# Architecture B — Sim-Faithful AH RTL Architecture Spec

**Status:** DRAFT, awaiting user review. No code changes until approved.
**Purpose:** Define the RTL target architecture derived from sim's actual `--model attention` execution path. All future cycle-alignment work derives from this spec.

---

## 1. Sim's actual architecture (verified by reading source)

**Entry point:** `azurelily/IMC/test.py --model attention --seq_length N --head_dim d`

**Execution path:**
1. `attention_model(N, d)` in `azurelily/models/attention.py` constructs 7 layers:
   - `linear_Q`, `linear_K`, `linear_V` (3 parallel Linear_Layer arms)
   - `mac_qk` (MAC_QK_Layer)
   - `exp` (Softmax_Exp_Layer)
   - `norm` (Softmax_Norm_Layer)
   - `mac_sv` (MAC_SV_Layer)

2. `run_transformer_model(model, imc)` walks the chain calling `imc.run_layer(layer)` for each.

3. Each `_run_*` (in `scheduler_stats/scheduler.py`) returns a **single full-stage ns** that internally accounts for row-pipelining via `gemm_log` / `gemm_dsp`.

4. **E2E = sum of all stage latencies, sequential composition.**
   - `_run_attention_pipeline` (which models cross-stage streaming) **EXISTS but is NOT invoked** by `--model attention`. It's dead code for this path.

**Sim cycle output (NL-DPE Fmax 102.9 MHz, AL Fmax 87.9 MHz):**

| Stage | NL-DPE cyc | AL cyc | Note |
|---|---:|---:|---|
| linear_qkv | 2,424 | 4,000 | max of Q/K/V arms (parallel) |
| mac_qk | 5,428 | 815 | full N×N scores, intra-row pipelined |
| softmax_exp | 41 | 12 | 128 rows × W=16 lane fold |
| softmax_norm | 24 | 47 | 128 rows × W=16 lane fold |
| mac_sv | 5,077 | 1,450 | full N×d outputs, intra-row pipelined |
| **E2E** | **12,994** | **6,324** | sum |

**Architectural truth:**
- Stages run **sequentially** (no inter-stage overlap).
- Within each stage, rows are **pipelined** (e.g., mac_qk's 128 Q rows pipeline internally; first row 248 cyc, steady ~41 cyc/row → total ≈ 5,428).
- linear_Q / linear_K / linear_V run in **parallel** (max latency, not sum).

---

## 2. Current RTL architecture (post-Bug-1, post-Bug-2)

**Entry point:** `tb_*_attn_head_v2.v` driving `*_attn_head_d64_c128.v`

**Execution flow (Architecture C — what we have today):**
```
1. Streaming linear_qkv: 3 FC arms run in parallel, producing N=128 Q rows + N=128 K rows + N=128 V rows
   → buffered in head's q_buffer / k_buffer / v_buffer
2. Outer loop, 128 iterations:
   for q_row_idx in 0..127:
       a. Soft-reset DIMM (4-cycle pulse, q_row_idx > 0)
       b. Load q_row_idx into DIMM's q_sram (14 words)
       c. Re-stream K, V into DIMM (already-known data, idempotent SRAM rewrite)
       d. Fire DIMM (mac_qk → softmax → mac_sv internally for this Q row)
       e. Wait for dimm_valid_n (full mac_sv output of this Q row)
       f. Move to next Q row
3. Output drained
```

**Cycle output (current):**

| Stage | NL-DPE cyc | AL cyc |
|---|---:|---:|
| linear_qkv | 3,315 | 4,458 |
| mac_qk (TB span) | 428,139 | 511,430 |
| softmax_exp (TB span) | 425,077 | 511,430 |
| softmax_norm (TB span) | 425,079 | 509,270 |
| mac_sv (TB span) | 427,861 | 509,208 |
| **E2E** | **433,429** | **517,742** |

**Why per-stage spans are nearly equal across stages:** each stage fires in every Q-row iteration, so its TB-measured span (first-fire to last-fire) covers nearly the full E2E.

**Architectural divergence from sim:**
- Sim: 3 sequential stages, each row-pipelined → E2E = sum(stages)
- Current RTL: 128 sequential per-Q-row iterations, each iteration full-pipeline → E2E = 128 × per-Q-row + glue

---

## 3. Architecture B RTL target (sim-faithful)

To match sim's stage composition exactly, the RTL head FSM must run **3 sequential phases**, each processing all 128 rows internally with row pipelining:

```
Phase 0 (linear_qkv):  ALREADY IMPLEMENTED — streaming FC arms in parallel, produces N=128 Q + N=128 K + N=128 V
Phase 1 (mac_qk):      Fire DIMM-top mac_qk back-to-back N=128 times, intra-pipelined.
                       Output: 128 rows × 128 scores = 16,384 scores accumulated in softmax SRAM.
Phase 2 (softmax):     Run softmax on all 128 rows of scores (W=16 lane fold).
                       Output: 128 rows × 128 weights in mac_sv SRAM.
Phase 3 (mac_sv):      Fire DIMM-top mac_sv back-to-back N=128 times, intra-pipelined.
                       Output: 128 rows × d=64 attention outputs.
```

**Key differences from current Architecture C:**

| Aspect | Architecture C (current) | Architecture B (target) |
|---|---|---|
| Outer structure | 128 iterations of (mac_qk → softmax → mac_sv) | 3 sequential phases (mac_qk_all → softmax_all → mac_sv_all) |
| Soft-reset between Q rows | Yes (S_DIMM_RST) | No — Q rows flow back-to-back within a phase |
| mac_qk row pipelining | No (full sequential per Q row) | Yes (Q row k+1's compute starts while Q row k drains) |
| mac_sv row pipelining | No | Yes (symmetric) |
| softmax all-rows accumulation | Per-Q-row (1 row at a time) | All 128 rows accumulated, then softmax over all |
| Score / weight buffer between stages | Implicit (DIMM-top internal SRAMs reused per Q row) | All N=128 rows of scores must persist between mac_qk and softmax phases; all 128 rows of weights between softmax and mac_sv |

---

## 4. RTL implementation implications

### 4a. Head FSM redesign (the big change)

**Current head FSM states:**
```
S_FEED → S_STREAM_TOKENS → S_DRAIN_FC → S_DIMM_RST → S_FIRE_DIMM → S_DIMM_WAIT → S_DIMM_RST → ... (loop) → S_OUTPUT
```

**Architecture B head FSM states:**
```
S_FEED                        # initial
S_STREAM_TOKENS               # linear_qkv runs (already exists)
S_DRAIN_FC                    # linear_qkv finishes
S_PHASE1_INIT                 # one-time DIMM init
S_PHASE1_FIRE_MAC_QK          # fire mac_qk N=128 times back-to-back
                              # advance q_row_idx every per-row-steady cycles
                              # mac_qk's output → softmax SRAM (existing path)
S_PHASE1_DRAIN                # last Q row drains through mac_qk
S_PHASE2_FIRE_SOFTMAX         # softmax processes all 128 rows of scores
                              # softmax SRAM is now fully populated
                              # softmax's output → mac_sv SRAM (need to confirm RTL path)
S_PHASE2_DRAIN
S_PHASE3_FIRE_MAC_SV          # fire mac_sv N=128 times back-to-back
S_PHASE3_DRAIN
S_OUTPUT
```

### 4b. DIMM-top changes needed

**Question 1: Does DIMM-top support back-to-back mac_qk firing without inter-Q-row reset?**

Need to check:
- `nldpe_dimm_top_d64_c128.v` `score` FSM internals
- Whether `q_w_addr` wraps cleanly when mac_qk fires for Q row k+1
- Whether the score-write into softmax SRAM uses absolute Q-row index for addressing

If not currently supported: need to add a "back-to-back mode" parameter to the score FSM (additive) that disables the post-completion idle and accepts a new Q-row trigger immediately.

**Question 2: Can softmax SRAM hold all N=128 rows of scores?**

Per-lane SRAM in NL-DPE softmax: needs to hold N×N / W = 128×128/16 = 1024 elements per lane. Each element is a score (~16 bits typically). That's 2 KB per lane × 16 lanes = 32 KB total. Need to verify current softmax SRAM is sized for this OR widen it.

**Question 3: Symmetric for mac_sv and weight SRAM?**

Same questions for mac_sv firing back-to-back, and for the weight buffer between softmax and mac_sv.

### 4c. Implications for Step 1's score_valid_o (already landed)

`score_valid_o` was added in Step 1 for cross-stage pipelining (Architecture A). In Architecture B, mac_qk → softmax communication doesn't need a per-cycle handshake (softmax waits for mac_qk to fully complete). The wire is harmless but unused.

**Recommendation:** keep Step 1's `score_valid_o` as-is — it's a valid debug probe and not in the way.

---

## 5. Step plan derived from Architecture B

Replacing the previous 5-step plan with sim-faithful steps:

| Step | What | Cycle change | Risk |
|---|---|---|---|
| **B-0 (this doc)** | Architecture spec, capability questions | None | None |
| **B-1** | Read DIMM-top score FSM + softmax module + mac_sv FSM. Document whether back-to-back firing is currently supported. Document SRAM capacities. **No code changes.** | None | None |
| **B-2** | Add "back-to-back mode" parameter to DIMM-top score FSM if needed (additive, default off). Verify back-to-back mac_qk firing produces correct score data. | None when off | Low — additive |
| **B-3** | Symmetric for mac_sv FSM. | None when off | Low |
| **B-4** | Verify softmax SRAM sized for N×N (or widen if needed). Functional check: full N=128 score row processed correctly. | None — functional verification | Medium if SRAM widening required |
| **B-5** | Head FSM redesign: replace 128-iteration outer loop with 3 sequential phases (mac_qk_all → softmax_all → mac_sv_all). Each phase enables back-to-back mode in the relevant DIMM-top stage. | **All cycle gain happens here.** Projected E2E: NL ~13K, AL ~6.3K. | High — biggest single change in the plan |

**Step B-5 success criterion:**
- Counter gate: green (event count unchanged)
- Functional gate: AH output bits match reference for all N=128 outputs
- Cycle gate: per-stage RTL TB cycles match sim's per-stage cycles within tolerance:
  - linear_qkv: ≈ 2,424 (NL) / 4,000 (AL) — already close
  - mac_qk: ≈ 5,428 (NL) / 815 (AL)
  - softmax_exp: ≈ 41 (NL) / 12 (AL)
  - softmax_norm: ≈ 24 (NL) / 47 (AL)
  - mac_sv: ≈ 5,077 (NL) / 1,450 (AL)
  - E2E: ≈ 12,993 (NL) / 6,324 (AL)
- TB probe semantics: per-stage probe = first-fire to last-fire within that phase only (NOT across all phases). Once Architecture B is in place, the probe spans naturally narrow to per-phase windows.

---

## 6. Open questions for user

1. **Confirm Architecture B is the desired target.** Architecture A (cross-stage pipelining within Q row + outer loop pipeline) was the original Step 2-5 plan; it's a different architecture that may or may not exactly hit sim's cycle numbers.

2. **SRAM widening tolerance.** If softmax SRAM is too small for all N×N=16,384 scores, do we (a) widen the SRAM (more BRAM consumption, may affect VTR Fmax) or (b) keep current size and accept that Architecture B can't be fully implemented (revert to Architecture A or accept the gap)?

3. **Step granularity.** B-2 / B-3 / B-4 / B-5 may need further sub-division depending on what B-1's analysis reveals. Spec is a starting point; sub-steps emerge from RTL reading.

4. **Dead code in sim.** `_run_attention_pipeline` exists in sim but isn't invoked. Should we (a) leave it (informational), (b) delete it (cleanup), or (c) make it the canonical entry path so sim's "streaming pipeline" model becomes accessible — but then sim would predict different cycles than current expected_cycles.json. **Recommend (a): leave alone, don't break expected_cycles.json.**

---

## 7. Methodology lessons captured

- Bug 1 picked an architectural option ("less invasive outer loop") without consulting sim's source. This is the wrong workflow for sim-alignment work.
- Going forward: every FSM-topology change must reference a specific sim function and execution path.
- Counter gate proves event-count equivalence but not architectural equivalence. To prove architectural equivalence, we need to verify temporal organization matches (e.g., per-stage TB spans match per-stage sim cycles within tolerance).
- "Architectural equivalence" includes:
  - Same stage composition (sequential vs pipelined)
  - Same intra-stage row-pipelining behavior
  - Same buffer ownership between stages
  - Same parallelism (W=16 lanes, K_id, etc. — already verified by counter gate)
