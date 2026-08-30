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

---

## B-1: DIMM-top Capability Analysis

**Status:** READ-ONLY analysis (Step B-1, 2026-04-26). No RTL/sim/TB/generator
changes — pure documentation appending feasibility data for B-2 / B-3 / B-4
planning.

**Sources read:**
- `fc_verification/rtl/nldpe_dimm_top_d64_c128.v` (1533 lines)
- `fc_verification/rtl/azurelily_dimm_top_d64_c128.v` (380 lines)
- `fc_verification/rtl/nldpe_attn_head_d64_c128.v` (332 lines)
- `fc_verification/rtl/azurelily_attn_head_d64_c128.v` (427 lines)
- `nl_dpe/gen_dimm_nldpe_top.py`, `nl_dpe/gen_attention_wrapper.py`
- `nl_dpe/gen_dimm_azurelily_top.py`

### B-1.1 mac_qk back-to-back firing analysis

#### NL-DPE — `dimm_score_matrix.score_inst` FSM

**State diagram** (`nldpe_dimm_top_d64_c128.v:244–321`):
```
S_IDLE (0) → S_LOAD_Q (1) → S_LOAD_K (2) → S_COMPUTE (3) ⇄ S_WAIT_DPE (4)
                                                ↑                ↓
                                                └──── S_WRITE_B (7) ←──── (via S_COMPUTE if more cols)
                                                                  ↓
                                                           S_OUTPUT (6) → (terminal)
```

Key transitions:
- `S_IDLE → S_LOAD_Q` on `valid_q || valid_k` (line 266).
- `S_LOAD_Q → S_LOAD_K` once `q_write_addr == 13` (14 Q packed words written).
- `S_LOAD_K → S_COMPUTE` once `k_write_addr == 1664` (1665 K packed words written).
- Inside `S_COMPUTE`, FSM iterates `score_idx` over the lane's assigned key columns; for each `(score_idx, score_idx+1)` pair it goes `S_COMPUTE → S_WAIT_DPE → S_WRITE_B`.
- `S_WRITE_B` decides: if `score_idx + W*2 >= N` → `S_OUTPUT`; else stays in the inner loop.
- `S_OUTPUT`: terminal — only reads `score_sram` out via `score_read_addr` increment when `ready_n` is asserted.

**Q-row identity:** the FSM is hard-coded to compute scores **only for the
single Q row currently in q_sram** (q_sram DEPTH=14, addresses 0..13 from
`S_LOAD_Q`). After all `score_idx` columns are processed, FSM lands in
`S_OUTPUT` and **stays there forever**. No path back to `S_LOAD_Q` exists.

**What happens between two consecutive Q rows today:**
- The head FSM (`nldpe_attn_head_d64_c128.v:243–268`) holds `dimm_soft_rst` high for `DIMM_RST_CYCLES = 4` cycles, which forces every `*_inst` FSM back to its rst defaults (`S_IDLE` for score_inst, `SM_IDLE` for softmax_inst, `WS_IDLE` for wsum_inst).
- The k_sram contents persist (regular SRAM doesn't clear on rst); valid_k is re-driven for 1664 cycles (idempotent re-write of the same K data into the same addresses).
- valid_q is re-driven for 14 cycles, but with a fresh starting `read_addr = q_row_idx * PACKED_NQ` from the head's q_buffer.

**Conclusion (NL-DPE):** **DOES NOT support back-to-back mac_qk firing.**
The FSM has no path to "absorb a new Q row and recompute scores"; once
`S_OUTPUT` is reached, the only escape is rst. The head FSM today exploits
this via its `S_DIMM_RST` soft-rst pulse.

**Estimated additive change for B-2:**
- Add a `back_to_back` mode: in `S_OUTPUT`, watch a new `valid_q` pulse and reload q_sram in-place (re-enter `S_LOAD_Q`) without rst.
- Reset `score_idx` to `LANE_IDX * 2` and `feed_phase / feed_half / mac_count` to 0.
- Need `q_write_addr` to wrap to 0 on Q-row boundary.
- The K SRAM **does NOT need re-loading** between Q rows (already persisted), so back-to-back can SKIP `S_LOAD_K` entirely (saving 1664 cyc/Q-row × 128 = 213,000 cyc).
- Need a way to write each Q-row's scores into a **distinct** region of `score_sram` (see B-1.4 — addressing change).
- **Diff size:** ~30–50 lines in `dimm_score_matrix` FSM + `score_write_addr` reorganization. Generator change in `gen_attention_wrapper.py` `_gen_dimm_score_matrix` (≤80 lines added, default-off parameter).

#### Azure-Lily — `mac_qk_inst` FSM (top-level FSM)

**State diagram** (`azurelily_dimm_top_d64_c128.v:60–95`):
```
S_IDLE (0) → S_LOAD (1) → S_FEED_QK (2) → S_WAIT_QK (3) → S_OUTPUT (6) → S_IDLE
```

Key transitions:
- `S_IDLE → S_LOAD` on `valid_q || valid_k || valid_v` (line 68).
- `S_LOAD → S_FEED_QK` once `q_w_addr >= 13 && k_w_addr >= 1664 && v_w_addr >= 1664` (all 3 SRAMs filled).
- `S_FEED_QK` iterates `(row_count, mac_count)` from `(0,0)` to `(127, 16+1)`. For **each row_count** in [0..127] it cycles `mac_count` 0..17 (PACKED_K=16 + 2 SRAM latency), feeding `q_r_addr=mac_count` and `k_r_addr=row_count*16+mac_count`.
- After row_count=127, transitions `S_FEED_QK → S_WAIT_QK`.

**Q-row identity:** **Different from NL-DPE.** AL's mac_qk FSM is structured to compute **all 128 Q-row × 128 K-row scores in a single pass** — `row_count` walks 0..127, and within each row_count it walks all 128 K rows via `k_r_addr=row_count*16+mac_count`. Wait — re-reading: `row_count` is actually walking the **K rows**, since `q_r_addr` always reads 0..16 (one Q row from q_sram, depth=17) while `k_r_addr` walks all 128 K rows for that one Q row.

Actually, looking more carefully — `q_sram` has `DEPTH=17` (line 49: `sram ... DEPTH(17)` for q_sram). It only stores ONE Q row. So `row_count` walking 0..127 fires the dsp_mac for K rows 0..127, all multiplied by the same Q row in q_sram. After row_count=127, FSM transitions to `S_WAIT_QK` and never returns to `S_FEED_QK`.

**What happens between two consecutive Q rows today:**
- Same as NL-DPE: head FSM holds `dimm_soft_rst` high for 4 cycles, FSM returns to S_IDLE.
- valid_k/v are re-driven for 1665 cycles (idempotent into k_sram/v_sram).
- valid_q is re-driven for 13 cycles with a fresh Q row from head's q_buf.

**Conclusion (AL):** **DOES NOT support back-to-back mac_qk firing.**
After a single full N=128 K-row sweep against the q_sram-resident Q row, FSM
is locked in `S_WAIT_QK` / `S_OUTPUT` until rst. The K-row sweep is
internally efficient (no per-row rst); but the **outer Q-row loop** is not
supported.

**Estimated additive change for B-2:**
- Add a `back_to_back` mode: in `S_WAIT_QK` (or after row_count=127 exit),
  watch new `valid_q` pulse → re-enter `S_LOAD` for q_sram only (skip k_sram
  reload, skip v_sram reload) → re-enter `S_FEED_QK` with fresh q_sram.
- Reset `row_count` to 0; do NOT reset `k_w_addr` / `v_w_addr` (preserve persistence flag).
- Need score-write-side widening (B-1.3 / B-1.4).
- Counters that today gate on `q_w_addr >= 13` need to be aware of "Q-row 1, 2, …" so they accept 13 new pulses each Q row even though `q_w_addr` would advance past 13.
- **Diff size:** ~30–40 lines in the top-level FSM + ~10 lines in `q_w_addr` rollover logic. Generator change in `gen_dimm_azurelily_top.py` ≤60 lines added.

### B-1.2 mac_sv back-to-back firing analysis

#### NL-DPE — `dimm_weighted_sum.wsum_inst` FSM

**State diagram** (`nldpe_dimm_top_d64_c128.v:716–778`):
```
WS_IDLE (0) → WS_LOAD_A (1) → WS_LOAD_V (2) → WS_LOG_FEED (3) → WS_LOG_DRAIN (4)
                                                                       ↓
                                              WS_EXP_FEED (6) ⇄ WS_EXP_DRAIN (7)
                                                                       ↓
                                                   (when ws_m == last col) → WS_OUTPUT (5) → (terminal)
```

Key transitions:
- `WS_IDLE → WS_LOAD_A` on `valid_attn || valid_v` (line 730).
- `WS_LOAD_A → WS_LOAD_V` once `attn_write_addr == (N/W) - 1 = 7` (one Q row's
  attention scores for this lane = 8 elements at N=128/W=16).
- `WS_LOAD_V → WS_LOG_FEED` once `v_write_addr == d*PACKED_N - 1 = 64*26 - 1 = 1663`.
- `WS_LOG_FEED → WS_LOG_DRAIN → WS_EXP_FEED` (one shot, configures DPE input source).
- `WS_EXP_FEED ⇄ WS_EXP_DRAIN` cycles for each output column `ws_m` in [LANE_IDX * M_PER_LANE, (LANE_IDX+1) * M_PER_LANE - 1]. M_PER_LANE = d/W = 64/16 = 4 columns per lane.
- After last column: `WS_EXP_DRAIN → WS_OUTPUT`. Terminal.

**Q-row identity:** the FSM computes **one Q row's attention output** (4
output columns per lane × 16 lanes = 64 = d). Once done, lands in `WS_OUTPUT`
and stays there. No path back.

**What happens between Q rows today:** dimm_soft_rst returns the FSM to
`WS_IDLE`; valid_attn re-driven for 8 cycles (next Q row's normalized
attention from softmax_inst's `out_sram`); valid_v re-driven for 1664 cycles
(idempotent re-write).

**Conclusion (NL-DPE):** **DOES NOT support back-to-back mac_sv firing.**
Same structural problem as score_inst.

**Estimated additive change for B-3:**
- Add a `back_to_back` mode: in `WS_OUTPUT`, watch new `valid_attn` →
  re-enter `WS_LOAD_A` (skip `WS_LOAD_V` since v_sram persisted).
- Reset `attn_write_addr`, `attn_read_addr` to 0; reset `ws_m` to
  `LANE_IDX * M_PER_LANE`; reset `ws_feed_count` to 0.
- Need output-write-side widening (each Q row's d/W=4 outputs go to a
  **distinct** region of `out_sram`).
- **Diff size:** ~40–60 lines in `dimm_weighted_sum` FSM + `out_write_addr` widening.
  Generator change in `gen_attention_wrapper.py` `_gen_dimm_weighted_sum` ≤90 lines.

#### Azure-Lily — `mac_sv_inst` (lane-internal `dsp_mac`)

**Important difference from NL-DPE:** AL's mac_sv is a **pure dsp_mac module
with no FSM** of its own — it's just an accumulator that fires on `valid` and
emits `valid_n` after K=32 inputs.

Looking at `mac_sv_inst` in `azurelily_dimm_top_d64_c128.v:133-140`:
```
dsp_mac #(.DATA_WIDTH(DATA_WIDTH), .K(32)) mac_sv_inst (
    .clk(clk), .rst(rst),
    .valid(attn_valid), .ready_n(1'b0),
    .data_a(attn), .data_b(v_sram_out),
    ...
);
```

And `dsp_mac` (line 165–228):
- accumulates `K=32` MAC ops (`accum`),
- emits `valid_n=1` for one cycle when `count == K - 1`,
- automatically resets `count` to 0 and continues accumulating the next K=32.

**Conclusion (AL):** **The dsp_mac is intrinsically back-to-back** — it
naturally takes a new K=32 sequence and accumulates it into a fresh `accum`
on the next `valid` pulse after `valid_n`. **No FSM change needed.**

However, the **driver of `valid` is the softmax stage's `attn_valid`**, which
is gated by `clb_softmax`'s S_NORM phase. Architecture B requires that mac_sv
fires N=128 times back-to-back, which means **128 different softmax outputs
must drive mac_sv** — this is conditioned on softmax having processed all 128
rows (B-2 phase).

**Estimated additive change for B-3 (AL):** Effectively **none** for the
mac_sv FSM itself. The change is at the top-level FSM (B-5): instead of one
S_FEED_QK pass, we need a phase that streams attn → mac_sv 128 times. AL's
top-level FSM doesn't currently have any FEED_SV state; one needs to be
added. **Diff size:** ~30–40 lines in top-level FSM, ~20 lines in
gen_dimm_azurelily_top.py.

### B-1.3 Softmax SRAM capacity analysis

#### NL-DPE — score storage path

mac_qk writes scores into `score_sram` inside `dimm_score_matrix`
(`nldpe_dimm_top_d64_c128.v:240–242`):
```
sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(129))
score_sram (...);
```

**Per lane:**
- score_sram depth = **129 elements** (each element = 40-bit packed score word, but only the lower 8 bits are used per `score_write_data <= accumulator_a[7:0]`).
- Per lane handles `N/W = 128/16 = 8` scores per Q row (the lane covers key indices `LANE_IDX*2, LANE_IDX*2+1, LANE_IDX*2+2W, LANE_IDX*2+2W+1, ...`).
- `score_write_addr` is set to `score_idx` (which walks 0, 2W=32, 4W=64, ... up to N-2=126), then `score_idx + 1` for the second score of the pair.

**Important:** Although score_sram depth is 129 (more than enough for 8 elements per lane), the addressing scheme uses absolute `score_idx` (0..127) as the write address. This means the SRAM is **addressed by absolute key index**, and 129 ≥ N=128 — so each lane's score_sram can hold the full N=128 scores for **one** Q row, but not multiple.

Then softmax_inst copies score_sram into its own `in_sram`
(`nldpe_dimm_top_d64_c128.v:349–351`):
```
sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(129))
in_sram (...);
```

Wait — looking again. softmax_inst's in_sram is fed from `data_in` (which is
`score_out`). softmax_inst's `SM_LOAD` only writes `N_PER_LANE = N/W = 8`
elements. So softmax_inst's in_sram only holds **8 scores** for one Q row's
lane.

For Architecture B, where 128 Q rows × 128 keys = 16,384 total scores, **per
lane** that means 1024 scores (lane covers 8 keys × 128 Q rows).

**Capacity gap:**
- Today: 8 scores per lane (1 Q row × 8 keys/lane).
- Need: 1024 scores per lane (128 Q rows × 8 keys/lane).
- **Factor: 128× widening.**

But wait — score_sram itself is depth=129 (already big enough for **one Q
row's full 128 scores**, even though only 8 are stored per lane today). So
score_sram has **partial** spare capacity, but the fundamental problem is
that `score_write_addr <= score_idx` doesn't include a Q-row dimension. The
addressing is single-Q-row.

If we widen to depth 1024 per lane and address by `q_row_idx*8 + lane_local_addr`:
- score_sram: 129 → 1024 (≈8× widening).
- in_sram (softmax_inst): 129 → 1024.
- Total per lane: 2 SRAMs × 1024 elements × 40 bits = ~80 Kbit per lane × 16 lanes = ~1.28 Mbit. At BRAM size ~32 Kbit, this is ~40 BRAMs just for score storage.

**Comparison to current grid budget:** 120×120 grid has ~472 BRAMs, with
per-replica budget of 4–6 BRAMs in the GEMV DSE. The DIMM-top is currently
using ~16 BRAMs for the W=16 lanes' SRAM family. Widening to 40 BRAMs is a
**~2.5× increase** for score storage alone.

**Concrete BRAM count estimate (NL-DPE):**

Per lane today, the DIMM SRAMs are:
- q_sram: depth=14, 1 BRAM
- k_sram_a + k_sram_b: depth=1665 each, ~3 BRAMs
- score_sram: depth=129, 1 BRAM
- (softmax) in_sram: depth=129, 1 BRAM
- (softmax) exp_sram: depth=129, 1 BRAM
- (softmax) out_sram: depth=129, 1 BRAM
- (wsum) attn_sram: depth=129, 1 BRAM
- (wsum) v_sram: depth=1665, ~3 BRAMs
- (wsum) log_attn_sram: depth=27, 1 BRAM
- (wsum) out_sram: depth=65, 1 BRAM

**Total per lane today: ~14 BRAMs × 16 lanes = ~224 BRAMs.** But many of
these BRAMs are likely fitting into smaller block-RAMs in VTR (which has
flexibility in mapping logical SRAMs to physical BRAMs).

To widen score_sram + in_sram from 129 to 1024:
- score_sram: 1 → ~3 BRAMs per lane (~3× widening).
- in_sram: 1 → ~3 BRAMs per lane.
- **Net add: ~4 BRAMs per lane × 16 lanes = ~64 BRAMs.**

Combined with the post-softmax buffer widening (B-1.5 below), total add is
roughly **~130 BRAMs**, well within the 472-BRAM grid budget but doubling
the DIMM-top's BRAM consumption. Will likely affect Fmax marginally
(routing congestion).

**Conclusion (NL-DPE softmax SRAM capacity):** **NEEDS WIDENING.**
Score-storage capacity per lane grows from 8 to 1024 effective scores; SRAM
depth needs to grow from 129 to ~1024.

#### Azure-Lily — score storage path

mac_qk writes scores into the lane-internal `clb_softmax`'s `sm_buf`
(`azurelily_dimm_top_d64_c128.v:251–257`):
```
sram #(.DATA_WIDTH(40), .DEPTH(N)) sm_buf (...);
```

**Per lane:**
- `sm_buf` depth = **N = 128 elements** (40-bit each).
- Each lane's mac_qk fires 128 times (once per K row), so `sm_buf` accumulates 128 scores for **one** Q row.
- `clb_softmax`'s S_LOAD state increments `w_addr` to N-1 then transitions to S_INV / S_NORM, which **read** from `sm_buf` (no further writes). After S_NORM, `state` returns to `S_LOAD` and `w_addr` resets — overwriting on next iteration.

**Capacity gap:**
- Today: 128 scores per lane (1 Q row × 128 keys).
- Need: 128 × 128 = 16,384 scores per lane?

Wait — for AL, mac_qk's k_r_addr walks 0..1664, meaning each `dsp_mac
mac_qk_inst` accumulates K=16 entries before emitting one `valid_n` pulse.
After 128 such pulses (one per K row), all 128 scores for the current Q row
are in sm_buf.

Per lane handles **all** scores for one Q row (W=16 lanes × 128 k_rows =
2048 dsp_mac fires; oh wait, that's W=16 per K row × 128 K rows = 2048; so
each lane fires for all 128 K rows). Hmm, that doesn't match the
per-lane structure cleanly.

Actually re-reading: each lane has its own `mac_qk_inst` (line 110), and
the FSM at the **top** drives all 16 lanes' mac_qk simultaneously with the
same `q_r_addr` and `k_r_addr` (just XOR-anti-merged). So **each lane is
actually computing the same scores** (or differing by anti-merge XOR).

Hmm — looking at line 109: `wire [DATA_WIDTH-1:0] lane_k_qk = k_sram_out ^
lane[DATA_WIDTH-1:0];` — yes, each lane gets a different K input via XOR
with the lane index. For lane 0 the XOR is 0 (no change). For lanes 1..15,
the K is corrupted, so they don't produce mathematically meaningful scores
but DO consume DSP fires.

This means in AL, **only lane 0 has mathematically correct scores** —
which is an existing pre-Architecture-B oddity (anti-merge for VTR). The
AL counter gate's `mac_qk dpe_fire_count = 2048 = W=16 lanes × 128 fires`
but only 128 of those are functionally meaningful scores.

For Architecture B's all-N-row score persistence, AL would need to extend
each lane's `sm_buf` from 128 to 128×128=16,384 — which is **way too big**
(40 bits × 16,384 = 640 Kbit per lane × 16 lanes = 10 Mbit, far exceeding
total BRAM budget).

**Better AL strategy for B:** since only lane 0 has meaningful scores,
widen only lane 0's `sm_buf` to N×N=16,384 (or use a single global score
buffer at the top, not per-lane). The other 15 lanes still fire mac_qk for
counter parity but their sm_buf stays at depth=128 and is overwritten each
Q row.

**Capacity decision (AL):**
- Per-lane `sm_buf` widening: minimal (keep at 128 for anti-merge lanes).
- Single shared score buffer at top: depth = N×N = 16,384 entries × 40 bits = 640 Kbit ≈ **20 BRAMs**.
- Alternative: per-lane lane-0 sm_buf widening: same ~20 BRAMs but cleaner.
- Either way: ~20 BRAMs added — within budget, similar order to NL-DPE.

**Conclusion (AL softmax SRAM capacity):** **NEEDS WIDENING (lane 0 only) or
a NEW top-level score buffer** sized for N×N.

### B-1.4 Softmax SRAM addressing analysis

#### NL-DPE

Today's score-write addressing (`nldpe_dimm_top_d64_c128.v:303-313`):
```
score_write_addr <= score_idx;            // S_WRITE_A
score_write_addr <= score_idx + 1;        // S_WRITE_B
```

`score_idx` walks `LANE_IDX*2, LANE_IDX*2+2W, ...` (covers 8 elements per
lane). Address range: 0..127 for the lane's pairs. **No Q-row dimension.**

For Architecture B accumulation:
- Need a `q_row_idx` field: `score_write_addr = q_row_idx * (per_lane_score_count) + score_idx_normalized`.
- Each lane stores 8 scores per Q row × 128 Q rows = 1024 elements.
- score_write_addr would be `q_row_idx * 8 + (score_idx_normalized)` where `score_idx_normalized = score_idx / (2*W)` × 2 + offset.
- Or, simpler: `score_write_addr = q_row_idx * N + score_idx` if SRAM is widened to depth ≥ N×N (overkill but simpler).

**Minimal additive change for B-4 addressing:**
- Add `q_row_idx` input to `dimm_score_matrix` (or sample it from the head FSM via a new port).
- Replace `score_write_addr <= score_idx` with `score_write_addr <= q_row_idx * (N/W * 2) + (score_idx % (2*W))` (for compact addressing).
- Same for softmax_inst's `in_sram` write-address for the SM_LOAD phase (currently increments 0..7; needs to read q_row_idx-base offset).
- **Diff size:** ~15 lines per FSM, plus port additions. Generator add ~30 lines.

#### Azure-Lily

Today's clb_softmax write addressing (`azurelily_dimm_top_d64_c128.v:288-294`):
```
S_LOAD: begin
    if (valid) begin
        sum_exp <= sum_exp + exp_val;
        if (w_addr == N-1) begin
            state <= S_INV; w_addr <= 0; ...
        end else
            w_addr <= w_addr + 1;
    end
end
```

`w_addr` walks 0..N-1 for one Q row, then resets (but immediately starts
S_INV/S_NORM, so no overwrite within the same Q row). On next Q row (after
soft rst), w_addr starts at 0 again — **overwrites** previous Q row.

**Minimal additive change for B-4 (AL):**
- Add `q_row_idx` field to write address: `actual_w_addr = q_row_idx * N + w_addr`.
- Widen `w_addr` reg from `$clog2(N)=7` to `$clog2(N*N)=14` (or `$clog2(N) + $clog2(N) = 14`).
- Don't reset `w_addr` between Q rows in back_to_back mode.
- Adjust S_INV/S_NORM to fold rows correctly.
- **Diff size:** ~20 lines in `clb_softmax` + ~5 lines for `q_row_idx` port. Generator add ~25 lines.

### B-1.5 mac_sv input SRAM (the post-softmax buffer)

#### NL-DPE — softmax → wsum buffer

After softmax_inst computes normalized weights, they're written to
softmax_inst's `out_sram` (`nldpe_dimm_top_d64_c128.v:449-451`):
```
sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(129))
out_sram (...);
```

This out_sram is then **read out** as `softmax_out` and feeds into
`dimm_weighted_sum`'s `attn_sram` (`nldpe_dimm_top_d64_c128.v:524-527`):
```
sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(129))
attn_sram (...);
```

attn_sram is written by wsum's `WS_LOAD_A` with `attn_write_addr <=
attn_write_addr + 1` until it reaches N/W - 1 = 7. **One Q row's normalized
weights for this lane = 8 elements.**

For Architecture B:
- softmax → wsum buffer needs to hold all 128 Q rows' normalized weights = 128 × N/W = 128 × 8 = 1024 elements per lane.
- attn_sram depth: 129 → 1024.

Same ~3 BRAMs per lane × 16 = ~48 BRAMs added (similar magnitude to score
buffer).

Addressing change:
- Today: `attn_write_addr` increments 0..7 per Q row, resets to 0 on next iteration via dimm_soft_rst.
- B-4: `attn_write_addr` = `q_row_idx * 8 + lane_local_pos`; same q_row_idx coupling as score_sram.

#### Azure-Lily

clb_softmax's S_NORM emits `data_out` directly (line 301: `data_out <=
norm_product`) — there's no intermediate buffer between softmax and mac_sv.
The mac_sv `dsp_mac` consumes `attn` (= softmax `data_out`) directly with
`valid = attn_valid` (line 135).

For Architecture B:
- The dsp_mac.accum auto-resets after K=32 inputs, so it's ready for the next sequence.
- BUT: the AL clb_softmax can only emit one Q row's normalized weights at a time, then has to wait for the next Q row's scores to flow through mac_qk. In Architecture B, we'd want all 128 softmax rows queued first, then all 128 mac_sv fires.
- **Solution:** add an intermediate `weight_buffer` SRAM between softmax and mac_sv. depth = N × N = 16,384 (per-lane: only lane 0 has meaningful weights, but per-lane is simpler).
- Or: keep clb_softmax's existing flow but add a FIFO between softmax and mac_sv that holds 128 rows of weights.
- Estimated: ~20 BRAMs (similar to score buffer).

### B-1.6 Cross-arch comparison summary

| Question | NL-DPE | Azure-Lily |
|---|---|---|
| **Q1: mac_qk back-to-back** | NOT supported. FSM lands in S_OUTPUT, no escape. | NOT supported. FSM lands in S_WAIT_QK, no escape. |
| **Q1: Diff size for B-2** | ~30–50 lines / ~80 gen | ~30–40 lines / ~60 gen |
| **Q2: mac_sv back-to-back** | NOT supported. FSM lands in WS_OUTPUT, no escape. | **Already supported** (dsp_mac auto-recycles). Top FSM needs new state. |
| **Q2: Diff size for B-3** | ~40–60 lines / ~90 gen | ~30–40 lines top FSM / ~20 gen |
| **Q3: Softmax SRAM N×N** | NOT adequate (8 per lane, need 1024). | NOT adequate (128 per lane, need 16,384 OR shared buffer). |
| **Q3: Diff size for B-4** | score_sram + in_sram widening; ~15 lines per FSM × 2 + ~30 gen. **~64 BRAMs added.** | Lane-0 widening or shared top-level score buffer; ~20 BRAMs added. |
| **Q4: Softmax addressing** | Single-Q-row only; needs q_row_idx port. | Single-Q-row only; needs q_row_idx port. |
| **Q4: Diff size for B-4 addressing** | ~15 lines per FSM. | ~20 lines + port addition. |
| **Q5: Post-softmax buffer N×N** | attn_sram + out_sram widening (129 → 1024). **~48 BRAMs added.** | NEW weight_buffer SRAM (no current intermediate). **~20 BRAMs added.** |

### B-1.7 Diff size estimate per follow-up step

**B-2 (mac_qk back-to-back):**
- NL-DPE: ~50 lines `dimm_score_matrix` + ~80 lines generator add.
- AL: ~40 lines top FSM + ~60 lines generator add.
- BRAM cost: 0 (FSM-only, no SRAM widening).
- **Combined: ~230 lines RTL/generator total.**

**B-3 (mac_sv back-to-back):**
- NL-DPE: ~60 lines `dimm_weighted_sum` + ~90 lines generator add.
- AL: ~40 lines top FSM (mac_sv inherently capable) + ~30 lines generator add.
- BRAM cost: 0.
- **Combined: ~220 lines.**

**B-4 (softmax SRAM N×N + addressing):**
- NL-DPE: ~30 lines per FSM × 4 SRAMs (score_sram, in_sram, exp_sram, out_sram, attn_sram) + ~80 lines generator add. Most actual change is in depth values + write_addr formula. The depth-only part is ~5 lines per SRAM, so net widening is small.
- AL: ~50 lines clb_softmax (depth + addressing) + new `weight_buffer` ~30 lines + ~70 lines generator.
- **BRAM cost (NL-DPE):** ~64 BRAMs added (score_sram + in_sram widened from 129→1024); plus ~48 BRAMs (attn_sram + out_sram widened from 129→1024); **total ~110 BRAMs added.** Current per-DIMM BRAM ≈ 224, so post-B-4 ≈ 334 BRAMs. With 472-BRAM grid budget → fits with 138 BRAMs spare.
- **BRAM cost (AL):** ~20 BRAMs (lane-0 score buffer) + ~20 BRAMs (weight_buffer) = ~40 BRAMs added. AL DIMM-top is currently smaller than NL-DPE; net BRAM consumption stays well within budget.
- **Fmax impact:** widening score/attn SRAMs may add 5–15% routing congestion overhead. Need VTR re-synth (Step 3 in deferred).
- **Combined: ~320 lines RTL/generator total.**

**B-5 (head FSM 3-phase redesign):**
- NL-DPE head: ~150 lines (replace 128-iter loop with 3 sequential phases — S_PHASE1_FIRE_MAC_QK / S_PHASE2_FIRE_SOFTMAX / S_PHASE3_FIRE_MAC_SV).
- AL head: ~120 lines (similar restructure).
- Generator changes: `gen_nldpe_attn_head_top.py` and `gen_azurelily_attn_head_top.py` rewrite of the FSM body section, ~200 lines each.
- BRAM cost: 0 (FSM-only).
- **Combined: ~670 lines.**

**Total Architecture B implementation estimate: ~1440 lines RTL/generator
diff + ~110 BRAMs added (NL-DPE) / ~40 BRAMs (AL).**

### B-1.8 Risk flags

1. **NL-DPE softmax sum semantics under all-N accumulation.** Today's softmax FSM computes `exp_sum` over `N_PER_LANE = 8` elements per Q row. In Architecture B, the FSM must compute exp_sum over 8 elements for **each** Q row separately (not over the full 1024-element widened in_sram). This requires `q_row_idx`-aware FSM control: SM_EXP iterates 8 elements, then transitions to SM_NORMALIZE for that Q row, then re-enters SM_LOAD/SM_EXP for the next Q row. Estimated extra: ~20 lines on top of B-4 capacity. **Minor structural change — does NOT block Architecture B.**

2. **AL anti-merge XOR vs softmax correctness.** Currently lanes 1..15 in AL produce bogus mac_qk outputs (XOR with lane index). For Architecture B's "process all 128 rows" semantics to be functionally meaningful, only lane 0 produces valid scores. This is a **pre-existing AL property** (T2v2 era), not a new Architecture B problem, but it means the per-lane Q row × N/W stride architecture used in NL-DPE doesn't directly apply to AL. AL's score accumulation is naturally **single-Q-row-at-a-time** at lane 0; Architecture B widens that to 128 Q rows in lane 0. **No structural blocker.**

3. **VTR Fmax impact from BRAM widening.** Adding ~110 BRAMs to NL-DPE DIMM-top may shift its BRAM utilization from ~50% (224/472) to ~70% (334/472). At 70% BRAM utilization the placer struggles to keep BRAMs near consumers, and Fmax may degrade ~5–15%. **Needs VTR re-synth verification (deferred Stage 3).** If Fmax drops below current 102.9 MHz, it could affect cycle budgets but NOT functional correctness. Architecture B remains viable; cycle gate may need re-tuning.

4. **No fundamental softmax-row-coupling block.** Both NL-DPE and AL softmax modules process rows independently (no inter-row state in S_NORM). Architecture B can decompose them naturally — no single-row "inseparability" risk.

5. **Score readback ordering.** In Architecture B, mac_qk writes 128 Q rows × 128 keys = 16,384 scores into score_sram. The softmax phase needs to read them back in the right order (per Q row, per lane). The simplest order is `q_row_idx * N + key_idx` major (write-side), which softmax can iterate as `for q_row in 0..127: for key in 0..127: read(q_row*N + key)`. This is **straightforward** but requires the addressing change in B-4. **No blocker.**

6. **Mac_sv back-to-back NL-DPE wsum_inst structural change is largest.** The wsum_inst FSM has 9 states and tightly couples WS_LOAD_A/V with WS_LOG_FEED/EXP_FEED for one Q row. Decoupling requires careful state preservation: ws_m, ws_feed_count, ws_acc_clear all need q_row_idx-awareness. ~60 lines is the right order; no architectural blocker, but **highest individual diff size in the plan.**

7. **No surprises requiring revising Architecture B plan.** All four capability questions can be answered with "additive RTL diff + BRAM widening" — no structural redesigns needed. Architecture B as-spec'd is feasible.

---

## B-1 Bottom-line conclusion

Architecture B is implementable as additive RTL diffs to both DIMM-top FSMs.
The biggest physical cost is **~110 BRAMs added in NL-DPE** (score + attn
SRAM widening from 129 to 1024-deep), still within 472-BRAM grid budget.
No architectural blockers found. The plan in §5 (B-2, B-3, B-4, B-5)
remains structurally correct; sub-step splits per arch are well-bounded.

**Recommendation:** proceed to B-2 (mac_qk back-to-back) for both archs in
parallel; SRAM widening (B-4) can be done independently of FSM changes.
B-5 (head FSM redesign) should be deferred until B-2/B-3/B-4 land.

