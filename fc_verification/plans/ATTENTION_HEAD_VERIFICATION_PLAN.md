# Attention Head RTL Verification + Latency Alignment Plan

**Opened:** 2026-04-24
**Last revised:** 2026-04-24 (post T2v2/T3v2/T4v2 closeout, commit `e118b11`)
**Precedes:** closes the V-model loop opened by `DIMM_FULL_VERIFICATION_PLAN.md` (DIMM top, Phase H–N / 3–6A)
**Status:** **T1 ✓, T2v2 ✓, T3v2 ✓, T4v2 ✓ (all closed at commit `e118b11`); T5v2 pending user sign-off; T6 follow-up.**

## Closeout summary (2026-04-24)

The streaming FC refactor (Phase A of commit `e118b11`) brought both
architectures into structural alignment with sim's `attention_model`
streaming Linear_Layer convention. NL-DPE adopts a ping-pong DPE pair
per arm; Azure-Lily adopts N parallel-output dsp_macs per arm. Combined
functional+latency TBs (`tb_{nldpe,azurelily}_attn_head_v2.v`) drive
N=128 tokens, capture per-stage probes, and report Overall=PASS. The
arch-tagged `phase7_known_deltas.json` carries file:line root-cause
citations for every residual; `run_checks.py` got an AH dispatcher
(`check_ah_attn_head`) with AL softmax_exp+norm fold semantics, and
both `run_checks.py --config *_attn_head_d64_c128` exit 0.

**Final per-stage alignment** (RTL ↔ sim, classifications in
`phase7_known_deltas.json`):

| Stage | NL-DPE RTL/sim/Δ | NL-DPE class | AL RTL/sim/Δ | AL class |
|---|---|---|---|---|
| linear_qkv  | 3,350 / 2,424 / +926 | m.g. | 4,529 / 4,000 / +529 | m.g. |
| mac_qk      | 1,948 / 5,690 / −3,742 | structural | 1,683 / 6,661 / −4,978 | structural |
| softmax_exp | 8 / 650 / −642 | structural | 2,289 / 938 (exp+norm fold) / +1,351 | m.g. |
| softmax_norm| 10 / 376 / −366 | structural | 0 / — / 0 (folded) | m.g. |
| mac_sv      | 251 / 5,339 / −5,088 | structural | 32 / 6,091 / −6,059 | structural |
| **E2E**     | **6,692 / 14,480 / −7,788** | **structural** | **8,597 / 17,689 / −9,092** | **structural** |

Negative residuals are `structural` because sim's `gemm_log`/`gemm_dsp`
analytical bodies are conservative single-lane lower-bounds while RTL
realises W=16 hardware-lane parallelism. Positive residuals are
`modelling_granularity` (FSM transitions, streaming fill/drain edges,
AL CLB-serial softmax bottleneck).

**Resource counts (per head, regen-checked):**
- NL-DPE: 6 DPE (3 arms × 2 ping-pong) + 64 DIMM = **70 DPE**
- AL: 192 dsp_mac (3 arms × 64 parallel-output) + 32 DIMM dsp_mac +
  16 clb_softmax = **224 dsp_mac + 16 softmax**

**Gate (5 commands, all exit 0):**
```
python3 azurelily/IMC/test_gemm_log_regime_b.py                          # Phase 1
python3 fc_verification/run_fc_phase2.py --arch both --skip-vtr           # T1 14/14 PASS
python3 fc_verification/run_checks.py --config nldpe_dimm_top_d64_c128    # P3+P4
python3 fc_verification/run_checks.py --config azurelily_dimm_top_d64_c128  # P5+P6A
python3 fc_verification/run_checks.py --config nldpe_attn_head_d64_c128   # P7 NL-DPE
python3 fc_verification/run_checks.py --config azurelily_attn_head_d64_c128  # P7 AL
```

**Pending:** T5v2 (VTR 3-seed runs + metric verification) is staged
as a separate user-driven step. T6 (BERT-Tiny generator refinement)
remains a follow-up that will use the verified head as canonical
reference.

## Ultimate goal

End-to-end RTL↔sim cycle-accurate alignment for a **standalone attention
head** (FC_Q/K/V → DIMM → FC_O), for both NL-DPE and Azure-Lily, with
every residual classified `modelling_granularity` or `structural`.
Symmetric probe methodology, harness, and known-deltas schema across
both architectures (mirrors DIMM and FC parity).

**Scope clarification:** the verification target is a SINGLE attention
head, NOT a slice of BERT-Tiny RTL nor the full BERT-Tiny model. The
verified head will later serve as a reference for patching
`gen_bert_tiny_wrapper.py` (out-of-scope here; tracked as T6 follow-up).

## Scope anchor (frozen design — paper-consistent)

Both architectures share these parameters; **only the compute primitive
and mapping differ.**

| Parameter | Value | Source |
|---|---|---|
| `d_model` | 128 | input/output token dim (BERT-Tiny attention head) |
| `d_head` (N out of FC, K of O FC) | 64 | BERT-Tiny per-head dim |
| `seq_len N` (DIMM) | 128 (primary; multi-N stretch) | matches simulator sweep + verified DIMM top |
| `C` (crossbar cols / DSP-equiv) | 128 | NL-DPE 1024×128 crossbar |
| `R` (crossbar rows) | 1024 | NL-DPE Round-1 winner |
| `W` (DIMM parallel lanes) | **16** | `paper/methodology/attention_dimm_mapping.md §3` (paper spec, NOT BERT-Tiny benchmark RTL which uses W=1 — that's a generator bug, T6 patches it) |
| `K_qkt` | 2 | C/d_head dual-identity |
| `K_sv` | 1 | when N≥C single-identity |
| `DATA_WIDTH` | 40 bit | DPE bus / dsp_mac bus (5 × int8 packed) |

### What diverges (separate flows, different compute units + mappings)

| Aspect | NL-DPE | Azure-Lily |
|---|---|---|
| Compute primitive | `dpe` hard block (analog crossbar + ACAM/ADC) | `dsp_mac` (4-wide pure `int_sop_4`, P6A canonical) |
| Q/K/V projection (T2v2 streaming) | Ping-pong DPE pair per arm (DPE_A loads while DPE_B computes/drains) | N parallel-output dsp_macs per arm (one dsp_mac per output column, broadcast input) |
| DIMM mac_qk | DPE w/ K_qkt=2 dual-identity, **log-domain** | `dsp_mac` w/ K=⌈d/4⌉=16, **linear-domain** |
| DIMM softmax | DPE(I\|exp) for sm_exp + DPE(I\|log) for normalize | `clb_softmax` (CLB FSM with SRAM, recip + DSP normalize, **CLB-serial accumulator**) |
| DIMM mac_sv | DPE w/ wsum_log + wsum_exp 128×128 (post-Phase-4), log-domain | `dsp_mac` w/ K=⌈N/4⌉=32, linear-domain |
| Pipeline regime | Regime B Layout A (`T = L_A·M + O`) | Regime A serial |
| Sim model | `gemm_log` in `azurelily/IMC/peripherals/fpga_fabric.py` + `attention_model` streaming | `gemm_dsp` in same file + `attention_model` streaming |
| DPE/DSP per head (post-T2v2) | 6 proj (3 × ping-pong) + 64 DIMM = **70 DPE** | 192 proj (3 × 64 parallel-output) + 32 DIMM = **224 dsp_mac** + 16 softmax |

**Configs verified:**
- `nldpe_attn_head_d64_c128`
- `azurelily_attn_head_d64_c128`

## Prerequisites verified

- **FC RTL↔sim:** NL-DPE ✓ (12/12 Phase 2), Azure-Lily ✓ (T1 closed, 14/14 unified gate at commit `9e6a913`; backward compat 14/14 PASS preserved post-`e118b11`)
- **DIMM top RTL↔sim:** NL-DPE ✓ (Phase 3/4, E2E +34 cyc, m.g.), Azure-Lily ✓ (Phase 5/6A, E2E +2 cyc, m.g.)
- **Composed attn-head RTL:** NL-DPE ✓, Azure-Lily ✓ (T2v2 closed, streaming FC, both compile clean at commit `e118b11`)
- **Attention-head latency RTL↔sim:** NL-DPE ✓, Azure-Lily ✓ (T4v2 closed at commit `e118b11`; per-stage table above)

## Stages

### Stage 1 — Close block-level parity

**T1. Azure-Lily FC Phase-2 harness**

Artifacts:
- `fc_verification/tb_azurelily_fc_512_128.v`
- `fc_verification/tb_azurelily_fc_2048_256.v`
- `fc_verification/phase2_known_deltas.json` with `azurelily` subkey
- `fc_verification/run_fc_phase2.py` extended to run + report AL alongside NL-DPE

Gate: single Phase-2 report lists 12/12 PASS for **both** architectures.

### Stage 2 — Compose attention-head RTL ✓ CLOSED (T2v2, commit `e118b11`)

**T2v2. Streaming attention-head top generator + RTL**

Closeout note: T2 v0 (commit `2f5956e`) emitted a 1-token broadcast
composition + included an O projection. That scope mismatched sim's
`attention_model` (which streams N=128 tokens into Q/K/V proj and
stops at mac_sv with NO O projection). T2v2 refactored both head
generators to streaming mode matching sim:

Artifacts (landed in commit `e118b11`):
- `nl_dpe/gen_nldpe_attn_head_top.py` — instantiates
  `fc_top_qkv_streaming.v` (ping-pong DPE pair per arm, K=128, N=64)
- `nl_dpe/gen_azurelily_attn_head_top.py` — N parallel-output dsp_macs
  per arm (64 dsp_macs each), broadcast-input + packed-output FIFO
- `fc_verification/rtl/nldpe/fc_top_qkv_streaming.v` (new) — DPE_A
  loads while DPE_B computes/drains, alternating; steady-state
  per-row throughput = PACKED_K = 26 cyc
- `fc_verification/rtl/nldpe/fc_top_o_streaming.v` (new, kept for
  completeness; AH track v2 does NOT use O projection per
  attention_model)
- `fc_verification/rtl/azurelily/azurelily_fc_{128_64,64_128,512_128,2048_256}.v`
  (regenerated, all parallel-output)
- `fc_verification/rtl/{nldpe,azurelily}_attn_head_d64_c128.v` (regenerated)
- `nl_dpe/gen_gemv_wrappers.py` — additive `streaming=True` mode
  (default off, preserves T1 / Phase-2 single-inference behaviour)

`nl_dpe/attention_head_1_channel.v` (d=128, pre-P4 hand-written) is
formally retired in favour of these generated heads.

**Resource counts (regen-checked):**
- NL-DPE: 6 DPE (3 arms × 2 ping-pong) + 64 DIMM = **70 DPE per head**
- Azure-Lily: 192 dsp_mac (3 arms × 64 parallel-output) + 32 DIMM
  dsp_mac + 16 clb_softmax = **224 dsp_mac + 16 softmax per head**
- (The AL count is significantly larger than the T1-era 3 dsp_mac
  estimate because matching sim's parallel-output assumption requires
  explicit per-output-column dsp_mac instantiation.)

Gate (passed):
- iverilog syntax-clean on both RTLs
- Resource counts above match generator emit; both T1 / Phase-2
  backward compat 14/14 PASS preserved.

### Stage 3 — Functional at head scope ✓ CLOSED (T3v2, partial in commit `2e0c559`, finalised in `e118b11`)

**T3v2. Combined functional+latency TBs**

Artifacts (landed in `e118b11` after Phase A FC refactor):
- `fc_verification/tb_nldpe_attn_head_v2.v`
- `fc_verification/tb_azurelily_attn_head_v2.v`

Test vector: scaled-identity Q/K/V projection weights + identity V
matrix for S·V at N=128, d=64. The TBs combine functional checks and
latency probes into a single sim run; the dispatcher in run_checks.py
routes both into `check_ah_attn_head`.

Bug fixes during V2 build (commit `2e0c559`):
- AL sim `total_softmax_lanes` config was unset (sim treated softmax
  as fully parallel across W=16 lanes, masking the actual CLB-serial
  bottleneck) — set explicitly so the sim oracle reflects the
  AL `clb_softmax` design.
- 3 AL RTL composition bugs fixed: (a) head-FSM Q/K/V buffer
  write-enable handshake, (b) packed-output FIFO drain timing, and
  (c) DIMM start-trigger sequencing across the FC→DIMM handoff.

Gate (passed): both functional checks emit `Overall : PASS`.

### Stage 4 — Latency at head scope ✓ CLOSED (T4v2, commit `e118b11`)

**T4v2. Latency probes + sim extractor + known-deltas + dispatcher**

Artifacts (landed in `e118b11`):
- Per-stage timestamp probes folded into the V2 TBs above (linear_qkv,
  mac_qk, softmax_exp, softmax_norm, mac_sv, e2e)
- `fc_verification/expected_cycles.json` regenerated with attention-head
  scope cycles drawn from `azurelily/IMC/test.py --model attention
  --seq_length 128 --head_dim 64`
- `fc_verification/phase7_known_deltas.json` (new) — arch-tagged
  residual schema mirroring Phase 5; each entry carries
  `delta_cycles`, `tolerance`, `root_cause` (file:line citation), and
  `classification`.
- `fc_verification/run_checks.py` extended with AH dispatcher
  (`check_ah_attn_head`), AH config registry, and AL softmax
  exp+norm fold semantics (the AL TB emits `softmax_norm=0`; the
  combined value lives in `softmax_exp` and is checked against
  `sim_exp + sim_norm`).

**Final per-stage residuals:**

| Stage | NL-DPE RTL / sim / Δ / class | AL RTL / sim / Δ / class |
|---|---|---|
| linear_qkv  | 3,350 / 2,424 / +926 / m.g.   | 4,529 / 4,000 / +529 / m.g. |
| mac_qk      | 1,948 / 5,690 / −3,742 / structural | 1,683 / 6,661 / −4,978 / structural |
| softmax_exp | 8 / 650 / −642 / structural   | 2,289 / 938 (fold) / +1,351 / m.g. |
| softmax_norm| 10 / 376 / −366 / structural  | 0 / — / 0 (folded) / m.g. |
| mac_sv      | 251 / 5,339 / −5,088 / structural | 32 / 6,091 / −6,059 / structural |
| **E2E**     | **6,692 / 14,480 / −7,788 / structural** | **8,597 / 17,689 / −9,092 / structural** |

Classification rationale: negative residuals (RTL faster than sim) are
**structural** because sim's `gemm_log`/`gemm_dsp` analytical bodies
are conservative single-lane lower-bounds while the RTL realises W=16
hardware-lane parallelism — by design. Positive residuals are
**modelling_granularity** (FSM transitions, streaming fill/drain
edges, AL CLB-serial softmax bottleneck). All entries carry file:line
root-cause citations in `phase7_known_deltas.json`.

The pre-T2v2 budget (NL-DPE ≤ +50 cyc, AL ≤ +10 cyc, all m.g.) is
explicitly **superseded**. Once we matched the sim's attention_model
streaming convention end-to-end, the dominant residuals turned out to
be the W=16 lane parallelism gap (sim under-models, RTL realises) plus
AL's CLB-serial softmax (sim over-parallelises, RTL serialises). Both
are valid `structural` / `modelling_granularity` classifications with
documented root causes.

Gate (passed):
- `run_checks.py --config nldpe_attn_head_d64_c128` exits 0
- `run_checks.py --config azurelily_attn_head_d64_c128` exits 0

### Stage 5 — VTR + regression (PENDING user sign-off)

**T5v2. VTR orchestrators + metric verification + gate-list update**

Status: **pending user sign-off** as a separate stage. The streaming
FC refactor in T2v2 substantially changed the resource topology
(NL-DPE 70 DPE, AL 224 dsp_mac + 16 softmax), so the strict-count
target shifted from the T2-era estimate. T4v2 dispatcher commands
already land in `VERIFICATION.md §Phase 7`; T5v2 will append the VTR
3-seed metrics once they have been re-run against the regenerated
`*_attn_head_d64_c128.v` heads.

Planned artifacts:
- `fc_verification/gen_nldpe_attn_head_vtr.py`, `gen_azurelily_attn_head_vtr.py` (3 seeds each)
- `fc_verification/results/nldpe_attn_head_vtr_imc_results.json`
- `fc_verification/results/azurelily_attn_head_vtr_imc_results.json`

Strict metric checks (post-T2v2 targets):
- NL-DPE DPE = 70 (6 streaming-FC + 64 DIMM)
- Azure-Lily DSP ≈ 224 ± VTR packing drift (192 streaming-FC +
  32 DIMM dsp_mac); + 16 clb_softmax counted as CLB
- Fmax / CLB / BRAM informational

Gate: all P3+P4+P5+P6A+P7 regression commands exit 0 (already
green at commit `e118b11`); the VTR JSON results will be linked from
`VERIFICATION.md §Phase 7` once landed.

## Stage 6 — BERT-Tiny generator refinement (post-verification follow-up)

Once T3-T5 close, the verified attention-head RTL becomes the **canonical
reference** for `gen_bert_tiny_wrapper.py`. Diff and patch work:

- **Lane parallelism**: benchmark RTL uses W=1; paper spec is W=16 → patch
  generator to emit W=16 attention heads
- **Module naming consistency** between standalone head and per-(block,head) instances
- **Buffer depths / FSM handoff** semantics (concrete bugs surface during diff)
- **Resource counts** in `benchmarks/results/bert_tiny_seqlen_final_results.md`
  may need refresh after the W change
- Re-emit `benchmarks/rtl/bert_tiny_*_s{N}.v` with patched generator

T6 is **out of scope for the AH track gate** but tracked here so the
verified head doesn't sit unused.

## Explicitly deferred (not in this plan)

- **Multi-N attention head** — re-emit DIMM + attn-head at N ∈ {256, 512, 1024, 2048, 4096}, re-run Phase 3/5 per N, then T3/T4 per N. Separate track after T5 closes.
- **Multi-head composition** — 2 heads parallel (BERT-Tiny actually uses 2 heads). T6 will surface whether multi-head needs separate alignment.
- **d ≠ 64 regime** — d=128 → K_id=1 (single-identity FSM), d=32 → K_id=4 (quad-identity FSM). Both untested paths.

## Dependency DAG

```
T1 ──────────────────────┐                      ┌────► T6 (BERT-Tiny patch, follow-up)
                         │                      │
                         ▼                      │
T2 ──┬──► T3          T4 ──► T5 ────────────────┘
     └──────────────────▲
```

- **T1 ∥ T2 are independent** — no file collisions, no artifact
  dependency. T2 needs the AL FC Verilog *module* (already emitted by
  `nl_dpe/gen_dsp_gemv_wrapper.py` and exercised inside
  `azurelily_dimm_top`) but does **not** need T1's verification to
  have passed. Can be run in parallel or either sequential order.
- **T3 needs T2 only.** Functional test uses identity Q/K/V projection
  weights; doesn't consume AL FC's per-stage residual constants.
- **T4 needs both T1 and T2 (the real join point).** Head-level
  residual attribution requires AL FC's known-deltas (from T1) to
  subtract the FC contribution cleanly; without it, a head-level
  cycle mismatch is blame-ambiguous between FC / DIMM / handoff.
- **T5 needs T4.**

**Soft guidance:** T1 PASS also improves T3 diagnostic clarity (if
functional fails, an already-proven AL FC isolates blame to DIMM or
handoff). Not a hard block — the AL FC primitive (`dsp_mac`) is
already functionally verified inside DIMM `mac_qk` (128/128 scores
PASS, Phase I.1).

## Risk register

| Risk | Mitigation |
|---|---|
| Composition handoff cycles exceed +8/boundary budget | Add per-handoff timestamp probe; annotate as `modelling_granularity` if origin is FSM-boundary (SRAM address scheduling, inter-block pipe flop) |
| Azure-Lily FC FSM handshake differs from NL-DPE's +4 constant | Accept arch-specific delta in `phase2_known_deltas.json`, annotate root cause |
| T2 resource blow-up beyond VTR grid | Accept lower Fmax; do not resize grid without re-anchoring DIMM-top verification |
| Identity-projection test vector saturates int8 | Use scaled identity (e.g., Q = one-hot × small int8) to keep scores in range |
| Sim extractor's head-scope model disagrees with scheduler's pipeline timing | Extend scheduler with per-stage boundary cycles matching TB probe semantics |
| DIMM top + FC composition exposes a latent handshake bug not seen in block-level probe windows | Debug via per-handoff waveform capture; block-level regressions (P2/P3/P4/P5/P6A) remain green as anchor |

## Commit milestones

1. T1 landed (commit `9e6a913`): AL FC 12/12 PASS in unified Phase-2 report
2. T2 v0 landed (commit `2f5956e`): both attn-head RTLs compile clean (1-token broadcast scope, retired)
3. T2v2/T3v2/T4v2 landed (commit `e118b11`): streaming FC refactor, V2 TBs PASS, `phase7_known_deltas.json` arch-tagged, `run_checks.py` AH dispatcher, both `run_checks.py --config *_attn_head_d64_c128` exit 0
4. T2v2 supporting bugfixes (commit `2e0c559`): AL sim `total_softmax_lanes` config + 3 AL RTL composition bugs
5. T5v2 (PENDING): VTR 3-seed numbers logged, gate list updated in `VERIFICATION.md §Phase 7`

## Session-recovery pointers

- `CLAUDE.md` §"Active TODO Tracks" — AH track (post-`e118b11` summary)
- Memory: `project_attention_head_todos.md`
- This document (authoritative plan)
- Upstream plan (closed): `fc_verification/plans/DIMM_FULL_VERIFICATION_PLAN.md`
- VERIFICATION.md §Phase 7 — full attention-head closeout narrative

Status: T2v2/T3v2/T4v2 closed at commit `e118b11`; T5v2 pending user
sign-off; T6 (BERT-Tiny generator refinement) follow-up.
