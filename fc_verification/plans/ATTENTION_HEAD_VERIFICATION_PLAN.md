# Attention Head RTL Verification + Latency Alignment Plan

**Opened:** 2026-04-24
**Last revised:** 2026-04-24 (post T1+T2 closeout, scope clarification)
**Precedes:** closes the V-model loop opened by `DIMM_FULL_VERIFICATION_PLAN.md` (DIMM top, Phase H–N / 3–6A)
**Status:** T1 ✓, T2 ✓, T3-T5 active

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
| Q/K/V/O projection | DPE in ACAM mode (V=H=1, ACAM-eligible, no CLB act) | single `dsp_mac` w/ K=PACKED_K |
| DIMM mac_qk | DPE w/ K_qkt=2 dual-identity, **log-domain** | `dsp_mac` w/ K=⌈d/4⌉=16, **linear-domain** |
| DIMM softmax | DPE(I\|exp) for sm_exp + DPE(I\|log) for normalize | `clb_softmax` (CLB FSM with SRAM, recip + DSP normalize) |
| DIMM mac_sv | DPE w/ wsum_log + wsum_exp 128×128 (post-Phase-4), log-domain | `dsp_mac` w/ K=⌈N/4⌉=32, linear-domain |
| Pipeline regime | Regime B Layout A (`T = L_A·M + O`) | Regime A serial |
| Sim model | `gemm_log` in `azurelily/IMC/peripherals/fpga_fabric.py` | `gemm_dsp` in same file |
| DPE/DSP per head | 4 proj + 64 DIMM = **68 DPE** | 4 proj + 32 DIMM + 16 clb_softmax = **36 dsp_mac** + 16 softmax |

**Configs verified:**
- `nldpe_attn_head_d64_c128`
- `azurelily_attn_head_d64_c128`

## Prerequisites verified

- **FC RTL↔sim:** NL-DPE ✓ (12/12 Phase 2), Azure-Lily ✓ (T1 closed, 14/14 unified gate at commit `9e6a913`)
- **DIMM top RTL↔sim:** NL-DPE ✓ (Phase 3/4, E2E +42 cyc, m.g.), Azure-Lily ✓ (Phase 5/6A, E2E +2 cyc, m.g.)
- **Composed attn-head RTL:** NL-DPE ✓, Azure-Lily ✓ (T2 closed, both compile clean at commit `2f5956e`)

## Stages

### Stage 1 — Close block-level parity

**T1. Azure-Lily FC Phase-2 harness**

Artifacts:
- `fc_verification/tb_azurelily_fc_512_128.v`
- `fc_verification/tb_azurelily_fc_2048_256.v`
- `fc_verification/phase2_known_deltas.json` with `azurelily` subkey
- `fc_verification/run_fc_phase2.py` extended to run + report AL alongside NL-DPE

Gate: single Phase-2 report lists 12/12 PASS for **both** architectures.

### Stage 2 — Compose attention-head RTL

**T2. Attention-head top generator + RTL**

Artifacts:
- `nl_dpe/gen_nldpe_attn_head_top.py` (new, mirrors `gen_dimm_nldpe_top.py`)
- `nl_dpe/gen_azurelily_attn_head_top.py` (new, mirrors `gen_dimm_azurelily_top.py`)
- `fc_verification/rtl/nldpe_attn_head_d64_c128.v`
- `fc_verification/rtl/azurelily_attn_head_d64_c128.v`

Wiring: Q/K/V FC → DIMM top → O FC with same `valid/ready_n/data_out`
interface as DIMM top. `ready_n = 1'b0` hardwired (no back-pressure,
matches DIMM top convention).

Decision: retire `nl_dpe/attention_head_1_channel.v` (d=128, pre-P4
hand-written — stale). Regenerate from T2 generator.

Gate:
- iverilog syntax-clean on both RTLs
- Resource-count sanity: NL-DPE ≈ 96 DPEs/head (16 proj + 64 DIMM + 16 O-proj), AL ≈ 40 DSPs/head (4 proj + 32 DIMM + 4 O-proj)

### Stage 3 — Functional at head scope

**T3. Functional TBs**

Artifacts:
- `fc_verification/tb_nldpe_attn_head_functional.v`
- `fc_verification/tb_azurelily_attn_head_functional.v`

Test vector: scaled-identity Q/K/V projection weights + identity V-matrix
for S·V + one-hot input → hand-computable output at N=128, d=64.

Gate: both archs produce the **same** numerical output within int8
tolerance. This is the first cross-architecture functional equivalence
check in the project.

### Stage 4 — Latency at head scope

**T4. Latency TBs + sim extractor + known-deltas**

Artifacts:
- `fc_verification/tb_nldpe_attn_head_latency.v`, `tb_azurelily_attn_head_latency.v`
  - Timestamp probes: 5 stages (FC_Q, DIMM_score, DIMM_softmax, DIMM_wsum, FC_O) + 2 handoff boundaries (FC_QKV → DIMM, DIMM → FC_O)
- `fc_verification/gen_expected_cycles.py` extended to emit head-scope cycles (wraps `azurelily/IMC/scheduler_stats/scheduler.py::_run_attention_pipeline`)
- `fc_verification/phase7_known_deltas.json` (new) with residual schema mirroring Phase 5
- `fc_verification/run_checks.py` dispatcher updated for `*_attn_head_d64_c128` configs

Residual budget (classifications must all be `modelling_granularity`):
- NL-DPE E2E ≤ +50 cyc (≈ 42 DIMM + 2 × ~4 handoff)
- Azure-Lily E2E ≤ +10 cyc (≈ 2 DIMM + 2 × ~4 handoff)

Gate:
- `run_checks.py --config nldpe_attn_head_d64_c128` exits 0
- `run_checks.py --config azurelily_attn_head_d64_c128` exits 0

### Stage 5 — VTR + regression

**T5. VTR orchestrators + metric verification + gate-list update**

Artifacts:
- `fc_verification/gen_nldpe_attn_head_vtr.py`, `gen_azurelily_attn_head_vtr.py` (3 seeds each)
- `fc_verification/results/nldpe_attn_head_vtr_imc_results.json`
- `fc_verification/results/azurelily_attn_head_vtr_imc_results.json`

Strict metric checks:
- NL-DPE DPE = 96 exact
- Azure-Lily DSP ≈ 40 ± VTR packing drift
- Fmax / CLB / BRAM informational

Append to gate list in `fc_verification/VERIFICATION.md §Phase 5`:
```
python3 fc_verification/run_checks.py --config nldpe_attn_head_d64_c128
python3 fc_verification/run_checks.py --config azurelily_attn_head_d64_c128
```

Gate: all 7 regression commands exit 0.

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

1. T1 landed: AL FC 12/12 PASS in unified Phase-2 report
2. T2.NL-DPE landed: `nldpe_attn_head_d64_c128.v` compiles clean
3. T2.AL landed: `azurelily_attn_head_d64_c128.v` compiles clean
4. T3 landed: cross-arch functional equivalence PASS
5. T4.NL-DPE landed: `run_checks.py --config nldpe_attn_head_d64_c128` exits 0
6. T4.AL landed: `run_checks.py --config azurelily_attn_head_d64_c128` exits 0
7. T5 landed: VTR 3-seed numbers logged, gate list updated

## Session-recovery pointers

- `CLAUDE.md` §"Active TODO Tracks" — AH track
- Memory: `project_attention_head_todos.md`
- This document (authoritative plan)
- Upstream plan (closed): `fc_verification/plans/DIMM_FULL_VERIFICATION_PLAN.md`

Status: Stage 1 (T1) ready to start — no blocking dependencies.
