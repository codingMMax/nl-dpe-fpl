# Attention Head RTL Verification + Latency Alignment Plan

**Opened:** 2026-04-24
**Precedes:** closes the V-model loop opened by `DIMM_FULL_VERIFICATION_PLAN.md` (DIMM top, Phase H–N / 3–6A)
**Status:** locked, pre-implementation

## Ultimate goal

End-to-end RTL↔sim cycle-accurate alignment for a composed attention head
(FC_Q/K/V → DIMM top → FC_O), for both NL-DPE and Azure-Lily, with every
residual classified `modelling_granularity`. Symmetric probe methodology,
harness, and known-deltas schema across both architectures.

## Scope anchor (frozen)

Single config point — inherits the already-verified DIMM-top surface.

| Parameter | Value | Source |
|---|---|---|
| N (seq len) | 128 | inherits verified DIMM top |
| d (head dim) | 64 | BERT-Tiny `d_head` |
| C (crossbar cols) | 128 | NL-DPE DIMM crossbar |
| R (crossbar rows) | 1024 | NL-DPE DPE config |
| W (parallel lanes) | 16 | paper §3 W=16 DIMM |
| W_DPE (bus width) | 40 bit | 5 × int8 packed |
| K_id (dual-identity) | 2 | = C / d |

Configs to verify:
- `nldpe_attn_head_d64_c128`
- `azurelily_attn_head_d64_c128`

## Prerequisites verified today

- **FC RTL↔sim:** NL-DPE ✓ (12/12 Phase 2, `phase2_fc_report.md`), Azure-Lily ✗ (no standalone harness — T1 closes this)
- **DIMM top RTL↔sim:** NL-DPE ✓ (Phase 3/4, E2E +42 cyc, m.g.), Azure-Lily ✓ (Phase 5/6A, E2E +2 cyc, m.g.)

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

## Explicitly deferred (not in this plan)

- **Multi-N attention head** — re-emit DIMM + attn-head at N ∈ {256, 512, 1024}, re-run Phase 3/5 per-N, then compose. Separate track after Stage 5 closes. Prerequisite for paper-wide seq_len scaling story.
- **Multi-head + BERT-block composition** — 2 heads parallel, LayerNorm, residual, embedding. Next V-model layer up. Partly covered by monolithic `benchmarks/rtl/bert_tiny_*.v` (VTR synthesis only, no cycle-accurate alignment).
- **d ≠ 64 regime** — d=128 → K_id=1 (single-identity FSM), d=32 → K_id=4 (quad-identity FSM). Both untested paths.

## Dependency DAG

```
T1 ──────────────────────┐
                         │
                         ▼
T2 ──┬──► T3          T4 ──► T5
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
