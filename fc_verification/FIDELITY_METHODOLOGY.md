# Simulator Fidelity Methodology

**Status:** Principle-locked, pre-implementation
**Authored:** 2026-04-30
**Anchor for:** simulator + RTL re-org under "analytical sim, RTL ground truth, measured fidelity" framing

---

## What this document is, and isn't

This is the **principle anchor** for the simulator + RTL re-org. It pins
down the methodology decisions before any code changes. It is *not* an
implementation plan; per-stage implementation plans get written separately,
*against* this doc, after each STOP gate.

If a future change conflicts with anything here, this doc is updated
first; code follows.

---

## §1 Principle

```
SIMULATOR  =  pure analytical workload performance model
              (work-volume × lane-parallelism + DPE-axiom only)
              No FSM overhead. No handshake fudge. No drain calibration.

RTL        =  ground truth for cycle count
              (all real overhead from controllers, FSMs, SRAMs)

Fidelity   =  (RTL_cyc − Sim_cyc) / Sim_cyc
              Reported per stage, per workload, per architecture.
              No pre-committed threshold; the number is a measurement.
```

The simulator is **not** expected to predict overhead. The simulator is
a fast-but-optimistic predictor; RTL discloses overhead reality; the gap
is the simulator's measured error rate. We report this honestly.

The previous "RTL ≈ sim within Δ" framing — with classified residuals,
modelling-granularity vs structural deltas, tolerance bands, and
overhead-shaped sim constants — is **superseded**. That work calibrated
the simulator to RTL post-hoc, which made alignment partly tautological.

---

## §2 Two simulators, one workload spec

Workloads are defined architecture-agnostically (e.g., "GEMM
M=128 K=64 N=64", "Attention head N=128 d=64"). Each architecture has
its own simulator that consumes the same workload spec and emits cycles
using its own architecture parameters.

**NL-DPE simulator:**
- W concurrent DPE+tree lanes (W = `total_softmax_lanes`, e.g., 16)
- DPE primitive includes ACAM with log/exp mode

**Azure-Lily simulator:**
- DSP_WIDTH concurrent dsp_macs (DSP_WIDTH = 4 int8 pairs/cycle)
- DPE primitive is pure VMM (ADC mode only, no log/exp)

Both predictions take the form:

```
total_cycles = (number of lane-passes) × cycles_per_pass
where:
  number_of_lane_passes = ceil(total_DPE_work / (C × n_parallel_lanes))
  cycles_per_pass        = LOAD + COMPUTE + OUTPUT  (DPE-axiom)
```

`total_DPE_work` is workload-derived. `n_parallel_lanes` is
workload-allocated (see §7 Tiling). `cycles_per_pass` is per-arch
config-derived (see §3). **No FSM overhead, no calibration constants,
no handshake fudge anywhere in the simulator.**

---

## §3 DPE architecture: config-derived, single source of truth

A DPE = analog R × C crossbar + ACAM conversion stage + private CLB
reduction tree of width K (the workload's inner dim).

**ACAM has three modes** (architecture-level, programmable per-pass):
- **ADC mode**: digitize the analog VMM result (no nonlinearity).
  Used by VMM workloads.
- **Activation mode**: apply piecewise activation (ReLU, etc.).
- **Log/exp mode**: crossbar configured as C × C identity. Per pass,
  ACAM applies `exp(input[c])` or `log(input[c])` for c in 0..C-1.
  Used by DIMM workloads.

**Per-pass cost (energy + cycles) is full crossbar + full tree** —
the DPE pays for all C columns and full tree depth regardless of how
many slots are useful for the workload, and regardless of which ACAM
mode is active. This is intentional: it keeps the simulator honest
about what the hardware fundamentally costs.

**Per-architecture config JSON owns DPE primitive parameters:**

```json
// nl_dpe.json (illustrative; actual keys finalized at implementation)
{
  "dpe_buf_width":         40,
  "kernel_width":         128,
  "num_cols":             128,
  "compute_cycles":         3,
  "has_acam":            true,
  "total_softmax_lanes":   16
}

// azurelily.json
{
  "dpe_buf_width":         16,
  "kernel_width":         512,
  "num_cols":             128,
  "compute_cycles":        44,
  "has_acam":           false,
  "total_softmax_lanes":   16
}
```

**Single source of truth:** both the simulator and the DPE generator
consume the same JSON. The DPE primitive cycle accounting
(`LOAD_STROBES = ceil(KW × 8 / BUF)`, `OUTPUT_CYCLES = ceil(C × 8 / BUF)`,
`COMPUTE_CYCLES = config_value`) is derived from config at runtime / RTL
emission time. **No hand-pinned constants in either side. No
`sram_read_latency` magic numbers.**

**DPE generator** (`nl_dpe/gen_dpe_stub.py`, to be written):

```
Inputs:   per-arch config JSON
Outputs:  fc_verification/rtl/dpe_stub_<arch>.v   (one per arch)

Both arch outputs share:
  - Same handshake protocol (w_buf_en, reg_full, nl_dpe_control,
    dpe_done, MSB_SA_Ready)
  - Same FSM (S_IDLE → S_LOAD → S_WAIT_EXEC → S_COMPUTE → S_OUTPUT
    → S_DRAIN)
  - Same behavioral 1-clock VMM at fire time
  - Same parameterized LOAD_STROBES, OUTPUT_CYCLES derivations

Differ only in:
  - ACAM_MODE branch (log/exp computation) present iff config.has_acam == true
  - COMPUTE_CYCLES, BUF_WIDTH baked from config (no defparam)
```

The behavioral VMM stays: full-precision MAC at fire time,
non-synthesizable but functionally correct, no bit-serial decomposition.
The cycle-emulation FSM stays: load_strobes / compute_cycles /
output_cycles hold-counters. The math/timing separation already exists
in today's `dpe_stub.v` and is preserved.

---

## §4 Pipeline model

**Single-buffered with drain-load overlap.**

Per pass:
- LOAD    (L cycles): SRAM → DPE input buffer
- COMPUTE (C_cyc cycles): DPE bit-serial computation (uses `compute_cycles` from config)
- OUTPUT  (O cycles): DPE output buffer → SRAM

Drain of pass k can overlap with load of pass k+1 (independent SRAM
ports). Compute is sandwiched between load and output and cannot
overlap across passes.

**Steady-state interval = max(L, C_cyc, O).**

Total latency for M passes (per parallel lane):

```
T(M) = (L + C_cyc + O) + (M − 1) × max(L, C_cyc, O)
       └─── T_fill ───┘    └────── (M−1) × T_steady ──────┘
```

**This model applies to BOTH the simulator and the RTL.** The simulator
emits T(M) analytically. The RTL is designed so its FSM achieves the
same drain-load overlap. Cycle delta between sim and RTL is purely
FSM/control overhead.

Note: this supersedes the "Regime A vs Regime B" terminology used in
`paper/methodology/dpe_pipeline_model.md` (which was framed around an
older sim that lacked the drain-load overlap). Those labels are not
used here.

---

## §5 Workload classes

The DPE+ACAM hardware supports two workload classes. They differ in
ACAM mode, crossbar contents, and the workload→pass mapping.

### VMM workload — weight-persistent matmul (Stage 1 GEMM)

Crossbar holds an R × C **weight matrix W**. ACAM in **ADC mode**.
Per pass: input vector of R elements × W → output vector of C elements.

For matmul A[M × K] × W[K × N], with crossbar R × C:
- K-tile: `ceil(K / R)` passes per output row (along inner dim)
- N-tile: `ceil(N / C)` parallel DPE column-tiles
- Per output row m: `ceil(K/R) × ceil(N/C / n_parallel_lanes)` passes

**Total VMM passes per lane:**

```
passes_per_lane = M × ceil(K/R) × ceil(N / C / n_parallel_lanes)
total_cycles    = T(passes_per_lane)   per §4 pipeline model
```

This is what `imc_core.run_gemm` computes for latency. The current
`run_gemm` bug: latency only multiplies by M, missing `ceil(K/R)`.

### DIMM workload — log-domain matmul (Stage 2 Attention's mac_qk / mac_sv)

Crossbar configured as **C × C identity**. ACAM in **log/exp mode**.

Algorithm for matmul A[M × K] × B[K × N] (e.g., Q × K^T → score):
1. **Phase 1a (log A)**: convert A → log_A. Cost: M × K log ops on DPE.
2. **Phase 1b (log B)**: convert B → log_B. Cost: K × N log ops on DPE.
3. **Phase 2 (CLB add)**: for each (m, n, k), `log_A[m][k] + log_B[k][n]`. Off-DPE; CLB hardware.
4. **Phase 3 (exp + sum)**: for each (m, n, k), `exp(log_A + log_B)`, then sum_k. Cost: M × N × K exp ops on DPE; reduction in private tree.

**Total DPE work** = (M × K) + (K × N) + (M × N × K).
The exp phase dominates (factor of K over the log conversions).

---

## §6 Workload definitions

### Stage 1 — GEMM at multiple shapes (VMM workload)

Single matmul: `C[M × N] = A[M × K] @ B[K × N]`. Work = M × K × N MACs.

Shape sweep (specific shapes finalized at Stage 1 implementation, but
the spec is "general matmul, multiple shapes for scaling probes").

This stage covers what was previously called "projection" (single FC
arm), "FC", and the linear projection portions of attention. They are
all the same primitive at different shapes.

### Stage 2 — Attention head (DIMM workload)

Composition: `linear_Q + linear_K + linear_V → mac_qk → softmax → mac_sv`.
Linear_O is dropped per current sim convention (`azurelily/models/
attention.py:96-102` does not include it).

Concrete shape: N=128, d_model=128, d_head=64, W=16 softmax lanes.

The linear projections (Q/K/V) use **VMM workload**; mac_qk and mac_sv
use **DIMM workload**.

### Out of scope for this re-org

- BERT-Tiny composition (multi-head + FFN + LayerNorm + residual + embedding)
- Multi-N scaling (N=256, 512, 1024)
- d ≠ 64 regime (d=128, d=32)
- All non-attention BERT-Tiny modules

---

## §7 Tiling model

### VMM workload tiling

Straightforward: K-tile × N-tile per output row, distributed across
`n_parallel_lanes` DPEs. Pass count formula in §5.

Each DPE has private SRAM for its weight tile (W matrix slice) and its
output column slice. Inputs broadcast naturally (one input row goes to
all DPE column-tiles). Reduction across K-tiles happens in CLB adder
tree per DPE.

### DIMM workload tiling — W-lane row-parallel with shared B + broadcast

For matmul A[M × K] × B[K × N] with W parallel DPE+tree lanes:
- Each lane owns `ceil(M / W)` rows of output (M-axis row tiling).
- A is **row-tiled** across lanes (each lane has its M/W rows of A in
  private SRAM).
- B is **shared** — single SRAM holds full B, content broadcast to all
  W lanes via fanout bus during phase 3.
- Each lane has its own private CLB reduction tree of width K.

**Why shared B (not replicated):** at W=16, replicating B would cost
~16 × (K × N + N × d) ≈ ~256KB BRAM for attention head, ≈ 37% of the
120 × 120 grid's 472 BRAMs. Sharing B with broadcast reduces this to
single-copy storage (~8KB per matrix) plus per-lane scratch — total
~9% BRAM utilization. Same cycle count (broadcast bus delivers same
data to all lanes simultaneously; W lanes work in lockstep on (n, k)
iteration but on their own m-rows).

**Per-lane DPE pass count:**

```
phase_1a = ceil((M/W × K) / C)              # log A subset, parallel across lanes
phase_1b = ceil((K × N) / C)                # log B once globally (shared)
phase_3  = ceil((M/W × N × K) / C)          # exp+sum, parallel across lanes
                                              (private tree handles reduction)

passes_per_lane = phase_1a + phase_1b + phase_3
total_cycles    = T(passes_per_lane)         per §4 pipeline model
```

For attention head N=128, d=K=64, W=16, C=128:
- phase_1a = ceil(8 × 64 / 128)   = 4 passes
- phase_1b = ceil(64 × 128 / 128) = 64 passes
- phase_3  = ceil(8 × 128 × 64 / 128) = 512 passes
- passes_per_lane = 4 + 64 + 512 = 580 passes

**Memory model (DIMM):**
- Lane-private SRAM: A subset (M/W × K), score buffer (M/W × N),
  softmax intermediate (M/W × N), output buffer.
- Shared SRAM: full B (K × N) + broadcast bus delivering one element
  per cycle to all W lanes.

**RTL implementation considerations** (forward-looking, for Stage 2 RTL):
- Synchronized (n, k) iteration FSM across W lanes (SIMD-style).
- Single shared B SRAM with W-way fanout (physical wire fanout, no
  switching network needed since all lanes consume identical data each
  cycle).
- Per-lane FSM differs only in m-axis indexing into private SRAMs.

### Why W-lane row-parallel for DIMM

1. Matches the row-parallel softmax structure already pinned (each
   softmax lane owns the same M/W rows as the upstream mac_qk lane).
2. Single allocation knob (`total_softmax_lanes`) controls W across
   mac_qk, softmax, and mac_sv — unified attention head structure.
3. Reduction tree per lane is naturally bounded by K (workload's inner
   dim), independent of M, N.
4. Scales cleanly to larger N (shared B SRAM scales as N × d, not as
   W × N × d).

---

## §8 RTL expected scope

Each top module contains:
- DPE primitive instance(s) — generated by DPE generator (§3)
- Handshake interconnect
- FSM (load → fire → compute → output → drain, single-buffered with
  drain-load overlap per §4)
- Necessary storage (lane-private SRAMs + shared SRAMs per §7)

Each top module **does not** contain:
- Cosmetic alignment knobs (`BACK_TO_BACK_MODE`, `WIDE_ADDR_MODE`,
  `SCORE_BACK_TO_BACK_MODE`, `WSUM_BACK_TO_BACK_MODE`, etc.)
- Drain states added to match sim cycle counts
- Per-Q-row K/V amortization unless architecturally correct
- Parameterized fudge factors

**Rule:** if a piece of RTL exists today only because it was added
during calibration, it gets deleted in the rewrite.

---

## §9 Fidelity metric

```
fidelity = (RTL_cyc − Sim_cyc) / Sim_cyc

Positive = simulator is optimistic (under-predicts cycles)
Negative = simulator is pessimistic (over-predicts cycles)
```

Reported:
- Per architecture (NL-DPE, Azure-Lily) — separately
- Per workload shape
- Per stage (only where stages are independently observable in RTL)
- E2E (always)

**No pre-committed acceptable threshold.** We observe → diagnose → decide:
- If gap is plausible overhead (FSM glue, drain, broadcast sync, SRAM
  access): accept, document.
- If gap is large (> 30%, say): inspect — is the simulator's work
  model wrong? Is the RTL doing extra work it shouldn't?

The fidelity number is a *measurement*, not a target. The paper section
that reports it is honest about what it represents.

---

## §10 What dies, what survives

### Deprecated (delete during re-org)

| Path | Status |
|---|---|
| `fc_verification/phase{2,3,5,7}_known_deltas.json` | DELETED 102a52b |
| `fc_verification/expected_cycles.json` | DELETED 102a52b |
| `fc_verification/per_stage_targets.json` | DELETED 102a52b |
| `fc_verification/known_count_deltas.json` | DELETED 102a52b |
| `fc_verification/expected_counters.json` | DELETED 102a52b |
| `fc_verification/functional_whitelist.json` | DELETED 102a52b |
| `fc_verification/run_checks.py` | DELETED 102a52b |
| `fc_verification/run_fc_phase2.py` | DELETED 102a52b |
| `fc_verification/tb_*.v` | DELETED 102a52b |
| `fc_verification/rtl/*` (alignment-era) | DELETED 102a52b |
| `nl_dpe/gen_*` (alignment-era) | DELETED 102a52b |
| `block_comp_apr_11/rtl/setup{0..5}/fc_*.v` | DELETE / archive (deferred) |
| `fc_verification/DIMM_pipeline_model_vs_rtl.md` | ARCHIVED 102a52b |
| `fc_verification/VERIFICATION.md` | ARCHIVED 102a52b |
| Submodule (`azurelily/`) post-`c15797f` history | REVERTED ec7ccd5 |

### Concepts deprecated

- "K_id" / "K-identity" / "single-identity" / "dual-identity" — replaced
  by §3's "ACAM modes" framing and §7's tiling. Per-pass cost is full
  crossbar regardless of how many output slots are useful; the
  simulator counts work in total exp/log ops divided by C.
- "Regime A" / "Regime B" labels from `paper/methodology/dpe_pipeline_model.md`
  — replaced by §4's single pipeline model (drain-load overlap).
- `sram_read_latency` magic-number constant — removed entirely.
  No magic-number physics in the simulator.

### Survives (keep, possibly with edits)

| Path | Status |
|---|---|
| `azurelily/IMC/` workload definitions (`attention.py`, `bert_tiny.py`) | KEEP — pure-work models, principle-aligned |
| `azurelily/IMC/imc_core/` | KEEP — architecture configs |
| `azurelily/IMC/scheduler_stats/scheduler.py` | KEEP after §11 strip pass |
| Per-arch JSON configs (`nl_dpe.json`, `azurelily.json`) | KEEP, become single source of truth (§3) |
| `gemv_dse.py` (DSE driver) | KEEP — DSE flow continues, decoupled from cycle alignment |
| `nl_dpe/run_vtr.py`, `nl_dpe/area_power.py`, `nl_dpe/gen_arch_xml.py` | KEEP — VTR + area/power infrastructure unchanged |

---

## §11 Per-stage verification workflow

```
1. Workload spec confirmed (shape, parameters)
2. Simulator: predicts cycles for the workload
   (analytical: §2 formula + §4 pipeline model + §7 tiling)
3. RTL: top module designed
   (handshake + FSM + DPE primitive, minimal scope per §8)
4. RTL generator: emits the top module from architecture config
5. TB: combined functional + cycle measurement on generated RTL
   (one TB per config; functional check uses real weights/inputs and
    compares against numpy oracle; cycle measurement parses TB output)
6. Fidelity computed: (RTL_cyc − Sim_cyc) / Sim_cyc, reported
7. Inspection:
   - Is sim's work model correct per workload definition?
   - Is RTL doing only expected work per §8 scope?
   - Does the gap match plausible overhead structure?
8. Decide: ship the fidelity number, fix the work model, or both.
9. STOP gate — proceed to next stage only after explicit confirmation.
```

Every STOP is an explicit gate. Do not proceed without confirmation.

---

## §12 Submodule revert plan (Option B) — EXECUTED

**Target:** parent commit `c15797f^` (predecessor of "Phase 1 (sim):
azurelily submodule → Regime B gemm_log").

**Executed at:** parent commit `ec7ccd5` (2026-04-30). Submodule reset
to `8cae3ea` and cherry-picked: `690d7fe` (Phase 2.1 --batch M),
`65750d8` (IMC parallelism: gemm_log intra-row + gemm_dsp
parallel-output + AL softmax W=16), `993aec5` (AH softmax parallelism
fix).

**Surfaced findings during simulator review** (§11 inspection of the
post-revert simulator):

1. `total_softmax_lanes` config field referenced but missing from JSON.
2. `sram_read_latency` config field referenced but missing from JSON
   (and identified as magic-number guess — to be deleted entirely).
3. K_id / "single-identity" terminology persisting in code comments
   (resolved by §3 + §7 reformulation).
4. Softmax `cols_per_lane = cols // W` double-counts parallelism
   (correction: row-parallel only per §7).
5. `imc_core.run_gemm` latency missing `k_tile` multiplier — real bug.
6. Regime B-shaped formula leaked into `gemm_log` via cherry-pick of
   `65750d8`. Per §4, the drain-load overlap *is* the right model — so
   the pipelined formula stays, but the K_id factor in `n_passes`
   computation is dropped.

These are addressed in the §13 code work below.

---

## §13 Execution flow

The methodology is executed in this order:

1. **Cleanup.** Delete deprecated artifacts and archive historical
   docs. ✅ Done at commit `102a52b`.

2. **Simulator review.** Submodule revert + post-revert simulator
   inspection. ✅ Done at commit `ec7ccd5` + this doc's §12.

3. **Simulator code work** (next). Apply the surfaced findings to the
   reverted simulator: drop K_id, drop `sram_read_latency`, fix
   `run_gemm` k_tile bug, cleanup softmax, add `total_softmax_lanes`
   to configs. Code plan to be discussed and locked before changes.

4. **Behavioral DPE generator.** Author `nl_dpe/gen_dpe_stub.py`
   per §3. Generator emits `dpe_stub_nldpe.v` and
   `dpe_stub_azurelily.v` from per-arch config JSON.

5. **DPE functional verification.** A simple TB drives each
   generated DPE with weights and inputs, checks the VMM result
   matches a numpy oracle. No top module yet.

6. **Stage 1 (GEMM).** Per §6 + §11: top module design, generator,
   TB, fidelity measurement, inspection.

7. **Stage 2 (Attention head).** Same pattern, composing the
   verified GEMM primitive into the attention pipeline.

Each step finishes before the next starts. Inspection at every
boundary is part of the methodology, not an exception to it.

---

## Glossary

- **Work-volume:** number of MACs (or higher-level ops) the workload
  fundamentally requires. Independent of architecture.
- **Lane parallelism:** number of MACs an architecture computes per
  cycle in steady state. Architecture-specific.
- **DPE-axiom:** per-DPE-fire latency =
  `LOAD_STROBES + COMPUTE_CYCLES + OUTPUT_CYCLES`. Derived from
  per-arch config; physics-bound.
- **Pipeline model:** §4 — single-buffered with drain-load overlap.
  Both sim and RTL implement this. Cycle delta = FSM/control overhead.
- **VMM workload:** weight-persistent matmul; ACAM in ADC mode;
  crossbar holds W matrix.
- **DIMM workload:** log-domain matmul; ACAM in log/exp mode;
  crossbar configured as C × C identity.
- **W lanes:** parallel DPE+tree units (config field
  `total_softmax_lanes`). Each lane owns M/W rows of DIMM output and
  the same M/W rows of softmax output downstream.
- **Shared B + broadcast:** §7 tiling pattern for DIMM. B held in
  single shared SRAM; broadcast bus fans out to all W lanes
  simultaneously (lockstep (n, k) iteration; lanes differ only on
  m-axis).
- **Fidelity:** relative under-prediction of the simulator vs RTL.
  `(RTL − Sim) / Sim`, reported per stage / workload / arch.
- **Overhead:** cycles spent in FSM transitions, handshake gaps,
  drain states, broadcast sync, etc. **Not modeled in simulator;**
  captured by RTL; reflected in fidelity number.
- **STOP gate:** explicit confirmation point before proceeding to
  the next sequence step.
