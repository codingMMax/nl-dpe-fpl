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
M=128 K=64 N=64"). Each architecture has its own simulator that
consumes the same workload spec and emits cycles using its own
architecture parameters.

**NL-DPE simulator:**
- W concurrent softmax/output lanes (W=16)
- Per-lane work-per-fire: depends on the analog crossbar's
  per-cell packing convention — held open pending simulator review,
  not pinned in this doc
- DPE primitive includes ACAM nonlinearity (post-VMM)

**Azure-Lily simulator:**
- DSP_WIDTH concurrent dsp_macs (DSP_WIDTH=4 int8 pairs/cycle)
- DPE primitive is pure VMM, no ACAM

Both predictions take the form:

```
cycles_workload = ceil(work_volume / lanes_per_fire)         # concurrent-fire model
                + DPE_axiom × number_of_fires                # per-fire physics overhead
```

`work_volume` is workload-derived (e.g., M × K × N MACs for GEMM).
`lanes_per_fire` is architecture-derived. `DPE_axiom` is per-arch
config-derived (see §3). **No FSM overhead, no calibration constants,
no handshake fudge anywhere in the simulator.**

---

## §3 DPE primitive: config-derived, single source of truth

Per-architecture config JSON owns DPE primitive parameters:

```json
// nl_dpe.json (illustrative; actual keys finalized at implementation)
{
  "dpe_buf_width": 40,
  "kernel_width": 128,
  "num_cols": 128,
  "compute_cycles": 3,
  "has_acam": true
}

// azurelily.json
{
  "dpe_buf_width": 16,
  "kernel_width": 512,
  "num_cols": 128,
  "compute_cycles": 44,
  "has_acam": false
}
```

**Single source of truth:** both the simulator and the DPE generator
consume the same JSON. The DPE primitive cycle accounting
(`LOAD_STROBES = ceil(KW × 8 / BUF)`, `OUTPUT_CYCLES = ceil(C × 8 / BUF)`,
`COMPUTE_CYCLES = config_value`) is derived from config at runtime / RTL
emission time. No hand-pinned constants in either side.

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
  - ACAM_MODE branch present iff config.has_acam == true
  - COMPUTE_CYCLES, BUF_WIDTH baked from config (no defparam)
```

The behavioral VMM stays: full-precision MAC at fire time,
non-synthesizable but functionally correct, no bit-serial decomposition.
The cycle-emulation FSM stays: load_strobes / compute_cycles / output_cycles
hold-counters. The math/timing separation already exists in today's
`dpe_stub.v` and is preserved.

What gets stripped from today's `dpe_stub.v`:
- ACAM_MODE post-processing in the no-ACAM (Azure-Lily) variant
- The `weight_wen` testbench-only port (TBs use hierarchical force)
- `initial` weight-zero block (rely on hierarchical force per-test)

---

## §4 Workload definitions

### Stage 1 — GEMM at multiple shapes

Single matmul: `C[M×N] = A[M×K] @ B[K×N]`. Work = M × K × N MACs.

Shape sweep (specific shapes finalized at Stage 1 implementation, but
the spec is "general matmul, multiple shapes for scaling probes").

This stage covers what was previously called "projection" (single FC
arm), "FC", and the linear projection portions of attention. They are
all the same primitive at different shapes.

### Stage 2 — Attention head

Composition: `linear_Q + linear_K + linear_V → mac_qk → softmax → mac_sv`.
Linear_O is dropped per current sim convention (`azurelily/models/
attention.py:96-102` does not include it).

Concrete shape: N=128, d_model=128, d_head=64, W=16 softmax lanes.
Per-cell K-packing convention held open pending simulator review.

### Out of scope for this re-org

- BERT-Tiny composition (multi-head + FFN + LayerNorm + residual + embedding)
- Multi-N scaling (N=256, 512, 1024)
- d ≠ 64 regime (d=128, d=32)
- All non-attention BERT-Tiny modules

---

## §5 RTL expected scope

Each top module contains:
- DPE primitive instance(s) — generated by DPE generator
- Handshake interconnect
- FSM (e.g., load → fire → compute → output → drain)
- Necessary storage (input/output SRAMs, weight stores)

Each top module **does not** contain:
- Cosmetic alignment knobs (`BACK_TO_BACK_MODE`, `WIDE_ADDR_MODE`,
  `SCORE_BACK_TO_BACK_MODE`, `WSUM_BACK_TO_BACK_MODE`, etc.)
- Drain states added to match sim cycle counts
- Per-Q-row K/V amortization unless architecturally correct
- Parameterized fudge factors

**Rule:** if a piece of RTL exists today only because it was added
during calibration, it gets deleted in the rewrite.

---

## §6 Fidelity metric

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
- If gap is plausible overhead (FSM glue, drain, SRAM access): accept,
  document.
- If gap is large (> 30%, say): inspect — is the simulator's work
  model wrong? Is the RTL doing extra work it shouldn't?

The fidelity number is a *measurement*, not a target. The paper section
that reports it is honest about what it represents.

---

## §7 What dies, what survives

### Deprecated (delete during re-org)

| Path | Status |
|---|---|
| `fc_verification/phase{2,3,5,7}_known_deltas.json` | DELETE |
| `fc_verification/expected_cycles.json` | DELETE |
| `fc_verification/per_stage_targets.json` | DELETE |
| `fc_verification/known_count_deltas.json` | DELETE |
| `fc_verification/expected_counters.json` | DELETE |
| `fc_verification/functional_whitelist.json` | DELETE |
| `fc_verification/run_checks.py` | DELETE (replaced by per-stage scripts) |
| `fc_verification/run_fc_phase2.py` | DELETE |
| `fc_verification/tb_*.v` | DELETE all |
| `fc_verification/rtl/*_attn_head_*.v` | DELETE |
| `fc_verification/rtl/*_dimm_*.v` (NL + AL) | DELETE |
| `fc_verification/rtl/azurelily/azurelily_fc_*.v` | DELETE |
| `fc_verification/rtl/dimm_*.v` (mac_qk/sv/softmax submodules) | DELETE |
| `fc_verification/rtl/fc_128_128_*.v` | DELETE |
| `fc_verification/rtl/dimm_pipeline_*.v` | DELETE |
| `nl_dpe/gen_attention_wrapper.py` | DELETE |
| `nl_dpe/gen_*_attn_head_top.py` (NL + AL) | DELETE |
| `nl_dpe/gen_dimm_*.py` (NL + AL DIMM tops + components) | DELETE |
| `nl_dpe/gen_gemv_wrappers.py` | DELETE (replaced by GEMM generator) |
| `nl_dpe/gen_azurelily_fc_wrapper.py` | DELETE |
| `nl_dpe/gen_gemm_wrapper.py` | DELETE |
| `block_comp_apr_11/rtl/setup{0..5}/fc_*.v` | DELETE / archive |
| `fc_verification/DIMM_pipeline_model_vs_rtl.md` | ARCHIVE (historical) |
| `fc_verification/VERIFICATION.md` Phase H–N narrative | ARCHIVE (historical) |
| Submodule (`azurelily/`) post-`c15797f` history | REVERT to pre-`c15797f` |

### Survives (keep, possibly with edits)

| Path | Status |
|---|---|
| `fc_verification/dpe_stub.v` | REPLACED by generator output (§3) |
| `azurelily/IMC/` workload definitions (`attention.py`, `bert_tiny.py`) | KEEP — pure-work models, principle-aligned |
| `azurelily/IMC/imc_core/` | KEEP — architecture configs |
| `azurelily/IMC/scheduler_stats/scheduler.py` | KEEP after revert + strip |
| Per-arch JSON configs (`nl_dpe.json`, `azurelily.json`) | KEEP, become single source of truth (§3) |
| `gemv_dse.py` (DSE driver) | KEEP — DSE flow continues, decoupled from cycle alignment |
| `nl_dpe/run_vtr.py`, `nl_dpe/area_power.py`, `nl_dpe/gen_arch_xml.py` | KEEP — VTR + area/power infrastructure unchanged |

---

## §8 Per-stage verification workflow

```
1. Workload spec confirmed (shape, parameters)
2. Simulator: predicts cycles for the workload
   (analytical: work-volume / lanes + DPE-axiom)
3. RTL: top module designed
   (handshake + FSM + DPE primitive, minimal scope)
4. RTL generator: emits the top module from architecture config
5. TB: combined functional + cycle measurement on generated RTL
   (one TB per config; functional check uses real weights/inputs and
    compares against numpy oracle; cycle measurement parses TB output)
6. Fidelity computed: (RTL_cyc − Sim_cyc) / Sim_cyc, reported
7. Inspection:
   - Is sim's work model correct per workload definition?
   - Is RTL doing only expected work per §5 scope?
   - Does the gap match plausible overhead structure?
8. Decide: ship the fidelity number, fix the work model, or both.
9. STOP gate — proceed to next stage only after explicit confirmation.
```

Every STOP is an explicit gate. Do not proceed without confirmation.

---

## §9 Submodule revert plan (Option B)

**Target:** parent commit `c15797f^` (predecessor of "Phase 1 (sim):
azurelily submodule → Regime B gemm_log").

**Why this line:** Regime B was added because RTL showed sequential
firing. But sequential firing is an FSM artifact (controllers serialize
lanes). Architecturally, both NL-DPE ACAM cells and Azure-Lily dsp_macs
*can* fire concurrently. Regime A models the concurrent-fire ideal;
Regime B leaks FSM shape into the simulator. Reverting past Regime B
restores the analytical purity §1 requires.

**Mechanical steps** (not yet executed; recorded here for traceability):

```
git checkout c15797f^ -- azurelily   # bump submodule pointer
cd azurelily
git checkout c15797f^                 # detach submodule HEAD
cd ..
git status                            # verify pointer change
git diff azurelily                    # inspect what we're rewinding
```

**Post-revert strip pass:** even at `c15797f^`, residual overhead
constants from older alignment work (e.g. `659b8bd` "DIMM simulator
RTL-matched: 66 sim vs 65 RTL cycles/pair", `d763db3` "DIMM simulator:
analytical pipeline model with clock domain crossing docs") may still
live in the simulator. After revert, inspect and strip:

- `compute + 4` handshake fudges
- `bridge_cyc` / `phase_drain_cyc` / `per_row_glue_cyc` (if present)
- Any `_with_kv_amortization` flags (probably already gone post-revert)

The strip surface is small once Regime B is gone.

---

## §10 Execution flow

The methodology is executed in this order:

1. **Cleanup.** Delete deprecated artifacts and archive historical
   docs. Packaged as a bash script (`fc_verification/cleanup.sh`)
   for review and one-shot execution.

2. **Simulator review.** The submodule revert (§9) bumps the
   simulator pointer to `c15797f^`. The user inspects the
   post-revert simulator code directly to verify the analytical
   model matches §1-§2 principle.

3. **Behavioral DPE generator.** Author `nl_dpe/gen_dpe_stub.py`
   per §3. Generator emits `dpe_stub_nldpe.v` and
   `dpe_stub_azurelily.v` from per-arch config JSON.

4. **DPE functional verification.** A simple TB drives each
   generated DPE with weights and inputs, checks the VMM result
   matches a numpy oracle. No top module yet.

5. **Stage 1 (GEMM).** Per §4 and §8: top module design, generator,
   TB, fidelity measurement, inspection.

6. **Stage 2 (Attention head).** Same pattern, composing the
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
- **Fidelity:** relative under-prediction of the simulator vs RTL.
  `(RTL − Sim) / Sim`, reported per stage / workload / arch.
- **Overhead:** cycles spent in FSM transitions, handshake gaps,
  drain states, SRAM access latencies, etc. **Not modeled in
  simulator;** captured by RTL; reflected in fidelity number.
- **STOP gate:** explicit confirmation point before proceeding to
  the next sequence step.
