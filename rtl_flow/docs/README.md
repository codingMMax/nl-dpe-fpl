# NL-DPE / Azure-Lily Verification Framework

End-to-end verification of the **faithful** DPE primitive (silicon-aligned
bit-serial behavior model) and the FC/GEMM workload wrapper that
instantiates it. Covers both architectures we're comparing for the paper:
NL-DPE (analog crossbar + ACAM) and Azure-Lily (DSP-MAC array).

**Verification status (last green run):**
- 8 / 8 IMC sim sanity PASS  (`azurelily/IMC/test.py`)
- 52 / 52 primitive smoke PASS  (`run_dpe_smoke.py`)
- 13 / 13 FC/GEMM workload smoke PASS  (`run_fc_smoke.py`)

Architecture is **double-buffered LOAD substrate** (Task #99) — pass-(k+1)
LOAD writes to substrate B while pass-k COMPUTE reads substrate A. No
LOAD-gate, no WAR hazard, no inter-pass stall.

## Read first

Open [`VERIFICATION_FRAMEWORK_OVERVIEW.html`](VERIFICATION_FRAMEWORK_OVERVIEW.html)
in a browser — self-contained 22-slide deck covering methodology,
primitive RTL architecture, workload mapping, FSM diagrams, cycle
accounting, and verification status. Same content lives in
[`VERIFICATION_FRAMEWORK_OVERVIEW.md`](VERIFICATION_FRAMEWORK_OVERVIEW.md).

Companion reading, in order:

1. [`FIDELITY_METHODOLOGY.md`](FIDELITY_METHODOLOGY.md) — canonical methodology anchor (DPE arch, pipeline model, workload classes, tiling)
2. [`CYCLE_ACCOUNTING.md`](CYCLE_ACCOUNTING.md) — per-test cycle delta evidence (primitive +2 NBA; workload +6/+7/+8 = wrap + tree + clb)
3. [`DPE_NLDPE_FAITHFUL_WALKTHROUGH.md`](DPE_NLDPE_FAITHFUL_WALKTHROUGH.md) — NL-DPE primitive RTL deep-dive (FSM, LOAD corner-turn, COMPUTE, OUTPUT)
4. [`DPE_AZURELILY_FAITHFUL_WALKTHROUGH.md`](DPE_AZURELILY_FAITHFUL_WALKTHROUGH.md) — Azure-Lily primitive RTL deep-dive
5. [`FC_GEMM_WALKTHROUGH.md`](FC_GEMM_WALKTHROUGH.md) — workload mapping (Path A weight-stationary, V×H tiling, fc_top + tb_fc)
6. [`FC_RTL_PLAN.md`](FC_RTL_PLAN.md) — Stage 1A→1D build-out plan

## Run the smoke

### Prereqs
- Python 3.8+
- `iverilog` (Icarus Verilog) — `apt install iverilog` or equivalent
- `numpy` — only used by oracles for regenerating test vectors (pre-generated vectors are checked in)
- Standard POSIX tools (bash, make)

### Primitive smoke — 52 cases, ~3 min
```
cd fc_verification
python3 run_dpe_smoke.py
```
Expected: `52 / 52 PASS, 0 FAIL`. Covers NL faithful (T1–T7 functional +
M-sweep) + AL faithful (T1–T5 + M-sweep) + legacy lazy primitives
(VMM, ACAM, DSP-MAC + their M-sweeps). Per-case log:
`results/dpe_smoke.log`.

### FC/GEMM workload smoke — 13 cases, ~15 min
```
cd fc_verification
python3 run_fc_smoke.py
```
Expected: `13 / 13 PASS`. Per-case row reports `LCYC CCYC OCYC`
per-pass cycles, `sim_exp` (architectural minimum), `rtl_obs`
(measured), and per-stage delta decomposition
`delta = nba + tree + clb + wrap`. Per-case log: `results/fc_smoke.log`.

### IMC sim sanity (Python sim only, no Verilog)
```
python3 ../azurelily/IMC/test.py
```
Expected: `8 / 8 PASS`. Validates the IMC simulator's hand-calc against
the unified `T_fill = LCYC + CCYC + OCYC`, `T_steady = max(LCYC, CCYC,
OCYC)` formula.

### Regenerate test vectors (optional)
```
python3 oracles/nldpe_mac_oracle.py     --gen-test-vectors
python3 oracles/azurelily_mac_oracle.py --gen-test-vectors
```
Writes `.mem` files to `oracles/test_vectors{,_al}/`.

### Regenerate primitives from arch JSON (optional)
```
python3 ../nl_dpe/gen_dpe_nldpe_faithful.py     --config ../azurelily/IMC/configs/nl_dpe.json
python3 ../nl_dpe/gen_dpe_azurelily_faithful.py --config ../azurelily/IMC/configs/azure_lily.json
```

## Layout

```
fc_verification/
  README.md                              ← you are here
  VERIFICATION_FRAMEWORK_OVERVIEW.html   ← start here (slide deck)
  VERIFICATION_FRAMEWORK_OVERVIEW.md     ← same content, MD form
  FIDELITY_METHODOLOGY.md                ← canonical methodology
  CYCLE_ACCOUNTING.md                    ← per-test cycle evidence
  DPE_*_FAITHFUL_WALKTHROUGH.md          ← per-arch primitive walkthroughs
  DPE_PRIMITIVE_WALKTHROUGH.md           ← legacy lazy-primitive walkthrough
  FC_GEMM_WALKTHROUGH.md                 ← workload walkthrough
  FC_RTL_PLAN.md                         ← Stage 1A→1D plan
  Makefile                               ← CLI build harness (R/C/BUF/M knobs)
  run_dpe_smoke.py                       ← primitive smoke (52 cases)
  run_fc_smoke.py                        ← FC/GEMM workload smoke (13 cases)
  rtl/
    dpe_nldpe_faithful.v                 ← faithful NL-DPE primitive (silicon-aligned)
    dpe_azurelily_faithful.v             ← faithful Azure-Lily primitive
    fc_top.v                             ← FC/GEMM workload wrapper (V×H tiles)
    dpe_nldpe.v, dpe_azurelily.v         ← legacy lazy primitives (retained for smoke)
    dsp_mac.v                            ← legacy DSP-MAC primitive
  tb_dpe_nldpe_faithful.v                ← faithful NL primitive TB
  tb_dpe_azurelily_faithful.v            ← faithful AL primitive TB
  tb_fc.v                                ← FC/GEMM workload TB
  tb_dpe_{vmm,acam,dsp_mac}.v            ← lazy primitive TBs (legacy)
  tb_dpe_vmm_msweep.v, tb_dsp_mac_msweep.v ← M-sweep TBs
  oracles/
    nldpe_mac_oracle.py                  ← NL faithful oracle (numpy bit-serial reference)
    azurelily_mac_oracle.py              ← AL faithful oracle
    test_vectors/                        ← pre-generated .mem files for NL TB
    test_vectors_al/                     ← pre-generated .mem files for AL TB

# Repo-level dependencies (siblings of fc_verification/):
../nl_dpe/
  gen_dpe_nldpe_faithful.py              ← regenerates dpe_nldpe_faithful.v from JSON
  gen_dpe_azurelily_faithful.py          ← regenerates dpe_azurelily_faithful.v from JSON
  gen_dpe_stub.py, gen_dsp_mac.py        ← legacy generators
../azurelily/IMC/configs/
  nl_dpe.json, azure_lily.json           ← per-arch parameters consumed by generators
```

## Architecture in one paragraph

The DPE is a bit-serial in-memory-computing tile: R rows × C columns of
weight cells, an analog crossbar (NL) or DSP-MAC array (AL) that fires on
one input-bit slice per cycle. Inputs are loaded into a **double-buffered
slice-major** input substrate: two physically separate substrates A and B
ping-pong, so pass-(k+1) LOAD overwrites substrate B while pass-k COMPUTE
reads substrate A — no LOAD-gate, no WAR hazard. The pipeline cadence is

```
T_fill   = LCYC + CCYC + OCYC                 (first pass)
T_steady = max(LCYC, CCYC, OCYC)              (each subsequent pass)
T(M)     = T_fill + (M − 1) · T_steady
```

For typical configs (NL R=256/C=256/BUF=40/P=8: LCYC=52, CCYC=10, OCYC=52
→ T_fill=114, T_steady=52). The wrapper `fc_top.v` tiles weights across
V × H DPEs (V = K-tile, H = N-tile) and adds a CLB adder tree (depth
⌈log₂V⌉) + activation LUT cycle when needed.

## Cycle-delta budget (RTL vs sim)

| Layer | Delta | Source |
|---|---|---|
| Primitive | +2 (constant) | NBA sub-FSM handoffs (LOAD→COMPUTE wake, COMPUTE→OUTPUT wake) |
| Workload | +6 / +7 / +8 (constant in M) | `wrap (6) + tree (⌈log₂V⌉) + clb (V>1 OR act_no_acam)` |

Deltas are paid **once in T_fill**, never in T_steady. T_steady is
bit-exact between sim and RTL — the double-buffered LOAD removes the
inter-pass stall.

The six contributors to `wrap = 6` are six structural registers in
`fc_top.v` (BRAM-read pipeline, registered DPE handshake, sign-extend
latch, BRAM-write tap, in-BRAM register, done-detect latch). See
`CYCLE_ACCOUNTING.md` §6 for the per-register breakdown.

## Known limits

- **Stage 1D (V > 1 AND H > 1) not yet verified.** Only V=1/H=1, V>1/H=1,
  V=1/H>1 are in the FC smoke. vgg_fc3 (V=16, H=4) is planned next.
- **DIMM and full attention head RTL not yet built.** The faithful
  primitive is verified end-to-end at the FC/GEMM level only.
- **VTR-synth path** is exercised separately (not part of the default
  smoke). Required for area/timing analysis but not for functional
  fidelity verification.

For deeper context, the slide deck and walkthrough docs are the
authoritative source. For per-test cycle traces with full decomposition,
see `CYCLE_ACCOUNTING.md`.
