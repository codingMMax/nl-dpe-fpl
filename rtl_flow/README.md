# rtl_flow — single entry point for all RTL work

Everything RTL lives here: primitives, the FC/GEMM top, generators, per-arch
specs, testbenches, smoke harnesses, and the methodology docs. If you are
working on RTL, start in this folder and never leave it.

## Layout

| Path | Role |
|------|------|
| `SPEC.md` | **Behavioral charter** — the normative spec of what each primitive must do (cycle cadence, dataflow, formats). Ground truth = this file + the oracles. |
| `specs/*.json` | Per-arch hard-block specs (generator inputs): geometry, LOAD/COMPUTE/OUTPUT cycles, precision, ACAM. Copied from the archived simulator configs; the archived copies remain the *sim-side* reference. |
| `gen/` | Python generators that *produce* RTL from `specs/*.json` — never hand-edit generated files. |
| `rtl/` | Generated + handwritten RTL: `dpe_{nldpe,azurelily}.v` (VTR contract stubs), `dpe_{nldpe,azurelily}_faithful.v` (silicon-truth cycle models), `dsp_mac.v`, `fc_top.v`. |
| `tb/` | Testbenches (`tb_dpe_*.v`, `tb_dsp_mac*.v`, `tb_fc.v`). |
| `smoke/` | Smoke harnesses + independent oracles: `run_dpe_smoke.py` (52 cases), `run_fc_smoke.py` (13 cases), `oracles/`. |
| `docs/` | Methodology: `FIDELITY_METHODOLOGY.md` (canonical anchor), `CYCLE_ACCOUNTING.md`, `FC_RTL_PLAN.md`, primitive/FC walkthroughs. |
| `vtr/` | Synthesis path: `fc_top_synth.v`, `dpe_blackbox.v` (VTR `<model name="dpe">` contract), `run_vtr_smoke.py`. VTR arch XMLs stay in `benchmarks/arch/`. |
| `results/` | Smoke logs (regenerated on each run). |
| `Makefile` | CLI build harness: `make regen`, `make smoke`, per-TB targets. |

## Ground-truth hierarchy (normative)

1. **`SPEC.md` charter** — what the hardware *must* do (dataflow + cycles).
2. **`smoke/oracles/` + NumPy oracles** — what the right *values* are (bit-exact).
3. **`rtl/*_faithful.v`** — the silicon-truth *enforcement*; cycle facts (CCYC)
   emerge structurally from the double-buffered substrate, they are not asserted.
4. `rtl/dpe_*.v` stubs are interface-contract models for VTR only — NOT ground truth.

## Commands

```bash
make -C rtl_flow regen                          # regenerate dpe_*.v, dsp_mac.v
python3 rtl_flow/smoke/run_dpe_smoke.py         # 52 primitive cases          (~1 min)
python3 rtl_flow/smoke/run_fc_smoke.py          # 13 FC cases, stages 1A+1B+1C (~13 min; iverilog ~50 s/case)
python3 rtl_flow/smoke/run_fc_smoke.py --stage 1A
python3 rtl_flow/vtr/run_vtr_smoke.py           # 3 VTR cases (needs VTR_ROOT)
```

## Re-verification status (bottom-up, started 2026-08-29)

Ladder: primitives → fc_top → softmax → projections + DIMM → (then) mapping +
simulator. Each rung = charter you approve + independent oracle + RTL that
matches both. Current ladder position: **Stage 1.1 (primitive charter)**.
Progress is tracked in the session TODO list and `CLAUDE.md`.
