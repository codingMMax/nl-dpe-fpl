# NL-DPE FPGA — Hard-Block Research Repo

NL-DPE FPGA hard block research: **crossbar-size DSE** (complete) + **RTL/sim fidelity alignment** (in progress) + **safe-softmax study** (complete), for a paper comparing NL-DPE vs Azure-Lily.

## Repo Map

| Path | Role |
|------|------|
| `rtl_flow/` | **All RTL work lives here**: primitives, `fc_top`, generators, specs, TBs, smoke harnesses, methodology docs (**live work**) |
| `softmax_study/` | Safe-softmax RTL + VTR + energy study, AL vs NL-DPE (**complete**) |
| `nl_dpe/` | DSE-era VTR arch XML / workload-wrapper generators, VTR runner, `area_power.py` |
| `dse/` | DSE results (CSVs, JSONs, plots) + Round-1 VTR outputs |
| `benchmarks/` | BERT-Tiny / CNN benchmark infrastructure (DSE-era) |
| `paper/` | Paper methodology, figures, scripts, writing materials |
| `archive/` | Reference-only: `azurelily_simulator/` (de-submoduled), legacy worktree-era tracks |
| `gemv_dse.py`, `flexscore_dse.py` | DSE orchestrators (root, DSE-era) |

## Current State

- **RTL flow** (single entry point: `rtl_flow/`; canonical anchor: `rtl_flow/docs/FIDELITY_METHODOLOGY.md`):
  - DPE primitives: NL-DPE (`dpe_nldpe.v`), Azure-Lily (`dpe_azurelily.v`), DSP-MAC (`dsp_mac.v`), faithful variants (`*_faithful.v`) — in `rtl_flow/rtl/`
  - FC/GEMM top `fc_top.v` — Path A weight-stationary V×H array; unified cycle formula `T(M) = T_fill + (M−1)·T_steady`
  - Stages 1A (V=1,H=1), 1B (V>1), 1C (H>1) validated; **Stage 1D (general V×H) pending**
  - Regression guards: `rtl_flow/smoke/run_dpe_smoke.py` (52 cases), `rtl_flow/smoke/run_fc_smoke.py` (13 cases), `rtl_flow/vtr/run_vtr_smoke.py` (3 cases, needs VTR_ROOT)
  - **Primitive re-verification in flight**: behavioral charter (`rtl_flow/SPEC.md`), independent NumPy oracles, re-derived expectations — see `rtl_flow/README.md`
- **Safe-softmax study**: complete; NL wins energy 3.1–3.9×/element vs AL at supply-matched port width. See `softmax_study/SOFTMAX_STUDY.md`.
- **DSE**: complete (Round 1: 12 crossbar configs, 512×128 optimal; Round 2: fixed-area density sweeps, DSP/BRAM constraints shift optimum to 512×128). Results: `dse/results/`.

## Quick Commands

```bash
# RTL flow (everything under rtl_flow/)
make -C rtl_flow regen          # regenerate dpe_*.v + dsp_mac.v from rtl_flow/specs/*.json
python3 rtl_flow/smoke/run_dpe_smoke.py   # 52 primitive cases
python3 rtl_flow/smoke/run_fc_smoke.py    # 13 FC cases (--stage 1A/1B/1C/1D)
python3 rtl_flow/vtr/run_vtr_smoke.py     # 3 VTR cases (requires VTR_ROOT)

# Softmax study (complete; rerun from softmax_study/)
python3 run_softmax_smoke.py
python3 run_vtr_softmax.py    # requires VTR_ROOT
```

## Notes

- Everything under `archive/` is **reference only** — do not edit as live work.
- DSE raw VTR run directories were deleted to save space; only summary CSVs/plots are kept (`dse/results/`).
- Session protocol and in-flight work are tracked in `CLAUDE.md` ("Active TODO Track").
