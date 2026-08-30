# NL-DPE FPGA — Hard-Block Research Repo

NL-DPE FPGA hard block research: **crossbar-size DSE** (complete) + **RTL/sim fidelity alignment** (in progress) + **safe-softmax study** (complete), for a paper comparing NL-DPE vs Azure-Lily.

## Repo Map

| Path | Role |
|------|------|
| `fc_verification/` | RTL behavior models, testbenches, smoke harnesses, methodology docs (**live work**) |
| `softmax_study/` | Safe-softmax RTL + VTR + energy study, AL vs NL-DPE (**complete**) |
| `nl_dpe/` | DPE physical specs (`area_power.py`), VTR arch XML / stub generators, VTR runner |
| `dse/` | DSE results (CSVs, JSONs, plots) + Round-1 VTR outputs |
| `benchmarks/` | BERT-Tiny / CNN benchmark infrastructure (DSE-era) |
| `paper/` | Paper methodology, figures, scripts, writing materials |
| `archive/` | Reference-only: `azurelily_simulator/` (de-submoduled), legacy worktree-era tracks |
| `gemv_dse.py`, `flexscore_dse.py` | DSE orchestrators (root, DSE-era) |

## Current State

- **RTL/sim alignment** (canonical anchor: `fc_verification/FIDELITY_METHODOLOGY.md`):
  - DPE primitives: NL-DPE (`dpe_nldpe.v`), Azure-Lily (`dpe_azurelily.v`), DSP-MAC (`dsp_mac.v`), faithful variants (`*_faithful.v`)
  - FC/GEMM top `fc_top.v` — Path A weight-stationary V×H array; unified cycle formula `T(M) = T_fill + (M−1)·T_steady`
  - Stages 1A (V=1,H=1), 1B (V>1), 1C (H>1) validated; **Stage 1D (general V×H) in flight**
  - Regression guards: `run_dpe_smoke.py` (52 cases), `run_fc_smoke.py` (13 cases), `run_vtr_smoke.py` (3 cases, needs VTR_ROOT)
- **Safe-softmax study**: complete; NL wins energy 3.1–3.9×/element vs AL at supply-matched port width. See `softmax_study/SOFTMAX_STUDY.md`.
- **DSE**: complete (Round 1: 12 crossbar configs, 512×128 optimal; Round 2: fixed-area density sweeps, DSP/BRAM constraints shift optimum to 512×128). Results: `dse/results/`.

## Quick Commands

```bash
# RTL/sim smoke suites (from fc_verification/)
python3 run_dpe_smoke.py      # 52 primitive cases
python3 run_fc_smoke.py       # 13 FC cases (--stage 1A/1B/1C/1D)
python3 run_vtr_smoke.py      # 3 VTR cases (requires VTR_ROOT)

# Softmax study (complete; rerun from softmax_study/)
python3 run_softmax_smoke.py
python3 run_vtr_softmax.py    # requires VTR_ROOT
```

## Notes

- Everything under `archive/` is **reference only** — do not edit as live work.
- DSE raw VTR run directories were deleted to save space; only summary CSVs/plots are kept (`dse/results/`).
- Session protocol and in-flight work are tracked in `CLAUDE.md` ("Active TODO Track").
