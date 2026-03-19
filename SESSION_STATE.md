# SESSION_STATE.md — Last updated: 2026-03-18

## Current Phase
**Round 1 DSE: Complete (54/54, 9 configs × 6 workloads). Next: attention head generator + Round 2.**

## Completed Work

### Infrastructure (all done)
- [x] `nl_dpe/gen_arch_xml.py` — auto mode: patches DPE tile W/H/area in XML template
- [x] `nl_dpe/gen_gemv_wrappers.py` — FC mode: generates V×H DPE tiling with adder tree + activation_lut
- [x] `azurelily/models/fc.py` — FC+activation model for IMC simulator (V=1 → ACAM free; V>1 → CLB)
- [x] `azurelily/IMC/test.py` — updated with fc_model import and `"fc"` registration
- [x] `gemv_dse.py` — full Round 1 orchestrator with SPEC-style normalized geomean ranking

### Key Bug Fixes Applied
- VTR path bug: `args.dse_dir = args.dse_dir.resolve()` in `main()` (relative path → VTR CWD issue)
- MSB_SA_Ready multi-driver: per-DPE `MSB_SA_Ready_c{col}_r{row}` wires, AND-aggregated
- Adder tree wire collision (H>1): `wire_prefix=f"col{col}"` in `_gen_adder_tree()`

### Sanity Check Results (256×256, 3 workloads)
| Workload | Fmax | Grid | Area (mm²) | Latency (ns) | Energy (pJ) | Tput/mm² | Tput/J |
|----------|------|------|------------|--------------|-------------|----------|--------|
| fc_64_64 | 334.5 MHz | 15×15 | 0.504 | 201.3 | 227.5 | 9.86M | 4.40B |
| fc_512_128 | 307.3 MHz | 26×26 | 1.514 | 1080.7 | 572.6 | 611K | 1.75B |
| fc_512_512 | 293.7 MHz | 28×28 | 1.755 | 1865.9 | 1321.1 | 305K | 757M |

### Round 1 Results (2026-03-18)
- 54/54 FC workload runs complete (9 configs × 6 workloads incl. fc_128_128)
- Top-3: 512×128 (GM=0.852), 512×256 (GM=0.635), 512×64 (GM=0.632)
- All top-3 are R=512 (ACAM-eligible on 5/6 workloads)
- Plots: `round1_ranking.pdf`, `round1_heatmap.pdf`
- Analysis: `dse/results/round1_analysis.md`

### Generated Artifacts
- `dse/configs/arch/` — 9 arch XMLs (128×64 through 512×256)
- `dse/rtl/` — 54 FC wrapper Verilog files (9 configs × 6 workloads)
- `dse/round1/` — VTR outputs for all 54 runs
- `dse/results/` — CSVs, JSONs, plots, analysis

## Open Problems / Decisions

| Issue | Status | Decision |
|-------|--------|----------|
| Round 1 ranking metric | **Resolved** | SPEC-style normalized geomean (per-workload best=1.0, geomean across workloads) |
| Fixed vs auto FPGA area | **Resolved** | Auto area for Round 1 (fair per-workload normalization eliminates bias); fixed area for Round 2 |
| Round 2 fixed area size | Pending | Derive analytically: max VTR grid across all configs × 2 (4× area buffer) |
| Full Round 1 DSE (FC) | **Complete** | 54/54 runs (6 workloads); top-3: 512x128, 512x256, 512x64 |
| Attention head integration | **Planned** | Option C: fc_128_128 in Round 1 + full attention as separate experiment |
| Attention head generator | Not started | `gen_attention_wrapper.py`: parameterized Q/K/V projections + fixed DIMM/softmax CLB |
| Attention IMC runner | Not started | Wire `--model attention` + `run_transformer_model` into `gemv_dse.py` |
| Round 2 implementation | Not started | CLB replacement + DSP/BRAM equivalence (gen_arch_xml.py fixed modes exist) |
| Paper Q1/Q2/Q3 | Blocked on DSE data | Q1 answered (512×128); Q2/Q3 need Round 2 + attention results |

## Design Decisions (Rationale)

- **ACAM activation**: V=1 → ACAM absorbs nonlinear activation at no CLB cost; V>1 → activation in CLB (energy penalty tracked separately)
- **Normalized geomean**: avoids cross-workload area bias (small workloads have tiny grids → huge tput/mm² numerically); every workload contributes equally
- **Auto_layout for Round 1**: VTR sizes FPGA to just-fit the design; allows fair comparison of inherent efficiency before adding fixed area overhead
- **Round 2 fixed layout**: fix FPGA to a worst-case grid, then sweep CLB→DPE replacement ratios to study area-efficiency tradeoff

## NL-DPE vs Azure-Lily Baseline (known data)
| Metric | NL-DPE | Azure-Lily |
|--------|--------|-----------|
| LeNet Fmax | 276.8 MHz | 256.4 MHz |
| LeNet DPEs | 6 | 5 |
| ResNet Fmax | 177.1 MHz | 215.1 MHz |
| ResNet DPEs | 40 | 35 |
| ACAM energy ratio (LeNet) | 0.548× | 1.0× (baseline) |

## Key Parameters (do not guess, verify from area_power.py)
- Azure-Lily e_conv = 2.33 (calibrated ground truth)
- ACAM power = 43.89 mW; remaining budget = 3.89 mW
- CLB_tile = 2239 µm²; SB = 688 µm²; CB = 303 µm²
