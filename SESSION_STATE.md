# SESSION_STATE.md — Last updated: 2026-03-19

## Current Phase
**Round 2 full sweep COMPLETE (2026-03-19). 300/300 points × 3 seeds = 900 VTR runs finished on 120×120 grid with P = min(P_dpe, P_clb, P_bram). Three publication figures generated: aggregate scalability heatmap, per-workload heatmaps, and Pareto front (area vs effective latency). Ready for paper writing (T8/T9).**

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

### 40-bit Data Width Migration (2026-03-18)
- Changed DPE data bus from 16-bit to 40-bit across all infrastructure
- Updated: arch XML template (pin counts), gen_gemv_wrappers.py, gen_attention_wrapper.py, hand-written RTL, nl_dpe.json (bram_width)
- Activation LUT saturation: clips to [-128, 127] (int8 precision, independent of bus width)
- All Round 1 + Attention results re-run with 40-bit

### Attention DSE Results (2026-03-18, 40-bit)
- 9/9 attention runs complete (9 configs × 1 workload: N=128, d=128)
- Best config: 512×256 (Fmax=131.6 MHz)
- CLB DIMM/softmax dominates energy (~90%+); DPE projections are minor
- Output: `dse/results/attention_results.csv`

### Round 2 DSE — OLD Results (2026-03-18, OBSOLETE)
- Previous CLB-only sweep (5%, 8%, 12%, 15%) showed flat throughput — CLB was never stressed
- Root cause: single GEMV uses 7-20 CLBs out of 8,500+ available; 106×106 grid massively oversized
- Previous DSP comparison was wrong methodology (standalone benchmark, not equivalent-area replacement)
- Old output files: `round2_results.csv`, `round2_dsp_comparison.csv` (kept for reference)

### Round 1 Extended with R=1024 (2026-03-19)
- Added 3 configs: 1024×64, 1024×128, 1024×256 → total 12 configs × 6 workloads = 72 rows
- R=1024 ranks #2-#4 but doesn't beat 512×128 (larger tile area offsets tiling benefit)
- Confirms R=512 is the sweet spot for FC workloads
- Output: `dse/results/round1_results.csv` (72 rows), updated `top3_configs.json`

### Round 2 DSE — Full Sweep (COMPLETE, 2026-03-19)
- **300/300 points × 3 seeds = 900 VTR runs completed** on 120×120 grid with BRAM cap + multi-seed
- **Fixes applied** (from prior buggy run):
  - Triple resource cap: P = min(P_dpe, P_clb, P_bram)
  - Grid 120×120 passed explicitly (was defaulting to 106×106)
  - Multi-seed: 3 seeds/point, averaged Fmax for stability
- **Scope**: 5 configs × 3 workloads × 20 (d,c) points
- **Output**: `dse/results/round2_full_results.csv` (300 rows, averaged Fmax)
- **Three publication figures**:
  1. Aggregate geomean normalized Fmax heatmaps (5 configs) → `round2_full_scalability.pdf`
  2. Per-workload heatmaps (mean across configs) → `round2_full_per_workload.pdf`
  3. Pareto front: DPE area % vs effective latency (5 configs) → `round2_full_pareto.pdf`
- Directory: `dse/round2_full/{config}/d{d}_c{c}/{workload}/seed{1,2,3}/`

### Cleaned Up Directories (2026-03-19)
- Deleted: `dse/round2/` (995MB, old CLB-only sweep, OBSOLETE)
- Deleted: `dse/round2_full/` (32GB, inconsistent grid sizes + missing BRAM cap)
- Deleted: `dse/sanity_40bit/` (4.4MB, one-time sanity check, results in SESSION_STATE)
- Deleted: `dse/sanity_attention/` (33MB, one-time sanity check, results in SESSION_STATE)

### Generated Artifacts
- `dse/configs/arch/` — 9 auto + 12 fixed + dsp_clb_replace arch XMLs
- `dse/rtl/` — 54 FC wrapper + 9 attention wrapper + GEMM wrapper Verilog files
- `dse/round1/` — VTR outputs for all 72 FC runs (12 configs × 6 workloads)
- `dse/attention/` — VTR outputs for all 9 attention runs
- `dse/round2_full/` — VTR outputs for Round 2 re-run (in progress, 300 points × 3 seeds)
- `dse/results/` — CSVs, JSONs, plots, analysis

## Open Problems / Decisions

| Issue | Status | Decision |
|-------|--------|----------|
| Round 1 ranking metric | **Resolved** | SPEC-style normalized geomean (per-workload best=1.0, geomean across workloads) |
| Fixed vs auto FPGA area | **Resolved** | Auto area for Round 1 (fair per-workload normalization eliminates bias); fixed area for Round 2 |
| Round 2 fixed area size | **Resolved** | 106×106 grid (max top-3 grid = 88 × 1.2 scale) |
| Full Round 1 DSE (FC) | **Complete** | 54/54 runs (6 workloads); top-3: 512x128, 512x64, 512x256 (40-bit) |
| Attention head integration | **Complete** | Option C: fc_128_128 in Round 1 + full attention as separate experiment |
| Attention head generator | **Complete** | `gen_attention_wrapper.py`: parameterized Q/K/V projections + fixed DIMM/softmax CLB |
| Attention IMC runner | **Complete** | `gemv_dse.py --attention` with grouped energy parsing (DPE/CLB/Memory) |
| Attention DSE | **Complete** | 9/9 runs (40-bit); CLB dominates (~90%+); best: 512×256 (131.6 MHz) |
| 40-bit data width | **Complete** | All RTL, arch XML, IMC config migrated from 16-bit to 40-bit DPE bus |
| Round 2 Part 1 (old CLB-only) | **Obsolete** | Flat throughput — CLB never stressed. Replaced by DSP+CLB sweep |
| Round 2 Part 1 (new DSP+CLB) | **COMPLETE** | 300/300 pts × 3 seeds on 120×120. CSV + 3 publication figures generated |
| Round 2 Part 2 (Attention) | **TODO** | Deferred until Part 1 validates methodology |
| Paper figures (Round 2) | **COMPLETE** | 3 figures: scalability heatmap, per-workload heatmap, Pareto front |
| Paper Q1/Q2/Q3 | **UNBLOCKED** | Q1 answered (512×128); Q2 from Round 2 Pareto; Q3 needs ACAM analysis |

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
