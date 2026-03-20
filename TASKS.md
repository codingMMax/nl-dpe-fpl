# TASKS.md — NL-DPE DSE Research Pipeline

## Active Sprint

### P0 — Round 1 DSE
- [x] **T1**: Run full Round 1 sweep: `python gemv_dse.py --round 1`
  - 54/54 runs completed (9 configs × 6 workloads). Bug fix applied: skip-existing now reloads pre-existing results into CSV.
  - Output: `dse/results/round1_results.csv`, `top3_configs.json`

- [x] **T2**: Verify Round 1 results
  - 54/54 rows present, no bad rows, ACAM-eligible V=1 configs dominate 5/6 workloads
  - Top-3: 512x128 (0.852), 512x256 (0.635), 512x64 (0.632)
  - Exception: fc_2048_256 (V=4 for all R≥512) — large GEMV forces multi-tile, no ACAM benefit

- [x] **T2b**: Add `fc_128_128` (attention projection proxy) to Round 1
  - 9/9 runs completed. All configs V=1 (ACAM-eligible). Differentiation via tile area only.
  - Ranking unchanged: top-3 still 512x128, 512x256, 512x64. Plots and analysis updated.

### P1 — Attention Head Integration
- [x] **T3a**: Build `nl_dpe/gen_attention_wrapper.py`
  - Parameterized by (R, C): generates Q/K/V projection DPEs via `gen_fc_wrapper` tiling
  - Fixed DIMM/softmax/weighted-sum CLB modules extracted from `attention_head_1_channel.v`
  - VTR sanity: 256x128 (H=1, 3 DPEs) and 256x64 (H=2, 6 DPEs) both pass
- [x] **T3b**: Add attention IMC runner to `gemv_dse.py`
  - `run_imc_attention()` + `run_attention_dse()` with `--attention` flag
  - Parses grouped energy breakdown (DPE / FPGA CLB / Memory)
- [x] **T3c**: Run attention DSE: 9 configs × 1 workload (N=128, d=128)
  - 9/9 runs complete. Output: `dse/results/attention_results.csv`
  - Best: 512x256 (Fmax=131.6 MHz, 40-bit width). CLB DIMM/softmax dominates energy (~90%+)
  - DPE projections are minor fraction of total energy
  - Plots: `attention_ranking.pdf`, `attention_energy_breakdown.pdf`

### P2 — Round 2 DSE (after P0 + P1)
- [x] **T4**: Derive fixed grid size
  - Fixed grid: 106×106 (from max top-3 grid = 88 × 1.2 scale)
  - Saved to `dse/results/round2_template.json`

- [x] **T5-old**: ~~Old Round 2 CLB-only sweep~~ (OBSOLETE)
  - 77/84 runs complete but results are flat — CLB never stressed (7-20 CLBs used out of 8,500+)
  - Root cause: single GEMV doesn't create enough CLB demand
  - Old output: `dse/results/round2_results.csv` (kept for reference)

- [x] **T5-new**: Round 2 Part 1 — DSP+CLB Replacement Sweep (§11.4 of experiment plan) (COMPLETE)
  - **Methodology**: sweep DSP→DPE and CLB→DPE replacement simultaneously with GEMM data parallelism
  - DSP replacement adds compute (P replicas), CLB replacement degrades routing → Fmax drops
  - 5 DSP ratios {20%, 40%, 60%, 80%, 100%} × 4 CLB ratios {0%, 20%, 40%, 60%} = 20 points per (config, workload)
  - 5 configs (512×128, 1024×128, 1024×64, 1024×256, 512×256) × 3 workloads = 300 points
  - Multi-seed: 3 VTR seeds per point, averaged Fmax for stability → 900 total VTR runs
  - Fixed grid: 120×120 (fits all 5 configs including 1024×256 tile 7×9)
  - Sole metric: utilization (fn/f0)
  - **Sub-tasks:**
    - [x] **T5a**: Analytical feasibility table for fc_2048_256 on 512×128 (all 25 points feasible)
    - [x] **T5b**: gen_arch_xml.py — new `fixed_dsp_clb_replace` mode for simultaneous DSP+CLB replacement
    - [x] **T5c**: gen_gemm_wrapper.py — P-replica wrapper around existing GEMV RTL
    - [x] **T5d**: Prototype VTR runs — fc_2048_256 on 512×128, 25/25 points completed (106×106 grid)
    - [x] **T5e**: Plot throughput utilization (5 lines × 5 points) → `round2_throughput_utilization.pdf`
    - [x] **T5f-old**: First full sweep — 273/300 rows, had bugs (missing BRAM cap, wrong grid 106→120)
    - [x] **T5g**: Fix BRAM feasibility — added `count_available_brams()`, `estimate_brams_per_replica()`, triple resource cap P=min(P_dpe,P_clb,P_bram)
    - [x] **T5h**: Add multi-seed support — 3 seeds/point in seed{1,2,3}/ subdirs, averaged Fmax, updated skip-existing
    - [x] **T5f-rerun**: Clean re-run of all 300 points on 120×120 grid with BRAM cap + multi-seed — 300/300 COMPLETE
    - [x] **T5i**: Publication figures — `round2_full_scalability.pdf` (aggregate heatmap), `round2_full_per_workload.pdf` (per-workload heatmap), `round2_full_pareto.pdf` (Pareto front: DPE area % vs effective latency)
  - **Bug fixes applied (2026-03-19)**:
    - `compute_feasibility()` now checks 3 resource limits: DPE, CLB, BRAM (was only DPE+CLB)
    - Grid size passed explicitly (was defaulting to 106×106 instead of 120×120)
    - BRAM per replica: 4 (K≤1024), 6 (K>1024). 120×120 grid has 472 BRAMs
    - All 300/300 points verified feasible in dry-run
  - **Deleted old data**: `dse/round2_full/` (32GB), `dse/round2/` (995MB), `dse/sanity_40bit/`, `dse/sanity_attention/`

- [x] **T6-new**: Round 2 Part 2 — Attention Head Exploration (COMPLETE, 2026-03-20)
  - 80/80 points × 3 seeds = 240 VTR runs on 120×120 grid
  - Architecture: (3V+4)×H DPEs/rep, DIMM stages use DPE(I|exp/log)
  - 4-resource constraint: P = min(P_dpe, P_clb, P_bram, P_dsp)
  - Key finding: BRAM wall at P=7 (64 BRAMs/rep, 472 total)
  - Output: round2_attention_results.csv (80 rows), 4 publication figures
  - Plots: scalability heatmap, per-config Pareto, merged Pareto, throughput ceiling

### P3 — Paper Results
- [x] **T7-old**: ~~Old paper figures~~ (Round 2 plots OBSOLETE, Round 1 + Attention plots still valid)
  - Valid: `round1_ranking.pdf`, `round1_heatmap.pdf`, `attention_ranking.pdf`, `attention_energy_breakdown.pdf`
  - Obsolete: `round2_clb_tput_mm2.pdf`, `round2_clb_tput_J.pdf`, `round2_geomean.pdf`, `round2_dsp_comparison.pdf`

- [x] **T7-new**: Round 2 paper figures COMPLETE
  - `round2_full_scalability.pdf` — aggregate geomean normalized Fmax heatmaps (5 configs, RdYlGn)
  - `round2_full_per_workload.pdf` — per-workload heatmaps mean across configs (PuBuGn)
  - `round2_full_pareto.pdf` — Pareto front: DPE area % vs effective latency, 5 configs, knee points annotated
  - Plot script: `dse/results/plot_round2_full.py`

- [ ] **T8**: Fill paper Q1/Q2/Q3 with actual data (UNBLOCKED)
  - Q1: Which (R,C) config is best? → from Round 1 ranking (512×128)
  - Q2: Optimal DPE density? → from new Round 2 throughput utilization curve
  - Q3: ACAM value? → compare V=1 vs V>1 energy across FC workloads + attention ACAM-as-log analysis

- [ ] **T9**: Write paper sections (after T7/T8)
  - Section 5 (Results): Q1, Q2, Q3 subsections
  - Section 6 (Discussion): ACAM dual-mode value (activation + log), limitations

## Backlog / Deferred
- [ ] Delete old GEMV wrapper files in `nl_dpe/to_delete/` (staged for removal)
- [ ] Delete `dse_implementation_plan.py` (permission issue — user to do manually)
- [ ] Delete `dse/round2_proto/` (1.9GB, superseded by round2_full)
- [ ] Azure-Lily Round 2 equivalent (for fair comparison baseline)

## Completed
- [x] fc.py (azurelily/models/) — FC+activation IMC model
- [x] test.py (azurelily/IMC/) — fc_model registered
- [x] gen_arch_xml.py — all 4 modes (auto, fixed_clb_replace, fixed_dsp_bram, fixed_dsp_clb_replace)
- [x] gen_gemv_wrappers.py — FC mode with V×H tiling, adder tree, activation_lut
- [x] gemv_dse.py — Round 1 orchestrator with normalized geomean ranking + skip-existing reload fix
- [x] 3-point sanity check — 256×256 × {fc_64_64, fc_512_128, fc_512_512} passed
- [x] 9 arch XMLs generated in dse/configs/arch/
- [x] 45 FC wrapper Verilog files generated in dse/rtl/
- [x] Round 1 full sweep — 54/54 runs (incl. fc_128_128), top-3: 512x128, 512x256, 512x64
- [x] Round 1 plots — `round1_ranking.pdf`, `round1_heatmap.pdf`
- [x] Round 1 analysis — `dse/results/round1_analysis.md`
- [x] nl_dpe/ cleanup — old files staged in `nl_dpe/to_delete/`, .gitignore updated
- [x] Attention head integration decision — Option C: fc_128_128 in Round 1, full attention head as separate experiment
- [x] gen_attention_wrapper.py — parameterized Q/K/V projections + fixed CLB modules (DIMM/softmax/weighted-sum)
- [x] gemv_dse.py `--attention` mode — attention IMC runner with grouped energy breakdown parsing
- [x] Attention DSE — 9/9 runs, best: 512×256 (131.6 MHz, 40-bit), CLB dominates ~90%+ energy
- [x] Attention plots — `attention_ranking.pdf`, `attention_energy_breakdown.pdf`
- [x] 40-bit data width migration — all RTL, arch XML, IMC config updated from 16-bit to 40-bit DPE bus
- [x] Round 1 + Attention DSE re-run with 40-bit — top-3 unchanged: 512x128, 512x64, 512x256
- [x] Fixed grid derivation — 106×106 from max top-3 grid (88) × 1.2
- [x] Round 2 implementation — CLB replacement sweep in gemv_dse.py (Part 1)
- [x] gen_arch_xml.py fix — FULL_LAYOUT_RE replaces entire `<layout>` section in fixed modes
- [x] Round 2 full sweep — 77/84 runs (512x256 at 5% placement failure expected)
- [x] Round 2 plots — `round2_clb_tput_mm2.pdf`, `round2_clb_tput_J.pdf`, `round2_geomean.pdf`
- [x] gen_dsp_gemv_wrapper.py — DSP-based FC RTL generator with explicit VTR single_port_ram primitives
- [x] gen_dsp_gemv_wrapper.py — DSP-based FC RTL generator, primary-input datapath passes VTR synthesis (8 dsp_top + 32-256 memory)
- [x] Round 2 Part 2 plot — `round2_dsp_comparison.pdf`, `round2_dsp_comparison.csv`
- [x] gen_arch_xml.py `fixed_dsp_clb_replace` mode — simultaneous DSP+CLB replacement with priority-based wc placement
- [x] gen_gemm_wrapper.py — P-replica GEMM wrapper (round-robin input distribution + output collection)
- [x] gemv_dse.py `--round2-proto` — grid-based feasibility + 25-point prototype orchestrator
- [x] Grid-based DPE counting — `count_available_wc()` simulates VTR column priority placement (replaced faulty area-based model)
- [x] Round 2 prototype sweep — 25/25 points (fc_2048_256, 512×128, 106×106 grid)
- [x] Round 2 throughput utilization plot — `round2_throughput_utilization.pdf`
