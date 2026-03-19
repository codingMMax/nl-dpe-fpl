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
- [ ] **T3a**: Build `nl_dpe/gen_attention_wrapper.py`
  - Parameterized by (R, C): generates Q/K/V projection DPEs via `gen_fc_wrapper` tiling
  - Fixed DIMM/softmax/weighted-sum CLB modules extracted from `attention_head_1_channel.v`
  - Stitch together: 3 × FC projection (parameterized) + DIMM score + softmax + DIMM weighted sum (fixed)
- [ ] **T3b**: Add attention IMC runner to `gemv_dse.py`
  - Call `--model attention` with `run_transformer_model` path (already in `azurelily/IMC/test.py`)
  - Parse attention-specific energy breakdown (linear_Q/K/V, mac_qk, exp, norm, mac_sv)
- [ ] **T3c**: Run attention DSE: 9 configs × 1 workload (N=128, d=128)
  - Separate results CSV: `dse/results/attention_results.csv`

### P2 — Round 2 DSE (after P0 + P1)
- [ ] **T4**: Derive fixed grid size analytically
  - Formula: max(grid_W, grid_H) across all Round 1 runs × 2 → fixed_grid_W = fixed_grid_H

- [ ] **T5**: Implement Round 2 in `gemv_dse.py`
  - Mode A (CLB replace): `gen_arch_xml(mode="fixed_clb_replace", ...)` — 4 CLB ratios × top-3 configs × 6 workloads
  - Mode B (DSP/BRAM): `gen_arch_xml(mode="fixed_dsp_bram", ...)` — 4 DSP/BRAM pairs × top-3 configs × 6 workloads
  - `--round 2` flag, reads `top3_configs.json` for config list

- [ ] **T6**: Run Round 2: `python gemv_dse.py --round 2`
  - 2 × 4 × top-3 × 6 = 144 VTR runs
  - Output: `dse/results/round2_results.csv`

### P3 — Paper Results
- [ ] **T7**: Generate paper figures
  - Fig 3c: Normalized geomean ranking (Round 1 summary) — DONE: `round1_ranking.pdf`
  - Fig 3d: Config × Workload heatmap — DONE: `round1_heatmap.pdf`
  - Fig 4 (Round 2): CLB replacement ratio vs throughput at fixed area
  - Fig 5 (Round 2): DSP/BRAM equivalence comparison
  - Fig 6: Attention head energy breakdown (DPE projections vs CLB DIMM/softmax)

- [ ] **T8**: Fill paper Q1/Q2/Q3 with actual data
  - Q1: Which (R,C) config is best? → from Round 1 ranking (512×128)
  - Q2: Optimal DPE density? → from Round 2 CLB replacement sweep
  - Q3: ACAM value? → compare V=1 vs V>1 energy across FC workloads + attention ACAM-as-log analysis

- [ ] **T9**: Write paper sections (after T7/T8)
  - Section 5 (Results): Q1, Q2, Q3 subsections
  - Section 6 (Discussion): ACAM dual-mode value (activation + log), limitations

## Backlog / Deferred
- [ ] Delete old GEMV wrapper files in `nl_dpe/to_delete/` (staged for removal)
- [ ] Delete `dse_implementation_plan.py` (permission issue — user to do manually)
- [ ] Azure-Lily Round 2 equivalent (for fair comparison baseline)

## Completed
- [x] fc.py (azurelily/models/) — FC+activation IMC model
- [x] test.py (azurelily/IMC/) — fc_model registered
- [x] gen_arch_xml.py — all 3 modes (auto, fixed_clb_replace, fixed_dsp_bram)
- [x] gen_gemv_wrappers.py — FC mode with V×H tiling, adder tree, activation_lut
- [x] gemv_dse.py — Round 1 orchestrator with normalized geomean ranking + skip-existing reload fix
- [x] 3-point sanity check — 256×256 × {fc_64_64, fc_512_128, fc_512_512} passed
- [x] 9 arch XMLs generated in dse/configs/arch/
- [x] 45 FC wrapper Verilog files generated in dse/rtl/
- [x] Round 1 full sweep — 45/45 runs, top-3: 512x128, 512x256, 512x64
- [x] Round 1 plots — `round1_ranking.pdf`, `round1_heatmap.pdf`
- [x] Round 1 analysis — `dse/results/round1_analysis.md`
- [x] nl_dpe/ cleanup — old files staged in `nl_dpe/to_delete/`, .gitignore updated
- [x] Attention head integration decision — Option C: fc_128_128 in Round 1, full attention head as separate experiment
