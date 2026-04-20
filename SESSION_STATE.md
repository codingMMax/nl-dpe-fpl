# SESSION_STATE.md — Last updated: 2026-04-18

## Current Phase (2026-04-18)

**Phase H–N (W=16 full DIMM verification) CLOSED.** RTL↔simulator
per-stage and end-to-end cycles aligned within the plan's ≤20 cycle
tolerance for NL-DPE at N=128, d=64:

| Stage | RTL | Sim | Δ |
|---|---:|---:|---:|
| score  | 260 | 267 | 7  ✓ |
| softmax|  27 |  26 | 1  ✓ |
| wsum   | 274 | 257 | 17 ✓ |
| **E2E**| **561** | **550** | **11 ✓** |

VTR 3-seed runs complete: NL-DPE DPE=**64 exact** (4 stages × 16 lanes)
at 90.1 MHz; Azure-Lily DSP=68 (≈32 int_sop_4 + packing drift) at
94.0 MHz. Final plot
`fc_verification/results/dimm_architecture_comparison.pdf` regenerated
at VTR-measured Fmax. Details: `fc_verification/VERIFICATION.md`
and rolling log
`fc_verification/results/dimm_top_w16_alignment_log.txt`.

**P4 open phases (multi-pass pipelined DPE model, opened 2026-04-18,
scope reduced 2026-04-19 to Layout A + Regime B only):** the track is
three phases under the committed **Layout A + Regime B** path.

- **Phase 1 — sim Regime B swap:** ✅ COMPLETE (2026-04-19, commit
  `c15797f` parent / `92bbb00` azurelily). `gemm_log` emits
  `T(M) = L_A · M + O`. Unit test + regression guard pass.
- **Phase 2 — FC RTL structural alignment + func + latency + VTR:**
  ✅ COMPLETE (2026-04-19, commits `1678443` harness, `86e539b` VTR
  check). 12/12 FC configs pass: feed/output Δ=0 exact;
  compute Δ=+4 annotated (FSM handshake); reduction+act Δ=+1
  annotated (valid_n latch). Activation routing matches policy
  table (ACAM-absorbed for V=1 ACAM, CLB-LUT otherwise). VTR DPE
  count = V·H exactly for all 12; CLB ∈ [7,39], BRAM ∈ [2,10],
  Fmax ∈ [250, 407] MHz. Block-level comparison figures can be
  regenerated from `block_comp_apr_11/results/block_comparison_results.csv`
  via `plot_block_comparison.py`.
- **Phase 3 — DIMM RTL re-verify (NEXT):** `mac_qk`, `softmax`,
  `mac_sv` against Regime-B sim. Structural verification of the
  attention DIMM pipeline in the RTL (vs what Phase 1 sim models).
  Functional + latency, with same convergence policy as Phase 2
  (residual deltas must be explainable as FSM / modelling
  granularity, not structural).

Retired / archived: Regime C / double-buffering (was T33, archived
as reference only) and the transpose block for Layout B (was T31,
retired) — archived in `paper/methodology/dpe_pipeline_model.md`
§§3.2, 4, 5.4, 5.7 as design-space reference only. Authoritative
reference: `paper/methodology/dpe_pipeline_model.md` §§5.3.1, 5.7,
8.1. See `CLAUDE.md` "Active TODO Tracks" for one-hop access to
model doc, verification baseline, and memory.

---

## Prior phase (2026-03-21, historical)
**All DSE experiments COMPLETE (2026-03-21). Four sweeps done: Round 1 (72 runs), Round 2 FC bare GEMV (900 runs), Round 2 Attention (240 runs), Round 2 FC+BN+Softmax (720 runs). Key findings: 512×128 is the robust balanced config across all workload types; DSP bottleneck confirmed for complete inference pipelines; BRAM wall for attention. Writing materials and plots consolidated.**

## BERT-Tiny Workload (NEW — Main Result, 2026-03-20)

### Architecture (TinyBERT: 2L/2H/128d)
| Parameter | Value |
|-----------|-------|
| Layers | 2 |
| Hidden dim (d_model) | 128 |
| Attention heads | 2 |
| Head dim (d_head) | 64 |
| FFN intermediate (d_ff) | 512 |
| Activation | GELU (modeled same as existing: ACAM if V=1, CLB if V>1) |
| Default seq_len | 1024 |

### IMC Simulator Model (COMPLETE)
- [x] `azurelily/models/bert_tiny.py` — full BERT-Tiny model definition
- [x] `azurelily/nn/layernorm_layer.py` — LayerNorm layer class
- [x] `azurelily/nn/embedding_layer.py` — Embedding layer class
- [x] `azurelily/IMC/peripherals/fpga_fabric.py` — `layernorm()` and `embedding_lookup()` methods
- [x] `azurelily/IMC/scheduler_stats/scheduler.py` — dispatch for `layernorm` and `embedding` types
- [x] `azurelily/IMC/test.py` — `bert_tiny` registered, `run_bert_model()` traversal with multi-head parallel
- [x] Multi-head strategy: run head layers num_heads times, subtract (num_heads-1)×latency for parallel execution

### IMC Simulator Sanity Check Results (256×256 DPE, default config)
| seq_len | Total Energy (pJ) | Latency (ns) | DPE % | FPGA % | Mem % |
|---------|--------------------|--------------|-------|--------|-------|
| 128 | 14.0M | 9.97M | 72.3% | 25.8% | 1.9% |
| 256 | 54.9M | 39.4M | 72.6% | 25.5% | 1.9% |
| 1024 | 868.7M | 625.6M | 72.9% | 25.2% | 1.9% |

- Scaling: ~4× per 2× seq_len (quadratic from attention) — correct
- ACAM: All projections V=1 on 256×256, no fpga_activation energy — correct
- Dimensions verified: Q/K/V (S,128,128), QK^T (S,64,S), FFN1 (S,128,512), FFN2 (S,512,128)

### RTL Implementation Status
**DPE-independent modules** (can implement now — pure CLB):
- [ ] LayerNorm RTL — CLB reduction trees + element-wise ops, 128-wide
- [ ] Residual Add RTL — CLB element-wise add, 128-wide
- [ ] Embedding Add RTL — CLB 3-way element-wise add, 128-wide

**DPE-dependent modules** (blocked on DPE config selection):
- [ ] Q/K/V/O projections — V=ceil(128/R), H=ceil(128/C) DPE tiling
- [ ] FFN1 — V=ceil(128/R), H=ceil(512/C) DPE tiling + GELU activation
- [ ] FFN2 — V=ceil(512/R), H=ceil(128/C) DPE tiling
- [ ] Attention DIMM (QK^T, Score×V) — uses DPE(I|exp) and DPE(I|log) via ACAM

### RTL Style
- Use plain `+` and `*` Verilog operators (VTR infers DSPs from arch XML)
- No explicit DSP instantiation — consistent with all existing RTL

## Completed Work (Prior DSE)

### Infrastructure (all done)
- [x] `nl_dpe/gen_arch_xml.py` — auto mode: patches DPE tile W/H/area in XML template
- [x] `nl_dpe/gen_gemv_wrappers.py` — FC mode: generates V×H DPE tiling with adder tree + activation_lut
- [x] `azurelily/models/fc.py` — FC+activation model for IMC simulator (V=1 → ACAM free; V>1 → CLB)
- [x] `azurelily/IMC/test.py` — updated with fc_model, attention, and bert_tiny registration
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

### Round 2 Attention DSE (COMPLETE, 2026-03-20)
- **80/80 points × 3 seeds = 240 VTR runs** on 120×120 grid
- **Architecture**: (3V+4)×H DPEs per replica (3 projections + 4 DIMM stages)
- **DIMM mapping (Fig 6c)**: DPE(I|exp/log) for nonlinear ops, CLB for add/reduce
- **4-resource constraint**: P = min(P_dpe, P_clb, P_bram, P_dsp)
- **Per-replica resources** (empirical VTR calibration): 145 CLBs, 64 BRAMs, 2 DSPs
- **BRAM wall**: P capped at 7 (64 BRAMs/rep, 472 total) — hard ceiling
- **d=100% excluded**: softmax normalization needs DSP blocks
- **Output**: `dse/results/plots/round2_attention/round2_attention_results.csv` (80 rows)
- **Plots**: scalability heatmap, per-config Pareto, merged Pareto, throughput ceiling

### IMC Simulator Fixes (2026-03-20)
- **mac_sv DPE(I|log)**: Added missing log-domain conversion for attention weights before log-domain matmul
- **softmax DPE(I|exp)**: Correctly uses DPE for exp (was CLB LUT)
- **softmax norm DPE(I|log)**: Replaces CLB inverse with DPE log
- **DPE counting**: Updated to (3V+4)×H formula (was 3VH, missed DIMM DPEs)

### Additional Publication Figures (2026-03-20)
- FC merged Pareto: `round2_full_pareto_merged.pdf` (cross-config, NL-DPE + AL-like groups)
- FC throughput ceiling: `round2_full_ceiling.pdf` (soft ceiling from routing degradation)
- Attention merged Pareto: `round2_attention_pareto_merged.pdf`
- Attention throughput ceiling: `round2_attention_ceiling.pdf` (BRAM wall)
- All plots organized into `dse/results/plots/{round1,round2_fc,round2_attention}/`

### Round 2 FC+BN+Softmax DSE (COMPLETE, 2026-03-21)
- **234/240 points × 3 seeds = 720 VTR runs** (6 failures at 1024×64 c=60%)
- **Benchmark**: Complete FC layer = GEMV (DPE) + BatchNorm (4 DSP MACs) + Softmax (12 DSP MACs + CLB)
- **16 mac_int_9x9 per replica** = 4 DSP tiles/rep, 93 CLBs/rep, 16 BRAMs/rep
- **4-resource constraint**: P = min(P_dpe, P_clb, P_bram, P_dsp)
- **DSP crossover confirmed**: fc_512_128 peaks at d=40% (3.66 inf/ns), drops 50% by d=80% (1.88 inf/ns)
- **Binding distribution**: DPE=43%, DSP=42%, BRAM=15%, CLB=0%
- **Merged Pareto balanced**: NL-DPE group → 512×128 at d=40% c=0% (differs from bare GEMV: 1024×128)
- **Key insight**: Non-DPE resource constraints (DSP, BRAM) shift the optimal config to 512×128
  - Bare GEMV (DPE-only) → 1024×128 wins (fewer DPEs/rep)
  - FC+BN+Softmax (DSP-constrained) → 512×128 wins (smaller tile, less waste under DSP cap)
  - Attention (BRAM-constrained) → 512×128 wins (same reason)
- **Output**: `dse/results/plots/round2_fc_softmax/round2_fc_softmax_results.csv` (234 rows)
- **Plots**: per-config Pareto, merged Pareto, DSP sweep curves, binding heatmap, comparison overlay

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
| Round 2 Part 2 (Attention) | **COMPLETE** | 80/80 pts × 3 seeds. BRAM wall at P=7. 4-resource constraint. |
| Round 2 Part 3 (FC+BN+Softmax) | **COMPLETE** | 234/240 pts. DSP bottleneck at d≥60%. Balanced: 512×128 (not 1024×128). |
| Paper figures (Round 2) | **COMPLETE** | FC: 5 figs, Attention: 4 figs, FC+Softmax: 6 figs. All in dse/results/plots/ |
| Paper Q1/Q2/Q3 | **UNBLOCKED** | Q1: 512×128; Q2: balanced config shifts with workload resource profile; Q3: ACAM analysis pending |
| Writing materials | **COMPLETE** | paper/dse_writing_materials.md consolidates all DSE details |
| Config robustness | **RESOLVED** | 512×128 is balanced for DSP/BRAM-constrained workloads; 1024×128 only wins for DPE-only |

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
