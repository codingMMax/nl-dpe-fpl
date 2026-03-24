# DPE Design Space Exploration Methodology

## 1. Problem Statement

Integrating DPE (Digital Processing Element) hard blocks into FPGA fabric requires trading general-purpose resources (CLBs, DSPs, BRAMs) for DPE tiles. This creates a fundamental tradeoff: more DPE area accelerates DL workloads but degrades non-DL FPGA performance.

**Key questions**:
1. *Which crossbar size (R×C) is best for DL workloads?* → Round 1
2. *What fraction of FPGA area should be allocated to DPE hard blocks?* → Round 2

## 2. DSE Overview

| Round | Question | Grid | Method |
|-------|----------|------|--------|
| **Round 1** | Which crossbar size is best? | Auto (per-workload) | 12 configs × 6 FC workloads, EDAP geomean ranking |
| **Round 2** | What is the optimal DPE area budget? | 60×60 fixed | 5 configs × 6 budgets, FlexScore + DL latency → Pareto front |

## 3. Round 1: Crossbar Size Selection

**Goal**: Identify the best DPE crossbar configuration (R×C) for FC workloads.

**Method**: VTR auto-layout (grid sized to each design). 12 crossbar configs × 6 FC workloads. EDAP (Energy-Delay-Area Product) ranking: per-workload best = 1.0, geomean across workloads determines rank.

**Metric**: EDAP = Energy(pJ) × Delay(ns) × Area(mm²). Lower is better.

**Design space justification**:
- R ∈ {128, 256, 512, 1024}: covers small to large crossbars
- C ∈ {64, 128, 256}: 3 column widths per row count
- Area overhead: smallest configs (256×64) have 47% peripheral overhead
- Power wall: at C≥256, ACAM power exceeds 50 mW per DPE

**Result**: Top-3 configs are R=512 (ACAM-eligible on 5/6 workloads). 512×128 ranked #1.

**Configs carried forward to Round 2**: 512×128, 1024×128, 1024×64 (NL-DPE group) + 1024×256, 512×256 (AL-like group).

## 4. Round 2: Proportional Budget Sweep (FlexScore DSE)

### 4.1 Problem

The 2D sweep (d% DSP, c% CLB) from earlier exploration was unfair — DPE replaces compute, memory, and routing functions, so all resource types should be reduced proportionally.

### 4.2 Proportional Area Budget

Single parameter: **area budget %**. Each FPGA resource type loses the same percentage of itself.

| Budget | CLB cols removed (of 50) | DSP cols removed (of 4) | BRAM cols removed (of 4) | CLBs left | DSPs left | BRAMs left |
|--------|--------------------------|-------------------------|--------------------------|-----------|-----------|------------|
| 0% | 0 | 0 | 0 | 2900 | 56 | 116 |
| 10% | 5 | 0 | 0 | 2610 | 56 | 116 |
| 20% | 10 | 1 | 1 | 2320 | 42 | 87 |
| 30% | 15 | 1 | 1 | 2030 | 42 | 87 |
| 40% | 20 | 2 | 2 | 1740 | 28 | 58 |
| 50% | 25 | 2 | 2 | 1450 | 28 | 58 |

Note: DSP/BRAM have only 4 columns each, so removal is discretized to 25% steps. CLB absorbs the rounding error.

### 4.3 Grid: 60×60

| Resource | 60×60 Grid |
|----------|-----------|
| CLBs | 2,908 (50 cols) |
| DSPs | 56 (4 cols) |
| BRAMs | 116 (4 cols) |

Sized so representative non-DL benchmarks fill 16–72% of resources at baseline, ensuring measurable degradation as DPE area increases.

### 4.4 DPE Configurations

| Config | Crossbar R×C | Tile W×H | Tile Group |
|--------|-------------|----------|------------|
| 512×128 | 512 rows, 128 cols | 3×8 | tw3 |
| 1024×128 | 1024 rows, 128 cols | 3×11 | tw3 |
| 1024×64 | 1024 rows, 64 cols | 3×6 | tw3 |
| 1024×256 | 1024 rows, 256 cols | 7×9 | tw7 |
| 512×256 | 512 rows, 256 cols | 4×12 | tw4 |

Configs with the same tile width produce identical FPGA layouts → identical FlexScore. FlexScore is determined by total DPE area, independent of crossbar configuration.

### 4.5 Dual-Metric Evaluation

**DL Performance: Effective Latency**

Bare GEMV workloads mapped to P data-parallel replicas:
```
Effective Latency = 1000 / (P × Fmax)    [ns/inference]
```
Geomean across 3 FC workloads (fc_512×128, fc_512×512, fc_2048×256). Bare GEMV is used because its throughput is CLB-limited (same resource FlexScore measures), creating a direct resource competition with non-DL FlexScore.

P = min(P_dpe, P_clb, P_bram) — 3-resource constraint. Per replica: ~25 CLBs, 4 BRAMs, V×H DPEs.

**Non-DL Performance: FlexScore**

Adopted from FlexScore (Tan et al., IEEE CAL 2021):
```
FlexScore(budget) = (1/N) × Σ [Fmax_i(budget) / Fmax_i(baseline)]
```
- Baseline = budget=0% (pure FPGA, no DPE tiles)
- If a benchmark cannot fit → ratio = 0 (zero performance)
- FlexScore = 1.0 means no degradation; 0 means all benchmarks fail

### 4.6 Non-DL Benchmarks

| Benchmark | CLBs | DSPs | BRAMs | Primary Stress | Expected Failure |
|-----------|------|------|-------|---------------|-----------------|
| bgm | 2,089 | 11 | 0 | CLB | budget ≥ 30% |
| LU8PEEng | 1,676 | 8 | 74 | CLB + BRAM | budget ≥ 40% |
| stereovision1 | 519 | 40 | 0 | DSP | budget ≥ 40% |
| arm_core | 662 | 0 | 42 | BRAM | survives to 50% |

Selected from VTR benchmark suite to cover diverse resource profiles.

### 4.7 DL Workloads (Bare GEMV)

| Workload | Matrix K×N |
|----------|-----------|
| fc_512_128 | 512×128 |
| fc_512_512 | 512×512 |
| fc_2048_256 | 2048×256 |

### 4.8 Multi-Seed and VTR Run Counts

3 random seeds per VTR point (averaged Fmax for stability).

| Experiment | Formula | VTR Runs |
|------------|---------|----------|
| Non-DL FlexScore (tw3) | 6 budgets × 4 benchmarks × 3 seeds | 72 |
| Non-DL FlexScore (tw4) | 6 × 4 × 3 | 72 |
| Non-DL FlexScore (tw7) | 6 × 4 × 3 | 72 |
| DL Bare GEMV (5 configs) | ~62 feasible points × 3 seeds | 186 |
| **Total Round 2** | | **~402** |

### 4.9 Pareto Front Visualization

**Plot axes**:
- X: Non-DL Performance Degradation (1 − FlexScore). Lower = more flexible (left = good).
- Y: Geomean Effective Latency (ns/inference). Lower = faster (bottom = good).
- Color: DPE Area % (lighter = less area, darker = more area).
- Bottom-left = ideal (low latency + low degradation).

**Layout**: 1×2 (NL-DPE Group | AL-like Group).

Each config shown with distinct marker shape. Pareto front (solid black line) connects non-dominated points. Knee point marks the balanced recommendation.

## 5. Generated Artifacts

### Scripts
| File | Description |
|------|-------------|
| `flexscore_dse.py` | Orchestrator: arch gen, non-DL sweep, DL sweep |
| `plot_pareto.py` | Pareto front plot |
| `plot_flexscore.py` | Scalability + per-benchmark + dual-Y plots |
| `plot_recommendation.py` | Sweet-spot recommendation plot |
| `gen_methodology_fig.py` | Methodology overview figure |

### Data Files
| File | Description |
|------|-------------|
| `flexscore_raw_results.csv` | Non-DL per-benchmark Fmax |
| `flexscore_summary.csv` | FlexScore per (tile_group, budget) |
| `flexscore_dl_gemv_results.csv` | DL bare GEMV per-workload latency |

### Plots
| File | Description |
|------|-------------|
| `round2_flexscore_pareto.pdf` | Main result: Pareto front |
| `round2_flexscore_scalability.pdf` | FlexScore vs budget per tile group |
| `round2_flexscore_per_benchmark.pdf` | Per-benchmark Fmax degradation |
| `round2_flexscore_main.pdf` | Dual-Y: latency + FlexScore vs budget |
| `round2_flexscore_recommendation.pdf` | Throughput + FlexScore sweet spot |
| `dse_methodology_overview.pdf` | Methodology flow diagram |
