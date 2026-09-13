# DSE Experiment: Writing Materials

This document captures all methodology, implementation details, assumptions, metrics,
and key findings from the NL-DPE crossbar-size DSE. Scoped to the DSE experiment only —
not the full paper narrative.

Last updated: 2026-03-20

---

## §1 DSE Overview

**Objective**: Find optimal ReRAM crossbar dimensions (rows R, columns C) for the NL-DPE
FPGA hard block, evaluating FC+activation and attention workloads.

**Three experimental phases:**

| Phase | Scope | VTR Runs | Grid | Purpose |
|-------|-------|----------|------|---------|
| Round 1 | 12 configs × 6 workloads | 72 | auto-layout | Block sizing: find optimal (R, C) |
| Round 2 FC | 5 configs × 3 workloads × 20 (d,c) × 3 seeds | 900 | 120×120 fixed | Density study: how much FPGA area to trade |
| Round 2 Attention | 5 configs × 1 workload × 16 (d,c) × 3 seeds | 240 | 120×120 fixed | Attention scalability: DIMM DPE mapping |

**Tools:**
- VTR 8.0 (place-and-route → Fmax, resource counts)
- IMC simulator (energy/latency analytical model)
- area_power.py (DPE tile specs from 32nm physical design data)
- gemv_dse.py (orchestrator: RTL gen → arch XML → VTR → IMC → CSV)

---

## §2 Metric Definitions

### Primary Metrics

| Metric | Formula | Unit | Notes |
|--------|---------|------|-------|
| Amortized time per inference | 1000 / (P × Fmax_MHz) | ns | Inverse throughput. Average time between consecutive outputs with P parallel replicas. NOT single-inference latency. |
| Throughput | P × Fmax | MHz·replicas | Proportional to inferences/second |
| Utilization | fₙ / f₀ | ratio | Normalized Fmax. f₀ = baseline Fmax at (d=20%, c=0%). Values >1.0 possible (VTR finds better routing at certain densities). |
| Gain | P × fₙ / f₀ | ratio | Throughput scaling factor relative to baseline |
| FPGA area | grid_W × grid_H × 2239 / 1e6 | mm² | 2239 µm² = CLB tile area (routing-aware) |
| Energy ratio | E₀ / Eₙ | ratio | >1 = improvement. E₀ = baseline energy. |

### Geomean (FC workloads)

For FC Pareto and ceiling plots, metrics are aggregated across 3 workloads using
**geometric mean**. Justification:
- SPEC-style normalization (standard in computer architecture benchmarks)
- Scale-invariant: prevents large workloads from dominating
- Balanced contribution from workloads with different absolute latencies

### Round 1 Ranking

SPEC-style normalized geomean combining throughput/mm² and throughput/J:
1. Per-workload best = 1.0 (normalize each metric)
2. Geomean across 6 workloads → per-config GM_tput_mm2, GM_tput_J
3. Combined = geomean(GM_tput_mm2, GM_tput_J)

---

## §3 Round 1: Auto-Layout Block Sizing

### Setup

- **12 configs**: R ∈ {128, 256, 512, 1024}, C ∈ {64, 128, 256}
- **6 workloads**: fc_64_64, fc_128_128, fc_512_128, fc_256_512, fc_512_512, fc_2048_256
- **72 VTR runs** (single seed, auto_layout — VTR sizes grid to fit each design)
- **Tiling**: V = ceil(K/R), H = ceil(N/C). ACAM-eligible iff V == 1.

### Key Results

**Configuration ranking (combined geomean):**

| Rank | Config | GM Combined | ACAM-eligible workloads |
|------|--------|-------------|------------------------|
| 1 | 512×128 | 0.852 | 5/6 |
| 2 | 1024×128 | 0.635 | 6/6 |
| 3 | 1024×64 | 0.626 | 6/6 |
| 4 | 1024×256 | 0.596 | 6/6 |
| 5 | 512×256 | 0.573 | 5/6 |
| 6 | 512×64 | 0.554 | 5/6 |

**ACAM eligibility by row count:**
- R=512: 5/6 workloads V=1 (all except fc_2048_256)
- R=1024: 6/6 workloads V=1
- R=256: 3/6 workloads V=1
- R=128: 2/6 workloads V=1

**Energy discontinuity (V=1 vs V>1):**
- fc_512_128: V=1 configs → 159–275 pJ; V=2 configs → 328–573 pJ (**2–4× gap**)
- This step-function is the ACAM value: V=1 absorbs activation for free

**Top-5 advance to Round 2**: 512×128, 1024×128, 1024×64, 1024×256, 512×256

**Figures**: round1_ranking.pdf, round1_heatmap.pdf
**Data**: round1_results.csv (72 rows)

---

## §4 Round 2 FC: Fixed-Layout Scalability

### Setup

- **5 configs**: 512×128, 1024×128, 1024×64, 1024×256, 512×256
- **3 workloads**: fc_512_128, fc_512_512, fc_2048_256
- **Grid**: 120×120 fixed (fits all 5 configs including 1024×256 tile 7×9)
- **DSP replacement sweep**: d ∈ {20%, 40%, 60%, 80%, 100%}
- **CLB replacement sweep**: c ∈ {0%, 20%, 40%, 60%}
- **20 (d,c) points per config×workload**, 3 seeds per point
- **Total**: 5 × 3 × 20 × 3 = **900 VTR runs** → 300 data points (Fmax averaged)

### Feasibility Model (3-resource constraint)

```
P = min(P_dpe, P_clb, P_bram)

P_dpe = count_available_wc(grid, tile_w, tile_h, d%, c%) // (V × H)
P_clb = (clbs_avail - 30) // (clbs_per_rep + 5)
        where clbs_avail = (1 - c%) × 10978
P_bram = 472 // brams_per_rep
        where brams_per_rep = 4 (K ≤ 1024) or 6 (K > 1024)
```

- Baseline: d=20%, c=0% for each (config, workload) → defines f₀

### Key Findings

**4.1 Soft ceiling from routing degradation:**
- Utilization drops from 100% → 49–55% at d=100%+c=60%
- Throughput per DPE drops ~50% from low to high density
- Negative marginal returns at some high-density points (adding DPEs hurts throughput)
- 80% of peak throughput achieved at only 57–86% of max DPE area

**4.2 Binding constraint distribution (300 points):**

| Workload | DPE-limited | CLB-limited | BRAM-limited |
|----------|-------------|-------------|--------------|
| fc_512_128 | 73% | 0% | 27% |
| fc_512_512 | 100% | 0% | 0% |
| fc_2048_256 | 100% | 0% | 0% |
| **Overall** | **91%** | **0%** | **9%** |

CLB is **never** the binding constraint. BRAM only binds for fc_512_128 (1 DPE/rep → very high P).

**4.3 Merged Pareto front (cross-config, geomean effective latency):**

| Group | Knee Config | Knee Point | DPE Area | Amortized Time |
|-------|-------------|------------|----------|----------------|
| NL-DPE (512×128, 1024×128, 1024×64) | 1024×128 | d=20%, c=20% | 9.17% | 0.336 ns/inf |
| AL-like (1024×256, 512×256) | 512×256 | d=60%, c=0% | 12.0% | 0.325 ns/inf |

**Figures**: round2_full_scalability.pdf, round2_full_per_workload.pdf,
round2_full_pareto.pdf, round2_full_pareto_merged.pdf, round2_full_ceiling.pdf
**Data**: round2_full_results.csv (300 rows)

---

## §5 Round 2 Attention: Fixed-Layout with DIMM DPEs

### Setup

- **5 configs**: same as FC
- **1 workload**: attention head (N=128 seq_length, d=128 head_dim)
- **DSP sweep**: d ∈ {20%, 40%, 60%, 80%} — d=100% **excluded** (softmax needs DSPs)
- **CLB sweep**: c ∈ {0%, 20%, 40%, 60%}
- **16 (d,c) points per config**, 3 seeds per point
- **Total**: 5 × 16 × 3 = **240 VTR runs** → 80 data points

### §5.1 Attention Architecture (Paper Fig 6c Mapping)

The NL-DPE attention pipeline operates in the log domain:

```
1. Linear_Q:  X_Q → DPE(W_Q | log) → log(Q)     [weight-persistent GEMV, ACAM=log]
2. Linear_K:  X_K → DPE(W_K | log) → log(K)
3. Linear_V:  X_V → DPE(W_V | log) → log(V)
4. DMMul_1:   CLB add(log_Q + log_K) → DPE(I|exp) → CLB reduce → S scores
5. Softmax:   DPE(I|exp) → CLB sum → CLB reciprocal → CLB multiply → attn weights
6. DMMul_2:   DPE(I|log) on attn → CLB add(log_attn + log_V) → DPE(I|exp) → CLB reduce → output
```

- `DPE(W|log)` = weight-persistent crossbar + ACAM configured as log
- `DPE(I|exp)` = identity-weight crossbar + ACAM configured as exp
- `DPE(I|log)` = identity-weight crossbar + ACAM configured as log
- CLB handles only linear operations (addition, reduction, reciprocal, normalization multiply)

### §5.2 DPE Counting Formula

```
V = ceil(d / R)              # vertical tiles per projection
H = ceil(max(d, N) / C)      # horizontal tiles (output limited by C columns)

Projection DPEs = 3 × V × H  (Q, K, V)
DIMM DPEs       = 4 × H       (score_exp, softmax_exp, wsum_log, wsum_exp)
Total           = (3V + 4) × H
```

| Config | V | H | Proj DPEs | DIMM DPEs | Total/rep |
|--------|---|---|-----------|-----------|-----------|
| 512×128 | 1 | 1 | 3 | 4 | **7** |
| 1024×128 | 1 | 1 | 3 | 4 | **7** |
| 1024×64 | 1 | 2 | 6 | 8 | **14** |
| 1024×256 | 1 | 1 | 3 | 4 | **7** |
| 512×256 | 1 | 1 | 3 | 4 | **7** |

### §5.3 Feasibility Model (4-resource constraint)

```
P = min(P_dpe, P_clb, P_bram, P_dsp)

P_dpe  = count_available_wc() // dpes_per_rep
P_clb  = (clbs_avail - 15) // 145           [145 CLBs/rep, empirical]
P_bram = (472 - 4) // 64                    [64 BRAMs/rep, empirical] = 7
P_dsp  = dsps_avail // 2                    [2 DSPs/rep for softmax multiply]
```

Per-replica resources (empirically calibrated via P=1,2,3 VTR runs):
- **145 CLBs** (adders, accumulators, FSMs, reciprocal LUT)
- **64 BRAMs** (Q/K/V/score/attn/output buffers for N=128, d=128)
- **2 DSPs** (softmax normalization multiply)

### §5.4 Key Findings

**BRAM wall (hard ceiling):**
- P capped at **7** (64 BRAMs/rep × 7 = 448, out of 472 available)
- Hard constraint: unlike FC soft ceiling, attention hits absolute wall at P=7
  regardless of how many DPEs are available
- Binding constraint: BRAM at most points; DPE only at low d%

**High utilization (misleading):**
- 85–105% utilization across all points
- But this is because P never exceeds 7 → design never stresses routing
- The "good scalability" is an artifact of being BRAM-constrained

**d=100% exclusion:**
- Softmax normalization requires DSP multiply blocks
- At d=100%, all DSPs replaced by DPEs → no DSPs for softmax → infeasible

**Cross-workload contrast (key paper insight):**

| | FC | Attention |
|---|---|---|
| Ceiling type | Soft (routing degradation) | Hard (BRAM wall) |
| P range | 1 → 52 | 1 → 7 |
| BRAMs/rep | 4–6 | **64** |
| Binding constraint | DPE (91%) | BRAM (most) |
| Utilization at max P | 49–55% | 85–105% |

**Merged Pareto:**
- Less differentiation between configs (all BRAM-capped at P=7)
- NL-DPE group and AL-like group show similar knee points

**Figures**: round2_attention_scalability.pdf, round2_attention_pareto.pdf,
round2_attention_pareto_merged.pdf, round2_attention_ceiling.pdf
**Data**: round2_attention_results.csv (80 rows)

---

## §5b Round 2 FC+BN+Softmax: DSP Bottleneck Benchmark

### Setup

- **Benchmark**: Complete FC layer = GEMV (DPE) + BatchNorm (DSP) + Softmax (CLB+DSP)
- **16 mac_int_9x9 per replica** = 4 DSP tiles/rep (4 BN + 12 softmax normalize)
- **Per-replica resources** (calibrated): 93 CLBs, 16 BRAMs, 4 DSP tiles
- **4-resource constraint**: P = min(P_dpe, P_clb, P_bram, P_dsp)
- **5 configs × 3 workloads × 16 (d,c) × 3 seeds = 720 VTR runs** → 234 data points (6 failures)
- **d=100% excluded**: 0 DSPs remaining → infeasible

### Key Findings

**5b.1 DSP crossover creates throughput peak:**

fc_512_128 on 512×128 (c=0%):
- d=20%: P=14 (DPE-limited) → 1.83 inf/ns
- d=40%: P=28 (DPE-limited) → **3.66 inf/ns** ← PEAK
- d=60%: P=21 (DSP-limited) → 2.75 inf/ns ← -25%
- d=80%: P=14 (DSP-limited) → 1.88 inf/ns ← -49%

**5b.2 Binding distribution:**

| Resource | Points | Fraction |
|----------|--------|----------|
| DPE | 100 | 43% |
| **DSP** | **99** | **42%** |
| BRAM | 35 | 15% |
| CLB | 0 | 0% |

vs Bare GEMV: DPE=91%, BRAM=9%, DSP=0%.

**5b.3 Balanced config shifts with workload resource profile:**

| Benchmark | NL-DPE group balanced | Why |
|-----------|----------------------|-----|
| Bare GEMV (DPE-only) | **1024×128** d=20% c=20% | Fewer DPEs/rep → more replicas |
| FC+BN+Softmax (DSP) | **512×128** d=40% c=0% | DSP caps P → smaller tile wastes less |
| Attention (BRAM) | **512×128** d=20% c=20% | BRAM caps P=7 → smaller tile wastes less |

**512×128 is the robust choice** — optimal for resource-constrained workloads (DSP or BRAM limited).
1024×128 only wins for purely DPE-limited workloads (bare GEMV).

**5b.4 Cross-workload contrast (updated):**

| | Bare FC | FC+BN+Softmax | Attention |
|---|---|---|---|
| Non-DPE bottleneck | None | **DSP** | **BRAM** |
| Throughput peak? | No (monotonic) | **Yes** (d=40%) | No (BRAM wall) |
| Binding at high d% | DPE | DSP | BRAM |
| Balanced config | 1024×128 | 512×128 | 512×128 |

**Figures**: fc_softmax_pareto.pdf, fc_softmax_pareto_merged.pdf,
fc_softmax_vs_gemv_throughput.pdf, fc_softmax_binding_heatmap.pdf,
fc_softmax_dsp_sweep_*.pdf
**Data**: round2_fc_softmax_results.csv (234 rows)

---

## §6 IMC Simulator: Energy Model Details

### §6.1 Architecture Configs

| Parameter | NL-DPE | Azure-Lily |
|-----------|--------|------------|
| e_analoge_pj | 3.89 | 0 |
| e_conv_pj | 0 | 2.33 (calibrated) |
| e_digital_pj | 0.171445 × cols | 0 |
| k_vmm (passes) | 8 (8-bit) | 8 |
| k_digital | 1 (fires once) | 0 |
| bram_width | 40 bits | 16 bits |
| Fmax (DPE core) | 1 GHz | 1 GHz |
| Fmax (FPGA fabric) | patched from VTR | patched from VTR |

### §6.2 FC Energy Breakdown

Per DPE pass (one bit-slice):
```
E_vmm     = k_vmm × e_analoge_pj = 8 × 3.89 = 31.12 pJ
E_conv    = k_conv × e_conv_pj   = 0 (NL-DPE) or 8 × 2.33 (Azure-Lily)
E_digital = k_digital × e_digital_pj × cols = 1 × 0.171445 × C
```

Additional per-layer:
```
E_reduction = (V-1) × N × e_clb_pj_per_mac × clb_coeff_add    [V>1 only]
E_activation = M × N × act_energy_pj_per_op                    [V>1 only]
E_sram = read + write energy (0.0495 pJ/access × accesses)
```

### §6.3 Attention Energy Mapping (aligned with Fig 6c)

| Stage | Simulator Path | NL-DPE Energy Source |
|-------|---------------|---------------------|
| Linear Q/K/V | `imc_core.run_gemm()` | DPE VMM + ACAM (log) |
| Q·K^T (DMMul_1) | `fpga.gemm_log()` | CLB add + DPE(I\|exp) + CLB reduce |
| Softmax exp | `imc_core.dimm_nonlinear(d, "exp")` | DPE(I\|exp) per row |
| Softmax norm | `fpga.norm_fpga()` + `dimm_nonlinear(1, "log")` | CLB sum + DPE(I\|log) replaces CLB inverse |
| Score·V (DMMul_2) | `dimm_nonlinear(N, "log")` + `fpga.gemm_log()` | DPE(I\|log) on attn + CLB add + DPE(I\|exp) + CLB reduce |

**Key fix applied**: mac_sv was missing `DPE(I|log)` step. Without it, attention weights
(linear domain from softmax) were fed directly into log-domain vector add — mathematically
wrong. Fixed by adding `imc_core.dimm_nonlinear(N, "log")` per row before `gemm_log()`.

### §6.4 Energy Breakdown Keys

```
DPE group:  imc_vmm, imc_conversion, imc_digital_post, imc_dimm_exp, imc_dimm_log
Memory:     sram_read, sram_write
FPGA:       clb_reduction, clb_add, clb_activation, clb_norm_sum, dsp_multiply
```

---

## §7 Implementation Constants & Assumptions

All assumptions requiring justification in the paper:

| # | Assumption | Value | Justification |
|---|-----------|-------|---------------|
| 1 | CLB tile area | 2239 µm² | Routing-aware: SB=688 + 2×CB=303 + logic=945. Validated against VTR grid sizing. |
| 2 | Fixed grid size | 120×120 | Fits largest config (1024×256, tile 7×9). Provides sufficient CLB/BRAM baseline. |
| 3 | Multi-seed count | 3 seeds | VTR placement stochasticity causes ±15% Fmax variation. 3-seed average for stability. |
| 4 | Baseline point | d=20%, c=0% | Minimal DPE footprint that still provides P≥1 for all configs. Defines reference f₀. |
| 5 | FC BRAM/rep | 4 (K≤1024), 6 (K>1024) | Empirical from VTR: weight SRAM depth requires 4 or 6 BRAMs. |
| 6 | Attention CLBs/rep | 145 | Empirical: VTR P=1,2,3 calibration runs. Includes adders, accumulators, FSMs. |
| 7 | Attention BRAMs/rep | 64 | Empirical: VTR calibration. Q/K/V/score/attn/output buffers for N=128, d=128. |
| 8 | Attention DSPs/rep | 2 | Softmax normalization multiply. One DSP per multiply instance. |
| 9 | DIMM identity crossbar | DPE(I\|exp/log) | Weight-persistent identity matrix + ACAM for nonlinear. Same DPE hard block, different programming. |
| 10 | Amortized time | 1000/(P×Fmax) | Assumes perfect pipelining across replicas. No inter-replica synchronization overhead. |
| 11 | Baseline CLBs | 10,978 | 120×120 grid, counted from VTR output. |
| 12 | Baseline BRAMs | 472 | height=2, startx=2, repeatx=16 → 8 columns × 59/column. |
| 13 | Baseline DSPs | ~210 | startx=6, repeatx=16 → ~14 columns on 120×120. |
| 14 | DPE tile overshoot | ≤10% | Routing-aware formula minimizes W×H grid cells. Accept tile if area overhead <10%. |
| 15 | d=100% excluded (attention) | No DSPs for softmax | Softmax normalization multiply requires DSP blocks. All DSPs traded → infeasible. |

---

## §8 Plot Inventory

### Round 1 (dse/results/plots/round1/)

| File | Type | Content |
|------|------|---------|
| round1_ranking.pdf | Bar chart | Config ranking by combined geomean |
| round1_heatmap.pdf | Heatmap | Workload × config Fmax/energy |

### Round 2 FC (dse/results/plots/round2_fc/)

| File | Type | Content |
|------|------|---------|
| round2_full_scalability.pdf | Heatmap (3+2) | Geomean utilization per config |
| round2_full_per_workload.pdf | Heatmap (1×3) | Per-workload mean utilization |
| round2_full_pareto.pdf | Scatter (2×3) | Per-config Pareto (area vs amortized time) |
| round2_full_pareto_merged.pdf | Scatter (1×2) | Cross-config Pareto (NL-DPE + AL-like groups) |
| round2_full_ceiling.pdf | Line (2×3) | Throughput ceiling: actual vs ideal |

### Round 2 Attention (dse/results/plots/round2_attention/)

| File | Type | Content |
|------|------|---------|
| round2_attention_scalability.pdf | Heatmap (3+2) | Utilization per config |
| round2_attention_pareto.pdf | Scatter (2×3) | Per-config Pareto |
| round2_attention_pareto_merged.pdf | Scatter (1×2) | Cross-config Pareto |
| round2_attention_ceiling.pdf | Line | Throughput ceiling (BRAM wall) |

### Data Files

| File | Rows | Content |
|------|------|---------|
| round1_results.csv | 72 | Round 1: 12 configs × 6 workloads |
| round2_full_results.csv | 300 | Round 2 FC: 5 configs × 3 workloads × 20 (d,c) |
| round2_attention_results.csv | 80 | Round 2 Attention: 5 configs × 16 (d,c) |
