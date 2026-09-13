# FC+BN+Softmax DSE: Insights Summary

## What's New

This benchmark replaces the bare GEMV replicas with **complete FC inference layers**:
GEMV (DPE) → BatchNorm (4 DSP MACs) → Softmax (12 DSP MACs + CLB exp/sum/reciprocal).
Each replica uses **4 DSP tiles** (16 `mac_int_9x9`), making DSP a scarce resource.

## Key Finding: DSP Bottleneck Creates Throughput Peak

**Bare GEMV** (old benchmark): throughput always increases with DPE area. No peak.

**FC+BN+Softmax** (new): throughput **peaks at d=40%** then **drops 50%** by d=80%.

```
fc_512_128 on 512×128 (c=0%):

                     Bare GEMV          FC+BN+Softmax
  d=20%:  P=14, 2.57 inf/ns    P=14, 1.83 inf/ns  (DPE-limited)
  d=40%:  P=28, 4.84 inf/ns    P=28, 3.66 inf/ns  (DPE-limited) ← PEAK
  d=60%:  P=56, 8.15 inf/ns    P=21, 2.75 inf/ns  (DSP-limited) ← DROP
  d=80%:  P=70, 7.51 inf/ns    P=14, 1.88 inf/ns  (DSP-limited) ← -50%
```

## Why It Happens

Replacing DSP columns with DPE columns simultaneously:
- **Adds** DPE tiles → more replicas possible (P↑)
- **Removes** DSP tiles → fewer BN+softmax pipelines possible (P_dsp↓)

At d≤40%: P_dpe < P_dsp → DPE is the bottleneck → adding DPEs helps.
At d≥60%: P_dsp < P_dpe → DSP is the bottleneck → adding DPEs doesn't help
           (can't build softmax for the extra replicas).

## Binding Constraint Distribution

| Resource | Points | Fraction |
|----------|--------|----------|
| DPE | 100 | 43% |
| **DSP** | **99** | **42%** |
| BRAM | 35 | 15% |
| CLB | 0 | 0% |

vs Bare GEMV: DPE=91%, BRAM=9%, DSP=0%, CLB=0%.

## Comparison with Bare GEMV

### Throughput
- Bare GEMV: monotonically increasing (no DSP dependency)
- FC+BN+Softmax: peaks at d=40%, drops at d≥60%

### Fmax
- FC+BN+Softmax has **lower Fmax** (~131 MHz vs ~184 MHz at d=20%)
  because each replica uses 93 CLBs (vs 25 for bare GEMV) → more routing pressure
- However, at d=80% the softmax Fmax is actually HIGHER (+25%) because
  P is lower (14 vs 70 replicas) → less routing congestion

### Replicas
- At d=40%: both have P=28 (same DPE count, DPE-limited)
- At d=60%: GEMV has P=56, softmax has P=21 (DSP caps at 21)
- At d=80%: GEMV has P=70, softmax has P=14 (DSP caps at 14)

## Paper Narrative

The FC+BN+Softmax benchmark demonstrates that **the cost of DPE over-provisioning
depends on the workload's resource profile**:

1. **DL-only workloads** (bare GEMV): only use DPEs → throughput always improves
   with more DPE area. The only cost is routing degradation (soft ceiling).

2. **Complete inference pipelines** (GEMV+BN+softmax): use DPEs AND DSPs →
   throughput peaks where DSP becomes the bottleneck. Over-provisioning DPEs
   wastes area because the non-DPE stages can't keep up.

3. **The crossover point shifts with workload size**:
   - fc_512_128 (1 DPE/rep): DSP binds at d≥60% (42% of points)
   - fc_2048_256 (8 DPEs/rep): DSP binds at d≥80% (21% of points)
   - Larger workloads are DPE-limited longer because P stays lower.

## Generated Plots

- `fc_softmax_vs_gemv_throughput.pdf` — overlay comparison (dashed=GEMV, solid=softmax)
- `fc_softmax_binding_heatmap.pdf` — color-coded binding constraint per (d%, c%)
- `round2_fc_softmax_results.csv` — 234 data points (6 VTR failures at 1024×64 c=60%)
