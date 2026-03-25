# BERT-Tiny Energy Breakdown (seq_len=128, d=128)

## Per-Layer Category

| Category | NL-DPE Proposed | | NL-DPE AL-Matched | | Azure-Lily | |
|----------|---:|---|---:|---|---:|---|
| | **Energy (pJ)** | **%** | **Energy (pJ)** | **%** | **Energy (pJ)** | **%** |
| Embedding + LN | 43,109 | 0.4% | 43,109 | 0.3% | 44,086 | 0.3% |
| Q/K/V Projections | 42,700 | 0.4% | 59,554 | 0.4% | 1,837,253 | 13.6% |
| **DIMM (QK^T)** | **6,656,187** | **61.2%** | **8,243,509** | **61.9%** | **3,810,912** | **28.3%** |
| Softmax | 119,371 | 1.1% | 141,843 | 1.1% | 67,197 | 0.5% |
| **DIMM (Score×V)** | **3,769,200** | **34.6%** | **4,586,525** | **34.4%** | **3,810,886** | **28.3%** |
| O Projection | 14,233 | 0.1% | 19,851 | 0.1% | 612,418 | 4.5% |
| Residual Add | 7,516 | 0.1% | 7,516 | 0.1% | 10,436 | 0.1% |
| LayerNorm | 158,702 | 1.5% | 158,702 | 1.2% | 158,717 | 1.2% |
| FFN (FFN1+FFN2) | 71,167 | 0.7% | 60,852 | 0.5% | 3,121,070 | 23.2% |
| **TOTAL** | **10,882,187** | **100%** | **13,321,462** | **100%** | **13,472,974** | **100%** |

## Execution Target

| Category | NL-DPE | Azure-Lily |
|----------|--------|------------|
| Q/K/V/O Projections | DPE + ACAM (log) | DPE + ADC |
| FFN1/FFN2 | DPE + ACAM (GELU/log) | DPE + ADC |
| DIMM (QK^T) | DPE(I\|exp) + CLB add/reduce | DSP MAC |
| DIMM (Score×V) | DPE(I\|exp/log) + CLB add/reduce | DSP MAC |
| Softmax | DPE(I\|exp) + CLB | CLB + DSP |
| LayerNorm | CLB + DSP | CLB + DSP |
| Embedding | SRAM + CLB | SRAM + CLB |
| Residual | CLB | CLB |

## Grouped Summary

| Group | NL-DPE Proposed | | Azure-Lily | | NL-DPE Advantage |
|-------|---:|---|---:|---|---:|
| | **Energy** | **%** | **Energy** | **%** | |
| Projections + FFN (DPE) | 128,101 | 1.2% | 5,570,740 | 41.3% | **43.5×** |
| Attention DIMM | 10,544,758 | 96.9% | 7,688,995 | 57.1% | **0.73×** (worse) |
| Other (LN+Res+Embed) | 209,327 | 1.9% | 213,239 | 1.6% | 1.0× |
| **TOTAL** | **10,882,187** | | **13,472,974** | | **1.2×** |

## Key Insight

NL-DPE's DIMM (log-domain DPE(I|exp) + CLB add) costs **more** than Azure-Lily's DSP MACs
for BERT-Tiny attention. The 43.5× projection savings is overwhelmed by the 0.73× DIMM penalty
because DIMM constitutes 96.9% of NL-DPE's total energy at seq_len=128.
