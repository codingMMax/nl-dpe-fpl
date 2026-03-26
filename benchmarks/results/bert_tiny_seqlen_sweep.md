# BERT-Tiny Energy vs Sequence Length Sweep

## DIMM % of Total Energy

| seq_len | NL-DPE DIMM% | Azure-Lily DIMM% | NL-DPE Total (pJ) | AL Total (pJ) | NL-DPE / AL |
|---------|-------------|-----------------|-------------------|---------------|-------------|
| 128 | 96.9% | 57.1% | 10,882,187 | 13,472,974 | **1.24×** (NL wins) |
| 256 | 98.3% | 72.7% | 39,976,255 | 42,313,025 | **1.06×** (NL wins) |
| 512 | 99.1% | 84.2% | 151,859,817 | 146,107,833 | **0.96×** (AL wins) |
| 1024 | 99.5% | 91.4% | 589,468,386 | 538,142,897 | **0.91×** (AL wins) |

## Per-Group Ratio (NL-DPE Advantage)

| seq_len | Proj+FFN | DIMM+Softmax | Total |
|---------|----------|-------------|-------|
| 128 | 43.5× | 0.73× | 1.24× |
| 256 | 43.5× | 0.78× | 1.06× |
| 512 | 43.5× | 0.82× | 0.96× |
| 1024 | 43.5× | 0.84× | 0.91× |

## Key Insight

**NL-DPE DIMM costs MORE than Azure-Lily DSP MACs** — ratio ranges from 0.73× to 0.84×.

As seq_len grows:
- DIMM energy scales O(N²) and dominates both architectures
- NL-DPE DIMM% grows from 96.9% → 99.5%
- Azure-Lily DIMM% grows from 57.1% → 91.4%
- **At seq_len ≥ 512, Azure-Lily becomes more energy-efficient overall** because its DSP MACs are cheaper per DIMM operation than NL-DPE's identity crossbar + ACAM exp

The crossover happens at seq_len ≈ 400:
- seq_len < 400: NL-DPE wins (projection savings dominate)
- seq_len > 400: Azure-Lily wins (DIMM cost dominates)

**Root cause**: NL-DPE's identity crossbar firing (75 pJ per pass) is more expensive
than Azure-Lily's DSP MAC (0.908 pJ × d_head MACs ≈ 58 pJ per element).
The ACAM advantage only applies to weight-persistent VMM (projections), not to
identity crossbar passes (DIMM).
