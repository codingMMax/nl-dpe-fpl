# NL-DPE Attention Mapping with Dual-Identity Optimization

## 1. Problem: DIMM Energy Bottleneck

The transformer attention mechanism requires Dynamic Input Matrix-Matrix Multiplication (DIMM):
- QK^T: S[i][j] = Σ_m Q[i][m] · K[j][m]  (N² elements, dot product of d_head)
- Score×V: O[i][m] = Σ_j attn[i][j] · V[j][m]  (N×d elements, dot product of N)

Unlike weight-persistent GEMV (projections), DIMM involves **two dynamic inputs** — the ReRAM
crossbar cannot store both matrices as persistent weights. The NL-DPE solves this by operating
in the **log domain**: replace multiply with add, use DPE ACAM for exp/log conversion.

**The challenge**: The DPE identity crossbar pass (for exp conversion) fires ALL rows and columns
regardless of how many elements are processed. For d_head=64 on a 1024×128 crossbar:
- Only 64 of 1024 rows have identity weight=1 (6.2% utilization)
- Only 64 of 128 columns produce useful outputs (50% utilization)
- Per-pass energy: 53.1 pJ for 64 exp values = 0.83 pJ/element

This makes DIMM the energy bottleneck (96.9% of total BERT-Tiny energy on NL-DPE).

## 2. Dual-Identity Mapping: Key Innovation

**Observation**: With d_head=64 and C=128 columns, half the crossbar columns are wasted on
zeros. We can pack **two independent 64-element identity blocks** into one 128-column crossbar:

```
Crossbar layout (1024 rows × 128 columns):

Columns:   [ 0 ─── 63 | 64 ─── 127 ]
           ┌──────────┬────────────┐
Row 0      │    1     │     0      │  ← Identity block A
Row 1      │    1     │     0      │
  ...      │   ...    │    ...     │
Row 63     │    1     │     0      │
Row 64     │    0     │     1      │  ← Identity block B
Row 65     │    0     │     1      │
  ...      │   ...    │    ...     │
Row 127    │    0     │     1      │
Row 128+   │    0     │     0      │  ← unused (zero)
           └──────────┴────────────┘

Input:  [vec_A (64 elements) | vec_B (64 elements) | zeros (896)]
Output: [exp(vec_A) (64 values) | exp(vec_B) (64 values)]
```

**One DPE pass now processes TWO d_head-sized vectors simultaneously.**

### Energy Savings

| | Single Identity (current) | Dual Identity |
|---|---|---|
| Elements per pass | 64 | **128** (2 × 64) |
| DPE energy per pass | 53.1 pJ | 53.1 pJ (same) |
| **Cost per element** | 0.83 pJ | **0.41 pJ** |
| QK^T per element (total) | 63.9 pJ | **37.3 pJ** (-42%) |
| vs DSP MAC (76.8 pJ) | 1.20× | **2.06×** |

### Applicability

- **QK^T (d_head=64, C=128)**: ✓ dual-pack — 2×64 = 128 = C. Perfect fit.
- **Score×V (N=128, C=128)**: ✗ no benefit — N already fills C. Already optimal.
- **Softmax exp (N=128, C=128)**: ✗ no benefit — same reason.
- **Requirement**: 2 × d_head ≤ C. Satisfied for all standard transformers (d_head=64, C=128).

### No Hardware Change Required

Dual-identity is purely a **weight programming** change. The same DPE hard block, same ACAM,
same interface. Only the ReRAM crossbar weights differ (two identity blocks instead of one).

## 3. Complete Attention Head Mapping (with Dual-Identity)

### NL-DPE FPGA (1024×128 or 512×128 crossbar)

**Config**: R≥128, C=128 (supports dual-identity for d_head=64)

```
=== PROJECTIONS (weight-persistent GEMV, ACAM=log) ===

X_Q → [DPE: W_Q crossbar | ACAM=log] → log(Q)     V=1, H=1, 1 DPE
X_K → [DPE: W_K crossbar | ACAM=log] → log(K)     V=1, H=1, 1 DPE
X_V → [DPE: W_V crossbar | ACAM=log] → log(V)     V=1, H=1, 1 DPE

  Energy: 53.1 pJ per DPE pass × N passes per projection
  ACAM configured as log → outputs in log domain (no separate conversion)
  V=1 for d_model=128 ≤ R → ACAM eligible on ALL projections
```

```
=== DMMul_1: Q·K^T SCORE MATRIX (dual-identity DIMM) ===

For each output S[i][j] = Σ_m exp(log_Q[i][m] + log_K[j][m]):

Step 1: CLB add       log_Q[i][m] + log_K[j][m] for m=0..63    (64 parallel CLB adders)
Step 2: DPE(I²|exp)   Feed 2 sets of 64 sums into DUAL-IDENTITY crossbar
                       → ACAM outputs 128 exp values (2 × 64)
                       → processes elements (i,j) and (i,j+1) simultaneously
Step 3: CLB reduce    Σ_m exp(...) for each of the 2 elements    (2 parallel reduction trees)

  DPE pass energy: 53.1 pJ for 2 elements → 26.6 pJ/element
  Total per element: 10.8 pJ (CLB) + 26.6 pJ (DPE) = 37.3 pJ
  vs DSP MAC: 64 × 1.2 = 76.8 pJ → 2.06× NL-DPE advantage

  Output: S scores matrix (N × N) in linear domain
```

```
=== SOFTMAX ===

Step 1: DPE(I|exp)    exp(S[i][j]) for each row        1 DPE, ACAM=exp
Step 2: CLB sum       Σ_j exp(S[i][j])                 CLB reduction tree
Step 3: CLB recip     1/sum                             CLB priority-encoder LUT
Step 4: CLB multiply  exp(S[i][j]) × (1/sum)           CLB or DSP multiply

  Output: attn weights (N × N) in linear domain
```

```
=== DMMul_2: ATTENTION × V WEIGHTED SUM ===

Step 1: DPE(I|log)    log(attn[i][j]) for all j         1 DPE, ACAM=log
Step 2: CLB add       log_attn[i][j] + log_V[j][m]      CLB parallel adders
Step 3: DPE(I|exp)    exp(sum) → N=128 elements          1 DPE, ACAM=exp
                       N=C=128 → single identity, fully utilized (no dual needed)
Step 4: CLB reduce    Σ_j exp(...) → output O[i][m]      CLB reduction tree

  Output: attention output (N × d_head) in linear domain
```

```
=== POST-ATTENTION ===

Concat heads → [DPE: W_O | ACAM=log]     1 DPE, output projection
→ [CLB + SRAM: Residual Add]
→ [CLB + DSP: LayerNorm]
```

```
=== FFN ===

→ [DPE: W_FFN1 | ACAM=GELU]    V=1, H=4 → 4 DPEs (128→512, GELU absorbed by ACAM)
→ [DPE: W_FFN2 | ACAM=log]     V=1, H=1 → 1 DPE (512→128)
→ [CLB + SRAM: Residual Add]
→ [CLB + DSP: LayerNorm]
```

## 4. ACAM Programming Summary

The same DPE hardware serves 5 different functions by reprogramming the ACAM:

| ACAM Mode | Where Used | What It Computes | What It Replaces |
|-----------|-----------|-----------------|-----------------|
| **log** | Q/K/V/O/FFN2 projections | log(VMM output) | Separate log conversion stage |
| **GELU** | FFN1 activation | GELU(VMM output) | CLB GELU LUT (~16 CLBs) |
| **exp** (single identity) | Softmax, DMMul_2 | exp(input) | CLB exp LUT or DSP |
| **exp** (dual identity) | DMMul_1 QK^T | exp(2 × 64 inputs) | 2× CLB/DSP exp operations |
| **log** (identity) | DMMul_2 attn→log | log(attn weights) | CLB log LUT |

## 5. DPE Resource Count (per transformer block)

| Component | DPEs | ACAM Mode | Crossbar Weights |
|-----------|------|-----------|-----------------|
| Q projection | 1 | log | W_Q (learned) |
| K projection | 1 | log | W_K (learned) |
| V projection | 1 | log | W_V (learned) |
| O projection | 1 | log | W_O (learned) |
| FFN1 | 4 | GELU | W_FFN1 (learned, H=4) |
| FFN2 | 1 | log | W_FFN2 (learned) |
| Head 0: QK^T score_exp | 1 | exp | **Dual identity** (2×64) |
| Head 0: softmax_exp | 1 | exp | Single identity (128) |
| Head 0: wsum_log | 1 | log | Single identity (128) |
| Head 0: wsum_exp | 1 | exp | Single identity (128) |
| Head 1: (same as head 0) | 4 | mixed | identity blocks |
| **Block total** | **17** | | |

**Full BERT-Tiny (2 blocks)**: 2 × 17 = **34 DPEs**

## 6. Energy Results (BERT-Tiny, N=128, d_head=64, dsp_pj=1.2)

| Architecture | Proj+FFN | DIMM | Total | vs Azure-Lily |
|---|---|---|---|---|
| NL-DPE dual-identity (1024×128) | 122K pJ | 4,895K pJ | **5,365K pJ** | **2.97×** |
| NL-DPE single-identity (1024×128) | 122K pJ | 6,634K pJ | 7,104K pJ | 2.24× |
| NL-DPE best C=64 config (128×64) | 259K pJ | 6,935K pJ | 7,541K pJ | 2.11× |
| Azure-Lily (512×128) | 5,571K pJ | 10,066K pJ | **15,937K pJ** | 1.00× |

**Key finding**: Dual-identity on C=128 (2.97×) beats the best C=64 config (2.11×).
Smaller columns cannot use dual-packing (2×64=128 > C=64), making C=128 strictly superior.

**This means one DPE config (512×128 or 1024×128) is optimal for BOTH CNN and transformer.**
No workload-specific config selection needed.

## 7. Scaling with Sequence Length

| N | NL-DPE dual-identity vs Azure-Lily |
|---|---|
| 128 | **2.97×** |
| 256 | **2.51×** |
| 512 | **2.27×** |
| 1024 | **2.14×** |

Advantage decreases at larger N because DIMM (O(N²)) dominates more, but NL-DPE
consistently wins 2-3× across all practical sequence lengths.
