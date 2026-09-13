# BERT-Tiny NL-DPE Mapping Flow — Paper Figure Prompt

## Figure Description

Generate a publication-quality dataflow diagram showing how BERT-Tiny maps onto the NL-DPE enhanced FPGA. This figure should appear in the Methodology or Results section of a top-tier architecture conference paper (ISCA/MICRO/FCCM).

## Model: BERT-Tiny (2 Layers, 2 Heads, d=128, d_ff=512)

## Architecture: NL-DPE FPGA

Three types of compute resources:
- **DPE** (green): ReRAM crossbar + ACAM — handles GEMV (weight-persistent) and nonlinear ops (log, exp, activation)
- **CLB** (blue): FPGA logic — handles addition, reduction, reciprocal, control
- **DSP** (orange): Multiply blocks — used only for LayerNorm normalization
- **SRAM/BRAM** (purple): On-chip memory — skip buffers, embedding tables, intermediate storage

## Complete Dataflow (Top to Bottom)

### Stage 1: Embedding (CLB + SRAM)

```
Token IDs ──→ [SRAM: Token Embed Table] ──┐
Position IDs → [SRAM: Position Embed Table]├→ [CLB: 3-way Add] → Embedded Vectors (S × 128)
Segment IDs ─→ [SRAM: Segment Embed Table]┘
```

→ **LayerNorm** (CLB reduction tree + DSP multiply for rsqrt normalization)

### Stage 2: Transformer Block (×2 identical blocks)

#### 2a. Q/K/V Projections (DPE, weight-persistent, ACAM=log)

```
Input (S × 128)
  ├→ [DPE: W_Q | ACAM=log] → log(Q)  (S × 128)
  ├→ [DPE: W_K | ACAM=log] → log(K)  (S × 128)
  └→ [DPE: W_V | ACAM=log] → log(V)  (S × 128)
```

- Each projection: V=1, H=1 → **1 DPE** (K=128 ≤ R=1024, N=128 ≤ C=128)
- ACAM configured as **log function** → outputs are in log domain
- **Key advantage**: projections output log(Q), log(K), log(V) directly — no separate log conversion needed for DIMM

#### 2b. Multi-Head Attention (×2 heads in parallel, d_head=64)

Each head slices log(Q), log(K), log(V) into d_head=64 dimensions.

**DMMul_1 — Score Matrix (Q·K^T):**
```
log(Q_h)[i][m] + log(K_h)[j][m]  →  [CLB: element-wise add]
                                  →  [DPE: I|exp (ACAM=exp)]  ← identity crossbar, ACAM computes exp
                                  →  [CLB: reduction sum]
                                  →  S[i][j] = Σ_m Q[i][m]·K[j][m]
```
- **1 DPE** (I|exp): identity-weight crossbar + ACAM configured for exp
- CLB adders replace multiplication (log domain: log(a) + log(b) = log(a×b))

**Softmax:**
```
S[i][j]  →  [DPE: I|exp (ACAM=exp)]  → exp(S[i][j])
         →  [CLB: reduction sum]       → Σ_j exp(S[i][j])
         →  [CLB: reciprocal LUT]      → 1/sum
         →  [CLB: multiply]            → attn[i][j] = exp(S[i][j]) / sum
```
- **1 DPE** (I|exp): computes element-wise exp
- CLB handles sum, reciprocal (priority-encoder LUT), and normalization multiply

**DMMul_2 — Weighted Sum (attn·V):**
```
attn[i][j]        →  [DPE: I|log (ACAM=log)]  → log(attn[i][j])
log(attn) + log(V) → [CLB: element-wise add]
                    → [DPE: I|exp (ACAM=exp)]
                    → [CLB: reduction sum]
                    → O[i][m] = Σ_j attn[i][j]·V[j][m]
```
- **1 DPE** (I|log): converts attention weights to log domain
- **1 DPE** (I|exp): converts back from log domain after addition

**Total per head:** 4 DIMM DPEs (score_exp + softmax_exp + wsum_log + wsum_exp)

#### 2c. Output Projection + Residual + LayerNorm

```
[Head 0 output | Head 1 output] → Concat (S × 128)
  → [DPE: W_O | ACAM=log] → O_proj output    (1 DPE)
  → [CLB + SRAM: Residual Add] (add input from before attention)
  → [CLB + DSP: LayerNorm]
```

#### 2d. Feed-Forward Network (FFN)

```
  → [DPE: W_FFN1 | ACAM=GELU] → FFN1 output   (V=1, H=4 → 4 DPEs)
    ↑ GELU activation absorbed by ACAM (free, no CLB needed)
  → [DPE: W_FFN2 | ACAM=log] → FFN2 output     (V=1, H=1 → 1 DPE)
  → [CLB + SRAM: Residual Add]
  → [CLB + DSP: LayerNorm]
```

- FFN1: ACAM configured for GELU activation → **zero CLB activation cost**
- This is a key NL-DPE advantage: ACAM is reprogrammable (log, exp, GELU, ReLU)

### DPE Resource Summary (per transformer block)

| Component | DPEs | ACAM Mode | Notes |
|-----------|------|-----------|-------|
| Q projection | 1 | log | Weight-persistent GEMV |
| K projection | 1 | log | Weight-persistent GEMV |
| V projection | 1 | log | Weight-persistent GEMV |
| O projection | 1 | log | Weight-persistent GEMV |
| FFN1 | 4 | GELU | 4 horizontal tiles (N=512, C=128) |
| FFN2 | 1 | log | Weight-persistent GEMV |
| Head 0: score_exp | 1 | exp | Identity crossbar |
| Head 0: softmax_exp | 1 | exp | Identity crossbar |
| Head 0: wsum_log | 1 | log | Identity crossbar |
| Head 0: wsum_exp | 1 | exp | Identity crossbar |
| Head 1: score_exp | 1 | exp | Identity crossbar |
| Head 1: softmax_exp | 1 | exp | Identity crossbar |
| Head 1: wsum_log | 1 | log | Identity crossbar |
| Head 1: wsum_exp | 1 | exp | Identity crossbar |
| **Block total** | **17** | | |

**Full model: 2 blocks × 17 = 34 DPEs** (for 1024×128 config)

### CLB Resource Summary

| Component | Per block | Total (2 blocks) | Notes |
|-----------|-----------|-------------------|-------|
| DIMM adders (log add) | ~8 instances | 16 | Element-wise add in log domain |
| DIMM accumulators | ~8 instances | 16 | Reduction sum trees |
| Softmax sum+recip+mul | 2 (per head) | 8 | CLB logic, no DSP |
| Residual add | 2 | 4 | + SRAM skip buffer |
| LayerNorm | 2 | 4 + 1 (embed) = 5 | CLB reduction + DSP multiply |
| Embedding | — | 1 | 3 SRAM lookups + CLB add |

### ACAM Modes Used (Key NL-DPE Feature)

| ACAM Mode | Where Used | What It Replaces |
|-----------|-----------|-----------------|
| **log** | Q/K/V/O/FFN2 projections | Separate log conversion stage |
| **exp** | DIMM score, softmax, weighted sum | CLB exp LUT (expensive) |
| **GELU** | FFN1 activation | CLB GELU LUT |

**The ACAM is reprogrammed per-layer** — same DPE hardware, different function.
This is the central architectural innovation: one DPE hard block serves as GEMV accelerator,
nonlinear function unit, AND activation layer depending on ACAM programming.

## Visual Style Notes

- Use the same color coding throughout: DPE=green, CLB=blue, DSP=orange, SRAM=purple
- Show the two attention heads side-by-side (parallel execution)
- Highlight the ACAM mode label on each DPE block (log/exp/GELU)
- The log-domain dataflow (add replacing multiply) should be visually prominent
- Show skip connections for residual as dashed backward arrows
- Two transformer blocks can be shown as one block with "×2" annotation
