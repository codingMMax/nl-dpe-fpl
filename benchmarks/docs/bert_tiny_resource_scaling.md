# BERT-Tiny Resource Scaling Design Document

## Overview

This document defines how DPE, DSP, and BRAM resources scale with input sequence length S for the BERT-Tiny seq_len sweep experiment. Both NL-DPE and Azure-Lily follow the **same computation-driven scaling logic** — the workload determines parallelism, each architecture implements it with its respective compute primitive.

## BERT-Tiny Architecture

| Parameter | Value |
|-----------|-------|
| Transformer layers | 2 |
| Attention heads | 2 |
| d_model (hidden dim) | 128 |
| d_head (per-head dim) | 64 |
| d_ff (FFN intermediate) | 512 |
| Sequence length S | 128, 256, 512, 1024, 2048, 4096 |

## Operations Per Transformer Block

| Operation | GEMM shape | Weight dims | Scales with S? |
|-----------|-----------|-------------|---------------|
| Q/K/V projection | (S, 128, 128) | 128×128 | S is batch dim only |
| O projection | (S, 128, 128) | 128×128 | S is batch dim only |
| FFN1 + GELU | (S, 128, 512) | 128×512 | S is batch dim only |
| FFN2 | (S, 512, 128) | 512×128 | S is batch dim only |
| QK^T (per head) | (S, 64, **S**) | — | **Output is S×S** |
| Softmax (per head) | S rows × S cols | — | **S² elements** |
| Score×V (per head) | (S, **S**, 64) | — | **Inner dim is S** |
| LayerNorm (×2+1) | 128 elements × S tokens | — | S is token count only |
| Residual Add (×2) | 128 elements × S tokens | — | S is token count only |

**Key insight**: Only DIMM operations (QK^T, Softmax, Score×V) have S in the matrix dimensions. Projections/FFN/LayerNorm process S tokens through **fixed-size** hardware.

## Why Projections Don't Scale

Pipeline throughput analysis at Fmax ≈ 100 MHz:

| Stage | Per-token time | Total for S tokens |
|-------|---------------|-------------------|
| Projection (1 DPE) | t_steady ≈ 260 ns (BRAM limited) | S × 260 ns |
| DIMM QK^T (W units) | S²/W × 64 MACs | S × 128 × 64 / freq |

With W = ceil(S/128), both scale as O(S). The ratio is constant:

```
DIMM_time / Proj_time = (S × 8192) / (S × 384) = 21.3×
```

**DIMM is always 21× slower** — projection never bottlenecks. Scaling projection DPEs provides zero throughput benefit.

## DIMM Parallelism: W = ceil(S / C)

The DIMM output has S² elements (QK^T) or S×64 elements (Score×V). With W parallel compute units, W elements are processed simultaneously:

```
W = h_dimm = ceil(S / C)
```

Where C = crossbar column count (128 for both proposed and Azure-Lily).

**Both architectures use the same W** — the difference is the compute unit type:
- **NL-DPE**: W DPE instances per DIMM stage (analog ACAM for exp/log)
- **Azure-Lily**: W DSP MAC instances per DIMM stage (digital multiply-accumulate)

## Resource Count Formulas

### DPE

**DPE tiling formula** for a linear layer with weight matrix (K×N):

```
V = ceil(K / R)    — vertical tiles (input dimension / crossbar rows)
H = ceil(N / C)    — horizontal tiles (output dimension / crossbar cols)
DPEs per layer = V × H
```

S tokens stream through the same V×H DPE array sequentially.

**Projection + FFN DPE count (all architectures):**

Azure-Lily (R=512, C=128) example:

```
Q/K/V/O projections: weight = 128×128
  V = ceil(128/512) = 1, H = ceil(128/128) = 1 → 1 DPE each, × 4 = 4

FFN1: weight = 128×512
  V = ceil(128/512) = 1, H = ceil(512/128) = 4 → 4 DPEs

FFN2: weight = 512×128
  V = ceil(512/512) = 1, H = ceil(128/128) = 1 → 1 DPE

Per block = 4 + 4 + 1 = 9, × 2 blocks = 18 DPEs
```

All architectures:

```
  proposed  (R=1024, C=128): (4×1×1 + 1×4 + 1×1) × 2 = 18
  al_like   (R=1024, C=256): (4×1×1 + 1×2 + 1×1) × 2 = 14
  azurelily (R=512,  C=128): (4×1×1 + 1×4 + 1×1) × 2 = 18
```

**DIMM DPE count (NL-DPE only):**

QK^T output is S×S. Each DPE covers C output columns. To compute one full
row of the S×S score matrix requires `h_dimm = ceil(S / C)` DPEs in parallel.
4 DIMM stages (score, softmax_exp, softmax_norm, weighted_sum) each need h_dimm DPEs.

```
  h_dimm = ceil(S / C)
  dimm_dpes = 4 stages × h_dimm × 2 heads × 2 blocks = 16 × h_dimm

  Total NL-DPE DPEs = functional + 16 × ceil(S / C)
```

**Azure-Lily DPEs = functional only (constant):**

Azure-Lily uses DSPs (not DPEs) for DIMM, so DPE count does not scale with S.

```
  Total Azure-Lily DPEs = 18 (independent of S)
```

### DSP

**LayerNorm DSPs (both architectures, fixed):**

LayerNorm normalizes d_model=128 elements per token. Per token it requires:
1. Mean: sum 128 elements → CLB adder tree (no DSP)
2. Variance: 128 × `(x - mean)²` → **1 multiply DSP** (time-shared across 128 elements)
3. Rsqrt(variance) → CLB LUT (no DSP)
4. Normalize: 128 × `(x - mean) * rsqrt` → **1 multiply DSP** (time-shared)

Each LN module has 2 multiply primitives, time-shared across 128 elements and S tokens.
More tokens (larger S) = more cycles through the same 2 multipliers, not more multipliers.

```
  5 LN modules × 2 multiply primitives = 10 DSPs
    - 1 embedding LN
    - 2 per block × 2 blocks = 4 (post-attention LN + post-FFN LN)
```

Why not scale with S? LayerNorm is never the bottleneck:
```
  LayerNorm latency ≈ S × 128 × (cycles per element)
  DIMM latency      ≈ S × 128 × 64 × (cycles per element)
```
DIMM is ~64× slower. Adding more LN multipliers saves <2% total latency.

**Azure-Lily DIMM DSPs (scales with S):**

Azure-Lily's DIMM uses DSP MAC units instead of DPEs. The same parallelism
formula applies: W = ceil(S / C) parallel `dsp_mac` instances per DIMM stage.
Each `dsp_mac` computes one output column's dot product via the Verilog `*`
operator, which VTR maps to DSP blocks.

```
  W = ceil(S / C) = ceil(S / 128)           — same as NL-DPE's h_dimm
  dimm_dsps = 4 stages × W × 2 heads × 2 blocks = 16 × W

  Total Azure-Lily DSPs = 10 (LN) + 16 × ceil(S / 128) (DIMM)
```

Note: parmys may merge some `dsp_mac` instances (synthesizable `*` is mergeable,
unlike NL-DPE's `dpe` hard block primitive). VTR-reported DSP count may be
lower than the intended formula. We use VTR-reported count as ground truth.

**NL-DPE DSPs (fixed):**

NL-DPE uses ACAM for DIMM nonlinear operations (exp/log), eliminating DSP need.

```
  Total NL-DPE DSPs = 10 (LayerNorm only, ACAM handles DIMM)
```

### BRAM

Driven by intermediate SRAM depths:
- DIMM buffers: depth = S × d_head per submodule (both archs)
- Softmax buffer: depth = S (Azure-Lily clb_softmax)
- Projection SRAMs: depth = 512 (fixed)
- CLB module SRAMs: depth = 128 (fixed)

Actual BRAM count determined by VTR packing.

## Expected Resource Counts

### NL-DPE Proposed (1024×128, C=128)

| S | h_dimm | DPE | DSP | Notes |
|---|--------|-----|-----|-------|
| 128 | 1 | 18 + 16 = **34** | **10** | Baseline |
| 256 | 2 | 18 + 32 = **50** | **10** | |
| 512 | 4 | 18 + 64 = **82** | **10** | |
| 1024 | 8 | 18 + 128 = **146** | **10** | |
| 2048 | 16 | 18 + 256 = **274** | **10** | |
| 4096 | 32 | 18 + 512 = **530** | **10** | |

### NL-DPE AL-Like (1024×256, C=256)

| S | h_dimm | DPE | DSP | Notes |
|---|--------|-----|-----|-------|
| 128 | 1 | 14 + 16 = **30** | **10** | d_head < C, so h_dimm=1 |
| 256 | 1 | 14 + 16 = **30** | **10** | 256/256 = 1 |
| 512 | 2 | 14 + 32 = **46** | **10** | |
| 1024 | 4 | 14 + 64 = **78** | **10** | |
| 2048 | 8 | 14 + 128 = **142** | **10** | |
| 4096 | 16 | 14 + 256 = **270** | **10** | |

### Azure-Lily (512×128, C=128)

| S | W | DPE | DSP | Notes |
|---|---|-----|-----|-------|
| 128 | 1 | **18** | 10 + 16 = **26** | Baseline |
| 256 | 2 | **18** | 10 + 32 = **42** | |
| 512 | 4 | **18** | 10 + 64 = **74** | |
| 1024 | 8 | **18** | 10 + 128 = **138** | |
| 2048 | 16 | **18** | 10 + 256 = **266** | |
| 4096 | 32 | **18** | 10 + 512 = **522** | |

## The Architectural Symmetry

The DIMM uses the **same number of parallel compute units** for both architectures:

```
W = ceil(S / C)    where C = 128 (crossbar column count)
DIMM parallel units = 4 stages × W × 2 heads × 2 blocks = 16 × W
```

### Different Roles, Same Parallelism

Although both architectures use W = ceil(S/C) parallel units per DIMM stage, each unit serves a **different role**:

**NL-DPE DPE (nonlinear function unit):**
- Identity crossbar + ACAM (Analog CAM)
- Role: applies exp() or log() to input elements in analog domain
- DIMM dataflow: CLB add (log Q + log K) → **DPE(I|exp)** → CLB reduce (accumulate)
- Per-element time: ~10 ns (bit-serial pipeline: fill + 7×steady + ACAM)
- The dot product is computed via log-domain add + exp, not MAC

**Azure-Lily DSP (multiply-accumulate unit):**
- Digital multiplier + accumulator
- Role: computes dot product by accumulating K products
- DIMM dataflow: BRAM read → **DSP MAC** (K accumulations) → BRAM write
- Per-element time: K cycles (K=64 for QK^T, K=S for Score×V)
- Direct arithmetic dot product

**Why same W despite different per-unit work?**

Both architectures need to produce the same number of output elements (S² for QK^T). With W parallel units, each unit handles ceil(S²/W) elements. The **per-element cycle count differs** (DPE ~10 cycles vs DSP K cycles), but the simulator models this accurately via different functions (`gemm_log()` vs `gemm_dsp()`). Keeping W the same provides a clean comparison: same parallelism, different area cost per unit.

### Area Comparison Per Compute Unit

| | NL-DPE (DPE) | Azure-Lily (DSP) |
|---|---|---|
| Compute primitive | Analog crossbar + ACAM | Digital multiply-accumulate |
| DIMM role | Nonlinear function (exp/log) | MAC (dot product) |
| Tile size | 3×7 = 21 grid cells | 1×4 = 4 grid cells |
| Area per unit | 21 × 2239 µm² = 47,019 µm² | 4 × 2239 µm² = 8,956 µm² |
| Area ratio | **5.25×** larger | 1× (baseline) |
| Free nonlinearity | Yes (ACAM exp/log) | No (needs CLB for exp/log) |
| Cycle count per element | ~10 (fixed, bit-serial) | K (scales with inner dim) |

**Paper story**: NL-DPE pays 5.25× more area per DIMM compute unit but gets analog nonlinearity (ACAM) for free, eliminating CLB/DSP cost for softmax exp/log. Azure-Lily is area-efficient per unit but needs additional CLB logic for nonlinear operations and takes more cycles per element for large K. The throughput/mm² comparison reveals which tradeoff wins at each sequence length.
