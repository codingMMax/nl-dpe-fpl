# BERT-Tiny Mapping Strategy: 4 FPGA Architectures

## Model Parameters

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| d_head | 64 |
| d_ff | 512 |
| num_heads | 2 |
| num_blocks | 2 |
| seq_len (N) | 128 |
| vocab_size | 30522 |
| max_pos | 512 |

## FPGA Resources (150×150 grid, VTR-verified)

| | Baseline | Proposed (1024×128) | AL-like (1024×256) | Azure-Lily (512×128) |
|---|---|---|---|---|
| DPEs available | 0 | 294 | 90 | 261 |
| DSPs available | 333 | 222 | 222 | 333 |
| BRAMs available | 740 | 518 | 444 | 740 |
| CLBs available | 19,092 | 13,806 | 16,528 | 11,262 |
| DPE tile | — | 3×7 | 5×8 | 6×5 |

### VTR-Verified Used Resources (BERT-Tiny, SDC-constrained)

| | Baseline | Proposed | AL-like | Azure-Lily |
|---|---|---|---|---|
| DPEs used | 0 | **274** (93%) | **78** (87%) | **252** (97%) |
| DSPs used | 0 | 4 (2%) | 4 (2%) | **333** (100%) |
| BRAMs used | 0 | 172 (33%) | 172 (39%) | 16 (2%) |
| CLBs used | 100 | 453 (3%) | 441 (3%) | 193 (2%) |
| Fmax (MHz) | 368.1 | **133.4** | **139.7** | **127.5** |

### Memory Bandwidth
- NL-DPE (Proposed/AL-like): **40-bit** BRAM width, each DPE has dedicated BRAM
- Azure-Lily: **16-bit** BRAM width, each DPE/DSP has dedicated BRAM
- Baseline: **16-bit** BRAM width

---

## Architecture 1: Baseline FPGA (No DPE)

### Compute Resources
- **DSP MACs**: 333 available, e_dsp = 1.2 pJ/MAC
- **CLBs**: 19,092 for softmax, LayerNorm, residual, control
- **BRAMs**: 740 for weight storage and activation buffers
- **No DPE hard blocks**

### Layer Mapping

| Layer | Operation | Compute Unit | Notes |
|-------|-----------|-------------|-------|
| Embedding | Lookup + Add | CLB + BRAM | Token + position + segment |
| LayerNorm | Mean + Var + Scale | CLB | |
| Q/K/V proj (128→64) | GEMM | **DSP MAC** | 333 DSPs time-shared |
| QK^T (64-dim dot product) | GEMM | **DSP MAC** | 333 DSPs time-shared |
| Softmax | exp + sum + recip + mul | **CLB + DSP** | CLB exp LUT, DSP normalize |
| S×V (128-dim dot product) | GEMM | **DSP MAC** | 333 DSPs time-shared |
| O proj (128→128) | GEMM | **DSP MAC** | 333 DSPs time-shared |
| Residual Add | Element-wise add | CLB | |
| FFN1 (128→512) | GEMM | **DSP MAC** | 333 DSPs time-shared |
| FFN2 (512→128) | GEMM | **DSP MAC** | 333 DSPs time-shared |

### Parallelism
- All 333 DSPs work on **one GEMM layer at a time** (time-multiplexed)
- Layers execute sequentially: Q → K → V → QK^T → Softmax → S×V → O → FFN1 → FFN2
- Both transformer blocks are sequential
- Both attention heads are sequential
- No pipeline parallelism across layers

### DPE Usage: **0 / 0** (no DPEs exist)

---

## Architecture 2: Proposed NL-DPE (1024×128)

### Compute Resources
- **DPEs**: 294 available (tile 3×7), ACAM supports exp/log/GELU
- **DSPs**: 222 for LayerNorm, residual
- **CLBs**: 13,806 for DIMM adders, reduction trees, softmax CLB ops
- **BRAMs**: 518 for weight storage

### Layer Mapping

| Layer | Operation | Compute Unit | ACAM Mode | K-identity |
|-------|-----------|-------------|-----------|------------|
| Embedding | Lookup + Add | CLB + BRAM | — | — |
| LayerNorm | Mean + Var + Scale | CLB + DSP | — | — |
| Q proj (128→64) | Weight GEMV | **DPE** (V=1, H=1) | log | — |
| K proj (128→64) | Weight GEMV | **DPE** (V=1, H=1) | log | — |
| V proj (128→64) | Weight GEMV | **DPE** (V=1, H=1) | log | — |
| QK^T score | Log-domain DIMM | **DPE** (I^K\|exp) + CLB add | exp | **K=2** (dual) |
| Softmax exp | Element-wise | **DPE** (I\|exp) | exp | K=1 |
| Softmax norm | Log-softmax fusion | **DPE** (I\|log) + CLB sub | log | K=1 |
| S×V wsum_log | Log conversion | **DPE** (I\|log) | log | K=1 (skipped w/ fusion) |
| S×V wsum_exp | Exp + reduce | **DPE** (I\|exp) + CLB reduce | exp | K=1 |
| O proj (128→128) | Weight GEMV | **DPE** (V=1, H=1) | log | — |
| Residual Add | Element-wise | CLB | — | — |
| FFN1 (128→512) | Weight GEMV | **DPE** (V=1, H=4) | GELU | — |
| FFN2 (512→128) | Weight GEMV | **DPE** (V=1, H=1) | log | — |

### K-Identity Details (C=128, d_head=64, N=128)
- **QK^T**: K = floor(128/64) = **2** → dual-identity, 2 elements per DPE pass
- **S×V (exp/log)**: K = floor(128/128) = **1** → single identity (N fills C)
- **Softmax**: K = floor(128/128) = **1** → single identity

### DIMM Parallelism (W=16 lanes)
- 4 DIMM stages per head × W=16 lanes × 2 heads × 2 blocks = **256 DPEs**
- QK^T: 16 lanes × K=2 = **32 elements/cycle**, 512 cycles for N²=16384
- S×V: 16 lanes × K=1 = **16 elements/cycle**, 512 cycles for N×d=8192

### DPE Allocation (per block)

| Component | DPEs | Notes |
|-----------|------|-------|
| Q projection | 1 | ACAM=log, weight-persistent |
| K projection | 1 | ACAM=log, weight-persistent |
| V projection | 1 | ACAM=log, weight-persistent |
| O projection | 1 | ACAM=log, weight-persistent |
| FFN1 | 4 | ACAM=GELU, H=4 tiles |
| FFN2 | 1 | ACAM=log, weight-persistent |
| DIMM head 0 (4 stages × W=16) | 64 | Identity crossbar, ACAM=exp/log |
| DIMM head 1 (4 stages × W=16) | 64 | Identity crossbar, ACAM=exp/log |
| **Block subtotal** | **137** | |

**Total (2 blocks)**: 2 × 137 = **274 DPEs** (93% of 294 available)

---

## Architecture 3: AL-like NL-DPE (1024×256)

### Differences from Proposed
- Wider crossbar (C=256) → higher K-identity, fewer FFN tiles
- Fewer available DPEs (90) → smaller W

### K-Identity Details (C=256, d_head=64, N=128)
- **QK^T**: K = floor(256/64) = **4** → quad-identity, 4 elements per DPE pass
- **S×V (exp/log)**: K = floor(256/128) = **2** → dual-identity
- **Softmax**: K = floor(256/128) = **2** → dual-identity

### DPE Allocation (per block)

| Component | DPEs | Notes |
|-----------|------|-------|
| Q/K/V projections | 3 | V=1, H=1 each |
| O projection | 1 | V=1, H=1 |
| FFN1 | 2 | H=ceil(512/256)=2 |
| FFN2 | 1 | V=1, H=1 |
| DIMM head 0 (4 stages × W=4) | 16 | K=4/2 identity |
| DIMM head 1 (4 stages × W=4) | 16 | K=4/2 identity |
| **Block subtotal** | **39** | |

**Total (2 blocks)**: 2 × 39 = **78 DPEs** (87% of 90 available)

### Throughput
- QK^T: 4 lanes × K=4 = **16 elements/cycle**, 1024 cycles
- S×V: 4 lanes × K=2 = **8 elements/cycle**, 1024 cycles

---

## Architecture 4: Azure-Lily (512×128)

### Compute Resources
- **DPEs**: 261 available (tile 6×5), ADC-based (NO ACAM analog nonlinear)
- **DSPs**: **333** available (separate columns from DPEs, startx=14 repeatx=16)
- **CLBs**: 11,262 for softmax, control, LayerNorm
- **BRAMs**: 740

### Layer Mapping

| Layer | Operation | Compute Unit | Notes |
|-------|-----------|-------------|-------|
| Q/K/V proj | Weight GEMV | **DPE** (V=1, H=1) | Digital accum (no ACAM) |
| QK^T | GEMM | **DSP MAC** | 333 DSPs in parallel |
| Softmax | exp + norm | **CLB + DSP** | CLB exp LUT + DSP multiply for normalization |
| S×V | GEMM | **DSP MAC** | 333 DSPs in parallel |
| O proj | Weight GEMV | **DPE** (V=1, H=1) | |
| FFN1 (128→512) | Weight GEMV | **DPE** (V=1, H=4) | |
| FFN2 (512→128) | Weight GEMV | **DPE** (V=1, H=1) | |

### Resource Allocation (per block)

| Component | DPEs | DSPs | Notes |
|-----------|------|------|-------|
| Q/K/V/O projections | 4 | 0 | ADC-based, weight-persistent |
| FFN1 | 4 | 0 | H=4 tiles |
| FFN2 | 1 | 0 | |
| DIMM (QK^T + S×V) | **0** | **all 333** | DSP MACs, time-shared across layers |
| Softmax normalization | 0 | incl. | DSP multiply for normalization |
| **Block subtotal** | **9** | | |

**Total (2 blocks)**:
- DPEs: 2 × 9 = **18** active + **234** extra replicas = **252 DPEs** (97% of 261)
- DSPs: **333** (100% utilization, time-shared across DIMM layers)

### Parallelism Strategy
- **Projections + FFN**: 14 pipeline replicas (252/18 = 14) — all DPEs utilized
- **DIMM**: All 333 DSPs work on one GEMM at a time, processing 333 MACs per cycle
- **Key limitation**: DPEs cannot perform DIMM (no ACAM analog nonlinear) —
  234 of 252 DPEs are padding replicas for realistic VTR Fmax, not active DIMM compute
---

## Summary: Simulator Parameters per Architecture

| Parameter | Baseline | Proposed | AL-like | Azure-Lily |
|-----------|----------|----------|---------|------------|
| Config JSON | baseline.json | nl_dpe.json | nl_dpe.json | azure_lily.json |
| analog_nonlinear | false | true | true | false |
| log_softmax_fusion | false | true | true | false |
| Crossbar (R×C) | — | 1024×128 | 1024×256 | 512×128 |
| BRAM width (bits) | 16 | 40 | 40 | 16 |
| total_dsp (available) | 333 | 222 | 222 | 333 |
| total_clb (available) | 19,092 | 13,806 | 16,528 | 11,262 |
| total_mem (available) | 740 | 518 | 444 | 740 |
| freq (VTR Fmax, MHz) | 368.1 | 133.4 | 139.7 | 127.5 |
| Linear layers route to | gemm_dsp | DPE VMM | DPE VMM | DPE VMM |
| DIMM layers route to | gemm_dsp | gemm_log (K-identity) | gemm_log (K-identity) | gemm_dsp (333 DSPs) |
| Softmax | CLB+DSP | DPE+CLB (fusion) | DPE+CLB (fusion) | CLB+DSP |
| DIMM W (parallel lanes) | N/A | 16 | 4 | N/A (333 DSPs) |
| K-identity (QK^T) | N/A | K=2 | K=4 | N/A |
| K-identity (S×V) | N/A | K=1 | K=2 | N/A |
| Memory model | per-DSP BRAM | per-DPE BRAM | per-DPE BRAM | per-DPE/DSP BRAM |
| K-identity (QK^T) | N/A | K=2 | K=4 | N/A |
| K-identity (S×V) | N/A | K=1 | K=2 | N/A |
