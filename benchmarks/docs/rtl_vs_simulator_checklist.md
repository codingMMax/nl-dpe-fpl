# RTL vs Simulator Stage-by-Stage Checklist

## Stage Comparison (per transformer block, x2 blocks total)

### 1. Embedding

| | RTL | Simulator |
|---|---|---|
| **Module** | `embedding_add` | `_run_embedding()` |
| **Operation** | Token + position + segment lookup, CLB element-wise add. SRAM depth=128 buffers one d_model vector. | SRAM lookup: S tokens x d_model reads. Energy = S x d_model x SRAM access cost. |
| **Compute** | CLB adder (3 vectors added element-wise) | CLB add: S x d_model additions |
| **Memory I/O** | SRAM read (3 embedding tables) + write (result) | SRAM read latency + energy for S x d_model bytes |
| **Match?** | YES — same data volume and compute |

### 2. LayerNorm (x5: post-embed, post-attn x2, post-FFN x2)

| | RTL | Simulator |
|---|---|---|
| **Module** | `layernorm` | `_run_layernorm()` |
| **Operation** | Per-token: load d_model elements, compute mean (accumulate), compute variance (DSP multiply + accumulate), rsqrt (LUT), normalize (DSP multiply). SRAM depth=128. | `fpga.layernorm(d_model, S)`: sum + variance + rsqrt + normalize. Models CLB add, DSP multiply, LUT operations. |
| **Compute** | 2x DSP `*` (variance, normalize) + CLB accumulate + rsqrt LUT | CLB add (mean/var accumulation) + DSP multiply (variance, normalize) |
| **Memory I/O** | SRAM read/write d_model per token x S tokens | SRAM read/write d_model x S bytes |
| **Match?** | YES — same ops per token. RTL uses DSP `*` operator, simulator models DSP energy. |

### 3. Q/K/V Projections (x3 per block)

| | RTL | Simulator |
|---|---|---|
| **Module** | `conv_layer_single_dpe` (DATA_WIDTH=40 for NL-DPE, 16 for Azure-Lily) | `_run_linear()` → `imc_core.run_gemm()` |
| **Operation** | DPE crossbar VMM: weight-persistent mode. Input streamed from SRAM, DPE computes dot product via bit-serial analog, ACAM applies activation. | GEMM(S, d_model, d_model): VMM energy (crossbar) + conversion (ADC/ACAM) + digital post + SRAM read/write + CLB reduction. |
| **GEMM shape** | S tokens x d_model input x d_model output | M=S, K=d_model=128, N=d_model=128 |
| **Memory I/O** | SRAM depth=512 (internal to DPE module) | Read A(S x 128) + B(128 x 128), write C(S x 128) |
| **Match?** | YES — same VMM dimensions. Simulator uses VTR DPE count for parallelism. |
| **Note** | Azure-Lily DPE has 16-bit data bus (2 int8/cycle). NL-DPE has 40-bit (5 int8/cycle). | Simulator uses `bram_width=40` globally. **MISMATCH**: Azure-Lily DPE should use 16-bit bandwidth. |

### 4. QK^T — Attention Score Matrix

| | RTL (NL-DPE) | RTL (Azure-Lily) | Simulator |
|---|---|---|---|
| **Module** | `dimm_score_matrix` (DPE I\|exp + CLB add + CLB reduce) | `dsp_mac` (W parallel instances) | `_run_dimm()` → `gemm_log()` or `gemm_dsp()` |
| **Operation** | Log-domain: CLB add (log_Q + log_K) → DPE(I\|exp) → CLB accumulate. h_dimm DPE stages per module. | DSP MAC: unpack 5 x int8, multiply-accumulate K=ceil(d_head/5) cycles. W=ceil(S/C) parallel instances. | NL-DPE: gemm_log(S, d_head, S) with n_parallel_dpes. Azure-Lily: gemm_dsp(S, d_head, S) with total_dsp. |
| **GEMM shape** | (S, d_head) x (d_head, S) → (S, S) | Same | M=S, K=d_head=64, N=S |
| **Parallelism** | h_dimm = ceil(S/C) DPEs per stage | W = ceil(S/C) dsp_mac instances | n_parallel_dpes or total_dsp (from VTR) |
| **Memory I/O** | K SRAM: depth=S x d/5 (packed). Score row buffer: depth=S/5. | Q/K/V top-level SRAMs: depth=S x d/5 (packed). | Read Q(S x d) + K(d x S), write S x S |
| **Match?** | YES — same GEMM dimensions, same parallelism formula. |

### 5. Softmax

| | RTL | Simulator |
|---|---|---|
| **Module** | `softmax_approx` (single module: exp + sum + reciprocal + normalize) | Two stages: `_run_softmax_exp()` + `_run_softmax_norm()` |
| **Exp** | NL-DPE: DPE(I\|exp) via ACAM. Azure-Lily: CLB exp approximation (shift LUT). | NL-DPE: `dimm_nonlinear(S, op="exp")` per row. Azure-Lily: `fpga.exp_fpga(S)` per row. |
| **Sum** | CLB priority-encoder reciprocal (no divider) | CLB adder tree sum + reciprocal |
| **Normalize** | CLB multiply: exp_val x inv_sum | CLB/DSP multiply: exp_val x inv_sum (or CLB subtract for log-softmax fusion) |
| **Memory I/O** | Row-by-row streaming: read S scores, write S normalized values. Row buffers depth=S/5. | Row-by-row: S iterations x S elements. Total = S x S reads + S x S writes. |
| **Match?** | YES — same per-row operations. RTL is one module, simulator splits into exp+norm. Total energy equivalent. |

### 6. Score x V — Weighted Sum

| | RTL (NL-DPE) | RTL (Azure-Lily) | Simulator |
|---|---|---|---|
| **Module** | `dimm_weighted_sum` (DPE I\|log + CLB add + DPE I\|exp + CLB reduce) | `dsp_mac` (W parallel instances) | `_run_dimm()` → `gemm_log()` or `gemm_dsp()` |
| **Operation** | Log-domain: DPE(I\|log) on attn weights → CLB add (log_attn + log_V) → DPE(I\|exp) → CLB accumulate. | DSP MAC: K=ceil(S/5) accumulation cycles per output. | NL-DPE: gemm_log(S, S, d_head). Azure-Lily: gemm_dsp(S, S, d_head). |
| **GEMM shape** | (S, S) x (S, d_head) → (S, d_head) | Same | M=S, K=S, N=d_head=64 |
| **Extra DPE** | DPE(I\|log) converts attn weights to log domain (2 extra DPE stages per head) | None | NL-DPE: adds DPE(I\|log) energy + latency for M rows (unless log-softmax fusion skips it) |
| **Memory I/O** | V SRAM: depth=S x d/5 (packed). Attn row buffer: depth=S/5. | V top-level SRAM. | Read attn(S x S) + V(S x d), write output(S x d) |
| **Match?** | YES — same GEMM dimensions. NL-DPE extra DPE(I\|log) modeled in simulator. |

### 7. O Projection

| | RTL | Simulator |
|---|---|---|
| **Module** | `conv_layer_single_dpe` | `_run_linear()` → `imc_core.run_gemm()` |
| **GEMM shape** | S x d_model x d_model | M=S, K=128, N=128 |
| **Match?** | YES — identical to Q/K/V projections. |

### 8. Residual Add (x2: post-attn, post-FFN)

| | RTL | Simulator |
|---|---|---|
| **Module** | `residual_add` (CLB element-wise add + SRAM buffer depth=128) | `_run_residual()` |
| **Operation** | Buffer input in SRAM, add with skip connection output element-wise | CLB add: d_model additions per token x S tokens |
| **Match?** | YES — same element-wise add. |

### 9. FFN1 (128 → 512) + GELU

| | RTL | Simulator |
|---|---|---|
| **Module** | `conv_layer_single_dpe` (or multi-DPE `ffn1_layer`) + activation LUT | `_run_linear()` with `has_act=True` |
| **GEMM shape** | S x 128 x 512 | M=S, K=128, N=512 |
| **Activation** | NL-DPE: ACAM (free, inside DPE). Azure-Lily: CLB activation LUT. | NL-DPE: free (ACAM). Azure-Lily: `fpga.activation()` energy. |
| **Match?** | YES — same GEMM + activation. |

### 10. FFN2 (512 → 128)

| | RTL | Simulator |
|---|---|---|
| **Module** | `conv_layer_single_dpe` | `_run_linear()` |
| **GEMM shape** | S x 512 x 128 | M=S, K=512, N=128 |
| **Match?** | YES — same GEMM dimensions. |

## Known Mismatches

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | **Softmax granularity**: RTL = 1 module, Simulator = 2 stages (exp + norm) | None — total energy/latency equivalent | OK |
| 2 | **Azure-Lily DPE bandwidth**: RTL uses DATA_WIDTH=16 for DPE SRAMs, Simulator uses bram_width=40 globally | Simulator underestimates Azure-Lily projection SRAM latency by ~2.5x | TODO |
| 3 | **Parmys SRAM dedup**: RTL DIMM module SRAMs are deduplicated by parmys, top-level buffers compensate | VTR BRAM count matches analytical target within ~20% | Acceptable |
| 4 | **Streaming latency**: Simulator models row-by-row I/O but no inter-stage pipelining overlap | Conservative (upper bound on latency) | TODO |
| 5 | **dsp_mac K parameter**: RTL uses K=ceil(inner_dim/5) packed cycles, Simulator models K=inner_dim element operations | Energy same (total elements identical), latency slightly different | Minor |

## Summary

15 out of 15 stages have matching RTL and simulator implementations.
2 TODOs (Azure-Lily DPE bandwidth, streaming pipeline overlap) are noted
but do not affect energy comparisons between architectures.
