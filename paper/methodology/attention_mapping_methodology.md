# Attention-Head Mapping Methodology: NL-DPE vs Azure-Lily

This document formalizes how a single attention head maps to NL-DPE and Azure-Lily
FPGA architectures. It serves as the reference for:
1. IMC simulator implementation (energy + latency models)
2. RTL design (resource allocation and scheduling)
3. Analytical comparison (NL-DPE vs Azure-Lily vs baseline FPGA)

Generalizes to multi-head attention, BERT-Tiny, BERT-Base.

---

## 1. Hardware Model

### 1.1 Shared Hardware (Identical for Both Architectures)

| Resource | Description |
|----------|-------------|
| DPE crossbar | R×C ReRAM array (e.g., 512×128). Weight-persistent VMM. |
| DSP blocks | Fixed-function multiply-accumulate (e.g., 132 on FPGA) |
| CLB blocks | Configurable logic (LUTs, adders, registers) |
| BRAM blocks | On-chip SRAM (e.g., 36Kb per block) |

### 1.2 The Only Difference: DPE Output Stage

| | NL-DPE | Azure-Lily |
|---|---|---|
| Output stage | **ACAM** (analog CAM) | **ADC** (analog-to-digital) |
| Programmable? | Yes: log, exp, activation, identity | No: linear conversion only |
| Energy | e_digital = 0.171 pJ/col | e_conv = 2.33 pJ/col |
| Implication | Can do nonlinear in-crossbar | Falls back to DSP/CLB for nonlinear |

**Critical**: both have the same crossbar, same DPE count, same DSP/CLB/BRAM.
The ACAM is the sole architectural advantage of NL-DPE.

---

## 2. Attention Head: Workload Definition

Single-head attention: Q, K, V ∈ ℝ^{N×d}, output O ∈ ℝ^{N×d}

```
Q = X_Q × W_Q          (projection, weight-persistent GEMV)
K = X_K × W_K
V = X_V × W_V
S = Q × K^T             (DIMM-1: dynamic input matmul, N²×d MACs)
A = softmax(S)           (element-wise exp + normalize)
O = A × V               (DIMM-2: dynamic input matmul, N×d×N MACs)
```

Parameters: N = sequence length, d = head dimension, R×C = crossbar size.

---

## 3. Stage-by-Stage Mapping

### 3.1 Projections (Q, K, V) — SAME for Both

Both architectures use DPE crossbar for weight-persistent GEMV.

```
Operation:  output[n] = W × input[n]    for n = 0..N-1
Tiling:     V = ceil(d/R), H = ceil(d/C)
DPEs:       V × H per projection, 3 projections
Schedule:   stream N inputs through DPE pipeline, 1 output per pipeline cycle
```

**Resource allocation:**
- DPEs: 3 × V × H (if all 3 projections run in parallel)
  OR V × H (if projections share DPEs sequentially, 3× latency)

**Output domain:**
- NL-DPE: ACAM configured as **log** → output = log(W × input)
- Azure-Lily: ADC → output = W × input (linear)

**Latency (per projection):**
```
t_proj = N × t_pipeline_per_input
t_pipeline_per_input = t_vmm + t_output_stage + t_readout
  NL-DPE:     t_vmm + t_acam + t_readout
  Azure-Lily: t_vmm + t_adc + t_readout
```

**Energy (per projection):**
```
e_proj = N × V × (e_vmm_per_pass + e_output_per_pass) + e_sram
  NL-DPE:     e_vmm = k_bit × e_analoge;  e_output = e_digital × C
  Azure-Lily: e_vmm = k_bit × e_analoge;  e_output = k_bit × e_conv × C
```

### 3.2 DIMM-1: Q·K^T — KEY DIFFERENCE

**Output**: S ∈ ℝ^{N×N}, where S[i][j] = Σ_m Q[i][m] × K[j][m]

#### NL-DPE: Log-Domain DIMM

Inputs are in log domain (from projection ACAM=log).
Replace multiply with add: Q[i][m] × K[j][m] = exp(log_Q[i][m] + log_K[j][m])

```
For each output element S[i][j]:
  1. CLB add:  sum_m = log_Q[i][m] + log_K[j][m]   for m=0..d-1    (d parallel adds)
  2. DPE exp:  exp_m = exp(sum_m)                    for m=0..d-1    (1 DPE pass, d columns)
  3. CLB sum:  S[i][j] = Σ_m exp_m                                   (log2(d) tree)
```

**Resource allocation:**
- DPE instances: **N** (one per output column j)
  Each DPE_j computes S[0][j], S[1][j], ..., S[N-1][j] sequentially over rows
- CLBs: N × d parallel adders (for step 1) + N × log2(d) tree reducers (step 3)

**Latency:**
```
t_dimm1 = N_rows × t_per_row
t_per_row = t_clb_add + t_dpe_exp + t_clb_reduce
  t_clb_add   = 1 cycle (d parallel adders, 1 cycle each)
  t_dpe_exp   = ceil(d/C) × t_dpe_pass  (t_dpe_pass ≈ 10 ns at 1 GHz)
  t_clb_reduce = ceil(log2(d)) cycles

t_dimm1 = N × (1 + ceil(d/C) × t_dpe_pass_cycles + ceil(log2(d)))
```

**Energy:**
```
e_dimm1 = N² × (d × e_clb_add + ceil(d/C) × e_dpe_pass + (d-1) × e_clb_reduce)
        + e_sram_read(N×d + N×d) + e_sram_write(N×N)
```

#### Azure-Lily: DSP MAC

Inputs are in linear domain. Standard matrix multiply on DSPs.

```
For each output element S[i][j]:
  S[i][j] = Σ_m Q[i][m] × K[j][m]    using DSP MAC units
```

**Resource allocation:**
- DSPs: total_dsp (all available, e.g., 132)
- Each DSP computes MAC for one output column

**Latency:**
```
t_dimm1 = ceil(N / total_dsp) × N × ceil(d / DSP_WIDTH) cycles
```

**Energy:**
```
e_dimm1 = N² × d × e_dsp_mac + e_sram_read + e_sram_write
```

### 3.3 Softmax — Different Nonlinear Mapping

**Operation**: A[i][j] = exp(S[i][j]) / Σ_k exp(S[i][k])

#### NL-DPE:

```
Per row i (N rows):
  1. DPE exp:   exp_j = exp(S[i][j])       for j=0..N-1    (ceil(N/C) DPE passes)
  2. CLB sum:   sum_exp = Σ_j exp_j                          (log2(N) tree)
  3. DPE log:   log_sum = log(sum_exp)                        (1 DPE pass, 1 element)
  4. CLB mul:   A[i][j] = exp_j / sum_exp   for j=0..N-1    (N multiplies, or DSP)

Resource: N DPE instances for step 1 (parallel across j), reuse for step 3
Latency: N × (ceil(N/C) × t_dpe_pass + log2(N) + t_dpe_pass + N × t_mul)
Energy: N × (N × e_dpe_pass + (N-1) × e_clb_add + e_dpe_pass + N × e_mul)
```

#### Azure-Lily:

```
Per row i:
  1. CLB exp:   exp_j via LUT                for j=0..N-1
  2. CLB sum:   sum_exp = Σ_j exp_j           (log2(N) tree)
  3. CLB inv:   inv_sum = 1/sum_exp           (reciprocal LUT)
  4. DSP mul:   A[i][j] = exp_j × inv_sum    for j=0..N-1

Resource: CLBs for exp LUT + tree, DSPs for normalize
Latency: N × (t_clb_exp + log2(N) + t_clb_inv + N × t_dsp_mul)
Energy: N × (N × e_clb_exp + (N-1) × e_clb_add + e_clb_inv + N × e_dsp_mul)
```

### 3.4 DIMM-2: Score·V — Same Structure as DIMM-1

**Output**: O ∈ ℝ^{N×d}, where O[i][m] = Σ_j A[i][j] × V[j][m]

#### NL-DPE:

Extra step: convert A (linear domain from softmax) to log domain.

```
Pre-step: DPE log on A[i][j] for all i,j   → log_A
  Resource: N DPE instances (parallel across j per row)
  Latency: N × ceil(N/C) × t_dpe_pass
  Energy: N² × e_dpe_pass

Then same as DIMM-1 with dimensions (N, N, d):
  CLB add(log_A + log_V) → DPE exp → CLB reduce
  Resource: d DPE instances (one per output column m)
  Latency: N × (1 + ceil(N/C) × t_dpe_pass + ceil(log2(N)))
  Energy: N×d × (N × e_clb_add + ceil(N/C) × e_dpe_pass + (N-1) × e_clb_reduce)
```

#### Azure-Lily:

Same as DIMM-1 but on DSPs with dimensions (N, N, d).

```
Latency: ceil(N / total_dsp) × N × ceil(N / DSP_WIDTH) cycles
Energy: N × d × N × e_dsp_mac
```

---

## 4. Total Resource Allocation

### 4.1 NL-DPE

```
Projection DPEs:  3 × V × H  (dedicated, weight-persistent)
DIMM DPEs:        max(N, d)   (reused across DIMM stages, identity weights)
  - DIMM-1:       N instances
  - Softmax exp:  N instances  (same DPEs as DIMM-1)
  - Softmax log:  1 instance
  - DIMM-2 log:   N instances  (same DPEs)
  - DIMM-2 exp:   d instances

Total DPEs = 3×V×H + max(N, d)

CLBs: N×d adders + N×log2(d) reducers + softmax tree + normalize
DSPs: softmax normalize multiply (small)
BRAMs: Q/K/V/S/A/O buffers (N×d + N×N + N×d)
```

### 4.2 Azure-Lily

```
Projection DPEs:  3 × V × H  (same as NL-DPE)
DIMM DSPs:        total_dsp   (all available)
  - (DPEs sit idle during DIMM — can't do nonlinear)

Total DPEs = 3×V×H  (DIMM doesn't use DPEs)
Total DSPs = total_dsp (for DIMM) + some for softmax normalize

CLBs: softmax exp LUT + trees + control
BRAMs: same as NL-DPE
```

### 4.3 Comparison

| Resource | NL-DPE | Azure-Lily |
|----------|--------|------------|
| DPEs (projections) | 3×V×H | 3×V×H (same) |
| DPEs (DIMM) | **max(N, d)** | 0 (idle) |
| DSPs (DIMM) | 0 | **total_dsp** |
| DSPs (softmax) | small | small |
| CLBs | adders + reducers | exp LUT + reducers |

**NL-DPE trades DSP usage for DPE reuse.** The same DPEs used for projections
are repurposed (with identity weights + ACAM) for DIMM stages.

---

## 5. Generalization to Multi-Layer Transformers

### 5.1 BERT-Tiny (2L, 2H, d_model=128, d_head=64, d_ff=512)

Per transformer block:
```
Multi-head attention (2 heads):
  Projections: 3 × GEMV(N, 128, 64) per head = 6 projections
  DIMM: QK^T(N,64,N) + Score·V(N,N,64) per head = 4 DIMM ops
  Softmax: per head

Output projection: GEMV(N, 128, 128)

FFN:
  FC1: GEMV(N, 128, 512) + GELU
    NL-DPE:     ACAM = GELU (if V=1) → free
    Azure-Lily: CLB LUT
  FC2: GEMV(N, 512, 128)

LayerNorm: CLB reduction + DSP scale/shift (SAME for both)
Residual add: CLB add (SAME for both)
```

DPE allocation for NL-DPE:
```
Projections: max across all layers
  Q/K/V: V=ceil(128/R)×H=ceil(64/C) = 1×1 = 1 DPE each (for 512×128)
  O proj: V=1, H=1 = 1 DPE
  FFN1: V=1, H=ceil(512/128)=4 → 4 DPEs
  FFN2: V=ceil(512/512)=1, H=1 = 1 DPE
  Max projection DPEs: 4 (FFN1)

DIMM: max(N, d_head) = max(N, 64)
  For N=128: 128 DPEs for DIMM
  For N=1024: 1024 DPEs for DIMM

Total: 4 + max(N, 64) DPEs
```

### 5.2 BERT-Base (12L, 12H, d_model=768, d_head=64, d_ff=3072)

```
Projections: max is FFN1 → V=ceil(768/512)=2, H=ceil(3072/128)=24 → 48 DPEs
DIMM: max(N, 64) DPEs
Total: 48 + max(N, 64) DPEs
```

---

## 6. Key Equations Summary

### Energy (per attention head)

```
NL-DPE:
  E_proj  = 3 × N × V × (k_bit × e_analoge + e_digital × C) + E_sram
  E_dimm1 = N² × (d × e_clb_add + ceil(d/C) × e_dpe_pass + (d-1) × e_clb_add) + E_sram
  E_soft  = N × (ceil(N/C) × e_dpe_pass + (N-1) × e_clb_add + e_dpe_pass + N × e_mul)
  E_dimm2 = N² × ceil(N/C) × e_dpe_pass  [log conversion]
          + N×d × (N × e_clb_add + ceil(N/C) × e_dpe_pass + (N-1) × e_clb_add) + E_sram
  E_total = E_proj + E_dimm1 + E_soft + E_dimm2

Azure-Lily:
  E_proj  = 3 × N × V × (k_bit × e_analoge + k_bit × e_conv × C) + E_sram
  E_dimm1 = N² × d × e_dsp_mac + E_sram
  E_soft  = N × (N × e_clb_exp + (N-1) × e_clb_add + e_clb_inv + N × e_dsp_mul)
  E_dimm2 = N × d × N × e_dsp_mac + E_sram
  E_total = E_proj + E_dimm1 + E_soft + E_dimm2
```

### Latency (per attention head, at FPGA clock freq f_fpga)

```
NL-DPE:
  L_proj  = 3 × N × t_pipeline / min(3, P_proj)
  L_dimm1 = N × (1 + ceil(d/C) × t_dpe_pass_cycles + ceil(log2(d))) / f_fpga
  L_soft  = N × (ceil(N/C) × t_dpe_pass_cycles + ceil(log2(N)) + 1 + N) / f_fpga
  L_dimm2 = N × ceil(N/C) × t_dpe_pass_cycles / f_fpga  [log]
          + N × (1 + ceil(N/C) × t_dpe_pass_cycles + ceil(log2(N))) / f_fpga  [exp+reduce]
  L_total = L_proj + L_dimm1 + L_soft + L_dimm2

Azure-Lily:
  L_proj  = 3 × N × t_pipeline / min(3, P_proj)
  L_dimm1 = ceil(N / total_dsp) × N × ceil(d / DSP_WIDTH) / f_fpga
  L_soft  = N × (1 + ceil(log2(N)) + 1 + N) / f_fpga
  L_dimm2 = ceil(N / total_dsp) × N × ceil(N / DSP_WIDTH) / f_fpga
  L_total = L_proj + L_dimm1 + L_soft + L_dimm2
```

---

## 7. Constants (from physical design / config files)

| Constant | NL-DPE | Azure-Lily | Unit |
|----------|--------|------------|------|
| e_analoge | 3.89 | 0 | pJ/pass |
| e_conv (ADC) | 0 | 2.33 × C | pJ/pass |
| e_digital (ACAM) | 0.171 × C | 0 | pJ/pass |
| e_clb_add | 0.085 | 0.085 | pJ/op |
| e_dsp_mac | 0.908 | 0.908 | pJ/op |
| e_clb_exp (LUT) | 2.112 | 2.112 | pJ/element |
| k_bit | 8 | 8 | bit-serial passes |
| t_dpe_pass | 10 | 10 | ns (at 1 GHz core) |
| DSP_WIDTH | 48 | 48 | bits |
| f_fpga | 300 | 300 | MHz |
