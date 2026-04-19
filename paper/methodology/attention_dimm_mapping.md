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

## 2. K-Identity Mapping: Key Innovation

### Motivation

In the log-domain DIMM pipeline, the DPE crossbar is programmed with an **identity matrix** to
perform element-wise exp/log conversion via the ACAM. A standard single-identity mapping places
one d_head×d_head identity block in the crossbar. For d_head=64 on a crossbar with C=128 columns,
this wastes half the columns (64/128 = 50% utilization). Wider crossbars waste even more:
C=256 gives 64/256 = 25% utilization.

The K-identity mapping eliminates this waste by packing **K independent identity blocks** into
one crossbar, where K = floor(C / d_head). One DPE pass now processes K vectors simultaneously.

### How It Works

**K = floor(C / d_head)** identity blocks are tiled across the crossbar columns:

```
Example: K=4 identity mapping (1024 rows × 256 columns, d_head=64):

Columns:   [ 0──63 | 64──127 | 128──191 | 192──255 ]
           ┌───────┬─────────┬──────────┬──────────┐
Row 0      │   1   │    0    │    0     │    0     │  ← Identity block 0
  ...      │  ...  │   ...   │   ...    │   ...    │
Row 63     │   1   │    0    │    0     │    0     │
Row 64     │   0   │    1    │    0     │    0     │  ← Identity block 1
  ...      │  ...  │   ...   │   ...    │   ...    │
Row 127    │   0   │    1    │    0     │    0     │
Row 128    │   0   │    0    │    1     │    0     │  ← Identity block 2
  ...      │  ...  │   ...   │   ...    │   ...    │
Row 191    │   0   │    0    │    1     │    0     │
Row 192    │   0   │    0    │    0     │    1     │  ← Identity block 3
  ...      │  ...  │   ...   │   ...    │   ...    │
Row 255    │   0   │    0    │    0     │    1     │
Row 256+   │   0   │    0    │    0     │    0     │  ← unused
           └───────┴─────────┴──────────┴──────────┘

Input:  [vec_0 | vec_1 | vec_2 | vec_3 | zeros(768)]   (K×d_head + padding)
Output: [f(vec_0) | f(vec_1) | f(vec_2) | f(vec_3)]    (f = exp or log via ACAM)
```

**One DPE pass processes K = 4 independent d_head-sized vectors.**

### Applicable Stages

K-identity applies wherever the operand dimension is smaller than C:

| DIMM Stage | Operand dim | K (C=128) | K (C=256) |
|---|---|---|---|
| **QK^T (score exp)** | d_head=64 | K=2 | **K=4** |
| **S×V (wsum exp/log)** | N=128 | K=1 | **K=2** |
| **Softmax exp** | N=128 | K=1 | **K=2** |

With C=256, **all** DIMM stages benefit from K-identity (K≥2).
With C=128, only QK^T benefits (K=2); S×V and softmax already fill C (K=1).

### Energy Impact

The DPE firing energy is fixed per pass (~53 pJ) regardless of how many identity blocks are
packed. K-identity reduces the **per-element** cost by K×:

| Config | C | K (QK^T) | Elements/pass | Cost/element | vs DSP MAC |
|---|---|---|---|---|---|
| Single identity | 64 | 1 | 64 | 0.83 pJ | 1.20× |
| **Proposed (C=128)** | 128 | **2** | 128 | **0.41 pJ** | **2.06×** |
| **AL-like (C=256)** | 256 | **4** | 256 | **0.21 pJ** | **3.71×** |

### Throughput Impact

K-identity directly multiplies the effective DIMM throughput. Combined with W parallel
DIMM lanes (spatial parallelism from multiple DPEs), the total throughput per cycle is:

```
QK^T elements/cycle = W × K
S×V  elements/cycle = W × K_sv    (K_sv = floor(C/N))
```

For BERT-Tiny (N=128, d_head=64) on the two evaluation configs:

**Peak compute bandwidth** (architectural ceiling — no per-pass overhead):

| | Proposed (C=128) | AL-like (C=256) |
|---|---|---|
| K (QK^T) | 2 | 4 |
| K (S×V) | 1 | 2 |
| W (parallel lanes) | 16 | 4 |
| **QK^T elem/cycle (peak)** | **32** | **16** |
| **S×V elem/cycle (peak)** | **16** | **8** |
| DPEs used | 274/294 (93%) | 78/90 (87%) |

These figures answer the question *"when every parallel unit is producing
one MAC output per cycle, how many outputs per cycle in total?"*. Peak is
the architectural ceiling, reached only under Layout B with `W_BRAM ≥ R`
in a many-pass streaming pipeline where per-pass overhead fully amortises
away.

**Realistic per-pass cycles** (dataflow-aware, N=128 × N=128 score matrix;
see `dpe_pipeline_model.md` §5–6 for formulas):

| Configuration | Dataflow | Single-pass cycles | Steady-state per additional pass | QK^T cycles for 128 queries |
|---|---|---:|---:|---:|
| Proposed, W_BRAM = W_DPE = 40 | **Layout A** (current RTL) | L+O = 103 + 26 ≈ 130 | max(103, 26) = 103 | ~13.4 k |
| Proposed, W_BRAM = W_DPE = 40 | **Layout B**, narrow BRAM | 104 + 26 ≈ 130 | 104 | ~13.5 k |
| Proposed, W_BRAM = R = 512 | **Layout B**, matched BRAM | 8 + 26 ≈ 34 | max(8, 26) = 26 | ~3.4 k |
| Idealised (peak-only)         | — | — | — | `N² / (W·K)` = **512** |

Proposed achieves higher **absolute** throughput (more DPEs), while AL-like
achieves higher **per-DPE** efficiency (4× vs 2× K-identity). Both fully
utilize the FPGA.

The measured RTL at Phase I.2 (561 compute cycles end-to-end, score stage
260 of those) aligns with the Layout A row above: the score stage is one
pass per lane × 4 passes per lane (4 dual-identity iterations × ~65 cycles
each). See `fc_verification/VERIFICATION.md` Phase I.2 and
`fc_verification/results/dimm_top_w16_alignment_log.txt` for the
cycle-by-cycle trace.

### No Hardware Change Required

K-identity is purely a **weight programming** change. The same DPE hard block, same ACAM,
same interface. Only the ReRAM crossbar weights differ (K identity blocks instead of one).
The ACAM configuration (exp or log) is unchanged.

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
=== DMMul_1: Q·K^T SCORE MATRIX (K-identity DIMM) ===

For each output S[i][j] = Σ_m exp(log_Q[i][m] + log_K[j][m]):

Step 1: CLB add       log_Q[i][m] + log_K[j][m] for m=0..63    (64 parallel CLB adders)
Step 2: DPE(I^K|exp)  Feed K sets of 64 sums into K-IDENTITY crossbar
                       → ACAM outputs K×64 exp values
                       → processes K elements (i,j)..(i,j+K-1) simultaneously
                       → K = floor(C / d_head): K=2 for C=128, K=4 for C=256
Step 3: CLB reduce    Σ_m exp(...) for each of the K elements    (K parallel reduction trees)

  C=128: DPE pass energy: 53.1 pJ for K=2 elements → 26.6 pJ/element
  C=256: DPE pass energy: 53.1 pJ for K=4 elements → 13.3 pJ/element

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

Step 1: DPE(I^K|log)  log(attn[i][j]) for all j          1 DPE, ACAM=log
                       K_sv = floor(C/N): K=1 for C=128, K=2 for C=256
Step 2: CLB add       log_attn[i][j] + log_V[j][m]       CLB parallel adders
Step 3: DPE(I^K|exp)  exp(sum) → K_sv output rows        1 DPE, ACAM=exp
Step 4: CLB reduce    Σ_j exp(...) → output O[i][m]      CLB reduction tree

  C=128: K_sv=1, N=128 fills all columns (single identity)
  C=256: K_sv=2, processes 2 output rows per DPE pass

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

The same DPE hardware serves multiple functions by reprogramming the ACAM:

| ACAM Mode | Where Used | K-Identity | What It Computes |
|-----------|-----------|------------|-----------------|
| **log** | Q/K/V/O/FFN2 projections | N/A (learned weights) | log(VMM output) |
| **GELU** | FFN1 activation | N/A (learned weights) | GELU(VMM output) |
| **exp** (K-identity) | DMMul_1 QK^T | K = floor(C/d_head) | exp(K×d_head inputs) |
| **exp** (K-identity) | DMMul_2 S×V, Softmax | K = floor(C/N) | exp(K×N inputs) |
| **log** (K-identity) | DMMul_2 attn→log | K = floor(C/N) | log(K×N inputs) |
## 5. DPE Resource Count (per transformer block)

### Baseline (W=1, no spatial parallelism)

| Component | DPEs (C=128) | DPEs (C=256) | ACAM | Crossbar Weights |
|---|---|---|---|---|
| Q projection | 1 | 1 | log | W_Q (learned) |
| K projection | 1 | 1 | log | W_K (learned) |
| V projection | 1 | 1 | log | W_V (learned) |
| O projection | 1 | 1 | log | W_O (learned) |
| FFN1 (128→512) | 4 (H=4) | 2 (H=2) | GELU | W_FFN1 (learned) |
| FFN2 (512→128) | 1 | 1 | log | W_FFN2 (learned) |
| Per-head DIMM (4 stages) | 4 | 4 | exp/log | K-identity |
| **Block total** | **17** | **15** | | |

**Full BERT-Tiny (2 blocks, 2 heads)**: Proposed = 2×17 = **34 DPEs**, AL-like = 2×15 = **30 DPEs**

### With DIMM Parallelism (W lanes per head)

Each additional DIMM lane adds 4 DPEs per head (4 stages). With W lanes:

| | Proposed (C=128) | AL-like (C=256) |
|---|---|---|
| Fixed (proj+FFN) × 2 blocks | 18 DPEs | 14 DPEs |
| DIMM: W × 4 stages × 2 heads × 2 blocks | W=16 → 256 DPEs | W=4 → 64 DPEs |
| **Total** | **274 / 294** (93%) | **78 / 90** (87%) |

## 6. K-Identity vs Conventional DIMM

| Architecture | K (QK^T) | K (S×V) | W lanes | QK^T elem/cycle | DPE util |
|---|---|---|---|---|---|
| **Proposed (C=128)** | 2 | 1 | 16 | **32** | 93% |
| **AL-like (C=256)** | 4 | 2 | 4 | **16** | 87% |
| Azure-Lily (DSP) | N/A | N/A | N/A | **0** (0 DSPs) | 7% idle |

Azure-Lily has 261 DPEs but **cannot use them for DIMM** (no log-domain support).
Its DPE columns replaced all DSP columns (0 DSPs remaining), creating a critical
bottleneck: massive DPE resources are idle during the O(N²) attention computation.

## 7. Scaling with Sequence Length

K-identity benefit increases with wider crossbars (higher K) but the overall advantage
decreases at larger N because DIMM (O(N²)) digital overhead dominates:

| N | NL-DPE K-identity vs Azure-Lily |
|---|---|
| 128 | ~1.77× |
| 256 | ~1.61× |
| 512 | ~1.52× |
| 1024 | ~1.48× |

The cap is set by the DIMM digital overhead (BRAM I/O, CLB reductions) which both
architectures share — this is an inherent property of the O(N²) attention workload,
not an NL-DPE limitation. For linear-layer-dominant workloads (CNNs), NL-DPE delivers
the full 40× efficiency advantage.

## 8. DIMM-only comparison setup (verification)

The "0 DSPs" Azure-Lily in §6 represents the extreme paper-canonical architecture
that replaced all DSP columns with DPE columns. For apples-to-apples DIMM
verification against NL-DPE's W=16 mapping, we instantiate a DIMM-only
**Azure-Lily W=16 DSP variant**: 16 `dsp_mac` per DIMM matmul stage, each = one
`int_sop_4` hard block = 4 MAC/cycle. This variant:

- Is used only for the block-level DIMM architecture comparison (Phase H-M of
  `fc_verification/`), not for the paper's main Proposed vs Canonical contrast.
- Preserves the paper's central per-lane throughput argument (NL-DPE W=16 vs
  Azure-Lily W=16) without forcing the strawman "0 DSP" setup into the DIMM
  verification loop.
- Matches the simulator's `DSP_WIDTH=4` assumption exactly (no CLB-multiply
  helper), so RTL ≡ simulator at the arithmetic level. See
  `fc_verification/VERIFICATION.md` §"Full DIMM Top Verification (W=16)" for
  the cycle alignment and resource-count results.

The paper's §3–6 architecture claims (40× CNN efficiency, 1.77× attention at
N=128, 93% DPE utilization) are unchanged by this verification variant — the
W=16 DSP lane count is a lane-match for the NL-DPE configuration and does not
alter the paper-canonical Azure-Lily's resource story.

## 9. Per-pass dataflow model (dated: 2026-04-18)

This document specifies **which primitive handles which attention stage**
and the per-stage parallelism (K-identity packing, W=16 lane count,
DPE/DSP counts). It does **not** specify the per-pass dataflow —
i.e. how bits move between BRAM and the crossbar inside a single DPE
pass. That model is captured separately in:

> `paper/methodology/dpe_pipeline_model.md` (added 2026-04-18)

The companion document covers:
- Layout A (natural-packed int8 + load-then-compute) vs Layout B
  (bit-plane transposed + streaming bit-serial).
- Transpose-buffer architecture for Layout B and its FPGA cost
  (`R · N_in_bits` flip-flops per DPE input).
- Multi-pass pipelined timing with overlap between the output of pass
  `k` and the load of pass `k+1`.
- Analytical per-pass cycle formulas.

**Assumption for cycle-budget numbers in this document.** The cycle
and DPE-utilisation tables in §5 and §6 assume **Layout A** with
`W_BRAM = W_DPE = 40 bits`. The mapping itself (K-identity, W=16)
is independent of the layout choice — only the per-pass cycle cost
and the read-BRAM port width differ.

If the project later adopts Layout B as the authoritative HW model
(because it better matches real analog IMC implementations), the
mapping structure in §§1–8 is unchanged, but the per-pass cycle
numbers in §5 and the scaling in §7 should be re-derived from
`dpe_pipeline_model.md` §6 formulas.

## 10. Phases (2026-04-18, revised 2026-04-19)

The P4 track is three phases, all under the committed
**Layout A + Regime B** path. See
`paper/methodology/dpe_pipeline_model.md` §§5.3.1, 5.7, and 8.1 for
the authoritative framing. Layout B is archived as design-space
reference (§§3.2, 4 of the model doc) but is not a separately-costed
alternative in the active plan.

At 512×128 under Regime B, Layout A: `T(M) = 111·M + 26`, i.e.
`T(M=8) = 914` cycles — the committed Phase 1 target for the
simulator. Today's RTL already runs this regime, so no RTL
architecture change is needed.

The three phases that affect this document once complete:

- **Phase 1 — Simulator Regime B swap.** Update `gemm_log` from
  `M · cycles_per_pass` (Regime A) to `T(M) = L_A · M + O` under
  Layout A. The "realistic per-pass cycles" table in §5 updates to
  the Regime B Layout A row.
- **Phase 2 — FC RTL re-verify.** No RTL architecture changes. Fix
  any TB / generator bugs that surface under the tighter ≤ 3 per
  stage / ≤ 5 end-to-end tolerance. Does not change this document's
  mapping claims (K-identity, W=16 lane counts); validates them
  under the Regime B simulator.
- **Phase 3 — DIMM RTL re-verify.** No RTL architecture changes.
  Re-align DIMM Phase I.2 + J (`mac_qk`, `softmax`, `mac_sv`)
  against the Regime-B sim under the same ≤ 3 / ≤ 5 tolerance.
  Peak-bandwidth table in §5 is unchanged. Realistic per-pass
  numbers refresh after re-alignment.

Full derivation and the 512×128 worked example for Layout A live in
`dpe_pipeline_model.md` §§5.3.1 and 6.3. The archived Layout B
alternative (transpose block, wide-BRAM port, cycle tables) lives in
the same doc §§3.2, 4, 5.7 as reference material only.
