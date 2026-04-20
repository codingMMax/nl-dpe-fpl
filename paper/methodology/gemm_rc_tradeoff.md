# Crossbar $(R, C)$ Tradeoff for GEMM under Regime B (Layout A)

**Scope.** This document analyses the per-pass pipeline cost and the
per-GEMM resource cost of a DPE hard block as a function of its
crossbar geometry $(R, C)$ — where $R$ is the number of rows (input
length) and $C$ is the number of columns (output length).

The analysis is **strictly under the committed path**: Layout A
(natural int8 packed, load-then-compute) and Regime B (single-buffered
input and output register). Layout B and Regime C are archived in the
pipeline-model doc and are *not* part of this DSE.

Symbols used throughout:

| Symbol | Meaning | Typical value |
|---|---|---|
| $R$ | crossbar rows (input length) | $\{128, 256, 512, 1024\}$ |
| $C$ | crossbar cols (output length) | $\{64, 128, 256\}$ |
| $W_{\text{DPE}}$ | DPE-to-fabric bus width | $40$ bits (= $5$ int8/cycle) |
| $N_{\text{in\_bits}}$ | input precision | $8$ |
| $K$ | GEMM inner (reduction) dimension | workload-dependent |
| $N$ | GEMM output dimension | workload-dependent |
| $M$ | GEMM outer (batch/sequence) dimension | workload-dependent |
| $V = \lceil K/R \rceil$ | vertical DPE tiles per output column | |
| $H = \lceil N/C \rceil$ | horizontal DPE tiles per output row | |

---

## 1. Per-pass cycle cost (Layout A)

A single DPE pass under Layout A is:
$$
\text{Load} \;\to\; \text{Analog fire} \;\to\; \text{Output}
$$

- **Load (feed)**: stream $R$ int8 values into the on-die input buffer
  through the $W_{\text{DPE}}$-bit bus. Once full, the DPE bit-slices
  the buffer internally and fires $N_{\text{in\_bits}}$ analog cycles.
$$
L_{\text{feed}} \;=\; \left\lceil \frac{R \cdot 8}{W_{\text{DPE}}} \right\rceil
$$
- **Compute (analog fires + digital post)**: $N_{\text{in\_bits}}$
  bit-serial fires through the crossbar plus ADC/ACAM:
$$
L_{\text{compute}} \;=\; N_{\text{in\_bits}} \;+\; t_{\text{ADC/ACAM}}
$$
  Typically $t_{\text{ACAM}} \approx 3$ and $t_{\text{ADC}} \approx 44$
  cycles.
- **Output (drain)**: stream $C$ int8 outputs through the bus:
$$
O \;=\; \left\lceil \frac{C \cdot 8}{W_{\text{DPE}}} \right\rceil
$$

Define the **load-side cost** (load + fire, before the drain stage):
$$
L_A \;\equiv\; L_{\text{feed}} \;+\; L_{\text{compute}}
      \;=\; \left\lceil \frac{R \cdot 8}{W_{\text{DPE}}} \right\rceil
      \;+\; N_{\text{in\_bits}} \;+\; t_{\text{ADC/ACAM}}
$$

Single-pass latency:
$$
T_{\text{single}} \;=\; L_A \;+\; O
$$

---

## 2. Multi-pass composition (Regime B)

For $M$ back-to-back passes under **Regime B** (single-buffered input +
output), the next pass's load begins only after the current pass's
fires complete (single input buffer), and the next pass's fire is
blocked until the current pass's drain releases the shift-add register
(single output register). When $L_A$ and $O$ are compared, the
inter-pass steady-state interval is:

$$
\boxed{\;
\text{steady}_B \;=\; \max(L_A,\, O)
\;}
$$

Total $M$-pass latency:
$$
\boxed{\;
T(M) \;=\; L_A \;+\; \text{steady}_B \cdot (M - 1) \;+\; O
\;}
$$

---

## 3. Two pipeline regimes, one model: **feed-bound vs drain-bound**

The behaviour of $\text{steady}_B = \max(L_A, O)$ splits the $(R, C)$
design space into two regions.

**Feed-bound** ($L_A > O$):
$$
\text{steady}_B = L_A \quad \Longleftrightarrow \quad
\left\lceil \frac{R \cdot 8}{W_{\text{DPE}}} \right\rceil + N_{\text{in\_bits}} + t_{\text{ADC/ACAM}}
> \left\lceil \frac{C \cdot 8}{W_{\text{DPE}}} \right\rceil
$$

In this region, **smaller $R$ means smaller $L_A$, which means faster
per-pass**. The drain is fully hidden by the next pass's feed; output
hardware is partly idle.

**Drain-bound** ($O > L_A$):
$$
\text{steady}_B = O \quad \Longleftrightarrow \quad
\left\lceil \frac{C \cdot 8}{W_{\text{DPE}}} \right\rceil
> \left\lceil \frac{R \cdot 8}{W_{\text{DPE}}} \right\rceil + N_{\text{in\_bits}} + t_{\text{ADC/ACAM}}
$$

In this region, **smaller $C$ is faster per-pass** (the drain
dominates). The feed is partly idle.

**Balanced point** ($L_A \approx O$):
$$
\left\lceil \frac{R \cdot 8}{W_{\text{DPE}}} \right\rceil
\;\approx\; \left\lceil \frac{C \cdot 8}{W_{\text{DPE}}} \right\rceil + N_{\text{in\_bits}} + t_{\text{ADC/ACAM}}
$$

At balance, both feed and drain hardware are fully utilised, but
further reducing either one no longer reduces $\text{steady}_B$.
Balance is the **local minimum** of $\text{steady}_B$ for a given
$(L_A, O)$ pair; it's the best the pipeline can do at that geometry.

For concrete numbers at $W_{\text{DPE}}=40$, $N_{\text{in\_bits}}=8$,
ACAM ($t_{\text{ADC/ACAM}}=3$):

| $R$ | $L_A$ (cyc) | Drain-bound threshold $C^\star$ |
|---:|---:|---:|
| 128 | $\lceil 128 \cdot 8 / 40 \rceil + 8 + 3 = 37$ | $C^\star \approx 180$ |
| 256 | $52 + 8 + 3 = 63$ | $C^\star \approx 310$ |
| 512 | $103 + 8 + 3 = 114$ | $C^\star \approx 565$ |
| 1024 | $205 + 8 + 3 = 216$ | $C^\star \approx 1070$ |

So under ACAM + $W_{\text{DPE}}=40$, only the corner
$(R=128,\, C=256)$ of our standard DSE grid is drain-bound; every
other cell is feed-bound.

---

## 4. Two pulls in tension

The $(R, C)$ sweep has **two competing pressures**:

### (a) Cycle-throughput pull — prefer small $R$ (feed-bound) or small $C$ (drain-bound)

- Feed-bound: $T(M) \sim L_A \cdot M = (\lceil R \cdot 8 / W_{\text{DPE}} \rceil + \text{const}) \cdot M$. Linear in $R$.
- Drain-bound: $T(M) \sim O \cdot M = \lceil C \cdot 8 / W_{\text{DPE}} \rceil \cdot M$. Linear in $C$.

Either way, shrinking the binding dimension wins per-pass cycles.

### (b) Resource / reduction / activation pull — prefer large $R$ (fewer V tiles)

A GEMM with reduction dimension $K$ is vertically tiled as
$V = \lceil K/R \rceil$. Larger $R$ means:

- **Fewer DPE hard blocks**: total DPE count scales as $V \cdot H$.
- **Smaller CLB reduction tree**: adder-tree depth is
  $\lceil \log_2 V \rceil$; width is $V - 1$ adder cells per output col.
  This tree is pipelined with the drain stage (fits when
  $\lceil \log_2 V \rceil < O$, which holds in all our configs).
  Even so, it consumes CLB area and routing.
- **In-place activation**: when $V = 1$ on ACAM hardware, the ACAM
  inside the DPE *absorbs* the activation function at zero extra CLB
  cost. When $V > 1$, a CLB LUT activation must be added after the
  reduction tree. This is a step-change, not smooth:
$$
\text{act\_cost}(V, \text{hw}) =
\begin{cases}
0 & \text{if } V = 1 \text{ and hw} = \text{ACAM} \\
\text{CLB LUT (1 cycle, $N$ LUT entries)} & \text{otherwise}
\end{cases}
$$

So the $V = 1$ regime — requiring $R \geq K$ — is architecturally
*special*: it eliminates both the reduction tree and the CLB
activation (on ACAM hardware).

### Tradeoff matrix

| | Small $R$ (feed-bound region) | Large $R$ ($V = 1$) |
|---|---|---|
| Per-pass cycles | ✓ smaller $L_A$ → faster | ✗ larger $L_A$ → slower |
| Total DPE count $V \cdot H$ | ✗ grows as $\lceil K/R \rceil \cdot H$ | ✓ $1 \cdot H = H$ |
| CLB reduction tree | ✗ $\log_2 V$ depth, $(V-1)H$ cells | ✓ none |
| Activation on ACAM hw | ✗ CLB LUT (forced by $V > 1$) | ✓ absorbed in ACAM |
| BRAM usage | ✗ more partial-sum buffers | ✓ fewer |

### Cycle-throughput pull for $C$

For fixed $R$, shrinking $C$ helps only in the drain-bound corner.
Otherwise $C$ is "free" cycle-wise but not resource-wise:

| | Small $C$ | Large $C$ |
|---|---|---|
| Per-pass cycles | ✓ (only if drain-bound) | ✗ (only if drain-bound) |
| Horizontal tile count $H = \lceil N/C \rceil$ | ✗ larger $H$, more DPEs/col | ✓ smaller $H$ |
| Output bus utilisation | drain small, bus partly idle | drain large, bus well-used |
| Activation LUT width | smaller $N/H$ per lane | wider per lane |

---

## 5. Per-GEMM total latency (Regime B, Layout A)

For a GEMM of shape $M \times K \times N$ tiled as $V \times H$ DPEs:

- Each output row requires $H$ DPE-passes worth of work in parallel
  across $H$ output column tiles, each of which handles
  $\text{cols\_per\_DPE} = \lceil N / (H \cdot C) \rceil$
  output cols. The $V$ vertical tiles run in parallel and their
  partial sums feed the CLB reduction tree.
- Total pass count per output row across the $H$ parallel lanes
  is $\text{cols\_per\_DPE}$ (what the sim calls `cols_per_dpe`), and
  the outer batch gives $M$ such rows — so the total sequential
  pass count per DPE is:
$$
M_{\text{eff}} \;=\; M \cdot \text{cols\_per\_DPE} \;=\; M \cdot \left\lceil \frac{N}{H \cdot C} \right\rceil
$$

Plugging into the Regime-B formula:
$$
\boxed{\;
T_{\text{GEMM}} \;=\; L_A \;+\; \max(L_A,\, O) \cdot (M_{\text{eff}} - 1) \;+\; O
\;}
$$

In the (typical) feed-bound regime this simplifies to:
$$
T_{\text{GEMM}} \;\approx\; L_A \cdot M_{\text{eff}} \;+\; O
\;=\; \left(\left\lceil \tfrac{R \cdot 8}{W_{\text{DPE}}} \right\rceil + N_{\text{in\_bits}} + t_{\text{ADC/ACAM}}\right) \cdot M \cdot \left\lceil \tfrac{N}{H C} \right\rceil \;+\; \left\lceil \tfrac{C \cdot 8}{W_{\text{DPE}}} \right\rceil
$$

---

## 6. Resource cost (per-inference, Regime B, Layout A)

Let $\text{CLB}_{\text{reduce}}(V)$ and $\text{CLB}_{\text{act}}(V,\text{hw})$
be the CLB contribution from the reduction tree and the activation
stage. Then per GEMM:

- **DPE hard blocks**:
$$
\#\text{DPE} \;=\; V \cdot H \;=\; \left\lceil \tfrac{K}{R} \right\rceil \cdot \left\lceil \tfrac{N}{C} \right\rceil
$$
- **CLB reduction tree**: pipelined with drain; area proportional to
$$
\#\text{CLB}_{\text{reduce}} \;\sim\; (V - 1) \cdot H
$$
  and logical depth $\lceil \log_2 V \rceil$.
- **CLB activation LUT** (on ACAM hardware only exists when $V > 1$):
$$
\#\text{CLB}_{\text{act}} \;=\; \begin{cases}
0 & V = 1 \land \text{hw} = \text{ACAM} \\
O(N) & \text{otherwise (LUT per output element)}
\end{cases}
$$
- **BRAM**: partial-sum storage scales with $V \cdot H$ and $M$.

---

## 7. DSE grid for Phase 2.1 — GEMM on $(R, C)$

### FC workload shapes (extended from the existing DSE spec)

The existing Round-1 DSE
(`dse_experiment_plan.md` §3) sweeps six FC shapes as $K \times N$ GEMV
($M = 1$). Phase 2.1 extends those to GEMM by adding a batch /
sequence dimension $M > 1$, because Regime B's $L_A \cdot M + O$
benefit is only visible for $M \gg 1$.

**Baseline FC shapes** (reused from Round-1):

| Shape | $K$ | $N$ | Origin |
|---|---:|---:|---|
| `fc_64_64` | 64 | 64 | small FC |
| `fc_128_128` | 128 | 128 | attention projection proxy ($d_{\text{model}}=128$) |
| `fc_512_128` | 512 | 128 | medium FC |
| `fc_512_512` | 512 | 512 | large-wide FC |
| `fc_256_512` | 256 | 512 | wide FC |
| `fc_2048_256` | 2048 | 256 | large FC (worst-case DPE count) |

**Extended for general DNN:**

| Shape | $K$ | $N$ | Origin |
|---|---:|---:|---|
| `fc_768_3072` | 768 | 3072 | Transformer-base FFN1 up-projection |
| `fc_3072_768` | 3072 | 768 | Transformer-base FFN2 down-projection |
| `fc_1024_4096` | 1024 | 4096 | GPT-2 medium FFN1 |
| `fc_4096_1024` | 4096 | 1024 | GPT-2 medium FFN2 |

**Batch / sequence axis** $M$:

| $M$ | Intent |
|---|---|
| $1$ | latency-critical single inference (= GEMV, Regime B reduces to Regime A) |
| $128$ | typical sequence length in BERT-tiny / transformer inference |
| $1024$ | long-context / large-batch throughput |

### $(R, C)$ axes

Match the existing Round-1 12-config grid:
$$
R \in \{128,\, 256,\, 512,\, 1024\}, \qquad C \in \{64,\, 128,\, 256\}
$$

Total grid points: $4 \times 3 = 12$ crossbar geometries, swept per
workload.

### Metrics per $(R, C, \text{workload})$

1. **Analytical per-pass cost** $L_A, O, \text{steady}_B$ — from the
   current sim (post-Phase-1, Regime B).
2. **Analytical total GEMM latency** $T_{\text{GEMM}}(M)$ in cycles
   and in ns (scaled by VTR Fmax).
3. **Regime diagnostic**: is this config feed-bound, drain-bound, or
   balanced?
4. **Resource counts** from VTR synthesis:
   - DPE hard blocks (assert $= V \cdot H$ exactly)
   - BRAMs
   - CLBs (differentiating *GEMV body*, *reduction tree*, *activation LUT*)
   - $F_{\max}$ (avg across 3 seeds)
5. **Activation routing**: ACAM-absorbed iff $V = 1$ on ACAM hardware,
   else CLB LUT.

### Expected Pareto front

Plot $T_{\text{GEMM}}$ (cycles or ns) versus total CLB count (or total
area). The Pareto front should expose:

- **$V = 1$ configs** (large $R$): lower CLB count due to no reduction
  tree and ACAM-absorbed activation, but higher per-pass cost.
- **$V > 1$ configs** (small $R$): lower per-pass cost, but the
  reduction tree and forced CLB activation inflate the CLB count.
- **Drain-bound corner** ($R = 128, C = 256$): a per-pass outlier
  (drain dominates $L_A$); smaller $C$ would have been faster.

The sweet spot for a given workload should land where the cycle
benefit of shrinking $R$ is not yet overtaken by the reduction /
activation CLB cost of the resulting $V > 1$ tiling.

---

## 8. Summary equations (presentation-ready)

Per-pass:
$$
L_A = \left\lceil \frac{R \cdot 8}{W_{\text{DPE}}} \right\rceil + N_{\text{in\_bits}} + t_{\text{ADC/ACAM}} \qquad
O = \left\lceil \frac{C \cdot 8}{W_{\text{DPE}}} \right\rceil
$$

Regime B steady-state:
$$
\text{steady}_B = \max(L_A,\, O)
$$

Total $M$-pass:
$$
T(M) = L_A + \max(L_A,\, O) \cdot (M - 1) + O
$$

Tiling:
$$
V = \left\lceil \frac{K}{R} \right\rceil, \qquad H = \left\lceil \frac{N}{C} \right\rceil, \qquad M_{\text{eff}} = M \cdot \left\lceil \frac{N}{H \cdot C} \right\rceil
$$

GEMM total (feed-bound approximation):
$$
T_{\text{GEMM}} \;\approx\; L_A \cdot M_{\text{eff}} + O
$$

DPE count:
$$
\#\text{DPE} = V \cdot H = \left\lceil \frac{K}{R} \right\rceil \cdot \left\lceil \frac{N}{C} \right\rceil
$$

Activation cost step function:
$$
\text{act\_cost}(V, \text{hw}) =
\begin{cases}
0 & V = 1 \land \text{hw} = \text{ACAM} \\
\text{CLB LUT} & \text{otherwise}
\end{cases}
$$

The $(R, C)$ DSE sweeps these quantities for each workload and exposes
the Pareto front between cycle-throughput (small $R$) and resource
cost ($V = 1$ favored by large $R$), with the drain-bound corner as
the boundary where shrinking $C$ can also help.
