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
| $W_{DPE}$ | DPE-to-fabric bus width | $40$ bits (= $5$ int8/cycle) |
| $N_b$ | input precision | $8$ |
| $K$ | GEMM inner (reduction) dimension | workload-dependent |
| $N$ | GEMM output dimension | workload-dependent |
| $M$ | GEMM outer (batch/sequence) dimension | workload-dependent |
| $V = \lceil K/R \rceil$ | vertical DPE tiles per output column | |
| $H = \lceil N/C \rceil$ | horizontal DPE tiles per output row | |

---

## 1. Per-pass cycle cost (Layout A)

A single DPE pass under Layout A is: $\text{Load} \to \text{Analog fire} \to \text{Output}$.

- **Load (feed)**: stream $R$ int8 values into the on-die input buffer
  through the $W_{DPE}$-bit bus. Once full, the DPE bit-slices the
  buffer internally and fires $N_b$ analog cycles. Load cycles:
  $L_{feed} = \lceil R \cdot 8 / W_{DPE} \rceil$.
- **Compute (analog fires + digital post)**: $N_b$ bit-serial fires
  plus ADC/ACAM, giving $L_{comp} = N_b + t_{ADC/ACAM}$. Typically
  $t_{ACAM} \approx 3$ and $t_{ADC} \approx 44$ cycles.
- **Output (drain)**: stream $C$ int8 outputs through the bus:
  $O = \lceil C \cdot 8 / W_{DPE} \rceil$.

Define the **load-side cost** (load + fire, before the drain stage):
$L_A \equiv L_{feed} + L_{comp} = \lceil R \cdot 8 / W_{DPE} \rceil + N_b + t_{ADC/ACAM}$.

Single-pass latency: $T_{single} = L_A + O$.

---

## 2. Multi-pass composition (Regime B)

For $M$ back-to-back passes under **Regime B** (single-buffered input
+ output), the next pass's load begins only after the current pass's
fires complete (single input buffer), and the next pass's fire is
blocked until the current pass's drain releases the shift-add register
(single output register). When $L_A$ and $O$ are compared, the
inter-pass steady-state interval is:

$\text{steady}_B = \max(L_A, O)$

Total $M$-pass latency:

$T(M) = L_A + \max(L_A, O) \cdot (M - 1) + O$

---

## 3. Two pipeline regimes, one model: feed-bound vs drain-bound

The behaviour of $\text{steady}_B = \max(L_A, O)$ splits the $(R, C)$
design space into two regions.

**Feed-bound** ($L_A > O$): occurs when
$\lceil R \cdot 8 / W_{DPE} \rceil + N_b + t_{ADC/ACAM} > \lceil C \cdot 8 / W_{DPE} \rceil$.
Then $\text{steady}_B = L_A$, so **smaller $R$ means smaller $L_A$
means faster per-pass**. The drain is fully hidden by the next
pass's feed; output hardware is partly idle.

**Drain-bound** ($O > L_A$): occurs when
$\lceil C \cdot 8 / W_{DPE} \rceil > \lceil R \cdot 8 / W_{DPE} \rceil + N_b + t_{ADC/ACAM}$.
Then $\text{steady}_B = O$, so **smaller $C$ is faster per-pass** (the
drain dominates). The feed is partly idle.

**Balanced point** ($L_A \approx O$): when
$\lceil R \cdot 8 / W_{DPE} \rceil \approx \lceil C \cdot 8 / W_{DPE} \rceil + N_b + t_{ADC/ACAM}$,
both feed and drain hardware are fully utilised, but further reducing
either one no longer reduces $\text{steady}_B$. Balance is the
**local minimum** of $\text{steady}_B$ for a given $(L_A, O)$ pair;
it's the best the pipeline can do at that geometry.

For concrete numbers at $W_{DPE}=40$, $N_b=8$, ACAM
($t_{ADC/ACAM}=3$):

| $R$ | $L_A$ (cyc) | Drain-bound threshold $C^\star$ |
|---:|---:|---:|
| 128 | $\lceil 128 \cdot 8 / 40 \rceil + 8 + 3 = 37$ | $C^\star \approx 180$ |
| 256 | $52 + 8 + 3 = 63$ | $C^\star \approx 310$ |
| 512 | $103 + 8 + 3 = 114$ | $C^\star \approx 565$ |
| 1024 | $205 + 8 + 3 = 216$ | $C^\star \approx 1070$ |

So under ACAM + $W_{DPE}=40$, only the corner $(R=128, C=256)$ of our
standard DSE grid is drain-bound; every other cell is feed-bound.

---

## 4. Two pulls in tension

The $(R, C)$ sweep has **two competing pressures**:

### (a) Cycle-throughput pull — prefer small $R$ (feed-bound) or small $C$ (drain-bound)

- Feed-bound: $T(M) \sim L_A \cdot M$, which is linear in $R$ through
  $L_A = \lceil R \cdot 8 / W_{DPE} \rceil + \text{const}$.
- Drain-bound: $T(M) \sim O \cdot M$, linear in $C$.

Either way, shrinking the binding dimension wins per-pass cycles.

### (b) Resource / reduction / activation pull — prefer large $R$ (fewer $V$ tiles)

A GEMM with reduction dimension $K$ is vertically tiled as
$V = \lceil K/R \rceil$. Larger $R$ means:

- **Fewer DPE hard blocks**: total DPE count scales as $V \cdot H$.
- **Smaller CLB reduction tree**: adder-tree depth is $\lceil \log_2 V \rceil$;
  width is $V - 1$ adder cells per output col. This tree is pipelined
  with the drain stage (fits when $\lceil \log_2 V \rceil < O$, which
  holds in all our configs). Even so, it consumes CLB area and
  routing.
- **In-place activation**: when $V = 1$ on ACAM hardware, the ACAM
  inside the DPE *absorbs* the activation function at zero extra CLB
  cost. When $V > 1$, a CLB LUT activation must be added after the
  reduction tree. The activation cost is a step function: $0$ if
  $V = 1$ and hw $= \text{ACAM}$, otherwise a CLB LUT (1 cycle,
  $N$ LUT entries).

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
  $\text{cols\_per\_DPE} = \lceil N / (H \cdot C) \rceil$ output cols.
  The $V$ vertical tiles run in parallel and their partial sums feed
  the CLB reduction tree.
- Total pass count per output row across the $H$ parallel lanes is
  $\text{cols\_per\_DPE}$ (the sim calls it `cols_per_dpe`), and the
  outer batch gives $M$ such rows — so the total sequential pass count
  per DPE is $M_{eff} = M \cdot \lceil N / (H \cdot C) \rceil$.

Plugging into the Regime-B formula:

$T_{GEMM} = L_A + \max(L_A, O) \cdot (M_{eff} - 1) + O$

In the (typical) feed-bound regime this simplifies to:

$T_{GEMM} \approx L_A \cdot M_{eff} + O = (\lceil R \cdot 8 / W_{DPE} \rceil + N_b + t_{ADC/ACAM}) \cdot M \cdot \lceil N / (HC) \rceil + \lceil C \cdot 8 / W_{DPE} \rceil$

---

## 6. Resource cost (per-inference, Regime B, Layout A)

Per GEMM:

- **DPE hard blocks**: $\#\text{DPE} = V \cdot H = \lceil K/R \rceil \cdot \lceil N/C \rceil$.
- **CLB reduction tree**: pipelined with drain; area proportional to
  $\#\text{CLB}_{reduce} \sim (V - 1) \cdot H$ and logical depth
  $\lceil \log_2 V \rceil$.
- **CLB activation LUT** (on ACAM hardware, exists only when $V > 1$):
  $\#\text{CLB}_{act} = 0$ if $V = 1$ and hw $= \text{ACAM}$, else
  $O(N)$ (LUT per output element).
- **BRAM**: partial-sum storage scales with $V \cdot H$ and $M$.

---

## 7. DSE grid for Phase 2.1 — GEMM on $(R, C)$

### GEMM workload shapes (6 real layers from target benchmarks)

Each workload is a concrete GEMM shape $(M, K, N)$ drawn from a layer
in our four target benchmarks (BERT-Tiny, Swin-Tiny, ResNet-9,
VGG-16). This replaces the earlier batch-sweep proposal: every
workload now has a clear network-layer provenance, and the six
together span the diversity of real DNN GEMM shapes we care about
(square / wide / tall, small / large reduction, small / large output).

Convolutions are listed in their im2col GEMM form
$M = H_{out} \cdot W_{out}$, $K = C_{in} \cdot k_h \cdot k_w$,
$N = C_{out}$.

| # | Benchmark | Layer | $(M, K, N)$ | Character |
|---|---|---|---|---|
| 1 | BERT-Tiny | Q/K/V/O projection ($d_{model}{=}128$, seq${=}128$) | $(128, 128, 128)$ | small square; $V{=}1$ for $R{\ge}128$ |
| 2 | BERT-Tiny | FFN1 up-projection ($d_{ff}{=}512$) | $(128, 128, 512)$ | wide output, stresses $H$ sweep |
| 3 | BERT-Tiny | FFN2 down-projection | $(128, 512, 128)$ | tall reduction, forces $V{>}1$ for $R{<}512$ |
| 4 | Swin-Tiny | stage-3 MLP up-projection (window${=}7{\times}7$, dim${=}384$, ratio${=}4$) | $(49, 384, 1536)$ | big $N$, stresses $H$ and per-pass drain |
| 5 | ResNet-9 | mid conv (3×3, 256→256, 16×16 spatial) — im2col | $(256, 2304, 256)$ | very large $K$, stresses $V$ even at $R{=}1024$ |
| 6 | VGG-16 | block-4 conv (3×3, 512→512, 14×14 spatial) — im2col | $(196, 4608, 512)$ | largest $K \cdot N$, worst-case DPE count |

Together these cover:

- **$K$ range**: 128 → 4608 (factor of 36). Exercises $V$ across the
  full $(R, C)$ grid.
- **$N$ range**: 128 → 1536 (factor of 12). Exercises $H$.
- **$M$ range**: 49 → 256. All $M \gg 1$ so Regime B amortisation is
  visible; no synthetic batch sweep needed.
- **Aspect ratios**: square ($d$-model), wide (FFN1), tall (FFN2),
  very wide (Swin MLP), balanced big (ResNet), widest (VGG).

### $(R, C)$ axes

Match the existing Round-1 12-config grid:
$R \in \{128, 256, 512, 1024\}$ and $C \in \{64, 128, 256\}$.
Total grid points: $4 \times 3 = 12$ crossbar geometries, swept per
workload.

### Metrics per $(R, C, \text{workload})$

1. **Analytical per-pass cost** $L_A$, $O$, $\text{steady}_B$ — from the
   current sim (post-Phase-1, Regime B).
2. **Analytical total GEMM latency** $T_{GEMM}(M)$ in cycles and in ns
   (scaled by VTR $F_{\max}$).
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

Plot $T_{GEMM}$ (cycles or ns) versus total CLB count (or total area).
The Pareto front should expose:

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

Per-pass: $L_A = \lceil R \cdot 8 / W_{DPE} \rceil + N_b + t_{ADC/ACAM}$
and $O = \lceil C \cdot 8 / W_{DPE} \rceil$.

Regime B steady-state: $\text{steady}_B = \max(L_A, O)$.

Total $M$-pass: $T(M) = L_A + \max(L_A, O) \cdot (M - 1) + O$.

Tiling: $V = \lceil K/R \rceil$, $H = \lceil N/C \rceil$, and
$M_{eff} = M \cdot \lceil N / (H \cdot C) \rceil$.

GEMM total (feed-bound approximation): $T_{GEMM} \approx L_A \cdot M_{eff} + O$.

DPE count: $\#\text{DPE} = V \cdot H = \lceil K/R \rceil \cdot \lceil N/C \rceil$.

Activation cost: $\text{act\_cost} = 0$ if $V = 1$ and hw $= \text{ACAM}$,
else CLB LUT.

The $(R, C)$ DSE sweeps these quantities for each workload and exposes
the Pareto front between cycle-throughput (small $R$) and resource
cost ($V = 1$ favored by large $R$), with the drain-bound corner as
the boundary where shrinking $C$ can also help.
