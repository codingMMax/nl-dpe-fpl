## Core Predictor Metric: Compute-Weighted ACAM Opportunity Ratio (CW-AOR)

**Definition:**
```
CW-AOR = Σ(M_i × K_i × N_i × [V_i=1 AND H_i=1])
         ──────────────────────────────────────────
                  Σ(M_i × K_i × N_i)
```

Where V_i = ceil(K_i / rows), H_i = ceil(N_i / cols), and [·] is 1 when single-tile.

**Why not plain AOR (layer-count-weighted):**
Layer-count AOR is trivially "bigger rows → fewer tiles → more ACAM." CW-AOR captures whether
ACAM activates on computationally heavy layers — which actually drives energy savings.
A workload with 80% single-tile layers by count but 5% by compute gets little real ACAM benefit.

**Under fixed FPGA area budget, System CW-AOR is non-trivial:**
```
System CW-AOR = n_blocks(area, block_size) × CW-AOR(rows, cols, workload)
```
Larger blocks → less tiling → higher per-layer CW-AOR, but fewer blocks fit → less parallelism.
Smaller blocks → more parallelism, but more tiling → lower CW-AOR.
The optimal block size maximizes System CW-AOR — and depends on the workload's (K, N) distribution.

**Role in paper:** CW-AOR is computed analytically from block specs and workload dims.
DSE (Q1+Q2) validates it as a predictor of actual energy efficiency.
Q3 shows that P2 gain ≈ f(CW-AOR). Main results confirm the predictor generalizes.

---

## Workload Strategy

| Role | Workloads | Rationale |
|---|---|---|
| DSE (Q1+Q2) | Weight-persistent GEMV shapes | Parameterizable, architecturally equivalent to CNN/transformer layers, avoids full-network reuse |
| Validation (main results) | ResNet, VGG11, BERT-Tiny | Full networks, clean separation from DSE |

**GEMV shapes for DSE** — span (K, N) space of real workload layers:

| Name | K | N | Represents |
|---|---|---|---|
| Small | 64 | 64 | Early CNN layers, attention head dim |
| Medium-K | 512 | 128 | ResNet conv2-4 range |
| Large-K | 2048 | 256 | ResNet/VGG11 deep layers |
| Wide-N | 256 | 512 | VGG11 conv5+, transformer FFN |
| Square-large | 512 | 512 | Large FC, transformer projection |

---

## Q1. Optimal NL-DPE Block Configuration for FPGA

| Field | Details |
|---|---|
| **Question** | What crossbar size and interface width maximizes System CW-AOR per unit area under realistic FPGA mapping? |
| **Goal** | Select one NL-DPE block configuration for Q2/Q3 and main evaluation. |
| **Knobs** | Crossbar size: 3 points (128×128, 256×256, 512×512); I/O width: 2 points (16-bit, wider) |
| **What stays fixed** | ACAM rows (re-characterization out of scope); CLB-based reduction; baseline FPGA style |
| **Workloads** | GEMV shapes (5 representative shapes covering small/medium/large K and N) |
| **Predictor** | Compute System CW-AOR analytically for each config; validate against simulator energy efficiency |
| **Metrics** | Block area (from `area_power.py`); block power; Fmax; System CW-AOR; throughput/area; energy efficiency |
| **Expected finding** | The FPGA-optimal block is not the largest — beyond a crossbar size where most GEMV K dims are already single-tile, area cost grows faster than CW-AOR gain. A 256×256 or modest variant is likely optimal. |
| **Main figures** | CW-AOR vs crossbar size (fixed area budget); energy efficiency vs crossbar size; CW-AOR as predictor vs simulated efficiency (scatter plot showing correlation) |
| **Main table** | Candidate configs: crossbar size, area, power, Fmax, System CW-AOR, energy eff, throughput/area. Final selected row highlighted. |

---

## Q2. Optimal NL-DPE Block Count in Fixed-Area FPGA

| Field | Details |
|---|---|
| **Question** | How many NL-DPE blocks should occupy a fixed-area FPGA fabric to maximize throughput/area and energy efficiency? |
| **Goal** | Identify the NL-DPE fabric density where compute capacity gain balances CLB displacement cost. |
| **Knobs** | NL-DPE block count: 5–6 points from 0 (pure CLB) to high density; block config fixed from Q1 |
| **What stays fixed** | Block design from Q1; DSP count; routing architecture; total FPGA area budget |
| **Workloads** | GEMV shapes (same set as Q1) |
| **Predictor** | System CW-AOR also decreases as block count grows beyond the point where CLB reduction budget is starved — show this analytically |
| **Metrics** | Throughput (inf/s); energy efficiency (inf/J); throughput/area; CLB utilization; CLB pressure proxy (CLBs remaining after reduction logic) |
| **Expected finding** | Performance improves with NL-DPE count up to a density point, then degrades as remaining CLBs become insufficient for adder-tree reduction. Sweet spot is workload-dependent but predictable from CW-AOR. |
| **Main figures** | Throughput/area vs block count; energy efficiency vs block count; CLB pressure vs block count — all with CW-AOR overlay showing correlation |
| **Main table** | Per-shape best block count, suite-optimal block count, and regret vs per-shape optimal. |

---

## Q3. Architectural Value of ACAM Dual-Mode Under Optimal Configuration

| Field | Details |
|---|---|
| **Question** | Using the optimal block config from Q1+Q2, how much additional energy efficiency does ACAM dual-mode (P2) provide over ADC-only (P1), and does CW-AOR predict the gain? |
| **Goal** | Validate CW-AOR as a predictor of P2 vs P1 gain; quantify the architectural value of dual-mode independent of circuit efficiency differences. |
| **Policies** | P1: ACAM always in ADC mode (activation in CLB fabric); P2: ACAM activates locally when V=1 and H=1, falls back to ADC otherwise |
| **Knobs** | Policy (P1 vs P2) at the Q1+Q2 optimal fabric point |
| **Workloads** | GEMV shapes (DSE) + ResNet, VGG11, BERT-Tiny (main results) |
| **Predictor** | P2 gain ≈ f(CW-AOR): validate on GEMV, confirm it holds for full networks |
| **Metrics** | Energy (pJ); energy efficiency (inf/J); FPGA fabric energy reduction; CLB activation ops eliminated |
| **Expected finding** | P2 gain is well-predicted by CW-AOR. BERT-Tiny attention heads (small K/N → high CW-AOR) see the largest gain. ResNet/VGG11 deep layers (large K → low CW-AOR) see limited gain from dual-mode. The predictor generalizes from GEMV to full networks without re-running DSE. |
| **Main figures** | P2/P1 energy ratio vs CW-AOR (one point per GEMV shape + one per full network) — show linear predictor fit; per-workload P1 vs P2 bar chart |
| **Main table** | Workload, CW-AOR, P1 energy, P2 energy, P2/P1 ratio, predicted ratio from CW-AOR model, prediction error |

---

## Main Evaluation: Full-Network Results

| Field | Details |
|---|---|
| **Workloads** | ResNet, VGG11, BERT-Tiny |
| **Configuration** | Optimal NL-DPE block from Q1, optimal fabric density from Q2, P2 mapping policy |
| **Metrics** | Throughput (inf/s), energy efficiency (inf/J), throughput/area, CW-AOR (reported for context) |
| **Baseline** | Azure-Lily (same FPGA fabric, same VTR flow) — used as architectural context, not primary claim |
| **Primary claim** | Under our systematic methodology (Q1+Q2+Q3), NL-DPE achieves X% better throughput/area and Y% better energy efficiency vs a naively-sized NL-DPE deployment. CW-AOR predicts results across all three workloads. |
| **Secondary claim** | NL-DPE compares favorably to Azure-Lily; differences attributed to architectural block design choices, not circuit-level factors. |

---

## Evaluation Matrix

| Question | Workloads | Sweep | Purpose |
|---|---|---|---|
| Q1 block sizing | 5 GEMV shapes | 3 crossbar sizes × 2 I/O widths | Find optimal NL-DPE block; validate CW-AOR as predictor |
| Q2 fabric density | 5 GEMV shapes | 5–6 block count points | Find optimal NL-DPE density; show CLB displacement tradeoff |
| Q3 dual-mode value | GEMV + ResNet + VGG11 + BERT-Tiny | P1 vs P2 at optimal fabric | Validate CW-AOR predicts P2 gain; quantify dual-mode benefit |
| Main results | ResNet, VGG11, BERT-Tiny | Fixed optimal config | Full-network validation of methodology |
