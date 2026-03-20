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
| DSE (Q1+Q2) | 6 FC workloads spanning (K,N) space | Parameterizable, architecturally equivalent to CNN/transformer layers |
| Attention case study | Single attention head (N=128, d=128) | Tests ACAM-as-log mode and CLB DIMM/softmax overhead |
| Validation (main results) | LeNet, ResNet | Full networks from prior work, Azure-Lily comparison points |

**FC workloads for DSE** — span (K, N) space of real workload layers:

| Name | K | N | Represents |
|---|---|---|---|
| fc_64_64 | 64 | 64 | Early CNN layers, tiny FC |
| fc_128_128 | 128 | 128 | Attention projection proxy (Q/K/V, d=128) |
| fc_512_128 | 512 | 128 | ResNet conv2-4 range |
| fc_2048_256 | 2048 | 256 | ResNet/VGG11 deep layers |
| fc_256_512 | 256 | 512 | VGG11 conv5+, transformer FFN |
| fc_512_512 | 512 | 512 | Large FC, transformer projection |

---

## Q1. Optimal NL-DPE Block Configuration for FPGA

| Field | Details |
|---|---|
| **Question** | What crossbar configuration (R × C) maximizes throughput per area and throughput per Joule under realistic FPGA mapping? |
| **Goal** | Select one NL-DPE block configuration for Q2/Q3 and main evaluation. |
| **Knobs** | R ∈ {128, 256, 512} × C ∈ {64, 128, 256} = 9 non-square configs; I/O width fixed at 16-bit |
| **What stays fixed** | I/O width (16-bit); ACAM rows (re-characterization out of scope); CLB-based reduction; VTR auto_layout |
| **Workloads** | 6 FC workloads (fc_64_64 through fc_2048_256) spanning the ACAM eligibility boundary |
| **Predictor** | CW-AOR can be computed analytically; Round 1 validates it against actual VTR+IMC results |
| **Metrics** | Block area (from `area_power.py`); FPGA area (grid × CLB_tile); Fmax; throughput/mm²; throughput/J |
| **Actual finding** | **512×128 is optimal** (GM combined = 0.852). R=512 enables ACAM on 5/6 workloads; R=256 on 3/6; R=128 on 2/6. C=128 balances tile area vs horizontal tiling. The row dimension is the dominant knob — ACAM eligibility (V=1 threshold) drives the ranking more than area or Fmax. |
| **Main figures** | Config ranking bar chart (GM tput/mm² and tput/J); 9×6 heatmap (normalized tput/mm², annotated with V and ACAM eligibility) |
| **Main table** | All 9 configs: area, tile W×H, power, ACAM-eligible workloads, GM tput/mm², GM tput/J, GM combined. 512×128 highlighted. |

---

## Q2. Optimal NL-DPE Block Count in Fixed-Area FPGA

| Field | Details |
|---|---|
| **Question** | How many NL-DPE blocks should occupy a fixed-area FPGA fabric to maximize throughput/area and energy efficiency? |
| **Goal** | Identify the NL-DPE fabric density where compute capacity gain balances CLB displacement cost. |
| **Knobs** | NL-DPE block count: 5–6 points from 0 (pure CLB) to high density; block config fixed from Q1 |
| **What stays fixed** | Block design from Q1; DSP count; routing architecture; total FPGA area budget |
| **Workloads** | 6 FC workloads (same set as Q1) |
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
| **Workloads** | 6 FC workloads (DSE) + attention head (N=128, d=128) + LeNet/ResNet (validation from prior work) |
| **Predictor** | ACAM benefit ≈ f(V=1 eligibility): validated by comparing V=1 vs V>1 configs on same workload within Round 1 data |
| **Metrics** | Energy (pJ); energy efficiency (inf/J); FPGA fabric energy reduction; CLB activation ops eliminated |
| **Actual finding** | Within Round 1 data, the V=1 boundary creates a 2-4× energy discontinuity (e.g., fc_512_128: V=1 at 159-275 pJ vs V=2 at 328-573 pJ). LeNet (prior work): 0.548× energy ratio vs Azure-Lily, ACAM on 4/5 layers. ResNet: ACAM on 2/9 layers only. Attention (pending): CLB DIMM/softmax dominates; ACAM-as-log is uniformly available. |
| **Main figures** | Energy vs config for a workload crossing V=1 boundary (step-function plot); attention energy breakdown (DPE vs CLB) |
| **Main table** | Per-workload ACAM eligibility count across 9 configs + energy gap at V=1 boundary |

---

## Main Evaluation: Full-Network Validation

| Field | Details |
|---|---|
| **Workloads** | LeNet, ResNet (from prior work); attention head (pending) |
| **Configuration** | Optimal NL-DPE block from Q1 (512×128), optimal fabric density from Q2 (pending Round 2) |
| **Metrics** | Throughput (inf/s), energy efficiency (inf/J), throughput/area |
| **Baseline** | Azure-Lily (512×128, ADC-only, same VTR flow) — used as architectural context |
| **Primary claim** | The crossbar-size DSE reveals R (row dimension) as the dominant FPGA design knob for ACAM-enabled hard blocks. 512×128 maximizes ACAM eligibility while minimizing tile area. |
| **Secondary claim** | With the DSE-optimal 512×128, NL-DPE has the same row dimension as Azure-Lily; ACAM provides additional energy savings on V=1 layers without a row-dimension penalty. |
| **Note** | VGG11 and BERT-Tiny are deferred — current scope focuses on FC DSE + attention head + LeNet/ResNet validation. May add if time permits before submission. |

---

## Evaluation Matrix

| Question | Workloads | Sweep | Status | Purpose |
|---|---|---|---|---|
| Q1 block sizing | 6 FC workloads | 9 configs (3R × 3C) × 6 WL = 54 runs | **Done** | Find optimal NL-DPE block → 512×128 |
| Q2 fabric density | 6 FC workloads | 4 ratios × 3 configs × 6 WL = 72 runs | Pending (Round 2) | Find optimal NL-DPE density; CLB displacement tradeoff |
| Q3 ACAM value | 6 FC + attention | V=1 vs V>1 within Round 1 data | **Partially done** (FC done, attention pending) | Quantify ACAM step-function benefit |
| Attention case study | N=128, d=128 | 9 configs × 1 WL = 9 runs | Pending (T3a-T3c) | ACAM-as-log + CLB DIMM/softmax analysis |
| Validation | LeNet, ResNet | Fixed config, NL-DPE vs Azure-Lily | Prior work data available | Full-network comparison |
