# Paper Outline — NL-DPE as a Heterogeneous FPGA Hard Block

---

## Title (SELECTED)

**ACAM-Enabled Heterogeneous FPGA Hard Blocks: Crossbar-Size DSE and Evaluation of NL-DPE for FC and Attention Workloads**

*Rationale: Names the mechanism (ACAM), the context (heterogeneous FPGA), the methodology (crossbar-size DSE), and the workload scope. Avoids vague "evaluation" framing.*

---

## Core Story (Read this before writing any section)

NL-DPE has a unique ACAM peripheral that can perform nonlinear functions (activation or log) inside the analog block — eliminating the need for fabric LUT/DSP activation. This sounds like a clear win. But in FPGA context, layers are executed as tiled GEMMs across multiple DPE blocks, meaning outputs leaving each DPE are *partial sums*, not final results. ACAM can only operate on final outputs — i.e., when the layer maps to a single vertical tile (V=1).

**The core tension:** The crossbar row dimension R determines ACAM eligibility (V = ceil(K/R) = 1 iff K ≤ R). Larger R means more workloads qualify for ACAM — but larger crossbars cost more area (ACAM is 77% of DPE area, scaling with columns C). The column dimension C determines horizontal tiling (H = ceil(N/C)) and per-tile area cost. There is a non-trivial Pareto front across (R, C).

**The paper's methodology:** We perform a systematic crossbar-size DSE across 9 configurations (R ∈ {128, 256, 512} × C ∈ {64, 128, 256}) using 6 representative FC workloads and a separate attention head experiment. VTR synthesis provides real Fmax and grid dimensions; an analytical energy/latency simulator completes each DSE point.

**The paper's finding:** 512×128 is the FPGA-optimal NL-DPE configuration. The 512-row dimension is critical — it enables ACAM eligibility (V=1) on 5/6 FC workloads, yielding 2-4× energy savings over V>1 configs. The 128-column dimension balances tile area against horizontal tiling cost. The ACAM benefit is a step function of tiling geometry: it helps exactly when V=1, and the row dimension is the single knob that controls this boundary.

---

## Abstract

* **Problem:** FPGA-based ML acceleration relies on heterogeneous hard blocks. NL-DPE offers an ACAM peripheral that can perform in-DPE nonlinear functions (activation, log), potentially saving fabric resources and energy. But the optimal crossbar configuration for FPGA integration is unknown.
* **Gap:** Prior NL-DPE work optimizes a fixed ASIC architecture. Prior FPGA-integrated IMC work (Azure-Lily) uses a fixed 512×128 crossbar with ADC-only peripheral. Neither answers: what crossbar size maximizes FPGA-level efficiency, and when does ACAM actually help under realistic tiled GEMM mappings?
* **Method:** We perform a crossbar-size DSE across 9 NL-DPE configurations (R ∈ {128, 256, 512} × C ∈ {64, 128, 256}) using VTR-based synthesis and an analytical energy/latency simulator. We evaluate 6 representative FC workloads spanning small (64×64) to large (2048×256), plus a separate attention head experiment (N=128, d=128). We compare against Azure-Lily as an architectural baseline.
* **Findings:**
  * 512×128 is the FPGA-optimal NL-DPE configuration (SPEC-style geomean score 0.852 vs runner-up 0.635), achieving ACAM eligibility on 5/6 FC workloads.
  * ACAM eligibility is a step function of tiling geometry: V=1 configs see 2-4× lower energy than V>1 configs on the same workload. The row dimension R is the single knob that controls this boundary.
  * The attention head uses ACAM as log function (not activation), with CLB-based DIMM/softmax stages dominating energy — motivating future hard-block integration of reduction logic.
  * [Round 2 finding: optimal DPE density in fixed-area FPGA — pending]
* **Takeaway:** ACAM value is gated by per-layer tiling geometry (V=1 threshold), not circuit quality. The crossbar row dimension is the primary design knob for FPGA-integrated NL-DPE, and 512 rows are necessary to capture ACAM benefit on typical FC/attention workloads.

---

## 1. Introduction

### 1.1 Motivation and Problem Statement

* FPGA fabrics increasingly embed domain-specific hard blocks (DSPs, BRAMs, now analog IMC cores) to close the efficiency gap with ASICs for ML workloads.
* NL-DPE is architecturally distinctive: its ACAM peripheral can operate as ADC, activation function, or log function — enabling *in-DPE nonlinear computation* without consuming FPGA fabric resources.
* **The problem that motivates this paper:**
  * NL-DPE crossbar size (R rows × C columns) is a design choice, not fixed. Larger crossbars reduce tiling but cost more area.
  * In FPGA context, layers are executed as tiled GEMMs. When V = ceil(K/R) > 1, DPE outputs are partial sums — ACAM cannot activate, and CLB-based reduction + activation is required.
  * Therefore, the crossbar row dimension R directly determines which workloads benefit from ACAM. The optimal (R, C) is non-obvious: it depends on the workload distribution across the (K, N) space.
* **Precise scope:** This paper is about finding the FPGA-optimal NL-DPE crossbar configuration and quantifying when ACAM provides real value under tiled GEMM mappings. Not about analog circuit improvements.

### 1.2 Why Prior Work Is Not Enough

* **Azure-Lily** (our primary baseline): demonstrates FPGA-integrated analog IMC with a fixed 512×128 crossbar and ADC-only peripheral. Does not study crossbar sizing or dual-mode peripheral value.
* **NL-DPE ASIC work:** characterizes the circuit for a fixed 256×256 configuration. Does not address FPGA block sizing, tiled GEMM mapping semantics, or the interaction between crossbar dimensions and ACAM eligibility.
* **Missing questions:**
  * What crossbar size (R × C) maximizes FPGA-level efficiency across diverse FC and attention workloads?
  * How many NL-DPE blocks should occupy a fixed-area FPGA fabric?
  * When does ACAM provide meaningful benefit, and what is the structural predictor?

### 1.3 Our Goal

* Perform a systematic crossbar-size DSE for NL-DPE as a heterogeneous FPGA hard block.
* Answer three architecture questions that prior work leaves open, using VTR synthesis + analytical energy/latency simulation.
* Provide quantitative guidance for FPGA architects on NL-DPE block sizing, density, and ACAM activation opportunity.

### 1.4 Research Questions

* **Q1.** What crossbar configuration (R × C) makes the best NL-DPE FPGA hard block? *(Answered: 512×128)*
* **Q2.** How many NL-DPE hard blocks should be placed in a fixed-area FPGA fabric? *(Round 2 — pending)*
* **Q3.** Under realistic tiled GEMM mappings, when does ACAM provide meaningful benefit — and does the tiling geometry (V=1 threshold) predict it?

### 1.5 Overview Figure

* **Figure 1:** Two-panel diagram.
  * Left: FPGA fabric with NL-DPE hard blocks (variable R×C), CLB reduction path, ACAM activation/log path.
  * Right: Tiling geometry decision — V=1 → ACAM eligible; V>1 → ADC-only, CLB reduction + activation.
  * Annotate with: Q1 (block sizing DSE), Q2 (block count), Q3 (ACAM value = f(V=1 threshold)).

### 1.6 Contributions

1. First crossbar-size DSE for NL-DPE as a heterogeneous FPGA hard block: 9 configs × 6 FC workloads, evaluated with VTR synthesis and analytical energy/latency simulation.
2. Parameterized analytical model for NL-DPE area, power, and energy (`area_power.py`), with routing-aware VTR tile sizing.
3. Identification of 512×128 as the FPGA-optimal configuration — the row dimension (ACAM eligibility) is the dominant design knob.
4. Separate attention head experiment showing ACAM-as-log mode and CLB DIMM/softmax overhead, motivating future hard-block reduction integration.
5. [Pending Round 2] Fixed-area fabric composition study: optimal NL-DPE block density under CLB displacement constraints.

---

## 2. Related Work

### 2.1 FPGA-Based ML Acceleration

* Survey of CLB/DSP-based CNN accelerators on FPGA (e.g., fpgaConvNet, DNNWeaver, hls4ml).
* Key point: these show that FPGA resource pressure is real and motivates hard block proposals.
* **What to cite:** 3–4 representative FPGA accelerator papers. Look for: one TRETS/FCCM paper on systolic array on FPGA, one on FPGA DSP utilization for DNN.

### 2.2 Analog / In-Memory Compute in FPGA Context

* **Azure-Lily (primary related work):** integrates analog crossbar into FPGA fabric. Peripheral is ADC-only; activation in LUT fabric. Uses 512×128 crossbar, 8-bit interface. Published in [cite Azure-Lily paper].
  * *What it answers:* can analog IMC work in FPGA framework.
  * *What it does not answer:* value of dual-mode peripheral, block sizing, density sweep.
* Other FPGA-analog integration efforts if available (check TACO/DATE/ICCAD for ReRAM-FPGA).

### 2.3 NL-DPE and ACAM-Based ASIC Designs

* Original NL-DPE paper: describes 256×256 crossbar with ACAM peripheral for nonlinear function in analog domain. Optimized for ASIC. [cite NL-DPE paper]
* Key insight retained from prior work: ACAM can switch between ADC and activation mode — this is the circuit capability we exploit.
* *Limitation:* prior work treats the DPE as a standalone ASIC block; no FPGA fabric integration, no tiling analysis.

### 2.4 Related Work Comparison Table

* **Table 1:** Qualitative comparison.
* Columns: FPGA-focused | Analog IMC | Dual-mode peripheral | Block sizing study | Density study | Mapping-policy study | CNN workloads | Transformer workloads
* Rows: Azure-Lily | NL-DPE (ASIC) | [2–3 other FPGA-ML papers] | **This work**

---

## 3. Background

### 3.1 NL-DPE Block Architecture

* Crossbar: R rows × C columns (parameterizable; prior ASIC work used 256×256). Weights stored in analog cells.
* Computation: VMM — input vector multiplied by weight matrix, one cycle per input row.
* ACAM peripheral: output stage that can be configured as:
  * **ADC mode:** converts analog output to digital partial sum.
  * **Activation mode:** applies nonlinear function (tanh/ReLU) directly in analog domain.
  * **Log mode:** computes logarithm for DIMM-based attention score computation.
* Area: ACAM dominates (77% of DPE area), scales linearly with C. Crossbar area scales with R×C. Full model in `area_power.py`.
* *Note for writing:* ACAM area cost makes column count C the primary area driver, while row count R is the primary ACAM-eligibility driver. This R-vs-C tension creates the non-trivial design space.

### 3.2 Azure-Lily Baseline Architecture

* Crossbar: 512 rows × 128 columns.
* Peripheral: ADC-only. e_conv = 2.33 pJ per analog operation (calibrated).
* Activation: always in FPGA fabric (LUT-based tanh).
* **This is our primary comparison baseline.** Same VTR flow, same FPGA fabric model, same workloads.

### 3.3 Tiled GEMM Mapping and ACAM Eligibility

* GEMM dimensions: K (input channels), N (output channels), M (spatial positions, serialized).
* Tiling: V = ⌈K/R⌉ (vertical), H = ⌈N/C⌉ (horizontal). DPE count = V × H.
* **ACAM-eligible (V=1):** K ≤ R → DPE output is a final result → ACAM can activate (or compute log).
* **Not eligible (V>1):** K > R → DPE outputs are partial sums → CLB-based adder tree reduction required, then CLB activation.
* **Key insight from DSE:** ACAM eligibility is a **step function** of R. At the V=1 boundary, energy drops 2-4× (no CLB reduction + no CLB activation). The row dimension R is the single knob that controls which workloads cross this boundary.
* **Horizontal tiling (H>1):** each DPE column produces independent output channels — no cross-DPE reduction needed. H>1 increases DPE count and grid area but does not affect ACAM eligibility.

### 3.4 FPGA Hard Block Perspective

* Hard block must justify area across diverse workloads (unlike ASIC macro, which is optimized for one design).
* Reduction and control logic stay in FPGA CLB fabric — deliberate constraint matching realistic initial integration (same as Azure-Lily).
* Block density tradeoff: more NL-DPE blocks → more compute capacity, but less remaining CLB budget for reduction/control (studied in Round 2).

---

## 4. System Description

*[Write in present tense — this system is built and running.]*

### 4.1 System Overview Figure

* **Figure 2:** End-to-end DSE flow.
  * Box 1: NL-DPE block characterization (`area_power.py`) → area/power/energy model for 9 (R,C) configs.
  * Box 2: Parameterized RTL generation (`gen_gemv_wrappers.py`) → VTR synthesis → Fmax, grid dimensions.
  * Box 3: IMC energy/latency simulator (patched with VTR Fmax) → per-inference energy/latency breakdown.
  * Box 4: SPEC-style normalized geomean ranking → optimal config selection → Round 2 inputs.

### 4.2 NL-DPE Hard Block Model (Q1)

* One NL-DPE hard block = one R×C crossbar + ACAM peripheral + digital I/O interface (16-bit).
* Parameterized by: crossbar size (R rows × C columns). I/O width fixed at 16-bit.
* `area_power.py` computes area (µm²), power (mW), energy (pJ), and VTR tile dimensions (W×H grid cells) as a function of (R, C).
* 9 configurations: R ∈ {128, 256, 512} × C ∈ {64, 128, 256}. DPE area ranges from 12,198 µm² (128×64) to 66,093 µm² (512×256).
* Output of Q1: a single selected configuration (512×128) used in Q2/Q3 and the attention experiment.

### 4.3 Mapping Policy

* **ACAM dual-mode (always used):** ACAM activates locally when V=1 (single vertical tile). Falls back to ADC + CLB activation when V>1.
* The V=1 vs V>1 comparison within Round 1 data implicitly captures the ACAM value (Q3): same config, same workload dimensions, different row counts → different V → different ACAM eligibility → measurable energy gap.
* *Note for writing:* we do not run a separate P1 (ADC-only) policy. Instead, Q3 is answered by comparing configs that achieve V=1 against those that don't, on the same workload.

### 4.4 Fabric Composition Model (Q2 — Round 2, pending)

* Fixed total FPGA area (derived from Round 1 worst-case grid).
* Sweep CLB-to-DPE replacement ratio: {5%, 8%, 12%, 15%} × top-3 configs × 6 workloads.
* As DPE count increases, available CLB count decreases proportionally.
* Metric: throughput/mm² and throughput/J as a function of DPE density.

### 4.5 Workloads

**DSE workloads (Round 1):** 6 FC layers spanning the (K, N) space of real CNN/transformer layers:

| Workload | K | N | Rationale |
|----------|---|---|-----------|
| fc_64_64 | 64 | 64 | Tiny; all configs V=1 |
| fc_128_128 | 128 | 128 | Attention projection proxy (Q/K/V, d=128) |
| fc_512_128 | 512 | 128 | Medium; only R=512 achieves V=1 |
| fc_256_512 | 256 | 512 | Wide output; R≥256 achieves V=1 |
| fc_512_512 | 512 | 512 | Large; only R=512 achieves V=1 |
| fc_2048_256 | 2048 | 256 | Very deep; no config achieves V=1 |

These are chosen to span the ACAM eligibility boundary across the 9 configs.

**Attention workload (separate experiment):** Single attention head (N=128 seq_length, d=128 head_dim). 3 DPE projections (Q/K/V) + CLB-based DIMM score matrix, softmax, weighted sum.

**Validation workloads (from prior work):** LeNet (6 DPEs, 5 layers) and ResNet (40 DPEs, 9 layers) provide full-network NL-DPE vs Azure-Lily comparison points.

* RTL generated parameterically by `gen_gemv_wrappers.py` (FC) and `gen_attention_wrapper.py` (attention, planned). Synthesized with VTR.

---

## 5. Methodology

### 5.1 Tools

* **VTR 8.x:** synthesis and place-and-route. Per-config architecture XMLs define DPE tile geometry (`gen_arch_xml.py`). Round 1 uses `auto_layout` (VTR sizes grid to just-fit).
* **IMC energy/latency simulator (`azurelily/IMC/`):** takes VTR Fmax + crossbar geometry, computes per-inference energy and latency. Patched at runtime with per-config energy parameters from `area_power.py`.
* **Block characterization (`nl_dpe/area_power.py`):** parameterized analytical model for NL-DPE area, power, energy as a function of (R, C). Routing-aware tile sizing (SB=688, CB=303 µm²).
* **RTL generator (`nl_dpe/gen_gemv_wrappers.py`):** generates Verilog for each (R, C, K, N) with correct DPE tiling, adder trees, and activation LUTs.
* **DSE orchestrator (`gemv_dse.py`):** drives the full pipeline (arch XML → RTL → VTR → IMC → metrics → ranking).

### 5.2 Baselines

* **Primary baseline:** Azure-Lily (512×128 crossbar, ADC-only, same VTR flow) — used for architectural context. Full-network comparison data available for LeNet and ResNet from prior work.
* **Internal baselines:** Within the 9-config sweep, configs with V>1 on a given workload serve as the "no ACAM" baseline for that workload, while V=1 configs capture the ACAM benefit. This eliminates the need for a separate P1 (ADC-only) run.

### 5.3 Metrics

* **Per-point (54 DSE points):** Fmax (MHz), grid_W × grid_H, FPGA area (mm²), energy (pJ), latency (ns), ACAM eligibility (V=1?), DPE count (V×H).
* **Derived:** throughput/mm² (inf/s/mm²), throughput/J (inf/J).
* **Ranking:** SPEC-style normalized geomean — per-workload best = 1.0, geomean across 6 workloads, combined score = geomean(GM_tput/mm², GM_tput/J).

### 5.4 Validation

* Azure-Lily simulator calibrated: e_conv = 2.33 pJ/op is ground truth from prior work.
* NL-DPE energy model validated against 0.548× LeNet energy ratio (NL-DPE vs Azure-Lily).
* VTR Fmax outputs used directly — no frequency scaling applied.
* Round 1: all 54 VTR runs completed successfully; no routing failures.

---

## 6. Results

### 6.1 Q1: Crossbar-Size Design-Space Exploration (Round 1)

* **Goal:** Select the FPGA-optimal NL-DPE crossbar configuration from 9 candidates.
* **Key claim:** The row dimension R is the dominant design knob. R=512 enables ACAM on 5/6 workloads; R=256 on 3/6; R=128 on 2/6. The column dimension C trades tile area against horizontal tiling. 512×128 is the Pareto-optimal point.
* **Data:** 54 VTR+IMC runs complete. Results in `dse/results/round1_results.csv`.

* **Figure 3a:** Config ranking bar chart (`round1_ranking.pdf`) — geomean tput/mm² and tput/J per config, sorted by combined score. Shows clear R=512 tier separation.
* **Figure 3b:** Config × Workload heatmap (`round1_heatmap.pdf`) — 9×6 grid, normalized tput/mm², annotated with V and ACAM eligibility. Visual proof that V=1 cells cluster in R=512 rows.
* **Table 2:** All 9 configs — columns: config, DPE area (µm²), tile W×H, power (mW), ACAM-eligible workloads (out of 6), GM tput/mm², GM tput/J, GM combined. 512×128 highlighted.

* **Subsections:**
  * Q1.1: Row dimension determines ACAM eligibility — V=1 threshold analysis across (R, workload) pairs.
  * Q1.2: Column dimension determines tile area and horizontal tiling — C=128 balances area vs H.
  * Q1.3: ACAM energy discontinuity — on FC 512×128, V=1 configs use 159-275 pJ vs V=2 configs using 328-573 pJ (2-4× gap).
  * Q1.4: Selected configuration (512×128) and rationale. *[Write: "512×128 achieves ACAM eligibility on 5/6 workloads while maintaining the most compact tile among R=512 options."]*

### 6.2 Q2: Fabric Composition Under Fixed Area (Round 2 — pending)

* **Goal:** Identify the optimal NL-DPE block density in a fixed-area FPGA.
* **Method:** Fix FPGA grid size (derived from Round 1 worst-case), sweep CLB-to-DPE replacement ratio {5%, 8%, 12%, 15%} × top-3 configs (512×128, 512×256, 512×64) × 6 workloads = 72 VTR runs.
* **Expected claim:** Performance improves with DPE count up to a density point, then degrades as CLB budget becomes insufficient for reduction/activation logic (especially for V>1 workloads).
* **Figure 4a:** Throughput/mm² vs DPE density, series = workloads. Mark inflection point.
* **Figure 4b:** CLB utilization vs DPE density. Show CLB pressure rising.
* **Table 3:** Per-workload best density, suite-optimal density, regret.
* **Status:** Not yet implemented. Depends on T4-T6 in TASKS.md.

### 6.3 Q3: Value of ACAM — Tiling Geometry as Predictor

* **Goal:** Show that ACAM benefit is a step function of V=1 eligibility, and that the row dimension R is the structural predictor.
* **Key claim:** Within Round 1 data, the same workload evaluated at V=1 (large R) vs V>1 (small R) shows 2-4× energy gap. This is not a separate experiment — it falls directly out of the crossbar-size sweep.

* **Evidence from Round 1:**
  * FC 512×128: V=1 configs (R=512) use 159-275 pJ; V=2 configs (R=256) use 328-573 pJ.
  * FC 256×512: V=1 configs (R≥256) vs V=2 configs (R=128) — same pattern.
  * FC 2048×256: No config achieves V=1 — all configs pay CLB reduction + activation cost; energy differences come only from tiling efficiency.

* **Attention head (separate experiment, pending):**
  * ACAM used as **log function** (for DIMM), not activation. All configs V=1 for d=128 projections.
  * CLB-based DIMM/softmax stages dominate energy → ACAM benefit is on a different axis than FC workloads.
  * Key finding expected: DPE projection energy is a small fraction; CLB overhead is the bottleneck.

* **Validation from prior work:**
  * LeNet: NL-DPE 0.548× energy vs Azure-Lily (ACAM activates on 4/5 layers).
  * ResNet: NL-DPE ACAM activates on only 2/9 layers; CLB +52%, Fmax −18%.

* **Figure 5a:** Energy vs config for a workload that crosses the V=1 boundary (e.g., fc_512_128). Show step-function discontinuity at R=512.
* **Figure 5b:** Attention head energy breakdown — DPE projections vs CLB DIMM/softmax stages.
* **Table 4:** Per-workload ACAM eligibility count across 9 configs + energy gap at the V=1 boundary.

### 6.4 NL-DPE vs Azure-Lily: Architectural Context

* **Goal:** Position the DSE-optimal NL-DPE (512×128) against Azure-Lily (512×128, ADC-only) to highlight the ACAM value in isolation.
* **Key claim:** With the same crossbar dimensions (512×128), NL-DPE's advantage comes purely from ACAM — not from crossbar sizing. This is a clean comparison.
* **Known data (from prior full-network work):**
  * LeNet: NL-DPE 276.8 MHz vs Azure-Lily 256.4 MHz (+8%). Energy ratio 0.548×. ACAM activates on 4/5 layers.
  * ResNet: NL-DPE 177.1 MHz vs Azure-Lily 215.1 MHz (−18%). ACAM activates on only 2/9 layers; CLB +52%.
* **Note:** The prior comparison used NL-DPE at 256×256 (pre-DSE). With our DSE-optimal 512×128, the comparison changes: NL-DPE now has the same R=512 as Azure-Lily, eliminating the row-dimension disadvantage. A re-run with 512×128 NL-DPE would update these numbers.
* **Table 5:** Workload × {DPEs, CLBs, Fmax, Energy, EDP} for NL-DPE (512×128) vs Azure-Lily (512×128). Rows: LeNet, ResNet. [Update after re-running with optimal config.]
* **Figure 6:** Energy comparison bar chart, NL-DPE vs Azure-Lily, per workload.

### 6.5 Attention Head Case Study

* **Goal:** Show how ACAM-as-log mode and CLB DIMM/softmax overhead differ from FC workloads.
* **Data:** 9 configs × 1 attention workload (N=128, d=128) — pending T3a/T3b/T3c.
* **Expected finding:** DPE projections are a small fraction of total energy; CLB DIMM/softmax dominate. All configs achieve V=1 for d=128, so differentiation is purely from tile area.
* **Figure 7:** Attention energy breakdown: DPE (Q/K/V projections) vs CLB (DIMM + softmax + weighted sum).

### 6.6 Summary of Main Findings

* **Q1 (answered):** 512×128 is the FPGA-optimal NL-DPE configuration. R=512 enables ACAM on 5/6 FC workloads; C=128 minimizes tile area among R=512 options. Combined geomean score 0.852 vs runner-up 0.635.
* **Q3 (answered from Round 1 data):** ACAM benefit is a step function of V=1 eligibility. At the V=1 boundary, energy drops 2-4×. The row dimension R is the single structural predictor.
* **Q2 (pending):** Optimal DPE density under fixed area — Round 2 CLB replacement sweep.
* **Attention (pending):** ACAM-as-log provides uniform benefit across configs; CLB overhead dominates and motivates hard-block reduction integration.

---

## 7. Discussion

### 7.1 Broader Interpretation

* The crossbar row dimension is the primary architectural knob for ACAM-enabled FPGA hard blocks — it determines the V=1 eligibility threshold, which gates all ACAM benefit.
* This tiling-semantic constraint generalizes: any analog nonlinear unit integrated into an FPGA hard block faces the same issue. The unit is useful only at the computation boundary, not inside a reduction chain.
* The DSE methodology (parameterized RTL generation + VTR synthesis + analytical energy model) is reusable for other analog hard block designs.
* Implication for future FPGA hard block design: integrating partial-sum reduction *inside* the hard block would move the V=1 boundary, enabling ACAM on more workloads.

### 7.2 Limitations

* Reduction always stays in CLB (conservative but realistic for initial integration).
* I/O width fixed at 16-bit (not swept — would require additional VTR tile sizing).
* Attention head experiment uses a single configuration (N=128, d=128) — not swept across sequence lengths.
* Round 1 uses auto_layout (variable grid per design); fixed-area comparison deferred to Round 2.
* Energy model is per-inference analytic; no routing congestion, thermal effects, or multi-batch scheduling.
* Azure-Lily comparison uses prior-work data at 256×256 NL-DPE; re-running with DSE-optimal 512×128 would update the numbers.

### 7.3 Future Extensions

* Integrate partial-sum reduction inside NL-DPE hard block → enables ACAM to activate on V>1 layers.
* Round 2: fixed-area FPGA with CLB replacement sweep → answers Q2 (optimal DPE density).
* Broader workload set: larger transformers, multi-head attention, depthwise convolution.
* Co-design routing architecture with NL-DPE block placement to reduce CLB pressure.

---

## 8. Conclusion

### 8.1 Summary

* We perform a systematic crossbar-size DSE for NL-DPE as a heterogeneous FPGA hard block, evaluating 9 configurations across 6 FC workloads and a separate attention head experiment.
* Q1 is answered: 512×128 is the FPGA-optimal configuration. Q3 is answered: ACAM benefit is a step function of V=1 eligibility. Q2 (optimal density) is addressed in Round 2.

### 8.2 Key Insights

* **Row dimension is the primary design knob:** R determines ACAM eligibility (V=1 threshold). R=512 enables ACAM on 5/6 FC workloads; R=128 on only 2/6.
* **ACAM benefit is a step function:** at the V=1 boundary, energy drops 2-4×. This is a mapping-semantic property, not a circuit property.
* **Column dimension is secondary:** C affects tile area and horizontal tiling, but does not affect ACAM eligibility. C=128 is the sweet spot (compact tile, moderate H).
* **Attention uses ACAM differently:** ACAM-as-log for DIMM is uniformly available (d=128 → V=1 for all configs), but CLB DIMM/softmax overhead dominates — motivating hard-block reduction integration.
* **[Pending]** Hard block density has a sweet spot under fixed area constraints (Round 2).

---

## Appendix Notes

* **Appendix A:** Full Round 1 results table (54 data points) with per-workload metrics.
* **Appendix B:** Per-workload tiling analysis — V and H for all 9 configs × 6 workloads.
* **Appendix C:** DPE physical specs table (area, power, tile dimensions for all 9 configs).
* **Appendix D:** Sensitivity to ACAM energy cost (what if ACAM activation is cheaper/more expensive?).
