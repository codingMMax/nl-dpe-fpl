# Paper Outline — NL-DPE as a Heterogeneous FPGA Hard Block

---

## Title (SELECTED)

**ACAM-Enabled Heterogeneous FPGA Hard Blocks: Evaluating NL-DPE for CNN and Transformer Inference**

*Rationale: Names the mechanism (ACAM), the context (heterogeneous FPGA), and the workload scope. Avoids vague "evaluation" framing.*

---

## Core Story (Read this before writing any section)

NL-DPE has a unique ACAM peripheral that can perform nonlinear activation inside the analog block — eliminating the need for fabric LUT/DSP activation. This sounds like a clear win. But in FPGA context, most CNN layers require tiling across multiple DPE blocks, meaning outputs leaving each DPE are *partial sums*, not final activations. ACAM can only activate when the output is final at block exit — i.e., when the layer maps to a single DPE tile (V=1, H=1).

**The core tension:** NL-DPE's wider column dimension (256 vs Azure-Lily's 128) eliminates horizontal tiling for N≤256 layers, always giving H=1. But NL-DPE's shallower row dimension (256 vs 512) forces more vertical tiles for K-large layers, giving V>1. When V>1, ACAM falls back to ADC-only mode, and activation is pushed to CLB fabric — negating the key advantage.

**The paper's finding:** Whether NL-DPE's ACAM provides real FPGA-level value depends entirely on workload tiling behavior, not just circuit quality. Small/shallow models (LeNet) see clear benefit; large/deep models (ResNet conv3-8, VGG11 conv5-8) see ACAM bypassed on the compute-heavy layers.

---

## Abstract

* **Problem:** FPGA-based ML acceleration relies on heterogeneous hard blocks. NL-DPE offers an ACAM peripheral that can perform in-DPE activation, potentially saving fabric resources and energy. Whether this advantage survives FPGA integration is unclear.
* **Gap:** Prior NL-DPE work optimizes an ASIC architecture. Prior FPGA-integrated IMC work (Azure-Lily) uses a passive ADC peripheral. Neither answers: how should NL-DPE be sized, how many should be in the fabric, and when does its ACAM actually help under realistic tiled GEMM mappings?
* **Method:** We evaluate NL-DPE as an FPGA hard block using VTR-based synthesis and an analytical energy/latency simulator across LeNet, ResNet, VGG11, and BERT-Tiny. We compare against Azure-Lily as a baseline FPGA IMC design.
* **Findings (fill in numbers before submission):**
  * NL-DPE achieves [X]% lower energy and [Y]% lower latency than Azure-Lily on LeNet, where ACAM activates on 4/5 layers.
  * On ResNet, NL-DPE's ACAM activates on only 2/9 layers; CLB pressure rises by 52% due to deeper vertical stacking; Fmax drops 18%.
  * A moderate NL-DPE block density of [Z] blocks per fixed-area fabric maximizes EDP across the workload suite.
* **Takeaway:** ACAM dual-mode provides real FPGA-level value only when layer shape permits single-tile mapping. This is a mapping-semantic constraint, not a circuit limitation — and it directly determines whether NL-DPE or a simpler ADC-based IMC is a better FPGA hard block choice.

---

## 1. Introduction

### 1.1 Motivation and Problem Statement

* FPGA fabrics increasingly embed domain-specific hard blocks (DSPs, BRAMs, now analog IMC cores) to close the efficiency gap with ASICs for ML workloads.
* NL-DPE is architecturally distinctive: its ACAM peripheral can operate as either an ADC or an activation function unit, enabling *in-DPE activation* without consuming FPGA fabric resources.
* **The problem that motivates this paper:**
  * In FPGA context, CNN layers are executed as tiled GEMMs across multiple DPE blocks.
  * When a layer spans multiple DPEs (V>1 or H>1), the output of each DPE is a partial sum that must be reduced in FPGA fabric before activation.
  * ACAM can only activate when its output is a *final* layer output, not a partial sum.
  * Therefore, the value of ACAM's dual-mode is not a circuit property — it is a mapping-semantic property.
* **Precise scope:** This paper is about NL-DPE as an FPGA hard block. Not about proving NL-DPE is a better ASIC, not about analog circuit improvements.

### 1.2 Why Prior Work Is Not Enough

* **Azure-Lily** (our primary baseline): demonstrates FPGA-integrated analog IMC, but uses a purely ADC-based peripheral — activation always occurs in FPGA fabric. Does not study what happens when the peripheral itself can activate.
* **NL-DPE ASIC work:** characterizes the circuit but optimizes for a fixed ASIC mapping. Does not address heterogeneous FPGA block sizing, density, or tiled GEMM mapping semantics.
* **Missing questions:**
  * What is the FPGA-optimal NL-DPE block configuration (crossbar size, I/O width)?
  * How many NL-DPE blocks should occupy a fixed-area FPGA fabric?
  * Under realistic tiled GEMM mappings, which layers actually benefit from ACAM dual-mode?

### 1.3 Our Goal

* Evaluate NL-DPE as a heterogeneous FPGA hard block, using Azure-Lily as the reference IMC design.
* Answer three architecture questions that prior work leaves open.
* Provide guidance for FPGA architects deciding whether and how to integrate NL-DPE-like blocks.

### 1.4 Research Questions

* **Q1.** What crossbar size and I/O configuration make the best NL-DPE FPGA hard block?
* **Q2.** How many NL-DPE hard blocks should be placed in a fixed-area FPGA fabric?
* **Q3.** Under realistic tiled GEMM mappings, when does ACAM dual-mode provide meaningful benefit over ADC-only operation?

### 1.5 Overview Figure

* **Figure 1:** Two-panel diagram.
  * Left: FPGA fabric with NL-DPE hard blocks, CLB reduction path, ACAM local activation path.
  * Right: Decision tree for ACAM mode — V=1 and H=1 → ACAM activates; otherwise → ADC-only, CLB activation.
  * Annotate with: Q1 (block sizing), Q2 (block count), Q3 (activation path decision).

### 1.6 Contributions

1. First evaluation of NL-DPE as a heterogeneous FPGA hard block, benchmarked against Azure-Lily as a baseline FPGA IMC design.
2. Hard block design study: power/area/timing characterization across crossbar sizes and I/O widths, implemented as a parameterized analytical model.
3. Fixed-area fabric composition study: identifies the NL-DPE block count that maximizes FPGA-level EDP across LeNet, ResNet, VGG11, and BERT-Tiny.
4. Mapping-semantic analysis of ACAM dual-mode: shows that ACAM activation benefit is gated by per-layer tiling geometry, not just circuit efficiency — and quantifies the per-workload opportunity fraction.

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

* Crossbar: 256 rows × 256 columns, weights stored in analog cells.
* Computation: VMM — input vector multiplied by weight matrix, one cycle per input row.
* ACAM peripheral: output stage that can be configured as:
  * **ADC mode:** converts analog output to digital partial sum. Energy: ~31.12 pJ/row.
  * **Activation mode:** applies nonlinear function (tanh/ReLU) directly in analog domain. Energy: ~31.12 + 43.89 pJ/row (ACAM cost added).
  * *Note for writing:* explain that ADC mode always runs; ACAM activation mode adds the 43.89 pJ — so dual-mode is not free even when it saves CLB area.

### 3.2 Azure-Lily Baseline Architecture

* Crossbar: 512 rows × 128 columns.
* Peripheral: ADC-only. e_conv = 2.33 pJ per analog operation (calibrated).
* Activation: always in FPGA fabric (LUT-based tanh).
* **This is our primary comparison baseline.** Same VTR flow, same FPGA fabric model, same workloads.

### 3.3 Tiled GEMM Mapping and ACAM Activation Opportunity

* GEMM dimensions: K (input channels/kernel), N (output channels), M (spatial positions, serialized).
* Tiling: V = ⌈K/crossbar\_rows⌉, H = ⌈N/crossbar\_cols⌉.
* **Single-tile case (V=1, H=1):** DPE output is a final layer output → ACAM can activate.
* **Multi-tile case (V>1 or H>1):** DPE output is a partial sum → must reduce in CLB → ACAM in ADC mode.
* **Key consequence:** NL-DPE (256×256) always achieves H=1 for N≤256 (all CNN layers in our workloads). But V>1 for K>256 layers, which are the compute-heavy layers in deep CNNs.
* Define: **local activation opportunity ratio** = fraction of total GEMM compute (by M×K) that occurs in single-tile layers.

### 3.4 FPGA Hard Block Perspective

* Hard block must justify area across diverse workloads (unlike ASIC macro, which is optimized for one design).
* Reduction and control logic stay in FPGA CLB fabric — this is a deliberate constraint in our model, matching the realistic initial integration point (same as Azure-Lily).
* Block density tradeoff: more NL-DPE blocks → more compute capacity, but less remaining CLB budget for reduction/control.

---

## 4. System Description

*[Write in present tense — this system is built and running.]*

### 4.1 System Overview Figure

* **Figure 2:** End-to-end evaluation flow.
  * Box 1: NL-DPE block characterization → area/power/timing model (Python script, Q1 input).
  * Box 2: RTL designs (LeNet, ResNet, VGG11, BERT-Tiny) → VTR synthesis → resource counts + Fmax.
  * Box 3: Analytical energy/latency simulator → per-layer energy breakdown using VTR outputs.
  * Box 4: Results for Q1/Q2/Q3.

### 4.2 NL-DPE Hard Block Model (Q1)

* One NL-DPE hard block = one 256×256 crossbar + ACAM peripheral + digital I/O interface.
* Parameterized by: crossbar size (rows × cols), input data width, output data width.
* Characterization data available for multiple configurations; `area_power.py` computes area, power, and timing as a function of these parameters.
* Fixed: ACAM row count (changing it requires full re-characterization — out of scope).
* Output of Q1: a single selected configuration used in all Q2/Q3 experiments.

### 4.3 Mapping Policies

* **P1 (ADC-only):** ACAM always operates in ADC mode. Activation always in CLB fabric. This is the Azure-Lily-equivalent mapping applied to NL-DPE.
* **P2 (ACAM dual-mode):** ACAM activates locally when the layer maps to a single DPE tile (V=1, H=1). Falls back to ADC + CLB activation for multi-tile layers.
* *Note for writing:* P1 gives a lower-bound on ACAM value; P2 gives the upper-bound assuming perfect local activation detection. No P3 (e.g., partial activation) — keeps the experiment clean.

### 4.4 Fabric Composition Model (Q2)

* Fixed total FPGA area (matched to Azure-Lily baseline fabric).
* Sweep NL-DPE block count: [0 (pure CLB baseline), low, medium, high].
* As NL-DPE count increases, available CLB count decreases proportionally.
* Routing architecture and DSP count held constant.
* Metric: EDP and latency as a function of block count, per workload.

### 4.5 Workloads and RTL

* **LeNet:** 5 layers, 6 NL-DPE tiles. Small model; ACAM activates on 4/5 layers (conv1, conv2, full2, full3 are single-tile; full1 is V2_H1 due to K=400>256). Overhead-sensitive case.
* **ResNet:** 9 layers, 40 NL-DPE tiles. ACAM activates on 2/9 layers (conv1: K=9, conv9: K=224). Compute-heavy layers (conv3-8: K=1008–2016) require V4 or V8 stacking — ACAM bypassed.
* **VGG11:** 9 layers, 146 NL-DPE tiles. [Fill in ACAM opportunity ratio after VTR run.]
* **BERT-Tiny:** transformer-like workload. QKV projection and FC layers have different K/N shapes — likely higher local activation opportunity for small attention heads. [In progress.]
* RTL for all workloads implemented in Verilog, synthesized with VTR using NL-DPE and Azure-Lily architecture XMLs.

---

## 5. Methodology

### 5.1 Tools

* **VTR 8.x:** synthesis and place-and-route. Architecture XMLs define NL-DPE and Azure-Lily hard block geometry, area, timing constants.
* **Analytical simulator (`run_imc_with_vtr_freq.py`):** takes VTR Fmax + DPE/CLB/BRAM counts, computes per-layer energy and latency using NL-DPE/Azure-Lily energy models.
* **Block characterization script (`nl_dpe/area_power.py`):** parameterized model for NL-DPE area, power, fmax as a function of crossbar size and I/O width. [Under construction — Q1 depends on this.]

### 5.2 Baselines

* **Primary baseline:** Azure-Lily (same FPGA fabric, same VTR flow, same workloads) — used for all Q2/Q3 comparisons.
* **Secondary baseline:** NL-DPE under P1 (ADC-only) — isolates the pure IMC advantage before adding ACAM benefit.
* **Tertiary baseline (Q2 only):** 0 NL-DPE blocks (pure CLB FPGA) — anchors the "no analog" point.

### 5.3 Metrics

* **Block-level (Q1):** area (µm²), power (mW), fmax (MHz), effective throughput per area.
* **FPGA-level (Q2/Q3):** inference latency (ms), total energy (mJ), EDP, Fmax, CLB utilization, DPE utilization.
* **ACAM-specific (Q3):** local activation opportunity ratio (per workload), energy saved by P2 vs P1, CLB activation count reduction under P2.

### 5.4 Validation

* Azure-Lily simulator calibrated: e_conv = 2.33 pJ/op is ground truth from prior work.
* NL-DPE energy model: 31.12 pJ/row (VMM) + 43.89 pJ/row (ACAM activation), validated analytically against 0.548x LeNet energy ratio (see Section 6.3).
* VTR frequency outputs used directly — no frequency scaling applied.

---

## 6. Results

### 6.1 Q1: NL-DPE Hard Block Design-Space Exploration

* **Goal:** Select the FPGA-optimal NL-DPE block configuration. Input: `area_power.py` characterization data.
* **Key claim to make:** The largest crossbar is not the FPGA-optimal choice — beyond a certain size, area overhead dominates and mapping efficiency plateaus because most CNN layers have K<512 anyway.
* **Figure 3a:** Crossbar size vs block area. Show that area scales super-linearly. Mark the 256×256 point.
* **Figure 3b:** Crossbar size vs Fmax. Larger crossbars reduce achievable frequency due to longer analog paths.
* **Figure 3c:** Area-normalized effective throughput for ResNet and BERT-Tiny. Show sweet spot.
* **Table 2:** Candidate configurations — columns: crossbar size, I/O width, area, power, Fmax, throughput/area. Final selected row highlighted.
* **Subsections:**
  * Q1.1: Impact of crossbar size on area and timing.
  * Q1.2: Impact of I/O width (16-bit vs alternatives) on interface cost and CLB routing demand.
  * Q1.3: Selected configuration and rationale. *[Write: "We select 256×256 with 16-bit interface because..."]*

### 6.2 Q2: Fabric Composition Under Fixed Area

* **Goal:** Identify the NL-DPE block count that maximizes FPGA-level efficiency across the workload suite.
* **Key claim to make:** Performance improves with NL-DPE block count up to a moderate density point, then saturates or degrades as remaining CLB budget becomes insufficient for reduction and control.
* **Known data points:**
  * LeNet needs 6 NL-DPE tiles; ResNet needs 40; VGG11 needs 146. Different models need different minimum block counts.
  * The "best" block count is suite-level, not per-workload — a table should show the per-workload optimal and the suite-optimal, and the regret of the latter.
* **Figure 4a:** Latency vs NL-DPE block count, series = workloads. Mark the inflection point.
* **Figure 4b:** EDP vs NL-DPE block count. Show moderate density is best.
* **Figure 4c:** CLB utilization vs block count. Show CLB pressure rising as blocks displace fabric.
* **Table 3:** Per-workload best block count, best EDP, suite-optimal block count, regret vs per-workload best.
* **Subsections:**
  * Q2.1: NL-DPE block count vs throughput — show scaling behavior.
  * Q2.2: NL-DPE block count vs EDP — show moderate density is best.
  * Q2.3: CLB pressure — explain why too many blocks hurts large models.

### 6.3 Q3: Value of ACAM Dual-Mode

* **Goal:** Quantify when and by how much ACAM dual-mode (P2) outperforms ADC-only (P1), and connect the benefit to per-layer tiling geometry.
* **Key claim to make:** ACAM dual-mode provides meaningful energy savings only when local activation opportunity ratio is high. For LeNet (ratio ≈ 4/5 layers by count), P2 reduces activation energy significantly. For ResNet (ratio ≈ 2/9), ACAM activates only on compute-light layers (conv1, conv9); the compute-heavy layers (V8_H1) bypass ACAM entirely.
* **Anchor finding (validated):** NL-DPE achieves 0.548x energy ratio vs Azure-Lily on LeNet, driven by ACAM activation savings on 4/5 layers plus elimination of activation layer FSMs (reflected in +8% Fmax, fewer controller states).
* **Figure 5a:** P1 vs P2 energy, per workload. Show that LeNet sees large gap; ResNet sees small gap.
* **Figure 5b:** Local activation opportunity ratio per workload (bar chart). Explains Figure 5a pattern directly.
* **Figure 5c:** Per-layer ACAM activation decision for ResNet — show which layers use ACAM vs bypass. Visual proof of the tiling-semantic argument.
* **Table 4:** P1 vs P2 — workload | policy | energy | latency | EDP | ACAM activation ratio | CLB activation ops saved.
* **Subsections:**
  * Q3.1: ACAM activation opportunity by workload — define and compute local activation ratio.
  * Q3.2: Energy and latency impact of P2 vs P1 — quantify the benefit where it exists.
  * Q3.3: Why ResNet and VGG11 see limited benefit — the V>1 tiling argument. *[Write: "Conv3-8 of ResNet require V=4–8 vertical tiles because K=1008–2016 > 256 rows. Each DPE outputs a partial sum; ACAM cannot activate. This is not a circuit limitation but a mapping constraint intrinsic to the FPGA integration model."]*

### 6.4 NL-DPE vs Azure-Lily: Head-to-Head Summary

* **Goal:** Provide a direct comparison across all four metrics for both architectures. This is the table reviewers will look at first.
* **Key claim to make:** NL-DPE beats Azure-Lily on small/shallow models (LeNet) due to ACAM savings and simpler pipeline. Azure-Lily is competitive on large/deep models (ResNet) because its 512-row crossbar reduces vertical tile count (fewer DPEs, shorter routing, higher Fmax).
* **Known data:**
  * LeNet: NL-DPE 276.8 MHz vs Azure-Lily 256.4 MHz (+8%). NL-DPE 6 DPEs vs 5 (+1). NL-DPE 53 CLBs vs 28 (+89%).
  * ResNet: NL-DPE 177.1 MHz vs Azure-Lily 215.1 MHz (−18%). NL-DPE 40 DPEs vs 35 (+14%). NL-DPE 281 CLBs vs 185 (+52%).
* **Table 5:** Workload × {DPEs, CLBs, Fmax, Latency, Energy, EDP} for NL-DPE (P2) vs Azure-Lily. Rows: LeNet, ResNet, VGG11, BERT-Tiny.
* **Figure 6:** Normalized EDP bar chart, NL-DPE vs Azure-Lily, per workload. Shows crossover between small and large models.

### 6.5 Flexibility / Adaptability Summary

> **STATUS: UNSOLVED** — adaptability regret metric is not yet defined or computed. Either define regret concretely (formula + source data) before submission, or fold this into Section 6.4 as a paragraph about which fixed fabric point generalizes best.

* Tentative goal: show that one NL-DPE block count generalizes across the workload suite without catastrophic per-workload regret.
* Candidate metric: for each workload, compute (EDP_chosen_fabric − EDP_optimal_fabric) / EDP_optimal_fabric. Average across workloads. Low average regret = good generalization.

### 6.6 Summary of Main Findings

* ACAM dual-mode benefit is gated by tiling geometry: only single-tile layers (V=1, H=1) can use local activation.
* NL-DPE outperforms Azure-Lily on LeNet (ACAM activates on 4/5 layers) but trails on ResNet (ACAM activates on 2/9 layers, CLB +52%, Fmax −18%).
* A moderate NL-DPE fabric density balances DPE compute capacity and CLB reduction budget.
* [Fill in Q1 finding after block characterization script runs.]
* [Fill in BERT-Tiny finding after RTL implementation.]

---

## 7. Discussion

### 7.1 Broader Interpretation

* Analog block circuit quality (ACAM, crossbar efficiency) is a necessary but not sufficient condition for FPGA hard block value.
* The tiling-semantic constraint generalizes: any analog activation unit integrated into an FPGA hard block faces the same issue. ACAM is useful only at the boundary of the computation graph, not inside a reduction chain.
* Implication for future FPGA hard block design: consider integrating partial-sum reduction *inside* the hard block before activation, so ACAM can activate more often (see Future Work).

### 7.2 Limitations

* Reduction always stays in CLB (conservative but realistic for initial integration).
* ACAM row count not swept in Q1 (would require new analog characterization).
* BERT-Tiny represents only one class of transformer-like workload.
* NL-DPE timing model currently uses Azure-Lily delay constants as a placeholder — updating with real NL-DPE timing data would change Fmax estimates.
* Simulator simplifications: energy model is per-layer analytic; no routing congestion or thermal effects.

### 7.3 Future Extensions

* Integrate partial-sum reduction inside NL-DPE hard block → enables ACAM to activate on more layers.
* Broader workload set: larger transformers (BERT-Base, GPT-2 small).
* Co-design routing architecture with NL-DPE block placement to reduce CLB pressure.

---

## 8. Conclusion

### 8.1 Summary

* We evaluate NL-DPE as a heterogeneous FPGA hard block and compare against Azure-Lily across LeNet, ResNet, VGG11, and BERT-Tiny.
* We answer three questions: optimal block configuration (Q1), optimal block density (Q2), and mapping-semantic value of ACAM dual-mode (Q3).

### 8.2 Key Insights

* **ACAM dual-mode is conditionally valuable:** it helps when and only when the layer is single-tile. The local activation opportunity ratio is the predictor.
* **Bigger FPGA crossbars are not always better:** NL-DPE's 256-row limit forces more vertical tiles than Azure-Lily's 512-row for K-heavy layers, increasing DPE count, CLB pressure, and routing delay.
* **NL-DPE wins on small models, Azure-Lily is competitive on large models** — the crossover is explained by tiling geometry, not circuit quality.
* **Hard block density has a sweet spot:** too few blocks underutilize the NL-DPE advantage; too many displace the CLB fabric needed for reduction.

---

## Appendix Notes

* **Appendix A:** Additional block configurations from Q1 sweep.
* **Appendix B:** Per-layer energy breakdown for ResNet (shows which layers consume most energy and whether ACAM activated).
* **Appendix C:** BERT-Tiny layer-by-layer tiling analysis and ACAM opportunity ratio.
* **Appendix D:** Sensitivity to ACAM energy cost (what if ACAM activation is cheaper/more expensive?).
