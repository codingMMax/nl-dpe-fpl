# IMC Developer Onboarding Guide

## 1) Scope and Objective

This guide covers only:
- `IMC/`
- `calibration_study/`

It does **not** cover legacy `IMC/`.

High-level objective of this codebase:
- Provide a fast analytical simulator for a heterogeneous IMC+FPGA system.
- Model three IMC architectures from JSON configs:
  - Azure-Lily
  - NL-DPE
  - SRAM-CHA
- Produce:
  - Layer-wise energy/latency estimates
  - End-to-end latency (critical path with cross-layer overlap)
  - Calibration comparisons against the reference event-driven simulator (`nn/`, `models/`).

---

## 2) Codebase Map (What each module does)

### Entry points
- `IMC/test.py`
  - Main simulator runner (LeNet/ResNet/VGG/single-layer/attention).
  - Prints:
    - energy by layer sum
    - energy by breakdown sum
    - latency critical path
    - latency raw sum

- `calibration_study/equivalence_study.py`
  - Calibration/report generator.
  - Runs both simulators (IMC and event-driven reference proxy/exact modes).
  - Generates markdown report (`calibration_study/equivalence_report.md`).

### Core simulator
- `IMC/simulator.py`
  - Wires config + modules:
    - `Config`
    - `MemoryModel`
    - `IMCCore`
    - `FPGAFabric`
    - `Scheduler`
    - `Stats`

- `IMC/imc_core/config.py`
  - Loads architecture and platform constants from JSON.
  - Provides IMC capability factors (`_get_arch_factors`) and frequencies/energy constants.

- `IMC/imc_core/imc_core.py`
  - IMC-core compute model for conv/linear GEMM execution.
  - Computes core+memory+reduction latency and energy.

- `IMC/peripherals/memory.py`
  - BRAM latency and BRAM energy models.
  - Records read/write latency/energy into stats breakdowns.

- `IMC/peripherals/fpga_fabric.py`
  - FPGA-side ops:
    - activation
    - maxpool
    - DSP GEMM / vector add
    - exp/norm utilities for attention.

- `IMC/scheduler_stats/scheduler.py`
  - Layer dispatcher/mapping:
    - conv/linear -> IMC core
    - maxpool/act/softmax/attention peripherals -> FPGA model
  - Produces timing tuple for each layer:
    - `first_output_ns`, `steady`, `events`, `required_upstream_outputs`

- `IMC/scheduler_stats/stats.py`
  - Keeps raw counters:
    - `energy_stats`
    - `energy_breakdown`
    - `latency_raw`
    - `latency_breakdown`
  - Delegates critical-path overlap accounting to `PipelineProfiler`.

- `IMC/scheduler_stats/pipeline_profiler.py`
  - Cross-layer latency orchestration model.
  - Converts per-layer timing tuples to overlap-aware end-to-end critical path.
  - Maintains detailed timeline trace for analysis.

---

## 3) Runtime Interaction Flow (Layer execution path)

For each layer, call chain is:
1. `IMC.run_layer(layer)` (`simulator.py`)
2. `Scheduler.run_layer(layer)` (`scheduler.py`)
3. Operator model executes:
   - `IMCCore.run_gemm(...)` for conv/linear
   - `FPGAFabric.maxpool(...)` for maxpool
   - `FPGAFabric.activation(...)` if needed
   - attention ops mapped to DSP/CLB methods as configured
4. Scheduler records:
   - layer energy via `Stats.record_energy`
   - layer latency via `Stats.record_latency(..., timing=...)`
5. `Stats.record_latency`:
   - stores raw latency (`latency_raw`)
   - forwards timing tuple to `PipelineProfiler.record(...)`
   - stores critical-path contribution in `latency_stats`

At end of run:
- `IMC.finalize_latency_stats()` flushes pending attention parallel groups in profiler.

---

## 4) Energy Model: assumptions, formulas, and functions

### 4.1 Global energy assumptions
- Unit in IMC is **pJ**.
- Data movement energy is BRAM-access based (no explicit off-chip DRAM in this model).
- Compute/peripheral energies are architecture- and resource-based constants.

### 4.2 Memory energy
Function:
- `MemoryModel._calc_energy(bytes)` in `peripherals/memory.py`

Formula:
- `bytes_per_access = floor(bram_width / 8)`
- `num_access = ceil(bytes / bytes_per_access)`
- `E_mem = e_bram_pj_per_access * num_access`

Recorded into breakdown keys:
- `sram_read`
- `sram_write`

### 4.3 IMC core energy (conv/linear)
Function:
- `IMCCore.run_gemm(M, K, N)` in `imc_core/imc_core.py`

Main terms:
- `E_vmm = M * total_imc_tiles * energy_per_vmm()`
- `E_conv = M * total_imc_tiles * energy_per_conversion(active_cols)`
- `E_digital = M * total_imc_tiles * energy_per_digital_post()`
- `E_reduction = clb_reduction_energy(k_tile) * (n_tile * active_cols)`
- `E_read`, `E_write` from `MemoryModel.energy(...)`

Breakdown keys:
- `imc_vmm`
- `imc_conversion`
- `imc_digital_post`
- `clb_reduction`
- `sram_read`, `sram_write`

### 4.4 FPGA peripheral energy
Functions and keys:
- Activation:
  - `FPGAFabric.activation` -> `fpga_activation`
  - `E_act = (M*N) * act_energy_pj_per_op`
- Maxpool:
  - `FPGAFabric.maxpool` -> `clb_compare`, plus BRAM read/write
- DSP GEMM:
  - `FPGAFabric.gemm_dsp` -> `dsp_gemm`, plus BRAM
- Residual vector add:
  - `FPGAFabric.vector_add` -> compute term recorded as `dsp_add`
- Softmax exp/norm:
  - `clb_exp`, `clb_norm_sum`, `clb_norm_inv`, `mul`

### 4.5 What controls total energy most
Typically (model-dependent):
- IMC conversion term (`imc_conversion`) dominates conv-heavy workloads.
- BRAM read/write dominates memory-heavy operators or small-channel maxpool.
- Activation/CLB terms matter when act/maxpool frequency is high.

---

## 5) Latency Model: assumptions, formulas, and scheduling logic

### 5.1 Intra-layer latency (operator-local)

#### Conv/Linear (IMC core + optional act)
Functions:
- `IMCCore.gemm_pipeline_profile(K,N)` and `IMCCore.run_gemm(M,K,N)`
- `Scheduler._merge_pipelined_activation_latency(...)`

Base pipeline shape:
- `t_fill = t_read_row + t_compute_row + t_write_row`
- `t_steady = max(t_read_row, t_compute_row, t_write_row)`
- `lat = t_fill + (M-1)*t_steady`

Where:
- `t_compute_row = max(vmm, conv, digital, reduction)` if `cfg.pipelinable`, else sum.

If activation exists and analog nonlinear is unavailable:
- act row latency is merged into pipeline in scheduler.

#### Maxpool
Function:
- `FPGAFabric.maxpool(...)`

Pipeline shape:
- `t_fill = t_read + t_compare + t_write`
- `t_steady = max(t_read, t_compare, t_write)`
- `lat = t_fill + (events-1)*t_steady`

#### Residual
Function:
- `FPGAFabric.vector_add(...)`
- modeled as read + compute + write aggregate.

### 5.2 Cross-layer pipeline strategy (critical path)

All detailed overlap logic is in:
- `IMC/scheduler_stats/pipeline_profiler.py`

Scheduler provides per-layer tuple:
- `first_output_ns`
- `steady_ns`
- `events`
- `required_upstream_outputs`

#### Token readiness-lite model (not event-queue)
For streaming layers (`events > 1`), profiler builds output ready times using:
- `T_i(1) = T_{i-1}(R_i(1)) + first_output_i`
- `T_i(n) = max(T_i(n-1) + steady_i, T_{i-1}(R_i(n)) + first_output_i)`

Where:
- `R_i(n)` = required upstream token index for downstream token `n`
- computed by `_required_upstream_index(...)` from window geometry:
  - output token -> `(oh, ow)`
  - downstream receptive-field max coordinate -> `(max_h, max_w)`
  - required upstream token derived from row-major index

For non-stream layers (`events <= 1`):
- launch from previous finish or required upstream token time
- finish = launch + layer aggregate latency

#### Critical path accumulation
For each layer:
- `contribution = max(0, finish - current_critical_end)`
- `current_critical_end = max(current_critical_end, finish)`

Final end-to-end latency:
- `sum(latency_stats.values())`

Raw non-overlapped sum:
- `sum(latency_raw.values())`

### 5.3 Why raw and critical can differ a lot
- `latency_raw`: sum of isolated layer latencies.
- `latency_stats`: only non-overlapped contribution to global critical path.
- In strongly overlapped pipelines, many layers contribute near zero to critical path.

---

## 6) How to run and where to inspect results

### 6.1 Run simulator (IMC)
Examples:
```bash
python IMC/test.py --model lenet --imc_file IMC/configs/azure_lily.json
python IMC/test.py --model resnet --imc_file IMC/configs/azure_lily.json
python IMC/test.py --model attention --imc_file IMC/configs/azure_lily.json
```

Console outputs include:
- `Energy total (by layer)` -> `sum(imc.energy_stats.values())`
- `Energy total (by breakdown)` -> `sum(imc.energy_breakdown.values())`
- `Latency total (critical path)` -> `sum(imc.latency_stats.values())`
- `Latency total (raw sum)` -> `sum(imc.latency_raw.values())`

### 6.2 Inspect breakdowns in code
- Energy per layer:
  - `imc.energy_stats`
- Energy component breakdown:
  - `imc.energy_breakdown`
- Raw layer latency:
  - `imc.latency_raw`
- Critical-path layer contribution:
  - `imc.latency_stats`
- Detailed overlap trace:
  - `imc.stats.pipeline_profiler.trace()`

### 6.3 Generate calibration report
```bash
python calibration_study/equivalence_study.py \
  --imc_file IMC/configs/azure_lily.json \
  --out calibration_study/equivalence_report.md
```

Primary report sections:
- assumptions and modeling notes
- cross-layer pipeline tutorial (timestamp walkthrough)
- layer-wise latency comparison (LeNet/ResNet)
- end-to-end totals and ratios

---

## 7) Calibration study structure (`calibration_study/`)

- `operator_model.py`
  - Defines small operator-focused test model used for controlled studies.

- `equivalence_study.py`
  - Runs both new and reference paths.
  - Produces:
    - bucketed energy comparisons (memory/core/peripheral)
    - latency proxy comparisons
    - end-to-end totals
    - tutorial/timeline markdown content

- `equivalence_report.md`
  - Generated artifact for reviews/presentations.

---

## 8) Practical onboarding workflow for a new maintainer

1. Validate baseline:
   - run LeNet and ResNet in `IMC/test.py`
   - confirm energy sum consistency and critical/raw latency values.
2. Run `equivalence_study.py` and inspect `equivalence_report.md`.
3. Read module order:
   - `simulator.py` -> `scheduler.py` -> `imc_core.py` / `fpga_fabric.py` -> `stats.py` -> `pipeline_profiler.py`.
4. If editing energy:
   - touch only operator modules (`imc_core.py`, `fpga_fabric.py`, `memory.py`)
   - keep breakdown keys consistent.
5. If editing latency overlap:
   - start in `pipeline_profiler.py`
   - avoid mixing orchestration logic back into core operator models.
6. Regenerate report and compare deltas.

---

## 9) Known modeling boundaries (important)

- This simulator is analytical, not cycle/event-queue exact.
- Cross-layer readiness is a readiness-lite token recurrence (geometry-based), not full event simulation.
- Memory assumes effective BRAM throughput from config (`bram_width`, utilization, port mode), without explicit bank-conflict network model.
- Therefore, calibration against event-driven reference should target bounded error and stable trend consistency, not exact equality.

---

## 10) Quick reference: key files to open first

- System wiring:
  - `IMC/simulator.py`
- Mapping and per-layer timing tuple:
  - `IMC/scheduler_stats/scheduler.py`
- Cross-layer critical-path model:
  - `IMC/scheduler_stats/pipeline_profiler.py`
- Energy/latency storage:
  - `IMC/scheduler_stats/stats.py`
- IMC core model:
  - `IMC/imc_core/imc_core.py`
- Peripheral + memory model:
  - `IMC/peripherals/fpga_fabric.py`
  - `IMC/peripherals/memory.py`
- Calibration/report generation:
  - `calibration_study/equivalence_study.py`
  - `calibration_study/equivalence_report.md`
