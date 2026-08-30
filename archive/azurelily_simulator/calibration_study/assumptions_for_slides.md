# IMC Simulator Calibration Notes (vs. Azure-Lily Event-Driven Simulator)

## 1. New Simulator latency/energy calibration (Azure-Lily as reference)

This note summarizes how the analytical IMC simulator is calibrated against the event-driven Azure-Lily simulator.

### Calibration objective
- We target **similar breakdown ratios** (memory/core/peripheral) and **similar total magnitude** (same order, small relative error), not exact event-by-event equality.
- We keep the IMC simulator analytical and fast, while using the event-driven simulator as a reference behavior model.

### Unit alignment (must be consistent first)
- IMC simulator energy unit is **pJ** (see `IMC/configs/*.json`, `IMC/imc_core/config.py`).
- Event-driven simulator energy constants are in **nJ** (see `nn/constant.py`).
- Conversion used in calibration: **1 nJ = 1000 pJ**.
- Latency unit in both is **ns**.

### What is calibrated
- IMC core stages: analog VMM, ADC conversion, digital post-processing (`IMC/imc_core/imc_core.py`).
- Memory model: BRAM access latency/energy (`IMC/peripherals/memory.py`).
- FPGA peripheral model: activation/maxpool/reduction/DSP ops (`IMC/peripherals/fpga_fabric.py`).
- Cross-layer overlap model: critical-path profiler (`IMC/scheduler_stats/pipeline_profiler.py`).

---

## 1.1 Assumptions for energy/latency modeling (and differences vs event-driven simulator)

### Side-by-side assumptions table

| Topic | IMC simulator assumption | Event-driven simulator assumption | Calibration implication |
|---|---|---|---|
| **Modeling granularity** | We model each layer with closed-form stage latencies/energies and then compose with a pipeline profiler. | We model per-event execution with readiness checks and resource availability updates. | IMC is faster to run; event-driven is finer-grain and can expose more stalls. |
| **Energy unit** | We report all energies in **pJ**. | We define constants in **nJ**. | We convert event-driven values to pJ before numeric comparison. |
| **Latency unit** | We report all latencies in **ns**. | We report all latencies in **ns**. | No unit conversion is needed for latency. |
| **Memory energy** | We model BRAM energy as `num_access * e_bram_pj_per_access`. | We model SRAM/buffer energy with per-byte constants (e.g., `4.95e-6 nJ/byte`). | Memory buckets are comparable only after mapping scope (BRAM-only vs SRAM+buffer terms). |
| **Memory latency** | We model BRAM latency from effective bytes/access and FPGA clock. | We model memory latency per event using `SRAM_LAT` and `ELEMS_PER_MV`. | Matching effective BW is key to reducing read/write mismatch. |
| **IMC core compute** | We model stages `VMM -> ADC -> digital` with config-driven factors (`_get_arch_factors`). | We model DPE/ADC/sum events explicitly per operation. | Core totals match after mapping stage constants and array factors. |
| **Intra-layer pipeline** | We model fill/steady pipeline (`t_fill + (M-1)*t_steady`) for GEMM/maxpool. | We realize pipeline behavior through event scheduling over time. | IMC captures throughput behavior with fewer state variables. |
| **Inter-layer overlap** | We model cross-layer critical path using token-ready recurrence (`PipelineProfiler`). | We model true event readiness (`check_ready`) and stage availability per event. | IMC overlap is controlled but coarser than full event-driven scheduling. |
| **Q/K/V parallelism (attention)** | We explicitly merge `linear_Q/K/V` as one parallel group, then serialize later attention stages. | We naturally overlap when events/resources allow. | IMC reproduces the intended parallel pattern at layer granularity. |
| **Activation latency** | We model FPGA activation throughput from `act_units` and `act_cycles_per_op`. | We use fixed event constants and event queue timing. | Activation mismatch is mostly a constant/parallelism mapping problem, not a unit problem. |

### Assumption summary for slides
- IMC simulator is **analytical + critical-path composition**.
- Event-driven simulator is **event-accurate + resource-availability scheduling**.
- The calibration target is to make analytical assumptions land near event-driven totals/breakdowns.

---

## 1.2 Constants, effective memory BW, and pipeline strategy (inter-layer & intra-layer)

### A) Constant mapping (Azure-Lily config to analytical model)

| Stage | IMC config field | IMC usage | Event-driven analog |
|---|---|---|---|
| Analog VMM latency/energy | `t_analoge`, `e_analoge_pj` | `latency_per_vmm`, `energy_per_vmm` | DPE event path |
| ADC latency/energy | `t_conv`, `e_conv_pj`, `cols_per_adc` | `latency_per_conversion`, `energy_per_conversion` | `ADC_LAT`, `ADC_ENERGY` |
| Digital post latency/energy | `t_digital`, `e_digital_pj` | `latency_per_digital_post`, `energy_per_digital_post` | SUM / shift-add path |
| BRAM latency/energy | `bram_width`, `mem_bw_utilization`, `freq`, `bram_pj_per_access` | `MemoryModel.latency`, `MemoryModel.energy` | `ELEMS_PER_MV`, `SRAM_LAT`, SRAM energy constants |
| FPGA peripheral constants | `dsp_pj_per_mac`, `clb_pj_per_mac`, `act_*` | DSP/CLB/activation models in `fpga_fabric.py` | `MAC_ENERGY`, `MAXPOOL_ENERGY`, `ACT_ENERGY`, etc. |

### B) Effective memory bandwidth check

For current Azure-Lily config (`IMC/configs/azure_lily.json`):
- `bram_width = 40 bits`
- `mem_bw_utilization = 0.4`
- Effective bytes/access used by IMC latency model:
  - `floor(40/8) * 0.4 = 5 * 0.4 = 2 bytes/access`
- At `freq = 300 MHz`, effective stream BW is about:
  - `2 bytes/cycle * 300M cycles/s = 600 MB/s`

For event-driven reference (`nn/constant.py`):
- `PHIT_SIZE = 16 bits`, `BIT_WIDTH = 8 bits` => `ELEMS_PER_MV = 2`
- Effective bytes/event = `2 bytes`
- `SRAM_LAT = 3.3 ns` gives about `2 / 3.3ns ≈ 606 MB/s`

**Takeaway:** the configured effective memory BW is intentionally close (~600 vs ~606 MB/s), so large residual mismatch usually comes from scheduling assumptions, not units.

### C) Intra-layer pipeline strategy

#### Conv2D / Linear (IMC GEMM form)
- Implemented in `IMC/imc_core/imc_core.py`:
  - `gemm_pipeline_profile(K, N)` computes per-row:
    - `t_read_row`
    - `t_compute_row` (pipelined max or serial sum)
    - `t_write_row`
  - `latency = t_fill + (M-1)*t_steady`
  - `t_fill = t_read_row + t_compute_row + t_write_row`
  - `t_steady = max(t_read_row, t_compute_row, t_write_row)`

#### MaxPool
- Implemented in `IMC/peripherals/fpga_fabric.py::maxpool`:
  - Same fill/steady template per output position.
  - Compare stage latency depends on window size and CLB parallelism.

### D) Inter-layer pipeline (critical-path orchestration)

- Implemented in `IMC/scheduler_stats/pipeline_profiler.py`.
- Scheduler passes each layer timing tuple:
  - `first_output_ns`, `steady`, `events`, `required_upstream_outputs`.
- Profiler computes output readiness recursively:
  - `T_i(1) = T_{i-1}(R_i(1)) + first_output_i`
  - `T_i(n) = max(T_i(n-1)+steady_i, T_{i-1}(R_i(n))+first_output_i)`
- For conv/maxpool, `R_i(n)` is derived from receptive-field geometry (row/col dependency), not a free heuristic.

### E) Practical differences from event-driven scheduling

| Aspect | IMC simulator | Event-driven simulator |
|---|---|---|
| Readiness state | Uses compact token-ready recurrence by layer. | Uses explicit event readiness checks (`check_ready`) per event. |
| Resource contention | Uses layer-level stage timing + critical-path merge. | Uses event-level availability times for read/compute/write stages. |
| Runtime cost | Low (analytical). | Higher (event simulation). |
| Accuracy mode | Good for fast design-space iteration. | Better for detailed micro-timing and stall diagnosis. |

---

## Slide-ready storyline (recommended order)

1. **Calibration target**: match ratios + total range, not event-by-event identity.  
2. **Unit sanity**: pJ vs nJ converted; latency both ns.  
3. **Constant mapping**: VMM/ADC/digital/memory/peripheral constants aligned.  
4. **Memory BW sanity**: show `2 bytes/access` on both sides for Azure-Lily.  
5. **Pipeline design**: intra-layer fill/steady + inter-layer token-ready critical path.  
6. **Residual differences**: mainly from event-level vs analytical scheduling granularity.  
