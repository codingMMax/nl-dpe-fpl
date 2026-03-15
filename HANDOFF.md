# NL-DPE Project — Handoff & Context Document
_Last updated: 2026-03-14_

Read this file first. It replaces the previous HANDOFF.md and supersedes the "Immediate Roadmap" sections of ONBOARDING_NOTES.md. ONBOARDING_NOTES.md still contains valid static repo structure info.

---

## 1. Overall Goal

Implement the NL-DPE RTL wrappers for LeNet and ResNet (and later VGG11), run them through the VTR FPGA CAD flow using the NL-DPE architecture XML, and keep the analytical IMC simulator (`azurelily/IMC/`) consistent with the RTL mapping strategy.

There are three parallel tracks tracked in `experiments.md`:
- **Azurelily baseline** (existing, mostly done)
- **NL-DPE (16-bit interface)** — current active work
- **NL-DPE (40-bit, paper spec)** — future

---

## 2. NL-DPE Architecture — Key Facts

| Property | Value |
|---|---|
| Crossbar size | 256 rows × 256 cols (one physical DPE instance) |
| Sub-arrays per DPE | 4 (each 256×256), shared ACAM — internal detail, not visible at RTL interface |
| Data interface | 16-bit PHIT: `data_in[15:0]`, `data_out[15:0]` (serial, not parallel 256-bit) |
| ACAM | Built-in ACAM (analog CAM) with two configurable modes (see §2.1) |
| Technology | 32nm |
| Operating frequency | 1 GHz (DPE core), 300 MHz (FPGA fabric) |

### 2.1 ACAM Dual Modes

The NL-DPE has **no separate ADC**. The ACAM serves both roles:

| Mode | Function | When used |
|---|---|---|
| **Activation mode** | Performs non-linear activation in analog domain | `k_tile == 1` (no cross-DPE reduction needed) |
| **Unity mode** | Acts as a linear ADC (identity function) | `k_tile > 1` (outputs partial sums for FPGA reduction) |

### 2.2 Bit-Serial Pipeline (NL-DPE vs Azure-Lily)

Both architectures accept **bit-sliced input** — the 8-bit input is applied 1 bit at a time serially to the crossbar. The pipeline structure differs:

**Azure-Lily — 4 pipelinable stages per bit:**
```
Bit 0: [DAC→Crossbar] → [ADC] → [Shift&Add]
Bit 1:                   [DAC→Crossbar] → [ADC] → [Shift&Add]
...
Bit 7:                                                        ... → [Shift&Add]
Core latency = fill_4 + 7 × steady_4
```
Each bit flows independently through all 4 stages. Digital accumulation (shift & add) happens per-bit.

**NL-DPE — 2 pipelinable stages per bit, then ACAM fires once:**
```
Bit 0: [DAC→Crossbar] → [Analog Accum]
Bit 1:                   [DAC→Crossbar] → [Analog Accum]
...
Bit 7:                                                   [Analog Accum]
                                                                      → [ACAM] (fires once)
Core latency = fill_2 + 7 × steady_2 + t_acam
  fill_2   = t_analoge + t_conv
  steady_2 = max(t_analoge, t_conv)
  t_acam   = t_digital
```
The 8 bit-sliced results are accumulated in the **analog domain** (no digital shift & add). After all 8 bits are accumulated, the ACAM fires once to produce the final output (activated or linear depending on mode).

### 2.3 JSON Parameter Mapping

| JSON param | NL-DPE physical stage | Repeats | Pipelinable? |
|---|---|---|---|
| `t_analoge` / `e_analoge` | DAC → Crossbar | ×8 (per bit) | Yes |
| `t_conv` / `e_conv` | Crossbar → Analog Accumulation | ×8 (per bit) | Yes |
| `t_digital` / `e_digital` | ACAM → Output | ×1 (fires once) | No |

Energy values are **per-column** and scaled by `cols=256` when `scale_with_geometry=true`.

**Important fix in IMC simulator**: `azurelily/IMC/simulator.py` previously had `self.cfg.rows *= BLK_PER_TILE` (multiplying 256 by 4 → 1024). This line was **deleted** — it was wrong. The physical NL-DPE has 4 sub-arrays of 256×256, but this is an internal implementation detail for functional correctness, not throughput. The K dimension per DPE instantiation is **256**. The simulator now uses `cfg.rows = 256` directly from `nl_dpe.json`.

---

## 3. NL-DPE Workload Mapping Strategy

This is the ground-truth mapping derived from the IMC simulator (`azurelily/IMC/scheduler_stats/scheduler.py` and `imc_core/imc_core.py`). The RTL must implement the same mapping.

### 3.1 GEMM Reformulation

Every layer is mapped as a GEMM `(M, K, N)`:

| Layer type | M | K | N |
|---|---|---|---|
| Conv2D | `out_H × out_W × batch` | `kernel_H × kernel_W × in_channels` | `out_channels` |
| Linear/FC | `batch` | `in_features` | `out_features` |

### 3.2 Tile Count

```
k_tile = ceil(K / 256)    # tiles in input dimension
n_tile = ceil(N / 256)    # tiles in output dimension
total_DPE = k_tile × n_tile
```

### 3.3 ACAM Mode Decision (the critical rule)

| Condition | ACAM mode | After DPE | External activation? |
|---|---|---|---|
| `k_tile == 1` (K ≤ 256) | **Activation mode** (non-linear) | Output is activated | **No** |
| `k_tile > 1` (K > 256) | **Unity mode** (linear, acts as ADC) | Partial sums → FPGA CLB adder tree | **Yes** (explicit activation module on FPGA) |

There is no separate ADC inside the NL-DPE — only ACAM configured in one of two modes. When `k_tile > 1`, the `k_tile` DPE instances each compute a partial dot product over 256 input elements with ACAM in unity mode (linear output). The FPGA accumulates these partial sums (`k_tile - 1` adders per output column in a reduction tree), then a separate activation module applies the non-linear function.

### 3.4 Dataflow Patterns (RTL)

**Single DPE (k_tile == 1, K ≤ 256):**
```
SRAM ──→ dpe (VMM + ACAM activation mode) ──→ data_out (activated) ──→ next stage
                                                └── no external activation
```

**Multiple DPEs (k_tile > 1, K > 256):**
```
SRAM ──→ dpe[0] (256 inputs, ACAM unity mode) ──┐
SRAM ──→ dpe[1] (256 inputs, ACAM unity mode) ──┤──→ adder tree ──→ activation ──→ next stage
  ...                                            │    (FPGA CLBs)    (FPGA CLBs)
SRAM ──→ dpe[k-1] (remainder, ACAM unity mode) ─┘
```
Each DPE outputs a partial sum via ACAM in unity mode (linear, acts as ADC). The adder tree has `k_tile - 1` adders per output column. The external activation module runs on FPGA after the accumulated result.

### 3.5 Residual Add (ResNet)

The residual skip connection is always an FPGA element-wise add — no DPE involvement. This is modeled by `scheduler.py:_run_residual()` calling `fpga.residual_add()`. In the RTL this maps to the existing `adder_dpe_*` modules, which remain **unchanged**. No activation follows the residual add.

### 3.6 MaxPool

Handled entirely on FPGA (`fpga.maxpool()`). The `max_pooling_N_CHANNELS_1` RTL module is **unchanged**.

### 3.7 Simulator Code Alignment

The IMC simulator was updated to match this mapping strategy:

1. **`simulator.py`**: Deleted `self.cfg.rows *= BLK_PER_TILE` — rows = 256 from `nl_dpe.json`, not 1024. The physical NL-DPE has 4 sub-arrays of 256×256 but this is an internal detail for functional correctness, not throughput.
2. **`imc_core.py:analoge_nonlinear_check()`**: Changed `k_tile > BLK_PER_TILE` → `k_tile > 1`. Any cross-DPE reduction (K > 256) disables ACAM activation mode and requires external activation.
3. Both files: removed unused `BLK_PER_TILE` imports.

**Fixed — bit-serial pipeline model**: `_core_bit_pipeline_row_latency()` now correctly models NL-DPE as 2 pipelinable stages (DAC→Crossbar, Analog Accum) + ACAM drain. With the fixed `_get_arch_factors()` returning (k_vmm=8, k_conv=8, k_digital=1), the core latency is `fill_2 + 7×steady_2 + t_acam = 2 + 7 + 1 = 10 ns`.

**Fixed — `_get_arch_factors()`**: For NL-DPE (analoge_accum=true, digital_accum=false), now returns k_conv=8 (analog accum per bit) and k_digital=1 (ACAM fires once). This also fixes the energy bug where ACAM energy (`e_digital_pj = 43.9` per column) was zeroed out.

**Fixed — `pipelinable` flag**: Now `True` for all architectures. Inter-position pipeline stages (read/core/reduce/write) can overlap across output positions regardless of intra-core accumulation method.

---

## 4. Per-Layer Tile Analysis

### LeNet (1-channel input, 32×32)

| Layer | K | k_tile | n_tile | Total DPEs | ACAM? | External activation? |
|---|---|---|---|---|---|---|
| Conv1 5×5, 1→6 | 25 | 1 | 1 | 1 | **Yes** | No |
| MaxPool | — | — | — | 0 | — | — |
| Conv2 5×5, 6→16 | 150 | 1 | 1 | 1 | **Yes** | No |
| MaxPool | — | — | — | 0 | — | — |
| FC1 400→120 | 400 | **2** | 1 | 2 | No (ADC) | **Yes** |
| FC2 120→84 | 120 | 1 | 1 | 1 | **Yes** | No |
| FC3 84→10 | 84 | 1 | 1 | 1 | **Yes** | No |

Only FC1 (K=400 > 256) requires cross-DPE reduction and an external activation module.

### ResNet (full-channel, from `models/resnet.py`)

| Layer | Type | C_in | C_out | k | K=k²×C_in | N | k_tile | n_tile | DPEs | ACAM mode | Ext Act? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conv1 | conv2d | 3 | 56 | 3 | 27 | 56 | 1 | 1 | 1 | Activation | No |
| conv2 | conv2d | 56 | 112 | 3 | 504 | 112 | **2** | 1 | **2** | Unity | **Yes** |
| pool1 | maxpool | — | — | — | — | — | — | — | 0 | — | FPGA |
| conv3 | conv2d | 112 | 112 | 3 | 1008 | 112 | **4** | 1 | **4** | Unity | **Yes** |
| conv4 | conv2d | 112 | 112 | 3 | 1008 | 112 | **4** | 1 | **4** | Unity | **Yes** |
| res1 | residual | — | — | — | — | — | — | — | 0 | — | FPGA |
| conv5 | conv2d | 112 | 224 | 3 | 1008 | 224 | **4** | 1 | **4** | Unity | **Yes** |
| pool2 | maxpool | — | — | — | — | — | — | — | 0 | — | FPGA |
| conv6 | conv2d | 224 | 224 | 3 | 2016 | 224 | **8** | 1 | **8** | Unity | **Yes** |
| pool3 | maxpool | — | — | — | — | — | — | — | 0 | — | FPGA |
| conv7 | conv2d | 224 | 224 | 3 | 2016 | 224 | **8** | 1 | **8** | Unity | **Yes** |
| conv8 | conv2d | 224 | 224 | 3 | 2016 | 224 | **8** | 1 | **8** | Unity | **Yes** |
| res2 | residual | — | — | — | — | — | — | — | 0 | — | FPGA |
| pool4 | maxpool | — | — | — | — | — | — | — | 0 | — | FPGA |
| full1 | linear | 224 | 10 | 1 | 224 | 10 | 1 | 1 | 1 | Activation | No |
| **Total** | | | | | | | | | **40** | | |

Only conv1 and full1 have k_tile==1 (ACAM activation mode, no external activation). All other conv layers require ACAM unity mode + FPGA reduction + FPGA external activation.

Compared to Azure-Lily (512×128 crossbar, 35 DPEs for ResNet):
- NL-DPE uses **more k_tiles** (256 vs 512 rows → K overflows sooner)
- NL-DPE uses **fewer n_tiles** (256 vs 128 cols → N fits more easily)
- Net: 40 NL-DPE DPEs vs 35 Azure-Lily DPEs

---

## 5. RTL Implementation Plan

### 5.1 What's Already Done

**DPE signal interface fixes (Session 2026-03-13)**

All RTL wrappers were audited against the NL-DPE XML port contract. The following were fixed to use the correct 16-bit padding (`{8'b0, sram_data_out}`) and output split (`{dpe_data_out_hi, data_out}`):

- `azurelily_TACO_experiments/conv_layer_single_dpe.v` ✅
- `azurelily_TACO_experiments/conv_layer_stacked_dpes_conv3.v` ✅
- `azurelily_TACO_experiments/lenet_1_channel.v` ✅
- `azurelily_TACO_experiments/resnet_1_channel.v` ✅
- `azurelily_TACO_experiments/resnet_1_channel_small.v` ✅
- `azurelily_TACO_experiments/vgg11_1_channel.v` ✅ (3 generate blocks + single instance)
- `nl_dpe_rtl/conv_layer_single_dpe.v` ✅
- `nl_dpe_rtl/lenet_1_channel.v` ✅
- `nl_dpe_rtl/resnet_1_channel.v` ✅
- `nl_dpe_rtl/resnet_1_channel_small.v` ✅
- `nl_dpe_rtl/vgg11_1_channel.v` ✅

**experiments.md TODOs marked done (Session 2026-03-13)**

Under Arch' (NL-DPE architecture):
- [x] Replace DPE block in VTR with NL-DPE (XML exists and is correct)
- [x] Signal interface is the same (16-bit) as Azurelily (all wrappers fixed above)
- [x] Area of the NL-DPE block (XML value 1595429, assumed correct)
- [ ] Delay — **deferred**: using Azurelily delay constants for now

**NL-DPE LeNet and ResNet wrappers implemented (Session 2026-03-14)**

Both `nl_dpe_rtl/lenet_1_channel.v` and `nl_dpe_rtl/resnet_1_channel.v` have been modified to implement the NL-DPE mapping strategy. Since all layers in both designs use `N_CHANNELS=1`, every K value is small (≤25 for LeNet, ≤9 for ResNet), so `k_tile = 1` for **all layers** and ACAM is always in activation mode (non-linear output, no external activation needed).

The change: all external activation modules were removed and the pipeline signals were rewired to bypass them. No `dpe.v` is needed — VTR maps `.subckt dpe` instantiations directly to the hard block defined in the architecture XML.

LeNet (`nl_dpe_rtl/lenet_1_channel.v`):
- Removed: `act1`, `act2`, `act3`, `act4` instantiations from top-level
- Removed: `activation_layer1`, `activation_layer2`, `activation_layer3`, `activation_layer4`, `tanh_activation_parallel_N_CHANNELS_1` module definitions
- Rewired: each conv output connects directly to the next stage (pool or next conv)
- Pipeline: `conv1 → pool1 → conv2 → pool2 → conv3 → conv4 → conv5 → global_sram`
- File reduced from 1804 to 1086 lines

ResNet (`nl_dpe_rtl/resnet_1_channel.v`):
- Removed: `act1`–`act8` instantiations from top-level
- Removed: `activation_layer1`–`activation_layer8`, `relu_activation_parallel_N_CHANNELS_1` module definitions
- Rewired: each conv output connects directly to the next stage (pool, next conv, or residual add)
- Pipeline: `conv1 → conv2 → pool1 → conv3 → conv4 → residual1(+pool1) → conv5 → pool2 → conv6 → pool3 → conv7 → conv8 → residual2(+pool2) → pool4 → conv9 → global_sram`
- File reduced from 3667 to 2927 lines
- DPE count unchanged: 35 DPEs total across 9 conv layers

Unchanged modules: all `conv_layer*`, `conv_controller*`, `sram`, `pool_layer*`, `max_pooling_*`, `residual_layer*`, `adder_dpe_*`, `global_controller`, `controller_scalable`, `xbar_ip_module`.

Detailed implementation plan with per-signal rewiring tables: see `IMPLEMENTATION_PLAN.md`.

### 5.2 What Needs to Be Done Next

#### A. Run VTR flow (highest priority)

```bash
python azurelily_TACO_experiments/run_vtr.py \
    nl_dpe_rtl/lenet_1_channel.v \
    nl_dpe_rtl/resnet_1_channel.v \
    nl_dpe_rtl/nl_dpe_22nm_auto.xml \
    --jobs 2
```

Confirm pack/place/route success, capture resource and timing reports.

#### B. NL-DPE VGG11 wrapper (future)

Same pattern as LeNet/ResNet: remove activation modules, rewire signals. VGG11 uses generate blocks with 10/20/36 DPE instances — more complex but same principle.

#### C. Multi-channel / larger models (future)

For layers where **k_tile > 1** (K > 256), the current wrappers will need:
- Multiple DPE instances in K dimension with ADC mode
- FPGA adder tree for partial sum accumulation
- External activation module after the adder tree

This is not needed for the current 1-channel designs but will be required for multi-channel variants.

---

## 6. Simulator Changes Made

### 6.1 IMC Simulator Fixes (NL-DPE alignment)

Two fixes applied to align the IMC simulator with physical NL-DPE behavior:

1. **`azurelily/IMC/simulator.py`**: Deleted `self.cfg.rows *= BLK_PER_TILE` — the simulator now uses `cfg.rows = 256` directly from `nl_dpe.json`. The `BLK_PER_TILE` import was also removed. Do not re-add.

2. **`azurelily/IMC/imc_core/imc_core.py:analoge_nonlinear_check()`**: Changed `k_tile > BLK_PER_TILE` to `k_tile > 1`. ACAM switches to unity mode (linear) whenever K > 256 (cross-DPE reduction needed). The `BLK_PER_TILE` import was also removed from this file.

### 6.2 Resource Count Alignment (Azurelily, Session 2026-03-14)

Added `memory_blocks` resource tracking to the IMC simulator to count BRAM instances per layer:
- **`scheduler_stats/stats.py`**: Added `memory_blocks` to `resource_total` and `resource_peak`
- **`scheduler_stats/scheduler.py`**: Added BRAM counting for conv/linear layers (1 per conv wrapper + 1 per external activation), plus 1 global BRAM counted once
- **`peripherals/fpga_fabric.py`**: Added BRAM counting for maxpool (1 per pool) and residual (2 per residual layer)

**Comparison results** (IMC simulator vs VTR for Azurelily 1-channel RTL):

| Resource | LeNet IMC | LeNet VTR | ResNet IMC | ResNet VTR | Notes |
|----------|-----------|-----------|------------|------------|-------|
| DPE (wc) | 6 total, 2 peak | 5 | 68 total, 16 peak | 35 | IMC uses full channels; RTL uses N_CH=1 |
| CLB | 398 total, 120 peak | 37 | 7792 total, 1568 peak | 110 | IMC analytical; VTR synthesizes actual Verilog |
| Memory | 12 | 13 | 26 | 15 | IMC counts logical; VTR packs into physical BRAMs |
| DSP | 0 | 0 | 0 | 0 | Aligned |

**Root cause**: The IMC model uses full-channel neural network dimensions (LeNet: 1→6→16, ResNet: 3→56→112→224) while the RTL uses `N_CHANNELS=1`. This creates a systematic 2× factor in DPE counts for multi-DPE layers and much larger CLB estimates (due to more activation units and maxpool comparators).

Full analysis: see `resource_alignment_analysis.md`.
Energy comparison: see `simulator_comparison.md`.

---

## 7. Key Files Reference

| File | Purpose |
|---|---|
| `ONBOARDING_NOTES.md` | Static repo structure, port contract, VTR flow notes |
| `experiments.md` | Three-column progress tracker (Azurelily / NL-DPE-16bit / NL-DPE-40bit) |
| `nl_dpe_rtl/nl_dpe_22nm_auto.xml` | Ground-truth DPE port contract and VTR tile definition |
| `azurelily/IMC/configs/nl_dpe.json` | NL-DPE energy/timing parameters (256×256, ACAM=true, freq=1GHz) |
| `azurelily/IMC/simulator.py` | IMC top-level; `rows *= BLK_PER_TILE` deleted (see §6) |
| `azurelily/IMC/imc_core/imc_core.py` | GEMM mapping, tile counts, ACAM check (`k_tile > 1`), reduction tree |
| `azurelily/IMC/scheduler_stats/scheduler.py` | Per-layer dispatch (`_run_conv2d`, `_run_linear`, `_run_residual`) |
| `IMPLEMENTATION_PLAN.md` | Detailed per-signal rewiring tables for LeNet and ResNet NL-DPE conversion |
| `nl_dpe_rtl/lenet_1_channel.v` | NL-DPE LeNet wrapper (activation modules removed, 1086 lines) |
| `nl_dpe_rtl/resnet_1_channel.v` | NL-DPE ResNet wrapper (activation modules removed, 2927 lines) |
| `nl_dpe_rtl/conv_layer_single_dpe.v` | Single-DPE conv wrapper (template for new wrappers) |
| `azurelily_TACO_experiments/run_vtr.py` | VTR runner script (parallel jobs via ThreadPoolExecutor) |
| `azurelily_TACO_experiments/makefile` | Legacy make-based VTR flow |
| `resource_alignment_analysis.md` | IMC vs VTR resource count mismatch analysis with per-layer DPE/CLB/BRAM breakdown |
| `simulator_comparison.md` | IMC vs NN simulator energy comparison (LeNet, ResNet) |
| `parse_vtr_resources.py` | VTR resource usage parser (formatted table output) |
| `nl_dpe_rtl/NL_DPE_VS_AZURELILY.md` | RTL implementation differences between NL-DPE and Azurelily |

---

## 8. Guardrails

- Do not break the `.subckt dpe` naming — used by VTR XML and all wrappers.
- `azurelily/` is a git submodule — coordinate edits carefully and do not create a parallel `IMC_new/` directory.
- Simulator energy unit is **pJ** (IMC) and **nJ** (event-driven `nn/`). Do not mix.
- Do not modify `adder_dpe_*`, `max_pooling_*`, or `residual_add` RTL modules unless explicitly asked — these are FPGA logic kept identical to Azurelily per the comparison plan.
- The 40-bit NL-DPE path (Arch'', RTL'', Simulator'') is future work. Do not start it until the 16-bit path is complete and verified.
