# Azurelily Latency & Energy Calibration Report (16-bit Data Width)

Comparison between the NN event-driven simulator and IMC GEMM-based simulator
for Azure-Lily architecture on LeNet and ResNet benchmarks.

## Architecture Constants & Assumptions

### DPE Crossbar

| Parameter | Value | Source |
|-----------|-------|--------|
| Crossbar rows (DPE_ROWS) | 512 | Paper / constant.py / azure_lily.json |
| Crossbar cols (DPE_COLS) | 128 | Paper / constant.py / azure_lily.json |
| Weight bits per cell | 8 | azure_lily.json |
| Columns per ADC | 16 | constant.py / azure_lily.json (cols_per_adc) |
| Precision (bit width) | 8 | constant.py / azure_lily.json (precision_bits) |

### FPGA / Memory Interface

| Parameter | NN Simulator | IMC Simulator | Notes |
|-----------|-------------|---------------|-------|
| SRAM access width | PHIT_SIZE = 16 bits (2 bytes) | bram_width = 16 bits (2 bytes) | Aligned to match RTL serial interface |
| SRAM clock period | CLK_LAT = 3.3 ns (~303 MHz) | 1/freq = 3.3333 ns (300 MHz) | ~1% residual mismatch |
| BRAM mode | N/A (implicit SP) | bram_mode = 0 (single-port) | |
| Read access pattern | Per spatial position: ceil(C_in/2) * k*k accesses | Streaming: ceil(C_in/elems_per_mv) * k*k accesses | Aligned to match RTL |
| Write access pattern | ceil(C_out/2) accesses | ceil(C_out/elems_per_mv) accesses | Naturally aligned |

### Timing Constants

| Parameter | NN Simulator (constant.py) | IMC Simulator (azure_lily.json) | Match? |
|-----------|---------------------------|-------------------------------|--------|
| DPE latency per bit | DPE_LAT = 1 ns | t_analoge = 1 * core_cycle(1.0) = 1.0 ns | Yes |
| ADC latency per bit (per ADC) | ADC_LAT = 1.09 ns | t_conv = 1.09 * core_cycle * cols_per_adc = 17.44 ns (for 16 cols) | Yes |
| Buffer transfer latency | BUFFER_LAT = 1 ns | Not modeled (minor) | ~1 ns gap |
| DPE+ADC pipeline (8-bit) | 140.52 ns | _core_bit_pipeline_row_latency() = 140.52 ns | Yes |
| Activation units | NUM_ACTS = 16 | act_units = 16 | Yes |
| Adder units | NUM_ADDS = 16 | Reduction via CLB model | Different model |
| Maxpool units | NUM_MAXPOOLS = 16 | CLB comparator model | Different model |

### Energy Constants

| Parameter | NN Simulator (nJ) | IMC Simulator (pJ) | Match? |
|-----------|-------------------|-------------------|--------|
| SRAM read/write per byte | 0.000005 nJ | bram_pj_per_access = 0.0495 pJ / (bram_width/8) bytes | Close (~1%) |
| ADC energy per bit per col | 0.00233 nJ | e_conv_pj = 2.33 pJ (scaled by geometry) | Yes (0.00233 nJ = 2.33 pJ) |
| Activation per op | 0.000453 nJ | act_energy_pj_per_op = 0.45 pJ | Close (0.4532 vs 0.45 pJ) |
| Sum (add) per op | 0.00008498 nJ | clb_pj_per_mac * clb_coeff_add | Calibrated via coefficients |

### Key Modeling Differences (known, accepted)

1. **Clock period**: NN sim uses CLK_LAT=3.3 ns (~303 MHz), IMC sim uses freq=300 MHz (3.333 ns). Causes ~1% systematic inflation in IMC results.
2. **Inter-layer pipelining**: NN sim uses event-driven backpressure with per-position `check_ready()`. IMC sim uses closed-form `PipelineProfiler` with receptive-field dependency model. Results are close but not identical.
3. **Pool/residual layers**: Different latency formulas (NN sim event-driven vs IMC sim closed-form pipeline). Minor impact on total latency.
4. **BUFFER_LAT**: NN sim adds 1 ns for external-to-internal buffer transfer per DPE compute. IMC sim does not model this. Negligible (~0.7% of compute stage).

## LeNet

### Summary

| Metric | NN Simulator | IMC Simulator | Ratio (IMC/NN) |
|--------|-------------|---------------|----------------|
| **Total Latency (ns)** | 114180.0 | 110881.5 (critical path) | 0.971 |
| **Total Energy (pJ)** | 126654.3 | 126901.8 | 1.002 |

### Layer-wise Latency Comparison

| Layer | Type | C_in | C_out | K | Positions | NN Duration (ns) | IMC Critical (ns) | IMC Raw (ns) | Ratio (Crit/NN) |
|-------|------|------|-------|---|-----------|-----------------|-------------------|-------------|-----------------|
| conv1 | conv2d | 1 | 6 | 5 | 784 | 111299.1 | 110264.3 | 110264.3 | 0.991 |
| pool1 | maxpool | 6 | 6 | 2 | 196 | 89.1 | 56.7 | 7856.7 | 0.636 |
| conv2 | conv2d | 6 | 16 | 5 | 100 | 442.2 | 420.5 | 25170.5 | 0.951 |
| pool2 | maxpool | 16 | 16 | 2 | 25 | 237.6 | 140.0 | 2700.0 | 0.589 |
| full1 | linear | 400 | 120 | 1 | 1 | 1197.9 | 0.0 | 1033.9 | 0.000 |
| full2 | linear | 120 | 84 | 1 | 1 | 617.1 | 0.0 | 500.5 | 0.000 |
| full3 | linear | 84 | 10 | 1 | 1 | 297.0 | 0.0 | 297.2 | 0.000 |
| **Total** | | | | | | **114180.0** | **110881.5** | **147823.1** | **0.971** |

### Per-Position Pipeline Stage Breakdown (conv2d/linear)

**NN Simulator** (per output position):

| Layer | t_read (ns) | t_compute (ns) | t_sum (ns) | t_act (ns) | t_write (ns) | fill (ns) | steady (ns) | Positions | Closed-form Total (ns) |
|-------|------------|----------------|-----------|-----------|-------------|----------|------------|-----------|----------------------|
| conv1 | 82.5 | 141.9 | 0.0 | 9.9 | 9.9 | 244.2 | 141.9 | 784 | 111351.9 |
| conv2 | 247.5 | 141.9 | 0.0 | 26.4 | 26.4 | 442.2 | 247.5 | 100 | 24944.7 |
| full1 | 660.0 | 141.9 | 0.0 | 198.0 | 198.0 | 1197.9 | 660.0 | 1 | 1197.9 |
| full2 | 198.0 | 141.9 | 0.0 | 138.6 | 138.6 | 617.1 | 198.0 | 1 | 617.1 |
| full3 | 138.6 | 141.9 | 0.0 | 0.0 | 16.5 | 297.0 | 141.9 | 1 | 297.0 |

**IMC Simulator** (per output position, GEMM pipeline):

| Layer | M | K | N | k_tile | n_tile | t_read (ns) | t_core (ns) | t_reduc (ns) | t_write (ns) | fill (ns) | steady (ns) | Total (ns) |
|-------|---|---|---|--------|--------|------------|------------|-------------|-------------|----------|------------|-----------|
| conv1 | 784 | 25 | 6 | 1 | 1 | 83.3 | 140.5 | 0.0 | 10.0 | 233.9 | 140.5 | 110261.0 |
| conv2 | 100 | 150 | 16 | 1 | 1 | 250.0 | 140.5 | 0.0 | 26.7 | 417.2 | 250.0 | 25167.2 |
| full1 | 1 | 400 | 120 | 1 | 1 | 666.7 | 140.5 | 0.0 | 200.0 | 1007.2 | 666.7 | 1007.2 |
| full2 | 1 | 120 | 84 | 1 | 1 | 200.0 | 140.5 | 0.0 | 140.0 | 480.5 | 200.0 | 480.5 |
| full3 | 1 | 84 | 10 | 1 | 1 | 140.0 | 140.5 | 0.0 | 16.7 | 297.2 | 140.5 | 297.2 |

### Energy Breakdown

**NN Simulator** (nJ):

| Component | Energy (nJ) |
|-----------|------------|
| sram_read_energy | 0.260746 |
| sram_write_energy | 0.080081 |
| external_buffer_write_energy | 0.197327 |
| external_buffer_read_energy | 0.174260 |
| internal_buffer_write_energy | 0.174260 |
| output_buffer_write_energy | 0.032264 |
| output_buffer_read_energy | 0.040065 |
| adc_energy | 121.495520 |
| maxpool_energy | 1.250052 |
| act_energy | 2.949724 |
| **Total** | **126.654299** |

**IMC Simulator** (pJ):

| Component | Energy (pJ) |
|-----------|------------|
| sram_read | 1027.32 |
| sram_write | 200.33 |
| imc_conversion | 121495.52 |
| fpga_activation | 2928.60 |
| clb_compare | 1250.05 |
| **Total** | **126901.82** |

## ResNet

### Summary

| Metric | NN Simulator | IMC Simulator | Ratio (IMC/NN) |
|--------|-------------|---------------|----------------|
| **Total Latency (ns)** | 992286.9 | 1046331.8 (critical path) | 1.054 |
| **Total Energy (pJ)** | 9319036.2 | 9853008.6 | 1.057 |

### Layer-wise Latency Comparison

| Layer | Type | C_in | C_out | K | Positions | NN Duration (ns) | IMC Critical (ns) | IMC Raw (ns) | Ratio (Crit/NN) |
|-------|------|------|-------|---|-----------|-----------------|-------------------|-------------|-----------------|
| conv1 | conv2d | 3 | 56 | 3 | 1024 | 145516.8 | 144059.1 | 144059.1 | 0.990 |
| conv2 | conv2d | 56 | 112 | 3 | 1024 | 676476.9 | 721395.7 | 860510.5 | 1.066 |
| pool1 | maxpool | 112 | 112 | 2 | 256 | 1663.2 | 940.0 | 191340.0 | 0.565 |
| conv3 | conv2d | 112 | 112 | 3 | 256 | 27700.2 | 30593.9 | 430433.9 | 1.104 |
| conv4 | conv2d | 112 | 112 | 3 | 256 | 29363.4 | 30593.9 | 430433.9 | 1.042 |
| res1 | residual | 112 | 112 | 1 | 256 | 372.9 | 563.3 | 95763.3 | 1.511 |
| conv5 | conv2d | 112 | 224 | 3 | 256 | 29756.1 | 30803.9 | 430643.9 | 1.035 |
| pool2 | maxpool | 224 | 224 | 2 | 64 | 3326.4 | 1873.3 | 95953.3 | 0.563 |
| conv6 | conv2d | 224 | 224 | 3 | 64 | 28693.5 | 34167.2 | 215607.2 | 1.191 |
| pool3 | maxpool | 224 | 224 | 2 | 16 | 3326.4 | 1873.3 | 24273.3 | 0.563 |
| conv7 | conv2d | 224 | 224 | 3 | 16 | 13909.5 | 20727.2 | 54327.2 | 1.490 |
| conv8 | conv2d | 224 | 224 | 3 | 16 | 18714.3 | 20727.2 | 54327.2 | 1.108 |
| res2 | residual | 224 | 224 | 1 | 16 | 742.5 | 1123.3 | 12323.3 | 1.513 |
| pool4 | maxpool | 224 | 224 | 4 | 1 | 12196.8 | 6360.0 | 6360.0 | 0.521 |
| full1 | linear | 224 | 10 | 1 | 1 | 528.0 | 530.5 | 530.5 | 1.005 |
| **Total** | | | | | | **992286.9** | **1046331.8** | **3046886.6** | **1.054** |

### Per-Position Pipeline Stage Breakdown (conv2d/linear)

**NN Simulator** (per output position):

| Layer | t_read (ns) | t_compute (ns) | t_sum (ns) | t_act (ns) | t_write (ns) | fill (ns) | steady (ns) | Positions | Closed-form Total (ns) |
|-------|------------|----------------|-----------|-----------|-------------|----------|------------|-----------|----------------------|
| conv1 | 59.4 | 141.9 | 0.0 | 92.4 | 92.4 | 386.1 | 141.9 | 1024 | 145549.8 |
| conv2 | 831.6 | 141.9 | 0.0 | 184.8 | 184.8 | 1343.1 | 831.6 | 1024 | 852069.9 |
| conv3 | 1663.2 | 141.9 | 23.1 | 184.8 | 184.8 | 2197.8 | 1663.2 | 256 | 426313.8 |
| conv4 | 1663.2 | 141.9 | 23.1 | 184.8 | 184.8 | 2197.8 | 1663.2 | 256 | 426313.8 |
| conv5 | 1663.2 | 141.9 | 46.2 | 369.6 | 369.6 | 2590.5 | 1663.2 | 256 | 426706.5 |
| conv6 | 3326.4 | 141.9 | 92.4 | 369.6 | 369.6 | 4299.9 | 3326.4 | 64 | 213863.1 |
| conv7 | 3326.4 | 141.9 | 92.4 | 369.6 | 369.6 | 4299.9 | 3326.4 | 16 | 54195.9 |
| conv8 | 3326.4 | 141.9 | 92.4 | 369.6 | 369.6 | 4299.9 | 3326.4 | 16 | 54195.9 |
| full1 | 369.6 | 141.9 | 0.0 | 0.0 | 16.5 | 528.0 | 369.6 | 1 | 528.0 |

**IMC Simulator** (per output position, GEMM pipeline):

| Layer | M | K | N | k_tile | n_tile | t_read (ns) | t_core (ns) | t_reduc (ns) | t_write (ns) | fill (ns) | steady (ns) | Total (ns) |
|-------|---|---|---|--------|--------|------------|------------|-------------|-------------|----------|------------|-----------|
| conv1 | 1024 | 27 | 56 | 1 | 1 | 60.0 | 140.5 | 0.0 | 93.3 | 293.9 | 140.5 | 144045.8 |
| conv2 | 1024 | 504 | 112 | 1 | 1 | 840.0 | 140.5 | 0.0 | 186.7 | 1167.2 | 840.0 | 860487.2 |
| conv3 | 256 | 1008 | 112 | 2 | 1 | 1680.0 | 140.5 | 3.3 | 186.7 | 2010.5 | 1680.0 | 430410.5 |
| conv4 | 256 | 1008 | 112 | 2 | 1 | 1680.0 | 140.5 | 3.3 | 186.7 | 2010.5 | 1680.0 | 430410.5 |
| conv5 | 256 | 1008 | 224 | 2 | 2 | 1680.0 | 140.5 | 3.3 | 373.3 | 2197.2 | 1680.0 | 430597.2 |
| conv6 | 64 | 2016 | 224 | 4 | 2 | 3360.0 | 140.5 | 6.7 | 373.3 | 3880.5 | 3360.0 | 215560.5 |
| conv7 | 16 | 2016 | 224 | 4 | 2 | 3360.0 | 140.5 | 6.7 | 373.3 | 3880.5 | 3360.0 | 54280.5 |
| conv8 | 16 | 2016 | 224 | 4 | 2 | 3360.0 | 140.5 | 6.7 | 373.3 | 3880.5 | 3360.0 | 54280.5 |
| full1 | 1 | 224 | 10 | 1 | 1 | 373.3 | 140.5 | 0.0 | 16.7 | 530.5 | 373.3 | 530.5 |

### Energy Breakdown

**NN Simulator** (nJ):

| Component | Energy (nJ) |
|-----------|------------|
| sram_read_energy | 10.425314 |
| sram_write_energy | 3.675959 |
| external_buffer_write_energy | 7.959343 |
| external_buffer_read_energy | 7.482658 |
| internal_buffer_write_energy | 7.482658 |
| output_buffer_write_energy | 1.525758 |
| output_buffer_read_energy | 1.917165 |
| adc_energy | 9085.769760 |
| maxpool_energy | 37.844209 |
| act_energy | 139.701233 |
| sum_energy | 15.252173 |
| **Total** | **9319.036229** |

**IMC Simulator** (pJ):

| Component | Energy (pJ) |
|-----------|------------|
| sram_read | 43711.27 |
| sram_write | 9585.82 |
| imc_conversion | 9620215.84 |
| clb_reduction | 209.40 |
| fpga_activation | 138700.80 |
| clb_compare | 37844.21 |
| clb_add | 2741.23 |
| **Total** | **9853008.57** |
