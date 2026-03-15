# NL-DPE vs Azurelily — RTL Implementation Differences

This document describes the RTL-level differences between the NL-DPE wrappers (`nl_dpe_rtl/`) and the Azurelily baseline (`azurelily_TACO_experiments/`).

---

## 1. Hardware Differences

| Property | Azurelily | NL-DPE |
|---|---|---|
| Crossbar size | 128 × 256 | 256 × 256 |
| Data interface | 16-bit PHIT (`data_in[15:0]`, `data_out[15:0]`) | Same |
| Non-linear activation | **External** — FPGA logic (tanh LUT or ReLU) | **Internal** — built-in ACAM (Analog CAM) |
| Sub-arrays per tile | 1 | 4 × 256×256 (internal detail, not visible at RTL) |

---

## 2. Key RTL Change: Activation Modules Removed

Azurelily relies on external FPGA logic to perform activation after each DPE output. NL-DPE's built-in ACAM performs non-linear activation internally, so the DPE `data_out` is already activated.

**Rule**: When `k_tile == 1` (K ≤ 256, i.e., the entire input vector fits in one DPE), ACAM is active and no external activation is needed. When `k_tile > 1` (cross-DPE reduction), ACAM acts as a linear ADC and external activation is still required after the adder tree.

For the current 1-channel designs (LeNet, ResNet), all layers have K ≤ 25 (conv) or K ≤ 9 (resnet conv), so `k_tile = 1` for every layer. Therefore **all activation modules were removed**.

### What was removed

**LeNet** (`lenet_1_channel.v`):
- 4 activation instantiations: `act1`, `act2`, `act3`, `act4`
- 5 module definitions: `activation_layer1`–`activation_layer4`, `tanh_activation_parallel_N_CHANNELS_1`

**ResNet** (`resnet_1_channel.v`):
- 8 activation instantiations: `act1`–`act8`
- 9 module definitions: `activation_layer1`–`activation_layer8`, `relu_activation_parallel_N_CHANNELS_1`

### Pipeline comparison

**Azurelily LeNet**:
```
conv1 → act1 → pool1 → conv2 → act2 → pool2 → conv3 → act3 → conv4 → act4 → conv5
```

**NL-DPE LeNet**:
```
conv1 → pool1 → conv2 → pool2 → conv3 → conv4 → conv5
```

**Azurelily ResNet**:
```
conv1 → act1 → conv2 → act2 → pool1 → conv3 → act3 → conv4 → act4 → res1
→ conv5 → act5 → pool2 → conv6 → act6 → pool3 → conv7 → act7 → conv8 → act8 → res2
→ pool4 → conv9
```

**NL-DPE ResNet**:
```
conv1 → conv2 → pool1 → conv3 → conv4 → res1
→ conv5 → pool2 → conv6 → pool3 → conv7 → conv8 → res2
→ pool4 → conv9
```

---

## 3. What Is Unchanged

The following modules are **identical** between Azurelily and NL-DPE:

| Module | Purpose |
|---|---|
| `conv_layer` / `conv_layer_single_dpe` | Single-DPE convolution wrapper (SRAM + controller + DPE) |
| `conv_layer_stacked_dpes_*` | Multi-DPE convolution wrappers (V2_H1, V2_H2, V4_H2) |
| `conv_controller` / `controller_scalable` | FSM for DPE read/write sequencing |
| `sram` | On-chip SRAM buffer |
| `pool_layer*` / `max_pooling_N_CHANNELS_1` | MaxPool — pure FPGA logic |
| `residual_layer*` / `adder_dpe_*` | Residual add — pure FPGA logic |
| `global_controller` | Top-level layer sequencer |
| `xbar_ip_module` | Crossbar IP wrapper |

The DPE instantiation pattern (`.subckt dpe` via VTR XML hard block) is also unchanged — both Azurelily and NL-DPE use the same Verilog `dpe dpe_inst(...)` syntax, mapped to the architecture XML.

---

## 4. Signal Interface

Both Azurelily and NL-DPE use the same 16-bit DPE port contract:

```verilog
dpe dpe_inst (
    .clk(clk),
    .reset(rst),
    .data_in({8'b0, sram_data_out}),      // 8-bit data zero-padded to 16-bit
    .data_out({dpe_data_out_hi, data_out}), // 16-bit split: upper 8 discarded
    .nl_dpe_control(2'b0),
    .shift_add_control(shift_add_control),
    .w_buf_en(w_buf_en),
    .shift_add_bypass(1'b0),
    .load_output_reg(load_output_reg),
    .load_input_reg(load_input_reg),
    .MSB_SA_Ready(),
    .dpe_done(dpe_done),
    .reg_full(reg_full),
    .shift_add_done(shift_add_done),
    .shift_add_bypass_ctrl()
);
```

The `{8'b0, sram_data_out}` padding and `{dpe_data_out_hi, data_out}` split were applied to all wrapper files in both directories during the signal interface audit.

---

## 5. When External Activation Is Still Needed

For future multi-channel designs where K > 256 (e.g., Conv 3×3 with 32+ input channels), `k_tile > 1` and the DPE operates in ADC mode (linear output). In that case:

1. Multiple DPEs compute partial dot products over 256-element slices
2. An FPGA adder tree accumulates the partial sums
3. An external activation module (tanh/ReLU on FPGA CLBs) applies the non-linear function

This pattern already exists in the Azurelily `conv_layer_stacked_dpes_*` wrappers — it just needs the activation module kept (rather than removed) for those specific layers.
