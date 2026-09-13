#!/usr/bin/env python3
"""Generate ResNet-9 and VGG-11 RTL for arbitrary DPE crossbar configs.

Usage:
    python nl_dpe/gen_model_wrapper.py --model resnet9 --rows 1024 --cols 128 --label proposed -o benchmarks/rtl/
    python nl_dpe/gen_model_wrapper.py --model vgg11 --rows 1024 --cols 256 --label al_matched -o benchmarks/rtl/

Reuses _gen_fc_layer() from gen_gemv_wrappers.py for DPE tiling + adder tree.
Each generated .v file is self-contained.
"""

import argparse
import math
import sys
from pathlib import Path

# Ensure nl_dpe/ is on the path
sys.path.insert(0, str(Path(__file__).parent))
from gen_gemv_wrappers import (
    _gen_fc_layer,
    _gen_activation_lut_module,
    _get_supporting_modules,
)

DATA_WIDTH = 40

# ═══════════════════════════════════════════════════════════════════════
# Model layer specifications
# ═══════════════════════════════════════════════════════════════════════

# Each conv entry: (name, K, N, W, H, has_act)
# K = kernel_h × kernel_w × in_channels (rows used in crossbar)
# N = out_channels (columns used in crossbar)
# has_act = whether this layer has activation (multi-DPE uses CLB activation_lut)

RESNET9_SPEC = {
    # Ground truth: Azure-Lily RTL (nl_dpe/resnet_1_channel.v)
    # Residual skip sources: res1 ← pool1, res2 ← pool2
    # Activation: V=1 → ACAM (no CLB), V>1 → CLB activation_lut
    "layers": [
        {"name": "conv1", "type": "conv", "K": 9, "N": 56, "W": 32, "H": 32, "has_act": True},
        {"name": "conv2", "type": "conv", "K": 504, "N": 112, "W": 32, "H": 32, "has_act": True},
        {"name": "pool1", "type": "pool", "kernel": 2},
        {"name": "conv3", "type": "conv", "K": 1008, "N": 112, "W": 16, "H": 16, "has_act": True},
        {"name": "conv4", "type": "conv", "K": 1008, "N": 112, "W": 16, "H": 16, "has_act": True},
        {"name": "res1", "type": "residual", "skip_from": "pool1"},
        {"name": "conv5", "type": "conv", "K": 1008, "N": 224, "W": 16, "H": 16, "has_act": True},
        {"name": "pool2", "type": "pool", "kernel": 2},
        {"name": "conv6", "type": "conv", "K": 2016, "N": 224, "W": 8, "H": 8, "has_act": True},
        {"name": "pool3", "type": "pool", "kernel": 2},
        {"name": "conv7", "type": "conv", "K": 2016, "N": 224, "W": 4, "H": 4, "has_act": True},
        {"name": "conv8", "type": "conv", "K": 2016, "N": 224, "W": 4, "H": 4, "has_act": True},
        {"name": "res2", "type": "residual", "skip_from": "pool2"},
        {"name": "pool4", "type": "pool", "kernel": 4},
        {"name": "conv9", "type": "conv", "K": 224, "N": 10, "W": 1, "H": 1, "has_act": False},
    ],
}

VGG11_SPEC = {
    # Ground truth: Azure-Lily RTL (nl_dpe/vgg11_1_channel.v)
    # All conv layers except conv9 have activation
    "layers": [
        {"name": "conv1", "type": "conv", "K": 9, "N": 64, "W": 33, "H": 33, "has_act": True},
        {"name": "pool1", "type": "pool", "kernel": 2},
        {"name": "conv2", "type": "conv", "K": 576, "N": 128, "W": 16, "H": 16, "has_act": True},
        {"name": "pool2", "type": "pool", "kernel": 2},
        {"name": "conv3", "type": "conv", "K": 1152, "N": 256, "W": 8, "H": 8, "has_act": True},
        {"name": "conv4", "type": "conv", "K": 2304, "N": 256, "W": 9, "H": 9, "has_act": True},
        {"name": "pool3", "type": "pool", "kernel": 2},
        {"name": "conv5", "type": "conv", "K": 2304, "N": 512, "W": 5, "H": 5, "has_act": True},
        {"name": "conv6", "type": "conv", "K": 4608, "N": 512, "W": 5, "H": 5, "has_act": True},
        {"name": "pool4", "type": "pool", "kernel": 2},
        {"name": "conv7", "type": "conv", "K": 4608, "N": 512, "W": 3, "H": 3, "has_act": True},
        {"name": "conv8", "type": "conv", "K": 4608, "N": 512, "W": 3, "H": 3, "has_act": True},
        {"name": "pool5", "type": "pool", "kernel": 2},
        {"name": "conv9", "type": "conv", "K": 512, "N": 10, "W": 1, "H": 1, "has_act": False},
    ],
}


def _gen_pool_module(name, kernel, data_width=40):
    """Simple max-pool module."""
    pool_size = kernel * kernel
    cnt_bits = max(1, math.ceil(math.log2(pool_size + 1)))
    return f"""module {name} #(parameter DATA_WIDTH = {data_width}) (
    input wire clk, rst, valid, ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, valid_n
);
    localparam POOL_SIZE = {pool_size};
    reg [DATA_WIDTH-1:0] max_val;
    reg [{cnt_bits}-1:0] count;
    reg out_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin max_val <= 0; count <= 0; out_valid <= 0; end
        else if (valid) begin
            if (count == 0) max_val <= data_in;
            else if ($signed(data_in) > $signed(max_val)) max_val <= data_in;
            if (count == POOL_SIZE - 1) begin count <= 0; out_valid <= 1; end
            else begin count <= count + 1; out_valid <= 0; end
        end else out_valid <= 0;
    end
    assign data_out = max_val;
    assign valid_n = out_valid;
    assign ready = 1'b1;
endmodule"""


def _gen_residual_module(name, data_width=40):
    """Residual add module with SRAM skip buffer."""
    return f"""module {name} #(parameter DATA_WIDTH = {data_width}, parameter DEPTH = 512) (
    input wire clk, rst, valid, ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, valid_n,
    input wire skip_valid,
    input wire [DATA_WIDTH-1:0] skip_data_in
);
    reg [DATA_WIDTH-1:0] skip_mem [0:DEPTH-1];
    reg [9:0] w_ptr, r_ptr;
    always @(posedge clk) begin
        if (rst) begin w_ptr <= 0; r_ptr <= 0; end
        else begin
            if (skip_valid) begin skip_mem[w_ptr] <= skip_data_in; w_ptr <= w_ptr + 1; end
            if (valid) r_ptr <= r_ptr + 1;
        end
    end
    assign data_out = $signed(data_in) + $signed(skip_mem[r_ptr]);
    assign valid_n = valid;
    assign ready = 1'b1;
endmodule"""


# ═══════════════════════════════════════════════════════════════════════
# Main generator
# ═══════════════════════════════════════════════════════════════════════

def gen_model(model_name, R, C, output_dir, label=None):
    spec = RESNET9_SPEC if model_name == "resnet9" else VGG11_SPEC
    layers = spec["layers"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{label}" if label else f"_{R}x{C}"
    filename = f"{model_name}{suffix}.v"
    top_name = model_name if not label else f"{model_name}_{label}"

    # Compute tiling per conv layer
    conv_info = []
    total_dpes = 0
    for layer in layers:
        if layer["type"] == "conv":
            K, N = layer["K"], layer["N"]
            V = math.ceil(K / R)
            H = math.ceil(N / C)
            dpes = V * H
            acam = (V == 1)  # ACAM eligibility: V=1 (k_tile=1), regardless of H
            conv_info.append({**layer, "V": V, "H": H, "dpes": dpes, "acam": acam})
            total_dpes += dpes

    parts = []

    # Header
    parts.append(f"// Auto-generated {model_name.upper()} RTL — DPE {R}×{C}")
    parts.append(f"// Total DPEs: {total_dpes}")
    parts.append(f"// Layer tiling:")
    for ci in conv_info:
        acam_str = "ACAM" if ci["acam"] else f"V={ci['V']}"
        parts.append(f"//   {ci['name']}: K={ci['K']} N={ci['N']} V={ci['V']} H={ci['H']} → {ci['dpes']} DPEs [{acam_str}]")
    parts.append(f"")

    # ─── Top-level module ──────────────────────────────────────────────
    parts.append(f"module {top_name} (")
    parts.append(f"    input wire clk, rst, valid, ready_n,")
    parts.append(f"    input wire [{DATA_WIDTH}-1:0] data_in,")
    parts.append(f"    output wire [{DATA_WIDTH}-1:0] data_out,")
    parts.append(f"    output wire ready, valid_n")
    parts.append(f");")
    parts.append(f"")

    # Wires for each layer
    for layer in layers:
        n = layer["name"]
        parts.append(f"    wire [{DATA_WIDTH}-1:0] data_out_{n};")
        parts.append(f"    wire valid_{n}, ready_{n};")
        # Extra wires for CLB activation after V>1 conv layers
        if layer["type"] == "conv" and layer.get("has_act", False):
            V = math.ceil(layer["K"] / R)
            if V > 1:  # V>1 → needs CLB activation_lut
                parts.append(f"    wire [{DATA_WIDTH}-1:0] data_out_act_{n};")
                parts.append(f"    wire valid_act_{n};")
    parts.append(f"    wire valid_g_out, ready_g_in;")
    parts.append(f"")

    # Chain layers
    prev_valid = "valid_g_out"
    prev_data = "data_in"
    first_ready = None

    for i, layer in enumerate(layers):
        n = layer["name"]
        ltype = layer["type"]

        if ltype == "conv":
            V = math.ceil(layer["K"] / R)
            H = math.ceil(layer["N"] / C)
            has_act = layer.get("has_act", False)
            needs_clb_act = has_act and V > 1  # V=1 → ACAM handles activation
            mod_name = f"{n}_layer"

            if V == 1 and H == 1:
                # Single DPE: instantiate conv_layer_single_dpe
                K, W, Ht = layer["K"], layer["W"], layer["H"]
                depth = max(512, K)
                addr_width = max(1, math.ceil(math.log2(depth)))
                acam_note = " (ACAM handles activation)" if has_act else ""
                parts.append(f"    // {n}: V=1 H=1 (single DPE){acam_note}")
                parts.append(f"    conv_layer_single_dpe #(")
                parts.append(f"        .N_CHANNELS(1), .ADDR_WIDTH({addr_width}),")
                parts.append(f"        .N_KERNELS(1), .KERNEL_WIDTH({K}), .KERNEL_HEIGHT(1),")
                parts.append(f"        .W({W}), .H({Ht}), .S(1),")
                parts.append(f"        .DEPTH({depth}), .DATA_WIDTH({DATA_WIDTH})")
                parts.append(f"    ) {n}_inst (")
            else:
                # Multi-DPE: use generated fc_layer with unique name
                act_note = " + CLB activation" if needs_clb_act else ""
                parts.append(f"    // {n}: V={V} H={H} ({V*H} DPEs){act_note}")
                parts.append(f"    {mod_name} #(.DATA_WIDTH({DATA_WIDTH})) {n}_inst (")

            parts.append(f"        .clk(clk), .rst(rst),")
            parts.append(f"        .valid({prev_valid}), .ready_n(ready_{n}),")
            parts.append(f"        .data_in({prev_data}),")
            parts.append(f"        .data_out(data_out_{n}),")
            parts.append(f"        .ready({f'ready_g_in' if first_ready is None else first_ready}), .valid_n(valid_{n})")
            parts.append(f"    );")

            if first_ready is None:
                first_ready = f"ready_{n}"

            prev_valid = f"valid_{n}"
            prev_data = f"data_out_{n}"

            # Add CLB activation_lut after V>1 conv layers
            # activation_lut is a simple registered LUT (clk, data_in → data_out)
            # No handshaking — valid/ready pass through with 1-cycle delay
            if needs_clb_act:
                parts.append(f"    // act_{n}: CLB activation (V>1, ACAM cannot absorb)")
                parts.append(f"    activation_lut #(.DATA_WIDTH({DATA_WIDTH})) act_{n}_inst (")
                parts.append(f"        .clk(clk),")
                parts.append(f"        .data_in({prev_data}),")
                parts.append(f"        .data_out(data_out_act_{n})")
                parts.append(f"    );")
                # Valid passes through with 1-cycle delay (registered output)
                parts.append(f"    reg valid_act_{n}_r;")
                parts.append(f"    always @(posedge clk) valid_act_{n}_r <= {prev_valid};")
                parts.append(f"    assign valid_act_{n} = valid_act_{n}_r;")
                prev_valid = f"valid_act_{n}"
                prev_data = f"data_out_act_{n}"

        elif ltype == "pool":
            kernel = layer["kernel"]
            mod_name = f"pool_mod_{n}"
            parts.append(f"    // {n}: max pool {kernel}×{kernel}")
            parts.append(f"    {mod_name} #(.DATA_WIDTH({DATA_WIDTH})) {n}_inst (")
            parts.append(f"        .clk(clk), .rst(rst),")
            parts.append(f"        .valid({prev_valid}), .ready_n(ready_{n}),")
            parts.append(f"        .data_in({prev_data}),")
            parts.append(f"        .data_out(data_out_{n}),")
            parts.append(f"        .ready({first_ready}), .valid_n(valid_{n})")
            parts.append(f"    );")
            first_ready = f"ready_{n}"
            prev_valid = f"valid_{n}"
            prev_data = f"data_out_{n}"

        elif ltype == "residual":
            mod_name = f"res_mod_{n}"
            skip_from = layer.get("skip_from", None)
            if skip_from:
                # Wire skip buffer to the named layer's output
                skip_valid_wire = f"valid_{skip_from}"
                skip_data_wire = f"data_out_{skip_from}"
                parts.append(f"    // {n}: residual add (skip from {skip_from})")
            else:
                skip_valid_wire = "1'b0"
                skip_data_wire = f"{DATA_WIDTH}'d0"
                parts.append(f"    // {n}: residual add (no skip source)")
            parts.append(f"    {mod_name} #(.DATA_WIDTH({DATA_WIDTH})) {n}_inst (")
            parts.append(f"        .clk(clk), .rst(rst),")
            parts.append(f"        .valid({prev_valid}), .ready_n(ready_{n}),")
            parts.append(f"        .data_in({prev_data}),")
            parts.append(f"        .data_out(data_out_{n}),")
            parts.append(f"        .ready({first_ready}), .valid_n(valid_{n}),")
            parts.append(f"        .skip_valid({skip_valid_wire}), .skip_data_in({skip_data_wire})")
            parts.append(f"    );")
            first_ready = f"ready_{n}"
            prev_valid = f"valid_{n}"
            prev_data = f"data_out_{n}"

        parts.append(f"")

    # Global controller
    parts.append(f"    global_controller #(.N_Layers(1)) g_ctrl (")
    parts.append(f"        .clk(clk), .rst(rst),")
    parts.append(f"        .ready_L1(ready_g_in), .valid_Ln({prev_valid}),")
    parts.append(f"        .valid(valid), .ready(ready),")
    parts.append(f"        .valid_L1(valid_g_out), .ready_Ln(ready_n)")
    parts.append(f"    );")
    parts.append(f"")
    parts.append(f"    assign data_out = {prev_data};")
    parts.append(f"    assign valid_n = {prev_valid};")
    parts.append(f"endmodule")
    parts.append(f"")

    # ─── Per-layer conv modules (V>1 or H>1 only) ─────────────────────
    generated_modules = set()
    for ci in conv_info:
        V, H = ci["V"], ci["H"]
        if V == 1 and H == 1:
            continue  # Uses conv_layer_single_dpe directly
        mod_name = f"{ci['name']}_layer"
        K, N = ci["K"], ci["N"]
        depth = max(512, K)
        addr_width = max(1, math.ceil(math.log2(depth)))
        parts.append(f"// ═══════════════════════════════════════════════")
        parts.append(f"// {mod_name}: V={V} H={H} K={K} N={N}")
        parts.append(f"// ═══════════════════════════════════════════════")
        parts.append(_gen_fc_layer(V, H, K, N, R, C, depth, addr_width,
                                   DATA_WIDTH, module_name=mod_name))
        parts.append(f"")

    # ─── Pool modules ──────────────────────────────────────────────────
    for layer in layers:
        if layer["type"] == "pool":
            mod_name = f"pool_mod_{layer['name']}"
            parts.append(_gen_pool_module(mod_name, layer["kernel"]))
            parts.append(f"")

    # ─── Residual modules ──────────────────────────────────────────────
    for layer in layers:
        if layer["type"] == "residual":
            mod_name = f"res_mod_{layer['name']}"
            parts.append(_gen_residual_module(mod_name))
            parts.append(f"")

    # ─── Activation LUT ────────────────────────────────────────────────
    parts.append(_gen_activation_lut_module())
    parts.append(f"")

    # ─── Supporting modules (sram, controllers, conv_layer_single_dpe) ─
    parts.append(f"// ═══════════════════════════════════════════════")
    parts.append(f"// Supporting modules")
    parts.append(f"// ═══════════════════════════════════════════════")
    parts.append(_get_supporting_modules())

    # Write
    out_path = out_dir / filename
    out_path.write_text("\n".join(parts))

    print(f"  Generated {filename}")
    print(f"    Model: {model_name}, Crossbar: {R}×{C}, Label: {label}")
    print(f"    Total DPEs: {total_dpes}")
    for ci in conv_info:
        acam_str = "ACAM" if ci["acam"] else "CLB"
        print(f"      {ci['name']}: V={ci['V']} H={ci['H']} → {ci['dpes']} DPEs [{acam_str}]")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate ResNet-9/VGG-11 RTL")
    parser.add_argument("--model", required=True, choices=["resnet9", "vgg11"])
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("-o", "--output-dir", default="benchmarks/rtl/")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    gen_model(args.model, args.rows, args.cols, args.output_dir, args.label)


if __name__ == "__main__":
    main()
