#!/usr/bin/env python3
"""Generate P-replica FC+softmax wrapper for Round 2 DSP bottleneck benchmark.

Each replica is a complete FC inference layer: GEMV → activation → softmax.
- GEMV: V×H DPE tiles (same as gen_gemm_wrapper)
- Softmax: CLB exp LUT + sum + reciprocal + DSP mac_int_9x9 normalize

Usage:
    python gen_fc_softmax_wrapper.py --rows 512 --cols 128 --K 2048 --N 256 --P 5 \
        --output-dir ../dse/rtl/
"""

import argparse
import math
from pathlib import Path

# Import single-replica GEMV generator internals
from gen_gemv_wrappers import (
    _gen_fc_top,
    _gen_fc_layer,
    _gen_activation_lut_module,
    _get_supporting_modules,
)
from gen_softmax_clb import gen_softmax_clb


def _gen_fc_softmax_top(v: int, h: int, k: int, n: int, rows: int, cols: int,
                         depth: int, addr_width: int,
                         data_width: int = 40) -> str:
    """Generate fc_softmax_top: fc_layer → bn_softmax_clb → output SRAM.

    Modified version of _gen_fc_top that chains softmax after fc_layer.
    """
    dw = data_width
    sm_aw = max(1, math.ceil(math.log2(n))) + 1
    lines = []
    lines.append(f"module fc_softmax_top #(")
    lines.append(f"    parameter DATA_WIDTH = {dw}")
    lines.append(f")(")
    lines.append(f"    input wire clk,")
    lines.append(f"    input wire rst,")
    lines.append(f"    input wire valid,")
    lines.append(f"    input wire ready_n,")
    lines.append(f"    input wire [DATA_WIDTH-1:0] data_in,")
    lines.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    lines.append(f"    output wire ready,")
    lines.append(f"    output wire valid_n")
    lines.append(f");")
    lines.append(f"")
    lines.append(f"    localparam V = {v};")
    lines.append(f"    localparam H = {h};")
    lines.append(f"    localparam DEPTH = {depth};")
    lines.append(f"    localparam ADDR_WIDTH = {addr_width};")
    lines.append(f"    localparam N_OUT = {n};")
    lines.append(f"")

    # ── Internal signals ──
    lines.append(f"    // Stage 1 (fc_layer) → Stage 2 (softmax) handshake")
    lines.append(f"    wire [DATA_WIDTH-1:0] fc_out;      // fc_layer output")
    lines.append(f"    wire fc_valid_n, fc_ready;")
    lines.append(f"    wire [DATA_WIDTH-1:0] sm_out;      // softmax output")
    lines.append(f"    wire sm_valid_n, sm_ready;")
    lines.append(f"")
    lines.append(f"    // Global controller signals")
    lines.append(f"    wire valid_g_out, ready_g_out;")
    lines.append(f"")

    # ── Stage 1: FC layer (GEMV + activation) ──
    if v == 1 and h == 1:
        lines.append(f"    // Stage 1: GEMV (V=1, H=1 → conv_layer_single_dpe)")
        lines.append(f"    conv_layer_single_dpe #(")
        lines.append(f"        .N_CHANNELS(1),")
        lines.append(f"        .ADDR_WIDTH(ADDR_WIDTH),")
        lines.append(f"        .N_KERNELS(1),")
        lines.append(f"        .KERNEL_WIDTH({k}),")
        lines.append(f"        .KERNEL_HEIGHT(1),")
        lines.append(f"        .W(1),")
        lines.append(f"        .H(1),")
        lines.append(f"        .S(1),")
        lines.append(f"        .DEPTH({depth}),")
        lines.append(f"        .DATA_WIDTH(DATA_WIDTH)")
        lines.append(f"    ) fc_layer_inst (")
        lines.append(f"        .clk(clk), .rst(rst),")
        lines.append(f"        .valid(valid_g_out), .ready_n(sm_ready),")
        lines.append(f"        .data_in(data_in),")
        lines.append(f"        .data_out(fc_out),")
        lines.append(f"        .ready(fc_ready), .valid_n(fc_valid_n)")
        lines.append(f"    );")
    else:
        lines.append(f"    // Stage 1: GEMV (V={v}, H={h} → fc_layer)")
        lines.append(f"    fc_layer #(")
        lines.append(f"        .DATA_WIDTH(DATA_WIDTH)")
        lines.append(f"    ) fc_layer_inst (")
        lines.append(f"        .clk(clk), .rst(rst),")
        lines.append(f"        .valid(valid_g_out), .ready_n(sm_ready),")
        lines.append(f"        .data_in(data_in),")
        lines.append(f"        .data_out(fc_out),")
        lines.append(f"        .ready(fc_ready), .valid_n(fc_valid_n)")
        lines.append(f"    );")
    lines.append(f"")

    # ── Stage 2: Softmax (CLB + DSP) ──
    lines.append(f"    // Stage 2: Softmax (CLB exp LUT + DSP normalize)")
    lines.append(f"    bn_softmax_clb #(")
    lines.append(f"        .N(N_OUT),")
    lines.append(f"        .DATA_WIDTH(DATA_WIDTH),")
    lines.append(f"        .ADDR_WIDTH({sm_aw}),")
    lines.append(f"        .DEPTH({n})")
    lines.append(f"    ) softmax_inst (")
    lines.append(f"        .clk(clk), .rst(rst),")
    lines.append(f"        .valid(fc_valid_n),")
    lines.append(f"        .ready_n(ready_n),")
    lines.append(f"        .data_in(fc_out),")
    lines.append(f"        .data_out(sm_out),")
    lines.append(f"        .ready(sm_ready),")
    lines.append(f"        .valid_n(sm_valid_n)")
    lines.append(f"    );")
    lines.append(f"")

    # ── Global controller ──
    lines.append(f"    // Global controller")
    lines.append(f"    global_controller #(")
    lines.append(f"        .N_Layers(1)")
    lines.append(f"    ) g_ctrl_inst (")
    lines.append(f"        .clk(clk), .rst(rst),")
    lines.append(f"        .ready_L1(fc_ready),")
    lines.append(f"        .valid_Ln(sm_valid_n),")
    lines.append(f"        .valid(valid),")
    lines.append(f"        .ready(ready),")
    lines.append(f"        .valid_L1(valid_g_out),")
    lines.append(f"        .ready_Ln(ready_g_out)")
    lines.append(f"    );")
    lines.append(f"")

    # ── Output SRAM buffer ──
    lines.append(f"    // Output SRAM buffer")
    lines.append(f"    reg [7:0] read_address, write_address;")
    lines.append(f"    wire [DATA_WIDTH-1:0] global_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(16))")
    lines.append(f"    global_sram (.clk(clk),.rst(rst),.w_en(sm_valid_n),")
    lines.append(f"                 .r_addr(read_address),.w_addr(write_address),")
    lines.append(f"                 .sram_data_in(sm_out),.sram_data_out(global_sram_out));")
    lines.append(f"")
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst) begin read_address <= 0; write_address <= 16; end")
    lines.append(f"        else begin")
    lines.append(f"            if (ready_g_out) read_address <= read_address + 1;")
    lines.append(f"            if (valid_g_out) write_address <= write_address + 1;")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")

    # ── Output connections ──
    lines.append(f"    assign data_out = global_sram_out;")
    lines.append(f"    assign ready = fc_ready;")
    lines.append(f"    assign valid_n = sm_valid_n;")
    lines.append(f"")
    lines.append(f"endmodule")
    return "\n".join(lines)


def _gen_gemm_softmax_top(p: int, data_width: int = 40) -> str:
    """Generate P-replica wrapper — same as gen_gemm_wrapper._gen_gemm_top
    but instantiates fc_softmax_top instead of fc_top."""
    sel_width = max(1, math.ceil(math.log2(p))) if p > 1 else 1
    lines = []

    lines.append(f"module gemm_softmax_top #(")
    lines.append(f"    parameter DATA_WIDTH = {data_width},")
    lines.append(f"    parameter P = {p}")
    lines.append(f")(")
    lines.append(f"    input wire clk,")
    lines.append(f"    input wire rst,")
    lines.append(f"    input wire valid,")
    lines.append(f"    input wire ready_n,")
    lines.append(f"    input wire [DATA_WIDTH-1:0] data_in,")
    lines.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    lines.append(f"    output wire ready,")
    lines.append(f"    output wire valid_n")
    lines.append(f");")
    lines.append(f"")

    # Per-replica signals
    for i in range(p):
        lines.append(f"    wire [DATA_WIDTH-1:0] rep{i}_data_out;")
        lines.append(f"    wire rep{i}_ready, rep{i}_valid_n;")
        lines.append(f"    reg  rep{i}_valid;")
    lines.append(f"")

    # Input distribution: round-robin
    lines.append(f"    reg [{sel_width}-1:0] in_sel;")
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst)")
    lines.append(f"            in_sel <= 0;")
    lines.append(f"        else if (valid)")
    if p > 1:
        lines.append(f"            in_sel <= (in_sel == {sel_width}'d{p - 1}) ? "
                     f"0 : in_sel + 1;")
    else:
        lines.append(f"            in_sel <= 0;")
    lines.append(f"    end")
    lines.append(f"")

    # Demux valid
    lines.append(f"    always @(*) begin")
    for i in range(p):
        lines.append(f"        rep{i}_valid = 1'b0;")
    lines.append(f"        case (in_sel)")
    for i in range(p):
        lines.append(f"            {sel_width}'d{i}: rep{i}_valid = valid;")
    lines.append(f"            default: ;")
    lines.append(f"        endcase")
    lines.append(f"    end")
    lines.append(f"")

    # Instantiate P copies of fc_softmax_top
    for i in range(p):
        lines.append(f"    fc_softmax_top #(.DATA_WIDTH(DATA_WIDTH)) replica_{i} (")
        lines.append(f"        .clk(clk), .rst(rst),")
        lines.append(f"        .valid(rep{i}_valid), .ready_n(1'b1),")
        lines.append(f"        .data_in(data_in),")
        lines.append(f"        .data_out(rep{i}_data_out),")
        lines.append(f"        .ready(rep{i}_ready), .valid_n(rep{i}_valid_n)")
        lines.append(f"    );")
        lines.append(f"")

    # Output collection: round-robin mux
    lines.append(f"    reg [{sel_width}-1:0] out_sel;")
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst)")
    lines.append(f"            out_sel <= 0;")
    if p > 1:
        lines.append(f"        else if (|{{" +
                     ", ".join(f"rep{i}_valid_n" for i in range(p)) +
                     f"}})")
        lines.append(f"            out_sel <= (out_sel == {sel_width}'d{p - 1}) ? "
                     f"0 : out_sel + 1;")
    lines.append(f"    end")
    lines.append(f"")

    # Output mux
    lines.append(f"    reg [DATA_WIDTH-1:0] data_out_mux;")
    lines.append(f"    reg valid_n_mux;")
    lines.append(f"    always @(*) begin")
    lines.append(f"        data_out_mux = {data_width}'d0;")
    lines.append(f"        valid_n_mux = 1'b0;")
    lines.append(f"        case (out_sel)")
    for i in range(p):
        lines.append(f"            {sel_width}'d{i}: begin")
        lines.append(f"                data_out_mux = rep{i}_data_out;")
        lines.append(f"                valid_n_mux = rep{i}_valid_n;")
        lines.append(f"            end")
    lines.append(f"            default: ;")
    lines.append(f"        endcase")
    lines.append(f"    end")
    lines.append(f"")

    lines.append(f"    assign data_out = data_out_mux;")
    lines.append(f"    assign valid_n = valid_n_mux;")
    ready_expr = " | ".join(f"rep{i}_ready" for i in range(p))
    lines.append(f"    assign ready = {ready_expr};")
    lines.append(f"")
    lines.append(f"endmodule")

    return "\n".join(lines)


def gen_fc_softmax_wrapper(k: int, n: int, rows: int, cols: int,
                            p: int, output_dir: str,
                            data_width: int = 40) -> Path:
    """Generate P-replica FC+softmax wrapper.

    Each replica: GEMV (fc_layer with V×H DPEs) → bn_softmax_clb (CLB + DSP).

    Returns path to the generated .v file.
    """
    v = math.ceil(k / rows)
    h = math.ceil(n / cols)
    total_dpes = p * v * h
    depth = max(512, k)
    addr_width = math.ceil(math.log2(depth)) if depth > 1 else 1
    acam_eligible = (v == 1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"gemm_softmax_{k}_{n}_{rows}x{cols}_P{p}.v"
    out_path = out_dir / filename

    parts = []

    # Header
    parts.append(f"// Auto-generated FC+softmax P-replica wrapper")
    parts.append(f"// Workload: K={k}, N={n}, P={p} replicas")
    parts.append(f"// Crossbar: {rows}x{cols}")
    parts.append(f"// Per-replica: V={v}, H={h}, DPEs={v * h}, +1 softmax (CLB+DSP)")
    parts.append(f"// Total DPEs: {total_dpes}")
    parts.append(f"// ACAM eligible (V=1): {'yes' if acam_eligible else 'no'}")
    parts.append(f"// Generated by gen_fc_softmax_wrapper.py")
    parts.append(f"")

    # gemm_softmax_top
    parts.append(f"// =====================================================")
    parts.append(f"// gemm_softmax_top — P-replica wrapper")
    parts.append(f"// =====================================================")
    parts.append(_gen_gemm_softmax_top(p, data_width))
    parts.append(f"")

    # fc_softmax_top (single replica)
    parts.append(f"// =====================================================")
    parts.append(f"// fc_softmax_top — GEMV + softmax (single replica)")
    parts.append(f"// =====================================================")
    parts.append(_gen_fc_softmax_top(v, h, k, n, rows, cols, depth, addr_width,
                                     data_width))
    parts.append(f"")

    # bn_softmax_clb
    parts.append(f"// =====================================================")
    parts.append(f"// bn_softmax_clb — CLB exp LUT + DSP normalize")
    parts.append(f"// =====================================================")
    parts.append(gen_softmax_clb(n, data_width))
    parts.append(f"")

    # fc_layer (only if not V=1 H=1)
    if not (v == 1 and h == 1):
        parts.append(f"// =====================================================")
        parts.append(f"// fc_layer — DPE stacking + reduction + activation")
        parts.append(f"// =====================================================")
        parts.append(_gen_fc_layer(v, h, k, n, rows, cols, depth, addr_width,
                                   data_width))
        parts.append(f"")

    # activation_lut (only if V > 1)
    if v > 1:
        parts.append(f"// =====================================================")
        parts.append(f"// activation_lut")
        parts.append(f"// =====================================================")
        parts.append(_gen_activation_lut_module())
        parts.append(f"")

    # Supporting modules
    parts.append(f"// =====================================================")
    parts.append(f"// Supporting modules")
    parts.append(f"// =====================================================")
    parts.append(_get_supporting_modules())
    parts.append(f"")

    out_path.write_text("\n".join(parts))
    print(f"  Generated {filename} (K={k}, N={n}, P={p}, "
          f"V={v}, H={h}, total_DPEs={total_dpes}, +{p} softmax DSPs)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate P-replica FC+softmax wrapper"
    )
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--P", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default=".")

    args = parser.parse_args()
    gen_fc_softmax_wrapper(args.K, args.N, args.rows, args.cols, args.P,
                           args.output_dir)
