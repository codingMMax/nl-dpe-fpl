#!/usr/bin/env python3
"""Generate parameterized attention head RTL for NL-DPE DSE.

Takes (n_seq, d_head, rows, cols) and produces a self-contained .v file
with Q/K/V projection DPEs + fixed CLB DIMM/softmax modules.

Q/K/V projections: reuse gen_gemv_wrappers.py tiling logic (fc_layer for H>1)
Fixed CLB modules: extracted verbatim from attention_head_1_channel.v

Usage:
    python gen_attention_wrapper.py --rows 512 --cols 128 --output-dir ../dse/rtl/
    python gen_attention_wrapper.py --all --output-dir ../dse/rtl/
"""

import argparse
import math
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ATTENTION_SRC = SCRIPT_DIR / "attention_head_1_channel.v"

# Import FC tiling helpers from gen_gemv_wrappers
from gen_gemv_wrappers import (
    _gen_fc_layer,
    _gen_activation_lut_module,
    _get_supporting_modules,
    _extract_modules,
)

# 9 DSE configs: R ∈ {128, 256, 512}, C ∈ {64, 128, 256}
DSE_CONFIGS = [
    (128, 64), (128, 128), (128, 256),
    (256, 64), (256, 128), (256, 256),
    (512, 64), (512, 128), (512, 256),
]


def _extract_attention_clb_modules() -> str:
    """Extract fixed CLB modules from attention_head_1_channel.v.

    These modules are independent of (R, C) and are used as-is:
    - dimm_score_matrix
    - softmax_approx
    - dimm_weighted_sum
    """
    src = ATTENTION_SRC.read_text()
    all_mods = _extract_modules(src)

    needed = ["dimm_score_matrix", "softmax_approx", "dimm_weighted_sum"]
    parts = []
    for name in needed:
        if name in all_mods:
            parts.append(all_mods[name])
        else:
            print(f"  WARNING: module '{name}' not found in {ATTENTION_SRC.name}")
    return "\n\n".join(parts)


def _gen_attention_top(v: int, h: int, n_seq: int, d_head: int,
                       rows: int, cols: int,
                       depth: int, addr_width: int,
                       data_width: int = 40) -> str:
    """Generate the attention_head top-level module.

    Follows the structure of the hand-written attention_head_1_channel.v
    but replaces conv_layer_single_dpe with fc_layer when H>1.
    """
    lines = []
    lines.append(f"module attention_head #(")
    lines.append(f"    parameter N = {n_seq},        // sequence length")
    lines.append(f"    parameter d = {d_head},        // head dimension")
    lines.append(f"    parameter DATA_WIDTH = {data_width}")
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

    # Internal signals (matching hand-written attention_head)
    lines.append(f"    // ========================================================================")
    lines.append(f"    // Internal signals")
    lines.append(f"    // ========================================================================")
    lines.append(f"    wire [DATA_WIDTH-1:0] data_out_q, data_out_k, data_out_v;")
    lines.append(f"    wire ready_q, valid_q, ready_k, valid_k, ready_v, valid_v;")
    lines.append(f"    wire ready_score, valid_score, ready_softmax, valid_softmax;")
    lines.append(f"    wire ready_wsum, valid_wsum;")
    lines.append(f"    wire valid_g_in, valid_g_out, ready_g_in, ready_g_out;")
    lines.append(f"    wire [DATA_WIDTH-1:0] global_sram_data_in;")
    lines.append(f"")
    lines.append(f"    reg [7:0] read_address, write_address;")
    lines.append(f"")

    # Q/K/V Projections — wiring matches hand-written attention_head exactly
    proj_specs = [
        # (name, ready_n_connection, ready_connection)
        ("q", "ready_k",    "ready_g_in"),
        ("k", "ready_v",    ""),   # not connected
        ("v", "ready_score", ""),   # not connected
    ]

    if v == 1 and h == 1:
        # V=1, H=1: use conv_layer_single_dpe (same as hand-written)
        for proj, rn_conn, r_conn in proj_specs:
            lines.append(f"    // ========================================================================")
            lines.append(f"    // {proj.upper()} Projection DPE (V=1, H=1: single DPE, ACAM=log)")
            lines.append(f"    // ========================================================================")
            lines.append(f"    conv_layer_single_dpe #(")
            lines.append(f"        .N_CHANNELS(1),")
            lines.append(f"        .ADDR_WIDTH({addr_width}),")
            lines.append(f"        .N_KERNELS(1),")
            lines.append(f"        .KERNEL_WIDTH(d),")
            lines.append(f"        .KERNEL_HEIGHT(1),")
            lines.append(f"        .W(1),")
            lines.append(f"        .H(1),")
            lines.append(f"        .S(1),")
            lines.append(f"        .DEPTH({depth}),")
            lines.append(f"        .DATA_WIDTH(DATA_WIDTH)")
            lines.append(f"    ) {proj}_projection (")
            lines.append(f"        .clk(clk),")
            lines.append(f"        .rst(rst),")
            lines.append(f"        .valid(valid_g_out),")
            lines.append(f"        .ready_n({rn_conn}),")
            lines.append(f"        .data_in(data_in),")
            lines.append(f"        .data_out(data_out_{proj}),")
            if r_conn:
                lines.append(f"        .ready({r_conn}),")
            else:
                lines.append(f"        .ready(),")
            lines.append(f"        .valid_n(valid_{proj})")
            lines.append(f"    );")
            lines.append(f"")
    else:
        # V>1 or H>1: use fc_layer for projections
        for proj, rn_conn, r_conn in proj_specs:
            lines.append(f"    // ========================================================================")
            lines.append(f"    // {proj.upper()} Projection ({v}x{h} DPE tiling, ACAM=log)")
            lines.append(f"    // ========================================================================")
            lines.append(f"    fc_layer #(")
            lines.append(f"        .DATA_WIDTH(DATA_WIDTH)")
            lines.append(f"    ) {proj}_projection (")
            lines.append(f"        .clk(clk),")
            lines.append(f"        .rst(rst),")
            lines.append(f"        .valid(valid_g_out),")
            lines.append(f"        .ready_n({rn_conn}),")
            lines.append(f"        .data_in(data_in),")
            lines.append(f"        .data_out(data_out_{proj}),")
            if r_conn:
                lines.append(f"        .ready({r_conn}),")
            else:
                lines.append(f"        .ready(),")
            lines.append(f"        .valid_n(valid_{proj})")
            lines.append(f"    );")
            lines.append(f"")

    # DIMM Score Matrix (CLB — unchanged across configs)
    lines.append(f"    // ========================================================================")
    lines.append(f"    // Stage 4: DIMM Score Matrix (CLB-based log-domain computation)")
    lines.append(f"    // S[i][j] = sum_m exp(log_Q[i][m] + log_K[j][m])")
    lines.append(f"    // ========================================================================")
    lines.append(f"    wire [DATA_WIDTH-1:0] data_out_score;")
    lines.append(f"")
    lines.append(f"    dimm_score_matrix #(")
    lines.append(f"        .N(N),")
    lines.append(f"        .d(d),")
    lines.append(f"        .DATA_WIDTH(DATA_WIDTH)")
    lines.append(f"    ) score_inst (")
    lines.append(f"        .clk(clk),")
    lines.append(f"        .rst(rst),")
    lines.append(f"        .valid_q(valid_q),")
    lines.append(f"        .valid_k(valid_k),")
    lines.append(f"        .ready_n(ready_softmax),")
    lines.append(f"        .data_in_q(data_out_q),")
    lines.append(f"        .data_in_k(data_out_k),")
    lines.append(f"        .data_out(data_out_score),")
    lines.append(f"        .ready_q(ready_q),")
    lines.append(f"        .ready_k(ready_k),")
    lines.append(f"        .valid_n(valid_score)")
    lines.append(f"    );")
    lines.append(f"")

    # Softmax (CLB — unchanged across configs)
    lines.append(f"    // ========================================================================")
    lines.append(f"    // Stage 5: Softmax (CLB-based exp + normalization)")
    lines.append(f"    // ========================================================================")
    lines.append(f"    wire [DATA_WIDTH-1:0] data_out_softmax;")
    lines.append(f"")
    lines.append(f"    softmax_approx #(")
    lines.append(f"        .N(N),")
    lines.append(f"        .d(d),")
    lines.append(f"        .DATA_WIDTH(DATA_WIDTH)")
    lines.append(f"    ) softmax_inst (")
    lines.append(f"        .clk(clk),")
    lines.append(f"        .rst(rst),")
    lines.append(f"        .valid(valid_score),")
    lines.append(f"        .ready_n(ready_wsum),")
    lines.append(f"        .data_in(data_out_score),")
    lines.append(f"        .data_out(data_out_softmax),")
    lines.append(f"        .ready(ready_softmax),")
    lines.append(f"        .valid_n(valid_softmax)")
    lines.append(f"    );")
    lines.append(f"")

    # DIMM Weighted Sum (CLB — unchanged across configs)
    lines.append(f"    // ========================================================================")
    lines.append(f"    // Stage 6: DIMM Weighted Sum (CLB-based log-domain computation)")
    lines.append(f"    // O[i][m] = sum_j exp(log(attn[i][j]) + log_V[j][m])")
    lines.append(f"    // ========================================================================")
    lines.append(f"    wire [DATA_WIDTH-1:0] data_out_wsum;")
    lines.append(f"")
    lines.append(f"    dimm_weighted_sum #(")
    lines.append(f"        .N(N),")
    lines.append(f"        .d(d),")
    lines.append(f"        .DATA_WIDTH(DATA_WIDTH)")
    lines.append(f"    ) wsum_inst (")
    lines.append(f"        .clk(clk),")
    lines.append(f"        .rst(rst),")
    lines.append(f"        .valid_attn(valid_softmax),")
    lines.append(f"        .valid_v(valid_v),")
    lines.append(f"        .ready_n(ready_g_out),")
    lines.append(f"        .data_in_attn(data_out_softmax),")
    lines.append(f"        .data_in_v(data_out_v),")
    lines.append(f"        .data_out(data_out_wsum),")
    lines.append(f"        .ready_attn(ready_wsum),")
    lines.append(f"        .ready_v(ready_v),")
    lines.append(f"        .valid_n(valid_g_in)")
    lines.append(f"    );")
    lines.append(f"")

    # Global Controller (N_Layers=6: 3 projections + score + softmax + wsum)
    lines.append(f"    // ========================================================================")
    lines.append(f"    // Global Controller")
    lines.append(f"    // ========================================================================")
    lines.append(f"    global_controller #(")
    lines.append(f"        .N_Layers(6)")
    lines.append(f"    ) g_ctrl_inst (")
    lines.append(f"        .clk(clk),")
    lines.append(f"        .rst(rst),")
    lines.append(f"        .ready_L1(ready_g_in),")
    lines.append(f"        .valid_Ln(valid_g_in),")
    lines.append(f"        .valid(valid),")
    lines.append(f"        .ready(ready),")
    lines.append(f"        .valid_L1(valid_g_out),")
    lines.append(f"        .ready_Ln(ready_g_out)")
    lines.append(f"    );")
    lines.append(f"")

    # Global Output SRAM
    lines.append(f"    // ========================================================================")
    lines.append(f"    // Global Output SRAM")
    lines.append(f"    // ========================================================================")
    lines.append(f"    sram #(")
    lines.append(f"        .N_CHANNELS(1),")
    lines.append(f"        .DATA_WIDTH(DATA_WIDTH),")
    lines.append(f"        .DEPTH(256)")
    lines.append(f"    ) global_sram_inst (")
    lines.append(f"        .clk(clk),")
    lines.append(f"        .rst(rst),")
    lines.append(f"        .w_en(valid_g_in),")
    lines.append(f"        .r_addr(read_address),")
    lines.append(f"        .w_addr(write_address),")
    lines.append(f"        .sram_data_in(data_out_wsum),")
    lines.append(f"        .sram_data_out(global_sram_data_in)")
    lines.append(f"    );")
    lines.append(f"")

    # Address counter logic (matching hand-written)
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst) begin")
    lines.append(f"            read_address <= 0;")
    lines.append(f"            write_address <= 128;")
    lines.append(f"        end else begin")
    lines.append(f"            if (ready_g_out)")
    lines.append(f"                read_address <= read_address + 1;")
    lines.append(f"            if (valid_g_out)")
    lines.append(f"                write_address <= write_address + 1;")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")

    # Output connections
    lines.append(f"    assign data_out = global_sram_data_in;")
    lines.append(f"    assign valid_n = valid_g_in;")
    lines.append(f"")
    lines.append(f"endmodule")

    return "\n".join(lines)


def gen_attention_wrapper(n_seq: int, d_head: int, rows: int, cols: int,
                          output_dir: str) -> Path:
    """Generate a self-contained attention head .v file for (rows x cols) crossbar.

    Returns the path to the generated file.
    """
    v = math.ceil(d_head / rows)
    h = math.ceil(d_head / cols)
    dpes_per_proj = v * h
    total_dpes = 3 * dpes_per_proj
    data_width = 40
    acam_eligible = (v == 1)

    # Depth/addr_width: match hand-written for V=1,H=1; gen_fc_wrapper for H>1
    if v == 1 and h == 1:
        depth = 256
        addr_width = 8
    else:
        depth = max(512, d_head)
        addr_width = math.ceil(math.log2(depth)) if depth > 1 else 1

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"attention_{rows}x{cols}.v"
    out_path = out_dir / filename

    parts = []

    # Header
    parts.append(f"// Auto-generated attention head wrapper")
    parts.append(f"// Crossbar: {rows}x{cols}")
    parts.append(f"// Projection tiling: V={v}, H={h}, DPEs/proj={dpes_per_proj}")
    parts.append(f"// Total DPEs: {total_dpes} (3 projections x {dpes_per_proj})")
    parts.append(f"// ACAM eligible (V=1): {'yes' if acam_eligible else 'no'}")
    parts.append(f"// N={n_seq} (seq length), d={d_head} (head dim)")
    parts.append(f"// Generated by gen_attention_wrapper.py")
    parts.append(f"")

    # Top-level attention_head module
    parts.append(f"// =====================================================")
    parts.append(f"// attention_head — top-level module")
    parts.append(f"// =====================================================")
    parts.append(_gen_attention_top(v, h, n_seq, d_head, rows, cols,
                                    depth, addr_width, data_width))
    parts.append(f"")

    # fc_layer module (only needed when not V=1 H=1)
    if not (v == 1 and h == 1):
        parts.append(f"// =====================================================")
        parts.append(f"// fc_layer — DPE tiling for projections ({v}x{h})")
        parts.append(f"// =====================================================")
        parts.append(_gen_fc_layer(v, h, d_head, d_head, rows, cols,
                                   depth, addr_width, data_width))
        parts.append(f"")

    # activation_lut (only if V > 1 — not expected for attention DSE)
    if v > 1:
        parts.append(f"// =====================================================")
        parts.append(f"// activation_lut — piecewise-linear tanh approximation")
        parts.append(f"// =====================================================")
        parts.append(_gen_activation_lut_module())
        parts.append(f"")

    # Fixed CLB modules from attention_head_1_channel.v
    parts.append(f"// =====================================================")
    parts.append(f"// Fixed CLB modules (from attention_head_1_channel.v)")
    parts.append(f"// =====================================================")
    parts.append(_extract_attention_clb_modules())
    parts.append(f"")

    # Supporting modules from gemv_1_channel.v
    parts.append(f"// =====================================================")
    parts.append(f"// Supporting modules (from gemv_1_channel.v)")
    parts.append(f"// =====================================================")
    parts.append(_get_supporting_modules())
    parts.append(f"")

    out_path.write_text("\n".join(parts))
    print(f"  Generated {filename} (V={v}, H={h}, DPEs/proj={dpes_per_proj}, "
          f"total_DPEs={total_dpes}, ACAM={'yes' if acam_eligible else 'no'})")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameterized attention head RTL for NL-DPE DSE"
    )
    parser.add_argument("--rows", type=int, default=None, help="Crossbar rows")
    parser.add_argument("--cols", type=int, default=None, help="Crossbar cols")
    parser.add_argument("--n-seq", type=int, default=128,
                        help="Sequence length (default: 128)")
    parser.add_argument("--d-head", type=int, default=128,
                        help="Head dimension (default: 128)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--all", action="store_true",
                        help="Generate for all 9 DSE configs")
    args = parser.parse_args()

    if args.all:
        print(f"Generating attention head wrappers for all 9 configs -> {args.output_dir}")
        for rows, cols in DSE_CONFIGS:
            gen_attention_wrapper(args.n_seq, args.d_head, rows, cols,
                                 args.output_dir)
    elif args.rows and args.cols:
        print(f"Generating attention head wrapper ({args.rows}x{args.cols}) "
              f"-> {args.output_dir}")
        gen_attention_wrapper(args.n_seq, args.d_head, args.rows, args.cols,
                              args.output_dir)
    else:
        parser.error("Either --all or both --rows and --cols are required")


if __name__ == "__main__":
    main()
