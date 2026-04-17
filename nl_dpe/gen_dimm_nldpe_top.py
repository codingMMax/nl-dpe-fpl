#!/usr/bin/env python3
"""Generate W=16 NL-DPE full DIMM top RTL.

The full DIMM pipeline: mac_qk + softmax + mac_sv, with W parallel lanes.
Each lane is a complete pipeline:
  dimm_score_matrix (stage 1, mac_qk + softmax_exp fused via ACAM)
  softmax_approx    (stage 2, normalize)
  dimm_weighted_sum (stage 3, wsum_log + mac_sv)

Lane i processes attention rows {i, i+W, i+2W, ...}. With N=128 and W=16,
each lane handles 8 rows.

Total DPE count: 4 stages × W lanes = 64 DPEs per DIMM (for W=16).

The top module interface:
  Q row (d_head packed)      — broadcast to all lanes
  K matrix (N×d_head packed) — broadcast to all lanes
  V matrix (N×d_head packed) — broadcast to all lanes
  Output (N×d packed)        — muxed from all 16 lanes

Generator reuses existing sub-modules from gen_attention_wrapper.py:
  _gen_dimm_score_matrix, _gen_softmax_dpe, _gen_dimm_weighted_sum

Usage:
    python3 nl_dpe/gen_dimm_nldpe_top.py --N 128 --d 64 --C 128 --W 16
"""
import argparse
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from gen_attention_wrapper import (
    _gen_dimm_score_matrix,
    _gen_softmax_dpe,
    _gen_dimm_weighted_sum,
)
from gen_gemv_wrappers import _get_supporting_modules


def _gen_dimm_top_nldpe(n_seq, d_head, crossbar_cols, W, data_width=40):
    """Generate nldpe_dimm_top with W parallel lanes.

    Architecture:
      - Q/K/V broadcast BRAMs at top level
      - W parallel lanes, each is a complete score→softmax→wsum pipeline
      - Each lane's SRAMs are sized for N/W rows (only one lane row at a time in flight)
      - Output muxing: lanes emit results in round-robin order
    """
    dw = data_width
    epw = dw // 8
    C = crossbar_cols
    K_qkt = max(1, C // d_head)           # dual-identity for mac_qk
    K_sv = max(1, C // n_seq)             # single-identity for mac_sv (K=1 at BERT-Tiny)
    packed_d = math.ceil(d_head / epw)
    packed_N = math.ceil(n_seq / epw)

    # Per-lane sub-module SRAM depths (the sub-modules expect these)
    # We reuse existing dimm_score_matrix/softmax_approx/dimm_weighted_sum unchanged,
    # each sized for 1 row of (N) scores or (d) outputs.
    depth_q = packed_d + 1
    depth_k = n_seq * packed_d + 1
    depth_v = n_seq * packed_d + 1
    depth_score = n_seq + 1
    depth_exp = n_seq + 1
    depth_out = d_head + 1
    depth_log = n_seq + 1

    # h_dimm for sub-modules: 1 in our BERT-Tiny config (C ≥ max(d, N))
    h_dimm = 1

    L = []
    L.append(f"// NL-DPE Full DIMM Top (W={W} parallel lanes)")
    L.append(f"// Per-head DPEs: 4 stages × {W} lanes = {4*W}")
    L.append(f"// K_qkt={K_qkt} (dual-identity for C={C}/d={d_head})")
    L.append(f"// K_sv={K_sv}")
    L.append(f"")
    L.append(f"module nldpe_dimm_top #(")
    L.append(f"    parameter N = {n_seq},")
    L.append(f"    parameter D = {d_head},")
    L.append(f"    parameter W = {W},")
    L.append(f"    parameter DATA_WIDTH = {dw}")
    L.append(f")(")
    L.append(f"    input wire clk, rst,")
    L.append(f"    input wire valid_q, valid_k, valid_v, ready_n,")
    L.append(f"    input wire [DATA_WIDTH-1:0] data_in_q, data_in_k, data_in_v,")
    L.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    L.append(f"    output wire ready_q, ready_k, ready_v, valid_n")
    L.append(f");")
    L.append(f"")

    # Per-lane output wires
    L.append(f"    // Per-lane valid/ready/data arrays")
    L.append(f"    wire [W-1:0] lane_valid_n;")
    L.append(f"    wire [W-1:0] lane_ready_q, lane_ready_k, lane_ready_v;")
    L.append(f"    wire [DATA_WIDTH-1:0] lane_data_out [0:W-1];")
    L.append(f"")

    # Instantiate W parallel pipelines via generate block
    L.append(f"    // W parallel DIMM pipelines (each = score_matrix + softmax + weighted_sum)")
    L.append(f"    genvar lane;")
    L.append(f"    generate for (lane = 0; lane < W; lane = lane + 1) begin : dimm_lane")
    L.append(f"        wire [DATA_WIDTH-1:0] score_out, softmax_out;")
    L.append(f"        wire score_valid, softmax_valid;")
    L.append(f"        wire score_ready, softmax_ready;")
    L.append(f"")
    L.append(f"        // Stage 1: dimm_score_matrix (mac_qk + softmax_exp fused)")
    L.append(f"        dimm_score_matrix #(.N(N), .d(D), .DATA_WIDTH(DATA_WIDTH)) score_inst (")
    L.append(f"            .clk(clk), .rst(rst),")
    L.append(f"            .valid_q(valid_q), .valid_k(valid_k),")
    L.append(f"            .ready_n(softmax_ready),")
    L.append(f"            .data_in_q(data_in_q), .data_in_k(data_in_k),")
    L.append(f"            .data_out(score_out),")
    L.append(f"            .ready_q(lane_ready_q[lane]), .ready_k(lane_ready_k[lane]),")
    L.append(f"            .valid_n(score_valid)")
    L.append(f"        );")
    L.append(f"")
    L.append(f"        // Stage 2: softmax_approx (normalize)")
    L.append(f"        softmax_approx #(.N(N), .d(D), .DATA_WIDTH(DATA_WIDTH)) softmax_inst (")
    L.append(f"            .clk(clk), .rst(rst),")
    L.append(f"            .valid(score_valid), .ready_n(1'b0),")
    L.append(f"            .data_in(score_out),")
    L.append(f"            .data_out(softmax_out),")
    L.append(f"            .ready(score_ready), .valid_n(softmax_valid)")
    L.append(f"        );")
    L.append(f"        assign softmax_ready = score_ready;")
    L.append(f"")
    L.append(f"        // Stage 3: dimm_weighted_sum (wsum_log + mac_sv)")
    L.append(f"        dimm_weighted_sum #(.N(N), .d(D), .DATA_WIDTH(DATA_WIDTH)) wsum_inst (")
    L.append(f"            .clk(clk), .rst(rst),")
    L.append(f"            .valid_attn(softmax_valid), .valid_v(valid_v),")
    L.append(f"            .ready_n(1'b0),")
    L.append(f"            .data_in_attn(softmax_out), .data_in_v(data_in_v),")
    L.append(f"            .data_out(lane_data_out[lane]),")
    L.append(f"            .ready_attn(), .ready_v(lane_ready_v[lane]),")
    L.append(f"            .valid_n(lane_valid_n[lane])")
    L.append(f"        );")
    L.append(f"    end endgenerate")
    L.append(f"")

    # Top-level output muxing (round-robin across lanes)
    L.append(f"    // Output mux: round-robin across W lanes")
    L.append(f"    reg [$clog2(W)-1:0] out_lane_sel;")
    L.append(f"    wire any_valid = |lane_valid_n;")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) out_lane_sel <= 0;")
    L.append(f"        else if (any_valid && !ready_n) out_lane_sel <= out_lane_sel + 1;")
    L.append(f"    end")
    L.append(f"    assign data_out = lane_data_out[out_lane_sel];")
    L.append(f"    assign valid_n = any_valid;")
    L.append(f"")

    # ready_q/ready_k/ready_v: aggregate across lanes (all lanes ready)
    L.append(f"    assign ready_q = &lane_ready_q;")
    L.append(f"    assign ready_k = &lane_ready_k;")
    L.append(f"    assign ready_v = &lane_ready_v;")
    L.append(f"")
    L.append(f"endmodule")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "fc_verification" / "rtl"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dw = 40
    epw = dw // 8
    packed_d = math.ceil(args.d / epw)
    packed_N = math.ceil(args.N / epw)
    h_dimm = 1

    # Existing sub-module generators need SRAM depths explicitly.
    score_rtl = _gen_dimm_score_matrix(
        n_seq=args.N, d_head=args.d, h_dimm=h_dimm,
        depth_q=packed_d + 1,
        depth_k=args.N * packed_d + 1,
        depth_score=args.N + 1,
        data_width=dw,
        dual_identity=(args.C // args.d >= 2),
        uid=0,
    )
    softmax_rtl = _gen_softmax_dpe(
        n_seq=args.N, d_head=args.d, h_dimm=h_dimm,
        depth_in=args.N + 1, depth_exp=args.N + 1, depth_out=args.N + 1,
        data_width=dw, uid=0,
    )
    wsum_rtl = _gen_dimm_weighted_sum(
        n_seq=args.N, d_head=args.d, h_dimm=h_dimm,
        depth_attn=args.N + 1,
        depth_v=args.N * args.d + 1,   # element-wise V: FSM writes N*d words, reads j*d+m
        depth_log=args.N + 1,
        depth_out=args.d + 1,
        data_width=dw, uid=0,
    )

    top_rtl = _gen_dimm_top_nldpe(
        n_seq=args.N, d_head=args.d, crossbar_cols=args.C, W=args.W, data_width=dw,
    )

    supporting = _get_supporting_modules()

    path = out_dir / f"nldpe_dimm_top_d{args.d}_c{args.C}.v"
    with open(path, "w") as f:
        f.write(f"// NL-DPE Full DIMM Top — W={args.W}, N={args.N}, d={args.d}, C={args.C}\n")
        f.write(f"// Generated by gen_dimm_nldpe_top.py\n\n")
        f.write(top_rtl)
        f.write("\n\n// ════ Sub-modules ════\n\n")
        f.write(score_rtl)
        f.write("\n\n")
        f.write(softmax_rtl)
        f.write("\n\n")
        f.write(wsum_rtl)
        f.write("\n\n// ════ Supporting modules ════\n\n")
        f.write(supporting)

    print(f"Generated: {path}")
    print(f"  W={args.W} lanes × 4 DIMM stages = {4*args.W} DPEs")


if __name__ == "__main__":
    main()
