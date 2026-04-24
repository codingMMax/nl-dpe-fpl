// NL-DPE Attention Head Top — composed FC_Q/K/V + DIMM + FC_O
// Generated for AH track T2 (config: N=128, d_head=64, C=128, W=16).
//
// Composition (no back-pressure: ready_n hardwired to 0 throughout):
//
//   data_in_x ──► fc_top_qkv (Q proj, K=128, N=64, V=1, H=1, ACAM)
//          └────► fc_top_qkv (K proj)
//          └────► fc_top_qkv (V proj)
//                   │             │             │
//                   ▼             ▼             ▼
//          nldpe_dimm_top #(N=128, D=64, C=128, W=16)
//                                 │
//                                 ▼
//                         fc_top_o (O proj, K=64, N=128, V=1, H=1, ACAM)
//                                 │
//                                 ▼
//                            data_out
//
// Resource expectation:
//   - 3 × fc_top_qkv = 3 DPE instances (Q/K/V projections, V=H=1)
//   - 1 × nldpe_dimm_top with W=16 lanes × 4 stages = 64 DPE instances
//   - 1 × fc_top_o = 1 DPE instance (O projection, V=H=1)
//   Total DPE instances: 3 + 64 + 1 = 68
//
// Functional caveat (deferred to T3/T4): single-token X drives Q/K/V FCs
// in parallel, producing one Q vector and one K/V row each. To populate
// the DIMM's K/V SRAMs (N=128 tokens), the upstream feeder must drive X
// 128 times sequentially or pre-load K/V SRAMs hierarchically. T2's gate
// is iverilog syntax-clean + resource count; full functional verification
// is T3.
//
// Module dependencies (passed to iverilog together):
//   - fc_verification/rtl/nldpe_dimm_top_d64_c128.v (provides nldpe_dimm_top
//     + dimm_score_matrix + softmax_approx + dimm_weighted_sum +
//     conv_layer_single_dpe + conv_controller + sram + global_controller +
//     controller_scalable + xbar_ip_module)
//   - fc_verification/rtl/nldpe/fc_top_qkv.v  (provides fc_top_qkv module)
//   - fc_verification/rtl/nldpe/fc_top_o.v    (provides fc_top_o module)
//   - fc_verification/dpe_stub.v              (provides dpe behavioral stub)

`timescale 1ns / 1ps

module nldpe_attn_head_d64_c128 #(
    parameter DATA_WIDTH = 40,
    parameter N_SEQ      = 128,
    parameter D_HEAD     = 64,
    parameter C          = 128,
    parameter W          = 16
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    valid_x,
    input  wire                    ready_n,
    input  wire [DATA_WIDTH-1:0]   data_in_x,
    output wire [DATA_WIDTH-1:0]   data_out,
    output wire                    valid_n,
    output wire                    ready_x
);

    // ── Q/K/V projections (each takes the same broadcast X) ────────────
    wire [DATA_WIDTH-1:0] q_out, k_out, v_out;
    wire q_valid, k_valid, v_valid;
    wire q_ready, k_ready, v_ready;

    fc_top_qkv #(.DATA_WIDTH(DATA_WIDTH)) fc_q_inst (
        .clk(clk), .rst(rst),
        .valid(valid_x), .ready_n(1'b0),
        .data_in(data_in_x),
        .data_out(q_out), .ready(q_ready), .valid_n(q_valid)
    );
    fc_top_qkv #(.DATA_WIDTH(DATA_WIDTH)) fc_k_inst (
        .clk(clk), .rst(rst),
        .valid(valid_x), .ready_n(1'b0),
        .data_in(data_in_x),
        .data_out(k_out), .ready(k_ready), .valid_n(k_valid)
    );
    fc_top_qkv #(.DATA_WIDTH(DATA_WIDTH)) fc_v_inst (
        .clk(clk), .rst(rst),
        .valid(valid_x), .ready_n(1'b0),
        .data_in(data_in_x),
        .data_out(v_out), .ready(v_ready), .valid_n(v_valid)
    );

    // ── DIMM (W=16 lanes; 64 internal DPE instances for QK^T+softmax+SV) ──
    wire [DATA_WIDTH-1:0] dimm_out;
    wire dimm_valid_n;
    nldpe_dimm_top #(
        .N(N_SEQ), .D(D_HEAD), .W(W), .DATA_WIDTH(DATA_WIDTH)
    ) dimm_inst (
        .clk(clk), .rst(rst),
        .valid_q(q_valid), .valid_k(k_valid), .valid_v(v_valid),
        .ready_n(1'b0),
        .data_in_q(q_out), .data_in_k(k_out), .data_in_v(v_out),
        .data_out(dimm_out),
        .ready_q(), .ready_k(), .ready_v(),
        .valid_n(dimm_valid_n)
    );

    // ── O projection (DIMM output → final output token) ────────────────
    fc_top_o #(.DATA_WIDTH(DATA_WIDTH)) fc_o_inst (
        .clk(clk), .rst(rst),
        .valid(dimm_valid_n), .ready_n(ready_n),
        .data_in(dimm_out),
        .data_out(data_out), .ready(), .valid_n(valid_n)
    );

    // ── Top-level handshake ───────────────────────────────────────────
    assign ready_x = q_ready & k_ready & v_ready;

endmodule
