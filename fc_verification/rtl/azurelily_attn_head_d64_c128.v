// Azure-Lily Attention Head Top — composed FC_Q/K/V + DIMM + FC_O
// Generated for AH track T2 (config: N=128, d_head=64, C=128, W=16).
//
// Composition (no back-pressure: ready_n hardwired to 0 throughout):
//
//   data_in_x ──► azurelily_fc_128_64 (Q proj, K=128, N=64, single-DSP serial)
//          └────► azurelily_fc_128_64 (K proj)
//          └────► azurelily_fc_128_64 (V proj)
//                   │             │             │
//                   ▼             ▼             ▼
//          azurelily_dimm_top #(N=128, D=64, W=16)
//                                 │
//                                 ▼
//                         azurelily_fc_64_128 (O proj, K=64, N=128)
//                                 │
//                                 ▼
//                            data_out
//
// Resource expectation:
//   - 3 × azurelily_fc_128_64 = 3 dsp_mac instances (Q/K/V projections)
//   - 1 × azurelily_dimm_top with W=16 lanes × 2 dsp stages = 32 dsp_mac
//     + 16 × clb_softmax (CLB-based, no DSP)
//   - 1 × azurelily_fc_64_128 = 1 dsp_mac (O projection)
//   Total dsp_mac instances: 3 + 32 + 1 = 36
//
// Functional caveat (deferred to T3/T4): same as NL-DPE attn head — single
// X drives Q/K/V FCs in parallel, producing one Q vector and one K/V row
// each. Real K/V SRAM population requires either a sequence of N tokens or
// hierarchical pre-load.
//
// Module dependencies (passed to iverilog together):
//   - fc_verification/rtl/azurelily_dimm_top_d64_c128.v (provides
//     azurelily_dimm_top + dsp_mac + clb_softmax + int_sop_4 + sram, all inline)
//   - fc_verification/rtl/azurelily/azurelily_fc_128_64.v (Q/K/V proj module)
//   - fc_verification/rtl/azurelily/azurelily_fc_64_128.v (O proj module)
//
// NOTE: do NOT pass azurelily_fc_stubs.v — its dsp_mac/sram/int_sop_4
// definitions collide with the inline ones in azurelily_dimm_top_d64_c128.v.

`timescale 1ns / 1ps

module azurelily_attn_head_d64_c128 #(
    parameter DATA_WIDTH = 40,
    parameter N_SEQ      = 128,
    parameter D_HEAD     = 64,
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

    // ── Q/K/V projections ──────────────────────────────────────────────
    wire [DATA_WIDTH-1:0] q_out, k_out, v_out;
    wire q_valid, k_valid, v_valid;
    wire q_ready, k_ready, v_ready;

    azurelily_fc_128_64 fc_q_inst (
        .clk(clk), .rst(rst),
        .valid(valid_x), .ready_n(1'b0),
        .data_in(data_in_x),
        .data_out(q_out), .ready(q_ready), .valid_n(q_valid)
    );
    azurelily_fc_128_64 fc_k_inst (
        .clk(clk), .rst(rst),
        .valid(valid_x), .ready_n(1'b0),
        .data_in(data_in_x),
        .data_out(k_out), .ready(k_ready), .valid_n(k_valid)
    );
    azurelily_fc_128_64 fc_v_inst (
        .clk(clk), .rst(rst),
        .valid(valid_x), .ready_n(1'b0),
        .data_in(data_in_x),
        .data_out(v_out), .ready(v_ready), .valid_n(v_valid)
    );

    // ── DIMM (W=16 lanes; 32 dsp_mac + 16 clb_softmax internally) ──────
    wire [DATA_WIDTH-1:0] dimm_out;
    wire dimm_valid_n;
    azurelily_dimm_top #(
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
    azurelily_fc_64_128 fc_o_inst (
        .clk(clk), .rst(rst),
        .valid(dimm_valid_n), .ready_n(ready_n),
        .data_in(dimm_out),
        .data_out(data_out), .ready(), .valid_n(valid_n)
    );

    // ── Top-level handshake ───────────────────────────────────────────
    assign ready_x = q_ready & k_ready & v_ready;

endmodule
