// Behavioral stubs for Azure-Lily FC iverilog simulation.
//
// These mirror the inline definitions in
// fc_verification/rtl/azurelily_dimm_top_d64_c128.v (Phase 6A canonical).
// Kept separate so that fc_verification/rtl/azurelily/azurelily_fc_*.v
// can be compiled standalone without dragging in the DIMM top module.
//
// Modules:
//   - int_sop_4 : behavioral 4-pair int8 sum-of-products (DSP hard block)
//   - dsp_mac   : pure 4-wide MAC accumulator wrapping int_sop_4
//   - sram      : parameterized single-port registered-read SRAM

`timescale 1ns / 1ps

// ════ int_sop_4 (behavioral DSP stub) ════
module int_sop_4 (
    input  wire                       clk,
    input  wire                       reset,
    input  wire [11:0]                mode_sigs,
    input  wire signed [8:0]          ax, ay,
    input  wire signed [8:0]          bx, by,
    input  wire signed [8:0]          cx, cy,
    input  wire signed [8:0]          dx, dy,
    input  wire signed [63:0]         chainin,
    output wire signed [63:0]         result,
    output wire signed [63:0]         chainout
);
    wire signed [17:0] p_a = ax * ay;
    wire signed [17:0] p_b = bx * by;
    wire signed [17:0] p_c = cx * cy;
    wire signed [17:0] p_d = dx * dy;
    wire signed [63:0] total = $signed(p_a) + $signed(p_b)
                             + $signed(p_c) + $signed(p_d)
                             + $signed(chainin);
    assign result   = total;
    assign chainout = total;
endmodule

// ════ dsp_mac (pure 4-wide int_sop_4, no CLB helper) — Phase 6A ════
module dsp_mac #(
    parameter DATA_WIDTH = 40,
    parameter K = 64
)(
    input  wire                   clk, rst, valid, ready_n,
    input  wire [DATA_WIDTH-1:0]  data_a,
    input  wire [DATA_WIDTH-1:0]  data_b,
    output wire [DATA_WIDTH-1:0]  data_out,
    output wire                   ready, valid_n
);
    localparam ADDR_W = $clog2(K+1);
    reg [ADDR_W-1:0] count;
    reg out_valid;

    wire [8:0] ax = {data_a[ 7], data_a[ 7: 0]};
    wire [8:0] ay = {data_b[ 7], data_b[ 7: 0]};
    wire [8:0] bx = {data_a[15], data_a[15: 8]};
    wire [8:0] by = {data_b[15], data_b[15: 8]};
    wire [8:0] cx = {data_a[23], data_a[23:16]};
    wire [8:0] cy = {data_b[23], data_b[23:16]};
    wire [8:0] dx = {data_a[31], data_a[31:24]};
    wire [8:0] dy = {data_b[31], data_b[31:24]};

    wire [63:0] sop_result;
    wire [63:0] sop_chainout;
    int_sop_4 sop_inst (
        .clk(clk), .reset(rst),
        .mode_sigs(12'b0),
        .ax(ax), .ay(ay),
        .bx(bx), .by(by),
        .cx(cx), .cy(cy),
        .dx(dx), .dy(dy),
        .chainin(64'b0),
        .result(sop_result),
        .chainout(sop_chainout)
    );

    wire signed [DATA_WIDTH-1:0] cycle_sum = sop_result[DATA_WIDTH-1:0];
    reg  signed [DATA_WIDTH-1:0] accum;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accum <= 0; count <= 0; out_valid <= 0;
        end else if (valid) begin
            if (count == 0) accum <= cycle_sum;
            else            accum <= accum + cycle_sum;
            if (count == K - 1) begin
                count <= 0;
                out_valid <= 1;
            end else begin
                count <= count + 1;
                out_valid <= 0;
            end
        end else begin
            out_valid <= 0;
        end
    end
    assign data_out = accum;
    assign valid_n  = out_valid;
    assign ready    = 1'b1;
endmodule

// ════ sram (parameterized single-port registered-read) ════
module sram #(
    parameter N_CHANNELS = 1,
    parameter DATA_WIDTH = 40 * N_CHANNELS,
    parameter DEPTH      = 512
)(
    input  wire                       clk,
    input  wire                       w_en,
    input  wire                       rst,
    input  wire [$clog2(DEPTH)-1:0]   r_addr,
    input  wire [$clog2(DEPTH)-1:0]   w_addr,
    input  wire [DATA_WIDTH-1:0]      sram_data_in,
    output reg  [DATA_WIDTH-1:0]      sram_data_out
);
    reg [DATA_WIDTH-1:0] mem [DEPTH-1:0];

    integer _sram_init_i;
    initial begin
        for (_sram_init_i = 0; _sram_init_i < DEPTH; _sram_init_i = _sram_init_i + 1)
            mem[_sram_init_i] = {DATA_WIDTH{1'b0}};
    end

    always @(posedge clk) begin
        if (rst) sram_data_out <= {DATA_WIDTH{1'b0}};
        else     sram_data_out <= mem[r_addr];
    end
    always @(posedge clk) begin
        if (w_en) mem[w_addr] <= sram_data_in;
    end
endmodule
