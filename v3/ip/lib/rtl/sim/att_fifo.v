// ============================================================================
// att_fifo.v — plain first-in first-out buffer between two stations of the attention
// kernel (rework M1, user 2026-10-08: stage-to-stage buffers are FIFOs; whether the
// array becomes memory blocks or logic is left to the synthesis flow). One write port,
// one read port; the read data is registered (valid the cycle after rd_en), so the
// array can map to a block RAM. The caller pops only when count != 0 and pushes only
// when count != DEPTH (or proves it by construction); `overflow` / `underflow` are
// sticky probes for the TBs. Any DEPTH >= 2 (the pointers wrap at DEPTH, so a FIFO sized
// to its exact content wastes no memory; M5: the layer's O FIFOs). Identical in the
// simulation and VTR sources.
// ============================================================================
`timescale 1ns / 1ps

module att_fifo #(
    parameter W     = 40,
    parameter DEPTH = 512,                     // >= 2
    parameter AW    = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  wire          clk,
    input  wire          reset,
    input  wire          wr_en,
    input  wire [W-1:0]  wr_data,
    input  wire          rd_en,
    output reg  [W-1:0]  rd_data,
    output reg  [AW:0]   count,
    output reg           overflow,
    output reg           underflow
);
    generate
        if (DEPTH < 2) begin : g_bad_param
            att_fifo_parameter_outside_domain u_bad ();
        end
    endgenerate
    reg [W-1:0]  mem [0:DEPTH-1];
    reg [AW-1:0] wp, rp;
    always @(posedge clk) begin
        if (wr_en) mem[wp] <= wr_data;
        if (rd_en) rd_data <= mem[rp];
    end
    always @(posedge clk) begin
        if (reset) begin
            wp <= 0; rp <= 0; count <= 0; overflow <= 1'b0; underflow <= 1'b0;
        end
        else begin
            if (wr_en) wp <= (wp == DEPTH - 1) ? {AW{1'b0}} : wp + 1'b1;
            if (rd_en) rp <= (rp == DEPTH - 1) ? {AW{1'b0}} : rp + 1'b1;
            count <= count + (wr_en ? 1'b1 : 1'b0) - (rd_en ? 1'b1 : 1'b0);
            if (wr_en && !rd_en && count == DEPTH) overflow <= 1'b1;
            if (rd_en && count == 0) underflow <= 1'b1;
        end
    end
endmodule
