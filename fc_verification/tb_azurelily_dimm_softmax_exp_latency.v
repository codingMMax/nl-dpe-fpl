// DEPRECATED (superseded by W=16 full DIMM top verification).
// This file was part of the Phase A-E per-stage W=1 DIMM exploration.
// Authoritative DIMM verification: fc_verification/rtl/nldpe_dimm_top_d64_c128.v
// and fc_verification/rtl/azurelily_dimm_top_d64_c128.v; see VERIFICATION.md.
//
// Azure-Lily DIMM softmax_exp Latency (S_LOAD phase only)
//
// Simulator reference: exp_fpga is in-simulator model for CLB-based exp.
// Per-element cycles: 1 (combinational exp + SRAM write per cycle).
// Total cycles for N elements: N.

`timescale 1ns / 1ps

module tb_azurelily_dimm_softmax_exp_latency;
    parameter DW = 40;
    parameter N = 128;

    reg clk, rst, valid, ready_n;
    reg  [DW-1:0] data_in;
    wire [DW-1:0] data_out;
    wire ready, valid_n;

    azurelily_dimm_softmax #(.DATA_WIDTH(DW), .N(N)) dut (
        .clk(clk), .rst(rst),
        .valid(valid), .ready_n(ready_n),
        .data_in(data_in),
        .data_out(data_out),
        .ready(ready), .valid_n(valid_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle, i;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    integer T_load_start, T_load_end;

    initial begin
        rst = 1; valid = 0; ready_n = 1; data_in = 0;
        T_load_start = -1; T_load_end = -1;
        #20; rst = 0; #15;

        T_load_start = cycle;
        for (i = 0; i < N; i = i + 1) begin
            @(posedge clk); #1;
            valid = 1;
            data_in = 0;
        end
        T_load_end = cycle;
        @(posedge clk); #1;
        valid = 0;

        $display("=== Azure-Lily softmax_exp Latency (S_LOAD phase) ===");
        $display("  Config: N=%0d", N);
        $display("");
        $display("RTL Measured:");
        $display("  T_load_start : cycle %0d", T_load_start);
        $display("  T_load_end   : cycle %0d", T_load_end);
        $display("  Total cycles : %0d", T_load_end - T_load_start);
        $display("  Per-element  : %.1f", (T_load_end - T_load_start) * 1.0 / N);
        $display("");
        $display("Simulator Reference (exp_fpga / CLB exp):");
        $display("  Per-element  : 1 cycle (combinational exp + SRAM write)");
        $display("  Total cycles : N = %0d", N);
        $display("");
        $display("Delta: RTL %0d vs Sim %0d = %0d cycles",
            T_load_end - T_load_start, N, (T_load_end - T_load_start) - N);

        $finish;
    end
endmodule
