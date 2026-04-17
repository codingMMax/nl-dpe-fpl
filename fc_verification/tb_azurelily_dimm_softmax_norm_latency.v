// DEPRECATED (superseded by W=16 full DIMM top verification).
// This file was part of the Phase A-E per-stage W=1 DIMM exploration.
// Authoritative DIMM verification: fc_verification/rtl/nldpe_dimm_top_d64_c128.v
// and fc_verification/rtl/azurelily_dimm_top_d64_c128.v; see VERIFICATION.md.
//
// Azure-Lily DIMM softmax_norm Latency (S_INV + S_NORM phases)
//
// Simulator reference: norm_fpga / CLB norm.
//   reduction (sum): log2(N) CLB add stages   = 7
//   reciprocal     : 1 cycle (LUT)
//   multiply+normalize: N cycles (streaming over N exp values)
// Total cycles: ~log2(N) + 1 + N = 7 + 1 + 128 = 136

`timescale 1ns / 1ps

module tb_azurelily_dimm_softmax_norm_latency;
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

    integer T_s_inv_start, T_s_norm_start, T_s_norm_end;
    reg [2:0] prev_state;
    always @(posedge clk) begin
        if (!rst) begin
            if (prev_state !== dut.inst.state) begin
                if (dut.inst.state == 2 && T_s_inv_start < 0) T_s_inv_start = cycle;
                if (dut.inst.state == 3 && T_s_norm_start < 0) T_s_norm_start = cycle;
                if (prev_state == 3 && dut.inst.state == 0) T_s_norm_end = cycle;
            end
            prev_state <= dut.inst.state;
        end
    end

    initial begin
        rst = 1; valid = 0; ready_n = 1; data_in = 0;
        T_s_inv_start = -1; T_s_norm_start = -1; T_s_norm_end = -1;
        prev_state = 0;
        #20; rst = 0; #15;

        // Feed N zeros through S_LOAD
        for (i = 0; i < N; i = i + 1) begin
            @(posedge clk); #1;
            valid = 1;
            data_in = 0;
        end
        @(posedge clk); #1;
        valid = 0;

        // Wait for state to return to S_LOAD (end of S_NORM)
        while (T_s_norm_end < 0 && cycle < 1000) @(posedge clk);

        $display("=== Azure-Lily softmax_norm Latency (S_INV + S_NORM) ===");
        $display("  Config: N=%0d", N);
        $display("");
        $display("RTL Measured:");
        $display("  T_s_inv_start  : cycle %0d", T_s_inv_start);
        $display("  T_s_norm_start : cycle %0d", T_s_norm_start);
        $display("  T_s_norm_end   : cycle %0d", T_s_norm_end);
        $display("  S_INV cycles   : %0d (expect 1)", T_s_norm_start - T_s_inv_start);
        $display("  S_NORM cycles  : %0d (expect N=%0d)", T_s_norm_end - T_s_norm_start, N);
        $display("  Total norm cyc : %0d", T_s_norm_end - T_s_inv_start);
        $display("");
        $display("Simulator Reference (norm_fpga):");
        $display("  reduction sum  : ceil(log2(N)) = %0d cycles", $clog2(N));
        $display("  reciprocal     : 1 cycle (LUT)");
        $display("  multiply       : N = %0d cycles", N);
        $display("  total          : %0d cycles", $clog2(N) + 1 + N);
        $display("");
        $display("Delta: RTL %0d vs Sim %0d = %0d cycles",
            T_s_norm_end - T_s_inv_start,
            $clog2(N) + 1 + N,
            (T_s_norm_end - T_s_inv_start) - ($clog2(N) + 1 + N));
        $display("Note: RTL does not pre-reduce; sum accumulates during S_LOAD (which is");
        $display("      counted in softmax_exp latency). Simulator models sum reduction");
        $display("      as a separate log2(N) stage — accounts for the delta.");

        $finish;
    end
endmodule
