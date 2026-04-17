// Azure-Lily DIMM softmax_exp Functional Test
//
// Covers the S_LOAD phase of clb_softmax: combinational exp approximation
// and running sum.
//
// clb_softmax exp formula (saturating shift):
//   exp_val = (1 << (data_in[4:0] + 10))  for input in [-20, 20]
//           = 2^20                         for input > 20
//           = 1                            for input < -20
//
// Test input: all-zeros (data_in = 0 for N=128 cycles)
//   exp_val per cycle = 1 << (0 + 10) = 1024
//   sum_exp after N cycles = 128 × 1024 = 131072 = 0x20000

`timescale 1ns / 1ps

module tb_azurelily_dimm_softmax_exp_functional;

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

    initial begin
        rst = 1; valid = 0; ready_n = 1; data_in = 0;
        #20; rst = 0; #15;

        $display("=== Azure-Lily softmax_exp Functional Test ===");
        $display("  Config: N=%0d, DATA_WIDTH=%0d", N, DW);
        $display("  Input: all-zeros (for N cycles)");
        $display("  exp_val per cycle = 1 << 10 = 1024");
        $display("  Expected sum_exp = %0d × 1024 = %0d", N, N * 1024);
        $display("");

        // Feed N zeros
        for (i = 0; i < N; i = i + 1) begin
            @(posedge clk); #1;
            valid = 1;
            data_in = 0;
        end
        @(posedge clk); #1;
        valid = 0;

        // Now state should be S_INV (=2) with sum_exp = 131072
        $display("Results after S_LOAD phase:");
        $display("  FSM state       : %0d (expect 2 = S_INV)", dut.inst.state);
        $display("  sum_exp         : %0d (expect %0d)", dut.inst.sum_exp, N * 1024);
        $display("");

        // Verify sum_exp
        if (dut.inst.sum_exp === N * 1024)
            $display("  PASS: sum_exp correct (= N × exp(0) = %0d)", N * 1024);
        else
            $display("  FAIL: sum_exp=%0d, expected %0d", dut.inst.sum_exp, N * 1024);

        // Spot-check SRAM entries
        $display("");
        $display("SRAM exp buffer spot check (all should be 1024):");
        $display("  sm_buf[0]   = %0d", dut.inst.sm_buf.mem[0]);
        $display("  sm_buf[63]  = %0d", dut.inst.sm_buf.mem[63]);
        $display("  sm_buf[127] = %0d", dut.inst.sm_buf.mem[127]);
        begin : sram_check
            integer err, si;
            err = 0;
            for (si = 0; si < N; si = si + 1)
                if (dut.inst.sm_buf.mem[si] !== 1024) err = err + 1;
            if (err == 0)
                $display("  PASS: all %0d SRAM entries = 1024", N);
            else
                $display("  FAIL: %0d SRAM mismatches", err);
        end

        $finish;
    end

endmodule
