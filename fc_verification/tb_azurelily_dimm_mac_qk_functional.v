// DEPRECATED (superseded by W=16 full DIMM top verification).
// This file was part of the Phase A-E per-stage W=1 DIMM exploration.
// Authoritative DIMM verification: fc_verification/rtl/nldpe_dimm_top_d64_c128.v
// and fc_verification/rtl/azurelily_dimm_top_d64_c128.v; see VERIFICATION.md.
//
// Azure-Lily DIMM mac_qk Functional Test
//
// Computes score[j] = Σ_k Q[k] × K[j][k]  for j = 0..N-1
//   Q: d=64 values packed
//   K: N=128 rows × d=64 cols, packed per row
//   score: N output values
//
// Test input:
//   Q    = [3, 2, 1, 0, 0, ..., 0]  (d=64 elements)
//   K[j] = identity-like: K[j][k] = δ(j,k) for j < d, else 0
//
// Expected scores:
//   score[0] = 3·K[0][0] + 2·K[0][1] + 1·K[0][2] + 0·... = 3·1 + 2·0 + 1·0 = 3
//   score[1] = 3·K[1][0] + 2·K[1][1] + 1·K[1][2] + 0·... = 3·0 + 2·1 + 1·0 = 2
//   score[2] = 3·K[2][0] + 2·K[2][1] + 1·K[2][2] + 0·... = 3·0 + 2·0 + 1·1 = 1
//   score[3..d-1] = 0
//   score[d..N-1] = 0 (K[j]=0 for j >= d)

`timescale 1ns / 1ps

module tb_azurelily_dimm_mac_qk_functional;

    parameter DW = 40;
    parameter N = 128;
    parameter D = 64;
    parameter EPW = DW / 8;
    parameter PACKED_D = (D + EPW - 1) / EPW;  // 13

    reg clk, rst, valid_q, valid_k, ready_n;
    reg  [DW-1:0] data_in_q, data_in_k;
    wire [DW-1:0] data_out;
    wire valid_n;

    azurelily_dimm_mac_qk #(.DATA_WIDTH(DW)) dut (
        .clk(clk), .rst(rst),
        .valid_q(valid_q), .valid_k(valid_k), .ready_n(ready_n),
        .data_in_q(data_in_q), .data_in_k(data_in_k),
        .data_out(data_out), .valid_n(valid_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle, i, j;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    reg [7:0] expected [0:N-1];

    initial begin
        rst = 1; valid_q = 0; valid_k = 0; ready_n = 1;
        data_in_q = 0; data_in_k = 0;
        #20; rst = 0; #15;

        $display("=== Azure-Lily DIMM mac_qk Functional Test ===");
        $display("  Config: N=%0d, d=%0d, DPE_bus=%0d", N, D, DW);
        $display("  Primitive: dsp_mac(K=PACKED_D=%0d)", PACKED_D);
        $display("");

        // Expected
        expected[0] = 8'd3;
        expected[1] = 8'd2;
        expected[2] = 8'd1;
        for (i = 3; i < N; i = i + 1) expected[i] = 8'd0;

        // Pre-load Q: [3, 2, 1, 0, 0, ..., 0]
        #1;
        for (i = 0; i < PACKED_D; i = i + 1) dut.q_sram.mem[i] = 0;
        dut.q_sram.mem[0][0*8 +: 8] = 8'd3;
        dut.q_sram.mem[0][1*8 +: 8] = 8'd2;
        dut.q_sram.mem[0][2*8 +: 8] = 8'd1;

        // Pre-load K: K[j][k] = δ(j,k) for j < D
        for (j = 0; j < N; j = j + 1) begin
            for (i = 0; i < PACKED_D; i = i + 1) dut.k_sram.mem[j * PACKED_D + i] = 0;
            if (j < D) begin
                dut.k_sram.mem[j * PACKED_D + j / EPW][(j % EPW) * 8 +: 8] = 8'd1;
            end
        end

        // Skip S_LOAD_Q/S_LOAD_K, set FSM to S_COMPUTE
        dut.q_w_addr = PACKED_D;
        dut.k_w_addr = N * PACKED_D;
        dut.state = 3;        // S_COMPUTE
        dut.j_count = 0;
        dut.k_count = 0;
        dut.mac_count = 0;

        $display("  Q: [3, 2, 1, 0, ..., 0]   (d=%0d)", D);
        $display("  K: identity-like for j < %0d", D);
        $display("  Expected scores: [3, 2, 1, 0, 0, ..., 0]");
        $display("");

        // Wait for S_OUTPUT (state=5)
        while (dut.state != 5 && cycle < 100000) @(posedge clk);

        if (dut.state == 5) begin
            @(posedge clk); @(posedge clk); #1;
            $display("=== Results ===");
            $display("  score[0] = %0d  (expect %0d)", dut.score_sram.mem[0][7:0], expected[0]);
            $display("  score[1] = %0d  (expect %0d)", dut.score_sram.mem[1][7:0], expected[1]);
            $display("  score[2] = %0d  (expect %0d)", dut.score_sram.mem[2][7:0], expected[2]);
            $display("  score[3] = %0d  (expect %0d)", dut.score_sram.mem[3][7:0], expected[3]);
            $display("  score[%0d] = %0d  (expect %0d)", D, dut.score_sram.mem[D][7:0], expected[D]);
            $display("  Total cycles: %0d", cycle);
            $display("");

            begin : check
                integer err, oi;
                err = 0;
                for (oi = 0; oi < N; oi = oi + 1) begin
                    if (dut.score_sram.mem[oi][7:0] !== expected[oi]) begin
                        if (err < 5) $display("  MISMATCH: score[%0d]=%0d (expect %0d)",
                            oi, dut.score_sram.mem[oi][7:0], expected[oi]);
                        err = err + 1;
                    end
                end
                if (err == 0)
                    $display("  PASS: mac_qk functional (%0d scores correct)", N);
                else
                    $display("  FAIL: %0d score mismatches out of %0d", err, N);
            end
        end else begin
            $display("  TIMEOUT at cycle %0d, state=%0d", cycle, dut.state);
        end

        $finish;
    end

endmodule
