// DEPRECATED (superseded by W=16 full DIMM top verification).
// This file was part of the Phase A-E per-stage W=1 DIMM exploration.
// Authoritative DIMM verification: fc_verification/rtl/nldpe_dimm_top_d64_c128.v
// and fc_verification/rtl/azurelily_dimm_top_d64_c128.v; see VERIFICATION.md.
//
// Azure-Lily DIMM mac_sv Functional Test
//
// Computes output[m] = Σ_k S[k] × V[k][m]  for m = 0..d-1
//   S: N=128 attention weights (packed)
//   V: N×d value matrix, stored column-major (col m at addr m*PACKED_N..(m+1)*PACKED_N-1)
//   output: d values, one per SRAM entry
//
// Test input:
//   S = [3, 2, 1, 0, 0, ..., 0]
//   V = identity-like: V[k][m] = δ(k, m) for k, m < min(N, d)
//
// Expected:
//   output[0] = 3·V[0][0] + 2·V[1][0] + 1·V[2][0] + 0·... = 3·1 = 3
//   output[1] = 3·0 + 2·1 + 1·0 = 2
//   output[2] = 3·0 + 2·0 + 1·1 = 1
//   output[3..d-1] = 0

`timescale 1ns / 1ps

module tb_azurelily_dimm_mac_sv_functional;

    parameter DW = 40;
    parameter N = 128;
    parameter D = 64;
    parameter EPW = DW / 8;
    parameter PACKED_N = (N + EPW - 1) / EPW;  // 26

    reg clk, rst, valid_s, valid_v, ready_n;
    reg  [DW-1:0] data_in_s, data_in_v;
    wire [DW-1:0] data_out;
    wire valid_n;

    azurelily_dimm_mac_sv #(.DATA_WIDTH(DW)) dut (
        .clk(clk), .rst(rst),
        .valid_s(valid_s), .valid_v(valid_v), .ready_n(ready_n),
        .data_in_s(data_in_s), .data_in_v(data_in_v),
        .data_out(data_out), .valid_n(valid_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle, i, j, k;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    reg [7:0] expected [0:D-1];

    initial begin
        rst = 1; valid_s = 0; valid_v = 0; ready_n = 1;
        data_in_s = 0; data_in_v = 0;
        #20; rst = 0; #15;

        $display("=== Azure-Lily DIMM mac_sv Functional Test ===");
        $display("  Config: N=%0d, d=%0d", N, D);
        $display("  Primitive: dsp_mac(K=PACKED_N=%0d)", PACKED_N);
        $display("");

        expected[0] = 8'd3;
        expected[1] = 8'd2;
        expected[2] = 8'd1;
        for (i = 3; i < D; i = i + 1) expected[i] = 8'd0;

        // Pre-load S: [3, 2, 1, 0, ..., 0]
        #1;
        for (i = 0; i < PACKED_N; i = i + 1) dut.s_sram.mem[i] = 0;
        dut.s_sram.mem[0][0*8 +: 8] = 8'd3;
        dut.s_sram.mem[0][1*8 +: 8] = 8'd2;
        dut.s_sram.mem[0][2*8 +: 8] = 8'd1;

        // Pre-load V: column-major, V[k][m] = δ(k, m).
        // Col m is at addresses m*PACKED_N..(m+1)*PACKED_N-1.
        // In col m: only word floor(m/EPW), byte (m%EPW) is 1 (=V[m][m]).
        for (j = 0; j < D; j = j + 1) begin
            for (k = 0; k < PACKED_N; k = k + 1) dut.v_sram.mem[j * PACKED_N + k] = 0;
            if (j < N) begin
                // V[j][j] = 1 for j < min(N, D) = D.
                // Store at col-j offset j*PACKED_N, word j/EPW, byte j%EPW.
                dut.v_sram.mem[j * PACKED_N + j / EPW][(j % EPW) * 8 +: 8] = 8'd1;
            end
        end

        // Skip S_LOAD_S/S_LOAD_V, set FSM to S_COMPUTE
        dut.s_w_addr = PACKED_N;
        dut.v_w_addr = D * PACKED_N;
        dut.state = 3;
        dut.m_count = 0;
        dut.k_count = 0;
        dut.mac_count = 0;

        $display("  S: [3, 2, 1, 0, ..., 0]   (N=%0d)", N);
        $display("  V: identity V[j][j]=1 for j<D");
        $display("  Expected output: [3, 2, 1, 0, 0, ..., 0]");
        $display("");

        // Wait for S_OUTPUT (state=4)
        while (dut.state != 4 && cycle < 100000) @(posedge clk);

        if (dut.state == 4) begin
            @(posedge clk); @(posedge clk); #1;
            $display("=== Results ===");
            $display("  output[0] = %0d  (expect %0d)", dut.o_sram.mem[0][7:0], expected[0]);
            $display("  output[1] = %0d  (expect %0d)", dut.o_sram.mem[1][7:0], expected[1]);
            $display("  output[2] = %0d  (expect %0d)", dut.o_sram.mem[2][7:0], expected[2]);
            $display("  output[3] = %0d  (expect %0d)", dut.o_sram.mem[3][7:0], expected[3]);
            $display("  output[%0d] = %0d  (expect %0d)", D-1,
                dut.o_sram.mem[D-1][7:0], expected[D-1]);
            $display("  Total cycles: %0d", cycle);
            $display("");

            begin : check
                integer err, oi;
                err = 0;
                for (oi = 0; oi < D; oi = oi + 1) begin
                    if (dut.o_sram.mem[oi][7:0] !== expected[oi]) begin
                        if (err < 5) $display("  MISMATCH: output[%0d]=%0d (expect %0d)",
                            oi, dut.o_sram.mem[oi][7:0], expected[oi]);
                        err = err + 1;
                    end
                end
                if (err == 0)
                    $display("  PASS: mac_sv functional (%0d outputs correct)", D);
                else
                    $display("  FAIL: %0d mismatches out of %0d", err, D);
            end
        end else begin
            $display("  TIMEOUT at cycle %0d, state=%0d", cycle, dut.state);
        end

        $finish;
    end

endmodule
