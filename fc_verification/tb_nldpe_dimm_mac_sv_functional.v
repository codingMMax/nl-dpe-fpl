// NL-DPE DIMM mac_sv Functional Test
//
// Computes output[m] = Σ_j attn[j] × V[j][m]  for m = 0..d-1
//   attn: N=128 attention weights (one row of softmax output)
//   V:    N×d value matrix, loaded into DPE crossbar as weights
//   output: d=64 values
//
// Test input:
//   attn = [3, 2, 1, 0, 0, ..., 0]  (N=128 elements, non-zero only in first 3)
//   V    = identity-like matrix: V[j][m] = δ(j,m) for j < d, else 0
//
// Expected output (d=64):
//   output[0] = 3×V[0][0] + 2×V[1][0] + 1×V[2][0] + 0×... = 3×1 + 2×0 + 1×0 = 3
//   output[1] = 3×V[0][1] + 2×V[1][1] + 1×V[2][1] + 0×... = 3×0 + 2×1 + 1×0 = 2
//   output[2] = 3×V[0][2] + 2×V[1][2] + 1×V[2][2] + 0×... = 3×0 + 2×0 + 1×1 = 1
//   output[3..63] = 0
//
// DPE: KERNEL_WIDTH = N (inputs), NUM_COLS = C (outputs), ACAM_MODE = 0 (plain VMM)

`timescale 1ns / 1ps

module tb_nldpe_dimm_mac_sv_functional;

    parameter DW = 40;
    parameter N = 128;
    parameter D = 64;
    parameter C = 128;
    parameter EPW = DW / 8;
    parameter PACKED_N = (N + EPW - 1) / EPW;  // 26
    parameter PACKED_D = (D + EPW - 1) / EPW;  // 13

    reg clk, rst, valid_in, ready_n;
    reg  [DW-1:0] data_in;
    wire [DW-1:0] data_out;
    wire ready_in, valid_n;

    // DPE parameters: KERNEL_WIDTH = N (input vector size)
    defparam dut.sv_dpe.KERNEL_WIDTH = N;
    defparam dut.sv_dpe.NUM_COLS = C;
    defparam dut.sv_dpe.ACAM_MODE = 0;       // plain VMM (no nonlinear)
    defparam dut.sv_dpe.DPE_BUF_WIDTH = DW;
    defparam dut.sv_dpe.COMPUTE_CYCLES = 3;

    nldpe_dimm_mac_sv #(.DATA_WIDTH(DW)) dut (
        .clk(clk), .rst(rst),
        .valid_in(valid_in), .ready_n(ready_n),
        .data_in(data_in),
        .data_out(data_out),
        .ready_in(ready_in), .valid_n(valid_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle, i, j;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    // V-weights: V[j][m] = δ(j,m) for j < D, else 0
    // Note: DPE weights dim is KERNEL_WIDTH × NUM_COLS = N × C = 128 × 128
    initial begin
        for (i = 0; i < N; i = i + 1)
            for (j = 0; j < C; j = j + 1)
                dut.sv_dpe.weights[i][j] = (i < D && i == j) ? 1 : 0;
    end

    reg [7:0] expected [0:D-1];

    initial begin
        rst = 1; valid_in = 0; ready_n = 1;
        data_in = 0;
        #20; rst = 0; #15;

        $display("=== NL-DPE DIMM mac_sv Functional Test ===");
        $display("  Config: N=%0d, d=%0d, C=%0d, dpe_buf_width=%0d", N, D, C, DW);
        $display("  DPE: KERNEL_WIDTH=%0d, ACAM_MODE=0 (plain VMM)", N);
        $display("");

        // Expected output
        expected[0] = 8'd3;
        expected[1] = 8'd2;
        expected[2] = 8'd1;
        for (i = 3; i < D; i = i + 1) expected[i] = 8'd0;

        // Pre-load input (attn) SRAM
        #1;
        for (i = 0; i < PACKED_N; i = i + 1) dut.in_sram.mem[i] = 0;
        dut.in_sram.mem[0][0*8 +: 8] = 8'd3;
        dut.in_sram.mem[0][1*8 +: 8] = 8'd2;
        dut.in_sram.mem[0][2*8 +: 8] = 8'd1;
        // rest = 0

        // Set FSM to S_COMPUTE
        dut.in_w_addr = PACKED_N;
        dut.state = 2;  // S_COMPUTE
        dut.mac_count = 0;

        $display("  attn loaded: [3, 2, 1, 0, 0, ..., 0] (%0d elements)", N);
        $display("  V loaded: identity for first %0d rows, else 0", D);
        $display("  Expected output: [3, 2, 1, 0, 0, ...] (first %0d)", D);
        $display("");

        // Wait for S_OUTPUT (state=4 for mac_sv)
        while (dut.state != 4 && cycle < 10000) @(posedge clk);

        if (dut.state == 4) begin
            @(posedge clk); @(posedge clk); #1;
            $display("=== Results ===");
            $display("  output[0] = %0d  (expect %0d)",
                $signed(dut.out_sram.mem[0][0*8 +: 8]), $signed(expected[0]));
            $display("  output[1] = %0d  (expect %0d)",
                $signed(dut.out_sram.mem[0][1*8 +: 8]), $signed(expected[1]));
            $display("  output[2] = %0d  (expect %0d)",
                $signed(dut.out_sram.mem[0][2*8 +: 8]), $signed(expected[2]));
            $display("  output[3] = %0d  (expect %0d)",
                $signed(dut.out_sram.mem[0][3*8 +: 8]), $signed(expected[3]));
            $display("  output[5] = %0d  (expect %0d)",
                $signed(dut.out_sram.mem[1][0*8 +: 8]), $signed(expected[5]));
            $display("  Total cycles: %0d", cycle);
            $display("");

            begin : check_outputs
                integer err, oi, word_idx, byte_idx;
                reg [7:0] actual;
                err = 0;
                for (oi = 0; oi < D; oi = oi + 1) begin
                    word_idx = oi / EPW;
                    byte_idx = oi % EPW;
                    actual = dut.out_sram.mem[word_idx][byte_idx*8 +: 8];
                    if (actual !== expected[oi]) begin
                        if (err < 5) $display("  MISMATCH: output[%0d]=%0d (expect %0d)",
                            oi, $signed(actual), $signed(expected[oi]));
                        err = err + 1;
                    end
                end
                if (err == 0)
                    $display("  PASS: mac_sv functional (%0d outputs correct)", D);
                else
                    $display("  FAIL: %0d output mismatches out of %0d", err, D);
            end
        end else begin
            $display("  TIMEOUT at cycle %0d, state=%0d", cycle, dut.state);
        end

        $finish;
    end

endmodule
