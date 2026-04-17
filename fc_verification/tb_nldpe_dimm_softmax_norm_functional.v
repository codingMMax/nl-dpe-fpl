// NL-DPE DIMM softmax_norm Functional Test
//
// Config: Proposed NL-DPE (R=1024, C=128, dpe_bw=40)
// Workload: N=128 exp-values → log-domain normalized values
//
// Pipeline: input SRAM → DPE(I|log) → CLB subtract log_sum → output SRAM
//
// Test input:
//   input[0] = 5, input[1] = 1, input[2] = 1, input[3] = 1, input[4..127] = 0
//   sum  = 5 + 1 + 1 + 1 + 0×124 = 8
//   log_sum = sum - 1 = 7  (log approx: log(x) ≈ x - 1)
//
// Expected output (log(x) - log_sum):
//   output[0] = log(5) - log_sum = 4 - 7 = -3  (int8: 253 = 0xFD)
//   output[1..3] = log(1) - log_sum = 0 - 7 = -7  (int8: 249 = 0xF9)
//   output[4..127] = log(0) - log_sum = -1 - 7 = -8  (int8: 248 = 0xF8)

`timescale 1ns / 1ps

module tb_nldpe_dimm_softmax_norm_functional;

    parameter DW = 40;
    parameter N = 128;
    parameter C = 128;
    parameter EPW = DW / 8;
    parameter PACKED_N = (N + EPW - 1) / EPW;  // 26

    reg clk, rst, valid_in, ready_n;
    reg  [DW-1:0] data_in;
    wire [DW-1:0] data_out;
    wire ready_in, valid_n;

    // DPE parameters
    defparam dut.norm_dpe.KERNEL_WIDTH = C;
    defparam dut.norm_dpe.NUM_COLS = C;
    defparam dut.norm_dpe.ACAM_MODE = 2;       // log
    defparam dut.norm_dpe.DPE_BUF_WIDTH = DW;
    defparam dut.norm_dpe.COMPUTE_CYCLES = 3;

    nldpe_dimm_softmax_norm #(.DATA_WIDTH(DW)) dut (
        .clk(clk), .rst(rst),
        .valid_in(valid_in), .ready_n(ready_n),
        .data_in(data_in),
        .data_out(data_out),
        .ready_in(ready_in), .valid_n(valid_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle, i, j, b;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    // Pre-load identity weights: 128×128 diagonal
    initial begin
        for (i = 0; i < C; i = i + 1)
            for (j = 0; j < C; j = j + 1)
                dut.norm_dpe.weights[i][j] = (i == j) ? 1 : 0;
    end

    reg [7:0] expected [0:N-1];

    initial begin
        rst = 1; valid_in = 0; ready_n = 1;
        data_in = 0;
        #20; rst = 0; #15;

        $display("=== NL-DPE DIMM softmax_norm Functional Test ===");
        $display("  Config: C=%0d, N=%0d, dpe_buf_width=%0d", C, N, DW);
        $display("  DPE: KERNEL_WIDTH=%0d, ACAM_MODE=2 (log), COMPUTE_CYCLES=3", C);
        $display("");

        // Build expected output table (log approx: log(x) = x - 1)
        for (i = 0; i < N; i = i + 1) begin
            case (i)
                0: expected[i] = 8'd5 - 1 - 8'd7;       // log(5) - log_sum = 4 - 7 = -3
                1, 2, 3: expected[i] = 8'd1 - 1 - 8'd7; // log(1) - log_sum = 0 - 7 = -7
                default: expected[i] = 8'd0 - 1 - 8'd7; // log(0) - log_sum = -1 - 7 = -8
            endcase
        end

        // Pre-load input SRAM: input[0]=5, input[1..3]=1, rest=0.
        #1;
        for (i = 0; i < PACKED_N; i = i + 1) dut.in_sram.mem[i] = 0;
        // Byte 0 of word 0 = 5
        dut.in_sram.mem[0][0*8 +: 8] = 8'd5;
        // Bytes 1,2,3 of word 0 = 1
        dut.in_sram.mem[0][1*8 +: 8] = 8'd1;
        dut.in_sram.mem[0][2*8 +: 8] = 8'd1;
        dut.in_sram.mem[0][3*8 +: 8] = 8'd1;
        // rest = 0

        // Set FSM to S_COMPUTE (skip S_LOAD)
        dut.in_w_addr = PACKED_N;
        dut.state = 2;    // S_COMPUTE
        dut.mac_count = 0;

        $display("  Input SRAM loaded (N=%0d values packed into %0d words)", N, PACKED_N);
        $display("    input[0]=5, input[1..3]=1, input[4..%0d]=0", N-1);
        $display("  Expected log_sum = 7, expected outputs: [-3, -7, -7, -7, -8, -8, ...]");
        $display("");

        // Wait for S_OUTPUT (state=5)
        while (dut.state != 5 && cycle < 10000) @(posedge clk);

        if (dut.state == 5) begin
            @(posedge clk); @(posedge clk); #1;
            $display("=== Results ===");
            $display("  output[0]   = %0d  (expect %0d = log(5)-log_sum)",
                $signed(dut.out_sram.mem[0][0*8 +: 8]), $signed(expected[0]));
            $display("  output[1]   = %0d  (expect %0d = log(1)-log_sum)",
                $signed(dut.out_sram.mem[0][1*8 +: 8]), $signed(expected[1]));
            $display("  output[4]   = %0d  (expect %0d = log(0)-log_sum)",
                $signed(dut.out_sram.mem[0][4*8 +: 8]), $signed(expected[4]));
            $display("  output[5]   = %0d  (expect %0d = log(0)-log_sum)",
                $signed(dut.out_sram.mem[1][0*8 +: 8]), $signed(expected[5]));
            $display("  Total cycles: %0d", cycle);
            $display("");

            begin : check_outputs
                integer err, oi, word_idx, byte_idx;
                reg [7:0] actual;
                err = 0;
                for (oi = 0; oi < N; oi = oi + 1) begin
                    word_idx = oi / EPW;
                    byte_idx = oi % EPW;
                    actual = dut.out_sram.mem[word_idx][byte_idx*8 +: 8];
                    if (actual !== expected[oi]) begin
                        if (err < 5) $display("  MISMATCH: output[%0d]=%0d (expect %0d) [word %0d byte %0d]",
                            oi, $signed(actual), $signed(expected[oi]), word_idx, byte_idx);
                        err = err + 1;
                    end
                end
                if (err == 0)
                    $display("  PASS: softmax_norm functional (%0d outputs correct)", N);
                else
                    $display("  FAIL: %0d output mismatches out of %0d", err, N);
            end
        end else begin
            $display("  TIMEOUT at cycle %0d, state=%0d", cycle, dut.state);
        end

        $finish;
    end

endmodule
