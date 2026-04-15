// Full DIMM pipeline verification: dimm_score_matrix with packed transfers.
//
// N=4, d=4, DATA_WIDTH=40, dual_identity=True.
// epw=5 (5 int8 per 40-bit word), packed_d=1, packed_Nd=4
//
// Q = [1,0,0,0] (one query row, stored in 1 packed word)
// K = identity matrix (4 key rows, stored in ceil(16/5)=4 packed words)
//
// S[0][j] = Σ_m exp(Q[m] + K[j][m])
// With exp(x) ≈ 1 + x + x²/2: exp(0)=1, exp(1)=2, exp(2)=5
// S[0][0] = exp(1+1) + exp(0+0) + exp(0+0) + exp(0+0) = 5+1+1+1 = 8
// S[0][1] = exp(1+0) + exp(0+1) + exp(0+0) + exp(0+0) = 2+2+1+1 = 6
// S[0][2] = exp(1+0) + exp(0+0) + exp(0+1) + exp(0+0) = 2+1+2+1 = 6
// S[0][3] = exp(1+0) + exp(0+0) + exp(0+0) + exp(0+1) = 2+1+1+2 = 6

`timescale 1ns / 1ps

module tb_dimm_score;

    parameter DW = 40;
    parameter N = 4;
    parameter D = 4;
    parameter EPW = DW / 8;  // 5

    reg clk, rst;
    reg valid_q, valid_k, ready_n;
    reg [DW-1:0] data_in_q, data_in_k;
    wire [DW-1:0] data_out;
    wire ready_q, ready_k, valid_n;

    // Override DPE inside dimm_score_matrix
    defparam dut.dimm_exp.dpe_inst.DPE_BUF_WIDTH = DW;
    defparam dut.dimm_exp.dpe_inst.COMPUTE_CYCLES = 3;
    defparam dut.dimm_exp.dpe_inst.ACAM_MODE = 1;  // exp
    defparam dut.dimm_exp.dpe_inst.KERNEL_WIDTH = 8;  // dual: 2*d=8
    defparam dut.dimm_exp.dpe_inst.NUM_COLS = 128;

    dimm_score_matrix #(.DATA_WIDTH(DW)) dut (
        .clk(clk), .rst(rst),
        .valid_q(valid_q), .valid_k(valid_k), .ready_n(ready_n),
        .data_in_q(data_in_q), .data_in_k(data_in_k),
        .data_out(data_out), .ready_q(ready_q), .ready_k(ready_k), .valid_n(valid_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    integer i, j, m;
    integer T_compute_start, T_output_start;

    // K matrix (identity) as flat array
    reg [7:0] K_flat [0:N*D-1];

    initial begin
        // Initialize K as identity
        for (i = 0; i < N*D; i = i + 1) K_flat[i] = 0;
        for (i = 0; i < N; i = i + 1) K_flat[i*D + i] = 1;

        rst = 1; valid_q = 0; valid_k = 0; ready_n = 1;
        data_in_q = 0; data_in_k = 0;
        #20; rst = 0; #10;

        $display("=== DIMM Score Matrix: N=%0d, d=%0d, epw=%0d ===", N, D, EPW);

        // ── Load Q: 1 packed word [Q0=1, Q1=0, Q2=0, Q3=0, pad=0] ──
        @(posedge clk);
        valid_q = 1;
        data_in_q = 0;
        data_in_q[0*8 +: 8] = 1;  // Q[0]=1
        // Q[1..3] = 0 (already zero)
        @(posedge clk);
        valid_q = 0;
        $display("T=%0t (cycle %0d): Q loaded (1 packed word)", $time, cycle);

        // Wait for S_LOAD_K
        wait(dut.state == 2);

        // ── Load K: 4 packed words (16 int8 → ceil(16/5)=4 words) ──
        // K flat: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        // Word 0: K[0..4] = [1,0,0,0, 0]
        // Word 1: K[5..9] = [1,0,0, 0,0]  → actually K_flat[5]=1 (K[1][1])
        // Word 2: K[10..14] = [1,0, 0,0,0] → K_flat[10]=1 (K[2][2])
        // Word 3: K[15] = [1, 0,0,0,0] → K_flat[15]=1 (K[3][3])
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
            valid_k = 1;
            data_in_k = 0;
            for (j = 0; j < EPW; j = j + 1) begin
                m = i * EPW + j;
                if (m < N * D)
                    data_in_k[j*8 +: 8] = K_flat[m];
            end
        end
        @(posedge clk);
        valid_k = 0;
        $display("T=%0t (cycle %0d): K loaded (4 packed words)", $time, cycle);

        // Wait for compute
        T_compute_start = cycle;
        wait(dut.state == 4 || dut.state == 5);  // S_OUTPUT or S_WRITE_B
        T_output_start = cycle;
        $display("T=%0t (cycle %0d): Compute done, output starting", $time, cycle);
        $display("  Compute cycles: %0d", T_output_start - T_compute_start);

        // Wait for S_OUTPUT
        wait(dut.state == 4);
        $display("T=%0t (cycle %0d): S_OUTPUT state", $time, cycle);

        // Read score outputs
        // The score SRAM has packed_N = ceil(4/5) = 1 word for all 4 scores
        // But the output FSM reads score_read_addr++ when ready_n=1
        // Each output word has up to 5 scores packed
        @(posedge clk); #1;
        if (valid_n) begin
            $display("  Score output word: %h", data_out);
            for (j = 0; j < N; j = j + 1) begin
                $display("    S[0][%0d] = %0d", j, $signed(data_out[j*8 +: 8]));
            end
        end

        // Expected: S[0][0]=8, S[0][1]=6, S[0][2]=6, S[0][3]=6
        $display("");
        $display("Expected: S[0][0]=8, S[0][1]=6, S[0][2]=6, S[0][3]=6");

        #100;
        $finish;
    end

endmodule
