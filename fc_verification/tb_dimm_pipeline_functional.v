// DIMM Pipeline Functional Test — Dual-Identity (K_id=2)
//
// Config: Proposed NL-DPE (R=1024, C=128, dpe_bw=40)
// BERT-Tiny: d_head=64, N=4 (small for fast sim)
// K_id = C//d = 128//64 = 2 (dual-identity, full column utilization)
//
// Test: Q = one-hot [1,0,...,0] (d=64 elements)
//       K = identity matrix (N×d = 4×64)
//       S[i][j] = Σ_m exp(Q[m] + K[j][m])
//
// Expected (exp(x) ≈ 1 + x + x²/2):
//   exp(0)=1, exp(1)=2, exp(2)=5
//   S[0] = exp(Q[0]+K[0][0]) + Σ_{m=1..63} exp(Q[m]+K[0][m])
//        = exp(1+1) + 63×exp(0+0) = 5 + 63 = 68
//   S[1] = exp(Q[0]+K[1][0]) + exp(Q[1]+K[1][1]) + 62×exp(0)
//        = exp(1+0) + exp(0+1) + 62×1 = 2 + 2 + 62 = 66
//   S[2] = 2 + 2 + 62 = 66
//   S[3] = 2 + 2 + 62 = 66

`timescale 1ns / 1ps

module tb_dimm_pipeline_functional;

    parameter DW = 40;
    parameter N = 128;
    parameter D = 64;
    parameter C = 128;
    parameter EPW = DW / 8;
    parameter PD = (D + EPW - 1) / EPW;  // 13

    reg clk, rst, valid_q, valid_k, ready_n;
    reg [DW-1:0] data_in_q, data_in_k;
    wire [DW-1:0] data_out;
    wire ready_q, ready_k, valid_n;

    // DPE parameters: KERNEL_WIDTH=128 (dual: 2×64), ACAM exp
    defparam dut.dimm_exp.KERNEL_WIDTH = C;  // = K_id × d = 128
    defparam dut.dimm_exp.NUM_COLS = C;
    defparam dut.dimm_exp.ACAM_MODE = 1;
    defparam dut.dimm_exp.DPE_BUF_WIDTH = DW;
    defparam dut.dimm_exp.COMPUTE_CYCLES = 3;

    dimm_score_matrix #(.DATA_WIDTH(DW)) dut (
        .clk(clk), .rst(rst),
        .valid_q(valid_q), .valid_k(valid_k), .ready_n(ready_n),
        .data_in_q(data_in_q), .data_in_k(data_in_k),
        .data_out(data_out), .ready_q(ready_q), .ready_k(ready_k), .valid_n(valid_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle, i, j, b;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    // Pre-load identity weights: 128×128 diagonal
    initial begin
        for (i = 0; i < C; i = i + 1)
            for (j = 0; j < C; j = j + 1)
                dut.dimm_exp.weights[i][j] = (i == j) ? 1 : 0;
    end

    initial begin
        rst = 1; valid_q = 0; valid_k = 0; ready_n = 1;
        data_in_q = 0; data_in_k = 0;
        #20; rst = 0; #15;

        $display("=== DIMM Pipeline Functional Test ===");
        $display("  Config: R=1024, C=%0d, d=%0d, N=%0d, K_id=2", C, D, N);
        $display("  DPE: KERNEL_WIDTH=%0d, DPE_BUF_WIDTH=%0d, ACAM=exp", C, DW);
        $display("");

        // Pre-load Q SRAM: Q = [1, 0, 0, ..., 0] (d=64 elements, PD=13 packed words)
        #1;
        for (i = 0; i < PD; i = i + 1) begin
            dut.q_sram.mem[i] = 0;
        end
        dut.q_sram.mem[0][0*8 +: 8] = 1;  // Q[0] = 1

        // Pre-load K SRAMs: K = identity (N×d, padded per-key)
        // K[j][m] = δ(j,m) for j=0..3, m=0..63
        // Each key: PD=13 words, padded to word boundary
        for (j = 0; j < N; j = j + 1) begin
            for (i = 0; i < PD; i = i + 1) begin
                dut.k_sram_a.mem[j * PD + i] = 0;
                dut.k_sram_b.mem[j * PD + i] = 0;
            end
            // Set identity diagonal
            if (j < D) begin
                dut.k_sram_a.mem[j * PD + j / EPW][(j % EPW) * 8 +: 8] = 1;
                dut.k_sram_b.mem[j * PD + j / EPW][(j % EPW) * 8 +: 8] = 1;
            end
        end

        // Set FSM to S_COMPUTE
        dut.q_write_addr = PD;
        dut.k_write_addr = N * PD;
        dut.state = 3;  // S_COMPUTE
        dut.score_idx = 0;
        dut.mac_count = 0;
        dut.acc_clear = 0;
        dut.feed_half = 0;
        dut.feed_phase = 0;

        $display("  Q SRAM loaded: %0d packed words (Q[0]=1, rest=0)", PD);
        $display("  K SRAMs loaded: %0d×%0d packed words (identity)", N, PD);
        $display("  Waiting for compute...");

        // Wait for S_OUTPUT (state=6)
        while (dut.state != 6 && cycle < 10000) @(posedge clk);

        if (dut.state == 6) begin
            @(posedge clk); @(posedge clk); #1;
            $display("");
            $display("=== Results ===");
            $display("  S[0]   = %0d (expect 68)", dut.score_sram.mem[0][7:0]);
            $display("  S[1]   = %0d (expect 66)", dut.score_sram.mem[1][7:0]);
            $display("  S[63]  = %0d (expect 66)", dut.score_sram.mem[63][7:0]);
            $display("  S[64]  = %0d (expect 65)", dut.score_sram.mem[64][7:0]);
            $display("  S[127] = %0d (expect 65)", dut.score_sram.mem[N-1][7:0]);
            $display("  Total compute: %0d cycles", cycle);
            $display("");

            // Check scores:
            // S[0] = 68 (exp(2) + 63×exp(0))
            // S[1..d-1] = 66 (2×exp(1) + 62×exp(0)) — K[j] has diagonal within d
            // S[d..N-1] = 65 (exp(1) + 63×exp(0)) — K[j] diagonal beyond d
            begin : check_scores
                integer err, si, expected;
                err = 0;
                for (si = 0; si < N; si = si + 1) begin
                    if (si == 0) expected = 68;
                    else if (si < D) expected = 66;
                    else expected = 65;
                    if (dut.score_sram.mem[si][7:0] != expected) begin
                        if (err < 5) $display("  MISMATCH: S[%0d]=%0d (expect %0d)", si,
                            dut.score_sram.mem[si][7:0], expected);
                        err = err + 1;
                    end
                end
                if (err == 0)
                    $display("  PASS: DIMM pipeline functional (K_id=2, d=%0d, C=%0d, N=%0d)", D, C, N);
                else
                    $display("  FAIL: %0d score mismatches", err);
            end
        end else begin
            $display("  TIMEOUT at cycle %0d, state=%0d", cycle, dut.state);
        end

        $finish;
    end

endmodule
