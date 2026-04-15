// DIMM Pipeline Latency Alignment — Dual-Identity (K_id=2)
//
// Config: Proposed NL-DPE (R=1024, C=128, dpe_bw=40)
// BERT-Tiny: d_head=64, N=4
//
// Measures per-score cycle count and compares with IMC simulator:
//   Simulator per DPE pass: read=26 + compute=3 + output=26 = 55 cycles
//   Simulator per row: 1(CLB_add) + 1×55(DPE) + 6(reduce) = 62 cycles
//   K_id=2: 2 scores per DPE pass → ~31 cycles/score amortized

`timescale 1ns / 1ps

module tb_dimm_pipeline_latency;

    parameter DW = 40;
    parameter N = 4;
    parameter D = 64;
    parameter C = 128;
    parameter EPW = DW / 8;
    parameter PD = (D + EPW - 1) / EPW;

    reg clk, rst, valid_q, valid_k, ready_n;
    reg [DW-1:0] data_in_q, data_in_k;
    wire [DW-1:0] data_out;
    wire ready_q, ready_k, valid_n;

    defparam dut.dimm_exp.KERNEL_WIDTH = C;
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

    integer cycle, i, j;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    // Identity weights
    initial begin
        for (i = 0; i < C; i = i + 1)
            for (j = 0; j < C; j = j + 1)
                dut.dimm_exp.weights[i][j] = (i == j) ? 1 : 0;
    end

    // Timestamps
    integer T_start, T_score_write[0:N-1];
    integer score_writes;
    wire probe_score_w_en = dut.score_w_en;
    wire probe_dpe_wbuf = dut.dimm_exp.w_buf_en;
    wire probe_dpe_done = dut.dimm_exp.dpe_done;
    wire probe_reg_full = dut.dimm_exp.reg_full;

    integer wbuf_count, done_count;

    always @(posedge clk) begin
        if (!rst) begin
            if (probe_dpe_wbuf) wbuf_count = wbuf_count + 1;
            if (probe_dpe_done) done_count = done_count + 1;
        end
    end

    initial begin
        rst = 1; valid_q = 0; valid_k = 0; ready_n = 1;
        data_in_q = 0; data_in_k = 0;
        wbuf_count = 0; done_count = 0;
        score_writes = 0;
        #20; rst = 0; #15;

        // Pre-load Q and K SRAMs (same as functional test)
        #1;
        for (i = 0; i < PD; i = i + 1) dut.q_sram.mem[i] = 0;
        dut.q_sram.mem[0][0*8 +: 8] = 1;

        for (j = 0; j < N; j = j + 1) begin
            for (i = 0; i < PD; i = i + 1) begin
                dut.k_sram_a.mem[j * PD + i] = 0;
                dut.k_sram_b.mem[j * PD + i] = 0;
            end
            if (j < D) begin
                dut.k_sram_a.mem[j * PD + j / EPW][(j % EPW) * 8 +: 8] = 1;
                dut.k_sram_b.mem[j * PD + j / EPW][(j % EPW) * 8 +: 8] = 1;
            end
        end

        dut.q_write_addr = PD; dut.k_write_addr = N * PD;
        dut.state = 3; dut.score_idx = 0; dut.mac_count = 0;
        dut.acc_clear = 0; dut.feed_half = 0; dut.feed_phase = 0;

        T_start = cycle;

        // Track score writes
        while (dut.state != 6 && cycle < 2000) begin
            @(posedge clk); #1;
            if (probe_score_w_en && score_writes < N) begin
                T_score_write[score_writes] = cycle;
                score_writes = score_writes + 1;
            end
        end

        $display("=== DIMM Pipeline Latency Alignment ===");
        $display("  Config: R=1024, C=%0d, d=%0d, N=%0d, K_id=2", C, D, N);
        $display("  DPE: KERNEL_WIDTH=%0d, DPE_BUF_WIDTH=%0d, COMPUTE=3", C, DW);
        $display("");
        $display("Per-Score Cycle Counts:");
        for (i = 0; i < score_writes; i = i + 1) begin
            if (i == 0)
                $display("  Score[%0d] at cycle %0d (from start: %0d cycles)",
                    i, T_score_write[i], T_score_write[i] - T_start);
            else
                $display("  Score[%0d] at cycle %0d (delta: %0d cycles)",
                    i, T_score_write[i], T_score_write[i] - T_score_write[i-1]);
        end
        $display("");
        $display("DPE Statistics:");
        $display("  w_buf_en pulses: %0d (expect %0d per pair × %0d pairs = %0d)",
            wbuf_count, C / EPW, N / 2, (C / EPW) * (N / 2));
        $display("  dpe_done pulses: %0d (expect %0d per pass × %0d passes = %0d)",
            done_count, C / EPW, N / 2, (C / EPW) * (N / 2));
        $display("");
        $display("Total: %0d cycles for %0d scores = %.1f cycles/score",
            cycle - T_start, N, (cycle - T_start) * 1.0 / N);
        $display("");
        $display("IMC Simulator Reference:");
        $display("  Per DPE pass: read=%0d + compute=3 + output=%0d = %0d cycles",
            C * 8 / DW, C * 8 / DW, C * 8 / DW + 3 + C * 8 / DW);
        $display("  Per row (gemm_log): 1(add) + 1×%0d(DPE) + %0d(reduce) = %0d cycles",
            C * 8 / DW + 3 + C * 8 / DW, 6, 1 + C * 8 / DW + 3 + C * 8 / DW + 6);
        $display("  K_id=2: %0d cycles per score pair → %.1f cycles/score amortized",
            1 + C * 8 / DW + 3 + C * 8 / DW + 6,
            (1 + C * 8 / DW + 3 + C * 8 / DW + 6) / 2.0);
        $display("");

        // Verify scores
        $display("Score Values:");
        $display("  S[0]=%0d S[1]=%0d S[2]=%0d S[3]=%0d",
            dut.score_sram.mem[0][7:0], dut.score_sram.mem[1][7:0],
            dut.score_sram.mem[2][7:0], dut.score_sram.mem[3][7:0]);
        if (dut.score_sram.mem[0][7:0]==68 && dut.score_sram.mem[1][7:0]==66 &&
            dut.score_sram.mem[2][7:0]==66 && dut.score_sram.mem[3][7:0]==66)
            $display("  Functional: PASS");
        else
            $display("  Functional: FAIL");

        $finish;
    end

endmodule
