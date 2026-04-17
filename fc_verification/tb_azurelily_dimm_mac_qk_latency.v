// DEPRECATED (superseded by W=16 full DIMM top verification).
// This file was part of the Phase A-E per-stage W=1 DIMM exploration.
// Authoritative DIMM verification: fc_verification/rtl/nldpe_dimm_top_d64_c128.v
// and fc_verification/rtl/azurelily_dimm_top_d64_c128.v; see VERIFICATION.md.
//
// Azure-Lily DIMM mac_qk Latency Alignment
//
// Measures cycles to compute all N=128 QK^T scores using dsp_mac (K=13).
//
// Simulator (gemm_dsp M=128, K=64, N=128):
//   DSP_WDITH=4 (simulator assumes 4 int8 products per DSP cycle)
//   k_tile      = ceil(K / DSP_WDITH) = ceil(64/4) = 16
//   dsp_batch   = ceil(M / total_dsp) = ceil(1 / 132) = 1   (M < N)
//   wave        = N = 128
//   total_cycles= (k_tile × dsp_batch + k_tile) × wave = 32 × 128 = 4096
//   per_score   = 32 cycles
//
// RTL (dsp_mac K=PACKED_D=13):
//   dsp_mac processes 5 int8 pairs per cycle (4 via int_sop_4 + 1 via CLB)
//   13 cycles to accumulate d=64 elements → 1 valid_n pulse → capture score
//   Expected per-score ≈ 13 + pipeline drain
//
// Known discrepancy: simulator assumes 4 products/cycle; RTL uses 5 (via CLB).
// Simulator is pessimistic by ~5/4 = 1.25× on per-row compute.

`timescale 1ns / 1ps

module tb_azurelily_dimm_mac_qk_latency;

    parameter DW = 40;
    parameter N = 128;
    parameter D = 64;
    parameter EPW = DW / 8;
    parameter PACKED_D = (D + EPW - 1) / EPW;

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

    // Probes
    wire dsp_valid = dut.mac_inst.valid;
    wire dsp_valid_n = dut.mac_inst.valid_n;
    wire score_wen = dut.s_w_en;

    integer dsp_valid_count, dsp_done_count, score_count;
    integer T_compute_start, T_first_score, T_last_score;

    always @(posedge clk) begin
        if (!rst) begin
            if (dsp_valid) dsp_valid_count = dsp_valid_count + 1;
            if (dsp_valid_n) dsp_done_count = dsp_done_count + 1;
            if (score_wen) begin
                if (T_first_score < 0) T_first_score = cycle;
                T_last_score = cycle;
                score_count = score_count + 1;
            end
        end
    end

    initial begin
        rst = 1; valid_q = 0; valid_k = 0; ready_n = 1;
        data_in_q = 0; data_in_k = 0;
        dsp_valid_count = 0; dsp_done_count = 0; score_count = 0;
        T_compute_start = -1; T_first_score = -1; T_last_score = -1;
        #20; rst = 0; #15;

        #1;
        // Preload with ones so we see activity
        for (i = 0; i < PACKED_D; i = i + 1) dut.q_sram.mem[i] = {5{8'd1}};
        for (j = 0; j < N; j = j + 1) begin
            for (i = 0; i < PACKED_D; i = i + 1)
                dut.k_sram.mem[j * PACKED_D + i] = {5{8'd1}};
        end

        dut.q_w_addr = PACKED_D;
        dut.k_w_addr = N * PACKED_D;
        dut.state = 3;       // S_COMPUTE
        dut.j_count = 0;
        dut.k_count = 0;
        dut.mac_count = 0;
        T_compute_start = cycle;

        while (dut.state != 5 && cycle < 100000) @(posedge clk);

        $display("=== Azure-Lily DIMM mac_qk Latency Alignment ===");
        $display("  Config: N=%0d, d=%0d", N, D);
        $display("  Primitive: dsp_mac(K=%0d)", PACKED_D);
        $display("");
        $display("RTL Measured:");
        $display("  T_compute_start : cycle %0d", T_compute_start);
        $display("  T_first_score   : cycle %0d", T_first_score);
        $display("  T_last_score    : cycle %0d", T_last_score);
        $display("  Total scores    : %0d (expect %0d)", score_count, N);
        $display("  Total cycles    : %0d", T_last_score - T_compute_start);
        $display("  dsp_valid pulses: %0d (expect %0d×%0d = %0d)",
            dsp_valid_count, N, PACKED_D, N * PACKED_D);
        $display("  dsp_done pulses : %0d (expect %0d)", dsp_done_count, N);
        $display("");
        $display("  Per-score cycles: %.1f", (T_last_score - T_first_score) * 1.0 / (N - 1));
        $display("");
        $display("IMC Simulator Reference (gemm_dsp M=%0d, K=%0d, N=%0d):", N, D, N);
        $display("  DSP_WDITH       = 4");
        $display("  k_tile          = ceil(%0d/4) = 16", D);
        $display("  dsp_batch       = 1");
        $display("  wave            = %0d", N);
        $display("  per_score       = k_tile×dsp_batch + k_tile = 32 cycles");
        $display("  total_cycles    = 32 × %0d = %0d", N, 32 * N);
        $display("");
        $display("Delta:");
        $display("  Sim per-score 32 vs RTL per-score %.1f",
            (T_last_score - T_first_score) * 1.0 / (N - 1));
        $display("  Note: Sim assumes DSP_WDITH=4 (pure DSP), RTL uses 5/cycle");
        $display("        (4 via int_sop_4 + 1 via CLB multiply).");

        $finish;
    end

endmodule
