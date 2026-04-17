// Azure-Lily DIMM mac_sv Latency Alignment
//
// Simulator (gemm_dsp M=1, K=N=128, N=d=64 — per query):
//   DSP_WDITH  = 4
//   k_tile     = ceil(K/4) = ceil(128/4) = 32
//   dsp_batch  = 1  (M < N)
//   wave       = d = 64
//   per_col    = k_tile × dsp_batch + k_tile = 64
//   total      = 64 × 64 = 4096 cycles per query
//
// RTL: dsp_mac with K=PACKED_N=26 per output column, × D=64 columns.

`timescale 1ns / 1ps

module tb_azurelily_dimm_mac_sv_latency;

    parameter DW = 40;
    parameter N = 128;
    parameter D = 64;
    parameter EPW = DW / 8;
    parameter PACKED_N = (N + EPW - 1) / EPW;

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

    wire dsp_valid = dut.mac_inst.valid;
    wire dsp_valid_n = dut.mac_inst.valid_n;

    integer dsp_valid_count, dsp_done_count, out_count;
    integer T_compute_start, T_first_out, T_last_out;

    always @(posedge clk) begin
        if (!rst) begin
            if (dsp_valid) dsp_valid_count = dsp_valid_count + 1;
            if (dsp_valid_n) begin
                dsp_done_count = dsp_done_count + 1;
                if (T_first_out < 0) T_first_out = cycle;
                T_last_out = cycle;
            end
            if (dut.o_w_en) out_count = out_count + 1;
        end
    end

    initial begin
        rst = 1; valid_s = 0; valid_v = 0; ready_n = 1;
        data_in_s = 0; data_in_v = 0;
        dsp_valid_count = 0; dsp_done_count = 0; out_count = 0;
        T_compute_start = -1; T_first_out = -1; T_last_out = -1;
        #20; rst = 0; #15;

        #1;
        // Fill S and V with ones for activity
        for (i = 0; i < PACKED_N; i = i + 1) dut.s_sram.mem[i] = {5{8'd1}};
        for (j = 0; j < D; j = j + 1)
            for (k = 0; k < PACKED_N; k = k + 1)
                dut.v_sram.mem[j * PACKED_N + k] = {5{8'd1}};

        dut.s_w_addr = PACKED_N;
        dut.v_w_addr = D * PACKED_N;
        dut.state = 3;
        dut.m_count = 0;
        dut.k_count = 0;
        dut.mac_count = 0;
        T_compute_start = cycle;

        while (dut.state != 4 && cycle < 100000) @(posedge clk);

        $display("=== Azure-Lily DIMM mac_sv Latency Alignment ===");
        $display("  Config: N=%0d, d=%0d", N, D);
        $display("  Primitive: dsp_mac(K=%0d)", PACKED_N);
        $display("");
        $display("RTL Measured:");
        $display("  T_compute_start : cycle %0d", T_compute_start);
        $display("  T_first_out     : cycle %0d", T_first_out);
        $display("  T_last_out      : cycle %0d", T_last_out);
        $display("  Total outputs   : %0d (expect %0d)", out_count, D);
        $display("  Total cycles    : %0d", T_last_out - T_compute_start);
        $display("  Per-column cyc  : %.1f", (T_last_out - T_first_out) * 1.0 / (D - 1));
        $display("  dsp_valid pulses: %0d (expect %0d)", dsp_valid_count, D * PACKED_N);
        $display("");
        $display("IMC Simulator Reference (gemm_dsp M=1, K=%0d, N=%0d):", N, D);
        $display("  DSP_WDITH       = 4");
        $display("  k_tile          = ceil(%0d/4) = 32", N);
        $display("  per_col         = 64 cycles");
        $display("  total           = 64 × %0d = %0d", D, 64 * D);
        $display("");
        $display("Delta: RTL per-col %.1f vs Sim per-col 64",
            (T_last_out - T_first_out) * 1.0 / (D - 1));
        $display("Note: Same structural mismatch as mac_qk — sim assumes DSP_WDITH=4,");
        $display("      RTL uses 5/cycle via int_sop_4 + CLB multiply.");

        $finish;
    end

endmodule
