// DEPRECATED (superseded by W=16 full DIMM top verification).
// This file was part of the Phase A-E per-stage W=1 DIMM exploration.
// Authoritative DIMM verification: fc_verification/rtl/nldpe_dimm_top_d64_c128.v
// and fc_verification/rtl/azurelily_dimm_top_d64_c128.v; see VERIFICATION.md.
//
// NL-DPE DIMM mac_sv Latency Alignment
//
// Config: Proposed NL-DPE (R=1024, C=128, dpe_bw=40)
// Workload: one row of attn (N=128) × V matrix (N=128 × d=64) → output d values
//
// Simulator (gemm_log M=N, K=N, N=d): per-pass cycles for mac_sv.
//   K_id = max(1, C//K) = 1  (no K_id packing since K=N=C=128)
//   dpe_passes_per_elem = ceil(K/C) = 1
//   read_cycles    = ceil(128*8/40) = 26
//   feed_cycles    = 26 + 2*1 = 28
//   compute_cycles = 4
//   output_cycles  = 26
//   reduce_cycles  = ceil(log2(128)) = 7
//   effective_out  = max(26, 7) = 26
//   per_pass       = 28 + 4 + 26 = 58 cycles
//
// RTL measures cycles from S_COMPUTE start to S_OUTPUT state.

`timescale 1ns / 1ps

module tb_nldpe_dimm_mac_sv_latency;

    parameter DW = 40;
    parameter N = 128;
    parameter D = 64;
    parameter C = 128;
    parameter EPW = DW / 8;
    parameter PACKED_N = (N + EPW - 1) / EPW;
    parameter PACKED_D = (D + EPW - 1) / EPW;

    reg clk, rst, valid_in, ready_n;
    reg  [DW-1:0] data_in;
    wire [DW-1:0] data_out;
    wire ready_in, valid_n;

    defparam dut.sv_dpe.KERNEL_WIDTH = N;
    defparam dut.sv_dpe.NUM_COLS = C;
    defparam dut.sv_dpe.ACAM_MODE = 0;
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

    initial begin
        for (i = 0; i < N; i = i + 1)
            for (j = 0; j < C; j = j + 1)
                dut.sv_dpe.weights[i][j] = (i < D && i == j) ? 1 : 0;
    end

    wire probe_wbuf = dut.sv_dpe_w_buf_en;
    wire probe_reg_full = dut.sv_dpe_reg_full;
    wire probe_dpe_done = dut.sv_dpe_dpe_done;

    integer wbuf_count, done_count;
    integer T_compute_start, T_first_wbuf, T_last_wbuf;
    integer T_reg_full, T_first_dpe_done, T_last_dpe_done, T_output_state;

    always @(posedge clk) begin
        if (!rst) begin
            if (probe_wbuf) begin
                wbuf_count = wbuf_count + 1;
                if (T_first_wbuf < 0) T_first_wbuf = cycle;
                T_last_wbuf = cycle;
            end
            if (probe_dpe_done) begin
                done_count = done_count + 1;
                if (T_first_dpe_done < 0) T_first_dpe_done = cycle;
                T_last_dpe_done = cycle;
            end
            if (probe_reg_full && T_reg_full < 0) T_reg_full = cycle;
        end
    end

    initial begin
        rst = 1; valid_in = 0; ready_n = 1; data_in = 0;
        wbuf_count = 0; done_count = 0;
        T_compute_start = -1; T_first_wbuf = -1; T_last_wbuf = -1;
        T_reg_full = -1; T_first_dpe_done = -1; T_last_dpe_done = -1;
        T_output_state = -1;
        #20; rst = 0; #15;

        #1;
        for (i = 0; i < PACKED_N; i = i + 1) dut.in_sram.mem[i] = {5{8'd1}};
        dut.in_w_addr = PACKED_N;
        dut.state = 2;       // S_COMPUTE
        dut.mac_count = 0;
        T_compute_start = cycle;

        while (dut.state != 4 && cycle < 10000) @(posedge clk);
        T_output_state = cycle;

        $display("=== NL-DPE DIMM mac_sv Latency Alignment ===");
        $display("  Config: N=%0d, d=%0d, C=%0d", N, D, C);
        $display("  DPE: KERNEL_WIDTH=%0d, ACAM_MODE=0 (plain VMM), COMPUTE=3", N);
        $display("");
        $display("RTL Measured (per DPE pass = 1 attn row × V → d outputs):");
        $display("  T_compute_start : cycle %0d", T_compute_start);
        $display("  T_first_wbuf    : cycle %0d", T_first_wbuf);
        $display("  T_last_wbuf     : cycle %0d", T_last_wbuf);
        $display("  T_reg_full      : cycle %0d", T_reg_full);
        $display("  T_first_dpe_done: cycle %0d", T_first_dpe_done);
        $display("  T_last_dpe_done : cycle %0d", T_last_dpe_done);
        $display("  T_output_state  : cycle %0d", T_output_state);
        $display("");
        $display("  w_buf_en strobes : %0d (expect %0d)", wbuf_count, PACKED_N);
        $display("  dpe_done pulses  : %0d (expect ~26 = ceil(C×8/DW))", done_count);
        $display("");
        $display("  Per-pass (compute_start → output_state): %0d cycles",
            T_output_state - T_compute_start);
        $display("");
        $display("IMC Simulator Reference (gemm_log M=%0d, K=%0d, N=%0d):", N, N, D);
        $display("  K_id            = max(1, C//K) = %0d", (C > N) ? C/N : 1);
        $display("  dpe_passes/elem = ceil(K/C) = 1");
        $display("  feed_cycles     = 26 + 2×1 = 28");
        $display("  compute_cycles  = 4");
        $display("  output_cycles   = 26");
        $display("  reduce_cycles   = ceil(log2(%0d)) = %0d", N, $clog2(N));
        $display("  effective_out   = max(26, %0d) = 26", $clog2(N));
        $display("  per_pass        = 28 + 4 + 26 = 58 cycles");
        $display("");
        $display("Delta: RTL %0d - Sim 58 = %0d cycles (FSM overhead)",
            T_output_state - T_compute_start,
            (T_output_state - T_compute_start) - 58);

        $finish;
    end

endmodule
