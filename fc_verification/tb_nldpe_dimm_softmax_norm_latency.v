// NL-DPE DIMM softmax_norm Latency Alignment
//
// Config: Proposed NL-DPE (R=1024, C=128, dpe_bw=40)
// Workload: N=128 single-identity DPE(I|log) pass + CLB divide
//
// Simulator (dimm_nonlinear(N=128, op='log')):
//   read_cycles    = ceil(128*8/40) = 26
//   feed_cycles    = read + sram_lat×1 = 26 + 2 = 28
//   compute_cycles = ceil((core_ns - output*t_clk)/t_clk) = 4
//   output_cycles  = ceil(128*8/40) = 26
//   reduce_cycles  = 1 (trivial for nonlinear)
//   effective_out  = max(output, reduce) = 26
//   per_pass       = 28 + 4 + 26 = 58 cycles
//
// RTL cycle count (measured) + per-stage probes: w_buf_en strobes,
// reg_full timing, dpe_done pulses, output capture.

`timescale 1ns / 1ps

module tb_nldpe_dimm_softmax_norm_latency;

    parameter DW = 40;
    parameter N = 128;
    parameter C = 128;
    parameter EPW = DW / 8;
    parameter PACKED_N = (N + EPW - 1) / EPW;

    reg clk, rst, valid_in, ready_n;
    reg  [DW-1:0] data_in;
    wire [DW-1:0] data_out;
    wire ready_in, valid_n;

    defparam dut.norm_dpe.KERNEL_WIDTH = C;
    defparam dut.norm_dpe.NUM_COLS = C;
    defparam dut.norm_dpe.ACAM_MODE = 2;
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

    integer cycle, i, j;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    // Identity weights
    initial begin
        for (i = 0; i < C; i = i + 1)
            for (j = 0; j < C; j = j + 1)
                dut.norm_dpe.weights[i][j] = (i == j) ? 1 : 0;
    end

    // Probes
    wire probe_wbuf = dut.norm_dpe_w_buf_en;
    wire probe_reg_full = dut.norm_dpe_reg_full;
    wire probe_dpe_done = dut.norm_dpe_dpe_done;
    wire probe_nl_ctrl_hi = (dut.norm_dpe_nl_dpe_control == 2'b11);

    integer wbuf_count, done_count;
    integer T_compute_start, T_first_wbuf, T_last_wbuf;
    integer T_reg_full, T_first_dpe_done, T_last_dpe_done;
    integer T_output_state;

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
        // Pre-load input SRAM (values don't matter for latency)
        for (i = 0; i < PACKED_N; i = i + 1) dut.in_sram.mem[i] = {5{8'd1}};
        dut.in_w_addr = PACKED_N;
        dut.state = 2;       // S_COMPUTE
        dut.mac_count = 0;
        T_compute_start = cycle;

        // Wait for S_OUTPUT (state=5)
        while (dut.state != 5 && cycle < 10000) @(posedge clk);
        T_output_state = cycle;

        $display("=== NL-DPE DIMM softmax_norm Latency Alignment ===");
        $display("  Config: C=%0d, N=%0d, dpe_buf_width=%0d", C, N, DW);
        $display("  DPE: KERNEL_WIDTH=%0d, ACAM_MODE=2 (log), COMPUTE_CYCLES=3", C);
        $display("");
        $display("RTL Measured (per DPE pass):");
        $display("  T_compute_start : cycle %0d", T_compute_start);
        $display("  T_first_wbuf    : cycle %0d (feed begins)", T_first_wbuf);
        $display("  T_last_wbuf     : cycle %0d", T_last_wbuf);
        $display("  T_reg_full      : cycle %0d (DPE buffer full)", T_reg_full);
        $display("  T_first_dpe_done: cycle %0d (DPE output begins)", T_first_dpe_done);
        $display("  T_last_dpe_done : cycle %0d", T_last_dpe_done);
        $display("  T_output_state  : cycle %0d (FSM S_OUTPUT)", T_output_state);
        $display("");
        $display("  w_buf_en strobes : %0d (expect %0d)", wbuf_count, PACKED_N);
        $display("  dpe_done pulses  : %0d (expect %0d)", done_count, PACKED_N);
        $display("");
        $display("  Per-pass total (first_wbuf → output_state):  %0d cycles",
            T_output_state - T_first_wbuf);
        $display("  Per-pass total (compute_start → output_state): %0d cycles",
            T_output_state - T_compute_start);
        $display("");
        $display("IMC Simulator Reference (dimm_nonlinear N=128, op='log'):");
        $display("  read_cycles    = %0d", PACKED_N);
        $display("  sram_read_lat  = 2");
        $display("  feed_cycles    = %0d + 2 = %0d", PACKED_N, PACKED_N + 2);
        $display("  compute_cycles = 4 (CDC: ceil(10ns / 2.93ns))");
        $display("  output_cycles  = %0d", PACKED_N);
        $display("  effective_out  = max(output, reduce=1) = %0d", PACKED_N);
        $display("  per_pass       = feed + compute + out = %0d + 4 + %0d = %0d cycles",
            PACKED_N + 2, PACKED_N, PACKED_N + 2 + 4 + PACKED_N);
        $display("");
        $display("Delta (RTL - Simulator):");
        $display("  %0d - %0d = %0d cycles (FSM overhead: drain + state transitions)",
            T_output_state - T_compute_start, PACKED_N + 2 + 4 + PACKED_N,
            (T_output_state - T_compute_start) - (PACKED_N + 2 + 4 + PACKED_N));

        $finish;
    end

endmodule
