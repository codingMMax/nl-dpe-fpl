// Instrumented testbench: measure per-stage cycle counts for RTL ↔ simulator alignment.
//
// Pre-loads the SRAM with packed data (bypassing input_feed), then measures
// the stages the simulator models: SRAM→DPE read, compute, output serialize.
//
// Usage:
//   iverilog -DK_INPUT=512 -DN_OUTPUT=128 -DDPE_BUF_WIDTH=16 -DDPE_COMPUTE_CYCLES=44 \
//     -o tb tb_alignment.v dpe_stub.v fc_*.v
//   vvp tb

`timescale 1ns / 1ps

`ifndef K_INPUT
`define K_INPUT 512
`endif
`ifndef N_OUTPUT
`define N_OUTPUT 128
`endif
`ifndef DPE_BUF_WIDTH
`define DPE_BUF_WIDTH 16
`endif
`ifndef DPE_COMPUTE_CYCLES
`define DPE_COMPUTE_CYCLES 44
`endif

module tb_alignment;

    parameter DATA_WIDTH = `DPE_BUF_WIDTH;
    parameter K = `K_INPUT;
    parameter N = `N_OUTPUT;
    parameter ELEMS_PER_WORD = DATA_WIDTH / 8;
    parameter PACKED_K = (K + ELEMS_PER_WORD - 1) / ELEMS_PER_WORD;

    reg clk, rst, valid, ready_n;
    reg [DATA_WIDTH-1:0] data_in;
    wire [DATA_WIDTH-1:0] data_out;
    wire ready, valid_n;

    // Override DPE parameters
    defparam dut.fc_layer_inst.dpe_inst.KERNEL_WIDTH = K;
    defparam dut.fc_layer_inst.dpe_inst.NUM_COLS = N;
    defparam dut.fc_layer_inst.dpe_inst.DPE_BUF_WIDTH = `DPE_BUF_WIDTH;
    defparam dut.fc_layer_inst.dpe_inst.COMPUTE_CYCLES = `DPE_COMPUTE_CYCLES;

    // DUT
    fc_top #(.DATA_WIDTH(DATA_WIDTH)) dut (
        .clk(clk),
        .rst(rst),
        .valid(valid),
        .ready_n(ready_n),
        .data_in(data_in),
        .data_out(data_out),
        .ready(ready),
        .valid_n(valid_n)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Cycle counter ───────────────────────────────────────────────
    integer cycle;
    always @(posedge clk) begin
        if (rst) cycle <= 0;
        else cycle <= cycle + 1;
    end

    // Stage timestamps
    integer T_wbuf_first = -1;  // first w_buf_en (SRAM→DPE starts)
    integer T_regfull    = -1;  // reg_full (DPE buffer full)
    integer T_done_first = -1;  // first dpe_done (output starts)
    integer T_done_last  = -1;  // last dpe_done (output ends)
    integer T_validn     = -1;  // first valid_n at top

    integer wbuf_count = 0;
    integer done_count = 0;

    // Probe internal signals
    wire probe_wbuf_en  = dut.fc_layer_inst.dpe_inst.w_buf_en;
    wire probe_reg_full = dut.fc_layer_inst.dpe_inst.reg_full;
    wire probe_dpe_done = dut.fc_layer_inst.dpe_inst.dpe_done;

    always @(posedge clk) begin
        if (!rst) begin
            if (probe_wbuf_en && T_wbuf_first < 0) T_wbuf_first = cycle;
            if (probe_wbuf_en) wbuf_count = wbuf_count + 1;
            if (probe_reg_full && T_regfull < 0) T_regfull = cycle;
            if (probe_dpe_done && T_done_first < 0) T_done_first = cycle;
            if (probe_dpe_done) begin done_count = done_count + 1; T_done_last = cycle; end
            if (valid_n && T_validn < 0) T_validn = cycle;
        end
    end

    // ── Main stimulus ───────────────────────────────────────────────
    integer i, j, elem_idx;
    integer output_count;

    initial begin
        $dumpfile("tb_alignment.vcd");
        $dumpvars(0, tb_alignment);

        // Reset
        rst = 1; valid = 0; ready_n = 1; data_in = 0;
        #20;
        rst = 0;
        #10;

        $display("=== Alignment TB (pre-load): K=%0d, N=%0d, DPE_BUF_WIDTH=%0d ===", K, N, DATA_WIDTH);
        $display("    elems_per_word=%0d, packed_words=%0d", ELEMS_PER_WORD, PACKED_K);

        // ── Phase 1: Pre-load SRAM with packed data ─────────────────
        // Write directly to the SRAM inside conv_layer_single_dpe.
        // Each SRAM word holds ELEMS_PER_WORD int8 values.
        $display("Pre-loading SRAM with %0d packed words...", PACKED_K);
        elem_idx = 0;
        for (i = 0; i < PACKED_K; i = i + 1) begin
            // Pack ELEMS_PER_WORD int8s into one word
            dut.fc_layer_inst.sram_inst.mem[i] = 0;
            for (j = 0; j < ELEMS_PER_WORD; j = j + 1) begin
                if (elem_idx + j < K)
                    dut.fc_layer_inst.sram_inst.mem[i][j*8 +: 8] = ((elem_idx + j + 1) & 8'hFF);
            end
            elem_idx = elem_idx + ELEMS_PER_WORD;
        end

        // Set controller's write_address_reg so memory_flag = 1
        // (write_addr > read_addr → data available in SRAM)
        dut.fc_layer_inst.controller_inst.write_address_reg = PACKED_K;
        $display("SRAM pre-loaded, write_addr=%0d", PACKED_K);

        // Wait a cycle for signals to settle
        @(posedge clk);
        @(posedge clk);

        // ── Phase 2: Measure — controller should start reading ──────
        // The controller sees memory_flag=1, reg_full=0, ready_n=1
        // → starts sending w_buf_en strobes to DPE
        $display("T=%0t (cycle %0d): Waiting for DPE pipeline...", $time, cycle);

        // Wait for output
        output_count = 0;
        while (T_validn < 0 && cycle < 5000) begin
            @(posedge clk);
        end

        // Count all valid_n pulses
        while (cycle < T_validn + PACKED_K + 100) begin
            @(posedge clk);
            if (valid_n) output_count = output_count + 1;
        end
        // Include the first valid_n
        output_count = output_count + 1;

        if (T_validn < 0)
            $display("TIMEOUT: no valid_n after 5000 cycles");

        // ── Report ──────────────────────────────────────────────────
        $display("");
        $display("=== RTL Cycle Count Report (pre-loaded SRAM) ===");
        $display("K=%0d, N=%0d, DPE_BUF_WIDTH=%0d, COMPUTE_CYCLES=%0d",
                 K, N, DATA_WIDTH, `DPE_COMPUTE_CYCLES);
        $display("");
        $display("Stage Timestamps (cycle):");
        $display("  T_wbuf_first (SRAM→DPE start) = %0d", T_wbuf_first);
        $display("  T_regfull    (DPE buffer full) = %0d", T_regfull);
        $display("  T_done_first (output start)    = %0d", T_done_first);
        $display("  T_done_last  (output end)      = %0d", T_done_last);
        $display("  T_validn     (first valid_n)   = %0d", T_validn);
        $display("");
        $display("Per-Stage Cycle Counts:");
        $display("  read (SRAM→DPE)  = %0d cycles  (wbuf_first→regfull)", T_regfull - T_wbuf_first);
        $display("  compute+overhead = %0d cycles  (regfull→done_first)", T_done_first - T_regfull);
        $display("  output_serialize = %0d cycles  (done_first→done_last+1 = %0d, done_pulses=%0d)",
                 T_done_last - T_done_first + 1, T_done_last - T_done_first + 1, done_count);
        $display("  valid_n pipeline = %0d cycles  (done_first→validn)", T_validn - T_done_first);
        $display("");
        $display("  w_buf_en pulses  = %0d (expect %0d)", wbuf_count, PACKED_K);
        $display("  output count     = %0d", output_count);
        $display("");

        // Simulator-comparable total: read + compute + output
        $display("Totals:");
        $display("  DPE pipeline (wbuf_first→done_last+1) = %0d cycles", T_done_last - T_wbuf_first + 1);
        $display("  Simulator expects: read=%0d + compute=%0d + output=%0d = %0d cycles",
                 PACKED_K, `DPE_COMPUTE_CYCLES, (N + ELEMS_PER_WORD - 1) / ELEMS_PER_WORD,
                 PACKED_K + `DPE_COMPUTE_CYCLES + (N + ELEMS_PER_WORD - 1) / ELEMS_PER_WORD);
        $display("");

        #50;
        $finish;
    end

endmodule
