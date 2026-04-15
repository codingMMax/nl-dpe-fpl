// Alignment testbench for fc_2048_256 with V=4, H=2 (Setup 0/1/2: 512×128)
// Pre-loads per-row SRAMs, measures parallel DPE pipeline.
//
// Hierarchy: dut(fc_top) → fc_layer_inst(fc_layer) → sram_r{0-3}, ctrl_r{0-3},
//            dpe_c{0-1}_r{0-3}

`timescale 1ns / 1ps

`ifndef DPE_BUF_WIDTH
`define DPE_BUF_WIDTH 16
`endif
`ifndef DPE_COMPUTE_CYCLES
`define DPE_COMPUTE_CYCLES 44
`endif
`ifndef N_ROWS
`define N_ROWS 4
`endif
`ifndef KW_ROW
`define KW_ROW 512
`endif

module tb_alignment_v4h2;

    parameter DATA_WIDTH = `DPE_BUF_WIDTH;
    parameter K = 2048;
    parameter N = 256;
    parameter V = `N_ROWS;
    parameter H = 2;
    parameter KW_ROW = `KW_ROW;  // elements per row
    parameter ELEMS_PER_WORD = DATA_WIDTH / 8;
    parameter PACKED_KW_ROW = (KW_ROW + ELEMS_PER_WORD - 1) / ELEMS_PER_WORD;
    parameter PACKED_N_COL = (N / H + ELEMS_PER_WORD - 1) / ELEMS_PER_WORD;

    reg clk, rst, valid, ready_n;
    reg [DATA_WIDTH-1:0] data_in;
    wire [DATA_WIDTH-1:0] data_out;
    wire ready, valid_n;

    // Override DPE parameters for all DPE instances (V=4, H=2 = 8 DPEs)
    defparam dut.fc_layer_inst.dpe_c0_r0.DPE_BUF_WIDTH = `DPE_BUF_WIDTH;
    defparam dut.fc_layer_inst.dpe_c0_r0.COMPUTE_CYCLES = `DPE_COMPUTE_CYCLES;
    defparam dut.fc_layer_inst.dpe_c1_r0.DPE_BUF_WIDTH = `DPE_BUF_WIDTH;
    defparam dut.fc_layer_inst.dpe_c1_r0.COMPUTE_CYCLES = `DPE_COMPUTE_CYCLES;
    defparam dut.fc_layer_inst.dpe_c0_r1.DPE_BUF_WIDTH = `DPE_BUF_WIDTH;
    defparam dut.fc_layer_inst.dpe_c0_r1.COMPUTE_CYCLES = `DPE_COMPUTE_CYCLES;
    defparam dut.fc_layer_inst.dpe_c1_r1.DPE_BUF_WIDTH = `DPE_BUF_WIDTH;
    defparam dut.fc_layer_inst.dpe_c1_r1.COMPUTE_CYCLES = `DPE_COMPUTE_CYCLES;
`ifdef HAS_ROW2
    defparam dut.fc_layer_inst.dpe_c0_r2.DPE_BUF_WIDTH = `DPE_BUF_WIDTH;
    defparam dut.fc_layer_inst.dpe_c0_r2.COMPUTE_CYCLES = `DPE_COMPUTE_CYCLES;
    defparam dut.fc_layer_inst.dpe_c1_r2.DPE_BUF_WIDTH = `DPE_BUF_WIDTH;
    defparam dut.fc_layer_inst.dpe_c1_r2.COMPUTE_CYCLES = `DPE_COMPUTE_CYCLES;
`endif
`ifdef HAS_ROW3
    defparam dut.fc_layer_inst.dpe_c0_r3.DPE_BUF_WIDTH = `DPE_BUF_WIDTH;
    defparam dut.fc_layer_inst.dpe_c0_r3.COMPUTE_CYCLES = `DPE_COMPUTE_CYCLES;
    defparam dut.fc_layer_inst.dpe_c1_r3.DPE_BUF_WIDTH = `DPE_BUF_WIDTH;
    defparam dut.fc_layer_inst.dpe_c1_r3.COMPUTE_CYCLES = `DPE_COMPUTE_CYCLES;
`endif

    // DUT
    fc_top #(.DATA_WIDTH(DATA_WIDTH)) dut (
        .clk(clk), .rst(rst), .valid(valid), .ready_n(ready_n),
        .data_in(data_in), .data_out(data_out), .ready(ready), .valid_n(valid_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Cycle counter
    integer cycle;
    always @(posedge clk) begin
        if (rst) cycle <= 0;
        else cycle <= cycle + 1;
    end

    // Stage timestamps — probe row 0's DPE (all rows are parallel, same timing)
    integer T_wbuf_first = -1;
    integer T_regfull    = -1;
    integer T_done_first = -1;
    integer T_done_last  = -1;
    integer T_validn     = -1;

    integer wbuf_count = 0;
    integer done_count = 0;

    wire probe_wbuf_en  = dut.fc_layer_inst.dpe_c0_r0.w_buf_en;
    wire probe_reg_full = dut.fc_layer_inst.dpe_c0_r0.reg_full;
    wire probe_dpe_done = dut.fc_layer_inst.dpe_c0_r0.dpe_done;

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

    integer i, j, row, elem_idx;

    initial begin
        $dumpfile("tb_alignment_v4h2.vcd");
        $dumpvars(0, tb_alignment_v4h2);

        rst = 1; valid = 0; ready_n = 1; data_in = 0;
        #20; rst = 0;
        // Wait for reset to fully propagate through all controllers
        @(posedge clk); @(posedge clk); @(posedge clk);

        $display("=== Alignment TB (pre-load): K=%0d, N=%0d, V=%0d, H=%0d ===", K, N, V, H);
        $display("    DPE_BUF_WIDTH=%0d, COMPUTE_CYCLES=%0d", DATA_WIDTH, `DPE_COMPUTE_CYCLES);
        $display("    elems_per_word=%0d, packed_kw_row=%0d", ELEMS_PER_WORD, PACKED_KW_ROW);

        // Pre-load all per-row SRAMs with packed data.
        // Force between clock edges so controller registers don't overwrite.
        #1;
        // Row 0
        elem_idx = 0;
        for (i = 0; i < PACKED_KW_ROW; i = i + 1) begin
            dut.fc_layer_inst.sram_r0.mem[i] = 0;
            for (j = 0; j < ELEMS_PER_WORD; j = j + 1)
                if (elem_idx + j < KW_ROW)
                    dut.fc_layer_inst.sram_r0.mem[i][j*8 +: 8] = ((elem_idx + j + 1) & 8'hFF);
            elem_idx = elem_idx + ELEMS_PER_WORD;
        end
        dut.fc_layer_inst.ctrl_r0.next_address = PACKED_KW_ROW;

        // Row 1
        elem_idx = 0;
        for (i = 0; i < PACKED_KW_ROW; i = i + 1) begin
            dut.fc_layer_inst.sram_r1.mem[i] = 0;
            for (j = 0; j < ELEMS_PER_WORD; j = j + 1)
                if (elem_idx + j < KW_ROW)
                    dut.fc_layer_inst.sram_r1.mem[i][j*8 +: 8] = ((elem_idx + j + 1) & 8'hFF);
            elem_idx = elem_idx + ELEMS_PER_WORD;
        end
        dut.fc_layer_inst.ctrl_r1.next_address = PACKED_KW_ROW;

`ifdef HAS_ROW2
        // Row 2
        elem_idx = 0;
        for (i = 0; i < PACKED_KW_ROW; i = i + 1) begin
            dut.fc_layer_inst.sram_r2.mem[i] = 0;
            for (j = 0; j < ELEMS_PER_WORD; j = j + 1)
                if (elem_idx + j < KW_ROW)
                    dut.fc_layer_inst.sram_r2.mem[i][j*8 +: 8] = ((elem_idx + j + 1) & 8'hFF);
            elem_idx = elem_idx + ELEMS_PER_WORD;
        end
        dut.fc_layer_inst.ctrl_r2.next_address = PACKED_KW_ROW;
`endif
`ifdef HAS_ROW3
        // Row 3
        elem_idx = 0;
        for (i = 0; i < PACKED_KW_ROW; i = i + 1) begin
            dut.fc_layer_inst.sram_r3.mem[i] = 0;
            for (j = 0; j < ELEMS_PER_WORD; j = j + 1)
                if (elem_idx + j < KW_ROW)
                    dut.fc_layer_inst.sram_r3.mem[i][j*8 +: 8] = ((elem_idx + j + 1) & 8'hFF);
            elem_idx = elem_idx + ELEMS_PER_WORD;
        end
        dut.fc_layer_inst.ctrl_r3.next_address = PACKED_KW_ROW;
`endif

        $display("SRAMs pre-loaded (%0d rows × %0d packed words)", V, PACKED_KW_ROW);

        @(posedge clk); @(posedge clk);
        $display("T=%0t (cycle %0d): Waiting for DPE pipeline...", $time, cycle);

        // Wait for valid_n
        while (T_validn < 0 && cycle < 5000) @(posedge clk);

        if (T_validn < 0) $display("TIMEOUT: no valid_n");

        // Wait for all DPE output cycles to complete (dpe_done goes low)
        wait(T_done_first >= 0);  // ensure output started
        while (probe_dpe_done) @(posedge clk);
        // One more cycle for drain
        @(posedge clk);

        // Report
        $display("");
        $display("=== RTL Cycle Count Report (fc_2048_256, V=%0d H=%0d) ===", V, H);
        $display("DPE_BUF_WIDTH=%0d, COMPUTE_CYCLES=%0d", DATA_WIDTH, `DPE_COMPUTE_CYCLES);
        $display("");
        $display("Stage Timestamps (cycle, measured at row 0 DPE):");
        $display("  T_wbuf_first = %0d", T_wbuf_first);
        $display("  T_regfull    = %0d", T_regfull);
        $display("  T_done_first = %0d", T_done_first);
        $display("  T_done_last  = %0d", T_done_last);
        $display("  T_validn     = %0d  (top-level, all rows AND'd)", T_validn);
        $display("");
        $display("Per-Stage Cycle Counts:");
        $display("  read (SRAM→DPE)  = %0d cycles", T_regfull - T_wbuf_first);
        $display("  compute+overhead = %0d cycles", T_done_first - T_regfull);
        $display("  output_serialize = %0d cycles  (done_pulses=%0d)", T_done_last - T_done_first + 1, done_count);
        $display("  reduction+act    = %0d cycles  (done_first→validn)", T_validn - T_done_first);
        $display("");
        $display("Totals:");
        $display("  DPE pipeline (wbuf→done_last+1) = %0d cycles", T_done_last - T_wbuf_first + 1);
        $display("  Full pipeline (wbuf→validn)      = %0d cycles", T_validn - T_wbuf_first);
        $display("  Simulator expects: read=%0d + compute=%0d + output=%0d + reduc=%0d = %0d",
                 PACKED_KW_ROW, `DPE_COMPUTE_CYCLES, PACKED_N_COL,
                 (V > 1 ? $clog2(V) : 0),
                 PACKED_KW_ROW + `DPE_COMPUTE_CYCLES + PACKED_N_COL + (V > 1 ? $clog2(V) : 0));
        $display("");

        #50; $finish;
    end

endmodule
