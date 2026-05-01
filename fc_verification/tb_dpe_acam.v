// tb_dpe_acam.v -- primitive-level ACAM smoke TB for the NL-DPE behavior
// model (FIDELITY_METHODOLOGY.md §3 ACAM modes — exp approximation).
//
// The DPE module name is `dpe` (matches VTR arch XML
// <model name="dpe"> contract). Compile and run only against
// dpe_nldpe.v (Azure-Lily has no ACAM).
//
// Test pattern:
//   ACAM_MODE = 1 (exp approximation: 1 + x + (x*x)/2)
//   weights[r][r] = 1 for r = 0 .. min(R,C)-1; all other weights = 0.
//   input vector = [1, 1, 1, ..., 1].
//   VMM result for c < R: vmm[c] = input[c] * 1 = 1.
//   ACAM result for c < R: 1 + 1 + (1*1)/2 = 1 + 1 + 0 = 2.
//   For c >= R: vmm[c] = 0, ACAM = 1 + 0 + 0 = 1. (Not exercised here:
//                                                  R == C for NL-DPE.)
//   Expected output[c] (8-bit slice): 2  for all c in 0..255.
//
// Compute timing (Model Y): the DPE itself has NO compute counter. The
// TB-as-controller asserts nl_dpe_control = 2'b11 from the first LOAD
// strobe through CCYC = (PRECISION + PIPELINE_DEPTH - 1) cycles past
// the fire posedge, then deasserts.
//
// Cycle measurement: identical to tb_dpe_vmm.v.
//   total_cycles == LOAD_STROBES + CCYC + OUTPUT_CYCLES
//                 = 52 + 10 + 52 = 114 for NL-DPE defaults
//                   (R=256, C=256, BUF=40, PRECISION=8, PIPELINE_DEPTH=3).

`timescale 1ns / 1ps

module tb_dpe_acam;
    // Clock / reset
    reg clk;
    reg reset;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Stimuli
    reg [39:0] data_in_full;
    reg        w_buf_en;
    reg [1:0]  nl_dpe_control;
    reg        shift_add_control;
    reg        shift_add_bypass;
    reg        load_output_reg;
    reg        load_input_reg;

    // DUT outputs
    wire        MSB_SA_Ready;
    wire [39:0] data_out_full;
    wire        dpe_done;
    wire        reg_full;
    wire        shift_add_done;
    wire        shift_add_bypass_ctrl;

    // Single source of truth: TB sets these localparams, DUT instantiated
    // with parameter overrides so the two cannot drift out of sync.
    // Override at compile time via:
    //   iverilog -DR_TB=1024 -DC_TB=128 -DBUF_TB=40 \
    //            -DPRECISION_TB=4 -DPIPELINE_DEPTH_TB=3 ...
    // (R >= C constraint applies. NL-DPE only.)
    //
    // Precision-driven compute pipeline (arch-agnostic):
    //   CCYC = PRECISION + PIPELINE_DEPTH - 1
    // For INT8 with 3-stage (fire -> VMM -> accumulate): CCYC = 10.
`ifndef R_TB
    `define R_TB 256
`endif
`ifndef C_TB
    `define C_TB 256
`endif
`ifndef BUF_TB
    `define BUF_TB 40
`endif
`ifndef PRECISION_TB
    `define PRECISION_TB 8
`endif
`ifndef PIPELINE_DEPTH_TB
    `define PIPELINE_DEPTH_TB 3
`endif
    localparam R              = `R_TB;
    localparam C              = `C_TB;
    localparam BUF            = `BUF_TB;
    localparam PRECISION      = `PRECISION_TB;
    localparam PIPELINE_DEPTH = `PIPELINE_DEPTH_TB;
    localparam CCYC           = PRECISION + PIPELINE_DEPTH - 1;
    localparam EPS  = BUF / 8;
    localparam LSTR = (R + EPS - 1) / EPS;
    localparam OCYC = (C + EPS - 1) / EPS;
    localparam T_FILL_EXPECTED = LSTR + CCYC + OCYC;

    dpe #(
        .KERNEL_WIDTH(R),
        .NUM_COLS(C),
        .DPE_BUF_WIDTH(BUF),
        .ACAM_MODE(1)   // 1 = exp approximation (1 + x + x^2/2)
    ) dut (
        .clk(clk),
        .reset(reset),
        .data_in(data_in_full),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready),
        .data_out(data_out_full),
        .dpe_done(dpe_done),
        .reg_full(reg_full),
        .shift_add_done(shift_add_done),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl)
    );

    integer cycle_count;
    always @(posedge clk) cycle_count <= cycle_count + 1;
    initial cycle_count = 0;

    integer T_first_load;
    integer T_done_last;
    integer i, k, b;

    reg [7:0] captured [0:1023];
    reg [7:0] expected [0:1023];

    wire [2:0] state_now = dut.state;

    integer error_count;
    integer load_strobe_idx;
    integer cap_idx;

    initial begin
        $display("[tb_dpe_acam] arch=NL-DPE ACAM_MODE=1 (exp 1+x+x^2/2) R=%0d C=%0d BUF=%0d EPS=%0d LSTR=%0d CCYC=%0d OCYC=%0d T_fill_expected=%0d", R, C, BUF, EPS, LSTR, CCYC, OCYC, T_FILL_EXPECTED);

        reset = 1;
        w_buf_en = 0;
        nl_dpe_control = 2'b00;
        shift_add_control = 0;
        shift_add_bypass = 0;
        load_output_reg = 0;
        load_input_reg = 0;
        data_in_full = 40'h0;
        T_first_load = -1;
        T_done_last = -1;
        error_count = 0;
        cap_idx = 0;
        for (i = 0; i < 1024; i = i + 1) begin
            captured[i] = 8'h00;
            expected[i] = 8'h00;
        end

        // Build expected: all 256 outputs = 2 (since input[c]=1, weights[r][r]=1 -> vmm[c]=1; ACAM exp(1) ≈ 2).
        for (i = 0; i < C; i = i + 1) begin
            if (i < R)
                expected[i] = 8'h02;
            else
                expected[i] = 8'h01;
        end

        repeat (3) @(posedge clk); #1;
        reset = 0;
        @(posedge clk); #1;

        // Hierarchical-force weights to identity on min(R, C).
        for (i = 0; i < R; i = i + 1)
            for (k = 0; k < C; k = k + 1)
                dut.weights[i][k] = 8'h00;
        for (i = 0; i < R; i = i + 1)
            if (i < C)
                dut.weights[i][i] = 8'h01;

        nl_dpe_control = 2'b11;

        // Drive 52 strobes, each carrying 5 bytes of `1`.
        for (load_strobe_idx = 0; load_strobe_idx < LSTR; load_strobe_idx = load_strobe_idx + 1) begin
            data_in_full = 40'h0;
            for (b = 0; b < EPS; b = b + 1) begin
                if (load_strobe_idx * EPS + b < R) begin
                    data_in_full[b*8 +: 8] = 8'h01;
                end
            end
            w_buf_en = 1'b1;
            @(posedge clk); #1;
            if (load_strobe_idx == 0) T_first_load = cycle_count;
        end
        w_buf_en = 1'b0;
        data_in_full = 40'h0;

        // ── Controller-side compute hold (Model Y) ──
        // Hold nl_dpe_control = 2'b11 for CCYC-1 more posedges past the
        // fire posedge, then deassert. The next posedge after deassertion
        // transitions S_COMPUTE -> S_OUTPUT, yielding total state==S_COMPUTE
        // duration = CCYC cycles.
        if (CCYC > 1) begin
            repeat (CCYC - 1) @(posedge clk);
            #1;
        end
        nl_dpe_control = 2'b00;

        // Wait for S_OUTPUT
        i = 0;
        while ((state_now != 3'd4) && (i < T_FILL_EXPECTED + 100)) begin
            @(posedge clk); #1;
            i = i + 1;
        end
        if (state_now != 3'd4) begin
            $display("[tb_dpe_acam] ERROR: FSM never reached S_OUTPUT (state=%0d)", state_now);
            error_count = error_count + 1;
        end

        // Capture (T_done_last set after each post-posedge sample so that
        // the LAST iteration captures the final strobe + S_DRAIN transition).
        cap_idx = 0;
        while (state_now == 3'd4) begin
            @(posedge clk); #1;
            for (b = 0; b < EPS; b = b + 1) begin
                if (cap_idx < C) begin
                    captured[cap_idx] = data_out_full[b*8 +: 8];
                    cap_idx = cap_idx + 1;
                end
            end
            T_done_last = cycle_count;
        end

        @(posedge clk); #1;

        // Compare
        for (i = 0; i < C; i = i + 1) begin
            if (captured[i] !== expected[i]) begin
                if (error_count < 10)
                    $display("[tb_dpe_acam] MISMATCH col=%0d expected=0x%02h got=0x%02h", i, expected[i], captured[i]);
                error_count = error_count + 1;
            end
        end

        $display("[tb_dpe_acam] T_first_load=%0d  T_done_last=%0d  total_cycles=%0d  T_fill_expected=%0d", T_first_load, T_done_last, T_done_last - T_first_load + 1, T_FILL_EXPECTED);

        if ((error_count == 0) && ((T_done_last - T_first_load + 1) == T_FILL_EXPECTED))
            $display("[tb_dpe_acam] PASS (NL-DPE ACAM=exp): %0d/%0d output bytes match, cycles=%0d (expected %0d)", C, C, T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        else
            $display("[tb_dpe_acam] FAIL (NL-DPE ACAM=exp): %0d byte mismatches; cycles=%0d (expected %0d)", error_count, T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        $finish;
    end

endmodule
