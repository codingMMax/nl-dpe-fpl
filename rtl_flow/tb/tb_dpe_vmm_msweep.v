// tb_dpe_vmm_msweep.v -- multi-pass DPE VMM TB exercising the §4
// drain-load overlap pipeline (single-buffered).
//
// Drives M back-to-back passes of LOAD strobes (LCYC * M strobes total),
// holds nl_dpe_control = 2'b11 throughout, then deasserts after the
// final pass. Captures M * OCYC output bytes from data_out IN PARALLEL
// with the LOAD strobing (since the §4 overlap means OUTPUT begins
// before LOAD finishes for M > 1).
//
// Verifies:
//   - Each pass's bytes match the expected VMM result for its input.
//   - total_cycles == T_fill + (M-1) * T_steady.
//
// Architecture switch via compile-time `+define+`:
//   ARCH_NLDPE -> dpe_nldpe.v   (R=256, C=256, BUF=40)
//   ARCH_AL    -> dpe_azurelily.v (R=512, C=128, BUF=16)
//
// Test pattern: per-pass constant inputs (m+1) for r in [0, min(R,C)).
//   weights[r][c] = 1 if r==c (identity); else 0.
//   input[m][r] = (m + 1) for r < min(R, C); else 0.
//   VMM[m][c] = (m + 1) for c < min(R, C); else 0.
//   Expected byte at (pass m, col c) = (m + 1) & 0xFF for c < min(R, C);
//                                       0           otherwise.

`timescale 1ns / 1ps

`ifndef ARCH_NLDPE
  `ifndef ARCH_AL
    `define ARCH_NLDPE
  `endif
`endif

module tb_dpe_vmm_msweep;
    reg clk;
    reg reset;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    reg [39:0] data_in_full;
    reg        w_buf_en;
    reg [1:0]  nl_dpe_control;
    reg        shift_add_control;
    reg        shift_add_bypass;
    reg        load_output_reg;
    reg        load_input_reg;

    wire        MSB_SA_Ready;
    wire [39:0] data_out_full;
    wire        dpe_done;
    wire        reg_full;
    wire        shift_add_done;
    wire        shift_add_bypass_ctrl;

`ifndef PRECISION_TB
  `define PRECISION_TB 8
`endif
`ifndef PIPELINE_DEPTH_TB
  `define PIPELINE_DEPTH_TB 3
`endif
`ifndef ACAM_CYCLES_TB
  `define ACAM_CYCLES_TB 0
`endif
`ifndef M_TB
  `define M_TB 2
`endif

// Per-arch CCYC decomposition (Task #86):
//   CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES
// NL-DPE     : PD=2 (MAC, Acc) + AC=1 (read-out)   → CCYC = P + 2
// Azure-Lily : PD=3 (MAC, ADC, SA), no ACAM         → CCYC = P + 2
// Legacy single-knob compat: ACAM_CYCLES_TB defaults to 0; passing
// -DPIPELINE_DEPTH_TB=3 alone matches pre-Task-#86 behaviour.

`ifdef ARCH_NLDPE
    `define ARCH_NAME "NL-DPE"
  `ifndef R_TB
    `define R_TB 256
  `endif
  `ifndef C_TB
    `define C_TB 256
  `endif
  `ifndef BUF_TB
    `define BUF_TB 40
  `endif
    localparam R              = `R_TB;
    localparam C              = `C_TB;
    localparam BUF            = `BUF_TB;
    localparam PRECISION      = `PRECISION_TB;
    localparam PIPELINE_DEPTH = `PIPELINE_DEPTH_TB;
    localparam ACAM_CYCLES    = `ACAM_CYCLES_TB;
    localparam CCYC           = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES;
    dpe #(
        .KERNEL_WIDTH(R),
        .NUM_COLS(C),
        .DPE_BUF_WIDTH(BUF),
        .COMPUTE_CYCLES(CCYC)
    ) dut (
        .clk(clk), .reset(reset),
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
`endif
`ifdef ARCH_AL
    `define ARCH_NAME "AzureLily"
  `ifndef R_TB
    `define R_TB 512
  `endif
  `ifndef C_TB
    `define C_TB 128
  `endif
  `ifndef BUF_TB
    `define BUF_TB 16
  `endif
    localparam R              = `R_TB;
    localparam C              = `C_TB;
    localparam BUF            = `BUF_TB;
    localparam PRECISION      = `PRECISION_TB;
    localparam PIPELINE_DEPTH = `PIPELINE_DEPTH_TB;
    localparam ACAM_CYCLES    = `ACAM_CYCLES_TB;
    localparam CCYC           = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES;
    wire [BUF-1:0] data_out_al;
    dpe #(
        .KERNEL_WIDTH(R),
        .NUM_COLS(C),
        .DPE_BUF_WIDTH(BUF),
        .COMPUTE_CYCLES(CCYC)
    ) dut (
        .clk(clk), .reset(reset),
        .data_in(data_in_full[BUF-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready),
        .data_out(data_out_al),
        .dpe_done(dpe_done),
        .reg_full(reg_full),
        .shift_add_done(shift_add_done),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl)
    );
    assign data_out_full = {{(40-BUF){1'b0}}, data_out_al};
`endif

    localparam M_PASSES = `M_TB;
    localparam EPS  = BUF / 8;
    localparam LCYC = (R + EPS - 1) / EPS;
    localparam OCYC = (C + EPS - 1) / EPS;
    // Task #87 Phase 1: T_fill = LCYC + CCYC + OCYC + 2 (FSM register-propagation
    // overhead). T_steady = max(LCYC, CCYC, OCYC) unchanged (handoffs amortised
    // by OUTPUT chain across passes).
    localparam T_FILL = LCYC + CCYC + OCYC + 2;
    localparam T_STEADY_AB = (LCYC > CCYC) ? LCYC : CCYC;
    localparam T_STEADY    = (T_STEADY_AB > OCYC) ? T_STEADY_AB : OCYC;
    localparam EXPECTED_CYCLES = T_FILL + (M_PASSES - 1) * T_STEADY;

    integer cycle_count;
    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
    end
    initial cycle_count = 0;

    integer T_first_load;
    integer T_done_last;
    integer i, k, b, mi;

    // Capture buffer: M passes × OCYC strobes × EPS bytes each.
    // Index: (m * OCYC + s) * EPS + b for byte b of strobe s in pass m.
    // For the test pattern, byte at column index (s*EPS+b) for pass m
    // should equal expected[m * C + col] when col < C.
    reg [7:0] captured [0:M_PASSES * 1024 - 1];
    reg [7:0] expected [0:M_PASSES * 1024 - 1];

    wire [2:0] state_now = dut.state;

    integer error_count;
    integer load_cycle_idx;
    integer pass_idx;

    // ── Continuous output capture ─────────────────────────────────────
    // Capture mechanism: reads data_out_full at every posedge where
    // dpe_done is sampled high. dpe_done is NBA'd to 1 by the DPE one
    // posedge after output_busy goes high, which is exactly the posedge
    // following any OUTPUT-block NBA of data_out. So `dpe_done` sample
    // = 1 implies that the previous posedge's data_out NBA produced a
    // real OUTPUT byte; we read that NBA value here (pre-NBA reads in
    // this always block return the previous posedge's committed value).
    //
    // For multi-pass with no gap (e.g., NL-DPE LCYC=OCYC=52), dpe_done
    // stays high continuously across all M passes (M*OCYC cycles).
    // For multi-pass with a gap (e.g., AL LCYC=256, OCYC=64), dpe_done
    // pulses high for OCYC cycles per pass, with idle periods between
    // passes. The capture indexer naturally pauses during gaps.
    //
    // T_done_last: the LOGICAL cycle when the DPE NBA'd the last byte
    // (= cycle_count - 1 at the capture posedge, since data_out we read
    // is the previous posedge's NBA). This matches T(M) = T_fill +
    // (M-1) * T_steady cycle-count semantics.
    integer cap_strobe_idx;       // strobe index within the capture stream
    initial begin
        cap_strobe_idx  = 0;
    end
    always @(posedge clk) begin
        if (dpe_done && cap_strobe_idx < M_PASSES * OCYC) begin
            for (b = 0; b < EPS; b = b + 1) begin
                if (cap_strobe_idx * EPS + b < M_PASSES * 1024)
                    captured[cap_strobe_idx * EPS + b] = data_out_full[b*8 +: 8];
            end
            cap_strobe_idx = cap_strobe_idx + 1;
            // T_done_last: post-NBA cycle_count at this capture posedge.
            // Matches T_first_load convention so total_cycles = T_done_last
            // - T_first_load + 1 corresponds to T_fill + (M-1)*T_steady.
            T_done_last = cycle_count;
        end
    end

    initial begin
        $display("[tb_dpe_vmm_msweep] arch=%0s M=%0d R=%0d C=%0d BUF=%0d EPS=%0d LCYC=%0d CCYC=%0d (P=%0d + (D=%0d - 1) + A=%0d) OCYC=%0d T_fill=%0d T_steady=%0d expected=%0d",
                 `ARCH_NAME, M_PASSES, R, C, BUF, EPS, LCYC, CCYC, PRECISION, PIPELINE_DEPTH, ACAM_CYCLES, OCYC, T_FILL, T_STEADY, EXPECTED_CYCLES);

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
        for (i = 0; i < M_PASSES * 1024; i = i + 1) begin
            captured[i] = 8'h00;
            expected[i] = 8'h00;
        end

        // Build expected outputs.
        // For pass m, the OUTPUT phase emits OCYC strobes of EPS bytes each.
        // captured[(m * OCYC + s) * EPS + b] should equal:
        //   (m+1) & 0xFF  if (s * EPS + b) < min(R, C)
        //   0             otherwise (padding for col >= C; col >= R also yields 0).
        for (mi = 0; mi < M_PASSES; mi = mi + 1) begin
            for (i = 0; i < OCYC * EPS; i = i + 1) begin
                if (i < R && i < C)
                    expected[mi * OCYC * EPS + i] = (mi + 1) & 8'hFF;
                else
                    expected[mi * OCYC * EPS + i] = 8'h00;
            end
        end

        repeat (3) @(posedge clk); #1;
        reset = 0;
        @(posedge clk); #1;

        // Identity weights (R x C, identity on min(R,C)).
        for (i = 0; i < R; i = i + 1)
            for (k = 0; k < C; k = k + 1)
                dut.weights[i][k] = 8'h00;
        for (i = 0; i < R; i = i + 1)
            if (i < C)
                dut.weights[i][i] = 8'h01;

        nl_dpe_control = 2'b11;

        // Drive M_PASSES * LCYC strobes back-to-back.
        for (pass_idx = 0; pass_idx < M_PASSES; pass_idx = pass_idx + 1) begin
            for (load_cycle_idx = 0; load_cycle_idx < LCYC; load_cycle_idx = load_cycle_idx + 1) begin
                data_in_full = 40'h0;
                for (b = 0; b < EPS; b = b + 1) begin
                    if (load_cycle_idx * EPS + b < R) begin
                        data_in_full[b*8 +: 8] = (pass_idx + 1) & 8'hFF;
                    end
                end
                w_buf_en = 1'b1;
                @(posedge clk); #1;
                if (pass_idx == 0 && load_cycle_idx == 0) T_first_load = cycle_count;
            end
        end
        w_buf_en = 1'b0;
        data_in_full = 40'h0;
        nl_dpe_control = 2'b00;

        // Wait for all M passes' OUTPUT to drain. The capture always-block
        // sets cap_strobe_idx; we wait until it reaches M*OCYC, plus a
        // few extra cycles for cleanliness.
        begin : wait_drain
            integer guard;
            guard = 0;
            while (cap_strobe_idx < M_PASSES * OCYC && guard < EXPECTED_CYCLES * 4 + 200) begin
                @(posedge clk); #1;
                guard = guard + 1;
            end
            if (cap_strobe_idx < M_PASSES * OCYC) begin
                $display("[tb_dpe_vmm_msweep] ERROR: only captured %0d/%0d strobes after %0d cycles",
                         cap_strobe_idx, M_PASSES * OCYC, guard);
                error_count = error_count + 1;
            end
        end

        @(posedge clk); #1;

        // Compare per-pass bytes (only the first min(R,C) of each pass's
        // OCYC*EPS slots; the rest are padding zeros).
        for (mi = 0; mi < M_PASSES; mi = mi + 1) begin
            for (i = 0; i < OCYC * EPS; i = i + 1) begin
                if (captured[mi * OCYC * EPS + i] !== expected[mi * OCYC * EPS + i]) begin
                    if (error_count < 10) begin
                        $display("[tb_dpe_vmm_msweep] MISMATCH pass=%0d slot=%0d expected=0x%02h got=0x%02h",
                                 mi, i,
                                 expected[mi * OCYC * EPS + i],
                                 captured[mi * OCYC * EPS + i]);
                    end
                    error_count = error_count + 1;
                end
            end
        end

        $display("[tb_dpe_vmm_msweep] T_first_load=%0d  T_done_last=%0d  total_cycles=%0d  expected=%0d",
                 T_first_load, T_done_last, T_done_last - T_first_load + 1, EXPECTED_CYCLES);

        if ((error_count == 0) &&
            ((T_done_last - T_first_load + 1) == EXPECTED_CYCLES)) begin
            $display("[tb_dpe_vmm_msweep] PASS (%0s M=%0d): %0d slots match, cycles=%0d (expected %0d)",
                     `ARCH_NAME, M_PASSES, M_PASSES * OCYC * EPS,
                     T_done_last - T_first_load + 1, EXPECTED_CYCLES);
        end else begin
            $display("[tb_dpe_vmm_msweep] FAIL (%0s M=%0d): %0d slot mismatches; cycles=%0d (expected %0d)",
                     `ARCH_NAME, M_PASSES, error_count,
                     T_done_last - T_first_load + 1, EXPECTED_CYCLES);
        end
        $finish;
    end

endmodule
