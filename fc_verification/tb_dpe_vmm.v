// tb_dpe_vmm.v -- primitive-level VMM smoke TB for the generated DPE
// behavior model (FIDELITY_METHODOLOGY.md §11 step 5: "TB drives generated
// RTL with weights and inputs, checks VMM result matches oracle").
//
// Architecture switch: compile-time `+define+ARCH_NLDPE` or `+define+ARCH_AL`.
//   ARCH_NLDPE  -> instantiates dpe_stub_nldpe     (R=256, C=256, BUF=40)
//   ARCH_AL     -> instantiates dpe_stub_azurelily (R=512, C=128, BUF=16)
// If neither is defined, defaults to ARCH_NLDPE.
//
// Test pattern (architecture-agnostic):
//   weights[r][r] = 1 for r = 0 .. min(R,C)-1; all other weights = 0.
//   input vector  = [1, 2, 3, ..., R]   (mod 128 so it stays int8).
//   Expected output[c] = input[c]                        for c < min(R, C)
//                      = 0                                otherwise.
//   For NL-DPE: ACAM_MODE=0 (ADC), so vmm_result is the raw VMM.
//
// Cycle measurement (per task spec):
//   T_first_load = cycle on which the first w_buf_en posedge is sampled
//   T_done_last  = last cycle in S_OUTPUT (i.e. final cycle in which
//                  data_out carries an output strobe)
//   total_cycles = T_done_last - T_first_load + 1
//   Expected     = LOAD_STROBES + COMPUTE_CYCLES + OUTPUT_CYCLES
//                  (= T_fill from §4 / per-arch dpe_stub header).

`timescale 1ns / 1ps

`ifndef ARCH_NLDPE
  `ifndef ARCH_AL
    `define ARCH_NLDPE
  `endif
`endif

module tb_dpe_vmm;
    // Clock / reset
    reg clk;
    reg reset;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;   // 100 MHz toy clock
    end

    // Stimuli to DUT
    reg [39:0] data_in_full;     // up to 40 bits available (NL-DPE width)
    reg        w_buf_en;
    reg [1:0]  nl_dpe_control;
    reg        shift_add_control;
    reg        shift_add_bypass;
    reg        load_output_reg;
    reg        load_input_reg;

    // DUT outputs
    wire        MSB_SA_Ready;
    wire [39:0] data_out_full;   // upper bits unused for AL
    wire        dpe_done;
    wire        reg_full;
    wire        shift_add_done;
    wire        shift_add_bypass_ctrl;

    // Per-arch RTL params. Single source of truth: the TB sets these
    // localparams, and the DUT is instantiated with parameter overrides
    // so the two cannot drift out of sync. To test a different geometry
    // or precision, either change the defaults below or override at
    // compile time:
    //   iverilog -DARCH_NLDPE -DR_TB=1024 -DC_TB=128 -DBUF_TB=40 \
    //            -DPRECISION_TB=4 -DPIPELINE_DEPTH_TB=3 ...
    // (R must be >= C; BUF in {16, 40}.)
    //
    // Precision-driven compute pipeline (arch-agnostic):
    //   CCYC = PRECISION + PIPELINE_DEPTH - 1
    // For INT8 with 3-stage (fire -> VMM -> accumulate): CCYC = 10.
`ifndef PRECISION_TB
  `define PRECISION_TB 8
`endif
`ifndef PIPELINE_DEPTH_TB
  `define PIPELINE_DEPTH_TB 3
`endif

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
    localparam CCYC           = PRECISION + PIPELINE_DEPTH - 1;
    dpe_stub_nldpe #(
        .KERNEL_WIDTH(R),
        .NUM_COLS(C),
        .DPE_BUF_WIDTH(BUF),
        .PRECISION_BITS(PRECISION),
        .PIPELINE_DEPTH(PIPELINE_DEPTH)
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
    localparam CCYC           = PRECISION + PIPELINE_DEPTH - 1;
    // AL DPE_BUF_WIDTH=BUF, so wire only the low BUF bits.
    wire [BUF-1:0] data_out_al;
    dpe_stub_azurelily #(
        .KERNEL_WIDTH(R),
        .NUM_COLS(C),
        .DPE_BUF_WIDTH(BUF),
        .PRECISION_BITS(PRECISION),
        .PIPELINE_DEPTH(PIPELINE_DEPTH)
    ) dut (
        .clk(clk),
        .reset(reset),
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

    // Derived from per-arch R / C / BUF / CCYC declared above (single
    // source of truth — both TB and DUT use the same values via the
    // parameter override at instantiation).
    localparam EPS  = BUF / 8;
    localparam LSTR = (R + EPS - 1) / EPS;
    localparam OCYC = (C + EPS - 1) / EPS;
    localparam T_FILL_EXPECTED = LSTR + CCYC + OCYC;

    // ── Cycle counter ──
    // Incremented in a free-running posedge always block. Read after the
    // TB inserts a #1 settling delay following each @(posedge clk) so that
    // NBA updates have committed before the procedural read.
    integer cycle_count;
    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
    end
    initial cycle_count = 0;

    integer T_first_load;
    integer T_done_last;
    integer i, k, b;

    // Output capture: store one byte per output column slot.
    reg [7:0] captured [0:1023];   // upper bound on max C
    reg [7:0] expected [0:1023];

    // FSM state probe
    wire [2:0] state_now = dut.state;

    // TB locals
    integer error_count;
    integer load_strobe_idx;
    integer cap_idx;

    initial begin
        $display("[tb_dpe_vmm] arch=%0s R=%0d C=%0d BUF=%0d EPS=%0d LSTR=%0d CCYC=%0d OCYC=%0d T_fill_expected=%0d", `ARCH_NAME, R, C, BUF, EPS, LSTR, CCYC, OCYC, T_FILL_EXPECTED);

        // Initialise stimuli + capture buffers
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

        // Build expected outputs (input[c] mod 128 for c < min(R,C); else 0).
        for (i = 0; i < C; i = i + 1) begin
            if (i < R)
                expected[i] = (i + 1) & 8'h7F;
            else
                expected[i] = 8'h00;
        end

        // Hold reset for a few cycles. Weight initialisation is deferred
        // until AFTER reset deassertion so the dut's own `initial` block
        // (which zeroes `weights` at time-0) cannot clobber our values.
        repeat (3) @(posedge clk); #1;
        reset = 0;
        @(posedge clk); #1;

        // ── Hierarchical-force the weight matrix: identity on min(R,C) ──
        for (i = 0; i < R; i = i + 1)
            for (k = 0; k < C; k = k + 1)
                dut.weights[i][k] = 8'h00;
        for (i = 0; i < R; i = i + 1)
            if (i < C)
                dut.weights[i][i] = 8'h01;

        // Pre-assert nl_dpe_control=2'b11 so the FSM fires on the last
        // LOAD strobe combinationally (T_fill = L+C+O exactly).
        nl_dpe_control = 2'b11;

        // ── Drive LOAD_STROBES strobes ──
        // Convention: T_first_load = cycle_count value sampled (after #1
        // settle) on the cycle when w_buf_en is FIRST seen high by the FSM.
        // The FSM samples on the next posedge after we set w_buf_en=1.
        for (load_strobe_idx = 0; load_strobe_idx < LSTR; load_strobe_idx = load_strobe_idx + 1) begin
            // Pack EPS input bytes [load_strobe_idx*EPS + b] (1-indexed values).
            data_in_full = 40'h0;
            for (b = 0; b < EPS; b = b + 1) begin
                if (load_strobe_idx * EPS + b < R) begin
                    // input[r] = (r+1) mod 128
                    data_in_full[b*8 +: 8] = ((load_strobe_idx * EPS + b + 1) & 8'h7F);
                end
            end
            w_buf_en = 1'b1;
            @(posedge clk); #1;
            // First strobe: snapshot cycle counter for T_first_load
            if (load_strobe_idx == 0) T_first_load = cycle_count;
        end
        // Stop driving w_buf_en
        w_buf_en = 1'b0;
        data_in_full = 40'h0;

        // ── Capture output bytes ──
        // Wait until state == S_OUTPUT (3'd4). Then sample data_out on every
        // cycle while in S_OUTPUT, recording bytes into captured[].
        i = 0;
        while ((state_now != 3'd4) && (i < T_FILL_EXPECTED + 100)) begin
            @(posedge clk); #1;
            i = i + 1;
        end
        if (state_now != 3'd4) begin
            $display("[tb_dpe_vmm] ERROR: FSM never reached S_OUTPUT (state=%0d after %0d cycles past LOAD)", state_now, i);
            error_count = error_count + 1;
        end

        // Walk through S_OUTPUT cycles. Each posedge in S_OUTPUT updates
        // data_out via NBA with the bytes for that strobe; state stays
        // S_OUTPUT until the final strobe, after which state <= S_DRAIN.
        //
        // Sample-point semantics (sample = "after posedge X #1"):
        //   - posedge K = first cycle in S_OUTPUT (entered from S_COMPUTE);
        //     data_out at sample K is still 0 (no S_OUTPUT NBA yet).
        //   - posedge K+1 .. K+OCYC: each runs the S_OUTPUT block and NBAs
        //     data_out for output_col_idx 0 .. OCYC-1 respectively.
        //   - posedge K+OCYC: state <= S_DRAIN (final strobe NBA also runs).
        //
        // So data_out becomes valid at sample K+1, K+2, ..., K+OCYC.
        // T_done_last = cycle index of posedge K+OCYC (the last "active"
        // S_OUTPUT cycle, i.e., the cycle where the S_OUTPUT block was
        // last executed combinationally — same cycle that transitions to
        // S_DRAIN).
        cap_idx = 0;
        // First step out of "entering S_OUTPUT" sample (data_out still 0).
        // After this posedge, the first strobe NBA has committed.
        while (state_now == 3'd4) begin
            @(posedge clk); #1;
            // Whatever data_out shows now is the strobe just written by
            // the just-completed posedge's S_OUTPUT block.
            for (b = 0; b < EPS; b = b + 1) begin
                if (cap_idx < C) begin
                    captured[cap_idx] = data_out_full[b*8 +: 8];
                    cap_idx = cap_idx + 1;
                end
            end
            // Mark T_done_last to the posedge we just stepped — i.e., the
            // last posedge that ran the S_OUTPUT block (whether or not it
            // also transitioned to S_DRAIN on that same posedge).
            T_done_last = cycle_count;
        end

        // Settle a couple cycles for cleanliness
        @(posedge clk); #1;

        // ── Compare ──
        for (i = 0; i < C; i = i + 1) begin
            if (captured[i] !== expected[i]) begin
                if (error_count < 10) begin
                    $display("[tb_dpe_vmm] MISMATCH col=%0d expected=0x%02h got=0x%02h", i, expected[i], captured[i]);
                end
                error_count = error_count + 1;
            end
        end

        // ── Cycle measurement ──
        $display("[tb_dpe_vmm] T_first_load=%0d  T_done_last=%0d  total_cycles=%0d  T_fill_expected=%0d", T_first_load, T_done_last, T_done_last - T_first_load + 1, T_FILL_EXPECTED);

        // ── Verdict ──
        if ((error_count == 0) &&
            ((T_done_last - T_first_load + 1) == T_FILL_EXPECTED)) begin
            $display("[tb_dpe_vmm] PASS (%0s): %0d/%0d output bytes match, cycles=%0d (expected %0d)", `ARCH_NAME, C, C, T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        end else begin
            $display("[tb_dpe_vmm] FAIL (%0s): %0d byte mismatches; cycles=%0d (expected %0d)", `ARCH_NAME, error_count, T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        end
        $finish;
    end

endmodule
