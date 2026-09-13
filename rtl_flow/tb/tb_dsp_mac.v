// tb_dsp_mac.v -- primitive-level smoke TB for the AL DSP-MAC behavior
// model (FIDELITY_METHODOLOGY.md §3 + §5: AL DIMM lane on int_sop_4 hard
// block, DSP_WIDTH = 4 int8 MACs/cycle).
//
// Test pattern (K-agnostic):
//   weight[k]       = 1   for all k
//   input_buffer[k] = 1   for all k
//   Expected MAC    = sum_{k=0..K-1}(1 * 1) = K_INPUT
//   Expected data_out[7:0] = K_INPUT mod 256
//
// Why this pattern: the original `(k+1) mod 256` input pattern aliased
// at K >= 256 because int8 sign-extension caused positive/negative
// cancellation in the MAC. The all-ones × all-ones pattern fits
// trivially in int8 (no wrapping), so mac_result = K exactly for any K
// up to ~2 billion (int32 accumulator).
//
// Cycle measurement: identical to tb_dpe_vmm.v.
//   total_cycles == LOAD + COMPUTE + OUTPUT
//                = ceil(K_INPUT*8/DPE_BUF_WIDTH)         // L
//                + ceil(K_INPUT/DSP_WIDTH)               // C
//                + max(1, ceil(PRECISION_BITS/DPE_BUF_WIDTH))  // O
//   For K=64, BUF=16, DSP_WIDTH=4: 32+16+1 = 49 cycles.
//
// Task #86 note: the DSP-MAC primitive has no bit-serial pipeline of
// the (PRECISION + PIPELINE_DEPTH - 1) form — its CCYC is K-driven
// (one DSP iteration per ceil(K/DSP_WIDTH) cycles). The per-arch
// (PIPELINE_DEPTH, ACAM_CYCLES) decomposition introduced in Task #86
// is orthogonal to DSP-MAC's compute model and does not apply here.
//
// Output check: the TB tracks the FULL 32-bit MAC accumulator hierarchically
// via dut.mac_result so we can compare the actual integer value (K_INPUT)
// and also confirm the data_out byte is the truncated low byte (K_INPUT mod 256).

`timescale 1ns / 1ps

module tb_dsp_mac;
    reg clk;
    reg reset;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // K_INPUT = 64 selected to match attention head N=128 d=64 (the
    // canonical DIMM K-tile in FIDELITY_METHODOLOGY.md §6 Stage 2).
    // Override at compile time via:
    //   iverilog -DK_TB=128 -DBUF_TB=16 -DDSP_WIDTH_TB=4 ...
`ifndef K_TB
    `define K_TB 64
`endif
`ifndef BUF_TB
    `define BUF_TB 16
`endif
`ifndef DSP_WIDTH_TB
    `define DSP_WIDTH_TB 4
`endif
    localparam K_INPUT        = `K_TB;
    localparam DPE_BUF_WIDTH  = `BUF_TB;
    localparam PRECISION_BITS = 8;
    localparam DSP_WIDTH      = `DSP_WIDTH_TB;

    // Expected mac_result for the test pattern below:
    //   weight[k] = 1   (all 1's)
    //   input[k]  = 1   (all 1's)
    // mac = sum_{k=0..K-1}(1 * 1) = K_INPUT
    // K-agnostic; no int8 sign-wrap cancellation for any K.
    // (Both operands fit trivially in int8; mac_result accumulates in
    //  int32, so exact integer K is the expected value for any K up to ~2B.)
    localparam EXPECTED_MAC  = K_INPUT;
    localparam [7:0] EXPECTED_BYTE = EXPECTED_MAC[7:0];
    localparam EPS  = DPE_BUF_WIDTH / 8;
    localparam LCYC = (K_INPUT * PRECISION_BITS + DPE_BUF_WIDTH - 1) / DPE_BUF_WIDTH;
    localparam CCYC = (K_INPUT + DSP_WIDTH - 1) / DSP_WIDTH;
    localparam OCYC_RAW = (PRECISION_BITS + DPE_BUF_WIDTH - 1) / DPE_BUF_WIDTH;
    localparam OCYC = (OCYC_RAW < 1) ? 1 : OCYC_RAW;
    // Task #87 Phase 1: T_fill = LCYC + CCYC + OCYC + 2 (FSM register-propagation
    // overhead — 1 cycle each for LOAD→COMPUTE and COMPUTE→OUTPUT handoffs).
    localparam T_FILL_EXPECTED = LCYC + CCYC + OCYC + 2;

    // Stimuli
    reg [DPE_BUF_WIDTH-1:0] data_in;
    reg                     w_buf_en;
    reg [1:0]               nl_dpe_control;
    reg                     shift_add_control;
    reg                     shift_add_bypass;
    reg                     load_output_reg;
    reg                     load_input_reg;

    // DUT outputs
    wire                    MSB_SA_Ready;
    wire [DPE_BUF_WIDTH-1:0] data_out;
    wire                    dpe_done;
    wire                    reg_full;
    wire                    shift_add_done;
    wire                    shift_add_bypass_ctrl;

    dsp_mac #(
        .K_INPUT(K_INPUT),
        .DPE_BUF_WIDTH(DPE_BUF_WIDTH),
        .PRECISION_BITS(PRECISION_BITS),
        .DSP_WIDTH(DSP_WIDTH)
    ) dut (
        .clk(clk),
        .reset(reset),
        .data_in(data_in),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready),
        .data_out(data_out),
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

    reg signed [31:0] mac_full_observed;
    reg [7:0]         data_out_low_observed;

    wire [2:0] state_now = dut.state;

    integer error_count;
    integer load_cycle_idx;
    integer cap_done;

    initial begin
        $display("[tb_dsp_mac] arch=AzureLily DSP-MAC K_INPUT=%0d BUF=%0d DSP_WIDTH=%0d EPS=%0d LCYC=%0d CCYC=%0d OCYC=%0d T_fill_expected=%0d", K_INPUT, DPE_BUF_WIDTH, DSP_WIDTH, EPS, LCYC, CCYC, OCYC, T_FILL_EXPECTED);

        reset = 1;
        w_buf_en = 0;
        nl_dpe_control = 2'b00;
        shift_add_control = 0;
        shift_add_bypass = 0;
        load_output_reg = 0;
        load_input_reg = 0;
        data_in = 0;
        T_first_load = -1;
        T_done_last = -1;
        error_count = 0;
        cap_done = 0;
        mac_full_observed = 0;
        data_out_low_observed = 0;

        repeat (3) @(posedge clk); #1;
        reset = 0;
        @(posedge clk); #1;

        // Hierarchical-force the weight vector: all 1's.
        // K-agnostic; no int8 sign-wrap concerns for any K_INPUT.
        for (k = 0; k < K_INPUT; k = k + 1)
            dut.weight[k] = 8'h01;

        nl_dpe_control = 2'b11;

        // Drive LCYC strobes; each carries EPS bytes of `1` (all 1's).
        // mac = sum_{k=0..K-1}(1 * 1) = K_INPUT (exact integer, any K).
        for (load_cycle_idx = 0; load_cycle_idx < LCYC; load_cycle_idx = load_cycle_idx + 1) begin
            data_in = 0;
            for (b = 0; b < EPS; b = b + 1) begin
                if (load_cycle_idx * EPS + b < K_INPUT) begin
                    data_in[b*8 +: 8] = 8'h01;
                end
            end
            w_buf_en = 1'b1;
            @(posedge clk); #1;
            if (load_cycle_idx == 0) T_first_load = cycle_count;
        end
        w_buf_en = 1'b0;
        data_in = 0;

        // Wait for S_OUTPUT
        i = 0;
        while ((state_now != 3'd4) && (i < T_FILL_EXPECTED + 100)) begin
            @(posedge clk); #1;
            i = i + 1;
        end
        if (state_now != 3'd4) begin
            $display("[tb_dsp_mac] ERROR: FSM never reached S_OUTPUT (state=%0d)", state_now);
            error_count = error_count + 1;
        end

        // Capture: OCYC=1 so just one strobe expected. Snapshot the full
        // mac_result hierarchically so we can compare both the truncated
        // byte and the integer accumulator value.
        while (state_now == 3'd4) begin
            @(posedge clk); #1;
            if (cap_done == 0) begin
                data_out_low_observed = data_out[7:0];
                mac_full_observed = dut.mac_result;
                cap_done = 1;
            end
            T_done_last = cycle_count;
        end

        @(posedge clk); #1;

        // Compare against EXPECTED_MAC = ((K+1)/2)^2 (sum of odd numbers 1..K-1
        // when K even, or 1..K when K odd — see localparam derivation above).
        if (mac_full_observed !== EXPECTED_MAC) begin
            $display("[tb_dsp_mac] MISMATCH mac_result expected=%0d got=%0d", EXPECTED_MAC, mac_full_observed);
            error_count = error_count + 1;
        end
        if (data_out_low_observed !== EXPECTED_BYTE) begin
            $display("[tb_dsp_mac] MISMATCH data_out[7:0] expected=0x%02h (%0d mod 256) got=0x%02h", EXPECTED_BYTE, EXPECTED_MAC, data_out_low_observed);
            error_count = error_count + 1;
        end

        $display("[tb_dsp_mac] T_first_load=%0d  T_done_last=%0d  total_cycles=%0d  T_fill_expected=%0d", T_first_load, T_done_last, T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        $display("[tb_dsp_mac] mac_result observed=%0d (expected %0d)  data_out[7:0]=0x%02h (expected 0x%02h)", mac_full_observed, EXPECTED_MAC, data_out_low_observed, EXPECTED_BYTE);

        if ((error_count == 0) && ((T_done_last - T_first_load + 1) == T_FILL_EXPECTED))
            $display("[tb_dsp_mac] PASS: mac_result=%0d, data_out=0x%02h, cycles=%0d (expected %0d)", EXPECTED_MAC, EXPECTED_BYTE, T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        else
            $display("[tb_dsp_mac] FAIL: %0d errors; cycles=%0d (expected %0d)", error_count, T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        $finish;
    end

endmodule
