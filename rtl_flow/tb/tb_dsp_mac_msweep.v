// tb_dsp_mac_msweep.v -- multi-pass DSP-MAC TB exercising the §4
// drain-load overlap pipeline (single-buffered).
//
// Drives M passes back-to-back of LCYC strobes each, holds
// nl_dpe_control = 2'b11 throughout, deasserts after the final pass.
// Captures M output bytes (one int8 mac result per pass) and verifies:
//   - Each pass's MAC equals K_INPUT (per the all-ones test pattern).
//   - total_cycles == T_fill + (M-1) * T_steady.
//
// Test pattern (K-agnostic):
//   weight[k] = 1 for all k
//   input[k]  = 1 for all k (every pass)
//   Expected MAC = K_INPUT for every pass.
//   Expected data_out byte = K_INPUT mod 256 for every pass.
//
// Task #86 note: DSP-MAC has no bit-serial pipeline of the
// (PRECISION + PIPELINE_DEPTH - 1) form — CCYC is K-driven
// (one DSP iteration per ceil(K/DSP_WIDTH) cycles). The per-arch
// (PIPELINE_DEPTH, ACAM_CYCLES) decomposition introduced in Task #86
// is orthogonal to DSP-MAC's compute model and does not apply here.

`timescale 1ns / 1ps

module tb_dsp_mac_msweep;
    reg clk;
    reg reset;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

`ifndef K_TB
    `define K_TB 64
`endif
`ifndef BUF_TB
    `define BUF_TB 16
`endif
`ifndef DSP_WIDTH_TB
    `define DSP_WIDTH_TB 4
`endif
`ifndef M_TB
    `define M_TB 2
`endif
    localparam K_INPUT        = `K_TB;
    localparam DPE_BUF_WIDTH  = `BUF_TB;
    localparam PRECISION_BITS = 8;
    localparam DSP_WIDTH      = `DSP_WIDTH_TB;
    localparam M_PASSES       = `M_TB;

    localparam EXPECTED_MAC   = K_INPUT;
    localparam [7:0] EXPECTED_BYTE = EXPECTED_MAC[7:0];
    localparam EPS  = DPE_BUF_WIDTH / 8;
    localparam LCYC = (K_INPUT * PRECISION_BITS + DPE_BUF_WIDTH - 1) / DPE_BUF_WIDTH;
    localparam CCYC = (K_INPUT + DSP_WIDTH - 1) / DSP_WIDTH;
    localparam OCYC_RAW = (PRECISION_BITS + DPE_BUF_WIDTH - 1) / DPE_BUF_WIDTH;
    localparam OCYC = (OCYC_RAW < 1) ? 1 : OCYC_RAW;
    // Task #87 Phase 1: T_fill = LCYC + CCYC + OCYC + 2 (FSM register-propagation
    // overhead). T_steady unchanged.
    localparam T_FILL = LCYC + CCYC + OCYC + 2;
    localparam T_STEADY_AB = (LCYC > CCYC) ? LCYC : CCYC;
    localparam T_STEADY    = (T_STEADY_AB > OCYC) ? T_STEADY_AB : OCYC;
    localparam EXPECTED_CYCLES = T_FILL + (M_PASSES - 1) * T_STEADY;

    reg [DPE_BUF_WIDTH-1:0] data_in;
    reg                     w_buf_en;
    reg [1:0]               nl_dpe_control;
    reg                     shift_add_control;
    reg                     shift_add_bypass;
    reg                     load_output_reg;
    reg                     load_input_reg;

    wire                     MSB_SA_Ready;
    wire [DPE_BUF_WIDTH-1:0] data_out;
    wire                     dpe_done;
    wire                     reg_full;
    wire                     shift_add_done;
    wire                     shift_add_bypass_ctrl;

    dsp_mac #(
        .K_INPUT(K_INPUT),
        .DPE_BUF_WIDTH(DPE_BUF_WIDTH),
        .PRECISION_BITS(PRECISION_BITS),
        .DSP_WIDTH(DSP_WIDTH)
    ) dut (
        .clk(clk), .reset(reset),
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
    integer i, k, b, mi;

    // Capture: M passes × OCYC strobes × EPS bytes each.
    reg [7:0] captured [0:M_PASSES * 64 - 1];

    integer error_count;
    integer load_cycle_idx;
    integer pass_idx;

    integer cap_strobe_idx;
    initial cap_strobe_idx = 0;
    always @(posedge clk) begin
        if (dpe_done && cap_strobe_idx < M_PASSES * OCYC) begin
            for (b = 0; b < EPS; b = b + 1) begin
                if (cap_strobe_idx * EPS + b < M_PASSES * 64)
                    captured[cap_strobe_idx * EPS + b] = data_out[b*8 +: 8];
            end
            cap_strobe_idx = cap_strobe_idx + 1;
            T_done_last = cycle_count;
        end
    end

    initial begin
        $display("[tb_dsp_mac_msweep] M=%0d K=%0d BUF=%0d DSP=%0d EPS=%0d LCYC=%0d CCYC=%0d OCYC=%0d T_fill=%0d T_steady=%0d expected=%0d",
                 M_PASSES, K_INPUT, DPE_BUF_WIDTH, DSP_WIDTH, EPS, LCYC, CCYC, OCYC, T_FILL, T_STEADY, EXPECTED_CYCLES);

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
        for (i = 0; i < M_PASSES * 64; i = i + 1)
            captured[i] = 8'h00;

        repeat (3) @(posedge clk); #1;
        reset = 0;
        @(posedge clk); #1;

        for (k = 0; k < K_INPUT; k = k + 1)
            dut.weight[k] = 8'h01;

        nl_dpe_control = 2'b11;

        // Drive M_PASSES * LCYC strobes back-to-back.
        for (pass_idx = 0; pass_idx < M_PASSES; pass_idx = pass_idx + 1) begin
            for (load_cycle_idx = 0; load_cycle_idx < LCYC; load_cycle_idx = load_cycle_idx + 1) begin
                data_in = 0;
                for (b = 0; b < EPS; b = b + 1) begin
                    if (load_cycle_idx * EPS + b < K_INPUT)
                        data_in[b*8 +: 8] = 8'h01;
                end
                w_buf_en = 1'b1;
                @(posedge clk); #1;
                if (pass_idx == 0 && load_cycle_idx == 0) T_first_load = cycle_count;
            end
        end
        w_buf_en = 1'b0;
        data_in = 0;
        nl_dpe_control = 2'b00;

        // Wait for all captures.
        begin : wait_drain
            integer guard;
            guard = 0;
            while (cap_strobe_idx < M_PASSES * OCYC && guard < EXPECTED_CYCLES * 4 + 200) begin
                @(posedge clk); #1;
                guard = guard + 1;
            end
            if (cap_strobe_idx < M_PASSES * OCYC) begin
                $display("[tb_dsp_mac_msweep] ERROR: only captured %0d/%0d strobes", cap_strobe_idx, M_PASSES * OCYC);
                error_count = error_count + 1;
            end
        end

        @(posedge clk); #1;

        // Compare: pass m's first byte (slot m*OCYC*EPS) should be EXPECTED_BYTE.
        for (mi = 0; mi < M_PASSES; mi = mi + 1) begin
            if (captured[mi * OCYC * EPS] !== EXPECTED_BYTE) begin
                if (error_count < 10)
                    $display("[tb_dsp_mac_msweep] MISMATCH pass=%0d expected=0x%02h got=0x%02h",
                             mi, EXPECTED_BYTE, captured[mi * OCYC * EPS]);
                error_count = error_count + 1;
            end
        end

        $display("[tb_dsp_mac_msweep] T_first_load=%0d  T_done_last=%0d  total_cycles=%0d  expected=%0d",
                 T_first_load, T_done_last, T_done_last - T_first_load + 1, EXPECTED_CYCLES);

        if ((error_count == 0) &&
            ((T_done_last - T_first_load + 1) == EXPECTED_CYCLES))
            $display("[tb_dsp_mac_msweep] PASS (M=%0d): %0d/%0d passes match, cycles=%0d (expected %0d)",
                     M_PASSES, M_PASSES, M_PASSES, T_done_last - T_first_load + 1, EXPECTED_CYCLES);
        else
            $display("[tb_dsp_mac_msweep] FAIL (M=%0d): %0d errors; cycles=%0d (expected %0d)",
                     M_PASSES, error_count, T_done_last - T_first_load + 1, EXPECTED_CYCLES);
        $finish;
    end

endmodule
