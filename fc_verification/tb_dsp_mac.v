// tb_dsp_mac.v -- primitive-level smoke TB for the AL DSP-MAC behavior
// model (FIDELITY_METHODOLOGY.md §3 + §5: AL DIMM lane on int_sop_4 hard
// block, DSP_WIDTH = 4 int8 MACs/cycle).
//
// Test pattern:
//   K_INPUT = 64
//   weight[k]       = 1 for even k, 0 for odd k    (alternating)
//   input_buffer[k] = (k + 1)                       (1, 2, 3, ..., 64)
//   Expected MAC    = sum_k input[k] * weight[k]
//                   = 1 + 3 + 5 + ... + 63
//                   = 32 * 32 = 1024  (overflow into 32-bit accumulator,
//                                      truncated to int8 on output port).
//   Expected data_out[7:0] = 1024 mod 256 = 0x00.
//
// Cycle measurement: identical to tb_dpe_vmm.v.
//   total_cycles == LOAD + COMPUTE + OUTPUT
//                = ceil(K_INPUT*8/DPE_BUF_WIDTH)         // 32
//                + ceil(K_INPUT/DSP_WIDTH)               // 16
//                + max(1, ceil(PRECISION_BITS/DPE_BUF_WIDTH))  // 1
//                = 49 cycles total.
//
// Output check: the TB tracks the FULL 32-bit MAC accumulator hierarchically
// via dut.mac_result so we can compare the actual integer value (1024) and
// also confirm the data_out byte is the truncated low byte (0x00).

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
    localparam K_INPUT        = 64;
    localparam DPE_BUF_WIDTH  = 16;
    localparam PRECISION_BITS = 8;
    localparam DSP_WIDTH      = 4;
    localparam EPS  = DPE_BUF_WIDTH / 8;
    localparam LSTR = (K_INPUT * PRECISION_BITS + DPE_BUF_WIDTH - 1) / DPE_BUF_WIDTH;
    localparam CCYC = (K_INPUT + DSP_WIDTH - 1) / DSP_WIDTH;
    localparam OCYC_RAW = (PRECISION_BITS + DPE_BUF_WIDTH - 1) / DPE_BUF_WIDTH;
    localparam OCYC = (OCYC_RAW < 1) ? 1 : OCYC_RAW;
    localparam T_FILL_EXPECTED = LSTR + CCYC + OCYC;

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
    integer load_strobe_idx;
    integer cap_done;

    initial begin
        $display("[tb_dsp_mac] arch=AzureLily DSP-MAC K_INPUT=%0d BUF=%0d DSP_WIDTH=%0d EPS=%0d LSTR=%0d CCYC=%0d OCYC=%0d T_fill_expected=%0d", K_INPUT, DPE_BUF_WIDTH, DSP_WIDTH, EPS, LSTR, CCYC, OCYC, T_FILL_EXPECTED);

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

        // Hierarchical-force the weight vector: alternating 1, 0, 1, 0, ...
        for (k = 0; k < K_INPUT; k = k + 1) begin
            if ((k % 2) == 0)
                dut.weight[k] = 8'h01;
            else
                dut.weight[k] = 8'h00;
        end

        nl_dpe_control = 2'b11;

        // Drive LSTR = 32 strobes; each carries 2 bytes (BUF=16).
        // Strobe k -> input_buffer[k*EPS + b] = (k*EPS + b + 1) for b in 0..1
        // -> packed: data_in[0..7] = byte0, data_in[8..15] = byte1.
        for (load_strobe_idx = 0; load_strobe_idx < LSTR; load_strobe_idx = load_strobe_idx + 1) begin
            data_in = 0;
            for (b = 0; b < EPS; b = b + 1) begin
                if (load_strobe_idx * EPS + b < K_INPUT) begin
                    data_in[b*8 +: 8] = (load_strobe_idx * EPS + b + 1);  // 1..64
                end
            end
            w_buf_en = 1'b1;
            @(posedge clk); #1;
            if (load_strobe_idx == 0) T_first_load = cycle_count;
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

        // Compare
        if (mac_full_observed !== 32'sd1024) begin
            $display("[tb_dsp_mac] MISMATCH mac_result expected=1024 got=%0d", mac_full_observed);
            error_count = error_count + 1;
        end
        if (data_out_low_observed !== 8'h00) begin
            $display("[tb_dsp_mac] MISMATCH data_out[7:0] expected=0x00 (1024 mod 256) got=0x%02h", data_out_low_observed);
            error_count = error_count + 1;
        end

        $display("[tb_dsp_mac] T_first_load=%0d  T_done_last=%0d  total_cycles=%0d  T_fill_expected=%0d", T_first_load, T_done_last, T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        $display("[tb_dsp_mac] mac_result observed=%0d (expected 1024)  data_out[7:0]=0x%02h (expected 0x00)", mac_full_observed, data_out_low_observed);

        if ((error_count == 0) && ((T_done_last - T_first_load + 1) == T_FILL_EXPECTED))
            $display("[tb_dsp_mac] PASS: mac_result=1024, data_out=0x00, cycles=%0d (expected %0d)", T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        else
            $display("[tb_dsp_mac] FAIL: %0d errors; cycles=%0d (expected %0d)", error_count, T_done_last - T_first_load + 1, T_FILL_EXPECTED);
        $finish;
    end

endmodule
