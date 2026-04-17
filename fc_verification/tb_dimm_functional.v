// DEPRECATED (superseded by W=16 full DIMM top verification).
// This file was part of the Phase A-E per-stage W=1 DIMM exploration.
// Authoritative DIMM verification: fc_verification/rtl/nldpe_dimm_top_d64_c128.v
// and fc_verification/rtl/azurelily_dimm_top_d64_c128.v; see VERIFICATION.md.
//
// DIMM functional verification: identity crossbar + ACAM exp/log.
//
// Test: R=4, C=4 DPE with identity weights and ACAM_MODE=1 (exp).
// Input: x = [1, 2, 3, 4]
// Identity weights: W[r][c] = (r==c) ? 1 : 0
// VMM output: vmm[c] = Σ_r x[r] * I[r][c] = x[c]  (identity pass-through)
// ACAM exp: output[c] = 1 + x[c] + x[c]²/2
//   c=0: 1 + 1 + 0 = 2
//   c=1: 1 + 2 + 2 = 5
//   c=2: 1 + 3 + 4 = 8   (3²/2 = 4 in integer)
//   c=3: 1 + 4 + 8 = 13  (4²/2 = 8)

`timescale 1ns / 1ps

module tb_dimm_functional;

    reg clk, reset;
    reg [39:0] data_in;
    reg [1:0] nl_dpe_control;
    reg shift_add_control, w_buf_en, shift_add_bypass;
    reg load_output_reg, load_input_reg;
    wire MSB_SA_Ready;
    wire [39:0] data_out;
    wire dpe_done, reg_full, shift_add_done, shift_add_bypass_ctrl;

    // Weight loading
    reg weight_wen;
    reg [7:0] weight_data;
    reg [15:0] weight_row_addr, weight_col_addr;

    dpe #(
        .KERNEL_WIDTH(4),
        .NUM_COLS(4),
        .DPE_BUF_WIDTH(8),      // 1 byte per strobe for simplicity
        .COMPUTE_CYCLES(3),      // ACAM compute
        .ACAM_MODE(1)            // exp mode
    ) uut (
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
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl),
        .weight_wen(weight_wen),
        .weight_data(weight_data),
        .weight_row_addr(weight_row_addr),
        .weight_col_addr(weight_col_addr)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer i, pass;
    reg [31:0] expected [0:3];
    reg [31:0] got [0:3];
    integer errors;

    initial begin
        $dumpfile("tb_dimm_functional.vcd");
        $dumpvars(0, tb_dimm_functional);

        // Reset
        reset = 1; data_in = 0; nl_dpe_control = 0;
        shift_add_control = 0; w_buf_en = 0; shift_add_bypass = 0;
        load_output_reg = 0; load_input_reg = 0;
        weight_wen = 0; weight_data = 0;
        weight_row_addr = 0; weight_col_addr = 0;
        #20; reset = 0; #10;

        // ── Phase 1: Load identity weights ──
        // I[r][c] = (r==c) ? 1 : 0
        $display("Loading identity weights (4x4)...");
        for (i = 0; i < 4; i = i + 1) begin
            load_weight(i, i, 1);  // diagonal = 1
        end
        @(posedge clk); weight_wen = 0;
        $display("Identity weights loaded.");

        // ── Phase 2: Feed input x = [1, 2, 3, 4] ──
        $display("Feeding input [1, 2, 3, 4]...");
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
            w_buf_en = 1;
            data_in = i + 1;
        end
        @(posedge clk); w_buf_en = 0;

        wait(reg_full == 1);
        $display("T=%0t: reg_full asserted", $time);

        // ── Phase 3: Fire VMM + ACAM ──
        @(posedge clk); nl_dpe_control = 2'b11;
        @(posedge clk); nl_dpe_control = 2'b00;

        // Wait for output
        $display("Waiting for output (ACAM_MODE=exp)...");
        wait(uut.state == 4);  // S_OUTPUT
        @(posedge clk); #1;
        got[0] = data_out[31:0];
        $display("  Column 0: got %0d", $signed(got[0]));
        @(posedge clk); #1;
        got[1] = data_out[31:0];
        $display("  Column 1: got %0d", $signed(got[1]));
        @(posedge clk); #1;
        got[2] = data_out[31:0];
        $display("  Column 2: got %0d", $signed(got[2]));
        @(posedge clk); #1;
        got[3] = data_out[31:0];
        $display("  Column 3: got %0d", $signed(got[3]));

        // ── Phase 4: Verify ──
        // exp(x) ≈ 1 + x + x²/2 (integer division)
        // x=1: 1+1+0 = 2
        // x=2: 1+2+2 = 5
        // x=3: 1+3+4 = 8
        // x=4: 1+4+8 = 13
        expected[0] = 2;
        expected[1] = 5;
        expected[2] = 8;
        expected[3] = 13;
        errors = 0;

        for (i = 0; i < 4; i = i + 1) begin
            if ($signed(got[i]) !== $signed(expected[i])) begin
                $display("FAIL: col %0d expected %0d got %0d", i, $signed(expected[i]), $signed(got[i]));
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("PASS: DIMM exp output matches expected [%0d, %0d, %0d, %0d]",
                $signed(expected[0]), $signed(expected[1]),
                $signed(expected[2]), $signed(expected[3]));
        else
            $display("FAIL: %0d errors", errors);

        #50;
        $finish;
    end

    // Helper task: load one weight
    task load_weight;
        input [15:0] row;
        input [15:0] col;
        input [7:0] val;
        begin
            @(posedge clk);
            weight_wen = 1;
            weight_row_addr = row;
            weight_col_addr = col;
            weight_data = val;
        end
    endtask

endmodule
