// Testbench: verify DPE behavioral model computes correct VMM.
//
// Test case: small DPE (R=4, C=2) with known weights and inputs.
// Weight matrix W[4][2]:  [[1,2],[3,4],[5,6],[7,8]]
// Input vector x[4]:      [1, 2, 3, 4]
// Expected output:
//   y[0] = 1*1 + 2*3 + 3*5 + 4*7 = 1 + 6 + 15 + 28 = 50
//   y[1] = 1*2 + 2*4 + 3*6 + 4*8 = 2 + 8 + 18 + 32 = 60

`timescale 1ns / 1ps

module tb_dpe_vmm;

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
        .NUM_COLS(2),
        .DPE_BUF_WIDTH(8),    // 1 byte per strobe for simple test
        .COMPUTE_CYCLES(8)    // 8-cycle compute for simple test
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
    reg [31:0] expected [0:1];
    reg [31:0] got [0:1];
    integer errors;

    initial begin
        $dumpfile("tb_dpe_vmm.vcd");
        $dumpvars(0, tb_dpe_vmm);

        // Reset
        reset = 1; data_in = 0; nl_dpe_control = 0;
        shift_add_control = 0; w_buf_en = 0; shift_add_bypass = 0;
        load_output_reg = 0; load_input_reg = 0;
        weight_wen = 0; weight_data = 0;
        weight_row_addr = 0; weight_col_addr = 0;
        #20; reset = 0; #10;

        // ── Phase 1: Load weights ──
        // W[r][c]: row 0=[1,2], row 1=[3,4], row 2=[5,6], row 3=[7,8]
        $display("Loading weights...");
        load_weight(0, 0, 1); load_weight(0, 1, 2);
        load_weight(1, 0, 3); load_weight(1, 1, 4);
        load_weight(2, 0, 5); load_weight(2, 1, 6);
        load_weight(3, 0, 7); load_weight(3, 1, 8);
        @(posedge clk); weight_wen = 0;
        $display("Weights loaded.");

        // ── Phase 2: Feed input x = [1, 2, 3, 4] ──
        $display("Feeding input [1, 2, 3, 4]...");
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
            w_buf_en = 1;
            data_in = i + 1;  // x = [1, 2, 3, 4]
        end
        @(posedge clk); w_buf_en = 0;

        // Wait for reg_full
        wait(reg_full == 1);
        $display("T=%0t: reg_full asserted", $time);

        // ── Phase 3: Fire VMM ──
        @(posedge clk); nl_dpe_control = 2'b11;
        @(posedge clk); nl_dpe_control = 2'b00;

        // Wait for first output: data_out is set in S_COMPUTE→S_OUTPUT transition,
        // one cycle before dpe_done. Capture on each posedge after compute completes.
        // The DPE sets data_out[col 0] when entering S_OUTPUT,
        // then data_out[col 1] on the next cycle, etc.
        $display("Waiting for output...");
        wait(uut.state == 4);  // S_OUTPUT
        @(posedge clk); #1;
        got[0] = data_out[31:0];
        $display("  Column 0: got %0d", $signed(got[0]));
        @(posedge clk); #1;
        got[1] = data_out[31:0];
        $display("  Column 1: got %0d", $signed(got[1]));

        // ── Phase 5: Verify ──
        expected[0] = 50;  // 1*1 + 2*3 + 3*5 + 4*7
        expected[1] = 60;  // 1*2 + 2*4 + 3*6 + 4*8
        errors = 0;

        if ($signed(got[0]) !== $signed(expected[0])) begin
            $display("FAIL: col 0 expected %0d got %0d", $signed(expected[0]), $signed(got[0]));
            errors = errors + 1;
        end
        if ($signed(got[1]) !== $signed(expected[1])) begin
            $display("FAIL: col 1 expected %0d got %0d", $signed(expected[1]), $signed(got[1]));
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS: VMM output matches expected [%0d, %0d]",
                $signed(expected[0]), $signed(expected[1]));
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
