`timescale 1ns / 1ps

module tb_fc_512_128;

    parameter DATA_WIDTH = 40;

    reg clk, rst, valid, ready_n;
    reg [DATA_WIDTH-1:0] data_in;
    wire [DATA_WIDTH-1:0] data_out;
    wire ready, valid_n;

    // Set KERNEL_WIDTH for behavioral DPE stub
    // V=1: single DPE inside conv_layer_single_dpe
    defparam dut.fc_layer_inst.dpe_inst.KERNEL_WIDTH = 512;

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

    integer i;
    integer cycle_count;
    reg got_output;

    initial begin
        $dumpfile("tb_fc_512_128.vcd");
        $dumpvars(0, tb_fc_512_128);

        // Reset
        rst = 1; valid = 0; ready_n = 1; data_in = 0;  // ready_n=1: downstream ready
        #20;
        rst = 0;
        #10;

        // Feed K=512 input values
        $display("T=%0t: Starting input feed (K=512)", $time);
        for (i = 0; i < 512; i = i + 1) begin
            @(posedge clk);
            valid = 1;
            data_in = i + 1;  // simple test pattern: 1, 2, 3, ...
        end
        @(posedge clk);
        valid = 0;
        data_in = 0;

        $display("T=%0t: Input feed complete, waiting for output", $time);

        // Wait for output
        got_output = 0;
        cycle_count = 0;
        while (!got_output && cycle_count < 10000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
            if (valid_n) begin
                $display("T=%0t: Got output! data_out = %0d (cycle %0d)", $time, data_out, cycle_count);
                got_output = 1;
            end
        end

        if (!got_output)
            $display("TIMEOUT: No output after %0d cycles", cycle_count);
        else
            $display("PASS: Output received after %0d cycles", cycle_count);

        #100;
        $finish;
    end

    // Monitor valid_n transitions
    always @(posedge clk) begin
        if (valid_n && !rst)
            $display("T=%0t: valid_n asserted, data_out=%0d", $time, data_out);
    end

endmodule
