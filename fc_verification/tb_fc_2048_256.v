`timescale 1ns / 1ps

module tb_fc_2048_256;

    parameter DATA_WIDTH = 40;

    reg clk, rst, valid, ready_n;
    reg [DATA_WIDTH-1:0] data_in;
    wire [DATA_WIDTH-1:0] data_out;
    wire ready, valid_n;

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

    initial clk = 0;
    always #5 clk = ~clk;

    integer i, cycle_count;
    reg got_output;

    initial begin
        $dumpfile("tb_fc_2048_256.vcd");
        $dumpvars(0, tb_fc_2048_256);

        rst = 1; valid = 0; ready_n = 1; data_in = 0;  // ready_n=1: downstream ready
        #20; rst = 0; #10;

        // Feed K=2048 input values
        $display("T=%0t: Starting input feed (K=2048, V=2 parallel rows)", $time);
        for (i = 0; i < 2048; i = i + 1) begin
            @(posedge clk);
            valid = 1;
            data_in = i + 1;
        end
        @(posedge clk);
        valid = 0; data_in = 0;

        $display("T=%0t: Input feed complete", $time);

        got_output = 0;
        cycle_count = 0;
        while (!got_output && cycle_count < 20000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
            if (valid_n) begin
                $display("T=%0t: Output! data_out=%0d (cycle %0d)", $time, data_out, cycle_count);
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

endmodule
