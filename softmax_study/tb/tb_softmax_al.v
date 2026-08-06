// tb_softmax_al.v -- functional + cycle TB for softmax_al (safe softmax, AL).
//
// Loads a lane-major score file (SCORES_HEX), pulses start, counts cycles
// to done, reads back every output byte and compares against EXPECTED_HEX.
// Prints:  CYCLES=<n>   MISMATCH lane=.. addr=.. got=.. exp=..   TB_PASS/TB_FAIL
//
// Defines: S_TB (128|256), SCORES_HEX, EXPECTED_HEX.
`timescale 1ns / 1ps

module tb_softmax_al;
    localparam S   = `S_TB;
    localparam W   = 16;
    localparam RPL = S / W;           // rows per lane
    localparam NB  = RPL * S;         // bytes per lane
    localparam TOT = W * NB;          // total bytes

    reg clk = 0, reset = 1, start = 0;
    wire done;
    reg         in_wen = 0;
    reg  [3:0]  in_lane = 0;
    reg  [13:0] in_addr = 0;
    reg  [7:0]  in_data = 0;
    reg  [3:0]  out_lane = 0;
    reg  [13:0] out_addr = 0;
    wire [7:0]  out_rdata;

    softmax_al #(.S(S)) dut (
        .clk(clk), .reset(reset), .start(start), .done(done),
        .in_wen(in_wen), .in_lane(in_lane), .in_addr(in_addr),
        .in_data(in_data),
        .out_lane(out_lane), .out_addr(out_addr), .out_rdata(out_rdata)
    );

    always #5 clk = ~clk;

    reg [7:0] scores   [0:TOT-1];
    reg [7:0] expected [0:TOT-1];

    integer k, a, cycles, errors;
    reg counting;

    // cycle counter: from the first posedge with start==1 up to (not
    // including) the first posedge with done==1.
    always @(posedge clk) begin
        if (start) counting <= 1;
        if ((counting || start) && !done) cycles <= cycles + 1;
    end

    initial begin
        $readmemh(`SCORES_HEX, scores);
        $readmemh(`EXPECTED_HEX, expected);
        cycles = 0; errors = 0; counting = 0;

        repeat (4) @(negedge clk);
        reset = 0;
        repeat (2) @(negedge clk);

        // ── load all lanes, byte-serial ──
        for (k = 0; k < W; k = k + 1) begin
            for (a = 0; a < NB; a = a + 1) begin
                @(negedge clk);
                in_wen  = 1;
                in_lane = k[3:0];
                in_addr = a[13:0];
                in_data = scores[k * NB + a];
            end
        end
        @(negedge clk);
        in_wen = 0;

        // ── run ──
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        wait (done === 1'b1);
        @(negedge clk);
        $display("CYCLES=%0d", cycles);

        // ── readback ──
        for (k = 0; k < W; k = k + 1) begin
            for (a = 0; a < NB; a = a + 1) begin
                @(negedge clk);
                out_lane = k[3:0];
                out_addr = a[13:0];
                @(negedge clk);
                @(negedge clk);
                if (out_rdata !== expected[k * NB + a]) begin
                    if (errors < 10)
                        $display("MISMATCH lane=%0d addr=%0d got=%02x exp=%02x",
                                 k, a, out_rdata, expected[k * NB + a]);
                    errors = errors + 1;
                end
            end
        end

        if (errors == 0) $display("TB_PASS");
        else             $display("TB_FAIL errors=%0d", errors);
        $finish;
    end

    // watchdog
    initial begin
        #200_000_000;  // 20M cycles
        $display("TB_FAIL timeout");
        $finish;
    end
endmodule
