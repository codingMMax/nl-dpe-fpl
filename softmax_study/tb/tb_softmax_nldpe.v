// tb_softmax_nldpe.v -- functional + cycle TB for softmax_nldpe (safe softmax,
// NL-DPE: DPE(I|exp) per lane + shared DPE(I|log), log-domain output).
//
// Same protocol as tb_softmax_al.v, plus identity-weight forcing into every
// DPE instance (weights are sim-only; the DPE is a VTR black box). Pattern
// follows fc_verification/tb_fc.v:229-249.
//
// Defines: S_TB, NEXP_TB, SCORES_HEX, EXPECTED_HEX. (C_TB accepted, unused:
// port time scales with elements moved E = S/N_EXP, not with crossbar C.)
`timescale 1ns / 1ps

module tb_softmax_nldpe;
    localparam S    = `S_TB;
    localparam NEXP = `NEXP_TB;
    localparam E    = S / NEXP;
    localparam W    = 16;
    localparam RPL  = S / W;
    localparam NB   = RPL * S;
    localparam TOT  = W * NB;

    reg clk = 0, reset = 1, start = 0;
    wire done;
    reg         in_wen = 0;
    reg  [3:0]  in_lane = 0;
    reg  [13:0] in_addr = 0;
    reg  [7:0]  in_data = 0;
    reg  [3:0]  out_lane = 0;
    reg  [13:0] out_addr = 0;
    wire [7:0]  out_rdata;

    softmax_nldpe #(.S(S), .N_EXP(NEXP)) dut (
        .clk(clk), .reset(reset), .start(start), .done(done),
        .in_wen(in_wen), .in_lane(in_lane), .in_addr(in_addr),
        .in_data(in_data),
        .out_lane(out_lane), .out_addr(out_addr), .out_rdata(out_rdata)
    );

    always #5 clk = ~clk;

    // ── identity-weight forcing (sim-only; DPE has no weight port) ──
    reg do_force;
    initial do_force = 1'b0;
    genvar fk, fe;
    generate
        for (fk = 0; fk < W; fk = fk + 1) begin : F
            for (fe = 0; fe < NEXP; fe = fe + 1) begin : FE
                integer fr;
                always @(posedge do_force) begin
                    for (fr = 0; fr < E; fr = fr + 1)
                        dut.lane[fk].edpe[fe].u_exp.weights[fr][fr] = 8'sd1;
                end
            end
        end
    endgenerate
    integer lr;
    always @(posedge do_force) begin
        for (lr = 0; lr < 16; lr = lr + 1)
            dut.u_log.weights[lr][lr] = 8'sd1;
    end

    reg [7:0] scores   [0:TOT-1];
    reg [7:0] expected [0:TOT-1];

    integer k, a, cycles, errors;
    reg counting;

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

        do_force = 1'b1;
        @(negedge clk);
        do_force = 1'b0;
        @(negedge clk);

        // ── load all lanes, byte-serial, strictly sequential ──
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

    initial begin
        #200_000_000;
        $display("TB_FAIL timeout");
        $finish;
    end
endmodule
