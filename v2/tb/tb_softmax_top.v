// ============================================================================
// tb_softmax_top.v — Stage-3 GATE-2 TB for the v2 softmax (`softmax_top`).
//
// Consumes case dirs produced by `v2/sim/softmax_sim.py::NldpeSoftmax.dump_case`
// (expanded by `v2/smoke/run_softmax_rtl.py` into <VDIR>/*.hex):
//
//   scores.hex  packed BUF-wide score words, ceil(S*S/BUF/8) lines
//   exrm.hex    S bytes      expected row_max
//   exin.hex    S*S bytes    expected exp_input (feed probe)
//   exout.hex   S*S bytes    expected exp_out   (ACAM_EXP probe)
//   exsum.hex   S int32      expected output_sum
//   exlq.hex    S bytes      expected log_input (lq)
//   exlg.hex    S bytes      expected log_output
//   expout.hex  S*S bytes    expected final result (row-major)
//
// Plusargs: +VDIR=<vectors dir> +CASE=<name> +EXPECTED=<used_cycles>
// Geometry: -DS_TB -DR_TB -DC_TB -DBUF_TB -DP_TB -DNE_TB -DNL_TB
//
// Gated: all six probe stages bit-exact, the final result read combinationally
// via `out_addr`/`data_out` (every element), and the measured cycles
// start->done vs +EXPECTED.
// ============================================================================

`timescale 1ns / 1ps

`ifndef S_TB
  `define S_TB 128
`endif
`ifndef R_TB
  `define R_TB 256
`endif
`ifndef C_TB
  `define C_TB 256
`endif
`ifndef BUF_TB
  `define BUF_TB 40
`endif
`ifndef P_TB
  `define P_TB 8
`endif
`ifndef NE_TB
  `define NE_TB 1
`endif
`ifndef NL_TB
  `define NL_TB 1
`endif

module tb_softmax_top;

    localparam S   = `S_TB;
    localparam R   = `R_TB;
    localparam C   = `C_TB;
    localparam BUF = `BUF_TB;
    localparam P   = `P_TB;
    localparam N_EXP = `NE_TB;
    localparam N_LOG = `NL_TB;

    localparam EPS = BUF / 8;
    localparam SQ  = S * S;
    localparam AW  = $clog2(SQ) + 1;
    localparam LOAD_WORDS = (SQ + EPS - 1) / EPS;

    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    reg reset;
    reg prog_start;
    reg [BUF-1:0] score_in;
    reg score_en;
    reg start;
    reg [AW-1:0] out_addr;

    wire prog_ready, load_ready, busy, done;
    wire [7:0] data_out;

    softmax_top #(
        .S(S), .R(R), .C(C), .BUF(BUF), .P(P), .N_EXP(N_EXP), .N_LOG(N_LOG)
    ) dut (
        .clk(clk), .reset(reset),
        .prog_start(prog_start), .prog_ready(prog_ready),
        .score_in(score_in), .score_en(score_en), .load_ready(load_ready),
        .start(start), .busy(busy),
        .out_addr(out_addr), .data_out(data_out), .done(done)
    );

    reg [BUF-1:0] scores_mem [0:LOAD_WORDS-1];
    reg [7:0] exp_rm [0:S-1];
    reg [7:0] exp_in [0:SQ-1];
    reg [7:0] exp_out [0:SQ-1];
    reg [31:0] exp_sum [0:S-1];
    reg [7:0] exp_lq [0:S-1];
    reg [7:0] exp_lg [0:S-1];
    reg [7:0] exp_stream [0:SQ-1];

    integer cycle_count;
    always @(posedge clk) cycle_count <= cycle_count + 1;

    integer t_start, t_done, c_words;
    reg saw_done;
    integer errors, guard, measured, expected, delta;
    integer i;
    reg [1023:0] vdir, case_name;
    reg [2047:0] f_scores, f_rm, f_in, f_out, f_sum, f_lq, f_lg, f_stream;

    always @(negedge clk) begin
        if (done) saw_done <= 1'b1;
        if (start && t_start < 0) t_start = cycle_count;
        if (done && t_done < 0) t_done = cycle_count;
    end

    initial begin
        reset = 1;
        prog_start = 0;
        score_in = 0;
        score_en = 0;
        start = 0;
        cycle_count = 0;
        t_start = -1;
        t_done = -1;
        c_words = 0;
        errors = 0;
        saw_done = 1'b0;
        measured = 0;
        expected = 0;
        delta = 0;

        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("VDIR=%s", vdir)) begin
            $display("[tb_softmax_top] FAIL %0s: missing +VDIR", case_name);
            $finish;
        end
        if (!$value$plusargs("EXPECTED=%d", expected)) expected = 0;

        $sformat(f_scores, "%0s/scores.hex", vdir);
        $sformat(f_rm, "%0s/exrm.hex", vdir);
        $sformat(f_in, "%0s/exin.hex", vdir);
        $sformat(f_out, "%0s/exout.hex", vdir);
        $sformat(f_sum, "%0s/exsum.hex", vdir);
        $sformat(f_lq, "%0s/exlq.hex", vdir);
        $sformat(f_lg, "%0s/exlg.hex", vdir);
        $sformat(f_stream, "%0s/expout.hex", vdir);
        $readmemh(f_scores, scores_mem);
        $readmemh(f_rm, exp_rm);
        $readmemh(f_in, exp_in);
        $readmemh(f_out, exp_out);
        $readmemh(f_sum, exp_sum);
        $readmemh(f_lq, exp_lq);
        $readmemh(f_lg, exp_lg);
        $readmemh(f_stream, exp_stream);

        $display("[tb_softmax_top] case=%0s S=%0d R=%0d C=%0d BUF=%0d P=%0d N_EXP=%0d N_LOG=%0d LOAD_WORDS=%0d",
                 case_name, S, R, C, BUF, P, N_EXP, N_LOG, LOAD_WORDS);

        repeat (3) @(posedge clk);
        #1 reset = 0;
        @(posedge clk); #1;

        // ---- weight preload (always-on): skip the R*C broadcast ---------
        // The generate blocks below write each dpe's weights/mode_q right
        // after reset. prog_ready is forced so the score path's load_ready
        // rises (the top gates score latching on prog_ready).
        force dut.u_wprog.prog_ready = 1'b1;
        @(posedge clk); #1;

        for (i = 0; i < LOAD_WORDS; i = i + 1) begin
            score_in = scores_mem[i];
            score_en = 1;
            @(posedge clk); #1;
        end
        score_en = 0;
        score_in = 0;

        start = 1;
        @(posedge clk); #1;
        start = 0;

        guard = 0;
        while (!done && guard < 20*SQ + 200000) begin
            @(posedge clk); #1;
            guard = guard + 1;
        end
        // `done` = whole result computed; the final result is read
        // combinationally (no drain). Sweep every element and compare.
        c_words = 0;
        for (i = 0; i < SQ; i = i + 1) begin
            out_addr = i;
            #1;
            if (data_out !== exp_stream[i]) errors = errors + 1;
            c_words = c_words + 1;
        end
        @(posedge clk); #1;
        if (!saw_done) begin
            $display("[tb_softmax_top]   ERROR: done never rose");
            errors = errors + 1;
        end

        for (i = 0; i < S; i = i + 1) begin
            if (dut.row_max_q[i] !== exp_rm[i]) errors = errors + 1;
            if (dut.sum_q[i] !== exp_sum[i]) errors = errors + 1;
            if (dut.lq_q[i] !== exp_lq[i]) errors = errors + 1;
            if (dut.log_out_q[i] !== exp_lg[i]) errors = errors + 1;
        end
        for (i = 0; i < SQ; i = i + 1) begin
            if (dut.exp_in_q[i] !== exp_in[i]) errors = errors + 1;
            if (dut.exp_out_q[i] !== exp_out[i]) errors = errors + 1;
        end

        measured = (t_start >= 0 && t_done >= 0) ? (t_done - t_start) : -1;
        delta = measured - expected;
        if (errors == 0 && delta == 0)
            $display("[tb_softmax_top] PASS %0s errors=0 measured=%0d expected=%0d delta=%0d words=%0d",
                     case_name, measured, expected, delta, c_words);
        else
            $display("[tb_softmax_top] FAIL %0s errors=%0d measured=%0d expected=%0d delta=%0d words=%0d",
                     case_name, errors, measured, expected, delta, c_words);
        $finish;
    end

    // ---- weight preload: identity-eye weights + mode into each dpe -------
    // (verification-only; replaces the R*C cycle broadcast at sim time 0)
    genvar wg;
    generate
        for (wg = 0; wg < N_EXP; wg = wg + 1) begin : wp_exp
            integer wi, wr, wc;
            initial begin
                @(negedge reset); #1;
                for (wi = 0; wi < R*C; wi = wi + 1) begin
                    wr = wi / C; wc = wi % C;
                    dut.gen_exp[wg].u_feed.u_dpe.weights[wi] =
                        (wr == wc) ? 8'sd1 : 8'sd0;
                end
                dut.gen_exp[wg].u_feed.u_dpe.mode_q = 2'b10;  // MODE_EXP
            end
        end
        for (wg = 0; wg < N_LOG; wg = wg + 1) begin : wp_log
            integer wi, wr, wc;
            initial begin
                @(negedge reset); #1;
                for (wi = 0; wi < R*C; wi = wi + 1) begin
                    wr = wi / C; wc = wi % C;
                    dut.gen_log[wg].u_log.u_dpe.weights[wi] =
                        (wr == wc) ? 8'sd1 : 8'sd0;
                end
                dut.gen_log[wg].u_log.u_dpe.mode_q = 2'b11;   // MODE_LOG
            end
        end
    endgenerate

endmodule
