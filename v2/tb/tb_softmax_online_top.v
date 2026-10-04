// ============================================================================
// tb_softmax_online_top.v — GATE-2 TB for the v2 online softmax.
//
// Consumes case dirs from `v2/sim/softmax_online_sim.py::dump_case` (expanded
// by `run_softmax_online_rtl.py` into <VDIR>/*.hex):
//   scores.hex    block-major BUF-wide score stream words
//   exblkmax.hex  B*S bytes
//   exblksp.hex   B*S int32   (per-block partials, crossbar-aggregated)
//   exfactor.hex  B*S int32
//   exL.hex       S int32
//   exlq.hex      S bytes
//   exls.hex      S bytes
//   exout.hex     S*S bytes   (row-major output stream)
//
// Plusargs: +VDIR +CASE +EXPECTED (compute_cycles)
// Geometry: -DS_TB -DR_TB -DC_TB -DBUF_TB -DP_TB -DBKV_TB -DNE_TB -DNL_TB
//           -DNF_TB (factor bank width; 0 => N_EXP)
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
`ifndef BKV_TB
  `define BKV_TB 16
`endif
`ifndef NE_TB
  `define NE_TB 1
`endif
`ifndef NL_TB
  `define NL_TB 1
`endif
`ifndef NF_TB
  `define NF_TB 0
`endif

module tb_softmax_online_top;

    localparam S    = `S_TB;
    localparam R    = `R_TB;
    localparam C    = `C_TB;
    localparam BUF  = `BUF_TB;
    localparam P    = `P_TB;
    localparam BKV  = `BKV_TB;
    localparam N_EXP = `NE_TB;
    localparam N_LOG = `NL_TB;
    localparam N_FAC = (`NF_TB == 0) ? N_EXP : `NF_TB;

    localparam EPS = BUF / 8;
    localparam B   = S / BKV;
    localparam SQ  = S * S;
    localparam NROWS = B * S;
    localparam AW  = $clog2(SQ) + 1;
    localparam STREAM = (S * BKV + EPS - 1) / EPS;
    localparam LOAD_WORDS = B * STREAM;

    reg clk;
    initial begin clk = 0; forever #5 clk = ~clk; end

    reg reset, prog_start, score_en, start;
    reg [BUF-1:0] score_in;
    reg [AW-1:0] out_addr;
    wire prog_ready, busy, done;
    wire [7:0] data_out;

    softmax_online_top #(
        .S(S), .R(R), .C(C), .BUF(BUF), .P(P), .BKV(BKV),
        .N_EXP(N_EXP), .N_LOG(N_LOG), .N_FAC(N_FAC)
    ) dut (
        .clk(clk), .reset(reset),
        .prog_start(prog_start), .prog_ready(prog_ready),
        .score_in(score_in), .score_en(score_en),
        .start(start), .busy(busy),
        .out_addr(out_addr), .data_out(data_out), .done(done)
    );

    reg [BUF-1:0] scores_mem [0:LOAD_WORDS-1];
    reg [7:0]  exp_bmax [0:NROWS-1];
    reg [31:0] exp_bsp  [0:NROWS-1];
    reg [31:0] exp_fac  [0:NROWS-1];
    reg [31:0] exp_L    [0:S-1];
    reg [7:0]  exp_lq   [0:S-1];
    reg [7:0]  exp_ls   [0:S-1];
    reg [7:0]  exp_out  [0:SQ-1];

    integer cycle_count;
    always @(posedge clk) cycle_count <= cycle_count + 1;

    integer t_start, t_done, c_words, n_drv, n_done1;
    reg saw_done;
    integer errors, guard, measured, expected, delta, i;
    integer e_bmax, e_bsp, e_fac, e_L, e_lq, e_ls, e_out;
    integer max_add0, max_cnt0, max_drb;
    reg [1023:0] vdir, case_name;
    reg [2047:0] f_scores, f_bmax, f_bsp, f_fac, f_L, f_lq, f_ls, f_out;

    always @(negedge clk) begin
        if (done) saw_done <= 1'b1;
        if (start && t_start < 0) t_start = cycle_count;
        if (done && t_done < 0) t_done = cycle_count;
        if (dut.exp_dr_valid[0]) n_drv = n_drv + 1;
        if (dut.u_sum.add_cnt[0] > max_add0) max_add0 = dut.u_sum.add_cnt[0];
        if (dut.u_sum.count[0] > max_cnt0) max_cnt0 = dut.u_sum.count[0];
        if (dut.exp_dr_base[0 +: 16] > max_drb) max_drb = dut.exp_dr_base[0 +: 16];
        if (dut.gen_exp[0].u_feed.dpe_reg_full) n_done1 = n_done1 + 1;
    end

    initial begin
        reset = 1; prog_start = 0; score_in = 0; score_en = 0; start = 0;
        cycle_count = 0; t_start = -1; t_done = -1; c_words = 0; errors = 0;
        saw_done = 0; measured = 0; expected = 0; delta = 0;
        e_bmax=0; e_bsp=0; e_fac=0; e_L=0; e_lq=0; e_ls=0; e_out=0;
        n_drv=0; n_done1=0; max_add0=0; max_cnt0=0; max_drb=0;

        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("VDIR=%s", vdir)) begin
            $display("[tb_softmax_online] FAIL %0s: missing +VDIR", case_name);
            $finish;
        end
        if (!$value$plusargs("EXPECTED=%d", expected)) expected = 0;

        $sformat(f_scores, "%0s/scores.hex", vdir);
        $sformat(f_bmax, "%0s/exblkmax.hex", vdir);
        $sformat(f_bsp,  "%0s/exblksp.hex", vdir);
        $sformat(f_fac,  "%0s/exfactor.hex", vdir);
        $sformat(f_L,    "%0s/exL.hex", vdir);
        $sformat(f_lq,   "%0s/exlq.hex", vdir);
        $sformat(f_ls,   "%0s/exls.hex", vdir);
        $sformat(f_out,  "%0s/exout.hex", vdir);
        $readmemh(f_scores, scores_mem);
        $readmemh(f_bmax, exp_bmax);
        $readmemh(f_bsp,  exp_bsp);
        $readmemh(f_fac,  exp_fac);
        $readmemh(f_L,    exp_L);
        $readmemh(f_lq,   exp_lq);
        $readmemh(f_ls,   exp_ls);
        $readmemh(f_out,  exp_out);

        $display("[tb_softmax_online] case=%0s S=%0d R=%0d C=%0d BUF=%0d P=%0d BKV=%0d N_EXP=%0d LOAD_WORDS=%0d",
                 case_name, S, R, C, BUF, P, BKV, N_EXP, LOAD_WORDS);

        repeat (3) @(posedge clk);
        #1 reset = 0;
        @(posedge clk); #1;

        // ---- weight preload (always-on): skip the R*C broadcast ---------
        // The generate/initial blocks below write each dpe's weights/mode_q
        // right after reset.
        force dut.u_wprog.prog_ready = 1'b1;
        @(posedge clk); #1;

        // start, then STREAM the block-major score words
        start = 1;
        @(posedge clk); #1;
        start = 0;
        for (i = 0; i < LOAD_WORDS; i = i + 1) begin
            score_in = scores_mem[i];
            score_en = 1;
            @(posedge clk); #1;
        end
        score_en = 0;
        score_in = 0;

        guard = 0;
        while (!done && guard < 40*SQ + 400000) begin
            @(posedge clk); #1; guard = guard + 1;
        end
        // `done` = whole result computed; the final result is read
        // combinationally (no drain). Sweep every element and compare.
        @(posedge clk); #1;
        if (!saw_done) begin
            $display("[tb_softmax_online]   t_start=%0d t_done=%0d", t_start, t_done);
            $display("[tb_softmax_online]   ERROR: done never rose");
            $display("  DBG started=%b blk_ready=%b iss0=%0d allexp=%b facd=%b logd=%b dcout=%0d lq0=%0d done=%b",
                     dut.started, dut.blk_ready, dut.exp_iss[0],
                     dut.all_exp_done, dut.fac_win_done, dut.log_win_done,
                     dut.exp_done_cnt, dut.lq_q[0], dut.done);
            errors = errors + 1;
        end

        for (i = 0; i < NROWS; i = i + 1) begin
            if (dut.blk_max_q[i] !== exp_bmax[i]) e_bmax = e_bmax + 1;
            if (dut.sp_q[i]      !== exp_bsp[i]) e_bsp = e_bsp + 1;
            if (dut.factor_q[i]  !== exp_fac[i]) e_fac = e_fac + 1;
        end
        for (i = 0; i < S; i = i + 1) begin
            if (dut.L_q[i]  !== exp_L[i])  e_L = e_L + 1;
            if (dut.lq_q[i] !== exp_lq[i]) e_lq = e_lq + 1;
            if (dut.ls_q[i] !== exp_ls[i]) e_ls = e_ls + 1;
        end
        c_words = 0;
        for (i = 0; i < SQ; i = i + 1) begin
            out_addr = i;
            #1;
            if (data_out !== exp_out[i]) e_out = e_out + 1;
            c_words = c_words + 1;
        end
        errors = e_bmax + e_bsp + e_fac + e_L + e_lq + e_ls + e_out;
        if (errors != 0)
            $display("  DBG err bmax=%0d bsp=%0d fac=%0d L=%0d lq=%0d ls=%0d out=%0d | bm0 e=%0d g=%0d | sp0 e=%0d g=%0d | L0 e=%0d g=%0d | out0 e=%0d g=%0d",
                     e_bmax, e_bsp, e_fac, e_L, e_lq, e_ls, e_out,
                     exp_bmax[0], dut.blk_max_q[0],
                     exp_bsp[0], dut.sp_q[0],
                     exp_L[0], dut.L_q[0],
                     exp_out[0], data_out);

        measured = (t_start >= 0 && t_done >= 0) ? (t_done - t_start) : -1;
        delta = measured - expected;
        if (errors == 0 && delta == 0) begin
            $display("[tb_softmax_online]   t_start=%0d t_done=%0d", t_start, t_done);
            $display("[tb_softmax_online] PASS %0s errors=0 measured=%0d expected=%0d delta=%0d words=%0d",
                     case_name, measured, expected, delta, c_words);
        end
        else begin
            $display("[tb_softmax_online]   t_start=%0d t_done=%0d", t_start, t_done);
            $display("[tb_softmax_online] FAIL %0s errors=%0d measured=%0d expected=%0d delta=%0d words=%0d",
                     case_name, errors, measured, expected, delta, c_words);
        end
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
    endgenerate
    integer fwi, fwr, fwc;
    genvar wf;
    generate
        for (wf = 0; wf < N_FAC; wf = wf + 1) begin : wp_fac
            initial begin  // online factor pass (ACAM_EXP)
                @(negedge reset); #1;
                for (fwi = 0; fwi < R*C; fwi = fwi + 1) begin
                    fwr = fwi / C; fwc = fwi % C;
                    dut.gen_fac[wf].u_fac.u_dpe.weights[fwi] =
                        (fwr == fwc) ? 8'sd1 : 8'sd0;
                end
                dut.gen_fac[wf].u_fac.u_dpe.mode_q = 2'b10;
            end
        end
    endgenerate
    integer lwi, lwr, lwc;
    initial begin  // online LOG pass
        @(negedge reset); #1;
        for (lwi = 0; lwi < R*C; lwi = lwi + 1) begin
            lwr = lwi / C; lwc = lwi % C;
            dut.u_log.u_dpe.weights[lwi] = (lwr == lwc) ? 8'sd1 : 8'sd0;
        end
        dut.u_log.u_dpe.mode_q = 2'b11;
    end

endmodule
