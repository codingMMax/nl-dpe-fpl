// ============================================================================
// tb_att_head.v — gate-H testbench of `att_head` (v3, top/spec.md §8). Case files are written by top/cases.py (model
// run + GATE 1); the runner top/tb/run_head_rtl.py compares this testbench's event lines with the model's.
//
// The testbench is the head workload's environment (sim/workloads/attention_head.py):
//   X sources    every copy's token bursts at the cycles the model planned (x_rdy must be high in the first cycle)
//   start        the start pulse at its planned cycle
//   O sink       takes every E9 window (o_win.txt: one window per epilogue wave, LW words of N_OC x 40 bits),
//                checks every value, no gap inside a window
// Logged (EV lines, the model's format) on every edge of the netlist (spec §4): windows by their first word, numbered
// by the stream rule (channel ch's k-th window is window ch + L k of the stream), and credits / pulses: the GEMMs'
// MSB_SA_Ready rising, x_rdy rising (after the first), the staging and stream credits, the job takes and j_rdy rising,
// the softmax starts and busy falling, run_done, the head's busy falling. Also checked: done at the model's cycle,
// the word counts, a watchdog.
// Weights: preloaded at zero time after reset (PLAN R8): the projections from wq / wk / wv.mem, identity elsewhere.
// Cycle 0 is the first cycle after reset.
// ============================================================================
`timescale 1ns / 1ps

`ifndef NSEQ_TB
  `define NSEQ_TB 32
`endif
`ifndef DM_TB
  `define DM_TB 64
`endif
`ifndef D_TB
  `define D_TB 16
`endif
`ifndef BKV_TB
  `define BKV_TB 16
`endif
`ifndef R_TB
  `define R_TB 64
`endif
`ifndef C_TB
  `define C_TB 64
`endif
`ifndef NEXP_TB
  `define NEXP_TB 2
`endif
`ifndef NE_TB
  `define NE_TB 32
`endif
`ifndef NEP_TB
  `define NEP_TB 32
`endif
`ifndef NOC_TB
  `define NOC_TB 2
`endif
`ifndef PP_TB
  `define PP_TB 1
`endif
`ifndef XS_TB
  `define XS_TB 16
`endif

// a window monitor: L channels, WORDS gapless words per window; channel ch's k-th window is window ch + L k of the
// stream, i.e. window (ch + L k) mod WPU of unit (ch + L k) / WPU. Prints `EV <t> <PRE><IDX><POST> win ch u w`.
module tb_mon_win #(parameter L = 1, parameter WORDS = 1, parameter WPU = 1, parameter PRE = "x", parameter IDX = -1,
                    parameter POST = "") (
    input wire clk, input wire run, input wire [31:0] tc, input wire [L-1:0] v);
    integer wd [0:L-1], k [0:L-1], i, idx;
    initial for (i = 0; i < L; i = i + 1) begin wd[i] = 0; k[i] = 0; end
    always @(posedge clk) begin
        #2;
        if (run)
            for (i = 0; i < L; i = i + 1)
                if (v[i]) begin
                    if (wd[i] == 0) begin
                        idx = i + L * k[i];
                        if (IDX >= 0) $display("EV %0d %0s%0d%0s win ch%0d u%0d w%0d", tc, PRE, IDX, POST, i,
                                               idx / WPU, idx % WPU);
                        else $display("EV %0d %0s%0s win ch%0d u%0d w%0d", tc, PRE, POST, i, idx / WPU, idx % WPU);
                    end
                    wd[i] = wd[i] + 1;
                    if (wd[i] == WORDS) begin wd[i] = 0; k[i] = k[i] + 1; end
                end
                else if (wd[i] != 0) begin
                    $display("[tb_head] ERROR: gap inside a window of %0s%0s ch%0d at %0d", PRE, POST, i, tc);
                    tb_att_head.errors = tb_att_head.errors + 1;
                end
    end
endmodule

// a credit / pulse monitor: MODE 0 = every cycle p is high, 1 = p rising, 2 = p falling; the first SKIP events are
// not credits. Prints `EV <t> <PRE><IDX><POST> cr u<n>`.
module tb_mon_cr #(parameter MODE = 0, parameter SKIP = 0, parameter PRE = "x", parameter IDX = -1,
                   parameter POST = "") (
    input wire clk, input wire run, input wire [31:0] tc, input wire p);
    integer n = 0, seen = 0;
    reg     prev = 1'b0, first = 1'b1;
    always @(posedge clk) begin
        #2;
        if (run) begin
            if ((MODE == 0 && p) || (!first && MODE == 1 && p && !prev) || (!first && MODE == 2 && !p && prev)) begin
                if (seen >= SKIP) begin
                    if (IDX >= 0) $display("EV %0d %0s%0d%0s cr u%0d", tc, PRE, IDX, POST, n);
                    else $display("EV %0d %0s%0s cr u%0d", tc, PRE, POST, n);
                    n = n + 1;
                end
                seen = seen + 1;
            end
            prev = p;
            first = 1'b0;
        end
    end
endmodule

module tb_att_head;
    localparam NSEQ = `NSEQ_TB, DM = `DM_TB, D = `D_TB, BKV = `BKV_TB, R = `R_TB, C = `C_TB, BUF = 40, P = 8;
    localparam N_EXP = `NEXP_TB, N_E = `NE_TB, N_EP = `NEP_TB, N_OC = `NOC_TB, P_PROJ = `PP_TB, X_S = `XS_TB;
    localparam I     = (R < C) ? R : C;
    localparam S     = (NSEQ < I) ? NSEQ : I;
    localparam RB    = NSEQ / S;
    localparam NBLK  = NSEQ / BKV;
    localparam V     = (DM + R - 1) / R;
    localparam ROWSX = (V == 1) ? DM : R;
    localparam LOADX = (ROWSX * 8 + BUF - 1) / BUF;
    localparam NGQ   = N_E / D;
    localparam NGP   = N_EP / BKV;
    localparam LW    = (I * 8 + BUF - 1) / BUF;
    localparam NWO   = (D + 4) / 5;
    localparam NWB   = (S * BKV) / I;                   // QK windows per job = softmax windows per block
    localparam WKP   = (S * D) / I;                     // PV windows per job
    localparam P_OC  = (S * D) / I;                     // accumulator windows per run
    localparam W9    = (P_OC + N_OC - 1) / N_OC;        // E9 windows (epilogue waves) per run
    localparam MAXO  = 1024;
    localparam XSQ   = (X_S < N_E) ? X_S : N_E, NSQ = N_E / XSQ;
    localparam XSP   = (X_S < N_EP) ? X_S : N_EP, NSP = N_EP / XSP;
    localparam MAXB  = 4096;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg reset = 1'b1;

    reg                     start = 1'b0;
    reg  [P_PROJ-1:0]       x_v = 0;
    reg  [P_PROJ*V*BUF-1:0] x_d = 0;
    wire                    busy, done, o_v;
    wire [P_PROJ-1:0]       x_rdy;
    wire [N_OC*BUF-1:0]     o_d;

    att_head #(.NSEQ(NSEQ), .DM(DM), .D(D), .BKV(BKV), .R(R), .C(C), .BUF(BUF), .P(P), .N_EXP(N_EXP), .N_E(N_E),
                .N_EP(N_EP), .N_OC(N_OC), .P_PROJ(P_PROJ), .X_S(X_S)) dut (
        .clk(clk), .reset(reset), .start(start), .busy(busy), .done(done), .x_rdy(x_rdy), .x_v(x_v), .x_d(x_d),
        .o_v(o_v), .o_d(o_d));

    // ---- case files --------------------------------------------------------------------------------------------
    reg [8*256-1:0] vdir, fn;
    integer fd, rc, i, k, nbur, tmax, t_start, t_done, nowin, ot_i, ou_i, ow_i, bad_o = 0;
    reg [7:0] wq_m [0:DM*D-1], wk_m [0:DM*D-1], wv_m [0:DM*D-1];
    integer bt [0:MAXB-1], bc [0:MAXB-1], bk [0:MAXB-1];
    reg [BUF-1:0] bd [0:MAXB*V*LOADX-1];
    reg [N_OC*BUF-1:0] ox [0:MAXO*LW-1];
    reg [BUF-1:0] word;
    reg [N_OC*BUF-1:0] oword;
    task load;
        begin
            $sformat(fn, "%0s/wq.mem", vdir); $readmemh(fn, wq_m);
            $sformat(fn, "%0s/wk.mem", vdir); $readmemh(fn, wk_m);
            $sformat(fn, "%0s/wv.mem", vdir); $readmemh(fn, wv_m);
            $sformat(fn, "%0s/x_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nbur; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d", bt[i], bc[i], bk[i]);
                for (k = 0; k < V * LOADX; k = k + 1) begin rc = $fscanf(fd, "%h", word); bd[i*V*LOADX + k] = word; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/o_win.txt", vdir);             // per line: t run wave, then LW words
            fd = $fopen(fn, "r");
            for (i = 0; i < nowin; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d", ot_i, ou_i, ow_i);
                if (ou_i != i / W9 || ow_i != i % W9) bad_o = bad_o + 1;     // windows in port order
                for (k = 0; k < LW; k = k + 1) begin rc = $fscanf(fd, "%h", oword); ox[i*LW + k] = oword; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/start.txt", vdir);
            fd = $fopen(fn, "r"); rc = $fscanf(fd, "%d %d", t_start, t_done); $fclose(fd);
        end
    endtask

    // ---- weights: zero-time preload after reset (PLAN R8) --------------------------------------------------------
    event preload;
    genvar gc, gv, gs, gf;
    generate
        for (gc = 0; gc < P_PROJ; gc = gc + 1) begin : w_cp
            for (gv = 0; gv < V; gv = gv + 1) begin : w_v                // GEMM tile v: rows v R .. of W
                integer wk, row, col;
                always @(preload) begin
                    for (wk = 0; wk < R*C; wk = wk + 1) begin
                        row = gv * R + wk / C;
                        col = wk % C;
                        dut.g_cp[gc].u_gq.gen_v[gv].gen_h[0].u_dpe.weights[wk] = (row < DM && col < D) ? wq_m[row*D + col] : 8'd0;
                        dut.g_cp[gc].u_gk.gen_v[gv].gen_h[0].u_dpe.weights[wk] = (row < DM && col < D) ? wk_m[row*D + col] : 8'd0;
                        dut.g_cp[gc].u_gv.gen_v[gv].gen_h[0].u_dpe.weights[wk] = (row < DM && col < D) ? wv_m[row*D + col] : 8'd0;
                    end
                    dut.g_cp[gc].u_gq.gen_v[gv].gen_h[0].u_dpe.mode_q = 2'b00;  // REGULAR
                    dut.g_cp[gc].u_gk.gen_v[gv].gen_h[0].u_dpe.mode_q = 2'b00;
                    dut.g_cp[gc].u_gv.gen_v[gv].gen_h[0].u_dpe.mode_q = 2'b00;
                end
            end
            integer wq;
            always @(preload) begin                                     // the DIMMs' LOG converters of copy c
                for (wq = 0; wq < R*C; wq = wq + 1) begin
                    dut.u_qk.g_sa[gc].g_conv.u_conv.u_dpe.weights[wq] = ((wq / C) == (wq % C)) ? 8'sd1 : 8'sd0;
                    dut.u_qk.g_sb[gc].g_conv.u_conv.u_dpe.weights[wq] = ((wq / C) == (wq % C)) ? 8'sd1 : 8'sd0;
                    dut.u_pv.g_sb[gc].g_conv.u_conv.u_dpe.weights[wq] = ((wq / C) == (wq % C)) ? 8'sd1 : 8'sd0;
                end
                dut.u_qk.g_sa[gc].g_conv.u_conv.u_dpe.mode_q = 2'b11;       // LOG
                dut.u_qk.g_sb[gc].g_conv.u_conv.u_dpe.mode_q = 2'b11;
                dut.u_pv.g_sb[gc].g_conv.u_conv.u_dpe.mode_q = 2'b11;
            end
        end
        for (gs = 0; gs < NSQ; gs = gs + 1) begin : w_qs                  // the QK farm
            for (gf = 0; gf < XSQ; gf = gf + 1) begin : w_x
                integer wk;
                always @(preload) begin
                    for (wk = 0; wk < R*C; wk = wk + 1)
                        dut.u_qk.g_sl[gs].u_slice.g_x[gf].u_dpe.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                    dut.u_qk.g_sl[gs].u_slice.g_x[gf].u_dpe.mode_q = 2'b10;  // EXP
                end
            end
        end
        for (gs = 0; gs < NSP; gs = gs + 1) begin : w_ps                  // the PV farm
            for (gf = 0; gf < XSP; gf = gf + 1) begin : w_x
                integer wk;
                always @(preload) begin
                    for (wk = 0; wk < R*C; wk = wk + 1)
                        dut.u_pv.g_sl[gs].u_slice.g_x[gf].u_dpe.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                    dut.u_pv.g_sl[gs].u_slice.g_x[gf].u_dpe.mode_q = 2'b10;  // EXP
                end
            end
        end
        for (gf = 0; gf < N_EXP; gf = gf + 1) begin : w_se               // the softmax's EXP crossbars
            integer wk;
            always @(preload) begin
                for (wk = 0; wk < R*C; wk = wk + 1)
                    dut.u_sm.gen_exp[gf].u_feed.u_x.u_dpe.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                dut.u_sm.gen_exp[gf].u_feed.u_x.u_dpe.mode_q = 2'b10;
            end
        end
        for (gf = 0; gf < N_OC; gf = gf + 1) begin : w_ac                // the accumulator's groups
            integer wk;
            always @(preload) begin
                for (wk = 0; wk < R*C; wk = wk + 1) begin
                    dut.u_acc.g_x[gf].u_lg.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                    dut.u_acc.g_x[gf].u_ex.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                end
                dut.u_acc.g_x[gf].u_lg.mode_q = 2'b11;                     // LOG
                dut.u_acc.g_x[gf].u_ex.mode_q = 2'b10;                     // EXP
            end
        end
    endgenerate
    integer ck;
    always @(preload) begin                                             // the softmax's carry crossbars
        for (ck = 0; ck < R*C; ck = ck + 1) begin
            dut.u_sm.u_carry.u_cexp.u_x.u_dpe.weights[ck] = ((ck / C) == (ck % C)) ? 8'sd1 : 8'sd0;
            dut.u_sm.u_carry.u_clog.u_x.u_dpe.weights[ck] = ((ck / C) == (ck % C)) ? 8'sd1 : 8'sd0;
        end
        dut.u_sm.u_carry.u_cexp.u_x.u_dpe.mode_q = 2'b10;
        dut.u_sm.u_carry.u_clog.u_x.u_dpe.mode_q = 2'b11;
    end

    // ---- cycle counter, errors, monitors -------------------------------------------------------------------------
    integer tc = 0;
    always @(posedge clk) if (!reset) tc <= tc + 1;
    integer errors = 0, err_o = 0, err_x = 0;            // all errors; O8 value and x_rdy errors among them
    reg     running = 1'b0;
    wire [31:0] tcw = tc;
    generate
        for (gc = 0; gc < P_PROJ; gc = gc + 1) begin : m_cp
            tb_mon_cr  #(.MODE(1), .PRE("gq"), .IDX(gc), .POST(".act")) mq (clk, running, tcw, dut.gq_rdy[gc]);
            tb_mon_cr  #(.MODE(1), .PRE("gk"), .IDX(gc), .POST(".act")) mk (clk, running, tcw, dut.gk_rdy[gc]);
            tb_mon_cr  #(.MODE(1), .PRE("gv"), .IDX(gc), .POST(".act")) mv (clk, running, tcw, dut.gv_rdy[gc]);
            tb_mon_cr  #(.MODE(1), .SKIP(1), .PRE("ctrl.x"), .IDX(gc)) mx (clk, running, tcw, x_rdy[gc]);
            tb_mon_win #(.WORDS(NWO), .PRE("gq"), .IDX(gc), .POST(".out")) oq (clk, running, tcw, dut.gq_v[gc]);
            tb_mon_win #(.WORDS(NWO), .PRE("gk"), .IDX(gc), .POST(".out")) ok (clk, running, tcw, dut.gk_v[gc]);
            tb_mon_win #(.WORDS(NWO), .PRE("gv"), .IDX(gc), .POST(".out")) ov (clk, running, tcw, dut.gv_v[gc]);
        end
    endgenerate
    tb_mon_cr  #(.PRE("qk.a"))                     m_qa (clk, running, tcw, dut.qa_cr);
    tb_mon_cr  #(.PRE("qk.b"))                     m_qb (clk, running, tcw, dut.qb_cr);
    tb_mon_cr  #(.PRE("pv.b"))                     m_pb (clk, running, tcw, dut.pb_cr);
    tb_mon_win #(.PRE("ctrl.qj"))                  m_qj (clk, running, tcw, dut.qj_v & dut.qj_rdy);
    tb_mon_win #(.PRE("ctrl.pj"))                  m_pj (clk, running, tcw, dut.pj_v & dut.pj_rdy);
    tb_mon_cr  #(.MODE(1), .PRE("qk.j"))           m_qr (clk, running, tcw, dut.qj_rdy);
    tb_mon_cr  #(.MODE(1), .PRE("pv.j"))           m_pr (clk, running, tcw, dut.pj_rdy);
    tb_mon_win #(.L(NGQ), .WORDS(LW), .WPU(NWB), .PRE("qk.c")) m_qc (clk, running, tcw, dut.s_v);
    tb_mon_cr  #(.PRE("sm.s"))                     m_sc (clk, running, tcw, dut.s_cr);
    tb_mon_win #(.PRE("ctrl.go"))                  m_go (clk, running, tcw, dut.sm_start);
    tb_mon_cr  #(.MODE(2), .PRE("sm.go"))          m_gb (clk, running, tcw, dut.sm_busy);
    tb_mon_win #(.L(N_EXP), .WORDS(LW), .WPU(NWB), .PRE("sm.lp")) m_lp (clk, running, tcw, dut.lp_v);
    tb_mon_cr  #(.PRE("pv.a"))                     m_pa (clk, running, tcw, dut.lp_cr);
    tb_mon_win #(.WPU(S), .PRE("sm.la"))           m_la (clk, running, tcw, dut.la_v);
    tb_mon_cr  #(.PRE("acc.la"))                   m_ac (clk, running, tcw, dut.la_cr);
    tb_mon_win #(.WORDS(S), .PRE("sm.ll"))         m_ll (clk, running, tcw, dut.ll_v);
    tb_mon_cr  #(.PRE("acc.ll"))                   m_lc (clk, running, tcw, dut.ll_cr);
    tb_mon_win #(.L(NGP), .WORDS(LW), .WPU(WKP), .PRE("pv.c")) m_pc (clk, running, tcw, dut.p_v);
    tb_mon_cr  #(.PRE("acc.p"))                    m_ap (clk, running, tcw, dut.p_cr);
    tb_mon_win #(.WORDS(LW), .WPU(W9), .PRE("acc.o")) m_o (clk, running, tcw, o_v);
    tb_mon_cr  #(.PRE("acc.run_done"))             m_rd (clk, running, tcw, dut.run_done);
    tb_mon_cr  #(.MODE(2), .PRE("ctrl.start"))     m_bz (clk, running, tcw, busy);

    // ---- drivers + the O check (cycle tc, just after the edge) ---------------------------------------------------
    // x_win.txt is sorted by copy, then token: copy c's k-th burst is line c T_X + k
    localparam T_X = NSEQ / P_PROJ;
    integer bp [0:P_PROJ-1], bw [0:P_PROJ-1], nb [0:P_PROJ-1];   // per copy: the burst, its word (-1: idle), bursts done
    integer n_ow = 0, owin = 0, owd = 0, n_done = 0, ln, c;
    initial for (c = 0; c < P_PROJ; c = c + 1) begin bp[c] = -1; bw[c] = -1; nb[c] = 0; end
    always @(posedge clk) begin
        #2;
        if (running) begin
            if (o_v) begin                                              // E9 window owin, word owd
                if (owin >= nowin) begin
                    if (err_o < 10) $display("[tb_head] ERROR o: a word past the last window at %0d", tc);
                    errors = errors + 1;
                    err_o = err_o + 1;
                end
                else if (o_d !== ox[owin * LW + owd]) begin
                    if (err_o < 10) $display("[tb_head] ERROR o: window %0d word %0d got %h exp %h", owin, owd, o_d,
                                             ox[owin * LW + owd]);
                    errors = errors + 1;
                    err_o = err_o + 1;
                end
                n_ow = n_ow + 1;
                owd = owd + 1;
                if (owd == LW) begin owd = 0; owin = owin + 1; end
            end
            if (done) begin
                if (tc != t_done) begin $display("[tb_head] ERROR: done at %0d (expected %0d)", tc, t_done);
                                        errors = errors + 1; end
                n_done = n_done + 1;
            end
            start = (tc == t_start);
            if (start) $display("EV %0d st.start win ch0 u0 w0", tc);
            for (c = 0; c < P_PROJ; c = c + 1) begin
                if (bw[c] < 0 && nb[c] < T_X && bt[c*T_X + nb[c]] == tc) begin   // the copy's next burst starts
                    bp[c] = c * T_X + nb[c];
                    bw[c] = 0;
                    nb[c] = nb[c] + 1;
                    if (bc[bp[c]] != c) begin $display("[tb_head] ERROR: x_win.txt order"); errors = errors + 1; end
                    if (!x_rdy[c]) begin
                        if (err_x < 10)
                            $display("[tb_head] ERROR: copy %0d token %0d burst at %0d without x_rdy", c, bk[bp[c]], tc);
                        errors = errors + 1;
                        err_x = err_x + 1;
                    end
                    for (ln = 0; ln < V; ln = ln + 1)
                        $display("EV %0d x%0d.x win ch%0d u%0d w%0d", tc, c, ln, bk[bp[c]], ln);
                end
                if (bw[c] >= 0) begin
                    x_v[c] = 1'b1;
                    for (ln = 0; ln < V; ln = ln + 1)
                        x_d[(c*V + ln)*BUF +: BUF] = bd[(bp[c]*V + ln)*LOADX + bw[c]];
                    bw[c] = bw[c] + 1;
                    if (bw[c] == LOADX) bw[c] = -1;
                end
                else begin
                    x_v[c] = 1'b0;
                    x_d[c*V*BUF +: V*BUF] = 0;
                end
            end
        end
    end

    // ---- main ----------------------------------------------------------------------------------------------------
    reg [8*128-1:0] case_name;
    initial begin
        if (!$value$plusargs("VDIR=%s", vdir)) begin $display("[tb_head] ERROR: +VDIR required"); $finish; end
        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("NBUR=%d", nbur)) nbur = 0;
        if (!$value$plusargs("TMAX=%d", tmax)) tmax = 200000;
        if (!$value$plusargs("NOWIN=%d", nowin)) nowin = 0;
        $display("[tb_head] case=%0s NSEQ=%0d DM=%0d D=%0d R=%0d C=%0d P_PROJ=%0d N_E=%0d N_EP=%0d N_OC=%0d V=%0d",
                 case_name, NSEQ, DM, D, R, C, P_PROJ, N_E, N_EP, N_OC, V);
        if (nowin != RB * W9 || nowin > MAXO) begin
            $display("[tb_head] ERROR: +NOWIN=%0d, expected %0d E9 windows (at most %0d)", nowin, RB * W9, MAXO);
            $finish;
        end
        load;
        if (bad_o != 0) begin $display("[tb_head] ERROR: o_win.txt: %0d windows out of order", bad_o);
                              errors = errors + 1; end
        repeat (3) @(posedge clk);
        #1 reset = 1'b0;
        -> preload;
        running = 1'b1;
        while ((owin < nowin || n_done < 1 || busy) && tc < tmax) @(posedge clk);
        repeat (4) @(posedge clk);
        #4;
        if (tc >= tmax) begin $display("[tb_head] ERROR: watchdog (tc %0d, windows out %0d of %0d)", tc, owin, nowin);
                              errors = errors + 1; end
        if (n_ow != nowin * LW) begin $display("[tb_head] ERROR: %0d O words, expected %0d", n_ow, nowin * LW);
                                      errors = errors + 1; end
        if (n_done != 1) begin $display("[tb_head] ERROR: %0d done pulses", n_done); errors = errors + 1; end
        $display("[tb_head] %0s %0s errors=%0d (O values %0d, bursts without x_rdy %0d) o_words=%0d end=%0d",
                 (errors == 0) ? "PASS" : "FAIL", case_name, errors, err_o, err_x, n_ow, tc);
        $finish;
    end
endmodule
