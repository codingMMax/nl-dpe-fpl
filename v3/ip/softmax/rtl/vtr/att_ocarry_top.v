// ============================================================================
// att_ocarry_top.v — attention head o-carry station (F3; implementation_plan.md
// "ATT-2c as designed (2026-10-08)"). VTR source: the bare `dpe` hard blocks
// (port contract v2/experiment/rtl/dpe_blackbox.v, arch-XML <model name="dpe">)
// directly in the two crossbar generate loops; no weight programming (the identity
// weights and the EXP / LOG modes are the hard blocks' configuration); never
// simulated. Simulation twin v2/sim/rtl/att_ocarry_top.v (gate: check_split.py).
//
// Ground truth: v2/sim/oracle/attention_ref.py (ocarry_stage_values + epilogue);
// behavior + cycle model v2/sim/simulator/kernels/ocarry_sim.py (GATE 1 in dump_case).
//
// Element e = r*D + c (token-major: the untransposed PV tile's accumulator, user
// 2026-10-08), window p = e div I (P_OC = S*D/I windows of RPW = I/D query rows),
// word q (5 elements), wave w = p div N_OC, crossbar i = p mod N_OC; the rows of
// crossbar i in wave w are (w*N_OC + i)*RPW .. + RPW-1 (log alpha / log l per row). Two groups of N_OC identity crossbars (P29 extent I x I), each
// lock-step from its crossbar 0:
//   EXP group, jobs per run: rescale (b, w), b = 1..NBLK-1, then the epilogue (w)
//     rescale feed  t  = clamp8(log_a[b][r] + lo)      drain e  -> e_mem (slot w)
//     epilogue feed tE = clamp8(lo - log_l[r])         drain O8 = sat8(+-(e * 2^(SO-LN))) -> o8_*
//   LOG group, jobs (b, w), b = 0..NBLK-1
//     feed  o' = (b ? (+-e) << SO : 0) + o_part, oq = min(|o'| >> SO, 127), so <- (o' < 0)
//     drain lo -> lo_mem (slot w)
// o_part input (rework M1, 2026-10-08): the window-major PV tile's output stream (in_valid /
// in_data: one word of each of its NGP groups per cycle; group f carries windows f, f + NGP, ...)
// lands in NGP FIFOs (att_fifo, window p in FIFO p mod NGP, two tiles deep); a LOG job reads
// word c of its windows w*N_OC + i from FIFOs (w*N_OC + i) mod NGP in lock-step (distinct
// FIFOs: N_OC <= NGP). op_free pulses after a block's last o_part read (the head lets the PV
// tile two blocks ahead start: the FIFOs never hold more than two tiles).
// Readiness: LOG(b, w): every window of the job complete in its FIFO (count >= LW); b >= 1: EXP(b, w) drained.
//            EXP(b, w): la_go of block b seen (slot b mod 2); LOG(b-1, w) drained.
//            EPI(w):    ll_go seen; LOG(NBLK-1, w) drained.
// A job starts (decide) when its group engine is idle, the job is ready and the
// leader's MSB_SA_Ready is high; word c is read at decide + c and fed (registered)
// at decide + c + 2 — a started burst is never stalled by the primitive.
// Head contract: the PV tile of block b+2 starts after op_free(b); log_a of block b written
// into slot b mod 2 after la_free(b-2); the next run's log_a / log_l writes and pulses after
// run_done (its o_part may arrive earlier: a run's state restarts at its last O8 word).
// O buffer = e_mem (rework M2, user 2026-10-09): the epilogue writes the run's O8 into e_mem at
// {w, q} (the rescale drains and the epilogue share its one write port, one cycle after the drain);
// the head's O feed reads it through the read port or_en / or_addr / or_data (registered, word
// {w, q} of every crossbar). While or_en is high the LOG feed's read of e is not served — it uses
// e only for b >= 1, after the run's first rescale. A run's EXP jobs wait for o_fed of the run
// before (its O8 read; none after reset), so no rescale or epilogue overwrites O8 still unread.
// Domain (elaboration guards): S, D, I, NGP powers of two, 8 <= S <= I, I | S*D,
// N_OC <= P_OC, N_OC <= NGP, NGP | P_OC, NBLK >= 1, BUF = 40.
// ============================================================================

`timescale 1ns / 1ps

module att_ocarry_top #(
    parameter S    = 16,
    parameter D    = 8,
    parameter R    = 64,
    parameter C    = 64,
    parameter BUF  = 40,
    parameter P    = 8,
    parameter N_OC = 2,
    parameter NBLK = 2,
    parameter SO   = 4,
    parameter LN   = 4,
    parameter NGP  = 2,    // PV tile groups (o_part FIFOs)
    parameter VWP  = 13,   // o_part lane width (signed sum of Bkv EXP bytes)
    // derived (port widths; do not override)
    parameter I    = (R < C) ? R : C,
    parameter P_OC = (S * D) / I,
    parameter W    = (P_OC + N_OC - 1) / N_OC,
    parameter LW   = (I + 4) / 5,
    parameter LS   = $clog2(S),
    parameter WSW  = (W > 1) ? $clog2(W) : 1,
    parameter QW   = $clog2(LW)
) (
    input  wire                 clk,
    input  wire                 reset,
    // log alpha (per block b >= 1, slot b mod 2) and log l (per run)
    input  wire                 la_wen,
    input  wire                 la_wslot,
    input  wire [LS-1:0]        la_wr,
    input  wire [8:0]           la_wdata,                    // log alpha (9 bit, >= -255)
    input  wire                 la_go,
    output reg                  la_free,
    input  wire                 ll_wen,
    input  wire [LS-1:0]        ll_wr,
    input  wire [7:0]           ll_wdata,
    input  wire                 ll_go,
    // o_part: the PV tile's output stream (every word taken)
    input  wire                 in_valid,
    input  wire [NGP*5*VWP-1:0] in_data,
    output reg                  op_free,
    // O8 (epilogue): word q of the windows of wave w, one 40-bit word per crossbar
    output reg                  o8_valid,
    output reg  [WSW-1:0]       o8_w,
    output reg  [QW-1:0]        o8_q,
    output reg  [N_OC*40-1:0]   o8_data,
    output reg  [N_OC-1:0]      o8_mask,
    output reg                  run_done,
    // the O buffer (e_mem) read port of the head's O feed, and its release
    input  wire                 or_en,
    input  wire [WSW+QW-1:0]    or_addr,
    output wire [N_OC*40-1:0]   or_data,
    input  wire                 o_fed
);

    localparam EPS = BUF / 8;
    localparam NJ  = NBLK * W;                          // jobs per group per run
    localparam JW  = $clog2(NJ + 1) + 1;
    localparam AW  = WSW + QW;                          // slot memory address {w, q}
    localparam LWC = P_OC - (W - 1) * N_OC;             // windows of the last wave
    localparam SHL = (SO > LN) ? SO - LN : 0;            // epilogue scale 2^(SO-LN)
    localparam SHR = (SO < LN) ? LN - SO : 0;
    localparam RPW = I / D;                             // query rows per window
    localparam LD  = $clog2(D);
    localparam LRP = (RPW > 1) ? $clog2(RPW) : 1;
    localparam LI  = $clog2(I);
    localparam NRS = (NBLK - 1) * W;                    // rescale jobs per run
    localparam ETW = WSW + 3;                           // EXP tag {lastw, slot, epi, w}
    localparam LTW = WSW + 2;                           // LOG tag {lastw, bnz, w}
    localparam LNG = (NGP > 1) ? $clog2(NGP) : 1;
    localparam TWP = P_OC / NGP;                        // PV waves per block
    localparam FD  = 1 << $clog2(2 * TWP * LW);         // FIFO depth: two tiles of o_part words
    localparam FAW = $clog2(FD);

    generate
        if ((S & (S - 1)) != 0 || (D & (D - 1)) != 0 || (I & (I - 1)) != 0 || S < 8 || S > I
            || ((S * D) % I) != 0 || D > I || N_OC < 1 || N_OC > P_OC || NBLK < 1 || EPS != 5
            || (NGP & (NGP - 1)) != 0 || N_OC > NGP || (P_OC % NGP) != 0)
        begin : g_bad_param
            att_ocarry_parameter_outside_domain u_bad ();
        end
    endgenerate

    // ---- log alpha (2 slots) and log l tables: one register per entry -------------
    wire [2*S*9-1:0] la_flat;
    wire [S*8-1:0]   ll_flat;
    genvar ga;
    generate
        for (ga = 0; ga < 2 * S; ga = ga + 1) begin : g_la
            reg [8:0] v;
            always @(posedge clk) if (la_wen && {la_wslot, la_wr} == ga) v <= la_wdata;
            assign la_flat[ga*9 +: 9] = v;
        end
        for (ga = 0; ga < S; ga = ga + 1) begin : g_ll
            reg [7:0] v;
            always @(posedge clk) if (ll_wen && ll_wr == ga) v <= ll_wdata;
            assign ll_flat[ga*8 +: 8] = v;
        end
    endgenerate

    // ---- o_part FIFO state (the FIFOs follow the engines): FIFO of window w*N_OC for the next
    // job / the job being read / the data stage, windows of the job
    wire           fifo_ok, fifo_err;
    reg  [LNG-1:0] lbase, rbase, cbase;
    reg  [LNG:0]   rnact;
    wire [LNG:0]   nact_l;
    wire [LNG-1:0] base_rd;
    wire [LNG:0]   nact_rd;
    wire [NGP*5*VWP-1:0] fq;                            // FIFO read data (one cycle after the pop)

    // ---- run state: input flags, job indices, drained counts ------------------
    reg            ll_v;
    reg  [1:0]     la_v;
    reg  [JW-1:0]  la_cnt;                              // la_go pulses this run (block of the next: la_cnt + 1)
    reg  [JW-1:0]  lj, ej;                              // next LOG / EXP job (linear index in the run)
    reg  [WSW-1:0] lw, ew;                              // its wave
    reg  [JW-1:0]  lb, eb;                              // its block (EXP: rescale block, 1..)
    reg  [JW-1:0]  ld_cnt, ed_cnt;                      // LOG / EXP jobs drained this run
    wire           e_epi = (ej >= NRS);

    // LOG / EXP engines (one per group; leader = crossbar 0 of the group)
    wire           lg_ready, lg_full, ex_ready, ex_full;
    wire           lg_job = (lj < NJ) && fifo_ok && (lb == 0 || ed_cnt + W > lj);
    reg            o_ok;                                // the previous run's O8 was read: e_mem may be written
    wire           ex_job = (ej < NJ) && (ld_cnt > ej) && (e_epi ? ll_v : la_v[eb[0]]) && o_ok;
    wire [LTW-1:0] lg_tag = {lw == W - 1, lb != 0, lw};
    wire [ETW-1:0] ex_tag = {ew == W - 1, eb[0], e_epi, ew};

    wire           lg_dec, lg_rv, lg_drv, lg_dlast, ex_dec, ex_rv, ex_drv, ex_dlast;
    wire [QW-1:0]  lg_rc, lg_dq, ex_rc, ex_dq;
    wire [LTW-1:0] lg_rtag, lg_dtag;
    wire [ETW-1:0] ex_rtag, ex_dtag;
    wire           lg_cv, lg_sv, ex_cv, ex_sv, lg_busy, ex_busy;
    wire [QW-1:0]  lg_cc, ex_cc, lg_rcq, ex_rcq;
    wire [LTW-1:0] lg_tq;
    wire [ETW-1:0] ex_tq;
    wire [LTW-1:0] lg_ctag, lg_stag;
    wire [ETW-1:0] ex_ctag, ex_stag;
    att_oc_engine #(.LW(LW), .QW(QW), .TAGW(LTW)) u_lg (
        .clk(clk), .reset(reset), .job_ready(lg_job), .job_tag(lg_tag),
        .leader_ready(lg_ready), .leader_full(lg_full), .busy(lg_busy), .rc(lg_rcq), .tag_q(lg_tq), .decide(lg_dec),
        .rd_v(lg_rv), .rd_c(lg_rc), .rd_tag(lg_rtag), .cs_v(lg_cv), .cs_c(lg_cc), .cs_tag(lg_ctag),
        .st_v(lg_sv), .st_tag(lg_stag), .drn_v(lg_drv), .drn_q(lg_dq), .drn_last(lg_dlast), .drn_tag(lg_dtag));
    att_oc_engine #(.LW(LW), .QW(QW), .TAGW(ETW)) u_ex (
        .clk(clk), .reset(reset), .job_ready(ex_job), .job_tag(ex_tag),
        .leader_ready(ex_ready), .leader_full(ex_full), .busy(ex_busy), .rc(ex_rcq), .tag_q(ex_tq), .decide(ex_dec),
        .rd_v(ex_rv), .rd_c(ex_rc), .rd_tag(ex_rtag), .cs_v(ex_cv), .cs_c(ex_cc), .cs_tag(ex_ctag),
        .st_v(ex_sv), .st_tag(ex_stag), .drn_v(ex_drv), .drn_q(ex_dq), .drn_last(ex_dlast), .drn_tag(ex_dtag));

    // ---- o_part FIFOs ---------------------------------------------------------------
    wire [NGP-1:0] fpop, f_ok, f_ovf, f_unf;
    genvar gf;
    generate
        for (gf = 0; gf < NGP; gf = gf + 1) begin : g_f
            wire [FAW:0] cnt;
            att_fifo #(.W(5*VWP), .DEPTH(FD)) u_f (
                .clk(clk), .reset(reset), .wr_en(in_valid), .wr_data(in_data[gf*5*VWP +: 5*VWP]),
                .rd_en(fpop[gf]), .rd_data(fq[gf*5*VWP +: 5*VWP]), .count(cnt),
                .overflow(f_ovf[gf]), .underflow(f_unf[gf]));
            wire [LNG-1:0] il = (gf + NGP - lbase) & (NGP - 1);       // crossbar of this FIFO in the next job
            wire [LNG-1:0] ir = (gf + NGP - base_rd) & (NGP - 1);     // ... in the job being read
            assign f_ok[gf] = (il >= nact_l) || (cnt >= LW);
            assign fpop[gf] = lg_rv && (ir < nact_rd);
        end
    endgenerate
    assign fifo_ok  = &f_ok;
    assign fifo_err = |f_ovf || |f_unf;                 // TB probe

    // EXP drain pipeline: drain word -> e_mem write (rescale e or the epilogue's O8) / O8 register
    reg            p1_e;                                // an EXP drain word (written into e_mem)
    reg            p1_v, p1_last;                       // ... of the epilogue (O8)
    reg  [WSW-1:0] p1_w;
    reg  [QW-1:0]  p1_q;
    wire           o8_last = o8_valid && o8_w == W - 1 && o8_q == LW - 1;

    // FIFO of the next job's first window (advances with the LOG job pointer), of the job being read
    assign nact_l  = (lw == W - 1) ? LWC : N_OC;
    assign base_rd = lg_dec ? lbase : rbase;
    assign nact_rd = lg_dec ? nact_l : rnact;
    always @(posedge clk) begin
        if (reset || o8_last) lbase <= 0;
        else if (lg_dec) lbase <= (lw == W - 1) ? {LNG{1'b0}} : ((lbase + N_OC) & (NGP - 1));
        if (lg_dec) begin rbase <= lbase; rnact <= nact_l; end
        cbase <= base_rd;
    end
    always @(posedge clk) begin
        if (reset || o8_last) begin
            ll_v <= 1'b0; la_v <= 2'b00; la_cnt <= 0;
            lj <= 0; lw <= 0; lb <= 0; ej <= 0; ew <= 0; eb <= 1;
            ld_cnt <= 0; ed_cnt <= 0;
            op_free <= 1'b0; la_free <= 1'b0;
        end
        else begin
            // input pulses (visible the next cycle)
            if (ll_go) ll_v <= 1'b1;
            if (la_go) begin la_v[~la_cnt[0]] <= 1'b1; la_cnt <= la_cnt + 1'b1; end
            // job pointers advance at the decision
            if (lg_dec) begin
                lj <= lj + 1'b1;
                if (lw == W - 1) begin lw <= 0; lb <= lb + 1'b1; end else lw <= lw + 1'b1;
            end
            if (ex_dec) begin
                ej <= ej + 1'b1;
                if (ew == W - 1) begin ew <= 0; eb <= eb + 1'b1; end else ew <= ew + 1'b1;
            end
            // drained counts (visible the next cycle)
            if (lg_dlast) ld_cnt <= ld_cnt + 1'b1;
            if (ex_dlast) ed_cnt <= ed_cnt + 1'b1;
            // o_part released after the block's last read; log_a slot after its last use
            op_free <= lg_rv && lg_rc == LW - 1 && lg_rtag[LTW-1];
            la_free <= ex_cv && ex_cc == LW - 1 && ex_ctag[ETW-1] && !ex_ctag[WSW];
            if (ex_cv && ex_cc == LW - 1 && ex_ctag[ETW-1] && !ex_ctag[WSW]) la_v[ex_ctag[WSW+1]] <= 1'b0;
        end
    end
    always @(posedge clk) run_done <= !reset && o8_last;
    always @(posedge clk) begin                         // O8 in e_mem from the run's last O8 word to o_fed
        if (reset || o_fed) o_ok <= 1'b1;
        else if (o8_last)   o_ok <= 1'b0;
    end

    // ---- lane rows within a window: element offset of the word read this cycle (shared by
    // every crossbar): lane j of word c is element 5c + j of its window, row (5c + j) >> LD.
    // Lanes past the window (last word) need no mask: the I x I pass never loads rows >= I.
    genvar gj, gr, gv;
    wire [EPS*(LI+4)-1:0] e_all;
    generate
        for (gj = 0; gj < EPS; gj = gj + 1) begin : g_lane
            reg [LI+3:0] e;                                     // offset of the running word's lane j
            always @(posedge clk) e <= ex_dec ? (gj + 5) : e + 5;
            assign e_all[gj*(LI+4) +: LI+4] = e;
        end
    endgenerate


    // ---- per crossbar index i: slot memories, the EXP and LOG crossbars --------
    wire [N_OC-1:0]    lg_rdy_all, lg_full_all, ex_rdy_all, ex_full_all;
    wire [N_OC*40-1:0] lg_out_all, o8_w_all;
    wire [N_OC-1:0]    o8_m_all;
    assign lg_ready = lg_rdy_all[0];
    assign lg_full  = lg_full_all[0];
    assign ex_ready = ex_rdy_all[0];
    assign ex_full  = ex_full_all[0];
    genvar gi, gw;                                  // gj, gr, gv declared above
    generate
        for (gi = 0; gi < N_OC; gi = gi + 1) begin : g_x
            // window of this crossbar in wave w: w*N_OC + gi (present iff w < W-1 or gi < LWC)
            wire act_l = (lg_stag[WSW-1:0] != W - 1) || (gi < LWC);
            wire act_e = (ex_stag[WSW-1:0] != W - 1) || (gi < LWC);
            // slot memories {w, q}: lo (LOG drain -> EXP feed), e (EXP drain -> LOG feed; the epilogue's
            // O8 -> the O feed), so (LOG feed -> LOG feed, epilogue drain)
            reg [39:0] lo_mem [0:(W << QW) - 1];
            reg [39:0] e_mem  [0:(W << QW) - 1];
            reg [4:0]  so_mem [0:(W << QW) - 1];
            reg [39:0] lo_rd, e_rd;
            reg [4:0]  so_rd, so_ep;
            wire [AW-1:0] lg_ra = {lg_rtag[WSW-1:0], lg_rc};
            wire [AW-1:0] ex_ra = {ex_rtag[WSW-1:0], ex_rc};
            // o_part word of this crossbar's window in the compute stage: FIFO (cbase + gi) mod NGP
            wire [LNG-1:0]   fsel = (cbase + gi) & (NGP - 1);
            reg  [5*VWP-1:0] opw;                                   // constant-offset select (no runtime product)
            integer          fsi;
            always @(*) begin
                opw = fq[0 +: 5*VWP];
                for (fsi = 1; fsi < NGP; fsi = fsi + 1) if (fsel == fsi) opw = fq[fsi*5*VWP +: 5*VWP];
            end
            // LOG feed: o' = (b ? (+-e) << SO : 0) + o_part -> oq, so
            wire [39:0] oq_w;
            wire [4:0]  so_w;
            for (gj = 0; gj < EPS; gj = gj + 1) begin : g_lf
                wire        [VWP-1:0] ox = opw[gj*VWP +: VWP];
                wire signed [31:0] op = {{(32-VWP){ox[VWP-1]}}, ox};
                wire signed [33:0] ev = so_rd[gj] ? -$signed({1'b0, e_rd[gj*8 +: 8]}) : $signed({1'b0, e_rd[gj*8 +: 8]});
                wire signed [33:0] on = (lg_ctag[WSW] ? (ev <<< SO) : 34'sd0) + op;
                wire        [33:0] mg = on[33] ? -on : on;
                wire        [33:0] sq = mg >> SO;
                assign oq_w[gj*8 +: 8] = (sq > 127) ? 8'd127 : sq[7:0];
                assign so_w[gj] = on[33];
            end
            // EXP x vector of this crossbar's RPW rows for the starting job (rescale: log alpha of
            // its slot, epilogue: -log l), chosen at the decision (a W:1 mux per row) and held
            wire [RPW*9-1:0] xv_c;
            reg  [RPW*9-1:0] xv_q;
            for (gr = 0; gr < RPW; gr = gr + 1) begin : g_xv
                wire [W*9-1:0] a0, a1, l9;
                for (gv = 0; gv < W; gv = gv + 1) begin : g_wv
                    localparam RW = (gv * N_OC + gi) * RPW + gr;    // row of window gv*N_OC + gi
                    localparam RC = (RW < S) ? RW : 0;
                    assign a0[gv*9 +: 9] = la_flat[RC*9 +: 9];
                    assign a1[gv*9 +: 9] = la_flat[(S + RC)*9 +: 9];
                    assign l9[gv*9 +: 9] = -{ll_flat[RC*8+7], ll_flat[RC*8 +: 8]};
                end
                wire [WSW-1:0] wj = ex_tag[WSW-1:0];
                reg  [8:0]     xs;                                  // constant-offset select (no runtime product)
                integer        xwi;
                always @(*) begin
                    xs = ex_tag[WSW] ? l9[0 +: 9] : (ex_tag[WSW+1] ? a1[0 +: 9] : a0[0 +: 9]);
                    for (xwi = 1; xwi < W; xwi = xwi + 1)
                        if (wj == xwi) xs = ex_tag[WSW] ? l9[xwi*9 +: 9] : (ex_tag[WSW+1] ? a1[xwi*9 +: 9] : a0[xwi*9 +: 9]);
                end
                assign xv_c[gr*9 +: 9] = xs;
            end
            always @(posedge clk) if (ex_dec) xv_q <= xv_c;
            // per lane: word 0 (the decision) takes row 0 of xv_c, later words the row of their offset
            reg  [EPS*9-1:0] ex_x;
            for (gj = 0; gj < EPS; gj = gj + 1) begin : g_xl
                wire [LI+3:0]  ej = e_all[gj*(LI+4) +: LI+4];
                wire [LRP-1:0] rj = (RPW > 1) ? (ej >> LD) : 1'b0;
                reg  [8:0]     xq;                                  // constant-offset select (no runtime product)
                integer        xri;
                always @(*) begin
                    xq = xv_q[0 +: 9];
                    for (xri = 1; xri < RPW; xri = xri + 1) if (rj == xri) xq = xv_q[xri*9 +: 9];
                end
                always @(posedge clk) ex_x[gj*9 +: 9] <= ex_dec ? xv_c[0 +: 9] : xq;
            end
            // EXP feed: t = clamp8(lo + x)
            wire [39:0] t_w;
            for (gj = 0; gj < EPS; gj = gj + 1) begin : g_ef
                wire signed [9:0] sm = $signed({lo_rd[gj*8+7], lo_rd[gj*8 +: 8]}) + $signed(ex_x[gj*9 +: 9]);
                assign t_w[gj*8 +: 8] = (sm > 127) ? 8'd127 : ((sm < -128) ? 8'h80 : sm[7:0]);
            end
            reg [39:0] lf_q, ef_q;                      // registered feed words
            always @(posedge clk) begin
                lo_rd <= lo_mem[ex_ra];
                e_rd  <= e_mem[or_en ? or_addr : lg_ra];
                so_rd <= so_mem[lg_ra];
                so_ep <= so_mem[{ex_dtag[WSW-1:0], ex_dq}];
                lf_q  <= oq_w;
                ef_q  <= t_w;
                if (lg_cv) so_mem[{lg_ctag[WSW-1:0], lg_cc}] <= so_w;
            end
            // LOG crossbar
            wire [BUF-1:0] lg_in = lf_q;
            wire [BUF-1:0] lg_out;
            wire           lg_done_x;
            dpe u_lg (
                .clk(clk), .reset(reset), .data_in(lg_in), .nl_dpe_control(2'b11),   // LOG
                .shift_add_control(1'b0), .w_buf_en(lg_sv && act_l), .shift_add_bypass(1'b0),
                .load_output_reg(1'b0), .load_input_reg(1'b0),
                .MSB_SA_Ready(lg_rdy_all[gi]), .data_out(lg_out), .dpe_done(lg_done_x),
                .reg_full(lg_full_all[gi]));
            always @(posedge clk)
                if (lg_drv) lo_mem[{lg_dtag[WSW-1:0], lg_dq}] <= lg_out;
            assign lg_out_all[gi*40 +: 40] = lg_out;
            // EXP crossbar
            wire [BUF-1:0] ex_in = ef_q;
            wire [BUF-1:0] ex_out;
            wire           ex_done_x;
            dpe u_ex (
                .clk(clk), .reset(reset), .data_in(ex_in), .nl_dpe_control(2'b10),   // EXP
                .shift_add_control(1'b0), .w_buf_en(ex_sv && act_e), .shift_add_bypass(1'b0),
                .load_output_reg(1'b0), .load_input_reg(1'b0),
                .MSB_SA_Ready(ex_rdy_all[gi]), .data_out(ex_out), .dpe_done(ex_done_x),
                .reg_full(ex_full_all[gi]));
            // epilogue output: O8 = sat8(+-(e * 2^SH)) with the final sign
            reg [39:0] ee_q;
            always @(posedge clk) ee_q <= ex_out;
            wire [39:0] o8_l;
            for (gj = 0; gj < EPS; gj = gj + 1) begin : g_o8
                wire [17:0] ez = {10'd0, ee_q[gj*8 +: 8]};
                wire [17:0] es = (ez << SHL) >> SHR;
                wire [QW+3:0] e5 = {p1_q, 2'b00} + p1_q + gj;
                assign o8_l[gj*8 +: 8] = !(e5 < I) ? 8'd0
                    : (so_ep[gj] ? ((es > 128) ? 8'h80 : (8'd0 - es[7:0])) : ((es > 127) ? 8'd127 : es[7:0]));
            end
            assign o8_w_all[gi*40 +: 40] = o8_l;
            // e_mem write port: every EXP drain word one cycle after the drain (rescale: e, epilogue: O8)
            always @(posedge clk)
                if (p1_e) e_mem[{p1_w, p1_q}] <= p1_v ? o8_l : ee_q;
            assign or_data[gi*40 +: 40] = e_rd;
            assign o8_m_all[gi] = (p1_w != W - 1) || (gi < LWC);
        end
    endgenerate

    always @(posedge clk) begin
        p1_e    <= !reset && ex_drv;
        p1_v    <= !reset && ex_drv && ex_dtag[WSW];
        p1_w    <= ex_dtag[WSW-1:0];
        p1_q    <= ex_dq;
        o8_valid <= !reset && p1_v;
        o8_w    <= p1_w;
        o8_q    <= p1_q;
        o8_data <= o8_w_all;
        o8_mask <= o8_m_all;
    end

    // probes (TB): LOG drain words (lo), the engines' feed state
    wire               lg_dv = lg_drv;
    wire [WSW-1:0]     lg_dw = lg_dtag[WSW-1:0];
    wire [N_OC*40-1:0] lg_dout = lg_out_all;
    // the O buffer while the O feed reads it: no e_mem write, no LOG read of e (b >= 1)
    wire               emem_err = or_en && (p1_e || (lg_rv && lg_rtag[WSW]));

endmodule


// ============================================================================
// att_oc_engine — one lock-step crossbar group's window engine. A job starts
// (decide) when the engine is idle (no word in its read / compute / strobe
// stages), the job is ready and the leader's MSB_SA_Ready is high; word c of the job is read at decide + c (rd_*), its
// memory data are valid the next cycle (cs_*), the registered feed word is
// strobed the cycle after (st_*). Drains are in order: the leader's reg_full
// marks a gapless burst of LW words (drn_*), tagged from a depth-4 job FIFO.
// ============================================================================
module att_oc_engine #(
    parameter LW = 13,
    parameter QW = 4,
    parameter TAGW = 4
) (
    input  wire            clk,
    input  wire            reset,
    input  wire            job_ready,
    input  wire [TAGW-1:0] job_tag,
    input  wire            leader_ready,
    input  wire            leader_full,
    output reg             busy,
    output reg  [QW-1:0]   rc,                 // word read next cycle while busy (1 .. LW-1)
    output reg  [TAGW-1:0] tag_q,              // the running job's tag
    output wire            decide,
    output wire            rd_v,
    output wire [QW-1:0]   rd_c,
    output wire [TAGW-1:0] rd_tag,
    output reg             cs_v,
    output reg  [QW-1:0]   cs_c,
    output reg  [TAGW-1:0] cs_tag,
    output reg             st_v,
    output reg  [TAGW-1:0] st_tag,
    output wire            drn_v,
    output wire [QW-1:0]   drn_q,
    output wire            drn_last,
    output wire [TAGW-1:0] drn_tag
);
    // a new job only once the previous burst has fully strobed (by then the leader's
    // MSB_SA_Ready has dropped; it rises again after that window's MSB fire)
    assign decide = !busy && !cs_v && !st_v && job_ready && leader_ready;
    assign rd_v   = decide || busy;
    assign rd_c   = decide ? {QW{1'b0}} : rc;
    assign rd_tag = decide ? job_tag : tag_q;

    // drain: word counter over the leader's reg_full burst, job tags in order
    reg [QW-1:0]   dq;
    reg [TAGW-1:0] f0, f1, f2, f3;                     // job tags in decision order
    reg [1:0]      wp, rp;
    assign drn_v    = leader_full;
    assign drn_q    = dq;
    assign drn_last = leader_full && dq == LW - 1;
    assign drn_tag  = (rp == 2'd0) ? f0 : (rp == 2'd1) ? f1 : (rp == 2'd2) ? f2 : f3;

    always @(posedge clk) begin
        if (reset) begin
            busy <= 1'b0; rc <= 0; cs_v <= 1'b0; st_v <= 1'b0; dq <= 0; wp <= 0; rp <= 0;
        end
        else begin
            if (decide) begin
                busy <= (LW > 1); rc <= 1; tag_q <= job_tag;
                wp <= wp + 1'b1;
                case (wp)
                    2'd0: f0 <= job_tag;
                    2'd1: f1 <= job_tag;
                    2'd2: f2 <= job_tag;
                    default: f3 <= job_tag;
                endcase
            end
            else if (busy) begin
                if (rc == LW - 1) busy <= 1'b0;
                else rc <= rc + 1'b1;
            end
            cs_v <= rd_v; cs_c <= rd_c; cs_tag <= rd_tag;
            st_v <= cs_v; st_tag <= cs_tag;
            dq <= leader_full ? dq + 1'b1 : {QW{1'b0}};
            if (drn_last) rp <= rp + 1'b1;
        end
    end
endmodule
