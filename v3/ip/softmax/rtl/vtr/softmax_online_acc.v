// ============================================================================
// softmax_online_acc.v — the online softmax's accumulator (v3, ip/softmax/acc_spec.md). VTR twin: bare `dpe` hard
// blocks (port contract ip/xbar/rtl/vtr/dpe_blackbox.v), weights preloaded, never simulated. Simulation twin
// ip/softmax/rtl/sim/softmax_online_acc.v (behavior `dpe`, sim-only weight ports); tools/check_twins.py gates the pair.
//
// Built from v2's o-carry station (v2/sim/rtl/att_ocarry_top.v): the crossbar groups, the job engines, the slot
// memories and the o_part FIFOs are unchanged. New in v3: the E6 / E7 stream inputs into the alpha ring and the log l
// table, the credits p_cr / la_cr / ll_cr, the E9 output (the epilogue's O8 words as they drain), run_done.
//
// Element e = m*D + c (row-major tile), window p = e div I (P_OC = S*D/I windows of RPW = I/D rows), word q
// (5 elements), wave w = p div N_OC, crossbar i = p mod N_OC. Two groups of N_OC identity crossbars (extent I x I),
// each lock-step from its crossbar 0:
//   EXP group, jobs per run: rescale (b, w), b = 1..NBLK-1, then the epilogue (w)
//     rescale  feed t  = clamp8(log_a[m] + lo)     drain e  -> e_mem (slot w)
//     epilogue feed tE = clamp8(lo - log_l[m])     drain O8 = sat8(+-(e * 2^(SO-LN))) -> E9 (o_d word i)
//   LOG group, jobs (b, w), b = 0..NBLK-1
//     feed o' = (b ? (+-e) << SO : 0) + o_part, oq = min(|o'| >> SO, 127), so <- (o' < 0); drain lo -> lo_mem
// Ports (spec §3):
//   E8 p_v / p_d: o_part windows, window p of a tile on channel p mod P_CH (a FIFO per channel, two tiles); p_cr one
//      cycle after a tile's last read
//   E6 la_v / la_d: log alpha, one row per word; unit u (block b >= 1) into ring slot u mod 2; la_cr after its last use
//   E7 ll_v / ll_d: the run's log l, one row per word; ll_cr after its last use
//   E9 o_v / o_d: the final O8 values, N_OC words of 5 int8 per cycle (word i from crossbar i: window w*N_OC + i,
//      word q), two cycles after the epilogue drains them; crossbars without a window in the last wave send zeros;
//      run_done one cycle after the run's end
// Readiness: LOG (b, w): its windows complete in their FIFOs; b >= 1: rescale (b, w) drained.
//            rescale (b, w): its alpha unit present; LOG (b-1, w) drained.
//            epilogue (w): log l present; LOG (NBLK-1, w) drained.
// A job starts (decide) when its group engine is idle, the job is ready and the leader's MSB_SA_Ready is high; word c is
// read at decide + c and fed (registered) at decide + c + 2. The run ends (o8_last) two cycles after its last epilogue
// drain word, with its last E9 word: the run state restarts.
// Domain (elaboration guards): S, D, I, P_CH powers of two, 8 <= S <= I, D <= I, I | S*D, 1 <= N_OC <= P_OC,
// N_OC <= P_CH, P_CH | P_OC, NBLK >= 1, BUF = 40, SO, LN >= 0, 1 <= VWP <= 32.
// ============================================================================

`timescale 1ns / 1ps

module softmax_online_acc #(
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
    parameter P_CH = 2,
    parameter VWP  = 13
) (
    input  wire                  clk,
    input  wire                  reset,
    // E8 o_part (the PV DIMM's output stream)
    input  wire [P_CH-1:0]       p_v,
    input  wire [P_CH*5*VWP-1:0] p_d,
    output reg                   p_cr,
    // E6 log alpha (9 bit, >= -255), E7 log l
    input  wire                  la_v,
    input  wire [8:0]            la_d,
    output reg                   la_cr,
    input  wire                  ll_v,
    input  wire [7:0]            ll_d,
    output reg                   ll_cr,
    // E9 the final O8 values: one word per crossbar
    output reg                   o_v,
    output reg  [N_OC*BUF-1:0]   o_d,
    output reg                   run_done
);

    localparam I    = (R < C) ? R : C;
    localparam P_OC = (S * D) / I;                      // windows per run
    localparam W    = (P_OC + N_OC - 1) / N_OC;         // waves
    localparam LW   = (I + 4) / 5;                      // words per window
    localparam LS   = $clog2(S);
    localparam WSW  = (W > 1) ? $clog2(W) : 1;
    localparam QW   = $clog2(LW);
    localparam EPS  = BUF / 8;
    localparam NJ   = NBLK * W;                         // jobs per group per run
    localparam JW   = $clog2(NJ + 1) + 1;
    localparam AW   = WSW + QW;                         // slot memory address {w, q}
    localparam LWC  = P_OC - (W - 1) * N_OC;            // windows of the last wave
    localparam SHL  = (SO > LN) ? SO - LN : 0;          // epilogue scale 2^(SO-LN)
    localparam SHR  = (SO < LN) ? LN - SO : 0;
    localparam RPW  = I / D;                            // rows per window
    localparam LD   = $clog2(D);
    localparam LRP  = (RPW > 1) ? $clog2(RPW) : 1;
    localparam LI   = $clog2(I);
    localparam NRS  = (NBLK - 1) * W;                   // rescale jobs per run
    localparam ETW  = WSW + 3;                          // EXP tag {lastw, slot, epi, w}
    localparam LTW  = WSW + 2;                          // LOG tag {lastw, bnz, w}
    localparam NRT  = N_OC % P_CH;                      // FIFO rotation from one wave to the next
    localparam OW   = ((9 + SO > VWP) ? 9 + SO : VWP) + 1;   // o' bits: |(+-e) << SO| + |o_part| < 2^(OW-1)
    localparam TWP  = P_OC / P_CH;                      // windows per channel per tile
    localparam FD   = 1 << $clog2(2 * TWP * LW);        // FIFO depth: two tiles of o_part words
    localparam FAW  = $clog2(FD);

    generate
        if ((S & (S - 1)) != 0 || (D & (D - 1)) != 0 || (I & (I - 1)) != 0 || (P_CH & (P_CH - 1)) != 0
            || S < 8 || S > I || D > I || ((S * D) % I) != 0 || N_OC < 1 || N_OC > P_OC || N_OC > P_CH
            || (P_OC % P_CH) != 0 || NBLK < 1 || EPS != 5 || SO < 0 || LN < 0 || VWP < 1 || VWP > 32)
        begin : g_bad_param
            softmax_online_acc_parameter_outside_domain u_bad ();
        end
    endgenerate

    // ---- E6 / E7: the alpha ring (2 slots) and the log l table, one register per entry ------------------------------
    reg            la_ws;                               // slot of the alpha unit being received (unit mod 2)
    reg  [LS-1:0]  la_wr, ll_wr;                        // the next row of the unit being received
    reg  [1:0]     la_ok;                               // the slot holds a complete unit not yet used up
    reg            ll_ok;
    reg            la_rs;                               // slot of the unit the next rescale job uses
    wire [2*S*9-1:0] la_flat;
    wire [S*8-1:0]   ll_flat;
    genvar ga;
    generate
        for (ga = 0; ga < 2 * S; ga = ga + 1) begin : g_la
            reg [8:0] v;
            always @(posedge clk) if (la_v && {la_ws, la_wr} == ga) v <= la_d;
            assign la_flat[ga*9 +: 9] = v;
        end
        for (ga = 0; ga < S; ga = ga + 1) begin : g_ll
            reg [7:0] v;
            always @(posedge clk) if (ll_v && ll_wr == ga) v <= ll_d;
            assign ll_flat[ga*8 +: 8] = v;
        end
    endgenerate

    // ---- o_part FIFO state, as one-hot vectors over the FIFOs (window p of a tile is in FIFO p mod P_CH; the LOG job
    // of wave w reads windows w*N_OC + i on crossbar i): nb1 / rb1 / cb1 = the FIFO of crossbar 0 for the next job /
    // the job being read / the compute stage; need_l / need_r = the FIFOs the next job / the job being read uses
    wire            fifo_ok, fifo_err;
    reg  [P_CH-1:0] nb1, rb1, cb1, need_l, need_r;
    wire [P_CH-1:0] m_first, m_last, nb1_rot, need_rot;
    wire [P_CH*5*VWP-1:0] fq;                           // FIFO read data (one cycle after the pop)

    // ---- run state: job indices, drained counts ----------------------------------------------------------------
    reg  [JW-1:0]  lj, ej;                              // next LOG / EXP job (linear index in the run)
    reg  [WSW-1:0] lw, ew;                              // its wave
    reg  [JW-1:0]  lb;                                  // the LOG job's block
    reg  [JW-1:0]  ld_cnt, ed_cnt;                      // LOG / EXP jobs drained this run
    wire           e_epi = (ej >= NRS);

    // LOG / EXP engines (one per group; leader = crossbar 0 of the group)
    wire           lg_ready, lg_full, ex_ready, ex_full;
    wire           lg_job = (lj < NJ) && fifo_ok && (lb == 0 || ed_cnt + W > lj);
    wire           ex_job = (ej < NJ) && (ld_cnt > ej) && (e_epi ? ll_ok : la_ok[la_rs]);
    wire [LTW-1:0] lg_tag = {lw == W - 1, lb != 0, lw};
    wire [ETW-1:0] ex_tag = {ew == W - 1, la_rs, e_epi, ew};

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
    softmax_online_acc_engine #(.LW(LW), .QW(QW), .TAGW(LTW)) u_lg (
        .clk(clk), .reset(reset), .job_ready(lg_job), .job_tag(lg_tag),
        .leader_ready(lg_ready), .leader_full(lg_full), .busy(lg_busy), .rc(lg_rcq), .tag_q(lg_tq), .decide(lg_dec),
        .rd_v(lg_rv), .rd_c(lg_rc), .rd_tag(lg_rtag), .cs_v(lg_cv), .cs_c(lg_cc), .cs_tag(lg_ctag),
        .st_v(lg_sv), .st_tag(lg_stag), .drn_v(lg_drv), .drn_q(lg_dq), .drn_last(lg_dlast), .drn_tag(lg_dtag));
    softmax_online_acc_engine #(.LW(LW), .QW(QW), .TAGW(ETW)) u_ex (
        .clk(clk), .reset(reset), .job_ready(ex_job), .job_tag(ex_tag),
        .leader_ready(ex_ready), .leader_full(ex_full), .busy(ex_busy), .rc(ex_rcq), .tag_q(ex_tq), .decide(ex_dec),
        .rd_v(ex_rv), .rd_c(ex_rc), .rd_tag(ex_rtag), .cs_v(ex_cv), .cs_c(ex_cc), .cs_tag(ex_ctag),
        .st_v(ex_sv), .st_tag(ex_stag), .drn_v(ex_drv), .drn_q(ex_dq), .drn_last(ex_dlast), .drn_tag(ex_dtag));

    // ---- o_part FIFOs ---------------------------------------------------------------------------------------------
    wire [P_CH-1:0] fpop, f_ok, f_ovf, f_unf;
    genvar gf;
    generate
        for (gf = 0; gf < P_CH; gf = gf + 1) begin : g_f
            wire [FAW:0] cnt;
            att_fifo #(.W(5*VWP), .DEPTH(FD)) u_f (
                .clk(clk), .reset(reset), .wr_en(p_v[gf]), .wr_data(p_d[gf*5*VWP +: 5*VWP]),
                .rd_en(fpop[gf]), .rd_data(fq[gf*5*VWP +: 5*VWP]), .count(cnt),
                .overflow(f_ovf[gf]), .underflow(f_unf[gf]));
            assign f_ok[gf] = !need_l[gf] || (cnt >= LW);                   // a whole window in every FIFO it uses
            assign fpop[gf] = (lg_dec && need_l[gf]) || (lg_busy && need_r[gf]);
            // constant masks: wave 0 reads FIFOs 0 .. n - 1; the last wave LWC FIFOs from ((W-1) N_OC) mod P_CH
            localparam integer OL = (gf + P_CH - ((W - 1) * N_OC) % P_CH) % P_CH;   // its crossbar in the last wave
            assign m_first[gf]  = (gf < ((W == 1) ? LWC : N_OC));
            assign m_last[gf]   = (OL < LWC);
            assign nb1_rot[gf]  = nb1[(gf + P_CH - NRT) % P_CH];             // the next wave: N_OC FIFOs further
            assign need_rot[gf] = need_l[(gf + P_CH - NRT) % P_CH];
        end
    endgenerate
    assign fifo_ok  = &f_ok;
    assign fifo_err = |f_ovf || |f_unf;                 // TB probe

    // EXP drain pipeline: drain word -> the rescale's e into e_mem, the epilogue's O8 to E9
    reg            p1_e;                                // an EXP drain word
    reg            p1_v;                                // ... of the epilogue (O8)
    reg  [WSW-1:0] p1_w;
    reg  [QW-1:0]  p1_q;
    reg            o8_last;                             // the run's last O8 word is on E9: the run ends

    // the FIFO sets advance with the LOG job pointer: the next wave's sets are the current ones rotated by N_OC, wave 0
    // and the last wave's mask are constants
    wire [WSW-1:0] lw_n = (lw == W - 1) ? {WSW{1'b0}} : lw + 1'b1;          // the next LOG job's wave
    always @(posedge clk) begin
        if (reset || o8_last) begin nb1 <= 1; need_l <= m_first; end
        else if (lg_dec) begin
            nb1    <= (lw_n == 0) ? 1 : nb1_rot;
            need_l <= (lw_n == 0) ? m_first : ((lw_n == W - 1) ? m_last : need_rot);
        end
        if (lg_dec) begin rb1 <= nb1; need_r <= need_l; end
        cb1 <= lg_dec ? nb1 : rb1;
    end
    always @(posedge clk) begin
        if (reset || o8_last) begin
            lj <= 0; lw <= 0; lb <= 0; ej <= 0; ew <= 0;
            ld_cnt <= 0; ed_cnt <= 0;
        end
        else begin
            // job pointers advance at the decision
            if (lg_dec) begin
                lj <= lj + 1'b1;
                if (lw == W - 1) begin lw <= 0; lb <= lb + 1'b1; end else lw <= lw + 1'b1;
            end
            if (ex_dec) begin
                ej <= ej + 1'b1;
                if (ew == W - 1) ew <= 0; else ew <= ew + 1'b1;
            end
            // drained counts (visible the next cycle)
            if (lg_dlast) ld_cnt <= ld_cnt + 1'b1;
            if (ex_dlast) ed_cnt <= ed_cnt + 1'b1;
        end
    end

    // inputs and credits (kept across runs: the next run's units may arrive before the run ends)
    wire la_end = la_v && la_wr == S - 1;               // an alpha unit's last row
    wire ll_end = ll_v && ll_wr == S - 1;
    wire la_use = ex_cv && ex_cc == LW - 1 && ex_ctag[ETW-1] && !ex_ctag[WSW];   // last read of the unit's last rescale
    wire ll_use = ex_cv && ex_cc == LW - 1 && ex_ctag[ETW-1] && ex_ctag[WSW];    // ... of the run's last epilogue job
    always @(posedge clk) begin
        if (reset) begin
            la_ws <= 1'b0; la_wr <= 0; ll_wr <= 0; la_ok <= 2'b00; ll_ok <= 1'b0; la_rs <= 1'b0;
            p_cr <= 1'b0; la_cr <= 1'b0; ll_cr <= 1'b0;
        end
        else begin
            if (la_v) la_wr <= la_wr + 1'b1;            // S a power of two: wraps after row S-1
            if (la_end) begin la_ws <= ~la_ws; la_ok[la_ws] <= 1'b1; end
            if (la_use) la_ok[ex_ctag[WSW+1]] <= 1'b0;
            if (ll_v) ll_wr <= ll_wr + 1'b1;
            if (ll_end) ll_ok <= 1'b1;
            if (ll_use) ll_ok <= 1'b0;
            if (ex_dec && !e_epi && ew == W - 1) la_rs <= ~la_rs;
            // o_part released after the block's last read; an alpha / log l unit after its last use
            p_cr  <= lg_rv && lg_rc == LW - 1 && lg_rtag[LTW-1];
            la_cr <= la_use;
            ll_cr <= ll_use;
        end
    end
    always @(posedge clk) o8_last <= !reset && p1_v && p1_w == W - 1 && p1_q == LW - 1;

    // ---- lane rows within a window: element offset of the word read this cycle (shared by every crossbar): lane j of
    // word c is element 5c + j of its window, row (5c + j) >> LD. Lanes past the window (last word) need no mask: the
    // I x I pass never loads rows >= I.
    genvar gj, gr, gv;
    wire [EPS*(LI+4)-1:0] e_all;
    generate
        for (gj = 0; gj < EPS; gj = gj + 1) begin : g_lane
            reg [LI+3:0] e;                                     // offset of the running word's lane j
            always @(posedge clk) e <= ex_dec ? (gj + 5) : e + 5;
            assign e_all[gj*(LI+4) +: LI+4] = e;
        end
    endgenerate

    // ---- per crossbar index i: slot memories, the EXP and LOG crossbars ---------------------------------------------
    wire [N_OC-1:0]    lg_rdy_all, lg_full_all, ex_rdy_all, ex_full_all;
    wire [N_OC*40-1:0] lg_out_all;
    wire [N_OC*40-1:0] o8_all;                          // the epilogue's O8 word of every crossbar (E9)
    assign lg_ready = lg_rdy_all[0];
    assign lg_full  = lg_full_all[0];
    assign ex_ready = ex_rdy_all[0];
    assign ex_full  = ex_full_all[0];
    genvar gi;
    generate
        for (gi = 0; gi < N_OC; gi = gi + 1) begin : g_x
            // window of this crossbar in wave w: w*N_OC + gi (present iff w < W-1 or gi < LWC)
            wire act_l = (lg_stag[WSW-1:0] != W - 1) || (gi < LWC);
            wire act_e = (ex_stag[WSW-1:0] != W - 1) || (gi < LWC);
            // slot memories {w, q}: lo (LOG drain -> EXP feed), e (rescale drain -> LOG feed), so (LOG feed -> LOG feed,
            // epilogue drain)
            reg [39:0] lo_mem [0:(W << QW) - 1];
            reg [39:0] e_mem  [0:(W << QW) - 1];
            reg [4:0]  so_mem [0:(W << QW) - 1];
            reg [39:0] lo_rd, e_rd;
            reg [4:0]  so_rd, so_ep;
            wire [AW-1:0] lg_ra = {lg_rtag[WSW-1:0], lg_rc};
            wire [AW-1:0] ex_ra = {ex_rtag[WSW-1:0], ex_rc};
            // o_part word of this crossbar's window in the compute stage: FIFO (that of cb1) + gi mod P_CH, one-hot
            wire [P_CH-1:0]  csel;
            for (gv = 0; gv < P_CH; gv = gv + 1) begin : g_cs
                assign csel[gv] = cb1[(gv + P_CH - gi % P_CH) % P_CH];
            end
            reg  [5*VWP-1:0] opw;                                   // AND-OR select (no runtime product)
            integer          fsi;
            always @(*) begin
                opw = {(5*VWP){1'b0}};
                for (fsi = 0; fsi < P_CH; fsi = fsi + 1) opw = opw | ({(5*VWP){csel[fsi]}} & fq[fsi*5*VWP +: 5*VWP]);
            end
            // LOG feed: o' = (b ? (+-e) << SO : 0) + o_part -> oq, so (OW bits hold every o')
            wire [39:0] oq_w;
            wire [4:0]  so_w;
            for (gj = 0; gj < EPS; gj = gj + 1) begin : g_lf
                wire        [VWP-1:0] ox = opw[gj*VWP +: VWP];
                wire signed [OW-1:0]  op = {{(OW-VWP){ox[VWP-1]}}, ox};
                wire signed [OW-1:0]  ev = so_rd[gj] ? -$signed({1'b0, e_rd[gj*8 +: 8]}) : $signed({1'b0, e_rd[gj*8 +: 8]});
                wire signed [OW-1:0]  on = (lg_ctag[WSW] ? (ev <<< SO) : $signed({OW{1'b0}})) + op;
                wire        [OW-1:0]  mg = on[OW-1] ? -on : on;
                wire        [OW-1:0]  sq = mg >> SO;
                assign oq_w[gj*8 +: 8] = (sq > 127) ? 8'd127 : sq[7:0];
                assign so_w[gj] = on[OW-1];
            end
            // EXP x vector of this crossbar's RPW rows for the starting job (rescale: log alpha of its slot, epilogue:
            // -log l), chosen at the decision (a W:1 mux per row) and held
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
                e_rd  <= e_mem[lg_ra];
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
            // the rescale's e -> e_mem one cycle after the drain; the epilogue's O8 -> E9 word gi (zeros without a window
            // in the last wave)
            always @(posedge clk)
                if (p1_e && !p1_v) e_mem[{p1_w, p1_q}] <= ee_q;
            assign o8_all[gi*40 +: 40] = ((p1_w != W - 1) || (gi < LWC)) ? o8_l : 40'd0;
        end
    endgenerate

    always @(posedge clk) begin
        p1_e <= !reset && ex_drv;
        p1_v <= !reset && ex_drv && ex_dtag[WSW];
        p1_w <= ex_dtag[WSW-1:0];
        p1_q <= ex_dq;
    end

    // ---- E9: the epilogue's O8 words, registered (two cycles after the drain); run_done one cycle after the run's end -
    always @(posedge clk) begin
        o_v      <= !reset && p1_v;
        o_d      <= o8_all;
        run_done <= !reset && o8_last;
    end

    // probes (TB): LOG drain words (lo)
    wire               lg_dv = lg_drv;
    wire [WSW-1:0]     lg_dw = lg_dtag[WSW-1:0];
    wire [N_OC*40-1:0] lg_dout = lg_out_all;

endmodule


// ============================================================================
// softmax_online_acc_engine — one lock-step crossbar group's window engine (v2's att_oc_engine, unchanged). A job
// starts (decide) when the engine is idle (no word in its read / compute / strobe stages), the job is ready and the
// leader's MSB_SA_Ready is high; word c of the job is read at decide + c (rd_*), its memory data are valid the next cycle
// (cs_*), the registered feed word is strobed the cycle after (st_*). Drains are in order: the leader's reg_full marks a
// gapless burst of LW words (drn_*), tagged from a depth-4 job FIFO.
// ============================================================================
module softmax_online_acc_engine #(
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
    // a new job only once the previous burst has fully strobed (by then the leader's MSB_SA_Ready has dropped; it rises
    // again after that window's MSB fire)
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
