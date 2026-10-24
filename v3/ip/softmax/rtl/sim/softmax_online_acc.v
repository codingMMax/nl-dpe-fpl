// ============================================================================
// softmax_online_acc.v — the online softmax's accumulator (v3, ip/softmax/acc_spec.md, the bit-plane pass).
// SIMULATION twin: the behavior `dpe` (ip/xbar/rtl/sim/dpe_nldpe.v) in the two crossbar generate loops, weights through
// the sim-only ports weight_in / weight_en (the testbench preloads them instead, PLAN R8). VTR twin
// ip/softmax/rtl/vtr/softmax_online_acc.v (bare `dpe`, dpe_blackbox.v). Model: simulator/models/softmaxAccumulator.py.
//
// Rewritten 2026-10-23 for the bit-plane pass (spec §10 item 8) from the pre-bit-plane module (git show
// 8acf73c:v3/ip/softmax/rtl/sim/softmax_online_acc.v). Kept: the two crossbar groups and their job order and
// readiness, the o_part FIFOs and their one-hot FIFO sets, the alpha ring and log l table logic, the credits, E9 and
// run_done. New: the crossbar ports carry plane words; a job decides PRE cycles before its group's leader is ready;
// the LOG feed corner-turns its e and so planes, computes at Q_W = I/8 lanes and corner-turns oq back into planes; the
// EXP feeds add bit-serially into an 8-plane hold with clamp flags; lo_mem and e_mem hold plane words, so_mem one plane
// per window; E9 is one plane window per epilogue wave. The trims of 2026-10-23 (spec §10 item 10): E8 carries value
// words, popped straight into the arithmetic; corner-turns A and B and the EXP hold are single buffers.
//
// Element e = m*D + c (row-major tile), window p = e div I (P_OC = S*D/I windows of RPW = I/D rows), wave
// w = p div N_OC, crossbar i = p mod N_OC. Two groups of N_OC identity crossbars (extent I x I), each lock-step from
// its crossbar 0:
//   EXP group, jobs per run: rescale (b, w), b = 1..NBLK-1, then the epilogue (w)
//     rescale  feed t  = clamp8(log_a[m] + lo)     drain e  -> e_mem (slot w)
//     epilogue feed tE = clamp8(lo - log_l[m])     drain e_E -> 8-plane hold -> O8 = sat8(+-(e * 2^(SO-LN))) -> E9
//   LOG group, jobs (b, w), b = 0..NBLK-1
//     feed o' = (b ? (+-e) << SO : 0) + o_part, oq = min(|o'| >> SO, 127), so <- (o' < 0); drain lo -> lo_mem
// Job timing (spec §5), a job decided at d:
//   LOG  reads d .. d+7 (e_mem, so_mem); corner-turn A whole at d+9; pops its N_OC FIFOs' o_part words d+8 .. d+15,
//        each word straight into the arithmetic at Q_W lanes d+9 .. d+16 into corner-turn B; the burst d+17 .. d+24
//        (PRE_LOG 17); the new so into so_mem at d+17; p_cr at d+16 for a block's last job
//   EXP  reads d .. d+7 (lo_mem); the bit-serial sums held d+1 .. d+8; the clamp flags at d+9; the burst d+10 ..
//        d+17 (PRE_EXP 10)
//   a group decides its next job at >= d + 16 (LOG_GAP) or d + 17 (EXP_GAP): one buffer per feed stage
//   the epilogue's drain planes at ds .. ds+7 into the O8 hold; the saturation flags at ds+8; its O8 planes registered
//   onto E9 at ds+10 .. ds+17 (E9_LAT 10)
// Readiness: LOG (b, w): its windows complete in their FIFOs; b >= 1: rescale (b, w) drained.
//            rescale (b, w): its alpha unit present; LOG (b-1, w) drained.
//            epilogue (w): log l present; LOG (NBLK-1, w) drained.
// so_mem has one read port: the LOG feed reads it at its decision; the epilogue reads its window's final so at drain
// plane 6 (or 7 when the LOG feed has the port), well before its first O8 plane.
// Domain (elaboration guards): S, D, I, P_CH powers of two, 8 <= S <= I, D <= I, I | S*D, 1 <= N_OC <= P_OC,
// N_OC <= P_CH, P_CH | P_OC, NBLK >= 1, P = 8, SO, LN >= 0, 1 <= VWP <= 16.
// ============================================================================

`timescale 1ns / 1ps

module softmax_online_acc #(
    parameter S    = 16,
    parameter D    = 8,
    parameter R    = 64,
    parameter C    = 64,
    parameter P    = 8,
    parameter N_OC = 2,
    parameter NBLK = 2,
    parameter SO   = 4,
    parameter LN   = 4,
    parameter P_CH = 2,
    parameter VWP  = 13,
    // derived (port widths; do not override)
    parameter I    = (R < C) ? R : C,
    parameter L_P  = (VWP + 7) / 8                      // lanes of an E8 plane word
) (
    input  wire                  clk,
    input  wire                  reset,
    // weight programming (sim only): R*C strobes broadcast to every crossbar
    input  wire [7:0]            weight_in,
    input  wire                  weight_en,
    // E8 o_part (the PV DIMM's output stream)
    input  wire [P_CH-1:0]       p_v,
    input  wire [P_CH*L_P*I-1:0] p_d,
    output reg                   p_cr,
    // E6 log alpha (9 bit: lane 1 carries bit 8), E7 log l
    input  wire                  la_v,
    input  wire [2*S-1:0]        la_d,
    output reg                   la_cr,
    input  wire                  ll_v,
    input  wire [S-1:0]          ll_d,
    output reg                   ll_cr,
    // E9 the final O8 planes: crossbar i at bits i*I +: I
    output reg                   o_v,
    output reg  [N_OC*I-1:0]     o_d,
    output reg                   run_done
);

    localparam P_OC = (S * D) / I;                      // windows per run
    localparam W    = (P_OC + N_OC - 1) / N_OC;         // waves
    localparam LS   = $clog2(S);
    localparam WSW  = (W > 1) ? $clog2(W) : 1;
    localparam PW   = $clog2(P);                        // plane index bits
    localparam NJ   = NBLK * W;                         // jobs per group per run
    localparam JW   = $clog2(NJ + 1) + 1;
    localparam MAW  = WSW + PW;                         // lo_mem / e_mem address {w, plane}
    localparam LWC  = P_OC - (W - 1) * N_OC;            // windows of the last wave
    localparam SHL  = (SO > LN) ? SO - LN : 0;          // epilogue scale 2^(SO-LN)
    localparam SHR  = (SO < LN) ? LN - SO : 0;
    localparam RPW  = I / D;                            // rows per window
    localparam LD   = $clog2(D);
    localparam Q_W  = I / 8;                            // the LOG feed's lanes
    localparam NRS  = (NBLK - 1) * W;                   // rescale jobs per run
    localparam ETW  = WSW + 3;                          // EXP tag {lastw, slot, epi, w}
    localparam LTW  = WSW + 2;                          // LOG tag {lastw, bnz, w}
    localparam NRT  = N_OC % P_CH;                      // FIFO rotation from one wave to the next
    localparam OW   = ((9 + SO > VWP) ? 9 + SO : VWP) + 1;   // o' bits: |(+-e) << SO| + |o_part| < 2^(OW-1)
    localparam TWP  = P_OC / P_CH;                      // windows per channel per tile
    localparam FD   = 1 << $clog2(2 * TWP * P);         // FIFO depth: two tiles of o_part planes
    localparam FAW  = $clog2(FD);
    localparam PL   = L_P * I;                          // an E8 plane word
    localparam EEW  = 8 + SHL;                          // e_E bits

    generate
        if ((S & (S - 1)) != 0 || (D & (D - 1)) != 0 || (I & (I - 1)) != 0 || (P_CH & (P_CH - 1)) != 0
            || S < 8 || S > I || D > I || ((S * D) % I) != 0 || N_OC < 1 || N_OC > P_OC || N_OC > P_CH
            || (P_OC % P_CH) != 0 || NBLK < 1 || P != 8 || SO < 0 || LN < 0 || VWP < 1 || VWP > 16)
        begin : g_bad_param
            softmax_online_acc_parameter_outside_domain u_bad ();
        end
    endgenerate

    // ---- E6 / E7: the alpha ring (2 slots) and the log l table, written plane by plane ------------------------------
    reg            la_ws;                               // slot of the alpha unit being received (unit mod 2)
    reg  [PW-1:0]  la_pc, ll_pc;                        // the plane of the unit being received
    reg  [1:0]     la_ok;                               // the slot holds a complete unit not yet used up
    reg            ll_ok;
    reg            la_rs;                               // slot of the unit the next rescale job uses
    reg  [2*S*9-1:0] la_flat;                           // slot s, row m at ((s*S + m)*9) +: 9
    reg  [S*8-1:0]   ll_flat;
    genvar ga;
    generate
        for (ga = 0; ga < 2 * S; ga = ga + 1) begin : g_la
            localparam integer SL = ga / S, RW = ga % S;
            always @(posedge clk)
                if (la_v && la_ws == SL) begin
                    la_flat[ga*9 +: 8] <= {la_d[RW], la_flat[ga*9 + 1 +: 7]};      // bits 0 .. 7, LSB plane first
                    if (la_pc == 0) la_flat[ga*9 + 8] <= la_d[S + RW];              // bit 8: lane 1 of plane 0
                end
        end
        for (ga = 0; ga < S; ga = ga + 1) begin : g_ll
            always @(posedge clk)
                if (ll_v) ll_flat[ga*8 +: 8] <= {ll_d[ga], ll_flat[ga*8 + 1 +: 7]};
        end
    endgenerate

    // ---- o_part FIFO state, as one-hot vectors over the FIFOs (window p of a tile is in FIFO p mod P_CH; the LOG job
    // of wave w reads windows w*N_OC + i on crossbar i): nb1 / rb1 = the FIFO of crossbar 0 for the next job / the job
    // being read; need_l / need_r = the FIFOs the next job / the job being read uses
    wire            fifo_ok, fifo_err;
    reg  [P_CH-1:0] nb1, rb1, need_l, need_r;
    wire [P_CH-1:0] m_first, m_last, nb1_rot, need_rot;
    wire [P_CH*PL-1:0] fq;                              // FIFO read data (one cycle after the pop)

    // ---- run state: job indices, drained counts ----------------------------------------------------------------
    reg  [JW-1:0]  lj, ej;                              // next LOG / EXP job (linear index in the run)
    reg  [WSW-1:0] lw, ew;                              // its wave
    reg  [JW-1:0]  lb;                                  // the LOG job's block
    reg  [JW-1:0]  ld_cnt, ed_cnt;                      // LOG / EXP jobs drained this run
    wire           e_epi = (ej >= NRS);
    reg  [4:0]     lcool, ecool;                        // cycles until the group may decide again (LOG_GAP, EXP_GAP)
    wire           lg_job = (lj < NJ) && fifo_ok && (lb == 0 || ed_cnt + W > lj);
    wire           ex_job = (ej < NJ) && (ld_cnt > ej) && (e_epi ? ll_ok : la_ok[la_rs]);
    wire           lg_dec = lg_job && (lcool == 0);
    wire           ex_dec = ex_job && (ecool == 0);
    wire [LTW-1:0] lg_tag = {lw == W - 1, lb != 0, lw};
    wire [ETW-1:0] ex_tag = {ew == W - 1, la_rs, e_epi, ew};

    // ---- the LOG group's phases: R (e, so reads, d .. d+7), F (fill A, d+1 .. d+8), the o_part word pops
    //      (d+8 .. d+15), G (arithmetic, d+9 .. d+16), B (burst, d+17 .. d+24); one buffer each, so the next decision
    //      waits LOG_GAP = 16; the drain follows the leader's reg_full, its tag handed over at the burst's end ---------
    reg            lr_on, lf_v, lgv, lbv;
    reg  [PW-1:0]  lr_c, lf_c, lg_c, lb_c;
    reg  [LTW-1:0] lr_tag, lf_tag, lg_tg, lb_tag, ld_tag;
    wire           l_rv   = lg_dec || lr_on;
    wire [PW-1:0]  l_rc   = lg_dec ? {PW{1'b0}} : lr_c;
    wire [LTW-1:0] l_rtag = lg_dec ? lg_tag : lr_tag;
    wire           l_pop  = (lf_v && lf_c == P - 1) || (lgv && lg_c != P - 1);   // the o_part words: d+8 .. d+15
    wire           lg_so_rd = l_rv && (l_rc == 0) && l_rtag[WSW];       // the so plane read (b >= 1)
    wire           lg_full, lg_done;                    // the leader's drain plane valid, its drain end
    reg  [PW-1:0]  ldc;                                 // the LOG drain's plane
    wire           l_dlast = lg_full && (ldc == P - 1);
    always @(posedge clk) begin
        if (reset) begin
            lr_on <= 1'b0; lr_c <= 0; lf_v <= 1'b0; lgv <= 1'b0; lg_c <= 0; lbv <= 1'b0; lb_c <= 0; ldc <= 0;
            lcool <= 0;
        end
        else begin
            lcool <= lg_dec ? 2 * P - 1 : ((lcool != 0) ? lcool - 1'b1 : 5'd0);
            // R
            if (l_rv) begin
                lr_on <= (l_rc != P - 1);
                lr_c  <= l_rc + 1'b1;
            end
            // F = R one cycle later
            lf_v <= l_rv;
            // G after F's last plane
            if (lf_v && lf_c == P - 1) begin lgv <= 1'b1; lg_c <= 0; end
            else if (lgv) begin
                if (lg_c == P - 1) lgv <= 1'b0;
                lg_c <= lg_c + 1'b1;
            end
            // B after G's last group
            if (lgv && lg_c == P - 1) begin lbv <= 1'b1; lb_c <= 0; end
            else if (lbv) begin
                if (lb_c == P - 1) lbv <= 1'b0;
                lb_c <= lb_c + 1'b1;
            end
            if (lg_full) ldc <= ldc + 1'b1;
        end
        if (l_rv) lr_tag <= l_rtag;
        lf_c <= l_rc; lf_tag <= l_rtag;
        if (lf_v && lf_c == P - 1) lg_tg <= lf_tag;
        if (lgv && lg_c == P - 1) lb_tag <= lg_tg;
        if (lbv && lb_c == P - 1) ld_tag <= lb_tag;
    end

    // ---- the EXP group's phases: R (lo reads, d .. d+7), F (bit-serial sums, d+1 .. d+8), the clamp flags (d+9),
    //      B (burst, d+10 .. d+17); the drain follows the leader; the epilogue's O8 stage after its drain -------------
    reg            er_on, ef_v, efl, ebv;
    reg  [PW-1:0]  er_c, ef_c, eb_c;
    reg  [ETW-1:0] er_tag, ef_tag, efl_tag, eb_tag, ed_tag, eo_tag;
    wire           e_rv   = ex_dec || er_on;
    wire [PW-1:0]  e_rc   = ex_dec ? {PW{1'b0}} : er_c;
    wire [ETW-1:0] e_rtag = ex_dec ? ex_tag : er_tag;
    wire           ex_full, ex_done;
    reg  [PW-1:0]  edc;                                 // the EXP drain's plane
    wire           e_dlast = ex_full && (edc == P - 1);
    reg            eo_f;                                // the O8 flags this cycle (ds + 8)
    reg            eo_v;                                // an O8 plane computed this cycle (ds + 9 .. ds + 16)
    reg  [PW-1:0]  eo_c;
    always @(posedge clk) begin
        if (reset) begin
            er_on <= 1'b0; er_c <= 0; ef_v <= 1'b0; efl <= 1'b0; ebv <= 1'b0; eb_c <= 0; edc <= 0;
            ecool <= 0; eo_f <= 1'b0; eo_v <= 1'b0; eo_c <= 0;
        end
        else begin
            ecool <= ex_dec ? 2 * P : ((ecool != 0) ? ecool - 1'b1 : 5'd0);   // EXP_GAP 17: the hold bursts to d+17
            if (e_rv) begin
                er_on <= (e_rc != P - 1);
                er_c  <= e_rc + 1'b1;
            end
            ef_v <= e_rv;
            efl  <= ef_v && (ef_c == P - 1);
            if (efl) begin ebv <= 1'b1; eb_c <= 0; end
            else if (ebv) begin
                if (eb_c == P - 1) ebv <= 1'b0;
                eb_c <= eb_c + 1'b1;
            end
            if (ex_full) edc <= edc + 1'b1;
            eo_f <= e_dlast && ed_tag[WSW];
            if (eo_f) begin eo_v <= 1'b1; eo_c <= 0; end
            else if (eo_v) begin
                if (eo_c == P - 1) eo_v <= 1'b0;
                eo_c <= eo_c + 1'b1;
            end
        end
        if (e_rv) er_tag <= e_rtag;
        ef_c <= e_rc; ef_tag <= e_rtag;
        if (ef_v && ef_c == P - 1) efl_tag <= ef_tag;
        if (efl) eb_tag <= efl_tag;
        if (ebv && eb_c == P - 1) ed_tag <= eb_tag;
        if (e_dlast) eo_tag <= ed_tag;
    end

    // ---- so_mem's read port: the LOG feed at its decision, the epilogue at its drain plane 6 (or 7) ----------------
    reg  ep_pend, ep_rd_q;
    wire ep_req = (ex_full && edc == 6 && ed_tag[WSW]) || ep_pend;
    wire ep_rd  = ep_req && !lg_so_rd;
    always @(posedge clk) begin
        ep_pend <= !reset && ep_req && lg_so_rd;
        ep_rd_q <= !reset && ep_rd;
    end
    wire [WSW-1:0] so_ra = lg_so_rd ? l_rtag[WSW-1:0] : ed_tag[WSW-1:0];

    // ---- o_part FIFOs ---------------------------------------------------------------------------------------------
    wire [P_CH-1:0] fpop, f_ok, f_ovf, f_unf;
    genvar gf;
    generate
        for (gf = 0; gf < P_CH; gf = gf + 1) begin : g_f
            wire [FAW:0] cnt;
            att_fifo #(.W(PL), .DEPTH(FD)) u_f (
                .clk(clk), .reset(reset), .wr_en(p_v[gf]), .wr_data(p_d[gf*PL +: PL]),
                .rd_en(fpop[gf]), .rd_data(fq[gf*PL +: PL]), .count(cnt),
                .overflow(f_ovf[gf]), .underflow(f_unf[gf]));
            assign f_ok[gf] = !need_l[gf] || (cnt >= P);                    // a whole window in every FIFO it uses
            assign fpop[gf] = l_pop && need_r[gf];                           // the job's 8 words, d+8 .. d+15
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

    reg o8_last;                                        // the run's last O8 plane is on E9: the run ends
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
            if (l_dlast) ld_cnt <= ld_cnt + 1'b1;
            if (e_dlast) ed_cnt <= ed_cnt + 1'b1;
        end
    end

    // inputs and credits (kept across runs: the next run's units may arrive before the run ends)
    wire la_end = la_v && la_pc == P - 1;               // an alpha unit's last plane
    wire ll_end = ll_v && ll_pc == P - 1;
    wire la_use = ef_v && ef_c == P - 1 && ef_tag[ETW-1] && !ef_tag[WSW];   // last lo read + 1 of the unit's last rescale
    wire ll_use = ef_v && ef_c == P - 1 && ef_tag[ETW-1] && ef_tag[WSW];    // ... of the run's last epilogue job
    always @(posedge clk) begin
        if (reset) begin
            la_ws <= 1'b0; la_pc <= 0; ll_pc <= 0; la_ok <= 2'b00; ll_ok <= 1'b0; la_rs <= 1'b0;
            p_cr <= 1'b0; la_cr <= 1'b0; ll_cr <= 1'b0;
        end
        else begin
            if (la_v) la_pc <= la_pc + 1'b1;            // P a power of two: wraps after plane P-1
            if (la_end) begin la_ws <= ~la_ws; la_ok[la_ws] <= 1'b1; end
            if (la_use) la_ok[ef_tag[WSW+1]] <= 1'b0;
            if (ll_v) ll_pc <= ll_pc + 1'b1;
            if (ll_end) ll_ok <= 1'b1;
            if (ll_use) ll_ok <= 1'b0;
            if (ex_dec && !e_epi && ew == W - 1) la_rs <= ~la_rs;
            // o_part released after the block's last word read (d+15); an alpha / log l unit after its last use
            p_cr  <= lgv && lg_c == P - 2 && lg_tg[LTW-1];
            la_cr <= la_use;
            ll_cr <= ll_use;
        end
    end

    // ---- per crossbar index i: slot memories, the feeds, the LOG and EXP crossbars, the O8 stage --------------------
    wire [N_OC-1:0]    lg_rdy_all, lg_full_all, lg_done_all, ex_rdy_all, ex_full_all, ex_done_all;
    wire [N_OC*I-1:0]  o8_all;                          // the O8 plane of every crossbar (E9)
    assign lg_full = lg_full_all[0];
    assign lg_done = lg_done_all[0];
    assign ex_full = ex_full_all[0];
    assign ex_done = ex_done_all[0];
    genvar gi, gj, gr, gv, gb, gq;
    generate
        for (gi = 0; gi < N_OC; gi = gi + 1) begin : g_x
            // this crossbar's window in wave w: w*N_OC + gi (present iff w < W-1 or gi < LWC)
            wire act_l = (lb_tag[WSW-1:0] != W - 1) || (gi < LWC);
            wire act_e = (eb_tag[WSW-1:0] != W - 1) || (gi < LWC);
            wire act_o = (eo_tag[WSW-1:0] != W - 1) || (gi < LWC);
            // slot memories: lo (LOG drain -> EXP feed) and e (rescale drain -> LOG feed), plane words {w, plane};
            // so (LOG feed -> LOG feed, the epilogue), one plane per window
            reg [I-1:0] lo_mem [0:(W << PW) - 1];
            reg [I-1:0] e_mem  [0:(W << PW) - 1];
            reg [I-1:0] so_mem [0:W - 1];
            reg [I-1:0] lo_rq, e_rq, so_rq;
            always @(posedge clk) begin
                lo_rq <= lo_mem[{e_rtag[WSW-1:0], e_rc}];
                e_rq  <= e_mem[{l_rtag[WSW-1:0], l_rc}];
                so_rq <= so_mem[so_ra];
            end

            // ======== the LOG feed ========
            // the o_part word of this crossbar's window during the arithmetic: FIFO (that of rb1) + gi mod P_CH, one-hot
            wire [P_CH-1:0] csel;
            for (gv = 0; gv < P_CH; gv = gv + 1) begin : g_cs
                assign csel[gv] = rb1[(gv + P_CH - gi % P_CH) % P_CH];
            end
            reg  [PL-1:0] opl;                                      // AND-OR select (no runtime product)
            integer       fsi;
            always @(*) begin
                opl = {PL{1'b0}};
                for (fsi = 0; fsi < P_CH; fsi = fsi + 1) opl = opl | ({PL{csel[fsi]}} & fq[fsi*PL +: PL]);
            end
            // corner-turn A (one buffer): the e and so planes, filled one plane per cycle; during the arithmetic each
            // plane shifts by Q_W, so the group in use is at bits 0 .. Q_W-1. The group's o_part is the FIFO word read
            // for it (E8 value words: element j at bits 8*L_P*j)
            reg [P*I-1:0] ea;                                       // plane j at bits j*I +: I
            reg [I-1:0]   sa;
            integer aj;
            always @(posedge clk) begin
                for (aj = 0; aj < P; aj = aj + 1)
                    if (lf_v && lf_c == aj) ea[aj*I +: I] <= e_rq;
                    else if (lgv)           ea[aj*I +: I] <= ea[aj*I +: I] >> Q_W;
                if (lf_v && lf_c == 0) sa <= so_rq;
                else if (lgv)          sa <= sa >> Q_W;
            end
            // the group's Q_W elements: o' = (b ? (+-e) << SO : 0) + o_part -> oq, so
            wire [Q_W*8-1:0] oq_w;
            wire [Q_W-1:0]   so_w;
            for (gj = 0; gj < Q_W; gj = gj + 1) begin : g_lf
                wire [7:0] eb;
                for (gq = 0; gq < 8; gq = gq + 1) begin : g_eb
                    assign eb[gq] = ea[gq*I + gj];
                end
                wire        [VWP-1:0] ox = opl[gj*8*L_P +: VWP];
                wire signed [OW-1:0]  op = {{(OW-VWP){ox[VWP-1]}}, ox};
                wire signed [OW-1:0]  ev = sa[gj] ? -$signed({1'b0, eb}) : $signed({1'b0, eb});
                wire signed [OW-1:0]  on = (lg_tg[WSW] ? (ev <<< SO) : $signed({OW{1'b0}})) + op;
                wire        [OW-1:0]  mg = on[OW-1] ? -on : on;
                wire        [OW-1:0]  sq = mg >> SO;
                assign oq_w[gj*8 +: 8] = (sq > 127) ? 8'd127 : sq[7:0];
                assign so_w[gj] = on[OW-1];
            end
            // corner-turn B (one buffer of I bytes): group g's oq bytes at g*Q_W; the burst shifts one bit per plane.
            // The new so of the window, collected group by group, goes to so_mem at the burst's first cycle.
            reg  [I-1:0]   so_n;
            reg  [8*I-1:0] bq;
            integer q;
            always @(posedge clk) begin
                for (q = 0; q < P; q = q + 1)
                    if (lgv && lg_c == q) bq[q*Q_W*8 +: Q_W*8] <= oq_w;
                if (lbv) bq <= bq >> 1;
            end
            wire [I-1:0] lg_plane;
            for (gq = 0; gq < I; gq = gq + 1) begin : g_p
                assign lg_plane[gq] = bq[gq*8];
            end
            integer qs;
            always @(posedge clk) begin
                for (qs = 0; qs < P; qs = qs + 1)
                    if (lgv && lg_c == qs) so_n[qs*Q_W +: Q_W] <= so_w;
                if (lbv && lb_c == 0) so_mem[lb_tag[WSW-1:0]] <= so_n;
            end
            // LOG crossbar
            wire [R-1:0] lg_in = weight_en ? {{(R-8){1'b0}}, weight_in} : lg_plane;   // rows >= I read 0
            wire [C-1:0] lg_out;
            dpe #(
                .KERNEL_WIDTH(R), .NUM_COLS(C), .PRECISION(P), .ROWS_USED(I), .COLS_USED(I)
            ) u_lg (
                .clk(clk), .reset(reset), .data_in(lg_in), .nl_dpe_control(2'b11),   // LOG
                .shift_add_control(1'b0), .w_buf_en(lbv && act_l), .shift_add_bypass(1'b0),
                .load_output_reg(1'b0), .load_input_reg(weight_en),
                .MSB_SA_Ready(lg_rdy_all[gi]), .data_out(lg_out), .dpe_done(lg_done_all[gi]),
                .reg_full(lg_full_all[gi]));
            always @(posedge clk)
                if (lg_full) lo_mem[{ld_tag[WSW-1:0], ldc}] <= lg_out[I-1:0];

            // ======== the EXP feed ========
            // x of this crossbar's RPW rows for the starting job (rescale: log alpha of its slot; epilogue: ~log l, the
            // +1 of -log l is the adders' carry-in), chosen at the decision (a W:1 mux per row) and shifted per plane
            wire [RPW*9-1:0] xv_c;
            for (gr = 0; gr < RPW; gr = gr + 1) begin : g_xv
                wire [W*9-1:0] a0, a1, l9;
                for (gv = 0; gv < W; gv = gv + 1) begin : g_wv
                    localparam RWX = (gv * N_OC + gi) * RPW + gr;   // row of window gv*N_OC + gi
                    localparam RC  = (RWX < S) ? RWX : 0;
                    assign a0[gv*9 +: 9] = la_flat[RC*9 +: 9];
                    assign a1[gv*9 +: 9] = la_flat[(S + RC)*9 +: 9];
                    assign l9[gv*9 +: 9] = ~{ll_flat[RC*8+7], ll_flat[RC*8 +: 8]};
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
            reg [RPW*8-1:0] xq;                                     // bits 0 .. 7 of each row's x, shifted per plane
            reg [RPW-1:0]   x8;                                     // bit 8 (the sign)
            integer         xr;
            always @(posedge clk)
                for (xr = 0; xr < RPW; xr = xr + 1)
                    if (ex_dec) begin
                        xq[xr*8 +: 8] <= xv_c[xr*9 +: 8];
                        x8[xr]        <= xv_c[xr*9 + 8];
                    end
                    else if (ef_v) xq[xr*8 +: 8] <= xq[xr*8 +: 8] >> 1;
            // the bit-serial adders: element k takes the x of its row k >> log2 D
            wire [I-1:0] xb, xs8;
            for (gj = 0; gj < I; gj = gj + 1) begin : g_xb
                assign xb[gj]  = xq[(gj >> LD) * 8];
                assign xs8[gj] = x8[gj >> LD];
            end
            reg  [I-1:0] cy, lo7;
            wire [I-1:0] cin = (ef_c == 0) ? {I{ef_tag[WSW]}} : cy;    // the epilogue: +1 (-log l = ~log l + 1)
            wire [I-1:0] su  = lo_rq ^ xb ^ cin;
            always @(posedge clk)
                if (ef_v) begin
                    cy <= (lo_rq & xb) | (lo_rq & cin) | (xb & cin);
                    if (ef_c == P - 1) lo7 <= lo_rq;
                end
            // the 8-plane hold (one buffer: the next job's sums wait EXP_GAP) and the clamp flags of t = clamp8(lo + x):
            // the sum's bits 9, 8 from the carry out of plane 7 and the operands' signs
            reg [P*I-1:0] hq;                                       // plane j at bits j*I +: I
            integer hj;
            always @(posedge clk)
                for (hj = 0; hj < P; hj = hj + 1)
                    if (ef_v && ef_c == hj) hq[hj*I +: I] <= su;
                    else if (ebv)           hq[hj*I +: I] <= (hj == P - 1) ? {I{1'b0}} : hq[(hj+1)*I +: I];
            wire [I-1:0] s7 = hq[(P-1)*I +: I];
            wire [I-1:0] s8 = lo7 ^ xs8 ^ cy;
            wire [I-1:0] c9 = (lo7 & xs8) | (lo7 & cy) | (xs8 & cy);
            wire [I-1:0] s9 = lo7 ^ xs8 ^ c9;
            reg  [I-1:0] cpos, cneg;                                // t > 127, t < -128
            always @(posedge clk)
                if (efl) begin
                    cpos <= ~s9 & (s8 | s7);
                    cneg <= s9 & ~(s8 & s7);
                end
            wire [I-1:0] hb = hq[0 +: I];
            wire [I-1:0] ex_plane = (cpos & {I{eb_c != P - 1}}) | (cneg & {I{eb_c == P - 1}}) | (hb & ~cpos & ~cneg);
            // EXP crossbar
            wire [R-1:0] ex_in = weight_en ? {{(R-8){1'b0}}, weight_in} : ex_plane;   // rows >= I read 0
            wire [C-1:0] ex_out;
            dpe #(
                .KERNEL_WIDTH(R), .NUM_COLS(C), .PRECISION(P), .ROWS_USED(I), .COLS_USED(I)
            ) u_ex (
                .clk(clk), .reset(reset), .data_in(ex_in), .nl_dpe_control(2'b10),   // EXP
                .shift_add_control(1'b0), .w_buf_en(ebv && act_e), .shift_add_bypass(1'b0),
                .load_output_reg(1'b0), .load_input_reg(weight_en),
                .MSB_SA_Ready(ex_rdy_all[gi]), .data_out(ex_out), .dpe_done(ex_done_all[gi]),
                .reg_full(ex_full_all[gi]));

            // ======== the EXP drain: the rescale's e -> e_mem; the epilogue's e_E -> the O8 hold -> E9 ========
            reg [P*I-1:0] oh;                                       // plane j at bits j*I +: I
            integer jh;
            always @(posedge clk)
                if (ex_full) begin
                    if (!ed_tag[WSW]) e_mem[{ed_tag[WSW-1:0], edc}] <= ex_out[I-1:0];
                    for (jh = 0; jh < P; jh = jh + 1)
                        if (ed_tag[WSW] && edc == jh) oh[jh*I +: I] <= ex_out[I-1:0];
                end
            reg [I-1:0] so_ep;                                      // the window's final so (read at drain plane 6 / 7)
            always @(posedge clk) if (ep_rd_q) so_ep <= so_rq;
            // e_E = (e << SHL) >> SHR; the flags of O8 = so ? max(-e_E, -128) : min(e_E, 127) at ds + 8; O8 plane b at
            // ds + 9 + b (the negation bit-serial: ~x + 1, LSB first), registered onto E9. Plane b reads hold plane
            // b - SHL; with SHL > 0 (SO > LN) the next window's drain rewrites those planes from ds + 10, so the planes
            // then read a copy of the hold taken at ds + 8 (the head has SO = LN: no copy)
            wire [P*I-1:0] ohp;                                     // the hold the O8 planes read
            if (SHL > 0) begin : g_ohc
                reg [P*I-1:0] ohc;
                always @(posedge clk) if (eo_f) ohc <= oh;
                assign ohp = ohc;
            end else begin : g_ohd
                assign ohp = oh;
            end
            wire [I-1:0] gt127, gt128, ex_b;
            for (gj = 0; gj < I; gj = gj + 1) begin : g_o8
                wire [7:0] eb8, ep8;
                for (gq = 0; gq < 8; gq = gq + 1) begin : g_e
                    assign eb8[gq] = oh[gq*I + gj];
                    assign ep8[gq] = ohp[gq*I + gj];
                end
                wire [EEW-1:0] e0 = eb8;
                wire [EEW-1:0] ee = (e0 << SHL) >> SHR;
                wire [EEW-1:0] e1 = ep8;
                wire [EEW-1:0] ep = (e1 << SHL) >> SHR;
                assign gt127[gj] = (ee > 127);
                assign gt128[gj] = (ee > 128);
                wire [7:0] ep8s = ep[7:0];
                assign ex_b[gj] = ep8s[eo_c];
            end
            reg  [I-1:0] f127, f128, ncy;
            always @(posedge clk)
                if (eo_f) begin f127 <= gt127; f128 <= gt128; end
            wire [I-1:0] ncin = (eo_c == 0) ? {I{1'b1}} : ncy;
            wire [I-1:0] neg  = (f128 & {I{eo_c == P - 1}}) | (~f128 & (~ex_b ^ ncin));
            wire [I-1:0] pos  = (f127 & {I{eo_c != P - 1}}) | (~f127 & ex_b);
            reg  [I-1:0] o8q;
            always @(posedge clk) begin
                if (eo_v) ncy <= ~ex_b & ncin;
                o8q <= (eo_v && act_o) ? ((so_ep & neg) | (~so_ep & pos)) : {I{1'b0}};
            end
            assign o8_all[gi*I +: I] = o8q;
        end
    endgenerate

    // ---- E9: the epilogue's O8 planes (registered); run_done one cycle after the run's end -------------------------
    always @(posedge clk) begin
        o_v      <= !reset && eo_v;
        o8_last  <= !reset && eo_v && eo_c == P - 1 && eo_tag[ETW-1];
        run_done <= !reset && o8_last;
    end
    always @(*) o_d = o8_all;

endmodule
