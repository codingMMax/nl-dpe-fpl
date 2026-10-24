// ============================================================================
// dimm_tile_top.v — tiled DIMM (v3, contract ip/dimm/spec.md, the bit-plane pass). SIMULATION twin: behavior `dpe`
// (ip/xbar/rtl/sim/dpe_nldpe.v) with weight programming on weight_in / weight_en (testbenches preload the weights,
// PLAN R8). VTR twin: ip/dimm/rtl/vtr/dimm_tile_top.v (bare `dpe`, no weight path). Model:
// simulator/models/dimmTile.py.
//
// One job: C[m][n] = sum_k (-1)^(sA ^ sB) EXP(trunc8(LA[m][k] + LB[k][n])), then sat8(C >> NARROW) if NARROW > 0.
// Every port carries a window as 8 plane words on consecutive cycles, LSB first (bit e of plane word b = bit b of
// element e; C in ECL = ceil(EC/8) lanes of I bits). Operand ports take rows (int8 through a LOG converter per
// channel, or log bytes); the module stages them into the farm's word layout, holds A_SLOTS / B_SLOTS matrices, takes
// jobs (slot pair, release bit) on a burst-on-ready job port into a one-entry register, and pushes C windows on NG
// channels against C_CAP credits.
//
// Rewritten 2026-10-23 for the bit-plane pass (spec §10 item 8) from the pre-bit-plane F3 module (git show
// 8acf73c:v3/ip/dimm/rtl/sim/dimm_tile_top.v). Kept: the loaders' control, the slots, the jobs and the credits. New:
// the stagers shift in one bit per plane; A_GRP = min(P, X_S'); the window stager writes through each A memory's own
// port, E_WP cycles per window; the farm is fed from shift registers; the drain is summed per plane (popcount trees,
// the root accumulator, NARROW on the exact value, the output word buffer).
// Revised 2026-10-23 (spec §10 item 11): where R >= 2I the add is the crossbar's row sum (X_ADD: the identity stacked
// twice, rows r and I + r; no adders), else I bit-serial adders per crossbar; the B word loads once per job and
// rotates; the drain input is (e & d_on) ^ s_A ^ s_B with the signs captured at the MSB plane; the root accumulates
// in right-shift form (u <= 2 n_neg, then floor(u/2) + POP_b per plane, -2 n_neg at plane 7; the bit shifted out is
// final and goes to a 7-bit low register).
//
// Structure (spec §4): per channel converter -> distribution register (plane index, window count) -> the slices.
// The loaders' control (completion, writer counters, memory addresses) is shared, the staging data is per slice.
// Slice s = the farm's columns [s*KS, (s+1)*KS) of all NG groups (X_S' crossbars): its A and B memories, its copy of
// the wave sequencer, its crossbars, the first log2 KS popcount-tree levels. Shared: slots, jobs, credits, the last
// log2 NS tree levels, the root accumulators, NARROW, the output plane buffer, the output channels.
// Every latency is named in spec §5; the model times every port event by the same rules (GATE 2: Delta = 0).
// Domain: the elaboration guards below (DimmTile.checkDomain; A_CH and B_CH powers of two).
// ============================================================================

`timescale 1ns / 1ps

module dimm_tile_top #(
    parameter M        = 64,
    parameter N        = 16,
    parameter K        = 16,
    parameter R        = 64,
    parameter C        = 64,
    parameter P        = 8,
    parameter N_E      = 32,
    parameter X_S      = 16,
    parameter A_DOM    = 0,
    parameter A_SIGNED = 1,
    parameter A_CH     = 1,
    parameter A_SLOTS  = 1,
    parameter A_FRAME  = 0,
    parameter A_ROWS   = 1,
    parameter B_DOM    = 0,
    parameter B_SIGNED = 1,
    parameter B_CH     = 1,
    parameter B_SLOTS  = 1,
    parameter TRANS_B  = 1,
    parameter NARROW   = 0,
    parameter C_CAP    = 2,
    // derived (port widths; do not override)
    parameter I        = (R < C) ? R : C,
    parameter NG       = N_E / K,
    parameter VW       = 9 + $clog2(K),
    parameter EC       = (NARROW > 0) ? 8 : VW,
    parameter SAW      = (A_SLOTS > 1) ? $clog2(A_SLOTS) : 1,
    parameter SBW      = (B_SLOTS > 1) ? $clog2(B_SLOTS) : 1,
    parameter A_W      = A_ROWS * K,                            // an A window's elements (plane word bits)
    parameter LBL      = TRANS_B ? K : N,                       // a B row's elements
    parameter ECL      = (EC + 7) / 8                           // the C port's plane lanes
) (
    input  wire                  clk,
    input  wire                  reset,
    // weight programming (simulation twin only): R*C identity strobes broadcast to every crossbar
    input  wire [7:0]            weight_in,
    input  wire                  weight_en,
    // A port: A_CH channels, push + credit
    input  wire [A_CH-1:0]       a_v,
    input  wire [A_CH*A_W-1:0]   a_d,
    output reg                   a_cr,
    // B port
    input  wire [B_CH-1:0]       b_v,
    input  wire [B_CH*LBL-1:0]   b_d,
    output reg                   b_cr,
    // job port: burst on ready into the job register
    input  wire                  j_v,
    input  wire [SAW-1:0]        j_a,
    input  wire [SBW-1:0]        j_b,
    input  wire                  j_ra,
    output wire                  j_rdy,
    // C port: NG channels, push + credit
    output wire [NG-1:0]         c_v,
    output wire [NG*I*ECL-1:0]   c_d,
    input  wire                  c_cr,
    output wire                  busy
);
    // ---- derived (spec §2) -------------------------------------------------------------------------------------
    localparam NB    = I / N;                                   // rows per farm window
    localparam W_K   = (M * N) / I;                             // windows per column
    localparam T_W   = W_K / NG;                                // waves
    localparam XS    = (X_S < N_E) ? X_S : N_E;                 // crossbars per slice
    localparam NS    = N_E / XS;                                // slices
    localparam KS    = XS / NG;                                 // columns per slice
    localparam AGC   = (P < XS) ? P : XS;
    localparam A_GRP = 1 << ($clog2(AGC + 1) - 1);              // crossbars per A memory: 2^k <= min(P, X_S')
    localparam B_GRP = (A_GRP < KS) ? A_GRP : KS;               // columns per B memory
    localparam NQW   = A_ROWS / NB;                             // NB-row groups per A input window (window frame)
    localparam NQ1   = (A_FRAME == 1) ? NQW : 1;
    localparam WPMA  = M / A_ROWS;                              // A input windows per matrix
    localparam PW    = $clog2(P);                               // plane index bits
    localparam KSW   = (KS > 1) ? $clog2(KS) : 1;
    localparam LKS   = $clog2(KS);
    localparam LNG   = $clog2(NG);
    localparam LG    = $clog2(A_GRP);
    localparam LBG   = $clog2(B_GRP);
    localparam LQ    = $clog2(NQ1);
    localparam LTW   = (T_W > 1) ? $clog2(T_W) : 1;
    localparam NAB   = XS / A_GRP;                              // A memories per slice, each with its own write port
    localparam NBM   = KS / B_GRP;                              // B memories per slice
    localparam LNBMW = (NBM > 1) ? $clog2(NBM) : 1;
    localparam LGW   = (LG > 0) ? LG : 1;
    localparam LBGW  = (LBG > 0) ? LBG : 1;
    localparam LQW   = (LQ > 0) ? LQ : 1;
    localparam AAW   = SAW + LTW + LGW;                         // A memory address {slot, wave, index}
    localparam BAW   = SBW + LBGW;                              // B memory address {slot, index}
    localparam LNS   = $clog2(NS);
    localparam CW    = 1 + LKS;                                 // a slice's per-plane counts
    localparam RW    = CW + LNS;                                // the root's per-plane counts: 1 + log2 K
    localparam CTLW  = 3 + PW;                                  // the drain control {n_neg, plane valid, plane, last}
    localparam CCW   = $clog2(C_CAP + 1);
    localparam LCHA  = (A_CH > 1) ? $clog2(A_CH) : 0;
    localparam LCHB  = (B_CH > 1) ? $clog2(B_CH) : 0;
    localparam LNB   = $clog2(NB);
    // the window stager (window frame, spec §5): a window's farm windows repeat their crossbar pattern every NPH windows
    localparam NPH   = (NQ1 < NG) ? NG / NQ1 : 1;
    localparam LPH   = $clog2(NPH);
    localparam LPHW  = (LPH > 0) ? LPH : 1;

    // word (q, i) of a window at phase ph (q < NQ1, i < KS; q outer, i inner) goes to the slice's crossbar
    // f = (ph*NQ1 + q) mod NG + NG i, in A memory f >> LG (spec §6)
    function integer wp_words;                                  // E_WP: the most words of a window one memory takes
        input integer dummy;
        integer ph, m, q, i, n, most;
        begin
            most = 0;
            for (ph = 0; ph < NPH; ph = ph + 1)
                for (m = 0; m < NAB; m = m + 1) begin
                    n = 0;
                    for (q = 0; q < NQ1; q = q + 1)
                        for (i = 0; i < KS; i = i + 1)
                            if ((((((ph << LQ) + q) & (NG - 1)) + (i << LNG)) >> LG) == m) n = n + 1;
                    if (n > most) most = n;
                end
            wp_words = most;
        end
    endfunction
    function integer wsel;                                      // memory m's e-th word at phase ph: {1, q, i} or 0
        input integer m, ph, e;
        integer q, i, n;
        begin
            wsel = 0;
            n = 0;
            for (q = 0; q < NQ1; q = q + 1)
                for (i = 0; i < KS; i = i + 1)
                    if ((((((ph << LQ) + q) & (NG - 1)) + (i << LNG)) >> LG) == m) begin
                        if (n == e) wsel = (1 << 20) | (q << 10) | i;
                        n = n + 1;
                    end
        end
    endfunction
    localparam E_WP  = wp_words(0);                             // the window stager's cycles per window
    localparam LEW   = $clog2(E_WP);
    localparam LEWW  = (LEW > 0) ? LEW : 1;
    localparam SELW  = 1 + LQW + KSW;                           // a port table entry {valid, q, i}

    generate
        if ((M & (M-1)) != 0 || (N & (N-1)) != 0 || (K & (K-1)) != 0 || (N_E & (N_E-1)) != 0
            || (X_S & (X_S-1)) != 0 || (I & (I-1)) != 0 || I < 8 || N > I || M * N < I || K < 2 || K > N_E
            || NG > W_K || NG > XS || K > I || LBL > I || P != 8
            || (A_CH & (A_CH-1)) != 0 || (B_CH & (B_CH-1)) != 0 || A_SLOTS < 1 || B_SLOTS < 1 || C_CAP < 1
            || (A_FRAME == 0 && (A_ROWS != 1 || A_CH > NB))
            || (A_FRAME == 1 && (A_ROWS < NB || (A_ROWS % NB) != 0 || (M % A_ROWS) != 0 || A_DOM != 1
                                 || (NQW & (NQW-1)) != 0 || A_ROWS * K > I))
            || NARROW < 0 || NARROW >= VW)
        begin : g_bad_param
            dimm_tile_parameter_outside_domain u_bad ();
        end
    endgenerate

    genvar gch, gs, gi, gr, gj, gm, gp, ge, gl;

    // =============================================================================================================
    // 1  streams: converter per channel (int8) or the port planes (log), then the distribution register (DIST_LAT 1)
    //    with the framing counters (plane in the window, windows on the channel; low bits only)
    // =============================================================================================================
    localparam ANW = (A_FRAME == 0) ? ((2 * NB / A_CH > 1) ? $clog2(2 * NB / A_CH) : 1) : 1;    // window count bits
    localparam BNW = TRANS_B ? ((2 * N / B_CH > 1) ? $clog2(2 * N / B_CH) : 1) : 1;
    wire [A_CH-1:0]      sa_v;
    wire [A_CH*A_W-1:0]  sa_d;
    wire [B_CH-1:0]      sb_v;
    wire [B_CH*LBL-1:0]  sb_d;
    wire [A_CH+B_CH-1:0] conv_err;
    generate
        for (gch = 0; gch < A_CH; gch = gch + 1) begin : g_sa
            if (A_DOM == 0) begin : g_conv
                dimm_conv #(.R(R), .C(C), .P(P), .L(A_W)) u_conv (
                    .clk(clk), .reset(reset), .weight_in(weight_in), .weight_en(weight_en),
                    .x_v(a_v[gch]), .x_d(a_d[gch*A_W +: A_W]),
                    .s_v(sa_v[gch]), .s_d(sa_d[gch*A_W +: A_W]), .err(conv_err[gch]));
            end else begin : g_log
                assign sa_v[gch] = a_v[gch];
                assign sa_d[gch*A_W +: A_W] = a_d[gch*A_W +: A_W];
                assign conv_err[gch] = 1'b0;
            end
        end
        for (gch = 0; gch < B_CH; gch = gch + 1) begin : g_sb
            if (B_DOM == 0) begin : g_conv
                dimm_conv #(.R(R), .C(C), .P(P), .L(LBL)) u_conv (
                    .clk(clk), .reset(reset), .weight_in(weight_in), .weight_en(weight_en),
                    .x_v(b_v[gch]), .x_d(b_d[gch*LBL +: LBL]),
                    .s_v(sb_v[gch]), .s_d(sb_d[gch*LBL +: LBL]), .err(conv_err[A_CH + gch]));
            end else begin : g_log
                assign sb_v[gch] = b_v[gch];
                assign sb_d[gch*LBL +: LBL] = b_d[gch*LBL +: LBL];
                assign conv_err[A_CH + gch] = 1'b0;
            end
        end
    endgenerate

    reg  [A_CH-1:0]     da_v;
    reg  [A_CH*A_W-1:0] da_d;
    reg  [A_CH*PW-1:0]  da_c;
    reg  [A_CH*ANW-1:0] da_n;
    reg  [B_CH-1:0]     db_v;
    reg  [B_CH*LBL-1:0] db_d;
    reg  [B_CH*PW-1:0]  db_c;
    reg  [B_CH*BNW-1:0] db_n;
    reg  [A_CH*PW-1:0]  fa_c;                                    // framing state: next plane index, windows seen
    reg  [A_CH*ANW-1:0] fa_n;
    reg  [B_CH*PW-1:0]  fb_c;
    reg  [B_CH*BNW-1:0] fb_n;
    integer ci;
    always @(posedge clk) begin
        for (ci = 0; ci < A_CH; ci = ci + 1) begin
            da_v[ci] <= !reset && sa_v[ci];
            da_d[ci*A_W +: A_W] <= sa_d[ci*A_W +: A_W];
            da_c[ci*PW +: PW] <= fa_c[ci*PW +: PW];
            da_n[ci*ANW +: ANW] <= fa_n[ci*ANW +: ANW];
            if (reset) begin
                fa_c[ci*PW +: PW] <= 0; fa_n[ci*ANW +: ANW] <= 0;
            end
            else if (sa_v[ci]) begin
                if (fa_c[ci*PW +: PW] == P - 1) begin
                    fa_c[ci*PW +: PW] <= 0;
                    fa_n[ci*ANW +: ANW] <= fa_n[ci*ANW +: ANW] + 1'b1;
                end
                else fa_c[ci*PW +: PW] <= fa_c[ci*PW +: PW] + 1'b1;
            end
        end
        for (ci = 0; ci < B_CH; ci = ci + 1) begin
            db_v[ci] <= !reset && sb_v[ci];
            db_d[ci*LBL +: LBL] <= sb_d[ci*LBL +: LBL];
            db_c[ci*PW +: PW] <= fb_c[ci*PW +: PW];
            db_n[ci*BNW +: BNW] <= fb_n[ci*BNW +: BNW];
            if (reset) begin
                fb_c[ci*PW +: PW] <= 0; fb_n[ci*BNW +: BNW] <= 0;
            end
            else if (sb_v[ci]) begin
                if (fb_c[ci*PW +: PW] == P - 1) begin
                    fb_c[ci*PW +: PW] <= 0;
                    fb_n[ci*BNW +: BNW] <= fb_n[ci*BNW +: BNW] + 1'b1;
                end
                else fb_c[ci*PW +: PW] <= fb_c[ci*PW +: PW] + 1'b1;
            end
        end
    end

    // =============================================================================================================
    // 2  A loader control (shared): row frame = transposer of NB-row groups; window frame = window stager writing
    //    through every A memory's own port. Outputs per A memory port: write, address; the staging read selects
    // =============================================================================================================
    wire [NAB-1:0]      aw_we;            // the A memory ports' writes this cycle
    wire [NAB*AAW-1:0]  aw_a;             // their addresses
    wire [NAB*LQW-1:0]  aw_q;             // window stager: per port, the NB-row group of its word
    wire [NAB*KSW-1:0]  aw_k;             // window stager: per port, the local column of its word
    reg  [KSW-1:0]      aw_i;             // transposer: local column
    reg                 aw_buf;           // staging buffer (row frame) / entry (window frame)
    reg  [((A_CH > 1) ? $clog2(A_CH) : 1)-1:0] aw_ch;          // staging channel (window frame)
    wire                a_unit_end;       // the writer's last cycle of a staged unit
    wire                a_matrix_end;     // ... that completes an A matrix
    reg  [SAW-1:0]      a_wslot;          // the slot being written
    localparam WKW = (W_K > 1) ? $clog2(W_K) : 1;
    localparam WPW = (WPMA > 1) ? $clog2(WPMA) : 1;

    generate
        if (A_FRAME == 0) begin : g_axc
            // rows done per buffer, complete flags, the writer (spec §5 transposer)
            reg [$clog2(NB+1)-1:0] cnt0, cnt1;
            reg                    cmp0, cmp1;
            reg [KSW:0]            wi;                 // 0 idle, else the next word index (counts 1..KS-1)
            reg                    qb;                 // the group being emitted: buffer
            reg [WKW-1:0]          qw;                 // its window in the matrix
            integer                ch, nd0, nd1;
            reg [31:0]             rowg;
            always @(*) begin
                nd0 = 0; nd1 = 0;
                for (ch = 0; ch < A_CH; ch = ch + 1)
                    if (da_v[ch] && da_c[ch*PW +: PW] == P - 1) begin             // a row's last plane
                        rowg = (da_n[ch*ANW +: ANW] << LCHA) | ch;          // the row's index (low bits)
                        if ((rowg >> LNB) & 1) nd1 = nd1 + 1; else nd0 = nd0 + 1;
                    end
            end
            wire start_w = (wi == 0) && (qb ? cmp1 : cmp0);
            wire [KSW-1:0] i_now = (wi == 0) ? {KSW{1'b0}} : wi[KSW-1:0];
            wire last_w = (wi == 0) ? (start_w && KS == 1) : (wi[KSW-1:0] == KS - 1 || KS == 1);
            wire we_now = start_w || (wi != 0);
            assign a_unit_end = we_now && last_w;
            assign a_matrix_end = a_unit_end && (qw == W_K - 1);
            // crossbar f = (w mod NG) | (i << LNG): memory f >> LG, index f mod A_GRP; address {slot, wave, index}
            wire [KSW+LNG:0] f_loc = ({{(LNG+1){1'b0}}, i_now} << LNG) | (qw & (NG - 1));
            wire [WKW:0]     wav   = qw >> LNG;
            wire [LGW-1:0]   idx   = f_loc[LGW-1:0] & (A_GRP - 1);
            for (gm = 0; gm < NAB; gm = gm + 1) begin : g_port
                assign aw_we[gm] = we_now && (NAB == 1 || (f_loc >> LG) == gm);
                assign aw_a[gm*AAW +: AAW] = {a_wslot, wav[LTW-1:0], idx};
                assign aw_q[gm*LQW +: LQW] = {LQW{1'b0}};
                assign aw_k[gm*KSW +: KSW] = i_now;
            end
            always @(*) begin
                aw_i   = i_now;
                aw_buf = qb;
                aw_ch  = 0;
            end
            always @(posedge clk) begin
                if (reset) begin
                    cnt0 <= 0; cnt1 <= 0; cmp0 <= 1'b0; cmp1 <= 1'b0; wi <= 0; qb <= 1'b0; qw <= 0; a_wslot <= 0;
                end
                else begin
                    // completion: the group's NB rows done (rows of the two buffers counted apart)
                    if (cnt0 + nd0 == NB && nd0 != 0) begin cnt0 <= 0; cmp0 <= 1'b1; end
                    else cnt0 <= cnt0 + nd0;
                    if (cnt1 + nd1 == NB && nd1 != 0) begin cnt1 <= 0; cmp1 <= 1'b1; end
                    else cnt1 <= cnt1 + nd1;
                    // writer
                    if (we_now) begin
                        if (last_w) begin
                            wi <= 0;
                            if (qb) cmp1 <= 1'b0; else cmp0 <= 1'b0;
                            qb <= !qb;
                            qw <= (qw == W_K - 1) ? {WKW{1'b0}} : qw + 1'b1;
                            if (qw == W_K - 1) a_wslot <= (a_wslot == A_SLOTS - 1) ? {SAW{1'b0}} : a_wslot + 1'b1;
                        end
                        else wi <= (wi == 0) ? 2'd1 : wi + 1'b1;
                    end
                end
            end
        end else begin : g_awc
            // window stager (spec §5): per channel two entries; windows written in global order, E_WP cycles each,
            // every A memory through its own port from one start; port m's e-th word at the window's phase is a table
            reg [A_CH-1:0]         cmp0, cmp1;          // entry 0 / 1 complete, per channel
            reg                    wr;                  // a window is being written
            reg [LEWW-1:0]         ei;                  // its next cycle
            reg [((A_CH > 1) ? $clog2(A_CH) : 1)-1:0] wch;   // the window being written: channel
            reg                    went;                //   entry
            reg [WPW-1:0]          wl;                  //   window in the matrix
            reg [WKW+1:0]          wbase;               //   its first farm window (wl * NQW)
            integer                ch;
            wire want    = went ? cmp1[wch] : cmp0[wch];
            wire start_w = !wr && want;
            wire we_now  = start_w || wr;
            wire [LEWW-1:0] e_now = wr ? ei : {LEWW{1'b0}};
            wire last_w  = (e_now == E_WP - 1);
            assign a_unit_end   = we_now && last_w;
            assign a_matrix_end = a_unit_end && (wl == WPMA - 1);
            wire [LPHW-1:0] ph = wl[LPHW-1:0] & (NPH - 1);            // the window's phase, wl mod NPH
            wire [LPHW+LEWW-1:0] ridx = (ph << LEW) | (e_now & (E_WP - 1));
            for (gm = 0; gm < NAB; gm = gm + 1) begin : g_port
                wire [NPH*E_WP*SELW-1:0] rom;                          // {valid, q, i} per (phase, cycle)
                for (gp = 0; gp < NPH; gp = gp + 1) begin : g_ph
                    for (ge = 0; ge < E_WP; ge = ge + 1) begin : g_e
                        localparam integer S  = wsel(gm, gp, ge);
                        localparam integer SV = (S >> 20) & 1, SQ = (S >> 10) & 1023, SI = S & 1023;
                        assign rom[(gp*E_WP + ge)*SELW +: SELW] = (SV << (LQW + KSW)) | (SQ << KSW) | SI;
                    end
                end
                reg [SELW-1:0] sel;
                integer j;
                always @(*) begin
                    sel = 0;
                    for (j = 0; j < NPH * E_WP; j = j + 1)
                        if (ridx == j) sel = rom[j*SELW +: SELW];
                end
                wire             v_now = sel[LQW + KSW];
                wire [LQW-1:0]   q_now = sel[KSW +: LQW];
                wire [KSW-1:0]   k_now = sel[0 +: KSW];
                wire [WKW+1:0]   w_a   = wbase + q_now;                 // the word's farm window
                wire [KSW+LNG:0] f_loc = ({{(LNG+1){1'b0}}, k_now} << LNG) | (w_a & (NG - 1));
                wire [WKW+1:0]   wav   = w_a >> LNG;
                wire [LGW-1:0]   idx   = f_loc[LGW-1:0] & (A_GRP - 1);
                assign aw_we[gm] = we_now && v_now;
                assign aw_a[gm*AAW +: AAW] = {a_wslot, wav[LTW-1:0], idx};
                assign aw_q[gm*LQW +: LQW] = q_now;
                assign aw_k[gm*KSW +: KSW] = k_now;
            end
            always @(*) begin
                aw_i   = 0;
                aw_buf = went;
                aw_ch  = wch;
            end
            always @(posedge clk) begin
                if (reset) begin
                    cmp0 <= 0; cmp1 <= 0; wr <= 1'b0; ei <= 0; wch <= 0; went <= 1'b0; wl <= 0; wbase <= 0;
                    a_wslot <= 0;
                end
                else begin
                    for (ch = 0; ch < A_CH; ch = ch + 1)
                        if (da_v[ch] && da_c[ch*PW +: PW] == P - 1) begin              // a window's last plane
                            if (da_n[ch*ANW] == 1'b1) cmp1[ch] <= 1'b1; else cmp0[ch] <= 1'b1;
                        end
                    if (we_now) begin
                        if (last_w) begin
                            wr <= 1'b0;
                            ei <= 0;
                            if (went) cmp1[wch] <= 1'b0; else cmp0[wch] <= 1'b0;
                            if (A_CH == 1 || wch == A_CH - 1) begin wch <= 0; went <= !went; end
                            else wch <= wch + 1'b1;
                            if (wl == WPMA - 1) begin
                                wl <= 0; wbase <= 0;
                                a_wslot <= (a_wslot == A_SLOTS - 1) ? {SAW{1'b0}} : a_wslot + 1'b1;
                            end
                            else begin wl <= wl + 1'b1; wbase <= wbase + NQW; end
                        end
                        else begin
                            wr <= 1'b1;
                            ei <= e_now + 1'b1;
                        end
                    end
                end
            end
        end
    endgenerate

    // =============================================================================================================
    // 3  B loader control (shared): TRANS_B = 1 transposer of N-row groups (one group = one matrix);
    //    TRANS_B = 0 row writer (two row registers per channel, one row per cycle in row order)
    // =============================================================================================================
    reg                 bw_we;
    reg  [KSW-1:0]      bw_i;                        // transposer: local column
    reg                 bw_buf;
    reg  [NS-1:0]       bw_sl;                       // row writer: the slice written
    reg  [LNBMW-1:0]    bw_m;
    reg  [BAW-1:0]      bw_a;
    reg  [N*8-1:0]      bw_rd;                       // row writer: the row
    wire                b_unit_end, b_matrix_end;
    reg  [SBW-1:0]      b_wslot;
    generate
        if (TRANS_B) begin : g_bxc
            reg [$clog2(N+1)-1:0] cnt0, cnt1;
            reg                   cmp0, cmp1;
            reg [KSW:0]           wi;
            reg                   qb;
            integer               ch, nd0, nd1;
            reg [31:0]            rowg;
            always @(*) begin
                nd0 = 0; nd1 = 0;
                for (ch = 0; ch < B_CH; ch = ch + 1)
                    if (db_v[ch] && db_c[ch*PW +: PW] == P - 1) begin
                        rowg = (db_n[ch*BNW +: BNW] << LCHB) | ch;
                        if ((rowg >> $clog2(N)) & 1) nd1 = nd1 + 1; else nd0 = nd0 + 1;
                    end
            end
            wire start_w = (wi == 0) && (qb ? cmp1 : cmp0);
            wire [KSW-1:0] i_now = (wi == 0) ? {KSW{1'b0}} : wi[KSW-1:0];
            wire [LBGW-1:0] idx_b = (B_GRP > 1) ? i_now[LBGW-1:0] : 1'b0;
            wire last_w = (wi == 0) ? (start_w && KS == 1) : (wi[KSW-1:0] == KS - 1 || KS == 1);
            assign b_unit_end = (start_w || wi != 0) && last_w;
            assign b_matrix_end = b_unit_end;
            always @(*) begin
                bw_we  = start_w || (wi != 0);
                bw_i   = i_now;
                bw_buf = qb;
                bw_sl  = {NS{1'b1}};
                bw_m   = (NBM > 1) ? (i_now >> LBG) : 0;
                bw_a   = {b_wslot, idx_b};
                bw_rd  = 0;
            end
            always @(posedge clk) begin
                if (reset) begin
                    cnt0 <= 0; cnt1 <= 0; cmp0 <= 1'b0; cmp1 <= 1'b0; wi <= 0; qb <= 1'b0; b_wslot <= 0;
                end
                else begin
                    if (cnt0 + nd0 == N && nd0 != 0) begin cnt0 <= 0; cmp0 <= 1'b1; end
                    else cnt0 <= cnt0 + nd0;
                    if (cnt1 + nd1 == N && nd1 != 0) begin cnt1 <= 0; cmp1 <= 1'b1; end
                    else cnt1 <= cnt1 + nd1;
                    if (bw_we) begin
                        if (last_w) begin
                            wi <= 0;
                            if (qb) cmp1 <= 1'b0; else cmp0 <= 1'b0;
                            qb <= !qb;
                            b_wslot <= (b_wslot == B_SLOTS - 1) ? {SBW{1'b0}} : b_wslot + 1'b1;
                        end
                        else wi <= (wi == 0) ? 2'd1 : wi + 1'b1;
                    end
                end
            end
        end else begin : g_brw
            // two row registers per channel; row t of the stream on channel t mod B_CH, register (t / B_CH) mod 2;
            // byte n of a row shifts in bit n of each of the row's planes (LSB first)
            reg  [B_CH*2*N*8-1:0] rr;
            reg  [B_CH-1:0] rdy0, rdy1;
            reg  [((B_CH > 1) ? $clog2(B_CH) : 1)-1:0] tch;   // next row to write: channel
            reg             treg;                              //   register
            reg  [$clog2(K+1)-1:0] tk;                         //   its row k in the matrix
            integer ch;
            for (gch = 0; gch < B_CH; gch = gch + 1) begin : g_rch
                for (gr = 0; gr < 2; gr = gr + 1) begin : g_rr
                    for (gj = 0; gj < N; gj = gj + 1) begin : g_rb
                        localparam B0 = ((gch*2 + gr)*N + gj)*8;
                        always @(posedge clk)
                            if (db_v[gch] && db_n[gch*BNW] == gr)
                                rr[B0 +: 8] <= {db_d[gch*LBL + gj], rr[B0 + 1 +: 7]};
                    end
                end
            end
            wire want = treg ? rdy1[tch] : rdy0[tch];
            assign b_unit_end = want;
            assign b_matrix_end = want && (tk == K - 1);
            wire [$clog2(K+1)-1:0] k_now = tk;
            wire [KSW-1:0]  i_k = k_now[KSW-1:0] & (KS - 1);
            wire [LBGW-1:0] idx_b = (B_GRP > 1) ? i_k[LBGW-1:0] : 1'b0;
            integer rsel;
            always @(*) begin
                bw_we  = want;
                bw_i   = i_k;
                bw_buf = 1'b0;
                bw_sl  = (NS > 1) ? ({{(NS-1){1'b0}}, want} << (k_now >> LKS)) : want;
                bw_m   = (NBM > 1) ? (i_k >> LBG) : 0;
                bw_a   = {b_wslot, idx_b};
                bw_rd  = 0;
                for (rsel = 0; rsel < 2 * B_CH; rsel = rsel + 1)
                    if (rsel == ((tch << 1) | treg)) bw_rd = rr[rsel*N*8 +: N*8];
            end
            always @(posedge clk) begin
                if (reset) begin
                    rdy0 <= 0; rdy1 <= 0; tch <= 0; treg <= 1'b0; tk <= 0; b_wslot <= 0;
                end
                else begin
                    for (ch = 0; ch < B_CH; ch = ch + 1)
                        if (db_v[ch] && db_c[ch*PW +: PW] == P - 1) begin             // a row's last plane
                            if (db_n[ch*BNW]) rdy1[ch] <= 1'b1; else rdy0[ch] <= 1'b1;
                        end
                    if (want) begin
                        if (treg) rdy1[tch] <= 1'b0; else rdy0[tch] <= 1'b0;
                        if (B_CH == 1 || tch == B_CH - 1) begin tch <= 0; treg <= !treg; end
                        else tch <= tch + 1'b1;
                        if (tk == K - 1) begin
                            tk <= 0;
                            b_wslot <= (b_wslot == B_SLOTS - 1) ? {SBW{1'b0}} : b_wslot + 1'b1;
                        end
                        else tk <= tk + 1'b1;
                    end
                end
            end
        end
    endgenerate

    // =============================================================================================================
    // 4  slots, credits, jobs (spec §5)
    // =============================================================================================================
    reg  [A_SLOTS-1:0] a_valid;
    reg  [B_SLOTS-1:0] b_valid;
    reg                jv_q, jra_q;
    reg  [SAW-1:0]     ja_q;
    reg  [SBW-1:0]     jb_q;
    reg                go_q;
    reg  [CCW-1:0]     ccnt;
    wire [NS-1:0]      s_idle;
    wire               idle_all = &s_idle;
    wire               done;
    assign j_rdy = !jv_q;
    wire take = j_v && !jv_q;
    wire dec  = jv_q && a_valid[ja_q] && b_valid[jb_q] && idle_all && (ccnt != 0);
    // release information of the jobs in flight (at most two: a job's output tail overlaps the next job's start):
    // pushed at the decision, popped at done
    reg  [1:0]         rq_n;
    reg  [SAW-1:0]     rq_s0, rq_s1;
    reg                rq_r0, rq_r1;
    assign busy = (rq_n != 0);
    always @(posedge clk) begin
        if (reset) begin
            jv_q <= 1'b0; go_q <= 1'b0; ccnt <= C_CAP; a_valid <= 0; b_valid <= 0; a_cr <= 1'b0; b_cr <= 1'b0;
            rq_n <= 0; ja_q <= 0; jb_q <= 0;
        end
        else begin
            go_q <= dec;
            if (take) begin jv_q <= 1'b1; ja_q <= j_a; jb_q <= j_b; jra_q <= j_ra; end
            else if (dec) jv_q <= 1'b0;
            ccnt <= ccnt - dec + c_cr;
            // slots valid (the slices are lock-step: the shared writer control)
            if (a_matrix_end) a_valid[a_wslot] <= 1'b1;
            if (b_matrix_end) b_valid[b_wslot] <= 1'b1;
            // credits: staging (row-frame A, TRANS_B B), per row (TRANS_B = 0 B), per released slot (window-frame A)
            a_cr <= (A_FRAME == 0) ? a_unit_end : (done && rq_r0);
            b_cr <= b_unit_end;
            if (A_FRAME == 1 && done && rq_r0) a_valid[rq_s0] <= 1'b0;
            // the release FIFO
            case ({dec, done})
                2'b10: begin
                    if (rq_n == 0) begin rq_s0 <= ja_q; rq_r0 <= jra_q; end
                    else begin rq_s1 <= ja_q; rq_r1 <= jra_q; end
                    rq_n <= rq_n + 1'b1;
                end
                2'b01: begin rq_s0 <= rq_s1; rq_r0 <= rq_r1; rq_n <= rq_n - 1'b1; end
                2'b11: begin
                    if (rq_n == 1) begin rq_s0 <= ja_q; rq_r0 <= jra_q; end
                    else begin rq_s0 <= rq_s1; rq_r0 <= rq_r1; rq_s1 <= ja_q; rq_r1 <= jra_q; end
                end
                default: ;
            endcase
        end
    end

    // =============================================================================================================
    // 5  slices
    // =============================================================================================================
    wire [NS*CTLW-1:0]    sl_ctl;
    wire [NS*NG*I*CW-1:0] sl_cnt;
    generate
        for (gs = 0; gs < NS; gs = gs + 1) begin : g_sl
            // A staging data of this slice -> the A memory ports' write words
            wire [NAB*NB*8-1:0] a_word;
            if (A_FRAME == 0) begin : g_ast
                wire [NB*8-1:0] xw;
                dimm_xpose_stage #(.G(NB), .KS(KS), .KSW(KSW), .CH(A_CH), .L(A_W), .NW(ANW), .COL0(gs * KS)) u_st (
                    .clk(clk), .d_v(da_v), .d_d(da_d), .d_n(da_n), .r_buf(aw_buf), .r_i(aw_i), .r_word(xw));
                assign a_word = {NAB{xw}};
            end else begin : g_awst
                dimm_wstage #(.ROWS(A_ROWS), .K(K), .KS(KS), .KSW(KSW), .NB(NB), .CH(A_CH), .NW(ANW),
                              .COL0(gs * KS), .NP(NAB), .LQW(LQW)) u_st (
                    .clk(clk), .d_v(da_v), .d_d(da_d), .d_n(da_n),
                    .r_ch(aw_ch), .r_ent(aw_buf), .r_q(aw_q), .r_i(aw_k), .r_word(a_word));
            end
            // B write word
            wire [N*8-1:0] b_word;
            if (TRANS_B) begin : g_bst
                dimm_xpose_stage #(.G(N), .KS(KS), .KSW(KSW), .CH(B_CH), .L(LBL), .NW(BNW), .COL0(gs * KS)) u_st (
                    .clk(clk), .d_v(db_v), .d_d(db_d), .d_n(db_n), .r_buf(bw_buf), .r_i(bw_i), .r_word(b_word));
            end else begin : g_brd
                assign b_word = bw_rd;
            end
            dimm_slice #(.M(M), .N(N), .R(R), .C(C), .P(P), .XS(XS), .NG(NG), .KS(KS), .T_W(T_W),
                         .A_GRP(A_GRP), .B_GRP(B_GRP), .A_SLOTS(A_SLOTS), .B_SLOTS(B_SLOTS),
                         .A_SIGNED(A_SIGNED), .B_SIGNED(B_SIGNED), .SAW(SAW), .SBW(SBW), .AAW(AAW), .BAW(BAW),
                         .NAB(NAB), .LNBMW(LNBMW), .CW(CW), .CTLW(CTLW)) u_slice (
                .clk(clk), .reset(reset), .weight_in(weight_in), .weight_en(weight_en),
                .a_we(aw_we), .a_wa(aw_a), .a_wd(a_word),
                .b_we(bw_we && bw_sl[gs]), .b_wm(bw_m), .b_wa(bw_a), .b_wd(b_word),
                .start(go_q), .a_sel(ja_q), .b_sel(jb_q), .idle(s_idle[gs]),
                .o_ctl(sl_ctl[gs*CTLW +: CTLW]), .o_cnt(sl_cnt[gs*NG*I*CW +: NG*I*CW]));
        end
    endgenerate

    // =============================================================================================================
    // 6  the sum of the NS slice counts (registered once when NS > 1: SHARED_LAT); per group the root accumulators in
    //    right-shift form (spec §4: u <= 2 n_neg at the n_neg plane, then u <= floor(u/2) + POP_b per plane, -2 n_neg
    //    at plane 7, the bit shifted out into the 7-bit low register; a = 128 u + low is exact after plane 7), NARROW
    //    on the exact value, the output word buffer (ROOT_LAT 2); done (DONE_LAT 2)
    // =============================================================================================================
    wire [CTLW-1:0]    t_ctl;
    wire [NG*I*RW-1:0] t_cnt;
    dimm_ptree #(.NG(NG), .DC(NS), .NE(I), .IW(CW), .CTLW(CTLW)) u_top (
        .clk(clk), .reset(reset), .in_ctl(sl_ctl[0 +: CTLW]), .in_val(sl_cnt), .out_ctl(t_ctl), .out_val(t_cnt));
    wire          t_n = t_ctl[CTLW-1];                // the n_neg counts at the root
    wire          t_v = t_ctl[CTLW-2];                // plane t_b's counts at the root
    wire [PW-1:0] t_b = t_ctl[1 +: PW];
    wire          t_l = t_ctl[0];                     // the job's last wave
    reg           fin, fin_l, ov, ol, dn1, dn2;
    reg  [PW-1:0] ob;
    always @(posedge clk) begin
        fin   <= !reset && t_v && (t_b == P - 1);     // the accumulators hold the wave's exact values
        fin_l <= t_l;
        if (reset) begin ov <= 1'b0; ob <= 0; ol <= 1'b0; end
        else if (fin) begin ov <= 1'b1; ob <= 0; ol <= fin_l; end
        else if (ov) begin
            if (ob == P - 1) ov <= 1'b0;
            ob <= ob + 1'b1;
        end
        dn1 <= !reset && ov && (ob == P - 1) && ol;   // the job's last output plane
        dn2 <= !reset && dn1;
    end
    assign done = dn2;
    assign c_v  = {NG{ov}};
    localparam AW1 = VW + 1;                          // the value a = 128 u + low (-255 K .. 255 K)
    localparam UW  = RW + 2;                          // the accumulator u = floor(a / 2^(b+1)) during the planes (-2K .. 2K)
    generate
        for (gi = 0; gi < NG; gi = gi + 1) begin : g_root
            reg  [I*UW-1:0]    u;                     // per element: the right-shift accumulator (spec §4)
            reg  [I*7-1:0]     lo;                    // per element: the finished low bits
            reg  [I*RW-1:0]    nn;                    // per element: n_neg, for the -2 n_neg term at plane 7
            reg  [I*8*ECL-1:0] obuf;
            reg  [I*UW-1:0]    un;                    // per element: the accumulator's next value
            reg  [I*EC-1:0]    y;                     // per element: the exact value, narrowed
            reg  [RW-1:0]      cx;
            reg  signed [UW-1:0] uv, term;
            reg  signed [AW1-1:0] av, sh;
            integer e;
            always @(*) begin
                for (e = 0; e < I; e = e + 1) begin
                    cx   = t_cnt[(gi*I + e)*RW +: RW];
                    uv   = u[e*UW +: UW];
                    // POP_b, and at plane 7 also -2 n_neg (the -256 n_neg part of the -255 n_neg preload)
                    term = (t_b == P - 1) ? $signed({{(UW-RW){1'b0}}, cx}) - $signed({{(UW-RW-1){1'b0}}, nn[e*RW +: RW], 1'b0})
                                          : $signed({{(UW-RW){1'b0}}, cx});
                    un[e*UW +: UW] = (uv >>> 1) + term;
                    av   = $signed({uv, lo[e*7 +: 7]});
                    sh   = av >>> NARROW;
                    y[e*EC +: EC] = (NARROW > 0) ? ((sh > 127) ? 8'd127 : ((sh < -128) ? 8'h80 : sh[7:0]))
                                                 : av[EC-1:0];
                end
            end
            always @(posedge clk)
                for (e = 0; e < I; e = e + 1) begin
                    if (t_n) begin                    // the n_neg counts: u <= 2 n_neg (plane 0 then adds n_neg + POP_0)
                        u[e*UW +: UW]  <= {{(UW-RW-1){1'b0}}, t_cnt[(gi*I + e)*RW +: RW], 1'b0};
                        nn[e*RW +: RW] <= t_cnt[(gi*I + e)*RW +: RW];
                    end
                    else if (t_v) begin
                        u[e*UW +: UW] <= un[e*UW +: UW];
                        lo[e*7 +: 7]  <= {u[e*UW], lo[e*7 + 1 +: 6]};   // the bit shifted out is final (plane 0's is 0)
                    end
                    if (fin)      obuf[e*8*ECL +: 8*ECL] <= y[e*EC +: EC];
                    else if (ov)  obuf[e*8*ECL +: 8*ECL] <= (e + I/8 < I) ? obuf[(e + I/8)*8*ECL +: 8*ECL] : {(8*ECL){1'b0}};
                end
            // value words (docs/architecture.md §3.1): word ob of channel g = elements ob*I/8 .. of the window, 8 ECL bits
            // each, from the low I/8 slots of the buffer, which shifts I/8 elements per word
            assign c_d[gi*I*ECL +: I*ECL] = obuf[0 +: I*ECL];
        end
    endgenerate
endmodule


// ============================================================================
// dimm_conv — LOG converter of one int8 input channel: the input plane register (CONV_IN_LAT 1) feeding an identity
// crossbar in LOG mode, extent L x L; its drain planes are the converted row (reg_full = plane valid). `err`: a
// plane while the crossbar is not ready (the producer broke the converter period; simulation probe).
// ============================================================================
module dimm_conv #(
    parameter R = 64, C = 64, P = 8, L = 16
) (
    input  wire         clk,
    input  wire         reset,
    input  wire [7:0]   weight_in,
    input  wire         weight_en,
    input  wire         x_v,
    input  wire [L-1:0] x_d,
    output wire         s_v,
    output wire [L-1:0] s_d,
    output wire         err
);
    reg [L-1:0] in_q;
    reg         in_v;
    always @(posedge clk) begin
        in_q <= x_d;
        in_v <= !reset && x_v;
    end
    wire rdy;
    wire [R-1:0] dpe_in = weight_en ? {{(R-8){1'b0}}, weight_in} : in_q;   // rows >= L read 0
    wire [C-1:0] dpe_out;
    dpe #(
        .KERNEL_WIDTH(R), .NUM_COLS(C), .PRECISION(P), .ROWS_USED(L), .COLS_USED(L)
    ) u_dpe (
        .clk(clk), .reset(reset), .data_in(dpe_in), .nl_dpe_control(2'b11),   // LOG
        .shift_add_control(1'b0), .w_buf_en(in_v), .shift_add_bypass(1'b0),
        .load_output_reg(1'b0), .load_input_reg(weight_en),
        .MSB_SA_Ready(rdy), .data_out(dpe_out), .dpe_done(), .reg_full(s_v));
    assign s_d = dpe_out[L-1:0];
    reg err_q;
    always @(posedge clk) err_q <= !reset && (err_q || (in_v && !rdy));
    assign err = err_q;
endmodule


// ============================================================================
// dimm_xpose_stage — one slice's transposer staging (spec §5): G-row groups, two buffers, bank = row mod G; the
// slice's columns COL0 .. COL0 + KS - 1 of every row. Byte (buffer, bank, i) shifts in bit COL0 + i of each plane of
// a row whose index (low bits) selects that buffer and bank, LSB plane first: after the row's 8 planes it holds the
// element. Read: word i of buffer r_buf = the bytes of banks 0 .. G-1, column i (bank = byte position).
// ============================================================================
module dimm_xpose_stage #(
    parameter G = 4, KS = 8, KSW = 3, CH = 1, L = 16, NW = 2, COL0 = 0
) (
    input  wire              clk,
    input  wire [CH-1:0]     d_v,
    input  wire [CH*L-1:0]   d_d,
    input  wire [CH*NW-1:0]  d_n,
    input  wire              r_buf,
    input  wire [KSW-1:0]    r_i,
    output reg  [G*8-1:0]    r_word
);
    localparam LCH = (CH > 1) ? $clog2(CH) : 0;
    reg [2*G*KS*8-1:0] st;                           // byte ((buffer*G + bank)*KS + i)
    genvar gb, gk, gi;
    generate
        for (gb = 0; gb < 2; gb = gb + 1) begin : g_buf
            for (gk = 0; gk < G; gk = gk + 1) begin : g_bank
                for (gi = 0; gi < KS; gi = gi + 1) begin : g_col
                    localparam ROW = gb * G + gk;     // the row's index mod 2G
                    localparam B0  = ((gb*G + gk)*KS + gi)*8;
                    integer ch;
                    always @(posedge clk)
                        for (ch = 0; ch < CH; ch = ch + 1)
                            if (d_v[ch] && ((((d_n[ch*NW +: NW] << LCH) | ch) & (2*G - 1)) == ROW))
                                st[B0 +: 8] <= {d_d[ch*L + COL0 + gi], st[B0 + 1 +: 7]};
                end
            end
        end
    endgenerate
    integer bk, i;
    always @(*) begin
        r_word = 0;
        for (bk = 0; bk < G; bk = bk + 1)
            for (i = 0; i < KS; i = i + 1)
                if (i == r_i) r_word[bk*8 +: 8] = r_buf ? st[((G + bk)*KS + i)*8 +: 8] : st[(bk*KS + i)*8 +: 8];
    end
endmodule


// ============================================================================
// dimm_wstage — one slice's window staging (window frame, spec §5): per channel two entries of ROWS rows x the
// slice's KS keys. Byte (ch, entry, r, i) = element r*K + COL0 + i of a window whose count selects the entry; it
// shifts in that bit of each of the window's planes, LSB first. NP read ports (one per A memory): port p reads the
// NB-row group r_q[p], key r_i[p] of channel r_ch, entry r_ent = NB bytes (rows q*NB .. q*NB + NB - 1).
// ============================================================================
module dimm_wstage #(
    parameter ROWS = 4, K = 16, KS = 8, KSW = 3, NB = 4, CH = 1, NW = 1, COL0 = 0, NP = 2, LQW = 1
) (
    input  wire                 clk,
    input  wire [CH-1:0]        d_v,
    input  wire [CH*ROWS*K-1:0] d_d,
    input  wire [CH*NW-1:0]     d_n,
    input  wire [((CH > 1) ? $clog2(CH) : 1)-1:0] r_ch,
    input  wire                 r_ent,
    input  wire [NP*LQW-1:0]    r_q,
    input  wire [NP*KSW-1:0]    r_i,
    output reg  [NP*NB*8-1:0]   r_word
);
    localparam AW = ROWS * K;
    reg [CH*2*ROWS*KS*8-1:0] st;                     // byte (((ch*2 + entry)*ROWS + r)*KS + i)
    genvar gc, ge, gr, gi;
    generate
        for (gc = 0; gc < CH; gc = gc + 1) begin : g_ch
            for (ge = 0; ge < 2; ge = ge + 1) begin : g_ent
                for (gr = 0; gr < ROWS; gr = gr + 1) begin : g_row
                    for (gi = 0; gi < KS; gi = gi + 1) begin : g_key
                        localparam B0 = (((gc*2 + ge)*ROWS + gr)*KS + gi)*8;
                        always @(posedge clk)
                            if (d_v[gc] && d_n[gc*NW] == ge)
                                st[B0 +: 8] <= {d_d[gc*AW + gr*K + COL0 + gi], st[B0 + 1 +: 7]};
                    end
                end
            end
        end
    endgenerate
    integer p, ch, e, r, i;
    always @(*) begin
        r_word = 0;
        for (p = 0; p < NP; p = p + 1)
            for (ch = 0; ch < CH; ch = ch + 1)
                for (e = 0; e < 2; e = e + 1)
                    for (r = 0; r < ROWS; r = r + 1)
                        for (i = 0; i < KS; i = i + 1)
                            if (ch == r_ch && e == r_ent && (r / NB) == r_q[p*LQW +: LQW] && i == r_i[p*KSW +: KSW])
                                r_word[(p*NB + (r % NB))*8 +: 8] = st[(((ch*2 + e)*ROWS + r)*KS + i)*8 +: 8];
    end
endmodule


// ============================================================================
// dimm_slice — one slice (spec §4): the A memories (blocked, A_GRP crossbars each, one write port each), the B
// memories, the wave sequencer, XS crossbars and the first log2 KS levels of each group's per-plane count tree.
// Crossbar f = g + NG*i: column i, group g; window (g + NG*t, i) in wave t.
// The feed (spec §4, §5, §10 item 11): wave t's A words are read at t_go + 10t + j into each crossbar's next-word
// register and load its shift register at t_go + 10t + START_LAT - 1; the job's B words are read in wave 0 straight
// into each column's rotating register. Plane b of row r of the crossbar's input is bit b of A byte r/N; with X_ADD
// (R >= 2I) row I + r carries bit b of B byte r mod N and the crossbar's row sum adds them (the identity stacked
// twice), else a bit-serial full adder per row forms trunc8(A + B) (the carry cleared at plane 0, the carry out of
// plane 7 dropped). The drain: x = (e & d_on) ^ s_A ^ s_B per plane, the n_neg plane first (e gated off); the tree
// counts each plane's ones per (group, element).
// ============================================================================
module dimm_slice #(
    parameter M = 64, N = 16, R = 64, C = 64, P = 8,
    parameter XS = 16, NG = 2, KS = 8, T_W = 4, A_GRP = 8, B_GRP = 8,
    parameter A_SLOTS = 1, B_SLOTS = 1, A_SIGNED = 1, B_SIGNED = 1,
    parameter SAW = 1, SBW = 1, AAW = 5, BAW = 4, NAB = 2, LNBMW = 1, CW = 4, CTLW = 6
) (
    input  wire                  clk,
    input  wire                  reset,
    input  wire [7:0]            weight_in,
    input  wire                  weight_en,
    input  wire [NAB-1:0]        a_we,
    input  wire [NAB*AAW-1:0]    a_wa,
    input  wire [NAB*(((R < C) ? R : C) / N)*8-1:0] a_wd,
    input  wire                  b_we,
    input  wire [LNBMW-1:0]      b_wm,
    input  wire [BAW-1:0]        b_wa,
    input  wire [N*8-1:0]        b_wd,
    input  wire                  start,
    input  wire [SAW-1:0]        a_sel,
    input  wire [SBW-1:0]        b_sel,
    output wire                  idle,
    output wire [CTLW-1:0]       o_ctl,
    output wire [NG*((R < C) ? R : C)*CW-1:0] o_cnt
);
    localparam I    = (R < C) ? R : C;
    localparam XADD = (R >= 2 * I) ? 1 : 0;       // the add is the crossbar's row sum (spec §10 item 11)
    localparam NB   = I / N;
    localparam LTW  = (T_W > 1) ? $clog2(T_W) : 1;
    localparam TWW  = $clog2(T_W + 1);
    localparam LN   = $clog2(N);
    localparam LNG  = $clog2(NG);
    localparam LG   = $clog2(A_GRP);
    localparam LGW  = (LG > 0) ? LG : 1;
    localparam NBM  = KS / B_GRP;
    localparam LBG  = $clog2(B_GRP);
    localparam LBGW = (LBG > 0) ? LBG : 1;
    localparam PW   = $clog2(P);

    function sgn;
        input [7:0] b;
        sgn = b[7] ^ (&b[6:0]);
    endfunction

    // ---- run state ------------------------------------------------------------------------------------------------
    // The memory read addresses come from this slice's registers only (Fmax, spec §10 item 7): the slots load from
    // the job register (a_sel / b_sel = the top's ja_q / jb_q) while the slice is idle; the top decides only when
    // every slice is idle, and its job register holds the job through the start cycle.
    wire          take_start = start && idle;
    reg [SAW-1:0] a_sel_q;
    reg [SBW-1:0] b_sel_q;
    always @(posedge clk) begin
        if (reset) begin a_sel_q <= 0; b_sel_q <= 0; end
        else if (idle) begin a_sel_q <= a_sel; b_sel_q <= b_sel; end
    end

    // ---- the wave sequencer (the drain side follows this slice's crossbar 0) -----------------------------------
    wire             rd_a, rd_b, ld, fv, d_pre, d_on, d_last;
    wire [LGW-1:0]   rd_i;
    wire [TWW-1:0]   rd_t;
    wire [PW-1:0]    fp, db;
    wire [XS-1:0]    x_ready, x_full, x_done, x_sad;
    dimm_tile_ctrl #(.T_W(T_W), .A_GRP(A_GRP), .B_GRP(B_GRP), .P(P), .TWW(TWW), .LGW(LGW)) u_ctrl (
        .clk(clk), .reset(reset), .start(take_start),
        .leader_sad(x_sad[0]), .leader_full(x_full[0]), .leader_done(x_done[0]),
        .rd_a(rd_a), .rd_b(rd_b), .rd_i(rd_i), .rd_t(rd_t), .ld(ld), .fv(fv), .fp(fp),
        .d_pre(d_pre), .d_on(d_on), .db(db), .d_last(d_last), .idle(idle));
    reg            rda_d, rdb_d;                  // a memory read's data this cycle
    reg [LGW-1:0]  rdi_d;                         // its index
    always @(posedge clk) begin
        rda_d <= !reset && rd_a;
        rdb_d <= !reset && rd_b;
        rdi_d <= rd_i;
    end

    // ---- B memories (B_GRP columns each, word {slot, index}); read in wave 0 (once per job) ---------------------
    wire [LBGW-1:0]    bi_rd = rd_i[LBGW-1:0] & (B_GRP - 1);
    wire [NBM*N*8-1:0] bm_rd;
    genvar gm;
    generate
        for (gm = 0; gm < NBM; gm = gm + 1) begin : g_bm
            reg [N*8-1:0] mem [0:(B_SLOTS << LBGW) - 1];
            reg [N*8-1:0] rq;
            wire [SBW+LBGW-1:0] ra = {b_sel_q, bi_rd};
            always @(posedge clk) begin
                if (b_we && (NBM == 1 || b_wm == gm)) mem[b_wa] <= b_wd;
                rq <= mem[ra];
            end
            assign bm_rd[gm*N*8 +: N*8] = rq;
        end
    endgenerate

    // ---- A memories (blocked: crossbar f in memory f >> LG at index f mod A_GRP; each with its own write port) --
    wire [SAW+LTW+LGW-1:0] a_ra = {a_sel_q, rd_t[LTW-1:0], rd_i};
    wire [NAB*NB*8-1:0]    ab_rd;
    genvar gk;
    generate
        for (gk = 0; gk < NAB; gk = gk + 1) begin : g_a
            reg [NB*8-1:0] mem [0:(A_SLOTS << (LTW + LGW)) - 1];
            reg [NB*8-1:0] rq;
            always @(posedge clk) begin
                if (a_we[gk]) mem[a_wa[gk*AAW +: AAW]] <= a_wd[gk*NB*8 +: NB*8];
                rq <= mem[a_ra];
            end
            assign ab_rd[gk*NB*8 +: NB*8] = rq;
        end
    endgenerate

    // ---- per column: the job's B word (loaded once per job, rotated per byte while feeding) and its signs --------
    wire [KS*N-1:0] b_bit;                        // column i, byte n: this plane's bit (the byte's bit 0)
    wire [KS*N-1:0] b_sgn;                        // column i, byte n: the sign of LB[k][n]
    genvar gs, gb;
    generate
        for (gs = 0; gs < KS; gs = gs + 1) begin : g_b
            localparam BM = gs >> LBG;
            localparam BI = gs & (B_GRP - 1);
            reg [N*8-1:0] sb;                         // the job's B word: loaded from the read in wave 0 (before the
            reg [N-1:0]   sg;                         //   feed), rotated per byte while feeding (spec §4)
            integer j;
            always @(posedge clk) begin
                if (rdb_d && rdi_d[LBGW-1:0] == BI) begin
                    sb <= bm_rd[BM*N*8 +: N*8];
                    for (j = 0; j < N; j = j + 1) sg[j] <= (B_SIGNED != 0) && sgn(bm_rd[BM*N*8 + j*8 +: 8]);
                end
                else if (fv)                          // plane b: bit b of every byte at the byte's bit 0
                    for (j = 0; j < N; j = j + 1) sb[j*8 +: 8] <= {sb[j*8], sb[j*8 + 1 +: 7]};
            end
            for (gb = 0; gb < N; gb = gb + 1) begin : g_n
                assign b_bit[gs*N + gb] = sb[gb*8];
                assign b_sgn[gs*N + gb] = sg[gb];
            end
        end
    endgenerate

    // ---- per crossbar f = g + NG*i: the wave's A word, its shift register and signs; the crossbar's input (X_ADD:
    //      rows r and I + r; else the I bit-serial adders); the crossbar (EXP); the drain's sign XOR -------------
    wire [XS*I-1:0] x_all;
    genvar gf, gr;
    generate
        for (gf = 0; gf < XS; gf = gf + 1) begin : g_x
            localparam COL = gf >> LNG;
            localparam GB  = gf >> LG;
            localparam GI  = gf & (A_GRP - 1);
            reg  [NB*8-1:0] na, sa;
            reg  [NB-1:0]   sga, sda;                 // the A signs of the wave being fed / drained
            integer j;
            always @(posedge clk) begin
                if (rda_d && rdi_d == GI) na <= ab_rd[GB*NB*8 +: NB*8];
                if (ld) begin
                    sa <= na;
                    for (j = 0; j < NB; j = j + 1) sga[j] <= (A_SIGNED != 0) && sgn(na[j*8 +: 8]);
                end
                else if (fv) sa <= sa >> 1;
                if (fv && fp == P - 1) sda <= sga;    // the MSB plane: the signs of the wave about to drain (spec §4)
            end
            // row r = element r of the window: A byte r / N, B byte r mod N of the column
            wire [I-1:0] fa, fb, s_d, s_b;
            for (gr = 0; gr < I; gr = gr + 1) begin : g_r
                assign fa[gr]  = sa[(gr >> LN) << 3];
                assign fb[gr]  = b_bit[COL*N + (gr & (N - 1))];
                assign s_d[gr] = sda[gr >> LN];
                assign s_b[gr] = b_sgn[COL*N + (gr & (N - 1))];
            end
            wire [R-1:0] x_in;                         // the crossbar's plane: rows past the extent read 0
            if (XADD) begin : g_xadd                   // the add is the crossbar's row sum: rows r (L_A), I + r (L_B)
                assign x_in[I-1:0]   = fa;
                assign x_in[2*I-1:I] = fb;
                if (R > 2*I) begin : g_z
                    assign x_in[R-1:2*I] = {(R-2*I){1'b0}};
                end
            end else begin : g_fadd                    // a square array: I bit-serial adders, the carry cleared at plane 0
                reg  [I-1:0] cy;
                wire [I-1:0] cin = (fp == 0) ? {I{1'b0}} : cy;
                always @(posedge clk) if (fv) cy <= (fa & fb) | (fa & cin) | (fb & cin);
                assign x_in[I-1:0] = fa ^ fb ^ cin;
                if (R > I) begin : g_z
                    assign x_in[R-1:I] = {(R-I){1'b0}};
                end
            end
            wire [R-1:0] dpe_in = weight_en ? {{(R-8){1'b0}}, weight_in} : x_in;
            wire [C-1:0] dpe_out;
            dpe #(
                .KERNEL_WIDTH(R), .NUM_COLS(C), .PRECISION(P), .ROWS_USED(XADD ? 2*I : I), .COLS_USED(I)
            ) u_dpe (
                .clk(clk), .reset(reset), .data_in(dpe_in), .nl_dpe_control(2'b10),   // EXP
                .shift_add_control(1'b0), .w_buf_en(fv), .shift_add_bypass(1'b0),
                .load_output_reg(1'b0), .load_input_reg(weight_en),
                .MSB_SA_Ready(x_ready[gf]), .data_out(dpe_out), .dpe_done(x_done[gf]), .reg_full(x_full[gf]),
                .shift_add_done(x_sad[gf]));
            // the drain: x = (e & d_on) ^ s_A ^ s_B straight into the slice's count tree (its one register is
            // SLICE_LAT): the n_neg plane (d_pre; e gated off) counts the signs, the 8 drain planes count e ^ s; the
            // tree's input while idle is ignored (its control carries no valid)
            assign x_all[gf*I +: I] = (dpe_out[I-1:0] & {I{d_on}}) ^ s_d ^ s_b;
        end
    endgenerate

    // ---- the slice's count tree: per group g, element r, the KS crossbars g + NG*i, registered once ----------------
    wire [CTLW-1:0] cx = {d_pre, d_on, d_on ? db : {PW{1'b0}}, d_last};   // the drain control: {n_neg, valid, plane, last}
    dimm_ptree #(.NG(NG), .DC(KS), .NE(I), .IW(1), .CTLW(CTLW), .OREG(1)) u_tree (
        .clk(clk), .reset(reset), .in_ctl(cx), .in_val(x_all), .out_ctl(o_ctl), .out_val(o_cnt));
endmodule


// ============================================================================
// dimm_tile_ctrl — the wave sequencer of a slice (spec §5 Core). From the start cycle t_go (period 0, cycle 0), each
// wave t owns the 10-cycle period t: its A memory reads at cycles 0 .. A_GRP-1 (the B words in period 0, cycles
// 0 .. B_GRP-1), the shift-register load at cycle START_LAT - 1 (START_LAT = A_GRP + 2), then its P feed planes.
// The drain side follows the leader crossbar: the n_neg plane at its shift_add_done (the cycle before the first
// drain plane), the P drain planes under its reg_full, the wave's end at its dpe_done. The slice is idle again from
// the job's last dpe_done + 1 (= the last drain plane + IDLE_LAT).
// ============================================================================
module dimm_tile_ctrl #(
    parameter T_W = 4, A_GRP = 8, B_GRP = 8, P = 8, TWW = 3, LGW = 3
) (
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 start,
    input  wire                 leader_sad,
    input  wire                 leader_full,
    input  wire                 leader_done,
    output wire                 rd_a,
    output wire                 rd_b,
    output wire [LGW-1:0]       rd_i,
    output wire [TWW-1:0]       rd_t,
    output wire                 ld,
    output reg                  fv,
    output reg  [$clog2(P)-1:0] fp,
    output wire                 d_pre,
    output wire                 d_on,
    output reg  [$clog2(P)-1:0] db,
    output wire                 d_last,
    output wire                 idle
);
    localparam SL = A_GRP + 2;                    // START_LAT
    localparam PH = P + 2;                        // a wave every P + 2 cycles
    reg           run;
    reg [3:0]     ph;                             // the cycle in the period
    reg [TWW-1:0] pc;                             // the period: the wave whose reads and load run in it (<= T_W)
    reg [TWW-1:0] dw;                             // the waves drained in this job
    wire           on   = start || run;
    wire [3:0]     ph_n = run ? ph : 4'd0;        // the start cycle is cycle 0 of period 0
    wire [TWW-1:0] pc_n = run ? pc : {TWW{1'b0}};
    wire           feed = on && (pc_n < T_W);
    assign rd_a   = feed && (ph_n < A_GRP);
    assign rd_b   = rd_a && (pc_n == 0) && (ph_n < B_GRP);
    assign rd_i   = ph_n[LGW-1:0];
    assign rd_t   = pc_n;
    assign ld     = feed && (ph_n == SL - 1);
    assign d_pre  = leader_sad;
    assign d_on   = leader_full;
    assign d_last = (dw == T_W - 1);
    assign idle   = !run;
    always @(posedge clk) begin
        if (reset) begin
            run <= 1'b0; ph <= 0; pc <= 0; dw <= 0; fv <= 1'b0; fp <= 0; db <= 0;
        end
        else begin
            if (start) run <= 1'b1;
            if (on) begin
                ph <= (ph_n == PH - 1) ? 4'd0 : ph_n + 1'b1;
                pc <= (ph_n == PH - 1 && pc_n < T_W) ? pc_n + 1'b1 : pc_n;
            end
            if (ld) begin fv <= 1'b1; fp <= 0; end
            else if (fv) begin
                if (fp == P - 1) fv <= 1'b0;
                fp <= fp + 1'b1;
            end
            if (d_pre) db <= 0;
            else if (d_on) db <= db + 1'b1;
            if (leader_done) begin
                dw <= (dw == T_W - 1) ? {TWW{1'b0}} : dw + 1'b1;
                if (dw == T_W - 1) run <= 1'b0;
            end
        end
    end
endmodule


// ============================================================================
// dimm_ptree — per-plane count trees, one per (group, element): log2 DC levels of adders, a balanced tree computed in a
// function on the element's own terms (one net per element, so a simulator wakes each tree only for its inputs), and
// the result registered once with its control (the trims of 2026-10-23; until then every level registered). Input
// index g + d*NG (group g, term d = 0 .. DC-1), NE elements of IW-bit unsigned counts each; outputs IW + log2 DC bits.
// DC = 1: no level; the input is registered when OREG = 1, else it passes through.
// ============================================================================
module dimm_ptree #(
    parameter NG = 2, DC = 2, NE = 64, IW = 1, CTLW = 6, OREG = 0
) (
    input  wire                             clk,
    input  wire                             reset,
    input  wire [CTLW-1:0]                  in_ctl,
    input  wire [DC*NG*NE*IW-1:0]           in_val,
    output wire [CTLW-1:0]                  out_ctl,
    output wire [NG*NE*(IW+$clog2(DC))-1:0] out_val
);
    localparam TL = $clog2(DC);
    localparam OW = IW + TL;
    // the sum of DC IW-bit terms as a balanced tree, reduced in place: level l adds the pairs of level l - 1
    function [OW-1:0] tsum;
        input [DC*IW-1:0] v;
        reg   [DC*OW-1:0] t;
        integer k, l;
        begin
            for (k = 0; k < DC; k = k + 1) t[k*OW +: OW] = v[k*IW +: IW];
            for (l = 0; l < TL; l = l + 1)
                for (k = 0; k < (DC >> (l + 1)); k = k + 1)
                    t[k*OW +: OW] = t[2*k*OW +: OW] + t[(2*k + 1)*OW +: OW];
            tsum = t[0 +: OW];
        end
    endfunction
    wire [NG*NE*OW-1:0] sum;                      // (group g, element e) at ((g*NE + e)*OW) +: OW
    genvar ga, gd, ge;
    generate
        for (ga = 0; ga < NG; ga = ga + 1) begin : g_a
            for (ge = 0; ge < NE; ge = ge + 1) begin : g_e
                wire [DC*IW-1:0] v;                // the element's DC terms
                for (gd = 0; gd < DC; gd = gd + 1) begin : g_d
                    assign v[gd*IW +: IW] = in_val[((ga + gd*NG)*NE + ge)*IW +: IW];
                end
                assign sum[(ga*NE + ge)*OW +: OW] = tsum(v);
            end
        end
        if (TL > 0 || OREG != 0) begin : g_reg                // the one register of the tree (SLICE_LAT / SHARED_LAT)
            reg [NG*NE*OW-1:0] oq;
            reg [CTLW-1:0]     cq;
            always @(posedge clk) begin
                oq <= sum;
                cq <= reset ? {CTLW{1'b0}} : in_ctl;
            end
            assign out_val = oq;
            assign out_ctl = cq;
        end else begin : g_pass
            assign out_val = sum;
            assign out_ctl = in_ctl;
        end
    endgenerate
endmodule
