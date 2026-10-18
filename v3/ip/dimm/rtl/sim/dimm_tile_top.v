// ============================================================================
// dimm_tile_top.v — tiled DIMM (v3, contract ip/dimm/spec.md). SIMULATION twin: behavior `dpe`
// (ip/xbar/rtl/sim/dpe_nldpe.v) with weight programming on weight_in / weight_en (testbenches preload the weights,
// PLAN R8). VTR twin: ip/dimm/rtl/vtr/dimm_tile_top.v (bare `dpe`, no weight path). Model: ip/dimm/model.py.
//
// One job: C[m][n] = sum_k (-1)^(sA ^ sB) EXP(trunc8(LA[m][k] + LB[k][n])), then sat8(C >> NARROW) if NARROW > 0.
// Operand ports take row-major rows (int8 through a LOG converter per channel, or log bytes); the module stages
// them into the farm's word layout, holds A_SLOTS / B_SLOTS matrices, takes jobs (slot pair, release bit) on a
// burst-on-ready job port into a one-entry register, and pushes C windows on NG channels against C_CAP credits.
//
// Structure (spec §4): per channel converter -> distribution register (word index, window count) -> the slices.
// The loaders' control (completion, writer counters, memory addresses) is shared, the staging data and the write
// ports are per slice. Slice s = the farm's columns [s*KS, (s+1)*KS) of all NG groups (X_S' crossbars): its A and B
// memories, its copy of the v2 window engine (leader = its crossbar 0), its crossbars, the first log2 KS tree
// levels. Shared: slots, jobs, credits, the last log2 NS tree levels, narrowing, the output channels.
// Every latency is named in spec §5; the model times every port event by the same rules (GATE 2: Delta = 0).
// Domain: the elaboration guards below (= DimmTile.guard).
// ============================================================================

`timescale 1ns / 1ps

module dimm_tile_top #(
    parameter M        = 64,
    parameter N        = 16,
    parameter K        = 16,
    parameter R        = 64,
    parameter C        = 64,
    parameter BUF      = 40,
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
    parameter SBW      = (B_SLOTS > 1) ? $clog2(B_SLOTS) : 1
) (
    input  wire                  clk,
    input  wire                  reset,
    // weight programming (simulation twin only): R*C identity strobes broadcast to every crossbar
    input  wire [7:0]            weight_in,
    input  wire                  weight_en,
    // A port: A_CH channels, push + credit
    input  wire [A_CH-1:0]       a_v,
    input  wire [A_CH*BUF-1:0]   a_d,
    output reg                   a_cr,
    // B port
    input  wire [B_CH-1:0]       b_v,
    input  wire [B_CH*BUF-1:0]   b_d,
    output reg                   b_cr,
    // job port: burst on ready into the job register
    input  wire                  j_v,
    input  wire [SAW-1:0]        j_a,
    input  wire [SBW-1:0]        j_b,
    input  wire                  j_ra,
    output wire                  j_rdy,
    // C port: NG channels, push + credit
    output wire [NG-1:0]         c_v,
    output wire [NG*5*EC-1:0]    c_d,
    input  wire                  c_cr,
    output wire                  busy
);
    // ---- derived (spec §2) -------------------------------------------------------------------------------------
    localparam EPS   = BUF / 8;
    localparam NB    = I / N;                                   // rows per farm window
    localparam W_K   = (M * N) / I;                             // windows per column
    localparam T_W   = W_K / NG;                                // waves
    localparam LW    = (I * 8 + BUF - 1) / BUF;                 // words per farm window
    localparam XS    = (X_S < N_E) ? X_S : N_E;                 // crossbars per slice
    localparam NS    = N_E / XS;                                // slices
    localparam KS    = XS / NG;                                 // columns per slice
    localparam AGC   = (LW < XS) ? LW : XS;
    localparam A_GRP = 1 << ($clog2(AGC + 1) - 1);              // crossbars per A memory
    localparam B_GRP = (A_GRP < KS) ? A_GRP : KS;               // columns per B memory
    localparam LBL   = TRANS_B ? K : N;                         // B row length
    localparam A_W   = A_ROWS * K;                              // elements per A input window
    localparam LWA   = (A_W + 4) / 5;                           // words per A input window
    localparam LWB   = (LBL + 4) / 5;                           // words per B row
    localparam NQW   = A_ROWS / NB;                             // NB-row groups per A input window (window frame)
    localparam EW    = NQW * KS;                                // words a window gives a slice (window frame)
    localparam WPMA  = M / A_ROWS;                              // A input windows per matrix
    localparam KSW   = (KS > 1) ? $clog2(KS) : 1;
    localparam LKS   = $clog2(KS);
    localparam LNG   = $clog2(NG);
    localparam LG    = $clog2(A_GRP);
    localparam LBG   = $clog2(B_GRP);
    localparam LTW   = (T_W > 1) ? $clog2(T_W) : 1;
    localparam NAB   = XS / A_GRP;                              // A memories per slice
    localparam NBM   = KS / B_GRP;                              // B memories per slice
    localparam LNABW = (NAB > 1) ? $clog2(NAB) : 1;
    localparam LNBMW = (NBM > 1) ? $clog2(NBM) : 1;
    localparam AAW   = SAW + LTW + ((LG > 0) ? LG : 1);         // A memory address {slot, wave, index}
    localparam BAW   = SBW + ((LBG > 0) ? LBG : 1);             // B memory address {slot, index}
    // (LGW / LBGW below: the index field widths)
    localparam VWS   = 9 + LKS;                                 // a slice's partial sum
    localparam LNS   = $clog2(NS);
    localparam TWW   = $clog2(T_W + 1) + 1;                     // engine counter widths (v2)
    localparam CW    = $clog2(LW + 1) + 1;
    localparam CTLW  = 2 + CW + TWW;                            // {valid, last, word, wave}
    localparam CCW   = $clog2(C_CAP + 1);
    localparam LCHA  = (A_CH > 1) ? $clog2(A_CH) : 0;
    localparam LCHB  = (B_CH > 1) ? $clog2(B_CH) : 0;
    localparam LNB   = $clog2(NB);
    localparam LGW   = (LG > 0) ? LG : 1;
    localparam LBGW  = (LBG > 0) ? LBG : 1;

    generate
        if ((M & (M-1)) != 0 || (N & (N-1)) != 0 || (K & (K-1)) != 0 || (N_E & (N_E-1)) != 0
            || (X_S & (X_S-1)) != 0 || (I & (I-1)) != 0 || I < 8 || N > I || M * N < I || K < 2 || K > N_E
            || NG > W_K || NG > XS || K > I || LBL > I || BUF != 40 || EPS != 5
            || (A_CH & (A_CH-1)) != 0 || (B_CH & (B_CH-1)) != 0 || A_SLOTS < 1 || B_SLOTS < 1 || C_CAP < 1
            || (A_FRAME == 0 && (A_ROWS != 1 || A_CH > NB))
            || (A_FRAME == 1 && (A_ROWS < NB || (A_ROWS % NB) != 0 || (M % A_ROWS) != 0 || A_DOM != 1
                                 || (NQW & (NQW-1)) != 0))
            || NARROW < 0 || NARROW >= VW)
        begin : g_bad_param
            dimm_tile_parameter_outside_domain u_bad ();
        end
    endgenerate

    genvar gch, gs, gi, gr, gq, gj, gm;

    // =============================================================================================================
    // 1  streams: converter per channel (int8) or the port word (log), then the distribution register (DIST_LAT 1)
    //    with the framing counters (word in window, windows on the channel; low bits only)
    // =============================================================================================================
    localparam ANW = (A_FRAME == 0) ? ((2 * NB / A_CH > 1) ? $clog2(2 * NB / A_CH) : 1) : 1;    // window count bits
    localparam BNW = TRANS_B ? ((2 * N / B_CH > 1) ? $clog2(2 * N / B_CH) : 1) : 1;
    localparam ACW = $clog2(LWA + 1);
    localparam BCW = $clog2(LWB + 1);
    wire [A_CH-1:0]     sa_v;
    wire [A_CH*BUF-1:0] sa_d;
    wire [B_CH-1:0]     sb_v;
    wire [B_CH*BUF-1:0] sb_d;
    wire [A_CH+B_CH-1:0] conv_err;
    generate
        for (gch = 0; gch < A_CH; gch = gch + 1) begin : g_sa
            if (A_DOM == 0) begin : g_conv
                dimm_conv #(.R(R), .C(C), .BUF(BUF), .P(P), .L(K)) u_conv (
                    .clk(clk), .reset(reset), .weight_in(weight_in), .weight_en(weight_en),
                    .x_v(a_v[gch]), .x_d(a_d[gch*BUF +: BUF]),
                    .s_v(sa_v[gch]), .s_d(sa_d[gch*BUF +: BUF]), .err(conv_err[gch]));
            end else begin : g_log
                assign sa_v[gch] = a_v[gch];
                assign sa_d[gch*BUF +: BUF] = a_d[gch*BUF +: BUF];
                assign conv_err[gch] = 1'b0;
            end
        end
        for (gch = 0; gch < B_CH; gch = gch + 1) begin : g_sb
            if (B_DOM == 0) begin : g_conv
                dimm_conv #(.R(R), .C(C), .BUF(BUF), .P(P), .L(LBL)) u_conv (
                    .clk(clk), .reset(reset), .weight_in(weight_in), .weight_en(weight_en),
                    .x_v(b_v[gch]), .x_d(b_d[gch*BUF +: BUF]),
                    .s_v(sb_v[gch]), .s_d(sb_d[gch*BUF +: BUF]), .err(conv_err[A_CH + gch]));
            end else begin : g_log
                assign sb_v[gch] = b_v[gch];
                assign sb_d[gch*BUF +: BUF] = b_d[gch*BUF +: BUF];
                assign conv_err[A_CH + gch] = 1'b0;
            end
        end
    endgenerate

    reg  [A_CH-1:0]     da_v;
    reg  [A_CH*BUF-1:0] da_d;
    reg  [A_CH*ACW-1:0] da_c;
    reg  [A_CH*ANW-1:0] da_n;
    reg  [B_CH-1:0]     db_v;
    reg  [B_CH*BUF-1:0] db_d;
    reg  [B_CH*BCW-1:0] db_c;
    reg  [B_CH*BNW-1:0] db_n;
    reg  [A_CH*ACW-1:0] fa_c;                                    // framing state: next word index, windows seen
    reg  [A_CH*ANW-1:0] fa_n;
    reg  [B_CH*BCW-1:0] fb_c;
    reg  [B_CH*BNW-1:0] fb_n;
    integer ci;
    always @(posedge clk) begin
        for (ci = 0; ci < A_CH; ci = ci + 1) begin
            da_v[ci] <= !reset && sa_v[ci];
            da_d[ci*BUF +: BUF] <= sa_d[ci*BUF +: BUF];
            da_c[ci*ACW +: ACW] <= fa_c[ci*ACW +: ACW];
            da_n[ci*ANW +: ANW] <= fa_n[ci*ANW +: ANW];
            if (reset) begin
                fa_c[ci*ACW +: ACW] <= 0; fa_n[ci*ANW +: ANW] <= 0;
            end
            else if (sa_v[ci]) begin
                if (fa_c[ci*ACW +: ACW] == LWA - 1) begin
                    fa_c[ci*ACW +: ACW] <= 0;
                    fa_n[ci*ANW +: ANW] <= fa_n[ci*ANW +: ANW] + 1'b1;
                end
                else fa_c[ci*ACW +: ACW] <= fa_c[ci*ACW +: ACW] + 1'b1;
            end
        end
        for (ci = 0; ci < B_CH; ci = ci + 1) begin
            db_v[ci] <= !reset && sb_v[ci];
            db_d[ci*BUF +: BUF] <= sb_d[ci*BUF +: BUF];
            db_c[ci*BCW +: BCW] <= fb_c[ci*BCW +: BCW];
            db_n[ci*BNW +: BNW] <= fb_n[ci*BNW +: BNW];
            if (reset) begin
                fb_c[ci*BCW +: BCW] <= 0; fb_n[ci*BNW +: BNW] <= 0;
            end
            else if (sb_v[ci]) begin
                if (fb_c[ci*BCW +: BCW] == LWB - 1) begin
                    fb_c[ci*BCW +: BCW] <= 0;
                    fb_n[ci*BNW +: BNW] <= fb_n[ci*BNW +: BNW] + 1'b1;
                end
                else fb_c[ci*BCW +: BCW] <= fb_c[ci*BCW +: BCW] + 1'b1;
            end
        end
    end

    // =============================================================================================================
    // 2  A loader control (shared): row frame = transposer of NB-row groups; window frame = window stager
    // =============================================================================================================
    reg                 aw_we;            // a slice write this cycle (combinational)
    reg  [KSW-1:0]      aw_i;             // local column
    reg  [7:0]          aw_q;             // NB-row group inside the window (window frame)
    reg                 aw_buf;           // staging buffer (row frame) / entry (window frame)
    reg  [((A_CH > 1) ? $clog2(A_CH) : 1)-1:0] aw_ch;          // staging channel (window frame)
    reg  [LNABW-1:0]    aw_m;             // A memory
    reg  [AAW-1:0]      aw_a;             // its address
    wire                a_unit_end;       // the writer's last word of a staged unit
    wire                a_matrix_end;     // ... that completes an A matrix
    reg  [SAW-1:0]      a_wslot;          // the slot being written
    localparam WKW = (W_K > 1) ? $clog2(W_K) : 1;
    localparam WPW = (WPMA > 1) ? $clog2(WPMA) : 1;

    generate
        if (A_FRAME == 0) begin : g_axc
            // rows done per buffer, complete flags, the writer (spec §5 transposer)
            reg [$clog2(NB+1)-1:0] cnt0, cnt1;
            reg                    cmp0, cmp1;
            reg [KSW:0]            wi;                 // 0 idle, else the next word index + 1 ... (counts 1..KS-1)
            reg                    qb;                 // the group being emitted: buffer
            reg [WKW-1:0]          qw;                 // its window in the matrix
            integer                ch, nd0, nd1;
            reg [31:0]             rowg;
            always @(*) begin
                nd0 = 0; nd1 = 0;
                for (ch = 0; ch < A_CH; ch = ch + 1)
                    if (da_v[ch] && da_c[ch*ACW +: ACW] == LWA - 1) begin
                        rowg = (da_n[ch*ANW +: ANW] << LCHA) | ch;          // the row's index (low bits)
                        if ((rowg >> LNB) & 1) nd1 = nd1 + 1; else nd0 = nd0 + 1;
                    end
            end
            wire start_w = (wi == 0) && (qb ? cmp1 : cmp0);
            wire [KSW-1:0] i_now = (wi == 0) ? {KSW{1'b0}} : wi[KSW-1:0];
            wire last_w = (wi == 0) ? (start_w && KS == 1) : (wi[KSW-1:0] == KS - 1 || KS == 1);
            assign a_unit_end = (start_w || wi != 0) && last_w;
            assign a_matrix_end = a_unit_end && (qw == W_K - 1);
            always @(*) begin
                aw_we  = start_w || (wi != 0);
                aw_i   = i_now;
                aw_q   = 0;
                aw_buf = qb;
                aw_ch  = 0;
            end
            // crossbar f = (w mod NG) | (i << LNG): memory f >> LG, index f mod A_GRP; address {slot, wave, index}
            wire [KSW+LNG:0] f_loc = ({{(LNG+1){1'b0}}, i_now} << LNG) | (qw & (NG - 1));
            wire [WKW:0]     wav   = qw >> LNG;
            always @(*) begin
                aw_m = (NAB > 1) ? (f_loc >> LG) : 0;
                aw_a = {a_wslot, wav[LTW-1:0], f_loc[LGW-1:0]};
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
                    if (aw_we) begin
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
            // window stager (spec §5): per channel two entries; windows emitted in global order
            reg [A_CH-1:0]         cmp0, cmp1;          // entry 0 / 1 complete, per channel
            reg [$clog2(EW+1):0]   ei;                  // 0 idle, else the next word + 1
            reg [((A_CH > 1) ? $clog2(A_CH) : 1)-1:0] wch;   // the window being emitted: channel
            reg                    went;                //   entry
            reg [WPW-1:0]          wl;                  //   window in the matrix
            reg [WKW+1:0]          wbase;               //   its first farm window (wl * NQW)
            integer                ch;
            wire want = went ? cmp1[wch] : cmp0[wch];
            wire start_w = (ei == 0) && want;
            wire [$clog2(EW+1):0] e_idx = (ei == 0) ? 0 : ei;
            wire last_w = (ei == 0) ? (start_w && EW == 1) : (ei == EW - 1);
            wire [KSW-1:0] i_now = e_idx & (KS - 1);
            wire [7:0]     q_now = e_idx >> LKS;
            assign a_unit_end = (start_w || ei != 0) && last_w;
            assign a_matrix_end = a_unit_end && (wl == WPMA - 1);
            wire [WKW+1:0] w_a = wbase + q_now;                      // farm window
            wire [KSW+LNG:0] f_loc = ({{(LNG+1){1'b0}}, i_now} << LNG) | (w_a & (NG - 1));
            wire [WKW+1:0] wav = w_a >> LNG;
            always @(*) begin
                aw_we  = start_w || (ei != 0);
                aw_i   = i_now;
                aw_q   = q_now;
                aw_buf = went;
                aw_ch  = wch;
                aw_m   = (NAB > 1) ? (f_loc >> LG) : 0;
                aw_a   = {a_wslot, wav[LTW-1:0], f_loc[LGW-1:0]};
            end
            always @(posedge clk) begin
                if (reset) begin
                    cmp0 <= 0; cmp1 <= 0; ei <= 0; wch <= 0; went <= 1'b0; wl <= 0; wbase <= 0; a_wslot <= 0;
                end
                else begin
                    for (ch = 0; ch < A_CH; ch = ch + 1)
                        if (da_v[ch] && da_c[ch*ACW +: ACW] == LWA - 1) begin
                            if (da_n[ch*ANW] == 1'b1) cmp1[ch] <= 1'b1; else cmp0[ch] <= 1'b1;
                        end
                    if (aw_we) begin
                        if (last_w) begin
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
                        else ei <= (ei == 0) ? 1 : ei + 1'b1;
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
                    if (db_v[ch] && db_c[ch*BCW +: BCW] == LWB - 1) begin
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
            // byte n of a row = lane n mod 5 of its word n / 5 (constants)
            reg  [B_CH*2*N*8-1:0] rr;
            reg  [B_CH-1:0] rdy0, rdy1;
            reg  [((B_CH > 1) ? $clog2(B_CH) : 1)-1:0] tch;   // next row to write: channel
            reg             treg;                              //   register
            reg  [$clog2(K+1)-1:0] tk;                         //   its row k in the matrix
            integer ch;
            for (gch = 0; gch < B_CH; gch = gch + 1) begin : g_rch
                for (gr = 0; gr < 2; gr = gr + 1) begin : g_rr
                    for (gj = 0; gj < N; gj = gj + 1) begin : g_rb
                        always @(posedge clk)
                            if (db_v[gch] && db_c[gch*BCW +: BCW] == (gj / 5) && db_n[gch*BNW] == gr)
                                rr[((gch*2 + gr)*N + gj)*8 +: 8] <= db_d[gch*BUF + (gj % 5)*8 +: 8];
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
                        if (db_v[ch] && db_c[ch*BCW +: BCW] == LWB - 1) begin
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
    wire [NS*CTLW-1:0]       sl_ctl;
    wire [NS*NG*5*VWS-1:0]   sl_val;
    generate
        for (gs = 0; gs < NS; gs = gs + 1) begin : g_sl
            // A staging data of this slice -> the A write word
            wire [NB*8-1:0] a_word;
            if (A_FRAME == 0) begin : g_ast
                dimm_xpose_stage #(.G(NB), .KS(KS), .KSW(KSW), .CH(A_CH), .LW(LWA), .CWD(ACW), .NW(ANW),
                                   .COL0(gs * KS), .BUF(BUF)) u_st (
                    .clk(clk), .d_v(da_v), .d_d(da_d), .d_c(da_c), .d_n(da_n),
                    .r_buf(aw_buf), .r_i(aw_i), .r_word(a_word));
            end else begin : g_awst
                dimm_wstage #(.ROWS(A_ROWS), .K(K), .KS(KS), .KSW(KSW), .NB(NB), .CH(A_CH), .LW(LWA), .CWD(ACW),
                              .NW(ANW), .COL0(gs * KS), .BUF(BUF)) u_st (
                    .clk(clk), .d_v(da_v), .d_d(da_d), .d_c(da_c), .d_n(da_n),
                    .r_ch(aw_ch), .r_ent(aw_buf), .r_q(aw_q), .r_i(aw_i), .r_word(a_word));
            end
            // B write word
            wire [N*8-1:0] b_word;
            if (TRANS_B) begin : g_bst
                dimm_xpose_stage #(.G(N), .KS(KS), .KSW(KSW), .CH(B_CH), .LW(LWB), .CWD(BCW), .NW(BNW),
                                   .COL0(gs * KS), .BUF(BUF)) u_st (
                    .clk(clk), .d_v(db_v), .d_d(db_d), .d_c(db_c), .d_n(db_n),
                    .r_buf(bw_buf), .r_i(bw_i), .r_word(b_word));
            end else begin : g_brd
                assign b_word = bw_rd;
            end
            dimm_slice #(.M(M), .N(N), .R(R), .C(C), .BUF(BUF), .P(P), .XS(XS), .NG(NG), .KS(KS), .T_W(T_W),
                         .W_K(W_K), .A_GRP(A_GRP), .B_GRP(B_GRP), .A_SLOTS(A_SLOTS), .B_SLOTS(B_SLOTS),
                         .A_SIGNED(A_SIGNED), .B_SIGNED(B_SIGNED), .SAW(SAW), .SBW(SBW), .AAW(AAW), .BAW(BAW),
                         .LNABW(LNABW), .LNBMW(LNBMW), .VWS(VWS), .TWW(TWW), .CW(CW)) u_slice (
                .clk(clk), .reset(reset), .weight_in(weight_in), .weight_en(weight_en),
                .a_we(aw_we), .a_wm(aw_m), .a_wa(aw_a), .a_wd(a_word),
                .b_we(bw_we && bw_sl[gs]), .b_wm(bw_m), .b_wa(bw_a), .b_wd(b_word),
                .start(go_q), .a_sel(ja_q), .b_sel(jb_q), .idle(s_idle[gs]),
                .o_ctl(sl_ctl[gs*CTLW +: CTLW]), .o_val(sl_val[gs*NG*5*VWS +: NG*5*VWS]));
        end
    endgenerate

    // =============================================================================================================
    // 6  the last log2 NS tree levels (registered), narrowing, the output channels; done
    // =============================================================================================================
    wire [CTLW-1:0]     t_ctl;
    wire [NG*5*VW-1:0]  t_val;
    dimm_tree #(.NG(NG), .DC(NS), .IW(VWS), .OW(VW), .CTLW(CTLW)) u_top (
        .clk(clk), .reset(reset), .in_ctl(sl_ctl[0 +: CTLW]), .in_val(sl_val), .out_ctl(t_ctl), .out_val(t_val));
    wire tv_v = t_ctl[CTLW-1];
    wire tv_l = t_ctl[CTLW-2];
    reg  done_q;
    always @(posedge clk) done_q <= !reset && tv_l;
    assign done = done_q;
    assign c_v = {NG{tv_v}};
    generate
        for (gi = 0; gi < NG; gi = gi + 1) begin : g_out
            for (gj = 0; gj < 5; gj = gj + 1) begin : g_lane
                wire signed [VW-1:0] v = t_val[(gi*5 + gj)*VW +: VW];
                if (NARROW > 0) begin : g_nar
                    wire signed [VW-1:0] sh = v >>> NARROW;
                    assign c_d[(gi*5 + gj)*EC +: EC] = (sh > 127) ? 8'sd127 : ((sh < -128) ? -8'sd128 : sh[7:0]);
                end else begin : g_wide
                    assign c_d[(gi*5 + gj)*EC +: EC] = v;
                end
            end
        end
    endgenerate
endmodule


// ============================================================================
// dimm_conv — LOG converter of one int8 input channel: the input word register (CONV_IN_LAT 1) feeding an identity
// crossbar in LOG mode, extent L x L; its drain is the converted row (reg_full = word valid). `err`: a load strobe
// while the crossbar is not ready (the producer broke the converter period; simulation probe).
// ============================================================================
module dimm_conv #(
    parameter R = 64, C = 64, BUF = 40, P = 8, L = 16
) (
    input  wire           clk,
    input  wire           reset,
    input  wire [7:0]     weight_in,
    input  wire           weight_en,
    input  wire           x_v,
    input  wire [BUF-1:0] x_d,
    output wire           s_v,
    output wire [BUF-1:0] s_d,
    output wire           err
);
    reg [BUF-1:0] in_q;
    reg           in_v;
    always @(posedge clk) begin
        in_q <= x_d;
        in_v <= !reset && x_v;
    end
    wire rdy;
    wire [BUF-1:0] dpe_in = weight_en ? {{(BUF-8){1'b0}}, weight_in} : in_q;
    dpe #(
        .KERNEL_WIDTH(R), .NUM_COLS(C), .DPE_BUF_WIDTH(BUF), .PRECISION(P), .ROWS_USED(L), .COLS_USED(L)
    ) u_dpe (
        .clk(clk), .reset(reset), .data_in(dpe_in), .nl_dpe_control(2'b11),   // LOG
        .shift_add_control(1'b0), .w_buf_en(in_v), .shift_add_bypass(1'b0),
        .load_output_reg(1'b0), .load_input_reg(weight_en),
        .MSB_SA_Ready(rdy), .data_out(s_d), .dpe_done(), .reg_full(s_v));
    reg err_q;
    always @(posedge clk) err_q <= !reset && (err_q || (in_v && !rdy));
    assign err = err_q;
endmodule


// ============================================================================
// dimm_xpose_stage — one slice's transposer staging (spec §5): G-row groups, two buffers, bank = row mod G; the
// slice's columns COL0 .. COL0 + KS - 1 of every row. Byte (buffer, bank, i) is written from word (COL0 + i) / 5,
// lane (COL0 + i) mod 5 of a row whose index (low bits) selects that buffer and bank (constants per byte).
// Read: word i of buffer r_buf = the bytes of banks 0 .. G-1, column i (bank = byte position).
// ============================================================================
module dimm_xpose_stage #(
    parameter G = 4, KS = 8, KSW = 3, CH = 1, LW = 4, CWD = 3, NW = 2, COL0 = 0, BUF = 40
) (
    input  wire              clk,
    input  wire [CH-1:0]     d_v,
    input  wire [CH*BUF-1:0] d_d,
    input  wire [CH*CWD-1:0] d_c,
    input  wire [CH*NW-1:0]  d_n,
    input  wire              r_buf,
    input  wire [KSW-1:0]    r_i,
    output reg  [G*8-1:0]    r_word
);
    localparam LCH = (CH > 1) ? $clog2(CH) : 0;
    localparam LG  = $clog2(G);
    reg [2*G*KS*8-1:0] st;                           // byte ((buffer*G + bank)*KS + i)
    genvar gb, gk, gi;
    generate
        for (gb = 0; gb < 2; gb = gb + 1) begin : g_buf
            for (gk = 0; gk < G; gk = gk + 1) begin : g_bank
                for (gi = 0; gi < KS; gi = gi + 1) begin : g_col
                    localparam WC = (COL0 + gi) / 5, LN = (COL0 + gi) % 5;
                    localparam ROW = gb * G + gk;     // the row's index mod 2G
                    integer ch;
                    always @(posedge clk)
                        for (ch = 0; ch < CH; ch = ch + 1)
                            if (d_v[ch] && d_c[ch*CWD +: CWD] == WC
                                && ((((d_n[ch*NW +: NW] << LCH) | ch) & (2*G - 1)) == ROW))
                                st[((gb*G + gk)*KS + gi)*8 +: 8] <= d_d[ch*BUF + LN*8 +: 8];
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
// slice's KS keys. Element e = r*K + COL0 + i of a window is word e / 5, lane e mod 5 (constants per byte); the
// entry is the channel's window count mod 2. Read: NB-row group r_q, key r_i of channel r_ch, entry r_ent = NB
// bytes (rows r_q*NB .. r_q*NB + NB - 1).
// ============================================================================
module dimm_wstage #(
    parameter ROWS = 4, K = 16, KS = 8, KSW = 3, NB = 4, CH = 1, LW = 13, CWD = 4, NW = 1, COL0 = 0, BUF = 40
) (
    input  wire              clk,
    input  wire [CH-1:0]     d_v,
    input  wire [CH*BUF-1:0] d_d,
    input  wire [CH*CWD-1:0] d_c,
    input  wire [CH*NW-1:0]  d_n,
    input  wire [((CH > 1) ? $clog2(CH) : 1)-1:0] r_ch,
    input  wire              r_ent,
    input  wire [7:0]        r_q,
    input  wire [KSW-1:0]    r_i,
    output reg  [NB*8-1:0]   r_word
);
    reg [CH*2*ROWS*KS*8-1:0] st;                     // byte (((ch*2 + entry)*ROWS + r)*KS + i)
    genvar gc, ge, gr, gi;
    generate
        for (gc = 0; gc < CH; gc = gc + 1) begin : g_ch
            for (ge = 0; ge < 2; ge = ge + 1) begin : g_ent
                for (gr = 0; gr < ROWS; gr = gr + 1) begin : g_row
                    for (gi = 0; gi < KS; gi = gi + 1) begin : g_key
                        localparam EL = gr * K + COL0 + gi;
                        always @(posedge clk)
                            if (d_v[gc] && d_c[gc*CWD +: CWD] == (EL / 5) && d_n[gc*NW] == ge)
                                st[(((gc*2 + ge)*ROWS + gr)*KS + gi)*8 +: 8] <= d_d[gc*BUF + (EL % 5)*8 +: 8];
                    end
                end
            end
        end
    endgenerate
    integer ch, e, r, i;
    always @(*) begin
        r_word = 0;
        for (ch = 0; ch < CH; ch = ch + 1)
            for (e = 0; e < 2; e = e + 1)
                for (r = 0; r < ROWS; r = r + 1)
                    for (i = 0; i < KS; i = i + 1)
                        if (ch == r_ch && e == r_ent && (r / NB) == r_q && i == r_i)
                            r_word[(r % NB)*8 +: 8] = st[(((ch*2 + e)*ROWS + r)*KS + i)*8 +: 8];
    end
endmodule


// ============================================================================
// dimm_slice — one slice: the verified v2 tile core (v2/sim/rtl/dimm_tile_top.v) on XS crossbars and KS columns,
// with one A and one B write port (blocked A memories, A_GRP crossbars each), the window engine copy led by the
// slice's crossbar 0, and the first log2 KS levels of each group's adder tree. Crossbar f = g + NG*i: column i,
// group g; window (g + NG*t, i) in wave t.
// ============================================================================
module dimm_slice #(
    parameter M = 64, N = 16, R = 64, C = 64, BUF = 40, P = 8,
    parameter XS = 16, NG = 2, KS = 8, T_W = 4, W_K = 8, A_GRP = 8, B_GRP = 8,
    parameter A_SLOTS = 1, B_SLOTS = 1, A_SIGNED = 1, B_SIGNED = 1,
    parameter SAW = 1, SBW = 1, AAW = 5, BAW = 4, LNABW = 1, LNBMW = 1, VWS = 12, TWW = 4, CW = 5
) (
    input  wire                  clk,
    input  wire                  reset,
    input  wire [7:0]            weight_in,
    input  wire                  weight_en,
    input  wire                  a_we,
    input  wire [LNABW-1:0]      a_wm,
    input  wire [AAW-1:0]        a_wa,
    input  wire [(((R < C) ? R : C) / N)*8-1:0] a_wd,
    input  wire                  b_we,
    input  wire [LNBMW-1:0]      b_wm,
    input  wire [BAW-1:0]        b_wa,
    input  wire [N*8-1:0]        b_wd,
    input  wire                  start,
    input  wire [SAW-1:0]        a_sel,
    input  wire [SBW-1:0]        b_sel,
    output wire                  idle,
    output wire [2+CW+TWW-1:0]   o_ctl,
    output wire [NG*5*VWS-1:0]   o_val
);
    localparam I    = (R < C) ? R : C;
    localparam NB   = I / N;
    localparam EPS  = BUF / 8;
    localparam EV   = I;                                     // M*N >= I: every window is full
    localparam PT   = KS * W_K;                              // windows of this slice per job
    localparam LCYC = (I * 8 + BUF - 1) / BUF;
    localparam LTW  = (T_W > 1) ? $clog2(T_W) : 1;
    localparam LXS  = $clog2(XS);
    localparam LN   = $clog2(N);
    localparam LNG  = $clog2(NG);
    localparam MW   = (NB > 1) ? $clog2(NB) : 1;
    localparam NW   = (N > 1) ? $clog2(N) : 1;
    localparam LG   = $clog2(A_GRP);
    localparam LGW  = (LG > 0) ? LG : 1;
    localparam NAB  = XS / A_GRP;
    localparam FR   = (A_GRP == 1) ? 1 : A_GRP + 1;
    localparam CTLW = 2 + CW + TWW;

    function sgn;
        input [7:0] b;
        sgn = b[7] ^ (&b[6:0]);
    endfunction

    // ---- run state (v2) ---------------------------------------------------------------------------------------
    // The memory read addresses come from this slice's registers only (Fmax, 2026-10-15): `start` crosses the chip
    // from the job logic, so it must not select an address in the start cycle. The slots load from the job register
    // (a_sel / b_sel = the top's ja_q / jb_q) while the slice is idle: the top decides only when every slice is idle,
    // and its job register holds the job through the start cycle. The wave count and the fill index are zero while
    // idle. Every address in the job's cycles is the same as before.
    wire          take_start = start && idle;
    reg [SAW-1:0] a_sel_q;
    reg [SBW-1:0] b_sel_q;
    always @(posedge clk) begin
        if (reset) begin a_sel_q <= 0; b_sel_q <= 0; end
        else if (idle) begin a_sel_q <= a_sel; b_sel_q <= b_sel; end
    end
    reg [LGW+1:0] fresh_c;
    always @(posedge clk)
        fresh_c <= reset ? 0 : (take_start ? FR : ((fresh_c != 0) ? fresh_c - 1'b1 : 0));

    // ---- window engine (leader = this slice's crossbar 0) -----------------------------------------------------
    wire              leader_ready, leader_full, leader_done;
    wire              feed_active, feed_strobe, start_feed, feed_busy, feed_more;
    wire [TWW-1:0]    feed_t, nst_next, drn_t, dt_next;
    wire [CW-1:0]     feed_c, drn_c;
    wire              drn_valid, last_win_done;
    dimm_tile_ctrl #(.T_W(T_W), .LCYC(LCYC), .TWW(TWW), .CW(CW)) u_ctrl (
        .clk(clk), .reset(reset), .start(take_start),
        .leader_ready(leader_ready && fresh_c == 0), .leader_full(leader_full), .leader_done(leader_done),
        .feed_active(feed_active), .feed_strobe(feed_strobe), .start_feed(start_feed), .feed_busy(feed_busy),
        .feed_more(feed_more),
        .feed_t(feed_t), .feed_c(feed_c), .nst_next(nst_next),
        .drn_valid(drn_valid), .drn_t(drn_t), .drn_c(drn_c), .dt_next(dt_next),
        .last_win_done(last_win_done), .idle(idle));

    reg           first_q;
    reg [TWW-1:0] ns_q;
    reg [CW+3:0]  eb_q;
    always @(posedge clk) begin
        first_q <= !reset && start_feed;
        eb_q    <= start_feed ? 10 : eb_q + 5;
        if (reset || idle)   ns_q <= 0;
        else if (start_feed) ns_q <= ns_q + 1'b1;
    end
    wire [TWW-1:0] pf_t = ns_q;
    wire [TWW-1:0] rw_t = ns_q - 1'b1;

    // ---- lane geometry (v2) -------------------------------------------------------------------------------------
    wire [EPS-1:0]    rl_valid, dl_valid;
    wire [EPS*MW-1:0] rl_m, dl_m;
    wire [EPS*NW-1:0] rl_n, dl_n;
    genvar gj, gr, gb;
    generate
        for (gj = 0; gj < EPS; gj = gj + 1) begin : g_lane
            wire [CW+3:0] er = eb_q + gj;
            wire [CW+3:0] ed = {drn_c, 2'b00} + drn_c + gj;
            assign rl_valid[gj] = er < EV;
            assign dl_valid[gj] = ed < EV;
            assign rl_m[gj*MW +: MW] = (NB > 1) ? (er >> LN) : {MW{1'b0}};
            assign dl_m[gj*MW +: MW] = (NB > 1) ? (ed >> LN) : {MW{1'b0}};
            assign rl_n[gj*NW +: NW] = (N > 1) ? er[NW-1:0] : {NW{1'b0}};
            assign dl_n[gj*NW +: NW] = (N > 1) ? ed[NW-1:0] : {NW{1'b0}};
        end
    endgenerate

    // ---- fill index (v2) ---------------------------------------------------------------------------------------
    reg  [LGW-1:0]       fi_q, fi_d;
    wire [LGW-1:0]       fi_rd = (A_GRP == 1) ? {LGW{1'b0}} : fi_q;
    always @(posedge clk) begin
        fi_q <= (idle && !take_start) ? {LGW{1'b0}} : fi_rd + 1'b1;   // 0 in the start cycle, then 1, 2, ...
        fi_d <= fi_rd;
    end

    // ---- B memories (B_GRP columns each, word {slot, index}) --------------------------------------------------
    localparam NBM   = KS / B_GRP;
    localparam LBG   = $clog2(B_GRP);
    localparam LBGW  = (LBG > 0) ? LBG : 1;
    wire [LBGW-1:0]    bi_rd = fi_rd[LBGW-1:0] & (B_GRP - 1);
    wire [LBGW-1:0]    bi_d  = fi_d[LBGW-1:0] & (B_GRP - 1);
    wire [NBM*N*8-1:0] bm_rd;
    genvar gm;
    generate
        for (gm = 0; gm < NBM; gm = gm + 1) begin : g_bm
            reg [N*8-1:0] mem [0:(B_SLOTS << LBGW) - 1];
            reg [N*8-1:0] rq;
            wire [SBW+LBGW-1:0] ra = ({{LBGW{1'b0}}, b_sel_q} << LBGW) | bi_rd;
            always @(posedge clk) begin
                if (b_we && (NBM == 1 || b_wm == gm)) mem[b_wa] <= b_wd;
                rq <= mem[ra];
            end
            assign bm_rd[gm*N*8 +: N*8] = rq;
        end
    endgenerate

    // ---- B words per column: feed lanes (bytes), signs (drain lanes) (v2) --------------------------------------
    wire [KS*EPS*8-1:0] b_f0, b_f1, b_fr;
    wire [KS*EPS-1:0]   b_sign;
    genvar gs;
    generate
        for (gs = 0; gs < KS; gs = gs + 1) begin : g_b
            localparam BM = gs >> LBG;
            localparam BI = gs & (B_GRP - 1);
            reg  [N*8-1:0] nxt_c;
            wire [N*8-1:0] nxt = (B_GRP == 1) ? bm_rd[BM*N*8 +: N*8] : nxt_c;
            always @(posedge clk)
                if (B_GRP > 1 && bi_d == BI) nxt_c <= bm_rd[BM*N*8 +: N*8];
            reg  [N*8-1:0] rb;
            for (gb = 0; gb < N; gb = gb + 1) begin : g_rb
                always @(posedge clk)
                    rb[gb*8 +: 8] <= feed_more ? rb[((gb + 5) % N)*8 +: 8] : nxt[((gb + 10) % N)*8 +: 8];
            end
            for (gj = 0; gj < EPS; gj = gj + 1) begin : g_l
                localparam N0 = gj & (N - 1), N1 = (gj + 5) & (N - 1);
                assign b_f0[(gs*EPS + gj)*8 +: 8] = nxt[N0*8 +: 8];
                assign b_f1[(gs*EPS + gj)*8 +: 8] = nxt[N1*8 +: 8];
                assign b_fr[(gs*EPS + gj)*8 +: 8] = rb[(gj % N)*8 +: 8];
            end
            wire [N-1:0] sd;
            for (gb = 0; gb < N; gb = gb + 1) begin : g_s
                assign sd[gb] = (B_SIGNED != 0) && sgn(nxt[gb*8 +: 8]);
            end
            for (gj = 0; gj < EPS; gj = gj + 1) begin : g_d
                assign b_sign[gs*EPS + gj] = sd[dl_n[gj*NW +: NW]];
            end
        end
    endgenerate

    // ---- A memories (blocked: crossbar f in memory f >> LG at index f mod A_GRP) -------------------------------
    wire [LTW+SAW+LGW-1:0] a_ra = ({{LGW{1'b0}}, a_sel_q, pf_t[LTW-1:0]} << LGW) | fi_rd;
    wire [NAB*NB*8-1:0] ab_rd;
    genvar gk;
    generate
        for (gk = 0; gk < NAB; gk = gk + 1) begin : g_a
            reg [NB*8-1:0] mem [0:(A_SLOTS << (LTW + LGW)) - 1];
            reg [NB*8-1:0] rq;
            always @(posedge clk) begin
                if (a_we && (NAB == 1 || a_wm == gk)) mem[a_wa] <= a_wd;
                rq <= mem[a_ra];
            end
            assign ab_rd[gk*NB*8 +: NB*8] = rq;
        end
    endgenerate

    // ---- crossbars: feed u = A + B (trunc8), dpe EXP, signed drain value (v2) -------------------------------------
    wire [XS-1:0]       x_ready, x_full, x_done;
    wire [XS*EPS*9-1:0] x_val;
    genvar gf;
    generate
        for (gf = 0; gf < XS; gf = gf + 1) begin : g_x
            localparam SLOT = gf >> LNG;
            localparam BS   = SLOT;
            localparam GB   = gf >> LG;
            localparam GI   = gf & (A_GRP - 1);
            reg  [NB*8-1:0] nxt_c, cur;
            wire [NB*8-1:0] nxt = (A_GRP == 1) ? ab_rd[GB*NB*8 +: NB*8] : nxt_c;
            always @(posedge clk) begin
                if (A_GRP > 1 && fi_d == GI) nxt_c <= ab_rd[GB*NB*8 +: NB*8];
                if (start_feed) cur <= nxt;
            end
            wire [TWW+LXS:0] g_feed = gf + ({{(LXS+1){1'b0}}, feed_t} << LXS);
            wire [TWW+LXS:0] g_drn  = gf + ({{(LXS+1){1'b0}}, drn_t} << LXS);
            wire act_f = g_feed < PT;
            wire act_d = g_drn < PT;
            wire [BUF-1:0] f0_w, f1_w, fr_w;
            reg  [BUF-1:0] f1_q, fn_q;
            for (gj = 0; gj < EPS; gj = gj + 1) begin : g_fl
                localparam V0 = gj < EV, V1 = gj + 5 < EV;
                localparam M0 = V0 ? (gj >> LN) : 0, M1 = V1 ? ((gj + 5) >> LN) : 0;
                assign f0_w[gj*8 +: 8] = V0 ? (nxt[M0*8 +: 8] + b_f0[(BS*EPS + gj)*8 +: 8]) : 8'd0;
                assign f1_w[gj*8 +: 8] = V1 ? (nxt[M1*8 +: 8] + b_f1[(BS*EPS + gj)*8 +: 8]) : 8'd0;
                assign fr_w[gj*8 +: 8] = rl_valid[gj]
                    ? (cur[rl_m[gj*MW +: MW]*8 +: 8] + b_fr[(BS*EPS + gj)*8 +: 8]) : 8'd0;
            end
            always @(posedge clk) begin
                f1_q <= f1_w;
                fn_q <= feed_more ? fr_w : f0_w;
            end
            wire [BUF-1:0] feed_bytes = first_q ? f1_q : fn_q;
            wire [BUF-1:0] dpe_in = weight_en ? {{(BUF-8){1'b0}}, weight_in} : feed_bytes;
            wire [BUF-1:0] dpe_out;
            dpe #(
                .KERNEL_WIDTH(R), .NUM_COLS(C), .DPE_BUF_WIDTH(BUF), .PRECISION(P),
                .ROWS_USED(I), .COLS_USED(I)
            ) u_dpe (
                .clk(clk), .reset(reset), .data_in(dpe_in), .nl_dpe_control(2'b10),   // EXP
                .shift_add_control(1'b0), .w_buf_en(feed_strobe && act_f), .shift_add_bypass(1'b0),
                .load_output_reg(1'b0), .load_input_reg(weight_en),
                .MSB_SA_Ready(x_ready[gf]), .data_out(dpe_out), .dpe_done(x_done[gf]),
                .reg_full(x_full[gf]));
            wire [4*NB-1:0] ring;
            for (gr = 0; gr < 4; gr = gr + 1) begin : g_r
                reg [NB-1:0] s;
                for (gb = 0; gb < NB; gb = gb + 1) begin : g_s
                    always @(posedge clk)
                        if (first_q && rw_t[1:0] == gr) s[gb] <= (A_SIGNED != 0) && sgn(cur[gb*8 +: 8]);
                end
                assign ring[gr*NB +: NB] = s;
            end
            reg  [NB-1:0] sd;
            always @(*)
                case (drn_t[1:0])
                    2'd0:    sd = ring[0 +: NB];
                    2'd1:    sd = ring[NB +: NB];
                    2'd2:    sd = ring[2*NB +: NB];
                    default: sd = ring[3*NB +: NB];
                endcase
            for (gj = 0; gj < EPS; gj = gj + 1) begin : g_dl
                wire       sa = sd[dl_m[gj*MW +: MW]];
                wire [8:0] e9 = {1'b0, dpe_out[gj*8 +: 8]};
                reg  [8:0] vq;
                always @(posedge clk)
                    vq <= (reset || !(drn_valid && act_d && dl_valid[gj])) ? 9'd0
                        : ((sa ^ b_sign[BS*EPS + gj]) ? (9'd0 - e9) : e9);
                assign x_val[(gf*EPS + gj)*9 +: 9] = vq;
            end
        end
    endgenerate
    assign leader_ready = x_ready[0];
    assign leader_full  = x_full[0];
    assign leader_done  = x_done[0];

    // ---- drain register + the slice's tree levels ---------------------------------------------------------------
    reg           d_valid, d_last;
    reg [CW-1:0]  d_c;
    reg [TWW-1:0] d_t;
    always @(posedge clk) begin
        d_valid <= !reset && drn_valid;
        d_last  <= !reset && last_win_done;
        d_c     <= drn_c;
        d_t     <= drn_t;
    end
    dimm_tree #(.NG(NG), .DC(KS), .IW(9), .OW(VWS), .CTLW(CTLW)) u_tree (
        .clk(clk), .reset(reset), .in_ctl({d_valid, d_last, d_c, d_t}), .in_val(x_val),
        .out_ctl(o_ctl), .out_val(o_val));
endmodule


// ============================================================================
// dimm_tile_ctrl — the window engine (v2 verbatim): take-now acceptance, feed overlapping the previous drain,
// depth-2 drain queue, driven by the leader crossbar's status.
// ============================================================================
module dimm_tile_ctrl #(
    parameter T_W  = 4,
    parameter LCYC = 13,
    parameter TWW  = 4,
    parameter CW   = 5
) (
    input  wire           clk,
    input  wire           reset,
    input  wire           start,
    input  wire           leader_ready,
    input  wire           leader_full,
    input  wire           leader_done,
    output wire           feed_active,
    output wire           feed_strobe,
    output wire           start_feed,
    output wire           feed_busy,
    output wire           feed_more,
    output wire [TWW-1:0] feed_t,
    output wire [CW-1:0]  feed_c,
    output wire [TWW-1:0] nst_next,
    output wire           drn_valid,
    output wire [TWW-1:0] drn_t,
    output wire [CW-1:0]  drn_c,
    output wire [TWW-1:0] dt_next,
    output wire           last_win_done,
    output wire           idle
);
    reg           run;
    reg [TWW-1:0] iss_t, nst, drained, ft, lat_t, dt, que_t;
    reg [CW-1:0]  feed_cnt, drn_cnt;
    reg           feeding, have_win, draining, que_valid;

    wire request  = run && (iss_t < T_W);
    wire win_busy = feeding || have_win;
    wire accept   = request && !win_busy;
    wire take_now = accept && leader_ready;
    wire [TWW-1:0] start_t = take_now ? iss_t : lat_t;
    assign start_feed  = take_now || (have_win && !feeding && leader_ready);
    assign feed_active = feeding || start_feed;
    assign feed_t      = start_feed ? start_t : ft;
    assign feed_c      = start_feed ? {CW{1'b0}} : feed_cnt;
    assign feed_strobe = feed_active && leader_ready;
    assign feed_busy   = feeding;
    assign feed_more   = feeding && (feed_cnt != LCYC - 1);
    wire feed_done_w   = start_feed ? (LCYC == 1) : (feeding && leader_ready && (feed_cnt == LCYC - 1));
    wire [TWW-1:0] feed_done_t = start_feed ? start_t : ft;
    wire drain_done_w  = draining && leader_done;

    assign drn_valid = draining && leader_full;
    assign drn_t     = dt;
    assign drn_c     = drn_cnt;
    assign last_win_done = drain_done_w && (drained == T_W - 1);
    assign idle      = !run;
    assign nst_next  = start ? {TWW{1'b0}} : (start_feed ? nst + 1'b1 : nst);
    assign dt_next   = drain_done_w ? (que_valid ? que_t : (feed_done_w ? feed_done_t : dt))
                     : ((feed_done_w && !draining) ? feed_done_t : dt);

    always @(posedge clk) begin
        if (reset) begin
            run <= 1'b0; iss_t <= 0; nst <= 0; drained <= 0; ft <= 0; lat_t <= 0;
            feed_cnt <= 0; feeding <= 1'b0; have_win <= 1'b0;
            dt <= 0; que_t <= 0; drn_cnt <= 0; draining <= 1'b0; que_valid <= 1'b0;
        end
        else begin
            nst <= nst_next;
            dt  <= dt_next;
            if (start) begin
                run <= 1'b1;
                iss_t <= 0;
                drained <= 0;
            end
            if (accept) iss_t <= iss_t + 1'b1;
            if (accept && !leader_ready) begin
                lat_t <= iss_t;
                have_win <= 1'b1;
            end
            if (start_feed) begin
                have_win <= 1'b0;
                if (LCYC == 1) feeding <= 1'b0;
                else begin feeding <= 1'b1; ft <= start_t; feed_cnt <= 1; end
            end
            else if (feeding && leader_ready && (feed_cnt != LCYC - 1)) feed_cnt <= feed_cnt + 1'b1;
            else if (feeding && leader_ready) feeding <= 1'b0;
            if (drain_done_w) begin
                drained <= drained + 1'b1;
                drn_cnt <= 0;
                if (que_valid) begin
                    que_t <= feed_done_t;
                    que_valid <= feed_done_w;
                end
                else if (!feed_done_w) draining <= 1'b0;
            end
            else if (feed_done_w) begin
                if (draining) begin que_t <= feed_done_t; que_valid <= 1'b1; end
                else begin draining <= 1'b1; drn_cnt <= 0; end
            end
            if (draining && leader_full && (drn_cnt != LCYC - 1)) drn_cnt <= drn_cnt + 1'b1;
            if (drain_done_w && (drained == T_W - 1)) run <= 1'b0;
        end
    end
endmodule


// ============================================================================
// dimm_tree — registered adder trees, one per group, with the control pipeline (v2 dimm_tile_out generalized).
// Input index g + d*NG (group g, term d = 0 .. DC-1), IW-bit signed values, 5 lanes each; TL = log2 DC registered
// levels; OW-bit outputs. DC = 1: no level (the input passes through).
// ============================================================================
module dimm_tree #(
    parameter NG = 2, DC = 2, IW = 9, OW = 10, CTLW = 11
) (
    input  wire                 clk,
    input  wire                 reset,
    input  wire [CTLW-1:0]      in_ctl,
    input  wire [DC*NG*5*IW-1:0] in_val,
    output wire [CTLW-1:0]      out_ctl,
    output wire [NG*5*OW-1:0]   out_val
);
    localparam TL = $clog2(DC);
    function integer lvl_off;
        input integer l;
        integer i;
        begin
            lvl_off = 0;
            for (i = 0; i < l; i = i + 1) lvl_off = lvl_off + NG * 5 * (DC >> i) * OW;
        end
    endfunction
    wire [lvl_off(TL + 1)-1:0] tree;
    wire [(TL+1)*CTLW-1:0]     ctl;
    genvar ga, gi, gj, gl;
    generate
        for (ga = 0; ga < NG; ga = ga + 1) begin : g_in_a
            for (gi = 0; gi < DC; gi = gi + 1) begin : g_in_i
                for (gj = 0; gj < 5; gj = gj + 1) begin : g_in_j
                    wire [IW-1:0] v = in_val[((ga + gi*NG)*5 + gj)*IW +: IW];
                    assign tree[((ga*DC + gi)*5 + gj)*OW +: OW] = {{(OW-IW){v[IW-1]}}, v};
                end
            end
        end
        assign ctl[0 +: CTLW] = in_ctl;
        for (gl = 0; gl < TL; gl = gl + 1) begin : g_lvl
            localparam OI = lvl_off(gl);
            localparam OO = lvl_off(gl + 1);
            localparam NO = DC >> (gl + 1);
            reg [CTLW-1:0] cq;
            for (ga = 0; ga < NG; ga = ga + 1) begin : g_a
                for (gi = 0; gi < NO; gi = gi + 1) begin : g_b
                    for (gj = 0; gj < 5; gj = gj + 1) begin : g_j
                        reg [OW-1:0] r;
                        always @(posedge clk)
                            r <= tree[OI + ((ga*2*NO + 2*gi)*5 + gj)*OW +: OW]
                               + tree[OI + ((ga*2*NO + 2*gi + 1)*5 + gj)*OW +: OW];
                        assign tree[OO + ((ga*NO + gi)*5 + gj)*OW +: OW] = r;
                    end
                end
            end
            always @(posedge clk) cq <= reset ? {CTLW{1'b0}} : ctl[gl*CTLW +: CTLW];
            assign ctl[(gl+1)*CTLW +: CTLW] = cq;
        end
    endgenerate
    assign out_ctl = ctl[TL*CTLW +: CTLW];
    assign out_val = tree[lvl_off(TL) +: NG*5*OW];
endmodule
