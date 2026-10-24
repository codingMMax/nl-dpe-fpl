// ============================================================================
// att_head.v — one attention head (v3, top/spec.md). Wiring only: 3 P_PROJ GEMM copies, the QK and PV tiled DIMMs,
// the online softmax, its accumulator and the controller att_ctrl, connected as top/spec.md §4. SIMULATION twin: the
// modules' simulation twins, their weight ports tied off (the testbench preloads every crossbar, PLAN R8). VTR twin
// top/rtl/vtr/att_head.v (the modules' VTR twins); tools/check_twins.py gates the pair.
// The bit-plane pass (2026-10-23, top/spec.md §10 item 6): every stream is 8 plane words per window.
//   E0  x_v / x_d (copy c: V lanes of ROWS_X bits, LOADX = P planes per token) -> GEMMs gq c, gk c, gv c; x_rdy from
//       att_ctrl
//   E1  gq c -> QK DIMM A channel c     E2  gk c -> QK DIMM B channel c     E3  gv c -> PV DIMM B channel c
//   E4  QK DIMM C -> softmax s          E5  softmax lp -> PV DIMM A        E6 / E7  softmax la / ll -> accumulator
//   E8  PV DIMM C -> accumulator p      E9  accumulator o -> o_v / o_d (the final O8 values, one plane window of
//                                           N_OC*I elements per epilogue wave; the O projection reads them)
// Domain (elaboration guards): every module's, plus NSEQ and P_PROJ powers of two, D <= C, ROWS_X >= D.
// ============================================================================
`timescale 1ns / 1ps

module att_head #(
    parameter NSEQ   = 128,
    parameter DM     = 128,
    parameter D      = 64,
    parameter BKV    = 16,
    parameter R      = 64,
    parameter C      = 64,
    parameter P      = 8,
    parameter N_EXP  = 2,
    parameter N_E    = 128,
    parameter N_EP   = 128,
    parameter N_OC   = 8,
    parameter P_PROJ = 1,
    parameter X_S    = 16,
    // derived (port widths; do not override)
    parameter V      = (DM + R - 1) / R,
    parameter ROWS_X = (V == 1) ? DM : R,               // GEMM rows in use (P29)
    parameter I      = (R < C) ? R : C
) (
    input  wire                       clk,
    input  wire                       reset,
    input  wire                       start,
    output wire                       busy,
    output wire                       done,
    output wire [P_PROJ-1:0]          x_rdy,
    input  wire [P_PROJ-1:0]          x_v,
    input  wire [P_PROJ*V*ROWS_X-1:0] x_d,
    output wire                       o_v,
    output wire [N_OC*I-1:0]          o_d
);
    localparam S     = (NSEQ < I) ? NSEQ : I;            // query rows per run
    localparam RB    = NSEQ / S;
    localparam NBLK  = NSEQ / BKV;
    localparam LOADX = P;                                // plane words per token burst
    localparam LN    = $clog2(NSEQ);
    localparam NGQ   = N_E / D;                          // QK DIMM output channels
    localparam NGP   = N_EP / BKV;                       // PV DIMM output channels
    localparam NWEL  = (S * BKV < I) ? S * BKV : I;
    localparam AROWS = NWEL / BKV;                       // rows per PV A window
    localparam VWP   = 9 + $clog2(BKV);                  // o_part width
    localparam LPV   = (VWP + 7) / 8;                    // o_part plane lanes
    localparam XW    = V * ROWS_X;                       // a copy's token plane word
    localparam QSA   = (RB > 1) ? $clog2(RB) : 1;
    localparam QSB   = (NBLK > 1) ? $clog2(NBLK) : 1;

    generate
        if ((NSEQ & (NSEQ - 1)) != 0 || (P_PROJ & (P_PROJ - 1)) != 0 || P_PROJ > NSEQ || D > C || ROWS_X < D)
        begin : g_bad_param
            att_head_parameter_outside_domain u_bad ();
        end
    endgenerate

    // ---- E0 -> the GEMM copies -> E1, E2, E3 --------------------------------------------------------------------
    wire [P_PROJ-1:0]   gq_rdy, gk_rdy, gv_rdy, gq_v, gk_v, gv_v;
    wire [P_PROJ*D-1:0] gq_d, gk_d, gv_d;
    genvar gc;
    generate
        for (gc = 0; gc < P_PROJ; gc = gc + 1) begin : g_cp
            wire dq, dk, dv;                            // dpe_done (unused); reg_full is a busy flag, out_valid the plane valid
            gemm_top #(.K(DM), .N(D), .R(R), .C(C), .P(P), .ROWS_USED(ROWS_X), .COLS_USED(D)) u_gq (
                .clk(clk), .reset(reset), .act_in(x_d[gc*XW +: XW]), .act_en(x_v[gc]),
                .weight_in(8'd0), .weight_en(1'b0),
                .MSB_SA_Ready(gq_rdy[gc]), .data_out(gq_d[gc*D +: D]), .dpe_done(dq), .reg_full(),
                .out_valid(gq_v[gc]));
            gemm_top #(.K(DM), .N(D), .R(R), .C(C), .P(P), .ROWS_USED(ROWS_X), .COLS_USED(D)) u_gk (
                .clk(clk), .reset(reset), .act_in(x_d[gc*XW +: XW]), .act_en(x_v[gc]),
                .weight_in(8'd0), .weight_en(1'b0),
                .MSB_SA_Ready(gk_rdy[gc]), .data_out(gk_d[gc*D +: D]), .dpe_done(dk), .reg_full(),
                .out_valid(gk_v[gc]));
            gemm_top #(.K(DM), .N(D), .R(R), .C(C), .P(P), .ROWS_USED(ROWS_X), .COLS_USED(D)) u_gv (
                .clk(clk), .reset(reset), .act_in(x_d[gc*XW +: XW]), .act_en(x_v[gc]),
                .weight_in(8'd0), .weight_en(1'b0),
                .MSB_SA_Ready(gv_rdy[gc]), .data_out(gv_d[gc*D +: D]), .dpe_done(dv), .reg_full(),
                .out_valid(gv_v[gc]));
        end
    endgenerate

    // ---- the QK DIMM: q8 rows (A), k8 rows as B^T, scores s8 (NARROW 5) -> E4 ------------------------------------
    wire           qa_cr, qb_cr, pb_cr, qj_v, qj_ra, qj_rdy, pj_v, pj_a, pj_ra, pj_rdy, qk_busy, pv_busy;
    wire [QSA-1:0] qj_a;
    wire [QSB-1:0] qj_b, pj_b;
    wire [NGQ-1:0]   s_v;
    wire [NGQ*I-1:0] s_d;
    wire             s_cr;
    dimm_tile_top #(.M(S), .N(BKV), .K(D), .R(R), .C(C), .P(P), .N_E(N_E), .X_S(X_S),
                    .A_DOM(0), .A_SIGNED(1), .A_CH(P_PROJ), .A_SLOTS(RB), .A_FRAME(0), .A_ROWS(1),
                    .B_DOM(0), .B_SIGNED(1), .B_CH(P_PROJ), .B_SLOTS(NBLK), .TRANS_B(1), .NARROW(5), .C_CAP(2)) u_qk (
        .clk(clk), .reset(reset),
        .weight_in(8'd0), .weight_en(1'b0),
        .a_v(gq_v), .a_d(gq_d), .a_cr(qa_cr), .b_v(gk_v), .b_d(gk_d), .b_cr(qb_cr),
        .j_v(qj_v), .j_a(qj_a), .j_b(qj_b), .j_ra(qj_ra), .j_rdy(qj_rdy),
        .c_v(s_v), .c_d(s_d), .c_cr(s_cr), .busy(qk_busy));

    // ---- the online softmax: E4 -> log P (E5), log alpha (E6), log l (E7) ---------------------------------------
    wire [N_EXP-1:0]   lp_v;
    wire [N_EXP*I-1:0] lp_d;
    wire               lp_cr, la_v, la_cr, ll_v, ll_cr, sm_start, sm_busy, sm_done;
    wire [2*S-1:0]     la_d;
    wire [S-1:0]       ll_d;
    softmax_online_top #(.S(S), .BKV(BKV), .N(NSEQ), .R(R), .C(C), .P(P), .N_EXP(N_EXP), .S_CH(NGQ),
                         .LP_CAP(2), .LA_CAP(2), .LL_CAP(1)) u_sm (
        .clk(clk), .reset(reset), .start(sm_start), .busy(sm_busy), .done(sm_done),
        .s_v(s_v), .s_d(s_d), .s_cr(s_cr), .lp_v(lp_v), .lp_d(lp_d), .lp_cr(lp_cr),
        .la_v(la_v), .la_d(la_d), .la_cr(la_cr), .ll_v(ll_v), .ll_d(ll_d), .ll_cr(ll_cr));

    // ---- the PV DIMM: log P (A, window frame, released), v8 rows (B), o_part -> E8 ------------------------------
    wire [NGP-1:0]       p_v;
    wire [NGP*LPV*I-1:0] p_d;
    wire                 p_cr;
    dimm_tile_top #(.M(S), .N(D), .K(BKV), .R(R), .C(C), .P(P), .N_E(N_EP), .X_S(X_S),
                    .A_DOM(1), .A_SIGNED(0), .A_CH(N_EXP), .A_SLOTS(2), .A_FRAME(1), .A_ROWS(AROWS),
                    .B_DOM(0), .B_SIGNED(1), .B_CH(P_PROJ), .B_SLOTS(NBLK), .TRANS_B(0), .NARROW(0), .C_CAP(2)) u_pv (
        .clk(clk), .reset(reset),
        .weight_in(8'd0), .weight_en(1'b0),
        .a_v(lp_v), .a_d(lp_d), .a_cr(lp_cr), .b_v(gv_v), .b_d(gv_d), .b_cr(pb_cr),
        .j_v(pj_v), .j_a(pj_a), .j_b(pj_b), .j_ra(pj_ra), .j_rdy(pj_rdy),
        .c_v(p_v), .c_d(p_d), .c_cr(p_cr), .busy(pv_busy));

    // ---- the accumulator: E8, E6, E7 -> the final O8 values (E9) ------------------------------------------------
    wire run_done;
    softmax_online_acc #(.S(S), .D(D), .R(R), .C(C), .P(P), .N_OC(N_OC), .NBLK(NBLK), .SO(LN), .LN(LN),
                         .P_CH(NGP), .VWP(VWP)) u_acc (
        .clk(clk), .reset(reset),
        .weight_in(8'd0), .weight_en(1'b0),
        .p_v(p_v), .p_d(p_d), .p_cr(p_cr), .la_v(la_v), .la_d(la_d), .la_cr(la_cr),
        .ll_v(ll_v), .ll_d(ll_d), .ll_cr(ll_cr), .o_v(o_v), .o_d(o_d), .run_done(run_done));

    // ---- the controller -----------------------------------------------------------------------------------------
    att_ctrl #(.NSEQ(NSEQ), .S(S), .BKV(BKV), .I(I), .P_PROJ(P_PROJ), .LOADX(LOADX)) u_ctrl (
        .clk(clk), .reset(reset), .start(start), .busy(busy), .done(done), .x_v(x_v), .x_rdy(x_rdy),
        .gq_rdy(gq_rdy), .gk_rdy(gk_rdy), .gv_rdy(gv_rdy), .qa_cr(qa_cr), .qb_cr(qb_cr), .pb_cr(pb_cr),
        .qj_v(qj_v), .qj_a(qj_a), .qj_b(qj_b), .qj_ra(qj_ra), .qj_rdy(qj_rdy),
        .pj_v(pj_v), .pj_a(pj_a), .pj_b(pj_b), .pj_ra(pj_ra), .pj_rdy(pj_rdy),
        .sm_start(sm_start), .sm_busy(sm_busy), .run_done(run_done));
endmodule
