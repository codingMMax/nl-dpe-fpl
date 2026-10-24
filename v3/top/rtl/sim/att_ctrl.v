// ============================================================================
// att_ctrl.v — the head's tiling controller (v3, top/spec.md §5). Counters only: no crossbar, no memory. The same file
// in both twins (no `dpe`).
//   admission  copy c may start its next token burst (x_rdy[c]) when the head runs, the copy has no burst in progress,
//              its three GEMMs have MSB_SA_Ready, a token is left, and the staging of the QK A, QK B and PV B ports
//              holds the token (freed + CAP > the token's unit); a burst starts when x_v rises, LOADX planes
//   jobs       QK job j = (A slot j / NBLK, B slot j mod NBLK), PV job j = (A slot j mod 2 released, B slot j mod NBLK),
//              offered from the run start and taken in the first cycle with the DIMM's j_rdy
//   runs       softmax run r starts in the first cycle with sm_busy low; done one cycle after the last run's run_done;
//              busy falls one cycle after done; one start per reset
// ============================================================================
`timescale 1ns / 1ps

module att_ctrl #(
    parameter NSEQ   = 128,
    parameter S      = 64,
    parameter BKV    = 16,
    parameter I      = 64,
    parameter P_PROJ = 1,
    parameter LOADX  = 8,                               // plane words per token burst (P)
    // derived (port widths; do not override)
    parameter RB     = NSEQ / S,
    parameter NBLK   = NSEQ / BKV,
    parameter QSA    = (RB > 1) ? $clog2(RB) : 1,
    parameter QSB    = (NBLK > 1) ? $clog2(NBLK) : 1
) (
    input  wire              clk,
    input  wire              reset,
    input  wire              start,
    output wire              busy,
    output reg               done,
    input  wire [P_PROJ-1:0] x_v,
    output wire [P_PROJ-1:0] x_rdy,
    input  wire [P_PROJ-1:0] gq_rdy,
    input  wire [P_PROJ-1:0] gk_rdy,
    input  wire [P_PROJ-1:0] gv_rdy,
    input  wire              qa_cr,
    input  wire              qb_cr,
    input  wire              pb_cr,
    output wire              qj_v,
    output wire [QSA-1:0]    qj_a,
    output wire [QSB-1:0]    qj_b,
    output wire              qj_ra,
    input  wire              qj_rdy,
    output wire              pj_v,
    output wire              pj_a,
    output wire [QSB-1:0]    pj_b,
    output wire              pj_ra,
    input  wire              pj_rdy,
    output wire              sm_start,
    input  wire              sm_busy,
    input  wire              run_done
);
    localparam NB_A = I / BKV;                          // rows per QK A staging group
    localparam LNA  = $clog2(NB_A);
    localparam LBK  = $clog2(BKV);
    localparam TW   = $clog2(NSEQ + P_PROJ) + 1;        // token indices and pulse counts
    localparam LXW  = (LOADX > 1) ? $clog2(LOADX) : 1;
    localparam NJ   = RB * NBLK;
    localparam JW   = $clog2(NJ + 1);
    localparam RW   = $clog2(RB + 1);

    generate
        if ((NSEQ & (NSEQ - 1)) != 0 || (S & (S - 1)) != 0 || (BKV & (BKV - 1)) != 0 || (I & (I - 1)) != 0
            || (P_PROJ & (P_PROJ - 1)) != 0 || S > NSEQ || BKV > NSEQ || BKV > I || P_PROJ > NSEQ || LOADX < 1)
        begin : g_bad_param
            att_ctrl_parameter_outside_domain u_bad ();
        end
    endgenerate

    // ---- run control: start taken at t_s, the head runs from t_s + 1 until done ---------------------------------
    reg running, started;
    assign busy = running;
    always @(posedge clk) begin
        if (reset) begin running <= 1'b0; started <= 1'b0; end
        else if (start && !running && !started) begin running <= 1'b1; started <= 1'b1; end
        else if (done) running <= 1'b0;
    end

    // ---- staging credits (a pulse at t counts from t + 1) -------------------------------------------------------
    reg [TW-1:0] qa_n, qb_n, pb_n;
    always @(posedge clk) begin
        if (reset) begin qa_n <= 0; qb_n <= 0; pb_n <= 0; end
        else begin
            if (qa_cr) qa_n <= qa_n + 1'b1;
            if (qb_cr) qb_n <= qb_n + 1'b1;
            if (pb_cr) pb_n <= pb_n + 1'b1;
        end
    end

    // ---- admission per copy: token tok = c, c + P_PROJ, ...; a burst of LOADX planes starts when x_v rises ------
    genvar gc;
    generate
        for (gc = 0; gc < P_PROJ; gc = gc + 1) begin : g_cp
            reg  [TW-1:0]  tok;                         // the copy's next token
            reg  [LXW-1:0] wc;                          // planes of the burst in progress (0: none)
            wire           bstart = x_v[gc] && (wc == 0);
            always @(posedge clk) begin
                if (reset) begin tok <= gc; wc <= 0; end
                else begin
                    if (bstart) tok <= tok + P_PROJ;
                    if (x_v[gc]) wc <= (wc == LOADX - 1) ? {LXW{1'b0}} : wc + 1'b1;
                end
            end
            wire qa_ok = (qa_n + 2) > (tok >> LNA);
            wire qb_ok = (qb_n + 2) > (tok >> LBK);
            wire pb_ok = (pb_n + 4 * P_PROJ) > tok;     // PV B (E3): 4 rows per copy
            assign x_rdy[gc] = running && (wc == 0) && gq_rdy[gc] && gk_rdy[gc] && gv_rdy[gc] && (tok < NSEQ)
                               && qa_ok && qb_ok && pb_ok;
        end
    endgenerate

    // ---- job lists: offered from the run start, taken with j_rdy -------------------------------------------------
    reg [JW-1:0]  qj_n, pj_n;
    reg [QSA-1:0] qj_r;
    reg [QSB-1:0] qj_k, pj_k;
    reg           pj_s;
    assign qj_v  = running && (qj_n < NJ);
    assign qj_a  = qj_r;
    assign qj_b  = qj_k;
    assign qj_ra = 1'b0;
    assign pj_v  = running && (pj_n < NJ);
    assign pj_a  = pj_s;
    assign pj_b  = pj_k;
    assign pj_ra = 1'b1;
    always @(posedge clk) begin
        if (reset) begin qj_n <= 0; qj_r <= 0; qj_k <= 0; pj_n <= 0; pj_s <= 1'b0; pj_k <= 0; end
        else begin
            if (qj_v && qj_rdy) begin
                qj_n <= qj_n + 1'b1;
                if (qj_k == NBLK - 1) begin qj_k <= 0; qj_r <= qj_r + 1'b1; end
                else qj_k <= qj_k + 1'b1;
            end
            if (pj_v && pj_rdy) begin
                pj_n <= pj_n + 1'b1;
                pj_s <= ~pj_s;
                pj_k <= (pj_k == NBLK - 1) ? {QSB{1'b0}} : pj_k + 1'b1;
            end
        end
    end

    // ---- softmax runs: one start per run, each in the first cycle with sm_busy low --------------------------------
    reg [RW-1:0] sr_n, rd_n;
    assign sm_start = running && !sm_busy && (sr_n < RB);
    always @(posedge clk) begin
        if (reset) begin sr_n <= 0; rd_n <= 0; done <= 1'b0; end
        else begin
            if (sm_start) sr_n <= sr_n + 1'b1;
            if (run_done) rd_n <= rd_n + 1'b1;
            done <= run_done && (rd_n == RB - 1);
        end
    end
endmodule
