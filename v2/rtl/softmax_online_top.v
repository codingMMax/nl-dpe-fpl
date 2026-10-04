// ============================================================================
// softmax_online_top.v — NL-DPE online (blocked) softmax, v2 clean-room.
//
// Operator (ACAM mapping, same ops as `softmax_ref`):
//   per key block b :  m_b[r] = max_j s ;  Sp_b[r] = sum_j ACAM_EXP(max(s-m_b,-128))
//   m[r] = max_b m_b[r]; factor_b[r] = ACAM_EXP(max(m_b[r]-m[r],-128))
//   L[r] = sum_b factor_b[r]*Sp_b[r]; lq = min(L>>log2 S,127); ls = ACAM_LOG(lq)
//   out  = clamp(s - m - ls)
// Only the deferred-alpha aggregate differs from the conventional softmax; the
// EXP / factor / LOG are all charged ACAM identity passes.
//
// `done` = results READY (after the LOG + clamp); the S^2 output stream is emitted
// for capture only and is NOT counted.
//
// Parametric: S, R, C, BUF, P, BKV, N_EXP, N_LOG (dedicated factor/LOG crossbars).
// Reuses the shared `softmax_wprog`/`softmax_exp_feed`/`softmax_sum_unit`/
// `softmax_log_unit` from `softmax_top.v` (compile together).
// ============================================================================

`timescale 1ns / 1ps


// ----------------------------------------------------------------------------
// online_blk_loader — BUF-wide block-major score stream -> block-contiguous
// store (per-block de-padding). Pulses blk_loaded[b] on the block's last word.
// ----------------------------------------------------------------------------
module online_blk_loader #(
    parameter S = 128,
    parameter B = 8,
    parameter BKV = 16,
    parameter BUF = 40,
    parameter AW = 16
) (
    input  wire clk,
    input  wire reset,
    input  wire [BUF-1:0] score_in,
    input  wire score_en,
    output wire wr_en,
    output wire [AW-1:0] wr_base,
    output wire [(BUF/8)-1:0] wr_lane,
    output wire [BUF-1:0] wr_data,
    output reg  [B-1:0] blk_loaded,
    output wire [AW-1:0] store_avail
);
    localparam EPS    = BUF / 8;
    localparam BLKEL  = S * BKV;
    localparam STREAM = (BLKEL + EPS - 1) / EPS;
    localparam LOAD_WORDS = B * STREAM;

    reg [AW-1:0] load_word;
    reg [(BUF/8)-1:0] lane_c;
    integer j, bb, lw_col, le;

    wire [AW-1:0] lw_blk = load_word / STREAM;
    wire [AW-1:0] lw_colw = load_word % STREAM;
    assign wr_en   = score_en && (load_word < LOAD_WORDS);
    assign wr_base = lw_blk * BLKEL + lw_colw * EPS;
    assign wr_data = score_in;
    always @(*) begin
        lw_col = load_word % STREAM;
        for (j = 0; j < EPS; j = j + 1) begin
            le = lw_col * EPS + j;
            lane_c[j] = (le < BLKEL);
        end
    end
    assign wr_lane = lane_c;
    // block-contiguous element count COMMITTED so far (words 0..load_word-1
    // are present in all_buf; the in-flight word commits next cycle)
    assign store_avail = (load_word / STREAM) * BLKEL
                         + (((load_word % STREAM) * EPS) < BLKEL
                            ? (load_word % STREAM) * EPS : BLKEL);

    always @(posedge clk) begin
        if (reset) begin load_word <= 0; blk_loaded <= 0; end
        else begin
            blk_loaded <= 0;
            if (score_en && (load_word < LOAD_WORDS)) begin
                for (bb = 0; bb < B; bb = bb + 1)
                    if (load_word == (bb+1)*STREAM - 1) blk_loaded[bb] <= 1'b1;
                load_word <= load_word + 1;
            end
        end
    end
endmodule


// ----------------------------------------------------------------------------
// online_max_unit — streaming fold of the block-major store into the per-block
// row maxima `blk_max[b*S+r]` (mirrors `softmax_max_unit`). Reads `W` bytes per
// cycle, gated on the store write position; emits one row max per GROUPS groups
// through a MAX_LAT pipeline.
// ----------------------------------------------------------------------------
module online_max_unit #(
    parameter S = 128,
    parameter BKV = 16,
    parameter B = 8,
    parameter W = 16,
    parameter MAX_LAT = 8,
    parameter AW = 16
) (
    input  wire clk, input wire reset, input wire start,
    input  wire [AW-1:0] store_avail,          // elements written so far
    output reg  [AW-1:0] rd_addr,
    input  wire [W*8-1:0] rd_data,
    output reg  we,
    output reg  [AW-1:0] waddr,
    output reg  signed [7:0] wdata,
    output reg  [31:0] rows_done,
    output wire [B-1:0] blk_ready,
    output reg  busy
);
    localparam GROUPS  = (BKV + W - 1) / W;
    localparam NROWS   = B * S;
    localparam NGROUPS = NROWS * GROUPS;

    integer group_cnt;
    reg signed [7:0] row_running;
    reg signed [7:0] pipe [0:MAX_LAT-1];
    reg [MAX_LAT-1:0] pipe_valid;
    integer i;
    reg signed [7:0] word_max;

    always @(*) begin
        word_max = rd_data[7:0];
        for (i = 1; i < W; i = i + 1)
            if ($signed(rd_data[i*8 +: 8]) > word_max)
                word_max = rd_data[i*8 +: 8];
    end
    wire last_group = (group_cnt % GROUPS) == GROUPS - 1;
    wire signed [7:0] row_max_value = (row_running > word_max)
                                      ? row_running : word_max;
    wire [31:0] rd_idx = group_cnt * W;
    wire can_read = busy && (group_cnt < NGROUPS)
                    && ((rd_idx + W) <= store_avail);

    always @(posedge clk) begin
        if (reset) begin
            rd_addr <= 0; we <= 1'b0; waddr <= 0; wdata <= 0;
            rows_done <= 0; group_cnt <= 0; row_running <= 8'h80;
            busy <= 1'b0; pipe_valid <= 0;
            for (i = 0; i < MAX_LAT; i = i + 1) pipe[i] <= 8'h80;
        end
        else begin
            we <= 1'b0;
            if (start && !busy && rows_done == 0) busy <= 1'b1;
            pipe_valid[0] <= 1'b0;
            for (i = MAX_LAT-1; i > 0; i = i - 1) begin
                pipe[i] <= pipe[i-1];
                pipe_valid[i] <= pipe_valid[i-1];
            end
            if (can_read) begin
                rd_addr <= (group_cnt + 1) * W;
                if (last_group) begin
                    pipe[0] <= row_max_value;
                    pipe_valid[0] <= 1'b1;
                    row_running <= 8'h80;
                end
                else if (word_max > row_running)
                    row_running <= word_max;
                group_cnt <= group_cnt + 1;
            end
            if (pipe_valid[MAX_LAT-1]) begin
                we <= 1'b1;
                waddr <= rows_done;
                wdata <= pipe[MAX_LAT-1];
                rows_done <= rows_done + 1;
            end
            if (busy && group_cnt >= NGROUPS && rows_done == NROWS)
                busy <= 1'b0;
        end
    end

    genvar gb;
    generate
        for (gb = 0; gb < B; gb = gb + 1)
            assign blk_ready[gb] = (rows_done >= (gb + 1) * S);
    endgenerate
endmodule


// ----------------------------------------------------------------------------
// online_combine — deferred-alpha merge: L = sum_b factor_b * Sp_b;
// lq = min(L>>log2 S,127). The factor comes from the dedicated factor feed.
// ----------------------------------------------------------------------------
module online_combine #(
    parameter S = 128,
    parameter B = 8,
    parameter NROWS = 1024,
    parameter LOG2S = 7
) (
    input  wire [NROWS*32-1:0] factor_in,   // flat [b*S+r]
    input  wire [NROWS*32-1:0] sp_in,       // flat [b*S+r]
    output reg  [S*32-1:0]     L_flat,      // sum_b factor*Sp
    output reg  [S*8-1:0]      lq_flat      // min(L>>log2 S,127)
);
    integer b4, r4;
    reg [63:0] acc;
    reg [31:0] lqv;
    always @(*) begin
        for (r4 = 0; r4 < S; r4 = r4 + 1) begin
            acc = 0;
            for (b4 = 0; b4 < B; b4 = b4 + 1)
                acc = acc + factor_in[(b4*S+r4)*32 +: 32]
                          * sp_in[(b4*S+r4)*32 +: 32];
            L_flat[r4*32 +: 32] = acc[31:0];
            lqv = ((acc >> LOG2S) > 127) ? 32'd127 : acc >> LOG2S;
            lq_flat[r4*8 +: 8] = lqv[7:0];
        end
    end
endmodule


// ----------------------------------------------------------------------------
// online_out_unit — final softmax result, purely combinational:
//   data_out = clamp(score[blk*S*BKV + r*BKV + (j mod BKV)] - m[r] - ls[r])
// (out_addr = row-major r*S + j). No drain/serialize stage.
// ----------------------------------------------------------------------------
module online_out_unit #(
    parameter S = 128,
    parameter BKV = 16,
    parameter AW = 20
) (
    input  wire [AW-1:0] out_addr,
    output wire [AW-1:0] score_addr,
    input  wire [7:0] score_rdata,
    output wire [AW-1:0] rmax_addr,
    input  wire [7:0] rmax_rdata,
    input  wire [7:0] log_rdata,
    output reg  [7:0] data_out
);
    localparam LOG2BKV = $clog2(BKV);

    wire [AW-1:0] r   = out_addr / S;
    wire [AW-1:0] j   = out_addr % S;
    wire [AW-1:0] blk = j >> LOG2BKV;
    wire [AW-1:0] cc  = j & (BKV - 1);

    assign score_addr = blk * (S * BKV) + r * BKV + cc;
    assign rmax_addr  = r;

    reg signed [15:0] acc;
    always @(*) begin
        acc = $signed(score_rdata) - $signed(rmax_rdata) - $signed(log_rdata);
        if (acc > 16'sd127) data_out = 8'd127;
        else if (acc < -16'sd128) data_out = 8'd128;
        else data_out = acc[7:0];
    end
endmodule


// ----------------------------------------------------------------------------
// softmax_online_top — wiring: loader, block/global max, EXP feeds, partial
// banks, dedicated factor + LOG passes, deferred-alpha combine, drain.
// ----------------------------------------------------------------------------
module softmax_online_top #(
    parameter S = 128,
    parameter R = 256,
    parameter C = 256,
    parameter BUF = 40,
    parameter P = 8,
    parameter BKV = 16,
    parameter N_EXP = 1,
    parameter N_LOG = 1,
    parameter N_FAC = 1
) (
    input  wire clk,
    input  wire reset,
    input  wire prog_start,
    output wire prog_ready,
    input  wire [BUF-1:0] score_in,
    input  wire score_en,
    input  wire start,
    output wire busy,
    input  wire [$clog2(S*S):0] out_addr,
    output wire [7:0] data_out,
    output wire done
);
    localparam EPS       = BUF / 8;
    localparam I         = (R < C) ? R : C;
    localparam B         = S / BKV;
    localparam LOG2S     = $clog2(S);
    localparam SQ        = S * S;
    localparam BLKEL     = S * BKV;
    localparam STREAM    = (BLKEL + EPS - 1) / EPS;
    localparam PEXP      = (SQ + I - 1) / I;   // global windows over the flat S^2
    localparam CLB_TREE  = LOG2S + 1;
    localparam CLAMP_LAT = 2;
    localparam COMBINE_LAT = $clog2(B) + 1;
    localparam AWX       = $clog2(SQ) + 1;
    localparam NROWS     = B * S;
    localparam NFEL      = B * S;          // factor table size

    integer c, j, k, bb, issue_i, g, n;
    integer dp, dfp, b3, r3, cc2;
    integer pf, pf2, pfbase, pfgend;

    // ---------------- buffers + probes -------------------------------------
    reg [7:0]  all_buf   [0:SQ-1];
    reg [7:0]  blk_max_q [0:NROWS-1];
    reg [31:0] sp_q      [0:NROWS-1];
    reg [31:0] factor_q  [0:NROWS-1];
    reg [31:0] L_q       [0:S-1];
    reg [7:0]  lq_q      [0:S-1];
    reg [7:0]  ls_q      [0:S-1];
    reg [7:0]  m_q       [0:S-1];

    // ---------------- loader ------------------------------------------------
    wire        ld_wr_en;
    wire [AWX-1:0] ld_wr_base;
    wire [EPS-1:0] ld_wr_lane;
    wire [BUF-1:0] ld_wr_data;
    wire [B-1:0]   blk_loaded;
    wire [AWX-1:0] store_avail;
    online_blk_loader #(.S(S), .B(B), .BKV(BKV), .BUF(BUF), .AW(AWX)) u_loader (
        .clk(clk), .reset(reset), .score_in(score_in), .score_en(score_en),
        .wr_en(ld_wr_en), .wr_base(ld_wr_base), .wr_lane(ld_wr_lane),
        .wr_data(ld_wr_data), .blk_loaded(blk_loaded),
        .store_avail(store_avail));

    always @(posedge clk)
        if (ld_wr_en)
            for (j = 0; j < EPS; j = j + 1)
                if (ld_wr_lane[j])
                    all_buf[ld_wr_base + j] <= ld_wr_data[j*8 +: 8];

    // ---------------- block max unit (streaming fold) ----------------------
    reg started;
    always @(posedge clk) begin
        if (reset) started <= 0;
        else if (start) started <= 1;
    end

    wire [AWX-1:0] max_rd_addr;
    reg  [BKV*8-1:0] max_rd_data;
    integer mi;
    always @(*) begin
        for (mi = 0; mi < BKV; mi = mi + 1)
            if (max_rd_addr + mi < SQ)
                max_rd_data[mi*8 +: 8] = all_buf[max_rd_addr + mi];
            else
                max_rd_data[mi*8 +: 8] = 8'h80;
    end
    wire max_we;
    wire [AWX-1:0] max_waddr;
    wire [7:0]  max_wdata;
    wire [31:0] rows_done;
    wire [B-1:0] blk_ready;
    wire max_busy;
    online_max_unit #(.S(S), .BKV(BKV), .B(B), .W(BKV),
                      .MAX_LAT(CLB_TREE), .AW(AWX)) u_max (
        .clk(clk), .reset(reset), .start(start), .store_avail(store_avail),
        .rd_addr(max_rd_addr), .rd_data(max_rd_data),
        .we(max_we), .waddr(max_waddr), .wdata(max_wdata),
        .rows_done(rows_done), .blk_ready(blk_ready), .busy(max_busy));
    always @(posedge clk)
        if (max_we) blk_max_q[max_waddr] <= max_wdata;

    wire all_blk_ready = &blk_ready;

    // global row max m[r] = max_b blk_max[b*S+r] (once all blocks loaded);
    // `m_ready` gates the factor pass one cycle after m_q commits.
    integer b5, r5;
    reg [7:0] mcomb;
    reg m_ready;
    always @(posedge clk) begin
        if (reset) m_ready <= 1'b0;
        else if (all_blk_ready) begin
            for (r5 = 0; r5 < S; r5 = r5 + 1) begin
                mcomb = blk_max_q[r5];
                for (b5 = 1; b5 < B; b5 = b5 + 1)
                    if ($signed(blk_max_q[b5*S+r5]) > $signed(mcomb))
                        mcomb = blk_max_q[b5*S+r5];
                m_q[r5] <= mcomb;
            end
            m_ready <= 1'b1;
        end
    end

    // ---------------- weight programming -----------------------------------
    wire [7:0] w_data; wire w_en; wire [1:0] mode_exp, mode_log;
    softmax_wprog #(.R(R), .C(C)) u_wprog (
        .clk(clk), .reset(reset), .prog_start(prog_start),
        .prog_ready(prog_ready), .w_data(w_data), .w_en(w_en),
        .mode_exp(mode_exp), .mode_log(mode_log));

    // ---------------- EXP feeds --------------------------------------------
    wire [EPS*AWX-1:0] exp_score_addrs [0:N_EXP-1];
    reg  [BUF-1:0]     exp_score_rdata [0:N_EXP-1];
    wire [EPS*AWX-1:0] exp_rmax_addrs  [0:N_EXP-1];
    reg  [BUF-1:0]     exp_rmax_rdata  [0:N_EXP-1];
    reg  [N_EXP-1:0]   exp_win_start;
    wire [N_EXP-1:0]   exp_win_busy, exp_win_done;
    reg  [N_EXP*AWX-1:0] exp_win_base, exp_win_span;
    wire [N_EXP-1:0]   exp_dr_valid;
    wire [N_EXP*AWX-1:0] exp_dr_base;
    wire [N_EXP*EPS-1:0] exp_dr_mask;
    wire [N_EXP*BUF-1:0] exp_dr_data;
    wire [N_EXP-1:0]   exp_in_valid;

    genvar gc, gi;
    generate
        for (gc = 0; gc < N_EXP; gc = gc + 1) begin : gen_exp
            softmax_exp_feed #(.S(S), .ROW_STRIDE(BKV), .NELEM(SQ),
                               .R(R), .C(C), .BUF(BUF), .P(P), .AW(AWX)) u_feed (
                .clk(clk), .reset(reset),
                .score_addrs(exp_score_addrs[gc]),
                .score_rdata(exp_score_rdata[gc]),
                .rmax_addrs(exp_rmax_addrs[gc]),
                .rmax_rdata(exp_rmax_rdata[gc]),
                .dpe_load_input_reg(w_en), .dpe_weight_data(w_data),
                .nl_dpe_control(mode_exp),
                .win_start(exp_win_start[gc]),
                .win_base(exp_win_base[gc*AWX +: AWX]),
                .win_span(exp_win_span[gc*AWX +: AWX]),
                .win_busy(exp_win_busy[gc]), .win_done(exp_win_done[gc]),
                .in_wr_valid(exp_in_valid[gc]),
                .in_wr_base(), .in_wr_mask(), .in_wr_data(),
                .dr_valid(exp_dr_valid[gc]),
                .dr_base(exp_dr_base[gc*AWX +: AWX]),
                .dr_mask(exp_dr_mask[gc*EPS +: EPS]),
                .dr_data(exp_dr_data[gc*BUF +: BUF]));
        end
    endgenerate

    always @(*) begin
        for (c = 0; c < N_EXP; c = c + 1)
            for (j = 0; j < EPS; j = j + 1) begin
                if (exp_score_addrs[c][j*AWX +: AWX] < SQ)
                    exp_score_rdata[c][j*8 +: 8] =
                        all_buf[exp_score_addrs[c][j*AWX +: AWX]];
                else exp_score_rdata[c][j*8 +: 8] = 8'h80;
                if (exp_rmax_addrs[c][j*AWX +: AWX] < NROWS)
                    exp_rmax_rdata[c][j*8 +: 8] =
                        blk_max_q[exp_rmax_addrs[c][j*AWX +: AWX]];
                else exp_rmax_rdata[c][j*8 +: 8] = 8'h80;
            end
    end

    wire sp_we; wire [AWX-1:0] sp_waddr; wire [31:0] sp_wdata;
    // per-crossbar direct-commit bus (sp written at its completion cycle)
    wire [N_EXP-1:0]     dc_we_s;
    wire [N_EXP*AWX-1:0] dc_waddr_s;
    wire [N_EXP*32-1:0]  dc_wdata_s;
    softmax_sum_unit #(.S(S), .RS(BKV), .NROWS(NROWS), .SPAN(BKV), .OUT_LQ(0),
                       .N_EXP(N_EXP), .PORTS(BUF), .SUM_LAT(1), .AW(AWX),
                       .DIRECT_COMMIT(1)) u_sum (
        .clk(clk), .reset(reset),
        .dr_valid(exp_dr_valid), .dr_base(exp_dr_base),
        .dr_mask(exp_dr_mask), .dr_data(exp_dr_data),
        .sp_we(sp_we), .sp_waddr(sp_waddr), .sp_wdata(sp_wdata),
        .lq_we(), .lq_waddr(), .lq_wdata(), .lq_done_count(),
        .dc_we(dc_we_s), .dc_waddr(dc_waddr_s), .dc_wdata(dc_wdata_s));
    genvar dw;
    generate
        for (dw = 0; dw < N_EXP; dw = dw + 1)
            always @(posedge clk)
                if (dc_we_s[dw])
                    sp_q[dc_waddr_s[dw*AWX +: AWX]] <= dc_wdata_s[dw*32 +: 32];
    endgenerate

    // ---------------- EXP window issue -------------------------------------
    integer exp_iss [0:N_EXP-1];
    reg exp_pend [0:N_EXP-1];
    reg [N_EXP-1:0] exp_need_ok;
    integer ne, g2, gbase2, gend2, blo2, bhi2, bb2;
    always @(*) begin
        exp_need_ok = 0;
        for (ne = 0; ne < N_EXP; ne = ne + 1) begin
            g2 = exp_iss[ne]*N_EXP + ne;
            gbase2 = g2 * I;
            gend2 = (gbase2 + I < SQ) ? (gbase2 + I) : SQ;
            blo2 = gbase2 / BLKEL;
            bhi2 = (gend2 > 0) ? (gend2 - 1) / BLKEL : 0;
            exp_need_ok[ne] = (g2 < PEXP);
            for (bb2 = 0; bb2 < B; bb2 = bb2 + 1)
                if ((bb2 >= blo2) && (bb2 <= bhi2) && !blk_ready[bb2])
                    exp_need_ok[ne] = 1'b0;
        end
    end

    always @(*) begin
        exp_win_start = 0; exp_win_base = 0; exp_win_span = 0;
        for (issue_i = 0; issue_i < N_EXP; issue_i = issue_i + 1) begin
            g = exp_iss[issue_i]*N_EXP + issue_i;
            exp_win_start[issue_i] = exp_pend[issue_i] ||
                (started && exp_need_ok[issue_i] && !exp_win_busy[issue_i]);
            exp_win_base[issue_i*AWX +: AWX] = g * I;
            exp_win_span[issue_i*AWX +: AWX] = ((g*I + I - 1) < SQ)
                ? (g*I + I - 1) : (SQ - 1);
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            for (issue_i = 0; issue_i < N_EXP; issue_i = issue_i + 1) begin
                exp_iss[issue_i] <= 0; exp_pend[issue_i] <= 0;
            end
        end
        else begin
            for (issue_i = 0; issue_i < N_EXP; issue_i = issue_i + 1) begin
                if (exp_pend[issue_i]) begin
                    if (exp_win_busy[issue_i]) begin
                        exp_pend[issue_i] <= 0;
                        exp_iss[issue_i] <= exp_iss[issue_i] + 1;
                    end
                end
                else if (started && exp_need_ok[issue_i] && !exp_win_busy[issue_i])
                    exp_pend[issue_i] <= 1;
            end
        end
    end

    // ---------------- dedicated factor pass (ACAM_EXP(m_b - m)) -------------
    // N_FAC parallel conversion crossbars (round-robin windows, same rule as
    // the EXP bank); each feed reads blk_max_q (score) and m_q (rmax).
    wire [EPS*AWX-1:0] fac_score_addrs [0:N_FAC-1];
    reg  [BUF-1:0]     fac_score_rdata [0:N_FAC-1];
    wire [EPS*AWX-1:0] fac_rmax_addrs  [0:N_FAC-1];
    reg  [BUF-1:0]     fac_rmax_rdata  [0:N_FAC-1];
    reg  [N_FAC-1:0]   fac_win_start;
    wire [N_FAC-1:0]   fac_win_busy, fac_win_done;
    wire [N_FAC-1:0]   fac_dr_valid;
    wire [N_FAC*AWX-1:0] fac_dr_base;
    wire [N_FAC*EPS-1:0] fac_dr_mask;
    wire [N_FAC*BUF-1:0] fac_dr_data;
    integer            fac_iss [0:N_FAC-1];
    reg                fac_pend [0:N_FAC-1];
    reg [N_FAC-1:0]    fac_need_ok;
    reg [N_FAC*AWX-1:0] fac_win_base, fac_win_span;

    genvar gf;
    generate
        for (gf = 0; gf < N_FAC; gf = gf + 1) begin : gen_fac
            softmax_exp_feed #(.S(S), .ROW_STRIDE(S), .NELEM(NFEL), .RMOD(1),
                               .R(R), .C(C), .BUF(BUF), .P(P), .AW(AWX)) u_fac (
                .clk(clk), .reset(reset),
                .score_addrs(fac_score_addrs[gf]),
                .score_rdata(fac_score_rdata[gf]),
                .rmax_addrs(fac_rmax_addrs[gf]),
                .rmax_rdata(fac_rmax_rdata[gf]),
                .dpe_load_input_reg(w_en), .dpe_weight_data(w_data),
                .nl_dpe_control(mode_exp),
                .win_start(fac_win_start[gf]),
                .win_base(fac_win_base[gf*AWX +: AWX]),
                .win_span(fac_win_span[gf*AWX +: AWX]),
                .win_busy(fac_win_busy[gf]), .win_done(fac_win_done[gf]),
                .in_wr_valid(), .in_wr_base(), .in_wr_mask(), .in_wr_data(),
                .dr_valid(fac_dr_valid[gf]),
                .dr_base(fac_dr_base[gf*AWX +: AWX]),
                .dr_mask(fac_dr_mask[gf*EPS +: EPS]),
                .dr_data(fac_dr_data[gf*BUF +: BUF]));
        end
    endgenerate

    always @(*) begin
        for (c = 0; c < N_FAC; c = c + 1)
            for (j = 0; j < EPS; j = j + 1) begin
                if (fac_score_addrs[c][j*AWX +: AWX] < NFEL)
                    fac_score_rdata[c][j*8 +: 8] =
                        blk_max_q[fac_score_addrs[c][j*AWX +: AWX]];
                else fac_score_rdata[c][j*8 +: 8] = 8'h80;
                if (fac_rmax_addrs[c][j*AWX +: AWX] < S)
                    fac_rmax_rdata[c][j*8 +: 8] =
                        m_q[fac_rmax_addrs[c][j*AWX +: AWX]];
                else fac_rmax_rdata[c][j*8 +: 8] = 8'h80;
            end
    end

    // factor pass = PFAC windows of I elements each, dealt round-robin over
    // the N_FAC factor crossbars (window j -> crossbar j mod N_FAC, mirrors
    // the EXP issue); `win_base` advances so the whole B*S table is covered.
    always @(*) begin
        fac_need_ok = 0;
        for (c = 0; c < N_FAC; c = c + 1)
            fac_need_ok[c] = ((fac_iss[c]*N_FAC + c) < PFAC);
    end
    always @(*) begin
        fac_win_start = 0; fac_win_base = 0; fac_win_span = 0;
        for (c = 0; c < N_FAC; c = c + 1) begin
            j = fac_iss[c]*N_FAC + c;
            fac_win_start[c] = fac_pend[c] ||
                (m_ready && fac_need_ok[c] && !fac_win_busy[c]);
            fac_win_base[c*AWX +: AWX] = j * I;
            fac_win_span[c*AWX +: AWX] = ((j*I + I - 1) < NFEL)
                ? (j*I + I - 1) : (NFEL - 1);
        end
    end
    always @(posedge clk) begin
        if (reset) begin
            for (c = 0; c < N_FAC; c = c + 1) begin
                fac_iss[c] <= 0; fac_pend[c] <= 0;
            end
        end
        else begin
            for (c = 0; c < N_FAC; c = c + 1) begin
                if (fac_pend[c]) begin
                    if (fac_win_busy[c]) begin
                        fac_pend[c] <= 0;
                        fac_iss[c] <= fac_iss[c] + 1;
                    end
                end
                else if (m_ready && fac_need_ok[c] && !fac_win_busy[c])
                    fac_pend[c] <= 1;
            end
        end
    end
    always @(posedge clk)  // factor table writes from the factor drains
        for (c = 0; c < N_FAC; c = c + 1)
            if (fac_dr_valid[c])
                for (j = 0; j < EPS; j = j + 1)
                    if (fac_dr_mask[c*EPS + j] &&
                        (fac_dr_base[c*AWX +: AWX] + j < NROWS))
                        factor_q[fac_dr_base[c*AWX +: AWX] + j] <=
                            fac_dr_data[c*BUF + j*8 +: 8];

    // ---------------- EXP done counter -------------------------------------
    reg [AWX-1:0] exp_done_cnt;
    reg all_exp_done;
    always @(*) begin
        dp = 0;
        for (k = 0; k < N_EXP; k = k + 1) dp = dp + exp_win_done[k];
    end
    always @(posedge clk) begin
        if (reset) begin exp_done_cnt <= 0; all_exp_done <= 0; end
        else begin
            if (exp_done_cnt + dp > PEXP) exp_done_cnt <= PEXP;
            else exp_done_cnt <= exp_done_cnt + dp;
            if ((exp_done_cnt + dp >= PEXP) && !all_exp_done) all_exp_done <= 1;
        end
    end

    // ---------------- factor-done counter (all PFAC windows) ---------------
    localparam PFAC = (NFEL + I - 1) / I;
    reg [AWX-1:0] fac_done_cnt;
    reg all_fac_done;
    always @(*) begin
        dfp = 0;
        for (k = 0; k < N_FAC; k = k + 1) dfp = dfp + fac_win_done[k];
    end
    always @(posedge clk) begin
        if (reset) begin fac_done_cnt <= 0; all_fac_done <= 1'b0; end
        else begin
            if (fac_done_cnt + dfp > PFAC) fac_done_cnt <= PFAC;
            else fac_done_cnt <= fac_done_cnt + dfp;
            if ((fac_done_cnt + dfp >= PFAC) && !all_fac_done) all_fac_done <= 1;
        end
    end

    // ---------------- committed-write handshake ----------------------------
    // sp_q / factor_q writes commit the cycle after the last drain valid; do
    // not fire the combine (or start the LOG) until every source is committed.
    // fac_last: OR over the parallel factor feeds of the drain word that
    // covers the last table index (NROWS-1).
    reg  fac_last;
    wire sp_last  = sp_we && (sp_waddr == NROWS - 1);
    always @(*) begin
        fac_last = 0;
        for (j = 0; j < N_FAC; j = j + 1)
            if (fac_dr_valid[j] &&
                ((fac_dr_base[j*AWX +: AWX] + EPS - 1) >= NROWS - 1))
                fac_last = 1'b1;
    end
    reg  sp_last_q, fac_last_q;
    always @(posedge clk) begin
        if (reset) begin sp_last_q <= 0; fac_last_q <= 0; end
        else begin
            if (sp_last)  sp_last_q  <= 1'b1;
            if (fac_last) fac_last_q <= 1'b1;
        end
    end
    wire comb_ready = all_exp_done && all_fac_done;
    reg  comb_fired;
    wire combine_fire = comb_ready && !comb_fired;
    always @(posedge clk) begin
        if (reset) comb_fired <= 1'b0;
        else if (combine_fire) comb_fired <= 1'b1;
    end
    reg [COMBINE_LAT-1:0] comb_pipe;
    always @(posedge clk) begin
        if (reset) comb_pipe <= 0;
        else begin
            comb_pipe[0] <= combine_fire;
            for (j = 1; j < COMBINE_LAT; j = j + 1) comb_pipe[j] <= comb_pipe[j-1];
        end
    end
    wire combine_done = comb_pipe[COMBINE_LAT-1];

    wire [NROWS*32-1:0] fac_flat, sp_flat;
    wire [S*32-1:0] cm_L;
    wire [S*8-1:0]  cm_lq;
    genvar gb;
    generate
        for (gb = 0; gb < NROWS; gb = gb + 1) begin : gen_flat
            assign fac_flat[gb*32 +: 32] = factor_q[gb];
            assign sp_flat[gb*32 +: 32]  = sp_q[gb];
        end
    endgenerate
    online_combine #(.S(S), .B(B), .NROWS(NROWS), .LOG2S(LOG2S)) u_comb (
        .factor_in(fac_flat), .sp_in(sp_flat),
        .L_flat(cm_L), .lq_flat(cm_lq));
    always @(posedge clk)
        if (combine_done)
            for (k = 0; k < S; k = k + 1) begin
                L_q[k]  <= cm_L[k*32 +: 32];
                lq_q[k] <= cm_lq[k*8 +: 8];
            end

    // ---------------- dedicated LOG pass (ACAM_LOG(lq)) --------------------
    wire [EPS*AWX-1:0] log_addrs;
    reg  [BUF-1:0]     log_rdata;
    reg                log_win_start;
    wire               log_win_busy, log_win_done;
    wire               log_dr_valid;
    wire [AWX-1:0]     log_dr_base;
    wire [EPS-1:0]     log_dr_mask;
    wire [BUF-1:0]     log_dr_data;

    // lq_q is written at `combine_done` (its posedge, so readable the next
    // cycle); start the LOG that cycle (single pulse; combine is one-shot).
    always @(posedge clk) begin
        if (reset) log_win_start <= 1'b0;
        else log_win_start <= combine_done;
    end

    softmax_log_unit #(.S(S), .R(R), .C(C), .BUF(BUF), .P(P), .AW(AWX)) u_log (
        .clk(clk), .reset(reset),
        .lq_addrs(log_addrs), .lq_rdata(log_rdata),
        .dpe_load_input_reg(w_en), .dpe_weight_data(w_data),
        .nl_dpe_control(mode_log),
        .win_start(log_win_start),
        .win_base({AWX{1'b0}}),
        .win_busy(log_win_busy), .win_done(log_win_done),
        .dr_valid(log_dr_valid), .dr_base(log_dr_base),
        .dr_mask(log_dr_mask), .dr_data(log_dr_data));

    always @(*) begin
        log_rdata = 0;
        for (j = 0; j < EPS; j = j + 1)
            if (log_addrs[j*AWX +: AWX] < S)
                log_rdata[j*8 +: 8] = lq_q[log_addrs[j*AWX +: AWX]];
            else log_rdata[j*8 +: 8] = 8'h80;
    end
    always @(posedge clk)
        if (log_dr_valid)
            for (j = 0; j < EPS; j = j + 1)
                if (log_dr_mask[j] && (log_dr_base + j < S))
                    ls_q[log_dr_base + j] <= log_dr_data[j*8 +: 8];

    // ---------------- results READY + output stream ------------------------
    reg [CLAMP_LAT-1:0] done_pipe;
    always @(posedge clk) begin
        if (reset) done_pipe <= 0;
        else begin
            done_pipe[0] <= log_win_done;
            for (k = 1; k < CLAMP_LAT; k = k + 1) done_pipe[k] <= done_pipe[k-1];
        end
    end
    assign done = done_pipe[CLAMP_LAT-1];




    wire [AWX-1:0] out_score_addr, out_rmax_addr;
    wire [7:0] out_score_rdata, out_rmax_rdata, out_log_rdata;
    assign out_score_rdata = all_buf[(out_score_addr < SQ) ? out_score_addr : 0];
    assign out_rmax_rdata  = m_q[(out_rmax_addr < S) ? out_rmax_addr : 0];
    assign out_log_rdata   = ls_q[(out_rmax_addr < S) ? out_rmax_addr : 0];

    online_out_unit #(.S(S), .BKV(BKV), .AW(AWX)) u_out (
        .out_addr(out_addr),
        .score_addr(out_score_addr), .score_rdata(out_score_rdata),
        .rmax_addr(out_rmax_addr), .rmax_rdata(out_rmax_rdata),
        .log_rdata(out_log_rdata), .data_out(data_out));

    reg done_seen;
    always @(posedge clk) begin
        if (reset) done_seen <= 0;
        else if (start) done_seen <= 0;
        else if (done) done_seen <= 1;
    end
    assign busy = started && !done_seen && !done;
endmodule
