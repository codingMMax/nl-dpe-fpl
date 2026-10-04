// ============================================================================
// softmax_top.v — NL-DPE softmax, v2 clean-room Stage 3.
//
// Ground truth: `v2/sim/softmax_sim.py` (fused behavior + timing simulator)
// and `v2/oracle/softmax_ref.py` (values). The certified `dpe` primitive is
// instantiated unchanged inside the feed/log engines.
//
// Operator:  softmax_out = clamp(scores - row_max - ACAM_LOG(lq))
//   row_max = max_j scores[r,j]          (streamed tree, CLB_WIDTH B/cycle)
//   EXP     = packed stride-I identity passes, window j -> crossbar j % N_EXP
//   lq      = min(sum(ACAM_EXP) >> log2 S, 127)
//   LOG     = packed stride-I identity passes over the S lq values
//
// Datapath (modules in this file, datapath order):
//   softmax_wprog     identity-eye broadcast + per-stage ACAM mode pins
//   softmax_max_unit  streaming row-max fold + rows_done handshake
//   softmax_exp_feed  one EXP crossbar: two-context window engine, fused
//                     exp_input = score - row_max in the feed path
//   softmax_sum_unit  per-crossbar partial banks + combine pipeline -> lq
//   softmax_log_unit  one LOG crossbar: window engine over the lq stream
//   softmax_out_unit  combinational final result: clamp(score - row_max - log)
//   softmax_top       buffers, instances, window issue, probes
//
// Scheduling contract (see `v2/sim/softmax_sim.py`):
//   * row r's max is written at `log2(S)+1 + ceil((r+1)*S/CLB_WIDTH)`; rows
//     complete in order (`rows_done` counts them);
//   * EXP window j issues once all its rows' maxes are done; windows on a
//     crossbar run back-to-back through the primitive event chain;
//   * lq[r] is written `log2(S)+1` cycles after row r's last element drains;
//   * a LOG window issues once its rows' lq values are all written;
//   * `done` (results computed) fires once the last normalizer row is ready;
//     the final result is read combinationally and is NOT drained/timed.
//
// Setup cycles (excluded from the measured run; reported by the harness):
//   weight programming  R*C broadcast strobes
//   score load          ceil(S*S*8/BUF) packed words
//
// TB probe contract (verification-only internals, NOT ports; frozen names):
//   row_max_q[0:S-1], exp_in_q[0:S*S-1], exp_out_q[0:S*S-1],
//   sum_q[0:S-1], lq_q[0:S-1], log_out_q[0:S-1], u_max.rows_done
// ============================================================================

`timescale 1ns / 1ps

module softmax_wprog #(
    parameter R = 256,
    parameter C = 256
) (
    input  wire clk,
    input  wire reset,
    input  wire prog_start,
    output reg  prog_ready,
    output reg  [7:0] w_data,
    output reg  w_en,
    output wire [1:0] mode_exp,
    output wire [1:0] mode_log
);

    localparam WR_CYC = R * C;
    localparam [1:0] MODE_EXP = 2'b10, MODE_LOG = 2'b11;

    assign mode_exp = MODE_EXP;
    assign mode_log = MODE_LOG;

    reg busy;
    integer wr_cnt;
    integer r, c;

    always @(*) begin
        w_en = busy;
        w_data = (r == c) ? 8'd1 : 8'd0;
    end

    always @(posedge clk) begin
        if (reset) begin
            busy <= 1'b0;
            prog_ready <= 1'b0;
            wr_cnt <= 0;
            r <= 0;
            c <= 0;
        end
        else begin
            if (prog_start && !busy && !prog_ready) begin
                busy <= 1'b1;
                wr_cnt <= 0;
                r <= 0;
                c <= 0;
            end
            if (busy) begin
                if (wr_cnt == WR_CYC - 1) begin
                    busy <= 1'b0;
                    prog_ready <= 1'b1;
                end
                else begin
                    wr_cnt <= wr_cnt + 1;
                    if (c == C - 1) begin
                        c <= 0;
                        r <= r + 1;
                    end
                    else c <= c + 1;
                end
            end
        end
    end

endmodule


// ============================================================================
// softmax_max_unit — streaming row-max fold.
//
// Reads the score buffer CLB_WIDTH bytes/cycle from element 0, folds each row
// into a running max, and pushes the completed row max into a MAX_LAT-deep
// shift pipeline; each value exits (and `rows_done` increments) exactly
// MAX_LAT cycles after its row's last byte entered.
// ============================================================================
module softmax_max_unit #(
    parameter S = 128,
    parameter CLB_WIDTH = 32,
    parameter MAX_LAT = 8,           // log2(S) + 1
    parameter AW = 16
) (
    input  wire clk,
    input  wire reset,
    input  wire start,
    output reg  [AW-1:0] rd_addr,               // group base element
    input  wire [CLB_WIDTH*8-1:0] rd_data,
    output reg  we,
    output reg  [AW-1:0] waddr,
    output reg signed [7:0] wdata,
    output reg  [31:0] rows_done,
    output reg  busy
);

    localparam GROUPS = S / CLB_WIDTH;
    localparam NGROUPS = S * GROUPS;

    integer group_cnt;
    reg signed [7:0] row_running;
    reg signed [7:0] pipe [0:MAX_LAT-1];
    reg [MAX_LAT-1:0] pipe_valid;

    integer i;
    reg signed [7:0] word_max;
    always @(*) begin
        word_max = rd_data[7:0];
        for (i = 1; i < CLB_WIDTH; i = i + 1)
            if ($signed(rd_data[i*8 +: 8]) > word_max)
                word_max = rd_data[i*8 +: 8];
    end

    wire last_group = (group_cnt % GROUPS) == GROUPS - 1;
    wire signed [7:0] row_max_value = (row_running > word_max)
                                      ? row_running : word_max;

    always @(posedge clk) begin
        if (reset) begin
            rd_addr <= 0;
            we <= 1'b0;
            waddr <= 0;
            wdata <= 0;
            rows_done <= 0;
            group_cnt <= 0;
            row_running <= 8'h80;
            busy <= 1'b0;
            pipe_valid <= 0;
            for (i = 0; i < MAX_LAT; i = i + 1) pipe[i] <= 8'h80;
        end
        else begin
            we <= 1'b0;
            if (start && !busy && rows_done == 0) busy <= 1'b1;

            // shift the pipeline every cycle
            pipe_valid[0] <= 1'b0;
            for (i = MAX_LAT-1; i > 0; i = i - 1) begin
                pipe[i] <= pipe[i-1];
                pipe_valid[i] <= pipe_valid[i-1];
            end

            if (busy && group_cnt < NGROUPS) begin
                rd_addr <= (group_cnt + 1) * CLB_WIDTH;   // next group
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
                // consumed: the shift already replaces this stage
            end

            if (busy && group_cnt >= NGROUPS && rows_done == S)
                busy <= 1'b0;
        end
    end

endmodule


// ============================================================================
// softmax_exp_feed — one EXP crossbar: identity pass with fused subtract.
// (two-context window engine; same handshake as the DIMM pool engine)
// ============================================================================
module softmax_exp_feed #(
    parameter S = 128,
    parameter ROW_STRIDE = 0,      // 0 => S (full-row); else the row stride (online = BKV)
    parameter NELEM = 0,           // 0 => S*S; else the flat element count
    parameter RMOD = 0,            // 0 => row index = ae/RS; 1 => ae % RS (factor table)
    parameter R = 256,
    parameter C = 256,
    parameter BUF = 40,
    parameter P = 8,
    parameter AW = 16
) (
    input  wire clk,
    input  wire reset,
    output reg  [(BUF/8)*AW-1:0] score_addrs,
    input  wire [BUF-1:0] score_rdata,
    output reg  [(BUF/8)*AW-1:0] rmax_addrs,
    input  wire [BUF-1:0] rmax_rdata,
    input  wire dpe_load_input_reg,
    input  wire [7:0] dpe_weight_data,
    input  wire [1:0] nl_dpe_control,
    input  wire win_start,
    input  wire [AW-1:0] win_base,
    input  wire [AW-1:0] win_span,         // inclusive element hi for this window
    output wire win_busy,
    output wire win_done,
    // feed-side probe write port (exp_input)
    output reg  in_wr_valid,
    output wire [AW-1:0] in_wr_base,
    output wire [(BUF/8)-1:0] in_wr_mask,
    output wire [BUF-1:0] in_wr_data,
    // drain port (sum unit + exp_out probe)
    output reg  dr_valid,
    output wire [AW-1:0] dr_base,
    output reg  [(BUF/8)-1:0] dr_mask,
    output wire [BUF-1:0] dr_data
);

    localparam EPS  = BUF / 8;
    localparam I    = (R < C) ? R : C;
    localparam LCYC = (R * 8 + BUF - 1) / BUF;
    localparam OCYC = (C * 8 + BUF - 1) / BUF;
    localparam RS   = (ROW_STRIDE == 0) ? S : ROW_STRIDE;
    localparam NEL  = (NELEM == 0) ? S * S : NELEM;

    reg  [BUF-1:0] dpe_in;
    reg            dpe_w_buf_en;
    wire           dpe_ready;
    wire [BUF-1:0] dpe_out;
    wire           dpe_done_w;
    wire           dpe_reg_full;

    dpe #(
        .KERNEL_WIDTH(R), .NUM_COLS(C), .DPE_BUF_WIDTH(BUF), .PRECISION(P)
    ) u_dpe (
        .clk(clk), .reset(reset),
        .data_in(dpe_in), .nl_dpe_control(nl_dpe_control),
        .shift_add_control(1'b0), .w_buf_en(dpe_w_buf_en),
        .shift_add_bypass(1'b0), .load_output_reg(1'b0),
        .load_input_reg(dpe_load_input_reg),
        .MSB_SA_Ready(dpe_ready), .data_out(dpe_out), .dpe_done(dpe_done_w),
        .reg_full(dpe_reg_full)
    );

    reg [AW-1:0] feed_base, feed_hi, lat_base, lat_hi;
    integer      feed_cnt;
    reg          feeding, have_win;
    reg [AW-1:0] drn_base, drn_hi;
    integer      drn_cnt;
    reg          draining;
    reg [AW-1:0] que_base, que_hi;
    reg          que_valid;

    integer jf, jd;
    integer acol, ae, arow, fcol, fe, dcol, de;

    wire take_now = win_start && !win_busy && dpe_ready;
    wire [AW-1:0] start_base = take_now ? win_base : lat_base;
    wire [AW-1:0] start_hi   = take_now ? win_span : lat_hi;
    wire start_feed = take_now || (have_win && !feeding && dpe_ready);
    wire feed_active = feeding || start_feed;
    wire [AW-1:0] feed_base_w = start_feed ? start_base : feed_base;
    wire [AW-1:0] feed_hi_w   = start_feed ? start_hi : feed_hi;
    wire [AW-1:0] feed_word = start_feed ? {AW{1'b0}} : feed_cnt;
    wire feed_done_w = start_feed ? (LCYC == 1)
                      : (feeding && dpe_ready && (feed_cnt == LCYC-1));
    wire [AW-1:0] feed_done_base = start_feed ? start_base : feed_base;
    wire [AW-1:0] feed_done_hi   = start_feed ? start_hi   : feed_hi;
    wire drain_done_w = draining && dpe_done_w;

    assign win_busy = feeding || have_win;
    assign win_done = draining && dpe_done_w;
    assign dr_base = drn_base + drn_cnt * EPS;
    assign dr_data = dpe_out;

    // feed probe registered together with in_wr_valid (write same cycle)
    reg [AW-1:0] in_wr_base_q;
    reg [(BUF/8)-1:0] in_wr_mask_q, in_wr_mask_next;
    reg [BUF-1:0] in_wr_data_q;
    always @(posedge clk) begin
        in_wr_base_q <= feed_base_w + feed_word * EPS;
        in_wr_mask_q <= in_wr_mask_next;
        in_wr_data_q <= dpe_in;
    end
    assign in_wr_base = in_wr_base_q;
    assign in_wr_mask = in_wr_mask_q;
    assign in_wr_data = in_wr_data_q;

    always @(*) begin
        score_addrs = 0;
        rmax_addrs = 0;
        in_wr_mask_next = 0;
        if (feed_active) begin
            for (jf = 0; jf < EPS; jf = jf + 1) begin
                acol = feed_word * EPS + jf;
                ae = feed_base_w + acol;
                if ((acol < I) && (ae < NEL) && (ae <= feed_hi_w)) begin
                    score_addrs[jf*AW +: AW] = ae;
                    arow = RMOD ? (ae % RS) : (ae / RS);
                    rmax_addrs[jf*AW +: AW] = arow;
                    in_wr_mask_next[jf] = 1'b1;
                end
            end
        end
    end

    reg signed [8:0] din_diff;
    always @(*) begin
        dpe_in = 0;
        if (dpe_load_input_reg) begin
            dpe_in = {{(BUF-8){1'b0}}, dpe_weight_data};
        end
        else if (feed_active) begin
            for (jf = 0; jf < EPS; jf = jf + 1) begin
                fcol = feed_word * EPS + jf;
                fe = feed_base_w + fcol;
                if ((fcol < I) && (fe < NEL) && (fe <= feed_hi_w)) begin
                    din_diff = $signed(score_rdata[jf*8 +: 8])
                               - $signed(rmax_rdata[jf*8 +: 8]);
                    if (din_diff < -9'sd128)
                        dpe_in[jf*8 +: 8] = 8'h80;
                    else
                        dpe_in[jf*8 +: 8] = din_diff[7:0];
                end
            end
        end
    end

    always @(*) dpe_w_buf_en = feed_active && dpe_ready;

    always @(*) begin
        dr_valid = 1'b0;
        dr_mask = 0;
        if (draining && dpe_reg_full) begin
            dr_valid = 1'b1;
            for (jd = 0; jd < EPS; jd = jd + 1) begin
                dcol = drn_cnt * EPS + jd;
                de = drn_base + dcol;
                if ((dcol < I) && (de < NEL) && (de <= drn_hi)) dr_mask[jd] = 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            feed_base <= 0; feed_hi <= 0; lat_base <= 0; lat_hi <= 0;
            feed_cnt <= 0; feeding <= 1'b0; have_win <= 1'b0;
            drn_base <= 0; drn_hi <= 0; drn_cnt <= 0; draining <= 1'b0;
            que_base <= 0; que_hi <= 0; que_valid <= 1'b0;
            in_wr_valid <= 1'b0;
        end
        else begin
            in_wr_valid <= feed_active;
            if (win_start && !win_busy && !dpe_ready) begin
                lat_base <= win_base; lat_hi <= win_span; have_win <= 1'b1;
            end
            if (start_feed) begin
                have_win <= 1'b0;
                if (LCYC == 1) feeding <= 1'b0;
                else begin
                    feeding <= 1'b1; feed_base <= start_base;
                    feed_hi <= start_hi; feed_cnt <= 1;
                end
            end
            else if (feeding && dpe_ready && (feed_cnt != LCYC-1))
                feed_cnt <= feed_cnt + 1;
            else if (feeding && dpe_ready)
                feeding <= 1'b0;

            if (drain_done_w) begin
                if (que_valid) begin
                    drn_base <= que_base; drn_hi <= que_hi; drn_cnt <= 0;
                    que_base <= feed_done_base; que_hi <= feed_done_hi;
                    que_valid <= feed_done_w;
                end
                else if (feed_done_w) begin
                    drn_base <= feed_done_base; drn_hi <= feed_done_hi;
                    drn_cnt <= 0;
                end
                else draining <= 1'b0;
            end
            else if (feed_done_w) begin
                if (draining) begin
                    que_base <= feed_done_base; que_hi <= feed_done_hi;
                    que_valid <= 1'b1;
                end
                else begin
                    draining <= 1'b1; drn_base <= feed_done_base;
                    drn_hi <= feed_done_hi; drn_cnt <= 0;
                end
            end

            if (draining && dpe_reg_full && (drn_cnt != OCYC-1))
                drn_cnt <= drn_cnt + 1;
        end
    end
endmodule



// ============================================================================
// softmax_sum_unit — per-crossbar partial banks + combine pipeline -> lq.
//
// Each drained element (crossbar c, element e) accumulates into
// bank[c][e/S]; when a row's element count reaches S, the partials are
// combined and shifted through SUM_LAT registers, then lq = min(sum >> log2 S,
// 127) is written and `lq_done_count` increments (rows complete in order).
// ============================================================================
module softmax_sum_unit #(
    parameter S = 128,
    parameter RS = 0,              // 0 => S; else the row stride (online = BKV)
    parameter NROWS = 0,           // 0 => S; else the row count (online = B*S)
    parameter SPAN = 0,            // 0 => S; else elements per completed row (online = BKV)
    parameter OUT_LQ = 1,          // 1 => also emit lq = min(sum>>log2 S,127)
    parameter N_EXP = 1,
    parameter PORTS = 16,
    parameter SUM_LAT = 8,
    parameter AW = 16,
    parameter DIRECT_COMMIT = 0
) (
    input  wire clk,
    input  wire reset,
    input  wire [N_EXP-1:0] dr_valid,
    input  wire [N_EXP*AW-1:0] dr_base,
    input  wire [N_EXP*(PORTS/8)-1:0] dr_mask,
    input  wire [N_EXP*PORTS-1:0] dr_data,
    output reg  sp_we,                     // raw combined sum for a completed row
    output reg  [AW-1:0] sp_waddr,
    output reg  [31:0] sp_wdata,
    output reg  lq_we,                     // (OUT_LQ) shifted / capped lq
    output reg  [AW-1:0] lq_waddr,
    output reg  [7:0] lq_wdata,
    output reg  [31:0] lq_done_count,
    // DIRECT_COMMIT per-crossbar write bus (online): a row's `sp` is written
    // at the cycle its element count completes (out-of-order, no row-order
    // pipe). Top writes sp_q from these; the scalar sp_we path is unused.
    output wire [N_EXP-1:0] dc_we,
    output wire [N_EXP*AW-1:0] dc_waddr,
    output wire [N_EXP*32-1:0] dc_wdata
);

    localparam EPS = PORTS / 8;
    localparam LOG2S = $clog2(S);
    localparam RST = (RS == 0) ? S : RS;
    localparam NR  = (NROWS == 0) ? S : NROWS;
    localparam SP  = (SPAN == 0) ? S : SPAN;

    reg [31:0] bank [0:N_EXP-1][0:NR-1];
    reg [31:0] count [0:NR-1];
    integer rows_pushed;

    reg [31:0] add_cnt [0:NR-1];
    reg [31:0] cnt_p [0:N_EXP-1][0:NR-1];
    reg [31:0] bank_add [0:N_EXP-1][0:NR-1];
    integer p, j, e, r;

    always @(*) begin
        for (r = 0; r < NR; r = r + 1) begin
            add_cnt[r] = 0;
            for (p = 0; p < N_EXP; p = p + 1) bank_add[p][r] = 0;
        end
        for (p = 0; p < N_EXP; p = p + 1)
            for (r = 0; r < NR; r = r + 1) cnt_p[p][r] = 0;
        for (p = 0; p < N_EXP; p = p + 1)
            if (dr_valid[p])
                for (j = 0; j < EPS; j = j + 1)
                    if (dr_mask[p*EPS + j]) begin
                        e = dr_base[p*AW +: AW] + j;
                        r = e / RST;
                        if (r < NR) begin
                            add_cnt[r] = add_cnt[r] + 1;
                            cnt_p[p][r] = cnt_p[p][r] + 1;
                            bank_add[p][r] = bank_add[p][r]
                                + {24'b0, dr_data[p*PORTS + j*8 +: 8]};
                        end
                    end
    end

    reg [SUM_LAT-1:0] pipe_valid;
    reg [AW-1:0] pipe_row [0:SUM_LAT-1];
    reg [31:0] total_r;
    reg [31:0] lq_value;
    integer cidx;
    always @(*) begin
        total_r = 0;
        for (cidx = 0; cidx < N_EXP; cidx = cidx + 1)
            total_r = total_r + bank[cidx][pipe_row[SUM_LAT-1]];
        lq_value = total_r >> LOG2S;
        if (lq_value > 32'd127) lq_value = 32'd127;
    end

    // ---------------- direct per-row commit (DIRECT_COMMIT=1, online) ------
    // A row's total = sum of the per-crossbar banks (+ this cycle's drain
    // words). Row r's live total is written at the cycle r's element count
    // completes, i.e. when (count < SP) && (count + add_cnt >= SP); count
    // itself then reaches SP, so each row commits exactly once. Rows are
    // single-window (BKV divides I); a crossbar's drain word completes at
    // most one row (two completing rows in one word need EPS > BKV), so one
    // write slot per crossbar per cycle suffices.
    reg [N_EXP-1:0]     dc_v;
    reg [N_EXP*AW-1:0]  dc_abus;
    reg [N_EXP*32-1:0]  dc_dbus;
    integer d1, d2, d3;
    always @(*) begin
        for (d1 = 0; d1 < N_EXP; d1 = d1 + 1) begin
            dc_v[d1] = 1'b0;
            dc_abus[d1*AW +: AW] = 0;
            dc_dbus[d1*32 +: 32] = 32'b0;
        end
        for (d1 = 0; d1 < N_EXP; d1 = d1 + 1)
            for (d2 = 0; d2 < NR; d2 = d2 + 1)
                if (!dc_v[d1] && (cnt_p[d1][d2] != 0)
                    && (count[d2] < SP)
                    && ((count[d2] + add_cnt[d2]) >= SP)) begin
                    dc_abus[d1*AW +: AW] = d2[AW-1:0];
                    dc_v[d1] = 1'b1;
                    for (d3 = 0; d3 < N_EXP; d3 = d3 + 1)
                        dc_dbus[d1*32 +: 32] = dc_dbus[d1*32 +: 32]
                            + bank[d3][d2] + bank_add[d3][d2];
                end
    end
    assign dc_we = (DIRECT_COMMIT != 0) ? dc_v : {N_EXP{1'b0}};
    assign dc_waddr = dc_abus;
    assign dc_wdata = dc_dbus;

    always @(posedge clk) begin
        if (reset) begin
            sp_we <= 1'b0; lq_we <= 1'b0; lq_done_count <= 0;
            rows_pushed <= 0; pipe_valid <= 0;
            for (p = 0; p < N_EXP; p = p + 1)
                for (r = 0; r < NR; r = r + 1) bank[p][r] <= 0;
            for (r = 0; r < NR; r = r + 1) count[r] <= 0;
        end
        else begin
            sp_we <= 1'b0;
            lq_we <= 1'b0;

            for (r = 0; r < NR; r = r + 1)
                if (add_cnt[r] != 0) count[r] <= count[r] + add_cnt[r];
            for (p = 0; p < N_EXP; p = p + 1)
                for (r = 0; r < NR; r = r + 1)
                    if (bank_add[p][r] != 0)
                        bank[p][r] <= bank[p][r] + bank_add[p][r];

            pipe_valid[0] <= 1'b0;
            for (j = SUM_LAT-1; j > 0; j = j - 1) begin
                pipe_valid[j] <= pipe_valid[j-1];
                pipe_row[j] <= pipe_row[j-1];
            end

            if ((rows_pushed < NR)
                && ((count[rows_pushed] + add_cnt[rows_pushed]) >= SP)) begin
                pipe_row[0] <= rows_pushed[AW-1:0];
                pipe_valid[0] <= 1'b1;
                rows_pushed <= rows_pushed + 1;
            end

            if (pipe_valid[SUM_LAT-1]) begin
                sp_we <= 1'b1;
                sp_waddr <= pipe_row[SUM_LAT-1];
                sp_wdata <= total_r;
                lq_done_count <= lq_done_count + 1;
                if (OUT_LQ) begin
                    lq_we <= 1'b1;
                    lq_waddr <= pipe_row[SUM_LAT-1];
                    lq_wdata <= lq_value[7:0];
                end
            end
        end
    end
endmodule



// ============================================================================
// softmax_log_unit — one LOG crossbar (identity | ACAM LOG) over the lq SRAM.
// ============================================================================
module softmax_log_unit #(
    parameter S = 128,
    parameter R = 256,
    parameter C = 256,
    parameter BUF = 40,
    parameter P = 8,
    parameter AW = 16
) (
    input  wire clk,
    input  wire reset,
    output reg  [(BUF/8)*AW-1:0] lq_addrs,
    input  wire [BUF-1:0] lq_rdata,
    input  wire dpe_load_input_reg,
    input  wire [7:0] dpe_weight_data,
    input  wire [1:0] nl_dpe_control,
    input  wire win_start,
    input  wire [AW-1:0] win_base,
    output wire win_busy,
    output wire win_done,
    output reg  dr_valid,
    output wire [AW-1:0] dr_base,
    output reg  [(BUF/8)-1:0] dr_mask,
    output wire [BUF-1:0] dr_data
);

    localparam EPS  = BUF / 8;
    localparam I    = (R < C) ? R : C;
    localparam LCYC = (R * 8 + BUF - 1) / BUF;
    localparam OCYC = (C * 8 + BUF - 1) / BUF;

    reg  [BUF-1:0] dpe_in;
    reg            dpe_w_buf_en;
    wire           dpe_ready;
    wire [BUF-1:0] dpe_out;
    wire           dpe_done_w;
    wire           dpe_reg_full;

    dpe #(
        .KERNEL_WIDTH(R), .NUM_COLS(C), .DPE_BUF_WIDTH(BUF), .PRECISION(P)
    ) u_dpe (
        .clk(clk), .reset(reset),
        .data_in(dpe_in), .nl_dpe_control(nl_dpe_control),
        .shift_add_control(1'b0), .w_buf_en(dpe_w_buf_en),
        .shift_add_bypass(1'b0), .load_output_reg(1'b0),
        .load_input_reg(dpe_load_input_reg),
        .MSB_SA_Ready(dpe_ready), .data_out(dpe_out), .dpe_done(dpe_done_w),
        .reg_full(dpe_reg_full)
    );

    reg [AW-1:0] feed_base, lat_base;
    integer      feed_cnt;
    reg          feeding, have_win;
    reg [AW-1:0] drn_base;
    integer      drn_cnt;
    reg          draining;
    reg [AW-1:0] que_base;
    reg          que_valid;

    integer jf, jd, acol, ae, fcol, fe, dcol, de;

    wire take_now = win_start && !win_busy && dpe_ready;
    wire [AW-1:0] start_base = take_now ? win_base : lat_base;
    wire start_feed = take_now || (have_win && !feeding && dpe_ready);
    wire feed_active = feeding || start_feed;
    wire [AW-1:0] feed_base_w = start_feed ? start_base : feed_base;
    wire [AW-1:0] feed_word = start_feed ? {AW{1'b0}} : feed_cnt;
    wire feed_done_w = start_feed ? (LCYC == 1)
                      : (feeding && dpe_ready && (feed_cnt == LCYC-1));
    wire [AW-1:0] feed_done_base = start_feed ? start_base : feed_base;
    wire drain_done_w = draining && dpe_done_w;

    assign win_busy = feeding || have_win;
    assign win_done = draining && dpe_done_w;
    assign dr_base = drn_base + drn_cnt * EPS;
    assign dr_data = dpe_out;

    always @(*) begin
        lq_addrs = 0;
        if (feed_active) begin
            for (jf = 0; jf < EPS; jf = jf + 1) begin
                acol = feed_word * EPS + jf;
                ae = feed_base_w + acol;
                if ((acol < I) && (ae < S))
                    lq_addrs[jf*AW +: AW] = ae;
            end
        end
    end

    always @(*) begin
        dpe_in = 0;
        if (dpe_load_input_reg)
            dpe_in = {{(BUF-8){1'b0}}, dpe_weight_data};
        else if (feed_active)
            for (jf = 0; jf < EPS; jf = jf + 1) begin
                fcol = feed_word * EPS + jf;
                fe = feed_base_w + fcol;
                if ((fcol < I) && (fe < S))
                    dpe_in[jf*8 +: 8] = lq_rdata[jf*8 +: 8];
            end
    end

    always @(*) dpe_w_buf_en = feed_active && dpe_ready;

    always @(*) begin
        dr_valid = 1'b0;
        dr_mask = 0;
        if (draining && dpe_reg_full) begin
            dr_valid = 1'b1;
            for (jd = 0; jd < EPS; jd = jd + 1) begin
                dcol = drn_cnt * EPS + jd;
                de = drn_base + dcol;
                if ((dcol < I) && (de < S)) dr_mask[jd] = 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            feed_base <= 0;
            lat_base <= 0;
            feed_cnt <= 0;
            feeding <= 1'b0;
            have_win <= 1'b0;
            drn_base <= 0;
            drn_cnt <= 0;
            draining <= 1'b0;
            que_base <= 0;
            que_valid <= 1'b0;
        end
        else begin
            if (win_start && !win_busy && !dpe_ready) begin
                lat_base <= win_base;
                have_win <= 1'b1;
            end
            if (start_feed) begin
                have_win <= 1'b0;
                if (LCYC == 1) feeding <= 1'b0;
                else begin
                    feeding  <= 1'b1;
                    feed_base <= start_base;
                    feed_cnt <= 1;
                end
            end
            else if (feeding && dpe_ready && (feed_cnt != LCYC-1))
                feed_cnt <= feed_cnt + 1;
            else if (feeding && dpe_ready)
                feeding <= 1'b0;

            if (drain_done_w) begin
                if (que_valid) begin
                    drn_base  <= que_base;
                    drn_cnt   <= 0;
                    que_base  <= feed_done_base;
                    que_valid <= feed_done_w;
                end
                else if (feed_done_w) begin
                    drn_base <= feed_done_base;
                    drn_cnt  <= 0;
                end
                else draining <= 1'b0;
            end
            else if (feed_done_w) begin
                if (draining) begin
                    que_base  <= feed_done_base;
                    que_valid <= 1'b1;
                end
                else begin
                    draining <= 1'b1;
                    drn_base <= feed_done_base;
                    drn_cnt  <= 0;
                end
            end

            if (draining && dpe_reg_full && (drn_cnt != OCYC-1))
                drn_cnt <= drn_cnt + 1;
        end
    end

endmodule


// ============================================================================
// softmax_out_unit — final softmax result, purely combinational:
//   data_out = clamp(score[out_addr] - row_max[out_addr/S] - log[out_addr/S])
// No drain/serialize stage; the top generates `done` (whole result computed).
// ============================================================================
module softmax_out_unit #(
    parameter S = 128,
    parameter AW = 16
) (
    input  wire [AW-1:0] out_addr,
    output wire [AW-1:0] score_addr,
    input  wire [7:0] score_rdata,
    output wire [AW-1:0] rmax_addr,
    input  wire [7:0] rmax_rdata,
    input  wire [7:0] log_rdata,
    output reg  [7:0] data_out
);

    assign score_addr = out_addr;
    assign rmax_addr  = out_addr / S;

    reg signed [15:0] acc;
    always @(*) begin
        acc = $signed(score_rdata) - $signed(rmax_rdata) - $signed(log_rdata);
        if (acc > 16'sd127) data_out = 8'd127;
        else if (acc < -16'sd128) data_out = 8'd128;
        else data_out = acc[7:0];
    end

endmodule


// ============================================================================
// softmax_top — buffers, instances, window issue, probes.
// ============================================================================
module softmax_top #(
    parameter S = 128,
    parameter R = 256,
    parameter C = 256,
    parameter BUF = 40,
    parameter P = 8,
    parameter N_EXP = 1,
    parameter N_LOG = 1
) (
    input  wire clk,
    input  wire reset,
    input  wire prog_start,
    output wire prog_ready,
    input  wire [BUF-1:0] score_in,
    input  wire score_en,
    output wire load_ready,
    input  wire start,
    output wire busy,
    input  wire [$clog2(S*S):0] out_addr,
    output wire [7:0] data_out,
    output reg  done
);

    localparam EPS     = BUF / 8;
    localparam I       = (R < C) ? R : C;
    localparam SQ      = S * S;
    localparam LOG2S   = $clog2(S);
    localparam PEXP    = (SQ + I - 1) / I;
    localparam PLOG    = (S + I - 1) / I;
    localparam CLB_WIDTH = 32;
    localparam MAX_LAT = LOG2S + 1;
    localparam SUM_LAT = LOG2S + 1;
    localparam CLAMP_LAT = 2;
    localparam AW      = $clog2(SQ) + 1;
    localparam AWM     = $clog2(S) + 1;
    localparam LOAD_WORDS = (SQ + EPS - 1) / EPS;

    integer c, j;
    integer g, n, issue_i;

    // ---------------- buffers + probes -------------------------------------
    reg signed [7:0]  score_buf [0:SQ-1];
    reg signed [7:0]  row_max_q [0:S-1];
    reg signed [7:0]  lq_q      [0:S-1];
    reg signed [7:0]  exp_in_q  [0:SQ-1];
    reg signed [7:0]  exp_out_q [0:SQ-1];
    reg signed [31:0] sum_q     [0:S-1];
    reg signed [7:0]  log_out_q [0:S-1];

    reg started;
    reg [AW-1:0] load_word;
    always @(posedge clk) begin
        if (reset) begin
            started <= 1'b0;
            load_word <= 0;
        end
        else begin
            if (start) started <= 1'b1;
            if (score_en && load_ready && (load_word < LOAD_WORDS)) begin
                for (j = 0; j < EPS; j = j + 1)
                    if (load_word * EPS + j < SQ)
                        score_buf[load_word * EPS + j] <= score_in[j*8 +: 8];
                load_word <= load_word + 1;
            end
        end
    end
    assign load_ready = prog_ready && !started;

    // ---------------- weight programming -----------------------------------
    wire [7:0] w_data;
    wire       w_en;
    wire [1:0] mode_exp, mode_log;
    softmax_wprog #(.R(R), .C(C)) u_wprog (
        .clk(clk), .reset(reset), .prog_start(prog_start),
        .prog_ready(prog_ready), .w_data(w_data), .w_en(w_en),
        .mode_exp(mode_exp), .mode_log(mode_log)
    );

    // ---------------- row-max unit -----------------------------------------
    wire [AW-1:0] max_rd_addr;
    wire [CLB_WIDTH*8-1:0] max_rd_data;
    reg  [CLB_WIDTH*8-1:0] max_rd_data_r;
    integer mi;
    always @(*) begin
        for (mi = 0; mi < CLB_WIDTH; mi = mi + 1) begin
            if (max_rd_addr + mi < SQ)
                max_rd_data_r[mi*8 +: 8] = score_buf[max_rd_addr + mi];
            else
                max_rd_data_r[mi*8 +: 8] = 8'h80;
        end
    end
    assign max_rd_data = max_rd_data_r;

    wire        max_we;
    wire [AW-1:0] max_waddr;
    wire [7:0]  max_wdata;
    wire [31:0] rows_done;
    wire        max_busy;
    softmax_max_unit #(
        .S(S), .CLB_WIDTH(CLB_WIDTH), .MAX_LAT(MAX_LAT), .AW(AW)
    ) u_max (
        .clk(clk), .reset(reset), .start(start),
        .rd_addr(max_rd_addr), .rd_data(max_rd_data),
        .we(max_we), .waddr(max_waddr), .wdata(max_wdata),
        .rows_done(rows_done), .busy(max_busy)
    );
    always @(posedge clk)
        if (max_we) row_max_q[max_waddr] <= max_wdata;

    // ---------------- EXP feeds --------------------------------------------
    wire [(BUF/8)*AW-1:0] exp_score_addrs [0:N_EXP-1];
    reg  [BUF-1:0]        exp_score_rdata [0:N_EXP-1];
    wire [(BUF/8)*AW-1:0] exp_rmax_addrs  [0:N_EXP-1];
    reg  [BUF-1:0]        exp_rmax_rdata  [0:N_EXP-1];
    reg  [N_EXP-1:0]      exp_win_start;
    wire [N_EXP-1:0]      exp_win_busy, exp_win_done;
    reg  [N_EXP*AW-1:0]   exp_win_base;
    wire [N_EXP-1:0]      exp_in_valid;
    wire [N_EXP*AW-1:0]   exp_in_base;
    wire [N_EXP*EPS-1:0]  exp_in_mask;
    wire [N_EXP*BUF-1:0]  exp_in_data;
    wire [N_EXP-1:0]      exp_dr_valid;
    wire [N_EXP*AW-1:0]   exp_dr_base;
    wire [N_EXP*EPS-1:0]  exp_dr_mask;
    wire [N_EXP*BUF-1:0]  exp_dr_data;

    genvar gc;
    generate
        for (gc = 0; gc < N_EXP; gc = gc + 1) begin : gen_exp
            softmax_exp_feed #(
                .S(S), .ROW_STRIDE(S), .NELEM(SQ),
                .R(R), .C(C), .BUF(BUF), .P(P), .AW(AW)
            ) u_feed (
                .clk(clk), .reset(reset),
                .score_addrs(exp_score_addrs[gc]),
                .score_rdata(exp_score_rdata[gc]),
                .rmax_addrs(exp_rmax_addrs[gc]),
                .rmax_rdata(exp_rmax_rdata[gc]),
                .dpe_load_input_reg(w_en), .dpe_weight_data(w_data),
                .nl_dpe_control(mode_exp),
                .win_start(exp_win_start[gc]),
                .win_base(exp_win_base[gc*AW +: AW]),
                .win_span(SQ - 1),
                .win_busy(exp_win_busy[gc]), .win_done(exp_win_done[gc]),
                .in_wr_valid(exp_in_valid[gc]),
                .in_wr_base(exp_in_base[gc*AW +: AW]),
                .in_wr_mask(exp_in_mask[gc*EPS +: EPS]),
                .in_wr_data(exp_in_data[gc*BUF +: BUF]),
                .dr_valid(exp_dr_valid[gc]),
                .dr_base(exp_dr_base[gc*AW +: AW]),
                .dr_mask(exp_dr_mask[gc*EPS +: EPS]),
                .dr_data(exp_dr_data[gc*BUF +: BUF])
            );
        end
    endgenerate

    always @(*) begin
        for (c = 0; c < N_EXP; c = c + 1) begin
            for (j = 0; j < EPS; j = j + 1) begin
                if (exp_score_addrs[c][j*AW +: AW] < SQ)
                    exp_score_rdata[c][j*8 +: 8] =
                        score_buf[exp_score_addrs[c][j*AW +: AW]];
                else
                    exp_score_rdata[c][j*8 +: 8] = 8'h80;
                if (exp_rmax_addrs[c][j*AW +: AW] < S) begin
                    // write-through bypass: the SRAM write commits one cycle
                    // after rows_done, so forward the in-flight write
                    if (max_we && (max_waddr == exp_rmax_addrs[c][j*AW +: AW]))
                        exp_rmax_rdata[c][j*8 +: 8] = max_wdata;
                    else
                        exp_rmax_rdata[c][j*8 +: 8] =
                            row_max_q[exp_rmax_addrs[c][j*AW +: AW]];
                end
                else
                    exp_rmax_rdata[c][j*8 +: 8] = 8'h80;
            end
        end
    end

    always @(posedge clk) begin
        for (c = 0; c < N_EXP; c = c + 1) begin
            if (exp_in_valid[c])
                for (j = 0; j < EPS; j = j + 1)
                    if (exp_in_mask[c*EPS + j]
                        && (exp_in_base[c*AW +: AW] + j < SQ))
                        exp_in_q[exp_in_base[c*AW +: AW] + j]
                            <= exp_in_data[c*BUF + j*8 +: 8];
            if (exp_dr_valid[c])
                for (j = 0; j < EPS; j = j + 1)
                    if (exp_dr_mask[c*EPS + j]
                        && (exp_dr_base[c*AW +: AW] + j < SQ))
                        exp_out_q[exp_dr_base[c*AW +: AW] + j]
                            <= exp_dr_data[c*BUF + j*8 +: 8];
        end
    end

    // ---------------- sum / lq ---------------------------------------------
    wire        sum_lq_we, sum_q_we;
    wire [AW-1:0] sum_lq_waddr, sum_sum_waddr;
    wire [7:0]  sum_lq_wdata;
    wire [31:0] sum_sum_wdata;
    wire [31:0] lq_done_count;
    softmax_sum_unit #(
        .S(S), .RS(S), .NROWS(S), .SPAN(S), .OUT_LQ(1),
        .N_EXP(N_EXP), .PORTS(BUF), .SUM_LAT(SUM_LAT), .AW(AW)
    ) u_sum (
        .clk(clk), .reset(reset),
        .dr_valid(exp_dr_valid), .dr_base(exp_dr_base),
        .dr_mask(exp_dr_mask), .dr_data(exp_dr_data),
        .lq_we(sum_lq_we), .lq_waddr(sum_lq_waddr), .lq_wdata(sum_lq_wdata),
        .lq_done_count(lq_done_count),
        .sp_we(sum_q_we), .sp_waddr(sum_sum_waddr),
        .sp_wdata(sum_sum_wdata)
    );
    always @(posedge clk) begin
        if (sum_lq_we) lq_q[sum_lq_waddr] <= sum_lq_wdata;
        if (sum_q_we)  sum_q[sum_sum_waddr] <= sum_sum_wdata;
    end

    // ---------------- LOG units --------------------------------------------
    wire [(BUF/8)*AW-1:0] log_addrs [0:N_LOG-1];
    reg  [BUF-1:0]        log_rdata [0:N_LOG-1];
    reg  [N_LOG-1:0]      log_win_start;
    wire [N_LOG-1:0]      log_win_busy, log_win_done;
    reg  [N_LOG*AW-1:0]   log_win_base;
    wire [N_LOG-1:0]      log_dr_valid;
    wire [N_LOG*AW-1:0]   log_dr_base;
    wire [N_LOG*EPS-1:0]  log_dr_mask;
    wire [N_LOG*BUF-1:0]  log_dr_data;

    generate
        for (gc = 0; gc < N_LOG; gc = gc + 1) begin : gen_log
            softmax_log_unit #(
                .S(S), .R(R), .C(C), .BUF(BUF), .P(P), .AW(AW)
            ) u_log (
                .clk(clk), .reset(reset),
                .lq_addrs(log_addrs[gc]), .lq_rdata(log_rdata[gc]),
                .dpe_load_input_reg(w_en), .dpe_weight_data(w_data),
                .nl_dpe_control(mode_log),
                .win_start(log_win_start[gc]),
                .win_base(log_win_base[gc*AW +: AW]),
                .win_busy(log_win_busy[gc]), .win_done(log_win_done[gc]),
                .dr_valid(log_dr_valid[gc]),
                .dr_base(log_dr_base[gc*AW +: AW]),
                .dr_mask(log_dr_mask[gc*EPS +: EPS]),
                .dr_data(log_dr_data[gc*BUF +: BUF])
            );
        end
    endgenerate

    always @(*) begin
        for (c = 0; c < N_LOG; c = c + 1) begin
            for (j = 0; j < EPS; j = j + 1) begin
                if (log_addrs[c][j*AW +: AW] < S)
                    log_rdata[c][j*8 +: 8] = lq_q[log_addrs[c][j*AW +: AW]];
                else
                    log_rdata[c][j*8 +: 8] = 8'h80;
            end
        end
    end

    always @(posedge clk) begin
        for (c = 0; c < N_LOG; c = c + 1)
            if (log_dr_valid[c])
                for (j = 0; j < EPS; j = j + 1)
                    if (log_dr_mask[c*EPS + j]
                        && (log_dr_base[c*AW +: AW] + j < S))
                        log_out_q[log_dr_base[c*AW +: AW] + j]
                            <= log_dr_data[c*BUF + j*8 +: 8];
    end

    // ---------------- final result (combinational; no drain) ---------------
    wire [AW-1:0] out_score_addr, out_rmax_addr;
    wire [7:0] out_score_rdata, out_rmax_rdata, out_log_rdata;
    assign out_score_rdata = score_buf[(out_score_addr < SQ)
                                       ? out_score_addr : 0];
    assign out_rmax_rdata  = row_max_q[(out_rmax_addr < S)
                                       ? out_rmax_addr : 0];
    assign out_log_rdata   = log_out_q[(out_rmax_addr < S)
                                       ? out_rmax_addr : 0];
    softmax_out_unit #(.S(S), .AW(AW)) u_out (
        .out_addr(out_addr),
        .score_addr(out_score_addr), .score_rdata(out_score_rdata),
        .rmax_addr(out_rmax_addr), .rmax_rdata(out_rmax_rdata),
        .log_rdata(out_log_rdata), .data_out(data_out)
    );

    // ---------------- results ready (whole result computed) ----------------
    // `done` at the cycle after the last normalizer row's LOG drain word
    // (replicates the old out_unit's CLAMP_LAT=1 marking). No drain is timed.
    reg [N_LOG-1:0]     dn_valid;
    reg [N_LOG*AW-1:0]  dn_base;
    reg [N_LOG*EPS-1:0] dn_mask;
    integer dn_j, dn_p;
    always @(posedge clk) begin
        if (reset) begin
            done <= 1'b0; dn_valid <= 0; dn_base <= 0; dn_mask <= 0;
        end
        else begin
            done <= 1'b0;
            dn_valid <= log_dr_valid;
            dn_base  <= log_dr_base;
            dn_mask  <= log_dr_mask;
            for (dn_p = 0; dn_p < N_LOG; dn_p = dn_p + 1)
                if (dn_valid[dn_p])
                    for (dn_j = 0; dn_j < EPS; dn_j = dn_j + 1)
                        if (dn_mask[dn_p*EPS + dn_j]
                            && ((dn_base[dn_p*AW +: AW] + dn_j) == S-1))
                            done <= 1'b1;
        end
    end
    // NOTE: for S <= I (the target corpus) PLOG == 1, so only LOG unit 0
    // carries windows; N_LOG > 1 is an idle-resource sweep axis.

    // ---------------- window issue (rows_done / lq_done gated) -------------
    integer exp_iss [0:N_EXP-1];
    reg     exp_pend [0:N_EXP-1];
    integer log_iss [0:N_LOG-1];
    reg     log_pend [0:N_LOG-1];

    wire [N_EXP-1:0] exp_need_ok;
    wire [N_LOG-1:0] log_need_ok;
    genvar gi;
    generate
        for (gi = 0; gi < N_EXP; gi = gi + 1) begin : gen_exp_need
            wire [31:0] gnext = exp_iss[gi] * N_EXP + gi;
            wire [31:0] nend  = (gnext + 1) * I;
            wire [31:0] nrows = ((nend > SQ ? SQ : nend) + S - 1) / S;
            assign exp_need_ok[gi] = (gnext < PEXP) && (rows_done >= nrows);
        end
        for (gi = 0; gi < N_LOG; gi = gi + 1) begin : gen_log_need
            wire [31:0] gnext = log_iss[gi] * N_LOG + gi;
            wire [31:0] nend  = (gnext + 1) * I;
            wire [31:0] nrows = (nend > S) ? S : nend;
            assign log_need_ok[gi] =
                (gnext < PLOG) && (lq_done_count >= nrows);
        end
    endgenerate

    always @(*) begin
        exp_win_start = 0;
        exp_win_base = 0;
        for (issue_i = 0; issue_i < N_EXP; issue_i = issue_i + 1) begin
            exp_win_start[issue_i] = exp_pend[issue_i]
                || (started && exp_need_ok[issue_i] && !exp_win_busy[issue_i]);
            exp_win_base[issue_i*AW +: AW] =
                (exp_iss[issue_i] * N_EXP + issue_i) * I;
        end
        log_win_start = 0;
        log_win_base = 0;
        for (issue_i = 0; issue_i < N_LOG; issue_i = issue_i + 1) begin
            log_win_start[issue_i] = log_pend[issue_i]
                || (started && log_need_ok[issue_i] && !log_win_busy[issue_i]);
            log_win_base[issue_i*AW +: AW] =
                (log_iss[issue_i] * N_LOG + issue_i) * I;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            for (issue_i = 0; issue_i < N_EXP; issue_i = issue_i + 1) begin
                exp_iss[issue_i] <= 0;
                exp_pend[issue_i] <= 1'b0;
            end
            for (issue_i = 0; issue_i < N_LOG; issue_i = issue_i + 1) begin
                log_iss[issue_i] <= 0;
                log_pend[issue_i] <= 1'b0;
            end
        end
        else begin
            for (issue_i = 0; issue_i < N_EXP; issue_i = issue_i + 1) begin
                if (exp_pend[issue_i]) begin
                    if (exp_win_busy[issue_i]) begin
                        exp_pend[issue_i] <= 1'b0;
                        exp_iss[issue_i] <= exp_iss[issue_i] + 1;
                    end
                end
                else if (started && exp_need_ok[issue_i]
                         && !exp_win_busy[issue_i])
                    exp_pend[issue_i] <= 1'b1;
            end
            for (issue_i = 0; issue_i < N_LOG; issue_i = issue_i + 1) begin
                if (log_pend[issue_i]) begin
                    if (log_win_busy[issue_i]) begin
                        log_pend[issue_i] <= 1'b0;
                        log_iss[issue_i] <= log_iss[issue_i] + 1;
                    end
                end
                else if (started && log_need_ok[issue_i]
                         && !log_win_busy[issue_i])
                    log_pend[issue_i] <= 1'b1;
            end
        end
    end

    // ---------------- top-level busy ---------------------------------------
    reg done_seen;
    always @(posedge clk) begin
        if (reset) done_seen <= 1'b0;
        else if (start) done_seen <= 1'b0;
        else if (done) done_seen <= 1'b1;
    end
    assign busy = started && !done_seen && !done;

endmodule
