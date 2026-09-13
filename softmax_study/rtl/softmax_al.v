// softmax_al.v -- Azure-Lily safe-softmax block: pure FPGA (CLB + DSP + BRAM).
//
// Spec: docs/superpowers/specs/2026-08-05-softmax-rtl-vtr-study-design.md §3.
// Plan: docs/superpowers/plans/2026-08-05-softmax-study-implementation.md.
//
// Workload: safe softmax over an S x S score matrix. W=16 lockstep lanes;
// lane k owns global rows {k + 16*i}. Per row, 3 passes:
//   A  (max)  : E-wide comparator tree over the row                [CLB]
//   B  (exp)  : u = max-x -> exp_lut ROM -> adder tree -> sum       [CLB]
//   Cs (recip): idx = sum >> log2(S) -> recip_lut ROM              [CLB]
//   D  (norm) : p = min(255, (e * rec) >> log2(S))                 [E x mac_int_9x9]
// Pipelined across rows: stage X processes row r+1 while X+1 processes row r.
// Output: linear-domain probabilities (uint8).
//
// E = datapath width (elements/cycle/lane). Two configurations are studied:
//   E = 16 : unconstrained AL -- as much operand bandwidth as the fabric will
//            give (128-bit words = 4 physical 512x40 BRAMs per memory).
//   E =  5 : supply-matched AL -- the same 40 bit/cycle operand feed a single
//            NL-DPE port provides, so the comparison isolates the exp
//            mechanism (LUT ROM vs ACAM) instead of interface width.
// S need not be divisible by E: the last word of each row carries
// TAIL = S - (WPR-1)*E valid elements and the rest are masked (-128 for the
// max tree, 0 for the sum tree).
//
// Memory plan (all single-port, Parmys-inference style of fc_top_synth.v):
//   score_a, score_b : two copies of the lane's score rows (one per reader)
//   exp_bank0/1      : one row of exp values, double-buffered by row parity
//   out_m            : output rows (written by D, read externally after done)
//
// External interface: byte-serial load pre-start (word address + byte offset
// supplied by the driver, so no divider is needed for non-power-of-2 E);
// byte read post-done. Cycle count = posedges from start acceptance to done.

`timescale 1ns / 1ps

module softmax_al #(
    parameter S = 128,
    parameter E = 16
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output reg         done,
    input  wire        in_wen,
    input  wire [3:0]  in_lane,
    input  wire [13:0] in_waddr,    // word index within lane
    input  wire [3:0]  in_boff,     // byte offset within word (0..E-1)
    input  wire [7:0]  in_data,
    input  wire [3:0]  out_lane,
    input  wire [13:0] out_waddr,
    input  wire [3:0]  out_boff,
    output wire [7:0]  out_rdata
);
    localparam W     = 16;                 // lanes
    localparam RPL   = S / W;              // rows per lane
    localparam WPR   = (S + E - 1) / E;    // words per row
    localparam NWORD = RPL * WPR;          // words per score copy per lane
    localparam AW    = $clog2(NWORD);
    localparam EW    = (WPR > 1) ? $clog2(WPR) : 1;
    localparam DW    = 8 * E;              // datapath / memory word width
    localparam SW    = 16 * E;             // sum-tree node vector width
    localparam TAIL  = S - (WPR - 1) * E;  // valid elements in last word
    localparam LV    = $clog2(E);          // reduction tree levels
    localparam R_SHIFT = $clog2(S);

    `include "softmax_luts.vh"

    // ── run control ─────────────────────────────────────────────────────
    reg running;
    wire go = start && !running;

    // ── loader: byte-serial; each word is written progressively, so the
    //    final write to a word address carries the complete word ─────────
    reg [DW-1:0] ld_stage;
    wire [DW-1:0] ld_byte = {{(DW-8){1'b0}}, in_data} << (in_boff * 8);
    wire [DW-1:0] ld_word = (in_boff == 4'd0) ? ld_byte : (ld_stage | ld_byte);
    wire [AW-1:0] ld_waddr = in_waddr[AW-1:0];
    always @(posedge clk) begin
        if (in_wen) ld_stage <= ld_word;
    end

    // ── stage issue FSMs (global control; lanes are pure datapath) ─────
    reg [15:0] a_row, a_wi, b_row, b_wi, c_row, d_row, d_wi;
    reg [15:0] a_done_c, b_done_c, c_done_c, d_done_c;

    wire a_issue = running && (a_row < RPL);
    reg        a_v, a_last_d;
    reg [15:0] a_row_d, a_wi_d;

    wire b_gate  = (a_done_c > b_row) && (d_done_c + 16'd1 >= b_row);
    wire b_issue = running && (b_row < RPL) && b_gate;
    reg        b_v, b_last_d;
    reg [15:0] b_row_d, b_wi_d;

    wire c_fire = running && (c_row < RPL) && (b_done_c > c_row);

    wire d_gate  = (c_done_c > d_row);
    wire d_issue = running && (d_row < RPL) && d_gate;
    reg        d_v, d_last_d, d_v2, d_last_d2;
    reg [15:0] d_row_d, d_wi_d, d_row_d2, d_wi_d2;

    always @(posedge clk) begin
        if (reset || go) begin
            running  <= go;
            done     <= 1'b0;
            a_row <= 0; a_wi <= 0; b_row <= 0; b_wi <= 0;
            c_row <= 0; d_row <= 0; d_wi <= 0;
            a_done_c <= 0; b_done_c <= 0; c_done_c <= 0; d_done_c <= 0;
            a_v <= 0; a_last_d <= 0; b_v <= 0; b_last_d <= 0;
            d_v <= 0; d_last_d <= 0; d_v2 <= 0; d_last_d2 <= 0;
            a_row_d <= 0; a_wi_d <= 0; b_row_d <= 0; b_wi_d <= 0;
            d_row_d <= 0; d_wi_d <= 0; d_row_d2 <= 0; d_wi_d2 <= 0;
        end else if (running) begin
            a_v      <= a_issue;
            a_row_d  <= a_row;
            a_wi_d   <= a_wi;
            a_last_d <= a_issue && (a_wi == WPR - 1);
            if (a_issue) begin
                if (a_wi == WPR - 1) begin a_wi <= 0; a_row <= a_row + 1; end
                else                 a_wi <= a_wi + 1;
            end
            if (a_last_d) a_done_c <= a_done_c + 1;

            b_v      <= b_issue;
            b_row_d  <= b_row;
            b_wi_d   <= b_wi;
            b_last_d <= b_issue && (b_wi == WPR - 1);
            if (b_issue) begin
                if (b_wi == WPR - 1) begin b_wi <= 0; b_row <= b_row + 1; end
                else                 b_wi <= b_wi + 1;
            end
            if (b_last_d) b_done_c <= b_done_c + 1;

            if (c_fire) begin
                c_row    <= c_row + 1;
                c_done_c <= c_done_c + 1;
            end

            d_v      <= d_issue;
            d_row_d  <= d_row;
            d_wi_d   <= d_wi;
            d_last_d <= d_issue && (d_wi == WPR - 1);
            if (d_issue) begin
                if (d_wi == WPR - 1) begin d_wi <= 0; d_row <= d_row + 1; end
                else                 d_wi <= d_wi + 1;
            end
            d_v2      <= d_v;
            d_row_d2  <= d_row_d;
            d_wi_d2   <= d_wi_d;
            d_last_d2 <= d_last_d;
            if (d_last_d2) d_done_c <= d_done_c + 1;

            if (d_done_c == RPL) done <= 1'b1;
        end
    end

    wire [AW-1:0] a_addr = a_row * WPR + a_wi;
    wire [AW-1:0] b_addr = b_row * WPR + b_wi;

    // last-word masks (only relevant when S % E != 0)
    wire a_tail = (a_wi_d == WPR - 1);
    wire b_tail = (b_wi_d == WPR - 1);

    reg [3:0] ext_lane_q, ext_boff_q;
    always @(posedge clk) begin
        ext_lane_q <= out_lane;
        ext_boff_q <= out_boff;
    end

    wire [DW-1:0] om_q_a [0:W-1];

    genvar gk, gj, lv, ti;
    generate
        for (gk = 0; gk < W; gk = gk + 1) begin : lane
            reg [DW-1:0] score_a  [0:NWORD-1];
            reg [DW-1:0] score_b  [0:NWORD-1];
            reg [DW-1:0] exp_bank0 [0:WPR-1];
            reg [DW-1:0] exp_bank1 [0:WPR-1];
            reg [DW-1:0] out_m    [0:NWORD-1];

            wire ld_w = in_wen && (in_lane == gk) && !running;

            reg  [DW-1:0] sa_q;
            wire [AW-1:0] sa_addr = running ? a_addr : ld_waddr;
            always @(posedge clk) begin
                if (ld_w) score_a[sa_addr] <= ld_word;
                sa_q <= score_a[sa_addr];
            end
            reg  [DW-1:0] sb_q;
            wire [AW-1:0] sb_addr = running ? b_addr : ld_waddr;
            always @(posedge clk) begin
                if (ld_w) score_b[sb_addr] <= ld_word;
                sb_q <= score_b[sb_addr];
            end

            // ── stage A: E-input balanced signed max tree ──
            // invalid lanes of a ragged last word are forced to -128.
            wire [DW-1:0] a_masked;
            for (ti = 0; ti < E; ti = ti + 1) begin : amask
                assign a_masked[ti*8 +: 8] =
                    (a_tail && (ti >= TAIL)) ? 8'h80 : sa_q[ti*8 +: 8];
            end

            wire [DW-1:0] mtree [0:LV];
            assign mtree[0] = a_masked;
            for (lv = 0; lv < LV; lv = lv + 1) begin : mlev
                localparam CNT = (E + (1 << lv) - 1) >> lv;
                localparam NXT = (CNT + 1) >> 1;
                for (ti = 0; ti < NXT; ti = ti + 1) begin : mnode
                    if (2*ti + 1 < CNT) begin : pair
                        assign mtree[lv+1][ti*8 +: 8] =
                            ($signed(mtree[lv][(2*ti)*8 +: 8]) >
                             $signed(mtree[lv][(2*ti+1)*8 +: 8]))
                            ? mtree[lv][(2*ti)*8 +: 8]
                            : mtree[lv][(2*ti+1)*8 +: 8];
                    end else begin : pass
                        assign mtree[lv+1][ti*8 +: 8] = mtree[lv][(2*ti)*8 +: 8];
                    end
                end
            end
            wire signed [7:0] max_word = mtree[LV][7:0];

            reg signed [7:0] run_max;
            // Flat vector, NOT a reg array: async-indexed reg arrays get
            // inferred as RAM primitives by Parmys.
            reg [8*RPL-1:0] max_hist;
            wire signed [7:0] max_now =
                (run_max > max_word) ? run_max : max_word;
            always @(posedge clk) begin
                if (go) run_max <= -8'sd128;
                else if (a_v) begin
                    if (a_last_d) begin
                        max_hist[a_row_d*8 +: 8] <= max_now;
                        run_max <= -8'sd128;
                    end else
                        run_max <= max_now;
                end
            end

            // ── stage B: subtract, exp ROM, adder tree, accumulate ──
            wire signed [7:0] bmax = max_hist[b_row_d*8 +: 8];
            wire [DW-1:0] e_word;
            for (gj = 0; gj < E; gj = gj + 1) begin : bsub
                wire signed [8:0] du = bmax - $signed(sb_q[gj*8 +: 8]);
                assign e_word[gj*8 +: 8] =
                    (b_tail && (gj >= TAIL)) ? 8'd0 : exp_lut(du[7:0]);
            end

            wire [SW-1:0] stree [0:LV];
            for (ti = 0; ti < E; ti = ti + 1) begin : sinit
                assign stree[0][ti*16 +: 16] = {8'b0, e_word[ti*8 +: 8]};
            end
            for (lv = 0; lv < LV; lv = lv + 1) begin : slev
                localparam CNT = (E + (1 << lv) - 1) >> lv;
                localparam NXT = (CNT + 1) >> 1;
                for (ti = 0; ti < NXT; ti = ti + 1) begin : snode
                    if (2*ti + 1 < CNT) begin : pair
                        assign stree[lv+1][ti*16 +: 16] =
                            stree[lv][(2*ti)*16 +: 16] +
                            stree[lv][(2*ti+1)*16 +: 16];
                    end else begin : pass
                        assign stree[lv+1][ti*16 +: 16] =
                            stree[lv][(2*ti)*16 +: 16];
                    end
                end
            end
            wire [15:0] tsum = stree[LV][15:0];

            reg [15:0] acc;
            reg [16*RPL-1:0] sum_hist;
            always @(posedge clk) begin
                if (go) acc <= 0;
                else if (b_v) begin
                    if (b_last_d) begin
                        sum_hist[b_row_d*16 +: 16] <= acc + tsum;
                        acc <= 0;
                    end else
                        acc <= acc + tsum;
                end
            end

            // exp banks: B writes bank (b_row_d%2); D reads bank (d_row%2).
            reg [DW-1:0] eb0_q, eb1_q;
            wire eb0_we = b_v && (b_row_d[0] == 1'b0);
            wire eb1_we = b_v && (b_row_d[0] == 1'b1);
            wire [EW-1:0] eb0_addr = eb0_we ? b_wi_d[EW-1:0] : d_wi[EW-1:0];
            wire [EW-1:0] eb1_addr = eb1_we ? b_wi_d[EW-1:0] : d_wi[EW-1:0];
            always @(posedge clk) begin
                if (eb0_we) exp_bank0[eb0_addr] <= e_word;
                eb0_q <= exp_bank0[eb0_addr];
            end
            always @(posedge clk) begin
                if (eb1_we) exp_bank1[eb1_addr] <= e_word;
                eb1_q <= exp_bank1[eb1_addr];
            end

            // ── stage Cs: reciprocal ROM ──
            wire [15:0] csum   = sum_hist[c_row*16 +: 16];
            wire [15:0] cshift = csum >> R_SHIFT;
            wire [7:0]  cidx   = (|cshift[15:8]) ? 8'd255 : cshift[7:0];
            reg  [8*RPL-1:0] rec_hist;
            always @(posedge clk) begin
                if (c_fire) rec_hist[c_row*8 +: 8] <= recip_lut(cidx);
            end

            // ── stage D: E x mac_int_9x9 normalize ──
            wire [DW-1:0] eq   = d_row_d[0] ? eb1_q : eb0_q;
            wire [7:0]    drec = rec_hist[d_row_d*8 +: 8];
            wire [DW-1:0] p_word;
            for (gj = 0; gj < E; gj = gj + 1) begin : dmac
                wire [17:0] mo;
                mac_int_9x9 u_mac (
                    .reset(reset),
                    .a({1'b0, eq[gj*8 +: 8]}),
                    .b({1'b0, drec}),
                    .out(mo),
                    .clk(clk)
                );
                wire [17:0] sh = mo >> R_SHIFT;
                assign p_word[gj*8 +: 8] = (|sh[17:8]) ? 8'd255 : sh[7:0];
            end

            reg [DW-1:0] om_q;
            wire [AW-1:0] d_waddr2 = d_row_d2 * WPR + d_wi_d2;
            wire [AW-1:0] om_addr  = done ? out_waddr[AW-1:0] : d_waddr2;
            always @(posedge clk) begin
                if (d_v2 && !done) out_m[om_addr] <= p_word;
                om_q <= out_m[om_addr];
            end
            assign om_q_a[gk] = om_q;
        end
    endgenerate

    wire [DW-1:0] om_sel = om_q_a[ext_lane_q];
    assign out_rdata = om_sel[ext_boff_q*8 +: 8];

endmodule
