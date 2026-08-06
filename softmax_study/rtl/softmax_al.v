// softmax_al.v -- Azure-Lily safe-softmax block: pure FPGA (CLB + DSP + BRAM).
//
// Spec: docs/superpowers/specs/2026-08-05-softmax-rtl-vtr-study-design.md §3.
// Plan: docs/superpowers/plans/2026-08-05-softmax-study-implementation.md.
//
// Workload: safe softmax over an S x S score matrix. W=16 lockstep lanes;
// lane k owns global rows {k + 16*i}. Per row, 3 passes:
//   A  (max)  : 16-wide comparator tree over the row            [CLB]
//   B  (exp)  : u = max-x -> exp_lut ROM -> adder tree -> sum   [CLB]
//   Cs (recip): idx = sum >> log2(S) -> recip_lut ROM           [CLB]
//   D  (norm) : p = min(255, (e * rec) >> log2(S))              [16x mac_int_9x9]
// Pipelined across rows: stage X processes row r+1 while X+1 processes row r.
// Output: linear-domain probabilities (uint8).
//
// Memory plan (all single-port, Parmys-inference style of fc_top_synth.v):
//   score_a, score_b : two copies of the lane's score rows (one per reader)
//   exp_bank0/1      : one row of exp values, double-buffered by row parity
//   out_m            : output rows (written by D, read externally after done)
//
// External interface: byte-serial load pre-start; byte read post-done.
// Cycle count = posedges from start acceptance to done.

`timescale 1ns / 1ps

module softmax_al #(
    parameter S = 128
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output reg         done,
    input  wire        in_wen,
    input  wire [3:0]  in_lane,
    input  wire [13:0] in_addr,     // element index r*S + j within lane
    input  wire [7:0]  in_data,
    input  wire [3:0]  out_lane,
    input  wire [13:0] out_addr,    // element index r*S + j within lane
    output wire [7:0]  out_rdata
);
    localparam W     = 16;             // lanes
    localparam RPL   = S / W;          // rows per lane
    localparam WPR   = S / 16;         // 128-bit words per row
    localparam NWORD = RPL * WPR;      // words per score copy per lane
    localparam AW    = $clog2(NWORD);
    localparam EW    = (WPR > 1) ? $clog2(WPR) : 1;
    localparam R_SHIFT = $clog2(S);

    `include "softmax_luts.vh"

    // ── run control ─────────────────────────────────────────────────────
    reg running;
    wire go = start && !running;

    // ── loader: byte-serial -> 128b words, written to both score copies ─
    reg [119:0] ld_stage;
    wire        ld_last_byte = in_wen && (in_addr[3:0] == 4'hF);
    wire [127:0] ld_word  = {in_data, ld_stage};
    wire [AW-1:0] ld_waddr = in_addr[13:4];
    always @(posedge clk) begin
        if (in_wen && in_addr[3:0] != 4'hF)
            ld_stage[in_addr[3:0]*8 +: 8] <= in_data;
    end

    // ── stage issue FSMs (global control; lanes are pure datapath) ─────
    // Streaming pattern: issue 1 word/cycle while the row gate holds;
    // consume side runs on 1-2 cycle delayed pipes.
    reg [7:0] a_row, a_wi, b_row, b_wi, c_row, d_row, d_wi;
    reg [7:0] a_done_c, b_done_c, c_done_c, d_done_c;

    // A: no gate.
    wire a_issue = running && (a_row < RPL);
    reg        a_v, a_last_d;
    reg [7:0]  a_row_d, a_wi_d;

    // B: needs A done on this row, and D done on row-2 (exp-bank parity safety).
    wire b_gate  = (a_done_c > b_row) && (d_done_c + 8'd1 >= b_row);
    wire b_issue = running && (b_row < RPL) && b_gate;
    reg        b_v, b_last_d;
    reg [7:0]  b_row_d, b_wi_d;

    // Cs: single cycle per row.
    wire c_fire = running && (c_row < RPL) && (b_done_c > c_row);

    // D: needs Cs done on this row.
    wire d_gate  = (c_done_c > d_row);
    wire d_issue = running && (d_row < RPL) && d_gate;
    reg        d_v, d_last_d, d_v2, d_last_d2;
    reg [7:0]  d_row_d, d_wi_d, d_row_d2, d_wi_d2;

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
            // A issue
            a_v      <= a_issue;
            a_row_d  <= a_row;
            a_wi_d   <= a_wi;
            a_last_d <= a_issue && (a_wi == WPR - 1);
            if (a_issue) begin
                if (a_wi == WPR - 1) begin a_wi <= 0; a_row <= a_row + 1; end
                else                 a_wi <= a_wi + 1;
            end
            if (a_last_d) a_done_c <= a_done_c + 1;

            // B issue
            b_v      <= b_issue;
            b_row_d  <= b_row;
            b_wi_d   <= b_wi;
            b_last_d <= b_issue && (b_wi == WPR - 1);
            if (b_issue) begin
                if (b_wi == WPR - 1) begin b_wi <= 0; b_row <= b_row + 1; end
                else                 b_wi <= b_wi + 1;
            end
            if (b_last_d) b_done_c <= b_done_c + 1;

            // Cs
            if (c_fire) begin
                c_row    <= c_row + 1;
                c_done_c <= c_done_c + 1;
            end

            // D issue (+2-deep pipe: BRAM read, then DSP register)
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

    // Global addresses shared by all lanes (lockstep).
    wire [AW-1:0] a_addr = a_row * WPR + a_wi;
    wire [AW-1:0] b_addr = b_row * WPR + b_wi;

    // External read staging (registered lane/byte select).
    reg [3:0] ext_lane_q, ext_boff_q;
    always @(posedge clk) begin
        ext_lane_q <= out_lane;
        ext_boff_q <= out_addr[3:0];
    end

    wire [127:0] om_q_a [0:W-1];

    // ── lanes ───────────────────────────────────────────────────────────
    genvar gk, gj;
    generate
        for (gk = 0; gk < W; gk = gk + 1) begin : lane
            // memories
            reg [127:0] score_a  [0:NWORD-1];
            reg [127:0] score_b  [0:NWORD-1];
            reg [127:0] exp_bank0 [0:WPR-1];
            reg [127:0] exp_bank1 [0:WPR-1];
            reg [127:0] out_m    [0:NWORD-1];

            wire ld_w = ld_last_byte && (in_lane == gk) && !running;

            // score copy A: loader write / stage-A read
            reg  [127:0] sa_q;
            wire [AW-1:0] sa_addr = running ? a_addr : ld_waddr;
            always @(posedge clk) begin
                if (ld_w) score_a[sa_addr] <= ld_word;
                sa_q <= score_a[sa_addr];
            end
            // score copy B: loader write / stage-B read
            reg  [127:0] sb_q;
            wire [AW-1:0] sb_addr = running ? b_addr : ld_waddr;
            always @(posedge clk) begin
                if (ld_w) score_b[sb_addr] <= ld_word;
                sb_q <= score_b[sb_addr];
            end

            // ── stage A: 16-input signed max tree ──
            wire signed [7:0] m2_0 = smax2(sa_q[7:0],    sa_q[15:8]);
            wire signed [7:0] m2_1 = smax2(sa_q[23:16],  sa_q[31:24]);
            wire signed [7:0] m2_2 = smax2(sa_q[39:32],  sa_q[47:40]);
            wire signed [7:0] m2_3 = smax2(sa_q[55:48],  sa_q[63:56]);
            wire signed [7:0] m2_4 = smax2(sa_q[71:64],  sa_q[79:72]);
            wire signed [7:0] m2_5 = smax2(sa_q[87:80],  sa_q[95:88]);
            wire signed [7:0] m2_6 = smax2(sa_q[103:96], sa_q[111:104]);
            wire signed [7:0] m2_7 = smax2(sa_q[119:112],sa_q[127:120]);
            wire signed [7:0] m4_0 = smax2(m2_0, m2_1);
            wire signed [7:0] m4_1 = smax2(m2_2, m2_3);
            wire signed [7:0] m4_2 = smax2(m2_4, m2_5);
            wire signed [7:0] m4_3 = smax2(m2_6, m2_7);
            wire signed [7:0] max16 = smax2(smax2(m4_0, m4_1), smax2(m4_2, m4_3));

            // Per-row history registers. Flat vectors with part-select
            // indexing, NOT reg arrays: async-indexed reg arrays get
            // inferred as RAM primitives by Parmys (measured: 10240
            // single_port_ram subckts / 1495 BRAMs); flat vectors
            // synthesize to the intended FFs + muxes.
            reg signed [7:0] run_max;
            reg [8*RPL-1:0] max_hist;
            wire signed [7:0] max_now = smax2(run_max, max16);
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
            wire [127:0] e_word;
            wire [11:0]  tsum;
            for (gj = 0; gj < 16; gj = gj + 1) begin : bsub
                wire signed [8:0] du = bmax - $signed(sb_q[gj*8 +: 8]);
                assign e_word[gj*8 +: 8] = exp_lut(du[7:0]);
            end
            assign tsum =
                ({4'b0, e_word[7:0]}     + {4'b0, e_word[15:8]})    +
                ({4'b0, e_word[23:16]}   + {4'b0, e_word[31:24]})   +
                ({4'b0, e_word[39:32]}   + {4'b0, e_word[47:40]})   +
                ({4'b0, e_word[55:48]}   + {4'b0, e_word[63:56]})   +
                ({4'b0, e_word[71:64]}   + {4'b0, e_word[79:72]})   +
                ({4'b0, e_word[87:80]}   + {4'b0, e_word[95:88]})   +
                ({4'b0, e_word[103:96]}  + {4'b0, e_word[111:104]}) +
                ({4'b0, e_word[119:112]} + {4'b0, e_word[127:120]});

            reg [15:0] acc;
            reg [16*RPL-1:0] sum_hist;
            always @(posedge clk) begin
                if (go) acc <= 0;
                else if (b_v) begin
                    if (b_last_d) begin
                        sum_hist[b_row_d*16 +: 16] <= acc + {4'b0, tsum};
                        acc <= 0;
                    end else
                        acc <= acc + {4'b0, tsum};
                end
            end

            // exp banks: B writes bank (b_row_d%2); D reads bank (d_row%2).
            // The d_done >= b_row-1 gate guarantees no same-bank conflict.
            reg [127:0] eb0_q, eb1_q;
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

            // ── stage D: 16x mac_int_9x9 normalize ──
            wire [127:0] eq   = d_row_d[0] ? eb1_q : eb0_q;
            wire [7:0]   drec = rec_hist[d_row_d*8 +: 8];
            wire [127:0] p_word;
            for (gj = 0; gj < 16; gj = gj + 1) begin : dmac
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

            // out memory: D writes during run, external reads after done.
            reg [127:0] om_q;
            wire [AW-1:0] d_waddr2 = d_row_d2 * WPR + d_wi_d2;
            wire [AW-1:0] om_addr  = done ? out_addr[13:4] : d_waddr2;
            always @(posedge clk) begin
                if (d_v2 && !done) out_m[om_addr] <= p_word;
                om_q <= out_m[om_addr];
            end
            assign om_q_a[gk] = om_q;
        end
    endgenerate

    wire [127:0] om_sel = om_q_a[ext_lane_q];
    assign out_rdata = om_sel[ext_boff_q*8 +: 8];

    // 2-input signed max helper
    function signed [7:0] smax2;
        input signed [7:0] x;
        input signed [7:0] y;
        begin
            smax2 = (x > y) ? x : y;
        end
    endfunction

endmodule
