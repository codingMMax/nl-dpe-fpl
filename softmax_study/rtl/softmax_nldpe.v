// softmax_nldpe.v -- NL-DPE safe-softmax block: DPE(I|exp) per lane +
// one shared DPE(I|log), log-domain output (log_softmax_fusion).
//
// Spec: docs/superpowers/specs/2026-08-05-softmax-rtl-vtr-study-design.md §4.
// Plan: docs/superpowers/plans/2026-08-05-softmax-study-implementation.md.
//
// Workload: safe softmax over an S x S score matrix. W=16 lockstep lanes;
// lane k owns global rows {k + 16*i}. Per row:
//   A  (max)  : 16-wide comparator tree over the row               [CLB]
//   B  (exp)  : d = max(x-max, -128) -> N_EXP parallel DPE(I|exp)  [DPE]
//               drained bytes -> adder tree -> row sum             [CLB]
//   Cs (log)  : lq = min(sum >> log2(S), 127), 16 lane sums batched
//               through ONE shared DPE(I|log)                      [DPE]
//   D  (norm) : out = clamp(x - max - log_sum, -128, 127)          [CLB]
// Output is LOG-DOMAIN (log p_i); downstream mac_sv consumes it directly.
//
// DPE count = 16*N_EXP + 1. N_EXP = ceil(S/C) is the free-split point:
// every pass stays full, so splitting adds parallelism without adding
// passes (energy is charged per pass on the full crossbar C).
//
// Port-honest timing: the ACAM emits E values at once but they move
// through the 40-bit port at 5 elements/cycle. LCYC = OCYC = ceil(E/5),
// CCYC = 10. Time is charged on E (elements moved), energy on C (spec §4).
//
// Memory plan (all single-port): score_a (A, 128b words), sp[N_EXP]
// (B DPE-feed geometry, 40b words), score_d (D, 128b), out_m (128b).
// No exp storage: the row sum is consumed immediately; pass D recomputes
// x - max from its own score copy.
//
// Sim vs VTR: `SYNTHESIS -> bare `dpe` instantiation (black box,
// fc_verification/rtl/dpe_blackbox.v); otherwise parameterized behavior
// model (fc_verification/rtl/dpe_nldpe.v). Weights (identity) are forced
// hierarchically by the TB -- sim-only, no weight port exists.

`timescale 1ns / 1ps

module softmax_nldpe #(
    parameter S     = 128,
    parameter N_EXP = 1                 // exp DPEs per lane = ceil(S/C)
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output reg         done,
    input  wire        in_wen,
    input  wire [3:0]  in_lane,
    input  wire [13:0] in_addr,        // element index r*S + j within lane
    input  wire [7:0]  in_data,
    input  wire [3:0]  out_lane,
    input  wire [13:0] out_addr,
    output wire [7:0]  out_rdata
);
    localparam W     = 16;
    localparam RPL   = S / W;
    localparam WPR   = S / 16;              // 128b words per row
    localparam NWORD = RPL * WPR;
    localparam AW    = $clog2(NWORD);
    localparam E     = S / N_EXP;           // elements per exp DPE per row
    localparam LCYC  = (E + 4) / 5;         // DPE LOAD strobes per pass
    localparam OCYC  = LCYC;                // DPE OUTPUT words per pass
    localparam PWORD = RPL * LCYC;          // sp words per segment
    localparam PAW   = $clog2(PWORD);
    localparam LOG_S = $clog2(S);
    localparam LOG_E = $clog2(E);

    // ── run control ─────────────────────────────────────────────────────
    reg running;
    wire go = start && !running;

    // ── loader: byte-serial, strictly sequential per lane ──────────────
    // 128b staging for score_a / score_d
    reg [119:0] ld_stage;
    wire        ld_last_byte = in_wen && (in_addr[3:0] == 4'hF);
    wire [127:0] ld_word  = {in_data, ld_stage};
    wire [AW-1:0] ld_waddr = in_addr[13:4];
    always @(posedge clk) begin
        if (in_wen && in_addr[3:0] != 4'hF)
            ld_stage[in_addr[3:0]*8 +: 8] <= in_data;
    end
    // 40b staging for sp segments (DPE-feed geometry)
    reg  [2:0]  sp_pos;
    reg  [15:0] sp_wcnt;
    reg  [39:0] sp_stage;
    wire [LOG_E-1:0] jE = in_addr[LOG_E-1:0];      // byte within segment
    wire sp_boundary = (sp_pos == 3'd4) || (jE == E - 1);
    wire        sp_wen_g  = in_wen && sp_boundary;
    wire [39:0] sp_wdata  = sp_stage | ({32'b0, in_data} << (sp_pos * 8));
    wire [PAW-1:0] sp_waddr =
        in_addr[13:LOG_S] * LCYC + sp_wcnt[PAW-1:0];
    wire sp_seg = (N_EXP > 1) ? in_addr[LOG_S-1] : 1'b0;
    always @(posedge clk) begin
        if (reset) begin
            sp_pos <= 0; sp_wcnt <= 0; sp_stage <= 0;
        end else if (in_wen) begin
            if (sp_boundary) begin
                sp_pos   <= 0;
                sp_stage <= 0;
                sp_wcnt  <= (jE == E - 1) ? 16'd0 : sp_wcnt + 16'd1;
            end else begin
                sp_stage[sp_pos*8 +: 8] <= in_data;
                sp_pos <= sp_pos + 3'd1;
            end
        end
    end

    // ── stage issue FSMs (global control) ──────────────────────────────
    reg [7:0] a_row, a_wi;            // A: max
    reg [7:0] bs_row, bs_wi;          // B strobe side
    reg [7:0] c_row;                  // Cs: log
    reg [7:0] d_row, d_wi;            // D: normalize
    reg [7:0] a_done_c, b_done_c, c_done_c, d_done_c;

    // A: no gate.
    wire a_issue = running && (a_row < RPL);
    reg        a_v, a_last_d;
    reg [7:0]  a_row_d;

    // B strobes: need the row max.
    wire bs_issue = running && (bs_row < RPL) && (a_done_c > bs_row);
    reg        bs_v, bs_last_d;
    reg [7:0]  bs_row_d;

    // D: needs log_sum of the row.
    wire d_issue = running && (d_row < RPL) && (c_done_c > d_row);
    reg        d_v, d_last_d;
    reg [7:0]  d_row_d, d_wi_d;

    // exp drain (global word counter; passes chain back-to-back)
    reg [7:0] exp_dcnt;

    // Cs FSM
    localparam C_IDLE = 2'd0, C_STROBE = 2'd1, C_WAIT = 2'd2, C_DRAIN = 2'd3;
    reg [1:0] cst;
    reg [1:0] c_str;                  // strobe index 0..3
    reg [1:0] log_widx;               // drain word index

    wire exp_drain_v;                 // lane0/seg0 dpe_done (all in lockstep)
    wire log_dn;                      // log DPE dpe_done
    wire [39:0] log_dout;

    always @(posedge clk) begin
        if (reset || go) begin
            running <= go;
            done    <= 1'b0;
            a_row <= 0; a_wi <= 0; bs_row <= 0; bs_wi <= 0;
            c_row <= 0; d_row <= 0; d_wi <= 0;
            a_done_c <= 0; b_done_c <= 0; c_done_c <= 0; d_done_c <= 0;
            a_v <= 0; a_last_d <= 0; a_row_d <= 0;
            bs_v <= 0; bs_last_d <= 0; bs_row_d <= 0;
            d_v <= 0; d_last_d <= 0; d_row_d <= 0; d_wi_d <= 0;
            exp_dcnt <= 0;
            cst <= C_IDLE; c_str <= 0; log_widx <= 0;
        end else if (running) begin
            // A issue
            a_v      <= a_issue;
            a_row_d  <= a_row;
            a_last_d <= a_issue && (a_wi == WPR - 1);
            if (a_issue) begin
                if (a_wi == WPR - 1) begin a_wi <= 0; a_row <= a_row + 1; end
                else                 a_wi <= a_wi + 1;
            end
            if (a_last_d) a_done_c <= a_done_c + 1;

            // B strobe issue (sp read; DPE strobed on bs_v cycle)
            bs_v      <= bs_issue;
            bs_row_d  <= bs_row;
            bs_last_d <= bs_issue && (bs_wi == LCYC - 1);
            if (bs_issue) begin
                if (bs_wi == LCYC - 1) begin bs_wi <= 0; bs_row <= bs_row + 1; end
                else                   bs_wi <= bs_wi + 1;
            end

            // exp drain word counter (rows complete in drain order)
            if (exp_drain_v) begin
                if (exp_dcnt == OCYC - 1) begin
                    exp_dcnt <= 0;
                    b_done_c <= b_done_c + 1;
                end else
                    exp_dcnt <= exp_dcnt + 1;
            end

            // Cs FSM: 4 strobes -> wait -> 4 drain words
            case (cst)
                C_IDLE: begin
                    if ((c_row < RPL) && (b_done_c > c_row)) begin
                        cst <= C_STROBE;
                        c_str <= 0;
                    end
                end
                C_STROBE: begin
                    if (c_str == 2'd3) cst <= C_WAIT;
                    c_str <= c_str + 2'd1;
                end
                C_WAIT: begin
                    if (log_dn) begin       // word 0 captured this cycle
                        log_widx <= 2'd1;
                        cst <= C_DRAIN;
                    end
                end
                C_DRAIN: begin
                    if (log_widx == 2'd3) begin
                        c_done_c <= c_done_c + 1;
                        c_row    <= c_row + 1;
                        cst      <= C_IDLE;
                    end
                    log_widx <= log_widx + 2'd1;
                end
            endcase

            // D issue
            d_v      <= d_issue;
            d_row_d  <= d_row;
            d_wi_d   <= d_wi;
            d_last_d <= d_issue && (d_wi == WPR - 1);
            if (d_issue) begin
                if (d_wi == WPR - 1) begin d_wi <= 0; d_row <= d_row + 1; end
                else                 d_wi <= d_wi + 1;
            end
            if (d_last_d) d_done_c <= d_done_c + 1;

            if (d_done_c == RPL) done <= 1'b1;
        end
    end

    // Global addresses (lockstep lanes share them).
    wire [AW-1:0]  a_addr  = a_row * WPR + a_wi;
    wire [PAW-1:0] sp_raddr = bs_row * LCYC + bs_wi;
    wire [AW-1:0]  d_addr  = d_row * WPR + d_wi;

    // Log-DPE capture strobes (word index of the arriving drain word).
    wire       log_cap  = ((cst == C_WAIT) && log_dn) || (cst == C_DRAIN);
    wire [1:0] log_wsel = (cst == C_WAIT) ? 2'd0 : log_widx;

    // External read staging.
    reg [3:0] ext_lane_q, ext_boff_q;
    always @(posedge clk) begin
        ext_lane_q <= out_lane;
        ext_boff_q <= out_addr[3:0];
    end

    wire [127:0] om_q_a [0:W-1];
    wire [7:0]   lq_a   [0:W-1];      // per-lane quantized sums -> log DPE
    wire         exp_dn_a [0:W*N_EXP-1];

    // ── lanes ───────────────────────────────────────────────────────────
    genvar gk, ge, gj;
    generate
        for (gk = 0; gk < W; gk = gk + 1) begin : lane
            // memories
            reg [127:0] score_a [0:NWORD-1];
            reg [127:0] score_d [0:NWORD-1];
            reg [127:0] out_m   [0:NWORD-1];

            wire ld_w = ld_last_byte && (in_lane == gk) && !running;

            reg  [127:0] sa_q;
            wire [AW-1:0] sa_addr = running ? a_addr : ld_waddr;
            always @(posedge clk) begin
                if (ld_w) score_a[sa_addr] <= ld_word;
                sa_q <= score_a[sa_addr];
            end
            reg  [127:0] sd_q;
            wire [AW-1:0] sd_addr = running ? d_addr : ld_waddr;
            always @(posedge clk) begin
                if (ld_w) score_d[sd_addr] <= ld_word;
                sd_q <= score_d[sd_addr];
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

            // Per-row history registers: flat vectors, NOT reg arrays --
            // async-indexed reg arrays get inferred as RAM primitives by
            // Parmys; flat vectors synthesize to the intended FFs + muxes.
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

            // ── stage B: subtract-clamp -> N_EXP exp DPEs -> sum tree ──
            wire signed [7:0] bmax = max_hist[bs_row_d*8 +: 8];
            wire [40*N_EXP-1:0] exp_din_flat;
            wire [40*N_EXP-1:0] exp_dout_flat;

            for (ge = 0; ge < N_EXP; ge = ge + 1) begin : edpe
                // sp segment memory (DPE-feed geometry, 40b words)
                reg [39:0] sp_mem [0:PWORD-1];
                reg [39:0] sp_q;
                wire sp_w = sp_wen_g && (in_lane == gk) &&
                            (sp_seg == ge[0]) && !running;
                wire [PAW-1:0] spa = running ? sp_raddr : sp_waddr;
                always @(posedge clk) begin
                    if (sp_w) sp_mem[spa] <= sp_wdata;
                    sp_q <= sp_mem[spa];
                end

                // subtract-clamp 5 bytes -> DPE data_in
                for (gj = 0; gj < 5; gj = gj + 1) begin : bsub
                    wire signed [8:0] du =
                        $signed(sp_q[gj*8 +: 8]) - bmax;      // in [-255, 0]
                    assign exp_din_flat[ge*40 + gj*8 +: 8] =
                        (du < -9'sd128) ? 8'h80 : du[7:0];
                end

                wire nc0, nc1, nc2, nc3;
`ifdef SYNTHESIS
                dpe u_exp (
                    .clk(clk), .reset(reset),
                    .data_in(exp_din_flat[ge*40 +: 40]),
                    .nl_dpe_control(2'b00), .shift_add_control(1'b0),
                    .w_buf_en(bs_v), .shift_add_bypass(1'b0),
                    .load_output_reg(1'b0), .load_input_reg(1'b0),
                    .MSB_SA_Ready(nc0),
                    .data_out(exp_dout_flat[ge*40 +: 40]),
                    .dpe_done(exp_dn_a[gk*N_EXP + ge]),
                    .reg_full(nc1), .shift_add_done(nc2),
                    .shift_add_bypass_ctrl(nc3)
                );
`else
                dpe #(.KERNEL_WIDTH(E), .NUM_COLS(E), .DPE_BUF_WIDTH(40),
                      .COMPUTE_CYCLES(10), .ACAM_MODE(1)) u_exp (
                    .clk(clk), .reset(reset),
                    .data_in(exp_din_flat[ge*40 +: 40]),
                    .nl_dpe_control(2'b00), .shift_add_control(1'b0),
                    .w_buf_en(bs_v), .shift_add_bypass(1'b0),
                    .load_output_reg(1'b0), .load_input_reg(1'b0),
                    .MSB_SA_Ready(nc0),
                    .data_out(exp_dout_flat[ge*40 +: 40]),
                    .dpe_done(exp_dn_a[gk*N_EXP + ge]),
                    .reg_full(nc1), .shift_add_done(nc2),
                    .shift_add_bypass_ctrl(nc3)
                );
`endif
            end

            // drained bytes -> adder tree -> row sum accumulator
            wire [12:0] ts;
            if (N_EXP == 1) begin : t1
                assign ts =
                    ({5'b0, exp_dout_flat[7:0]}   + {5'b0, exp_dout_flat[15:8]}) +
                    ({5'b0, exp_dout_flat[23:16]} + {5'b0, exp_dout_flat[31:24]}) +
                    {5'b0, exp_dout_flat[39:32]};
            end else begin : t2
                assign ts =
                    (({5'b0, exp_dout_flat[7:0]}   + {5'b0, exp_dout_flat[15:8]}) +
                     ({5'b0, exp_dout_flat[23:16]} + {5'b0, exp_dout_flat[31:24]}) +
                      {5'b0, exp_dout_flat[39:32]}) +
                    (({5'b0, exp_dout_flat[47:40]} + {5'b0, exp_dout_flat[55:48]}) +
                     ({5'b0, exp_dout_flat[63:56]} + {5'b0, exp_dout_flat[71:64]}) +
                      {5'b0, exp_dout_flat[79:72]});
            end

            reg [15:0] acc;
            reg [16*RPL-1:0] sum_hist;
            always @(posedge clk) begin
                if (go) acc <= 0;
                else if (exp_drain_v) begin
                    if (exp_dcnt == OCYC - 1) begin
                        sum_hist[b_done_c*16 +: 16] <= acc + {3'b0, ts};
                        acc <= 0;
                    end else
                        acc <= acc + {3'b0, ts};
                end
            end

            // ── stage Cs (lane side): quantize sum, capture log_sum ──
            wire [15:0] csum = sum_hist[c_row*16 +: 16];
            wire [15:0] csh  = csum >> LOG_S;
            assign lq_a[gk] = (|csh[15:7]) ? 8'd127 : {1'b0, csh[6:0]};

            reg [8*RPL-1:0] ls_hist;
            always @(posedge clk) begin
                if (log_cap && (log_wsel == gk / 5))
                    ls_hist[c_row*8 +: 8] <= log_dout[(gk % 5)*8 +: 8];
            end

            // ── stage D: out = clamp(x - max - log_sum) ──
            wire signed [7:0] dmax = max_hist[d_row_d*8 +: 8];
            wire signed [7:0] dls  = ls_hist[d_row_d*8 +: 8];
            wire [127:0] p_word;
            for (gj = 0; gj < 16; gj = gj + 1) begin : dsub
                wire signed [9:0] t =
                    {{2{sd_q[gj*8+7]}}, sd_q[gj*8 +: 8]}
                    - {{2{dmax[7]}}, dmax}
                    - {{2{dls[7]}},  dls};
                assign p_word[gj*8 +: 8] =
                    (t < -10'sd128) ? 8'h80 :
                    (t >  10'sd127) ? 8'h7F : t[7:0];
            end

            reg [127:0] om_q;
            wire [AW-1:0] d_waddr = d_row_d * WPR + d_wi_d;
            wire [AW-1:0] om_addr = done ? out_addr[13:4] : d_waddr;
            always @(posedge clk) begin
                if (d_v && !done) out_m[om_addr] <= p_word;
                om_q <= out_m[om_addr];
            end
            assign om_q_a[gk] = om_q;
        end
    endgenerate

    assign exp_drain_v = exp_dn_a[0];

    // ── shared log DPE: 16 lane sums per pass, ACAM_MODE=2 ─────────────
    wire log_wen = (cst == C_STROBE);
    wire [39:0] log_din =
        (c_str == 2'd0) ? {lq_a[4],  lq_a[3],  lq_a[2],  lq_a[1],  lq_a[0]}  :
        (c_str == 2'd1) ? {lq_a[9],  lq_a[8],  lq_a[7],  lq_a[6],  lq_a[5]}  :
        (c_str == 2'd2) ? {lq_a[14], lq_a[13], lq_a[12], lq_a[11], lq_a[10]} :
                          {32'b0, lq_a[15]};
    wire lnc0, lnc1, lnc2, lnc3;
`ifdef SYNTHESIS
    dpe u_log (
        .clk(clk), .reset(reset), .data_in(log_din),
        .nl_dpe_control(2'b00), .shift_add_control(1'b0),
        .w_buf_en(log_wen), .shift_add_bypass(1'b0),
        .load_output_reg(1'b0), .load_input_reg(1'b0),
        .MSB_SA_Ready(lnc0), .data_out(log_dout), .dpe_done(log_dn),
        .reg_full(lnc1), .shift_add_done(lnc2), .shift_add_bypass_ctrl(lnc3)
    );
`else
    dpe #(.KERNEL_WIDTH(16), .NUM_COLS(16), .DPE_BUF_WIDTH(40),
          .COMPUTE_CYCLES(10), .ACAM_MODE(2)) u_log (
        .clk(clk), .reset(reset), .data_in(log_din),
        .nl_dpe_control(2'b00), .shift_add_control(1'b0),
        .w_buf_en(log_wen), .shift_add_bypass(1'b0),
        .load_output_reg(1'b0), .load_input_reg(1'b0),
        .MSB_SA_Ready(lnc0), .data_out(log_dout), .dpe_done(log_dn),
        .reg_full(lnc1), .shift_add_done(lnc2), .shift_add_bypass_ctrl(lnc3)
    );
`endif

    wire [127:0] om_sel = om_q_a[ext_lane_q];
    assign out_rdata = om_sel[ext_boff_q*8 +: 8];

    function signed [7:0] smax2;
        input signed [7:0] x;
        input signed [7:0] y;
        begin
            smax2 = (x > y) ? x : y;
        end
    endfunction

endmodule
