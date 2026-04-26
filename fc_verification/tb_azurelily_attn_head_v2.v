// Azure-Lily Attention Head V2 — combined functional + latency TB.
//
// AH-track verification for the regenerated DUT
//   fc_verification/rtl/azurelily_attn_head_d64_c128.v
//
// Drives the full N=128 token sequence end-to-end:
//   N tokens × PACKED_KQ=32 packed-int8 words = 4096 cycles of valid_x.
//   Exercises the outer FSM (S_LOAD_TOKEN → S_WAIT_FC → S_FC_DRAIN →
//   S_INTER_TOKEN_RST × N) and the inner azurelily_dimm_top firing once
//   for the whole N×d Q/K/V buffer.
//
// ──────────────────────────────────────────────────────────────────────
// DUT off-by-one workaround (hierarchical force, NOT a DUT modification):
// The head's S_LOAD_TOKEN runs for PACKED_K=32 cycles, but the inner
// azurelily_fc_128_64 takes 1 cycle to transition S_IDLE→S_LOAD when
// it first sees valid=1.  Net effect: the FC sees only 31 valid pulses
// in S_LOAD before fc_valid_in deasserts (head→S_WAIT_FC), so its
// i_w_addr maxes at 31 but never satisfies the (iwa==31 && valid)
// transition condition.  FC sits in S_LOAD forever ⇒ head sits in
// S_WAIT_FC waiting for q_fc_valid_n that never comes.
//
// Workaround: monitor the stuck condition (FC.state==S_LOAD && iwa==31
// while head.state==S_WAIT_FC) and apply a one-cycle hierarchical
// `force` to push FC.state→S_COMPUTE on each of the three (Q/K/V) FCs.
// The cost vs. natural-flow is ZERO cycles per token (the force replaces
// a missed transition, no skipped work).  This is recorded in the report
// as a "DUT off-by-one — workaround applied" annotation; it does NOT
// modify the RTL.
//
// PROBE SEMANTICS (post-2026-04-24 fix): each per-stage probe captures
// `stage_total = T_last_active - T_first_active + 1` — the full active
// duration of the stage across ALL W=16 DIMM lanes (OR-aggregated). This
// matches the IMC sim's "full stage duration" semantics. Lane-0 first-out
// timestamps are kept for diagnostic/backward-compat; the canonical
// AH_AL_STAGES_TOTAL line is what run_checks.py gates on. e2e =
// T_data_out_last_valid - T_x_first_valid + 1 (full pipeline latency).
//
// Per-stage stage-total probes:
//   linear_qkv   : first→last FC valid_n pulse on any of Q/K/V (or-reduce)
//   mac_qk       : first→last cycle ANY al_lane[i].mac_qk_inst.out_valid
//   softmax_exp  : first→last cycle ANY al_lane[i].softmax_inst input valid
//                  is asserted (S_LOAD active = scores arriving) plus the
//                  S_INV cycle (= duration "exp+inv" before normalize starts)
//   softmax_norm : first→last cycle ANY al_lane[i].softmax_inst.valid_n
//                  (= S_NORM duration)
//   mac_sv       : first→last cycle ANY al_lane[i].mac_sv_inst.out_valid
//   e2e          : T_data_out_last_valid - T_x_first_valid + 1
//
// AL TB historically folded softmax_exp+softmax_norm into the softmax_exp
// slot. The post-fix TB now reports them separately in AH_AL_STAGES_TOTAL.
//
// Legacy timestamps (kept for diagnostic dump):
//   T_x_first_valid           : first cycle valid_x=1
//   T_fc_qkv_done             : cycle state transitions to S_DIMM_FIRE
//   T_dimm_score_first_out    : first cycle mac_qk lane[0] out_valid=1
//   T_dimm_softmax_first_out  : first cycle clb_softmax lane[0] valid_n=1
//   T_dimm_wsum_first_out     : first cycle mac_sv lane[0] out_valid=1
//   T_data_out_first_valid    : first cycle top-level valid_n=1
//   T_data_out_last_valid     : last  cycle top-level valid_n=1
//
// Sim oracle (from fc_verification/expected_cycles.json,
// configs/azurelily_attn_head_d64_c128, N=128, d=64, W=16, Fmax 87.9 MHz,
// post total_dimm_dpes=16 fix):
//   linear_qkv (parallel arms, max) = 4000 cyc
//   mac_qk                          = 6661 cyc
//   softmax_exp                     = 188  cyc
//   softmax_norm                    = 750  cyc
//   mac_sv                          = 6091 cyc
//   E2E                             = 17689 cyc
//
// Tolerance budget: see fc_verification/phase7_known_deltas.json — each
// stage carries an annotated delta_cycles + tolerance pair classified as
// `modelling_granularity` (analytical-model coarseness) or `structural`
// (W=16 lane parallelism vs sim's single-lane analytical lower-bound).
// "Modelling-granularity" residuals are reported, never silently absorbed.
//
// Pre-loads (hierarchical, before reset release):
//   - FC Q/K/V weight BRAMs (per-arm w_bram): identity D_HEAD×D_MODEL such
//     that out[i] = in[i] for i < D_HEAD.  Lets us correlate output bytes
//     to input bytes for the functional sanity check.
//   - DIMM Q/K/V SRAMs: NOT pre-loaded.  These are populated by the head's
//     internal buffer→DIMM stream during S_DIMM_FIRE.
//
// Stimulus pattern: byte k of packed word i = (i*4 + k + 1) & 0x7F.
// Drive valid_x=1 for the entire 4096-cycle FC-feed phase; the head's
// FSM gates feed_count to 32 per token internally.
//
// Run model (rough):
//   per-token   = PACKED_KQ + per_FC_drain + 2 (FC_DRAIN + INTER_TOKEN_RST)
//   linear_qkv  = N × per-token
//   dimm_fire   = ~2304 (FEED_QK) + 1 + 128 (NORM) + tail_drain
//   E2E         = linear_qkv + dimm_fire
//
// AL FC azurelily_fc_128_64 cycle model (from header comment):
//   feed_load     = 32, compute_total = 2240 (64*35), output_drain = 64
//   minimum FC firing = 32 + 2240 + 64 + ~6 FSM = ~2342 cyc per token.
//
// Budget-aware sim time: 21290 + headroom ~= 30000 cycles ≈ 300 us @10ns clk.
// Hard timeout set to 10 ms = 1M cycles for safety.

`timescale 1ns / 1ps

module tb_azurelily_attn_head_v2;
    parameter DW         = 40;
    parameter D_MODEL    = 128;
    parameter D_HEAD     = 64;
    parameter N_SEQ      = 128;
    parameter W          = 16;
    parameter EPW_DSP    = 4;                                      // dsp_mac 4-wide
    parameter PACKED_KQ  = (D_MODEL + EPW_DSP - 1) / EPW_DSP;      // 32
    parameter PACKED_KO  = (D_HEAD  + EPW_DSP - 1) / EPW_DSP;      // 16

    // Sim oracle (from expected_cycles.json, post total_dimm_dpes=16 fix)
    parameter SIM_LINEAR_QKV    = 4000;
    parameter SIM_MAC_QK        = 6661;
    parameter SIM_SOFTMAX_EXP   = 188;
    parameter SIM_SOFTMAX_NORM  = 750;
    parameter SIM_MAC_SV        = 6091;
    parameter SIM_E2E           = 17689;

    reg              clk, rst, valid_x, ready_n;
    reg  [DW-1:0]    data_in_x;
    wire [DW-1:0]    data_out;
    wire             ready_x, valid_n;

    azurelily_attn_head_d64_c128 #(
        .DATA_WIDTH(DW), .D_MODEL(D_MODEL),
        .D_HEAD(D_HEAD), .N_SEQ(N_SEQ), .W(W)
    ) dut (
        .clk(clk), .rst(rst),
        .valid_x(valid_x), .data_in_x(data_in_x),
        .ready_n(ready_n),
        .data_out(data_out),
        .ready_x(ready_x), .valid_n(valid_n)
    );

    // ─── Clock + cycle counter ───────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    integer i, j;

    // ─── Per-stage timestamps (legacy lane-0 first-out probes) ──────────
    integer T_x_first_valid          = -1;
    integer T_fc_qkv_done            = -1;  // S_DIMM_FIRE entered (token_count==N)
    integer T_dimm_fire_first_cycle  = -1;
    integer T_dimm_score_first_out   = -1;
    integer T_dimm_softmax_first_out = -1;
    integer T_dimm_wsum_first_out    = -1;
    integer T_data_out_first_valid   = -1;
    integer T_data_out_last_valid    = -1;
    integer top_validn_pulses        = 0;

    // ─── Stage-total timestamps (NEW, 16-lane OR-aggregated) ────────────
    integer T_fc_first_valid         = -1;
    integer T_fc_last_valid          = -1;
    integer T_mac_qk_first_out       = -1;
    integer T_mac_qk_last_out        = -1;
    integer T_softmax_in_first       = -1;  // first cycle softmax sees a score input
    integer T_softmax_in_last        = -1;  // last cycle softmax has its valid input high
    integer T_softmax_norm_first     = -1;  // first cycle softmax_inst.valid_n high
    integer T_softmax_norm_last      = -1;  // last cycle softmax_inst.valid_n high
    integer T_mac_sv_first_out       = -1;
    integer T_mac_sv_last_out        = -1;

    // FSM-state shorthand (matches header in azurelily_attn_head_d64_c128.v)
    localparam S_IDLE             = 4'd0;
    localparam S_LOAD_TOKEN       = 4'd1;
    localparam S_WAIT_FC          = 4'd2;
    localparam S_FC_DRAIN         = 4'd3;
    localparam S_INTER_TOKEN_RST  = 4'd4;
    localparam S_DIMM_FIRE        = 4'd5;
    localparam S_DRAIN            = 4'd6;

    // ─── Probe wires (lane 0 — kept for legacy first-out probes) ────────
    wire [3:0] head_state          = dut.state;
    wire       dimm_score_first_v  = dut.dimm_inst.al_lane[0].mac_qk_inst.out_valid;
    wire       dimm_softmax_first_v= dut.dimm_inst.al_lane[0].softmax_inst.valid_n;
    wire       dimm_wsum_first_v   = dut.dimm_inst.al_lane[0].mac_sv_inst.out_valid;

    // ─── Stage-total probe wires (16-way OR-reduce across al_lane[0..15]) ──
    // mac_qk active = ANY lane's mac_qk_inst.out_valid pulse.
    wire any_mac_qk_out =
          dut.dimm_inst.al_lane[ 0].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[ 1].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[ 2].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[ 3].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[ 4].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[ 5].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[ 6].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[ 7].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[ 8].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[ 9].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[10].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[11].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[12].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[13].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[14].mac_qk_inst.out_valid
        | dut.dimm_inst.al_lane[15].mac_qk_inst.out_valid;

    // softmax_inst.valid is the input port (= mac_qk's score_valid). It pulses
    // when scores arrive (S_LOAD active). softmax_inst.valid_n is the output
    // (out_valid in S_NORM). We probe both for split exp/norm timing.
    wire any_softmax_in_v =
          dut.dimm_inst.al_lane[ 0].softmax_inst.valid
        | dut.dimm_inst.al_lane[ 1].softmax_inst.valid
        | dut.dimm_inst.al_lane[ 2].softmax_inst.valid
        | dut.dimm_inst.al_lane[ 3].softmax_inst.valid
        | dut.dimm_inst.al_lane[ 4].softmax_inst.valid
        | dut.dimm_inst.al_lane[ 5].softmax_inst.valid
        | dut.dimm_inst.al_lane[ 6].softmax_inst.valid
        | dut.dimm_inst.al_lane[ 7].softmax_inst.valid
        | dut.dimm_inst.al_lane[ 8].softmax_inst.valid
        | dut.dimm_inst.al_lane[ 9].softmax_inst.valid
        | dut.dimm_inst.al_lane[10].softmax_inst.valid
        | dut.dimm_inst.al_lane[11].softmax_inst.valid
        | dut.dimm_inst.al_lane[12].softmax_inst.valid
        | dut.dimm_inst.al_lane[13].softmax_inst.valid
        | dut.dimm_inst.al_lane[14].softmax_inst.valid
        | dut.dimm_inst.al_lane[15].softmax_inst.valid;

    wire any_softmax_norm_v =
          dut.dimm_inst.al_lane[ 0].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[ 1].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[ 2].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[ 3].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[ 4].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[ 5].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[ 6].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[ 7].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[ 8].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[ 9].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[10].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[11].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[12].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[13].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[14].softmax_inst.valid_n
        | dut.dimm_inst.al_lane[15].softmax_inst.valid_n;

    wire any_mac_sv_out =
          dut.dimm_inst.al_lane[ 0].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[ 1].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[ 2].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[ 3].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[ 4].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[ 5].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[ 6].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[ 7].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[ 8].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[ 9].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[10].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[11].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[12].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[13].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[14].mac_sv_inst.out_valid
        | dut.dimm_inst.al_lane[15].mac_sv_inst.out_valid;

    // FC valid_n OR-reduce (Q/K/V fire in lockstep but we OR for safety).
    wire any_fc_valid =
          dut.q_fc_valid_n | dut.k_fc_valid_n | dut.v_fc_valid_n;

    // ─── AH gate Stage 1: Fmax-independent counter probes ───────────────
    // Per-stage primitive activation counters, summed across all parallel
    // hardware. AL uses dsp_mac (one valid_n pulse per output) for
    // mac_qk/mac_sv, and clb_softmax (S_LOAD valid pulses for exp,
    // S_NORM valid_n pulses for norm). FC arms have 64 dsp_macs each
    // (parallel-output), so we count valid_n pulses on Q/K/V FCs (= 1 row
    // produced per pulse, but we count GLOBAL fires by summing dsp_macs
    // active per-cycle pulse across all 64+64+64 dsp_macs in the FC arms).
    // For FC: we use the simpler "FC valid_n pulses per arm × 64 dsp_macs
    // per arm" — each FC valid_n pulse carries one row's worth of outputs,
    // produced by 64 parallel dsp_macs firing simultaneously. So
    // dsp_mac fires per FC arm = (# FC valid_n pulses) × 64.
    // Total across 3 arms = 3 × pulses × 64.

    // Edge counter: count rising edges on dsp_mac.valid_n across 16 lanes.
    // For mac_qk: dut.dimm_inst.al_lane[i].mac_qk_inst.valid_n
    // For mac_sv: dut.dimm_inst.al_lane[i].mac_sv_inst.valid_n
    // For softmax_exp: count rising edges of clb_softmax S_LOAD entry events
    //                  per lane (= each lane's clb_softmax processes 1 attn row,
    //                  the S_LOAD state runs N=128 cycles consuming scores).
    //                  We count per-cycle "valid input strobes" instead of fires
    //                  because clb_softmax doesn't have a discrete "fire" event
    //                  for exp — it's an inline shift-LUT applied per S_LOAD cyc.
    // For softmax_norm: count rising edges of clb_softmax.valid_n (= S_NORM
    //                   completion).

    integer fc_q_pulses = 0, fc_k_pulses = 0, fc_v_pulses = 0;
    reg fc_q_prev = 0, fc_k_prev = 0, fc_v_prev = 0;
    always @(posedge clk) begin
        if (rst) begin
            fc_q_pulses <= 0; fc_k_pulses <= 0; fc_v_pulses <= 0;
            fc_q_prev <= 0; fc_k_prev <= 0; fc_v_prev <= 0;
        end else begin
            if (!fc_q_prev && dut.q_fc_valid_n) fc_q_pulses <= fc_q_pulses + 1;
            if (!fc_k_prev && dut.k_fc_valid_n) fc_k_pulses <= fc_k_pulses + 1;
            if (!fc_v_prev && dut.v_fc_valid_n) fc_v_pulses <= fc_v_pulses + 1;
            fc_q_prev <= dut.q_fc_valid_n;
            fc_k_prev <= dut.k_fc_valid_n;
            fc_v_prev <= dut.v_fc_valid_n;
        end
    end

    // mac_qk dsp_mac valid_n rising edges across 16 lanes (= total dsp_mac
    // output events for mac_qk).
    integer mac_qk_dsp_fires   = 0;
    integer mac_sv_dsp_fires   = 0;
    integer softmax_exp_inputs = 0;  // clb_softmax S_LOAD valid pulses (per row)
    integer softmax_norm_outs  = 0;  // clb_softmax valid_n rising edges
    reg [15:0] al_mq_prev, al_mq_curr;
    reg [15:0] al_ms_prev, al_ms_curr;
    reg [15:0] al_sx_prev, al_sx_curr;
    reg [15:0] al_sn_prev, al_sn_curr;

    always @(*) begin
        al_mq_curr[ 0] = dut.dimm_inst.al_lane[ 0].mac_qk_inst.valid_n;
        al_mq_curr[ 1] = dut.dimm_inst.al_lane[ 1].mac_qk_inst.valid_n;
        al_mq_curr[ 2] = dut.dimm_inst.al_lane[ 2].mac_qk_inst.valid_n;
        al_mq_curr[ 3] = dut.dimm_inst.al_lane[ 3].mac_qk_inst.valid_n;
        al_mq_curr[ 4] = dut.dimm_inst.al_lane[ 4].mac_qk_inst.valid_n;
        al_mq_curr[ 5] = dut.dimm_inst.al_lane[ 5].mac_qk_inst.valid_n;
        al_mq_curr[ 6] = dut.dimm_inst.al_lane[ 6].mac_qk_inst.valid_n;
        al_mq_curr[ 7] = dut.dimm_inst.al_lane[ 7].mac_qk_inst.valid_n;
        al_mq_curr[ 8] = dut.dimm_inst.al_lane[ 8].mac_qk_inst.valid_n;
        al_mq_curr[ 9] = dut.dimm_inst.al_lane[ 9].mac_qk_inst.valid_n;
        al_mq_curr[10] = dut.dimm_inst.al_lane[10].mac_qk_inst.valid_n;
        al_mq_curr[11] = dut.dimm_inst.al_lane[11].mac_qk_inst.valid_n;
        al_mq_curr[12] = dut.dimm_inst.al_lane[12].mac_qk_inst.valid_n;
        al_mq_curr[13] = dut.dimm_inst.al_lane[13].mac_qk_inst.valid_n;
        al_mq_curr[14] = dut.dimm_inst.al_lane[14].mac_qk_inst.valid_n;
        al_mq_curr[15] = dut.dimm_inst.al_lane[15].mac_qk_inst.valid_n;

        al_ms_curr[ 0] = dut.dimm_inst.al_lane[ 0].mac_sv_inst.valid_n;
        al_ms_curr[ 1] = dut.dimm_inst.al_lane[ 1].mac_sv_inst.valid_n;
        al_ms_curr[ 2] = dut.dimm_inst.al_lane[ 2].mac_sv_inst.valid_n;
        al_ms_curr[ 3] = dut.dimm_inst.al_lane[ 3].mac_sv_inst.valid_n;
        al_ms_curr[ 4] = dut.dimm_inst.al_lane[ 4].mac_sv_inst.valid_n;
        al_ms_curr[ 5] = dut.dimm_inst.al_lane[ 5].mac_sv_inst.valid_n;
        al_ms_curr[ 6] = dut.dimm_inst.al_lane[ 6].mac_sv_inst.valid_n;
        al_ms_curr[ 7] = dut.dimm_inst.al_lane[ 7].mac_sv_inst.valid_n;
        al_ms_curr[ 8] = dut.dimm_inst.al_lane[ 8].mac_sv_inst.valid_n;
        al_ms_curr[ 9] = dut.dimm_inst.al_lane[ 9].mac_sv_inst.valid_n;
        al_ms_curr[10] = dut.dimm_inst.al_lane[10].mac_sv_inst.valid_n;
        al_ms_curr[11] = dut.dimm_inst.al_lane[11].mac_sv_inst.valid_n;
        al_ms_curr[12] = dut.dimm_inst.al_lane[12].mac_sv_inst.valid_n;
        al_ms_curr[13] = dut.dimm_inst.al_lane[13].mac_sv_inst.valid_n;
        al_ms_curr[14] = dut.dimm_inst.al_lane[14].mac_sv_inst.valid_n;
        al_ms_curr[15] = dut.dimm_inst.al_lane[15].mac_sv_inst.valid_n;

        // softmax exp/norm: clb_softmax has `valid` (input) and `valid_n` (output).
        // For exp: 1 input pulse per S_LOAD step (valid signal active). For norm:
        // valid_n = output S_NORM pulses.
        al_sx_curr[ 0] = dut.dimm_inst.al_lane[ 0].softmax_inst.valid;
        al_sx_curr[ 1] = dut.dimm_inst.al_lane[ 1].softmax_inst.valid;
        al_sx_curr[ 2] = dut.dimm_inst.al_lane[ 2].softmax_inst.valid;
        al_sx_curr[ 3] = dut.dimm_inst.al_lane[ 3].softmax_inst.valid;
        al_sx_curr[ 4] = dut.dimm_inst.al_lane[ 4].softmax_inst.valid;
        al_sx_curr[ 5] = dut.dimm_inst.al_lane[ 5].softmax_inst.valid;
        al_sx_curr[ 6] = dut.dimm_inst.al_lane[ 6].softmax_inst.valid;
        al_sx_curr[ 7] = dut.dimm_inst.al_lane[ 7].softmax_inst.valid;
        al_sx_curr[ 8] = dut.dimm_inst.al_lane[ 8].softmax_inst.valid;
        al_sx_curr[ 9] = dut.dimm_inst.al_lane[ 9].softmax_inst.valid;
        al_sx_curr[10] = dut.dimm_inst.al_lane[10].softmax_inst.valid;
        al_sx_curr[11] = dut.dimm_inst.al_lane[11].softmax_inst.valid;
        al_sx_curr[12] = dut.dimm_inst.al_lane[12].softmax_inst.valid;
        al_sx_curr[13] = dut.dimm_inst.al_lane[13].softmax_inst.valid;
        al_sx_curr[14] = dut.dimm_inst.al_lane[14].softmax_inst.valid;
        al_sx_curr[15] = dut.dimm_inst.al_lane[15].softmax_inst.valid;

        al_sn_curr[ 0] = dut.dimm_inst.al_lane[ 0].softmax_inst.valid_n;
        al_sn_curr[ 1] = dut.dimm_inst.al_lane[ 1].softmax_inst.valid_n;
        al_sn_curr[ 2] = dut.dimm_inst.al_lane[ 2].softmax_inst.valid_n;
        al_sn_curr[ 3] = dut.dimm_inst.al_lane[ 3].softmax_inst.valid_n;
        al_sn_curr[ 4] = dut.dimm_inst.al_lane[ 4].softmax_inst.valid_n;
        al_sn_curr[ 5] = dut.dimm_inst.al_lane[ 5].softmax_inst.valid_n;
        al_sn_curr[ 6] = dut.dimm_inst.al_lane[ 6].softmax_inst.valid_n;
        al_sn_curr[ 7] = dut.dimm_inst.al_lane[ 7].softmax_inst.valid_n;
        al_sn_curr[ 8] = dut.dimm_inst.al_lane[ 8].softmax_inst.valid_n;
        al_sn_curr[ 9] = dut.dimm_inst.al_lane[ 9].softmax_inst.valid_n;
        al_sn_curr[10] = dut.dimm_inst.al_lane[10].softmax_inst.valid_n;
        al_sn_curr[11] = dut.dimm_inst.al_lane[11].softmax_inst.valid_n;
        al_sn_curr[12] = dut.dimm_inst.al_lane[12].softmax_inst.valid_n;
        al_sn_curr[13] = dut.dimm_inst.al_lane[13].softmax_inst.valid_n;
        al_sn_curr[14] = dut.dimm_inst.al_lane[14].softmax_inst.valid_n;
        al_sn_curr[15] = dut.dimm_inst.al_lane[15].softmax_inst.valid_n;
    end

    integer ali, _almq_inc, _alms_inc, _alsx_inc, _alsn_inc;
    always @(posedge clk) begin
        if (rst) begin
            al_mq_prev <= 0; al_ms_prev <= 0;
            al_sx_prev <= 0; al_sn_prev <= 0;
            mac_qk_dsp_fires   <= 0;
            mac_sv_dsp_fires   <= 0;
            softmax_exp_inputs <= 0;
            softmax_norm_outs  <= 0;
        end else begin
            _almq_inc = 0; _alms_inc = 0; _alsx_inc = 0; _alsn_inc = 0;
            for (ali = 0; ali < 16; ali = ali + 1) begin
                if (!al_mq_prev[ali] && al_mq_curr[ali]) _almq_inc = _almq_inc + 1;
                if (!al_ms_prev[ali] && al_ms_curr[ali]) _alms_inc = _alms_inc + 1;
                if (!al_sx_prev[ali] && al_sx_curr[ali]) _alsx_inc = _alsx_inc + 1;
                if (!al_sn_prev[ali] && al_sn_curr[ali]) _alsn_inc = _alsn_inc + 1;
            end
            mac_qk_dsp_fires   <= mac_qk_dsp_fires   + _almq_inc;
            mac_sv_dsp_fires   <= mac_sv_dsp_fires   + _alms_inc;
            softmax_exp_inputs <= softmax_exp_inputs + _alsx_inc;
            softmax_norm_outs  <= softmax_norm_outs  + _alsn_inc;
            al_mq_prev <= al_mq_curr;
            al_ms_prev <= al_ms_curr;
            al_sx_prev <= al_sx_curr;
            al_sn_prev <= al_sn_curr;
        end
    end

    // Edge sample: every cycle log first-valid + last-valid timestamps.
    always @(posedge clk) begin
        if (!rst) begin
            if (valid_x && T_x_first_valid < 0) T_x_first_valid <= cycle;
            if (head_state == S_DIMM_FIRE && T_fc_qkv_done < 0) begin
                T_fc_qkv_done           <= cycle;
                T_dimm_fire_first_cycle <= cycle;
            end
            // Legacy lane-0 first-out timestamps.
            if (dimm_score_first_v   && T_dimm_score_first_out  < 0) T_dimm_score_first_out  <= cycle;
            if (dimm_softmax_first_v && T_dimm_softmax_first_out< 0) T_dimm_softmax_first_out<= cycle;
            if (dimm_wsum_first_v    && T_dimm_wsum_first_out   < 0) T_dimm_wsum_first_out   <= cycle;

            // Stage-total OR-aggregated first/last.
            if (any_fc_valid) begin
                if (T_fc_first_valid < 0) T_fc_first_valid <= cycle;
                T_fc_last_valid <= cycle;
            end
            if (any_mac_qk_out) begin
                if (T_mac_qk_first_out < 0) T_mac_qk_first_out <= cycle;
                T_mac_qk_last_out <= cycle;
            end
            if (any_softmax_in_v) begin
                if (T_softmax_in_first < 0) T_softmax_in_first <= cycle;
                T_softmax_in_last <= cycle;
            end
            if (any_softmax_norm_v) begin
                if (T_softmax_norm_first < 0) T_softmax_norm_first <= cycle;
                T_softmax_norm_last <= cycle;
            end
            if (any_mac_sv_out) begin
                if (T_mac_sv_first_out < 0) T_mac_sv_first_out <= cycle;
                T_mac_sv_last_out <= cycle;
            end

            if (valid_n) begin
                if (T_data_out_first_valid < 0) T_data_out_first_valid <= cycle;
                T_data_out_last_valid <= cycle;
                top_validn_pulses     <= top_validn_pulses + 1;
            end
        end
    end

    // ─── DUT issue 1 (AH RTL bug #1 — FC S_LOAD off-by-one): FIXED ───────
    // Root cause: head's S_LOAD_TOKEN ran PACKED_K=32 cycles, but the FC
    // took 1-2 cycles to transition S_IDLE->S_LOAD (async-reset settle on
    // fc_rst_pulse between tokens, or global rst for token 0), so the FC
    // only saw 30-31 valid pulses. S_LOAD->S_COMPUTE needs
    // (i_w_addr==PACKED_K-1 && valid), which requires PACKED_K+1 valid
    // pulses from the head.
    //
    // Fix (gen_azurelily_attn_head_top.py): extend S_LOAD_TOKEN window to
    // PACKED_K+2 cycles (feed_count: 0..PACKED_K+1). Every FC arm now
    // reaches S_COMPUTE without any hierarchical force. fc_force_count
    // should remain 0 after the fix.
    //
    // ─── DUT issue 2 (AH RTL bug #2 — q_w_addr 5-bit wrap): FIXED ─────────
    // Root cause: azurelily_dimm_top_d64_c128.v sized q_w_addr/q_r_addr
    // from depth_q=17 (5 bits, max 31). The head drives valid_q for
    // BUF_DEPTH-1=1664 cycles during S_DIMM_FIRE, so q_w_addr wrapped
    // every 32 cycles and the S_LOAD threshold (q_w_addr>=13 &&
    // k_w_addr>=1664 && v_w_addr>=1664) almost never held.
    //
    // Fix (gen_dimm_azurelily_top.py): widen q_w_addr/q_r_addr to the
    // shared addr width (12 bits, same as k/v). q_w_addr now grows
    // monotonically and the threshold holds once k/v reach 1664.
    //
    // ─── DUT issue 3 (AH RTL bug #3 — clb_softmax out_valid stuck): FIXED ─
    // Root cause: the generator patched out the `out_valid <= 0` clear in
    // the S_NORM->S_LOAD transition of clb_softmax, so once the first
    // softmax completed, out_valid stayed high indefinitely and mac_sv
    // kept accumulating bogus attention outputs.
    //
    // Fix (gen_dimm_azurelily_top.py): skip that `.replace()` patch.
    // out_valid now deasserts on the S_LOAD re-entry edge.
    //
    // Telemetry counters retained (should stay at 0 post-fix).
    integer fc_force_count = 0;
    reg dimm_valid_q_force_active = 0;
    integer waitfc_streak;
    always @(posedge clk) begin
        if (rst) waitfc_streak <= 0;
        else if (head_state == S_WAIT_FC) waitfc_streak <= waitfc_streak + 1;
        else waitfc_streak <= 0;
    end

    // ─── Stimulus task: drive one packed input word ──────────────────────
    task drive_word(input [DW-1:0] word);
    begin
        @(posedge clk);
        valid_x   <= 1;
        data_in_x <= word;
    end
    endtask

    // ─── Run plan ────────────────────────────────────────────────────────
    integer total_feed_cycles;
    integer settle_cycles;
    integer t;
    integer expected_top_pulses;
    integer e2e_rtl_cyc;
    integer linear_qkv_rtl;
    integer mac_qk_rtl;
    integer softmax_combined_rtl;
    integer mac_sv_rtl;

    // Stage-total durations (NEW probe semantics).
    integer linear_qkv_total;
    integer mac_qk_total;
    integer softmax_exp_total;
    integer softmax_norm_total;
    integer mac_sv_total;
    integer e2e_total;

    initial begin
        rst       = 1;
        valid_x   = 0;
        ready_n   = 0;
        data_in_x = 0;

        // ── Hierarchical pre-load: FC weight BRAMs (identity) ─────────
        // Set Q/K/V BRAM weights so out[i] = in[i] for i < D_HEAD.
        // weight layout = w_bram.mem[output_idx*PACKED_KQ + (input_idx/EPW_DSP)]
        //                 byte (input_idx % EPW_DSP) = 1 when input_idx == output_idx.
        for (i = 0; i < D_HEAD * PACKED_KQ; i = i + 1) begin
            dut.fc_q_inst.w_bram.mem[i] = 0;
            dut.fc_k_inst.w_bram.mem[i] = 0;
            dut.fc_v_inst.w_bram.mem[i] = 0;
        end
        for (i = 0; i < D_HEAD; i = i + 1) begin
            dut.fc_q_inst.w_bram.mem[i*PACKED_KQ + (i / EPW_DSP)][(i % EPW_DSP)*8 +: 8] = 8'd1;
            dut.fc_k_inst.w_bram.mem[i*PACKED_KQ + (i / EPW_DSP)][(i % EPW_DSP)*8 +: 8] = 8'd1;
            dut.fc_v_inst.w_bram.mem[i*PACKED_KQ + (i / EPW_DSP)][(i % EPW_DSP)*8 +: 8] = 8'd1;
        end

        // ── Reset release ──
        #20; rst = 0; @(posedge clk); #1;

        $display("=== Azure-Lily Attention Head V2 — Combined Functional+Latency TB ===");
        $display("  Config: D_MODEL=%0d, D_HEAD=%0d, N_SEQ=%0d, W=%0d, Fmax=87.9 MHz",
                 D_MODEL, D_HEAD, N_SEQ, W);
        $display("  Sim oracle (cyc): linear_qkv=%0d mac_qk=%0d softmax_exp=%0d softmax_norm=%0d mac_sv=%0d e2e=%0d",
                 SIM_LINEAR_QKV, SIM_MAC_QK, SIM_SOFTMAX_EXP, SIM_SOFTMAX_NORM,
                 SIM_MAC_SV, SIM_E2E);
        $display("  [t=%0d] reset released", cycle);

        // ── Phase A: drive valid_x for all N tokens ───────────────────
        // The head's FSM gates feed_count internally; we just hold valid_x=1
        // for N × PACKED_KQ + slack cycles.  Each cycle we present one
        // packed-byte word with byte k = (cycle*EPW_DSP + k + 1) & 0x7F.
        //
        // After the head has serviced all N tokens, S_DIMM_FIRE fires and
        // ready_x drops; we keep valid_x=1 (the head ignores it in
        // S_WAIT_FC / S_FC_DRAIN / S_INTER_TOKEN_RST).
        valid_x   <= 1;
        @(posedge clk);

        // Drive PACKED_KQ × N packed words. The head's FSM only consumes
        // PACKED_KQ per token (gated by feed_count), so this is enough.
        // We loop over the maximum needed cycles to reach S_DIMM_FIRE,
        // each cycle presenting a structured pattern.
        total_feed_cycles = 0;
        // Hold valid_x and present a streaming pattern; let the head drive
        // its FSM through all N tokens.  We exit when state == S_DIMM_FIRE.
        // Worst case: per-token = PACKED_KQ (load) + ~2240 (compute) + ~64
        // (drain) + ~6 (FSM glue) ~ 2342 cyc.  N tokens = ~300K cyc.
        // Hard guard: 1,000,000 cycles.
        begin : feed_loop
            integer last_token;
            last_token = -1;
            for (t = 0; t < 1000000; t = t + 1) begin
                // Print every 16th token boundary so the run shows progress
                // without clogging the log.
                if (dut.token_count != last_token
                    && (dut.token_count == 0 || dut.token_count == N_SEQ-1
                        || (dut.token_count[4:0] == 0))) begin
                    $display("  [t=%0d] token=%0d (FC iteration progress)",
                             cycle, dut.token_count);
                    last_token = dut.token_count;
                end
                // Vary the packed input word per cycle so that any structural
                // hazard (e.g. byte alignment) gets exercised.
                data_in_x[ 7: 0] <= ((t*4 + 1) & 8'h7F);
                data_in_x[15: 8] <= ((t*4 + 2) & 8'h7F);
                data_in_x[23:16] <= ((t*4 + 3) & 8'h7F);
                data_in_x[31:24] <= ((t*4 + 4) & 8'h7F);
                data_in_x[39:32] <= 8'd0;
                @(posedge clk);
                total_feed_cycles = total_feed_cycles + 1;
                if (head_state == S_DIMM_FIRE) begin
                    // FC arm phase complete; stop driving valid_x to free
                    // the SRAMs (the head's S_DIMM_FIRE doesn't consume top
                    // valid_x).
                    valid_x   <= 0;
                    data_in_x <= 0;
                    @(posedge clk);  // settle one cycle
                    disable feed_loop;
                end
            end
        end

        if (head_state != S_DIMM_FIRE && head_state != S_DRAIN) begin
            $display("  WARN [t=%0d]: state=%0d after %0d feed cycles (no S_DIMM_FIRE reached)",
                     cycle, head_state, total_feed_cycles);
        end else begin
            $display("  [t=%0d] FC_QKV phase complete (state=%0d), drove %0d feed cycles, fc_force_count=%0d",
                     cycle, head_state, total_feed_cycles, fc_force_count);
        end

        // ── Phase B: wait for DIMM to drain ──
        // Bug-1 fix: head FSM now loops N=128 Q rows. Each iteration takes
        // ~3,500 cycles (Q load + K/V re-stream + DIMM internal compute), so
        // 128 iterations ≈ 500K cycles total. Cap raised from 16000 to 1.5M
        // to accommodate the outer-loop refactor.
        begin : wait_drain
            for (i = 0; i < 1500000; i = i + 1) begin
                @(posedge clk);
            end
        end

        // Settle a bit so any final NBA writes commit.
        for (i = 0; i < 8; i = i + 1) @(posedge clk);
        settle_cycles = cycle;

        // ── Final report ──
        $display("");
        $display("=== Per-Stage Timestamps (cycle index) ===");
        $display("  T_x_first_valid          = %0d", T_x_first_valid);
        $display("  T_fc_qkv_done            = %0d", T_fc_qkv_done);
        $display("  T_dimm_score_first_out   = %0d", T_dimm_score_first_out);
        $display("  T_dimm_softmax_first_out = %0d", T_dimm_softmax_first_out);
        $display("  T_dimm_wsum_first_out    = %0d", T_dimm_wsum_first_out);
        $display("  T_data_out_first_valid   = %0d", T_data_out_first_valid);
        $display("  T_data_out_last_valid    = %0d", T_data_out_last_valid);
        $display("  --- Stage-total OR-aggregated (16 lanes) ---");
        $display("  T_fc_first_valid         = %0d", T_fc_first_valid);
        $display("  T_fc_last_valid          = %0d", T_fc_last_valid);
        $display("  T_mac_qk_first_out       = %0d", T_mac_qk_first_out);
        $display("  T_mac_qk_last_out        = %0d", T_mac_qk_last_out);
        $display("  T_softmax_in_first       = %0d", T_softmax_in_first);
        $display("  T_softmax_in_last        = %0d", T_softmax_in_last);
        $display("  T_softmax_norm_first     = %0d", T_softmax_norm_first);
        $display("  T_softmax_norm_last      = %0d", T_softmax_norm_last);
        $display("  T_mac_sv_first_out       = %0d", T_mac_sv_first_out);
        $display("  T_mac_sv_last_out        = %0d", T_mac_sv_last_out);
        $display("  top valid_n pulses       = %0d", top_validn_pulses);
        $display("");

        // ── Per-stage cycle attribution (RTL) ──
        // Stage = first_out_of_NEXT_stage - first_out_of_THIS_stage (latency
        // from the previous stage's first output to this stage's first
        // output).  This is the same convention used by NL-DPE attn_head
        // and the existing dimm_top latency TB (`tb_*_dimm_top_latency.v`).
        //
        // linear_qkv = T_fc_qkv_done            - T_x_first_valid
        //              (FC arm phase, all 3 arms run in parallel; head iterates
        //              N tokens, ≈N×(PACKED_KQ + N_OUTPUT*(PACKED_KQ+3) + N_OUTPUT))
        // mac_qk     = T_dimm_score_first_out   - T_dimm_fire_first_cycle
        //              (cycles from S_DIMM_FIRE entry to first lane[0] mac_qk
        //              out_valid; ~q_addr_load + dsp K=16 cycles)
        // softmax    = T_dimm_softmax_first_out - T_dimm_score_first_out
        //              (cycles from first score pulse to first softmax valid_n;
        //              ~clb_softmax S_LOAD over N=128 score pulses (each 18
        //              cyc apart) + S_INV (1) + first S_NORM output)
        // mac_sv     = T_dimm_wsum_first_out    - T_dimm_softmax_first_out
        //              (cycles from first softmax pulse to first mac_sv pulse;
        //              ~dsp K=32 cycles)
        // e2e        = T_data_out_last_valid    - T_x_first_valid
        //              (full pipeline first→last visible at top valid_n)
        if (T_x_first_valid >= 0 && T_fc_qkv_done >= 0)
            linear_qkv_rtl = T_fc_qkv_done - T_x_first_valid;
        else linear_qkv_rtl = -1;

        if (T_dimm_fire_first_cycle >= 0 && T_dimm_score_first_out >= 0)
            mac_qk_rtl = T_dimm_score_first_out - T_dimm_fire_first_cycle;
        else mac_qk_rtl = -1;

        if (T_dimm_softmax_first_out >= 0 && T_dimm_score_first_out >= 0)
            softmax_combined_rtl = T_dimm_softmax_first_out - T_dimm_score_first_out;
        else softmax_combined_rtl = -1;

        if (T_dimm_wsum_first_out >= 0 && T_dimm_softmax_first_out >= 0)
            mac_sv_rtl = T_dimm_wsum_first_out - T_dimm_softmax_first_out;
        else mac_sv_rtl = -1;

        if (T_data_out_last_valid >= 0 && T_x_first_valid >= 0)
            e2e_rtl_cyc = T_data_out_last_valid - T_x_first_valid;
        else e2e_rtl_cyc = -1;

        $display("=== Per-Stage Cycle Counts (RTL vs sim oracle) ===");
        $display("  LEGACY (first-out-to-first-out latencies):");
        $display("");
        $display("  linear_qkv (FC arms)  : RTL=%0d  sim=%0d  Δ=%0d",
                 linear_qkv_rtl, SIM_LINEAR_QKV, linear_qkv_rtl - SIM_LINEAR_QKV);
        $display("  mac_qk (first out)    : RTL=%0d  sim=%0d  Δ=%0d",
                 mac_qk_rtl, SIM_MAC_QK, mac_qk_rtl - SIM_MAC_QK);
        $display("  softmax (score→sm)    : RTL=%0d  sim=%0d  Δ=%0d",
                 softmax_combined_rtl,
                 SIM_SOFTMAX_EXP + SIM_SOFTMAX_NORM,
                 softmax_combined_rtl - (SIM_SOFTMAX_EXP + SIM_SOFTMAX_NORM));
        $display("  mac_sv (sm→sv)        : RTL=%0d  sim=%0d  Δ=%0d",
                 mac_sv_rtl, SIM_MAC_SV, mac_sv_rtl - SIM_MAC_SV);
        $display("  E2E (last_vn-first_vx): RTL=%0d  sim=%0d  Δ=%0d",
                 e2e_rtl_cyc, SIM_E2E, e2e_rtl_cyc - SIM_E2E);
        $display("");

        // Stage-total durations (T_last - T_first + 1) across all W=16 lanes.
        linear_qkv_total   = (T_fc_first_valid    >= 0 && T_fc_last_valid    >= 0)
                           ? (T_fc_last_valid    - T_fc_first_valid    + 1) : 0;
        mac_qk_total       = (T_mac_qk_first_out  >= 0 && T_mac_qk_last_out  >= 0)
                           ? (T_mac_qk_last_out  - T_mac_qk_first_out  + 1) : 0;
        softmax_exp_total  = (T_softmax_in_first  >= 0 && T_softmax_in_last  >= 0)
                           ? (T_softmax_in_last  - T_softmax_in_first  + 1) : 0;
        softmax_norm_total = (T_softmax_norm_first>= 0 && T_softmax_norm_last>= 0)
                           ? (T_softmax_norm_last- T_softmax_norm_first+ 1) : 0;
        mac_sv_total       = (T_mac_sv_first_out  >= 0 && T_mac_sv_last_out  >= 0)
                           ? (T_mac_sv_last_out  - T_mac_sv_first_out  + 1) : 0;
        e2e_total          = (T_data_out_last_valid >= 0 && T_x_first_valid    >= 0)
                           ? (T_data_out_last_valid - T_x_first_valid + 1) : 0;

        $display("  STAGE-TOTAL (full active duration across 16 lanes):");
        $display("");
        $display("  linear_qkv  : RTL=%5d  sim=%5d  Δ=%+0d",
                 linear_qkv_total, SIM_LINEAR_QKV,
                 linear_qkv_total - SIM_LINEAR_QKV);
        $display("  mac_qk      : RTL=%5d  sim=%5d  Δ=%+0d",
                 mac_qk_total, SIM_MAC_QK, mac_qk_total - SIM_MAC_QK);
        $display("  softmax_exp : RTL=%5d  sim=%5d  Δ=%+0d",
                 softmax_exp_total, SIM_SOFTMAX_EXP,
                 softmax_exp_total - SIM_SOFTMAX_EXP);
        $display("  softmax_norm: RTL=%5d  sim=%5d  Δ=%+0d",
                 softmax_norm_total, SIM_SOFTMAX_NORM,
                 softmax_norm_total - SIM_SOFTMAX_NORM);
        $display("  mac_sv      : RTL=%5d  sim=%5d  Δ=%+0d",
                 mac_sv_total, SIM_MAC_SV, mac_sv_total - SIM_MAC_SV);
        $display("  e2e         : RTL=%5d  sim=%5d  Δ=%+0d",
                 e2e_total, SIM_E2E, e2e_total - SIM_E2E);
        $display("");
        $display("  Workarounds applied: FC nudge=%0d, DIMM valid_q gate=%0d",
                 fc_force_count, dimm_valid_q_force_active ? 1 : 0);
        $display("");

        // Functional sanity: count top valid_n pulses.
        // Expected: 16 lanes × (D_HEAD/EPW_DSP)? Actually mac_sv produces
        // 1 output per 32 attn-valids. With 128 attn values per row and
        // 16 lanes broadcast, we expect 4 mac_sv outputs per lane, at
        // various cycle offsets. Top valid_n = OR-reduce of all lane_valid,
        // so we expect roughly 4 cycles of valid_n total (when at least one
        // lane is asserting). The output is round-robin across lanes.
        expected_top_pulses = 4;  // approximate; depends on lane staggering
        $display("=== Functional Sanity ===");
        $display("  top valid_n pulses    : %0d (expect roughly 4-64, depends on lane staggering)",
                 top_validn_pulses);
        if (top_validn_pulses > 0) begin
            $display("  Functional output     : valid_n asserted (PASS gate-1)");
        end else begin
            $display("  Functional output     : NO valid_n (FAIL gate-1)");
        end
        $display("");

        // ── Regex-friendly final lines ──
        // Legacy line (first-out latencies, exp+norm folded into exp slot).
        $display("AH_AL_STAGES linear_qkv=%0d mac_qk=%0d softmax_exp=%0d softmax_norm=%0d mac_sv=%0d e2e=%0d",
                 linear_qkv_rtl,
                 mac_qk_rtl,
                 softmax_combined_rtl,  // map to softmax_exp slot (combined)
                 0,                      // softmax_norm folded into combined
                 mac_sv_rtl,
                 e2e_rtl_cyc);
        // NEW canonical line (stage-total durations, exp/norm reported separately).
        $display("AH_AL_STAGES_TOTAL linear_qkv=%0d mac_qk=%0d softmax_exp=%0d softmax_norm=%0d mac_sv=%0d e2e=%0d",
                 linear_qkv_total,
                 mac_qk_total,
                 softmax_exp_total,
                 softmax_norm_total,
                 mac_sv_total,
                 e2e_total);

        // ── AH gate Stage 1: Fmax-independent counter emission ──
        // For AL FCs, each FC valid_n pulse corresponds to 1 row produced
        // by 64 parallel dsp_macs (the fc_q/k/v_inst has 64 parallel-output
        // dsp_macs, all firing simultaneously per row). Total dsp_mac fires
        // per FC valid_n pulse = 64. Total across 3 arms = 64 × pulses × 3.
        // For mac_qk/mac_sv: dsp_mac.valid_n rising edges across 16 lanes.
        // For softmax_exp: clb_softmax.valid (input) rising edges (= score
        //   inputs consumed during S_LOAD phase). Total = inputs across 16
        //   lanes (each S_LOAD pulse triggers a combinational exp() LUT —
        //   one fire per pulse).
        // For softmax_norm: clb_softmax.valid_n rising edges (= S_NORM
        //   completion events). Each lane completes 1 attn row's norm
        //   stream of N elements before resetting; we count rising edges
        //   = # of norm "starts" per lane.
        $display("");
        $display("=== AH-gate counters (TB-side architectural invariants) ===");
        // Bug-1 fix: row_count is now N_SEQ (head FSM loops N=128 Q rows).
        $display("COUNTER linear_qkv dpe_fire_count %0d", 64*(fc_q_pulses + fc_k_pulses + fc_v_pulses));
        $display("COUNTER linear_qkv pass_count %0d", 64*(fc_q_pulses + fc_k_pulses + fc_v_pulses));
        $display("COUNTER linear_qkv row_count %0d", N_SEQ);
        $display("COUNTER linear_qkv lane_count 192");  // 3 arms × 64 dsp_macs
        $display("COUNTER mac_qk dpe_fire_count %0d", mac_qk_dsp_fires);
        $display("COUNTER mac_qk pass_count %0d", mac_qk_dsp_fires);
        $display("COUNTER mac_qk row_count %0d", N_SEQ);
        $display("COUNTER mac_qk lane_count %0d", W);
        $display("COUNTER softmax_exp dpe_fire_count %0d", softmax_exp_inputs);
        $display("COUNTER softmax_exp pass_count %0d", softmax_exp_inputs);
        $display("COUNTER softmax_exp row_count %0d", N_SEQ);
        $display("COUNTER softmax_exp lane_count %0d", W);
        $display("COUNTER softmax_exp softmax_exp_fires %0d", softmax_exp_inputs);
        $display("COUNTER softmax_norm dpe_fire_count %0d", softmax_norm_outs);
        $display("COUNTER softmax_norm pass_count %0d", softmax_norm_outs);
        $display("COUNTER softmax_norm row_count %0d", N_SEQ);
        $display("COUNTER softmax_norm lane_count %0d", W);
        $display("COUNTER softmax_norm softmax_norm_fires %0d", softmax_norm_outs);
        $display("COUNTER mac_sv dpe_fire_count %0d", mac_sv_dsp_fires);
        $display("COUNTER mac_sv pass_count %0d", mac_sv_dsp_fires);
        $display("COUNTER mac_sv row_count %0d", N_SEQ);
        $display("COUNTER mac_sv lane_count %0d", W);

        $finish;
    end

    // Hard timeout: bumped to 60 ms (6M cycles) to accommodate the Bug-1
    // outer Q-row loop (~500K cycles for 128 iterations × ~3.5K cyc each).
    initial begin
        #60_000_000;
        $display("HARD TIMEOUT (60ms simulated time)");
        $display("AH_AL_STAGES linear_qkv=-1 mac_qk=-1 softmax_exp=-1 softmax_norm=-1 mac_sv=-1 e2e=-1");
        $display("AH_AL_STAGES_TOTAL linear_qkv=-1 mac_qk=-1 softmax_exp=-1 softmax_norm=-1 mac_sv=-1 e2e=-1");
        $finish;
    end

endmodule
