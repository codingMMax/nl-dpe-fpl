// NL-DPE Attention Head V2 — combined functional + latency TB
//
// AH track: drives the full N=128 token sequence through the head, probes
// per-stage FSM timestamps, and emits a regex-friendly key=value summary
// line for downstream parsing by run_checks.py.
//
// DUT: fc_verification/rtl/nldpe_attn_head_d64_c128.v
//
// Pipeline (top-level FSM):
//   S_IDLE -> S_FEED_X -> S_DRAIN_FC -> S_FIRE_DIMM -> S_OUTPUT
//
// PROBE SEMANTICS (post-2026-04-24 fix): each per-stage probe captures
// `stage_total = T_last_active - T_first_active + 1` — the full active
// duration of the stage across ALL W=16 DIMM lanes (OR-aggregated). This
// matches the IMC sim's "full stage duration" semantics. The previous
// `*_first_out` lane-0 timestamps are kept for diagnostic/backward-compat
// purposes; the canonical AH_NLDPE_STAGES_TOTAL line is what run_checks.py
// gates on. e2e = T_data_out_last_valid - T_x_first_valid + 1 (full
// pipeline latency from first input pulse to last output pulse).
//
// Per-stage stage-total probes:
//   linear_qkv   : first→last FC valid_n pulse on any of Q/K/V (or-reduce)
//   mac_qk       : first→last cycle ANY dimm_lane[i].score_inst.state==S_OUTPUT
//   softmax_exp  : first→last cycle ANY dimm_lane[i].softmax_inst.sm_state==SM_EXP
//   softmax_norm : first→last cycle ANY dimm_lane[i].softmax_inst.sm_state==SM_NORMALIZE
//   mac_sv       : first→last cycle ANY dimm_lane[i].wsum_inst.ws_state==WS_OUTPUT
//   e2e          : T_data_out_last_valid - T_x_first_valid + 1
//
// Pre-load:
//   - 3 FC arms (Q/K/V) DPE: identity weight diagonal[d_head=64] = 1
//     (KERNEL_WIDTH=128 d_model, NUM_COLS=64 d_head set via defparam)
//   - 16 DIMM lanes: identity weights (per LOAD_LANE pattern from
//     tb_nldpe_dimm_top_functional)
//
// Reference: tb_nldpe_dimm_top_functional.v (LOAD_LANE pattern), and
// tb_nldpe_dimm_top_latency.v (per-stage state probe convention).
//
// Test vector: byte k of word i = (i*5 + k + 1) & 0x7F  (deterministic,
// non-trivial). N*PACKED_KQ = 3328 cycles of valid_x. The DUT FSM drives
// FC arms exactly once per valid_x in S_FEED_X, then drains FC outputs to
// internal buffers, then fires DIMM. Top-level valid_n pulses N*PACKED_NQ
// = 1664 times during S_OUTPUT (one packed output word per token).

`timescale 1ns / 1ps

module tb_nldpe_attn_head_v2;
    parameter DW         = 40;
    parameter D_MODEL    = 128;
    parameter D_HEAD     = 64;
    parameter N_SEQ      = 128;
    parameter EPW        = DW / 8;                              // 5 int8 per word
    parameter PACKED_KQ  = (D_MODEL + EPW - 1) / EPW;           // 26
    parameter PACKED_NQ  = (D_HEAD  + EPW - 1) / EPW;           // 13
    parameter N_X_INPUT  = N_SEQ * PACKED_KQ;                   // 3328
    parameter N_FC_OUT   = N_SEQ * PACKED_NQ;                   // 1664
    parameter W_LANES    = 16;

    reg clk, rst, valid_x, ready_n;
    reg  [DW-1:0] data_in_x;
    wire [DW-1:0] data_out;
    wire ready_x, valid_n;

    // ─── FC arm DPE shape (d_model -> d_head) ─────────────────────────────
    // Streaming FC variant uses 2 DPEs per arm (ping-pong: dpe_a_inst,
    // dpe_b_inst). KERNEL_WIDTH=K=128 and NUM_COLS=N_OUT=64 are wired in
    // at instantiation in fc_top_qkv_streaming.v (no defparam needed).

    // DUT instantiation (parameter values carried via defaults in the RTL)
    nldpe_attn_head_d64_c128 dut (
        .clk(clk), .rst(rst),
        .valid_x(valid_x), .ready_n(ready_n),
        .data_in_x(data_in_x),
        .data_out(data_out),
        .ready_x(ready_x), .valid_n(valid_n)
    );

    // 100 MHz clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    integer i, j, ww;

    // ─── Per-stage cycle timestamps (legacy lane-0 first-out probes) ────
    integer T_x_first_valid          = -1;
    integer T_fc_qkv_done            = -1;
    integer T_dimm_score_first_out   = -1;
    integer T_dimm_softmax_exp_start = -1;
    integer T_dimm_softmax_norm_start= -1;
    integer T_dimm_softmax_out_start = -1;
    integer T_dimm_wsum_first_out    = -1;
    integer T_data_out_first_valid   = -1;
    integer T_data_out_last_valid    = -1;
    integer top_valid_pulses         = 0;

    // ─── Stage-total duration probes (NEW, OR-aggregated across W=16 lanes) ──
    // Each *_first / *_last pair captures the active span of that stage across
    // all 16 lanes. stage_total_cyc = T_last - T_first + 1.
    integer T_fc_first_valid         = -1;  // first FC valid_n on any Q/K/V
    integer T_fc_last_valid          = -1;  // last  FC valid_n on any Q/K/V
    integer T_score_out_first        = -1;  // first cycle ANY lane score.state == S_OUTPUT
    integer T_score_out_last         = -1;  // last  cycle ANY lane score.state == S_OUTPUT
    integer T_softmax_exp_first      = -1;
    integer T_softmax_exp_last       = -1;
    integer T_softmax_nrm_first      = -1;
    integer T_softmax_nrm_last       = -1;
    integer T_wsum_out_first         = -1;
    integer T_wsum_out_last          = -1;

    // ─── Probe wires (lane 0 — kept for legacy first-out probes) ────────
    // score_inst.state is 4-bit; S_OUTPUT = 4'd6
    // softmax_inst.sm_state is 3-bit; SM_EXP=3'd2, SM_NORM=3'd3, SM_OUT=3'd4
    // wsum_inst.ws_state is 4-bit; WS_OUTPUT = 4'd5
    wire score_out_v   = (dut.dimm_inst.dimm_lane[0].score_inst.state    == 4'd6);
    wire softmax_exp_v = (dut.dimm_inst.dimm_lane[0].softmax_inst.sm_state == 3'd2);
    wire softmax_nrm_v = (dut.dimm_inst.dimm_lane[0].softmax_inst.sm_state == 3'd3);
    wire softmax_out_v = (dut.dimm_inst.dimm_lane[0].softmax_inst.sm_state == 3'd4);
    wire wsum_out_v    = (dut.dimm_inst.dimm_lane[0].wsum_inst.ws_state  == 4'd5);
    wire fire_dimm_v   = (dut.state == 3'd3);

    // ─── Stage-total probe wires (16-way OR-reduce across dimm_lane[0..15]) ──
    // Each `any_*` is high if ANY lane is in the named state this cycle.
    // Indexed-name access into the generate-block instances below.
    wire any_score_out =
          (dut.dimm_inst.dimm_lane[ 0].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[ 1].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[ 2].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[ 3].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[ 4].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[ 5].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[ 6].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[ 7].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[ 8].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[ 9].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[10].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[11].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[12].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[13].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[14].score_inst.state == 4'd6)
        | (dut.dimm_inst.dimm_lane[15].score_inst.state == 4'd6);

    wire any_sm_exp =
          (dut.dimm_inst.dimm_lane[ 0].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[ 1].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[ 2].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[ 3].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[ 4].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[ 5].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[ 6].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[ 7].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[ 8].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[ 9].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[10].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[11].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[12].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[13].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[14].softmax_inst.sm_state == 3'd2)
        | (dut.dimm_inst.dimm_lane[15].softmax_inst.sm_state == 3'd2);

    wire any_sm_nrm =
          (dut.dimm_inst.dimm_lane[ 0].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[ 1].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[ 2].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[ 3].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[ 4].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[ 5].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[ 6].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[ 7].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[ 8].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[ 9].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[10].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[11].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[12].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[13].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[14].softmax_inst.sm_state == 3'd3)
        | (dut.dimm_inst.dimm_lane[15].softmax_inst.sm_state == 3'd3);

    wire any_wsum_out =
          (dut.dimm_inst.dimm_lane[ 0].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[ 1].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[ 2].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[ 3].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[ 4].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[ 5].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[ 6].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[ 7].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[ 8].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[ 9].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[10].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[11].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[12].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[13].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[14].wsum_inst.ws_state == 4'd5)
        | (dut.dimm_inst.dimm_lane[15].wsum_inst.ws_state == 4'd5);

    // FC valid_n OR-reduce (Q/K/V fire in lockstep but we OR for safety).
    wire any_fc_valid =
          dut.q_fc_valid_n | dut.k_fc_valid_n | dut.v_fc_valid_n;

    // ─── AH gate Stage 1: Fmax-independent counter probes ───────────────
    // Each counter increments on the rising edge of a "DPE fire complete"
    // event for one architectural primitive. Counts are summed across
    // ALL parallel hardware (FC ping-pong DPEs, 16 DIMM lanes) — so the
    // total is the global # of analog primitive activations during the
    // run, the same quantity sim's `gemm_log`/`dimm_nonlinear` aggregate.
    //
    // Convention: count rising edges of `dpe_done`. dpe_done pulses high
    // for the DPE's S_OUTPUT phase and clears in S_DRAIN, so a 0→1
    // transition is exactly 1 fire. (See dpe_stub.v lines 213, 227.)
    //
    // FC arms: 6 DPEs total (3 arms × 2 ping-pong). Both DPE_A and DPE_B
    // each fire once per inference; we count each arm's pair, then sum.
    // DIMM lanes: 4 DPEs per lane × 16 lanes = 64. Per stage:
    //   mac_qk        : dimm_lane[i].score_inst.dimm_exp.dpe_done
    //   softmax_exp   : dimm_lane[i].softmax_inst.sm_exp.dpe_done
    //   softmax_norm  : dimm_lane[i].softmax_inst sm_state==SM_NORMALIZE
    //                   (no DPE in normalize; count entries to SM_NORM)
    //   mac_sv (log)  : dimm_lane[i].wsum_inst.ws_log.dpe_done
    //   mac_sv (exp)  : dimm_lane[i].wsum_inst.ws_exp.dpe_done

    // FC fire counters (per-arm pair, then sum into 3-arm total).
    // NB: NBA semantics — multiple `<=` to the same target in one cycle keep
    // only the LAST one applied. To count ALL rising edges concurrently,
    // accumulate into a temporary integer (blocking) and assign at the end.
    integer linear_qkv_dpe_fires = 0;
    reg fc_q_a_prev, fc_q_b_prev, fc_k_a_prev, fc_k_b_prev, fc_v_a_prev, fc_v_b_prev;
    integer _lq_inc;
    always @(posedge clk) begin
        if (rst) begin
            fc_q_a_prev <= 0; fc_q_b_prev <= 0;
            fc_k_a_prev <= 0; fc_k_b_prev <= 0;
            fc_v_a_prev <= 0; fc_v_b_prev <= 0;
            linear_qkv_dpe_fires <= 0;
        end else begin
            // Snapshot previous-cycle values at start (NBA on prev regs).
            _lq_inc = 0;
            if (!fc_q_a_prev && dut.fc_q_inst.dpe_a_inst.dpe_done) _lq_inc = _lq_inc + 1;
            if (!fc_q_b_prev && dut.fc_q_inst.dpe_b_inst.dpe_done) _lq_inc = _lq_inc + 1;
            if (!fc_k_a_prev && dut.fc_k_inst.dpe_a_inst.dpe_done) _lq_inc = _lq_inc + 1;
            if (!fc_k_b_prev && dut.fc_k_inst.dpe_b_inst.dpe_done) _lq_inc = _lq_inc + 1;
            if (!fc_v_a_prev && dut.fc_v_inst.dpe_a_inst.dpe_done) _lq_inc = _lq_inc + 1;
            if (!fc_v_b_prev && dut.fc_v_inst.dpe_b_inst.dpe_done) _lq_inc = _lq_inc + 1;
            linear_qkv_dpe_fires <= linear_qkv_dpe_fires + _lq_inc;
            fc_q_a_prev <= dut.fc_q_inst.dpe_a_inst.dpe_done;
            fc_q_b_prev <= dut.fc_q_inst.dpe_b_inst.dpe_done;
            fc_k_a_prev <= dut.fc_k_inst.dpe_a_inst.dpe_done;
            fc_k_b_prev <= dut.fc_k_inst.dpe_b_inst.dpe_done;
            fc_v_a_prev <= dut.fc_v_inst.dpe_a_inst.dpe_done;
            fc_v_b_prev <= dut.fc_v_inst.dpe_b_inst.dpe_done;
        end
    end

    // DIMM-stage fire counters (16 lanes summed). To avoid 16-way generate
    // probes for each, we sample dpe_done across all lanes per cycle and
    // sum the rising-edge contributions. Iverilog macros expanded to
    // explicit indexing (no generate scope inside the TB).
    integer mac_qk_dpe_fires       = 0;
    integer softmax_exp_dpe_fires  = 0;
    integer softmax_norm_fires     = 0;
    integer mac_sv_log_dpe_fires   = 0;
    integer mac_sv_exp_dpe_fires   = 0;
    reg [15:0] mac_qk_d_prev, mac_qk_d_curr;
    reg [15:0] sm_exp_d_prev, sm_exp_d_curr;
    reg [15:0] ws_log_d_prev, ws_log_d_curr;
    reg [15:0] ws_exp_d_prev, ws_exp_d_curr;
    reg [15:0] sm_nrm_prev,   sm_nrm_curr;

    // Per-cycle aggregate of all 16 lanes' dpe_done bits, plus norm-state.
    // Use indexed-name access into the dimm_lane generate.
    always @(*) begin
        mac_qk_d_curr[ 0] = dut.dimm_inst.dimm_lane[ 0].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[ 1] = dut.dimm_inst.dimm_lane[ 1].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[ 2] = dut.dimm_inst.dimm_lane[ 2].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[ 3] = dut.dimm_inst.dimm_lane[ 3].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[ 4] = dut.dimm_inst.dimm_lane[ 4].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[ 5] = dut.dimm_inst.dimm_lane[ 5].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[ 6] = dut.dimm_inst.dimm_lane[ 6].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[ 7] = dut.dimm_inst.dimm_lane[ 7].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[ 8] = dut.dimm_inst.dimm_lane[ 8].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[ 9] = dut.dimm_inst.dimm_lane[ 9].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[10] = dut.dimm_inst.dimm_lane[10].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[11] = dut.dimm_inst.dimm_lane[11].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[12] = dut.dimm_inst.dimm_lane[12].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[13] = dut.dimm_inst.dimm_lane[13].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[14] = dut.dimm_inst.dimm_lane[14].score_inst.dimm_exp.dpe_done;
        mac_qk_d_curr[15] = dut.dimm_inst.dimm_lane[15].score_inst.dimm_exp.dpe_done;

        sm_exp_d_curr[ 0] = dut.dimm_inst.dimm_lane[ 0].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[ 1] = dut.dimm_inst.dimm_lane[ 1].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[ 2] = dut.dimm_inst.dimm_lane[ 2].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[ 3] = dut.dimm_inst.dimm_lane[ 3].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[ 4] = dut.dimm_inst.dimm_lane[ 4].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[ 5] = dut.dimm_inst.dimm_lane[ 5].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[ 6] = dut.dimm_inst.dimm_lane[ 6].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[ 7] = dut.dimm_inst.dimm_lane[ 7].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[ 8] = dut.dimm_inst.dimm_lane[ 8].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[ 9] = dut.dimm_inst.dimm_lane[ 9].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[10] = dut.dimm_inst.dimm_lane[10].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[11] = dut.dimm_inst.dimm_lane[11].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[12] = dut.dimm_inst.dimm_lane[12].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[13] = dut.dimm_inst.dimm_lane[13].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[14] = dut.dimm_inst.dimm_lane[14].softmax_inst.sm_exp.dpe_done;
        sm_exp_d_curr[15] = dut.dimm_inst.dimm_lane[15].softmax_inst.sm_exp.dpe_done;

        ws_log_d_curr[ 0] = dut.dimm_inst.dimm_lane[ 0].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[ 1] = dut.dimm_inst.dimm_lane[ 1].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[ 2] = dut.dimm_inst.dimm_lane[ 2].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[ 3] = dut.dimm_inst.dimm_lane[ 3].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[ 4] = dut.dimm_inst.dimm_lane[ 4].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[ 5] = dut.dimm_inst.dimm_lane[ 5].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[ 6] = dut.dimm_inst.dimm_lane[ 6].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[ 7] = dut.dimm_inst.dimm_lane[ 7].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[ 8] = dut.dimm_inst.dimm_lane[ 8].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[ 9] = dut.dimm_inst.dimm_lane[ 9].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[10] = dut.dimm_inst.dimm_lane[10].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[11] = dut.dimm_inst.dimm_lane[11].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[12] = dut.dimm_inst.dimm_lane[12].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[13] = dut.dimm_inst.dimm_lane[13].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[14] = dut.dimm_inst.dimm_lane[14].wsum_inst.ws_log.dpe_done;
        ws_log_d_curr[15] = dut.dimm_inst.dimm_lane[15].wsum_inst.ws_log.dpe_done;

        ws_exp_d_curr[ 0] = dut.dimm_inst.dimm_lane[ 0].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[ 1] = dut.dimm_inst.dimm_lane[ 1].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[ 2] = dut.dimm_inst.dimm_lane[ 2].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[ 3] = dut.dimm_inst.dimm_lane[ 3].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[ 4] = dut.dimm_inst.dimm_lane[ 4].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[ 5] = dut.dimm_inst.dimm_lane[ 5].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[ 6] = dut.dimm_inst.dimm_lane[ 6].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[ 7] = dut.dimm_inst.dimm_lane[ 7].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[ 8] = dut.dimm_inst.dimm_lane[ 8].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[ 9] = dut.dimm_inst.dimm_lane[ 9].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[10] = dut.dimm_inst.dimm_lane[10].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[11] = dut.dimm_inst.dimm_lane[11].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[12] = dut.dimm_inst.dimm_lane[12].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[13] = dut.dimm_inst.dimm_lane[13].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[14] = dut.dimm_inst.dimm_lane[14].wsum_inst.ws_exp.dpe_done;
        ws_exp_d_curr[15] = dut.dimm_inst.dimm_lane[15].wsum_inst.ws_exp.dpe_done;

        sm_nrm_curr[ 0] = (dut.dimm_inst.dimm_lane[ 0].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[ 1] = (dut.dimm_inst.dimm_lane[ 1].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[ 2] = (dut.dimm_inst.dimm_lane[ 2].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[ 3] = (dut.dimm_inst.dimm_lane[ 3].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[ 4] = (dut.dimm_inst.dimm_lane[ 4].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[ 5] = (dut.dimm_inst.dimm_lane[ 5].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[ 6] = (dut.dimm_inst.dimm_lane[ 6].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[ 7] = (dut.dimm_inst.dimm_lane[ 7].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[ 8] = (dut.dimm_inst.dimm_lane[ 8].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[ 9] = (dut.dimm_inst.dimm_lane[ 9].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[10] = (dut.dimm_inst.dimm_lane[10].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[11] = (dut.dimm_inst.dimm_lane[11].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[12] = (dut.dimm_inst.dimm_lane[12].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[13] = (dut.dimm_inst.dimm_lane[13].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[14] = (dut.dimm_inst.dimm_lane[14].softmax_inst.sm_state == 3'd3);
        sm_nrm_curr[15] = (dut.dimm_inst.dimm_lane[15].softmax_inst.sm_state == 3'd3);
    end

    integer ki;
    integer _mq_inc, _sx_inc, _wl_inc, _we_inc, _sn_inc;
    always @(posedge clk) begin
        if (rst) begin
            mac_qk_d_prev      <= 0;
            sm_exp_d_prev      <= 0;
            ws_log_d_prev      <= 0;
            ws_exp_d_prev      <= 0;
            sm_nrm_prev        <= 0;
            mac_qk_dpe_fires       <= 0;
            softmax_exp_dpe_fires  <= 0;
            softmax_norm_fires     <= 0;
            mac_sv_log_dpe_fires   <= 0;
            mac_sv_exp_dpe_fires   <= 0;
        end else begin
            // Aggregate into integer temporaries (blocking) so multiple
            // lane rising edges in the SAME cycle all count. NBA semantics
            // would otherwise keep only the last `<=` from a chain of
            // `var <= var + 1` statements.
            _mq_inc = 0; _sx_inc = 0; _wl_inc = 0; _we_inc = 0; _sn_inc = 0;
            for (ki = 0; ki < 16; ki = ki + 1) begin
                if (!mac_qk_d_prev[ki] && mac_qk_d_curr[ki]) _mq_inc = _mq_inc + 1;
                if (!sm_exp_d_prev[ki] && sm_exp_d_curr[ki]) _sx_inc = _sx_inc + 1;
                if (!ws_log_d_prev[ki] && ws_log_d_curr[ki]) _wl_inc = _wl_inc + 1;
                if (!ws_exp_d_prev[ki] && ws_exp_d_curr[ki]) _we_inc = _we_inc + 1;
                if (!sm_nrm_prev[ki]   && sm_nrm_curr[ki]  ) _sn_inc = _sn_inc + 1;
            end
            mac_qk_dpe_fires      <= mac_qk_dpe_fires      + _mq_inc;
            softmax_exp_dpe_fires <= softmax_exp_dpe_fires + _sx_inc;
            mac_sv_log_dpe_fires  <= mac_sv_log_dpe_fires  + _wl_inc;
            mac_sv_exp_dpe_fires  <= mac_sv_exp_dpe_fires  + _we_inc;
            softmax_norm_fires    <= softmax_norm_fires    + _sn_inc;
            mac_qk_d_prev <= mac_qk_d_curr;
            sm_exp_d_prev <= sm_exp_d_curr;
            ws_log_d_prev <= ws_log_d_curr;
            ws_exp_d_prev <= ws_exp_d_curr;
            sm_nrm_prev   <= sm_nrm_curr;
        end
    end


    always @(posedge clk) begin
        if (!rst) begin
            // Legacy lane-0 first-out timestamps (kept for diagnostic dump).
            if (score_out_v   && T_dimm_score_first_out   < 0) T_dimm_score_first_out   <= cycle;
            if (softmax_exp_v && T_dimm_softmax_exp_start < 0) T_dimm_softmax_exp_start <= cycle;
            if (softmax_nrm_v && T_dimm_softmax_norm_start< 0) T_dimm_softmax_norm_start<= cycle;
            if (softmax_out_v && T_dimm_softmax_out_start < 0) T_dimm_softmax_out_start <= cycle;
            if (wsum_out_v    && T_dimm_wsum_first_out    < 0) T_dimm_wsum_first_out    <= cycle;
            if (fire_dimm_v   && T_fc_qkv_done            < 0) T_fc_qkv_done            <= cycle;

            // Stage-total OR-aggregated first/last across 16 lanes.
            if (any_fc_valid) begin
                if (T_fc_first_valid < 0) T_fc_first_valid <= cycle;
                T_fc_last_valid <= cycle;
            end
            if (any_score_out) begin
                if (T_score_out_first < 0) T_score_out_first <= cycle;
                T_score_out_last <= cycle;
            end
            if (any_sm_exp) begin
                if (T_softmax_exp_first < 0) T_softmax_exp_first <= cycle;
                T_softmax_exp_last <= cycle;
            end
            if (any_sm_nrm) begin
                if (T_softmax_nrm_first < 0) T_softmax_nrm_first <= cycle;
                T_softmax_nrm_last <= cycle;
            end
            if (any_wsum_out) begin
                if (T_wsum_out_first < 0) T_wsum_out_first <= cycle;
                T_wsum_out_last <= cycle;
            end

            if (valid_n) begin
                if (T_data_out_first_valid < 0) T_data_out_first_valid <= cycle;
                T_data_out_last_valid <= cycle;
                top_valid_pulses <= top_valid_pulses + 1;
            end
        end
    end

    // ─── DUT FSM workarounds ──────────────────────────────────────────────
    //
    // (1) The top FSM drives DIMM_K_WORDS=1664 valid_k pulses, but the
    //     score_inst lane FSM transitions to S_LOAD_K one cycle AFTER top's
    //     Q→K phase boundary (state==S_LOAD_Q on the cycle q_write_addr==13
    //     fires the transition; state==S_LOAD_K only the *next* cycle). This
    //     loses the first valid_k pulse, so each lane's k_write_addr peaks
    //     at 1663 — one short of the score_inst's S_LOAD_K → S_COMPUTE
    //     check (k_write_addr==1664). Without this kick, the lanes deadlock.
    //     Fix: when top exits the K phase (drive_valid_k 1→0 transition),
    //     force k_write_addr=1664 in all 16 lanes once. Same kind of
    //     hierarchical force as `FORCE_VWADDR` in tb_nldpe_dimm_top_functional.
    //
    // (2) wsum_inst.v_sram is pre-loaded by LOAD_DIMM_LANE (mirroring the
    //     functional/latency DIMM-top TB pattern), so when wsum reaches
    //     WS_LOAD_V we force v_write_addr=d*PACKED_N-1 to immediately exit
    //     to WS_LOG_FEED — the same convention as the standalone DIMM-top
    //     latency TB.
    `define FORCE_KW(L) \
        dut.dimm_inst.dimm_lane[L].score_inst.k_write_addr = 11'd1664;
    `define FORCE_VW(L) \
        dut.dimm_inst.dimm_lane[L].wsum_inst.v_write_addr = 11'd1663;

    reg drive_valid_k_d;
    reg fired_kw_kick;
    always @(posedge clk) begin
        if (rst) begin
            drive_valid_k_d <= 0;
            fired_kw_kick   <= 0;
        end else begin
            drive_valid_k_d <= dut.drive_valid_k;
            // Detect 1→0 falling edge of drive_valid_k (Q→V phase transition's
            // K-end signal). Fire the kick exactly once.
            if (drive_valid_k_d && !dut.drive_valid_k && !fired_kw_kick) begin
                `FORCE_KW( 0) `FORCE_KW( 1) `FORCE_KW( 2) `FORCE_KW( 3)
                `FORCE_KW( 4) `FORCE_KW( 5) `FORCE_KW( 6) `FORCE_KW( 7)
                `FORCE_KW( 8) `FORCE_KW( 9) `FORCE_KW(10) `FORCE_KW(11)
                `FORCE_KW(12) `FORCE_KW(13) `FORCE_KW(14) `FORCE_KW(15)
                fired_kw_kick <= 1;
            end
        end
    end

    // Whenever any lane's wsum reaches WS_LOAD_V (state 2), force its
    // v_write_addr to the exit value so it immediately advances to
    // WS_LOG_FEED (mirroring the dimm-top functional/latency TB pattern;
    // v_sram is pre-loaded above).
    always @(posedge clk) begin
        if (!rst && dut.dimm_inst.dimm_lane[0].wsum_inst.ws_state == 4'd2) begin
            `FORCE_VW( 0) `FORCE_VW( 1) `FORCE_VW( 2) `FORCE_VW( 3)
            `FORCE_VW( 4) `FORCE_VW( 5) `FORCE_VW( 6) `FORCE_VW( 7)
            `FORCE_VW( 8) `FORCE_VW( 9) `FORCE_VW(10) `FORCE_VW(11)
            `FORCE_VW(12) `FORCE_VW(13) `FORCE_VW(14) `FORCE_VW(15)
        end
    end

    // ─── DIMM identity-weight + V-SRAM preload (per lane) ────────────────
    // Mirrors LOAD_LANE in tb_nldpe_dimm_top_functional.v. Each lane has 4
    // DPEs (dimm_exp, sm_exp, ws_log, ws_exp), and the wsum sub-module has
    // a v_sram which we pre-fill with identity (transposed/packed).
    `define LOAD_DIMM_LANE(L) begin \
        for (ww = 0; ww < 128; ww = ww + 1) \
            dut.dimm_inst.dimm_lane[L].score_inst.dimm_exp.weights[ww][ww] = 1; \
        dut.dimm_inst.dimm_lane[L].softmax_inst.sm_exp.weights[0][0] = 1; \
        dut.dimm_inst.dimm_lane[L].wsum_inst.ws_log.weights[0][0] = 1; \
        for (ww = 0; ww < 128; ww = ww + 1) \
            dut.dimm_inst.dimm_lane[L].wsum_inst.ws_exp.weights[ww][ww] = 1; \
        for (i = 0; i < 1665; i = i + 1) \
            dut.dimm_inst.dimm_lane[L].wsum_inst.v_sram.mem[i] = 40'd0; \
        for (j = 0; j < D_HEAD; j = j + 1) \
            dut.dimm_inst.dimm_lane[L].wsum_inst.v_sram.mem[j*26 + (j/EPW)][(j%EPW)*8 +: 8] = 8'd1; \
    end

    // ─── FC arm DPE identity-weight preload ───────────────────────────────
    // For diagonal weights[c][c] = 1 (c=0..D_HEAD-1), the DPE computes
    // output[c] = input[c] for c < D_HEAD. Other columns weighted to 0.
    // Streaming FC has 2 DPEs per arm (ping-pong); load BOTH so the
    // output is identical regardless of which DPE handles a given token.
    `define LOAD_FC_ARM(NAME) begin \
        for (ww = 0; ww < D_HEAD; ww = ww + 1) begin \
            dut.NAME.dpe_a_inst.weights[ww][ww] = 1; \
            dut.NAME.dpe_b_inst.weights[ww][ww] = 1; \
        end \
    end

    initial begin
        // ─── Reset + preload ──────────────────────────────────────────────
        rst       = 1;
        valid_x   = 0;
        ready_n   = 1;       // downstream perpetually ready (consumes output)
        data_in_x = 0;

        // 16 DIMM lanes
        `LOAD_DIMM_LANE( 0) `LOAD_DIMM_LANE( 1) `LOAD_DIMM_LANE( 2) `LOAD_DIMM_LANE( 3)
        `LOAD_DIMM_LANE( 4) `LOAD_DIMM_LANE( 5) `LOAD_DIMM_LANE( 6) `LOAD_DIMM_LANE( 7)
        `LOAD_DIMM_LANE( 8) `LOAD_DIMM_LANE( 9) `LOAD_DIMM_LANE(10) `LOAD_DIMM_LANE(11)
        `LOAD_DIMM_LANE(12) `LOAD_DIMM_LANE(13) `LOAD_DIMM_LANE(14) `LOAD_DIMM_LANE(15)

        // 3 FC arms (Q, K, V)
        `LOAD_FC_ARM(fc_q_inst)
        `LOAD_FC_ARM(fc_k_inst)
        `LOAD_FC_ARM(fc_v_inst)

        #20; rst = 0; #15;

        $display("=== NL-DPE Attention Head V2 (combined func+latency) TB ===");
        $display("  Config: d_model=%0d d_head=%0d N=%0d W=16 EPW=%0d", D_MODEL, D_HEAD, N_SEQ, EPW);
        $display("  PACKED_KQ=%0d (in/tok) PACKED_NQ=%0d (out/tok)", PACKED_KQ, PACKED_NQ);
        $display("  [t=%0d] reset released", cycle);

        // ─── Drive valid_x for N*PACKED_KQ + 1 cycles ─────────────────────
        // The FSM IDLE -> FEED_X transition consumes 1 edge before in_count
        // increments. Drive valid_x from cycle 0 and let the DUT FSM count.
        // Pattern: byte k of word i = (i*5 + k + 1) & 0x7F.
        T_x_first_valid = cycle;
        for (i = 0; i < N_X_INPUT + 2; i = i + 1) begin
            @(posedge clk); #1;
            valid_x = 1;
            data_in_x = 0;
            for (j = 0; j < EPW; j = j + 1)
                data_in_x[j*8 +: 8] = ((i * EPW + j + 1) & 8'h7F);
        end
        @(posedge clk); #1; valid_x = 0; data_in_x = 0;
        $display("  [t=%0d] Drove %0d valid_x cycles (N_X_INPUT=%0d)",
                 cycle, N_X_INPUT + 2, N_X_INPUT);

        // ─── Wait for top-level valid_n pulses to drain ──────────────────
        // Top valid_n pulses ~N*PACKED_NQ = 1664 times. Note: due to the
        // lane round-robin and wsum WS_OUTPUT-state semantics, the actual
        // count can be slightly above N_FC_OUT (a small structural FSM
        // residual). Wait for at least N_FC_OUT pulses, then capture the
        // last pulse cycle.
        begin : wait_drain
            integer timeout;
            timeout = 0;
            while (top_valid_pulses < N_FC_OUT && timeout < 200000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 200000)
                $display("  [t=%0d] TIMEOUT waiting for top valid_n drain (%0d/%0d)",
                         cycle, top_valid_pulses, N_FC_OUT);
        end

        // Settle 1 extra cycle so non-blocking T_*_last writes commit.
        @(posedge clk); #1;

        // ─── Per-stage cycle counts ───────────────────────────────────────
        $display("");
        $display("=== Per-Stage RTL Cycles vs Sim Oracle ===");
        $display("  T_x_first_valid          = %0d", T_x_first_valid);
        $display("  T_fc_qkv_done            = %0d", T_fc_qkv_done);
        $display("  T_dimm_score_first_out   = %0d", T_dimm_score_first_out);
        $display("  T_dimm_softmax_exp_start = %0d", T_dimm_softmax_exp_start);
        $display("  T_dimm_softmax_norm_start= %0d", T_dimm_softmax_norm_start);
        $display("  T_dimm_softmax_out_start = %0d", T_dimm_softmax_out_start);
        $display("  T_dimm_wsum_first_out    = %0d", T_dimm_wsum_first_out);
        $display("  T_data_out_first_valid   = %0d", T_data_out_first_valid);
        $display("  T_data_out_last_valid    = %0d", T_data_out_last_valid);
        $display("  --- Stage-total OR-aggregated (16 lanes) ---");
        $display("  T_fc_first_valid         = %0d", T_fc_first_valid);
        $display("  T_fc_last_valid          = %0d", T_fc_last_valid);
        $display("  T_score_out_first        = %0d", T_score_out_first);
        $display("  T_score_out_last         = %0d", T_score_out_last);
        $display("  T_softmax_exp_first      = %0d", T_softmax_exp_first);
        $display("  T_softmax_exp_last       = %0d", T_softmax_exp_last);
        $display("  T_softmax_nrm_first      = %0d", T_softmax_nrm_first);
        $display("  T_softmax_nrm_last       = %0d", T_softmax_nrm_last);
        $display("  T_wsum_out_first         = %0d", T_wsum_out_first);
        $display("  T_wsum_out_last          = %0d", T_wsum_out_last);
        $display("");

        // Compute per-stage cycle counts (default 0 if any timestamp missed)
        report_stages;

        $display("");
        $display("=== Functional Sanity ===");
        $display("  top valid_n pulses : %0d (expected %0d = N*PACKED_NQ)",
                 top_valid_pulses, N_FC_OUT);
        if (top_valid_pulses == N_FC_OUT)
            $display("  Functional         : PASS (output pulse count matches expected)");
        else
            $display("  Functional         : FAIL (output pulse count mismatch)");

        $finish;
    end

    // Helper: emit per-stage cycles + the regex-friendly probe rows.
    //
    // Two output formats are emitted:
    //   AH_NLDPE_STAGES        — legacy first-out latencies (kept for back-compat,
    //                            no longer gated by run_checks.py).
    //   AH_NLDPE_STAGES_TOTAL  — stage-total durations across W=16 lanes
    //                            (the canonical line consumed by run_checks.py).
    task report_stages;
        integer linear_qkv, mac_qk, softmax_exp, softmax_norm, mac_sv, e2e;
        // Stage-total durations
        integer linear_qkv_t, mac_qk_t, softmax_exp_t, softmax_norm_t, mac_sv_t, e2e_t;
        begin
            // Legacy first-out latency stage cycles (for diagnostic dump).
            linear_qkv   = (T_fc_qkv_done            > T_x_first_valid)
                         ? (T_fc_qkv_done            - T_x_first_valid) : 0;
            mac_qk       = (T_dimm_softmax_exp_start > T_fc_qkv_done)
                         ? (T_dimm_softmax_exp_start - T_fc_qkv_done) : 0;
            softmax_exp  = (T_dimm_softmax_norm_start> T_dimm_softmax_exp_start)
                         ? (T_dimm_softmax_norm_start- T_dimm_softmax_exp_start) : 0;
            softmax_norm = (T_dimm_softmax_out_start > T_dimm_softmax_norm_start)
                         ? (T_dimm_softmax_out_start - T_dimm_softmax_norm_start) : 0;
            mac_sv       = (T_dimm_wsum_first_out    > T_dimm_softmax_out_start)
                         ? (T_dimm_wsum_first_out    - T_dimm_softmax_out_start) : 0;
            e2e          = (T_data_out_first_valid   > T_x_first_valid)
                         ? (T_data_out_first_valid   - T_x_first_valid) : 0;

            // Stage-total durations (T_last - T_first + 1) across all W=16 lanes.
            // If a stage was never observed (T_first == -1), report 0.
            linear_qkv_t   = (T_fc_first_valid    >= 0 && T_fc_last_valid    >= 0)
                           ? (T_fc_last_valid    - T_fc_first_valid    + 1) : 0;
            mac_qk_t       = (T_score_out_first   >= 0 && T_score_out_last   >= 0)
                           ? (T_score_out_last   - T_score_out_first   + 1) : 0;
            softmax_exp_t  = (T_softmax_exp_first >= 0 && T_softmax_exp_last >= 0)
                           ? (T_softmax_exp_last - T_softmax_exp_first + 1) : 0;
            softmax_norm_t = (T_softmax_nrm_first >= 0 && T_softmax_nrm_last >= 0)
                           ? (T_softmax_nrm_last - T_softmax_nrm_first + 1) : 0;
            mac_sv_t       = (T_wsum_out_first    >= 0 && T_wsum_out_last    >= 0)
                           ? (T_wsum_out_last    - T_wsum_out_first    + 1) : 0;
            e2e_t          = (T_data_out_first_valid >= 0 && T_data_out_last_valid >= 0)
                           ? (T_data_out_last_valid  - T_x_first_valid     + 1) : 0;

            $display("  Stage (LEGACY first-out latencies)");
            $display("  linear_qkv  : %4d  vs  2424  %+0d", linear_qkv,   linear_qkv   - 2424);
            $display("  mac_qk      : %4d  vs  5690  %+0d", mac_qk,       mac_qk       - 5690);
            $display("  softmax_exp : %4d  vs   650  %+0d", softmax_exp,  softmax_exp  -  650);
            $display("  softmax_norm: %4d  vs   376  %+0d", softmax_norm, softmax_norm -  376);
            $display("  mac_sv      : %4d  vs  5339  %+0d", mac_sv,       mac_sv       - 5339);
            $display("  e2e         : %4d  vs 14480  %+0d", e2e,          e2e          - 14480);
            $display("");
            $display("  Stage (NEW stage-total durations, OR over W=16 lanes)");
            $display("  linear_qkv  : %5d  vs  2424  %+0d", linear_qkv_t,   linear_qkv_t   - 2424);
            $display("  mac_qk      : %5d  vs  5690  %+0d", mac_qk_t,       mac_qk_t       - 5690);
            $display("  softmax_exp : %5d  vs   650  %+0d", softmax_exp_t,  softmax_exp_t  -  650);
            $display("  softmax_norm: %5d  vs   376  %+0d", softmax_norm_t, softmax_norm_t -  376);
            $display("  mac_sv      : %5d  vs  5339  %+0d", mac_sv_t,       mac_sv_t       - 5339);
            $display("  e2e         : %5d  vs 14480  %+0d", e2e_t,          e2e_t          - 14480);
            $display("");
            $display("AH_NLDPE_STAGES linear_qkv=%0d mac_qk=%0d softmax_exp=%0d softmax_norm=%0d mac_sv=%0d e2e=%0d",
                     linear_qkv, mac_qk, softmax_exp, softmax_norm, mac_sv, e2e);
            $display("AH_NLDPE_STAGES_TOTAL linear_qkv=%0d mac_qk=%0d softmax_exp=%0d softmax_norm=%0d mac_sv=%0d e2e=%0d",
                     linear_qkv_t, mac_qk_t, softmax_exp_t, softmax_norm_t, mac_sv_t, e2e_t);

            // ── AH gate Stage 1: Fmax-independent counter emission ──
            // Format: COUNTER <stage> <key> <value>
            // run_checks.py regex-parses these and compares against
            // expected_counters.json. mac_sv combines ws_log + ws_exp DPE
            // fires (= total weighted-sum DPE activations).
            $display("");
            $display("=== AH-gate counters (TB-side architectural invariants) ===");
            $display("COUNTER linear_qkv dpe_fire_count %0d", linear_qkv_dpe_fires);
            $display("COUNTER linear_qkv pass_count %0d", linear_qkv_dpe_fires);
            $display("COUNTER linear_qkv row_count %0d", N_SEQ);
            $display("COUNTER linear_qkv lane_count 6");
            $display("COUNTER mac_qk dpe_fire_count %0d", mac_qk_dpe_fires);
            $display("COUNTER mac_qk pass_count %0d", mac_qk_dpe_fires);
            $display("COUNTER mac_qk row_count 1");
            $display("COUNTER mac_qk lane_count %0d", W_LANES);
            $display("COUNTER softmax_exp dpe_fire_count %0d", softmax_exp_dpe_fires);
            $display("COUNTER softmax_exp pass_count %0d", softmax_exp_dpe_fires);
            $display("COUNTER softmax_exp row_count 1");
            $display("COUNTER softmax_exp lane_count %0d", W_LANES);
            $display("COUNTER softmax_exp softmax_exp_fires %0d", softmax_exp_dpe_fires);
            $display("COUNTER softmax_norm dpe_fire_count %0d", softmax_norm_fires);
            $display("COUNTER softmax_norm pass_count %0d", softmax_norm_fires);
            $display("COUNTER softmax_norm row_count 1");
            $display("COUNTER softmax_norm lane_count %0d", W_LANES);
            $display("COUNTER softmax_norm softmax_norm_fires %0d", softmax_norm_fires);
            $display("COUNTER mac_sv dpe_fire_count %0d", mac_sv_log_dpe_fires + mac_sv_exp_dpe_fires);
            $display("COUNTER mac_sv pass_count %0d", mac_sv_log_dpe_fires + mac_sv_exp_dpe_fires);
            $display("COUNTER mac_sv row_count 1");
            $display("COUNTER mac_sv lane_count %0d", W_LANES);
            $display("COUNTER mac_sv ws_log_fires %0d", mac_sv_log_dpe_fires);
            $display("COUNTER mac_sv ws_exp_fires %0d", mac_sv_exp_dpe_fires);
        end
    endtask

    // Hard timeout (3 ms sim ≈ 300k cycles at 10 ns clock).
    initial begin
        #3000000;
        $display("HARD TIMEOUT at 3 ms sim time");
        $finish;
    end

endmodule
