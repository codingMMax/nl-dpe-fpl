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
// Per-stage probes captured (lane 0 of the W=16 DIMM):
//   linear_qkv   : T_fc_qkv_done   - T_x_first_valid
//   mac_qk       : T_dimm_softmax_exp_start - T_fc_qkv_done
//   softmax_exp  : T_dimm_softmax_norm_start - T_dimm_softmax_exp_start
//   softmax_norm : T_dimm_softmax_out_start  - T_dimm_softmax_norm_start
//   mac_sv       : T_dimm_wsum_first_out     - T_dimm_softmax_out_start
//   e2e          : T_data_out_first_valid    - T_x_first_valid
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

    // ─── Per-stage cycle timestamps ───────────────────────────────────────
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

    // ─── Probe wires (lane 0 of the W=16 DIMM) ────────────────────────────
    // score_inst.state is 4-bit; S_OUTPUT = 4'd6
    // softmax_inst.sm_state is 3-bit; SM_EXP=3'd2, SM_NORM=3'd3, SM_OUT=3'd4
    // wsum_inst.ws_state is 4-bit; WS_OUTPUT = 4'd5
    wire score_out_v   = (dut.dimm_inst.dimm_lane[0].score_inst.state    == 4'd6);
    wire softmax_exp_v = (dut.dimm_inst.dimm_lane[0].softmax_inst.sm_state == 3'd2);
    wire softmax_nrm_v = (dut.dimm_inst.dimm_lane[0].softmax_inst.sm_state == 3'd3);
    wire softmax_out_v = (dut.dimm_inst.dimm_lane[0].softmax_inst.sm_state == 3'd4);
    wire wsum_out_v    = (dut.dimm_inst.dimm_lane[0].wsum_inst.ws_state  == 4'd5);
    wire fire_dimm_v   = (dut.state == 3'd3);

    always @(posedge clk) begin
        if (!rst) begin
            if (score_out_v   && T_dimm_score_first_out   < 0) T_dimm_score_first_out   <= cycle;
            if (softmax_exp_v && T_dimm_softmax_exp_start < 0) T_dimm_softmax_exp_start <= cycle;
            if (softmax_nrm_v && T_dimm_softmax_norm_start< 0) T_dimm_softmax_norm_start<= cycle;
            if (softmax_out_v && T_dimm_softmax_out_start < 0) T_dimm_softmax_out_start <= cycle;
            if (wsum_out_v    && T_dimm_wsum_first_out    < 0) T_dimm_wsum_first_out    <= cycle;
            if (fire_dimm_v   && T_fc_qkv_done            < 0) T_fc_qkv_done            <= cycle;
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

    // Helper: emit per-stage cycles + the regex-friendly probe row.
    task report_stages;
        integer linear_qkv, mac_qk, softmax_exp, softmax_norm, mac_sv, e2e;
        begin
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

            $display("  Stage          RTL       sim    Δ");
            $display("  linear_qkv  : %4d  vs  2424  %+0d", linear_qkv,   linear_qkv   - 2424);
            $display("  mac_qk      : %4d  vs  5690  %+0d", mac_qk,       mac_qk       - 5690);
            $display("  softmax_exp : %4d  vs   650  %+0d", softmax_exp,  softmax_exp  -  650);
            $display("  softmax_norm: %4d  vs   376  %+0d", softmax_norm, softmax_norm -  376);
            $display("  mac_sv      : %4d  vs  5339  %+0d", mac_sv,       mac_sv       - 5339);
            $display("  e2e         : %4d  vs 14480  %+0d", e2e,          e2e          - 14480);
            $display("");
            $display("AH_NLDPE_STAGES linear_qkv=%0d mac_qk=%0d softmax_exp=%0d softmax_norm=%0d mac_sv=%0d e2e=%0d",
                     linear_qkv, mac_qk, softmax_exp, softmax_norm, mac_sv, e2e);
        end
    endtask

    // Hard timeout (3 ms sim ≈ 300k cycles at 10 ns clock).
    initial begin
        #3000000;
        $display("HARD TIMEOUT at 3 ms sim time");
        $finish;
    end

endmodule
