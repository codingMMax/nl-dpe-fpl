// NL-DPE Full DIMM Top — Latency Measurement (W=16, N=128, d=64)
//
// Measures end-to-end cycle count from compute-phase start to final valid_n
// pulse.  Uses the same preload + force pattern as the functional TB.
//
// Reports two numbers:
//   t_compute_plus_output : FEED_QK entry -> lane 0 wsum_inst.ws_state == WS_OUTPUT
//   total_rtl_cycles      : reset deassertion -> lane 0 ws_state == WS_OUTPUT
//
// These are compared to the simulator's attention model prediction in Phase J.

`timescale 1ns / 1ps

module tb_nldpe_dimm_top_latency;

    parameter DW = 40;
    parameter N = 128;
    parameter D = 64;
    parameter W = 16;
    parameter EPW = DW / 8;
    parameter PD = (D + EPW - 1) / EPW;

    reg clk, rst, valid_q, valid_k, valid_v, ready_n;
    reg  [DW-1:0] data_in_q, data_in_k, data_in_v;
    wire [DW-1:0] data_out;
    wire ready_q, ready_k, ready_v, valid_n;

    nldpe_dimm_top #(.N(N), .D(D), .W(W), .DATA_WIDTH(DW)) dut (
        .clk(clk), .rst(rst),
        .valid_q(valid_q), .valid_k(valid_k), .valid_v(valid_v), .ready_n(ready_n),
        .data_in_q(data_in_q), .data_in_k(data_in_k), .data_in_v(data_in_v),
        .data_out(data_out),
        .ready_q(ready_q), .ready_k(ready_k), .ready_v(ready_v),
        .valid_n(valid_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    integer i, j, m, ww;
    integer start_cyc, feed_qk_cyc, end_cyc;
    // Per-stage timestamps (lane 0)
    integer score_start_cyc, score_end_cyc;
    integer softmax_start_cyc, softmax_end_cyc;
    integer wsum_start_cyc, wsum_end_cyc;

    // Detect first rising edge of each stage's valid_n on lane 0.
    reg score_caught, softmax_caught, wsum_caught;
    always @(posedge clk) begin
        if (rst) begin
            score_start_cyc <= 0; score_end_cyc <= 0;
            softmax_start_cyc <= 0; softmax_end_cyc <= 0;
            wsum_start_cyc <= 0; wsum_end_cyc <= 0;
            score_caught <= 0; softmax_caught <= 0; wsum_caught <= 0;
        end else begin
            // Score: first cycle state==S_OUTPUT (6)
            if (!score_caught && dut.dimm_lane[0].score_inst.state == 4'd6) begin
                score_end_cyc <= cycle;
                score_caught <= 1;
                softmax_start_cyc <= cycle;
            end
            // Softmax: first cycle sm_state==SM_OUTPUT (4)
            if (!softmax_caught && dut.dimm_lane[0].softmax_inst.sm_state == 3'd4) begin
                softmax_end_cyc <= cycle;
                softmax_caught <= 1;
                wsum_start_cyc <= cycle;
            end
            // Wsum: first cycle ws_state==WS_OUTPUT (5)
            if (!wsum_caught && dut.dimm_lane[0].wsum_inst.ws_state == 3'd5) begin
                wsum_end_cyc <= cycle;
                wsum_caught <= 1;
            end
        end
    end

    // Identity-weight preload for all 64 DPE instances per lane.
    // Phase 4: ws_exp widened to KW=128, NUM_COLS=128 — preload 128-diagonal
    // identity. ws_log remains scalar (KW=1) — weights[0][0] = 1.
    `define LOAD_WEIGHTS(L) begin \
        for (ww = 0; ww < 128; ww = ww + 1) \
            dut.dimm_lane[L].score_inst.dimm_exp.weights[ww][ww] = 1; \
        dut.dimm_lane[L].softmax_inst.sm_exp.weights[0][0] = 1; \
        dut.dimm_lane[L].wsum_inst.ws_log.weights[0][0] = 1; \
        for (ww = 0; ww < 128; ww = ww + 1) \
            dut.dimm_lane[L].wsum_inst.ws_exp.weights[ww][ww] = 1; \
    end

    // Preload Q/K/V SRAMs with one-hot Q + identity K + transposed/packed V.
    // Phase-4 v_sram layout: v_sram[m*26 + (j/EPW)][(j%EPW)*8 +: 8] = V[j][m].
    `define PRELOAD_SRAMS(L) begin \
        for (i = 0; i < 14; i = i + 1) \
            dut.dimm_lane[L].score_inst.q_sram.mem[i] = 0; \
        dut.dimm_lane[L].score_inst.q_sram.mem[0][0*8 +: 8] = 1; \
        for (i = 0; i < 1665; i = i + 1) begin \
            dut.dimm_lane[L].score_inst.k_sram_a.mem[i] = 0; \
            dut.dimm_lane[L].score_inst.k_sram_b.mem[i] = 0; \
            dut.dimm_lane[L].wsum_inst.v_sram.mem[i] = 0; \
        end \
        for (j = 0; j < D; j = j + 1) begin \
            dut.dimm_lane[L].score_inst.k_sram_a.mem[j*PD + (j/EPW)][(j%EPW)*8 +: 8] = 1; \
            dut.dimm_lane[L].score_inst.k_sram_b.mem[j*PD + (j/EPW)][(j%EPW)*8 +: 8] = 1; \
        end \
        for (j = 0; j < D; j = j + 1) \
            for (m = 0; m < D; m = m + 1) \
                if (j == m) \
                    dut.dimm_lane[L].wsum_inst.v_sram.mem[m*26 + (j/EPW)][(j%EPW)*8 +: 8] = 8'd1; \
    end

    // Force FSM into COMPUTE + past all LOAD states so we only measure
    // compute+output time.  Also pre-advance address counters so FSM doesn't
    // get stuck on LOAD exit checks.
    // Phase-4: wsum v_write_addr target is d*PACKED_N - 1 = 1663 (was N*D-1).
    `define FORCE_FSM(L) begin \
        dut.dimm_lane[L].score_inst.state = 3;       /* S_COMPUTE */ \
        dut.dimm_lane[L].score_inst.q_write_addr = 13; \
        dut.dimm_lane[L].score_inst.k_write_addr = 1664; \
        dut.dimm_lane[L].score_inst.score_idx = 0; \
        dut.dimm_lane[L].score_inst.mac_count = 0; \
        dut.dimm_lane[L].score_inst.feed_half = 0; \
        dut.dimm_lane[L].score_inst.feed_phase = 0; \
        dut.dimm_lane[L].wsum_inst.v_write_addr = D*26 - 1; \
    end

    // Force v_write_addr each cycle while in WS_LOAD_V so state exits quickly.
    always @(posedge clk) begin
        if (!rst && dut.dimm_lane[0].wsum_inst.ws_state == 3'd2) begin
            dut.dimm_lane[ 0].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[ 1].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[ 2].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[ 3].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[ 4].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[ 5].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[ 6].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[ 7].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[ 8].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[ 9].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[10].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[11].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[12].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[13].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[14].wsum_inst.v_write_addr = D*26 - 1;
            dut.dimm_lane[15].wsum_inst.v_write_addr = D*26 - 1;
        end
    end

    initial begin
        rst = 1; valid_q = 0; valid_k = 0; valid_v = 0; ready_n = 1;
        data_in_q = 0; data_in_k = 0; data_in_v = 0;

        `LOAD_WEIGHTS( 0) `LOAD_WEIGHTS( 1) `LOAD_WEIGHTS( 2) `LOAD_WEIGHTS( 3)
        `LOAD_WEIGHTS( 4) `LOAD_WEIGHTS( 5) `LOAD_WEIGHTS( 6) `LOAD_WEIGHTS( 7)
        `LOAD_WEIGHTS( 8) `LOAD_WEIGHTS( 9) `LOAD_WEIGHTS(10) `LOAD_WEIGHTS(11)
        `LOAD_WEIGHTS(12) `LOAD_WEIGHTS(13) `LOAD_WEIGHTS(14) `LOAD_WEIGHTS(15)

        `PRELOAD_SRAMS( 0) `PRELOAD_SRAMS( 1) `PRELOAD_SRAMS( 2) `PRELOAD_SRAMS( 3)
        `PRELOAD_SRAMS( 4) `PRELOAD_SRAMS( 5) `PRELOAD_SRAMS( 6) `PRELOAD_SRAMS( 7)
        `PRELOAD_SRAMS( 8) `PRELOAD_SRAMS( 9) `PRELOAD_SRAMS(10) `PRELOAD_SRAMS(11)
        `PRELOAD_SRAMS(12) `PRELOAD_SRAMS(13) `PRELOAD_SRAMS(14) `PRELOAD_SRAMS(15)

        #20; rst = 0; #15;

        start_cyc = cycle;
        $display("=== NL-DPE Full DIMM Top Latency Measurement ===");
        $display("  Config: N=%0d, D=%0d, W=%0d lanes, DW=%0d", N, D, W, DW);
        $display("  [t=%0d] reset released", start_cyc);

        // Skip LOAD by forcing FSM to S_COMPUTE.  Wait 1 cycle for propagation.
        @(posedge clk); #1;
        `FORCE_FSM( 0) `FORCE_FSM( 1) `FORCE_FSM( 2) `FORCE_FSM( 3)
        `FORCE_FSM( 4) `FORCE_FSM( 5) `FORCE_FSM( 6) `FORCE_FSM( 7)
        `FORCE_FSM( 8) `FORCE_FSM( 9) `FORCE_FSM(10) `FORCE_FSM(11)
        `FORCE_FSM(12) `FORCE_FSM(13) `FORCE_FSM(14) `FORCE_FSM(15)

        feed_qk_cyc = cycle;
        $display("  [t=%0d] FSM forced to S_COMPUTE (compute timing starts here)", feed_qk_cyc);

        // Wait for lane 0's wsum to reach WS_OUTPUT (state 5).
        begin : wait_done
            integer timeout;
            timeout = 0;
            while ((dut.dimm_lane[0].wsum_inst.ws_state != 3'd5) && (timeout < 100000)) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 100000)
                $display("  TIMEOUT at %0d cycles post-force", timeout);
        end

        end_cyc = cycle;
        $display("  [t=%0d] Lane 0 WS_OUTPUT reached", end_cyc);

        // Force score_start_cyc = FSM force time (everyone starts after force).
        score_start_cyc = feed_qk_cyc;

        // Sample NBA-assigned end_cyc registers: the always-block at line 52
        // updates *_end_cyc via non-blocking assignments.  Without an extra
        // clock edge between WS_OUTPUT detection and the $display below, the
        // wsum_end_cyc NBA from the current timestep has not yet been
        // committed, producing spurious "end=0" readings (Phase 3 TB fix).
        @(posedge clk);

        $display("");
        $display("=== Per-Stage Latency Report (lane 0) ===");
        $display("  Score stage   : %0d cycles  (start=%0d  end=%0d)",
                 score_end_cyc - score_start_cyc, score_start_cyc, score_end_cyc);
        $display("  Softmax stage : %0d cycles  (start=%0d  end=%0d)",
                 softmax_end_cyc - softmax_start_cyc, softmax_start_cyc, softmax_end_cyc);
        $display("  Wsum stage    : %0d cycles  (start=%0d  end=%0d)",
                 wsum_end_cyc - wsum_start_cyc, wsum_start_cyc, wsum_end_cyc);
        $display("");
        $display("=== End-to-end ===");
        $display("  Reset release      : cycle %0d",  start_cyc);
        $display("  FSM force (compute): cycle %0d",  feed_qk_cyc);
        $display("  WS_OUTPUT reached  : cycle %0d",  end_cyc);
        $display("  Compute+output     : %0d cycles", end_cyc - feed_qk_cyc);
        $display("");
        $display("  Phase J comparison: these per-stage RTL cycles are the ground");
        $display("  truth that sim's gemm_log/row_timing must match within 20 cyc.");

        $finish;
    end

    // Hard timeout
    initial begin
        #2000000;
        $display("HARD TIMEOUT at 2ms sim time");
        $finish;
    end

endmodule
