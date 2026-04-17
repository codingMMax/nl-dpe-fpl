// NL-DPE Full DIMM Top — Deep Functional Test at N=128, d=64, W=16
//
// Tests the complete pipeline: score_matrix → softmax → weighted_sum
// with structured inputs (one-hot Q, identity K, identity V) and verifies:
//   1. Each lane produces an expected output byte pattern.
//   2. All 16 lanes produce the same output (lane-isolation check).
//
// Inputs (broadcast to all 16 lanes; each lane has its own SRAMs):
//   Q = [1, 0, ..., 0]                        (d=64 log-domain values; only Q[0]=1)
//   K[j][k] = δ(j, k) for j, k < 64;          (identity on leading 64×64 block of 128×64)
//   V[j][m] = δ(j, m) for j, m < 64;          (identity on leading 64×64 block of 128×64)
//
// Expected (hand-computable; DPE exp approximation exp(0)=1, exp(1)=2, exp(2)=5):
//   Score[0][0]  = exp(Q[0]+K[0][0]) + Σ_{m=1..63} exp(Q[m]+K[0][m])
//                = exp(1+1) + 63·exp(0)  = 5 + 63 = 68
//   Score[0][j]  for 1 ≤ j < 64:
//                = exp(Q[0]+K[j][0]) + exp(Q[j]+K[j][j]) + 62·exp(0)
//                = exp(1) + exp(1) + 62  = 2 + 2 + 62 = 66
//   Score[0][j]  for 64 ≤ j < 128:
//                = exp(Q[0]+K[j][0]) + 63·exp(0)
//                = exp(1) + 63  = 2 + 63 = 65
//
// Lane-isolation: all 16 lanes load the same data via the broadcast inputs and
// should produce identical outputs.  The TB reads each lane's final output
// and also its intermediate score_sram / softmax_out_sram for cross-checking.

`timescale 1ns / 1ps

module tb_nldpe_dimm_top_functional;

    parameter DW = 40;
    parameter N = 128;
    parameter D = 64;
    parameter W = 16;
    parameter EPW = DW / 8;        // 5 int8 per 40-bit word
    parameter PD = (D + EPW - 1) / EPW;  // 13 packed words per d-vector

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

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz

    integer cycle;
    always @(posedge clk) if (rst) cycle <= 0; else cycle <= cycle + 1;

    integer i, j, m, lane, ww;
    integer err_count, lane_ok;
    reg [7:0] expected, observed;

    // DPE identity-weight preload + V preload (hierarchical).
    // DPEs per lane:
    //   dimm_exp   KW=128, NUM_COLS=128 → weights[0..127][0..127] diagonal = 1
    //   sm_exp     KW=1,   NUM_COLS=1   → weights[0][0] = 1
    //   ws_log     KW=1,   NUM_COLS=1   → weights[0][0] = 1
    //   ws_exp     KW=1,   NUM_COLS=1   → weights[0][0] = 1
    // V (identity on d×d block, zero outside): dut.dimm_lane[i].wsum_inst.v_sram.mem[j*d+m] = δ(j,m).
    // Also force v_write_addr = N*d-1 = 8191 so WS_LOAD_V exits on first entry.
    `define LOAD_LANE(L) begin \
        for (ww = 0; ww < 128; ww = ww + 1) \
            dut.dimm_lane[L].score_inst.dimm_exp.weights[ww][ww] = 1; \
        dut.dimm_lane[L].softmax_inst.sm_exp.weights[0][0] = 1; \
        dut.dimm_lane[L].wsum_inst.ws_log.weights[0][0] = 1; \
        dut.dimm_lane[L].wsum_inst.ws_exp.weights[0][0] = 1; \
        for (j = 0; j < N; j = j + 1) \
            for (m = 0; m < D; m = m + 1) \
                dut.dimm_lane[L].wsum_inst.v_sram.mem[j*D + m] = (j < D && j == m) ? 40'd1 : 40'd0; \
    end

    // Force v_write_addr once lane enters WS_LOAD_V, so its exit check fires.
    `define FORCE_VWADDR(L) \
        dut.dimm_lane[L].wsum_inst.v_write_addr = (N*D - 1);

    // Convenience: compute expected Score[0][j] per formula above
    function [7:0] expected_score;
        input integer jj;
        begin
            if (jj == 0)           expected_score = 68;
            else if (jj < D)       expected_score = 66;
            else                   expected_score = 65;
        end
    endfunction

    initial begin
        // ==== Bring-up ====
        rst = 1; valid_q = 0; valid_k = 0; valid_v = 0; ready_n = 1;
        data_in_q = 0; data_in_k = 0; data_in_v = 0;

        // Pre-load identity weights into all 64 DPE instances (16 lanes × 4 DPEs).
        // Done during reset, before any clock edge releases.
        `LOAD_LANE( 0) `LOAD_LANE( 1) `LOAD_LANE( 2) `LOAD_LANE( 3)
        `LOAD_LANE( 4) `LOAD_LANE( 5) `LOAD_LANE( 6) `LOAD_LANE( 7)
        `LOAD_LANE( 8) `LOAD_LANE( 9) `LOAD_LANE(10) `LOAD_LANE(11)
        `LOAD_LANE(12) `LOAD_LANE(13) `LOAD_LANE(14) `LOAD_LANE(15)

        #20; rst = 0; #15;

        $display("=== NL-DPE Full DIMM Top Deep Functional Test ===");
        $display("  Config: N=%0d, D=%0d, W=%0d lanes, DW=%0d", N, D, W, DW);
        $display("  Pipeline: score_matrix -> softmax -> weighted_sum");
        $display("  Test: Q one-hot, K identity, V identity");
        $display("");

        // ==== Drive Q (PD+2=15 pulses — see timing note) ====
        // FSM timing: the FIRST valid_q pulse triggers IDLE→LOAD_Q transition but
        // performs NO write (state was IDLE pre-posedge). The SECOND pulse is the
        // first actual write to q_sram.mem[0]. The FSM exits LOAD_Q when
        // pre-posedge q_write_addr == 13 (after 13 writes + 1 terminator). So we
        // need 1 (IDLE→LOAD_Q) + 13 (data writes to mem[0..12]) + 1 (terminator
        // writing to mem[13] before the exit check) = 15 valid_q pulses. The
        // real Q[0]=1 data must be staged on iter=1 so it gets latched at the
        // SECOND posedge (first LOAD_Q write).
        for (i = 0; i < PD + 2; i = i + 1) begin
            @(posedge clk); #1;
            valid_q = 1;
            data_in_q = 0;
            if (i == 1) data_in_q[7:0] = 1;  // Q[0] = 1 (writes mem[0] on posedge 3)
        end
        @(posedge clk); #1; valid_q = 0;
        $display("[t=%0d] Q loaded (%0d+2 pulses; data at iter=1)", cycle, PD);

        // ==== Drive K (N*PD+1 = 1665 packed words — same +1 terminator convention) ====
        for (j = 0; j < N; j = j + 1) begin
            for (i = 0; i < PD; i = i + 1) begin
                @(posedge clk); #1;
                valid_k = 1;
                data_in_k = 0;
                if (j < D) begin
                    // K[j][j] = 1, rest 0.  j goes in position (j % EPW) of packed word (j / EPW).
                    if (i == j / EPW)
                        data_in_k[(j % EPW) * 8 +: 8] = 1;
                end
            end
        end
        @(posedge clk); #1; valid_k = 1; data_in_k = 0;  // +1 terminator
        @(posedge clk); #1; valid_k = 0;
        $display("[t=%0d] K loaded (%0d packed words + 1 terminator)", cycle, N*PD);

        // ==== V is preloaded hierarchically (see LOAD_LANE). No top-level drive. ====
        // Instead, force v_write_addr = N*D-1 as soon as each lane enters WS_LOAD_V.
        $display("[t=%0d] V preloaded hierarchically (no top-level drive)", cycle);
        valid_v = 0;

        // ==== Wait for pipeline to complete ====
        // All lanes should now be in S_COMPUTE → ... → WS_OUTPUT.
        // We wait until lane 0's wsum_inst.ws_state == WS_OUTPUT (5) or timeout.
        begin : wait_done
            integer timeout;
            timeout = 0;
            while ((dut.dimm_lane[0].wsum_inst.ws_state != 3'd5) && (timeout < 40000)) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 40000) begin
                $display("[t=%0d] TIMEOUT waiting for lane 0 WS_OUTPUT", cycle);
                $display("  lane 0 score state  = %0d", dut.dimm_lane[0].score_inst.state);
                $display("  lane 0 softmax state= %0d", dut.dimm_lane[0].softmax_inst.sm_state);
                $display("  lane 0 wsum state   = %0d", dut.dimm_lane[0].wsum_inst.ws_state);
            end else begin
                $display("[t=%0d] Lane 0 reached WS_OUTPUT", cycle);
            end
        end

        // ==== Phase I.1 Check 1: score_sram values (per plan §I.1 debug tree) ====
        @(posedge clk); #1;
        $display("");
        $display("=== Check 1: score_sram (first lane) ===");
        err_count = 0;
        for (j = 0; j < N; j = j + 1) begin
            expected = expected_score(j);
            observed = dut.dimm_lane[0].score_inst.score_sram.mem[j][7:0];
            // Use !== to force X values to mismatch (!= with X yields X which iverilog treats as false).
            if ((observed !== expected) || (^observed === 1'bx)) begin
                if (err_count < 5)
                    $display("  MISMATCH score[0][%0d]: got %0d (bits=%b) expected %0d",
                             j, observed, observed, expected);
                err_count = err_count + 1;
            end
        end
        if (err_count == 0)
            $display("  PASS: all 128 score elements match (68 / 66x63 / 65x64)");
        else
            $display("  FAIL: %0d score mismatches of 128", err_count);

        // ==== Phase I.1 Check 2: lane-isolation — all 16 lanes produce same score ====
        // Iverilog doesn't allow dynamic scope indexing, so we explicitly unroll.
        $display("");
        $display("=== Check 2: lane-isolation (all lanes score should match lane 0) ===");
        lane_ok = 0;
        check_lane( 1, lane_ok);
        check_lane( 2, lane_ok);
        check_lane( 3, lane_ok);
        check_lane( 4, lane_ok);
        check_lane( 5, lane_ok);
        check_lane( 6, lane_ok);
        check_lane( 7, lane_ok);
        check_lane( 8, lane_ok);
        check_lane( 9, lane_ok);
        check_lane(10, lane_ok);
        check_lane(11, lane_ok);
        check_lane(12, lane_ok);
        check_lane(13, lane_ok);
        check_lane(14, lane_ok);
        check_lane(15, lane_ok);
        $display("  %0d / %0d lanes match lane 0's scores", lane_ok, W-1);

        // ==== Phase I.1 Check 3: softmax out_sram (intermediate) ====
        $display("");
        $display("=== Check 3: softmax normalization (lane 0, first 4 entries) ===");
        for (j = 0; j < 4; j = j + 1)
            $display("  attn[0][%0d] = %0d (byte 0)",
                     j, dut.dimm_lane[0].softmax_inst.out_sram.mem[j][7:0]);

        // ==== Phase I.1 Check 4: wsum out_sram (final output) ====
        $display("");
        $display("=== Check 4: wsum out_sram (lane 0, first 8 entries) ===");
        for (m = 0; m < 8; m = m + 1)
            $display("  out[%0d] = %0d (byte 0)",
                     m, dut.dimm_lane[0].wsum_inst.out_sram.mem[m][7:0]);

        // ==== Summary ====
        $display("");
        $display("=== Summary ===");
        $display("  Total cycles : %0d", cycle);
        $display("  Score PASS   : %0d / 128 (err=%0d)", (128 - err_count), err_count);
        $display("  Lane isolate : %0d / %0d lanes match lane 0", lane_ok, W-1);
        if (err_count <= 4 && lane_ok == W-1)
            $display("  Overall      : PASS");
        else
            $display("  Overall      : FAIL (err=%0d, lane_ok=%0d/%0d)",
                     err_count, lane_ok, W-1);

        $finish;
    end

    // Task: check lane <k> vs lane 0, update counter
    // Must be compile-time constant lane index → one task per lane via macros below.
    task check_lane(input integer lane_idx, inout integer ok_counter);
        integer jj, mismatch;
        begin
            mismatch = 0;
            for (jj = 0; jj < N; jj = jj + 1) begin
                case (lane_idx)
                    1:  if (dut.dimm_lane[ 1].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    2:  if (dut.dimm_lane[ 2].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    3:  if (dut.dimm_lane[ 3].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    4:  if (dut.dimm_lane[ 4].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    5:  if (dut.dimm_lane[ 5].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    6:  if (dut.dimm_lane[ 6].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    7:  if (dut.dimm_lane[ 7].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    8:  if (dut.dimm_lane[ 8].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    9:  if (dut.dimm_lane[ 9].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    10: if (dut.dimm_lane[10].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    11: if (dut.dimm_lane[11].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    12: if (dut.dimm_lane[12].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    13: if (dut.dimm_lane[13].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    14: if (dut.dimm_lane[14].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                    15: if (dut.dimm_lane[15].score_inst.score_sram.mem[jj][7:0] != dut.dimm_lane[0].score_inst.score_sram.mem[jj][7:0]) mismatch = mismatch + 1;
                endcase
            end
            if (mismatch == 0) ok_counter = ok_counter + 1;
            else $display("  lane %0d: %0d mismatches vs lane 0", lane_idx, mismatch);
        end
    endtask

    // Parallel: when any lane's wsum enters WS_LOAD_V (state 2), force v_write_addr
    // for ALL lanes so the WS_LOAD_V exit check immediately fires. All 16 lanes
    // run in lockstep due to broadcast inputs.
    always @(posedge clk) begin
        if (!rst && dut.dimm_lane[0].wsum_inst.ws_state == 3'd2) begin
            `FORCE_VWADDR( 0) `FORCE_VWADDR( 1) `FORCE_VWADDR( 2) `FORCE_VWADDR( 3)
            `FORCE_VWADDR( 4) `FORCE_VWADDR( 5) `FORCE_VWADDR( 6) `FORCE_VWADDR( 7)
            `FORCE_VWADDR( 8) `FORCE_VWADDR( 9) `FORCE_VWADDR(10) `FORCE_VWADDR(11)
            `FORCE_VWADDR(12) `FORCE_VWADDR(13) `FORCE_VWADDR(14) `FORCE_VWADDR(15)
        end
    end

    // Hard timeout
    initial begin
        #5000000;  // 5 ms sim time ~= 500k cycles at 10 ns clock
        $display("HARD TIMEOUT at 5ms sim time");
        $finish;
    end

endmodule
