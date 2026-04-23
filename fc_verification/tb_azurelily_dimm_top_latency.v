// Azure-Lily Full DIMM Top — Latency Measurement (W=16, N=128, d=64)
//
// Per-stage latency probes for Phase 5 parity with NL-DPE's latency TB.
// Stage-name mapping (NL-DPE convention → Azure-Lily DUT module):
//   Score   = mac_qk_inst   (dsp_mac, K=13 cycles per row, one row per QK^T score)
//   Softmax = softmax_inst  (clb_softmax FSM: S_LOAD→S_INV→S_NORM)
//   Wsum    = mac_sv_inst   (dsp_mac, K=26 cycles per mac_sv output element)
//
// Rationale for NL-DPE naming conventions (Score / Softmax / Wsum): the shared
// `run_checks.py::RE_STAGE` regex at line 144 matches those exact tags so a
// single harness parses both architectures' TB stdout identically.
//
// Probe pattern mirrors `tb_nldpe_dimm_top_latency.v:60–77` — first-valid
// pulse of each stage on lane 0 latches the end timestamp; the subsequent
// stage's start cycle is stitched from the prior stage's end. The
// `@(posedge clk);` before the final `$display` block is the NBA-commit fix
// applied in Phase 3 so non-blocking `*_end_cyc` registers are sampled after
// the scheduler has committed their values.

`timescale 1ns / 1ps

module tb_azurelily_dimm_top_latency;

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

    azurelily_dimm_top #(.N(N), .D(D), .W(W), .DATA_WIDTH(DW)) dut (
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

    integer i, j, m, out_valid_count;
    integer start_cyc, feed_qk_cyc, end_cyc;

    // Per-stage timestamps (lane 0).
    integer score_start_cyc, score_end_cyc;
    integer softmax_start_cyc, softmax_end_cyc;
    integer wsum_start_cyc, wsum_end_cyc;

    // Detect first rising edge of each stage's valid pulse on lane 0.
    reg score_caught, softmax_caught, wsum_caught;
    always @(posedge clk) begin
        if (rst) begin
            score_start_cyc <= 0;   score_end_cyc   <= 0;
            softmax_start_cyc <= 0; softmax_end_cyc <= 0;
            wsum_start_cyc <= 0;    wsum_end_cyc    <= 0;
            score_caught <= 0; softmax_caught <= 0; wsum_caught <= 0;
        end else begin
            // Score: first cycle mac_qk_inst.out_valid=1 on lane 0.
            // This is mac_qk finishing the FIRST row (K=13 valid cycles after
            // feed begins).  Per stage-name mapping this is the "Score" stage.
            if (!score_caught && dut.al_lane[0].mac_qk_inst.out_valid) begin
                score_end_cyc <= cycle;
                score_caught  <= 1;
                softmax_start_cyc <= cycle;
            end
            // Softmax: first cycle clb_softmax produces an output (state==S_NORM=3
            // with out_valid=1).  The clb_softmax FSM walks
            // S_LOAD(0)→S_INV(2)→S_NORM(3); its first data_out commit is when
            // it first enters S_NORM and raises out_valid.
            if (!softmax_caught && dut.al_lane[0].softmax_inst.valid_n) begin
                softmax_end_cyc <= cycle;
                softmax_caught  <= 1;
                wsum_start_cyc  <= cycle;
            end
            // Wsum: first cycle mac_sv_inst.out_valid=1 on lane 0.
            // mac_sv accumulates K=26 cycles of attn*V before its first
            // out_valid pulse.
            if (!wsum_caught && dut.al_lane[0].mac_sv_inst.out_valid) begin
                wsum_end_cyc <= cycle;
                wsum_caught  <= 1;
            end
        end
    end

    // Count out_valid pulses from lane 0 (one per row) for legacy mac_qk
    // total-row reporting.
    always @(posedge clk) begin
        if (rst) out_valid_count <= 0;
        else if (dut.al_lane[0].mac_qk_inst.out_valid) out_valid_count <= out_valid_count + 1;
    end

    initial begin
        rst = 1; valid_q = 0; valid_k = 0; valid_v = 0; ready_n = 1;
        data_in_q = 0; data_in_k = 0; data_in_v = 0;

        // Preload Q (one-hot), K (identity), V (identity).
        for (i = 0; i < 14; i = i + 1)
            dut.q_sram.mem[i] = 0;
        dut.q_sram.mem[0][0*8 +: 8] = 1;

        for (i = 0; i < 1665; i = i + 1)
            dut.k_sram.mem[i] = 0;
        for (j = 0; j < D; j = j + 1)
            dut.k_sram.mem[j*PD + (j/EPW)][(j%EPW)*8 +: 8] = 1;

        for (i = 0; i < 1665; i = i + 1)
            dut.v_sram.mem[i] = 0;
        for (j = 0; j < D; j = j + 1)
            dut.v_sram.mem[j*PD + (j/EPW)][(j%EPW)*8 +: 8] = 1;

        #20; rst = 0; #15;

        start_cyc = cycle;
        $display("=== Azure-Lily Full DIMM Top Latency Measurement ===");
        $display("  Config: N=%0d, D=%0d, W=%0d lanes, DW=%0d", N, D, W, DW);
        $display("  [t=%0d] reset released", start_cyc);

        // Drive one valid pulse to kick S_IDLE → S_LOAD, then force w_addr.
        @(posedge clk); #1;
        dut.q_w_addr = 13;
        dut.k_w_addr = 1664;
        dut.v_w_addr = 1664;
        valid_q = 1;
        @(posedge clk); #1;
        valid_q = 0;

        // Wait for FSM to reach S_FEED_QK.
        while (dut.state != 3'd2) @(posedge clk);
        feed_qk_cyc = cycle;
        $display("  [t=%0d] S_FEED_QK reached (compute timing starts here)", feed_qk_cyc);

        // Anchor Score stage start at S_FEED_QK entry (mirrors NL-DPE's
        // feed_qk_cyc as the start anchor for the first stage).
        score_start_cyc = feed_qk_cyc;

        // Wait until lane 0's mac_sv produces its FIRST output pulse, OR timeout.
        // That delimits the full Score→Softmax→Wsum cascade for the first
        // output row.
        begin : wait_wsum_done
            integer timeout;
            timeout = 0;
            while (!wsum_caught && (timeout < 100000)) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 100000)
                $display("  TIMEOUT: wsum did not complete (softmax_caught=%0d, score_caught=%0d)",
                         softmax_caught, score_caught);
        end

        end_cyc = cycle;
        $display("  [t=%0d] Lane 0 mac_sv first out_valid reached", end_cyc);

        // NBA-commit fix (Phase 3 pattern): the per-stage end_cyc registers
        // are assigned via non-blocking assigns in the always block above.
        // Insert one clock edge before the report so their committed values
        // are visible to $display, avoiding spurious "end=0" readings.
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
        $display("  Wsum first output  : cycle %0d",  end_cyc);
        $display("  Compute+output     : %0d cycles", end_cyc - feed_qk_cyc);
        $display("");
        $display("=== Legacy mac_qk-only counters (for reference) ===");
        $display("  mac_qk out_valid pulses : %0d / %0d", out_valid_count, N);
        $display("");
        $display("  Phase 5 comparison: these per-stage RTL cycles are the ground");
        $display("  truth that sim's gemm_dsp + softmax model must match within");
        $display("  tolerance (see phase5_known_deltas.json for annotated residuals).");

        $finish;
    end

    // Hard timeout
    initial begin
        #2000000;
        $display("HARD TIMEOUT at 2ms sim time");
        $finish;
    end

endmodule
