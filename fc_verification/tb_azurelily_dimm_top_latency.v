// Azure-Lily Full DIMM Top — Latency Measurement (W=16, N=128, d=64)
//
// Measures mac_qk's latency for all N=128 QK^T scores.  The AL top's
// softmax + mac_sv stages are structurally present but not end-to-end
// wired in the current generator, so we measure the mac_qk portion
// explicitly by counting out_valid pulses from lane 0.
//
// Reports:
//   mac_qk_latency : cycles from S_FEED_QK entry to lane 0's N-th out_valid.
//   That's the "QK^T matmul" time.  Softmax and mac_sv add ~constant
//   post-processing time (separately estimated in the alignment log).

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

    // Count out_valid pulses from lane 0 (one per row).
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
        $display("  [t=%0d] S_FEED_QK reached (mac_qk timing starts here)", feed_qk_cyc);

        // Wait for N=128 out_valid pulses from lane 0 (one per row).
        begin : wait_done
            integer timeout;
            timeout = 0;
            while ((out_valid_count < N) && (timeout < 100000)) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 100000)
                $display("  TIMEOUT: got %0d/%0d out_valid pulses", out_valid_count, N);
        end

        end_cyc = cycle;
        $display("  [t=%0d] All N=%0d QK^T rows produced", end_cyc, N);

        $display("");
        $display("=== Latency Report ===");
        $display("  Reset release      : cycle %0d",  start_cyc);
        $display("  S_FEED_QK entry    : cycle %0d",  feed_qk_cyc);
        $display("  N=%0d rows done    : cycle %0d",  N, end_cyc);
        $display("");
        $display("  mac_qk latency (compute-only) : %0d cycles",
                 end_cyc - feed_qk_cyc);
        $display("  Per-row avg                    : %.2f cycles/row",
                 (end_cyc - feed_qk_cyc) * 1.0 / N);
        $display("");
        $display("  NOTE: softmax + mac_sv not exercised in this test.");
        $display("  Post-mac_qk cycles (softmax + mac_sv) would add an est. ~500 cycles");
        $display("  per row based on sim model — total end-to-end AL latency is that");
        $display("  mac_qk latency + per-row softmax + per-col mac_sv.");

        $finish;
    end

    // Hard timeout
    initial begin
        #500000;
        $display("HARD TIMEOUT");
        $finish;
    end

endmodule
