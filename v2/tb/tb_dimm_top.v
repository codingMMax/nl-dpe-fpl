// ============================================================================
// tb_dimm_top.v — Stage-4 GATE-2 TB for the v2 DIMM (`dimm_top`).
//
// Consumes case dirs produced by `v2/sim/dimm_sim.py::NldpeDimm.dump_case`
// (expanded by `v2/smoke/run_dimm_rtl.py` into <VDIR>/*.hex):
//
//   a.hex    one 10-hex BUF-wide word per line, ceil(M*K*8/BUF)   (row-major)
//   b.hex    one 10-hex word per line,          ceil(K*N*8/BUF)
//   expla.hex one 2-hex byte per line, K*M  (LA_T probe order: index k*M+m)
//   explb.hex one 2-hex byte per line, K*N  (index k*N+n)
//   expacc.hex one 8-hex int32 per line, M*N (index m*N+n, valid at `done`)
//   expc.hex  one 8-hex int32 per line, M*N  (C stream order, row-major)
//
// Plusargs: +VDIR=<vectors dir> [+ODIR=<dir>] [+CASE=<name>] +EXPECTED=<int>
// Geometry: -DM_TB -DN_TB -DK_TB -DR_TB -DC_TB -DBUF_TB -DP_TB
//           -DNA_TB -DNB_TB -DNE_TB
//
// What is gated:
//   * staged compare (bit-exact vs the GATE-1-certified expected bits):
//       la_q (K*M int8, transposed) -> lb_q (K*N int8) ->
//       acc_q (M*N int32) -> C stream (M*N int32)
//   * cycles: measured (start -> done) vs +EXPECTED (model.total +
//     serialize_cycles); delta is printed, the harness gates constancy
//   * readiness: no a_en/b_en accepted while load_ready is low
//   * drain integrity: exactly M*N C words
//
// TB probe contract (frozen names; see the dimm_top.v header):
//   dut.la_q, dut.lb_q, dut.u_reduce.acc_q, dut.u_sched.a_win_total /
//   b_win_total
// ============================================================================

`timescale 1ns / 1ps

`ifndef M_TB
  `define M_TB 8
`endif
`ifndef N_TB
  `define N_TB 10
`endif
`ifndef K_TB
  `define K_TB 6
`endif
`ifndef R_TB
  `define R_TB 256
`endif
`ifndef C_TB
  `define C_TB 256
`endif
`ifndef BUF_TB
  `define BUF_TB 40
`endif
`ifndef P_TB
  `define P_TB 8
`endif
`ifndef NA_TB
  `define NA_TB 1
`endif
`ifndef NB_TB
  `define NB_TB 1
`endif
`ifndef NE_TB
  `define NE_TB 2
`endif

module tb_dimm_top;

    localparam M   = `M_TB;
    localparam N   = `N_TB;
    localparam K   = `K_TB;
    localparam R   = `R_TB;
    localparam C   = `C_TB;
    localparam BUF = `BUF_TB;
    localparam P   = `P_TB;
    localparam N_A = `NA_TB;
    localparam N_B = `NB_TB;
    localparam N_E = `NE_TB;

    localparam EPS = BUF / 8;
    localparam A_WORDS = (M*K*8 + BUF - 1) / BUF;
    localparam B_WORDS = (K*N*8 + BUF - 1) / BUF;
    localparam LA_N = K * M;
    localparam LB_N = K * N;
    localparam ACC_N = M * N;

    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    reg reset;
    reg prog_start;
    reg [BUF-1:0] a_in, b_in;
    reg a_en, b_en;
    reg start;

    wire prog_ready, load_ready, busy, out_valid, done;
    wire [31:0] data_out;

    dimm_top #(
        .M(M), .N(N), .K(K), .R(R), .C(C), .BUF(BUF), .P(P),
        .N_A(N_A), .N_B(N_B), .N_E(N_E)
    ) dut (
        .clk(clk),
        .reset(reset),
        .prog_start(prog_start),
        .prog_ready(prog_ready),
        .a_in(a_in),
        .a_en(a_en),
        .b_in(b_in),
        .b_en(b_en),
        .load_ready(load_ready),
        .start(start),
        .busy(busy),
        .data_out(data_out),
        .out_valid(out_valid),
        .done(done)
    );

    // -- stimulus / expected memories ---------------------------------------
    reg [BUF-1:0] a_mem [0:A_WORDS-1];
    reg [BUF-1:0] b_mem [0:B_WORDS-1];
    reg [7:0]  exp_la [0:LA_N-1];
    reg [7:0]  exp_lb [0:LB_N-1];
    reg signed [31:0] exp_acc [0:ACC_N-1];
    reg signed [31:0] exp_c [0:ACC_N-1];

    // -- capture ------------------------------------------------------------
    reg [7:0]  cap_la [0:LA_N-1];
    reg [7:0]  cap_lb [0:LB_N-1];
    reg signed [31:0] cap_acc [0:ACC_N-1];
    reg signed [31:0] cap_c [0:ACC_N-1];

    integer cycle_count;
    always @(posedge clk) cycle_count <= cycle_count + 1;

    integer t_prog, t_load0, t_load1, t_start, t_done;
    integer c_words;
    integer ready_errs;
    integer i;      // driver (initial block) only
    integer mi;     // monitor (always block) only — do not share with `i`
    // per-stage timestamps (calibration: span_* vs model T_A/T_B/T_E)
    integer t_a0, t_aN, t_b0, t_bN, t_f0, t_fN, t_c0, t_cN;
    integer a_wins, b_wins, f_wins;
    integer span_a, span_b, span_f, fill, tail, ser;

    // Event monitor: sampled on the NEGEDGE so combinational probes
    // (dpe_w_buf_en, *_win_done, out_valid/data_out) are read mid-cycle,
    // race-free. Cycle numbering matches the posedge counter for differences.
    always @(negedge clk) begin
        if (a_en && !load_ready) ready_errs = ready_errs + 1;
        if (b_en && !load_ready) ready_errs = ready_errs + 1;
        if (prog_ready && t_prog < 0) t_prog = cycle_count;
        if (start && t_start < 0) t_start = cycle_count;
        if (out_valid && c_words < ACC_N) begin
            cap_c[c_words] = data_out;
            c_words = c_words + 1;
            if (t_c0 < 0) t_c0 = cycle_count;
            t_cN = cycle_count;
        end
        // span reference: the cycle the first ACT strobe is *presented*
        // (primitive TB convention: presentation cycle -> dpe_done)
        if (dut.gen_poolA[0].u_pool.dpe_w_buf_en && t_a0 < 0)
            t_a0 = cycle_count;
        if (dut.gen_poolB[0].u_pool.dpe_w_buf_en && t_b0 < 0)
            t_b0 = cycle_count;
        if (dut.gen_farm[0].u_farm.dpe_w_buf_en && t_f0 < 0)
            t_f0 = cycle_count;
        if (|dut.a_win_done) begin
            t_aN = cycle_count;
            for (mi = 0; mi < N_A; mi = mi + 1) a_wins = a_wins + dut.a_win_done[mi];
        end
        if (|dut.b_win_done) begin
            t_bN = cycle_count;
            for (mi = 0; mi < N_B; mi = mi + 1) b_wins = b_wins + dut.b_win_done[mi];
        end
        if (|dut.f_win_done) begin
            t_fN = cycle_count;
            for (mi = 0; mi < N_E; mi = mi + 1) f_wins = f_wins + dut.f_win_done[mi];
        end
        if (done && t_done < 0) t_done = cycle_count;
    end

    // -- driver --------------------------------------------------------------
    integer M_ign, errors, cycle_errs;
    integer measured, expected, delta, guard;
    reg [1023:0] vdir, odir, case_name;
    reg [2047:0] fa, fb, fla, flb, facc, fc;
    reg [2047:0] ola, olb, oacc, oc, ocyc;
    reg odir_set;
    integer fda, fdb, fdl, fdm, fdn, fdo;

    initial begin
        reset = 1;
        prog_start = 0;
        a_in = 0; b_in = 0; a_en = 0; b_en = 0; start = 0;
        cycle_count = 0;
        t_prog = -1; t_load0 = -1; t_load1 = -1; t_start = -1; t_done = -1;
        t_a0 = -1; t_aN = -1; t_b0 = -1; t_bN = -1;
        t_f0 = -1; t_fN = -1; t_c0 = -1; t_cN = -1;
        a_wins = 0; b_wins = 0; f_wins = 0;
        span_a = 0; span_b = 0; span_f = 0; fill = 0; tail = 0; ser = 0;
        c_words = 0; ready_errs = 0; errors = 0; cycle_errs = 0;
        measured = 0; expected = 0; delta = 0;
        odir_set = 0;

        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("VDIR=%s", vdir)) begin
            $display("[tb_dimm_top] FAIL %0s: missing +VDIR=<path>", case_name);
            $finish;
        end
        if ($value$plusargs("ODIR=%s", odir)) odir_set = 1;
        if (!$value$plusargs("EXPECTED=%d", expected)) expected = 0;

        $sformat(fa, "%0s/a.hex", vdir);
        $sformat(fb, "%0s/b.hex", vdir);
        $sformat(fla, "%0s/expla.hex", vdir);
        $sformat(flb, "%0s/explb.hex", vdir);
        $sformat(facc, "%0s/expacc.hex", vdir);
        $sformat(fc, "%0s/expc.hex", vdir);
        $readmemh(fa, a_mem);
        $readmemh(fb, b_mem);
        $readmemh(fla, exp_la);
        $readmemh(flb, exp_lb);
        $readmemh(facc, exp_acc);
        $readmemh(fc, exp_c);

        $display("[tb_dimm_top] case=%0s M=%0d N=%0d K=%0d R=%0d C=%0d BUF=%0d P=%0d N_A=%0d N_B=%0d N_E=%0d A_WORDS=%0d B_WORDS=%0d",
                 case_name, M, N, K, R, C, BUF, P, N_A, N_B, N_E, A_WORDS, B_WORDS);

        repeat (3) @(posedge clk);
        #1 reset = 0;
        @(posedge clk); #1;

        // -- weight programming (identity eye broadcast, WR = R*C) ---------
        prog_start = 1;
        @(posedge clk); #1;
        prog_start = 0;
        guard = 0;
        while (!prog_ready && guard < 4*R*C + 1000) begin
            @(posedge clk); #1; guard = guard + 1;
        end
        if (!prog_ready) begin
            $display("[tb_dimm_top]   ERROR: prog_ready never rose");
            cycle_errs = cycle_errs + 1;
        end

        // -- A/B loads (setup; excluded from the measured run) --------------
        t_load0 = cycle_count;
        for (i = 0; i < A_WORDS; i = i + 1) begin
            a_in = a_mem[i];
            a_en = 1;
            @(posedge clk); #1;
        end
        a_en = 0; a_in = 0;
        for (i = 0; i < B_WORDS; i = i + 1) begin
            b_in = b_mem[i];
            b_en = 1;
            @(posedge clk); #1;
        end
        b_en = 0; b_in = 0;
        t_load1 = cycle_count;

        // -- run -------------------------------------------------------------
        start = 1;
        @(posedge clk); #1;
        start = 0;

        guard = 0;
        while (t_done < 0 && guard < 8*expected + 100000) begin
            @(posedge clk); #1; guard = guard + 1;
        end
        @(posedge clk); #1;

        // frozen probes sampled at end of run (done or timeout);
        // the reduce keeps per-port banks — the TB sums them (acc_q value)
        for (i = 0; i < LA_N; i = i + 1) cap_la[i] = dut.la_q[i];
        for (i = 0; i < LB_N; i = i + 1) cap_lb[i] = dut.lb_q[i];
        for (i = 0; i < ACC_N; i = i + 1) begin
            cap_acc[i] = 32'sd0;
            for (mi = 0; mi < N_E; mi = mi + 1)
                cap_acc[i] = cap_acc[i] + dut.u_reduce.acc_bank[mi][i];
        end

        if (t_done < 0) begin
            $display("[tb_dimm_top]   ERROR: done never pulsed (timeout)");
            cycle_errs = cycle_errs + 1;
        end
        if (c_words != ACC_N) begin
            $display("[tb_dimm_top]   ERROR: drained %0d/%0d C words", c_words, ACC_N);
            cycle_errs = cycle_errs + 1;
        end

        // -- staged compare --------------------------------------------------
        for (i = 0; i < LA_N; i = i + 1)
            if (cap_la[i] !== exp_la[i]) begin
                if (errors < 10)
                    $display("[tb_dimm_top]   la MISMATCH idx=%0d got=%02x exp=%02x",
                             i, cap_la[i], exp_la[i]);
                errors = errors + 1;
            end
        for (i = 0; i < LB_N; i = i + 1)
            if (cap_lb[i] !== exp_lb[i]) begin
                if (errors < 10)
                    $display("[tb_dimm_top]   lb MISMATCH idx=%0d got=%02x exp=%02x",
                             i, cap_lb[i], exp_lb[i]);
                errors = errors + 1;
            end
        for (i = 0; i < ACC_N; i = i + 1)
            if (cap_acc[i] !== exp_acc[i]) begin
                if (errors < 10)
                    $display("[tb_dimm_top]   acc MISMATCH idx=%0d got=%0d exp=%0d",
                             i, cap_acc[i], exp_acc[i]);
                errors = errors + 1;
            end
        for (i = 0; i < ACC_N; i = i + 1)
            if (cap_c[i] !== exp_c[i]) begin
                if (errors < 10)
                    $display("[tb_dimm_top]   C MISMATCH idx=%0d got=%0d exp=%0d",
                             i, cap_c[i], exp_c[i]);
                errors = errors + 1;
            end

        if (t_done > 0 && t_start > 0) measured = t_done - t_start;
        delta = measured - expected;   // reported; harness gates constancy

        // per-stage spans (calibration vs model T_A/T_B/T_E)
        if (t_a0 >= 0 && t_aN >= 0) span_a = t_aN - t_a0;
        if (t_b0 >= 0 && t_bN >= 0) span_b = t_bN - t_b0;
        if (t_f0 >= 0 && t_fN >= 0) span_f = t_fN - t_f0;
        if (t_f0 >= 0 && t_start >= 0) fill = t_f0 - t_start;
        if (t_done >= 0 && t_fN >= 0) tail = t_done - t_fN;
        if (t_c0 >= 0 && t_cN >= 0) ser = t_cN - t_c0 + 1;

        // -- observed dumps (optional; +ODIR=<dir> must exist) --------------
        if (odir_set) begin
            $sformat(ola, "%0s/observed_la.hex", odir);
            fda = $fopen(ola, "w");
            for (i = 0; i < LA_N; i = i + 1) $fwrite(fda, "%02x\n", cap_la[i]);
            $fclose(fda);

            $sformat(olb, "%0s/observed_lb.hex", odir);
            fdb = $fopen(olb, "w");
            for (i = 0; i < LB_N; i = i + 1) $fwrite(fdb, "%02x\n", cap_lb[i]);
            $fclose(fdb);

            $sformat(oacc, "%0s/observed_acc.hex", odir);
            fdm = $fopen(oacc, "w");
            for (i = 0; i < ACC_N; i = i + 1)
                $fwrite(fdm, "%08x\n", cap_acc[i]);
            $fclose(fdm);

            $sformat(oc, "%0s/observed_c.hex", odir);
            fdn = $fopen(oc, "w");
            for (i = 0; i < ACC_N; i = i + 1)
                $fwrite(fdn, "%08x\n", cap_c[i]);
            $fclose(fdn);

            $sformat(ocyc, "%0s/observed_cycles.txt", odir);
            fdo = $fopen(ocyc, "w");
            $fwrite(fdo, "prog=%0d load0=%0d load1=%0d start=%0d done=%0d measured=%0d expected=%0d delta=%0d errors=%0d ready_err=%0d cycle_err=%0d verdict=%0s\n",
                    t_prog, t_load0, t_load1, t_start, t_done,
                    measured, expected, delta, errors, ready_errs, cycle_errs,
                    (errors == 0 && ready_errs == 0 && cycle_errs == 0)
                        ? "PASS" : "FAIL");
            $fwrite(fdo, "load_cycles=%0d weight_cycles=%0d\n",
                    (t_load1 > t_load0) ? (t_load1 - t_load0) : -1, R*C);
            $fwrite(fdo, "t_a0=%0d t_aN=%0d a_wins=%0d span_A=%0d t_b0=%0d t_bN=%0d b_wins=%0d span_B=%0d\n",
                    t_a0, t_aN, a_wins, span_a, t_b0, t_bN, b_wins, span_b);
            $fwrite(fdo, "t_f0=%0d t_fN=%0d f_wins=%0d span_F=%0d fill=%0d tail=%0d ser=%0d t_c0=%0d t_cN=%0d\n",
                    t_f0, t_fN, f_wins, span_f, fill, tail, ser, t_c0, t_cN);
            $fclose(fdo);
        end

        if (errors == 0 && ready_errs == 0 && cycle_errs == 0) begin
            $display("[tb_dimm_top] PASS case=%0s measured=%0d expected=%0d delta=%0d span_A=%0d span_B=%0d span_F=%0d fill=%0d ser=%0d wins_A=%0d wins_B=%0d wins_F=%0d",
                     case_name, measured, expected, delta,
                     span_a, span_b, span_f, fill, ser,
                     a_wins, b_wins, f_wins);
        end else begin
            $display("[tb_dimm_top] FAIL case=%0s errors=%0d ready_err=%0d cycle_err=%0d measured=%0d expected=%0d delta=%0d span_A=%0d span_B=%0d span_F=%0d fill=%0d ser=%0d wins_A=%0d wins_B=%0d wins_F=%0d",
                     case_name, errors, ready_errs, cycle_errs,
                     measured, expected, delta,
                     span_a, span_b, span_f, fill, ser,
                     a_wins, b_wins, f_wins);
        end
        $finish;
    end

endmodule
