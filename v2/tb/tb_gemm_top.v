// ============================================================================
// tb_gemm_top.v — Stage-2 GATE-2 TB for the v2 GEMM array (`gemm_top`).
//
// Consumes case dirs produced by `v2/sim/gemm_sim.py::dump_case` (expanded by
// `v2/smoke/run_gemm_rtl.py` into <VDIR>/*.hex):
//
//   w.hex    one 2-hex byte per line,  V*H*R*C  (§4.1: v-major, h, row-outer)
//   act.hex  one 10-hex word per line, M*V*LCYC (burst m = V lane blocks)
//   expS.hex one 8-hex int32 per line, M*H*C (wide partial; padding cols 0)
//   expo.hex one 2-hex byte per line,  M*H*C (lane h = cols h*C..h*C+C-1)
//
// Plusargs:  +M=<passes>  +VDIR=<vectors dir>  [+CASE=<name>]  [+ODIR=<dir>]
// Geometry:  -DK_TB -DN_TB -DR_TB -DC_TB -DBUF_TB -DP_TB
//
// What is gated (charter v0.3):
//   * dual compare: lane bytes (`data_out`) AND the wide reduced partial
//     `dut.S_col` vs the GATE-1-certified expected bits
//   * readiness: no act strobe accepted while MSB_SA_Ready low
//   * drain integrity: exactly OCYC words per pass, M dpe_done pulses
//   * cycles: measured == T_fill_array + (M-1)*T_steady + delta
//     (delta is printed; the harness gates constancy + T_steady steps)
//
// TB probe contract (internal names frozen so the RTL header can pin them):
//   S_col[0:H*C-1] : reg signed [31:0] — wide reduced partial per column,
//                    complete at dpe_done, stable until the next pass
//   out_valid      : reg — 1-cycle pulse per drained word; lanes valid
// The primitive has no per-word valid port, so the RTL derives its drain
// from `reg_full` + OUTPUT_CYC_prim (documented in the RTL header).
// ============================================================================

`timescale 1ns / 1ps

`ifndef K_TB
  `define K_TB 128
`endif
`ifndef N_TB
  `define N_TB 128
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
`ifndef MAXM_TB
  `define MAXM_TB 16
`endif

module tb_gemm_top;

    localparam K   = `K_TB;
    localparam N   = `N_TB;
    localparam R   = `R_TB;
    localparam C   = `C_TB;
    localparam BUF = `BUF_TB;
    localparam P   = `P_TB;
    localparam MAXM = `MAXM_TB;

    localparam V   = (K + R - 1) / R;
    localparam H   = (N + C - 1) / C;
    localparam EPS = BUF / 8;
    localparam LCYC = (R * 8 + BUF - 1) / BUF;
    localparam OCYC = (C * 8 + BUF - 1) / BUF;
    localparam CCYC = P + 2;

    function integer clog2_fn;
        input integer val;
        integer i;
        begin
            clog2_fn = 0;
            for (i = val - 1; i > 0; i = i >> 1) clog2_fn = clog2_fn + 1;
        end
    endfunction
    localparam TREE_PIPE = (V > 1) ? clog2_fn(V) : 0;
    localparam L_W       = TREE_PIPE + 1;
    localparam T_FILL    = LCYC + CCYC + OCYC + L_W;
    localparam TS_A      = (LCYC + P > CCYC) ? (LCYC + P) : CCYC;
    localparam T_STEADY  = (TS_A > OCYC + 1) ? TS_A : (OCYC + 1);

    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    reg reset;
    reg [V*BUF-1:0] act_in;
    reg             act_en;
    reg [7:0]       weight_in;
    reg             weight_en;

    wire               MSB_SA_Ready;
    wire [H*BUF-1:0]   data_out;
    wire               dpe_done;
    wire               reg_full;

    gemm_top #(
        .K(K), .N(N), .R(R), .C(C), .BUF(BUF), .P(P)
    ) dut (
        .clk(clk),
        .reset(reset),
        .act_in(act_in),
        .act_en(act_en),
        .weight_in(weight_in),
        .weight_en(weight_en),
        .MSB_SA_Ready(MSB_SA_Ready),
        .data_out(data_out),
        .dpe_done(dpe_done),
        .reg_full(reg_full)
    );

    // -- stimulus / expected memories ---------------------------------------
    reg [7:0]  w_mem   [0:V*H*R*C-1];
    reg [BUF-1:0] act_mem [0:MAXM*V*LCYC-1];
    reg signed [31:0] exp_s [0:MAXM*H*C-1];
    reg [7:0]  exp_o   [0:MAXM*H*C-1];

    // -- capture ------------------------------------------------------------
    reg [7:0] cap [0:MAXM*OCYC*H*EPS-1];      // lane bytes, per pass/word/lane
    reg signed [31:0] s_cap [0:MAXM*H*C-1];   // wide partials sampled at done

    integer cycle_count;
    always @(posedge clk) cycle_count <= cycle_count + 1;

    integer done_idx, wpass, word_total;
    integer t_first_load;
    integer t_done [0:MAXM-1];
    integer ready_violations, errors, ready_errs, cycle_errs;
    integer measured, expected, delta;
    reg mark_first_load;
    integer i, m, w, v, hh, bb, kk;

    always @(posedge clk) begin
        // I3: a strobe presented in this cycle requires ready high in-cycle
        if (act_en && !MSB_SA_Ready)
            ready_violations = ready_violations + 1;

        if (mark_first_load) begin
            t_first_load = cycle_count;
            mark_first_load = 0;
        end

        if (dut.out_valid) begin
            for (hh = 0; hh < H; hh = hh + 1)
                for (bb = 0; bb < EPS; bb = bb + 1)
                    cap[(done_idx*OCYC + wpass)*H*EPS + hh*EPS + bb] =
                        data_out[hh*BUF + bb*8 +: 8];
            wpass = wpass + 1;
            word_total = word_total + 1;
        end

        if (dpe_done) begin
            if (done_idx < MAXM)
                for (i = 0; i < H*C; i = i + 1)
                    s_cap[done_idx*H*C + i] = dut.S_col[i];
            t_done[done_idx] = cycle_count;
            done_idx = done_idx + 1;
            wpass = 0;
        end
    end

    // -- driver --------------------------------------------------------------
    integer M, guard;
    reg [1023:0] vdir, odir, case_name;
    reg [2047:0] fw, fa, fs, fo, foy, foo, foc;
    reg odir_set;
    integer fdy, fdo, fdc;

    initial begin
        reset = 1;
        act_in = 0;
        act_en = 0;
        weight_in = 0;
        weight_en = 0;
        mark_first_load = 0;
        cycle_count = 0;
        done_idx = 0; wpass = 0; word_total = 0;
        t_first_load = -1;
        ready_violations = 0;
        errors = 0; ready_errs = 0; cycle_errs = 0;
        measured = 0; expected = 0; delta = 0;
        odir_set = 0;
        for (i = 0; i < MAXM; i = i + 1) t_done[i] = -1;
        for (i = 0; i < MAXM*H*C; i = i + 1) begin
            exp_s[i] = 32'sd0; exp_o[i] = 8'h00; s_cap[i] = 32'sd0;
        end
        for (i = 0; i < MAXM*OCYC*H*EPS; i = i + 1) cap[i] = 8'h00;

        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("M=%d", M)) M = 1;
        if (!$value$plusargs("VDIR=%s", vdir)) begin
            $display("[tb_gemm_top] FAIL %0s: missing +VDIR=<path>", case_name);
            $finish;
        end
        if ($value$plusargs("ODIR=%s", odir)) odir_set = 1;

        $sformat(fw, "%0s/w.hex", vdir);
        $sformat(fa, "%0s/act.hex", vdir);
        $sformat(fs, "%0s/expS.hex", vdir);
        $sformat(fo, "%0s/expo.hex", vdir);
        $readmemh(fw, w_mem);
        $readmemh(fa, act_mem);
        $readmemh(fs, exp_s);
        $readmemh(fo, exp_o);

        $display("[tb_gemm_top] case=%0s M=%0d K=%0d N=%0d R=%0d C=%0d P=%0d BUF=%0d V=%0d H=%0d LCYC=%0d OCYC=%0d T_FILL=%0d T_STEADY=%0d",
                 case_name, M, K, N, R, C, P, BUF, V, H, LCYC, OCYC, T_FILL, T_STEADY);

        repeat (3) @(posedge clk);
        #1 reset = 0;
        @(posedge clk); #1;

        // -- WEIGHT strobes (one-time; §4.1 order; no mode latch in v0.3) ----
        for (i = 0; i < V*H*R*C; i = i + 1) begin
            weight_en = 1;
            weight_in = w_mem[i];
            @(posedge clk); #1;
        end
        weight_en = 0;
        weight_in = 0;
        @(posedge clk); #1;

        // -- M ACT bursts, paced strictly by MSB_SA_Ready (P1/A9) ------------
        for (m = 0; m < M; m = m + 1) begin
            if (m > 0) begin
                guard = 0;
                while (MSB_SA_Ready && guard < 100000) begin
                    @(posedge clk); #1; guard = guard + 1;
                end
                if (MSB_SA_Ready) begin
                    $display("[tb_gemm_top]   ERROR: MSB_SA_Ready never fell after pass %0d burst", m-1);
                    ready_errs = ready_errs + 1;
                end
            end
            guard = 0;
            while (!MSB_SA_Ready && guard < 100000) begin
                @(posedge clk); #1; guard = guard + 1;
            end
            if (!MSB_SA_Ready) begin
                $display("[tb_gemm_top]   ERROR: MSB_SA_Ready never rose before pass %0d", m);
                ready_errs = ready_errs + 1;
            end
            for (w = 0; w < LCYC; w = w + 1) begin
                for (v = 0; v < V; v = v + 1)
                    act_in[v*BUF +: BUF] = act_mem[(m*V + v)*LCYC + w];
                act_en = 1;
                if (m == 0 && w == 0) mark_first_load = 1;
                @(posedge clk); #1;
            end
            act_en = 0;
            act_in = 0;
        end

        // -- wait for the final drain, then compare --------------------------
        expected = T_FILL + (M - 1) * T_STEADY;
        guard = 0;
        while (done_idx < M && guard < 4*expected + 1000) begin
            @(posedge clk); #1; guard = guard + 1;
        end
        @(posedge clk); #1;

        if (done_idx != M) begin
            $display("[tb_gemm_top]   ERROR: only %0d/%0d dpe_done pulses seen", done_idx, M);
            cycle_errs = cycle_errs + 1;
        end
        if (word_total != M*OCYC) begin
            $display("[tb_gemm_top]   ERROR: drained %0d/%0d lane words", word_total, M*OCYC);
            cycle_errs = cycle_errs + 1;
        end

        // lane bytes vs expected (lane h = columns h*C .. h*C+C-1)
        for (m = 0; m < M; m = m + 1) begin
            for (hh = 0; hh < H; hh = hh + 1) begin
                for (kk = 0; kk < C; kk = kk + 1) begin
                    if (cap[(m*OCYC + kk/EPS)*H*EPS + hh*EPS + kk%EPS]
                        !== exp_o[m*H*C + hh*C + kk]) begin
                        if (errors < 10)
                            $display("[tb_gemm_top]   out MISMATCH pass=%0d lane=%0d col=%0d got=%02x exp=%02x",
                                     m, hh, kk,
                                     cap[(m*OCYC + kk/EPS)*H*EPS + hh*EPS + kk%EPS],
                                     exp_o[m*H*C + hh*C + kk]);
                        errors = errors + 1;
                    end
                end
            end
        end

        // wide partial S vs expected (all H*C columns; padding cols equal 0)
        for (m = 0; m < M; m = m + 1) begin
            for (i = 0; i < H*C; i = i + 1) begin
                if (s_cap[m*H*C + i] !== exp_s[m*H*C + i]) begin
                    if (errors < 10)
                        $display("[tb_gemm_top]   S MISMATCH pass=%0d col=%0d got=%0d exp=%0d",
                                 m, i, s_cap[m*H*C + i], exp_s[m*H*C + i]);
                    errors = errors + 1;
                end
            end
        end

        if (done_idx == M) begin
            measured = t_done[M-1] - t_first_load;
            delta = measured - expected;   // reported; harness gates constancy
        end

        // -- observed dumps (optional; +ODIR=<dir> must exist) ---------------
        if (odir_set) begin
            $sformat(foy, "%0s/observed_psum.hex", odir);
            fdy = $fopen(foy, "w");
            for (m = 0; m < M; m = m + 1)
                for (i = 0; i < H*C; i = i + 1)
                    $fwrite(fdy, "%08x\n", s_cap[m*H*C + i]);
            $fclose(fdy);

            $sformat(foo, "%0s/observed_out.hex", odir);
            fdo = $fopen(foo, "w");
            for (m = 0; m < M; m = m + 1)
                for (hh = 0; hh < H; hh = hh + 1)
                    for (kk = 0; kk < C; kk = kk + 1)
                        $fwrite(fdo, "%02x\n",
                                cap[(m*OCYC + kk/EPS)*H*EPS + hh*EPS + kk%EPS]);
            $fclose(fdo);

            $sformat(foc, "%0s/observed_cycles.txt", odir);
            fdc = $fopen(foc, "w");
            $fwrite(fdc, "first_load=%0d done=%0d measured=%0d expected=%0d delta=%0d errors=%0d ready_err=%0d cycle_err=%0d verdict=%0s\n",
                    t_first_load, (done_idx == M) ? t_done[M-1] : -1,
                    measured, expected, delta,
                    errors, ready_errs, cycle_errs,
                    (errors == 0 && ready_errs == 0 && cycle_errs == 0) ? "PASS" : "FAIL");
            for (m = 0; m < M; m = m + 1)
                $fwrite(fdc, "pass=%0d done=%0d\n", m, t_done[m]);
            $fclose(fdc);
        end

        if (errors == 0 && ready_errs == 0 && cycle_errs == 0) begin
            $display("[tb_gemm_top] PASS case=%0s M=%0d measured=%0d expected=%0d delta=%0d",
                     case_name, M, measured, expected, delta);
        end else begin
            $display("[tb_gemm_top] FAIL case=%0s M=%0d errors=%0d ready_err=%0d cycle_err=%0d measured=%0d expected=%0d delta=%0d",
                     case_name, M, errors, ready_errs, cycle_errs,
                     measured, expected, delta);
        end
        $finish;
    end

endmodule
