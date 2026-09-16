// ============================================================================
// tb_dpe_nldpe.v — Stage 1.5 cross-check TB for the v2 NL-DPE primitive.
//
// Written from `v2/spec/dpe_nldpe.md` (v2.0.1) only. Consumes the case
// directories produced by `v2/sim/nldpe_sim.py::dump_case` (expanded by
// `v2/smoke/run_dpe_rtl.py` into `$readmemh` files):
//
//   <VDIR>/w.hex    one 2-hex byte per line, R*C weights (P23 row-major)
//   <VDIR>/act.hex  one 10-hex word per line, M*LOAD_CYC ACT words (§4.3)
//   <VDIR>/expy.hex one 8-hex int32 word per line, M*C (F2 hierarchical, P26)
//   <VDIR>/expo.hex one 2-hex byte per line, M*C (drained stream, §4.5)
//
// Plusargs:  +M=<passes>  +MODE=<0..3>  +VDIR=<vectors dir>  [+CASE=<name>]
// The mode is workload configuration (P27): driven on `nl_dpe_control` before
// and during the WEIGHT strobes, then held for all passes.
// Geometry:  -DR_TB= -DC_TB= -DBUF_TB= -DP_TB=  (defaults 256/256/40/8)
//
// What is gated (spec §8):
//   I1 functional   : hierarchical int32 `y` (probe `dut.acc`) at every
//                     acam_fire, and the drained 8-bit stream (probe
//                     `dut.drain_valid`) — dual compare P26
//   I2 cycles       : measured == T_fill + (M-1)*T_steady + delta; delta is
//                     printed (one implementation constant, §5.3)
//   I3 readiness    : no ACT strobe accepted while MSB_SA_Ready low; ready
//                     falls after a burst and rises after MSB fire (P1/A9)
//   I4/P27 mode     : one workload mode, programmed with the weights, held
//                     across all M passes (no per-pass changes)
//   P10 emergence   : shift_add_done - compute_entry + 1 == P+1 (fires + MSB
//                     S&A) for every pass; pass 0 ACAM span == P+2
//
// Probe contract (verification-only internals, D8; NOT ports — see the header
// of v2/rtl/dpe_nldpe.v): `dut.state`, `dut.acc`, `dut.acam_fire`,
// `dut.drain_valid`.
// ============================================================================

`timescale 1ns / 1ps

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

module tb_dpe_nldpe;

    localparam R   = `R_TB;
    localparam C   = `C_TB;
    localparam BUF = `BUF_TB;
    localparam P   = `P_TB;
    localparam MAXM = `MAXM_TB;

    localparam EPS  = BUF / 8;
    localparam LCYC = (R * 8 + BUF - 1) / BUF;   // §4.3
    localparam OCYC = (C * 8 + BUF - 1) / BUF;   // §4.5
    localparam T_FILL = LCYC + (P + 2) + OCYC;   // §5.3
    localparam TS_A = (LCYC + P > P + 2) ? (LCYC + P) : (P + 2);
    localparam T_STEADY = (TS_A > OCYC + 1) ? TS_A : (OCYC + 1);

    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    reg reset;
    reg [BUF-1:0] data_in;
    reg [1:0] nl_dpe_control;
    reg shift_add_control;
    reg w_buf_en;
    reg shift_add_bypass;
    reg load_output_reg;
    reg load_input_reg;

    wire        MSB_SA_Ready;
    wire [BUF-1:0] data_out;
    wire        dpe_done;
    wire        reg_full;
    wire        shift_add_done;
    wire        shift_add_bypass_ctrl;

    dpe #(
        .KERNEL_WIDTH(R),
        .NUM_COLS(C),
        .DPE_BUF_WIDTH(BUF),
        .PRECISION(P)
    ) dut (
        .clk(clk),
        .reset(reset),
        .data_in(data_in),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready),
        .data_out(data_out),
        .dpe_done(dpe_done),
        .reg_full(reg_full),
        .shift_add_done(shift_add_done),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl)
    );

    // -- stimulus / expected memories ---------------------------------------
    reg [7:0]  w_mem   [0:R*C-1];
    reg [39:0] act_mem [0:MAXM*LCYC-1];
    reg signed [31:0] exp_y [0:MAXM*C-1];
    reg [7:0]  exp_o   [0:MAXM*C-1];

    // -- capture -------------------------------------------------------------
    reg [7:0] cap [0:MAXM*OCYC*EPS-1];          // drained bytes, pass-major
    reg signed [31:0] y_cap [0:MAXM*C-1];       // pre-ACAM y per pass

    integer cycle_count;
    always @(posedge clk) cycle_count <= cycle_count + 1;

    integer done_idx, acam_idx, sad_idx, comp_idx, cap_words;
    integer t_first_load;
    integer t_done [0:MAXM-1];
    integer t_acams [0:MAXM-1];
    integer t_sad [0:MAXM-1];
    integer t_comp [0:MAXM-1];
    integer ready_violations;
    reg mark_first_load;
    reg [2:0] state_d;
    integer i;

    always @(posedge clk) begin
        // I3: a strobe presented in this cycle requires ready high in-cycle
        if (w_buf_en && !MSB_SA_Ready)
            ready_violations = ready_violations + 1;

        if (mark_first_load) begin
            t_first_load = cycle_count;
            mark_first_load = 0;
        end
        if (dut.state == 3'd2 && state_d != 3'd2 && comp_idx < MAXM) begin
            t_comp[comp_idx] = cycle_count;
            comp_idx = comp_idx + 1;
        end
        if (shift_add_done && sad_idx < MAXM) begin
            t_sad[sad_idx] = cycle_count;
            sad_idx = sad_idx + 1;
        end
        if (dut.acam_fire && acam_idx < MAXM) begin
            for (i = 0; i < C; i = i + 1)
                y_cap[acam_idx*C + i] = dut.acc[i];
            t_acams[acam_idx] = cycle_count;
            acam_idx = acam_idx + 1;
        end
        if (dut.drain_valid && cap_words < MAXM*OCYC) begin
            for (i = 0; i < EPS; i = i + 1)
                cap[cap_words*EPS + i] = data_out[8*i +: 8];
            cap_words = cap_words + 1;
        end
        if (dpe_done && done_idx < MAXM) begin
            t_done[done_idx] = cycle_count;
            done_idx = done_idx + 1;
        end
        state_d <= dut.state;
    end

    // -- driver --------------------------------------------------------------
    integer M, mode;
    integer m, t, k, guard;
    integer errors, ready_errs, span_errs, cycle_errs;
    integer measured, expected, delta;
    reg [1023:0] vdir, case_name;
    reg [2047:0] fw, fa, fy, fo;

    initial begin
        reset = 1;
        data_in = 0;
        nl_dpe_control = 0;
        shift_add_control = 0;
        w_buf_en = 0;
        shift_add_bypass = 0;
        load_output_reg = 0;
        load_input_reg = 0;
        mark_first_load = 0;
        state_d = 0;
        cycle_count = 0;
        done_idx = 0; acam_idx = 0; sad_idx = 0; comp_idx = 0; cap_words = 0;
        ready_violations = 0;
        errors = 0; ready_errs = 0; span_errs = 0; cycle_errs = 0;
        measured = 0; expected = 0; delta = 0;
        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("M=%d", M)) M = 1;
        if (!$value$plusargs("MODE=%d", mode)) mode = 0;
        if (!$value$plusargs("VDIR=%s", vdir)) begin
            $display("[tb_dpe_nldpe] FAIL %0s: missing +VDIR=<path>", case_name);
            $finish;
        end
        $sformat(fw, "%0s/w.hex", vdir);
        $sformat(fa, "%0s/act.hex", vdir);
        $sformat(fy, "%0s/expy.hex", vdir);
        $sformat(fo, "%0s/expo.hex", vdir);
        $readmemh(fw, w_mem);
        $readmemh(fa, act_mem);
        $readmemh(fy, exp_y);
        $readmemh(fo, exp_o);

        $display("[tb_dpe_nldpe] case=%0s M=%0d mode=%0d R=%0d C=%0d P=%0d BUF=%0d LCYC=%0d OCYC=%0d T_FILL=%0d T_STEADY=%0d",
                 case_name, M, mode, R, C, P, BUF, LCYC, OCYC, T_FILL, T_STEADY);

        if (mode < 0 || mode > 3) begin
            $display("[tb_dpe_nldpe] FAIL %0s: invalid +MODE=%0d (must be 0..3)",
                     case_name, mode);
            $finish;
        end

        repeat (3) @(posedge clk);
        #1 reset = 0;
        @(posedge clk); #1;

        // -- WORKLOAD CONFIG (P27): drive the mode with the WEIGHT strobes --
        // (held constant for the whole workload; latched into mode_q by the
        // RTL during the strobe sequence)
        nl_dpe_control = mode[1:0];

        // -- WEIGHT strobes (one-time, WR_CYC = R*C; P23; untimed setup) ----
        for (k = 0; k < R*C; k = k + 1) begin
            load_input_reg = 1;
            data_in = {{(BUF-8){1'b0}}, w_mem[k]};
            @(posedge clk); #1;
        end
        load_input_reg = 0;
        data_in = 0;
        nl_dpe_control = mode[1:0];   // keep the configured mode on the port
        @(posedge clk); #1;

        // -- M ACT bursts, paced strictly by MSB_SA_Ready (P1/A9) -----------
        for (m = 0; m < M; m = m + 1) begin
            if (m > 0) begin
                // ready must fall once the previous vector is complete ...
                guard = 0;
                while (MSB_SA_Ready && guard < 100000) begin
                    @(posedge clk); #1; guard = guard + 1;
                end
                if (MSB_SA_Ready) begin
                    $display("[tb_dpe_nldpe]   ERROR: MSB_SA_Ready never fell after pass %0d burst", m-1);
                    ready_errs = ready_errs + 1;
                end
            end
            // ... and rise again after the in-flight pass's MSB fire
            guard = 0;
            while (!MSB_SA_Ready && guard < 100000) begin
                @(posedge clk); #1; guard = guard + 1;
            end
            if (!MSB_SA_Ready) begin
                $display("[tb_dpe_nldpe]   ERROR: MSB_SA_Ready never rose before pass %0d", m);
                ready_errs = ready_errs + 1;
            end
            for (t = 0; t < LCYC; t = t + 1) begin
                data_in = act_mem[m*LCYC + t][BUF-1:0];
                w_buf_en = 1;
                if (m == 0 && t == 0) mark_first_load = 1;
                @(posedge clk); #1;
            end
            w_buf_en = 0;
            data_in = 0;
        end

        // -- wait for the final drain, then compare (P26 dual compare) ------
        expected = T_FILL + (M - 1) * T_STEADY;
        guard = 0;
        while (done_idx < M && guard < 4*expected + 1000) begin
            @(posedge clk); #1; guard = guard + 1;
        end
        @(posedge clk); #1;

        if (done_idx != M) begin
            $display("[tb_dpe_nldpe]   ERROR: only %0d/%0d dpe_done pulses seen", done_idx, M);
            cycle_errs = cycle_errs + 1;
        end

        for (m = 0; m < M; m = m + 1) begin
            for (k = 0; k < C; k = k + 1) begin
                if (y_cap[m*C + k] !== exp_y[m*C + k]) begin
                    if (errors < 10)
                        $display("[tb_dpe_nldpe]   y MISMATCH pass=%0d col=%0d got=%0d exp=%0d",
                                 m, k, y_cap[m*C + k], exp_y[m*C + k]);
                    errors = errors + 1;
                end
            end
        end

        if (cap_words != M*OCYC) begin
            $display("[tb_dpe_nldpe]   ERROR: captured %0d/%0d drain words", cap_words, M*OCYC);
            cycle_errs = cycle_errs + 1;
        end
        for (m = 0; m < M; m = m + 1) begin
            for (k = 0; k < C; k = k + 1) begin
                if (cap[m*OCYC*EPS + k] !== exp_o[m*C + k]) begin
                    if (errors < 10)
                        $display("[tb_dpe_nldpe]   out MISMATCH pass=%0d col=%0d got=%02x exp=%02x",
                                 m, k, cap[m*OCYC*EPS + k], exp_o[m*C + k]);
                    errors = errors + 1;
                end
            end
        end

        // -- cycle + structural-emergence checks ----------------------------
        if (done_idx == M) begin
            measured = t_done[M-1] - t_first_load;
            delta = measured - expected;   // Δ_impl: reported, harness gates constancy
        end
        for (m = 0; m < M; m = m + 1) begin
            if (t_sad[m] - t_comp[m] + 1 != P + 1) begin
                $display("[tb_dpe_nldpe]   SPAN ERROR pass=%0d fires+sad=%0d expected=%0d",
                         m, t_sad[m] - t_comp[m] + 1, P + 1);
                span_errs = span_errs + 1;
            end
        end
        if (t_acams[0] - t_comp[0] + 1 != P + 2) begin
            $display("[tb_dpe_nldpe]   SPAN ERROR pass=0 compute+acam=%0d expected=%0d",
                     t_acams[0] - t_comp[0] + 1, P + 2);
            span_errs = span_errs + 1;
        end

        if (errors == 0 && ready_errs == 0 && span_errs == 0 && cycle_errs == 0) begin
            $display("[tb_dpe_nldpe] PASS case=%0s M=%0d measured=%0d expected=%0d delta=%0d",
                     case_name, M, measured, expected, delta);
        end else begin
            $display("[tb_dpe_nldpe] FAIL case=%0s M=%0d errors=%0d ready_err=%0d span_err=%0d cycle_err=%0d measured=%0d expected=%0d delta=%0d",
                     case_name, M, errors, ready_errs, span_errs, cycle_errs,
                     measured, expected, delta);
        end
        $finish;
    end

endmodule
