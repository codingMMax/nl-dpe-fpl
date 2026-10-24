// ============================================================================
// tb_softmax_online_acc.v — GATE-2 testbench of `softmax_online_acc` (v3, ip/softmax/acc_spec.md §8). Case files are
// written by ip/softmax/acc_cases.py (model run + GATE 1); the runner ip/softmax/tb/run_acc_rtl.py compares this
// testbench's event lines with the model's, line for line. The bit-plane pass (2026-10-23): every window is 8 words
// on consecutive cycles (E8 L_P*I bits per channel, E6 2S bits, E7 S bits, E9 N_OC*I bits): E8 value words (the trims
// of 2026-10-23), the others planes.
//
// The testbench is the module's environment of the model's test graph (ip/softmax/acc_cases.py, sim/core/env.py):
//   p source    every E8 window at the cycle the model planned, on its channel; before a tile's first word the
//               testbench checks that the RTL has granted its credit (p_cr of tile u - 2)
//   la source   every E6 window at its planned cycle (credit: la_cr of unit u - 2)
//   ll source   every E7 window at its planned cycle (credit: ll_cr of run r - 1)
//   o sink      takes every E9 window, checks every plane, no gap inside a window
// Logged (EV lines, the model's event format): the p / la / ll windows (first plane), p_cr / la_cr / ll_cr, the E9
// windows (first plane). Also checked: run_done at the model's cycle, once per run; the plane counts; the FIFO probe;
// every crossbar's sticky err flag; a watchdog.
// Weights: preloaded at zero time after reset (identity in every crossbar, mode_q LOG / EXP; PLAN R8).
// Cycle 0 is the first cycle after reset.
// ============================================================================
`timescale 1ns / 1ps

`ifndef S_TB
  `define S_TB 16
`endif
`ifndef D_TB
  `define D_TB 16
`endif
`ifndef R_TB
  `define R_TB 64
`endif
`ifndef C_TB
  `define C_TB 64
`endif
`ifndef NOC_TB
  `define NOC_TB 2
`endif
`ifndef NBLK_TB
  `define NBLK_TB 2
`endif
`ifndef SO_TB
  `define SO_TB 4
`endif
`ifndef LN_TB
  `define LN_TB 4
`endif
`ifndef PCH_TB
  `define PCH_TB 2
`endif
`ifndef VWP_TB
  `define VWP_TB 13
`endif

module tb_softmax_online_acc;
    localparam S = `S_TB, D = `D_TB, R = `R_TB, C = `C_TB, P = 8, N_OC = `NOC_TB, NBLK = `NBLK_TB;
    localparam SO = `SO_TB, LN = `LN_TB, P_CH = `PCH_TB, VWP = `VWP_TB;
    localparam I    = (R < C) ? R : C;
    localparam P_OC = (S * D) / I;
    localparam W    = (P_OC + N_OC - 1) / N_OC;
    localparam L_P  = (VWP + 7) / 8;
    localparam PL   = L_P * I;                          // an E8 word (I/8 elements of L_P bytes)
    localparam OWW  = N_OC * I;                         // an E9 plane word
    localparam MAXW = 16384, MAXLA = 4096, MAXR = 64, MAXO = 4096;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg reset = 1'b1;

    reg  [P_CH-1:0]    p_v = 0;
    reg  [P_CH*PL-1:0] p_d = 0;
    reg                la_v = 1'b0, ll_v = 1'b0;
    reg  [2*S-1:0]     la_d = 0;
    reg  [S-1:0]       ll_d = 0;
    wire               p_cr, la_cr, ll_cr, o_v, run_done;
    wire [OWW-1:0]     o_d;

    softmax_online_acc #(.S(S), .D(D), .R(R), .C(C), .P(P), .N_OC(N_OC), .NBLK(NBLK), .SO(SO), .LN(LN),
                         .P_CH(P_CH), .VWP(VWP)) dut (
        .clk(clk), .reset(reset), .weight_in(8'd0), .weight_en(1'b0),
        .p_v(p_v), .p_d(p_d), .p_cr(p_cr), .la_v(la_v), .la_d(la_d), .la_cr(la_cr),
        .ll_v(ll_v), .ll_d(ll_d), .ll_cr(ll_cr), .o_v(o_v), .o_d(o_d), .run_done(run_done));

    // ---- weights: zero-time preload after reset (PLAN R8) ----------------------------------------------------------
    event preload;
    wire [2*N_OC-1:0] x_err;
    genvar wg;
    generate
        for (wg = 0; wg < N_OC; wg = wg + 1) begin : wp_x
            integer wk;
            always @(preload) begin
                for (wk = 0; wk < R*C; wk = wk + 1) begin
                    dut.g_x[wg].u_lg.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                    dut.g_x[wg].u_ex.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                end
                dut.g_x[wg].u_lg.mode_q = 2'b11;                                 // LOG
                dut.g_x[wg].u_ex.mode_q = 2'b10;                                 // EXP
            end
            assign x_err[2*wg]     = dut.g_x[wg].u_lg.err;
            assign x_err[2*wg + 1] = dut.g_x[wg].u_ex.err;
        end
    endgenerate

    // ---- case files ------------------------------------------------------------------------------------------------
    reg [8*256-1:0] vdir, fn;
    integer fd, rc, i, k;
    integer np, nla, nrun, no, tmax;
    integer pw_t [0:MAXW-1], pw_ch [0:MAXW-1], pw_u [0:MAXW-1], pw_w [0:MAXW-1];
    reg [PL-1:0]  pw_d [0:MAXW*P-1];
    integer la_t [0:MAXLA-1], la_u [0:MAXLA-1];
    reg [2*S-1:0] la_x [0:MAXLA*P-1];
    integer ll_t [0:MAXR-1], ll_u [0:MAXR-1];
    reg [S-1:0]   ll_x [0:MAXR*P-1];
    integer ow_t [0:MAXO-1], ow_r [0:MAXO-1], ow_w [0:MAXO-1];
    reg [OWW-1:0] ox [0:MAXO*P-1];
    integer rd_t [0:MAXR-1];
    reg [PL-1:0]  pword;
    reg [2*S-1:0] aword;
    reg [S-1:0]   lword;
    reg [OWW-1:0] oword;

    task load;
        begin
            $sformat(fn, "%0s/p_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < np; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d %d", pw_t[i], pw_ch[i], pw_u[i], pw_w[i]);
                for (k = 0; k < P; k = k + 1) begin rc = $fscanf(fd, "%h", pword); pw_d[i*P + k] = pword; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/la_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nla; i = i + 1) begin
                rc = $fscanf(fd, "%d %d", la_t[i], la_u[i]);
                for (k = 0; k < P; k = k + 1) begin rc = $fscanf(fd, "%h", aword); la_x[i*P + k] = aword; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/ll_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nrun; i = i + 1) begin
                rc = $fscanf(fd, "%d %d", ll_t[i], ll_u[i]);
                for (k = 0; k < P; k = k + 1) begin rc = $fscanf(fd, "%h", lword); ll_x[i*P + k] = lword; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/o_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < no; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d", ow_t[i], ow_r[i], ow_w[i]);
                for (k = 0; k < P; k = k + 1) begin rc = $fscanf(fd, "%h", oword); ox[i*P + k] = oword; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/run_done.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nrun; i = i + 1) rc = $fscanf(fd, "%d", rd_t[i]);
            $fclose(fd);
        end
    endtask

    // ---- cycle counter, errors -------------------------------------------------------------------------------------
    integer tc = 0;
    always @(posedge clk) if (!reset) tc <= tc + 1;
    integer errors = 0;

    // ---- monitor + drivers (cycle tc, just after the edge) ---------------------------------------------------------
    integer n_pcr = 0, n_lacr = 0, n_llcr = 0, n_done = 0, n_op = 0, owin = 0, opl = 0;
    integer sp [0:P_CH-1], sc [0:P_CH-1];               // per channel: window (array index), plane in it
    integer lap = 0, lac = -1, llp = 0, llc = -1;
    reg     running = 1'b0;
    integer ch;
    initial for (ch = 0; ch < P_CH; ch = ch + 1) begin sp[ch] = ch; sc[ch] = -1; end
    // channel ch's k-th E8 window is window ch + P_CH*k of the stream (windows dealt round robin). Runs at posedge + 2.
    always @(posedge clk) begin
        #2;
        if (running) begin
            // -- DUT outputs of this cycle --
            if (p_cr)  begin $display("EV %0d acc.p cr u%0d", tc, n_pcr);  n_pcr  = n_pcr + 1;  end
            if (la_cr) begin $display("EV %0d acc.la cr u%0d", tc, n_lacr); n_lacr = n_lacr + 1; end
            if (ll_cr) begin $display("EV %0d acc.ll cr u%0d", tc, n_llcr); n_llcr = n_llcr + 1; end
            if (o_v) begin
                if (owin >= no) begin
                    if (errors < 10) $display("[tb_acc] ERROR: an E9 plane after the last window at %0d", tc);
                    errors = errors + 1;
                end
                else begin
                    if (opl == 0) $display("EV %0d acc.o win ch0 u%0d w%0d", tc, ow_r[owin], ow_w[owin]);
                    if (o_d !== ox[owin * P + opl]) begin
                        if (errors < 10) $display("[tb_acc] ERROR o: run %0d wave %0d plane %0d got %h exp %h",
                                                  ow_r[owin], ow_w[owin], opl, o_d, ox[owin * P + opl]);
                        errors = errors + 1;
                    end
                    n_op = n_op + 1;
                    opl = opl + 1;
                    if (opl == P) begin opl = 0; owin = owin + 1; end
                end
            end
            else if (opl != 0) begin
                $display("[tb_acc] ERROR: gap inside an E9 window (window %0d)", owin); errors = errors + 1;
            end
            if (run_done) begin
                if (n_done >= nrun || rd_t[n_done] != tc) begin
                    $display("[tb_acc] ERROR: run_done at %0d (expected %0d)", tc, (n_done < nrun) ? rd_t[n_done] : -1);
                    errors = errors + 1;
                end
                n_done = n_done + 1;
            end
            if (dut.fifo_err) begin
                if (errors < 10) $display("[tb_acc] ERROR: FIFO probe at %0d", tc);
                errors = errors + 1;
            end
            // -- drivers --
            p_v = 0;
            for (ch = 0; ch < P_CH; ch = ch + 1) begin
                if (sc[ch] < 0 && sp[ch] < np && pw_t[sp[ch]] == tc) begin
                    sc[ch] = 0;
                    if (pw_ch[sp[ch]] != ch) begin $display("[tb_acc] ERROR: E8 window channel"); errors = errors + 1; end
                    if (pw_w[sp[ch]] == 0 && pw_u[sp[ch]] >= 2 && n_pcr <= pw_u[sp[ch]] - 2) begin
                        $display("[tb_acc] ERROR: tile %0d sent at %0d without credit %0d", pw_u[sp[ch]], tc,
                                 pw_u[sp[ch]] - 2);
                        errors = errors + 1;
                    end
                    $display("EV %0d p_src.p win ch%0d u%0d w%0d", tc, ch, pw_u[sp[ch]], pw_w[sp[ch]]);
                end
                if (sc[ch] >= 0) begin
                    p_v[ch] = 1'b1;
                    p_d[ch*PL +: PL] = pw_d[sp[ch]*P + sc[ch]];
                    sc[ch] = sc[ch] + 1;
                    if (sc[ch] == P) begin sc[ch] = -1; sp[ch] = sp[ch] + P_CH; end
                end
            end
            la_v = 1'b0;
            if (lac < 0 && lap < nla && la_t[lap] == tc) begin
                lac = 0;
                if (la_u[lap] >= 2 && n_lacr <= la_u[lap] - 2) begin
                    $display("[tb_acc] ERROR: alpha unit %0d sent at %0d without credit %0d", la_u[lap], tc,
                             la_u[lap] - 2);
                    errors = errors + 1;
                end
                $display("EV %0d la_src.la win ch0 u%0d w0", tc, la_u[lap]);
            end
            if (lac >= 0) begin
                la_v = 1'b1;
                la_d = la_x[lap * P + lac];
                lac = lac + 1;
                if (lac == P) begin lac = -1; lap = lap + 1; end
            end
            ll_v = 1'b0;
            if (llc < 0 && llp < nrun && ll_t[llp] == tc) begin
                llc = 0;
                if (ll_u[llp] >= 1 && n_llcr <= ll_u[llp] - 1) begin
                    $display("[tb_acc] ERROR: log l of run %0d sent at %0d without credit %0d", ll_u[llp], tc,
                             ll_u[llp] - 1);
                    errors = errors + 1;
                end
                $display("EV %0d ll_src.ll win ch0 u%0d w0", tc, ll_u[llp]);
            end
            if (llc >= 0) begin
                ll_v = 1'b1;
                ll_d = ll_x[llp * P + llc];
                llc = llc + 1;
                if (llc == P) begin llc = -1; llp = llp + 1; end
            end
        end
    end

    // ---- main ----------------------------------------------------------------------------------------------------
    reg [8*128-1:0] case_name;
    initial begin
        if (!$value$plusargs("VDIR=%s", vdir)) begin $display("[tb_acc] ERROR: +VDIR required"); $finish; end
        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("NP=%d", np)) np = 0;
        if (!$value$plusargs("NLA=%d", nla)) nla = 0;
        if (!$value$plusargs("NRUN=%d", nrun)) nrun = 0;
        if (!$value$plusargs("TMAX=%d", tmax)) tmax = 100000;
        no = nrun * W;
        $display("[tb_acc] case=%0s S=%0d D=%0d R=%0d C=%0d N_OC=%0d NBLK=%0d SO=%0d LN=%0d P_CH=%0d P_OC=%0d W=%0d runs=%0d",
                 case_name, S, D, R, C, N_OC, NBLK, SO, LN, P_CH, P_OC, W, nrun);
        load;
        repeat (3) @(posedge clk);
        #1 reset = 1'b0;
        -> preload;
        running = 1'b1;
        while ((owin < no || n_done < nrun || n_pcr < nrun * NBLK || n_lacr < nla || n_llcr < nrun) && tc < tmax)
            @(posedge clk);
        repeat (4) @(posedge clk);
        #4;
        if (tc >= tmax) begin $display("[tb_acc] ERROR: watchdog (tc %0d, E9 windows out %0d of %0d)", tc, owin, no);
                              errors = errors + 1; end
        if (n_op != no * P) begin $display("[tb_acc] ERROR: %0d E9 planes, expected %0d", n_op, no * P);
                                  errors = errors + 1; end
        if (n_done != nrun) begin $display("[tb_acc] ERROR: %0d run_done pulses, expected %0d", n_done, nrun);
                                  errors = errors + 1; end
        if (n_pcr != nrun * NBLK || n_lacr != nla || n_llcr != nrun) begin
            $display("[tb_acc] ERROR: credits p %0d la %0d ll %0d, expected %0d %0d %0d", n_pcr, n_lacr, n_llcr,
                     nrun * NBLK, nla, nrun);
            errors = errors + 1;
        end
        if (|x_err) begin $display("[tb_acc] ERROR: a crossbar's err flag is set (%b)", x_err); errors = errors + 1; end
        $display("[tb_acc] %0s %0s errors=%0d o_planes=%0d runs=%0d end=%0d", (errors == 0) ? "PASS" : "FAIL",
                 case_name, errors, n_op, n_done, tc);
        $finish;
    end
endmodule
