// ============================================================================
// tb_softmax_online_acc.v — GATE-2 testbench of `softmax_online_acc` (v3, ip/softmax/acc_spec.md §8). Case files are
// written by ip/softmax/acc_cases.py (model run + GATE 1); the runner ip/softmax/tb/run_acc_rtl.py compares this
// testbench's event lines with the model's, line for line.
//
// The testbench is the module's environment of the model's test graph (ip/softmax/acc_cases.py, sim/core/env.py):
//   p source    every E8 window at the cycle the model planned, on its channel; before a tile's first word the
//               testbench checks that the RTL has granted its credit (p_cr of tile u - 2)
//   la source   every E6 row at its planned cycle (credit: la_cr of unit u - 2)
//   ll source   every E7 unit, S rows on consecutive cycles (credit: ll_cr of run r - 1)
//   o sink      takes every E9 word, checks every value, no gap inside a row
// Logged (EV lines, the model's event format): the p / la / ll windows (first word), p_cr / la_cr / ll_cr, the E9 rows
// (first word). Also checked: run_done at the model's cycle, once per run; the word counts; the FIFO, o8_mem and O
// reader probes; a watchdog.
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
    localparam S = `S_TB, D = `D_TB, R = `R_TB, C = `C_TB, BUF = 40, P = 8, N_OC = `NOC_TB, NBLK = `NBLK_TB;
    localparam SO = `SO_TB, LN = `LN_TB, P_CH = `PCH_TB, VWP = `VWP_TB;
    localparam I    = (R < C) ? R : C;
    localparam P_OC = (S * D) / I;
    localparam LW   = (I * 8 + BUF - 1) / BUF;
    localparam NWO  = (D + 4) / 5;
    localparam PWB  = 5 * VWP;
    localparam MAXW = 16384, MAXLA = 65536, MAXR = 64, MAXO = 8192;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg reset = 1'b1;

    reg  [P_CH-1:0]     p_v = 0;
    reg  [P_CH*PWB-1:0] p_d = 0;
    reg                 la_v = 1'b0, ll_v = 1'b0;
    reg  [8:0]          la_d = 0;
    reg  [7:0]          ll_d = 0;
    wire                p_cr, la_cr, ll_cr, o_v, run_done;
    wire [BUF-1:0]      o_d;

    softmax_online_acc #(.S(S), .D(D), .R(R), .C(C), .BUF(BUF), .P(P), .N_OC(N_OC), .NBLK(NBLK), .SO(SO), .LN(LN),
                         .P_CH(P_CH), .VWP(VWP)) dut (
        .clk(clk), .reset(reset), .weight_in(8'd0), .weight_en(1'b0),
        .p_v(p_v), .p_d(p_d), .p_cr(p_cr), .la_v(la_v), .la_d(la_d), .la_cr(la_cr),
        .ll_v(ll_v), .ll_d(ll_d), .ll_cr(ll_cr), .o_v(o_v), .o_d(o_d), .run_done(run_done));

    // ---- weights: zero-time preload after reset (PLAN R8) ----------------------------------------------------------
    event preload;
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
        end
    endgenerate

    // ---- case files ------------------------------------------------------------------------------------------------
    reg [8*256-1:0] vdir, fn;
    integer fd, rc, i, k;
    integer np, nla, nrun, tmax;
    integer pw_t [0:MAXW-1], pw_ch [0:MAXW-1], pw_u [0:MAXW-1], pw_w [0:MAXW-1];
    reg [PWB-1:0] pw_d [0:MAXW*LW-1];
    integer la_t [0:MAXLA-1], la_u [0:MAXLA-1], la_w [0:MAXLA-1];
    reg [8:0]     la_x [0:MAXLA-1];
    integer ll_t [0:MAXR-1], ll_u [0:MAXR-1];
    reg [7:0]     ll_x [0:MAXR*S-1];
    reg [BUF-1:0] ox [0:MAXO*NWO-1];
    integer rd_t [0:MAXR-1];
    reg [PWB-1:0] pword;
    reg [BUF-1:0] word;
    reg [8:0]     w9;
    reg [7:0]     w8;

    task load;
        begin
            $sformat(fn, "%0s/p_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < np; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d %d", pw_t[i], pw_ch[i], pw_u[i], pw_w[i]);
                for (k = 0; k < LW; k = k + 1) begin rc = $fscanf(fd, "%h", pword); pw_d[i*LW + k] = pword; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/la_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nla; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d %h", la_t[i], la_u[i], la_w[i], w9); la_x[i] = w9;
            end
            $fclose(fd);
            $sformat(fn, "%0s/ll_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nrun; i = i + 1) begin
                rc = $fscanf(fd, "%d %d", ll_t[i], ll_u[i]);
                for (k = 0; k < S; k = k + 1) begin rc = $fscanf(fd, "%h", w8); ll_x[i*S + k] = w8; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/o_exp.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nrun * S * NWO; i = i + 1) begin rc = $fscanf(fd, "%h", word); ox[i] = word; end
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
    integer n_pcr = 0, n_lacr = 0, n_llcr = 0, n_done = 0, n_ow = 0, orow = 0, owd = 0;
    integer sp [0:P_CH-1], sc [0:P_CH-1];               // per channel: window (array index), word in it
    integer lap = 0, llp = 0, llc = -1;
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
                if (owd == 0) $display("EV %0d acc.o win ch0 u%0d w0", tc, orow);
                if (o_d !== ox[orow * NWO + owd]) begin
                    if (errors < 10) $display("[tb_acc] ERROR o: row %0d word %0d got %h exp %h", orow, owd, o_d,
                                              ox[orow * NWO + owd]);
                    errors = errors + 1;
                end
                n_ow = n_ow + 1;
                owd = owd + 1;
                if (owd == NWO) begin owd = 0; orow = orow + 1; end
            end
            else if (owd != 0) begin
                $display("[tb_acc] ERROR: gap inside an O8 row (row %0d)", orow); errors = errors + 1;
            end
            if (run_done) begin
                if (n_done >= nrun || rd_t[n_done] != tc) begin
                    $display("[tb_acc] ERROR: run_done at %0d (expected %0d)", tc, (n_done < nrun) ? rd_t[n_done] : -1);
                    errors = errors + 1;
                end
                n_done = n_done + 1;
            end
            if (dut.fifo_err || dut.omem_err || dut.or_err) begin
                if (errors < 10) $display("[tb_acc] ERROR: probe at %0d (fifo %0d, o8_mem %0d, reader %0d)", tc,
                                          dut.fifo_err, dut.omem_err, dut.or_err);
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
                    p_d[ch*PWB +: PWB] = pw_d[sp[ch]*LW + sc[ch]];
                    sc[ch] = sc[ch] + 1;
                    if (sc[ch] == LW) begin sc[ch] = -1; sp[ch] = sp[ch] + P_CH; end
                end
            end
            la_v = 1'b0;
            if (lap < nla && la_t[lap] == tc) begin
                if (la_w[lap] == 0 && la_u[lap] >= 2 && n_lacr <= la_u[lap] - 2) begin
                    $display("[tb_acc] ERROR: alpha unit %0d sent at %0d without credit %0d", la_u[lap], tc,
                             la_u[lap] - 2);
                    errors = errors + 1;
                end
                $display("EV %0d la_src.la win ch0 u%0d w%0d", tc, la_u[lap], la_w[lap]);
                la_v = 1'b1;
                la_d = la_x[lap];
                lap = lap + 1;
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
                ll_d = ll_x[llp * S + llc];
                llc = llc + 1;
                if (llc == S) begin llc = -1; llp = llp + 1; end
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
        $display("[tb_acc] case=%0s S=%0d D=%0d R=%0d C=%0d N_OC=%0d NBLK=%0d SO=%0d LN=%0d P_CH=%0d P_OC=%0d LW=%0d runs=%0d",
                 case_name, S, D, R, C, N_OC, NBLK, SO, LN, P_CH, P_OC, LW, nrun);
        load;
        repeat (3) @(posedge clk);
        #1 reset = 1'b0;
        -> preload;
        running = 1'b1;
        while ((orow < nrun * S || n_done < nrun || n_pcr < nrun * NBLK || n_lacr < nla / S || n_llcr < nrun)
               && tc < tmax)
            @(posedge clk);
        repeat (4) @(posedge clk);
        #4;
        if (tc >= tmax) begin $display("[tb_acc] ERROR: watchdog (tc %0d, rows out %0d of %0d)", tc, orow, nrun * S);
                              errors = errors + 1; end
        if (n_ow != nrun * S * NWO) begin $display("[tb_acc] ERROR: %0d O8 words, expected %0d", n_ow, nrun * S * NWO);
                                          errors = errors + 1; end
        if (n_done != nrun) begin $display("[tb_acc] ERROR: %0d run_done pulses, expected %0d", n_done, nrun);
                                  errors = errors + 1; end
        if (n_pcr != nrun * NBLK || n_lacr != nla / S || n_llcr != nrun) begin
            $display("[tb_acc] ERROR: credits p %0d la %0d ll %0d, expected %0d %0d %0d", n_pcr, n_lacr, n_llcr,
                     nrun * NBLK, nla / S, nrun);
            errors = errors + 1;
        end
        $display("[tb_acc] %0s %0s errors=%0d o_words=%0d runs=%0d end=%0d", (errors == 0) ? "PASS" : "FAIL",
                 case_name, errors, n_ow, n_done, tc);
        $finish;
    end
endmodule
