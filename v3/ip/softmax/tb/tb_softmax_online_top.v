// ============================================================================
// tb_softmax_online_top.v — GATE-2 testbench of `softmax_online_top` (v3, ip/softmax/spec.md §8). Case files are
// written by ip/softmax/cases.py (model run + GATE 1); the runner ip/softmax/tb/run_softmax_rtl.py compares this
// testbench's event lines with the model's, line for line. The bit-plane pass (2026-10-23): every window is 8 words
// on consecutive cycles (E4 and E5 I bits per channel, E6 2S bits in 2 lanes, E7 S bits): E4 value words (the trims of
// 2026-10-23), the others planes.
//
// The testbench is the module's environment of the model's test graph (ip/softmax/cases.py, sim/core/env.py):
//   score source   every E4 window is driven at the cycle the model planned; before a tile's first word the
//                  testbench checks that the RTL has granted its credit (s_cr of tile u - 2)
//   start source   every start pulse at the planned cycle; busy must be low then
//   lp / la / ll   sinks: take every word, check every value, pulse the credit LPCR / LACR / LLCR cycles after a
//                  unit's last word
// Logged (EV lines, the model's event format): E4 windows, s_cr, start pulses, busy falling (the start port's
// credit), lp / la / ll windows (first word), lp_cr / la_cr / ll_cr. Also checked: no gap inside a window, the word
// counts, one done per run, a watchdog.
// Weights: preloaded at zero time after reset (identity in every crossbar, mode_q EXP / LOG; PLAN R8).
// Cycle 0 is the first cycle after reset.
// ============================================================================
`timescale 1ns / 1ps

`ifndef S_TB
  `define S_TB 32
`endif
`ifndef BKV_TB
  `define BKV_TB 16
`endif
`ifndef N_TB
  `define N_TB 64
`endif
`ifndef R_TB
  `define R_TB 64
`endif
`ifndef C_TB
  `define C_TB 64
`endif
`ifndef NE_TB
  `define NE_TB 2
`endif
`ifndef SCH_TB
  `define SCH_TB 1
`endif
`ifndef LPC_TB
  `define LPC_TB 2
`endif
`ifndef LAC_TB
  `define LAC_TB 2
`endif
`ifndef LLC_TB
  `define LLC_TB 1
`endif

module tb_softmax_online_top;
    localparam S = `S_TB, BKV = `BKV_TB, N = `N_TB, R = `R_TB, C = `C_TB, P = 8, N_EXP = `NE_TB;
    localparam S_CH = `SCH_TB, LP_CAP = `LPC_TB, LA_CAP = `LAC_TB, LL_CAP = `LLC_TB;
    localparam I     = (R < C) ? R : C;
    localparam B     = N / BKV;
    localparam NWB   = (S * BKV) / I;
    localparam LW    = P;                              // plane words per window
    localparam MAXW = 4096, MAXU = 1024, MAXR = 64, MAXLA = 65536, MAXLP = 131072;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg reset = 1'b1;

    reg                  start = 1'b0;
    reg  [S_CH-1:0]      s_v = 0;
    reg  [S_CH*I-1:0]    s_d = 0;
    reg                  lp_cr = 1'b0, la_cr = 1'b0, ll_cr = 1'b0;
    wire                 busy, done, s_cr, la_v, ll_v;
    wire [N_EXP-1:0]     lp_v;
    wire [N_EXP*I-1:0]   lp_d;
    wire [2*S-1:0]       la_d;
    wire [S-1:0]         ll_d;

    softmax_online_top #(.S(S), .BKV(BKV), .N(N), .R(R), .C(C), .P(P), .N_EXP(N_EXP), .S_CH(S_CH),
                         .LP_CAP(LP_CAP), .LA_CAP(LA_CAP), .LL_CAP(LL_CAP)) dut (
        .clk(clk), .reset(reset), .start(start), .busy(busy), .done(done),
        .s_v(s_v), .s_d(s_d), .s_cr(s_cr), .lp_v(lp_v), .lp_d(lp_d), .lp_cr(lp_cr),
        .la_v(la_v), .la_d(la_d), .la_cr(la_cr), .ll_v(ll_v), .ll_d(ll_d), .ll_cr(ll_cr));

    // ---- weights: zero-time preload after reset (PLAN R8) ----------------------------------------------------------
    event preload;
    genvar wg;
    generate
        for (wg = 0; wg < N_EXP; wg = wg + 1) begin : wp_exp
            integer wk;
            always @(preload) begin
                for (wk = 0; wk < R*C; wk = wk + 1)
                    dut.gen_exp[wg].u_feed.u_x.u_dpe.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                dut.gen_exp[wg].u_feed.u_x.u_dpe.mode_q = 2'b10;                // EXP
            end
        end
    endgenerate
    integer ck;
    always @(preload) begin
        for (ck = 0; ck < R*C; ck = ck + 1) begin
            dut.u_carry.u_cexp.u_x.u_dpe.weights[ck] = ((ck / C) == (ck % C)) ? 8'sd1 : 8'sd0;
            dut.u_carry.u_clog.u_x.u_dpe.weights[ck] = ((ck / C) == (ck % C)) ? 8'sd1 : 8'sd0;
        end
        dut.u_carry.u_cexp.u_x.u_dpe.mode_q = 2'b10;                            // EXP
        dut.u_carry.u_clog.u_x.u_dpe.mode_q = 2'b11;                            // LOG
    end

    // ---- case files ------------------------------------------------------------------------------------------------
    reg [8*256-1:0] vdir, fn;
    integer fd, rc, i, k;
    integer ns, nrun, nblk, nla, lpcr, lacr, llcr, tmax;
    integer sw_t [0:MAXW-1], sw_ch [0:MAXW-1], sw_u [0:MAXW-1], sw_w [0:MAXW-1];
    reg [I-1:0]   sw_d [0:MAXW*LW-1];
    integer go_t [0:MAXR-1];
    reg [I-1:0]   lpx [0:MAXLP-1];                      // expected lp planes, window-major in (unit, window) order
    reg [2*S-1:0] lax [0:MAXLA-1];                      // expected la planes, per unit
    reg [S-1:0]   llx [0:MAXR*LW-1];                    // expected ll planes, per run
    reg [I-1:0]   word;
    reg [2*S-1:0] w9;
    reg [S-1:0]   w8;

    task load;
        begin
            $sformat(fn, "%0s/s_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < ns; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d %d", sw_t[i], sw_ch[i], sw_u[i], sw_w[i]);
                for (k = 0; k < LW; k = k + 1) begin rc = $fscanf(fd, "%h", word); sw_d[i*LW + k] = word; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/go.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nrun; i = i + 1) rc = $fscanf(fd, "%d", go_t[i]);
            $fclose(fd);
            $sformat(fn, "%0s/lp_exp.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nblk * NWB * LW; i = i + 1) begin rc = $fscanf(fd, "%h", word); lpx[i] = word; end
            $fclose(fd);
            $sformat(fn, "%0s/la_exp.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nla * LW; i = i + 1) begin rc = $fscanf(fd, "%h", w9); lax[i] = w9; end
            $fclose(fd);
            $sformat(fn, "%0s/ll_exp.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nrun * LW; i = i + 1) begin rc = $fscanf(fd, "%h", w8); llx[i] = w8; end
            $fclose(fd);
        end
    endtask

    // ---- cycle counter, errors -------------------------------------------------------------------------------------
    integer tc = 0;
    always @(posedge clk) if (!reset) tc <= tc + 1;
    integer errors = 0;
    integer n_done = 0;
    always @(posedge clk) if (!reset && done) n_done <= n_done + 1;

    // ---- monitor + drivers (cycle tc, just after the edge) ---------------------------------------------------------
    integer n_scr = 0, n_gocr = 0, n_lpcr = 0, n_lacr = 0, n_llcr = 0;
    integer n_lpw = 0, n_la = 0, n_llw = 0, llw = 0, llrun = 0, gp = 0;
    integer lwin [0:N_EXP-1], lwd [0:N_EXP-1];          // per lane: windows done, word in the window
    integer ucnt [0:MAXU-1], lp_due [0:MAXU-1], la_due [0:MAXU-1];
    integer ll_due [0:MAXR-1];
    integer sp [0:S_CH-1], sc [0:S_CH-1];               // per channel: window (array index), word in it
    reg     busy_d = 1'b0;
    reg     running = 1'b0;
    integer ch, g, gi, u, wl, idx;
    initial begin
        for (g = 0; g < N_EXP; g = g + 1) begin lwin[g] = 0; lwd[g] = 0; end
        for (i = 0; i < MAXU; i = i + 1) begin ucnt[i] = 0; lp_due[i] = -1; la_due[i] = -1; end
        for (i = 0; i < MAXR; i = i + 1) ll_due[i] = -1;
        for (ch = 0; ch < S_CH; ch = ch + 1) begin sp[ch] = ch; sc[ch] = -1; end
    end
    // lane g's k-th lp window is window g + N_EXP*k of the stream (window i = unit*NWB + w on lane i mod N_EXP);
    // channel ch's k-th E4 window is window ch + S_CH*k (windows dealt round robin). Runs at posedge + 2.
    always @(posedge clk) begin
        #2;
        if (running) begin
            // -- DUT outputs of this cycle --
            if (s_cr) begin $display("EV %0d sm.s cr u%0d", tc, n_scr); n_scr = n_scr + 1; end
            if (busy_d && !busy) begin $display("EV %0d sm.go cr u%0d", tc, n_gocr); n_gocr = n_gocr + 1; end
            busy_d = busy;
            for (g = 0; g < N_EXP; g = g + 1)
                if (lp_v[g]) begin
                    gi = g + N_EXP * lwin[g];
                    u  = gi / NWB;
                    wl = gi % NWB;
                    if (lwd[g] == 0) $display("EV %0d sm.lp win ch%0d u%0d w%0d", tc, g, u, wl);
                    idx = gi * LW + lwd[g];
                    if (lp_d[g*I +: I] !== lpx[idx]) begin
                        if (errors < 10)
                            $display("[tb_smol] ERROR lp: unit %0d window %0d plane %0d got %h exp %h",
                                     u, wl, lwd[g], lp_d[g*I +: I], lpx[idx]);
                        errors = errors + 1;
                    end
                    n_lpw = n_lpw + 1;
                    lwd[g] = lwd[g] + 1;
                    if (lwd[g] == LW) begin
                        lwd[g] = 0;
                        lwin[g] = lwin[g] + 1;
                        ucnt[u] = ucnt[u] + 1;
                        if (ucnt[u] == NWB) lp_due[u] = tc + lpcr;
                    end
                end
                else if (lwd[g] != 0) begin
                    $display("[tb_smol] ERROR: gap inside an lp window (lane %0d)", g); errors = errors + 1;
                end
            if (la_v) begin
                if (n_la % LW == 0) $display("EV %0d sm.la win ch0 u%0d w0", tc, n_la / LW);
                if (la_d !== lax[n_la]) begin
                    if (errors < 10) $display("[tb_smol] ERROR la: unit %0d plane %0d got %h exp %h",
                                              n_la / LW, n_la % LW, la_d, lax[n_la]);
                    errors = errors + 1;
                end
                if (n_la % LW == LW - 1) la_due[n_la / LW] = tc + lacr;
                n_la = n_la + 1;
            end
            else if (n_la % LW != 0) begin
                $display("[tb_smol] ERROR: gap inside an la window"); errors = errors + 1;
            end
            if (ll_v) begin
                if (llw == 0) $display("EV %0d sm.ll win ch0 u%0d w0", tc, llrun);
                if (ll_d !== llx[llrun * LW + llw]) begin
                    if (errors < 10) $display("[tb_smol] ERROR ll: run %0d plane %0d got %h exp %h",
                                              llrun, llw, ll_d, llx[llrun * LW + llw]);
                    errors = errors + 1;
                end
                n_llw = n_llw + 1;
                llw = llw + 1;
                if (llw == LW) begin ll_due[llrun] = tc + llcr; llw = 0; llrun = llrun + 1; end
            end
            else if (llw != 0) begin
                $display("[tb_smol] ERROR: gap inside an ll window"); errors = errors + 1;
            end
            // -- drivers --
            lp_cr = 1'b0;
            if (n_lpcr < nblk && lp_due[n_lpcr] == tc) begin
                lp_cr = 1'b1; $display("EV %0d lp_snk.lp cr u%0d", tc, n_lpcr); n_lpcr = n_lpcr + 1;
            end
            la_cr = 1'b0;
            if (n_lacr < nla && la_due[n_lacr] == tc) begin
                la_cr = 1'b1; $display("EV %0d la_snk.la cr u%0d", tc, n_lacr); n_lacr = n_lacr + 1;
            end
            ll_cr = 1'b0;
            if (n_llcr < nrun && ll_due[n_llcr] == tc) begin
                ll_cr = 1'b1; $display("EV %0d ll_snk.ll cr u%0d", tc, n_llcr); n_llcr = n_llcr + 1;
            end
            s_v = 0;
            for (ch = 0; ch < S_CH; ch = ch + 1) begin
                if (sc[ch] < 0 && sp[ch] < ns && sw_t[sp[ch]] == tc) begin
                    sc[ch] = 0;
                    if (sw_ch[sp[ch]] != ch) begin $display("[tb_smol] ERROR: E4 window channel"); errors = errors + 1; end
                    if (sw_w[sp[ch]] == 0 && sw_u[sp[ch]] >= 2 && n_scr <= sw_u[sp[ch]] - 2) begin
                        $display("[tb_smol] ERROR: tile %0d sent at %0d without credit %0d", sw_u[sp[ch]], tc,
                                 sw_u[sp[ch]] - 2);
                        errors = errors + 1;
                    end
                    $display("EV %0d s_src.s win ch%0d u%0d w%0d", tc, ch, sw_u[sp[ch]], sw_w[sp[ch]]);
                end
                if (sc[ch] >= 0) begin
                    s_v[ch] = 1'b1;
                    s_d[ch*I +: I] = sw_d[sp[ch]*LW + sc[ch]];
                    sc[ch] = sc[ch] + 1;
                    if (sc[ch] == LW) begin sc[ch] = -1; sp[ch] = sp[ch] + S_CH; end
                end
            end
            start = 1'b0;
            if (gp < nrun && go_t[gp] == tc) begin
                if (busy) begin $display("[tb_smol] ERROR: run %0d started at %0d while busy", gp, tc);
                                errors = errors + 1; end
                start = 1'b1;
                $display("EV %0d go_src.go win ch0 u%0d w0", tc, gp);
                gp = gp + 1;
            end
        end
    end

    // ---- main ----------------------------------------------------------------------------------------------------
    reg [8*128-1:0] case_name;
    initial begin
        if (!$value$plusargs("VDIR=%s", vdir)) begin $display("[tb_smol] ERROR: +VDIR required"); $finish; end
        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("NS=%d", ns)) ns = 0;
        if (!$value$plusargs("NRUN=%d", nrun)) nrun = 0;
        if (!$value$plusargs("NBLK=%d", nblk)) nblk = 0;
        if (!$value$plusargs("NLA=%d", nla)) nla = 0;
        if (!$value$plusargs("LPCR=%d", lpcr)) lpcr = 1;
        if (!$value$plusargs("LACR=%d", lacr)) lacr = 1;
        if (!$value$plusargs("LLCR=%d", llcr)) llcr = 1;
        if (!$value$plusargs("TMAX=%d", tmax)) tmax = 100000;
        $display("[tb_smol] case=%0s S=%0d BKV=%0d N=%0d R=%0d C=%0d N_EXP=%0d S_CH=%0d NWB=%0d LW=%0d runs=%0d",
                 case_name, S, BKV, N, R, C, N_EXP, S_CH, NWB, LW, nrun);
        load;
        repeat (3) @(posedge clk);
        #1 reset = 1'b0;
        -> preload;
        running = 1'b1;
        while ((llrun < nrun || n_llcr < nrun || n_lpcr < nblk || n_lacr < nla || n_gocr < nrun) && tc < tmax)
            @(posedge clk);
        repeat (4) @(posedge clk);
        #4;
        if (tc >= tmax) begin $display("[tb_smol] ERROR: watchdog (tc %0d, runs out %0d of %0d)", tc, llrun, nrun);
                              errors = errors + 1; end
        if (n_lpw != nblk * NWB * LW) begin $display("[tb_smol] ERROR: %0d lp words, expected %0d", n_lpw,
                                                     nblk * NWB * LW); errors = errors + 1; end
        if (n_la != nla * LW) begin $display("[tb_smol] ERROR: %0d la planes, expected %0d", n_la, nla * LW);
                                    errors = errors + 1; end
        if (n_llw != nrun * LW) begin $display("[tb_smol] ERROR: %0d ll planes, expected %0d", n_llw, nrun * LW);
                                      errors = errors + 1; end
        if (n_done != nrun) begin $display("[tb_smol] ERROR: %0d done pulses, expected %0d", n_done, nrun);
                                  errors = errors + 1; end
        $display("[tb_smol] %0s %0s errors=%0d lp_words=%0d la=%0d ll=%0d end=%0d", (errors == 0) ? "PASS" : "FAIL",
                 case_name, errors, n_lpw, n_la, n_llw, tc);
        $finish;
    end
endmodule
