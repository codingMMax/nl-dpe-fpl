// ============================================================================
// tb_dimm_tile_top.v — GATE-2 testbench of `dimm_tile_top` (v3, ip/dimm/spec.md §8). Case files are written by
// ip/dimm/cases.py (model run + GATE 1); the runner ip/dimm/tb/run_dimm_rtl.py compares this testbench's event
// lines with the model's, line for line. The bit-plane pass (2026-10-23): every window is 8 words on consecutive
// cycles (A_W, L_B and I*ceil(EC/8) bits per channel): planes on A and B, value words on C (the trims of 2026-10-23,
// compared word for word); the engine state compared across slices is the wave sequencer's.
//
// The testbench is the module's environment of the model's test graph (sim/core/env.py):
//   A / B / job sources   every input window (and job) is driven at the cycle the model planned; before a unit's
//                         first word the testbench checks that the RTL has already granted the unit's credit
//                         (credit u - CAP for u >= CAP; for a job, j_rdy), so the replay never runs ahead of the RTL
//   C sink                takes every word, checks every value, and pulses c_cr CRLAT cycles after a job's last word
// Logged (EV lines, the model's event format): input windows, a_cr / b_cr pulses, j_rdy rising (the job port's
// credit), output windows (first word), c_cr pulses.
// Also checked: every slice's A and B memory image at the end (gate D1), every slice's wave sequencer equal to slice
// 0's in every cycle (gate D2), the converters' overrun probes, every crossbar's sticky err flag, a watchdog.
// Weights: preloaded at zero time after reset (identity in every crossbar; in the farm stacked twice where R >= 2I,
// spec §10 item 11; mode_q LOG / EXP; PLAN R8).
// Cycle 0 is the first cycle after reset.
// ============================================================================
`timescale 1ns / 1ps

`ifndef M_TB
  `define M_TB 32
`endif
`ifndef N_TB
  `define N_TB 16
`endif
`ifndef K_TB
  `define K_TB 16
`endif
`ifndef R_TB
  `define R_TB 64
`endif
`ifndef C_TB
  `define C_TB 64
`endif
`ifndef NE_TB
  `define NE_TB 32
`endif
`ifndef XS_TB
  `define XS_TB 16
`endif
`ifndef ADOM_TB
  `define ADOM_TB 0
`endif
`ifndef ASIG_TB
  `define ASIG_TB 1
`endif
`ifndef ACH_TB
  `define ACH_TB 1
`endif
`ifndef ASL_TB
  `define ASL_TB 1
`endif
`ifndef AFR_TB
  `define AFR_TB 0
`endif
`ifndef AROWS_TB
  `define AROWS_TB 1
`endif
`ifndef BDOM_TB
  `define BDOM_TB 0
`endif
`ifndef BSIG_TB
  `define BSIG_TB 1
`endif
`ifndef BCH_TB
  `define BCH_TB 1
`endif
`ifndef BSL_TB
  `define BSL_TB 1
`endif
`ifndef TRB_TB
  `define TRB_TB 1
`endif
`ifndef NAR_TB
  `define NAR_TB 0
`endif
`ifndef CCAP_TB
  `define CCAP_TB 2
`endif

module tb_dimm_tile_top;
    localparam M = `M_TB, N = `N_TB, K = `K_TB, R = `R_TB, C = `C_TB, P = 8, N_E = `NE_TB, X_S = `XS_TB;
    localparam A_DOM = `ADOM_TB, A_SIGNED = `ASIG_TB, A_CH = `ACH_TB, A_SLOTS = `ASL_TB, A_FRAME = `AFR_TB;
    localparam A_ROWS = `AROWS_TB, B_DOM = `BDOM_TB, B_SIGNED = `BSIG_TB, B_CH = `BCH_TB, B_SLOTS = `BSL_TB;
    localparam TRANS_B = `TRB_TB, NARROW = `NAR_TB, C_CAP = `CCAP_TB;
    localparam I    = (R < C) ? R : C;
    localparam NG   = N_E / K;
    localparam NB   = I / N;
    localparam W_K  = (M * N) / I;
    localparam T_W  = W_K / NG;
    localparam LW   = P;                                           // words per window (plane words)
    localparam XS   = (X_S < N_E) ? X_S : N_E;
    localparam NS   = N_E / XS;
    localparam KS   = XS / NG;
    localparam AGC  = (P < XS) ? P : XS;
    localparam A_GRP = 1 << ($clog2(AGC + 1) - 1);
    localparam B_GRP = (A_GRP < KS) ? A_GRP : KS;
    localparam NAB  = XS / A_GRP;
    localparam NBM  = KS / B_GRP;
    localparam VW   = 9 + $clog2(K);
    localparam EC   = (NARROW > 0) ? 8 : VW;
    localparam LBL  = TRANS_B ? K : N;
    localparam A_W  = A_ROWS * K;                                  // A plane word bits
    localparam ECL  = (EC + 7) / 8;                                // C bytes per element
    localparam CWW  = I * ECL;                                     // C word bits per channel
    localparam LWA  = P;
    localparam LWB  = P;
    localparam SAW  = (A_SLOTS > 1) ? $clog2(A_SLOTS) : 1;
    localparam SBW  = (B_SLOTS > 1) ? $clog2(B_SLOTS) : 1;
    localparam CAP_A = (A_FRAME == 0) ? 2 : A_SLOTS;
    localparam UN_A  = (A_FRAME == 0) ? NB : (M / A_ROWS);         // A windows per credit unit
    localparam CAP_B = TRANS_B ? 2 : 4 * B_CH;                     // TRANS_B = 0: 4 B_CH rows (spec §10 item 6)
    localparam UN_B  = TRANS_B ? N : 1;
    localparam MAXW = 4096, MAXJ = 256, MAXC = 8192, MAXMEM = 65536;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg reset = 1'b1;

    reg  [A_CH-1:0]     a_v = 0;
    reg  [A_CH*A_W-1:0] a_d = 0;
    reg  [B_CH-1:0]     b_v = 0;
    reg  [B_CH*LBL-1:0] b_d = 0;
    reg                 j_v = 0, j_ra = 0;
    reg  [SAW-1:0]      j_a = 0;
    reg  [SBW-1:0]      j_b = 0;
    reg                 c_cr = 0;
    wire                a_cr, b_cr, j_rdy, busy;
    wire [NG-1:0]       c_v;
    wire [NG*CWW-1:0]   c_d;

    dimm_tile_top #(
        .M(M), .N(N), .K(K), .R(R), .C(C), .P(P), .N_E(N_E), .X_S(X_S),
        .A_DOM(A_DOM), .A_SIGNED(A_SIGNED), .A_CH(A_CH), .A_SLOTS(A_SLOTS), .A_FRAME(A_FRAME), .A_ROWS(A_ROWS),
        .B_DOM(B_DOM), .B_SIGNED(B_SIGNED), .B_CH(B_CH), .B_SLOTS(B_SLOTS), .TRANS_B(TRANS_B), .NARROW(NARROW),
        .C_CAP(C_CAP)
    ) dut (
        .clk(clk), .reset(reset), .weight_in(8'd0), .weight_en(1'b0),
        .a_v(a_v), .a_d(a_d), .a_cr(a_cr), .b_v(b_v), .b_d(b_d), .b_cr(b_cr),
        .j_v(j_v), .j_a(j_a), .j_b(j_b), .j_ra(j_ra), .j_rdy(j_rdy),
        .c_v(c_v), .c_d(c_d), .c_cr(c_cr), .busy(busy));

    // ---- weights: zero-time preload after reset (PLAN R8) ----------------------------------------------------------
    event preload;
    genvar gs, gf, gch, gm;
    generate
        for (gs = 0; gs < NS; gs = gs + 1) begin : w_sl
            for (gf = 0; gf < XS; gf = gf + 1) begin : w_x
                integer wk;
                always @(preload) begin
                    for (wk = 0; wk < R*C; wk = wk + 1)
                        dut.g_sl[gs].u_slice.g_x[gf].u_dpe.weights[wk] = ((wk / C) == (wk % C) || (R >= 2*I && (wk / C) == (wk % C) + I)) ? 8'sd1 : 8'sd0;   // the identity, stacked twice with X_ADD
                    dut.g_sl[gs].u_slice.g_x[gf].u_dpe.mode_q = 2'b10;          // EXP
                end
            end
        end
        if (A_DOM == 0) begin : w_ca
            for (gch = 0; gch < A_CH; gch = gch + 1) begin : w_c
                integer wk;
                always @(preload) begin
                    for (wk = 0; wk < R*C; wk = wk + 1)
                        dut.g_sa[gch].g_conv.u_conv.u_dpe.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                    dut.g_sa[gch].g_conv.u_conv.u_dpe.mode_q = 2'b11;           // LOG
                end
            end
        end
        if (B_DOM == 0) begin : w_cb
            for (gch = 0; gch < B_CH; gch = gch + 1) begin : w_c
                integer wk;
                always @(preload) begin
                    for (wk = 0; wk < R*C; wk = wk + 1)
                        dut.g_sb[gch].g_conv.u_conv.u_dpe.weights[wk] = ((wk / C) == (wk % C)) ? 8'sd1 : 8'sd0;
                    dut.g_sb[gch].g_conv.u_conv.u_dpe.mode_q = 2'b11;           // LOG
                end
            end
        end
    endgenerate

    // ---- case files ------------------------------------------------------------------------------------------------
    reg [8*256-1:0] vdir, fn;
    integer fd, rc, i, j, w;
    integer na, nb, nj, nc, nmem, crlat, tmax;
    integer aw_t [0:MAXW-1], aw_ch [0:MAXW-1], aw_u [0:MAXW-1], aw_w [0:MAXW-1];
    reg [A_W-1:0] aw_d [0:MAXW*LWA-1];
    integer bw_t [0:MAXW-1], bw_ch [0:MAXW-1], bw_u [0:MAXW-1], bw_w [0:MAXW-1];
    reg [LBL-1:0] bw_d [0:MAXW*LWB-1];
    integer jw_t [0:MAXJ-1], jw_a [0:MAXJ-1], jw_b [0:MAXJ-1], jw_r [0:MAXJ-1];
    reg [CWW-1:0] cx [0:MAXC*LW-1];                     // expected output value words, window-major
    integer mk [0:MAXMEM-1], ms [0:MAXMEM-1], mm [0:MAXMEM-1], ma [0:MAXMEM-1];
    reg [N*8-1:0] md [0:MAXMEM-1];
    reg [A_W-1:0] aword;
    reg [LBL-1:0] bword;
    reg [CWW-1:0] cword;
    reg [N*8-1:0] mword;

    task load;
        begin
            $sformat(fn, "%0s/a_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < na; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d %d", aw_t[i], aw_ch[i], aw_u[i], aw_w[i]);
                for (w = 0; w < LWA; w = w + 1) begin rc = $fscanf(fd, "%h", aword); aw_d[i*LWA + w] = aword; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/b_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nb; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d %d", bw_t[i], bw_ch[i], bw_u[i], bw_w[i]);
                for (w = 0; w < LWB; w = w + 1) begin rc = $fscanf(fd, "%h", bword); bw_d[i*LWB + w] = bword; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/j_win.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nj; i = i + 1) rc = $fscanf(fd, "%d %d %d %d", jw_t[i], jw_a[i], jw_b[i], jw_r[i]);
            $fclose(fd);
            $sformat(fn, "%0s/c_exp.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nc * LW; i = i + 1) begin rc = $fscanf(fd, "%h", cword); cx[i] = cword; end
            $fclose(fd);
            $sformat(fn, "%0s/mem.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < nmem; i = i + 1) begin
                rc = $fscanf(fd, "%d %d %d %d %h", mk[i], ms[i], mm[i], ma[i], mword); md[i] = mword;
            end
            $fclose(fd);
        end
    endtask

    // ---- cycle counter, errors -------------------------------------------------------------------------------------
    integer tc = 0;
    always @(posedge clk) if (!reset) tc <= tc + 1;
    integer errors = 0;

    // ---- monitor + drivers (cycle tc, just after the edge) ---------------------------------------------------------
    integer n_acr = 0, n_bcr = 0, n_jcr = 0, n_ccr = 0, n_cw = 0;
    integer ap [0:A_CH-1], ac [0:A_CH-1], bp [0:B_CH-1], bc [0:B_CH-1];     // per channel: window, word in it
    integer jp = 0;
    integer cwin [0:NG-1], cwd [0:NG-1];                                    // per channel: windows, word in window
    integer cr_due [0:MAXJ-1];
    integer jobs_out = 0, last_t;
    reg     jrdy_d = 1'b1;
    reg     running = 1'b0;
    integer ch, idx, wv, u_need, g, l;
    initial begin
        for (ch = 0; ch < A_CH; ch = ch + 1) begin ap[ch] = ch; ac[ch] = -1; end
        for (ch = 0; ch < B_CH; ch = ch + 1) begin bp[ch] = ch; bc[ch] = -1; end
        for (g = 0; g < NG; g = g + 1) begin cwin[g] = 0; cwd[g] = 0; end
        for (i = 0; i < MAXJ; i = i + 1) cr_due[i] = -1;
    end
    // window index of channel ch's k-th window: ch + CH*k (windows are dealt round robin over the stream).
    // Runs at posedge + 2 (reset is released at posedge + 1, so the first visit is cycle 0).
    always @(posedge clk) begin
        #2;
        if (running) begin
            // -- DUT outputs of this cycle --
            if (a_cr) begin $display("EV %0d dimm.a cr u%0d", tc, n_acr); n_acr = n_acr + 1; end
            if (b_cr) begin $display("EV %0d dimm.b cr u%0d", tc, n_bcr); n_bcr = n_bcr + 1; end
            if (j_rdy && !jrdy_d) begin $display("EV %0d dimm.j cr u%0d", tc, n_jcr); n_jcr = n_jcr + 1; end
            jrdy_d = j_rdy;
            for (g = 0; g < NG; g = g + 1)
                if (c_v[g]) begin
                    if (cwd[g] == 0)
                        $display("EV %0d dimm.c win ch%0d u%0d w%0d", tc, g, cwin[g] / T_W, g + NG * (cwin[g] % T_W));
                    idx = (((cwin[g] / T_W) * W_K) + g + NG * (cwin[g] % T_W)) * LW + cwd[g];
                    if (c_d[g*CWW +: CWW] !== cx[idx]) begin
                        if (errors < 10)
                            $display("[tb_dimm] ERROR value: job %0d window %0d word %0d got %h exp %h",
                                     cwin[g] / T_W, g + NG * (cwin[g] % T_W), cwd[g], c_d[g*CWW +: CWW], cx[idx]);
                        errors = errors + 1;
                    end
                    n_cw = n_cw + 1;
                    cwd[g] = cwd[g] + 1;
                    if (cwd[g] == LW) begin
                        cwd[g] = 0;
                        cwin[g] = cwin[g] + 1;
                        if (g == NG - 1 && cwin[g] % T_W == 0) begin       // the job's last word
                            cr_due[cwin[g] / T_W - 1] = tc + crlat;
                            jobs_out = jobs_out + 1;
                            last_t = tc;
                        end
                    end
                end
                else if (cwd[g] != 0) begin
                    $display("[tb_dimm] ERROR: gap inside an output window (channel %0d)", g); errors = errors + 1;
                end
            // -- drivers --
            c_cr = 1'b0;
            if (n_ccr < nj && cr_due[n_ccr] == tc) begin
                c_cr = 1'b1; $display("EV %0d c_snk.c cr u%0d", tc, n_ccr); n_ccr = n_ccr + 1;
            end
            a_v = 0;
            for (ch = 0; ch < A_CH; ch = ch + 1) begin
                if (ac[ch] < 0 && ap[ch] < na && aw_t[ap[ch]] == tc) begin
                    ac[ch] = 0;
                    if (aw_ch[ap[ch]] != ch) begin $display("[tb_dimm] ERROR: A window channel"); errors = errors + 1; end
                    u_need = aw_u[ap[ch]] - CAP_A;
                    if (aw_w[ap[ch]] == 0 && u_need >= 0 && n_acr <= u_need) begin
                        $display("[tb_dimm] ERROR: A unit %0d sent at %0d without credit %0d", aw_u[ap[ch]], tc, u_need);
                        errors = errors + 1;
                    end
                    $display("EV %0d a_src.a win ch%0d u%0d w%0d", tc, ch, aw_u[ap[ch]], aw_w[ap[ch]]);
                end
                if (ac[ch] >= 0) begin
                    a_v[ch] = 1'b1;
                    a_d[ch*A_W +: A_W] = aw_d[ap[ch]*LWA + ac[ch]];
                    ac[ch] = ac[ch] + 1;
                    if (ac[ch] == LWA) begin ac[ch] = -1; ap[ch] = ap[ch] + A_CH; end
                end
            end
            b_v = 0;
            for (ch = 0; ch < B_CH; ch = ch + 1) begin
                if (bc[ch] < 0 && bp[ch] < nb && bw_t[bp[ch]] == tc) begin
                    bc[ch] = 0;
                    if (bw_ch[bp[ch]] != ch) begin $display("[tb_dimm] ERROR: B window channel"); errors = errors + 1; end
                    u_need = bw_u[bp[ch]] - CAP_B;
                    if (bw_w[bp[ch]] == 0 && u_need >= 0 && n_bcr <= u_need) begin
                        $display("[tb_dimm] ERROR: B unit %0d sent at %0d without credit %0d", bw_u[bp[ch]], tc, u_need);
                        errors = errors + 1;
                    end
                    $display("EV %0d b_src.b win ch%0d u%0d w%0d", tc, ch, bw_u[bp[ch]], bw_w[bp[ch]]);
                end
                if (bc[ch] >= 0) begin
                    b_v[ch] = 1'b1;
                    b_d[ch*LBL +: LBL] = bw_d[bp[ch]*LWB + bc[ch]];
                    bc[ch] = bc[ch] + 1;
                    if (bc[ch] == LWB) begin bc[ch] = -1; bp[ch] = bp[ch] + B_CH; end
                end
            end
            j_v = 1'b0;
            if (jp < nj && jw_t[jp] == tc) begin
                if (!j_rdy) begin $display("[tb_dimm] ERROR: job %0d offered at %0d while j_rdy is low", jp, tc);
                                  errors = errors + 1; end
                j_v = 1'b1; j_a = jw_a[jp]; j_b = jw_b[jp]; j_ra = jw_r[jp];
                $display("EV %0d j_src.j win ch0 u%0d w0", tc, jp);
                jp = jp + 1;
            end
        end
    end

    // ---- gate D2: every slice's wave sequencer equal to slice 0's in every cycle ------------------------------------
    generate
        for (gs = 1; gs < NS; gs = gs + 1) begin : d2
            always @(posedge clk) begin
                #3;
                if (running && ({dut.g_sl[gs].u_slice.u_ctrl.run, dut.g_sl[gs].u_slice.u_ctrl.ph,
                                 dut.g_sl[gs].u_slice.u_ctrl.pc, dut.g_sl[gs].u_slice.u_ctrl.dw,
                                 dut.g_sl[gs].u_slice.u_ctrl.fv, dut.g_sl[gs].u_slice.u_ctrl.fp,
                                 dut.g_sl[gs].u_slice.u_ctrl.db}
                             !== {dut.g_sl[0].u_slice.u_ctrl.run, dut.g_sl[0].u_slice.u_ctrl.ph,
                                 dut.g_sl[0].u_slice.u_ctrl.pc, dut.g_sl[0].u_slice.u_ctrl.dw,
                                 dut.g_sl[0].u_slice.u_ctrl.fv, dut.g_sl[0].u_slice.u_ctrl.fp,
                                 dut.g_sl[0].u_slice.u_ctrl.db})) begin
                    if (errors < 10) $display("[tb_dimm] ERROR D2: slice %0d sequencer differs from slice 0 at %0d", gs, tc);
                    errors = errors + 1;
                end
            end
        end
    endgenerate

    // ---- every crossbar's sticky err flag (the behavior model: a burst while not ready, a gap, a binding drain) ----
    wire [NS*XS-1:0] x_err;
    generate
        for (gs = 0; gs < NS; gs = gs + 1) begin : xe_s
            for (gf = 0; gf < XS; gf = gf + 1) begin : xe_f
                assign x_err[gs*XS + gf] = dut.g_sl[gs].u_slice.g_x[gf].u_dpe.err;
            end
        end
    endgenerate

    // ---- gate D1: the memory images at the end -----------------------------------------------------------------------
    event check_mem;
    integer n_memchk = 0;
    generate
        for (gs = 0; gs < NS; gs = gs + 1) begin : d1
            for (gm = 0; gm < NAB; gm = gm + 1) begin : a
                integer q;
                always @(check_mem)
                    for (q = 0; q < nmem; q = q + 1)
                        if (mk[q] == 0 && ms[q] == gs && mm[q] == gm) begin
                            n_memchk = n_memchk + 1;
                            if (dut.g_sl[gs].u_slice.g_a[gm].mem[ma[q]] !== md[q][NB*8-1:0]) begin
                                if (errors < 10) $display("[tb_dimm] ERROR D1: A slice %0d mem %0d addr %0d", gs, gm, ma[q]);
                                errors = errors + 1;
                            end
                        end
            end
            for (gm = 0; gm < NBM; gm = gm + 1) begin : b
                integer q;
                always @(check_mem)
                    for (q = 0; q < nmem; q = q + 1)
                        if (mk[q] == 1 && ms[q] == gs && mm[q] == gm) begin
                            n_memchk = n_memchk + 1;
                            if (dut.g_sl[gs].u_slice.g_bm[gm].mem[ma[q]] !== md[q]) begin
                                if (errors < 10) $display("[tb_dimm] ERROR D1: B slice %0d mem %0d addr %0d", gs, gm, ma[q]);
                                errors = errors + 1;
                            end
                        end
            end
        end
    endgenerate

    // ---- main ----------------------------------------------------------------------------------------------------
    reg [8*128-1:0] case_name;
    initial begin
        if (!$value$plusargs("VDIR=%s", vdir)) begin $display("[tb_dimm] ERROR: +VDIR required"); $finish; end
        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("NA=%d", na)) na = 0;
        if (!$value$plusargs("NBW=%d", nb)) nb = 0;
        if (!$value$plusargs("NJ=%d", nj)) nj = 0;
        if (!$value$plusargs("NMEM=%d", nmem)) nmem = 0;
        if (!$value$plusargs("CRLAT=%d", crlat)) crlat = 1;
        if (!$value$plusargs("TMAX=%d", tmax)) tmax = 100000;
        nc = nj * W_K;
        $display("[tb_dimm] case=%0s M=%0d N=%0d K=%0d R=%0d C=%0d N_E=%0d NS=%0d KS=%0d NG=%0d T_W=%0d A_GRP=%0d",
                 case_name, M, N, K, R, C, N_E, NS, KS, NG, T_W, A_GRP);
        load;
        repeat (3) @(posedge clk);
        #1 reset = 1'b0;
        -> preload;
        running = 1'b1;
        while ((jobs_out < nj || n_ccr < nj) && tc < tmax) @(posedge clk);
        repeat (4) @(posedge clk);
        #4;
        -> check_mem;
        #1;
        if (tc >= tmax) begin $display("[tb_dimm] ERROR: watchdog (tc %0d, jobs out %0d of %0d)", tc, jobs_out, nj);
                              errors = errors + 1; end
        if (n_memchk != nmem) begin $display("[tb_dimm] ERROR D1: %0d of %0d memory words checked", n_memchk, nmem);
                                    errors = errors + 1; end
        if (n_cw != nc * LW) begin $display("[tb_dimm] ERROR: %0d output words, expected %0d", n_cw, nc * LW);
                                   errors = errors + 1; end
        if (|dut.conv_err) begin $display("[tb_dimm] ERROR: converter overrun probe"); errors = errors + 1; end
        if (|x_err) begin $display("[tb_dimm] ERROR: a farm crossbar's err flag"); errors = errors + 1; end
        $display("[tb_dimm] %0s %0s errors=%0d last_word=%0d words=%0d mem=%0d", (errors == 0) ? "PASS" : "FAIL",
                 case_name, errors, last_t, n_cw, n_memchk);
        $finish;
    end
endmodule
