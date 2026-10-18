// ============================================================================
// tb_att_ocarry_top.v — GATE-2 TB for the o-carry station (`att_ocarry_top`).
// Case dirs are written by `v2/sim/simulator/kernels/ocarry_sim.py::dump_case`
// (via `v2/sim/smoke/gen_ocarry_cases.py`); the runner `v2/sim/smoke/run_ocarry_rtl.py`
// compares every event cycle printed here (EV lines) with the model, strictly.
//
// Flow: reset -> weight programming (R*C identity strobes, broadcast) -> origin
// (tc = 0) -> the schedule (sched.txt, cycles from the origin): log_a rows of block b
// written in the S cycles before la_go(b), log_l rows before ll_go, o_part of block b
// streamed as the window-major PV tile emits it: from op_start(b) on, wave t (+PACE
// cycles apart) carries word c of window gi + NGP*t in group gi, one word per cycle.
// O buffer (rework M2): after each run the TB acts as the head's O feed — it reads the run's
// O8 back from e_mem through the read port (or_en / or_addr, every {w, q} once, one per
// cycle, in the OR_READS cycles before o_fed - 1) and pulses o_fed (sched.txt) — so the
// next run's EXP jobs wait for it.
// Checks: every LOG drain word (lo bytes, lo_<r>_<b>.mem), every O8 word
// (o8_<r>.mem) on the output stream and read back from e_mem, the masks, no feed strobe
// while the group's leader is not ready, no o_part FIFO overflow / underflow, no e_mem
// write or LOG read of e while the read port is in use, all words / events present.
//   +VDIR=<case dir>  +CASE=<name>  +PACE=<wave spacing>  +MIDRESET=<tc> (reset at that cycle
//   of a first pass, then re-program and replay the whole schedule; only the second pass is checked)
// ============================================================================
`timescale 1ns / 1ps
`ifndef S_TB
  `define S_TB 16
`endif
`ifndef D_TB
  `define D_TB 8
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
`ifndef NRUN_TB
  `define NRUN_TB 1
`endif
`ifndef NGP_TB
  `define NGP_TB 2
`endif
`ifndef VWP_TB
  `define VWP_TB 13
`endif

module tb_att_ocarry_top;
    localparam S = `S_TB, D = `D_TB, R = `R_TB, C = `C_TB, N_OC = `NOC_TB, NBLK = `NBLK_TB;
    localparam SO = `SO_TB, LN = `LN_TB, NRUN = `NRUN_TB, BUF = 40, NGP = `NGP_TB, VWP = `VWP_TB;
    localparam I    = (R < C) ? R : C;
    localparam P_OC = (S * D) / I;
    localparam W    = (P_OC + N_OC - 1) / N_OC;
    localparam LW   = (I + 4) / 5;
    localparam LS   = $clog2(S);
    localparam WSW  = (W > 1) ? $clog2(W) : 1;
    localparam QW   = $clog2(LW);
    localparam NW   = P_OC * LW;                        // words per block
    localparam NLA  = (NBLK > 1) ? NBLK - 1 : 1;         // la_free pulses per run (>= 1 for the division)
    localparam TWP  = P_OC / NGP;                        // PV waves per block
    localparam ORD  = W * LW;                            // O8 read-back words per run

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg reset = 1'b1;

    reg  [7:0]          weight_in = 0;
    reg                 weight_en = 1'b0;
    reg                 la_wen = 1'b0, la_wslot = 1'b0, la_go = 1'b0;
    reg  [LS-1:0]       la_wr = 0;
    reg  [8:0]          la_wdata = 0;
    reg                 ll_wen = 1'b0, ll_go = 1'b0;
    reg  [LS-1:0]       ll_wr = 0;
    reg  [7:0]          ll_wdata = 0;
    reg                 in_valid = 1'b0;
    reg  [NGP*5*VWP-1:0] in_data = 0;
    wire                la_free, op_free, o8_valid, run_done;
    wire [QW-1:0]       o8_q;
    wire [WSW-1:0]      o8_w;
    wire [N_OC*40-1:0]  o8_data;
    wire [N_OC-1:0]     o8_mask;
    reg                 or_en = 1'b0, o_fed = 1'b0;
    reg  [WSW+QW-1:0]   or_addr = 0;
    wire [N_OC*40-1:0]  or_data;

    att_ocarry_top #(
        .S(S), .D(D), .R(R), .C(C), .BUF(BUF), .P(8), .N_OC(N_OC), .NBLK(NBLK), .SO(SO), .LN(LN), .NGP(NGP), .VWP(VWP)
    ) dut (
        .clk(clk), .reset(reset), .weight_in(weight_in), .weight_en(weight_en),
        .la_wen(la_wen), .la_wslot(la_wslot), .la_wr(la_wr), .la_wdata(la_wdata), .la_go(la_go),
        .la_free(la_free), .ll_wen(ll_wen), .ll_wr(ll_wr), .ll_wdata(ll_wdata), .ll_go(ll_go),
        .in_valid(in_valid), .in_data(in_data), .op_free(op_free),
        .o8_valid(o8_valid), .o8_w(o8_w), .o8_q(o8_q), .o8_data(o8_data), .o8_mask(o8_mask),
        .run_done(run_done), .or_en(or_en), .or_addr(or_addr), .or_data(or_data), .o_fed(o_fed));

    // ---- files ------------------------------------------------------------------
    reg [8*256-1:0] vdir, case_name, fn;
    integer op_t [0:NRUN*NBLK-1], la_t [0:NRUN*NBLK-1], ll_t [0:NRUN-1], of_t [0:NRUN-1];
    reg [159:0] opall  [0:NRUN*NBLK*NW-1];             // o_part of every block (p-major words)
    reg [159:0] optmp  [0:NW-1];
    reg [39:0]  lo_exp [0:NRUN*NBLK*NW-1];
    reg [39:0]  o8_exp [0:NRUN*NW-1];
    reg [8:0]   la_rows [0:NRUN*NBLK*S-1];
    reg [7:0]   ll_rows [0:NRUN*S-1];
    reg [39:0]  tmp40 [0:NW-1];
    reg [7:0]   tmp8 [0:S-1];
    reg [8:0]   tmp9 [0:S-1];
    integer fd, rv, r, b, k, x;
    task load_files;
        begin
            $sformat(fn, "%0s/sched.txt", vdir);
            fd = $fopen(fn, "r");
            for (r = 0; r < NRUN; r = r + 1) begin
                for (b = 0; b < NBLK; b = b + 1) begin rv = $fscanf(fd, "%d", x); op_t[r*NBLK + b] = x; end
                for (b = 0; b < NBLK; b = b + 1) begin rv = $fscanf(fd, "%d", x); la_t[r*NBLK + b] = x; end
                rv = $fscanf(fd, "%d", x); ll_t[r] = x;
                rv = $fscanf(fd, "%d", x); of_t[r] = x;
            end
            $fclose(fd);
            for (r = 0; r < NRUN; r = r + 1) begin
                for (b = 0; b < NBLK; b = b + 1) begin
                    $sformat(fn, "%0s/op_%0d_%0d.mem", vdir, r, b);
                    $readmemh(fn, optmp);
                    for (k = 0; k < NW; k = k + 1) opall[(r*NBLK + b)*NW + k] = optmp[k];
                    $sformat(fn, "%0s/lo_%0d_%0d.mem", vdir, r, b);
                    $readmemh(fn, tmp40);
                    for (k = 0; k < NW; k = k + 1) lo_exp[(r*NBLK + b)*NW + k] = tmp40[k];
                    if (b >= 1) begin
                        $sformat(fn, "%0s/la_%0d_%0d.mem", vdir, r, b);
                        $readmemh(fn, tmp9);
                        for (k = 0; k < S; k = k + 1) la_rows[(r*NBLK + b)*S + k] = tmp9[k];
                    end
                end
                $sformat(fn, "%0s/o8_%0d.mem", vdir, r);
                $readmemh(fn, tmp40);
                for (k = 0; k < NW; k = k + 1) o8_exp[r*NW + k] = tmp40[k];
                $sformat(fn, "%0s/ll_%0d.mem", vdir, r);
                $readmemh(fn, tmp8);
                for (k = 0; k < S; k = k + 1) ll_rows[r*S + k] = tmp8[k];
            end
        end
    endtask

    // ---- time origin: tc = 0 in the first cycle after weight programming --------
    integer tc = 0;
    reg t_on = 1'b0;
    always @(posedge clk) tc <= t_on ? tc + 1 : 0;

    // ---- drivers (inputs of cycle tc, set just after the edge) -------------------
    integer errors = 0, kk, wt, wc, gg, ln, pace, nstream = 0, or_r = 0, or_j = 0;
    reg [159:0] ow;
    always @(posedge clk) begin
        #1;
        la_go = 1'b0; la_wen = 1'b0; ll_go = 1'b0; ll_wen = 1'b0; in_valid = 1'b0; in_data = 0;
        or_en = 1'b0; or_addr = 0; o_fed = 1'b0;
        if (t_on && !reset) begin
            for (k = 0; k < NRUN*NBLK; k = k + 1) begin
                // o_part stream of block k: wave wt word wc at op_t[k] + wt*pace + wc
                for (wt = 0; wt < TWP; wt = wt + 1)
                    if (op_t[k] + wt*pace <= tc && tc < op_t[k] + wt*pace + LW) begin
                        wc = tc - (op_t[k] + wt*pace);
                        if (in_valid) begin
                            $display("[tb_ocarry]   ERROR: two o_part streams overlap at tc=%0d", tc);
                            errors = errors + 1;
                        end
                        in_valid = 1'b1;
                        for (gg = 0; gg < NGP; gg = gg + 1) begin
                            ow = opall[k*NW + (gg + NGP*wt)*LW + wc];
                            for (ln = 0; ln < 5; ln = ln + 1)
                                in_data[(gg*5 + ln)*VWP +: VWP] = ow[ln*32 +: VWP];
                        end
                        nstream = nstream + 1;
                    end
                if ((k % NBLK) >= 1 && la_t[k] - S <= tc && tc < la_t[k]) begin
                    la_wen = 1'b1; la_wslot = (k % NBLK) & 1; la_wr = tc - (la_t[k] - S);
                    la_wdata = la_rows[k*S + (tc - (la_t[k] - S))];
                end
                if ((k % NBLK) >= 1 && la_t[k] == tc) la_go = 1'b1;
            end
            for (r = 0; r < NRUN; r = r + 1) begin
                if (ll_t[r] - S <= tc && tc < ll_t[r]) begin
                    ll_wen = 1'b1; ll_wr = tc - (ll_t[r] - S); ll_wdata = ll_rows[r*S + (tc - (ll_t[r] - S))];
                end
                if (ll_t[r] == tc) ll_go = 1'b1;
                // the O feed: read word j = {w, q} of run r at of_t - ORD - 1 + j, then o_fed
                if (of_t[r] - ORD - 1 <= tc && tc < of_t[r] - 1) begin
                    or_en = 1'b1; or_r = r; or_j = tc - (of_t[r] - ORD - 1);
                    or_addr = ((or_j / LW) << QW) | (or_j % LW);
                end
                if (of_t[r] == tc) o_fed = 1'b1;
            end
        end
    end

    // ---- monitors (sampled at the negative edge: every signal of cycle tc settled)
    integer n_opf = 0, n_laf = 0, n_run = 0, n_lgw = 0, n_lgj = 0, n_o8 = 0, lane, i, p, e, nbad = 0;
    integer lg_run, lg_b, n_or = 0, rq_r = 0, rq_j = 0, rq_p, rq_q;
    reg chk_on = 1'b0, rq_v = 1'b0;
    always @(negedge clk) if (t_on && !reset && chk_on) begin
        if (op_free) begin
            $display("EV opfree %0d %0d %0d", n_opf / NBLK, n_opf % NBLK, tc);
            n_opf = n_opf + 1;
        end
        if (la_free) begin
            $display("EV lafree %0d %0d %0d", n_laf / NLA, n_laf % NLA + 1, tc);
            n_laf = n_laf + 1;
        end
        if (dut.fifo_err && nbad < 8) begin
            $display("[tb_ocarry]   ERROR: o_part FIFO overflow / underflow by tc=%0d", tc);
            nbad = 8; errors = errors + 1;
        end
        if ((dut.lg_sv && !dut.lg_ready) || (dut.ex_sv && !dut.ex_ready)) begin
            if (nbad < 8) $display("[tb_ocarry]   ERROR: feed strobe while the leader is not ready, tc=%0d", tc);
            nbad = nbad + 1; errors = errors + 1;
        end
        // LOG drain words: job n_lgj = (run, b, w) in order
        if (dut.lg_dv) begin
            lg_run = n_lgj / (NBLK * W);
            lg_b   = (n_lgj / W) % NBLK;
            for (i = 0; i < N_OC; i = i + 1) begin
                p = dut.lg_dw * N_OC + i;
                if (p < P_OC && dut.lg_dout[i*40 +: 40] !== lo_exp[(lg_run*NBLK + lg_b)*NW + p*LW + dut.lg_dq]) begin
                    if (nbad < 8)
                        $display("[tb_ocarry]   ERROR: lo run=%0d b=%0d p=%0d q=%0d got=%h exp=%h", lg_run, lg_b, p,
                                 dut.lg_dq, dut.lg_dout[i*40 +: 40], lo_exp[(lg_run*NBLK + lg_b)*NW + p*LW + dut.lg_dq]);
                    nbad = nbad + 1; errors = errors + 1;
                end
            end
            n_lgw = n_lgw + 1;
            if (dut.lg_dq == LW - 1) begin
                $display("EV logend %0d %0d %0d", lg_run, n_lgj % (NBLK * W), tc);
                n_lgj = n_lgj + 1;
            end
        end
        if (o8_valid) begin
            if (o8_q == 0) $display("EV o8first %0d %0d %0d", n_run, o8_w, tc);
            if (o8_q == LW - 1) $display("EV o8last %0d %0d %0d", n_run, o8_w, tc);
            for (i = 0; i < N_OC; i = i + 1) begin
                p = o8_w * N_OC + i;
                if (o8_mask[i] !== (p < P_OC)) begin
                    nbad = nbad + 1; errors = errors + 1;
                end
                if (p < P_OC && o8_data[i*40 +: 40] !== o8_exp[n_run*NW + p*LW + o8_q]) begin
                    if (nbad < 8)
                        $display("[tb_ocarry]   ERROR: O8 run=%0d p=%0d q=%0d got=%h exp=%h", n_run, p, o8_q,
                                 o8_data[i*40 +: 40], o8_exp[n_run*NW + p*LW + o8_q]);
                    nbad = nbad + 1; errors = errors + 1;
                end
            end
            n_o8 = n_o8 + 1;
        end
        if (run_done) begin
            $display("EV rundone %0d %0d", n_run, tc);
            n_run = n_run + 1;
        end
        // O8 read back from e_mem: the read issued last cycle is on or_data now
        if (rq_v) begin
            rq_q = rq_j % LW;
            for (i = 0; i < N_OC; i = i + 1) begin
                rq_p = (rq_j / LW) * N_OC + i;
                if (rq_p < P_OC && or_data[i*40 +: 40] !== o8_exp[rq_r*NW + rq_p*LW + rq_q]) begin
                    if (nbad < 8)
                        $display("[tb_ocarry]   ERROR: e_mem O8 run=%0d p=%0d q=%0d got=%h exp=%h", rq_r, rq_p, rq_q,
                                 or_data[i*40 +: 40], o8_exp[rq_r*NW + rq_p*LW + rq_q]);
                    nbad = nbad + 1; errors = errors + 1;
                end
            end
            n_or = n_or + 1;
        end
        rq_v = or_en; rq_r = or_r; rq_j = or_j;
        if (dut.emem_err) begin
            if (nbad < 8) $display("[tb_ocarry]   ERROR: e_mem written / LOG-read while the O feed reads, tc=%0d", tc);
            nbad = nbad + 1; errors = errors + 1;
        end
        if (o_fed) $display("EV ofed %0d %0d", n_run - 1, tc);
    end

    // ---- main --------------------------------------------------------------------
    integer wi, guard, midreset;
    task program_weights;
        begin
            for (wi = 0; wi < R*C; wi = wi + 1) begin
                weight_en = 1'b1;
                weight_in = ((wi / C) == (wi % C)) ? 8'd1 : 8'd0;      // identity eye, row-major
                @(posedge clk); #1;
            end
            weight_en = 1'b0; weight_in = 8'd0;
        end
    endtask
    initial begin
        if (!$value$plusargs("VDIR=%s", vdir)) begin
            $display("[tb_ocarry] ERROR: +VDIR=<case dir> required"); $finish;
        end
        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("MIDRESET=%d", midreset)) midreset = 0;
        if (!$value$plusargs("PACE=%d", pace)) pace = LW;
        $display("[tb_ocarry] case=%0s S=%0d D=%0d R=%0d C=%0d N_OC=%0d NBLK=%0d SO=%0d LN=%0d NRUN=%0d P_OC=%0d W=%0d LW=%0d",
                 case_name, S, D, R, C, N_OC, NBLK, SO, LN, NRUN, P_OC, W, LW);
        load_files;
        repeat (3) @(posedge clk);
        #1 reset = 1'b0;
        @(posedge clk); #1;
        program_weights;
        if (midreset > 0) begin                                    // first pass, abandoned at tc = midreset
            t_on = 1'b1;
            while (tc < midreset) @(posedge clk);
            #1 reset = 1'b1; t_on = 1'b0;
            repeat (3) @(posedge clk);
            #1 reset = 1'b0;
            @(posedge clk); #1;
            program_weights;
        end
        chk_on = 1'b1;
        t_on = 1'b1;
        guard = 0;
        while ((n_run < NRUN || tc <= of_t[NRUN-1] + 1) && guard < 4000000) begin @(posedge clk); guard = guard + 1; end
        repeat (4) @(posedge clk);
        if (n_run < NRUN) begin
            $display("[tb_ocarry]   ERROR: only %0d of %0d runs done", n_run, NRUN); errors = errors + 1;
        end
        if (dut.fifo_err) begin $display("[tb_ocarry]   ERROR: o_part FIFO overflow / underflow"); errors = errors + 1; end
        if (n_lgj != NRUN * NBLK * W || n_lgw != NRUN * NBLK * W * LW || n_o8 != NRUN * W * LW
            || n_opf != NRUN * NBLK || n_laf != NRUN * (NBLK - 1) || n_or != NRUN * ORD) begin
            $display("[tb_ocarry]   ERROR: counts lg_jobs=%0d lg_words=%0d o8=%0d op_free=%0d la_free=%0d or=%0d",
                     n_lgj, n_lgw, n_o8, n_opf, n_laf, n_or);
            errors = errors + 1;
        end
        $display("[tb_ocarry] %0s %0s errors=%0d", (errors == 0) ? "PASS" : "FAIL", case_name, errors);
        $finish;
    end
endmodule
