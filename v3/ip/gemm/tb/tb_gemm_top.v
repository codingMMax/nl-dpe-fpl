// ============================================================================
// tb_gemm_top.v — GATE-2 testbench of `gemm_top` (v3, ip/gemm/spec.md §7). Case files are written by
// ip/gemm/cases.py (model run + GATE 1); the runner ip/gemm/tb/run_gemm_gate2.py compares this testbench's event
// lines with the model's, line for line.
//
// Forked 2026-10-23 from v2's Stage-2 testbench (git show v2-final:v2/sim/tb/tb_gemm_top.v; sha256 in
// tools/copies.lock and the LEDGER) for the bit-plane interface: the weights are preloaded, the rows are plane words
// replayed at the model's cycles, the wide S_col probe is gone (model-only, spec §1).
//
// The testbench is the module's environment of the model's test graph (simulator/models/fc.py):
//   X source   row u's burst starts at the cycle the model planned: 8 plane words of every lane, one per cycle, under
//              act_en; before it the testbench checks that the RTL has already given the row's credit (credit u - 1
//              for u >= 1: MSB_SA_Ready rose u times) and that MSB_SA_Ready is high
//   Y sink     takes every output plane word (out_valid) and checks every lane's word
// Logged (EV lines, the model's event format): the rows' lane windows, the MSB_SA_Ready rises (the act port's
// credit), the output windows (first word).
// Also checked: 8 output words per row without a gap, dpe_done exactly one cycle after each row's last word, every
// tile's sticky `err` (a burst while not ready, a gap in a burst, the output register binding), a watchdog.
// Weights: preloaded at zero time after reset from w.mem (tile v*H + h, row-major; mode_q REGULAR; PLAN R8).
// Cycle 0 is the first cycle after reset.
// ============================================================================
`timescale 1ns / 1ps

`ifndef K_TB
  `define K_TB 128
`endif
`ifndef N_TB
  `define N_TB 128
`endif
`ifndef R_TB
  `define R_TB 64
`endif
`ifndef C_TB
  `define C_TB 64
`endif
// P29 pass extent (default: the whole array)
`ifndef RU_TB
  `define RU_TB `R_TB
`endif
`ifndef CU_TB
  `define CU_TB `C_TB
`endif

module tb_gemm_top;
    localparam K = `K_TB, N = `N_TB, R = `R_TB, C = `C_TB, P = 8, RU = `RU_TB, CU = `CU_TB;
    localparam V = (K + R - 1) / R;
    localparam H = (N + C - 1) / C;
    localparam MAXM = 64;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg reset = 1'b1;

    reg  [V*RU-1:0] act_in = 0;
    reg             act_en = 1'b0;
    wire            MSB_SA_Ready, dpe_done, reg_full, out_valid;
    wire [H*CU-1:0] data_out;

    gemm_top #(
        .K(K), .N(N), .R(R), .C(C), .P(P), .ROWS_USED(RU), .COLS_USED(CU)
    ) dut (
        .clk(clk), .reset(reset), .act_in(act_in), .act_en(act_en), .weight_in(8'd0), .weight_en(1'b0),
        .MSB_SA_Ready(MSB_SA_Ready), .data_out(data_out), .dpe_done(dpe_done), .reg_full(reg_full),
        .out_valid(out_valid));

    // ---- weights: zero-time preload after reset (PLAN R8) ----------------------------------------------------------
    reg [7:0] wmem [0:V*H*R*C-1];
    event preload;
    wire [V*H-1:0] tile_err;
    genvar gv, gh;
    generate
        for (gv = 0; gv < V; gv = gv + 1) begin : w_v
            for (gh = 0; gh < H; gh = gh + 1) begin : w_h
                integer wk;
                always @(preload) begin
                    for (wk = 0; wk < R*C; wk = wk + 1)
                        dut.gen_v[gv].gen_h[gh].u_dpe.weights[wk] = wmem[(gv*H + gh)*R*C + wk];
                    dut.gen_v[gv].gen_h[gh].u_dpe.mode_q = 2'b00;                  // REGULAR
                end
                assign tile_err[gv*H + gh] = dut.gen_v[gv].gen_h[gh].u_dpe.err;
            end
        end
    endgenerate

    // ---- case files ------------------------------------------------------------------------------------------------
    reg [8*256-1:0] vdir, fn;
    integer fd, rc, i, b, v, h;
    integer M, tmax;
    integer at_t [0:MAXM-1];                     // row u's burst cycle
    reg [RU-1:0] ax [0:MAXM*V*P-1];              // row u, lane v, plane b
    reg [CU-1:0] yx [0:MAXM*H*P-1];              // row u, lane h, plane b
    reg [RU-1:0] aword;
    reg [CU-1:0] yword;

    task load;
        begin
            $sformat(fn, "%0s/w.mem", vdir);
            $readmemh(fn, wmem);
            $sformat(fn, "%0s/act.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < M; i = i + 1) begin
                rc = $fscanf(fd, "%d", at_t[i]);
                for (b = 0; b < V*P; b = b + 1) begin rc = $fscanf(fd, "%h", aword); ax[i*V*P + b] = aword; end
            end
            $fclose(fd);
            $sformat(fn, "%0s/y_exp.txt", vdir);
            fd = $fopen(fn, "r");
            for (i = 0; i < M*H*P; i = i + 1) begin rc = $fscanf(fd, "%h", yword); yx[i] = yword; end
            $fclose(fd);
        end
    endtask

    // ---- cycle counter, errors -------------------------------------------------------------------------------------
    integer tc = 0;
    always @(posedge clk) if (!reset) tc <= tc + 1;
    integer errors = 0;

    // ---- monitor + driver (cycle tc, just after the edge) ----------------------------------------------------------
    // Runs at posedge + 2 (reset is released at posedge + 1, so the first visit is cycle 0).
    integer n_cr = 0, n_done = 0, n_w = 0;
    integer ap = 0, ab = -1;                     // the row being sent, its plane (-1: none)
    integer orow = 0, ow = 0;                    // the output row, its plane
    integer done_due = -1, last_t = -1;
    reg     rdy_d = 1'b1;
    reg     running = 1'b0;
    always @(posedge clk) begin
        #2;
        if (running) begin
            // -- DUT outputs of this cycle --
            if (MSB_SA_Ready && !rdy_d) begin $display("EV %0d gemm.act cr u%0d", tc, n_cr); n_cr = n_cr + 1; end
            rdy_d = MSB_SA_Ready;
            if (out_valid) begin
                if (orow >= M) begin
                    if (errors < 10) $display("[tb_gemm] ERROR: an output word after the last row at %0d", tc);
                    errors = errors + 1;
                end else begin
                    if (ow == 0)
                        for (h = 0; h < H; h = h + 1) $display("EV %0d gemm.out win ch%0d u%0d w%0d", tc, h, orow, h);
                    for (h = 0; h < H; h = h + 1)
                        if (data_out[h*CU +: CU] !== yx[(orow*H + h)*P + ow]) begin
                            if (errors < 10)
                                $display("[tb_gemm] ERROR value: row %0d lane %0d plane %0d got %h exp %h", orow, h, ow,
                                         data_out[h*CU +: CU], yx[(orow*H + h)*P + ow]);
                            errors = errors + 1;
                        end
                    n_w = n_w + 1;
                    ow = ow + 1;
                    if (ow == P) begin ow = 0; orow = orow + 1; done_due = tc + 1; last_t = tc; end
                end
            end else if (ow != 0) begin
                $display("[tb_gemm] ERROR: a gap inside row %0d's output window at %0d", orow, tc);
                errors = errors + 1;
            end
            if (dpe_done) begin
                if (tc != done_due) begin
                    $display("[tb_gemm] ERROR: dpe_done at %0d, the last output word + 1 is %0d", tc, done_due);
                    errors = errors + 1;
                end
                n_done = n_done + 1;
            end else if (tc == done_due) begin
                $display("[tb_gemm] ERROR: no dpe_done at %0d (one cycle after the row's last word)", tc);
                errors = errors + 1;
            end
            // -- driver: the X source --
            act_en = 1'b0;
            act_in = 0;
            if (ab < 0 && ap < M && at_t[ap] == tc) begin
                if (!MSB_SA_Ready || (ap >= 1 && n_cr < ap)) begin
                    $display("[tb_gemm] ERROR: row %0d sent at %0d without its credit (credits %0d, ready %b)", ap,
                             tc, n_cr, MSB_SA_Ready);
                    errors = errors + 1;
                end
                for (v = 0; v < V; v = v + 1) $display("EV %0d x.act win ch%0d u%0d w%0d", tc, v, ap, v);
                ab = 0;
            end
            if (ab >= 0) begin
                act_en = 1'b1;
                for (v = 0; v < V; v = v + 1) act_in[v*RU +: RU] = ax[(ap*V + v)*P + ab];
                ab = ab + 1;
                if (ab == P) begin ab = -1; ap = ap + 1; end
            end
        end
    end

    // ---- main ----------------------------------------------------------------------------------------------------
    reg [8*128-1:0] case_name;
    initial begin
        if (!$value$plusargs("VDIR=%s", vdir)) begin $display("[tb_gemm] ERROR: +VDIR required"); $finish; end
        if (!$value$plusargs("CASE=%s", case_name)) case_name = "case";
        if (!$value$plusargs("M=%d", M)) M = 1;
        if (!$value$plusargs("TMAX=%d", tmax)) tmax = 100000;
        if (M > MAXM) begin $display("[tb_gemm] ERROR: M %0d > MAXM %0d", M, MAXM); $finish; end
        $display("[tb_gemm] case=%0s M=%0d K=%0d N=%0d R=%0d C=%0d V=%0d H=%0d ROWS_USED=%0d COLS_USED=%0d",
                 case_name, M, K, N, R, C, V, H, RU, CU);
        load;
        repeat (3) @(posedge clk);
        #1 reset = 1'b0;
        -> preload;
        running = 1'b1;
        while ((orow < M || tc <= done_due) && tc < tmax) @(posedge clk);
        repeat (4) @(posedge clk);
        #4;
        if (tc >= tmax) begin $display("[tb_gemm] ERROR: watchdog (tc %0d, rows out %0d of %0d)", tc, orow, M);
                              errors = errors + 1; end
        if (n_w != M * P) begin $display("[tb_gemm] ERROR: %0d output words, expected %0d", n_w, M * P);
                                errors = errors + 1; end
        if (n_done != M) begin $display("[tb_gemm] ERROR: %0d dpe_done pulses, expected %0d", n_done, M);
                               errors = errors + 1; end
        if (n_cr != M) begin $display("[tb_gemm] ERROR: %0d MSB_SA_Ready credits, expected %0d", n_cr, M);
                             errors = errors + 1; end
        if (|tile_err) begin $display("[tb_gemm] ERROR: a tile's err flag is set (%b)", tile_err);
                             errors = errors + 1; end
        $display("[tb_gemm] %0s %0s errors=%0d last_word=%0d words=%0d", (errors == 0) ? "PASS" : "FAIL",
                 case_name, errors, last_t, n_w);
        $finish;
    end
endmodule
