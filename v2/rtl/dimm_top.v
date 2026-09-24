// ============================================================================
// dimm_top.v — NL-DPE DIMM (pool/farm, true overlap), v2 clean-room Stage 4.
//
// Ground truth: `v2/spec/dimm.md` (v0.1) + `v2/sim/dimm_sim.py` (values ≡
// `v2/oracle/dimm_ref.py`; cycles = `DimmCycleModel`). Hand-written from the
// charters; the certified primitive (`dpe`, spec v2.0.2) is instantiated
// unchanged inside the pool/farm modules.
//
// Operator:  C[m,n] = sum_k ACAM_EXP(trunc8(LA[k*M+m] + LB[k*N+n]))
//   LA = ACAM_LOG(A) through N_A identity crossbars (mode LOG, 2'b11)
//   LB = ACAM_LOG(B) through N_B identity crossbars (mode LOG)
//   farm = N_E identity crossbars (mode EXP, 2'b10) + exact int32 reduction
//
// Datapath (modules in this file, datapath order):
//   dimm_wprog     identity-eye broadcast + per-pool ACAM mode pins   [agent]
//   dimm_log_pool  n x dpe(identity|LOG): buffer feed -> LA/LB writes  [you]
//   dimm_exp_farm  n x dpe(identity|EXP): CLB add feed -> byte drain   [you]
//   dimm_reduce    acc[M*N] int32 RMW + exact sum + C serializer       [you]
//   dimm_sched     pool starts, round-robin windows, block-k gating    [you]
//   dimm_top       buffers, loader, instances, probe contract          [agent]
//
// True overlap (no policy parameter): A is written TRANSPOSED into `a_buf`
// (address k*M+m), so producer windows read sequentially and LA_T is written
// sequentially; B is row-major (address k*N+n). Farm block k starts when the
// producers finished the k-th slices:
//     a_win_total >= ceil((k+1)*M / I)   and   b_win_total >= ceil((k+1)*N / I)
// with I = min(R,C) the identity-pass capacity (F5). Window q (producer or
// farm) covers elements [q*I, q*I+I) of its production-order stream.
//
// Values: int8 linear in -> int32 linear out (matches dimm_sim/dimm_ref).
// Log-domain fusion (skip output-EXP + input-LOG back-to-back) is a deferred
// interface optimization and is NOT part of this baseline.
//
// Setup cycles (excluded from the measured run; reported by the harness):
//   weight programming  WR = R*C broadcast strobes (dimm_wprog)
//   A/B loads           ceil(M*K*8/BUF) + ceil(K*N*8/BUF) stream words
// Runtime: measured = model.total + serialize_cycles exactly (delta_impl = 0,
// dimm.md v0.2 §5):
//   model.total = max(T_A, T_B, T_start + T_E)   (overlap; T_start = fill)
//   serialize   = M*N                            (C words, one per cycle)
// Zero-overhead mechanisms (all verified by the GATE-2 strict gates):
//   * `take_now` window acceptance: word 0 is presented in the very cycle a
//     window is requested, so a window span equals primitive T(p) exactly;
//   * combinational `win_done` (same cycle as dpe_done) and a
//     producer-completion fast issue path in `dimm_sched`, so the farm's
//     first word coincides with the producer's last drain -> fill == T_start;
//   * serializer word 0 in the flush cycle, `done` one cycle after word
//     M*N-1 -> the output stage spans exactly M*N cycles.
// Every fast-path issue is accounted in the scheduler's issued counters, so a
// window is never re-issued (no duplicates, no skipped windows).
//
// TB probe contract (verification-only internals, NOT ports; frozen names):
//   la_q[0:K*M-1] : reg signed [7:0]  — LA_T, index k*M+m
//   lb_q[0:K*N-1] : reg signed [7:0]  — LB,   index k*N+n
//   u_reduce.acc_q[0:M*N-1] : reg signed [31:0] — valid at `done`
//   u_sched.a_win_total / u_sched.b_win_total : integer — completed windows
// ============================================================================

`timescale 1ns / 1ps

module dimm_top #(
    parameter M   = 8,     // A rows / C rows
    parameter N   = 10,    // B cols / C cols
    parameter K   = 6,     // reduction length
    parameter R   = 256,   // dpe rows
    parameter C   = 256,   // dpe cols
    parameter BUF = 40,    // port width (5 bytes/cycle)
    parameter P   = 8,     // activation bit precision
    parameter N_A = 1,     // logA crossbars (derive-by-default from n_E)
    parameter N_B = 1,     // logB crossbars
    parameter N_E = 2      // exp farm crossbars
) (
    input  wire clk,
    input  wire reset,
    // weight programming (identity eye broadcast; WR = R*C strobes)
    input  wire prog_start,
    output wire prog_ready,
    // A/B load streams (row-major bytes, EPS per BUF-wide word)
    input  wire [BUF-1:0] a_in,
    input  wire a_en,
    input  wire [BUF-1:0] b_in,
    input  wire b_en,
    output wire load_ready,
    // run
    input  wire start,
    output wire busy,
    output wire [31:0] data_out,   // C stream, one int32/cycle, row-major
    output wire out_valid,
    output wire done               // 1-cycle pulse after the last C word
);

    localparam EPS  = BUF / 8;
    localparam I    = (R < C) ? R : C;
    localparam A_DEPTH  = M * K;
    localparam B_DEPTH  = K * N;
    localparam LA_DEPTH = K * M;
    localparam LB_DEPTH = K * N;
    localparam ACC_DEPTH = M * N;
    localparam A_WORDS = (A_DEPTH * 8 + BUF - 1) / BUF;
    localparam B_WORDS = (B_DEPTH * 8 + BUF - 1) / BUF;

    function integer clog2_fn;
        input integer val;
        integer i;
        begin
            clog2_fn = 0;
            for (i = val - 1; i > 0; i = i >> 1) clog2_fn = clog2_fn + 1;
        end
    endfunction

    localparam MAXD = (A_DEPTH > B_DEPTH)
                      ? ((A_DEPTH > ACC_DEPTH) ? A_DEPTH : ACC_DEPTH)
                      : ((B_DEPTH > ACC_DEPTH) ? B_DEPTH : ACC_DEPTH);
    localparam AW = clog2_fn(MAXD) + 1;   // byte/window address width

    // ------------------------------------------------------------------
    // buffers + frozen probes
    //   a_buf: raw A written TRANSPOSED at load time (address k*M+m)
    //   b_buf: raw B row-major (address k*N+n)
    //   la_q / lb_q: converted operands (producer write ports below)
    // ------------------------------------------------------------------
    reg signed [7:0] a_buf [0:A_DEPTH-1];
    reg signed [7:0] b_buf [0:B_DEPTH-1];
    reg signed [7:0] la_q  [0:LA_DEPTH-1];   // TB probe (do not rename)
    reg signed [7:0] lb_q  [0:LB_DEPTH-1];   // TB probe (do not rename)

    // ------------------------------------------------------------------
    // load phase (setup; excluded from the measured run)
    //   A element i = m*K + k (row-major stream) -> a_buf[k*M + m]
    //   B element i = k*N + n (row-major stream) -> b_buf[k*N + n]
    //   Counters advance EPS elements per accepted strobe word.
    // ------------------------------------------------------------------
    reg started;
    reg [AW-1:0] a_word, b_word;
    integer lm, lk, ln, j;        // loader (own block)
    integer jj, jw;               // a/b mux / write block
    integer gn, jn;               // farm mux (own block)
    always @(posedge clk) begin
        if (reset) begin
            started <= 1'b0;
            a_word <= 0;
            b_word <= 0;
            lm <= 0;
            lk <= 0;
            ln <= 0;
        end
        else begin
            if (start) started <= 1'b1;

            if (a_en && load_ready) begin
                for (j = 0; j < EPS; j = j + 1) begin
                    if (lm < M) a_buf[lk*M + lm] <= a_in[j*8 +: 8];
                    lk = lk + 1;
                    if (lk == K) begin
                        lk = 0;
                        lm = lm + 1;
                    end
                end
                a_word <= a_word + 1;
            end

            if (b_en && load_ready) begin
                for (j = 0; j < EPS; j = j + 1) begin
                    if (b_word*EPS + j < B_DEPTH)
                        b_buf[b_word*EPS + j] <= b_in[j*8 +: 8];
                end
                b_word <= b_word + 1;
            end
        end
    end

    assign load_ready = prog_ready && !started;

    // ------------------------------------------------------------------
    // dimm_wprog — identity-eye broadcast (complete; agent-owned)
    // ------------------------------------------------------------------
    wire [7:0] w_data;
    wire       w_en;
    wire [1:0] mode_log, mode_exp;

    dimm_wprog #(.R(R), .C(C)) u_wprog (
        .clk(clk),
        .reset(reset),
        .prog_start(prog_start),
        .prog_ready(prog_ready),
        .w_data(w_data),
        .w_en(w_en),
        .mode_log(mode_log),
        .mode_exp(mode_exp)
    );

    // ------------------------------------------------------------------
    // producers — N_A logA pools + N_B logB pools
    //   buffer read ports are muxed combinationally from the internal arrays;
    //   converted bytes are written into la_q / lb_q.
    // ------------------------------------------------------------------
    wire [N_A-1:0]        a_win_start, a_win_busy, a_win_done;
    wire [N_A*AW-1:0]     a_win_base;
    wire [N_A*EPS-1:0]    a_wr_mask;
    wire [N_A*EPS*AW-1:0] a_wr_addrs;
    wire [N_A*BUF-1:0]    a_wr_data;
    wire [(BUF/8)*AW-1:0] a_buf_addrs [0:N_A-1];
    reg  [BUF-1:0]        a_buf_rdata [0:N_A-1];

    wire [N_B-1:0]        b_win_start, b_win_busy, b_win_done;
    wire [N_B*AW-1:0]     b_win_base;
    wire [N_B*EPS-1:0]    b_wr_mask;
    wire [N_B*EPS*AW-1:0] b_wr_addrs;
    wire [N_B*BUF-1:0]    b_wr_data;
    wire [(BUF/8)*AW-1:0] b_buf_addrs [0:N_B-1];
    reg  [BUF-1:0]        b_buf_rdata [0:N_B-1];

    genvar ga, gb, gf;

    generate
        for (ga = 0; ga < N_A; ga = ga + 1) begin : gen_poolA
            dimm_log_pool #(
                .DEPTH(A_DEPTH), .R(R), .C(C), .BUF(BUF), .P(P), .AW(AW)
            ) u_pool (
                .clk(clk),
                .reset(reset),
                .buf_addrs(a_buf_addrs[ga]),
                .buf_rdata(a_buf_rdata[ga]),
                .dpe_load_input_reg(w_en),
                .dpe_weight_data(w_data),
                .nl_dpe_control(mode_log),
                .win_start(a_win_start[ga]),
                .win_base(a_win_base[ga*AW +: AW]),
                .win_busy(a_win_busy[ga]),
                .win_done(a_win_done[ga]),
                .wr_mask(a_wr_mask[ga*EPS +: EPS]),
                .wr_addrs(a_wr_addrs[ga*EPS*AW +: EPS*AW]),
                .wr_data(a_wr_data[ga*BUF +: BUF])
            );
        end
    endgenerate

    generate
        for (gb = 0; gb < N_B; gb = gb + 1) begin : gen_poolB
            dimm_log_pool #(
                .DEPTH(B_DEPTH), .R(R), .C(C), .BUF(BUF), .P(P), .AW(AW)
            ) u_pool (
                .clk(clk),
                .reset(reset),
                .buf_addrs(b_buf_addrs[gb]),
                .buf_rdata(b_buf_rdata[gb]),
                .dpe_load_input_reg(w_en),
                .dpe_weight_data(w_data),
                .nl_dpe_control(mode_log),
                .win_start(b_win_start[gb]),
                .win_base(b_win_base[gb*AW +: AW]),
                .win_busy(b_win_busy[gb]),
                .win_done(b_win_done[gb]),
                .wr_mask(b_wr_mask[gb*EPS +: EPS]),
                .wr_addrs(b_wr_addrs[gb*EPS*AW +: EPS*AW]),
                .wr_data(b_wr_data[gb*BUF +: BUF])
            );
        end
    endgenerate

    // buffer read mux (one EPS-byte word per pool per cycle)
    integer gm;
    always @(*) begin
        for (gm = 0; gm < N_A; gm = gm + 1)
            for (jj = 0; jj < BUF/8; jj = jj + 1)
                a_buf_rdata[gm][jj*8 +: 8] = a_buf[a_buf_addrs[gm][jj*AW +: AW]];
        for (gm = 0; gm < N_B; gm = gm + 1)
            for (jj = 0; jj < BUF/8; jj = jj + 1)
                b_buf_rdata[gm][jj*8 +: 8] = b_buf[b_buf_addrs[gm][jj*AW +: AW]];
    end

    // converted-output write ports (normative padding discard inside pools:
    // one drained word per cycle, per-lane mask)
    integer gw;
    always @(posedge clk) begin
        for (gw = 0; gw < N_A; gw = gw + 1)
            for (jw = 0; jw < EPS; jw = jw + 1)
                if (a_wr_mask[gw*EPS + jw])
                    la_q[a_wr_addrs[gw*EPS*AW + jw*AW +: AW]] <=
                        a_wr_data[gw*BUF + jw*8 +: 8];
        for (gw = 0; gw < N_B; gw = gw + 1)
            for (jw = 0; jw < EPS; jw = jw + 1)
                if (b_wr_mask[gw*EPS + jw])
                    lb_q[b_wr_addrs[gw*EPS*AW + jw*AW +: AW]] <=
                        b_wr_data[gw*BUF + jw*8 +: 8];
    end

    // ------------------------------------------------------------------
    // farm — N_E identity-EXP crossbars
    //   LA_T / LB read ports muxed combinationally; drain ports go to reduce.
    // ------------------------------------------------------------------
    wire [N_E-1:0]        f_win_start, f_win_busy, f_win_done;
    wire [N_E*AW-1:0]     f_win_base, f_k_idx;
    wire [(BUF/8)*AW-1:0] f_la_addrs [0:N_E-1];
    wire [(BUF/8)*AW-1:0] f_lb_addrs [0:N_E-1];
    reg  [BUF-1:0]        f_la_rdata [0:N_E-1];
    reg  [BUF-1:0]        f_lb_rdata [0:N_E-1];
    wire [N_E-1:0]        dr_valid;
    wire [N_E*AW-1:0]     dr_base;
    wire [N_E*EPS-1:0]    dr_mask;
    wire [N_E*BUF-1:0]    dr_data;

    generate
        for (gf = 0; gf < N_E; gf = gf + 1) begin : gen_farm
            dimm_exp_farm #(
                .M(M), .N(N), .K(K), .R(R), .C(C), .BUF(BUF), .P(P), .AW(AW)
            ) u_farm (
                .clk(clk),
                .reset(reset),
                .la_addrs(f_la_addrs[gf]),
                .lb_addrs(f_lb_addrs[gf]),
                .la_rdata(f_la_rdata[gf]),
                .lb_rdata(f_lb_rdata[gf]),
                .dpe_load_input_reg(w_en),
                .dpe_weight_data(w_data),
                .nl_dpe_control(mode_exp),
                .win_start(f_win_start[gf]),
                .win_base(f_win_base[gf*AW +: AW]),
                .k_idx(f_k_idx[gf*AW +: AW]),
                .win_busy(f_win_busy[gf]),
                .win_done(f_win_done[gf]),
                .dr_valid(dr_valid[gf]),
                .dr_base(dr_base[gf*AW +: AW]),
                .dr_mask(dr_mask[gf*EPS +: EPS]),
                .dr_data(dr_data[gf*BUF +: BUF])
            );
        end
    endgenerate

    // LA_T / LB read mux (EPS bytes per farm crossbar per cycle)
    always @(*) begin
        for (gn = 0; gn < N_E; gn = gn + 1)
            for (jn = 0; jn < BUF/8; jn = jn + 1) begin
                f_la_rdata[gn][jn*8 +: 8] =
                    la_q[f_la_addrs[gn][jn*AW +: AW]];
                f_lb_rdata[gn][jn*8 +: 8] =
                    lb_q[f_lb_addrs[gn][jn*AW +: AW]];
            end
    end

    // ------------------------------------------------------------------
    // reduce — acc[M*N] int32 + exact sum + C serializer
    // ------------------------------------------------------------------
    wire farm_all_done;   // from dimm_sched (level: all farm windows done)

    dimm_reduce #(
        .M(M), .N(N), .BUF(BUF), .AW(AW), .NPORT(N_E)
    ) u_reduce (
        .clk(clk),
        .reset(reset),
        .dr_valid(dr_valid),
        .dr_base(dr_base),
        .dr_mask(dr_mask),
        .dr_data(dr_data),
        .flush(farm_all_done),
        .c_data(data_out),
        .c_valid(out_valid),
        .done(done)
    );

    // ------------------------------------------------------------------
    // scheduler — pool/farm window issue + block-k gating
    // ------------------------------------------------------------------
    dimm_sched #(
        .M(M), .N(N), .K(K), .R(R), .C(C), .BUF(BUF),
        .N_A(N_A), .N_B(N_B), .N_E(N_E), .AW(AW)
    ) u_sched (
        .clk(clk),
        .reset(reset),
        .start(start),
        .a_win_start(a_win_start),
        .a_win_base(a_win_base),
        .a_win_done(a_win_done),
        .a_win_busy(a_win_busy),
        .b_win_start(b_win_start),
        .b_win_base(b_win_base),
        .b_win_done(b_win_done),
        .b_win_busy(b_win_busy),
        .f_win_start(f_win_start),
        .f_win_base(f_win_base),
        .f_k_idx(f_k_idx),
        .f_win_done(f_win_done),
        .f_win_busy(f_win_busy),
        .farm_all_done(farm_all_done),
        .busy(busy)
    );

endmodule


// ============================================================================
// dimm_wprog — identity-eye weight programmer (complete; agent-owned)
//
// Broadcasts WR_CYC = R*C WEIGHT strobes to every pool/farm dpe with
//   w_data = 1 if (r == c) else 0     (row-major eye, r = cnt/C, c = cnt%C)
// and holds the per-pool ACAM mode pins: LOG (2'b11) for the producers,
// EXP (2'b10) for the farm. Mode is latched by the dpe on the weight strobes
// (P27), so one broadcast programs all crossbars in parallel.
// ============================================================================
module dimm_wprog #(
    parameter R = 256,
    parameter C = 256
) (
    input  wire clk,
    input  wire reset,
    input  wire prog_start,
    output reg  prog_ready,
    output reg  [7:0] w_data,
    output reg  w_en,
    output wire [1:0] mode_log,
    output wire [1:0] mode_exp
);

    localparam WR_CYC = R * C;
    localparam [1:0] MODE_EXP = 2'b10, MODE_LOG = 2'b11;

    assign mode_log = MODE_LOG;
    assign mode_exp = MODE_EXP;

    reg busy;
    integer wr_cnt;
    integer r, c;

    // strobe + eye payload are combinational from the busy/counter state, so
    // the first and last strobes are not lost (WR_CYC cycles exactly)
    always @(*) begin
        w_en = busy;
        w_data = (r == c) ? 8'd1 : 8'd0;
    end

    always @(posedge clk) begin
        if (reset) begin
            busy <= 1'b0;
            prog_ready <= 1'b0;
            wr_cnt <= 0;
            r <= 0;
            c <= 0;
        end
        else begin
            if (prog_start && !busy && !prog_ready) begin
                busy <= 1'b1;
                wr_cnt <= 0;
                r <= 0;
                c <= 0;
            end
            if (busy) begin
                if (wr_cnt == WR_CYC - 1) begin
                    busy <= 1'b0;
                    prog_ready <= 1'b1;
                end
                else begin
                    wr_cnt <= wr_cnt + 1;
                    if (c == C - 1) begin
                        c <= 0;
                        r <= r + 1;
                    end
                    else c <= c + 1;
                end
            end
        end
    end

endmodule


// ============================================================================
// dimm_log_pool — one producer crossbar (identity | LOG)
//
// Two-context pipelined window engine (required for T_steady fidelity: the
// next window's feed starts at the previous pass's MSB fire while the
// previous drain is still in flight):
//
//   feed context  : latched window -> LCYC ACT words while dpe_ready;
//                   lane j of word w carries e = base + w*EPS + j, 0-padded
//                   when (w*EPS+j >= I) || (e >= DEPTH)  (F6 padding)
//   drain context : OCYC words on reg_full; per-lane write mask discards
//                   padding; win_done pulses the cycle after dpe_done
//   handshake     : win_start is a level held until win_busy acknowledges;
//                   win_busy = feeding || latched (drops at the last feed
//                   word, so the scheduler can keep passes back-to-back)
//
// No cycle counters/constants: everything fires from dpe_ready / reg_full /
// dpe_done.
// ============================================================================
module dimm_log_pool #(
    parameter DEPTH = 64,    // buffer elements (M*K for A, K*N for B)
    parameter R = 256,
    parameter C = 256,
    parameter BUF = 40,
    parameter P = 8,
    parameter AW = 8
) (
    input  wire clk,
    input  wire reset,
    // input-buffer read port (one address per lane, EPS bytes/cycle)
    output reg  [(BUF/8)*AW-1:0] buf_addrs,
    input  wire [BUF-1:0] buf_rdata,
    // weight strobe + payload + ACAM mode (from dimm_wprog, into the dpe)
    input  wire dpe_load_input_reg,
    input  wire [7:0] dpe_weight_data,
    input  wire [1:0] nl_dpe_control,
    // window interface (from dimm_sched): level handshake — `win_start` is
    // held high until `win_busy` acknowledges (the window is latched); the
    // scheduler clears its request when it sees win_busy rise.
    input  wire win_start,
    input  wire [AW-1:0] win_base,
    output wire win_busy,          // cannot accept another window
    output wire win_done,          // combinational: same cycle as dpe_done
    // converted-output write port (top routes to la_q / lb_q):
    // one drained word per cycle; wr_mask[j] gates lane j (padding discard)
    output reg  [(BUF/8)-1:0]    wr_mask,
    output reg  [(BUF/8)*AW-1:0] wr_addrs,
    output reg  [BUF-1:0]        wr_data
);

    localparam EPS  = BUF / 8;
    localparam I    = (R < C) ? R : C;
    localparam LCYC = (R * 8 + BUF - 1) / BUF;
    localparam OCYC = (C * 8 + BUF - 1) / BUF;

    reg  [BUF-1:0] dpe_in;
    reg            dpe_w_buf_en;
    wire           dpe_ready;
    wire [BUF-1:0] dpe_out;
    wire           dpe_done_w;
    wire           dpe_reg_full;

    dpe #(
        .KERNEL_WIDTH(R),
        .NUM_COLS(C),
        .DPE_BUF_WIDTH(BUF),
        .PRECISION(P)
    ) u_dpe (
        .clk(clk),
        .reset(reset),
        .data_in(dpe_in),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(1'b0),
        .w_buf_en(dpe_w_buf_en),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(dpe_load_input_reg),
        .MSB_SA_Ready(dpe_ready),
        .data_out(dpe_out),
        .dpe_done(dpe_done_w),
        .reg_full(dpe_reg_full)
    );

    // ---- contexts ------------------------------------------------------
    reg [AW-1:0] feed_base, lat_base;   // feed window / latched window
    integer      feed_cnt;              // feed word index 0..LCYC-1
    reg          feeding, have_win;
    reg [AW-1:0] drn_base;              // active drain window
    integer      drn_cnt;               // drain word index 0..OCYC-1
    reg          draining;
    reg [AW-1:0] que_base;              // queued drain window (depth 2)
    reg          que_valid;

    integer jf, jd;
    integer acol, ae, fcol, fe, dcol, de;
    // zero-latency acceptance: when idle and the DPE is ready, the requested
    // window starts feeding in the *same* cycle (word 0 presented now), so
    // the window span equals the primitive T(p) exactly
    wire take_now = win_start && !win_busy && dpe_ready;
    wire [AW-1:0] start_base = take_now ? win_base : lat_base;
    wire start_feed = take_now || (have_win && !feeding && dpe_ready);
    wire feed_active = feeding || start_feed;
    wire [AW-1:0] feed_base_w = start_feed ? start_base : feed_base;
    // last feed word of the current window is accepted this cycle
    wire feed_done_w = start_feed ? (LCYC == 1)
                      : (feeding && dpe_ready && (feed_cnt == LCYC-1));
    wire [AW-1:0] feed_done_base = start_feed ? start_base : feed_base;
    wire drain_done_w = draining && dpe_done_w;

    assign win_busy = feeding || have_win;
    assign win_done = draining && dpe_done_w;   // no registered delay

    // feed addresses: lane j reads element e = base + word*EPS + j (clamped
    // when the pass capacity I or the buffer depth DEPTH is exceeded).
    // Addresses depend only on counters — never on the read data — so there
    // is no combinational loop through the top-level buffer mux.
    always @(*) begin
        buf_addrs = 0;
        if (feed_active) begin
            for (jf = 0; jf < EPS; jf = jf + 1) begin
                acol = (start_feed ? 0 : feed_cnt)*EPS + jf;
                ae = feed_base_w + acol;
                if ((acol < I) && (ae < DEPTH)) buf_addrs[jf*AW +: AW] = ae;
            end
        end
    end

    // feed data: zero-padded window bytes; during WEIGHT strobes the same
    // port carries the weight byte (P23)
    always @(*) begin
        dpe_in = 0;
        if (dpe_load_input_reg) begin
            dpe_in = {{(BUF-8){1'b0}}, dpe_weight_data};
        end
        else if (feed_active) begin
            for (jf = 0; jf < EPS; jf = jf + 1) begin
                fcol = (start_feed ? 0 : feed_cnt)*EPS + jf;
                fe = feed_base_w + fcol;
                if ((fcol < I) && (fe < DEPTH))
                    dpe_in[jf*8 +: 8] = buf_rdata[jf*8 +: 8];
            end
        end
    end

    always @(*) dpe_w_buf_en = feed_active && dpe_ready;

    // drain word: per-lane write mask discards padding (F6)
    always @(*) begin
        wr_mask = 0;
        wr_addrs = 0;
        wr_data = 0;
        if (draining && dpe_reg_full) begin
            for (jd = 0; jd < EPS; jd = jd + 1) begin
                dcol = drn_cnt*EPS + jd;
                de = drn_base + dcol;
                if ((dcol < I) && (de < DEPTH)) begin
                    wr_mask[jd] = 1'b1;
                    wr_addrs[jd*AW +: AW] = de;
                    wr_data[jd*8 +: 8] = dpe_out[jd*8 +: 8];
                end
            end
        end
    end

    // window engine (two contexts: feed runs while the previous drains)
    always @(posedge clk) begin
        if (reset) begin
            feed_base <= 0;
            lat_base <= 0;
            feed_cnt <= 0;
            feeding <= 1'b0;
            have_win <= 1'b0;
            drn_base <= 0;
            drn_cnt <= 0;
            draining <= 1'b0;
            que_base <= 0;
            que_valid <= 1'b0;
        end
        else begin
            // latch a requested window that cannot start immediately
            // (level handshake; one per request; `take_now` consumes the
            // request in the same cycle when the DPE is ready)
            if (win_start && !win_busy && !dpe_ready) begin
                lat_base <= win_base;
                have_win <= 1'b1;
            end

            // feed advance
            if (start_feed) begin
                have_win <= 1'b0;
                if (LCYC == 1) feeding <= 1'b0;
                else begin
                    feeding  <= 1'b1;
                    feed_base <= start_base;
                    feed_cnt <= 1;
                end
            end
            else if (feeding && dpe_ready && (feed_cnt != LCYC-1))
                feed_cnt <= feed_cnt + 1;
            else if (feeding && dpe_ready)
                feeding <= 1'b0;

            // window handoff into the drain pipeline (depth 2: a pass's
            // drain can outlast the next pass's feed)
            if (drain_done_w) begin
                if (que_valid) begin
                    drn_base  <= que_base;
                    drn_cnt   <= 0;
                    que_base  <= feed_done_base;
                    que_valid <= feed_done_w;
                end
                else if (feed_done_w) begin
                    drn_base <= feed_done_base;
                    drn_cnt  <= 0;
                end
                else begin
                    draining <= 1'b0;
                end
            end
            else if (feed_done_w) begin
                if (draining) begin
                    que_base  <= feed_done_base;
                    que_valid <= 1'b1;
                end
                else begin
                    draining <= 1'b1;
                    drn_base <= feed_done_base;
                    drn_cnt  <= 0;
                end
            end

            // drain word counter
            if (draining && dpe_reg_full && (drn_cnt != OCYC-1))
                drn_cnt <= drn_cnt + 1;
        end
    end

endmodule


// ============================================================================
// dimm_exp_farm — one farm crossbar (identity | EXP)
//
// Same two-context pipelined engine as dimm_log_pool, with the farm datapath:
//
//   feed : lane j of word w computes e = base + w*EPS + j;
//          valid = (w*EPS+j < I) && (e < M*N)
//          m = e / N, n = e % N
//          u = (la_rdata + lb_rdata)[7:0]     (CLB add + trunc8, F7)
//          reads la_q[k*M+m] / lb_q[k*N+n]; invalid lanes are 0
//   drain: dr_valid per reg_full word, dr_base = base + word*EPS, and a
//          per-lane mask (invalid lanes are NOT accumulated — padding
//          EXP(0)=1 bytes must never enter the reduction)
//
// Handshake identical to dimm_log_pool (win_start level until win_busy
// acknowledges; win_done after dpe_done). No cycle counters/constants.
// ============================================================================
module dimm_exp_farm #(
    parameter M = 8,
    parameter N = 10,
    parameter K = 6,
    parameter R = 256,
    parameter C = 256,
    parameter BUF = 40,
    parameter P = 8,
    parameter AW = 8
) (
    input  wire clk,
    input  wire reset,
    // LA_T / LB read ports (one address per lane, EPS bytes/cycle each)
    output reg  [(BUF/8)*AW-1:0] la_addrs,
    output reg  [(BUF/8)*AW-1:0] lb_addrs,
    input  wire [BUF-1:0] la_rdata,
    input  wire [BUF-1:0] lb_rdata,
    // weight strobe + payload + ACAM mode (from dimm_wprog, into the dpe)
    input  wire dpe_load_input_reg,
    input  wire [7:0] dpe_weight_data,
    input  wire [1:0] nl_dpe_control,
    // window interface (from dimm_sched): level handshake (see dimm_log_pool)
    input  wire win_start,
    input  wire [AW-1:0] win_base,   // element base within M*N
    input  wire [AW-1:0] k_idx,      // reduction block k
    output wire win_busy,            // cannot accept another window
    output wire win_done,            // combinational: same cycle as dpe_done
    // drain port (to dimm_reduce): one word per cycle, per-lane validity mask
    output reg  dr_valid,
    output reg  [AW-1:0] dr_base,
    output reg  [(BUF/8)-1:0] dr_mask,
    output reg  [BUF-1:0] dr_data
);

    localparam EPS  = BUF / 8;
    localparam I    = (R < C) ? R : C;
    localparam LCYC = (R * 8 + BUF - 1) / BUF;
    localparam OCYC = (C * 8 + BUF - 1) / BUF;

    reg  [BUF-1:0] dpe_in;
    reg            dpe_w_buf_en;
    wire           dpe_ready;
    wire [BUF-1:0] dpe_out;
    wire           dpe_done_w;
    wire           dpe_reg_full;

    dpe #(
        .KERNEL_WIDTH(R),
        .NUM_COLS(C),
        .DPE_BUF_WIDTH(BUF),
        .PRECISION(P)
    ) u_dpe (
        .clk(clk),
        .reset(reset),
        .data_in(dpe_in),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(1'b0),
        .w_buf_en(dpe_w_buf_en),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(dpe_load_input_reg),
        .MSB_SA_Ready(dpe_ready),
        .data_out(dpe_out),
        .dpe_done(dpe_done_w),
        .reg_full(dpe_reg_full)
    );

    // ---- contexts ------------------------------------------------------
    reg [AW-1:0] feed_base, lat_base;   // element base within M*N
    reg [AW-1:0] feed_k, lat_k;         // reduction block of the window
    integer      feed_cnt;
    reg          feeding, have_win;
    reg [AW-1:0] drn_base;
    integer      drn_cnt;
    reg          draining;
    reg [AW-1:0] que_base;              // queued drain window (depth 2)
    reg          que_valid;

    integer jf, jd;
    integer acol, ae, am, an, fcol, fe, dcol, de;
    // zero-latency acceptance (see dimm_log_pool): the requested window
    // starts feeding in the same cycle when idle and the DPE is ready
    wire take_now = win_start && !win_busy && dpe_ready;
    wire [AW-1:0] start_base = take_now ? win_base : lat_base;
    wire [AW-1:0] start_k = take_now ? k_idx : lat_k;
    wire start_feed = take_now || (have_win && !feeding && dpe_ready);
    wire feed_active = feeding || start_feed;
    wire [AW-1:0] feed_base_w = start_feed ? start_base : feed_base;
    wire [AW-1:0] feed_k_w = start_feed ? start_k : feed_k;
    wire feed_done_w = start_feed ? (LCYC == 1)
                      : (feeding && dpe_ready && (feed_cnt == LCYC-1));
    wire [AW-1:0] feed_done_base = start_feed ? start_base : feed_base;
    wire drain_done_w = draining && dpe_done_w;

    assign win_busy = feeding || have_win;
    assign win_done = draining && dpe_done_w;   // no registered delay

    // feed addresses: LA_T[k*M+m] / LB[k*N+n] for lane j's element
    // e = base + word*EPS + j (clamped). Counter-only — no read-data
    // dependency, so no combinational loop through the top-level muxes.
    always @(*) begin
        la_addrs = 0;
        lb_addrs = 0;
        if (feed_active) begin
            for (jf = 0; jf < EPS; jf = jf + 1) begin
                acol = (start_feed ? 0 : feed_cnt)*EPS + jf;
                ae = feed_base_w + acol;
                if ((acol < I) && (ae < M*N)) begin
                    am = ae / N;
                    an = ae % N;
                    la_addrs[jf*AW +: AW] = feed_k_w*M + am;
                    lb_addrs[jf*AW +: AW] = feed_k_w*N + an;
                end
            end
        end
    end

    // feed data: CLB add of the two converted operands (low byte = trunc8);
    // during WEIGHT strobes the same port carries the weight byte (P23)
    always @(*) begin
        dpe_in = 0;
        if (dpe_load_input_reg) begin
            dpe_in = {{(BUF-8){1'b0}}, dpe_weight_data};
        end
        else if (feed_active) begin
            for (jf = 0; jf < EPS; jf = jf + 1) begin
                fcol = (start_feed ? 0 : feed_cnt)*EPS + jf;
                fe = feed_base_w + fcol;
                if ((fcol < I) && (fe < M*N))
                    dpe_in[jf*8 +: 8] =
                        la_rdata[jf*8 +: 8] + lb_rdata[jf*8 +: 8];
            end
        end
    end

    always @(*) dpe_w_buf_en = feed_active && dpe_ready;

    // drain word: mask keeps padding out of the reduction
    always @(*) begin
        dr_valid = 1'b0;
        dr_base = 0;
        dr_mask = 0;
        dr_data = 0;
        if (draining && dpe_reg_full) begin
            dr_valid = 1'b1;
            dr_base = drn_base + drn_cnt*EPS;
            dr_data = dpe_out;
            for (jd = 0; jd < EPS; jd = jd + 1) begin
                dcol = drn_cnt*EPS + jd;
                de = drn_base + dcol;
                if ((dcol < I) && (de < M*N)) dr_mask[jd] = 1'b1;
            end
        end
    end

    // window engine (feed overlaps the previous drain)
    always @(posedge clk) begin
        if (reset) begin
            feed_base <= 0;
            lat_base <= 0;
            feed_k <= 0;
            lat_k <= 0;
            feed_cnt <= 0;
            feeding <= 1'b0;
            have_win <= 1'b0;
            drn_base <= 0;
            drn_cnt <= 0;
            draining <= 1'b0;
            que_base <= 0;
            que_valid <= 1'b0;
        end
        else begin
            // latch only a request that cannot start immediately
            if (win_start && !win_busy && !dpe_ready) begin
                lat_base <= win_base;
                lat_k <= k_idx;
                have_win <= 1'b1;
            end

            // feed advance
            if (start_feed) begin
                have_win <= 1'b0;
                if (LCYC == 1) feeding <= 1'b0;
                else begin
                    feeding  <= 1'b1;
                    feed_base <= start_base;
                    feed_k   <= start_k;
                    feed_cnt <= 1;
                end
            end
            else if (feeding && dpe_ready && (feed_cnt != LCYC-1))
                feed_cnt <= feed_cnt + 1;
            else if (feeding && dpe_ready)
                feeding <= 1'b0;

            // window handoff into the drain pipeline (depth 2)
            if (drain_done_w) begin
                if (que_valid) begin
                    drn_base  <= que_base;
                    drn_cnt   <= 0;
                    que_base  <= feed_done_base;
                    que_valid <= feed_done_w;
                end
                else if (feed_done_w) begin
                    drn_base <= feed_done_base;
                    drn_cnt  <= 0;
                end
                else begin
                    draining <= 1'b0;
                end
            end
            else if (feed_done_w) begin
                if (draining) begin
                    que_base  <= feed_done_base;
                    que_valid <= 1'b1;
                end
                else begin
                    draining <= 1'b1;
                    drn_base <= feed_done_base;
                    drn_cnt  <= 0;
                end
            end

            if (draining && dpe_reg_full && (drn_cnt != OCYC-1))
                drn_cnt <= drn_cnt + 1;
        end
    end

endmodule


// ============================================================================
// dimm_reduce — exact int32 reduction + C serializer
//
//   * accumulation: NPORT drain ports (one per farm crossbar). Windows from
//     DIFFERENT k-blocks cover the same (m,n) elements, so two ports can
//     target the same element in the same cycle — a single shared array
//     would drop one contribution under non-blocking assignment. Each port
//     therefore owns a private bank: acc_bank[p][e], collision-free.
//   * serializer: `flush` (level from dimm_sched when every farm window is
//     done) streams M*N int32 words, one per cycle, row-major, emitting the
//     exact sum over the port banks; `done` pulses the cycle after the last
//     word; `serialized` prevents a re-run. No extra cycles.
//   * TB probe: `acc_bank[0:NPORT-1][0:M*N-1]` (the TB sums the banks; the
//     sum is the frozen `acc_q` value).
// ============================================================================
module dimm_reduce #(
    parameter M = 8,
    parameter N = 10,
    parameter BUF = 40,
    parameter AW = 8,
    parameter NPORT = 1
) (
    input  wire clk,
    input  wire reset,
    input  wire [NPORT-1:0] dr_valid,
    input  wire [NPORT*AW-1:0] dr_base,
    input  wire [NPORT*(BUF/8)-1:0] dr_mask,
    input  wire [NPORT*BUF-1:0] dr_data,
    input  wire flush,
    output wire [31:0] c_data,
    output wire c_valid,
    output reg  done
);

    localparam EPS = BUF / 8;

    // per-port accumulator banks (TB probe; the TB sums them into acc_q)
    reg signed [31:0] acc_bank [0:NPORT-1][0:M*N-1];

    integer p, j, e, i, pp;
    integer s_cnt;
    reg serializing, serialized;

    // serializer output stage: combinational, so word 0 is emitted in the
    // same cycle the last farm window drains (`flush`); `done` is one cycle
    // after word M*N-1, i.e. the span from flush is exactly M*N cycles
    wire ser_start = flush && !serialized && !serializing;
    wire ser_run = ser_start || serializing;

    // serializer output index + exact sum over the port banks (combinational
    // always block, so the bank reads are in the sensitivity)
    wire [31:0] ser_idx = ser_start ? 32'd0 : s_cnt;
    reg signed [31:0] acc_sum;
    integer qq;
    always @(*) begin
        acc_sum = 32'sd0;
        for (qq = 0; qq < NPORT; qq = qq + 1)
            acc_sum = acc_sum + acc_bank[qq][ser_idx];
    end

    assign c_valid = ser_run;
    assign c_data = acc_sum;

    always @(posedge clk) begin
        if (reset) begin
            for (p = 0; p < NPORT; p = p + 1)
                for (i = 0; i < M*N; i = i + 1) acc_bank[p][i] <= 32'sd0;
            done <= 1'b0;
            s_cnt <= 0;
            serializing <= 1'b0;
            serialized <= 1'b0;
        end
        else begin
            done <= 1'b0;

            // exact unsigned accumulation (F3: bytes are unsigned); each port
            // writes only its own bank, so same-cycle collisions are impossible
            for (p = 0; p < NPORT; p = p + 1)
                if (dr_valid[p])
                    for (j = 0; j < EPS; j = j + 1)
                        if (dr_mask[p*EPS + j]) begin
                            e = dr_base[p*AW +: AW] + j;
                            acc_bank[p][e] <= acc_bank[p][e]
                                + {24'b0, dr_data[p*BUF + j*8 +: 8]};
                        end

            // C serializer: M*N words, one per cycle, row-major; each word is
            // the exact sum over the port banks (no extra cycles)
            if (ser_start) begin
                if (M*N == 1) begin
                    serialized <= 1'b1;
                    done <= 1'b1;
                end
                else begin
                    serializing <= 1'b1;
                    s_cnt <= 1;
                end
            end
            else if (serializing) begin
                if (s_cnt == M*N-1) begin
                    serializing <= 1'b0;
                    serialized <= 1'b1;
                    done <= 1'b1;
                end
                else s_cnt <= s_cnt + 1;
            end
        end
    end

endmodule


// ============================================================================
// dimm_sched — window scheduler (pool round-robin + farm block-k gating)
//
//   * Producer pools: pool a (a = 0..N_A-1) issues windows
//     q = a, a+N_A, a+2*N_A, ... (P_A = ceil(M*K/I) total); win_base = q*I.
//     Same for B with P_B = ceil(K*N/I). One request outstanding per pool:
//     `*_pend` is held as a level until the pool acknowledges with win_busy,
//     then the window counter increments (no double-issue, no dead cycle).
//     A zero-latency fast issue is asserted combinationally in the `start`
//     cycle; it is counted in the same sequential block (issued counter) so
//     the window is not re-issued by the pending path.
//   * Farm crossbar f issues windows g = f, f+N_E, f+2*N_E, ... of the
//     flattened (k, window) stream: k = g / PW, w = g % PW,
//     PW = ceil(M*N/I), total PT = K*PW. Block-k gating:
//       a_win_total >= ceil((k+1)*M / I)  and  b_win_total >= ceil((k+1)*N / I)
//     (completed-window counters + this cycle's completions; exact prefix
//     completion holds because the pools share geometry and start together).
//     A producer-completion fast issue asserts a farm request in the same
//     cycle a producer window drains (fill == T_start); it is counted like
//     the pending path so each g is issued exactly once.
//   * `farm_all_done` (level) when every farm window is drained; `busy` high
//     from `start` until then (the reduce serializer then runs; top's `done`
//     comes from dimm_reduce). a/b/f_win_total are the frozen TB probes.
//   * No cycle counters/constants: fire from win_done/win_busy only.
// ============================================================================
module dimm_sched #(
    parameter M = 8,
    parameter N = 10,
    parameter K = 6,
    parameter R = 256,
    parameter C = 256,
    parameter BUF = 40,
    parameter N_A = 1,
    parameter N_B = 1,
    parameter N_E = 2,
    parameter AW = 8
) (
    input  wire clk,
    input  wire reset,
    input  wire start,
    // producer pools
    output reg  [N_A-1:0] a_win_start,
    output reg  [N_A*AW-1:0] a_win_base,
    input  wire [N_A-1:0] a_win_done,
    input  wire [N_A-1:0] a_win_busy,
    output reg  [N_B-1:0] b_win_start,
    output reg  [N_B*AW-1:0] b_win_base,
    input  wire [N_B-1:0] b_win_done,
    input  wire [N_B-1:0] b_win_busy,
    // farm crossbars
    output reg  [N_E-1:0] f_win_start,
    output reg  [N_E*AW-1:0] f_win_base,
    output reg  [N_E*AW-1:0] f_k_idx,
    input  wire [N_E-1:0] f_win_done,
    input  wire [N_E-1:0] f_win_busy,
    output wire farm_all_done,     // combinational: flush in the last drain cycle
    output wire busy
);

    localparam I  = (R < C) ? R : C;
    localparam PA = (M * K + I - 1) / I;   // logA windows
    localparam PB = (K * N + I - 1) / I;   // logB windows
    localparam PW = (M * N + I - 1) / I;   // farm windows per block
    localparam PT = K * PW;                // farm windows total

    integer a_win_total, b_win_total, f_win_total;  // TB probes (do not rename)

    // per-pool / per-farm issue state
    integer a_iss [0:N_A-1];
    integer b_iss [0:N_B-1];
    integer f_iss [0:N_E-1];
    reg     a_pend [0:N_A-1];
    reg     b_pend [0:N_B-1];
    reg     f_pend [0:N_E-1];

    integer a, b, f, g, k, need_a, need_b;   // sequential block only
    integer ca, cb, cf, cg;                   // combinational block only
    integer ca_inc, cb_inc, cf_inc;           // combinational popcounts
    integer ck, cneed_a, cneed_b;
    reg     cgate_ok;
    integer a_inc, b_inc, f_inc;
    reg     started;

    // window requests (combinational; held until acknowledged) with two
    // zero-latency issue paths:
    //   * `start`-cycle issue for the producers (first window of each pool
    //     is presented in the same cycle `start` is high);
    //   * producer-completion issue for the farm (a farm window may be
    //     presented in the same cycle a producer window's drain completes,
    //     so the fill is exactly T_start and not T_start+1).
    // Acks stay registered (`*_win_busy`), so there is no combinational loop.
    always @(*) begin
        ca_inc = 0;
        for (ca = 0; ca < N_A; ca = ca + 1) ca_inc = ca_inc + a_win_done[ca];
        cb_inc = 0;
        for (cb = 0; cb < N_B; cb = cb + 1) cb_inc = cb_inc + b_win_done[cb];
        cf_inc = 0;
        for (cf = 0; cf < N_E; cf = cf + 1) cf_inc = cf_inc + f_win_done[cf];

        a_win_start = 0;
        a_win_base = 0;
        for (ca = 0; ca < N_A; ca = ca + 1) begin
            a_win_start[ca] = a_pend[ca] ||
                (start && !a_win_busy[ca] && ((a_iss[ca]*N_A + ca) < PA));
            a_win_base[ca*AW +: AW] = (a_iss[ca]*N_A + ca) * I;
        end
        b_win_start = 0;
        b_win_base = 0;
        for (cb = 0; cb < N_B; cb = cb + 1) begin
            b_win_start[cb] = b_pend[cb] ||
                (start && !b_win_busy[cb] && ((b_iss[cb]*N_B + cb) < PB));
            b_win_base[cb*AW +: AW] = (b_iss[cb]*N_B + cb) * I;
        end
        f_win_start = 0;
        f_win_base = 0;
        f_k_idx = 0;
        for (cf = 0; cf < N_E; cf = cf + 1) begin
            cg = f_iss[cf]*N_E + cf;
            ck = cg / PW;
            cneed_a = ((ck+1)*M + I - 1) / I;
            cneed_b = ((ck+1)*N + I - 1) / I;
            cgate_ok = ((a_win_total + ca_inc) >= cneed_a) &&
                       ((b_win_total + cb_inc) >= cneed_b);
            f_win_start[cf] = f_pend[cf] ||
                (started && (ca_inc || cb_inc) && cgate_ok
                 && !f_win_busy[cf]
                 && (f_iss[cf] < ((cf < PT)
                      ? ((PT - cf + N_E - 1) / N_E) : 0)));
            f_win_base[cf*AW +: AW] = (cg % PW) * I;
            f_k_idx[cf*AW +: AW] = ck;
        end
    end

    // flush in the same cycle as the last farm drain (no registered delay)
    reg      farm_done_comb;
    integer  cf2, cf2_inc;      // own scratch: never shared across blocks
    always @(*) begin
        cf2_inc = 0;
        for (cf2 = 0; cf2 < N_E; cf2 = cf2 + 1)
            cf2_inc = cf2_inc + f_win_done[cf2];
        farm_done_comb = (f_win_total + cf2_inc) == PT;
    end
    assign farm_all_done = started && farm_done_comb;
    assign busy = started && !farm_done_comb;

    always @(posedge clk) begin
        if (reset) begin
            started <= 1'b0;
            a_win_total <= 0;
            b_win_total <= 0;
            f_win_total <= 0;
            for (a = 0; a < N_A; a = a + 1) begin
                a_iss[a] <= 0;
                a_pend[a] <= 1'b0;
            end
            for (b = 0; b < N_B; b = b + 1) begin
                b_iss[b] <= 0;
                b_pend[b] <= 1'b0;
            end
            for (f = 0; f < N_E; f = f + 1) begin
                f_iss[f] <= 0;
                f_pend[f] <= 1'b0;
            end
        end
        else begin
            if (start) started <= 1'b1;

            // completion counters (gating + probes)
            a_inc = 0;
            for (a = 0; a < N_A; a = a + 1) a_inc = a_inc + a_win_done[a];
            b_inc = 0;
            for (b = 0; b < N_B; b = b + 1) b_inc = b_inc + b_win_done[b];
            f_inc = 0;
            for (f = 0; f < N_E; f = f + 1) f_inc = f_inc + f_win_done[f];
            a_win_total <= a_win_total + a_inc;
            b_win_total <= b_win_total + b_inc;
            f_win_total <= f_win_total + f_inc;

            // producer requests (account for the combinational start-cycle
            // fast issue; otherwise the same window would be re-issued)
            for (a = 0; a < N_A; a = a + 1) begin
                if (a_pend[a]) begin
                    if (a_win_busy[a]) begin
                        a_pend[a] <= 1'b0;
                        a_iss[a] <= a_iss[a] + 1;
                    end
                end
                else if (start && !a_win_busy[a]
                         && ((a_iss[a]*N_A + a) < PA))
                    a_iss[a] <= a_iss[a] + 1;      // fast issue (this cycle)
                else if (started && !a_win_busy[a]
                         && ((a_iss[a]*N_A + a) < PA))
                    a_pend[a] <= 1'b1;
            end
            for (b = 0; b < N_B; b = b + 1) begin
                if (b_pend[b]) begin
                    if (b_win_busy[b]) begin
                        b_pend[b] <= 1'b0;
                        b_iss[b] <= b_iss[b] + 1;
                    end
                end
                else if (start && !b_win_busy[b]
                         && ((b_iss[b]*N_B + b) < PB))
                    b_iss[b] <= b_iss[b] + 1;      // fast issue (this cycle)
                else if (started && !b_win_busy[b]
                         && ((b_iss[b]*N_B + b) < PB))
                    b_pend[b] <= 1'b1;
            end

            // farm requests (block-k gated; the producer-completion fast path
            // is counted here so the same window is never re-issued)
            for (f = 0; f < N_E; f = f + 1) begin
                if (f_pend[f]) begin
                    if (f_win_busy[f]) begin
                        f_pend[f] <= 1'b0;
                        f_iss[f] <= f_iss[f] + 1;
                    end
                end
                else begin
                    g = f_iss[f]*N_E + f;
                    k = g / PW;
                    need_a = ((k+1)*M + I - 1) / I;
                    need_b = ((k+1)*N + I - 1) / I;
                    if (started && !f_win_busy[f]
                        && (f_iss[f] < ((f < PT)
                             ? ((PT - f + N_E - 1) / N_E) : 0))
                        && (a_win_total + a_inc >= need_a)
                        && (b_win_total + b_inc >= need_b)) begin
                        if (a_inc || b_inc)
                            f_iss[f] <= f_iss[f] + 1;  // fast issue (this cycle)
                        else
                            f_pend[f] <= 1'b1;
                    end
                end
            end
        end
    end

endmodule
