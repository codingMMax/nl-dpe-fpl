// ============================================================================
// softmax_online_top.v — NL-DPE online softmax, log-domain carry; VTR twin (v3, F3 / phase B).
// Synthesis input only — never simulated. Contract: v3/ip/softmax/spec.md; value / cycle contract = the
// simulation twin v3/ip/softmax/rtl/sim/softmax_online_top.v (GATE 2 against v3/ip/softmax/model.py).
//
// Maintained by hand next to its twin; the two share modules, parameters and datapath, and differ only in:
//   * `dpe` is the arch-XML hard block (<model name="dpe">, the `wc` subckt): no behavior model is compiled and
//     every instance is bare (the model declares no parameters); load_input_reg is tied 0. Identity weights are
//     preloaded configuration (there is no weight programmer in either twin).
//   * no verification-only probe state.
// Consistency gate: v3/tools/check_twins.py. No runtime products: v3/tools/lint_rtl.py.
//
// v3 adds to the verified v2 core: the input stage (FIFO per E4 channel, the tile reader, the packer, s_cr), the
// tile gate (run armed, store slot, log P credit, alpha credit) with registered credit counters, the output framing
// (lp = the EXP feeds' pops, la = log a of the blocks b >= 1, ll = the final log l row by row, done after it), and
// the P29 extents (EXP passes I x I, carry passes S x S).
//
// Domain (elaboration guards): S, N, BKV, I = min(R, C), N_EXP, S_CH powers of two; 8 <= S <= I;
// 8 <= BKV <= min(N, I); EPS = BUF/8 = 5; S*BKV >= I; S_CH <= NWB; B*NWB a multiple of N_EXP; capacities >= 1.
// ============================================================================

`timescale 1ns / 1ps

// ----------------------------------------------------------------------------
// softmax_online_dpe — one NL-DPE crossbar with identity weights and a fixed ACAM mode.
// ----------------------------------------------------------------------------
module softmax_online_dpe #(
    parameter R = 256,
    parameter C = 256,
    parameter BUF = 40,
    parameter P = 8,
    parameter [1:0] MODE = 2'b10,
    parameter RU = R,                  // P29 pass extent: rows loaded, columns drained
    parameter CU = C
) (
    input  wire           clk,
    input  wire           reset,
    input  wire [BUF-1:0] data_in,
    input  wire           w_buf_en,
    output wire           ready,
    output wire [BUF-1:0] data_out,
    output wire           done,
    output wire           full
);
    dpe u_dpe (
        .clk(clk), .reset(reset),
        .data_in(data_in), .nl_dpe_control(MODE),
        .shift_add_control(1'b0), .w_buf_en(w_buf_en),
        .shift_add_bypass(1'b0), .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(ready), .data_out(data_out), .dpe_done(done),
        .reg_full(full)
    );
endmodule


// ----------------------------------------------------------------------------
// softmax_online_ohmux — one-hot select of N W-bit words (flat): per-bit reduction OR of
// the masked inputs (generate-only, no combinational loops).
// ----------------------------------------------------------------------------
module softmax_online_ohmux #(
    parameter N = 4,
    parameter W = 8
) (
    input  wire [N-1:0]   sel,
    input  wire [N*W-1:0] in,
    output wire [W-1:0]   out
);
    genvar k, n;
    generate
        for (k = 0; k < W; k = k + 1) begin : g_bit
            wire [N-1:0] col;
            for (n = 0; n < N; n = n + 1) begin : g_n
                assign col[n] = sel[n] & in[n*W + k];
            end
            assign out[k] = |col;
        end
    endgenerate
endmodule


// ----------------------------------------------------------------------------
// softmax_online_store — ring of score blocks. Block element e (row-major within the block)
// is in window w = e >> log2 NWEL (global window g = b*NWB + w) at position
// pos = e mod NWEL; bank g mod NB, lane pos mod EPS, address
// ((g / NB) mod DW) * WPW + pos / EPS. A stream word hits each lane RAM of a
// bank at most once and straddles at most two windows (different banks, NB >= 2).
// Each block starts on a fresh stream word (per-block padding).
// ----------------------------------------------------------------------------
module softmax_online_store #(
    parameter NWEL   = 256,
    parameter BLKEL  = 1024,
    parameter WPW    = 52,
    parameter NB     = 2,
    parameter BDEPTH = 104,
    parameter BAW    = 7,
    parameter EPS    = 5
) (
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 ld_valid,
    input  wire [8*EPS-1:0]     ld_data,
    input  wire [NB*BAW-1:0]    rd_addr,
    output wire [NB*8*EPS-1:0]  rd_data
);
    localparam LNB = $clog2(NB);
    localparam PW  = $clog2(NWEL + EPS + 1);
    localparam QW  = $clog2(WPW + 1);
    localparam EW  = $clog2(BLKEL + EPS + 1);

    // next-word tracking (lane 0 of the incoming word)
    reg [EW-1:0]  t_e;          // element index within the block
    reg [PW-1:0]  t_pos;
    reg [2:0]     t_r;
    reg [QW-1:0]  t_q;
    reg [LNB-1:0] t_b;
    reg [BAW-1:0] t_wb;
    wire [LNB-1:0] t_b1  = t_b + 1'b1;
    wire [BAW-1:0] t_wbn = (t_wb + WPW == BDEPTH) ? {BAW{1'b0}} : t_wb + WPW;
    wire [BAW-1:0] t_wb1 = (t_b1 == 0) ? t_wbn : t_wb;
    wire           last  = (t_e + EPS >= BLKEL);
    wire           cross = (t_pos + EPS >= NWEL);

    // stage L0
    reg              l0_v;
    reg [8*EPS-1:0]  l0_d;
    reg [EW-1:0]     l0_e;
    reg [PW-1:0]     l0_pos;
    reg [2:0]        l0_r;
    reg [LNB-1:0]    l0_b;
    reg [BAW-1:0]    l0_a0, l0_a1, l0_wb1;

    always @(posedge clk) begin
        if (reset) begin
            t_e <= 0; t_pos <= 0; t_r <= 0; t_q <= 0; t_b <= 0; t_wb <= 0;
            l0_v <= 1'b0;
        end
        else begin
            l0_v <= ld_valid;
            if (ld_valid) begin
                l0_d   <= ld_data;
                l0_e   <= t_e;
                l0_pos <= t_pos;
                l0_r   <= t_r;
                l0_b   <= t_b;
                l0_a0  <= t_wb + t_q;
                l0_a1  <= t_wb + t_q + 1'b1;
                l0_wb1 <= t_wb1;
                if (last) begin                    // next block: fresh word, next window
                    t_e <= 0; t_pos <= 0; t_r <= 0; t_q <= 0;
                    t_b <= t_b1; t_wb <= t_wb1;
                end
                else if (cross) begin
                    t_e   <= t_e + EPS;
                    t_pos <= t_pos + EPS - NWEL;
                    t_r   <= t_pos + EPS - NWEL;
                    t_q   <= 0;
                    t_b   <= t_b1; t_wb <= t_wb1;
                end
                else begin
                    t_e   <= t_e + EPS;
                    t_pos <= t_pos + EPS;
                    t_q   <= t_q + 1'b1;
                end
            end
        end
    end

    genvar gb, gl;
    generate
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_b
            for (gl = 0; gl < EPS; gl = gl + 1) begin : g_l
                wire [3:0]  t_in  = (gl >= l0_r) ? (gl - l0_r) : (gl + EPS - l0_r);
                wire [PW:0] t_nx  = gl + NWEL - l0_pos;
                wire in_ok = l0_v && (l0_b == gb) && (l0_pos + t_in < NWEL)
                             && (l0_e + t_in < BLKEL);
                wire nx_ok = l0_v && (((l0_b + 1'b1) & (NB - 1)) == gb) && (t_nx < EPS)
                             && (l0_e + t_nx < BLKEL);
                wire [2:0]     t_sel = in_ok ? t_in[2:0] : t_nx[2:0];
                wire           we    = in_ok | nx_ok;
                wire [BAW-1:0] wa    = in_ok ? ((gl >= l0_r) ? l0_a0 : l0_a1) : l0_wb1;
                wire [7:0]     wd    = l0_d[t_sel*8 +: 8];

                // write controls registered before the RAM port (ladder 2026-10-06)
                reg            w2_we;
                reg [BAW-1:0]  w2_a;
                reg [7:0]      w2_d;
                always @(posedge clk) begin
                    if (reset) w2_we <= 1'b0;
                    else       w2_we <= we;
                    w2_a <= wa;
                    w2_d <= wd;
                end

                reg [7:0] mem [0:BDEPTH-1];
                reg [7:0] rq;
                always @(posedge clk) begin
                    if (w2_we) mem[w2_a] <= w2_d;
                    rq <= mem[rd_addr[gb*BAW +: BAW]];
                end
                assign rd_data[(gb*EPS + gl)*8 +: 8] = rq;
            end
        end
    endgenerate
endmodule


// ----------------------------------------------------------------------------
// softmax_online_loadmax — per-block row maxima while the block streams in. Rows are BKV
// consecutive elements (BKV > EPS: a word ends at most one row). M1 masked
// lane maxima, M2 running max, M3 per completed row r of block b:
//   m_new = m_blk (b = 0) | max(m[r], m_blk);  log_a = m[r] - m_new (0 at b = 0)
// writes mn_<parity>[r], la_<parity>[r] (rows complete in order, one-hot
// pointer) and emits a la event. max_cnt = blocks whose rows are all done.
// The running max m is a rotating ring (rows complete in order every block):
// row r is at the head when it completes, so no S:1 read mux (ladder lc1);
// after the last block the ring is aligned again (m_q[r*8 +: 8] = m[r]).
// ----------------------------------------------------------------------------
module softmax_online_loadmax #(
    parameter S   = 64,
    parameter BKV = 16,
    parameter EPS = 5,
    parameter MCW = 3,
    parameter LS  = $clog2(S)
) (
    input  wire              clk,
    input  wire              reset,
    input  wire              ld_valid,
    input  wire [8*EPS-1:0]  ld_data,
    output reg  [8*S-1:0]    m_q,
    output reg  [8*S-1:0]    mn0,
    output reg  [8*S-1:0]    mn1,
    output reg  [9*S-1:0]    la0,
    output reg  [9*S-1:0]    la1,
    output reg  [MCW-1:0]    max_cnt,
    output reg               la_valid,
    output reg  [MCW-1:0]    la_blk,
    output reg  [LS-1:0]     la_row,
    output reg  [8:0]        la_val
);
    localparam BLKEL = S * BKV;
    localparam LB    = $clog2(BKV);
    localparam EW    = $clog2(BLKEL + EPS + 1);

    // M0
    reg              m0_v;
    reg [8*EPS-1:0]  m0_d;
    reg [EW-1:0]     m0_e;
    reg [EW-1:0]     t_e;
    always @(posedge clk) begin
        if (reset) begin m0_v <= 1'b0; t_e <= 0; end
        else begin
            m0_v <= ld_valid;
            if (ld_valid) begin
                m0_d <= ld_data; m0_e <= t_e;
                t_e  <= (t_e + EPS >= BLKEL) ? {EW{1'b0}} : t_e + EPS;
            end
        end
    end

    wire [LB-1:0]  p0 = m0_e[LB-1:0];
    wire [EPS-1:0] va, vb, lr;
    genvar j;
    generate
        for (j = 0; j < EPS; j = j + 1) begin : g_c
            wire [LB:0] pj = p0 + j;
            wire        v  = m0_v && (m0_e + j < BLKEL);
            assign va[j] = v && (pj < BKV);
            assign vb[j] = v && (pj >= BKV);
            assign lr[j] = v && (pj == BKV - 1);
        end
    endgenerate

    function signed [7:0] smax;
        input signed [7:0] a, b;
        smax = (a > b) ? a : b;
    endfunction
    function signed [7:0] lane;
        input [8*EPS-1:0] d;
        input integer k;
        input m;
        lane = m ? d[k*8 +: 8] : 8'sh80;
    endfunction

    // M1a / M1b
    reg              m1_v, m1_done, m1_hasb;
    reg signed [7:0] a01, a23, a4, b01, b23, b4;
    always @(posedge clk) begin
        if (reset) m1_v <= 1'b0;
        else begin
            m1_v    <= m0_v;
            m1_done <= |lr;
            m1_hasb <= |vb;
            a01 <= smax(lane(m0_d, 0, va[0]), lane(m0_d, 1, va[1]));
            a23 <= smax(lane(m0_d, 2, va[2]), lane(m0_d, 3, va[3]));
            a4  <= lane(m0_d, 4, va[4]);
            b01 <= smax(lane(m0_d, 0, vb[0]), lane(m0_d, 1, vb[1]));
            b23 <= smax(lane(m0_d, 2, vb[2]), lane(m0_d, 3, vb[3]));
            b4  <= lane(m0_d, 4, vb[4]);
        end
    end
    reg              m2_v, m2_done, m2_hasb;
    reg signed [7:0] ma, mb;
    always @(posedge clk) begin
        if (reset) m2_v <= 1'b0;
        else begin
            m2_v    <= m1_v;
            m2_done <= m1_done;
            m2_hasb <= m1_hasb;
            ma <= smax(smax(a01, a23), a4);
            mb <= smax(smax(b01, b23), b4);
        end
    end

    // M2: running block-row max
    reg signed [7:0] run;
    reg [S-1:0]      rp_oh;
    reg [LS-1:0]     r2;
    wire signed [7:0] fold = smax(run, ma);
    reg              w3_v;
    reg [S-1:0]      w3_oh;
    reg [LS-1:0]     w3_row;
    reg signed [7:0] w3_val;
    always @(posedge clk) begin
        if (reset) begin
            run <= 8'sh80; rp_oh <= 1; r2 <= 0; w3_v <= 1'b0;
        end
        else begin
            w3_v <= 1'b0;
            if (m2_v) begin
                if (m2_done) begin
                    w3_v   <= 1'b1;
                    w3_oh  <= rp_oh;
                    w3_row <= r2;
                    w3_val <= fold;
                    run    <= m2_hasb ? mb : 8'sh80;
                    rp_oh  <= rp_oh[S-1] ? {{(S-1){1'b0}}, 1'b1} : rp_oh << 1;
                    r2     <= r2 + 1'b1;
                end
                else run <= fold;
            end
        end
    end

    // M3: running max, m_new, log_a
    reg [MCW-1:0] cb3;
    wire          first = (cb3 == 0);
    wire          par   = cb3[0];
    wire [7:0]    m_old = m_q[7:0];               // ring head = the completing row
    genvar gr;
    wire signed [7:0] mnv = first ? w3_val : smax($signed(m_old), w3_val);
    wire signed [8:0] lav = first ? 9'sd0 : ($signed({m_old[7], m_old}) - $signed({mnv[7], mnv}));

    generate
        for (gr = 0; gr < S; gr = gr + 1) begin : g_w
            always @(posedge clk)
                if (w3_v && w3_oh[gr]) begin
                    if (par) begin mn1[gr*8 +: 8] <= mnv; la1[gr*9 +: 9] <= lav; end
                    else     begin mn0[gr*8 +: 8] <= mnv; la0[gr*9 +: 9] <= lav; end
                end
        end
    endgenerate

    always @(posedge clk) begin
        if (reset) begin
            cb3 <= 0; max_cnt <= 0; la_valid <= 1'b0;
        end
        else begin
            la_valid <= w3_v;
            la_blk   <= cb3;
            la_row   <= w3_row;
            la_val   <= lav;
            if (w3_v) m_q <= {mnv, m_q[8*S-1:8]};     // rotate: next row to the head
            if (w3_v && w3_oh[S-1]) begin
                cb3 <= cb3 + 1'b1;
                max_cnt <= max_cnt + 1'b1;
            end
        end
    end

endmodule


// ----------------------------------------------------------------------------
// softmax_online_xfeed — EXP crossbar CX of N_EXP. Global windows g = CX, CX+N_EXP, ...
// (PW = B*NWB windows of NWEL elements, block-major). Window g belongs to block
// g >> log2 NWB and issues once that block's row maxima are done (max_cnt).
// Producer per word k: bank read (window base + k), subtract m_new of the
// lane's row (rows of lane 0 and the next; BKV > EPS), clamp -128, mask lanes
// >= NWEL; FIFO entry two cycles later. The dpe pops whenever ready; popped
// words are the log|P| output. Drain windows in order.
// ----------------------------------------------------------------------------
module softmax_online_xfeed #(
    parameter CX    = 0,
    parameter N_EXP = 1,
    parameter S     = 64,
    parameter BKV   = 16,
    parameter R     = 256,
    parameter C     = 256,
    parameter BUF   = 40,
    parameter P     = 8,
    parameter NWEL  = 256,
    parameter NWB   = 4,
    parameter B     = 4,
    parameter WPW   = 52,
    parameter NB    = 2,
    parameter DW    = 4,
    parameter BAW   = 8,
    parameter AW    = 13,
    parameter MCW   = 3
) (
    input  wire                      clk,
    input  wire                      reset,          // run clear (re-arm) + power-on reset
    input  wire                      dpe_reset,      // power-on reset of the crossbar only
    input  wire                      go,
    input  wire [MCW-1:0]            max_cnt,
    input  wire [8*S-1:0]            mn0,
    input  wire [8*S-1:0]            mn1,
    output wire [BAW-1:0]            rd_addr,
    input  wire [(NB/N_EXP)*BUF-1:0] rd_data,
    output wire                      lp_valid,
    output wire [AW-1:0]             lp_base,
    output wire [BUF/8-1:0]          lp_mask,
    output wire [BUF-1:0]            lp_data,
    output wire                      dr_valid,
    output wire [AW-1:0]             dr_base,
    output wire [BUF/8-1:0]          dr_mask,
    output wire [BUF-1:0]            dr_data
);
    localparam EPS   = BUF / 8;
    localparam XI    = (R < C) ? R : C;                // P29: the pass loads and drains I = min(R, C)
    localparam LCYC  = (XI * 8 + BUF - 1) / BUF;
    localparam OCYC  = (XI * 8 + BUF - 1) / BUF;
    localparam NBX   = NB / N_EXP;
    localparam PW    = B * NWB;
    localparam NWC   = (PW > CX) ? (PW - CX + N_EXP - 1) / N_EXP : 0;
    localparam LS    = $clog2(S);
    localparam LB    = $clog2(BKV);
    localparam LNWEL = $clog2(NWEL);
    localparam LNWB  = $clog2(NWB);
    localparam LNB   = $clog2(NB);
    localparam LNX   = $clog2(N_EXP);
    localparam RSH   = LNWEL - LB;                 // rows per window = 2^RSH
    localparam GW    = $clog2(PW + N_EXP + 1);
    localparam KW    = $clog2(LCYC + 1);
    localparam DCW   = $clog2(OCYC + 1);
    // column of lane 0 runs over every loaded / drained word (LCYC, OCYC words),
    // not just the window: sized for both (S=8 regression, 2026-10-06)
    localparam CPW  = $clog2(((LCYC > OCYC) ? LCYC : OCYC) * EPS + EPS + 1);
    localparam LBW   = (NBX > 1) ? $clog2(NBX) : 1;
    localparam FD    = 4;
    localparam EW    = BUF + AW + EPS + 1;          // FIFO entry: last, mask, base, data

    function [31:0] mulk;
        input [31:0] x;
        input integer k;
        integer b;
        begin
            mulk = 32'd0;
            for (b = 0; b < 31; b = b + 1)
                if ((k >> b) & 1) mulk = mulk + (x << b);
        end
    endfunction

    // window-derived values (shifts / masks of the window index; no multiply)
    function [BAW-1:0] w_addr;      // bank address of word 0
        input [GW-1:0] g;
        w_addr = mulk((g >> LNB) & (DW - 1), WPW);
    endfunction
    function [AW-1:0] w_elem;       // global element index of word 0
        input [GW-1:0] g;
        w_elem = g << LNWEL;
    endfunction
    function [LS-1:0] w_row;        // block row of word 0
        input [GW-1:0] g;
        w_row = (g & (NWB - 1)) << RSH;
    endfunction
    function [LBW-1:0] w_lb;        // local bank among this crossbar's banks
        input [GW-1:0] g;
        w_lb = (g & (NB - 1)) >> LNX;
    endfunction

    // ---------------- dpe ---------------------------------------------------
    wire           dpe_ready, dpe_done_w, dpe_full;
    wire [BUF-1:0] dpe_out, dpe_in;
    wire           dpe_wen;
    softmax_online_dpe #(.R(R), .C(C), .BUF(BUF), .P(P), .MODE(2'b10), .RU(XI), .CU(XI)) u_x (
        .clk(clk), .reset(dpe_reset), .data_in(dpe_in), .w_buf_en(dpe_wen),
        .ready(dpe_ready), .data_out(dpe_out), .done(dpe_done_w), .full(dpe_full));

    // ---------------- producer ---------------------------------------------
    reg            p_act;
    reg [GW-1:0]   p_g;
    reg [KW-1:0]   p_k;
    reg [CPW-1:0]  p_col;
    reg [BAW-1:0]  p_addr;
    reg [AW-1:0]   p_e;
    reg [LS-1:0]   p_ra;
    reg [LB:0]     p_rp;
    wire [GW-1:0]  p_blk = p_g >> LNWB;
    wire           p_par = p_blk[0];
    wire [GW-1:0]  n_g   = p_g + N_EXP;

    reg [1:0]       f_wp, f_rp;
    reg [2:0]       f_cnt;
    reg             i1_v, i2_v;
    wire            k_last = (p_k == LCYC - 1);
    // block gate registered (ladder lc4): `max_cnt > p_blk` evaluated one cycle
    // ahead and cleared on a window change, so it is never stale; keeps the compare
    // off the issue enable that fans out to every producer register
    reg             blk_ok;
    // FIFO credit registered too (ladder lc5): credit_ok == (f_cnt + i1_v + i2_v < FD)
    // of the current cycle, computed one edge early from the next-state values, so
    // `issue` is an AND of registers
    reg             credit_ok;
    wire            issue  = go && p_act && blk_ok && credit_ok;
    always @(posedge clk) begin
        if (reset) begin blk_ok <= 1'b0; credit_ok <= 1'b1; end
        else begin
            blk_ok    <= (issue && k_last) ? 1'b0 : (max_cnt > p_blk);
            credit_ok <= ({1'b0, f_cnt} + i2_v - pop + issue + i1_v) < FD;
        end
    end
    wire            pop    = (f_cnt != 0) && dpe_ready;
    // the RAM reads the registered address, aligned with the 2-stage m_new select
    reg  [BAW-1:0]  ra_q;
    always @(posedge clk) ra_q <= p_addr;
    assign rd_addr = ra_q;

    always @(posedge clk) begin
        if (reset) begin
            p_act <= (NWC > 0); p_g <= CX; p_k <= 0; p_col <= 0;
            p_addr <= w_addr(CX); p_e <= w_elem(CX); p_ra <= w_row(CX); p_rp <= 0;
        end
        else if (issue) begin
            if (k_last) begin
                p_g    <= n_g;
                p_k    <= 0;
                p_col  <= 0;
                p_addr <= w_addr(n_g);
                p_e    <= w_elem(n_g);
                p_ra   <= w_row(n_g);
                p_rp   <= 0;
                if (n_g >= PW) p_act <= 1'b0;
            end
            else begin
                p_k    <= p_k + 1'b1;
                p_col  <= p_col + EPS;
                p_addr <= p_addr + 1'b1;
                p_e    <= p_e + EPS;
                if (p_rp + EPS >= BKV) begin
                    p_rp <= p_rp + EPS - BKV;
                    p_ra <= p_ra + 1'b1;
                end
                else p_rp <= p_rp + EPS;
            end
        end
    end

    // ---------------- I1 / I2: meta + 2-stage m_new select (ladder lc2) ----
    // I1 selects the 2^RL-row group of the lane-0 row and of the next row;
    // I2 the row inside the group; the RAM word arrives with I2.
    localparam RL = LS - LS / 2;                    // low row bits
    localparam GWD = 8 << RL;                       // group width in bits
    reg [EPS-1:0]  i1_mask, i1_split, i2_mask, i2_split;
    reg [GWD-1:0]  i1_ga, i1_gb;
    reg [RL-1:0]   i1_la, i1_lb1;
    reg [7:0]      i2_rma, i2_rmb;
    reg [LBW-1:0]  i1_lb, i2_lb;
    reg            i1_last, i2_last;
    reg [AW-1:0]   i1_e, i2_e;
    wire [8*S-1:0] mn  = p_par ? mn1 : mn0;
    wire [LS-1:0]  ra1 = p_ra + 1'b1;
    wire [31:0]    ra_w  = p_ra;                   // zero-extended: the group base below is wider than a row index
    wire [31:0]    ra1_w = ra1;
    integer jm;
    always @(posedge clk) begin
        if (reset) begin i1_v <= 1'b0; i2_v <= 1'b0; end
        else begin i1_v <= issue; i2_v <= i1_v; end
        for (jm = 0; jm < EPS; jm = jm + 1) begin
            i1_mask[jm]  <= (p_col + jm < NWEL);
            i1_split[jm] <= (p_rp + jm >= BKV);
        end
        i1_ga   <= mn[((ra_w >> RL) << (RL + 3)) +: GWD];     // group base = group * GWD, GWD = 2^(RL+3)
        i1_gb   <= mn[((ra1_w >> RL) << (RL + 3)) +: GWD];
        i1_la   <= p_ra[RL-1:0];
        i1_lb1  <= ra1[RL-1:0];
        i1_lb   <= w_lb(p_g);
        i1_last <= k_last;
        i1_e    <= p_e;
        i2_rma   <= i1_ga[i1_la*8 +: 8];
        i2_rmb   <= i1_gb[i1_lb1*8 +: 8];
        i2_mask  <= i1_mask;
        i2_split <= i1_split;
        i2_lb    <= i1_lb;
        i2_last  <= i1_last;
        i2_e     <= i1_e;
    end

    // ---------------- I3: subtract + clamp, FIFO --------------------------
    genvar gq;
    wire [NBX-1:0] lb_sel;                     // one-hot of the local bank (NBX <= 2)
    generate
        for (gq = 0; gq < NBX; gq = gq + 1) begin : g_lb
            assign lb_sel[gq] = (i2_lb == gq);
        end
    endgenerate
    wire [BUF-1:0] rword;
    softmax_online_ohmux #(.N(NBX), .W(BUF)) u_rw (.sel(lb_sel), .in(rd_data), .out(rword));
    reg  [BUF-1:0] sub_w;
    reg signed [8:0] dd;
    integer js;
    always @(*) begin
        sub_w = 0;
        for (js = 0; js < EPS; js = js + 1) begin
            dd = $signed(rword[js*8 +: 8]) - $signed(i2_split[js] ? i2_rmb : i2_rma);
            if (i2_mask[js])
                sub_w[js*8 +: 8] = (dd < -9'sd128) ? 8'h80 : dd[7:0];
        end
    end
    always @(posedge clk) begin
        if (reset) begin f_wp <= 0; f_rp <= 0; f_cnt <= 0; end
        else begin
            if (i2_v) f_wp <= f_wp + 1'b1;
            if (pop) f_rp <= f_rp + 1'b1;
            f_cnt <= f_cnt + i2_v - pop;
        end
    end
    // FIFO entries (one register each), head by a one-hot read select
    wire [FD*EW-1:0] f_q;
    wire [FD-1:0]    f_rsel;
    generate
        for (gq = 0; gq < FD; gq = gq + 1) begin : g_fq
            reg [EW-1:0] q;
            always @(posedge clk) if (!reset && i2_v && (f_wp == gq)) q <= {i2_last, i2_mask, i2_e, sub_w};
            assign f_q[gq*EW +: EW] = q;
            assign f_rsel[gq] = (f_rp == gq);
        end
    endgenerate
    wire [EW-1:0] head;
    softmax_online_ohmux #(.N(FD), .W(EW)) u_head (.sel(f_rsel), .in(f_q), .out(head));
    assign dpe_in   = head[BUF-1:0];
    assign dpe_wen  = pop;
    assign lp_valid = pop;
    assign lp_data  = head[BUF-1:0];
    assign lp_base  = head[BUF +: AW];
    assign lp_mask  = head[BUF+AW +: EPS];
    wire   head_last = head[EW-1];

    // ---------------- drain -----------------------------------------------
    reg [1:0]     fed, drained;
    reg [DCW-1:0] d_cnt;
    reg [CPW-1:0] d_col;
    reg [AW-1:0]  d_base, d_eb;
    wire          pend = (fed != drained);
    assign dr_valid = pend && dpe_full;
    assign dr_base  = d_base;
    assign dr_data  = dpe_out;
    genvar gd;
    generate
        for (gd = 0; gd < EPS; gd = gd + 1) begin : g_dm
            assign dr_mask[gd] = dr_valid && (d_col + gd < NWEL);
        end
    endgenerate
    localparam [AW-1:0] WSTEP = N_EXP * NWEL;
    always @(posedge clk) begin
        if (reset) begin
            fed <= 0; drained <= 0; d_cnt <= 0; d_col <= 0;
            d_base <= w_elem(CX); d_eb <= w_elem(CX);
        end
        else begin
            if (pop && head_last) fed <= fed + 1'b1;
            if (pend && dpe_done_w) begin
                drained <= drained + 1'b1;
                d_cnt  <= 0;
                d_col  <= 0;
                d_eb   <= d_eb + WSTEP;
                d_base <= d_eb + WSTEP;
            end
            else if (dr_valid && (d_cnt != OCYC - 1)) begin
                d_cnt  <= d_cnt + 1'b1;
                d_col  <= d_col + EPS;
                d_base <= d_base + EPS;
            end
        end
    end
endmodule


// ----------------------------------------------------------------------------
// softmax_online_row_acc — streaming row-sum accumulator, 2-stage (Fmax ladder 2026-10-06).
// Each crossbar drains its windows in order; with RS | I a row never straddles a
// window, so one open-row accumulator per crossbar suffices. A drain word
// (EPS <= 8 lanes, RS > EPS) touches at most two rows: lanes of lane 0's row (a)
// and of the next row (b).
//   stage A (registered): lane classes, masked lane values, balanced lane-sum
//                         tree for a and b
//   stage B (registered): acc <= done_a ? sum_b : acc + sum_a; event (row,
//                         acc + sum_a) when row a ends in the word
// Events leave 2 cycles after the completing drain word, one per crossbar.
// ----------------------------------------------------------------------------
module softmax_online_row_acc #(
    parameter RS    = 128,
    parameter N_EXP = 1,
    parameter PORTS = 40,
    parameter AW    = 16
) (
    input  wire                       clk,
    input  wire                       reset,
    input  wire [N_EXP-1:0]           dr_valid,
    input  wire [N_EXP*AW-1:0]        dr_base,
    input  wire [N_EXP*(PORTS/8)-1:0] dr_mask,
    input  wire [N_EXP*PORTS-1:0]     dr_data,
    output wire [N_EXP-1:0]           ev_we,
    output wire [N_EXP*AW-1:0]        ev_row,
    output wire [N_EXP*32-1:0]        ev_sum
);
    localparam EPS = PORTS / 8;
    localparam LRS = $clog2(RS);
    localparam SW  = 8 + LRS;
    localparam LW  = 11;                 // 8 lanes * 255 < 2^11

    genvar c, j;
    generate
        for (c = 0; c < N_EXP; c = c + 1) begin : g_x
            wire [AW-1:0]   base = dr_base[c*AW +: AW];
            wire [LRS:0]    pos0 = {1'b0, base[LRS-1:0]};
            wire [8*LW-1:0] ma, mb;          // masked lanes, padded to 8
            wire [7:0]      lastl;
            for (j = 0; j < 8; j = j + 1) begin : g_l
                if (j < EPS) begin : g_in
                    wire         v    = dr_valid[c] & dr_mask[c*EPS + j];
                    wire [LRS:0] pj   = pos0 + j;
                    wire         in_b = (pj >= RS);
                    wire [7:0]   d    = dr_data[c*PORTS + j*8 +: 8];
                    assign ma[j*LW +: LW] = (v & ~in_b) ? {{(LW-8){1'b0}}, d} : {LW{1'b0}};
                    assign mb[j*LW +: LW] = (v &  in_b) ? {{(LW-8){1'b0}}, d} : {LW{1'b0}};
                    assign lastl[j] = v & (pj == RS - 1);
                end else begin : g_pad
                    assign ma[j*LW +: LW] = {LW{1'b0}};
                    assign mb[j*LW +: LW] = {LW{1'b0}};
                    assign lastl[j] = 1'b0;
                end
            end
            wire [LW-1:0] sa = ((ma[0*LW +: LW] + ma[1*LW +: LW]) + (ma[2*LW +: LW] + ma[3*LW +: LW]))
                             + ((ma[4*LW +: LW] + ma[5*LW +: LW]) + (ma[6*LW +: LW] + ma[7*LW +: LW]));
            wire [LW-1:0] sb = ((mb[0*LW +: LW] + mb[1*LW +: LW]) + (mb[2*LW +: LW] + mb[3*LW +: LW]))
                             + ((mb[4*LW +: LW] + mb[5*LW +: LW]) + (mb[6*LW +: LW] + mb[7*LW +: LW]));

            // stage A
            reg          a_v, a_done;
            reg [LW-1:0] a_sa, a_sb;
            reg [AW-1:0] a_row;
            always @(posedge clk) begin
                if (reset) a_v <= 1'b0;
                else       a_v <= dr_valid[c];
                a_done <= |lastl;
                a_sa   <= sa;
                a_sb   <= sb;
                a_row  <= base >> LRS;
            end

            // stage B
            reg  [SW-1:0] acc;
            wire [SW-1:0] tot_a = acc + a_sa;
            reg           e_we;
            reg  [AW-1:0] e_row;
            reg  [SW-1:0] e_sum;
            always @(posedge clk) begin
                if (reset) begin acc <= {SW{1'b0}}; e_we <= 1'b0; end
                else begin
                    if (a_v) acc <= a_done ? {{(SW-LW){1'b0}}, a_sb} : tot_a;
                    e_we <= a_v & a_done;
                end
                e_row <= a_row;
                e_sum <= tot_a;
            end
            assign ev_we[c]           = e_we;
            assign ev_row[c*AW +: AW] = e_row;
            assign ev_sum[c*32 +: 32] = {{(32-SW){1'b0}}, e_sum};
        end
    endgenerate
endmodule


// ----------------------------------------------------------------------------
// softmax_online_wfeed — one window of S row values through a carry crossbar (mode MODE).
// On `start`, the producer emits LCYC words; word k carries rows EPS*k + l,
// selected by the parent from the one-hot pointer `k_oh` (zero past the row
// words). The parent registers its selection, so `word` belongs to the word
// issued in the previous cycle (ladder lc2). FIFO + pop-when-ready as in softmax_online_xfeed. Drain: rows by the one-hot
// word pointer `d_oh` (row r = word r/EPS, lane r%EPS); `last_row` marks the
// drain word that carries row S-1; `d_row0` is the row of lane 0.
// ----------------------------------------------------------------------------
module softmax_online_wfeed #(
    parameter S   = 64,
    parameter R   = 256,
    parameter C   = 256,
    parameter BUF = 40,
    parameter P   = 8,
    parameter [1:0] MODE = 2'b10,
    parameter WL  = (S + BUF / 8 - 1) / (BUF / 8),
    parameter LS  = $clog2(S)
) (
    input  wire            clk,
    input  wire            reset,          // run clear (re-arm) + power-on reset
    input  wire            dpe_reset,      // power-on reset of the crossbar only
    input  wire            start,
    output reg  [WL-1:0]   k_oh,
    input  wire [BUF-1:0]  word,
    output wire            dr_v,
    output reg  [WL:0]     d_oh,
    output reg  [LS:0]     d_row0,
    output wire [BUF-1:0]  dr_data,
    output wire            last_row
);
    localparam EPS  = BUF / 8;
    localparam LCYC = (S * 8 + BUF - 1) / BUF;         // P29: the pass loads and drains the S rows in use
    localparam OCYC = (S * 8 + BUF - 1) / BUF;
    localparam KW   = $clog2(LCYC + 1);
    localparam DCW  = $clog2(OCYC + 1);
    localparam FD   = 4;

    wire           ready, ddone, full;
    wire [BUF-1:0] dout, din;
    wire           wen;
    softmax_online_dpe #(.R(R), .C(C), .BUF(BUF), .P(P), .MODE(MODE), .RU(S), .CU(S)) u_x (
        .clk(clk), .reset(dpe_reset), .data_in(din), .w_buf_en(wen),
        .ready(ready), .data_out(dout), .done(ddone), .full(full));

    reg                  p_act;
    reg [KW-1:0]         p_k;
    reg [1:0]            f_wp, f_rp;
    reg [2:0]            f_cnt;
    reg                  i1_v, i1_last;
    wire issue = p_act && ((f_cnt + i1_v) < FD);
    wire pop   = (f_cnt != 0) && ready;
    always @(posedge clk) begin
        if (reset) begin
            p_act <= 1'b0; p_k <= 0; k_oh <= 0; f_wp <= 0; f_rp <= 0; f_cnt <= 0;
            i1_v <= 1'b0; i1_last <= 1'b0;
        end
        else begin
            i1_v    <= issue;
            i1_last <= (p_k == LCYC - 1);
            if (start && !p_act) begin
                p_act <= 1'b1; p_k <= 0; k_oh <= {{(WL-1){1'b0}}, 1'b1};
            end
            else if (issue) begin
                p_k  <= p_k + 1'b1;
                k_oh <= k_oh << 1;
                if (p_k == LCYC - 1) p_act <= 1'b0;
            end
            if (i1_v) f_wp <= f_wp + 1'b1;
            if (pop) f_rp <= f_rp + 1'b1;
            f_cnt <= f_cnt + i1_v - pop;
        end
    end
    // FIFO entries (one register each), head by a one-hot read select
    genvar gq;
    wire [FD*(BUF+1)-1:0] f_q;
    wire [FD-1:0]         f_rsel;
    generate
        for (gq = 0; gq < FD; gq = gq + 1) begin : g_fq
            reg [BUF:0] q;
            always @(posedge clk) if (!reset && i1_v && (f_wp == gq)) q <= {i1_last, word};
            assign f_q[gq*(BUF+1) +: BUF+1] = q;
            assign f_rsel[gq] = (f_rp == gq);
        end
    endgenerate
    wire [BUF:0] head;
    softmax_online_ohmux #(.N(FD), .W(BUF+1)) u_head (.sel(f_rsel), .in(f_q), .out(head));
    assign din = head[BUF-1:0];
    assign wen = pop;

    reg [1:0]     fed, drained;
    reg [DCW-1:0] d_cnt;
    wire pend = (fed != drained);
    assign dr_v     = pend && full;
    assign dr_data  = dout;
    assign last_row = dr_v && d_oh[(S - 1) / EPS];
    always @(posedge clk) begin
        if (reset) begin
            fed <= 0; drained <= 0; d_cnt <= 0;
            d_oh <= {{WL{1'b0}}, 1'b1}; d_row0 <= 0;
        end
        else begin
            if (pop && head[BUF]) fed <= fed + 1'b1;
            if (pend && ddone) begin
                drained <= drained + 1'b1; d_cnt <= 0;
                d_oh <= {{WL{1'b0}}, 1'b1}; d_row0 <= 0;
            end
            else if (dr_v && (d_cnt != OCYC - 1)) begin
                d_cnt  <= d_cnt + 1'b1;
                d_oh   <= d_oh << 1;
                d_row0 <= d_row0 + EPS;
            end
        end
    end
endmodule


// ----------------------------------------------------------------------------
// softmax_online_carry — per-block carry update, blocks in order (cb = 0..B-1):
//   WAIT : sp of block cb complete and its row maxima done
//   b = 0: WALK  lq[r] = min(sp >> log2 N, 127), one word (EPS rows) per cycle
//   b > 0: CEXP  carry-EXP crossbar on t = clamp(log_a + log_l, -128, 127);
//               each drain word e gives lq[r] = min(e + (sp >> log2 N), 127)
//                (= min(((e << log2 N) + sp) >> log2 N, 127), exactly)
//   LOG  : carry-LOG crossbar on lq; drain writes log_l[r] (+ ll events);
//          the word with row S-1 retires the block.
// ----------------------------------------------------------------------------
module softmax_online_carry #(
    parameter S   = 64,
    parameter B   = 4,
    parameter R   = 256,
    parameter C   = 256,
    parameter BUF = 40,
    parameter P   = 8,
    parameter SPW = 12,
    parameter MCW = 3,
    parameter LS  = $clog2(S),
    parameter LN  = $clog2(S)          // log2 of the key length N (mean shift)
) (
    input  wire              clk,
    input  wire              reset,
    input  wire              dpe_reset,
    input  wire [MCW-1:0]    max_cnt,
    input  wire [1:0]        sp_full,
    input  wire [SPW*S-1:0]  sp0,
    input  wire [SPW*S-1:0]  sp1,
    input  wire [9*S-1:0]    la0,
    input  wire [9*S-1:0]    la1,
    output reg  [1:0]        sp_clr,
    output reg  [MCW-1:0]    ret_cnt,
    output reg  [8*S-1:0]    ll_q,
    output wire              ll_valid,
    output wire [MCW-1:0]    ll_blk,
    output wire [LS:0]       ll_base,
    output wire [BUF/8-1:0]  ll_mask,
    output wire [BUF-1:0]    ll_data,
    output reg               done
);
    localparam EPS = BUF / 8;
    localparam WL  = (S + EPS - 1) / EPS;
    localparam [2:0] ST_WAIT = 3'd0, ST_WALK = 3'd1, ST_CEXP = 3'd2, ST_LOG = 3'd3, ST_END = 3'd4;

    reg [2:0]     st;
    reg [MCW-1:0] cb;
    wire          par = cb[0];
    reg [8*S-1:0] lq_q;
    reg [WL-1:0]  w_oh;              // WALK pointer (b = 0)
    reg           cexp_go, clog_go;

    // ---- carry-EXP feed: t = clamp(log_a + log_l, -128, 127) per row --------
    wire [WL-1:0]  cx_k;
    wire [9*S-1:0] la = par ? la1 : la0;
    wire [BUF-1:0] cx_word;
    // per lane l: rows EPS*k + l (k < WL; rows >= S read as zero)
    genvar gl, gk;
    generate
        for (gl = 0; gl < EPS; gl = gl + 1) begin : g_cx
            wire [WL*9-1:0] la_c;
            wire [WL*8-1:0] ll_c;
            for (gk = 0; gk < WL; gk = gk + 1) begin : g_k
                if (EPS * gk + gl < S) begin : g_in
                    assign la_c[gk*9 +: 9] = la[(EPS*gk + gl)*9 +: 9];
                    assign ll_c[gk*8 +: 8] = ll_q[(EPS*gk + gl)*8 +: 8];
                end else begin : g_out
                    assign la_c[gk*9 +: 9] = 9'd0;
                    assign ll_c[gk*8 +: 8] = 8'd0;
                end
            end
            wire [8:0] sla;
            wire [7:0] sll;
            softmax_online_ohmux #(.N(WL), .W(9)) u_la (.sel(cx_k), .in(la_c), .out(sla));
            softmax_online_ohmux #(.N(WL), .W(8)) u_ll (.sel(cx_k), .in(ll_c), .out(sll));
            reg [8:0] sla_q;
            reg [7:0] sll_q;
            always @(posedge clk) begin sla_q <= sla; sll_q <= sll; end
            wire signed [9:0] tsum = $signed(sla_q) + $signed(sll_q);
            assign cx_word[gl*8 +: 8] = (tsum < -10'sd128) ? 8'h80
                                      : (tsum > 10'sd127) ? 8'h7f : tsum[7:0];
        end
    endgenerate
    wire            cx_dv, cx_last;
    wire [WL:0]     cx_doh;
    wire [LS:0]     cx_row0;
    wire [BUF-1:0]  cx_dd;
    softmax_online_wfeed #(.S(S), .R(R), .C(C), .BUF(BUF), .P(P), .MODE(2'b10)) u_cexp (
        .clk(clk), .reset(reset), .dpe_reset(dpe_reset), .start(cexp_go), .k_oh(cx_k), .word(cx_word),
        .dr_v(cx_dv), .d_oh(cx_doh), .d_row0(cx_row0), .dr_data(cx_dd), .last_row(cx_last));

    // ---- lq walker: (pointer, e lanes) from WALK or from the carry-EXP drain --
    wire           wv0  = (st == ST_WALK) || ((st == ST_CEXP) && cx_dv);
    wire [WL-1:0]  woh0 = (st == ST_WALK) ? w_oh : cx_doh[WL-1:0];
    wire [BUF-1:0] e_w0 = (st == ST_WALK) ? {BUF{1'b0}} : cx_dd;
    // walker stage 2 (ladder lc2): selection registered, add/clamp/write next cycle
    reg            wv;
    reg [WL-1:0]   woh;
    reg [BUF-1:0]  e_w;
    reg [MCW-1:0]  wcb;
    always @(posedge clk) begin
        if (reset) wv <= 1'b0;
        else       wv <= wv0;
        woh <= woh0; e_w <= e_w0; wcb <= cb;
    end
    wire [SPW*S-1:0] sp = par ? sp1 : sp0;
    wire [8*EPS-1:0] lq_lane;
    generate
        for (gl = 0; gl < EPS; gl = gl + 1) begin : g_wk
            wire [WL*SPW-1:0] sp_c;
            for (gk = 0; gk < WL; gk = gk + 1) begin : g_k
                if (EPS * gk + gl < S) begin : g_in
                    assign sp_c[gk*SPW +: SPW] = sp[(EPS*gk + gl)*SPW +: SPW];
                end else begin : g_out
                    assign sp_c[gk*SPW +: SPW] = {SPW{1'b0}};
                end
            end
            wire [SPW-1:0] ssp0;
            softmax_online_ohmux #(.N(WL), .W(SPW)) u_sp (.sel(woh0), .in(sp_c), .out(ssp0));
            reg  [SPW-1:0] ssp;
            always @(posedge clk) ssp <= ssp0;
            wire [SPW:0] lsum = e_w[gl*8 +: 8] + (ssp >> LN);
            assign lq_lane[gl*8 +: 8] = (lsum > 127) ? 8'd127 : lsum[7:0];
        end
    endgenerate
    genvar gr;
    generate
        for (gr = 0; gr < S; gr = gr + 1) begin : g_lq
            always @(posedge clk)
                if (wv && woh[gr / EPS]) lq_q[gr*8 +: 8] <= lq_lane[(gr % EPS)*8 +: 8];
        end
    endgenerate

    // ---- carry-LOG feed on lq; drain -> log_l -------------------------------
    wire [WL-1:0]  cl_k;
    wire [BUF-1:0] cl_word;
    generate
        for (gl = 0; gl < EPS; gl = gl + 1) begin : g_cl
            wire [WL*8-1:0] lq_c;
            for (gk = 0; gk < WL; gk = gk + 1) begin : g_k
                if (EPS * gk + gl < S) begin : g_in
                    assign lq_c[gk*8 +: 8] = lq_q[(EPS*gk + gl)*8 +: 8];
                end else begin : g_out
                    assign lq_c[gk*8 +: 8] = 8'd0;
                end
            end
            wire [7:0] slq;
            softmax_online_ohmux #(.N(WL), .W(8)) u_lq (.sel(cl_k), .in(lq_c), .out(slq));
            reg [7:0] slq_q;
            always @(posedge clk) slq_q <= slq;
            assign cl_word[gl*8 +: 8] = slq_q;
        end
    endgenerate
    wire            cl_dv, cl_last;
    wire [WL:0]     cl_doh;
    wire [LS:0]     cl_row0;
    wire [BUF-1:0]  cl_dd;
    softmax_online_wfeed #(.S(S), .R(R), .C(C), .BUF(BUF), .P(P), .MODE(2'b11)) u_clog (
        .clk(clk), .reset(reset), .dpe_reset(dpe_reset), .start(clog_go), .k_oh(cl_k), .word(cl_word),
        .dr_v(cl_dv), .d_oh(cl_doh), .d_row0(cl_row0), .dr_data(cl_dd), .last_row(cl_last));
    wire lg_rows = (st == ST_LOG) && cl_dv && (|cl_doh[WL-1:0]);
    generate
        for (gr = 0; gr < S; gr = gr + 1) begin : g_ll
            always @(posedge clk)
                if (lg_rows && cl_doh[gr / EPS]) ll_q[gr*8 +: 8] <= cl_dd[(gr % EPS)*8 +: 8];
        end
    endgenerate
    assign ll_valid = lg_rows;
    assign ll_blk   = cb;
    assign ll_base  = cl_row0;
    assign ll_data  = cl_dd;
    genvar gm;
    generate
        for (gm = 0; gm < EPS; gm = gm + 1) begin : g_llm
            assign ll_mask[gm] = lg_rows && (cl_row0 + gm < S);
        end
    endgenerate

    // ---- sequencer ----------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            st <= ST_WAIT; cb <= 0; ret_cnt <= 0; w_oh <= 0;
            cexp_go <= 1'b0; clog_go <= 1'b0; sp_clr <= 2'b00; done <= 1'b0;
        end
        else begin
            cexp_go <= 1'b0; clog_go <= 1'b0; sp_clr <= 2'b00; done <= 1'b0;
            case (st)
                ST_WAIT:
                    if (sp_full[par] && (max_cnt > cb)) begin
                        if (cb == 0) begin
                            st <= ST_WALK; w_oh <= {{(WL-1){1'b0}}, 1'b1};
                        end
                        else begin
                            st <= ST_CEXP; cexp_go <= 1'b1;
                        end
                    end
                ST_WALK: begin
                    w_oh <= w_oh << 1;
                    if (w_oh[WL-1]) begin st <= ST_LOG; clog_go <= 1'b1; end
                end
                ST_CEXP:
                    if (cx_last) begin st <= ST_LOG; clog_go <= 1'b1; end
                ST_LOG:
                    if (cl_last) begin
                        sp_clr[par] <= 1'b1;
                        ret_cnt <= ret_cnt + 1'b1;
                        cb <= cb + 1'b1;
                        if (cb == B - 1) begin st <= ST_END; done <= 1'b1; end
                        else st <= ST_WAIT;
                    end
                default: ;
            endcase
        end
    end

endmodule


// ----------------------------------------------------------------------------
// softmax_online_instage — the input stage (spec §5: FIFO_LAT, OPEN_LAT, PACK_LAT, SCR_LAT).
//   FIFOs   one per E4 channel: 2^DFW words of BUF bits, 1W1R, registered read (two tiles)
//   reader  a tile opens on `open` (idle, open_ok, FIFO 0 non-empty) and is read from the next cycle: window w
//           from FIFO w mod S_CH, its LW words, one per cycle whenever the FIFO holds the next word; s_cr pulses
//           one cycle after the tile's last read; the reader is idle for one cycle before the next tile opens
//   packer  window words (5 bytes; L_LAST in a window's last word) -> 5-byte stream words, each tile's last word
//           padded; a word completed by the read at r leaves at r + 2, a second word of the same read at r + 3
// ----------------------------------------------------------------------------
module softmax_online_instage #(
    parameter S_CH   = 1,
    parameter BUF    = 40,
    parameter LW     = 13,
    parameter NWB    = 2,
    parameter DFW    = 5,
    parameter L_LAST = 4
) (
    input  wire                clk,
    input  wire                reset,
    input  wire [S_CH-1:0]     s_v,
    input  wire [S_CH*BUF-1:0] s_d,
    input  wire                open_ok,
    output wire                open,
    output reg                 s_cr,
    output reg                 out_v,
    output reg  [BUF-1:0]      out_d
);
    localparam EPS = BUF / 8;
    localparam LWW = $clog2(LW + 1);
    localparam NWW = $clog2(NWB + 1);
    localparam [BUF-1:0] LMASK = {BUF{1'b1}} >> (BUF - 8 * L_LAST);   // the bytes of a window's last word

    // ---------------- FIFOs ----------------------------------------------
    wire [S_CH-1:0]     ne, rd_ch;
    wire [S_CH*BUF-1:0] rq_all;
    genvar gc;
    generate
        for (gc = 0; gc < S_CH; gc = gc + 1) begin : g_f
            reg [DFW:0]   wp, rp;
            reg [BUF-1:0] mem [0:(1 << DFW) - 1];
            reg [BUF-1:0] rq;
            always @(posedge clk) begin
                if (reset) begin wp <= 0; rp <= 0; end
                else begin
                    if (s_v[gc])   wp <= wp + 1'b1;
                    if (rd_ch[gc]) rp <= rp + 1'b1;
                end
            end
            always @(posedge clk) begin
                if (s_v[gc]) mem[wp[DFW-1:0]] <= s_d[gc*BUF +: BUF];
                rq <= mem[rp[DFW-1:0]];
            end
            assign ne[gc] = (wp != rp);
            assign rq_all[gc*BUF +: BUF] = rq;
        end
    endgenerate

    // ---------------- reader ----------------------------------------------
    reg            act;
    reg [LWW-1:0]  c;
    reg [NWW-1:0]  w;
    reg [S_CH-1:0] fsel;                       // one-hot: the FIFO of window w
    wire [S_CH-1:0] fnext = (fsel << 1) | (fsel >> (S_CH - 1));
    wire           rd     = act && (|(ne & fsel));
    wire           c_last = (c == LW - 1);
    wire           w_last = (w == NWB - 1);
    assign open  = !act && open_ok && ne[0];
    assign rd_ch = rd ? fsel : {S_CH{1'b0}};
    always @(posedge clk) begin
        if (reset) begin
            act <= 1'b0; c <= 0; w <= 0; fsel <= 1; s_cr <= 1'b0;
        end
        else begin
            s_cr <= rd && c_last && w_last;
            if (open) begin
                act <= 1'b1; c <= 0; w <= 0; fsel <= 1;
            end
            else if (rd) begin
                if (c_last) begin
                    c <= 0;
                    fsel <= fnext;
                    if (w_last) begin act <= 1'b0; w <= 0; end
                    else w <= w + 1'b1;
                end
                else c <= c + 1'b1;
            end
        end
    end

    // ---------------- packer ----------------------------------------------
    reg            p_v, p_last, p_short;
    reg [S_CH-1:0] p_sel;
    always @(posedge clk) begin
        if (reset) p_v <= 1'b0;
        else       p_v <= rd;
        p_last  <= c_last && w_last;
        p_short <= c_last;
        p_sel   <= fsel;
    end
    wire [BUF-1:0] p_raw;
    softmax_online_ohmux #(.N(S_CH), .W(BUF)) u_sel (.sel(p_sel), .in(rq_all), .out(p_raw));
    wire [BUF-1:0] p_d = p_short ? (p_raw & LMASK) : p_raw;
    wire [3:0]     p_n = p_short ? L_LAST : EPS;
    reg  [8*(EPS-1)-1:0]   rs_d;               // residual bytes (at most EPS - 1)
    reg  [2:0]             rs_n;
    reg                    fl_v;               // the tile's last (partial) word is pending
    wire [8*(2*EPS-1)-1:0] cat = {{(8*EPS){1'b0}}, rs_d} | ({{(8*(EPS-1)){1'b0}}, p_d} << {rs_n, 3'b000});
    wire [3:0]             tot = rs_n + p_n;
    always @(posedge clk) begin
        if (reset) begin out_v <= 1'b0; rs_d <= 0; rs_n <= 0; fl_v <= 1'b0; end
        else begin
            out_v <= 1'b0;
            if (p_v) begin
                if (tot >= EPS) begin                       // a whole word
                    out_v <= 1'b1;
                    out_d <= cat[BUF-1:0];
                    rs_d  <= cat[BUF +: 8*(EPS-1)];
                    rs_n  <= tot - EPS;
                    fl_v  <= p_last && (tot != EPS);
                end
                else if (p_last) begin                      // the tile ends inside this word
                    out_v <= 1'b1;
                    out_d <= cat[BUF-1:0];
                    rs_d  <= 0;
                    rs_n  <= 0;
                end
                else begin
                    rs_d <= cat[8*(EPS-1)-1:0];
                    rs_n <= tot;
                end
            end
            else if (fl_v) begin                            // the reader's idle cycle: the last partial word
                out_v <= 1'b1;
                out_d <= {{8{1'b0}}, rs_d};
                rs_d  <= 0;
                rs_n  <= 0;
                fl_v  <= 1'b0;
            end
        end
    end
endmodule


// ----------------------------------------------------------------------------
// softmax_online_top — the online softmax (v3/ip/softmax/spec.md).
//   E4 s_v / s_d (S_CH channels) -> input stage -> the core's block stream; s_cr per tile
//   E5 lp_v / lp_d: the EXP feeds' pops (window g of the run on lane g mod N_EXP); lp_cr per block
//   E6 la_v / la_d: log a of every row of the blocks b >= 1, as the row's maximum completes; la_cr per block
//   E7 ll_v / ll_d: the run's final log l, rows 0 .. S-1 on consecutive cycles; ll_cr per run
//   start / busy: the start port (busy falling is its credit); done one cycle after the last E7 word
// ----------------------------------------------------------------------------
module softmax_online_top #(
    parameter S      = 64,
    parameter BKV    = 16,
    parameter N      = S,
    parameter R      = 256,
    parameter C      = 256,
    parameter BUF    = 40,
    parameter P      = 8,
    parameter N_EXP  = 2,
    parameter S_CH   = 1,
    parameter LP_CAP = 2,
    parameter LA_CAP = 2,
    parameter LL_CAP = 1
) (
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 start,
    output wire                 busy,
    output reg                  done,
    input  wire [S_CH-1:0]      s_v,
    input  wire [S_CH*BUF-1:0]  s_d,
    output wire                 s_cr,
    output wire [N_EXP-1:0]     lp_v,
    output wire [N_EXP*BUF-1:0] lp_d,
    input  wire                 lp_cr,
    output wire                 la_v,
    output wire [8:0]           la_d,
    input  wire                 la_cr,
    output wire                 ll_v,
    output wire [7:0]           ll_d,
    input  wire                 ll_cr
);
    localparam EPS    = BUF / 8;
    localparam LS     = $clog2(S);
    localparam AW     = $clog2(S * N) + 1;
    localparam MCW    = $clog2(N / BKV + 2);
    localparam I      = (R < C) ? R : C;
    localparam B      = N / BKV;
    localparam LN     = $clog2(N);
    localparam LB     = $clog2(BKV);
    localparam BLKEL  = S * BKV;
    localparam NWEL   = (BLKEL < I) ? BLKEL : I;
    localparam NWB    = BLKEL / NWEL;
    localparam PW     = B * NWB;
    localparam WPW    = (NWEL + EPS - 1) / EPS;
    localparam NB     = (N_EXP < 2) ? 2 : N_EXP;
    localparam NBX    = NB / N_EXP;
    localparam DW     = (2 * NWB >= NB) ? (2 * NWB) / NB : 1;
    localparam BDEPTH = DW * WPW;
    localparam BAW    = (BDEPTH > 1) ? $clog2(BDEPTH) : 1;
    localparam SPW    = 8 + LB;
    localparam LW     = (I * 8 + BUF - 1) / BUF;
    localparam L_LAST = I - EPS * (LW - 1);
    localparam T_W    = NWB / S_CH;
    localparam DFW    = $clog2(2 * T_W * LW);
    localparam LPW    = $clog2(LP_CAP + 1);
    localparam LAW    = $clog2(LA_CAP + 1);
    localparam LLW    = $clog2(LL_CAP + 1);

    // ---------------- domain guards ------------------------------------------
    generate
        if ((S & (S - 1)) != 0 || (BKV & (BKV - 1)) != 0 || (I & (I - 1)) != 0
            || (N & (N - 1)) != 0 || (N_EXP & (N_EXP - 1)) != 0 || (S_CH & (S_CH - 1)) != 0
            || BKV < 8 || BKV > N || BKV > I || S < 8 || S > I || EPS != 5
            || BLKEL < I || S_CH > NWB || (PW % N_EXP) != 0
            || LP_CAP < 1 || LA_CAP < 1 || LL_CAP < 1)
        begin : g_bad_param
            softmax_online_parameter_outside_domain u_bad ();
        end
    endgenerate

    // ---------------- run control ------------------------------------------------
    // `start` while not busy is registered into `rearm`; clr = power-on reset | rearm clears the run state of the
    // core (the crossbars, the input stage and the credit counters keep theirs). The run is armed from 2 cycles
    // after `start` until `done`; busy = rearm | armed (busy falling is the start port's credit).
    reg  rearm, started, done_seen;
    wire clr   = reset || rearm;
    wire armed = started && !done_seen;
    assign busy = rearm || armed;
    always @(posedge clk) begin
        if (reset) begin rearm <= 1'b0; started <= 1'b0; end
        else begin
            rearm <= start && !busy;
            if (rearm) started <= 1'b1;
        end
    end
    always @(posedge clk) begin
        if (clr) done_seen <= 1'b0;
        else if (done) done_seen <= 1'b1;
    end

    // ---------------- tile gate + credit counters --------------------------------
    wire [MCW-1:0] ret_cnt;
    wire           open, e7_go;
    reg  [MCW-1:0] rb;                         // tiles opened in this run
    reg  [LPW-1:0] lp_cnt;
    reg  [LAW-1:0] la_cnt;
    reg  [LLW-1:0] ll_cnt;
    wire           rb0     = (rb == 0);
    wire           open_ok = armed && (rb < B) && (rb < ret_cnt + 2) && (lp_cnt != 0) && (rb0 || (la_cnt != 0));
    always @(posedge clk) begin
        if (clr) rb <= 0;
        else if (open) rb <= rb + 1'b1;
    end
    always @(posedge clk) begin
        if (reset) begin lp_cnt <= LP_CAP; la_cnt <= LA_CAP; ll_cnt <= LL_CAP; end
        else begin
            lp_cnt <= lp_cnt + lp_cr - open;
            la_cnt <= la_cnt + la_cr - (open && !rb0);
            ll_cnt <= ll_cnt + ll_cr - e7_go;
        end
    end

    // ---------------- input stage ------------------------------------------------
    wire           ld_valid;
    wire [BUF-1:0] ld_data;
    softmax_online_instage #(.S_CH(S_CH), .BUF(BUF), .LW(LW), .NWB(NWB), .DFW(DFW), .L_LAST(L_LAST)) u_in (
        .clk(clk), .reset(reset), .s_v(s_v), .s_d(s_d), .open_ok(open_ok), .open(open), .s_cr(s_cr),
        .out_v(ld_valid), .out_d(ld_data));

    // ---------------- store + block max --------------------------------------
    wire [NB*BAW-1:0] st_raddr;
    wire [NB*BUF-1:0] st_rdata;
    softmax_online_store #(.NWEL(NWEL), .BLKEL(BLKEL), .WPW(WPW), .NB(NB), .BDEPTH(BDEPTH),
               .BAW(BAW), .EPS(EPS)) u_store (
        .clk(clk), .reset(clr), .ld_valid(ld_valid), .ld_data(ld_data),
        .rd_addr(st_raddr), .rd_data(st_rdata));

    wire [8*S-1:0] m_q, mn0, mn1;
    wire [9*S-1:0] la0, la1;
    wire [MCW-1:0] max_cnt;
    wire           la_valid;
    wire [MCW-1:0] la_blk;
    wire [LS-1:0]  la_row;
    wire [8:0]     la_val;
    softmax_online_loadmax #(.S(S), .BKV(BKV), .EPS(EPS), .MCW(MCW)) u_lmax (
        .clk(clk), .reset(clr), .ld_valid(ld_valid), .ld_data(ld_data),
        .m_q(m_q), .mn0(mn0), .mn1(mn1), .la0(la0), .la1(la1), .max_cnt(max_cnt),
        .la_valid(la_valid), .la_blk(la_blk), .la_row(la_row), .la_val(la_val));
    assign la_v = la_valid && (la_blk != 0);
    assign la_d = la_val;

    // ---------------- EXP crossbars ------------------------------------------
    wire [N_EXP*BAW-1:0] fx_raddr;
    wire [N_EXP*AW-1:0]  lp_base;
    wire [N_EXP*EPS-1:0] lp_mask;
    wire [N_EXP-1:0]     dr_valid;
    wire [N_EXP*AW-1:0]  dr_base;
    wire [N_EXP*EPS-1:0] dr_mask;
    wire [N_EXP*BUF-1:0] dr_data;
    genvar gc, gb;
    generate
        for (gc = 0; gc < N_EXP; gc = gc + 1) begin : gen_exp
            wire [NBX*BUF-1:0] own;
            for (gb = 0; gb < NBX; gb = gb + 1) begin : g_own
                assign own[gb*BUF +: BUF] = st_rdata[(gb*N_EXP + gc)*BUF +: BUF];
            end
            softmax_online_xfeed #(.CX(gc), .N_EXP(N_EXP), .S(S), .BKV(BKV), .R(R), .C(C), .BUF(BUF),
                       .P(P), .NWEL(NWEL), .NWB(NWB), .B(B), .WPW(WPW), .NB(NB), .DW(DW),
                       .BAW(BAW), .AW(AW), .MCW(MCW)) u_feed (
                .clk(clk), .reset(clr), .dpe_reset(reset), .go(started), .max_cnt(max_cnt),
                .mn0(mn0), .mn1(mn1),
                .rd_addr(fx_raddr[gc*BAW +: BAW]), .rd_data(own),
                .lp_valid(lp_v[gc]), .lp_base(lp_base[gc*AW +: AW]),
                .lp_mask(lp_mask[gc*EPS +: EPS]), .lp_data(lp_d[gc*BUF +: BUF]),
                .dr_valid(dr_valid[gc]), .dr_base(dr_base[gc*AW +: AW]),
                .dr_mask(dr_mask[gc*EPS +: EPS]), .dr_data(dr_data[gc*BUF +: BUF]));
        end
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_bank
            assign st_raddr[gb*BAW +: BAW] = fx_raddr[(gb % N_EXP)*BAW +: BAW];
        end
    endgenerate

    // ---------------- row sums (sp) per block, ping-pong by parity -----------
    wire [N_EXP-1:0]    acc_we;
    wire [N_EXP*AW-1:0] acc_row;           // global row = b*S + r
    wire [N_EXP*32-1:0] acc_sum;
    softmax_online_row_acc #(.RS(BKV), .N_EXP(N_EXP), .PORTS(BUF), .AW(AW)) u_acc (
        .clk(clk), .reset(clr),
        .dr_valid(dr_valid), .dr_base(dr_base), .dr_mask(dr_mask), .dr_data(dr_data),
        .ev_we(acc_we), .ev_row(acc_row), .ev_sum(acc_sum));

    wire [2*SPW*S-1:0] sp_both;
    wire [1:0]         sp_full;
    wire [1:0]         sp_clr;
    genvar gp, gr;
    generate
        for (gp = 0; gp < 2; gp = gp + 1) begin : g_sp
            for (gr = 0; gr < S; gr = gr + 1) begin : g_r
                reg [SPW-1:0] v;
                integer q;
                always @(posedge clk)
                    for (q = 0; q < N_EXP; q = q + 1)
                        if (acc_we[q] && (acc_row[q*AW + LS] == gp)
                            && (acc_row[q*AW +: LS] == gr))
                            v <= acc_sum[q*32 +: SPW];
                assign sp_both[(gp*S + gr)*SPW +: SPW] = v;
            end
            // completed rows of the slot
            reg [LS:0] cnt;
            reg [$clog2(N_EXP + 1):0] n;   // holds N_EXP simultaneous completions (was [3:0]: wrapped at 16, PRE-1)
            integer q2;
            always @(*) begin
                n = 0;
                for (q2 = 0; q2 < N_EXP; q2 = q2 + 1)
                    n = n + (acc_we[q2] && (acc_row[q2*AW + LS] == gp));
            end
            always @(posedge clk) begin
                if (clr || sp_clr[gp]) cnt <= 0;
                else cnt <= cnt + n;
            end
            assign sp_full[gp] = (cnt == S);
        end
    endgenerate
    wire [SPW*S-1:0] sp0 = sp_both[0 +: SPW*S];
    wire [SPW*S-1:0] sp1 = sp_both[SPW*S +: SPW*S];

    // ---------------- carry ---------------------------------------------------
    wire [8*S-1:0] ll_q;
    wire           c_ll_valid, core_done;
    wire [MCW-1:0] c_ll_blk;
    wire [LS:0]    c_ll_base;
    wire [EPS-1:0] c_ll_mask;
    wire [BUF-1:0] c_ll_data;
    softmax_online_carry #(.S(S), .B(B), .R(R), .C(C), .BUF(BUF), .P(P), .SPW(SPW), .MCW(MCW), .LN(LN)) u_carry (
        .clk(clk), .reset(clr), .dpe_reset(reset), .max_cnt(max_cnt), .sp_full(sp_full),
        .sp0(sp0), .sp1(sp1), .la0(la0), .la1(la1),
        .sp_clr(sp_clr), .ret_cnt(ret_cnt), .ll_q(ll_q),
        .ll_valid(c_ll_valid), .ll_blk(c_ll_blk), .ll_base(c_ll_base), .ll_mask(c_ll_mask),
        .ll_data(c_ll_data), .done(core_done));

    // ---------------- E7: the final log l, row by row; done -----------------------
    // Decided when the core is done (or later, when an ll credit arrives): e7_row walks 0 .. S-1, the 2-cycle
    // registered read gives row e7_row two cycles later on ll_d; done one cycle after the last row.
    reg           ll_pend, e7_act, e7_v1, e7_v2;
    reg [LS:0]    e7_row;
    assign e7_go = (core_done || ll_pend) && !e7_act && (ll_cnt != 0);
    always @(posedge clk) begin
        if (clr) begin
            ll_pend <= 1'b0; e7_act <= 1'b0; e7_row <= 0; e7_v1 <= 1'b0; e7_v2 <= 1'b0; done <= 1'b0;
        end
        else begin
            e7_v1 <= e7_act;
            e7_v2 <= e7_v1;
            done  <= e7_v2 && !e7_v1;
            if (e7_go) begin
                ll_pend <= 1'b0; e7_act <= 1'b1; e7_row <= 0;
            end
            else begin
                if (core_done) ll_pend <= 1'b1;
                if (e7_act) begin
                    if (e7_row == S - 1) e7_act <= 1'b0;
                    else e7_row <= e7_row + 1'b1;
                end
            end
        end
    end
    reg [LS-1:0] rd_row_q;
    reg [7:0]    rd_ll;
    always @(posedge clk) begin
        rd_row_q <= e7_row[LS-1:0];
        rd_ll    <= ll_q[rd_row_q*8 +: 8];
    end
    assign ll_v = e7_v2;
    assign ll_d = rd_ll;
endmodule
