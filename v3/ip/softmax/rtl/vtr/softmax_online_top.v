// ============================================================================
// softmax_online_top.v — NL-DPE online softmax, log-domain carry; VTR twin (v3, the bit-plane pass): bare `dpe`
// hard blocks (port contract ip/xbar/rtl/vtr/dpe_blackbox.v; the widths from the arch XML, vtr/arch_rowpar/), never
// simulated. Simulation twin: ip/softmax/rtl/sim/softmax_online_top.v (gate: tools/check_twins.py).
// Contract: v3/ip/softmax/spec.md; behavior model v3/simulator/models/softmaxOnline.py (GATE 2: every port event and
// value equal). Value contract: softmax_online_ref.logcarry_stage_values.
//
// Rewritten 2026-10-23 for the bit-plane pass (spec §10 item 6) from the pre-bit-plane F3 module (git show
// 8acf73c:v3/ip/softmax/rtl/sim/softmax_online_top.v). Kept: the run control, the tile gate and the credit counters,
// the row max with its rotating m ring, the m_new / log a / sp tables by block parity, the carry's sequence. New:
// plane-word FIFOs and a double corner-turn instead of the packer; store words of Q_W = I/8 bytes; the EXP feeds
// subtract and clamp at Q_W lanes into a double corner-turn and send one burst per window (the E5 window); the row
// sums add per-plane counts; the carry's inputs and outputs are planes of the tables; E6 and E7 are plane windows.
//
// Behavioral simulation: compile with v3/ip/xbar/rtl/sim/dpe_nldpe.v (the `dpe` model; the TB preloads the identity
// weights through the hierarchical paths below). The VTR twin is v3/ip/softmax/rtl/vtr/softmax_online_top.v: the
// same modules and datapath with a bare `dpe` hard block (gate: v3/tools/check_twins.py).
//
// Per row r, key blocks b = 0..B-1 (B = N/BKV), P24 ACAM forms:
//   m_blk = max_j s ;  m_new = m_blk (b=0) | max(m, m_blk) ;  log_a = m - m_new
//   logp_j = max(s_j - m_new, -128)            -> lp_* ports (log|P| to P.V), the EXP crossbars' input bursts
//   sp     = sum_j ACAM_EXP(logp_j)            (per-plane counts of the EXP drain, weighted by 2^b)
//   l'     = sp (b=0) | (ACAM_EXP(clamp(log_a + log_l)) << log2 N) + sp
//   log_l' = ACAM_LOG(min(l' >> log2 N, 127))  -> ll_* ports; carry (m, log_l)
//
// Structure (modules in datapath order):
//   softmax_online_dpe      one crossbar with identity weights and a fixed ACAM mode
//   softmax_online_ohmux    one-hot read mux helper
//   softmax_online_popcnt   balanced population count (combinational)
//   softmax_online_instage  plane FIFOs, the tile reader, the double corner-turn -> the core stream (Q_W bytes/word)
//   softmax_online_store    2-block ring score store: NB banks of Q_W-byte words, 1W/1R sync-read RAMs
//   softmax_online_loadmax  per-block row max while the block streams in; running max m, m_new / log_a tables
//   softmax_online_xfeed    one EXP crossbar: window reads, s - m_new and the clamp, a double corner-turn, the burst;
//                           the drain's row sums (count register, weighted add)
//   softmax_online_wfeed    one carry crossbar (S x S) fed with a plane register
//   softmax_online_carry    per-block carry: [EXP of clamp(log_a + log_l)] -> lq -> LOG -> log_l; retires blocks
//   softmax_online_top
//
// Flow control: block b streams in once block b-2 has retired (its carry is done), so every ping-pong slot (store,
// m_new, log_a, sp) of b is free. EXP windows of block b start once its row maxima are complete.
//
// Shapes: S query rows per run (S <= I), N keys in B = N/BKV blocks. A run re-arms on `start` while idle
// (back-to-back row blocks; the crossbars keep their power-on state; the input stage and the credit counters keep
// theirs).
//
// Domain (elaboration guards): S, N, BKV, I = min(R, C), N_EXP, S_CH powers of two; 8 <= S <= I;
// 8 <= BKV <= min(N, I); P = 8; S*BKV >= I; S_CH <= NWB; B*NWB a multiple of N_EXP; credit capacities >= 1.
//
// TB preload paths: gen_exp[g].u_feed.u_x.u_dpe, u_carry.u_cexp.u_x.u_dpe, u_carry.u_clog.u_x.u_dpe.
// ============================================================================

`timescale 1ns / 1ps

// ----------------------------------------------------------------------------
// softmax_online_dpe — one NL-DPE crossbar with identity weights and a fixed ACAM mode: R-bit input planes, C-bit
// drain planes (ip/xbar/spec.md).
// ----------------------------------------------------------------------------
module softmax_online_dpe #(
    parameter R = 256,
    parameter C = 256,
    parameter P = 8,
    parameter [1:0] MODE = 2'b10,
    parameter RU = R,                  // P29 pass extent: rows loaded, columns drained
    parameter CU = C
) (
    input  wire         clk,
    input  wire         reset,
    input  wire [R-1:0] data_in,
    input  wire         w_buf_en,
    output wire         ready,
    output wire [C-1:0] data_out,
    output wire         done,
    output wire         full
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
// softmax_online_ohmux — one-hot select of N W-bit words (flat): per-bit reduction OR of the masked inputs
// (generate-only, no combinational loops).
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
// softmax_online_popcnt — the number of set bits of a W-bit word (W a power of two), a balanced tree of adds
// (combinational).
// ----------------------------------------------------------------------------
module softmax_online_popcnt #(
    parameter W = 16
) (
    input  wire [W-1:0]         x,
    output wire [$clog2(W):0]   n
);
    localparam L = $clog2(W);
    function integer off;                      // the first bit of level l (level l: W >> l counts of l + 1 bits)
        input integer l;
        integer i;
        begin
            off = 0;
            for (i = 0; i < l; i = i + 1) off = off + (W >> i) * (i + 1);
        end
    endfunction
    wire [off(L + 1)-1:0] t;
    assign t[0 +: W] = x;
    genvar gl, gk;
    generate
        for (gl = 0; gl < L; gl = gl + 1) begin : g_lvl
            localparam OI = off(gl), OO = off(gl + 1), NO = W >> (gl + 1), WI = gl + 1;
            for (gk = 0; gk < NO; gk = gk + 1) begin : g_k
                assign t[OO + gk*(WI+1) +: WI+1] = t[OI + 2*gk*WI +: WI] + t[OI + (2*gk+1)*WI +: WI];
            end
        end
    endgenerate
    assign n = t[off(L) +: L+1];
endmodule


// ----------------------------------------------------------------------------
// softmax_online_instage — the input stage (spec §5: FIFO_LAT, OPEN_LAT, CT_LAT, SCR_LAT).
//   FIFOs        one per E4 channel: 2^DFW plane words of I bits, 1W1R, registered read (two tiles)
//   reader       a tile opens on `open` (idle, open_ok, FIFO 0 non-empty) and is read from the next cycle: window w
//                from FIFO w mod S_CH, its P planes, one per cycle whenever the FIFO holds the next plane; s_cr one
//                cycle after the tile's last read; the reader is idle for one cycle before the next tile opens
//   corner-turn  two buffers of P planes x I bits: a window's planes are written as they arrive from the registered
//                read; once its last plane is in, its P core words leave one per cycle (word c = bytes c*Q_W .. of the
//                window: bit b of byte k = plane b's bit c*Q_W + k), while the next window fills the other buffer
// ----------------------------------------------------------------------------
module softmax_online_instage #(
    parameter S_CH = 1,
    parameter I    = 64,
    parameter P    = 8,
    parameter NWB  = 2,
    parameter DFW  = 5
) (
    input  wire              clk,
    input  wire              reset,
    input  wire [S_CH-1:0]   s_v,
    input  wire [S_CH*I-1:0] s_d,
    input  wire              open_ok,
    output wire              open,
    output reg               s_cr,
    output wire              out_v,
    output wire [I-1:0]      out_d
);
    localparam Q_W = I / 8;
    localparam PW  = $clog2(P);
    localparam NWW = $clog2(NWB + 1);

    // ---------------- FIFOs ----------------------------------------------
    wire [S_CH-1:0]   ne, rd_ch;
    wire [S_CH*I-1:0] rq_all;
    genvar gc;
    generate
        for (gc = 0; gc < S_CH; gc = gc + 1) begin : g_f
            reg [DFW:0] wp, rp;
            reg [I-1:0] mem [0:(1 << DFW) - 1];
            reg [I-1:0] rq;
            always @(posedge clk) begin
                if (reset) begin wp <= 0; rp <= 0; end
                else begin
                    if (s_v[gc])   wp <= wp + 1'b1;
                    if (rd_ch[gc]) rp <= rp + 1'b1;
                end
            end
            always @(posedge clk) begin
                if (s_v[gc]) mem[wp[DFW-1:0]] <= s_d[gc*I +: I];
                rq <= mem[rp[DFW-1:0]];
            end
            assign ne[gc] = (wp != rp);
            assign rq_all[gc*I +: I] = rq;
        end
    endgenerate

    // ---------------- reader ----------------------------------------------
    reg            act;
    reg [PW-1:0]   c;
    reg [NWW-1:0]  w;
    reg [S_CH-1:0] fsel;                       // one-hot: the FIFO of window w
    wire [S_CH-1:0] fnext = (fsel << 1) | (fsel >> (S_CH - 1));
    wire           rd     = act && (|(ne & fsel));
    wire           c_last = (c == P - 1);
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

    // ---------------- the registered read: the word read at r is the core's word at r + 1 (RD_LAT) -------------
    // E4 carries value words (the trims of 2026-10-23): word c of a window is bytes c*Q_W .. of the window, as the
    // core takes them, so no corner-turn
    reg            p_v;
    reg [S_CH-1:0] p_sel;
    always @(posedge clk) begin
        p_v   <= !reset && rd;
        p_sel <= fsel;
    end
    softmax_online_ohmux #(.N(S_CH), .W(I)) u_sel (.sel(p_sel), .in(rq_all), .out(out_d));
    assign out_v = p_v;
endmodule


// ----------------------------------------------------------------------------
// softmax_online_store — ring of score blocks (spec §6): NB banks of BDEPTH = DW*P words of Q_W bytes (I bits),
// 1W/1R sync-read RAMs. Core stream word c of the run's window g goes to bank g mod NB, address {(g / NB) mod DW, c};
// the write controls are registered before the RAM port.
// ----------------------------------------------------------------------------
module softmax_online_store #(
    parameter I   = 64,
    parameter P   = 8,
    parameter NB  = 2,
    parameter DW  = 4,
    parameter BAW = 5
) (
    input  wire              clk,
    input  wire              reset,
    input  wire              ld_valid,
    input  wire [I-1:0]      ld_data,
    input  wire [NB*BAW-1:0] rd_addr,
    output wire [NB*I-1:0]   rd_data
);
    localparam LNB = $clog2(NB);
    localparam LDW = $clog2(DW);
    localparam PW  = $clog2(P);
    localparam GW  = LNB + LDW + 1;
    reg [GW-1:0] t_g;                          // the incoming word's window (low bits)
    reg [PW-1:0] t_c;                          // its word in the window
    always @(posedge clk) begin
        if (reset) begin t_g <= 0; t_c <= 0; end
        else if (ld_valid) begin
            t_c <= t_c + 1'b1;
            if (t_c == P - 1) t_g <= t_g + 1'b1;
        end
    end
    wire [BAW-1:0] addr = (((t_g >> LNB) & (DW - 1)) << PW) | t_c;
    reg            w_we;
    reg [LNB-1:0]  w_b;
    reg [BAW-1:0]  w_a;
    reg [I-1:0]    w_d;
    always @(posedge clk) begin
        w_we <= !reset && ld_valid;
        w_b  <= t_g[LNB-1:0];
        w_a  <= addr;
        w_d  <= ld_data;
    end
    genvar gb;
    generate
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_b
            reg [I-1:0] mem [0:DW*P-1];
            reg [I-1:0] rq;
            always @(posedge clk) begin
                if (w_we && w_b == gb) mem[w_a] <= w_d;
                rq <= mem[rd_addr[gb*BAW +: BAW]];
            end
            assign rd_data[gb*I +: I] = rq;
        end
    endgenerate
endmodule


// ----------------------------------------------------------------------------
// softmax_online_loadmax — per-block row maxima while the block streams in (spec §5 MAX_LAT = log2 G + MAX_REG):
// the input register, log2 G registered levels of a signed max over each group of G = min(Q_W, BKV) elements of a
// word, then the running max with m. A row spans WPR = BKV/G words; a word holds NGR = Q_W/G rows. At a row's last
// word: m_new = m_blk (b = 0) | max(m, m_blk) and log_a = m - m_new (0 at b = 0) are written into the block parity's
// tables. The running max m is a rotating ring (rows complete in order every block, so the completing rows are at
// its head: no S:1 read mux). max_cnt = blocks whose rows are all done; blk_done is high in the cycle the block's last
// m_new is readable (a_last + MAX_LAT), with the block's parity and first-block flag.
// ----------------------------------------------------------------------------
module softmax_online_loadmax #(
    parameter S   = 64,
    parameter BKV = 16,
    parameter Q_W = 8,
    parameter MCW = 3
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             ld_valid,
    input  wire [8*Q_W-1:0] ld_data,
    output reg  [8*S-1:0]   mn0,
    output reg  [8*S-1:0]   mn1,
    output reg  [9*S-1:0]   la0,
    output reg  [9*S-1:0]   la1,
    output reg  [MCW-1:0]   max_cnt,
    output reg              blk_done,
    output reg              blk_par,
    output reg              blk_first
);
    localparam G      = (Q_W < BKV) ? Q_W : BKV;
    localparam L      = $clog2(G);
    localparam NGR    = Q_W / G;                // rows completed by a row's last word
    localparam WPR    = BKV / G;                // words per row
    localparam STREAM = (S * BKV) / Q_W;        // words per block
    localparam SW     = $clog2(STREAM + 1);
    localparam NRG    = S / NGR;                // row groups (written together)

    function signed [7:0] smax;
        input signed [7:0] a, b;
        smax = (a > b) ? a : b;
    endfunction
    function integer lo;                        // the first bit of tree level l
        input integer l;
        integer i;
        begin
            lo = 0;
            for (i = 0; i < l; i = i + 1) lo = lo + 8 * (Q_W >> i);
        end
    endfunction

    // the input register
    reg             m0_v, m0_first, m0_last, m0_blast;
    reg [8*Q_W-1:0] m0_d;
    reg [SW-1:0]    t_i;                         // the incoming word's index in the block
    always @(posedge clk) begin
        if (reset) begin m0_v <= 1'b0; t_i <= 0; end
        else begin
            m0_v <= ld_valid;
            if (ld_valid) t_i <= (t_i == STREAM - 1) ? {SW{1'b0}} : t_i + 1'b1;
        end
        m0_d     <= ld_data;
        m0_first <= ((t_i & (WPR - 1)) == 0);
        m0_last  <= ((t_i & (WPR - 1)) == WPR - 1);
        m0_blast <= (t_i == STREAM - 1);
    end

    // the max tree: level l holds Q_W >> l maxima; the control travels with it
    wire [lo(L + 1)-1:0] tv;
    wire [L:0]           cv, cf, cl, cb;
    assign tv[0 +: 8*Q_W] = m0_d;
    assign cv[0] = m0_v;
    assign cf[0] = m0_first;
    assign cl[0] = m0_last;
    assign cb[0] = m0_blast;
    genvar gl;
    generate
        for (gl = 0; gl < L; gl = gl + 1) begin : g_lvl
            localparam OI = lo(gl), OO = lo(gl + 1), NO = Q_W >> (gl + 1);
            reg [8*NO-1:0] r;
            reg            v_q, f_q, l_q, b_q;
            integer k;
            always @(posedge clk) begin
                for (k = 0; k < NO; k = k + 1)
                    r[k*8 +: 8] <= smax(tv[OI + 2*k*8 +: 8], tv[OI + (2*k+1)*8 +: 8]);
                v_q <= !reset && cv[gl];
                f_q <= cf[gl];
                l_q <= cl[gl];
                b_q <= cb[gl];
            end
            assign tv[OO +: 8*NO] = r;
            assign cv[gl+1] = v_q;
            assign cf[gl+1] = f_q;
            assign cl[gl+1] = l_q;
            assign cb[gl+1] = b_q;
        end
    endgenerate

    // the running max with m
    wire           f_v = cv[L], f_first = cf[L], f_last = cl[L], f_blast = cb[L];
    reg [MCW-1:0]  cb3;                          // the block in the run
    wire           bfirst = (cb3 == 0);
    wire           par    = cb3[0];
    reg  [8*S-1:0] m_q;                          // the ring: the completing rows at the head
    reg  signed [7:0] run;                       // the running max of the row in progress (WPR > 1)
    reg  [NRG-1:0] rp_oh;                        // the completing row group (one-hot)
    wire [8*NGR-1:0] mnv;
    wire [9*NGR-1:0] lav;
    genvar gj, gr;
    generate
        for (gj = 0; gj < NGR; gj = gj + 1) begin : g_r
            wire signed [7:0] gm   = tv[lo(L) + gj*8 +: 8];
            wire signed [7:0] mo   = m_q[gj*8 +: 8];
            wire signed [7:0] prev = f_first ? (bfirst ? 8'sh80 : mo) : run;
            wire signed [7:0] nv   = smax(prev, gm);
            assign mnv[gj*8 +: 8] = nv;
            assign lav[gj*9 +: 9] = bfirst ? 9'sd0 : ($signed({mo[7], mo}) - $signed({nv[7], nv}));
        end
        for (gr = 0; gr < S; gr = gr + 1) begin : g_w
            always @(posedge clk)
                if (f_v && f_last && rp_oh[gr / NGR]) begin
                    if (par) begin
                        mn1[gr*8 +: 8] <= mnv[(gr % NGR)*8 +: 8];
                        la1[gr*9 +: 9] <= lav[(gr % NGR)*9 +: 9];
                    end
                    else begin
                        mn0[gr*8 +: 8] <= mnv[(gr % NGR)*8 +: 8];
                        la0[gr*9 +: 9] <= lav[(gr % NGR)*9 +: 9];
                    end
                end
        end
    endgenerate
    always @(posedge clk) begin
        if (reset) begin
            cb3 <= 0; max_cnt <= 0; rp_oh <= 1; blk_done <= 1'b0;
        end
        else begin
            blk_done <= 1'b0;
            if (f_v) begin
                run <= mnv[7:0];
                if (f_last) begin
                    m_q   <= {mnv, m_q[8*S-1:8*NGR]};                  // rotate: the next rows to the head
                    rp_oh <= rp_oh[NRG-1] ? {{(NRG-1){1'b0}}, 1'b1} : rp_oh << 1;
                    if (f_blast) begin
                        cb3       <= cb3 + 1'b1;
                        max_cnt   <= max_cnt + 1'b1;
                        blk_done  <= 1'b1;
                        blk_par   <= par;
                        blk_first <= bfirst;
                    end
                end
            end
        end
    end
endmodule


// ----------------------------------------------------------------------------
// softmax_online_xfeed — EXP crossbar CX of N_EXP (spec §5 FEED_LAT, SP_LAT). Its windows are the run's windows
// g = CX, CX + N_EXP, ... (block g / NWB, window g mod NWB), in order. Window g starts at the first cycle s at which
// its block's row maxima are done and the previous window started P + 2 cycles before (the crossbar's pass period):
//   s .. s+7    its P store words are read (bank g mod NB); at s the window's rows' m_new are registered
//   s+1+c       the registered read of word c: s - m_new and the clamp at -128 on its Q_W bytes, registered
//   s+2+c       the bytes written into the window's corner-turn buffer (two buffers, alternating)
//   s+10 ..     the burst: plane b = bit b of every byte, into the crossbar (the E5 window, lp_*), LSB first
// Drain (the crossbar's P planes): per plane, the count of each row's BKV set bits (the count register), then the
// row sum += count << b (the weighted add); after plane 7 the window's row sums are offered (sp_we, the window add).
// ----------------------------------------------------------------------------
module softmax_online_xfeed #(
    parameter CX    = 0,
    parameter N_EXP = 1,
    parameter S     = 64,
    parameter BKV   = 16,
    parameter R     = 256,
    parameter C     = 256,
    parameter P     = 8,
    parameter NWB   = 4,
    parameter B     = 4,
    parameter NB    = 2,
    parameter DW    = 4,
    parameter BAW   = 5,
    parameter MCW   = 3,
    parameter SPW   = 12
) (
    input  wire                         clk,
    input  wire                         reset,          // run clear (re-arm) + power-on reset
    input  wire                         dpe_reset,      // power-on reset of the crossbar and of the pass spacing
    input  wire [MCW-1:0]               max_cnt,
    input  wire [8*S-1:0]               mn0,
    input  wire [8*S-1:0]               mn1,
    output wire [BAW-1:0]               rd_addr,
    input  wire [(NB/N_EXP)*((R < C) ? R : C)-1:0] rd_data,
    output wire                         lp_valid,
    output wire [((R < C) ? R : C)-1:0] lp_data,
    output wire                         sp_we,
    output wire                         sp_par,
    output wire [$clog2(NWB+1)-1:0]     sp_w,
    output wire [(((R < C) ? R : C) / BKV)*SPW-1:0] sp_sum
);
    localparam I     = (R < C) ? R : C;
    localparam Q_W   = I / 8;
    localparam LQ    = $clog2(Q_W);
    localparam PWR   = B * NWB;                    // windows per run
    localparam NWC   = PWR / N_EXP;                // this crossbar's windows per run
    localparam LNB   = $clog2(NB);
    localparam LNWB  = $clog2(NWB);
    localparam LB    = $clog2(BKV);
    localparam RWIN  = I / BKV;                    // rows per window
    localparam GW    = $clog2(PWR + N_EXP + 1);
    localparam NBX   = NB / N_EXP;                 // banks this crossbar reads (2 when N_EXP = 1)
    localparam PWB   = $clog2(P);
    localparam NWW   = $clog2(NWB + 1);

    // ---------------- the crossbar --------------------------------------------
    wire           x_ready, x_done, x_full;
    wire [C-1:0]   x_out;
    wire           bv;
    wire [I-1:0]   bplane;
    wire [R-1:0]   x_in = bplane;                   // rows >= I read 0
    softmax_online_dpe #(.R(R), .C(C), .P(P), .MODE(2'b10), .RU(I), .CU(I)) u_x (
        .clk(clk), .reset(dpe_reset), .data_in(x_in), .w_buf_en(bv),
        .ready(x_ready), .data_out(x_out), .done(x_done), .full(x_full));
    assign lp_valid = bv;
    assign lp_data  = bplane;

    // ---------------- the window reader ----------------------------------------
    reg            p_act;                           // windows remain in this run
    reg [GW-1:0]   p_g;                             // the next window
    reg            rd_on;
    reg [PWB-1:0]  rd_c;                            // the next word
    reg [3:0]      cool;                            // cycles until the next window may start
    wire [GW-1:0]  p_blk = p_g >> LNWB;
    wire           go_rd = p_act && !rd_on && (cool == 0) && (max_cnt > p_blk);
    wire           r_v   = go_rd || rd_on;
    wire [PWB-1:0] c_now = rd_on ? rd_c : {PWB{1'b0}};
    wire [BAW-1:0] base  = ((p_g >> LNB) & (DW - 1)) << PWB;
    assign rd_addr = base | c_now;
    always @(posedge clk) begin
        if (reset) begin
            p_act <= (NWC > 0); p_g <= CX; rd_on <= 1'b0; rd_c <= 0;
        end
        else if (r_v) begin
            if (c_now == P - 1) begin
                rd_on <= 1'b0;
                rd_c  <= 0;
                p_g   <= p_g + N_EXP;
                if (p_g + N_EXP >= PWR) p_act <= 1'b0;
            end
            else begin
                rd_on <= 1'b1;
                rd_c  <= c_now + 1'b1;
            end
        end
    end
    always @(posedge clk) begin
        if (dpe_reset)       cool <= 0;
        else if (go_rd)      cool <= P + 1;         // the next window at s + P + 2 at the earliest
        else if (cool != 0)  cool <= cool - 1'b1;
    end
    // the window's rows' m_new, registered at s
    reg  [8*RWIN-1:0] slice, slice_n;
    integer jw;
    always @(*) begin
        slice_n = 0;
        for (jw = 0; jw < NWB; jw = jw + 1)
            if ((p_g & (NWB - 1)) == jw)
                slice_n = p_blk[0] ? mn1[jw*RWIN*8 +: RWIN*8] : mn0[jw*RWIN*8 +: RWIN*8];
    end
    always @(posedge clk) if (go_rd) slice <= slice_n;

    // ---------------- s - m_new, the clamp (FEED_REG: the registered read, the registered result) -------------
    reg            a_v;
    reg [PWB-1:0]  a_c;
    reg            a_bk;                            // the bank (NBX = 2)
    always @(posedge clk) begin
        a_v  <= !reset && r_v;
        a_c  <= c_now;
        a_bk <= p_g[0];
    end
    wire [I-1:0] rword = (NBX > 1 && a_bk) ? rd_data[(NBX-1)*I +: I] : rd_data[0 +: I];
    reg  [8*Q_W-1:0] sub_w;
    reg  signed [8:0] dd;
    reg  signed [7:0] mrow;
    integer js, jr;
    always @(*) begin
        sub_w = 0;
        for (js = 0; js < Q_W; js = js + 1) begin
            mrow = 0;
            for (jr = 0; jr < RWIN; jr = jr + 1)
                if (((({{(LB+LQ){1'b0}}, a_c} << LQ) | js) >> LB) == jr) mrow = slice[jr*8 +: 8];
            dd = $signed(rword[js*8 +: 8]) - mrow;
            sub_w[js*8 +: 8] = (dd < -9'sd128) ? 8'h80 : dd[7:0];
        end
    end
    reg            b_v;
    reg [PWB-1:0]  b_c;
    reg [8*Q_W-1:0] sq;
    always @(posedge clk) begin
        b_v <= !reset && a_v;
        b_c <= a_c;
        sq  <= sub_w;
    end

    // ---------------- the corner-turn: two buffers of I bytes; the burst shifts its buffer one bit per plane --------
    reg            fb;                              // the buffer being filled
    wire [1:0]     on_w;
    wire [2*I-1:0] pl_w;
    always @(posedge clk) begin
        if (reset) fb <= 1'b0;
        else if (b_v && b_c == P - 1) fb <= !fb;
    end
    genvar gb, ge;
    generate
        for (gb = 0; gb < 2; gb = gb + 1) begin : g_ct
            reg [8*I-1:0] by;                       // byte e of the window at bits e*8 +: 8
            reg           on;                       // bursting
            reg [PWB-1:0] k;                        // the burst's plane
            wire          wr = b_v && (fb == gb);
            integer q;
            always @(posedge clk) begin
                if (dpe_reset) begin on <= 1'b0; k <= 0; end
                else if (wr && b_c == P - 1) begin on <= 1'b1; k <= 0; end
                else if (on) begin
                    if (k == P - 1) on <= 1'b0;
                    k <= k + 1'b1;
                end
                for (q = 0; q < P; q = q + 1)
                    if (wr && b_c == q) by[q*Q_W*8 +: Q_W*8] <= sq;
                if (on) by <= by >> 1;              // plane b: bit b of every byte at the byte's bit 0
            end
            for (ge = 0; ge < I; ge = ge + 1) begin : g_e
                assign pl_w[gb*I + ge] = by[ge*8];
            end
            assign on_w[gb] = on;
        end
    endgenerate
    assign bv     = |on_w;
    assign bplane = on_w[1] ? pl_w[I +: I] : pl_w[0 +: I];

    // ---------------- the drain's row sums ----------------------------------------------------------------------
    reg [PWB-1:0]  db;                              // the drain plane
    reg [GW-1:0]   dg;                              // the window being drained
    always @(posedge clk) begin
        if (reset) begin db <= 0; dg <= CX; end
        else if (x_full) begin
            db <= db + 1'b1;
            if (db == P - 1) dg <= dg + N_EXP;
        end
    end
    wire [GW-1:0] dg_blk = dg >> LNWB;
    // the count register: per row of the window, its set bits in the plane
    wire [RWIN*(LB+1)-1:0] cnt_w;
    genvar gc;
    generate
        for (gc = 0; gc < RWIN; gc = gc + 1) begin : g_cnt
            softmax_online_popcnt #(.W(BKV)) u_pc (.x(x_out[gc*BKV +: BKV]), .n(cnt_w[gc*(LB+1) +: LB+1]));
        end
    endgenerate
    reg                   c_v, c_last, c_par;
    reg [PWB-1:0]         c_b;
    reg [NWW-1:0]         c_w;
    reg [RWIN*(LB+1)-1:0] cnt;
    always @(posedge clk) begin
        c_v    <= !reset && x_full;
        c_b    <= db;
        c_last <= (db == P - 1);
        c_par  <= dg_blk[0];
        c_w    <= dg & (NWB - 1);
        cnt    <= cnt_w;
    end
    // the weighted add: sum += count << b (the sum restarts at plane 0)
    reg                 s_v, s_last, s_par;
    reg [NWW-1:0]       s_w;
    reg [RWIN*SPW-1:0]  acc;
    reg [SPW-1:0]       cx, cs;
    integer ja;
    always @(posedge clk) begin
        s_v    <= !reset && c_v;
        s_last <= c_last;
        s_par  <= c_par;
        s_w    <= c_w;
        if (c_v)
            for (ja = 0; ja < RWIN; ja = ja + 1) begin
                cx = cnt[ja*(LB+1) +: LB+1];
                case (c_b)
                    3'd0:    cs = cx;
                    3'd1:    cs = cx << 1;
                    3'd2:    cs = cx << 2;
                    3'd3:    cs = cx << 3;
                    3'd4:    cs = cx << 4;
                    3'd5:    cs = cx << 5;
                    3'd6:    cs = cx << 6;
                    default: cs = cx << 7;
                endcase
                acc[ja*SPW +: SPW] <= ((c_b == 0) ? {SPW{1'b0}} : acc[ja*SPW +: SPW]) + cs;
            end
    end
    // the window add: the window's row sums, written into the block's sp table by the top
    assign sp_we  = s_v && s_last;
    assign sp_par = s_par;
    assign sp_w   = s_w;
    assign sp_sum = acc;
endmodule


// ----------------------------------------------------------------------------
// softmax_online_wfeed — one carry crossbar (mode MODE, extent S x S): its input planes come from the carry's plane
// register (d, v); its drain planes (q, full) go to the carry's tables.
// ----------------------------------------------------------------------------
module softmax_online_wfeed #(
    parameter S   = 64,
    parameter R   = 256,
    parameter C   = 256,
    parameter P   = 8,
    parameter [1:0] MODE = 2'b10
) (
    input  wire         clk,
    input  wire         dpe_reset,
    input  wire [S-1:0] d,
    input  wire         v,
    output wire         full,
    output wire [S-1:0] q
);
    wire         ready, ddone;
    wire [C-1:0] dout;
    wire [R-1:0] din = d;                           // rows >= S read 0
    softmax_online_dpe #(.R(R), .C(C), .P(P), .MODE(MODE), .RU(S), .CU(S)) u_x (
        .clk(clk), .reset(dpe_reset), .data_in(din), .w_buf_en(v),
        .ready(ready), .data_out(dout), .done(ddone), .full(full));
    assign q = dout[S-1:0];
endmodule


// ----------------------------------------------------------------------------
// softmax_online_carry — per-block carry update, blocks in order (cb = 0..B-1), spec §5 Core:
//   decision   block cb's row sums complete, its row maxima done and block cb - 1 retired: in WAIT, or in the cycle
//              of block cb - 1's last carry-LOG drain plane (WAIT_LAT: registered)
//   b = 0      the carry-LOG input lq = min(sp >> log2 N, 127)
//   b > 0      the carry-EXP input t = clamp(log_a + log_l, -128, 127); its drain writes e plane by plane; then the
//              carry-LOG input lq = min(e + (sp >> log2 N), 127) (= min(((e << log2 N) + sp) >> log2 N, 127))
//   inputs     computed from the tables into a byte register, then the plane register (GO_LAT 2), one burst
//   LOG drain  writes log_l plane by plane; its last plane retires the block (ret_cnt, the sp slot cleared)
//   done       one cycle after the last block's retire (the core's done)
// ----------------------------------------------------------------------------
module softmax_online_carry #(
    parameter S   = 64,
    parameter B   = 4,
    parameter R   = 256,
    parameter C   = 256,
    parameter P   = 8,
    parameter SPW = 12,
    parameter MCW = 3,
    parameter LN  = 6                  // log2 of the key length N (mean shift)
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             dpe_reset,
    input  wire [MCW-1:0]   max_cnt,
    input  wire [1:0]       sp_full,
    input  wire [SPW*S-1:0] sp0,
    input  wire [SPW*S-1:0] sp1,
    input  wire [9*S-1:0]   la0,
    input  wire [9*S-1:0]   la1,
    output reg  [1:0]       sp_clr,
    output reg  [MCW-1:0]   ret_cnt,
    output wire [8*S-1:0]   ll_q,
    output reg              done
);
    localparam [1:0] ST_WAIT = 2'd0, ST_CEXP = 2'd1, ST_LOG = 2'd2, ST_END = 2'd3;
    localparam PWB = $clog2(P);

    reg [1:0]     st;
    reg [MCW-1:0] cb;
    wire          par = cb[0];
    wire [MCW-1:0] cb1 = cb + 1'b1;

    // ---- the carry crossbars, fed by one plane register --------------------
    reg  [S-1:0]  pl;                  // the input plane
    reg           pv, pe;              // a plane this cycle; it goes to the carry-EXP (else the carry-LOG)
    wire          ce_full, cl_full;
    wire [S-1:0]  ce_out, cl_out;
    softmax_online_wfeed #(.S(S), .R(R), .C(C), .P(P), .MODE(2'b10)) u_cexp (
        .clk(clk), .dpe_reset(dpe_reset), .d(pl), .v(pv && pe), .full(ce_full), .q(ce_out));
    softmax_online_wfeed #(.S(S), .R(R), .C(C), .P(P), .MODE(2'b11)) u_clog (
        .clk(clk), .dpe_reset(dpe_reset), .d(pl), .v(pv && !pe), .full(cl_full), .q(cl_out));

    // ---- the drains: e and log_l, written plane by plane (LSB first) -------
    reg [PWB-1:0] ce_b, cl_b;
    reg [8*S-1:0] e_q, l_q;
    reg           e_done;              // e is whole (the cycle after the carry-EXP's last drain plane)
    wire          ret_now = cl_full && (cl_b == P - 1);
    integer jd;
    always @(posedge clk) begin
        if (dpe_reset) begin ce_b <= 0; cl_b <= 0; end
        else begin
            if (ce_full) ce_b <= ce_b + 1'b1;
            if (cl_full) cl_b <= cl_b + 1'b1;
        end
        e_done <= !reset && ce_full && (ce_b == P - 1);
        for (jd = 0; jd < S; jd = jd + 1) begin
            if (ce_full) e_q[jd*8 +: 8] <= {ce_out[jd], e_q[jd*8 + 1 +: 7]};
            if (cl_full) l_q[jd*8 +: 8] <= {cl_out[jd], l_q[jd*8 + 1 +: 7]};
        end
    end
    assign ll_q = l_q;

    // ---- the sequencer -------------------------------------------------------
    wire ok_cb  = sp_full[cb[0]] && (max_cnt > cb);
    wire ok_cb1 = sp_full[cb1[0]] && (max_cnt > cb1);
    wire dec_w  = (st == ST_WAIT) && ok_cb;                                  // block cb decided now
    wire dec_r  = (st == ST_LOG) && ret_now && (cb != B - 1) && ok_cb1;     // block cb + 1, at cb's retire
    reg  go_q;                         // the decision registered: the `wait` cycle of block cb
    always @(posedge clk) begin
        if (reset) begin
            st <= ST_WAIT; cb <= 0; ret_cnt <= 0; sp_clr <= 2'b00; done <= 1'b0; go_q <= 1'b0;
        end
        else begin
            sp_clr <= 2'b00;
            done   <= 1'b0;
            go_q   <= dec_w || dec_r;
            case (st)
                ST_WAIT:
                    if (ok_cb) st <= (cb == 0) ? ST_LOG : ST_CEXP;
                ST_CEXP:
                    if (e_done) st <= ST_LOG;
                ST_LOG:
                    if (ret_now) begin
                        sp_clr[par] <= 1'b1;
                        ret_cnt <= ret_cnt + 1'b1;
                        cb <= cb1;
                        if (cb == B - 1) begin st <= ST_END; done <= 1'b1; end
                        else if (ok_cb1) st <= ST_CEXP;
                        else st <= ST_WAIT;
                    end
                default: ;
            endcase
        end
    end

    // ---- the carry inputs: computed from the tables, registered, then planes -------------------------------------
    wire          ld_t  = go_q && (cb != 0);                  // the carry-EXP input (b > 0) at `wait`
    wire          ld_lq = (go_q && (cb == 0)) || e_done;     // the carry-LOG input at `wait` (b = 0) or e whole
    wire [9*S-1:0]   la = par ? la1 : la0;
    wire [SPW*S-1:0] sp = par ? sp1 : sp0;
    reg  [8*S-1:0]   t_vec, lq_vec;
    reg  signed [9:0] ts;
    reg  [SPW:0]     ls;
    integer jv;
    always @(*) begin
        for (jv = 0; jv < S; jv = jv + 1) begin
            ts = $signed(la[jv*9 +: 9]) + $signed(l_q[jv*8 +: 8]);
            t_vec[jv*8 +: 8] = (ts < -10'sd128) ? 8'h80 : ((ts > 10'sd127) ? 8'h7f : ts[7:0]);
            ls = ((cb == 0) ? {(SPW+1){1'b0}} : {{(SPW-7){1'b0}}, e_q[jv*8 +: 8]}) + (sp[jv*SPW +: SPW] >> LN);
            lq_vec[jv*8 +: 8] = (ls > 127) ? 8'd127 : ls[7:0];
        end
    end
    reg [8*S-1:0] cin_q;               // the input bytes; the plane register takes their bit 0, then they shift
    reg           lg, lg_e;            // cin_q was loaded in the previous cycle (for the carry-EXP)
    reg [PWB-1:0] pk;
    integer jp;
    always @(posedge clk) begin
        if (ld_t)       cin_q <= t_vec;
        else if (ld_lq) cin_q <= lq_vec;
        else if (lg || pv) cin_q <= cin_q >> 1;
        if (lg || pv)
            for (jp = 0; jp < S; jp = jp + 1) pl[jp] <= cin_q[jp*8];
        if (reset) begin lg <= 1'b0; lg_e <= 1'b0; pv <= 1'b0; pk <= 0; end
        else begin
            lg   <= ld_t || ld_lq;
            lg_e <= ld_t;
            if (lg) begin pv <= 1'b1; pk <= 0; pe <= lg_e; end
            else if (pv) begin
                if (pk == P - 1) pv <= 1'b0;
                pk <= pk + 1'b1;
            end
        end
    end
endmodule


// ----------------------------------------------------------------------------
// softmax_online_top — the online softmax (v3/ip/softmax/spec.md).
//   E4 s_v / s_d (S_CH channels of I-bit planes) -> input stage -> the core's block stream; s_cr per tile
//   E5 lp_v / lp_d: the EXP crossbars' input bursts (window g of the run on lane g mod N_EXP); lp_cr per block
//   E6 la_v / la_d: log a of the S rows of a block b >= 1, one window (2 lanes: lane 1 carries bit 8); la_cr per block
//   E7 ll_v / ll_d: the run's final log l, one window of S rows; ll_cr per run
//   start / busy: the start port (busy falling is its credit); done one cycle after the last E7 plane
// ----------------------------------------------------------------------------
module softmax_online_top #(
    parameter S      = 64,
    parameter BKV    = 16,
    parameter N      = S,
    parameter R      = 256,
    parameter C      = 256,
    parameter P      = 8,
    parameter N_EXP  = 2,
    parameter S_CH   = 1,
    parameter LP_CAP = 2,
    parameter LA_CAP = 2,
    parameter LL_CAP = 1,
    // derived (port widths; do not override)
    parameter I      = (R < C) ? R : C
) (
    input  wire               clk,
    input  wire               reset,
    input  wire               start,
    output wire               busy,
    output reg                done,
    input  wire [S_CH-1:0]    s_v,
    input  wire [S_CH*I-1:0]  s_d,
    output wire               s_cr,
    output wire [N_EXP-1:0]   lp_v,
    output wire [N_EXP*I-1:0] lp_d,
    input  wire               lp_cr,
    output wire               la_v,
    output wire [2*S-1:0]     la_d,
    input  wire               la_cr,
    output wire               ll_v,
    output wire [S-1:0]       ll_d,
    input  wire               ll_cr
);
    localparam LS     = $clog2(S);
    localparam MCW    = $clog2(N / BKV + 2);
    localparam B      = N / BKV;
    localparam LN     = $clog2(N);
    localparam LB     = $clog2(BKV);
    localparam BLKEL  = S * BKV;
    localparam NWB    = BLKEL / I;
    localparam PW     = B * NWB;
    localparam Q_W    = I / 8;
    localparam NB     = (N_EXP < 2) ? 2 : N_EXP;
    localparam NBX    = NB / N_EXP;
    localparam DW     = (2 * NWB >= NB) ? (2 * NWB) / NB : 1;
    localparam BDEPTH = DW * P;
    localparam BAW    = $clog2(BDEPTH);
    localparam SPW    = 8 + LB;
    localparam T_W    = NWB / S_CH;
    localparam DFW    = $clog2(2 * T_W * P);
    localparam LPW    = $clog2(LP_CAP + 1);
    localparam LAW    = $clog2(LA_CAP + 1);
    localparam LLW    = $clog2(LL_CAP + 1);
    localparam RWIN   = I / BKV;                     // rows per window
    localparam LRW    = $clog2(RWIN);
    localparam NWW    = $clog2(NWB + 1);
    localparam PWB    = $clog2(P);

    // ---------------- domain guards ------------------------------------------
    generate
        if ((S & (S - 1)) != 0 || (BKV & (BKV - 1)) != 0 || (I & (I - 1)) != 0
            || (N & (N - 1)) != 0 || (N_EXP & (N_EXP - 1)) != 0 || (S_CH & (S_CH - 1)) != 0
            || BKV < 8 || BKV > N || BKV > I || S < 8 || S > I || P != 8
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
    wire         ld_valid;
    wire [I-1:0] ld_data;
    softmax_online_instage #(.S_CH(S_CH), .I(I), .P(P), .NWB(NWB), .DFW(DFW)) u_in (
        .clk(clk), .reset(reset), .s_v(s_v), .s_d(s_d), .open_ok(open_ok), .open(open), .s_cr(s_cr),
        .out_v(ld_valid), .out_d(ld_data));

    // ---------------- store + block max --------------------------------------
    wire [NB*BAW-1:0] st_raddr;
    wire [NB*I-1:0]   st_rdata;
    softmax_online_store #(.I(I), .P(P), .NB(NB), .DW(DW), .BAW(BAW)) u_store (
        .clk(clk), .reset(clr), .ld_valid(ld_valid), .ld_data(ld_data),
        .rd_addr(st_raddr), .rd_data(st_rdata));

    wire [8*S-1:0] mn0, mn1;
    wire [9*S-1:0] la0, la1;
    wire [MCW-1:0] max_cnt;
    wire           blk_done, blk_par, blk_first;
    softmax_online_loadmax #(.S(S), .BKV(BKV), .Q_W(Q_W), .MCW(MCW)) u_lmax (
        .clk(clk), .reset(clr), .ld_valid(ld_valid), .ld_data(ld_data),
        .mn0(mn0), .mn1(mn1), .la0(la0), .la1(la1), .max_cnt(max_cnt),
        .blk_done(blk_done), .blk_par(blk_par), .blk_first(blk_first));

    // ---------------- E6: a block b >= 1's log a, one window of S rows, from the cycle after its last m_new ---------
    reg            e6_v, e6_p;
    reg  [PWB:0]   e6_k;                       // the next plane
    reg  [2*S-1:0] e6_q;
    wire           e6_go = blk_done && !blk_first;
    wire [9*S-1:0] e6_t  = (e6_go ? blk_par : e6_p) ? la1 : la0;
    wire [PWB:0]   e6_b  = e6_go ? {(PWB+1){1'b0}} : e6_k;
    wire [2*S-1:0] e6_pl;                      // plane e6_b: lane 0 = bit b of every row, lane 1 = bit 8 + b
    genvar g6;
    generate
        for (g6 = 0; g6 < S; g6 = g6 + 1) begin : g_e6
            wire [8:0] v9 = e6_t[g6*9 +: 9];
            assign e6_pl[g6]     = v9[e6_b[PWB-1:0]];
            assign e6_pl[S + g6] = (e6_b == 0) ? v9[8] : 1'b0;
        end
    endgenerate
    always @(posedge clk) begin
        if (clr) begin e6_v <= 1'b0; e6_k <= 0; end
        else if (e6_go) begin e6_v <= 1'b1; e6_k <= 1; e6_p <= blk_par; e6_q <= e6_pl; end
        else if (e6_v) begin
            if (e6_k == P) e6_v <= 1'b0;
            else begin e6_q <= e6_pl; e6_k <= e6_k + 1'b1; end
        end
    end
    assign la_v = e6_v;
    assign la_d = e6_q;

    // ---------------- EXP crossbars ------------------------------------------
    wire [N_EXP*BAW-1:0]       fx_raddr;
    wire [N_EXP-1:0]           sp_we;
    wire [N_EXP-1:0]           sp_par;
    wire [N_EXP*NWW-1:0]       sp_w;
    wire [N_EXP*RWIN*SPW-1:0]  sp_sum;
    genvar gc, gb;
    generate
        for (gc = 0; gc < N_EXP; gc = gc + 1) begin : gen_exp
            wire [NBX*I-1:0] own;
            for (gb = 0; gb < NBX; gb = gb + 1) begin : g_own
                assign own[gb*I +: I] = st_rdata[(gb*N_EXP + gc)*I +: I];
            end
            softmax_online_xfeed #(.CX(gc), .N_EXP(N_EXP), .S(S), .BKV(BKV), .R(R), .C(C), .P(P), .NWB(NWB), .B(B),
                                   .NB(NB), .DW(DW), .BAW(BAW), .MCW(MCW), .SPW(SPW)) u_feed (
                .clk(clk), .reset(clr), .dpe_reset(reset), .max_cnt(max_cnt), .mn0(mn0), .mn1(mn1),
                .rd_addr(fx_raddr[gc*BAW +: BAW]), .rd_data(own),
                .lp_valid(lp_v[gc]), .lp_data(lp_d[gc*I +: I]),
                .sp_we(sp_we[gc]), .sp_par(sp_par[gc]), .sp_w(sp_w[gc*NWW +: NWW]),
                .sp_sum(sp_sum[gc*RWIN*SPW +: RWIN*SPW]));
        end
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_bank
            assign st_raddr[gb*BAW +: BAW] = fx_raddr[(gb % N_EXP)*BAW +: BAW];
        end
    endgenerate

    // ---------------- row sums (sp) per block, ping-pong by parity -----------
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
                        if (sp_we[q] && (sp_par[q] == gp) && (sp_w[q*NWW +: NWW] == (gr >> LRW)))
                            v <= sp_sum[(q*RWIN + (gr & (RWIN - 1)))*SPW +: SPW];
                assign sp_both[(gp*S + gr)*SPW +: SPW] = v;
            end
            // completed rows of the slot
            reg [LS:0] cnt;
            reg [LS:0] n;
            integer q2;
            always @(*) begin
                n = 0;
                for (q2 = 0; q2 < N_EXP; q2 = q2 + 1)
                    if (sp_we[q2] && (sp_par[q2] == gp)) n = n + RWIN;
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
    wire           core_done;
    softmax_online_carry #(.S(S), .B(B), .R(R), .C(C), .P(P), .SPW(SPW), .MCW(MCW), .LN(LN)) u_carry (
        .clk(clk), .reset(clr), .dpe_reset(reset), .max_cnt(max_cnt), .sp_full(sp_full),
        .sp0(sp0), .sp1(sp1), .la0(la0), .la1(la1),
        .sp_clr(sp_clr), .ret_cnt(ret_cnt), .ll_q(ll_q), .done(core_done));

    // ---------------- E7: the final log l, one window of S rows; done -------------------------------------------
    // Decided when the core is done (or later, when an ll credit arrives); the decision register, then the plane
    // register: plane b of every row's log l two cycles after the decision, one plane per cycle; done one cycle after
    // the last plane.
    reg           ll_pend, e7_act, e7_v;
    reg [PWB:0]   e7_k;
    reg [S-1:0]   e7_q;
    assign e7_go = (core_done || ll_pend) && !e7_act && (ll_cnt != 0);
    wire [S-1:0]  e7_pl;
    genvar g7;
    generate
        for (g7 = 0; g7 < S; g7 = g7 + 1) begin : g_e7
            wire [7:0] v8 = ll_q[g7*8 +: 8];
            assign e7_pl[g7] = v8[e7_k[PWB-1:0]];
        end
    endgenerate
    always @(posedge clk) begin
        if (clr) begin
            ll_pend <= 1'b0; e7_act <= 1'b0; e7_k <= 0; e7_v <= 1'b0; done <= 1'b0;
        end
        else begin
            done <= e7_v && !e7_act;
            if (e7_go) begin
                ll_pend <= 1'b0; e7_act <= 1'b1; e7_k <= 0;
            end
            else if (core_done) ll_pend <= 1'b1;
            if (e7_act) begin
                e7_q <= e7_pl;
                e7_v <= 1'b1;
                if (e7_k == P - 1) e7_act <= 1'b0;
                e7_k <= e7_k + 1'b1;
            end
            else e7_v <= 1'b0;
        end
    end
    assign ll_v = e7_v;
    assign ll_d = e7_q;
endmodule
