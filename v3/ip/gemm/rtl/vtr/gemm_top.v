// ============================================================================
// gemm_top.v — GEMM array (V×H dpe instances + byte-reduction tree + lane
// serializer), v2 clean-room Stage-2 skeleton.
//
// VTR SOURCE — synthesis input only, never simulated. Hand-maintained twin of
// the simulation source `v2/sim/rtl/gemm_top.v` (same modules, parameters and
// datapath); the twins differ only in: `dpe` is the arch-XML hard block
// (`<model name="dpe">`, the `wc` subckt; port contract
// `v2/experiment/rtl/dpe_blackbox.v`): no behavior model is compiled and each
// instance is bare (the model declares no parameters). Weight programming is
// removed (weights are preloaded configuration): no `weight_in`/`weight_en`
// ports, no block 1 demux, `load_input_reg` tied 0, `data_in` = the act lane.
// The V*H instance grid is still generated from (K, N, R, C). Consistency gate:
// `v2/experiment/rtl/check_split.py`.
//
// Ground truth: `v2/sim/spec/gemm.md` (**v0.3 FROZEN 2026-09-17**).
// Hand-written from the charter; no MAC/ACAM logic is re-implemented here —
// this module ONLY composes the certified primitive (`dpe`, spec v2.0.1):
//
//     Y[M,N] = X[M,K] * W[K,N]
//     V = ceil(K/R) K-tiles, H = ceil(N/C) N-tiles
//     per tile (v,h): REGULAR ACAM -> out8_v,  S = sum_v sign_extend(out8_v)
//     array output   : low byte of S (serializer; NO ACAM after reduction)
//
// Interface (charter §4, streaming):
//   act_in[V*BUF-1:0]  V lanes; lane v carries tile-row v's ACT word (§4.2);
//                      all H tiles of a v-row receive the same lane data
//   act_en             ACT strobe (accepted only while MSB_SA_Ready high)
//   MSB_SA_Ready       array refill permit = AND of all tiles' ready
//   data_out[H*BUF-1:0] H output lanes, 5 bytes/cycle each; lane h carries
//                      columns h*C .. h*C+C-1 (§4.4; padding bytes are 0)
//   dpe_done           1-cycle pulse after the last lane word of a pass
//   reg_full           output busy (observability; A1)
//
// TB probe contract (verification-only internals, D8 — NOT ports; names are
// frozen so `v2/sim/tb/tb_gemm_top.v` can read them):
//   S_col[0:H*C-1] : reg signed [31:0] — wide reduced partial per column
//                    (G5); complete at `dpe_done`, stable until the next
//                    pass's first reduced word. Wide half of the dual compare.
//   out_valid      : reg — 1-cycle pulse per drained word; every lane's
//                    `lane_q` word is valid in that cycle.
//
// Primitive protocol note: `dpe` exposes no per-word drain-valid port (its
// `drain_valid` is an internal TB probe), but it exposes the drain as signals:
// `reg_full` is high for exactly the drain-word cycles (one word per cycle,
// gapless) and `dpe_done` pulses the cycle after the last word (A7). The
// wrapper fires everything from those handshakes — `word_en` is the reg_full
// window, `pass_end` is the dpe_done pulse — and pushes both through the same
// marker pipeline as the data (TREE_PIPE tree stages + serializer = L_w
// registers, charter §5.2). T_steady is untouched.
//
// Block map (all implemented; bring-up 1A -> 1B -> 1C -> 1D done, GATE 2
// green with delta_impl = 0 on every geometry):
//   1 weight demux: simulation source only
//   2 readiness: MSB_SA_Ready = AND of all V*H tile ready outs    §4.4 / I3
//   3 drain phase: reg_full/dpe_done handshakes -> marker pipe    §5.2 / A7
//   4 reduction tree: sign-extend V tile bytes, TREE_PIPE folds    G5
//   5 serializer + lanes: low byte of S -> lane_q, S_col, out_valid §4.4 / A4
//   6 pass end: dpe_done pulse, reg_full busy window               A7
//
// Cycles: measured = T_fill_array + (M-1)*T_steady + Δ_impl (charter §5.3);
// COMPUTE_CYC = P+2 must emerge structurally (P10) — no hold-counters.
// The cycle contract (LOAD_CYC, COMPUTE_CYC, OUTPUT_CYC, L_w, WR_CYC) is
// owned by tb_gemm_top.v / run_gemm_rtl.py / case.json; RTL logic must never
// gate on it.
// ============================================================================

`timescale 1ns / 1ps

module gemm_top #(
    parameter K   = 128,   // reduction length
    parameter N   = 128,   // output width
    parameter R   = 256,   // dpe rows   (crossbar)
    parameter C   = 256,   // dpe cols   (crossbar)
    parameter BUF = 40,    // port width (5 bytes/cycle)
    parameter P   = 8,     // activation bit precision
    // P29 pass extent (dpe_nldpe.md v2.0.3), passed to every tile; default =
    // the certified full extent. Allowed: ROWS_USED = R, or K <= ROWS_USED <= R
    // when V == 1; COLS_USED = C, or N <= COLS_USED <= C when H == 1 (the
    // lock-step tiles of a split dimension stay full).
    parameter ROWS_USED = R,
    parameter COLS_USED = C
) (
    input  wire clk,
    input  wire reset,
    input  wire [((K + R - 1) / R) * BUF - 1:0] act_in,
    input  wire act_en,
    output wire  MSB_SA_Ready,
    output wire [(((N + C - 1) / C) * BUF) - 1:0] data_out,
    output reg  dpe_done,
    output reg  reg_full,
    output reg  out_valid           // word valid of data_out (v3: a port; it was a TB probe)
);

    localparam V   = (K + R - 1) / R;
    localparam H   = (N + C - 1) / C;
    localparam EPS = BUF / 8;                        // bytes per word (5)
    function integer clog2_fn;
        input integer val;
        integer i;
        begin
            clog2_fn = 0;
            for (i = val - 1; i > 0; i = i >> 1) clog2_fn = clog2_fn + 1;
        end
    endfunction
    localparam TREE_PIPE = (V > 1) ? clog2_fn(V) : 0;   // G5: registered fold depth
    localparam NTILES    = V * H;

    // P29 extent guards (elaboration error outside the allowed extents)
    generate
        if (ROWS_USED < 1 || ROWS_USED > R || COLS_USED < 1 || COLS_USED > C
            || (V > 1 && ROWS_USED != R) || (V == 1 && ROWS_USED < K)
            || (H > 1 && COLS_USED != C) || (H == 1 && COLS_USED < N))
        begin : g_bad_extent
            gemm_top_pass_extent_outside_domain u_bad ();
        end
    endgenerate

    // ------------------------------------------------------------------
    // dpe grid (V*H hard-block instances; all REGULAR — v0.3 has no mode
    // port). act lane v is broadcast to all H tiles of row v; weights are
    // preloaded (load_input_reg tied 0); readiness/drains come back per tile.
    // The P29 extent (ROWS_USED / COLS_USED) is the hard block's configuration:
    // the bare `dpe` has no parameters, so the twin carries them for parity only.
    // ------------------------------------------------------------------
    wire [BUF-1:0] tile_dout   [0:NTILES-1];
    wire [NTILES-1:0] tile_ready;
    wire [NTILES-1:0] tile_done;
    wire [NTILES-1:0] tile_regfull;
    wire [NTILES-1:0] tile_sad;      // shift_add_done (unused; observability)
    wire [NTILES-1:0] tile_sabc;     // shift_add_bypass_ctrl (unused, drives 0)

    genvar gv, gh;
    generate
        for (gv = 0; gv < V; gv = gv + 1) begin : gen_v
            for (gh = 0; gh < H; gh = gh + 1) begin : gen_h
                dpe u_dpe (
                    .clk(clk),
                    .reset(reset),
                    .data_in(act_in[gv * BUF +: BUF]),
                    .nl_dpe_control(2'b00),          // REGULAR (v0.3)
                    .shift_add_control(1'b0),
                    .w_buf_en(act_en),
                    .shift_add_bypass(1'b0),
                    .load_output_reg(1'b0),
                    .load_input_reg(1'b0),           // weights preloaded
                    .MSB_SA_Ready(tile_ready[gv*H + gh]),
                    .data_out(tile_dout[gv*H + gh]),
                    .dpe_done(tile_done[gv*H + gh]),
                    .reg_full(tile_regfull[gv*H + gh]),
                    .shift_add_done(tile_sad[gv*H + gh]),
                    .shift_add_bypass_ctrl(tile_sabc[gv*H + gh])
                );
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // 2 — readiness (§4.4 / I3)
    //   MSB_SA_Ready = AND of every tile's MSB_SA_Ready (lockstep tiles, so
    //   they are aligned; the AND is the contract).
    // ------------------------------------------------------------------

    assign MSB_SA_Ready = &tile_ready;

    // ------------------------------------------------------------------
    // 3 — drain phase, signal-fired (§5.2 / A7)
    //   No cycle constants: the primitive exposes the drain as signals.
    //     word_en  = &tile_regfull   // level: one drained word per cycle
    //     pass_end = &tile_done      // 1-cycle handshake, the cycle after
    //                                // the last drain word (A7)
    //   Push word_en and pass_end through the SAME marker pipeline as the
    //   data path (TREE_PIPE tree stages + serializer = L_w stages). The
    //   delayed pass_end IS the array `dpe_done` output: one cycle after the
    //   last lane word, for any V — no OCYC/L_w arithmetic in the DUT.
    //   Column index is address bookkeeping only (never gates execution):
    //     idx <= 0 on pass_end ;  idx <= idx + 1 on word_en
    // ------------------------------------------------------------------
    wire word_en  = &tile_regfull;   // common drain-word window (lockstep)
    wire pass_end = &tile_done;      // end-of-drain handshake (A7), 1 cycle

    // marker pipeline: same depth as the data path (TREE_PIPE tree stages +
    // serializer = L_w registers). valid_d[i] is word_en delayed (i+1).
    reg [TREE_PIPE:0] valid_d, end_d;
    integer pi;
    always @(posedge clk) begin
        if (reset) begin
            valid_d <= 0;
            end_d   <= 0;
        end
        else begin
            valid_d[0] <= word_en;
            end_d[0]   <= pass_end;
            for (pi = 1; pi <= TREE_PIPE; pi = pi + 1) begin
                valid_d[pi] <= valid_d[pi-1];
                end_d[pi]   <= end_d[pi-1];
            end
        end
    end

    // stage aligned with the serializer input (delay TREE_PIPE); V=1 has no
    // tree, so the serializer consumes the leaves directly.
    wire serializer_valid, serializer_end;
    generate
        if (TREE_PIPE == 0) begin : gen_sv0
            assign serializer_valid = word_en;
            assign serializer_end   = pass_end;
        end else begin : gen_svn
            assign serializer_valid = valid_d[TREE_PIPE-1];
            assign serializer_end   = end_d[TREE_PIPE-1];
        end
    endgenerate

    // ------------------------------------------------------------------
    // 4 — reduction tree (G5): per output element (lane h, word byte b) there
    //   is ONE V-way reduction; the H*EPS elements of a drained word reduce
    //   in parallel, and TREE_PIPE registered pairwise stages fold the V
    //   leaves (V=1 has no tree). Layout is group-major:
    //     g = h*EPS + b   (which output element),   slot = g*V + v
    //   so each group's V leaves are contiguous and slot g+0 of the last
    //   stage holds that element's total. Arithmetic is exact integer, so
    //   the pairing order is immaterial (charter T5/P22); the tree depth is
    //   what sets the fill latency L_w = TREE_PIPE + 1.
    //   Bit-width: |S| <= 128*V, so 32-bit signed per column (B8).
    // ------------------------------------------------------------------
    localparam NGROUP = H * EPS;                 // output elements per word
    localparam NSLOT  = NGROUP * V;              // leaves per word
    reg signed [31:0] leaf [0:NSLOT-1];          // stage 0 (combinational)
    integer lh, lb, lv;
    always @(*) begin
        for (lh = 0; lh < H; lh = lh + 1)
            for (lb = 0; lb < EPS; lb = lb + 1)
                for (lv = 0; lv < V; lv = lv + 1)
                    leaf[(lh*EPS + lb)*V + lv] =
                        $signed(tile_dout[lv*H + lh][lb*8 +: 8]);
    end

    // pairwise fold within each group:
    //   tree[s][g+v] = tree[s-1][g+2v] + tree[s-1][g+2v+1]
    // odd leaf passes through; unused slots are zero. The tree is one wide
    // register, stage s slot k at tree[(s*NSLOT + k)*32 +: 32] (stage 0 is
    // unused), each index a function of the loop counters only: as an
    // unpacked array, synthesis collected it into one memory with a port per
    // access (v3 DSE elaboration check, 2026-10-15).
    reg [(TREE_PIPE+1)*NSLOT*32-1:0] tree;
    integer fs, fv, fh, fb;
    generate
        if (V > 1) begin : gen_tree
            always @(posedge clk) begin
                for (fs = 1; fs <= TREE_PIPE; fs = fs + 1)
                    for (fh = 0; fh < H; fh = fh + 1)
                        for (fb = 0; fb < EPS; fb = fb + 1)
                            for (fv = 0; fv < V; fv = fv + 1) begin
                                if (2*fv + 1 < V)
                                    tree[(fs*NSLOT + (fh*EPS + fb)*V + fv)*32 +: 32] <=
                                        ((fs == 1) ? leaf[(fh*EPS + fb)*V + 2*fv]
                                                   : $signed(tree[((fs-1)*NSLOT + (fh*EPS + fb)*V + 2*fv)*32 +: 32]))
                                      + ((fs == 1) ? leaf[(fh*EPS + fb)*V + 2*fv + 1]
                                                   : $signed(tree[((fs-1)*NSLOT + (fh*EPS + fb)*V + 2*fv + 1)*32 +: 32]));
                                else if (2*fv < V)
                                    tree[(fs*NSLOT + (fh*EPS + fb)*V + fv)*32 +: 32] <=
                                        (fs == 1) ? leaf[(fh*EPS + fb)*V + 2*fv]
                                                  : $signed(tree[((fs-1)*NSLOT + (fh*EPS + fb)*V + 2*fv)*32 +: 32]);
                                else
                                    tree[(fs*NSLOT + (fh*EPS + fb)*V + fv)*32 +: 32] <= 32'sd0;
                            end
            end
        end
    endgenerate

    // serializer source: the last tree stage (V>1) or the leaves (V=1);
    // element (h,b)'s total lives at slot (h*EPS + b)*V + 0
    reg signed [31:0] red [0:NSLOT-1];
    integer ri;
    generate
        if (V == 1) begin : gen_red1
            always @(*) for (ri = 0; ri < NSLOT; ri = ri + 1) red[ri] = leaf[ri];
        end else begin : gen_redn
            always @(*) for (ri = 0; ri < NSLOT; ri = ri + 1)
                red[ri] = $signed(tree[(TREE_PIPE*NSLOT + ri)*32 +: 32]);
        end
    endgenerate

    // ------------------------------------------------------------------
    // 5 — serializer + lanes (§4.4 / A4)
    //   For each lane h and column c, `S_col[h*C + c]` gets the wide sum;
    //   the emitted byte is its LOW BYTE (trunc8 semantics; NO ACAM after
    //   reduction, charter v0.3). The low bytes are registered into
    //   `lane_q[h*BUF +: BUF]` (byte i in bits 8i+7:8i), `data_out` is
    //   driven from `lane_q`, and `out_valid` pulses in the same cycle.
    //   L_w = TREE_PIPE + 1 registers in total.
    //   Padding columns (n >= N) and absent K-rows are zero by construction
    //   (zero weights/activations, A5).
    // ------------------------------------------------------------------
    reg signed [31:0] S_col [0:H*C-1];   // TB probe (do not rename)
    // out_valid: the word valid of data_out (a port since v3; do not rename: the v2 TB reads dut.out_valid)
    reg [H*BUF-1:0] lane_q;

    assign data_out = lane_q;             // serializer output (combinational)

    // column index: address bookkeeping only, cleared by the end handshake
    integer s_idx;
    always @(posedge clk) begin
        if (reset)                 s_idx <= 0;
        else if (serializer_end)   s_idx <= 0;
        else if (serializer_valid) s_idx <= s_idx + 1;
    end

    integer sh, sb;
    always @(posedge clk) begin
        if (reset) begin
            out_valid <= 1'b0;
        end
        else begin
            out_valid <= serializer_valid;   // serializer output valid
            if (serializer_valid) begin
                for (sh = 0; sh < H; sh = sh + 1) begin
                    for (sb = 0; sb < EPS; sb = sb + 1) begin
                        lane_q[sh*BUF + sb*8 +: 8] <= red[(sh*EPS + sb)*V][7:0];
                        // guard: the last word's padding bytes must not spill
                        // into the next lane's column range
                        if (s_idx*EPS + sb < C)
                            S_col[sh*C + s_idx*EPS + sb] <= red[(sh*EPS + sb)*V];
                    end
                end
            end
            else begin
                lane_q <= {H*BUF{1'b0}};     // idle lanes read zero (§4.4)
            end
        end
    end

    // ------------------------------------------------------------------
    // 6 — pass end + busy flag (A7)
    //   `dpe_done` = 1-cycle pulse after the last lane word of a pass
    //   (`serializer_end`, registered); `reg_full` is high from the first
    //   output word until the pass end (observability; A1). Both are fired
    //   by the drain handshakes only — no cycle constants.
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            dpe_done <= 1'b0;
            reg_full <= 1'b0;
        end
        else begin
            dpe_done <= serializer_end;      // one cycle after the last word
            if (out_valid) reg_full <= 1'b1;
            if (dpe_done)  reg_full <= 1'b0;
        end
    end

endmodule
