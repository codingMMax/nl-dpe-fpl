// ============================================================================
// gemm_top.v — GEMM: V x H `dpe` tiles in lock-step, a bit-serial mod-256 tree over the V row tiles of each column,
// and an output register. Simulation twin.
//
// Contract: v3/ip/gemm/spec.md (the bit-plane pass, 2026-10-22/23). Rewritten 2026-10-23 from the v2-derived module
// (git show 8acf73c:v3/ip/gemm/rtl/sim/gemm_top.v): the values are v2's (Y = the low byte of the sum of the V tiles'
// REGULAR bytes), the interface and the timing are the bit-plane ones.
//
//     Y[M,N] = trunc8(X[M,K] W[K,N]);  V = ceil(K/R) row tiles, H = ceil(N/C) column tiles
//
// Ports (spec §3):
//   act_in [V*ROWS_USED]   a row's burst: 8 plane words, LSB first, gapless, started in a cycle with MSB_SA_Ready;
//                          lane v = rows v*R .. v*R + ROWS_USED - 1 of the row (each tile's rows >= ROWS_USED are 0)
//   act_en                 the plane strobe, high in each of the burst's 8 cycles
//   weight_in, weight_en   one int8 weight per strobe, simulation twin only (the testbenches preload instead, R8)
//   MSB_SA_Ready           the AND of the tiles' ready (the burst port's credit)
//   data_out [H*COLS_USED] lane h = columns h*C .. of the row's result, 8 plane words; out_valid is the word valid
//   dpe_done               a pulse one cycle after the row's last output plane;  reg_full: a busy flag
//
// Timing (spec §5): output plane b of a row at drain_start + L_W + b, L_W = TREE_PIPE + 1 (the tree's registered
// levels, then the output register); dpe_done one cycle after plane 7. Back to back, a row every 10 cycles.
//
// The tree: per column, ceil(log2 V) registered levels of bit-serial full adders. A node adds its two input planes
// with its carry flip-flop; the carry is cleared at plane 0, and the carry out of plane 7 is dropped (mod 256). An odd
// input passes through, registered. The plane-0 marker travels with the planes, one register per level.
//
// The probe names `gen_v[v].gen_h[h].u_dpe` (weights, mode_q) are what the testbenches preload; keep them.
// ============================================================================

`timescale 1ns / 1ps

module gemm_top #(
    parameter K   = 128,   // reduction length
    parameter N   = 128,   // output width
    parameter R   = 256,   // dpe rows
    parameter C   = 256,   // dpe columns
    parameter P   = 8,     // bits per element = plane words per window
    // P29 pass extent: ROWS_USED = R, or K <= ROWS_USED <= R when V == 1; COLS_USED = C, or N <= COLS_USED <= C when
    // H == 1
    parameter ROWS_USED = R,
    parameter COLS_USED = C
) (
    input  wire clk,
    input  wire reset,
    input  wire [((K + R - 1) / R) * ROWS_USED - 1:0] act_in,
    input  wire act_en,
    input  wire [7:0] weight_in,
    input  wire weight_en,
    output wire MSB_SA_Ready,
    output wire [((N + C - 1) / C) * COLS_USED - 1:0] data_out,
    output reg  dpe_done,
    output reg  reg_full,
    output reg  out_valid
);

    localparam V  = (K + R - 1) / R;
    localparam H  = (N + C - 1) / C;
    localparam RU = ROWS_USED;
    localparam CU = COLS_USED;
    function integer clog2_fn;
        input integer val;
        integer i;
        begin
            clog2_fn = 0;
            for (i = val - 1; i > 0; i = i >> 1) clog2_fn = clog2_fn + 1;
        end
    endfunction
    localparam TREE_PIPE = (V > 1) ? clog2_fn(V) : 0;   // registered tree levels
    localparam NTILES    = V * H;
    localparam PW        = clog2_fn(P);                 // plane counter width

    // P29 extent guards (an elaboration error outside the allowed extents)
    generate
        if (ROWS_USED < 1 || ROWS_USED > R || COLS_USED < 1 || COLS_USED > C
            || (V > 1 && ROWS_USED != R) || (V == 1 && ROWS_USED < K)
            || (H > 1 && COLS_USED != C) || (H == 1 && COLS_USED < N))
        begin : g_bad_extent
            gemm_top_pass_extent_outside_domain u_bad ();
        end
    endgenerate

    // ------------------------------------------------------------------
    // 1 — weight demux (simulation twin only): one int8 weight per strobe; order v-major, then h, then row-major
    // ------------------------------------------------------------------
    wire [NTILES-1:0] tile_workload;
    integer wr_tile, wr_cnt;
    always @(posedge clk) begin
        if (reset) begin
            wr_tile <= 0;
            wr_cnt  <= 0;
        end else if (weight_en) begin
            if (wr_cnt == R * C - 1) begin
                wr_cnt  <= 0;
                wr_tile <= (wr_tile == NTILES - 1) ? 0 : wr_tile + 1;
            end else begin
                wr_cnt <= wr_cnt + 1;
            end
        end
    end

    genvar t;
    generate
        for (t = 0; t < NTILES; t = t + 1) begin : gen_wl
            assign tile_workload[t] = weight_en && (wr_tile == t);
        end
    endgenerate

    // ------------------------------------------------------------------
    // 2 — the tiles: lane v's planes go straight into the H tiles of row v
    // ------------------------------------------------------------------
    wire [C-1:0]      tile_dout [0:NTILES-1];
    wire [NTILES-1:0] tile_ready;
    wire [NTILES-1:0] tile_done;
    wire [NTILES-1:0] tile_regfull;
    wire [NTILES-1:0] tile_sad;
    wire [NTILES-1:0] tile_sabc;

    genvar gv, gh;
    generate
        for (gv = 0; gv < V; gv = gv + 1) begin : gen_v
            for (gh = 0; gh < H; gh = gh + 1) begin : gen_h
                wire [R-1:0] dpe_data_in = tile_workload[gv*H + gh] ? {{(R-8){1'b0}}, weight_in}
                                                                   : act_in[gv*RU +: RU];   // rows >= RU read 0
                dpe #(
                    .KERNEL_WIDTH(R),
                    .NUM_COLS(C),
                    .PRECISION(P),
                    .ROWS_USED(ROWS_USED),
                    .COLS_USED(COLS_USED)
                ) u_dpe (
                    .clk(clk),
                    .reset(reset),
                    .data_in(dpe_data_in),
                    .nl_dpe_control(2'b00),          // REGULAR
                    .shift_add_control(1'b0),
                    .w_buf_en(act_en),
                    .shift_add_bypass(1'b0),
                    .load_output_reg(1'b0),
                    .load_input_reg(tile_workload[gv*H + gh]),
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

    assign MSB_SA_Ready = &tile_ready;

    // ------------------------------------------------------------------
    // 3 — the drain markers: the plane valid, plane 0 and the pass end, delayed one register per tree level
    // ------------------------------------------------------------------
    wire plane_v  = &tile_regfull;       // the tiles drain one plane per cycle, in lock-step
    wire pass_end = &tile_done;          // the cycle after the last drain plane
    reg  [PW-1:0] pcnt;                  // the drain plane's index
    always @(posedge clk) begin
        if (reset)        pcnt <= 0;
        else if (plane_v) pcnt <= pcnt + 1'b1;
    end
    wire plane_0 = plane_v && (pcnt == 0);

    wire [TREE_PIPE:0] valid_d, first_d, end_d;    // [s]: the markers at the output of tree level s (0: the tiles)
    assign valid_d[0] = plane_v;
    assign first_d[0] = plane_0;
    assign end_d[0]   = pass_end;
    genvar ms;
    generate
        for (ms = 1; ms <= TREE_PIPE; ms = ms + 1) begin : gen_mark
            reg v_q, f_q, e_q;
            always @(posedge clk) begin
                v_q <= reset ? 1'b0 : valid_d[ms-1];
                f_q <= reset ? 1'b0 : first_d[ms-1];
                e_q <= reset ? 1'b0 : end_d[ms-1];
            end
            assign valid_d[ms] = v_q;
            assign first_d[ms] = f_q;
            assign end_d[ms]   = e_q;
        end
    endgenerate

    // ------------------------------------------------------------------
    // 4 — the bit-serial mod-256 tree and the output register, per lane h: level s node k adds level s-1 nodes 2k and
    //     2k+1; tsum holds every level's node planes, level s node k at bits ((s*V + k)*CU) +: CU (level 0 = the tiles)
    // ------------------------------------------------------------------
    genvar th, ts, tk;
    generate
        for (th = 0; th < H; th = th + 1) begin : gen_lane
            wire [(TREE_PIPE+1)*V*CU-1:0] tsum;
            for (tk = 0; tk < V; tk = tk + 1) begin : gen_leaf
                assign tsum[tk*CU +: CU] = tile_dout[tk*H + th][CU-1:0];
            end
            for (ts = 1; ts <= TREE_PIPE; ts = ts + 1) begin : gen_lvl
                localparam integer NIN  = (V + (1 << (ts-1)) - 1) >> (ts-1);   // nodes at level ts-1
                localparam integer NOUT = (NIN + 1) >> 1;                       // nodes at level ts
                for (tk = 0; tk < NOUT; tk = tk + 1) begin : gen_node
                    reg [CU-1:0] s_q, c_q;               // the node's sum plane and its carries
                    wire [CU-1:0] a = tsum[((ts-1)*V + 2*tk)*CU +: CU];
                    if (2*tk + 1 < NIN) begin : gen_add
                        wire [CU-1:0] b   = tsum[((ts-1)*V + 2*tk + 1)*CU +: CU];
                        wire [CU-1:0] cin = first_d[ts-1] ? {CU{1'b0}} : c_q;   // cleared at plane 0
                        always @(posedge clk) begin
                            s_q <= a ^ b ^ cin;
                            c_q <= (a & b) | (a & cin) | (b & cin);
                        end
                    end else begin : gen_pass
                        always @(posedge clk) begin
                            s_q <= a;
                            c_q <= {CU{1'b0}};
                        end
                    end
                    assign tsum[(ts*V + tk)*CU +: CU] = s_q;
                end
                for (tk = NOUT; tk < V; tk = tk + 1) begin : gen_unused
                    assign tsum[(ts*V + tk)*CU +: CU] = {CU{1'b0}};
                end
            end
            // the output register: the root's plane (level TREE_PIPE, node 0); idle lanes read zero
            reg [CU-1:0] oq;
            always @(posedge clk) begin
                if (reset) oq <= {CU{1'b0}};
                else       oq <= valid_d[TREE_PIPE] ? tsum[(TREE_PIPE*V)*CU +: CU] : {CU{1'b0}};
            end
            assign data_out[th*CU +: CU] = oq;
        end
    endgenerate

    // ------------------------------------------------------------------
    // 5 — the output valid (the output register itself is in each lane's block above)
    // ------------------------------------------------------------------
    always @(posedge clk) out_valid <= reset ? 1'b0 : valid_d[TREE_PIPE];

    // ------------------------------------------------------------------
    // 6 — the pass end: dpe_done one cycle after the last output plane; reg_full from the first plane to it
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            dpe_done <= 1'b0;
            reg_full <= 1'b0;
        end else begin
            dpe_done <= end_d[TREE_PIPE];
            if (out_valid) reg_full <= 1'b1;
            if (dpe_done)  reg_full <= 1'b0;
        end
    end

endmodule
