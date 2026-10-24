// ============================================================================
// dpe_nldpe.v — NL-DPE crossbar primitive, behavior model (simulation twin)
//
// Contract: v3/ip/xbar/spec.md (the row-parallel bit-plane interface, user 2026-10-21/22). Forked 2026-10-23 from
// v2's clean-room model (git show v2-final:v2/sim/rtl/dpe_nldpe.v; its sha256 is in tools/copies.lock and the
// LEDGER): the numeric contract is v2's, the interface and the timing are new.
//
// Ports (spec §3):
//   data_in [R]   during a burst, bit r = bit b of x_r, plane b in the burst's cycle b (LSB first, gapless)
//   w_buf_en      the plane strobe: high in each of a burst's P cycles; the first only with MSB_SA_Ready high
//   data_out [C]  during the drain, bit c = bit b of output byte c, plane b in the drain's cycle b
//   reg_full      high in each of the drain's P cycles (the drain's word valid)
//   MSB_SA_Ready  high from reset and during a burst; low from load_start + P to the ACAM; high again from ACAM + 1
//   dpe_done      a pulse one cycle after the drain's last plane;  shift_add_done: one cycle after the MSB fire
//   load_input_reg + data_in[7:0]: the weight strobe (one int8 weight per cycle, row-major), simulation twin only;
//   nl_dpe_control is latched into mode_q on the weight strobes. The testbenches preload `weights` and `mode_q`
//   through the hierarchy instead (PLAN R8); those two names are frozen.
//
// Pass (spec §4), for a burst whose plane 0 arrives at ls:
//   planes ls .. ls+7; plane b fires at ls+1+b (acc += s_b << b, the MSB subtracted); msb_fire = ls+8;
//   ACAM at ls+9 (obuf <= the mode form of acc); drain planes ls+10 .. ls+17; dpe_done at ls+18;
//   MSB_SA_Ready low at ls+8 and ls+9, high from ls+10: back to back, a pass every P + 2 = 10 cycles.
// The output register never binds under the burst rule (the next ACAM is at least 10 cycles later); a drain still
// running at an ACAM sets `err`.
//
// Numeric contract (v2): int8 weights and activations, an exact integer MAC into a 32-bit accumulator, the integer
// ACAM forms (REGULAR, ACTIVATION, EXP = 1 + y + floor(y^2/2) with the int32 clamp, LOG = y - 1), then trunc8.
// P29 extent: rows >= ROWS_USED read as zero; columns >= COLS_USED drain as zero.
//
// `err` (simulation only, sticky): a strobe that starts a burst while MSB_SA_Ready is low, a gap inside a burst, or a
// drain still running at an ACAM. The testbenches check it.
// ============================================================================

`timescale 1ns / 1ps

module dpe #(
    parameter KERNEL_WIDTH   = 256,   // R
    parameter NUM_COLS       = 256,   // C
    parameter PRECISION      = 8,     // P
    parameter ROWS_USED      = KERNEL_WIDTH,
    parameter COLS_USED      = NUM_COLS
)(
    input  wire                       clk,
    input  wire                       reset,
    input  wire [KERNEL_WIDTH-1:0]    data_in,
    input  wire [1:0]                 nl_dpe_control,
    input  wire                       shift_add_control,
    input  wire                       w_buf_en,
    input  wire                       shift_add_bypass,
    input  wire                       load_output_reg,
    input  wire                       load_input_reg,
    output reg                        MSB_SA_Ready,
    output reg  [NUM_COLS-1:0]        data_out,
    output reg                        dpe_done,
    output reg                        reg_full,
    output reg                        shift_add_done,
    output reg                        shift_add_bypass_ctrl
);

    localparam R  = KERNEL_WIDTH;
    localparam C  = NUM_COLS;
    localparam P  = PRECISION;
    localparam RU = ROWS_USED;
    localparam CU = COLS_USED;
    localparam BW = $clog2(P);                          // plane index width

    // extent guard: 1 <= ROWS_USED <= R, 1 <= COLS_USED <= C
    initial begin
        if (RU < 1 || RU > R || CU < 1 || CU > C) begin
            $display("ERROR dpe: pass extent ROWS_USED=%0d COLS_USED=%0d outside 1..%0d / 1..%0d", RU, CU, R, C);
            $finish;
        end
    end

    // ------------------------------------------------------------------
    // 1 — weights + mode_q (the weight strobe; simulation twin only)
    // ------------------------------------------------------------------
    reg signed [P-1:0] weights [0:R*C-1];
    reg [1:0] mode_q;
    integer wr_ptr;
    always @(posedge clk) begin
        if (reset) begin
            wr_ptr <= 0;
            mode_q <= 0;
        end else if (load_input_reg) begin
            weights[wr_ptr] <= data_in[P-1:0];
            wr_ptr <= (wr_ptr == R * C - 1) ? 0 : wr_ptr + 1;
            mode_q <= nl_dpe_control;
        end
    end

    // ------------------------------------------------------------------
    // 2 — the burst: one plane register; plane b fires the cycle after it arrives
    // ------------------------------------------------------------------
    reg [R-1:0]  plane_q;                // the plane that fires this cycle
    reg          fire_v;                 // plane_q fires this cycle
    reg [BW-1:0] fire_b;                 // its bit index
    reg [BW:0]   ld_cnt;                 // planes received in the current burst (0 = no burst in progress)
    reg          err;                    // sticky producer / model error (simulation only)
    reg          acam_fire;              // the ACAM: one cycle after the MSB fire
    reg          draining;               // the drain of the output register is running
    integer      r1;

    always @(posedge clk) begin
        if (reset) begin
            ld_cnt <= 0;
            fire_v <= 1'b0;
            fire_b <= 0;
            err    <= 1'b0;
        end else begin
            fire_v <= w_buf_en;
            if (w_buf_en) begin
                if (ld_cnt == 0 && !MSB_SA_Ready) err <= 1'b1;      // a burst started while not ready
                for (r1 = 0; r1 < R; r1 = r1 + 1)
                    plane_q[r1] <= (r1 < RU) ? data_in[r1] : 1'b0;  // rows >= ROWS_USED read as zero
                fire_b <= ld_cnt[BW-1:0];
                ld_cnt <= (ld_cnt == P - 1) ? 0 : ld_cnt + 1;
            end else if (ld_cnt != 0) begin
                err <= 1'b1;                                         // a gap inside a burst
            end
            if (acam_fire && draining) err <= 1'b1;                  // the output register would bind
        end
    end

    wire fire_msb = fire_v & (fire_b == P - 1);

    // ------------------------------------------------------------------
    // 3 — the shift-add accumulator: s_b[c] = sum_r plane_b[r] ? W[r][c] : 0; plane 0 loads, the MSB subtracts
    // ------------------------------------------------------------------
    reg signed [31:0] acc [0:C-1];
    integer c2, r2, s_local;
    always @(posedge clk) begin
        if (fire_v) begin
            for (c2 = 0; c2 < C; c2 = c2 + 1) begin
                s_local = 0;
                for (r2 = 0; r2 < RU; r2 = r2 + 1)
                    if (plane_q[r2]) s_local = s_local + weights[r2 * C + c2];
                if (fire_b == 0)          acc[c2] <= s_local;
                else if (fire_b == P - 1) acc[c2] <= acc[c2] - (s_local << (P - 1));
                else                      acc[c2] <= acc[c2] + (s_local << fire_b);
            end
        end
    end

    // ------------------------------------------------------------------
    // 4 — the ACAM (the cycle after the MSB fire): the mode form, then trunc8, into the output register
    // ------------------------------------------------------------------
    localparam [1:0] STD = 0, ACT = 1, EXP = 2, LOG = 3;
    reg signed [7:0]  y8 [0:C-1];
    reg signed [63:0] e64;
    integer c3;
    always @(*) begin
        for (c3 = 0; c3 < C; c3 = c3 + 1) begin
            case (mode_q)
                STD: y8[c3] = acc[c3][7:0];
                ACT: y8[c3] = acc[c3][31] ? 8'b0 : acc[c3][7:0];
                EXP: begin
                    e64 = 1 + acc[c3] + ((acc[c3] * acc[c3]) >>> 1);
                    if      (e64 >  64'sd2147483647) e64 =  64'sd2147483647;    // the int32 clamp
                    else if (e64 < -64'sd2147483648) e64 = -64'sd2147483648;
                    y8[c3] = e64[7:0];
                end
                LOG: y8[c3] = acc[c3][7:0] - 8'd1;
                default: y8[c3] = 8'sd0;
            endcase
        end
    end

    reg [P-1:0] obuf [0:C-1];
    integer c4;
    always @(posedge clk) begin
        if (acam_fire)
            for (c4 = 0; c4 < C; c4 = c4 + 1) obuf[c4] <= (c4 < CU) ? y8[c4] : 8'd0;   // columns >= COLS_USED drain 0
    end

    // ------------------------------------------------------------------
    // 5 — control: ACAM, MSB_SA_Ready, the drain of P planes, dpe_done
    // ------------------------------------------------------------------
    reg [BW-1:0] drain_b;
    always @(posedge clk) begin
        if (reset) begin
            acam_fire      <= 1'b0;
            shift_add_done <= 1'b0;
            MSB_SA_Ready   <= 1'b1;
            draining       <= 1'b0;
            drain_b        <= 0;
            reg_full       <= 1'b0;
            dpe_done       <= 1'b0;
            shift_add_bypass_ctrl <= 1'b0;                              // reserved output, tied 0 (v2)
        end else begin
            shift_add_bypass_ctrl <= 1'b0;
            acam_fire      <= fire_msb;
            shift_add_done <= fire_msb;
            dpe_done       <= 1'b0;
            if (w_buf_en && ld_cnt == P - 1) MSB_SA_Ready <= 1'b0;      // low from load_start + P
            if (acam_fire) MSB_SA_Ready <= 1'b1;                        // high again from ACAM + 1
            if (acam_fire) begin
                draining <= 1'b1;
                drain_b  <= 0;
                reg_full <= 1'b1;
            end else if (draining) begin
                if (drain_b == P - 1) begin
                    draining <= 1'b0;
                    reg_full <= 1'b0;
                    dpe_done <= 1'b1;
                end else begin
                    drain_b <= drain_b + 1;
                end
            end
        end
    end

    integer c5;
    always @(*) begin
        for (c5 = 0; c5 < C; c5 = c5 + 1) data_out[c5] = draining ? obuf[c5][drain_b] : 1'b0;
    end


endmodule
