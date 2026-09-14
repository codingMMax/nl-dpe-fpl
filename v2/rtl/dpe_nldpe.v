// ============================================================================
// dpe_nldpe.v — NL-DPE primitive, v2 clean-room skeleton
//
// Ground truth: `v2/spec/dpe_nldpe.md` (**v2.0 integer dataflow, 2026-09-14**).
// Hand-written from the spec; legacy RTL (`rtl_flow/rtl/dpe_nldpe_faithful.v`)
// is a read-only witness — do not consult or copy it while implementing.
//
// Port surface is frozen and identical to `rtl_flow/vtr/dpe_blackbox.v` (I8):
//   in : clk, reset, data_in[39:0], nl_dpe_control[1:0], shift_add_control,
//        w_buf_en, shift_add_bypass, load_output_reg, load_input_reg
//   out: MSB_SA_Ready, data_out[39:0], dpe_done, reg_full, shift_add_done,
//        shift_add_bypass_ctrl
// Port semantics: spec §4. `w_buf_en` = ACT strobe; `load_input_reg` =
// WEIGHT strobe (one int8 weight per cycle on data_in[7:0], row-major
// row-outer, P23). Reserved inputs are tied 0 by the wrapper;
// `shift_add_bypass_ctrl` drives 0.
//
// Numeric contract: int8 weights/activations, exact integer MAC into a 32-bit
// accumulator (|y| <= R*2^14, safe for R <= 131072, P25), integer ACAM mode
// forms, then `trunc8` (clamp to int32, keep the low byte, P16). No rounding,
// no fp32 cores.
//
// TODO(you) blocks (spec anchors):
//   1 weight storage + WEIGHT strobe path            §3 / §4.2 / P23
//   2 input buffer: P banks x R bits + corner-turn   §3 / §4.3 / P1
//   3 crossbar integer slice partials s_b            §6 F2
//   4 partial-shift accumulate + MSB subtract        §6 F2 / P2 / P22
//   5 ACAM integer modes -> trunc8 (low byte)        §6 F3 / P16 / P24
//   6 output buffer + drain (5 bytes/cyc) + done     §4.5 / P11
//   7 timing FSM + readiness                         §5.1-5.3 / P1 / P10
//
// COMPUTE_CYC = P+2 must emerge from the structure (P10); Δ_impl is a single
// implementation-declared constant, invariant across M and modes (§5.3).
// The TB compares both the hierarchical int32 `y` and the 8-bit stream (P26).
// ============================================================================

`timescale 1ns / 1ps

module dpe #(
    parameter integer R   = 256,   // crossbar rows (§1)
    parameter integer C   = 256,   // crossbar cols (reference: 512)
    parameter integer P   = 8,     // input bit-slices (§1)
    parameter integer BUF = 40     // external port width in bits (§1)
) (
    input  wire        clk,
    input  wire        reset,
    input  wire [39:0] data_in,
    input  wire [1:0]  nl_dpe_control,
    input  wire        shift_add_control,
    input  wire        w_buf_en,
    input  wire        shift_add_bypass,
    input  wire        load_output_reg,
    input  wire        load_input_reg,
    output wire        MSB_SA_Ready,
    output wire [39:0] data_out,
    output wire        dpe_done,
    output wire        reg_full,
    output wire        shift_add_done,
    output wire        shift_add_bypass_ctrl
);

    // ------------------------------------------------------------------
    // Derived stage lengths (§5.1/§5.3) — no other constants allowed
    // ------------------------------------------------------------------
    localparam integer LOAD_CYC    = (R * 8 + BUF - 1) / BUF;   // §4.3
    localparam integer COMPUTE_CYC = P + 2;                     // §5.1
    localparam integer OUTPUT_CYC  = (C * 8 + BUF - 1) / BUF;   // §4.5
    localparam integer WR_CYC      = R * C;                     // P23, one-time

    // ------------------------------------------------------------------
    // TODO(you): 1 — weight storage + WEIGHT strobe path
    //   §3: R*C int8 words, one per (r, c); §4.2: one byte per strobe
    //   cycle on data_in[7:0], order row-major row-outer (P23);
    //   stationary after programming (A5/A12). WR_CYC = R*C one-time,
    //   excluded from T_fill/T_steady, reported separately.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(you): 2 — input buffer (single, P1) + corner-turn (§4.3)
    //   P banks x R bits; byte-major writes: bank[b][j] = bit b of x[j].
    //   Burst = R bytes, 5 bytes/cycle. Refill permitted only from the
    //   cycle after MSB fire of the in-flight pass (MSB_SA_Ready rises).
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(you): 3 — crossbar integer slice partials (§6 F2)
    //   Per slice b (LSB -> MSB), per column c:
    //   s_b[c] = sum_r ( bank_b[r] ? W[r,c] : 0 )    — exact integer.
    //   One slice per cycle. No rounding, no order constraint (P22).
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(you): 4 — partial-shift accumulate + MSB subtract (§6 F2 / P22)
    //   y = sum_{b=0..P-2} (s_b << b)  -  (s_{P-1} << (P-1)).
    //   The partial enters at its own significance; the accumulator does
    //   not shift; the MSB slice is subtracted (two's complement, P2).
    //   int32 accumulator suffices for R <= 131072 (P25).
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(you): 5 — ACAM integer modes -> trunc8 (§6 F3 / P16 / P24)
    //   REGULAR: y;  ACTIVATION: relu(y);  EXP: 1 + y + floor(y^2/2)
    //   (wide intermediate; only y mod 512 affects the output byte);
    //   LOG: y - 1.  Then trunc8: clamp to int32, keep the low byte as
    //   signed int8. C units parallel, 1 cycle, mode sampled at compute
    //   start (A11).
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(you): 6 — output buffer + drain (§4.5 / P11)
    //   C x 8 bits, written wholesale by ACAM; drained 5 bytes/cycle in
    //   column order, gapless; dpe_done 1-cycle pulse after the last byte;
    //   reg_full marks the buffer busy.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(you): 7 — timing FSM + readiness (§5.1-5.3 / P1 / P10)
    //   LOAD_CYC -> COMPUTE_CYC (= P+2, must emerge structurally) ->
    //   ACAM -> OUTPUT_CYC; next ACT burst may start the cycle after MSB
    //   fire; ACAM strictly after the previous drain (P11); reset
    //   synchronous active-high (A10); measured(M) = T_fill + (M-1)*
    //   T_steady + Δ_impl, Δ_impl invariant (I2).
    // ------------------------------------------------------------------

    // Placeholder drivers — replace as each TODO block lands (keeps this
    // file elaborating cleanly in the meantime).
    assign MSB_SA_Ready          = 1'b0;
    assign data_out              = 40'b0;
    assign dpe_done              = 1'b0;
    assign reg_full              = 1'b0;
    assign shift_add_done        = 1'b0;
    assign shift_add_bypass_ctrl = 1'b0;

endmodule
