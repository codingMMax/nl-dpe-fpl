// ============================================================================
// dpe_nldpe.v — NL-DPE primitive, v2 clean-room skeleton
//
// Ground truth: `v2/spec/dpe_nldpe.md` (v1.1 amended 2026-09-12).
// Hand-written from the spec; legacy RTL (`rtl_flow/rtl/dpe_nldpe_faithful.v`)
// is a read-only witness — do not consult or copy it while implementing.
//
// Port surface is frozen and identical to `rtl_flow/vtr/dpe_blackbox.v` (I8):
//   in : clk, reset, data_in[39:0], nl_dpe_control[1:0], shift_add_control,
//        w_buf_en, shift_add_bypass, load_output_reg, load_input_reg
//   out: MSB_SA_Ready, data_out[39:0], dpe_done, reg_full, shift_add_done,
//        shift_add_bypass_ctrl
// Port semantics: spec §4. `w_buf_en` = ACT strobe; `load_input_reg` =
// WEIGHT strobe (data_in[31:0], one fp32 word/cycle, row-major row-outer,
// P17). Reserved inputs are tied 0 by the wrapper; `shift_add_bypass_ctrl`
// drives 0.
//
// TODO(you) blocks (spec anchors):
//   1 weight storage + WEIGHT strobe path            §3 / §4.2 / P17
//   2 input buffer: P banks x R bits + corner-turn   §3 / §4.3 / P1
//   3 crossbar fires + structural fp32 MAC           §6 F2 / P15 / P19
//   4 shift&acc (partial-shift, MSB subtract)        §6 F2 / P2 / P20
//   5 ACAM modes -> trunc8 (functional form first)   §6 F3 / P16 / P18
//   6 output buffer + drain (5 bytes/cyc) + done     §4.5 / P11
//   7 timing FSM + readiness                         §5.1-5.3 / P1 / P10
//   8 fp32 add/mul cores                             §9 option A (do first)
//
// COMPUTE_CYC = P+2 must emerge from the structure (P10); Δ_impl is a single
// implementation-declared constant, invariant across M and modes (§5.3).
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
    localparam integer WR_CYC      = R * C;                     // P17, one-time

    // ------------------------------------------------------------------
    // TODO(you): 1 — weight storage + WEIGHT strobe path
    //   §3: R*C fp32 words, one per (r, c); §4.2: one word per strobe
    //   cycle on data_in[31:0], order row-major row-outer (P17);
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
    // TODO(you): 3 — crossbar fires + structural fp32 MAC (§6 F2)
    //   Per slice b (LSB -> MSB), per column c: p_b[c] = fp32 running sum
    //   over r ascending of W[r,c] where bit b of x[r] is set. No FMA,
    //   RNE, gradual underflow (P15/P19). One slice per cycle.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(you): 4 — shift&acc (§6 F2 / P2 / P20)
    //   y = sum_{b=0..P-2} 2^b * p_b, then y -= 2^(P-1) * p_{P-1}.
    //   The partial enters at its own significance; the accumulator does
    //   NOT shift (P20); MSB slice is subtracted (two's complement, P2).
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // TODO(you): 5 — ACAM modes -> trunc8 (§6 F3 / P16 / P18)
    //   REGULAR: v;  ACTIVATION: relu(v);  EXP: 1 + v + v^2/2 (normative
    //   order);  LOG: v - 1. Functional form FIRST in fp32, then trunc8
    //   (truncate toward zero, clamp int32, keep low byte). C units
    //   parallel, 1 cycle, mode sampled at compute start (A11).
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

    // ------------------------------------------------------------------
    // TODO(you): 8 — fp32 add/mul cores (§9 option A)
    //   Verify bit-exact against NumPy float32 vectors BEFORE DPE-level
    //   checks: IEEE-754 binary32 RNE, no FMA, gradual underflow (P19).
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
