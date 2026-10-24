// ============================================================================
// dpe_blackbox.v — NL-DPE crossbar primitive, VTR source: the port surface of the `dpe` hard block only, no body.
//
// Contract: v3/ip/xbar/spec.md §3 (the row-parallel bit-plane interface). Forked 2026-10-23 from v2's black box
// (git show v2-final:v2/experiment/rtl/dpe_blackbox.v; sha256 in tools/copies.lock and the LEDGER): data_in is R bits
// and data_out C bits, one bit plane per cycle, where v2 had 40 bits each way.
//
// In the VTR flow every bare `dpe` instance of a VTR twin binds to the architecture's <model name="dpe"> (the `wc`
// block of vtr/arch_rowpar/nl_dpe_<R>x<C>_auto.xml, data_in R pins, data_out C pins); Parmys takes the widths from the
// XML, so this file never enters a circuit. It is the port contract the twins are checked against
// (tools/check_twins.py: these ports == dpe_nldpe.v's, the widths by the same expressions; every VTR twin elaborates
// against it, and the width gate elaborates the head at an R != C build with these defaults overridden).
// ============================================================================

`timescale 1ns / 1ps

(* blackbox *)
module dpe #(
    parameter KERNEL_WIDTH = 256,     // R
    parameter NUM_COLS     = 256      // C
)(
    input  wire                    clk,
    input  wire                    reset,
    input  wire [KERNEL_WIDTH-1:0] data_in,
    input  wire [1:0]              nl_dpe_control,
    input  wire                    shift_add_control,
    input  wire                    w_buf_en,
    input  wire                    shift_add_bypass,
    input  wire                    load_output_reg,
    input  wire                    load_input_reg,
    output wire                    MSB_SA_Ready,
    output wire [NUM_COLS-1:0]     data_out,
    output wire                    dpe_done,
    output wire                    reg_full,
    output wire                    shift_add_done,
    output wire                    shift_add_bypass_ctrl
);
    // No body: VTR maps every instance to the <model name="dpe"> hard block.
endmodule
