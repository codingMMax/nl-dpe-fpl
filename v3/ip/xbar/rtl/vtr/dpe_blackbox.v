// ============================================================================
// dpe_blackbox.v — NL-DPE primitive, VTR source (v2 clean-room Stage 1).
// Synthesis-side twin of the behavior model v2/sim/rtl/dpe_nldpe.v: the port
// surface of the `dpe` hard block only, no body. In the VTR flow every bare
// `dpe` instance of a kernel's VTR source binds to the architecture's
// `<model name="dpe">` (the `wc` subckt); the arch XML owns placement, routing
// and timing, so this file is not part of the circuits `vtr_tops.py` builds.
// It is the port contract the VTR sources are checked against
// (v2/experiment/rtl/check_split.py: ports == dpe_nldpe.v ports at
// DPE_BUF_WIDTH = 40 == the arch-XML model's ports; every VTR source
// elaborates against it).
//
// Ground truth: v2/sim/spec/dpe_nldpe.md (interface freeze 2026-09-15).
// ============================================================================

`timescale 1ns / 1ps

(* blackbox *)
module dpe (
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
    // No body: VTR maps every instance to the <model name="dpe"> hard block.
endmodule
