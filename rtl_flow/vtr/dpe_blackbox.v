// dpe_blackbox.v — VTR-targeted black-box declaration for the NL-DPE / AL DPE
// hard block.
//
// This file declares ONLY the port surface of the `dpe` module — no body.
// During VTR's CAD flow (Parmys → ABC → VPR), instances of `dpe(...)` in the
// netlist bind to the architecture's `<model name="dpe">` hard block; the
// FPGA arch XML (`nl_dpe_22nm_auto.xml` or `azure_lily_22nm_with_dpe_*.xml`)
// owns the cycle-accurate / power / area model.
//
// Spec anchors:
//   - nl_dpe_22nm_auto.xml `<model name="dpe">` port contract (line 219)
//   - rtl_flow/rtl/dpe_nldpe.v  (behavioral primitive — sim-only)
//   - rtl_flow/rtl/dpe_azurelily.v  (behavioral primitive — sim-only)
//
// Port widths assume DPE_BUF_WIDTH = 40 (NL-DPE default). For the Azure-Lily
// variant (BUF=16) build a parallel `dpe_blackbox_al.v` if/when AL VTR smoke
// is added.
//
// The `(* blackbox *)` attribute tells Parmys/Yosys not to recurse into the
// body — VTR will emit `.subckt dpe ...` for each instance in the BLIF and
// rely on the arch XML for placement / routing / timing.

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
    // Empty body — VTR maps to <model name="dpe"> hard block from arch XML.
endmodule
