#!/usr/bin/env python3
"""
DPE behavior model generator (FIDELITY_METHODOLOGY.md §3).

Reads a per-arch config JSON and emits one Verilog file per architecture
into fc_verification/rtl/dpe_<arch>.v.

The emitted module is named `dpe` (not `dpe_<arch>`) so that it matches
the VTR arch XML's <model name="dpe"> blackbox port contract for both
nl_dpe/nl_dpe_22nm_auto.xml and
azurelily_TACO_experiments/azure_lily_22nm_with_dpe_550x550.xml.

Both arch XMLs declare an identical port list:
  inputs : clk, reset, data_in, nl_dpe_control, shift_add_control,
           w_buf_en, shift_add_bypass, load_output_reg, load_input_reg
  outputs: data_out, MSB_SA_Ready, dpe_done, reg_full, shift_add_done,
           shift_add_bypass_ctrl

VTR ignores Verilog parameters on the blackbox; the parameters are
sim-only. To exercise both archs in the same simulator run is impossible
because both files declare `module dpe`; pick the appropriate file when
compiling for a given workload.

Both arch outputs share:
  - Handshake protocol (clk, reset, data_in, nl_dpe_control[1:0],
    shift_add_control, w_buf_en, shift_add_bypass, load_output_reg,
    load_input_reg, MSB_SA_Ready, data_out, dpe_done, reg_full,
    shift_add_done, shift_add_bypass_ctrl).
  - FSM (S_IDLE -> S_LOAD -> S_WAIT_EXEC -> S_COMPUTE -> S_OUTPUT
    -> S_DRAIN).
  - Behavioral 1-clock VMM at the moment S_WAIT_EXEC samples
    nl_dpe_control == 2'b11 (or, on the last LOAD strobe combinationally
    when the same condition holds).
  - Cycle counters (LOAD_STROBES, OUTPUT_CYCLES) derived from the
    arch's KERNEL_WIDTH / NUM_COLS / DPE_BUF_WIDTH parameters.

Compute timing (Model Y):
  - DPE stays in S_COMPUTE while nl_dpe_control == 2'b11.
  - DPE transitions to S_OUTPUT on the cycle where nl_dpe_control
    deasserts.
  - The VMM math runs ONCE upon entering S_COMPUTE; vmm_result[]
    holds the values; S_COMPUTE is a wait-loop on control deassert.
  - The DPE has NO internal compute counter. Compute duration is
    controller-driven: the controller (workload top / TB) asserts
    nl_dpe_control = 2'b11 for (PRECISION + PIPELINE_DEPTH - 1)
    cycles to emulate the bit-serial pipeline.

Differ between archs only by:
  - ACAM_MODE param + branch present iff capabilities.analog_nonlinear
    is true (NL-DPE only).

Usage:
    # one arch:
    python nl_dpe/gen_dpe_stub.py \
        --config azurelily/IMC/configs/nl_dpe.json
    python nl_dpe/gen_dpe_stub.py \
        --config azurelily/IMC/configs/azure_lily.json

    # both archs at once (default config paths):
    python nl_dpe/gen_dpe_stub.py
"""
import argparse
import json
import math
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_CFG_PATHS = [
    os.path.join(REPO, "azurelily/IMC/configs/nl_dpe.json"),
    os.path.join(REPO, "azurelily/IMC/configs/azure_lily.json"),
]
OUT_DIR = os.path.join(REPO, "fc_verification/rtl")

# core_name -> filename suffix.
ARCH_SUFFIX = {
    "NL-DPE":     "nldpe",
    "Azure-Lily": "azurelily",
}


def _load_arch_cfg(cfg_path):
    """Load arch geometry / capabilities via simulator's Config class."""
    imc_root = os.path.join(REPO, "azurelily", "IMC")
    if imc_root not in sys.path:
        sys.path.insert(0, imc_root)
    from imc_core.config import Config  # noqa: WPS433
    return Config(cfg_path)


def derive_arch_params(cfg_path):
    """Read per-arch JSON and return canonical RTL params.

    Returns dict with keys:
        arch_suffix, R, C, BUF, has_acam, load_strobes, output_cycles,
        core_name
    """
    with open(cfg_path, "r") as fh:
        cfg_json = json.load(fh)

    core_name = cfg_json.get("core_name", "")
    arch_suffix = ARCH_SUFFIX.get(core_name)
    if arch_suffix is None:
        raise ValueError(
            f"Unknown core_name {core_name!r} in {cfg_path}; "
            f"known: {list(ARCH_SUFFIX)}"
        )

    cfg = _load_arch_cfg(cfg_path)
    R = int(cfg.rows)
    C = int(cfg.cols)
    BUF = int(cfg.dpe_buf_width)
    has_acam = bool(cfg.analoge_nonlinear_support)

    elems_per_strobe = BUF // 8
    rtl_load = math.ceil(R * 8 / BUF)
    rtl_output = math.ceil(C * 8 / BUF)
    assert elems_per_strobe >= 1, f"BUF={BUF} < 8 not supported"

    return {
        "arch_suffix": arch_suffix,
        "R": R,
        "C": C,
        "BUF": BUF,
        "has_acam": has_acam,
        "load_strobes": int(rtl_load),
        "output_cycles": int(rtl_output),
        "core_name": core_name,
    }


def emit_dpe_verilog(params):
    """Return Verilog source string for the DPE behavior model.

    Module name is always `dpe` to match the VTR arch XML
    <model name="dpe"> port contract.
    """
    arch = params["arch_suffix"]
    R = params["R"]
    C = params["C"]
    BUF = params["BUF"]
    has_acam = params["has_acam"]
    load_strobes = params["load_strobes"]
    output_cycles = params["output_cycles"]
    elems_per_strobe = BUF // 8

    lines = []
    a = lines.append

    a(f"// dpe_{arch}.v -- behavioral DPE model")
    a(f"// Generated by nl_dpe/gen_dpe_stub.py from per-arch JSON config.")
    a(f"// Module name      : dpe  (matches VTR arch XML <model name=\"dpe\"> contract)")
    a(f"// File arch tag    : {arch} ({params['core_name']})")
    a(f"// KERNEL_WIDTH (R) : {R}")
    a(f"// NUM_COLS     (C) : {C}")
    a(f"// DPE_BUF_WIDTH    : {BUF}  (elems/strobe = {elems_per_strobe})")
    a(f"// LOAD_STROBES     : {load_strobes}")
    a(f"// OUTPUT_CYCLES    : {output_cycles}")
    a(f"// ACAM_MODE param  : {'present' if has_acam else 'absent'}")
    a("//")
    a("// Compute timing (Model Y):")
    a("//   DPE stays in S_COMPUTE while nl_dpe_control == 2'b11.")
    a("//   Transitions to S_OUTPUT on the cycle nl_dpe_control deasserts.")
    a("//   No internal compute counter; precision/pipeline depth is owned")
    a("//   by the controller (workload top / TB), which holds")
    a("//   nl_dpe_control = 2'b11 for (PRECISION + PIPELINE_DEPTH - 1)")
    a("//   cycles to emulate the bit-serial pipeline.")
    a("//")
    a("// FSM:  S_IDLE -> S_LOAD -> S_WAIT_EXEC -> S_COMPUTE -> S_OUTPUT -> S_DRAIN")
    a("// Behavioral single-clock VMM at S_WAIT_EXEC fire-time")
    a("// (or on the last LOAD strobe combinationally when ctrl is already 2'b11).")
    a("// Weights are loaded by the TB through hierarchical force:")
    a("//     dut.weights[r][c] = <int8>;   (no weight_wen port).")
    a("")
    a("module dpe #(")
    a(f"    parameter KERNEL_WIDTH   = {R},")
    a(f"    parameter NUM_COLS       = {C},")
    a(f"    parameter DPE_BUF_WIDTH  = {BUF}" + ("," if has_acam else ""))
    if has_acam:
        a("    parameter ACAM_MODE      = 0  // 0=ADC/VMM, 1=exp(approx 1+x+x^2/2), 2=log(approx x-1)")
    a(")(")
    a("    input  wire                       clk,")
    a("    input  wire                       reset,")
    a("    input  wire [DPE_BUF_WIDTH-1:0]   data_in,")
    a("    input  wire [1:0]                 nl_dpe_control,")
    a("    input  wire                       shift_add_control,")
    a("    input  wire                       w_buf_en,")
    a("    input  wire                       shift_add_bypass,")
    a("    input  wire                       load_output_reg,")
    a("    input  wire                       load_input_reg,")
    a("    output reg                        MSB_SA_Ready,")
    a("    output reg  [DPE_BUF_WIDTH-1:0]   data_out,")
    a("    output reg                        dpe_done,")
    a("    output reg                        reg_full,")
    a("    output reg                        shift_add_done,")
    a("    output reg                        shift_add_bypass_ctrl")
    a(");")
    a("")
    a("    // Tie-off auxiliary handshake inputs (kept in port list for")
    a("    // protocol compatibility with the legacy controller).")
    a("    wire _unused_aux = shift_add_control | shift_add_bypass |")
    a("                       load_output_reg   | load_input_reg;")
    a("")
    a("    // Derived per-pass cycle counters")
    a("    localparam ELEMS_PER_STROBE = DPE_BUF_WIDTH / 8;")
    a("    localparam LOAD_STROBES  = (KERNEL_WIDTH + ELEMS_PER_STROBE - 1) / ELEMS_PER_STROBE;")
    a("    localparam OUTPUT_CYCLES = (NUM_COLS    + ELEMS_PER_STROBE - 1) / ELEMS_PER_STROBE;")
    a("")
    a("    // Weight memory (hierarchical-force loaded from TB)")
    a("    reg signed [7:0] weights [0:KERNEL_WIDTH-1][0:NUM_COLS-1];")
    a("")
    a("    // Input buffer (one int8 per row)")
    a("    reg signed [7:0] input_buffer [0:KERNEL_WIDTH-1];")
    a("    reg [15:0] load_count;")
    a("    reg [15:0] strobe_count;")
    a("")
    a("    // VMM accumulators (full int32 per output column)")
    a("    reg signed [31:0] vmm_result [0:NUM_COLS-1];")
    a("    reg [15:0] output_col_idx;")
    a("")
    a("    // Initialise weight memory to zero (sim-only)")
    a("    integer wi, wj;")
    a("    initial begin")
    a("        for (wi = 0; wi < KERNEL_WIDTH; wi = wi + 1)")
    a("            for (wj = 0; wj < NUM_COLS; wj = wj + 1)")
    a("                weights[wi][wj] = 0;")
    a("    end")
    a("")
    a("    // FSM state encoding")
    a("    localparam S_IDLE      = 3'd0;")
    a("    localparam S_LOAD      = 3'd1;")
    a("    localparam S_WAIT_EXEC = 3'd2;")
    a("    localparam S_COMPUTE   = 3'd3;")
    a("    localparam S_OUTPUT    = 3'd4;")
    a("    localparam S_DRAIN     = 3'd5;")
    a("    reg [2:0] state;")
    a("")
    a("    integer r, c, b;")
    a("")
    a("    always @(posedge clk or posedge reset) begin")
    a("        if (reset) begin")
    a("            state <= S_IDLE;")
    a("            load_count <= 0;")
    a("            strobe_count <= 0;")
    a("            data_out <= 0;")
    a("            dpe_done <= 0;")
    a("            reg_full <= 0;")
    a("            MSB_SA_Ready <= 1;")
    a("            shift_add_done <= 1;")
    a("            shift_add_bypass_ctrl <= 1;")
    a("            output_col_idx <= 0;")
    a("        end else begin")
    a("            dpe_done <= 0;  // default low (one-shot in S_OUTPUT)")
    a("            case (state)")
    a("                S_IDLE: begin")
    a("                    reg_full <= 0;")
    a("                    MSB_SA_Ready <= 1;")
    a("                    shift_add_done <= 1;")
    a("                    load_count <= 0;")
    a("                    strobe_count <= 0;")
    a("                    output_col_idx <= 0;")
    a("                    if (w_buf_en) begin")
    a("                        for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)")
    a("                            if (b < KERNEL_WIDTH)")
    a("                                input_buffer[b] <= data_in[b*8 +: 8];")
    a("                        load_count <= (ELEMS_PER_STROBE < KERNEL_WIDTH) ?")
    a("                                       ELEMS_PER_STROBE : KERNEL_WIDTH;")
    a("                        strobe_count <= 1;")
    a("                        if (1 >= LOAD_STROBES) begin")
    a("                            reg_full <= 1;")
    a("                            MSB_SA_Ready <= 0;")
    a("                            shift_add_done <= 0;")
    a("                            state <= S_WAIT_EXEC;")
    a("                        end else begin")
    a("                            state <= S_LOAD;")
    a("                        end")
    a("                    end")
    a("                end")
    a("                S_LOAD: begin")
    a("                    if (w_buf_en) begin")
    a("                        if (strobe_count + 1 >= LOAD_STROBES) begin")
    a("                            // Last LOAD strobe — write input_buffer with")
    a("                            // BLOCKING assignment so the VMM below reads the")
    a("                            // freshly-loaded final bytes (mixing blocking +")
    a("                            // non-blocking is sim-only behavioural; there is")
    a("                            // no synthesised hardware here). Then fire VMM")
    a("                            // combinationally when ctrl is already 2'b11.")
    a("                            for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)")
    a("                                if (load_count + b < KERNEL_WIDTH)")
    a("                                    input_buffer[load_count + b] = data_in[b*8 +: 8];")
    a("                            load_count <= load_count + ELEMS_PER_STROBE;")
    a("                            strobe_count <= strobe_count + 1;")
    a("                            reg_full <= 1;")
    a("                            MSB_SA_Ready <= 0;")
    a("                            shift_add_done <= 0;")
    a("                            if (nl_dpe_control == 2'b11) begin")
    a("                                for (c = 0; c < NUM_COLS; c = c + 1) begin")
    a("                                    vmm_result[c] = 0;")
    a("                                    for (r = 0; r < KERNEL_WIDTH; r = r + 1) begin")
    a("                                        vmm_result[c] = vmm_result[c] +")
    a("                                            input_buffer[r] * weights[r][c];")
    a("                                    end")
    a("                                end")
    if has_acam:
        a("                                if (ACAM_MODE == 1) begin")
        a("                                    for (c = 0; c < NUM_COLS; c = c + 1)")
        a("                                        vmm_result[c] = 1 + vmm_result[c] +")
        a("                                            (vmm_result[c] * vmm_result[c]) / 2;")
        a("                                end else if (ACAM_MODE == 2) begin")
        a("                                    for (c = 0; c < NUM_COLS; c = c + 1)")
        a("                                        vmm_result[c] = vmm_result[c] - 1;")
        a("                                end")
    a("                                output_col_idx <= 0;")
    a("                                state <= S_COMPUTE;")
    a("                            end else begin")
    a("                                state <= S_WAIT_EXEC;")
    a("                            end")
    a("                        end else begin")
    a("                            // Non-final LOAD strobe — keep NBA semantics.")
    a("                            for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)")
    a("                                if (load_count + b < KERNEL_WIDTH)")
    a("                                    input_buffer[load_count + b] <= data_in[b*8 +: 8];")
    a("                            load_count <= load_count + ELEMS_PER_STROBE;")
    a("                            strobe_count <= strobe_count + 1;")
    a("                        end")
    a("                    end")
    a("                end")
    a("                S_WAIT_EXEC: begin")
    a("                    if (nl_dpe_control == 2'b11) begin")
    a("                        // Single-clock behavioral VMM:")
    a("                        // vmm_result[c] = sum_r input_buffer[r] * weights[r][c]")
    a("                        for (c = 0; c < NUM_COLS; c = c + 1) begin")
    a("                            vmm_result[c] = 0;")
    a("                            for (r = 0; r < KERNEL_WIDTH; r = r + 1) begin")
    a("                                vmm_result[c] = vmm_result[c] +")
    a("                                    input_buffer[r] * weights[r][c];")
    a("                            end")
    a("                        end")
    if has_acam:
        a("                        // ACAM_MODE: 0=ADC, 1=exp approx, 2=log approx")
        a("                        if (ACAM_MODE == 1) begin")
        a("                            for (c = 0; c < NUM_COLS; c = c + 1)")
        a("                                vmm_result[c] = 1 + vmm_result[c] +")
        a("                                    (vmm_result[c] * vmm_result[c]) / 2;")
        a("                        end else if (ACAM_MODE == 2) begin")
        a("                            for (c = 0; c < NUM_COLS; c = c + 1)")
        a("                                vmm_result[c] = vmm_result[c] - 1;")
        a("                        end")
    a("                        output_col_idx <= 0;")
    a("                        state <= S_COMPUTE;")
    a("                    end")
    a("                end")
    a("                S_COMPUTE: begin")
    a("                    // Model Y: stay here while controller holds nl_dpe_control = 2'b11.")
    a("                    // Transition to S_OUTPUT when controller deasserts (precision-driven).")
    a("                    if (nl_dpe_control != 2'b11) begin")
    a("                        MSB_SA_Ready <= 1;")
    a("                        shift_add_done <= 1;")
    a("                        output_col_idx <= 0;")
    a("                        state <= S_OUTPUT;")
    a("                    end")
    a("                end")
    a("                S_OUTPUT: begin")
    a("                    dpe_done <= 1;  // held high during output phase")
    a("                    data_out <= 0;")
    a("                    for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)")
    a("                        if (output_col_idx * ELEMS_PER_STROBE + b < NUM_COLS)")
    a("                            data_out[b*8 +: 8] <= vmm_result[output_col_idx * ELEMS_PER_STROBE + b][7:0];")
    a("                    if (output_col_idx < OUTPUT_CYCLES - 1) begin")
    a("                        output_col_idx <= output_col_idx + 1;")
    a("                    end else begin")
    a("                        state <= S_DRAIN;")
    a("                    end")
    a("                end")
    a("                S_DRAIN: begin")
    a("                    dpe_done <= 0;")
    a("                    reg_full <= 0;")
    a("                    load_count <= 0;")
    a("                    strobe_count <= 0;")
    a("                    output_col_idx <= 0;")
    a("                    MSB_SA_Ready <= 1;")
    a("                    shift_add_done <= 1;")
    a("                    state <= S_IDLE;")
    a("                end")
    a("                default: state <= S_IDLE;")
    a("            endcase")
    a("        end")
    a("    end")
    a("")
    a("endmodule")
    a("")
    return "\n".join(lines)


def write_arch(cfg_path):
    params = derive_arch_params(cfg_path)
    src = emit_dpe_verilog(params)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"dpe_{params['arch_suffix']}.v")
    with open(out_path, "w") as fh:
        fh.write(src)
    print(f"[gen_dpe_stub] wrote {out_path}")
    print(f"               module=dpe arch={params['arch_suffix']} R={params['R']} "
          f"C={params['C']} BUF={params['BUF']} "
          f"L={params['load_strobes']} O={params['output_cycles']} "
          f"has_acam={params['has_acam']}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="DPE behavior model generator")
    p.add_argument("--config", default=None,
                   help="Per-arch JSON config path. If omitted, both default "
                        "configs are emitted.")
    args = p.parse_args(argv)

    if args.config is not None:
        write_arch(args.config)
    else:
        for cfg_path in DEFAULT_CFG_PATHS:
            write_arch(cfg_path)


if __name__ == "__main__":
    main()
