#!/usr/bin/env python3
"""
DPE behavior model generator (FIDELITY_METHODOLOGY.md §3 + §4).

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
  - Three parallel sub-FSMs implementing FIDELITY_METHODOLOGY.md §4
    single-buffered drain-load overlap pipeline:
      LOAD    : ingests w_buf_en strobes, fires VMM at the LCYC-th
                strobe of each pass into vmm_queue[q_load_tail].
      COMPUTE : runs COMPUTE_CYCLES cycles per queued pass; tightly
                coupled with LOAD's fire so M=1 cycle parity is
                preserved (S_LOAD->S_COMPUTE on last strobe).
      OUTPUT  : drains OUTPUT_CYCLES strobes per pass; chains
                back-to-back across passes when ready (n_pending_output
                > 1 or compute completes this same cycle).
  - Behavioral 1-clock VMM at the moment of LOAD-fire (last LOAD
    strobe of a pass): vmm_queue[q_load_tail][c] = sum_r input_buffer[r]
    * weights[r][c]. Optionally extended via ACAM_MODE.
  - Cycle counters (LOAD_CYCLES, OUTPUT_CYCLES) derived from the
    arch's KERNEL_WIDTH / NUM_COLS / DPE_BUF_WIDTH parameters.
  - COMPUTE_CYCLES parameter (per-arch default) parametrising the
    bit-serial compute duration. Controllers / TBs pass the runtime
    CCYC via parameter override.

Per-arch CCYC decomposition (Task #86):
  CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES

  NL-DPE (2-stage internal pipeline: crossbar MAC -> analog accumulator,
          + 1-cycle ACAM read-out path that always fires regardless of
          activation mode — ACAM_MODE just selects the LUT contents):
      capabilities.pipeline_depth = 2
      capabilities.acam_cycles    = 1
      CCYC = PRECISION + 1 + 1 = PRECISION + 2
      INT8  -> 10 cycles
      INT4  ->  6 cycles
      INT16 -> 18 cycles

  Azure-Lily (3-stage internal pipeline: crossbar MAC -> ADC ->
              shift-add; no ACAM stage):
      capabilities.pipeline_depth = 3
      capabilities.acam_cycles    = 0
      CCYC = PRECISION + 2 + 0 = PRECISION + 2
      INT8  -> 10 cycles
      INT4  ->  6 cycles
      INT16 -> 18 cycles

Both archs give CCYC = PRECISION + 2 under the current parameters —
this is structural symmetry, not coincidence: (D_NL - 1) + ACAM_NL =
(D_AL - 1) + ACAM_AL = 2. If either side's pipeline depth or ACAM
latency changes the symmetry breaks. Backward compatibility: if the
JSON capabilities block lacks pipeline_depth / acam_cycles, the
generator falls back to legacy (3, 0) (single-knob 3-stage pipeline
with no separate ACAM cycle).

Differ between archs only by:
  - ACAM_MODE param + branch present iff capabilities.analog_nonlinear
    is true (NL-DPE only).
  - COMPUTE_CYCLES default baked from the per-arch decomposition above.

Cycle accounting (faithful NBA-FSM, RTL behavior):
  T_rtl(M)     = T_fill_rtl + (M-1) * T_steady
  T_fill_rtl   = LOAD_CYCLES + COMPUTE_CYCLES + OUTPUT_CYCLES + 2
  T_steady     = max(LOAD_CYCLES, COMPUTE_CYCLES, OUTPUT_CYCLES)

The +2 in T_fill_rtl is the **faithful FSM register-propagation
overhead**: 1 cycle for the LOAD→COMPUTE handoff (q_load_tail NBA must
commit before COMPUTE's idle branch reads n_pending_compute>0) plus 1
cycle for the COMPUTE→OUTPUT handoff (q_compute_head NBA must commit
before OUTPUT's idle branch reads n_pending_output>0). This is real
silicon overhead of our particular NBA-propagating FSM design — NOT a
methodology constant. The simulator emits the ideal T_fill_ideal =
L + C + O (per FIDELITY_METHODOLOGY §4); the RTL pays +2; fidelity is
the measurement. See FIDELITY_METHODOLOGY §4.2 for the framing
(Option A, post Task #90).

Steady-state cadence T_steady = max(LOAD_CYCLES, COMPUTE_CYCLES,
OUTPUT_CYCLES) is unchanged in both sim and RTL because the chain
mechanism amortises subsequent handoffs into the per-pass pipeline
depth.

This supersedes the same-cycle blocking-pulse (`fire_now`,
`compute_done_now`) trick used pre-Task #87, which collapsed both
handoffs into 0 cycles and made the model implausibly tight (no real
hardware FSM can transition states in 0 cycles).

The legacy `state` reg surface is retained as a wire computed from
sub-FSM activity (priority OUTPUT > COMPUTE > LOAD > IDLE) so TBs that
probe `dut.state == 3'd4` continue to observe the OUTPUT phase boundary
(now at cycle (LOAD_CYCLES + COMPUTE_CYCLES + 2)..(T_fill - 1) for
pass 0).

Usage:
    # one arch:
    python nl_dpe/gen_dpe_stub.py \
        --config archive/azurelily_simulator/IMC/configs/nl_dpe.json
    python nl_dpe/gen_dpe_stub.py \
        --config archive/azurelily_simulator/IMC/configs/azure_lily.json

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
    os.path.join(REPO, "archive/azurelily_simulator/IMC/configs/nl_dpe.json"),
    os.path.join(REPO, "archive/azurelily_simulator/IMC/configs/azure_lily.json"),
]
OUT_DIR = os.path.join(REPO, "fc_verification/rtl")

# core_name -> filename suffix.
ARCH_SUFFIX = {
    "NL-DPE":     "nldpe",
    "Azure-Lily": "azurelily",
}


def _load_arch_cfg(cfg_path):
    """Load arch geometry / capabilities via simulator's Config class."""
    imc_root = os.path.join(REPO, "archive/azurelily_simulator", "IMC")
    if imc_root not in sys.path:
        sys.path.insert(0, imc_root)
    from imc_core.config import Config  # noqa: WPS433
    return Config(cfg_path)


def derive_arch_params(cfg_path):
    """Read per-arch JSON and return canonical RTL params.

    Returns dict with keys:
        arch_suffix, R, C, BUF, has_acam, load_cycles, output_cycles,
        core_name, precision_bits, pipeline_depth, acam_cycles,
        compute_cycles_default

    Per-arch CCYC decomposition (Task #86):
        compute_cycles_default = precision_bits
                               + (pipeline_depth - 1)  # last-bit drain
                               + acam_cycles            # ACAM read-out (NL only)

    Backward compat: missing capabilities.pipeline_depth / .acam_cycles
    fall back to the legacy single-knob defaults (3, 0).
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

    # Per-arch CCYC decomposition (Task #86): read from capabilities block.
    capabilities = cfg_json.get("capabilities", {})
    precision_bits = int(cfg_json.get("precision_bits", 8))
    pipeline_depth = int(capabilities.get("pipeline_depth", 3))
    acam_cycles = int(capabilities.get("acam_cycles", 0))
    compute_cycles_default = precision_bits + (pipeline_depth - 1) + acam_cycles

    return {
        "arch_suffix": arch_suffix,
        "R": R,
        "C": C,
        "BUF": BUF,
        "has_acam": has_acam,
        "load_cycles": int(rtl_load),
        "output_cycles": int(rtl_output),
        "core_name": core_name,
        "precision_bits": precision_bits,
        "pipeline_depth": pipeline_depth,
        "acam_cycles": acam_cycles,
        "compute_cycles_default": int(compute_cycles_default),
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
    load_cycles = params["load_cycles"]
    output_cycles = params["output_cycles"]
    precision_bits = params["precision_bits"]
    pipeline_depth = params["pipeline_depth"]
    acam_cycles = params["acam_cycles"]
    compute_cycles_default = params["compute_cycles_default"]
    elems_per_strobe = BUF // 8

    # Per-arch CCYC decomposition string (Task #86) — used in the
    # emitted Verilog header so a reader can derive COMPUTE_CYCLES
    # without leaving the file.
    if has_acam:
        stages_desc = "(MAC, Acc)"
    else:
        stages_desc = "(MAC, ADC, SA)"

    lines = []
    a = lines.append

    a(f"// dpe_{arch}.v -- behavioral DPE model with single-buffered drain-load overlap.")
    a(f"// Generated by nl_dpe/gen_dpe_stub.py from per-arch JSON config.")
    a(f"// Module name      : dpe  (matches VTR arch XML <model name=\"dpe\"> contract)")
    a(f"// File arch tag    : {arch} ({params['core_name']})")
    a(f"// KERNEL_WIDTH (R) : {R}")
    a(f"// NUM_COLS     (C) : {C}")
    a(f"// DPE_BUF_WIDTH    : {BUF}  (elems/strobe = {elems_per_strobe})")
    a(f"// LOAD_CYCLES     : {load_cycles}")
    a(f"// OUTPUT_CYCLES    : {output_cycles}")
    a(f"// COMPUTE_CYCLES   : parameter (default {compute_cycles_default} = per-arch decomposition below)")
    a(f"// ACAM_MODE param  : {'present' if has_acam else 'absent'}")
    a("//")
    a(f"// COMPUTE_CYCLES decomposition ({params['core_name']}):")
    a(f"//   PRECISION ({precision_bits}) cycles    -- bit-slices entering crossbar pipeline")
    a(f"// + PIPELINE_DEPTH ({pipeline_depth}) - 1  -- last-bit drain through {stages_desc} stages")
    if has_acam:
        a(f"// + ACAM_CYCLES ({acam_cycles})         -- ACAM read-out (always fires)")
    else:
        a(f"// + ACAM_CYCLES ({acam_cycles})         -- no ACAM in Azure-Lily")
    a(f"// = {compute_cycles_default} cycles (INT{precision_bits})")
    a("//")
    a("// Both arches give CCYC = PRECISION + 2 under current parameters —")
    a("// structural symmetry, not coincidence: (D_NL - 1) + ACAM_NL =")
    a("// (D_AL - 1) + ACAM_AL = 2. If either side's PIPELINE_DEPTH or")
    a("// ACAM_CYCLES changes the symmetry breaks. See")
    a("// fc_verification/FIDELITY_METHODOLOGY.md §3 and")
    a("// fc_verification/DPE_PRIMITIVE_WALKTHROUGH.md §10a for the")
    a("// per-arch pipeline diagrams.")
    a("//")
    a("// Pipeline model (FIDELITY_METHODOLOGY.md §4): single-buffered drain-load")
    a("// overlap, T(M) = T_fill + (M-1) * T_steady where")
    a("// T_fill_rtl = LOAD_CYCLES + COMPUTE_CYCLES + OUTPUT_CYCLES + 2  (faithful FSM)")
    a("//   (sim emits ideal T_fill = L + C + O; +2 surfaces as fidelity, Option A)")
    a("// T_steady = max(LOAD_CYCLES, COMPUTE_CYCLES, OUTPUT_CYCLES).")
    a("//")
    a("// The +2 in T_fill is the faithful FSM register-propagation overhead:")
    a("//   +1 cycle for LOAD→COMPUTE handoff: at the LOAD last strobe cycle T,")
    a("//      q_load_tail NBAs from t→t+1; the COMPUTE idle branch only reads")
    a("//      the updated n_pending_compute at cycle T+1, NBAs compute_busy<=1,")
    a("//      so compute_busy is observable from cycle T+2 onward.")
    a("//   +1 cycle for COMPUTE→OUTPUT handoff: at COMPUTE done cycle U,")
    a("//      q_compute_head NBAs; OUTPUT idle branch reads n_pending_output>0")
    a("//      at cycle U+1, NBAs output_busy<=1, so OUTPUT is observable from")
    a("//      cycle U+2 onward.")
    a("//")
    a("// Three parallel sub-FSMs operate concurrently:")
    a("//   LOAD    : ingests w_buf_en strobes; every LOAD_CYCLES strobes fires")
    a("//             the behavioral VMM into vmm_queue[q_load_tail] and re-arms")
    a("//             for the next pass.")
    a("//   COMPUTE : when q_compute_head != q_load_tail and not busy, runs")
    a("//             COMPUTE_CYCLES cycles, then advances q_compute_head.")
    a("//             COMPUTE cannot overlap across passes (§4).")
    a("//   OUTPUT  : when q_output_head != q_compute_head and not busy, drains")
    a("//             OUTPUT_CYCLES strobes from vmm_queue[q_output_head] onto")
    a("//             data_out, advances q_output_head.")
    a("//")
    a("// Single-buffer semantics: physically there is one input buffer; bytes")
    a("// stream through the analog crossbar bit-serially. Behaviorally the VMM")
    a("// math runs in a single posedge (the last LOAD strobe of a pass) and")
    a("// the result is parked in vmm_queue, freeing input_buffer immediately")
    a("// for the next pass's LOAD strobes.")
    a("//")
    a("// Sub-FSM coordination via queue indices (Task #87 Phase 1):")
    a("//   - LOAD fires VMM math and NBAs q_load_tail++.")
    a("//   - COMPUTE wakes the next cycle when n_pending_compute>0.")
    a("//   - COMPUTE done NBAs q_compute_head++.")
    a("//   - OUTPUT wakes the next cycle when n_pending_output>0.")
    a("//   - OUTPUT chains across passes via `n_pending_output > 1` at the")
    a("//     last strobe — when COMPUTE has already advanced past the current")
    a("//     OUTPUT pass by the time OUTPUT's last strobe fires.")
    a("//")
    a("// Legacy `state` reg surface preserved as a wire (priority OUTPUT >")
    a("// COMPUTE > LOAD > IDLE) for TB compat with `dut.state == 3'd4`.")
    a("//")
    a("// Weights are loaded by the TB through hierarchical force:")
    a("//     dut.weights[r][c] = <int8>;   (no weight_wen port).")
    a("")
    a("module dpe #(")
    a(f"    parameter KERNEL_WIDTH   = {R},")
    a(f"    parameter NUM_COLS       = {C},")
    a(f"    parameter DPE_BUF_WIDTH  = {BUF},")
    a(f"    parameter COMPUTE_CYCLES = {compute_cycles_default}" + ("," if has_acam else ""))
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
    a("    localparam LOAD_CYCLES  = (KERNEL_WIDTH + ELEMS_PER_STROBE - 1) / ELEMS_PER_STROBE;")
    a("    localparam OUTPUT_CYCLES = (NUM_COLS    + ELEMS_PER_STROBE - 1) / ELEMS_PER_STROBE;")
    a("")
    a("    // Ring buffer queue depth: at most 3 passes in flight (LOAD just")
    a("    // finished awaiting compute; COMPUTE running; OUTPUT running).")
    a("    // QDEPTH=4 for headroom; q_*_tail/head are 4-bit counters mod QDEPTH.")
    a("    localparam QDEPTH = 4;")
    a("")
    a("    // Weight memory (hierarchical-force loaded from TB)")
    a("    reg signed [7:0] weights [0:KERNEL_WIDTH-1][0:NUM_COLS-1];")
    a("")
    a("    // Input buffer (single physical buffer reused across passes).")
    a("    reg signed [7:0] input_buffer [0:KERNEL_WIDTH-1];")
    a("")
    a("    // Per-pass VMM result ring buffer")
    a("    reg signed [31:0] vmm_queue [0:QDEPTH-1][0:NUM_COLS-1];")
    a("")
    a("    // ── LOAD sub-FSM ────────────────────────────────────────────────────")
    a("    reg [15:0] load_count;       // byte offset into input_buffer")
    a("    reg [15:0] load_cycle_idx;  // 0..LOAD_CYCLES-1 within current pass")
    a("    reg [3:0]  q_load_tail;      // next vmm_queue slot for VMM fire")
    a("")
    a("    // ── COMPUTE sub-FSM ─────────────────────────────────────────────────")
    a("    reg        compute_busy;")
    a("    reg [15:0] compute_cycle;    // 0..COMPUTE_CYCLES-1")
    a("    reg [3:0]  q_compute_head;")
    a("")
    a("    // ── OUTPUT sub-FSM ──────────────────────────────────────────────────")
    a("    reg        output_busy;")
    a("    reg [15:0] output_col_idx;   // 0..OUTPUT_CYCLES-1")
    a("    reg [3:0]  q_output_head;")
    a("")
    a("    // Combinational queue counts (mod QDEPTH).")
    a("    wire [3:0] n_pending_compute = (q_load_tail >= q_compute_head) ?")
    a("                                    (q_load_tail - q_compute_head) :")
    a("                                    (QDEPTH[3:0] + q_load_tail - q_compute_head);")
    a("    wire [3:0] n_pending_output  = (q_compute_head >= q_output_head) ?")
    a("                                    (q_compute_head - q_output_head) :")
    a("                                    (QDEPTH[3:0] + q_compute_head - q_output_head);")
    a("")
    a("    // Legacy state encoding (combinational, priority OUTPUT > COMPUTE > LOAD > IDLE).")
    a("    localparam S_IDLE      = 3'd0;")
    a("    localparam S_LOAD      = 3'd1;")
    a("    localparam S_COMPUTE   = 3'd3;")
    a("    localparam S_OUTPUT    = 3'd4;")
    a("    wire [2:0] state = output_busy ? S_OUTPUT :")
    a("                       compute_busy ? S_COMPUTE :")
    a("                       (w_buf_en || load_cycle_idx != 0) ? S_LOAD : S_IDLE;")
    a("")
    a("    // Init")
    a("    integer wi, wj, wq;")
    a("    initial begin")
    a("        for (wi = 0; wi < KERNEL_WIDTH; wi = wi + 1)")
    a("            for (wj = 0; wj < NUM_COLS; wj = wj + 1)")
    a("                weights[wi][wj] = 0;")
    a("        for (wq = 0; wq < QDEPTH; wq = wq + 1)")
    a("            for (wj = 0; wj < NUM_COLS; wj = wj + 1)")
    a("                vmm_queue[wq][wj] = 0;")
    a("    end")
    a("")
    a("    integer r, c, b;")
    a("")
    a("    always @(posedge clk or posedge reset) begin")
    a("        if (reset) begin")
    a("            load_count       <= 0;")
    a("            load_cycle_idx  <= 0;")
    a("            q_load_tail      <= 0;")
    a("            compute_busy     <= 0;")
    a("            compute_cycle    <= 0;")
    a("            q_compute_head   <= 0;")
    a("            output_busy      <= 0;")
    a("            output_col_idx   <= 0;")
    a("            q_output_head    <= 0;")
    a("            data_out         <= 0;")
    a("            dpe_done         <= 0;")
    a("            reg_full         <= 0;")
    a("            MSB_SA_Ready     <= 1;")
    a("            shift_add_done   <= 1;")
    a("            shift_add_bypass_ctrl <= 1;")
    a("        end else begin")
    a("            // ── LOAD sub-FSM ────────────────────────────────────────────")
    a("            if (w_buf_en) begin")
    a("                if (load_cycle_idx == LOAD_CYCLES - 1) begin")
    a("                    // Last strobe of pass: write final EPS bytes BLOCKING")
    a("                    // so the VMM math below reads the freshly-loaded final")
    a("                    // bytes (sim-only behavioural; no synthesised hardware).")
    a("                    for (b = 0; b < ELEMS_PER_STROBE; b = b + 1) begin")
    a("                        if (load_count + b < KERNEL_WIDTH)")
    a("                            input_buffer[load_count + b] = data_in[b*8 +: 8];")
    a("                    end")
    a("                    // Fire VMM into vmm_queue[q_load_tail].")
    a("                    for (c = 0; c < NUM_COLS; c = c + 1) begin")
    a("                        vmm_queue[q_load_tail][c] = 0;")
    a("                        for (r = 0; r < KERNEL_WIDTH; r = r + 1) begin")
    a("                            vmm_queue[q_load_tail][c] = vmm_queue[q_load_tail][c] +")
    a("                                input_buffer[r] * weights[r][c];")
    a("                        end")
    a("                    end")
    if has_acam:
        a("                    if (ACAM_MODE == 1) begin")
        a("                        for (c = 0; c < NUM_COLS; c = c + 1)")
        a("                            vmm_queue[q_load_tail][c] = 1 + vmm_queue[q_load_tail][c] +")
        a("                                (vmm_queue[q_load_tail][c] * vmm_queue[q_load_tail][c]) / 2;")
        a("                    end else if (ACAM_MODE == 2) begin")
        a("                        for (c = 0; c < NUM_COLS; c = c + 1)")
        a("                            vmm_queue[q_load_tail][c] = vmm_queue[q_load_tail][c] - 1;")
        a("                    end")
    a("                    load_count      <= 0;")
    a("                    load_cycle_idx <= 0;")
    a("                    q_load_tail     <= (q_load_tail + 1) % QDEPTH;")
    a("                end else begin")
    a("                    // Non-final strobe: NBA write.")
    a("                    for (b = 0; b < ELEMS_PER_STROBE; b = b + 1) begin")
    a("                        if (load_count + b < KERNEL_WIDTH)")
    a("                            input_buffer[load_count + b] <= data_in[b*8 +: 8];")
    a("                    end")
    a("                    load_count      <= load_count + ELEMS_PER_STROBE;")
    a("                    load_cycle_idx <= load_cycle_idx + 1;")
    a("                end")
    a("            end")
    a("")
    a("            // ── COMPUTE sub-FSM (Task #87 Phase 1: NBA-wake on queue) ──")
    a("            if (compute_busy) begin")
    a("                if (compute_cycle + 1 >= COMPUTE_CYCLES) begin")
    a("                    compute_busy     <= 0;")
    a("                    compute_cycle    <= 0;")
    a("                    q_compute_head   <= (q_compute_head + 1) % QDEPTH;")
    a("                    MSB_SA_Ready     <= 1;")
    a("                    shift_add_done   <= 1;")
    a("                end else begin")
    a("                    compute_cycle <= compute_cycle + 1;")
    a("                end")
    a("            end else begin")
    a("                // Wake one cycle after LOAD's last strobe: q_load_tail's NBA")
    a("                // must commit before n_pending_compute > 0 is observable.")
    a("                if (n_pending_compute > 0) begin")
    a("                    compute_busy     <= 1;")
    a("                    compute_cycle    <= 0;")
    a("                    MSB_SA_Ready     <= 0;")
    a("                    shift_add_done   <= 0;")
    a("                end")
    a("            end")
    a("")
    a("            // ── OUTPUT sub-FSM (Task #87 Phase 1: NBA-wake on queue) ───")
    a("            if (output_busy) begin")
    a("                data_out <= 0;")
    a("                for (b = 0; b < ELEMS_PER_STROBE; b = b + 1) begin")
    a("                    if (output_col_idx * ELEMS_PER_STROBE + b < NUM_COLS) begin")
    a("                        data_out[b*8 +: 8] <= vmm_queue[q_output_head][output_col_idx * ELEMS_PER_STROBE + b][7:0];")
    a("                    end")
    a("                end")
    a("                if (output_col_idx + 1 >= OUTPUT_CYCLES) begin")
    a("                    q_output_head  <= (q_output_head + 1) % QDEPTH;")
    a("                    output_col_idx <= 0;")
    a("                    // Chain to next pass when COMPUTE has already advanced")
    a("                    // its head past the current OUTPUT pass.  Without the")
    a("                    // chain, OUTPUT would idle one cycle between passes;")
    a("                    // with it, contiguous OUTPUT passes flow back-to-back.")
    a("                    if (n_pending_output > 1) begin")
    a("                        output_busy <= 1;")
    a("                    end else begin")
    a("                        output_busy <= 0;")
    a("                    end")
    a("                end else begin")
    a("                    output_col_idx <= output_col_idx + 1;")
    a("                end")
    a("            end else begin")
    a("                // Wake one cycle after COMPUTE done: q_compute_head's NBA")
    a("                // must commit before n_pending_output > 0 is observable.")
    a("                if (n_pending_output > 0) begin")
    a("                    output_busy    <= 1;")
    a("                    output_col_idx <= 0;")
    a("                end")
    a("            end")
    a("")
    a("            if (output_busy)")
    a("                dpe_done <= 1;")
    a("            else")
    a("                dpe_done <= 0;")
    a("")
    a("            if (compute_busy || output_busy)")
    a("                reg_full <= 1;")
    a("            else")
    a("                reg_full <= 0;")
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
          f"L={params['load_cycles']} O={params['output_cycles']} "
          f"has_acam={params['has_acam']}")
    print(f"               CCYC = PRECISION({params['precision_bits']}) "
          f"+ (PIPELINE_DEPTH({params['pipeline_depth']}) - 1) "
          f"+ ACAM_CYCLES({params['acam_cycles']}) "
          f"= {params['compute_cycles_default']}")
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
