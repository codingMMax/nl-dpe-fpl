#!/usr/bin/env python3
"""
Generator for the FAITHFUL NL-DPE behavior model (Task #91 + silicon-faithful refactor).

Reads `archive/azurelily_simulator/IMC/configs/nl_dpe.json` for arch parameters
(KERNEL_WIDTH, NUM_COLS, DPE_BUF_WIDTH, PRECISION_BITS, capabilities.
pipeline_depth, capabilities.acam_cycles) and emits

    fc_verification/rtl/dpe_nldpe_faithful.v

Module name `dpe` matches the VTR arch XML `<model name="dpe">`
blackbox port contract (same as the lazy primitive).

DIFFERENCE FROM `gen_dpe_stub.py`:

* Lazy primitive  (gen_dpe_stub.py)   - COMPUTE_CYCLES is a Verilog
                                         parameter that BURNS that many
                                         cycles after a single-posedge
                                         VMM fire. Cycle count is set.
* Faithful prim.  (this generator)    - NO `COMPUTE_CYCLES` parameter.
                                         PRECISION, PIPELINE_DEPTH and
                                         ACAM_CYCLES are baked in;
                                         compute cycle count EMERGES
                                         from advancing bit_idx 0..P-1
                                         (1 cycle each), then one
                                         cycle of pipeline drain (last-
                                         bit Acc commit), then 1 cycle
                                         of ACAM read-out.

SILICON-FAITHFUL REFACTOR (Task #93 + Task #99 double-buffer):

* Double-buffered input substrate (Task #99):
    - input_buf_slice_a / input_buf_slice_b [PRECISION][R] ping-pong pair,
      each bit-stratified slice-major. load_phase selects the LOAD-side;
      compute_phase (captured at COMPUTE wake-time as ~load_phase) selects
      the COMPUTE-side. LOAD-pass-(k+1) starts back-to-back after
      pass-k's last strobe (no LOAD-gate, no inter-pass stall).
* Single-buffered downstream substrates:
    - mac_acc         [NUM_COLS]     single-buffered.
    - acam_out        [NUM_COLS]     single-buffered.
* LOAD uses corner-turn: each strobe writes EPS rows of ALL PRECISION
  bit positions in parallel into the load_phase-selected substrate.
* COMPUTE bit-serial sweep reads `slice[compute_phase][bit_idx_s0][r]`
  directly via a 1-bit conditional.
* Sub-FSM coordination via `buf_loaded` and `compute_done` flags
  (replacing legacy q_load_tail / q_compute_head / q_output_head
  pointers). Cycle-by-cycle handoffs (one NBA cycle each) preserved.

The generator is intended as a single-source-of-truth-from-JSON
companion to `gen_dpe_stub.py`. Running it emits an RTL file that is
functionally and structurally equivalent to the hand-written
`dpe_nldpe_faithful.v`. The hand-written file remains the canonical
reference; this generator can be used to re-emit it from JSON when the
arch parameters change.

Usage:
    python nl_dpe/gen_dpe_nldpe_faithful.py
    python nl_dpe/gen_dpe_nldpe_faithful.py --config archive/azurelily_simulator/IMC/configs/nl_dpe.json
    python nl_dpe/gen_dpe_nldpe_faithful.py --out-dir /path/to/some/dir
"""

import argparse
import json
import math
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_CFG_PATH = os.path.join(REPO, "archive/azurelily_simulator/IMC/configs/nl_dpe.json")
DEFAULT_OUT_DIR = os.path.join(REPO, "fc_verification/rtl")


def derive_arch_params(cfg_path):
    """Load NL-DPE config JSON and return RTL parameters."""
    with open(cfg_path, "r") as fh:
        cfg = json.load(fh)
    if cfg.get("core_name") != "NL-DPE":
        raise ValueError(
            f"{cfg_path}: core_name='{cfg.get('core_name')}' "
            "is not 'NL-DPE'; this generator emits the NL-DPE faithful primitive only."
        )

    geometry = cfg.get("geometry", {})
    capabilities = cfg.get("capabilities", {})
    fpga_specs = cfg.get("fpga_specs", {})

    R = int(geometry["array_rows"])
    C = int(geometry["array_cols"])
    BUF = int(fpga_specs["dpe_buf_width"])
    precision_bits = int(cfg.get("precision_bits", 8))
    pipeline_depth = int(capabilities.get("pipeline_depth", 2))
    acam_cycles = int(capabilities.get("acam_cycles", 1))
    has_acam = bool(capabilities.get("analog_nonlinear", True))
    if not has_acam:
        raise ValueError(
            f"{cfg_path}: capabilities.analog_nonlinear=False "
            "but this generator emits the NL-DPE primitive with ACAM."
        )

    eps = BUF // 8
    assert eps >= 1, f"BUF={BUF} < 8 not supported"
    load_cycles = math.ceil(R * 8 / BUF)
    output_cycles = math.ceil(C * 8 / BUF)
    # Declared CCYC (not used by RTL -- emergent in the RTL, but written
    # into the header comment for documentation).
    ccyc_declared = precision_bits + (pipeline_depth - 1) + acam_cycles

    return {
        "R": R,
        "C": C,
        "BUF": BUF,
        "precision_bits": precision_bits,
        "pipeline_depth": pipeline_depth,
        "acam_cycles": acam_cycles,
        "load_cycles": load_cycles,
        "output_cycles": output_cycles,
        "ccyc_declared": ccyc_declared,
        "core_name": "NL-DPE",
    }


HEADER_TEMPLATE = """\
// dpe_nldpe_faithful.v  --  FAITHFUL behavioral model of the NL-DPE primitive.
// Generated by nl_dpe/gen_dpe_nldpe_faithful.py from {cfg_path}.
// Module name : dpe   (matches VTR arch XML <model name="dpe"> contract).
// File arch   : NL-DPE (faithful track -- Task #91, silicon-faithful refactor)
//
// Architecture parameters (single source of truth: the JSON config):
//   R (KERNEL_WIDTH)       = {R}
//   C (NUM_COLS)           = {C}
//   BUF (DPE_BUF_WIDTH)    = {BUF}    (elems_per_strobe = {eps})
//   PRECISION              = {P}      (INT{P})
//   PIPELINE_DEPTH         = {PD}     (Crossbar -> Acc; 2-stage)
//   ACAM_CYCLES            = {AC}     (1-cycle read-out)
//
// Derived per-pass cycle counters:
//   LOAD_CYCLES   = ceil(R * 8 / BUF) = {LCYC}
//   OUTPUT_CYCLES = ceil(C * 8 / BUF) = {OCYC}
//   CCYC          = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES
//                 = {P} + {PDm1} + {AC} = {CCYC}    [emergent in RTL]
//
// ┌────────────────────────────────────────────────────────────────────┐
// │ DESIGN CHARTER -- silicon-faithful DOUBLE-BUFFERED input substrate │
// │  (Task #99: double-buffer refactor; LOAD-gate eliminated.)         │
// │                                                                    │
// │  Substrates:                                                       │
// │    input_buf_slice_a / input_buf_slice_b -- ping-pong pair         │
// │      (each [PRECISION][R] bit-stratified slice-major).             │
// │    mac_acc                              -- single-buffered MAC.    │
// │    acam_out                             -- single-buffered output. │
// │                                                                    │
// │  Multi-pass overlap is now back-pressure-free on LOAD:             │
// │  LOAD writes to substrate selected by `load_phase`; COMPUTE reads  │
// │  the other substrate (captured at wake-time as `compute_phase`).   │
// │  LOAD-pass-(k+1) can begin IMMEDIATELY after pass-k's last strobe  │
// │  -- it targets the substrate just vacated by pass-k's COMPUTE.     │
// │                                                                    │
// │  Area cost: +1 extra input substrate (PRECISION*R single-bit       │
// │  flops per DPE). Accounted for at the architectural area budget    │
// │  level (out of scope here).                                        │
// │                                                                    │
// │  EMERGENT CYCLE COUNT (NOT a parameter):                           │
// │     CCYC      = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES     │
// │               = {CCYC} cycles (INT{P}, NL-DPE).                    │
// │     T_steady  = max(LCYC, CCYC, OCYC)                              │
// │               = LCYC for typical configs (LCYC >> CCYC).           │
// │     LOAD-pass-(k+1) launches RIGHT AFTER pass-k's last LOAD strobe.│
// │                                                                    │
// │  Assumption: LCYC >> CCYC. COMPUTE finishes well before LOAD       │
// │  finishes the next pass; no back-pressure on the LOAD side.        │
// │  The OUTPUT substrate is still single-buffered, so pass-(k+1)'s    │
// │  COMPUTE waits (via the buf_loaded handshake) for pass-k's COMPUTE │
// │  to drain.                                                          │
// └────────────────────────────────────────────────────────────────────┘
"""


def emit_faithful_rtl(params, cfg_path):
    """Emit the faithful primitive RTL source as a string."""
    header = HEADER_TEMPLATE.format(
        cfg_path=cfg_path,
        R=params["R"],
        C=params["C"],
        BUF=params["BUF"],
        eps=params["BUF"] // 8,
        P=params["precision_bits"],
        PD=params["pipeline_depth"],
        AC=params["acam_cycles"],
        LCYC=params["load_cycles"],
        OCYC=params["output_cycles"],
        CCYC=params["ccyc_declared"],
        PDm1=params["pipeline_depth"] - 1,
    )

    R = params["R"]
    C = params["C"]
    BUF = params["BUF"]
    P = params["precision_bits"]
    PD = params["pipeline_depth"]
    AC = params["acam_cycles"]

    body = f"""//
// Physical pipeline model (declared at the architecture level --
// the RTL realises it; the TB measures it):
//
//   LOAD (corner-turn)
//        |
//        v
//   input_buf_slice_{{a,b}}[bit][row]  <-- ping-pong substrate pair
//   (LOAD writes whichever load_phase selects; COMPUTE reads the OTHER)
//        |
//        v
//   Stage 0 (combinational + REG, crossbar fire on bit_idx_s0)
//     crossbar_sum_comb[c] = sum_r (slice[compute_phase][bit_idx_s0][r]
//                                   ? weights[r][c] : 0)
//     -> latched into crossbar_sum_reg[c]
//        |
//        v
//   Stage 1 (registered, signed accumulator on bit_idx_s1)
//     if (bit_idx_s1 == {P-1}):  // MSB subtract
//          mac_acc[c] <= mac_acc[c] - (crossbar_sum_reg[c] <<< bit_idx_s1)
//     else
//          mac_acc[c] <= mac_acc[c] + (crossbar_sum_reg[c] <<< bit_idx_s1)
//        |
//        v
//   Stage 2 (ACAM, registered, 1 cycle, mode-dependent)
//        mode 0 (ADC/identity): acam_out[c] <= mac_acc[c]
//        mode 1 (exp):          acam_out[c] <= 1 + mac_acc[c]
//                                + (mac_acc[c] * mac_acc[c]) >>> 1
//        mode 2 (log):          acam_out[c] <= mac_acc[c] - 1
//        |
//        v
//   OUTPUT drain (single-substrate acam_out -> byte-streamed)
//
// Module name `dpe` (NOT `dpe_faithful`) keeps the VTR arch XML
// blackbox port contract.

module dpe #(
    parameter KERNEL_WIDTH   = {R},
    parameter NUM_COLS       = {C},
    parameter DPE_BUF_WIDTH  = {BUF},
    parameter PRECISION      = {P},
    parameter PIPELINE_DEPTH = {PD},
    parameter ACAM_CYCLES    = {AC},
    parameter ACAM_MODE      = 0
)(
    input  wire                       clk,
    input  wire                       reset,
    input  wire [DPE_BUF_WIDTH-1:0]   data_in,
    input  wire [1:0]                 nl_dpe_control,
    input  wire                       shift_add_control,
    input  wire                       w_buf_en,
    input  wire                       shift_add_bypass,
    input  wire                       load_output_reg,
    input  wire                       load_input_reg,
    output reg                        MSB_SA_Ready,
    output reg  [DPE_BUF_WIDTH-1:0]   data_out,
    output reg                        dpe_done,
    output reg                        reg_full,
    output reg                        shift_add_done,
    output reg                        shift_add_bypass_ctrl
);

    wire _unused_aux = shift_add_control | shift_add_bypass |
                       load_output_reg   | load_input_reg   |
                       (|nl_dpe_control);

    localparam ELEMS_PER_STROBE = DPE_BUF_WIDTH / 8;
    localparam LOAD_CYCLES      = (KERNEL_WIDTH + ELEMS_PER_STROBE - 1) / ELEMS_PER_STROBE;
    localparam OUTPUT_CYCLES    = (NUM_COLS    + ELEMS_PER_STROBE - 1) / ELEMS_PER_STROBE;

    reg signed [7:0]  weights      [0:KERNEL_WIDTH-1][0:NUM_COLS-1];
    // Double-buffered input substrate in slice-major form (bit-stratified).
    // LOAD writes to A or B based on load_phase; COMPUTE reads from the
    // other (latched as compute_phase at wake-time).
    reg input_buf_slice_a [0:PRECISION-1][0:KERNEL_WIDTH-1];
    reg input_buf_slice_b [0:PRECISION-1][0:KERNEL_WIDTH-1];
    reg signed [31:0] mac_acc          [0:NUM_COLS-1];
    reg signed [31:0] crossbar_sum_reg [0:NUM_COLS-1];
    reg signed [31:0] acam_out         [0:NUM_COLS-1];

    // LOAD sub-FSM
    reg [15:0] load_cycle_cnt;

    // COMPUTE sub-FSM
    reg        compute_busy;
    reg [4:0]  bit_idx_s0;
    reg [4:0]  bit_idx_s1;
    reg        s1_valid;
    reg        acam_fire;
    reg        compute_first_bit_pulse;
    reg        acam_commit_pulse;

    // OUTPUT sub-FSM
    reg        output_busy;
    reg [15:0] output_col_idx;

    // Sub-FSM coordination (replaces ring-buffer pointers).
    reg buf_loaded;
    reg compute_done;
    // Double-buffer phase selectors (Task #99). load_phase selects which
    // substrate LOAD writes; compute_phase selects which substrate
    // COMPUTE reads (captured at COMPUTE wake-time as ~load_phase, since
    // LOAD toggles load_phase to the NEXT substrate at end of each pass).
    reg load_phase;
    reg compute_phase;

    localparam S_IDLE    = 3'd0;
    localparam S_LOAD    = 3'd1;
    localparam S_COMPUTE = 3'd3;
    localparam S_OUTPUT  = 3'd4;
    wire [2:0] state = output_busy ? S_OUTPUT :
                       compute_busy ? S_COMPUTE :
                       (w_buf_en || load_cycle_cnt != 0) ? S_LOAD : S_IDLE;

    integer r_idx, c_idx;
    reg signed [31:0] crossbar_sum_comb [0:NUM_COLS-1];
    // Combinational pick of the COMPUTE-side substrate slice cell.
    // (Verilog-2005 has no clean way to index a packed-array dimension by
    // a variable, so we expand the per-row read via a conditional.)
    reg slice_bit;
    always @* begin
        for (c_idx = 0; c_idx < NUM_COLS; c_idx = c_idx + 1)
            crossbar_sum_comb[c_idx] = 32'sd0;
        if (compute_busy && bit_idx_s0 < PRECISION[4:0]) begin
            for (r_idx = 0; r_idx < KERNEL_WIDTH; r_idx = r_idx + 1) begin
                slice_bit = (compute_phase == 1'b0)
                              ? input_buf_slice_a[bit_idx_s0][r_idx]
                              : input_buf_slice_b[bit_idx_s0][r_idx];
                if (slice_bit) begin
                    for (c_idx = 0; c_idx < NUM_COLS; c_idx = c_idx + 1) begin
                        crossbar_sum_comb[c_idx] = crossbar_sum_comb[c_idx]
                            + {{{{24{{weights[r_idx][c_idx][7]}}}}, weights[r_idx][c_idx]}};
                    end
                end
            end
        end
    end

    integer wi, wj, wb;
    initial begin
        for (wi = 0; wi < KERNEL_WIDTH; wi = wi + 1)
            for (wj = 0; wj < NUM_COLS; wj = wj + 1)
                weights[wi][wj] = 0;
        for (wb = 0; wb < PRECISION; wb = wb + 1)
            for (wj = 0; wj < KERNEL_WIDTH; wj = wj + 1) begin
                input_buf_slice_a[wb][wj] = 1'b0;
                input_buf_slice_b[wb][wj] = 1'b0;
            end
        for (wj = 0; wj < NUM_COLS; wj = wj + 1) begin
            mac_acc[wj]          = 0;
            acam_out[wj]         = 0;
            crossbar_sum_reg[wj] = 0;
        end
    end

    integer cc, bb, ib;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            load_cycle_cnt         <= 0;
            buf_loaded             <= 0;
            compute_busy           <= 0;
            bit_idx_s0             <= 0;
            bit_idx_s1             <= 0;
            s1_valid               <= 0;
            acam_fire              <= 0;
            compute_first_bit_pulse<= 0;
            acam_commit_pulse      <= 0;
            compute_done           <= 0;
            output_busy            <= 0;
            output_col_idx         <= 0;
            data_out               <= 0;
            reg_full               <= 0;
            MSB_SA_Ready           <= 1;
            shift_add_done         <= 1;
            shift_add_bypass_ctrl  <= 1;
            dpe_done               <= 0;
            load_phase             <= 1'b0;
            compute_phase          <= 1'b0;
        end else begin
            compute_first_bit_pulse <= 0;
            acam_commit_pulse       <= 0;

            // LOAD sub-FSM (corner-turn into bit-stratified slice).
            // Task #99 double-buffer: write to substrate selected by
            // load_phase. No LOAD-gate; pass-(k+1) writes IMMEDIATELY
            // begin after pass-k's last strobe -- targeting the other
            // substrate, which pass-k's COMPUTE has already vacated
            // (or will vacate well before LOAD wraps, since LCYC >> CCYC).
            if (w_buf_en) begin
                for (bb = 0; bb < ELEMS_PER_STROBE; bb = bb + 1) begin
                    if (load_cycle_cnt * ELEMS_PER_STROBE + bb < KERNEL_WIDTH) begin
                        for (ib = 0; ib < PRECISION; ib = ib + 1) begin
                            if (load_phase == 1'b0)
                                input_buf_slice_a[ib][load_cycle_cnt * ELEMS_PER_STROBE + bb]
                                    <= data_in[bb*8 + ib];
                            else
                                input_buf_slice_b[ib][load_cycle_cnt * ELEMS_PER_STROBE + bb]
                                    <= data_in[bb*8 + ib];
                        end
                    end
                end
                if (load_cycle_cnt == LOAD_CYCLES - 1) begin
                    load_cycle_cnt <= 0;
                    buf_loaded     <= 1;
                    // Toggle load_phase so the NEXT pass's LOAD targets
                    // the OTHER substrate.
                    load_phase     <= ~load_phase;
                end else begin
                    load_cycle_cnt <= load_cycle_cnt + 1;
                end
            end

            // COMPUTE sub-FSM.
            if (compute_busy) begin
                if (bit_idx_s0 < PRECISION[4:0]) begin
                    for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                        crossbar_sum_reg[cc] <= crossbar_sum_comb[cc];
                    bit_idx_s0 <= bit_idx_s0 + 5'd1;
                end

                if (s1_valid && bit_idx_s1 < PRECISION[4:0]) begin
                    if (bit_idx_s1 == (PRECISION[4:0] - 5'd1)) begin
                        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                            mac_acc[cc] <= mac_acc[cc]
                                           - (crossbar_sum_reg[cc] <<< bit_idx_s1);
                        acam_fire <= 1;
                    end else begin
                        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                            mac_acc[cc] <= mac_acc[cc]
                                           + (crossbar_sum_reg[cc] <<< bit_idx_s1);
                    end
                    bit_idx_s1 <= bit_idx_s1 + 5'd1;
                end

                if (!s1_valid && bit_idx_s0 == 5'd0) begin
                    s1_valid <= 1;
                end

                if (acam_fire) begin
                    if (ACAM_MODE == 0) begin
                        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                            acam_out[cc] <= mac_acc[cc];
                    end else if (ACAM_MODE == 1) begin
                        for (cc = 0; cc < NUM_COLS; cc = cc + 1) begin
                            acam_out[cc] <= 32'sd1
                                + mac_acc[cc]
                                + ((mac_acc[cc] * mac_acc[cc]) >>> 1);
                        end
                    end else if (ACAM_MODE == 2) begin
                        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                            acam_out[cc] <= mac_acc[cc] - 32'sd1;
                    end
                    acam_commit_pulse <= 1;
                    acam_fire         <= 0;
                    compute_busy   <= 0;
                    bit_idx_s0     <= 0;
                    bit_idx_s1     <= 0;
                    s1_valid       <= 0;
                    compute_done   <= 1;
                    MSB_SA_Ready   <= 1;
                    shift_add_done <= 1;
                end
            end else begin
                if (buf_loaded) begin
                    compute_busy            <= 1;
                    bit_idx_s0              <= 0;
                    bit_idx_s1              <= 0;
                    s1_valid                <= 0;
                    acam_fire               <= 0;
                    compute_first_bit_pulse <= 1;
                    MSB_SA_Ready            <= 0;
                    shift_add_done          <= 0;
                    buf_loaded              <= 0;
                    // Double-buffer (Task #99): COMPUTE wakes on the
                    // substrate that LOAD JUST FILLED. The LOAD path has
                    // already NBA-toggled load_phase to point at the
                    // NEXT (other) substrate; the just-filled substrate
                    // is therefore ~load_phase from the COMPUTE side.
                    compute_phase           <= ~load_phase;
                    for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                        mac_acc[cc] <= 32'sd0;
                end
            end

            // OUTPUT sub-FSM.
            if (output_busy) begin
                data_out <= 0;
                for (bb = 0; bb < ELEMS_PER_STROBE; bb = bb + 1) begin
                    if (output_col_idx * ELEMS_PER_STROBE + bb < NUM_COLS) begin
                        data_out[bb*8 +: 8] <=
                            acam_out[output_col_idx * ELEMS_PER_STROBE + bb][7:0];
                    end
                end
                if (output_col_idx + 1 >= OUTPUT_CYCLES) begin
                    output_col_idx <= 0;
                    if (compute_done) begin
                        output_busy <= 1;
                    end else begin
                        output_busy <= 0;
                    end
                end else begin
                    output_col_idx <= output_col_idx + 1;
                end
            end else begin
                if (compute_done) begin
                    output_busy    <= 1;
                    output_col_idx <= 0;
                    compute_done   <= 0;
                end
            end

            if (output_busy && output_col_idx + 1 >= OUTPUT_CYCLES && compute_done) begin
                compute_done <= 0;
            end

            if (output_busy)
                dpe_done <= 1;
            else
                dpe_done <= 0;

            // reg_full asserts when the producer (TB / fc_top wrapper)
            // must back off. Under the Task #99 double-buffered design,
            // LOAD is never gated by COMPUTE (the two substrates are
            // independent), so reg_full is driven only by the OUTPUT
            // drain stage (single-substrate acam_out). LCYC >> CCYC for
            // our typical configs, so back-pressure from OUTPUT is also
            // not expected in steady state, but the wire is retained
            // for observability / safety.
            if (output_busy)
                reg_full <= 1;
            else
                reg_full <= 0;
        end
    end

endmodule
"""
    return header + body


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CFG_PATH,
                   help=f"NL-DPE JSON config path (default: {DEFAULT_CFG_PATH})")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--check", action="store_true",
                   help="Print the emitted RTL to stdout but don't write the file")
    args = p.parse_args(argv)

    params = derive_arch_params(args.config)
    src = emit_faithful_rtl(params, args.config)

    if args.check:
        sys.stdout.write(src)
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "dpe_nldpe_faithful.v")
    with open(out_path, "w") as fh:
        fh.write(src)
    print(f"[gen_dpe_nldpe_faithful] wrote {out_path}")
    print(f"               R={params['R']} C={params['C']} BUF={params['BUF']} "
          f"PRECISION={params['precision_bits']} PIPELINE_DEPTH={params['pipeline_depth']} "
          f"ACAM_CYCLES={params['acam_cycles']}")
    print(f"               LCYC={params['load_cycles']} "
          f"OCYC={params['output_cycles']} "
          f"CCYC(declared, emergent in RTL)={params['ccyc_declared']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
