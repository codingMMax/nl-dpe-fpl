#!/usr/bin/env python3
"""
Generator for the FAITHFUL Azure-Lily DPE behavior model (Task #92 + silicon-faithful refactor).

Reads `archive/azurelily_simulator/IMC/configs/azure_lily.json` for arch parameters
(KERNEL_WIDTH, NUM_COLS, DPE_BUF_WIDTH, PRECISION_BITS, capabilities.
pipeline_depth, capabilities.acam_cycles) and emits

    rtl_flow/rtl/dpe_azurelily_faithful.v

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
                                         (1 cycle each) through 3
                                         internal stages (Crossbar +
                                         ADC + ShiftAdd), then 0
                                         cycles of ACAM read-out (AL
                                         has no ACAM).

DIFFERENCE FROM `gen_dpe_nldpe_faithful.py`:

* NL faithful   : 2-stage pipeline (Crossbar + Acc) + 1-cycle ACAM
                  read-out stage (3 modes: ADC identity, exp, log).
* AL faithful   : 3-stage pipeline (Crossbar + ADC + ShiftAdd) + NO
                  ACAM stage. AL is purely VMM; nonlinearity must be
                  done by CLB activation downstream.

Both yield CCYC = PRECISION + 2 for INT8 by structural symmetry, but
the physical decomposition is different.

SILICON-FAITHFUL REFACTOR (Task #93 + Task #99 double-buffer):

* Double-buffered input substrate (Task #99):
    - input_buf_slice_a / input_buf_slice_b [PRECISION][R] ping-pong pair,
      each bit-stratified slice-major. load_phase selects the LOAD-side;
      compute_phase (captured at COMPUTE wake-time as ~load_phase) selects
      the COMPUTE-side. LOAD-pass-(k+1) starts back-to-back after
      pass-k's last strobe (no LOAD-gate, no inter-pass stall).
* Single-buffered downstream substrate:
    - mac_acc         [NUM_COLS]     single-buffered (AL drains mac_acc
                                     directly; no separate acam_out).
* LOAD uses corner-turn: each strobe writes EPS rows of ALL PRECISION
  bit positions in parallel into the load_phase-selected substrate.
* COMPUTE bit-serial sweep reads `slice[compute_phase][bit_idx_s0][r]`
  directly via a 1-bit conditional.
* Sub-FSM coordination via `buf_loaded` and `compute_done` flags
  (replacing legacy q_load_tail / q_compute_head / q_output_head
  pointers).

The generator is intended as a single-source-of-truth-from-JSON
companion to `gen_dpe_stub.py`. Running it emits an RTL file that is
functionally and structurally equivalent to the hand-written
`dpe_azurelily_faithful.v`. The hand-written file remains the
canonical reference; this generator can be used to re-emit it from
JSON when the arch parameters change.

Usage:
    python nl_dpe/gen_dpe_azurelily_faithful.py
    python nl_dpe/gen_dpe_azurelily_faithful.py --config archive/azurelily_simulator/IMC/configs/azure_lily.json
    python nl_dpe/gen_dpe_azurelily_faithful.py --out-dir /path/to/some/dir
"""

import argparse
import json
import math
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_CFG_PATH = os.path.join(REPO, "rtl_flow/specs/azure_lily.json")
DEFAULT_OUT_DIR = os.path.join(REPO, "rtl_flow/rtl")


def derive_arch_params(cfg_path):
    """Load Azure-Lily config JSON and return RTL parameters."""
    with open(cfg_path, "r") as fh:
        cfg = json.load(fh)
    if cfg.get("core_name") != "Azure-Lily":
        raise ValueError(
            f"{cfg_path}: core_name='{cfg.get('core_name')}' "
            "is not 'Azure-Lily'; this generator emits the Azure-Lily faithful primitive only."
        )

    geometry = cfg.get("geometry", {})
    capabilities = cfg.get("capabilities", {})
    fpga_specs = cfg.get("fpga_specs", {})

    R = int(geometry["array_rows"])
    C = int(geometry["array_cols"])
    BUF = int(fpga_specs["dpe_buf_width"])
    precision_bits = int(cfg.get("precision_bits", 8))
    pipeline_depth = int(capabilities.get("pipeline_depth", 3))
    acam_cycles = int(capabilities.get("acam_cycles", 0))
    has_acam = bool(capabilities.get("analog_nonlinear", False))
    if has_acam:
        raise ValueError(
            f"{cfg_path}: capabilities.analog_nonlinear=True "
            "but this generator emits the Azure-Lily primitive WITHOUT ACAM."
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
        "core_name": "Azure-Lily",
    }


HEADER_TEMPLATE = """\
// dpe_azurelily_faithful.v -- FAITHFUL behavioral model of the Azure-Lily DPE primitive.
// Generated by nl_dpe/gen_dpe_azurelily_faithful.py from {cfg_path}.
// Module name : dpe   (matches VTR arch XML <model name="dpe"> contract).
// File arch   : Azure-Lily (faithful track -- Task #92, silicon-faithful refactor)
//
// Architecture parameters (single source of truth: the JSON config):
//   R (KERNEL_WIDTH)       = {R}
//   C (NUM_COLS)           = {C}
//   BUF (DPE_BUF_WIDTH)    = {BUF}    (elems_per_strobe = {eps})
//   PRECISION              = {P}      (INT{P})
//   PIPELINE_DEPTH         = {PD}     (Crossbar -> ADC -> ShiftAdd; 3-stage)
//   ACAM_CYCLES            = {AC}     (Azure-Lily has NO ACAM)
//
// Derived per-pass cycle counters:
//   LOAD_CYCLES   = ceil(R * 8 / BUF) = {LCYC}
//   OUTPUT_CYCLES = ceil(C * 8 / BUF) = {OCYC}
//   CCYC          = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES
//                 = {P} + {PDm1} + {AC} = {CCYC}    [emergent in RTL]
//
// +--------------------------------------------------------------------+
// | DESIGN CHARTER -- silicon-faithful DOUBLE-BUFFERED input substrate |
// |  (Task #99: double-buffer refactor; LOAD-gate eliminated.)         |
// |                                                                    |
// |  Substrates:                                                       |
// |    input_buf_slice_a / input_buf_slice_b -- ping-pong pair         |
// |      (each [PRECISION][R] bit-stratified slice-major).             |
// |    mac_acc                              -- single-buffered (AL     |
// |                                            drains mac_acc as the   |
// |                                            OUTPUT substrate).      |
// |                                                                    |
// |  Multi-pass overlap is now back-pressure-free on LOAD:             |
// |  LOAD writes to substrate selected by `load_phase`; COMPUTE reads  |
// |  the other substrate (captured at wake-time as `compute_phase`).   |
// |  LOAD-pass-(k+1) can begin IMMEDIATELY after pass-k's last strobe  |
// |  -- it targets the substrate just vacated by pass-k's COMPUTE.     |
// |                                                                    |
// |  Area cost: +1 extra input substrate (PRECISION*R single-bit       |
// |  flops per DPE). Accounted for at the architectural area budget    |
// |  level (out of scope here).                                        |
// |                                                                    |
// |  EMERGENT CYCLE COUNT (NOT a parameter):                           |
// |     CCYC      = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES     |
// |               = {CCYC} cycles (INT{P}, Azure-Lily).                |
// |     T_steady  = max(LCYC, CCYC, OCYC)                              |
// |               = LCYC for typical AL configs (LCYC=256 >> CCYC=10). |
// |     LOAD-pass-(k+1) launches RIGHT AFTER pass-k's last LOAD strobe.|
// |                                                                    |
// |  Assumption: LCYC >> CCYC. For AL (LCYC=256 CCYC=10), COMPUTE      |
// |  finishes well before LOAD finishes the next pass, so no back-     |
// |  pressure is needed on the LOAD side. The OUTPUT substrate         |
// |  (mac_acc) is still single-buffered, so pass-(k+1)'s COMPUTE waits |
// |  (via the buf_loaded handshake) for pass-k's COMPUTE to drain.     |
// |                                                                    |
// |  Same total CCYC as NL faithful (8 + 1 + 1 = 10), different        |
// |  physical decomposition: AL trades ACAM for an extra ADC stage.    |
// +--------------------------------------------------------------------+
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
//   Stage 1 (registered, ADC sample/quantize on bit_idx_s1)
//     adc_reg[c] <= crossbar_sum_reg[c]
//        |
//        v
//   Stage 2 (registered, shift-add accumulator on bit_idx_s2)
//     if (bit_idx_s2 == {P-1}):  // MSB subtract
//          mac_acc[c] <= mac_acc[c] - (adc_reg[c] <<< bit_idx_s2)
//     else
//          mac_acc[c] <= mac_acc[c] + (adc_reg[c] <<< bit_idx_s2)
//        |
//        v
//        NO ACAM stage in Azure-Lily!
//        |
//        v
//   OUTPUT drain (single-substrate mac_acc -> byte-streamed)
//
// Module name `dpe` (NOT `dpe_faithful`) keeps the VTR arch XML
// blackbox port contract.

module dpe #(
    parameter KERNEL_WIDTH   = {R},
    parameter NUM_COLS       = {C},
    parameter DPE_BUF_WIDTH  = {BUF},
    parameter PRECISION      = {P},
    parameter PIPELINE_DEPTH = {PD},
    parameter ACAM_CYCLES    = {AC}
    // NO ACAM_MODE -- Azure-Lily has no ACAM nonlinear stage.
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
    // Double-buffered input substrate (Task #99 ping-pong pair).
    reg input_buf_slice_a [0:PRECISION-1][0:KERNEL_WIDTH-1];
    reg input_buf_slice_b [0:PRECISION-1][0:KERNEL_WIDTH-1];
    reg signed [31:0] mac_acc          [0:NUM_COLS-1];
    reg signed [31:0] crossbar_sum_reg [0:NUM_COLS-1];
    reg signed [31:0] adc_reg          [0:NUM_COLS-1];

    reg [15:0] load_cycle_cnt;

    reg        compute_busy;
    reg [4:0]  bit_idx_s0;
    reg [4:0]  bit_idx_s1;
    reg [4:0]  bit_idx_s2;
    reg        s1_valid;
    reg        s2_valid;
    reg        compute_first_bit_pulse;
    reg        compute_last_bit_pulse;

    reg        output_busy;
    reg [15:0] output_col_idx;

    reg buf_loaded;
    reg compute_done;
    // Double-buffer phase selectors (Task #99). See dpe_nldpe_faithful.v
    // for the design rationale; same mechanism applies here.
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
            crossbar_sum_reg[wj] = 0;
            adc_reg[wj]          = 0;
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
            bit_idx_s2             <= 0;
            s1_valid               <= 0;
            s2_valid               <= 0;
            compute_first_bit_pulse<= 0;
            compute_last_bit_pulse <= 0;
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
            compute_last_bit_pulse  <= 0;

            // LOAD sub-FSM (corner-turn into bit-stratified slice).
            // Task #99 double-buffer: write to substrate selected by
            // load_phase. No LOAD-gate; pass-(k+1) writes IMMEDIATELY
            // begin after pass-k's last strobe -- targeting the other
            // substrate.
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
                    load_phase     <= ~load_phase;
                end else begin
                    load_cycle_cnt <= load_cycle_cnt + 1;
                end
            end

            // COMPUTE sub-FSM (3 stages).
            if (compute_busy) begin
                if (bit_idx_s0 < PRECISION[4:0]) begin
                    for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                        crossbar_sum_reg[cc] <= crossbar_sum_comb[cc];
                    bit_idx_s0 <= bit_idx_s0 + 5'd1;
                end

                if (s1_valid && bit_idx_s1 < PRECISION[4:0]) begin
                    for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                        adc_reg[cc] <= crossbar_sum_reg[cc];
                    bit_idx_s1 <= bit_idx_s1 + 5'd1;
                end

                if (s2_valid && bit_idx_s2 < PRECISION[4:0]) begin
                    if (bit_idx_s2 == (PRECISION[4:0] - 5'd1)) begin
                        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                            mac_acc[cc] <= mac_acc[cc]
                                           - (adc_reg[cc] <<< bit_idx_s2);
                        compute_last_bit_pulse <= 1;
                        compute_busy   <= 0;
                        bit_idx_s0     <= 0;
                        bit_idx_s1     <= 0;
                        bit_idx_s2     <= 0;
                        s1_valid       <= 0;
                        s2_valid       <= 0;
                        compute_done   <= 1;
                        MSB_SA_Ready   <= 1;
                        shift_add_done <= 1;
                    end else begin
                        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                            mac_acc[cc] <= mac_acc[cc]
                                           + (adc_reg[cc] <<< bit_idx_s2);
                    end
                    bit_idx_s2 <= bit_idx_s2 + 5'd1;
                end

                if (!s1_valid && bit_idx_s0 == 5'd0) begin
                    s1_valid <= 1;
                end

                if (s1_valid && !s2_valid && bit_idx_s1 == 5'd0) begin
                    s2_valid <= 1;
                end
            end else begin
                if (buf_loaded) begin
                    compute_busy            <= 1;
                    bit_idx_s0              <= 0;
                    bit_idx_s1              <= 0;
                    bit_idx_s2              <= 0;
                    s1_valid                <= 0;
                    s2_valid                <= 0;
                    compute_first_bit_pulse <= 1;
                    MSB_SA_Ready            <= 0;
                    shift_add_done          <= 0;
                    buf_loaded              <= 0;
                    // Double-buffer (Task #99): COMPUTE wakes on the
                    // substrate that LOAD JUST FILLED (= ~load_phase,
                    // since LOAD already toggled to the OTHER substrate).
                    compute_phase           <= ~load_phase;
                    for (cc = 0; cc < NUM_COLS; cc = cc + 1)
                        mac_acc[cc] <= 32'sd0;
                end
            end

            // OUTPUT sub-FSM (drain mac_acc directly -- AL has no ACAM).
            if (output_busy) begin
                data_out <= 0;
                for (bb = 0; bb < ELEMS_PER_STROBE; bb = bb + 1) begin
                    if (output_col_idx * ELEMS_PER_STROBE + bb < NUM_COLS) begin
                        data_out[bb*8 +: 8] <=
                            mac_acc[output_col_idx * ELEMS_PER_STROBE + bb][7:0];
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
            // independent). reg_full is driven only by the OUTPUT
            // drain stage (single-substrate mac_acc).
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
                   help=f"Azure-Lily JSON config path (default: {DEFAULT_CFG_PATH})")
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
    out_path = os.path.join(args.out_dir, "dpe_azurelily_faithful.v")
    with open(out_path, "w") as fh:
        fh.write(src)
    print(f"[gen_dpe_azurelily_faithful] wrote {out_path}")
    print(f"               R={params['R']} C={params['C']} BUF={params['BUF']} "
          f"PRECISION={params['precision_bits']} PIPELINE_DEPTH={params['pipeline_depth']} "
          f"ACAM_CYCLES={params['acam_cycles']}")
    print(f"               LCYC={params['load_cycles']} "
          f"OCYC={params['output_cycles']} "
          f"CCYC(declared, emergent in RTL)={params['ccyc_declared']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
