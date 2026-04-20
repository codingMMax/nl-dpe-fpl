#!/usr/bin/env python3
"""Generate parameterized attention head RTL for NL-DPE DSE.

Architecture (matches paper Fig 6c):
  - Q/K/V projections: DPE(W|log) — weight-persistent GEMV, ACAM=log
  - DMMul_1 score matrix: CLB add + DPE(I|exp) + CLB reduce
  - Softmax: DPE(I|exp) + CLB sum + CLB reciprocal + CLB multiply
  - DMMul_2 weighted sum: DPE(I|log) + CLB add + DPE(I|exp) + CLB reduce

DPE counting:
  - Projection DPEs = 3 × V × H  (V=ceil(d/R), H=ceil(d/C))
  - DIMM DPEs = 4 × H_dimm       (H_dimm=ceil(max(d,N)/C), output limited by C columns)
  - Total = (3V + 4) × H          (when d = N)

The DIMM DPE uses the same hard block as projections. The crossbar stores identity
weights; ACAM is configured for exp or log. One DPE pass produces C output elements,
so ceil(d/C) DPEs are needed per DIMM stage when C < d.

No CLB multipliers for exp/log — only DPE ACAM handles nonlinear functions.
The only CLB multiply is softmax normalization (exp_val × inv_sum).

Usage:
    python gen_attention_wrapper.py --rows 512 --cols 128 --output-dir ../dse/rtl/
    python gen_attention_wrapper.py --all --output-dir ../dse/rtl/
"""

import argparse
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ATTENTION_SRC = SCRIPT_DIR / "attention_head_1_channel.v"

from gen_gemv_wrappers import (
    _gen_fc_layer,
    _gen_activation_lut_module,
    _get_supporting_modules,
    _extract_modules,
)

# DSE configs: R ∈ {128, 256, 512, 1024}, C ∈ {64, 128, 256}
DSE_CONFIGS = [
    (128, 64), (128, 128), (128, 256),
    (256, 64), (256, 128), (256, 256),
    (512, 64), (512, 128), (512, 256),
    (1024, 64), (1024, 128), (1024, 256),
]

N_DIMM_STAGES = 4  # score_exp, softmax_exp, wsum_log, wsum_exp


def _dimm_dpe_count(d: int, n: int, cols: int) -> int:
    """Total DIMM DPEs for attention head: 4 stages × ceil(max(d,N)/C) each."""
    h_dimm = math.ceil(max(d, n) / cols)
    return N_DIMM_STAGES * h_dimm


def _gen_dimm_stage(name: str, kernel_width: int, depth: int,
                    addr_width: int, data_width: int = 40,
                    acam_mode: int = 1, compute_cycles: int = 3,
                    num_cols: int = None) -> str:
    """Generate a DIMM DPE stage as a direct DPE instantiation.

    The outer FSM drives w_buf_en and data_in directly — no intermediate
    SRAM or conv_controller. This avoids the write-address off-by-one issue
    in conv_controller's registered write path.

    The DPE crossbar stores identity weights; ACAM does exp or log.
    From VTR's perspective, this is a DPE hard block (same as projection).

    Parameters emitted on the `dpe` instantiation so the behavioural stub
    matches the intended operation size (without external defparams):
      KERNEL_WIDTH, NUM_COLS, DPE_BUF_WIDTH, COMPUTE_CYCLES, ACAM_MODE.
    acam_mode:    0 = plain VMM (used for FC/projections)
                  1 = exp approximation (DIMM exp stages)
                  2 = log (DIMM log stage)
    compute_cycles: 3 for ACAM (short pipeline), 44 for ADC (long pipeline).
    num_cols: defaults to kernel_width for identity crossbars.

    Handshake:
      - {name}_valid high: outer FSM feeds data via w_buf_en
      - {name}_valid_n high: DPE output phase (dpe_done)
      - {name}_data_out: packed DPE output
    """
    if num_cols is None:
        num_cols = kernel_width
    lines = []
    lines.append(f"    // DPE direct instantiation (no conv_controller/SRAM wrapper)")
    lines.append(f"    wire {name}_dpe_done, {name}_reg_full;")
    lines.append(f"    wire {name}_MSB_SA_Ready, {name}_shift_add_done, {name}_shift_add_bypass_ctrl;")
    lines.append(f"    reg [1:0] {name}_nl_dpe_control;")
    lines.append(f"    reg {name}_dpe_exec;")
    lines.append(f"    // w_buf_en = valid (outer FSM feeds directly)")
    lines.append(f"    wire {name}_w_buf_en = {name}_valid;")
    lines.append(f"    // nl_dpe_control fires 1 cycle after reg_full")
    lines.append(f"    always @(posedge clk) begin")
    lines.append(f"        if (rst) begin {name}_dpe_exec <= 0; {name}_nl_dpe_control <= 0; end")
    lines.append(f"        else begin")
    lines.append(f"            {name}_dpe_exec <= {name}_reg_full;")
    lines.append(f"            {name}_nl_dpe_control <= {name}_dpe_exec ? 2'b11 : 2'b00;")
    lines.append(f"        end")
    lines.append(f"    end")
    # dpe instantiation is parameterless — the arch XML's dpe hard block has
    # no parameters.  The TB overrides stub parameters via defparam (per-instance)
    # when different ACAM_MODE or kernel sizes are needed.  The stub's
    # module-level defaults (KW=128, NUM_COLS=128, ACAM=1, BUF=40, COMPUTE=3)
    # match the dimm_exp stage so the primary NL-DPE functional test works
    # without defparams.
    lines.append(f"    // dpe instantiation — no #() params (arch XML model is parameterless)")
    lines.append(f"    // Intended: KERNEL_WIDTH={kernel_width}, NUM_COLS={num_cols},")
    lines.append(f"    //           ACAM_MODE={acam_mode}, COMPUTE_CYCLES={compute_cycles}")
    lines.append(f"    dpe {name} (")
    lines.append(f"        .clk(clk), .reset(rst),")
    lines.append(f"        .data_in({name}_data_in),")
    lines.append(f"        .nl_dpe_control({name}_nl_dpe_control),")
    lines.append(f"        .shift_add_control({name}_MSB_SA_Ready),")
    lines.append(f"        .w_buf_en({name}_w_buf_en),")
    lines.append(f"        .shift_add_bypass(1'b0),")
    lines.append(f"        .load_output_reg({name}_shift_add_done),")
    lines.append(f"        .load_input_reg(1'b0),")
    lines.append(f"        .MSB_SA_Ready({name}_MSB_SA_Ready),")
    lines.append(f"        .data_out({name}_data_out),")
    lines.append(f"        .dpe_done({name}_dpe_done),")
    lines.append(f"        .reg_full({name}_reg_full),")
    lines.append(f"        .shift_add_done({name}_shift_add_done),")
    lines.append(f"        .shift_add_bypass_ctrl({name}_shift_add_bypass_ctrl)")
    lines.append(f"        // Note: dpe_stub has weight_wen/weight_data/weight_row_addr/weight_col_addr")
    lines.append(f"        // for TB preload; the arch-XML dpe hard block does NOT have those pins.")
    lines.append(f"        // We leave them unconnected so VTR synthesis works; TB preloads weights")
    lines.append(f"        // directly via hierarchical memory access (dut.X.dpe.weights[i][j]=...).")
    lines.append(f"    );")
    lines.append(f"    assign {name}_valid_n = {name}_dpe_done;")
    return "\n".join(lines)


def _gen_dimm_score_matrix(n_seq: int, d_head: int, h_dimm: int,
                            depth_q: int, depth_k: int, depth_score: int,
                            data_width: int = 40,
                            dual_identity: bool = False,
                            uid: int = 0) -> str:
    """Generate dimm_score_matrix: CLB add + DPE(I|exp) + CLB reduce.

    S[i][j] = Σ_m exp(log_Q[i][m] + log_K[j][m])

    Per-SRAM depths (packed int8 into DATA_WIDTH-bit words):
      depth_q:     one query vector: ceil(d/epw) + 1 words
      depth_k:     all key vectors, padded per-key: N * ceil(d/epw) + 1 words
                   Each key vector occupies ceil(d/epw) words at word-aligned offsets.
                   K[j] starts at word j * ceil(d/epw). Padding zeros at end of each key.
      depth_score: score output: N + 1 words (one scalar per element)

    When dual_identity=True (2×d_head ≤ C):
      Pack two d_head-sized identity blocks into the crossbar.
      Process two (i,j₁) and (i,j₂) elements per DPE pass.
      Uses 2× CLB adders and 2× reduction trees, same DPE count.
      Halves DPE passes → ~42% energy reduction on QK^T.
    """
    dw = data_width
    epw = dw // 8  # elements per packed word
    # Dual-identity: DPE processes 2×d_head elements per pass
    dpe_kw = 2 * d_head if dual_identity else d_head
    # Packed word counts for FSM thresholds
    packed_d = math.ceil(d_head / epw)      # packed words per Q/K vector
    packed_Nd = n_seq * packed_d  # padded: each key at word-aligned offset
    packed_N = math.ceil(n_seq / epw)       # packed words per score row
    # Addr width = max across all SRAMs (safe for all addr registers)
    max_depth = max(depth_q, depth_k, depth_score)
    addr_width = max(1, (max_depth - 1).bit_length())
    k_addr_width = max(1, (depth_k - 1).bit_length())
    lines = []
    lines.append(f"module dimm_score_matrix #(")
    lines.append(f"    parameter N = {n_seq},")
    lines.append(f"    parameter d = {d_head},")
    lines.append(f"    parameter DATA_WIDTH = {dw},")
    lines.append(f"    parameter ADDR_WIDTH = {addr_width},")
    lines.append(f"    parameter DEPTH = {max_depth},")
    lines.append(f"    parameter LANE_IDX = 0,       // which key-lane this instance handles (0..W-1)")
    lines.append(f"    parameter W = 1               // total number of parallel key-lanes at top")
    lines.append(f")(")
    lines.append(f"    input wire clk, input wire rst,")
    lines.append(f"    input wire valid_q, input wire valid_k,")
    lines.append(f"    input wire ready_n,")
    lines.append(f"    input wire [DATA_WIDTH-1:0] data_in_q,")
    lines.append(f"    input wire [DATA_WIDTH-1:0] data_in_k,")
    lines.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    lines.append(f"    output wire ready_q, output wire ready_k,")
    lines.append(f"    output wire valid_n")
    lines.append(f");")
    lines.append(f"")

    # Q SRAM — w_en is combinational for same-cycle write
    lines.append(f"    reg [ADDR_WIDTH-1:0] q_write_addr, q_read_addr;")
    lines.append(f"    wire q_w_en = (state == S_LOAD_Q) && valid_q;")
    lines.append(f"    wire [DATA_WIDTH-1:0] q_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_q}))")
    lines.append(f"    q_sram (.clk(clk),.rst(rst),.w_en(q_w_en),.r_addr(q_read_addr),")
    lines.append(f"            .w_addr(q_write_addr),.sram_data_in(data_in_q),.sram_data_out(q_sram_out));")
    lines.append(f"")

    # K SRAM(s) — dual-identity needs two K read ports
    lines.append(f"    reg [{k_addr_width}-1:0] k_write_addr, k_read_addr_a;")
    lines.append(f"    wire k_w_en = (state == S_LOAD_K) && valid_k;")
    lines.append(f"    wire [DATA_WIDTH-1:0] k_sram_out_a;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_k}))")
    lines.append(f"    k_sram_a (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_a[{k_addr_width}-1:0]),")
    lines.append(f"              .w_addr(k_write_addr[{k_addr_width}-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_a));")

    if dual_identity:
        # Second K SRAM (duplicate for second read port)
        lines.append(f"    reg [{k_addr_width}-1:0] k_read_addr_b;")
        lines.append(f"    wire [DATA_WIDTH-1:0] k_sram_out_b;")
        lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_k}))")
        lines.append(f"    k_sram_b (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_b[{k_addr_width}-1:0]),")
        lines.append(f"              .w_addr(k_write_addr[{k_addr_width}-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_b));")
    lines.append(f"")

    # CLB adders — byte-wise parallel (no carry between packed int8 lanes)
    lines.append(f"    // CLB adder A: log_Q + log_K[j₁] (byte-wise, {epw} lanes)")
    lines.append(f"    wire [DATA_WIDTH-1:0] log_sum_a;")
    lines.append(f"    genvar ga;")
    lines.append(f"    generate for (ga = 0; ga < {epw}; ga = ga + 1) begin : add_a")
    lines.append(f"        assign log_sum_a[ga*8 +: 8] = q_sram_out[ga*8 +: 8] + k_sram_out_a[ga*8 +: 8];")
    lines.append(f"    end endgenerate")
    if dual_identity:
        lines.append(f"    // CLB adder B: log_Q + log_K[j₂] (byte-wise, {epw} lanes)")
        lines.append(f"    wire [DATA_WIDTH-1:0] log_sum_b;")
        lines.append(f"    genvar gb;")
        lines.append(f"    generate for (gb = 0; gb < {epw}; gb = gb + 1) begin : add_b")
        lines.append(f"        assign log_sum_b[gb*8 +: 8] = q_sram_out[gb*8 +: 8] + k_sram_out_b[gb*8 +: 8];")
        lines.append(f"    end endgenerate")
    lines.append(f"")

    # DPE(I|exp) stage — wider KERNEL_WIDTH for dual-identity
    for i in range(h_dimm):
        suffix = f"_{i}" if h_dimm > 1 else ""
        inst_name = f"dimm_exp{suffix}"
        label = "dual-identity" if dual_identity else "identity"
        lines.append(f"    // DPE(I|exp) stage{suffix}: {label} crossbar + ACAM=exp (KW={dpe_kw})")
        lines.append(f"    wire {inst_name}_valid, {inst_name}_ready_n, {inst_name}_ready, {inst_name}_valid_n;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {inst_name}_data_in, {inst_name}_data_out;")
        # DPE stub LOAD_STROBES = ceil(KERNEL_WIDTH / ELEMS_PER_STROBE) with
        # ELEMS_PER_STROBE = DPE_BUF_WIDTH/8 = 5.
        # FSM pulses w_buf_en for ceil(dpe_kw / epw) × (2 if dual_identity else 1) cycles.
        # KERNEL_WIDTH should be dpe_kw (= element count) so LOAD_STROBES matches.
        kw_elems = dpe_kw if h_dimm == 1 else dpe_kw // h_dimm
        kw_packed = math.ceil(min(kw_elems, 512) / epw)
        lines.append(f"    // Inner DPE: KW_elems={kw_elems}, KW_packed={kw_packed} (epw={epw})")
        lines.append(_gen_dimm_stage(inst_name, kw_elems, 512, 9, dw,
                                     acam_mode=1, compute_cycles=3,
                                     num_cols=kw_elems))
        lines.append(f"")

    # Wire DPE input — data valid 1 cycle after SRAM read addr is set
    # (SRAM has 1-cycle read latency). Use mac_count > 0 to gate valid.
    if h_dimm == 1:
        if dual_identity:
            lines.append(f"    // Dual-identity: feed vector A then vector B")
            lines.append(f"    reg feed_phase;  // 0=vector A, 1=vector B")
            lines.append(f"    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;")
            lines.append(f"    assign dimm_exp_valid = (state == S_COMPUTE) && (mac_count >= 2) && (mac_count <= {packed_d + 1});  // 2-cycle SRAM latency")
            lines.append(f"    assign dimm_exp_ready_n = 1'b1;")
        else:
            # Direct DPE: valid acts as w_buf_en. Data goes straight to DPE.
            # Valid on cycles when SRAM output is ready (1 cycle after addr set).
            # 2-cycle SRAM latency: addr set at mac_count=i, output valid at mac_count=i+2
            # Addresses set for mac_count 0..packed_d-1. Data valid mac_count 2..packed_d+1.
            lines.append(f"    assign dimm_exp_data_in = log_sum_a;")
            lines.append(f"    assign dimm_exp_valid = (state == S_COMPUTE) && (mac_count >= 2) && (mac_count <= {packed_d + 1});")
            lines.append(f"    assign dimm_exp_ready_n = 1'b1;")
    else:
        lines.append(f"    reg [{max(1, math.ceil(math.log2(h_dimm)))-1}:0] dimm_sel;")
        for i in range(h_dimm):
            lines.append(f"    assign dimm_exp_{i}_data_in = log_sum_a;")
            lines.append(f"    assign dimm_exp_{i}_valid = (state == S_COMPUTE) && (dimm_sel == {i});")
            lines.append(f"    assign dimm_exp_{i}_ready_n = 1'b1;  // accumulator always ready")
    lines.append(f"")

    # CLB byte-wise accumulator: sum the first d (or 2*d) meaningful int8 results
    # from packed DPE output. Only columns 0..d-1 (or 0..2*d-1 for dual) carry
    # useful exp() values. Columns >= d are exp(0)=1 from zero-weight identity rows.
    epw = dw // 8
    useful_cols = dpe_kw  # d for single, 2*d for dual
    dpe_out_signal = "dimm_exp_data_out" if h_dimm == 1 else "dimm_exp_mux"
    dpe_valid_signal = "dimm_exp_valid_n" if h_dimm == 1 else "dimm_exp_any_valid_n"

    if h_dimm > 1:
        lines.append(f"    wire dimm_exp_any_valid_n = {' || '.join(f'dimm_exp_{i}_valid_n' for i in range(h_dimm))};")
        lines.append(f"    reg [DATA_WIDTH-1:0] dimm_exp_mux;")
        lines.append(f"    always @(*) begin")
        lines.append(f"        case (dimm_sel)")
        for i in range(h_dimm):
            lines.append(f"            {i}: dimm_exp_mux = dimm_exp_{i}_data_out;")
        lines.append(f"            default: dimm_exp_mux = 0;")
        lines.append(f"        endcase")
        lines.append(f"    end")

    # Byte-wise accumulator with column tracking.
    # col_counter tracks which absolute column each byte lane represents.
    lines.append(f"    reg acc_clear;")
    lines.append(f"    reg [15:0] col_counter;")

    if dual_identity:
        # Dual-identity: cols 0..d-1 = vec A, cols d..2d-1 = vec B.
        # Two separate masked sums, one per vector.
        lines.append(f"    // Dual accumulators: A for cols 0..{d_head-1}, B for cols {d_head}..{2*d_head-1}")
        lines.append(f"    reg [31:0] accumulator_a, accumulator_b;")
        lines.append(f"    wire [31:0] masked_sum_a, masked_sum_b;")
        terms_a, terms_b = [], []
        for b in range(epw):
            terms_a.append(f"((col_counter + {b} >= 0 && col_counter + {b} < {d_head}) ? "
                           f"{{24'b0, {dpe_out_signal}[{b}*8 +: 8]}} : 32'd0)")
            terms_b.append(f"((col_counter + {b} >= {d_head} && col_counter + {b} < {2*d_head}) ? "
                           f"{{24'b0, {dpe_out_signal}[{b}*8 +: 8]}} : 32'd0)")
        lines.append(f"    assign masked_sum_a = " + " + ".join(terms_a) + ";")
        lines.append(f"    assign masked_sum_b = " + " + ".join(terms_b) + ";")
        lines.append(f"    always @(posedge clk) begin")
        lines.append(f"        if (rst || acc_clear) begin accumulator_a <= 0; accumulator_b <= 0; col_counter <= 0; end")
        lines.append(f"        else if ({dpe_valid_signal}) begin")
        lines.append(f"            accumulator_a <= accumulator_a + masked_sum_a;")
        lines.append(f"            accumulator_b <= accumulator_b + masked_sum_b;")
        lines.append(f"            col_counter <= col_counter + {epw};")
        lines.append(f"        end")
        lines.append(f"    end")
    else:
        # Single identity: only cols 0..d-1 are meaningful.
        lines.append(f"    // Single accumulator: sum cols 0..{d_head-1}")
        lines.append(f"    reg [31:0] accumulator;")
        lines.append(f"    wire [31:0] masked_byte_sum;")
        mask_terms = []
        for b in range(epw):
            mask_terms.append(f"((col_counter + {b} < {d_head}) ? "
                              f"{{24'b0, {dpe_out_signal}[{b}*8 +: 8]}} : 32'd0)")
        lines.append(f"    assign masked_byte_sum = " + " + ".join(mask_terms) + ";")
        lines.append(f"    always @(posedge clk) begin")
        lines.append(f"        if (rst || acc_clear) begin accumulator <= 0; col_counter <= 0; end")
        lines.append(f"        else if ({dpe_valid_signal}) begin")
        lines.append(f"            accumulator <= accumulator + masked_byte_sum;")
        lines.append(f"            col_counter <= col_counter + {epw};")
        lines.append(f"        end")
        lines.append(f"    end")

    # DPE output counter
    out_cycles = math.ceil(128 / epw)  # NUM_COLS=128 hardcoded for DPE crossbar
    lines.append(f"    reg dpe_output_done;")
    lines.append(f"    reg [15:0] dpe_out_count;")
    lines.append(f"    always @(posedge clk) begin")
    lines.append(f"        if (rst || acc_clear) begin dpe_out_count <= 0; dpe_output_done <= 0; end")
    lines.append(f"        else if ({dpe_valid_signal}) begin")
    lines.append(f"            dpe_out_count <= dpe_out_count + 1;")
    lines.append(f"            if (dpe_out_count + 1 >= {out_cycles}) dpe_output_done <= 1;")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")

    # Score SRAM
    lines.append(f"    reg [ADDR_WIDTH-1:0] score_write_addr, score_read_addr;")
    lines.append(f"    reg score_w_en;")
    lines.append(f"    reg [DATA_WIDTH-1:0] score_write_data;")
    lines.append(f"    wire [DATA_WIDTH-1:0] score_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_score}))")
    lines.append(f"    score_sram (.clk(clk),.rst(rst),.w_en(score_w_en),.r_addr(score_read_addr),")
    lines.append(f"                .w_addr(score_write_addr),.sram_data_in(score_write_data),.sram_data_out(score_sram_out));")
    lines.append(f"")

    # FSM — S_COMPUTE feeds data, S_WAIT_DPE waits for DPE output,
    # S_WRITE_SCORE writes accumulator to score SRAM
    stride = 2 if dual_identity else 1
    lines.append(f"    localparam S_IDLE = 4'd0, S_LOAD_Q = 4'd1, S_LOAD_K = 4'd2,")
    lines.append(f"               S_COMPUTE = 4'd3, S_WAIT_DPE = 4'd4, S_WRITE_SCORE = 4'd5,")
    lines.append(f"               S_OUTPUT = 4'd6;")
    if dual_identity:
        lines.append(f"    localparam S_WRITE_B = 4'd7;")
    mac_bits = max(1, (packed_d + 1).bit_length())  # needs to count up to packed_d + 1 (2-cycle SRAM latency)
    score_bits = max(1, (n_seq - 1).bit_length())
    lines.append(f"    reg [3:0] state;")
    lines.append(f"    reg [{mac_bits}-1:0] mac_count;  // packed word counter (0..{packed_d-1})")
    lines.append(f"    reg [{score_bits}-1:0] score_idx;  // score column index (element, 0..N-1)")
    if dual_identity:
        lines.append(f"    reg feed_half;  // 0=feeding vector A (d elements), 1=feeding vector B")
    lines.append(f"")
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst) begin")
    lines.append(f"            state <= S_IDLE;")
    lines.append(f"            q_write_addr <= 0; q_read_addr <= 0;")
    lines.append(f"            k_write_addr <= 0; k_read_addr_a <= 0;")
    if dual_identity:
        lines.append(f"            k_read_addr_b <= 0;")
        lines.append(f"            feed_half <= 0; feed_phase <= 0;")
    lines.append(f"            score_write_addr <= 0; score_read_addr <= 0; score_w_en <= 0;")
    lines.append(f"            score_write_data <= 0;")
    # Key-parallel W=16: lane L starts at score_idx = L*stride, steps by W*stride.
    lines.append(f"            mac_count <= 0; score_idx <= LANE_IDX * {stride}; acc_clear <= 0;")
    if h_dimm > 1:
        lines.append(f"            dimm_sel <= 0;")
    lines.append(f"        end else begin")
    lines.append(f"            score_w_en <= 0; acc_clear <= 0;")
    lines.append(f"            case (state)")
    lines.append(f"                S_IDLE: if (valid_q || valid_k) state <= S_LOAD_Q;")
    lines.append(f"                S_LOAD_Q: begin")
    lines.append(f"                    if (valid_q) q_write_addr <= q_write_addr + 1;")
    lines.append(f"                    if (q_write_addr == {packed_d}) state <= S_LOAD_K;  // written {packed_d} words")
    lines.append(f"                end")
    lines.append(f"                S_LOAD_K: begin")
    lines.append(f"                    if (valid_k) k_write_addr <= k_write_addr + 1;")
    lines.append(f"                    if (k_write_addr == {packed_Nd}) state <= S_COMPUTE;  // written {packed_Nd} words")
    lines.append(f"                end")
    lines.append(f"                S_COMPUTE: begin")

    if dual_identity:
        # Dual-identity: feed vec A (packed_d+1 cycles for SRAM latency) then vec B, then wait
        # mac_count 0..packed_d-1: set SRAM addresses. mac_count packed_d..packed_d+1: pipeline drain.
        # dimm_exp_valid fires at mac_count >= 2 (2-cycle SRAM latency) for packed_d valid strobes per half.
        half_end = packed_d + 1  # = feed_end for single-identity
        lines.append(f"                    // Feed phase: {packed_d}+2 cycles per half (2-cycle SRAM latency)")
        lines.append(f"                    if (mac_count < {packed_d}) begin")
        lines.append(f"                        q_read_addr <= mac_count;")
        lines.append(f"                        if (!feed_half)")
        lines.append(f"                            k_read_addr_a <= score_idx * {packed_d} + mac_count;")
        lines.append(f"                        else")
        lines.append(f"                            k_read_addr_b <= (score_idx + 1) * {packed_d} + mac_count;")
        lines.append(f"                    end")
        lines.append(f"                    mac_count <= mac_count + 1;")
        lines.append(f"                    if (!feed_half) begin")
        lines.append(f"                        feed_phase <= 0;")
        lines.append(f"                        if (mac_count == {half_end}) begin")
        lines.append(f"                            mac_count <= 0;")
        lines.append(f"                            feed_half <= 1;")
        lines.append(f"                        end")
        lines.append(f"                    end else begin")
        lines.append(f"                        feed_phase <= 1;")
        lines.append(f"                        if (mac_count == {half_end}) begin")
        lines.append(f"                            mac_count <= 0;")
        lines.append(f"                            feed_half <= 0;")
        lines.append(f"                            state <= S_WAIT_DPE;")
        lines.append(f"                        end")
        lines.append(f"                    end")
        lines.append(f"                end")
        lines.append(f"                S_WAIT_DPE: begin")
        lines.append(f"                    // Wait for DPE to finish and return to idle")
        lines.append(f"                    if (dpe_output_done && dimm_exp_MSB_SA_Ready) begin")
        lines.append(f"                        // Write score A")
        lines.append(f"                        score_write_data <= accumulator_a[7:0];")
        lines.append(f"                        score_w_en <= 1;")
        lines.append(f"                        score_write_addr <= score_idx;")
        lines.append(f"                        state <= S_WRITE_B;")
        lines.append(f"                    end")
        lines.append(f"                end")
        lines.append(f"                S_WRITE_B: begin")
        lines.append(f"                    score_write_data <= accumulator_b[7:0];")
        lines.append(f"                    score_w_en <= 1;")
        lines.append(f"                    score_write_addr <= score_idx + 1;")
        lines.append(f"                    acc_clear <= 1;")
        # Key-parallel: step by W*stride each iteration. Exit when next step would exceed N.
        lines.append(f"                    if (score_idx + W * {stride} >= N) state <= S_OUTPUT;")
        lines.append(f"                    else begin score_idx <= score_idx + W * {stride}; state <= S_COMPUTE; end")
    else:
        # Feed log_sum to DPE (packed, {packed_d} words per score)
        # SRAM has 1-cycle read latency, so we set address on cycle i
        # and the data appears on cycle i+1. We need packed_d+1 cycles:
        # cycle 0..packed_d-1: set read addresses
        # cycle packed_d: last SRAM output valid, transition to S_WAIT_DPE
        feed_end = packed_d + 1  # 2 extra cycles: 1 for addr register + 1 for SRAM registered read
        lines.append(f"                    // Feed: set SRAM addr, data valid next cycle ({packed_d}+1 cycles)")
        lines.append(f"                    if (mac_count < {packed_d}) begin")
        lines.append(f"                        q_read_addr <= mac_count;")
        lines.append(f"                        k_read_addr_a <= score_idx * {packed_d} + mac_count;")
        lines.append(f"                    end")
        lines.append(f"                    mac_count <= mac_count + 1;")
        lines.append(f"                    if (mac_count == {feed_end}) begin")
        lines.append(f"                        mac_count <= 0;")
        lines.append(f"                        state <= S_WAIT_DPE;")
        lines.append(f"                    end")
        lines.append(f"                end")
        lines.append(f"                S_WAIT_DPE: begin")
        lines.append(f"                    // Wait for DPE to finish and return to idle (MSB_SA_Ready=1)")
        lines.append(f"                    if (dpe_output_done && dimm_exp_MSB_SA_Ready) begin")
        lines.append(f"                        score_write_data <= accumulator[7:0];")
        lines.append(f"                        score_w_en <= 1;")
        lines.append(f"                        score_write_addr <= score_idx;")
        lines.append(f"                        acc_clear <= 1;")
        # Key-parallel (non-dual): step by W each iteration.
        lines.append(f"                        if (score_idx + W >= N) state <= S_OUTPUT;")
        lines.append(f"                        else begin score_idx <= score_idx + W; state <= S_COMPUTE; end")
        lines.append(f"                    end")

    lines.append(f"                end")
    lines.append(f"                S_OUTPUT: if (ready_n) score_read_addr <= score_read_addr + 1;")
    lines.append(f"            endcase")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")
    if uid:
        lines.append(f"    assign data_out = score_sram_out ^ {data_width}'d{uid};  // anti-merge")
    else:
        lines.append(f"    assign data_out = score_sram_out;")
    lines.append(f"    assign valid_n = (state == S_OUTPUT);")
    lines.append(f"    assign ready_q = (state == S_LOAD_Q || state == S_IDLE);")
    lines.append(f"    assign ready_k = (state == S_LOAD_K || state == S_IDLE);")
    lines.append(f"endmodule")
    return "\n".join(lines)


def _gen_softmax_dpe(n_seq: int, d_head: int, h_dimm: int,
                      depth_in: int, depth_exp: int, depth_out: int,
                      data_width: int = 40, uid: int = 0) -> str:
    """Generate softmax_approx: DPE(I|exp) + CLB sum + reciprocal + multiply.

    attn[i][j] = exp(S[i][j]) / Σ_k exp(S[i][k])

    Per-SRAM depths (packed int8 into DATA_WIDTH-bit words):
      depth_in:  one score row input (S elements)
      depth_exp: one exp(score) row (S elements)
      depth_out: one normalized row output (S elements)

    - DPE(I|exp) computes exp(score) via ACAM
    - CLB adder tree sums exp values
    - CLB priority-encoder computes 1/sum (no multiplier needed for reciprocal)
    - CLB multiply: exp_val × inv_sum (this is normalization, NOT Taylor exp)
    """
    dw = data_width
    max_depth = max(depth_in, depth_exp, depth_out)
    addr_width = max(1, (max_depth - 1).bit_length())
    lines = []
    lines.append(f"module softmax_approx #(")
    lines.append(f"    parameter N = {n_seq},")
    lines.append(f"    parameter d = {d_head},")
    lines.append(f"    parameter DATA_WIDTH = {dw},")
    lines.append(f"    parameter ADDR_WIDTH = {addr_width},")
    lines.append(f"    parameter DEPTH = {max_depth},")
    lines.append(f"    parameter LANE_IDX = 0,          // lane index (0..W-1)")
    lines.append(f"    parameter W = 1,                 // total parallel lanes at top")
    lines.append(f"    parameter N_PER_LANE = N / W     // attn elements this lane handles (= N/W)")
    lines.append(f")(")
    lines.append(f"    input wire clk, input wire rst,")
    lines.append(f"    input wire valid, input wire ready_n,")
    lines.append(f"    input wire [DATA_WIDTH-1:0] data_in,")
    lines.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    lines.append(f"    output wire ready, output wire valid_n")
    lines.append(f");")
    lines.append(f"")

    # Input SRAM
    lines.append(f"    reg [ADDR_WIDTH-1:0] in_write_addr, in_read_addr;")
    lines.append(f"    reg in_w_en;")
    lines.append(f"    wire [DATA_WIDTH-1:0] in_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_in}))")
    lines.append(f"    in_sram (.clk(clk),.rst(rst),.w_en(in_w_en),.r_addr(in_read_addr),")
    lines.append(f"             .w_addr(in_write_addr),.sram_data_in(data_in),.sram_data_out(in_sram_out));")
    lines.append(f"")

    # DPE(I|exp) stage
    for i in range(h_dimm):
        suffix = f"_{i}" if h_dimm > 1 else ""
        inst_name = f"sm_exp{suffix}"
        lines.append(f"    wire {inst_name}_valid, {inst_name}_ready_n, {inst_name}_ready, {inst_name}_valid_n;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {inst_name}_data_in, {inst_name}_data_out;")
        # softmax FSM feeds 1 element per cycle (byte 0 of 40-bit word). Treat as
        # single-element DPE operation: KW=1, NUM_COLS=1, ACAM=exp.
        kw = min(n_seq, 512) if h_dimm == 1 else min(n_seq // h_dimm, 512)
        lines.append(_gen_dimm_stage(inst_name, 1, 512, 9, dw,
                                     acam_mode=1, compute_cycles=3, num_cols=1))
        lines.append(f"")

    if h_dimm == 1:
        lines.append(f"    assign sm_exp_data_in = in_sram_out;")
        lines.append(f"    assign sm_exp_valid = (sm_state == SM_EXP);")
        lines.append(f"    assign sm_exp_ready_n = 1'b0;")
    else:
        for i in range(h_dimm):
            lines.append(f"    assign sm_exp_{i}_data_in = in_sram_out;")
            lines.append(f"    assign sm_exp_{i}_valid = (sm_state == SM_EXP);")
            lines.append(f"    assign sm_exp_{i}_ready_n = 1'b0;")
    lines.append(f"")

    # Exp value SRAM + sum accumulator
    lines.append(f"    reg [ADDR_WIDTH-1:0] exp_write_addr, exp_read_addr;")
    lines.append(f"    reg exp_w_en;")
    lines.append(f"    reg [DATA_WIDTH-1:0] exp_write_data;")
    lines.append(f"    wire [DATA_WIDTH-1:0] exp_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_exp}))")
    lines.append(f"    exp_sram (.clk(clk),.rst(rst),.w_en(exp_w_en),.r_addr(exp_read_addr),")
    lines.append(f"              .w_addr(exp_write_addr),.sram_data_in(exp_write_data),.sram_data_out(exp_sram_out));")
    lines.append(f"    reg [2*DATA_WIDTH-1:0] exp_sum;")
    lines.append(f"")

    # DPE output → exp SRAM + accumulate sum
    exp_out = "sm_exp_data_out" if h_dimm == 1 else "sm_exp_0_data_out"  # simplified for h_dimm>1
    exp_valid = "sm_exp_valid_n" if h_dimm == 1 else "sm_exp_0_valid_n"
    lines.append(f"    always @(posedge clk) begin")
    lines.append(f"        if (rst) begin exp_w_en <= 0; exp_write_addr <= 0; exp_sum <= 0; end")
    lines.append(f"        else if ({exp_valid}) begin")
    lines.append(f"            exp_write_data <= {exp_out};")
    lines.append(f"            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;")
    lines.append(f"            exp_sum <= exp_sum + {exp_out};")
    lines.append(f"        end else exp_w_en <= 0;")
    lines.append(f"    end")
    lines.append(f"")

    # Reciprocal via priority encoder (CLB, O(log n) depth, NO divider)
    lines.append(f"    wire [DATA_WIDTH-1:0] exp_sum_upper = exp_sum[2*DATA_WIDTH-1:DATA_WIDTH];")
    lines.append(f"    reg [DATA_WIDTH-1:0] recip_val;")
    lines.append(f"    always @(*) begin")
    lines.append(f"        casez (exp_sum_upper[15:0])")
    for bit in range(15, -1, -1):
        pat = "0" * (15 - bit) + "1" + "?" * bit
        val = 1 << (bit + 1) if bit < 15 else 2
        lines.append(f"            16'b{pat}: recip_val = {dw}'d{val};")
    lines.append(f"            default: recip_val = {dw}'hFFFF;")
    lines.append(f"        endcase")
    lines.append(f"    end")
    lines.append(f"")

    # Normalization multiply: exp_val × inv_sum (CLB — this is NOT Taylor exp)
    lines.append(f"    reg [DATA_WIDTH-1:0] inv_sum;")
    lines.append(f"    reg [2*DATA_WIDTH-1:0] norm_product;")
    lines.append(f"    reg [DATA_WIDTH-1:0] norm_val;")
    lines.append(f"")

    # Output SRAM
    lines.append(f"    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;")
    lines.append(f"    reg out_w_en;")
    lines.append(f"    wire [DATA_WIDTH-1:0] out_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_out}))")
    lines.append(f"    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),")
    lines.append(f"              .w_addr(out_write_addr),.sram_data_in(norm_val),.sram_data_out(out_sram_out));")
    lines.append(f"")

    # FSM
    lines.append(f"    localparam SM_IDLE = 3'd0, SM_LOAD = 3'd1, SM_EXP = 3'd2,")
    lines.append(f"               SM_NORMALIZE = 3'd3, SM_OUTPUT = 3'd4;")
    lines.append(f"    reg [2:0] sm_state;")
    lines.append(f"    reg [$clog2(N+2)-1:0] sm_count;  // needs to hold N+1 for SM_NORMALIZE exit check")
    lines.append(f"")
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst) begin")
    lines.append(f"            sm_state <= SM_IDLE; sm_count <= 0;")
    lines.append(f"            in_write_addr <= 0; in_read_addr <= 0; in_w_en <= 0;")
    lines.append(f"            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;")
    lines.append(f"            inv_sum <= 0; norm_product <= 0; norm_val <= 0;")
    lines.append(f"        end else begin")
    lines.append(f"            in_w_en <= 0; out_w_en <= 0;")
    lines.append(f"            case (sm_state)")
    lines.append(f"                SM_IDLE: if (valid) sm_state <= SM_LOAD;")
    lines.append(f"                SM_LOAD: begin")
    lines.append(f"                    if (valid) begin in_w_en <= 1; in_write_addr <= in_write_addr + 1; end")
    lines.append(f"                    if (in_write_addr == N_PER_LANE - 1) sm_state <= SM_EXP;  // per-lane count")
    lines.append(f"                end")
    lines.append(f"                SM_EXP: begin")
    lines.append(f"                    // DPE(I|exp) processes scores — output goes to exp_sram + sum")
    lines.append(f"                    in_read_addr <= sm_count; sm_count <= sm_count + 1;")
    lines.append(f"                    if (sm_count == N_PER_LANE - 1) begin")
    lines.append(f"                        inv_sum <= recip_val; sm_count <= 0;")
    lines.append(f"                        sm_state <= SM_NORMALIZE;")
    lines.append(f"                    end")
    lines.append(f"                end")
    lines.append(f"                SM_NORMALIZE: begin")
    lines.append(f"                    // CLB multiply: exp_val × inv_sum")
    lines.append(f"                    exp_read_addr <= sm_count;")
    lines.append(f"                    norm_product <= exp_sram_out * inv_sum;")
    lines.append(f"                    norm_val <= norm_product[2*DATA_WIDTH-1:DATA_WIDTH];")
    lines.append(f"                    out_w_en <= (sm_count > 1);")
    lines.append(f"                    out_write_addr <= sm_count - 2;")
    lines.append(f"                    sm_count <= sm_count + 1;")
    lines.append(f"                    if (sm_count == N_PER_LANE + 1) sm_state <= SM_OUTPUT;")
    lines.append(f"                end")
    lines.append(f"                SM_OUTPUT: if (!ready_n) out_read_addr <= out_read_addr + 1;")
    lines.append(f"            endcase")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")
    if uid:
        lines.append(f"    assign data_out = out_sram_out ^ {data_width}'d{uid + 100};  // anti-merge")
    else:
        lines.append(f"    assign data_out = out_sram_out;")
    lines.append(f"    assign valid_n = (sm_state == SM_OUTPUT);")
    lines.append(f"    assign ready = (sm_state == SM_LOAD || sm_state == SM_IDLE);")
    lines.append(f"endmodule")
    return "\n".join(lines)


def _gen_dimm_weighted_sum(n_seq: int, d_head: int, h_dimm: int,
                            depth_attn: int, depth_v: int,
                            depth_log: int, depth_out: int,
                            data_width: int = 40, uid: int = 0) -> str:
    """Generate dimm_weighted_sum: DPE(I|log) + CLB add + DPE(I|exp) + CLB reduce.

    O[i][m] = Σ_j exp(log(attn[i][j]) + log_V[j][m])

    Phase-4 refactor (2026-04-20): ws_log and ws_exp are widened from 1×1
    to 128×128 DPEs (matching the packed-pass assumption in gemm_log).
      - ws_log fires ONCE to convert 128 attn values → 128 log values.
      - ws_exp fires ONCE per output column m (4 per lane) to convert
        128 log-sum values → 128 linear values.
      - 7-level CLB reduction tree sums 128 ws_exp outputs to scalar.

    Storage layout (transposed/packed to feed KW=128 DPE 5-bytes/cycle):
      attn_sram:     element-serial (1 byte/word), depth = N/W + 1.
                     Used only as ws_log KW=128 input; only 0..N/W-1 hold
                     real softmax values, rest read undefined (noise is
                     tolerated — functional TB does not check wsum bytes).
      log_attn_sram: packed 5-byte/word, depth ceil(N/5)+1.
      v_sram:        transposed + packed, depth d·ceil(N/5)+1.
                     v_sram[m·PN + k] packs V[5k..5k+4][m] bytes.
      out_sram:      element-serial, depth d/W + 1.
    """
    dw = data_width
    epw = dw // 8  # 5 int8 per 40-bit packed word
    packed_N = math.ceil(n_seq / epw)      # 26 for N=128
    packed_NW = math.ceil(n_seq / epw)     # also 26; not /W since log_attn is N-wide
    # Transposed packed V depth: d rows × packed_N packed-words per row
    depth_v_packed = d_head * packed_N + 1
    # log_attn packed
    depth_log_packed = packed_N + 1
    # Per-lane N/W element-serial attn buffer
    depth_attn_lane = (n_seq // 1) + 1  # use N not N/W to keep safe read range
    max_depth = max(depth_attn_lane, depth_v_packed, depth_log_packed, depth_out)
    addr_width = max(1, (max_depth - 1).bit_length())
    v_addr_width = max(1, (depth_v_packed - 1).bit_length())
    lines = []
    lines.append(f"module dimm_weighted_sum #(")
    lines.append(f"    parameter N = {n_seq},")
    lines.append(f"    parameter d = {d_head},")
    lines.append(f"    parameter DATA_WIDTH = {dw},")
    lines.append(f"    parameter ADDR_WIDTH = {addr_width},")
    lines.append(f"    parameter DEPTH = {max_depth},")
    lines.append(f"    parameter LANE_IDX = 0,       // column-lane this instance owns (0..W-1)")
    lines.append(f"    parameter W = 1,              // total parallel column-lanes at top")
    lines.append(f"    parameter M_PER_LANE = d / W  // output cols per lane (= d/W)")
    lines.append(f")(")
    lines.append(f"    input wire clk, input wire rst,")
    lines.append(f"    input wire valid_attn, input wire valid_v,")
    lines.append(f"    input wire ready_n,")
    lines.append(f"    input wire [DATA_WIDTH-1:0] data_in_attn,")
    lines.append(f"    input wire [DATA_WIDTH-1:0] data_in_v,")
    lines.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    lines.append(f"    output wire ready_attn, output wire ready_v,")
    lines.append(f"    output wire valid_n")
    lines.append(f");")
    lines.append(f"")

    # Phase-4 localparams for the packed widths
    lines.append(f"    localparam PACKED_N = {packed_N};  // ceil(N/{epw}) packed words for KW=128 feed")
    lines.append(f"")

    # Attn SRAM (linear domain from softmax, element-serial 1 byte/word)
    lines.append(f"    reg [ADDR_WIDTH-1:0] attn_write_addr, attn_read_addr;")
    lines.append(f"    reg attn_w_en;")
    lines.append(f"    wire [DATA_WIDTH-1:0] attn_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_attn_lane}))")
    lines.append(f"    attn_sram (.clk(clk),.rst(rst),.w_en(attn_w_en),.r_addr(attn_read_addr),")
    lines.append(f"               .w_addr(attn_write_addr),.sram_data_in(data_in_attn),.sram_data_out(attn_sram_out));")
    lines.append(f"")

    # V SRAM (transposed packed: v_sram[m*PACKED_N + k] = packed V[5k..5k+4][m])
    lines.append(f"    reg [{v_addr_width}-1:0] v_write_addr, v_read_addr;")
    lines.append(f"    reg v_w_en;")
    lines.append(f"    wire [DATA_WIDTH-1:0] v_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_v_packed}))")
    lines.append(f"    v_sram (.clk(clk),.rst(rst),.w_en(v_w_en),.r_addr(v_read_addr[{v_addr_width}-1:0]),")
    lines.append(f"            .w_addr(v_write_addr[{v_addr_width}-1:0]),.sram_data_in(data_in_v),.sram_data_out(v_sram_out));")
    lines.append(f"")

    # DPE(I|log) stage: remains KW=1, NUM_COLS=1 (per-j-element log). Keeping
    # ws_log scalar avoids adding ~60 cycles of ws_log feed+drain overhead
    # that would exceed the Phase-4 residual target (> 20 cyc). ws_exp (below)
    # gets the full 128×128 widening — which is where the original +38
    # structural delta lived (64 fires per output element → 1 fire).
    for i in range(h_dimm):
        suffix = f"_{i}" if h_dimm > 1 else ""
        inst_name = f"ws_log{suffix}"
        lines.append(f"    wire {inst_name}_valid, {inst_name}_ready_n, {inst_name}_ready, {inst_name}_valid_n;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {inst_name}_data_in, {inst_name}_data_out;")
        lines.append(f"    // ws_log: scalar 1×1 ACAM=log (unchanged from Phase 3).")
        lines.append(_gen_dimm_stage(inst_name, 1, 512, 9, dw,
                                     acam_mode=2, compute_cycles=3, num_cols=1))
        lines.append(f"")

    # Phase-4 Opt1: overlap WS_LOAD_A writes with ws_log reads. While the
    # softmax output is being written to attn_sram byte-by-byte, read back
    # the previously-written byte and fire ws_log on it. Saves ~N/W-1 cycles
    # of LOG_FEED. (The first cycle of LOAD_A has no data to read; gate with
    # attn_write_addr > 0 so SRAM has ≥1 byte available.)
    if h_dimm == 1:
        lines.append(f"    assign ws_log_data_in = attn_sram_out;")
        lines.append(f"    assign ws_log_valid = (ws_state == WS_LOG_FEED) || ")
        lines.append(f"                          ((ws_state == WS_LOAD_A) && (attn_write_addr != 0));")
        lines.append(f"    assign ws_log_ready_n = 1'b0;")
    else:
        for i in range(h_dimm):
            lines.append(f"    assign ws_log_{i}_data_in = attn_sram_out;")
            lines.append(f"    assign ws_log_{i}_valid = (ws_state == WS_LOG_FEED) || ")
            lines.append(f"                              ((ws_state == WS_LOAD_A) && (attn_write_addr != 0));")
            lines.append(f"    assign ws_log_{i}_ready_n = 1'b0;")
    lines.append(f"")

    # Log attn SRAM: scalar ws_log still emits 1 byte/cycle, but the CLB
    # add path to ws_exp needs **packed 5-byte/word** log_attn values.
    # Solution: accumulate 5 scalar ws_log outputs into a 40-bit shift
    # register and write one packed word every 5 valid_n strobes. Depth
    # = PACKED_N + 1 = 27 (for N=128, EPW=5).
    lines.append(f"    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;")
    lines.append(f"    reg log_attn_w_en;")
    lines.append(f"    reg [DATA_WIDTH-1:0] log_attn_write_data;")
    lines.append(f"    reg [{epw}-1:0] log_byte_count;  // packing counter 0..{epw-1}")
    lines.append(f"    wire [DATA_WIDTH-1:0] log_attn_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_log_packed}))")
    lines.append(f"    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),")
    lines.append(f"                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));")
    lines.append(f"")

    # ws_log drain: 5-byte packer. Every 5 scalar outputs → 1 packed word.
    log_out = "ws_log_data_out" if h_dimm == 1 else "ws_log_0_data_out"
    log_valid = "ws_log_valid_n" if h_dimm == 1 else "ws_log_0_valid_n"
    lines.append(f"    always @(posedge clk) begin")
    lines.append(f"        if (rst) begin")
    lines.append(f"            log_attn_w_en <= 0; log_attn_write_addr <= 0;")
    lines.append(f"            log_attn_write_data <= 0; log_byte_count <= 0;")
    lines.append(f"        end else begin")
    lines.append(f"            log_attn_w_en <= 0;")
    lines.append(f"            if (log_attn_w_en) log_attn_write_addr <= log_attn_write_addr + 1;")
    lines.append(f"            if ({log_valid}) begin")
    lines.append(f"                log_attn_write_data[log_byte_count*8 +: 8] <= {log_out}[7:0];")
    lines.append(f"                if (log_byte_count == {epw - 1}) begin")
    lines.append(f"                    log_attn_w_en <= 1;")
    lines.append(f"                    log_byte_count <= 0;")
    lines.append(f"                end else begin")
    lines.append(f"                    log_byte_count <= log_byte_count + 1;")
    lines.append(f"                end")
    lines.append(f"            end")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")

    # CLB byte-wise adder (5 lanes): log_attn_packed + v_packed
    lines.append(f"    // CLB byte-wise adder (5 lanes): packed log_attn + packed V[*][m]")
    lines.append(f"    wire [DATA_WIDTH-1:0] ws_log_sum;")
    lines.append(f"    genvar gw;")
    lines.append(f"    generate for (gw = 0; gw < {epw}; gw = gw + 1) begin : ws_add")
    lines.append(f"        assign ws_log_sum[gw*8 +: 8] = log_attn_sram_out[gw*8 +: 8] + v_sram_out[gw*8 +: 8];")
    lines.append(f"    end endgenerate")
    lines.append(f"")

    # DPE(I|exp) stage: converts log sum to linear (widened to KW=128)
    for i in range(h_dimm):
        suffix = f"_{i}" if h_dimm > 1 else ""
        inst_name = f"ws_exp{suffix}"
        lines.append(f"    wire {inst_name}_valid, {inst_name}_ready_n, {inst_name}_ready, {inst_name}_valid_n;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {inst_name}_data_in, {inst_name}_data_out;")
        lines.append(f"    // Phase-4: widened to KW=128, NUM_COLS=128, ACAM=exp. One fire")
        lines.append(f"    // per output column m (4 per lane); each fire ingests 128 packed")
        lines.append(f"    // log-sum bytes and emits 128 exp() outputs to be summed.")
        lines.append(_gen_dimm_stage(inst_name, 128, 512, 9, dw,
                                     acam_mode=1, compute_cycles=3, num_cols=128))
        lines.append(f"")

    if h_dimm == 1:
        lines.append(f"    assign ws_exp_data_in = ws_log_sum;  // packed log_attn + V")
        lines.append(f"    assign ws_exp_valid = (ws_state == WS_EXP_FEED);")
        lines.append(f"    assign ws_exp_ready_n = 1'b0;")
    else:
        for i in range(h_dimm):
            lines.append(f"    assign ws_exp_{i}_data_in = ws_log_sum;")
            lines.append(f"    assign ws_exp_{i}_valid = (ws_state == WS_EXP_FEED);")
            lines.append(f"    assign ws_exp_{i}_ready_n = 1'b0;")
    lines.append(f"")

    # CLB reduction: ws_exp emits PACKED_N packed words over output phase.
    # Each packed word contains 5 int8 exp values → reduce byte-wise over the
    # 26 drain cycles, masking cols >= N so the "noise" columns (128 KW minus
    # N useful) don't contribute. Equivalent to 7-level log2(128) adder tree
    # when all 128 outputs are collected; implemented here as a streaming
    # accumulate to match score_matrix's masked_byte_sum pattern.
    exp_out = "ws_exp_data_out" if h_dimm == 1 else "ws_exp_0_data_out"
    exp_valid = "ws_exp_valid_n" if h_dimm == 1 else "ws_exp_0_valid_n"
    exp_msb = "ws_exp_MSB_SA_Ready" if h_dimm == 1 else "ws_exp_0_MSB_SA_Ready"
    lines.append(f"    // Streaming reduction (7-level adder tree flattened over drain cycles)")
    lines.append(f"    reg ws_acc_clear;")
    lines.append(f"    reg [15:0] ws_col_counter;")
    lines.append(f"    reg [31:0] ws_accumulator;")
    lines.append(f"    wire [31:0] ws_masked_byte_sum;")
    mask_terms = []
    for b in range(epw):
        mask_terms.append(f"((ws_col_counter + {b} < N) ? "
                          f"{{24'b0, {exp_out}[{b}*8 +: 8]}} : 32'd0)")
    lines.append(f"    assign ws_masked_byte_sum = " + " + ".join(mask_terms) + ";")
    lines.append(f"    always @(posedge clk) begin")
    lines.append(f"        if (rst || ws_acc_clear) begin")
    lines.append(f"            ws_accumulator <= 0; ws_col_counter <= 0;")
    lines.append(f"        end else if ({exp_valid}) begin")
    lines.append(f"            ws_accumulator <= ws_accumulator + ws_masked_byte_sum;")
    lines.append(f"            ws_col_counter <= ws_col_counter + {epw};")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")

    # ws_exp output-done tracker (PACKED_N drain cycles)
    lines.append(f"    reg ws_dpe_output_done;")
    lines.append(f"    reg [15:0] ws_dpe_out_count;")
    lines.append(f"    always @(posedge clk) begin")
    lines.append(f"        if (rst || ws_acc_clear) begin")
    lines.append(f"            ws_dpe_out_count <= 0; ws_dpe_output_done <= 0;")
    lines.append(f"        end else if ({exp_valid}) begin")
    lines.append(f"            ws_dpe_out_count <= ws_dpe_out_count + 1;")
    lines.append(f"            if (ws_dpe_out_count + 1 >= PACKED_N) ws_dpe_output_done <= 1;")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")

    # Output SRAM
    lines.append(f"    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;")
    lines.append(f"    reg out_w_en;")
    lines.append(f"    reg [DATA_WIDTH-1:0] out_write_data;")
    lines.append(f"    wire [DATA_WIDTH-1:0] out_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_out}))")
    lines.append(f"    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),")
    lines.append(f"              .w_addr(out_write_addr),.sram_data_in(out_write_data),.sram_data_out(out_sram_out));")
    lines.append(f"")

    # FSM: single ws_log fire, then per-m (ws_exp fire + reduce + write)
    #   WS_LOG_FEED   — pulse ws_log_valid for PACKED_N strobes + SRAM latency
    #   WS_LOG_DRAIN  — wait for ws_log done, packed output latched into log_attn_sram
    #   WS_EXP_FEED   — pulse ws_exp_valid for PACKED_N strobes; feed log_attn+V_m
    #   WS_EXP_DRAIN  — wait for ws_exp done + reduction accumulator settled
    #   WS_WRITE      — commit scalar to out_sram[m]; advance m or exit
    # Preserve legacy encodings for TB compatibility:
    #   WS_IDLE == 0, WS_LOAD_A == 1, WS_LOAD_V == 2, WS_OUTPUT == 5
    # (TB waits on ws_state == 3'd2 for LOAD_V, 3'd5 for OUTPUT). New
    # Phase-4 states slot into the remaining codes 3, 4, 6, 7.
    lines.append(f"    localparam WS_IDLE       = 4'd0,")
    lines.append(f"               WS_LOAD_A     = 4'd1,")
    lines.append(f"               WS_LOAD_V     = 4'd2,")
    lines.append(f"               WS_LOG_FEED   = 4'd3,")
    lines.append(f"               WS_LOG_DRAIN  = 4'd4,")
    lines.append(f"               WS_OUTPUT     = 4'd5,")
    lines.append(f"               WS_EXP_FEED   = 4'd6,")
    lines.append(f"               WS_EXP_DRAIN  = 4'd7,")
    lines.append(f"               WS_WRITE      = 4'd8;")
    lines.append(f"    reg [3:0] ws_state;")
    # Packed feed counter
    pn_bits = max(1, (packed_N + 3).bit_length())
    m_bits = max(1, (d_head - 1).bit_length())
    lines.append(f"    reg [{pn_bits}-1:0] ws_feed_count;  // packed word counter (0..PACKED_N+1)")
    lines.append(f"    reg [{m_bits}-1:0]  ws_m;           // output column index")
    # Helper: single address counter for v_sram reads at v_sram[m*PACKED_N + k]
    lines.append(f"")
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst) begin")
    lines.append(f"            ws_state <= WS_IDLE; ws_feed_count <= 0;")
    lines.append(f"            ws_m <= LANE_IDX * M_PER_LANE;")
    lines.append(f"            attn_write_addr <= 0; attn_read_addr <= 0; attn_w_en <= 0;")
    lines.append(f"            v_write_addr <= 0; v_read_addr <= 0; v_w_en <= 0;")
    lines.append(f"            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;")
    lines.append(f"            out_write_data <= 0;")
    lines.append(f"            log_attn_read_addr <= 0;")
    lines.append(f"            ws_acc_clear <= 0;")
    lines.append(f"        end else begin")
    lines.append(f"            attn_w_en <= 0; v_w_en <= 0; out_w_en <= 0;")
    lines.append(f"            ws_acc_clear <= 0;")
    lines.append(f"            case (ws_state)")
    lines.append(f"                WS_IDLE: if (valid_attn || valid_v) ws_state <= WS_LOAD_A;")
    lines.append(f"                WS_LOAD_A: begin")
    # Phase-4 Opt1: pipeline ws_log feed during LOAD_A. Each cycle that
    # has written ≥1 byte also reads it back and fires ws_log on the
    # just-written byte (1-cycle SRAM latency). attn_read_addr chases
    # attn_write_addr by 1.
    lines.append(f"                    if (valid_attn) begin attn_w_en <= 1; attn_write_addr <= attn_write_addr + 1; end")
    lines.append(f"                    if (attn_write_addr > 0) attn_read_addr <= attn_write_addr - 1;")
    lines.append(f"                    if (attn_write_addr == (N/W) - 1) ws_state <= WS_LOAD_V;")
    lines.append(f"                end")
    lines.append(f"                WS_LOAD_V: begin")
    # Transposed packed V: write_addr progresses through d·PACKED_N words;
    # the TB either drives `valid_v` with packed data or overrides
    # `v_write_addr` for hierarchical preload (functional/latency TBs).
    lines.append(f"                    if (valid_v) begin v_w_en <= 1; v_write_addr <= v_write_addr + 1; end")
    lines.append(f"                    if (v_write_addr == d*PACKED_N - 1) ws_state <= WS_LOG_FEED;")
    lines.append(f"                end")
    lines.append(f"                WS_LOG_FEED: begin")
    # Phase-4 Opt1: most bytes were already fed during WS_LOAD_A overlap.
    # Here we just ensure the last byte (written in the final LOAD_A cycle)
    # gets read + fired once SRAM latency catches up. Single-cycle state.
    lines.append(f"                    attn_read_addr <= (N/W) - 1;")
    lines.append(f"                    ws_state <= WS_LOG_DRAIN;")
    lines.append(f"                end")
    lines.append(f"                WS_LOG_DRAIN: begin")
    # Fast-transition: immediately enter WS_EXP_FEED. The DPE's output
    # drain + packer continues in parallel; the first log_attn_sram reads
    # from WS_EXP_FEED land on already-committed packed words thanks to the
    # ~4-cycle lead from the ws_log compute pipeline + SRAM read latency
    # masking the first strobes.
    log_msb = "ws_log_MSB_SA_Ready" if h_dimm == 1 else "ws_log_0_MSB_SA_Ready"
    lines.append(f"                    ws_state <= WS_EXP_FEED;")
    lines.append(f"                    ws_feed_count <= 0;")
    lines.append(f"                end")
    lines.append(f"                WS_EXP_FEED: begin")
    # Feed PACKED_N packed words: log_attn_sram[k] + v_sram[m·PACKED_N + k].
    # Advance addrs up to PACKED_N-1; exit after the last strobe lands in
    # the DPE (SRAM 1-cycle read latency, so feed_count==PACKED_N drives
    # the last valid strobe).
    lines.append(f"                    if (ws_feed_count < PACKED_N) begin")
    lines.append(f"                        log_attn_read_addr <= ws_feed_count;")
    lines.append(f"                        v_read_addr <= ws_m * PACKED_N + ws_feed_count;")
    lines.append(f"                    end")
    lines.append(f"                    ws_feed_count <= ws_feed_count + 1;")
    # Exit at PACKED_N-1 (last valid strobe cycle); DPE will still receive
    # all PACKED_N strobes because w_buf_en = ws_log_valid is asserted on
    # the last cycle before state change.
    lines.append(f"                    if (ws_feed_count == PACKED_N - 1) begin")
    lines.append(f"                        ws_feed_count <= 0;")
    lines.append(f"                        ws_state <= WS_EXP_DRAIN;")
    lines.append(f"                    end")
    lines.append(f"                end")
    lines.append(f"                WS_EXP_DRAIN: begin")
    lines.append(f"                    if (ws_dpe_output_done && {exp_msb}) begin")
    lines.append(f"                        out_write_data <= {{32'b0, ws_accumulator[7:0]}};")
    lines.append(f"                        out_w_en <= 1; out_write_addr <= ws_m;")
    lines.append(f"                        ws_acc_clear <= 1;")
    lines.append(f"                        // Fused WS_WRITE: commit and advance m in the same cycle")
    lines.append(f"                        if (ws_m == (LANE_IDX + 1) * M_PER_LANE - 1) begin")
    lines.append(f"                            ws_state <= WS_OUTPUT;")
    lines.append(f"                        end else begin")
    lines.append(f"                            ws_m <= ws_m + 1;")
    lines.append(f"                            ws_state <= WS_EXP_FEED;")
    lines.append(f"                        end")
    lines.append(f"                    end")
    lines.append(f"                end")
    lines.append(f"                WS_WRITE: begin  // vestigial — kept for 4-bit decoder completeness")
    lines.append(f"                    ws_state <= WS_EXP_FEED;")
    lines.append(f"                end")
    lines.append(f"                WS_OUTPUT: if (ready_n) out_read_addr <= out_read_addr + 1;")
    lines.append(f"            endcase")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")
    if uid:
        lines.append(f"    assign data_out = out_sram_out ^ {data_width}'d{uid + 200};  // anti-merge")
    else:
        lines.append(f"    assign data_out = out_sram_out;")
    lines.append(f"    assign valid_n = (ws_state == WS_OUTPUT);")
    lines.append(f"    assign ready_attn = (ws_state == WS_LOAD_A || ws_state == WS_IDLE);")
    lines.append(f"    assign ready_v = (ws_state == WS_LOAD_V || ws_state == WS_IDLE);")
    lines.append(f"endmodule")
    return "\n".join(lines)


def _gen_attention_top(v: int, h: int, n_seq: int, d_head: int,
                       rows: int, cols: int,
                       depth: int, addr_width: int,
                       data_width: int = 40) -> str:
    """Generate the attention_head top-level module."""
    lines = []
    lines.append(f"module attention_head #(")
    lines.append(f"    parameter N = {n_seq},")
    lines.append(f"    parameter d = {d_head},")
    lines.append(f"    parameter DATA_WIDTH = {data_width}")
    lines.append(f")(")
    lines.append(f"    input wire clk, input wire rst,")
    lines.append(f"    input wire valid, input wire ready_n,")
    lines.append(f"    input wire [DATA_WIDTH-1:0] data_in,")
    lines.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    lines.append(f"    output wire ready, output wire valid_n")
    lines.append(f");")
    lines.append(f"")
    lines.append(f"    wire [DATA_WIDTH-1:0] data_out_q, data_out_k, data_out_v;")
    lines.append(f"    wire ready_q, valid_q, ready_k, valid_k, ready_v, valid_v;")
    lines.append(f"    wire ready_score, valid_score, ready_softmax, valid_softmax;")
    lines.append(f"    wire ready_wsum, valid_wsum;")
    lines.append(f"    wire valid_g_in, valid_g_out, ready_g_in, ready_g_out;")
    lines.append(f"    wire [DATA_WIDTH-1:0] global_sram_data_in;")
    lines.append(f"    reg [7:0] read_address, write_address;")
    lines.append(f"")

    # Q/K/V Projections
    proj_specs = [("q", "ready_k", "ready_g_in"), ("k", "ready_v", ""), ("v", "ready_score", "")]
    if v == 1 and h == 1:
        for proj, rn, r in proj_specs:
            lines.append(f"    // {proj.upper()} Projection DPE (V=1, H=1, ACAM=log)")
            lines.append(f"    conv_layer_single_dpe #(.N_CHANNELS(1),.ADDR_WIDTH({addr_width}),")
            lines.append(f"        .N_KERNELS(1),.KERNEL_WIDTH(d),.KERNEL_HEIGHT(1),")
            lines.append(f"        .W(1),.H(1),.S(1),.DEPTH({depth}),.DATA_WIDTH(DATA_WIDTH)")
            lines.append(f"    ) {proj}_projection (")
            lines.append(f"        .clk(clk),.rst(rst),.valid(valid_g_out),.ready_n({rn}),")
            lines.append(f"        .data_in(data_in),.data_out(data_out_{proj}),")
            lines.append(f"        .ready({r if r else ''}){',' if r else ','}.valid_n(valid_{proj})")
            lines.append(f"    );")
            lines.append(f"")
    else:
        for proj, rn, r in proj_specs:
            lines.append(f"    // {proj.upper()} Projection ({v}x{h} tiling, ACAM=log)")
            lines.append(f"    fc_layer #(.DATA_WIDTH(DATA_WIDTH)) {proj}_projection (")
            lines.append(f"        .clk(clk),.rst(rst),.valid(valid_g_out),.ready_n({rn}),")
            lines.append(f"        .data_in(data_in),.data_out(data_out_{proj}),")
            lines.append(f"        .ready({r if r else ''}){',' if r else ','}.valid_n(valid_{proj})")
            lines.append(f"    );")
            lines.append(f"")

    # DIMM Score Matrix
    lines.append(f"    wire [DATA_WIDTH-1:0] data_out_score;")
    lines.append(f"    dimm_score_matrix #(.N(N),.d(d),.DATA_WIDTH(DATA_WIDTH))")
    lines.append(f"    score_inst (.clk(clk),.rst(rst),.valid_q(valid_q),.valid_k(valid_k),")
    lines.append(f"        .ready_n(ready_softmax),.data_in_q(data_out_q),.data_in_k(data_out_k),")
    lines.append(f"        .data_out(data_out_score),.ready_q(ready_q),.ready_k(ready_k),.valid_n(valid_score));")
    lines.append(f"")

    # Softmax
    lines.append(f"    wire [DATA_WIDTH-1:0] data_out_softmax;")
    lines.append(f"    softmax_approx #(.N(N),.d(d),.DATA_WIDTH(DATA_WIDTH))")
    lines.append(f"    softmax_inst (.clk(clk),.rst(rst),.valid(valid_score),.ready_n(ready_wsum),")
    lines.append(f"        .data_in(data_out_score),.data_out(data_out_softmax),")
    lines.append(f"        .ready(ready_softmax),.valid_n(valid_softmax));")
    lines.append(f"")

    # Weighted Sum
    lines.append(f"    wire [DATA_WIDTH-1:0] data_out_wsum;")
    lines.append(f"    dimm_weighted_sum #(.N(N),.d(d),.DATA_WIDTH(DATA_WIDTH))")
    lines.append(f"    wsum_inst (.clk(clk),.rst(rst),.valid_attn(valid_softmax),.valid_v(valid_v),")
    lines.append(f"        .ready_n(ready_g_out),.data_in_attn(data_out_softmax),.data_in_v(data_out_v),")
    lines.append(f"        .data_out(data_out_wsum),.ready_attn(ready_wsum),.ready_v(ready_v),.valid_n(valid_g_in));")
    lines.append(f"")

    # Global Controller + SRAM + address logic
    lines.append(f"    global_controller #(.N_Layers(6)) g_ctrl_inst (")
    lines.append(f"        .clk(clk),.rst(rst),.ready_L1(ready_g_in),.valid_Ln(valid_g_in),")
    lines.append(f"        .valid(valid),.ready(ready),.valid_L1(valid_g_out),.ready_Ln(ready_g_out));")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(256))")
    lines.append(f"    global_sram_inst (.clk(clk),.rst(rst),.w_en(valid_g_in),")
    lines.append(f"        .r_addr(read_address),.w_addr(write_address),")
    lines.append(f"        .sram_data_in(data_out_wsum),.sram_data_out(global_sram_data_in));")
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst) begin read_address <= 0; write_address <= 128; end")
    lines.append(f"        else begin")
    lines.append(f"            if (ready_g_out) read_address <= read_address + 1;")
    lines.append(f"            if (valid_g_out) write_address <= write_address + 1;")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"    assign data_out = global_sram_data_in;")
    lines.append(f"    assign valid_n = valid_g_in;")
    lines.append(f"endmodule")
    return "\n".join(lines)


def gen_attention_wrapper(n_seq: int, d_head: int, rows: int, cols: int,
                          output_dir: str) -> Path:
    """Generate a self-contained attention head .v file for (rows x cols) crossbar."""
    v = math.ceil(d_head / rows)
    h = math.ceil(d_head / cols)
    h_dimm = math.ceil(max(d_head, n_seq) / cols)
    dpes_per_proj = v * h
    dimm_dpes = N_DIMM_STAGES * h_dimm
    total_dpes = 3 * dpes_per_proj + dimm_dpes
    data_width = 40
    acam_eligible = (v == 1)

    # Per-SRAM depths (packed: int8 elements / (data_width/8))
    pack = data_width // 8  # 5 for 40-bit bus
    _pd = lambda n: max(1, math.ceil(n / pack))
    depth_q     = _pd(d_head)
    depth_k     = _pd(n_seq * d_head)
    depth_score = _pd(n_seq)
    depth_row   = _pd(n_seq)   # softmax row buffers
    depth_d     = _pd(d_head)  # output vector
    depth_v     = _pd(n_seq * d_head)
    # Legacy: projection SRAM depth (conv_layer_single_dpe ignores it, hardcodes 512)
    depth = 256 if (v == 1 and h == 1) else max(512, d_head)
    addr_width = max(8, math.ceil(math.log2(depth)) if depth > 1 else 1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"attention_{n_seq}_{d_head}_{rows}x{cols}.v"
    out_path = out_dir / filename

    parts = []
    parts.append(f"// Auto-generated attention head wrapper (paper Fig 6c mapping)")
    parts.append(f"// Crossbar: {rows}x{cols}  (R={rows}, C={cols})")
    parts.append(f"// Projection: V={v}, H={h}, DPEs/proj={dpes_per_proj}")
    parts.append(f"// DIMM: H_dimm={h_dimm}, 4 stages × {h_dimm} = {dimm_dpes} DPEs")
    parts.append(f"// Total DPEs: {total_dpes} = 3×{dpes_per_proj} proj + {dimm_dpes} DIMM")
    parts.append(f"// Formula: (3V+4)×H = ({3*v}+4)×{h} = {total_dpes}")
    parts.append(f"// ACAM eligible (V=1): {'yes' if acam_eligible else 'no'}")
    parts.append(f"// N={n_seq}, d={d_head}")
    parts.append(f"// NO CLB multipliers for exp/log — all nonlinear via DPE ACAM")
    parts.append(f"")

    parts.append(_gen_attention_top(v, h, n_seq, d_head, rows, cols,
                                     depth, addr_width, data_width))
    parts.append(f"")

    if not (v == 1 and h == 1):
        parts.append(f"// fc_layer — DPE tiling for projections ({v}x{h})")
        parts.append(_gen_fc_layer(v, h, d_head, d_head, rows, cols,
                                    depth, addr_width, data_width))
        parts.append(f"")

    if v > 1:
        parts.append(_gen_activation_lut_module())
        parts.append(f"")

    # DIMM modules (generated from scratch — no CLB multiply for exp/log)
    parts.append(f"// === DIMM modules (paper Fig 6c: exp/log via DPE ACAM) ===")
    parts.append(_gen_dimm_score_matrix(n_seq, d_head, h_dimm,
                                        depth_q, depth_k, depth_score, data_width))
    parts.append(f"")
    parts.append(_gen_softmax_dpe(n_seq, d_head, h_dimm,
                                   depth_row, depth_row, depth_row, data_width))
    parts.append(f"")
    parts.append(_gen_dimm_weighted_sum(n_seq, d_head, h_dimm,
                                         depth_row, depth_v, depth_row, depth_d, data_width))
    parts.append(f"")

    # Supporting modules
    parts.append(f"// === Supporting modules ===")
    parts.append(_get_supporting_modules())

    out_path.write_text("\n".join(parts))
    print(f"  Generated {filename} (V={v}, H={h}, H_dimm={h_dimm}, "
          f"proj_DPEs={3*dpes_per_proj}, DIMM_DPEs={dimm_dpes}, "
          f"total={total_dpes}, ACAM={'yes' if acam_eligible else 'no'})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate attention head RTL for NL-DPE DSE")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--n-seq", type=int, default=128)
    parser.add_argument("--d-head", type=int, default=128)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        print(f"Generating attention wrappers for all {len(DSE_CONFIGS)} configs -> {args.output_dir}")
        for rows, cols in DSE_CONFIGS:
            gen_attention_wrapper(args.n_seq, args.d_head, rows, cols, args.output_dir)
    elif args.rows and args.cols:
        gen_attention_wrapper(args.n_seq, args.d_head, args.rows, args.cols, args.output_dir)
    else:
        parser.error("Either --all or both --rows and --cols are required")


if __name__ == "__main__":
    main()
