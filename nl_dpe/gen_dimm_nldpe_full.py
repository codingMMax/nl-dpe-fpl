#!/usr/bin/env python3
"""Generate NL-DPE DIMM extension RTL — softmax_norm and mac_sv.

Two new DIMM stages to complement the existing score-matrix (mac_qk + softmax_exp fused):
  - nldpe_dimm_softmax_norm: identity crossbar + ACAM-log + CLB divide-by-sum
  - nldpe_dimm_mac_sv: crossbar GEMV with V-weights (gemm_log shape)

Together with the existing dimm_score_matrix, these cover the full attention DIMM scope
(= everything in attention except the Q/K/V/O FC projections).

Naming:
  - RTL: nldpe_dimm_<stage>_d<D>_c<C>.v
  - Module names in RTL file: nldpe_dimm_softmax_norm, nldpe_dimm_mac_sv

Usage:
    python3 nl_dpe/gen_dimm_nldpe_full.py                # generate default d=64, C=128
    python3 nl_dpe/gen_dimm_nldpe_full.py --N 4 --d 4    # small test variant
"""
import argparse
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Reuse supporting modules (sram, conv_controller, etc.) from existing generator.
from gen_gemv_wrappers import _get_supporting_modules


def _gen_dimm_softmax_norm(n_seq: int, crossbar_cols: int = 128,
                           data_width: int = 40) -> str:
    """Generate nldpe_dimm_softmax_norm module.

    Pipeline:
      1. S_LOAD:        receive N exp-values packed into input SRAM (not stressed here)
      2. S_COMPUTE:     stream N packed words from input SRAM through CLB sum + DPE(I|log)
      3. S_WAIT_DPE:    wait for DPE to finish output serialization
      4. S_CLB_NORM:    combinationally subtract log_sum from each output byte
                        (= divide in log domain) and write to output SRAM
      5. S_OUTPUT:      downstream consumer reads output SRAM

    CLB operations modeled:
      - Running sum (N-stage add, overlapped with feed — "free" structurally)
      - Reciprocal/log_sum (1 cycle, LUT-based)
      - Element-wise subtract (combinational byte-wise, EPW lanes)

    Input  SRAM depth = ceil(N / epw) + 1
    Output SRAM depth = ceil(N / epw) + 1
    Crossbar width C columns hits with one DPE pass when N <= C (the BERT-Tiny case).
    """
    dw = data_width
    epw = dw // 8
    C = crossbar_cols

    # Packed input: each word carries epw int8 elements
    packed_n = math.ceil(n_seq / epw)  # 26 for N=128, epw=5
    depth_in = packed_n + 1
    depth_out = packed_n + 1

    # DPE pass count. For N=128, C=128 → 1 pass (N ≤ C uses one DPE pass).
    # For N > C → multiple passes.
    dpe_passes = math.ceil(n_seq / C)
    kernel_width_elems = min(n_seq, C)           # elements consumed by DPE per pass
    kernel_width_packed = math.ceil(kernel_width_elems / epw)  # packed words per pass

    # Addr widths
    addr_in = max(1, (depth_in - 1).bit_length())
    addr_out = max(1, (depth_out - 1).bit_length())
    mac_bits = max(1, (packed_n + 2).bit_length())
    score_bits = max(1, packed_n.bit_length())

    L = []
    L.append(f"// nldpe_dimm_softmax_norm")
    L.append(f"// N={n_seq} (sequence length), C={C} (crossbar cols), DATA_WIDTH={dw}")
    L.append(f"// Pipeline: input SRAM → DPE(I|log) → CLB subtract log_sum → output SRAM")
    L.append(f"// DPE passes per invocation: {dpe_passes}  (N/C = {n_seq}/{C})")
    L.append(f"")
    L.append(f"module nldpe_dimm_softmax_norm #(")
    L.append(f"    parameter N = {n_seq},")
    L.append(f"    parameter C = {C},")
    L.append(f"    parameter DATA_WIDTH = {dw}")
    L.append(f")(")
    L.append(f"    input wire clk, input wire rst,")
    L.append(f"    input wire valid_in, input wire ready_n,")
    L.append(f"    input wire [DATA_WIDTH-1:0] data_in,")
    L.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    L.append(f"    output wire ready_in, output wire valid_n")
    L.append(f");")
    L.append(f"")
    L.append(f"    localparam EPW = DATA_WIDTH / 8;  // {epw}")
    L.append(f"    localparam PACKED_N = {packed_n};   // packed input words")
    L.append(f"    localparam KW_PACKED = {kernel_width_packed};  // packed words fed per DPE pass")
    L.append(f"")

    # ── Input SRAM ────────────────────────────────────────────────────
    L.append(f"    // Input SRAM: N={n_seq} exp-values packed into {packed_n} words")
    L.append(f"    reg [{addr_in}-1:0] in_w_addr, in_r_addr;")
    L.append(f"    wire in_w_en = (state == S_LOAD) && valid_in;")
    L.append(f"    wire [DATA_WIDTH-1:0] in_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_in}))")
    L.append(f"        in_sram (.clk(clk),.rst(rst),.w_en(in_w_en),")
    L.append(f"                 .r_addr(in_r_addr),.w_addr(in_w_addr),")
    L.append(f"                 .sram_data_in(data_in),.sram_data_out(in_sram_out));")
    L.append(f"")

    # ── Running-sum accumulator (CLB, overlapped with feed) ────────────
    L.append(f"    // CLB running sum: byte-wise, lane-parallel. One 32-bit total.")
    L.append(f"    // Accumulated during feed; used for log_sum subtract after DPE output.")
    L.append(f"    reg [31:0] sum_accum;")
    L.append(f"    wire [31:0] sum_lane_total;")
    terms = []
    for b in range(epw):
        terms.append(f"{{24'b0, in_sram_out[{b}*8 +: 8]}}")
    L.append(f"    assign sum_lane_total = " + " + ".join(terms) + ";")
    L.append(f"    wire feed_active = (state == S_COMPUTE)")
    L.append(f"                       && (mac_count >= 2) && (mac_count <= PACKED_N + 1);")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin sum_accum <= 0; end")
    L.append(f"        else if (feed_active) begin")
    L.append(f"            sum_accum <= sum_accum + sum_lane_total;")
    L.append(f"        end else if (state == S_IDLE || state == S_LOAD) begin")
    L.append(f"            sum_accum <= 0;")
    L.append(f"        end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    // log_sum = log(sum_accum) via LUT (modeled: log(x) ≈ x - 1 for int8 scale)")
    L.append(f"    // Kept as 1-cycle combinational for simulator alignment; FSM waits 1 cycle.")
    L.append(f"    wire [7:0] log_sum = sum_accum[7:0] - 8'd1;")
    L.append(f"")

    # ── DPE(I|log) stage ──────────────────────────────────────────────
    L.append(f"    // DPE(I|log): identity crossbar, ACAM configured for log.")
    L.append(f"    wire norm_dpe_dpe_done, norm_dpe_reg_full;")
    L.append(f"    wire norm_dpe_MSB_SA_Ready, norm_dpe_shift_add_done, norm_dpe_shift_add_bypass_ctrl;")
    L.append(f"    reg  [1:0] norm_dpe_nl_dpe_control;")
    L.append(f"    reg        norm_dpe_dpe_exec;")
    L.append(f"    wire [DATA_WIDTH-1:0] norm_dpe_data_in, norm_dpe_data_out;")
    L.append(f"    wire       norm_dpe_w_buf_en;")
    L.append(f"")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin norm_dpe_dpe_exec <= 0; norm_dpe_nl_dpe_control <= 0; end")
    L.append(f"        else begin")
    L.append(f"            norm_dpe_dpe_exec <= norm_dpe_reg_full;")
    L.append(f"            norm_dpe_nl_dpe_control <= norm_dpe_dpe_exec ? 2'b11 : 2'b00;")
    L.append(f"        end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    // Feed: input SRAM out → DPE data_in. Valid when SRAM output ready (2-cyc latency).")
    L.append(f"    assign norm_dpe_data_in = in_sram_out;")
    L.append(f"    assign norm_dpe_w_buf_en = (state == S_COMPUTE) &&")
    L.append(f"                               (mac_count >= 2) && (mac_count <= PACKED_N + 1);")
    L.append(f"")
    L.append(f"    dpe norm_dpe (")
    L.append(f"        .clk(clk), .reset(rst),")
    L.append(f"        .data_in(norm_dpe_data_in),")
    L.append(f"        .nl_dpe_control(norm_dpe_nl_dpe_control),")
    L.append(f"        .shift_add_control(norm_dpe_MSB_SA_Ready),")
    L.append(f"        .w_buf_en(norm_dpe_w_buf_en),")
    L.append(f"        .shift_add_bypass(1'b0),")
    L.append(f"        .load_output_reg(norm_dpe_shift_add_done),")
    L.append(f"        .load_input_reg(1'b0),")
    L.append(f"        .MSB_SA_Ready(norm_dpe_MSB_SA_Ready),")
    L.append(f"        .data_out(norm_dpe_data_out),")
    L.append(f"        .dpe_done(norm_dpe_dpe_done),")
    L.append(f"        .reg_full(norm_dpe_reg_full),")
    L.append(f"        .shift_add_done(norm_dpe_shift_add_done),")
    L.append(f"        .shift_add_bypass_ctrl(norm_dpe_shift_add_bypass_ctrl)")
    L.append(f"    );")
    L.append(f"")

    # ── Output stage: CLB subtract log_sum + write to output SRAM ────
    L.append(f"    // Output SRAM: N normalized log values packed into {packed_n} words.")
    L.append(f"    reg [{addr_out}-1:0] out_w_addr, out_r_addr;")
    L.append(f"    reg out_w_en;")
    L.append(f"    reg [DATA_WIDTH-1:0] out_write_data;")
    L.append(f"    wire [DATA_WIDTH-1:0] out_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_out}))")
    L.append(f"        out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),")
    L.append(f"                  .r_addr(out_r_addr),.w_addr(out_w_addr),")
    L.append(f"                  .sram_data_in(out_write_data),.sram_data_out(out_sram_out));")
    L.append(f"")

    L.append(f"    // CLB byte-wise subtract: out_byte[b] = dpe_out[b] - log_sum")
    L.append(f"    wire [DATA_WIDTH-1:0] norm_out_word;")
    norm_terms = []
    for b in range(epw):
        norm_terms.append(f"norm_out_word[{b}*8 +: 8] = norm_dpe_data_out[{b}*8 +: 8] - log_sum;")
    L.append(f"    genvar gn;")
    L.append(f"    generate for (gn = 0; gn < EPW; gn = gn + 1) begin : norm_sub")
    L.append(f"        assign norm_out_word[gn*8 +: 8] = norm_dpe_data_out[gn*8 +: 8] - log_sum;")
    L.append(f"    end endgenerate")
    L.append(f"")

    # ── FSM ───────────────────────────────────────────────────────────
    L.append(f"    localparam S_IDLE = 3'd0, S_LOAD = 3'd1, S_COMPUTE = 3'd2,")
    L.append(f"               S_WAIT_DPE = 3'd3, S_WRITE_OUT = 3'd4, S_OUTPUT = 3'd5;")
    L.append(f"")
    L.append(f"    reg [2:0] state;")
    L.append(f"    reg [{mac_bits}-1:0] mac_count;")
    L.append(f"    reg [{score_bits}-1:0] out_write_count;")
    L.append(f"    reg dpe_output_done;")
    L.append(f"    reg [15:0] dpe_out_count;")
    L.append(f"")
    L.append(f"    // Track DPE output progress")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin dpe_out_count <= 0; dpe_output_done <= 0; end")
    L.append(f"        else if (state == S_IDLE) begin dpe_out_count <= 0; dpe_output_done <= 0; end")
    L.append(f"        else if (norm_dpe_dpe_done) begin")
    L.append(f"            dpe_out_count <= dpe_out_count + 1;")
    L.append(f"            if (dpe_out_count + 1 >= KW_PACKED) dpe_output_done <= 1;")
    L.append(f"        end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    // Capture DPE output into output SRAM (with CLB log_sum subtract)")
    L.append(f"    reg [{addr_out}-1:0] capture_addr;")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin capture_addr <= 0; out_w_en <= 0; end")
    L.append(f"        else if (state == S_IDLE) begin capture_addr <= 0; out_w_en <= 0; end")
    L.append(f"        else if (norm_dpe_dpe_done) begin")
    L.append(f"            out_write_data <= norm_out_word;")
    L.append(f"            out_w_addr <= capture_addr;")
    L.append(f"            out_w_en <= 1;")
    L.append(f"            capture_addr <= capture_addr + 1;")
    L.append(f"        end else begin out_w_en <= 0; end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin")
    L.append(f"            state <= S_IDLE;")
    L.append(f"            in_w_addr <= 0; in_r_addr <= 0;")
    L.append(f"            out_r_addr <= 0;")
    L.append(f"            mac_count <= 0; out_write_count <= 0;")
    L.append(f"        end else begin")
    L.append(f"            case (state)")
    L.append(f"                S_IDLE: if (valid_in) state <= S_LOAD;")
    L.append(f"                S_LOAD: begin")
    L.append(f"                    if (valid_in) in_w_addr <= in_w_addr + 1;")
    L.append(f"                    if (in_w_addr == PACKED_N) begin")
    L.append(f"                        state <= S_COMPUTE;")
    L.append(f"                        mac_count <= 0;")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"                S_COMPUTE: begin")
    L.append(f"                    // Set SRAM read addr; 2-cycle SRAM latency → valid from mac=2")
    L.append(f"                    if (mac_count < PACKED_N) in_r_addr <= mac_count;")
    L.append(f"                    mac_count <= mac_count + 1;")
    L.append(f"                    if (mac_count == PACKED_N + 1) state <= S_WAIT_DPE;")
    L.append(f"                end")
    L.append(f"                S_WAIT_DPE: begin")
    L.append(f"                    if (dpe_output_done && norm_dpe_MSB_SA_Ready) begin")
    L.append(f"                        state <= S_OUTPUT;")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"                S_OUTPUT: begin")
    L.append(f"                    // Ready for downstream; loop back to idle after read")
    L.append(f"                    if (!ready_n) begin")
    L.append(f"                        out_r_addr <= out_r_addr + 1;")
    L.append(f"                        if (out_r_addr == PACKED_N) state <= S_IDLE;")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"            endcase")
    L.append(f"        end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    assign ready_in = (state == S_LOAD);")
    L.append(f"    assign valid_n = (state == S_OUTPUT);")
    L.append(f"    assign data_out = out_sram_out;")
    L.append(f"")
    L.append(f"endmodule")
    return "\n".join(L)


def _gen_dimm_mac_sv(n_seq: int, d_head: int, crossbar_rows: int = 1024,
                     crossbar_cols: int = 128, data_width: int = 40) -> str:
    """Generate nldpe_dimm_mac_sv module.

    Computes output[m] = Σ_j attn[j] × V[j][m] for m = 0..d-1.
    V weights loaded into crossbar rows (N rows × d cols). attn row streamed as input.

    Uses the FC/GEMV pattern: crossbar holds V matrix, input is attn row,
    output is d values. Log-domain fusion: attn is log-domain (ACAM-log'd),
    CLB adds log_V, DPE sums exp to get output.

    For simplicity in standalone verification, we model this as a direct
    crossbar matmul (ACAM_MODE=0, plain VMM) with V preloaded as weights.
    The log-domain semantics are verified at the simulator level where the
    full attention pipeline chains the stages.

    Input  SRAM depth = ceil(N / epw) + 1  (attn row packed)
    Output SRAM depth = ceil(d / epw) + 1  (d output values packed)
    Crossbar width: C cols (= d when d ≤ C). V rows = N.
    """
    dw = data_width
    epw = dw // 8
    C = crossbar_cols

    packed_n = math.ceil(n_seq / epw)
    packed_d = math.ceil(d_head / epw)
    depth_in = packed_n + 1
    depth_out = packed_d + 1

    # DPE passes. If N <= R (crossbar rows), one pass handles all inputs.
    dpe_passes = math.ceil(n_seq / crossbar_rows)

    addr_in = max(1, (depth_in - 1).bit_length())
    addr_out = max(1, (depth_out - 1).bit_length())
    mac_bits = max(1, (packed_n + 2).bit_length())

    L = []
    L.append(f"// nldpe_dimm_mac_sv")
    L.append(f"// N={n_seq} (seq_len, input), d={d_head} (output), C={C} (crossbar cols)")
    L.append(f"// Pipeline: attn SRAM → DPE(VMM with V weights) → output SRAM")
    L.append(f"// Crossbar weights = V matrix ({n_seq} rows × {d_head} cols), preloaded")
    L.append(f"// DPE passes per invocation: {dpe_passes}")
    L.append(f"")
    L.append(f"module nldpe_dimm_mac_sv #(")
    L.append(f"    parameter N = {n_seq},")
    L.append(f"    parameter D = {d_head},")
    L.append(f"    parameter C = {C},")
    L.append(f"    parameter DATA_WIDTH = {dw}")
    L.append(f")(")
    L.append(f"    input wire clk, input wire rst,")
    L.append(f"    input wire valid_in, input wire ready_n,")
    L.append(f"    input wire [DATA_WIDTH-1:0] data_in,")
    L.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    L.append(f"    output wire ready_in, output wire valid_n")
    L.append(f");")
    L.append(f"")
    L.append(f"    localparam EPW = DATA_WIDTH / 8;")
    L.append(f"    localparam PACKED_N = {packed_n};")
    L.append(f"    localparam PACKED_D = {packed_d};")
    L.append(f"")

    # ── Input (attn row) SRAM ─────────────────────────────────────────
    L.append(f"    reg [{addr_in}-1:0] in_w_addr, in_r_addr;")
    L.append(f"    wire in_w_en = (state == S_LOAD) && valid_in;")
    L.append(f"    wire [DATA_WIDTH-1:0] in_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_in}))")
    L.append(f"        in_sram (.clk(clk),.rst(rst),.w_en(in_w_en),")
    L.append(f"                 .r_addr(in_r_addr),.w_addr(in_w_addr),")
    L.append(f"                 .sram_data_in(data_in),.sram_data_out(in_sram_out));")
    L.append(f"")

    # ── DPE with V weights ────────────────────────────────────────────
    L.append(f"    // DPE with V-weights (preloaded). ACAM_MODE=0 for plain VMM.")
    L.append(f"    wire sv_dpe_dpe_done, sv_dpe_reg_full;")
    L.append(f"    wire sv_dpe_MSB_SA_Ready, sv_dpe_shift_add_done, sv_dpe_shift_add_bypass_ctrl;")
    L.append(f"    reg  [1:0] sv_dpe_nl_dpe_control;")
    L.append(f"    reg        sv_dpe_dpe_exec;")
    L.append(f"    wire [DATA_WIDTH-1:0] sv_dpe_data_in, sv_dpe_data_out;")
    L.append(f"    wire       sv_dpe_w_buf_en;")
    L.append(f"")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin sv_dpe_dpe_exec <= 0; sv_dpe_nl_dpe_control <= 0; end")
    L.append(f"        else begin")
    L.append(f"            sv_dpe_dpe_exec <= sv_dpe_reg_full;")
    L.append(f"            sv_dpe_nl_dpe_control <= sv_dpe_dpe_exec ? 2'b11 : 2'b00;")
    L.append(f"        end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    assign sv_dpe_data_in = in_sram_out;")
    L.append(f"    assign sv_dpe_w_buf_en = (state == S_COMPUTE) &&")
    L.append(f"                             (mac_count >= 2) && (mac_count <= PACKED_N + 1);")
    L.append(f"")
    L.append(f"    dpe sv_dpe (")
    L.append(f"        .clk(clk), .reset(rst),")
    L.append(f"        .data_in(sv_dpe_data_in),")
    L.append(f"        .nl_dpe_control(sv_dpe_nl_dpe_control),")
    L.append(f"        .shift_add_control(sv_dpe_MSB_SA_Ready),")
    L.append(f"        .w_buf_en(sv_dpe_w_buf_en),")
    L.append(f"        .shift_add_bypass(1'b0),")
    L.append(f"        .load_output_reg(sv_dpe_shift_add_done),")
    L.append(f"        .load_input_reg(1'b0),")
    L.append(f"        .MSB_SA_Ready(sv_dpe_MSB_SA_Ready),")
    L.append(f"        .data_out(sv_dpe_data_out),")
    L.append(f"        .dpe_done(sv_dpe_dpe_done),")
    L.append(f"        .reg_full(sv_dpe_reg_full),")
    L.append(f"        .shift_add_done(sv_dpe_shift_add_done),")
    L.append(f"        .shift_add_bypass_ctrl(sv_dpe_shift_add_bypass_ctrl)")
    L.append(f"    );")
    L.append(f"")

    # ── Output SRAM ───────────────────────────────────────────────────
    L.append(f"    // Output SRAM: d output values packed into {packed_d} words.")
    L.append(f"    reg [{addr_out}-1:0] out_w_addr, out_r_addr;")
    L.append(f"    reg out_w_en;")
    L.append(f"    reg [DATA_WIDTH-1:0] out_write_data;")
    L.append(f"    wire [DATA_WIDTH-1:0] out_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_out}))")
    L.append(f"        out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),")
    L.append(f"                  .r_addr(out_r_addr),.w_addr(out_w_addr),")
    L.append(f"                  .sram_data_in(out_write_data),.sram_data_out(out_sram_out));")
    L.append(f"")

    # ── FSM ───────────────────────────────────────────────────────────
    L.append(f"    localparam S_IDLE = 3'd0, S_LOAD = 3'd1, S_COMPUTE = 3'd2,")
    L.append(f"               S_WAIT_DPE = 3'd3, S_OUTPUT = 3'd4;")
    L.append(f"")
    L.append(f"    reg [2:0] state;")
    L.append(f"    reg [{mac_bits}-1:0] mac_count;")
    L.append(f"    reg dpe_output_done;")
    L.append(f"    reg [15:0] dpe_out_count;")
    L.append(f"")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin dpe_out_count <= 0; dpe_output_done <= 0; end")
    L.append(f"        else if (state == S_IDLE) begin dpe_out_count <= 0; dpe_output_done <= 0; end")
    L.append(f"        else if (sv_dpe_dpe_done) begin")
    L.append(f"            dpe_out_count <= dpe_out_count + 1;")
    L.append(f"            if (dpe_out_count + 1 >= PACKED_D) dpe_output_done <= 1;")
    L.append(f"        end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    // Capture DPE output into output SRAM (first PACKED_D words)")
    L.append(f"    reg [{addr_out}-1:0] capture_addr;")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin capture_addr <= 0; out_w_en <= 0; end")
    L.append(f"        else if (state == S_IDLE) begin capture_addr <= 0; out_w_en <= 0; end")
    L.append(f"        else if (sv_dpe_dpe_done && capture_addr < PACKED_D) begin")
    L.append(f"            out_write_data <= sv_dpe_data_out;")
    L.append(f"            out_w_addr <= capture_addr;")
    L.append(f"            out_w_en <= 1;")
    L.append(f"            capture_addr <= capture_addr + 1;")
    L.append(f"        end else begin out_w_en <= 0; end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin")
    L.append(f"            state <= S_IDLE;")
    L.append(f"            in_w_addr <= 0; in_r_addr <= 0;")
    L.append(f"            out_r_addr <= 0;")
    L.append(f"            mac_count <= 0;")
    L.append(f"        end else begin")
    L.append(f"            case (state)")
    L.append(f"                S_IDLE: if (valid_in) state <= S_LOAD;")
    L.append(f"                S_LOAD: begin")
    L.append(f"                    if (valid_in) in_w_addr <= in_w_addr + 1;")
    L.append(f"                    if (in_w_addr == PACKED_N) begin")
    L.append(f"                        state <= S_COMPUTE;")
    L.append(f"                        mac_count <= 0;")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"                S_COMPUTE: begin")
    L.append(f"                    if (mac_count < PACKED_N) in_r_addr <= mac_count;")
    L.append(f"                    mac_count <= mac_count + 1;")
    L.append(f"                    if (mac_count == PACKED_N + 1) state <= S_WAIT_DPE;")
    L.append(f"                end")
    L.append(f"                S_WAIT_DPE: begin")
    L.append(f"                    if (dpe_output_done && sv_dpe_MSB_SA_Ready) state <= S_OUTPUT;")
    L.append(f"                end")
    L.append(f"                S_OUTPUT: begin")
    L.append(f"                    if (!ready_n) begin")
    L.append(f"                        out_r_addr <= out_r_addr + 1;")
    L.append(f"                        if (out_r_addr == PACKED_D) state <= S_IDLE;")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"            endcase")
    L.append(f"        end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    assign ready_in = (state == S_LOAD);")
    L.append(f"    assign valid_n = (state == S_OUTPUT);")
    L.append(f"    assign data_out = out_sram_out;")
    L.append(f"")
    L.append(f"endmodule")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--R", type=int, default=1024)
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "fc_verification" / "rtl"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    supporting = _get_supporting_modules()

    # softmax_norm
    sn_rtl = _gen_dimm_softmax_norm(n_seq=args.N, crossbar_cols=args.C, data_width=40)
    sn_path = out_dir / f"nldpe_dimm_softmax_norm_d{args.d}_c{args.C}.v"
    with open(sn_path, "w") as f:
        f.write(f"// NL-DPE DIMM softmax_norm — N={args.N}, C={args.C}\n")
        f.write(f"// Generated by gen_dimm_nldpe_full.py\n\n")
        f.write(sn_rtl)
        f.write("\n\n")
        f.write(supporting)
    print(f"Generated: {sn_path}")

    # mac_sv
    ms_rtl = _gen_dimm_mac_sv(n_seq=args.N, d_head=args.d,
                               crossbar_rows=args.R, crossbar_cols=args.C, data_width=40)
    ms_path = out_dir / f"nldpe_dimm_mac_sv_d{args.d}_c{args.C}.v"
    with open(ms_path, "w") as f:
        f.write(f"// NL-DPE DIMM mac_sv — N={args.N}, d={args.d}, C={args.C}, R={args.R}\n")
        f.write(f"// Generated by gen_dimm_nldpe_full.py\n\n")
        f.write(ms_rtl)
        f.write("\n\n")
        f.write(supporting)
    print(f"Generated: {ms_path}")


if __name__ == "__main__":
    main()
