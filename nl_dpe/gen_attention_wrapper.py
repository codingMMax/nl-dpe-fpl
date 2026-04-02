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
                    addr_width: int, data_width: int = 40) -> str:
    """Generate a DIMM DPE stage as a conv_layer_single_dpe instance.

    This wraps a single DPE hard block with a controller and SRAM.
    For H_dimm=1, one instance suffices. For H_dimm>1, the caller
    generates multiple instances.

    The DPE crossbar stores identity weights; ACAM does exp or log.
    From VTR's perspective, this is identical to a projection DPE.

    Note: conv_layer_single_dpe hardcodes its internal SRAM to DEPTH=512
    regardless of the parameter, so parmys merges identical instances.
    The caller (gen_bert_tiny_wrapper) adds bare DPE hard blocks at the
    top level to ensure the correct total DPE count for area/routing.
    """
    lines = []
    lines.append(f"    conv_layer_single_dpe #(")
    lines.append(f"        .N_CHANNELS(1),")
    lines.append(f"        .ADDR_WIDTH({addr_width}),")
    lines.append(f"        .N_KERNELS(1),")
    lines.append(f"        .KERNEL_WIDTH({kernel_width}),")
    lines.append(f"        .KERNEL_HEIGHT(1),")
    lines.append(f"        .W(1),")
    lines.append(f"        .H(1),")
    lines.append(f"        .S(1),")
    lines.append(f"        .DEPTH({depth}),")
    lines.append(f"        .DATA_WIDTH({data_width})")
    lines.append(f"    ) {name} (")
    lines.append(f"        .clk(clk),")
    lines.append(f"        .rst(rst),")
    lines.append(f"        .valid({name}_valid),")
    lines.append(f"        .ready_n({name}_ready_n),")
    lines.append(f"        .data_in({name}_data_in),")
    lines.append(f"        .data_out({name}_data_out),")
    lines.append(f"        .ready({name}_ready),")
    lines.append(f"        .valid_n({name}_valid_n)")
    lines.append(f"    );")
    return "\n".join(lines)


def _gen_dimm_score_matrix(n_seq: int, d_head: int, h_dimm: int,
                            depth: int, addr_width: int,
                            data_width: int = 40,
                            dual_identity: bool = False) -> str:
    """Generate dimm_score_matrix: CLB add + DPE(I|exp) + CLB reduce.

    S[i][j] = Σ_m exp(log_Q[i][m] + log_K[j][m])

    When dual_identity=True (2×d_head ≤ C):
      Pack two d_head-sized identity blocks into the crossbar.
      Process two (i,j₁) and (i,j₂) elements per DPE pass.
      Uses 2× CLB adders and 2× reduction trees, same DPE count.
      Halves DPE passes → ~42% energy reduction on QK^T.
    """
    dw = data_width
    # Dual-identity: DPE processes 2×d_head elements per pass
    dpe_kw = 2 * d_head if dual_identity else d_head
    lines = []
    lines.append(f"module dimm_score_matrix #(")
    lines.append(f"    parameter N = {n_seq},")
    lines.append(f"    parameter d = {d_head},")
    lines.append(f"    parameter DATA_WIDTH = {dw},")
    lines.append(f"    parameter ADDR_WIDTH = {addr_width},")
    lines.append(f"    parameter DEPTH = {depth}")
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

    # Q SRAM
    lines.append(f"    reg [ADDR_WIDTH-1:0] q_write_addr, q_read_addr;")
    lines.append(f"    reg q_w_en;")
    lines.append(f"    wire [DATA_WIDTH-1:0] q_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
    lines.append(f"    q_sram (.clk(clk),.rst(rst),.w_en(q_w_en),.r_addr(q_read_addr),")
    lines.append(f"            .w_addr(q_write_addr),.sram_data_in(data_in_q),.sram_data_out(q_sram_out));")
    lines.append(f"")

    # K SRAM(s) — dual-identity needs two K read ports
    lines.append(f"    reg [$clog2(N*d)-1:0] k_write_addr, k_read_addr_a;")
    lines.append(f"    reg k_w_en;")
    lines.append(f"    wire [DATA_WIDTH-1:0] k_sram_out_a;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))")
    lines.append(f"    k_sram_a (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_a[$clog2(N*d)-1:0]),")
    lines.append(f"              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_a));")

    if dual_identity:
        # Second K SRAM (duplicate for second read port)
        lines.append(f"    reg [$clog2(N*d)-1:0] k_read_addr_b;")
        lines.append(f"    wire [DATA_WIDTH-1:0] k_sram_out_b;")
        lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))")
        lines.append(f"    k_sram_b (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_b[$clog2(N*d)-1:0]),")
        lines.append(f"              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_b));")
    lines.append(f"")

    # CLB adders
    lines.append(f"    // CLB adder A: log_Q + log_K[j₁] (log-domain addition)")
    lines.append(f"    wire [DATA_WIDTH-1:0] log_sum_a = q_sram_out + k_sram_out_a;")
    if dual_identity:
        lines.append(f"    // CLB adder B: log_Q + log_K[j₂] (second vector for dual-identity)")
        lines.append(f"    wire [DATA_WIDTH-1:0] log_sum_b = q_sram_out + k_sram_out_b;")
    lines.append(f"")

    # DPE(I|exp) stage — wider KERNEL_WIDTH for dual-identity
    for i in range(h_dimm):
        suffix = f"_{i}" if h_dimm > 1 else ""
        inst_name = f"dimm_exp{suffix}"
        label = "dual-identity" if dual_identity else "identity"
        lines.append(f"    // DPE(I|exp) stage{suffix}: {label} crossbar + ACAM=exp (KW={dpe_kw})")
        lines.append(f"    wire {inst_name}_valid, {inst_name}_ready_n, {inst_name}_ready, {inst_name}_valid_n;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {inst_name}_data_in, {inst_name}_data_out;")
        kw = min(dpe_kw, depth) if h_dimm == 1 else min(dpe_kw // h_dimm, depth)
        lines.append(_gen_dimm_stage(inst_name, kw, depth, addr_width, dw))
        lines.append(f"")

    # Wire DPE input — for dual-identity, interleave two vectors
    if h_dimm == 1:
        if dual_identity:
            # Feed alternating: d_head elements from sum_a, then d_head from sum_b
            lines.append(f"    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)")
            lines.append(f"    reg feed_phase;  // 0=vector A, 1=vector B")
            lines.append(f"    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;")
            lines.append(f"    assign dimm_exp_valid = (state == S_COMPUTE);")
            lines.append(f"    assign dimm_exp_ready_n = 1'b0;")
        else:
            lines.append(f"    assign dimm_exp_data_in = log_sum_a;")
            lines.append(f"    assign dimm_exp_valid = (state == S_COMPUTE);")
            lines.append(f"    assign dimm_exp_ready_n = 1'b0;")
    else:
        lines.append(f"    reg [{max(1, math.ceil(math.log2(h_dimm)))-1}:0] dimm_sel;")
        for i in range(h_dimm):
            lines.append(f"    assign dimm_exp_{i}_data_in = log_sum_a;")
            lines.append(f"    assign dimm_exp_{i}_valid = (state == S_COMPUTE) && (dimm_sel == {i});")
            lines.append(f"    assign dimm_exp_{i}_ready_n = 1'b0;")
    lines.append(f"")

    # CLB accumulators — dual-identity needs two independent accumulators
    if dual_identity:
        lines.append(f"    // Dual accumulators: one per vector (A for j₁, B for j₂)")
        lines.append(f"    reg [2*DATA_WIDTH-1:0] accumulator_a, accumulator_b;")
        lines.append(f"    reg acc_phase;  // 0=accumulating A outputs, 1=accumulating B outputs")
        lines.append(f"    always @(posedge clk) begin")
        lines.append(f"        if (rst) begin accumulator_a <= 0; accumulator_b <= 0; end")
        lines.append(f"        else if (dimm_exp_valid_n) begin")
        lines.append(f"            if (!acc_phase) accumulator_a <= accumulator_a + dimm_exp_data_out;")
        lines.append(f"            else accumulator_b <= accumulator_b + dimm_exp_data_out;")
        lines.append(f"        end")
        lines.append(f"    end")
    else:
        lines.append(f"    reg [2*DATA_WIDTH-1:0] accumulator;")
        dpe_done_signal = "dimm_exp_valid_n" if h_dimm == 1 else " || ".join(f"dimm_exp_{i}_valid_n" for i in range(h_dimm))
        if h_dimm == 1:
            lines.append(f"    always @(posedge clk) begin")
            lines.append(f"        if (rst) accumulator <= 0;")
            lines.append(f"        else if (dimm_exp_valid_n) accumulator <= accumulator + dimm_exp_data_out;")
            lines.append(f"    end")
        else:
            lines.append(f"    wire dimm_exp_any_valid_n = {dpe_done_signal};")
            lines.append(f"    reg [DATA_WIDTH-1:0] dimm_exp_mux;")
            lines.append(f"    always @(*) begin")
            lines.append(f"        case (dimm_sel)")
            for i in range(h_dimm):
                lines.append(f"            {i}: dimm_exp_mux = dimm_exp_{i}_data_out;")
            lines.append(f"            default: dimm_exp_mux = 0;")
            lines.append(f"        endcase")
            lines.append(f"    end")
            lines.append(f"    always @(posedge clk) begin")
            lines.append(f"        if (rst) accumulator <= 0;")
            lines.append(f"        else if (dimm_exp_any_valid_n) accumulator <= accumulator + dimm_exp_mux;")
            lines.append(f"    end")
    lines.append(f"")

    # Score SRAM
    lines.append(f"    reg [ADDR_WIDTH-1:0] score_write_addr, score_read_addr;")
    lines.append(f"    reg score_w_en;")
    lines.append(f"    reg [DATA_WIDTH-1:0] score_write_data;")
    lines.append(f"    wire [DATA_WIDTH-1:0] score_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
    lines.append(f"    score_sram (.clk(clk),.rst(rst),.w_en(score_w_en),.r_addr(score_read_addr),")
    lines.append(f"                .w_addr(score_write_addr),.sram_data_in(score_write_data),.sram_data_out(score_sram_out));")
    lines.append(f"")

    # FSM — dual-identity processes 2 score elements per iteration
    stride = 2 if dual_identity else 1
    lines.append(f"    localparam S_IDLE = 3'd0, S_LOAD_Q = 3'd1, S_LOAD_K = 3'd2,")
    lines.append(f"               S_COMPUTE = 3'd3, S_OUTPUT = 3'd4;")
    if dual_identity:
        lines.append(f"    localparam S_WRITE_B = 3'd5;  // write second score for dual-identity")
    lines.append(f"    reg [2:0] state;")
    lines.append(f"    reg [$clog2(d)-1:0] mac_count;")
    lines.append(f"    reg [$clog2(N)-1:0] score_idx;")
    if dual_identity:
        lines.append(f"    reg feed_half;  // 0=feeding vector A (d elements), 1=feeding vector B")
    lines.append(f"")
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst) begin")
    lines.append(f"            state <= S_IDLE;")
    lines.append(f"            q_write_addr <= 0; q_read_addr <= 0; q_w_en <= 0;")
    lines.append(f"            k_write_addr <= 0; k_read_addr_a <= 0; k_w_en <= 0;")
    if dual_identity:
        lines.append(f"            k_read_addr_b <= 0;")
        lines.append(f"            feed_half <= 0; feed_phase <= 0; acc_phase <= 0;")
    lines.append(f"            score_write_addr <= 0; score_read_addr <= 0; score_w_en <= 0;")
    lines.append(f"            score_write_data <= 0;")
    lines.append(f"            mac_count <= 0; score_idx <= 0;")
    if h_dimm > 1:
        lines.append(f"            dimm_sel <= 0;")
    lines.append(f"        end else begin")
    lines.append(f"            q_w_en <= 0; k_w_en <= 0; score_w_en <= 0;")
    lines.append(f"            case (state)")
    lines.append(f"                S_IDLE: if (valid_q || valid_k) state <= S_LOAD_Q;")
    lines.append(f"                S_LOAD_Q: begin")
    lines.append(f"                    if (valid_q) begin q_w_en <= 1; q_write_addr <= q_write_addr + 1; end")
    lines.append(f"                    if (q_write_addr == d-1) state <= S_LOAD_K;")
    lines.append(f"                end")
    lines.append(f"                S_LOAD_K: begin")
    lines.append(f"                    if (valid_k) begin k_w_en <= 1; k_write_addr <= k_write_addr + 1; end")
    lines.append(f"                    if (k_write_addr == N*d-1) state <= S_COMPUTE;")
    lines.append(f"                end")
    lines.append(f"                S_COMPUTE: begin")

    if dual_identity:
        # Dual-identity FSM: feed d elements for vec A, then d elements for vec B
        lines.append(f"                    // Dual-identity: feed vector A then vector B per DPE pass")
        lines.append(f"                    q_read_addr <= mac_count;")
        lines.append(f"                    k_read_addr_a <= (score_idx << $clog2(d)) + mac_count;")
        lines.append(f"                    k_read_addr_b <= ((score_idx + 1) << $clog2(d)) + mac_count;")
        lines.append(f"                    if (!feed_half) begin")
        lines.append(f"                        // Feeding vector A (log_sum_a)")
        lines.append(f"                        feed_phase <= 0; acc_phase <= 0;")
        lines.append(f"                        mac_count <= mac_count + 1;")
        lines.append(f"                        if (mac_count == d-1) begin")
        lines.append(f"                            mac_count <= 0;")
        lines.append(f"                            feed_half <= 1;")
        lines.append(f"                        end")
        lines.append(f"                    end else begin")
        lines.append(f"                        // Feeding vector B (log_sum_b)")
        lines.append(f"                        feed_phase <= 1; acc_phase <= 1;")
        lines.append(f"                        mac_count <= mac_count + 1;")
        lines.append(f"                        if (mac_count == d-1) begin")
        lines.append(f"                            // Both vectors done — write score A, then B")
        lines.append(f"                            score_write_data <= accumulator_a[2*DATA_WIDTH-1:DATA_WIDTH];")
        lines.append(f"                            score_w_en <= 1;")
        lines.append(f"                            score_write_addr <= score_idx;")
        lines.append(f"                            mac_count <= 0;")
        lines.append(f"                            feed_half <= 0;")
        lines.append(f"                            state <= S_WRITE_B;")
        lines.append(f"                        end")
        lines.append(f"                    end")
        lines.append(f"                end")
        lines.append(f"                S_WRITE_B: begin")
        lines.append(f"                    // Write second score (j₂ = score_idx + 1)")
        lines.append(f"                    score_write_data <= accumulator_b[2*DATA_WIDTH-1:DATA_WIDTH];")
        lines.append(f"                    score_w_en <= 1;")
        lines.append(f"                    score_write_addr <= score_idx + 1;")
        lines.append(f"                    accumulator_a <= 0; accumulator_b <= 0;")
        lines.append(f"                    if (score_idx >= N-2) state <= S_OUTPUT;")
        lines.append(f"                    else begin score_idx <= score_idx + {stride}; state <= S_COMPUTE; end")
    else:
        lines.append(f"                    // Feed log_sum to DPE, accumulate exp results")
        lines.append(f"                    q_read_addr <= mac_count;")
        lines.append(f"                    k_read_addr_a <= (score_idx << $clog2(d)) + mac_count;")
        lines.append(f"                    mac_count <= mac_count + 1;")
        lines.append(f"                    if (mac_count == d-1) begin")
        lines.append(f"                        score_write_data <= accumulator[2*DATA_WIDTH-1:DATA_WIDTH];")
        lines.append(f"                        score_w_en <= 1;")
        lines.append(f"                        score_write_addr <= score_idx;")
        lines.append(f"                        mac_count <= 0;")
        lines.append(f"                        if (score_idx == N-1) state <= S_OUTPUT;")
        lines.append(f"                        else score_idx <= score_idx + 1;")
        lines.append(f"                    end")

    lines.append(f"                end")
    lines.append(f"                S_OUTPUT: if (ready_n) score_read_addr <= score_read_addr + 1;")
    lines.append(f"            endcase")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")
    lines.append(f"    assign data_out = score_sram_out;")
    lines.append(f"    assign valid_n = (state == S_OUTPUT);")
    lines.append(f"    assign ready_q = (state == S_LOAD_Q || state == S_IDLE);")
    lines.append(f"    assign ready_k = (state == S_LOAD_K || state == S_IDLE);")
    lines.append(f"endmodule")
    return "\n".join(lines)


def _gen_softmax_dpe(n_seq: int, d_head: int, h_dimm: int,
                      depth: int, addr_width: int,
                      data_width: int = 40) -> str:
    """Generate softmax_approx: DPE(I|exp) + CLB sum + reciprocal + multiply.

    attn[i][j] = exp(S[i][j]) / Σ_k exp(S[i][k])

    - DPE(I|exp) computes exp(score) via ACAM
    - CLB adder tree sums exp values
    - CLB priority-encoder computes 1/sum (no multiplier needed for reciprocal)
    - CLB multiply: exp_val × inv_sum (this is normalization, NOT Taylor exp)
    """
    dw = data_width
    lines = []
    lines.append(f"module softmax_approx #(")
    lines.append(f"    parameter N = {n_seq},")
    lines.append(f"    parameter d = {d_head},")
    lines.append(f"    parameter DATA_WIDTH = {dw},")
    lines.append(f"    parameter ADDR_WIDTH = {addr_width},")
    lines.append(f"    parameter DEPTH = {depth}")
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
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
    lines.append(f"    in_sram (.clk(clk),.rst(rst),.w_en(in_w_en),.r_addr(in_read_addr),")
    lines.append(f"             .w_addr(in_write_addr),.sram_data_in(data_in),.sram_data_out(in_sram_out));")
    lines.append(f"")

    # DPE(I|exp) stage
    for i in range(h_dimm):
        suffix = f"_{i}" if h_dimm > 1 else ""
        inst_name = f"sm_exp{suffix}"
        lines.append(f"    wire {inst_name}_valid, {inst_name}_ready_n, {inst_name}_ready, {inst_name}_valid_n;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {inst_name}_data_in, {inst_name}_data_out;")
        kw = min(n_seq, depth) if h_dimm == 1 else min(n_seq // h_dimm, depth)
        lines.append(_gen_dimm_stage(inst_name, kw, depth, addr_width, dw))
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
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
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
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
    lines.append(f"    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),")
    lines.append(f"              .w_addr(out_write_addr),.sram_data_in(norm_val),.sram_data_out(out_sram_out));")
    lines.append(f"")

    # FSM
    lines.append(f"    localparam SM_IDLE = 3'd0, SM_LOAD = 3'd1, SM_EXP = 3'd2,")
    lines.append(f"               SM_NORMALIZE = 3'd3, SM_OUTPUT = 3'd4;")
    lines.append(f"    reg [2:0] sm_state;")
    lines.append(f"    reg [$clog2(N)-1:0] sm_count;")
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
    lines.append(f"                    if (in_write_addr == N-1) sm_state <= SM_EXP;")
    lines.append(f"                end")
    lines.append(f"                SM_EXP: begin")
    lines.append(f"                    // DPE(I|exp) processes scores — output goes to exp_sram + sum")
    lines.append(f"                    in_read_addr <= sm_count; sm_count <= sm_count + 1;")
    lines.append(f"                    if (sm_count == N-1) begin")
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
    lines.append(f"                    if (sm_count == N+1) sm_state <= SM_OUTPUT;")
    lines.append(f"                end")
    lines.append(f"                SM_OUTPUT: if (!ready_n) out_read_addr <= out_read_addr + 1;")
    lines.append(f"            endcase")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")
    lines.append(f"    assign data_out = out_sram_out;")
    lines.append(f"    assign valid_n = (sm_state == SM_OUTPUT);")
    lines.append(f"    assign ready = (sm_state == SM_LOAD || sm_state == SM_IDLE);")
    lines.append(f"endmodule")
    return "\n".join(lines)


def _gen_dimm_weighted_sum(n_seq: int, d_head: int, h_dimm: int,
                            depth: int, addr_width: int,
                            data_width: int = 40) -> str:
    """Generate dimm_weighted_sum: DPE(I|log) + CLB add + DPE(I|exp) + CLB reduce.

    O[i][m] = Σ_j exp(log(attn[i][j]) + log_V[j][m])

    - DPE(I|log) converts attention weights to log domain
    - CLB adder: log_attn + log_V (no multiply)
    - DPE(I|exp) converts back to linear domain
    - CLB accumulator sums results
    """
    dw = data_width
    lines = []
    lines.append(f"module dimm_weighted_sum #(")
    lines.append(f"    parameter N = {n_seq},")
    lines.append(f"    parameter d = {d_head},")
    lines.append(f"    parameter DATA_WIDTH = {dw},")
    lines.append(f"    parameter ADDR_WIDTH = {addr_width},")
    lines.append(f"    parameter DEPTH = {depth}")
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

    # Attn SRAM (linear domain from softmax)
    lines.append(f"    reg [ADDR_WIDTH-1:0] attn_write_addr, attn_read_addr;")
    lines.append(f"    reg attn_w_en;")
    lines.append(f"    wire [DATA_WIDTH-1:0] attn_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
    lines.append(f"    attn_sram (.clk(clk),.rst(rst),.w_en(attn_w_en),.r_addr(attn_read_addr),")
    lines.append(f"               .w_addr(attn_write_addr),.sram_data_in(data_in_attn),.sram_data_out(attn_sram_out));")
    lines.append(f"")

    # V SRAM (log domain from V projection DPE)
    lines.append(f"    reg [$clog2(N*d)-1:0] v_write_addr, v_read_addr;")
    lines.append(f"    reg v_w_en;")
    lines.append(f"    wire [DATA_WIDTH-1:0] v_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))")
    lines.append(f"    v_sram (.clk(clk),.rst(rst),.w_en(v_w_en),.r_addr(v_read_addr[$clog2(N*d)-1:0]),")
    lines.append(f"            .w_addr(v_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_v),.sram_data_out(v_sram_out));")
    lines.append(f"")

    # DPE(I|log) stage: converts attn weights to log domain
    for i in range(h_dimm):
        suffix = f"_{i}" if h_dimm > 1 else ""
        inst_name = f"ws_log{suffix}"
        lines.append(f"    wire {inst_name}_valid, {inst_name}_ready_n, {inst_name}_ready, {inst_name}_valid_n;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {inst_name}_data_in, {inst_name}_data_out;")
        kw = min(n_seq, depth) if h_dimm == 1 else min(n_seq // h_dimm, depth)
        lines.append(_gen_dimm_stage(inst_name, kw, depth, addr_width, dw))
        lines.append(f"")

    if h_dimm == 1:
        lines.append(f"    assign ws_log_data_in = attn_sram_out;")
        lines.append(f"    assign ws_log_valid = (ws_state == WS_LOG);")
        lines.append(f"    assign ws_log_ready_n = 1'b0;")
    else:
        for i in range(h_dimm):
            lines.append(f"    assign ws_log_{i}_data_in = attn_sram_out;")
            lines.append(f"    assign ws_log_{i}_valid = (ws_state == WS_LOG);")
            lines.append(f"    assign ws_log_{i}_ready_n = 1'b0;")
    lines.append(f"")

    # Log attn SRAM (stores DPE log output)
    lines.append(f"    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;")
    lines.append(f"    reg log_attn_w_en;")
    lines.append(f"    reg [DATA_WIDTH-1:0] log_attn_write_data;")
    lines.append(f"    wire [DATA_WIDTH-1:0] log_attn_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
    lines.append(f"    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),")
    lines.append(f"                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));")
    lines.append(f"")

    log_out = "ws_log_data_out" if h_dimm == 1 else "ws_log_0_data_out"
    log_valid = "ws_log_valid_n" if h_dimm == 1 else "ws_log_0_valid_n"
    lines.append(f"    always @(posedge clk) begin")
    lines.append(f"        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end")
    lines.append(f"        else if ({log_valid}) begin")
    lines.append(f"            log_attn_write_data <= {log_out};")
    lines.append(f"            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;")
    lines.append(f"        end else log_attn_w_en <= 0;")
    lines.append(f"    end")
    lines.append(f"")

    # CLB adder: log_attn + log_V (NO multiply)
    lines.append(f"    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;")
    lines.append(f"")

    # DPE(I|exp) stage: converts log sum back to linear domain
    for i in range(h_dimm):
        suffix = f"_{i}" if h_dimm > 1 else ""
        inst_name = f"ws_exp{suffix}"
        lines.append(f"    wire {inst_name}_valid, {inst_name}_ready_n, {inst_name}_ready, {inst_name}_valid_n;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {inst_name}_data_in, {inst_name}_data_out;")
        kw = min(d_head, depth) if h_dimm == 1 else min(d_head // h_dimm, depth)
        lines.append(_gen_dimm_stage(inst_name, kw, depth, addr_width, dw))
        lines.append(f"")

    if h_dimm == 1:
        lines.append(f"    assign ws_exp_data_in = ws_log_sum;")
        lines.append(f"    assign ws_exp_valid = (ws_state == WS_COMPUTE);")
        lines.append(f"    assign ws_exp_ready_n = 1'b0;")
    else:
        for i in range(h_dimm):
            lines.append(f"    assign ws_exp_{i}_data_in = ws_log_sum;")
            lines.append(f"    assign ws_exp_{i}_valid = (ws_state == WS_COMPUTE);")
            lines.append(f"    assign ws_exp_{i}_ready_n = 1'b0;")
    lines.append(f"")

    # CLB accumulator
    exp_out = "ws_exp_data_out" if h_dimm == 1 else "ws_exp_0_data_out"
    exp_valid = "ws_exp_valid_n" if h_dimm == 1 else "ws_exp_0_valid_n"
    lines.append(f"    reg [2*DATA_WIDTH-1:0] ws_accumulator;")
    lines.append(f"    always @(posedge clk) begin")
    lines.append(f"        if (rst) ws_accumulator <= 0;")
    lines.append(f"        else if ({exp_valid}) ws_accumulator <= ws_accumulator + {exp_out};")
    lines.append(f"    end")
    lines.append(f"")

    # Output SRAM
    lines.append(f"    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;")
    lines.append(f"    reg out_w_en;")
    lines.append(f"    reg [DATA_WIDTH-1:0] out_write_data;")
    lines.append(f"    wire [DATA_WIDTH-1:0] out_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
    lines.append(f"    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),")
    lines.append(f"              .w_addr(out_write_addr),.sram_data_in(out_write_data),.sram_data_out(out_sram_out));")
    lines.append(f"")

    # FSM
    lines.append(f"    localparam WS_IDLE = 3'd0, WS_LOAD_A = 3'd1, WS_LOAD_V = 3'd2,")
    lines.append(f"               WS_LOG = 3'd3, WS_COMPUTE = 3'd4, WS_OUTPUT = 3'd5;")
    lines.append(f"    reg [2:0] ws_state;")
    lines.append(f"    reg [$clog2(N)-1:0] ws_j;")
    lines.append(f"    reg [$clog2(d)-1:0] ws_m;")
    lines.append(f"")
    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst) begin")
    lines.append(f"            ws_state <= WS_IDLE; ws_j <= 0; ws_m <= 0;")
    lines.append(f"            attn_write_addr <= 0; attn_read_addr <= 0; attn_w_en <= 0;")
    lines.append(f"            v_write_addr <= 0; v_read_addr <= 0; v_w_en <= 0;")
    lines.append(f"            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;")
    lines.append(f"            out_write_data <= 0;")
    lines.append(f"        end else begin")
    lines.append(f"            attn_w_en <= 0; v_w_en <= 0; out_w_en <= 0;")
    lines.append(f"            case (ws_state)")
    lines.append(f"                WS_IDLE: if (valid_attn || valid_v) ws_state <= WS_LOAD_A;")
    lines.append(f"                WS_LOAD_A: begin")
    lines.append(f"                    if (valid_attn) begin attn_w_en <= 1; attn_write_addr <= attn_write_addr + 1; end")
    lines.append(f"                    if (attn_write_addr == N-1) ws_state <= WS_LOAD_V;")
    lines.append(f"                end")
    lines.append(f"                WS_LOAD_V: begin")
    lines.append(f"                    if (valid_v) begin v_w_en <= 1; v_write_addr <= v_write_addr + 1; end")
    lines.append(f"                    if (v_write_addr == N*d-1) ws_state <= WS_LOG;")
    lines.append(f"                end")
    lines.append(f"                WS_LOG: begin")
    lines.append(f"                    // DPE(I|log) converts attn to log domain")
    lines.append(f"                    attn_read_addr <= ws_j;")
    lines.append(f"                    ws_j <= ws_j + 1;")
    lines.append(f"                    if (ws_j == N-1) begin ws_j <= 0; ws_state <= WS_COMPUTE; end")
    lines.append(f"                end")
    lines.append(f"                WS_COMPUTE: begin")
    lines.append(f"                    // CLB add + DPE(I|exp) + accumulate")
    lines.append(f"                    log_attn_read_addr <= ws_j;")
    lines.append(f"                    v_read_addr <= (ws_j << $clog2(d)) + ws_m;")
    lines.append(f"                    ws_j <= ws_j + 1;")
    lines.append(f"                    if (ws_j == N-1) begin")
    lines.append(f"                        out_write_data <= ws_accumulator[2*DATA_WIDTH-1:DATA_WIDTH];")
    lines.append(f"                        out_w_en <= 1; out_write_addr <= ws_m;")
    lines.append(f"                        ws_j <= 0; ws_m <= ws_m + 1;")
    lines.append(f"                        if (ws_m == d-1) ws_state <= WS_OUTPUT;")
    lines.append(f"                    end")
    lines.append(f"                end")
    lines.append(f"                WS_OUTPUT: if (ready_n) out_read_addr <= out_read_addr + 1;")
    lines.append(f"            endcase")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")
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
    parts.append(_gen_dimm_score_matrix(n_seq, d_head, h_dimm, depth, addr_width, data_width))
    parts.append(f"")
    parts.append(_gen_softmax_dpe(n_seq, d_head, h_dimm, depth, addr_width, data_width))
    parts.append(f"")
    parts.append(_gen_dimm_weighted_sum(n_seq, d_head, h_dimm, depth, addr_width, data_width))
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
