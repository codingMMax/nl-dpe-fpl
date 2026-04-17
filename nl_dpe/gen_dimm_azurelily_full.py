#!/usr/bin/env python3
"""Generate Azure-Lily DIMM RTL — mac_qk, softmax, mac_sv.

Azure-Lily attention has no crossbar involvement in the DIMM path. MAC_QK
and MAC_SV run on FPGA DSPs; softmax exp+norm runs on CLB. This generator
lifts the dsp_mac and clb_softmax modules from gen_bert_tiny_wrapper.py
and wraps them with Q/K/V/S SRAM buffers, paralleling the NL-DPE DIMM
structure so the two architectures can be compared stage-by-stage.

Files produced under fc_verification/rtl/:
  azurelily_dimm_mac_qk_d<D>_c<C>.v         Q/K SRAM + dsp_mac(K=packed_d)
  azurelily_dimm_softmax_d<D>_n<N>.v        score SRAM + clb_softmax
  azurelily_dimm_mac_sv_d<D>_c<C>.v         S/V SRAM + dsp_mac(K=packed_n)
  azurelily_dimm_int_sop_4_stub.v           behavioral stub of int_sop_4

Usage:
    python3 nl_dpe/gen_dimm_azurelily_full.py
"""
import argparse
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from gen_gemv_wrappers import _get_supporting_modules
from gen_bert_tiny_wrapper import _gen_dsp_mac_module, _gen_clb_softmax_module


def _get_sram_module_only():
    """Azure-Lily DIMM RTL only needs the sram module from supporting.
    The full _get_supporting_modules pulls in conv_layer_single_dpe which
    references `dpe`; we don't need that for DSP/CLB-based DIMM."""
    full = _get_supporting_modules()
    # Extract just the sram module (between `module sram` and its `endmodule`)
    import re
    m = re.search(r'(module\s+sram\s+#[\s\S]*?endmodule)', full)
    assert m, "sram module not found in supporting modules"
    return m.group(1)


def _gen_int_sop_4_stub(data_width: int = 40) -> str:
    """Behavioral stub for Intel int_sop_4 primitive used by dsp_mac.
    result = ax*ay + bx*by + cx*cy + dx*dy + chainin
    Synthesizable int_sop_4 is a DSP hard block; this stub enables iverilog sim.
    """
    return """// Behavioral stub of int_sop_4 for iverilog simulation.
// Synthesis (VTR) uses the real DSP hard block from the architecture XML.
module int_sop_4 (
    input wire clk,
    input wire reset,
    input wire [11:0] mode_sigs,
    input wire signed [8:0] ax, input wire signed [8:0] ay,
    input wire signed [8:0] bx, input wire signed [8:0] by,
    input wire signed [8:0] cx, input wire signed [8:0] cy,
    input wire signed [8:0] dx, input wire signed [8:0] dy,
    input wire signed [63:0] chainin,
    output wire signed [63:0] result,
    output wire signed [63:0] chainout
);
    wire signed [17:0] p_a = ax * ay;
    wire signed [17:0] p_b = bx * by;
    wire signed [17:0] p_c = cx * cy;
    wire signed [17:0] p_d = dx * dy;
    wire signed [63:0] total = $signed(p_a) + $signed(p_b) + $signed(p_c) + $signed(p_d) + $signed(chainin);
    assign result = total;
    assign chainout = total;
endmodule
"""


def _gen_azurelily_mac_qk(n_seq: int, d_head: int, data_width: int = 40) -> str:
    """azurelily_dimm_mac_qk: Q/K SRAM + dsp_mac + score SRAM.

    Pipeline per query row i (fixed here), over all N K rows:
      For each j in 0..N-1:
        For k in 0..K_PACKED-1:
          dsp_mac.valid=1, data_a=Q_packed[k], data_b=K_packed[j,k]
        After K_PACKED cycles: dsp_mac.valid_n pulses, data_out = score[j]
        Capture score[j] → score SRAM[j]

    SRAM depths:
      Q SRAM    = ceil(d / epw) + 1
      K SRAM    = N * ceil(d / epw) + 1  (each key at word-aligned offset)
      Score SRAM = N + 1
    """
    dw = data_width
    epw = dw // 8
    packed_d = math.ceil(d_head / epw)
    depth_q = packed_d + 1
    depth_k = n_seq * packed_d + 1
    depth_score = n_seq + 1

    addr_q = max(1, (depth_q - 1).bit_length())
    addr_k = max(1, (depth_k - 1).bit_length())
    addr_s = max(1, (depth_score - 1).bit_length())

    L = []
    L.append(f"// azurelily_dimm_mac_qk — Q(d={d_head}) × K^T(N={n_seq}×d={d_head}) on DSP")
    L.append(f"// Pipeline: Q SRAM + K SRAM → dsp_mac(K={packed_d}) → score SRAM")
    L.append(f"")
    L.append(f"module azurelily_dimm_mac_qk #(")
    L.append(f"    parameter N = {n_seq},")
    L.append(f"    parameter D = {d_head},")
    L.append(f"    parameter DATA_WIDTH = {dw}")
    L.append(f")(")
    L.append(f"    input wire clk, input wire rst,")
    L.append(f"    input wire valid_q, input wire valid_k,")
    L.append(f"    input wire ready_n,")
    L.append(f"    input wire [DATA_WIDTH-1:0] data_in_q,")
    L.append(f"    input wire [DATA_WIDTH-1:0] data_in_k,")
    L.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    L.append(f"    output wire valid_n")
    L.append(f");")
    L.append(f"")
    L.append(f"    localparam EPW = DATA_WIDTH / 8;")
    L.append(f"    localparam PACKED_D = {packed_d};")
    L.append(f"")

    # FSM states
    L.append(f"    localparam S_IDLE = 3'd0, S_LOAD_Q = 3'd1, S_LOAD_K = 3'd2,")
    L.append(f"               S_COMPUTE = 3'd3, S_DRAIN = 3'd4, S_OUTPUT = 3'd5;")
    L.append(f"    reg [2:0] state;")
    L.append(f"    reg [{addr_q}-1:0] q_w_addr, q_r_addr;")
    L.append(f"    reg [{addr_k}-1:0] k_w_addr, k_r_addr;")
    L.append(f"    reg [{addr_s}-1:0] s_r_addr, s_w_addr;")
    L.append(f"    reg s_w_en;")
    L.append(f"    reg [DATA_WIDTH-1:0] s_w_data;")
    L.append(f"    reg [15:0] j_count;      // which K row (0..N-1)")
    L.append(f"    reg [15:0] k_count;      // which packed word (0..PACKED_D-1)")
    L.append(f"")

    # SRAMs
    L.append(f"    wire q_w_en = (state == S_LOAD_Q) && valid_q;")
    L.append(f"    wire [DATA_WIDTH-1:0] q_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_q}))")
    L.append(f"        q_sram (.clk(clk),.rst(rst),.w_en(q_w_en),")
    L.append(f"                .r_addr(q_r_addr),.w_addr(q_w_addr),")
    L.append(f"                .sram_data_in(data_in_q),.sram_data_out(q_sram_out));")
    L.append(f"")
    L.append(f"    wire k_w_en = (state == S_LOAD_K) && valid_k;")
    L.append(f"    wire [DATA_WIDTH-1:0] k_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_k}))")
    L.append(f"        k_sram (.clk(clk),.rst(rst),.w_en(k_w_en),")
    L.append(f"                .r_addr(k_r_addr),.w_addr(k_w_addr),")
    L.append(f"                .sram_data_in(data_in_k),.sram_data_out(k_sram_out));")
    L.append(f"")
    L.append(f"    wire [DATA_WIDTH-1:0] score_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_score}))")
    L.append(f"        score_sram (.clk(clk),.rst(rst),.w_en(s_w_en),")
    L.append(f"                    .r_addr(s_r_addr),.w_addr(s_w_addr),")
    L.append(f"                    .sram_data_in(s_w_data),.sram_data_out(score_sram_out));")
    L.append(f"")

    # dsp_mac: feed Q packed word, K packed word, accumulate K=PACKED_D times
    L.append(f"    // DSP MAC: K=PACKED_D accumulation over packed pairs of Q, K[j]")
    L.append(f"    wire dsp_valid = (state == S_COMPUTE) && (mac_count >= 2);  // 2-cyc SRAM latency")
    L.append(f"    wire [DATA_WIDTH-1:0] dsp_out;")
    L.append(f"    wire dsp_out_valid;")
    L.append(f"    reg [15:0] mac_count;")
    L.append(f"")
    L.append(f"    dsp_mac #(.DATA_WIDTH(DATA_WIDTH), .K(PACKED_D)) mac_inst (")
    L.append(f"        .clk(clk), .rst(rst),")
    L.append(f"        .valid(dsp_valid), .ready_n(1'b0),")
    L.append(f"        .data_a(q_sram_out),")
    L.append(f"        .data_b(k_sram_out),")
    L.append(f"        .data_out(dsp_out),")
    L.append(f"        .ready(), .valid_n(dsp_out_valid)")
    L.append(f"    );")
    L.append(f"")

    # FSM logic
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin")
    L.append(f"            state <= S_IDLE;")
    L.append(f"            q_w_addr <= 0; q_r_addr <= 0;")
    L.append(f"            k_w_addr <= 0; k_r_addr <= 0;")
    L.append(f"            s_r_addr <= 0; s_w_addr <= 0; s_w_en <= 0; s_w_data <= 0;")
    L.append(f"            j_count <= 0; k_count <= 0; mac_count <= 0;")
    L.append(f"        end else begin")
    L.append(f"            s_w_en <= 0;   // default")
    L.append(f"            case (state)")
    L.append(f"                S_IDLE: if (valid_q || valid_k) state <= S_LOAD_Q;")
    L.append(f"                S_LOAD_Q: begin")
    L.append(f"                    if (valid_q) q_w_addr <= q_w_addr + 1;")
    L.append(f"                    if (q_w_addr == PACKED_D) state <= S_LOAD_K;")
    L.append(f"                end")
    L.append(f"                S_LOAD_K: begin")
    L.append(f"                    if (valid_k) k_w_addr <= k_w_addr + 1;")
    L.append(f"                    if (k_w_addr == N * PACKED_D) begin")
    L.append(f"                        state <= S_COMPUTE;")
    L.append(f"                        j_count <= 0; k_count <= 0; mac_count <= 0;")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"                S_COMPUTE: begin")
    L.append(f"                    // Set SRAM addresses each cycle (2-cyc latency)")
    L.append(f"                    if (k_count < PACKED_D) begin")
    L.append(f"                        q_r_addr <= k_count;")
    L.append(f"                        k_r_addr <= j_count * PACKED_D + k_count;")
    L.append(f"                    end")
    L.append(f"                    mac_count <= mac_count + 1;")
    L.append(f"                    if (mac_count < PACKED_D + 1) k_count <= k_count + 1;")
    L.append(f"                    // When dsp_mac finishes a row: capture score and advance j")
    L.append(f"                    if (dsp_out_valid) begin")
    L.append(f"                        s_w_data <= dsp_out;")
    L.append(f"                        s_w_addr <= j_count;")
    L.append(f"                        s_w_en <= 1;")
    L.append(f"                        if (j_count == N - 1) state <= S_OUTPUT;")
    L.append(f"                        else begin")
    L.append(f"                            j_count <= j_count + 1;")
    L.append(f"                            k_count <= 0;")
    L.append(f"                            mac_count <= 0;")
    L.append(f"                        end")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"                S_OUTPUT: begin")
    L.append(f"                    if (!ready_n) begin")
    L.append(f"                        s_r_addr <= s_r_addr + 1;")
    L.append(f"                        if (s_r_addr == N - 1) state <= S_IDLE;")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"            endcase")
    L.append(f"        end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    assign valid_n = (state == S_OUTPUT);")
    L.append(f"    assign data_out = score_sram_out;")
    L.append(f"")
    L.append(f"endmodule")
    return "\n".join(L)


def _gen_azurelily_softmax_wrapper(n_seq: int, d_head: int, data_width: int = 40) -> str:
    """Standalone wrapper around clb_softmax for verification.

    The clb_softmax module has its own SRAM buffer, FSM (S_LOAD → S_INV →
    S_NORM), and implements exp + sum + reciprocal + normalize internally.
    Our wrapper adds a thin shell so we can probe the FSM phases and run
    standalone testbenches.
    """
    dw = data_width
    epw = dw // 8

    L = []
    L.append(f"// azurelily_dimm_softmax — wrapper around clb_softmax for verification")
    L.append(f"// Covers both softmax_exp (S_LOAD phase) and softmax_norm (S_INV + S_NORM phases)")
    L.append(f"")
    L.append(f"module azurelily_dimm_softmax #(")
    L.append(f"    parameter N = {n_seq},")
    L.append(f"    parameter DATA_WIDTH = {dw}")
    L.append(f")(")
    L.append(f"    input wire clk, input wire rst,")
    L.append(f"    input wire valid, input wire ready_n,")
    L.append(f"    input wire [DATA_WIDTH-1:0] data_in,")
    L.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    L.append(f"    output wire ready, output wire valid_n")
    L.append(f");")
    L.append(f"    clb_softmax #(.DATA_WIDTH(DATA_WIDTH), .N(N)) inst (")
    L.append(f"        .clk(clk), .rst(rst),")
    L.append(f"        .valid(valid), .ready_n(ready_n),")
    L.append(f"        .data_in(data_in),")
    L.append(f"        .data_out(data_out),")
    L.append(f"        .ready(ready), .valid_n(valid_n)")
    L.append(f"    );")
    L.append(f"endmodule")
    return "\n".join(L)


def _gen_azurelily_mac_sv(n_seq: int, d_head: int, data_width: int = 40) -> str:
    """azurelily_dimm_mac_sv: S/V SRAM + dsp_mac + output SRAM.

    For each output column m in 0..d-1:
      For each k in 0..PACKED_N-1:
        dsp_mac.valid=1, data_a=S_packed[k], data_b=V_col_m_packed[k]
      After PACKED_N cycles: dsp_mac.valid_n, data_out = output[m]

    V is stored column-major (per-output-column packed rows). This means the
    V SRAM is arranged so that column m's N values are at addresses
    m*PACKED_N + 0..PACKED_N-1.
    """
    dw = data_width
    epw = dw // 8
    packed_n = math.ceil(n_seq / epw)
    packed_d = math.ceil(d_head / epw)
    depth_s = packed_n + 1
    depth_v = d_head * packed_n + 1
    # One SRAM entry per output (no packing) — simpler capture, each word has low
    # byte containing one score. TBs check low 8 bits of o_sram.mem[m].
    depth_o = d_head + 1

    addr_s = max(1, (depth_s - 1).bit_length())
    addr_v = max(1, (depth_v - 1).bit_length())
    addr_o = max(1, (depth_o - 1).bit_length())

    L = []
    L.append(f"// azurelily_dimm_mac_sv — S(N={n_seq}) × V(N={n_seq}×d={d_head}) on DSP")
    L.append(f"// Pipeline: S SRAM + V SRAM (col-major) → dsp_mac(K={packed_n}) → output SRAM")
    L.append(f"// Output is d packed bytes (first PACKED_D words of output SRAM)")
    L.append(f"")
    L.append(f"module azurelily_dimm_mac_sv #(")
    L.append(f"    parameter N = {n_seq},")
    L.append(f"    parameter D = {d_head},")
    L.append(f"    parameter DATA_WIDTH = {dw}")
    L.append(f")(")
    L.append(f"    input wire clk, input wire rst,")
    L.append(f"    input wire valid_s, input wire valid_v,")
    L.append(f"    input wire ready_n,")
    L.append(f"    input wire [DATA_WIDTH-1:0] data_in_s,")
    L.append(f"    input wire [DATA_WIDTH-1:0] data_in_v,")
    L.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    L.append(f"    output wire valid_n")
    L.append(f");")
    L.append(f"")
    L.append(f"    localparam EPW = DATA_WIDTH / 8;")
    L.append(f"    localparam PACKED_N = {packed_n};")
    L.append(f"    localparam PACKED_D = {packed_d};")
    L.append(f"")
    L.append(f"    localparam S_IDLE = 3'd0, S_LOAD_S = 3'd1, S_LOAD_V = 3'd2,")
    L.append(f"               S_COMPUTE = 3'd3, S_OUTPUT = 3'd4;")
    L.append(f"    reg [2:0] state;")
    L.append(f"    reg [{addr_s}-1:0] s_w_addr, s_r_addr;")
    L.append(f"    reg [{addr_v}-1:0] v_w_addr, v_r_addr;")
    L.append(f"    reg [{addr_o}-1:0] o_r_addr, o_w_addr;")
    L.append(f"    reg o_w_en;")
    L.append(f"    reg [DATA_WIDTH-1:0] o_w_data;")
    L.append(f"    reg [15:0] m_count;     // output column index (0..D-1)")
    L.append(f"    reg [15:0] k_count;     // inner index (0..PACKED_N-1)")
    L.append(f"    reg [15:0] mac_count;")
    L.append(f"")

    L.append(f"    wire s_w_en = (state == S_LOAD_S) && valid_s;")
    L.append(f"    wire [DATA_WIDTH-1:0] s_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_s}))")
    L.append(f"        s_sram (.clk(clk),.rst(rst),.w_en(s_w_en),")
    L.append(f"                .r_addr(s_r_addr),.w_addr(s_w_addr),")
    L.append(f"                .sram_data_in(data_in_s),.sram_data_out(s_sram_out));")
    L.append(f"")
    L.append(f"    wire v_w_en = (state == S_LOAD_V) && valid_v;")
    L.append(f"    wire [DATA_WIDTH-1:0] v_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_v}))")
    L.append(f"        v_sram (.clk(clk),.rst(rst),.w_en(v_w_en),")
    L.append(f"                .r_addr(v_r_addr),.w_addr(v_w_addr),")
    L.append(f"                .sram_data_in(data_in_v),.sram_data_out(v_sram_out));")
    L.append(f"")
    L.append(f"    wire [DATA_WIDTH-1:0] o_sram_out;")
    L.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH({depth_o}))")
    L.append(f"        o_sram (.clk(clk),.rst(rst),.w_en(o_w_en),")
    L.append(f"                .r_addr(o_r_addr),.w_addr(o_w_addr),")
    L.append(f"                .sram_data_in(o_w_data),.sram_data_out(o_sram_out));")
    L.append(f"")
    L.append(f"    wire dsp_valid = (state == S_COMPUTE) && (mac_count >= 2);")
    L.append(f"    wire [DATA_WIDTH-1:0] dsp_out;")
    L.append(f"    wire dsp_out_valid;")
    L.append(f"    dsp_mac #(.DATA_WIDTH(DATA_WIDTH), .K(PACKED_N)) mac_inst (")
    L.append(f"        .clk(clk), .rst(rst),")
    L.append(f"        .valid(dsp_valid), .ready_n(1'b0),")
    L.append(f"        .data_a(s_sram_out),")
    L.append(f"        .data_b(v_sram_out),")
    L.append(f"        .data_out(dsp_out),")
    L.append(f"        .ready(), .valid_n(dsp_out_valid)")
    L.append(f"    );")
    L.append(f"")

    L.append(f"    // Capture dsp_out directly to o_sram — one entry per output")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin o_w_en <= 0; o_w_addr <= 0; o_w_data <= 0; end")
    L.append(f"        else if (dsp_out_valid && state == S_COMPUTE) begin")
    L.append(f"            o_w_en <= 1;")
    L.append(f"            o_w_addr <= m_count;")
    L.append(f"            o_w_data <= dsp_out;")
    L.append(f"        end else begin o_w_en <= 0; end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    always @(posedge clk or posedge rst) begin")
    L.append(f"        if (rst) begin")
    L.append(f"            state <= S_IDLE;")
    L.append(f"            s_w_addr <= 0; s_r_addr <= 0;")
    L.append(f"            v_w_addr <= 0; v_r_addr <= 0;")
    L.append(f"            o_r_addr <= 0;")
    L.append(f"            m_count <= 0; k_count <= 0; mac_count <= 0;")
    L.append(f"        end else begin")
    L.append(f"            case (state)")
    L.append(f"                S_IDLE: if (valid_s || valid_v) state <= S_LOAD_S;")
    L.append(f"                S_LOAD_S: begin")
    L.append(f"                    if (valid_s) s_w_addr <= s_w_addr + 1;")
    L.append(f"                    if (s_w_addr == PACKED_N) state <= S_LOAD_V;")
    L.append(f"                end")
    L.append(f"                S_LOAD_V: begin")
    L.append(f"                    if (valid_v) v_w_addr <= v_w_addr + 1;")
    L.append(f"                    if (v_w_addr == D * PACKED_N) begin")
    L.append(f"                        state <= S_COMPUTE;")
    L.append(f"                        m_count <= 0; k_count <= 0; mac_count <= 0;")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"                S_COMPUTE: begin")
    L.append(f"                    if (k_count < PACKED_N) begin")
    L.append(f"                        s_r_addr <= k_count;")
    L.append(f"                        v_r_addr <= m_count * PACKED_N + k_count;")
    L.append(f"                    end")
    L.append(f"                    mac_count <= mac_count + 1;")
    L.append(f"                    if (mac_count < PACKED_N + 1) k_count <= k_count + 1;")
    L.append(f"                    if (dsp_out_valid) begin")
    L.append(f"                        if (m_count == D - 1) begin")
    L.append(f"                            state <= S_OUTPUT;")
    L.append(f"                        end else begin")
    L.append(f"                            m_count <= m_count + 1;")
    L.append(f"                            k_count <= 0;")
    L.append(f"                            mac_count <= 0;")
    L.append(f"                        end")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"                S_OUTPUT: begin")
    L.append(f"                    if (!ready_n) begin")
    L.append(f"                        o_r_addr <= o_r_addr + 1;")
    L.append(f"                        if (o_r_addr == D - 1) state <= S_IDLE;")
    L.append(f"                    end")
    L.append(f"                end")
    L.append(f"            endcase")
    L.append(f"        end")
    L.append(f"    end")
    L.append(f"")
    L.append(f"    assign valid_n = (state == S_OUTPUT);")
    L.append(f"    assign data_out = o_sram_out;")
    L.append(f"")
    L.append(f"endmodule")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "fc_verification" / "rtl"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dw = 40
    supporting = _get_sram_module_only()
    dsp_mac_mod = _gen_dsp_mac_module(data_width=dw)
    # Patch existing bug: `output reg data_out` conflicts with `assign data_out = accum`.
    # (clb_softmax legitimately uses `output reg` — non-blocking assign.)
    dsp_mac_mod = dsp_mac_mod.replace(
        "output reg  [DATA_WIDTH-1:0]  data_out,",
        "output wire [DATA_WIDTH-1:0]  data_out,")
    clb_softmax_mod = _gen_clb_softmax_module(data_width=dw)
    # Patch: w_en registered 1 cycle late causes SRAM[0] to be skipped.
    # Replace registered w_en with combinational enable.
    clb_softmax_mod = clb_softmax_mod.replace(
        "    reg [ADDR_W-1:0] w_addr, r_addr;\n    reg w_en;",
        "    reg [ADDR_W-1:0] w_addr, r_addr;\n    wire w_en;  // combinational fix for SRAM[0] skip bug")
    clb_softmax_mod = clb_softmax_mod.replace(
        "            S_LOAD: begin\n                w_en <= valid;\n                if (valid) begin",
        "            S_LOAD: begin\n                // w_en is now combinational — see assign below\n                if (valid) begin")
    clb_softmax_mod = clb_softmax_mod.replace(
        "                    if (w_addr == N-1) begin\n                        state <= S_INV; w_addr <= 0; w_en <= 0; r_addr <= 0;\n                    end else",
        "                    if (w_addr == N-1) begin\n                        state <= S_INV; w_addr <= 0; r_addr <= 0;\n                    end else")
    clb_softmax_mod = clb_softmax_mod.replace(
        "            default: begin state <= S_LOAD; out_valid <= 0; w_en <= 0; end",
        "            default: begin state <= S_LOAD; out_valid <= 0; end")
    clb_softmax_mod = clb_softmax_mod.replace(
        "            sum_exp <= 0; out_valid <= 0; w_en <= 0;",
        "            sum_exp <= 0; out_valid <= 0;")
    # Fix: last output in S_NORM gets dropped because out_valid clears on same
    # cycle as final data_out. Keep out_valid=1 through the final cycle.
    clb_softmax_mod = clb_softmax_mod.replace(
        "                if (r_addr == N-1) begin\n                    state <= S_LOAD; r_addr <= 0; sum_exp <= 0; out_valid <= 0;\n                end else",
        "                if (r_addr == N-1) begin\n                    state <= S_LOAD; r_addr <= 0; sum_exp <= 0; /* out_valid kept high */\n                end else")
    # Insert combinational w_en after wire decl.
    clb_softmax_mod = clb_softmax_mod.replace(
        "    wire w_en;  // combinational fix for SRAM[0] skip bug",
        "    wire w_en;  // combinational fix for SRAM[0] skip bug\n"
        "    assign w_en = (state == 3'd0 /*S_LOAD*/) && valid;")
    int_sop_4_stub = _gen_int_sop_4_stub(data_width=dw)

    # Write int_sop_4 stub (shared by all Azure-Lily RTL files)
    stub_path = out_dir / "azurelily_dimm_int_sop_4_stub.v"
    with open(stub_path, "w") as f:
        f.write(int_sop_4_stub)
    print(f"Generated: {stub_path}")

    # mac_qk
    mk_rtl = _gen_azurelily_mac_qk(n_seq=args.N, d_head=args.d, data_width=dw)
    mk_path = out_dir / f"azurelily_dimm_mac_qk_d{args.d}_c{args.C}.v"
    with open(mk_path, "w") as f:
        f.write(f"// Azure-Lily DIMM mac_qk — N={args.N}, d={args.d}\n")
        f.write(f"// Generated by gen_dimm_azurelily_full.py\n\n")
        f.write(mk_rtl)
        f.write("\n\n")
        f.write(dsp_mac_mod)
        f.write("\n\n")
        f.write(supporting)
    print(f"Generated: {mk_path}")

    # softmax
    sm_rtl = _gen_azurelily_softmax_wrapper(n_seq=args.N, d_head=args.d, data_width=dw)
    sm_path = out_dir / f"azurelily_dimm_softmax_d{args.d}_n{args.N}.v"
    with open(sm_path, "w") as f:
        f.write(f"// Azure-Lily DIMM softmax — N={args.N}, d={args.d}\n\n")
        f.write(sm_rtl)
        f.write("\n\n")
        f.write(clb_softmax_mod)
        f.write("\n\n")
        f.write(supporting)
    print(f"Generated: {sm_path}")

    # mac_sv
    ms_rtl = _gen_azurelily_mac_sv(n_seq=args.N, d_head=args.d, data_width=dw)
    ms_path = out_dir / f"azurelily_dimm_mac_sv_d{args.d}_c{args.C}.v"
    with open(ms_path, "w") as f:
        f.write(f"// Azure-Lily DIMM mac_sv — N={args.N}, d={args.d}\n\n")
        f.write(ms_rtl)
        f.write("\n\n")
        f.write(dsp_mac_mod)
        f.write("\n\n")
        f.write(supporting)
    print(f"Generated: {ms_path}")


if __name__ == "__main__":
    main()
