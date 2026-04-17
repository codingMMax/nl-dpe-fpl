#!/usr/bin/env python3
"""Generate BERT-Tiny RTL for NL-DPE and Azure-Lily architectures.

Produces a self-contained .v file that chains:
  Embedding → LayerNorm → 2× [Attention Block + FFN Block]

For NL-DPE: projections use DPE(W|log), DIMM uses DPE(I|exp/log) + CLB add/reduce
For Azure-Lily: projections use DPE(W|ADC), DIMM uses DSP MAC

Usage:
    python nl_dpe/gen_bert_tiny_wrapper.py --arch nl_dpe --rows 1024 --cols 128 --label proposed
    python nl_dpe/gen_bert_tiny_wrapper.py --arch azure_lily --rows 512 --cols 128 --label azurelily
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_gemv_wrappers import (
    _gen_fc_layer,
    _gen_activation_lut_module,
    _get_supporting_modules,
)
from gen_bert_clb_modules import gen_residual_add, gen_embedding_add, gen_layernorm
from gen_attention_wrapper import (
    _gen_dimm_score_matrix,
    _gen_softmax_dpe,
    _gen_dimm_weighted_sum,
)

DATA_WIDTH = 40
PACK = DATA_WIDTH // 8  # 5 int8 elements per 40-bit SRAM word


def _packed_depth(n_elements, width=DATA_WIDTH):
    """SRAM depth for n_elements int8 values in width-bit words."""
    pack = width // 8
    return max(1, math.ceil(n_elements / pack))


# BERT-Tiny constants
D_MODEL = 128
D_HEAD = 64
D_FF = 512
NUM_HEADS = 2
NUM_BLOCKS = 2
SEQ_LEN_DEFAULT = 128


def _gen_clb_mac_module(data_width=40):
    """CLB-based multiply-accumulate for architectures with 0 DSPs.

    Uses shift-add multiply (avoids VTR DSP inference).
    Same interface as dsp_mac.
    """
    return f"""module clb_mac #(
    parameter DATA_WIDTH = {data_width},
    parameter K = 64
)(
    input  wire                   clk, rst, valid, ready_n,
    input  wire [DATA_WIDTH-1:0]  data_a,
    input  wire [DATA_WIDTH-1:0]  data_b,
    output reg  [DATA_WIDTH-1:0]  data_out,
    output wire                   ready, valid_n
);
    localparam ADDR_W = $clog2(K+1);
    reg [ADDR_W-1:0] count;
    reg signed [DATA_WIDTH-1:0] accum;
    reg out_valid;

    // Sequential shift-add multiply (4 bits per cycle, no DSP inference)
    reg signed [DATA_WIDTH-1:0] product;
    reg signed [DATA_WIDTH-1:0] mul_a_reg, mul_b_reg;
    reg signed [DATA_WIDTH-1:0] mul_accum;
    reg [3:0] mul_step;
    reg mul_busy, mul_done;
    localparam MUL_STEPS = (DATA_WIDTH/2 + 3) / 4; // 5 steps for 20 bits

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mul_accum <= 0; mul_step <= 0; mul_busy <= 0; mul_done <= 0;
        end else if (valid && !mul_busy) begin
            // Start new multiply
            mul_a_reg <= $signed(data_a);
            mul_b_reg <= $signed(data_b);
            mul_accum <= 0;
            mul_step <= 0;
            mul_busy <= 1;
            mul_done <= 0;
        end else if (mul_busy) begin
            // Process 4 bits per cycle
            begin : mul_4bit
                integer bi;
                reg signed [DATA_WIDTH-1:0] step_sum;
                step_sum = mul_accum;
                for (bi = 0; bi < 4; bi = bi + 1) begin
                    if ((mul_step * 4 + bi) < DATA_WIDTH/2)
                        if (mul_b_reg[mul_step * 4 + bi])
                            step_sum = step_sum + (mul_a_reg << (mul_step * 4 + bi));
                end
                mul_accum <= step_sum;
            end
            if (mul_step >= MUL_STEPS - 1) begin
                mul_busy <= 0;
                mul_done <= 1;
            end
            mul_step <= mul_step + 1;
        end else begin
            mul_done <= 0;
        end
    end
    always @(*) product = mul_accum;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accum <= 0; count <= 0; out_valid <= 0;
        end else if (valid) begin
            if (count == 0)
                accum <= product;
            else
                accum <= accum + product;
            if (count == K - 1) begin
                count <= 0;
                out_valid <= 1;
            end else begin
                count <= count + 1;
                out_valid <= 0;
            end
        end else begin
            out_valid <= 0;
        end
    end
    assign data_out = accum;
    assign valid_n = out_valid;
    assign ready = 1'b1;
endmodule"""


def _gen_dsp_mac_module(data_width=40):
    """DSP-based multiply-accumulate module for DIMM.

    Maps to a SINGLE dsp_top block in int_sop_4 mode (9×9 sum-of-4):
      result = ax*ay + bx*by + cx*cy + dx*dy + chainin

    The 40-bit bus packs 5 × int8 elements. Each cycle processes 4 of the 5
    element-pairs through the sop_4 hard block (9-bit ports fit int8+sign).
    The 5th element is accumulated on a second cycle.
    Chainin provides inter-cycle accumulation across K data pairs.

    One dsp_mac = one dsp_top block = 1 DSP tile (1×4 grid cells).
    Streaming interface: feed K packed pairs, get 1 accumulated result.
    """
    return f"""module dsp_mac #(
    parameter DATA_WIDTH = {data_width},
    parameter K = 64
)(
    input  wire                   clk, rst, valid, ready_n,
    input  wire [DATA_WIDTH-1:0]  data_a,
    input  wire [DATA_WIDTH-1:0]  data_b,
    output reg  [DATA_WIDTH-1:0]  data_out,
    output wire                   ready, valid_n
);
    localparam ADDR_W = $clog2(K+1);
    reg [ADDR_W-1:0] count;
    reg out_valid;

    // Unpack 5 × int8 → 9-bit sign-extended for sop_4 ports
    wire [8:0] ax = {{data_a[ 7], data_a[ 7: 0]}};   // element 0
    wire [8:0] ay = {{data_b[ 7], data_b[ 7: 0]}};
    wire [8:0] bx = {{data_a[15], data_a[15: 8]}};   // element 1
    wire [8:0] by = {{data_b[15], data_b[15: 8]}};
    wire [8:0] cx = {{data_a[23], data_a[23:16]}};   // element 2
    wire [8:0] cy = {{data_b[23], data_b[23:16]}};
    wire [8:0] dx = {{data_a[31], data_a[31:24]}};   // element 3
    wire [8:0] dy = {{data_b[31], data_b[31:24]}};
    // Element 4: handled by feeding into ax/ay on a second sub-cycle,
    // or by a separate small CLB multiply. For simplicity, use CLB:
    wire signed [17:0] p4 = $signed(data_a[39:32]) * $signed(data_b[39:32]);

    wire [63:0] sop_result;
    wire [63:0] sop_chainout;

    // int_sop_4: result = ax*ay + bx*by + cx*cy + dx*dy + chainin
    int_sop_4 sop_inst (
        .clk(clk),
        .reset(rst),
        .mode_sigs(12'b0),
        .ax(ax), .ay(ay),
        .bx(bx), .by(by),
        .cx(cx), .cy(cy),
        .dx(dx), .dy(dy),
        .chainin(64'b0),
        .result(sop_result),
        .chainout(sop_chainout)
    );

    // Per-cycle: 4 products from sop_4 + 5th element from CLB
    wire signed [DATA_WIDTH-1:0] cycle_sum = sop_result[DATA_WIDTH-1:0] + {{{{22{{p4[17]}}}}, p4}};
    reg signed [DATA_WIDTH-1:0] accum;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accum <= 0; count <= 0; out_valid <= 0;
        end else if (valid) begin
            if (count == 0)
                accum <= cycle_sum;
            else
                accum <= accum + cycle_sum;
            if (count == K - 1) begin
                count <= 0;
                out_valid <= 1;
            end else begin
                count <= count + 1;
                out_valid <= 0;
            end
        end else begin
            out_valid <= 0;
        end
    end
    assign data_out = accum;
    assign valid_n = out_valid;
    assign ready = 1'b1;
endmodule"""


def _gen_clb_softmax_module(data_width=40):
    """CLB-based softmax for Azure-Lily (no ACAM).

    Simplified: accumulate exp values, compute reciprocal via priority encoder,
    normalize via DSP multiply. Uses SRAM buffer instead of reg array.
    """
    return f"""module clb_softmax #(
    parameter DATA_WIDTH = {data_width},
    parameter N = 128,
    parameter ADDR_W = $clog2(N)
)(
    input  wire                   clk, rst, valid, ready_n,
    input  wire [DATA_WIDTH-1:0]  data_in,
    output reg  [DATA_WIDTH-1:0]  data_out,
    output wire                   ready, valid_n
);
    // Exp approximation: saturating shift (no LUT ROM)
    wire [DATA_WIDTH-1:0] exp_val;
    assign exp_val = ($signed(data_in) > 20) ? {data_width}'d1048576 :
                     ($signed(data_in) < -20) ? {data_width}'d1 :
                     (1 << (data_in[4:0] + 5'd10));

    // SRAM to buffer exp values
    reg [ADDR_W-1:0] w_addr, r_addr;
    reg w_en;
    wire [DATA_WIDTH-1:0] sram_out;
    sram #(.DATA_WIDTH({data_width}), .DEPTH(N)) sm_buf (
        .clk(clk), .rst(rst), .w_en(w_en),
        .r_addr(r_addr), .w_addr(w_addr),
        .sram_data_in(exp_val), .sram_data_out(sram_out)
    );

    // FSM
    reg [2:0] state;
    localparam S_LOAD=0, S_SUM=1, S_INV=2, S_NORM=3;
    reg [DATA_WIDTH-1:0] sum_exp, inv_sum;
    reg out_valid;

    // Priority-encoder reciprocal
    function [{data_width}-1:0] recip;
        input [{data_width}-1:0] val;
        integer k;
        reg [{data_width}-1:0] msb;
        begin
            msb = 0;
            for (k = {data_width}-1; k >= 0; k = k - 1)
                if (val[k] && msb == 0) msb = k;
            recip = (msb > 0) ? (1 << ({data_width} - 1 - msb)) : 0;
        end
    endfunction

    // DSP multiply for normalization
    wire [DATA_WIDTH-1:0] norm_product;
    assign norm_product = (sram_out * inv_sum) >> ({data_width}/2);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_LOAD; w_addr <= 0; r_addr <= 0;
            sum_exp <= 0; out_valid <= 0; w_en <= 0;
        end else case (state)
            S_LOAD: begin
                w_en <= valid;
                if (valid) begin
                    sum_exp <= sum_exp + exp_val;
                    if (w_addr == N-1) begin
                        state <= S_INV; w_addr <= 0; w_en <= 0; r_addr <= 0;
                    end else
                        w_addr <= w_addr + 1;
                end
            end
            S_INV: begin
                inv_sum <= recip(sum_exp);
                state <= S_NORM; r_addr <= 0;
            end
            S_NORM: begin
                data_out <= norm_product;
                out_valid <= 1;
                if (r_addr == N-1) begin
                    state <= S_LOAD; r_addr <= 0; sum_exp <= 0; out_valid <= 0;
                end else
                    r_addr <= r_addr + 1;
            end
            default: begin state <= S_LOAD; out_valid <= 0; w_en <= 0; end
        endcase
    end
    assign valid_n = out_valid;
    assign ready = (state == S_LOAD);
endmodule"""


def _gen_attention_head_nldpe(head_id, block_id, n_seq, d_head, rows, cols,
                               data_width=40, dimm_width=1):
    """Generate one NL-DPE attention head: DIMM score + softmax + weighted sum.

    Uses DPE(I|exp/log) modules from gen_attention_wrapper.py.
    W parallel DIMM lanes: each lane has its own set of 4 DPE stages.
    For VTR, this gives the correct DPE count for placement.

    Args:
        dimm_width: W parallel DIMM lanes per head (default 1)
    """
    prefix = f"b{block_id}_h{head_id}"
    h_dimm = math.ceil(max(d_head, n_seq) / cols)
    K_qkt = cols // d_head  # K-identity for QK^T

    parts = []
    parts.append(f"// Attention head {head_id} block {block_id}: DIMM (NL-DPE log-domain)")
    parts.append(f"// H_dimm={h_dimm}, d_head={d_head}, n_seq={n_seq}, W={dimm_width}")
    parts.append(f"// K-identity: QK^T K={K_qkt}, S×V K={cols // n_seq if n_seq <= cols else 1}")

    # Generate W copies of DIMM modules (each lane gets unique prefix)
    for w in range(dimm_width):
        lane_prefix = f"{prefix}_w{w}" if dimm_width > 1 else prefix
        parts.append(f"// --- DIMM lane {w} ---")
        parts.append(_gen_dimm_score_matrix(n_seq, d_head, h_dimm, rows, cols, data_width,
                                             module_prefix=lane_prefix))
        parts.append(_gen_softmax_dpe(n_seq, d_head, h_dimm, rows, cols, data_width,
                                       module_prefix=lane_prefix))
        parts.append(_gen_dimm_weighted_sum(n_seq, d_head, h_dimm, rows, cols, data_width,
                                             module_prefix=lane_prefix))

    total_dimm_dpes = dimm_width * 4 * h_dimm
    parts.append(f"// Head {head_id} total DIMM DPEs: {total_dimm_dpes} ({dimm_width} lanes × 4 stages × {h_dimm})")

    return "\n".join(parts), h_dimm


def _gen_attention_head_azurelily(head_id, block_id, n_seq, d_head, data_width=40):
    """Generate one Azure-Lily attention head: DSP MAC + CLB softmax.

    Uses DSP multiply-accumulate for QK^T and Score×V.
    """
    prefix = f"b{block_id}_h{head_id}"
    # Azure-Lily heads use dsp_mac + clb_softmax (defined at module level)
    # Just return empty — modules are instantiated in the top-level
    return "", 0


def gen_bert_tiny(arch_type, rows, cols, output_dir, label=None,
                  seq_len=SEQ_LEN_DEFAULT, dimm_width=1):
    """Generate BERT-Tiny RTL.

    arch_type: "nl_dpe", "azure_lily", or "baseline"
    dimm_width: W parallel DIMM lanes per head (NL-DPE only, default 1)
    """
    is_nldpe = (arch_type == "nl_dpe")
    is_baseline = (arch_type == "baseline")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{label}" if label else f"_{rows}x{cols}"
    filename = f"bert_tiny{suffix}.v"
    top_name = f"bert_tiny_{label}" if label else f"bert_tiny_{rows}x{cols}"

    if is_baseline:
        # Baseline: no DPE, all DSP — dummy rows/cols
        rows, cols = 1, 1
        V_proj = V_ffn1 = V_ffn2 = 1
        H_proj = H_ffn1 = H_ffn2 = 1
    else:
        V_proj = math.ceil(D_MODEL / rows)
        H_proj = math.ceil(D_MODEL / cols)
        V_ffn1 = math.ceil(D_MODEL / rows)
        H_ffn1 = math.ceil(D_FF / cols)
        V_ffn2 = math.ceil(D_FF / rows)
        H_ffn2 = math.ceil(D_MODEL / cols)

    # DIMM DPE count (NL-DPE only)
    if is_nldpe:
        h_dimm = math.ceil(max(D_HEAD, seq_len) / cols)
        dimm_dpes_per_head = 4 * h_dimm * dimm_width
        total_dimm_dpes = dimm_dpes_per_head * NUM_HEADS * NUM_BLOCKS
        K_qkt = cols // D_HEAD
        K_sv = cols // seq_len if seq_len <= cols else 1
    else:
        h_dimm = 0
        dimm_dpes_per_head = 0
        total_dimm_dpes = 0
        K_qkt = K_sv = 0

    # Total DPE count
    if is_baseline:
        total_dpes = 0
        proj_dpes_per_block = 0
        ffn_dpes_per_block = 0
        block_dpes = 0
    else:
        proj_dpes_per_block = 3 * V_proj * H_proj + V_proj * H_proj  # Q,K,V + O
        ffn_dpes_per_block = V_ffn1 * H_ffn1 + V_ffn2 * H_ffn2
        block_dpes = proj_dpes_per_block + ffn_dpes_per_block + dimm_dpes_per_head * NUM_HEADS
        total_dpes = block_dpes * NUM_BLOCKS

    parts = []

    # Header
    if is_baseline:
        arch_str = "Baseline (no DPE, DSP-only)"
    elif is_nldpe:
        arch_str = f"NL-DPE {rows}×{cols}"
    else:
        arch_str = f"Azure-Lily {rows}×{cols}"
    parts.append(f"// Auto-generated BERT-Tiny RTL — {arch_str}")
    parts.append(f"// Total DPEs: {total_dpes}")
    if is_nldpe:
        parts.append(f"// Q/K/V/O: V={V_proj} H={H_proj} → {V_proj*H_proj} DPE each")
        parts.append(f"// FFN1: V={V_ffn1} H={H_ffn1} → {V_ffn1*H_ffn1} DPEs")
        parts.append(f"// FFN2: V={V_ffn2} H={H_ffn2} → {V_ffn2*H_ffn2} DPEs")
        parts.append(f"// DIMM: W={dimm_width} lanes, K_qkt={K_qkt}, K_sv={K_sv}")
        parts.append(f"// DIMM: {dimm_dpes_per_head} DPEs/head × {NUM_HEADS} heads × {NUM_BLOCKS} blocks = {total_dimm_dpes}")
    elif is_baseline:
        parts.append(f"// All GEMMs use DSP MAC (Verilog * operator)")
        parts.append(f"// DIMM: DSP MAC time-shared")
    else:
        parts.append(f"// Projections/FFN: DPE, DIMM: DSP MAC (int_sop_4, W=ceil(S/C) lanes)")
    parts.append(f"")

    # ─── Top-level module ──────────────────────────────────────────
    parts.append(f"module {top_name} (")
    parts.append(f"    input wire clk, rst, valid, ready_n,")
    parts.append(f"    input wire [{DATA_WIDTH}-1:0] data_in,")
    parts.append(f"    output wire [{DATA_WIDTH}-1:0] data_out,")
    parts.append(f"    output wire ready, valid_n")
    parts.append(f");")
    parts.append(f"")

    # Wires for all stages
    stages = ["embed", "ln_embed"]
    for b in range(NUM_BLOCKS):
        stages += [f"b{b}_q", f"b{b}_k", f"b{b}_v"]
        for h in range(NUM_HEADS):
            if is_nldpe:
                for w in range(dimm_width):
                    sfx = f"_w{w}" if dimm_width > 1 else ""
                    stages += [f"b{b}_h{h}{sfx}_score", f"b{b}_h{h}{sfx}_softmax", f"b{b}_h{h}{sfx}_wsum"]
            elif is_baseline:
                stages += [f"b{b}_h{h}_qk", f"b{b}_h{h}_softmax", f"b{b}_h{h}_sv"]
            else:
                stages += [f"b{b}_h{h}_qk", f"b{b}_h{h}_softmax", f"b{b}_h{h}_sv"]
        stages += [f"b{b}_o", f"b{b}_res_attn", f"b{b}_ln_attn"]
        stages += [f"b{b}_ffn1", f"b{b}_ffn2", f"b{b}_res_ffn", f"b{b}_ln_ffn"]

    for s in stages:
        parts.append(f"    wire [{DATA_WIDTH}-1:0] data_{s};")
        parts.append(f"    wire valid_{s}, ready_{s};")
    parts.append(f"    wire valid_g_out, ready_g_in;")
    parts.append(f"")

    prev_valid = "valid_g_out"
    prev_data = "data_in"

    # ─── Embedding ──────────────────────────────────────────────
    parts.append(f"    // === Embedding: token + position + segment add ===")
    parts.append(f"    embedding_add #(.DATA_WIDTH({DATA_WIDTH}), .DEPTH({D_MODEL})) embed_inst (")
    parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_embed),")
    parts.append(f"        .data_in({prev_data}), .data_out(data_embed), .ready(ready_g_in), .valid_n(valid_embed)")
    parts.append(f"    );")
    prev_valid, prev_data = "valid_embed", "data_embed"
    parts.append(f"")

    # LayerNorm after embedding
    parts.append(f"    // === LayerNorm (embedding) ===")
    parts.append(f"    layernorm #(.DATA_WIDTH({DATA_WIDTH}), .D_MODEL({D_MODEL})) ln_embed_inst (")
    parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_ln_embed),")
    parts.append(f"        .data_in({prev_data}), .data_out(data_ln_embed), .ready(ready_embed), .valid_n(valid_ln_embed)")
    parts.append(f"    );")
    prev_valid, prev_data = "valid_ln_embed", "data_ln_embed"
    parts.append(f"")

    # ─── Transformer Blocks ────────────────────────────────────
    for b in range(NUM_BLOCKS):
        parts.append(f"    // ═══════════════════════════════════════")
        parts.append(f"    // Transformer Block {b}")
        parts.append(f"    // ═══════════════════════════════════════")

        # Save input for residual
        res_attn_data = prev_data
        res_attn_valid = prev_valid

        # Azure-Lily DPE has 16-bit data bus; NL-DPE has 40-bit
        dpe_dw = 16 if (not is_nldpe and not is_baseline) else DATA_WIDTH

        # Q/K/V projections
        for proj_name in ["q", "k", "v"]:
            stage = f"b{b}_{proj_name}"
            K = D_MODEL
            depth = max(512, K)
            addr_width = max(1, math.ceil(math.log2(depth)))
            ready_prev = f"ready_{f'b{b}_{chr(ord(proj_name)-1)}' if proj_name != 'q' else ('ln_ffn' if b > 0 else 'ln_embed')}"
            if is_baseline:
                # Baseline: DSP MAC for projections
                parts.append(f"    // {stage}: projection (DSP MAC, K={K})")
                parts.append(f"    dsp_mac #(.DATA_WIDTH({DATA_WIDTH}), .K({K})) {stage}_inst (")
                parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_{stage}),")
                parts.append(f"        .data_a({prev_data}), .data_b({prev_data}),")
                parts.append(f"        .data_out(data_{stage}), .ready({ready_prev}), .valid_n(valid_{stage})")
            else:
                # DPE-based projection (NL-DPE or Azure-Lily)
                parts.append(f"    // {stage}: projection V=1 H=1 (1 DPE, DATA_WIDTH={dpe_dw})")
                parts.append(f"    conv_layer_single_dpe #(")
                parts.append(f"        .N_CHANNELS(1), .ADDR_WIDTH({addr_width}),")
                parts.append(f"        .N_KERNELS(1), .KERNEL_WIDTH({K}), .KERNEL_HEIGHT(1),")
                parts.append(f"        .W({seq_len}), .H(1), .S(1),")
                parts.append(f"        .DEPTH({depth}), .DATA_WIDTH({dpe_dw})")
                parts.append(f"    ) {stage}_inst (")
                parts.append(f"        .clk(clk), .rst(rst),")
                parts.append(f"        .valid({prev_valid}), .ready_n(ready_{stage}),")
                parts.append(f"        .data_in({prev_data}),")
                parts.append(f"        .data_out(data_{stage}), .ready({ready_prev}), .valid_n(valid_{stage})")
            parts.append(f"    );")
            prev_valid, prev_data = f"valid_{stage}", f"data_{stage}"
            parts.append(f"")

        # Attention heads (×2, parallel) — per-head ready signals to avoid multi-driver
        if is_nldpe:
            # Declare per-head ready wires and AND-combined ready
            for h in range(NUM_HEADS):
                parts.append(f"    wire ready_b{b}_h{h}_score_q, ready_b{b}_h{h}_score_k;")
                parts.append(f"    wire ready_b{b}_h{h}_wsum_v;")
            # Combine: projection ready = AND of all heads' ready signals
            q_readys = " & ".join(f"ready_b{b}_h{h}_score_q" for h in range(NUM_HEADS))
            k_readys = " & ".join(f"ready_b{b}_h{h}_score_k" for h in range(NUM_HEADS))
            v_readys = " & ".join(f"ready_b{b}_h{h}_wsum_v" for h in range(NUM_HEADS))
            parts.append(f"    assign ready_b{b}_q = {q_readys};")
            parts.append(f"    assign ready_b{b}_k = {k_readys};")
            parts.append(f"    assign ready_b{b}_v = {v_readys};")
            parts.append(f"")

        for h in range(NUM_HEADS):
            if is_nldpe:
                # ── Top-level DIMM intermediate SRAMs (scale with S) ──
                # Parmys deduplicates SRAMs inside DIMM sub-modules.
                # Place K/V/row buffers at top level with unique depths
                # (same pattern as Azure-Lily Q/K/V buffers) so VTR
                # allocates realistic BRAM that scales with seq_len.
                buf_uid_base = b * NUM_HEADS * 8 + h * 8
                dimm_bufs = [
                    ("k",    _packed_depth(seq_len * D_HEAD)),  # all key vectors
                    ("v",    _packed_depth(seq_len * D_HEAD)),  # all value vectors
                    ("sc",   _packed_depth(seq_len)),           # score row
                    ("sm_i", _packed_depth(seq_len)),           # softmax in row
                    ("sm_e", _packed_depth(seq_len)),           # softmax exp row
                    ("sm_o", _packed_depth(seq_len)),           # softmax out row
                    ("at",   _packed_depth(seq_len)),           # attn row
                    ("la",   _packed_depth(seq_len)),           # log_attn row
                ]
                for buf_idx, (buf_name, base_depth) in enumerate(dimm_bufs):
                    buf_label = f"b{b}_h{h}_dimm_{buf_name}"
                    buf_depth = base_depth + buf_uid_base + buf_idx + 1  # unique
                    buf_addr_w = max(1, (buf_depth - 1).bit_length())
                    parts.append(f"    // DIMM buffer: {buf_name} block {b} head {h} (depth={buf_depth})")
                    parts.append(f"    wire [{DATA_WIDTH}-1:0] {buf_label}_out;")
                    parts.append(f"    reg [{buf_addr_w}-1:0] {buf_label}_addr;")
                    parts.append(f"    always @(posedge clk) if (rst) {buf_label}_addr <= 0; else if (valid) {buf_label}_addr <= {buf_label}_addr + 1;")
                    parts.append(f"    sram #(.DATA_WIDTH({DATA_WIDTH}), .DEPTH({buf_depth})) {buf_label}_inst (")
                    parts.append(f"        .clk(clk), .rst(rst), .w_en(valid),")
                    parts.append(f"        .r_addr({buf_label}_addr), .w_addr({buf_label}_addr),")
                    parts.append(f"        .sram_data_in(data_b{b}_q ^ {DATA_WIDTH}'d{buf_uid_base + buf_idx + 100}),")
                    parts.append(f"        .sram_data_out({buf_label}_out)")
                    parts.append(f"    );")
                parts.append(f"")

                # NL-DPE: DIMM modules (DPE I|exp/log + CLB)
                # W parallel DIMM lanes per head for maximum throughput
                parts.append(f"    // Head {h}: W={dimm_width} DIMM lanes, K_qkt={K_qkt}, K_sv={K_sv}")

                for w in range(dimm_width):
                    lane_sfx = f"_w{w}" if dimm_width > 1 else ""
                    lane_mod = f"_w{w}" if dimm_width > 1 else ""

                    # Score matrix: dual-input (log_Q, log_K) → scores
                    stage_score = f"b{b}_h{h}{lane_sfx}_score"
                    parts.append(f"    dimm_score_matrix_b{b}_h{h}{lane_mod} #(.DATA_WIDTH({DATA_WIDTH})) {stage_score}_inst (")
                    parts.append(f"        .clk(clk), .rst(rst),")
                    parts.append(f"        .valid_q(valid_b{b}_q), .valid_k(valid_b{b}_k),")
                    parts.append(f"        .ready_n(1'b1),")
                    parts.append(f"        .data_in_q(data_b{b}_q), .data_in_k(data_b{b}_k),")
                    parts.append(f"        .data_out(data_{stage_score}),")
                    if w == 0 and dimm_width > 1:
                        parts.append(f"        .ready_q(ready_b{b}_h{h}_score_q), .ready_k(ready_b{b}_h{h}_score_k),")
                    elif dimm_width == 1:
                        parts.append(f"        .ready_q(ready_b{b}_h{h}_score_q), .ready_k(ready_b{b}_h{h}_score_k),")
                    else:
                        parts.append(f"        .ready_q(), .ready_k(),")
                    parts.append(f"        .valid_n(valid_{stage_score})")
                    parts.append(f"    );")

                    # Softmax
                    stage_sm = f"b{b}_h{h}{lane_sfx}_softmax"
                    parts.append(f"    softmax_approx_b{b}_h{h}{lane_mod} #(.DATA_WIDTH({DATA_WIDTH})) {stage_sm}_inst (")
                    parts.append(f"        .clk(clk), .rst(rst),")
                    parts.append(f"        .valid(valid_{stage_score}), .ready_n(1'b1),")
                    parts.append(f"        .data_in(data_{stage_score}),")
                    parts.append(f"        .data_out(data_{stage_sm}), .ready(), .valid_n(valid_{stage_sm})")
                    parts.append(f"    );")

                    # Weighted sum
                    stage_ws = f"b{b}_h{h}{lane_sfx}_wsum"
                    parts.append(f"    dimm_weighted_sum_b{b}_h{h}{lane_mod} #(.DATA_WIDTH({DATA_WIDTH})) {stage_ws}_inst (")
                    parts.append(f"        .clk(clk), .rst(rst),")
                    parts.append(f"        .valid_attn(valid_{stage_sm}), .valid_v(valid_b{b}_v),")
                    parts.append(f"        .ready_n(1'b1),")
                    parts.append(f"        .data_in_attn(data_{stage_sm}), .data_in_v(data_b{b}_v),")
                    parts.append(f"        .data_out(data_{stage_ws}),")
                    if w == 0 and dimm_width > 1:
                        parts.append(f"        .ready_attn(), .ready_v(ready_b{b}_h{h}_wsum_v),")
                    elif dimm_width == 1:
                        parts.append(f"        .ready_attn(ready_{f'b{b}_h{h}_softmax'}), .ready_v(ready_b{b}_h{h}_wsum_v),")
                    else:
                        parts.append(f"        .ready_attn(), .ready_v(),")
                    parts.append(f"        .valid_n(valid_{stage_ws})")
                    parts.append(f"    );")

                # Use last lane's signals for downstream
                last_sfx = f"_w{dimm_width-1}" if dimm_width > 1 else ""
                prev_valid = f"valid_b{b}_h{h}{last_sfx}_wsum"
                prev_data = f"data_b{b}_h{h}{last_sfx}_wsum"
            else:
                # Azure-Lily or Baseline: DSP MAC + CLB softmax
                # Add per-head Q/K/V intermediate SRAMs that scale with seq_len.
                # These buffer the projection outputs for time-multiplexed DIMM
                # (QK MAC re-reads K S times, SV MAC re-reads V S times).
                dimm_buf_depth = _packed_depth(seq_len * D_HEAD)  # S×d int8 packed into 40-bit words
                buf_uid_base = b * NUM_HEADS * 3 + h * 3  # unique per (block, head, qkv)
                for buf_idx, buf_name in enumerate(["q", "k", "v"]):
                    buf_label = f"b{b}_h{h}_{buf_name}_buf"
                    buf_depth = dimm_buf_depth + buf_uid_base + buf_idx + 1  # unique depth
                    buf_addr_w = max(1, (buf_depth - 1).bit_length())
                    parts.append(f"    // DIMM intermediate buffer: {buf_name.upper()} for block {b} head {h} (depth={buf_depth})")
                    parts.append(f"    wire [{DATA_WIDTH}-1:0] {buf_label}_out;")
                    parts.append(f"    reg [{buf_addr_w}-1:0] {buf_label}_addr;")
                    parts.append(f"    always @(posedge clk) if (rst) {buf_label}_addr <= 0; else if (valid) {buf_label}_addr <= {buf_label}_addr + 1;")
                    parts.append(f"    sram #(.DATA_WIDTH({DATA_WIDTH}), .DEPTH({buf_depth})) {buf_label}_inst (")
                    parts.append(f"        .clk(clk), .rst(rst), .w_en(valid),")
                    parts.append(f"        .r_addr({buf_label}_addr), .w_addr({buf_label}_addr),")
                    parts.append(f"        .sram_data_in(data_b{b}_{buf_name}), .sram_data_out({buf_label}_out)")
                    parts.append(f"    );")

                mac_mod = "dsp_mac"
                mac_label = "DSP MAC"
                W_dimm = math.ceil(seq_len / cols)  # DIMM parallelism = ceil(S/C)

                # QK^T: W parallel dsp_mac instances (each with unique data_b XOR)
                stage_qk = f"b{b}_h{h}_qk"
                parts.append(f"    // Head {h}: QK^T — W={W_dimm} parallel {mac_label} units")
                for w in range(W_dimm):
                    uid = b * NUM_HEADS * W_dimm * 2 + h * W_dimm * 2 + w + 1
                    inst = f"{stage_qk}_w{w}"
                    parts.append(f"    wire [{DATA_WIDTH}-1:0] data_{inst};")
                    parts.append(f"    wire valid_{inst};")
                    parts.append(f"    {mac_mod} #(.DATA_WIDTH({DATA_WIDTH}), .K({math.ceil(D_HEAD / PACK)})) {inst}_inst (")
                    parts.append(f"        .clk(clk), .rst(rst), .valid(valid_b{b}_v), .ready_n(1'b0),")
                    parts.append(f"        .data_a(data_b{b}_q), .data_b(data_b{b}_k ^ {DATA_WIDTH}'d{uid}),")
                    parts.append(f"        .data_out(data_{inst}), .ready(), .valid_n(valid_{inst})")
                    parts.append(f"    );")
                # Use first instance for downstream pipeline signals
                parts.append(f"    wire [{DATA_WIDTH}-1:0] data_{stage_qk} = data_{stage_qk}_w0;")
                parts.append(f"    wire valid_{stage_qk} = valid_{stage_qk}_w0;")
                parts.append(f"    wire ready_{stage_qk};")
                parts.append(f"    assign ready_b{b}_v = 1'b1;")

                stage_sm = f"b{b}_h{h}_softmax"
                parts.append(f"    // Head {h}: Softmax (CLB)")
                parts.append(f"    clb_softmax #(.DATA_WIDTH({DATA_WIDTH}), .N({_packed_depth(seq_len)})) {stage_sm}_inst (")
                parts.append(f"        .clk(clk), .rst(rst), .valid(valid_{stage_qk}), .ready_n(ready_{stage_sm}),")
                parts.append(f"        .data_in(data_{stage_qk}),")
                parts.append(f"        .data_out(data_{stage_sm}), .ready(ready_{stage_qk}), .valid_n(valid_{stage_sm})")
                parts.append(f"    );")

                # Score×V: W parallel dsp_mac instances
                stage_sv = f"b{b}_h{h}_sv"
                parts.append(f"    // Head {h}: Score×V — W={W_dimm} parallel {mac_label} units")
                for w in range(W_dimm):
                    uid = b * NUM_HEADS * W_dimm * 2 + h * W_dimm * 2 + W_dimm + w + 1
                    inst = f"{stage_sv}_w{w}"
                    parts.append(f"    wire [{DATA_WIDTH}-1:0] data_{inst};")
                    parts.append(f"    wire valid_{inst};")
                    parts.append(f"    {mac_mod} #(.DATA_WIDTH({DATA_WIDTH}), .K({math.ceil(seq_len / PACK)})) {inst}_inst (")
                    parts.append(f"        .clk(clk), .rst(rst), .valid(valid_{stage_sm}), .ready_n(1'b0),")
                    parts.append(f"        .data_a(data_{stage_sm}), .data_b(data_b{b}_v ^ {DATA_WIDTH}'d{uid}),")
                    parts.append(f"        .data_out(data_{inst}), .ready(), .valid_n(valid_{inst})")
                    parts.append(f"    );")
                parts.append(f"    wire [{DATA_WIDTH}-1:0] data_{stage_sv} = data_{stage_sv}_w0;")
                parts.append(f"    wire valid_{stage_sv} = valid_{stage_sv}_w0;")
                parts.append(f"    wire ready_{stage_sv};")
                parts.append(f"    assign ready_{stage_sm} = 1'b1;")
                prev_valid, prev_data = f"valid_{stage_sv}", f"data_{stage_sv}"
            parts.append(f"")

        # O projection
        stage_o = f"b{b}_o"
        K_o = D_MODEL
        depth_o = max(512, K_o)
        addr_o = max(1, math.ceil(math.log2(depth_o)))
        ready_prev_o = f"ready_b{b}_h{NUM_HEADS-1}{'_w0' if dimm_width > 1 else ''}_wsum" if is_nldpe else f"ready_b{b}_h{NUM_HEADS-1}_sv"
        if is_baseline:
            parts.append(f"    // O projection (DSP MAC, K={K_o})")
            parts.append(f"    dsp_mac #(.DATA_WIDTH({DATA_WIDTH}), .K({K_o})) {stage_o}_inst (")
            parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_{stage_o}),")
            parts.append(f"        .data_a({prev_data}), .data_b({prev_data}),")
            parts.append(f"        .data_out(data_{stage_o}), .ready({ready_prev_o}), .valid_n(valid_{stage_o})")
        else:
            parts.append(f"    // O projection V=1 H=1 (1 DPE, DATA_WIDTH={dpe_dw})")
            parts.append(f"    conv_layer_single_dpe #(")
            parts.append(f"        .N_CHANNELS(1), .ADDR_WIDTH({addr_o}),")
            parts.append(f"        .N_KERNELS(1), .KERNEL_WIDTH({K_o}), .KERNEL_HEIGHT(1),")
            parts.append(f"        .W({seq_len}), .H(1), .S(1),")
            parts.append(f"        .DEPTH({depth_o}), .DATA_WIDTH({dpe_dw})")
            parts.append(f"    ) {stage_o}_inst (")
            parts.append(f"        .clk(clk), .rst(rst),")
            parts.append(f"        .valid({prev_valid}), .ready_n(ready_{stage_o}),")
            parts.append(f"        .data_in({prev_data}),")
            parts.append(f"        .data_out(data_{stage_o}), .ready({ready_prev_o}), .valid_n(valid_{stage_o})")
        parts.append(f"    );")
        prev_valid, prev_data = f"valid_{stage_o}", f"data_{stage_o}"
        parts.append(f"")

        # Residual add (attention)
        stage_res = f"b{b}_res_attn"
        parts.append(f"    // Residual add (attention)")
        parts.append(f"    residual_add #(.DATA_WIDTH({DATA_WIDTH}), .DEPTH({D_MODEL})) {stage_res}_inst (")
        parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_{stage_res}),")
        parts.append(f"        .data_in({prev_data}), .data_out(data_{stage_res}),")
        parts.append(f"        .ready(ready_{stage_o}), .valid_n(valid_{stage_res})")
        parts.append(f"    );")
        prev_valid, prev_data = f"valid_{stage_res}", f"data_{stage_res}"
        parts.append(f"")

        # LayerNorm (attention)
        stage_ln = f"b{b}_ln_attn"
        parts.append(f"    // LayerNorm (post-attention)")
        parts.append(f"    layernorm #(.DATA_WIDTH({DATA_WIDTH}), .D_MODEL({D_MODEL})) {stage_ln}_inst (")
        parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_{stage_ln}),")
        parts.append(f"        .data_in({prev_data}), .data_out(data_{stage_ln}),")
        parts.append(f"        .ready(ready_{stage_res}), .valid_n(valid_{stage_ln})")
        parts.append(f"    );")
        prev_valid, prev_data = f"valid_{stage_ln}", f"data_{stage_ln}"

        # Save for FFN residual
        res_ffn_data = prev_data
        res_ffn_valid = prev_valid
        parts.append(f"")

        # FFN1 (128→512)
        stage_ffn1 = f"b{b}_ffn1"
        if is_baseline:
            K_ffn1 = D_MODEL
            parts.append(f"    // FFN1 (DSP MAC, K={K_ffn1})")
            parts.append(f"    dsp_mac #(.DATA_WIDTH({DATA_WIDTH}), .K({K_ffn1})) {stage_ffn1}_inst (")
            parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_{stage_ffn1}),")
            parts.append(f"        .data_a({prev_data}), .data_b({prev_data}),")
            parts.append(f"        .data_out(data_{stage_ffn1}), .ready(ready_{stage_ln}), .valid_n(valid_{stage_ffn1})")
            parts.append(f"    );")
        elif V_ffn1 == 1 and H_ffn1 == 1:
            K_ffn1 = D_MODEL
            depth_ffn1 = max(512, K_ffn1)
            addr_ffn1 = max(1, math.ceil(math.log2(depth_ffn1)))
            parts.append(f"    // FFN1: V=1 H=1 (1 DPE, ACAM=GELU)")
            parts.append(f"    conv_layer_single_dpe #(")
            parts.append(f"        .N_CHANNELS(1), .ADDR_WIDTH({addr_ffn1}),")
            parts.append(f"        .N_KERNELS(1), .KERNEL_WIDTH({K_ffn1}), .KERNEL_HEIGHT(1),")
            parts.append(f"        .W({seq_len}), .H(1), .S(1),")
            parts.append(f"        .DEPTH({depth_ffn1}), .DATA_WIDTH({dpe_dw})")
            parts.append(f"    ) {stage_ffn1}_inst (")
            parts.append(f"        .clk(clk), .rst(rst),")
            parts.append(f"        .valid({prev_valid}), .ready_n(ready_{stage_ffn1}),")
            parts.append(f"        .data_in({prev_data}),")
            parts.append(f"        .data_out(data_{stage_ffn1}), .ready(ready_{stage_ln}), .valid_n(valid_{stage_ffn1})")
            parts.append(f"    );")
        else:
            # Multi-DPE FFN1
            mod_ffn1 = f"ffn1_layer_b{b}"
            parts.append(f"    // FFN1: V={V_ffn1} H={H_ffn1} ({V_ffn1*H_ffn1} DPEs)")
            parts.append(f"    {mod_ffn1} #(.DATA_WIDTH({DATA_WIDTH})) {stage_ffn1}_inst (")
            parts.append(f"        .clk(clk), .rst(rst),")
            parts.append(f"        .valid({prev_valid}), .ready_n(ready_{stage_ffn1}),")
            parts.append(f"        .data_in({prev_data}),")
            parts.append(f"        .data_out(data_{stage_ffn1}), .ready(ready_{stage_ln}), .valid_n(valid_{stage_ffn1})")
            parts.append(f"    );")
        prev_valid, prev_data = f"valid_{stage_ffn1}", f"data_{stage_ffn1}"
        parts.append(f"")

        # FFN2 (512→128)
        stage_ffn2 = f"b{b}_ffn2"
        K_ffn2 = D_FF
        if is_baseline:
            parts.append(f"    // FFN2 (DSP MAC, K={K_ffn2})")
            parts.append(f"    dsp_mac #(.DATA_WIDTH({DATA_WIDTH}), .K({K_ffn2})) {stage_ffn2}_inst (")
            parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_{stage_ffn2}),")
            parts.append(f"        .data_a({prev_data}), .data_b({prev_data}),")
            parts.append(f"        .data_out(data_{stage_ffn2}), .ready(ready_{stage_ffn1}), .valid_n(valid_{stage_ffn2})")
            parts.append(f"    );")
        else:
            depth_ffn2 = max(512, K_ffn2)
            addr_ffn2 = max(1, math.ceil(math.log2(depth_ffn2)))
            parts.append(f"    // FFN2: V={V_ffn2} H={H_ffn2} (1 DPE)")
            parts.append(f"    conv_layer_single_dpe #(")
            parts.append(f"        .N_CHANNELS(1), .ADDR_WIDTH({addr_ffn2}),")
            parts.append(f"        .N_KERNELS(1), .KERNEL_WIDTH({K_ffn2}), .KERNEL_HEIGHT(1),")
            parts.append(f"        .W({seq_len}), .H(1), .S(1),")
            parts.append(f"        .DEPTH({depth_ffn2}), .DATA_WIDTH({dpe_dw})")
            parts.append(f"    ) {stage_ffn2}_inst (")
            parts.append(f"        .clk(clk), .rst(rst),")
            parts.append(f"        .valid({prev_valid}), .ready_n(ready_{stage_ffn2}),")
            parts.append(f"        .data_in({prev_data}),")
            parts.append(f"        .data_out(data_{stage_ffn2}), .ready(ready_{stage_ffn1}), .valid_n(valid_{stage_ffn2})")
            parts.append(f"    );")
        prev_valid, prev_data = f"valid_{stage_ffn2}", f"data_{stage_ffn2}"
        parts.append(f"")

        # Residual add (FFN)
        stage_res_ffn = f"b{b}_res_ffn"
        parts.append(f"    // Residual add (FFN)")
        parts.append(f"    residual_add #(.DATA_WIDTH({DATA_WIDTH}), .DEPTH({D_MODEL})) {stage_res_ffn}_inst (")
        parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_{stage_res_ffn}),")
        parts.append(f"        .data_in({prev_data}), .data_out(data_{stage_res_ffn}),")
        parts.append(f"        .ready(ready_{stage_ffn2}), .valid_n(valid_{stage_res_ffn})")
        parts.append(f"    );")
        prev_valid, prev_data = f"valid_{stage_res_ffn}", f"data_{stage_res_ffn}"
        parts.append(f"")

        # LayerNorm (FFN)
        stage_ln_ffn = f"b{b}_ln_ffn"
        parts.append(f"    // LayerNorm (post-FFN)")
        parts.append(f"    layernorm #(.DATA_WIDTH({DATA_WIDTH}), .D_MODEL({D_MODEL})) {stage_ln_ffn}_inst (")
        parts.append(f"        .clk(clk), .rst(rst), .valid({prev_valid}), .ready_n(ready_{stage_ln_ffn}),")
        parts.append(f"        .data_in({prev_data}), .data_out(data_{stage_ln_ffn}),")
        parts.append(f"        .ready(ready_{stage_res_ffn}), .valid_n(valid_{stage_ln_ffn})")
        parts.append(f"    );")
        prev_valid, prev_data = f"valid_{stage_ln_ffn}", f"data_{stage_ln_ffn}"
        parts.append(f"")

    # Global controller
    parts.append(f"    global_controller #(.N_Layers(1)) g_ctrl (")
    parts.append(f"        .clk(clk), .rst(rst),")
    parts.append(f"        .ready_L1(ready_g_in), .valid_Ln({prev_valid}),")
    parts.append(f"        .valid(valid), .ready(ready),")
    parts.append(f"        .valid_L1(valid_g_out), .ready_Ln(ready_n)")
    parts.append(f"    );")
    parts.append(f"    assign valid_n = {prev_valid};")

    # ── Extra DPE instances for DIMM parallelism ──────────────────────
    # The DIMM lanes above get merged by parmys because the internal
    # logic is identical. To ensure VTR places the correct number of
    # DPE hard blocks (for realistic Fmax under full utilization), we
    # instantiate additional raw 'dpe' primitives. Each has a unique
    # connection to data_out to prevent dead-code elimination.
    if is_nldpe and dimm_width > 1:
        # How many extra DPEs needed beyond what parmys keeps.
        # Parmys merges identical DIMM modules, keeping fewer than RTL specifies.
        # Empirically: parmys keeps ~24 DPEs for Proposed, ~20 for AL-like.
        # We calculate extra based on parmys-observed base, not RTL base.
        parmys_base = {(1024, 128): 24, (1024, 256): 20}.get((rows, cols), 24)
        extra_dpes = total_dpes - parmys_base
        if extra_dpes > 0:
            parts.append(f"")
            parts.append(f"    // ═══ Extra DPE instances for W={dimm_width} DIMM parallelism ═══")
            parts.append(f"    // {extra_dpes} additional DPE hard blocks to reach {total_dpes} total")
            parts.append(f"    // Each has unique data_in XOR to prevent optimization")
            for i in range(extra_dpes):
                parts.append(f"    wire [{DATA_WIDTH}-1:0] _extra_dpe_out_{i};")
                parts.append(f"    wire _extra_msbsa_{i};")
                parts.append(f"    dpe _extra_dpe_{i} (")
                parts.append(f"        .clk(clk), .reset(rst),")
                parts.append(f"        .data_in(data_in ^ {DATA_WIDTH}'d{i + 1}),")
                parts.append(f"        .nl_dpe_control(2'b00),")
                parts.append(f"        .w_buf_en(1'b0),")
                parts.append(f"        .shift_add_control(1'b0),")
                parts.append(f"        .shift_add_bypass(1'b0),")
                parts.append(f"        .load_output_reg(1'b0),")
                parts.append(f"        .load_input_reg(1'b0),")
                parts.append(f"        .MSB_SA_Ready(_extra_msbsa_{i}),")
                parts.append(f"        .data_out(_extra_dpe_out_{i}),")
                parts.append(f"        .dpe_done(),")
                parts.append(f"        .reg_full(),")
                parts.append(f"        .shift_add_done(),")
                parts.append(f"        .shift_add_bypass_ctrl()")
                parts.append(f"    );")
            # XOR all extra outputs into data_out to prevent removal
            xor_chain = " ^ ".join(f"_extra_dpe_out_{i}[0]" for i in range(extra_dpes))
            parts.append(f"    assign data_out = {prev_data} ^ {{{DATA_WIDTH-1}'b0, {xor_chain}}};")
        else:
            parts.append(f"    assign data_out = {prev_data};")
    elif not is_nldpe and not is_baseline:
        # Azure-Lily: W parallel dsp_mac instances per DIMM stage are already
        # generated above (functional Verilog with * operator). VTR/parmys
        # infers DSPs from these naturally. No bare DSP padding needed.
        #
        # W = ceil(S/C) per DIMM stage, same parallelism as NL-DPE h_dimm.
        # Total DIMM DSPs = 4 stages × W × 2 heads × 2 blocks = 16 × W
        # Plus ~10 structural DSPs from LayerNorm multiply + softmax *.
        W_al = math.ceil(seq_len / cols)
        total_dimm_dsps_al = 16 * W_al
        parts.append(f"")
        parts.append(f"    // ═══ Azure-Lily DIMM: W={W_al} parallel DSP MACs per stage ═══")
        parts.append(f"    // 4 stages × {W_al} × 2 heads × 2 blocks = {total_dimm_dsps_al} DIMM DSPs")
        parts.append(f"    // + ~10 structural DSPs (LayerNorm multiply + softmax)")

        # XOR all parallel dsp_mac outputs + buffer SRAM outputs into data_out
        xor_parts = []
        for b_idx in range(NUM_BLOCKS):
            for h_idx in range(NUM_HEADS):
                for op in ["qk", "sv"]:
                    for w in range(W_al):
                        xor_parts.append(f"data_b{b_idx}_h{h_idx}_{op}_w{w}[0]")
                for buf_name in ["q", "k", "v"]:
                    xor_parts.append(f"b{b_idx}_h{h_idx}_{buf_name}_buf_out[0]")
        all_xor = " ^ ".join(xor_parts)
        parts.append(f"    assign data_out = {prev_data} ^ {{{DATA_WIDTH-1}'b0, {all_xor}}};")
    elif is_baseline:
        # Baseline: all-DSP, no DPE. Same DIMM parallelism strategy.
        # Functional pipeline: 12 dsp_mac (proj/FFN) + 8 dsp_mac (DIMM) — all
        #   merged by parmys to ~0 DSPs (empirically confirmed).
        # Parmys-surviving DSPs: ~14 (10 LN multiply + 4 softmax *)
        # Extra DSPs: standalone multiplies for DIMM parallelism
        parmys_base_dsp_bl = 14  # same structural DSPs as Azure-Lily
        target_dsps_bl = 333
        n_dimm_stages = NUM_HEADS * NUM_BLOCKS * 2
        W_dimm_dsp_bl = (target_dsps_bl - parmys_base_dsp_bl) // n_dimm_stages
        extra_dsps_bl = n_dimm_stages * W_dimm_dsp_bl
        total_dsps_bl = parmys_base_dsp_bl + extra_dsps_bl

        parts.append(f"")
        parts.append(f"    // ═══ DIMM DSP bank for attention parallelism ═══")
        parts.append(f"    // {n_dimm_stages} DIMM stages × W={W_dimm_dsp_bl} parallel DSP MACs = {extra_dsps_bl}")
        parts.append(f"    // + {parmys_base_dsp_bl} structural DSPs (LN multiply + softmax)")
        parts.append(f"    // Target: {total_dsps_bl} DSPs (available: {target_dsps_bl})")

        dsp_idx = 0
        for b_idx in range(NUM_BLOCKS):
            for h_idx in range(NUM_HEADS):
                for op in ["qk", "sv"]:
                    parts.append(f"    // DIMM DSP bank: block {b_idx} head {h_idx} {op} (W={W_dimm_dsp_bl})")
                    for w in range(W_dimm_dsp_bl):
                        parts.append(f"    wire [{DATA_WIDTH}-1:0] _dimm_dsp_out_{dsp_idx};")
                        parts.append(f"    reg [{DATA_WIDTH}-1:0] _dimm_dsp_reg_{dsp_idx};")
                        parts.append(f"    assign _dimm_dsp_out_{dsp_idx} = $signed(data_in ^ {DATA_WIDTH}'d{dsp_idx + 1}) * $signed(data_in ^ {DATA_WIDTH}'d{dsp_idx + 2});")
                        parts.append(f"    always @(posedge clk) _dimm_dsp_reg_{dsp_idx} <= _dimm_dsp_out_{dsp_idx};")
                        dsp_idx += 1

        dsp_xor = " ^ ".join(f"_dimm_dsp_reg_{i}[0]" for i in range(extra_dsps_bl))
        parts.append(f"    assign data_out = {prev_data} ^ {{{DATA_WIDTH-1}'b0, {dsp_xor}}};")
    else:
        # NL-DPE: add bare DPE instances for DIMM DPEs to ensure correct
        # resource count. Parmys merges the conv_layer_single_dpe instances
        # inside DIMM modules (identical structure), so we add standalone
        # DPE hard blocks with unique data_in XOR to prevent optimization.
        if is_nldpe and total_dimm_dpes > 0:
            parts.append(f"")
            parts.append(f"    // ═══ DIMM DPE instances (bare hard blocks) ═══")
            parts.append(f"    // {total_dimm_dpes} DPEs for DIMM stages across all blocks/heads")
            parts.append(f"    // Each has unique data_in XOR to prevent parmys optimization")
            xor_bits = []
            for i in range(total_dimm_dpes):
                parts.append(f"    wire [{DATA_WIDTH}-1:0] _dimm_dpe_out_{i};")
                parts.append(f"    wire _dimm_msbsa_{i};")
                parts.append(f"    dpe _dimm_dpe_{i} (")
                parts.append(f"        .clk(clk), .reset(rst),")
                parts.append(f"        .data_in(data_in ^ {DATA_WIDTH}'d{i + 1}),")
                parts.append(f"        .nl_dpe_control(2'b00),")
                parts.append(f"        .w_buf_en(1'b0),")
                parts.append(f"        .shift_add_control(1'b0),")
                parts.append(f"        .shift_add_bypass(1'b0),")
                parts.append(f"        .load_output_reg(1'b0),")
                parts.append(f"        .load_input_reg(1'b0),")
                parts.append(f"        .MSB_SA_Ready(_dimm_msbsa_{i}),")
                parts.append(f"        .data_out(_dimm_dpe_out_{i}),")
                parts.append(f"        .dpe_done(), .reg_full(),")
                parts.append(f"        .shift_add_done(), .shift_add_bypass_ctrl()")
                parts.append(f"    );")
                xor_bits.append(f"_dimm_dpe_out_{i}[0]")
            # XOR one bit from each DPE + top-level DIMM buffers into data_out
            # Include DIMM buffer outputs to prevent dead-code elimination
            dimm_buf_xor = []
            for b_idx in range(NUM_BLOCKS):
                for h_idx in range(NUM_HEADS):
                    for bn in ["k", "v", "sc", "sm_i", "sm_e", "sm_o", "at", "la"]:
                        dimm_buf_xor.append(f"b{b_idx}_h{h_idx}_dimm_{bn}_out[0]")
            xor_bits.extend(dimm_buf_xor)
            xor_expr = " ^ ".join(xor_bits)
            parts.append(f"    assign data_out = {prev_data} ^ {{{DATA_WIDTH-1}'b0, {xor_expr}}};")
        else:
            parts.append(f"    assign data_out = {prev_data};")

    parts.append(f"endmodule")
    parts.append(f"")

    # ─── Sub-modules ──────────────────────────────────────────

    # FFN1 multi-DPE modules (if needed)
    if not (V_ffn1 == 1 and H_ffn1 == 1):
        for b in range(NUM_BLOCKS):
            mod_name = f"ffn1_layer_b{b}"
            K_ffn1 = D_MODEL
            depth = max(512, K_ffn1)
            addr_width = max(1, math.ceil(math.log2(depth)))
            parts.append(f"// FFN1 block {b}: V={V_ffn1} H={H_ffn1}")
            parts.append(_gen_fc_layer(V_ffn1, H_ffn1, K_ffn1, D_FF, rows, cols,
                                       depth, addr_width, DATA_WIDTH, module_name=mod_name))
            parts.append(f"")

    # DIMM modules (NL-DPE only) — per-block, per-head, per-lane unique names
    # Each lane uses a slightly different SRAM depth to prevent parmys from
    # merging structurally identical modules. This ensures VTR places all
    # DPE instances separately, giving realistic Fmax under full utilization.
    if is_nldpe:
        # Per-SRAM depths (packed int8 into 40-bit words)
        dimm_depth_q     = _packed_depth(D_HEAD)           # one query vector
        dimm_depth_k     = _packed_depth(seq_len * D_HEAD) # all key vectors
        dimm_depth_score = _packed_depth(seq_len)          # one score row
        dimm_depth_row   = _packed_depth(seq_len)          # softmax/attn row
        dimm_depth_v     = _packed_depth(seq_len * D_HEAD) # all V vectors
        dimm_depth_d     = _packed_depth(D_HEAD)           # one output row
        use_dual = (2 * D_HEAD <= cols)  # K-identity: K = cols // D_HEAD
        lane_idx = 0
        for b in range(NUM_BLOCKS):
            for h in range(NUM_HEADS):
                for w in range(dimm_width):
                    lane_mod = f"_w{w}" if dimm_width > 1 else ""
                    mod_suffix = f"_b{b}_h{h}{lane_mod}"
                    lane_idx += 1
                    # Anti-merge: offset SRAM depths by uid×513 so each instance
                    # crosses a different BRAM block boundary (block=512 entries).
                    # This forces parmys to create different numbers of dual_port_ram
                    # primitives per instance, preventing structural deduplication.
                    # XOR on data_out also prevents module-level dedup.
                    uid = lane_idx  # 1, 2, 3, 4 for 2 heads × 2 blocks
                    off = uid * 513  # cross block boundaries: +513, +1026, +1539, +2052
                    dk = dimm_depth_k + off
                    dv = dimm_depth_v + off
                    ds = dimm_depth_score + off
                    dr = dimm_depth_row + off
                    dq = dimm_depth_q + off
                    dd = dimm_depth_d + off
                    parts.append(f"// DIMM — Block {b}, Head {h}, Lane {w} (uid={uid}, off={off})")
                    parts.append(f"//   Q={dq}, K={dk}, score={ds}, row={dr}, V={dv}, out={dd}")

                    score_v = _gen_dimm_score_matrix(seq_len, D_HEAD, h_dimm,
                                                      dq, dk, ds,
                                                      DATA_WIDTH, dual_identity=use_dual, uid=uid)
                    score_v = score_v.replace("module dimm_score_matrix",
                                             f"module dimm_score_matrix{mod_suffix}")
                    parts.append(score_v)

                    softmax_v = _gen_softmax_dpe(seq_len, D_HEAD, h_dimm,
                                                  dr, dr + 1, dr + 2,
                                                  DATA_WIDTH, uid=uid)
                    softmax_v = softmax_v.replace("module softmax_approx",
                                                  f"module softmax_approx{mod_suffix}")
                    parts.append(softmax_v)

                    wsum_v = _gen_dimm_weighted_sum(seq_len, D_HEAD, h_dimm,
                                                     dr + 3, dv, dr + 4, dd,
                                                     DATA_WIDTH, uid=uid)
                    wsum_v = wsum_v.replace("module dimm_weighted_sum",
                                           f"module dimm_weighted_sum{mod_suffix}")
                    parts.append(wsum_v)
                    parts.append(f"")

    # Non-DPE modules (baseline and Azure-Lily)
    if not is_nldpe:
        # Both baseline and Azure-Lily use DSP MACs (DSPs now available)
        parts.append(_gen_dsp_mac_module(DATA_WIDTH))
        parts.append(f"")
        parts.append(_gen_clb_softmax_module(DATA_WIDTH))
        parts.append(f"")

    # CLB modules (both architectures)
    # Strip inline sram definitions — _get_supporting_modules() provides it
    def _strip_inline_sram(verilog_str):
        """Remove embedded 'module sram ... endmodule' from CLB module output."""
        lines = verilog_str.split('\n')
        result = []
        skip = False
        for line in lines:
            if line.strip().startswith('module sram'):
                skip = True
            if not skip:
                result.append(line)
            if skip and line.strip() == 'endmodule':
                skip = False
        return '\n'.join(result)

    parts.append(f"// ═══════════════════════════════════════")
    parts.append(f"// CLB Modules (DPE-independent)")
    parts.append(f"// ═══════════════════════════════════════")
    parts.append(_strip_inline_sram(gen_residual_add(depth=D_MODEL, data_width=DATA_WIDTH)))
    parts.append(f"")
    parts.append(_strip_inline_sram(gen_embedding_add(depth=D_MODEL, data_width=DATA_WIDTH)))
    parts.append(f"")
    parts.append(_strip_inline_sram(gen_layernorm(d_model=D_MODEL, data_width=DATA_WIDTH)))
    parts.append(f"")

    # Note: LayerNorm uses 'multiply' primitive (VTR built-in in primitives.v).
    # Maps to .subckt multiply in BLIF → DSP hard block.
    # No need to define it here — VTR provides it.

    # Activation LUT (needed for Azure-Lily and baseline GELU)
    if not is_nldpe:
        parts.append(_gen_activation_lut_module())
        parts.append(f"")

    # Supporting modules (DPE primitives only for DPE-enabled architectures)
    if not is_baseline:
        parts.append(f"// ═══════════════════════════════════════")
        parts.append(f"// Supporting modules (DPE primitives)")
        parts.append(f"// ═══════════════════════════════════════")
        parts.append(_get_supporting_modules())
    else:
        # Baseline only needs sram and global_controller (no DPE)
        parts.append(f"// ═══════════════════════════════════════")
        parts.append(f"// Supporting modules (baseline, no DPE)")
        parts.append(f"// ═══════════════════════════════════════")
        # Extract just sram and global_controller from supporting modules
        all_mods = _get_supporting_modules()
        for mod_name in ["sram", "global_controller"]:
            start = all_mods.find(f"module {mod_name}")
            if start >= 0:
                end = all_mods.find("endmodule", start) + len("endmodule")
                parts.append(all_mods[start:end])
                parts.append("")

    # Write
    out_path = out_dir / filename
    out_path.write_text("\n".join(parts))

    print(f"  Generated {filename}")
    print(f"    Arch: {arch_str}, Crossbar: {rows}×{cols}")
    print(f"    Total DPEs: {total_dpes}")
    print(f"    Proj DPEs/block: {proj_dpes_per_block} (Q/K/V/O)")
    print(f"    FFN DPEs/block: {ffn_dpes_per_block} (FFN1={V_ffn1*H_ffn1} + FFN2={V_ffn2*H_ffn2})")
    if is_nldpe:
        print(f"    DIMM DPEs/block: {dimm_dpes_per_head*NUM_HEADS} ({dimm_dpes_per_head}/head × {NUM_HEADS} heads)")
    else:
        print(f"    DIMM: DSP MAC (0 DPEs)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate BERT-Tiny RTL")
    parser.add_argument("--arch", required=True, choices=["nl_dpe", "azure_lily", "baseline"])
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("-o", "--output-dir", default="benchmarks/rtl/")
    parser.add_argument("--label", default=None)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--dimm-width", type=int, default=1,
                        help="W parallel DIMM lanes per head (NL-DPE only)")
    args = parser.parse_args()
    gen_bert_tiny(args.arch, args.rows, args.cols, args.output_dir,
                  args.label, args.seq_len, args.dimm_width)


if __name__ == "__main__":
    main()
