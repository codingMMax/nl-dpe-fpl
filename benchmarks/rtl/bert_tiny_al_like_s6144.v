// Auto-generated BERT-Tiny RTL — NL-DPE 1024×256
// Total DPEs: 398
// Q/K/V/O: V=1 H=1 → 1 DPE each
// FFN1: V=1 H=2 → 2 DPEs
// FFN2: V=1 H=1 → 1 DPEs
// DIMM: W=1 lanes, K_qkt=4, K_sv=1
// DIMM: 96 DPEs/head × 2 heads × 2 blocks = 384

module bert_tiny_al_like_s6144 (
    input wire clk, rst, valid, ready_n,
    input wire [40-1:0] data_in,
    output wire [40-1:0] data_out,
    output wire ready, valid_n
);

    wire [40-1:0] data_embed;
    wire valid_embed, ready_embed;
    wire [40-1:0] data_ln_embed;
    wire valid_ln_embed, ready_ln_embed;
    wire [40-1:0] data_b0_q;
    wire valid_b0_q, ready_b0_q;
    wire [40-1:0] data_b0_k;
    wire valid_b0_k, ready_b0_k;
    wire [40-1:0] data_b0_v;
    wire valid_b0_v, ready_b0_v;
    wire [40-1:0] data_b0_h0_score;
    wire valid_b0_h0_score, ready_b0_h0_score;
    wire [40-1:0] data_b0_h0_softmax;
    wire valid_b0_h0_softmax, ready_b0_h0_softmax;
    wire [40-1:0] data_b0_h0_wsum;
    wire valid_b0_h0_wsum, ready_b0_h0_wsum;
    wire [40-1:0] data_b0_h1_score;
    wire valid_b0_h1_score, ready_b0_h1_score;
    wire [40-1:0] data_b0_h1_softmax;
    wire valid_b0_h1_softmax, ready_b0_h1_softmax;
    wire [40-1:0] data_b0_h1_wsum;
    wire valid_b0_h1_wsum, ready_b0_h1_wsum;
    wire [40-1:0] data_b0_o;
    wire valid_b0_o, ready_b0_o;
    wire [40-1:0] data_b0_res_attn;
    wire valid_b0_res_attn, ready_b0_res_attn;
    wire [40-1:0] data_b0_ln_attn;
    wire valid_b0_ln_attn, ready_b0_ln_attn;
    wire [40-1:0] data_b0_ffn1;
    wire valid_b0_ffn1, ready_b0_ffn1;
    wire [40-1:0] data_b0_ffn2;
    wire valid_b0_ffn2, ready_b0_ffn2;
    wire [40-1:0] data_b0_res_ffn;
    wire valid_b0_res_ffn, ready_b0_res_ffn;
    wire [40-1:0] data_b0_ln_ffn;
    wire valid_b0_ln_ffn, ready_b0_ln_ffn;
    wire [40-1:0] data_b1_q;
    wire valid_b1_q, ready_b1_q;
    wire [40-1:0] data_b1_k;
    wire valid_b1_k, ready_b1_k;
    wire [40-1:0] data_b1_v;
    wire valid_b1_v, ready_b1_v;
    wire [40-1:0] data_b1_h0_score;
    wire valid_b1_h0_score, ready_b1_h0_score;
    wire [40-1:0] data_b1_h0_softmax;
    wire valid_b1_h0_softmax, ready_b1_h0_softmax;
    wire [40-1:0] data_b1_h0_wsum;
    wire valid_b1_h0_wsum, ready_b1_h0_wsum;
    wire [40-1:0] data_b1_h1_score;
    wire valid_b1_h1_score, ready_b1_h1_score;
    wire [40-1:0] data_b1_h1_softmax;
    wire valid_b1_h1_softmax, ready_b1_h1_softmax;
    wire [40-1:0] data_b1_h1_wsum;
    wire valid_b1_h1_wsum, ready_b1_h1_wsum;
    wire [40-1:0] data_b1_o;
    wire valid_b1_o, ready_b1_o;
    wire [40-1:0] data_b1_res_attn;
    wire valid_b1_res_attn, ready_b1_res_attn;
    wire [40-1:0] data_b1_ln_attn;
    wire valid_b1_ln_attn, ready_b1_ln_attn;
    wire [40-1:0] data_b1_ffn1;
    wire valid_b1_ffn1, ready_b1_ffn1;
    wire [40-1:0] data_b1_ffn2;
    wire valid_b1_ffn2, ready_b1_ffn2;
    wire [40-1:0] data_b1_res_ffn;
    wire valid_b1_res_ffn, ready_b1_res_ffn;
    wire [40-1:0] data_b1_ln_ffn;
    wire valid_b1_ln_ffn, ready_b1_ln_ffn;
    wire valid_g_out, ready_g_in;

    // === Embedding: token + position + segment add ===
    embedding_add #(.DATA_WIDTH(40), .DEPTH(128)) embed_inst (
        .clk(clk), .rst(rst), .valid(valid_g_out), .ready_n(ready_embed),
        .data_in(data_in), .data_out(data_embed), .ready(ready_g_in), .valid_n(valid_embed)
    );

    // === LayerNorm (embedding) ===
    layernorm #(.DATA_WIDTH(40), .D_MODEL(128)) ln_embed_inst (
        .clk(clk), .rst(rst), .valid(valid_embed), .ready_n(ready_ln_embed),
        .data_in(data_embed), .data_out(data_ln_embed), .ready(ready_embed), .valid_n(valid_ln_embed)
    );

    // ═══════════════════════════════════════
    // Transformer Block 0
    // ═══════════════════════════════════════
    // b0_q: projection V=1 H=1 (1 DPE, ACAM)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b0_q_inst (
        .clk(clk), .rst(rst),
        .valid(valid_ln_embed), .ready_n(ready_b0_q),
        .data_in(data_ln_embed),
        .data_out(data_b0_q), .ready(ready_ln_embed), .valid_n(valid_b0_q)
    );

    // b0_k: projection V=1 H=1 (1 DPE, ACAM)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b0_k_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_q), .ready_n(ready_b0_k),
        .data_in(data_b0_q),
        .data_out(data_b0_k), .ready(ready_b0_j), .valid_n(valid_b0_k)
    );

    // b0_v: projection V=1 H=1 (1 DPE, ACAM)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b0_v_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_k), .ready_n(ready_b0_v),
        .data_in(data_b0_k),
        .data_out(data_b0_v), .ready(ready_b0_u), .valid_n(valid_b0_v)
    );

    wire ready_b0_h0_score_q, ready_b0_h0_score_k;
    wire ready_b0_h0_wsum_v;
    wire ready_b0_h1_score_q, ready_b0_h1_score_k;
    wire ready_b0_h1_wsum_v;
    assign ready_b0_q = ready_b0_h0_score_q & ready_b0_h1_score_q;
    assign ready_b0_k = ready_b0_h0_score_k & ready_b0_h1_score_k;
    assign ready_b0_v = ready_b0_h0_wsum_v & ready_b0_h1_wsum_v;

    // Head 0: W=1 DIMM lanes, K_qkt=4, K_sv=1
    dimm_score_matrix_b0_h0 #(.DATA_WIDTH(40)) b0_h0_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h0_score),
        .ready_q(ready_b0_h0_score_q), .ready_k(ready_b0_h0_score_k),
        .valid_n(valid_b0_h0_score)
    );
    softmax_approx_b0_h0 #(.DATA_WIDTH(40)) b0_h0_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h0_score), .ready_n(1'b1),
        .data_in(data_b0_h0_score),
        .data_out(data_b0_h0_softmax), .ready(), .valid_n(valid_b0_h0_softmax)
    );
    dimm_weighted_sum_b0_h0 #(.DATA_WIDTH(40)) b0_h0_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h0_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h0_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h0_wsum),
        .ready_attn(ready_b0_h0_softmax), .ready_v(ready_b0_h0_wsum_v),
        .valid_n(valid_b0_h0_wsum)
    );

    // Head 1: W=1 DIMM lanes, K_qkt=4, K_sv=1
    dimm_score_matrix_b0_h1 #(.DATA_WIDTH(40)) b0_h1_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h1_score),
        .ready_q(ready_b0_h1_score_q), .ready_k(ready_b0_h1_score_k),
        .valid_n(valid_b0_h1_score)
    );
    softmax_approx_b0_h1 #(.DATA_WIDTH(40)) b0_h1_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h1_score), .ready_n(1'b1),
        .data_in(data_b0_h1_score),
        .data_out(data_b0_h1_softmax), .ready(), .valid_n(valid_b0_h1_softmax)
    );
    dimm_weighted_sum_b0_h1 #(.DATA_WIDTH(40)) b0_h1_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h1_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h1_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h1_wsum),
        .ready_attn(ready_b0_h1_softmax), .ready_v(ready_b0_h1_wsum_v),
        .valid_n(valid_b0_h1_wsum)
    );

    // O projection V=1 H=1 (1 DPE)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b0_o_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h1_wsum), .ready_n(ready_b0_o),
        .data_in(data_b0_h1_wsum),
        .data_out(data_b0_o), .ready(ready_b0_h1_wsum), .valid_n(valid_b0_o)
    );

    // Residual add (attention)
    residual_add #(.DATA_WIDTH(40), .DEPTH(128)) b0_res_attn_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_o), .ready_n(ready_b0_res_attn),
        .data_in(data_b0_o), .data_out(data_b0_res_attn),
        .ready(ready_b0_o), .valid_n(valid_b0_res_attn)
    );

    // LayerNorm (post-attention)
    layernorm #(.DATA_WIDTH(40), .D_MODEL(128)) b0_ln_attn_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_res_attn), .ready_n(ready_b0_ln_attn),
        .data_in(data_b0_res_attn), .data_out(data_b0_ln_attn),
        .ready(ready_b0_res_attn), .valid_n(valid_b0_ln_attn)
    );

    // FFN1: V=1 H=2 (2 DPEs)
    ffn1_layer_b0 #(.DATA_WIDTH(40)) b0_ffn1_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_ln_attn), .ready_n(ready_b0_ffn1),
        .data_in(data_b0_ln_attn),
        .data_out(data_b0_ffn1), .ready(ready_b0_ln_attn), .valid_n(valid_b0_ffn1)
    );

    // FFN2: V=1 H=1 (1 DPE)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(512), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b0_ffn2_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_ffn1), .ready_n(ready_b0_ffn2),
        .data_in(data_b0_ffn1),
        .data_out(data_b0_ffn2), .ready(ready_b0_ffn1), .valid_n(valid_b0_ffn2)
    );

    // Residual add (FFN)
    residual_add #(.DATA_WIDTH(40), .DEPTH(128)) b0_res_ffn_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_ffn2), .ready_n(ready_b0_res_ffn),
        .data_in(data_b0_ffn2), .data_out(data_b0_res_ffn),
        .ready(ready_b0_ffn2), .valid_n(valid_b0_res_ffn)
    );

    // LayerNorm (post-FFN)
    layernorm #(.DATA_WIDTH(40), .D_MODEL(128)) b0_ln_ffn_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_res_ffn), .ready_n(ready_b0_ln_ffn),
        .data_in(data_b0_res_ffn), .data_out(data_b0_ln_ffn),
        .ready(ready_b0_res_ffn), .valid_n(valid_b0_ln_ffn)
    );

    // ═══════════════════════════════════════
    // Transformer Block 1
    // ═══════════════════════════════════════
    // b1_q: projection V=1 H=1 (1 DPE, ACAM)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b1_q_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_ln_ffn), .ready_n(ready_b1_q),
        .data_in(data_b0_ln_ffn),
        .data_out(data_b1_q), .ready(ready_ln_ffn), .valid_n(valid_b1_q)
    );

    // b1_k: projection V=1 H=1 (1 DPE, ACAM)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b1_k_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_q), .ready_n(ready_b1_k),
        .data_in(data_b1_q),
        .data_out(data_b1_k), .ready(ready_b1_j), .valid_n(valid_b1_k)
    );

    // b1_v: projection V=1 H=1 (1 DPE, ACAM)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b1_v_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_k), .ready_n(ready_b1_v),
        .data_in(data_b1_k),
        .data_out(data_b1_v), .ready(ready_b1_u), .valid_n(valid_b1_v)
    );

    wire ready_b1_h0_score_q, ready_b1_h0_score_k;
    wire ready_b1_h0_wsum_v;
    wire ready_b1_h1_score_q, ready_b1_h1_score_k;
    wire ready_b1_h1_wsum_v;
    assign ready_b1_q = ready_b1_h0_score_q & ready_b1_h1_score_q;
    assign ready_b1_k = ready_b1_h0_score_k & ready_b1_h1_score_k;
    assign ready_b1_v = ready_b1_h0_wsum_v & ready_b1_h1_wsum_v;

    // Head 0: W=1 DIMM lanes, K_qkt=4, K_sv=1
    dimm_score_matrix_b1_h0 #(.DATA_WIDTH(40)) b1_h0_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h0_score),
        .ready_q(ready_b1_h0_score_q), .ready_k(ready_b1_h0_score_k),
        .valid_n(valid_b1_h0_score)
    );
    softmax_approx_b1_h0 #(.DATA_WIDTH(40)) b1_h0_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h0_score), .ready_n(1'b1),
        .data_in(data_b1_h0_score),
        .data_out(data_b1_h0_softmax), .ready(), .valid_n(valid_b1_h0_softmax)
    );
    dimm_weighted_sum_b1_h0 #(.DATA_WIDTH(40)) b1_h0_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h0_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h0_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h0_wsum),
        .ready_attn(ready_b1_h0_softmax), .ready_v(ready_b1_h0_wsum_v),
        .valid_n(valid_b1_h0_wsum)
    );

    // Head 1: W=1 DIMM lanes, K_qkt=4, K_sv=1
    dimm_score_matrix_b1_h1 #(.DATA_WIDTH(40)) b1_h1_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h1_score),
        .ready_q(ready_b1_h1_score_q), .ready_k(ready_b1_h1_score_k),
        .valid_n(valid_b1_h1_score)
    );
    softmax_approx_b1_h1 #(.DATA_WIDTH(40)) b1_h1_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h1_score), .ready_n(1'b1),
        .data_in(data_b1_h1_score),
        .data_out(data_b1_h1_softmax), .ready(), .valid_n(valid_b1_h1_softmax)
    );
    dimm_weighted_sum_b1_h1 #(.DATA_WIDTH(40)) b1_h1_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h1_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h1_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h1_wsum),
        .ready_attn(ready_b1_h1_softmax), .ready_v(ready_b1_h1_wsum_v),
        .valid_n(valid_b1_h1_wsum)
    );

    // O projection V=1 H=1 (1 DPE)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b1_o_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h1_wsum), .ready_n(ready_b1_o),
        .data_in(data_b1_h1_wsum),
        .data_out(data_b1_o), .ready(ready_b1_h1_wsum), .valid_n(valid_b1_o)
    );

    // Residual add (attention)
    residual_add #(.DATA_WIDTH(40), .DEPTH(128)) b1_res_attn_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_o), .ready_n(ready_b1_res_attn),
        .data_in(data_b1_o), .data_out(data_b1_res_attn),
        .ready(ready_b1_o), .valid_n(valid_b1_res_attn)
    );

    // LayerNorm (post-attention)
    layernorm #(.DATA_WIDTH(40), .D_MODEL(128)) b1_ln_attn_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_res_attn), .ready_n(ready_b1_ln_attn),
        .data_in(data_b1_res_attn), .data_out(data_b1_ln_attn),
        .ready(ready_b1_res_attn), .valid_n(valid_b1_ln_attn)
    );

    // FFN1: V=1 H=2 (2 DPEs)
    ffn1_layer_b1 #(.DATA_WIDTH(40)) b1_ffn1_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_ln_attn), .ready_n(ready_b1_ffn1),
        .data_in(data_b1_ln_attn),
        .data_out(data_b1_ffn1), .ready(ready_b1_ln_attn), .valid_n(valid_b1_ffn1)
    );

    // FFN2: V=1 H=1 (1 DPE)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(512), .KERNEL_HEIGHT(1),
        .W(6144), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b1_ffn2_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_ffn1), .ready_n(ready_b1_ffn2),
        .data_in(data_b1_ffn1),
        .data_out(data_b1_ffn2), .ready(ready_b1_ffn1), .valid_n(valid_b1_ffn2)
    );

    // Residual add (FFN)
    residual_add #(.DATA_WIDTH(40), .DEPTH(128)) b1_res_ffn_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_ffn2), .ready_n(ready_b1_res_ffn),
        .data_in(data_b1_ffn2), .data_out(data_b1_res_ffn),
        .ready(ready_b1_ffn2), .valid_n(valid_b1_res_ffn)
    );

    // LayerNorm (post-FFN)
    layernorm #(.DATA_WIDTH(40), .D_MODEL(128)) b1_ln_ffn_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_res_ffn), .ready_n(ready_b1_ln_ffn),
        .data_in(data_b1_res_ffn), .data_out(data_b1_ln_ffn),
        .ready(ready_b1_res_ffn), .valid_n(valid_b1_ln_ffn)
    );

    global_controller #(.N_Layers(1)) g_ctrl (
        .clk(clk), .rst(rst),
        .ready_L1(ready_g_in), .valid_Ln(valid_b1_ln_ffn),
        .valid(valid), .ready(ready),
        .valid_L1(valid_g_out), .ready_Ln(ready_n)
    );
    assign valid_n = valid_b1_ln_ffn;

    // ═══ DIMM DPE instances (bare hard blocks) ═══
    // 384 DPEs for DIMM stages across all blocks/heads
    // Each has unique data_in XOR to prevent parmys optimization
    wire [40-1:0] _dimm_dpe_out_0;
    wire _dimm_msbsa_0;
    dpe _dimm_dpe_0 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd1),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_0),
        .data_out(_dimm_dpe_out_0),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_1;
    wire _dimm_msbsa_1;
    dpe _dimm_dpe_1 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd2),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_1),
        .data_out(_dimm_dpe_out_1),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_2;
    wire _dimm_msbsa_2;
    dpe _dimm_dpe_2 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd3),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_2),
        .data_out(_dimm_dpe_out_2),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_3;
    wire _dimm_msbsa_3;
    dpe _dimm_dpe_3 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd4),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_3),
        .data_out(_dimm_dpe_out_3),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_4;
    wire _dimm_msbsa_4;
    dpe _dimm_dpe_4 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd5),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_4),
        .data_out(_dimm_dpe_out_4),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_5;
    wire _dimm_msbsa_5;
    dpe _dimm_dpe_5 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd6),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_5),
        .data_out(_dimm_dpe_out_5),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_6;
    wire _dimm_msbsa_6;
    dpe _dimm_dpe_6 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd7),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_6),
        .data_out(_dimm_dpe_out_6),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_7;
    wire _dimm_msbsa_7;
    dpe _dimm_dpe_7 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd8),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_7),
        .data_out(_dimm_dpe_out_7),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_8;
    wire _dimm_msbsa_8;
    dpe _dimm_dpe_8 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd9),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_8),
        .data_out(_dimm_dpe_out_8),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_9;
    wire _dimm_msbsa_9;
    dpe _dimm_dpe_9 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd10),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_9),
        .data_out(_dimm_dpe_out_9),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_10;
    wire _dimm_msbsa_10;
    dpe _dimm_dpe_10 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd11),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_10),
        .data_out(_dimm_dpe_out_10),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_11;
    wire _dimm_msbsa_11;
    dpe _dimm_dpe_11 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd12),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_11),
        .data_out(_dimm_dpe_out_11),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_12;
    wire _dimm_msbsa_12;
    dpe _dimm_dpe_12 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd13),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_12),
        .data_out(_dimm_dpe_out_12),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_13;
    wire _dimm_msbsa_13;
    dpe _dimm_dpe_13 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd14),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_13),
        .data_out(_dimm_dpe_out_13),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_14;
    wire _dimm_msbsa_14;
    dpe _dimm_dpe_14 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd15),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_14),
        .data_out(_dimm_dpe_out_14),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_15;
    wire _dimm_msbsa_15;
    dpe _dimm_dpe_15 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd16),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_15),
        .data_out(_dimm_dpe_out_15),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_16;
    wire _dimm_msbsa_16;
    dpe _dimm_dpe_16 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd17),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_16),
        .data_out(_dimm_dpe_out_16),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_17;
    wire _dimm_msbsa_17;
    dpe _dimm_dpe_17 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd18),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_17),
        .data_out(_dimm_dpe_out_17),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_18;
    wire _dimm_msbsa_18;
    dpe _dimm_dpe_18 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd19),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_18),
        .data_out(_dimm_dpe_out_18),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_19;
    wire _dimm_msbsa_19;
    dpe _dimm_dpe_19 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd20),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_19),
        .data_out(_dimm_dpe_out_19),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_20;
    wire _dimm_msbsa_20;
    dpe _dimm_dpe_20 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd21),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_20),
        .data_out(_dimm_dpe_out_20),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_21;
    wire _dimm_msbsa_21;
    dpe _dimm_dpe_21 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd22),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_21),
        .data_out(_dimm_dpe_out_21),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_22;
    wire _dimm_msbsa_22;
    dpe _dimm_dpe_22 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd23),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_22),
        .data_out(_dimm_dpe_out_22),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_23;
    wire _dimm_msbsa_23;
    dpe _dimm_dpe_23 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd24),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_23),
        .data_out(_dimm_dpe_out_23),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_24;
    wire _dimm_msbsa_24;
    dpe _dimm_dpe_24 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd25),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_24),
        .data_out(_dimm_dpe_out_24),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_25;
    wire _dimm_msbsa_25;
    dpe _dimm_dpe_25 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd26),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_25),
        .data_out(_dimm_dpe_out_25),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_26;
    wire _dimm_msbsa_26;
    dpe _dimm_dpe_26 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd27),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_26),
        .data_out(_dimm_dpe_out_26),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_27;
    wire _dimm_msbsa_27;
    dpe _dimm_dpe_27 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd28),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_27),
        .data_out(_dimm_dpe_out_27),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_28;
    wire _dimm_msbsa_28;
    dpe _dimm_dpe_28 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd29),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_28),
        .data_out(_dimm_dpe_out_28),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_29;
    wire _dimm_msbsa_29;
    dpe _dimm_dpe_29 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd30),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_29),
        .data_out(_dimm_dpe_out_29),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_30;
    wire _dimm_msbsa_30;
    dpe _dimm_dpe_30 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd31),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_30),
        .data_out(_dimm_dpe_out_30),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_31;
    wire _dimm_msbsa_31;
    dpe _dimm_dpe_31 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd32),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_31),
        .data_out(_dimm_dpe_out_31),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_32;
    wire _dimm_msbsa_32;
    dpe _dimm_dpe_32 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd33),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_32),
        .data_out(_dimm_dpe_out_32),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_33;
    wire _dimm_msbsa_33;
    dpe _dimm_dpe_33 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd34),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_33),
        .data_out(_dimm_dpe_out_33),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_34;
    wire _dimm_msbsa_34;
    dpe _dimm_dpe_34 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd35),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_34),
        .data_out(_dimm_dpe_out_34),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_35;
    wire _dimm_msbsa_35;
    dpe _dimm_dpe_35 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd36),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_35),
        .data_out(_dimm_dpe_out_35),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_36;
    wire _dimm_msbsa_36;
    dpe _dimm_dpe_36 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd37),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_36),
        .data_out(_dimm_dpe_out_36),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_37;
    wire _dimm_msbsa_37;
    dpe _dimm_dpe_37 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd38),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_37),
        .data_out(_dimm_dpe_out_37),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_38;
    wire _dimm_msbsa_38;
    dpe _dimm_dpe_38 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd39),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_38),
        .data_out(_dimm_dpe_out_38),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_39;
    wire _dimm_msbsa_39;
    dpe _dimm_dpe_39 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd40),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_39),
        .data_out(_dimm_dpe_out_39),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_40;
    wire _dimm_msbsa_40;
    dpe _dimm_dpe_40 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd41),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_40),
        .data_out(_dimm_dpe_out_40),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_41;
    wire _dimm_msbsa_41;
    dpe _dimm_dpe_41 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd42),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_41),
        .data_out(_dimm_dpe_out_41),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_42;
    wire _dimm_msbsa_42;
    dpe _dimm_dpe_42 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd43),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_42),
        .data_out(_dimm_dpe_out_42),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_43;
    wire _dimm_msbsa_43;
    dpe _dimm_dpe_43 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd44),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_43),
        .data_out(_dimm_dpe_out_43),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_44;
    wire _dimm_msbsa_44;
    dpe _dimm_dpe_44 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd45),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_44),
        .data_out(_dimm_dpe_out_44),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_45;
    wire _dimm_msbsa_45;
    dpe _dimm_dpe_45 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd46),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_45),
        .data_out(_dimm_dpe_out_45),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_46;
    wire _dimm_msbsa_46;
    dpe _dimm_dpe_46 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd47),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_46),
        .data_out(_dimm_dpe_out_46),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_47;
    wire _dimm_msbsa_47;
    dpe _dimm_dpe_47 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd48),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_47),
        .data_out(_dimm_dpe_out_47),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_48;
    wire _dimm_msbsa_48;
    dpe _dimm_dpe_48 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd49),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_48),
        .data_out(_dimm_dpe_out_48),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_49;
    wire _dimm_msbsa_49;
    dpe _dimm_dpe_49 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd50),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_49),
        .data_out(_dimm_dpe_out_49),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_50;
    wire _dimm_msbsa_50;
    dpe _dimm_dpe_50 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd51),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_50),
        .data_out(_dimm_dpe_out_50),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_51;
    wire _dimm_msbsa_51;
    dpe _dimm_dpe_51 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd52),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_51),
        .data_out(_dimm_dpe_out_51),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_52;
    wire _dimm_msbsa_52;
    dpe _dimm_dpe_52 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd53),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_52),
        .data_out(_dimm_dpe_out_52),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_53;
    wire _dimm_msbsa_53;
    dpe _dimm_dpe_53 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd54),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_53),
        .data_out(_dimm_dpe_out_53),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_54;
    wire _dimm_msbsa_54;
    dpe _dimm_dpe_54 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd55),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_54),
        .data_out(_dimm_dpe_out_54),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_55;
    wire _dimm_msbsa_55;
    dpe _dimm_dpe_55 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd56),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_55),
        .data_out(_dimm_dpe_out_55),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_56;
    wire _dimm_msbsa_56;
    dpe _dimm_dpe_56 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd57),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_56),
        .data_out(_dimm_dpe_out_56),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_57;
    wire _dimm_msbsa_57;
    dpe _dimm_dpe_57 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd58),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_57),
        .data_out(_dimm_dpe_out_57),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_58;
    wire _dimm_msbsa_58;
    dpe _dimm_dpe_58 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd59),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_58),
        .data_out(_dimm_dpe_out_58),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_59;
    wire _dimm_msbsa_59;
    dpe _dimm_dpe_59 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd60),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_59),
        .data_out(_dimm_dpe_out_59),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_60;
    wire _dimm_msbsa_60;
    dpe _dimm_dpe_60 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd61),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_60),
        .data_out(_dimm_dpe_out_60),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_61;
    wire _dimm_msbsa_61;
    dpe _dimm_dpe_61 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd62),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_61),
        .data_out(_dimm_dpe_out_61),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_62;
    wire _dimm_msbsa_62;
    dpe _dimm_dpe_62 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd63),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_62),
        .data_out(_dimm_dpe_out_62),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_63;
    wire _dimm_msbsa_63;
    dpe _dimm_dpe_63 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd64),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_63),
        .data_out(_dimm_dpe_out_63),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_64;
    wire _dimm_msbsa_64;
    dpe _dimm_dpe_64 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd65),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_64),
        .data_out(_dimm_dpe_out_64),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_65;
    wire _dimm_msbsa_65;
    dpe _dimm_dpe_65 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd66),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_65),
        .data_out(_dimm_dpe_out_65),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_66;
    wire _dimm_msbsa_66;
    dpe _dimm_dpe_66 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd67),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_66),
        .data_out(_dimm_dpe_out_66),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_67;
    wire _dimm_msbsa_67;
    dpe _dimm_dpe_67 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd68),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_67),
        .data_out(_dimm_dpe_out_67),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_68;
    wire _dimm_msbsa_68;
    dpe _dimm_dpe_68 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd69),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_68),
        .data_out(_dimm_dpe_out_68),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_69;
    wire _dimm_msbsa_69;
    dpe _dimm_dpe_69 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd70),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_69),
        .data_out(_dimm_dpe_out_69),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_70;
    wire _dimm_msbsa_70;
    dpe _dimm_dpe_70 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd71),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_70),
        .data_out(_dimm_dpe_out_70),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_71;
    wire _dimm_msbsa_71;
    dpe _dimm_dpe_71 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd72),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_71),
        .data_out(_dimm_dpe_out_71),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_72;
    wire _dimm_msbsa_72;
    dpe _dimm_dpe_72 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd73),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_72),
        .data_out(_dimm_dpe_out_72),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_73;
    wire _dimm_msbsa_73;
    dpe _dimm_dpe_73 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd74),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_73),
        .data_out(_dimm_dpe_out_73),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_74;
    wire _dimm_msbsa_74;
    dpe _dimm_dpe_74 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd75),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_74),
        .data_out(_dimm_dpe_out_74),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_75;
    wire _dimm_msbsa_75;
    dpe _dimm_dpe_75 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd76),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_75),
        .data_out(_dimm_dpe_out_75),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_76;
    wire _dimm_msbsa_76;
    dpe _dimm_dpe_76 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd77),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_76),
        .data_out(_dimm_dpe_out_76),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_77;
    wire _dimm_msbsa_77;
    dpe _dimm_dpe_77 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd78),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_77),
        .data_out(_dimm_dpe_out_77),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_78;
    wire _dimm_msbsa_78;
    dpe _dimm_dpe_78 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd79),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_78),
        .data_out(_dimm_dpe_out_78),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_79;
    wire _dimm_msbsa_79;
    dpe _dimm_dpe_79 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd80),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_79),
        .data_out(_dimm_dpe_out_79),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_80;
    wire _dimm_msbsa_80;
    dpe _dimm_dpe_80 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd81),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_80),
        .data_out(_dimm_dpe_out_80),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_81;
    wire _dimm_msbsa_81;
    dpe _dimm_dpe_81 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd82),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_81),
        .data_out(_dimm_dpe_out_81),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_82;
    wire _dimm_msbsa_82;
    dpe _dimm_dpe_82 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd83),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_82),
        .data_out(_dimm_dpe_out_82),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_83;
    wire _dimm_msbsa_83;
    dpe _dimm_dpe_83 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd84),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_83),
        .data_out(_dimm_dpe_out_83),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_84;
    wire _dimm_msbsa_84;
    dpe _dimm_dpe_84 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd85),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_84),
        .data_out(_dimm_dpe_out_84),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_85;
    wire _dimm_msbsa_85;
    dpe _dimm_dpe_85 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd86),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_85),
        .data_out(_dimm_dpe_out_85),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_86;
    wire _dimm_msbsa_86;
    dpe _dimm_dpe_86 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd87),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_86),
        .data_out(_dimm_dpe_out_86),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_87;
    wire _dimm_msbsa_87;
    dpe _dimm_dpe_87 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd88),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_87),
        .data_out(_dimm_dpe_out_87),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_88;
    wire _dimm_msbsa_88;
    dpe _dimm_dpe_88 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd89),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_88),
        .data_out(_dimm_dpe_out_88),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_89;
    wire _dimm_msbsa_89;
    dpe _dimm_dpe_89 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd90),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_89),
        .data_out(_dimm_dpe_out_89),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_90;
    wire _dimm_msbsa_90;
    dpe _dimm_dpe_90 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd91),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_90),
        .data_out(_dimm_dpe_out_90),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_91;
    wire _dimm_msbsa_91;
    dpe _dimm_dpe_91 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd92),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_91),
        .data_out(_dimm_dpe_out_91),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_92;
    wire _dimm_msbsa_92;
    dpe _dimm_dpe_92 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd93),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_92),
        .data_out(_dimm_dpe_out_92),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_93;
    wire _dimm_msbsa_93;
    dpe _dimm_dpe_93 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd94),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_93),
        .data_out(_dimm_dpe_out_93),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_94;
    wire _dimm_msbsa_94;
    dpe _dimm_dpe_94 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd95),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_94),
        .data_out(_dimm_dpe_out_94),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_95;
    wire _dimm_msbsa_95;
    dpe _dimm_dpe_95 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd96),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_95),
        .data_out(_dimm_dpe_out_95),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_96;
    wire _dimm_msbsa_96;
    dpe _dimm_dpe_96 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd97),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_96),
        .data_out(_dimm_dpe_out_96),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_97;
    wire _dimm_msbsa_97;
    dpe _dimm_dpe_97 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd98),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_97),
        .data_out(_dimm_dpe_out_97),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_98;
    wire _dimm_msbsa_98;
    dpe _dimm_dpe_98 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd99),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_98),
        .data_out(_dimm_dpe_out_98),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_99;
    wire _dimm_msbsa_99;
    dpe _dimm_dpe_99 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd100),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_99),
        .data_out(_dimm_dpe_out_99),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_100;
    wire _dimm_msbsa_100;
    dpe _dimm_dpe_100 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd101),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_100),
        .data_out(_dimm_dpe_out_100),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_101;
    wire _dimm_msbsa_101;
    dpe _dimm_dpe_101 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd102),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_101),
        .data_out(_dimm_dpe_out_101),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_102;
    wire _dimm_msbsa_102;
    dpe _dimm_dpe_102 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd103),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_102),
        .data_out(_dimm_dpe_out_102),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_103;
    wire _dimm_msbsa_103;
    dpe _dimm_dpe_103 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd104),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_103),
        .data_out(_dimm_dpe_out_103),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_104;
    wire _dimm_msbsa_104;
    dpe _dimm_dpe_104 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd105),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_104),
        .data_out(_dimm_dpe_out_104),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_105;
    wire _dimm_msbsa_105;
    dpe _dimm_dpe_105 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd106),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_105),
        .data_out(_dimm_dpe_out_105),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_106;
    wire _dimm_msbsa_106;
    dpe _dimm_dpe_106 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd107),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_106),
        .data_out(_dimm_dpe_out_106),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_107;
    wire _dimm_msbsa_107;
    dpe _dimm_dpe_107 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd108),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_107),
        .data_out(_dimm_dpe_out_107),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_108;
    wire _dimm_msbsa_108;
    dpe _dimm_dpe_108 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd109),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_108),
        .data_out(_dimm_dpe_out_108),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_109;
    wire _dimm_msbsa_109;
    dpe _dimm_dpe_109 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd110),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_109),
        .data_out(_dimm_dpe_out_109),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_110;
    wire _dimm_msbsa_110;
    dpe _dimm_dpe_110 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd111),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_110),
        .data_out(_dimm_dpe_out_110),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_111;
    wire _dimm_msbsa_111;
    dpe _dimm_dpe_111 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd112),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_111),
        .data_out(_dimm_dpe_out_111),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_112;
    wire _dimm_msbsa_112;
    dpe _dimm_dpe_112 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd113),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_112),
        .data_out(_dimm_dpe_out_112),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_113;
    wire _dimm_msbsa_113;
    dpe _dimm_dpe_113 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd114),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_113),
        .data_out(_dimm_dpe_out_113),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_114;
    wire _dimm_msbsa_114;
    dpe _dimm_dpe_114 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd115),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_114),
        .data_out(_dimm_dpe_out_114),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_115;
    wire _dimm_msbsa_115;
    dpe _dimm_dpe_115 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd116),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_115),
        .data_out(_dimm_dpe_out_115),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_116;
    wire _dimm_msbsa_116;
    dpe _dimm_dpe_116 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd117),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_116),
        .data_out(_dimm_dpe_out_116),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_117;
    wire _dimm_msbsa_117;
    dpe _dimm_dpe_117 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd118),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_117),
        .data_out(_dimm_dpe_out_117),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_118;
    wire _dimm_msbsa_118;
    dpe _dimm_dpe_118 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd119),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_118),
        .data_out(_dimm_dpe_out_118),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_119;
    wire _dimm_msbsa_119;
    dpe _dimm_dpe_119 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd120),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_119),
        .data_out(_dimm_dpe_out_119),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_120;
    wire _dimm_msbsa_120;
    dpe _dimm_dpe_120 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd121),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_120),
        .data_out(_dimm_dpe_out_120),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_121;
    wire _dimm_msbsa_121;
    dpe _dimm_dpe_121 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd122),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_121),
        .data_out(_dimm_dpe_out_121),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_122;
    wire _dimm_msbsa_122;
    dpe _dimm_dpe_122 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd123),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_122),
        .data_out(_dimm_dpe_out_122),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_123;
    wire _dimm_msbsa_123;
    dpe _dimm_dpe_123 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd124),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_123),
        .data_out(_dimm_dpe_out_123),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_124;
    wire _dimm_msbsa_124;
    dpe _dimm_dpe_124 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd125),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_124),
        .data_out(_dimm_dpe_out_124),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_125;
    wire _dimm_msbsa_125;
    dpe _dimm_dpe_125 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd126),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_125),
        .data_out(_dimm_dpe_out_125),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_126;
    wire _dimm_msbsa_126;
    dpe _dimm_dpe_126 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd127),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_126),
        .data_out(_dimm_dpe_out_126),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_127;
    wire _dimm_msbsa_127;
    dpe _dimm_dpe_127 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd128),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_127),
        .data_out(_dimm_dpe_out_127),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_128;
    wire _dimm_msbsa_128;
    dpe _dimm_dpe_128 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd129),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_128),
        .data_out(_dimm_dpe_out_128),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_129;
    wire _dimm_msbsa_129;
    dpe _dimm_dpe_129 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd130),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_129),
        .data_out(_dimm_dpe_out_129),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_130;
    wire _dimm_msbsa_130;
    dpe _dimm_dpe_130 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd131),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_130),
        .data_out(_dimm_dpe_out_130),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_131;
    wire _dimm_msbsa_131;
    dpe _dimm_dpe_131 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd132),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_131),
        .data_out(_dimm_dpe_out_131),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_132;
    wire _dimm_msbsa_132;
    dpe _dimm_dpe_132 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd133),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_132),
        .data_out(_dimm_dpe_out_132),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_133;
    wire _dimm_msbsa_133;
    dpe _dimm_dpe_133 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd134),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_133),
        .data_out(_dimm_dpe_out_133),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_134;
    wire _dimm_msbsa_134;
    dpe _dimm_dpe_134 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd135),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_134),
        .data_out(_dimm_dpe_out_134),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_135;
    wire _dimm_msbsa_135;
    dpe _dimm_dpe_135 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd136),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_135),
        .data_out(_dimm_dpe_out_135),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_136;
    wire _dimm_msbsa_136;
    dpe _dimm_dpe_136 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd137),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_136),
        .data_out(_dimm_dpe_out_136),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_137;
    wire _dimm_msbsa_137;
    dpe _dimm_dpe_137 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd138),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_137),
        .data_out(_dimm_dpe_out_137),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_138;
    wire _dimm_msbsa_138;
    dpe _dimm_dpe_138 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd139),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_138),
        .data_out(_dimm_dpe_out_138),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_139;
    wire _dimm_msbsa_139;
    dpe _dimm_dpe_139 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd140),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_139),
        .data_out(_dimm_dpe_out_139),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_140;
    wire _dimm_msbsa_140;
    dpe _dimm_dpe_140 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd141),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_140),
        .data_out(_dimm_dpe_out_140),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_141;
    wire _dimm_msbsa_141;
    dpe _dimm_dpe_141 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd142),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_141),
        .data_out(_dimm_dpe_out_141),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_142;
    wire _dimm_msbsa_142;
    dpe _dimm_dpe_142 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd143),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_142),
        .data_out(_dimm_dpe_out_142),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_143;
    wire _dimm_msbsa_143;
    dpe _dimm_dpe_143 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd144),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_143),
        .data_out(_dimm_dpe_out_143),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_144;
    wire _dimm_msbsa_144;
    dpe _dimm_dpe_144 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd145),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_144),
        .data_out(_dimm_dpe_out_144),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_145;
    wire _dimm_msbsa_145;
    dpe _dimm_dpe_145 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd146),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_145),
        .data_out(_dimm_dpe_out_145),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_146;
    wire _dimm_msbsa_146;
    dpe _dimm_dpe_146 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd147),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_146),
        .data_out(_dimm_dpe_out_146),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_147;
    wire _dimm_msbsa_147;
    dpe _dimm_dpe_147 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd148),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_147),
        .data_out(_dimm_dpe_out_147),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_148;
    wire _dimm_msbsa_148;
    dpe _dimm_dpe_148 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd149),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_148),
        .data_out(_dimm_dpe_out_148),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_149;
    wire _dimm_msbsa_149;
    dpe _dimm_dpe_149 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd150),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_149),
        .data_out(_dimm_dpe_out_149),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_150;
    wire _dimm_msbsa_150;
    dpe _dimm_dpe_150 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd151),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_150),
        .data_out(_dimm_dpe_out_150),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_151;
    wire _dimm_msbsa_151;
    dpe _dimm_dpe_151 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd152),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_151),
        .data_out(_dimm_dpe_out_151),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_152;
    wire _dimm_msbsa_152;
    dpe _dimm_dpe_152 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd153),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_152),
        .data_out(_dimm_dpe_out_152),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_153;
    wire _dimm_msbsa_153;
    dpe _dimm_dpe_153 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd154),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_153),
        .data_out(_dimm_dpe_out_153),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_154;
    wire _dimm_msbsa_154;
    dpe _dimm_dpe_154 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd155),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_154),
        .data_out(_dimm_dpe_out_154),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_155;
    wire _dimm_msbsa_155;
    dpe _dimm_dpe_155 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd156),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_155),
        .data_out(_dimm_dpe_out_155),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_156;
    wire _dimm_msbsa_156;
    dpe _dimm_dpe_156 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd157),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_156),
        .data_out(_dimm_dpe_out_156),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_157;
    wire _dimm_msbsa_157;
    dpe _dimm_dpe_157 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd158),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_157),
        .data_out(_dimm_dpe_out_157),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_158;
    wire _dimm_msbsa_158;
    dpe _dimm_dpe_158 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd159),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_158),
        .data_out(_dimm_dpe_out_158),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_159;
    wire _dimm_msbsa_159;
    dpe _dimm_dpe_159 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd160),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_159),
        .data_out(_dimm_dpe_out_159),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_160;
    wire _dimm_msbsa_160;
    dpe _dimm_dpe_160 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd161),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_160),
        .data_out(_dimm_dpe_out_160),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_161;
    wire _dimm_msbsa_161;
    dpe _dimm_dpe_161 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd162),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_161),
        .data_out(_dimm_dpe_out_161),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_162;
    wire _dimm_msbsa_162;
    dpe _dimm_dpe_162 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd163),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_162),
        .data_out(_dimm_dpe_out_162),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_163;
    wire _dimm_msbsa_163;
    dpe _dimm_dpe_163 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd164),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_163),
        .data_out(_dimm_dpe_out_163),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_164;
    wire _dimm_msbsa_164;
    dpe _dimm_dpe_164 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd165),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_164),
        .data_out(_dimm_dpe_out_164),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_165;
    wire _dimm_msbsa_165;
    dpe _dimm_dpe_165 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd166),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_165),
        .data_out(_dimm_dpe_out_165),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_166;
    wire _dimm_msbsa_166;
    dpe _dimm_dpe_166 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd167),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_166),
        .data_out(_dimm_dpe_out_166),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_167;
    wire _dimm_msbsa_167;
    dpe _dimm_dpe_167 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd168),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_167),
        .data_out(_dimm_dpe_out_167),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_168;
    wire _dimm_msbsa_168;
    dpe _dimm_dpe_168 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd169),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_168),
        .data_out(_dimm_dpe_out_168),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_169;
    wire _dimm_msbsa_169;
    dpe _dimm_dpe_169 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd170),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_169),
        .data_out(_dimm_dpe_out_169),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_170;
    wire _dimm_msbsa_170;
    dpe _dimm_dpe_170 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd171),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_170),
        .data_out(_dimm_dpe_out_170),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_171;
    wire _dimm_msbsa_171;
    dpe _dimm_dpe_171 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd172),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_171),
        .data_out(_dimm_dpe_out_171),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_172;
    wire _dimm_msbsa_172;
    dpe _dimm_dpe_172 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd173),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_172),
        .data_out(_dimm_dpe_out_172),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_173;
    wire _dimm_msbsa_173;
    dpe _dimm_dpe_173 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd174),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_173),
        .data_out(_dimm_dpe_out_173),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_174;
    wire _dimm_msbsa_174;
    dpe _dimm_dpe_174 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd175),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_174),
        .data_out(_dimm_dpe_out_174),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_175;
    wire _dimm_msbsa_175;
    dpe _dimm_dpe_175 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd176),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_175),
        .data_out(_dimm_dpe_out_175),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_176;
    wire _dimm_msbsa_176;
    dpe _dimm_dpe_176 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd177),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_176),
        .data_out(_dimm_dpe_out_176),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_177;
    wire _dimm_msbsa_177;
    dpe _dimm_dpe_177 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd178),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_177),
        .data_out(_dimm_dpe_out_177),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_178;
    wire _dimm_msbsa_178;
    dpe _dimm_dpe_178 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd179),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_178),
        .data_out(_dimm_dpe_out_178),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_179;
    wire _dimm_msbsa_179;
    dpe _dimm_dpe_179 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd180),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_179),
        .data_out(_dimm_dpe_out_179),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_180;
    wire _dimm_msbsa_180;
    dpe _dimm_dpe_180 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd181),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_180),
        .data_out(_dimm_dpe_out_180),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_181;
    wire _dimm_msbsa_181;
    dpe _dimm_dpe_181 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd182),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_181),
        .data_out(_dimm_dpe_out_181),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_182;
    wire _dimm_msbsa_182;
    dpe _dimm_dpe_182 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd183),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_182),
        .data_out(_dimm_dpe_out_182),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_183;
    wire _dimm_msbsa_183;
    dpe _dimm_dpe_183 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd184),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_183),
        .data_out(_dimm_dpe_out_183),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_184;
    wire _dimm_msbsa_184;
    dpe _dimm_dpe_184 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd185),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_184),
        .data_out(_dimm_dpe_out_184),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_185;
    wire _dimm_msbsa_185;
    dpe _dimm_dpe_185 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd186),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_185),
        .data_out(_dimm_dpe_out_185),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_186;
    wire _dimm_msbsa_186;
    dpe _dimm_dpe_186 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd187),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_186),
        .data_out(_dimm_dpe_out_186),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_187;
    wire _dimm_msbsa_187;
    dpe _dimm_dpe_187 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd188),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_187),
        .data_out(_dimm_dpe_out_187),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_188;
    wire _dimm_msbsa_188;
    dpe _dimm_dpe_188 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd189),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_188),
        .data_out(_dimm_dpe_out_188),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_189;
    wire _dimm_msbsa_189;
    dpe _dimm_dpe_189 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd190),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_189),
        .data_out(_dimm_dpe_out_189),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_190;
    wire _dimm_msbsa_190;
    dpe _dimm_dpe_190 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd191),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_190),
        .data_out(_dimm_dpe_out_190),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_191;
    wire _dimm_msbsa_191;
    dpe _dimm_dpe_191 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd192),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_191),
        .data_out(_dimm_dpe_out_191),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_192;
    wire _dimm_msbsa_192;
    dpe _dimm_dpe_192 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd193),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_192),
        .data_out(_dimm_dpe_out_192),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_193;
    wire _dimm_msbsa_193;
    dpe _dimm_dpe_193 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd194),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_193),
        .data_out(_dimm_dpe_out_193),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_194;
    wire _dimm_msbsa_194;
    dpe _dimm_dpe_194 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd195),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_194),
        .data_out(_dimm_dpe_out_194),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_195;
    wire _dimm_msbsa_195;
    dpe _dimm_dpe_195 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd196),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_195),
        .data_out(_dimm_dpe_out_195),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_196;
    wire _dimm_msbsa_196;
    dpe _dimm_dpe_196 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd197),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_196),
        .data_out(_dimm_dpe_out_196),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_197;
    wire _dimm_msbsa_197;
    dpe _dimm_dpe_197 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd198),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_197),
        .data_out(_dimm_dpe_out_197),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_198;
    wire _dimm_msbsa_198;
    dpe _dimm_dpe_198 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd199),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_198),
        .data_out(_dimm_dpe_out_198),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_199;
    wire _dimm_msbsa_199;
    dpe _dimm_dpe_199 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd200),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_199),
        .data_out(_dimm_dpe_out_199),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_200;
    wire _dimm_msbsa_200;
    dpe _dimm_dpe_200 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd201),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_200),
        .data_out(_dimm_dpe_out_200),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_201;
    wire _dimm_msbsa_201;
    dpe _dimm_dpe_201 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd202),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_201),
        .data_out(_dimm_dpe_out_201),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_202;
    wire _dimm_msbsa_202;
    dpe _dimm_dpe_202 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd203),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_202),
        .data_out(_dimm_dpe_out_202),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_203;
    wire _dimm_msbsa_203;
    dpe _dimm_dpe_203 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd204),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_203),
        .data_out(_dimm_dpe_out_203),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_204;
    wire _dimm_msbsa_204;
    dpe _dimm_dpe_204 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd205),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_204),
        .data_out(_dimm_dpe_out_204),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_205;
    wire _dimm_msbsa_205;
    dpe _dimm_dpe_205 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd206),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_205),
        .data_out(_dimm_dpe_out_205),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_206;
    wire _dimm_msbsa_206;
    dpe _dimm_dpe_206 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd207),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_206),
        .data_out(_dimm_dpe_out_206),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_207;
    wire _dimm_msbsa_207;
    dpe _dimm_dpe_207 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd208),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_207),
        .data_out(_dimm_dpe_out_207),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_208;
    wire _dimm_msbsa_208;
    dpe _dimm_dpe_208 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd209),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_208),
        .data_out(_dimm_dpe_out_208),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_209;
    wire _dimm_msbsa_209;
    dpe _dimm_dpe_209 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd210),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_209),
        .data_out(_dimm_dpe_out_209),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_210;
    wire _dimm_msbsa_210;
    dpe _dimm_dpe_210 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd211),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_210),
        .data_out(_dimm_dpe_out_210),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_211;
    wire _dimm_msbsa_211;
    dpe _dimm_dpe_211 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd212),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_211),
        .data_out(_dimm_dpe_out_211),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_212;
    wire _dimm_msbsa_212;
    dpe _dimm_dpe_212 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd213),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_212),
        .data_out(_dimm_dpe_out_212),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_213;
    wire _dimm_msbsa_213;
    dpe _dimm_dpe_213 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd214),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_213),
        .data_out(_dimm_dpe_out_213),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_214;
    wire _dimm_msbsa_214;
    dpe _dimm_dpe_214 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd215),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_214),
        .data_out(_dimm_dpe_out_214),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_215;
    wire _dimm_msbsa_215;
    dpe _dimm_dpe_215 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd216),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_215),
        .data_out(_dimm_dpe_out_215),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_216;
    wire _dimm_msbsa_216;
    dpe _dimm_dpe_216 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd217),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_216),
        .data_out(_dimm_dpe_out_216),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_217;
    wire _dimm_msbsa_217;
    dpe _dimm_dpe_217 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd218),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_217),
        .data_out(_dimm_dpe_out_217),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_218;
    wire _dimm_msbsa_218;
    dpe _dimm_dpe_218 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd219),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_218),
        .data_out(_dimm_dpe_out_218),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_219;
    wire _dimm_msbsa_219;
    dpe _dimm_dpe_219 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd220),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_219),
        .data_out(_dimm_dpe_out_219),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_220;
    wire _dimm_msbsa_220;
    dpe _dimm_dpe_220 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd221),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_220),
        .data_out(_dimm_dpe_out_220),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_221;
    wire _dimm_msbsa_221;
    dpe _dimm_dpe_221 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd222),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_221),
        .data_out(_dimm_dpe_out_221),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_222;
    wire _dimm_msbsa_222;
    dpe _dimm_dpe_222 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd223),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_222),
        .data_out(_dimm_dpe_out_222),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_223;
    wire _dimm_msbsa_223;
    dpe _dimm_dpe_223 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd224),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_223),
        .data_out(_dimm_dpe_out_223),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_224;
    wire _dimm_msbsa_224;
    dpe _dimm_dpe_224 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd225),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_224),
        .data_out(_dimm_dpe_out_224),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_225;
    wire _dimm_msbsa_225;
    dpe _dimm_dpe_225 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd226),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_225),
        .data_out(_dimm_dpe_out_225),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_226;
    wire _dimm_msbsa_226;
    dpe _dimm_dpe_226 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd227),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_226),
        .data_out(_dimm_dpe_out_226),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_227;
    wire _dimm_msbsa_227;
    dpe _dimm_dpe_227 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd228),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_227),
        .data_out(_dimm_dpe_out_227),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_228;
    wire _dimm_msbsa_228;
    dpe _dimm_dpe_228 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd229),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_228),
        .data_out(_dimm_dpe_out_228),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_229;
    wire _dimm_msbsa_229;
    dpe _dimm_dpe_229 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd230),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_229),
        .data_out(_dimm_dpe_out_229),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_230;
    wire _dimm_msbsa_230;
    dpe _dimm_dpe_230 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd231),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_230),
        .data_out(_dimm_dpe_out_230),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_231;
    wire _dimm_msbsa_231;
    dpe _dimm_dpe_231 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd232),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_231),
        .data_out(_dimm_dpe_out_231),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_232;
    wire _dimm_msbsa_232;
    dpe _dimm_dpe_232 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd233),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_232),
        .data_out(_dimm_dpe_out_232),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_233;
    wire _dimm_msbsa_233;
    dpe _dimm_dpe_233 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd234),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_233),
        .data_out(_dimm_dpe_out_233),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_234;
    wire _dimm_msbsa_234;
    dpe _dimm_dpe_234 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd235),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_234),
        .data_out(_dimm_dpe_out_234),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_235;
    wire _dimm_msbsa_235;
    dpe _dimm_dpe_235 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd236),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_235),
        .data_out(_dimm_dpe_out_235),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_236;
    wire _dimm_msbsa_236;
    dpe _dimm_dpe_236 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd237),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_236),
        .data_out(_dimm_dpe_out_236),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_237;
    wire _dimm_msbsa_237;
    dpe _dimm_dpe_237 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd238),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_237),
        .data_out(_dimm_dpe_out_237),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_238;
    wire _dimm_msbsa_238;
    dpe _dimm_dpe_238 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd239),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_238),
        .data_out(_dimm_dpe_out_238),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_239;
    wire _dimm_msbsa_239;
    dpe _dimm_dpe_239 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd240),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_239),
        .data_out(_dimm_dpe_out_239),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_240;
    wire _dimm_msbsa_240;
    dpe _dimm_dpe_240 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd241),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_240),
        .data_out(_dimm_dpe_out_240),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_241;
    wire _dimm_msbsa_241;
    dpe _dimm_dpe_241 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd242),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_241),
        .data_out(_dimm_dpe_out_241),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_242;
    wire _dimm_msbsa_242;
    dpe _dimm_dpe_242 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd243),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_242),
        .data_out(_dimm_dpe_out_242),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_243;
    wire _dimm_msbsa_243;
    dpe _dimm_dpe_243 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd244),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_243),
        .data_out(_dimm_dpe_out_243),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_244;
    wire _dimm_msbsa_244;
    dpe _dimm_dpe_244 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd245),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_244),
        .data_out(_dimm_dpe_out_244),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_245;
    wire _dimm_msbsa_245;
    dpe _dimm_dpe_245 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd246),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_245),
        .data_out(_dimm_dpe_out_245),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_246;
    wire _dimm_msbsa_246;
    dpe _dimm_dpe_246 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd247),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_246),
        .data_out(_dimm_dpe_out_246),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_247;
    wire _dimm_msbsa_247;
    dpe _dimm_dpe_247 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd248),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_247),
        .data_out(_dimm_dpe_out_247),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_248;
    wire _dimm_msbsa_248;
    dpe _dimm_dpe_248 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd249),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_248),
        .data_out(_dimm_dpe_out_248),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_249;
    wire _dimm_msbsa_249;
    dpe _dimm_dpe_249 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd250),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_249),
        .data_out(_dimm_dpe_out_249),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_250;
    wire _dimm_msbsa_250;
    dpe _dimm_dpe_250 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd251),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_250),
        .data_out(_dimm_dpe_out_250),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_251;
    wire _dimm_msbsa_251;
    dpe _dimm_dpe_251 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd252),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_251),
        .data_out(_dimm_dpe_out_251),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_252;
    wire _dimm_msbsa_252;
    dpe _dimm_dpe_252 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd253),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_252),
        .data_out(_dimm_dpe_out_252),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_253;
    wire _dimm_msbsa_253;
    dpe _dimm_dpe_253 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd254),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_253),
        .data_out(_dimm_dpe_out_253),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_254;
    wire _dimm_msbsa_254;
    dpe _dimm_dpe_254 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd255),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_254),
        .data_out(_dimm_dpe_out_254),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_255;
    wire _dimm_msbsa_255;
    dpe _dimm_dpe_255 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd256),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_255),
        .data_out(_dimm_dpe_out_255),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_256;
    wire _dimm_msbsa_256;
    dpe _dimm_dpe_256 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd257),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_256),
        .data_out(_dimm_dpe_out_256),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_257;
    wire _dimm_msbsa_257;
    dpe _dimm_dpe_257 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd258),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_257),
        .data_out(_dimm_dpe_out_257),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_258;
    wire _dimm_msbsa_258;
    dpe _dimm_dpe_258 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd259),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_258),
        .data_out(_dimm_dpe_out_258),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_259;
    wire _dimm_msbsa_259;
    dpe _dimm_dpe_259 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd260),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_259),
        .data_out(_dimm_dpe_out_259),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_260;
    wire _dimm_msbsa_260;
    dpe _dimm_dpe_260 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd261),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_260),
        .data_out(_dimm_dpe_out_260),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_261;
    wire _dimm_msbsa_261;
    dpe _dimm_dpe_261 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd262),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_261),
        .data_out(_dimm_dpe_out_261),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_262;
    wire _dimm_msbsa_262;
    dpe _dimm_dpe_262 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd263),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_262),
        .data_out(_dimm_dpe_out_262),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_263;
    wire _dimm_msbsa_263;
    dpe _dimm_dpe_263 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd264),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_263),
        .data_out(_dimm_dpe_out_263),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_264;
    wire _dimm_msbsa_264;
    dpe _dimm_dpe_264 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd265),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_264),
        .data_out(_dimm_dpe_out_264),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_265;
    wire _dimm_msbsa_265;
    dpe _dimm_dpe_265 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd266),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_265),
        .data_out(_dimm_dpe_out_265),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_266;
    wire _dimm_msbsa_266;
    dpe _dimm_dpe_266 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd267),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_266),
        .data_out(_dimm_dpe_out_266),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_267;
    wire _dimm_msbsa_267;
    dpe _dimm_dpe_267 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd268),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_267),
        .data_out(_dimm_dpe_out_267),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_268;
    wire _dimm_msbsa_268;
    dpe _dimm_dpe_268 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd269),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_268),
        .data_out(_dimm_dpe_out_268),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_269;
    wire _dimm_msbsa_269;
    dpe _dimm_dpe_269 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd270),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_269),
        .data_out(_dimm_dpe_out_269),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_270;
    wire _dimm_msbsa_270;
    dpe _dimm_dpe_270 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd271),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_270),
        .data_out(_dimm_dpe_out_270),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_271;
    wire _dimm_msbsa_271;
    dpe _dimm_dpe_271 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd272),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_271),
        .data_out(_dimm_dpe_out_271),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_272;
    wire _dimm_msbsa_272;
    dpe _dimm_dpe_272 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd273),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_272),
        .data_out(_dimm_dpe_out_272),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_273;
    wire _dimm_msbsa_273;
    dpe _dimm_dpe_273 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd274),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_273),
        .data_out(_dimm_dpe_out_273),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_274;
    wire _dimm_msbsa_274;
    dpe _dimm_dpe_274 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd275),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_274),
        .data_out(_dimm_dpe_out_274),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_275;
    wire _dimm_msbsa_275;
    dpe _dimm_dpe_275 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd276),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_275),
        .data_out(_dimm_dpe_out_275),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_276;
    wire _dimm_msbsa_276;
    dpe _dimm_dpe_276 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd277),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_276),
        .data_out(_dimm_dpe_out_276),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_277;
    wire _dimm_msbsa_277;
    dpe _dimm_dpe_277 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd278),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_277),
        .data_out(_dimm_dpe_out_277),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_278;
    wire _dimm_msbsa_278;
    dpe _dimm_dpe_278 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd279),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_278),
        .data_out(_dimm_dpe_out_278),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_279;
    wire _dimm_msbsa_279;
    dpe _dimm_dpe_279 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd280),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_279),
        .data_out(_dimm_dpe_out_279),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_280;
    wire _dimm_msbsa_280;
    dpe _dimm_dpe_280 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd281),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_280),
        .data_out(_dimm_dpe_out_280),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_281;
    wire _dimm_msbsa_281;
    dpe _dimm_dpe_281 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd282),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_281),
        .data_out(_dimm_dpe_out_281),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_282;
    wire _dimm_msbsa_282;
    dpe _dimm_dpe_282 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd283),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_282),
        .data_out(_dimm_dpe_out_282),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_283;
    wire _dimm_msbsa_283;
    dpe _dimm_dpe_283 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd284),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_283),
        .data_out(_dimm_dpe_out_283),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_284;
    wire _dimm_msbsa_284;
    dpe _dimm_dpe_284 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd285),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_284),
        .data_out(_dimm_dpe_out_284),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_285;
    wire _dimm_msbsa_285;
    dpe _dimm_dpe_285 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd286),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_285),
        .data_out(_dimm_dpe_out_285),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_286;
    wire _dimm_msbsa_286;
    dpe _dimm_dpe_286 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd287),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_286),
        .data_out(_dimm_dpe_out_286),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_287;
    wire _dimm_msbsa_287;
    dpe _dimm_dpe_287 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd288),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_287),
        .data_out(_dimm_dpe_out_287),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_288;
    wire _dimm_msbsa_288;
    dpe _dimm_dpe_288 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd289),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_288),
        .data_out(_dimm_dpe_out_288),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_289;
    wire _dimm_msbsa_289;
    dpe _dimm_dpe_289 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd290),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_289),
        .data_out(_dimm_dpe_out_289),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_290;
    wire _dimm_msbsa_290;
    dpe _dimm_dpe_290 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd291),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_290),
        .data_out(_dimm_dpe_out_290),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_291;
    wire _dimm_msbsa_291;
    dpe _dimm_dpe_291 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd292),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_291),
        .data_out(_dimm_dpe_out_291),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_292;
    wire _dimm_msbsa_292;
    dpe _dimm_dpe_292 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd293),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_292),
        .data_out(_dimm_dpe_out_292),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_293;
    wire _dimm_msbsa_293;
    dpe _dimm_dpe_293 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd294),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_293),
        .data_out(_dimm_dpe_out_293),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_294;
    wire _dimm_msbsa_294;
    dpe _dimm_dpe_294 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd295),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_294),
        .data_out(_dimm_dpe_out_294),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_295;
    wire _dimm_msbsa_295;
    dpe _dimm_dpe_295 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd296),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_295),
        .data_out(_dimm_dpe_out_295),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_296;
    wire _dimm_msbsa_296;
    dpe _dimm_dpe_296 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd297),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_296),
        .data_out(_dimm_dpe_out_296),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_297;
    wire _dimm_msbsa_297;
    dpe _dimm_dpe_297 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd298),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_297),
        .data_out(_dimm_dpe_out_297),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_298;
    wire _dimm_msbsa_298;
    dpe _dimm_dpe_298 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd299),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_298),
        .data_out(_dimm_dpe_out_298),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_299;
    wire _dimm_msbsa_299;
    dpe _dimm_dpe_299 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd300),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_299),
        .data_out(_dimm_dpe_out_299),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_300;
    wire _dimm_msbsa_300;
    dpe _dimm_dpe_300 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd301),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_300),
        .data_out(_dimm_dpe_out_300),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_301;
    wire _dimm_msbsa_301;
    dpe _dimm_dpe_301 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd302),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_301),
        .data_out(_dimm_dpe_out_301),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_302;
    wire _dimm_msbsa_302;
    dpe _dimm_dpe_302 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd303),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_302),
        .data_out(_dimm_dpe_out_302),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_303;
    wire _dimm_msbsa_303;
    dpe _dimm_dpe_303 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd304),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_303),
        .data_out(_dimm_dpe_out_303),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_304;
    wire _dimm_msbsa_304;
    dpe _dimm_dpe_304 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd305),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_304),
        .data_out(_dimm_dpe_out_304),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_305;
    wire _dimm_msbsa_305;
    dpe _dimm_dpe_305 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd306),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_305),
        .data_out(_dimm_dpe_out_305),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_306;
    wire _dimm_msbsa_306;
    dpe _dimm_dpe_306 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd307),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_306),
        .data_out(_dimm_dpe_out_306),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_307;
    wire _dimm_msbsa_307;
    dpe _dimm_dpe_307 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd308),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_307),
        .data_out(_dimm_dpe_out_307),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_308;
    wire _dimm_msbsa_308;
    dpe _dimm_dpe_308 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd309),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_308),
        .data_out(_dimm_dpe_out_308),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_309;
    wire _dimm_msbsa_309;
    dpe _dimm_dpe_309 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd310),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_309),
        .data_out(_dimm_dpe_out_309),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_310;
    wire _dimm_msbsa_310;
    dpe _dimm_dpe_310 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd311),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_310),
        .data_out(_dimm_dpe_out_310),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_311;
    wire _dimm_msbsa_311;
    dpe _dimm_dpe_311 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd312),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_311),
        .data_out(_dimm_dpe_out_311),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_312;
    wire _dimm_msbsa_312;
    dpe _dimm_dpe_312 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd313),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_312),
        .data_out(_dimm_dpe_out_312),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_313;
    wire _dimm_msbsa_313;
    dpe _dimm_dpe_313 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd314),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_313),
        .data_out(_dimm_dpe_out_313),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_314;
    wire _dimm_msbsa_314;
    dpe _dimm_dpe_314 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd315),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_314),
        .data_out(_dimm_dpe_out_314),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_315;
    wire _dimm_msbsa_315;
    dpe _dimm_dpe_315 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd316),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_315),
        .data_out(_dimm_dpe_out_315),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_316;
    wire _dimm_msbsa_316;
    dpe _dimm_dpe_316 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd317),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_316),
        .data_out(_dimm_dpe_out_316),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_317;
    wire _dimm_msbsa_317;
    dpe _dimm_dpe_317 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd318),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_317),
        .data_out(_dimm_dpe_out_317),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_318;
    wire _dimm_msbsa_318;
    dpe _dimm_dpe_318 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd319),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_318),
        .data_out(_dimm_dpe_out_318),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_319;
    wire _dimm_msbsa_319;
    dpe _dimm_dpe_319 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd320),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_319),
        .data_out(_dimm_dpe_out_319),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_320;
    wire _dimm_msbsa_320;
    dpe _dimm_dpe_320 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd321),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_320),
        .data_out(_dimm_dpe_out_320),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_321;
    wire _dimm_msbsa_321;
    dpe _dimm_dpe_321 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd322),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_321),
        .data_out(_dimm_dpe_out_321),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_322;
    wire _dimm_msbsa_322;
    dpe _dimm_dpe_322 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd323),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_322),
        .data_out(_dimm_dpe_out_322),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_323;
    wire _dimm_msbsa_323;
    dpe _dimm_dpe_323 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd324),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_323),
        .data_out(_dimm_dpe_out_323),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_324;
    wire _dimm_msbsa_324;
    dpe _dimm_dpe_324 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd325),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_324),
        .data_out(_dimm_dpe_out_324),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_325;
    wire _dimm_msbsa_325;
    dpe _dimm_dpe_325 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd326),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_325),
        .data_out(_dimm_dpe_out_325),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_326;
    wire _dimm_msbsa_326;
    dpe _dimm_dpe_326 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd327),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_326),
        .data_out(_dimm_dpe_out_326),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_327;
    wire _dimm_msbsa_327;
    dpe _dimm_dpe_327 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd328),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_327),
        .data_out(_dimm_dpe_out_327),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_328;
    wire _dimm_msbsa_328;
    dpe _dimm_dpe_328 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd329),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_328),
        .data_out(_dimm_dpe_out_328),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_329;
    wire _dimm_msbsa_329;
    dpe _dimm_dpe_329 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd330),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_329),
        .data_out(_dimm_dpe_out_329),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_330;
    wire _dimm_msbsa_330;
    dpe _dimm_dpe_330 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd331),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_330),
        .data_out(_dimm_dpe_out_330),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_331;
    wire _dimm_msbsa_331;
    dpe _dimm_dpe_331 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd332),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_331),
        .data_out(_dimm_dpe_out_331),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_332;
    wire _dimm_msbsa_332;
    dpe _dimm_dpe_332 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd333),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_332),
        .data_out(_dimm_dpe_out_332),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_333;
    wire _dimm_msbsa_333;
    dpe _dimm_dpe_333 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd334),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_333),
        .data_out(_dimm_dpe_out_333),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_334;
    wire _dimm_msbsa_334;
    dpe _dimm_dpe_334 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd335),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_334),
        .data_out(_dimm_dpe_out_334),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_335;
    wire _dimm_msbsa_335;
    dpe _dimm_dpe_335 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd336),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_335),
        .data_out(_dimm_dpe_out_335),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_336;
    wire _dimm_msbsa_336;
    dpe _dimm_dpe_336 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd337),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_336),
        .data_out(_dimm_dpe_out_336),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_337;
    wire _dimm_msbsa_337;
    dpe _dimm_dpe_337 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd338),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_337),
        .data_out(_dimm_dpe_out_337),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_338;
    wire _dimm_msbsa_338;
    dpe _dimm_dpe_338 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd339),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_338),
        .data_out(_dimm_dpe_out_338),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_339;
    wire _dimm_msbsa_339;
    dpe _dimm_dpe_339 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd340),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_339),
        .data_out(_dimm_dpe_out_339),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_340;
    wire _dimm_msbsa_340;
    dpe _dimm_dpe_340 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd341),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_340),
        .data_out(_dimm_dpe_out_340),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_341;
    wire _dimm_msbsa_341;
    dpe _dimm_dpe_341 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd342),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_341),
        .data_out(_dimm_dpe_out_341),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_342;
    wire _dimm_msbsa_342;
    dpe _dimm_dpe_342 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd343),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_342),
        .data_out(_dimm_dpe_out_342),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_343;
    wire _dimm_msbsa_343;
    dpe _dimm_dpe_343 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd344),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_343),
        .data_out(_dimm_dpe_out_343),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_344;
    wire _dimm_msbsa_344;
    dpe _dimm_dpe_344 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd345),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_344),
        .data_out(_dimm_dpe_out_344),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_345;
    wire _dimm_msbsa_345;
    dpe _dimm_dpe_345 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd346),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_345),
        .data_out(_dimm_dpe_out_345),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_346;
    wire _dimm_msbsa_346;
    dpe _dimm_dpe_346 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd347),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_346),
        .data_out(_dimm_dpe_out_346),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_347;
    wire _dimm_msbsa_347;
    dpe _dimm_dpe_347 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd348),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_347),
        .data_out(_dimm_dpe_out_347),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_348;
    wire _dimm_msbsa_348;
    dpe _dimm_dpe_348 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd349),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_348),
        .data_out(_dimm_dpe_out_348),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_349;
    wire _dimm_msbsa_349;
    dpe _dimm_dpe_349 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd350),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_349),
        .data_out(_dimm_dpe_out_349),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_350;
    wire _dimm_msbsa_350;
    dpe _dimm_dpe_350 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd351),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_350),
        .data_out(_dimm_dpe_out_350),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_351;
    wire _dimm_msbsa_351;
    dpe _dimm_dpe_351 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd352),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_351),
        .data_out(_dimm_dpe_out_351),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_352;
    wire _dimm_msbsa_352;
    dpe _dimm_dpe_352 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd353),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_352),
        .data_out(_dimm_dpe_out_352),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_353;
    wire _dimm_msbsa_353;
    dpe _dimm_dpe_353 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd354),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_353),
        .data_out(_dimm_dpe_out_353),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_354;
    wire _dimm_msbsa_354;
    dpe _dimm_dpe_354 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd355),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_354),
        .data_out(_dimm_dpe_out_354),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_355;
    wire _dimm_msbsa_355;
    dpe _dimm_dpe_355 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd356),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_355),
        .data_out(_dimm_dpe_out_355),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_356;
    wire _dimm_msbsa_356;
    dpe _dimm_dpe_356 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd357),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_356),
        .data_out(_dimm_dpe_out_356),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_357;
    wire _dimm_msbsa_357;
    dpe _dimm_dpe_357 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd358),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_357),
        .data_out(_dimm_dpe_out_357),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_358;
    wire _dimm_msbsa_358;
    dpe _dimm_dpe_358 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd359),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_358),
        .data_out(_dimm_dpe_out_358),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_359;
    wire _dimm_msbsa_359;
    dpe _dimm_dpe_359 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd360),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_359),
        .data_out(_dimm_dpe_out_359),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_360;
    wire _dimm_msbsa_360;
    dpe _dimm_dpe_360 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd361),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_360),
        .data_out(_dimm_dpe_out_360),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_361;
    wire _dimm_msbsa_361;
    dpe _dimm_dpe_361 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd362),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_361),
        .data_out(_dimm_dpe_out_361),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_362;
    wire _dimm_msbsa_362;
    dpe _dimm_dpe_362 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd363),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_362),
        .data_out(_dimm_dpe_out_362),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_363;
    wire _dimm_msbsa_363;
    dpe _dimm_dpe_363 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd364),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_363),
        .data_out(_dimm_dpe_out_363),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_364;
    wire _dimm_msbsa_364;
    dpe _dimm_dpe_364 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd365),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_364),
        .data_out(_dimm_dpe_out_364),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_365;
    wire _dimm_msbsa_365;
    dpe _dimm_dpe_365 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd366),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_365),
        .data_out(_dimm_dpe_out_365),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_366;
    wire _dimm_msbsa_366;
    dpe _dimm_dpe_366 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd367),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_366),
        .data_out(_dimm_dpe_out_366),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_367;
    wire _dimm_msbsa_367;
    dpe _dimm_dpe_367 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd368),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_367),
        .data_out(_dimm_dpe_out_367),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_368;
    wire _dimm_msbsa_368;
    dpe _dimm_dpe_368 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd369),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_368),
        .data_out(_dimm_dpe_out_368),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_369;
    wire _dimm_msbsa_369;
    dpe _dimm_dpe_369 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd370),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_369),
        .data_out(_dimm_dpe_out_369),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_370;
    wire _dimm_msbsa_370;
    dpe _dimm_dpe_370 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd371),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_370),
        .data_out(_dimm_dpe_out_370),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_371;
    wire _dimm_msbsa_371;
    dpe _dimm_dpe_371 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd372),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_371),
        .data_out(_dimm_dpe_out_371),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_372;
    wire _dimm_msbsa_372;
    dpe _dimm_dpe_372 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd373),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_372),
        .data_out(_dimm_dpe_out_372),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_373;
    wire _dimm_msbsa_373;
    dpe _dimm_dpe_373 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd374),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_373),
        .data_out(_dimm_dpe_out_373),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_374;
    wire _dimm_msbsa_374;
    dpe _dimm_dpe_374 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd375),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_374),
        .data_out(_dimm_dpe_out_374),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_375;
    wire _dimm_msbsa_375;
    dpe _dimm_dpe_375 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd376),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_375),
        .data_out(_dimm_dpe_out_375),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_376;
    wire _dimm_msbsa_376;
    dpe _dimm_dpe_376 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd377),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_376),
        .data_out(_dimm_dpe_out_376),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_377;
    wire _dimm_msbsa_377;
    dpe _dimm_dpe_377 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd378),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_377),
        .data_out(_dimm_dpe_out_377),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_378;
    wire _dimm_msbsa_378;
    dpe _dimm_dpe_378 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd379),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_378),
        .data_out(_dimm_dpe_out_378),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_379;
    wire _dimm_msbsa_379;
    dpe _dimm_dpe_379 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd380),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_379),
        .data_out(_dimm_dpe_out_379),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_380;
    wire _dimm_msbsa_380;
    dpe _dimm_dpe_380 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd381),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_380),
        .data_out(_dimm_dpe_out_380),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_381;
    wire _dimm_msbsa_381;
    dpe _dimm_dpe_381 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd382),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_381),
        .data_out(_dimm_dpe_out_381),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_382;
    wire _dimm_msbsa_382;
    dpe _dimm_dpe_382 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd383),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_382),
        .data_out(_dimm_dpe_out_382),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _dimm_dpe_out_383;
    wire _dimm_msbsa_383;
    dpe _dimm_dpe_383 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd384),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_dimm_msbsa_383),
        .data_out(_dimm_dpe_out_383),
        .dpe_done(), .reg_full(),
        .shift_add_done(), .shift_add_bypass_ctrl()
    );
    assign data_out = data_b1_ln_ffn ^ {39'b0, _dimm_dpe_out_0[0] ^ _dimm_dpe_out_1[0] ^ _dimm_dpe_out_2[0] ^ _dimm_dpe_out_3[0] ^ _dimm_dpe_out_4[0] ^ _dimm_dpe_out_5[0] ^ _dimm_dpe_out_6[0] ^ _dimm_dpe_out_7[0] ^ _dimm_dpe_out_8[0] ^ _dimm_dpe_out_9[0] ^ _dimm_dpe_out_10[0] ^ _dimm_dpe_out_11[0] ^ _dimm_dpe_out_12[0] ^ _dimm_dpe_out_13[0] ^ _dimm_dpe_out_14[0] ^ _dimm_dpe_out_15[0] ^ _dimm_dpe_out_16[0] ^ _dimm_dpe_out_17[0] ^ _dimm_dpe_out_18[0] ^ _dimm_dpe_out_19[0] ^ _dimm_dpe_out_20[0] ^ _dimm_dpe_out_21[0] ^ _dimm_dpe_out_22[0] ^ _dimm_dpe_out_23[0] ^ _dimm_dpe_out_24[0] ^ _dimm_dpe_out_25[0] ^ _dimm_dpe_out_26[0] ^ _dimm_dpe_out_27[0] ^ _dimm_dpe_out_28[0] ^ _dimm_dpe_out_29[0] ^ _dimm_dpe_out_30[0] ^ _dimm_dpe_out_31[0] ^ _dimm_dpe_out_32[0] ^ _dimm_dpe_out_33[0] ^ _dimm_dpe_out_34[0] ^ _dimm_dpe_out_35[0] ^ _dimm_dpe_out_36[0] ^ _dimm_dpe_out_37[0] ^ _dimm_dpe_out_38[0] ^ _dimm_dpe_out_39[0] ^ _dimm_dpe_out_40[0] ^ _dimm_dpe_out_41[0] ^ _dimm_dpe_out_42[0] ^ _dimm_dpe_out_43[0] ^ _dimm_dpe_out_44[0] ^ _dimm_dpe_out_45[0] ^ _dimm_dpe_out_46[0] ^ _dimm_dpe_out_47[0] ^ _dimm_dpe_out_48[0] ^ _dimm_dpe_out_49[0] ^ _dimm_dpe_out_50[0] ^ _dimm_dpe_out_51[0] ^ _dimm_dpe_out_52[0] ^ _dimm_dpe_out_53[0] ^ _dimm_dpe_out_54[0] ^ _dimm_dpe_out_55[0] ^ _dimm_dpe_out_56[0] ^ _dimm_dpe_out_57[0] ^ _dimm_dpe_out_58[0] ^ _dimm_dpe_out_59[0] ^ _dimm_dpe_out_60[0] ^ _dimm_dpe_out_61[0] ^ _dimm_dpe_out_62[0] ^ _dimm_dpe_out_63[0] ^ _dimm_dpe_out_64[0] ^ _dimm_dpe_out_65[0] ^ _dimm_dpe_out_66[0] ^ _dimm_dpe_out_67[0] ^ _dimm_dpe_out_68[0] ^ _dimm_dpe_out_69[0] ^ _dimm_dpe_out_70[0] ^ _dimm_dpe_out_71[0] ^ _dimm_dpe_out_72[0] ^ _dimm_dpe_out_73[0] ^ _dimm_dpe_out_74[0] ^ _dimm_dpe_out_75[0] ^ _dimm_dpe_out_76[0] ^ _dimm_dpe_out_77[0] ^ _dimm_dpe_out_78[0] ^ _dimm_dpe_out_79[0] ^ _dimm_dpe_out_80[0] ^ _dimm_dpe_out_81[0] ^ _dimm_dpe_out_82[0] ^ _dimm_dpe_out_83[0] ^ _dimm_dpe_out_84[0] ^ _dimm_dpe_out_85[0] ^ _dimm_dpe_out_86[0] ^ _dimm_dpe_out_87[0] ^ _dimm_dpe_out_88[0] ^ _dimm_dpe_out_89[0] ^ _dimm_dpe_out_90[0] ^ _dimm_dpe_out_91[0] ^ _dimm_dpe_out_92[0] ^ _dimm_dpe_out_93[0] ^ _dimm_dpe_out_94[0] ^ _dimm_dpe_out_95[0] ^ _dimm_dpe_out_96[0] ^ _dimm_dpe_out_97[0] ^ _dimm_dpe_out_98[0] ^ _dimm_dpe_out_99[0] ^ _dimm_dpe_out_100[0] ^ _dimm_dpe_out_101[0] ^ _dimm_dpe_out_102[0] ^ _dimm_dpe_out_103[0] ^ _dimm_dpe_out_104[0] ^ _dimm_dpe_out_105[0] ^ _dimm_dpe_out_106[0] ^ _dimm_dpe_out_107[0] ^ _dimm_dpe_out_108[0] ^ _dimm_dpe_out_109[0] ^ _dimm_dpe_out_110[0] ^ _dimm_dpe_out_111[0] ^ _dimm_dpe_out_112[0] ^ _dimm_dpe_out_113[0] ^ _dimm_dpe_out_114[0] ^ _dimm_dpe_out_115[0] ^ _dimm_dpe_out_116[0] ^ _dimm_dpe_out_117[0] ^ _dimm_dpe_out_118[0] ^ _dimm_dpe_out_119[0] ^ _dimm_dpe_out_120[0] ^ _dimm_dpe_out_121[0] ^ _dimm_dpe_out_122[0] ^ _dimm_dpe_out_123[0] ^ _dimm_dpe_out_124[0] ^ _dimm_dpe_out_125[0] ^ _dimm_dpe_out_126[0] ^ _dimm_dpe_out_127[0] ^ _dimm_dpe_out_128[0] ^ _dimm_dpe_out_129[0] ^ _dimm_dpe_out_130[0] ^ _dimm_dpe_out_131[0] ^ _dimm_dpe_out_132[0] ^ _dimm_dpe_out_133[0] ^ _dimm_dpe_out_134[0] ^ _dimm_dpe_out_135[0] ^ _dimm_dpe_out_136[0] ^ _dimm_dpe_out_137[0] ^ _dimm_dpe_out_138[0] ^ _dimm_dpe_out_139[0] ^ _dimm_dpe_out_140[0] ^ _dimm_dpe_out_141[0] ^ _dimm_dpe_out_142[0] ^ _dimm_dpe_out_143[0] ^ _dimm_dpe_out_144[0] ^ _dimm_dpe_out_145[0] ^ _dimm_dpe_out_146[0] ^ _dimm_dpe_out_147[0] ^ _dimm_dpe_out_148[0] ^ _dimm_dpe_out_149[0] ^ _dimm_dpe_out_150[0] ^ _dimm_dpe_out_151[0] ^ _dimm_dpe_out_152[0] ^ _dimm_dpe_out_153[0] ^ _dimm_dpe_out_154[0] ^ _dimm_dpe_out_155[0] ^ _dimm_dpe_out_156[0] ^ _dimm_dpe_out_157[0] ^ _dimm_dpe_out_158[0] ^ _dimm_dpe_out_159[0] ^ _dimm_dpe_out_160[0] ^ _dimm_dpe_out_161[0] ^ _dimm_dpe_out_162[0] ^ _dimm_dpe_out_163[0] ^ _dimm_dpe_out_164[0] ^ _dimm_dpe_out_165[0] ^ _dimm_dpe_out_166[0] ^ _dimm_dpe_out_167[0] ^ _dimm_dpe_out_168[0] ^ _dimm_dpe_out_169[0] ^ _dimm_dpe_out_170[0] ^ _dimm_dpe_out_171[0] ^ _dimm_dpe_out_172[0] ^ _dimm_dpe_out_173[0] ^ _dimm_dpe_out_174[0] ^ _dimm_dpe_out_175[0] ^ _dimm_dpe_out_176[0] ^ _dimm_dpe_out_177[0] ^ _dimm_dpe_out_178[0] ^ _dimm_dpe_out_179[0] ^ _dimm_dpe_out_180[0] ^ _dimm_dpe_out_181[0] ^ _dimm_dpe_out_182[0] ^ _dimm_dpe_out_183[0] ^ _dimm_dpe_out_184[0] ^ _dimm_dpe_out_185[0] ^ _dimm_dpe_out_186[0] ^ _dimm_dpe_out_187[0] ^ _dimm_dpe_out_188[0] ^ _dimm_dpe_out_189[0] ^ _dimm_dpe_out_190[0] ^ _dimm_dpe_out_191[0] ^ _dimm_dpe_out_192[0] ^ _dimm_dpe_out_193[0] ^ _dimm_dpe_out_194[0] ^ _dimm_dpe_out_195[0] ^ _dimm_dpe_out_196[0] ^ _dimm_dpe_out_197[0] ^ _dimm_dpe_out_198[0] ^ _dimm_dpe_out_199[0] ^ _dimm_dpe_out_200[0] ^ _dimm_dpe_out_201[0] ^ _dimm_dpe_out_202[0] ^ _dimm_dpe_out_203[0] ^ _dimm_dpe_out_204[0] ^ _dimm_dpe_out_205[0] ^ _dimm_dpe_out_206[0] ^ _dimm_dpe_out_207[0] ^ _dimm_dpe_out_208[0] ^ _dimm_dpe_out_209[0] ^ _dimm_dpe_out_210[0] ^ _dimm_dpe_out_211[0] ^ _dimm_dpe_out_212[0] ^ _dimm_dpe_out_213[0] ^ _dimm_dpe_out_214[0] ^ _dimm_dpe_out_215[0] ^ _dimm_dpe_out_216[0] ^ _dimm_dpe_out_217[0] ^ _dimm_dpe_out_218[0] ^ _dimm_dpe_out_219[0] ^ _dimm_dpe_out_220[0] ^ _dimm_dpe_out_221[0] ^ _dimm_dpe_out_222[0] ^ _dimm_dpe_out_223[0] ^ _dimm_dpe_out_224[0] ^ _dimm_dpe_out_225[0] ^ _dimm_dpe_out_226[0] ^ _dimm_dpe_out_227[0] ^ _dimm_dpe_out_228[0] ^ _dimm_dpe_out_229[0] ^ _dimm_dpe_out_230[0] ^ _dimm_dpe_out_231[0] ^ _dimm_dpe_out_232[0] ^ _dimm_dpe_out_233[0] ^ _dimm_dpe_out_234[0] ^ _dimm_dpe_out_235[0] ^ _dimm_dpe_out_236[0] ^ _dimm_dpe_out_237[0] ^ _dimm_dpe_out_238[0] ^ _dimm_dpe_out_239[0] ^ _dimm_dpe_out_240[0] ^ _dimm_dpe_out_241[0] ^ _dimm_dpe_out_242[0] ^ _dimm_dpe_out_243[0] ^ _dimm_dpe_out_244[0] ^ _dimm_dpe_out_245[0] ^ _dimm_dpe_out_246[0] ^ _dimm_dpe_out_247[0] ^ _dimm_dpe_out_248[0] ^ _dimm_dpe_out_249[0] ^ _dimm_dpe_out_250[0] ^ _dimm_dpe_out_251[0] ^ _dimm_dpe_out_252[0] ^ _dimm_dpe_out_253[0] ^ _dimm_dpe_out_254[0] ^ _dimm_dpe_out_255[0] ^ _dimm_dpe_out_256[0] ^ _dimm_dpe_out_257[0] ^ _dimm_dpe_out_258[0] ^ _dimm_dpe_out_259[0] ^ _dimm_dpe_out_260[0] ^ _dimm_dpe_out_261[0] ^ _dimm_dpe_out_262[0] ^ _dimm_dpe_out_263[0] ^ _dimm_dpe_out_264[0] ^ _dimm_dpe_out_265[0] ^ _dimm_dpe_out_266[0] ^ _dimm_dpe_out_267[0] ^ _dimm_dpe_out_268[0] ^ _dimm_dpe_out_269[0] ^ _dimm_dpe_out_270[0] ^ _dimm_dpe_out_271[0] ^ _dimm_dpe_out_272[0] ^ _dimm_dpe_out_273[0] ^ _dimm_dpe_out_274[0] ^ _dimm_dpe_out_275[0] ^ _dimm_dpe_out_276[0] ^ _dimm_dpe_out_277[0] ^ _dimm_dpe_out_278[0] ^ _dimm_dpe_out_279[0] ^ _dimm_dpe_out_280[0] ^ _dimm_dpe_out_281[0] ^ _dimm_dpe_out_282[0] ^ _dimm_dpe_out_283[0] ^ _dimm_dpe_out_284[0] ^ _dimm_dpe_out_285[0] ^ _dimm_dpe_out_286[0] ^ _dimm_dpe_out_287[0] ^ _dimm_dpe_out_288[0] ^ _dimm_dpe_out_289[0] ^ _dimm_dpe_out_290[0] ^ _dimm_dpe_out_291[0] ^ _dimm_dpe_out_292[0] ^ _dimm_dpe_out_293[0] ^ _dimm_dpe_out_294[0] ^ _dimm_dpe_out_295[0] ^ _dimm_dpe_out_296[0] ^ _dimm_dpe_out_297[0] ^ _dimm_dpe_out_298[0] ^ _dimm_dpe_out_299[0] ^ _dimm_dpe_out_300[0] ^ _dimm_dpe_out_301[0] ^ _dimm_dpe_out_302[0] ^ _dimm_dpe_out_303[0] ^ _dimm_dpe_out_304[0] ^ _dimm_dpe_out_305[0] ^ _dimm_dpe_out_306[0] ^ _dimm_dpe_out_307[0] ^ _dimm_dpe_out_308[0] ^ _dimm_dpe_out_309[0] ^ _dimm_dpe_out_310[0] ^ _dimm_dpe_out_311[0] ^ _dimm_dpe_out_312[0] ^ _dimm_dpe_out_313[0] ^ _dimm_dpe_out_314[0] ^ _dimm_dpe_out_315[0] ^ _dimm_dpe_out_316[0] ^ _dimm_dpe_out_317[0] ^ _dimm_dpe_out_318[0] ^ _dimm_dpe_out_319[0] ^ _dimm_dpe_out_320[0] ^ _dimm_dpe_out_321[0] ^ _dimm_dpe_out_322[0] ^ _dimm_dpe_out_323[0] ^ _dimm_dpe_out_324[0] ^ _dimm_dpe_out_325[0] ^ _dimm_dpe_out_326[0] ^ _dimm_dpe_out_327[0] ^ _dimm_dpe_out_328[0] ^ _dimm_dpe_out_329[0] ^ _dimm_dpe_out_330[0] ^ _dimm_dpe_out_331[0] ^ _dimm_dpe_out_332[0] ^ _dimm_dpe_out_333[0] ^ _dimm_dpe_out_334[0] ^ _dimm_dpe_out_335[0] ^ _dimm_dpe_out_336[0] ^ _dimm_dpe_out_337[0] ^ _dimm_dpe_out_338[0] ^ _dimm_dpe_out_339[0] ^ _dimm_dpe_out_340[0] ^ _dimm_dpe_out_341[0] ^ _dimm_dpe_out_342[0] ^ _dimm_dpe_out_343[0] ^ _dimm_dpe_out_344[0] ^ _dimm_dpe_out_345[0] ^ _dimm_dpe_out_346[0] ^ _dimm_dpe_out_347[0] ^ _dimm_dpe_out_348[0] ^ _dimm_dpe_out_349[0] ^ _dimm_dpe_out_350[0] ^ _dimm_dpe_out_351[0] ^ _dimm_dpe_out_352[0] ^ _dimm_dpe_out_353[0] ^ _dimm_dpe_out_354[0] ^ _dimm_dpe_out_355[0] ^ _dimm_dpe_out_356[0] ^ _dimm_dpe_out_357[0] ^ _dimm_dpe_out_358[0] ^ _dimm_dpe_out_359[0] ^ _dimm_dpe_out_360[0] ^ _dimm_dpe_out_361[0] ^ _dimm_dpe_out_362[0] ^ _dimm_dpe_out_363[0] ^ _dimm_dpe_out_364[0] ^ _dimm_dpe_out_365[0] ^ _dimm_dpe_out_366[0] ^ _dimm_dpe_out_367[0] ^ _dimm_dpe_out_368[0] ^ _dimm_dpe_out_369[0] ^ _dimm_dpe_out_370[0] ^ _dimm_dpe_out_371[0] ^ _dimm_dpe_out_372[0] ^ _dimm_dpe_out_373[0] ^ _dimm_dpe_out_374[0] ^ _dimm_dpe_out_375[0] ^ _dimm_dpe_out_376[0] ^ _dimm_dpe_out_377[0] ^ _dimm_dpe_out_378[0] ^ _dimm_dpe_out_379[0] ^ _dimm_dpe_out_380[0] ^ _dimm_dpe_out_381[0] ^ _dimm_dpe_out_382[0] ^ _dimm_dpe_out_383[0]};
endmodule

// FFN1 block 0: V=1 H=2
module ffn1_layer_b0 #(
    parameter DATA_WIDTH = 40
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    localparam V = 1;
    localparam H = 2;
    localparam DEPTH = 512;
    localparam ADDR_WIDTH = 9;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [1-1:0] reg_full_sig;
    wire [2-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [1-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [1-1:0] dpe_sel;
    wire [1-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(512)
    ) sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_out)
    );

    // Controller
    controller_scalable #(
        .N_CHANNELS(1),
        .N_BRAM_R(1),
        .N_BRAM_W(1),
        .N_DPE_V(1),
        .N_DPE_H(2),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1)
    ) ctrl_inst (
        .clk(clk),
        .rst(rst),
        .MSB_SA_Ready(MSB_SA_Ready),
        .valid(valid),
        .ready_n(ready_n),
        .dpe_done(dpe_done),
        .reg_full(reg_full_sig),
        .reg_empty(reg_empty),
        .shift_add_done(shift_add_done),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl),
        .dpe_accum_done(dpe_accum_done),
        .read_address(read_address),
        .write_address(write_address),
        .w_en_dec(w_en),
        .w_buf_en(w_buf_en),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .dpe_sel(dpe_sel),
        .dpe_sel_h(dpe_sel_h),
        .ready(ready),
        .valid_n(valid_n),
        .dpe_accum_ready(dpe_accum_ready)
    );

    wire [DATA_WIDTH-1:0] dpe_out_c0_r0;
    wire dpe_done_c0_r0;
    wire reg_full_c0_r0;
    wire shift_add_done_c0_r0;
    wire shift_add_bypass_ctrl_c0_r0;
    wire MSB_SA_Ready_c0_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r0;
    wire dpe_done_c1_r0;
    wire reg_full_c1_r0;
    wire shift_add_done_c1_r0;
    wire shift_add_bypass_ctrl_c1_r0;
    wire MSB_SA_Ready_c1_r0;

    // DPE instantiations: 1 vertical x 2 horizontal = 2 DPEs
    dpe dpe_c0_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r0),
        .data_out(dpe_out_c0_r0),
        .dpe_done(dpe_done_c0_r0),
        .reg_full(reg_full_c0_r0),
        .shift_add_done(shift_add_done_c0_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r0)
    );

    dpe dpe_c1_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r0),
        .data_out(dpe_out_c1_r0),
        .dpe_done(dpe_done_c1_r0),
        .reg_full(reg_full_c1_r0),
        .shift_add_done(shift_add_done_c1_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r0)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c1_r0;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c1_r0;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c1_r0;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c1_r0;
    assign reg_full_sig = {reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Horizontal output mux (2 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            1'd0: data_out_mux = dpe_out_c0_r0;
            1'd1: data_out_mux = dpe_out_c1_r0;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

// FFN1 block 1: V=1 H=2
module ffn1_layer_b1 #(
    parameter DATA_WIDTH = 40
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    localparam V = 1;
    localparam H = 2;
    localparam DEPTH = 512;
    localparam ADDR_WIDTH = 9;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [1-1:0] reg_full_sig;
    wire [2-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [1-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [1-1:0] dpe_sel;
    wire [1-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(512)
    ) sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_out)
    );

    // Controller
    controller_scalable #(
        .N_CHANNELS(1),
        .N_BRAM_R(1),
        .N_BRAM_W(1),
        .N_DPE_V(1),
        .N_DPE_H(2),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1)
    ) ctrl_inst (
        .clk(clk),
        .rst(rst),
        .MSB_SA_Ready(MSB_SA_Ready),
        .valid(valid),
        .ready_n(ready_n),
        .dpe_done(dpe_done),
        .reg_full(reg_full_sig),
        .reg_empty(reg_empty),
        .shift_add_done(shift_add_done),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl),
        .dpe_accum_done(dpe_accum_done),
        .read_address(read_address),
        .write_address(write_address),
        .w_en_dec(w_en),
        .w_buf_en(w_buf_en),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .dpe_sel(dpe_sel),
        .dpe_sel_h(dpe_sel_h),
        .ready(ready),
        .valid_n(valid_n),
        .dpe_accum_ready(dpe_accum_ready)
    );

    wire [DATA_WIDTH-1:0] dpe_out_c0_r0;
    wire dpe_done_c0_r0;
    wire reg_full_c0_r0;
    wire shift_add_done_c0_r0;
    wire shift_add_bypass_ctrl_c0_r0;
    wire MSB_SA_Ready_c0_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r0;
    wire dpe_done_c1_r0;
    wire reg_full_c1_r0;
    wire shift_add_done_c1_r0;
    wire shift_add_bypass_ctrl_c1_r0;
    wire MSB_SA_Ready_c1_r0;

    // DPE instantiations: 1 vertical x 2 horizontal = 2 DPEs
    dpe dpe_c0_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r0),
        .data_out(dpe_out_c0_r0),
        .dpe_done(dpe_done_c0_r0),
        .reg_full(reg_full_c0_r0),
        .shift_add_done(shift_add_done_c0_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r0)
    );

    dpe dpe_c1_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r0),
        .data_out(dpe_out_c1_r0),
        .dpe_done(dpe_done_c1_r0),
        .reg_full(reg_full_c1_r0),
        .shift_add_done(shift_add_done_c1_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r0)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c1_r0;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c1_r0;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c1_r0;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c1_r0;
    assign reg_full_sig = {reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Horizontal output mux (2 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            1'd0: data_out_mux = dpe_out_c0_r0;
            1'd1: data_out_mux = dpe_out_c1_r0;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

// DIMM — Block 0, Head 0, Lane 0 (depth=393216)
module dimm_score_matrix_b0_h0 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid_q, input wire valid_k,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_q,
    input wire [DATA_WIDTH-1:0] data_in_k,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_q, output wire ready_k,
    output wire valid_n
);

    reg [ADDR_WIDTH-1:0] q_write_addr, q_read_addr;
    reg q_w_en;
    wire [DATA_WIDTH-1:0] q_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    q_sram (.clk(clk),.rst(rst),.w_en(q_w_en),.r_addr(q_read_addr),
            .w_addr(q_write_addr),.sram_data_in(data_in_q),.sram_data_out(q_sram_out));

    reg [$clog2(N*d)-1:0] k_write_addr, k_read_addr_a;
    reg k_w_en;
    wire [DATA_WIDTH-1:0] k_sram_out_a;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    k_sram_a (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_a[$clog2(N*d)-1:0]),
              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_a));
    reg [$clog2(N*d)-1:0] k_read_addr_b;
    wire [DATA_WIDTH-1:0] k_sram_out_b;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    k_sram_b (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_b[$clog2(N*d)-1:0]),
              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_b));

    // CLB adder A: log_Q + log_K[j₁] (log-domain addition)
    wire [DATA_WIDTH-1:0] log_sum_a = q_sram_out + k_sram_out_a;
    // CLB adder B: log_Q + log_K[j₂] (second vector for dual-identity)
    wire [DATA_WIDTH-1:0] log_sum_b = q_sram_out + k_sram_out_b;

    // DPE(I|exp) stage_0: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_0_valid, dimm_exp_0_ready_n, dimm_exp_0_ready, dimm_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_0_data_in, dimm_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_0_valid),
        .ready_n(dimm_exp_0_ready_n),
        .data_in(dimm_exp_0_data_in),
        .data_out(dimm_exp_0_data_out),
        .ready(dimm_exp_0_ready),
        .valid_n(dimm_exp_0_valid_n)
    );

    // DPE(I|exp) stage_1: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_1_valid, dimm_exp_1_ready_n, dimm_exp_1_ready, dimm_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_1_data_in, dimm_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_1_valid),
        .ready_n(dimm_exp_1_ready_n),
        .data_in(dimm_exp_1_data_in),
        .data_out(dimm_exp_1_data_out),
        .ready(dimm_exp_1_ready),
        .valid_n(dimm_exp_1_valid_n)
    );

    // DPE(I|exp) stage_2: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_2_valid, dimm_exp_2_ready_n, dimm_exp_2_ready, dimm_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_2_data_in, dimm_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_2_valid),
        .ready_n(dimm_exp_2_ready_n),
        .data_in(dimm_exp_2_data_in),
        .data_out(dimm_exp_2_data_out),
        .ready(dimm_exp_2_ready),
        .valid_n(dimm_exp_2_valid_n)
    );

    // DPE(I|exp) stage_3: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_3_valid, dimm_exp_3_ready_n, dimm_exp_3_ready, dimm_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_3_data_in, dimm_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_3_valid),
        .ready_n(dimm_exp_3_ready_n),
        .data_in(dimm_exp_3_data_in),
        .data_out(dimm_exp_3_data_out),
        .ready(dimm_exp_3_ready),
        .valid_n(dimm_exp_3_valid_n)
    );

    // DPE(I|exp) stage_4: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_4_valid, dimm_exp_4_ready_n, dimm_exp_4_ready, dimm_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_4_data_in, dimm_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_4_valid),
        .ready_n(dimm_exp_4_ready_n),
        .data_in(dimm_exp_4_data_in),
        .data_out(dimm_exp_4_data_out),
        .ready(dimm_exp_4_ready),
        .valid_n(dimm_exp_4_valid_n)
    );

    // DPE(I|exp) stage_5: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_5_valid, dimm_exp_5_ready_n, dimm_exp_5_ready, dimm_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_5_data_in, dimm_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_5_valid),
        .ready_n(dimm_exp_5_ready_n),
        .data_in(dimm_exp_5_data_in),
        .data_out(dimm_exp_5_data_out),
        .ready(dimm_exp_5_ready),
        .valid_n(dimm_exp_5_valid_n)
    );

    // DPE(I|exp) stage_6: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_6_valid, dimm_exp_6_ready_n, dimm_exp_6_ready, dimm_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_6_data_in, dimm_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_6_valid),
        .ready_n(dimm_exp_6_ready_n),
        .data_in(dimm_exp_6_data_in),
        .data_out(dimm_exp_6_data_out),
        .ready(dimm_exp_6_ready),
        .valid_n(dimm_exp_6_valid_n)
    );

    // DPE(I|exp) stage_7: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_7_valid, dimm_exp_7_ready_n, dimm_exp_7_ready, dimm_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_7_data_in, dimm_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_7_valid),
        .ready_n(dimm_exp_7_ready_n),
        .data_in(dimm_exp_7_data_in),
        .data_out(dimm_exp_7_data_out),
        .ready(dimm_exp_7_ready),
        .valid_n(dimm_exp_7_valid_n)
    );

    // DPE(I|exp) stage_8: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_8_valid, dimm_exp_8_ready_n, dimm_exp_8_ready, dimm_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_8_data_in, dimm_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_8_valid),
        .ready_n(dimm_exp_8_ready_n),
        .data_in(dimm_exp_8_data_in),
        .data_out(dimm_exp_8_data_out),
        .ready(dimm_exp_8_ready),
        .valid_n(dimm_exp_8_valid_n)
    );

    // DPE(I|exp) stage_9: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_9_valid, dimm_exp_9_ready_n, dimm_exp_9_ready, dimm_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_9_data_in, dimm_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_9_valid),
        .ready_n(dimm_exp_9_ready_n),
        .data_in(dimm_exp_9_data_in),
        .data_out(dimm_exp_9_data_out),
        .ready(dimm_exp_9_ready),
        .valid_n(dimm_exp_9_valid_n)
    );

    // DPE(I|exp) stage_10: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_10_valid, dimm_exp_10_ready_n, dimm_exp_10_ready, dimm_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_10_data_in, dimm_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_10_valid),
        .ready_n(dimm_exp_10_ready_n),
        .data_in(dimm_exp_10_data_in),
        .data_out(dimm_exp_10_data_out),
        .ready(dimm_exp_10_ready),
        .valid_n(dimm_exp_10_valid_n)
    );

    // DPE(I|exp) stage_11: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_11_valid, dimm_exp_11_ready_n, dimm_exp_11_ready, dimm_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_11_data_in, dimm_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_11_valid),
        .ready_n(dimm_exp_11_ready_n),
        .data_in(dimm_exp_11_data_in),
        .data_out(dimm_exp_11_data_out),
        .ready(dimm_exp_11_ready),
        .valid_n(dimm_exp_11_valid_n)
    );

    // DPE(I|exp) stage_12: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_12_valid, dimm_exp_12_ready_n, dimm_exp_12_ready, dimm_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_12_data_in, dimm_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_12_valid),
        .ready_n(dimm_exp_12_ready_n),
        .data_in(dimm_exp_12_data_in),
        .data_out(dimm_exp_12_data_out),
        .ready(dimm_exp_12_ready),
        .valid_n(dimm_exp_12_valid_n)
    );

    // DPE(I|exp) stage_13: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_13_valid, dimm_exp_13_ready_n, dimm_exp_13_ready, dimm_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_13_data_in, dimm_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_13_valid),
        .ready_n(dimm_exp_13_ready_n),
        .data_in(dimm_exp_13_data_in),
        .data_out(dimm_exp_13_data_out),
        .ready(dimm_exp_13_ready),
        .valid_n(dimm_exp_13_valid_n)
    );

    // DPE(I|exp) stage_14: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_14_valid, dimm_exp_14_ready_n, dimm_exp_14_ready, dimm_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_14_data_in, dimm_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_14_valid),
        .ready_n(dimm_exp_14_ready_n),
        .data_in(dimm_exp_14_data_in),
        .data_out(dimm_exp_14_data_out),
        .ready(dimm_exp_14_ready),
        .valid_n(dimm_exp_14_valid_n)
    );

    // DPE(I|exp) stage_15: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_15_valid, dimm_exp_15_ready_n, dimm_exp_15_ready, dimm_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_15_data_in, dimm_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_15_valid),
        .ready_n(dimm_exp_15_ready_n),
        .data_in(dimm_exp_15_data_in),
        .data_out(dimm_exp_15_data_out),
        .ready(dimm_exp_15_ready),
        .valid_n(dimm_exp_15_valid_n)
    );

    // DPE(I|exp) stage_16: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_16_valid, dimm_exp_16_ready_n, dimm_exp_16_ready, dimm_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_16_data_in, dimm_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_16_valid),
        .ready_n(dimm_exp_16_ready_n),
        .data_in(dimm_exp_16_data_in),
        .data_out(dimm_exp_16_data_out),
        .ready(dimm_exp_16_ready),
        .valid_n(dimm_exp_16_valid_n)
    );

    // DPE(I|exp) stage_17: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_17_valid, dimm_exp_17_ready_n, dimm_exp_17_ready, dimm_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_17_data_in, dimm_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_17_valid),
        .ready_n(dimm_exp_17_ready_n),
        .data_in(dimm_exp_17_data_in),
        .data_out(dimm_exp_17_data_out),
        .ready(dimm_exp_17_ready),
        .valid_n(dimm_exp_17_valid_n)
    );

    // DPE(I|exp) stage_18: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_18_valid, dimm_exp_18_ready_n, dimm_exp_18_ready, dimm_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_18_data_in, dimm_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_18_valid),
        .ready_n(dimm_exp_18_ready_n),
        .data_in(dimm_exp_18_data_in),
        .data_out(dimm_exp_18_data_out),
        .ready(dimm_exp_18_ready),
        .valid_n(dimm_exp_18_valid_n)
    );

    // DPE(I|exp) stage_19: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_19_valid, dimm_exp_19_ready_n, dimm_exp_19_ready, dimm_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_19_data_in, dimm_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_19_valid),
        .ready_n(dimm_exp_19_ready_n),
        .data_in(dimm_exp_19_data_in),
        .data_out(dimm_exp_19_data_out),
        .ready(dimm_exp_19_ready),
        .valid_n(dimm_exp_19_valid_n)
    );

    // DPE(I|exp) stage_20: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_20_valid, dimm_exp_20_ready_n, dimm_exp_20_ready, dimm_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_20_data_in, dimm_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_20_valid),
        .ready_n(dimm_exp_20_ready_n),
        .data_in(dimm_exp_20_data_in),
        .data_out(dimm_exp_20_data_out),
        .ready(dimm_exp_20_ready),
        .valid_n(dimm_exp_20_valid_n)
    );

    // DPE(I|exp) stage_21: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_21_valid, dimm_exp_21_ready_n, dimm_exp_21_ready, dimm_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_21_data_in, dimm_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_21_valid),
        .ready_n(dimm_exp_21_ready_n),
        .data_in(dimm_exp_21_data_in),
        .data_out(dimm_exp_21_data_out),
        .ready(dimm_exp_21_ready),
        .valid_n(dimm_exp_21_valid_n)
    );

    // DPE(I|exp) stage_22: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_22_valid, dimm_exp_22_ready_n, dimm_exp_22_ready, dimm_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_22_data_in, dimm_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_22_valid),
        .ready_n(dimm_exp_22_ready_n),
        .data_in(dimm_exp_22_data_in),
        .data_out(dimm_exp_22_data_out),
        .ready(dimm_exp_22_ready),
        .valid_n(dimm_exp_22_valid_n)
    );

    // DPE(I|exp) stage_23: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_23_valid, dimm_exp_23_ready_n, dimm_exp_23_ready, dimm_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_23_data_in, dimm_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_23_valid),
        .ready_n(dimm_exp_23_ready_n),
        .data_in(dimm_exp_23_data_in),
        .data_out(dimm_exp_23_data_out),
        .ready(dimm_exp_23_ready),
        .valid_n(dimm_exp_23_valid_n)
    );

    reg [4:0] dimm_sel;
    assign dimm_exp_0_data_in = log_sum_a;
    assign dimm_exp_0_valid = (state == S_COMPUTE) && (dimm_sel == 0);
    assign dimm_exp_0_ready_n = 1'b0;
    assign dimm_exp_1_data_in = log_sum_a;
    assign dimm_exp_1_valid = (state == S_COMPUTE) && (dimm_sel == 1);
    assign dimm_exp_1_ready_n = 1'b0;
    assign dimm_exp_2_data_in = log_sum_a;
    assign dimm_exp_2_valid = (state == S_COMPUTE) && (dimm_sel == 2);
    assign dimm_exp_2_ready_n = 1'b0;
    assign dimm_exp_3_data_in = log_sum_a;
    assign dimm_exp_3_valid = (state == S_COMPUTE) && (dimm_sel == 3);
    assign dimm_exp_3_ready_n = 1'b0;
    assign dimm_exp_4_data_in = log_sum_a;
    assign dimm_exp_4_valid = (state == S_COMPUTE) && (dimm_sel == 4);
    assign dimm_exp_4_ready_n = 1'b0;
    assign dimm_exp_5_data_in = log_sum_a;
    assign dimm_exp_5_valid = (state == S_COMPUTE) && (dimm_sel == 5);
    assign dimm_exp_5_ready_n = 1'b0;
    assign dimm_exp_6_data_in = log_sum_a;
    assign dimm_exp_6_valid = (state == S_COMPUTE) && (dimm_sel == 6);
    assign dimm_exp_6_ready_n = 1'b0;
    assign dimm_exp_7_data_in = log_sum_a;
    assign dimm_exp_7_valid = (state == S_COMPUTE) && (dimm_sel == 7);
    assign dimm_exp_7_ready_n = 1'b0;
    assign dimm_exp_8_data_in = log_sum_a;
    assign dimm_exp_8_valid = (state == S_COMPUTE) && (dimm_sel == 8);
    assign dimm_exp_8_ready_n = 1'b0;
    assign dimm_exp_9_data_in = log_sum_a;
    assign dimm_exp_9_valid = (state == S_COMPUTE) && (dimm_sel == 9);
    assign dimm_exp_9_ready_n = 1'b0;
    assign dimm_exp_10_data_in = log_sum_a;
    assign dimm_exp_10_valid = (state == S_COMPUTE) && (dimm_sel == 10);
    assign dimm_exp_10_ready_n = 1'b0;
    assign dimm_exp_11_data_in = log_sum_a;
    assign dimm_exp_11_valid = (state == S_COMPUTE) && (dimm_sel == 11);
    assign dimm_exp_11_ready_n = 1'b0;
    assign dimm_exp_12_data_in = log_sum_a;
    assign dimm_exp_12_valid = (state == S_COMPUTE) && (dimm_sel == 12);
    assign dimm_exp_12_ready_n = 1'b0;
    assign dimm_exp_13_data_in = log_sum_a;
    assign dimm_exp_13_valid = (state == S_COMPUTE) && (dimm_sel == 13);
    assign dimm_exp_13_ready_n = 1'b0;
    assign dimm_exp_14_data_in = log_sum_a;
    assign dimm_exp_14_valid = (state == S_COMPUTE) && (dimm_sel == 14);
    assign dimm_exp_14_ready_n = 1'b0;
    assign dimm_exp_15_data_in = log_sum_a;
    assign dimm_exp_15_valid = (state == S_COMPUTE) && (dimm_sel == 15);
    assign dimm_exp_15_ready_n = 1'b0;
    assign dimm_exp_16_data_in = log_sum_a;
    assign dimm_exp_16_valid = (state == S_COMPUTE) && (dimm_sel == 16);
    assign dimm_exp_16_ready_n = 1'b0;
    assign dimm_exp_17_data_in = log_sum_a;
    assign dimm_exp_17_valid = (state == S_COMPUTE) && (dimm_sel == 17);
    assign dimm_exp_17_ready_n = 1'b0;
    assign dimm_exp_18_data_in = log_sum_a;
    assign dimm_exp_18_valid = (state == S_COMPUTE) && (dimm_sel == 18);
    assign dimm_exp_18_ready_n = 1'b0;
    assign dimm_exp_19_data_in = log_sum_a;
    assign dimm_exp_19_valid = (state == S_COMPUTE) && (dimm_sel == 19);
    assign dimm_exp_19_ready_n = 1'b0;
    assign dimm_exp_20_data_in = log_sum_a;
    assign dimm_exp_20_valid = (state == S_COMPUTE) && (dimm_sel == 20);
    assign dimm_exp_20_ready_n = 1'b0;
    assign dimm_exp_21_data_in = log_sum_a;
    assign dimm_exp_21_valid = (state == S_COMPUTE) && (dimm_sel == 21);
    assign dimm_exp_21_ready_n = 1'b0;
    assign dimm_exp_22_data_in = log_sum_a;
    assign dimm_exp_22_valid = (state == S_COMPUTE) && (dimm_sel == 22);
    assign dimm_exp_22_ready_n = 1'b0;
    assign dimm_exp_23_data_in = log_sum_a;
    assign dimm_exp_23_valid = (state == S_COMPUTE) && (dimm_sel == 23);
    assign dimm_exp_23_ready_n = 1'b0;

    // Dual accumulators: one per vector (A for j₁, B for j₂)
    reg [2*DATA_WIDTH-1:0] accumulator_a, accumulator_b;
    reg acc_phase;  // 0=accumulating A outputs, 1=accumulating B outputs
    always @(posedge clk) begin
        if (rst) begin accumulator_a <= 0; accumulator_b <= 0; end
        else if (dimm_exp_valid_n) begin
            if (!acc_phase) accumulator_a <= accumulator_a + dimm_exp_data_out;
            else accumulator_b <= accumulator_b + dimm_exp_data_out;
        end
    end

    reg [ADDR_WIDTH-1:0] score_write_addr, score_read_addr;
    reg score_w_en;
    reg [DATA_WIDTH-1:0] score_write_data;
    wire [DATA_WIDTH-1:0] score_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    score_sram (.clk(clk),.rst(rst),.w_en(score_w_en),.r_addr(score_read_addr),
                .w_addr(score_write_addr),.sram_data_in(score_write_data),.sram_data_out(score_sram_out));

    localparam S_IDLE = 3'd0, S_LOAD_Q = 3'd1, S_LOAD_K = 3'd2,
               S_COMPUTE = 3'd3, S_OUTPUT = 3'd4;
    localparam S_WRITE_B = 3'd5;  // write second score for dual-identity
    reg [2:0] state;
    reg [$clog2(d)-1:0] mac_count;
    reg [$clog2(N)-1:0] score_idx;
    reg feed_half;  // 0=feeding vector A (d elements), 1=feeding vector B

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            q_write_addr <= 0; q_read_addr <= 0; q_w_en <= 0;
            k_write_addr <= 0; k_read_addr_a <= 0; k_w_en <= 0;
            k_read_addr_b <= 0;
            feed_half <= 0; feed_phase <= 0; acc_phase <= 0;
            score_write_addr <= 0; score_read_addr <= 0; score_w_en <= 0;
            score_write_data <= 0;
            mac_count <= 0; score_idx <= 0;
            dimm_sel <= 0;
        end else begin
            q_w_en <= 0; k_w_en <= 0; score_w_en <= 0;
            case (state)
                S_IDLE: if (valid_q || valid_k) state <= S_LOAD_Q;
                S_LOAD_Q: begin
                    if (valid_q) begin q_w_en <= 1; q_write_addr <= q_write_addr + 1; end
                    if (q_write_addr == d-1) state <= S_LOAD_K;
                end
                S_LOAD_K: begin
                    if (valid_k) begin k_w_en <= 1; k_write_addr <= k_write_addr + 1; end
                    if (k_write_addr == N*d-1) state <= S_COMPUTE;
                end
                S_COMPUTE: begin
                    // Dual-identity: feed vector A then vector B per DPE pass
                    q_read_addr <= mac_count;
                    k_read_addr_a <= (score_idx << $clog2(d)) + mac_count;
                    k_read_addr_b <= ((score_idx + 1) << $clog2(d)) + mac_count;
                    if (!feed_half) begin
                        // Feeding vector A (log_sum_a)
                        feed_phase <= 0; acc_phase <= 0;
                        mac_count <= mac_count + 1;
                        if (mac_count == d-1) begin
                            mac_count <= 0;
                            feed_half <= 1;
                        end
                    end else begin
                        // Feeding vector B (log_sum_b)
                        feed_phase <= 1; acc_phase <= 1;
                        mac_count <= mac_count + 1;
                        if (mac_count == d-1) begin
                            // Both vectors done — write score A, then B
                            score_write_data <= accumulator_a[2*DATA_WIDTH-1:DATA_WIDTH];
                            score_w_en <= 1;
                            score_write_addr <= score_idx;
                            mac_count <= 0;
                            feed_half <= 0;
                            state <= S_WRITE_B;
                        end
                    end
                end
                S_WRITE_B: begin
                    // Write second score (j₂ = score_idx + 1)
                    score_write_data <= accumulator_b[2*DATA_WIDTH-1:DATA_WIDTH];
                    score_w_en <= 1;
                    score_write_addr <= score_idx + 1;
                    accumulator_a <= 0; accumulator_b <= 0;
                    if (score_idx >= N-2) state <= S_OUTPUT;
                    else begin score_idx <= score_idx + 2; state <= S_COMPUTE; end
                end
                S_OUTPUT: if (ready_n) score_read_addr <= score_read_addr + 1;
            endcase
        end
    end

    assign data_out = score_sram_out;
    assign valid_n = (state == S_OUTPUT);
    assign ready_q = (state == S_LOAD_Q || state == S_IDLE);
    assign ready_k = (state == S_LOAD_K || state == S_IDLE);
endmodule
module softmax_approx_b0_h0 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid, input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, output wire valid_n
);

    reg [ADDR_WIDTH-1:0] in_write_addr, in_read_addr;
    reg in_w_en;
    wire [DATA_WIDTH-1:0] in_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    in_sram (.clk(clk),.rst(rst),.w_en(in_w_en),.r_addr(in_read_addr),
             .w_addr(in_write_addr),.sram_data_in(data_in),.sram_data_out(in_sram_out));

    wire sm_exp_0_valid, sm_exp_0_ready_n, sm_exp_0_ready, sm_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_0_data_in, sm_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_0_valid),
        .ready_n(sm_exp_0_ready_n),
        .data_in(sm_exp_0_data_in),
        .data_out(sm_exp_0_data_out),
        .ready(sm_exp_0_ready),
        .valid_n(sm_exp_0_valid_n)
    );

    wire sm_exp_1_valid, sm_exp_1_ready_n, sm_exp_1_ready, sm_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_1_data_in, sm_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_1_valid),
        .ready_n(sm_exp_1_ready_n),
        .data_in(sm_exp_1_data_in),
        .data_out(sm_exp_1_data_out),
        .ready(sm_exp_1_ready),
        .valid_n(sm_exp_1_valid_n)
    );

    wire sm_exp_2_valid, sm_exp_2_ready_n, sm_exp_2_ready, sm_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_2_data_in, sm_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_2_valid),
        .ready_n(sm_exp_2_ready_n),
        .data_in(sm_exp_2_data_in),
        .data_out(sm_exp_2_data_out),
        .ready(sm_exp_2_ready),
        .valid_n(sm_exp_2_valid_n)
    );

    wire sm_exp_3_valid, sm_exp_3_ready_n, sm_exp_3_ready, sm_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_3_data_in, sm_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_3_valid),
        .ready_n(sm_exp_3_ready_n),
        .data_in(sm_exp_3_data_in),
        .data_out(sm_exp_3_data_out),
        .ready(sm_exp_3_ready),
        .valid_n(sm_exp_3_valid_n)
    );

    wire sm_exp_4_valid, sm_exp_4_ready_n, sm_exp_4_ready, sm_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_4_data_in, sm_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_4_valid),
        .ready_n(sm_exp_4_ready_n),
        .data_in(sm_exp_4_data_in),
        .data_out(sm_exp_4_data_out),
        .ready(sm_exp_4_ready),
        .valid_n(sm_exp_4_valid_n)
    );

    wire sm_exp_5_valid, sm_exp_5_ready_n, sm_exp_5_ready, sm_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_5_data_in, sm_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_5_valid),
        .ready_n(sm_exp_5_ready_n),
        .data_in(sm_exp_5_data_in),
        .data_out(sm_exp_5_data_out),
        .ready(sm_exp_5_ready),
        .valid_n(sm_exp_5_valid_n)
    );

    wire sm_exp_6_valid, sm_exp_6_ready_n, sm_exp_6_ready, sm_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_6_data_in, sm_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_6_valid),
        .ready_n(sm_exp_6_ready_n),
        .data_in(sm_exp_6_data_in),
        .data_out(sm_exp_6_data_out),
        .ready(sm_exp_6_ready),
        .valid_n(sm_exp_6_valid_n)
    );

    wire sm_exp_7_valid, sm_exp_7_ready_n, sm_exp_7_ready, sm_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_7_data_in, sm_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_7_valid),
        .ready_n(sm_exp_7_ready_n),
        .data_in(sm_exp_7_data_in),
        .data_out(sm_exp_7_data_out),
        .ready(sm_exp_7_ready),
        .valid_n(sm_exp_7_valid_n)
    );

    wire sm_exp_8_valid, sm_exp_8_ready_n, sm_exp_8_ready, sm_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_8_data_in, sm_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_8_valid),
        .ready_n(sm_exp_8_ready_n),
        .data_in(sm_exp_8_data_in),
        .data_out(sm_exp_8_data_out),
        .ready(sm_exp_8_ready),
        .valid_n(sm_exp_8_valid_n)
    );

    wire sm_exp_9_valid, sm_exp_9_ready_n, sm_exp_9_ready, sm_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_9_data_in, sm_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_9_valid),
        .ready_n(sm_exp_9_ready_n),
        .data_in(sm_exp_9_data_in),
        .data_out(sm_exp_9_data_out),
        .ready(sm_exp_9_ready),
        .valid_n(sm_exp_9_valid_n)
    );

    wire sm_exp_10_valid, sm_exp_10_ready_n, sm_exp_10_ready, sm_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_10_data_in, sm_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_10_valid),
        .ready_n(sm_exp_10_ready_n),
        .data_in(sm_exp_10_data_in),
        .data_out(sm_exp_10_data_out),
        .ready(sm_exp_10_ready),
        .valid_n(sm_exp_10_valid_n)
    );

    wire sm_exp_11_valid, sm_exp_11_ready_n, sm_exp_11_ready, sm_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_11_data_in, sm_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_11_valid),
        .ready_n(sm_exp_11_ready_n),
        .data_in(sm_exp_11_data_in),
        .data_out(sm_exp_11_data_out),
        .ready(sm_exp_11_ready),
        .valid_n(sm_exp_11_valid_n)
    );

    wire sm_exp_12_valid, sm_exp_12_ready_n, sm_exp_12_ready, sm_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_12_data_in, sm_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_12_valid),
        .ready_n(sm_exp_12_ready_n),
        .data_in(sm_exp_12_data_in),
        .data_out(sm_exp_12_data_out),
        .ready(sm_exp_12_ready),
        .valid_n(sm_exp_12_valid_n)
    );

    wire sm_exp_13_valid, sm_exp_13_ready_n, sm_exp_13_ready, sm_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_13_data_in, sm_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_13_valid),
        .ready_n(sm_exp_13_ready_n),
        .data_in(sm_exp_13_data_in),
        .data_out(sm_exp_13_data_out),
        .ready(sm_exp_13_ready),
        .valid_n(sm_exp_13_valid_n)
    );

    wire sm_exp_14_valid, sm_exp_14_ready_n, sm_exp_14_ready, sm_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_14_data_in, sm_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_14_valid),
        .ready_n(sm_exp_14_ready_n),
        .data_in(sm_exp_14_data_in),
        .data_out(sm_exp_14_data_out),
        .ready(sm_exp_14_ready),
        .valid_n(sm_exp_14_valid_n)
    );

    wire sm_exp_15_valid, sm_exp_15_ready_n, sm_exp_15_ready, sm_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_15_data_in, sm_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_15_valid),
        .ready_n(sm_exp_15_ready_n),
        .data_in(sm_exp_15_data_in),
        .data_out(sm_exp_15_data_out),
        .ready(sm_exp_15_ready),
        .valid_n(sm_exp_15_valid_n)
    );

    wire sm_exp_16_valid, sm_exp_16_ready_n, sm_exp_16_ready, sm_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_16_data_in, sm_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_16_valid),
        .ready_n(sm_exp_16_ready_n),
        .data_in(sm_exp_16_data_in),
        .data_out(sm_exp_16_data_out),
        .ready(sm_exp_16_ready),
        .valid_n(sm_exp_16_valid_n)
    );

    wire sm_exp_17_valid, sm_exp_17_ready_n, sm_exp_17_ready, sm_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_17_data_in, sm_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_17_valid),
        .ready_n(sm_exp_17_ready_n),
        .data_in(sm_exp_17_data_in),
        .data_out(sm_exp_17_data_out),
        .ready(sm_exp_17_ready),
        .valid_n(sm_exp_17_valid_n)
    );

    wire sm_exp_18_valid, sm_exp_18_ready_n, sm_exp_18_ready, sm_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_18_data_in, sm_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_18_valid),
        .ready_n(sm_exp_18_ready_n),
        .data_in(sm_exp_18_data_in),
        .data_out(sm_exp_18_data_out),
        .ready(sm_exp_18_ready),
        .valid_n(sm_exp_18_valid_n)
    );

    wire sm_exp_19_valid, sm_exp_19_ready_n, sm_exp_19_ready, sm_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_19_data_in, sm_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_19_valid),
        .ready_n(sm_exp_19_ready_n),
        .data_in(sm_exp_19_data_in),
        .data_out(sm_exp_19_data_out),
        .ready(sm_exp_19_ready),
        .valid_n(sm_exp_19_valid_n)
    );

    wire sm_exp_20_valid, sm_exp_20_ready_n, sm_exp_20_ready, sm_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_20_data_in, sm_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_20_valid),
        .ready_n(sm_exp_20_ready_n),
        .data_in(sm_exp_20_data_in),
        .data_out(sm_exp_20_data_out),
        .ready(sm_exp_20_ready),
        .valid_n(sm_exp_20_valid_n)
    );

    wire sm_exp_21_valid, sm_exp_21_ready_n, sm_exp_21_ready, sm_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_21_data_in, sm_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_21_valid),
        .ready_n(sm_exp_21_ready_n),
        .data_in(sm_exp_21_data_in),
        .data_out(sm_exp_21_data_out),
        .ready(sm_exp_21_ready),
        .valid_n(sm_exp_21_valid_n)
    );

    wire sm_exp_22_valid, sm_exp_22_ready_n, sm_exp_22_ready, sm_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_22_data_in, sm_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_22_valid),
        .ready_n(sm_exp_22_ready_n),
        .data_in(sm_exp_22_data_in),
        .data_out(sm_exp_22_data_out),
        .ready(sm_exp_22_ready),
        .valid_n(sm_exp_22_valid_n)
    );

    wire sm_exp_23_valid, sm_exp_23_ready_n, sm_exp_23_ready, sm_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_23_data_in, sm_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_23_valid),
        .ready_n(sm_exp_23_ready_n),
        .data_in(sm_exp_23_data_in),
        .data_out(sm_exp_23_data_out),
        .ready(sm_exp_23_ready),
        .valid_n(sm_exp_23_valid_n)
    );

    assign sm_exp_0_data_in = in_sram_out;
    assign sm_exp_0_valid = (sm_state == SM_EXP);
    assign sm_exp_0_ready_n = 1'b0;
    assign sm_exp_1_data_in = in_sram_out;
    assign sm_exp_1_valid = (sm_state == SM_EXP);
    assign sm_exp_1_ready_n = 1'b0;
    assign sm_exp_2_data_in = in_sram_out;
    assign sm_exp_2_valid = (sm_state == SM_EXP);
    assign sm_exp_2_ready_n = 1'b0;
    assign sm_exp_3_data_in = in_sram_out;
    assign sm_exp_3_valid = (sm_state == SM_EXP);
    assign sm_exp_3_ready_n = 1'b0;
    assign sm_exp_4_data_in = in_sram_out;
    assign sm_exp_4_valid = (sm_state == SM_EXP);
    assign sm_exp_4_ready_n = 1'b0;
    assign sm_exp_5_data_in = in_sram_out;
    assign sm_exp_5_valid = (sm_state == SM_EXP);
    assign sm_exp_5_ready_n = 1'b0;
    assign sm_exp_6_data_in = in_sram_out;
    assign sm_exp_6_valid = (sm_state == SM_EXP);
    assign sm_exp_6_ready_n = 1'b0;
    assign sm_exp_7_data_in = in_sram_out;
    assign sm_exp_7_valid = (sm_state == SM_EXP);
    assign sm_exp_7_ready_n = 1'b0;
    assign sm_exp_8_data_in = in_sram_out;
    assign sm_exp_8_valid = (sm_state == SM_EXP);
    assign sm_exp_8_ready_n = 1'b0;
    assign sm_exp_9_data_in = in_sram_out;
    assign sm_exp_9_valid = (sm_state == SM_EXP);
    assign sm_exp_9_ready_n = 1'b0;
    assign sm_exp_10_data_in = in_sram_out;
    assign sm_exp_10_valid = (sm_state == SM_EXP);
    assign sm_exp_10_ready_n = 1'b0;
    assign sm_exp_11_data_in = in_sram_out;
    assign sm_exp_11_valid = (sm_state == SM_EXP);
    assign sm_exp_11_ready_n = 1'b0;
    assign sm_exp_12_data_in = in_sram_out;
    assign sm_exp_12_valid = (sm_state == SM_EXP);
    assign sm_exp_12_ready_n = 1'b0;
    assign sm_exp_13_data_in = in_sram_out;
    assign sm_exp_13_valid = (sm_state == SM_EXP);
    assign sm_exp_13_ready_n = 1'b0;
    assign sm_exp_14_data_in = in_sram_out;
    assign sm_exp_14_valid = (sm_state == SM_EXP);
    assign sm_exp_14_ready_n = 1'b0;
    assign sm_exp_15_data_in = in_sram_out;
    assign sm_exp_15_valid = (sm_state == SM_EXP);
    assign sm_exp_15_ready_n = 1'b0;
    assign sm_exp_16_data_in = in_sram_out;
    assign sm_exp_16_valid = (sm_state == SM_EXP);
    assign sm_exp_16_ready_n = 1'b0;
    assign sm_exp_17_data_in = in_sram_out;
    assign sm_exp_17_valid = (sm_state == SM_EXP);
    assign sm_exp_17_ready_n = 1'b0;
    assign sm_exp_18_data_in = in_sram_out;
    assign sm_exp_18_valid = (sm_state == SM_EXP);
    assign sm_exp_18_ready_n = 1'b0;
    assign sm_exp_19_data_in = in_sram_out;
    assign sm_exp_19_valid = (sm_state == SM_EXP);
    assign sm_exp_19_ready_n = 1'b0;
    assign sm_exp_20_data_in = in_sram_out;
    assign sm_exp_20_valid = (sm_state == SM_EXP);
    assign sm_exp_20_ready_n = 1'b0;
    assign sm_exp_21_data_in = in_sram_out;
    assign sm_exp_21_valid = (sm_state == SM_EXP);
    assign sm_exp_21_ready_n = 1'b0;
    assign sm_exp_22_data_in = in_sram_out;
    assign sm_exp_22_valid = (sm_state == SM_EXP);
    assign sm_exp_22_ready_n = 1'b0;
    assign sm_exp_23_data_in = in_sram_out;
    assign sm_exp_23_valid = (sm_state == SM_EXP);
    assign sm_exp_23_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] exp_write_addr, exp_read_addr;
    reg exp_w_en;
    reg [DATA_WIDTH-1:0] exp_write_data;
    wire [DATA_WIDTH-1:0] exp_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    exp_sram (.clk(clk),.rst(rst),.w_en(exp_w_en),.r_addr(exp_read_addr),
              .w_addr(exp_write_addr),.sram_data_in(exp_write_data),.sram_data_out(exp_sram_out));
    reg [2*DATA_WIDTH-1:0] exp_sum;

    always @(posedge clk) begin
        if (rst) begin exp_w_en <= 0; exp_write_addr <= 0; exp_sum <= 0; end
        else if (sm_exp_0_valid_n) begin
            exp_write_data <= sm_exp_0_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_0_data_out;
        end else exp_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] exp_sum_upper = exp_sum[2*DATA_WIDTH-1:DATA_WIDTH];
    reg [DATA_WIDTH-1:0] recip_val;
    always @(*) begin
        casez (exp_sum_upper[15:0])
            16'b1???????????????: recip_val = 40'd2;
            16'b01??????????????: recip_val = 40'd32768;
            16'b001?????????????: recip_val = 40'd16384;
            16'b0001????????????: recip_val = 40'd8192;
            16'b00001???????????: recip_val = 40'd4096;
            16'b000001??????????: recip_val = 40'd2048;
            16'b0000001?????????: recip_val = 40'd1024;
            16'b00000001????????: recip_val = 40'd512;
            16'b000000001???????: recip_val = 40'd256;
            16'b0000000001??????: recip_val = 40'd128;
            16'b00000000001?????: recip_val = 40'd64;
            16'b000000000001????: recip_val = 40'd32;
            16'b0000000000001???: recip_val = 40'd16;
            16'b00000000000001??: recip_val = 40'd8;
            16'b000000000000001?: recip_val = 40'd4;
            16'b0000000000000001: recip_val = 40'd2;
            default: recip_val = 40'hFFFF;
        endcase
    end

    reg [DATA_WIDTH-1:0] inv_sum;
    reg [2*DATA_WIDTH-1:0] norm_product;
    reg [DATA_WIDTH-1:0] norm_val;

    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    wire [DATA_WIDTH-1:0] out_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),
              .w_addr(out_write_addr),.sram_data_in(norm_val),.sram_data_out(out_sram_out));

    localparam SM_IDLE = 3'd0, SM_LOAD = 3'd1, SM_EXP = 3'd2,
               SM_NORMALIZE = 3'd3, SM_OUTPUT = 3'd4;
    reg [2:0] sm_state;
    reg [$clog2(N)-1:0] sm_count;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sm_state <= SM_IDLE; sm_count <= 0;
            in_write_addr <= 0; in_read_addr <= 0; in_w_en <= 0;
            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;
            inv_sum <= 0; norm_product <= 0; norm_val <= 0;
        end else begin
            in_w_en <= 0; out_w_en <= 0;
            case (sm_state)
                SM_IDLE: if (valid) sm_state <= SM_LOAD;
                SM_LOAD: begin
                    if (valid) begin in_w_en <= 1; in_write_addr <= in_write_addr + 1; end
                    if (in_write_addr == N-1) sm_state <= SM_EXP;
                end
                SM_EXP: begin
                    // DPE(I|exp) processes scores — output goes to exp_sram + sum
                    in_read_addr <= sm_count; sm_count <= sm_count + 1;
                    if (sm_count == N-1) begin
                        inv_sum <= recip_val; sm_count <= 0;
                        sm_state <= SM_NORMALIZE;
                    end
                end
                SM_NORMALIZE: begin
                    // CLB multiply: exp_val × inv_sum
                    exp_read_addr <= sm_count;
                    norm_product <= exp_sram_out * inv_sum;
                    norm_val <= norm_product[2*DATA_WIDTH-1:DATA_WIDTH];
                    out_w_en <= (sm_count > 1);
                    out_write_addr <= sm_count - 2;
                    sm_count <= sm_count + 1;
                    if (sm_count == N+1) sm_state <= SM_OUTPUT;
                end
                SM_OUTPUT: if (!ready_n) out_read_addr <= out_read_addr + 1;
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (sm_state == SM_OUTPUT);
    assign ready = (sm_state == SM_LOAD || sm_state == SM_IDLE);
endmodule
module dimm_weighted_sum_b0_h0 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid_attn, input wire valid_v,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_attn,
    input wire [DATA_WIDTH-1:0] data_in_v,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_attn, output wire ready_v,
    output wire valid_n
);

    reg [ADDR_WIDTH-1:0] attn_write_addr, attn_read_addr;
    reg attn_w_en;
    wire [DATA_WIDTH-1:0] attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    attn_sram (.clk(clk),.rst(rst),.w_en(attn_w_en),.r_addr(attn_read_addr),
               .w_addr(attn_write_addr),.sram_data_in(data_in_attn),.sram_data_out(attn_sram_out));

    reg [$clog2(N*d)-1:0] v_write_addr, v_read_addr;
    reg v_w_en;
    wire [DATA_WIDTH-1:0] v_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    v_sram (.clk(clk),.rst(rst),.w_en(v_w_en),.r_addr(v_read_addr[$clog2(N*d)-1:0]),
            .w_addr(v_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_v),.sram_data_out(v_sram_out));

    wire ws_log_0_valid, ws_log_0_ready_n, ws_log_0_ready, ws_log_0_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_0_data_in, ws_log_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_0 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_0_valid),
        .ready_n(ws_log_0_ready_n),
        .data_in(ws_log_0_data_in),
        .data_out(ws_log_0_data_out),
        .ready(ws_log_0_ready),
        .valid_n(ws_log_0_valid_n)
    );

    wire ws_log_1_valid, ws_log_1_ready_n, ws_log_1_ready, ws_log_1_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_1_data_in, ws_log_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_1 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_1_valid),
        .ready_n(ws_log_1_ready_n),
        .data_in(ws_log_1_data_in),
        .data_out(ws_log_1_data_out),
        .ready(ws_log_1_ready),
        .valid_n(ws_log_1_valid_n)
    );

    wire ws_log_2_valid, ws_log_2_ready_n, ws_log_2_ready, ws_log_2_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_2_data_in, ws_log_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_2 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_2_valid),
        .ready_n(ws_log_2_ready_n),
        .data_in(ws_log_2_data_in),
        .data_out(ws_log_2_data_out),
        .ready(ws_log_2_ready),
        .valid_n(ws_log_2_valid_n)
    );

    wire ws_log_3_valid, ws_log_3_ready_n, ws_log_3_ready, ws_log_3_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_3_data_in, ws_log_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_3 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_3_valid),
        .ready_n(ws_log_3_ready_n),
        .data_in(ws_log_3_data_in),
        .data_out(ws_log_3_data_out),
        .ready(ws_log_3_ready),
        .valid_n(ws_log_3_valid_n)
    );

    wire ws_log_4_valid, ws_log_4_ready_n, ws_log_4_ready, ws_log_4_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_4_data_in, ws_log_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_4 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_4_valid),
        .ready_n(ws_log_4_ready_n),
        .data_in(ws_log_4_data_in),
        .data_out(ws_log_4_data_out),
        .ready(ws_log_4_ready),
        .valid_n(ws_log_4_valid_n)
    );

    wire ws_log_5_valid, ws_log_5_ready_n, ws_log_5_ready, ws_log_5_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_5_data_in, ws_log_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_5 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_5_valid),
        .ready_n(ws_log_5_ready_n),
        .data_in(ws_log_5_data_in),
        .data_out(ws_log_5_data_out),
        .ready(ws_log_5_ready),
        .valid_n(ws_log_5_valid_n)
    );

    wire ws_log_6_valid, ws_log_6_ready_n, ws_log_6_ready, ws_log_6_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_6_data_in, ws_log_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_6 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_6_valid),
        .ready_n(ws_log_6_ready_n),
        .data_in(ws_log_6_data_in),
        .data_out(ws_log_6_data_out),
        .ready(ws_log_6_ready),
        .valid_n(ws_log_6_valid_n)
    );

    wire ws_log_7_valid, ws_log_7_ready_n, ws_log_7_ready, ws_log_7_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_7_data_in, ws_log_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_7 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_7_valid),
        .ready_n(ws_log_7_ready_n),
        .data_in(ws_log_7_data_in),
        .data_out(ws_log_7_data_out),
        .ready(ws_log_7_ready),
        .valid_n(ws_log_7_valid_n)
    );

    wire ws_log_8_valid, ws_log_8_ready_n, ws_log_8_ready, ws_log_8_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_8_data_in, ws_log_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_8 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_8_valid),
        .ready_n(ws_log_8_ready_n),
        .data_in(ws_log_8_data_in),
        .data_out(ws_log_8_data_out),
        .ready(ws_log_8_ready),
        .valid_n(ws_log_8_valid_n)
    );

    wire ws_log_9_valid, ws_log_9_ready_n, ws_log_9_ready, ws_log_9_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_9_data_in, ws_log_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_9 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_9_valid),
        .ready_n(ws_log_9_ready_n),
        .data_in(ws_log_9_data_in),
        .data_out(ws_log_9_data_out),
        .ready(ws_log_9_ready),
        .valid_n(ws_log_9_valid_n)
    );

    wire ws_log_10_valid, ws_log_10_ready_n, ws_log_10_ready, ws_log_10_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_10_data_in, ws_log_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_10 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_10_valid),
        .ready_n(ws_log_10_ready_n),
        .data_in(ws_log_10_data_in),
        .data_out(ws_log_10_data_out),
        .ready(ws_log_10_ready),
        .valid_n(ws_log_10_valid_n)
    );

    wire ws_log_11_valid, ws_log_11_ready_n, ws_log_11_ready, ws_log_11_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_11_data_in, ws_log_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_11 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_11_valid),
        .ready_n(ws_log_11_ready_n),
        .data_in(ws_log_11_data_in),
        .data_out(ws_log_11_data_out),
        .ready(ws_log_11_ready),
        .valid_n(ws_log_11_valid_n)
    );

    wire ws_log_12_valid, ws_log_12_ready_n, ws_log_12_ready, ws_log_12_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_12_data_in, ws_log_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_12 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_12_valid),
        .ready_n(ws_log_12_ready_n),
        .data_in(ws_log_12_data_in),
        .data_out(ws_log_12_data_out),
        .ready(ws_log_12_ready),
        .valid_n(ws_log_12_valid_n)
    );

    wire ws_log_13_valid, ws_log_13_ready_n, ws_log_13_ready, ws_log_13_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_13_data_in, ws_log_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_13 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_13_valid),
        .ready_n(ws_log_13_ready_n),
        .data_in(ws_log_13_data_in),
        .data_out(ws_log_13_data_out),
        .ready(ws_log_13_ready),
        .valid_n(ws_log_13_valid_n)
    );

    wire ws_log_14_valid, ws_log_14_ready_n, ws_log_14_ready, ws_log_14_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_14_data_in, ws_log_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_14 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_14_valid),
        .ready_n(ws_log_14_ready_n),
        .data_in(ws_log_14_data_in),
        .data_out(ws_log_14_data_out),
        .ready(ws_log_14_ready),
        .valid_n(ws_log_14_valid_n)
    );

    wire ws_log_15_valid, ws_log_15_ready_n, ws_log_15_ready, ws_log_15_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_15_data_in, ws_log_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_15 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_15_valid),
        .ready_n(ws_log_15_ready_n),
        .data_in(ws_log_15_data_in),
        .data_out(ws_log_15_data_out),
        .ready(ws_log_15_ready),
        .valid_n(ws_log_15_valid_n)
    );

    wire ws_log_16_valid, ws_log_16_ready_n, ws_log_16_ready, ws_log_16_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_16_data_in, ws_log_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_16 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_16_valid),
        .ready_n(ws_log_16_ready_n),
        .data_in(ws_log_16_data_in),
        .data_out(ws_log_16_data_out),
        .ready(ws_log_16_ready),
        .valid_n(ws_log_16_valid_n)
    );

    wire ws_log_17_valid, ws_log_17_ready_n, ws_log_17_ready, ws_log_17_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_17_data_in, ws_log_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_17 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_17_valid),
        .ready_n(ws_log_17_ready_n),
        .data_in(ws_log_17_data_in),
        .data_out(ws_log_17_data_out),
        .ready(ws_log_17_ready),
        .valid_n(ws_log_17_valid_n)
    );

    wire ws_log_18_valid, ws_log_18_ready_n, ws_log_18_ready, ws_log_18_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_18_data_in, ws_log_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_18 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_18_valid),
        .ready_n(ws_log_18_ready_n),
        .data_in(ws_log_18_data_in),
        .data_out(ws_log_18_data_out),
        .ready(ws_log_18_ready),
        .valid_n(ws_log_18_valid_n)
    );

    wire ws_log_19_valid, ws_log_19_ready_n, ws_log_19_ready, ws_log_19_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_19_data_in, ws_log_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_19 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_19_valid),
        .ready_n(ws_log_19_ready_n),
        .data_in(ws_log_19_data_in),
        .data_out(ws_log_19_data_out),
        .ready(ws_log_19_ready),
        .valid_n(ws_log_19_valid_n)
    );

    wire ws_log_20_valid, ws_log_20_ready_n, ws_log_20_ready, ws_log_20_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_20_data_in, ws_log_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_20 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_20_valid),
        .ready_n(ws_log_20_ready_n),
        .data_in(ws_log_20_data_in),
        .data_out(ws_log_20_data_out),
        .ready(ws_log_20_ready),
        .valid_n(ws_log_20_valid_n)
    );

    wire ws_log_21_valid, ws_log_21_ready_n, ws_log_21_ready, ws_log_21_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_21_data_in, ws_log_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_21 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_21_valid),
        .ready_n(ws_log_21_ready_n),
        .data_in(ws_log_21_data_in),
        .data_out(ws_log_21_data_out),
        .ready(ws_log_21_ready),
        .valid_n(ws_log_21_valid_n)
    );

    wire ws_log_22_valid, ws_log_22_ready_n, ws_log_22_ready, ws_log_22_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_22_data_in, ws_log_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_22 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_22_valid),
        .ready_n(ws_log_22_ready_n),
        .data_in(ws_log_22_data_in),
        .data_out(ws_log_22_data_out),
        .ready(ws_log_22_ready),
        .valid_n(ws_log_22_valid_n)
    );

    wire ws_log_23_valid, ws_log_23_ready_n, ws_log_23_ready, ws_log_23_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_23_data_in, ws_log_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_23 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_23_valid),
        .ready_n(ws_log_23_ready_n),
        .data_in(ws_log_23_data_in),
        .data_out(ws_log_23_data_out),
        .ready(ws_log_23_ready),
        .valid_n(ws_log_23_valid_n)
    );

    assign ws_log_0_data_in = attn_sram_out;
    assign ws_log_0_valid = (ws_state == WS_LOG);
    assign ws_log_0_ready_n = 1'b0;
    assign ws_log_1_data_in = attn_sram_out;
    assign ws_log_1_valid = (ws_state == WS_LOG);
    assign ws_log_1_ready_n = 1'b0;
    assign ws_log_2_data_in = attn_sram_out;
    assign ws_log_2_valid = (ws_state == WS_LOG);
    assign ws_log_2_ready_n = 1'b0;
    assign ws_log_3_data_in = attn_sram_out;
    assign ws_log_3_valid = (ws_state == WS_LOG);
    assign ws_log_3_ready_n = 1'b0;
    assign ws_log_4_data_in = attn_sram_out;
    assign ws_log_4_valid = (ws_state == WS_LOG);
    assign ws_log_4_ready_n = 1'b0;
    assign ws_log_5_data_in = attn_sram_out;
    assign ws_log_5_valid = (ws_state == WS_LOG);
    assign ws_log_5_ready_n = 1'b0;
    assign ws_log_6_data_in = attn_sram_out;
    assign ws_log_6_valid = (ws_state == WS_LOG);
    assign ws_log_6_ready_n = 1'b0;
    assign ws_log_7_data_in = attn_sram_out;
    assign ws_log_7_valid = (ws_state == WS_LOG);
    assign ws_log_7_ready_n = 1'b0;
    assign ws_log_8_data_in = attn_sram_out;
    assign ws_log_8_valid = (ws_state == WS_LOG);
    assign ws_log_8_ready_n = 1'b0;
    assign ws_log_9_data_in = attn_sram_out;
    assign ws_log_9_valid = (ws_state == WS_LOG);
    assign ws_log_9_ready_n = 1'b0;
    assign ws_log_10_data_in = attn_sram_out;
    assign ws_log_10_valid = (ws_state == WS_LOG);
    assign ws_log_10_ready_n = 1'b0;
    assign ws_log_11_data_in = attn_sram_out;
    assign ws_log_11_valid = (ws_state == WS_LOG);
    assign ws_log_11_ready_n = 1'b0;
    assign ws_log_12_data_in = attn_sram_out;
    assign ws_log_12_valid = (ws_state == WS_LOG);
    assign ws_log_12_ready_n = 1'b0;
    assign ws_log_13_data_in = attn_sram_out;
    assign ws_log_13_valid = (ws_state == WS_LOG);
    assign ws_log_13_ready_n = 1'b0;
    assign ws_log_14_data_in = attn_sram_out;
    assign ws_log_14_valid = (ws_state == WS_LOG);
    assign ws_log_14_ready_n = 1'b0;
    assign ws_log_15_data_in = attn_sram_out;
    assign ws_log_15_valid = (ws_state == WS_LOG);
    assign ws_log_15_ready_n = 1'b0;
    assign ws_log_16_data_in = attn_sram_out;
    assign ws_log_16_valid = (ws_state == WS_LOG);
    assign ws_log_16_ready_n = 1'b0;
    assign ws_log_17_data_in = attn_sram_out;
    assign ws_log_17_valid = (ws_state == WS_LOG);
    assign ws_log_17_ready_n = 1'b0;
    assign ws_log_18_data_in = attn_sram_out;
    assign ws_log_18_valid = (ws_state == WS_LOG);
    assign ws_log_18_ready_n = 1'b0;
    assign ws_log_19_data_in = attn_sram_out;
    assign ws_log_19_valid = (ws_state == WS_LOG);
    assign ws_log_19_ready_n = 1'b0;
    assign ws_log_20_data_in = attn_sram_out;
    assign ws_log_20_valid = (ws_state == WS_LOG);
    assign ws_log_20_ready_n = 1'b0;
    assign ws_log_21_data_in = attn_sram_out;
    assign ws_log_21_valid = (ws_state == WS_LOG);
    assign ws_log_21_ready_n = 1'b0;
    assign ws_log_22_data_in = attn_sram_out;
    assign ws_log_22_valid = (ws_state == WS_LOG);
    assign ws_log_22_ready_n = 1'b0;
    assign ws_log_23_data_in = attn_sram_out;
    assign ws_log_23_valid = (ws_state == WS_LOG);
    assign ws_log_23_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_0_valid_n) begin
            log_attn_write_data <= ws_log_0_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_0_valid, ws_exp_0_ready_n, ws_exp_0_ready, ws_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_0_data_in, ws_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_0_valid),
        .ready_n(ws_exp_0_ready_n),
        .data_in(ws_exp_0_data_in),
        .data_out(ws_exp_0_data_out),
        .ready(ws_exp_0_ready),
        .valid_n(ws_exp_0_valid_n)
    );

    wire ws_exp_1_valid, ws_exp_1_ready_n, ws_exp_1_ready, ws_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_1_data_in, ws_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_1_valid),
        .ready_n(ws_exp_1_ready_n),
        .data_in(ws_exp_1_data_in),
        .data_out(ws_exp_1_data_out),
        .ready(ws_exp_1_ready),
        .valid_n(ws_exp_1_valid_n)
    );

    wire ws_exp_2_valid, ws_exp_2_ready_n, ws_exp_2_ready, ws_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_2_data_in, ws_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_2_valid),
        .ready_n(ws_exp_2_ready_n),
        .data_in(ws_exp_2_data_in),
        .data_out(ws_exp_2_data_out),
        .ready(ws_exp_2_ready),
        .valid_n(ws_exp_2_valid_n)
    );

    wire ws_exp_3_valid, ws_exp_3_ready_n, ws_exp_3_ready, ws_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_3_data_in, ws_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_3_valid),
        .ready_n(ws_exp_3_ready_n),
        .data_in(ws_exp_3_data_in),
        .data_out(ws_exp_3_data_out),
        .ready(ws_exp_3_ready),
        .valid_n(ws_exp_3_valid_n)
    );

    wire ws_exp_4_valid, ws_exp_4_ready_n, ws_exp_4_ready, ws_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_4_data_in, ws_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_4_valid),
        .ready_n(ws_exp_4_ready_n),
        .data_in(ws_exp_4_data_in),
        .data_out(ws_exp_4_data_out),
        .ready(ws_exp_4_ready),
        .valid_n(ws_exp_4_valid_n)
    );

    wire ws_exp_5_valid, ws_exp_5_ready_n, ws_exp_5_ready, ws_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_5_data_in, ws_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_5_valid),
        .ready_n(ws_exp_5_ready_n),
        .data_in(ws_exp_5_data_in),
        .data_out(ws_exp_5_data_out),
        .ready(ws_exp_5_ready),
        .valid_n(ws_exp_5_valid_n)
    );

    wire ws_exp_6_valid, ws_exp_6_ready_n, ws_exp_6_ready, ws_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_6_data_in, ws_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_6_valid),
        .ready_n(ws_exp_6_ready_n),
        .data_in(ws_exp_6_data_in),
        .data_out(ws_exp_6_data_out),
        .ready(ws_exp_6_ready),
        .valid_n(ws_exp_6_valid_n)
    );

    wire ws_exp_7_valid, ws_exp_7_ready_n, ws_exp_7_ready, ws_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_7_data_in, ws_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_7_valid),
        .ready_n(ws_exp_7_ready_n),
        .data_in(ws_exp_7_data_in),
        .data_out(ws_exp_7_data_out),
        .ready(ws_exp_7_ready),
        .valid_n(ws_exp_7_valid_n)
    );

    wire ws_exp_8_valid, ws_exp_8_ready_n, ws_exp_8_ready, ws_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_8_data_in, ws_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_8_valid),
        .ready_n(ws_exp_8_ready_n),
        .data_in(ws_exp_8_data_in),
        .data_out(ws_exp_8_data_out),
        .ready(ws_exp_8_ready),
        .valid_n(ws_exp_8_valid_n)
    );

    wire ws_exp_9_valid, ws_exp_9_ready_n, ws_exp_9_ready, ws_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_9_data_in, ws_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_9_valid),
        .ready_n(ws_exp_9_ready_n),
        .data_in(ws_exp_9_data_in),
        .data_out(ws_exp_9_data_out),
        .ready(ws_exp_9_ready),
        .valid_n(ws_exp_9_valid_n)
    );

    wire ws_exp_10_valid, ws_exp_10_ready_n, ws_exp_10_ready, ws_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_10_data_in, ws_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_10_valid),
        .ready_n(ws_exp_10_ready_n),
        .data_in(ws_exp_10_data_in),
        .data_out(ws_exp_10_data_out),
        .ready(ws_exp_10_ready),
        .valid_n(ws_exp_10_valid_n)
    );

    wire ws_exp_11_valid, ws_exp_11_ready_n, ws_exp_11_ready, ws_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_11_data_in, ws_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_11_valid),
        .ready_n(ws_exp_11_ready_n),
        .data_in(ws_exp_11_data_in),
        .data_out(ws_exp_11_data_out),
        .ready(ws_exp_11_ready),
        .valid_n(ws_exp_11_valid_n)
    );

    wire ws_exp_12_valid, ws_exp_12_ready_n, ws_exp_12_ready, ws_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_12_data_in, ws_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_12_valid),
        .ready_n(ws_exp_12_ready_n),
        .data_in(ws_exp_12_data_in),
        .data_out(ws_exp_12_data_out),
        .ready(ws_exp_12_ready),
        .valid_n(ws_exp_12_valid_n)
    );

    wire ws_exp_13_valid, ws_exp_13_ready_n, ws_exp_13_ready, ws_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_13_data_in, ws_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_13_valid),
        .ready_n(ws_exp_13_ready_n),
        .data_in(ws_exp_13_data_in),
        .data_out(ws_exp_13_data_out),
        .ready(ws_exp_13_ready),
        .valid_n(ws_exp_13_valid_n)
    );

    wire ws_exp_14_valid, ws_exp_14_ready_n, ws_exp_14_ready, ws_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_14_data_in, ws_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_14_valid),
        .ready_n(ws_exp_14_ready_n),
        .data_in(ws_exp_14_data_in),
        .data_out(ws_exp_14_data_out),
        .ready(ws_exp_14_ready),
        .valid_n(ws_exp_14_valid_n)
    );

    wire ws_exp_15_valid, ws_exp_15_ready_n, ws_exp_15_ready, ws_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_15_data_in, ws_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_15_valid),
        .ready_n(ws_exp_15_ready_n),
        .data_in(ws_exp_15_data_in),
        .data_out(ws_exp_15_data_out),
        .ready(ws_exp_15_ready),
        .valid_n(ws_exp_15_valid_n)
    );

    wire ws_exp_16_valid, ws_exp_16_ready_n, ws_exp_16_ready, ws_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_16_data_in, ws_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_16_valid),
        .ready_n(ws_exp_16_ready_n),
        .data_in(ws_exp_16_data_in),
        .data_out(ws_exp_16_data_out),
        .ready(ws_exp_16_ready),
        .valid_n(ws_exp_16_valid_n)
    );

    wire ws_exp_17_valid, ws_exp_17_ready_n, ws_exp_17_ready, ws_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_17_data_in, ws_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_17_valid),
        .ready_n(ws_exp_17_ready_n),
        .data_in(ws_exp_17_data_in),
        .data_out(ws_exp_17_data_out),
        .ready(ws_exp_17_ready),
        .valid_n(ws_exp_17_valid_n)
    );

    wire ws_exp_18_valid, ws_exp_18_ready_n, ws_exp_18_ready, ws_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_18_data_in, ws_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_18_valid),
        .ready_n(ws_exp_18_ready_n),
        .data_in(ws_exp_18_data_in),
        .data_out(ws_exp_18_data_out),
        .ready(ws_exp_18_ready),
        .valid_n(ws_exp_18_valid_n)
    );

    wire ws_exp_19_valid, ws_exp_19_ready_n, ws_exp_19_ready, ws_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_19_data_in, ws_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_19_valid),
        .ready_n(ws_exp_19_ready_n),
        .data_in(ws_exp_19_data_in),
        .data_out(ws_exp_19_data_out),
        .ready(ws_exp_19_ready),
        .valid_n(ws_exp_19_valid_n)
    );

    wire ws_exp_20_valid, ws_exp_20_ready_n, ws_exp_20_ready, ws_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_20_data_in, ws_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_20_valid),
        .ready_n(ws_exp_20_ready_n),
        .data_in(ws_exp_20_data_in),
        .data_out(ws_exp_20_data_out),
        .ready(ws_exp_20_ready),
        .valid_n(ws_exp_20_valid_n)
    );

    wire ws_exp_21_valid, ws_exp_21_ready_n, ws_exp_21_ready, ws_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_21_data_in, ws_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_21_valid),
        .ready_n(ws_exp_21_ready_n),
        .data_in(ws_exp_21_data_in),
        .data_out(ws_exp_21_data_out),
        .ready(ws_exp_21_ready),
        .valid_n(ws_exp_21_valid_n)
    );

    wire ws_exp_22_valid, ws_exp_22_ready_n, ws_exp_22_ready, ws_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_22_data_in, ws_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_22_valid),
        .ready_n(ws_exp_22_ready_n),
        .data_in(ws_exp_22_data_in),
        .data_out(ws_exp_22_data_out),
        .ready(ws_exp_22_ready),
        .valid_n(ws_exp_22_valid_n)
    );

    wire ws_exp_23_valid, ws_exp_23_ready_n, ws_exp_23_ready, ws_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_23_data_in, ws_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_23_valid),
        .ready_n(ws_exp_23_ready_n),
        .data_in(ws_exp_23_data_in),
        .data_out(ws_exp_23_data_out),
        .ready(ws_exp_23_ready),
        .valid_n(ws_exp_23_valid_n)
    );

    assign ws_exp_0_data_in = ws_log_sum;
    assign ws_exp_0_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_0_ready_n = 1'b0;
    assign ws_exp_1_data_in = ws_log_sum;
    assign ws_exp_1_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_1_ready_n = 1'b0;
    assign ws_exp_2_data_in = ws_log_sum;
    assign ws_exp_2_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_2_ready_n = 1'b0;
    assign ws_exp_3_data_in = ws_log_sum;
    assign ws_exp_3_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_3_ready_n = 1'b0;
    assign ws_exp_4_data_in = ws_log_sum;
    assign ws_exp_4_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_4_ready_n = 1'b0;
    assign ws_exp_5_data_in = ws_log_sum;
    assign ws_exp_5_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_5_ready_n = 1'b0;
    assign ws_exp_6_data_in = ws_log_sum;
    assign ws_exp_6_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_6_ready_n = 1'b0;
    assign ws_exp_7_data_in = ws_log_sum;
    assign ws_exp_7_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_7_ready_n = 1'b0;
    assign ws_exp_8_data_in = ws_log_sum;
    assign ws_exp_8_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_8_ready_n = 1'b0;
    assign ws_exp_9_data_in = ws_log_sum;
    assign ws_exp_9_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_9_ready_n = 1'b0;
    assign ws_exp_10_data_in = ws_log_sum;
    assign ws_exp_10_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_10_ready_n = 1'b0;
    assign ws_exp_11_data_in = ws_log_sum;
    assign ws_exp_11_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_11_ready_n = 1'b0;
    assign ws_exp_12_data_in = ws_log_sum;
    assign ws_exp_12_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_12_ready_n = 1'b0;
    assign ws_exp_13_data_in = ws_log_sum;
    assign ws_exp_13_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_13_ready_n = 1'b0;
    assign ws_exp_14_data_in = ws_log_sum;
    assign ws_exp_14_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_14_ready_n = 1'b0;
    assign ws_exp_15_data_in = ws_log_sum;
    assign ws_exp_15_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_15_ready_n = 1'b0;
    assign ws_exp_16_data_in = ws_log_sum;
    assign ws_exp_16_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_16_ready_n = 1'b0;
    assign ws_exp_17_data_in = ws_log_sum;
    assign ws_exp_17_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_17_ready_n = 1'b0;
    assign ws_exp_18_data_in = ws_log_sum;
    assign ws_exp_18_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_18_ready_n = 1'b0;
    assign ws_exp_19_data_in = ws_log_sum;
    assign ws_exp_19_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_19_ready_n = 1'b0;
    assign ws_exp_20_data_in = ws_log_sum;
    assign ws_exp_20_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_20_ready_n = 1'b0;
    assign ws_exp_21_data_in = ws_log_sum;
    assign ws_exp_21_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_21_ready_n = 1'b0;
    assign ws_exp_22_data_in = ws_log_sum;
    assign ws_exp_22_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_22_ready_n = 1'b0;
    assign ws_exp_23_data_in = ws_log_sum;
    assign ws_exp_23_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_23_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_0_valid_n) ws_accumulator <= ws_accumulator + ws_exp_0_data_out;
    end

    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    reg [DATA_WIDTH-1:0] out_write_data;
    wire [DATA_WIDTH-1:0] out_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),
              .w_addr(out_write_addr),.sram_data_in(out_write_data),.sram_data_out(out_sram_out));

    localparam WS_IDLE = 3'd0, WS_LOAD_A = 3'd1, WS_LOAD_V = 3'd2,
               WS_LOG = 3'd3, WS_COMPUTE = 3'd4, WS_OUTPUT = 3'd5;
    reg [2:0] ws_state;
    reg [$clog2(N)-1:0] ws_j;
    reg [$clog2(d)-1:0] ws_m;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ws_state <= WS_IDLE; ws_j <= 0; ws_m <= 0;
            attn_write_addr <= 0; attn_read_addr <= 0; attn_w_en <= 0;
            v_write_addr <= 0; v_read_addr <= 0; v_w_en <= 0;
            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;
            out_write_data <= 0;
        end else begin
            attn_w_en <= 0; v_w_en <= 0; out_w_en <= 0;
            case (ws_state)
                WS_IDLE: if (valid_attn || valid_v) ws_state <= WS_LOAD_A;
                WS_LOAD_A: begin
                    if (valid_attn) begin attn_w_en <= 1; attn_write_addr <= attn_write_addr + 1; end
                    if (attn_write_addr == N-1) ws_state <= WS_LOAD_V;
                end
                WS_LOAD_V: begin
                    if (valid_v) begin v_w_en <= 1; v_write_addr <= v_write_addr + 1; end
                    if (v_write_addr == N*d-1) ws_state <= WS_LOG;
                end
                WS_LOG: begin
                    // DPE(I|log) converts attn to log domain
                    attn_read_addr <= ws_j;
                    ws_j <= ws_j + 1;
                    if (ws_j == N-1) begin ws_j <= 0; ws_state <= WS_COMPUTE; end
                end
                WS_COMPUTE: begin
                    // CLB add + DPE(I|exp) + accumulate
                    log_attn_read_addr <= ws_j;
                    v_read_addr <= (ws_j << $clog2(d)) + ws_m;
                    ws_j <= ws_j + 1;
                    if (ws_j == N-1) begin
                        out_write_data <= ws_accumulator[2*DATA_WIDTH-1:DATA_WIDTH];
                        out_w_en <= 1; out_write_addr <= ws_m;
                        ws_j <= 0; ws_m <= ws_m + 1;
                        if (ws_m == d-1) ws_state <= WS_OUTPUT;
                    end
                end
                WS_OUTPUT: if (ready_n) out_read_addr <= out_read_addr + 1;
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (ws_state == WS_OUTPUT);
    assign ready_attn = (ws_state == WS_LOAD_A || ws_state == WS_IDLE);
    assign ready_v = (ws_state == WS_LOAD_V || ws_state == WS_IDLE);
endmodule

// DIMM — Block 0, Head 1, Lane 0 (depth=393216)
module dimm_score_matrix_b0_h1 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid_q, input wire valid_k,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_q,
    input wire [DATA_WIDTH-1:0] data_in_k,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_q, output wire ready_k,
    output wire valid_n
);

    reg [ADDR_WIDTH-1:0] q_write_addr, q_read_addr;
    reg q_w_en;
    wire [DATA_WIDTH-1:0] q_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    q_sram (.clk(clk),.rst(rst),.w_en(q_w_en),.r_addr(q_read_addr),
            .w_addr(q_write_addr),.sram_data_in(data_in_q),.sram_data_out(q_sram_out));

    reg [$clog2(N*d)-1:0] k_write_addr, k_read_addr_a;
    reg k_w_en;
    wire [DATA_WIDTH-1:0] k_sram_out_a;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    k_sram_a (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_a[$clog2(N*d)-1:0]),
              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_a));
    reg [$clog2(N*d)-1:0] k_read_addr_b;
    wire [DATA_WIDTH-1:0] k_sram_out_b;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    k_sram_b (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_b[$clog2(N*d)-1:0]),
              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_b));

    // CLB adder A: log_Q + log_K[j₁] (log-domain addition)
    wire [DATA_WIDTH-1:0] log_sum_a = q_sram_out + k_sram_out_a;
    // CLB adder B: log_Q + log_K[j₂] (second vector for dual-identity)
    wire [DATA_WIDTH-1:0] log_sum_b = q_sram_out + k_sram_out_b;

    // DPE(I|exp) stage_0: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_0_valid, dimm_exp_0_ready_n, dimm_exp_0_ready, dimm_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_0_data_in, dimm_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_0_valid),
        .ready_n(dimm_exp_0_ready_n),
        .data_in(dimm_exp_0_data_in),
        .data_out(dimm_exp_0_data_out),
        .ready(dimm_exp_0_ready),
        .valid_n(dimm_exp_0_valid_n)
    );

    // DPE(I|exp) stage_1: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_1_valid, dimm_exp_1_ready_n, dimm_exp_1_ready, dimm_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_1_data_in, dimm_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_1_valid),
        .ready_n(dimm_exp_1_ready_n),
        .data_in(dimm_exp_1_data_in),
        .data_out(dimm_exp_1_data_out),
        .ready(dimm_exp_1_ready),
        .valid_n(dimm_exp_1_valid_n)
    );

    // DPE(I|exp) stage_2: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_2_valid, dimm_exp_2_ready_n, dimm_exp_2_ready, dimm_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_2_data_in, dimm_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_2_valid),
        .ready_n(dimm_exp_2_ready_n),
        .data_in(dimm_exp_2_data_in),
        .data_out(dimm_exp_2_data_out),
        .ready(dimm_exp_2_ready),
        .valid_n(dimm_exp_2_valid_n)
    );

    // DPE(I|exp) stage_3: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_3_valid, dimm_exp_3_ready_n, dimm_exp_3_ready, dimm_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_3_data_in, dimm_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_3_valid),
        .ready_n(dimm_exp_3_ready_n),
        .data_in(dimm_exp_3_data_in),
        .data_out(dimm_exp_3_data_out),
        .ready(dimm_exp_3_ready),
        .valid_n(dimm_exp_3_valid_n)
    );

    // DPE(I|exp) stage_4: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_4_valid, dimm_exp_4_ready_n, dimm_exp_4_ready, dimm_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_4_data_in, dimm_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_4_valid),
        .ready_n(dimm_exp_4_ready_n),
        .data_in(dimm_exp_4_data_in),
        .data_out(dimm_exp_4_data_out),
        .ready(dimm_exp_4_ready),
        .valid_n(dimm_exp_4_valid_n)
    );

    // DPE(I|exp) stage_5: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_5_valid, dimm_exp_5_ready_n, dimm_exp_5_ready, dimm_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_5_data_in, dimm_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_5_valid),
        .ready_n(dimm_exp_5_ready_n),
        .data_in(dimm_exp_5_data_in),
        .data_out(dimm_exp_5_data_out),
        .ready(dimm_exp_5_ready),
        .valid_n(dimm_exp_5_valid_n)
    );

    // DPE(I|exp) stage_6: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_6_valid, dimm_exp_6_ready_n, dimm_exp_6_ready, dimm_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_6_data_in, dimm_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_6_valid),
        .ready_n(dimm_exp_6_ready_n),
        .data_in(dimm_exp_6_data_in),
        .data_out(dimm_exp_6_data_out),
        .ready(dimm_exp_6_ready),
        .valid_n(dimm_exp_6_valid_n)
    );

    // DPE(I|exp) stage_7: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_7_valid, dimm_exp_7_ready_n, dimm_exp_7_ready, dimm_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_7_data_in, dimm_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_7_valid),
        .ready_n(dimm_exp_7_ready_n),
        .data_in(dimm_exp_7_data_in),
        .data_out(dimm_exp_7_data_out),
        .ready(dimm_exp_7_ready),
        .valid_n(dimm_exp_7_valid_n)
    );

    // DPE(I|exp) stage_8: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_8_valid, dimm_exp_8_ready_n, dimm_exp_8_ready, dimm_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_8_data_in, dimm_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_8_valid),
        .ready_n(dimm_exp_8_ready_n),
        .data_in(dimm_exp_8_data_in),
        .data_out(dimm_exp_8_data_out),
        .ready(dimm_exp_8_ready),
        .valid_n(dimm_exp_8_valid_n)
    );

    // DPE(I|exp) stage_9: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_9_valid, dimm_exp_9_ready_n, dimm_exp_9_ready, dimm_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_9_data_in, dimm_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_9_valid),
        .ready_n(dimm_exp_9_ready_n),
        .data_in(dimm_exp_9_data_in),
        .data_out(dimm_exp_9_data_out),
        .ready(dimm_exp_9_ready),
        .valid_n(dimm_exp_9_valid_n)
    );

    // DPE(I|exp) stage_10: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_10_valid, dimm_exp_10_ready_n, dimm_exp_10_ready, dimm_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_10_data_in, dimm_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_10_valid),
        .ready_n(dimm_exp_10_ready_n),
        .data_in(dimm_exp_10_data_in),
        .data_out(dimm_exp_10_data_out),
        .ready(dimm_exp_10_ready),
        .valid_n(dimm_exp_10_valid_n)
    );

    // DPE(I|exp) stage_11: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_11_valid, dimm_exp_11_ready_n, dimm_exp_11_ready, dimm_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_11_data_in, dimm_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_11_valid),
        .ready_n(dimm_exp_11_ready_n),
        .data_in(dimm_exp_11_data_in),
        .data_out(dimm_exp_11_data_out),
        .ready(dimm_exp_11_ready),
        .valid_n(dimm_exp_11_valid_n)
    );

    // DPE(I|exp) stage_12: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_12_valid, dimm_exp_12_ready_n, dimm_exp_12_ready, dimm_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_12_data_in, dimm_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_12_valid),
        .ready_n(dimm_exp_12_ready_n),
        .data_in(dimm_exp_12_data_in),
        .data_out(dimm_exp_12_data_out),
        .ready(dimm_exp_12_ready),
        .valid_n(dimm_exp_12_valid_n)
    );

    // DPE(I|exp) stage_13: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_13_valid, dimm_exp_13_ready_n, dimm_exp_13_ready, dimm_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_13_data_in, dimm_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_13_valid),
        .ready_n(dimm_exp_13_ready_n),
        .data_in(dimm_exp_13_data_in),
        .data_out(dimm_exp_13_data_out),
        .ready(dimm_exp_13_ready),
        .valid_n(dimm_exp_13_valid_n)
    );

    // DPE(I|exp) stage_14: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_14_valid, dimm_exp_14_ready_n, dimm_exp_14_ready, dimm_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_14_data_in, dimm_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_14_valid),
        .ready_n(dimm_exp_14_ready_n),
        .data_in(dimm_exp_14_data_in),
        .data_out(dimm_exp_14_data_out),
        .ready(dimm_exp_14_ready),
        .valid_n(dimm_exp_14_valid_n)
    );

    // DPE(I|exp) stage_15: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_15_valid, dimm_exp_15_ready_n, dimm_exp_15_ready, dimm_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_15_data_in, dimm_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_15_valid),
        .ready_n(dimm_exp_15_ready_n),
        .data_in(dimm_exp_15_data_in),
        .data_out(dimm_exp_15_data_out),
        .ready(dimm_exp_15_ready),
        .valid_n(dimm_exp_15_valid_n)
    );

    // DPE(I|exp) stage_16: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_16_valid, dimm_exp_16_ready_n, dimm_exp_16_ready, dimm_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_16_data_in, dimm_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_16_valid),
        .ready_n(dimm_exp_16_ready_n),
        .data_in(dimm_exp_16_data_in),
        .data_out(dimm_exp_16_data_out),
        .ready(dimm_exp_16_ready),
        .valid_n(dimm_exp_16_valid_n)
    );

    // DPE(I|exp) stage_17: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_17_valid, dimm_exp_17_ready_n, dimm_exp_17_ready, dimm_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_17_data_in, dimm_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_17_valid),
        .ready_n(dimm_exp_17_ready_n),
        .data_in(dimm_exp_17_data_in),
        .data_out(dimm_exp_17_data_out),
        .ready(dimm_exp_17_ready),
        .valid_n(dimm_exp_17_valid_n)
    );

    // DPE(I|exp) stage_18: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_18_valid, dimm_exp_18_ready_n, dimm_exp_18_ready, dimm_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_18_data_in, dimm_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_18_valid),
        .ready_n(dimm_exp_18_ready_n),
        .data_in(dimm_exp_18_data_in),
        .data_out(dimm_exp_18_data_out),
        .ready(dimm_exp_18_ready),
        .valid_n(dimm_exp_18_valid_n)
    );

    // DPE(I|exp) stage_19: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_19_valid, dimm_exp_19_ready_n, dimm_exp_19_ready, dimm_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_19_data_in, dimm_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_19_valid),
        .ready_n(dimm_exp_19_ready_n),
        .data_in(dimm_exp_19_data_in),
        .data_out(dimm_exp_19_data_out),
        .ready(dimm_exp_19_ready),
        .valid_n(dimm_exp_19_valid_n)
    );

    // DPE(I|exp) stage_20: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_20_valid, dimm_exp_20_ready_n, dimm_exp_20_ready, dimm_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_20_data_in, dimm_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_20_valid),
        .ready_n(dimm_exp_20_ready_n),
        .data_in(dimm_exp_20_data_in),
        .data_out(dimm_exp_20_data_out),
        .ready(dimm_exp_20_ready),
        .valid_n(dimm_exp_20_valid_n)
    );

    // DPE(I|exp) stage_21: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_21_valid, dimm_exp_21_ready_n, dimm_exp_21_ready, dimm_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_21_data_in, dimm_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_21_valid),
        .ready_n(dimm_exp_21_ready_n),
        .data_in(dimm_exp_21_data_in),
        .data_out(dimm_exp_21_data_out),
        .ready(dimm_exp_21_ready),
        .valid_n(dimm_exp_21_valid_n)
    );

    // DPE(I|exp) stage_22: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_22_valid, dimm_exp_22_ready_n, dimm_exp_22_ready, dimm_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_22_data_in, dimm_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_22_valid),
        .ready_n(dimm_exp_22_ready_n),
        .data_in(dimm_exp_22_data_in),
        .data_out(dimm_exp_22_data_out),
        .ready(dimm_exp_22_ready),
        .valid_n(dimm_exp_22_valid_n)
    );

    // DPE(I|exp) stage_23: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_23_valid, dimm_exp_23_ready_n, dimm_exp_23_ready, dimm_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_23_data_in, dimm_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_23_valid),
        .ready_n(dimm_exp_23_ready_n),
        .data_in(dimm_exp_23_data_in),
        .data_out(dimm_exp_23_data_out),
        .ready(dimm_exp_23_ready),
        .valid_n(dimm_exp_23_valid_n)
    );

    reg [4:0] dimm_sel;
    assign dimm_exp_0_data_in = log_sum_a;
    assign dimm_exp_0_valid = (state == S_COMPUTE) && (dimm_sel == 0);
    assign dimm_exp_0_ready_n = 1'b0;
    assign dimm_exp_1_data_in = log_sum_a;
    assign dimm_exp_1_valid = (state == S_COMPUTE) && (dimm_sel == 1);
    assign dimm_exp_1_ready_n = 1'b0;
    assign dimm_exp_2_data_in = log_sum_a;
    assign dimm_exp_2_valid = (state == S_COMPUTE) && (dimm_sel == 2);
    assign dimm_exp_2_ready_n = 1'b0;
    assign dimm_exp_3_data_in = log_sum_a;
    assign dimm_exp_3_valid = (state == S_COMPUTE) && (dimm_sel == 3);
    assign dimm_exp_3_ready_n = 1'b0;
    assign dimm_exp_4_data_in = log_sum_a;
    assign dimm_exp_4_valid = (state == S_COMPUTE) && (dimm_sel == 4);
    assign dimm_exp_4_ready_n = 1'b0;
    assign dimm_exp_5_data_in = log_sum_a;
    assign dimm_exp_5_valid = (state == S_COMPUTE) && (dimm_sel == 5);
    assign dimm_exp_5_ready_n = 1'b0;
    assign dimm_exp_6_data_in = log_sum_a;
    assign dimm_exp_6_valid = (state == S_COMPUTE) && (dimm_sel == 6);
    assign dimm_exp_6_ready_n = 1'b0;
    assign dimm_exp_7_data_in = log_sum_a;
    assign dimm_exp_7_valid = (state == S_COMPUTE) && (dimm_sel == 7);
    assign dimm_exp_7_ready_n = 1'b0;
    assign dimm_exp_8_data_in = log_sum_a;
    assign dimm_exp_8_valid = (state == S_COMPUTE) && (dimm_sel == 8);
    assign dimm_exp_8_ready_n = 1'b0;
    assign dimm_exp_9_data_in = log_sum_a;
    assign dimm_exp_9_valid = (state == S_COMPUTE) && (dimm_sel == 9);
    assign dimm_exp_9_ready_n = 1'b0;
    assign dimm_exp_10_data_in = log_sum_a;
    assign dimm_exp_10_valid = (state == S_COMPUTE) && (dimm_sel == 10);
    assign dimm_exp_10_ready_n = 1'b0;
    assign dimm_exp_11_data_in = log_sum_a;
    assign dimm_exp_11_valid = (state == S_COMPUTE) && (dimm_sel == 11);
    assign dimm_exp_11_ready_n = 1'b0;
    assign dimm_exp_12_data_in = log_sum_a;
    assign dimm_exp_12_valid = (state == S_COMPUTE) && (dimm_sel == 12);
    assign dimm_exp_12_ready_n = 1'b0;
    assign dimm_exp_13_data_in = log_sum_a;
    assign dimm_exp_13_valid = (state == S_COMPUTE) && (dimm_sel == 13);
    assign dimm_exp_13_ready_n = 1'b0;
    assign dimm_exp_14_data_in = log_sum_a;
    assign dimm_exp_14_valid = (state == S_COMPUTE) && (dimm_sel == 14);
    assign dimm_exp_14_ready_n = 1'b0;
    assign dimm_exp_15_data_in = log_sum_a;
    assign dimm_exp_15_valid = (state == S_COMPUTE) && (dimm_sel == 15);
    assign dimm_exp_15_ready_n = 1'b0;
    assign dimm_exp_16_data_in = log_sum_a;
    assign dimm_exp_16_valid = (state == S_COMPUTE) && (dimm_sel == 16);
    assign dimm_exp_16_ready_n = 1'b0;
    assign dimm_exp_17_data_in = log_sum_a;
    assign dimm_exp_17_valid = (state == S_COMPUTE) && (dimm_sel == 17);
    assign dimm_exp_17_ready_n = 1'b0;
    assign dimm_exp_18_data_in = log_sum_a;
    assign dimm_exp_18_valid = (state == S_COMPUTE) && (dimm_sel == 18);
    assign dimm_exp_18_ready_n = 1'b0;
    assign dimm_exp_19_data_in = log_sum_a;
    assign dimm_exp_19_valid = (state == S_COMPUTE) && (dimm_sel == 19);
    assign dimm_exp_19_ready_n = 1'b0;
    assign dimm_exp_20_data_in = log_sum_a;
    assign dimm_exp_20_valid = (state == S_COMPUTE) && (dimm_sel == 20);
    assign dimm_exp_20_ready_n = 1'b0;
    assign dimm_exp_21_data_in = log_sum_a;
    assign dimm_exp_21_valid = (state == S_COMPUTE) && (dimm_sel == 21);
    assign dimm_exp_21_ready_n = 1'b0;
    assign dimm_exp_22_data_in = log_sum_a;
    assign dimm_exp_22_valid = (state == S_COMPUTE) && (dimm_sel == 22);
    assign dimm_exp_22_ready_n = 1'b0;
    assign dimm_exp_23_data_in = log_sum_a;
    assign dimm_exp_23_valid = (state == S_COMPUTE) && (dimm_sel == 23);
    assign dimm_exp_23_ready_n = 1'b0;

    // Dual accumulators: one per vector (A for j₁, B for j₂)
    reg [2*DATA_WIDTH-1:0] accumulator_a, accumulator_b;
    reg acc_phase;  // 0=accumulating A outputs, 1=accumulating B outputs
    always @(posedge clk) begin
        if (rst) begin accumulator_a <= 0; accumulator_b <= 0; end
        else if (dimm_exp_valid_n) begin
            if (!acc_phase) accumulator_a <= accumulator_a + dimm_exp_data_out;
            else accumulator_b <= accumulator_b + dimm_exp_data_out;
        end
    end

    reg [ADDR_WIDTH-1:0] score_write_addr, score_read_addr;
    reg score_w_en;
    reg [DATA_WIDTH-1:0] score_write_data;
    wire [DATA_WIDTH-1:0] score_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    score_sram (.clk(clk),.rst(rst),.w_en(score_w_en),.r_addr(score_read_addr),
                .w_addr(score_write_addr),.sram_data_in(score_write_data),.sram_data_out(score_sram_out));

    localparam S_IDLE = 3'd0, S_LOAD_Q = 3'd1, S_LOAD_K = 3'd2,
               S_COMPUTE = 3'd3, S_OUTPUT = 3'd4;
    localparam S_WRITE_B = 3'd5;  // write second score for dual-identity
    reg [2:0] state;
    reg [$clog2(d)-1:0] mac_count;
    reg [$clog2(N)-1:0] score_idx;
    reg feed_half;  // 0=feeding vector A (d elements), 1=feeding vector B

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            q_write_addr <= 0; q_read_addr <= 0; q_w_en <= 0;
            k_write_addr <= 0; k_read_addr_a <= 0; k_w_en <= 0;
            k_read_addr_b <= 0;
            feed_half <= 0; feed_phase <= 0; acc_phase <= 0;
            score_write_addr <= 0; score_read_addr <= 0; score_w_en <= 0;
            score_write_data <= 0;
            mac_count <= 0; score_idx <= 0;
            dimm_sel <= 0;
        end else begin
            q_w_en <= 0; k_w_en <= 0; score_w_en <= 0;
            case (state)
                S_IDLE: if (valid_q || valid_k) state <= S_LOAD_Q;
                S_LOAD_Q: begin
                    if (valid_q) begin q_w_en <= 1; q_write_addr <= q_write_addr + 1; end
                    if (q_write_addr == d-1) state <= S_LOAD_K;
                end
                S_LOAD_K: begin
                    if (valid_k) begin k_w_en <= 1; k_write_addr <= k_write_addr + 1; end
                    if (k_write_addr == N*d-1) state <= S_COMPUTE;
                end
                S_COMPUTE: begin
                    // Dual-identity: feed vector A then vector B per DPE pass
                    q_read_addr <= mac_count;
                    k_read_addr_a <= (score_idx << $clog2(d)) + mac_count;
                    k_read_addr_b <= ((score_idx + 1) << $clog2(d)) + mac_count;
                    if (!feed_half) begin
                        // Feeding vector A (log_sum_a)
                        feed_phase <= 0; acc_phase <= 0;
                        mac_count <= mac_count + 1;
                        if (mac_count == d-1) begin
                            mac_count <= 0;
                            feed_half <= 1;
                        end
                    end else begin
                        // Feeding vector B (log_sum_b)
                        feed_phase <= 1; acc_phase <= 1;
                        mac_count <= mac_count + 1;
                        if (mac_count == d-1) begin
                            // Both vectors done — write score A, then B
                            score_write_data <= accumulator_a[2*DATA_WIDTH-1:DATA_WIDTH];
                            score_w_en <= 1;
                            score_write_addr <= score_idx;
                            mac_count <= 0;
                            feed_half <= 0;
                            state <= S_WRITE_B;
                        end
                    end
                end
                S_WRITE_B: begin
                    // Write second score (j₂ = score_idx + 1)
                    score_write_data <= accumulator_b[2*DATA_WIDTH-1:DATA_WIDTH];
                    score_w_en <= 1;
                    score_write_addr <= score_idx + 1;
                    accumulator_a <= 0; accumulator_b <= 0;
                    if (score_idx >= N-2) state <= S_OUTPUT;
                    else begin score_idx <= score_idx + 2; state <= S_COMPUTE; end
                end
                S_OUTPUT: if (ready_n) score_read_addr <= score_read_addr + 1;
            endcase
        end
    end

    assign data_out = score_sram_out;
    assign valid_n = (state == S_OUTPUT);
    assign ready_q = (state == S_LOAD_Q || state == S_IDLE);
    assign ready_k = (state == S_LOAD_K || state == S_IDLE);
endmodule
module softmax_approx_b0_h1 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid, input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, output wire valid_n
);

    reg [ADDR_WIDTH-1:0] in_write_addr, in_read_addr;
    reg in_w_en;
    wire [DATA_WIDTH-1:0] in_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    in_sram (.clk(clk),.rst(rst),.w_en(in_w_en),.r_addr(in_read_addr),
             .w_addr(in_write_addr),.sram_data_in(data_in),.sram_data_out(in_sram_out));

    wire sm_exp_0_valid, sm_exp_0_ready_n, sm_exp_0_ready, sm_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_0_data_in, sm_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_0_valid),
        .ready_n(sm_exp_0_ready_n),
        .data_in(sm_exp_0_data_in),
        .data_out(sm_exp_0_data_out),
        .ready(sm_exp_0_ready),
        .valid_n(sm_exp_0_valid_n)
    );

    wire sm_exp_1_valid, sm_exp_1_ready_n, sm_exp_1_ready, sm_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_1_data_in, sm_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_1_valid),
        .ready_n(sm_exp_1_ready_n),
        .data_in(sm_exp_1_data_in),
        .data_out(sm_exp_1_data_out),
        .ready(sm_exp_1_ready),
        .valid_n(sm_exp_1_valid_n)
    );

    wire sm_exp_2_valid, sm_exp_2_ready_n, sm_exp_2_ready, sm_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_2_data_in, sm_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_2_valid),
        .ready_n(sm_exp_2_ready_n),
        .data_in(sm_exp_2_data_in),
        .data_out(sm_exp_2_data_out),
        .ready(sm_exp_2_ready),
        .valid_n(sm_exp_2_valid_n)
    );

    wire sm_exp_3_valid, sm_exp_3_ready_n, sm_exp_3_ready, sm_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_3_data_in, sm_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_3_valid),
        .ready_n(sm_exp_3_ready_n),
        .data_in(sm_exp_3_data_in),
        .data_out(sm_exp_3_data_out),
        .ready(sm_exp_3_ready),
        .valid_n(sm_exp_3_valid_n)
    );

    wire sm_exp_4_valid, sm_exp_4_ready_n, sm_exp_4_ready, sm_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_4_data_in, sm_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_4_valid),
        .ready_n(sm_exp_4_ready_n),
        .data_in(sm_exp_4_data_in),
        .data_out(sm_exp_4_data_out),
        .ready(sm_exp_4_ready),
        .valid_n(sm_exp_4_valid_n)
    );

    wire sm_exp_5_valid, sm_exp_5_ready_n, sm_exp_5_ready, sm_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_5_data_in, sm_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_5_valid),
        .ready_n(sm_exp_5_ready_n),
        .data_in(sm_exp_5_data_in),
        .data_out(sm_exp_5_data_out),
        .ready(sm_exp_5_ready),
        .valid_n(sm_exp_5_valid_n)
    );

    wire sm_exp_6_valid, sm_exp_6_ready_n, sm_exp_6_ready, sm_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_6_data_in, sm_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_6_valid),
        .ready_n(sm_exp_6_ready_n),
        .data_in(sm_exp_6_data_in),
        .data_out(sm_exp_6_data_out),
        .ready(sm_exp_6_ready),
        .valid_n(sm_exp_6_valid_n)
    );

    wire sm_exp_7_valid, sm_exp_7_ready_n, sm_exp_7_ready, sm_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_7_data_in, sm_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_7_valid),
        .ready_n(sm_exp_7_ready_n),
        .data_in(sm_exp_7_data_in),
        .data_out(sm_exp_7_data_out),
        .ready(sm_exp_7_ready),
        .valid_n(sm_exp_7_valid_n)
    );

    wire sm_exp_8_valid, sm_exp_8_ready_n, sm_exp_8_ready, sm_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_8_data_in, sm_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_8_valid),
        .ready_n(sm_exp_8_ready_n),
        .data_in(sm_exp_8_data_in),
        .data_out(sm_exp_8_data_out),
        .ready(sm_exp_8_ready),
        .valid_n(sm_exp_8_valid_n)
    );

    wire sm_exp_9_valid, sm_exp_9_ready_n, sm_exp_9_ready, sm_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_9_data_in, sm_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_9_valid),
        .ready_n(sm_exp_9_ready_n),
        .data_in(sm_exp_9_data_in),
        .data_out(sm_exp_9_data_out),
        .ready(sm_exp_9_ready),
        .valid_n(sm_exp_9_valid_n)
    );

    wire sm_exp_10_valid, sm_exp_10_ready_n, sm_exp_10_ready, sm_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_10_data_in, sm_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_10_valid),
        .ready_n(sm_exp_10_ready_n),
        .data_in(sm_exp_10_data_in),
        .data_out(sm_exp_10_data_out),
        .ready(sm_exp_10_ready),
        .valid_n(sm_exp_10_valid_n)
    );

    wire sm_exp_11_valid, sm_exp_11_ready_n, sm_exp_11_ready, sm_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_11_data_in, sm_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_11_valid),
        .ready_n(sm_exp_11_ready_n),
        .data_in(sm_exp_11_data_in),
        .data_out(sm_exp_11_data_out),
        .ready(sm_exp_11_ready),
        .valid_n(sm_exp_11_valid_n)
    );

    wire sm_exp_12_valid, sm_exp_12_ready_n, sm_exp_12_ready, sm_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_12_data_in, sm_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_12_valid),
        .ready_n(sm_exp_12_ready_n),
        .data_in(sm_exp_12_data_in),
        .data_out(sm_exp_12_data_out),
        .ready(sm_exp_12_ready),
        .valid_n(sm_exp_12_valid_n)
    );

    wire sm_exp_13_valid, sm_exp_13_ready_n, sm_exp_13_ready, sm_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_13_data_in, sm_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_13_valid),
        .ready_n(sm_exp_13_ready_n),
        .data_in(sm_exp_13_data_in),
        .data_out(sm_exp_13_data_out),
        .ready(sm_exp_13_ready),
        .valid_n(sm_exp_13_valid_n)
    );

    wire sm_exp_14_valid, sm_exp_14_ready_n, sm_exp_14_ready, sm_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_14_data_in, sm_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_14_valid),
        .ready_n(sm_exp_14_ready_n),
        .data_in(sm_exp_14_data_in),
        .data_out(sm_exp_14_data_out),
        .ready(sm_exp_14_ready),
        .valid_n(sm_exp_14_valid_n)
    );

    wire sm_exp_15_valid, sm_exp_15_ready_n, sm_exp_15_ready, sm_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_15_data_in, sm_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_15_valid),
        .ready_n(sm_exp_15_ready_n),
        .data_in(sm_exp_15_data_in),
        .data_out(sm_exp_15_data_out),
        .ready(sm_exp_15_ready),
        .valid_n(sm_exp_15_valid_n)
    );

    wire sm_exp_16_valid, sm_exp_16_ready_n, sm_exp_16_ready, sm_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_16_data_in, sm_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_16_valid),
        .ready_n(sm_exp_16_ready_n),
        .data_in(sm_exp_16_data_in),
        .data_out(sm_exp_16_data_out),
        .ready(sm_exp_16_ready),
        .valid_n(sm_exp_16_valid_n)
    );

    wire sm_exp_17_valid, sm_exp_17_ready_n, sm_exp_17_ready, sm_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_17_data_in, sm_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_17_valid),
        .ready_n(sm_exp_17_ready_n),
        .data_in(sm_exp_17_data_in),
        .data_out(sm_exp_17_data_out),
        .ready(sm_exp_17_ready),
        .valid_n(sm_exp_17_valid_n)
    );

    wire sm_exp_18_valid, sm_exp_18_ready_n, sm_exp_18_ready, sm_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_18_data_in, sm_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_18_valid),
        .ready_n(sm_exp_18_ready_n),
        .data_in(sm_exp_18_data_in),
        .data_out(sm_exp_18_data_out),
        .ready(sm_exp_18_ready),
        .valid_n(sm_exp_18_valid_n)
    );

    wire sm_exp_19_valid, sm_exp_19_ready_n, sm_exp_19_ready, sm_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_19_data_in, sm_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_19_valid),
        .ready_n(sm_exp_19_ready_n),
        .data_in(sm_exp_19_data_in),
        .data_out(sm_exp_19_data_out),
        .ready(sm_exp_19_ready),
        .valid_n(sm_exp_19_valid_n)
    );

    wire sm_exp_20_valid, sm_exp_20_ready_n, sm_exp_20_ready, sm_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_20_data_in, sm_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_20_valid),
        .ready_n(sm_exp_20_ready_n),
        .data_in(sm_exp_20_data_in),
        .data_out(sm_exp_20_data_out),
        .ready(sm_exp_20_ready),
        .valid_n(sm_exp_20_valid_n)
    );

    wire sm_exp_21_valid, sm_exp_21_ready_n, sm_exp_21_ready, sm_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_21_data_in, sm_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_21_valid),
        .ready_n(sm_exp_21_ready_n),
        .data_in(sm_exp_21_data_in),
        .data_out(sm_exp_21_data_out),
        .ready(sm_exp_21_ready),
        .valid_n(sm_exp_21_valid_n)
    );

    wire sm_exp_22_valid, sm_exp_22_ready_n, sm_exp_22_ready, sm_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_22_data_in, sm_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_22_valid),
        .ready_n(sm_exp_22_ready_n),
        .data_in(sm_exp_22_data_in),
        .data_out(sm_exp_22_data_out),
        .ready(sm_exp_22_ready),
        .valid_n(sm_exp_22_valid_n)
    );

    wire sm_exp_23_valid, sm_exp_23_ready_n, sm_exp_23_ready, sm_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_23_data_in, sm_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_23_valid),
        .ready_n(sm_exp_23_ready_n),
        .data_in(sm_exp_23_data_in),
        .data_out(sm_exp_23_data_out),
        .ready(sm_exp_23_ready),
        .valid_n(sm_exp_23_valid_n)
    );

    assign sm_exp_0_data_in = in_sram_out;
    assign sm_exp_0_valid = (sm_state == SM_EXP);
    assign sm_exp_0_ready_n = 1'b0;
    assign sm_exp_1_data_in = in_sram_out;
    assign sm_exp_1_valid = (sm_state == SM_EXP);
    assign sm_exp_1_ready_n = 1'b0;
    assign sm_exp_2_data_in = in_sram_out;
    assign sm_exp_2_valid = (sm_state == SM_EXP);
    assign sm_exp_2_ready_n = 1'b0;
    assign sm_exp_3_data_in = in_sram_out;
    assign sm_exp_3_valid = (sm_state == SM_EXP);
    assign sm_exp_3_ready_n = 1'b0;
    assign sm_exp_4_data_in = in_sram_out;
    assign sm_exp_4_valid = (sm_state == SM_EXP);
    assign sm_exp_4_ready_n = 1'b0;
    assign sm_exp_5_data_in = in_sram_out;
    assign sm_exp_5_valid = (sm_state == SM_EXP);
    assign sm_exp_5_ready_n = 1'b0;
    assign sm_exp_6_data_in = in_sram_out;
    assign sm_exp_6_valid = (sm_state == SM_EXP);
    assign sm_exp_6_ready_n = 1'b0;
    assign sm_exp_7_data_in = in_sram_out;
    assign sm_exp_7_valid = (sm_state == SM_EXP);
    assign sm_exp_7_ready_n = 1'b0;
    assign sm_exp_8_data_in = in_sram_out;
    assign sm_exp_8_valid = (sm_state == SM_EXP);
    assign sm_exp_8_ready_n = 1'b0;
    assign sm_exp_9_data_in = in_sram_out;
    assign sm_exp_9_valid = (sm_state == SM_EXP);
    assign sm_exp_9_ready_n = 1'b0;
    assign sm_exp_10_data_in = in_sram_out;
    assign sm_exp_10_valid = (sm_state == SM_EXP);
    assign sm_exp_10_ready_n = 1'b0;
    assign sm_exp_11_data_in = in_sram_out;
    assign sm_exp_11_valid = (sm_state == SM_EXP);
    assign sm_exp_11_ready_n = 1'b0;
    assign sm_exp_12_data_in = in_sram_out;
    assign sm_exp_12_valid = (sm_state == SM_EXP);
    assign sm_exp_12_ready_n = 1'b0;
    assign sm_exp_13_data_in = in_sram_out;
    assign sm_exp_13_valid = (sm_state == SM_EXP);
    assign sm_exp_13_ready_n = 1'b0;
    assign sm_exp_14_data_in = in_sram_out;
    assign sm_exp_14_valid = (sm_state == SM_EXP);
    assign sm_exp_14_ready_n = 1'b0;
    assign sm_exp_15_data_in = in_sram_out;
    assign sm_exp_15_valid = (sm_state == SM_EXP);
    assign sm_exp_15_ready_n = 1'b0;
    assign sm_exp_16_data_in = in_sram_out;
    assign sm_exp_16_valid = (sm_state == SM_EXP);
    assign sm_exp_16_ready_n = 1'b0;
    assign sm_exp_17_data_in = in_sram_out;
    assign sm_exp_17_valid = (sm_state == SM_EXP);
    assign sm_exp_17_ready_n = 1'b0;
    assign sm_exp_18_data_in = in_sram_out;
    assign sm_exp_18_valid = (sm_state == SM_EXP);
    assign sm_exp_18_ready_n = 1'b0;
    assign sm_exp_19_data_in = in_sram_out;
    assign sm_exp_19_valid = (sm_state == SM_EXP);
    assign sm_exp_19_ready_n = 1'b0;
    assign sm_exp_20_data_in = in_sram_out;
    assign sm_exp_20_valid = (sm_state == SM_EXP);
    assign sm_exp_20_ready_n = 1'b0;
    assign sm_exp_21_data_in = in_sram_out;
    assign sm_exp_21_valid = (sm_state == SM_EXP);
    assign sm_exp_21_ready_n = 1'b0;
    assign sm_exp_22_data_in = in_sram_out;
    assign sm_exp_22_valid = (sm_state == SM_EXP);
    assign sm_exp_22_ready_n = 1'b0;
    assign sm_exp_23_data_in = in_sram_out;
    assign sm_exp_23_valid = (sm_state == SM_EXP);
    assign sm_exp_23_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] exp_write_addr, exp_read_addr;
    reg exp_w_en;
    reg [DATA_WIDTH-1:0] exp_write_data;
    wire [DATA_WIDTH-1:0] exp_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    exp_sram (.clk(clk),.rst(rst),.w_en(exp_w_en),.r_addr(exp_read_addr),
              .w_addr(exp_write_addr),.sram_data_in(exp_write_data),.sram_data_out(exp_sram_out));
    reg [2*DATA_WIDTH-1:0] exp_sum;

    always @(posedge clk) begin
        if (rst) begin exp_w_en <= 0; exp_write_addr <= 0; exp_sum <= 0; end
        else if (sm_exp_0_valid_n) begin
            exp_write_data <= sm_exp_0_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_0_data_out;
        end else exp_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] exp_sum_upper = exp_sum[2*DATA_WIDTH-1:DATA_WIDTH];
    reg [DATA_WIDTH-1:0] recip_val;
    always @(*) begin
        casez (exp_sum_upper[15:0])
            16'b1???????????????: recip_val = 40'd2;
            16'b01??????????????: recip_val = 40'd32768;
            16'b001?????????????: recip_val = 40'd16384;
            16'b0001????????????: recip_val = 40'd8192;
            16'b00001???????????: recip_val = 40'd4096;
            16'b000001??????????: recip_val = 40'd2048;
            16'b0000001?????????: recip_val = 40'd1024;
            16'b00000001????????: recip_val = 40'd512;
            16'b000000001???????: recip_val = 40'd256;
            16'b0000000001??????: recip_val = 40'd128;
            16'b00000000001?????: recip_val = 40'd64;
            16'b000000000001????: recip_val = 40'd32;
            16'b0000000000001???: recip_val = 40'd16;
            16'b00000000000001??: recip_val = 40'd8;
            16'b000000000000001?: recip_val = 40'd4;
            16'b0000000000000001: recip_val = 40'd2;
            default: recip_val = 40'hFFFF;
        endcase
    end

    reg [DATA_WIDTH-1:0] inv_sum;
    reg [2*DATA_WIDTH-1:0] norm_product;
    reg [DATA_WIDTH-1:0] norm_val;

    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    wire [DATA_WIDTH-1:0] out_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),
              .w_addr(out_write_addr),.sram_data_in(norm_val),.sram_data_out(out_sram_out));

    localparam SM_IDLE = 3'd0, SM_LOAD = 3'd1, SM_EXP = 3'd2,
               SM_NORMALIZE = 3'd3, SM_OUTPUT = 3'd4;
    reg [2:0] sm_state;
    reg [$clog2(N)-1:0] sm_count;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sm_state <= SM_IDLE; sm_count <= 0;
            in_write_addr <= 0; in_read_addr <= 0; in_w_en <= 0;
            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;
            inv_sum <= 0; norm_product <= 0; norm_val <= 0;
        end else begin
            in_w_en <= 0; out_w_en <= 0;
            case (sm_state)
                SM_IDLE: if (valid) sm_state <= SM_LOAD;
                SM_LOAD: begin
                    if (valid) begin in_w_en <= 1; in_write_addr <= in_write_addr + 1; end
                    if (in_write_addr == N-1) sm_state <= SM_EXP;
                end
                SM_EXP: begin
                    // DPE(I|exp) processes scores — output goes to exp_sram + sum
                    in_read_addr <= sm_count; sm_count <= sm_count + 1;
                    if (sm_count == N-1) begin
                        inv_sum <= recip_val; sm_count <= 0;
                        sm_state <= SM_NORMALIZE;
                    end
                end
                SM_NORMALIZE: begin
                    // CLB multiply: exp_val × inv_sum
                    exp_read_addr <= sm_count;
                    norm_product <= exp_sram_out * inv_sum;
                    norm_val <= norm_product[2*DATA_WIDTH-1:DATA_WIDTH];
                    out_w_en <= (sm_count > 1);
                    out_write_addr <= sm_count - 2;
                    sm_count <= sm_count + 1;
                    if (sm_count == N+1) sm_state <= SM_OUTPUT;
                end
                SM_OUTPUT: if (!ready_n) out_read_addr <= out_read_addr + 1;
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (sm_state == SM_OUTPUT);
    assign ready = (sm_state == SM_LOAD || sm_state == SM_IDLE);
endmodule
module dimm_weighted_sum_b0_h1 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid_attn, input wire valid_v,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_attn,
    input wire [DATA_WIDTH-1:0] data_in_v,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_attn, output wire ready_v,
    output wire valid_n
);

    reg [ADDR_WIDTH-1:0] attn_write_addr, attn_read_addr;
    reg attn_w_en;
    wire [DATA_WIDTH-1:0] attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    attn_sram (.clk(clk),.rst(rst),.w_en(attn_w_en),.r_addr(attn_read_addr),
               .w_addr(attn_write_addr),.sram_data_in(data_in_attn),.sram_data_out(attn_sram_out));

    reg [$clog2(N*d)-1:0] v_write_addr, v_read_addr;
    reg v_w_en;
    wire [DATA_WIDTH-1:0] v_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    v_sram (.clk(clk),.rst(rst),.w_en(v_w_en),.r_addr(v_read_addr[$clog2(N*d)-1:0]),
            .w_addr(v_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_v),.sram_data_out(v_sram_out));

    wire ws_log_0_valid, ws_log_0_ready_n, ws_log_0_ready, ws_log_0_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_0_data_in, ws_log_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_0 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_0_valid),
        .ready_n(ws_log_0_ready_n),
        .data_in(ws_log_0_data_in),
        .data_out(ws_log_0_data_out),
        .ready(ws_log_0_ready),
        .valid_n(ws_log_0_valid_n)
    );

    wire ws_log_1_valid, ws_log_1_ready_n, ws_log_1_ready, ws_log_1_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_1_data_in, ws_log_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_1 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_1_valid),
        .ready_n(ws_log_1_ready_n),
        .data_in(ws_log_1_data_in),
        .data_out(ws_log_1_data_out),
        .ready(ws_log_1_ready),
        .valid_n(ws_log_1_valid_n)
    );

    wire ws_log_2_valid, ws_log_2_ready_n, ws_log_2_ready, ws_log_2_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_2_data_in, ws_log_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_2 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_2_valid),
        .ready_n(ws_log_2_ready_n),
        .data_in(ws_log_2_data_in),
        .data_out(ws_log_2_data_out),
        .ready(ws_log_2_ready),
        .valid_n(ws_log_2_valid_n)
    );

    wire ws_log_3_valid, ws_log_3_ready_n, ws_log_3_ready, ws_log_3_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_3_data_in, ws_log_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_3 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_3_valid),
        .ready_n(ws_log_3_ready_n),
        .data_in(ws_log_3_data_in),
        .data_out(ws_log_3_data_out),
        .ready(ws_log_3_ready),
        .valid_n(ws_log_3_valid_n)
    );

    wire ws_log_4_valid, ws_log_4_ready_n, ws_log_4_ready, ws_log_4_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_4_data_in, ws_log_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_4 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_4_valid),
        .ready_n(ws_log_4_ready_n),
        .data_in(ws_log_4_data_in),
        .data_out(ws_log_4_data_out),
        .ready(ws_log_4_ready),
        .valid_n(ws_log_4_valid_n)
    );

    wire ws_log_5_valid, ws_log_5_ready_n, ws_log_5_ready, ws_log_5_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_5_data_in, ws_log_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_5 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_5_valid),
        .ready_n(ws_log_5_ready_n),
        .data_in(ws_log_5_data_in),
        .data_out(ws_log_5_data_out),
        .ready(ws_log_5_ready),
        .valid_n(ws_log_5_valid_n)
    );

    wire ws_log_6_valid, ws_log_6_ready_n, ws_log_6_ready, ws_log_6_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_6_data_in, ws_log_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_6 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_6_valid),
        .ready_n(ws_log_6_ready_n),
        .data_in(ws_log_6_data_in),
        .data_out(ws_log_6_data_out),
        .ready(ws_log_6_ready),
        .valid_n(ws_log_6_valid_n)
    );

    wire ws_log_7_valid, ws_log_7_ready_n, ws_log_7_ready, ws_log_7_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_7_data_in, ws_log_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_7 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_7_valid),
        .ready_n(ws_log_7_ready_n),
        .data_in(ws_log_7_data_in),
        .data_out(ws_log_7_data_out),
        .ready(ws_log_7_ready),
        .valid_n(ws_log_7_valid_n)
    );

    wire ws_log_8_valid, ws_log_8_ready_n, ws_log_8_ready, ws_log_8_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_8_data_in, ws_log_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_8 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_8_valid),
        .ready_n(ws_log_8_ready_n),
        .data_in(ws_log_8_data_in),
        .data_out(ws_log_8_data_out),
        .ready(ws_log_8_ready),
        .valid_n(ws_log_8_valid_n)
    );

    wire ws_log_9_valid, ws_log_9_ready_n, ws_log_9_ready, ws_log_9_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_9_data_in, ws_log_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_9 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_9_valid),
        .ready_n(ws_log_9_ready_n),
        .data_in(ws_log_9_data_in),
        .data_out(ws_log_9_data_out),
        .ready(ws_log_9_ready),
        .valid_n(ws_log_9_valid_n)
    );

    wire ws_log_10_valid, ws_log_10_ready_n, ws_log_10_ready, ws_log_10_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_10_data_in, ws_log_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_10 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_10_valid),
        .ready_n(ws_log_10_ready_n),
        .data_in(ws_log_10_data_in),
        .data_out(ws_log_10_data_out),
        .ready(ws_log_10_ready),
        .valid_n(ws_log_10_valid_n)
    );

    wire ws_log_11_valid, ws_log_11_ready_n, ws_log_11_ready, ws_log_11_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_11_data_in, ws_log_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_11 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_11_valid),
        .ready_n(ws_log_11_ready_n),
        .data_in(ws_log_11_data_in),
        .data_out(ws_log_11_data_out),
        .ready(ws_log_11_ready),
        .valid_n(ws_log_11_valid_n)
    );

    wire ws_log_12_valid, ws_log_12_ready_n, ws_log_12_ready, ws_log_12_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_12_data_in, ws_log_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_12 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_12_valid),
        .ready_n(ws_log_12_ready_n),
        .data_in(ws_log_12_data_in),
        .data_out(ws_log_12_data_out),
        .ready(ws_log_12_ready),
        .valid_n(ws_log_12_valid_n)
    );

    wire ws_log_13_valid, ws_log_13_ready_n, ws_log_13_ready, ws_log_13_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_13_data_in, ws_log_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_13 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_13_valid),
        .ready_n(ws_log_13_ready_n),
        .data_in(ws_log_13_data_in),
        .data_out(ws_log_13_data_out),
        .ready(ws_log_13_ready),
        .valid_n(ws_log_13_valid_n)
    );

    wire ws_log_14_valid, ws_log_14_ready_n, ws_log_14_ready, ws_log_14_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_14_data_in, ws_log_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_14 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_14_valid),
        .ready_n(ws_log_14_ready_n),
        .data_in(ws_log_14_data_in),
        .data_out(ws_log_14_data_out),
        .ready(ws_log_14_ready),
        .valid_n(ws_log_14_valid_n)
    );

    wire ws_log_15_valid, ws_log_15_ready_n, ws_log_15_ready, ws_log_15_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_15_data_in, ws_log_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_15 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_15_valid),
        .ready_n(ws_log_15_ready_n),
        .data_in(ws_log_15_data_in),
        .data_out(ws_log_15_data_out),
        .ready(ws_log_15_ready),
        .valid_n(ws_log_15_valid_n)
    );

    wire ws_log_16_valid, ws_log_16_ready_n, ws_log_16_ready, ws_log_16_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_16_data_in, ws_log_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_16 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_16_valid),
        .ready_n(ws_log_16_ready_n),
        .data_in(ws_log_16_data_in),
        .data_out(ws_log_16_data_out),
        .ready(ws_log_16_ready),
        .valid_n(ws_log_16_valid_n)
    );

    wire ws_log_17_valid, ws_log_17_ready_n, ws_log_17_ready, ws_log_17_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_17_data_in, ws_log_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_17 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_17_valid),
        .ready_n(ws_log_17_ready_n),
        .data_in(ws_log_17_data_in),
        .data_out(ws_log_17_data_out),
        .ready(ws_log_17_ready),
        .valid_n(ws_log_17_valid_n)
    );

    wire ws_log_18_valid, ws_log_18_ready_n, ws_log_18_ready, ws_log_18_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_18_data_in, ws_log_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_18 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_18_valid),
        .ready_n(ws_log_18_ready_n),
        .data_in(ws_log_18_data_in),
        .data_out(ws_log_18_data_out),
        .ready(ws_log_18_ready),
        .valid_n(ws_log_18_valid_n)
    );

    wire ws_log_19_valid, ws_log_19_ready_n, ws_log_19_ready, ws_log_19_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_19_data_in, ws_log_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_19 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_19_valid),
        .ready_n(ws_log_19_ready_n),
        .data_in(ws_log_19_data_in),
        .data_out(ws_log_19_data_out),
        .ready(ws_log_19_ready),
        .valid_n(ws_log_19_valid_n)
    );

    wire ws_log_20_valid, ws_log_20_ready_n, ws_log_20_ready, ws_log_20_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_20_data_in, ws_log_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_20 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_20_valid),
        .ready_n(ws_log_20_ready_n),
        .data_in(ws_log_20_data_in),
        .data_out(ws_log_20_data_out),
        .ready(ws_log_20_ready),
        .valid_n(ws_log_20_valid_n)
    );

    wire ws_log_21_valid, ws_log_21_ready_n, ws_log_21_ready, ws_log_21_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_21_data_in, ws_log_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_21 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_21_valid),
        .ready_n(ws_log_21_ready_n),
        .data_in(ws_log_21_data_in),
        .data_out(ws_log_21_data_out),
        .ready(ws_log_21_ready),
        .valid_n(ws_log_21_valid_n)
    );

    wire ws_log_22_valid, ws_log_22_ready_n, ws_log_22_ready, ws_log_22_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_22_data_in, ws_log_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_22 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_22_valid),
        .ready_n(ws_log_22_ready_n),
        .data_in(ws_log_22_data_in),
        .data_out(ws_log_22_data_out),
        .ready(ws_log_22_ready),
        .valid_n(ws_log_22_valid_n)
    );

    wire ws_log_23_valid, ws_log_23_ready_n, ws_log_23_ready, ws_log_23_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_23_data_in, ws_log_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_23 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_23_valid),
        .ready_n(ws_log_23_ready_n),
        .data_in(ws_log_23_data_in),
        .data_out(ws_log_23_data_out),
        .ready(ws_log_23_ready),
        .valid_n(ws_log_23_valid_n)
    );

    assign ws_log_0_data_in = attn_sram_out;
    assign ws_log_0_valid = (ws_state == WS_LOG);
    assign ws_log_0_ready_n = 1'b0;
    assign ws_log_1_data_in = attn_sram_out;
    assign ws_log_1_valid = (ws_state == WS_LOG);
    assign ws_log_1_ready_n = 1'b0;
    assign ws_log_2_data_in = attn_sram_out;
    assign ws_log_2_valid = (ws_state == WS_LOG);
    assign ws_log_2_ready_n = 1'b0;
    assign ws_log_3_data_in = attn_sram_out;
    assign ws_log_3_valid = (ws_state == WS_LOG);
    assign ws_log_3_ready_n = 1'b0;
    assign ws_log_4_data_in = attn_sram_out;
    assign ws_log_4_valid = (ws_state == WS_LOG);
    assign ws_log_4_ready_n = 1'b0;
    assign ws_log_5_data_in = attn_sram_out;
    assign ws_log_5_valid = (ws_state == WS_LOG);
    assign ws_log_5_ready_n = 1'b0;
    assign ws_log_6_data_in = attn_sram_out;
    assign ws_log_6_valid = (ws_state == WS_LOG);
    assign ws_log_6_ready_n = 1'b0;
    assign ws_log_7_data_in = attn_sram_out;
    assign ws_log_7_valid = (ws_state == WS_LOG);
    assign ws_log_7_ready_n = 1'b0;
    assign ws_log_8_data_in = attn_sram_out;
    assign ws_log_8_valid = (ws_state == WS_LOG);
    assign ws_log_8_ready_n = 1'b0;
    assign ws_log_9_data_in = attn_sram_out;
    assign ws_log_9_valid = (ws_state == WS_LOG);
    assign ws_log_9_ready_n = 1'b0;
    assign ws_log_10_data_in = attn_sram_out;
    assign ws_log_10_valid = (ws_state == WS_LOG);
    assign ws_log_10_ready_n = 1'b0;
    assign ws_log_11_data_in = attn_sram_out;
    assign ws_log_11_valid = (ws_state == WS_LOG);
    assign ws_log_11_ready_n = 1'b0;
    assign ws_log_12_data_in = attn_sram_out;
    assign ws_log_12_valid = (ws_state == WS_LOG);
    assign ws_log_12_ready_n = 1'b0;
    assign ws_log_13_data_in = attn_sram_out;
    assign ws_log_13_valid = (ws_state == WS_LOG);
    assign ws_log_13_ready_n = 1'b0;
    assign ws_log_14_data_in = attn_sram_out;
    assign ws_log_14_valid = (ws_state == WS_LOG);
    assign ws_log_14_ready_n = 1'b0;
    assign ws_log_15_data_in = attn_sram_out;
    assign ws_log_15_valid = (ws_state == WS_LOG);
    assign ws_log_15_ready_n = 1'b0;
    assign ws_log_16_data_in = attn_sram_out;
    assign ws_log_16_valid = (ws_state == WS_LOG);
    assign ws_log_16_ready_n = 1'b0;
    assign ws_log_17_data_in = attn_sram_out;
    assign ws_log_17_valid = (ws_state == WS_LOG);
    assign ws_log_17_ready_n = 1'b0;
    assign ws_log_18_data_in = attn_sram_out;
    assign ws_log_18_valid = (ws_state == WS_LOG);
    assign ws_log_18_ready_n = 1'b0;
    assign ws_log_19_data_in = attn_sram_out;
    assign ws_log_19_valid = (ws_state == WS_LOG);
    assign ws_log_19_ready_n = 1'b0;
    assign ws_log_20_data_in = attn_sram_out;
    assign ws_log_20_valid = (ws_state == WS_LOG);
    assign ws_log_20_ready_n = 1'b0;
    assign ws_log_21_data_in = attn_sram_out;
    assign ws_log_21_valid = (ws_state == WS_LOG);
    assign ws_log_21_ready_n = 1'b0;
    assign ws_log_22_data_in = attn_sram_out;
    assign ws_log_22_valid = (ws_state == WS_LOG);
    assign ws_log_22_ready_n = 1'b0;
    assign ws_log_23_data_in = attn_sram_out;
    assign ws_log_23_valid = (ws_state == WS_LOG);
    assign ws_log_23_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_0_valid_n) begin
            log_attn_write_data <= ws_log_0_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_0_valid, ws_exp_0_ready_n, ws_exp_0_ready, ws_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_0_data_in, ws_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_0_valid),
        .ready_n(ws_exp_0_ready_n),
        .data_in(ws_exp_0_data_in),
        .data_out(ws_exp_0_data_out),
        .ready(ws_exp_0_ready),
        .valid_n(ws_exp_0_valid_n)
    );

    wire ws_exp_1_valid, ws_exp_1_ready_n, ws_exp_1_ready, ws_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_1_data_in, ws_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_1_valid),
        .ready_n(ws_exp_1_ready_n),
        .data_in(ws_exp_1_data_in),
        .data_out(ws_exp_1_data_out),
        .ready(ws_exp_1_ready),
        .valid_n(ws_exp_1_valid_n)
    );

    wire ws_exp_2_valid, ws_exp_2_ready_n, ws_exp_2_ready, ws_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_2_data_in, ws_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_2_valid),
        .ready_n(ws_exp_2_ready_n),
        .data_in(ws_exp_2_data_in),
        .data_out(ws_exp_2_data_out),
        .ready(ws_exp_2_ready),
        .valid_n(ws_exp_2_valid_n)
    );

    wire ws_exp_3_valid, ws_exp_3_ready_n, ws_exp_3_ready, ws_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_3_data_in, ws_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_3_valid),
        .ready_n(ws_exp_3_ready_n),
        .data_in(ws_exp_3_data_in),
        .data_out(ws_exp_3_data_out),
        .ready(ws_exp_3_ready),
        .valid_n(ws_exp_3_valid_n)
    );

    wire ws_exp_4_valid, ws_exp_4_ready_n, ws_exp_4_ready, ws_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_4_data_in, ws_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_4_valid),
        .ready_n(ws_exp_4_ready_n),
        .data_in(ws_exp_4_data_in),
        .data_out(ws_exp_4_data_out),
        .ready(ws_exp_4_ready),
        .valid_n(ws_exp_4_valid_n)
    );

    wire ws_exp_5_valid, ws_exp_5_ready_n, ws_exp_5_ready, ws_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_5_data_in, ws_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_5_valid),
        .ready_n(ws_exp_5_ready_n),
        .data_in(ws_exp_5_data_in),
        .data_out(ws_exp_5_data_out),
        .ready(ws_exp_5_ready),
        .valid_n(ws_exp_5_valid_n)
    );

    wire ws_exp_6_valid, ws_exp_6_ready_n, ws_exp_6_ready, ws_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_6_data_in, ws_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_6_valid),
        .ready_n(ws_exp_6_ready_n),
        .data_in(ws_exp_6_data_in),
        .data_out(ws_exp_6_data_out),
        .ready(ws_exp_6_ready),
        .valid_n(ws_exp_6_valid_n)
    );

    wire ws_exp_7_valid, ws_exp_7_ready_n, ws_exp_7_ready, ws_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_7_data_in, ws_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_7_valid),
        .ready_n(ws_exp_7_ready_n),
        .data_in(ws_exp_7_data_in),
        .data_out(ws_exp_7_data_out),
        .ready(ws_exp_7_ready),
        .valid_n(ws_exp_7_valid_n)
    );

    wire ws_exp_8_valid, ws_exp_8_ready_n, ws_exp_8_ready, ws_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_8_data_in, ws_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_8_valid),
        .ready_n(ws_exp_8_ready_n),
        .data_in(ws_exp_8_data_in),
        .data_out(ws_exp_8_data_out),
        .ready(ws_exp_8_ready),
        .valid_n(ws_exp_8_valid_n)
    );

    wire ws_exp_9_valid, ws_exp_9_ready_n, ws_exp_9_ready, ws_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_9_data_in, ws_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_9_valid),
        .ready_n(ws_exp_9_ready_n),
        .data_in(ws_exp_9_data_in),
        .data_out(ws_exp_9_data_out),
        .ready(ws_exp_9_ready),
        .valid_n(ws_exp_9_valid_n)
    );

    wire ws_exp_10_valid, ws_exp_10_ready_n, ws_exp_10_ready, ws_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_10_data_in, ws_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_10_valid),
        .ready_n(ws_exp_10_ready_n),
        .data_in(ws_exp_10_data_in),
        .data_out(ws_exp_10_data_out),
        .ready(ws_exp_10_ready),
        .valid_n(ws_exp_10_valid_n)
    );

    wire ws_exp_11_valid, ws_exp_11_ready_n, ws_exp_11_ready, ws_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_11_data_in, ws_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_11_valid),
        .ready_n(ws_exp_11_ready_n),
        .data_in(ws_exp_11_data_in),
        .data_out(ws_exp_11_data_out),
        .ready(ws_exp_11_ready),
        .valid_n(ws_exp_11_valid_n)
    );

    wire ws_exp_12_valid, ws_exp_12_ready_n, ws_exp_12_ready, ws_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_12_data_in, ws_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_12_valid),
        .ready_n(ws_exp_12_ready_n),
        .data_in(ws_exp_12_data_in),
        .data_out(ws_exp_12_data_out),
        .ready(ws_exp_12_ready),
        .valid_n(ws_exp_12_valid_n)
    );

    wire ws_exp_13_valid, ws_exp_13_ready_n, ws_exp_13_ready, ws_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_13_data_in, ws_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_13_valid),
        .ready_n(ws_exp_13_ready_n),
        .data_in(ws_exp_13_data_in),
        .data_out(ws_exp_13_data_out),
        .ready(ws_exp_13_ready),
        .valid_n(ws_exp_13_valid_n)
    );

    wire ws_exp_14_valid, ws_exp_14_ready_n, ws_exp_14_ready, ws_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_14_data_in, ws_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_14_valid),
        .ready_n(ws_exp_14_ready_n),
        .data_in(ws_exp_14_data_in),
        .data_out(ws_exp_14_data_out),
        .ready(ws_exp_14_ready),
        .valid_n(ws_exp_14_valid_n)
    );

    wire ws_exp_15_valid, ws_exp_15_ready_n, ws_exp_15_ready, ws_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_15_data_in, ws_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_15_valid),
        .ready_n(ws_exp_15_ready_n),
        .data_in(ws_exp_15_data_in),
        .data_out(ws_exp_15_data_out),
        .ready(ws_exp_15_ready),
        .valid_n(ws_exp_15_valid_n)
    );

    wire ws_exp_16_valid, ws_exp_16_ready_n, ws_exp_16_ready, ws_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_16_data_in, ws_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_16_valid),
        .ready_n(ws_exp_16_ready_n),
        .data_in(ws_exp_16_data_in),
        .data_out(ws_exp_16_data_out),
        .ready(ws_exp_16_ready),
        .valid_n(ws_exp_16_valid_n)
    );

    wire ws_exp_17_valid, ws_exp_17_ready_n, ws_exp_17_ready, ws_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_17_data_in, ws_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_17_valid),
        .ready_n(ws_exp_17_ready_n),
        .data_in(ws_exp_17_data_in),
        .data_out(ws_exp_17_data_out),
        .ready(ws_exp_17_ready),
        .valid_n(ws_exp_17_valid_n)
    );

    wire ws_exp_18_valid, ws_exp_18_ready_n, ws_exp_18_ready, ws_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_18_data_in, ws_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_18_valid),
        .ready_n(ws_exp_18_ready_n),
        .data_in(ws_exp_18_data_in),
        .data_out(ws_exp_18_data_out),
        .ready(ws_exp_18_ready),
        .valid_n(ws_exp_18_valid_n)
    );

    wire ws_exp_19_valid, ws_exp_19_ready_n, ws_exp_19_ready, ws_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_19_data_in, ws_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_19_valid),
        .ready_n(ws_exp_19_ready_n),
        .data_in(ws_exp_19_data_in),
        .data_out(ws_exp_19_data_out),
        .ready(ws_exp_19_ready),
        .valid_n(ws_exp_19_valid_n)
    );

    wire ws_exp_20_valid, ws_exp_20_ready_n, ws_exp_20_ready, ws_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_20_data_in, ws_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_20_valid),
        .ready_n(ws_exp_20_ready_n),
        .data_in(ws_exp_20_data_in),
        .data_out(ws_exp_20_data_out),
        .ready(ws_exp_20_ready),
        .valid_n(ws_exp_20_valid_n)
    );

    wire ws_exp_21_valid, ws_exp_21_ready_n, ws_exp_21_ready, ws_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_21_data_in, ws_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_21_valid),
        .ready_n(ws_exp_21_ready_n),
        .data_in(ws_exp_21_data_in),
        .data_out(ws_exp_21_data_out),
        .ready(ws_exp_21_ready),
        .valid_n(ws_exp_21_valid_n)
    );

    wire ws_exp_22_valid, ws_exp_22_ready_n, ws_exp_22_ready, ws_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_22_data_in, ws_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_22_valid),
        .ready_n(ws_exp_22_ready_n),
        .data_in(ws_exp_22_data_in),
        .data_out(ws_exp_22_data_out),
        .ready(ws_exp_22_ready),
        .valid_n(ws_exp_22_valid_n)
    );

    wire ws_exp_23_valid, ws_exp_23_ready_n, ws_exp_23_ready, ws_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_23_data_in, ws_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_23_valid),
        .ready_n(ws_exp_23_ready_n),
        .data_in(ws_exp_23_data_in),
        .data_out(ws_exp_23_data_out),
        .ready(ws_exp_23_ready),
        .valid_n(ws_exp_23_valid_n)
    );

    assign ws_exp_0_data_in = ws_log_sum;
    assign ws_exp_0_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_0_ready_n = 1'b0;
    assign ws_exp_1_data_in = ws_log_sum;
    assign ws_exp_1_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_1_ready_n = 1'b0;
    assign ws_exp_2_data_in = ws_log_sum;
    assign ws_exp_2_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_2_ready_n = 1'b0;
    assign ws_exp_3_data_in = ws_log_sum;
    assign ws_exp_3_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_3_ready_n = 1'b0;
    assign ws_exp_4_data_in = ws_log_sum;
    assign ws_exp_4_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_4_ready_n = 1'b0;
    assign ws_exp_5_data_in = ws_log_sum;
    assign ws_exp_5_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_5_ready_n = 1'b0;
    assign ws_exp_6_data_in = ws_log_sum;
    assign ws_exp_6_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_6_ready_n = 1'b0;
    assign ws_exp_7_data_in = ws_log_sum;
    assign ws_exp_7_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_7_ready_n = 1'b0;
    assign ws_exp_8_data_in = ws_log_sum;
    assign ws_exp_8_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_8_ready_n = 1'b0;
    assign ws_exp_9_data_in = ws_log_sum;
    assign ws_exp_9_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_9_ready_n = 1'b0;
    assign ws_exp_10_data_in = ws_log_sum;
    assign ws_exp_10_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_10_ready_n = 1'b0;
    assign ws_exp_11_data_in = ws_log_sum;
    assign ws_exp_11_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_11_ready_n = 1'b0;
    assign ws_exp_12_data_in = ws_log_sum;
    assign ws_exp_12_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_12_ready_n = 1'b0;
    assign ws_exp_13_data_in = ws_log_sum;
    assign ws_exp_13_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_13_ready_n = 1'b0;
    assign ws_exp_14_data_in = ws_log_sum;
    assign ws_exp_14_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_14_ready_n = 1'b0;
    assign ws_exp_15_data_in = ws_log_sum;
    assign ws_exp_15_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_15_ready_n = 1'b0;
    assign ws_exp_16_data_in = ws_log_sum;
    assign ws_exp_16_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_16_ready_n = 1'b0;
    assign ws_exp_17_data_in = ws_log_sum;
    assign ws_exp_17_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_17_ready_n = 1'b0;
    assign ws_exp_18_data_in = ws_log_sum;
    assign ws_exp_18_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_18_ready_n = 1'b0;
    assign ws_exp_19_data_in = ws_log_sum;
    assign ws_exp_19_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_19_ready_n = 1'b0;
    assign ws_exp_20_data_in = ws_log_sum;
    assign ws_exp_20_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_20_ready_n = 1'b0;
    assign ws_exp_21_data_in = ws_log_sum;
    assign ws_exp_21_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_21_ready_n = 1'b0;
    assign ws_exp_22_data_in = ws_log_sum;
    assign ws_exp_22_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_22_ready_n = 1'b0;
    assign ws_exp_23_data_in = ws_log_sum;
    assign ws_exp_23_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_23_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_0_valid_n) ws_accumulator <= ws_accumulator + ws_exp_0_data_out;
    end

    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    reg [DATA_WIDTH-1:0] out_write_data;
    wire [DATA_WIDTH-1:0] out_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),
              .w_addr(out_write_addr),.sram_data_in(out_write_data),.sram_data_out(out_sram_out));

    localparam WS_IDLE = 3'd0, WS_LOAD_A = 3'd1, WS_LOAD_V = 3'd2,
               WS_LOG = 3'd3, WS_COMPUTE = 3'd4, WS_OUTPUT = 3'd5;
    reg [2:0] ws_state;
    reg [$clog2(N)-1:0] ws_j;
    reg [$clog2(d)-1:0] ws_m;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ws_state <= WS_IDLE; ws_j <= 0; ws_m <= 0;
            attn_write_addr <= 0; attn_read_addr <= 0; attn_w_en <= 0;
            v_write_addr <= 0; v_read_addr <= 0; v_w_en <= 0;
            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;
            out_write_data <= 0;
        end else begin
            attn_w_en <= 0; v_w_en <= 0; out_w_en <= 0;
            case (ws_state)
                WS_IDLE: if (valid_attn || valid_v) ws_state <= WS_LOAD_A;
                WS_LOAD_A: begin
                    if (valid_attn) begin attn_w_en <= 1; attn_write_addr <= attn_write_addr + 1; end
                    if (attn_write_addr == N-1) ws_state <= WS_LOAD_V;
                end
                WS_LOAD_V: begin
                    if (valid_v) begin v_w_en <= 1; v_write_addr <= v_write_addr + 1; end
                    if (v_write_addr == N*d-1) ws_state <= WS_LOG;
                end
                WS_LOG: begin
                    // DPE(I|log) converts attn to log domain
                    attn_read_addr <= ws_j;
                    ws_j <= ws_j + 1;
                    if (ws_j == N-1) begin ws_j <= 0; ws_state <= WS_COMPUTE; end
                end
                WS_COMPUTE: begin
                    // CLB add + DPE(I|exp) + accumulate
                    log_attn_read_addr <= ws_j;
                    v_read_addr <= (ws_j << $clog2(d)) + ws_m;
                    ws_j <= ws_j + 1;
                    if (ws_j == N-1) begin
                        out_write_data <= ws_accumulator[2*DATA_WIDTH-1:DATA_WIDTH];
                        out_w_en <= 1; out_write_addr <= ws_m;
                        ws_j <= 0; ws_m <= ws_m + 1;
                        if (ws_m == d-1) ws_state <= WS_OUTPUT;
                    end
                end
                WS_OUTPUT: if (ready_n) out_read_addr <= out_read_addr + 1;
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (ws_state == WS_OUTPUT);
    assign ready_attn = (ws_state == WS_LOAD_A || ws_state == WS_IDLE);
    assign ready_v = (ws_state == WS_LOAD_V || ws_state == WS_IDLE);
endmodule

// DIMM — Block 1, Head 0, Lane 0 (depth=393216)
module dimm_score_matrix_b1_h0 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid_q, input wire valid_k,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_q,
    input wire [DATA_WIDTH-1:0] data_in_k,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_q, output wire ready_k,
    output wire valid_n
);

    reg [ADDR_WIDTH-1:0] q_write_addr, q_read_addr;
    reg q_w_en;
    wire [DATA_WIDTH-1:0] q_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    q_sram (.clk(clk),.rst(rst),.w_en(q_w_en),.r_addr(q_read_addr),
            .w_addr(q_write_addr),.sram_data_in(data_in_q),.sram_data_out(q_sram_out));

    reg [$clog2(N*d)-1:0] k_write_addr, k_read_addr_a;
    reg k_w_en;
    wire [DATA_WIDTH-1:0] k_sram_out_a;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    k_sram_a (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_a[$clog2(N*d)-1:0]),
              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_a));
    reg [$clog2(N*d)-1:0] k_read_addr_b;
    wire [DATA_WIDTH-1:0] k_sram_out_b;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    k_sram_b (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_b[$clog2(N*d)-1:0]),
              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_b));

    // CLB adder A: log_Q + log_K[j₁] (log-domain addition)
    wire [DATA_WIDTH-1:0] log_sum_a = q_sram_out + k_sram_out_a;
    // CLB adder B: log_Q + log_K[j₂] (second vector for dual-identity)
    wire [DATA_WIDTH-1:0] log_sum_b = q_sram_out + k_sram_out_b;

    // DPE(I|exp) stage_0: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_0_valid, dimm_exp_0_ready_n, dimm_exp_0_ready, dimm_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_0_data_in, dimm_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_0_valid),
        .ready_n(dimm_exp_0_ready_n),
        .data_in(dimm_exp_0_data_in),
        .data_out(dimm_exp_0_data_out),
        .ready(dimm_exp_0_ready),
        .valid_n(dimm_exp_0_valid_n)
    );

    // DPE(I|exp) stage_1: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_1_valid, dimm_exp_1_ready_n, dimm_exp_1_ready, dimm_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_1_data_in, dimm_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_1_valid),
        .ready_n(dimm_exp_1_ready_n),
        .data_in(dimm_exp_1_data_in),
        .data_out(dimm_exp_1_data_out),
        .ready(dimm_exp_1_ready),
        .valid_n(dimm_exp_1_valid_n)
    );

    // DPE(I|exp) stage_2: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_2_valid, dimm_exp_2_ready_n, dimm_exp_2_ready, dimm_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_2_data_in, dimm_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_2_valid),
        .ready_n(dimm_exp_2_ready_n),
        .data_in(dimm_exp_2_data_in),
        .data_out(dimm_exp_2_data_out),
        .ready(dimm_exp_2_ready),
        .valid_n(dimm_exp_2_valid_n)
    );

    // DPE(I|exp) stage_3: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_3_valid, dimm_exp_3_ready_n, dimm_exp_3_ready, dimm_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_3_data_in, dimm_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_3_valid),
        .ready_n(dimm_exp_3_ready_n),
        .data_in(dimm_exp_3_data_in),
        .data_out(dimm_exp_3_data_out),
        .ready(dimm_exp_3_ready),
        .valid_n(dimm_exp_3_valid_n)
    );

    // DPE(I|exp) stage_4: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_4_valid, dimm_exp_4_ready_n, dimm_exp_4_ready, dimm_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_4_data_in, dimm_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_4_valid),
        .ready_n(dimm_exp_4_ready_n),
        .data_in(dimm_exp_4_data_in),
        .data_out(dimm_exp_4_data_out),
        .ready(dimm_exp_4_ready),
        .valid_n(dimm_exp_4_valid_n)
    );

    // DPE(I|exp) stage_5: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_5_valid, dimm_exp_5_ready_n, dimm_exp_5_ready, dimm_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_5_data_in, dimm_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_5_valid),
        .ready_n(dimm_exp_5_ready_n),
        .data_in(dimm_exp_5_data_in),
        .data_out(dimm_exp_5_data_out),
        .ready(dimm_exp_5_ready),
        .valid_n(dimm_exp_5_valid_n)
    );

    // DPE(I|exp) stage_6: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_6_valid, dimm_exp_6_ready_n, dimm_exp_6_ready, dimm_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_6_data_in, dimm_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_6_valid),
        .ready_n(dimm_exp_6_ready_n),
        .data_in(dimm_exp_6_data_in),
        .data_out(dimm_exp_6_data_out),
        .ready(dimm_exp_6_ready),
        .valid_n(dimm_exp_6_valid_n)
    );

    // DPE(I|exp) stage_7: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_7_valid, dimm_exp_7_ready_n, dimm_exp_7_ready, dimm_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_7_data_in, dimm_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_7_valid),
        .ready_n(dimm_exp_7_ready_n),
        .data_in(dimm_exp_7_data_in),
        .data_out(dimm_exp_7_data_out),
        .ready(dimm_exp_7_ready),
        .valid_n(dimm_exp_7_valid_n)
    );

    // DPE(I|exp) stage_8: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_8_valid, dimm_exp_8_ready_n, dimm_exp_8_ready, dimm_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_8_data_in, dimm_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_8_valid),
        .ready_n(dimm_exp_8_ready_n),
        .data_in(dimm_exp_8_data_in),
        .data_out(dimm_exp_8_data_out),
        .ready(dimm_exp_8_ready),
        .valid_n(dimm_exp_8_valid_n)
    );

    // DPE(I|exp) stage_9: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_9_valid, dimm_exp_9_ready_n, dimm_exp_9_ready, dimm_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_9_data_in, dimm_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_9_valid),
        .ready_n(dimm_exp_9_ready_n),
        .data_in(dimm_exp_9_data_in),
        .data_out(dimm_exp_9_data_out),
        .ready(dimm_exp_9_ready),
        .valid_n(dimm_exp_9_valid_n)
    );

    // DPE(I|exp) stage_10: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_10_valid, dimm_exp_10_ready_n, dimm_exp_10_ready, dimm_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_10_data_in, dimm_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_10_valid),
        .ready_n(dimm_exp_10_ready_n),
        .data_in(dimm_exp_10_data_in),
        .data_out(dimm_exp_10_data_out),
        .ready(dimm_exp_10_ready),
        .valid_n(dimm_exp_10_valid_n)
    );

    // DPE(I|exp) stage_11: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_11_valid, dimm_exp_11_ready_n, dimm_exp_11_ready, dimm_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_11_data_in, dimm_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_11_valid),
        .ready_n(dimm_exp_11_ready_n),
        .data_in(dimm_exp_11_data_in),
        .data_out(dimm_exp_11_data_out),
        .ready(dimm_exp_11_ready),
        .valid_n(dimm_exp_11_valid_n)
    );

    // DPE(I|exp) stage_12: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_12_valid, dimm_exp_12_ready_n, dimm_exp_12_ready, dimm_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_12_data_in, dimm_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_12_valid),
        .ready_n(dimm_exp_12_ready_n),
        .data_in(dimm_exp_12_data_in),
        .data_out(dimm_exp_12_data_out),
        .ready(dimm_exp_12_ready),
        .valid_n(dimm_exp_12_valid_n)
    );

    // DPE(I|exp) stage_13: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_13_valid, dimm_exp_13_ready_n, dimm_exp_13_ready, dimm_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_13_data_in, dimm_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_13_valid),
        .ready_n(dimm_exp_13_ready_n),
        .data_in(dimm_exp_13_data_in),
        .data_out(dimm_exp_13_data_out),
        .ready(dimm_exp_13_ready),
        .valid_n(dimm_exp_13_valid_n)
    );

    // DPE(I|exp) stage_14: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_14_valid, dimm_exp_14_ready_n, dimm_exp_14_ready, dimm_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_14_data_in, dimm_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_14_valid),
        .ready_n(dimm_exp_14_ready_n),
        .data_in(dimm_exp_14_data_in),
        .data_out(dimm_exp_14_data_out),
        .ready(dimm_exp_14_ready),
        .valid_n(dimm_exp_14_valid_n)
    );

    // DPE(I|exp) stage_15: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_15_valid, dimm_exp_15_ready_n, dimm_exp_15_ready, dimm_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_15_data_in, dimm_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_15_valid),
        .ready_n(dimm_exp_15_ready_n),
        .data_in(dimm_exp_15_data_in),
        .data_out(dimm_exp_15_data_out),
        .ready(dimm_exp_15_ready),
        .valid_n(dimm_exp_15_valid_n)
    );

    // DPE(I|exp) stage_16: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_16_valid, dimm_exp_16_ready_n, dimm_exp_16_ready, dimm_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_16_data_in, dimm_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_16_valid),
        .ready_n(dimm_exp_16_ready_n),
        .data_in(dimm_exp_16_data_in),
        .data_out(dimm_exp_16_data_out),
        .ready(dimm_exp_16_ready),
        .valid_n(dimm_exp_16_valid_n)
    );

    // DPE(I|exp) stage_17: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_17_valid, dimm_exp_17_ready_n, dimm_exp_17_ready, dimm_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_17_data_in, dimm_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_17_valid),
        .ready_n(dimm_exp_17_ready_n),
        .data_in(dimm_exp_17_data_in),
        .data_out(dimm_exp_17_data_out),
        .ready(dimm_exp_17_ready),
        .valid_n(dimm_exp_17_valid_n)
    );

    // DPE(I|exp) stage_18: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_18_valid, dimm_exp_18_ready_n, dimm_exp_18_ready, dimm_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_18_data_in, dimm_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_18_valid),
        .ready_n(dimm_exp_18_ready_n),
        .data_in(dimm_exp_18_data_in),
        .data_out(dimm_exp_18_data_out),
        .ready(dimm_exp_18_ready),
        .valid_n(dimm_exp_18_valid_n)
    );

    // DPE(I|exp) stage_19: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_19_valid, dimm_exp_19_ready_n, dimm_exp_19_ready, dimm_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_19_data_in, dimm_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_19_valid),
        .ready_n(dimm_exp_19_ready_n),
        .data_in(dimm_exp_19_data_in),
        .data_out(dimm_exp_19_data_out),
        .ready(dimm_exp_19_ready),
        .valid_n(dimm_exp_19_valid_n)
    );

    // DPE(I|exp) stage_20: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_20_valid, dimm_exp_20_ready_n, dimm_exp_20_ready, dimm_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_20_data_in, dimm_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_20_valid),
        .ready_n(dimm_exp_20_ready_n),
        .data_in(dimm_exp_20_data_in),
        .data_out(dimm_exp_20_data_out),
        .ready(dimm_exp_20_ready),
        .valid_n(dimm_exp_20_valid_n)
    );

    // DPE(I|exp) stage_21: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_21_valid, dimm_exp_21_ready_n, dimm_exp_21_ready, dimm_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_21_data_in, dimm_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_21_valid),
        .ready_n(dimm_exp_21_ready_n),
        .data_in(dimm_exp_21_data_in),
        .data_out(dimm_exp_21_data_out),
        .ready(dimm_exp_21_ready),
        .valid_n(dimm_exp_21_valid_n)
    );

    // DPE(I|exp) stage_22: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_22_valid, dimm_exp_22_ready_n, dimm_exp_22_ready, dimm_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_22_data_in, dimm_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_22_valid),
        .ready_n(dimm_exp_22_ready_n),
        .data_in(dimm_exp_22_data_in),
        .data_out(dimm_exp_22_data_out),
        .ready(dimm_exp_22_ready),
        .valid_n(dimm_exp_22_valid_n)
    );

    // DPE(I|exp) stage_23: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_23_valid, dimm_exp_23_ready_n, dimm_exp_23_ready, dimm_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_23_data_in, dimm_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_23_valid),
        .ready_n(dimm_exp_23_ready_n),
        .data_in(dimm_exp_23_data_in),
        .data_out(dimm_exp_23_data_out),
        .ready(dimm_exp_23_ready),
        .valid_n(dimm_exp_23_valid_n)
    );

    reg [4:0] dimm_sel;
    assign dimm_exp_0_data_in = log_sum_a;
    assign dimm_exp_0_valid = (state == S_COMPUTE) && (dimm_sel == 0);
    assign dimm_exp_0_ready_n = 1'b0;
    assign dimm_exp_1_data_in = log_sum_a;
    assign dimm_exp_1_valid = (state == S_COMPUTE) && (dimm_sel == 1);
    assign dimm_exp_1_ready_n = 1'b0;
    assign dimm_exp_2_data_in = log_sum_a;
    assign dimm_exp_2_valid = (state == S_COMPUTE) && (dimm_sel == 2);
    assign dimm_exp_2_ready_n = 1'b0;
    assign dimm_exp_3_data_in = log_sum_a;
    assign dimm_exp_3_valid = (state == S_COMPUTE) && (dimm_sel == 3);
    assign dimm_exp_3_ready_n = 1'b0;
    assign dimm_exp_4_data_in = log_sum_a;
    assign dimm_exp_4_valid = (state == S_COMPUTE) && (dimm_sel == 4);
    assign dimm_exp_4_ready_n = 1'b0;
    assign dimm_exp_5_data_in = log_sum_a;
    assign dimm_exp_5_valid = (state == S_COMPUTE) && (dimm_sel == 5);
    assign dimm_exp_5_ready_n = 1'b0;
    assign dimm_exp_6_data_in = log_sum_a;
    assign dimm_exp_6_valid = (state == S_COMPUTE) && (dimm_sel == 6);
    assign dimm_exp_6_ready_n = 1'b0;
    assign dimm_exp_7_data_in = log_sum_a;
    assign dimm_exp_7_valid = (state == S_COMPUTE) && (dimm_sel == 7);
    assign dimm_exp_7_ready_n = 1'b0;
    assign dimm_exp_8_data_in = log_sum_a;
    assign dimm_exp_8_valid = (state == S_COMPUTE) && (dimm_sel == 8);
    assign dimm_exp_8_ready_n = 1'b0;
    assign dimm_exp_9_data_in = log_sum_a;
    assign dimm_exp_9_valid = (state == S_COMPUTE) && (dimm_sel == 9);
    assign dimm_exp_9_ready_n = 1'b0;
    assign dimm_exp_10_data_in = log_sum_a;
    assign dimm_exp_10_valid = (state == S_COMPUTE) && (dimm_sel == 10);
    assign dimm_exp_10_ready_n = 1'b0;
    assign dimm_exp_11_data_in = log_sum_a;
    assign dimm_exp_11_valid = (state == S_COMPUTE) && (dimm_sel == 11);
    assign dimm_exp_11_ready_n = 1'b0;
    assign dimm_exp_12_data_in = log_sum_a;
    assign dimm_exp_12_valid = (state == S_COMPUTE) && (dimm_sel == 12);
    assign dimm_exp_12_ready_n = 1'b0;
    assign dimm_exp_13_data_in = log_sum_a;
    assign dimm_exp_13_valid = (state == S_COMPUTE) && (dimm_sel == 13);
    assign dimm_exp_13_ready_n = 1'b0;
    assign dimm_exp_14_data_in = log_sum_a;
    assign dimm_exp_14_valid = (state == S_COMPUTE) && (dimm_sel == 14);
    assign dimm_exp_14_ready_n = 1'b0;
    assign dimm_exp_15_data_in = log_sum_a;
    assign dimm_exp_15_valid = (state == S_COMPUTE) && (dimm_sel == 15);
    assign dimm_exp_15_ready_n = 1'b0;
    assign dimm_exp_16_data_in = log_sum_a;
    assign dimm_exp_16_valid = (state == S_COMPUTE) && (dimm_sel == 16);
    assign dimm_exp_16_ready_n = 1'b0;
    assign dimm_exp_17_data_in = log_sum_a;
    assign dimm_exp_17_valid = (state == S_COMPUTE) && (dimm_sel == 17);
    assign dimm_exp_17_ready_n = 1'b0;
    assign dimm_exp_18_data_in = log_sum_a;
    assign dimm_exp_18_valid = (state == S_COMPUTE) && (dimm_sel == 18);
    assign dimm_exp_18_ready_n = 1'b0;
    assign dimm_exp_19_data_in = log_sum_a;
    assign dimm_exp_19_valid = (state == S_COMPUTE) && (dimm_sel == 19);
    assign dimm_exp_19_ready_n = 1'b0;
    assign dimm_exp_20_data_in = log_sum_a;
    assign dimm_exp_20_valid = (state == S_COMPUTE) && (dimm_sel == 20);
    assign dimm_exp_20_ready_n = 1'b0;
    assign dimm_exp_21_data_in = log_sum_a;
    assign dimm_exp_21_valid = (state == S_COMPUTE) && (dimm_sel == 21);
    assign dimm_exp_21_ready_n = 1'b0;
    assign dimm_exp_22_data_in = log_sum_a;
    assign dimm_exp_22_valid = (state == S_COMPUTE) && (dimm_sel == 22);
    assign dimm_exp_22_ready_n = 1'b0;
    assign dimm_exp_23_data_in = log_sum_a;
    assign dimm_exp_23_valid = (state == S_COMPUTE) && (dimm_sel == 23);
    assign dimm_exp_23_ready_n = 1'b0;

    // Dual accumulators: one per vector (A for j₁, B for j₂)
    reg [2*DATA_WIDTH-1:0] accumulator_a, accumulator_b;
    reg acc_phase;  // 0=accumulating A outputs, 1=accumulating B outputs
    always @(posedge clk) begin
        if (rst) begin accumulator_a <= 0; accumulator_b <= 0; end
        else if (dimm_exp_valid_n) begin
            if (!acc_phase) accumulator_a <= accumulator_a + dimm_exp_data_out;
            else accumulator_b <= accumulator_b + dimm_exp_data_out;
        end
    end

    reg [ADDR_WIDTH-1:0] score_write_addr, score_read_addr;
    reg score_w_en;
    reg [DATA_WIDTH-1:0] score_write_data;
    wire [DATA_WIDTH-1:0] score_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    score_sram (.clk(clk),.rst(rst),.w_en(score_w_en),.r_addr(score_read_addr),
                .w_addr(score_write_addr),.sram_data_in(score_write_data),.sram_data_out(score_sram_out));

    localparam S_IDLE = 3'd0, S_LOAD_Q = 3'd1, S_LOAD_K = 3'd2,
               S_COMPUTE = 3'd3, S_OUTPUT = 3'd4;
    localparam S_WRITE_B = 3'd5;  // write second score for dual-identity
    reg [2:0] state;
    reg [$clog2(d)-1:0] mac_count;
    reg [$clog2(N)-1:0] score_idx;
    reg feed_half;  // 0=feeding vector A (d elements), 1=feeding vector B

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            q_write_addr <= 0; q_read_addr <= 0; q_w_en <= 0;
            k_write_addr <= 0; k_read_addr_a <= 0; k_w_en <= 0;
            k_read_addr_b <= 0;
            feed_half <= 0; feed_phase <= 0; acc_phase <= 0;
            score_write_addr <= 0; score_read_addr <= 0; score_w_en <= 0;
            score_write_data <= 0;
            mac_count <= 0; score_idx <= 0;
            dimm_sel <= 0;
        end else begin
            q_w_en <= 0; k_w_en <= 0; score_w_en <= 0;
            case (state)
                S_IDLE: if (valid_q || valid_k) state <= S_LOAD_Q;
                S_LOAD_Q: begin
                    if (valid_q) begin q_w_en <= 1; q_write_addr <= q_write_addr + 1; end
                    if (q_write_addr == d-1) state <= S_LOAD_K;
                end
                S_LOAD_K: begin
                    if (valid_k) begin k_w_en <= 1; k_write_addr <= k_write_addr + 1; end
                    if (k_write_addr == N*d-1) state <= S_COMPUTE;
                end
                S_COMPUTE: begin
                    // Dual-identity: feed vector A then vector B per DPE pass
                    q_read_addr <= mac_count;
                    k_read_addr_a <= (score_idx << $clog2(d)) + mac_count;
                    k_read_addr_b <= ((score_idx + 1) << $clog2(d)) + mac_count;
                    if (!feed_half) begin
                        // Feeding vector A (log_sum_a)
                        feed_phase <= 0; acc_phase <= 0;
                        mac_count <= mac_count + 1;
                        if (mac_count == d-1) begin
                            mac_count <= 0;
                            feed_half <= 1;
                        end
                    end else begin
                        // Feeding vector B (log_sum_b)
                        feed_phase <= 1; acc_phase <= 1;
                        mac_count <= mac_count + 1;
                        if (mac_count == d-1) begin
                            // Both vectors done — write score A, then B
                            score_write_data <= accumulator_a[2*DATA_WIDTH-1:DATA_WIDTH];
                            score_w_en <= 1;
                            score_write_addr <= score_idx;
                            mac_count <= 0;
                            feed_half <= 0;
                            state <= S_WRITE_B;
                        end
                    end
                end
                S_WRITE_B: begin
                    // Write second score (j₂ = score_idx + 1)
                    score_write_data <= accumulator_b[2*DATA_WIDTH-1:DATA_WIDTH];
                    score_w_en <= 1;
                    score_write_addr <= score_idx + 1;
                    accumulator_a <= 0; accumulator_b <= 0;
                    if (score_idx >= N-2) state <= S_OUTPUT;
                    else begin score_idx <= score_idx + 2; state <= S_COMPUTE; end
                end
                S_OUTPUT: if (ready_n) score_read_addr <= score_read_addr + 1;
            endcase
        end
    end

    assign data_out = score_sram_out;
    assign valid_n = (state == S_OUTPUT);
    assign ready_q = (state == S_LOAD_Q || state == S_IDLE);
    assign ready_k = (state == S_LOAD_K || state == S_IDLE);
endmodule
module softmax_approx_b1_h0 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid, input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, output wire valid_n
);

    reg [ADDR_WIDTH-1:0] in_write_addr, in_read_addr;
    reg in_w_en;
    wire [DATA_WIDTH-1:0] in_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    in_sram (.clk(clk),.rst(rst),.w_en(in_w_en),.r_addr(in_read_addr),
             .w_addr(in_write_addr),.sram_data_in(data_in),.sram_data_out(in_sram_out));

    wire sm_exp_0_valid, sm_exp_0_ready_n, sm_exp_0_ready, sm_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_0_data_in, sm_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_0_valid),
        .ready_n(sm_exp_0_ready_n),
        .data_in(sm_exp_0_data_in),
        .data_out(sm_exp_0_data_out),
        .ready(sm_exp_0_ready),
        .valid_n(sm_exp_0_valid_n)
    );

    wire sm_exp_1_valid, sm_exp_1_ready_n, sm_exp_1_ready, sm_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_1_data_in, sm_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_1_valid),
        .ready_n(sm_exp_1_ready_n),
        .data_in(sm_exp_1_data_in),
        .data_out(sm_exp_1_data_out),
        .ready(sm_exp_1_ready),
        .valid_n(sm_exp_1_valid_n)
    );

    wire sm_exp_2_valid, sm_exp_2_ready_n, sm_exp_2_ready, sm_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_2_data_in, sm_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_2_valid),
        .ready_n(sm_exp_2_ready_n),
        .data_in(sm_exp_2_data_in),
        .data_out(sm_exp_2_data_out),
        .ready(sm_exp_2_ready),
        .valid_n(sm_exp_2_valid_n)
    );

    wire sm_exp_3_valid, sm_exp_3_ready_n, sm_exp_3_ready, sm_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_3_data_in, sm_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_3_valid),
        .ready_n(sm_exp_3_ready_n),
        .data_in(sm_exp_3_data_in),
        .data_out(sm_exp_3_data_out),
        .ready(sm_exp_3_ready),
        .valid_n(sm_exp_3_valid_n)
    );

    wire sm_exp_4_valid, sm_exp_4_ready_n, sm_exp_4_ready, sm_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_4_data_in, sm_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_4_valid),
        .ready_n(sm_exp_4_ready_n),
        .data_in(sm_exp_4_data_in),
        .data_out(sm_exp_4_data_out),
        .ready(sm_exp_4_ready),
        .valid_n(sm_exp_4_valid_n)
    );

    wire sm_exp_5_valid, sm_exp_5_ready_n, sm_exp_5_ready, sm_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_5_data_in, sm_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_5_valid),
        .ready_n(sm_exp_5_ready_n),
        .data_in(sm_exp_5_data_in),
        .data_out(sm_exp_5_data_out),
        .ready(sm_exp_5_ready),
        .valid_n(sm_exp_5_valid_n)
    );

    wire sm_exp_6_valid, sm_exp_6_ready_n, sm_exp_6_ready, sm_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_6_data_in, sm_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_6_valid),
        .ready_n(sm_exp_6_ready_n),
        .data_in(sm_exp_6_data_in),
        .data_out(sm_exp_6_data_out),
        .ready(sm_exp_6_ready),
        .valid_n(sm_exp_6_valid_n)
    );

    wire sm_exp_7_valid, sm_exp_7_ready_n, sm_exp_7_ready, sm_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_7_data_in, sm_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_7_valid),
        .ready_n(sm_exp_7_ready_n),
        .data_in(sm_exp_7_data_in),
        .data_out(sm_exp_7_data_out),
        .ready(sm_exp_7_ready),
        .valid_n(sm_exp_7_valid_n)
    );

    wire sm_exp_8_valid, sm_exp_8_ready_n, sm_exp_8_ready, sm_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_8_data_in, sm_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_8_valid),
        .ready_n(sm_exp_8_ready_n),
        .data_in(sm_exp_8_data_in),
        .data_out(sm_exp_8_data_out),
        .ready(sm_exp_8_ready),
        .valid_n(sm_exp_8_valid_n)
    );

    wire sm_exp_9_valid, sm_exp_9_ready_n, sm_exp_9_ready, sm_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_9_data_in, sm_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_9_valid),
        .ready_n(sm_exp_9_ready_n),
        .data_in(sm_exp_9_data_in),
        .data_out(sm_exp_9_data_out),
        .ready(sm_exp_9_ready),
        .valid_n(sm_exp_9_valid_n)
    );

    wire sm_exp_10_valid, sm_exp_10_ready_n, sm_exp_10_ready, sm_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_10_data_in, sm_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_10_valid),
        .ready_n(sm_exp_10_ready_n),
        .data_in(sm_exp_10_data_in),
        .data_out(sm_exp_10_data_out),
        .ready(sm_exp_10_ready),
        .valid_n(sm_exp_10_valid_n)
    );

    wire sm_exp_11_valid, sm_exp_11_ready_n, sm_exp_11_ready, sm_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_11_data_in, sm_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_11_valid),
        .ready_n(sm_exp_11_ready_n),
        .data_in(sm_exp_11_data_in),
        .data_out(sm_exp_11_data_out),
        .ready(sm_exp_11_ready),
        .valid_n(sm_exp_11_valid_n)
    );

    wire sm_exp_12_valid, sm_exp_12_ready_n, sm_exp_12_ready, sm_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_12_data_in, sm_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_12_valid),
        .ready_n(sm_exp_12_ready_n),
        .data_in(sm_exp_12_data_in),
        .data_out(sm_exp_12_data_out),
        .ready(sm_exp_12_ready),
        .valid_n(sm_exp_12_valid_n)
    );

    wire sm_exp_13_valid, sm_exp_13_ready_n, sm_exp_13_ready, sm_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_13_data_in, sm_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_13_valid),
        .ready_n(sm_exp_13_ready_n),
        .data_in(sm_exp_13_data_in),
        .data_out(sm_exp_13_data_out),
        .ready(sm_exp_13_ready),
        .valid_n(sm_exp_13_valid_n)
    );

    wire sm_exp_14_valid, sm_exp_14_ready_n, sm_exp_14_ready, sm_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_14_data_in, sm_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_14_valid),
        .ready_n(sm_exp_14_ready_n),
        .data_in(sm_exp_14_data_in),
        .data_out(sm_exp_14_data_out),
        .ready(sm_exp_14_ready),
        .valid_n(sm_exp_14_valid_n)
    );

    wire sm_exp_15_valid, sm_exp_15_ready_n, sm_exp_15_ready, sm_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_15_data_in, sm_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_15_valid),
        .ready_n(sm_exp_15_ready_n),
        .data_in(sm_exp_15_data_in),
        .data_out(sm_exp_15_data_out),
        .ready(sm_exp_15_ready),
        .valid_n(sm_exp_15_valid_n)
    );

    wire sm_exp_16_valid, sm_exp_16_ready_n, sm_exp_16_ready, sm_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_16_data_in, sm_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_16_valid),
        .ready_n(sm_exp_16_ready_n),
        .data_in(sm_exp_16_data_in),
        .data_out(sm_exp_16_data_out),
        .ready(sm_exp_16_ready),
        .valid_n(sm_exp_16_valid_n)
    );

    wire sm_exp_17_valid, sm_exp_17_ready_n, sm_exp_17_ready, sm_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_17_data_in, sm_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_17_valid),
        .ready_n(sm_exp_17_ready_n),
        .data_in(sm_exp_17_data_in),
        .data_out(sm_exp_17_data_out),
        .ready(sm_exp_17_ready),
        .valid_n(sm_exp_17_valid_n)
    );

    wire sm_exp_18_valid, sm_exp_18_ready_n, sm_exp_18_ready, sm_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_18_data_in, sm_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_18_valid),
        .ready_n(sm_exp_18_ready_n),
        .data_in(sm_exp_18_data_in),
        .data_out(sm_exp_18_data_out),
        .ready(sm_exp_18_ready),
        .valid_n(sm_exp_18_valid_n)
    );

    wire sm_exp_19_valid, sm_exp_19_ready_n, sm_exp_19_ready, sm_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_19_data_in, sm_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_19_valid),
        .ready_n(sm_exp_19_ready_n),
        .data_in(sm_exp_19_data_in),
        .data_out(sm_exp_19_data_out),
        .ready(sm_exp_19_ready),
        .valid_n(sm_exp_19_valid_n)
    );

    wire sm_exp_20_valid, sm_exp_20_ready_n, sm_exp_20_ready, sm_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_20_data_in, sm_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_20_valid),
        .ready_n(sm_exp_20_ready_n),
        .data_in(sm_exp_20_data_in),
        .data_out(sm_exp_20_data_out),
        .ready(sm_exp_20_ready),
        .valid_n(sm_exp_20_valid_n)
    );

    wire sm_exp_21_valid, sm_exp_21_ready_n, sm_exp_21_ready, sm_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_21_data_in, sm_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_21_valid),
        .ready_n(sm_exp_21_ready_n),
        .data_in(sm_exp_21_data_in),
        .data_out(sm_exp_21_data_out),
        .ready(sm_exp_21_ready),
        .valid_n(sm_exp_21_valid_n)
    );

    wire sm_exp_22_valid, sm_exp_22_ready_n, sm_exp_22_ready, sm_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_22_data_in, sm_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_22_valid),
        .ready_n(sm_exp_22_ready_n),
        .data_in(sm_exp_22_data_in),
        .data_out(sm_exp_22_data_out),
        .ready(sm_exp_22_ready),
        .valid_n(sm_exp_22_valid_n)
    );

    wire sm_exp_23_valid, sm_exp_23_ready_n, sm_exp_23_ready, sm_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_23_data_in, sm_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_23_valid),
        .ready_n(sm_exp_23_ready_n),
        .data_in(sm_exp_23_data_in),
        .data_out(sm_exp_23_data_out),
        .ready(sm_exp_23_ready),
        .valid_n(sm_exp_23_valid_n)
    );

    assign sm_exp_0_data_in = in_sram_out;
    assign sm_exp_0_valid = (sm_state == SM_EXP);
    assign sm_exp_0_ready_n = 1'b0;
    assign sm_exp_1_data_in = in_sram_out;
    assign sm_exp_1_valid = (sm_state == SM_EXP);
    assign sm_exp_1_ready_n = 1'b0;
    assign sm_exp_2_data_in = in_sram_out;
    assign sm_exp_2_valid = (sm_state == SM_EXP);
    assign sm_exp_2_ready_n = 1'b0;
    assign sm_exp_3_data_in = in_sram_out;
    assign sm_exp_3_valid = (sm_state == SM_EXP);
    assign sm_exp_3_ready_n = 1'b0;
    assign sm_exp_4_data_in = in_sram_out;
    assign sm_exp_4_valid = (sm_state == SM_EXP);
    assign sm_exp_4_ready_n = 1'b0;
    assign sm_exp_5_data_in = in_sram_out;
    assign sm_exp_5_valid = (sm_state == SM_EXP);
    assign sm_exp_5_ready_n = 1'b0;
    assign sm_exp_6_data_in = in_sram_out;
    assign sm_exp_6_valid = (sm_state == SM_EXP);
    assign sm_exp_6_ready_n = 1'b0;
    assign sm_exp_7_data_in = in_sram_out;
    assign sm_exp_7_valid = (sm_state == SM_EXP);
    assign sm_exp_7_ready_n = 1'b0;
    assign sm_exp_8_data_in = in_sram_out;
    assign sm_exp_8_valid = (sm_state == SM_EXP);
    assign sm_exp_8_ready_n = 1'b0;
    assign sm_exp_9_data_in = in_sram_out;
    assign sm_exp_9_valid = (sm_state == SM_EXP);
    assign sm_exp_9_ready_n = 1'b0;
    assign sm_exp_10_data_in = in_sram_out;
    assign sm_exp_10_valid = (sm_state == SM_EXP);
    assign sm_exp_10_ready_n = 1'b0;
    assign sm_exp_11_data_in = in_sram_out;
    assign sm_exp_11_valid = (sm_state == SM_EXP);
    assign sm_exp_11_ready_n = 1'b0;
    assign sm_exp_12_data_in = in_sram_out;
    assign sm_exp_12_valid = (sm_state == SM_EXP);
    assign sm_exp_12_ready_n = 1'b0;
    assign sm_exp_13_data_in = in_sram_out;
    assign sm_exp_13_valid = (sm_state == SM_EXP);
    assign sm_exp_13_ready_n = 1'b0;
    assign sm_exp_14_data_in = in_sram_out;
    assign sm_exp_14_valid = (sm_state == SM_EXP);
    assign sm_exp_14_ready_n = 1'b0;
    assign sm_exp_15_data_in = in_sram_out;
    assign sm_exp_15_valid = (sm_state == SM_EXP);
    assign sm_exp_15_ready_n = 1'b0;
    assign sm_exp_16_data_in = in_sram_out;
    assign sm_exp_16_valid = (sm_state == SM_EXP);
    assign sm_exp_16_ready_n = 1'b0;
    assign sm_exp_17_data_in = in_sram_out;
    assign sm_exp_17_valid = (sm_state == SM_EXP);
    assign sm_exp_17_ready_n = 1'b0;
    assign sm_exp_18_data_in = in_sram_out;
    assign sm_exp_18_valid = (sm_state == SM_EXP);
    assign sm_exp_18_ready_n = 1'b0;
    assign sm_exp_19_data_in = in_sram_out;
    assign sm_exp_19_valid = (sm_state == SM_EXP);
    assign sm_exp_19_ready_n = 1'b0;
    assign sm_exp_20_data_in = in_sram_out;
    assign sm_exp_20_valid = (sm_state == SM_EXP);
    assign sm_exp_20_ready_n = 1'b0;
    assign sm_exp_21_data_in = in_sram_out;
    assign sm_exp_21_valid = (sm_state == SM_EXP);
    assign sm_exp_21_ready_n = 1'b0;
    assign sm_exp_22_data_in = in_sram_out;
    assign sm_exp_22_valid = (sm_state == SM_EXP);
    assign sm_exp_22_ready_n = 1'b0;
    assign sm_exp_23_data_in = in_sram_out;
    assign sm_exp_23_valid = (sm_state == SM_EXP);
    assign sm_exp_23_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] exp_write_addr, exp_read_addr;
    reg exp_w_en;
    reg [DATA_WIDTH-1:0] exp_write_data;
    wire [DATA_WIDTH-1:0] exp_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    exp_sram (.clk(clk),.rst(rst),.w_en(exp_w_en),.r_addr(exp_read_addr),
              .w_addr(exp_write_addr),.sram_data_in(exp_write_data),.sram_data_out(exp_sram_out));
    reg [2*DATA_WIDTH-1:0] exp_sum;

    always @(posedge clk) begin
        if (rst) begin exp_w_en <= 0; exp_write_addr <= 0; exp_sum <= 0; end
        else if (sm_exp_0_valid_n) begin
            exp_write_data <= sm_exp_0_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_0_data_out;
        end else exp_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] exp_sum_upper = exp_sum[2*DATA_WIDTH-1:DATA_WIDTH];
    reg [DATA_WIDTH-1:0] recip_val;
    always @(*) begin
        casez (exp_sum_upper[15:0])
            16'b1???????????????: recip_val = 40'd2;
            16'b01??????????????: recip_val = 40'd32768;
            16'b001?????????????: recip_val = 40'd16384;
            16'b0001????????????: recip_val = 40'd8192;
            16'b00001???????????: recip_val = 40'd4096;
            16'b000001??????????: recip_val = 40'd2048;
            16'b0000001?????????: recip_val = 40'd1024;
            16'b00000001????????: recip_val = 40'd512;
            16'b000000001???????: recip_val = 40'd256;
            16'b0000000001??????: recip_val = 40'd128;
            16'b00000000001?????: recip_val = 40'd64;
            16'b000000000001????: recip_val = 40'd32;
            16'b0000000000001???: recip_val = 40'd16;
            16'b00000000000001??: recip_val = 40'd8;
            16'b000000000000001?: recip_val = 40'd4;
            16'b0000000000000001: recip_val = 40'd2;
            default: recip_val = 40'hFFFF;
        endcase
    end

    reg [DATA_WIDTH-1:0] inv_sum;
    reg [2*DATA_WIDTH-1:0] norm_product;
    reg [DATA_WIDTH-1:0] norm_val;

    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    wire [DATA_WIDTH-1:0] out_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),
              .w_addr(out_write_addr),.sram_data_in(norm_val),.sram_data_out(out_sram_out));

    localparam SM_IDLE = 3'd0, SM_LOAD = 3'd1, SM_EXP = 3'd2,
               SM_NORMALIZE = 3'd3, SM_OUTPUT = 3'd4;
    reg [2:0] sm_state;
    reg [$clog2(N)-1:0] sm_count;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sm_state <= SM_IDLE; sm_count <= 0;
            in_write_addr <= 0; in_read_addr <= 0; in_w_en <= 0;
            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;
            inv_sum <= 0; norm_product <= 0; norm_val <= 0;
        end else begin
            in_w_en <= 0; out_w_en <= 0;
            case (sm_state)
                SM_IDLE: if (valid) sm_state <= SM_LOAD;
                SM_LOAD: begin
                    if (valid) begin in_w_en <= 1; in_write_addr <= in_write_addr + 1; end
                    if (in_write_addr == N-1) sm_state <= SM_EXP;
                end
                SM_EXP: begin
                    // DPE(I|exp) processes scores — output goes to exp_sram + sum
                    in_read_addr <= sm_count; sm_count <= sm_count + 1;
                    if (sm_count == N-1) begin
                        inv_sum <= recip_val; sm_count <= 0;
                        sm_state <= SM_NORMALIZE;
                    end
                end
                SM_NORMALIZE: begin
                    // CLB multiply: exp_val × inv_sum
                    exp_read_addr <= sm_count;
                    norm_product <= exp_sram_out * inv_sum;
                    norm_val <= norm_product[2*DATA_WIDTH-1:DATA_WIDTH];
                    out_w_en <= (sm_count > 1);
                    out_write_addr <= sm_count - 2;
                    sm_count <= sm_count + 1;
                    if (sm_count == N+1) sm_state <= SM_OUTPUT;
                end
                SM_OUTPUT: if (!ready_n) out_read_addr <= out_read_addr + 1;
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (sm_state == SM_OUTPUT);
    assign ready = (sm_state == SM_LOAD || sm_state == SM_IDLE);
endmodule
module dimm_weighted_sum_b1_h0 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid_attn, input wire valid_v,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_attn,
    input wire [DATA_WIDTH-1:0] data_in_v,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_attn, output wire ready_v,
    output wire valid_n
);

    reg [ADDR_WIDTH-1:0] attn_write_addr, attn_read_addr;
    reg attn_w_en;
    wire [DATA_WIDTH-1:0] attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    attn_sram (.clk(clk),.rst(rst),.w_en(attn_w_en),.r_addr(attn_read_addr),
               .w_addr(attn_write_addr),.sram_data_in(data_in_attn),.sram_data_out(attn_sram_out));

    reg [$clog2(N*d)-1:0] v_write_addr, v_read_addr;
    reg v_w_en;
    wire [DATA_WIDTH-1:0] v_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    v_sram (.clk(clk),.rst(rst),.w_en(v_w_en),.r_addr(v_read_addr[$clog2(N*d)-1:0]),
            .w_addr(v_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_v),.sram_data_out(v_sram_out));

    wire ws_log_0_valid, ws_log_0_ready_n, ws_log_0_ready, ws_log_0_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_0_data_in, ws_log_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_0 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_0_valid),
        .ready_n(ws_log_0_ready_n),
        .data_in(ws_log_0_data_in),
        .data_out(ws_log_0_data_out),
        .ready(ws_log_0_ready),
        .valid_n(ws_log_0_valid_n)
    );

    wire ws_log_1_valid, ws_log_1_ready_n, ws_log_1_ready, ws_log_1_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_1_data_in, ws_log_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_1 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_1_valid),
        .ready_n(ws_log_1_ready_n),
        .data_in(ws_log_1_data_in),
        .data_out(ws_log_1_data_out),
        .ready(ws_log_1_ready),
        .valid_n(ws_log_1_valid_n)
    );

    wire ws_log_2_valid, ws_log_2_ready_n, ws_log_2_ready, ws_log_2_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_2_data_in, ws_log_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_2 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_2_valid),
        .ready_n(ws_log_2_ready_n),
        .data_in(ws_log_2_data_in),
        .data_out(ws_log_2_data_out),
        .ready(ws_log_2_ready),
        .valid_n(ws_log_2_valid_n)
    );

    wire ws_log_3_valid, ws_log_3_ready_n, ws_log_3_ready, ws_log_3_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_3_data_in, ws_log_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_3 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_3_valid),
        .ready_n(ws_log_3_ready_n),
        .data_in(ws_log_3_data_in),
        .data_out(ws_log_3_data_out),
        .ready(ws_log_3_ready),
        .valid_n(ws_log_3_valid_n)
    );

    wire ws_log_4_valid, ws_log_4_ready_n, ws_log_4_ready, ws_log_4_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_4_data_in, ws_log_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_4 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_4_valid),
        .ready_n(ws_log_4_ready_n),
        .data_in(ws_log_4_data_in),
        .data_out(ws_log_4_data_out),
        .ready(ws_log_4_ready),
        .valid_n(ws_log_4_valid_n)
    );

    wire ws_log_5_valid, ws_log_5_ready_n, ws_log_5_ready, ws_log_5_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_5_data_in, ws_log_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_5 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_5_valid),
        .ready_n(ws_log_5_ready_n),
        .data_in(ws_log_5_data_in),
        .data_out(ws_log_5_data_out),
        .ready(ws_log_5_ready),
        .valid_n(ws_log_5_valid_n)
    );

    wire ws_log_6_valid, ws_log_6_ready_n, ws_log_6_ready, ws_log_6_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_6_data_in, ws_log_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_6 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_6_valid),
        .ready_n(ws_log_6_ready_n),
        .data_in(ws_log_6_data_in),
        .data_out(ws_log_6_data_out),
        .ready(ws_log_6_ready),
        .valid_n(ws_log_6_valid_n)
    );

    wire ws_log_7_valid, ws_log_7_ready_n, ws_log_7_ready, ws_log_7_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_7_data_in, ws_log_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_7 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_7_valid),
        .ready_n(ws_log_7_ready_n),
        .data_in(ws_log_7_data_in),
        .data_out(ws_log_7_data_out),
        .ready(ws_log_7_ready),
        .valid_n(ws_log_7_valid_n)
    );

    wire ws_log_8_valid, ws_log_8_ready_n, ws_log_8_ready, ws_log_8_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_8_data_in, ws_log_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_8 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_8_valid),
        .ready_n(ws_log_8_ready_n),
        .data_in(ws_log_8_data_in),
        .data_out(ws_log_8_data_out),
        .ready(ws_log_8_ready),
        .valid_n(ws_log_8_valid_n)
    );

    wire ws_log_9_valid, ws_log_9_ready_n, ws_log_9_ready, ws_log_9_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_9_data_in, ws_log_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_9 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_9_valid),
        .ready_n(ws_log_9_ready_n),
        .data_in(ws_log_9_data_in),
        .data_out(ws_log_9_data_out),
        .ready(ws_log_9_ready),
        .valid_n(ws_log_9_valid_n)
    );

    wire ws_log_10_valid, ws_log_10_ready_n, ws_log_10_ready, ws_log_10_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_10_data_in, ws_log_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_10 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_10_valid),
        .ready_n(ws_log_10_ready_n),
        .data_in(ws_log_10_data_in),
        .data_out(ws_log_10_data_out),
        .ready(ws_log_10_ready),
        .valid_n(ws_log_10_valid_n)
    );

    wire ws_log_11_valid, ws_log_11_ready_n, ws_log_11_ready, ws_log_11_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_11_data_in, ws_log_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_11 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_11_valid),
        .ready_n(ws_log_11_ready_n),
        .data_in(ws_log_11_data_in),
        .data_out(ws_log_11_data_out),
        .ready(ws_log_11_ready),
        .valid_n(ws_log_11_valid_n)
    );

    wire ws_log_12_valid, ws_log_12_ready_n, ws_log_12_ready, ws_log_12_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_12_data_in, ws_log_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_12 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_12_valid),
        .ready_n(ws_log_12_ready_n),
        .data_in(ws_log_12_data_in),
        .data_out(ws_log_12_data_out),
        .ready(ws_log_12_ready),
        .valid_n(ws_log_12_valid_n)
    );

    wire ws_log_13_valid, ws_log_13_ready_n, ws_log_13_ready, ws_log_13_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_13_data_in, ws_log_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_13 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_13_valid),
        .ready_n(ws_log_13_ready_n),
        .data_in(ws_log_13_data_in),
        .data_out(ws_log_13_data_out),
        .ready(ws_log_13_ready),
        .valid_n(ws_log_13_valid_n)
    );

    wire ws_log_14_valid, ws_log_14_ready_n, ws_log_14_ready, ws_log_14_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_14_data_in, ws_log_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_14 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_14_valid),
        .ready_n(ws_log_14_ready_n),
        .data_in(ws_log_14_data_in),
        .data_out(ws_log_14_data_out),
        .ready(ws_log_14_ready),
        .valid_n(ws_log_14_valid_n)
    );

    wire ws_log_15_valid, ws_log_15_ready_n, ws_log_15_ready, ws_log_15_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_15_data_in, ws_log_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_15 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_15_valid),
        .ready_n(ws_log_15_ready_n),
        .data_in(ws_log_15_data_in),
        .data_out(ws_log_15_data_out),
        .ready(ws_log_15_ready),
        .valid_n(ws_log_15_valid_n)
    );

    wire ws_log_16_valid, ws_log_16_ready_n, ws_log_16_ready, ws_log_16_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_16_data_in, ws_log_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_16 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_16_valid),
        .ready_n(ws_log_16_ready_n),
        .data_in(ws_log_16_data_in),
        .data_out(ws_log_16_data_out),
        .ready(ws_log_16_ready),
        .valid_n(ws_log_16_valid_n)
    );

    wire ws_log_17_valid, ws_log_17_ready_n, ws_log_17_ready, ws_log_17_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_17_data_in, ws_log_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_17 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_17_valid),
        .ready_n(ws_log_17_ready_n),
        .data_in(ws_log_17_data_in),
        .data_out(ws_log_17_data_out),
        .ready(ws_log_17_ready),
        .valid_n(ws_log_17_valid_n)
    );

    wire ws_log_18_valid, ws_log_18_ready_n, ws_log_18_ready, ws_log_18_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_18_data_in, ws_log_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_18 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_18_valid),
        .ready_n(ws_log_18_ready_n),
        .data_in(ws_log_18_data_in),
        .data_out(ws_log_18_data_out),
        .ready(ws_log_18_ready),
        .valid_n(ws_log_18_valid_n)
    );

    wire ws_log_19_valid, ws_log_19_ready_n, ws_log_19_ready, ws_log_19_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_19_data_in, ws_log_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_19 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_19_valid),
        .ready_n(ws_log_19_ready_n),
        .data_in(ws_log_19_data_in),
        .data_out(ws_log_19_data_out),
        .ready(ws_log_19_ready),
        .valid_n(ws_log_19_valid_n)
    );

    wire ws_log_20_valid, ws_log_20_ready_n, ws_log_20_ready, ws_log_20_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_20_data_in, ws_log_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_20 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_20_valid),
        .ready_n(ws_log_20_ready_n),
        .data_in(ws_log_20_data_in),
        .data_out(ws_log_20_data_out),
        .ready(ws_log_20_ready),
        .valid_n(ws_log_20_valid_n)
    );

    wire ws_log_21_valid, ws_log_21_ready_n, ws_log_21_ready, ws_log_21_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_21_data_in, ws_log_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_21 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_21_valid),
        .ready_n(ws_log_21_ready_n),
        .data_in(ws_log_21_data_in),
        .data_out(ws_log_21_data_out),
        .ready(ws_log_21_ready),
        .valid_n(ws_log_21_valid_n)
    );

    wire ws_log_22_valid, ws_log_22_ready_n, ws_log_22_ready, ws_log_22_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_22_data_in, ws_log_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_22 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_22_valid),
        .ready_n(ws_log_22_ready_n),
        .data_in(ws_log_22_data_in),
        .data_out(ws_log_22_data_out),
        .ready(ws_log_22_ready),
        .valid_n(ws_log_22_valid_n)
    );

    wire ws_log_23_valid, ws_log_23_ready_n, ws_log_23_ready, ws_log_23_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_23_data_in, ws_log_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_23 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_23_valid),
        .ready_n(ws_log_23_ready_n),
        .data_in(ws_log_23_data_in),
        .data_out(ws_log_23_data_out),
        .ready(ws_log_23_ready),
        .valid_n(ws_log_23_valid_n)
    );

    assign ws_log_0_data_in = attn_sram_out;
    assign ws_log_0_valid = (ws_state == WS_LOG);
    assign ws_log_0_ready_n = 1'b0;
    assign ws_log_1_data_in = attn_sram_out;
    assign ws_log_1_valid = (ws_state == WS_LOG);
    assign ws_log_1_ready_n = 1'b0;
    assign ws_log_2_data_in = attn_sram_out;
    assign ws_log_2_valid = (ws_state == WS_LOG);
    assign ws_log_2_ready_n = 1'b0;
    assign ws_log_3_data_in = attn_sram_out;
    assign ws_log_3_valid = (ws_state == WS_LOG);
    assign ws_log_3_ready_n = 1'b0;
    assign ws_log_4_data_in = attn_sram_out;
    assign ws_log_4_valid = (ws_state == WS_LOG);
    assign ws_log_4_ready_n = 1'b0;
    assign ws_log_5_data_in = attn_sram_out;
    assign ws_log_5_valid = (ws_state == WS_LOG);
    assign ws_log_5_ready_n = 1'b0;
    assign ws_log_6_data_in = attn_sram_out;
    assign ws_log_6_valid = (ws_state == WS_LOG);
    assign ws_log_6_ready_n = 1'b0;
    assign ws_log_7_data_in = attn_sram_out;
    assign ws_log_7_valid = (ws_state == WS_LOG);
    assign ws_log_7_ready_n = 1'b0;
    assign ws_log_8_data_in = attn_sram_out;
    assign ws_log_8_valid = (ws_state == WS_LOG);
    assign ws_log_8_ready_n = 1'b0;
    assign ws_log_9_data_in = attn_sram_out;
    assign ws_log_9_valid = (ws_state == WS_LOG);
    assign ws_log_9_ready_n = 1'b0;
    assign ws_log_10_data_in = attn_sram_out;
    assign ws_log_10_valid = (ws_state == WS_LOG);
    assign ws_log_10_ready_n = 1'b0;
    assign ws_log_11_data_in = attn_sram_out;
    assign ws_log_11_valid = (ws_state == WS_LOG);
    assign ws_log_11_ready_n = 1'b0;
    assign ws_log_12_data_in = attn_sram_out;
    assign ws_log_12_valid = (ws_state == WS_LOG);
    assign ws_log_12_ready_n = 1'b0;
    assign ws_log_13_data_in = attn_sram_out;
    assign ws_log_13_valid = (ws_state == WS_LOG);
    assign ws_log_13_ready_n = 1'b0;
    assign ws_log_14_data_in = attn_sram_out;
    assign ws_log_14_valid = (ws_state == WS_LOG);
    assign ws_log_14_ready_n = 1'b0;
    assign ws_log_15_data_in = attn_sram_out;
    assign ws_log_15_valid = (ws_state == WS_LOG);
    assign ws_log_15_ready_n = 1'b0;
    assign ws_log_16_data_in = attn_sram_out;
    assign ws_log_16_valid = (ws_state == WS_LOG);
    assign ws_log_16_ready_n = 1'b0;
    assign ws_log_17_data_in = attn_sram_out;
    assign ws_log_17_valid = (ws_state == WS_LOG);
    assign ws_log_17_ready_n = 1'b0;
    assign ws_log_18_data_in = attn_sram_out;
    assign ws_log_18_valid = (ws_state == WS_LOG);
    assign ws_log_18_ready_n = 1'b0;
    assign ws_log_19_data_in = attn_sram_out;
    assign ws_log_19_valid = (ws_state == WS_LOG);
    assign ws_log_19_ready_n = 1'b0;
    assign ws_log_20_data_in = attn_sram_out;
    assign ws_log_20_valid = (ws_state == WS_LOG);
    assign ws_log_20_ready_n = 1'b0;
    assign ws_log_21_data_in = attn_sram_out;
    assign ws_log_21_valid = (ws_state == WS_LOG);
    assign ws_log_21_ready_n = 1'b0;
    assign ws_log_22_data_in = attn_sram_out;
    assign ws_log_22_valid = (ws_state == WS_LOG);
    assign ws_log_22_ready_n = 1'b0;
    assign ws_log_23_data_in = attn_sram_out;
    assign ws_log_23_valid = (ws_state == WS_LOG);
    assign ws_log_23_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_0_valid_n) begin
            log_attn_write_data <= ws_log_0_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_0_valid, ws_exp_0_ready_n, ws_exp_0_ready, ws_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_0_data_in, ws_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_0_valid),
        .ready_n(ws_exp_0_ready_n),
        .data_in(ws_exp_0_data_in),
        .data_out(ws_exp_0_data_out),
        .ready(ws_exp_0_ready),
        .valid_n(ws_exp_0_valid_n)
    );

    wire ws_exp_1_valid, ws_exp_1_ready_n, ws_exp_1_ready, ws_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_1_data_in, ws_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_1_valid),
        .ready_n(ws_exp_1_ready_n),
        .data_in(ws_exp_1_data_in),
        .data_out(ws_exp_1_data_out),
        .ready(ws_exp_1_ready),
        .valid_n(ws_exp_1_valid_n)
    );

    wire ws_exp_2_valid, ws_exp_2_ready_n, ws_exp_2_ready, ws_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_2_data_in, ws_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_2_valid),
        .ready_n(ws_exp_2_ready_n),
        .data_in(ws_exp_2_data_in),
        .data_out(ws_exp_2_data_out),
        .ready(ws_exp_2_ready),
        .valid_n(ws_exp_2_valid_n)
    );

    wire ws_exp_3_valid, ws_exp_3_ready_n, ws_exp_3_ready, ws_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_3_data_in, ws_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_3_valid),
        .ready_n(ws_exp_3_ready_n),
        .data_in(ws_exp_3_data_in),
        .data_out(ws_exp_3_data_out),
        .ready(ws_exp_3_ready),
        .valid_n(ws_exp_3_valid_n)
    );

    wire ws_exp_4_valid, ws_exp_4_ready_n, ws_exp_4_ready, ws_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_4_data_in, ws_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_4_valid),
        .ready_n(ws_exp_4_ready_n),
        .data_in(ws_exp_4_data_in),
        .data_out(ws_exp_4_data_out),
        .ready(ws_exp_4_ready),
        .valid_n(ws_exp_4_valid_n)
    );

    wire ws_exp_5_valid, ws_exp_5_ready_n, ws_exp_5_ready, ws_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_5_data_in, ws_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_5_valid),
        .ready_n(ws_exp_5_ready_n),
        .data_in(ws_exp_5_data_in),
        .data_out(ws_exp_5_data_out),
        .ready(ws_exp_5_ready),
        .valid_n(ws_exp_5_valid_n)
    );

    wire ws_exp_6_valid, ws_exp_6_ready_n, ws_exp_6_ready, ws_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_6_data_in, ws_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_6_valid),
        .ready_n(ws_exp_6_ready_n),
        .data_in(ws_exp_6_data_in),
        .data_out(ws_exp_6_data_out),
        .ready(ws_exp_6_ready),
        .valid_n(ws_exp_6_valid_n)
    );

    wire ws_exp_7_valid, ws_exp_7_ready_n, ws_exp_7_ready, ws_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_7_data_in, ws_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_7_valid),
        .ready_n(ws_exp_7_ready_n),
        .data_in(ws_exp_7_data_in),
        .data_out(ws_exp_7_data_out),
        .ready(ws_exp_7_ready),
        .valid_n(ws_exp_7_valid_n)
    );

    wire ws_exp_8_valid, ws_exp_8_ready_n, ws_exp_8_ready, ws_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_8_data_in, ws_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_8_valid),
        .ready_n(ws_exp_8_ready_n),
        .data_in(ws_exp_8_data_in),
        .data_out(ws_exp_8_data_out),
        .ready(ws_exp_8_ready),
        .valid_n(ws_exp_8_valid_n)
    );

    wire ws_exp_9_valid, ws_exp_9_ready_n, ws_exp_9_ready, ws_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_9_data_in, ws_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_9_valid),
        .ready_n(ws_exp_9_ready_n),
        .data_in(ws_exp_9_data_in),
        .data_out(ws_exp_9_data_out),
        .ready(ws_exp_9_ready),
        .valid_n(ws_exp_9_valid_n)
    );

    wire ws_exp_10_valid, ws_exp_10_ready_n, ws_exp_10_ready, ws_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_10_data_in, ws_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_10_valid),
        .ready_n(ws_exp_10_ready_n),
        .data_in(ws_exp_10_data_in),
        .data_out(ws_exp_10_data_out),
        .ready(ws_exp_10_ready),
        .valid_n(ws_exp_10_valid_n)
    );

    wire ws_exp_11_valid, ws_exp_11_ready_n, ws_exp_11_ready, ws_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_11_data_in, ws_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_11_valid),
        .ready_n(ws_exp_11_ready_n),
        .data_in(ws_exp_11_data_in),
        .data_out(ws_exp_11_data_out),
        .ready(ws_exp_11_ready),
        .valid_n(ws_exp_11_valid_n)
    );

    wire ws_exp_12_valid, ws_exp_12_ready_n, ws_exp_12_ready, ws_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_12_data_in, ws_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_12_valid),
        .ready_n(ws_exp_12_ready_n),
        .data_in(ws_exp_12_data_in),
        .data_out(ws_exp_12_data_out),
        .ready(ws_exp_12_ready),
        .valid_n(ws_exp_12_valid_n)
    );

    wire ws_exp_13_valid, ws_exp_13_ready_n, ws_exp_13_ready, ws_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_13_data_in, ws_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_13_valid),
        .ready_n(ws_exp_13_ready_n),
        .data_in(ws_exp_13_data_in),
        .data_out(ws_exp_13_data_out),
        .ready(ws_exp_13_ready),
        .valid_n(ws_exp_13_valid_n)
    );

    wire ws_exp_14_valid, ws_exp_14_ready_n, ws_exp_14_ready, ws_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_14_data_in, ws_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_14_valid),
        .ready_n(ws_exp_14_ready_n),
        .data_in(ws_exp_14_data_in),
        .data_out(ws_exp_14_data_out),
        .ready(ws_exp_14_ready),
        .valid_n(ws_exp_14_valid_n)
    );

    wire ws_exp_15_valid, ws_exp_15_ready_n, ws_exp_15_ready, ws_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_15_data_in, ws_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_15_valid),
        .ready_n(ws_exp_15_ready_n),
        .data_in(ws_exp_15_data_in),
        .data_out(ws_exp_15_data_out),
        .ready(ws_exp_15_ready),
        .valid_n(ws_exp_15_valid_n)
    );

    wire ws_exp_16_valid, ws_exp_16_ready_n, ws_exp_16_ready, ws_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_16_data_in, ws_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_16_valid),
        .ready_n(ws_exp_16_ready_n),
        .data_in(ws_exp_16_data_in),
        .data_out(ws_exp_16_data_out),
        .ready(ws_exp_16_ready),
        .valid_n(ws_exp_16_valid_n)
    );

    wire ws_exp_17_valid, ws_exp_17_ready_n, ws_exp_17_ready, ws_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_17_data_in, ws_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_17_valid),
        .ready_n(ws_exp_17_ready_n),
        .data_in(ws_exp_17_data_in),
        .data_out(ws_exp_17_data_out),
        .ready(ws_exp_17_ready),
        .valid_n(ws_exp_17_valid_n)
    );

    wire ws_exp_18_valid, ws_exp_18_ready_n, ws_exp_18_ready, ws_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_18_data_in, ws_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_18_valid),
        .ready_n(ws_exp_18_ready_n),
        .data_in(ws_exp_18_data_in),
        .data_out(ws_exp_18_data_out),
        .ready(ws_exp_18_ready),
        .valid_n(ws_exp_18_valid_n)
    );

    wire ws_exp_19_valid, ws_exp_19_ready_n, ws_exp_19_ready, ws_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_19_data_in, ws_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_19_valid),
        .ready_n(ws_exp_19_ready_n),
        .data_in(ws_exp_19_data_in),
        .data_out(ws_exp_19_data_out),
        .ready(ws_exp_19_ready),
        .valid_n(ws_exp_19_valid_n)
    );

    wire ws_exp_20_valid, ws_exp_20_ready_n, ws_exp_20_ready, ws_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_20_data_in, ws_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_20_valid),
        .ready_n(ws_exp_20_ready_n),
        .data_in(ws_exp_20_data_in),
        .data_out(ws_exp_20_data_out),
        .ready(ws_exp_20_ready),
        .valid_n(ws_exp_20_valid_n)
    );

    wire ws_exp_21_valid, ws_exp_21_ready_n, ws_exp_21_ready, ws_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_21_data_in, ws_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_21_valid),
        .ready_n(ws_exp_21_ready_n),
        .data_in(ws_exp_21_data_in),
        .data_out(ws_exp_21_data_out),
        .ready(ws_exp_21_ready),
        .valid_n(ws_exp_21_valid_n)
    );

    wire ws_exp_22_valid, ws_exp_22_ready_n, ws_exp_22_ready, ws_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_22_data_in, ws_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_22_valid),
        .ready_n(ws_exp_22_ready_n),
        .data_in(ws_exp_22_data_in),
        .data_out(ws_exp_22_data_out),
        .ready(ws_exp_22_ready),
        .valid_n(ws_exp_22_valid_n)
    );

    wire ws_exp_23_valid, ws_exp_23_ready_n, ws_exp_23_ready, ws_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_23_data_in, ws_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_23_valid),
        .ready_n(ws_exp_23_ready_n),
        .data_in(ws_exp_23_data_in),
        .data_out(ws_exp_23_data_out),
        .ready(ws_exp_23_ready),
        .valid_n(ws_exp_23_valid_n)
    );

    assign ws_exp_0_data_in = ws_log_sum;
    assign ws_exp_0_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_0_ready_n = 1'b0;
    assign ws_exp_1_data_in = ws_log_sum;
    assign ws_exp_1_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_1_ready_n = 1'b0;
    assign ws_exp_2_data_in = ws_log_sum;
    assign ws_exp_2_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_2_ready_n = 1'b0;
    assign ws_exp_3_data_in = ws_log_sum;
    assign ws_exp_3_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_3_ready_n = 1'b0;
    assign ws_exp_4_data_in = ws_log_sum;
    assign ws_exp_4_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_4_ready_n = 1'b0;
    assign ws_exp_5_data_in = ws_log_sum;
    assign ws_exp_5_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_5_ready_n = 1'b0;
    assign ws_exp_6_data_in = ws_log_sum;
    assign ws_exp_6_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_6_ready_n = 1'b0;
    assign ws_exp_7_data_in = ws_log_sum;
    assign ws_exp_7_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_7_ready_n = 1'b0;
    assign ws_exp_8_data_in = ws_log_sum;
    assign ws_exp_8_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_8_ready_n = 1'b0;
    assign ws_exp_9_data_in = ws_log_sum;
    assign ws_exp_9_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_9_ready_n = 1'b0;
    assign ws_exp_10_data_in = ws_log_sum;
    assign ws_exp_10_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_10_ready_n = 1'b0;
    assign ws_exp_11_data_in = ws_log_sum;
    assign ws_exp_11_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_11_ready_n = 1'b0;
    assign ws_exp_12_data_in = ws_log_sum;
    assign ws_exp_12_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_12_ready_n = 1'b0;
    assign ws_exp_13_data_in = ws_log_sum;
    assign ws_exp_13_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_13_ready_n = 1'b0;
    assign ws_exp_14_data_in = ws_log_sum;
    assign ws_exp_14_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_14_ready_n = 1'b0;
    assign ws_exp_15_data_in = ws_log_sum;
    assign ws_exp_15_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_15_ready_n = 1'b0;
    assign ws_exp_16_data_in = ws_log_sum;
    assign ws_exp_16_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_16_ready_n = 1'b0;
    assign ws_exp_17_data_in = ws_log_sum;
    assign ws_exp_17_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_17_ready_n = 1'b0;
    assign ws_exp_18_data_in = ws_log_sum;
    assign ws_exp_18_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_18_ready_n = 1'b0;
    assign ws_exp_19_data_in = ws_log_sum;
    assign ws_exp_19_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_19_ready_n = 1'b0;
    assign ws_exp_20_data_in = ws_log_sum;
    assign ws_exp_20_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_20_ready_n = 1'b0;
    assign ws_exp_21_data_in = ws_log_sum;
    assign ws_exp_21_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_21_ready_n = 1'b0;
    assign ws_exp_22_data_in = ws_log_sum;
    assign ws_exp_22_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_22_ready_n = 1'b0;
    assign ws_exp_23_data_in = ws_log_sum;
    assign ws_exp_23_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_23_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_0_valid_n) ws_accumulator <= ws_accumulator + ws_exp_0_data_out;
    end

    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    reg [DATA_WIDTH-1:0] out_write_data;
    wire [DATA_WIDTH-1:0] out_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),
              .w_addr(out_write_addr),.sram_data_in(out_write_data),.sram_data_out(out_sram_out));

    localparam WS_IDLE = 3'd0, WS_LOAD_A = 3'd1, WS_LOAD_V = 3'd2,
               WS_LOG = 3'd3, WS_COMPUTE = 3'd4, WS_OUTPUT = 3'd5;
    reg [2:0] ws_state;
    reg [$clog2(N)-1:0] ws_j;
    reg [$clog2(d)-1:0] ws_m;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ws_state <= WS_IDLE; ws_j <= 0; ws_m <= 0;
            attn_write_addr <= 0; attn_read_addr <= 0; attn_w_en <= 0;
            v_write_addr <= 0; v_read_addr <= 0; v_w_en <= 0;
            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;
            out_write_data <= 0;
        end else begin
            attn_w_en <= 0; v_w_en <= 0; out_w_en <= 0;
            case (ws_state)
                WS_IDLE: if (valid_attn || valid_v) ws_state <= WS_LOAD_A;
                WS_LOAD_A: begin
                    if (valid_attn) begin attn_w_en <= 1; attn_write_addr <= attn_write_addr + 1; end
                    if (attn_write_addr == N-1) ws_state <= WS_LOAD_V;
                end
                WS_LOAD_V: begin
                    if (valid_v) begin v_w_en <= 1; v_write_addr <= v_write_addr + 1; end
                    if (v_write_addr == N*d-1) ws_state <= WS_LOG;
                end
                WS_LOG: begin
                    // DPE(I|log) converts attn to log domain
                    attn_read_addr <= ws_j;
                    ws_j <= ws_j + 1;
                    if (ws_j == N-1) begin ws_j <= 0; ws_state <= WS_COMPUTE; end
                end
                WS_COMPUTE: begin
                    // CLB add + DPE(I|exp) + accumulate
                    log_attn_read_addr <= ws_j;
                    v_read_addr <= (ws_j << $clog2(d)) + ws_m;
                    ws_j <= ws_j + 1;
                    if (ws_j == N-1) begin
                        out_write_data <= ws_accumulator[2*DATA_WIDTH-1:DATA_WIDTH];
                        out_w_en <= 1; out_write_addr <= ws_m;
                        ws_j <= 0; ws_m <= ws_m + 1;
                        if (ws_m == d-1) ws_state <= WS_OUTPUT;
                    end
                end
                WS_OUTPUT: if (ready_n) out_read_addr <= out_read_addr + 1;
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (ws_state == WS_OUTPUT);
    assign ready_attn = (ws_state == WS_LOAD_A || ws_state == WS_IDLE);
    assign ready_v = (ws_state == WS_LOAD_V || ws_state == WS_IDLE);
endmodule

// DIMM — Block 1, Head 1, Lane 0 (depth=393216)
module dimm_score_matrix_b1_h1 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid_q, input wire valid_k,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_q,
    input wire [DATA_WIDTH-1:0] data_in_k,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_q, output wire ready_k,
    output wire valid_n
);

    reg [ADDR_WIDTH-1:0] q_write_addr, q_read_addr;
    reg q_w_en;
    wire [DATA_WIDTH-1:0] q_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    q_sram (.clk(clk),.rst(rst),.w_en(q_w_en),.r_addr(q_read_addr),
            .w_addr(q_write_addr),.sram_data_in(data_in_q),.sram_data_out(q_sram_out));

    reg [$clog2(N*d)-1:0] k_write_addr, k_read_addr_a;
    reg k_w_en;
    wire [DATA_WIDTH-1:0] k_sram_out_a;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    k_sram_a (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_a[$clog2(N*d)-1:0]),
              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_a));
    reg [$clog2(N*d)-1:0] k_read_addr_b;
    wire [DATA_WIDTH-1:0] k_sram_out_b;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    k_sram_b (.clk(clk),.rst(rst),.w_en(k_w_en),.r_addr(k_read_addr_b[$clog2(N*d)-1:0]),
              .w_addr(k_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_k),.sram_data_out(k_sram_out_b));

    // CLB adder A: log_Q + log_K[j₁] (log-domain addition)
    wire [DATA_WIDTH-1:0] log_sum_a = q_sram_out + k_sram_out_a;
    // CLB adder B: log_Q + log_K[j₂] (second vector for dual-identity)
    wire [DATA_WIDTH-1:0] log_sum_b = q_sram_out + k_sram_out_b;

    // DPE(I|exp) stage_0: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_0_valid, dimm_exp_0_ready_n, dimm_exp_0_ready, dimm_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_0_data_in, dimm_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_0_valid),
        .ready_n(dimm_exp_0_ready_n),
        .data_in(dimm_exp_0_data_in),
        .data_out(dimm_exp_0_data_out),
        .ready(dimm_exp_0_ready),
        .valid_n(dimm_exp_0_valid_n)
    );

    // DPE(I|exp) stage_1: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_1_valid, dimm_exp_1_ready_n, dimm_exp_1_ready, dimm_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_1_data_in, dimm_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_1_valid),
        .ready_n(dimm_exp_1_ready_n),
        .data_in(dimm_exp_1_data_in),
        .data_out(dimm_exp_1_data_out),
        .ready(dimm_exp_1_ready),
        .valid_n(dimm_exp_1_valid_n)
    );

    // DPE(I|exp) stage_2: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_2_valid, dimm_exp_2_ready_n, dimm_exp_2_ready, dimm_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_2_data_in, dimm_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_2_valid),
        .ready_n(dimm_exp_2_ready_n),
        .data_in(dimm_exp_2_data_in),
        .data_out(dimm_exp_2_data_out),
        .ready(dimm_exp_2_ready),
        .valid_n(dimm_exp_2_valid_n)
    );

    // DPE(I|exp) stage_3: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_3_valid, dimm_exp_3_ready_n, dimm_exp_3_ready, dimm_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_3_data_in, dimm_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_3_valid),
        .ready_n(dimm_exp_3_ready_n),
        .data_in(dimm_exp_3_data_in),
        .data_out(dimm_exp_3_data_out),
        .ready(dimm_exp_3_ready),
        .valid_n(dimm_exp_3_valid_n)
    );

    // DPE(I|exp) stage_4: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_4_valid, dimm_exp_4_ready_n, dimm_exp_4_ready, dimm_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_4_data_in, dimm_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_4_valid),
        .ready_n(dimm_exp_4_ready_n),
        .data_in(dimm_exp_4_data_in),
        .data_out(dimm_exp_4_data_out),
        .ready(dimm_exp_4_ready),
        .valid_n(dimm_exp_4_valid_n)
    );

    // DPE(I|exp) stage_5: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_5_valid, dimm_exp_5_ready_n, dimm_exp_5_ready, dimm_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_5_data_in, dimm_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_5_valid),
        .ready_n(dimm_exp_5_ready_n),
        .data_in(dimm_exp_5_data_in),
        .data_out(dimm_exp_5_data_out),
        .ready(dimm_exp_5_ready),
        .valid_n(dimm_exp_5_valid_n)
    );

    // DPE(I|exp) stage_6: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_6_valid, dimm_exp_6_ready_n, dimm_exp_6_ready, dimm_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_6_data_in, dimm_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_6_valid),
        .ready_n(dimm_exp_6_ready_n),
        .data_in(dimm_exp_6_data_in),
        .data_out(dimm_exp_6_data_out),
        .ready(dimm_exp_6_ready),
        .valid_n(dimm_exp_6_valid_n)
    );

    // DPE(I|exp) stage_7: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_7_valid, dimm_exp_7_ready_n, dimm_exp_7_ready, dimm_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_7_data_in, dimm_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_7_valid),
        .ready_n(dimm_exp_7_ready_n),
        .data_in(dimm_exp_7_data_in),
        .data_out(dimm_exp_7_data_out),
        .ready(dimm_exp_7_ready),
        .valid_n(dimm_exp_7_valid_n)
    );

    // DPE(I|exp) stage_8: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_8_valid, dimm_exp_8_ready_n, dimm_exp_8_ready, dimm_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_8_data_in, dimm_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_8_valid),
        .ready_n(dimm_exp_8_ready_n),
        .data_in(dimm_exp_8_data_in),
        .data_out(dimm_exp_8_data_out),
        .ready(dimm_exp_8_ready),
        .valid_n(dimm_exp_8_valid_n)
    );

    // DPE(I|exp) stage_9: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_9_valid, dimm_exp_9_ready_n, dimm_exp_9_ready, dimm_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_9_data_in, dimm_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_9_valid),
        .ready_n(dimm_exp_9_ready_n),
        .data_in(dimm_exp_9_data_in),
        .data_out(dimm_exp_9_data_out),
        .ready(dimm_exp_9_ready),
        .valid_n(dimm_exp_9_valid_n)
    );

    // DPE(I|exp) stage_10: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_10_valid, dimm_exp_10_ready_n, dimm_exp_10_ready, dimm_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_10_data_in, dimm_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_10_valid),
        .ready_n(dimm_exp_10_ready_n),
        .data_in(dimm_exp_10_data_in),
        .data_out(dimm_exp_10_data_out),
        .ready(dimm_exp_10_ready),
        .valid_n(dimm_exp_10_valid_n)
    );

    // DPE(I|exp) stage_11: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_11_valid, dimm_exp_11_ready_n, dimm_exp_11_ready, dimm_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_11_data_in, dimm_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_11_valid),
        .ready_n(dimm_exp_11_ready_n),
        .data_in(dimm_exp_11_data_in),
        .data_out(dimm_exp_11_data_out),
        .ready(dimm_exp_11_ready),
        .valid_n(dimm_exp_11_valid_n)
    );

    // DPE(I|exp) stage_12: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_12_valid, dimm_exp_12_ready_n, dimm_exp_12_ready, dimm_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_12_data_in, dimm_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_12_valid),
        .ready_n(dimm_exp_12_ready_n),
        .data_in(dimm_exp_12_data_in),
        .data_out(dimm_exp_12_data_out),
        .ready(dimm_exp_12_ready),
        .valid_n(dimm_exp_12_valid_n)
    );

    // DPE(I|exp) stage_13: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_13_valid, dimm_exp_13_ready_n, dimm_exp_13_ready, dimm_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_13_data_in, dimm_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_13_valid),
        .ready_n(dimm_exp_13_ready_n),
        .data_in(dimm_exp_13_data_in),
        .data_out(dimm_exp_13_data_out),
        .ready(dimm_exp_13_ready),
        .valid_n(dimm_exp_13_valid_n)
    );

    // DPE(I|exp) stage_14: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_14_valid, dimm_exp_14_ready_n, dimm_exp_14_ready, dimm_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_14_data_in, dimm_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_14_valid),
        .ready_n(dimm_exp_14_ready_n),
        .data_in(dimm_exp_14_data_in),
        .data_out(dimm_exp_14_data_out),
        .ready(dimm_exp_14_ready),
        .valid_n(dimm_exp_14_valid_n)
    );

    // DPE(I|exp) stage_15: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_15_valid, dimm_exp_15_ready_n, dimm_exp_15_ready, dimm_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_15_data_in, dimm_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_15_valid),
        .ready_n(dimm_exp_15_ready_n),
        .data_in(dimm_exp_15_data_in),
        .data_out(dimm_exp_15_data_out),
        .ready(dimm_exp_15_ready),
        .valid_n(dimm_exp_15_valid_n)
    );

    // DPE(I|exp) stage_16: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_16_valid, dimm_exp_16_ready_n, dimm_exp_16_ready, dimm_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_16_data_in, dimm_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_16_valid),
        .ready_n(dimm_exp_16_ready_n),
        .data_in(dimm_exp_16_data_in),
        .data_out(dimm_exp_16_data_out),
        .ready(dimm_exp_16_ready),
        .valid_n(dimm_exp_16_valid_n)
    );

    // DPE(I|exp) stage_17: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_17_valid, dimm_exp_17_ready_n, dimm_exp_17_ready, dimm_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_17_data_in, dimm_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_17_valid),
        .ready_n(dimm_exp_17_ready_n),
        .data_in(dimm_exp_17_data_in),
        .data_out(dimm_exp_17_data_out),
        .ready(dimm_exp_17_ready),
        .valid_n(dimm_exp_17_valid_n)
    );

    // DPE(I|exp) stage_18: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_18_valid, dimm_exp_18_ready_n, dimm_exp_18_ready, dimm_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_18_data_in, dimm_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_18_valid),
        .ready_n(dimm_exp_18_ready_n),
        .data_in(dimm_exp_18_data_in),
        .data_out(dimm_exp_18_data_out),
        .ready(dimm_exp_18_ready),
        .valid_n(dimm_exp_18_valid_n)
    );

    // DPE(I|exp) stage_19: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_19_valid, dimm_exp_19_ready_n, dimm_exp_19_ready, dimm_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_19_data_in, dimm_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_19_valid),
        .ready_n(dimm_exp_19_ready_n),
        .data_in(dimm_exp_19_data_in),
        .data_out(dimm_exp_19_data_out),
        .ready(dimm_exp_19_ready),
        .valid_n(dimm_exp_19_valid_n)
    );

    // DPE(I|exp) stage_20: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_20_valid, dimm_exp_20_ready_n, dimm_exp_20_ready, dimm_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_20_data_in, dimm_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_20_valid),
        .ready_n(dimm_exp_20_ready_n),
        .data_in(dimm_exp_20_data_in),
        .data_out(dimm_exp_20_data_out),
        .ready(dimm_exp_20_ready),
        .valid_n(dimm_exp_20_valid_n)
    );

    // DPE(I|exp) stage_21: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_21_valid, dimm_exp_21_ready_n, dimm_exp_21_ready, dimm_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_21_data_in, dimm_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_21_valid),
        .ready_n(dimm_exp_21_ready_n),
        .data_in(dimm_exp_21_data_in),
        .data_out(dimm_exp_21_data_out),
        .ready(dimm_exp_21_ready),
        .valid_n(dimm_exp_21_valid_n)
    );

    // DPE(I|exp) stage_22: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_22_valid, dimm_exp_22_ready_n, dimm_exp_22_ready, dimm_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_22_data_in, dimm_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_22_valid),
        .ready_n(dimm_exp_22_ready_n),
        .data_in(dimm_exp_22_data_in),
        .data_out(dimm_exp_22_data_out),
        .ready(dimm_exp_22_ready),
        .valid_n(dimm_exp_22_valid_n)
    );

    // DPE(I|exp) stage_23: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_23_valid, dimm_exp_23_ready_n, dimm_exp_23_ready, dimm_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_23_data_in, dimm_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) dimm_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_23_valid),
        .ready_n(dimm_exp_23_ready_n),
        .data_in(dimm_exp_23_data_in),
        .data_out(dimm_exp_23_data_out),
        .ready(dimm_exp_23_ready),
        .valid_n(dimm_exp_23_valid_n)
    );

    reg [4:0] dimm_sel;
    assign dimm_exp_0_data_in = log_sum_a;
    assign dimm_exp_0_valid = (state == S_COMPUTE) && (dimm_sel == 0);
    assign dimm_exp_0_ready_n = 1'b0;
    assign dimm_exp_1_data_in = log_sum_a;
    assign dimm_exp_1_valid = (state == S_COMPUTE) && (dimm_sel == 1);
    assign dimm_exp_1_ready_n = 1'b0;
    assign dimm_exp_2_data_in = log_sum_a;
    assign dimm_exp_2_valid = (state == S_COMPUTE) && (dimm_sel == 2);
    assign dimm_exp_2_ready_n = 1'b0;
    assign dimm_exp_3_data_in = log_sum_a;
    assign dimm_exp_3_valid = (state == S_COMPUTE) && (dimm_sel == 3);
    assign dimm_exp_3_ready_n = 1'b0;
    assign dimm_exp_4_data_in = log_sum_a;
    assign dimm_exp_4_valid = (state == S_COMPUTE) && (dimm_sel == 4);
    assign dimm_exp_4_ready_n = 1'b0;
    assign dimm_exp_5_data_in = log_sum_a;
    assign dimm_exp_5_valid = (state == S_COMPUTE) && (dimm_sel == 5);
    assign dimm_exp_5_ready_n = 1'b0;
    assign dimm_exp_6_data_in = log_sum_a;
    assign dimm_exp_6_valid = (state == S_COMPUTE) && (dimm_sel == 6);
    assign dimm_exp_6_ready_n = 1'b0;
    assign dimm_exp_7_data_in = log_sum_a;
    assign dimm_exp_7_valid = (state == S_COMPUTE) && (dimm_sel == 7);
    assign dimm_exp_7_ready_n = 1'b0;
    assign dimm_exp_8_data_in = log_sum_a;
    assign dimm_exp_8_valid = (state == S_COMPUTE) && (dimm_sel == 8);
    assign dimm_exp_8_ready_n = 1'b0;
    assign dimm_exp_9_data_in = log_sum_a;
    assign dimm_exp_9_valid = (state == S_COMPUTE) && (dimm_sel == 9);
    assign dimm_exp_9_ready_n = 1'b0;
    assign dimm_exp_10_data_in = log_sum_a;
    assign dimm_exp_10_valid = (state == S_COMPUTE) && (dimm_sel == 10);
    assign dimm_exp_10_ready_n = 1'b0;
    assign dimm_exp_11_data_in = log_sum_a;
    assign dimm_exp_11_valid = (state == S_COMPUTE) && (dimm_sel == 11);
    assign dimm_exp_11_ready_n = 1'b0;
    assign dimm_exp_12_data_in = log_sum_a;
    assign dimm_exp_12_valid = (state == S_COMPUTE) && (dimm_sel == 12);
    assign dimm_exp_12_ready_n = 1'b0;
    assign dimm_exp_13_data_in = log_sum_a;
    assign dimm_exp_13_valid = (state == S_COMPUTE) && (dimm_sel == 13);
    assign dimm_exp_13_ready_n = 1'b0;
    assign dimm_exp_14_data_in = log_sum_a;
    assign dimm_exp_14_valid = (state == S_COMPUTE) && (dimm_sel == 14);
    assign dimm_exp_14_ready_n = 1'b0;
    assign dimm_exp_15_data_in = log_sum_a;
    assign dimm_exp_15_valid = (state == S_COMPUTE) && (dimm_sel == 15);
    assign dimm_exp_15_ready_n = 1'b0;
    assign dimm_exp_16_data_in = log_sum_a;
    assign dimm_exp_16_valid = (state == S_COMPUTE) && (dimm_sel == 16);
    assign dimm_exp_16_ready_n = 1'b0;
    assign dimm_exp_17_data_in = log_sum_a;
    assign dimm_exp_17_valid = (state == S_COMPUTE) && (dimm_sel == 17);
    assign dimm_exp_17_ready_n = 1'b0;
    assign dimm_exp_18_data_in = log_sum_a;
    assign dimm_exp_18_valid = (state == S_COMPUTE) && (dimm_sel == 18);
    assign dimm_exp_18_ready_n = 1'b0;
    assign dimm_exp_19_data_in = log_sum_a;
    assign dimm_exp_19_valid = (state == S_COMPUTE) && (dimm_sel == 19);
    assign dimm_exp_19_ready_n = 1'b0;
    assign dimm_exp_20_data_in = log_sum_a;
    assign dimm_exp_20_valid = (state == S_COMPUTE) && (dimm_sel == 20);
    assign dimm_exp_20_ready_n = 1'b0;
    assign dimm_exp_21_data_in = log_sum_a;
    assign dimm_exp_21_valid = (state == S_COMPUTE) && (dimm_sel == 21);
    assign dimm_exp_21_ready_n = 1'b0;
    assign dimm_exp_22_data_in = log_sum_a;
    assign dimm_exp_22_valid = (state == S_COMPUTE) && (dimm_sel == 22);
    assign dimm_exp_22_ready_n = 1'b0;
    assign dimm_exp_23_data_in = log_sum_a;
    assign dimm_exp_23_valid = (state == S_COMPUTE) && (dimm_sel == 23);
    assign dimm_exp_23_ready_n = 1'b0;

    // Dual accumulators: one per vector (A for j₁, B for j₂)
    reg [2*DATA_WIDTH-1:0] accumulator_a, accumulator_b;
    reg acc_phase;  // 0=accumulating A outputs, 1=accumulating B outputs
    always @(posedge clk) begin
        if (rst) begin accumulator_a <= 0; accumulator_b <= 0; end
        else if (dimm_exp_valid_n) begin
            if (!acc_phase) accumulator_a <= accumulator_a + dimm_exp_data_out;
            else accumulator_b <= accumulator_b + dimm_exp_data_out;
        end
    end

    reg [ADDR_WIDTH-1:0] score_write_addr, score_read_addr;
    reg score_w_en;
    reg [DATA_WIDTH-1:0] score_write_data;
    wire [DATA_WIDTH-1:0] score_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    score_sram (.clk(clk),.rst(rst),.w_en(score_w_en),.r_addr(score_read_addr),
                .w_addr(score_write_addr),.sram_data_in(score_write_data),.sram_data_out(score_sram_out));

    localparam S_IDLE = 3'd0, S_LOAD_Q = 3'd1, S_LOAD_K = 3'd2,
               S_COMPUTE = 3'd3, S_OUTPUT = 3'd4;
    localparam S_WRITE_B = 3'd5;  // write second score for dual-identity
    reg [2:0] state;
    reg [$clog2(d)-1:0] mac_count;
    reg [$clog2(N)-1:0] score_idx;
    reg feed_half;  // 0=feeding vector A (d elements), 1=feeding vector B

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            q_write_addr <= 0; q_read_addr <= 0; q_w_en <= 0;
            k_write_addr <= 0; k_read_addr_a <= 0; k_w_en <= 0;
            k_read_addr_b <= 0;
            feed_half <= 0; feed_phase <= 0; acc_phase <= 0;
            score_write_addr <= 0; score_read_addr <= 0; score_w_en <= 0;
            score_write_data <= 0;
            mac_count <= 0; score_idx <= 0;
            dimm_sel <= 0;
        end else begin
            q_w_en <= 0; k_w_en <= 0; score_w_en <= 0;
            case (state)
                S_IDLE: if (valid_q || valid_k) state <= S_LOAD_Q;
                S_LOAD_Q: begin
                    if (valid_q) begin q_w_en <= 1; q_write_addr <= q_write_addr + 1; end
                    if (q_write_addr == d-1) state <= S_LOAD_K;
                end
                S_LOAD_K: begin
                    if (valid_k) begin k_w_en <= 1; k_write_addr <= k_write_addr + 1; end
                    if (k_write_addr == N*d-1) state <= S_COMPUTE;
                end
                S_COMPUTE: begin
                    // Dual-identity: feed vector A then vector B per DPE pass
                    q_read_addr <= mac_count;
                    k_read_addr_a <= (score_idx << $clog2(d)) + mac_count;
                    k_read_addr_b <= ((score_idx + 1) << $clog2(d)) + mac_count;
                    if (!feed_half) begin
                        // Feeding vector A (log_sum_a)
                        feed_phase <= 0; acc_phase <= 0;
                        mac_count <= mac_count + 1;
                        if (mac_count == d-1) begin
                            mac_count <= 0;
                            feed_half <= 1;
                        end
                    end else begin
                        // Feeding vector B (log_sum_b)
                        feed_phase <= 1; acc_phase <= 1;
                        mac_count <= mac_count + 1;
                        if (mac_count == d-1) begin
                            // Both vectors done — write score A, then B
                            score_write_data <= accumulator_a[2*DATA_WIDTH-1:DATA_WIDTH];
                            score_w_en <= 1;
                            score_write_addr <= score_idx;
                            mac_count <= 0;
                            feed_half <= 0;
                            state <= S_WRITE_B;
                        end
                    end
                end
                S_WRITE_B: begin
                    // Write second score (j₂ = score_idx + 1)
                    score_write_data <= accumulator_b[2*DATA_WIDTH-1:DATA_WIDTH];
                    score_w_en <= 1;
                    score_write_addr <= score_idx + 1;
                    accumulator_a <= 0; accumulator_b <= 0;
                    if (score_idx >= N-2) state <= S_OUTPUT;
                    else begin score_idx <= score_idx + 2; state <= S_COMPUTE; end
                end
                S_OUTPUT: if (ready_n) score_read_addr <= score_read_addr + 1;
            endcase
        end
    end

    assign data_out = score_sram_out;
    assign valid_n = (state == S_OUTPUT);
    assign ready_q = (state == S_LOAD_Q || state == S_IDLE);
    assign ready_k = (state == S_LOAD_K || state == S_IDLE);
endmodule
module softmax_approx_b1_h1 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid, input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, output wire valid_n
);

    reg [ADDR_WIDTH-1:0] in_write_addr, in_read_addr;
    reg in_w_en;
    wire [DATA_WIDTH-1:0] in_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    in_sram (.clk(clk),.rst(rst),.w_en(in_w_en),.r_addr(in_read_addr),
             .w_addr(in_write_addr),.sram_data_in(data_in),.sram_data_out(in_sram_out));

    wire sm_exp_0_valid, sm_exp_0_ready_n, sm_exp_0_ready, sm_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_0_data_in, sm_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_0_valid),
        .ready_n(sm_exp_0_ready_n),
        .data_in(sm_exp_0_data_in),
        .data_out(sm_exp_0_data_out),
        .ready(sm_exp_0_ready),
        .valid_n(sm_exp_0_valid_n)
    );

    wire sm_exp_1_valid, sm_exp_1_ready_n, sm_exp_1_ready, sm_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_1_data_in, sm_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_1_valid),
        .ready_n(sm_exp_1_ready_n),
        .data_in(sm_exp_1_data_in),
        .data_out(sm_exp_1_data_out),
        .ready(sm_exp_1_ready),
        .valid_n(sm_exp_1_valid_n)
    );

    wire sm_exp_2_valid, sm_exp_2_ready_n, sm_exp_2_ready, sm_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_2_data_in, sm_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_2_valid),
        .ready_n(sm_exp_2_ready_n),
        .data_in(sm_exp_2_data_in),
        .data_out(sm_exp_2_data_out),
        .ready(sm_exp_2_ready),
        .valid_n(sm_exp_2_valid_n)
    );

    wire sm_exp_3_valid, sm_exp_3_ready_n, sm_exp_3_ready, sm_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_3_data_in, sm_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_3_valid),
        .ready_n(sm_exp_3_ready_n),
        .data_in(sm_exp_3_data_in),
        .data_out(sm_exp_3_data_out),
        .ready(sm_exp_3_ready),
        .valid_n(sm_exp_3_valid_n)
    );

    wire sm_exp_4_valid, sm_exp_4_ready_n, sm_exp_4_ready, sm_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_4_data_in, sm_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_4_valid),
        .ready_n(sm_exp_4_ready_n),
        .data_in(sm_exp_4_data_in),
        .data_out(sm_exp_4_data_out),
        .ready(sm_exp_4_ready),
        .valid_n(sm_exp_4_valid_n)
    );

    wire sm_exp_5_valid, sm_exp_5_ready_n, sm_exp_5_ready, sm_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_5_data_in, sm_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_5_valid),
        .ready_n(sm_exp_5_ready_n),
        .data_in(sm_exp_5_data_in),
        .data_out(sm_exp_5_data_out),
        .ready(sm_exp_5_ready),
        .valid_n(sm_exp_5_valid_n)
    );

    wire sm_exp_6_valid, sm_exp_6_ready_n, sm_exp_6_ready, sm_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_6_data_in, sm_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_6_valid),
        .ready_n(sm_exp_6_ready_n),
        .data_in(sm_exp_6_data_in),
        .data_out(sm_exp_6_data_out),
        .ready(sm_exp_6_ready),
        .valid_n(sm_exp_6_valid_n)
    );

    wire sm_exp_7_valid, sm_exp_7_ready_n, sm_exp_7_ready, sm_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_7_data_in, sm_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_7_valid),
        .ready_n(sm_exp_7_ready_n),
        .data_in(sm_exp_7_data_in),
        .data_out(sm_exp_7_data_out),
        .ready(sm_exp_7_ready),
        .valid_n(sm_exp_7_valid_n)
    );

    wire sm_exp_8_valid, sm_exp_8_ready_n, sm_exp_8_ready, sm_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_8_data_in, sm_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_8_valid),
        .ready_n(sm_exp_8_ready_n),
        .data_in(sm_exp_8_data_in),
        .data_out(sm_exp_8_data_out),
        .ready(sm_exp_8_ready),
        .valid_n(sm_exp_8_valid_n)
    );

    wire sm_exp_9_valid, sm_exp_9_ready_n, sm_exp_9_ready, sm_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_9_data_in, sm_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_9_valid),
        .ready_n(sm_exp_9_ready_n),
        .data_in(sm_exp_9_data_in),
        .data_out(sm_exp_9_data_out),
        .ready(sm_exp_9_ready),
        .valid_n(sm_exp_9_valid_n)
    );

    wire sm_exp_10_valid, sm_exp_10_ready_n, sm_exp_10_ready, sm_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_10_data_in, sm_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_10_valid),
        .ready_n(sm_exp_10_ready_n),
        .data_in(sm_exp_10_data_in),
        .data_out(sm_exp_10_data_out),
        .ready(sm_exp_10_ready),
        .valid_n(sm_exp_10_valid_n)
    );

    wire sm_exp_11_valid, sm_exp_11_ready_n, sm_exp_11_ready, sm_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_11_data_in, sm_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_11_valid),
        .ready_n(sm_exp_11_ready_n),
        .data_in(sm_exp_11_data_in),
        .data_out(sm_exp_11_data_out),
        .ready(sm_exp_11_ready),
        .valid_n(sm_exp_11_valid_n)
    );

    wire sm_exp_12_valid, sm_exp_12_ready_n, sm_exp_12_ready, sm_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_12_data_in, sm_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_12_valid),
        .ready_n(sm_exp_12_ready_n),
        .data_in(sm_exp_12_data_in),
        .data_out(sm_exp_12_data_out),
        .ready(sm_exp_12_ready),
        .valid_n(sm_exp_12_valid_n)
    );

    wire sm_exp_13_valid, sm_exp_13_ready_n, sm_exp_13_ready, sm_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_13_data_in, sm_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_13_valid),
        .ready_n(sm_exp_13_ready_n),
        .data_in(sm_exp_13_data_in),
        .data_out(sm_exp_13_data_out),
        .ready(sm_exp_13_ready),
        .valid_n(sm_exp_13_valid_n)
    );

    wire sm_exp_14_valid, sm_exp_14_ready_n, sm_exp_14_ready, sm_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_14_data_in, sm_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_14_valid),
        .ready_n(sm_exp_14_ready_n),
        .data_in(sm_exp_14_data_in),
        .data_out(sm_exp_14_data_out),
        .ready(sm_exp_14_ready),
        .valid_n(sm_exp_14_valid_n)
    );

    wire sm_exp_15_valid, sm_exp_15_ready_n, sm_exp_15_ready, sm_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_15_data_in, sm_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_15_valid),
        .ready_n(sm_exp_15_ready_n),
        .data_in(sm_exp_15_data_in),
        .data_out(sm_exp_15_data_out),
        .ready(sm_exp_15_ready),
        .valid_n(sm_exp_15_valid_n)
    );

    wire sm_exp_16_valid, sm_exp_16_ready_n, sm_exp_16_ready, sm_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_16_data_in, sm_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_16_valid),
        .ready_n(sm_exp_16_ready_n),
        .data_in(sm_exp_16_data_in),
        .data_out(sm_exp_16_data_out),
        .ready(sm_exp_16_ready),
        .valid_n(sm_exp_16_valid_n)
    );

    wire sm_exp_17_valid, sm_exp_17_ready_n, sm_exp_17_ready, sm_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_17_data_in, sm_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_17_valid),
        .ready_n(sm_exp_17_ready_n),
        .data_in(sm_exp_17_data_in),
        .data_out(sm_exp_17_data_out),
        .ready(sm_exp_17_ready),
        .valid_n(sm_exp_17_valid_n)
    );

    wire sm_exp_18_valid, sm_exp_18_ready_n, sm_exp_18_ready, sm_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_18_data_in, sm_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_18_valid),
        .ready_n(sm_exp_18_ready_n),
        .data_in(sm_exp_18_data_in),
        .data_out(sm_exp_18_data_out),
        .ready(sm_exp_18_ready),
        .valid_n(sm_exp_18_valid_n)
    );

    wire sm_exp_19_valid, sm_exp_19_ready_n, sm_exp_19_ready, sm_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_19_data_in, sm_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_19_valid),
        .ready_n(sm_exp_19_ready_n),
        .data_in(sm_exp_19_data_in),
        .data_out(sm_exp_19_data_out),
        .ready(sm_exp_19_ready),
        .valid_n(sm_exp_19_valid_n)
    );

    wire sm_exp_20_valid, sm_exp_20_ready_n, sm_exp_20_ready, sm_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_20_data_in, sm_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_20_valid),
        .ready_n(sm_exp_20_ready_n),
        .data_in(sm_exp_20_data_in),
        .data_out(sm_exp_20_data_out),
        .ready(sm_exp_20_ready),
        .valid_n(sm_exp_20_valid_n)
    );

    wire sm_exp_21_valid, sm_exp_21_ready_n, sm_exp_21_ready, sm_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_21_data_in, sm_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_21_valid),
        .ready_n(sm_exp_21_ready_n),
        .data_in(sm_exp_21_data_in),
        .data_out(sm_exp_21_data_out),
        .ready(sm_exp_21_ready),
        .valid_n(sm_exp_21_valid_n)
    );

    wire sm_exp_22_valid, sm_exp_22_ready_n, sm_exp_22_ready, sm_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_22_data_in, sm_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_22_valid),
        .ready_n(sm_exp_22_ready_n),
        .data_in(sm_exp_22_data_in),
        .data_out(sm_exp_22_data_out),
        .ready(sm_exp_22_ready),
        .valid_n(sm_exp_22_valid_n)
    );

    wire sm_exp_23_valid, sm_exp_23_ready_n, sm_exp_23_ready, sm_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_23_data_in, sm_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) sm_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_23_valid),
        .ready_n(sm_exp_23_ready_n),
        .data_in(sm_exp_23_data_in),
        .data_out(sm_exp_23_data_out),
        .ready(sm_exp_23_ready),
        .valid_n(sm_exp_23_valid_n)
    );

    assign sm_exp_0_data_in = in_sram_out;
    assign sm_exp_0_valid = (sm_state == SM_EXP);
    assign sm_exp_0_ready_n = 1'b0;
    assign sm_exp_1_data_in = in_sram_out;
    assign sm_exp_1_valid = (sm_state == SM_EXP);
    assign sm_exp_1_ready_n = 1'b0;
    assign sm_exp_2_data_in = in_sram_out;
    assign sm_exp_2_valid = (sm_state == SM_EXP);
    assign sm_exp_2_ready_n = 1'b0;
    assign sm_exp_3_data_in = in_sram_out;
    assign sm_exp_3_valid = (sm_state == SM_EXP);
    assign sm_exp_3_ready_n = 1'b0;
    assign sm_exp_4_data_in = in_sram_out;
    assign sm_exp_4_valid = (sm_state == SM_EXP);
    assign sm_exp_4_ready_n = 1'b0;
    assign sm_exp_5_data_in = in_sram_out;
    assign sm_exp_5_valid = (sm_state == SM_EXP);
    assign sm_exp_5_ready_n = 1'b0;
    assign sm_exp_6_data_in = in_sram_out;
    assign sm_exp_6_valid = (sm_state == SM_EXP);
    assign sm_exp_6_ready_n = 1'b0;
    assign sm_exp_7_data_in = in_sram_out;
    assign sm_exp_7_valid = (sm_state == SM_EXP);
    assign sm_exp_7_ready_n = 1'b0;
    assign sm_exp_8_data_in = in_sram_out;
    assign sm_exp_8_valid = (sm_state == SM_EXP);
    assign sm_exp_8_ready_n = 1'b0;
    assign sm_exp_9_data_in = in_sram_out;
    assign sm_exp_9_valid = (sm_state == SM_EXP);
    assign sm_exp_9_ready_n = 1'b0;
    assign sm_exp_10_data_in = in_sram_out;
    assign sm_exp_10_valid = (sm_state == SM_EXP);
    assign sm_exp_10_ready_n = 1'b0;
    assign sm_exp_11_data_in = in_sram_out;
    assign sm_exp_11_valid = (sm_state == SM_EXP);
    assign sm_exp_11_ready_n = 1'b0;
    assign sm_exp_12_data_in = in_sram_out;
    assign sm_exp_12_valid = (sm_state == SM_EXP);
    assign sm_exp_12_ready_n = 1'b0;
    assign sm_exp_13_data_in = in_sram_out;
    assign sm_exp_13_valid = (sm_state == SM_EXP);
    assign sm_exp_13_ready_n = 1'b0;
    assign sm_exp_14_data_in = in_sram_out;
    assign sm_exp_14_valid = (sm_state == SM_EXP);
    assign sm_exp_14_ready_n = 1'b0;
    assign sm_exp_15_data_in = in_sram_out;
    assign sm_exp_15_valid = (sm_state == SM_EXP);
    assign sm_exp_15_ready_n = 1'b0;
    assign sm_exp_16_data_in = in_sram_out;
    assign sm_exp_16_valid = (sm_state == SM_EXP);
    assign sm_exp_16_ready_n = 1'b0;
    assign sm_exp_17_data_in = in_sram_out;
    assign sm_exp_17_valid = (sm_state == SM_EXP);
    assign sm_exp_17_ready_n = 1'b0;
    assign sm_exp_18_data_in = in_sram_out;
    assign sm_exp_18_valid = (sm_state == SM_EXP);
    assign sm_exp_18_ready_n = 1'b0;
    assign sm_exp_19_data_in = in_sram_out;
    assign sm_exp_19_valid = (sm_state == SM_EXP);
    assign sm_exp_19_ready_n = 1'b0;
    assign sm_exp_20_data_in = in_sram_out;
    assign sm_exp_20_valid = (sm_state == SM_EXP);
    assign sm_exp_20_ready_n = 1'b0;
    assign sm_exp_21_data_in = in_sram_out;
    assign sm_exp_21_valid = (sm_state == SM_EXP);
    assign sm_exp_21_ready_n = 1'b0;
    assign sm_exp_22_data_in = in_sram_out;
    assign sm_exp_22_valid = (sm_state == SM_EXP);
    assign sm_exp_22_ready_n = 1'b0;
    assign sm_exp_23_data_in = in_sram_out;
    assign sm_exp_23_valid = (sm_state == SM_EXP);
    assign sm_exp_23_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] exp_write_addr, exp_read_addr;
    reg exp_w_en;
    reg [DATA_WIDTH-1:0] exp_write_data;
    wire [DATA_WIDTH-1:0] exp_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    exp_sram (.clk(clk),.rst(rst),.w_en(exp_w_en),.r_addr(exp_read_addr),
              .w_addr(exp_write_addr),.sram_data_in(exp_write_data),.sram_data_out(exp_sram_out));
    reg [2*DATA_WIDTH-1:0] exp_sum;

    always @(posedge clk) begin
        if (rst) begin exp_w_en <= 0; exp_write_addr <= 0; exp_sum <= 0; end
        else if (sm_exp_0_valid_n) begin
            exp_write_data <= sm_exp_0_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_0_data_out;
        end else exp_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] exp_sum_upper = exp_sum[2*DATA_WIDTH-1:DATA_WIDTH];
    reg [DATA_WIDTH-1:0] recip_val;
    always @(*) begin
        casez (exp_sum_upper[15:0])
            16'b1???????????????: recip_val = 40'd2;
            16'b01??????????????: recip_val = 40'd32768;
            16'b001?????????????: recip_val = 40'd16384;
            16'b0001????????????: recip_val = 40'd8192;
            16'b00001???????????: recip_val = 40'd4096;
            16'b000001??????????: recip_val = 40'd2048;
            16'b0000001?????????: recip_val = 40'd1024;
            16'b00000001????????: recip_val = 40'd512;
            16'b000000001???????: recip_val = 40'd256;
            16'b0000000001??????: recip_val = 40'd128;
            16'b00000000001?????: recip_val = 40'd64;
            16'b000000000001????: recip_val = 40'd32;
            16'b0000000000001???: recip_val = 40'd16;
            16'b00000000000001??: recip_val = 40'd8;
            16'b000000000000001?: recip_val = 40'd4;
            16'b0000000000000001: recip_val = 40'd2;
            default: recip_val = 40'hFFFF;
        endcase
    end

    reg [DATA_WIDTH-1:0] inv_sum;
    reg [2*DATA_WIDTH-1:0] norm_product;
    reg [DATA_WIDTH-1:0] norm_val;

    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    wire [DATA_WIDTH-1:0] out_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),
              .w_addr(out_write_addr),.sram_data_in(norm_val),.sram_data_out(out_sram_out));

    localparam SM_IDLE = 3'd0, SM_LOAD = 3'd1, SM_EXP = 3'd2,
               SM_NORMALIZE = 3'd3, SM_OUTPUT = 3'd4;
    reg [2:0] sm_state;
    reg [$clog2(N)-1:0] sm_count;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sm_state <= SM_IDLE; sm_count <= 0;
            in_write_addr <= 0; in_read_addr <= 0; in_w_en <= 0;
            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;
            inv_sum <= 0; norm_product <= 0; norm_val <= 0;
        end else begin
            in_w_en <= 0; out_w_en <= 0;
            case (sm_state)
                SM_IDLE: if (valid) sm_state <= SM_LOAD;
                SM_LOAD: begin
                    if (valid) begin in_w_en <= 1; in_write_addr <= in_write_addr + 1; end
                    if (in_write_addr == N-1) sm_state <= SM_EXP;
                end
                SM_EXP: begin
                    // DPE(I|exp) processes scores — output goes to exp_sram + sum
                    in_read_addr <= sm_count; sm_count <= sm_count + 1;
                    if (sm_count == N-1) begin
                        inv_sum <= recip_val; sm_count <= 0;
                        sm_state <= SM_NORMALIZE;
                    end
                end
                SM_NORMALIZE: begin
                    // CLB multiply: exp_val × inv_sum
                    exp_read_addr <= sm_count;
                    norm_product <= exp_sram_out * inv_sum;
                    norm_val <= norm_product[2*DATA_WIDTH-1:DATA_WIDTH];
                    out_w_en <= (sm_count > 1);
                    out_write_addr <= sm_count - 2;
                    sm_count <= sm_count + 1;
                    if (sm_count == N+1) sm_state <= SM_OUTPUT;
                end
                SM_OUTPUT: if (!ready_n) out_read_addr <= out_read_addr + 1;
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (sm_state == SM_OUTPUT);
    assign ready = (sm_state == SM_LOAD || sm_state == SM_IDLE);
endmodule
module dimm_weighted_sum_b1_h1 #(
    parameter N = 6144,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 19,
    parameter DEPTH = 393216
)(
    input wire clk, input wire rst,
    input wire valid_attn, input wire valid_v,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_attn,
    input wire [DATA_WIDTH-1:0] data_in_v,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_attn, output wire ready_v,
    output wire valid_n
);

    reg [ADDR_WIDTH-1:0] attn_write_addr, attn_read_addr;
    reg attn_w_en;
    wire [DATA_WIDTH-1:0] attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    attn_sram (.clk(clk),.rst(rst),.w_en(attn_w_en),.r_addr(attn_read_addr),
               .w_addr(attn_write_addr),.sram_data_in(data_in_attn),.sram_data_out(attn_sram_out));

    reg [$clog2(N*d)-1:0] v_write_addr, v_read_addr;
    reg v_w_en;
    wire [DATA_WIDTH-1:0] v_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N*d))
    v_sram (.clk(clk),.rst(rst),.w_en(v_w_en),.r_addr(v_read_addr[$clog2(N*d)-1:0]),
            .w_addr(v_write_addr[$clog2(N*d)-1:0]),.sram_data_in(data_in_v),.sram_data_out(v_sram_out));

    wire ws_log_0_valid, ws_log_0_ready_n, ws_log_0_ready, ws_log_0_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_0_data_in, ws_log_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_0 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_0_valid),
        .ready_n(ws_log_0_ready_n),
        .data_in(ws_log_0_data_in),
        .data_out(ws_log_0_data_out),
        .ready(ws_log_0_ready),
        .valid_n(ws_log_0_valid_n)
    );

    wire ws_log_1_valid, ws_log_1_ready_n, ws_log_1_ready, ws_log_1_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_1_data_in, ws_log_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_1 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_1_valid),
        .ready_n(ws_log_1_ready_n),
        .data_in(ws_log_1_data_in),
        .data_out(ws_log_1_data_out),
        .ready(ws_log_1_ready),
        .valid_n(ws_log_1_valid_n)
    );

    wire ws_log_2_valid, ws_log_2_ready_n, ws_log_2_ready, ws_log_2_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_2_data_in, ws_log_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_2 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_2_valid),
        .ready_n(ws_log_2_ready_n),
        .data_in(ws_log_2_data_in),
        .data_out(ws_log_2_data_out),
        .ready(ws_log_2_ready),
        .valid_n(ws_log_2_valid_n)
    );

    wire ws_log_3_valid, ws_log_3_ready_n, ws_log_3_ready, ws_log_3_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_3_data_in, ws_log_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_3 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_3_valid),
        .ready_n(ws_log_3_ready_n),
        .data_in(ws_log_3_data_in),
        .data_out(ws_log_3_data_out),
        .ready(ws_log_3_ready),
        .valid_n(ws_log_3_valid_n)
    );

    wire ws_log_4_valid, ws_log_4_ready_n, ws_log_4_ready, ws_log_4_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_4_data_in, ws_log_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_4 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_4_valid),
        .ready_n(ws_log_4_ready_n),
        .data_in(ws_log_4_data_in),
        .data_out(ws_log_4_data_out),
        .ready(ws_log_4_ready),
        .valid_n(ws_log_4_valid_n)
    );

    wire ws_log_5_valid, ws_log_5_ready_n, ws_log_5_ready, ws_log_5_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_5_data_in, ws_log_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_5 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_5_valid),
        .ready_n(ws_log_5_ready_n),
        .data_in(ws_log_5_data_in),
        .data_out(ws_log_5_data_out),
        .ready(ws_log_5_ready),
        .valid_n(ws_log_5_valid_n)
    );

    wire ws_log_6_valid, ws_log_6_ready_n, ws_log_6_ready, ws_log_6_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_6_data_in, ws_log_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_6 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_6_valid),
        .ready_n(ws_log_6_ready_n),
        .data_in(ws_log_6_data_in),
        .data_out(ws_log_6_data_out),
        .ready(ws_log_6_ready),
        .valid_n(ws_log_6_valid_n)
    );

    wire ws_log_7_valid, ws_log_7_ready_n, ws_log_7_ready, ws_log_7_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_7_data_in, ws_log_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_7 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_7_valid),
        .ready_n(ws_log_7_ready_n),
        .data_in(ws_log_7_data_in),
        .data_out(ws_log_7_data_out),
        .ready(ws_log_7_ready),
        .valid_n(ws_log_7_valid_n)
    );

    wire ws_log_8_valid, ws_log_8_ready_n, ws_log_8_ready, ws_log_8_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_8_data_in, ws_log_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_8 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_8_valid),
        .ready_n(ws_log_8_ready_n),
        .data_in(ws_log_8_data_in),
        .data_out(ws_log_8_data_out),
        .ready(ws_log_8_ready),
        .valid_n(ws_log_8_valid_n)
    );

    wire ws_log_9_valid, ws_log_9_ready_n, ws_log_9_ready, ws_log_9_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_9_data_in, ws_log_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_9 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_9_valid),
        .ready_n(ws_log_9_ready_n),
        .data_in(ws_log_9_data_in),
        .data_out(ws_log_9_data_out),
        .ready(ws_log_9_ready),
        .valid_n(ws_log_9_valid_n)
    );

    wire ws_log_10_valid, ws_log_10_ready_n, ws_log_10_ready, ws_log_10_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_10_data_in, ws_log_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_10 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_10_valid),
        .ready_n(ws_log_10_ready_n),
        .data_in(ws_log_10_data_in),
        .data_out(ws_log_10_data_out),
        .ready(ws_log_10_ready),
        .valid_n(ws_log_10_valid_n)
    );

    wire ws_log_11_valid, ws_log_11_ready_n, ws_log_11_ready, ws_log_11_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_11_data_in, ws_log_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_11 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_11_valid),
        .ready_n(ws_log_11_ready_n),
        .data_in(ws_log_11_data_in),
        .data_out(ws_log_11_data_out),
        .ready(ws_log_11_ready),
        .valid_n(ws_log_11_valid_n)
    );

    wire ws_log_12_valid, ws_log_12_ready_n, ws_log_12_ready, ws_log_12_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_12_data_in, ws_log_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_12 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_12_valid),
        .ready_n(ws_log_12_ready_n),
        .data_in(ws_log_12_data_in),
        .data_out(ws_log_12_data_out),
        .ready(ws_log_12_ready),
        .valid_n(ws_log_12_valid_n)
    );

    wire ws_log_13_valid, ws_log_13_ready_n, ws_log_13_ready, ws_log_13_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_13_data_in, ws_log_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_13 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_13_valid),
        .ready_n(ws_log_13_ready_n),
        .data_in(ws_log_13_data_in),
        .data_out(ws_log_13_data_out),
        .ready(ws_log_13_ready),
        .valid_n(ws_log_13_valid_n)
    );

    wire ws_log_14_valid, ws_log_14_ready_n, ws_log_14_ready, ws_log_14_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_14_data_in, ws_log_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_14 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_14_valid),
        .ready_n(ws_log_14_ready_n),
        .data_in(ws_log_14_data_in),
        .data_out(ws_log_14_data_out),
        .ready(ws_log_14_ready),
        .valid_n(ws_log_14_valid_n)
    );

    wire ws_log_15_valid, ws_log_15_ready_n, ws_log_15_ready, ws_log_15_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_15_data_in, ws_log_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_15 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_15_valid),
        .ready_n(ws_log_15_ready_n),
        .data_in(ws_log_15_data_in),
        .data_out(ws_log_15_data_out),
        .ready(ws_log_15_ready),
        .valid_n(ws_log_15_valid_n)
    );

    wire ws_log_16_valid, ws_log_16_ready_n, ws_log_16_ready, ws_log_16_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_16_data_in, ws_log_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_16 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_16_valid),
        .ready_n(ws_log_16_ready_n),
        .data_in(ws_log_16_data_in),
        .data_out(ws_log_16_data_out),
        .ready(ws_log_16_ready),
        .valid_n(ws_log_16_valid_n)
    );

    wire ws_log_17_valid, ws_log_17_ready_n, ws_log_17_ready, ws_log_17_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_17_data_in, ws_log_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_17 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_17_valid),
        .ready_n(ws_log_17_ready_n),
        .data_in(ws_log_17_data_in),
        .data_out(ws_log_17_data_out),
        .ready(ws_log_17_ready),
        .valid_n(ws_log_17_valid_n)
    );

    wire ws_log_18_valid, ws_log_18_ready_n, ws_log_18_ready, ws_log_18_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_18_data_in, ws_log_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_18 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_18_valid),
        .ready_n(ws_log_18_ready_n),
        .data_in(ws_log_18_data_in),
        .data_out(ws_log_18_data_out),
        .ready(ws_log_18_ready),
        .valid_n(ws_log_18_valid_n)
    );

    wire ws_log_19_valid, ws_log_19_ready_n, ws_log_19_ready, ws_log_19_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_19_data_in, ws_log_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_19 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_19_valid),
        .ready_n(ws_log_19_ready_n),
        .data_in(ws_log_19_data_in),
        .data_out(ws_log_19_data_out),
        .ready(ws_log_19_ready),
        .valid_n(ws_log_19_valid_n)
    );

    wire ws_log_20_valid, ws_log_20_ready_n, ws_log_20_ready, ws_log_20_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_20_data_in, ws_log_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_20 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_20_valid),
        .ready_n(ws_log_20_ready_n),
        .data_in(ws_log_20_data_in),
        .data_out(ws_log_20_data_out),
        .ready(ws_log_20_ready),
        .valid_n(ws_log_20_valid_n)
    );

    wire ws_log_21_valid, ws_log_21_ready_n, ws_log_21_ready, ws_log_21_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_21_data_in, ws_log_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_21 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_21_valid),
        .ready_n(ws_log_21_ready_n),
        .data_in(ws_log_21_data_in),
        .data_out(ws_log_21_data_out),
        .ready(ws_log_21_ready),
        .valid_n(ws_log_21_valid_n)
    );

    wire ws_log_22_valid, ws_log_22_ready_n, ws_log_22_ready, ws_log_22_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_22_data_in, ws_log_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_22 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_22_valid),
        .ready_n(ws_log_22_ready_n),
        .data_in(ws_log_22_data_in),
        .data_out(ws_log_22_data_out),
        .ready(ws_log_22_ready),
        .valid_n(ws_log_22_valid_n)
    );

    wire ws_log_23_valid, ws_log_23_ready_n, ws_log_23_ready, ws_log_23_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_23_data_in, ws_log_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(256),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_log_23 (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_23_valid),
        .ready_n(ws_log_23_ready_n),
        .data_in(ws_log_23_data_in),
        .data_out(ws_log_23_data_out),
        .ready(ws_log_23_ready),
        .valid_n(ws_log_23_valid_n)
    );

    assign ws_log_0_data_in = attn_sram_out;
    assign ws_log_0_valid = (ws_state == WS_LOG);
    assign ws_log_0_ready_n = 1'b0;
    assign ws_log_1_data_in = attn_sram_out;
    assign ws_log_1_valid = (ws_state == WS_LOG);
    assign ws_log_1_ready_n = 1'b0;
    assign ws_log_2_data_in = attn_sram_out;
    assign ws_log_2_valid = (ws_state == WS_LOG);
    assign ws_log_2_ready_n = 1'b0;
    assign ws_log_3_data_in = attn_sram_out;
    assign ws_log_3_valid = (ws_state == WS_LOG);
    assign ws_log_3_ready_n = 1'b0;
    assign ws_log_4_data_in = attn_sram_out;
    assign ws_log_4_valid = (ws_state == WS_LOG);
    assign ws_log_4_ready_n = 1'b0;
    assign ws_log_5_data_in = attn_sram_out;
    assign ws_log_5_valid = (ws_state == WS_LOG);
    assign ws_log_5_ready_n = 1'b0;
    assign ws_log_6_data_in = attn_sram_out;
    assign ws_log_6_valid = (ws_state == WS_LOG);
    assign ws_log_6_ready_n = 1'b0;
    assign ws_log_7_data_in = attn_sram_out;
    assign ws_log_7_valid = (ws_state == WS_LOG);
    assign ws_log_7_ready_n = 1'b0;
    assign ws_log_8_data_in = attn_sram_out;
    assign ws_log_8_valid = (ws_state == WS_LOG);
    assign ws_log_8_ready_n = 1'b0;
    assign ws_log_9_data_in = attn_sram_out;
    assign ws_log_9_valid = (ws_state == WS_LOG);
    assign ws_log_9_ready_n = 1'b0;
    assign ws_log_10_data_in = attn_sram_out;
    assign ws_log_10_valid = (ws_state == WS_LOG);
    assign ws_log_10_ready_n = 1'b0;
    assign ws_log_11_data_in = attn_sram_out;
    assign ws_log_11_valid = (ws_state == WS_LOG);
    assign ws_log_11_ready_n = 1'b0;
    assign ws_log_12_data_in = attn_sram_out;
    assign ws_log_12_valid = (ws_state == WS_LOG);
    assign ws_log_12_ready_n = 1'b0;
    assign ws_log_13_data_in = attn_sram_out;
    assign ws_log_13_valid = (ws_state == WS_LOG);
    assign ws_log_13_ready_n = 1'b0;
    assign ws_log_14_data_in = attn_sram_out;
    assign ws_log_14_valid = (ws_state == WS_LOG);
    assign ws_log_14_ready_n = 1'b0;
    assign ws_log_15_data_in = attn_sram_out;
    assign ws_log_15_valid = (ws_state == WS_LOG);
    assign ws_log_15_ready_n = 1'b0;
    assign ws_log_16_data_in = attn_sram_out;
    assign ws_log_16_valid = (ws_state == WS_LOG);
    assign ws_log_16_ready_n = 1'b0;
    assign ws_log_17_data_in = attn_sram_out;
    assign ws_log_17_valid = (ws_state == WS_LOG);
    assign ws_log_17_ready_n = 1'b0;
    assign ws_log_18_data_in = attn_sram_out;
    assign ws_log_18_valid = (ws_state == WS_LOG);
    assign ws_log_18_ready_n = 1'b0;
    assign ws_log_19_data_in = attn_sram_out;
    assign ws_log_19_valid = (ws_state == WS_LOG);
    assign ws_log_19_ready_n = 1'b0;
    assign ws_log_20_data_in = attn_sram_out;
    assign ws_log_20_valid = (ws_state == WS_LOG);
    assign ws_log_20_ready_n = 1'b0;
    assign ws_log_21_data_in = attn_sram_out;
    assign ws_log_21_valid = (ws_state == WS_LOG);
    assign ws_log_21_ready_n = 1'b0;
    assign ws_log_22_data_in = attn_sram_out;
    assign ws_log_22_valid = (ws_state == WS_LOG);
    assign ws_log_22_ready_n = 1'b0;
    assign ws_log_23_data_in = attn_sram_out;
    assign ws_log_23_valid = (ws_state == WS_LOG);
    assign ws_log_23_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_0_valid_n) begin
            log_attn_write_data <= ws_log_0_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_0_valid, ws_exp_0_ready_n, ws_exp_0_ready, ws_exp_0_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_0_data_in, ws_exp_0_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_0 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_0_valid),
        .ready_n(ws_exp_0_ready_n),
        .data_in(ws_exp_0_data_in),
        .data_out(ws_exp_0_data_out),
        .ready(ws_exp_0_ready),
        .valid_n(ws_exp_0_valid_n)
    );

    wire ws_exp_1_valid, ws_exp_1_ready_n, ws_exp_1_ready, ws_exp_1_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_1_data_in, ws_exp_1_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_1 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_1_valid),
        .ready_n(ws_exp_1_ready_n),
        .data_in(ws_exp_1_data_in),
        .data_out(ws_exp_1_data_out),
        .ready(ws_exp_1_ready),
        .valid_n(ws_exp_1_valid_n)
    );

    wire ws_exp_2_valid, ws_exp_2_ready_n, ws_exp_2_ready, ws_exp_2_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_2_data_in, ws_exp_2_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_2 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_2_valid),
        .ready_n(ws_exp_2_ready_n),
        .data_in(ws_exp_2_data_in),
        .data_out(ws_exp_2_data_out),
        .ready(ws_exp_2_ready),
        .valid_n(ws_exp_2_valid_n)
    );

    wire ws_exp_3_valid, ws_exp_3_ready_n, ws_exp_3_ready, ws_exp_3_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_3_data_in, ws_exp_3_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_3 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_3_valid),
        .ready_n(ws_exp_3_ready_n),
        .data_in(ws_exp_3_data_in),
        .data_out(ws_exp_3_data_out),
        .ready(ws_exp_3_ready),
        .valid_n(ws_exp_3_valid_n)
    );

    wire ws_exp_4_valid, ws_exp_4_ready_n, ws_exp_4_ready, ws_exp_4_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_4_data_in, ws_exp_4_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_4 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_4_valid),
        .ready_n(ws_exp_4_ready_n),
        .data_in(ws_exp_4_data_in),
        .data_out(ws_exp_4_data_out),
        .ready(ws_exp_4_ready),
        .valid_n(ws_exp_4_valid_n)
    );

    wire ws_exp_5_valid, ws_exp_5_ready_n, ws_exp_5_ready, ws_exp_5_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_5_data_in, ws_exp_5_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_5 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_5_valid),
        .ready_n(ws_exp_5_ready_n),
        .data_in(ws_exp_5_data_in),
        .data_out(ws_exp_5_data_out),
        .ready(ws_exp_5_ready),
        .valid_n(ws_exp_5_valid_n)
    );

    wire ws_exp_6_valid, ws_exp_6_ready_n, ws_exp_6_ready, ws_exp_6_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_6_data_in, ws_exp_6_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_6 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_6_valid),
        .ready_n(ws_exp_6_ready_n),
        .data_in(ws_exp_6_data_in),
        .data_out(ws_exp_6_data_out),
        .ready(ws_exp_6_ready),
        .valid_n(ws_exp_6_valid_n)
    );

    wire ws_exp_7_valid, ws_exp_7_ready_n, ws_exp_7_ready, ws_exp_7_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_7_data_in, ws_exp_7_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_7 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_7_valid),
        .ready_n(ws_exp_7_ready_n),
        .data_in(ws_exp_7_data_in),
        .data_out(ws_exp_7_data_out),
        .ready(ws_exp_7_ready),
        .valid_n(ws_exp_7_valid_n)
    );

    wire ws_exp_8_valid, ws_exp_8_ready_n, ws_exp_8_ready, ws_exp_8_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_8_data_in, ws_exp_8_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_8 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_8_valid),
        .ready_n(ws_exp_8_ready_n),
        .data_in(ws_exp_8_data_in),
        .data_out(ws_exp_8_data_out),
        .ready(ws_exp_8_ready),
        .valid_n(ws_exp_8_valid_n)
    );

    wire ws_exp_9_valid, ws_exp_9_ready_n, ws_exp_9_ready, ws_exp_9_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_9_data_in, ws_exp_9_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_9 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_9_valid),
        .ready_n(ws_exp_9_ready_n),
        .data_in(ws_exp_9_data_in),
        .data_out(ws_exp_9_data_out),
        .ready(ws_exp_9_ready),
        .valid_n(ws_exp_9_valid_n)
    );

    wire ws_exp_10_valid, ws_exp_10_ready_n, ws_exp_10_ready, ws_exp_10_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_10_data_in, ws_exp_10_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_10 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_10_valid),
        .ready_n(ws_exp_10_ready_n),
        .data_in(ws_exp_10_data_in),
        .data_out(ws_exp_10_data_out),
        .ready(ws_exp_10_ready),
        .valid_n(ws_exp_10_valid_n)
    );

    wire ws_exp_11_valid, ws_exp_11_ready_n, ws_exp_11_ready, ws_exp_11_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_11_data_in, ws_exp_11_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_11 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_11_valid),
        .ready_n(ws_exp_11_ready_n),
        .data_in(ws_exp_11_data_in),
        .data_out(ws_exp_11_data_out),
        .ready(ws_exp_11_ready),
        .valid_n(ws_exp_11_valid_n)
    );

    wire ws_exp_12_valid, ws_exp_12_ready_n, ws_exp_12_ready, ws_exp_12_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_12_data_in, ws_exp_12_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_12 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_12_valid),
        .ready_n(ws_exp_12_ready_n),
        .data_in(ws_exp_12_data_in),
        .data_out(ws_exp_12_data_out),
        .ready(ws_exp_12_ready),
        .valid_n(ws_exp_12_valid_n)
    );

    wire ws_exp_13_valid, ws_exp_13_ready_n, ws_exp_13_ready, ws_exp_13_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_13_data_in, ws_exp_13_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_13 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_13_valid),
        .ready_n(ws_exp_13_ready_n),
        .data_in(ws_exp_13_data_in),
        .data_out(ws_exp_13_data_out),
        .ready(ws_exp_13_ready),
        .valid_n(ws_exp_13_valid_n)
    );

    wire ws_exp_14_valid, ws_exp_14_ready_n, ws_exp_14_ready, ws_exp_14_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_14_data_in, ws_exp_14_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_14 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_14_valid),
        .ready_n(ws_exp_14_ready_n),
        .data_in(ws_exp_14_data_in),
        .data_out(ws_exp_14_data_out),
        .ready(ws_exp_14_ready),
        .valid_n(ws_exp_14_valid_n)
    );

    wire ws_exp_15_valid, ws_exp_15_ready_n, ws_exp_15_ready, ws_exp_15_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_15_data_in, ws_exp_15_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_15 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_15_valid),
        .ready_n(ws_exp_15_ready_n),
        .data_in(ws_exp_15_data_in),
        .data_out(ws_exp_15_data_out),
        .ready(ws_exp_15_ready),
        .valid_n(ws_exp_15_valid_n)
    );

    wire ws_exp_16_valid, ws_exp_16_ready_n, ws_exp_16_ready, ws_exp_16_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_16_data_in, ws_exp_16_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_16 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_16_valid),
        .ready_n(ws_exp_16_ready_n),
        .data_in(ws_exp_16_data_in),
        .data_out(ws_exp_16_data_out),
        .ready(ws_exp_16_ready),
        .valid_n(ws_exp_16_valid_n)
    );

    wire ws_exp_17_valid, ws_exp_17_ready_n, ws_exp_17_ready, ws_exp_17_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_17_data_in, ws_exp_17_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_17 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_17_valid),
        .ready_n(ws_exp_17_ready_n),
        .data_in(ws_exp_17_data_in),
        .data_out(ws_exp_17_data_out),
        .ready(ws_exp_17_ready),
        .valid_n(ws_exp_17_valid_n)
    );

    wire ws_exp_18_valid, ws_exp_18_ready_n, ws_exp_18_ready, ws_exp_18_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_18_data_in, ws_exp_18_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_18 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_18_valid),
        .ready_n(ws_exp_18_ready_n),
        .data_in(ws_exp_18_data_in),
        .data_out(ws_exp_18_data_out),
        .ready(ws_exp_18_ready),
        .valid_n(ws_exp_18_valid_n)
    );

    wire ws_exp_19_valid, ws_exp_19_ready_n, ws_exp_19_ready, ws_exp_19_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_19_data_in, ws_exp_19_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_19 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_19_valid),
        .ready_n(ws_exp_19_ready_n),
        .data_in(ws_exp_19_data_in),
        .data_out(ws_exp_19_data_out),
        .ready(ws_exp_19_ready),
        .valid_n(ws_exp_19_valid_n)
    );

    wire ws_exp_20_valid, ws_exp_20_ready_n, ws_exp_20_ready, ws_exp_20_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_20_data_in, ws_exp_20_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_20 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_20_valid),
        .ready_n(ws_exp_20_ready_n),
        .data_in(ws_exp_20_data_in),
        .data_out(ws_exp_20_data_out),
        .ready(ws_exp_20_ready),
        .valid_n(ws_exp_20_valid_n)
    );

    wire ws_exp_21_valid, ws_exp_21_ready_n, ws_exp_21_ready, ws_exp_21_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_21_data_in, ws_exp_21_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_21 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_21_valid),
        .ready_n(ws_exp_21_ready_n),
        .data_in(ws_exp_21_data_in),
        .data_out(ws_exp_21_data_out),
        .ready(ws_exp_21_ready),
        .valid_n(ws_exp_21_valid_n)
    );

    wire ws_exp_22_valid, ws_exp_22_ready_n, ws_exp_22_ready, ws_exp_22_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_22_data_in, ws_exp_22_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_22 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_22_valid),
        .ready_n(ws_exp_22_ready_n),
        .data_in(ws_exp_22_data_in),
        .data_out(ws_exp_22_data_out),
        .ready(ws_exp_22_ready),
        .valid_n(ws_exp_22_valid_n)
    );

    wire ws_exp_23_valid, ws_exp_23_ready_n, ws_exp_23_ready, ws_exp_23_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_23_data_in, ws_exp_23_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(19),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(393216),
        .DATA_WIDTH(40)
    ) ws_exp_23 (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_23_valid),
        .ready_n(ws_exp_23_ready_n),
        .data_in(ws_exp_23_data_in),
        .data_out(ws_exp_23_data_out),
        .ready(ws_exp_23_ready),
        .valid_n(ws_exp_23_valid_n)
    );

    assign ws_exp_0_data_in = ws_log_sum;
    assign ws_exp_0_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_0_ready_n = 1'b0;
    assign ws_exp_1_data_in = ws_log_sum;
    assign ws_exp_1_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_1_ready_n = 1'b0;
    assign ws_exp_2_data_in = ws_log_sum;
    assign ws_exp_2_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_2_ready_n = 1'b0;
    assign ws_exp_3_data_in = ws_log_sum;
    assign ws_exp_3_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_3_ready_n = 1'b0;
    assign ws_exp_4_data_in = ws_log_sum;
    assign ws_exp_4_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_4_ready_n = 1'b0;
    assign ws_exp_5_data_in = ws_log_sum;
    assign ws_exp_5_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_5_ready_n = 1'b0;
    assign ws_exp_6_data_in = ws_log_sum;
    assign ws_exp_6_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_6_ready_n = 1'b0;
    assign ws_exp_7_data_in = ws_log_sum;
    assign ws_exp_7_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_7_ready_n = 1'b0;
    assign ws_exp_8_data_in = ws_log_sum;
    assign ws_exp_8_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_8_ready_n = 1'b0;
    assign ws_exp_9_data_in = ws_log_sum;
    assign ws_exp_9_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_9_ready_n = 1'b0;
    assign ws_exp_10_data_in = ws_log_sum;
    assign ws_exp_10_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_10_ready_n = 1'b0;
    assign ws_exp_11_data_in = ws_log_sum;
    assign ws_exp_11_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_11_ready_n = 1'b0;
    assign ws_exp_12_data_in = ws_log_sum;
    assign ws_exp_12_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_12_ready_n = 1'b0;
    assign ws_exp_13_data_in = ws_log_sum;
    assign ws_exp_13_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_13_ready_n = 1'b0;
    assign ws_exp_14_data_in = ws_log_sum;
    assign ws_exp_14_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_14_ready_n = 1'b0;
    assign ws_exp_15_data_in = ws_log_sum;
    assign ws_exp_15_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_15_ready_n = 1'b0;
    assign ws_exp_16_data_in = ws_log_sum;
    assign ws_exp_16_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_16_ready_n = 1'b0;
    assign ws_exp_17_data_in = ws_log_sum;
    assign ws_exp_17_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_17_ready_n = 1'b0;
    assign ws_exp_18_data_in = ws_log_sum;
    assign ws_exp_18_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_18_ready_n = 1'b0;
    assign ws_exp_19_data_in = ws_log_sum;
    assign ws_exp_19_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_19_ready_n = 1'b0;
    assign ws_exp_20_data_in = ws_log_sum;
    assign ws_exp_20_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_20_ready_n = 1'b0;
    assign ws_exp_21_data_in = ws_log_sum;
    assign ws_exp_21_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_21_ready_n = 1'b0;
    assign ws_exp_22_data_in = ws_log_sum;
    assign ws_exp_22_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_22_ready_n = 1'b0;
    assign ws_exp_23_data_in = ws_log_sum;
    assign ws_exp_23_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_23_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_0_valid_n) ws_accumulator <= ws_accumulator + ws_exp_0_data_out;
    end

    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    reg [DATA_WIDTH-1:0] out_write_data;
    wire [DATA_WIDTH-1:0] out_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    out_sram (.clk(clk),.rst(rst),.w_en(out_w_en),.r_addr(out_read_addr),
              .w_addr(out_write_addr),.sram_data_in(out_write_data),.sram_data_out(out_sram_out));

    localparam WS_IDLE = 3'd0, WS_LOAD_A = 3'd1, WS_LOAD_V = 3'd2,
               WS_LOG = 3'd3, WS_COMPUTE = 3'd4, WS_OUTPUT = 3'd5;
    reg [2:0] ws_state;
    reg [$clog2(N)-1:0] ws_j;
    reg [$clog2(d)-1:0] ws_m;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ws_state <= WS_IDLE; ws_j <= 0; ws_m <= 0;
            attn_write_addr <= 0; attn_read_addr <= 0; attn_w_en <= 0;
            v_write_addr <= 0; v_read_addr <= 0; v_w_en <= 0;
            out_write_addr <= 0; out_read_addr <= 0; out_w_en <= 0;
            out_write_data <= 0;
        end else begin
            attn_w_en <= 0; v_w_en <= 0; out_w_en <= 0;
            case (ws_state)
                WS_IDLE: if (valid_attn || valid_v) ws_state <= WS_LOAD_A;
                WS_LOAD_A: begin
                    if (valid_attn) begin attn_w_en <= 1; attn_write_addr <= attn_write_addr + 1; end
                    if (attn_write_addr == N-1) ws_state <= WS_LOAD_V;
                end
                WS_LOAD_V: begin
                    if (valid_v) begin v_w_en <= 1; v_write_addr <= v_write_addr + 1; end
                    if (v_write_addr == N*d-1) ws_state <= WS_LOG;
                end
                WS_LOG: begin
                    // DPE(I|log) converts attn to log domain
                    attn_read_addr <= ws_j;
                    ws_j <= ws_j + 1;
                    if (ws_j == N-1) begin ws_j <= 0; ws_state <= WS_COMPUTE; end
                end
                WS_COMPUTE: begin
                    // CLB add + DPE(I|exp) + accumulate
                    log_attn_read_addr <= ws_j;
                    v_read_addr <= (ws_j << $clog2(d)) + ws_m;
                    ws_j <= ws_j + 1;
                    if (ws_j == N-1) begin
                        out_write_data <= ws_accumulator[2*DATA_WIDTH-1:DATA_WIDTH];
                        out_w_en <= 1; out_write_addr <= ws_m;
                        ws_j <= 0; ws_m <= ws_m + 1;
                        if (ws_m == d-1) ws_state <= WS_OUTPUT;
                    end
                end
                WS_OUTPUT: if (ready_n) out_read_addr <= out_read_addr + 1;
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (ws_state == WS_OUTPUT);
    assign ready_attn = (ws_state == WS_LOAD_A || ws_state == WS_IDLE);
    assign ready_v = (ws_state == WS_LOAD_V || ws_state == WS_IDLE);
endmodule

// ═══════════════════════════════════════
// CLB Modules (DPE-independent)
// ═══════════════════════════════════════
// Auto-generated residual_add module
// Element-wise add of two tensors (DEPTH=128 elements)
// Pure CLB: 1 SRAM + 1 adder + 2-state FSM
// Generated by gen_bert_clb_modules.py

module residual_add #(
    parameter DATA_WIDTH = 40,
    parameter DEPTH      = 128
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    valid,
    input  wire                    ready_n,
    input  wire [DATA_WIDTH-1:0]  data_in,
    output reg  [DATA_WIDTH-1:0]  data_out,
    output wire                    ready,
    output wire                    valid_n
);

    localparam ADDR_WIDTH = 7;

    // FSM
    localparam S_IDLE = 2'd0;
    localparam S_LOAD = 2'd1;
    localparam S_ADD  = 2'd2;

    reg [1:0] state, next_state;

    // SRAM for first operand
    reg                      sram_w_en;
    reg  [ADDR_WIDTH-1:0]    sram_w_addr;
    reg  [ADDR_WIDTH-1:0]    sram_r_addr;
    wire [DATA_WIDTH-1:0]    sram_out;

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) buf_a (
        .clk(clk),
        .rst(rst),
        .w_en(sram_w_en),
        .r_addr(sram_r_addr),
        .w_addr(sram_w_addr),
        .sram_data_in(data_in),
        .sram_data_out(sram_out)
    );

    // Registered add result (1-cycle latency from SRAM read)
    reg                      add_valid;
    reg  [ADDR_WIDTH-1:0]    add_addr;

    // FSM transitions
    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    always @* begin
        next_state = state;
        case (state)
            S_IDLE: if (valid)                        next_state = S_LOAD;
            S_LOAD: if (sram_w_addr == DEPTH - 1)     next_state = S_ADD;
            S_ADD:  if (add_addr == DEPTH - 1 && add_valid) next_state = S_IDLE;
            default:                                   next_state = S_IDLE;
        endcase
    end

    // Datapath
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sram_w_en   <= 1'b0;
            sram_w_addr <= {ADDR_WIDTH{1'b0}};
            sram_r_addr <= {ADDR_WIDTH{1'b0}};
            add_valid   <= 1'b0;
            add_addr    <= {ADDR_WIDTH{1'b0}};
            data_out    <= {DATA_WIDTH{1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    sram_w_en   <= 1'b0;
                    sram_w_addr <= {ADDR_WIDTH{1'b0}};
                    sram_r_addr <= {ADDR_WIDTH{1'b0}};
                    add_valid   <= 1'b0;
                    add_addr    <= {ADDR_WIDTH{1'b0}};
                    if (valid) begin
                        sram_w_en   <= 1'b1;
                        sram_w_addr <= {ADDR_WIDTH{1'b0}};
                    end
                end

                S_LOAD: begin
                    sram_w_en <= valid;
                    if (valid) begin
                        sram_w_addr <= sram_w_addr + 1'b1;
                    end
                    if (sram_w_addr == DEPTH - 1) begin
                        // Transition: start reading from addr 0
                        sram_w_en   <= 1'b0;
                        sram_r_addr <= {ADDR_WIDTH{1'b0}};
                    end
                end

                S_ADD: begin
                    sram_w_en <= 1'b0;
                    // Pipeline: SRAM read (cycle N) -> add + output (cycle N+1)
                    if (valid) begin
                        sram_r_addr <= sram_r_addr + 1'b1;
                    end
                    // Registered add: sram_out is available 1 cycle after r_addr
                    add_valid <= valid;
                    if (add_valid)
                        add_addr <= add_addr + 1'b1;
                    data_out <= $signed(sram_out) + $signed(data_in);
                end

                default: begin
                    sram_w_en <= 1'b0;
                    add_valid <= 1'b0;
                end
            endcase
        end
    end

    assign ready   = (state == S_LOAD && valid) || (state == S_ADD && valid) || (state == S_IDLE);
    assign valid_n = ~add_valid;

endmodule




// Auto-generated embedding_add module
// 3-way element-wise add: token + position + segment (DEPTH=128 elements)
// Pure CLB: 2 SRAMs + 2 adders + 3-state FSM
// Generated by gen_bert_clb_modules.py

module embedding_add #(
    parameter DATA_WIDTH = 40,
    parameter DEPTH      = 128
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    valid,
    input  wire                    ready_n,
    input  wire [DATA_WIDTH-1:0]  data_in,
    output reg  [DATA_WIDTH-1:0]  data_out,
    output wire                    ready,
    output wire                    valid_n
);

    localparam ADDR_WIDTH = 7;

    // FSM
    localparam S_IDLE     = 2'd0;
    localparam S_LOAD_TOK = 2'd1;
    localparam S_LOAD_POS = 2'd2;
    localparam S_ADD      = 2'd3;

    reg [1:0] state, next_state;

    // Token SRAM
    reg                      tok_w_en;
    reg  [ADDR_WIDTH-1:0]    tok_w_addr, tok_r_addr;
    wire [DATA_WIDTH-1:0]    tok_out;

    sram #(.N_CHANNELS(1), .DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH)) sram_tok (
        .clk(clk), .rst(rst), .w_en(tok_w_en),
        .r_addr(tok_r_addr), .w_addr(tok_w_addr),
        .sram_data_in(data_in), .sram_data_out(tok_out)
    );

    // Position SRAM
    reg                      pos_w_en;
    reg  [ADDR_WIDTH-1:0]    pos_w_addr, pos_r_addr;
    wire [DATA_WIDTH-1:0]    pos_out;

    sram #(.N_CHANNELS(1), .DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH)) sram_pos (
        .clk(clk), .rst(rst), .w_en(pos_w_en),
        .r_addr(pos_r_addr), .w_addr(pos_w_addr),
        .sram_data_in(data_in), .sram_data_out(pos_out)
    );

    // Pipeline registers
    reg                      pipe1_valid;
    reg  [DATA_WIDTH-1:0]    pipe1_seg;        // segment value delayed 1 cycle
    reg  [DATA_WIDTH-1:0]    pipe1_tok_pos;    // tok + pos (stage 1)
    reg                      pipe2_valid;
    reg  [ADDR_WIDTH-1:0]    out_addr;

    // FSM transitions
    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    always @* begin
        next_state = state;
        case (state)
            S_IDLE:     if (valid)                        next_state = S_LOAD_TOK;
            S_LOAD_TOK: if (tok_w_addr == DEPTH - 1)      next_state = S_LOAD_POS;
            S_LOAD_POS: if (pos_w_addr == DEPTH - 1)      next_state = S_ADD;
            S_ADD:      if (out_addr == DEPTH - 1 && pipe2_valid) next_state = S_IDLE;
            default:                                       next_state = S_IDLE;
        endcase
    end

    // Datapath
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tok_w_en    <= 1'b0;
            tok_w_addr  <= {ADDR_WIDTH{1'b0}};
            tok_r_addr  <= {ADDR_WIDTH{1'b0}};
            pos_w_en    <= 1'b0;
            pos_w_addr  <= {ADDR_WIDTH{1'b0}};
            pos_r_addr  <= {ADDR_WIDTH{1'b0}};
            pipe1_valid <= 1'b0;
            pipe1_seg   <= {DATA_WIDTH{1'b0}};
            pipe1_tok_pos <= {DATA_WIDTH{1'b0}};
            pipe2_valid <= 1'b0;
            out_addr    <= {ADDR_WIDTH{1'b0}};
            data_out    <= {DATA_WIDTH{1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    tok_w_en   <= 1'b0;
                    tok_w_addr <= {ADDR_WIDTH{1'b0}};
                    pos_w_en   <= 1'b0;
                    pos_w_addr <= {ADDR_WIDTH{1'b0}};
                    tok_r_addr <= {ADDR_WIDTH{1'b0}};
                    pos_r_addr <= {ADDR_WIDTH{1'b0}};
                    pipe1_valid <= 1'b0;
                    pipe2_valid <= 1'b0;
                    out_addr    <= {ADDR_WIDTH{1'b0}};
                    if (valid) begin
                        tok_w_en   <= 1'b1;
                        tok_w_addr <= {ADDR_WIDTH{1'b0}};
                    end
                end

                S_LOAD_TOK: begin
                    tok_w_en <= valid;
                    if (valid)
                        tok_w_addr <= tok_w_addr + 1'b1;
                    pos_w_en <= 1'b0;
                    if (tok_w_addr == DEPTH - 1) begin
                        tok_w_en   <= 1'b0;
                        pos_w_en   <= 1'b1;
                        pos_w_addr <= {ADDR_WIDTH{1'b0}};
                    end
                end

                S_LOAD_POS: begin
                    tok_w_en <= 1'b0;
                    pos_w_en <= valid;
                    if (valid)
                        pos_w_addr <= pos_w_addr + 1'b1;
                    if (pos_w_addr == DEPTH - 1) begin
                        pos_w_en   <= 1'b0;
                        tok_r_addr <= {ADDR_WIDTH{1'b0}};
                        pos_r_addr <= {ADDR_WIDTH{1'b0}};
                    end
                end

                S_ADD: begin
                    tok_w_en <= 1'b0;
                    pos_w_en <= 1'b0;
                    // Cycle 0: drive r_addr, capture segment
                    if (valid) begin
                        tok_r_addr <= tok_r_addr + 1'b1;
                        pos_r_addr <= pos_r_addr + 1'b1;
                    end
                    // Cycle 1: SRAM outputs valid, compute tok + pos
                    pipe1_valid   <= valid;
                    pipe1_seg     <= data_in;
                    pipe1_tok_pos <= $signed(tok_out) + $signed(pos_out);
                    // Cycle 2: add segment
                    pipe2_valid <= pipe1_valid;
                    data_out    <= $signed(pipe1_tok_pos) + $signed(pipe1_seg);
                    if (pipe2_valid)
                        out_addr <= out_addr + 1'b1;
                end

                default: begin
                    tok_w_en    <= 1'b0;
                    pos_w_en    <= 1'b0;
                    pipe1_valid <= 1'b0;
                    pipe2_valid <= 1'b0;
                end
            endcase
        end
    end

    assign ready   = (state == S_LOAD_TOK) || (state == S_LOAD_POS) || (state == S_ADD) || (state == S_IDLE);
    assign valid_n = ~pipe2_valid;

endmodule




// Auto-generated layernorm module
// Per-token LayerNorm over D_MODEL=128 elements
// Pure CLB + DSP: 1 SRAM + accumulator + 2x explicit DSP multiply (27x27) + rsqrt LUT
// FSM: LOAD -> MEAN -> VARIANCE -> RSQRT -> NORMALIZE (repeats per token)
// Multiplies use explicit 'multiply' primitive to force DSP mapping (not DPE).
// Generated by gen_bert_clb_modules.py

module layernorm #(
    parameter DATA_WIDTH = 40,
    parameter D_MODEL    = 128
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    valid,
    input  wire                    ready_n,
    input  wire [DATA_WIDTH-1:0]  data_in,
    output reg  [DATA_WIDTH-1:0]  data_out,
    output wire                    ready,
    output wire                    valid_n
);

    localparam ADDR_WIDTH = 7;
    localparam ACC_WIDTH  = 80;
    localparam LOG2_D     = 7;
    localparam MULT_W     = 27;
    localparam PROD_W     = 54;

    // FSM states
    localparam S_IDLE      = 3'd0;
    localparam S_LOAD      = 3'd1;
    localparam S_MEAN      = 3'd2;
    localparam S_VARIANCE  = 3'd3;
    localparam S_RSQRT     = 3'd4;
    localparam S_NORMALIZE = 3'd5;
    localparam S_DRAIN     = 3'd6;

    reg [2:0] state, next_state;

    // SRAM: stores one token (d_model elements), read 3 times
    reg                      sram_w_en;
    reg  [ADDR_WIDTH-1:0]    sram_w_addr, sram_r_addr;
    wire [DATA_WIDTH-1:0]    sram_out;

    sram #(.N_CHANNELS(1), .DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL)) token_buf (
        .clk(clk), .rst(rst), .w_en(sram_w_en),
        .r_addr(sram_r_addr), .w_addr(sram_w_addr),
        .sram_data_in(data_in), .sram_data_out(sram_out)
    );

    // Accumulator (reused for mean and variance)
    reg signed [ACC_WIDTH-1:0] accumulator;

    // Computed statistics (held for one token)
    reg signed [DATA_WIDTH-1:0] mean_val;
    reg        [DATA_WIDTH-1:0] rsqrt_val_reg;

    // Counters
    reg [ADDR_WIDTH-1:0] elem_count;

    // ========================================================================
    // Explicit DSP multiply primitives (27x27 -> 54)
    // Prevents VTR from mapping wide multiplies to DPE hard blocks.
    // Two multipliers: var_mult (diff²) and norm_mult (diff * rsqrt)
    // ========================================================================

    // Variance multiplier: diff * diff (truncated to 27 bits)
    reg  [MULT_W-1:0] var_mult_a, var_mult_b;
    wire [PROD_W-1:0] var_mult_out;

    multiply var_dsp (
        .a(var_mult_a),
        .b(var_mult_b),
        .out(var_mult_out)
    );

    // Normalize multiplier: diff * rsqrt_val (truncated to 27 bits)
    reg  [MULT_W-1:0] norm_mult_a, norm_mult_b;
    wire [PROD_W-1:0] norm_mult_out;

    multiply norm_dsp (
        .a(norm_mult_a),
        .b(norm_mult_b),
        .out(norm_mult_out)
    );

    // Normalize pipeline registers
    reg signed [DATA_WIDTH-1:0]  norm_pipe1_diff;
    reg                           norm_pipe1_valid;
    reg signed [PROD_W-1:0]      norm_pipe2_product;
    reg                           norm_pipe2_valid;
    reg                           norm_pipe3_valid;
    reg [1:0]                     drain_count;

    // rsqrt LUT (priority encoder — same pattern as reciprocal in attention_head)
    reg [DATA_WIDTH-1:0] rsqrt_val;
    wire [DATA_WIDTH-1:0] var_upper = accumulator[ACC_WIDTH-1:DATA_WIDTH];

    always @* begin
        casez (var_upper)
            40'b1???????????????????????????????????????: rsqrt_val = 40'd1048576;
            40'b01??????????????????????????????????????: rsqrt_val = 40'd1048576;
            40'b001?????????????????????????????????????: rsqrt_val = 40'd2097152;
            40'b0001????????????????????????????????????: rsqrt_val = 40'd2097152;
            40'b00001???????????????????????????????????: rsqrt_val = 40'd4194304;
            40'b000001??????????????????????????????????: rsqrt_val = 40'd4194304;
            40'b0000001?????????????????????????????????: rsqrt_val = 40'd8388608;
            40'b00000001????????????????????????????????: rsqrt_val = 40'd8388608;
            40'b000000001???????????????????????????????: rsqrt_val = 40'd16777216;
            40'b0000000001??????????????????????????????: rsqrt_val = 40'd16777216;
            40'b00000000001?????????????????????????????: rsqrt_val = 40'd33554432;
            40'b000000000001????????????????????????????: rsqrt_val = 40'd33554432;
            40'b0000000000001???????????????????????????: rsqrt_val = 40'd67108864;
            40'b00000000000001??????????????????????????: rsqrt_val = 40'd67108864;
            40'b000000000000001?????????????????????????: rsqrt_val = 40'd134217728;
            40'b0000000000000001????????????????????????: rsqrt_val = 40'd134217728;
            40'b00000000000000001???????????????????????: rsqrt_val = 40'd268435456;
            40'b000000000000000001??????????????????????: rsqrt_val = 40'd268435456;
            40'b0000000000000000001?????????????????????: rsqrt_val = 40'd536870912;
            40'b00000000000000000001????????????????????: rsqrt_val = 40'd536870912;
            40'b000000000000000000001???????????????????: rsqrt_val = 40'd1073741824;
            40'b0000000000000000000001??????????????????: rsqrt_val = 40'd1073741824;
            40'b00000000000000000000001?????????????????: rsqrt_val = 40'd2147483648;
            40'b000000000000000000000001????????????????: rsqrt_val = 40'd2147483648;
            40'b0000000000000000000000001???????????????: rsqrt_val = 40'd4294967296;
            40'b00000000000000000000000001??????????????: rsqrt_val = 40'd4294967296;
            40'b000000000000000000000000001?????????????: rsqrt_val = 40'd8589934592;
            40'b0000000000000000000000000001????????????: rsqrt_val = 40'd8589934592;
            40'b00000000000000000000000000001???????????: rsqrt_val = 40'd17179869184;
            40'b000000000000000000000000000001??????????: rsqrt_val = 40'd17179869184;
            40'b0000000000000000000000000000001?????????: rsqrt_val = 40'd34359738368;
            40'b00000000000000000000000000000001????????: rsqrt_val = 40'd34359738368;
            40'b000000000000000000000000000000001???????: rsqrt_val = 40'd68719476736;
            40'b0000000000000000000000000000000001??????: rsqrt_val = 40'd68719476736;
            40'b00000000000000000000000000000000001?????: rsqrt_val = 40'd137438953472;
            40'b000000000000000000000000000000000001????: rsqrt_val = 40'd137438953472;
            40'b0000000000000000000000000000000000001???: rsqrt_val = 40'd274877906944;
            40'b00000000000000000000000000000000000001??: rsqrt_val = 40'd274877906944;
            40'b000000000000000000000000000000000000001?: rsqrt_val = 40'd549755813888;
            40'b0000000000000000000000000000000000000001: rsqrt_val = 40'd549755813888;
            default: rsqrt_val = {DATA_WIDTH{1'b1}};
        endcase
    end

    // FSM transitions
    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    always @* begin
        next_state = state;
        case (state)
            S_IDLE:      if (valid)                           next_state = S_LOAD;
            S_LOAD:      if (sram_w_addr == D_MODEL - 1)      next_state = S_MEAN;
            S_MEAN:      if (elem_count == D_MODEL - 1)        next_state = S_VARIANCE;
            S_VARIANCE:  if (elem_count == D_MODEL - 1)        next_state = S_RSQRT;
            S_RSQRT:                                           next_state = S_NORMALIZE;
            S_NORMALIZE: if (elem_count == D_MODEL - 1)        next_state = S_DRAIN;
            S_DRAIN:     if (drain_count == 2'd2)              next_state = S_IDLE;
            default:                                           next_state = S_IDLE;
        endcase
    end

    // Datapath
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sram_w_en       <= 1'b0;
            sram_w_addr     <= {ADDR_WIDTH{1'b0}};
            sram_r_addr     <= {ADDR_WIDTH{1'b0}};
            accumulator     <= {ACC_WIDTH{1'b0}};
            mean_val        <= {DATA_WIDTH{1'b0}};
            rsqrt_val_reg   <= {DATA_WIDTH{1'b0}};
            elem_count      <= {ADDR_WIDTH{1'b0}};
            var_mult_a      <= {MULT_W{1'b0}};
            var_mult_b      <= {MULT_W{1'b0}};
            norm_mult_a     <= {MULT_W{1'b0}};
            norm_mult_b     <= {MULT_W{1'b0}};
            norm_pipe1_diff  <= {DATA_WIDTH{1'b0}};
            norm_pipe1_valid <= 1'b0;
            norm_pipe2_product <= {PROD_W{1'b0}};
            norm_pipe2_valid <= 1'b0;
            norm_pipe3_valid <= 1'b0;
            drain_count      <= 2'd0;
            data_out         <= {DATA_WIDTH{1'b0}};
        end else begin
            case (state)
                // ---- IDLE: reset for next token ----
                S_IDLE: begin
                    sram_w_en    <= 1'b0;
                    sram_w_addr  <= {ADDR_WIDTH{1'b0}};
                    sram_r_addr  <= {ADDR_WIDTH{1'b0}};
                    accumulator  <= {ACC_WIDTH{1'b0}};
                    elem_count   <= {ADDR_WIDTH{1'b0}};
                    norm_pipe1_valid <= 1'b0;
                    norm_pipe2_valid <= 1'b0;
                    norm_pipe3_valid <= 1'b0;
                    drain_count  <= 2'd0;
                    if (valid) begin
                        sram_w_en   <= 1'b1;
                        sram_w_addr <= {ADDR_WIDTH{1'b0}};
                    end
                end

                // ---- LOAD: stream d_model elements into SRAM ----
                S_LOAD: begin
                    sram_w_en <= valid;
                    if (valid)
                        sram_w_addr <= sram_w_addr + 1'b1;
                    if (sram_w_addr == D_MODEL - 1) begin
                        sram_w_en   <= 1'b0;
                        sram_r_addr <= {ADDR_WIDTH{1'b0}};
                        accumulator <= {ACC_WIDTH{1'b0}};
                        elem_count  <= {ADDR_WIDTH{1'b0}};
                    end
                end

                // ---- MEAN: accumulate sum, then right-shift to get mean ----
                S_MEAN: begin
                    sram_w_en <= 1'b0;
                    sram_r_addr <= sram_r_addr + 1'b1;
                    accumulator <= accumulator + {{(ACC_WIDTH-DATA_WIDTH){sram_out[DATA_WIDTH-1]}}, sram_out};
                    elem_count  <= elem_count + 1'b1;
                    if (elem_count == D_MODEL - 1) begin
                        mean_val    <= accumulator[DATA_WIDTH-1+LOG2_D:LOG2_D];
                        accumulator <= {ACC_WIDTH{1'b0}};
                        sram_r_addr <= {ADDR_WIDTH{1'b0}};
                        elem_count  <= {ADDR_WIDTH{1'b0}};
                    end
                end

                // ---- VARIANCE: accumulate (x - mean)^2 via DSP multiply ----
                S_VARIANCE: begin
                    sram_r_addr <= sram_r_addr + 1'b1;
                    elem_count  <= elem_count + 1'b1;
                    begin : var_block
                        reg signed [DATA_WIDTH-1:0] diff;
                        diff = $signed(sram_out) - $signed(mean_val);
                        // Truncate diff to MULT_W bits for DSP multiply
                        var_mult_a <= diff[MULT_W-1:0];
                        var_mult_b <= diff[MULT_W-1:0];
                    end
                    // Accumulate DSP product (var_mult_out is combinational)
                    accumulator <= accumulator + {{(ACC_WIDTH-PROD_W){var_mult_out[PROD_W-1]}}, var_mult_out};
                    if (elem_count == D_MODEL - 1) begin
                        sram_r_addr <= {ADDR_WIDTH{1'b0}};
                        elem_count  <= {ADDR_WIDTH{1'b0}};
                    end
                end

                // ---- RSQRT: lookup rsqrt(variance) in 1 cycle ----
                S_RSQRT: begin
                    rsqrt_val_reg <= rsqrt_val;
                    sram_r_addr   <= {ADDR_WIDTH{1'b0}};
                    elem_count    <= {ADDR_WIDTH{1'b0}};
                end

                // ---- NORMALIZE: y = (x - mean) * rsqrt_val via DSP multiply ----
                S_NORMALIZE: begin
                    sram_r_addr <= sram_r_addr + 1'b1;
                    elem_count  <= elem_count + 1'b1;

                    // Pipeline stage 1: diff = x - mean, feed DSP
                    norm_pipe1_diff  <= $signed(sram_out) - $signed(mean_val);
                    norm_pipe1_valid <= 1'b1;
                    norm_mult_a      <= norm_pipe1_diff[MULT_W-1:0];
                    norm_mult_b      <= rsqrt_val_reg[MULT_W-1:0];

                    // Pipeline stage 2: capture DSP product
                    norm_pipe2_product <= norm_mult_out;
                    norm_pipe2_valid   <= norm_pipe1_valid;

                    // Pipeline stage 3: truncate to DATA_WIDTH and output
                    data_out         <= norm_pipe2_product[PROD_W-2:PROD_W-1-DATA_WIDTH];
                    norm_pipe3_valid <= norm_pipe2_valid;

                    if (elem_count == D_MODEL - 1) begin
                        norm_pipe1_valid <= 1'b0;
                    end
                end

                // ---- DRAIN: flush pipeline (2 extra cycles) ----
                S_DRAIN: begin
                    drain_count <= drain_count + 1'b1;
                    // Stage 2
                    norm_mult_a      <= norm_pipe1_diff[MULT_W-1:0];
                    norm_mult_b      <= rsqrt_val_reg[MULT_W-1:0];
                    norm_pipe2_product <= norm_mult_out;
                    norm_pipe2_valid   <= norm_pipe1_valid;
                    norm_pipe1_valid   <= 1'b0;
                    // Stage 3
                    data_out         <= norm_pipe2_product[PROD_W-2:PROD_W-1-DATA_WIDTH];
                    norm_pipe3_valid <= norm_pipe2_valid;
                    if (drain_count == 2'd2)
                        norm_pipe3_valid <= 1'b0;
                end

                default: begin
                    sram_w_en        <= 1'b0;
                    norm_pipe1_valid <= 1'b0;
                    norm_pipe2_valid <= 1'b0;
                    norm_pipe3_valid <= 1'b0;
                end
            endcase
        end
    end

    assign ready   = (state == S_LOAD) || (state == S_IDLE);
    assign valid_n = ~norm_pipe3_valid;

endmodule




// ═══════════════════════════════════════
// Supporting modules (DPE primitives)
// ═══════════════════════════════════════
module conv_layer_single_dpe #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 5,
    parameter KERNEL_HEIGHT = 5,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter DATA_WIDTH = 40

)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [(N_CHANNELS*DATA_WIDTH)-1:0] data_in,
    output wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    // Internal signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire reg_full;
    wire reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire [DATA_WIDTH-1:0] sram_data_out;

    // Instantiate the SRAM module
    sram #(
        .N_CHANNELS(1),
        .DEPTH(512)
    ) sram_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_out)
    );

    // Instantiate the DPE module (16-bit direct, no zero padding)
    dpe dpe_inst (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready),
        .data_out(data_out),
        .dpe_done(dpe_done),
        .reg_full(reg_full),
        .shift_add_done(shift_add_done),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl)
    );

    // Instantiate the Controller module
    conv_controller #(
        .N_CHANNELS(N_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(N_KERNELS),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .KERNEL_HEIGHT(KERNEL_HEIGHT),
        .W(W),
        .H(H),
        .S(S)
    ) controller_inst (
        .clk(clk),
        .rst(rst),
        .MSB_SA_Ready(MSB_SA_Ready),
        .valid(valid),
        .ready_n(ready_n),
        .dpe_done(dpe_done),
        .reg_full(reg_full),
        .shift_add_done(shift_add_done),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl),
        .read_address(read_address),
        .write_address(write_address),
        .w_buf_en(w_buf_en),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .w_en(w_en),
        .load_input_reg(load_input_reg),
        .ready(ready),
        .valid_n(valid_n)
    );

endmodule


module conv_controller #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 5,
    parameter KERNEL_HEIGHT = 5,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter B_ADDR_WIDTH = $clog2(KERNEL_WIDTH * KERNEL_HEIGHT),
    parameter S_BITWIDTH = S,
    parameter KW_BITWIDTH = $clog2(KERNEL_WIDTH),
    parameter KH_BITWIDTH = $clog2(KERNEL_HEIGHT)
)(
    input wire clk,
    input wire rst,
    input wire MSB_SA_Ready,
    input wire valid,
    input wire ready_n,
    input wire dpe_done,
    input wire reg_full,
    input wire shift_add_done,
    input wire shift_add_bypass_ctrl,
    output reg [ADDR_WIDTH-1:0] read_address,
    output reg [ADDR_WIDTH-1:0] write_address,
    output reg w_buf_en,
    output reg [1:0] nl_dpe_control,
    output reg shift_add_control,
    output reg shift_add_bypass,
    output reg load_output_reg,
    output reg w_en,
    output wire load_input_reg,
    output reg ready,
    output reg valid_n
);

    reg [ADDR_WIDTH-1:0] write_address_reg, read_address_reg;
    wire buf_load, busy;
    reg memory_flag;
    reg stall;
    reg memory_stall;
    reg dpe_exec_signal;
    reg [ADDR_WIDTH-1:0] next_address;

    reg [S_BITWIDTH-1:0] s, sv;
    reg [KW_BITWIDTH-1:0] n;
    reg [KH_BITWIDTH-1:0] m;
    reg [ADDR_WIDTH-1:0] pointer_offset;

    // always block for sram control
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            write_address_reg <= {ADDR_WIDTH{1'b0}};
            w_en <= 1'b0;
            w_buf_en <= 0;
            next_address <= {ADDR_WIDTH{1'b0}};
            pointer_offset <= 0;
            sv <= 0;
            s <= 0;
            n <= 0;
            m <= 0;
        end else begin
            if (valid && ~memory_stall) begin
                write_address_reg <= write_address_reg + 1;
                w_en <= 1'b1;
            end else begin
                w_en <= 1'b0;
            end

            if (~reg_full && ready_n && memory_flag) begin
                pointer_offset <= n + W * (m + sv) + s * S;
                if (n < KERNEL_WIDTH-1) begin
                    n <= n + 1;
                end else begin
                    n <= 0;
                    if (m < KERNEL_HEIGHT-1) begin
                        m <= m + 1;
                    end else begin
                        m <= 0;
                        if (s < W - KERNEL_WIDTH + S - 1) begin
                            s <= s + 1;
                        end else begin
                            s <= 0;
                            if (sv < H - KERNEL_HEIGHT + S - 1) begin
                                sv <= sv + 1;
                            end else begin
                                sv <= 0;
                            end
                        end
                    end
                end
                w_buf_en <= 1;
            end else begin
                w_buf_en <= 0;
            end
        end
    end

    always @* begin
        read_address_reg <= pointer_offset;
    end

    always @* begin
        if ((write_address_reg > read_address_reg) ||
            ((write_address_reg == {ADDR_WIDTH{1'b0}}) && (read_address_reg == {ADDR_WIDTH{1'b1}}))) begin
            memory_flag <= 1;
        end else begin
            memory_flag <= 0;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            dpe_exec_signal <= 1'b0;
        end else begin
            if (reg_full) begin
                dpe_exec_signal <= 1'b1;
            end else begin
                dpe_exec_signal <= 1'b0;
            end
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            nl_dpe_control <= 2'b0;
        end else begin
            if (dpe_exec_signal) begin
                nl_dpe_control <= 2'b11;
            end else begin
                nl_dpe_control <= 2'b00;
            end
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_n <= 1'b0;
            ready <= 1'b0;
            stall <= 0;
            memory_stall <= 0;
        end else begin
            if ((read_address_reg < write_address_reg - 2) && (write_address_reg == {ADDR_WIDTH{1'b1}})) begin
                memory_stall <= 1;
            end else begin
                memory_stall <= 0;
            end

            if (memory_stall) begin
                ready <= 1'b0;
            end else begin
                ready <= 1'b1;
            end

            if (dpe_done) begin
                valid_n <= 1;
            end else begin
                valid_n <= 0;
            end

            if (ready_n) begin
                stall <= 0;
            end else begin
                stall <= 1;
            end
        end
    end

    always @* begin
        read_address <= read_address_reg;
        write_address <= write_address_reg;
        shift_add_bypass <= shift_add_bypass_ctrl;
        shift_add_control <= MSB_SA_Ready;
        load_output_reg <= shift_add_done;
    end

    assign load_input_reg = reg_full;
    assign busy = ~MSB_SA_Ready;

endmodule


module sram #(
    parameter N_CHANNELS = 1,
    parameter DATA_WIDTH = 40*N_CHANNELS,  // Data width (default: 16 bits) 16 x number of channels
    parameter DEPTH = 512       // Memory depth (default: 512)

)(
    input wire clk,
    input wire w_en,
	input wire rst,
    input wire [$clog2(DEPTH)-1:0] r_addr,
    input wire [$clog2(DEPTH)-1:0] w_addr,
    input wire [DATA_WIDTH-1:0] sram_data_in,
    output reg [DATA_WIDTH-1:0] sram_data_out
);

    // Memory array with parameterized depth and width
    reg [DATA_WIDTH-1:0] mem [DEPTH-1:0];

    // Read/Write operations
    always @(posedge clk) begin
        if (rst) begin
            sram_data_out <= {DATA_WIDTH{1'b0}};
        end else begin
            sram_data_out <= mem[r_addr];
        end
    end
    always @(posedge clk) begin
            if (w_en) begin
                mem[w_addr] <= sram_data_in;
            end
    end

endmodule


module global_controller #(
    parameter N_Layers = 1
)(
    input wire clk,
    input wire rst,
    input wire ready_L1,
    input wire valid_Ln,
    input wire valid,
    output reg ready,
    output reg valid_L1,
    output reg ready_Ln
);

    wire busy;
    reg stall;

    // valid and ready control
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            stall <= 0;
        end else begin

            if (stall) begin
                ready_Ln <= 1'b0;
            end else begin
                ready_Ln <= 1'b1;
            end

            if(~valid) begin
                stall <= 0;
            end else begin
                stall <= 1;
            end
        end
    end

    always @* begin
        ready <= ready_L1;
        valid_L1 <= valid;
    end

endmodule


module controller_scalable #(
    parameter N_CHANNELS = 1,
    parameter N_BRAM_R = 4,
    parameter N_BRAM_W = 8,
    parameter N_DPE_V = 2,
    parameter N_DPE_H = 16,
    parameter ADDR_WIDTH = 10,
    parameter N_KERNELS = 6,
    parameter KERNEL_WIDTH = 5,
    parameter KERNEL_HEIGHT = 5,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter B_ADDR_WIDTH = $clog2(KERNEL_WIDTH * KERNEL_HEIGHT),
    parameter S_BITWIDTH = S,
    parameter KW_BITWIDTH = $clog2(KERNEL_WIDTH),
    parameter KH_BITWIDTH = $clog2(KERNEL_HEIGHT)
)(
    // Clock and reset
    input wire clk,
    input wire rst,

    // Control inputs
    input wire MSB_SA_Ready,
    input wire valid,
    input wire ready_n,
    input wire dpe_done,
    input wire [N_DPE_V-1:0] reg_full,
    input wire [N_DPE_H-1:0] reg_empty,
    input wire shift_add_done,
    input wire shift_add_bypass_ctrl,
    input wire dpe_accum_done,

    // Memory interface
    output reg [ADDR_WIDTH-1:0] read_address,
    output reg [ADDR_WIDTH-1:0] write_address,
    output wire [(N_BRAM_W)-1:0] w_en_dec,

    // DPE control outputs
    output reg [N_DPE_V-1:0] w_buf_en,
    output reg [1:0] nl_dpe_control,
    output reg shift_add_control,
    output reg shift_add_bypass,
    output reg load_output_reg,
    output wire load_input_reg,
    output reg [$clog2(N_DPE_V)-1:0] dpe_sel,
    output reg [$clog2(N_DPE_H)-1:0] dpe_sel_h,

    // Status signals
    output reg ready,
    output reg valid_n,
    output reg dpe_accum_ready
);
    // Local parameters
    localparam VR_ADDR_WIDTH = $clog2(N_BRAM_R) + ADDR_WIDTH;
    localparam VW_ADDR_WIDTH = $clog2(N_BRAM_W) + ADDR_WIDTH;
    localparam N_DPE_SEL_V = $clog2(N_DPE_V);
    localparam N_DPE_SEL_H = $clog2(N_DPE_H);

    // Internal registers
    reg w_en;
    reg [VW_ADDR_WIDTH-1:0] write_address_reg;
    reg [VR_ADDR_WIDTH-1:0] read_address_reg;
    reg [$clog2(N_DPE_H)-1:0] dpe_h_count;
    reg [$clog2(N_DPE_V)-1:0] dpe_v_count;
    reg memory_flag;
    reg stall;
    reg memory_stall;
    reg dpe_exec_signal;
    reg [VW_ADDR_WIDTH-1:0] next_address;

    // Address generation registers
    reg [S_BITWIDTH-1:0] s, sv;
    reg [KW_BITWIDTH-1:0] n;
    reg [KH_BITWIDTH-1:0] m;
    reg [VR_ADDR_WIDTH-1:0] pointer_offset;

    // Bank select registers
    reg [N_BRAM_W-1:0] w_dec;
    reg [N_BRAM_R-1:0] r_dec;

    // Internal wires
    wire busy;

    // SRAM control logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            w_en <= 1'b0;
            w_buf_en <= 0;
            next_address <= {VW_ADDR_WIDTH{1'b0}};
            pointer_offset <= 0;
            sv <= 0;
            s <= 0;
            n <= 0;
            m <= 0;
        end else begin
            if (valid && ~memory_stall) begin
                next_address <= next_address + 1;
                w_en <= 1'b1;
            end else begin
                w_en <= 0;
            end

            if (~(&reg_full) && ready_n && memory_flag) begin
                pointer_offset <= n + W * (m + sv) + s * S;
                if (n < KERNEL_WIDTH-1) begin
                    n <= n + 1;
                end else begin
                    n <= 0;
                    if (m < KERNEL_HEIGHT-1) begin
                        m <= m + 1;
                    end else begin
                        m <= 0;
                        if (s < W - KERNEL_WIDTH + S - 1) begin
                            s <= s + 1;
                        end else begin
                            s <= 0;
                            if (sv < H - KERNEL_HEIGHT + S - 1) begin
                                sv <= sv + 1;
                            end else begin
                                sv <= 0;
                            end
                        end
                    end
                end
                w_buf_en <= 1<<dpe_v_count;
            end else begin
                w_buf_en <= 0;
            end
        end
    end

    // Address assignment
    always @* begin
        read_address_reg = pointer_offset;
        write_address_reg = next_address;
    end

    // Memory flag logic
    always @* begin
        if ((write_address_reg > read_address_reg) ||
            ((write_address_reg == {VW_ADDR_WIDTH{1'b0}}) && (read_address_reg == {VR_ADDR_WIDTH{1'b1}}))) begin
            memory_flag = 1;
        end else begin
            memory_flag = 0;
        end
    end

    // DPE execution control
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            dpe_exec_signal <= 1'b0;
        end else begin
            if (&reg_full) begin
                dpe_exec_signal <= 1'b1;
            end else begin
                dpe_exec_signal <= 1'b0;
            end
        end
    end

    // DPE control logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            nl_dpe_control <= 2'b0;
        end else begin
            if (dpe_exec_signal) begin
                nl_dpe_control <= 2'b11;
            end else begin
                nl_dpe_control <= 2'b00;
            end
        end
    end

    // Status control
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_n <= 1'b0;
            ready <= 1'b0;
            stall <= 0;
            memory_stall <= 0;
        end else begin
            if ((read_address_reg < write_address_reg - 2) && (write_address_reg == {VW_ADDR_WIDTH{1'b1}})) begin
                memory_stall <= 1;
            end else begin
                memory_stall <= 0;
            end

            if (memory_stall) begin
                ready <= 1'b0;
            end else begin
                ready <= 1'b1;
            end

            if (dpe_done) begin
                valid_n <= 1;
            end else begin
                valid_n <= 0;
            end

            if (ready_n) begin
                stall <= 0;
            end else begin
                stall <= 1;
            end
        end
    end

    // Priority encoder for reg_full
    integer v_bit;
    always @* begin
        dpe_v_count = {N_DPE_SEL_V{1'b0}};  // Default output is 0

        for (v_bit = N_DPE_V-1; v_bit >= 0; v_bit = v_bit - 1) begin
            if (reg_full[v_bit]) begin
                // Properly convert integer to the correct bit width
                dpe_v_count = v_bit + 1;
            end
        end
    end

    // Priority encoder for reg_empty
    integer h_bit;
    always @* begin
        dpe_h_count = {N_DPE_SEL_H{1'b0}};  // Default output is 0

        for (h_bit = 0; h_bit < N_DPE_H; h_bit = h_bit + 1) begin
            if (reg_empty[h_bit]) begin
                // Properly convert integer to the correct bit width
                dpe_h_count = h_bit + 1;
            end
        end
    end

    // Output assignments
    always @* begin
        read_address = read_address_reg[ADDR_WIDTH-1:0];
        write_address = write_address_reg[ADDR_WIDTH-1:0];
        dpe_sel = dpe_v_count;
        dpe_sel_h = dpe_h_count;
        shift_add_bypass = shift_add_bypass_ctrl;
        shift_add_control = MSB_SA_Ready;
        dpe_accum_ready = shift_add_done;
        load_output_reg = dpe_accum_done;

        // Fixed bank selection logic using proper bit ranges
        w_dec = {N_BRAM_W{1'b0}};
        r_dec = {N_BRAM_R{1'b0}};

        // Only perform shift if index is within valid range
        if (N_BRAM_W > 1) begin
            if (write_address_reg[VW_ADDR_WIDTH-1:ADDR_WIDTH] < N_BRAM_W) begin
                w_dec = 1'b1 << write_address_reg[VW_ADDR_WIDTH-1:ADDR_WIDTH];
            end
        end
        else begin
            w_dec = 1'b1 << write_address_reg[ADDR_WIDTH];
        end

        if (N_BRAM_R > 1) begin
            if (read_address_reg[VR_ADDR_WIDTH-1:ADDR_WIDTH] < N_BRAM_R) begin
                r_dec = 1'b1 << read_address_reg[VR_ADDR_WIDTH-1:ADDR_WIDTH];
            end
        end
        else begin
            r_dec = 1'b1 << write_address_reg[ADDR_WIDTH];
        end
    end

    // Output assignments
    assign w_en_dec = w_dec & {N_BRAM_W{w_en}};
    assign load_input_reg = reg_full;
    assign busy = ~MSB_SA_Ready;

endmodule


module xbar_ip_module #(
    parameter DATA_WIDTH = 40,
    parameter NUM_INPUTS = 1,
    parameter NUM_OUTPUTS = 4,
    // Derived parameters to handle special cases
    parameter IN_SEL_WIDTH = (NUM_INPUTS <= 1) ? 1 : $clog2(NUM_INPUTS),
    parameter OUT_SEL_WIDTH = (NUM_OUTPUTS <= 1) ? 1 : $clog2(NUM_OUTPUTS)
)(
    input  wire [NUM_INPUTS*DATA_WIDTH-1:0]  in_data,
    input  wire [IN_SEL_WIDTH-1:0]           in_sel,
    input  wire [OUT_SEL_WIDTH-1:0]          out_sel,
    output wire [NUM_OUTPUTS*DATA_WIDTH-1:0] out_data
);

    // Generate appropriate implementation based on parameters
    generate
        // Case 1: Both inputs and outputs > 1 (full crossbar)
        if (NUM_INPUTS > 1 && NUM_OUTPUTS > 1) begin : full_crossbar
            reg [NUM_OUTPUTS*DATA_WIDTH-1:0] out_data_reg;

            integer i, j;
            reg [DATA_WIDTH-1:0] in_data_arr [NUM_INPUTS-1:0];
            reg [DATA_WIDTH-1:0] out_data_arr [NUM_OUTPUTS-1:0];

            always @(*) begin
                // Unpack input data
                for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                    in_data_arr[i] = in_data[i*DATA_WIDTH +: DATA_WIDTH];
                end

                // Default outputs to zero
                for (j = 0; j < NUM_OUTPUTS; j = j + 1) begin
                    out_data_arr[j] = {DATA_WIDTH{1'b0}};
                end

                // Route data based on selectors
                for (j = 0; j < NUM_OUTPUTS; j = j + 1) begin
                    for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                        if ((in_sel == i) && (out_sel == j)) begin
                            out_data_arr[j] = in_data_arr[i];
                        end
                    end
                end

                // Pack output data
                for (j = 0; j < NUM_OUTPUTS; j = j + 1) begin
                    out_data_reg[j*DATA_WIDTH +: DATA_WIDTH] = out_data_arr[j];
                end
            end

            assign out_data = out_data_reg;
        end

        // Case 2: Single input, multiple outputs (fan-out)
        else if (NUM_INPUTS == 1 && NUM_OUTPUTS > 1) begin : single_input_multi_output
            wire [DATA_WIDTH-1:0] input_data = in_data[DATA_WIDTH-1:0];
            reg [NUM_OUTPUTS*DATA_WIDTH-1:0] out_data_reg;

            always @(*) begin
                // Default all outputs to 0
                out_data_reg = {NUM_OUTPUTS*DATA_WIDTH{1'b0}};

                // Send input data to selected output
                if (out_sel < NUM_OUTPUTS) begin
                    out_data_reg[out_sel*DATA_WIDTH +: DATA_WIDTH] = input_data;
                end
            end

            assign out_data = out_data_reg;
        end

        // Case 3: Multiple inputs, single output (multiplexer)
        else if (NUM_INPUTS > 1 && NUM_OUTPUTS == 1) begin : multi_input_single_output
            reg [DATA_WIDTH-1:0] out_data_reg;

            always @(*) begin
                // Default output to 0
                out_data_reg = {DATA_WIDTH{1'b0}};

                // Select from inputs
                if (in_sel < NUM_INPUTS) begin
                    out_data_reg = in_data[in_sel*DATA_WIDTH +: DATA_WIDTH];
                end
            end

            assign out_data = out_data_reg;
        end

        // Case 4: Single input, single output (direct connection)
        else begin : single_input_single_output
            // Just pass through - selectors are ignored
            assign out_data = in_data;
        end
    endgenerate

endmodule
