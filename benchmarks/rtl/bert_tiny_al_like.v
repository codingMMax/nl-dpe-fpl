// Auto-generated BERT-Tiny RTL — NL-DPE 1024×256
// Total DPEs: 78
// Q/K/V/O: V=1 H=1 → 1 DPE each
// FFN1: V=1 H=2 → 2 DPEs
// FFN2: V=1 H=1 → 1 DPEs
// DIMM: W=4 lanes, K_qkt=4, K_sv=2
// DIMM: 16 DPEs/head × 2 heads × 2 blocks = 64

module bert_tiny_al_like (
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
    wire [40-1:0] data_b0_h0_w0_score;
    wire valid_b0_h0_w0_score, ready_b0_h0_w0_score;
    wire [40-1:0] data_b0_h0_w0_softmax;
    wire valid_b0_h0_w0_softmax, ready_b0_h0_w0_softmax;
    wire [40-1:0] data_b0_h0_w0_wsum;
    wire valid_b0_h0_w0_wsum, ready_b0_h0_w0_wsum;
    wire [40-1:0] data_b0_h0_w1_score;
    wire valid_b0_h0_w1_score, ready_b0_h0_w1_score;
    wire [40-1:0] data_b0_h0_w1_softmax;
    wire valid_b0_h0_w1_softmax, ready_b0_h0_w1_softmax;
    wire [40-1:0] data_b0_h0_w1_wsum;
    wire valid_b0_h0_w1_wsum, ready_b0_h0_w1_wsum;
    wire [40-1:0] data_b0_h0_w2_score;
    wire valid_b0_h0_w2_score, ready_b0_h0_w2_score;
    wire [40-1:0] data_b0_h0_w2_softmax;
    wire valid_b0_h0_w2_softmax, ready_b0_h0_w2_softmax;
    wire [40-1:0] data_b0_h0_w2_wsum;
    wire valid_b0_h0_w2_wsum, ready_b0_h0_w2_wsum;
    wire [40-1:0] data_b0_h0_w3_score;
    wire valid_b0_h0_w3_score, ready_b0_h0_w3_score;
    wire [40-1:0] data_b0_h0_w3_softmax;
    wire valid_b0_h0_w3_softmax, ready_b0_h0_w3_softmax;
    wire [40-1:0] data_b0_h0_w3_wsum;
    wire valid_b0_h0_w3_wsum, ready_b0_h0_w3_wsum;
    wire [40-1:0] data_b0_h1_w0_score;
    wire valid_b0_h1_w0_score, ready_b0_h1_w0_score;
    wire [40-1:0] data_b0_h1_w0_softmax;
    wire valid_b0_h1_w0_softmax, ready_b0_h1_w0_softmax;
    wire [40-1:0] data_b0_h1_w0_wsum;
    wire valid_b0_h1_w0_wsum, ready_b0_h1_w0_wsum;
    wire [40-1:0] data_b0_h1_w1_score;
    wire valid_b0_h1_w1_score, ready_b0_h1_w1_score;
    wire [40-1:0] data_b0_h1_w1_softmax;
    wire valid_b0_h1_w1_softmax, ready_b0_h1_w1_softmax;
    wire [40-1:0] data_b0_h1_w1_wsum;
    wire valid_b0_h1_w1_wsum, ready_b0_h1_w1_wsum;
    wire [40-1:0] data_b0_h1_w2_score;
    wire valid_b0_h1_w2_score, ready_b0_h1_w2_score;
    wire [40-1:0] data_b0_h1_w2_softmax;
    wire valid_b0_h1_w2_softmax, ready_b0_h1_w2_softmax;
    wire [40-1:0] data_b0_h1_w2_wsum;
    wire valid_b0_h1_w2_wsum, ready_b0_h1_w2_wsum;
    wire [40-1:0] data_b0_h1_w3_score;
    wire valid_b0_h1_w3_score, ready_b0_h1_w3_score;
    wire [40-1:0] data_b0_h1_w3_softmax;
    wire valid_b0_h1_w3_softmax, ready_b0_h1_w3_softmax;
    wire [40-1:0] data_b0_h1_w3_wsum;
    wire valid_b0_h1_w3_wsum, ready_b0_h1_w3_wsum;
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
    wire [40-1:0] data_b1_h0_w0_score;
    wire valid_b1_h0_w0_score, ready_b1_h0_w0_score;
    wire [40-1:0] data_b1_h0_w0_softmax;
    wire valid_b1_h0_w0_softmax, ready_b1_h0_w0_softmax;
    wire [40-1:0] data_b1_h0_w0_wsum;
    wire valid_b1_h0_w0_wsum, ready_b1_h0_w0_wsum;
    wire [40-1:0] data_b1_h0_w1_score;
    wire valid_b1_h0_w1_score, ready_b1_h0_w1_score;
    wire [40-1:0] data_b1_h0_w1_softmax;
    wire valid_b1_h0_w1_softmax, ready_b1_h0_w1_softmax;
    wire [40-1:0] data_b1_h0_w1_wsum;
    wire valid_b1_h0_w1_wsum, ready_b1_h0_w1_wsum;
    wire [40-1:0] data_b1_h0_w2_score;
    wire valid_b1_h0_w2_score, ready_b1_h0_w2_score;
    wire [40-1:0] data_b1_h0_w2_softmax;
    wire valid_b1_h0_w2_softmax, ready_b1_h0_w2_softmax;
    wire [40-1:0] data_b1_h0_w2_wsum;
    wire valid_b1_h0_w2_wsum, ready_b1_h0_w2_wsum;
    wire [40-1:0] data_b1_h0_w3_score;
    wire valid_b1_h0_w3_score, ready_b1_h0_w3_score;
    wire [40-1:0] data_b1_h0_w3_softmax;
    wire valid_b1_h0_w3_softmax, ready_b1_h0_w3_softmax;
    wire [40-1:0] data_b1_h0_w3_wsum;
    wire valid_b1_h0_w3_wsum, ready_b1_h0_w3_wsum;
    wire [40-1:0] data_b1_h1_w0_score;
    wire valid_b1_h1_w0_score, ready_b1_h1_w0_score;
    wire [40-1:0] data_b1_h1_w0_softmax;
    wire valid_b1_h1_w0_softmax, ready_b1_h1_w0_softmax;
    wire [40-1:0] data_b1_h1_w0_wsum;
    wire valid_b1_h1_w0_wsum, ready_b1_h1_w0_wsum;
    wire [40-1:0] data_b1_h1_w1_score;
    wire valid_b1_h1_w1_score, ready_b1_h1_w1_score;
    wire [40-1:0] data_b1_h1_w1_softmax;
    wire valid_b1_h1_w1_softmax, ready_b1_h1_w1_softmax;
    wire [40-1:0] data_b1_h1_w1_wsum;
    wire valid_b1_h1_w1_wsum, ready_b1_h1_w1_wsum;
    wire [40-1:0] data_b1_h1_w2_score;
    wire valid_b1_h1_w2_score, ready_b1_h1_w2_score;
    wire [40-1:0] data_b1_h1_w2_softmax;
    wire valid_b1_h1_w2_softmax, ready_b1_h1_w2_softmax;
    wire [40-1:0] data_b1_h1_w2_wsum;
    wire valid_b1_h1_w2_wsum, ready_b1_h1_w2_wsum;
    wire [40-1:0] data_b1_h1_w3_score;
    wire valid_b1_h1_w3_score, ready_b1_h1_w3_score;
    wire [40-1:0] data_b1_h1_w3_softmax;
    wire valid_b1_h1_w3_softmax, ready_b1_h1_w3_softmax;
    wire [40-1:0] data_b1_h1_w3_wsum;
    wire valid_b1_h1_w3_wsum, ready_b1_h1_w3_wsum;
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
        .W(128), .H(1), .S(1),
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
        .W(128), .H(1), .S(1),
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
        .W(128), .H(1), .S(1),
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

    // Head 0: W=4 DIMM lanes, K_qkt=4, K_sv=2
    dimm_score_matrix_b0_h0_w0 #(.DATA_WIDTH(40)) b0_h0_w0_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h0_w0_score),
        .ready_q(ready_b0_h0_score_q), .ready_k(ready_b0_h0_score_k),
        .valid_n(valid_b0_h0_w0_score)
    );
    softmax_approx_b0_h0_w0 #(.DATA_WIDTH(40)) b0_h0_w0_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h0_w0_score), .ready_n(1'b1),
        .data_in(data_b0_h0_w0_score),
        .data_out(data_b0_h0_w0_softmax), .ready(), .valid_n(valid_b0_h0_w0_softmax)
    );
    dimm_weighted_sum_b0_h0_w0 #(.DATA_WIDTH(40)) b0_h0_w0_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h0_w0_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h0_w0_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h0_w0_wsum),
        .ready_attn(), .ready_v(ready_b0_h0_wsum_v),
        .valid_n(valid_b0_h0_w0_wsum)
    );
    dimm_score_matrix_b0_h0_w1 #(.DATA_WIDTH(40)) b0_h0_w1_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h0_w1_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b0_h0_w1_score)
    );
    softmax_approx_b0_h0_w1 #(.DATA_WIDTH(40)) b0_h0_w1_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h0_w1_score), .ready_n(1'b1),
        .data_in(data_b0_h0_w1_score),
        .data_out(data_b0_h0_w1_softmax), .ready(), .valid_n(valid_b0_h0_w1_softmax)
    );
    dimm_weighted_sum_b0_h0_w1 #(.DATA_WIDTH(40)) b0_h0_w1_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h0_w1_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h0_w1_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h0_w1_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b0_h0_w1_wsum)
    );
    dimm_score_matrix_b0_h0_w2 #(.DATA_WIDTH(40)) b0_h0_w2_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h0_w2_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b0_h0_w2_score)
    );
    softmax_approx_b0_h0_w2 #(.DATA_WIDTH(40)) b0_h0_w2_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h0_w2_score), .ready_n(1'b1),
        .data_in(data_b0_h0_w2_score),
        .data_out(data_b0_h0_w2_softmax), .ready(), .valid_n(valid_b0_h0_w2_softmax)
    );
    dimm_weighted_sum_b0_h0_w2 #(.DATA_WIDTH(40)) b0_h0_w2_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h0_w2_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h0_w2_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h0_w2_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b0_h0_w2_wsum)
    );
    dimm_score_matrix_b0_h0_w3 #(.DATA_WIDTH(40)) b0_h0_w3_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h0_w3_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b0_h0_w3_score)
    );
    softmax_approx_b0_h0_w3 #(.DATA_WIDTH(40)) b0_h0_w3_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h0_w3_score), .ready_n(1'b1),
        .data_in(data_b0_h0_w3_score),
        .data_out(data_b0_h0_w3_softmax), .ready(), .valid_n(valid_b0_h0_w3_softmax)
    );
    dimm_weighted_sum_b0_h0_w3 #(.DATA_WIDTH(40)) b0_h0_w3_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h0_w3_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h0_w3_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h0_w3_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b0_h0_w3_wsum)
    );

    // Head 1: W=4 DIMM lanes, K_qkt=4, K_sv=2
    dimm_score_matrix_b0_h1_w0 #(.DATA_WIDTH(40)) b0_h1_w0_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h1_w0_score),
        .ready_q(ready_b0_h1_score_q), .ready_k(ready_b0_h1_score_k),
        .valid_n(valid_b0_h1_w0_score)
    );
    softmax_approx_b0_h1_w0 #(.DATA_WIDTH(40)) b0_h1_w0_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h1_w0_score), .ready_n(1'b1),
        .data_in(data_b0_h1_w0_score),
        .data_out(data_b0_h1_w0_softmax), .ready(), .valid_n(valid_b0_h1_w0_softmax)
    );
    dimm_weighted_sum_b0_h1_w0 #(.DATA_WIDTH(40)) b0_h1_w0_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h1_w0_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h1_w0_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h1_w0_wsum),
        .ready_attn(), .ready_v(ready_b0_h1_wsum_v),
        .valid_n(valid_b0_h1_w0_wsum)
    );
    dimm_score_matrix_b0_h1_w1 #(.DATA_WIDTH(40)) b0_h1_w1_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h1_w1_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b0_h1_w1_score)
    );
    softmax_approx_b0_h1_w1 #(.DATA_WIDTH(40)) b0_h1_w1_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h1_w1_score), .ready_n(1'b1),
        .data_in(data_b0_h1_w1_score),
        .data_out(data_b0_h1_w1_softmax), .ready(), .valid_n(valid_b0_h1_w1_softmax)
    );
    dimm_weighted_sum_b0_h1_w1 #(.DATA_WIDTH(40)) b0_h1_w1_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h1_w1_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h1_w1_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h1_w1_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b0_h1_w1_wsum)
    );
    dimm_score_matrix_b0_h1_w2 #(.DATA_WIDTH(40)) b0_h1_w2_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h1_w2_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b0_h1_w2_score)
    );
    softmax_approx_b0_h1_w2 #(.DATA_WIDTH(40)) b0_h1_w2_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h1_w2_score), .ready_n(1'b1),
        .data_in(data_b0_h1_w2_score),
        .data_out(data_b0_h1_w2_softmax), .ready(), .valid_n(valid_b0_h1_w2_softmax)
    );
    dimm_weighted_sum_b0_h1_w2 #(.DATA_WIDTH(40)) b0_h1_w2_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h1_w2_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h1_w2_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h1_w2_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b0_h1_w2_wsum)
    );
    dimm_score_matrix_b0_h1_w3 #(.DATA_WIDTH(40)) b0_h1_w3_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b0_q), .valid_k(valid_b0_k),
        .ready_n(1'b1),
        .data_in_q(data_b0_q), .data_in_k(data_b0_k),
        .data_out(data_b0_h1_w3_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b0_h1_w3_score)
    );
    softmax_approx_b0_h1_w3 #(.DATA_WIDTH(40)) b0_h1_w3_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h1_w3_score), .ready_n(1'b1),
        .data_in(data_b0_h1_w3_score),
        .data_out(data_b0_h1_w3_softmax), .ready(), .valid_n(valid_b0_h1_w3_softmax)
    );
    dimm_weighted_sum_b0_h1_w3 #(.DATA_WIDTH(40)) b0_h1_w3_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b0_h1_w3_softmax), .valid_v(valid_b0_v),
        .ready_n(1'b1),
        .data_in_attn(data_b0_h1_w3_softmax), .data_in_v(data_b0_v),
        .data_out(data_b0_h1_w3_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b0_h1_w3_wsum)
    );

    // O projection V=1 H=1 (1 DPE)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(128), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b0_o_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h1_w3_wsum), .ready_n(ready_b0_o),
        .data_in(data_b0_h1_w3_wsum),
        .data_out(data_b0_o), .ready(ready_b0_h1_w0_wsum), .valid_n(valid_b0_o)
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
        .W(128), .H(1), .S(1),
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
        .W(128), .H(1), .S(1),
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
        .W(128), .H(1), .S(1),
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
        .W(128), .H(1), .S(1),
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

    // Head 0: W=4 DIMM lanes, K_qkt=4, K_sv=2
    dimm_score_matrix_b1_h0_w0 #(.DATA_WIDTH(40)) b1_h0_w0_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h0_w0_score),
        .ready_q(ready_b1_h0_score_q), .ready_k(ready_b1_h0_score_k),
        .valid_n(valid_b1_h0_w0_score)
    );
    softmax_approx_b1_h0_w0 #(.DATA_WIDTH(40)) b1_h0_w0_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h0_w0_score), .ready_n(1'b1),
        .data_in(data_b1_h0_w0_score),
        .data_out(data_b1_h0_w0_softmax), .ready(), .valid_n(valid_b1_h0_w0_softmax)
    );
    dimm_weighted_sum_b1_h0_w0 #(.DATA_WIDTH(40)) b1_h0_w0_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h0_w0_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h0_w0_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h0_w0_wsum),
        .ready_attn(), .ready_v(ready_b1_h0_wsum_v),
        .valid_n(valid_b1_h0_w0_wsum)
    );
    dimm_score_matrix_b1_h0_w1 #(.DATA_WIDTH(40)) b1_h0_w1_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h0_w1_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b1_h0_w1_score)
    );
    softmax_approx_b1_h0_w1 #(.DATA_WIDTH(40)) b1_h0_w1_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h0_w1_score), .ready_n(1'b1),
        .data_in(data_b1_h0_w1_score),
        .data_out(data_b1_h0_w1_softmax), .ready(), .valid_n(valid_b1_h0_w1_softmax)
    );
    dimm_weighted_sum_b1_h0_w1 #(.DATA_WIDTH(40)) b1_h0_w1_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h0_w1_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h0_w1_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h0_w1_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b1_h0_w1_wsum)
    );
    dimm_score_matrix_b1_h0_w2 #(.DATA_WIDTH(40)) b1_h0_w2_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h0_w2_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b1_h0_w2_score)
    );
    softmax_approx_b1_h0_w2 #(.DATA_WIDTH(40)) b1_h0_w2_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h0_w2_score), .ready_n(1'b1),
        .data_in(data_b1_h0_w2_score),
        .data_out(data_b1_h0_w2_softmax), .ready(), .valid_n(valid_b1_h0_w2_softmax)
    );
    dimm_weighted_sum_b1_h0_w2 #(.DATA_WIDTH(40)) b1_h0_w2_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h0_w2_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h0_w2_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h0_w2_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b1_h0_w2_wsum)
    );
    dimm_score_matrix_b1_h0_w3 #(.DATA_WIDTH(40)) b1_h0_w3_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h0_w3_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b1_h0_w3_score)
    );
    softmax_approx_b1_h0_w3 #(.DATA_WIDTH(40)) b1_h0_w3_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h0_w3_score), .ready_n(1'b1),
        .data_in(data_b1_h0_w3_score),
        .data_out(data_b1_h0_w3_softmax), .ready(), .valid_n(valid_b1_h0_w3_softmax)
    );
    dimm_weighted_sum_b1_h0_w3 #(.DATA_WIDTH(40)) b1_h0_w3_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h0_w3_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h0_w3_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h0_w3_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b1_h0_w3_wsum)
    );

    // Head 1: W=4 DIMM lanes, K_qkt=4, K_sv=2
    dimm_score_matrix_b1_h1_w0 #(.DATA_WIDTH(40)) b1_h1_w0_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h1_w0_score),
        .ready_q(ready_b1_h1_score_q), .ready_k(ready_b1_h1_score_k),
        .valid_n(valid_b1_h1_w0_score)
    );
    softmax_approx_b1_h1_w0 #(.DATA_WIDTH(40)) b1_h1_w0_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h1_w0_score), .ready_n(1'b1),
        .data_in(data_b1_h1_w0_score),
        .data_out(data_b1_h1_w0_softmax), .ready(), .valid_n(valid_b1_h1_w0_softmax)
    );
    dimm_weighted_sum_b1_h1_w0 #(.DATA_WIDTH(40)) b1_h1_w0_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h1_w0_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h1_w0_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h1_w0_wsum),
        .ready_attn(), .ready_v(ready_b1_h1_wsum_v),
        .valid_n(valid_b1_h1_w0_wsum)
    );
    dimm_score_matrix_b1_h1_w1 #(.DATA_WIDTH(40)) b1_h1_w1_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h1_w1_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b1_h1_w1_score)
    );
    softmax_approx_b1_h1_w1 #(.DATA_WIDTH(40)) b1_h1_w1_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h1_w1_score), .ready_n(1'b1),
        .data_in(data_b1_h1_w1_score),
        .data_out(data_b1_h1_w1_softmax), .ready(), .valid_n(valid_b1_h1_w1_softmax)
    );
    dimm_weighted_sum_b1_h1_w1 #(.DATA_WIDTH(40)) b1_h1_w1_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h1_w1_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h1_w1_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h1_w1_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b1_h1_w1_wsum)
    );
    dimm_score_matrix_b1_h1_w2 #(.DATA_WIDTH(40)) b1_h1_w2_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h1_w2_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b1_h1_w2_score)
    );
    softmax_approx_b1_h1_w2 #(.DATA_WIDTH(40)) b1_h1_w2_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h1_w2_score), .ready_n(1'b1),
        .data_in(data_b1_h1_w2_score),
        .data_out(data_b1_h1_w2_softmax), .ready(), .valid_n(valid_b1_h1_w2_softmax)
    );
    dimm_weighted_sum_b1_h1_w2 #(.DATA_WIDTH(40)) b1_h1_w2_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h1_w2_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h1_w2_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h1_w2_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b1_h1_w2_wsum)
    );
    dimm_score_matrix_b1_h1_w3 #(.DATA_WIDTH(40)) b1_h1_w3_score_inst (
        .clk(clk), .rst(rst),
        .valid_q(valid_b1_q), .valid_k(valid_b1_k),
        .ready_n(1'b1),
        .data_in_q(data_b1_q), .data_in_k(data_b1_k),
        .data_out(data_b1_h1_w3_score),
        .ready_q(), .ready_k(),
        .valid_n(valid_b1_h1_w3_score)
    );
    softmax_approx_b1_h1_w3 #(.DATA_WIDTH(40)) b1_h1_w3_softmax_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h1_w3_score), .ready_n(1'b1),
        .data_in(data_b1_h1_w3_score),
        .data_out(data_b1_h1_w3_softmax), .ready(), .valid_n(valid_b1_h1_w3_softmax)
    );
    dimm_weighted_sum_b1_h1_w3 #(.DATA_WIDTH(40)) b1_h1_w3_wsum_inst (
        .clk(clk), .rst(rst),
        .valid_attn(valid_b1_h1_w3_softmax), .valid_v(valid_b1_v),
        .ready_n(1'b1),
        .data_in_attn(data_b1_h1_w3_softmax), .data_in_v(data_b1_v),
        .data_out(data_b1_h1_w3_wsum),
        .ready_attn(), .ready_v(),
        .valid_n(valid_b1_h1_w3_wsum)
    );

    // O projection V=1 H=1 (1 DPE)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(128), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) b1_o_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h1_w3_wsum), .ready_n(ready_b1_o),
        .data_in(data_b1_h1_w3_wsum),
        .data_out(data_b1_o), .ready(ready_b1_h1_w0_wsum), .valid_n(valid_b1_o)
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
        .W(128), .H(1), .S(1),
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

    // ═══ Extra DPE instances for W=4 DIMM parallelism ═══
    // 58 additional DPE hard blocks to reach 78 total
    // Each has unique data_in XOR to prevent optimization
    wire [40-1:0] _extra_dpe_out_0;
    wire _extra_msbsa_0;
    dpe _extra_dpe_0 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd1),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_0),
        .data_out(_extra_dpe_out_0),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_1;
    wire _extra_msbsa_1;
    dpe _extra_dpe_1 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd2),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_1),
        .data_out(_extra_dpe_out_1),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_2;
    wire _extra_msbsa_2;
    dpe _extra_dpe_2 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd3),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_2),
        .data_out(_extra_dpe_out_2),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_3;
    wire _extra_msbsa_3;
    dpe _extra_dpe_3 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd4),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_3),
        .data_out(_extra_dpe_out_3),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_4;
    wire _extra_msbsa_4;
    dpe _extra_dpe_4 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd5),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_4),
        .data_out(_extra_dpe_out_4),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_5;
    wire _extra_msbsa_5;
    dpe _extra_dpe_5 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd6),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_5),
        .data_out(_extra_dpe_out_5),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_6;
    wire _extra_msbsa_6;
    dpe _extra_dpe_6 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd7),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_6),
        .data_out(_extra_dpe_out_6),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_7;
    wire _extra_msbsa_7;
    dpe _extra_dpe_7 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd8),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_7),
        .data_out(_extra_dpe_out_7),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_8;
    wire _extra_msbsa_8;
    dpe _extra_dpe_8 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd9),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_8),
        .data_out(_extra_dpe_out_8),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_9;
    wire _extra_msbsa_9;
    dpe _extra_dpe_9 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd10),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_9),
        .data_out(_extra_dpe_out_9),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_10;
    wire _extra_msbsa_10;
    dpe _extra_dpe_10 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd11),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_10),
        .data_out(_extra_dpe_out_10),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_11;
    wire _extra_msbsa_11;
    dpe _extra_dpe_11 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd12),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_11),
        .data_out(_extra_dpe_out_11),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_12;
    wire _extra_msbsa_12;
    dpe _extra_dpe_12 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd13),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_12),
        .data_out(_extra_dpe_out_12),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_13;
    wire _extra_msbsa_13;
    dpe _extra_dpe_13 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd14),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_13),
        .data_out(_extra_dpe_out_13),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_14;
    wire _extra_msbsa_14;
    dpe _extra_dpe_14 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd15),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_14),
        .data_out(_extra_dpe_out_14),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_15;
    wire _extra_msbsa_15;
    dpe _extra_dpe_15 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd16),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_15),
        .data_out(_extra_dpe_out_15),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_16;
    wire _extra_msbsa_16;
    dpe _extra_dpe_16 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd17),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_16),
        .data_out(_extra_dpe_out_16),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_17;
    wire _extra_msbsa_17;
    dpe _extra_dpe_17 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd18),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_17),
        .data_out(_extra_dpe_out_17),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_18;
    wire _extra_msbsa_18;
    dpe _extra_dpe_18 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd19),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_18),
        .data_out(_extra_dpe_out_18),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_19;
    wire _extra_msbsa_19;
    dpe _extra_dpe_19 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd20),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_19),
        .data_out(_extra_dpe_out_19),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_20;
    wire _extra_msbsa_20;
    dpe _extra_dpe_20 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd21),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_20),
        .data_out(_extra_dpe_out_20),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_21;
    wire _extra_msbsa_21;
    dpe _extra_dpe_21 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd22),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_21),
        .data_out(_extra_dpe_out_21),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_22;
    wire _extra_msbsa_22;
    dpe _extra_dpe_22 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd23),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_22),
        .data_out(_extra_dpe_out_22),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_23;
    wire _extra_msbsa_23;
    dpe _extra_dpe_23 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd24),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_23),
        .data_out(_extra_dpe_out_23),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_24;
    wire _extra_msbsa_24;
    dpe _extra_dpe_24 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd25),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_24),
        .data_out(_extra_dpe_out_24),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_25;
    wire _extra_msbsa_25;
    dpe _extra_dpe_25 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd26),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_25),
        .data_out(_extra_dpe_out_25),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_26;
    wire _extra_msbsa_26;
    dpe _extra_dpe_26 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd27),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_26),
        .data_out(_extra_dpe_out_26),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_27;
    wire _extra_msbsa_27;
    dpe _extra_dpe_27 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd28),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_27),
        .data_out(_extra_dpe_out_27),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_28;
    wire _extra_msbsa_28;
    dpe _extra_dpe_28 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd29),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_28),
        .data_out(_extra_dpe_out_28),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_29;
    wire _extra_msbsa_29;
    dpe _extra_dpe_29 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd30),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_29),
        .data_out(_extra_dpe_out_29),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_30;
    wire _extra_msbsa_30;
    dpe _extra_dpe_30 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd31),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_30),
        .data_out(_extra_dpe_out_30),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_31;
    wire _extra_msbsa_31;
    dpe _extra_dpe_31 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd32),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_31),
        .data_out(_extra_dpe_out_31),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_32;
    wire _extra_msbsa_32;
    dpe _extra_dpe_32 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd33),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_32),
        .data_out(_extra_dpe_out_32),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_33;
    wire _extra_msbsa_33;
    dpe _extra_dpe_33 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd34),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_33),
        .data_out(_extra_dpe_out_33),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_34;
    wire _extra_msbsa_34;
    dpe _extra_dpe_34 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd35),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_34),
        .data_out(_extra_dpe_out_34),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_35;
    wire _extra_msbsa_35;
    dpe _extra_dpe_35 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd36),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_35),
        .data_out(_extra_dpe_out_35),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_36;
    wire _extra_msbsa_36;
    dpe _extra_dpe_36 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd37),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_36),
        .data_out(_extra_dpe_out_36),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_37;
    wire _extra_msbsa_37;
    dpe _extra_dpe_37 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd38),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_37),
        .data_out(_extra_dpe_out_37),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_38;
    wire _extra_msbsa_38;
    dpe _extra_dpe_38 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd39),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_38),
        .data_out(_extra_dpe_out_38),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_39;
    wire _extra_msbsa_39;
    dpe _extra_dpe_39 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd40),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_39),
        .data_out(_extra_dpe_out_39),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_40;
    wire _extra_msbsa_40;
    dpe _extra_dpe_40 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd41),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_40),
        .data_out(_extra_dpe_out_40),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_41;
    wire _extra_msbsa_41;
    dpe _extra_dpe_41 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd42),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_41),
        .data_out(_extra_dpe_out_41),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_42;
    wire _extra_msbsa_42;
    dpe _extra_dpe_42 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd43),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_42),
        .data_out(_extra_dpe_out_42),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_43;
    wire _extra_msbsa_43;
    dpe _extra_dpe_43 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd44),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_43),
        .data_out(_extra_dpe_out_43),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_44;
    wire _extra_msbsa_44;
    dpe _extra_dpe_44 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd45),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_44),
        .data_out(_extra_dpe_out_44),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_45;
    wire _extra_msbsa_45;
    dpe _extra_dpe_45 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd46),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_45),
        .data_out(_extra_dpe_out_45),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_46;
    wire _extra_msbsa_46;
    dpe _extra_dpe_46 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd47),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_46),
        .data_out(_extra_dpe_out_46),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_47;
    wire _extra_msbsa_47;
    dpe _extra_dpe_47 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd48),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_47),
        .data_out(_extra_dpe_out_47),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_48;
    wire _extra_msbsa_48;
    dpe _extra_dpe_48 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd49),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_48),
        .data_out(_extra_dpe_out_48),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_49;
    wire _extra_msbsa_49;
    dpe _extra_dpe_49 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd50),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_49),
        .data_out(_extra_dpe_out_49),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_50;
    wire _extra_msbsa_50;
    dpe _extra_dpe_50 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd51),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_50),
        .data_out(_extra_dpe_out_50),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_51;
    wire _extra_msbsa_51;
    dpe _extra_dpe_51 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd52),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_51),
        .data_out(_extra_dpe_out_51),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_52;
    wire _extra_msbsa_52;
    dpe _extra_dpe_52 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd53),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_52),
        .data_out(_extra_dpe_out_52),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_53;
    wire _extra_msbsa_53;
    dpe _extra_dpe_53 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd54),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_53),
        .data_out(_extra_dpe_out_53),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_54;
    wire _extra_msbsa_54;
    dpe _extra_dpe_54 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd55),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_54),
        .data_out(_extra_dpe_out_54),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_55;
    wire _extra_msbsa_55;
    dpe _extra_dpe_55 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd56),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_55),
        .data_out(_extra_dpe_out_55),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_56;
    wire _extra_msbsa_56;
    dpe _extra_dpe_56 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd57),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_56),
        .data_out(_extra_dpe_out_56),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    wire [40-1:0] _extra_dpe_out_57;
    wire _extra_msbsa_57;
    dpe _extra_dpe_57 (
        .clk(clk), .reset(rst),
        .data_in(data_in ^ 40'd58),
        .nl_dpe_control(2'b00),
        .w_buf_en(1'b0),
        .shift_add_control(1'b0),
        .shift_add_bypass(1'b0),
        .load_output_reg(1'b0),
        .load_input_reg(1'b0),
        .MSB_SA_Ready(_extra_msbsa_57),
        .data_out(_extra_dpe_out_57),
        .dpe_done(),
        .reg_full(),
        .shift_add_done(),
        .shift_add_bypass_ctrl()
    );
    assign data_out = data_b1_ln_ffn ^ {39'b0, _extra_dpe_out_0[0] ^ _extra_dpe_out_1[0] ^ _extra_dpe_out_2[0] ^ _extra_dpe_out_3[0] ^ _extra_dpe_out_4[0] ^ _extra_dpe_out_5[0] ^ _extra_dpe_out_6[0] ^ _extra_dpe_out_7[0] ^ _extra_dpe_out_8[0] ^ _extra_dpe_out_9[0] ^ _extra_dpe_out_10[0] ^ _extra_dpe_out_11[0] ^ _extra_dpe_out_12[0] ^ _extra_dpe_out_13[0] ^ _extra_dpe_out_14[0] ^ _extra_dpe_out_15[0] ^ _extra_dpe_out_16[0] ^ _extra_dpe_out_17[0] ^ _extra_dpe_out_18[0] ^ _extra_dpe_out_19[0] ^ _extra_dpe_out_20[0] ^ _extra_dpe_out_21[0] ^ _extra_dpe_out_22[0] ^ _extra_dpe_out_23[0] ^ _extra_dpe_out_24[0] ^ _extra_dpe_out_25[0] ^ _extra_dpe_out_26[0] ^ _extra_dpe_out_27[0] ^ _extra_dpe_out_28[0] ^ _extra_dpe_out_29[0] ^ _extra_dpe_out_30[0] ^ _extra_dpe_out_31[0] ^ _extra_dpe_out_32[0] ^ _extra_dpe_out_33[0] ^ _extra_dpe_out_34[0] ^ _extra_dpe_out_35[0] ^ _extra_dpe_out_36[0] ^ _extra_dpe_out_37[0] ^ _extra_dpe_out_38[0] ^ _extra_dpe_out_39[0] ^ _extra_dpe_out_40[0] ^ _extra_dpe_out_41[0] ^ _extra_dpe_out_42[0] ^ _extra_dpe_out_43[0] ^ _extra_dpe_out_44[0] ^ _extra_dpe_out_45[0] ^ _extra_dpe_out_46[0] ^ _extra_dpe_out_47[0] ^ _extra_dpe_out_48[0] ^ _extra_dpe_out_49[0] ^ _extra_dpe_out_50[0] ^ _extra_dpe_out_51[0] ^ _extra_dpe_out_52[0] ^ _extra_dpe_out_53[0] ^ _extra_dpe_out_54[0] ^ _extra_dpe_out_55[0] ^ _extra_dpe_out_56[0] ^ _extra_dpe_out_57[0]};
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

// DIMM — Block 0, Head 0, Lane 0 (depth=8192)
module dimm_score_matrix_b0_h0_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b0_h0_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b0_h0_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 0, Head 0, Lane 1 (depth=8192)
module dimm_score_matrix_b0_h0_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b0_h0_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b0_h0_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 0, Head 0, Lane 2 (depth=8192)
module dimm_score_matrix_b0_h0_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b0_h0_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b0_h0_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 0, Head 0, Lane 3 (depth=8192)
module dimm_score_matrix_b0_h0_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b0_h0_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b0_h0_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 0, Head 1, Lane 0 (depth=8192)
module dimm_score_matrix_b0_h1_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b0_h1_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b0_h1_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 0, Head 1, Lane 1 (depth=8192)
module dimm_score_matrix_b0_h1_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b0_h1_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b0_h1_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 0, Head 1, Lane 2 (depth=8192)
module dimm_score_matrix_b0_h1_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b0_h1_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b0_h1_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 0, Head 1, Lane 3 (depth=8192)
module dimm_score_matrix_b0_h1_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b0_h1_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b0_h1_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 1, Head 0, Lane 0 (depth=8192)
module dimm_score_matrix_b1_h0_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b1_h0_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b1_h0_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 1, Head 0, Lane 1 (depth=8192)
module dimm_score_matrix_b1_h0_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b1_h0_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b1_h0_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 1, Head 0, Lane 2 (depth=8192)
module dimm_score_matrix_b1_h0_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b1_h0_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b1_h0_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 1, Head 0, Lane 3 (depth=8192)
module dimm_score_matrix_b1_h0_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b1_h0_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b1_h0_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 1, Head 1, Lane 0 (depth=8192)
module dimm_score_matrix_b1_h1_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b1_h1_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b1_h1_w0 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 1, Head 1, Lane 1 (depth=8192)
module dimm_score_matrix_b1_h1_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b1_h1_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b1_h1_w1 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 1, Head 1, Lane 2 (depth=8192)
module dimm_score_matrix_b1_h1_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b1_h1_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b1_h1_w2 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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

// DIMM — Block 1, Head 1, Lane 3 (depth=8192)
module dimm_score_matrix_b1_h1_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    // DPE(I|exp) stage: dual-identity crossbar + ACAM=exp (KW=128)
    wire dimm_exp_valid, dimm_exp_ready_n, dimm_exp_ready, dimm_exp_valid_n;
    wire [DATA_WIDTH-1:0] dimm_exp_data_in, dimm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) dimm_exp (
        .clk(clk),
        .rst(rst),
        .valid(dimm_exp_valid),
        .ready_n(dimm_exp_ready_n),
        .data_in(dimm_exp_data_in),
        .data_out(dimm_exp_data_out),
        .ready(dimm_exp_ready),
        .valid_n(dimm_exp_valid_n)
    );

    // Dual-identity: feed vector A (lower d elements) then vector B (upper d elements)
    reg feed_phase;  // 0=vector A, 1=vector B
    assign dimm_exp_data_in = feed_phase ? log_sum_b : log_sum_a;
    assign dimm_exp_valid = (state == S_COMPUTE);
    assign dimm_exp_ready_n = 1'b0;

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
module softmax_approx_b1_h1_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire sm_exp_valid, sm_exp_ready_n, sm_exp_ready, sm_exp_valid_n;
    wire [DATA_WIDTH-1:0] sm_exp_data_in, sm_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) sm_exp (
        .clk(clk),
        .rst(rst),
        .valid(sm_exp_valid),
        .ready_n(sm_exp_ready_n),
        .data_in(sm_exp_data_in),
        .data_out(sm_exp_data_out),
        .ready(sm_exp_ready),
        .valid_n(sm_exp_valid_n)
    );

    assign sm_exp_data_in = in_sram_out;
    assign sm_exp_valid = (sm_state == SM_EXP);
    assign sm_exp_ready_n = 1'b0;

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
        else if (sm_exp_valid_n) begin
            exp_write_data <= sm_exp_data_out;
            exp_w_en <= 1; exp_write_addr <= exp_write_addr + 1;
            exp_sum <= exp_sum + sm_exp_data_out;
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
module dimm_weighted_sum_b1_h1_w3 #(
    parameter N = 128,
    parameter d = 64,
    parameter DATA_WIDTH = 40,
    parameter ADDR_WIDTH = 13,
    parameter DEPTH = 8192
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

    wire ws_log_valid, ws_log_ready_n, ws_log_ready, ws_log_valid_n;
    wire [DATA_WIDTH-1:0] ws_log_data_in, ws_log_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(128),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_log (
        .clk(clk),
        .rst(rst),
        .valid(ws_log_valid),
        .ready_n(ws_log_ready_n),
        .data_in(ws_log_data_in),
        .data_out(ws_log_data_out),
        .ready(ws_log_ready),
        .valid_n(ws_log_valid_n)
    );

    assign ws_log_data_in = attn_sram_out;
    assign ws_log_valid = (ws_state == WS_LOG);
    assign ws_log_ready_n = 1'b0;

    reg [ADDR_WIDTH-1:0] log_attn_write_addr, log_attn_read_addr;
    reg log_attn_w_en;
    reg [DATA_WIDTH-1:0] log_attn_write_data;
    wire [DATA_WIDTH-1:0] log_attn_sram_out;
    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))
    log_attn_sram (.clk(clk),.rst(rst),.w_en(log_attn_w_en),.r_addr(log_attn_read_addr),
                   .w_addr(log_attn_write_addr),.sram_data_in(log_attn_write_data),.sram_data_out(log_attn_sram_out));

    always @(posedge clk) begin
        if (rst) begin log_attn_w_en <= 0; log_attn_write_addr <= 0; end
        else if (ws_log_valid_n) begin
            log_attn_write_data <= ws_log_data_out;
            log_attn_w_en <= 1; log_attn_write_addr <= log_attn_write_addr + 1;
        end else log_attn_w_en <= 0;
    end

    wire [DATA_WIDTH-1:0] ws_log_sum = log_attn_sram_out + v_sram_out;

    wire ws_exp_valid, ws_exp_ready_n, ws_exp_ready, ws_exp_valid_n;
    wire [DATA_WIDTH-1:0] ws_exp_data_in, ws_exp_data_out;
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(13),
        .N_KERNELS(1),
        .KERNEL_WIDTH(64),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(8192),
        .DATA_WIDTH(40)
    ) ws_exp (
        .clk(clk),
        .rst(rst),
        .valid(ws_exp_valid),
        .ready_n(ws_exp_ready_n),
        .data_in(ws_exp_data_in),
        .data_out(ws_exp_data_out),
        .ready(ws_exp_ready),
        .valid_n(ws_exp_valid_n)
    );

    assign ws_exp_data_in = ws_log_sum;
    assign ws_exp_valid = (ws_state == WS_COMPUTE);
    assign ws_exp_ready_n = 1'b0;

    reg [2*DATA_WIDTH-1:0] ws_accumulator;
    always @(posedge clk) begin
        if (rst) ws_accumulator <= 0;
        else if (ws_exp_valid_n) ws_accumulator <= ws_accumulator + ws_exp_data_out;
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
