// Auto-generated BERT-Tiny RTL — Baseline (no DPE, DSP-only)
// Total DPEs: 0
// All GEMMs use DSP MAC (Verilog * operator)
// DIMM: DSP MAC time-shared

module bert_tiny_baseline (
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
    wire [40-1:0] data_b0_h0_qk;
    wire valid_b0_h0_qk, ready_b0_h0_qk;
    wire [40-1:0] data_b0_h0_softmax;
    wire valid_b0_h0_softmax, ready_b0_h0_softmax;
    wire [40-1:0] data_b0_h0_sv;
    wire valid_b0_h0_sv, ready_b0_h0_sv;
    wire [40-1:0] data_b0_h1_qk;
    wire valid_b0_h1_qk, ready_b0_h1_qk;
    wire [40-1:0] data_b0_h1_softmax;
    wire valid_b0_h1_softmax, ready_b0_h1_softmax;
    wire [40-1:0] data_b0_h1_sv;
    wire valid_b0_h1_sv, ready_b0_h1_sv;
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
    wire [40-1:0] data_b1_h0_qk;
    wire valid_b1_h0_qk, ready_b1_h0_qk;
    wire [40-1:0] data_b1_h0_softmax;
    wire valid_b1_h0_softmax, ready_b1_h0_softmax;
    wire [40-1:0] data_b1_h0_sv;
    wire valid_b1_h0_sv, ready_b1_h0_sv;
    wire [40-1:0] data_b1_h1_qk;
    wire valid_b1_h1_qk, ready_b1_h1_qk;
    wire [40-1:0] data_b1_h1_softmax;
    wire valid_b1_h1_softmax, ready_b1_h1_softmax;
    wire [40-1:0] data_b1_h1_sv;
    wire valid_b1_h1_sv, ready_b1_h1_sv;
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
    // b0_q: projection (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b0_q_inst (
        .clk(clk), .rst(rst), .valid(valid_ln_embed), .ready_n(ready_b0_q),
        .data_a(data_ln_embed), .data_b(data_ln_embed),
        .data_out(data_b0_q), .ready(ready_ln_embed), .valid_n(valid_b0_q)
    );

    // b0_k: projection (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b0_k_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_q), .ready_n(ready_b0_k),
        .data_a(data_b0_q), .data_b(data_b0_q),
        .data_out(data_b0_k), .ready(ready_b0_j), .valid_n(valid_b0_k)
    );

    // b0_v: projection (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b0_v_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_k), .ready_n(ready_b0_v),
        .data_a(data_b0_k), .data_b(data_b0_k),
        .data_out(data_b0_v), .ready(ready_b0_u), .valid_n(valid_b0_v)
    );

    // Head 0: QK^T (DSP MAC)
    dsp_mac #(.DATA_WIDTH(40), .K(64)) b0_h0_qk_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(ready_b0_h0_qk),
        .data_a(data_b0_q), .data_b(data_b0_k),
        .data_out(data_b0_h0_qk), .ready(ready_b0_v), .valid_n(valid_b0_h0_qk)
    );
    // Head 0: Softmax (CLB)
    clb_softmax #(.DATA_WIDTH(40), .N(128)) b0_h0_softmax_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_qk), .ready_n(ready_b0_h0_softmax),
        .data_in(data_b0_h0_qk),
        .data_out(data_b0_h0_softmax), .ready(ready_b0_h0_qk), .valid_n(valid_b0_h0_softmax)
    );
    // Head 0: Score×V (DSP MAC)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b0_h0_sv_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_softmax), .ready_n(ready_b0_h0_sv),
        .data_a(data_b0_h0_softmax), .data_b(data_b0_v),
        .data_out(data_b0_h0_sv), .ready(ready_b0_h0_softmax), .valid_n(valid_b0_h0_sv)
    );

    // Head 1: QK^T (DSP MAC)
    dsp_mac #(.DATA_WIDTH(40), .K(64)) b0_h1_qk_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(ready_b0_h1_qk),
        .data_a(data_b0_q), .data_b(data_b0_k),
        .data_out(data_b0_h1_qk), .ready(ready_b0_v), .valid_n(valid_b0_h1_qk)
    );
    // Head 1: Softmax (CLB)
    clb_softmax #(.DATA_WIDTH(40), .N(128)) b0_h1_softmax_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_qk), .ready_n(ready_b0_h1_softmax),
        .data_in(data_b0_h1_qk),
        .data_out(data_b0_h1_softmax), .ready(ready_b0_h1_qk), .valid_n(valid_b0_h1_softmax)
    );
    // Head 1: Score×V (DSP MAC)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b0_h1_sv_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_softmax), .ready_n(ready_b0_h1_sv),
        .data_a(data_b0_h1_softmax), .data_b(data_b0_v),
        .data_out(data_b0_h1_sv), .ready(ready_b0_h1_softmax), .valid_n(valid_b0_h1_sv)
    );

    // O projection (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b0_o_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_sv), .ready_n(ready_b0_o),
        .data_a(data_b0_h1_sv), .data_b(data_b0_h1_sv),
        .data_out(data_b0_o), .ready(ready_b0_h1_sv), .valid_n(valid_b0_o)
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

    // FFN1 (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b0_ffn1_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_ln_attn), .ready_n(ready_b0_ffn1),
        .data_a(data_b0_ln_attn), .data_b(data_b0_ln_attn),
        .data_out(data_b0_ffn1), .ready(ready_b0_ln_attn), .valid_n(valid_b0_ffn1)
    );

    // FFN2 (DSP MAC, K=512)
    dsp_mac #(.DATA_WIDTH(40), .K(512)) b0_ffn2_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_ffn1), .ready_n(ready_b0_ffn2),
        .data_a(data_b0_ffn1), .data_b(data_b0_ffn1),
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
    // b1_q: projection (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b1_q_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_ln_ffn), .ready_n(ready_b1_q),
        .data_a(data_b0_ln_ffn), .data_b(data_b0_ln_ffn),
        .data_out(data_b1_q), .ready(ready_ln_ffn), .valid_n(valid_b1_q)
    );

    // b1_k: projection (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b1_k_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_q), .ready_n(ready_b1_k),
        .data_a(data_b1_q), .data_b(data_b1_q),
        .data_out(data_b1_k), .ready(ready_b1_j), .valid_n(valid_b1_k)
    );

    // b1_v: projection (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b1_v_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_k), .ready_n(ready_b1_v),
        .data_a(data_b1_k), .data_b(data_b1_k),
        .data_out(data_b1_v), .ready(ready_b1_u), .valid_n(valid_b1_v)
    );

    // Head 0: QK^T (DSP MAC)
    dsp_mac #(.DATA_WIDTH(40), .K(64)) b1_h0_qk_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(ready_b1_h0_qk),
        .data_a(data_b1_q), .data_b(data_b1_k),
        .data_out(data_b1_h0_qk), .ready(ready_b1_v), .valid_n(valid_b1_h0_qk)
    );
    // Head 0: Softmax (CLB)
    clb_softmax #(.DATA_WIDTH(40), .N(128)) b1_h0_softmax_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_qk), .ready_n(ready_b1_h0_softmax),
        .data_in(data_b1_h0_qk),
        .data_out(data_b1_h0_softmax), .ready(ready_b1_h0_qk), .valid_n(valid_b1_h0_softmax)
    );
    // Head 0: Score×V (DSP MAC)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b1_h0_sv_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_softmax), .ready_n(ready_b1_h0_sv),
        .data_a(data_b1_h0_softmax), .data_b(data_b1_v),
        .data_out(data_b1_h0_sv), .ready(ready_b1_h0_softmax), .valid_n(valid_b1_h0_sv)
    );

    // Head 1: QK^T (DSP MAC)
    dsp_mac #(.DATA_WIDTH(40), .K(64)) b1_h1_qk_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(ready_b1_h1_qk),
        .data_a(data_b1_q), .data_b(data_b1_k),
        .data_out(data_b1_h1_qk), .ready(ready_b1_v), .valid_n(valid_b1_h1_qk)
    );
    // Head 1: Softmax (CLB)
    clb_softmax #(.DATA_WIDTH(40), .N(128)) b1_h1_softmax_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_qk), .ready_n(ready_b1_h1_softmax),
        .data_in(data_b1_h1_qk),
        .data_out(data_b1_h1_softmax), .ready(ready_b1_h1_qk), .valid_n(valid_b1_h1_softmax)
    );
    // Head 1: Score×V (DSP MAC)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b1_h1_sv_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_softmax), .ready_n(ready_b1_h1_sv),
        .data_a(data_b1_h1_softmax), .data_b(data_b1_v),
        .data_out(data_b1_h1_sv), .ready(ready_b1_h1_softmax), .valid_n(valid_b1_h1_sv)
    );

    // O projection (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b1_o_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_sv), .ready_n(ready_b1_o),
        .data_a(data_b1_h1_sv), .data_b(data_b1_h1_sv),
        .data_out(data_b1_o), .ready(ready_b1_h1_sv), .valid_n(valid_b1_o)
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

    // FFN1 (DSP MAC, K=128)
    dsp_mac #(.DATA_WIDTH(40), .K(128)) b1_ffn1_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_ln_attn), .ready_n(ready_b1_ffn1),
        .data_a(data_b1_ln_attn), .data_b(data_b1_ln_attn),
        .data_out(data_b1_ffn1), .ready(ready_b1_ln_attn), .valid_n(valid_b1_ffn1)
    );

    // FFN2 (DSP MAC, K=512)
    dsp_mac #(.DATA_WIDTH(40), .K(512)) b1_ffn2_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_ffn1), .ready_n(ready_b1_ffn2),
        .data_a(data_b1_ffn1), .data_b(data_b1_ffn1),
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

    // ═══ Extra DSP MACs for full baseline utilization ═══
    // 333 extra DSP MACs to reach 333 total
    wire [40-1:0] _extra_dsp_out_0;
    reg [40-1:0] _extra_dsp_reg_0;
    assign _extra_dsp_out_0 = $signed(data_in ^ 40'd1) * $signed(data_in ^ 40'd2);
    always @(posedge clk) _extra_dsp_reg_0 <= _extra_dsp_out_0;
    wire [40-1:0] _extra_dsp_out_1;
    reg [40-1:0] _extra_dsp_reg_1;
    assign _extra_dsp_out_1 = $signed(data_in ^ 40'd2) * $signed(data_in ^ 40'd3);
    always @(posedge clk) _extra_dsp_reg_1 <= _extra_dsp_out_1;
    wire [40-1:0] _extra_dsp_out_2;
    reg [40-1:0] _extra_dsp_reg_2;
    assign _extra_dsp_out_2 = $signed(data_in ^ 40'd3) * $signed(data_in ^ 40'd4);
    always @(posedge clk) _extra_dsp_reg_2 <= _extra_dsp_out_2;
    wire [40-1:0] _extra_dsp_out_3;
    reg [40-1:0] _extra_dsp_reg_3;
    assign _extra_dsp_out_3 = $signed(data_in ^ 40'd4) * $signed(data_in ^ 40'd5);
    always @(posedge clk) _extra_dsp_reg_3 <= _extra_dsp_out_3;
    wire [40-1:0] _extra_dsp_out_4;
    reg [40-1:0] _extra_dsp_reg_4;
    assign _extra_dsp_out_4 = $signed(data_in ^ 40'd5) * $signed(data_in ^ 40'd6);
    always @(posedge clk) _extra_dsp_reg_4 <= _extra_dsp_out_4;
    wire [40-1:0] _extra_dsp_out_5;
    reg [40-1:0] _extra_dsp_reg_5;
    assign _extra_dsp_out_5 = $signed(data_in ^ 40'd6) * $signed(data_in ^ 40'd7);
    always @(posedge clk) _extra_dsp_reg_5 <= _extra_dsp_out_5;
    wire [40-1:0] _extra_dsp_out_6;
    reg [40-1:0] _extra_dsp_reg_6;
    assign _extra_dsp_out_6 = $signed(data_in ^ 40'd7) * $signed(data_in ^ 40'd8);
    always @(posedge clk) _extra_dsp_reg_6 <= _extra_dsp_out_6;
    wire [40-1:0] _extra_dsp_out_7;
    reg [40-1:0] _extra_dsp_reg_7;
    assign _extra_dsp_out_7 = $signed(data_in ^ 40'd8) * $signed(data_in ^ 40'd9);
    always @(posedge clk) _extra_dsp_reg_7 <= _extra_dsp_out_7;
    wire [40-1:0] _extra_dsp_out_8;
    reg [40-1:0] _extra_dsp_reg_8;
    assign _extra_dsp_out_8 = $signed(data_in ^ 40'd9) * $signed(data_in ^ 40'd10);
    always @(posedge clk) _extra_dsp_reg_8 <= _extra_dsp_out_8;
    wire [40-1:0] _extra_dsp_out_9;
    reg [40-1:0] _extra_dsp_reg_9;
    assign _extra_dsp_out_9 = $signed(data_in ^ 40'd10) * $signed(data_in ^ 40'd11);
    always @(posedge clk) _extra_dsp_reg_9 <= _extra_dsp_out_9;
    wire [40-1:0] _extra_dsp_out_10;
    reg [40-1:0] _extra_dsp_reg_10;
    assign _extra_dsp_out_10 = $signed(data_in ^ 40'd11) * $signed(data_in ^ 40'd12);
    always @(posedge clk) _extra_dsp_reg_10 <= _extra_dsp_out_10;
    wire [40-1:0] _extra_dsp_out_11;
    reg [40-1:0] _extra_dsp_reg_11;
    assign _extra_dsp_out_11 = $signed(data_in ^ 40'd12) * $signed(data_in ^ 40'd13);
    always @(posedge clk) _extra_dsp_reg_11 <= _extra_dsp_out_11;
    wire [40-1:0] _extra_dsp_out_12;
    reg [40-1:0] _extra_dsp_reg_12;
    assign _extra_dsp_out_12 = $signed(data_in ^ 40'd13) * $signed(data_in ^ 40'd14);
    always @(posedge clk) _extra_dsp_reg_12 <= _extra_dsp_out_12;
    wire [40-1:0] _extra_dsp_out_13;
    reg [40-1:0] _extra_dsp_reg_13;
    assign _extra_dsp_out_13 = $signed(data_in ^ 40'd14) * $signed(data_in ^ 40'd15);
    always @(posedge clk) _extra_dsp_reg_13 <= _extra_dsp_out_13;
    wire [40-1:0] _extra_dsp_out_14;
    reg [40-1:0] _extra_dsp_reg_14;
    assign _extra_dsp_out_14 = $signed(data_in ^ 40'd15) * $signed(data_in ^ 40'd16);
    always @(posedge clk) _extra_dsp_reg_14 <= _extra_dsp_out_14;
    wire [40-1:0] _extra_dsp_out_15;
    reg [40-1:0] _extra_dsp_reg_15;
    assign _extra_dsp_out_15 = $signed(data_in ^ 40'd16) * $signed(data_in ^ 40'd17);
    always @(posedge clk) _extra_dsp_reg_15 <= _extra_dsp_out_15;
    wire [40-1:0] _extra_dsp_out_16;
    reg [40-1:0] _extra_dsp_reg_16;
    assign _extra_dsp_out_16 = $signed(data_in ^ 40'd17) * $signed(data_in ^ 40'd18);
    always @(posedge clk) _extra_dsp_reg_16 <= _extra_dsp_out_16;
    wire [40-1:0] _extra_dsp_out_17;
    reg [40-1:0] _extra_dsp_reg_17;
    assign _extra_dsp_out_17 = $signed(data_in ^ 40'd18) * $signed(data_in ^ 40'd19);
    always @(posedge clk) _extra_dsp_reg_17 <= _extra_dsp_out_17;
    wire [40-1:0] _extra_dsp_out_18;
    reg [40-1:0] _extra_dsp_reg_18;
    assign _extra_dsp_out_18 = $signed(data_in ^ 40'd19) * $signed(data_in ^ 40'd20);
    always @(posedge clk) _extra_dsp_reg_18 <= _extra_dsp_out_18;
    wire [40-1:0] _extra_dsp_out_19;
    reg [40-1:0] _extra_dsp_reg_19;
    assign _extra_dsp_out_19 = $signed(data_in ^ 40'd20) * $signed(data_in ^ 40'd21);
    always @(posedge clk) _extra_dsp_reg_19 <= _extra_dsp_out_19;
    wire [40-1:0] _extra_dsp_out_20;
    reg [40-1:0] _extra_dsp_reg_20;
    assign _extra_dsp_out_20 = $signed(data_in ^ 40'd21) * $signed(data_in ^ 40'd22);
    always @(posedge clk) _extra_dsp_reg_20 <= _extra_dsp_out_20;
    wire [40-1:0] _extra_dsp_out_21;
    reg [40-1:0] _extra_dsp_reg_21;
    assign _extra_dsp_out_21 = $signed(data_in ^ 40'd22) * $signed(data_in ^ 40'd23);
    always @(posedge clk) _extra_dsp_reg_21 <= _extra_dsp_out_21;
    wire [40-1:0] _extra_dsp_out_22;
    reg [40-1:0] _extra_dsp_reg_22;
    assign _extra_dsp_out_22 = $signed(data_in ^ 40'd23) * $signed(data_in ^ 40'd24);
    always @(posedge clk) _extra_dsp_reg_22 <= _extra_dsp_out_22;
    wire [40-1:0] _extra_dsp_out_23;
    reg [40-1:0] _extra_dsp_reg_23;
    assign _extra_dsp_out_23 = $signed(data_in ^ 40'd24) * $signed(data_in ^ 40'd25);
    always @(posedge clk) _extra_dsp_reg_23 <= _extra_dsp_out_23;
    wire [40-1:0] _extra_dsp_out_24;
    reg [40-1:0] _extra_dsp_reg_24;
    assign _extra_dsp_out_24 = $signed(data_in ^ 40'd25) * $signed(data_in ^ 40'd26);
    always @(posedge clk) _extra_dsp_reg_24 <= _extra_dsp_out_24;
    wire [40-1:0] _extra_dsp_out_25;
    reg [40-1:0] _extra_dsp_reg_25;
    assign _extra_dsp_out_25 = $signed(data_in ^ 40'd26) * $signed(data_in ^ 40'd27);
    always @(posedge clk) _extra_dsp_reg_25 <= _extra_dsp_out_25;
    wire [40-1:0] _extra_dsp_out_26;
    reg [40-1:0] _extra_dsp_reg_26;
    assign _extra_dsp_out_26 = $signed(data_in ^ 40'd27) * $signed(data_in ^ 40'd28);
    always @(posedge clk) _extra_dsp_reg_26 <= _extra_dsp_out_26;
    wire [40-1:0] _extra_dsp_out_27;
    reg [40-1:0] _extra_dsp_reg_27;
    assign _extra_dsp_out_27 = $signed(data_in ^ 40'd28) * $signed(data_in ^ 40'd29);
    always @(posedge clk) _extra_dsp_reg_27 <= _extra_dsp_out_27;
    wire [40-1:0] _extra_dsp_out_28;
    reg [40-1:0] _extra_dsp_reg_28;
    assign _extra_dsp_out_28 = $signed(data_in ^ 40'd29) * $signed(data_in ^ 40'd30);
    always @(posedge clk) _extra_dsp_reg_28 <= _extra_dsp_out_28;
    wire [40-1:0] _extra_dsp_out_29;
    reg [40-1:0] _extra_dsp_reg_29;
    assign _extra_dsp_out_29 = $signed(data_in ^ 40'd30) * $signed(data_in ^ 40'd31);
    always @(posedge clk) _extra_dsp_reg_29 <= _extra_dsp_out_29;
    wire [40-1:0] _extra_dsp_out_30;
    reg [40-1:0] _extra_dsp_reg_30;
    assign _extra_dsp_out_30 = $signed(data_in ^ 40'd31) * $signed(data_in ^ 40'd32);
    always @(posedge clk) _extra_dsp_reg_30 <= _extra_dsp_out_30;
    wire [40-1:0] _extra_dsp_out_31;
    reg [40-1:0] _extra_dsp_reg_31;
    assign _extra_dsp_out_31 = $signed(data_in ^ 40'd32) * $signed(data_in ^ 40'd33);
    always @(posedge clk) _extra_dsp_reg_31 <= _extra_dsp_out_31;
    wire [40-1:0] _extra_dsp_out_32;
    reg [40-1:0] _extra_dsp_reg_32;
    assign _extra_dsp_out_32 = $signed(data_in ^ 40'd33) * $signed(data_in ^ 40'd34);
    always @(posedge clk) _extra_dsp_reg_32 <= _extra_dsp_out_32;
    wire [40-1:0] _extra_dsp_out_33;
    reg [40-1:0] _extra_dsp_reg_33;
    assign _extra_dsp_out_33 = $signed(data_in ^ 40'd34) * $signed(data_in ^ 40'd35);
    always @(posedge clk) _extra_dsp_reg_33 <= _extra_dsp_out_33;
    wire [40-1:0] _extra_dsp_out_34;
    reg [40-1:0] _extra_dsp_reg_34;
    assign _extra_dsp_out_34 = $signed(data_in ^ 40'd35) * $signed(data_in ^ 40'd36);
    always @(posedge clk) _extra_dsp_reg_34 <= _extra_dsp_out_34;
    wire [40-1:0] _extra_dsp_out_35;
    reg [40-1:0] _extra_dsp_reg_35;
    assign _extra_dsp_out_35 = $signed(data_in ^ 40'd36) * $signed(data_in ^ 40'd37);
    always @(posedge clk) _extra_dsp_reg_35 <= _extra_dsp_out_35;
    wire [40-1:0] _extra_dsp_out_36;
    reg [40-1:0] _extra_dsp_reg_36;
    assign _extra_dsp_out_36 = $signed(data_in ^ 40'd37) * $signed(data_in ^ 40'd38);
    always @(posedge clk) _extra_dsp_reg_36 <= _extra_dsp_out_36;
    wire [40-1:0] _extra_dsp_out_37;
    reg [40-1:0] _extra_dsp_reg_37;
    assign _extra_dsp_out_37 = $signed(data_in ^ 40'd38) * $signed(data_in ^ 40'd39);
    always @(posedge clk) _extra_dsp_reg_37 <= _extra_dsp_out_37;
    wire [40-1:0] _extra_dsp_out_38;
    reg [40-1:0] _extra_dsp_reg_38;
    assign _extra_dsp_out_38 = $signed(data_in ^ 40'd39) * $signed(data_in ^ 40'd40);
    always @(posedge clk) _extra_dsp_reg_38 <= _extra_dsp_out_38;
    wire [40-1:0] _extra_dsp_out_39;
    reg [40-1:0] _extra_dsp_reg_39;
    assign _extra_dsp_out_39 = $signed(data_in ^ 40'd40) * $signed(data_in ^ 40'd41);
    always @(posedge clk) _extra_dsp_reg_39 <= _extra_dsp_out_39;
    wire [40-1:0] _extra_dsp_out_40;
    reg [40-1:0] _extra_dsp_reg_40;
    assign _extra_dsp_out_40 = $signed(data_in ^ 40'd41) * $signed(data_in ^ 40'd42);
    always @(posedge clk) _extra_dsp_reg_40 <= _extra_dsp_out_40;
    wire [40-1:0] _extra_dsp_out_41;
    reg [40-1:0] _extra_dsp_reg_41;
    assign _extra_dsp_out_41 = $signed(data_in ^ 40'd42) * $signed(data_in ^ 40'd43);
    always @(posedge clk) _extra_dsp_reg_41 <= _extra_dsp_out_41;
    wire [40-1:0] _extra_dsp_out_42;
    reg [40-1:0] _extra_dsp_reg_42;
    assign _extra_dsp_out_42 = $signed(data_in ^ 40'd43) * $signed(data_in ^ 40'd44);
    always @(posedge clk) _extra_dsp_reg_42 <= _extra_dsp_out_42;
    wire [40-1:0] _extra_dsp_out_43;
    reg [40-1:0] _extra_dsp_reg_43;
    assign _extra_dsp_out_43 = $signed(data_in ^ 40'd44) * $signed(data_in ^ 40'd45);
    always @(posedge clk) _extra_dsp_reg_43 <= _extra_dsp_out_43;
    wire [40-1:0] _extra_dsp_out_44;
    reg [40-1:0] _extra_dsp_reg_44;
    assign _extra_dsp_out_44 = $signed(data_in ^ 40'd45) * $signed(data_in ^ 40'd46);
    always @(posedge clk) _extra_dsp_reg_44 <= _extra_dsp_out_44;
    wire [40-1:0] _extra_dsp_out_45;
    reg [40-1:0] _extra_dsp_reg_45;
    assign _extra_dsp_out_45 = $signed(data_in ^ 40'd46) * $signed(data_in ^ 40'd47);
    always @(posedge clk) _extra_dsp_reg_45 <= _extra_dsp_out_45;
    wire [40-1:0] _extra_dsp_out_46;
    reg [40-1:0] _extra_dsp_reg_46;
    assign _extra_dsp_out_46 = $signed(data_in ^ 40'd47) * $signed(data_in ^ 40'd48);
    always @(posedge clk) _extra_dsp_reg_46 <= _extra_dsp_out_46;
    wire [40-1:0] _extra_dsp_out_47;
    reg [40-1:0] _extra_dsp_reg_47;
    assign _extra_dsp_out_47 = $signed(data_in ^ 40'd48) * $signed(data_in ^ 40'd49);
    always @(posedge clk) _extra_dsp_reg_47 <= _extra_dsp_out_47;
    wire [40-1:0] _extra_dsp_out_48;
    reg [40-1:0] _extra_dsp_reg_48;
    assign _extra_dsp_out_48 = $signed(data_in ^ 40'd49) * $signed(data_in ^ 40'd50);
    always @(posedge clk) _extra_dsp_reg_48 <= _extra_dsp_out_48;
    wire [40-1:0] _extra_dsp_out_49;
    reg [40-1:0] _extra_dsp_reg_49;
    assign _extra_dsp_out_49 = $signed(data_in ^ 40'd50) * $signed(data_in ^ 40'd51);
    always @(posedge clk) _extra_dsp_reg_49 <= _extra_dsp_out_49;
    wire [40-1:0] _extra_dsp_out_50;
    reg [40-1:0] _extra_dsp_reg_50;
    assign _extra_dsp_out_50 = $signed(data_in ^ 40'd51) * $signed(data_in ^ 40'd52);
    always @(posedge clk) _extra_dsp_reg_50 <= _extra_dsp_out_50;
    wire [40-1:0] _extra_dsp_out_51;
    reg [40-1:0] _extra_dsp_reg_51;
    assign _extra_dsp_out_51 = $signed(data_in ^ 40'd52) * $signed(data_in ^ 40'd53);
    always @(posedge clk) _extra_dsp_reg_51 <= _extra_dsp_out_51;
    wire [40-1:0] _extra_dsp_out_52;
    reg [40-1:0] _extra_dsp_reg_52;
    assign _extra_dsp_out_52 = $signed(data_in ^ 40'd53) * $signed(data_in ^ 40'd54);
    always @(posedge clk) _extra_dsp_reg_52 <= _extra_dsp_out_52;
    wire [40-1:0] _extra_dsp_out_53;
    reg [40-1:0] _extra_dsp_reg_53;
    assign _extra_dsp_out_53 = $signed(data_in ^ 40'd54) * $signed(data_in ^ 40'd55);
    always @(posedge clk) _extra_dsp_reg_53 <= _extra_dsp_out_53;
    wire [40-1:0] _extra_dsp_out_54;
    reg [40-1:0] _extra_dsp_reg_54;
    assign _extra_dsp_out_54 = $signed(data_in ^ 40'd55) * $signed(data_in ^ 40'd56);
    always @(posedge clk) _extra_dsp_reg_54 <= _extra_dsp_out_54;
    wire [40-1:0] _extra_dsp_out_55;
    reg [40-1:0] _extra_dsp_reg_55;
    assign _extra_dsp_out_55 = $signed(data_in ^ 40'd56) * $signed(data_in ^ 40'd57);
    always @(posedge clk) _extra_dsp_reg_55 <= _extra_dsp_out_55;
    wire [40-1:0] _extra_dsp_out_56;
    reg [40-1:0] _extra_dsp_reg_56;
    assign _extra_dsp_out_56 = $signed(data_in ^ 40'd57) * $signed(data_in ^ 40'd58);
    always @(posedge clk) _extra_dsp_reg_56 <= _extra_dsp_out_56;
    wire [40-1:0] _extra_dsp_out_57;
    reg [40-1:0] _extra_dsp_reg_57;
    assign _extra_dsp_out_57 = $signed(data_in ^ 40'd58) * $signed(data_in ^ 40'd59);
    always @(posedge clk) _extra_dsp_reg_57 <= _extra_dsp_out_57;
    wire [40-1:0] _extra_dsp_out_58;
    reg [40-1:0] _extra_dsp_reg_58;
    assign _extra_dsp_out_58 = $signed(data_in ^ 40'd59) * $signed(data_in ^ 40'd60);
    always @(posedge clk) _extra_dsp_reg_58 <= _extra_dsp_out_58;
    wire [40-1:0] _extra_dsp_out_59;
    reg [40-1:0] _extra_dsp_reg_59;
    assign _extra_dsp_out_59 = $signed(data_in ^ 40'd60) * $signed(data_in ^ 40'd61);
    always @(posedge clk) _extra_dsp_reg_59 <= _extra_dsp_out_59;
    wire [40-1:0] _extra_dsp_out_60;
    reg [40-1:0] _extra_dsp_reg_60;
    assign _extra_dsp_out_60 = $signed(data_in ^ 40'd61) * $signed(data_in ^ 40'd62);
    always @(posedge clk) _extra_dsp_reg_60 <= _extra_dsp_out_60;
    wire [40-1:0] _extra_dsp_out_61;
    reg [40-1:0] _extra_dsp_reg_61;
    assign _extra_dsp_out_61 = $signed(data_in ^ 40'd62) * $signed(data_in ^ 40'd63);
    always @(posedge clk) _extra_dsp_reg_61 <= _extra_dsp_out_61;
    wire [40-1:0] _extra_dsp_out_62;
    reg [40-1:0] _extra_dsp_reg_62;
    assign _extra_dsp_out_62 = $signed(data_in ^ 40'd63) * $signed(data_in ^ 40'd64);
    always @(posedge clk) _extra_dsp_reg_62 <= _extra_dsp_out_62;
    wire [40-1:0] _extra_dsp_out_63;
    reg [40-1:0] _extra_dsp_reg_63;
    assign _extra_dsp_out_63 = $signed(data_in ^ 40'd64) * $signed(data_in ^ 40'd65);
    always @(posedge clk) _extra_dsp_reg_63 <= _extra_dsp_out_63;
    wire [40-1:0] _extra_dsp_out_64;
    reg [40-1:0] _extra_dsp_reg_64;
    assign _extra_dsp_out_64 = $signed(data_in ^ 40'd65) * $signed(data_in ^ 40'd66);
    always @(posedge clk) _extra_dsp_reg_64 <= _extra_dsp_out_64;
    wire [40-1:0] _extra_dsp_out_65;
    reg [40-1:0] _extra_dsp_reg_65;
    assign _extra_dsp_out_65 = $signed(data_in ^ 40'd66) * $signed(data_in ^ 40'd67);
    always @(posedge clk) _extra_dsp_reg_65 <= _extra_dsp_out_65;
    wire [40-1:0] _extra_dsp_out_66;
    reg [40-1:0] _extra_dsp_reg_66;
    assign _extra_dsp_out_66 = $signed(data_in ^ 40'd67) * $signed(data_in ^ 40'd68);
    always @(posedge clk) _extra_dsp_reg_66 <= _extra_dsp_out_66;
    wire [40-1:0] _extra_dsp_out_67;
    reg [40-1:0] _extra_dsp_reg_67;
    assign _extra_dsp_out_67 = $signed(data_in ^ 40'd68) * $signed(data_in ^ 40'd69);
    always @(posedge clk) _extra_dsp_reg_67 <= _extra_dsp_out_67;
    wire [40-1:0] _extra_dsp_out_68;
    reg [40-1:0] _extra_dsp_reg_68;
    assign _extra_dsp_out_68 = $signed(data_in ^ 40'd69) * $signed(data_in ^ 40'd70);
    always @(posedge clk) _extra_dsp_reg_68 <= _extra_dsp_out_68;
    wire [40-1:0] _extra_dsp_out_69;
    reg [40-1:0] _extra_dsp_reg_69;
    assign _extra_dsp_out_69 = $signed(data_in ^ 40'd70) * $signed(data_in ^ 40'd71);
    always @(posedge clk) _extra_dsp_reg_69 <= _extra_dsp_out_69;
    wire [40-1:0] _extra_dsp_out_70;
    reg [40-1:0] _extra_dsp_reg_70;
    assign _extra_dsp_out_70 = $signed(data_in ^ 40'd71) * $signed(data_in ^ 40'd72);
    always @(posedge clk) _extra_dsp_reg_70 <= _extra_dsp_out_70;
    wire [40-1:0] _extra_dsp_out_71;
    reg [40-1:0] _extra_dsp_reg_71;
    assign _extra_dsp_out_71 = $signed(data_in ^ 40'd72) * $signed(data_in ^ 40'd73);
    always @(posedge clk) _extra_dsp_reg_71 <= _extra_dsp_out_71;
    wire [40-1:0] _extra_dsp_out_72;
    reg [40-1:0] _extra_dsp_reg_72;
    assign _extra_dsp_out_72 = $signed(data_in ^ 40'd73) * $signed(data_in ^ 40'd74);
    always @(posedge clk) _extra_dsp_reg_72 <= _extra_dsp_out_72;
    wire [40-1:0] _extra_dsp_out_73;
    reg [40-1:0] _extra_dsp_reg_73;
    assign _extra_dsp_out_73 = $signed(data_in ^ 40'd74) * $signed(data_in ^ 40'd75);
    always @(posedge clk) _extra_dsp_reg_73 <= _extra_dsp_out_73;
    wire [40-1:0] _extra_dsp_out_74;
    reg [40-1:0] _extra_dsp_reg_74;
    assign _extra_dsp_out_74 = $signed(data_in ^ 40'd75) * $signed(data_in ^ 40'd76);
    always @(posedge clk) _extra_dsp_reg_74 <= _extra_dsp_out_74;
    wire [40-1:0] _extra_dsp_out_75;
    reg [40-1:0] _extra_dsp_reg_75;
    assign _extra_dsp_out_75 = $signed(data_in ^ 40'd76) * $signed(data_in ^ 40'd77);
    always @(posedge clk) _extra_dsp_reg_75 <= _extra_dsp_out_75;
    wire [40-1:0] _extra_dsp_out_76;
    reg [40-1:0] _extra_dsp_reg_76;
    assign _extra_dsp_out_76 = $signed(data_in ^ 40'd77) * $signed(data_in ^ 40'd78);
    always @(posedge clk) _extra_dsp_reg_76 <= _extra_dsp_out_76;
    wire [40-1:0] _extra_dsp_out_77;
    reg [40-1:0] _extra_dsp_reg_77;
    assign _extra_dsp_out_77 = $signed(data_in ^ 40'd78) * $signed(data_in ^ 40'd79);
    always @(posedge clk) _extra_dsp_reg_77 <= _extra_dsp_out_77;
    wire [40-1:0] _extra_dsp_out_78;
    reg [40-1:0] _extra_dsp_reg_78;
    assign _extra_dsp_out_78 = $signed(data_in ^ 40'd79) * $signed(data_in ^ 40'd80);
    always @(posedge clk) _extra_dsp_reg_78 <= _extra_dsp_out_78;
    wire [40-1:0] _extra_dsp_out_79;
    reg [40-1:0] _extra_dsp_reg_79;
    assign _extra_dsp_out_79 = $signed(data_in ^ 40'd80) * $signed(data_in ^ 40'd81);
    always @(posedge clk) _extra_dsp_reg_79 <= _extra_dsp_out_79;
    wire [40-1:0] _extra_dsp_out_80;
    reg [40-1:0] _extra_dsp_reg_80;
    assign _extra_dsp_out_80 = $signed(data_in ^ 40'd81) * $signed(data_in ^ 40'd82);
    always @(posedge clk) _extra_dsp_reg_80 <= _extra_dsp_out_80;
    wire [40-1:0] _extra_dsp_out_81;
    reg [40-1:0] _extra_dsp_reg_81;
    assign _extra_dsp_out_81 = $signed(data_in ^ 40'd82) * $signed(data_in ^ 40'd83);
    always @(posedge clk) _extra_dsp_reg_81 <= _extra_dsp_out_81;
    wire [40-1:0] _extra_dsp_out_82;
    reg [40-1:0] _extra_dsp_reg_82;
    assign _extra_dsp_out_82 = $signed(data_in ^ 40'd83) * $signed(data_in ^ 40'd84);
    always @(posedge clk) _extra_dsp_reg_82 <= _extra_dsp_out_82;
    wire [40-1:0] _extra_dsp_out_83;
    reg [40-1:0] _extra_dsp_reg_83;
    assign _extra_dsp_out_83 = $signed(data_in ^ 40'd84) * $signed(data_in ^ 40'd85);
    always @(posedge clk) _extra_dsp_reg_83 <= _extra_dsp_out_83;
    wire [40-1:0] _extra_dsp_out_84;
    reg [40-1:0] _extra_dsp_reg_84;
    assign _extra_dsp_out_84 = $signed(data_in ^ 40'd85) * $signed(data_in ^ 40'd86);
    always @(posedge clk) _extra_dsp_reg_84 <= _extra_dsp_out_84;
    wire [40-1:0] _extra_dsp_out_85;
    reg [40-1:0] _extra_dsp_reg_85;
    assign _extra_dsp_out_85 = $signed(data_in ^ 40'd86) * $signed(data_in ^ 40'd87);
    always @(posedge clk) _extra_dsp_reg_85 <= _extra_dsp_out_85;
    wire [40-1:0] _extra_dsp_out_86;
    reg [40-1:0] _extra_dsp_reg_86;
    assign _extra_dsp_out_86 = $signed(data_in ^ 40'd87) * $signed(data_in ^ 40'd88);
    always @(posedge clk) _extra_dsp_reg_86 <= _extra_dsp_out_86;
    wire [40-1:0] _extra_dsp_out_87;
    reg [40-1:0] _extra_dsp_reg_87;
    assign _extra_dsp_out_87 = $signed(data_in ^ 40'd88) * $signed(data_in ^ 40'd89);
    always @(posedge clk) _extra_dsp_reg_87 <= _extra_dsp_out_87;
    wire [40-1:0] _extra_dsp_out_88;
    reg [40-1:0] _extra_dsp_reg_88;
    assign _extra_dsp_out_88 = $signed(data_in ^ 40'd89) * $signed(data_in ^ 40'd90);
    always @(posedge clk) _extra_dsp_reg_88 <= _extra_dsp_out_88;
    wire [40-1:0] _extra_dsp_out_89;
    reg [40-1:0] _extra_dsp_reg_89;
    assign _extra_dsp_out_89 = $signed(data_in ^ 40'd90) * $signed(data_in ^ 40'd91);
    always @(posedge clk) _extra_dsp_reg_89 <= _extra_dsp_out_89;
    wire [40-1:0] _extra_dsp_out_90;
    reg [40-1:0] _extra_dsp_reg_90;
    assign _extra_dsp_out_90 = $signed(data_in ^ 40'd91) * $signed(data_in ^ 40'd92);
    always @(posedge clk) _extra_dsp_reg_90 <= _extra_dsp_out_90;
    wire [40-1:0] _extra_dsp_out_91;
    reg [40-1:0] _extra_dsp_reg_91;
    assign _extra_dsp_out_91 = $signed(data_in ^ 40'd92) * $signed(data_in ^ 40'd93);
    always @(posedge clk) _extra_dsp_reg_91 <= _extra_dsp_out_91;
    wire [40-1:0] _extra_dsp_out_92;
    reg [40-1:0] _extra_dsp_reg_92;
    assign _extra_dsp_out_92 = $signed(data_in ^ 40'd93) * $signed(data_in ^ 40'd94);
    always @(posedge clk) _extra_dsp_reg_92 <= _extra_dsp_out_92;
    wire [40-1:0] _extra_dsp_out_93;
    reg [40-1:0] _extra_dsp_reg_93;
    assign _extra_dsp_out_93 = $signed(data_in ^ 40'd94) * $signed(data_in ^ 40'd95);
    always @(posedge clk) _extra_dsp_reg_93 <= _extra_dsp_out_93;
    wire [40-1:0] _extra_dsp_out_94;
    reg [40-1:0] _extra_dsp_reg_94;
    assign _extra_dsp_out_94 = $signed(data_in ^ 40'd95) * $signed(data_in ^ 40'd96);
    always @(posedge clk) _extra_dsp_reg_94 <= _extra_dsp_out_94;
    wire [40-1:0] _extra_dsp_out_95;
    reg [40-1:0] _extra_dsp_reg_95;
    assign _extra_dsp_out_95 = $signed(data_in ^ 40'd96) * $signed(data_in ^ 40'd97);
    always @(posedge clk) _extra_dsp_reg_95 <= _extra_dsp_out_95;
    wire [40-1:0] _extra_dsp_out_96;
    reg [40-1:0] _extra_dsp_reg_96;
    assign _extra_dsp_out_96 = $signed(data_in ^ 40'd97) * $signed(data_in ^ 40'd98);
    always @(posedge clk) _extra_dsp_reg_96 <= _extra_dsp_out_96;
    wire [40-1:0] _extra_dsp_out_97;
    reg [40-1:0] _extra_dsp_reg_97;
    assign _extra_dsp_out_97 = $signed(data_in ^ 40'd98) * $signed(data_in ^ 40'd99);
    always @(posedge clk) _extra_dsp_reg_97 <= _extra_dsp_out_97;
    wire [40-1:0] _extra_dsp_out_98;
    reg [40-1:0] _extra_dsp_reg_98;
    assign _extra_dsp_out_98 = $signed(data_in ^ 40'd99) * $signed(data_in ^ 40'd100);
    always @(posedge clk) _extra_dsp_reg_98 <= _extra_dsp_out_98;
    wire [40-1:0] _extra_dsp_out_99;
    reg [40-1:0] _extra_dsp_reg_99;
    assign _extra_dsp_out_99 = $signed(data_in ^ 40'd100) * $signed(data_in ^ 40'd101);
    always @(posedge clk) _extra_dsp_reg_99 <= _extra_dsp_out_99;
    wire [40-1:0] _extra_dsp_out_100;
    reg [40-1:0] _extra_dsp_reg_100;
    assign _extra_dsp_out_100 = $signed(data_in ^ 40'd101) * $signed(data_in ^ 40'd102);
    always @(posedge clk) _extra_dsp_reg_100 <= _extra_dsp_out_100;
    wire [40-1:0] _extra_dsp_out_101;
    reg [40-1:0] _extra_dsp_reg_101;
    assign _extra_dsp_out_101 = $signed(data_in ^ 40'd102) * $signed(data_in ^ 40'd103);
    always @(posedge clk) _extra_dsp_reg_101 <= _extra_dsp_out_101;
    wire [40-1:0] _extra_dsp_out_102;
    reg [40-1:0] _extra_dsp_reg_102;
    assign _extra_dsp_out_102 = $signed(data_in ^ 40'd103) * $signed(data_in ^ 40'd104);
    always @(posedge clk) _extra_dsp_reg_102 <= _extra_dsp_out_102;
    wire [40-1:0] _extra_dsp_out_103;
    reg [40-1:0] _extra_dsp_reg_103;
    assign _extra_dsp_out_103 = $signed(data_in ^ 40'd104) * $signed(data_in ^ 40'd105);
    always @(posedge clk) _extra_dsp_reg_103 <= _extra_dsp_out_103;
    wire [40-1:0] _extra_dsp_out_104;
    reg [40-1:0] _extra_dsp_reg_104;
    assign _extra_dsp_out_104 = $signed(data_in ^ 40'd105) * $signed(data_in ^ 40'd106);
    always @(posedge clk) _extra_dsp_reg_104 <= _extra_dsp_out_104;
    wire [40-1:0] _extra_dsp_out_105;
    reg [40-1:0] _extra_dsp_reg_105;
    assign _extra_dsp_out_105 = $signed(data_in ^ 40'd106) * $signed(data_in ^ 40'd107);
    always @(posedge clk) _extra_dsp_reg_105 <= _extra_dsp_out_105;
    wire [40-1:0] _extra_dsp_out_106;
    reg [40-1:0] _extra_dsp_reg_106;
    assign _extra_dsp_out_106 = $signed(data_in ^ 40'd107) * $signed(data_in ^ 40'd108);
    always @(posedge clk) _extra_dsp_reg_106 <= _extra_dsp_out_106;
    wire [40-1:0] _extra_dsp_out_107;
    reg [40-1:0] _extra_dsp_reg_107;
    assign _extra_dsp_out_107 = $signed(data_in ^ 40'd108) * $signed(data_in ^ 40'd109);
    always @(posedge clk) _extra_dsp_reg_107 <= _extra_dsp_out_107;
    wire [40-1:0] _extra_dsp_out_108;
    reg [40-1:0] _extra_dsp_reg_108;
    assign _extra_dsp_out_108 = $signed(data_in ^ 40'd109) * $signed(data_in ^ 40'd110);
    always @(posedge clk) _extra_dsp_reg_108 <= _extra_dsp_out_108;
    wire [40-1:0] _extra_dsp_out_109;
    reg [40-1:0] _extra_dsp_reg_109;
    assign _extra_dsp_out_109 = $signed(data_in ^ 40'd110) * $signed(data_in ^ 40'd111);
    always @(posedge clk) _extra_dsp_reg_109 <= _extra_dsp_out_109;
    wire [40-1:0] _extra_dsp_out_110;
    reg [40-1:0] _extra_dsp_reg_110;
    assign _extra_dsp_out_110 = $signed(data_in ^ 40'd111) * $signed(data_in ^ 40'd112);
    always @(posedge clk) _extra_dsp_reg_110 <= _extra_dsp_out_110;
    wire [40-1:0] _extra_dsp_out_111;
    reg [40-1:0] _extra_dsp_reg_111;
    assign _extra_dsp_out_111 = $signed(data_in ^ 40'd112) * $signed(data_in ^ 40'd113);
    always @(posedge clk) _extra_dsp_reg_111 <= _extra_dsp_out_111;
    wire [40-1:0] _extra_dsp_out_112;
    reg [40-1:0] _extra_dsp_reg_112;
    assign _extra_dsp_out_112 = $signed(data_in ^ 40'd113) * $signed(data_in ^ 40'd114);
    always @(posedge clk) _extra_dsp_reg_112 <= _extra_dsp_out_112;
    wire [40-1:0] _extra_dsp_out_113;
    reg [40-1:0] _extra_dsp_reg_113;
    assign _extra_dsp_out_113 = $signed(data_in ^ 40'd114) * $signed(data_in ^ 40'd115);
    always @(posedge clk) _extra_dsp_reg_113 <= _extra_dsp_out_113;
    wire [40-1:0] _extra_dsp_out_114;
    reg [40-1:0] _extra_dsp_reg_114;
    assign _extra_dsp_out_114 = $signed(data_in ^ 40'd115) * $signed(data_in ^ 40'd116);
    always @(posedge clk) _extra_dsp_reg_114 <= _extra_dsp_out_114;
    wire [40-1:0] _extra_dsp_out_115;
    reg [40-1:0] _extra_dsp_reg_115;
    assign _extra_dsp_out_115 = $signed(data_in ^ 40'd116) * $signed(data_in ^ 40'd117);
    always @(posedge clk) _extra_dsp_reg_115 <= _extra_dsp_out_115;
    wire [40-1:0] _extra_dsp_out_116;
    reg [40-1:0] _extra_dsp_reg_116;
    assign _extra_dsp_out_116 = $signed(data_in ^ 40'd117) * $signed(data_in ^ 40'd118);
    always @(posedge clk) _extra_dsp_reg_116 <= _extra_dsp_out_116;
    wire [40-1:0] _extra_dsp_out_117;
    reg [40-1:0] _extra_dsp_reg_117;
    assign _extra_dsp_out_117 = $signed(data_in ^ 40'd118) * $signed(data_in ^ 40'd119);
    always @(posedge clk) _extra_dsp_reg_117 <= _extra_dsp_out_117;
    wire [40-1:0] _extra_dsp_out_118;
    reg [40-1:0] _extra_dsp_reg_118;
    assign _extra_dsp_out_118 = $signed(data_in ^ 40'd119) * $signed(data_in ^ 40'd120);
    always @(posedge clk) _extra_dsp_reg_118 <= _extra_dsp_out_118;
    wire [40-1:0] _extra_dsp_out_119;
    reg [40-1:0] _extra_dsp_reg_119;
    assign _extra_dsp_out_119 = $signed(data_in ^ 40'd120) * $signed(data_in ^ 40'd121);
    always @(posedge clk) _extra_dsp_reg_119 <= _extra_dsp_out_119;
    wire [40-1:0] _extra_dsp_out_120;
    reg [40-1:0] _extra_dsp_reg_120;
    assign _extra_dsp_out_120 = $signed(data_in ^ 40'd121) * $signed(data_in ^ 40'd122);
    always @(posedge clk) _extra_dsp_reg_120 <= _extra_dsp_out_120;
    wire [40-1:0] _extra_dsp_out_121;
    reg [40-1:0] _extra_dsp_reg_121;
    assign _extra_dsp_out_121 = $signed(data_in ^ 40'd122) * $signed(data_in ^ 40'd123);
    always @(posedge clk) _extra_dsp_reg_121 <= _extra_dsp_out_121;
    wire [40-1:0] _extra_dsp_out_122;
    reg [40-1:0] _extra_dsp_reg_122;
    assign _extra_dsp_out_122 = $signed(data_in ^ 40'd123) * $signed(data_in ^ 40'd124);
    always @(posedge clk) _extra_dsp_reg_122 <= _extra_dsp_out_122;
    wire [40-1:0] _extra_dsp_out_123;
    reg [40-1:0] _extra_dsp_reg_123;
    assign _extra_dsp_out_123 = $signed(data_in ^ 40'd124) * $signed(data_in ^ 40'd125);
    always @(posedge clk) _extra_dsp_reg_123 <= _extra_dsp_out_123;
    wire [40-1:0] _extra_dsp_out_124;
    reg [40-1:0] _extra_dsp_reg_124;
    assign _extra_dsp_out_124 = $signed(data_in ^ 40'd125) * $signed(data_in ^ 40'd126);
    always @(posedge clk) _extra_dsp_reg_124 <= _extra_dsp_out_124;
    wire [40-1:0] _extra_dsp_out_125;
    reg [40-1:0] _extra_dsp_reg_125;
    assign _extra_dsp_out_125 = $signed(data_in ^ 40'd126) * $signed(data_in ^ 40'd127);
    always @(posedge clk) _extra_dsp_reg_125 <= _extra_dsp_out_125;
    wire [40-1:0] _extra_dsp_out_126;
    reg [40-1:0] _extra_dsp_reg_126;
    assign _extra_dsp_out_126 = $signed(data_in ^ 40'd127) * $signed(data_in ^ 40'd128);
    always @(posedge clk) _extra_dsp_reg_126 <= _extra_dsp_out_126;
    wire [40-1:0] _extra_dsp_out_127;
    reg [40-1:0] _extra_dsp_reg_127;
    assign _extra_dsp_out_127 = $signed(data_in ^ 40'd128) * $signed(data_in ^ 40'd129);
    always @(posedge clk) _extra_dsp_reg_127 <= _extra_dsp_out_127;
    wire [40-1:0] _extra_dsp_out_128;
    reg [40-1:0] _extra_dsp_reg_128;
    assign _extra_dsp_out_128 = $signed(data_in ^ 40'd129) * $signed(data_in ^ 40'd130);
    always @(posedge clk) _extra_dsp_reg_128 <= _extra_dsp_out_128;
    wire [40-1:0] _extra_dsp_out_129;
    reg [40-1:0] _extra_dsp_reg_129;
    assign _extra_dsp_out_129 = $signed(data_in ^ 40'd130) * $signed(data_in ^ 40'd131);
    always @(posedge clk) _extra_dsp_reg_129 <= _extra_dsp_out_129;
    wire [40-1:0] _extra_dsp_out_130;
    reg [40-1:0] _extra_dsp_reg_130;
    assign _extra_dsp_out_130 = $signed(data_in ^ 40'd131) * $signed(data_in ^ 40'd132);
    always @(posedge clk) _extra_dsp_reg_130 <= _extra_dsp_out_130;
    wire [40-1:0] _extra_dsp_out_131;
    reg [40-1:0] _extra_dsp_reg_131;
    assign _extra_dsp_out_131 = $signed(data_in ^ 40'd132) * $signed(data_in ^ 40'd133);
    always @(posedge clk) _extra_dsp_reg_131 <= _extra_dsp_out_131;
    wire [40-1:0] _extra_dsp_out_132;
    reg [40-1:0] _extra_dsp_reg_132;
    assign _extra_dsp_out_132 = $signed(data_in ^ 40'd133) * $signed(data_in ^ 40'd134);
    always @(posedge clk) _extra_dsp_reg_132 <= _extra_dsp_out_132;
    wire [40-1:0] _extra_dsp_out_133;
    reg [40-1:0] _extra_dsp_reg_133;
    assign _extra_dsp_out_133 = $signed(data_in ^ 40'd134) * $signed(data_in ^ 40'd135);
    always @(posedge clk) _extra_dsp_reg_133 <= _extra_dsp_out_133;
    wire [40-1:0] _extra_dsp_out_134;
    reg [40-1:0] _extra_dsp_reg_134;
    assign _extra_dsp_out_134 = $signed(data_in ^ 40'd135) * $signed(data_in ^ 40'd136);
    always @(posedge clk) _extra_dsp_reg_134 <= _extra_dsp_out_134;
    wire [40-1:0] _extra_dsp_out_135;
    reg [40-1:0] _extra_dsp_reg_135;
    assign _extra_dsp_out_135 = $signed(data_in ^ 40'd136) * $signed(data_in ^ 40'd137);
    always @(posedge clk) _extra_dsp_reg_135 <= _extra_dsp_out_135;
    wire [40-1:0] _extra_dsp_out_136;
    reg [40-1:0] _extra_dsp_reg_136;
    assign _extra_dsp_out_136 = $signed(data_in ^ 40'd137) * $signed(data_in ^ 40'd138);
    always @(posedge clk) _extra_dsp_reg_136 <= _extra_dsp_out_136;
    wire [40-1:0] _extra_dsp_out_137;
    reg [40-1:0] _extra_dsp_reg_137;
    assign _extra_dsp_out_137 = $signed(data_in ^ 40'd138) * $signed(data_in ^ 40'd139);
    always @(posedge clk) _extra_dsp_reg_137 <= _extra_dsp_out_137;
    wire [40-1:0] _extra_dsp_out_138;
    reg [40-1:0] _extra_dsp_reg_138;
    assign _extra_dsp_out_138 = $signed(data_in ^ 40'd139) * $signed(data_in ^ 40'd140);
    always @(posedge clk) _extra_dsp_reg_138 <= _extra_dsp_out_138;
    wire [40-1:0] _extra_dsp_out_139;
    reg [40-1:0] _extra_dsp_reg_139;
    assign _extra_dsp_out_139 = $signed(data_in ^ 40'd140) * $signed(data_in ^ 40'd141);
    always @(posedge clk) _extra_dsp_reg_139 <= _extra_dsp_out_139;
    wire [40-1:0] _extra_dsp_out_140;
    reg [40-1:0] _extra_dsp_reg_140;
    assign _extra_dsp_out_140 = $signed(data_in ^ 40'd141) * $signed(data_in ^ 40'd142);
    always @(posedge clk) _extra_dsp_reg_140 <= _extra_dsp_out_140;
    wire [40-1:0] _extra_dsp_out_141;
    reg [40-1:0] _extra_dsp_reg_141;
    assign _extra_dsp_out_141 = $signed(data_in ^ 40'd142) * $signed(data_in ^ 40'd143);
    always @(posedge clk) _extra_dsp_reg_141 <= _extra_dsp_out_141;
    wire [40-1:0] _extra_dsp_out_142;
    reg [40-1:0] _extra_dsp_reg_142;
    assign _extra_dsp_out_142 = $signed(data_in ^ 40'd143) * $signed(data_in ^ 40'd144);
    always @(posedge clk) _extra_dsp_reg_142 <= _extra_dsp_out_142;
    wire [40-1:0] _extra_dsp_out_143;
    reg [40-1:0] _extra_dsp_reg_143;
    assign _extra_dsp_out_143 = $signed(data_in ^ 40'd144) * $signed(data_in ^ 40'd145);
    always @(posedge clk) _extra_dsp_reg_143 <= _extra_dsp_out_143;
    wire [40-1:0] _extra_dsp_out_144;
    reg [40-1:0] _extra_dsp_reg_144;
    assign _extra_dsp_out_144 = $signed(data_in ^ 40'd145) * $signed(data_in ^ 40'd146);
    always @(posedge clk) _extra_dsp_reg_144 <= _extra_dsp_out_144;
    wire [40-1:0] _extra_dsp_out_145;
    reg [40-1:0] _extra_dsp_reg_145;
    assign _extra_dsp_out_145 = $signed(data_in ^ 40'd146) * $signed(data_in ^ 40'd147);
    always @(posedge clk) _extra_dsp_reg_145 <= _extra_dsp_out_145;
    wire [40-1:0] _extra_dsp_out_146;
    reg [40-1:0] _extra_dsp_reg_146;
    assign _extra_dsp_out_146 = $signed(data_in ^ 40'd147) * $signed(data_in ^ 40'd148);
    always @(posedge clk) _extra_dsp_reg_146 <= _extra_dsp_out_146;
    wire [40-1:0] _extra_dsp_out_147;
    reg [40-1:0] _extra_dsp_reg_147;
    assign _extra_dsp_out_147 = $signed(data_in ^ 40'd148) * $signed(data_in ^ 40'd149);
    always @(posedge clk) _extra_dsp_reg_147 <= _extra_dsp_out_147;
    wire [40-1:0] _extra_dsp_out_148;
    reg [40-1:0] _extra_dsp_reg_148;
    assign _extra_dsp_out_148 = $signed(data_in ^ 40'd149) * $signed(data_in ^ 40'd150);
    always @(posedge clk) _extra_dsp_reg_148 <= _extra_dsp_out_148;
    wire [40-1:0] _extra_dsp_out_149;
    reg [40-1:0] _extra_dsp_reg_149;
    assign _extra_dsp_out_149 = $signed(data_in ^ 40'd150) * $signed(data_in ^ 40'd151);
    always @(posedge clk) _extra_dsp_reg_149 <= _extra_dsp_out_149;
    wire [40-1:0] _extra_dsp_out_150;
    reg [40-1:0] _extra_dsp_reg_150;
    assign _extra_dsp_out_150 = $signed(data_in ^ 40'd151) * $signed(data_in ^ 40'd152);
    always @(posedge clk) _extra_dsp_reg_150 <= _extra_dsp_out_150;
    wire [40-1:0] _extra_dsp_out_151;
    reg [40-1:0] _extra_dsp_reg_151;
    assign _extra_dsp_out_151 = $signed(data_in ^ 40'd152) * $signed(data_in ^ 40'd153);
    always @(posedge clk) _extra_dsp_reg_151 <= _extra_dsp_out_151;
    wire [40-1:0] _extra_dsp_out_152;
    reg [40-1:0] _extra_dsp_reg_152;
    assign _extra_dsp_out_152 = $signed(data_in ^ 40'd153) * $signed(data_in ^ 40'd154);
    always @(posedge clk) _extra_dsp_reg_152 <= _extra_dsp_out_152;
    wire [40-1:0] _extra_dsp_out_153;
    reg [40-1:0] _extra_dsp_reg_153;
    assign _extra_dsp_out_153 = $signed(data_in ^ 40'd154) * $signed(data_in ^ 40'd155);
    always @(posedge clk) _extra_dsp_reg_153 <= _extra_dsp_out_153;
    wire [40-1:0] _extra_dsp_out_154;
    reg [40-1:0] _extra_dsp_reg_154;
    assign _extra_dsp_out_154 = $signed(data_in ^ 40'd155) * $signed(data_in ^ 40'd156);
    always @(posedge clk) _extra_dsp_reg_154 <= _extra_dsp_out_154;
    wire [40-1:0] _extra_dsp_out_155;
    reg [40-1:0] _extra_dsp_reg_155;
    assign _extra_dsp_out_155 = $signed(data_in ^ 40'd156) * $signed(data_in ^ 40'd157);
    always @(posedge clk) _extra_dsp_reg_155 <= _extra_dsp_out_155;
    wire [40-1:0] _extra_dsp_out_156;
    reg [40-1:0] _extra_dsp_reg_156;
    assign _extra_dsp_out_156 = $signed(data_in ^ 40'd157) * $signed(data_in ^ 40'd158);
    always @(posedge clk) _extra_dsp_reg_156 <= _extra_dsp_out_156;
    wire [40-1:0] _extra_dsp_out_157;
    reg [40-1:0] _extra_dsp_reg_157;
    assign _extra_dsp_out_157 = $signed(data_in ^ 40'd158) * $signed(data_in ^ 40'd159);
    always @(posedge clk) _extra_dsp_reg_157 <= _extra_dsp_out_157;
    wire [40-1:0] _extra_dsp_out_158;
    reg [40-1:0] _extra_dsp_reg_158;
    assign _extra_dsp_out_158 = $signed(data_in ^ 40'd159) * $signed(data_in ^ 40'd160);
    always @(posedge clk) _extra_dsp_reg_158 <= _extra_dsp_out_158;
    wire [40-1:0] _extra_dsp_out_159;
    reg [40-1:0] _extra_dsp_reg_159;
    assign _extra_dsp_out_159 = $signed(data_in ^ 40'd160) * $signed(data_in ^ 40'd161);
    always @(posedge clk) _extra_dsp_reg_159 <= _extra_dsp_out_159;
    wire [40-1:0] _extra_dsp_out_160;
    reg [40-1:0] _extra_dsp_reg_160;
    assign _extra_dsp_out_160 = $signed(data_in ^ 40'd161) * $signed(data_in ^ 40'd162);
    always @(posedge clk) _extra_dsp_reg_160 <= _extra_dsp_out_160;
    wire [40-1:0] _extra_dsp_out_161;
    reg [40-1:0] _extra_dsp_reg_161;
    assign _extra_dsp_out_161 = $signed(data_in ^ 40'd162) * $signed(data_in ^ 40'd163);
    always @(posedge clk) _extra_dsp_reg_161 <= _extra_dsp_out_161;
    wire [40-1:0] _extra_dsp_out_162;
    reg [40-1:0] _extra_dsp_reg_162;
    assign _extra_dsp_out_162 = $signed(data_in ^ 40'd163) * $signed(data_in ^ 40'd164);
    always @(posedge clk) _extra_dsp_reg_162 <= _extra_dsp_out_162;
    wire [40-1:0] _extra_dsp_out_163;
    reg [40-1:0] _extra_dsp_reg_163;
    assign _extra_dsp_out_163 = $signed(data_in ^ 40'd164) * $signed(data_in ^ 40'd165);
    always @(posedge clk) _extra_dsp_reg_163 <= _extra_dsp_out_163;
    wire [40-1:0] _extra_dsp_out_164;
    reg [40-1:0] _extra_dsp_reg_164;
    assign _extra_dsp_out_164 = $signed(data_in ^ 40'd165) * $signed(data_in ^ 40'd166);
    always @(posedge clk) _extra_dsp_reg_164 <= _extra_dsp_out_164;
    wire [40-1:0] _extra_dsp_out_165;
    reg [40-1:0] _extra_dsp_reg_165;
    assign _extra_dsp_out_165 = $signed(data_in ^ 40'd166) * $signed(data_in ^ 40'd167);
    always @(posedge clk) _extra_dsp_reg_165 <= _extra_dsp_out_165;
    wire [40-1:0] _extra_dsp_out_166;
    reg [40-1:0] _extra_dsp_reg_166;
    assign _extra_dsp_out_166 = $signed(data_in ^ 40'd167) * $signed(data_in ^ 40'd168);
    always @(posedge clk) _extra_dsp_reg_166 <= _extra_dsp_out_166;
    wire [40-1:0] _extra_dsp_out_167;
    reg [40-1:0] _extra_dsp_reg_167;
    assign _extra_dsp_out_167 = $signed(data_in ^ 40'd168) * $signed(data_in ^ 40'd169);
    always @(posedge clk) _extra_dsp_reg_167 <= _extra_dsp_out_167;
    wire [40-1:0] _extra_dsp_out_168;
    reg [40-1:0] _extra_dsp_reg_168;
    assign _extra_dsp_out_168 = $signed(data_in ^ 40'd169) * $signed(data_in ^ 40'd170);
    always @(posedge clk) _extra_dsp_reg_168 <= _extra_dsp_out_168;
    wire [40-1:0] _extra_dsp_out_169;
    reg [40-1:0] _extra_dsp_reg_169;
    assign _extra_dsp_out_169 = $signed(data_in ^ 40'd170) * $signed(data_in ^ 40'd171);
    always @(posedge clk) _extra_dsp_reg_169 <= _extra_dsp_out_169;
    wire [40-1:0] _extra_dsp_out_170;
    reg [40-1:0] _extra_dsp_reg_170;
    assign _extra_dsp_out_170 = $signed(data_in ^ 40'd171) * $signed(data_in ^ 40'd172);
    always @(posedge clk) _extra_dsp_reg_170 <= _extra_dsp_out_170;
    wire [40-1:0] _extra_dsp_out_171;
    reg [40-1:0] _extra_dsp_reg_171;
    assign _extra_dsp_out_171 = $signed(data_in ^ 40'd172) * $signed(data_in ^ 40'd173);
    always @(posedge clk) _extra_dsp_reg_171 <= _extra_dsp_out_171;
    wire [40-1:0] _extra_dsp_out_172;
    reg [40-1:0] _extra_dsp_reg_172;
    assign _extra_dsp_out_172 = $signed(data_in ^ 40'd173) * $signed(data_in ^ 40'd174);
    always @(posedge clk) _extra_dsp_reg_172 <= _extra_dsp_out_172;
    wire [40-1:0] _extra_dsp_out_173;
    reg [40-1:0] _extra_dsp_reg_173;
    assign _extra_dsp_out_173 = $signed(data_in ^ 40'd174) * $signed(data_in ^ 40'd175);
    always @(posedge clk) _extra_dsp_reg_173 <= _extra_dsp_out_173;
    wire [40-1:0] _extra_dsp_out_174;
    reg [40-1:0] _extra_dsp_reg_174;
    assign _extra_dsp_out_174 = $signed(data_in ^ 40'd175) * $signed(data_in ^ 40'd176);
    always @(posedge clk) _extra_dsp_reg_174 <= _extra_dsp_out_174;
    wire [40-1:0] _extra_dsp_out_175;
    reg [40-1:0] _extra_dsp_reg_175;
    assign _extra_dsp_out_175 = $signed(data_in ^ 40'd176) * $signed(data_in ^ 40'd177);
    always @(posedge clk) _extra_dsp_reg_175 <= _extra_dsp_out_175;
    wire [40-1:0] _extra_dsp_out_176;
    reg [40-1:0] _extra_dsp_reg_176;
    assign _extra_dsp_out_176 = $signed(data_in ^ 40'd177) * $signed(data_in ^ 40'd178);
    always @(posedge clk) _extra_dsp_reg_176 <= _extra_dsp_out_176;
    wire [40-1:0] _extra_dsp_out_177;
    reg [40-1:0] _extra_dsp_reg_177;
    assign _extra_dsp_out_177 = $signed(data_in ^ 40'd178) * $signed(data_in ^ 40'd179);
    always @(posedge clk) _extra_dsp_reg_177 <= _extra_dsp_out_177;
    wire [40-1:0] _extra_dsp_out_178;
    reg [40-1:0] _extra_dsp_reg_178;
    assign _extra_dsp_out_178 = $signed(data_in ^ 40'd179) * $signed(data_in ^ 40'd180);
    always @(posedge clk) _extra_dsp_reg_178 <= _extra_dsp_out_178;
    wire [40-1:0] _extra_dsp_out_179;
    reg [40-1:0] _extra_dsp_reg_179;
    assign _extra_dsp_out_179 = $signed(data_in ^ 40'd180) * $signed(data_in ^ 40'd181);
    always @(posedge clk) _extra_dsp_reg_179 <= _extra_dsp_out_179;
    wire [40-1:0] _extra_dsp_out_180;
    reg [40-1:0] _extra_dsp_reg_180;
    assign _extra_dsp_out_180 = $signed(data_in ^ 40'd181) * $signed(data_in ^ 40'd182);
    always @(posedge clk) _extra_dsp_reg_180 <= _extra_dsp_out_180;
    wire [40-1:0] _extra_dsp_out_181;
    reg [40-1:0] _extra_dsp_reg_181;
    assign _extra_dsp_out_181 = $signed(data_in ^ 40'd182) * $signed(data_in ^ 40'd183);
    always @(posedge clk) _extra_dsp_reg_181 <= _extra_dsp_out_181;
    wire [40-1:0] _extra_dsp_out_182;
    reg [40-1:0] _extra_dsp_reg_182;
    assign _extra_dsp_out_182 = $signed(data_in ^ 40'd183) * $signed(data_in ^ 40'd184);
    always @(posedge clk) _extra_dsp_reg_182 <= _extra_dsp_out_182;
    wire [40-1:0] _extra_dsp_out_183;
    reg [40-1:0] _extra_dsp_reg_183;
    assign _extra_dsp_out_183 = $signed(data_in ^ 40'd184) * $signed(data_in ^ 40'd185);
    always @(posedge clk) _extra_dsp_reg_183 <= _extra_dsp_out_183;
    wire [40-1:0] _extra_dsp_out_184;
    reg [40-1:0] _extra_dsp_reg_184;
    assign _extra_dsp_out_184 = $signed(data_in ^ 40'd185) * $signed(data_in ^ 40'd186);
    always @(posedge clk) _extra_dsp_reg_184 <= _extra_dsp_out_184;
    wire [40-1:0] _extra_dsp_out_185;
    reg [40-1:0] _extra_dsp_reg_185;
    assign _extra_dsp_out_185 = $signed(data_in ^ 40'd186) * $signed(data_in ^ 40'd187);
    always @(posedge clk) _extra_dsp_reg_185 <= _extra_dsp_out_185;
    wire [40-1:0] _extra_dsp_out_186;
    reg [40-1:0] _extra_dsp_reg_186;
    assign _extra_dsp_out_186 = $signed(data_in ^ 40'd187) * $signed(data_in ^ 40'd188);
    always @(posedge clk) _extra_dsp_reg_186 <= _extra_dsp_out_186;
    wire [40-1:0] _extra_dsp_out_187;
    reg [40-1:0] _extra_dsp_reg_187;
    assign _extra_dsp_out_187 = $signed(data_in ^ 40'd188) * $signed(data_in ^ 40'd189);
    always @(posedge clk) _extra_dsp_reg_187 <= _extra_dsp_out_187;
    wire [40-1:0] _extra_dsp_out_188;
    reg [40-1:0] _extra_dsp_reg_188;
    assign _extra_dsp_out_188 = $signed(data_in ^ 40'd189) * $signed(data_in ^ 40'd190);
    always @(posedge clk) _extra_dsp_reg_188 <= _extra_dsp_out_188;
    wire [40-1:0] _extra_dsp_out_189;
    reg [40-1:0] _extra_dsp_reg_189;
    assign _extra_dsp_out_189 = $signed(data_in ^ 40'd190) * $signed(data_in ^ 40'd191);
    always @(posedge clk) _extra_dsp_reg_189 <= _extra_dsp_out_189;
    wire [40-1:0] _extra_dsp_out_190;
    reg [40-1:0] _extra_dsp_reg_190;
    assign _extra_dsp_out_190 = $signed(data_in ^ 40'd191) * $signed(data_in ^ 40'd192);
    always @(posedge clk) _extra_dsp_reg_190 <= _extra_dsp_out_190;
    wire [40-1:0] _extra_dsp_out_191;
    reg [40-1:0] _extra_dsp_reg_191;
    assign _extra_dsp_out_191 = $signed(data_in ^ 40'd192) * $signed(data_in ^ 40'd193);
    always @(posedge clk) _extra_dsp_reg_191 <= _extra_dsp_out_191;
    wire [40-1:0] _extra_dsp_out_192;
    reg [40-1:0] _extra_dsp_reg_192;
    assign _extra_dsp_out_192 = $signed(data_in ^ 40'd193) * $signed(data_in ^ 40'd194);
    always @(posedge clk) _extra_dsp_reg_192 <= _extra_dsp_out_192;
    wire [40-1:0] _extra_dsp_out_193;
    reg [40-1:0] _extra_dsp_reg_193;
    assign _extra_dsp_out_193 = $signed(data_in ^ 40'd194) * $signed(data_in ^ 40'd195);
    always @(posedge clk) _extra_dsp_reg_193 <= _extra_dsp_out_193;
    wire [40-1:0] _extra_dsp_out_194;
    reg [40-1:0] _extra_dsp_reg_194;
    assign _extra_dsp_out_194 = $signed(data_in ^ 40'd195) * $signed(data_in ^ 40'd196);
    always @(posedge clk) _extra_dsp_reg_194 <= _extra_dsp_out_194;
    wire [40-1:0] _extra_dsp_out_195;
    reg [40-1:0] _extra_dsp_reg_195;
    assign _extra_dsp_out_195 = $signed(data_in ^ 40'd196) * $signed(data_in ^ 40'd197);
    always @(posedge clk) _extra_dsp_reg_195 <= _extra_dsp_out_195;
    wire [40-1:0] _extra_dsp_out_196;
    reg [40-1:0] _extra_dsp_reg_196;
    assign _extra_dsp_out_196 = $signed(data_in ^ 40'd197) * $signed(data_in ^ 40'd198);
    always @(posedge clk) _extra_dsp_reg_196 <= _extra_dsp_out_196;
    wire [40-1:0] _extra_dsp_out_197;
    reg [40-1:0] _extra_dsp_reg_197;
    assign _extra_dsp_out_197 = $signed(data_in ^ 40'd198) * $signed(data_in ^ 40'd199);
    always @(posedge clk) _extra_dsp_reg_197 <= _extra_dsp_out_197;
    wire [40-1:0] _extra_dsp_out_198;
    reg [40-1:0] _extra_dsp_reg_198;
    assign _extra_dsp_out_198 = $signed(data_in ^ 40'd199) * $signed(data_in ^ 40'd200);
    always @(posedge clk) _extra_dsp_reg_198 <= _extra_dsp_out_198;
    wire [40-1:0] _extra_dsp_out_199;
    reg [40-1:0] _extra_dsp_reg_199;
    assign _extra_dsp_out_199 = $signed(data_in ^ 40'd200) * $signed(data_in ^ 40'd201);
    always @(posedge clk) _extra_dsp_reg_199 <= _extra_dsp_out_199;
    wire [40-1:0] _extra_dsp_out_200;
    reg [40-1:0] _extra_dsp_reg_200;
    assign _extra_dsp_out_200 = $signed(data_in ^ 40'd201) * $signed(data_in ^ 40'd202);
    always @(posedge clk) _extra_dsp_reg_200 <= _extra_dsp_out_200;
    wire [40-1:0] _extra_dsp_out_201;
    reg [40-1:0] _extra_dsp_reg_201;
    assign _extra_dsp_out_201 = $signed(data_in ^ 40'd202) * $signed(data_in ^ 40'd203);
    always @(posedge clk) _extra_dsp_reg_201 <= _extra_dsp_out_201;
    wire [40-1:0] _extra_dsp_out_202;
    reg [40-1:0] _extra_dsp_reg_202;
    assign _extra_dsp_out_202 = $signed(data_in ^ 40'd203) * $signed(data_in ^ 40'd204);
    always @(posedge clk) _extra_dsp_reg_202 <= _extra_dsp_out_202;
    wire [40-1:0] _extra_dsp_out_203;
    reg [40-1:0] _extra_dsp_reg_203;
    assign _extra_dsp_out_203 = $signed(data_in ^ 40'd204) * $signed(data_in ^ 40'd205);
    always @(posedge clk) _extra_dsp_reg_203 <= _extra_dsp_out_203;
    wire [40-1:0] _extra_dsp_out_204;
    reg [40-1:0] _extra_dsp_reg_204;
    assign _extra_dsp_out_204 = $signed(data_in ^ 40'd205) * $signed(data_in ^ 40'd206);
    always @(posedge clk) _extra_dsp_reg_204 <= _extra_dsp_out_204;
    wire [40-1:0] _extra_dsp_out_205;
    reg [40-1:0] _extra_dsp_reg_205;
    assign _extra_dsp_out_205 = $signed(data_in ^ 40'd206) * $signed(data_in ^ 40'd207);
    always @(posedge clk) _extra_dsp_reg_205 <= _extra_dsp_out_205;
    wire [40-1:0] _extra_dsp_out_206;
    reg [40-1:0] _extra_dsp_reg_206;
    assign _extra_dsp_out_206 = $signed(data_in ^ 40'd207) * $signed(data_in ^ 40'd208);
    always @(posedge clk) _extra_dsp_reg_206 <= _extra_dsp_out_206;
    wire [40-1:0] _extra_dsp_out_207;
    reg [40-1:0] _extra_dsp_reg_207;
    assign _extra_dsp_out_207 = $signed(data_in ^ 40'd208) * $signed(data_in ^ 40'd209);
    always @(posedge clk) _extra_dsp_reg_207 <= _extra_dsp_out_207;
    wire [40-1:0] _extra_dsp_out_208;
    reg [40-1:0] _extra_dsp_reg_208;
    assign _extra_dsp_out_208 = $signed(data_in ^ 40'd209) * $signed(data_in ^ 40'd210);
    always @(posedge clk) _extra_dsp_reg_208 <= _extra_dsp_out_208;
    wire [40-1:0] _extra_dsp_out_209;
    reg [40-1:0] _extra_dsp_reg_209;
    assign _extra_dsp_out_209 = $signed(data_in ^ 40'd210) * $signed(data_in ^ 40'd211);
    always @(posedge clk) _extra_dsp_reg_209 <= _extra_dsp_out_209;
    wire [40-1:0] _extra_dsp_out_210;
    reg [40-1:0] _extra_dsp_reg_210;
    assign _extra_dsp_out_210 = $signed(data_in ^ 40'd211) * $signed(data_in ^ 40'd212);
    always @(posedge clk) _extra_dsp_reg_210 <= _extra_dsp_out_210;
    wire [40-1:0] _extra_dsp_out_211;
    reg [40-1:0] _extra_dsp_reg_211;
    assign _extra_dsp_out_211 = $signed(data_in ^ 40'd212) * $signed(data_in ^ 40'd213);
    always @(posedge clk) _extra_dsp_reg_211 <= _extra_dsp_out_211;
    wire [40-1:0] _extra_dsp_out_212;
    reg [40-1:0] _extra_dsp_reg_212;
    assign _extra_dsp_out_212 = $signed(data_in ^ 40'd213) * $signed(data_in ^ 40'd214);
    always @(posedge clk) _extra_dsp_reg_212 <= _extra_dsp_out_212;
    wire [40-1:0] _extra_dsp_out_213;
    reg [40-1:0] _extra_dsp_reg_213;
    assign _extra_dsp_out_213 = $signed(data_in ^ 40'd214) * $signed(data_in ^ 40'd215);
    always @(posedge clk) _extra_dsp_reg_213 <= _extra_dsp_out_213;
    wire [40-1:0] _extra_dsp_out_214;
    reg [40-1:0] _extra_dsp_reg_214;
    assign _extra_dsp_out_214 = $signed(data_in ^ 40'd215) * $signed(data_in ^ 40'd216);
    always @(posedge clk) _extra_dsp_reg_214 <= _extra_dsp_out_214;
    wire [40-1:0] _extra_dsp_out_215;
    reg [40-1:0] _extra_dsp_reg_215;
    assign _extra_dsp_out_215 = $signed(data_in ^ 40'd216) * $signed(data_in ^ 40'd217);
    always @(posedge clk) _extra_dsp_reg_215 <= _extra_dsp_out_215;
    wire [40-1:0] _extra_dsp_out_216;
    reg [40-1:0] _extra_dsp_reg_216;
    assign _extra_dsp_out_216 = $signed(data_in ^ 40'd217) * $signed(data_in ^ 40'd218);
    always @(posedge clk) _extra_dsp_reg_216 <= _extra_dsp_out_216;
    wire [40-1:0] _extra_dsp_out_217;
    reg [40-1:0] _extra_dsp_reg_217;
    assign _extra_dsp_out_217 = $signed(data_in ^ 40'd218) * $signed(data_in ^ 40'd219);
    always @(posedge clk) _extra_dsp_reg_217 <= _extra_dsp_out_217;
    wire [40-1:0] _extra_dsp_out_218;
    reg [40-1:0] _extra_dsp_reg_218;
    assign _extra_dsp_out_218 = $signed(data_in ^ 40'd219) * $signed(data_in ^ 40'd220);
    always @(posedge clk) _extra_dsp_reg_218 <= _extra_dsp_out_218;
    wire [40-1:0] _extra_dsp_out_219;
    reg [40-1:0] _extra_dsp_reg_219;
    assign _extra_dsp_out_219 = $signed(data_in ^ 40'd220) * $signed(data_in ^ 40'd221);
    always @(posedge clk) _extra_dsp_reg_219 <= _extra_dsp_out_219;
    wire [40-1:0] _extra_dsp_out_220;
    reg [40-1:0] _extra_dsp_reg_220;
    assign _extra_dsp_out_220 = $signed(data_in ^ 40'd221) * $signed(data_in ^ 40'd222);
    always @(posedge clk) _extra_dsp_reg_220 <= _extra_dsp_out_220;
    wire [40-1:0] _extra_dsp_out_221;
    reg [40-1:0] _extra_dsp_reg_221;
    assign _extra_dsp_out_221 = $signed(data_in ^ 40'd222) * $signed(data_in ^ 40'd223);
    always @(posedge clk) _extra_dsp_reg_221 <= _extra_dsp_out_221;
    wire [40-1:0] _extra_dsp_out_222;
    reg [40-1:0] _extra_dsp_reg_222;
    assign _extra_dsp_out_222 = $signed(data_in ^ 40'd223) * $signed(data_in ^ 40'd224);
    always @(posedge clk) _extra_dsp_reg_222 <= _extra_dsp_out_222;
    wire [40-1:0] _extra_dsp_out_223;
    reg [40-1:0] _extra_dsp_reg_223;
    assign _extra_dsp_out_223 = $signed(data_in ^ 40'd224) * $signed(data_in ^ 40'd225);
    always @(posedge clk) _extra_dsp_reg_223 <= _extra_dsp_out_223;
    wire [40-1:0] _extra_dsp_out_224;
    reg [40-1:0] _extra_dsp_reg_224;
    assign _extra_dsp_out_224 = $signed(data_in ^ 40'd225) * $signed(data_in ^ 40'd226);
    always @(posedge clk) _extra_dsp_reg_224 <= _extra_dsp_out_224;
    wire [40-1:0] _extra_dsp_out_225;
    reg [40-1:0] _extra_dsp_reg_225;
    assign _extra_dsp_out_225 = $signed(data_in ^ 40'd226) * $signed(data_in ^ 40'd227);
    always @(posedge clk) _extra_dsp_reg_225 <= _extra_dsp_out_225;
    wire [40-1:0] _extra_dsp_out_226;
    reg [40-1:0] _extra_dsp_reg_226;
    assign _extra_dsp_out_226 = $signed(data_in ^ 40'd227) * $signed(data_in ^ 40'd228);
    always @(posedge clk) _extra_dsp_reg_226 <= _extra_dsp_out_226;
    wire [40-1:0] _extra_dsp_out_227;
    reg [40-1:0] _extra_dsp_reg_227;
    assign _extra_dsp_out_227 = $signed(data_in ^ 40'd228) * $signed(data_in ^ 40'd229);
    always @(posedge clk) _extra_dsp_reg_227 <= _extra_dsp_out_227;
    wire [40-1:0] _extra_dsp_out_228;
    reg [40-1:0] _extra_dsp_reg_228;
    assign _extra_dsp_out_228 = $signed(data_in ^ 40'd229) * $signed(data_in ^ 40'd230);
    always @(posedge clk) _extra_dsp_reg_228 <= _extra_dsp_out_228;
    wire [40-1:0] _extra_dsp_out_229;
    reg [40-1:0] _extra_dsp_reg_229;
    assign _extra_dsp_out_229 = $signed(data_in ^ 40'd230) * $signed(data_in ^ 40'd231);
    always @(posedge clk) _extra_dsp_reg_229 <= _extra_dsp_out_229;
    wire [40-1:0] _extra_dsp_out_230;
    reg [40-1:0] _extra_dsp_reg_230;
    assign _extra_dsp_out_230 = $signed(data_in ^ 40'd231) * $signed(data_in ^ 40'd232);
    always @(posedge clk) _extra_dsp_reg_230 <= _extra_dsp_out_230;
    wire [40-1:0] _extra_dsp_out_231;
    reg [40-1:0] _extra_dsp_reg_231;
    assign _extra_dsp_out_231 = $signed(data_in ^ 40'd232) * $signed(data_in ^ 40'd233);
    always @(posedge clk) _extra_dsp_reg_231 <= _extra_dsp_out_231;
    wire [40-1:0] _extra_dsp_out_232;
    reg [40-1:0] _extra_dsp_reg_232;
    assign _extra_dsp_out_232 = $signed(data_in ^ 40'd233) * $signed(data_in ^ 40'd234);
    always @(posedge clk) _extra_dsp_reg_232 <= _extra_dsp_out_232;
    wire [40-1:0] _extra_dsp_out_233;
    reg [40-1:0] _extra_dsp_reg_233;
    assign _extra_dsp_out_233 = $signed(data_in ^ 40'd234) * $signed(data_in ^ 40'd235);
    always @(posedge clk) _extra_dsp_reg_233 <= _extra_dsp_out_233;
    wire [40-1:0] _extra_dsp_out_234;
    reg [40-1:0] _extra_dsp_reg_234;
    assign _extra_dsp_out_234 = $signed(data_in ^ 40'd235) * $signed(data_in ^ 40'd236);
    always @(posedge clk) _extra_dsp_reg_234 <= _extra_dsp_out_234;
    wire [40-1:0] _extra_dsp_out_235;
    reg [40-1:0] _extra_dsp_reg_235;
    assign _extra_dsp_out_235 = $signed(data_in ^ 40'd236) * $signed(data_in ^ 40'd237);
    always @(posedge clk) _extra_dsp_reg_235 <= _extra_dsp_out_235;
    wire [40-1:0] _extra_dsp_out_236;
    reg [40-1:0] _extra_dsp_reg_236;
    assign _extra_dsp_out_236 = $signed(data_in ^ 40'd237) * $signed(data_in ^ 40'd238);
    always @(posedge clk) _extra_dsp_reg_236 <= _extra_dsp_out_236;
    wire [40-1:0] _extra_dsp_out_237;
    reg [40-1:0] _extra_dsp_reg_237;
    assign _extra_dsp_out_237 = $signed(data_in ^ 40'd238) * $signed(data_in ^ 40'd239);
    always @(posedge clk) _extra_dsp_reg_237 <= _extra_dsp_out_237;
    wire [40-1:0] _extra_dsp_out_238;
    reg [40-1:0] _extra_dsp_reg_238;
    assign _extra_dsp_out_238 = $signed(data_in ^ 40'd239) * $signed(data_in ^ 40'd240);
    always @(posedge clk) _extra_dsp_reg_238 <= _extra_dsp_out_238;
    wire [40-1:0] _extra_dsp_out_239;
    reg [40-1:0] _extra_dsp_reg_239;
    assign _extra_dsp_out_239 = $signed(data_in ^ 40'd240) * $signed(data_in ^ 40'd241);
    always @(posedge clk) _extra_dsp_reg_239 <= _extra_dsp_out_239;
    wire [40-1:0] _extra_dsp_out_240;
    reg [40-1:0] _extra_dsp_reg_240;
    assign _extra_dsp_out_240 = $signed(data_in ^ 40'd241) * $signed(data_in ^ 40'd242);
    always @(posedge clk) _extra_dsp_reg_240 <= _extra_dsp_out_240;
    wire [40-1:0] _extra_dsp_out_241;
    reg [40-1:0] _extra_dsp_reg_241;
    assign _extra_dsp_out_241 = $signed(data_in ^ 40'd242) * $signed(data_in ^ 40'd243);
    always @(posedge clk) _extra_dsp_reg_241 <= _extra_dsp_out_241;
    wire [40-1:0] _extra_dsp_out_242;
    reg [40-1:0] _extra_dsp_reg_242;
    assign _extra_dsp_out_242 = $signed(data_in ^ 40'd243) * $signed(data_in ^ 40'd244);
    always @(posedge clk) _extra_dsp_reg_242 <= _extra_dsp_out_242;
    wire [40-1:0] _extra_dsp_out_243;
    reg [40-1:0] _extra_dsp_reg_243;
    assign _extra_dsp_out_243 = $signed(data_in ^ 40'd244) * $signed(data_in ^ 40'd245);
    always @(posedge clk) _extra_dsp_reg_243 <= _extra_dsp_out_243;
    wire [40-1:0] _extra_dsp_out_244;
    reg [40-1:0] _extra_dsp_reg_244;
    assign _extra_dsp_out_244 = $signed(data_in ^ 40'd245) * $signed(data_in ^ 40'd246);
    always @(posedge clk) _extra_dsp_reg_244 <= _extra_dsp_out_244;
    wire [40-1:0] _extra_dsp_out_245;
    reg [40-1:0] _extra_dsp_reg_245;
    assign _extra_dsp_out_245 = $signed(data_in ^ 40'd246) * $signed(data_in ^ 40'd247);
    always @(posedge clk) _extra_dsp_reg_245 <= _extra_dsp_out_245;
    wire [40-1:0] _extra_dsp_out_246;
    reg [40-1:0] _extra_dsp_reg_246;
    assign _extra_dsp_out_246 = $signed(data_in ^ 40'd247) * $signed(data_in ^ 40'd248);
    always @(posedge clk) _extra_dsp_reg_246 <= _extra_dsp_out_246;
    wire [40-1:0] _extra_dsp_out_247;
    reg [40-1:0] _extra_dsp_reg_247;
    assign _extra_dsp_out_247 = $signed(data_in ^ 40'd248) * $signed(data_in ^ 40'd249);
    always @(posedge clk) _extra_dsp_reg_247 <= _extra_dsp_out_247;
    wire [40-1:0] _extra_dsp_out_248;
    reg [40-1:0] _extra_dsp_reg_248;
    assign _extra_dsp_out_248 = $signed(data_in ^ 40'd249) * $signed(data_in ^ 40'd250);
    always @(posedge clk) _extra_dsp_reg_248 <= _extra_dsp_out_248;
    wire [40-1:0] _extra_dsp_out_249;
    reg [40-1:0] _extra_dsp_reg_249;
    assign _extra_dsp_out_249 = $signed(data_in ^ 40'd250) * $signed(data_in ^ 40'd251);
    always @(posedge clk) _extra_dsp_reg_249 <= _extra_dsp_out_249;
    wire [40-1:0] _extra_dsp_out_250;
    reg [40-1:0] _extra_dsp_reg_250;
    assign _extra_dsp_out_250 = $signed(data_in ^ 40'd251) * $signed(data_in ^ 40'd252);
    always @(posedge clk) _extra_dsp_reg_250 <= _extra_dsp_out_250;
    wire [40-1:0] _extra_dsp_out_251;
    reg [40-1:0] _extra_dsp_reg_251;
    assign _extra_dsp_out_251 = $signed(data_in ^ 40'd252) * $signed(data_in ^ 40'd253);
    always @(posedge clk) _extra_dsp_reg_251 <= _extra_dsp_out_251;
    wire [40-1:0] _extra_dsp_out_252;
    reg [40-1:0] _extra_dsp_reg_252;
    assign _extra_dsp_out_252 = $signed(data_in ^ 40'd253) * $signed(data_in ^ 40'd254);
    always @(posedge clk) _extra_dsp_reg_252 <= _extra_dsp_out_252;
    wire [40-1:0] _extra_dsp_out_253;
    reg [40-1:0] _extra_dsp_reg_253;
    assign _extra_dsp_out_253 = $signed(data_in ^ 40'd254) * $signed(data_in ^ 40'd255);
    always @(posedge clk) _extra_dsp_reg_253 <= _extra_dsp_out_253;
    wire [40-1:0] _extra_dsp_out_254;
    reg [40-1:0] _extra_dsp_reg_254;
    assign _extra_dsp_out_254 = $signed(data_in ^ 40'd255) * $signed(data_in ^ 40'd256);
    always @(posedge clk) _extra_dsp_reg_254 <= _extra_dsp_out_254;
    wire [40-1:0] _extra_dsp_out_255;
    reg [40-1:0] _extra_dsp_reg_255;
    assign _extra_dsp_out_255 = $signed(data_in ^ 40'd256) * $signed(data_in ^ 40'd257);
    always @(posedge clk) _extra_dsp_reg_255 <= _extra_dsp_out_255;
    wire [40-1:0] _extra_dsp_out_256;
    reg [40-1:0] _extra_dsp_reg_256;
    assign _extra_dsp_out_256 = $signed(data_in ^ 40'd257) * $signed(data_in ^ 40'd258);
    always @(posedge clk) _extra_dsp_reg_256 <= _extra_dsp_out_256;
    wire [40-1:0] _extra_dsp_out_257;
    reg [40-1:0] _extra_dsp_reg_257;
    assign _extra_dsp_out_257 = $signed(data_in ^ 40'd258) * $signed(data_in ^ 40'd259);
    always @(posedge clk) _extra_dsp_reg_257 <= _extra_dsp_out_257;
    wire [40-1:0] _extra_dsp_out_258;
    reg [40-1:0] _extra_dsp_reg_258;
    assign _extra_dsp_out_258 = $signed(data_in ^ 40'd259) * $signed(data_in ^ 40'd260);
    always @(posedge clk) _extra_dsp_reg_258 <= _extra_dsp_out_258;
    wire [40-1:0] _extra_dsp_out_259;
    reg [40-1:0] _extra_dsp_reg_259;
    assign _extra_dsp_out_259 = $signed(data_in ^ 40'd260) * $signed(data_in ^ 40'd261);
    always @(posedge clk) _extra_dsp_reg_259 <= _extra_dsp_out_259;
    wire [40-1:0] _extra_dsp_out_260;
    reg [40-1:0] _extra_dsp_reg_260;
    assign _extra_dsp_out_260 = $signed(data_in ^ 40'd261) * $signed(data_in ^ 40'd262);
    always @(posedge clk) _extra_dsp_reg_260 <= _extra_dsp_out_260;
    wire [40-1:0] _extra_dsp_out_261;
    reg [40-1:0] _extra_dsp_reg_261;
    assign _extra_dsp_out_261 = $signed(data_in ^ 40'd262) * $signed(data_in ^ 40'd263);
    always @(posedge clk) _extra_dsp_reg_261 <= _extra_dsp_out_261;
    wire [40-1:0] _extra_dsp_out_262;
    reg [40-1:0] _extra_dsp_reg_262;
    assign _extra_dsp_out_262 = $signed(data_in ^ 40'd263) * $signed(data_in ^ 40'd264);
    always @(posedge clk) _extra_dsp_reg_262 <= _extra_dsp_out_262;
    wire [40-1:0] _extra_dsp_out_263;
    reg [40-1:0] _extra_dsp_reg_263;
    assign _extra_dsp_out_263 = $signed(data_in ^ 40'd264) * $signed(data_in ^ 40'd265);
    always @(posedge clk) _extra_dsp_reg_263 <= _extra_dsp_out_263;
    wire [40-1:0] _extra_dsp_out_264;
    reg [40-1:0] _extra_dsp_reg_264;
    assign _extra_dsp_out_264 = $signed(data_in ^ 40'd265) * $signed(data_in ^ 40'd266);
    always @(posedge clk) _extra_dsp_reg_264 <= _extra_dsp_out_264;
    wire [40-1:0] _extra_dsp_out_265;
    reg [40-1:0] _extra_dsp_reg_265;
    assign _extra_dsp_out_265 = $signed(data_in ^ 40'd266) * $signed(data_in ^ 40'd267);
    always @(posedge clk) _extra_dsp_reg_265 <= _extra_dsp_out_265;
    wire [40-1:0] _extra_dsp_out_266;
    reg [40-1:0] _extra_dsp_reg_266;
    assign _extra_dsp_out_266 = $signed(data_in ^ 40'd267) * $signed(data_in ^ 40'd268);
    always @(posedge clk) _extra_dsp_reg_266 <= _extra_dsp_out_266;
    wire [40-1:0] _extra_dsp_out_267;
    reg [40-1:0] _extra_dsp_reg_267;
    assign _extra_dsp_out_267 = $signed(data_in ^ 40'd268) * $signed(data_in ^ 40'd269);
    always @(posedge clk) _extra_dsp_reg_267 <= _extra_dsp_out_267;
    wire [40-1:0] _extra_dsp_out_268;
    reg [40-1:0] _extra_dsp_reg_268;
    assign _extra_dsp_out_268 = $signed(data_in ^ 40'd269) * $signed(data_in ^ 40'd270);
    always @(posedge clk) _extra_dsp_reg_268 <= _extra_dsp_out_268;
    wire [40-1:0] _extra_dsp_out_269;
    reg [40-1:0] _extra_dsp_reg_269;
    assign _extra_dsp_out_269 = $signed(data_in ^ 40'd270) * $signed(data_in ^ 40'd271);
    always @(posedge clk) _extra_dsp_reg_269 <= _extra_dsp_out_269;
    wire [40-1:0] _extra_dsp_out_270;
    reg [40-1:0] _extra_dsp_reg_270;
    assign _extra_dsp_out_270 = $signed(data_in ^ 40'd271) * $signed(data_in ^ 40'd272);
    always @(posedge clk) _extra_dsp_reg_270 <= _extra_dsp_out_270;
    wire [40-1:0] _extra_dsp_out_271;
    reg [40-1:0] _extra_dsp_reg_271;
    assign _extra_dsp_out_271 = $signed(data_in ^ 40'd272) * $signed(data_in ^ 40'd273);
    always @(posedge clk) _extra_dsp_reg_271 <= _extra_dsp_out_271;
    wire [40-1:0] _extra_dsp_out_272;
    reg [40-1:0] _extra_dsp_reg_272;
    assign _extra_dsp_out_272 = $signed(data_in ^ 40'd273) * $signed(data_in ^ 40'd274);
    always @(posedge clk) _extra_dsp_reg_272 <= _extra_dsp_out_272;
    wire [40-1:0] _extra_dsp_out_273;
    reg [40-1:0] _extra_dsp_reg_273;
    assign _extra_dsp_out_273 = $signed(data_in ^ 40'd274) * $signed(data_in ^ 40'd275);
    always @(posedge clk) _extra_dsp_reg_273 <= _extra_dsp_out_273;
    wire [40-1:0] _extra_dsp_out_274;
    reg [40-1:0] _extra_dsp_reg_274;
    assign _extra_dsp_out_274 = $signed(data_in ^ 40'd275) * $signed(data_in ^ 40'd276);
    always @(posedge clk) _extra_dsp_reg_274 <= _extra_dsp_out_274;
    wire [40-1:0] _extra_dsp_out_275;
    reg [40-1:0] _extra_dsp_reg_275;
    assign _extra_dsp_out_275 = $signed(data_in ^ 40'd276) * $signed(data_in ^ 40'd277);
    always @(posedge clk) _extra_dsp_reg_275 <= _extra_dsp_out_275;
    wire [40-1:0] _extra_dsp_out_276;
    reg [40-1:0] _extra_dsp_reg_276;
    assign _extra_dsp_out_276 = $signed(data_in ^ 40'd277) * $signed(data_in ^ 40'd278);
    always @(posedge clk) _extra_dsp_reg_276 <= _extra_dsp_out_276;
    wire [40-1:0] _extra_dsp_out_277;
    reg [40-1:0] _extra_dsp_reg_277;
    assign _extra_dsp_out_277 = $signed(data_in ^ 40'd278) * $signed(data_in ^ 40'd279);
    always @(posedge clk) _extra_dsp_reg_277 <= _extra_dsp_out_277;
    wire [40-1:0] _extra_dsp_out_278;
    reg [40-1:0] _extra_dsp_reg_278;
    assign _extra_dsp_out_278 = $signed(data_in ^ 40'd279) * $signed(data_in ^ 40'd280);
    always @(posedge clk) _extra_dsp_reg_278 <= _extra_dsp_out_278;
    wire [40-1:0] _extra_dsp_out_279;
    reg [40-1:0] _extra_dsp_reg_279;
    assign _extra_dsp_out_279 = $signed(data_in ^ 40'd280) * $signed(data_in ^ 40'd281);
    always @(posedge clk) _extra_dsp_reg_279 <= _extra_dsp_out_279;
    wire [40-1:0] _extra_dsp_out_280;
    reg [40-1:0] _extra_dsp_reg_280;
    assign _extra_dsp_out_280 = $signed(data_in ^ 40'd281) * $signed(data_in ^ 40'd282);
    always @(posedge clk) _extra_dsp_reg_280 <= _extra_dsp_out_280;
    wire [40-1:0] _extra_dsp_out_281;
    reg [40-1:0] _extra_dsp_reg_281;
    assign _extra_dsp_out_281 = $signed(data_in ^ 40'd282) * $signed(data_in ^ 40'd283);
    always @(posedge clk) _extra_dsp_reg_281 <= _extra_dsp_out_281;
    wire [40-1:0] _extra_dsp_out_282;
    reg [40-1:0] _extra_dsp_reg_282;
    assign _extra_dsp_out_282 = $signed(data_in ^ 40'd283) * $signed(data_in ^ 40'd284);
    always @(posedge clk) _extra_dsp_reg_282 <= _extra_dsp_out_282;
    wire [40-1:0] _extra_dsp_out_283;
    reg [40-1:0] _extra_dsp_reg_283;
    assign _extra_dsp_out_283 = $signed(data_in ^ 40'd284) * $signed(data_in ^ 40'd285);
    always @(posedge clk) _extra_dsp_reg_283 <= _extra_dsp_out_283;
    wire [40-1:0] _extra_dsp_out_284;
    reg [40-1:0] _extra_dsp_reg_284;
    assign _extra_dsp_out_284 = $signed(data_in ^ 40'd285) * $signed(data_in ^ 40'd286);
    always @(posedge clk) _extra_dsp_reg_284 <= _extra_dsp_out_284;
    wire [40-1:0] _extra_dsp_out_285;
    reg [40-1:0] _extra_dsp_reg_285;
    assign _extra_dsp_out_285 = $signed(data_in ^ 40'd286) * $signed(data_in ^ 40'd287);
    always @(posedge clk) _extra_dsp_reg_285 <= _extra_dsp_out_285;
    wire [40-1:0] _extra_dsp_out_286;
    reg [40-1:0] _extra_dsp_reg_286;
    assign _extra_dsp_out_286 = $signed(data_in ^ 40'd287) * $signed(data_in ^ 40'd288);
    always @(posedge clk) _extra_dsp_reg_286 <= _extra_dsp_out_286;
    wire [40-1:0] _extra_dsp_out_287;
    reg [40-1:0] _extra_dsp_reg_287;
    assign _extra_dsp_out_287 = $signed(data_in ^ 40'd288) * $signed(data_in ^ 40'd289);
    always @(posedge clk) _extra_dsp_reg_287 <= _extra_dsp_out_287;
    wire [40-1:0] _extra_dsp_out_288;
    reg [40-1:0] _extra_dsp_reg_288;
    assign _extra_dsp_out_288 = $signed(data_in ^ 40'd289) * $signed(data_in ^ 40'd290);
    always @(posedge clk) _extra_dsp_reg_288 <= _extra_dsp_out_288;
    wire [40-1:0] _extra_dsp_out_289;
    reg [40-1:0] _extra_dsp_reg_289;
    assign _extra_dsp_out_289 = $signed(data_in ^ 40'd290) * $signed(data_in ^ 40'd291);
    always @(posedge clk) _extra_dsp_reg_289 <= _extra_dsp_out_289;
    wire [40-1:0] _extra_dsp_out_290;
    reg [40-1:0] _extra_dsp_reg_290;
    assign _extra_dsp_out_290 = $signed(data_in ^ 40'd291) * $signed(data_in ^ 40'd292);
    always @(posedge clk) _extra_dsp_reg_290 <= _extra_dsp_out_290;
    wire [40-1:0] _extra_dsp_out_291;
    reg [40-1:0] _extra_dsp_reg_291;
    assign _extra_dsp_out_291 = $signed(data_in ^ 40'd292) * $signed(data_in ^ 40'd293);
    always @(posedge clk) _extra_dsp_reg_291 <= _extra_dsp_out_291;
    wire [40-1:0] _extra_dsp_out_292;
    reg [40-1:0] _extra_dsp_reg_292;
    assign _extra_dsp_out_292 = $signed(data_in ^ 40'd293) * $signed(data_in ^ 40'd294);
    always @(posedge clk) _extra_dsp_reg_292 <= _extra_dsp_out_292;
    wire [40-1:0] _extra_dsp_out_293;
    reg [40-1:0] _extra_dsp_reg_293;
    assign _extra_dsp_out_293 = $signed(data_in ^ 40'd294) * $signed(data_in ^ 40'd295);
    always @(posedge clk) _extra_dsp_reg_293 <= _extra_dsp_out_293;
    wire [40-1:0] _extra_dsp_out_294;
    reg [40-1:0] _extra_dsp_reg_294;
    assign _extra_dsp_out_294 = $signed(data_in ^ 40'd295) * $signed(data_in ^ 40'd296);
    always @(posedge clk) _extra_dsp_reg_294 <= _extra_dsp_out_294;
    wire [40-1:0] _extra_dsp_out_295;
    reg [40-1:0] _extra_dsp_reg_295;
    assign _extra_dsp_out_295 = $signed(data_in ^ 40'd296) * $signed(data_in ^ 40'd297);
    always @(posedge clk) _extra_dsp_reg_295 <= _extra_dsp_out_295;
    wire [40-1:0] _extra_dsp_out_296;
    reg [40-1:0] _extra_dsp_reg_296;
    assign _extra_dsp_out_296 = $signed(data_in ^ 40'd297) * $signed(data_in ^ 40'd298);
    always @(posedge clk) _extra_dsp_reg_296 <= _extra_dsp_out_296;
    wire [40-1:0] _extra_dsp_out_297;
    reg [40-1:0] _extra_dsp_reg_297;
    assign _extra_dsp_out_297 = $signed(data_in ^ 40'd298) * $signed(data_in ^ 40'd299);
    always @(posedge clk) _extra_dsp_reg_297 <= _extra_dsp_out_297;
    wire [40-1:0] _extra_dsp_out_298;
    reg [40-1:0] _extra_dsp_reg_298;
    assign _extra_dsp_out_298 = $signed(data_in ^ 40'd299) * $signed(data_in ^ 40'd300);
    always @(posedge clk) _extra_dsp_reg_298 <= _extra_dsp_out_298;
    wire [40-1:0] _extra_dsp_out_299;
    reg [40-1:0] _extra_dsp_reg_299;
    assign _extra_dsp_out_299 = $signed(data_in ^ 40'd300) * $signed(data_in ^ 40'd301);
    always @(posedge clk) _extra_dsp_reg_299 <= _extra_dsp_out_299;
    wire [40-1:0] _extra_dsp_out_300;
    reg [40-1:0] _extra_dsp_reg_300;
    assign _extra_dsp_out_300 = $signed(data_in ^ 40'd301) * $signed(data_in ^ 40'd302);
    always @(posedge clk) _extra_dsp_reg_300 <= _extra_dsp_out_300;
    wire [40-1:0] _extra_dsp_out_301;
    reg [40-1:0] _extra_dsp_reg_301;
    assign _extra_dsp_out_301 = $signed(data_in ^ 40'd302) * $signed(data_in ^ 40'd303);
    always @(posedge clk) _extra_dsp_reg_301 <= _extra_dsp_out_301;
    wire [40-1:0] _extra_dsp_out_302;
    reg [40-1:0] _extra_dsp_reg_302;
    assign _extra_dsp_out_302 = $signed(data_in ^ 40'd303) * $signed(data_in ^ 40'd304);
    always @(posedge clk) _extra_dsp_reg_302 <= _extra_dsp_out_302;
    wire [40-1:0] _extra_dsp_out_303;
    reg [40-1:0] _extra_dsp_reg_303;
    assign _extra_dsp_out_303 = $signed(data_in ^ 40'd304) * $signed(data_in ^ 40'd305);
    always @(posedge clk) _extra_dsp_reg_303 <= _extra_dsp_out_303;
    wire [40-1:0] _extra_dsp_out_304;
    reg [40-1:0] _extra_dsp_reg_304;
    assign _extra_dsp_out_304 = $signed(data_in ^ 40'd305) * $signed(data_in ^ 40'd306);
    always @(posedge clk) _extra_dsp_reg_304 <= _extra_dsp_out_304;
    wire [40-1:0] _extra_dsp_out_305;
    reg [40-1:0] _extra_dsp_reg_305;
    assign _extra_dsp_out_305 = $signed(data_in ^ 40'd306) * $signed(data_in ^ 40'd307);
    always @(posedge clk) _extra_dsp_reg_305 <= _extra_dsp_out_305;
    wire [40-1:0] _extra_dsp_out_306;
    reg [40-1:0] _extra_dsp_reg_306;
    assign _extra_dsp_out_306 = $signed(data_in ^ 40'd307) * $signed(data_in ^ 40'd308);
    always @(posedge clk) _extra_dsp_reg_306 <= _extra_dsp_out_306;
    wire [40-1:0] _extra_dsp_out_307;
    reg [40-1:0] _extra_dsp_reg_307;
    assign _extra_dsp_out_307 = $signed(data_in ^ 40'd308) * $signed(data_in ^ 40'd309);
    always @(posedge clk) _extra_dsp_reg_307 <= _extra_dsp_out_307;
    wire [40-1:0] _extra_dsp_out_308;
    reg [40-1:0] _extra_dsp_reg_308;
    assign _extra_dsp_out_308 = $signed(data_in ^ 40'd309) * $signed(data_in ^ 40'd310);
    always @(posedge clk) _extra_dsp_reg_308 <= _extra_dsp_out_308;
    wire [40-1:0] _extra_dsp_out_309;
    reg [40-1:0] _extra_dsp_reg_309;
    assign _extra_dsp_out_309 = $signed(data_in ^ 40'd310) * $signed(data_in ^ 40'd311);
    always @(posedge clk) _extra_dsp_reg_309 <= _extra_dsp_out_309;
    wire [40-1:0] _extra_dsp_out_310;
    reg [40-1:0] _extra_dsp_reg_310;
    assign _extra_dsp_out_310 = $signed(data_in ^ 40'd311) * $signed(data_in ^ 40'd312);
    always @(posedge clk) _extra_dsp_reg_310 <= _extra_dsp_out_310;
    wire [40-1:0] _extra_dsp_out_311;
    reg [40-1:0] _extra_dsp_reg_311;
    assign _extra_dsp_out_311 = $signed(data_in ^ 40'd312) * $signed(data_in ^ 40'd313);
    always @(posedge clk) _extra_dsp_reg_311 <= _extra_dsp_out_311;
    wire [40-1:0] _extra_dsp_out_312;
    reg [40-1:0] _extra_dsp_reg_312;
    assign _extra_dsp_out_312 = $signed(data_in ^ 40'd313) * $signed(data_in ^ 40'd314);
    always @(posedge clk) _extra_dsp_reg_312 <= _extra_dsp_out_312;
    wire [40-1:0] _extra_dsp_out_313;
    reg [40-1:0] _extra_dsp_reg_313;
    assign _extra_dsp_out_313 = $signed(data_in ^ 40'd314) * $signed(data_in ^ 40'd315);
    always @(posedge clk) _extra_dsp_reg_313 <= _extra_dsp_out_313;
    wire [40-1:0] _extra_dsp_out_314;
    reg [40-1:0] _extra_dsp_reg_314;
    assign _extra_dsp_out_314 = $signed(data_in ^ 40'd315) * $signed(data_in ^ 40'd316);
    always @(posedge clk) _extra_dsp_reg_314 <= _extra_dsp_out_314;
    wire [40-1:0] _extra_dsp_out_315;
    reg [40-1:0] _extra_dsp_reg_315;
    assign _extra_dsp_out_315 = $signed(data_in ^ 40'd316) * $signed(data_in ^ 40'd317);
    always @(posedge clk) _extra_dsp_reg_315 <= _extra_dsp_out_315;
    wire [40-1:0] _extra_dsp_out_316;
    reg [40-1:0] _extra_dsp_reg_316;
    assign _extra_dsp_out_316 = $signed(data_in ^ 40'd317) * $signed(data_in ^ 40'd318);
    always @(posedge clk) _extra_dsp_reg_316 <= _extra_dsp_out_316;
    wire [40-1:0] _extra_dsp_out_317;
    reg [40-1:0] _extra_dsp_reg_317;
    assign _extra_dsp_out_317 = $signed(data_in ^ 40'd318) * $signed(data_in ^ 40'd319);
    always @(posedge clk) _extra_dsp_reg_317 <= _extra_dsp_out_317;
    wire [40-1:0] _extra_dsp_out_318;
    reg [40-1:0] _extra_dsp_reg_318;
    assign _extra_dsp_out_318 = $signed(data_in ^ 40'd319) * $signed(data_in ^ 40'd320);
    always @(posedge clk) _extra_dsp_reg_318 <= _extra_dsp_out_318;
    wire [40-1:0] _extra_dsp_out_319;
    reg [40-1:0] _extra_dsp_reg_319;
    assign _extra_dsp_out_319 = $signed(data_in ^ 40'd320) * $signed(data_in ^ 40'd321);
    always @(posedge clk) _extra_dsp_reg_319 <= _extra_dsp_out_319;
    wire [40-1:0] _extra_dsp_out_320;
    reg [40-1:0] _extra_dsp_reg_320;
    assign _extra_dsp_out_320 = $signed(data_in ^ 40'd321) * $signed(data_in ^ 40'd322);
    always @(posedge clk) _extra_dsp_reg_320 <= _extra_dsp_out_320;
    wire [40-1:0] _extra_dsp_out_321;
    reg [40-1:0] _extra_dsp_reg_321;
    assign _extra_dsp_out_321 = $signed(data_in ^ 40'd322) * $signed(data_in ^ 40'd323);
    always @(posedge clk) _extra_dsp_reg_321 <= _extra_dsp_out_321;
    wire [40-1:0] _extra_dsp_out_322;
    reg [40-1:0] _extra_dsp_reg_322;
    assign _extra_dsp_out_322 = $signed(data_in ^ 40'd323) * $signed(data_in ^ 40'd324);
    always @(posedge clk) _extra_dsp_reg_322 <= _extra_dsp_out_322;
    wire [40-1:0] _extra_dsp_out_323;
    reg [40-1:0] _extra_dsp_reg_323;
    assign _extra_dsp_out_323 = $signed(data_in ^ 40'd324) * $signed(data_in ^ 40'd325);
    always @(posedge clk) _extra_dsp_reg_323 <= _extra_dsp_out_323;
    wire [40-1:0] _extra_dsp_out_324;
    reg [40-1:0] _extra_dsp_reg_324;
    assign _extra_dsp_out_324 = $signed(data_in ^ 40'd325) * $signed(data_in ^ 40'd326);
    always @(posedge clk) _extra_dsp_reg_324 <= _extra_dsp_out_324;
    wire [40-1:0] _extra_dsp_out_325;
    reg [40-1:0] _extra_dsp_reg_325;
    assign _extra_dsp_out_325 = $signed(data_in ^ 40'd326) * $signed(data_in ^ 40'd327);
    always @(posedge clk) _extra_dsp_reg_325 <= _extra_dsp_out_325;
    wire [40-1:0] _extra_dsp_out_326;
    reg [40-1:0] _extra_dsp_reg_326;
    assign _extra_dsp_out_326 = $signed(data_in ^ 40'd327) * $signed(data_in ^ 40'd328);
    always @(posedge clk) _extra_dsp_reg_326 <= _extra_dsp_out_326;
    wire [40-1:0] _extra_dsp_out_327;
    reg [40-1:0] _extra_dsp_reg_327;
    assign _extra_dsp_out_327 = $signed(data_in ^ 40'd328) * $signed(data_in ^ 40'd329);
    always @(posedge clk) _extra_dsp_reg_327 <= _extra_dsp_out_327;
    wire [40-1:0] _extra_dsp_out_328;
    reg [40-1:0] _extra_dsp_reg_328;
    assign _extra_dsp_out_328 = $signed(data_in ^ 40'd329) * $signed(data_in ^ 40'd330);
    always @(posedge clk) _extra_dsp_reg_328 <= _extra_dsp_out_328;
    wire [40-1:0] _extra_dsp_out_329;
    reg [40-1:0] _extra_dsp_reg_329;
    assign _extra_dsp_out_329 = $signed(data_in ^ 40'd330) * $signed(data_in ^ 40'd331);
    always @(posedge clk) _extra_dsp_reg_329 <= _extra_dsp_out_329;
    wire [40-1:0] _extra_dsp_out_330;
    reg [40-1:0] _extra_dsp_reg_330;
    assign _extra_dsp_out_330 = $signed(data_in ^ 40'd331) * $signed(data_in ^ 40'd332);
    always @(posedge clk) _extra_dsp_reg_330 <= _extra_dsp_out_330;
    wire [40-1:0] _extra_dsp_out_331;
    reg [40-1:0] _extra_dsp_reg_331;
    assign _extra_dsp_out_331 = $signed(data_in ^ 40'd332) * $signed(data_in ^ 40'd333);
    always @(posedge clk) _extra_dsp_reg_331 <= _extra_dsp_out_331;
    wire [40-1:0] _extra_dsp_out_332;
    reg [40-1:0] _extra_dsp_reg_332;
    assign _extra_dsp_out_332 = $signed(data_in ^ 40'd333) * $signed(data_in ^ 40'd334);
    always @(posedge clk) _extra_dsp_reg_332 <= _extra_dsp_out_332;
    assign data_out = data_b1_ln_ffn ^ {39'b0, _extra_dsp_reg_0[0] ^ _extra_dsp_reg_1[0] ^ _extra_dsp_reg_2[0] ^ _extra_dsp_reg_3[0] ^ _extra_dsp_reg_4[0] ^ _extra_dsp_reg_5[0] ^ _extra_dsp_reg_6[0] ^ _extra_dsp_reg_7[0] ^ _extra_dsp_reg_8[0] ^ _extra_dsp_reg_9[0] ^ _extra_dsp_reg_10[0] ^ _extra_dsp_reg_11[0] ^ _extra_dsp_reg_12[0] ^ _extra_dsp_reg_13[0] ^ _extra_dsp_reg_14[0] ^ _extra_dsp_reg_15[0] ^ _extra_dsp_reg_16[0] ^ _extra_dsp_reg_17[0] ^ _extra_dsp_reg_18[0] ^ _extra_dsp_reg_19[0] ^ _extra_dsp_reg_20[0] ^ _extra_dsp_reg_21[0] ^ _extra_dsp_reg_22[0] ^ _extra_dsp_reg_23[0] ^ _extra_dsp_reg_24[0] ^ _extra_dsp_reg_25[0] ^ _extra_dsp_reg_26[0] ^ _extra_dsp_reg_27[0] ^ _extra_dsp_reg_28[0] ^ _extra_dsp_reg_29[0] ^ _extra_dsp_reg_30[0] ^ _extra_dsp_reg_31[0] ^ _extra_dsp_reg_32[0] ^ _extra_dsp_reg_33[0] ^ _extra_dsp_reg_34[0] ^ _extra_dsp_reg_35[0] ^ _extra_dsp_reg_36[0] ^ _extra_dsp_reg_37[0] ^ _extra_dsp_reg_38[0] ^ _extra_dsp_reg_39[0] ^ _extra_dsp_reg_40[0] ^ _extra_dsp_reg_41[0] ^ _extra_dsp_reg_42[0] ^ _extra_dsp_reg_43[0] ^ _extra_dsp_reg_44[0] ^ _extra_dsp_reg_45[0] ^ _extra_dsp_reg_46[0] ^ _extra_dsp_reg_47[0] ^ _extra_dsp_reg_48[0] ^ _extra_dsp_reg_49[0] ^ _extra_dsp_reg_50[0] ^ _extra_dsp_reg_51[0] ^ _extra_dsp_reg_52[0] ^ _extra_dsp_reg_53[0] ^ _extra_dsp_reg_54[0] ^ _extra_dsp_reg_55[0] ^ _extra_dsp_reg_56[0] ^ _extra_dsp_reg_57[0] ^ _extra_dsp_reg_58[0] ^ _extra_dsp_reg_59[0] ^ _extra_dsp_reg_60[0] ^ _extra_dsp_reg_61[0] ^ _extra_dsp_reg_62[0] ^ _extra_dsp_reg_63[0] ^ _extra_dsp_reg_64[0] ^ _extra_dsp_reg_65[0] ^ _extra_dsp_reg_66[0] ^ _extra_dsp_reg_67[0] ^ _extra_dsp_reg_68[0] ^ _extra_dsp_reg_69[0] ^ _extra_dsp_reg_70[0] ^ _extra_dsp_reg_71[0] ^ _extra_dsp_reg_72[0] ^ _extra_dsp_reg_73[0] ^ _extra_dsp_reg_74[0] ^ _extra_dsp_reg_75[0] ^ _extra_dsp_reg_76[0] ^ _extra_dsp_reg_77[0] ^ _extra_dsp_reg_78[0] ^ _extra_dsp_reg_79[0] ^ _extra_dsp_reg_80[0] ^ _extra_dsp_reg_81[0] ^ _extra_dsp_reg_82[0] ^ _extra_dsp_reg_83[0] ^ _extra_dsp_reg_84[0] ^ _extra_dsp_reg_85[0] ^ _extra_dsp_reg_86[0] ^ _extra_dsp_reg_87[0] ^ _extra_dsp_reg_88[0] ^ _extra_dsp_reg_89[0] ^ _extra_dsp_reg_90[0] ^ _extra_dsp_reg_91[0] ^ _extra_dsp_reg_92[0] ^ _extra_dsp_reg_93[0] ^ _extra_dsp_reg_94[0] ^ _extra_dsp_reg_95[0] ^ _extra_dsp_reg_96[0] ^ _extra_dsp_reg_97[0] ^ _extra_dsp_reg_98[0] ^ _extra_dsp_reg_99[0] ^ _extra_dsp_reg_100[0] ^ _extra_dsp_reg_101[0] ^ _extra_dsp_reg_102[0] ^ _extra_dsp_reg_103[0] ^ _extra_dsp_reg_104[0] ^ _extra_dsp_reg_105[0] ^ _extra_dsp_reg_106[0] ^ _extra_dsp_reg_107[0] ^ _extra_dsp_reg_108[0] ^ _extra_dsp_reg_109[0] ^ _extra_dsp_reg_110[0] ^ _extra_dsp_reg_111[0] ^ _extra_dsp_reg_112[0] ^ _extra_dsp_reg_113[0] ^ _extra_dsp_reg_114[0] ^ _extra_dsp_reg_115[0] ^ _extra_dsp_reg_116[0] ^ _extra_dsp_reg_117[0] ^ _extra_dsp_reg_118[0] ^ _extra_dsp_reg_119[0] ^ _extra_dsp_reg_120[0] ^ _extra_dsp_reg_121[0] ^ _extra_dsp_reg_122[0] ^ _extra_dsp_reg_123[0] ^ _extra_dsp_reg_124[0] ^ _extra_dsp_reg_125[0] ^ _extra_dsp_reg_126[0] ^ _extra_dsp_reg_127[0] ^ _extra_dsp_reg_128[0] ^ _extra_dsp_reg_129[0] ^ _extra_dsp_reg_130[0] ^ _extra_dsp_reg_131[0] ^ _extra_dsp_reg_132[0] ^ _extra_dsp_reg_133[0] ^ _extra_dsp_reg_134[0] ^ _extra_dsp_reg_135[0] ^ _extra_dsp_reg_136[0] ^ _extra_dsp_reg_137[0] ^ _extra_dsp_reg_138[0] ^ _extra_dsp_reg_139[0] ^ _extra_dsp_reg_140[0] ^ _extra_dsp_reg_141[0] ^ _extra_dsp_reg_142[0] ^ _extra_dsp_reg_143[0] ^ _extra_dsp_reg_144[0] ^ _extra_dsp_reg_145[0] ^ _extra_dsp_reg_146[0] ^ _extra_dsp_reg_147[0] ^ _extra_dsp_reg_148[0] ^ _extra_dsp_reg_149[0] ^ _extra_dsp_reg_150[0] ^ _extra_dsp_reg_151[0] ^ _extra_dsp_reg_152[0] ^ _extra_dsp_reg_153[0] ^ _extra_dsp_reg_154[0] ^ _extra_dsp_reg_155[0] ^ _extra_dsp_reg_156[0] ^ _extra_dsp_reg_157[0] ^ _extra_dsp_reg_158[0] ^ _extra_dsp_reg_159[0] ^ _extra_dsp_reg_160[0] ^ _extra_dsp_reg_161[0] ^ _extra_dsp_reg_162[0] ^ _extra_dsp_reg_163[0] ^ _extra_dsp_reg_164[0] ^ _extra_dsp_reg_165[0] ^ _extra_dsp_reg_166[0] ^ _extra_dsp_reg_167[0] ^ _extra_dsp_reg_168[0] ^ _extra_dsp_reg_169[0] ^ _extra_dsp_reg_170[0] ^ _extra_dsp_reg_171[0] ^ _extra_dsp_reg_172[0] ^ _extra_dsp_reg_173[0] ^ _extra_dsp_reg_174[0] ^ _extra_dsp_reg_175[0] ^ _extra_dsp_reg_176[0] ^ _extra_dsp_reg_177[0] ^ _extra_dsp_reg_178[0] ^ _extra_dsp_reg_179[0] ^ _extra_dsp_reg_180[0] ^ _extra_dsp_reg_181[0] ^ _extra_dsp_reg_182[0] ^ _extra_dsp_reg_183[0] ^ _extra_dsp_reg_184[0] ^ _extra_dsp_reg_185[0] ^ _extra_dsp_reg_186[0] ^ _extra_dsp_reg_187[0] ^ _extra_dsp_reg_188[0] ^ _extra_dsp_reg_189[0] ^ _extra_dsp_reg_190[0] ^ _extra_dsp_reg_191[0] ^ _extra_dsp_reg_192[0] ^ _extra_dsp_reg_193[0] ^ _extra_dsp_reg_194[0] ^ _extra_dsp_reg_195[0] ^ _extra_dsp_reg_196[0] ^ _extra_dsp_reg_197[0] ^ _extra_dsp_reg_198[0] ^ _extra_dsp_reg_199[0] ^ _extra_dsp_reg_200[0] ^ _extra_dsp_reg_201[0] ^ _extra_dsp_reg_202[0] ^ _extra_dsp_reg_203[0] ^ _extra_dsp_reg_204[0] ^ _extra_dsp_reg_205[0] ^ _extra_dsp_reg_206[0] ^ _extra_dsp_reg_207[0] ^ _extra_dsp_reg_208[0] ^ _extra_dsp_reg_209[0] ^ _extra_dsp_reg_210[0] ^ _extra_dsp_reg_211[0] ^ _extra_dsp_reg_212[0] ^ _extra_dsp_reg_213[0] ^ _extra_dsp_reg_214[0] ^ _extra_dsp_reg_215[0] ^ _extra_dsp_reg_216[0] ^ _extra_dsp_reg_217[0] ^ _extra_dsp_reg_218[0] ^ _extra_dsp_reg_219[0] ^ _extra_dsp_reg_220[0] ^ _extra_dsp_reg_221[0] ^ _extra_dsp_reg_222[0] ^ _extra_dsp_reg_223[0] ^ _extra_dsp_reg_224[0] ^ _extra_dsp_reg_225[0] ^ _extra_dsp_reg_226[0] ^ _extra_dsp_reg_227[0] ^ _extra_dsp_reg_228[0] ^ _extra_dsp_reg_229[0] ^ _extra_dsp_reg_230[0] ^ _extra_dsp_reg_231[0] ^ _extra_dsp_reg_232[0] ^ _extra_dsp_reg_233[0] ^ _extra_dsp_reg_234[0] ^ _extra_dsp_reg_235[0] ^ _extra_dsp_reg_236[0] ^ _extra_dsp_reg_237[0] ^ _extra_dsp_reg_238[0] ^ _extra_dsp_reg_239[0] ^ _extra_dsp_reg_240[0] ^ _extra_dsp_reg_241[0] ^ _extra_dsp_reg_242[0] ^ _extra_dsp_reg_243[0] ^ _extra_dsp_reg_244[0] ^ _extra_dsp_reg_245[0] ^ _extra_dsp_reg_246[0] ^ _extra_dsp_reg_247[0] ^ _extra_dsp_reg_248[0] ^ _extra_dsp_reg_249[0] ^ _extra_dsp_reg_250[0] ^ _extra_dsp_reg_251[0] ^ _extra_dsp_reg_252[0] ^ _extra_dsp_reg_253[0] ^ _extra_dsp_reg_254[0] ^ _extra_dsp_reg_255[0] ^ _extra_dsp_reg_256[0] ^ _extra_dsp_reg_257[0] ^ _extra_dsp_reg_258[0] ^ _extra_dsp_reg_259[0] ^ _extra_dsp_reg_260[0] ^ _extra_dsp_reg_261[0] ^ _extra_dsp_reg_262[0] ^ _extra_dsp_reg_263[0] ^ _extra_dsp_reg_264[0] ^ _extra_dsp_reg_265[0] ^ _extra_dsp_reg_266[0] ^ _extra_dsp_reg_267[0] ^ _extra_dsp_reg_268[0] ^ _extra_dsp_reg_269[0] ^ _extra_dsp_reg_270[0] ^ _extra_dsp_reg_271[0] ^ _extra_dsp_reg_272[0] ^ _extra_dsp_reg_273[0] ^ _extra_dsp_reg_274[0] ^ _extra_dsp_reg_275[0] ^ _extra_dsp_reg_276[0] ^ _extra_dsp_reg_277[0] ^ _extra_dsp_reg_278[0] ^ _extra_dsp_reg_279[0] ^ _extra_dsp_reg_280[0] ^ _extra_dsp_reg_281[0] ^ _extra_dsp_reg_282[0] ^ _extra_dsp_reg_283[0] ^ _extra_dsp_reg_284[0] ^ _extra_dsp_reg_285[0] ^ _extra_dsp_reg_286[0] ^ _extra_dsp_reg_287[0] ^ _extra_dsp_reg_288[0] ^ _extra_dsp_reg_289[0] ^ _extra_dsp_reg_290[0] ^ _extra_dsp_reg_291[0] ^ _extra_dsp_reg_292[0] ^ _extra_dsp_reg_293[0] ^ _extra_dsp_reg_294[0] ^ _extra_dsp_reg_295[0] ^ _extra_dsp_reg_296[0] ^ _extra_dsp_reg_297[0] ^ _extra_dsp_reg_298[0] ^ _extra_dsp_reg_299[0] ^ _extra_dsp_reg_300[0] ^ _extra_dsp_reg_301[0] ^ _extra_dsp_reg_302[0] ^ _extra_dsp_reg_303[0] ^ _extra_dsp_reg_304[0] ^ _extra_dsp_reg_305[0] ^ _extra_dsp_reg_306[0] ^ _extra_dsp_reg_307[0] ^ _extra_dsp_reg_308[0] ^ _extra_dsp_reg_309[0] ^ _extra_dsp_reg_310[0] ^ _extra_dsp_reg_311[0] ^ _extra_dsp_reg_312[0] ^ _extra_dsp_reg_313[0] ^ _extra_dsp_reg_314[0] ^ _extra_dsp_reg_315[0] ^ _extra_dsp_reg_316[0] ^ _extra_dsp_reg_317[0] ^ _extra_dsp_reg_318[0] ^ _extra_dsp_reg_319[0] ^ _extra_dsp_reg_320[0] ^ _extra_dsp_reg_321[0] ^ _extra_dsp_reg_322[0] ^ _extra_dsp_reg_323[0] ^ _extra_dsp_reg_324[0] ^ _extra_dsp_reg_325[0] ^ _extra_dsp_reg_326[0] ^ _extra_dsp_reg_327[0] ^ _extra_dsp_reg_328[0] ^ _extra_dsp_reg_329[0] ^ _extra_dsp_reg_330[0] ^ _extra_dsp_reg_331[0] ^ _extra_dsp_reg_332[0]};
endmodule

module dsp_mac #(
    parameter DATA_WIDTH = 40,
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
    wire signed [DATA_WIDTH-1:0] product;
    reg out_valid;

    // DSP multiply (VTR infers DSP from '*')
    assign product = $signed(data_a) * $signed(data_b);

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
endmodule

module clb_softmax #(
    parameter DATA_WIDTH = 40,
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
    assign exp_val = ($signed(data_in) > 20) ? 40'd1048576 :
                     ($signed(data_in) < -20) ? 40'd1 :
                     (1 << (data_in[4:0] + 5'd10));

    // SRAM to buffer exp values
    reg [ADDR_W-1:0] w_addr, r_addr;
    reg w_en;
    wire [DATA_WIDTH-1:0] sram_out;
    sram #(.DATA_WIDTH(40), .DEPTH(N)) sm_buf (
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
    function [40-1:0] recip;
        input [40-1:0] val;
        integer k;
        reg [40-1:0] msb;
        begin
            msb = 0;
            for (k = 40-1; k >= 0; k = k - 1)
                if (val[k] && msb == 0) msb = k;
            recip = (msb > 0) ? (1 << (40 - 1 - msb)) : 0;
        end
    endfunction

    // DSP multiply for normalization
    wire [DATA_WIDTH-1:0] norm_product;
    assign norm_product = (sram_out * inv_sum) >> (40/2);

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




module activation_lut #(
    parameter DATA_WIDTH = 40
)(
    input wire clk,
    input wire [DATA_WIDTH-1:0] data_in,
    output reg [DATA_WIDTH-1:0] data_out
);
    // Saturation to int8 range [-128, 127]
    always @(posedge clk) begin
        if ($signed(data_in) > $signed(127))
            data_out <= 127;
        else if ($signed(data_in) < $signed(-128))
            data_out <= -128;
        else
            data_out <= data_in;
    end
endmodule

// ═══════════════════════════════════════
// Supporting modules (baseline, no DPE)
// ═══════════════════════════════════════
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
