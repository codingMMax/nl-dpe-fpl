// Auto-generated BERT-Tiny RTL — Azure-Lily 512×128
// Total DPEs: 18
// Projections/FFN: DPE, DIMM: CLB MAC (0 DSPs)

module bert_tiny_azurelily_s1024 (
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
    // b0_q: projection V=1 H=1 (1 DPE, DATA_WIDTH=16)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
    ) b0_q_inst (
        .clk(clk), .rst(rst),
        .valid(valid_ln_embed), .ready_n(ready_b0_q),
        .data_in(data_ln_embed),
        .data_out(data_b0_q), .ready(ready_ln_embed), .valid_n(valid_b0_q)
    );

    // b0_k: projection V=1 H=1 (1 DPE, DATA_WIDTH=16)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
    ) b0_k_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_q), .ready_n(ready_b0_k),
        .data_in(data_b0_q),
        .data_out(data_b0_k), .ready(ready_b0_j), .valid_n(valid_b0_k)
    );

    // b0_v: projection V=1 H=1 (1 DPE, DATA_WIDTH=16)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
    ) b0_v_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_k), .ready_n(ready_b0_v),
        .data_in(data_b0_k),
        .data_out(data_b0_v), .ready(ready_b0_u), .valid_n(valid_b0_v)
    );

    // DIMM intermediate buffer: Q for block 0 head 0 (depth=13109)
    wire [40-1:0] b0_h0_q_buf_out;
    reg [14-1:0] b0_h0_q_buf_addr;
    always @(posedge clk) if (rst) b0_h0_q_buf_addr <= 0; else if (valid) b0_h0_q_buf_addr <= b0_h0_q_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13109)) b0_h0_q_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b0_h0_q_buf_addr), .w_addr(b0_h0_q_buf_addr),
        .sram_data_in(data_b0_q), .sram_data_out(b0_h0_q_buf_out)
    );
    // DIMM intermediate buffer: K for block 0 head 0 (depth=13110)
    wire [40-1:0] b0_h0_k_buf_out;
    reg [14-1:0] b0_h0_k_buf_addr;
    always @(posedge clk) if (rst) b0_h0_k_buf_addr <= 0; else if (valid) b0_h0_k_buf_addr <= b0_h0_k_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13110)) b0_h0_k_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b0_h0_k_buf_addr), .w_addr(b0_h0_k_buf_addr),
        .sram_data_in(data_b0_k), .sram_data_out(b0_h0_k_buf_out)
    );
    // DIMM intermediate buffer: V for block 0 head 0 (depth=13111)
    wire [40-1:0] b0_h0_v_buf_out;
    reg [14-1:0] b0_h0_v_buf_addr;
    always @(posedge clk) if (rst) b0_h0_v_buf_addr <= 0; else if (valid) b0_h0_v_buf_addr <= b0_h0_v_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13111)) b0_h0_v_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b0_h0_v_buf_addr), .w_addr(b0_h0_v_buf_addr),
        .sram_data_in(data_b0_v), .sram_data_out(b0_h0_v_buf_out)
    );
    // Head 0: QK^T — W=8 parallel DSP MAC units
    wire [40-1:0] data_b0_h0_qk_w0;
    wire valid_b0_h0_qk_w0;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h0_qk_w0_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd1),
        .data_out(data_b0_h0_qk_w0), .ready(), .valid_n(valid_b0_h0_qk_w0)
    );
    wire [40-1:0] data_b0_h0_qk_w1;
    wire valid_b0_h0_qk_w1;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h0_qk_w1_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd2),
        .data_out(data_b0_h0_qk_w1), .ready(), .valid_n(valid_b0_h0_qk_w1)
    );
    wire [40-1:0] data_b0_h0_qk_w2;
    wire valid_b0_h0_qk_w2;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h0_qk_w2_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd3),
        .data_out(data_b0_h0_qk_w2), .ready(), .valid_n(valid_b0_h0_qk_w2)
    );
    wire [40-1:0] data_b0_h0_qk_w3;
    wire valid_b0_h0_qk_w3;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h0_qk_w3_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd4),
        .data_out(data_b0_h0_qk_w3), .ready(), .valid_n(valid_b0_h0_qk_w3)
    );
    wire [40-1:0] data_b0_h0_qk_w4;
    wire valid_b0_h0_qk_w4;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h0_qk_w4_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd5),
        .data_out(data_b0_h0_qk_w4), .ready(), .valid_n(valid_b0_h0_qk_w4)
    );
    wire [40-1:0] data_b0_h0_qk_w5;
    wire valid_b0_h0_qk_w5;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h0_qk_w5_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd6),
        .data_out(data_b0_h0_qk_w5), .ready(), .valid_n(valid_b0_h0_qk_w5)
    );
    wire [40-1:0] data_b0_h0_qk_w6;
    wire valid_b0_h0_qk_w6;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h0_qk_w6_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd7),
        .data_out(data_b0_h0_qk_w6), .ready(), .valid_n(valid_b0_h0_qk_w6)
    );
    wire [40-1:0] data_b0_h0_qk_w7;
    wire valid_b0_h0_qk_w7;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h0_qk_w7_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd8),
        .data_out(data_b0_h0_qk_w7), .ready(), .valid_n(valid_b0_h0_qk_w7)
    );
    wire [40-1:0] data_b0_h0_qk = data_b0_h0_qk_w0;
    wire valid_b0_h0_qk = valid_b0_h0_qk_w0;
    wire ready_b0_h0_qk;
    assign ready_b0_v = 1'b1;
    // Head 0: Softmax (CLB)
    clb_softmax #(.DATA_WIDTH(40), .N(205)) b0_h0_softmax_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_qk), .ready_n(ready_b0_h0_softmax),
        .data_in(data_b0_h0_qk),
        .data_out(data_b0_h0_softmax), .ready(ready_b0_h0_qk), .valid_n(valid_b0_h0_softmax)
    );
    // Head 0: Score×V — W=8 parallel DSP MAC units
    wire [40-1:0] data_b0_h0_sv_w0;
    wire valid_b0_h0_sv_w0;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h0_sv_w0_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_softmax), .ready_n(1'b0),
        .data_a(data_b0_h0_softmax), .data_b(data_b0_v ^ 40'd9),
        .data_out(data_b0_h0_sv_w0), .ready(), .valid_n(valid_b0_h0_sv_w0)
    );
    wire [40-1:0] data_b0_h0_sv_w1;
    wire valid_b0_h0_sv_w1;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h0_sv_w1_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_softmax), .ready_n(1'b0),
        .data_a(data_b0_h0_softmax), .data_b(data_b0_v ^ 40'd10),
        .data_out(data_b0_h0_sv_w1), .ready(), .valid_n(valid_b0_h0_sv_w1)
    );
    wire [40-1:0] data_b0_h0_sv_w2;
    wire valid_b0_h0_sv_w2;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h0_sv_w2_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_softmax), .ready_n(1'b0),
        .data_a(data_b0_h0_softmax), .data_b(data_b0_v ^ 40'd11),
        .data_out(data_b0_h0_sv_w2), .ready(), .valid_n(valid_b0_h0_sv_w2)
    );
    wire [40-1:0] data_b0_h0_sv_w3;
    wire valid_b0_h0_sv_w3;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h0_sv_w3_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_softmax), .ready_n(1'b0),
        .data_a(data_b0_h0_softmax), .data_b(data_b0_v ^ 40'd12),
        .data_out(data_b0_h0_sv_w3), .ready(), .valid_n(valid_b0_h0_sv_w3)
    );
    wire [40-1:0] data_b0_h0_sv_w4;
    wire valid_b0_h0_sv_w4;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h0_sv_w4_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_softmax), .ready_n(1'b0),
        .data_a(data_b0_h0_softmax), .data_b(data_b0_v ^ 40'd13),
        .data_out(data_b0_h0_sv_w4), .ready(), .valid_n(valid_b0_h0_sv_w4)
    );
    wire [40-1:0] data_b0_h0_sv_w5;
    wire valid_b0_h0_sv_w5;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h0_sv_w5_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_softmax), .ready_n(1'b0),
        .data_a(data_b0_h0_softmax), .data_b(data_b0_v ^ 40'd14),
        .data_out(data_b0_h0_sv_w5), .ready(), .valid_n(valid_b0_h0_sv_w5)
    );
    wire [40-1:0] data_b0_h0_sv_w6;
    wire valid_b0_h0_sv_w6;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h0_sv_w6_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_softmax), .ready_n(1'b0),
        .data_a(data_b0_h0_softmax), .data_b(data_b0_v ^ 40'd15),
        .data_out(data_b0_h0_sv_w6), .ready(), .valid_n(valid_b0_h0_sv_w6)
    );
    wire [40-1:0] data_b0_h0_sv_w7;
    wire valid_b0_h0_sv_w7;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h0_sv_w7_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h0_softmax), .ready_n(1'b0),
        .data_a(data_b0_h0_softmax), .data_b(data_b0_v ^ 40'd16),
        .data_out(data_b0_h0_sv_w7), .ready(), .valid_n(valid_b0_h0_sv_w7)
    );
    wire [40-1:0] data_b0_h0_sv = data_b0_h0_sv_w0;
    wire valid_b0_h0_sv = valid_b0_h0_sv_w0;
    wire ready_b0_h0_sv;
    assign ready_b0_h0_softmax = 1'b1;

    // DIMM intermediate buffer: Q for block 0 head 1 (depth=13112)
    wire [40-1:0] b0_h1_q_buf_out;
    reg [14-1:0] b0_h1_q_buf_addr;
    always @(posedge clk) if (rst) b0_h1_q_buf_addr <= 0; else if (valid) b0_h1_q_buf_addr <= b0_h1_q_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13112)) b0_h1_q_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b0_h1_q_buf_addr), .w_addr(b0_h1_q_buf_addr),
        .sram_data_in(data_b0_q), .sram_data_out(b0_h1_q_buf_out)
    );
    // DIMM intermediate buffer: K for block 0 head 1 (depth=13113)
    wire [40-1:0] b0_h1_k_buf_out;
    reg [14-1:0] b0_h1_k_buf_addr;
    always @(posedge clk) if (rst) b0_h1_k_buf_addr <= 0; else if (valid) b0_h1_k_buf_addr <= b0_h1_k_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13113)) b0_h1_k_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b0_h1_k_buf_addr), .w_addr(b0_h1_k_buf_addr),
        .sram_data_in(data_b0_k), .sram_data_out(b0_h1_k_buf_out)
    );
    // DIMM intermediate buffer: V for block 0 head 1 (depth=13114)
    wire [40-1:0] b0_h1_v_buf_out;
    reg [14-1:0] b0_h1_v_buf_addr;
    always @(posedge clk) if (rst) b0_h1_v_buf_addr <= 0; else if (valid) b0_h1_v_buf_addr <= b0_h1_v_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13114)) b0_h1_v_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b0_h1_v_buf_addr), .w_addr(b0_h1_v_buf_addr),
        .sram_data_in(data_b0_v), .sram_data_out(b0_h1_v_buf_out)
    );
    // Head 1: QK^T — W=8 parallel DSP MAC units
    wire [40-1:0] data_b0_h1_qk_w0;
    wire valid_b0_h1_qk_w0;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h1_qk_w0_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd17),
        .data_out(data_b0_h1_qk_w0), .ready(), .valid_n(valid_b0_h1_qk_w0)
    );
    wire [40-1:0] data_b0_h1_qk_w1;
    wire valid_b0_h1_qk_w1;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h1_qk_w1_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd18),
        .data_out(data_b0_h1_qk_w1), .ready(), .valid_n(valid_b0_h1_qk_w1)
    );
    wire [40-1:0] data_b0_h1_qk_w2;
    wire valid_b0_h1_qk_w2;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h1_qk_w2_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd19),
        .data_out(data_b0_h1_qk_w2), .ready(), .valid_n(valid_b0_h1_qk_w2)
    );
    wire [40-1:0] data_b0_h1_qk_w3;
    wire valid_b0_h1_qk_w3;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h1_qk_w3_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd20),
        .data_out(data_b0_h1_qk_w3), .ready(), .valid_n(valid_b0_h1_qk_w3)
    );
    wire [40-1:0] data_b0_h1_qk_w4;
    wire valid_b0_h1_qk_w4;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h1_qk_w4_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd21),
        .data_out(data_b0_h1_qk_w4), .ready(), .valid_n(valid_b0_h1_qk_w4)
    );
    wire [40-1:0] data_b0_h1_qk_w5;
    wire valid_b0_h1_qk_w5;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h1_qk_w5_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd22),
        .data_out(data_b0_h1_qk_w5), .ready(), .valid_n(valid_b0_h1_qk_w5)
    );
    wire [40-1:0] data_b0_h1_qk_w6;
    wire valid_b0_h1_qk_w6;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h1_qk_w6_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd23),
        .data_out(data_b0_h1_qk_w6), .ready(), .valid_n(valid_b0_h1_qk_w6)
    );
    wire [40-1:0] data_b0_h1_qk_w7;
    wire valid_b0_h1_qk_w7;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b0_h1_qk_w7_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_v), .ready_n(1'b0),
        .data_a(data_b0_q), .data_b(data_b0_k ^ 40'd24),
        .data_out(data_b0_h1_qk_w7), .ready(), .valid_n(valid_b0_h1_qk_w7)
    );
    wire [40-1:0] data_b0_h1_qk = data_b0_h1_qk_w0;
    wire valid_b0_h1_qk = valid_b0_h1_qk_w0;
    wire ready_b0_h1_qk;
    assign ready_b0_v = 1'b1;
    // Head 1: Softmax (CLB)
    clb_softmax #(.DATA_WIDTH(40), .N(205)) b0_h1_softmax_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_qk), .ready_n(ready_b0_h1_softmax),
        .data_in(data_b0_h1_qk),
        .data_out(data_b0_h1_softmax), .ready(ready_b0_h1_qk), .valid_n(valid_b0_h1_softmax)
    );
    // Head 1: Score×V — W=8 parallel DSP MAC units
    wire [40-1:0] data_b0_h1_sv_w0;
    wire valid_b0_h1_sv_w0;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h1_sv_w0_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_softmax), .ready_n(1'b0),
        .data_a(data_b0_h1_softmax), .data_b(data_b0_v ^ 40'd25),
        .data_out(data_b0_h1_sv_w0), .ready(), .valid_n(valid_b0_h1_sv_w0)
    );
    wire [40-1:0] data_b0_h1_sv_w1;
    wire valid_b0_h1_sv_w1;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h1_sv_w1_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_softmax), .ready_n(1'b0),
        .data_a(data_b0_h1_softmax), .data_b(data_b0_v ^ 40'd26),
        .data_out(data_b0_h1_sv_w1), .ready(), .valid_n(valid_b0_h1_sv_w1)
    );
    wire [40-1:0] data_b0_h1_sv_w2;
    wire valid_b0_h1_sv_w2;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h1_sv_w2_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_softmax), .ready_n(1'b0),
        .data_a(data_b0_h1_softmax), .data_b(data_b0_v ^ 40'd27),
        .data_out(data_b0_h1_sv_w2), .ready(), .valid_n(valid_b0_h1_sv_w2)
    );
    wire [40-1:0] data_b0_h1_sv_w3;
    wire valid_b0_h1_sv_w3;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h1_sv_w3_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_softmax), .ready_n(1'b0),
        .data_a(data_b0_h1_softmax), .data_b(data_b0_v ^ 40'd28),
        .data_out(data_b0_h1_sv_w3), .ready(), .valid_n(valid_b0_h1_sv_w3)
    );
    wire [40-1:0] data_b0_h1_sv_w4;
    wire valid_b0_h1_sv_w4;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h1_sv_w4_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_softmax), .ready_n(1'b0),
        .data_a(data_b0_h1_softmax), .data_b(data_b0_v ^ 40'd29),
        .data_out(data_b0_h1_sv_w4), .ready(), .valid_n(valid_b0_h1_sv_w4)
    );
    wire [40-1:0] data_b0_h1_sv_w5;
    wire valid_b0_h1_sv_w5;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h1_sv_w5_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_softmax), .ready_n(1'b0),
        .data_a(data_b0_h1_softmax), .data_b(data_b0_v ^ 40'd30),
        .data_out(data_b0_h1_sv_w5), .ready(), .valid_n(valid_b0_h1_sv_w5)
    );
    wire [40-1:0] data_b0_h1_sv_w6;
    wire valid_b0_h1_sv_w6;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h1_sv_w6_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_softmax), .ready_n(1'b0),
        .data_a(data_b0_h1_softmax), .data_b(data_b0_v ^ 40'd31),
        .data_out(data_b0_h1_sv_w6), .ready(), .valid_n(valid_b0_h1_sv_w6)
    );
    wire [40-1:0] data_b0_h1_sv_w7;
    wire valid_b0_h1_sv_w7;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b0_h1_sv_w7_inst (
        .clk(clk), .rst(rst), .valid(valid_b0_h1_softmax), .ready_n(1'b0),
        .data_a(data_b0_h1_softmax), .data_b(data_b0_v ^ 40'd32),
        .data_out(data_b0_h1_sv_w7), .ready(), .valid_n(valid_b0_h1_sv_w7)
    );
    wire [40-1:0] data_b0_h1_sv = data_b0_h1_sv_w0;
    wire valid_b0_h1_sv = valid_b0_h1_sv_w0;
    wire ready_b0_h1_sv;
    assign ready_b0_h1_softmax = 1'b1;

    // O projection V=1 H=1 (1 DPE, DATA_WIDTH=16)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
    ) b0_o_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_h1_sv), .ready_n(ready_b0_o),
        .data_in(data_b0_h1_sv),
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

    // FFN1: V=1 H=4 (4 DPEs)
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
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
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
    // b1_q: projection V=1 H=1 (1 DPE, DATA_WIDTH=16)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
    ) b1_q_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b0_ln_ffn), .ready_n(ready_b1_q),
        .data_in(data_b0_ln_ffn),
        .data_out(data_b1_q), .ready(ready_ln_ffn), .valid_n(valid_b1_q)
    );

    // b1_k: projection V=1 H=1 (1 DPE, DATA_WIDTH=16)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
    ) b1_k_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_q), .ready_n(ready_b1_k),
        .data_in(data_b1_q),
        .data_out(data_b1_k), .ready(ready_b1_j), .valid_n(valid_b1_k)
    );

    // b1_v: projection V=1 H=1 (1 DPE, DATA_WIDTH=16)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
    ) b1_v_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_k), .ready_n(ready_b1_v),
        .data_in(data_b1_k),
        .data_out(data_b1_v), .ready(ready_b1_u), .valid_n(valid_b1_v)
    );

    // DIMM intermediate buffer: Q for block 1 head 0 (depth=13115)
    wire [40-1:0] b1_h0_q_buf_out;
    reg [14-1:0] b1_h0_q_buf_addr;
    always @(posedge clk) if (rst) b1_h0_q_buf_addr <= 0; else if (valid) b1_h0_q_buf_addr <= b1_h0_q_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13115)) b1_h0_q_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b1_h0_q_buf_addr), .w_addr(b1_h0_q_buf_addr),
        .sram_data_in(data_b1_q), .sram_data_out(b1_h0_q_buf_out)
    );
    // DIMM intermediate buffer: K for block 1 head 0 (depth=13116)
    wire [40-1:0] b1_h0_k_buf_out;
    reg [14-1:0] b1_h0_k_buf_addr;
    always @(posedge clk) if (rst) b1_h0_k_buf_addr <= 0; else if (valid) b1_h0_k_buf_addr <= b1_h0_k_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13116)) b1_h0_k_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b1_h0_k_buf_addr), .w_addr(b1_h0_k_buf_addr),
        .sram_data_in(data_b1_k), .sram_data_out(b1_h0_k_buf_out)
    );
    // DIMM intermediate buffer: V for block 1 head 0 (depth=13117)
    wire [40-1:0] b1_h0_v_buf_out;
    reg [14-1:0] b1_h0_v_buf_addr;
    always @(posedge clk) if (rst) b1_h0_v_buf_addr <= 0; else if (valid) b1_h0_v_buf_addr <= b1_h0_v_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13117)) b1_h0_v_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b1_h0_v_buf_addr), .w_addr(b1_h0_v_buf_addr),
        .sram_data_in(data_b1_v), .sram_data_out(b1_h0_v_buf_out)
    );
    // Head 0: QK^T — W=8 parallel DSP MAC units
    wire [40-1:0] data_b1_h0_qk_w0;
    wire valid_b1_h0_qk_w0;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h0_qk_w0_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd33),
        .data_out(data_b1_h0_qk_w0), .ready(), .valid_n(valid_b1_h0_qk_w0)
    );
    wire [40-1:0] data_b1_h0_qk_w1;
    wire valid_b1_h0_qk_w1;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h0_qk_w1_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd34),
        .data_out(data_b1_h0_qk_w1), .ready(), .valid_n(valid_b1_h0_qk_w1)
    );
    wire [40-1:0] data_b1_h0_qk_w2;
    wire valid_b1_h0_qk_w2;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h0_qk_w2_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd35),
        .data_out(data_b1_h0_qk_w2), .ready(), .valid_n(valid_b1_h0_qk_w2)
    );
    wire [40-1:0] data_b1_h0_qk_w3;
    wire valid_b1_h0_qk_w3;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h0_qk_w3_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd36),
        .data_out(data_b1_h0_qk_w3), .ready(), .valid_n(valid_b1_h0_qk_w3)
    );
    wire [40-1:0] data_b1_h0_qk_w4;
    wire valid_b1_h0_qk_w4;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h0_qk_w4_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd37),
        .data_out(data_b1_h0_qk_w4), .ready(), .valid_n(valid_b1_h0_qk_w4)
    );
    wire [40-1:0] data_b1_h0_qk_w5;
    wire valid_b1_h0_qk_w5;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h0_qk_w5_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd38),
        .data_out(data_b1_h0_qk_w5), .ready(), .valid_n(valid_b1_h0_qk_w5)
    );
    wire [40-1:0] data_b1_h0_qk_w6;
    wire valid_b1_h0_qk_w6;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h0_qk_w6_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd39),
        .data_out(data_b1_h0_qk_w6), .ready(), .valid_n(valid_b1_h0_qk_w6)
    );
    wire [40-1:0] data_b1_h0_qk_w7;
    wire valid_b1_h0_qk_w7;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h0_qk_w7_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd40),
        .data_out(data_b1_h0_qk_w7), .ready(), .valid_n(valid_b1_h0_qk_w7)
    );
    wire [40-1:0] data_b1_h0_qk = data_b1_h0_qk_w0;
    wire valid_b1_h0_qk = valid_b1_h0_qk_w0;
    wire ready_b1_h0_qk;
    assign ready_b1_v = 1'b1;
    // Head 0: Softmax (CLB)
    clb_softmax #(.DATA_WIDTH(40), .N(205)) b1_h0_softmax_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_qk), .ready_n(ready_b1_h0_softmax),
        .data_in(data_b1_h0_qk),
        .data_out(data_b1_h0_softmax), .ready(ready_b1_h0_qk), .valid_n(valid_b1_h0_softmax)
    );
    // Head 0: Score×V — W=8 parallel DSP MAC units
    wire [40-1:0] data_b1_h0_sv_w0;
    wire valid_b1_h0_sv_w0;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h0_sv_w0_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_softmax), .ready_n(1'b0),
        .data_a(data_b1_h0_softmax), .data_b(data_b1_v ^ 40'd41),
        .data_out(data_b1_h0_sv_w0), .ready(), .valid_n(valid_b1_h0_sv_w0)
    );
    wire [40-1:0] data_b1_h0_sv_w1;
    wire valid_b1_h0_sv_w1;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h0_sv_w1_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_softmax), .ready_n(1'b0),
        .data_a(data_b1_h0_softmax), .data_b(data_b1_v ^ 40'd42),
        .data_out(data_b1_h0_sv_w1), .ready(), .valid_n(valid_b1_h0_sv_w1)
    );
    wire [40-1:0] data_b1_h0_sv_w2;
    wire valid_b1_h0_sv_w2;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h0_sv_w2_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_softmax), .ready_n(1'b0),
        .data_a(data_b1_h0_softmax), .data_b(data_b1_v ^ 40'd43),
        .data_out(data_b1_h0_sv_w2), .ready(), .valid_n(valid_b1_h0_sv_w2)
    );
    wire [40-1:0] data_b1_h0_sv_w3;
    wire valid_b1_h0_sv_w3;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h0_sv_w3_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_softmax), .ready_n(1'b0),
        .data_a(data_b1_h0_softmax), .data_b(data_b1_v ^ 40'd44),
        .data_out(data_b1_h0_sv_w3), .ready(), .valid_n(valid_b1_h0_sv_w3)
    );
    wire [40-1:0] data_b1_h0_sv_w4;
    wire valid_b1_h0_sv_w4;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h0_sv_w4_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_softmax), .ready_n(1'b0),
        .data_a(data_b1_h0_softmax), .data_b(data_b1_v ^ 40'd45),
        .data_out(data_b1_h0_sv_w4), .ready(), .valid_n(valid_b1_h0_sv_w4)
    );
    wire [40-1:0] data_b1_h0_sv_w5;
    wire valid_b1_h0_sv_w5;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h0_sv_w5_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_softmax), .ready_n(1'b0),
        .data_a(data_b1_h0_softmax), .data_b(data_b1_v ^ 40'd46),
        .data_out(data_b1_h0_sv_w5), .ready(), .valid_n(valid_b1_h0_sv_w5)
    );
    wire [40-1:0] data_b1_h0_sv_w6;
    wire valid_b1_h0_sv_w6;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h0_sv_w6_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_softmax), .ready_n(1'b0),
        .data_a(data_b1_h0_softmax), .data_b(data_b1_v ^ 40'd47),
        .data_out(data_b1_h0_sv_w6), .ready(), .valid_n(valid_b1_h0_sv_w6)
    );
    wire [40-1:0] data_b1_h0_sv_w7;
    wire valid_b1_h0_sv_w7;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h0_sv_w7_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h0_softmax), .ready_n(1'b0),
        .data_a(data_b1_h0_softmax), .data_b(data_b1_v ^ 40'd48),
        .data_out(data_b1_h0_sv_w7), .ready(), .valid_n(valid_b1_h0_sv_w7)
    );
    wire [40-1:0] data_b1_h0_sv = data_b1_h0_sv_w0;
    wire valid_b1_h0_sv = valid_b1_h0_sv_w0;
    wire ready_b1_h0_sv;
    assign ready_b1_h0_softmax = 1'b1;

    // DIMM intermediate buffer: Q for block 1 head 1 (depth=13118)
    wire [40-1:0] b1_h1_q_buf_out;
    reg [14-1:0] b1_h1_q_buf_addr;
    always @(posedge clk) if (rst) b1_h1_q_buf_addr <= 0; else if (valid) b1_h1_q_buf_addr <= b1_h1_q_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13118)) b1_h1_q_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b1_h1_q_buf_addr), .w_addr(b1_h1_q_buf_addr),
        .sram_data_in(data_b1_q), .sram_data_out(b1_h1_q_buf_out)
    );
    // DIMM intermediate buffer: K for block 1 head 1 (depth=13119)
    wire [40-1:0] b1_h1_k_buf_out;
    reg [14-1:0] b1_h1_k_buf_addr;
    always @(posedge clk) if (rst) b1_h1_k_buf_addr <= 0; else if (valid) b1_h1_k_buf_addr <= b1_h1_k_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13119)) b1_h1_k_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b1_h1_k_buf_addr), .w_addr(b1_h1_k_buf_addr),
        .sram_data_in(data_b1_k), .sram_data_out(b1_h1_k_buf_out)
    );
    // DIMM intermediate buffer: V for block 1 head 1 (depth=13120)
    wire [40-1:0] b1_h1_v_buf_out;
    reg [14-1:0] b1_h1_v_buf_addr;
    always @(posedge clk) if (rst) b1_h1_v_buf_addr <= 0; else if (valid) b1_h1_v_buf_addr <= b1_h1_v_buf_addr + 1;
    sram #(.DATA_WIDTH(40), .DEPTH(13120)) b1_h1_v_buf_inst (
        .clk(clk), .rst(rst), .w_en(valid),
        .r_addr(b1_h1_v_buf_addr), .w_addr(b1_h1_v_buf_addr),
        .sram_data_in(data_b1_v), .sram_data_out(b1_h1_v_buf_out)
    );
    // Head 1: QK^T — W=8 parallel DSP MAC units
    wire [40-1:0] data_b1_h1_qk_w0;
    wire valid_b1_h1_qk_w0;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h1_qk_w0_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd49),
        .data_out(data_b1_h1_qk_w0), .ready(), .valid_n(valid_b1_h1_qk_w0)
    );
    wire [40-1:0] data_b1_h1_qk_w1;
    wire valid_b1_h1_qk_w1;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h1_qk_w1_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd50),
        .data_out(data_b1_h1_qk_w1), .ready(), .valid_n(valid_b1_h1_qk_w1)
    );
    wire [40-1:0] data_b1_h1_qk_w2;
    wire valid_b1_h1_qk_w2;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h1_qk_w2_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd51),
        .data_out(data_b1_h1_qk_w2), .ready(), .valid_n(valid_b1_h1_qk_w2)
    );
    wire [40-1:0] data_b1_h1_qk_w3;
    wire valid_b1_h1_qk_w3;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h1_qk_w3_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd52),
        .data_out(data_b1_h1_qk_w3), .ready(), .valid_n(valid_b1_h1_qk_w3)
    );
    wire [40-1:0] data_b1_h1_qk_w4;
    wire valid_b1_h1_qk_w4;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h1_qk_w4_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd53),
        .data_out(data_b1_h1_qk_w4), .ready(), .valid_n(valid_b1_h1_qk_w4)
    );
    wire [40-1:0] data_b1_h1_qk_w5;
    wire valid_b1_h1_qk_w5;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h1_qk_w5_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd54),
        .data_out(data_b1_h1_qk_w5), .ready(), .valid_n(valid_b1_h1_qk_w5)
    );
    wire [40-1:0] data_b1_h1_qk_w6;
    wire valid_b1_h1_qk_w6;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h1_qk_w6_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd55),
        .data_out(data_b1_h1_qk_w6), .ready(), .valid_n(valid_b1_h1_qk_w6)
    );
    wire [40-1:0] data_b1_h1_qk_w7;
    wire valid_b1_h1_qk_w7;
    dsp_mac #(.DATA_WIDTH(40), .K(13)) b1_h1_qk_w7_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_v), .ready_n(1'b0),
        .data_a(data_b1_q), .data_b(data_b1_k ^ 40'd56),
        .data_out(data_b1_h1_qk_w7), .ready(), .valid_n(valid_b1_h1_qk_w7)
    );
    wire [40-1:0] data_b1_h1_qk = data_b1_h1_qk_w0;
    wire valid_b1_h1_qk = valid_b1_h1_qk_w0;
    wire ready_b1_h1_qk;
    assign ready_b1_v = 1'b1;
    // Head 1: Softmax (CLB)
    clb_softmax #(.DATA_WIDTH(40), .N(205)) b1_h1_softmax_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_qk), .ready_n(ready_b1_h1_softmax),
        .data_in(data_b1_h1_qk),
        .data_out(data_b1_h1_softmax), .ready(ready_b1_h1_qk), .valid_n(valid_b1_h1_softmax)
    );
    // Head 1: Score×V — W=8 parallel DSP MAC units
    wire [40-1:0] data_b1_h1_sv_w0;
    wire valid_b1_h1_sv_w0;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h1_sv_w0_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_softmax), .ready_n(1'b0),
        .data_a(data_b1_h1_softmax), .data_b(data_b1_v ^ 40'd57),
        .data_out(data_b1_h1_sv_w0), .ready(), .valid_n(valid_b1_h1_sv_w0)
    );
    wire [40-1:0] data_b1_h1_sv_w1;
    wire valid_b1_h1_sv_w1;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h1_sv_w1_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_softmax), .ready_n(1'b0),
        .data_a(data_b1_h1_softmax), .data_b(data_b1_v ^ 40'd58),
        .data_out(data_b1_h1_sv_w1), .ready(), .valid_n(valid_b1_h1_sv_w1)
    );
    wire [40-1:0] data_b1_h1_sv_w2;
    wire valid_b1_h1_sv_w2;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h1_sv_w2_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_softmax), .ready_n(1'b0),
        .data_a(data_b1_h1_softmax), .data_b(data_b1_v ^ 40'd59),
        .data_out(data_b1_h1_sv_w2), .ready(), .valid_n(valid_b1_h1_sv_w2)
    );
    wire [40-1:0] data_b1_h1_sv_w3;
    wire valid_b1_h1_sv_w3;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h1_sv_w3_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_softmax), .ready_n(1'b0),
        .data_a(data_b1_h1_softmax), .data_b(data_b1_v ^ 40'd60),
        .data_out(data_b1_h1_sv_w3), .ready(), .valid_n(valid_b1_h1_sv_w3)
    );
    wire [40-1:0] data_b1_h1_sv_w4;
    wire valid_b1_h1_sv_w4;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h1_sv_w4_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_softmax), .ready_n(1'b0),
        .data_a(data_b1_h1_softmax), .data_b(data_b1_v ^ 40'd61),
        .data_out(data_b1_h1_sv_w4), .ready(), .valid_n(valid_b1_h1_sv_w4)
    );
    wire [40-1:0] data_b1_h1_sv_w5;
    wire valid_b1_h1_sv_w5;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h1_sv_w5_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_softmax), .ready_n(1'b0),
        .data_a(data_b1_h1_softmax), .data_b(data_b1_v ^ 40'd62),
        .data_out(data_b1_h1_sv_w5), .ready(), .valid_n(valid_b1_h1_sv_w5)
    );
    wire [40-1:0] data_b1_h1_sv_w6;
    wire valid_b1_h1_sv_w6;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h1_sv_w6_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_softmax), .ready_n(1'b0),
        .data_a(data_b1_h1_softmax), .data_b(data_b1_v ^ 40'd63),
        .data_out(data_b1_h1_sv_w6), .ready(), .valid_n(valid_b1_h1_sv_w6)
    );
    wire [40-1:0] data_b1_h1_sv_w7;
    wire valid_b1_h1_sv_w7;
    dsp_mac #(.DATA_WIDTH(40), .K(205)) b1_h1_sv_w7_inst (
        .clk(clk), .rst(rst), .valid(valid_b1_h1_softmax), .ready_n(1'b0),
        .data_a(data_b1_h1_softmax), .data_b(data_b1_v ^ 40'd64),
        .data_out(data_b1_h1_sv_w7), .ready(), .valid_n(valid_b1_h1_sv_w7)
    );
    wire [40-1:0] data_b1_h1_sv = data_b1_h1_sv_w0;
    wire valid_b1_h1_sv = valid_b1_h1_sv_w0;
    wire ready_b1_h1_sv;
    assign ready_b1_h1_softmax = 1'b1;

    // O projection V=1 H=1 (1 DPE, DATA_WIDTH=16)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(128), .KERNEL_HEIGHT(1),
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
    ) b1_o_inst (
        .clk(clk), .rst(rst),
        .valid(valid_b1_h1_sv), .ready_n(ready_b1_o),
        .data_in(data_b1_h1_sv),
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

    // FFN1: V=1 H=4 (4 DPEs)
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
        .W(1024), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(16)
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

    // ═══ Azure-Lily DIMM: W=8 parallel DSP MACs per stage ═══
    // 4 stages × 8 × 2 heads × 2 blocks = 128 DIMM DSPs
    // + ~10 structural DSPs (LayerNorm multiply + softmax)
    assign data_out = data_b1_ln_ffn ^ {39'b0, data_b0_h0_qk_w0[0] ^ data_b0_h0_qk_w1[0] ^ data_b0_h0_qk_w2[0] ^ data_b0_h0_qk_w3[0] ^ data_b0_h0_qk_w4[0] ^ data_b0_h0_qk_w5[0] ^ data_b0_h0_qk_w6[0] ^ data_b0_h0_qk_w7[0] ^ data_b0_h0_sv_w0[0] ^ data_b0_h0_sv_w1[0] ^ data_b0_h0_sv_w2[0] ^ data_b0_h0_sv_w3[0] ^ data_b0_h0_sv_w4[0] ^ data_b0_h0_sv_w5[0] ^ data_b0_h0_sv_w6[0] ^ data_b0_h0_sv_w7[0] ^ b0_h0_q_buf_out[0] ^ b0_h0_k_buf_out[0] ^ b0_h0_v_buf_out[0] ^ data_b0_h1_qk_w0[0] ^ data_b0_h1_qk_w1[0] ^ data_b0_h1_qk_w2[0] ^ data_b0_h1_qk_w3[0] ^ data_b0_h1_qk_w4[0] ^ data_b0_h1_qk_w5[0] ^ data_b0_h1_qk_w6[0] ^ data_b0_h1_qk_w7[0] ^ data_b0_h1_sv_w0[0] ^ data_b0_h1_sv_w1[0] ^ data_b0_h1_sv_w2[0] ^ data_b0_h1_sv_w3[0] ^ data_b0_h1_sv_w4[0] ^ data_b0_h1_sv_w5[0] ^ data_b0_h1_sv_w6[0] ^ data_b0_h1_sv_w7[0] ^ b0_h1_q_buf_out[0] ^ b0_h1_k_buf_out[0] ^ b0_h1_v_buf_out[0] ^ data_b1_h0_qk_w0[0] ^ data_b1_h0_qk_w1[0] ^ data_b1_h0_qk_w2[0] ^ data_b1_h0_qk_w3[0] ^ data_b1_h0_qk_w4[0] ^ data_b1_h0_qk_w5[0] ^ data_b1_h0_qk_w6[0] ^ data_b1_h0_qk_w7[0] ^ data_b1_h0_sv_w0[0] ^ data_b1_h0_sv_w1[0] ^ data_b1_h0_sv_w2[0] ^ data_b1_h0_sv_w3[0] ^ data_b1_h0_sv_w4[0] ^ data_b1_h0_sv_w5[0] ^ data_b1_h0_sv_w6[0] ^ data_b1_h0_sv_w7[0] ^ b1_h0_q_buf_out[0] ^ b1_h0_k_buf_out[0] ^ b1_h0_v_buf_out[0] ^ data_b1_h1_qk_w0[0] ^ data_b1_h1_qk_w1[0] ^ data_b1_h1_qk_w2[0] ^ data_b1_h1_qk_w3[0] ^ data_b1_h1_qk_w4[0] ^ data_b1_h1_qk_w5[0] ^ data_b1_h1_qk_w6[0] ^ data_b1_h1_qk_w7[0] ^ data_b1_h1_sv_w0[0] ^ data_b1_h1_sv_w1[0] ^ data_b1_h1_sv_w2[0] ^ data_b1_h1_sv_w3[0] ^ data_b1_h1_sv_w4[0] ^ data_b1_h1_sv_w5[0] ^ data_b1_h1_sv_w6[0] ^ data_b1_h1_sv_w7[0] ^ b1_h1_q_buf_out[0] ^ b1_h1_k_buf_out[0] ^ b1_h1_v_buf_out[0]};
endmodule

// FFN1 block 0: V=1 H=4
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
    localparam H = 4;
    localparam DEPTH = 512;
    localparam ADDR_WIDTH = 9;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [1-1:0] reg_full_sig;
    wire [4-1:0] reg_empty;
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
    wire [2-1:0] dpe_sel_h;

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
        .N_DPE_H(4),
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
    wire [DATA_WIDTH-1:0] dpe_out_c2_r0;
    wire dpe_done_c2_r0;
    wire reg_full_c2_r0;
    wire shift_add_done_c2_r0;
    wire shift_add_bypass_ctrl_c2_r0;
    wire MSB_SA_Ready_c2_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r0;
    wire dpe_done_c3_r0;
    wire reg_full_c3_r0;
    wire shift_add_done_c3_r0;
    wire shift_add_bypass_ctrl_c3_r0;
    wire MSB_SA_Ready_c3_r0;

    // DPE instantiations: 1 vertical x 4 horizontal = 4 DPEs
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

    dpe dpe_c2_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r0),
        .data_out(dpe_out_c2_r0),
        .dpe_done(dpe_done_c2_r0),
        .reg_full(reg_full_c2_r0),
        .shift_add_done(shift_add_done_c2_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r0)
    );

    dpe dpe_c3_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r0),
        .data_out(dpe_out_c3_r0),
        .dpe_done(dpe_done_c3_r0),
        .reg_full(reg_full_c3_r0),
        .shift_add_done(shift_add_done_c3_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r0)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c1_r0 | dpe_done_c2_r0 | dpe_done_c3_r0;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c1_r0 & shift_add_done_c2_r0 & shift_add_done_c3_r0;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c2_r0 & shift_add_bypass_ctrl_c3_r0;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c2_r0 & MSB_SA_Ready_c3_r0;
    assign reg_full_sig = {reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0, 1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Horizontal output mux (4 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            2'd0: data_out_mux = dpe_out_c0_r0;
            2'd1: data_out_mux = dpe_out_c1_r0;
            2'd2: data_out_mux = dpe_out_c2_r0;
            2'd3: data_out_mux = dpe_out_c3_r0;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

// FFN1 block 1: V=1 H=4
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
    localparam H = 4;
    localparam DEPTH = 512;
    localparam ADDR_WIDTH = 9;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [1-1:0] reg_full_sig;
    wire [4-1:0] reg_empty;
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
    wire [2-1:0] dpe_sel_h;

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
        .N_DPE_H(4),
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
    wire [DATA_WIDTH-1:0] dpe_out_c2_r0;
    wire dpe_done_c2_r0;
    wire reg_full_c2_r0;
    wire shift_add_done_c2_r0;
    wire shift_add_bypass_ctrl_c2_r0;
    wire MSB_SA_Ready_c2_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r0;
    wire dpe_done_c3_r0;
    wire reg_full_c3_r0;
    wire shift_add_done_c3_r0;
    wire shift_add_bypass_ctrl_c3_r0;
    wire MSB_SA_Ready_c3_r0;

    // DPE instantiations: 1 vertical x 4 horizontal = 4 DPEs
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

    dpe dpe_c2_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r0),
        .data_out(dpe_out_c2_r0),
        .dpe_done(dpe_done_c2_r0),
        .reg_full(reg_full_c2_r0),
        .shift_add_done(shift_add_done_c2_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r0)
    );

    dpe dpe_c3_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r0),
        .data_out(dpe_out_c3_r0),
        .dpe_done(dpe_done_c3_r0),
        .reg_full(reg_full_c3_r0),
        .shift_add_done(shift_add_done_c3_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r0)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c1_r0 | dpe_done_c2_r0 | dpe_done_c3_r0;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c1_r0 & shift_add_done_c2_r0 & shift_add_done_c3_r0;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c2_r0 & shift_add_bypass_ctrl_c3_r0;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c2_r0 & MSB_SA_Ready_c3_r0;
    assign reg_full_sig = {reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0, 1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Horizontal output mux (4 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            2'd0: data_out_mux = dpe_out_c0_r0;
            2'd1: data_out_mux = dpe_out_c1_r0;
            2'd2: data_out_mux = dpe_out_c2_r0;
            2'd3: data_out_mux = dpe_out_c3_r0;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

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
    reg out_valid;

    // Unpack 5 × int8 → 9-bit sign-extended for sop_4 ports
    wire [8:0] ax = {data_a[ 7], data_a[ 7: 0]};   // element 0
    wire [8:0] ay = {data_b[ 7], data_b[ 7: 0]};
    wire [8:0] bx = {data_a[15], data_a[15: 8]};   // element 1
    wire [8:0] by = {data_b[15], data_b[15: 8]};
    wire [8:0] cx = {data_a[23], data_a[23:16]};   // element 2
    wire [8:0] cy = {data_b[23], data_b[23:16]};
    wire [8:0] dx = {data_a[31], data_a[31:24]};   // element 3
    wire [8:0] dy = {data_b[31], data_b[31:24]};
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
    wire signed [DATA_WIDTH-1:0] cycle_sum = sop_result[DATA_WIDTH-1:0] + {{22{p4[17]}}, p4};
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
    // Multiply operations (27x27 -> 54) — parmys infers DSP from *
    // Two multipliers: variance (diff²) and normalize (diff * rsqrt)
    // ========================================================================

    // Variance multiplier: diff * diff
    reg  [MULT_W-1:0] var_mult_a, var_mult_b;
    wire [PROD_W-1:0] var_mult_out;
    assign var_mult_out = var_mult_a * var_mult_b;

    // Normalize multiplier: diff * rsqrt_val
    reg  [MULT_W-1:0] norm_mult_a, norm_mult_b;
    wire [PROD_W-1:0] norm_mult_out;
    assign norm_mult_out = norm_mult_a * norm_mult_b;

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
