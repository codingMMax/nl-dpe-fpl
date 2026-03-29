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
    assign data_out = data_b1_ln_ffn;
    assign valid_n = valid_b1_ln_ffn;
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
