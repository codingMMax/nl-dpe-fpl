// NL-DPE Attention Head RTL - Log-domain DIMM mapping
// 3 DPE hard blocks for Q, K, V linear projections (ACAM configured as log function)
// Score matrix and weighted sum use DIMM: add(log values) -> exp_LUT -> reduction
// This replaces standard multiply-accumulate with log-domain operations in CLB fabric.
//
// Data flow (matching IMC simulator attention model):
//   1. linear_Q/K/V: DPE projections with ACAM=log -> outputs are log(Wx)
//   2. dimm_score_matrix: S[i][j] = sum_m exp(log_Q[i][m] + log_K[j][m])
//      - CLB adder (log addition) + CLB exp LUT + CLB reduction adder
//   3. softmax: exp(S[i][j]) / sum_j exp(S[i][j])
//   4. dimm_weighted_sum: O[i][m] = sum_j exp(log(attn[i][j]) + log_V[j][m])
//      - CLB log LUT + CLB adder + CLB exp LUT + CLB reduction adder
//
// N=128 (sequence length), d=128 (head dimension), DATA_WIDTH=16

module attention_head #(
    parameter N = 128,        // sequence length
    parameter d = 128,        // head dimension
    parameter DATA_WIDTH = 16
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

    // ========================================================================
    // Internal signals
    // ========================================================================
    wire [DATA_WIDTH-1:0] data_out_q, data_out_k, data_out_v;
    wire ready_q, valid_q, ready_k, valid_k, ready_v, valid_v;
    wire ready_score, valid_score, ready_softmax, valid_softmax;
    wire ready_wsum, valid_wsum;
    wire valid_g_in, valid_g_out, ready_g_in, ready_g_out;
    wire [DATA_WIDTH-1:0] global_sram_data_in;

    reg [7:0] read_address, write_address;

    // ========================================================================
    // Stage 1: Q Projection DPE (single DPE, d=128 < 256 crossbar rows)
    // ACAM configured as log function (V=1 so ACAM eligible)
    // Output: log(W_Q * x) for each input token
    // ========================================================================
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(8),
        .N_KERNELS(1),
        .KERNEL_WIDTH(d),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(256),
        .DATA_WIDTH(DATA_WIDTH)
    ) q_projection (
        .clk(clk),
        .rst(rst),
        .valid(valid_g_out),
        .ready_n(ready_k),
        .data_in(data_in),
        .data_out(data_out_q),
        .ready(ready_g_in),
        .valid_n(valid_q)
    );

    // ========================================================================
    // Stage 2: K Projection DPE (single DPE, ACAM=log)
    // Output: log(W_K * x) for each input token
    // ========================================================================
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(8),
        .N_KERNELS(1),
        .KERNEL_WIDTH(d),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(256),
        .DATA_WIDTH(DATA_WIDTH)
    ) k_projection (
        .clk(clk),
        .rst(rst),
        .valid(valid_g_out),
        .ready_n(ready_v),
        .data_in(data_in),
        .data_out(data_out_k),
        .ready(),
        .valid_n(valid_k)
    );

    // ========================================================================
    // Stage 3: V Projection DPE (single DPE, ACAM=log)
    // Output: log(W_V * x) for each input token
    // ========================================================================
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(8),
        .N_KERNELS(1),
        .KERNEL_WIDTH(d),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(256),
        .DATA_WIDTH(DATA_WIDTH)
    ) v_projection (
        .clk(clk),
        .rst(rst),
        .valid(valid_g_out),
        .ready_n(ready_score),
        .data_in(data_in),
        .data_out(data_out_v),
        .ready(),
        .valid_n(valid_v)
    );

    // ========================================================================
    // Stage 4: DIMM Score Matrix (CLB-based log-domain computation)
    // S[i][j] = sum_m exp(log_Q[i][m] + log_K[j][m])
    // Replaces standard multiply-accumulate with add + exp_LUT + reduction
    // ========================================================================
    wire [DATA_WIDTH-1:0] data_out_score;

    dimm_score_matrix #(
        .N(N),
        .d(d),
        .DATA_WIDTH(DATA_WIDTH)
    ) score_inst (
        .clk(clk),
        .rst(rst),
        .valid_q(valid_q),
        .valid_k(valid_k),
        .ready_n(ready_softmax),
        .data_in_q(data_out_q),
        .data_in_k(data_out_k),
        .data_out(data_out_score),
        .ready_q(ready_q),
        .ready_k(ready_k),
        .valid_n(valid_score)
    );

    // ========================================================================
    // Stage 5: Softmax (CLB-based exp + normalization)
    // ========================================================================
    wire [DATA_WIDTH-1:0] data_out_softmax;

    softmax_approx #(
        .N(N),
        .d(d),
        .DATA_WIDTH(DATA_WIDTH)
    ) softmax_inst (
        .clk(clk),
        .rst(rst),
        .valid(valid_score),
        .ready_n(ready_wsum),
        .data_in(data_out_score),
        .data_out(data_out_softmax),
        .ready(ready_softmax),
        .valid_n(valid_softmax)
    );

    // ========================================================================
    // Stage 6: DIMM Weighted Sum (CLB-based log-domain computation)
    // O[i][m] = sum_j exp(log(attn[i][j]) + log_V[j][m])
    // attn weights need CLB log LUT; V values already in log domain from DPE
    // ========================================================================
    wire [DATA_WIDTH-1:0] data_out_wsum;

    dimm_weighted_sum #(
        .N(N),
        .d(d),
        .DATA_WIDTH(DATA_WIDTH)
    ) wsum_inst (
        .clk(clk),
        .rst(rst),
        .valid_attn(valid_softmax),
        .valid_v(valid_v),
        .ready_n(ready_g_out),
        .data_in_attn(data_out_softmax),
        .data_in_v(data_out_v),
        .data_out(data_out_wsum),
        .ready_attn(ready_wsum),
        .ready_v(ready_v),
        .valid_n(valid_g_in)
    );

    // ========================================================================
    // Global Controller
    // ========================================================================
    global_controller #(
        .N_Layers(6)
    ) g_ctrl_inst (
        .clk(clk),
        .rst(rst),
        .ready_L1(ready_g_in),
        .valid_Ln(valid_g_in),
        .valid(valid),
        .ready(ready),
        .valid_L1(valid_g_out),
        .ready_Ln(ready_g_out)
    );

    // ========================================================================
    // Global Output SRAM
    // ========================================================================
    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(256)
    ) global_sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(valid_g_in),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_out_wsum),
        .sram_data_out(global_sram_data_in)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            read_address <= 0;
            write_address <= 128;
        end else begin
            if (ready_g_out)
                read_address <= read_address + 1;
            if (valid_g_out)
                write_address <= write_address + 1;
        end
    end

    assign data_out = global_sram_data_in;
    assign valid_n = valid_g_in;

endmodule


// ============================================================================
// DIMM Score Matrix Module (CLB-based log-domain dot product)
//
// Computes S[i][j] = sum_m exp(log_Q[i][m] + log_K[j][m])  for all i,j
//
// Instead of CLB multiplier for Q*K, uses:
//   1. CLB adder: log_Q[m] + log_K[j*d+m]  (addition in log domain)
//   2. CLB exp LUT: exp(sum) via Taylor approximation
//   3. CLB reduction adder: accumulate exp values over d dimensions
//
// This matches the IMC simulator's gemm_log / DIMM operation for mac_qk.
// ============================================================================
module dimm_score_matrix #(
    parameter N = 128,
    parameter d = 128,
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 8,
    parameter DEPTH = 256
)(
    input wire clk,
    input wire rst,
    input wire valid_q,
    input wire valid_k,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_q,
    input wire [DATA_WIDTH-1:0] data_in_k,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_q,
    output wire ready_k,
    output wire valid_n
);

    // Q SRAM: stores d log-domain elements for current query token
    reg [ADDR_WIDTH-1:0] q_write_addr, q_read_addr;
    reg q_w_en;
    wire [DATA_WIDTH-1:0] q_sram_out;

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) q_sram (
        .clk(clk),
        .rst(rst),
        .w_en(q_w_en),
        .r_addr(q_read_addr),
        .w_addr(q_write_addr),
        .sram_data_in(data_in_q),
        .sram_data_out(q_sram_out)
    );

    // K SRAM: stores N*d log-domain elements for all key tokens
    reg [$clog2(N*d)-1:0] k_write_addr, k_read_addr;
    reg k_w_en;
    wire [DATA_WIDTH-1:0] k_sram_out;

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(N*d)
    ) k_sram (
        .clk(clk),
        .rst(rst),
        .w_en(k_w_en),
        .r_addr(k_read_addr[$clog2(N*d)-1:0]),
        .w_addr(k_write_addr[$clog2(N*d)-1:0]),
        .sram_data_in(data_in_k),
        .sram_data_out(k_sram_out)
    );

    // ---- Pipelined DIMM compute pipeline (CLB logic) ----
    // Pipeline stages to break critical path at multiply boundaries
    //   Stage 1 (registered): log_sum = q + k
    //   Stage 2 (registered): x_squared = log_sum * log_sum
    //   Stage 3 (registered): exp_val = 1 + log_sum + x_squared/2
    //   Accumulate: accumulator += exp_val
    // Pipeline latency = 3 cycles; throughput = 1 element/cycle after fill

    // Stage 1: Log-domain addition
    reg [DATA_WIDTH-1:0] pipe1_log_sum;
    reg pipe1_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pipe1_log_sum <= 0;
            pipe1_valid <= 0;
        end else begin
            pipe1_log_sum <= q_sram_out + k_sram_out;
            pipe1_valid <= (state == S_COMPUTE);
        end
    end

    // Stage 2: Multiply for exp approximation
    reg [2*DATA_WIDTH-1:0] pipe2_x_squared;
    reg [DATA_WIDTH-1:0] pipe2_log_sum;
    reg pipe2_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pipe2_x_squared <= 0;
            pipe2_log_sum <= 0;
            pipe2_valid <= 0;
        end else begin
            pipe2_x_squared <= pipe1_log_sum * pipe1_log_sum;
            pipe2_log_sum <= pipe1_log_sum;
            pipe2_valid <= pipe1_valid;
        end
    end

    // Stage 3: Exp approximation result
    reg [DATA_WIDTH-1:0] pipe3_exp_val;
    reg pipe3_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pipe3_exp_val <= 0;
            pipe3_valid <= 0;
        end else begin
            pipe3_exp_val <= {{(DATA_WIDTH-1){1'b0}}, 1'b1}
                           + pipe2_log_sum
                           + pipe2_x_squared[2*DATA_WIDTH-1:DATA_WIDTH+1];
            pipe3_valid <= pipe2_valid;
        end
    end

    // Reduction accumulator
    reg [2*DATA_WIDTH-1:0] accumulator;
    reg [$clog2(d)-1:0] mac_count;
    reg [$clog2(N)-1:0] score_idx;
    // Pipeline drain counter: after last address driven, wait 3 more cycles
    localparam PIPE_DEPTH = 3;
    reg [1:0] drain_count;

    // Output SRAM for scores
    reg [ADDR_WIDTH-1:0] score_write_addr, score_read_addr;
    reg score_w_en;
    wire [DATA_WIDTH-1:0] score_sram_out;
    reg [DATA_WIDTH-1:0] score_write_data;

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) score_sram (
        .clk(clk),
        .rst(rst),
        .w_en(score_w_en),
        .r_addr(score_read_addr),
        .w_addr(score_write_addr),
        .sram_data_in(score_write_data),
        .sram_data_out(score_sram_out)
    );

    // FSM states
    localparam S_IDLE    = 3'd0;
    localparam S_LOAD_Q  = 3'd1;
    localparam S_LOAD_K  = 3'd2;
    localparam S_COMPUTE = 3'd3;
    localparam S_OUTPUT  = 3'd4;

    reg [2:0] state, next_state;

    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    always @* begin
        next_state = state;
        case (state)
            S_IDLE:    if (valid_q || valid_k) next_state = S_LOAD_Q;
            S_LOAD_Q:  if (q_write_addr == d-1) next_state = S_LOAD_K;
            S_LOAD_K:  if (k_write_addr == N*d-1) next_state = S_COMPUTE;
            S_COMPUTE: begin
                if (drain_count == PIPE_DEPTH && score_idx == N-1)
                    next_state = S_OUTPUT;
            end
            S_OUTPUT:  if (!ready_n) next_state = S_IDLE;
            default:   next_state = S_IDLE;
        endcase
    end

    // Datapath
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            q_write_addr <= 0;
            q_read_addr <= 0;
            q_w_en <= 0;
            k_write_addr <= 0;
            k_read_addr <= 0;
            k_w_en <= 0;
            accumulator <= 0;
            mac_count <= 0;
            score_idx <= 0;
            drain_count <= 0;
            score_write_addr <= 0;
            score_read_addr <= 0;
            score_w_en <= 0;
            score_write_data <= 0;
        end else begin
            q_w_en <= 0;
            k_w_en <= 0;
            score_w_en <= 0;

            // Pipeline accumulation (runs regardless of address-driving state)
            if (pipe3_valid) begin
                accumulator <= accumulator + pipe3_exp_val;
            end

            case (state)
                S_IDLE: begin
                    q_write_addr <= 0;
                    k_write_addr <= 0;
                    score_write_addr <= 0;
                    score_idx <= 0;
                    mac_count <= 0;
                    drain_count <= 0;
                end

                S_LOAD_Q: begin
                    if (valid_q) begin
                        q_w_en <= 1;
                        q_write_addr <= q_write_addr + 1;
                    end
                end

                S_LOAD_K: begin
                    if (valid_k) begin
                        k_w_en <= 1;
                        k_write_addr <= k_write_addr + 1;
                    end
                end

                S_COMPUTE: begin
                    if (drain_count == 0) begin
                        // Drive SRAM addresses (feed pipeline)
                        q_read_addr <= mac_count;
                        k_read_addr <= (score_idx << $clog2(d)) + mac_count;

                        if (mac_count == d-1) begin
                            // Last address driven, start draining pipeline
                            drain_count <= 1;
                        end else begin
                            mac_count <= mac_count + 1;
                        end
                    end else begin
                        // Drain phase: wait for pipeline to flush
                        drain_count <= drain_count + 1;
                        if (drain_count == PIPE_DEPTH) begin
                            // Pipeline fully drained, store result
                            score_write_data <= accumulator[2*DATA_WIDTH-1:DATA_WIDTH];
                            score_w_en <= 1;
                            score_write_addr <= score_idx;
                            accumulator <= 0;
                            mac_count <= 0;
                            drain_count <= 0;
                            if (score_idx < N-1)
                                score_idx <= score_idx + 1;
                        end
                    end
                end

                S_OUTPUT: begin
                    if (ready_n) begin
                        score_read_addr <= score_read_addr + 1;
                    end
                end
            endcase
        end
    end

    assign data_out = score_sram_out;
    assign valid_n = (state == S_OUTPUT);
    assign ready_q = (state == S_LOAD_Q || state == S_IDLE);
    assign ready_k = (state == S_LOAD_K || state == S_IDLE);

endmodule


// ============================================================================
// Softmax Approximation Module (CLB-based)
// Implements exp + normalization in CLB fabric
// Three passes: find-max, exp+sum, normalize
// ============================================================================
module softmax_approx #(
    parameter N = 128,
    parameter d = 128,
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 8,
    parameter DEPTH = 256
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

    // Input SRAM: buffer scores
    reg [ADDR_WIDTH-1:0] in_write_addr, in_read_addr;
    reg in_w_en;
    wire [DATA_WIDTH-1:0] in_sram_out;

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) in_sram (
        .clk(clk),
        .rst(rst),
        .w_en(in_w_en),
        .r_addr(in_read_addr),
        .w_addr(in_write_addr),
        .sram_data_in(data_in),
        .sram_data_out(in_sram_out)
    );

    // Pipelined exp approximation (CLB): exp(x) ~ 1 + x + x^2/2
    // Stage 1: multiply (registered)
    // Stage 2: exp result (registered)
    reg [2*DATA_WIDTH-1:0] sm_pipe1_x_squared;
    reg [DATA_WIDTH-1:0] sm_pipe1_x;
    reg sm_pipe1_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sm_pipe1_x_squared <= 0;
            sm_pipe1_x <= 0;
            sm_pipe1_valid <= 0;
        end else begin
            sm_pipe1_x_squared <= in_sram_out * in_sram_out;
            sm_pipe1_x <= in_sram_out;
            sm_pipe1_valid <= (sm_state == SM_EXP_SUM);
        end
    end

    reg [DATA_WIDTH-1:0] exp_val;
    reg sm_pipe2_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            exp_val <= 0;
            sm_pipe2_valid <= 0;
        end else begin
            exp_val <= {{(DATA_WIDTH-1){1'b0}}, 1'b1} + sm_pipe1_x + sm_pipe1_x_squared[2*DATA_WIDTH-1:DATA_WIDTH+1];
            sm_pipe2_valid <= sm_pipe1_valid;
        end
    end

    // Sum accumulator for normalization
    reg [2*DATA_WIDTH-1:0] exp_sum;
    reg [DATA_WIDTH-1:0] max_score;

    localparam SM_IDLE      = 3'd0;
    localparam SM_LOAD      = 3'd1;
    localparam SM_FIND_MAX  = 3'd2;
    localparam SM_EXP_SUM   = 3'd3;
    localparam SM_EXP_DRAIN = 3'd4;
    localparam SM_NORMALIZE = 3'd5;
    localparam SM_NORM_DRAIN= 3'd6;
    localparam SM_OUTPUT    = 3'd7;

    reg [2:0] sm_state, sm_next_state;
    reg [$clog2(N)-1:0] sm_count;
    reg [1:0] sm_drain;
    // Separate write counters that follow the pipeline
    reg [ADDR_WIDTH-1:0] sm_exp_wr_cnt;
    reg [ADDR_WIDTH-1:0] sm_norm_wr_cnt;

    // Exp values SRAM
    reg [ADDR_WIDTH-1:0] exp_write_addr, exp_read_addr;
    reg exp_w_en;
    wire [DATA_WIDTH-1:0] exp_sram_out;

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) exp_sram (
        .clk(clk),
        .rst(rst),
        .w_en(exp_w_en),
        .r_addr(exp_read_addr),
        .w_addr(exp_write_addr),
        .sram_data_in(exp_val),
        .sram_data_out(exp_sram_out)
    );

    // Output SRAM for normalized values
    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    reg [DATA_WIDTH-1:0] norm_val;
    wire [DATA_WIDTH-1:0] out_sram_out;

    // Reciprocal approximation: priority encoder + shift (O(log n) depth)
    // Replaces combinational divider ({16{1'b1}} / x) which was ~50 LUT levels
    wire [DATA_WIDTH-1:0] recip_approx;
    wire [DATA_WIDTH-1:0] exp_sum_upper = exp_sum[2*DATA_WIDTH-1:DATA_WIDTH];
    reg [DATA_WIDTH-1:0] recip_val;
    always @* begin
        casez (exp_sum_upper)
            16'b1???????????????: recip_val = 16'd2;
            16'b01??????????????: recip_val = 16'd4;
            16'b001?????????????: recip_val = 16'd8;
            16'b0001????????????: recip_val = 16'd16;
            16'b00001???????????: recip_val = 16'd32;
            16'b000001??????????: recip_val = 16'd64;
            16'b0000001?????????: recip_val = 16'd128;
            16'b00000001????????: recip_val = 16'd256;
            16'b000000001???????: recip_val = 16'd512;
            16'b0000000001??????: recip_val = 16'd1024;
            16'b00000000001?????: recip_val = 16'd2048;
            16'b000000000001????: recip_val = 16'd4096;
            16'b0000000000001???: recip_val = 16'd8192;
            16'b00000000000001??: recip_val = 16'd16384;
            16'b000000000000001?: recip_val = 16'd32768;
            16'b0000000000000001: recip_val = 16'hFFFF;
            default:             recip_val = 16'hFFFF;
        endcase
    end
    assign recip_approx = recip_val;

    // Pipelined normalization multiply
    reg [DATA_WIDTH-1:0] inv_sum;
    reg [2*DATA_WIDTH-1:0] norm_pipe1_product;
    reg norm_pipe1_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            norm_pipe1_product <= 0;
            norm_pipe1_valid <= 0;
        end else begin
            norm_pipe1_product <= exp_sram_out * inv_sum;
            norm_pipe1_valid <= (sm_state == SM_NORMALIZE);
        end
    end

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) out_sram (
        .clk(clk),
        .rst(rst),
        .w_en(out_w_en),
        .r_addr(out_read_addr),
        .w_addr(out_write_addr),
        .sram_data_in(norm_val),
        .sram_data_out(out_sram_out)
    );

    always @(posedge clk or posedge rst) begin
        if (rst)
            sm_state <= SM_IDLE;
        else
            sm_state <= sm_next_state;
    end

    always @* begin
        sm_next_state = sm_state;
        case (sm_state)
            SM_IDLE:       if (valid) sm_next_state = SM_LOAD;
            SM_LOAD:       if (in_write_addr == N-1) sm_next_state = SM_FIND_MAX;
            SM_FIND_MAX:   if (sm_count == N-1) sm_next_state = SM_EXP_SUM;
            SM_EXP_SUM:    if (sm_count == N-1) sm_next_state = SM_EXP_DRAIN;
            SM_EXP_DRAIN:  if (sm_drain == 2) sm_next_state = SM_NORMALIZE;
            SM_NORMALIZE:  if (sm_count == N-1) sm_next_state = SM_NORM_DRAIN;
            SM_NORM_DRAIN: if (sm_drain == 1) sm_next_state = SM_OUTPUT;
            SM_OUTPUT:     if (!ready_n) sm_next_state = SM_IDLE;
            default:       sm_next_state = SM_IDLE;
        endcase
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            in_write_addr <= 0;
            in_read_addr <= 0;
            in_w_en <= 0;
            exp_write_addr <= 0;
            exp_read_addr <= 0;
            exp_w_en <= 0;
            out_write_addr <= 0;
            out_read_addr <= 0;
            out_w_en <= 0;
            sm_count <= 0;
            sm_drain <= 0;
            sm_exp_wr_cnt <= 0;
            sm_norm_wr_cnt <= 0;
            max_score <= 0;
            exp_sum <= 0;
            inv_sum <= 0;
            norm_val <= 0;
        end else begin
            in_w_en <= 0;
            exp_w_en <= 0;
            out_w_en <= 0;

            // Pipeline-driven exp SRAM writes and accumulation (works in EXP_SUM and EXP_DRAIN)
            if (sm_pipe2_valid) begin
                exp_w_en <= 1;
                exp_write_addr <= sm_exp_wr_cnt;
                sm_exp_wr_cnt <= sm_exp_wr_cnt + 1;
                exp_sum <= exp_sum + exp_val;
            end

            // Pipeline-driven norm SRAM writes (works in NORMALIZE and NORM_DRAIN)
            if (norm_pipe1_valid) begin
                norm_val <= norm_pipe1_product[2*DATA_WIDTH-1:DATA_WIDTH];
                out_w_en <= 1;
                out_write_addr <= sm_norm_wr_cnt;
                sm_norm_wr_cnt <= sm_norm_wr_cnt + 1;
            end

            case (sm_state)
                SM_IDLE: begin
                    in_write_addr <= 0;
                    sm_count <= 0;
                    sm_drain <= 0;
                    sm_exp_wr_cnt <= 0;
                    sm_norm_wr_cnt <= 0;
                    max_score <= 0;
                    exp_sum <= 0;
                end

                SM_LOAD: begin
                    if (valid) begin
                        in_w_en <= 1;
                        in_write_addr <= in_write_addr + 1;
                    end
                end

                SM_FIND_MAX: begin
                    in_read_addr <= sm_count;
                    if (sm_count > 0 && in_sram_out > max_score)
                        max_score <= in_sram_out;
                    sm_count <= sm_count + 1;
                    if (sm_count == N-1) sm_count <= 0;
                end

                SM_EXP_SUM: begin
                    // Drive SRAM read addresses
                    in_read_addr <= sm_count;
                    sm_count <= sm_count + 1;
                    if (sm_count == N-1) begin
                        sm_count <= 0;
                        sm_drain <= 0;
                    end
                end

                SM_EXP_DRAIN: begin
                    // Wait for pipeline to flush (2 cycles)
                    sm_drain <= sm_drain + 1;
                    if (sm_drain == 2) begin
                        // Reciprocal via priority encoder (O(log n) depth)
                        // replaces combinational divider that was the critical path
                        inv_sum <= recip_approx;
                        sm_drain <= 0;
                    end
                end

                SM_NORMALIZE: begin
                    // Drive exp SRAM read addresses
                    exp_read_addr <= sm_count;
                    sm_count <= sm_count + 1;
                    if (sm_count == N-1) begin
                        sm_count <= 0;
                        sm_drain <= 0;
                    end
                end

                SM_NORM_DRAIN: begin
                    // Wait for norm pipeline to flush (1 cycle)
                    sm_drain <= sm_drain + 1;
                    if (sm_drain == 1) sm_drain <= 0;
                end

                SM_OUTPUT: begin
                    if (ready_n) begin
                        out_read_addr <= out_read_addr + 1;
                    end
                end
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (sm_state == SM_OUTPUT);
    assign ready = (sm_state == SM_LOAD || sm_state == SM_IDLE);

endmodule


// ============================================================================
// DIMM Weighted Sum Module (CLB-based log-domain computation)
//
// Computes O[i][m] = sum_j exp(log(attn[i][j]) + log_V[j][m])  for all m
//
// attn weights (from softmax) are NOT in log domain, so we apply CLB log LUT.
// V values are already in log domain (from DPE with ACAM=log).
//
// CLB pipeline per element:
//   1. log_LUT(attn[j])        -> CLB log approximation
//   2. log_attn + log_V[j][m]  -> CLB adder
//   3. exp(sum)                -> CLB exp approximation
//   4. accumulate              -> CLB reduction adder
//
// This matches the IMC simulator's gemm_log / DIMM operation for mac_sv.
// ============================================================================
module dimm_weighted_sum #(
    parameter N = 128,
    parameter d = 128,
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 8,
    parameter DEPTH = 256
)(
    input wire clk,
    input wire rst,
    input wire valid_attn,
    input wire valid_v,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in_attn,
    input wire [DATA_WIDTH-1:0] data_in_v,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready_attn,
    output wire ready_v,
    output wire valid_n
);

    // Attention weights SRAM (N values, linear domain from softmax)
    reg [ADDR_WIDTH-1:0] attn_write_addr, attn_read_addr;
    reg attn_w_en;
    wire [DATA_WIDTH-1:0] attn_sram_out;

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) attn_sram (
        .clk(clk),
        .rst(rst),
        .w_en(attn_w_en),
        .r_addr(attn_read_addr),
        .w_addr(attn_write_addr),
        .sram_data_in(data_in_attn),
        .sram_data_out(attn_sram_out)
    );

    // V matrix SRAM (N*d values, already in log domain from DPE)
    reg [$clog2(N*d)-1:0] v_write_addr, v_read_addr;
    reg v_w_en;
    wire [DATA_WIDTH-1:0] v_sram_out;

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(N*d)
    ) v_sram (
        .clk(clk),
        .rst(rst),
        .w_en(v_w_en),
        .r_addr(v_read_addr[$clog2(N*d)-1:0]),
        .w_addr(v_write_addr[$clog2(N*d)-1:0]),
        .sram_data_in(data_in_v),
        .sram_data_out(v_sram_out)
    );

    // Output SRAM (d values)
    reg [ADDR_WIDTH-1:0] out_write_addr, out_read_addr;
    reg out_w_en;
    reg [DATA_WIDTH-1:0] out_data;
    wire [DATA_WIDTH-1:0] out_sram_out;

    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) out_sram (
        .clk(clk),
        .rst(rst),
        .w_en(out_w_en),
        .r_addr(out_read_addr),
        .w_addr(out_write_addr),
        .sram_data_in(out_data),
        .sram_data_out(out_sram_out)
    );

    // ---- Pipelined DIMM compute pipeline (CLB logic) ----
    // 4 pipeline stages to break both multiply critical paths:
    //   Stage 1 (registered): attn_sq = attn * attn  (log approx multiply)
    //   Stage 2 (registered): log_attn = attn - attn_sq/2; log_sum = log_attn + v
    //   Stage 3 (registered): sum_sq = log_sum * log_sum  (exp approx multiply)
    //   Stage 4 (registered): exp_val = 1 + log_sum + sum_sq/2
    //   Accumulate: accumulator += exp_val
    // Pipeline latency = 4 cycles; throughput = 1 element/cycle after fill

    // Stage 1: Log multiply (attn^2)
    reg [2*DATA_WIDTH-1:0] ws_pipe1_attn_sq;
    reg [DATA_WIDTH-1:0] ws_pipe1_attn;
    reg [DATA_WIDTH-1:0] ws_pipe1_v;
    reg ws_pipe1_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ws_pipe1_attn_sq <= 0;
            ws_pipe1_attn <= 0;
            ws_pipe1_v <= 0;
            ws_pipe1_valid <= 0;
        end else begin
            ws_pipe1_attn_sq <= attn_sram_out * attn_sram_out;
            ws_pipe1_attn <= attn_sram_out;
            ws_pipe1_v <= v_sram_out;
            ws_pipe1_valid <= (ws_state == WS_COMPUTE && ws_addr_active);
        end
    end

    // Stage 2: Log subtraction + addition
    reg [DATA_WIDTH-1:0] ws_pipe2_log_sum;
    reg ws_pipe2_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ws_pipe2_log_sum <= 0;
            ws_pipe2_valid <= 0;
        end else begin
            ws_pipe2_log_sum <= (ws_pipe1_attn - ws_pipe1_attn_sq[2*DATA_WIDTH-1:DATA_WIDTH+1])
                              + ws_pipe1_v;
            ws_pipe2_valid <= ws_pipe1_valid;
        end
    end

    // Stage 3: Exp multiply (log_sum^2)
    reg [2*DATA_WIDTH-1:0] ws_pipe3_sum_sq;
    reg [DATA_WIDTH-1:0] ws_pipe3_log_sum;
    reg ws_pipe3_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ws_pipe3_sum_sq <= 0;
            ws_pipe3_log_sum <= 0;
            ws_pipe3_valid <= 0;
        end else begin
            ws_pipe3_sum_sq <= ws_pipe2_log_sum * ws_pipe2_log_sum;
            ws_pipe3_log_sum <= ws_pipe2_log_sum;
            ws_pipe3_valid <= ws_pipe2_valid;
        end
    end

    // Stage 4: Exp result
    reg [DATA_WIDTH-1:0] ws_pipe4_exp_val;
    reg ws_pipe4_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ws_pipe4_exp_val <= 0;
            ws_pipe4_valid <= 0;
        end else begin
            ws_pipe4_exp_val <= {{(DATA_WIDTH-1){1'b0}}, 1'b1}
                              + ws_pipe3_log_sum
                              + ws_pipe3_sum_sq[2*DATA_WIDTH-1:DATA_WIDTH+1];
            ws_pipe4_valid <= ws_pipe3_valid;
        end
    end

    // Reduction accumulator
    reg [2*DATA_WIDTH-1:0] accumulator;

    // FSM
    localparam WS_IDLE      = 3'd0;
    localparam WS_LOAD_ATTN = 3'd1;
    localparam WS_LOAD_V    = 3'd2;
    localparam WS_COMPUTE   = 3'd3;
    localparam WS_OUTPUT    = 3'd4;

    localparam WS_PIPE_DEPTH = 4;

    reg [2:0] ws_state, ws_next_state;
    reg [$clog2(N)-1:0] ws_n_count;
    reg [$clog2(d)-1:0] ws_d_count;
    reg [2:0] ws_drain_count;
    reg ws_addr_active;  // 1 when driving addresses, 0 during drain

    // Track pipeline outputs per dimension for store decisions
    reg [$clog2(N)-1:0] ws_pipe_out_count;
    reg ws_dim_done;

    always @(posedge clk or posedge rst) begin
        if (rst)
            ws_state <= WS_IDLE;
        else
            ws_state <= ws_next_state;
    end

    always @* begin
        ws_next_state = ws_state;
        case (ws_state)
            WS_IDLE:      if (valid_attn || valid_v) ws_next_state = WS_LOAD_ATTN;
            WS_LOAD_ATTN: if (attn_write_addr == N-1) ws_next_state = WS_LOAD_V;
            WS_LOAD_V:    if (v_write_addr == N*d-1) ws_next_state = WS_COMPUTE;
            WS_COMPUTE: begin
                if (ws_dim_done && ws_d_count == d-1)
                    ws_next_state = WS_OUTPUT;
            end
            WS_OUTPUT:    if (!ready_n) ws_next_state = WS_IDLE;
            default:      ws_next_state = WS_IDLE;
        endcase
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            attn_write_addr <= 0;
            attn_read_addr <= 0;
            attn_w_en <= 0;
            v_write_addr <= 0;
            v_read_addr <= 0;
            v_w_en <= 0;
            out_write_addr <= 0;
            out_read_addr <= 0;
            out_w_en <= 0;
            out_data <= 0;
            accumulator <= 0;
            ws_n_count <= 0;
            ws_d_count <= 0;
            ws_drain_count <= 0;
            ws_addr_active <= 0;
            ws_pipe_out_count <= 0;
            ws_dim_done <= 0;
        end else begin
            attn_w_en <= 0;
            v_w_en <= 0;
            out_w_en <= 0;
            ws_dim_done <= 0;

            // Pipeline accumulation (runs in COMPUTE state)
            if (ws_pipe4_valid) begin
                accumulator <= accumulator + ws_pipe4_exp_val;
                ws_pipe_out_count <= ws_pipe_out_count + 1;
            end

            case (ws_state)
                WS_IDLE: begin
                    attn_write_addr <= 0;
                    v_write_addr <= 0;
                    out_write_addr <= 0;
                    ws_n_count <= 0;
                    ws_d_count <= 0;
                    ws_drain_count <= 0;
                    ws_addr_active <= 0;
                    ws_pipe_out_count <= 0;
                    accumulator <= 0;
                end

                WS_LOAD_ATTN: begin
                    if (valid_attn) begin
                        attn_w_en <= 1;
                        attn_write_addr <= attn_write_addr + 1;
                    end
                end

                WS_LOAD_V: begin
                    if (valid_v) begin
                        v_w_en <= 1;
                        v_write_addr <= v_write_addr + 1;
                    end
                end

                WS_COMPUTE: begin
                    if (ws_addr_active == 0 && ws_drain_count == 0) begin
                        // Start new dimension: drive first address
                        ws_addr_active <= 1;
                        ws_n_count <= 0;
                        ws_pipe_out_count <= 0;
                        attn_read_addr <= 0;
                        v_read_addr <= ws_d_count;
                    end else if (ws_addr_active) begin
                        // Drive SRAM addresses (feed pipeline)
                        attn_read_addr <= ws_n_count + 1;
                        v_read_addr <= ((ws_n_count + 1) << $clog2(d)) + ws_d_count;

                        if (ws_n_count == N-1) begin
                            // Last address driven, start draining
                            ws_addr_active <= 0;
                            ws_drain_count <= 1;
                        end else begin
                            ws_n_count <= ws_n_count + 1;
                        end
                    end else begin
                        // Drain phase: wait for pipeline to flush
                        ws_drain_count <= ws_drain_count + 1;
                        if (ws_drain_count == WS_PIPE_DEPTH) begin
                            // Pipeline fully drained, store result
                            out_data <= accumulator[2*DATA_WIDTH-1:DATA_WIDTH];
                            out_w_en <= 1;
                            out_write_addr <= ws_d_count;
                            accumulator <= 0;
                            ws_drain_count <= 0;
                            ws_dim_done <= 1;
                            if (ws_d_count < d-1)
                                ws_d_count <= ws_d_count + 1;
                        end
                    end
                end

                WS_OUTPUT: begin
                    if (ready_n) begin
                        out_read_addr <= out_read_addr + 1;
                    end
                end
            endcase
        end
    end

    assign data_out = out_sram_out;
    assign valid_n = (ws_state == WS_OUTPUT);
    assign ready_attn = (ws_state == WS_LOAD_ATTN || ws_state == WS_IDLE);
    assign ready_v = (ws_state == WS_LOAD_V || ws_state == WS_IDLE);

endmodule


// ============================================================================
// Supporting Modules (reused from resnet_1_channel.v for VTR self-containment)
// ============================================================================

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
    parameter DATA_WIDTH = 16
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


module sram #(
    parameter N_CHANNELS = 1,
    parameter DATA_WIDTH = 16*N_CHANNELS,
    parameter DEPTH = 512
)(
    input wire clk,
    input wire w_en,
    input wire rst,
    input wire [$clog2(DEPTH)-1:0] r_addr,
    input wire [$clog2(DEPTH)-1:0] w_addr,
    input wire [DATA_WIDTH-1:0] sram_data_in,
    output reg [DATA_WIDTH-1:0] sram_data_out
);

    reg [DATA_WIDTH-1:0] mem [DEPTH-1:0];

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
