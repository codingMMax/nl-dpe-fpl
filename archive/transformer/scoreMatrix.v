module scoreMatrix #(
    parameter N = 64,
    parameter d = 256,
    parameter DATA_WIDTH = 8
) (

    input clk,
    input rst,
    // Q input of size 4x1 per batch
    input [(4*DATA_WIDTH)-1:0] q_data_in,
    input q_valid_in,
    output q_ready_in,

    // K input of size 4xN per batch
    input [(4*N*DATA_WIDTH)-1:0] k_data_in,
    input k_valid_in,
    output k_ready_in,

    input start,  // indicated start of computation for a (Nxd) batch
    input in_last, // indicate last dimension

    // Output score matrix of size (1xN)
    output [(N*DATA_WIDTH)-1:0] score_data_out,
    output [DATA_WIDTH-1:0] score_sum_out, // output 1/sum of exp scores for softmax normalization
    output score_valid_out,
    input score_ready_out
);

    // Internal widths
    localparam SUM_WIDTH = 16;  // DSP result is 16-bit
    localparam SCALE_SHIFT = $clog2(d) >> 1;  // Shift for approx /sqrt(d)
    // store exp results (DATA_WIDTH bits)
    reg [DATA_WIDTH-1:0] score_array [0:N-1];
    reg [SUM_WIDTH-1:0] score_sum_reg;

    // Accumulators for dot-products
    reg [SUM_WIDTH-1:0] accum [0:N-1];

    // Exp LUT output
    wire [DATA_WIDTH-1:0] exp_result [0:N-1];

    // Exponential LUT
    // TODO: Ruthwik: Update LUT values based on precision used in simulator/ parameterize LUT depth
    genvar ei;
    generate
        for (ei = 0; ei < N; ei = ei + 1) begin : exp_lut_gen
            wire [SUM_WIDTH-1:0] scaled_accum = accum[ei] >> SCALE_SHIFT;  // divide by sqrt(d)
            reg [DATA_WIDTH-1:0] lut_out;
            always @(*) begin
                if (scaled_accum == 0) lut_out = 8'd0;    // exp(0) = 1, scaled = 0.012
                else if (scaled_accum == 1) lut_out = 8'd0;    // exp(1) = 2.7, scaled = 0.031
                else if (scaled_accum == 2) lut_out = 8'd0;    // exp(2) = 7.4, scaled = 0.085
                else if (scaled_accum == 3) lut_out = 8'd0;    // exp(3) = 20.1, scaled = 0.232
                else if (scaled_accum == 4) lut_out = 8'd1;    // exp(4) = 54.6, scaled = 0.631
                else if (scaled_accum == 5) lut_out = 8'd2;    // exp(5) = 148.4, scaled = 1.716
                else if (scaled_accum == 6) lut_out = 8'd5;    // exp(6) = 403.4, scaled = 4.67
                else if (scaled_accum == 7) lut_out = 8'd13;   // exp(7) = 1096.6, scaled = 12.68
                else if (scaled_accum == 8) lut_out = 8'd34;   // exp(8) = 2981, scaled = 34.46
                else if (scaled_accum == 9) lut_out = 8'd94;   // exp(9) = 8103, scaled = 93.75
                else if (scaled_accum == 10) lut_out = 8'd255; // exp(10) = 22026, scaled = 255
                else lut_out = 8'd255;  // saturate for >10
            end
            assign exp_result[ei] = lut_out;
        end
    endgenerate

    // FSM states
    localparam S_IDLE = 2'd0;
    localparam S_COLLECT = 2'd1;
    localparam S_PROCESS = 2'd2;
    localparam S_OUTPUT_WAIT = 2'd3;
    localparam S_OUTPUT = 2'd4;

    reg [2:0] state;
    reg [$clog2(d)-1:0] dim_count;
    reg [1:0] pipeline_stage;  // 0,1,2 for 3 stages
    reg q_latched_flag;

    // ready/valid outputs
    reg q_ready_reg;
    reg k_ready_reg;
    reg score_valid_reg;

    assign q_ready_in = q_ready_reg;
    assign k_ready_in = k_ready_reg;
    assign score_valid_out = score_valid_reg;

    // Latch Q and K per batch of 4 dimensions
    reg [(4*DATA_WIDTH)-1:0] q_latched;
    reg [(4*N*DATA_WIDTH)-1:0] k_latched;

    // Pack scores to output 
    integer ii;
    reg [N*DATA_WIDTH-1:0] score_pack_reg;
    always @(*) begin
        score_pack_reg = {N*DATA_WIDTH{1'b0}};
        for (ii = 0; ii < N; ii = ii + 1) begin
            // take the least-significant DATA_WIDTH bits of each computed score
            score_pack_reg[ii*DATA_WIDTH +: DATA_WIDTH] = score_array[ii][DATA_WIDTH-1:0];
        end
    end
    assign score_data_out = score_pack_reg;
    assign score_sum_out = reciprocal_result; // 1/sum for softmax normalization


    // DSP for accumulation: accum[j] + sum_{k=0 to 3} q[k] * k[j][k]
    wire [SUM_WIDTH-1:0] new_accum [0:N-1];
    genvar j;
    generate
        for (j = 0; j < N; j = j + 1) begin : accum_dsp
            int_sop_4 mac_inst (
                .a0(q_latched[0*DATA_WIDTH +: DATA_WIDTH]),
                .a1(q_latched[1*DATA_WIDTH +: DATA_WIDTH]),
                .a2(q_latched[2*DATA_WIDTH +: DATA_WIDTH]),
                .a3(q_latched[3*DATA_WIDTH +: DATA_WIDTH]),
                .b0(k_latched[0*N*DATA_WIDTH + j*DATA_WIDTH +: DATA_WIDTH]),
                .b1(k_latched[1*N*DATA_WIDTH + j*DATA_WIDTH +: DATA_WIDTH]),
                .b2(k_latched[2*N*DATA_WIDTH + j*DATA_WIDTH +: DATA_WIDTH]),
                .b3(k_latched[3*N*DATA_WIDTH + j*DATA_WIDTH +: DATA_WIDTH]),
                .chainin(accum[j]),
                .chainout(),
                .result(new_accum[j])
            );
        end
    endgenerate


    // Dynamic adder tree parameters based on N (assuming N is multiple of 4)
    localparam L1_DSPS = N / 4;
    localparam L2_DSPS = L1_DSPS / 4;
    localparam L3_DSPS = L2_DSPS / 4;
    localparam L4_DSPS = (L3_DSPS > 1) ? 1 : 0;  // 1 DSP if L3_DSPS > 1

    // Adder tree wires
    wire [SUM_WIDTH-1:0] level1_sum [0:L1_DSPS-1];
    wire [SUM_WIDTH-1:0] level2_sum [0:L2_DSPS-1];
    wire [SUM_WIDTH-1:0] level3_sum [0:L3_DSPS-1];
    wire [SUM_WIDTH-1:0] level4_sum;

    // Pipeline registers for adder tree
    reg [SUM_WIDTH-1:0] level1_reg [0:L1_DSPS-1];
    reg [SUM_WIDTH-1:0] level2_reg [0:L2_DSPS-1];

    genvar l1, l2, l3;
    generate
        // Level 1: L1_DSPS DSPs, each summing 4 exp_result
        for (l1 = 0; l1 < L1_DSPS; l1 = l1 + 1) begin : level1
            int_sop_4 adder_l1 (
                .a0({{(SUM_WIDTH-DATA_WIDTH){1'b0}}, exp_result[4*l1]}),
                .a1({{(SUM_WIDTH-DATA_WIDTH){1'b0}}, exp_result[4*l1+1]}),
                .a2({{(SUM_WIDTH-DATA_WIDTH){1'b0}}, exp_result[4*l1+2]}),
                .a3({{(SUM_WIDTH-DATA_WIDTH){1'b0}}, exp_result[4*l1+3]}),
                .b0(8'd1), .b1(8'd1), .b2(8'd1), .b3(8'd1),
                .chainin({SUM_WIDTH{1'b0}}),
                .chainout(),
                .result(level1_sum[l1])
            );
        end

        // Level 2: L2_DSPS DSPs, each summing 4 level1_reg
        for (l2 = 0; l2 < L2_DSPS; l2 = l2 + 1) begin : level2
            int_sop_4 adder_l2 (
                .a0(level1_reg[4*l2]),
                .a1(level1_reg[4*l2+1]),
                .a2(level1_reg[4*l2+2]),
                .a3(level1_reg[4*l2+3]),
                .b0(8'd1), .b1(8'd1), .b2(8'd1), .b3(8'd1),
                .chainin({SUM_WIDTH{1'b0}}),
                .chainout(),
                .result(level2_sum[l2])
            );
        end

        // Level 3: L3_DSPS DSPs, each summing 4 level2_reg
        for (l3 = 0; l3 < L3_DSPS; l3 = l3 + 1) begin : level3
            int_sop_4 adder_l3 (
                .a0(level2_reg[4*l3]),
                .a1(level2_reg[4*l3+1]),
                .a2(level2_reg[4*l3+2]),
                .a3(level2_reg[4*l3+3]),
                .b0(8'd1), .b1(8'd1), .b2(8'd1), .b3(8'd1),
                .chainin({SUM_WIDTH{1'b0}}),
                .chainout(),
                .result(level3_sum[l3])
            );
        end

        // Level 4: if L3_DSPS > 1, sum level3_sum with DSP
        if (L3_DSPS == 1) begin
            assign level4_sum = level3_sum[0];
        end else begin
            int_sop_4 adder_l4 (
                .a0(level3_sum[0]),
                .a1(level3_sum[1]),
                .a2((L3_DSPS > 2) ? level3_sum[2] : {SUM_WIDTH{1'b0}}),
                .a3((L3_DSPS > 3) ? level3_sum[3] : {SUM_WIDTH{1'b0}}),
                .b0(8'd1), .b1(8'd1), .b2(8'd1), .b3(8'd1),
                .chainin({SUM_WIDTH{1'b0}}),
                .chainout(),
                .result(level4_sum)
            );
        end
    endgenerate


    // LUT for reciprocal of sum (1/sum), output 8-bit fixed-point (0=0.0, 255=1.0)
    wire [DATA_WIDTH-1:0] reciprocal_result;
    reg [DATA_WIDTH-1:0] lut_recip;
    always @(*) begin
        case (score_sum_reg[11:0])  // Index on lower 12 bits of sum
            12'd1: lut_recip = 8'd255;  // 1/1 = 1.0
            12'd2: lut_recip = 8'd128;  // 1/2 = 0.5
            12'd3: lut_recip = 8'd85;   // 1/3 = 0.333
            12'd4: lut_recip = 8'd64;   // 1/4 = 0.25
            12'd5: lut_recip = 8'd51;   // 1/5 = 0.2
            12'd6: lut_recip = 8'd43;   // 1/6 = 0.167
            12'd7: lut_recip = 8'd36;   // 1/7 = 0.143
            12'd8: lut_recip = 8'd32;   // 1/8 = 0.125
            12'd9: lut_recip = 8'd28;   // 1/9 = 0.111
            12'd10: lut_recip = 8'd26;  // 1/10 = 0.1
            12'd11: lut_recip = 8'd23;  // 1/11 = 0.091
            12'd12: lut_recip = 8'd21;  // 1/12 = 0.083
            12'd13: lut_recip = 8'd20;  // 1/13 = 0.077
            12'd14: lut_recip = 8'd18;  // 1/14 = 0.071
            12'd15: lut_recip = 8'd17;  // 1/15 = 0.067
            12'd16: lut_recip = 8'd16;  // 1/16 = 0.0625
            12'd17: lut_recip = 8'd15;  // 1/17 = 0.059
            12'd18: lut_recip = 8'd14;  // 1/18 = 0.056
            12'd19: lut_recip = 8'd13;  // 1/19 = 0.053
            12'd20: lut_recip = 8'd13;  // 1/20 = 0.05
            // For larger values, approximate as small value - 1
            default: lut_recip = (score_sum_reg > 20) ? 8'd1 : 8'd255;  
        endcase
    end
    assign reciprocal_result = lut_recip;





    // main FSM
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            dim_count <= 0;
            pipeline_stage <= 0;
            q_ready_reg <= 1'b1;
            k_ready_reg <= 1'b1;
            score_valid_reg <= 1'b0;
            score_sum_reg <= 0;
            for (ii = 0; ii < N; ii = ii + 1) begin
                score_array[ii] <= 0;
                accum[ii] <= 0;
            end
        end else begin
            case (state)
                S_IDLE: begin
                    score_valid_reg <= 1'b0;
                    score_sum_reg <= 0;
                    dim_count <= 0;
                    q_ready_reg <= 1'b1;
                    k_ready_reg <= 1'b1;
                    if (start) begin
                        state <= S_COLLECT;
                        for (ii = 0; ii < N; ii = ii + 1) begin
                            accum[ii] <= 0;
                        end
                    end
                end
                S_COLLECT: begin
                    if (q_valid_in && k_valid_in && q_ready_reg && k_ready_reg) begin
                        q_latched <= q_data_in;
                        k_latched <= k_data_in;
                        // Update accumulators
                        for (ii = 0; ii < N; ii = ii + 1) begin
                            accum[ii] <= new_accum[ii];
                        end
                        if (in_last) begin
                            q_ready_reg <= 1'b0;
                            k_ready_reg <= 1'b0;
                            state <= S_PROCESS;
                        end else begin
                            dim_count <= dim_count + 4;
                        end
                    end
                end
                S_PROCESS: begin
                    case (pipeline_stage)
                        0: begin
                            // Compute exp and store scores
                            for (ii = 0; ii < N; ii = ii + 1) begin
                                score_array[ii] <= exp_result[ii];
                            end
                            // Latch level1
                            level1_reg <= level1_sum;
                            pipeline_stage <= 1;
                        end
                        1: begin
                            // Latch level2
                            level2_reg <= level2_sum;
                            pipeline_stage <= 2;
                        end
                        2: begin
                            // Compute final sum
                            score_sum_reg <= level4_sum;
                            pipeline_stage <= 0;
                            state <= S_OUTPUT_WAIT;
                        end
                    endcase
                end
                S_OUTPUT_WAIT: begin
                    score_valid_reg <= 1'b1;
                    state <= S_OUTPUT;
                end
                S_OUTPUT: begin
                    if (score_valid_reg && score_ready_out) begin
                        score_valid_reg <= 1'b0;
                        q_ready_reg <= 1'b1;
                        k_ready_reg <= 1'b1;
                        state <= S_IDLE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end



endmodule