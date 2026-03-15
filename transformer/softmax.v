module softmax #(
    parameter N = 64,
    parameter d = 256,

    parameter DATA_WIDTH = 8
) (

    input clk,
    input rst,

    input [(N*DATA_WIDTH)-1:0] score_data_in, // score matrix input of size (1xN)
    input [DATA_WIDTH-1:0] score_sum_in, // 1/sum of all scores for normalization
    input score_valid_in,
    output score_ready_in,

    input start,  // indicated start of computation

    // Softmax output of size (1xN)
    output [(N*DATA_WIDTH)-1:0] softmax_data_out,  
    output softmax_valid_out,
    input softmax_ready_out
);

    // FSM states
    localparam S_IDLE = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_OUTPUT = 2'd2;

    reg [1:0] state;
    reg score_ready_reg;
    reg softmax_valid_reg;

    assign score_ready_in = score_ready_reg;
    assign softmax_valid_out = softmax_valid_reg;

    // Latch inputs
    reg [N*DATA_WIDTH-1:0] score_data_reg;
    reg [DATA_WIDTH-1:0] score_sum_reg;

    // Use mac_int_9x9 primitives for multiplication
    wire [N*18-1:0] mul_result;
    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : softmax_mul
            mac_int_9x9 mul_inst (
                .reset(rst),
                .a({1'b0, score_data_reg[gi*DATA_WIDTH +: DATA_WIDTH]}),  // pad to 9 bits
                .b({1'b0, score_sum_reg}),  // pad to 9 bits
                .out(mul_result[gi*18 +: 18]),
                .clk(clk)
            );
        end
    endgenerate

    // Pipeline the multiplier results
    reg [N*18-1:0] mul_result_reg;
    always @(posedge clk) begin
        if (rst) begin
            mul_result_reg <= 0;
        end else begin
            mul_result_reg <= mul_result;
        end
    end

    // Extract the normalized scores 
    // taking [15:8] for 8-bit output
    reg [N*DATA_WIDTH-1:0] softmax_data_reg;
    always @(*) begin
        for (integer i = 0; i < N; i = i + 1) begin
            softmax_data_reg[i*DATA_WIDTH +: DATA_WIDTH] = mul_result_reg[i*18 + 15 -: 8];
        end
    end
    assign softmax_data_out = softmax_data_reg;

    // FSM
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            score_ready_reg <= 1'b1;
            softmax_valid_reg <= 1'b0;
            score_data_reg <= 0;
            score_sum_reg <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    softmax_valid_reg <= 1'b0;
                    score_ready_reg <= 1'b1;
                    if (score_valid_in && start) begin
                        score_data_reg <= score_data_in;
                        score_sum_reg <= score_sum_in;
                        score_ready_reg <= 1'b0;
                        state <= S_COMPUTE;
                    end
                end
                S_COMPUTE: begin
                    // One cycle for multipliers
                    state <= S_OUTPUT;
                end
                S_OUTPUT: begin
                    softmax_valid_reg <= 1'b1;
                    if (softmax_valid_reg && softmax_ready_out) begin
                        softmax_valid_reg <= 1'b0;
                        score_ready_reg <= 1'b1;
                        state <= S_IDLE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule