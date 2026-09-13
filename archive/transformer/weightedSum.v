module weightedSum #(
    parameter N = 64,
    parameter d = 256,

    parameter DATA_WIDTH = 8
) (
    input clk,
    input rst,

    input [(N*DATA_WIDTH)-1:0] softmax_data_in,
    input softmax_valid_in,
    output softmax_ready_in,

    // V inputs, 4 rows each of size (1xd)
    input [(d*DATA_WIDTH)-1:0] v0_data_in,
    input [(d*DATA_WIDTH)-1:0] v1_data_in,
    input [(d*DATA_WIDTH)-1:0] v2_data_in,
    input [(d*DATA_WIDTH)-1:0] v3_data_in,
    input v_valid_in,
    output v_ready_in,

    // Start and last signals
    input start,  // indicated start of computation for a (Nxd) batch
    input in_last, // indicate last row of V of size (Nxd)

    // Output weighted sum of size (1xd)
    output [(d*DATA_WIDTH)-1:0] weighted_sum_out,
    output valid_out,
    input ready_out
);

    localparam SUM_WIDTH = 16;  // DSP result is 16-bit

    // FSM states
    localparam S_IDLE = 2'd0;
    localparam S_ACCUM = 2'd1;
    localparam S_OUTPUT = 2'd2;

    reg [1:0] state;
    reg softmax_ready_reg;
    reg v_ready_reg;
    reg valid_out_reg;

    assign softmax_ready_in = softmax_ready_reg;
    assign v_ready_in = v_ready_reg;
    assign valid_out = valid_out_reg;

    // Latch softmax
    reg [N*DATA_WIDTH-1:0] softmax_data_reg;

    // Accumulators
    reg [SUM_WIDTH-1:0] accum [0:d-1];

    // Count for index of N
    reg [$clog2(N)-1:0] i_count;

    // DSP for MAC: accum[j] + sum_{k=0 to 3} softmax[i*4 + k] * v[k][j]
    wire [SUM_WIDTH-1:0] new_accum [0:d-1];
    genvar gi;
    generate
        for (gi = 0; gi < d; gi = gi + 1) begin : accum_mac
            int_sop_4 mac_inst (
                .a0(softmax_data_reg[(i_count*4 + 0)*DATA_WIDTH +: DATA_WIDTH]),
                .a1(softmax_data_reg[(i_count*4 + 1)*DATA_WIDTH +: DATA_WIDTH]),
                .a2(softmax_data_reg[(i_count*4 + 2)*DATA_WIDTH +: DATA_WIDTH]),
                .a3(softmax_data_reg[(i_count*4 + 3)*DATA_WIDTH +: DATA_WIDTH]),
                .b0(v0_data_in[gi*DATA_WIDTH +: DATA_WIDTH]),
                .b1(v1_data_in[gi*DATA_WIDTH +: DATA_WIDTH]),
                .b2(v2_data_in[gi*DATA_WIDTH +: DATA_WIDTH]),
                .b3(v3_data_in[gi*DATA_WIDTH +: DATA_WIDTH]),
                .chainin(accum[gi]),
                .chainout(),
                .result(new_accum[gi])
            );
        end
    endgenerate

    // Pack output (truncate accum to DATA_WIDTH)
    reg [d*DATA_WIDTH-1:0] weighted_sum_reg;
    always @(*) begin
        for (integer j = 0; j < d; j = j + 1) begin
            weighted_sum_reg[j*DATA_WIDTH +: DATA_WIDTH] = accum[j][DATA_WIDTH-1:0];  // lower 8 bits
        end
    end
    assign weighted_sum_out = weighted_sum_reg;

    // FSM
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            softmax_ready_reg <= 1'b1;
            v_ready_reg <= 1'b0;
            valid_out_reg <= 1'b0;
            i_count <= 0;
            for (integer j = 0; j < d; j = j + 1) begin
                accum[j] <= 0;
            end
        end else begin
            case (state)
                S_IDLE: begin
                    valid_out_reg <= 1'b0;
                    if (softmax_valid_in && start) begin
                        softmax_data_reg <= softmax_data_in;
                        softmax_ready_reg <= 1'b0;
                        v_ready_reg <= 1'b1;
                        i_count <= 0;
                        for (integer j = 0; j < d; j = j + 1) begin
                            accum[j] <= 0;
                        end
                        state <= S_ACCUM;
                    end
                end
                S_ACCUM: begin
                    if (v_valid_in && v_ready_reg) begin
                        // Update accumulators
                        for (integer j = 0; j < d; j = j + 1) begin
                            accum[j] <= new_accum[j];
                        end
                        if (in_last) begin
                            v_ready_reg <= 1'b0;
                            state <= S_OUTPUT;
                        end else begin
                            i_count <= i_count + 4;
                        end
                    end
                end
                S_OUTPUT: begin
                    valid_out_reg <= 1'b1;
                    if (valid_out_reg && ready_out) begin
                        valid_out_reg <= 1'b0;
                        softmax_ready_reg <= 1'b1;
                        state <= S_IDLE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule