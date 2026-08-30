module xbar_module #(
    parameter DATA_WIDTH = 8,
    parameter NUM_INPUTS = 4,
    parameter NUM_OUTPUTS = 4
)(
    input  wire [NUM_INPUTS*DATA_WIDTH-1:0]  in_data,
    input  wire [$clog2(NUM_INPUTS)-1:0]     in_sel,   // Select input for each output
    input  wire [$clog2(NUM_OUTPUTS)-1:0]    out_sel,   // Select output for each input
    output reg  [NUM_OUTPUTS*DATA_WIDTH-1:0] out_data
);

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
            if ((in_sel[j] == i) && (out_sel[i] == j)) begin
                out_data_arr[j] = in_data_arr[i];
            end
        end
    end

    // Pack output data
    for (j = 0; j < NUM_OUTPUTS; j = j + 1) begin
        out_data[j*DATA_WIDTH +: DATA_WIDTH] = out_data_arr[j];
    end
end

endmodule