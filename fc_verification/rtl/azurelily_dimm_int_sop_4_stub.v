// Behavioral stub of int_sop_4 for iverilog simulation.
// Synthesis (VTR) uses the real DSP hard block from the architecture XML.
module int_sop_4 (
    input wire clk,
    input wire reset,
    input wire [11:0] mode_sigs,
    input wire signed [8:0] ax, input wire signed [8:0] ay,
    input wire signed [8:0] bx, input wire signed [8:0] by,
    input wire signed [8:0] cx, input wire signed [8:0] cy,
    input wire signed [8:0] dx, input wire signed [8:0] dy,
    input wire signed [63:0] chainin,
    output wire signed [63:0] result,
    output wire signed [63:0] chainout
);
    wire signed [17:0] p_a = ax * ay;
    wire signed [17:0] p_b = bx * by;
    wire signed [17:0] p_c = cx * cy;
    wire signed [17:0] p_d = dx * dy;
    wire signed [63:0] total = $signed(p_a) + $signed(p_b) + $signed(p_c) + $signed(p_d) + $signed(chainin);
    assign result = total;
    assign chainout = total;
endmodule
