// sim_models.v -- behavioral models of arch hard blocks for iverilog only.
// NEVER include this file in a VTR circuit: Parmys binds bare instantiations
// of these module names to the arch XML <model> entries instead.

// mac_int_9x9: 9x9 unsigned multiply, registered output (1-cycle latency).
// Port contract: benchmarks/arch/azure_lily_auto.xml line 487 <model>;
// usage precedent transformer/softmax.v:45-51.
module mac_int_9x9 (
    input  wire        reset,
    input  wire [8:0]  a,
    input  wire [8:0]  b,
    output reg  [17:0] out,
    input  wire        clk
);
    always @(posedge clk) begin
        if (reset) out <= 18'd0;
        else       out <= a * b;
    end
endmodule
