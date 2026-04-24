// Extracted fc_top from fc_64_128_1024x128_acam_dw40.v, renamed to fc_top_o
// Supporting modules (conv_layer_single_dpe, conv_controller, sram,
// global_controller, controller_scalable, xbar_ip_module) come from
// nldpe_dimm_top_d64_c128.v.

module fc_top_o #(
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
    localparam H = 1;
    localparam DEPTH = 103;
    localparam ADDR_WIDTH = 7;

    // Internal signals for global controller
    wire valid_g_in, valid_g_out, ready_g_in, ready_g_out;
    wire [DATA_WIDTH-1:0] data_out_layer;
    wire [DATA_WIDTH-1:0] global_sram_data_in;
    reg [7:0] read_address, write_address;

    // V=1, H=1: conv_layer_single_dpe (ACAM handles activation)
    // Packed: 5 int8/word, 13 words for K=64
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(13),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(103),
        .DATA_WIDTH(40)
    ) fc_layer_inst (
        .clk(clk),
        .rst(rst),
        .valid(valid_g_out),
        .ready_n(ready_g_out),
        .data_in(data_in),
        .data_out(data_out_layer),
        .ready(ready_g_in),
        .valid_n(valid_g_in)
    );

    // Global controller
    global_controller #(
        .N_Layers(1)
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

    // Output SRAM buffer (packed: 5 cols/word)
    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(28)
    ) global_sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(valid_g_in),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_out_layer),
        .sram_data_out(global_sram_data_in)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            read_address <= 0;
            write_address <= 16;
        end else begin
            if (ready_g_out)
                read_address <= read_address + 1;
            if (valid_g_out)
                write_address <= write_address + 1;
        end
    end

    assign data_out = global_sram_data_in;
    assign ready = ready_g_in;
    assign valid_n = valid_g_in;

endmodule
