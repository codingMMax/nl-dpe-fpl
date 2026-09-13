// NL-DPE VGG11 RTL - 256x256 crossbar mapping, 16-bit data width
// Total DPEs: 146 (1+3+5+9+18+36+36+36+2)
// conv1 (single DPE): ACAM handles activation internally
// conv2-conv8 (multi-DPE): ACAM in unity/ADC mode, fabric ReLU activation
// conv9 (FC, V2_H1): no activation (output layer)

module vgg11 (
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [15:0] data_in,
    output wire [15:0] data_out,
    output wire ready,
    output wire valid_n
);

    // Internal data signals
    wire [15:0] data_out_conv1, data_out_pool1;
    wire [15:0] data_out_conv2, data_out_act2, data_out_pool2;
    wire [15:0] data_out_conv3, data_out_act3;
    wire [15:0] data_out_conv4, data_out_act4, data_out_pool3;
    wire [15:0] data_out_conv5, data_out_act5;
    wire [15:0] data_out_conv6, data_out_act6, data_out_pool4;
    wire [15:0] data_out_conv7, data_out_act7;
    wire [15:0] data_out_conv8, data_out_act8, data_out_pool5;
    wire [15:0] data_out_avg_pool1;
    wire [15:0] data_out_conv9;
    wire [15:0] global_sram_data_in;

    // Internal ready/valid signals
    wire ready_conv1, valid_conv1;
    wire ready_pool1, valid_pool1;
    wire ready_conv2, valid_conv2;
    wire ready_act2, valid_act2;
    wire ready_pool2, valid_pool2;
    wire ready_conv3, valid_conv3;
    wire ready_act3, valid_act3;
    wire ready_conv4, valid_conv4;
    wire ready_act4, valid_act4;
    wire ready_pool3, valid_pool3;
    wire ready_conv5, valid_conv5;
    wire ready_act5, valid_act5;
    wire ready_conv6, valid_conv6;
    wire ready_act6, valid_act6;
    wire ready_pool4, valid_pool4;
    wire ready_conv7, valid_conv7;
    wire ready_act7, valid_act7;
    wire ready_conv8, valid_conv8;
    wire ready_act8, valid_act8;
    wire ready_pool5, valid_pool5;
    wire ready_avg_pool1, valid_avg_pool1;
    wire valid_g_in, valid_g_out, ready_g_in, ready_g_out;

    reg [7:0] read_address, write_address;

    // ============================================================
    // Layer 1: conv1 (1->64, K=9, single DPE) - ACAM handles ReLU
    // ============================================================
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(10), .N_KERNELS(1),
        .KERNEL_WIDTH(3), .KERNEL_HEIGHT(3),
        .W(33), .H(33), .S(1), .DEPTH(1024), .DATA_WIDTH(16)
    ) conv1 (
        .clk(clk), .rst(rst),
        .valid(valid_g_out), .ready_n(ready_conv1),
        .data_in(data_in), .data_out(data_out_conv1),
        .ready(ready_g_in), .valid_n(valid_conv1)
    );

    // NL-DPE: activation_layer removed (single DPE, ACAM handles it)

    // Pool 1
    pool_layer1 #(
        .N_CHANNELS(1), .ADDR_WIDTH(10), .DATA_WIDTH(16), .DEPTH(1024)
    ) pool1 (
        .clk(clk), .rst(rst),
        .valid(valid_conv1), .ready_n(ready_pool1),
        .layer_done(1'b0),
        .data_in(data_out_conv1), .data_out(data_out_pool1),
        .ready(ready_conv1), .valid_n(valid_pool1)
    );

    // ============================================================
    // Layer 2: conv2 (64->128, K=576, V3_H1) + fabric ReLU
    // ============================================================
    conv_layer_stacked_dpes_V3_H1 #(
        .N_CHANNELS(1), .ADDR_WIDTH(9), .N_KERNELS(1),
        .KERNEL_WIDTH(3), .KERNEL_HEIGHT(3),
        .W(16), .H(16), .S(1),
        .N_DPE_V(3), .N_DPE_H(1), .N_BRAM_R(1), .N_BRAM_W(1),
        .DATA_WIDTH(16), .DEPTH(512)
    ) conv2 (
        .clk(clk), .rst(rst),
        .valid(valid_pool1), .ready_n(ready_conv2),
        .data_in(data_out_pool1), .data_out(data_out_conv2),
        .ready(ready_pool1), .valid_n(valid_conv2)
    );

    activation_layer_relu #(
        .N_CHANNELS(1), .ADDR_WIDTH(9), .DATA_WIDTH(16), .DEPTH(512)
    ) act2 (
        .clk(clk), .rst(rst),
        .valid(valid_conv2), .ready_n(ready_act2),
        .data_in(data_out_conv2), .data_out(data_out_act2),
        .ready(ready_conv2), .valid_n(valid_act2)
    );

    // Pool 2
    pool_layer2 #(
        .N_CHANNELS(1), .ADDR_WIDTH(8), .DATA_WIDTH(16), .DEPTH(256)
    ) pool2 (
        .clk(clk), .rst(rst),
        .valid(valid_act2), .ready_n(ready_pool2),
        .layer_done(1'b0),
        .data_in(data_out_act2), .data_out(data_out_pool2),
        .ready(ready_act2), .valid_n(valid_pool2)
    );

    // ============================================================
    // Layer 3: conv3 (128->256, K=1152, V5_H1) + fabric ReLU
    // ============================================================
    conv_layer_stacked_dpes_V5_H1 #(
        .N_CHANNELS(1), .ADDR_WIDTH(9), .N_KERNELS(1),
        .KERNEL_WIDTH(3), .KERNEL_HEIGHT(3),
        .W(8), .H(8), .S(1),
        .N_DPE_V(5), .N_DPE_H(1), .N_BRAM_R(1), .N_BRAM_W(1),
        .DATA_WIDTH(16), .DEPTH(512)
    ) conv3 (
        .clk(clk), .rst(rst),
        .valid(valid_pool2), .ready_n(ready_conv3),
        .data_in(data_out_pool2), .data_out(data_out_conv3),
        .ready(ready_pool2), .valid_n(valid_conv3)
    );

    activation_layer_relu #(
        .N_CHANNELS(1), .ADDR_WIDTH(6), .DATA_WIDTH(16), .DEPTH(64)
    ) act3 (
        .clk(clk), .rst(rst),
        .valid(valid_conv3), .ready_n(ready_act3),
        .data_in(data_out_conv3), .data_out(data_out_act3),
        .ready(ready_conv3), .valid_n(valid_act3)
    );

    // ============================================================
    // Layer 4: conv4 (256->256, K=2304, V9_H1) + fabric ReLU
    // ============================================================
    conv_layer_stacked_dpes_V9_H1 #(
        .N_CHANNELS(1), .ADDR_WIDTH(9), .N_KERNELS(1),
        .KERNEL_WIDTH(3), .KERNEL_HEIGHT(3),
        .W(9), .H(9), .S(1),
        .N_DPE_V(9), .N_DPE_H(1), .N_BRAM_R(1), .N_BRAM_W(1),
        .DATA_WIDTH(16), .DEPTH(512)
    ) conv4 (
        .clk(clk), .rst(rst),
        .valid(valid_act3), .ready_n(ready_conv4),
        .data_in(data_out_act3), .data_out(data_out_conv4),
        .ready(ready_act3), .valid_n(valid_conv4)
    );

    activation_layer_relu #(
        .N_CHANNELS(1), .ADDR_WIDTH(6), .DATA_WIDTH(16), .DEPTH(64)
    ) act4 (
        .clk(clk), .rst(rst),
        .valid(valid_conv4), .ready_n(ready_act4),
        .data_in(data_out_conv4), .data_out(data_out_act4),
        .ready(ready_conv4), .valid_n(valid_act4)
    );

    // Pool 3
    pool_layer3 #(
        .N_CHANNELS(1), .ADDR_WIDTH(6), .DATA_WIDTH(16), .DEPTH(64)
    ) pool3 (
        .clk(clk), .rst(rst),
        .valid(valid_act4), .ready_n(ready_pool3),
        .layer_done(1'b0),
        .data_in(data_out_act4), .data_out(data_out_pool3),
        .ready(ready_act4), .valid_n(valid_pool3)
    );

    // ============================================================
    // Layer 5: conv5 (256->512, K=2304, V9_H2) + fabric ReLU
    // ============================================================
    conv_layer_stacked_dpes_V9_H2 #(
        .N_CHANNELS(1), .ADDR_WIDTH(9), .N_KERNELS(1),
        .KERNEL_WIDTH(3), .KERNEL_HEIGHT(3),
        .W(5), .H(5), .S(1),
        .N_DPE_V(9), .N_DPE_H(2), .N_BRAM_R(1), .N_BRAM_W(1),
        .DATA_WIDTH(16), .DEPTH(512)
    ) conv5 (
        .clk(clk), .rst(rst),
        .valid(valid_pool3), .ready_n(ready_conv5),
        .data_in(data_out_pool3), .data_out(data_out_conv5),
        .ready(ready_pool3), .valid_n(valid_conv5)
    );

    activation_layer_relu #(
        .N_CHANNELS(1), .ADDR_WIDTH(4), .DATA_WIDTH(16), .DEPTH(16)
    ) act5 (
        .clk(clk), .rst(rst),
        .valid(valid_conv5), .ready_n(ready_act5),
        .data_in(data_out_conv5), .data_out(data_out_act5),
        .ready(ready_conv5), .valid_n(valid_act5)
    );

    // ============================================================
    // Layer 6: conv6 (512->512, K=4608, V18_H2) + fabric ReLU
    // ============================================================
    conv_layer_stacked_dpes_V18_H2 #(
        .N_CHANNELS(1), .ADDR_WIDTH(9), .N_KERNELS(1),
        .KERNEL_WIDTH(3), .KERNEL_HEIGHT(3),
        .W(5), .H(5), .S(1),
        .N_DPE_V(18), .N_DPE_H(2), .N_BRAM_R(1), .N_BRAM_W(1),
        .DATA_WIDTH(16), .DEPTH(512)
    ) conv6 (
        .clk(clk), .rst(rst),
        .valid(valid_act5), .ready_n(ready_conv6),
        .data_in(data_out_act5), .data_out(data_out_conv6),
        .ready(ready_act5), .valid_n(valid_conv6)
    );

    activation_layer_relu #(
        .N_CHANNELS(1), .ADDR_WIDTH(4), .DATA_WIDTH(16), .DEPTH(16)
    ) act6 (
        .clk(clk), .rst(rst),
        .valid(valid_conv6), .ready_n(ready_act6),
        .data_in(data_out_conv6), .data_out(data_out_act6),
        .ready(ready_conv6), .valid_n(valid_act6)
    );

    // Pool 4
    pool_layer4 #(
        .N_CHANNELS(1), .ADDR_WIDTH(4), .DATA_WIDTH(16), .DEPTH(16)
    ) pool4 (
        .clk(clk), .rst(rst),
        .valid(valid_act6), .ready_n(ready_pool4),
        .layer_done(1'b0),
        .data_in(data_out_act6), .data_out(data_out_pool4),
        .ready(ready_act6), .valid_n(valid_pool4)
    );

    // ============================================================
    // Layer 7: conv7 (512->512, K=4608, V18_H2) + fabric ReLU
    // ============================================================
    conv_layer_stacked_dpes_V18_H2 #(
        .N_CHANNELS(1), .ADDR_WIDTH(9), .N_KERNELS(1),
        .KERNEL_WIDTH(3), .KERNEL_HEIGHT(3),
        .W(5), .H(5), .S(1),
        .N_DPE_V(18), .N_DPE_H(2), .N_BRAM_R(1), .N_BRAM_W(1),
        .DATA_WIDTH(16), .DEPTH(512)
    ) conv7 (
        .clk(clk), .rst(rst),
        .valid(valid_pool4), .ready_n(ready_conv7),
        .data_in(data_out_pool4), .data_out(data_out_conv7),
        .ready(ready_pool4), .valid_n(valid_conv7)
    );

    activation_layer_relu #(
        .N_CHANNELS(1), .ADDR_WIDTH(2), .DATA_WIDTH(16), .DEPTH(4)
    ) act7 (
        .clk(clk), .rst(rst),
        .valid(valid_conv7), .ready_n(ready_act7),
        .data_in(data_out_conv7), .data_out(data_out_act7),
        .ready(ready_conv7), .valid_n(valid_act7)
    );

    // ============================================================
    // Layer 8: conv8 (512->512, K=4608, V18_H2) + fabric ReLU
    // ============================================================
    conv_layer_stacked_dpes_V18_H2 #(
        .N_CHANNELS(1), .ADDR_WIDTH(9), .N_KERNELS(1),
        .KERNEL_WIDTH(3), .KERNEL_HEIGHT(3),
        .W(3), .H(3), .S(1),
        .N_DPE_V(18), .N_DPE_H(2), .N_BRAM_R(1), .N_BRAM_W(1),
        .DATA_WIDTH(16), .DEPTH(512)
    ) conv8 (
        .clk(clk), .rst(rst),
        .valid(valid_act7), .ready_n(ready_conv8),
        .data_in(data_out_act7), .data_out(data_out_conv8),
        .ready(ready_act7), .valid_n(valid_conv8)
    );

    activation_layer_relu #(
        .N_CHANNELS(1), .ADDR_WIDTH(2), .DATA_WIDTH(16), .DEPTH(4)
    ) act8 (
        .clk(clk), .rst(rst),
        .valid(valid_conv8), .ready_n(ready_act8),
        .data_in(data_out_conv8), .data_out(data_out_act8),
        .ready(ready_conv8), .valid_n(valid_act8)
    );

    // Pool 5
    pool_layer5 #(
        .N_CHANNELS(1), .ADDR_WIDTH(2), .DATA_WIDTH(16), .DEPTH(4)
    ) pool5 (
        .clk(clk), .rst(rst),
        .valid(valid_act8), .ready_n(ready_pool5),
        .layer_done(1'b0),
        .data_in(data_out_act8), .data_out(data_out_pool5),
        .ready(ready_act8), .valid_n(valid_pool5)
    );

    // Avg Pool
    avg_pool_layer1 #(
        .N_CHANNELS(1), .ADDR_WIDTH(2), .DATA_WIDTH(16), .DEPTH(2)
    ) avg_pool1 (
        .clk(clk), .rst(rst),
        .valid(valid_pool5), .ready_n(ready_avg_pool1),
        .layer_done(1'b0),
        .data_in(data_out_pool5), .data_out(data_out_avg_pool1),
        .ready(ready_pool5), .valid_n(valid_avg_pool1)
    );

    // ============================================================
    // Layer 9: conv9/FC (512->10, K=512, V2_H1) - no activation
    // ============================================================
    conv_layer_stacked_dpes_V2_H1 #(
        .N_CHANNELS(1), .ADDR_WIDTH(8), .N_KERNELS(1),
        .KERNEL_WIDTH(1), .KERNEL_HEIGHT(1),
        .W(1), .H(1), .S(1),
        .N_DPE_V(2), .N_DPE_H(1), .N_BRAM_R(1), .N_BRAM_W(1),
        .DATA_WIDTH(16), .DEPTH(256)
    ) conv9 (
        .clk(clk), .rst(rst),
        .valid(valid_avg_pool1),
        .ready_n(valid_g_out),
        .data_in(data_out_avg_pool1),
        .data_out(data_out_conv9),
        .ready(ready_avg_pool1),
        .valid_n(valid_g_in)
    );

    // Global controller
    global_controller #(.N_Layers(5)) g_ctrl_inst (
        .clk(clk), .rst(rst),
        .ready_L1(ready_g_in),
        .valid_Ln(valid_g_in),
        .valid(valid),
        .ready(ready),
        .valid_L1(valid_g_out),
        .ready_Ln(ready_g_out)
    );

    // Global SRAM
    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(16),
        .DEPTH(16)
    ) global_sram_inst (
        .clk(clk), .rst(rst),
        .w_en(valid_g_in),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_out_conv9),
        .sram_data_out(global_sram_data_in)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            read_address <= 0;
            write_address <= 16;
        end else begin
            if(ready_g_out)
                read_address <= read_address + 1;
            if(valid_g_out)
                write_address <= write_address + 1;
        end
    end

    assign data_out = global_sram_data_in;

endmodule


// ============================================================
// NEW NL-DPE STACKING MODULES
// ============================================================

// conv_layer_stacked_dpes_V3_H1: 3 DPEs vertical, 1 horizontal
module conv_layer_stacked_dpes_V3_H1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 3,
    parameter N_DPE_H = 1,
    parameter N_BRAM_R = 1,
    parameter N_BRAM_W = 1,
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [(N_CHANNELS*DATA_WIDTH)-1:0] data_in,
    output reg [(N_KERNELS*DATA_WIDTH)-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    wire MSB_SA_Ready;
    wire dpe_done;
    wire [N_DPE_V-1:0] reg_full_sig;
    wire [N_DPE_H-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire load_input_reg;

    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out1, data_out2, data_out3;
    wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_temp1;
    wire [N_BRAM_W-1:0] w_en_dec_signal;
    wire [N_DPE_V-1:0] dpe_sel_signal;
    wire [N_DPE_H-1:0] dpe_sel_h_signal;
    wire [DATA_WIDTH-1:0] sram_data;
    wire dpe_done1, dpe_done2, dpe_done3;
    wire accum_ready, accum_done;
    wire accum_done1;
    wire shift_add_done1, shift_add_done2, shift_add_done3;
    wire shift_add_bypass_ctrl1, shift_add_bypass_ctrl2, shift_add_bypass_ctrl3;
    wire MSB_SA_Ready1, MSB_SA_Ready2, MSB_SA_Ready3;

    sram #(.N_CHANNELS(1), .DEPTH(512)) sram_inst_1 (
        .clk(clk), .rst(rst),
        .w_en(w_en_dec_signal),
        .r_addr(read_address), .w_addr(write_address),
        .sram_data_in(data_in), .sram_data_out(sram_data)
    );

    xbar_ip_module #(.DATA_WIDTH(16), .NUM_INPUTS(1), .NUM_OUTPUTS(3)) u_xbar (
        .in_data(sram_data), .in_sel(N_BRAM_R),
        .out_sel(dpe_sel_signal), .out_data(dpe_data)
    );

    dpe dpe_R1_C1 (
        .clk(clk), .reset(rst), .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control), .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en), .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg), .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready1), .data_out(data_out1),
        .dpe_done(dpe_done1), .reg_full(reg_full_sig[0]),
        .shift_add_done(shift_add_done1), .shift_add_bypass_ctrl(shift_add_bypass_ctrl1)
    );

    dpe dpe_R2_C1 (
        .clk(clk), .reset(rst), .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control), .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en), .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg), .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready2), .data_out(data_out2),
        .dpe_done(dpe_done2), .reg_full(reg_full_sig[1]),
        .shift_add_done(shift_add_done2), .shift_add_bypass_ctrl(shift_add_bypass_ctrl2)
    );

    dpe dpe_R3_C1 (
        .clk(clk), .reset(rst), .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control), .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en), .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg), .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready3), .data_out(data_out3),
        .dpe_done(dpe_done3), .reg_full(reg_full_sig[2]),
        .shift_add_done(shift_add_done3), .shift_add_bypass_ctrl(shift_add_bypass_ctrl3)
    );

    assign shift_add_done = shift_add_done1 & shift_add_done2 & shift_add_done3;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl1 & shift_add_bypass_ctrl2 & shift_add_bypass_ctrl3;
    assign dpe_done = dpe_done1 & dpe_done2 & dpe_done3;
    assign MSB_SA_Ready = MSB_SA_Ready1 & MSB_SA_Ready2 & MSB_SA_Ready3;

    // Adder tree: 1 pair adder + registered final sum
    adder_dpe_N_CHANNELS_1 #(.N_CHANNELS(1)) adder_inst_R1_R2 (
        .clk(clk), .reset(rst), .en(accum_ready), .add_done(accum_done1),
        .input1(data_out1), .input2(data_out2), .output_data(data_out_temp1)
    );

    assign accum_done = accum_done1;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
        end else begin
            data_out <= data_out_temp1 + data_out3;
        end
    end

    controller_scalable #(
        .N_CHANNELS(N_CHANNELS), .N_BRAM_R(N_BRAM_R), .N_BRAM_W(N_BRAM_W),
        .N_DPE_V(N_DPE_V), .N_DPE_H(N_DPE_H), .ADDR_WIDTH(ADDR_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH), .KERNEL_HEIGHT(KERNEL_HEIGHT),
        .W(W), .H(H), .S(S)
    ) controller_scalable_inst (
        .clk(clk), .rst(rst), .MSB_SA_Ready(MSB_SA_Ready),
        .valid(valid), .ready_n(ready_n), .dpe_done(dpe_done),
        .reg_full(reg_full_sig), .reg_empty(reg_empty),
        .shift_add_done(shift_add_done), .shift_add_bypass_ctrl(shift_add_bypass_ctrl),
        .dpe_accum_done(accum_done),
        .read_address(read_address), .write_address(write_address),
        .w_buf_en(w_buf_en), .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control), .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg), .w_en_dec(w_en_dec_signal),
        .load_input_reg(load_input_reg), .ready(ready), .valid_n(valid_n),
        .dpe_accum_ready(accum_ready), .dpe_sel(dpe_sel_signal), .dpe_sel_h(dpe_sel_h_signal)
    );

endmodule


// conv_layer_stacked_dpes_V5_H1: 5 DPEs vertical, 1 horizontal
module conv_layer_stacked_dpes_V5_H1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 5,
    parameter N_DPE_H = 1,
    parameter N_BRAM_R = 1,
    parameter N_BRAM_W = 1,
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [(N_CHANNELS*DATA_WIDTH)-1:0] data_in,
    output reg [(N_KERNELS*DATA_WIDTH)-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_arr [0:4];
    wire shift_add_done_arr [0:4];
    wire shift_add_bypass_ctrl_arr [0:4];
    wire MSB_SA_Ready_arr [0:4];
    wire dpe_done_arr [0:4];
    wire reg_full_arr [0:4];
    wire [N_DPE_H-1:0] reg_empty_sig;
    wire [ADDR_WIDTH-1:0] read_address, write_address;
    wire w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire load_input_reg;
    wire [(N_BRAM_W-1):0] w_en_dec_signal;
    wire [N_DPE_V-1:0] dpe_sel_signal;
    wire [N_DPE_H-1:0] dpe_sel_h_signal;
    wire [DATA_WIDTH-1:0] sram_data;
    wire accum_ready;

    sram #(.N_CHANNELS(1), .DEPTH(512)) sram_inst_1 (
        .clk(clk), .rst(rst),
        .w_en(w_en_dec_signal),
        .r_addr(read_address), .w_addr(write_address),
        .sram_data_in(data_in), .sram_data_out(sram_data)
    );

    xbar_ip_module #(.DATA_WIDTH(16), .NUM_INPUTS(1), .NUM_OUTPUTS(5)) u_xbar (
        .in_data(sram_data), .in_sel(N_BRAM_R),
        .out_sel(dpe_sel_signal), .out_data(dpe_data)
    );

    genvar i;
    generate
        for (i = 0; i < 5; i = i + 1) begin: DPE_ARRAY
            dpe dpe_inst (
                .clk(clk), .reset(rst), .data_in(dpe_data),
                .nl_dpe_control(nl_dpe_control), .shift_add_control(shift_add_control),
                .w_buf_en(w_buf_en), .shift_add_bypass(shift_add_bypass),
                .load_output_reg(load_output_reg), .load_input_reg(load_input_reg),
                .MSB_SA_Ready(MSB_SA_Ready_arr[i]), .data_out(data_out_arr[i]),
                .dpe_done(dpe_done_arr[i]), .reg_full(reg_full_arr[i]),
                .shift_add_done(shift_add_done_arr[i]),
                .shift_add_bypass_ctrl(shift_add_bypass_ctrl_arr[i])
            );
        end
    endgenerate

    wire MSB_SA_Ready = MSB_SA_Ready_arr[0] & MSB_SA_Ready_arr[1] & MSB_SA_Ready_arr[2] & MSB_SA_Ready_arr[3] & MSB_SA_Ready_arr[4];
    wire dpe_done = dpe_done_arr[0] & dpe_done_arr[1] & dpe_done_arr[2] & dpe_done_arr[3] & dpe_done_arr[4];
    wire shift_add_done = shift_add_done_arr[0] & shift_add_done_arr[1] & shift_add_done_arr[2] & shift_add_done_arr[3] & shift_add_done_arr[4];
    wire shift_add_bypass_ctrl = shift_add_bypass_ctrl_arr[0] & shift_add_bypass_ctrl_arr[1] & shift_add_bypass_ctrl_arr[2] & shift_add_bypass_ctrl_arr[3] & shift_add_bypass_ctrl_arr[4];
    wire reg_full_sig = reg_full_arr[0] & reg_full_arr[1] & reg_full_arr[2] & reg_full_arr[3] & reg_full_arr[4];

    wire [(N_KERNELS*DATA_WIDTH)-1:0] sum1, sum2;
    wire done1, done2;

    adder_dpe_N_CHANNELS_1 add1 (.clk(clk), .reset(rst), .en(accum_ready), .add_done(done1), .input1(data_out_arr[0]), .input2(data_out_arr[1]), .output_data(sum1));
    adder_dpe_N_CHANNELS_1 add2 (.clk(clk), .reset(rst), .en(accum_ready), .add_done(done2), .input1(data_out_arr[2]), .input2(data_out_arr[3]), .output_data(sum2));

    wire accum_done = done1 & done2;

    reg [(N_KERNELS*DATA_WIDTH)-1:0] temp_sum1;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
            temp_sum1 <= 0;
        end else begin
            temp_sum1 <= sum1 + sum2;
            data_out <= temp_sum1 + data_out_arr[4];
        end
    end

    controller_scalable #(
        .N_CHANNELS(N_CHANNELS), .N_BRAM_R(N_BRAM_R), .N_BRAM_W(N_BRAM_W),
        .N_DPE_V(N_DPE_V), .N_DPE_H(N_DPE_H), .ADDR_WIDTH(ADDR_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH), .KERNEL_HEIGHT(KERNEL_HEIGHT),
        .W(W), .H(H), .S(S)
    ) controller_scalable_inst (
        .clk(clk), .rst(rst), .MSB_SA_Ready(MSB_SA_Ready),
        .valid(valid), .ready_n(ready_n), .dpe_done(dpe_done),
        .reg_full(reg_full_sig), .reg_empty(reg_empty_sig),
        .shift_add_done(shift_add_done), .shift_add_bypass_ctrl(shift_add_bypass_ctrl),
        .dpe_accum_done(accum_done),
        .read_address(read_address), .write_address(write_address),
        .w_buf_en(w_buf_en), .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control), .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg), .w_en_dec(w_en_dec_signal),
        .load_input_reg(load_input_reg), .ready(ready), .valid_n(valid_n),
        .dpe_accum_ready(accum_ready), .dpe_sel(dpe_sel_signal), .dpe_sel_h(dpe_sel_h_signal)
    );

endmodule


// conv_layer_stacked_dpes_V9_H1: 9 DPEs vertical, 1 horizontal
module conv_layer_stacked_dpes_V9_H1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 9,
    parameter N_DPE_H = 1,
    parameter N_BRAM_R = 1,
    parameter N_BRAM_W = 1,
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [(N_CHANNELS*DATA_WIDTH)-1:0] data_in,
    output reg [(N_KERNELS*DATA_WIDTH)-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_arr [0:8];
    wire shift_add_done_arr [0:8];
    wire shift_add_bypass_ctrl_arr [0:8];
    wire MSB_SA_Ready_arr [0:8];
    wire dpe_done_arr [0:8];
    wire reg_full_arr [0:8];
    wire [N_DPE_H-1:0] reg_empty_sig;
    wire [ADDR_WIDTH-1:0] read_address, write_address;
    wire w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire load_input_reg;
    wire [(N_BRAM_W-1):0] w_en_dec_signal;
    wire [N_DPE_V-1:0] dpe_sel_signal;
    wire [N_DPE_H-1:0] dpe_sel_h_signal;
    wire [DATA_WIDTH-1:0] sram_data;
    wire accum_ready;

    sram #(.N_CHANNELS(1), .DEPTH(512)) sram_inst_1 (
        .clk(clk), .rst(rst),
        .w_en(w_en_dec_signal),
        .r_addr(read_address), .w_addr(write_address),
        .sram_data_in(data_in), .sram_data_out(sram_data)
    );

    xbar_ip_module #(.DATA_WIDTH(16), .NUM_INPUTS(1), .NUM_OUTPUTS(9)) u_xbar (
        .in_data(sram_data), .in_sel(N_BRAM_R),
        .out_sel(dpe_sel_signal), .out_data(dpe_data)
    );

    genvar i;
    generate
        for (i = 0; i < 9; i = i + 1) begin: DPE_ARRAY
            dpe dpe_inst (
                .clk(clk), .reset(rst), .data_in(dpe_data),
                .nl_dpe_control(nl_dpe_control), .shift_add_control(shift_add_control),
                .w_buf_en(w_buf_en), .shift_add_bypass(shift_add_bypass),
                .load_output_reg(load_output_reg), .load_input_reg(load_input_reg),
                .MSB_SA_Ready(MSB_SA_Ready_arr[i]), .data_out(data_out_arr[i]),
                .dpe_done(dpe_done_arr[i]), .reg_full(reg_full_arr[i]),
                .shift_add_done(shift_add_done_arr[i]),
                .shift_add_bypass_ctrl(shift_add_bypass_ctrl_arr[i])
            );
        end
    endgenerate

    wire MSB_SA_Ready = MSB_SA_Ready_arr[0] & MSB_SA_Ready_arr[1] & MSB_SA_Ready_arr[2] & MSB_SA_Ready_arr[3] & MSB_SA_Ready_arr[4] & MSB_SA_Ready_arr[5] & MSB_SA_Ready_arr[6] & MSB_SA_Ready_arr[7] & MSB_SA_Ready_arr[8];
    wire dpe_done = dpe_done_arr[0] & dpe_done_arr[1] & dpe_done_arr[2] & dpe_done_arr[3] & dpe_done_arr[4] & dpe_done_arr[5] & dpe_done_arr[6] & dpe_done_arr[7] & dpe_done_arr[8];
    wire shift_add_done = shift_add_done_arr[0] & shift_add_done_arr[1] & shift_add_done_arr[2] & shift_add_done_arr[3] & shift_add_done_arr[4] & shift_add_done_arr[5] & shift_add_done_arr[6] & shift_add_done_arr[7] & shift_add_done_arr[8];
    wire shift_add_bypass_ctrl = shift_add_bypass_ctrl_arr[0] & shift_add_bypass_ctrl_arr[1] & shift_add_bypass_ctrl_arr[2] & shift_add_bypass_ctrl_arr[3] & shift_add_bypass_ctrl_arr[4] & shift_add_bypass_ctrl_arr[5] & shift_add_bypass_ctrl_arr[6] & shift_add_bypass_ctrl_arr[7] & shift_add_bypass_ctrl_arr[8];
    wire reg_full_sig = reg_full_arr[0] & reg_full_arr[1] & reg_full_arr[2] & reg_full_arr[3] & reg_full_arr[4] & reg_full_arr[5] & reg_full_arr[6] & reg_full_arr[7] & reg_full_arr[8];

    // Adder tree: 4 pair adders at level 1
    wire [(N_KERNELS*DATA_WIDTH)-1:0] sum [0:3];
    wire done [0:3];

    adder_dpe_N_CHANNELS_1 add1 (.clk(clk), .reset(rst), .en(accum_ready), .add_done(done[0]), .input1(data_out_arr[0]), .input2(data_out_arr[1]), .output_data(sum[0]));
    adder_dpe_N_CHANNELS_1 add2 (.clk(clk), .reset(rst), .en(accum_ready), .add_done(done[1]), .input1(data_out_arr[2]), .input2(data_out_arr[3]), .output_data(sum[1]));
    adder_dpe_N_CHANNELS_1 add3 (.clk(clk), .reset(rst), .en(accum_ready), .add_done(done[2]), .input1(data_out_arr[4]), .input2(data_out_arr[5]), .output_data(sum[2]));
    adder_dpe_N_CHANNELS_1 add4 (.clk(clk), .reset(rst), .en(accum_ready), .add_done(done[3]), .input1(data_out_arr[6]), .input2(data_out_arr[7]), .output_data(sum[3]));

    wire accum_done = done[0] & done[1] & done[2] & done[3];

    // Pipelined reduction tree
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s0_0, s0_1;
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s1_0;
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s2_0;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
            s0_0 <= 0; s0_1 <= 0;
            s1_0 <= 0; s2_0 <= 0;
        end else begin
            // Stage 0: pair sums from level 1 adders
            s0_0 <= sum[0] + sum[1];
            s0_1 <= sum[2] + sum[3];
            // Stage 1: second level
            s1_0 <= s0_0 + s0_1;
            // Stage 2: add leftover R9
            s2_0 <= s1_0 + data_out_arr[8];
            // Output
            data_out <= s2_0;
        end
    end

    controller_scalable #(
        .N_CHANNELS(N_CHANNELS), .N_BRAM_R(N_BRAM_R), .N_BRAM_W(N_BRAM_W),
        .N_DPE_V(N_DPE_V), .N_DPE_H(N_DPE_H), .ADDR_WIDTH(ADDR_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH), .KERNEL_HEIGHT(KERNEL_HEIGHT),
        .W(W), .H(H), .S(S)
    ) controller_scalable_inst (
        .clk(clk), .rst(rst), .MSB_SA_Ready(MSB_SA_Ready),
        .valid(valid), .ready_n(ready_n), .dpe_done(dpe_done),
        .reg_full(reg_full_sig), .reg_empty(reg_empty_sig),
        .shift_add_done(shift_add_done), .shift_add_bypass_ctrl(shift_add_bypass_ctrl),
        .dpe_accum_done(accum_done),
        .read_address(read_address), .write_address(write_address),
        .w_buf_en(w_buf_en), .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control), .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg), .w_en_dec(w_en_dec_signal),
        .load_input_reg(load_input_reg), .ready(ready), .valid_n(valid_n),
        .dpe_accum_ready(accum_ready), .dpe_sel(dpe_sel_signal), .dpe_sel_h(dpe_sel_h_signal)
    );

endmodule


// conv_layer_stacked_dpes_V9_H2: 18 DPEs (9 vertical x 2 horizontal)
module conv_layer_stacked_dpes_V9_H2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 9,
    parameter N_DPE_H = 2,
    parameter N_BRAM_R = 1,
    parameter N_BRAM_W = 1,
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [(N_CHANNELS*DATA_WIDTH)-1:0] data_in,
    output reg [(N_KERNELS*DATA_WIDTH)-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_arr [0:17];
    wire shift_add_done_arr [0:17];
    wire shift_add_bypass_ctrl_arr [0:17];
    wire MSB_SA_Ready_arr [0:17];
    wire dpe_done_arr [0:17];
    wire reg_full_arr [0:17];
    wire [N_DPE_H-1:0] reg_empty_sig;
    wire [ADDR_WIDTH-1:0] read_address, write_address;
    wire w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire load_input_reg;
    wire [(N_BRAM_W-1):0] w_en_dec_signal;
    wire [N_DPE_V-1:0] dpe_sel_signal;
    wire [N_DPE_H-1:0] dpe_sel_h_signal;
    wire [DATA_WIDTH-1:0] sram_data;
    wire accum_ready;

    sram #(.N_CHANNELS(1), .DEPTH(512)) sram_inst_1 (
        .clk(clk), .rst(rst),
        .w_en(w_en_dec_signal),
        .r_addr(read_address), .w_addr(write_address),
        .sram_data_in(data_in), .sram_data_out(sram_data)
    );

    xbar_ip_module #(.DATA_WIDTH(16), .NUM_INPUTS(1), .NUM_OUTPUTS(18)) u_xbar (
        .in_data(sram_data), .in_sel(N_BRAM_R),
        .out_sel(dpe_sel_signal), .out_data(dpe_data)
    );

    genvar i;
    generate
        for (i = 0; i < 18; i = i + 1) begin: DPE_ARRAY
            reg [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data_reg;
            reg [1:0] nl_dpe_control_reg;
            reg shift_add_control_reg;
            reg w_buf_en_reg;
            reg shift_add_bypass_reg;
            reg load_output_reg_reg;
            reg load_input_reg_reg;

            always @(posedge clk) begin
                if (rst) begin
                    dpe_data_reg <= 0;
                    nl_dpe_control_reg <= 0;
                    shift_add_control_reg <= 0;
                    w_buf_en_reg <= 0;
                    shift_add_bypass_reg <= 0;
                    load_output_reg_reg <= 0;
                    load_input_reg_reg <= 0;
                end else begin
                    dpe_data_reg <= dpe_data;
                    nl_dpe_control_reg <= nl_dpe_control;
                    shift_add_control_reg <= shift_add_control;
                    w_buf_en_reg <= w_buf_en;
                    shift_add_bypass_reg <= shift_add_bypass;
                    load_output_reg_reg <= load_output_reg;
                    load_input_reg_reg <= load_input_reg;
                end
            end

            dpe dpe_inst (
                .clk(clk), .reset(rst),
                .data_in(dpe_data_reg),
                .nl_dpe_control(nl_dpe_control_reg),
                .shift_add_control(shift_add_control_reg),
                .w_buf_en(w_buf_en_reg),
                .shift_add_bypass(shift_add_bypass_reg),
                .load_output_reg(load_output_reg_reg),
                .load_input_reg(load_input_reg_reg),
                .MSB_SA_Ready(MSB_SA_Ready_arr[i]),
                .data_out(data_out_arr[i]),
                .dpe_done(dpe_done_arr[i]),
                .reg_full(reg_full_arr[i]),
                .shift_add_done(shift_add_done_arr[i]),
                .shift_add_bypass_ctrl(shift_add_bypass_ctrl_arr[i])
            );
        end
    endgenerate

    wire MSB_SA_Ready =
        MSB_SA_Ready_arr[0] & MSB_SA_Ready_arr[1] & MSB_SA_Ready_arr[2] & MSB_SA_Ready_arr[3] & MSB_SA_Ready_arr[4] &
        MSB_SA_Ready_arr[5] & MSB_SA_Ready_arr[6] & MSB_SA_Ready_arr[7] & MSB_SA_Ready_arr[8] & MSB_SA_Ready_arr[9] &
        MSB_SA_Ready_arr[10] & MSB_SA_Ready_arr[11] & MSB_SA_Ready_arr[12] & MSB_SA_Ready_arr[13] & MSB_SA_Ready_arr[14] &
        MSB_SA_Ready_arr[15] & MSB_SA_Ready_arr[16] & MSB_SA_Ready_arr[17];

    wire dpe_done =
        dpe_done_arr[0] & dpe_done_arr[1] & dpe_done_arr[2] & dpe_done_arr[3] & dpe_done_arr[4] & dpe_done_arr[5] &
        dpe_done_arr[6] & dpe_done_arr[7] & dpe_done_arr[8] & dpe_done_arr[9] & dpe_done_arr[10] & dpe_done_arr[11] &
        dpe_done_arr[12] & dpe_done_arr[13] & dpe_done_arr[14] & dpe_done_arr[15] & dpe_done_arr[16] & dpe_done_arr[17];

    wire shift_add_done =
        shift_add_done_arr[0] & shift_add_done_arr[1] & shift_add_done_arr[2] & shift_add_done_arr[3] & shift_add_done_arr[4] &
        shift_add_done_arr[5] & shift_add_done_arr[6] & shift_add_done_arr[7] & shift_add_done_arr[8] & shift_add_done_arr[9] &
        shift_add_done_arr[10] & shift_add_done_arr[11] & shift_add_done_arr[12] & shift_add_done_arr[13] & shift_add_done_arr[14] &
        shift_add_done_arr[15] & shift_add_done_arr[16] & shift_add_done_arr[17];

    wire shift_add_bypass_ctrl =
        shift_add_bypass_ctrl_arr[0] & shift_add_bypass_ctrl_arr[1] & shift_add_bypass_ctrl_arr[2] & shift_add_bypass_ctrl_arr[3] & shift_add_bypass_ctrl_arr[4] &
        shift_add_bypass_ctrl_arr[5] & shift_add_bypass_ctrl_arr[6] & shift_add_bypass_ctrl_arr[7] & shift_add_bypass_ctrl_arr[8] & shift_add_bypass_ctrl_arr[9] &
        shift_add_bypass_ctrl_arr[10] & shift_add_bypass_ctrl_arr[11] & shift_add_bypass_ctrl_arr[12] & shift_add_bypass_ctrl_arr[13] & shift_add_bypass_ctrl_arr[14] &
        shift_add_bypass_ctrl_arr[15] & shift_add_bypass_ctrl_arr[16] & shift_add_bypass_ctrl_arr[17];

    wire reg_full_sig =
        reg_full_arr[0] & reg_full_arr[1] & reg_full_arr[2] & reg_full_arr[3] & reg_full_arr[4] & reg_full_arr[5] &
        reg_full_arr[6] & reg_full_arr[7] & reg_full_arr[8] & reg_full_arr[9] & reg_full_arr[10] & reg_full_arr[11] &
        reg_full_arr[12] & reg_full_arr[13] & reg_full_arr[14] & reg_full_arr[15] & reg_full_arr[16] & reg_full_arr[17];

    // Adder tree: 9 pair adders at level 1
    wire [(N_KERNELS*DATA_WIDTH)-1:0] sum [0:8];
    wire done [0:8];

    generate
        for (i = 0; i < 9; i = i + 1) begin: ADDER_LAYER_1
            reg [DATA_WIDTH-1:0] input1_reg, input2_reg;
            reg accum_ready_reg;

            always @(posedge clk) begin
                if (rst) begin
                    input1_reg <= 0;
                    input2_reg <= 0;
                    accum_ready_reg <= 0;
                end else begin
                    input1_reg <= data_out_arr[2*i];
                    input2_reg <= data_out_arr[2*i + 1];
                    accum_ready_reg <= accum_ready;
                end
            end

            adder_dpe_N_CHANNELS_1 add_inst (
                .clk(clk), .reset(rst), .en(accum_ready_reg), .add_done(done[i]),
                .input1(input1_reg), .input2(input2_reg), .output_data(sum[i])
            );
        end
    endgenerate

    wire accum_done = done[0] & done[1] & done[2] & done[3] & done[4] & done[5] & done[6] & done[7] & done[8];

    // Pipelined reduction tree
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s0_0, s0_1, s0_2, s0_3, s0_4;
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s1_0, s1_1;
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s2_0;
    reg [(N_KERNELS*DATA_WIDTH)-1:0] final_sum;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            s0_0 <= 0; s0_1 <= 0; s0_2 <= 0; s0_3 <= 0; s0_4 <= 0;
            s1_0 <= 0; s1_1 <= 0;
            s2_0 <= 0;
            final_sum <= 0;
            data_out <= 0;
        end else if (accum_done) begin
            // Stage 0
            s0_0 <= sum[0] + sum[1];
            s0_1 <= sum[2] + sum[3];
            s0_2 <= sum[4] + sum[5];
            s0_3 <= sum[6] + sum[7];
            s0_4 <= sum[8];
            // Stage 1
            s1_0 <= s0_0 + s0_1;
            s1_1 <= s0_2 + s0_3;
            // Stage 2
            s2_0 <= s1_0 + s1_1;
            // Stage 3
            final_sum <= s2_0 + s0_4;
            // Output
            data_out <= final_sum;
        end
    end

    controller_scalable #(
        .N_CHANNELS(N_CHANNELS), .N_BRAM_R(N_BRAM_R), .N_BRAM_W(N_BRAM_W),
        .N_DPE_V(N_DPE_V), .N_DPE_H(N_DPE_H), .ADDR_WIDTH(ADDR_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH), .KERNEL_HEIGHT(KERNEL_HEIGHT),
        .W(W), .H(H), .S(S)
    ) controller_scalable_inst (
        .clk(clk), .rst(rst), .MSB_SA_Ready(MSB_SA_Ready),
        .valid(valid), .ready_n(ready_n), .dpe_done(dpe_done),
        .reg_full(reg_full_sig), .reg_empty(reg_empty_sig),
        .shift_add_done(shift_add_done), .shift_add_bypass_ctrl(shift_add_bypass_ctrl),
        .dpe_accum_done(accum_done),
        .read_address(read_address), .write_address(write_address),
        .w_buf_en(w_buf_en), .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control), .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg), .w_en_dec(w_en_dec_signal),
        .load_input_reg(load_input_reg), .ready(ready), .valid_n(valid_n),
        .dpe_accum_ready(accum_ready), .dpe_sel(dpe_sel_signal), .dpe_sel_h(dpe_sel_h_signal)
    );

endmodule


// conv_layer_stacked_dpes_V18_H2: 36 DPEs (18 vertical x 2 horizontal)
// Adapted from Azure-Lily V18_H2 for NL-DPE 16-bit interface
module conv_layer_stacked_dpes_V18_H2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 18,
    parameter N_DPE_H = 2,
    parameter N_BRAM_R = 1,
    parameter N_BRAM_W = 1,
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [(N_CHANNELS*DATA_WIDTH)-1:0] data_in,
    output reg [(N_KERNELS*DATA_WIDTH)-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_arr [0:35];
    wire shift_add_done_arr [0:35];
    wire shift_add_bypass_ctrl_arr [0:35];
    wire MSB_SA_Ready_arr [0:35];
    wire dpe_done_arr [0:35];
    wire reg_full_arr [0:35];
    wire [N_DPE_H-1:0] reg_empty_sig;
    wire [ADDR_WIDTH-1:0] read_address, write_address;
    wire w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire load_input_reg;
    wire [(N_BRAM_W-1):0] w_en_dec_signal;
    wire [N_DPE_V-1:0] dpe_sel_signal;
    wire [N_DPE_H-1:0] dpe_sel_h_signal;
    wire [DATA_WIDTH-1:0] sram_data;
    wire accum_ready;

    sram #(.N_CHANNELS(1), .DEPTH(512)) sram_inst_1 (
        .clk(clk), .rst(rst),
        .w_en(w_en_dec_signal),
        .r_addr(read_address), .w_addr(write_address),
        .sram_data_in(data_in), .sram_data_out(sram_data)
    );

    xbar_ip_module #(.DATA_WIDTH(16), .NUM_INPUTS(1), .NUM_OUTPUTS(36)) u_xbar (
        .in_data(sram_data), .in_sel(N_BRAM_R),
        .out_sel(dpe_sel_signal), .out_data(dpe_data)
    );

    genvar i;
    generate
        for (i = 0; i < 36; i = i + 1) begin: DPE_ARRAY
            reg [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data_reg;
            reg [1:0] nl_dpe_control_reg;
            reg shift_add_control_reg;
            reg w_buf_en_reg;
            reg shift_add_bypass_reg;
            reg load_output_reg_reg;
            reg load_input_reg_reg;

            always @(posedge clk) begin
                if (rst) begin
                    dpe_data_reg         <= 0;
                    nl_dpe_control_reg   <= 0;
                    shift_add_control_reg <= 0;
                    w_buf_en_reg         <= 0;
                    shift_add_bypass_reg <= 0;
                    load_output_reg_reg  <= 0;
                    load_input_reg_reg   <= 0;
                end else begin
                    dpe_data_reg         <= dpe_data;
                    nl_dpe_control_reg   <= nl_dpe_control;
                    shift_add_control_reg <= shift_add_control;
                    w_buf_en_reg         <= w_buf_en;
                    shift_add_bypass_reg <= shift_add_bypass;
                    load_output_reg_reg  <= load_output_reg;
                    load_input_reg_reg   <= load_input_reg;
                end
            end

            dpe dpe_inst (
                .clk(clk), .reset(rst),
                .data_in(dpe_data_reg),
                .nl_dpe_control(nl_dpe_control_reg),
                .shift_add_control(shift_add_control_reg),
                .w_buf_en(w_buf_en_reg),
                .shift_add_bypass(shift_add_bypass_reg),
                .load_output_reg(load_output_reg_reg),
                .load_input_reg(load_input_reg_reg),
                .MSB_SA_Ready(MSB_SA_Ready_arr[i]),
                .data_out(data_out_arr[i]),
                .dpe_done(dpe_done_arr[i]),
                .reg_full(reg_full_arr[i]),
                .shift_add_done(shift_add_done_arr[i]),
                .shift_add_bypass_ctrl(shift_add_bypass_ctrl_arr[i])
            );
        end
    endgenerate

    wire MSB_SA_Ready =
        MSB_SA_Ready_arr[0] & MSB_SA_Ready_arr[1] & MSB_SA_Ready_arr[2] & MSB_SA_Ready_arr[3] & MSB_SA_Ready_arr[4] &
        MSB_SA_Ready_arr[5] & MSB_SA_Ready_arr[6] & MSB_SA_Ready_arr[7] & MSB_SA_Ready_arr[8] & MSB_SA_Ready_arr[9] &
        MSB_SA_Ready_arr[10] & MSB_SA_Ready_arr[11] & MSB_SA_Ready_arr[12] & MSB_SA_Ready_arr[13] & MSB_SA_Ready_arr[14] &
        MSB_SA_Ready_arr[15] & MSB_SA_Ready_arr[16] & MSB_SA_Ready_arr[17] & MSB_SA_Ready_arr[18] & MSB_SA_Ready_arr[19] &
        MSB_SA_Ready_arr[20] & MSB_SA_Ready_arr[21] & MSB_SA_Ready_arr[22] & MSB_SA_Ready_arr[23] & MSB_SA_Ready_arr[24] &
        MSB_SA_Ready_arr[25] & MSB_SA_Ready_arr[26] & MSB_SA_Ready_arr[27] & MSB_SA_Ready_arr[28] & MSB_SA_Ready_arr[29] &
        MSB_SA_Ready_arr[30] & MSB_SA_Ready_arr[31] & MSB_SA_Ready_arr[32] & MSB_SA_Ready_arr[33] & MSB_SA_Ready_arr[34] &
        MSB_SA_Ready_arr[35];

    wire dpe_done =
        dpe_done_arr[0] & dpe_done_arr[1] & dpe_done_arr[2] & dpe_done_arr[3] & dpe_done_arr[4] & dpe_done_arr[5] &
        dpe_done_arr[6] & dpe_done_arr[7] & dpe_done_arr[8] & dpe_done_arr[9] & dpe_done_arr[10] & dpe_done_arr[11] &
        dpe_done_arr[12] & dpe_done_arr[13] & dpe_done_arr[14] & dpe_done_arr[15] & dpe_done_arr[16] & dpe_done_arr[17] &
        dpe_done_arr[18] & dpe_done_arr[19] & dpe_done_arr[20] & dpe_done_arr[21] & dpe_done_arr[22] & dpe_done_arr[23] &
        dpe_done_arr[24] & dpe_done_arr[25] & dpe_done_arr[26] & dpe_done_arr[27] & dpe_done_arr[28] & dpe_done_arr[29] &
        dpe_done_arr[30] & dpe_done_arr[31] & dpe_done_arr[32] & dpe_done_arr[33] & dpe_done_arr[34] & dpe_done_arr[35];

    wire shift_add_done =
        shift_add_done_arr[0] & shift_add_done_arr[1] & shift_add_done_arr[2] & shift_add_done_arr[3] & shift_add_done_arr[4] &
        shift_add_done_arr[5] & shift_add_done_arr[6] & shift_add_done_arr[7] & shift_add_done_arr[8] & shift_add_done_arr[9] &
        shift_add_done_arr[10] & shift_add_done_arr[11] & shift_add_done_arr[12] & shift_add_done_arr[13] & shift_add_done_arr[14] &
        shift_add_done_arr[15] & shift_add_done_arr[16] & shift_add_done_arr[17] & shift_add_done_arr[18] & shift_add_done_arr[19] &
        shift_add_done_arr[20] & shift_add_done_arr[21] & shift_add_done_arr[22] & shift_add_done_arr[23] & shift_add_done_arr[24] &
        shift_add_done_arr[25] & shift_add_done_arr[26] & shift_add_done_arr[27] & shift_add_done_arr[28] & shift_add_done_arr[29] &
        shift_add_done_arr[30] & shift_add_done_arr[31] & shift_add_done_arr[32] & shift_add_done_arr[33] & shift_add_done_arr[34] &
        shift_add_done_arr[35];

    wire shift_add_bypass_ctrl =
        shift_add_bypass_ctrl_arr[0] & shift_add_bypass_ctrl_arr[1] & shift_add_bypass_ctrl_arr[2] & shift_add_bypass_ctrl_arr[3] & shift_add_bypass_ctrl_arr[4] &
        shift_add_bypass_ctrl_arr[5] & shift_add_bypass_ctrl_arr[6] & shift_add_bypass_ctrl_arr[7] & shift_add_bypass_ctrl_arr[8] & shift_add_bypass_ctrl_arr[9] &
        shift_add_bypass_ctrl_arr[10] & shift_add_bypass_ctrl_arr[11] & shift_add_bypass_ctrl_arr[12] & shift_add_bypass_ctrl_arr[13] & shift_add_bypass_ctrl_arr[14] &
        shift_add_bypass_ctrl_arr[15] & shift_add_bypass_ctrl_arr[16] & shift_add_bypass_ctrl_arr[17] & shift_add_bypass_ctrl_arr[18] & shift_add_bypass_ctrl_arr[19] &
        shift_add_bypass_ctrl_arr[20] & shift_add_bypass_ctrl_arr[21] & shift_add_bypass_ctrl_arr[22] & shift_add_bypass_ctrl_arr[23] & shift_add_bypass_ctrl_arr[24] &
        shift_add_bypass_ctrl_arr[25] & shift_add_bypass_ctrl_arr[26] & shift_add_bypass_ctrl_arr[27] & shift_add_bypass_ctrl_arr[28] & shift_add_bypass_ctrl_arr[29] &
        shift_add_bypass_ctrl_arr[30] & shift_add_bypass_ctrl_arr[31] & shift_add_bypass_ctrl_arr[32] & shift_add_bypass_ctrl_arr[33] & shift_add_bypass_ctrl_arr[34] &
        shift_add_bypass_ctrl_arr[35];

    wire reg_full_sig =
        reg_full_arr[0] & reg_full_arr[1] & reg_full_arr[2] & reg_full_arr[3] & reg_full_arr[4] & reg_full_arr[5] &
        reg_full_arr[6] & reg_full_arr[7] & reg_full_arr[8] & reg_full_arr[9] & reg_full_arr[10] & reg_full_arr[11] &
        reg_full_arr[12] & reg_full_arr[13] & reg_full_arr[14] & reg_full_arr[15] & reg_full_arr[16] & reg_full_arr[17] &
        reg_full_arr[18] & reg_full_arr[19] & reg_full_arr[20] & reg_full_arr[21] & reg_full_arr[22] & reg_full_arr[23] &
        reg_full_arr[24] & reg_full_arr[25] & reg_full_arr[26] & reg_full_arr[27] & reg_full_arr[28] & reg_full_arr[29] &
        reg_full_arr[30] & reg_full_arr[31] & reg_full_arr[32] & reg_full_arr[33] & reg_full_arr[34] & reg_full_arr[35];

    // Adder tree: 18 pair adders at level 1
    wire [(N_KERNELS*DATA_WIDTH)-1:0] sum [0:17];
    wire done [0:17];

    generate
        for (i = 0; i < 18; i = i + 1) begin: ADDER_LAYER_1
            reg [DATA_WIDTH-1:0] input1_reg, input2_reg;
            reg accum_ready_reg;

            always @(posedge clk) begin
                if (rst) begin
                    input1_reg <= 0;
                    input2_reg <= 0;
                    accum_ready_reg <= 0;
                end else begin
                    input1_reg <= data_out_arr[2*i];
                    input2_reg <= data_out_arr[2*i + 1];
                    accum_ready_reg <= accum_ready;
                end
            end

            adder_dpe_N_CHANNELS_1 add_inst (
                .clk(clk), .reset(rst), .en(accum_ready_reg), .add_done(done[i]),
                .input1(input1_reg), .input2(input2_reg), .output_data(sum[i])
            );
        end
    endgenerate

    wire accum_done =
        done[0] & done[1] & done[2] & done[3] & done[4] & done[5] & done[6] & done[7] & done[8] &
        done[9] & done[10] & done[11] & done[12] & done[13] & done[14] & done[15] & done[16] & done[17];

    // Pipelined reduction tree (18 -> 9 -> 5 -> 3 -> 2 -> 1)
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7, s0_8;
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s1_0, s1_1, s1_2, s1_3;
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s2_0, s2_1;
    reg [(N_KERNELS*DATA_WIDTH)-1:0] s3_0;
    reg [(N_KERNELS*DATA_WIDTH)-1:0] final_sum;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            s0_0 <= 0; s0_1 <= 0; s0_2 <= 0; s0_3 <= 0; s0_4 <= 0;
            s0_5 <= 0; s0_6 <= 0; s0_7 <= 0; s0_8 <= 0;
            s1_0 <= 0; s1_1 <= 0; s1_2 <= 0; s1_3 <= 0;
            s2_0 <= 0; s2_1 <= 0;
            s3_0 <= 0;
            final_sum <= 0;
            data_out <= 0;
        end else if (accum_done) begin
            // Stage 0 (9 pair sums)
            s0_0 <= sum[0] + sum[1];
            s0_1 <= sum[2] + sum[3];
            s0_2 <= sum[4] + sum[5];
            s0_3 <= sum[6] + sum[7];
            s0_4 <= sum[8] + sum[9];
            s0_5 <= sum[10] + sum[11];
            s0_6 <= sum[12] + sum[13];
            s0_7 <= sum[14] + sum[15];
            s0_8 <= sum[16] + sum[17];
            // Stage 1 (4 pair sums)
            s1_0 <= s0_0 + s0_1;
            s1_1 <= s0_2 + s0_3;
            s1_2 <= s0_4 + s0_5;
            s1_3 <= s0_6 + s0_7;
            // Stage 2 (2 pair sums)
            s2_0 <= s1_0 + s1_1;
            s2_1 <= s1_2 + s1_3;
            // Stage 3 (1 sum)
            s3_0 <= s2_0 + s2_1;
            // Stage 4 (final + leftover s0_8)
            final_sum <= s3_0 + s0_8;
            // Output
            data_out <= final_sum;
        end
    end

    controller_scalable #(
        .N_CHANNELS(N_CHANNELS), .N_BRAM_R(N_BRAM_R), .N_BRAM_W(N_BRAM_W),
        .N_DPE_V(N_DPE_V), .N_DPE_H(N_DPE_H), .ADDR_WIDTH(ADDR_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH), .KERNEL_HEIGHT(KERNEL_HEIGHT),
        .W(W), .H(H), .S(S)
    ) controller_scalable_inst (
        .clk(clk), .rst(rst), .MSB_SA_Ready(MSB_SA_Ready),
        .valid(valid), .ready_n(ready_n), .dpe_done(dpe_done),
        .reg_full(reg_full_sig), .reg_empty(reg_empty_sig),
        .shift_add_done(shift_add_done), .shift_add_bypass_ctrl(shift_add_bypass_ctrl),
        .dpe_accum_done(accum_done),
        .read_address(read_address), .write_address(write_address),
        .w_buf_en(w_buf_en), .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control), .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg), .w_en_dec(w_en_dec_signal),
        .load_input_reg(load_input_reg), .ready(ready), .valid_n(valid_n),
        .dpe_accum_ready(accum_ready), .dpe_sel(dpe_sel_signal), .dpe_sel_h(dpe_sel_h_signal)
    );

endmodule


// ============================================================
// NEW VGG11-SPECIFIC MODULES (pool_layer5, avg_pool, avg_pooling 16-bit)
// ============================================================

module pool_layer5 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,
    parameter KERNEL_SIZE = 2
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire layer_done,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire ready,
    output wire valid_n,
    output wire [DATA_WIDTH-1:0] data_out
);
    wire [ADDR_WIDTH-1:0] read_address, write_address;
    wire w_buf_en, w_en, en, reg_full, pooling_done, load_output_reg;
    wire [1:0] pooling_control;
    wire [DATA_WIDTH-1:0] sram_data_in;

    pooling_controller #(
        .N_CHANNELS(N_CHANNELS), .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1), .KERNEL_SIZE(2)
    ) pooling_ctrl_inst (
        .clk(clk), .rst(rst), .pooling_done(pooling_done),
        .valid(valid), .ready_n(ready_n), .layer_done(layer_done),
        .reg_full(reg_full), .read_address(read_address), .write_address(write_address),
        .w_buf_en(w_buf_en), .p_en(en), .pooling_control(pooling_control),
        .load_output_reg(load_output_reg), .w_en(w_en), .ready(ready), .valid_n(valid_n)
    );

    sram #(.N_CHANNELS(N_CHANNELS), .DEPTH(DEPTH)) sram_inst (
        .clk(clk), .rst(rst), .w_en(w_en),
        .r_addr(read_address), .w_addr(write_address),
        .sram_data_in(data_in), .sram_data_out(sram_data_in)
    );

    max_pooling_N_CHANNELS_1 #(.N_CHANNELS(N_CHANNELS)) max_pooling_inst (
        .clk(clk), .reset(rst), .en(en),
        .load_input_reg(w_buf_en), .reg_full(reg_full),
        .load_output_reg(load_output_reg), .pooling_done(pooling_done),
        .sram_data_in(sram_data_in), .sram_data_out(data_out)
    );
endmodule


module avg_pool_layer1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 2,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 4,
    parameter KERNEL_SIZE = 2
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire layer_done,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire ready,
    output wire valid_n,
    output wire [DATA_WIDTH-1:0] data_out
);
    wire [ADDR_WIDTH-1:0] read_address, write_address;
    wire w_buf_en, w_en, en, reg_full, pooling_done, load_output_reg;
    wire [1:0] pooling_control;
    wire [DATA_WIDTH-1:0] sram_data_in;

    pooling_controller #(
        .N_CHANNELS(N_CHANNELS), .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1), .KERNEL_SIZE(2)
    ) pooling_ctrl_inst (
        .clk(clk), .rst(rst), .pooling_done(pooling_done),
        .valid(valid), .ready_n(ready_n), .layer_done(layer_done),
        .reg_full(reg_full), .read_address(read_address), .write_address(write_address),
        .w_buf_en(w_buf_en), .p_en(en), .pooling_control(pooling_control),
        .load_output_reg(load_output_reg), .w_en(w_en), .ready(ready), .valid_n(valid_n)
    );

    sram #(.N_CHANNELS(N_CHANNELS), .DEPTH(DEPTH)) sram_inst (
        .clk(clk), .rst(rst), .w_en(w_en),
        .r_addr(read_address), .w_addr(write_address),
        .sram_data_in(data_in), .sram_data_out(sram_data_in)
    );

    avg_pooling_N_CHANNELS_1 #(.N_CHANNELS(N_CHANNELS)) avg_pooling_inst (
        .clk(clk), .reset(rst), .en(en),
        .load_input_reg(w_buf_en), .reg_full(reg_full),
        .load_output_reg(load_output_reg), .pooling_done(pooling_done),
        .sram_data_in(sram_data_in), .sram_data_out(data_out)
    );
endmodule


// 16-bit average pooling
module avg_pooling_N_CHANNELS_1 #(
    parameter N_CHANNELS = 1
)(
    input wire clk,
    input wire reset,
    input wire en,
    input wire load_input_reg,
    output reg reg_full,
    input wire load_output_reg,
    output reg pooling_done,
    input wire [16*N_CHANNELS-1:0] sram_data_in,
    output reg [16*N_CHANNELS-1:0] sram_data_out
);
    reg [15:0] input_0_0, input_1_0, input_2_0, input_3_0;
    reg [17:0] sum_0_1, sum_2_3, total_sum;
    reg [15:0] avg_value;
    reg [1:0] read_count;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            input_0_0 <= 16'd0; input_1_0 <= 16'd0;
            input_2_0 <= 16'd0; input_3_0 <= 16'd0;
            read_count <= 2'd0;
            reg_full <= 1'b0;
        end else if (en && load_input_reg) begin
            read_count <= read_count + 1;
            case (read_count)
                2'd0: input_0_0 <= sram_data_in[15:0];
                2'd1: input_1_0 <= sram_data_in[15:0];
                2'd2: input_2_0 <= sram_data_in[15:0];
                2'd3: begin
                    input_3_0 <= sram_data_in[15:0];
                    reg_full <= 1'b1;
                end
            endcase
        end else begin
            reg_full <= 1'b0;
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            sum_0_1 <= 18'd0; sum_2_3 <= 18'd0; total_sum <= 18'd0;
            avg_value <= 16'd0;
            pooling_done <= 1'b0;
        end else if (reg_full) begin
            sum_0_1 <= input_0_0 + input_1_0;
            sum_2_3 <= input_2_0 + input_3_0;
            total_sum <= sum_0_1 + sum_2_3;
            avg_value <= total_sum[17:2]; // divide by 4
            pooling_done <= 1'b1;
        end else begin
            pooling_done <= 1'b0;
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset)
            sram_data_out[15:0] <= 16'd0;
        else if (load_output_reg)
            sram_data_out[15:0] <= avg_value;
    end
endmodule
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

module conv_layer_stacked_dpes_V2_H1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 2,
    parameter N_DPE_H = 1,
    parameter N_BRAM_R = 1,
    parameter N_BRAM_W = 1,
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

    // Internal signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [N_DPE_V-1:0] reg_full_sig;
    wire [N_DPE_H-1:0] reg_empty;
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
    wire [DATA_WIDTH-1:0] sram_data_in;
    wire [DATA_WIDTH-1:0] sram_data_out;

	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out1;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out2;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_temp;
	wire [N_BRAM_W-1:0] w_en_dec_signal;
	wire [N_DPE_V-1:0] dpe_sel_signal;
	wire [N_DPE_H-1:0] dpe_sel_h_signal;
	wire [DATA_WIDTH-1:0] sram_data;
	wire dpe_done1, dpe_done2;
	wire accum_ready, accum_done;
	wire shift_add_done1, shift_add_done2;
	wire shift_add_bypass_ctrl1, shift_add_bypass_ctrl2;
	wire MSB_SA_Ready1, MSB_SA_Ready2;

    // Instantiate the SRAM module
    sram #(
        .N_CHANNELS(1),
        .DEPTH(512)
    ) sram_inst_1 (
        .clk(clk),
		.rst(rst),
        .w_en(w_en_dec_signal),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data)
    );

	// Instantiate crossbar module for inputs
	xbar_ip_module #(
        .DATA_WIDTH(16),
        .NUM_INPUTS(1),
        .NUM_OUTPUTS(2)
    ) u_xbar (
        .in_data(sram_data),
        .in_sel(N_BRAM_R),
        .out_sel(dpe_sel_signal),
        .out_data(dpe_data)
    );

    // Instantiate the DPE modules (16-bit direct, no zero padding)
    dpe dpe_R1_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready1),
        .data_out(data_out1),
        .dpe_done(dpe_done1),
        .reg_full(reg_full_sig[0]),
        .shift_add_done(shift_add_done1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl1)
    );

 	dpe dpe_R2_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready2),
        .data_out(data_out2),
        .dpe_done(dpe_done2),
        .reg_full(reg_full_sig[1]),
        .shift_add_done(shift_add_done2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl2)
    );

  	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done),
        .input1(data_out1),
        .input2(data_out2),
        .output_data(data_out)
    );

	assign shift_add_done = shift_add_done1 & shift_add_done2;
	assign shift_add_bypass_ctrl = shift_add_bypass_ctrl1 & shift_add_bypass_ctrl2;
	assign dpe_done = dpe_done1 & dpe_done2;
	assign MSB_SA_Ready = MSB_SA_Ready1 & MSB_SA_Ready2;

    // Instantiate the Controller module
 	controller_scalable #(
		.N_CHANNELS(N_CHANNELS),
		.N_BRAM_R(N_BRAM_R),
		.N_BRAM_W(N_BRAM_W),
		.N_DPE_V(N_DPE_V),
		.N_DPE_H(N_DPE_H),
		.ADDR_WIDTH(ADDR_WIDTH),
		.KERNEL_WIDTH(KERNEL_WIDTH),
		.KERNEL_HEIGHT(KERNEL_HEIGHT),
		.W(W),
		.H(H),
		.S(S)
	) controller_scalable_inst (
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
		.dpe_accum_done(accum_done),
		.read_address(read_address),
        .write_address(write_address),
        .w_buf_en(w_buf_en),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
		.w_en_dec(w_en_dec_signal),
		.load_input_reg(load_input_reg),
		.ready(ready),
        .valid_n(valid_n),
		.dpe_accum_ready(accum_ready),
		.dpe_sel(dpe_sel_signal),
		.dpe_sel_h(dpe_sel_h_signal)
    );

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

module sram #(
    parameter N_CHANNELS = 1,
    parameter DATA_WIDTH = 16*N_CHANNELS,  // Data width (default: 16 bits) 16 x number of channels
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


module xbar_ip_module #(
    parameter DATA_WIDTH = 16,
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


module pooling_controller #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_SIZE = 3,
    parameter B_ADDR_WIDTH = $clog2(KERNEL_SIZE * KERNEL_SIZE)
)(
    input wire clk,
    input wire rst,
    input wire pooling_done,
    input wire valid,
    input wire ready_n,
    input wire layer_done,
    input wire reg_full,
    output reg [ADDR_WIDTH-1:0] read_address,
    output reg [ADDR_WIDTH-1:0] write_address,
    output reg w_buf_en,
    output reg p_en,
    output reg [1:0] pooling_control,
    output reg load_output_reg,
    output reg w_en,
    output reg ready,
    output reg valid_n
);

    reg [ADDR_WIDTH-1:0] write_address_reg, read_address_reg;
    wire buf_load, busy;
    reg memory_flag;
    reg stall;
    reg memory_stall;
    reg pooling_exec_signal;

    // always block for sram control
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            write_address_reg <= {ADDR_WIDTH{1'b0}};
            read_address_reg <= {ADDR_WIDTH{1'b0}};
            w_en <= 1'b0;
            w_buf_en <= 0;
            p_en <= 0;
        end else begin
            // enable write
            if (valid && ~memory_stall) begin
                write_address_reg <= write_address_reg + 1;
                w_en <= 1'b1;
            end else begin
                write_address_reg <= write_address_reg;
                w_en <= 1'b0;
            end
            if (~reg_full && ready_n && memory_flag) begin
                read_address_reg <= read_address_reg + 1;
                w_buf_en <= 1;
            end else begin
                read_address_reg <= read_address_reg;
                w_buf_en <= 0;
            end
            if (~stall) begin
                p_en <= 1;
            end else begin
                p_en <= 0;
            end
        end
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
            pooling_exec_signal <= 1'b0;
        end else begin
            if (reg_full) begin
                pooling_exec_signal <= 1'b1;
            end else begin
                pooling_exec_signal <= 1'b0;
            end
        end
    end

    // dpe control
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pooling_control <= 2'b0;
        end else begin
            if (pooling_exec_signal) begin
                pooling_control <= 2'b11;
            end else begin
                pooling_control <= 2'b00;
            end
        end
    end

    // valid and ready control
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_n <= 1'b0;
            ready <= 1'b0;
            stall <= 0;
            memory_stall <= 0;
        end else begin
            if((read_address_reg < write_address_reg - 2) && (write_address_reg == {ADDR_WIDTH{1'b1}})) begin
                memory_stall <= 1;
            end else begin
                memory_stall <= 0;
            end

            if (memory_stall) begin
                ready <= 1'b0;
            end else begin
                ready <= 1'b1;
            end

            if(layer_done) begin
                valid_n <= 1;
            end else begin
                valid_n <= 0;
            end

            if(ready_n) begin
                stall <= 0;
            end else begin
                stall <= 1;
            end
        end
    end

    always @* begin
        read_address <= read_address_reg;
        write_address <= write_address_reg;
        load_output_reg <= pooling_done;
    end

    assign busy = ~pooling_done;

endmodule

module pool_layer1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,
    parameter KERNEL_SIZE = 2
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire layer_done,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire ready,
    output wire valid_n,
    output wire [DATA_WIDTH-1:0] data_out
);

    // Wires for interconnections
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire w_buf_en;
    wire [1:0] pooling_control;
    wire load_output_reg;
    wire w_en;
    wire en;
    wire reg_full;
    wire pooling_done;
    wire [DATA_WIDTH-1:0] sram_data_in;

    // Instantiate pooling_controller
    pooling_controller #(
        .N_CHANNELS(N_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_SIZE(2)
    ) pooling_ctrl_inst (
        .clk(clk),
        .rst(rst),
        .pooling_done(pooling_done),
        .valid(valid),
        .ready_n(ready_n),
        .layer_done(layer_done),
        .reg_full(reg_full),
        .read_address(read_address),
        .write_address(write_address),
        .w_buf_en(w_buf_en),
        .p_en(en),
        .pooling_control(pooling_control),
        .load_output_reg(load_output_reg),
        .w_en(w_en),
        .ready(ready),
        .valid_n(valid_n)
    );

    // Instantiate sram
    sram #(
        .N_CHANNELS(N_CHANNELS),
        .DEPTH(DEPTH)
    ) sram_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_in)
    );

    // Instantiate max pooling
    max_pooling_N_CHANNELS_1 #(
        .N_CHANNELS(N_CHANNELS)
    ) max_pooling_inst (
        .clk(clk),
        .reset(rst),
        .en(en),
        .load_input_reg(w_buf_en),
        .reg_full(reg_full),
        .load_output_reg(load_output_reg),
        .pooling_done(pooling_done),
        .sram_data_in(sram_data_in),
        .sram_data_out(data_out)
    );

endmodule

module pool_layer2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,
    parameter KERNEL_SIZE = 2
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire layer_done,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire ready,
    output wire valid_n,
    output wire [DATA_WIDTH-1:0] data_out
);

    // Wires for interconnections
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire w_buf_en;
    wire [1:0] pooling_control;
    wire load_output_reg;
    wire w_en;
    wire en;
    wire reg_full;
    wire pooling_done;
    wire [DATA_WIDTH-1:0] sram_data_in;

    // Instantiate pooling_controller
    pooling_controller #(
        .N_CHANNELS(N_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_SIZE(2)
    ) pooling_ctrl_inst (
        .clk(clk),
        .rst(rst),
        .pooling_done(pooling_done),
        .valid(valid),
        .ready_n(ready_n),
        .layer_done(layer_done),
        .reg_full(reg_full),
        .read_address(read_address),
        .write_address(write_address),
        .w_buf_en(w_buf_en),
        .p_en(en),
        .pooling_control(pooling_control),
        .load_output_reg(load_output_reg),
        .w_en(w_en),
        .ready(ready),
        .valid_n(valid_n)
    );

    // Instantiate sram
    sram #(
        .N_CHANNELS(N_CHANNELS),
        .DEPTH(DEPTH)
    ) sram_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_in)
    );

    // Instantiate max pooling
    max_pooling_N_CHANNELS_1 #(
        .N_CHANNELS(N_CHANNELS)
    ) max_pooling_inst (
        .clk(clk),
        .reset(rst),
        .en(en),
        .load_input_reg(w_buf_en),
        .reg_full(reg_full),
        .load_output_reg(load_output_reg),
        .pooling_done(pooling_done),
        .sram_data_in(sram_data_in),
        .sram_data_out(data_out)
    );

endmodule

module pool_layer3 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,
    parameter KERNEL_SIZE = 2
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire layer_done,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire ready,
    output wire valid_n,
    output wire [DATA_WIDTH-1:0] data_out
);

    // Wires for interconnections
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire w_buf_en;
    wire [1:0] pooling_control;
    wire load_output_reg;
    wire w_en;
    wire en;
    wire reg_full;
    wire pooling_done;
    wire [DATA_WIDTH-1:0] sram_data_in;

    // Instantiate pooling_controller
    pooling_controller #(
        .N_CHANNELS(N_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_SIZE(2)
    ) pooling_ctrl_inst (
        .clk(clk),
        .rst(rst),
        .pooling_done(pooling_done),
        .valid(valid),
        .ready_n(ready_n),
        .layer_done(layer_done),
        .reg_full(reg_full),
        .read_address(read_address),
        .write_address(write_address),
        .w_buf_en(w_buf_en),
        .p_en(en),
        .pooling_control(pooling_control),
        .load_output_reg(load_output_reg),
        .w_en(w_en),
        .ready(ready),
        .valid_n(valid_n)
    );

    // Instantiate sram
    sram #(
        .N_CHANNELS(N_CHANNELS),
        .DEPTH(DEPTH)
    ) sram_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_in)
    );

    // Instantiate max pooling
    max_pooling_N_CHANNELS_1 #(
        .N_CHANNELS(N_CHANNELS)
    ) max_pooling_inst (
        .clk(clk),
        .reset(rst),
        .en(en),
        .load_input_reg(w_buf_en),
        .reg_full(reg_full),
        .load_output_reg(load_output_reg),
        .pooling_done(pooling_done),
        .sram_data_in(sram_data_in),
        .sram_data_out(data_out)
    );

endmodule

module pool_layer4 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,
    parameter KERNEL_SIZE = 2
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire layer_done,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire ready,
    output wire valid_n,
    output wire [DATA_WIDTH-1:0] data_out
);

    // Wires for interconnections
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire w_buf_en;
    wire [1:0] pooling_control;
    wire load_output_reg;
    wire w_en;
    wire en;
    wire reg_full;
    wire pooling_done;
    wire [DATA_WIDTH-1:0] sram_data_in;

    // Instantiate pooling_controller
    pooling_controller #(
        .N_CHANNELS(N_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_SIZE(2)
    ) pooling_ctrl_inst (
        .clk(clk),
        .rst(rst),
        .pooling_done(pooling_done),
        .valid(valid),
        .ready_n(ready_n),
        .layer_done(layer_done),
        .reg_full(reg_full),
        .read_address(read_address),
        .write_address(write_address),
        .w_buf_en(w_buf_en),
        .p_en(en),
        .pooling_control(pooling_control),
        .load_output_reg(load_output_reg),
        .w_en(w_en),
        .ready(ready),
        .valid_n(valid_n)
    );

    // Instantiate sram
    sram #(
        .N_CHANNELS(N_CHANNELS),
        .DEPTH(DEPTH)
    ) sram_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_in)
    );

    // Instantiate max pooling
    max_pooling_N_CHANNELS_1 #(
        .N_CHANNELS(N_CHANNELS)
    ) max_pooling_inst (
        .clk(clk),
        .reset(rst),
        .en(en),
        .load_input_reg(w_buf_en),
        .reg_full(reg_full),
        .load_output_reg(load_output_reg),
        .pooling_done(pooling_done),
        .sram_data_in(sram_data_in),
        .sram_data_out(data_out)
    );

endmodule


module max_pooling_N_CHANNELS_1 #(
    parameter N_CHANNELS = 1  // Number of channels, each channel has 4 inputs
)(
    input wire clk,
    input wire reset,
    input wire en,
    input wire load_input_reg,
    output reg reg_full,
    input wire load_output_reg,
    output reg pooling_done,
    input wire [16*N_CHANNELS-1:0] sram_data_in,
    output reg [16*N_CHANNELS-1:0] sram_data_out
);

    reg [15:0] input_0_0, input_1_0, input_2_0, input_3_0; // Channel 0

    reg [15:0] max_0_1_0, max_2_3_0, max_pool_value_0; // Max values for channel 0

    reg [1:0] read_count;  // Counter to track the number of read operations

    // Block 1: Reading and Storing Data
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            input_0_0 <= 16'd0; input_1_0 <= 16'd0; input_2_0 <= 16'd0; input_3_0 <= 16'd0;
            read_count <= 2'd0;
            reg_full <= 1'b0;
        end else if (en && load_input_reg) begin
            // Increment read count and load inputs based on read count
            read_count <= read_count + 1;
            case (read_count)
                2'd0: begin
                    input_0_0 <= sram_data_in[15:0];
                end
                2'd1: begin
                    input_1_0 <= sram_data_in[15:0];
                end
                2'd2: begin
                    input_2_0 <= sram_data_in[15:0];
                end
                2'd3: begin
                    input_3_0 <= sram_data_in[15:0];
                    reg_full <= 1'b1; // Indicate that all inputs are filled
                end
            endcase
        end else begin
            reg_full <= 1'b0;
        end
    end

    // Block 2: Max-Pooling Logic
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            max_0_1_0 <= 16'd0; max_2_3_0 <= 16'd0; max_pool_value_0 <= 16'd0;
            pooling_done <= 1'b0;
        end else if (reg_full) begin
            max_0_1_0 <= (input_0_0 > input_1_0) ? input_0_0 : input_1_0;
            max_2_3_0 <= (input_2_0 > input_3_0) ? input_2_0 : input_3_0;
            max_pool_value_0 <= (max_0_1_0 > max_2_3_0) ? max_0_1_0 : max_2_3_0;
            pooling_done <= 1'b1; // Signal that pooling is done
        end else begin
            pooling_done <= 1'b0;
        end
    end

    // Block 3: Storing Max-Pooled Data into Output Registers
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            sram_data_out[15:0] <= 16'd0;
        end else if (load_output_reg) begin
            sram_data_out[15:0] <= max_pool_value_0;
        end
    end
endmodule


module residual_layer1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 256,
    parameter KERNEL_SIZE = 2
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire layer_done,
    input wire [DATA_WIDTH-1:0] data_in,
    input wire [DATA_WIDTH-1:0] data_in2,
    output wire ready,
    output wire valid_n,
    output wire [DATA_WIDTH-1:0] data_out
);

    // Wires for interconnections
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire w_buf_en;
    wire [1:0] pooling_control;
    wire load_output_reg;
    wire w_en;
    wire en;
    wire reg_full;
    wire pooling_done;
    wire [DATA_WIDTH-1:0] sram_data_in;
    wire [DATA_WIDTH-1:0] sram_data_in2;

    // Instantiate pooling_controller
    pooling_controller #(
        .N_CHANNELS(N_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_SIZE(2)
    ) pooling_ctrl_inst (
        .clk(clk),
        .rst(rst),
        .pooling_done(pooling_done),
        .valid(valid),
        .ready_n(ready_n),
        .layer_done(layer_done),
        .reg_full(reg_full),
        .read_address(read_address),
        .write_address(write_address),
        .w_buf_en(w_buf_en),
        .p_en(en),
        .pooling_control(pooling_control),
        .load_output_reg(load_output_reg),
        .w_en(w_en),
        .ready(ready),
        .valid_n(valid_n)
    );

    // Instantiate sram
    sram #(
        .N_CHANNELS(N_CHANNELS),
        .DEPTH(DEPTH)
    ) sram_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_in)
    );

    // Instantiate adder for residual connection
    adder_dpe_N_CHANNELS_1_with_reg #(
        .N_CHANNELS(1)
    ) adder_inst (
        .clk(clk),
        .reset(rst),
        .en(en),
        .load_input_reg(w_buf_en),
        .reg_full(reg_full),
        .load_output_reg(load_output_reg),
        .add_done(pooling_done),
        .input1(sram_data_in),
        .input2(sram_data_in2),
        .output_data(data_out)
    );

	// Instantiate sram for second input
    sram #(
        .N_CHANNELS(N_CHANNELS),
        .DEPTH(DEPTH)
    ) sram2_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in2),
        .sram_data_out(sram_data_in2)
    );

endmodule

module residual_layer2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 4,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 16,
    parameter KERNEL_SIZE = 2
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire layer_done,
    input wire [DATA_WIDTH-1:0] data_in,
    input wire [DATA_WIDTH-1:0] data_in2,
    output wire ready,
    output wire valid_n,
    output wire [DATA_WIDTH-1:0] data_out
);

    // Wires for interconnections
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire w_buf_en;
    wire [1:0] pooling_control;
    wire load_output_reg;
    wire w_en;
    wire en;
    wire reg_full;
    wire pooling_done;
    wire [DATA_WIDTH-1:0] sram_data_in;
    wire [DATA_WIDTH-1:0] sram_data_in2;

    // Instantiate pooling_controller
    pooling_controller #(
        .N_CHANNELS(N_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_SIZE(2)
    ) pooling_ctrl_inst (
        .clk(clk),
        .rst(rst),
        .pooling_done(pooling_done),
        .valid(valid),
        .ready_n(ready_n),
        .layer_done(layer_done),
        .reg_full(reg_full),
        .read_address(read_address),
        .write_address(write_address),
        .w_buf_en(w_buf_en),
        .p_en(en),
        .pooling_control(pooling_control),
        .load_output_reg(load_output_reg),
        .w_en(w_en),
        .ready(ready),
        .valid_n(valid_n)
    );

    // Instantiate sram
    sram #(
        .N_CHANNELS(N_CHANNELS),
        .DEPTH(DEPTH)
    ) sram_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_in)
    );

    // Instantiate adder for residual connection
    adder_dpe_N_CHANNELS_1_with_reg #(
        .N_CHANNELS(1)
    ) adder_inst (
        .clk(clk),
        .reset(rst),
        .en(en),
        .load_input_reg(w_buf_en),
        .reg_full(reg_full),
        .load_output_reg(load_output_reg),
        .add_done(pooling_done),
        .input1(sram_data_in),
        .input2(sram_data_in2),
        .output_data(data_out)
    );

	// Instantiate sram for second input
    sram #(
        .N_CHANNELS(N_CHANNELS),
        .DEPTH(DEPTH)
    ) sram2_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in2),
        .sram_data_out(sram_data_in2)
    );

endmodule

module adder_dpe_N_CHANNELS_1 #(
    parameter N_CHANNELS = 1  // Number of channels
)(
    input wire clk,
    input wire reset,
    input wire en,
    output reg add_done,
    input wire [16*N_CHANNELS-1:0] input1,
    input wire [16*N_CHANNELS-1:0] input2,
    output reg [16*N_CHANNELS-1:0] output_data
);

    reg [15:0] in_data1_0;  // Input storage for first set of inputs for channel 0
    reg [15:0] in_data2_0;  // Input storage for second set of inputs for channel 0
    reg [16:0] sum_0;       // 17-bit sum storage for channel 0 (includes LSB to drop)
    reg [15:0] result_0;    // 16-bit result after dropping LSB for channel 0

    reg [7:0] channel_index;  // Index to track current channel being processed
    reg processing;           // Flag to indicate if processing is ongoing

    // Block 1: Reading and Storing Data
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            in_data1_0 <= 16'd0;
            in_data2_0 <= 16'd0;
        end else if (en) begin
            in_data1_0 <= input1[15:0];
            in_data2_0 <= input2[15:0];
        end else begin
        end
    end

    // Block 2: Addition Logic with LSB Dropped
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            result_0 <= 16'd0;
            channel_index <= 8'd0;
            processing <= 1'b0;
            add_done <= 1'b0;
        end else if (!processing) begin
            sum_0 <= in_data1_0 + in_data2_0;
            result_0 <= sum_0[16:1];  // Right shift to drop LSB
            processing <= 1'b1;
        end else if (processing) begin
            channel_index <= channel_index + 1;
            if (channel_index == (N_CHANNELS - 1)) begin
                add_done <= 1'b1;
                processing <= 1'b0;
                channel_index <= 8'd0;
            end else begin
                processing <= 1'b0;
            end
        end else begin
            add_done <= 1'b0;
        end
    end

    // Block 3: Storing Output Data into Output Registers
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            output_data[15:0] <= 16'd0;
        end else begin
            output_data[15:0] <= result_0;
        end
    end
endmodule

module adder_dpe_N_CHANNELS_1_with_reg #(
    parameter N_CHANNELS = 1  // Number of channels
)(
    input wire clk,
    input wire reset,
    input wire en,
    input wire load_input_reg,
    output reg reg_full,
    input wire load_output_reg,
    output reg add_done,
    input wire [16*N_CHANNELS-1:0] input1,
    input wire [16*N_CHANNELS-1:0] input2,
    output reg [16*N_CHANNELS-1:0] output_data
);

    reg [15:0] in_data1_0;  // Input storage for first set of inputs for channel 0
    reg [15:0] in_data2_0;  // Input storage for second set of inputs for channel 0
    reg [16:0] sum_0;       // 17-bit sum storage for channel 0 (includes LSB to drop)
    reg [15:0] result_0;    // 16-bit result after dropping LSB for channel 0

    reg [7:0] channel_index;  // Index to track current channel being processed
    reg processing;           // Flag to indicate if processing is ongoing

    // Block 1: Reading and Storing Data
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            in_data1_0 <= 16'd0;
            in_data2_0 <= 16'd0;
            reg_full <= 1'b0;
        end else if (en && load_input_reg) begin
            in_data1_0 <= input1[15:0];
            in_data2_0 <= input2[15:0];
            reg_full <= 1'b1; // Indicate that all registers are full
        end else begin
            reg_full <= 1'b0;
        end
    end

    // Block 2: Addition Logic with LSB Dropped
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            result_0 <= 16'd0;
            channel_index <= 8'd0;
            processing <= 1'b0;
            add_done <= 1'b0;
        end else if (reg_full && !processing) begin
            sum_0 <= in_data1_0 + in_data2_0;
            result_0 <= sum_0[16:1];  // Right shift to drop LSB
            processing <= 1'b1;
        end else if (processing) begin
            channel_index <= channel_index + 1;
            if (channel_index == (N_CHANNELS - 1)) begin
                add_done <= 1'b1;
                processing <= 1'b0;
                channel_index <= 8'd0;
            end else begin
                processing <= 1'b0;
            end
        end else begin
            add_done <= 1'b0;
        end
    end

    // Block 3: Storing Output Data into Output Registers
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            output_data[15:0] <= 16'd0;
        end else if (load_output_reg) begin
            output_data[15:0] <= result_0;
        end
    end
endmodule

module relu_activation_parallel_N_CHANNELS_1 #(
    parameter N_CHANNELS = 1
)(
    input wire clk,
    input wire reset,
    input wire en,
    input wire load_input_reg,
    output reg reg_full,
    input wire load_output_reg,
    output reg activation_done,
    input wire [16*N_CHANNELS-1:0] sram_data_in,
    output reg [16*N_CHANNELS-1:0] sram_data_out
);
    reg [15:0] input_data_0;
    reg [15:0] relu_output_0;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            input_data_0 <= 16'd0;
            reg_full <= 1'b0;
        end else if (en && load_input_reg) begin
            input_data_0 <= sram_data_in[15:0];
            reg_full <= 1'b1;
        end else begin
            reg_full <= 1'b0;
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            relu_output_0 <= 16'd0;
            activation_done <= 1'b0;
        end else if (reg_full) begin
            relu_output_0 <= (input_data_0[15] == 1'b0) ? input_data_0 : 16'd0;
            activation_done <= 1'b1;
        end else begin
            activation_done <= 1'b0;
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            sram_data_out[15:0] <= 16'd0;
        end else if (load_output_reg) begin
            sram_data_out[15:0] <= relu_output_0;
        end
    end
endmodule

module activation_layer_relu #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512
)(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire ready,
    output wire valid_n,
    output wire [DATA_WIDTH-1:0] data_out
);
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire w_buf_en;
    wire [1:0] pooling_control;
    wire load_output_reg;
    wire w_en;
    wire en;
    wire reg_full;
    wire pooling_done;
    wire [DATA_WIDTH-1:0] sram_data_in;

    pooling_controller #(
        .N_CHANNELS(N_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_SIZE(2)
    ) pooling_ctrl_inst (
        .clk(clk),
        .rst(rst),
        .pooling_done(pooling_done),
        .valid(valid),
        .ready_n(ready_n),
        .layer_done(1'b0),
        .reg_full(reg_full),
        .read_address(read_address),
        .write_address(write_address),
        .w_buf_en(w_buf_en),
        .p_en(en),
        .pooling_control(pooling_control),
        .load_output_reg(load_output_reg),
        .w_en(w_en),
        .ready(ready),
        .valid_n(valid_n)
    );

    sram #(
        .N_CHANNELS(N_CHANNELS),
        .DEPTH(DEPTH)
    ) sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_in)
    );

    relu_activation_parallel_N_CHANNELS_1 #(
        .N_CHANNELS(N_CHANNELS)
    ) relu_inst (
        .clk(clk),
        .reset(rst),
        .en(en),
        .load_input_reg(w_buf_en),
        .reg_full(reg_full),
        .load_output_reg(load_output_reg),
        .activation_done(pooling_done),
        .sram_data_in(sram_data_in),
        .sram_data_out(data_out)
    );
endmodule
