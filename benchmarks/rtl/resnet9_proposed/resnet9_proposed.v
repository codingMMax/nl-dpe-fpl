// Auto-generated RESNET9 RTL — DPE 1024×128
// Total DPEs: 19
// Layer tiling:
//   conv1: K=9 N=56 V=1 H=1 → 1 DPEs [ACAM]
//   conv2: K=504 N=112 V=1 H=1 → 1 DPEs [ACAM]
//   conv3: K=1008 N=112 V=1 H=1 → 1 DPEs [ACAM]
//   conv4: K=1008 N=112 V=1 H=1 → 1 DPEs [ACAM]
//   conv5: K=1008 N=224 V=1 H=2 → 2 DPEs [ACAM]
//   conv6: K=2016 N=224 V=2 H=2 → 4 DPEs [V=2]
//   conv7: K=2016 N=224 V=2 H=2 → 4 DPEs [V=2]
//   conv8: K=2016 N=224 V=2 H=2 → 4 DPEs [V=2]
//   conv9: K=224 N=10 V=1 H=1 → 1 DPEs [ACAM]

module resnet9_proposed (
    input wire clk, rst, valid, ready_n,
    input wire [40-1:0] data_in,
    output wire [40-1:0] data_out,
    output wire ready, valid_n
);

    wire [40-1:0] data_out_conv1;
    wire valid_conv1, ready_conv1;
    wire [40-1:0] data_out_conv2;
    wire valid_conv2, ready_conv2;
    wire [40-1:0] data_out_pool1;
    wire valid_pool1, ready_pool1;
    wire [40-1:0] data_out_conv3;
    wire valid_conv3, ready_conv3;
    wire [40-1:0] data_out_conv4;
    wire valid_conv4, ready_conv4;
    wire [40-1:0] data_out_res1;
    wire valid_res1, ready_res1;
    wire [40-1:0] data_out_conv5;
    wire valid_conv5, ready_conv5;
    wire [40-1:0] data_out_pool2;
    wire valid_pool2, ready_pool2;
    wire [40-1:0] data_out_conv6;
    wire valid_conv6, ready_conv6;
    wire [40-1:0] data_out_act_conv6;
    wire valid_act_conv6;
    wire [40-1:0] data_out_pool3;
    wire valid_pool3, ready_pool3;
    wire [40-1:0] data_out_conv7;
    wire valid_conv7, ready_conv7;
    wire [40-1:0] data_out_act_conv7;
    wire valid_act_conv7;
    wire [40-1:0] data_out_conv8;
    wire valid_conv8, ready_conv8;
    wire [40-1:0] data_out_act_conv8;
    wire valid_act_conv8;
    wire [40-1:0] data_out_res2;
    wire valid_res2, ready_res2;
    wire [40-1:0] data_out_pool4;
    wire valid_pool4, ready_pool4;
    wire [40-1:0] data_out_conv9;
    wire valid_conv9, ready_conv9;
    wire valid_g_out, ready_g_in;

    // conv1: V=1 H=1 (single DPE) (ACAM handles activation)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(9), .KERNEL_HEIGHT(1),
        .W(32), .H(32), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) conv1_inst (
        .clk(clk), .rst(rst),
        .valid(valid_g_out), .ready_n(ready_conv1),
        .data_in(data_in),
        .data_out(data_out_conv1),
        .ready(ready_g_in), .valid_n(valid_conv1)
    );

    // conv2: V=1 H=1 (single DPE) (ACAM handles activation)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(504), .KERNEL_HEIGHT(1),
        .W(32), .H(32), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) conv2_inst (
        .clk(clk), .rst(rst),
        .valid(valid_conv1), .ready_n(ready_conv2),
        .data_in(data_out_conv1),
        .data_out(data_out_conv2),
        .ready(ready_conv1), .valid_n(valid_conv2)
    );

    // pool1: max pool 2×2
    pool_mod_pool1 #(.DATA_WIDTH(40)) pool1_inst (
        .clk(clk), .rst(rst),
        .valid(valid_conv2), .ready_n(ready_pool1),
        .data_in(data_out_conv2),
        .data_out(data_out_pool1),
        .ready(ready_conv1), .valid_n(valid_pool1)
    );

    // conv3: V=1 H=1 (single DPE) (ACAM handles activation)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(10),
        .N_KERNELS(1), .KERNEL_WIDTH(1008), .KERNEL_HEIGHT(1),
        .W(16), .H(16), .S(1),
        .DEPTH(1008), .DATA_WIDTH(40)
    ) conv3_inst (
        .clk(clk), .rst(rst),
        .valid(valid_pool1), .ready_n(ready_conv3),
        .data_in(data_out_pool1),
        .data_out(data_out_conv3),
        .ready(ready_pool1), .valid_n(valid_conv3)
    );

    // conv4: V=1 H=1 (single DPE) (ACAM handles activation)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(10),
        .N_KERNELS(1), .KERNEL_WIDTH(1008), .KERNEL_HEIGHT(1),
        .W(16), .H(16), .S(1),
        .DEPTH(1008), .DATA_WIDTH(40)
    ) conv4_inst (
        .clk(clk), .rst(rst),
        .valid(valid_conv3), .ready_n(ready_conv4),
        .data_in(data_out_conv3),
        .data_out(data_out_conv4),
        .ready(ready_pool1), .valid_n(valid_conv4)
    );

    // res1: residual add (skip from pool1)
    res_mod_res1 #(.DATA_WIDTH(40)) res1_inst (
        .clk(clk), .rst(rst),
        .valid(valid_conv4), .ready_n(ready_res1),
        .data_in(data_out_conv4),
        .data_out(data_out_res1),
        .ready(ready_pool1), .valid_n(valid_res1),
        .skip_valid(valid_pool1), .skip_data_in(data_out_pool1)
    );

    // conv5: V=1 H=2 (2 DPEs)
    conv5_layer #(.DATA_WIDTH(40)) conv5_inst (
        .clk(clk), .rst(rst),
        .valid(valid_res1), .ready_n(ready_conv5),
        .data_in(data_out_res1),
        .data_out(data_out_conv5),
        .ready(ready_res1), .valid_n(valid_conv5)
    );

    // pool2: max pool 2×2
    pool_mod_pool2 #(.DATA_WIDTH(40)) pool2_inst (
        .clk(clk), .rst(rst),
        .valid(valid_conv5), .ready_n(ready_pool2),
        .data_in(data_out_conv5),
        .data_out(data_out_pool2),
        .ready(ready_res1), .valid_n(valid_pool2)
    );

    // conv6: V=2 H=2 (4 DPEs) + CLB activation
    conv6_layer #(.DATA_WIDTH(40)) conv6_inst (
        .clk(clk), .rst(rst),
        .valid(valid_pool2), .ready_n(ready_conv6),
        .data_in(data_out_pool2),
        .data_out(data_out_conv6),
        .ready(ready_pool2), .valid_n(valid_conv6)
    );
    // act_conv6: CLB activation (V>1, ACAM cannot absorb)
    activation_lut #(.DATA_WIDTH(40)) act_conv6_inst (
        .clk(clk),
        .data_in(data_out_conv6),
        .data_out(data_out_act_conv6)
    );
    reg valid_act_conv6_r;
    always @(posedge clk) valid_act_conv6_r <= valid_conv6;
    assign valid_act_conv6 = valid_act_conv6_r;

    // pool3: max pool 2×2
    pool_mod_pool3 #(.DATA_WIDTH(40)) pool3_inst (
        .clk(clk), .rst(rst),
        .valid(valid_act_conv6), .ready_n(ready_pool3),
        .data_in(data_out_act_conv6),
        .data_out(data_out_pool3),
        .ready(ready_pool2), .valid_n(valid_pool3)
    );

    // conv7: V=2 H=2 (4 DPEs) + CLB activation
    conv7_layer #(.DATA_WIDTH(40)) conv7_inst (
        .clk(clk), .rst(rst),
        .valid(valid_pool3), .ready_n(ready_conv7),
        .data_in(data_out_pool3),
        .data_out(data_out_conv7),
        .ready(ready_pool3), .valid_n(valid_conv7)
    );
    // act_conv7: CLB activation (V>1, ACAM cannot absorb)
    activation_lut #(.DATA_WIDTH(40)) act_conv7_inst (
        .clk(clk),
        .data_in(data_out_conv7),
        .data_out(data_out_act_conv7)
    );
    reg valid_act_conv7_r;
    always @(posedge clk) valid_act_conv7_r <= valid_conv7;
    assign valid_act_conv7 = valid_act_conv7_r;

    // conv8: V=2 H=2 (4 DPEs) + CLB activation
    conv8_layer #(.DATA_WIDTH(40)) conv8_inst (
        .clk(clk), .rst(rst),
        .valid(valid_act_conv7), .ready_n(ready_conv8),
        .data_in(data_out_act_conv7),
        .data_out(data_out_conv8),
        .ready(ready_pool3), .valid_n(valid_conv8)
    );
    // act_conv8: CLB activation (V>1, ACAM cannot absorb)
    activation_lut #(.DATA_WIDTH(40)) act_conv8_inst (
        .clk(clk),
        .data_in(data_out_conv8),
        .data_out(data_out_act_conv8)
    );
    reg valid_act_conv8_r;
    always @(posedge clk) valid_act_conv8_r <= valid_conv8;
    assign valid_act_conv8 = valid_act_conv8_r;

    // res2: residual add (skip from pool2)
    res_mod_res2 #(.DATA_WIDTH(40)) res2_inst (
        .clk(clk), .rst(rst),
        .valid(valid_act_conv8), .ready_n(ready_res2),
        .data_in(data_out_act_conv8),
        .data_out(data_out_res2),
        .ready(ready_pool3), .valid_n(valid_res2),
        .skip_valid(valid_pool2), .skip_data_in(data_out_pool2)
    );

    // pool4: max pool 4×4
    pool_mod_pool4 #(.DATA_WIDTH(40)) pool4_inst (
        .clk(clk), .rst(rst),
        .valid(valid_res2), .ready_n(ready_pool4),
        .data_in(data_out_res2),
        .data_out(data_out_pool4),
        .ready(ready_res2), .valid_n(valid_pool4)
    );

    // conv9: V=1 H=1 (single DPE)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(224), .KERNEL_HEIGHT(1),
        .W(1), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) conv9_inst (
        .clk(clk), .rst(rst),
        .valid(valid_pool4), .ready_n(ready_conv9),
        .data_in(data_out_pool4),
        .data_out(data_out_conv9),
        .ready(ready_pool4), .valid_n(valid_conv9)
    );

    global_controller #(.N_Layers(1)) g_ctrl (
        .clk(clk), .rst(rst),
        .ready_L1(ready_g_in), .valid_Ln(valid_conv9),
        .valid(valid), .ready(ready),
        .valid_L1(valid_g_out), .ready_Ln(ready_n)
    );

    assign data_out = data_out_conv9;
    assign valid_n = valid_conv9;
endmodule

// ═══════════════════════════════════════════════
// conv5_layer: V=1 H=2 K=1008 N=224
// ═══════════════════════════════════════════════
module conv5_layer #(
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
    localparam H = 2;
    localparam DEPTH = 1008;
    localparam ADDR_WIDTH = 10;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [1-1:0] reg_full_sig;
    wire [2-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [1-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [1-1:0] dpe_sel;
    wire [1-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(1008)
    ) sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_out)
    );

    // Controller
    controller_scalable #(
        .N_CHANNELS(1),
        .N_BRAM_R(1),
        .N_BRAM_W(1),
        .N_DPE_V(1),
        .N_DPE_H(2),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(1008),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1)
    ) ctrl_inst (
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
        .dpe_accum_done(dpe_accum_done),
        .read_address(read_address),
        .write_address(write_address),
        .w_en_dec(w_en),
        .w_buf_en(w_buf_en),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .dpe_sel(dpe_sel),
        .dpe_sel_h(dpe_sel_h),
        .ready(ready),
        .valid_n(valid_n),
        .dpe_accum_ready(dpe_accum_ready)
    );

    wire [DATA_WIDTH-1:0] dpe_out_c0_r0;
    wire dpe_done_c0_r0;
    wire reg_full_c0_r0;
    wire shift_add_done_c0_r0;
    wire shift_add_bypass_ctrl_c0_r0;
    wire MSB_SA_Ready_c0_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r0;
    wire dpe_done_c1_r0;
    wire reg_full_c1_r0;
    wire shift_add_done_c1_r0;
    wire shift_add_bypass_ctrl_c1_r0;
    wire MSB_SA_Ready_c1_r0;

    // DPE instantiations: 1 vertical x 2 horizontal = 2 DPEs
    dpe dpe_c0_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r0),
        .data_out(dpe_out_c0_r0),
        .dpe_done(dpe_done_c0_r0),
        .reg_full(reg_full_c0_r0),
        .shift_add_done(shift_add_done_c0_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r0)
    );

    dpe dpe_c1_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r0),
        .data_out(dpe_out_c1_r0),
        .dpe_done(dpe_done_c1_r0),
        .reg_full(reg_full_c1_r0),
        .shift_add_done(shift_add_done_c1_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r0)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c1_r0;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c1_r0;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c1_r0;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c1_r0;
    assign reg_full_sig = {reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Horizontal output mux (2 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            1'd0: data_out_mux = dpe_out_c0_r0;
            1'd1: data_out_mux = dpe_out_c1_r0;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

// ═══════════════════════════════════════════════
// conv6_layer: V=2 H=2 K=2016 N=224
// ═══════════════════════════════════════════════
module conv6_layer #(
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

    localparam V = 2;
    localparam H = 2;
    localparam DEPTH = 2016;
    localparam ADDR_WIDTH = 11;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [2-1:0] reg_full_sig;
    wire [2-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [2-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [1-1:0] dpe_sel;
    wire [1-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(2016)
    ) sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_out)
    );

    // Controller
    controller_scalable #(
        .N_CHANNELS(1),
        .N_BRAM_R(1),
        .N_BRAM_W(1),
        .N_DPE_V(2),
        .N_DPE_H(2),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2016),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1)
    ) ctrl_inst (
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
        .dpe_accum_done(dpe_accum_done),
        .read_address(read_address),
        .write_address(write_address),
        .w_en_dec(w_en),
        .w_buf_en(w_buf_en),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .dpe_sel(dpe_sel),
        .dpe_sel_h(dpe_sel_h),
        .ready(ready),
        .valid_n(valid_n),
        .dpe_accum_ready(dpe_accum_ready)
    );

    wire [DATA_WIDTH-1:0] dpe_out_c0_r0;
    wire dpe_done_c0_r0;
    wire reg_full_c0_r0;
    wire shift_add_done_c0_r0;
    wire shift_add_bypass_ctrl_c0_r0;
    wire MSB_SA_Ready_c0_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c0_r1;
    wire dpe_done_c0_r1;
    wire reg_full_c0_r1;
    wire shift_add_done_c0_r1;
    wire shift_add_bypass_ctrl_c0_r1;
    wire MSB_SA_Ready_c0_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r0;
    wire dpe_done_c1_r0;
    wire reg_full_c1_r0;
    wire shift_add_done_c1_r0;
    wire shift_add_bypass_ctrl_c1_r0;
    wire MSB_SA_Ready_c1_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r1;
    wire dpe_done_c1_r1;
    wire reg_full_c1_r1;
    wire shift_add_done_c1_r1;
    wire shift_add_bypass_ctrl_c1_r1;
    wire MSB_SA_Ready_c1_r1;

    // DPE instantiations: 2 vertical x 2 horizontal = 4 DPEs
    dpe dpe_c0_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r0),
        .data_out(dpe_out_c0_r0),
        .dpe_done(dpe_done_c0_r0),
        .reg_full(reg_full_c0_r0),
        .shift_add_done(shift_add_done_c0_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r0)
    );

    dpe dpe_c0_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r1),
        .data_out(dpe_out_c0_r1),
        .dpe_done(dpe_done_c0_r1),
        .reg_full(reg_full_c0_r1),
        .shift_add_done(shift_add_done_c0_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r1)
    );

    dpe dpe_c1_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r0),
        .data_out(dpe_out_c1_r0),
        .dpe_done(dpe_done_c1_r0),
        .reg_full(reg_full_c1_r0),
        .shift_add_done(shift_add_done_c1_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r0)
    );

    dpe dpe_c1_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r1),
        .data_out(dpe_out_c1_r1),
        .dpe_done(dpe_done_c1_r1),
        .reg_full(reg_full_c1_r1),
        .shift_add_done(shift_add_done_c1_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r1)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c0_r1 | dpe_done_c1_r0 | dpe_done_c1_r1;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c0_r1 & shift_add_done_c1_r0 & shift_add_done_c1_r1;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c0_r1 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c1_r1;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c0_r1 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c1_r1;
    assign reg_full_sig = {reg_full_c0_r1, reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Column 0 adder tree (V=2)
    wire signed [DATA_WIDTH-1:0] col0_dpe_0;
    assign col0_dpe_0 = dpe_out_c0_r0;
    wire signed [DATA_WIDTH-1:0] col0_dpe_1;
    assign col0_dpe_1 = dpe_out_c0_r1;
    wire [DATA_WIDTH-1:0] col0_sum;
    wire signed [40:0] col0_sum_L0_0;
    assign col0_sum_L0_0 = $signed(col0_dpe_0) + $signed(col0_dpe_1);
    // Truncate adder tree result to 40 bits
    assign col0_sum = col0_sum_L0_0[39:0];

    wire [DATA_WIDTH-1:0] col0_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c0 (
        .clk(clk),
        .data_in(col0_sum),
        .data_out(col0_act)
    );

    // Column 1 adder tree (V=2)
    wire signed [DATA_WIDTH-1:0] col1_dpe_0;
    assign col1_dpe_0 = dpe_out_c1_r0;
    wire signed [DATA_WIDTH-1:0] col1_dpe_1;
    assign col1_dpe_1 = dpe_out_c1_r1;
    wire [DATA_WIDTH-1:0] col1_sum;
    wire signed [40:0] col1_sum_L0_0;
    assign col1_sum_L0_0 = $signed(col1_dpe_0) + $signed(col1_dpe_1);
    // Truncate adder tree result to 40 bits
    assign col1_sum = col1_sum_L0_0[39:0];

    wire [DATA_WIDTH-1:0] col1_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c1 (
        .clk(clk),
        .data_in(col1_sum),
        .data_out(col1_act)
    );

    // Horizontal output mux (2 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            1'd0: data_out_mux = col0_act;
            1'd1: data_out_mux = col1_act;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

// ═══════════════════════════════════════════════
// conv7_layer: V=2 H=2 K=2016 N=224
// ═══════════════════════════════════════════════
module conv7_layer #(
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

    localparam V = 2;
    localparam H = 2;
    localparam DEPTH = 2016;
    localparam ADDR_WIDTH = 11;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [2-1:0] reg_full_sig;
    wire [2-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [2-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [1-1:0] dpe_sel;
    wire [1-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(2016)
    ) sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_out)
    );

    // Controller
    controller_scalable #(
        .N_CHANNELS(1),
        .N_BRAM_R(1),
        .N_BRAM_W(1),
        .N_DPE_V(2),
        .N_DPE_H(2),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2016),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1)
    ) ctrl_inst (
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
        .dpe_accum_done(dpe_accum_done),
        .read_address(read_address),
        .write_address(write_address),
        .w_en_dec(w_en),
        .w_buf_en(w_buf_en),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .dpe_sel(dpe_sel),
        .dpe_sel_h(dpe_sel_h),
        .ready(ready),
        .valid_n(valid_n),
        .dpe_accum_ready(dpe_accum_ready)
    );

    wire [DATA_WIDTH-1:0] dpe_out_c0_r0;
    wire dpe_done_c0_r0;
    wire reg_full_c0_r0;
    wire shift_add_done_c0_r0;
    wire shift_add_bypass_ctrl_c0_r0;
    wire MSB_SA_Ready_c0_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c0_r1;
    wire dpe_done_c0_r1;
    wire reg_full_c0_r1;
    wire shift_add_done_c0_r1;
    wire shift_add_bypass_ctrl_c0_r1;
    wire MSB_SA_Ready_c0_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r0;
    wire dpe_done_c1_r0;
    wire reg_full_c1_r0;
    wire shift_add_done_c1_r0;
    wire shift_add_bypass_ctrl_c1_r0;
    wire MSB_SA_Ready_c1_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r1;
    wire dpe_done_c1_r1;
    wire reg_full_c1_r1;
    wire shift_add_done_c1_r1;
    wire shift_add_bypass_ctrl_c1_r1;
    wire MSB_SA_Ready_c1_r1;

    // DPE instantiations: 2 vertical x 2 horizontal = 4 DPEs
    dpe dpe_c0_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r0),
        .data_out(dpe_out_c0_r0),
        .dpe_done(dpe_done_c0_r0),
        .reg_full(reg_full_c0_r0),
        .shift_add_done(shift_add_done_c0_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r0)
    );

    dpe dpe_c0_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r1),
        .data_out(dpe_out_c0_r1),
        .dpe_done(dpe_done_c0_r1),
        .reg_full(reg_full_c0_r1),
        .shift_add_done(shift_add_done_c0_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r1)
    );

    dpe dpe_c1_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r0),
        .data_out(dpe_out_c1_r0),
        .dpe_done(dpe_done_c1_r0),
        .reg_full(reg_full_c1_r0),
        .shift_add_done(shift_add_done_c1_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r0)
    );

    dpe dpe_c1_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r1),
        .data_out(dpe_out_c1_r1),
        .dpe_done(dpe_done_c1_r1),
        .reg_full(reg_full_c1_r1),
        .shift_add_done(shift_add_done_c1_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r1)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c0_r1 | dpe_done_c1_r0 | dpe_done_c1_r1;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c0_r1 & shift_add_done_c1_r0 & shift_add_done_c1_r1;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c0_r1 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c1_r1;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c0_r1 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c1_r1;
    assign reg_full_sig = {reg_full_c0_r1, reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Column 0 adder tree (V=2)
    wire signed [DATA_WIDTH-1:0] col0_dpe_0;
    assign col0_dpe_0 = dpe_out_c0_r0;
    wire signed [DATA_WIDTH-1:0] col0_dpe_1;
    assign col0_dpe_1 = dpe_out_c0_r1;
    wire [DATA_WIDTH-1:0] col0_sum;
    wire signed [40:0] col0_sum_L0_0;
    assign col0_sum_L0_0 = $signed(col0_dpe_0) + $signed(col0_dpe_1);
    // Truncate adder tree result to 40 bits
    assign col0_sum = col0_sum_L0_0[39:0];

    wire [DATA_WIDTH-1:0] col0_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c0 (
        .clk(clk),
        .data_in(col0_sum),
        .data_out(col0_act)
    );

    // Column 1 adder tree (V=2)
    wire signed [DATA_WIDTH-1:0] col1_dpe_0;
    assign col1_dpe_0 = dpe_out_c1_r0;
    wire signed [DATA_WIDTH-1:0] col1_dpe_1;
    assign col1_dpe_1 = dpe_out_c1_r1;
    wire [DATA_WIDTH-1:0] col1_sum;
    wire signed [40:0] col1_sum_L0_0;
    assign col1_sum_L0_0 = $signed(col1_dpe_0) + $signed(col1_dpe_1);
    // Truncate adder tree result to 40 bits
    assign col1_sum = col1_sum_L0_0[39:0];

    wire [DATA_WIDTH-1:0] col1_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c1 (
        .clk(clk),
        .data_in(col1_sum),
        .data_out(col1_act)
    );

    // Horizontal output mux (2 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            1'd0: data_out_mux = col0_act;
            1'd1: data_out_mux = col1_act;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

// ═══════════════════════════════════════════════
// conv8_layer: V=2 H=2 K=2016 N=224
// ═══════════════════════════════════════════════
module conv8_layer #(
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

    localparam V = 2;
    localparam H = 2;
    localparam DEPTH = 2016;
    localparam ADDR_WIDTH = 11;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [2-1:0] reg_full_sig;
    wire [2-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [2-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [1-1:0] dpe_sel;
    wire [1-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(2016)
    ) sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_out)
    );

    // Controller
    controller_scalable #(
        .N_CHANNELS(1),
        .N_BRAM_R(1),
        .N_BRAM_W(1),
        .N_DPE_V(2),
        .N_DPE_H(2),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2016),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1)
    ) ctrl_inst (
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
        .dpe_accum_done(dpe_accum_done),
        .read_address(read_address),
        .write_address(write_address),
        .w_en_dec(w_en),
        .w_buf_en(w_buf_en),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .dpe_sel(dpe_sel),
        .dpe_sel_h(dpe_sel_h),
        .ready(ready),
        .valid_n(valid_n),
        .dpe_accum_ready(dpe_accum_ready)
    );

    wire [DATA_WIDTH-1:0] dpe_out_c0_r0;
    wire dpe_done_c0_r0;
    wire reg_full_c0_r0;
    wire shift_add_done_c0_r0;
    wire shift_add_bypass_ctrl_c0_r0;
    wire MSB_SA_Ready_c0_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c0_r1;
    wire dpe_done_c0_r1;
    wire reg_full_c0_r1;
    wire shift_add_done_c0_r1;
    wire shift_add_bypass_ctrl_c0_r1;
    wire MSB_SA_Ready_c0_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r0;
    wire dpe_done_c1_r0;
    wire reg_full_c1_r0;
    wire shift_add_done_c1_r0;
    wire shift_add_bypass_ctrl_c1_r0;
    wire MSB_SA_Ready_c1_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r1;
    wire dpe_done_c1_r1;
    wire reg_full_c1_r1;
    wire shift_add_done_c1_r1;
    wire shift_add_bypass_ctrl_c1_r1;
    wire MSB_SA_Ready_c1_r1;

    // DPE instantiations: 2 vertical x 2 horizontal = 4 DPEs
    dpe dpe_c0_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r0),
        .data_out(dpe_out_c0_r0),
        .dpe_done(dpe_done_c0_r0),
        .reg_full(reg_full_c0_r0),
        .shift_add_done(shift_add_done_c0_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r0)
    );

    dpe dpe_c0_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r1),
        .data_out(dpe_out_c0_r1),
        .dpe_done(dpe_done_c0_r1),
        .reg_full(reg_full_c0_r1),
        .shift_add_done(shift_add_done_c0_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r1)
    );

    dpe dpe_c1_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r0),
        .data_out(dpe_out_c1_r0),
        .dpe_done(dpe_done_c1_r0),
        .reg_full(reg_full_c1_r0),
        .shift_add_done(shift_add_done_c1_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r0)
    );

    dpe dpe_c1_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r1),
        .data_out(dpe_out_c1_r1),
        .dpe_done(dpe_done_c1_r1),
        .reg_full(reg_full_c1_r1),
        .shift_add_done(shift_add_done_c1_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r1)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c0_r1 | dpe_done_c1_r0 | dpe_done_c1_r1;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c0_r1 & shift_add_done_c1_r0 & shift_add_done_c1_r1;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c0_r1 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c1_r1;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c0_r1 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c1_r1;
    assign reg_full_sig = {reg_full_c0_r1, reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Column 0 adder tree (V=2)
    wire signed [DATA_WIDTH-1:0] col0_dpe_0;
    assign col0_dpe_0 = dpe_out_c0_r0;
    wire signed [DATA_WIDTH-1:0] col0_dpe_1;
    assign col0_dpe_1 = dpe_out_c0_r1;
    wire [DATA_WIDTH-1:0] col0_sum;
    wire signed [40:0] col0_sum_L0_0;
    assign col0_sum_L0_0 = $signed(col0_dpe_0) + $signed(col0_dpe_1);
    // Truncate adder tree result to 40 bits
    assign col0_sum = col0_sum_L0_0[39:0];

    wire [DATA_WIDTH-1:0] col0_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c0 (
        .clk(clk),
        .data_in(col0_sum),
        .data_out(col0_act)
    );

    // Column 1 adder tree (V=2)
    wire signed [DATA_WIDTH-1:0] col1_dpe_0;
    assign col1_dpe_0 = dpe_out_c1_r0;
    wire signed [DATA_WIDTH-1:0] col1_dpe_1;
    assign col1_dpe_1 = dpe_out_c1_r1;
    wire [DATA_WIDTH-1:0] col1_sum;
    wire signed [40:0] col1_sum_L0_0;
    assign col1_sum_L0_0 = $signed(col1_dpe_0) + $signed(col1_dpe_1);
    // Truncate adder tree result to 40 bits
    assign col1_sum = col1_sum_L0_0[39:0];

    wire [DATA_WIDTH-1:0] col1_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c1 (
        .clk(clk),
        .data_in(col1_sum),
        .data_out(col1_act)
    );

    // Horizontal output mux (2 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            1'd0: data_out_mux = col0_act;
            1'd1: data_out_mux = col1_act;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

module pool_mod_pool1 #(parameter DATA_WIDTH = 40) (
    input wire clk, rst, valid, ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, valid_n
);
    localparam POOL_SIZE = 4;
    reg [DATA_WIDTH-1:0] max_val;
    reg [3-1:0] count;
    reg out_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin max_val <= 0; count <= 0; out_valid <= 0; end
        else if (valid) begin
            if (count == 0) max_val <= data_in;
            else if ($signed(data_in) > $signed(max_val)) max_val <= data_in;
            if (count == POOL_SIZE - 1) begin count <= 0; out_valid <= 1; end
            else begin count <= count + 1; out_valid <= 0; end
        end else out_valid <= 0;
    end
    assign data_out = max_val;
    assign valid_n = out_valid;
    assign ready = 1'b1;
endmodule

module pool_mod_pool2 #(parameter DATA_WIDTH = 40) (
    input wire clk, rst, valid, ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, valid_n
);
    localparam POOL_SIZE = 4;
    reg [DATA_WIDTH-1:0] max_val;
    reg [3-1:0] count;
    reg out_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin max_val <= 0; count <= 0; out_valid <= 0; end
        else if (valid) begin
            if (count == 0) max_val <= data_in;
            else if ($signed(data_in) > $signed(max_val)) max_val <= data_in;
            if (count == POOL_SIZE - 1) begin count <= 0; out_valid <= 1; end
            else begin count <= count + 1; out_valid <= 0; end
        end else out_valid <= 0;
    end
    assign data_out = max_val;
    assign valid_n = out_valid;
    assign ready = 1'b1;
endmodule

module pool_mod_pool3 #(parameter DATA_WIDTH = 40) (
    input wire clk, rst, valid, ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, valid_n
);
    localparam POOL_SIZE = 4;
    reg [DATA_WIDTH-1:0] max_val;
    reg [3-1:0] count;
    reg out_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin max_val <= 0; count <= 0; out_valid <= 0; end
        else if (valid) begin
            if (count == 0) max_val <= data_in;
            else if ($signed(data_in) > $signed(max_val)) max_val <= data_in;
            if (count == POOL_SIZE - 1) begin count <= 0; out_valid <= 1; end
            else begin count <= count + 1; out_valid <= 0; end
        end else out_valid <= 0;
    end
    assign data_out = max_val;
    assign valid_n = out_valid;
    assign ready = 1'b1;
endmodule

module pool_mod_pool4 #(parameter DATA_WIDTH = 40) (
    input wire clk, rst, valid, ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, valid_n
);
    localparam POOL_SIZE = 16;
    reg [DATA_WIDTH-1:0] max_val;
    reg [5-1:0] count;
    reg out_valid;
    always @(posedge clk or posedge rst) begin
        if (rst) begin max_val <= 0; count <= 0; out_valid <= 0; end
        else if (valid) begin
            if (count == 0) max_val <= data_in;
            else if ($signed(data_in) > $signed(max_val)) max_val <= data_in;
            if (count == POOL_SIZE - 1) begin count <= 0; out_valid <= 1; end
            else begin count <= count + 1; out_valid <= 0; end
        end else out_valid <= 0;
    end
    assign data_out = max_val;
    assign valid_n = out_valid;
    assign ready = 1'b1;
endmodule

module res_mod_res1 #(parameter DATA_WIDTH = 40, parameter DEPTH = 512) (
    input wire clk, rst, valid, ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, valid_n,
    input wire skip_valid,
    input wire [DATA_WIDTH-1:0] skip_data_in
);
    reg [DATA_WIDTH-1:0] skip_mem [0:DEPTH-1];
    reg [9:0] w_ptr, r_ptr;
    always @(posedge clk) begin
        if (rst) begin w_ptr <= 0; r_ptr <= 0; end
        else begin
            if (skip_valid) begin skip_mem[w_ptr] <= skip_data_in; w_ptr <= w_ptr + 1; end
            if (valid) r_ptr <= r_ptr + 1;
        end
    end
    assign data_out = $signed(data_in) + $signed(skip_mem[r_ptr]);
    assign valid_n = valid;
    assign ready = 1'b1;
endmodule

module res_mod_res2 #(parameter DATA_WIDTH = 40, parameter DEPTH = 512) (
    input wire clk, rst, valid, ready_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire ready, valid_n,
    input wire skip_valid,
    input wire [DATA_WIDTH-1:0] skip_data_in
);
    reg [DATA_WIDTH-1:0] skip_mem [0:DEPTH-1];
    reg [9:0] w_ptr, r_ptr;
    always @(posedge clk) begin
        if (rst) begin w_ptr <= 0; r_ptr <= 0; end
        else begin
            if (skip_valid) begin skip_mem[w_ptr] <= skip_data_in; w_ptr <= w_ptr + 1; end
            if (valid) r_ptr <= r_ptr + 1;
        end
    end
    assign data_out = $signed(data_in) + $signed(skip_mem[r_ptr]);
    assign valid_n = valid;
    assign ready = 1'b1;
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

// ═══════════════════════════════════════════════
// Supporting modules
// ═══════════════════════════════════════════════
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
    parameter DATA_WIDTH = 40

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


module xbar_ip_module #(
    parameter DATA_WIDTH = 40,
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
