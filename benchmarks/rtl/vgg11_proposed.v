// Auto-generated VGG11 RTL — DPE 1024×128
// Total DPEs: 85
// Layer tiling:
//   conv1: K=9 N=64 V=1 H=1 → 1 DPEs [ACAM]
//   conv2: K=576 N=128 V=1 H=1 → 1 DPEs [ACAM]
//   conv3: K=1152 N=256 V=2 H=2 → 4 DPEs [V=2]
//   conv4: K=2304 N=256 V=3 H=2 → 6 DPEs [V=3]
//   conv5: K=2304 N=512 V=3 H=4 → 12 DPEs [V=3]
//   conv6: K=4608 N=512 V=5 H=4 → 20 DPEs [V=5]
//   conv7: K=4608 N=512 V=5 H=4 → 20 DPEs [V=5]
//   conv8: K=4608 N=512 V=5 H=4 → 20 DPEs [V=5]
//   conv9: K=512 N=10 V=1 H=1 → 1 DPEs [ACAM]

module vgg11_proposed (
    input wire clk, rst, valid, ready_n,
    input wire [40-1:0] data_in,
    output wire [40-1:0] data_out,
    output wire ready, valid_n
);

    wire [40-1:0] data_out_conv1;
    wire valid_conv1, ready_conv1;
    wire [40-1:0] data_out_pool1;
    wire valid_pool1, ready_pool1;
    wire [40-1:0] data_out_conv2;
    wire valid_conv2, ready_conv2;
    wire [40-1:0] data_out_pool2;
    wire valid_pool2, ready_pool2;
    wire [40-1:0] data_out_conv3;
    wire valid_conv3, ready_conv3;
    wire [40-1:0] data_out_act_conv3;
    wire valid_act_conv3;
    wire [40-1:0] data_out_conv4;
    wire valid_conv4, ready_conv4;
    wire [40-1:0] data_out_act_conv4;
    wire valid_act_conv4;
    wire [40-1:0] data_out_pool3;
    wire valid_pool3, ready_pool3;
    wire [40-1:0] data_out_conv5;
    wire valid_conv5, ready_conv5;
    wire [40-1:0] data_out_act_conv5;
    wire valid_act_conv5;
    wire [40-1:0] data_out_conv6;
    wire valid_conv6, ready_conv6;
    wire [40-1:0] data_out_act_conv6;
    wire valid_act_conv6;
    wire [40-1:0] data_out_pool4;
    wire valid_pool4, ready_pool4;
    wire [40-1:0] data_out_conv7;
    wire valid_conv7, ready_conv7;
    wire [40-1:0] data_out_act_conv7;
    wire valid_act_conv7;
    wire [40-1:0] data_out_conv8;
    wire valid_conv8, ready_conv8;
    wire [40-1:0] data_out_act_conv8;
    wire valid_act_conv8;
    wire [40-1:0] data_out_pool5;
    wire valid_pool5, ready_pool5;
    wire [40-1:0] data_out_conv9;
    wire valid_conv9, ready_conv9;
    wire valid_g_out, ready_g_in;

    // conv1: V=1 H=1 (single DPE) (ACAM handles activation)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(9), .KERNEL_HEIGHT(1),
        .W(33), .H(33), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) conv1_inst (
        .clk(clk), .rst(rst),
        .valid(valid_g_out), .ready_n(ready_conv1),
        .data_in(data_in),
        .data_out(data_out_conv1),
        .ready(ready_g_in), .valid_n(valid_conv1)
    );

    // pool1: max pool 2×2
    pool_mod_pool1 #(.DATA_WIDTH(40)) pool1_inst (
        .clk(clk), .rst(rst),
        .valid(valid_conv1), .ready_n(ready_pool1),
        .data_in(data_out_conv1),
        .data_out(data_out_pool1),
        .ready(ready_conv1), .valid_n(valid_pool1)
    );

    // conv2: V=1 H=1 (single DPE) (ACAM handles activation)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(10),
        .N_KERNELS(1), .KERNEL_WIDTH(576), .KERNEL_HEIGHT(1),
        .W(16), .H(16), .S(1),
        .DEPTH(576), .DATA_WIDTH(40)
    ) conv2_inst (
        .clk(clk), .rst(rst),
        .valid(valid_pool1), .ready_n(ready_conv2),
        .data_in(data_out_pool1),
        .data_out(data_out_conv2),
        .ready(ready_pool1), .valid_n(valid_conv2)
    );

    // pool2: max pool 2×2
    pool_mod_pool2 #(.DATA_WIDTH(40)) pool2_inst (
        .clk(clk), .rst(rst),
        .valid(valid_conv2), .ready_n(ready_pool2),
        .data_in(data_out_conv2),
        .data_out(data_out_pool2),
        .ready(ready_pool1), .valid_n(valid_pool2)
    );

    // conv3: V=2 H=2 (4 DPEs) + CLB activation
    conv3_layer #(.DATA_WIDTH(40)) conv3_inst (
        .clk(clk), .rst(rst),
        .valid(valid_pool2), .ready_n(ready_conv3),
        .data_in(data_out_pool2),
        .data_out(data_out_conv3),
        .ready(ready_pool2), .valid_n(valid_conv3)
    );
    // act_conv3: CLB activation (V>1, ACAM cannot absorb)
    activation_lut #(.DATA_WIDTH(40)) act_conv3_inst (
        .clk(clk),
        .data_in(data_out_conv3),
        .data_out(data_out_act_conv3)
    );
    reg valid_act_conv3_r;
    always @(posedge clk) valid_act_conv3_r <= valid_conv3;
    assign valid_act_conv3 = valid_act_conv3_r;

    // conv4: V=3 H=2 (6 DPEs) + CLB activation
    conv4_layer #(.DATA_WIDTH(40)) conv4_inst (
        .clk(clk), .rst(rst),
        .valid(valid_act_conv3), .ready_n(ready_conv4),
        .data_in(data_out_act_conv3),
        .data_out(data_out_conv4),
        .ready(ready_pool2), .valid_n(valid_conv4)
    );
    // act_conv4: CLB activation (V>1, ACAM cannot absorb)
    activation_lut #(.DATA_WIDTH(40)) act_conv4_inst (
        .clk(clk),
        .data_in(data_out_conv4),
        .data_out(data_out_act_conv4)
    );
    reg valid_act_conv4_r;
    always @(posedge clk) valid_act_conv4_r <= valid_conv4;
    assign valid_act_conv4 = valid_act_conv4_r;

    // pool3: max pool 2×2
    pool_mod_pool3 #(.DATA_WIDTH(40)) pool3_inst (
        .clk(clk), .rst(rst),
        .valid(valid_act_conv4), .ready_n(ready_pool3),
        .data_in(data_out_act_conv4),
        .data_out(data_out_pool3),
        .ready(ready_pool2), .valid_n(valid_pool3)
    );

    // conv5: V=3 H=4 (12 DPEs) + CLB activation
    conv5_layer #(.DATA_WIDTH(40)) conv5_inst (
        .clk(clk), .rst(rst),
        .valid(valid_pool3), .ready_n(ready_conv5),
        .data_in(data_out_pool3),
        .data_out(data_out_conv5),
        .ready(ready_pool3), .valid_n(valid_conv5)
    );
    // act_conv5: CLB activation (V>1, ACAM cannot absorb)
    activation_lut #(.DATA_WIDTH(40)) act_conv5_inst (
        .clk(clk),
        .data_in(data_out_conv5),
        .data_out(data_out_act_conv5)
    );
    reg valid_act_conv5_r;
    always @(posedge clk) valid_act_conv5_r <= valid_conv5;
    assign valid_act_conv5 = valid_act_conv5_r;

    // conv6: V=5 H=4 (20 DPEs) + CLB activation
    conv6_layer #(.DATA_WIDTH(40)) conv6_inst (
        .clk(clk), .rst(rst),
        .valid(valid_act_conv5), .ready_n(ready_conv6),
        .data_in(data_out_act_conv5),
        .data_out(data_out_conv6),
        .ready(ready_pool3), .valid_n(valid_conv6)
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

    // pool4: max pool 2×2
    pool_mod_pool4 #(.DATA_WIDTH(40)) pool4_inst (
        .clk(clk), .rst(rst),
        .valid(valid_act_conv6), .ready_n(ready_pool4),
        .data_in(data_out_act_conv6),
        .data_out(data_out_pool4),
        .ready(ready_pool3), .valid_n(valid_pool4)
    );

    // conv7: V=5 H=4 (20 DPEs) + CLB activation
    conv7_layer #(.DATA_WIDTH(40)) conv7_inst (
        .clk(clk), .rst(rst),
        .valid(valid_pool4), .ready_n(ready_conv7),
        .data_in(data_out_pool4),
        .data_out(data_out_conv7),
        .ready(ready_pool4), .valid_n(valid_conv7)
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

    // conv8: V=5 H=4 (20 DPEs) + CLB activation
    conv8_layer #(.DATA_WIDTH(40)) conv8_inst (
        .clk(clk), .rst(rst),
        .valid(valid_act_conv7), .ready_n(ready_conv8),
        .data_in(data_out_act_conv7),
        .data_out(data_out_conv8),
        .ready(ready_pool4), .valid_n(valid_conv8)
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

    // pool5: max pool 2×2
    pool_mod_pool5 #(.DATA_WIDTH(40)) pool5_inst (
        .clk(clk), .rst(rst),
        .valid(valid_act_conv8), .ready_n(ready_pool5),
        .data_in(data_out_act_conv8),
        .data_out(data_out_pool5),
        .ready(ready_pool4), .valid_n(valid_pool5)
    );

    // conv9: V=1 H=1 (single DPE)
    conv_layer_single_dpe #(
        .N_CHANNELS(1), .ADDR_WIDTH(9),
        .N_KERNELS(1), .KERNEL_WIDTH(512), .KERNEL_HEIGHT(1),
        .W(1), .H(1), .S(1),
        .DEPTH(512), .DATA_WIDTH(40)
    ) conv9_inst (
        .clk(clk), .rst(rst),
        .valid(valid_pool5), .ready_n(ready_conv9),
        .data_in(data_out_pool5),
        .data_out(data_out_conv9),
        .ready(ready_pool5), .valid_n(valid_conv9)
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
// conv3_layer: V=2 H=2 K=1152 N=256
// ═══════════════════════════════════════════════
module conv3_layer #(
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
    localparam DEPTH = 1152;
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
        .DEPTH(1152)
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
        .KERNEL_WIDTH(1152),
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
// conv4_layer: V=3 H=2 K=2304 N=256
// ═══════════════════════════════════════════════
module conv4_layer #(
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

    localparam V = 3;
    localparam H = 2;
    localparam DEPTH = 2304;
    localparam ADDR_WIDTH = 12;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [3-1:0] reg_full_sig;
    wire [2-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [3-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [2-1:0] dpe_sel;
    wire [1-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(2304)
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
        .N_DPE_V(3),
        .N_DPE_H(2),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2304),
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
    wire [DATA_WIDTH-1:0] dpe_out_c0_r2;
    wire dpe_done_c0_r2;
    wire reg_full_c0_r2;
    wire shift_add_done_c0_r2;
    wire shift_add_bypass_ctrl_c0_r2;
    wire MSB_SA_Ready_c0_r2;
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
    wire [DATA_WIDTH-1:0] dpe_out_c1_r2;
    wire dpe_done_c1_r2;
    wire reg_full_c1_r2;
    wire shift_add_done_c1_r2;
    wire shift_add_bypass_ctrl_c1_r2;
    wire MSB_SA_Ready_c1_r2;

    // DPE instantiations: 3 vertical x 2 horizontal = 6 DPEs
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

    dpe dpe_c0_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r2),
        .data_out(dpe_out_c0_r2),
        .dpe_done(dpe_done_c0_r2),
        .reg_full(reg_full_c0_r2),
        .shift_add_done(shift_add_done_c0_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r2)
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

    dpe dpe_c1_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r2),
        .data_out(dpe_out_c1_r2),
        .dpe_done(dpe_done_c1_r2),
        .reg_full(reg_full_c1_r2),
        .shift_add_done(shift_add_done_c1_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r2)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c0_r1 | dpe_done_c0_r2 | dpe_done_c1_r0 | dpe_done_c1_r1 | dpe_done_c1_r2;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c0_r1 & shift_add_done_c0_r2 & shift_add_done_c1_r0 & shift_add_done_c1_r1 & shift_add_done_c1_r2;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c0_r1 & shift_add_bypass_ctrl_c0_r2 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c1_r1 & shift_add_bypass_ctrl_c1_r2;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c0_r1 & MSB_SA_Ready_c0_r2 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c1_r1 & MSB_SA_Ready_c1_r2;
    assign reg_full_sig = {reg_full_c0_r2, reg_full_c0_r1, reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Column 0 adder tree (V=3)
    wire signed [DATA_WIDTH-1:0] col0_dpe_0;
    assign col0_dpe_0 = dpe_out_c0_r0;
    wire signed [DATA_WIDTH-1:0] col0_dpe_1;
    assign col0_dpe_1 = dpe_out_c0_r1;
    wire signed [DATA_WIDTH-1:0] col0_dpe_2;
    assign col0_dpe_2 = dpe_out_c0_r2;
    wire [DATA_WIDTH-1:0] col0_sum;
    wire signed [40:0] col0_sum_L0_0;
    assign col0_sum_L0_0 = $signed(col0_dpe_0) + $signed(col0_dpe_1);
    wire signed [40:0] col0_pass_L0;
    assign col0_pass_L0 = {{col0_dpe_2[39]}, col0_dpe_2};
    wire signed [41:0] col0_sum_L1_0;
    assign col0_sum_L1_0 = $signed(col0_sum_L0_0) + $signed(col0_pass_L0);
    // Truncate adder tree result to 40 bits
    assign col0_sum = col0_sum_L1_0[39:0];

    wire [DATA_WIDTH-1:0] col0_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c0 (
        .clk(clk),
        .data_in(col0_sum),
        .data_out(col0_act)
    );

    // Column 1 adder tree (V=3)
    wire signed [DATA_WIDTH-1:0] col1_dpe_0;
    assign col1_dpe_0 = dpe_out_c1_r0;
    wire signed [DATA_WIDTH-1:0] col1_dpe_1;
    assign col1_dpe_1 = dpe_out_c1_r1;
    wire signed [DATA_WIDTH-1:0] col1_dpe_2;
    assign col1_dpe_2 = dpe_out_c1_r2;
    wire [DATA_WIDTH-1:0] col1_sum;
    wire signed [40:0] col1_sum_L0_0;
    assign col1_sum_L0_0 = $signed(col1_dpe_0) + $signed(col1_dpe_1);
    wire signed [40:0] col1_pass_L0;
    assign col1_pass_L0 = {{col1_dpe_2[39]}, col1_dpe_2};
    wire signed [41:0] col1_sum_L1_0;
    assign col1_sum_L1_0 = $signed(col1_sum_L0_0) + $signed(col1_pass_L0);
    // Truncate adder tree result to 40 bits
    assign col1_sum = col1_sum_L1_0[39:0];

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
// conv5_layer: V=3 H=4 K=2304 N=512
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

    localparam V = 3;
    localparam H = 4;
    localparam DEPTH = 2304;
    localparam ADDR_WIDTH = 12;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [3-1:0] reg_full_sig;
    wire [4-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [3-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [2-1:0] dpe_sel;
    wire [2-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(2304)
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
        .N_DPE_V(3),
        .N_DPE_H(4),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(2304),
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
    wire [DATA_WIDTH-1:0] dpe_out_c0_r2;
    wire dpe_done_c0_r2;
    wire reg_full_c0_r2;
    wire shift_add_done_c0_r2;
    wire shift_add_bypass_ctrl_c0_r2;
    wire MSB_SA_Ready_c0_r2;
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
    wire [DATA_WIDTH-1:0] dpe_out_c1_r2;
    wire dpe_done_c1_r2;
    wire reg_full_c1_r2;
    wire shift_add_done_c1_r2;
    wire shift_add_bypass_ctrl_c1_r2;
    wire MSB_SA_Ready_c1_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r0;
    wire dpe_done_c2_r0;
    wire reg_full_c2_r0;
    wire shift_add_done_c2_r0;
    wire shift_add_bypass_ctrl_c2_r0;
    wire MSB_SA_Ready_c2_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r1;
    wire dpe_done_c2_r1;
    wire reg_full_c2_r1;
    wire shift_add_done_c2_r1;
    wire shift_add_bypass_ctrl_c2_r1;
    wire MSB_SA_Ready_c2_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r2;
    wire dpe_done_c2_r2;
    wire reg_full_c2_r2;
    wire shift_add_done_c2_r2;
    wire shift_add_bypass_ctrl_c2_r2;
    wire MSB_SA_Ready_c2_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r0;
    wire dpe_done_c3_r0;
    wire reg_full_c3_r0;
    wire shift_add_done_c3_r0;
    wire shift_add_bypass_ctrl_c3_r0;
    wire MSB_SA_Ready_c3_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r1;
    wire dpe_done_c3_r1;
    wire reg_full_c3_r1;
    wire shift_add_done_c3_r1;
    wire shift_add_bypass_ctrl_c3_r1;
    wire MSB_SA_Ready_c3_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r2;
    wire dpe_done_c3_r2;
    wire reg_full_c3_r2;
    wire shift_add_done_c3_r2;
    wire shift_add_bypass_ctrl_c3_r2;
    wire MSB_SA_Ready_c3_r2;

    // DPE instantiations: 3 vertical x 4 horizontal = 12 DPEs
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

    dpe dpe_c0_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r2),
        .data_out(dpe_out_c0_r2),
        .dpe_done(dpe_done_c0_r2),
        .reg_full(reg_full_c0_r2),
        .shift_add_done(shift_add_done_c0_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r2)
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

    dpe dpe_c1_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r2),
        .data_out(dpe_out_c1_r2),
        .dpe_done(dpe_done_c1_r2),
        .reg_full(reg_full_c1_r2),
        .shift_add_done(shift_add_done_c1_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r2)
    );

    dpe dpe_c2_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r0),
        .data_out(dpe_out_c2_r0),
        .dpe_done(dpe_done_c2_r0),
        .reg_full(reg_full_c2_r0),
        .shift_add_done(shift_add_done_c2_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r0)
    );

    dpe dpe_c2_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r1),
        .data_out(dpe_out_c2_r1),
        .dpe_done(dpe_done_c2_r1),
        .reg_full(reg_full_c2_r1),
        .shift_add_done(shift_add_done_c2_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r1)
    );

    dpe dpe_c2_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r2),
        .data_out(dpe_out_c2_r2),
        .dpe_done(dpe_done_c2_r2),
        .reg_full(reg_full_c2_r2),
        .shift_add_done(shift_add_done_c2_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r2)
    );

    dpe dpe_c3_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r0),
        .data_out(dpe_out_c3_r0),
        .dpe_done(dpe_done_c3_r0),
        .reg_full(reg_full_c3_r0),
        .shift_add_done(shift_add_done_c3_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r0)
    );

    dpe dpe_c3_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r1),
        .data_out(dpe_out_c3_r1),
        .dpe_done(dpe_done_c3_r1),
        .reg_full(reg_full_c3_r1),
        .shift_add_done(shift_add_done_c3_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r1)
    );

    dpe dpe_c3_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r2),
        .data_out(dpe_out_c3_r2),
        .dpe_done(dpe_done_c3_r2),
        .reg_full(reg_full_c3_r2),
        .shift_add_done(shift_add_done_c3_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r2)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c0_r1 | dpe_done_c0_r2 | dpe_done_c1_r0 | dpe_done_c1_r1 | dpe_done_c1_r2 | dpe_done_c2_r0 | dpe_done_c2_r1 | dpe_done_c2_r2 | dpe_done_c3_r0 | dpe_done_c3_r1 | dpe_done_c3_r2;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c0_r1 & shift_add_done_c0_r2 & shift_add_done_c1_r0 & shift_add_done_c1_r1 & shift_add_done_c1_r2 & shift_add_done_c2_r0 & shift_add_done_c2_r1 & shift_add_done_c2_r2 & shift_add_done_c3_r0 & shift_add_done_c3_r1 & shift_add_done_c3_r2;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c0_r1 & shift_add_bypass_ctrl_c0_r2 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c1_r1 & shift_add_bypass_ctrl_c1_r2 & shift_add_bypass_ctrl_c2_r0 & shift_add_bypass_ctrl_c2_r1 & shift_add_bypass_ctrl_c2_r2 & shift_add_bypass_ctrl_c3_r0 & shift_add_bypass_ctrl_c3_r1 & shift_add_bypass_ctrl_c3_r2;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c0_r1 & MSB_SA_Ready_c0_r2 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c1_r1 & MSB_SA_Ready_c1_r2 & MSB_SA_Ready_c2_r0 & MSB_SA_Ready_c2_r1 & MSB_SA_Ready_c2_r2 & MSB_SA_Ready_c3_r0 & MSB_SA_Ready_c3_r1 & MSB_SA_Ready_c3_r2;
    assign reg_full_sig = {reg_full_c0_r2, reg_full_c0_r1, reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0, 1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Column 0 adder tree (V=3)
    wire signed [DATA_WIDTH-1:0] col0_dpe_0;
    assign col0_dpe_0 = dpe_out_c0_r0;
    wire signed [DATA_WIDTH-1:0] col0_dpe_1;
    assign col0_dpe_1 = dpe_out_c0_r1;
    wire signed [DATA_WIDTH-1:0] col0_dpe_2;
    assign col0_dpe_2 = dpe_out_c0_r2;
    wire [DATA_WIDTH-1:0] col0_sum;
    wire signed [40:0] col0_sum_L0_0;
    assign col0_sum_L0_0 = $signed(col0_dpe_0) + $signed(col0_dpe_1);
    wire signed [40:0] col0_pass_L0;
    assign col0_pass_L0 = {{col0_dpe_2[39]}, col0_dpe_2};
    wire signed [41:0] col0_sum_L1_0;
    assign col0_sum_L1_0 = $signed(col0_sum_L0_0) + $signed(col0_pass_L0);
    // Truncate adder tree result to 40 bits
    assign col0_sum = col0_sum_L1_0[39:0];

    wire [DATA_WIDTH-1:0] col0_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c0 (
        .clk(clk),
        .data_in(col0_sum),
        .data_out(col0_act)
    );

    // Column 1 adder tree (V=3)
    wire signed [DATA_WIDTH-1:0] col1_dpe_0;
    assign col1_dpe_0 = dpe_out_c1_r0;
    wire signed [DATA_WIDTH-1:0] col1_dpe_1;
    assign col1_dpe_1 = dpe_out_c1_r1;
    wire signed [DATA_WIDTH-1:0] col1_dpe_2;
    assign col1_dpe_2 = dpe_out_c1_r2;
    wire [DATA_WIDTH-1:0] col1_sum;
    wire signed [40:0] col1_sum_L0_0;
    assign col1_sum_L0_0 = $signed(col1_dpe_0) + $signed(col1_dpe_1);
    wire signed [40:0] col1_pass_L0;
    assign col1_pass_L0 = {{col1_dpe_2[39]}, col1_dpe_2};
    wire signed [41:0] col1_sum_L1_0;
    assign col1_sum_L1_0 = $signed(col1_sum_L0_0) + $signed(col1_pass_L0);
    // Truncate adder tree result to 40 bits
    assign col1_sum = col1_sum_L1_0[39:0];

    wire [DATA_WIDTH-1:0] col1_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c1 (
        .clk(clk),
        .data_in(col1_sum),
        .data_out(col1_act)
    );

    // Column 2 adder tree (V=3)
    wire signed [DATA_WIDTH-1:0] col2_dpe_0;
    assign col2_dpe_0 = dpe_out_c2_r0;
    wire signed [DATA_WIDTH-1:0] col2_dpe_1;
    assign col2_dpe_1 = dpe_out_c2_r1;
    wire signed [DATA_WIDTH-1:0] col2_dpe_2;
    assign col2_dpe_2 = dpe_out_c2_r2;
    wire [DATA_WIDTH-1:0] col2_sum;
    wire signed [40:0] col2_sum_L0_0;
    assign col2_sum_L0_0 = $signed(col2_dpe_0) + $signed(col2_dpe_1);
    wire signed [40:0] col2_pass_L0;
    assign col2_pass_L0 = {{col2_dpe_2[39]}, col2_dpe_2};
    wire signed [41:0] col2_sum_L1_0;
    assign col2_sum_L1_0 = $signed(col2_sum_L0_0) + $signed(col2_pass_L0);
    // Truncate adder tree result to 40 bits
    assign col2_sum = col2_sum_L1_0[39:0];

    wire [DATA_WIDTH-1:0] col2_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c2 (
        .clk(clk),
        .data_in(col2_sum),
        .data_out(col2_act)
    );

    // Column 3 adder tree (V=3)
    wire signed [DATA_WIDTH-1:0] col3_dpe_0;
    assign col3_dpe_0 = dpe_out_c3_r0;
    wire signed [DATA_WIDTH-1:0] col3_dpe_1;
    assign col3_dpe_1 = dpe_out_c3_r1;
    wire signed [DATA_WIDTH-1:0] col3_dpe_2;
    assign col3_dpe_2 = dpe_out_c3_r2;
    wire [DATA_WIDTH-1:0] col3_sum;
    wire signed [40:0] col3_sum_L0_0;
    assign col3_sum_L0_0 = $signed(col3_dpe_0) + $signed(col3_dpe_1);
    wire signed [40:0] col3_pass_L0;
    assign col3_pass_L0 = {{col3_dpe_2[39]}, col3_dpe_2};
    wire signed [41:0] col3_sum_L1_0;
    assign col3_sum_L1_0 = $signed(col3_sum_L0_0) + $signed(col3_pass_L0);
    // Truncate adder tree result to 40 bits
    assign col3_sum = col3_sum_L1_0[39:0];

    wire [DATA_WIDTH-1:0] col3_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c3 (
        .clk(clk),
        .data_in(col3_sum),
        .data_out(col3_act)
    );

    // Horizontal output mux (4 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            2'd0: data_out_mux = col0_act;
            2'd1: data_out_mux = col1_act;
            2'd2: data_out_mux = col2_act;
            2'd3: data_out_mux = col3_act;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

// ═══════════════════════════════════════════════
// conv6_layer: V=5 H=4 K=4608 N=512
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

    localparam V = 5;
    localparam H = 4;
    localparam DEPTH = 4608;
    localparam ADDR_WIDTH = 13;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [5-1:0] reg_full_sig;
    wire [4-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [5-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [3-1:0] dpe_sel;
    wire [2-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(4608)
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
        .N_DPE_V(5),
        .N_DPE_H(4),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(4608),
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
    wire [DATA_WIDTH-1:0] dpe_out_c0_r2;
    wire dpe_done_c0_r2;
    wire reg_full_c0_r2;
    wire shift_add_done_c0_r2;
    wire shift_add_bypass_ctrl_c0_r2;
    wire MSB_SA_Ready_c0_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c0_r3;
    wire dpe_done_c0_r3;
    wire reg_full_c0_r3;
    wire shift_add_done_c0_r3;
    wire shift_add_bypass_ctrl_c0_r3;
    wire MSB_SA_Ready_c0_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c0_r4;
    wire dpe_done_c0_r4;
    wire reg_full_c0_r4;
    wire shift_add_done_c0_r4;
    wire shift_add_bypass_ctrl_c0_r4;
    wire MSB_SA_Ready_c0_r4;
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
    wire [DATA_WIDTH-1:0] dpe_out_c1_r2;
    wire dpe_done_c1_r2;
    wire reg_full_c1_r2;
    wire shift_add_done_c1_r2;
    wire shift_add_bypass_ctrl_c1_r2;
    wire MSB_SA_Ready_c1_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r3;
    wire dpe_done_c1_r3;
    wire reg_full_c1_r3;
    wire shift_add_done_c1_r3;
    wire shift_add_bypass_ctrl_c1_r3;
    wire MSB_SA_Ready_c1_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r4;
    wire dpe_done_c1_r4;
    wire reg_full_c1_r4;
    wire shift_add_done_c1_r4;
    wire shift_add_bypass_ctrl_c1_r4;
    wire MSB_SA_Ready_c1_r4;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r0;
    wire dpe_done_c2_r0;
    wire reg_full_c2_r0;
    wire shift_add_done_c2_r0;
    wire shift_add_bypass_ctrl_c2_r0;
    wire MSB_SA_Ready_c2_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r1;
    wire dpe_done_c2_r1;
    wire reg_full_c2_r1;
    wire shift_add_done_c2_r1;
    wire shift_add_bypass_ctrl_c2_r1;
    wire MSB_SA_Ready_c2_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r2;
    wire dpe_done_c2_r2;
    wire reg_full_c2_r2;
    wire shift_add_done_c2_r2;
    wire shift_add_bypass_ctrl_c2_r2;
    wire MSB_SA_Ready_c2_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r3;
    wire dpe_done_c2_r3;
    wire reg_full_c2_r3;
    wire shift_add_done_c2_r3;
    wire shift_add_bypass_ctrl_c2_r3;
    wire MSB_SA_Ready_c2_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r4;
    wire dpe_done_c2_r4;
    wire reg_full_c2_r4;
    wire shift_add_done_c2_r4;
    wire shift_add_bypass_ctrl_c2_r4;
    wire MSB_SA_Ready_c2_r4;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r0;
    wire dpe_done_c3_r0;
    wire reg_full_c3_r0;
    wire shift_add_done_c3_r0;
    wire shift_add_bypass_ctrl_c3_r0;
    wire MSB_SA_Ready_c3_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r1;
    wire dpe_done_c3_r1;
    wire reg_full_c3_r1;
    wire shift_add_done_c3_r1;
    wire shift_add_bypass_ctrl_c3_r1;
    wire MSB_SA_Ready_c3_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r2;
    wire dpe_done_c3_r2;
    wire reg_full_c3_r2;
    wire shift_add_done_c3_r2;
    wire shift_add_bypass_ctrl_c3_r2;
    wire MSB_SA_Ready_c3_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r3;
    wire dpe_done_c3_r3;
    wire reg_full_c3_r3;
    wire shift_add_done_c3_r3;
    wire shift_add_bypass_ctrl_c3_r3;
    wire MSB_SA_Ready_c3_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r4;
    wire dpe_done_c3_r4;
    wire reg_full_c3_r4;
    wire shift_add_done_c3_r4;
    wire shift_add_bypass_ctrl_c3_r4;
    wire MSB_SA_Ready_c3_r4;

    // DPE instantiations: 5 vertical x 4 horizontal = 20 DPEs
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

    dpe dpe_c0_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r2),
        .data_out(dpe_out_c0_r2),
        .dpe_done(dpe_done_c0_r2),
        .reg_full(reg_full_c0_r2),
        .shift_add_done(shift_add_done_c0_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r2)
    );

    dpe dpe_c0_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r3),
        .data_out(dpe_out_c0_r3),
        .dpe_done(dpe_done_c0_r3),
        .reg_full(reg_full_c0_r3),
        .shift_add_done(shift_add_done_c0_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r3)
    );

    dpe dpe_c0_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r4),
        .data_out(dpe_out_c0_r4),
        .dpe_done(dpe_done_c0_r4),
        .reg_full(reg_full_c0_r4),
        .shift_add_done(shift_add_done_c0_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r4)
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

    dpe dpe_c1_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r2),
        .data_out(dpe_out_c1_r2),
        .dpe_done(dpe_done_c1_r2),
        .reg_full(reg_full_c1_r2),
        .shift_add_done(shift_add_done_c1_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r2)
    );

    dpe dpe_c1_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r3),
        .data_out(dpe_out_c1_r3),
        .dpe_done(dpe_done_c1_r3),
        .reg_full(reg_full_c1_r3),
        .shift_add_done(shift_add_done_c1_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r3)
    );

    dpe dpe_c1_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r4),
        .data_out(dpe_out_c1_r4),
        .dpe_done(dpe_done_c1_r4),
        .reg_full(reg_full_c1_r4),
        .shift_add_done(shift_add_done_c1_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r4)
    );

    dpe dpe_c2_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r0),
        .data_out(dpe_out_c2_r0),
        .dpe_done(dpe_done_c2_r0),
        .reg_full(reg_full_c2_r0),
        .shift_add_done(shift_add_done_c2_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r0)
    );

    dpe dpe_c2_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r1),
        .data_out(dpe_out_c2_r1),
        .dpe_done(dpe_done_c2_r1),
        .reg_full(reg_full_c2_r1),
        .shift_add_done(shift_add_done_c2_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r1)
    );

    dpe dpe_c2_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r2),
        .data_out(dpe_out_c2_r2),
        .dpe_done(dpe_done_c2_r2),
        .reg_full(reg_full_c2_r2),
        .shift_add_done(shift_add_done_c2_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r2)
    );

    dpe dpe_c2_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r3),
        .data_out(dpe_out_c2_r3),
        .dpe_done(dpe_done_c2_r3),
        .reg_full(reg_full_c2_r3),
        .shift_add_done(shift_add_done_c2_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r3)
    );

    dpe dpe_c2_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r4),
        .data_out(dpe_out_c2_r4),
        .dpe_done(dpe_done_c2_r4),
        .reg_full(reg_full_c2_r4),
        .shift_add_done(shift_add_done_c2_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r4)
    );

    dpe dpe_c3_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r0),
        .data_out(dpe_out_c3_r0),
        .dpe_done(dpe_done_c3_r0),
        .reg_full(reg_full_c3_r0),
        .shift_add_done(shift_add_done_c3_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r0)
    );

    dpe dpe_c3_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r1),
        .data_out(dpe_out_c3_r1),
        .dpe_done(dpe_done_c3_r1),
        .reg_full(reg_full_c3_r1),
        .shift_add_done(shift_add_done_c3_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r1)
    );

    dpe dpe_c3_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r2),
        .data_out(dpe_out_c3_r2),
        .dpe_done(dpe_done_c3_r2),
        .reg_full(reg_full_c3_r2),
        .shift_add_done(shift_add_done_c3_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r2)
    );

    dpe dpe_c3_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r3),
        .data_out(dpe_out_c3_r3),
        .dpe_done(dpe_done_c3_r3),
        .reg_full(reg_full_c3_r3),
        .shift_add_done(shift_add_done_c3_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r3)
    );

    dpe dpe_c3_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r4),
        .data_out(dpe_out_c3_r4),
        .dpe_done(dpe_done_c3_r4),
        .reg_full(reg_full_c3_r4),
        .shift_add_done(shift_add_done_c3_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r4)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c0_r1 | dpe_done_c0_r2 | dpe_done_c0_r3 | dpe_done_c0_r4 | dpe_done_c1_r0 | dpe_done_c1_r1 | dpe_done_c1_r2 | dpe_done_c1_r3 | dpe_done_c1_r4 | dpe_done_c2_r0 | dpe_done_c2_r1 | dpe_done_c2_r2 | dpe_done_c2_r3 | dpe_done_c2_r4 | dpe_done_c3_r0 | dpe_done_c3_r1 | dpe_done_c3_r2 | dpe_done_c3_r3 | dpe_done_c3_r4;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c0_r1 & shift_add_done_c0_r2 & shift_add_done_c0_r3 & shift_add_done_c0_r4 & shift_add_done_c1_r0 & shift_add_done_c1_r1 & shift_add_done_c1_r2 & shift_add_done_c1_r3 & shift_add_done_c1_r4 & shift_add_done_c2_r0 & shift_add_done_c2_r1 & shift_add_done_c2_r2 & shift_add_done_c2_r3 & shift_add_done_c2_r4 & shift_add_done_c3_r0 & shift_add_done_c3_r1 & shift_add_done_c3_r2 & shift_add_done_c3_r3 & shift_add_done_c3_r4;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c0_r1 & shift_add_bypass_ctrl_c0_r2 & shift_add_bypass_ctrl_c0_r3 & shift_add_bypass_ctrl_c0_r4 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c1_r1 & shift_add_bypass_ctrl_c1_r2 & shift_add_bypass_ctrl_c1_r3 & shift_add_bypass_ctrl_c1_r4 & shift_add_bypass_ctrl_c2_r0 & shift_add_bypass_ctrl_c2_r1 & shift_add_bypass_ctrl_c2_r2 & shift_add_bypass_ctrl_c2_r3 & shift_add_bypass_ctrl_c2_r4 & shift_add_bypass_ctrl_c3_r0 & shift_add_bypass_ctrl_c3_r1 & shift_add_bypass_ctrl_c3_r2 & shift_add_bypass_ctrl_c3_r3 & shift_add_bypass_ctrl_c3_r4;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c0_r1 & MSB_SA_Ready_c0_r2 & MSB_SA_Ready_c0_r3 & MSB_SA_Ready_c0_r4 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c1_r1 & MSB_SA_Ready_c1_r2 & MSB_SA_Ready_c1_r3 & MSB_SA_Ready_c1_r4 & MSB_SA_Ready_c2_r0 & MSB_SA_Ready_c2_r1 & MSB_SA_Ready_c2_r2 & MSB_SA_Ready_c2_r3 & MSB_SA_Ready_c2_r4 & MSB_SA_Ready_c3_r0 & MSB_SA_Ready_c3_r1 & MSB_SA_Ready_c3_r2 & MSB_SA_Ready_c3_r3 & MSB_SA_Ready_c3_r4;
    assign reg_full_sig = {reg_full_c0_r4, reg_full_c0_r3, reg_full_c0_r2, reg_full_c0_r1, reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0, 1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Column 0 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col0_dpe_0;
    assign col0_dpe_0 = dpe_out_c0_r0;
    wire signed [DATA_WIDTH-1:0] col0_dpe_1;
    assign col0_dpe_1 = dpe_out_c0_r1;
    wire signed [DATA_WIDTH-1:0] col0_dpe_2;
    assign col0_dpe_2 = dpe_out_c0_r2;
    wire signed [DATA_WIDTH-1:0] col0_dpe_3;
    assign col0_dpe_3 = dpe_out_c0_r3;
    wire signed [DATA_WIDTH-1:0] col0_dpe_4;
    assign col0_dpe_4 = dpe_out_c0_r4;
    wire [DATA_WIDTH-1:0] col0_sum;
    wire signed [40:0] col0_sum_L0_0;
    assign col0_sum_L0_0 = $signed(col0_dpe_0) + $signed(col0_dpe_1);
    wire signed [40:0] col0_sum_L0_1;
    assign col0_sum_L0_1 = $signed(col0_dpe_2) + $signed(col0_dpe_3);
    wire signed [40:0] col0_pass_L0;
    assign col0_pass_L0 = {{col0_dpe_4[39]}, col0_dpe_4};
    wire signed [41:0] col0_sum_L1_0;
    assign col0_sum_L1_0 = $signed(col0_sum_L0_0) + $signed(col0_sum_L0_1);
    wire signed [41:0] col0_pass_L1;
    assign col0_pass_L1 = {{col0_pass_L0[40]}, col0_pass_L0};
    wire signed [42:0] col0_sum_L2_0;
    assign col0_sum_L2_0 = $signed(col0_sum_L1_0) + $signed(col0_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col0_sum = col0_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col0_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c0 (
        .clk(clk),
        .data_in(col0_sum),
        .data_out(col0_act)
    );

    // Column 1 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col1_dpe_0;
    assign col1_dpe_0 = dpe_out_c1_r0;
    wire signed [DATA_WIDTH-1:0] col1_dpe_1;
    assign col1_dpe_1 = dpe_out_c1_r1;
    wire signed [DATA_WIDTH-1:0] col1_dpe_2;
    assign col1_dpe_2 = dpe_out_c1_r2;
    wire signed [DATA_WIDTH-1:0] col1_dpe_3;
    assign col1_dpe_3 = dpe_out_c1_r3;
    wire signed [DATA_WIDTH-1:0] col1_dpe_4;
    assign col1_dpe_4 = dpe_out_c1_r4;
    wire [DATA_WIDTH-1:0] col1_sum;
    wire signed [40:0] col1_sum_L0_0;
    assign col1_sum_L0_0 = $signed(col1_dpe_0) + $signed(col1_dpe_1);
    wire signed [40:0] col1_sum_L0_1;
    assign col1_sum_L0_1 = $signed(col1_dpe_2) + $signed(col1_dpe_3);
    wire signed [40:0] col1_pass_L0;
    assign col1_pass_L0 = {{col1_dpe_4[39]}, col1_dpe_4};
    wire signed [41:0] col1_sum_L1_0;
    assign col1_sum_L1_0 = $signed(col1_sum_L0_0) + $signed(col1_sum_L0_1);
    wire signed [41:0] col1_pass_L1;
    assign col1_pass_L1 = {{col1_pass_L0[40]}, col1_pass_L0};
    wire signed [42:0] col1_sum_L2_0;
    assign col1_sum_L2_0 = $signed(col1_sum_L1_0) + $signed(col1_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col1_sum = col1_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col1_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c1 (
        .clk(clk),
        .data_in(col1_sum),
        .data_out(col1_act)
    );

    // Column 2 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col2_dpe_0;
    assign col2_dpe_0 = dpe_out_c2_r0;
    wire signed [DATA_WIDTH-1:0] col2_dpe_1;
    assign col2_dpe_1 = dpe_out_c2_r1;
    wire signed [DATA_WIDTH-1:0] col2_dpe_2;
    assign col2_dpe_2 = dpe_out_c2_r2;
    wire signed [DATA_WIDTH-1:0] col2_dpe_3;
    assign col2_dpe_3 = dpe_out_c2_r3;
    wire signed [DATA_WIDTH-1:0] col2_dpe_4;
    assign col2_dpe_4 = dpe_out_c2_r4;
    wire [DATA_WIDTH-1:0] col2_sum;
    wire signed [40:0] col2_sum_L0_0;
    assign col2_sum_L0_0 = $signed(col2_dpe_0) + $signed(col2_dpe_1);
    wire signed [40:0] col2_sum_L0_1;
    assign col2_sum_L0_1 = $signed(col2_dpe_2) + $signed(col2_dpe_3);
    wire signed [40:0] col2_pass_L0;
    assign col2_pass_L0 = {{col2_dpe_4[39]}, col2_dpe_4};
    wire signed [41:0] col2_sum_L1_0;
    assign col2_sum_L1_0 = $signed(col2_sum_L0_0) + $signed(col2_sum_L0_1);
    wire signed [41:0] col2_pass_L1;
    assign col2_pass_L1 = {{col2_pass_L0[40]}, col2_pass_L0};
    wire signed [42:0] col2_sum_L2_0;
    assign col2_sum_L2_0 = $signed(col2_sum_L1_0) + $signed(col2_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col2_sum = col2_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col2_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c2 (
        .clk(clk),
        .data_in(col2_sum),
        .data_out(col2_act)
    );

    // Column 3 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col3_dpe_0;
    assign col3_dpe_0 = dpe_out_c3_r0;
    wire signed [DATA_WIDTH-1:0] col3_dpe_1;
    assign col3_dpe_1 = dpe_out_c3_r1;
    wire signed [DATA_WIDTH-1:0] col3_dpe_2;
    assign col3_dpe_2 = dpe_out_c3_r2;
    wire signed [DATA_WIDTH-1:0] col3_dpe_3;
    assign col3_dpe_3 = dpe_out_c3_r3;
    wire signed [DATA_WIDTH-1:0] col3_dpe_4;
    assign col3_dpe_4 = dpe_out_c3_r4;
    wire [DATA_WIDTH-1:0] col3_sum;
    wire signed [40:0] col3_sum_L0_0;
    assign col3_sum_L0_0 = $signed(col3_dpe_0) + $signed(col3_dpe_1);
    wire signed [40:0] col3_sum_L0_1;
    assign col3_sum_L0_1 = $signed(col3_dpe_2) + $signed(col3_dpe_3);
    wire signed [40:0] col3_pass_L0;
    assign col3_pass_L0 = {{col3_dpe_4[39]}, col3_dpe_4};
    wire signed [41:0] col3_sum_L1_0;
    assign col3_sum_L1_0 = $signed(col3_sum_L0_0) + $signed(col3_sum_L0_1);
    wire signed [41:0] col3_pass_L1;
    assign col3_pass_L1 = {{col3_pass_L0[40]}, col3_pass_L0};
    wire signed [42:0] col3_sum_L2_0;
    assign col3_sum_L2_0 = $signed(col3_sum_L1_0) + $signed(col3_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col3_sum = col3_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col3_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c3 (
        .clk(clk),
        .data_in(col3_sum),
        .data_out(col3_act)
    );

    // Horizontal output mux (4 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            2'd0: data_out_mux = col0_act;
            2'd1: data_out_mux = col1_act;
            2'd2: data_out_mux = col2_act;
            2'd3: data_out_mux = col3_act;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

// ═══════════════════════════════════════════════
// conv7_layer: V=5 H=4 K=4608 N=512
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

    localparam V = 5;
    localparam H = 4;
    localparam DEPTH = 4608;
    localparam ADDR_WIDTH = 13;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [5-1:0] reg_full_sig;
    wire [4-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [5-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [3-1:0] dpe_sel;
    wire [2-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(4608)
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
        .N_DPE_V(5),
        .N_DPE_H(4),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(4608),
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
    wire [DATA_WIDTH-1:0] dpe_out_c0_r2;
    wire dpe_done_c0_r2;
    wire reg_full_c0_r2;
    wire shift_add_done_c0_r2;
    wire shift_add_bypass_ctrl_c0_r2;
    wire MSB_SA_Ready_c0_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c0_r3;
    wire dpe_done_c0_r3;
    wire reg_full_c0_r3;
    wire shift_add_done_c0_r3;
    wire shift_add_bypass_ctrl_c0_r3;
    wire MSB_SA_Ready_c0_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c0_r4;
    wire dpe_done_c0_r4;
    wire reg_full_c0_r4;
    wire shift_add_done_c0_r4;
    wire shift_add_bypass_ctrl_c0_r4;
    wire MSB_SA_Ready_c0_r4;
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
    wire [DATA_WIDTH-1:0] dpe_out_c1_r2;
    wire dpe_done_c1_r2;
    wire reg_full_c1_r2;
    wire shift_add_done_c1_r2;
    wire shift_add_bypass_ctrl_c1_r2;
    wire MSB_SA_Ready_c1_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r3;
    wire dpe_done_c1_r3;
    wire reg_full_c1_r3;
    wire shift_add_done_c1_r3;
    wire shift_add_bypass_ctrl_c1_r3;
    wire MSB_SA_Ready_c1_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r4;
    wire dpe_done_c1_r4;
    wire reg_full_c1_r4;
    wire shift_add_done_c1_r4;
    wire shift_add_bypass_ctrl_c1_r4;
    wire MSB_SA_Ready_c1_r4;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r0;
    wire dpe_done_c2_r0;
    wire reg_full_c2_r0;
    wire shift_add_done_c2_r0;
    wire shift_add_bypass_ctrl_c2_r0;
    wire MSB_SA_Ready_c2_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r1;
    wire dpe_done_c2_r1;
    wire reg_full_c2_r1;
    wire shift_add_done_c2_r1;
    wire shift_add_bypass_ctrl_c2_r1;
    wire MSB_SA_Ready_c2_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r2;
    wire dpe_done_c2_r2;
    wire reg_full_c2_r2;
    wire shift_add_done_c2_r2;
    wire shift_add_bypass_ctrl_c2_r2;
    wire MSB_SA_Ready_c2_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r3;
    wire dpe_done_c2_r3;
    wire reg_full_c2_r3;
    wire shift_add_done_c2_r3;
    wire shift_add_bypass_ctrl_c2_r3;
    wire MSB_SA_Ready_c2_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r4;
    wire dpe_done_c2_r4;
    wire reg_full_c2_r4;
    wire shift_add_done_c2_r4;
    wire shift_add_bypass_ctrl_c2_r4;
    wire MSB_SA_Ready_c2_r4;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r0;
    wire dpe_done_c3_r0;
    wire reg_full_c3_r0;
    wire shift_add_done_c3_r0;
    wire shift_add_bypass_ctrl_c3_r0;
    wire MSB_SA_Ready_c3_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r1;
    wire dpe_done_c3_r1;
    wire reg_full_c3_r1;
    wire shift_add_done_c3_r1;
    wire shift_add_bypass_ctrl_c3_r1;
    wire MSB_SA_Ready_c3_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r2;
    wire dpe_done_c3_r2;
    wire reg_full_c3_r2;
    wire shift_add_done_c3_r2;
    wire shift_add_bypass_ctrl_c3_r2;
    wire MSB_SA_Ready_c3_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r3;
    wire dpe_done_c3_r3;
    wire reg_full_c3_r3;
    wire shift_add_done_c3_r3;
    wire shift_add_bypass_ctrl_c3_r3;
    wire MSB_SA_Ready_c3_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r4;
    wire dpe_done_c3_r4;
    wire reg_full_c3_r4;
    wire shift_add_done_c3_r4;
    wire shift_add_bypass_ctrl_c3_r4;
    wire MSB_SA_Ready_c3_r4;

    // DPE instantiations: 5 vertical x 4 horizontal = 20 DPEs
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

    dpe dpe_c0_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r2),
        .data_out(dpe_out_c0_r2),
        .dpe_done(dpe_done_c0_r2),
        .reg_full(reg_full_c0_r2),
        .shift_add_done(shift_add_done_c0_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r2)
    );

    dpe dpe_c0_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r3),
        .data_out(dpe_out_c0_r3),
        .dpe_done(dpe_done_c0_r3),
        .reg_full(reg_full_c0_r3),
        .shift_add_done(shift_add_done_c0_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r3)
    );

    dpe dpe_c0_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r4),
        .data_out(dpe_out_c0_r4),
        .dpe_done(dpe_done_c0_r4),
        .reg_full(reg_full_c0_r4),
        .shift_add_done(shift_add_done_c0_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r4)
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

    dpe dpe_c1_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r2),
        .data_out(dpe_out_c1_r2),
        .dpe_done(dpe_done_c1_r2),
        .reg_full(reg_full_c1_r2),
        .shift_add_done(shift_add_done_c1_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r2)
    );

    dpe dpe_c1_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r3),
        .data_out(dpe_out_c1_r3),
        .dpe_done(dpe_done_c1_r3),
        .reg_full(reg_full_c1_r3),
        .shift_add_done(shift_add_done_c1_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r3)
    );

    dpe dpe_c1_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r4),
        .data_out(dpe_out_c1_r4),
        .dpe_done(dpe_done_c1_r4),
        .reg_full(reg_full_c1_r4),
        .shift_add_done(shift_add_done_c1_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r4)
    );

    dpe dpe_c2_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r0),
        .data_out(dpe_out_c2_r0),
        .dpe_done(dpe_done_c2_r0),
        .reg_full(reg_full_c2_r0),
        .shift_add_done(shift_add_done_c2_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r0)
    );

    dpe dpe_c2_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r1),
        .data_out(dpe_out_c2_r1),
        .dpe_done(dpe_done_c2_r1),
        .reg_full(reg_full_c2_r1),
        .shift_add_done(shift_add_done_c2_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r1)
    );

    dpe dpe_c2_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r2),
        .data_out(dpe_out_c2_r2),
        .dpe_done(dpe_done_c2_r2),
        .reg_full(reg_full_c2_r2),
        .shift_add_done(shift_add_done_c2_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r2)
    );

    dpe dpe_c2_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r3),
        .data_out(dpe_out_c2_r3),
        .dpe_done(dpe_done_c2_r3),
        .reg_full(reg_full_c2_r3),
        .shift_add_done(shift_add_done_c2_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r3)
    );

    dpe dpe_c2_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r4),
        .data_out(dpe_out_c2_r4),
        .dpe_done(dpe_done_c2_r4),
        .reg_full(reg_full_c2_r4),
        .shift_add_done(shift_add_done_c2_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r4)
    );

    dpe dpe_c3_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r0),
        .data_out(dpe_out_c3_r0),
        .dpe_done(dpe_done_c3_r0),
        .reg_full(reg_full_c3_r0),
        .shift_add_done(shift_add_done_c3_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r0)
    );

    dpe dpe_c3_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r1),
        .data_out(dpe_out_c3_r1),
        .dpe_done(dpe_done_c3_r1),
        .reg_full(reg_full_c3_r1),
        .shift_add_done(shift_add_done_c3_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r1)
    );

    dpe dpe_c3_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r2),
        .data_out(dpe_out_c3_r2),
        .dpe_done(dpe_done_c3_r2),
        .reg_full(reg_full_c3_r2),
        .shift_add_done(shift_add_done_c3_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r2)
    );

    dpe dpe_c3_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r3),
        .data_out(dpe_out_c3_r3),
        .dpe_done(dpe_done_c3_r3),
        .reg_full(reg_full_c3_r3),
        .shift_add_done(shift_add_done_c3_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r3)
    );

    dpe dpe_c3_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r4),
        .data_out(dpe_out_c3_r4),
        .dpe_done(dpe_done_c3_r4),
        .reg_full(reg_full_c3_r4),
        .shift_add_done(shift_add_done_c3_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r4)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c0_r1 | dpe_done_c0_r2 | dpe_done_c0_r3 | dpe_done_c0_r4 | dpe_done_c1_r0 | dpe_done_c1_r1 | dpe_done_c1_r2 | dpe_done_c1_r3 | dpe_done_c1_r4 | dpe_done_c2_r0 | dpe_done_c2_r1 | dpe_done_c2_r2 | dpe_done_c2_r3 | dpe_done_c2_r4 | dpe_done_c3_r0 | dpe_done_c3_r1 | dpe_done_c3_r2 | dpe_done_c3_r3 | dpe_done_c3_r4;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c0_r1 & shift_add_done_c0_r2 & shift_add_done_c0_r3 & shift_add_done_c0_r4 & shift_add_done_c1_r0 & shift_add_done_c1_r1 & shift_add_done_c1_r2 & shift_add_done_c1_r3 & shift_add_done_c1_r4 & shift_add_done_c2_r0 & shift_add_done_c2_r1 & shift_add_done_c2_r2 & shift_add_done_c2_r3 & shift_add_done_c2_r4 & shift_add_done_c3_r0 & shift_add_done_c3_r1 & shift_add_done_c3_r2 & shift_add_done_c3_r3 & shift_add_done_c3_r4;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c0_r1 & shift_add_bypass_ctrl_c0_r2 & shift_add_bypass_ctrl_c0_r3 & shift_add_bypass_ctrl_c0_r4 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c1_r1 & shift_add_bypass_ctrl_c1_r2 & shift_add_bypass_ctrl_c1_r3 & shift_add_bypass_ctrl_c1_r4 & shift_add_bypass_ctrl_c2_r0 & shift_add_bypass_ctrl_c2_r1 & shift_add_bypass_ctrl_c2_r2 & shift_add_bypass_ctrl_c2_r3 & shift_add_bypass_ctrl_c2_r4 & shift_add_bypass_ctrl_c3_r0 & shift_add_bypass_ctrl_c3_r1 & shift_add_bypass_ctrl_c3_r2 & shift_add_bypass_ctrl_c3_r3 & shift_add_bypass_ctrl_c3_r4;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c0_r1 & MSB_SA_Ready_c0_r2 & MSB_SA_Ready_c0_r3 & MSB_SA_Ready_c0_r4 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c1_r1 & MSB_SA_Ready_c1_r2 & MSB_SA_Ready_c1_r3 & MSB_SA_Ready_c1_r4 & MSB_SA_Ready_c2_r0 & MSB_SA_Ready_c2_r1 & MSB_SA_Ready_c2_r2 & MSB_SA_Ready_c2_r3 & MSB_SA_Ready_c2_r4 & MSB_SA_Ready_c3_r0 & MSB_SA_Ready_c3_r1 & MSB_SA_Ready_c3_r2 & MSB_SA_Ready_c3_r3 & MSB_SA_Ready_c3_r4;
    assign reg_full_sig = {reg_full_c0_r4, reg_full_c0_r3, reg_full_c0_r2, reg_full_c0_r1, reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0, 1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Column 0 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col0_dpe_0;
    assign col0_dpe_0 = dpe_out_c0_r0;
    wire signed [DATA_WIDTH-1:0] col0_dpe_1;
    assign col0_dpe_1 = dpe_out_c0_r1;
    wire signed [DATA_WIDTH-1:0] col0_dpe_2;
    assign col0_dpe_2 = dpe_out_c0_r2;
    wire signed [DATA_WIDTH-1:0] col0_dpe_3;
    assign col0_dpe_3 = dpe_out_c0_r3;
    wire signed [DATA_WIDTH-1:0] col0_dpe_4;
    assign col0_dpe_4 = dpe_out_c0_r4;
    wire [DATA_WIDTH-1:0] col0_sum;
    wire signed [40:0] col0_sum_L0_0;
    assign col0_sum_L0_0 = $signed(col0_dpe_0) + $signed(col0_dpe_1);
    wire signed [40:0] col0_sum_L0_1;
    assign col0_sum_L0_1 = $signed(col0_dpe_2) + $signed(col0_dpe_3);
    wire signed [40:0] col0_pass_L0;
    assign col0_pass_L0 = {{col0_dpe_4[39]}, col0_dpe_4};
    wire signed [41:0] col0_sum_L1_0;
    assign col0_sum_L1_0 = $signed(col0_sum_L0_0) + $signed(col0_sum_L0_1);
    wire signed [41:0] col0_pass_L1;
    assign col0_pass_L1 = {{col0_pass_L0[40]}, col0_pass_L0};
    wire signed [42:0] col0_sum_L2_0;
    assign col0_sum_L2_0 = $signed(col0_sum_L1_0) + $signed(col0_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col0_sum = col0_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col0_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c0 (
        .clk(clk),
        .data_in(col0_sum),
        .data_out(col0_act)
    );

    // Column 1 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col1_dpe_0;
    assign col1_dpe_0 = dpe_out_c1_r0;
    wire signed [DATA_WIDTH-1:0] col1_dpe_1;
    assign col1_dpe_1 = dpe_out_c1_r1;
    wire signed [DATA_WIDTH-1:0] col1_dpe_2;
    assign col1_dpe_2 = dpe_out_c1_r2;
    wire signed [DATA_WIDTH-1:0] col1_dpe_3;
    assign col1_dpe_3 = dpe_out_c1_r3;
    wire signed [DATA_WIDTH-1:0] col1_dpe_4;
    assign col1_dpe_4 = dpe_out_c1_r4;
    wire [DATA_WIDTH-1:0] col1_sum;
    wire signed [40:0] col1_sum_L0_0;
    assign col1_sum_L0_0 = $signed(col1_dpe_0) + $signed(col1_dpe_1);
    wire signed [40:0] col1_sum_L0_1;
    assign col1_sum_L0_1 = $signed(col1_dpe_2) + $signed(col1_dpe_3);
    wire signed [40:0] col1_pass_L0;
    assign col1_pass_L0 = {{col1_dpe_4[39]}, col1_dpe_4};
    wire signed [41:0] col1_sum_L1_0;
    assign col1_sum_L1_0 = $signed(col1_sum_L0_0) + $signed(col1_sum_L0_1);
    wire signed [41:0] col1_pass_L1;
    assign col1_pass_L1 = {{col1_pass_L0[40]}, col1_pass_L0};
    wire signed [42:0] col1_sum_L2_0;
    assign col1_sum_L2_0 = $signed(col1_sum_L1_0) + $signed(col1_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col1_sum = col1_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col1_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c1 (
        .clk(clk),
        .data_in(col1_sum),
        .data_out(col1_act)
    );

    // Column 2 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col2_dpe_0;
    assign col2_dpe_0 = dpe_out_c2_r0;
    wire signed [DATA_WIDTH-1:0] col2_dpe_1;
    assign col2_dpe_1 = dpe_out_c2_r1;
    wire signed [DATA_WIDTH-1:0] col2_dpe_2;
    assign col2_dpe_2 = dpe_out_c2_r2;
    wire signed [DATA_WIDTH-1:0] col2_dpe_3;
    assign col2_dpe_3 = dpe_out_c2_r3;
    wire signed [DATA_WIDTH-1:0] col2_dpe_4;
    assign col2_dpe_4 = dpe_out_c2_r4;
    wire [DATA_WIDTH-1:0] col2_sum;
    wire signed [40:0] col2_sum_L0_0;
    assign col2_sum_L0_0 = $signed(col2_dpe_0) + $signed(col2_dpe_1);
    wire signed [40:0] col2_sum_L0_1;
    assign col2_sum_L0_1 = $signed(col2_dpe_2) + $signed(col2_dpe_3);
    wire signed [40:0] col2_pass_L0;
    assign col2_pass_L0 = {{col2_dpe_4[39]}, col2_dpe_4};
    wire signed [41:0] col2_sum_L1_0;
    assign col2_sum_L1_0 = $signed(col2_sum_L0_0) + $signed(col2_sum_L0_1);
    wire signed [41:0] col2_pass_L1;
    assign col2_pass_L1 = {{col2_pass_L0[40]}, col2_pass_L0};
    wire signed [42:0] col2_sum_L2_0;
    assign col2_sum_L2_0 = $signed(col2_sum_L1_0) + $signed(col2_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col2_sum = col2_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col2_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c2 (
        .clk(clk),
        .data_in(col2_sum),
        .data_out(col2_act)
    );

    // Column 3 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col3_dpe_0;
    assign col3_dpe_0 = dpe_out_c3_r0;
    wire signed [DATA_WIDTH-1:0] col3_dpe_1;
    assign col3_dpe_1 = dpe_out_c3_r1;
    wire signed [DATA_WIDTH-1:0] col3_dpe_2;
    assign col3_dpe_2 = dpe_out_c3_r2;
    wire signed [DATA_WIDTH-1:0] col3_dpe_3;
    assign col3_dpe_3 = dpe_out_c3_r3;
    wire signed [DATA_WIDTH-1:0] col3_dpe_4;
    assign col3_dpe_4 = dpe_out_c3_r4;
    wire [DATA_WIDTH-1:0] col3_sum;
    wire signed [40:0] col3_sum_L0_0;
    assign col3_sum_L0_0 = $signed(col3_dpe_0) + $signed(col3_dpe_1);
    wire signed [40:0] col3_sum_L0_1;
    assign col3_sum_L0_1 = $signed(col3_dpe_2) + $signed(col3_dpe_3);
    wire signed [40:0] col3_pass_L0;
    assign col3_pass_L0 = {{col3_dpe_4[39]}, col3_dpe_4};
    wire signed [41:0] col3_sum_L1_0;
    assign col3_sum_L1_0 = $signed(col3_sum_L0_0) + $signed(col3_sum_L0_1);
    wire signed [41:0] col3_pass_L1;
    assign col3_pass_L1 = {{col3_pass_L0[40]}, col3_pass_L0};
    wire signed [42:0] col3_sum_L2_0;
    assign col3_sum_L2_0 = $signed(col3_sum_L1_0) + $signed(col3_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col3_sum = col3_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col3_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c3 (
        .clk(clk),
        .data_in(col3_sum),
        .data_out(col3_act)
    );

    // Horizontal output mux (4 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            2'd0: data_out_mux = col0_act;
            2'd1: data_out_mux = col1_act;
            2'd2: data_out_mux = col2_act;
            2'd3: data_out_mux = col3_act;
            default: data_out_mux = 40'd0;
        endcase
    end
    assign data_out = data_out_mux;

endmodule

// ═══════════════════════════════════════════════
// conv8_layer: V=5 H=4 K=4608 N=512
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

    localparam V = 5;
    localparam H = 4;
    localparam DEPTH = 4608;
    localparam ADDR_WIDTH = 13;

    // Controller signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [5-1:0] reg_full_sig;
    wire [4-1:0] reg_empty;
    wire shift_add_done;
    wire shift_add_bypass_ctrl;
    wire [ADDR_WIDTH-1:0] read_address;
    wire [ADDR_WIDTH-1:0] write_address;
    wire [5-1:0] w_buf_en;
    wire [1:0] nl_dpe_control;
    wire shift_add_control;
    wire shift_add_bypass;
    wire load_output_reg;
    wire w_en;
    wire load_input_reg;
    wire dpe_accum_ready;
    wire dpe_accum_done;
    wire [3-1:0] dpe_sel;
    wire [2-1:0] dpe_sel_h;

    wire [DATA_WIDTH-1:0] sram_data_out;

    // Input SRAM
    sram #(
        .N_CHANNELS(1),
        .DEPTH(4608)
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
        .N_DPE_V(5),
        .N_DPE_H(4),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(1),
        .KERNEL_WIDTH(4608),
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
    wire [DATA_WIDTH-1:0] dpe_out_c0_r2;
    wire dpe_done_c0_r2;
    wire reg_full_c0_r2;
    wire shift_add_done_c0_r2;
    wire shift_add_bypass_ctrl_c0_r2;
    wire MSB_SA_Ready_c0_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c0_r3;
    wire dpe_done_c0_r3;
    wire reg_full_c0_r3;
    wire shift_add_done_c0_r3;
    wire shift_add_bypass_ctrl_c0_r3;
    wire MSB_SA_Ready_c0_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c0_r4;
    wire dpe_done_c0_r4;
    wire reg_full_c0_r4;
    wire shift_add_done_c0_r4;
    wire shift_add_bypass_ctrl_c0_r4;
    wire MSB_SA_Ready_c0_r4;
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
    wire [DATA_WIDTH-1:0] dpe_out_c1_r2;
    wire dpe_done_c1_r2;
    wire reg_full_c1_r2;
    wire shift_add_done_c1_r2;
    wire shift_add_bypass_ctrl_c1_r2;
    wire MSB_SA_Ready_c1_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r3;
    wire dpe_done_c1_r3;
    wire reg_full_c1_r3;
    wire shift_add_done_c1_r3;
    wire shift_add_bypass_ctrl_c1_r3;
    wire MSB_SA_Ready_c1_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c1_r4;
    wire dpe_done_c1_r4;
    wire reg_full_c1_r4;
    wire shift_add_done_c1_r4;
    wire shift_add_bypass_ctrl_c1_r4;
    wire MSB_SA_Ready_c1_r4;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r0;
    wire dpe_done_c2_r0;
    wire reg_full_c2_r0;
    wire shift_add_done_c2_r0;
    wire shift_add_bypass_ctrl_c2_r0;
    wire MSB_SA_Ready_c2_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r1;
    wire dpe_done_c2_r1;
    wire reg_full_c2_r1;
    wire shift_add_done_c2_r1;
    wire shift_add_bypass_ctrl_c2_r1;
    wire MSB_SA_Ready_c2_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r2;
    wire dpe_done_c2_r2;
    wire reg_full_c2_r2;
    wire shift_add_done_c2_r2;
    wire shift_add_bypass_ctrl_c2_r2;
    wire MSB_SA_Ready_c2_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r3;
    wire dpe_done_c2_r3;
    wire reg_full_c2_r3;
    wire shift_add_done_c2_r3;
    wire shift_add_bypass_ctrl_c2_r3;
    wire MSB_SA_Ready_c2_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c2_r4;
    wire dpe_done_c2_r4;
    wire reg_full_c2_r4;
    wire shift_add_done_c2_r4;
    wire shift_add_bypass_ctrl_c2_r4;
    wire MSB_SA_Ready_c2_r4;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r0;
    wire dpe_done_c3_r0;
    wire reg_full_c3_r0;
    wire shift_add_done_c3_r0;
    wire shift_add_bypass_ctrl_c3_r0;
    wire MSB_SA_Ready_c3_r0;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r1;
    wire dpe_done_c3_r1;
    wire reg_full_c3_r1;
    wire shift_add_done_c3_r1;
    wire shift_add_bypass_ctrl_c3_r1;
    wire MSB_SA_Ready_c3_r1;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r2;
    wire dpe_done_c3_r2;
    wire reg_full_c3_r2;
    wire shift_add_done_c3_r2;
    wire shift_add_bypass_ctrl_c3_r2;
    wire MSB_SA_Ready_c3_r2;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r3;
    wire dpe_done_c3_r3;
    wire reg_full_c3_r3;
    wire shift_add_done_c3_r3;
    wire shift_add_bypass_ctrl_c3_r3;
    wire MSB_SA_Ready_c3_r3;
    wire [DATA_WIDTH-1:0] dpe_out_c3_r4;
    wire dpe_done_c3_r4;
    wire reg_full_c3_r4;
    wire shift_add_done_c3_r4;
    wire shift_add_bypass_ctrl_c3_r4;
    wire MSB_SA_Ready_c3_r4;

    // DPE instantiations: 5 vertical x 4 horizontal = 20 DPEs
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

    dpe dpe_c0_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r2),
        .data_out(dpe_out_c0_r2),
        .dpe_done(dpe_done_c0_r2),
        .reg_full(reg_full_c0_r2),
        .shift_add_done(shift_add_done_c0_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r2)
    );

    dpe dpe_c0_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r3),
        .data_out(dpe_out_c0_r3),
        .dpe_done(dpe_done_c0_r3),
        .reg_full(reg_full_c0_r3),
        .shift_add_done(shift_add_done_c0_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r3)
    );

    dpe dpe_c0_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c0_r4),
        .data_out(dpe_out_c0_r4),
        .dpe_done(dpe_done_c0_r4),
        .reg_full(reg_full_c0_r4),
        .shift_add_done(shift_add_done_c0_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c0_r4)
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

    dpe dpe_c1_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r2),
        .data_out(dpe_out_c1_r2),
        .dpe_done(dpe_done_c1_r2),
        .reg_full(reg_full_c1_r2),
        .shift_add_done(shift_add_done_c1_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r2)
    );

    dpe dpe_c1_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r3),
        .data_out(dpe_out_c1_r3),
        .dpe_done(dpe_done_c1_r3),
        .reg_full(reg_full_c1_r3),
        .shift_add_done(shift_add_done_c1_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r3)
    );

    dpe dpe_c1_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1_r4),
        .data_out(dpe_out_c1_r4),
        .dpe_done(dpe_done_c1_r4),
        .reg_full(reg_full_c1_r4),
        .shift_add_done(shift_add_done_c1_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1_r4)
    );

    dpe dpe_c2_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r0),
        .data_out(dpe_out_c2_r0),
        .dpe_done(dpe_done_c2_r0),
        .reg_full(reg_full_c2_r0),
        .shift_add_done(shift_add_done_c2_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r0)
    );

    dpe dpe_c2_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r1),
        .data_out(dpe_out_c2_r1),
        .dpe_done(dpe_done_c2_r1),
        .reg_full(reg_full_c2_r1),
        .shift_add_done(shift_add_done_c2_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r1)
    );

    dpe dpe_c2_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r2),
        .data_out(dpe_out_c2_r2),
        .dpe_done(dpe_done_c2_r2),
        .reg_full(reg_full_c2_r2),
        .shift_add_done(shift_add_done_c2_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r2)
    );

    dpe dpe_c2_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r3),
        .data_out(dpe_out_c2_r3),
        .dpe_done(dpe_done_c2_r3),
        .reg_full(reg_full_c2_r3),
        .shift_add_done(shift_add_done_c2_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r3)
    );

    dpe dpe_c2_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2_r4),
        .data_out(dpe_out_c2_r4),
        .dpe_done(dpe_done_c2_r4),
        .reg_full(reg_full_c2_r4),
        .shift_add_done(shift_add_done_c2_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2_r4)
    );

    dpe dpe_c3_r0 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[0]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r0),
        .data_out(dpe_out_c3_r0),
        .dpe_done(dpe_done_c3_r0),
        .reg_full(reg_full_c3_r0),
        .shift_add_done(shift_add_done_c3_r0),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r0)
    );

    dpe dpe_c3_r1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[1]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r1),
        .data_out(dpe_out_c3_r1),
        .dpe_done(dpe_done_c3_r1),
        .reg_full(reg_full_c3_r1),
        .shift_add_done(shift_add_done_c3_r1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r1)
    );

    dpe dpe_c3_r2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[2]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r2),
        .data_out(dpe_out_c3_r2),
        .dpe_done(dpe_done_c3_r2),
        .reg_full(reg_full_c3_r2),
        .shift_add_done(shift_add_done_c3_r2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r2)
    );

    dpe dpe_c3_r3 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[3]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r3),
        .data_out(dpe_out_c3_r3),
        .dpe_done(dpe_done_c3_r3),
        .reg_full(reg_full_c3_r3),
        .shift_add_done(shift_add_done_c3_r3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r3)
    );

    dpe dpe_c3_r4 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en[4]),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c3_r4),
        .data_out(dpe_out_c3_r4),
        .dpe_done(dpe_done_c3_r4),
        .reg_full(reg_full_c3_r4),
        .shift_add_done(shift_add_done_c3_r4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c3_r4)
    );

    // Aggregate control signals
    assign dpe_done = dpe_done_c0_r0 | dpe_done_c0_r1 | dpe_done_c0_r2 | dpe_done_c0_r3 | dpe_done_c0_r4 | dpe_done_c1_r0 | dpe_done_c1_r1 | dpe_done_c1_r2 | dpe_done_c1_r3 | dpe_done_c1_r4 | dpe_done_c2_r0 | dpe_done_c2_r1 | dpe_done_c2_r2 | dpe_done_c2_r3 | dpe_done_c2_r4 | dpe_done_c3_r0 | dpe_done_c3_r1 | dpe_done_c3_r2 | dpe_done_c3_r3 | dpe_done_c3_r4;
    assign shift_add_done = shift_add_done_c0_r0 & shift_add_done_c0_r1 & shift_add_done_c0_r2 & shift_add_done_c0_r3 & shift_add_done_c0_r4 & shift_add_done_c1_r0 & shift_add_done_c1_r1 & shift_add_done_c1_r2 & shift_add_done_c1_r3 & shift_add_done_c1_r4 & shift_add_done_c2_r0 & shift_add_done_c2_r1 & shift_add_done_c2_r2 & shift_add_done_c2_r3 & shift_add_done_c2_r4 & shift_add_done_c3_r0 & shift_add_done_c3_r1 & shift_add_done_c3_r2 & shift_add_done_c3_r3 & shift_add_done_c3_r4;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c0_r0 & shift_add_bypass_ctrl_c0_r1 & shift_add_bypass_ctrl_c0_r2 & shift_add_bypass_ctrl_c0_r3 & shift_add_bypass_ctrl_c0_r4 & shift_add_bypass_ctrl_c1_r0 & shift_add_bypass_ctrl_c1_r1 & shift_add_bypass_ctrl_c1_r2 & shift_add_bypass_ctrl_c1_r3 & shift_add_bypass_ctrl_c1_r4 & shift_add_bypass_ctrl_c2_r0 & shift_add_bypass_ctrl_c2_r1 & shift_add_bypass_ctrl_c2_r2 & shift_add_bypass_ctrl_c2_r3 & shift_add_bypass_ctrl_c2_r4 & shift_add_bypass_ctrl_c3_r0 & shift_add_bypass_ctrl_c3_r1 & shift_add_bypass_ctrl_c3_r2 & shift_add_bypass_ctrl_c3_r3 & shift_add_bypass_ctrl_c3_r4;
    assign MSB_SA_Ready = MSB_SA_Ready_c0_r0 & MSB_SA_Ready_c0_r1 & MSB_SA_Ready_c0_r2 & MSB_SA_Ready_c0_r3 & MSB_SA_Ready_c0_r4 & MSB_SA_Ready_c1_r0 & MSB_SA_Ready_c1_r1 & MSB_SA_Ready_c1_r2 & MSB_SA_Ready_c1_r3 & MSB_SA_Ready_c1_r4 & MSB_SA_Ready_c2_r0 & MSB_SA_Ready_c2_r1 & MSB_SA_Ready_c2_r2 & MSB_SA_Ready_c2_r3 & MSB_SA_Ready_c2_r4 & MSB_SA_Ready_c3_r0 & MSB_SA_Ready_c3_r1 & MSB_SA_Ready_c3_r2 & MSB_SA_Ready_c3_r3 & MSB_SA_Ready_c3_r4;
    assign reg_full_sig = {reg_full_c0_r4, reg_full_c0_r3, reg_full_c0_r2, reg_full_c0_r1, reg_full_c0_r0};
    assign reg_empty = {1'b0, 1'b0, 1'b0, 1'b0};
    assign dpe_accum_done = dpe_done;

    // Column 0 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col0_dpe_0;
    assign col0_dpe_0 = dpe_out_c0_r0;
    wire signed [DATA_WIDTH-1:0] col0_dpe_1;
    assign col0_dpe_1 = dpe_out_c0_r1;
    wire signed [DATA_WIDTH-1:0] col0_dpe_2;
    assign col0_dpe_2 = dpe_out_c0_r2;
    wire signed [DATA_WIDTH-1:0] col0_dpe_3;
    assign col0_dpe_3 = dpe_out_c0_r3;
    wire signed [DATA_WIDTH-1:0] col0_dpe_4;
    assign col0_dpe_4 = dpe_out_c0_r4;
    wire [DATA_WIDTH-1:0] col0_sum;
    wire signed [40:0] col0_sum_L0_0;
    assign col0_sum_L0_0 = $signed(col0_dpe_0) + $signed(col0_dpe_1);
    wire signed [40:0] col0_sum_L0_1;
    assign col0_sum_L0_1 = $signed(col0_dpe_2) + $signed(col0_dpe_3);
    wire signed [40:0] col0_pass_L0;
    assign col0_pass_L0 = {{col0_dpe_4[39]}, col0_dpe_4};
    wire signed [41:0] col0_sum_L1_0;
    assign col0_sum_L1_0 = $signed(col0_sum_L0_0) + $signed(col0_sum_L0_1);
    wire signed [41:0] col0_pass_L1;
    assign col0_pass_L1 = {{col0_pass_L0[40]}, col0_pass_L0};
    wire signed [42:0] col0_sum_L2_0;
    assign col0_sum_L2_0 = $signed(col0_sum_L1_0) + $signed(col0_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col0_sum = col0_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col0_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c0 (
        .clk(clk),
        .data_in(col0_sum),
        .data_out(col0_act)
    );

    // Column 1 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col1_dpe_0;
    assign col1_dpe_0 = dpe_out_c1_r0;
    wire signed [DATA_WIDTH-1:0] col1_dpe_1;
    assign col1_dpe_1 = dpe_out_c1_r1;
    wire signed [DATA_WIDTH-1:0] col1_dpe_2;
    assign col1_dpe_2 = dpe_out_c1_r2;
    wire signed [DATA_WIDTH-1:0] col1_dpe_3;
    assign col1_dpe_3 = dpe_out_c1_r3;
    wire signed [DATA_WIDTH-1:0] col1_dpe_4;
    assign col1_dpe_4 = dpe_out_c1_r4;
    wire [DATA_WIDTH-1:0] col1_sum;
    wire signed [40:0] col1_sum_L0_0;
    assign col1_sum_L0_0 = $signed(col1_dpe_0) + $signed(col1_dpe_1);
    wire signed [40:0] col1_sum_L0_1;
    assign col1_sum_L0_1 = $signed(col1_dpe_2) + $signed(col1_dpe_3);
    wire signed [40:0] col1_pass_L0;
    assign col1_pass_L0 = {{col1_dpe_4[39]}, col1_dpe_4};
    wire signed [41:0] col1_sum_L1_0;
    assign col1_sum_L1_0 = $signed(col1_sum_L0_0) + $signed(col1_sum_L0_1);
    wire signed [41:0] col1_pass_L1;
    assign col1_pass_L1 = {{col1_pass_L0[40]}, col1_pass_L0};
    wire signed [42:0] col1_sum_L2_0;
    assign col1_sum_L2_0 = $signed(col1_sum_L1_0) + $signed(col1_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col1_sum = col1_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col1_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c1 (
        .clk(clk),
        .data_in(col1_sum),
        .data_out(col1_act)
    );

    // Column 2 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col2_dpe_0;
    assign col2_dpe_0 = dpe_out_c2_r0;
    wire signed [DATA_WIDTH-1:0] col2_dpe_1;
    assign col2_dpe_1 = dpe_out_c2_r1;
    wire signed [DATA_WIDTH-1:0] col2_dpe_2;
    assign col2_dpe_2 = dpe_out_c2_r2;
    wire signed [DATA_WIDTH-1:0] col2_dpe_3;
    assign col2_dpe_3 = dpe_out_c2_r3;
    wire signed [DATA_WIDTH-1:0] col2_dpe_4;
    assign col2_dpe_4 = dpe_out_c2_r4;
    wire [DATA_WIDTH-1:0] col2_sum;
    wire signed [40:0] col2_sum_L0_0;
    assign col2_sum_L0_0 = $signed(col2_dpe_0) + $signed(col2_dpe_1);
    wire signed [40:0] col2_sum_L0_1;
    assign col2_sum_L0_1 = $signed(col2_dpe_2) + $signed(col2_dpe_3);
    wire signed [40:0] col2_pass_L0;
    assign col2_pass_L0 = {{col2_dpe_4[39]}, col2_dpe_4};
    wire signed [41:0] col2_sum_L1_0;
    assign col2_sum_L1_0 = $signed(col2_sum_L0_0) + $signed(col2_sum_L0_1);
    wire signed [41:0] col2_pass_L1;
    assign col2_pass_L1 = {{col2_pass_L0[40]}, col2_pass_L0};
    wire signed [42:0] col2_sum_L2_0;
    assign col2_sum_L2_0 = $signed(col2_sum_L1_0) + $signed(col2_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col2_sum = col2_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col2_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c2 (
        .clk(clk),
        .data_in(col2_sum),
        .data_out(col2_act)
    );

    // Column 3 adder tree (V=5)
    wire signed [DATA_WIDTH-1:0] col3_dpe_0;
    assign col3_dpe_0 = dpe_out_c3_r0;
    wire signed [DATA_WIDTH-1:0] col3_dpe_1;
    assign col3_dpe_1 = dpe_out_c3_r1;
    wire signed [DATA_WIDTH-1:0] col3_dpe_2;
    assign col3_dpe_2 = dpe_out_c3_r2;
    wire signed [DATA_WIDTH-1:0] col3_dpe_3;
    assign col3_dpe_3 = dpe_out_c3_r3;
    wire signed [DATA_WIDTH-1:0] col3_dpe_4;
    assign col3_dpe_4 = dpe_out_c3_r4;
    wire [DATA_WIDTH-1:0] col3_sum;
    wire signed [40:0] col3_sum_L0_0;
    assign col3_sum_L0_0 = $signed(col3_dpe_0) + $signed(col3_dpe_1);
    wire signed [40:0] col3_sum_L0_1;
    assign col3_sum_L0_1 = $signed(col3_dpe_2) + $signed(col3_dpe_3);
    wire signed [40:0] col3_pass_L0;
    assign col3_pass_L0 = {{col3_dpe_4[39]}, col3_dpe_4};
    wire signed [41:0] col3_sum_L1_0;
    assign col3_sum_L1_0 = $signed(col3_sum_L0_0) + $signed(col3_sum_L0_1);
    wire signed [41:0] col3_pass_L1;
    assign col3_pass_L1 = {{col3_pass_L0[40]}, col3_pass_L0};
    wire signed [42:0] col3_sum_L2_0;
    assign col3_sum_L2_0 = $signed(col3_sum_L1_0) + $signed(col3_pass_L1);
    // Truncate adder tree result to 40 bits
    assign col3_sum = col3_sum_L2_0[39:0];

    wire [DATA_WIDTH-1:0] col3_act;
    activation_lut #(.DATA_WIDTH(DATA_WIDTH)) act_c3 (
        .clk(clk),
        .data_in(col3_sum),
        .data_out(col3_act)
    );

    // Horizontal output mux (4 columns)
    reg [DATA_WIDTH-1:0] data_out_mux;
    always @(*) begin
        case (dpe_sel_h)
            2'd0: data_out_mux = col0_act;
            2'd1: data_out_mux = col1_act;
            2'd2: data_out_mux = col2_act;
            2'd3: data_out_mux = col3_act;
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

module pool_mod_pool5 #(parameter DATA_WIDTH = 40) (
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
