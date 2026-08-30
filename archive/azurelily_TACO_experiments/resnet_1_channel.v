module resnet (
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [15:0] data_in,
    output wire [15:0] data_out,
    output wire ready,
    output wire valid_n
);

    // Internal signals
	wire [15:0] data_out_conv1, data_out_act1, data_out_pool1, global_sram_data_in;
	wire [15:0] data_out_conv2, data_out_act2, data_out_pool2;
	wire [15:0] data_out_act3, data_out_conv4, data_out_act4, data_out_conv5;
	wire [15:0] data_out_dummy_res1, data_out_dummy_res2, data_out_conv3;
	wire [15:0] data_out_act5, data_out_conv6, data_out_act6, data_out_pool3, data_out_conv7;
	wire [15:0] data_out_act7, data_out_conv8, data_out_act8, data_out_pool4, data_out_conv9;
    wire ready_conv1, valid_conv1, ready_conv2, valid_conv2, ready_conv3, valid_conv3;
    wire ready_act1, valid_act1, ready_act2, valid_act2, valid_act3;
    wire ready_pool1, valid_pool1, ready_pool2, valid_pool2, ready_pool3, valid_pool3, ready_pool4, valid_pool4;
    wire ready_res1, valid_res1, ready_res2, valid_res2;
    wire ready_conv3a, valid_n_pool2, valid_n_conv3b, ready_accum, valid_n_conv3a, ready_conv3b;
    wire valid_n_accum, ready_act3, ready_conv4, valid_conv4, valid_n_act3;
    wire ready_act4, valid_act4, ready_act5, valid_act5, valid_n_conv4;
    wire ready_act6, valid_act6, ready_act7, valid_act7, ready_act8, valid_act8;
    wire ready_conv5, valid_conv5, ready_conv6, valid_conv6, ready_conv7, valid_conv7, ready_conv8, valid_conv8;
    wire valid_g_in,valid_g_out,ready_g_in,ready_g_out;

    reg [7:0] read_address, write_address;
    
    // Instantiate the first conv_layer
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(10),
        .N_KERNELS(1),
        .KERNEL_WIDTH(3),
        .KERNEL_HEIGHT(3),
        .W(32),
        .H(32),
        .S(1),
        .DEPTH(1024),
        .DATA_WIDTH(16)
    ) conv1 (
        .clk(clk),
        .rst(rst),
        .valid(valid_g_out),
        .ready_n(ready_conv1), // udpated with ready_Ln
        .data_in(data_in),
        .data_out(data_out_conv1), // this wasn't correct
        .ready(ready_g_in), // or ready_conv1
        .valid_n(valid_conv1)
    );
	
	// Instantiate the first activation_layer
    activation_layer1 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(10),
        .DATA_WIDTH(16),
        .DEPTH(1024)
    ) act1 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv1),
        .ready_n(ready_act1),
        .data_in(data_out_conv1),
        .data_out(data_out_act1),
        .ready(ready_conv1),
        .valid_n(valid_act1)
    );
	
	// Instantiate the second conv_layer
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(10),
        .N_KERNELS(1),
        .KERNEL_WIDTH(3),
        .KERNEL_HEIGHT(3),
        .W(32),
        .H(32),
        .S(1),
        .DEPTH(1024),
        .DATA_WIDTH(16)
    ) conv2 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act1),
        .ready_n(ready_conv2), // udpated with ready_Ln
        .data_in(data_out_act1),
        .data_out(data_out_conv2), // this wasn't correct
        .ready(ready_act1), // or ready_conv1
        .valid_n(valid_conv2)
    );
	
	// Instantiate the second activation_layer
    activation_layer2 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(10),
        .DATA_WIDTH(16),
        .DEPTH(1024)
    ) act2 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv2),
        .ready_n(ready_act2),
        .data_in(data_out_conv2),
        .data_out(data_out_act2),
        .ready(ready_conv2),
        .valid_n(valid_act2)
    ); 
	
	
	// Instantiate the first pool_layer
    pool_layer1 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(10),
        .DATA_WIDTH(16),
        .DEPTH(1024)
    ) pool1 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act2),
        .ready_n(ready_pool1),
        .layer_done(1'b0),
        .data_in(data_out_act2),
        .data_out(data_out_pool1),
        .ready(ready_act2),
		.valid_n(valid_pool1)
    ); 
	
	// Instantiate the third conv_layer
 	conv_layer_stacked_dpes_V2_H1 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(3),
        .KERNEL_HEIGHT(3),
        .W(16),
        .H(16),
        .S(1),
		.N_DPE_V(2),
		.N_DPE_H(1),
		.N_BRAM_R(1),
		.N_BRAM_W(1),
		.DATA_WIDTH (16),
        .DEPTH(512)
    ) conv3 (
        .clk(clk),
        .rst(rst),
        .valid(valid_pool1),
        .ready_n(ready_conv3),
        .data_in(data_out_pool1),
        .data_out(data_out_conv3),
        .ready(ready_pool1),
        .valid_n(valid_conv3)
    );

	// Instantiate the third activation_layer
    activation_layer3 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(8),
        .DATA_WIDTH(16),
        .DEPTH(256)
    )act3 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv3),
        .ready_n(ready_act3),
        .data_in(data_out_conv3),
        .data_out(data_out_act3),
        .ready(ready_conv3),
        .valid_n(valid_act3)
    );
	
	// Instantiate the fourth conv_layer
 	conv_layer_stacked_dpes_V2_H1 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(3),
        .KERNEL_HEIGHT(3),
        .W(16),
        .H(16),
        .S(1),
		.N_DPE_V(2),
		.N_DPE_H(1),
		.N_BRAM_R(1),
		.N_BRAM_W(1),
		.DATA_WIDTH (16),
        .DEPTH(512)
    ) conv4 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act3),
        .ready_n(ready_conv4),
        .data_in(data_out_act3),
        .data_out(data_out_conv4),
        .ready(ready_act3),
        .valid_n(valid_conv4)
    );
	
	// Instantiate the fourth activation_layer
      activation_layer4 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(8),
        .DATA_WIDTH(16),
        .DEPTH(256)
    )act4 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv4),
        .ready_n(ready_act4),
        .data_in(data_out_conv4),
        .data_out(data_out_act4),
        .ready(ready_conv4),
        .valid_n(valid_act4)
    ); 
	
	// Instantiate the first (dummy) residual layer
    residual_layer1 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(8),
        .DEPTH(256)
        ) residual1 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act4),
        .ready_n(ready_res1),
        .data_in(data_out_act4),
        .data_in2(data_out_pool1),
        .data_out(data_out_dummy_res1),
        .ready(ready_act4),
        .valid_n(valid_res1)
    );	
	
	// Instantiate the fifth conv_layer
	conv_layer_stacked_dpes_V2_H2 #(
		.N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(3),
        .KERNEL_HEIGHT(3),
        .W(16),
        .H(16),
        .S(1),
		.N_DPE_V(2),
		.N_DPE_H(2),
		.N_BRAM_R(1),
		.N_BRAM_W(1),
		.DATA_WIDTH (16),
        .DEPTH(512)
    ) conv5 (
        .clk(clk),
        .rst(rst),
        .valid(valid_res1),
        .ready_n(ready_conv5),
        .data_in(data_out_dummy_res1),
        .data_out(data_out_conv5),
        .ready(ready_res1),
        .valid_n(valid_conv5)
    );
	
	// Instantiate the fifth activation_layer
    activation_layer5 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(8),
        .DATA_WIDTH(16),
        .DEPTH(256)
    )act5 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv5),
        .ready_n(ready_act5),
        .data_in(data_out_conv5),
        .data_out(data_out_act5),
        .ready(ready_conv5),
        .valid_n(valid_act5)
    );
	
	// Instantiate the second pool_layer
    pool_layer2 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(8),
        .DATA_WIDTH(16),
        .DEPTH(256)
    ) pool2 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act5),
        .ready_n(ready_pool2),
        .layer_done(1'b0),
        .data_in(data_out_act5),
        .data_out(data_out_pool2),
        .ready(ready_act5),
        .valid_n(valid_pool2)
    );
	
	// Instantiate the sixth conv_layer
    conv_layer_stacked_dpes_V4_H2 #(
		.N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(3),
        .KERNEL_HEIGHT(3),
        .W(8),
        .H(8),
        .S(1),
		.N_DPE_V(4),
		.N_DPE_H(2),
		.N_BRAM_R(1),
		.N_BRAM_W(1),
		.DATA_WIDTH (16),
        .DEPTH(512)
    ) conv6 (
        .clk(clk),
        .rst(rst),
        .valid(valid_pool2),
        .ready_n(ready_conv6),
        .data_in(data_out_pool2),
        .data_out(data_out_conv6),
        .ready(ready_pool2),
        .valid_n(valid_conv6)
    );
	
	// Instantiate the sixth activation_layer
    activation_layer6 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(6),
        .DATA_WIDTH(16),
        .DEPTH(64)
    )act6 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv6),
        .ready_n(ready_act6),
        .data_in(data_out_conv6),
        .data_out(data_out_act6),
        .ready(ready_conv6),
        .valid_n(valid_act6)
    );
	
	// Instantiate the third pool_layer
    pool_layer3 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(64),
        .DATA_WIDTH(16),
        .DEPTH(64)
    ) pool3 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act6),
        .ready_n(ready_pool3),
        .layer_done(1'b0),
        .data_in(data_out_act6),
        .data_out(data_out_pool3),
        .ready(ready_act6),
        .valid_n(valid_pool3)
    );
	
	// Instantiate the seventh conv_layer
    conv_layer_stacked_dpes_V4_H2 #(
		.N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(3),
        .KERNEL_HEIGHT(3),
        .W(4),
        .H(4),
        .S(1),
		.N_DPE_V(4),
		.N_DPE_H(2),
		.N_BRAM_R(1),
		.N_BRAM_W(1),
		.DATA_WIDTH (16),
        .DEPTH(512)
    ) conv7 (
        .clk(clk),
        .rst(rst),
        .valid(valid_pool3),
        .ready_n(ready_conv7),
        .data_in(data_out_pool3),
        .data_out(data_out_conv7),
        .ready(ready_pool3),
        .valid_n(valid_conv7)
    );
	
	// Instantiate the seventh activation_layer
    activation_layer7 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(4),
        .DATA_WIDTH(16),
        .DEPTH(16)
    )act7 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv7),
        .ready_n(ready_act7),
        .data_in(data_out_conv7),
        .data_out(data_out_act7),
        .ready(ready_conv7),
        .valid_n(valid_act7)
    );
	
	// Instantiate the eighth conv_layer
    conv_layer_stacked_dpes_V4_H2 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(3),
        .KERNEL_HEIGHT(3),
        .W(4),
        .H(4),
        .S(1),
		.N_DPE_V(4),
		.N_DPE_H(2),
		.N_BRAM_R(1),
		.N_BRAM_W(1),
		.DATA_WIDTH (16),
        .DEPTH(512)
    ) conv8 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act7),
        .ready_n(ready_conv8),
        .data_in(data_out_act7),
        .data_out(data_out_conv8),
        .ready(ready_act7),
        .valid_n(valid_conv8)
    ); 
	
	// Instantiate the eighth activation_layer
    activation_layer8 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(4),
        .DATA_WIDTH(16),
        .DEPTH(16)
    )act8 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv8),
        .ready_n(ready_act8),
        .data_in(data_out_conv8),
        .data_out(data_out_act8),
        .ready(ready_conv8),
        .valid_n(valid_act8)
    );
	
	// Instantiate the second (dummy) residual layer
    residual_layer2 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(4),
        .DEPTH(16)
        ) residual2 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act8),
        .ready_n(ready_res2),
        .data_in(data_out_act8),
        .data_in2(data_out_pool2),
        .data_out(data_out_dummy_res2),
        .ready(ready_act8),
        .valid_n(valid_res2)
    );

	// Instantiate the fourth pool_layer
    pool_layer4 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(4),
        .DATA_WIDTH(16),
        .DEPTH(16)
    ) pool4 (
        .clk(clk),
        .rst(rst),
        .valid(valid_res2),
        .ready_n(ready_pool4),
        .layer_done(1'b0),
        .data_in(data_out_dummy_res2),
        .data_out(data_out_pool4),
        .ready(ready_res2),
        .valid_n(valid_pool4)
    );
	
	// Instantiate the ninth conv_layer
    conv_layer_single_dpe #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(7),
        .N_KERNELS(1),
        .KERNEL_WIDTH(1),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(128)
    ) conv9 (
        .clk(clk),
        .rst(rst),
        .valid(valid_pool4),
        .ready_n(ready_g_out),
        .data_in(data_out_pool4),
        .data_out(data_out_conv9),
        .ready(ready_pool4),
        .valid_n(valid_g_in)
    );
	
	
    global_controller #(
    .N_Layers(5)        
    ) g_ctrl_inst(
    .clk(clk),              
    .rst(rst),             
    .ready_L1(ready_g_in),     // trigger/signal from nl_dpe indicating new data can be read
    .valid_Ln(valid_g_in),            // Valid signal to enable new operation
    .valid(valid),                   //corrected
    .ready(ready),                 //corrected
    .valid_L1(valid_g_out),     //corrected
    .ready_Ln (ready_g_out)    //corrected
);

// Global SRAM
sram #(
    .N_CHANNELS(1),
    .DATA_WIDTH(16), //redundant
    .DEPTH(16)
) global_sram_inst (
    .clk(clk),
	.rst(rst),
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
        if(ready_g_out) begin
            read_address <= read_address + 1;
        end
        else begin
            read_address <= read_address;
        end
        if(valid_g_out) begin
            write_address <= write_address + 1;
        end
        else begin
            write_address <= write_address;
        end
    end
end

    // Final output connections
    assign data_out = global_sram_data_in;
    assign ready = ready_g_in;
    assign valid_n = valid_g_in;

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
    wire [DATA_WIDTH-1:0] sram_data_in;
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

    // Instantiate the DPE module
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
        //.w_en(w_en),
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

    // Instantiate the DPE modules
	// instance: dpe_R{number}_C{number} - to easily identify horizontal and vertical stacks
    dpe dpe_R1_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
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
        .data_in(dpe_data[DATA_WIDTH-1:0]),
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
		//.r_en_dec(1),
		.load_input_reg(load_input_reg),
		.ready(ready),
        .valid_n(valid_n),
		.dpe_accum_ready(accum_ready),
		.dpe_sel(dpe_sel_signal),
		.dpe_sel_h(dpe_sel_h_signal)
    );

endmodule


module conv_layer_stacked_dpes_V2_H2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 5,
    parameter KERNEL_HEIGHT = 5,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 2,
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
    output wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    // Internal signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [N_DPE_V-1:0] reg_full_sig1;
    wire [N_DPE_V-1:0] reg_full_sig2;
    wire [N_DPE_H-1:0] reg_empty_sig;
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
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_temp1;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_temp2;
	wire [N_BRAM_W-1:0] w_en_dec_signal;
	wire [N_DPE_V-1:0] dpe_sel_signal;
	wire [N_DPE_H-1:0] dpe_sel_h_signal;
	wire [DATA_WIDTH-1:0] sram_data;
	wire dpe_done1, dpe_done2, dpe_done3, dpe_done4;
	wire accum_ready, accum_done;
	wire shift_add_done1, shift_add_done2, shift_add_done3, shift_add_done4;
	wire shift_add_bypass_ctrl1, shift_add_bypass_ctrl2, shift_add_bypass_ctrl3, shift_add_bypass_ctrl4;
	wire MSB_SA_Ready1, MSB_SA_Ready2, MSB_SA_Ready3, MSB_SA_Ready4;	

    // Instantiate the SRAM module
    sram #(
        .N_CHANNELS(1),
        .DEPTH(512)
    ) sram_inst_1 (
        .clk(clk),
		.rst(rst),
        //.w_en(w_en),
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

    // Instantiate the DPE modules
	// R1C1 R1C2 -- reg1
	// R2C1 R2C2 -- reg2
	// instance: dpe_R{number}_C{number} - to easily identify horizontal and vertical stacks
    dpe dpe_R1_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready1),
        .data_out(data_out1),
        .dpe_done(dpe_done1),
        .reg_full(reg_full_sig1[0]),
        .shift_add_done(shift_add_done1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl1)
    );
	
	dpe dpe_R1_C2 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready2),
        .data_out(data_out2),
        .dpe_done(dpe_done2),
        .reg_full(reg_full_sig1[1]),
        .shift_add_done(shift_add_done2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl2)
    );
	
	dpe dpe_R2_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready3),
        .data_out(data_out3),
        .dpe_done(dpe_done3),
        .reg_full(reg_full_sig2[0]),
        .shift_add_done(shift_add_done3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl3)
    );

	dpe dpe_R2_C2 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready4),
        .data_out(data_out4),
        .dpe_done(dpe_done4),
        .reg_full(reg_full_sig2[1]),
        .shift_add_done(shift_add_done4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl4)
    );
	
	assign shift_add_done = shift_add_done1 & shift_add_done2 & shift_add_done3 & shift_add_done4;
	assign shift_add_bypass_ctrl = shift_add_bypass_ctrl1 & shift_add_bypass_ctrl2 & shift_add_bypass_ctrl3 & shift_add_bypass_ctrl4;
	assign dpe_done = dpe_done1 & dpe_done2 & dpe_done3 & dpe_done4;
	assign reg_full_sig = reg_full_sig1 & reg_full_sig2;
	assign MSB_SA_Ready = MSB_SA_Ready1 & MSB_SA_Ready2 & MSB_SA_Ready3 & MSB_SA_Ready4;
	
  	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R1C1_R2C1 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done1),
        .input1(data_out1),
        .input2(data_out3),
        .output_data(data_out_temp1)
    );
	
	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R1C2_R2C2 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done2),
        .input1(data_out2),
        .input2(data_out4),
        .output_data(data_out_temp2)
    );
	
	assign accum_done = accum_done1 & accum_done2;
	assign data_out = data_out_temp1 + data_out_temp2;
	
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
		//.r_en_dec(1),
		.load_input_reg(load_input_reg),
		.ready(ready),
        .valid_n(valid_n),
		.dpe_accum_ready(accum_ready),
		.dpe_sel(dpe_sel_signal),
		.dpe_sel_h(dpe_sel_h_signal)
    );

endmodule

module conv_layer_stacked_dpes_V4_H2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 5,
    parameter KERNEL_HEIGHT = 5,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 4,
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
    output wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out,
    output wire ready,
    output wire valid_n
);

    // Internal signals
    wire MSB_SA_Ready;
    wire dpe_done;
    wire [N_DPE_V-1:0] reg_full_sig1;
    wire [N_DPE_V-1:0] reg_full_sig2;
    wire [N_DPE_V-1:0] reg_full_sig3;
    wire [N_DPE_V-1:0] reg_full_sig4;
    wire [N_DPE_H-1:0] reg_empty_sig;
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

	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out1, data_out2, data_out3, data_out4;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out5, data_out6, data_out7, data_out8;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_temp1, data_out_temp2, data_out_temp3, data_out_temp4;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_part1, data_out_part2;
	wire [N_BRAM_W-1:0] w_en_dec_signal;
	wire [N_DPE_V-1:0] dpe_sel_signal;
	wire [N_DPE_H-1:0] dpe_sel_h_signal;
	wire [DATA_WIDTH-1:0] sram_data;
	wire dpe_done1, dpe_done2, dpe_done3, dpe_done4;
	wire dpe_done5, dpe_done6, dpe_done7, dpe_done8;
	wire accum_ready, accum_done;
	wire shift_add_done1, shift_add_done2, shift_add_done3, shift_add_done4;
	wire shift_add_done5, shift_add_done6, shift_add_done7, shift_add_done8;
	wire shift_add_bypass_ctrl1, shift_add_bypass_ctrl2, shift_add_bypass_ctrl3, shift_add_bypass_ctrl4;
	wire shift_add_bypass_ctrl5, shift_add_bypass_ctrl6, shift_add_bypass_ctrl7, shift_add_bypass_ctrl8;
	wire MSB_SA_Ready1, MSB_SA_Ready2, MSB_SA_Ready3, MSB_SA_Ready4;	
	wire MSB_SA_Ready5, MSB_SA_Ready6, MSB_SA_Ready7, MSB_SA_Ready8;	

    // Instantiate the SRAM module
    sram #(
        .N_CHANNELS(1),        
        .DEPTH(512)
    ) sram_inst_1 (
        .clk(clk),
		.rst(rst),
        //.w_en(w_en),
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
        .NUM_OUTPUTS(4)
    ) u_xbar (
        .in_data(sram_data),
        .in_sel(N_BRAM_R),
        .out_sel(dpe_sel_signal),
        .out_data(dpe_data)
    );

    // Instantiate the DPE modules
	// R1C1 R1C2 -- reg1
	// R2C1 R2C2 -- reg2
	// instance: dpe_R{number}_C{number} - to easily identify horizontal and vertical stacks
    dpe dpe_R1_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready1),
        .data_out(data_out1),
        .dpe_done(dpe_done1),
        .reg_full(reg_full_sig1[0]),
        .shift_add_done(shift_add_done1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl1)
    );
	
	dpe dpe_R1_C2 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready2),
        .data_out(data_out2),
        .dpe_done(dpe_done2),
        .reg_full(reg_full_sig1[1]),
        .shift_add_done(shift_add_done2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl2)
    );
	
	dpe dpe_R2_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready3),
        .data_out(data_out3),
        .dpe_done(dpe_done3),
        .reg_full(reg_full_sig2[0]),
        .shift_add_done(shift_add_done3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl3)
    );

	dpe dpe_R2_C2 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready4),
        .data_out(data_out4),
        .dpe_done(dpe_done4),
        .reg_full(reg_full_sig2[1]),
        .shift_add_done(shift_add_done4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl4)
    );
	
	dpe dpe_R3_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready5),
        .data_out(data_out5),
        .dpe_done(dpe_done5),
        .reg_full(reg_full_sig3[0]),
        .shift_add_done(shift_add_done5),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl5)
    );
	
	dpe dpe_R3_C2 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready6),
        .data_out(data_out6),
        .dpe_done(dpe_done6),
        .reg_full(reg_full_sig3[1]),
        .shift_add_done(shift_add_done6),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl6)
    );
	
	dpe dpe_R4_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready7),
        .data_out(data_out7),
        .dpe_done(dpe_done7),
        .reg_full(reg_full_sig4[0]),
        .shift_add_done(shift_add_done7),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl7)
    );

	dpe dpe_R4_C2 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data[DATA_WIDTH-1:0]),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready8),
        .data_out(data_out8),
        .dpe_done(dpe_done8),
        .reg_full(reg_full_sig4[1]),
        .shift_add_done(shift_add_done8),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl8)
    );
	
	assign shift_add_done = shift_add_done1 & shift_add_done2 & shift_add_done3 & shift_add_done4 &
							shift_add_done5 & shift_add_done6 & shift_add_done7 & shift_add_done8;
	assign shift_add_bypass_ctrl = shift_add_bypass_ctrl1 & shift_add_bypass_ctrl2 & shift_add_bypass_ctrl3 & shift_add_bypass_ctrl4 &
								   shift_add_bypass_ctrl5 & shift_add_bypass_ctrl6 & shift_add_bypass_ctrl7 & shift_add_bypass_ctrl8;
	assign dpe_done = dpe_done1 & dpe_done2 & dpe_done3 & dpe_done4 &
					  dpe_done5 & dpe_done6 & dpe_done7 & dpe_done8;
	assign reg_full_sig = reg_full_sig1 & reg_full_sig2 & reg_full_sig3 & reg_full_sig4;
	assign MSB_SA_Ready = MSB_SA_Ready1 & MSB_SA_Ready2 & MSB_SA_Ready3 & MSB_SA_Ready4 &
						  MSB_SA_Ready5 & MSB_SA_Ready6 & MSB_SA_Ready7 & MSB_SA_Ready8;
	
  	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R1C1_R2C1 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        //.load_input_reg(w_buf_en),
        //.reg_full(reg_full_sig),
        //.load_output_reg(load_output_reg),
        .add_done(accum_done1),
        .input1(data_out1),
        .input2(data_out3),
        .output_data(data_out_temp1)
    );
	
	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R1C2_R2C2 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done2),
        .input1(data_out2),
        .input2(data_out4),
        .output_data(data_out_temp2)
    );
	
	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R3C1_R4C1 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done1),
        .input1(data_out5),
        .input2(data_out7),
        .output_data(data_out_temp3)
    );
	
	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R3C2_R4C2 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        //.load_input_reg(w_buf_en),
        //.reg_full(reg_full_sig),
        //.load_output_reg(load_output_reg),
        .add_done(accum_done2),
        .input1(data_out6),
        .input2(data_out8),
        .output_data(data_out_temp4)
    );
	
	assign accum_done = accum_done1 & accum_done2 & accum_done3 & accum_done4;
	
	always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
            data_out_part1 <= 0;
            data_out_part2 <= 0;
        end else begin
			data_out_part1 <= data_out_temp1 + data_out_temp2;
			data_out_part2 <= data_out_temp3 + data_out_temp4;
			data_out <= data_out_part1 + data_out_part2;
        end
    end
	
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
		//.r_en_dec(1),
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
    //output wire [(N_BRAM_R)-1:0] r_en_dec,
    
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
    //reg r_en;
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
            //r_en <= 1'b0;
            w_buf_en <= 0;
            next_address <= {VW_ADDR_WIDTH{1'b0}};
            pointer_offset <= 0;
            sv <= 0;
            s <= 0;
            n <= 0;
            m <= 0;
        end else begin
            //r_en <= ~(&reg_full);
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
		/* if (write_address_reg[VW_ADDR_WIDTH-1:ADDR_WIDTH] < N_BRAM_W) begin
            w_dec = 1'b1 << write_address_reg[VW_ADDR_WIDTH-1:ADDR_WIDTH];
        end
        
        if (read_address_reg[VR_ADDR_WIDTH-1:ADDR_WIDTH] < N_BRAM_R) begin
            r_dec = 1'b1 << read_address_reg[VR_ADDR_WIDTH-1:ADDR_WIDTH];
        end */
    end

    // Output assignments
    assign w_en_dec = w_dec & {N_BRAM_W{w_en}};
    //assign r_en_dec = r_dec & {N_BRAM_R{r_en}};
    assign load_input_reg = reg_full;
    assign busy = ~MSB_SA_Ready;

endmodule

module global_controller #(
    parameter N_Layers = 1        
)(
    input wire clk,              
    input wire rst,             
    input wire ready_L1,     // trigger/signal from nl_dpe indicating new data can be read
    input wire valid_Ln,            // Valid signal to enable new operation
    input wire valid,
    output reg ready,                 // Ready signal indicating operation is done
    output reg valid_L1,
    output reg ready_Ln
);
    
    wire busy;
    reg stall;     

    // valid and ready control
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            //valid_L1 <= 1'b0;
            //ready <= 1'b0;
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
    parameter DATA_WIDTH = 16*N_CHANNELS,  // Data width (default: 8 bits) 8 x number of channels
    parameter DEPTH = 512       // Memory depth (default: 512)
    
)(
    input wire clk,           
    input wire w_en,
	input wire rst,
    input wire [$clog2(DEPTH)-1:0] r_addr,  // Address input (width based on depth)
    input wire [$clog2(DEPTH)-1:0] w_addr,
    input wire [DATA_WIDTH-1:0] sram_data_in,  // Data input for writing
    output reg [DATA_WIDTH-1:0] sram_data_out  // Data output for reading
);

    // Memory array with parameterized depth and width
    //reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] mem [DEPTH-1:0];

    // Read/Write operations
    //always @(posedge clk or posedge rst) begin
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

module activation_layer1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate ReLu_PE
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

module relu_activation_parallel_N_CHANNELS_1 #(
    parameter N_CHANNELS = 1  // Number of channels
)(
    input wire clk,
    input wire reset,
    input wire en,
    input wire load_input_reg,  // Signal to start filling data into registers
    output reg reg_full,        // Signal indicating all registers are full
    input wire load_output_reg, // Signal to load the output registers
    output reg activation_done, // Internal signal indicating ReLU activation is complete
    input wire [16*N_CHANNELS-1:0] sram_data_in,  // Input data from SRAM for all channels
    output reg [16*N_CHANNELS-1:0] sram_data_out  // ReLU activation output for all channels
);

    reg [15:0] input_data_0;  // Input storage for channel 0
    reg [15:0] relu_output_0; // ReLU output for channel 0

    // Block 1: Reading and Storing Data
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            input_data_0 <= 16'd0;
            reg_full <= 1'b0;
        end else if (en && load_input_reg) begin
            input_data_0 <= sram_data_in[15:0];
            reg_full <= 1'b1; // Indicate that all registers are full
        end else begin
            reg_full <= 1'b0;
        end
    end

    // Block 2: Optimized ReLU Activation Logic using MSB Check
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            relu_output_0 <= 16'd0;
            activation_done <= 1'b0;
        end else if (reg_full) begin
            relu_output_0 <= (input_data_0[15] == 1'b0) ? input_data_0 : 16'd0;
            activation_done <= 1'b1; // Set activation done flag
        end else begin
            activation_done <= 1'b0;
        end
    end

    // Block 3: Storing ReLU Output into Output Registers
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            sram_data_out[15:0] <= 16'd0;
        end else if (load_output_reg) begin
            sram_data_out[15:0] <= relu_output_0;
        end
    end
endmodule


module pooling_controller #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,  // Parameter for address width (default: 7 bits)
    parameter N_KERNELS = 1,
    parameter KERNEL_SIZE = 3,
    parameter B_ADDR_WIDTH = $clog2(KERNEL_SIZE * KERNEL_SIZE)
)(
    input wire clk,              
    input wire rst,             
    input wire pooling_done,     // trigger/signal from nl_dpe indicating new data can be read
    input wire valid,            // Valid signal to enable new operation
    input wire ready_n,
    input wire layer_done,
    input wire reg_full,
    output reg [ADDR_WIDTH-1:0] read_address, // Read address for SRAM (parameterized width)
    output reg [ADDR_WIDTH-1:0] write_address, // Write address for SRAM (parameterized width)
    output reg w_buf_en,           // load input register
    output reg p_en,
    output reg [1:0] pooling_control, // Op mode signal for nl_dpe (2 bits)
    output reg load_output_reg,      // Signal indicating shift-and-add logic is done
    output reg w_en,                 // Write enable signal
    output reg ready,                 // Ready signal indicating operation is done
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
        load_output_reg <= pooling_done; // Replacing shift_add_done with pooling_done
    end

    assign busy = ~pooling_done;

endmodule

module pool_layer1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate max_pooling_28x28_pipelined
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
    input wire load_input_reg,  // Signal to start filling data into registers
    output reg reg_full,        // Signal indicating all registers are full
    input wire load_output_reg, // Signal to load the output registers
    output reg pooling_done,    // Internal signal indicating pooling is complete
    input wire [16*N_CHANNELS-1:0] sram_data_in,  // Input data from SRAM for all channels
    output reg [16*N_CHANNELS-1:0] sram_data_out    // Max-pooling output for all channels
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

module activation_layer2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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
        .N_CHANNELS(1),
        .DEPTH(512)
    ) sram_inst (
        .clk(clk),
		.rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_in),
        .sram_data_out(sram_data_in)
    );

    // Instantiate ReLu_PE
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

module activation_layer4 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 128,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate ReLu_PE
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

module residual_layer2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 4,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 16,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate max_pooling_28x28_pipelined
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
	
	// Instantiate sram
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
    //input wire load_input_reg,  // Signal to start filling data into registers
    //output reg reg_full,        // Signal indicating all registers are full
    //input wire load_output_reg, // Signal to load the output registers
    output reg add_done,        // Internal signal indicating the addition is complete
    input wire [16*N_CHANNELS-1:0] input1,  // First set of 8-bit inputs for all channels
    input wire [16*N_CHANNELS-1:0] input2,  // Second set of 8-bit inputs for all channels
    output reg [16*N_CHANNELS-1:0] output_data  // 8-bit addition output for all channels
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
            //reg_full <= 1'b0;
        //end else if (en && load_input_reg) begin
        end else if (en) begin
            in_data1_0 <= input1[15:0];
            in_data2_0 <= input2[15:0];
            //reg_full <= 1'b1; // Indicate that all registers are full
        end else begin
            //reg_full <= 1'b0;
        end
    end

    // Block 2: Addition Logic with LSB Dropped
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            result_0 <= 16'd0;
            channel_index <= 8'd0;
            processing <= 1'b0;
            add_done <= 1'b0;
        //end else if (reg_full && !processing) begin
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
        //end else if (load_output_reg) begin
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
    input wire load_input_reg,  // Signal to start filling data into registers
    output reg reg_full,        // Signal indicating all registers are full
    input wire load_output_reg, // Signal to load the output registers
    output reg add_done,        // Internal signal indicating the addition is complete
    input wire [16*N_CHANNELS-1:0] input1,  // First set of 8-bit inputs for all channels
    input wire [16*N_CHANNELS-1:0] input2,  // Second set of 8-bit inputs for all channels
    output reg [16*N_CHANNELS-1:0] output_data  // 8-bit addition output for all channels
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
        //end else if (en) begin
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
        //end else if (!processing) begin
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
        //end else begin
            output_data[15:0] <= result_0;
        end
    end
endmodule


module pool_layer4 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate max_pooling_28x28_pipelined
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

module activation_layer7 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 128,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate ReLu_PE
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

module residual_layer1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 256,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate max_pooling_28x28_pipelined
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
	
	// Instantiate sram
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

module pool_layer3 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate max_pooling_28x28_pipelined
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

module activation_layer3 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 128,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate ReLu_PE
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

module pool_layer2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 512,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate max_pooling_28x28_pipelined
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

module activation_layer5 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 128,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate ReLu_PE
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

module activation_layer6 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 128,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate ReLu_PE
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

module activation_layer8 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = N_CHANNELS*16,
    parameter DEPTH = 128,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [16*N_CHANNELS-1:0] sram_data_out;

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

    // Instantiate ReLu_PE
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

module xbar_ip_module #(
    parameter DATA_WIDTH = 16,
    parameter NUM_INPUTS = 1,
    parameter NUM_OUTPUTS = 4,
    // Derived parameters to handle special cases
    parameter IN_SEL_WIDTH = (NUM_INPUTS <= 1) ? 1 : $clog2(NUM_INPUTS),
    parameter OUT_SEL_WIDTH = (NUM_OUTPUTS <= 1) ? 1 : $clog2(NUM_OUTPUTS)
)(
    input  wire [NUM_INPUTS*DATA_WIDTH-1:0]  in_data,
    input  wire [IN_SEL_WIDTH-1:0]           in_sel,  // Will be 1 bit when NUM_INPUTS=1
    input  wire [OUT_SEL_WIDTH-1:0]          out_sel, // Will be 1 bit when NUM_OUTPUTS=1
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
