module LeNet3_0 (
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [7:0] data_in,
    output wire [7:0] data_out,
    output wire ready,
    output wire valid_n
);

    // Internal signals
	wire [7:0] data_out_conv1, data_out_act1, data_out_pool1, global_sram_data_in;
	wire [7:0] data_out_conv2, data_out_act2, data_out_pool2;
	wire [7:0] data_out_conv3a, data_out_conv3b, data_out_act3, data_out_conv4, data_out_act4, data_out_conv5, data_out_accum;
    wire ready_conv1, valid_conv1, ready_conv2, valid_conv2;
    wire ready_act1, valid_act1, ready_act2, valid_act2;
    wire ready_pool2, valid_pool2;
    wire ready_conv3a, valid_n_pool2, valid_n_conv3b, ready_accum, valid_n_conv3a, ready_conv3b;
    wire valid_n_accum, ready_act3, ready_conv4, valid_n_act3;
    wire ready_act4, valid_n_conv4;
    wire ready_conv5, valid_conv5;
    wire valid_g_in,valid_g_out,ready_g_in,ready_g_out;


    reg [7:0] read_address, write_address;
    
    // Instantiate the first conv_layer
    conv_layer #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(5),
        .W(32),
        .H(32),
        .S(1),
        .DEPTH(512),
        .DATA_WIDTH(8)
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
        .ADDR_WIDTH(9),
        .DATA_WIDTH(8),
        .DEPTH(512)
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
	
	// Instantiate the first pool_layer
    pool_layer1 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(10),
        .DATA_WIDTH(8),
        .DEPTH(1024)
    ) pool1 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act1),
        .ready_n(ready_conv2),
        .layer_done(1'b0),
        .data_in(data_out_act1),
        .data_out(data_out_pool1),
        .ready(ready_act1),
        .valid_n(valid_conv2)
    );
	
	// Instantiate the second conv_layer
    conv_layer #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(5),
        .KERNEL_HEIGHT(5),
        .W(14),
        .H(14),
        .S(1),
        .DEPTH(512)
    ) conv2 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv2),
        .ready_n(ready_act2),
        .data_in(data_out_pool1),
        .data_out(data_out_conv2),
        .ready(ready_conv2),
        .valid_n(valid_act2)
    );
	
	// Instantiate the second activation_layer
    activation_layer2 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(7),
        .DATA_WIDTH(128),
        .DEPTH(128)
    )act2 (
        .clk(clk),
        .rst(rst),
        .valid(valid_act2),
        .ready_n(ready_pool2),
        .data_in(data_out_conv2),
        .data_out(data_out_act2),
        .ready(ready_act2),
        .valid_n(valid_pool2)
    );
	
	// Instantiate the second pool_layer
    pool_layer2 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(7),
        .DATA_WIDTH(128),
        .DEPTH(128)
    ) pool2 (
        .clk(clk),
        .rst(rst),
        .valid(valid_pool2),
        .ready_n(ready_conv3a),
        .layer_done(1'b0),
        .data_in(data_out_act2),
        .data_out(data_out_pool2),
        .ready(ready_pool2),
        .valid_n(valid_n_pool2)
    );
	
	// Instantiate the third conv_layer
    conv_layer #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(1),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(512)
    ) conv3a (
        .clk(clk),
        .rst(rst),
        .valid(valid_n_pool2),
        .ready_n(ready_conv3b),
        .data_in(data_out_pool2),
        .data_out(data_out_conv3a),
        .ready(ready_conv3a),
        .valid_n(valid_n_conv3a)
    );

	conv_layer #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(1),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(512)
    ) conv3b (
        .clk(clk),
        .rst(rst),
        .valid(valid_n_conv3a),
        .ready_n(ready_accum),
        .data_in(data_out_pool2),
        .data_out(data_out_conv3b),
        .ready(ready_conv3b),
        .valid_n(valid_n_conv3b)
    );
    // Instantiate the accumulation_layer
    accumulation_layer #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(7),
        .DEPTH(128)
        ) accum (
        .clk(clk),
        .rst(rst),
        .valid(valid_n_conv3b),
        .ready_n(ready_act3),
        .data_in(data_out_conv3a),
        .data_in2(data_out_conv3b),
        .data_out(data_out_accum),
        .ready(ready_accum),
        .valid_n(valid_n_accum)
    );
	
	// Instantiate the third activation_layer
    activation_layer3 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(7),
        .DATA_WIDTH(8),
        .DEPTH(128)
    ) act3 (
        .clk(clk),
        .rst(rst),
        .valid(valid_n_accum),
        .ready_n(ready_conv4),
        .data_in(data_out_accum),
        .data_out(data_out_act3),
        .ready(ready_act3),
        .valid_n(valid_n_act3)
    );
	
	// Instantiate the fourth conv_layer
    conv_layer #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(9),
        .N_KERNELS(1),
        .KERNEL_WIDTH(1),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(512)
    ) conv4 (
        .clk(clk),
        .rst(rst),
        .valid(valid_n_act3),
        .ready_n(ready_act4),
        .data_in(data_out_act3),
        .data_out(data_out_conv4),
        .ready(ready_conv4),
        .valid_n(valid_n_conv4)
    );
	
	// Instantiate the fourth activation_layer
    activation_layer4 #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(7),
        .DATA_WIDTH(8),
        .DEPTH(128)
    ) act4 (
        .clk(clk),
        .rst(rst),
        .valid(valid_n_conv4),
        .ready_n(ready_conv5),
        .data_in(data_out_conv4),
        .data_out(data_out_act4),
        .ready(ready_act4),
        .valid_n(valid_conv5)
    );
	
	    // Instantiate the fifth conv_layer
    conv_layer #(
        .N_CHANNELS(1),
        .ADDR_WIDTH(7),
        .N_KERNELS(1),
        .KERNEL_WIDTH(1),
        .KERNEL_HEIGHT(1),
        .W(1),
        .H(1),
        .S(1),
        .DEPTH(128)
    ) conv5 (
        .clk(clk),
        .rst(rst),
        .valid(valid_conv5),
        .ready_n(ready_g_out),
        .data_in(data_out_act4),
        .data_out(data_out_conv5),
        .ready(ready_conv5),
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
    .DATA_WIDTH(8), //redundant
    .DEPTH(128)
) global_sram_inst (
    .clk(clk),
	.rst(rst),
    .w_en(valid_g_in),
    .r_addr(read_address),
    .w_addr(write_address),
    .sram_data_in(data_out_conv5),
    .sram_data_out(global_sram_data_in)
);


always @(posedge clk or posedge rst) begin
    if (rst) begin
        read_address <= 0;
        write_address <= 128;
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

// Top Module
module conv_layer #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 5,
    parameter KERNEL_HEIGHT = 5,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter DATA_WIDTH = 8
    
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
        .N_CHANNELS(N_CHANNELS),        
        .DEPTH(DEPTH)
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
    parameter DATA_WIDTH = 8*N_CHANNELS,  // Data width (default: 8 bits) 8 x number of channels
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
    parameter DATA_WIDTH = N_CHANNELS*8,
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
    //wire [8*N_CHANNELS-1:0] sram_data_out;

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
    tanh_activation_parallel_N_CHANNELS_1 #(
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

module tanh_activation_parallel_N_CHANNELS_1 #(
    parameter N_CHANNELS = 1  // Number of channels
)(
    input wire clk,
    input wire reset,
    input wire en,
    input wire load_input_reg,  // Signal to start filling data into registers
    output reg reg_full,        // Signal indicating all registers are full
    input wire load_output_reg, // Signal to load the output registers
    output reg activation_done, // Internal signal indicating tanh activation is complete
    input wire [8*N_CHANNELS-1:0] sram_data_in,  // Input data from SRAM for all channels
    output reg [8*N_CHANNELS-1:0] sram_data_out  // Tanh activation output for all channels
);

    reg [7:0] input_data_0;  // Input storage for channel 0
    reg [7:0] tanh_output_0; // Tanh output for channel 0

    // Block 1: Reading and Storing Data
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            input_data_0 <= 8'd0;
            reg_full <= 1'b0;
        end else if (en && load_input_reg) begin
            input_data_0 <= sram_data_in[7:0];
            reg_full <= 1'b1; // Indicate that all registers are full
        end else begin
            reg_full <= 1'b0;
        end
    end

    // Block 2: Tanh Activation Logic using Lookup Table
    reg [7:0] tanh_lut [255:0];  // Tanh lookup table with 256 values
    initial begin
        // Example values - Populate this with actual tanh LUT
        tanh_lut[0] = 8'd0;
        tanh_lut[1] = 8'd1;
        tanh_lut[2] = 8'd2;
        tanh_lut[3] = 8'd3;
        tanh_lut[4] = 8'd4;
        tanh_lut[5] = 8'd5;
        tanh_lut[6] = 8'd6;
        tanh_lut[7] = 8'd7;
        tanh_lut[8] = 8'd8;
        tanh_lut[9] = 8'd9;
        tanh_lut[10] = 8'd10;
        tanh_lut[11] = 8'd11;
        tanh_lut[12] = 8'd12;
        tanh_lut[13] = 8'd13;
        tanh_lut[14] = 8'd14;
        tanh_lut[15] = 8'd15;
        tanh_lut[16] = 8'd16;
        tanh_lut[17] = 8'd17;
        tanh_lut[18] = 8'd18;
        tanh_lut[19] = 8'd19;
        tanh_lut[20] = 8'd20;
        tanh_lut[21] = 8'd21;
        tanh_lut[22] = 8'd22;
        tanh_lut[23] = 8'd23;
        tanh_lut[24] = 8'd24;
        tanh_lut[25] = 8'd25;
        tanh_lut[26] = 8'd26;
        tanh_lut[27] = 8'd27;
        tanh_lut[28] = 8'd28;
        tanh_lut[29] = 8'd29;
        tanh_lut[30] = 8'd30;
        tanh_lut[31] = 8'd31;
        tanh_lut[32] = 8'd32;
        tanh_lut[33] = 8'd33;
        tanh_lut[34] = 8'd34;
        tanh_lut[35] = 8'd35;
        tanh_lut[36] = 8'd36;
        tanh_lut[37] = 8'd37;
        tanh_lut[38] = 8'd38;
        tanh_lut[39] = 8'd39;
        tanh_lut[40] = 8'd40;
        tanh_lut[41] = 8'd41;
        tanh_lut[42] = 8'd42;
        tanh_lut[43] = 8'd43;
        tanh_lut[44] = 8'd44;
        tanh_lut[45] = 8'd45;
        tanh_lut[46] = 8'd46;
        tanh_lut[47] = 8'd47;
        tanh_lut[48] = 8'd48;
        tanh_lut[49] = 8'd49;
        tanh_lut[50] = 8'd50;
        tanh_lut[51] = 8'd51;
        tanh_lut[52] = 8'd52;
        tanh_lut[53] = 8'd53;
        tanh_lut[54] = 8'd54;
        tanh_lut[55] = 8'd55;
        tanh_lut[56] = 8'd56;
        tanh_lut[57] = 8'd57;
        tanh_lut[58] = 8'd58;
        tanh_lut[59] = 8'd59;
        tanh_lut[60] = 8'd60;
        tanh_lut[61] = 8'd61;
        tanh_lut[62] = 8'd62;
        tanh_lut[63] = 8'd63;
        tanh_lut[64] = 8'd64;
        tanh_lut[65] = 8'd65;
        tanh_lut[66] = 8'd66;
        tanh_lut[67] = 8'd67;
        tanh_lut[68] = 8'd68;
        tanh_lut[69] = 8'd69;
        tanh_lut[70] = 8'd70;
        tanh_lut[71] = 8'd71;
        tanh_lut[72] = 8'd72;
        tanh_lut[73] = 8'd73;
        tanh_lut[74] = 8'd74;
        tanh_lut[75] = 8'd75;
        tanh_lut[76] = 8'd76;
        tanh_lut[77] = 8'd77;
        tanh_lut[78] = 8'd78;
        tanh_lut[79] = 8'd79;
        tanh_lut[80] = 8'd80;
        tanh_lut[81] = 8'd81;
        tanh_lut[82] = 8'd82;
        tanh_lut[83] = 8'd83;
        tanh_lut[84] = 8'd84;
        tanh_lut[85] = 8'd85;
        tanh_lut[86] = 8'd86;
        tanh_lut[87] = 8'd87;
        tanh_lut[88] = 8'd88;
        tanh_lut[89] = 8'd89;
        tanh_lut[90] = 8'd90;
        tanh_lut[91] = 8'd91;
        tanh_lut[92] = 8'd92;
        tanh_lut[93] = 8'd93;
        tanh_lut[94] = 8'd94;
        tanh_lut[95] = 8'd95;
        tanh_lut[96] = 8'd96;
        tanh_lut[97] = 8'd97;
        tanh_lut[98] = 8'd98;
        tanh_lut[99] = 8'd99;
        tanh_lut[100] = 8'd100;
        tanh_lut[101] = 8'd101;
        tanh_lut[102] = 8'd102;
        tanh_lut[103] = 8'd103;
        tanh_lut[104] = 8'd104;
        tanh_lut[105] = 8'd105;
        tanh_lut[106] = 8'd106;
        tanh_lut[107] = 8'd107;
        tanh_lut[108] = 8'd108;
        tanh_lut[109] = 8'd109;
        tanh_lut[110] = 8'd110;
        tanh_lut[111] = 8'd111;
        tanh_lut[112] = 8'd112;
        tanh_lut[113] = 8'd113;
        tanh_lut[114] = 8'd114;
        tanh_lut[115] = 8'd115;
        tanh_lut[116] = 8'd116;
        tanh_lut[117] = 8'd117;
        tanh_lut[118] = 8'd118;
        tanh_lut[119] = 8'd119;
        tanh_lut[120] = 8'd120;
        tanh_lut[121] = 8'd121;
        tanh_lut[122] = 8'd122;
        tanh_lut[123] = 8'd123;
        tanh_lut[124] = 8'd124;
        tanh_lut[125] = 8'd125;
        tanh_lut[126] = 8'd126;
        tanh_lut[127] = 8'd127;
        tanh_lut[128] = 8'd128;
        tanh_lut[129] = 8'd129;
        tanh_lut[130] = 8'd130;
        tanh_lut[131] = 8'd131;
        tanh_lut[132] = 8'd132;
        tanh_lut[133] = 8'd133;
        tanh_lut[134] = 8'd134;
        tanh_lut[135] = 8'd135;
        tanh_lut[136] = 8'd136;
        tanh_lut[137] = 8'd137;
        tanh_lut[138] = 8'd138;
        tanh_lut[139] = 8'd139;
        tanh_lut[140] = 8'd140;
        tanh_lut[141] = 8'd141;
        tanh_lut[142] = 8'd142;
        tanh_lut[143] = 8'd143;
        tanh_lut[144] = 8'd144;
        tanh_lut[145] = 8'd145;
        tanh_lut[146] = 8'd146;
        tanh_lut[147] = 8'd147;
        tanh_lut[148] = 8'd148;
        tanh_lut[149] = 8'd149;
        tanh_lut[150] = 8'd150;
        tanh_lut[151] = 8'd151;
        tanh_lut[152] = 8'd152;
        tanh_lut[153] = 8'd153;
        tanh_lut[154] = 8'd154;
        tanh_lut[155] = 8'd155;
        tanh_lut[156] = 8'd156;
        tanh_lut[157] = 8'd157;
        tanh_lut[158] = 8'd158;
        tanh_lut[159] = 8'd159;
        tanh_lut[160] = 8'd160;
        tanh_lut[161] = 8'd161;
        tanh_lut[162] = 8'd162;
        tanh_lut[163] = 8'd163;
        tanh_lut[164] = 8'd164;
        tanh_lut[165] = 8'd165;
        tanh_lut[166] = 8'd166;
        tanh_lut[167] = 8'd167;
        tanh_lut[168] = 8'd168;
        tanh_lut[169] = 8'd169;
        tanh_lut[170] = 8'd170;
        tanh_lut[171] = 8'd171;
        tanh_lut[172] = 8'd172;
        tanh_lut[173] = 8'd173;
        tanh_lut[174] = 8'd174;
        tanh_lut[175] = 8'd175;
        tanh_lut[176] = 8'd176;
        tanh_lut[177] = 8'd177;
        tanh_lut[178] = 8'd178;
        tanh_lut[179] = 8'd179;
        tanh_lut[180] = 8'd180;
        tanh_lut[181] = 8'd181;
        tanh_lut[182] = 8'd182;
        tanh_lut[183] = 8'd183;
        tanh_lut[184] = 8'd184;
        tanh_lut[185] = 8'd185;
        tanh_lut[186] = 8'd186;
        tanh_lut[187] = 8'd187;
        tanh_lut[188] = 8'd188;
        tanh_lut[189] = 8'd189;
        tanh_lut[190] = 8'd190;
        tanh_lut[191] = 8'd191;
        tanh_lut[192] = 8'd192;
        tanh_lut[193] = 8'd193;
        tanh_lut[194] = 8'd194;
        tanh_lut[195] = 8'd195;
        tanh_lut[196] = 8'd196;
        tanh_lut[197] = 8'd197;
        tanh_lut[198] = 8'd198;
        tanh_lut[199] = 8'd199;
        tanh_lut[200] = 8'd200;
        tanh_lut[201] = 8'd201;
        tanh_lut[202] = 8'd202;
        tanh_lut[203] = 8'd203;
        tanh_lut[204] = 8'd204;
        tanh_lut[205] = 8'd205;
        tanh_lut[206] = 8'd206;
        tanh_lut[207] = 8'd207;
        tanh_lut[208] = 8'd208;
        tanh_lut[209] = 8'd209;
        tanh_lut[210] = 8'd210;
        tanh_lut[211] = 8'd211;
        tanh_lut[212] = 8'd212;
        tanh_lut[213] = 8'd213;
        tanh_lut[214] = 8'd214;
        tanh_lut[215] = 8'd215;
        tanh_lut[216] = 8'd216;
        tanh_lut[217] = 8'd217;
        tanh_lut[218] = 8'd218;
        tanh_lut[219] = 8'd219;
        tanh_lut[220] = 8'd220;
        tanh_lut[221] = 8'd221;
        tanh_lut[222] = 8'd222;
        tanh_lut[223] = 8'd223;
        tanh_lut[224] = 8'd224;
        tanh_lut[225] = 8'd225;
        tanh_lut[226] = 8'd226;
        tanh_lut[227] = 8'd227;
        tanh_lut[228] = 8'd228;
        tanh_lut[229] = 8'd229;
        tanh_lut[230] = 8'd230;
        tanh_lut[231] = 8'd231;
        tanh_lut[232] = 8'd232;
        tanh_lut[233] = 8'd233;
        tanh_lut[234] = 8'd234;
        tanh_lut[235] = 8'd235;
        tanh_lut[236] = 8'd236;
        tanh_lut[237] = 8'd237;
        tanh_lut[238] = 8'd238;
        tanh_lut[239] = 8'd239;
        tanh_lut[240] = 8'd240;
        tanh_lut[241] = 8'd241;
        tanh_lut[242] = 8'd242;
        tanh_lut[243] = 8'd243;
        tanh_lut[244] = 8'd244;
        tanh_lut[245] = 8'd245;
        tanh_lut[246] = 8'd246;
        tanh_lut[247] = 8'd247;
        tanh_lut[248] = 8'd248;
        tanh_lut[249] = 8'd249;
        tanh_lut[250] = 8'd250;
        tanh_lut[251] = 8'd251;
        tanh_lut[252] = 8'd252;
        tanh_lut[253] = 8'd253;
        tanh_lut[254] = 8'd254;
        tanh_lut[255] = 8'd255;
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            tanh_output_0 <= 8'd0;
            activation_done <= 1'b0;
        end else if (reg_full) begin
            tanh_output_0 <= tanh_lut[input_data_0];
            activation_done <= 1'b1; // Set activation done flag
        end else begin
            activation_done <= 1'b0;
        end
    end

    // Block 3: Storing Tanh Output into Output Registers
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            sram_data_out[7:0] <= 8'd0;
        end else if (load_output_reg) begin
            sram_data_out[7:0] <= tanh_output_0;
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
    parameter DATA_WIDTH = N_CHANNELS*8,
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
    //wire [8*N_CHANNELS-1:0] sram_data_out;

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
    input wire [8*N_CHANNELS-1:0] sram_data_in,  // Input data from SRAM for all channels
    output reg [8*N_CHANNELS-1:0] sram_data_out    // Max-pooling output for all channels
);

    reg [7:0] input_0_0, input_1_0, input_2_0, input_3_0; // Channel 0

    reg [7:0] max_0_1_0, max_2_3_0, max_pool_value_0; // Max values for channel 0

    reg [1:0] read_count;  // Counter to track the number of read operations

    // Block 1: Reading and Storing Data
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            input_0_0 <= 8'd0; input_1_0 <= 8'd0; input_2_0 <= 8'd0; input_3_0 <= 8'd0;
            read_count <= 2'd0;
            reg_full <= 1'b0;
        end else if (en && load_input_reg) begin
            // Increment read count and load inputs based on read count
            read_count <= read_count + 1;
            case (read_count)
                2'd0: begin
                    input_0_0 <= sram_data_in[7:0];
                end
                2'd1: begin
                    input_1_0 <= sram_data_in[7:0];
                end
                2'd2: begin
                    input_2_0 <= sram_data_in[7:0];
                end
                2'd3: begin
                    input_3_0 <= sram_data_in[7:0];
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
            max_0_1_0 <= 8'd0; max_2_3_0 <= 8'd0; max_pool_value_0 <= 8'd0;
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
            sram_data_out[7:0] <= 8'd0;
        end else if (load_output_reg) begin
            sram_data_out[7:0] <= max_pool_value_0;
        end
    end
endmodule

module activation_layer2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter DATA_WIDTH = N_CHANNELS*8,
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
    //wire [8*N_CHANNELS-1:0] sram_data_out;

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
    tanh_activation_parallel_N_CHANNELS_1 #(
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
    parameter DATA_WIDTH = N_CHANNELS*8,
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
    //wire [8*N_CHANNELS-1:0] sram_data_out;

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

module accumulation_layer #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = N_CHANNELS*8,
    parameter DEPTH = 128,   // Memory depth, can be replaced by 2^ADDR_WIDTH
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
    //wire [8*N_CHANNELS-1:0] sram_data_out;

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
    adder_dpe_N_CHANNELS_1 #(
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
    input wire load_input_reg,  // Signal to start filling data into registers
    output reg reg_full,        // Signal indicating all registers are full
    input wire load_output_reg, // Signal to load the output registers
    output reg add_done,        // Internal signal indicating the addition is complete
    input wire [8*N_CHANNELS-1:0] input1,  // First set of 8-bit inputs for all channels
    input wire [8*N_CHANNELS-1:0] input2,  // Second set of 8-bit inputs for all channels
    output reg [8*N_CHANNELS-1:0] output_data  // 8-bit addition output for all channels
);

    reg [7:0] in_data1_0;  // Input storage for first set of inputs for channel 0
    reg [7:0] in_data2_0;  // Input storage for second set of inputs for channel 0
    reg [8:0] sum_0;       // 9-bit sum storage for channel 0 (includes LSB to drop)
    reg [7:0] result_0;    // 8-bit result after dropping LSB for channel 0

    reg [7:0] channel_index;  // Index to track current channel being processed
    reg processing;           // Flag to indicate if processing is ongoing

    // Block 1: Reading and Storing Data
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            in_data1_0 <= 8'd0;
            in_data2_0 <= 8'd0;
            reg_full <= 1'b0;
        end else if (en && load_input_reg) begin
            in_data1_0 <= input1[7:0];
            in_data2_0 <= input2[7:0];
            reg_full <= 1'b1; // Indicate that all registers are full
        end else begin
            reg_full <= 1'b0;
        end
    end

    // Block 2: Addition Logic with LSB Dropped
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            result_0 <= 8'd0;
            channel_index <= 8'd0;
            processing <= 1'b0;
            add_done <= 1'b0;
        end else if (reg_full && !processing) begin
            sum_0 <= in_data1_0 + in_data2_0;
            result_0 <= sum_0[8:1];  // Right shift to drop LSB
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
            output_data[7:0] <= 8'd0;
        end else if (load_output_reg) begin
            output_data[7:0] <= result_0;
        end
    end
endmodule

module activation_layer3 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = N_CHANNELS*8,
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
    //wire [8*N_CHANNELS-1:0] sram_data_out;

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
    tanh_activation_parallel_N_CHANNELS_1 #(
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
    parameter DATA_WIDTH = N_CHANNELS*8,
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
    //wire [8*N_CHANNELS-1:0] sram_data_out;

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
    tanh_activation_parallel_N_CHANNELS_1 #(
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