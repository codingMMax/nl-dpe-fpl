// Top Module
module dpe_one_layer(
    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [(16*8)-1:0] data_in,      
    output wire [(128*8)-1:0] data_out,  
    output wire ready,
    output wire valid_n
);

	wire valid_g_out, ready_g_out, ready_g_in, valid_g_in;
    reg [8:0] read_address, write_address;    // Fix: 9 bits for 512-deep SRAM
    wire [1023:0] data_out_conv1;              // Fix: 1024 bits
    wire [1023:0] global_sram_data_in; 

	// Instantiate the dpe conv_layer
    (* keep = "true" *) conv_layer #(
        .N_CHANNELS(16),
        .ADDR_WIDTH(9),
        .N_KERNELS(128),
        .KERNEL_WIDTH(4),
        .KERNEL_HEIGHT(4),
        .W(32),
        .H(32),
        .S(1),
        .DEPTH(512),
        .DATA_WIDTH(8)
    ) conv1 (
        .clk(clk),
        .rst(rst),
        .valid(valid_g_out),
        .ready_n(ready_g_out), // udpated with ready_Ln
        .data_in(data_in),
        .data_out(data_out_conv1), // this wasn't correct
        .ready(ready_g_in), // or ready_conv1
        .valid_n(valid_g_in)
    );
	
	global_controller #(
    .N_Layers(1)        
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
		.N_CHANNELS(128),
		// .DATA_WIDTH(8), //redundant
		.DEPTH(512)
	) global_sram_inst (
		.clk(clk),
		.rst(rst),
		.w_en(valid_g_in),
		.r_addr(read_address),
		.w_addr(write_address),
		.sram_data_in(data_out_conv1),
		.sram_data_out(global_sram_data_in)
	);
	
	
	always @(posedge clk or posedge rst) begin
		if (rst) begin
            read_address <= 9'b0;             
            write_address <= 9'b0;  
		end else begin
			if(ready_g_out) begin
				read_address <= read_address + 1;
			end
			if(valid_g_out) begin
				write_address <= write_address + 1;
			end
		end
	end

    // Final output connections
    assign data_out = global_sram_data_in;
    // assign ready = ready_g_in;
    assign valid_n = valid_g_in;

  
endmodule

// Top Module
module conv_layer #(
    parameter N_CHANNELS = 16,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 128,
    parameter KERNEL_WIDTH = 4,
    parameter KERNEL_HEIGHT = 4,
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
    // wire [DATA_WIDTH-1:0] sram_data_in;
    wire [DATA_WIDTH-1:0] sram_data_out;
     wire [DATA_WIDTH-1:0] dpe_single_output;

    // Internal registers
    reg [DATA_WIDTH-1:0] sram_data_in;
    reg [$clog2(N_CHANNELS)-1:0] channel_counter;
    reg [ADDR_WIDTH-1:0] w_addr;

    // Output accumulation buffer
    reg [(N_KERNELS*DATA_WIDTH)-1:0] kernel_results;
    reg [$clog2(N_KERNELS)-1:0] kernel_counter;

    // SRAM instance
    sram #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(w_en),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(sram_data_in),
        .sram_data_out(sram_data_out)
    );

    wire [15:0] temp_data_out;
    // Instantiate the DPE module
    dpe dpe_inst (
        .clk(clk),
        .reset(rst),
        .data_in({8'b0, sram_data_out}),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready),
        .data_out(temp_data_out),
        .dpe_done(dpe_done),
        .reg_full(reg_full),
        .shift_add_done(shift_add_done),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl)
    );
    assign dpe_single_output = temp_data_out[DATA_WIDTH-1:0]; 
    
    // Instantiate Controller module
    conv_controller #(
        .N_CHANNELS(N_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_KERNELS(N_KERNELS),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .KERNEL_HEIGHT(KERNEL_HEIGHT),
        .W(W),
        .H(H),
        .S(S),
        .DEPTH(DEPTH)
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

localparam N_CHAN_MIN_1 = N_CHANNELS - 1;
localparam N_KERN_MIN_1 = N_KERNELS - 1;

    // Channel-wise SRAM write logic
always @(posedge clk or posedge rst) begin
    if (rst) begin
        channel_counter <= 0;
        sram_data_in <= 0;
    end else begin
        if (valid && load_input_reg) begin
            w_addr <= write_address;

            case (channel_counter)
                0: sram_data_in <= data_in[7:0];
                1: sram_data_in <= data_in[15:8];
                2: sram_data_in <= data_in[23:16];
                3: sram_data_in <= data_in[31:24];
                4: sram_data_in <= data_in[39:32];
                5: sram_data_in <= data_in[47:40];
                6: sram_data_in <= data_in[55:48];
                7: sram_data_in <= data_in[63:56];
                8: sram_data_in <= data_in[71:64];
                9: sram_data_in <= data_in[79:72];
                10: sram_data_in <= data_in[87:80];
                11: sram_data_in <= data_in[95:88];
                12: sram_data_in <= data_in[103:96];
                13: sram_data_in <= data_in[111:104];
                14: sram_data_in <= data_in[119:112];
                15: sram_data_in <= data_in[127:120];
            endcase

            if (channel_counter == N_CHANNELS - 1) begin
                channel_counter <= 0;
            end else begin
                channel_counter <= channel_counter + 1;
            end
        end
    end
end

    // Accumulate kernel results
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            kernel_counter <= 0;
        end else begin
            if (dpe_done) begin                
                if (kernel_counter == N_KERN_MIN_1) begin
                    kernel_counter <= 0;
                end else begin
                    kernel_counter <= kernel_counter + 1;
                end
            end
        end
    end

always @(posedge clk or posedge rst) begin 
    if (rst) begin 
        kernel_results <= 1024'b0; 
    end else begin 
        if (dpe_done) begin 
            // Case-based MUX for kernel assignment 
            case (kernel_counter) 
                0: kernel_results[7:0] <= dpe_single_output; 
                1: kernel_results[15:8] <= dpe_single_output; 
                2: kernel_results[23:16] <= dpe_single_output; 
                3: kernel_results[31:24] <= dpe_single_output; 
                4: kernel_results[39:32] <= dpe_single_output; 
                5: kernel_results[47:40] <= dpe_single_output; 
                6: kernel_results[55:48] <= dpe_single_output; 
                7: kernel_results[63:56] <= dpe_single_output; 
                8: kernel_results[71:64] <= dpe_single_output; 
                9: kernel_results[79:72] <= dpe_single_output; 
                10: kernel_results[87:80] <= dpe_single_output; 
                11: kernel_results[95:88] <= dpe_single_output; 
                12: kernel_results[103:96] <= dpe_single_output; 
                13: kernel_results[111:104] <= dpe_single_output; 
                14: kernel_results[119:112] <= dpe_single_output; 
                15: kernel_results[127:120] <= dpe_single_output; 
                16: kernel_results[135:128] <= dpe_single_output; 
                17: kernel_results[143:136] <= dpe_single_output; 
                18: kernel_results[151:144] <= dpe_single_output; 
                19: kernel_results[159:152] <= dpe_single_output; 
                20: kernel_results[167:160] <= dpe_single_output; 
                21: kernel_results[175:168] <= dpe_single_output; 
                22: kernel_results[183:176] <= dpe_single_output; 
                23: kernel_results[191:184] <= dpe_single_output; 
                24: kernel_results[199:192] <= dpe_single_output; 
                25: kernel_results[207:200] <= dpe_single_output; 
                26: kernel_results[215:208] <= dpe_single_output; 
                27: kernel_results[223:216] <= dpe_single_output; 
                28: kernel_results[231:224] <= dpe_single_output; 
                29: kernel_results[239:232] <= dpe_single_output; 
                30: kernel_results[247:240] <= dpe_single_output; 
                31: kernel_results[255:248] <= dpe_single_output; 
                32: kernel_results[263:256] <= dpe_single_output; 
                33: kernel_results[271:264] <= dpe_single_output; 
                34: kernel_results[279:272] <= dpe_single_output; 
                35: kernel_results[287:280] <= dpe_single_output; 
                36: kernel_results[295:288] <= dpe_single_output; 
                37: kernel_results[303:296] <= dpe_single_output; 
                38: kernel_results[311:304] <= dpe_single_output; 
                39: kernel_results[319:312] <= dpe_single_output; 
                40: kernel_results[327:320] <= dpe_single_output; 
                41: kernel_results[335:328] <= dpe_single_output; 
                42: kernel_results[343:336] <= dpe_single_output; 
                43: kernel_results[351:344] <= dpe_single_output; 
                44: kernel_results[359:352] <= dpe_single_output; 
                45: kernel_results[367:360] <= dpe_single_output; 
                46: kernel_results[375:368] <= dpe_single_output; 
                47: kernel_results[383:376] <= dpe_single_output; 
                48: kernel_results[391:384] <= dpe_single_output; 
                49: kernel_results[399:392] <= dpe_single_output; 
                50: kernel_results[407:400] <= dpe_single_output; 
                51: kernel_results[415:408] <= dpe_single_output; 
                52: kernel_results[423:416] <= dpe_single_output; 
                53: kernel_results[431:424] <= dpe_single_output; 
                54: kernel_results[439:432] <= dpe_single_output; 
                55: kernel_results[447:440] <= dpe_single_output; 
                56: kernel_results[455:448] <= dpe_single_output; 
                57: kernel_results[463:456] <= dpe_single_output; 
                58: kernel_results[471:464] <= dpe_single_output; 
                59: kernel_results[479:472] <= dpe_single_output; 
                60: kernel_results[487:480] <= dpe_single_output; 
                61: kernel_results[495:488] <= dpe_single_output; 
                62: kernel_results[503:496] <= dpe_single_output; 
                63: kernel_results[511:504] <= dpe_single_output; 
                64: kernel_results[519:512] <= dpe_single_output; 
                65: kernel_results[527:520] <= dpe_single_output; 
                66: kernel_results[535:528] <= dpe_single_output; 
                67: kernel_results[543:536] <= dpe_single_output; 
                68: kernel_results[551:544] <= dpe_single_output; 
                69: kernel_results[559:552] <= dpe_single_output; 
                70: kernel_results[567:560] <= dpe_single_output; 
                71: kernel_results[575:568] <= dpe_single_output; 
                72: kernel_results[583:576] <= dpe_single_output; 
                73: kernel_results[591:584] <= dpe_single_output; 
                74: kernel_results[599:592] <= dpe_single_output; 
                75: kernel_results[607:600] <= dpe_single_output; 
                76: kernel_results[615:608] <= dpe_single_output; 
                77: kernel_results[623:616] <= dpe_single_output; 
                78: kernel_results[631:624] <= dpe_single_output; 
                79: kernel_results[639:632] <= dpe_single_output; 
                80: kernel_results[647:640] <= dpe_single_output; 
                81: kernel_results[655:648] <= dpe_single_output; 
                82: kernel_results[663:656] <= dpe_single_output; 
                83: kernel_results[671:664] <= dpe_single_output; 
                84: kernel_results[679:672] <= dpe_single_output; 
                85: kernel_results[687:680] <= dpe_single_output; 
                86: kernel_results[695:688] <= dpe_single_output; 
                87: kernel_results[703:696] <= dpe_single_output; 
                88: kernel_results[711:704] <= dpe_single_output; 
                89: kernel_results[719:712] <= dpe_single_output; 
                90: kernel_results[727:720] <= dpe_single_output; 
                91: kernel_results[735:728] <= dpe_single_output; 
                92: kernel_results[743:736] <= dpe_single_output; 
                93: kernel_results[751:744] <= dpe_single_output; 
                94: kernel_results[759:752] <= dpe_single_output; 
                95: kernel_results[767:760] <= dpe_single_output; 
                96: kernel_results[775:768] <= dpe_single_output; 
                97: kernel_results[783:776] <= dpe_single_output; 
                98: kernel_results[791:784] <= dpe_single_output; 
                99: kernel_results[799:792] <= dpe_single_output; 
                100: kernel_results[807:800] <= dpe_single_output; 
                101: kernel_results[815:808] <= dpe_single_output; 
                102: kernel_results[823:816] <= dpe_single_output; 
                103: kernel_results[831:824] <= dpe_single_output; 
                104: kernel_results[839:832] <= dpe_single_output; 
                105: kernel_results[847:840] <= dpe_single_output; 
                106: kernel_results[855:848] <= dpe_single_output; 
                107: kernel_results[863:856] <= dpe_single_output; 
                108: kernel_results[871:864] <= dpe_single_output; 
                109: kernel_results[879:872] <= dpe_single_output; 
                110: kernel_results[887:880] <= dpe_single_output; 
                111: kernel_results[895:888] <= dpe_single_output; 
                112: kernel_results[903:896] <= dpe_single_output; 
                113: kernel_results[911:904] <= dpe_single_output; 
                114: kernel_results[919:912] <= dpe_single_output; 
                115: kernel_results[927:920] <= dpe_single_output; 
                116: kernel_results[935:928] <= dpe_single_output; 
                117: kernel_results[943:936] <= dpe_single_output; 
                118: kernel_results[951:944] <= dpe_single_output; 
                119: kernel_results[959:952] <= dpe_single_output; 
                120: kernel_results[967:960] <= dpe_single_output; 
                121: kernel_results[975:968] <= dpe_single_output; 
                122: kernel_results[983:976] <= dpe_single_output; 
                123: kernel_results[991:984] <= dpe_single_output; 
                124: kernel_results[999:992] <= dpe_single_output; 
                125: kernel_results[1007:1000] <= dpe_single_output; 
                126: kernel_results[1015:1008] <= dpe_single_output; 
                127: kernel_results[1023:1016] <= dpe_single_output; 
                // default: kernel_results <= kernel_results; // No change 
            endcase 
        end 
    end 
end 

    // Output the accumulated results
    assign data_out = kernel_results;


endmodule

module conv_controller #(
    parameter N_CHANNELS = 16,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 128,
    parameter KERNEL_WIDTH = 4,
    parameter KERNEL_HEIGHT = 4,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter S_BITWIDTH = $clog2(W + S),
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

    output wire [ADDR_WIDTH-1:0] read_address,
    output wire [ADDR_WIDTH-1:0] write_address,
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

    // Constants
    localparam PATCH_SIZE = N_CHANNELS * KERNEL_WIDTH * KERNEL_HEIGHT;

    // State Machine
localparam IDLE        = 2'd0;
localparam LOAD_PATCH  = 2'd1;
localparam RUN_DPE     = 2'd2;
localparam OUTPUT_RESULT = 2'd3;

reg [1:0] state, next_state;

    // Counters
    reg [$clog2(PATCH_SIZE)-1:0] patch_counter;
    reg [$clog2(N_KERNELS)-1:0] kernel_counter;

    // Spatial counters
    reg [$clog2(KERNEL_WIDTH)-1:0] n;
    reg [$clog2(KERNEL_HEIGHT)-1:0] m;
    reg [$clog2(W + S)-1:0] s;
    reg [$clog2(H + S)-1:0] sv;
    
    // Internal registers
    reg [ADDR_WIDTH-1:0] read_address_reg;
    reg [ADDR_WIDTH-1:0] write_address_reg;
    reg [ADDR_WIDTH-1:0] pointer_offset;

    // Output assignments
    assign read_address = read_address_reg;
    assign write_address = write_address_reg;
    assign load_input_reg = (state == LOAD_PATCH && patch_counter == PATCH_SIZE - 1);

        // Memory flag: full/empty status
    wire memory_flag;
    assign memory_flag = (write_address_reg > read_address_reg) || 
                         ((write_address_reg == 0) && (read_address_reg == DEPTH - 1));

    // FSM Logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            next_state <= IDLE;

            patch_counter <= 0;
            kernel_counter <= 0;

            read_address_reg <= 0;
            write_address_reg <= 0;

            n <= 0;
            m <= 0;
            s <= 0;
            sv <= 0;

            // w_buf_en <= 0;
            // nl_dpe_control <= 2'b00;
            // shift_add_control <= 0;
            // shift_add_bypass <= 0;
            // load_output_reg <= 0;
            w_en <= 0;
            ready <= 1'b1;
            valid_n <= 1'b0;

        end else begin
            state <= next_state;

            case (state)
                IDLE: begin
                    ready <= 1'b1;
                    valid_n <= 1'b0;

                    if (valid) begin
                        next_state <= LOAD_PATCH;
                        patch_counter <= 0;
                        w_en <= 1'b1;
                        write_address_reg <= 0;
                        n <= 0;
                        m <= 0;
                        s <= 0;
                        sv <= 0;
                    end else begin
                        next_state <= IDLE;
                    end
                end

                LOAD_PATCH: begin
                    ready <= 1'b0;

                    write_address_reg <= write_address_reg + 1;
                    pointer_offset <= (n + W * (m + sv) + s * S) * N_CHANNELS;

                    if (patch_counter == PATCH_SIZE - 1) begin
                        w_en <= 1'b0;
                        next_state <= RUN_DPE;
                    end else begin
                        patch_counter <= patch_counter + 1;

                        // Update spatial counters
                        if (n < KERNEL_WIDTH - 1) begin
                            n <= n + 1;
                        end else if (m < KERNEL_HEIGHT - 1) begin
                            n <= 0;
                            m <= m + 1;
                        end else if (s < W - KERNEL_WIDTH + S - 1) begin
                            n <= 0;
                            m <= 0;
                            s <= s + 1;
                        end else if (sv < H - KERNEL_HEIGHT + S - 1) begin
                            n <= 0;
                            m <= 0;
                            s <= 0;
                            sv <= sv + 1;
                        end else begin
                            n <= 0;
                            m <= 0;
                            s <= 0;
                            sv <= 0;
                        end
                    end
                end

                RUN_DPE: begin
                    ready <= 1'b0;
                    if (memory_flag) begin
                        read_address_reg <= read_address_reg + 1;
                    end
                    if (dpe_done) begin
                        next_state <= OUTPUT_RESULT;
                    end
                end

                OUTPUT_RESULT: begin
                    valid_n <= 1'b1;
                    read_address_reg <= 0;

                    if (kernel_counter == N_KERNELS - 1) begin
                        kernel_counter <= 0;
                        next_state <= IDLE;
                    end else begin
                        kernel_counter <= kernel_counter + 1;
                        next_state <= LOAD_PATCH;
                        patch_counter <= 0;
                        write_address_reg <= write_address_reg + 1;
                    end
                end
            endcase
        end
    end

    // Control Signal Assignments
    always @* begin
        case (state)
            IDLE: begin
                w_buf_en = 1'b0;
                nl_dpe_control = 2'b00;
                shift_add_control = 1'b0;
                // shift_add_bypass = shift_add_bypass_ctrl;
                shift_add_bypass = 1'b0;
                load_output_reg = 1'b0;
            end

            LOAD_PATCH: begin
                w_buf_en = 1'b0;
                nl_dpe_control = 2'b00;
                shift_add_control = 1'b0;
                shift_add_bypass = shift_add_bypass_ctrl;
                load_output_reg = 1'b0;
            end

            RUN_DPE: begin
                w_buf_en = 1'b1;
                nl_dpe_control = 2'b11;
                shift_add_control = MSB_SA_Ready;
                shift_add_bypass = shift_add_bypass_ctrl;
                load_output_reg = dpe_done;
            end

            OUTPUT_RESULT: begin
                w_buf_en = 1'b0;
                nl_dpe_control = 2'b00;
                shift_add_control = MSB_SA_Ready;
                shift_add_bypass = shift_add_bypass_ctrl;
                load_output_reg = dpe_done;
            end

            // default: begin
            //     w_buf_en = 1'b0;
            //     nl_dpe_control = 2'b00;
            //     shift_add_control = 1'b0;
            //     shift_add_bypass = 1'b0;
            //     load_output_reg = 1'b0;
            // end
        endcase
    end

endmodule

module sram #(
    parameter N_CHANNELS = 128,
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



module global_controller #(
    parameter N_Layers = 1        
)(
    input wire clk,              
    input wire rst,             
    input wire ready_L1,     // Ready from downstream
    input wire valid_Ln,     // Valid from next layer (added dependency)
    input wire valid,        // Valid from upstream
    output reg ready,       // Ready to upstream
    output reg valid_L1,    // Valid to downstream
    output reg ready_Ln     // Ready to next layer
);

    (* keep = "true" *) reg stall;
    reg valid_Ln_sync1, valid_Ln_sync2;

    // Synchronize valid_Ln to avoid metastability
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_Ln_sync1 <= 1'b0;
            valid_Ln_sync2 <= 1'b0;
        end else begin
            valid_Ln_sync1 <= valid_Ln;
            valid_Ln_sync2 <= valid_Ln_sync1;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            stall <= 0;
            ready_Ln <= 1'b0;
        end else begin
            if (valid && !ready_L1 && !valid_Ln_sync2) begin
                stall <= 1; // Stall if upstream is valid, downstream isn't ready, and next layer isn't ready
            end else if (!valid) begin
                stall <= 0;
            end
            ready_Ln <= ~stall && !valid_Ln_sync2; // Only ready if not stalled AND next layer is ready
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_L1 <= 1'b0;
            ready <= 1'b0;
        end else begin
            valid_L1 <= valid;  // Sync valid to downstream
            ready <= ready_L1;  // Sync ready to upstream
        end
    end

endmodule