// NL-DPE GEMV (y = Wx) RTL - parameterizable for VTR synthesis
// Maps a single GEMV layer to NL-DPE crossbar tiles (256x256 each)
// Supports 5 configurations:
//   Small     (K=64,  N=64):   V=1, H=1 -> 1 DPE
//   Medium-K  (K=512, N=128):  V=2, H=1 -> 2 DPEs
//   Large-K   (K=2048,N=256):  V=8, H=1 -> 8 DPEs
//   Wide-N    (K=256, N=512):  V=1, H=2 -> 2 DPEs
//   Square-lg (K=512, N=512):  V=2, H=2 -> 4 DPEs
//
// GEMV maps to conv_layer pattern: KERNEL_WIDTH=K, KERNEL_HEIGHT=1, W=1, H=1, S=1
// No pooling or activation — just the matrix-vector multiply.

module gemv #(
    parameter K = 64,           // Input dimension (rows of W)
    parameter N = 64,           // Output dimension (cols of W)
    parameter ROWS = 256,       // Crossbar rows (Q1 sweep: 128, 256, 512)
    parameter COLS = 256,       // Crossbar cols (Q1 sweep: 128, 256, 512)
    parameter DATA_WIDTH = 16
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

    // Tile counts
    localparam V = (K + ROWS - 1) / ROWS;  // ceil(K/ROWS) — vertical stacking (rows)
    localparam H = (N + COLS - 1) / COLS;   // ceil(N/COLS) — horizontal stacking (cols)

    // SRAM depth: at least K entries to buffer the input vector
    localparam DEPTH = (K < 512) ? 512 : K;
    localparam ADDR_WIDTH = $clog2(DEPTH);

    // Internal signals for global controller and output SRAM
    wire valid_g_in, valid_g_out, ready_g_in, ready_g_out;
    wire [DATA_WIDTH-1:0] data_out_gemv_layer;
    wire [DATA_WIDTH-1:0] global_sram_data_in;
    reg [7:0] read_address, write_address;

    // =========================================================================
    // GEMV layer instantiation via generate — selects correct DPE tiling
    // =========================================================================
    generate
        if (V == 1 && H == 1) begin : single_dpe
            // 1 DPE tile (e.g. K=64, N=64)
            conv_layer_single_dpe #(
                .N_CHANNELS(1),
                .ADDR_WIDTH(ADDR_WIDTH),
                .N_KERNELS(1),
                .KERNEL_WIDTH(K),
                .KERNEL_HEIGHT(1),
                .W(1),
                .H(1),
                .S(1),
                .DEPTH(DEPTH),
                .DATA_WIDTH(DATA_WIDTH)
            ) gemv_layer (
                .clk(clk),
                .rst(rst),
                .valid(valid_g_out),
                .ready_n(ready_g_out),
                .data_in(data_in),
                .data_out(data_out_gemv_layer),
                .ready(ready_g_in),
                .valid_n(valid_g_in)
            );
        end else if (V == 2 && H == 1) begin : v2_h1
            // 2 DPE tiles vertical (e.g. K=512, N=128)
            conv_layer_stacked_dpes_V2_H1 #(
                .N_CHANNELS(1),
                .ADDR_WIDTH(ADDR_WIDTH),
                .N_KERNELS(1),
                .KERNEL_WIDTH(K),
                .KERNEL_HEIGHT(1),
                .W(1),
                .H(1),
                .S(1),
                .N_DPE_V(2),
                .N_DPE_H(1),
                .N_BRAM_R(1),
                .N_BRAM_W(1),
                .DATA_WIDTH(DATA_WIDTH),
                .DEPTH(DEPTH)
            ) gemv_layer (
                .clk(clk),
                .rst(rst),
                .valid(valid_g_out),
                .ready_n(ready_g_out),
                .data_in(data_in),
                .data_out(data_out_gemv_layer),
                .ready(ready_g_in),
                .valid_n(valid_g_in)
            );
        end else if (V <= 4 && H == 1) begin : v4_h1
            // 4 DPE tiles vertical (e.g. K=1024, N=256)
            conv_layer_stacked_dpes_V4_H1 #(
                .N_CHANNELS(1),
                .ADDR_WIDTH(ADDR_WIDTH),
                .N_KERNELS(1),
                .KERNEL_WIDTH(K),
                .KERNEL_HEIGHT(1),
                .W(1),
                .H(1),
                .S(1),
                .N_DPE_V(4),
                .N_DPE_H(1),
                .N_BRAM_R(1),
                .N_BRAM_W(1),
                .DATA_WIDTH(DATA_WIDTH),
                .DEPTH(DEPTH)
            ) gemv_layer (
                .clk(clk),
                .rst(rst),
                .valid(valid_g_out),
                .ready_n(ready_g_out),
                .data_in(data_in),
                .data_out(data_out_gemv_layer),
                .ready(ready_g_in),
                .valid_n(valid_g_in)
            );
        end else if (V <= 8 && H == 1) begin : v8_h1
            // 8 DPE tiles vertical (e.g. K=2048, N=256)
            conv_layer_stacked_dpes_V8_H1 #(
                .N_CHANNELS(1),
                .ADDR_WIDTH(ADDR_WIDTH),
                .N_KERNELS(1),
                .KERNEL_WIDTH(K),
                .KERNEL_HEIGHT(1),
                .W(1),
                .H(1),
                .S(1),
                .N_DPE_V(8),
                .N_DPE_H(1),
                .N_BRAM_R(1),
                .N_BRAM_W(1),
                .DATA_WIDTH(DATA_WIDTH),
                .DEPTH(DEPTH)
            ) gemv_layer (
                .clk(clk),
                .rst(rst),
                .valid(valid_g_out),
                .ready_n(ready_g_out),
                .data_in(data_in),
                .data_out(data_out_gemv_layer),
                .ready(ready_g_in),
                .valid_n(valid_g_in)
            );
        end else if (V == 1 && H == 2) begin : v1_h2
            // 2 DPE tiles horizontal (e.g. K=256, N=512)
            conv_layer_stacked_dpes_V1_H2 #(
                .N_CHANNELS(1),
                .ADDR_WIDTH(ADDR_WIDTH),
                .N_KERNELS(1),
                .KERNEL_WIDTH(K),
                .KERNEL_HEIGHT(1),
                .W(1),
                .H(1),
                .S(1),
                .DEPTH(DEPTH),
                .DATA_WIDTH(DATA_WIDTH)
            ) gemv_layer (
                .clk(clk),
                .rst(rst),
                .valid(valid_g_out),
                .ready_n(ready_g_out),
                .data_in(data_in),
                .data_out(data_out_gemv_layer),
                .ready(ready_g_in),
                .valid_n(valid_g_in)
            );
        end else if (V == 2 && H == 2) begin : v2_h2
            // 4 DPE tiles in 2x2 grid (e.g. K=512, N=512)
            conv_layer_stacked_dpes_V2_H2 #(
                .N_CHANNELS(1),
                .ADDR_WIDTH(ADDR_WIDTH),
                .N_KERNELS(1),
                .KERNEL_WIDTH(K),
                .KERNEL_HEIGHT(1),
                .W(1),
                .H(1),
                .S(1),
                .DEPTH(DEPTH),
                .DATA_WIDTH(DATA_WIDTH)
            ) gemv_layer (
                .clk(clk),
                .rst(rst),
                .valid(valid_g_out),
                .ready_n(ready_g_out),
                .data_in(data_in),
                .data_out(data_out_gemv_layer),
                .ready(ready_g_in),
                .valid_n(valid_g_in)
            );
        end
    endgenerate

    // =========================================================================
    // Global controller — handshaking between top-level and GEMV layer
    // =========================================================================
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

    // =========================================================================
    // Global output SRAM buffer
    // =========================================================================
    sram #(
        .N_CHANNELS(1),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(16)
    ) global_sram_inst (
        .clk(clk),
        .rst(rst),
        .w_en(valid_g_in),
        .r_addr(read_address),
        .w_addr(write_address),
        .sram_data_in(data_out_gemv_layer),
        .sram_data_out(global_sram_data_in)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            read_address <= 0;
            write_address <= 16;
        end else begin
            if (ready_g_out) begin
                read_address <= read_address + 1;
            end else begin
                read_address <= read_address;
            end
            if (valid_g_out) begin
                write_address <= write_address + 1;
            end else begin
                write_address <= write_address;
            end
        end
    end

    // Final output connections
    assign data_out = global_sram_data_in;
    assign ready = ready_g_in;
    assign valid_n = valid_g_in;

endmodule


// =============================================================================
// conv_layer_single_dpe — single DPE tile (V=1, H=1)
// =============================================================================
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


// =============================================================================
// conv_controller — kernel scan controller for single DPE
// =============================================================================
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


// =============================================================================
// conv_layer_stacked_dpes_V2_H1 — 2 DPEs vertical, 1 horizontal
// =============================================================================
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


// =============================================================================
// conv_layer_stacked_dpes_V4_H1 — 4 DPEs vertical, 1 horizontal
// =============================================================================
module conv_layer_stacked_dpes_V4_H1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 4,
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

	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out1, data_out2, data_out3, data_out4;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_temp1, data_out_temp2;
	reg [(N_KERNELS*DATA_WIDTH)-1:0] data_out_part1, data_out_part2;
	wire [N_BRAM_W-1:0] w_en_dec_signal;
	wire [N_DPE_V-1:0] dpe_sel_signal;
	wire [N_DPE_H-1:0] dpe_sel_h_signal;
	wire [DATA_WIDTH-1:0] sram_data;
	wire dpe_done1, dpe_done2, dpe_done3, dpe_done4;
	wire accum_ready, accum_done;
	wire accum_done1, accum_done2;
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

    // Instantiate the DPE modules (16-bit direct, no zero padding)
    // 4 DPEs vertical, 1 horizontal
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

	dpe dpe_R3_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready3),
        .data_out(data_out3),
        .dpe_done(dpe_done3),
        .reg_full(reg_full_sig[2]),
        .shift_add_done(shift_add_done3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl3)
    );

	dpe dpe_R4_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready4),
        .data_out(data_out4),
        .dpe_done(dpe_done4),
        .reg_full(reg_full_sig[3]),
        .shift_add_done(shift_add_done4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl4)
    );

	assign shift_add_done = shift_add_done1 & shift_add_done2 & shift_add_done3 & shift_add_done4;
	assign shift_add_bypass_ctrl = shift_add_bypass_ctrl1 & shift_add_bypass_ctrl2 & shift_add_bypass_ctrl3 & shift_add_bypass_ctrl4;
	assign dpe_done = dpe_done1 & dpe_done2 & dpe_done3 & dpe_done4;
	assign MSB_SA_Ready = MSB_SA_Ready1 & MSB_SA_Ready2 & MSB_SA_Ready3 & MSB_SA_Ready4;

	// Adder tree: Level 1 - pair adders
  	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R1_R2 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done1),
        .input1(data_out1),
        .input2(data_out2),
        .output_data(data_out_temp1)
    );

	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R3_R4 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done2),
        .input1(data_out3),
        .input2(data_out4),
        .output_data(data_out_temp2)
    );

	assign accum_done = accum_done1 & accum_done2;

	// Level 2 - registered final sum
	always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
            data_out_part1 <= 0;
            data_out_part2 <= 0;
        end else begin
			data_out_part1 <= data_out_temp1;
			data_out_part2 <= data_out_temp2;
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
		.load_input_reg(load_input_reg),
		.ready(ready),
        .valid_n(valid_n),
		.dpe_accum_ready(accum_ready),
		.dpe_sel(dpe_sel_signal),
		.dpe_sel_h(dpe_sel_h_signal)
    );

endmodule


// =============================================================================
// conv_layer_stacked_dpes_V8_H1 — 8 DPEs vertical, 1 horizontal
// =============================================================================
module conv_layer_stacked_dpes_V8_H1 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter N_DPE_V = 8,
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

	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out1, data_out2, data_out3, data_out4;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out5, data_out6, data_out7, data_out8;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
	wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_temp1, data_out_temp2, data_out_temp3, data_out_temp4;
	reg [(N_KERNELS*DATA_WIDTH)-1:0] data_out_temp1_r, data_out_temp2_r, data_out_temp3_r, data_out_temp4_r;
	reg [(N_KERNELS*DATA_WIDTH)-1:0] data_out_part1, data_out_part2;
	wire [N_BRAM_W-1:0] w_en_dec_signal;
	wire [N_DPE_V-1:0] dpe_sel_signal;
	wire [N_DPE_H-1:0] dpe_sel_h_signal;
	wire [DATA_WIDTH-1:0] sram_data;
	wire dpe_done1, dpe_done2, dpe_done3, dpe_done4;
	wire dpe_done5, dpe_done6, dpe_done7, dpe_done8;
	wire accum_ready, accum_done;
	wire accum_done1, accum_done2, accum_done3, accum_done4;
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
        .NUM_OUTPUTS(8)
    ) u_xbar (
        .in_data(sram_data),
        .in_sel(N_BRAM_R),
        .out_sel(dpe_sel_signal),
        .out_data(dpe_data)
    );

    // Instantiate the DPE modules (16-bit direct, no zero padding)
    // 8 DPEs vertical, 1 horizontal
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

	dpe dpe_R3_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready3),
        .data_out(data_out3),
        .dpe_done(dpe_done3),
        .reg_full(reg_full_sig[2]),
        .shift_add_done(shift_add_done3),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl3)
    );

	dpe dpe_R4_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready4),
        .data_out(data_out4),
        .dpe_done(dpe_done4),
        .reg_full(reg_full_sig[3]),
        .shift_add_done(shift_add_done4),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl4)
    );

	dpe dpe_R5_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready5),
        .data_out(data_out5),
        .dpe_done(dpe_done5),
        .reg_full(reg_full_sig[4]),
        .shift_add_done(shift_add_done5),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl5)
    );

	dpe dpe_R6_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready6),
        .data_out(data_out6),
        .dpe_done(dpe_done6),
        .reg_full(reg_full_sig[5]),
        .shift_add_done(shift_add_done6),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl6)
    );

	dpe dpe_R7_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready7),
        .data_out(data_out7),
        .dpe_done(dpe_done7),
        .reg_full(reg_full_sig[6]),
        .shift_add_done(shift_add_done7),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl7)
    );

	dpe dpe_R8_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready8),
        .data_out(data_out8),
        .dpe_done(dpe_done8),
        .reg_full(reg_full_sig[7]),
        .shift_add_done(shift_add_done8),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl8)
    );

	assign shift_add_done = shift_add_done1 & shift_add_done2 & shift_add_done3 & shift_add_done4 &
							shift_add_done5 & shift_add_done6 & shift_add_done7 & shift_add_done8;
	assign shift_add_bypass_ctrl = shift_add_bypass_ctrl1 & shift_add_bypass_ctrl2 & shift_add_bypass_ctrl3 & shift_add_bypass_ctrl4 &
								   shift_add_bypass_ctrl5 & shift_add_bypass_ctrl6 & shift_add_bypass_ctrl7 & shift_add_bypass_ctrl8;
	assign dpe_done = dpe_done1 & dpe_done2 & dpe_done3 & dpe_done4 &
					  dpe_done5 & dpe_done6 & dpe_done7 & dpe_done8;
	assign MSB_SA_Ready = MSB_SA_Ready1 & MSB_SA_Ready2 & MSB_SA_Ready3 & MSB_SA_Ready4 &
						  MSB_SA_Ready5 & MSB_SA_Ready6 & MSB_SA_Ready7 & MSB_SA_Ready8;

	// Adder tree: Level 1 - 4 pair adders
  	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R1_R2 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done1),
        .input1(data_out1),
        .input2(data_out2),
        .output_data(data_out_temp1)
    );

	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R3_R4 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done2),
        .input1(data_out3),
        .input2(data_out4),
        .output_data(data_out_temp2)
    );

	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R5_R6 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done3),
        .input1(data_out5),
        .input2(data_out6),
        .output_data(data_out_temp3)
    );

	adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_inst_R7_R8 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done4),
        .input1(data_out7),
        .input2(data_out8),
        .output_data(data_out_temp4)
    );

	assign accum_done = accum_done1 & accum_done2 & accum_done3 & accum_done4;

	// Adder tree: Level 2 - pipelined with buffer registers to break critical path
	// Stage 1: Register Level 1 adder outputs (breaks routing from DPE pairs to central adder)
	// Stage 2: Registered second-level sums
	// Stage 3: Registered final sum
	always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
            data_out_temp1_r <= 0;
            data_out_temp2_r <= 0;
            data_out_temp3_r <= 0;
            data_out_temp4_r <= 0;
            data_out_part1 <= 0;
            data_out_part2 <= 0;
        end else begin
			// Stage 1: buffer register (breaks long routing path)
			data_out_temp1_r <= data_out_temp1;
			data_out_temp2_r <= data_out_temp2;
			data_out_temp3_r <= data_out_temp3;
			data_out_temp4_r <= data_out_temp4;
			// Stage 2: second-level sums (add from registered values)
			data_out_part1 <= data_out_temp1_r + data_out_temp2_r;
			data_out_part2 <= data_out_temp3_r + data_out_temp4_r;
			// Stage 3: final sum
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
		.load_input_reg(load_input_reg),
		.ready(ready),
        .valid_n(valid_n),
		.dpe_accum_ready(accum_ready),
		.dpe_sel(dpe_sel_signal),
		.dpe_sel_h(dpe_sel_h_signal)
    );

endmodule


// =============================================================================
// conv_layer_stacked_dpes_V1_H2 — 2 DPEs horizontal (V=1, H=2)
// For N > 256: two column-groups, each single DPE handles 256 columns.
// Outputs are sequenced (not summed) — first group processes, then second.
// For VTR: exposes 2 DPE hard block instances.
// =============================================================================
module conv_layer_stacked_dpes_V1_H2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
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

    // DPE outputs
    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_c1;
    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_c2;
    wire dpe_done_c1, dpe_done_c2;
    wire reg_full_c1, reg_full_c2;
    wire shift_add_done_c1, shift_add_done_c2;
    wire shift_add_bypass_ctrl_c1, shift_add_bypass_ctrl_c2;
    wire MSB_SA_Ready_c1, MSB_SA_Ready_c2;

    // Column group sequencing: alternate between column groups
    reg col_group_sel;  // 0 = first column group, 1 = second

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            col_group_sel <= 1'b0;
        end else begin
            if (dpe_done) begin
                col_group_sel <= ~col_group_sel;
            end
        end
    end

    // Instantiate the SRAM module (shared input buffer)
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

    // DPE Column 1
    dpe dpe_R1_C1 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c1),
        .data_out(data_out_c1),
        .dpe_done(dpe_done_c1),
        .reg_full(reg_full_c1),
        .shift_add_done(shift_add_done_c1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c1)
    );

    // DPE Column 2
    dpe dpe_R1_C2 (
        .clk(clk),
        .reset(rst),
        .data_in(sram_data_out),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_c2),
        .data_out(data_out_c2),
        .dpe_done(dpe_done_c2),
        .reg_full(reg_full_c2),
        .shift_add_done(shift_add_done_c2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_c2)
    );

    // Horizontal: outputs from different column groups are NOT summed
    // Mux output based on which column group completed
    assign data_out = col_group_sel ? data_out_c2 : data_out_c1;

    // Synchronize control signals across both DPEs
    assign shift_add_done = shift_add_done_c1 & shift_add_done_c2;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_c1 & shift_add_bypass_ctrl_c2;
    assign dpe_done = dpe_done_c1 & dpe_done_c2;
    assign MSB_SA_Ready = MSB_SA_Ready_c1 & MSB_SA_Ready_c2;
    assign reg_full = reg_full_c1 & reg_full_c2;

    // Instantiate the Controller module (single-DPE controller works here)
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


// =============================================================================
// conv_layer_stacked_dpes_V2_H2 — 4 DPEs in 2x2 grid (V=2, H=2)
// For K > 256 AND N > 256: two vertical pairs, each pair handles 256 columns.
// Vertical outputs are summed (adder tree), horizontal outputs are sequenced.
// For VTR: exposes 4 DPE hard block instances.
// =============================================================================
module conv_layer_stacked_dpes_V2_H2 #(
    parameter N_CHANNELS = 1,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 1,
    parameter KERNEL_WIDTH = 3,
    parameter KERNEL_HEIGHT = 3,
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

    localparam N_DPE_V = 2;
    localparam N_DPE_H = 2;
    localparam N_BRAM_R = 1;
    localparam N_BRAM_W = 1;

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

    wire [DATA_WIDTH-1:0] sram_data;
    wire [(N_KERNELS*DATA_WIDTH)-1:0] dpe_data;
    wire [N_BRAM_W-1:0] w_en_dec_signal;
    wire [N_DPE_V-1:0] dpe_sel_signal;
    wire [N_DPE_H-1:0] dpe_sel_h_signal;

    // DPE outputs — 2x2 grid: R{row}_C{col}
    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_r1_c1, data_out_r2_c1;
    wire [(N_KERNELS*DATA_WIDTH)-1:0] data_out_r1_c2, data_out_r2_c2;
    wire dpe_done_r1_c1, dpe_done_r2_c1, dpe_done_r1_c2, dpe_done_r2_c2;
    wire shift_add_done_r1_c1, shift_add_done_r2_c1, shift_add_done_r1_c2, shift_add_done_r2_c2;
    wire shift_add_bypass_ctrl_r1_c1, shift_add_bypass_ctrl_r2_c1;
    wire shift_add_bypass_ctrl_r1_c2, shift_add_bypass_ctrl_r2_c2;
    wire MSB_SA_Ready_r1_c1, MSB_SA_Ready_r2_c1, MSB_SA_Ready_r1_c2, MSB_SA_Ready_r2_c2;

    // Adder outputs for vertical summing within each column group
    wire [(N_KERNELS*DATA_WIDTH)-1:0] col1_sum, col2_sum;
    wire accum_ready, accum_done;
    wire accum_done_c1, accum_done_c2;

    // Column group sequencing
    reg col_group_sel;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            col_group_sel <= 1'b0;
        end else begin
            if (dpe_done) begin
                col_group_sel <= ~col_group_sel;
            end
        end
    end

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

    // Instantiate crossbar module for inputs (fan out to V=2 DPEs)
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

    // DPE Row 1, Col 1
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
        .MSB_SA_Ready(MSB_SA_Ready_r1_c1),
        .data_out(data_out_r1_c1),
        .dpe_done(dpe_done_r1_c1),
        .reg_full(reg_full_sig[0]),
        .shift_add_done(shift_add_done_r1_c1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_r1_c1)
    );

    // DPE Row 2, Col 1
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
        .MSB_SA_Ready(MSB_SA_Ready_r2_c1),
        .data_out(data_out_r2_c1),
        .dpe_done(dpe_done_r2_c1),
        .reg_full(reg_full_sig[1]),
        .shift_add_done(shift_add_done_r2_c1),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_r2_c1)
    );

    // DPE Row 1, Col 2
    dpe dpe_R1_C2 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_r1_c2),
        .data_out(data_out_r1_c2),
        .dpe_done(dpe_done_r1_c2),
        .reg_full(),
        .shift_add_done(shift_add_done_r1_c2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_r1_c2)
    );

    // DPE Row 2, Col 2
    dpe dpe_R2_C2 (
        .clk(clk),
        .reset(rst),
        .data_in(dpe_data),
        .nl_dpe_control(nl_dpe_control),
        .shift_add_control(shift_add_control),
        .w_buf_en(w_buf_en),
        .shift_add_bypass(shift_add_bypass),
        .load_output_reg(load_output_reg),
        .load_input_reg(load_input_reg),
        .MSB_SA_Ready(MSB_SA_Ready_r2_c2),
        .data_out(data_out_r2_c2),
        .dpe_done(dpe_done_r2_c2),
        .reg_full(),
        .shift_add_done(shift_add_done_r2_c2),
        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_r2_c2)
    );

    // Combine status signals across all 4 DPEs
    assign shift_add_done = shift_add_done_r1_c1 & shift_add_done_r2_c1 &
                            shift_add_done_r1_c2 & shift_add_done_r2_c2;
    assign shift_add_bypass_ctrl = shift_add_bypass_ctrl_r1_c1 & shift_add_bypass_ctrl_r2_c1 &
                                   shift_add_bypass_ctrl_r1_c2 & shift_add_bypass_ctrl_r2_c2;
    assign dpe_done = dpe_done_r1_c1 & dpe_done_r2_c1 &
                      dpe_done_r1_c2 & dpe_done_r2_c2;
    assign MSB_SA_Ready = MSB_SA_Ready_r1_c1 & MSB_SA_Ready_r2_c1 &
                          MSB_SA_Ready_r1_c2 & MSB_SA_Ready_r2_c2;

    // Vertical adder for column 1 (R1+R2 partial sums)
    adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_col1 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done_c1),
        .input1(data_out_r1_c1),
        .input2(data_out_r2_c1),
        .output_data(col1_sum)
    );

    // Vertical adder for column 2 (R1+R2 partial sums)
    adder_dpe_N_CHANNELS_1 #(
        .N_CHANNELS(1)
    ) adder_col2 (
        .clk(clk),
        .reset(rst),
        .en(accum_ready),
        .add_done(accum_done_c2),
        .input1(data_out_r1_c2),
        .input2(data_out_r2_c2),
        .output_data(col2_sum)
    );

    assign accum_done = accum_done_c1 & accum_done_c2;

    // Horizontal mux: select output from column group
    assign data_out = col_group_sel ? col2_sum : col1_sum;

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


// =============================================================================
// controller_scalable — controller for stacked DPE configurations
// =============================================================================
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


// =============================================================================
// global_controller — top-level handshaking
// =============================================================================
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


// =============================================================================
// sram — parameterized SRAM module
// =============================================================================
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


// =============================================================================
// xbar_ip_module — crossbar fan-out module
// =============================================================================
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


// =============================================================================
// adder_dpe_N_CHANNELS_1 — adder for stacked DPE outputs
// =============================================================================
module adder_dpe_N_CHANNELS_1 #(
    parameter N_CHANNELS = 1,  // Number of channels
    parameter DATA_WIDTH = 16
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
