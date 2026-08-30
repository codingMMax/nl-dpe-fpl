// input_projections_dpe.v
// Input projections (Q, K, V) for attention mechanism using DPE instances
//
// Q Projection: SRAM depth=N*(d/4), width=32 bits, temp buffer for 4 DPE outputs
// K Projection: N SRAMs each depth=d/4, width=32 bits, temp buffer per SRAM
// V Projection: d SRAMs each depth=N, width=8 bits, direct write

module input_projections_dpe #(
    parameter N = 128,
    parameter d = 128,
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 10
)(
    input wire clk,
    input wire rst,
    
    // Input stream
    input wire [DATA_WIDTH-1:0] data_in,
    input wire valid_in,
    output wire ready_in,
    
    // Control
    input wire start_q,
    input wire start_k,
    input wire start_v,
    
    // Q outputs: single SRAM outputs 4 elements (32 bits)
    output wire [4*DATA_WIDTH-1:0] q_data_out,
    input wire [$clog2(N*(d/4))-1:0] q_sram_read_addr,
    
    // K outputs: read same address from N SRAMs to get 4xN
    output wire [4*N*DATA_WIDTH-1:0] k_data_out,
    input wire [$clog2(d/4)-1:0] k_sram_read_addr,
    
    // V outputs: 4 rows from d SRAMs
    output wire [d*DATA_WIDTH-1:0] v0_data_out,
    output wire [d*DATA_WIDTH-1:0] v1_data_out,
    output wire [d*DATA_WIDTH-1:0] v2_data_out,
    output wire [d*DATA_WIDTH-1:0] v3_data_out,
    input wire [$clog2(N)-1:0] v_sram_read_addr_0,
    input wire [$clog2(N)-1:0] v_sram_read_addr_1,
    input wire [$clog2(N)-1:0] v_sram_read_addr_2,
    input wire [$clog2(N)-1:0] v_sram_read_addr_3,
    
    // Status
    output wire q_done,
    output wire k_done,
    output wire v_done
);

    // ========================================
    // Q Projection
    // ========================================
    
    wire [ADDR_WIDTH-1:0] q_ctrl_read_address [0:3];
    wire [ADDR_WIDTH-1:0] q_ctrl_write_address [0:3];
    wire q_ctrl_w_en [0:3], q_ctrl_valid_n [0:3], q_ctrl_ready [0:3];
    wire q_ctrl_w_buf_en [0:3];
    wire [1:0] q_ctrl_nl_dpe_control [0:3];
    wire q_ctrl_shift_add_control [0:3], q_ctrl_shift_add_bypass [0:3];
    wire q_ctrl_load_output_reg [0:3], q_ctrl_load_input_reg [0:3];
    
    wire [DATA_WIDTH-1:0] q_dpe_data_out [0:3];
    wire q_dpe_done [0:3];
    wire q_reg_full [0:3], q_shift_add_done [0:3];
    wire q_MSB_SA_Ready [0:3], q_shift_add_bypass_ctrl [0:3];
    
    genvar q_i;
    generate
        for (q_i = 0; q_i < 4; q_i = q_i + 1) begin : q_dpe_gen
            dpeController #(
                .KERNEL_WIDTH(d),
                .READ_ADDR_WIDTH(ADDR_WIDTH),
                .WRITE_ADDR_WIDTH($clog2(N*(d/4)))
            ) q_controller (
                .clk(clk), .rst(rst),
                .MSB_SA_Ready(q_MSB_SA_Ready[q_i]),
                .dpe_done(q_dpe_done[q_i]),
                .reg_full(q_reg_full[q_i]),
                .shift_add_done(q_shift_add_done[q_i]),
                .shift_add_bypass_ctrl(q_shift_add_bypass_ctrl[q_i]),
                .read_address(q_ctrl_read_address[q_i]),
                .write_address(q_ctrl_write_address[q_i]),
                .w_en(q_ctrl_w_en[q_i]),
                .valid(valid_in && start_q),
                .ready_n(1'b0), // This is to be interfaced from next stage ready
                .valid_n(q_ctrl_valid_n[q_i]),
                .ready(q_ctrl_ready[q_i]),
                .w_buf_en(q_ctrl_w_buf_en[q_i]),
                .nl_dpe_control(q_ctrl_nl_dpe_control[q_i]),
                .shift_add_control(q_ctrl_shift_add_control[q_i]),
                .shift_add_bypass(q_ctrl_shift_add_bypass[q_i]),
                .load_output_reg(q_ctrl_load_output_reg[q_i]),
                .load_input_reg(q_ctrl_load_input_reg[q_i])
            );
            
            dpe q_dpe_inst (
                .clk(clk), .reset(rst),
                .data_in(data_in),
                .nl_dpe_control(q_ctrl_nl_dpe_control[q_i]),
                .shift_add_control(q_ctrl_shift_add_control[q_i]),
                .w_buf_en(q_ctrl_w_buf_en[q_i]),
                .shift_add_bypass(q_ctrl_shift_add_bypass[q_i]),
                .load_output_reg(q_ctrl_load_output_reg[q_i]),
                .load_input_reg(q_ctrl_load_input_reg[q_i]),
                .MSB_SA_Ready(q_MSB_SA_Ready[q_i]),
                .data_out(q_dpe_data_out[q_i]),
                .dpe_done(q_dpe_done[q_i]),
                .reg_full(q_reg_full[q_i]),
                .shift_add_done(q_shift_add_done[q_i]),
                .shift_add_bypass_ctrl(q_shift_add_bypass_ctrl[q_i])
            );
        end
    endgenerate
    
    genvar q_s;
    generate
        for (q_s = 0; q_s < 4; q_s = q_s + 1) begin : q_sram_gen
            wire [DATA_WIDTH-1:0] q_sram_out;
            sram #(
                .DEPTH(N*(d/4)),
                .WIDTH(DATA_WIDTH),
                .ADDR_WIDTH($clog2(N*(d/4)))
            ) q_sram (
                .clk(clk),
                .w_en(q_ctrl_w_en[q_s]),
                .write_addr(q_ctrl_write_address[q_s]),
                .read_addr(q_sram_read_addr),
                .data_in(q_dpe_data_out[q_s]),
                .data_out(q_sram_out)
            );
            assign q_data_out[q_s*DATA_WIDTH +: DATA_WIDTH] = q_sram_out;
        end
    endgenerate
    
    // Q is done when all 4 SRAMs are filled (N*d/4 iterations total)
    // Check that all controllers have reached their final write address
    assign q_done = (q_ctrl_write_address[0] == (N*(d/4) - 1)) && q_ctrl_valid_n[0] && 
                    (q_ctrl_write_address[1] == (N*(d/4) - 1)) && q_ctrl_valid_n[1] &&
                    (q_ctrl_write_address[2] == (N*(d/4) - 1)) && q_ctrl_valid_n[2] &&
                    (q_ctrl_write_address[3] == (N*(d/4) - 1)) && q_ctrl_valid_n[3];
    
    // ========================================
    // K Projection
    // ========================================
    
    wire [ADDR_WIDTH-1:0] k_ctrl_read_address [0:3];
    wire [ADDR_WIDTH-1:0] k_ctrl_write_address [0:3];
    wire k_ctrl_w_en [0:3], k_ctrl_valid_n [0:3], k_ctrl_ready [0:3];
    wire k_ctrl_w_buf_en [0:3];
    wire [1:0] k_ctrl_nl_dpe_control [0:3];
    wire k_ctrl_shift_add_control [0:3], k_ctrl_shift_add_bypass [0:3];
    wire k_ctrl_load_output_reg [0:3], k_ctrl_load_input_reg [0:3];
    
    wire [DATA_WIDTH-1:0] k_dpe_data_out [0:3];
    wire k_dpe_done [0:3];
    wire k_reg_full [0:3], k_shift_add_done [0:3];
    wire k_MSB_SA_Ready [0:3], k_shift_add_bypass_ctrl [0:3];
    
    // K SRAM group selector: tracks which group of 4 SRAMs (out of N groups) to write to
    reg [$clog2(N)-1:0] k_current_sram_group;
    
    genvar k_i;
    generate
        for (k_i = 0; k_i < 4; k_i = k_i + 1) begin : k_dpe_gen
            dpeController #(
                .KERNEL_WIDTH(d/4),
                .READ_ADDR_WIDTH(ADDR_WIDTH),
                .WRITE_ADDR_WIDTH($clog2(d/4))
            ) k_controller (
                .clk(clk), .rst(rst),
                .MSB_SA_Ready(k_MSB_SA_Ready[k_i]),
                .dpe_done(k_dpe_done[k_i]),
                .reg_full(k_reg_full[k_i]),
                .shift_add_done(k_shift_add_done[k_i]),
                .shift_add_bypass_ctrl(k_shift_add_bypass_ctrl[k_i]),
                .read_address(k_ctrl_read_address[k_i]),
                .write_address(k_ctrl_write_address[k_i]),
                .w_en(k_ctrl_w_en[k_i]),
                .valid(valid_in && start_k),
                .ready_n(1'b0), // next stage ready to be interfaced
                .valid_n(k_ctrl_valid_n[k_i]),
                .ready(k_ctrl_ready[k_i]),
                .w_buf_en(k_ctrl_w_buf_en[k_i]),
                .nl_dpe_control(k_ctrl_nl_dpe_control[k_i]),
                .shift_add_control(k_ctrl_shift_add_control[k_i]),
                .shift_add_bypass(k_ctrl_shift_add_bypass[k_i]),
                .load_output_reg(k_ctrl_load_output_reg[k_i]),
                .load_input_reg(k_ctrl_load_input_reg[k_i])
            );
            
            dpe k_dpe_inst (
                .clk(clk), .reset(rst),
                .data_in(data_in),
                .nl_dpe_control(k_ctrl_nl_dpe_control[k_i]),
                .shift_add_control(k_ctrl_shift_add_control[k_i]),
                .w_buf_en(k_ctrl_w_buf_en[k_i]),
                .shift_add_bypass(k_ctrl_shift_add_bypass[k_i]),
                .load_output_reg(k_ctrl_load_output_reg[k_i]),
                .load_input_reg(k_ctrl_load_input_reg[k_i]),
                .MSB_SA_Ready(k_MSB_SA_Ready[k_i]),
                .data_out(k_dpe_data_out[k_i]),
                .dpe_done(k_dpe_done[k_i]),
                .reg_full(k_reg_full[k_i]),
                .shift_add_done(k_shift_add_done[k_i]),
                .shift_add_bypass_ctrl(k_shift_add_bypass_ctrl[k_i])
            );
        end
    endgenerate
    
    // Track which group of 4 SRAMs to write to (advances when depth is full)
    // This is made based on the assumption that SRAMs are clustered into groups of 4 for writing and you fill one then the other as N proceeds
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            k_current_sram_group <= 0;
        end else begin
            if (k_ctrl_valid_n[0] && start_k) begin  // DPE done signal
                if (k_ctrl_write_address[0] == (d/4 - 1)) begin  // Depth full, move to next group
                    k_current_sram_group <= k_current_sram_group + 1;
                end
            end
        end
    end
    
    // Generate 4×N SRAMs with decoded write enables
    genvar k_s, k_d;
    generate
        for (k_d = 0; k_d < 4; k_d = k_d + 1) begin : k_dpe_sram_gen
            for (k_s = 0; k_s < N; k_s = k_s + 1) begin : k_sram_gen
                wire [DATA_WIDTH-1:0] k_sram_out;
                wire k_sram_w_en_decoded;
                
                // Decode: enable this SRAM only if it's in the current group
                assign k_sram_w_en_decoded = k_ctrl_w_en[k_d] && (k_current_sram_group == k_s);
                
                sram #(
                    .DEPTH(d/4),
                    .WIDTH(DATA_WIDTH),
                    .ADDR_WIDTH($clog2(d/4))
                ) k_sram (
                    .clk(clk),
                    .w_en(k_sram_w_en_decoded),
                    .write_addr(k_ctrl_write_address[k_d]),
                    .read_addr(k_sram_read_addr),
                    .data_in(k_dpe_data_out[k_d]),
                    .data_out(k_sram_out)
                );
                assign k_data_out[(k_s*4 + k_d)*DATA_WIDTH +: DATA_WIDTH] = k_sram_out;
            end
        end
    endgenerate
    
    // K is done when all N groups of 4 SRAMs are filled (N*d/4 total iterations)
    assign k_done = (k_current_sram_group == (N - 1)) && 
                    (k_ctrl_write_address[0] == (d/4 - 1)) && k_ctrl_valid_n[0];
    
    // ========================================
    // V Projection  
    // ========================================
    
    wire [ADDR_WIDTH-1:0] v_ctrl_read_address;
    wire [ADDR_WIDTH-1:0] v_ctrl_write_address;
    wire v_ctrl_w_en, v_ctrl_valid_n, v_ctrl_ready;
    wire v_ctrl_w_buf_en;
    wire [1:0] v_ctrl_nl_dpe_control;
    wire v_ctrl_shift_add_control, v_ctrl_shift_add_bypass;
    wire v_ctrl_load_output_reg, v_ctrl_load_input_reg;
    
    wire [DATA_WIDTH-1:0] v_dpe_data_out;
    wire v_dpe_done;
    wire v_reg_full, v_shift_add_done;
    wire v_MSB_SA_Ready, v_shift_add_bypass_ctrl;
    
    // V SRAM selector: tracks which SRAM (out of d) to write to
    reg [$clog2(d)-1:0] v_current_sram;
    reg [$clog2(N)-1:0] v_current_row;
    
    // FSM states
    localparam IDLE = 2'b00, WRITING = 2'b01;
    reg [1:0] v_state, v_next_state;
    
    // Sequential read buffering
    reg [d*DATA_WIDTH-1:0] v0_data_reg, v1_data_reg, v2_data_reg, v3_data_reg;
    reg [1:0] read_phase;
    wire [d*DATA_WIDTH-1:0] v_read_bus;
    
    // State register and counter updates
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            v_state <= IDLE;
            v_current_sram <= 0;
            v_current_row <= 0;
        end else begin
            v_state <= v_next_state;
            if (v_state == WRITING && v_ctrl_valid_n) begin
                if (v_current_sram == (d - 1)) begin
                    v_current_sram <= 0;
                    if (v_current_row == (N - 1)) begin
                        v_current_row <= 0;
                    end else begin
                        v_current_row <= v_current_row + 1;
                    end
                end else begin
                    v_current_sram <= v_current_sram + 1;
                end
            end
        end
    end
    
    // Next state logic
    always @* begin
        v_next_state = v_state;
        case (v_state)
            IDLE: if (start_v) v_next_state = WRITING;
            WRITING: begin
                if (v_ctrl_valid_n && v_current_sram == (d - 1) && v_current_row == (N - 1)) begin
                    v_next_state = IDLE;
                end
            end
        endcase
    end
    
    // Sequential read buffering: accumulate data over 4 cycles
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            read_phase <= 0;
            v0_data_reg <= 0;
            v1_data_reg <= 0;
            v2_data_reg <= 0;
            v3_data_reg <= 0;
        end else begin
            case (read_phase)
                0: v0_data_reg <= v_read_bus;
                1: v1_data_reg <= v_read_bus;
                2: v2_data_reg <= v_read_bus;
                3: v3_data_reg <= v_read_bus;
            endcase
            read_phase <= read_phase + 1;
        end
    end
    
    dpeController #(
        .KERNEL_WIDTH(d),
        .READ_ADDR_WIDTH(ADDR_WIDTH),
        .WRITE_ADDR_WIDTH($clog2(N))
    ) v_controller (
        .clk(clk), .rst(rst),
        .MSB_SA_Ready(v_MSB_SA_Ready),
        .dpe_done(v_dpe_done),
        .reg_full(v_reg_full),
        .shift_add_done(v_shift_add_done),
        .shift_add_bypass_ctrl(v_shift_add_bypass_ctrl),
        .read_address(v_ctrl_read_address),
        .write_address(v_ctrl_write_address),
        .w_en(v_ctrl_w_en),
        .valid(valid_in && start_v),
        .ready_n(1'b0), // next stage ready to be interfaced
        .valid_n(v_ctrl_valid_n),
        .ready(v_ctrl_ready),
        .w_buf_en(v_ctrl_w_buf_en),
        .nl_dpe_control(v_ctrl_nl_dpe_control),
        .shift_add_control(v_ctrl_shift_add_control),
        .shift_add_bypass(v_ctrl_shift_add_bypass),
        .load_output_reg(v_ctrl_load_output_reg),
        .load_input_reg(v_ctrl_load_input_reg)
    );
    
    dpe v_dpe_inst (
        .clk(clk), .reset(rst),
        .data_in(data_in),
        .nl_dpe_control(v_ctrl_nl_dpe_control),
        .shift_add_control(v_ctrl_shift_add_control),
        .w_buf_en(v_ctrl_w_buf_en),
        .shift_add_bypass(v_ctrl_shift_add_bypass),
        .load_output_reg(v_ctrl_load_output_reg),
        .load_input_reg(v_ctrl_load_input_reg),
        .MSB_SA_Ready(v_MSB_SA_Ready),
        .data_out(v_dpe_data_out),
        .dpe_done(v_dpe_done),
        .reg_full(v_reg_full),
        .shift_add_done(v_shift_add_done),
        .shift_add_bypass_ctrl(v_shift_add_bypass_ctrl)
    );
    
    // V SRAMs with 4 read ports (read 4 sequential rows over 4 cycles)
    genvar v_s;
    generate
        for (v_s = 0; v_s < d; v_s = v_s + 1) begin : v_sram_gen
            wire [DATA_WIDTH-1:0] v_out_0;
            wire v_sram_w_en_decoded;
            
            // Decode: enable this SRAM only if it's the current one being written
            assign v_sram_w_en_decoded = v_ctrl_w_en && (v_current_sram == v_s);
            
            sram #(.DEPTH(N), .WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(N)))
            v_sram_0 (.clk(clk), .w_en(v_sram_w_en_decoded), 
                     .write_addr(v_current_row), 
                     .read_addr(v_sram_read_addr_0), 
                     .data_in(v_dpe_data_out), .data_out(v_out_0));
            
            assign v_read_bus[v_s*DATA_WIDTH +: DATA_WIDTH] = v_out_0;
            
            assign v0_data_out[v_s*DATA_WIDTH +: DATA_WIDTH] = v0_data_reg[v_s*DATA_WIDTH +: DATA_WIDTH];
            assign v1_data_out[v_s*DATA_WIDTH +: DATA_WIDTH] = v1_data_reg[v_s*DATA_WIDTH +: DATA_WIDTH];
            assign v2_data_out[v_s*DATA_WIDTH +: DATA_WIDTH] = v2_data_reg[v_s*DATA_WIDTH +: DATA_WIDTH];
            assign v3_data_out[v_s*DATA_WIDTH +: DATA_WIDTH] = v3_data_reg[v_s*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate
    
    // V is done when all d SRAMs are filled (d*N total iterations)
    assign v_done = (v_current_sram == (d - 1)) && (v_current_row == (N - 1)) && v_ctrl_valid_n;
    assign ready_in = (start_q && q_ctrl_ready[0]) || (start_k && k_ctrl_ready[0]) || (start_v && v_ctrl_ready);

endmodule
