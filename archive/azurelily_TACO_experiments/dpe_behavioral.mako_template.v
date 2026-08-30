<%
# Python variables for configuration
N_CHANNELS = 1
N_KERNELS = 128
KERNEL_WIDTH = 1
KERNEL_HEIGHT = 1
DATA_WIDTH = 8

# Derived parameters
INPUT_SIZE = KERNEL_WIDTH * KERNEL_HEIGHT * N_CHANNELS
NUM_DSP_PER_KERNEL = (INPUT_SIZE + 3) // 4
BUFFER_PTR_WIDTH = (INPUT_SIZE - 1).bit_length()
OUT_PTR_WIDTH = (N_KERNELS - 1).bit_length()
KERNEL_PTR_WIDTH = (N_KERNELS - 1).bit_length()
WEIGHT_ADDR_WIDTH = (INPUT_SIZE*N_KERNELS - 1).bit_length()

# Calculate total DSP requirements
TOTAL_DSPS = N_KERNELS * NUM_DSP_PER_KERNEL
%>

module dpe_behavioral_fully_parallel #(
    parameter N_CHANNELS = ${N_CHANNELS},
    parameter N_KERNELS = ${N_KERNELS},
    parameter KERNEL_WIDTH = ${KERNEL_WIDTH},
    parameter KERNEL_HEIGHT = ${KERNEL_HEIGHT},
    parameter DATA_WIDTH = ${DATA_WIDTH},
    parameter INPUT_SIZE = ${INPUT_SIZE},
    parameter NUM_DSP_PER_KERNEL = ${NUM_DSP_PER_KERNEL},
    parameter BUFFER_PTR_WIDTH = ${BUFFER_PTR_WIDTH},
    parameter OUT_PTR_WIDTH = ${OUT_PTR_WIDTH}
)(
    // Standard interface
    input wire clk,
    input wire reset,
    input wire [DATA_WIDTH-1:0] data_in,
    input wire [1:0] nl_dpe_control,
    input wire w_buf_en,
    input wire load_output_reg,
    
    // Weight loading interface - parallel for all kernels
    input wire [N_KERNELS-1:0] weight_write_en,
    input wire [N_KERNELS*INPUT_SIZE*DATA_WIDTH-1:0] weight_write_data,
    
    // Outputs
    output reg MSB_SA_Ready,
    output reg [N_KERNELS*DATA_WIDTH-1:0] data_out_parallel, // All kernel outputs
    output reg dpe_done,
    output reg reg_full
);

    // Shared input buffer (same input for all kernels)
    (* keep = "true" *) reg [INPUT_SIZE*DATA_WIDTH-1:0] input_buffer;
    
    // Individual weight data for each kernel
    (* keep = "true" *) wire [INPUT_SIZE*DATA_WIDTH-1:0] weight_data [N_KERNELS-1:0];
    
    // Individual results from each kernel
    wire [63:0] kernel_results [N_KERNELS-1:0];
    
    // Pointers and control
    reg [BUFFER_PTR_WIDTH-1:0] input_ptr;
    reg computation_ready;
    reg computation_done;
    
    // Simplified state machine for parallel processing
    reg [1:0] state;
    localparam IDLE = 2'd0;
    localparam LOAD_INPUT = 2'd1;
    localparam COMPUTE_ALL = 2'd2;
    localparam OUTPUT_DATA = 2'd3;
    
    // DSP control signals
    wire [11:0] mode_sigs;
    assign mode_sigs = 12'b0;

    // PARALLEL WEIGHT MEMORIES - One for each kernel
% for k in range(N_KERNELS):
    weight_memory #(
        .DATA_WIDTH(INPUT_SIZE*DATA_WIDTH),
        .DEPTH(1)  // Each kernel has one set of weights
    ) weight_mem_${k} (
        .clk(clk),
        .rst(reset),
        .w_en(weight_write_en[${k}]),
        .w_addr(1'b0),
        .r_addr(1'b0),
        .wt_data_in(weight_write_data[${k}*INPUT_SIZE*DATA_WIDTH +: INPUT_SIZE*DATA_WIDTH]),
        .wt_data_out(weight_data[${k}])
    );
    
% endfor

    // PARALLEL DSP CHAINS - One complete chain for each kernel
% for k in range(N_KERNELS):
    // ========== KERNEL ${k} DSP CHAIN ==========
    % for j in range(NUM_DSP_PER_KERNEL):
    // DSP instance ${j} for kernel ${k}
    wire [8:0] ax_k${k}_${j}, ay_k${k}_${j}, bx_k${k}_${j}, by_k${k}_${j};
    wire [8:0] cx_k${k}_${j}, cy_k${k}_${j}, dx_k${k}_${j}, dy_k${k}_${j};
    wire [63:0] chainin_k${k}_${j}, result_k${k}_${j}, chainout_k${k}_${j};
    
    // Input connections (shared input buffer)
    assign ax_k${k}_${j} = {1'b0, input_buffer[${j}*DATA_WIDTH +: DATA_WIDTH]};
    assign bx_k${k}_${j} = ((${j}+1) < INPUT_SIZE) ? {1'b0, input_buffer[(${j}+1)*DATA_WIDTH +: DATA_WIDTH]} : 9'b0;
    assign cx_k${k}_${j} = ((${j}+2) < INPUT_SIZE) ? {1'b0, input_buffer[(${j}+2)*DATA_WIDTH +: DATA_WIDTH]} : 9'b0;
    assign dx_k${k}_${j} = ((${j}+3) < INPUT_SIZE) ? {1'b0, input_buffer[(${j}+3)*DATA_WIDTH +: DATA_WIDTH]} : 9'b0;
    
    // Weight connections (unique for each kernel)
    assign ay_k${k}_${j} = {1'b0, weight_data[${k}][${j}*DATA_WIDTH +: DATA_WIDTH]};
    assign by_k${k}_${j} = ((${j}+1) < INPUT_SIZE) ? {1'b0, weight_data[${k}][(${j}+1)*DATA_WIDTH +: DATA_WIDTH]} : 9'b0;
    assign cy_k${k}_${j} = ((${j}+2) < INPUT_SIZE) ? {1'b0, weight_data[${k}][(${j}+2)*DATA_WIDTH +: DATA_WIDTH]} : 9'b0;
    assign dy_k${k}_${j} = ((${j}+3) < INPUT_SIZE) ? {1'b0, weight_data[${k}][(${j}+3)*DATA_WIDTH +: DATA_WIDTH]} : 9'b0;
    
    // Chain connections within this kernel
        % if j == 0:
    assign chainin_k${k}_${j} = 64'd0;  // First DSP in kernel ${k} chain
        % else:
    assign chainin_k${k}_${j} = chainout_k${k}_${j-1};  // Chain from previous DSP in kernel ${k}
        % endif
    
    // MAC unit instantiation
    int_sop_4 mac_k${k}_${j} (
        .clk(clk),
        .reset(reset),
        .mode_sigs(mode_sigs),
        .ax(ax_k${k}_${j}),
        .ay(ay_k${k}_${j}),
        .bx(bx_k${k}_${j}),
        .by(by_k${k}_${j}),
        .cx(cx_k${k}_${j}),
        .cy(cy_k${k}_${j}),
        .dx(dx_k${k}_${j}),
        .dy(dy_k${k}_${j}),
        .chainin(chainin_k${k}_${j}),
        .result(result_k${k}_${j}),
        .chainout(chainout_k${k}_${j})
    );
    
    % endfor
    // Final result for kernel ${k} is from the last DSP in its chain
    assign kernel_results[${k}] = result_k${k}_${NUM_DSP_PER_KERNEL-1};
    
% endfor

    // MAIN CONTROL FSM 
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            MSB_SA_Ready <= 1'b1;
            data_out_parallel <= {N_KERNELS*DATA_WIDTH{1'b0}};
            reg_full <= 1'b0;
            dpe_done <= 1'b0;
            input_ptr <= 0;
            computation_ready <= 1'b0;
            computation_done <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    dpe_done <= 1'b0;
                    reg_full <= 1'b0;
                    input_ptr <= 0;
                    computation_ready <= 1'b0;
                    computation_done <= 1'b0;
                    MSB_SA_Ready <= 1'b1;
                    
                    if (w_buf_en) begin
                        state <= LOAD_INPUT;
                        MSB_SA_Ready <= 1'b0;
                    end
                end
                
                LOAD_INPUT: begin
                    if (w_buf_en && input_ptr < INPUT_SIZE) begin
                        input_buffer[input_ptr*DATA_WIDTH +: DATA_WIDTH] <= data_in;
                        input_ptr <= input_ptr + 1'b1;
                        
                        if (input_ptr == INPUT_SIZE-1) begin
                            reg_full <= 1'b1;
                            computation_ready <= 1'b1;
                            state <= COMPUTE_ALL;
                        end
                    end
                end
                
                COMPUTE_ALL: begin
                    // All ${N_KERNELS} kernels compute in parallel
                    // Wait for computation to stabilize (1-2 cycles for DSP pipeline)
                    if (nl_dpe_control == 2'b11 && computation_ready) begin
                        computation_done <= 1'b1;
                        state <= OUTPUT_DATA;
                    end
                end
                
                OUTPUT_DATA: begin
                    if (load_output_reg) begin
                        // Output all kernel results in parallel
% for k in range(N_KERNELS):
                        data_out_parallel[${k}*DATA_WIDTH +: DATA_WIDTH] <= kernel_results[${k}][DATA_WIDTH-1:0];
% endfor
                        dpe_done <= 1'b1;
                        MSB_SA_Ready <= 1'b1;
                        state <= IDLE;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule




module weight_memory #(
    parameter DATA_WIDTH = ${DATA_WIDTH}*${INPUT_SIZE},
    parameter DEPTH = ${N_KERNELS}
)(
    input wire clk,           
    input wire w_en,
	input wire rst,
    input wire [$clog2(DEPTH)-1:0] r_addr,  // Address input (width based on depth)
    input wire [$clog2(DEPTH)-1:0] w_addr,
    input wire [DATA_WIDTH-1:0] wt_data_in,  // Data input for writing
    output reg [DATA_WIDTH-1:0] wt_data_out  // Data output for reading
);

    // Memory array with parameterized depth and width
    //reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] mem [DEPTH-1:0];

    // Read/Write operations
    //always @(posedge clk or posedge rst) begin
    always @(posedge clk) begin
        if (rst) begin
            wt_data_out <= {DATA_WIDTH{1'b0}};
        end else begin
            wt_data_out <= mem[r_addr];
        end
    end
    always @(posedge clk) begin
            if (w_en) begin
                mem[w_addr] <= wt_data_in;
            end        
    end

endmodule

/*
RESOURCE USAGE(Generated by Mako):
=====================================
Total Kernels: ${N_KERNELS}
Input Size per Kernel: ${INPUT_SIZE}
DSPs per Kernel: ${NUM_DSP_PER_KERNEL}
Total DSPs Required: ${TOTAL_DSPS}
Weight Memory Instances: ${N_KERNELS}
Total Weight Storage: ${N_KERNELS * INPUT_SIZE * DATA_WIDTH} bits

PERFORMANCE:
============
Latency: ~3-4 clock cycles (vs ${N_KERNELS} cycles for sequential)
Throughput: ${N_KERNELS}x improvement
All kernels compute simultaneously
*/