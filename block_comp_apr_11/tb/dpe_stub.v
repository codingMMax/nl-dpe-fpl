// Behavioral DPE model with real VMM computation.
//
// Models an R×C analog crossbar at behavioral level:
//   - Weight memory: R × C int8 weights (pre-loaded via weight_load interface)
//   - Input buffer: R int8 values loaded via w_buf_en
//   - VMM: output[c] = Σ_{r=0}^{R-1} input[r] × weight[r][c]  for each column c
//   - Output: C results serialized through data_out port, one per cycle
//
// Weight loading: before normal operation, assert weight_wen with weight_data
//   and weight_col_idx to load one column of weights at a time.
//   Each column load takes R cycles (one weight per cycle).
//   This is a testbench-only interface (not part of the real DPE hard block).
//
// Handshake protocol (matches controller_scalable and conv_controller):
//   1. Controller asserts w_buf_en → DPE buffers data_in into input_buffer[r]
//   2. After KERNEL_WIDTH elements → DPE asserts reg_full
//   3. Controller fires nl_dpe_control=2'b11 → DPE computes VMM
//   4. DPE outputs C results serially → asserts dpe_done
//   5. DPE resets → ready for next pass

module dpe #(
    parameter KERNEL_WIDTH = 512,   // R: number of input elements per VMM pass
    parameter NUM_COLS     = 128    // C: number of crossbar columns (output elements)
)(
    input  wire        clk,
    input  wire        reset,
    input  wire [39:0] data_in,
    input  wire [1:0]  nl_dpe_control,
    input  wire        shift_add_control,
    input  wire        w_buf_en,
    input  wire        shift_add_bypass,
    input  wire        load_output_reg,
    input  wire        load_input_reg,
    output reg         MSB_SA_Ready,
    output reg  [39:0] data_out,
    output reg         dpe_done,
    output reg         reg_full,
    output reg         shift_add_done,
    output reg         shift_add_bypass_ctrl,

    // Testbench-only: weight loading interface
    input  wire        weight_wen,
    input  wire [7:0]  weight_data,      // one int8 weight per cycle
    input  wire [15:0] weight_row_addr,  // which row to write
    input  wire [15:0] weight_col_addr   // which column to write
);

    // ── Weight memory: R × C int8 ──
    reg signed [7:0] weights [0:KERNEL_WIDTH-1][0:NUM_COLS-1];

    // Weight loading (testbench only)
    always @(posedge clk) begin
        if (weight_wen)
            weights[weight_row_addr][weight_col_addr] <= weight_data;
    end

    // ── Input buffer: R int8 ──
    reg signed [7:0] input_buffer [0:KERNEL_WIDTH-1];
    reg [15:0] load_count;

    // ── VMM result: C columns ──
    reg signed [31:0] vmm_result [0:NUM_COLS-1];
    reg [15:0] output_col_idx;   // which column is being output

    // ── FSM ──
    localparam S_IDLE      = 3'd0;
    localparam S_LOAD      = 3'd1;
    localparam S_WAIT_EXEC = 3'd2;
    localparam S_COMPUTE   = 3'd3;
    localparam S_OUTPUT    = 3'd4;
    localparam S_DRAIN     = 3'd5;

    reg [2:0]  state;
    reg [7:0]  compute_cycles;

    integer r, c;  // loop variables for VMM

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S_IDLE;
            load_count <= 0;
            data_out <= 0;
            dpe_done <= 0;
            reg_full <= 0;
            MSB_SA_Ready <= 1;
            shift_add_done <= 1;
            shift_add_bypass_ctrl <= 1;
            compute_cycles <= 0;
            output_col_idx <= 0;
        end else begin
            dpe_done <= 0;  // default: pulse

            case (state)
                S_IDLE: begin
                    reg_full <= 0;
                    MSB_SA_Ready <= 1;
                    shift_add_done <= 1;
                    load_count <= 0;
                    compute_cycles <= 0;
                    output_col_idx <= 0;

                    if (w_buf_en) begin
                        input_buffer[0] <= data_in[7:0];  // store int8
                        load_count <= 1;
                        state <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    if (w_buf_en) begin
                        input_buffer[load_count] <= data_in[7:0];
                        load_count <= load_count + 1;
                        if (load_count + 1 >= KERNEL_WIDTH) begin
                            reg_full <= 1;
                            MSB_SA_Ready <= 0;
                            shift_add_done <= 0;
                            state <= S_WAIT_EXEC;
                        end
                    end
                end

                S_WAIT_EXEC: begin
                    if (nl_dpe_control == 2'b11) begin
                        // Compute VMM: output[c] = Σ_r input[r] × weight[r][c]
                        for (c = 0; c < NUM_COLS; c = c + 1) begin
                            vmm_result[c] = 0;
                            for (r = 0; r < KERNEL_WIDTH; r = r + 1) begin
                                vmm_result[c] = vmm_result[c] +
                                    input_buffer[r] * weights[r][c];
                            end
                        end
                        compute_cycles <= 0;
                        output_col_idx <= 0;
                        state <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    // Simulate 8-cycle bit-serial compute delay
                    compute_cycles <= compute_cycles + 1;
                    if (compute_cycles >= 7) begin
                        MSB_SA_Ready <= 1;
                        shift_add_done <= 1;
                        output_col_idx <= 0;
                        state <= S_OUTPUT;
                    end
                end

                S_OUTPUT: begin
                    // Serial output: one column per cycle
                    dpe_done <= 1;  // held high during output phase
                    data_out <= {{8{vmm_result[output_col_idx][31]}},
                                 vmm_result[output_col_idx][31:0]};
                    if (output_col_idx < NUM_COLS - 1) begin
                        output_col_idx <= output_col_idx + 1;
                    end else begin
                        state <= S_DRAIN;
                    end
                end

                S_DRAIN: begin
                    dpe_done <= 0;
                    reg_full <= 0;
                    load_count <= 0;
                    output_col_idx <= 0;
                    MSB_SA_Ready <= 1;
                    shift_add_done <= 1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
