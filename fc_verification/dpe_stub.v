// Behavioral DPE model with real VMM computation — cycle-accurate.
//
// Models an R×C analog crossbar at behavioral level:
//   - Weight memory: R × C int8 weights (pre-loaded via weight_load interface)
//   - Input buffer: R int8 values loaded via w_buf_en
//   - VMM: output[c] = Σ_{r=0}^{R-1} input[r] × weight[r][c]  for each column c
//   - Output: C results serialized through data_out port
//
// Cycle-accurate parameters:
//   - DPE_BUF_WIDTH: bits per w_buf_en strobe (16 or 40)
//     S_LOAD takes ceil(KERNEL_WIDTH * 8 / DPE_BUF_WIDTH) strobes
//     S_OUTPUT takes ceil(NUM_COLS * 8 / DPE_BUF_WIDTH) cycles
//   - COMPUTE_CYCLES: bit-serial pipeline delay
//     ADC (Azure-Lily): 44 cycles (8 bit-slices × 3-stage pipeline + ADC)
//     ACAM (NL-DPE): 3 cycles (8 bit-slices × 2-stage pipeline + ACAM drain)
//
// Weight loading: testbench-only interface (weight_wen, weight_data, etc.)
//
// Handshake protocol (matches controller_scalable and conv_controller):
//   1. Controller asserts w_buf_en → DPE buffers data_in
//   2. After enough strobes → DPE asserts reg_full
//   3. Controller fires nl_dpe_control=2'b11 → DPE computes VMM
//   4. DPE outputs results serially → asserts dpe_done
//   5. DPE resets → ready for next pass

// Default parameters match the NL-DPE W=16 full DIMM top's primary dimm_exp
// stage (KW=128 dual-identity exp, BUF_WIDTH=40 → 5 elements/strobe, COMPUTE=3
// ACAM pipeline). Other DPE uses (sm_exp/ws_log/ws_exp) override via defparam
// in the TB when needed. These module-level defaults are IGNORED by VTR — the
// arch XML's dpe model is parameterless — so they only affect iverilog sim.
module dpe #(
    parameter KERNEL_WIDTH   = 128,
    parameter NUM_COLS       = 128,
    parameter DPE_BUF_WIDTH  = 40,
    parameter COMPUTE_CYCLES = 3,
    parameter ACAM_MODE      = 1
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

    // ── Derived parameters ──
    // Packed transfers: each w_buf_en strobe carries DPE_BUF_WIDTH bits
    // = ELEMS_PER_STROBE int8 elements. Controller sends LOAD_STROBES strobes.
    localparam ELEMS_PER_STROBE = DPE_BUF_WIDTH / 8;
    localparam LOAD_STROBES  = (KERNEL_WIDTH + ELEMS_PER_STROBE - 1) / ELEMS_PER_STROBE;
    localparam OUTPUT_CYCLES = (NUM_COLS + ELEMS_PER_STROBE - 1) / ELEMS_PER_STROBE;

    // ── Weight memory: R × C int8 ──
    reg signed [7:0] weights [0:KERNEL_WIDTH-1][0:NUM_COLS-1];

    // Weight initialization (zero all weights at reset)
    integer wi, wj;
    initial begin
        for (wi = 0; wi < KERNEL_WIDTH; wi = wi + 1)
            for (wj = 0; wj < NUM_COLS; wj = wj + 1)
                weights[wi][wj] = 0;
    end

    // Weight loading (testbench only)
    always @(posedge clk) begin
        if (weight_wen)
            weights[weight_row_addr][weight_col_addr] <= weight_data;
    end

    // ── Input buffer: R int8 ──
    reg signed [7:0] input_buffer [0:KERNEL_WIDTH-1];
    reg [15:0] load_count;  // counts elements loaded (1 per w_buf_en strobe)

    // ── VMM result: C columns ──
    reg signed [31:0] vmm_result [0:NUM_COLS-1];
    reg [15:0] output_col_idx;   // which output column we're on

    // ── FSM ──
    localparam S_IDLE      = 3'd0;
    localparam S_LOAD      = 3'd1;
    localparam S_WAIT_EXEC = 3'd2;
    localparam S_COMPUTE   = 3'd3;
    localparam S_OUTPUT    = 3'd4;
    localparam S_DRAIN     = 3'd5;

    reg [2:0]  state;
    reg [15:0] compute_cycle_cnt;

    integer r, c, b;  // loop variables

    // Track strobes (not individual elements)
    reg [15:0] strobe_count;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S_IDLE;
            load_count <= 0;
            strobe_count <= 0;
            data_out <= 0;
            dpe_done <= 0;
            reg_full <= 0;
            MSB_SA_Ready <= 1;
            shift_add_done <= 1;
            shift_add_bypass_ctrl <= 1;
            compute_cycle_cnt <= 0;
            output_col_idx <= 0;
        end else begin
            dpe_done <= 0;  // default: pulse

            case (state)
                S_IDLE: begin
                    reg_full <= 0;
                    MSB_SA_Ready <= 1;
                    shift_add_done <= 1;
                    load_count <= 0;
                    strobe_count <= 0;
                    compute_cycle_cnt <= 0;
                    output_col_idx <= 0;

                    if (w_buf_en) begin
                        // Unpack ELEMS_PER_STROBE bytes from data_in
                        for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)
                            if (b < KERNEL_WIDTH)
                                input_buffer[b] <= data_in[b*8 +: 8];
                        load_count <= (ELEMS_PER_STROBE < KERNEL_WIDTH) ?
                                       ELEMS_PER_STROBE : KERNEL_WIDTH;
                        strobe_count <= 1;
                        if (1 >= LOAD_STROBES) begin
                            reg_full <= 1;
                            MSB_SA_Ready <= 0;
                            shift_add_done <= 0;
                            state <= S_WAIT_EXEC;
                        end else begin
                            state <= S_LOAD;
                        end
                    end
                end

                S_LOAD: begin
                    if (w_buf_en) begin
                        // Unpack ELEMS_PER_STROBE bytes from data_in
                        for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)
                            if (load_count + b < KERNEL_WIDTH)
                                input_buffer[load_count + b] <= data_in[b*8 +: 8];
                        load_count <= load_count + ELEMS_PER_STROBE;
                        strobe_count <= strobe_count + 1;
                        if (strobe_count + 1 >= LOAD_STROBES) begin
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
                        // Apply ACAM nonlinear function (behavioral, per column)
                        // For identity weights: vmm_result[c] = input[c]
                        // ACAM_MODE=0: no ACAM (FC/ADC), output = VMM result
                        // ACAM_MODE=1: exp approximation (DIMM exp)
                        // ACAM_MODE=2: log approximation (DIMM log)
                        if (ACAM_MODE == 1) begin
                            for (c = 0; c < NUM_COLS; c = c + 1)
                                vmm_result[c] = 1 + vmm_result[c] +
                                    (vmm_result[c] * vmm_result[c]) / 2;
                        end else if (ACAM_MODE == 2) begin
                            for (c = 0; c < NUM_COLS; c = c + 1)
                                vmm_result[c] = vmm_result[c] - 1;
                        end
                        compute_cycle_cnt <= 0;
                        output_col_idx <= 0;
                        state <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    // Parameterized bit-serial compute delay
                    compute_cycle_cnt <= compute_cycle_cnt + 1;
                    if (compute_cycle_cnt >= COMPUTE_CYCLES - 1) begin
                        MSB_SA_Ready <= 1;
                        shift_add_done <= 1;
                        output_col_idx <= 0;
                        state <= S_OUTPUT;
                    end
                end

                S_OUTPUT: begin
                    // Packed output: OUTPUT_CYCLES cycles, ELEMS_PER_STROBE cols per cycle
                    dpe_done <= 1;  // held high during output phase
                    // Pack ELEMS_PER_STROBE column results into data_out
                    data_out <= 0;
                    for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)
                        if (output_col_idx * ELEMS_PER_STROBE + b < NUM_COLS)
                            data_out[b*8 +: 8] <= vmm_result[output_col_idx * ELEMS_PER_STROBE + b][7:0];
                    if (output_col_idx < OUTPUT_CYCLES - 1) begin
                        output_col_idx <= output_col_idx + 1;
                    end else begin
                        state <= S_DRAIN;
                    end
                end

                S_DRAIN: begin
                    dpe_done <= 0;
                    reg_full <= 0;
                    load_count <= 0;
                    strobe_count <= 0;
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
