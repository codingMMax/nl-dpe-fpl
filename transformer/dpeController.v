// dpeController.v
// Controller for Deep Processing Engine (DPE) operations
// Handles memory addressing and control signals for MVM operations
module dpeController #(
    parameter KERNEL_WIDTH = 128,    // Vector length for MVM
    parameter READ_ADDR_WIDTH  = 7,  // Address width for read operations
    parameter WRITE_ADDR_WIDTH = 7   // Address width for write operations
)(
    // Clock and reset
    input  wire                    clk,
    input  wire                    rst,

    // DPE interface signals
    input  wire                    MSB_SA_Ready,     // Most significant bit shift/add ready
    input  wire                    dpe_done,         // DPE operation complete
    input  wire                    reg_full,         // Register buffer full
    input  wire                    shift_add_done,   // Shift/add operation complete
    input  wire                    shift_add_bypass_ctrl, // Control bypass of shift/add

    // Memory interface
    output reg  [READ_ADDR_WIDTH-1:0]   read_address,     // SRAM read address
    output reg  [WRITE_ADDR_WIDTH-1:0]  write_address,    // SRAM write address
    output reg                          w_en,             // Write enable
    
    // Handshaking signals
    input  wire                    valid,            // Input data valid
    input  wire                    ready_n,          // Downstream ready (active low)
    output reg                     valid_n,          // Output data valid (active high)
    output reg                     ready,            // Ready for input
    
    // DPE control signals
    output reg                     w_buf_en,         // Write buffer enable
    output reg  [1:0]             nl_dpe_control,   // Non-linear DPE control
    output reg                     shift_add_control, // Shift/add control
    output reg                     shift_add_bypass,  // Shift/add bypass
    output reg                     load_output_reg,  // Load output register
    output wire                    load_input_reg    // Load input register
);

    // Internal registers for addressing and control
    reg [WRITE_ADDR_WIDTH-1:0] write_address_reg;  // Tracks write position
    reg [READ_ADDR_WIDTH-1:0] read_address_reg;    // Tracks read position

    // Control flags
    reg stall;         // Pipeline stall indicator
    reg memory_stall;  // Memory access stall indicator
    reg dpe_exec_signal; // DPE execution control

    // ----------------------------------------
    // Memory Control Logic (DPE processing mode)
    // - Fill: load internal buffer until reg_full
    // - Execute: shift data to DPE, then write result on dpe_done
    // - Read address incremented until reg_full, then ready deasserted
    // - Write address incremented on dpe_done, max VEC_LEN-1
    // ----------------------------------------
    // Define vector length. Reuse KERNEL_WIDTH as a default vector length.
    localparam VEC_LEN = KERNEL_WIDTH;

    // Memory controller states
    localparam MEM_IDLE    = 2'd0;
    localparam MEM_FILL    = 2'd1;
    localparam MEM_EXECUTE = 2'd2;

    reg [1:0] mem_state, mem_next_state;
    reg in_execute;  // Flag for execute phase

    // Combinational next-state logic
    always @* begin
        // Defaults: hold current state and registers
        mem_next_state = mem_state;

        case (mem_state)
            MEM_IDLE: begin
                if (valid && ~memory_stall) begin
                    mem_next_state = MEM_FILL;
                end
            end

            MEM_FILL: begin
                if (reg_full) begin
                    mem_next_state = MEM_EXECUTE;
                end
            end

            MEM_EXECUTE: begin
                if (dpe_done) begin
                    if (write_address_reg == VEC_LEN-1) begin
                        mem_next_state = MEM_IDLE;
                    end else if (valid) begin
                        mem_next_state = MEM_FILL;
                    end
                end
                else begin
                    mem_next_state = MEM_EXECUTE;
                end
            end

            default: mem_next_state = MEM_IDLE;
        endcase
    end

    // Sequential state update
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mem_state <= MEM_IDLE;
        end else begin
            mem_state <= mem_next_state;
        end
    end

    // Sequential register updates
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            write_address_reg <= {WRITE_ADDR_WIDTH{1'b0}};
            read_address_reg <= {READ_ADDR_WIDTH{1'b0}};
            in_execute <= 1'b0;
        end else begin
            case (mem_state)
                MEM_IDLE: begin
                    if (valid && ~memory_stall) begin
                        write_address_reg <= {WRITE_ADDR_WIDTH{1'b0}};  // Start writing at address 0
                        read_address_reg <= {READ_ADDR_WIDTH{1'b0}};   // Reset read address
                        in_execute <= 1'b0;
                    end
                end

                MEM_FILL: begin
                    // placeholder if read_address increment needed
                    /*
                    if (valid && ~memory_stall) begin
                        read_address_reg <= read_address_reg + 1'b1;  // Increment read address until reg_full
                    end
                    */
                end

                MEM_EXECUTE: begin
                    if (dpe_done) begin
                        write_address_reg <= write_address_reg + 1'b1;  // Increment write address on dpe_done
                        if (write_address_reg == VEC_LEN-1) begin
                            write_address_reg <= {WRITE_ADDR_WIDTH{1'b0}}; // Reset write address
                            in_execute <= 1'b0;  // Reset when done
                        end
                    end
                end
            endcase
        end
    end

    // (Address generation and memory_flag were removed for MVM flow - read/write registers are driven by state machine)

    // ----------------------------------------
    // DPE Control Logic
    // ----------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            dpe_exec_signal <= 1'b0;
            nl_dpe_control <= 2'b0;
        end else begin
            // Set execution signal when buffer is full
            dpe_exec_signal <= reg_full;
            
            // Control DPE non-linear unit
            nl_dpe_control <= dpe_exec_signal ? 2'b11 : 2'b00;
        end
    end

    // ----------------------------------------
    // Handshaking and Status Logic
    // ----------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_n <= 1'b0;
            ready <= 1'b0;
            stall <= 1'b0;
            memory_stall <= 1'b0;
        end else begin
            // Check for memory near full condition
            memory_stall <= reg_full && ((mem_state == MEM_FILL) || (mem_state == MEM_EXECUTE && ~dpe_done)); 

            // Set ready based on memory stall
            ready <= ~memory_stall;

            // Set valid when DPE completes
            valid_n <= dpe_done;

            // Update stall status
            stall <= ~ready_n;
        end
    end

    // ----------------------------------------
    // Output Assignments (combinational, based on mem_state)
    // ----------------------------------------
    always @* begin
        // defaults
        read_address = read_address_reg;
        write_address = write_address_reg;
        w_en = 1'b0;
        w_buf_en = 1'b0;
        shift_add_bypass = shift_add_bypass_ctrl;
        shift_add_control = MSB_SA_Ready;
        load_output_reg = shift_add_done;

        case (mem_state)
            MEM_FILL: begin
                // enable writes while filling and not stalled
                w_en = valid && ~memory_stall && dpe_done;
                w_buf_en = 1'b1;
            end

            MEM_EXECUTE: begin
                // shift data to DPE
                w_buf_en = 1'b0;
                // write results when dpe_done
                w_en = dpe_done;
            end

            default: begin
                // no-op
            end
        endcase
    end

    // Direct assignments
    assign load_input_reg = reg_full;

endmodule