
// Top Module
module dpe_one_layer_behavioral_smallest(
    //     input weight_write_en,
    // input [6:0] weight_write_kernel_idx,  // clog2(128) - num kernel pointer width
    // input [63:0] weight_write_data,   // input size * data w
    // output [127:0] all_results_concatenated, // num dsp * 64
    // output [14:0] dummy_out_dpe_in,
    // output [12:0] dummy_out_dpe_out,

    input wire clk,
    input wire rst,
    input wire valid,
    input wire ready_n,
    input wire [(2*8)-1:0] data_in,      // num channels * dataw
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
    .N_CHANNELS(2),
    .ADDR_WIDTH(9),
    .N_KERNELS(128),
    .KERNEL_WIDTH(2),
    .KERNEL_HEIGHT(2),
    .W(32),
    .H(32),
    .S(1),
    .DEPTH(512),
    .DATA_WIDTH(8)
  ) conv1 (
    // .weight_write_en(weight_write_en),
    // .weight_write_kernel_idx(weight_write_kernel_idx),
    // .weight_write_data(weight_write_data),
    // .all_results_concatenated(all_results_concatenated),
    // .dummy_out_dpe_out(dummy_out_dpe_out),
    // .dummy_out_dpe_in(dummy_out_dpe_in),

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


  always @(posedge clk or posedge rst)
  begin
    if (rst)
    begin
      read_address <= 9'b0;
      write_address <= 9'b0;
    end
    else
    begin
      if(ready_g_out)
      begin
        read_address <= read_address + 1;
      end
      if(valid_g_out)
      begin
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
    parameter N_CHANNELS = 2,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 128,
    parameter KERNEL_WIDTH = 2,
    parameter KERNEL_HEIGHT = 2,
    parameter W = 32,
    parameter H = 32,
    parameter S = 1,
    parameter DEPTH = 512,
    parameter DATA_WIDTH = 8
  )(
    //     input weight_write_en,
    // input [6:0] weight_write_kernel_idx,  // clog2(128) - num kernel pointer width
    // input [63:0] weight_write_data,   // input size * data w
    // output [127:0] all_results_concatenated, // num dsp * 64
    // output [14:0] dummy_out_dpe_in,
    // output [12:0] dummy_out_dpe_out,

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

  // Instantiate the Controller module
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
  always @(posedge clk or posedge rst)
  begin
    if (rst)
    begin
      channel_counter <= 0;
      sram_data_in <= 0;
    end
    else
    begin
      if (valid && load_input_reg)
      begin
        w_addr <= write_address;

        case (channel_counter)
          0:
            sram_data_in <= data_in[7:0];
          1:
            sram_data_in <= data_in[15:8];
        endcase

        if (channel_counter == N_CHAN_MIN_1)
        begin
          channel_counter <= 0;
        end
        else
        begin
          channel_counter <= channel_counter + 1;
        end
      end
    end
  end

  // Accumulate kernel results
  always @(posedge clk or posedge rst)
  begin
    if (rst)
    begin
      kernel_results <= {(N_KERNELS*DATA_WIDTH){1'b0}};
      kernel_counter <= 0;
    end
    else
    begin
      if (dpe_done)
      begin
        kernel_results[kernel_counter*DATA_WIDTH +: DATA_WIDTH] <= dpe_single_output;
        if (kernel_counter == N_KERN_MIN_1)
        begin
          kernel_counter <= 0;
        end
        else
        begin
          kernel_counter <= kernel_counter + 1;
        end
      end
    end
  end

  // Output the accumulated results
  assign data_out = kernel_results;

endmodule

module conv_controller #(
    parameter N_CHANNELS = 2,
    parameter ADDR_WIDTH = 9,
    parameter N_KERNELS = 128,
    parameter KERNEL_WIDTH = 2,
    parameter KERNEL_HEIGHT = 2,
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
  always @(posedge clk or posedge rst)
  begin
    if (rst)
    begin
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

    end
    else
    begin
      state <= next_state;

      case (state)
        IDLE:
        begin
          ready <= 1'b1;
          valid_n <= 1'b0;

          if (valid)
          begin
            next_state <= LOAD_PATCH;
            patch_counter <= 0;
            w_en <= 1'b1;
            write_address_reg <= 0;
            n <= 0;
            m <= 0;
            s <= 0;
            sv <= 0;
          end
          else
          begin
            next_state <= IDLE;
          end
        end

        LOAD_PATCH:
        begin
          write_address_reg <= write_address_reg + 1;
          pointer_offset <= (n + W * (m + sv) + s * S) * N_CHANNELS;

          if (patch_counter == PATCH_SIZE - 1)
          begin
            w_en <= 1'b0;
            next_state <= RUN_DPE;
          end
          else
          begin
            patch_counter <= patch_counter + 1;

            // Update spatial counters
            if (n < KERNEL_WIDTH - 1)
            begin
              n <= n + 1;
            end
            else if (m < KERNEL_HEIGHT - 1)
            begin
              n <= 0;
              m <= m + 1;
            end
            else if (s < W - KERNEL_WIDTH + S - 1)
            begin
              n <= 0;
              m <= 0;
              s <= s + 1;
            end
            else if (sv < H - KERNEL_HEIGHT + S - 1)
            begin
              n <= 0;
              m <= 0;
              s <= 0;
              sv <= sv + 1;
            end
            else
            begin
              n <= 0;
              m <= 0;
              s <= 0;
              sv <= 0;
            end
          end
        end

        RUN_DPE:
        begin
          if (memory_flag)
          begin
            read_address_reg <= read_address_reg + 1;
          end
          if (dpe_done)
          begin
            next_state <= OUTPUT_RESULT;
          end
        end

        OUTPUT_RESULT:
        begin
          valid_n <= 1'b1;
          read_address_reg <= 0;

          if (kernel_counter == N_KERNELS - 1)
          begin
            kernel_counter <= 0;
            next_state <= IDLE;
          end
          else
          begin
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
  always @*
  begin
    case (state)
      IDLE:
      begin
        w_buf_en = 1'b0;
        nl_dpe_control = 2'b00;
        shift_add_control = 1'b0;
        // shift_add_bypass = shift_add_bypass_ctrl;
        shift_add_bypass = 1'b0;
        load_output_reg = 1'b0;
      end

      LOAD_PATCH:
      begin
        w_buf_en = 1'b0;
        nl_dpe_control = 2'b00;
        shift_add_control = 1'b0;
        shift_add_bypass = shift_add_bypass_ctrl;
        load_output_reg = 1'b0;
      end

      RUN_DPE:
      begin
        w_buf_en = 1'b1;
        nl_dpe_control = 2'b11;
        shift_add_control = MSB_SA_Ready;
        shift_add_bypass = shift_add_bypass_ctrl;
        load_output_reg = dpe_done;
      end

      OUTPUT_RESULT:
      begin
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
  always @(posedge clk)
  begin
    if (rst)
    begin
      sram_data_out <= {DATA_WIDTH{1'b0}};
    end
    else
    begin
      sram_data_out <= mem[r_addr];
    end
  end
  always @(posedge clk)
  begin
    if (w_en)
    begin
      mem[w_addr] <= sram_data_in;
    end
  end

endmodule


// module global_controller #(
//     parameter N_Layers = 1
// )(
//     input wire clk,
//     input wire rst,
//     input wire ready_L1,     // trigger/signal from nl_dpe indicating new data can be read
//     input wire valid_Ln,            // Valid signal to enable new operation
//     input wire valid,
//     output reg ready,                 // Ready signal indicating operation is done
//     output reg valid_L1,
//     output reg ready_Ln
// );

//     wire busy;
//     reg stall;

//     // valid and ready control
//     always @(posedge clk or posedge rst) begin
//         if (rst) begin
//             //valid_L1 <= 1'b0;
//             //ready <= 1'b0;
//             stall <= 0;
//         end else begin

//             if (stall) begin
//                 ready_Ln <= 1'b0;
//             end else begin
//                 ready_Ln <= 1'b1;
//             end

//             if(~valid) begin
//                 stall <= 0;
//             end else begin
//                 stall <= 1;
//             end
//         end
//     end

//     // always @* begin
//     //     ready <= ready_L1;
//     //     valid_L1 <= valid;
//     // end
//     always @(posedge clk or posedge rst) begin
//         if (rst) begin
//             valid_L1 <= 1'b0;
//             ready <= 1'b0;
//         end else begin
//             valid_L1 <= valid;
//             ready <= ready_L1;
//         end
//     end


// endmodule

module global_controller #(
    parameter N_Layers = 1
  )(
    input wire clk,
    input wire rst,
    input wire ready_L1,     // Ready from layer 1
    input wire valid_Ln,     // Valid from last layer (backpressure)
    input wire valid,        // External valid
    output reg ready,        // External ready
    output reg valid_L1,     // Valid to layer 1
    output reg ready_Ln      // Ready to last layer
  );

  // Combinational logic for stall (based on valid and backpressure)
  wire stall;
  assign stall = valid && (~ready_Ln || ~valid_Ln); // Stall if upstream valid but downstream not ready

  // Sequential logic for handshake signals
  always @(posedge clk or posedge rst)
  begin
    if (rst)
    begin
      valid_L1 <= 1'b0;
      ready_Ln <= 1'b0;
      ready <= 1'b0;
    end
    else
    begin
      // Propagate valid to layer 1 if not stalled
      valid_L1 <= (stall) ? 1'b0 : valid;

      // Propagate ready to last layer if not stalled
      ready_Ln <= (stall) ? 1'b0 : ready;

      // External ready follows ready_L1
      ready <= ready_L1;
    end
  end
endmodule
