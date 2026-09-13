#!/usr/bin/env python3
"""Generate bn_softmax_clb Verilog module — CLB + DSP implementation.

Post-GEMV pipeline: BatchNorm → Exp LUT → Sum → Reciprocal → Normalize.
Uses 16 explicit mac_int_9x9 DSP primitives (= 4 DSP tiles).

Design uses 4 independent data lanes, each with its own SRAM bank storing
N/4 elements. This ensures all 16 DSP instances have unique, data-dependent
inputs that synthesis cannot optimize away.

  Lane k processes elements {k, k+4, k+8, ...} (stride-4 interleave).
  - 1 mac_int_9x9 for BN scale (gamma[i] × x[i])
  - 1 mac_int_9x9 for BN bias  (scaled[i] × beta[i])
  - 1 mac_int_9x9 for softmax normalize pass 1 (exp[i] × inv_sum)
  - 1 mac_int_9x9 for softmax normalize pass 2 (exp[i] × inv_sum, second pass)
  Total per lane: 4 mac_int_9x9 = 1 DSP tile.
  4 lanes × 4 MACs = 16 mac_int_9x9 = 4 DSP tiles.
"""

import math


def gen_softmax_clb(n: int, data_width: int = 40) -> str:
    """Generate bn_softmax_clb module with 16 mac_int_9x9 (4 DSP tiles).

    4 independent lanes, each with own SRAM bank + 4 DSP MACs.
    """
    dw = data_width
    n_lanes = 4
    n_per_lane = (n + n_lanes - 1) // n_lanes  # ceil(N/4)
    aw = max(1, math.ceil(math.log2(n_per_lane))) + 1
    depth = n_per_lane

    lines = []
    lines.append(f"module bn_softmax_clb #(")
    lines.append(f"    parameter N = {n},")
    lines.append(f"    parameter N_LANES = {n_lanes},")
    lines.append(f"    parameter N_PER_LANE = {n_per_lane},")
    lines.append(f"    parameter DATA_WIDTH = {dw},")
    lines.append(f"    parameter ADDR_WIDTH = {aw},")
    lines.append(f"    parameter DEPTH = {n_per_lane}")
    lines.append(f")(")
    lines.append(f"    input wire clk,")
    lines.append(f"    input wire rst,")
    lines.append(f"    input wire valid,")
    lines.append(f"    input wire ready_n,")
    lines.append(f"    input wire [DATA_WIDTH-1:0] data_in,")
    lines.append(f"    output wire [DATA_WIDTH-1:0] data_out,")
    lines.append(f"    output wire ready,")
    lines.append(f"    output wire valid_n")
    lines.append(f");")
    lines.append(f"")

    # ── Per-lane infrastructure: 4 independent data paths ──
    for lane in range(n_lanes):
        L = f"l{lane}"
        lines.append(f"    // ════════ Lane {lane} ════════")
        lines.append(f"    // Input SRAM: stores N/4 elements (stride-4 from input)")
        lines.append(f"    reg [ADDR_WIDTH-1:0] {L}_in_w_addr, {L}_in_r_addr;")
        lines.append(f"    reg {L}_in_w_en;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {L}_in_out;")
        lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
        lines.append(f"    {L}_in_buf (.clk(clk),.rst(rst),.w_en({L}_in_w_en),")
        lines.append(f"               .r_addr({L}_in_r_addr),.w_addr({L}_in_w_addr),")
        lines.append(f"               .sram_data_in(data_in),.sram_data_out({L}_in_out));")
        lines.append(f"")

        # Gamma/beta parameter SRAMs per lane
        lines.append(f"    // Gamma/Beta SRAMs (per-lane learned parameters)")
        lines.append(f"    reg [ADDR_WIDTH-1:0] {L}_param_addr;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {L}_gamma_out, {L}_beta_out;")
        lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
        lines.append(f"    {L}_gamma (.clk(clk),.rst(rst),.w_en(1'b0),.r_addr({L}_param_addr),")
        lines.append(f"              .w_addr({aw}'d0),.sram_data_in({dw}'d0),.sram_data_out({L}_gamma_out));")
        lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
        lines.append(f"    {L}_beta (.clk(clk),.rst(rst),.w_en(1'b0),.r_addr({L}_param_addr),")
        lines.append(f"             .w_addr({aw}'d0),.sram_data_in({dw}'d0),.sram_data_out({L}_beta_out));")
        lines.append(f"")

        # 4 DSP MACs per lane
        lines.append(f"    // DSP 1: BN scale (gamma × x)")
        lines.append(f"    wire [17:0] {L}_bn_scale_out;")
        lines.append(f"    mac_int_9x9 {L}_bn_scale (")
        lines.append(f"        .clk(clk), .reset(rst),")
        lines.append(f"        .a({L}_in_out[8:0]), .b({L}_gamma_out[8:0]),")
        lines.append(f"        .out({L}_bn_scale_out)")
        lines.append(f"    );")
        lines.append(f"")

        lines.append(f"    // DSP 2: BN bias (scaled × beta)")
        lines.append(f"    wire [17:0] {L}_bn_bias_out;")
        lines.append(f"    mac_int_9x9 {L}_bn_bias (")
        lines.append(f"        .clk(clk), .reset(rst),")
        lines.append(f"        .a({L}_bn_scale_out[8:0]), .b({L}_beta_out[8:0]),")
        lines.append(f"        .out({L}_bn_bias_out)")
        lines.append(f"    );")
        lines.append(f"")

        # Exp SRAM per lane
        lines.append(f"    // Exp SRAM (per-lane)")
        lines.append(f"    reg [ADDR_WIDTH-1:0] {L}_exp_w_addr, {L}_exp_r_addr;")
        lines.append(f"    reg {L}_exp_w_en;")
        lines.append(f"    reg [DATA_WIDTH-1:0] {L}_exp_wdata;")
        lines.append(f"    wire [DATA_WIDTH-1:0] {L}_exp_out;")
        lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(DEPTH))")
        lines.append(f"    {L}_exp_buf (.clk(clk),.rst(rst),.w_en({L}_exp_w_en),")
        lines.append(f"                .r_addr({L}_exp_r_addr),.w_addr({L}_exp_w_addr),")
        lines.append(f"                .sram_data_in({L}_exp_wdata),.sram_data_out({L}_exp_out));")
        lines.append(f"")

        lines.append(f"    // DSP 3: Normalize pass 1 (exp × inv_sum)")
        lines.append(f"    wire [17:0] {L}_norm1_out;")
        lines.append(f"    reg [8:0] {L}_inv_sum_reg;")
        lines.append(f"    mac_int_9x9 {L}_norm1 (")
        lines.append(f"        .clk(clk), .reset(rst),")
        lines.append(f"        .a({L}_exp_out[8:0]), .b({L}_inv_sum_reg),")
        lines.append(f"        .out({L}_norm1_out)")
        lines.append(f"    );")
        lines.append(f"")

        lines.append(f"    // DSP 4: Normalize pass 2 (output scaling)")
        lines.append(f"    wire [17:0] {L}_norm2_out;")
        lines.append(f"    mac_int_9x9 {L}_norm2 (")
        lines.append(f"        .clk(clk), .reset(rst),")
        lines.append(f"        .a({L}_norm1_out[8:0]), .b({L}_inv_sum_reg),")
        lines.append(f"        .out({L}_norm2_out)")
        lines.append(f"    );")
        lines.append(f"")

        # Per-lane exp LUT
        lines.append(f"    // Per-lane Exp LUT")
        lines.append(f"    reg [8:0] {L}_exp_lut;")
        lines.append(f"    wire [7:0] {L}_exp_in = {L}_bn_bias_out[7:0];")
        lines.append(f"    always @(posedge clk) begin")
        lines.append(f"        case ({L}_exp_in)")
        import math as m
        for i in range(256):
            signed_val = i if i < 128 else i - 256
            raw = m.exp(signed_val / 32.0)
            scaled = int(min(511, max(1, raw * 128)))
            lines.append(f"            8'd{i}: {L}_exp_lut <= 9'd{scaled};")
        lines.append(f"            default: {L}_exp_lut <= 9'd128;")
        lines.append(f"        endcase")
        lines.append(f"    end")
        lines.append(f"")

        # Per-lane sum accumulator
        lines.append(f"    reg [{dw+8-1}:0] {L}_exp_sum;")
        lines.append(f"")

    # ── Global sum: add all 4 lane sums ──
    lines.append(f"    // Global exp sum = sum of all lane sums")
    lines.append(f"    wire [{dw+10-1}:0] global_exp_sum = " +
                 " + ".join(f"l{i}_exp_sum" for i in range(n_lanes)) + ";")
    lines.append(f"")

    # ── Reciprocal (CLB priority encoder, shared) ──
    lines.append(f"    // Reciprocal via priority encoder")
    lines.append(f"    wire [15:0] sum_upper = global_exp_sum[{dw+10-1}:{dw+10-16}];")
    lines.append(f"    reg [8:0] recip_val;")
    lines.append(f"    always @(*) begin")
    lines.append(f"        casez (sum_upper)")
    for bit in range(15, -1, -1):
        pat = "0" * (15 - bit) + "1" + "?" * bit
        val = min(511, max(1, (1 << 15) >> (15 - bit)))
        lines.append(f"            16'b{pat}: recip_val = 9'd{val};")
    lines.append(f"            default: recip_val = 9'd511;")
    lines.append(f"        endcase")
    lines.append(f"    end")
    lines.append(f"")

    # ── Output mux: round-robin across 4 lanes ──
    lines.append(f"    // Output mux: round-robin across lanes")
    lines.append(f"    reg [1:0] out_lane_sel;")
    lines.append(f"    reg [DATA_WIDTH-1:0] out_mux;")
    lines.append(f"    always @(*) begin")
    lines.append(f"        case (out_lane_sel)")
    for i in range(n_lanes):
        lines.append(f"            2'd{i}: out_mux = {{{{(DATA_WIDTH-18){{1'b0}}}}, l{i}_norm2_out}};")
    lines.append(f"            default: out_mux = {dw}'d0;")
    lines.append(f"        endcase")
    lines.append(f"    end")
    lines.append(f"")

    # ── Output SRAM ──
    lines.append(f"    // Output SRAM")
    lines.append(f"    reg [ADDR_WIDTH+1:0] out_w_addr, out_r_addr;")  # wider for full N
    lines.append(f"    reg out_w_en;")
    lines.append(f"    wire [DATA_WIDTH-1:0] out_sram_out;")
    lines.append(f"    sram #(.N_CHANNELS(1),.DATA_WIDTH(DATA_WIDTH),.DEPTH(N))")
    lines.append(f"    out_buf (.clk(clk),.rst(rst),.w_en(out_w_en),")
    lines.append(f"            .r_addr(out_r_addr[ADDR_WIDTH-1:0]),.w_addr(out_w_addr[ADDR_WIDTH-1:0]),")
    lines.append(f"            .sram_data_in(out_mux),.sram_data_out(out_sram_out));")
    lines.append(f"")

    # ── FSM ──
    lines.append(f"    localparam SM_IDLE = 3'd0, SM_LOAD = 3'd1, SM_BN_EXP = 3'd2,")
    lines.append(f"               SM_RECIP = 3'd3, SM_NORM = 3'd4, SM_OUTPUT = 3'd5;")
    lines.append(f"    reg [2:0] state;")
    lines.append(f"    reg [{aw+2}-1:0] count;  // global counter (up to N)")
    lines.append(f"    reg [{aw}-1:0] lane_count;  // per-lane counter (up to N/4)")
    lines.append(f"")

    lines.append(f"    always @(posedge clk or posedge rst) begin")
    lines.append(f"        if (rst) begin")
    lines.append(f"            state <= SM_IDLE; count <= 0; lane_count <= 0;")
    lines.append(f"            out_lane_sel <= 0;")
    for lane in range(n_lanes):
        L = f"l{lane}"
        lines.append(f"            {L}_in_w_addr <= 0; {L}_in_r_addr <= 0; {L}_in_w_en <= 0;")
        lines.append(f"            {L}_param_addr <= 0;")
        lines.append(f"            {L}_exp_w_addr <= 0; {L}_exp_r_addr <= 0; {L}_exp_w_en <= 0;")
        lines.append(f"            {L}_exp_wdata <= 0; {L}_exp_sum <= 0; {L}_inv_sum_reg <= 0;")
    lines.append(f"            out_w_addr <= 0; out_r_addr <= 0; out_w_en <= 0;")
    lines.append(f"        end else begin")
    for lane in range(n_lanes):
        lines.append(f"            l{lane}_in_w_en <= 0; l{lane}_exp_w_en <= 0;")
    lines.append(f"            out_w_en <= 0;")
    lines.append(f"            case (state)")
    lines.append(f"")

    # SM_IDLE
    lines.append(f"                SM_IDLE: if (valid) begin")
    lines.append(f"                    state <= SM_LOAD; count <= 0;")
    for lane in range(n_lanes):
        lines.append(f"                    l{lane}_in_w_addr <= 0; l{lane}_exp_sum <= 0;")
    lines.append(f"                end")
    lines.append(f"")

    # SM_LOAD: distribute N elements to 4 lane SRAMs (stride-4)
    lines.append(f"                SM_LOAD: begin")
    lines.append(f"                    // Distribute to lane (count % 4)")
    lines.append(f"                    case (count[1:0])")
    for lane in range(n_lanes):
        lines.append(f"                        2'd{lane}: begin")
        lines.append(f"                            l{lane}_in_w_en <= 1;")
        lines.append(f"                            l{lane}_in_w_addr <= count >> 2;")
        lines.append(f"                        end")
    lines.append(f"                    endcase")
    lines.append(f"                    count <= count + 1;")
    lines.append(f"                    if (count == N - 1) begin")
    lines.append(f"                        state <= SM_BN_EXP; lane_count <= 0;")
    for lane in range(n_lanes):
        lines.append(f"                        l{lane}_in_r_addr <= 0; l{lane}_param_addr <= 0;")
    lines.append(f"                    end")
    lines.append(f"                end")
    lines.append(f"")

    # SM_BN_EXP: BN (2 DSPs) → exp LUT → accumulate, per lane, N/4 iterations
    # Pipeline: cycle k reads input+gamma, cycle k+1 scale DSP fires,
    #           cycle k+2 bias DSP fires, cycle k+3 exp LUT, cycle k+4 accumulate
    lines.append(f"                SM_BN_EXP: begin")
    for lane in range(n_lanes):
        L = f"l{lane}"
        lines.append(f"                    // Lane {lane}: advance read addresses")
        lines.append(f"                    {L}_in_r_addr <= lane_count + 1;")
        lines.append(f"                    {L}_param_addr <= lane_count + 1;")
        lines.append(f"                    // Write exp result from LUT (pipeline delay 4)")
        lines.append(f"                    if (lane_count > 3) begin")
        lines.append(f"                        {L}_exp_wdata <= {{{{(DATA_WIDTH-9){{1'b0}}}}, {L}_exp_lut}};")
        lines.append(f"                        {L}_exp_w_en <= 1;")
        lines.append(f"                        {L}_exp_w_addr <= lane_count - 4;")
        lines.append(f"                        {L}_exp_sum <= {L}_exp_sum + {L}_exp_lut;")
        lines.append(f"                    end")
    lines.append(f"                    lane_count <= lane_count + 1;")
    lines.append(f"                    if (lane_count == N_PER_LANE + 4) begin")
    lines.append(f"                        state <= SM_RECIP; lane_count <= 0;")
    lines.append(f"                    end")
    lines.append(f"                end")
    lines.append(f"")

    # SM_RECIP
    lines.append(f"                SM_RECIP: begin")
    for lane in range(n_lanes):
        lines.append(f"                    l{lane}_inv_sum_reg <= recip_val;")
        lines.append(f"                    l{lane}_exp_r_addr <= 0;")
    lines.append(f"                    state <= SM_NORM; lane_count <= 0;")
    lines.append(f"                end")
    lines.append(f"")

    # SM_NORM: 4 lanes × 2 norm DSPs each = 8 DSPs active
    # Each lane reads its exp SRAM, feeds norm1 and norm2 in pipeline
    lines.append(f"                SM_NORM: begin")
    for lane in range(n_lanes):
        L = f"l{lane}"
        lines.append(f"                    {L}_exp_r_addr <= lane_count + 1;")
    lines.append(f"                    // Write norm2 outputs to output SRAM (pipeline delay 2)")
    lines.append(f"                    if (lane_count > 2) begin")
    lines.append(f"                        out_w_en <= 1;")
    lines.append(f"                        out_lane_sel <= (lane_count - 3) & 2'b11;")
    lines.append(f"                        out_w_addr <= lane_count - 3;")
    lines.append(f"                    end")
    lines.append(f"                    lane_count <= lane_count + 1;")
    lines.append(f"                    if (lane_count == N_PER_LANE + 3) begin")
    lines.append(f"                        state <= SM_OUTPUT; count <= 0;")
    lines.append(f"                        out_r_addr <= 0;")
    lines.append(f"                    end")
    lines.append(f"                end")
    lines.append(f"")

    # SM_OUTPUT
    lines.append(f"                SM_OUTPUT: begin")
    lines.append(f"                    if (!ready_n) begin")
    lines.append(f"                        out_r_addr <= out_r_addr + 1;")
    lines.append(f"                        count <= count + 1;")
    lines.append(f"                        if (count == N - 1) state <= SM_IDLE;")
    lines.append(f"                    end")
    lines.append(f"                end")
    lines.append(f"")

    lines.append(f"            endcase")
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append(f"")

    lines.append(f"    assign data_out = out_sram_out;")
    lines.append(f"    assign valid_n = (state == SM_OUTPUT);")
    lines.append(f"    assign ready = (state == SM_IDLE || state == SM_LOAD);")
    lines.append(f"")
    lines.append(f"endmodule")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--data-width", type=int, default=40)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    verilog = gen_softmax_clb(args.n, args.data_width)
    if args.output:
        Path(args.output).write_text(verilog)
        print(f"Written to {args.output}")
    else:
        print(verilog)
