// fc_top_synth.v — Synthesis-targeted variant of `fc_top.v`.
//
// Purpose:
//   The behavioral `fc_top.v` (Phase 2) is verified for cycle-accurate
//   functional simulation. Two constructs in that file are sim-only:
//
//   (1) DPE parameter overrides:
//         `dpe #(.KERNEL_WIDTH(R), .NUM_COLS(C), .DPE_BUF_WIDTH(BUF),
//                .COMPUTE_CYCLES(CCYC), .ACAM_MODE(0)) dpe_inst (...)`
//       These parameters configure the behavioral primitive's weight
//       array depth and FSM cycle count. The VTR arch XML's
//       `<model name="dpe">` declares NO parameters — VTR's Parmys
//       frontend rejects parameterized black-box instances.
//
//   (2) Hierarchical state access for debug:
//         `wire [2:0] dpe_state = gen_v[0].gen_h[0].gen_active.dpe_inst.state;`
//       References a register internal to the behavioral primitive that
//       does not exist in the VTR black-box model.
//
// This file is a `fc_top.v` clone — same FSM, same BRAM banks, same
// pipelined CLB tree — with module renamed to `fc_top_synth` and the
// two sim-only constructs stripped. The DPE instance now uses a bare
// `dpe dpe_inst (...)` form binding to the arch-XML hard block via
// `dpe_blackbox.v`.
//
// Used by `fc_verification/run_vtr_smoke.py` for the Task #89 VTR
// resource-and-Fmax smoke test. NOT used for iverilog functional
// simulation (which continues to drive the behavioral `fc_top.v` from
// `tb_fc.v` and `run_fc_smoke.py`).
//
// Spec anchors:
//   - fc_verification/rtl/fc_top.v (sim-verified Phase 2 master)
//   - fc_verification/rtl/dpe_blackbox.v (black-box DPE port contract)
//   - nl_dpe/nl_dpe_22nm_auto.xml (arch XML, `<model name="dpe">`)

`timescale 1ns / 1ps

module fc_top_synth #(
    parameter M               = 1,
    parameter K               = 128,
    parameter N               = 128,
    parameter R               = 256,
    parameter C               = 256,
    parameter BUF             = 40,
    parameter PRECISION       = 8,
    parameter PIPELINE_DEPTH  = 3,
    parameter ACAM_CYCLES     = 0,
    parameter ACTIVATION_MODE = 0,
    parameter HAS_ACAM        = 1
) (
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 start,
    output reg                  done,

    // Phase 2: input BRAM write port (TB-driven, pre-start)
    input  wire                 in_bram_wen,
    input  wire [31:0]          in_bram_waddr,
    input  wire [BUF-1:0]       in_bram_wdata,

    // Phase 2: output BRAM read port (TB-driven, post-done)
    input  wire [31:0]          out_bram_raddr,
    output wire [BUF-1:0]       out_bram_rdata
);

    // ── Derived geometry ──────────────────────────────────────────────
    localparam V              = (K + R - 1) / R;
    localparam H              = (N + C - 1) / C;
    localparam EPS            = BUF / 8;
    localparam LCYC           = (R + EPS - 1) / EPS;
    localparam OCYC           = (C + EPS - 1) / EPS;
    localparam CCYC           = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES;
    localparam VH             = V * H;
    localparam IN_BANK_DEPTH  = M * LCYC;
    localparam OUT_BANK_DEPTH = M * OCYC;

    function integer clog2_fn;
        input integer val;
        integer i;
        begin
            clog2_fn = 0;
            for (i = val - 1; i > 0; i = i >> 1) clog2_fn = clog2_fn + 1;
        end
    endfunction
    localparam TREE_PIPE = (V > 1) ? clog2_fn(V) : 0;

    localparam ACT_NEEDED = ((V > 1) ||
                             ((ACTIVATION_MODE == 1) && (HAS_ACAM == 0))) ? 1 : 0;
    localparam DO_RELU = (ACTIVATION_MODE == 1) ? 1 : 0;

    integer cycle_count;
    initial cycle_count = 0;
    always @(posedge clk) cycle_count <= cycle_count + 1;

    integer t_first_load;
    integer t_done;

    // Shared DPE control
    reg                 w_buf_en;
    reg  [1:0]          nl_dpe_control;
    reg                 shift_add_control;
    reg                 shift_add_bypass;
    reg                 load_output_reg;
    reg                 load_input_reg;

    reg  [BUF-1:0]      data_in_v      [0:V-1];

    wire [BUF-1:0]      data_out_vh [0:V*H-1];
    wire                dpe_done_vh [0:V*H-1];

    // FSM states
    localparam S_IDLE       = 4'd0;
    localparam S_LOAD       = 4'd2;
    localparam S_HOLD       = 4'd3;
    localparam S_OUT_CAP    = 4'd4;
    localparam S_ACT_FINAL  = 4'd5;
    localparam S_DONE       = 4'd6;
    reg [3:0] state;

    reg [31:0] read_idx_flat;
    reg [31:0] drive_idx_flat;
    reg        read_active;
    reg        drive_active;
    reg [31:0] hold_count;

    // Input BRAM banks (V parallel; BUF-bit wide; depth M*LCYC)
    wire [BUF-1:0] in_bram_rdata_v [0:V-1];

    genvar gvb;
    generate
        for (gvb = 0; gvb < V; gvb = gvb + 1) begin : gen_in_bram
            localparam integer GV_BASE = gvb * IN_BANK_DEPTH;
            reg [BUF-1:0] storage [0:IN_BANK_DEPTH-1];
            reg [BUF-1:0] rdata_reg;
            integer iii;
            initial begin
                for (iii = 0; iii < IN_BANK_DEPTH; iii = iii + 1)
                    storage[iii] = {BUF{1'b0}};
                rdata_reg = {BUF{1'b0}};
            end
            always @(posedge clk) begin
                if (in_bram_wen &&
                    in_bram_waddr >= GV_BASE &&
                    in_bram_waddr <  GV_BASE + IN_BANK_DEPTH) begin
                    storage[in_bram_waddr - GV_BASE] <= in_bram_wdata;
                end
                if (read_active && read_idx_flat < IN_BANK_DEPTH)
                    rdata_reg <= storage[read_idx_flat];
                else
                    rdata_reg <= {BUF{1'b0}};
            end
            assign in_bram_rdata_v[gvb] = rdata_reg;
        end
    endgenerate

    // Output BRAM banks (H parallel; BUF-bit wide; depth M*OCYC)
    reg  [31:0]    out_bram_waddr_h [0:H-1];
    reg            out_bram_wen_h   [0:H-1];
    reg  [BUF-1:0] out_bram_wdata_h [0:H-1];

    wire [BUF-1:0] out_bram_rdata_h [0:H-1];
    wire           out_rd_gh_match_h [0:H-1];

    genvar ghb;
    generate
        for (ghb = 0; ghb < H; ghb = ghb + 1) begin : gen_out_bram
            localparam integer GH_BASE = ghb * OUT_BANK_DEPTH;
            reg [BUF-1:0] storage [0:OUT_BANK_DEPTH-1];
            reg [BUF-1:0] rdata_reg;
            reg           rd_match_reg;
            integer iii;
            initial begin
                for (iii = 0; iii < OUT_BANK_DEPTH; iii = iii + 1)
                    storage[iii] = {BUF{1'b0}};
                rdata_reg    = {BUF{1'b0}};
                rd_match_reg = 1'b0;
            end
            wire rd_match = (out_bram_raddr >= GH_BASE) &&
                            (out_bram_raddr <  GH_BASE + OUT_BANK_DEPTH);
            wire [31:0] rd_inner = out_bram_raddr - GH_BASE;
            wire [31:0] wr_inner = out_bram_waddr_h[ghb];
            always @(posedge clk) begin
                if (out_bram_wen_h[ghb]) begin
                    storage[wr_inner] <= out_bram_wdata_h[ghb];
                end
                if (rd_inner < OUT_BANK_DEPTH)
                    rdata_reg <= storage[rd_inner];
                else
                    rdata_reg <= {BUF{1'b0}};
                rd_match_reg <= rd_match;
            end
            assign out_bram_rdata_h[ghb] = rdata_reg;
            assign out_rd_gh_match_h[ghb] = rd_match_reg;
        end
    endgenerate

    integer rd_sel;
    reg [BUF-1:0] out_bram_rdata_mux;
    always @(*) begin
        out_bram_rdata_mux = {BUF{1'b0}};
        for (rd_sel = 0; rd_sel < H; rd_sel = rd_sel + 1) begin
            if (out_rd_gh_match_h[rd_sel])
                out_bram_rdata_mux = out_bram_rdata_h[rd_sel];
        end
    end
    assign out_bram_rdata = out_bram_rdata_mux;

    // ── DPE instances (V*H grid) — bare black-box, no parameters ──────
    // Width contract: data_in/data_out are BUF=40 bits (NL-DPE).
    genvar gv, gh;
    generate
        for (gv = 0; gv < V; gv = gv + 1) begin : gen_v
            for (gh = 0; gh < H; gh = gh + 1) begin : gen_h
                begin : gen_active
                    wire                msb_sa_ready_w;
                    wire                reg_full_w;
                    wire                shift_add_done_w;
                    wire                shift_add_bypass_ctrl_w;
                    dpe dpe_inst (
                        .clk(clk),
                        .reset(reset),
                        .data_in(data_in_v[gv]),
                        .nl_dpe_control(nl_dpe_control),
                        .shift_add_control(shift_add_control),
                        .w_buf_en(w_buf_en),
                        .shift_add_bypass(shift_add_bypass),
                        .load_output_reg(load_output_reg),
                        .load_input_reg(load_input_reg),
                        .MSB_SA_Ready(msb_sa_ready_w),
                        .data_out(data_out_vh[gv*H + gh]),
                        .dpe_done(dpe_done_vh[gv*H + gh]),
                        .reg_full(reg_full_w),
                        .shift_add_done(shift_add_done_w),
                        .shift_add_bypass_ctrl(shift_add_bypass_ctrl_w)
                    );
                end
            end
        end
    endgenerate

    // (Sim-only `dpe_state` hierarchical wire removed — VTR black-box
    //  has no `state` register.)

    // ── Registered handshake on dpe_done + data_out_vh ────────────────
    wire dpe_done_or = dpe_done_vh[0];
    reg  dpe_done_r;
    reg  [BUF-1:0] data_out_vh_r [0:V*H-1];
    integer dor_i;
    always @(posedge clk) begin
        if (reset) begin
            dpe_done_r <= 1'b0;
            for (dor_i = 0; dor_i < V*H; dor_i = dor_i + 1)
                data_out_vh_r[dor_i] <= {BUF{1'b0}};
        end else begin
            dpe_done_r <= dpe_done_or;
            for (dor_i = 0; dor_i < V*H; dor_i = dor_i + 1)
                data_out_vh_r[dor_i] <= data_out_vh[dor_i];
        end
    end

    reg [31:0] m_out_idx_ahead;
    reg [31:0] out_idx_ahead;

    localparam OUT_PIPE = TREE_PIPE + 2;

    reg        valid_pipe [0:OUT_PIPE-1];
    reg [31:0] m_pipe     [0:OUT_PIPE-1];
    reg [31:0] oidx_pipe  [0:OUT_PIPE-1];

    reg signed [31:0] sxt_data [0:H*EPS*V-1];
    integer sx_gh, sx_b, sx_gv;
    reg signed [7:0] sx_byte;
    always @(*) begin
        for (sx_gh = 0; sx_gh < H; sx_gh = sx_gh + 1) begin
            for (sx_b = 0; sx_b < EPS; sx_b = sx_b + 1) begin
                for (sx_gv = 0; sx_gv < V; sx_gv = sx_gv + 1) begin
                    sx_byte = data_out_vh_r[sx_gv*H + sx_gh][sx_b*8 +: 8];
                    sxt_data[sx_gh*EPS*V + sx_b*V + sx_gv]
                        = $signed({{24{sx_byte[7]}}, sx_byte});
                end
            end
        end
    end

    reg signed [31:0] tree_data [0:TREE_PIPE][0:H*EPS*V-1];

    integer ts_s, ts_gh, ts_b, ts_v;
    always @(posedge clk) begin
        if (dpe_done_r) begin
            for (ts_v = 0; ts_v < H*EPS*V; ts_v = ts_v + 1)
                tree_data[0][ts_v] <= sxt_data[ts_v];
        end
        for (ts_s = 1; ts_s <= TREE_PIPE; ts_s = ts_s + 1) begin
            for (ts_gh = 0; ts_gh < H; ts_gh = ts_gh + 1) begin
                for (ts_b = 0; ts_b < EPS; ts_b = ts_b + 1) begin
                    for (ts_v = 0; ts_v < V; ts_v = ts_v + 1) begin
                        if (ts_v * 2 + 1 < V) begin
                            tree_data[ts_s][ts_gh*EPS*V + ts_b*V + ts_v]
                                <= tree_data[ts_s-1][ts_gh*EPS*V + ts_b*V + ts_v*2]
                                 + tree_data[ts_s-1][ts_gh*EPS*V + ts_b*V + ts_v*2 + 1];
                        end else if (ts_v * 2 < V) begin
                            tree_data[ts_s][ts_gh*EPS*V + ts_b*V + ts_v]
                                <= tree_data[ts_s-1][ts_gh*EPS*V + ts_b*V + ts_v*2];
                        end else begin
                            tree_data[ts_s][ts_gh*EPS*V + ts_b*V + ts_v] <= 32'sd0;
                        end
                    end
                end
            end
        end
    end

    integer pi;
    always @(posedge clk) begin
        if (reset) begin
            for (pi = 0; pi < OUT_PIPE; pi = pi + 1) begin
                valid_pipe[pi] <= 1'b0;
                m_pipe[pi]     <= 32'd0;
                oidx_pipe[pi]  <= 32'd0;
            end
        end else begin
            for (pi = OUT_PIPE - 1; pi > 0; pi = pi - 1) begin
                valid_pipe[pi] <= valid_pipe[pi-1];
                m_pipe[pi]     <= m_pipe[pi-1];
                oidx_pipe[pi]  <= oidx_pipe[pi-1];
            end
            if (dpe_done_r && m_out_idx_ahead < M) begin
                valid_pipe[0] <= 1'b1;
                m_pipe[0]     <= m_out_idx_ahead;
                oidx_pipe[0]  <= out_idx_ahead;
            end else begin
                valid_pipe[0] <= 1'b0;
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            m_out_idx_ahead <= 32'd0;
            out_idx_ahead   <= 32'd0;
        end else if (dpe_done_r && m_out_idx_ahead < M) begin
            if (out_idx_ahead + 1 >= OCYC) begin
                out_idx_ahead   <= 32'd0;
                m_out_idx_ahead <= m_out_idx_ahead + 1;
            end else begin
                out_idx_ahead <= out_idx_ahead + 1;
            end
        end
    end

    integer wt_gh, wt_b;
    always @(posedge clk) begin
        if (reset) begin
            for (wt_gh = 0; wt_gh < H; wt_gh = wt_gh + 1) begin
                out_bram_wen_h[wt_gh]   <= 1'b0;
                out_bram_waddr_h[wt_gh] <= 32'd0;
                out_bram_wdata_h[wt_gh] <= {BUF{1'b0}};
            end
        end else begin
            for (wt_gh = 0; wt_gh < H; wt_gh = wt_gh + 1) begin
                out_bram_wen_h[wt_gh] <= 1'b0;
            end
            if (valid_pipe[TREE_PIPE]) begin
                for (wt_gh = 0; wt_gh < H; wt_gh = wt_gh + 1) begin
                    out_bram_waddr_h[wt_gh]
                        <= m_pipe[TREE_PIPE] * OCYC + oidx_pipe[TREE_PIPE];
                    out_bram_wen_h[wt_gh] <= 1'b1;
                    for (wt_b = 0; wt_b < EPS; wt_b = wt_b + 1) begin
                        if (DO_RELU && ACT_NEEDED &&
                            tree_data[TREE_PIPE]
                                [wt_gh*EPS*V + wt_b*V + 0][7]) begin
                            out_bram_wdata_h[wt_gh][wt_b*8 +: 8] <= 8'h00;
                        end else begin
                            out_bram_wdata_h[wt_gh][wt_b*8 +: 8]
                                <= tree_data[TREE_PIPE]
                                    [wt_gh*EPS*V + wt_b*V + 0][7:0];
                        end
                    end
                end
            end
        end
    end

    wire last_cap_commit_now = valid_pipe[OUT_PIPE-1] &&
                                (m_pipe[OUT_PIPE-1] == M - 1) &&
                                (oidx_pipe[OUT_PIPE-1] == OCYC - 1);

    integer fsm_gv;
    always @(posedge clk) begin
        if (reset) begin
            state             <= S_IDLE;
            done              <= 1'b0;
            w_buf_en          <= 1'b0;
            nl_dpe_control    <= 2'b00;
            shift_add_control <= 1'b0;
            shift_add_bypass  <= 1'b0;
            load_output_reg   <= 1'b0;
            load_input_reg    <= 1'b0;
            hold_count        <= 32'd0;
            t_first_load      <= -1;
            t_done            <= -1;
            read_idx_flat     <= 32'd0;
            drive_idx_flat    <= 32'd0;
            read_active       <= 1'b0;
            drive_active      <= 1'b0;
            for (fsm_gv = 0; fsm_gv < V; fsm_gv = fsm_gv + 1)
                data_in_v[fsm_gv] <= {BUF{1'b0}};
        end else begin
            w_buf_en      <= 1'b0;
            drive_active  <= read_active;
            drive_idx_flat <= read_idx_flat;

            if (state == S_LOAD) begin
                if (read_idx_flat + 1 < M * LCYC) begin
                    read_idx_flat <= read_idx_flat + 1;
                    read_active   <= 1'b1;
                end else if (read_idx_flat + 1 == M * LCYC) begin
                    read_idx_flat <= read_idx_flat + 1;
                    read_active   <= 1'b0;
                end else begin
                    read_active <= 1'b0;
                end
            end

            if (drive_active) begin
                w_buf_en <= 1'b1;
                for (fsm_gv = 0; fsm_gv < V; fsm_gv = fsm_gv + 1)
                    data_in_v[fsm_gv] <= in_bram_rdata_v[fsm_gv];
            end

            case (state)
                S_IDLE: begin
                    done           <= 1'b0;
                    nl_dpe_control <= 2'b00;
                    if (start) begin
                        read_idx_flat <= 32'd0;
                        read_active   <= 1'b1;
                        drive_active  <= 1'b0;
                        nl_dpe_control <= 2'b11;
                        t_first_load  <= cycle_count + 3;
                        state         <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    nl_dpe_control <= 2'b11;
                    if (drive_active && !read_active) begin
                        hold_count <= CCYC;
                        state      <= S_HOLD;
                    end
                end

                S_HOLD: begin
                    nl_dpe_control <= 2'b11;
                    if (hold_count > 1) begin
                        hold_count <= hold_count - 1;
                    end else begin
                        nl_dpe_control <= 2'b00;
                        state          <= S_OUT_CAP;
                    end
                end

                S_OUT_CAP: begin
                    nl_dpe_control <= 2'b00;
                    if (last_cap_commit_now) begin
                        t_done <= cycle_count;
                        if (ACT_NEEDED) state <= S_ACT_FINAL;
                        else            state <= S_DONE;
                    end
                end

                S_ACT_FINAL: begin
                    nl_dpe_control <= 2'b00;
                    t_done <= cycle_count;
                    state  <= S_DONE;
                end

                S_DONE: begin
                    done           <= 1'b1;
                    nl_dpe_control <= 2'b00;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
