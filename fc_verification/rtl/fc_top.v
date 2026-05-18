// fc_top.v -- FC / GEMM top module (Stages 1A + 1B + 1C) with
// drain-load overlap controller (FIDELITY_METHODOLOGY.md §4).
//
// Phase 2 (Task #88, 2026-05-15): SYNTHESIZABLE WRAPPER REWRITE
//
//   The wrapper now uses inferrable BRAM ports for both input and output
//   storage (no more hierarchical-forced flat-reg arrays), a pipelined
//   CLB adder tree (⌈log₂(V)⌉ registered stages for V > 1; pass-through
//   for V = 1), and a registered handshake on the dpe_done sample.
//
//   These additions surface the real-silicon overhead that the Phase 1
//   model abstracted away.  Phase 2 cycle formula (measured empirically;
//   see FC_GEMM_WALKTHROUGH §8 for the cycle-math derivation):
//
//       T_fill_phase2 = LCYC + CCYC + OCYC + 6 + TREE_PIPE
//                     = T_fill_phase1 + 4 + TREE_PIPE
//       T_steady      = max(LCYC, CCYC, OCYC)        (Task #99 double-buffer:
//                                                     no +PRECISION term)
//       T(M)          = T_fill_phase2 + (M-1)*T_steady
//       total_cycles  = T(M) + (1 if CLB_NEEDED else 0)
//       TREE_PIPE     = ⌈log₂(V)⌉ for V > 1, else 0
//       CLB_NEEDED    = (V > 1) OR (ACTIVATION_MODE == 1 AND HAS_ACAM == 0)
//
//   The structural +4 (over Phase 1's +2 baseline) decomposes as:
//     +1 BRAM-read pipeline on LOAD (net delta after t_first_load shift)
//     +1 data_out_vh_r register (DPE→wrapper handshake)
//     +1 stage-0 sign-extend latch entering the CLB tree pipeline
//     +1 BRAM-write tap NBA + BRAM internal write commit
//   For TREE_PIPE registered tree stages (V > 1), each stage adds 1
//   more cycle of capture-side latency.
//
//   T_steady = max(LCYC, CCYC, OCYC) is unchanged: the wrapper's BRAM-
//   read, tree, and BRAM-write pipelines are all single-cycle pipelined
//   and absorb into steady-state cadence at M=M+1 boundaries.
//
// Module name      : fc_top
// Workload         : Y[M x N] = X[M x K] @ W[K x N] (+ optional ReLU)
//
// Phase 2 port surface (synthesizable BRAM ports):
//   input  wire                  clk
//   input  wire                  reset             // synchronous, active-high
//   input  wire                  start             // pulse to begin (TB asserts after BRAM setup)
//   output reg                   done              // FSM in S_DONE
//   // Input BRAM write port (TB-driven, pre-start)
//   input  wire                  in_bram_wen
//   input  wire [31:0]           in_bram_waddr     // flat: gv*(M*LCYC) + m*LCYC + strobe
//   input  wire [BUF-1:0]        in_bram_wdata     // EPS bytes packed (BUF bits)
//   // Output BRAM read port (TB-driven, post-done)
//   input  wire [31:0]           out_bram_raddr    // flat: gh*(M*OCYC) + m*OCYC + out_strobe
//   output wire [BUF-1:0]        out_bram_rdata    // EPS bytes packed; 2-cycle read latency
//
//   The DPE primitive's weight array is still TB-hierarchical-forced
//   (the DPE is a VTR black box; its internals are sim-only). Only the
//   wrapper's input/output SRAM gets synthesizable BRAM ports.
//
// Spec anchors:
//   - fc_verification/FC_RTL_PLAN.md §3-§5 (Stages 1A-1C)
//   - fc_verification/FIDELITY_METHODOLOGY.md §4 + §4.1 + §5 + §7
//   - fc_verification/FC_GEMM_WALKTHROUGH.md
//   - azurelily/IMC/imc_core/imc_core.py:run_gemm (simulator's cycle model)

`timescale 1ns / 1ps

module fc_top #(
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

    // ── Phase 2: input BRAM write port (TB-driven, pre-start) ───────
    input  wire                 in_bram_wen,
    input  wire [31:0]          in_bram_waddr,
    input  wire [BUF-1:0]       in_bram_wdata,

    // ── Phase 2: output BRAM read port (TB-driven, post-done) ───────
    input  wire [31:0]          out_bram_raddr,
    output wire [BUF-1:0]       out_bram_rdata
);

    // ── Derived geometry ──────────────────────────────────────────────────
    localparam V              = (K + R - 1) / R;     // K-tile count
    localparam H              = (N + C - 1) / C;     // N-tile count
    localparam EPS            = BUF / 8;              // bytes per strobe
    localparam LCYC           = (R + EPS - 1) / EPS;  // LOAD strobes per pass
    localparam OCYC           = (C + EPS - 1) / EPS;  // OUTPUT strobes per pass
    localparam CCYC           = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES;
    localparam VH             = V * H;

    // Per-bank BRAM depths.
    localparam IN_BANK_DEPTH  = M * LCYC;
    localparam OUT_BANK_DEPTH = M * OCYC;

    // CLB adder tree pipeline depth = ⌈log₂(V)⌉ for V > 1, else 0.
    function integer clog2_fn;
        input integer val;
        integer i;
        begin
            clog2_fn = 0;
            for (i = val - 1; i > 0; i = i >> 1) clog2_fn = clog2_fn + 1;
        end
    endfunction
    localparam TREE_PIPE = (V > 1) ? clog2_fn(V) : 0;

    // CLB-stage gating (mirrors simulator's run_gemm gate — see
    // FIDELITY_METHODOLOGY §5):
    //   CLB_NEEDED = (V > 1) || ((ACTIVATION_MODE == 1) && (HAS_ACAM == 0))
    localparam ACT_NEEDED = ((V > 1) ||
                             ((ACTIVATION_MODE == 1) && (HAS_ACAM == 0))) ? 1 : 0;
    localparam DO_RELU = (ACTIVATION_MODE == 1) ? 1 : 0;

    // ── Free-running cycle counter ──────────────────────────────────────
    integer cycle_count;
    initial cycle_count = 0;
    always @(posedge clk) cycle_count <= cycle_count + 1;

    integer t_first_load;
    integer t_done;

    // ── Shared DPE control ──────────────────────────────────────────────
    reg                 w_buf_en;
    reg  [1:0]          nl_dpe_control;
    reg                 shift_add_control;
    reg                 shift_add_bypass;
    reg                 load_output_reg;
    reg                 load_input_reg;

    // Per-gv data_in (NBA-driven from BRAM read result).
    reg  [BUF-1:0]      data_in_v      [0:V-1];

    // Per-(gv, gh) DPE outputs.
    wire [BUF-1:0]      data_out_vh [0:V*H-1];
    wire                dpe_done_vh [0:V*H-1];

    // ── FSM state encoding ──────────────────────────────────────────────
    //   S_IDLE      : reset / waiting for start
    //   S_LOAD      : driving w_buf_en for LCYC*M strobes (combined LOAD+
    //                 chained-row controller, with 1-cycle BRAM-read
    //                 lookahead pipeline)
    //   S_HOLD      : holding nl_dpe_control=2'b11 through compute
    //   S_OUT_CAP   : waiting for the last BRAM-write commit
    //   S_ACT_FINAL : +1 CLB cycle (V>1 tree fold or AL no_acam ReLU)
    //   S_DONE      : terminal; done=1
    localparam S_IDLE       = 4'd0;
    localparam S_LOAD       = 4'd2;
    localparam S_HOLD       = 4'd3;
    localparam S_OUT_CAP    = 4'd4;
    localparam S_ACT_FINAL  = 4'd5;
    localparam S_DONE       = 4'd6;
    reg [3:0] state;

    // ── LOAD controller pipeline registers ─────────────────────────────
    //   read_idx_flat = next BRAM read address (gv-agnostic: m*LCYC + s,
    //                   each bank reads its own GV_BASE-offset version)
    //   drive_idx_flat = the m*LCYC+s of the strobe being driven THIS
    //                   cycle (one cycle behind read_idx_flat)
    //   read_active   = BRAM read is fetching a real strobe this cycle
    //                   (raddr in range)
    //   drive_active  = w_buf_en should fire this cycle (one cycle
    //                   behind read_active)
    reg [31:0] read_idx_flat;
    reg [31:0] drive_idx_flat;     // unused for FSM logic; kept for
                                    // diagnostics
    reg        read_active;
    reg        drive_active;
    reg [31:0] hold_count;

    // ── Per-pass strobe tracking (Task #99 — double-buffered primitive) ─
    //
    // The faithful primitive's input substrate is now double-buffered
    // (Task #99): pass-(k+1)'s LOAD writes to the OTHER substrate while
    // pass-k's COMPUTE reads its substrate, so there is no inter-pass
    // stall in the wrapper -- read_active stays high continuously across
    // pass boundaries until the very last strobe of the last pass.
    //
    //   pass_strobe_idx : 0..LCYC-1   (strobe count within the current pass)
    //   pass_idx        : 0..M-1      (current pass index)
    //
    // Cadence math (see FIDELITY_METHODOLOGY.md and FC_GEMM_WALKTHROUGH):
    //   Pass-k strobe 0 DPE-accepted at U + k*LCYC.
    //   read_active=1 for M*LCYC consecutive cycles (back-to-back).
    //   This produces T_steady = LCYC cycles per pass.
    reg [31:0] pass_strobe_idx;
    reg [31:0] pass_idx;

    // ── Input BRAM banks (V parallel; BUF-bit wide; depth M*LCYC) ───────
    // BRAM read latency = 1 cycle. Read address is `read_idx_flat`
    // (combinational from a register, so deterministic at each posedge).
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

    // ── Output BRAM banks (H parallel; BUF-bit wide; depth M*OCYC) ──────
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

    // ── DPE instances (V*H grid) ───────────────────────────────────────
    genvar gv, gh;
    generate
        for (gv = 0; gv < V; gv = gv + 1) begin : gen_v
            for (gh = 0; gh < H; gh = gh + 1) begin : gen_h
                begin : gen_active
                    wire                msb_sa_ready_w;
                    wire                reg_full_w;
                    wire                shift_add_done_w;
                    wire                shift_add_bypass_ctrl_w;
                    `ifdef FC_TOP_ARCH_NLDPE
                    // Faithful NL-DPE primitive (Task #91/#97): bit-stratified
                    // single-substrate slice with LOAD-gated cadence. Takes
                    // PRECISION/PIPELINE_DEPTH/ACAM_CYCLES (not COMPUTE_CYCLES).
                    dpe #(
                        .KERNEL_WIDTH(R),
                        .NUM_COLS(C),
                        .DPE_BUF_WIDTH(BUF),
                        .PRECISION(PRECISION),
                        .PIPELINE_DEPTH(PIPELINE_DEPTH),
                        .ACAM_CYCLES(ACAM_CYCLES),
                        .ACAM_MODE(0)
                    ) dpe_inst (
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
                    `else
                    // Faithful Azure-Lily primitive (Task #92/#97): single-
                    // substrate 3-stage MAC+ADC+ShiftAdd; same LOAD-gate.
                    dpe #(
                        .KERNEL_WIDTH(R),
                        .NUM_COLS(C),
                        .DPE_BUF_WIDTH(BUF),
                        .PRECISION(PRECISION),
                        .PIPELINE_DEPTH(PIPELINE_DEPTH),
                        .ACAM_CYCLES(ACAM_CYCLES)
                    ) dpe_inst (
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
                    `endif
                end
            end
        end
    endgenerate

    wire [2:0] dpe_state = gen_v[0].gen_h[0].gen_active.dpe_inst.state;

    // ── DPE back-pressure (Task #99 — double-buffered primitive) ──────
    //
    // The faithful primitive asserts reg_full only during OUTPUT drain
    // (output_busy) now that the input substrate is double-buffered
    // (Task #99). The wrapper does NOT stall between passes -- LOAD is
    // a back-to-back stream of M*LCYC strobes -- because pass-(k+1)
    // writes the OTHER substrate while pass-k's COMPUTE consumes its
    // substrate. Under typical LCYC >> CCYC configs, OUTPUT drain
    // completes well before pass-(k+1)'s COMPUTE finishes, so back-
    // pressure is not expected in steady state.
    //
    // dpe_reg_full is kept exported for observability (waveform debug,
    // assertions); it is not used to directly gate w_buf_en.
    wire dpe_reg_full = gen_v[0].gen_h[0].gen_active.reg_full_w;

    // ── Registered handshake: dpe_done sample + data_out_vh sample ────
    //
    // Phase 2: register dpe_done AND the DPE's data_out_vh bus on the
    // boundary between the DPE primitive and the fc_top capture
    // pipeline. This is the "registered handshake" that adds +1 cycle
    // to the OUTPUT side of T_fill, and (equally important) gives the
    // synthesized wrapper a clean per-stage register barrier on a real
    // VTR-emitted netlist.
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

    // ── OUTPUT pipeline cursors ────────────────────────────────────────
    reg [31:0] m_out_idx_ahead;
    reg [31:0] out_idx_ahead;

    // OUT_PIPE stages:
    //   0           : sign-extend latch (cycle after dpe_done_r=1)
    //   1..TREE_PIPE: pairwise tree fold
    //   TREE_PIPE+1 : BRAM-write commit (cycle after out_bram_wen=1)
    localparam OUT_PIPE = TREE_PIPE + 2;

    reg        valid_pipe [0:OUT_PIPE-1];
    reg [31:0] m_pipe     [0:OUT_PIPE-1];
    reg [31:0] oidx_pipe  [0:OUT_PIPE-1];

    // Sign-extended raw bytes (combinational from data_out_vh_r, the
    // registered version of the DPE primitive's data_out bus).
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

    // Tree pipeline storage. Stage 0 = sign-extended slots (V per (gh, b)).
    // Stages 1..TREE_PIPE = pairwise reductions.
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

    // BRAM write tap: assert wen + wdata on the cycle valid_pipe[TREE_PIPE]
    // is high. BRAM write latency = 1 cycle (write commits on next posedge).
    // valid_pipe[OUT_PIPE-1] (= TREE_PIPE+1) marks the cycle the write
    // has committed.
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
                        // Apply ReLU LUT inline with the BRAM-write tap:
                        //   - If ACT_NEEDED && DO_RELU && bit-7 of the
                        //     truncated low-8-bits is set, clamp to 0.
                        //   - Otherwise pass through the low-8 bits.
                        // Bit-7 check matches the test-pattern oracle in
                        // tb_fc.v's expected_byte_fn (byte-level ReLU
                        // post-truncation, consistent across Phase 1/2).
                        // The +1 cycle for the CLB stage (ACT_NEEDED) is
                        // accounted for in S_ACT_FINAL (post-pipeline);
                        // the per-byte LUT itself folds combinationally
                        // into the BRAM-write tap (no extra latency).
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

    // ── Main FSM ────────────────────────────────────────────────────────
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
            pass_strobe_idx   <= 32'd0;
            pass_idx          <= 32'd0;
            for (fsm_gv = 0; fsm_gv < V; fsm_gv = fsm_gv + 1)
                data_in_v[fsm_gv] <= {BUF{1'b0}};
        end else begin
            // Default each cycle
            w_buf_en      <= 1'b0;
            drive_active  <= read_active;     // 1-cycle shift register
            // drive_idx_flat tracks the strobe BEING DRIVEN this cycle
            // (which was READ at the previous cycle).
            drive_idx_flat <= read_idx_flat;

            // ── BRAM-read advance (LOAD pipeline, Task #99 double-buffer) ──
            // Per-pass strobe sequencing (no inter-pass stall):
            //   - During pass-k strobes (pass_strobe_idx < LCYC-1):
            //       advance read_idx_flat, keep read_active=1.
            //   - At pass_strobe_idx == LCYC-1 (the LAST strobe of pass k):
            //       If pass_idx+1 < M:  IMMEDIATELY queue pass-(k+1)'s
            //           first strobe (advance read_idx_flat by 1, reset
            //           pass_strobe_idx to 0, bump pass_idx). The DPE's
            //           double-buffered substrate accepts new writes
            //           with no inter-pass gap.
            //       Else (last pass):   hold read_idx_flat and let
            //           read_active fall -- the existing S_LOAD branch's
            //           drive_active && !read_active detector then fires
            //           S_HOLD transition.
            //
            // This produces T_steady = LCYC cycles per pass and a clean
            // back-to-back LOAD stream of M*LCYC strobes.
            if (state == S_LOAD) begin
                // Normal LOAD: advance one strobe per cycle.
                if (pass_strobe_idx + 1 < LCYC) begin
                    // Not the last strobe of this pass yet.
                    read_idx_flat   <= read_idx_flat + 1;
                    read_active     <= 1'b1;
                    pass_strobe_idx <= pass_strobe_idx + 1;
                end else begin
                    // pass_strobe_idx == LCYC-1: this IS the last
                    // strobe read of pass-pass_idx. The BRAM is
                    // reading storage[pass_idx*LCYC + LCYC-1] THIS
                    // cycle; drive_active will fire 1 cycle later.
                    if (pass_idx + 1 < M) begin
                        // Double-buffer: queue NEXT pass's first strobe
                        // immediately. read_active stays HIGH; the DPE's
                        // load_phase will toggle (next-pass strobes write
                        // the OTHER substrate).
                        read_idx_flat   <= read_idx_flat + 1;
                        read_active     <= 1'b1;
                        pass_strobe_idx <= 32'd0;
                        pass_idx        <= pass_idx + 1;
                    end else begin
                        // Last pass: do nothing fancy. read_active
                        // falls; drive_active=1 at next cycle drives
                        // the last strobe; the
                        // `drive_active && !read_active` detector
                        // (below) then triggers S_HOLD.
                        read_active <= 1'b0;
                    end
                end
            end

            // ── DRIVE side (one cycle behind READ side) ───────────────
            //
            // If drive_active is set (= read_active was set last cycle),
            // fire w_buf_en and latch BRAM rdata into data_in_v.
            //
            // Back-pressure mechanism (Task #99 double-buffered primitive):
            //
            //   The faithful primitive's input substrate is now double-
            //   buffered (Task #99). Pass-(k+1)'s LOAD writes to the
            //   OTHER substrate while pass-k's COMPUTE reads its own
            //   substrate, so the wrapper does NOT need to stall between
            //   passes -- read_active stays high continuously, producing
            //   a back-to-back stream of M*LCYC strobes.
            //
            //   T_steady = LCYC (no +PRECISION term). dpe_reg_full only
            //   asserts during OUTPUT drain (single-substrate acam_out
            //   on NL, mac_acc on AL); under typical LCYC >> CCYC
            //   configs the OUTPUT drain completes well before pass-
            //   (k+1)'s COMPUTE finishes, so back-pressure is not
            //   expected in steady state. The dpe_reg_full wire is
            //   retained for observability / safety.
            //
            //   See FIDELITY_METHODOLOGY.md and FC_GEMM_WALKTHROUGH.md
            //   for cycle-math details.
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
                        // Initiate the LOAD pipeline.
                        //
                        // At THIS posedge:
                        //   - We NBA read_idx_flat = 0, read_active = 1
                        //     so the BRAM starts reading storage[0] on
                        //     the NEXT posedge (cycle T+1).
                        //   - We NBA drive_active = 0 (no strobe driven
                        //     yet).
                        //
                        // At T+1: BRAM internally reads storage[0],
                        //   NBAs rdata_reg <= storage[0]. After T+1:
                        //   rdata_reg = storage[0]. drive_active was
                        //   NBA'd from read_active (which was 1 at T),
                        //   so drive_active=1 at T+1. read_active at T+1
                        //   advances to next strobe (= 1).
                        //
                        // At T+2: FSM sees drive_active=1 (current).
                        //   NBAs: w_buf_en <= 1, data_in_v <= rdata_reg
                        //   (= storage[0]). After T+2: w_buf_en=1.
                        //
                        // At T+3: DPE samples w_buf_en=1 first time.
                        //
                        // First DPE-observed strobe: cycle T+3.
                        read_idx_flat   <= 32'd0;
                        read_active     <= 1'b1;
                        drive_active    <= 1'b0;
                        pass_strobe_idx <= 32'd0;
                        pass_idx        <= 32'd0;
                        nl_dpe_control  <= 2'b11;
                        t_first_load    <= cycle_count + 3;
                        state           <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    nl_dpe_control <= 2'b11;
                    // Detect end-of-LOAD (Task #99 double-buffer): for
                    // pass-(M-1) (the LAST pass), the LOAD-pipeline's
                    // last cycle has drive_active=1, read_active=0,
                    // pass_idx == M-1 -- that's our cue to S_HOLD.
                    // For inner passes (pass_idx < M-1), read_active is
                    // continuously high (the next pass's strobes follow
                    // immediately), so this condition doesn't fire.
                    if (drive_active && !read_active &&
                        (pass_idx == M - 1)) begin
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
                    // +1 cycle for CLB tree fold / activation LUT.
                    //
                    // Phase 2 NOTE: the ReLU bit-mask is NOT applied
                    // here (the BRAM banks are inside a generate block;
                    // iverilog doesn't allow runtime-indexed access to
                    // genvar-scoped storage). For the smoke test pattern
                    // (all-ones × all-ones identity weights), output
                    // bytes are positive (K & 0xFF for K <= 127, or
                    // wrap-around K & 0xFF for K = 128, 256, etc.) so
                    // ReLU == identity in our test pattern. The +1
                    // cycle still fires for the CLB-stage cost.
                    //
                    // (Real-silicon ReLU would be a single-cycle CLB
                    // LUT inline with the BRAM-write tap. Phase 2 wraps
                    // it into the +1 ACT_NEEDED cycle; for non-
                    // identity pattern testing we'd need to widen the
                    // tree pipeline to optionally mask negative bytes
                    // before BRAM commit.)
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
