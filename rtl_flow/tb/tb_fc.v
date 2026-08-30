// tb_fc.v -- FC / GEMM top-level smoke TB for fc_top.v (Stages 1A + 1B + 1C).
//
// Phase 2 (Task #88, 2026-05-15): port-based input/output BRAM access.
//   - Inputs are written via the fc_top input BRAM port (in_bram_wen,
//     in_bram_waddr, in_bram_wdata). Each write delivers one BUF-bit word
//     (= EPS bytes packed). TB iterates (gv, m, strobe) and packs the EPS
//     bytes into wdata.
//   - Outputs are read via the fc_top output BRAM port (out_bram_raddr,
//     out_bram_rdata). 2-cycle read latency (BRAM read + H-way mux);
//     TB applies 2 wait cycles per read.
//   - Weights remain hierarchical-forced into the DPE primitive's
//     weights[r][c] array (DPE is a VTR black box; weights are sim-only).
//
//   The functional check verifies the same test pattern as Phase 1
//   (W_global=all-ones, X_global=all-ones → output K & 0xFF, ReLU'd if
//   DO_RELU). The cycle expectation now uses the Phase 2 formula
//   measured against the synthesizable wrapper.
//
// Architecture switch via compile-time `+define+`:
//   ARCH_NLDPE -> compiles dpe_nldpe.v + fc_top with FC_TOP_ARCH_NLDPE
//   ARCH_AL    -> compiles dpe_azurelily.v + fc_top without it
// (only one DPE backend may be linked into a given iverilog run since
//  both define `module dpe`).
//
// Cycle measurement (Task #98 — unified sim formula, no compensation
// constants in the analytical output):
//
//   total_cycles  = dut.t_done - dut.t_first_load + 1
//
//   SIM_CYCLES (single source of truth — architectural minimum):
//
//     T_fill_sim   = LCYC + CCYC + OCYC
//     T_steady_sim = max(LCYC + PRECISION, CCYC, OCYC)
//     SIM_CYCLES   = T_fill_sim + (M-1) * T_steady_sim
//
//   The synthesizable RTL pays additional real silicon costs that
//   surface as the rtl_obs - sim_exp delta (info-only):
//     - +2 NBA-handoff cost in the primitive (LOAD->COMPUTE + COMPUTE->OUTPUT)
//     - +TREE_PIPE = ⌈log₂(V)⌉ for V > 1 (CLB adder tree pipeline depth)
//     - +1 CLB cycle if (V > 1) || ((act_mode == 1) && !HAS_ACAM)
//     - +N wrapper plumbing (BRAM read/write pipeline, DPE handshake,
//                            sign-extend latch, BRAM internal write commit,
//                            last-strobe drive-cycle)
//
//   delta = total_cycles - SIM_CYCLES   (varies by V, H, activation, arch)
//
// Pass criterion (per Task #97, retained Task #98): FUNCTIONAL only.
// Cycle delta is reported but does NOT gate PASS/FAIL.

`timescale 1ns / 1ps

`ifndef ARCH_NLDPE
  `ifndef ARCH_AL
    `define ARCH_NLDPE
  `endif
`endif

`ifdef ARCH_NLDPE
  `define FC_TOP_ARCH_NLDPE
  `define ARCH_NAME "NL-DPE"
  `define HAS_ACAM_VAL 1
  `ifndef R_TB
    `define R_TB 256
  `endif
  `ifndef C_TB
    `define C_TB 256
  `endif
  `ifndef BUF_TB
    `define BUF_TB 40
  `endif
`endif
`ifdef ARCH_AL
  `define ARCH_NAME "AzureLily"
  `define HAS_ACAM_VAL 0
  `ifndef R_TB
    `define R_TB 512
  `endif
  `ifndef C_TB
    `define C_TB 128
  `endif
  `ifndef BUF_TB
    `define BUF_TB 16
  `endif
`endif

`ifndef M_TB
  `define M_TB 1
`endif
`ifndef K_TB
  `define K_TB 128
`endif
`ifndef N_TB
  `define N_TB 128
`endif
`ifndef PRECISION_TB
  `define PRECISION_TB 8
`endif
`ifndef PIPELINE_DEPTH_TB
  `define PIPELINE_DEPTH_TB 3
`endif
`ifndef ACAM_CYCLES_TB
  `define ACAM_CYCLES_TB 0
`endif
`ifndef ACTIVATION_TB
  `define ACTIVATION_TB 0
`endif

module tb_fc;
    // Clock / reset
    reg clk;
    reg reset;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ── Locals ─────────────────────────────────────────────────────────
    localparam M               = `M_TB;
    localparam K               = `K_TB;
    localparam N               = `N_TB;
    localparam R               = `R_TB;
    localparam C               = `C_TB;
    localparam BUF             = `BUF_TB;
    localparam PRECISION       = `PRECISION_TB;
    localparam PIPELINE_DEPTH  = `PIPELINE_DEPTH_TB;
    localparam ACAM_CYCLES     = `ACAM_CYCLES_TB;
    localparam ACTIVATION_MODE = `ACTIVATION_TB;
    localparam HAS_ACAM        = `HAS_ACAM_VAL;

    localparam V               = (K + R - 1) / R;
    localparam H               = (N + C - 1) / C;
    localparam EPS             = BUF / 8;
    localparam LCYC            = (R + EPS - 1) / EPS;
    localparam OCYC            = (C + EPS - 1) / EPS;
    localparam CCYC            = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES;

    // ⌈log₂(V)⌉ for V > 1, else 0
    function integer clog2_fn_tb;
        input integer val;
        integer i;
        begin
            clog2_fn_tb = 0;
            for (i = val - 1; i > 0; i = i >> 1) clog2_fn_tb = clog2_fn_tb + 1;
        end
    endfunction
    localparam TREE_PIPE = (V > 1) ? clog2_fn_tb(V) : 0;

    // SIM cycle formula (Task #98 unified — single source of truth,
    // architectural minimum, no compensation constants):
    //   T_fill_sim   = LCYC + CCYC + OCYC
    //   T_steady_sim = max(LCYC + PRECISION, CCYC, OCYC)
    //   SIM_CYCLES   = T_fill_sim + (M-1) * T_steady_sim
    //
    // The RTL pays additional real silicon costs reported as deltas:
    //   - +2 NBA-handoff (LOAD->COMPUTE + COMPUTE->OUTPUT) in the primitive
    //   - +TREE_PIPE = ⌈log₂(V)⌉ for V > 1 (CLB adder tree depth)
    //   - +1 CLB cycle if (V>1) || ((act_mode==1) && !HAS_ACAM)
    //   - +wrapper plumbing (BRAM/handshake/latch/tap/commit/drive)
    // These are NO LONGER absorbed into SIM_CYCLES; rtl_obs - SIM_CYCLES
    // surfaces as info-only cycle delta.
    localparam T_FILL_SIM = LCYC + CCYC + OCYC;

    // ACT_NEEDED gate (info-only; reported as delta, not added to sim):
    //   ACT_NEEDED = (V>1) || ((ACTIVATION_MODE==1) && (HAS_ACAM==0))
    localparam ACT_NEEDED = ((V > 1) ||
                             ((ACTIVATION_MODE == 1) && (HAS_ACAM == 0))) ? 1 : 0;
    localparam DO_RELU = (ACTIVATION_MODE == 1) ? 1 : 0;

    // T_steady (Option A1) — both SIM and RTL pay this cadence.
    localparam T_STEADY_SIM_LC = ((LCYC + PRECISION) > CCYC) ?
                                 (LCYC + PRECISION) : CCYC;
    localparam T_STEADY_SIM    = (T_STEADY_SIM_LC > OCYC) ?
                                  T_STEADY_SIM_LC : OCYC;

    localparam SIM_CYCLES = T_FILL_SIM + (M - 1) * T_STEADY_SIM;

    localparam IN_BANK_DEPTH  = M * LCYC;
    localparam OUT_BANK_DEPTH = M * OCYC;

    // Expected output byte (same test pattern as Phase 1).
    function [7:0] expected_byte_fn(input integer kval, input integer relu_int);
        reg [7:0] raw;
        begin
            raw = kval & 8'hFF;
            if (relu_int && raw[7])
                expected_byte_fn = 8'h00;
            else
                expected_byte_fn = raw;
        end
    endfunction

    localparam APPLY_RELU = (ACT_NEEDED && DO_RELU) ? 1 : 0;
    localparam [7:0] EXPECTED_BYTE = expected_byte_fn(K, APPLY_RELU);

    // ── DUT ─────────────────────────────────────────────────────────────
    reg                 start;
    wire                done;
    reg                 in_bram_wen;
    reg  [31:0]         in_bram_waddr;
    reg  [BUF-1:0]      in_bram_wdata;
    reg  [31:0]         out_bram_raddr;
    wire [BUF-1:0]      out_bram_rdata;

    fc_top #(
        .M(M),
        .K(K),
        .N(N),
        .R(R),
        .C(C),
        .BUF(BUF),
        .PRECISION(PRECISION),
        .PIPELINE_DEPTH(PIPELINE_DEPTH),
        .ACAM_CYCLES(ACAM_CYCLES),
        .ACTIVATION_MODE(ACTIVATION_MODE),
        .HAS_ACAM(HAS_ACAM)
    ) dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .done(done),
        .in_bram_wen(in_bram_wen),
        .in_bram_waddr(in_bram_waddr),
        .in_bram_wdata(in_bram_wdata),
        .out_bram_raddr(out_bram_raddr),
        .out_bram_rdata(out_bram_rdata)
    );

    // ── Per-tile weight forcing (generate block) ───────────────────────
    // Hierarchical paths to dut.gen_v[gv].gen_h[gh].gen_active.dpe_inst.weights
    // require gv/gh to be elaboration-time constants. We unroll a nested
    // generate that, when triggered, sets every per-tile weight per the
    // global pattern: W_global[r][n] = 1 iff (r < K) AND (n < N).
    reg do_force_weights;
    initial do_force_weights = 1'b0;

    genvar fwv, fwh;
    generate
        for (fwv = 0; fwv < V; fwv = fwv + 1) begin : g_force_v
            for (fwh = 0; fwh < H; fwh = fwh + 1) begin : g_force_h
                integer fr, fc, gr, gn;
                always @(posedge do_force_weights) begin
                    for (fr = 0; fr < R; fr = fr + 1) begin
                        for (fc = 0; fc < C; fc = fc + 1) begin
                            gr = fwv * R + fr;
                            gn = fwh * C + fc;
                            if (gr < K && gn < N)
                                dut.gen_v[fwv].gen_h[fwh].gen_active.dpe_inst.weights[fr][fc] = 8'sh01;
                            else
                                dut.gen_v[fwv].gen_h[fwh].gen_active.dpe_inst.weights[fr][fc] = 8'sh00;
                        end
                    end
                end
            end
        end
    endgenerate

    // Helper to pack EPS bytes of all-ones (limited by K and the
    // gv-local R offset) into a BUF-bit word.
    function [BUF-1:0] input_word_pack;
        input integer gv;
        input integer m;
        input integer strobe;
        integer p_b, p_k, p_kg;
        reg [BUF-1:0] w;
        begin
            w = {BUF{1'b0}};
            for (p_b = 0; p_b < EPS; p_b = p_b + 1) begin
                p_k  = strobe * EPS + p_b;
                p_kg = gv * R + p_k;
                if (p_k < R && p_kg < K)
                    w[p_b*8 +: 8] = 8'sh01;
                else
                    w[p_b*8 +: 8] = 8'sh00;
            end
            input_word_pack = w;
        end
    endfunction

    // ── Test sequence ──────────────────────────────────────────────────
    integer mi, ki, ni, ri, ci, gvi, ghi, si;
    integer error_count;
    integer total_cycles;
    // (fidelity_num/den removed in Task #97; now report only signed delta.)
    reg [7:0] obs_byte;

    initial begin
        $display("[tb_fc] arch=%0s M=%0d K=%0d N=%0d R=%0d C=%0d BUF=%0d V=%0d H=%0d",
                 `ARCH_NAME, M, K, N, R, C, BUF, V, H);
        $display("[tb_fc]   EPS=%0d LCYC=%0d CCYC=%0d (P=%0d + (D=%0d - 1) + A=%0d) OCYC=%0d TREE_PIPE=%0d (info-only)",
                 EPS, LCYC, CCYC, PRECISION, PIPELINE_DEPTH, ACAM_CYCLES, OCYC, TREE_PIPE);
        $display("[tb_fc]   T_FILL_SIM=%0d (= LCYC + CCYC + OCYC) T_STEADY_SIM=%0d (= max(LCYC+P, CCYC, OCYC))",
                 T_FILL_SIM, T_STEADY_SIM);
        $display("[tb_fc]   ACTIVATION_MODE=%0d HAS_ACAM=%0d ACT_NEEDED=%0d (info-only, reported as delta)",
                 ACTIVATION_MODE, HAS_ACAM, ACT_NEEDED);
        $display("[tb_fc]   SIM_CYCLES=%0d (Task #98 unified, architectural minimum)", SIM_CYCLES);
        $display("[tb_fc]   EXPECTED_BYTE=0x%02h", EXPECTED_BYTE);

        reset          = 1'b1;
        start          = 1'b0;
        in_bram_wen    = 1'b0;
        in_bram_waddr  = 32'd0;
        in_bram_wdata  = {BUF{1'b0}};
        out_bram_raddr = 32'd0;
        error_count    = 0;

        repeat (3) @(posedge clk); #1;
        reset = 1'b0;
        @(posedge clk); #1;

        // ── Drive input BRAM via the port (flat address per gv) ──
        // For each gv in [0, V), m in [0, M), strobe in [0, LCYC), write
        // one BUF-bit packed word at flat address
        //   gv * (M*LCYC) + m * LCYC + strobe.
        for (gvi = 0; gvi < V; gvi = gvi + 1) begin
            for (mi = 0; mi < M; mi = mi + 1) begin
                for (si = 0; si < LCYC; si = si + 1) begin
                    in_bram_wen   = 1'b1;
                    in_bram_waddr = gvi * (M * LCYC) + mi * LCYC + si;
                    in_bram_wdata = input_word_pack(gvi, mi, si);
                    @(posedge clk); #1;
                end
            end
        end
        in_bram_wen   = 1'b0;
        in_bram_wdata = {BUF{1'b0}};

        // ── Hierarchical-force weights (every (gv, gh) tile) ──
        do_force_weights = 1'b1;
        #1;
        do_force_weights = 1'b0;

        // Settle one cycle so forces are stable before start.
        @(posedge clk); #1;

        // ── Pulse start ──
        start = 1'b1;
        @(posedge clk); #1;
        start = 1'b0;

        // ── Wait for done ──
        begin : wait_done
            integer guard;
            guard = 0;
            // Generous guard: the worst delta we expect is wrapper(+6) + TREE
            // + CLB ≈ +10, so 4× SIM + 400 is plenty.
            while (done !== 1'b1 && guard < SIM_CYCLES * 4 + 400) begin
                @(posedge clk); #1;
                guard = guard + 1;
            end
            if (done !== 1'b1) begin
                $display("[tb_fc] FUNCTIONAL_FAIL: done never asserted within %0d cycles", guard);
                error_count = error_count + 1;
            end
        end

        // ── Read latency markers ──
        total_cycles = dut.t_done - dut.t_first_load + 1;

        // ── Functional check ──
        // For each (m, n), compute (gh, col_local) and read from the
        // appropriate output bank. Read latency = 2 cycles (BRAM read +
        // H-way mux register).
        for (mi = 0; mi < M; mi = mi + 1) begin
            for (ni = 0; ni < N; ni = ni + 1) begin : check_n_loop
                integer gh_sel;
                integer col_local;
                integer strobe_n;
                integer b_n;
                reg [BUF-1:0] word;
                gh_sel    = ni / C;
                col_local = ni - gh_sel * C;
                strobe_n  = col_local / EPS;
                b_n       = col_local - strobe_n * EPS;
                out_bram_raddr = gh_sel * (M * OCYC) + mi * OCYC + strobe_n;
                @(posedge clk); #1;
                @(posedge clk); #1;
                word     = out_bram_rdata;
                obs_byte = word[b_n*8 +: 8];
                if (obs_byte !== EXPECTED_BYTE) begin
                    if (error_count < 10) begin
                        $display("[tb_fc] MISMATCH m=%0d n=%0d expected=0x%02h got=0x%02h",
                                 mi, ni, EXPECTED_BYTE, obs_byte);
                    end
                    error_count = error_count + 1;
                end
            end
        end

        // ── Cycle reporting (informational; no cycle gate per Task #97/#98) ──
        $display("[tb_fc] t_first_load=%0d  t_done=%0d  total_cycles=%0d  SIM_CYCLES=%0d  delta=%0d",
                 dut.t_first_load, dut.t_done, total_cycles,
                 SIM_CYCLES,
                 total_cycles - SIM_CYCLES);

        // ── Verdict: FUNCTIONAL only (cycle delta is informational) ──
        if (error_count == 0) begin
            $display("[tb_fc] FUNCTIONAL_PASS (%0s): %0d/%0d output bytes match, cycles=%0d  sim=%0d  delta=%0d",
                     `ARCH_NAME, M*N, M*N, total_cycles, SIM_CYCLES,
                     total_cycles - SIM_CYCLES);
        end else begin
            $display("[tb_fc] FUNCTIONAL_FAIL (%0s): %0d byte mismatches; cycles=%0d sim=%0d delta=%0d",
                     `ARCH_NAME, error_count, total_cycles, SIM_CYCLES,
                     total_cycles - SIM_CYCLES);
        end
        $finish;
    end

endmodule
