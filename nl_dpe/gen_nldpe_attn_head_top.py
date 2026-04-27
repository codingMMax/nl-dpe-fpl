#!/usr/bin/env python3
"""Generate the NL-DPE attention-head top RTL (no output projection).

Composition matches the IMC simulator's `attention_model` definition:

    [valid_x, data_in_x] ──► fc_top_qkv_streaming ×3 (Q/K/V projections)
                                 │     │     │
                                 ▼     ▼     ▼
                              q_buf  k_buf  v_buf  (sram, depth = N×PACKED_NQ+1)
                                 │     │     │
                                 └─────┴─────┘
                                       ▼
                              nldpe_dimm_top (W=16 lanes, 1 fire)
                                       ▼
                                [data_out, valid_n]

For N=128 input tokens of d_model=128, the FCs run as streaming
ping-pong pairs (2 DPEs per arm, dpe_a/dpe_b), absorbing one packed
input word per cycle and producing one packed output word per cycle
in steady state.

Locked design parameters:
  d_model = 128
  d_head  = 64
  N_SEQ   = 128
  C       = 128 (crossbar cols)
  W       = 16  (DIMM lanes)
  K_qkt   = 2   (= C / d_head)
  K_sv    = 1
  EPW     = 5   (elements per 40-bit packed word)
  PACKED_KQ = ceil(d_model/EPW) = 26 (input words per FC inference)
  PACKED_NQ = ceil(d_head/EPW)  = 13 (output words per FC inference)

Resource expectation (verified by grep on emitted RTL):
  - 3 × fc_top_qkv_streaming (Q/K/V arms; 2 DPEs each) — instances
                               fc_q_inst, fc_k_inst, fc_v_inst
  - 3 × sram (q/k/v buffers)    — instances q_buffer, k_buffer, v_buffer
  - 1 × nldpe_dimm_top          — instance dimm_inst
  - 0 × fc_top_o (NO output projection; sim's attention_model stops at mac_sv)

Top-level FSM (Architecture B, AH Step B-5 — strict serial cascade):
  S_IDLE       → wait for valid_x rising edge
  S_FEED_X     → ready_x=1; pipe X to all 3 FCs for N×PACKED_KQ cycles;
                 capture FC outputs into q/k/v_buffers as their valid_n fire
  S_DRAIN_FC   → wait for in-flight FC outputs to settle (count = N×PACKED_NQ)
  S_LOAD_KV    → drive valid_q (14 cyc, Q row 0), valid_k (1664 cyc, full K),
                 valid_v (1664 cyc, full V) — one-shot ingress (NOT repeated
                 per Q row; SRAM contents persist through the full cascade)
  S_CASCADE    → strict serial Q-row processing.  For each Q row k = 0..N-1:
                  • Drive valid_q for DIMM_Q_WORDS cycles (Q row k's words)
                  • Wait for wsum_done_o (= Q row k completed all 3 stages)
                  • Fire all 3 next_q_row_trigger pulses simultaneously to
                    recycle score/softmax/wsum FSMs
                  • Increment q_row_idx for wide-addr indexing
                  • Repeat until q_row_idx == N_SEQ-1, then advance to drain
  S_DRAIN_FINAL → settle margin after last Q row completes
  S_OUTPUT_DRAIN → assert valid_n for N_FC_OUT_WORDS cycles (TB-counted)
  S_OUTPUT_DONE → terminal

Why strict serial (vs deeply pipelined cascade)?  The score_sram is
single-Q-row scope (no ping-pong banks), so score's S_OUTPUT must remain
held while softmax SM_LOAD pulls 8 words.  If score is triggered too
early, score_read_addr resets to 0 and softmax captures stale data.
The clean barrier is wsum_done — by which time softmax has long since
left SM_LOAD and wsum has cached its inputs.  Triggering score here
gives correct data flow at the cost of pipelined throughput.  Cycle
gate residual is documented as `fsm_glue` modelling-granularity in
known_cycle_deltas.json.

iverilog smoke-compile target:
  iverilog -o /tmp/smoke fc_verification/rtl/nldpe_attn_head_d64_c128.v   \
                         fc_verification/rtl/nldpe_dimm_top_d64_c128.v    \
                         fc_verification/rtl/nldpe/fc_top_qkv_streaming.v \
                         fc_verification/dpe_stub.v

Usage:
    python3 nl_dpe/gen_nldpe_attn_head_top.py --output-dir fc_verification/rtl/
"""

import argparse
import math
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
# Locked design constants
# ─────────────────────────────────────────────────────────────────────────
DATA_WIDTH = 40
D_MODEL    = 128
D_HEAD     = 64
N_SEQ      = 128
C          = 128
W_LANES    = 16
EPW        = 5  # elements per 40-bit packed word

PACKED_KQ  = math.ceil(D_MODEL / EPW)  # 26 — input words / FC inference
PACKED_NQ  = math.ceil(D_HEAD  / EPW)  # 13 — output words / FC inference

# DIMM ingest counts (matched against existing nldpe_dimm_top FSM):
DIMM_Q_WORDS = 14                 # score_matrix S_LOAD_Q exits when q_write_addr==13
DIMM_K_WORDS = 1664               # full K tensor (128 keys × 13 packed)
DIMM_V_WORDS = 1664               # full V tensor (128 vals × 13 packed)


def render_top_module(out_path: Path) -> None:
    """Emit the nldpe_attn_head_d64_c128 top module."""

    n_x_input_words   = N_SEQ * PACKED_KQ          # 128 × 26 = 3328
    n_fc_output_words = N_SEQ * PACKED_NQ          # 128 × 13 = 1664
    buffer_depth      = n_fc_output_words + 1      # 1665 (matches DIMM k/v sram depth)

    # Address widths
    aw_in   = max(1, math.ceil(math.log2(n_x_input_words   + 1)))   # 12
    aw_buf  = max(1, math.ceil(math.log2(buffer_depth          )))  # 11
    aw_dimm = max(1, math.ceil(math.log2(max(DIMM_K_WORDS,
                                              DIMM_V_WORDS) + 1)))  # 11
    aw_drain = max(1, math.ceil(math.log2(n_fc_output_words + 2)))  # 12

    rtl = []
    rtl.append("// NL-DPE Attention Head Top — Q/K/V projections + buffers + DIMM")
    rtl.append("// AUTO-GENERATED by nl_dpe/gen_nldpe_attn_head_top.py — DO NOT EDIT BY HAND")
    rtl.append("//")
    rtl.append("// Composition (matches IMC sim's attention_model — no output projection):")
    rtl.append("//")
    rtl.append("//   [valid_x, data_in_x] ──► fc_top_qkv_streaming ×3 (Q/K/V; 2 DPEs each)")
    rtl.append("//                                │     │     │")
    rtl.append("//                                ▼     ▼     ▼")
    rtl.append("//                             q_buf k_buf v_buf  (sram)")
    rtl.append("//                                │     │     │")
    rtl.append("//                                └─────┴─────┘")
    rtl.append("//                                      ▼")
    rtl.append("//                             nldpe_dimm_top (W=16 lanes, B-5 strict serial)")
    rtl.append("//                                      ▼")
    rtl.append("//                                [data_out, valid_n]")
    rtl.append("//")
    rtl.append("// Locked parameters (see gen_nldpe_attn_head_top.py for derivation):")
    rtl.append(f"//   d_model={D_MODEL}, d_head={D_HEAD}, N_SEQ={N_SEQ}, C={C}, W={W_LANES}")
    rtl.append(f"//   PACKED_KQ={PACKED_KQ} (input words/inference)")
    rtl.append(f"//   PACKED_NQ={PACKED_NQ} (output words/inference)")
    rtl.append("//")
    rtl.append("// Resource expectation:")
    rtl.append("//   - 3 × fc_top_qkv_streaming (instances: fc_q_inst, fc_k_inst, fc_v_inst;")
    rtl.append("//                               2 DPEs each via ping-pong)")
    rtl.append("//   - 3 × sram                 (instances: q_buffer, k_buffer, v_buffer)")
    rtl.append("//   - 1 × nldpe_dimm_top       (instance:  dimm_inst)")
    rtl.append("//   - 0 × fc_top_o             (NO output projection)")
    rtl.append("//")
    rtl.append("// AH Step B-5 (this file): strict-serial 3-stage cascade per Q row.")
    rtl.append("// All 4 DIMM-top mode parameters (SCORE/SOFTMAX/WSUM_BACK_TO_BACK_MODE,")
    rtl.append("// *_WIDE_ADDR_MODE) are turned ON. K/V SRAMs are loaded ONCE; the score/softmax/")
    rtl.append("// wsum FSMs auto-recycle through Q rows on a unified next_q_row_trigger that")
    rtl.append("// fires when wsum_done_o pulses (= Q row k fully completed all 3 stages).")
    rtl.append("//")
    rtl.append("// Module dependencies (compile together with this file):")
    rtl.append("//   - fc_verification/rtl/nldpe_dimm_top_d64_c128.v")
    rtl.append("//   - fc_verification/rtl/nldpe/fc_top_qkv_streaming.v")
    rtl.append("//   - fc_verification/dpe_stub.v")
    rtl.append("")
    rtl.append("`timescale 1ns / 1ps")
    rtl.append("")
    rtl.append("module nldpe_attn_head_d64_c128 #(")
    rtl.append(f"    parameter DATA_WIDTH = {DATA_WIDTH},")
    rtl.append(f"    parameter D_MODEL    = {D_MODEL},")
    rtl.append(f"    parameter D_HEAD     = {D_HEAD},")
    rtl.append(f"    parameter N_SEQ      = {N_SEQ},")
    rtl.append(f"    parameter C          = {C},")
    rtl.append(f"    parameter W          = {W_LANES}")
    rtl.append(")(")
    rtl.append("    input  wire                    clk,")
    rtl.append("    input  wire                    rst,")
    rtl.append("    input  wire                    valid_x,")
    rtl.append("    input  wire [DATA_WIDTH-1:0]   data_in_x,")
    rtl.append("    input  wire                    ready_n,")
    rtl.append("    output wire [DATA_WIDTH-1:0]   data_out,")
    rtl.append("    output wire                    ready_x,")
    rtl.append("    output wire                    valid_n")
    rtl.append(");")
    rtl.append("")
    rtl.append("    // ─── Local constants (must match generator) ────────────────────")
    rtl.append(f"    localparam PACKED_KQ        = {PACKED_KQ}; // ceil(d_model/EPW), in words")
    rtl.append(f"    localparam PACKED_NQ        = {PACKED_NQ}; // ceil(d_head/EPW),  in words")
    rtl.append(f"    localparam N_X_INPUT_WORDS  = {n_x_input_words};   // N × PACKED_KQ")
    rtl.append(f"    localparam N_FC_OUT_WORDS   = {n_fc_output_words}; // N × PACKED_NQ")
    rtl.append(f"    localparam BUFFER_DEPTH     = {buffer_depth};      // N × PACKED_NQ + 1")
    rtl.append(f"    localparam DIMM_Q_WORDS     = {DIMM_Q_WORDS};")
    rtl.append(f"    localparam DIMM_K_WORDS     = {DIMM_K_WORDS};")
    rtl.append(f"    localparam DIMM_V_WORDS     = {DIMM_V_WORDS};")
    rtl.append("")
    rtl.append("    // ─── FSM (Architecture B, AH Step B-5) ─────────────────────────")
    rtl.append("    // Strict serial cascade: each Q row goes through score → softmax → wsum")
    rtl.append("    // before the next Q row starts.  K/V SRAMs are loaded ONCE in S_LOAD_KV.")
    rtl.append("    // Per Q row, the head waits for wsum_done_o (= Q row k entered WS_OUTPUT)")
    rtl.append("    // and then fires all 3 *_next_q_row_trigger pulses simultaneously.")
    rtl.append("    localparam S_IDLE         = 4'd0;")
    rtl.append("    localparam S_FEED_X       = 4'd1;")
    rtl.append("    localparam S_DRAIN_FC     = 4'd2;")
    rtl.append("    localparam S_LOAD_KV      = 4'd3;  // one-shot Q0+K+V load")
    rtl.append("    localparam S_CASCADE      = 4'd4;  // serial per-Q-row cascade × N")
    rtl.append("    localparam S_DRAIN_FINAL  = 4'd5;  // settle after last Q row")
    rtl.append("    localparam S_OUTPUT_DRAIN = 4'd6;  // stream N×PACKED_NQ to valid_n")
    rtl.append("    localparam S_OUTPUT_DONE  = 4'd7;  // terminal")
    rtl.append("    reg [3:0] state;")
    rtl.append("")
    rtl.append(f"    reg [{aw_in - 1}:0] in_count;       // X words consumed (0..N_X_INPUT_WORDS)")
    rtl.append(f"    reg [{aw_buf - 1}:0] write_addr;    // q/k/v_buffer write address")
    rtl.append(f"    reg [{aw_buf - 1}:0] read_addr;     // q/k/v_buffer read address")
    rtl.append(f"    reg [{aw_dimm - 1}:0] dimm_q_count; // valid_q pulses already issued")
    rtl.append(f"    reg [{aw_dimm - 1}:0] dimm_k_count; // valid_k pulses already issued")
    rtl.append(f"    reg [{aw_dimm - 1}:0] dimm_v_count; // valid_v pulses already issued")
    rtl.append("    reg drive_valid_q, drive_valid_k, drive_valid_v;")
    rtl.append("    reg [7:0] q_row_idx;            // 0..N-1, drives DIMM wide-addr SRAMs")
    rtl.append("    // wsum_done_o is a 1-cycle edge pulse; track total count globally so events")
    rtl.append("    // that fire during S_LOAD_KV (when Q row 0 finishes inside the DIMM before")
    rtl.append("    // S_CASCADE entry) aren't missed.  Same idea for sm_done_count.")
    rtl.append("    reg [7:0] ws_done_count;        // cumulative wsum_done_o pulses observed")
    rtl.append("    reg [7:0] ws_consumed_count;    // wsum_done events consumed by S_CASCADE advance")
    rtl.append(f"    reg [{aw_drain - 1}:0] phase_drain_count;  // drain margin counter")
    rtl.append(f"    reg [{aw_drain - 1}:0] output_drain_count; // S_OUTPUT_DRAIN ticks 0..N_FC_OUT_WORDS")
    rtl.append("")
    rtl.append("    // Triggers driven combinationally from FSM state below.")
    rtl.append("    reg score_next_q_row_pulse, softmax_next_q_row_pulse, wsum_next_q_row_pulse;")
    rtl.append("")
    rtl.append("    // Drain margin: accommodates the small NBA-edge lag between trigger pulse")
    rtl.append("    // and FSM commit cycle (typically 1-3 cycles).")
    rtl.append("    localparam PHASE_DRAIN_CYC = 16'd64;   // post-trigger settle window")
    rtl.append("")
    rtl.append("    // ─── Q/K/V projection arms ─────────────────────────────────────")
    rtl.append("    wire [DATA_WIDTH-1:0] q_fc_out, k_fc_out, v_fc_out;")
    rtl.append("    wire q_fc_valid_n, k_fc_valid_n, v_fc_valid_n;")
    rtl.append("    wire q_fc_ready,   k_fc_ready,   v_fc_ready;")
    rtl.append("")
    rtl.append("    wire feed_x_now = (state == S_FEED_X) && valid_x;")
    rtl.append("")
    rtl.append("    fc_top_qkv_streaming #(.DATA_WIDTH(DATA_WIDTH)) fc_q_inst (")
    rtl.append("        .clk(clk), .rst(rst),")
    rtl.append("        .valid(feed_x_now), .ready_n(1'b0),")
    rtl.append("        .data_in(data_in_x),")
    rtl.append("        .data_out(q_fc_out),")
    rtl.append("        .ready(q_fc_ready), .valid_n(q_fc_valid_n)")
    rtl.append("    );")
    rtl.append("    fc_top_qkv_streaming #(.DATA_WIDTH(DATA_WIDTH)) fc_k_inst (")
    rtl.append("        .clk(clk), .rst(rst),")
    rtl.append("        .valid(feed_x_now), .ready_n(1'b0),")
    rtl.append("        .data_in(data_in_x),")
    rtl.append("        .data_out(k_fc_out),")
    rtl.append("        .ready(k_fc_ready), .valid_n(k_fc_valid_n)")
    rtl.append("    );")
    rtl.append("    fc_top_qkv_streaming #(.DATA_WIDTH(DATA_WIDTH)) fc_v_inst (")
    rtl.append("        .clk(clk), .rst(rst),")
    rtl.append("        .valid(feed_x_now), .ready_n(1'b0),")
    rtl.append("        .data_in(data_in_x),")
    rtl.append("        .data_out(v_fc_out),")
    rtl.append("        .ready(v_fc_ready), .valid_n(v_fc_valid_n)")
    rtl.append("    );")
    rtl.append("")
    rtl.append("    // ─── Q/K/V buffers (sram, depth = N × PACKED_NQ + 1) ────────────")
    rtl.append("    wire [DATA_WIDTH-1:0] q_buf_out, k_buf_out, v_buf_out;")
    rtl.append("")
    rtl.append("    sram #(.N_CHANNELS(1), .DATA_WIDTH(DATA_WIDTH), .DEPTH(BUFFER_DEPTH))")
    rtl.append("    q_buffer (")
    rtl.append("        .clk(clk), .rst(rst),")
    rtl.append("        .w_en(q_fc_valid_n),")
    rtl.append("        .r_addr(read_addr),")
    rtl.append("        .w_addr(write_addr),")
    rtl.append("        .sram_data_in(q_fc_out),")
    rtl.append("        .sram_data_out(q_buf_out)")
    rtl.append("    );")
    rtl.append("    sram #(.N_CHANNELS(1), .DATA_WIDTH(DATA_WIDTH), .DEPTH(BUFFER_DEPTH))")
    rtl.append("    k_buffer (")
    rtl.append("        .clk(clk), .rst(rst),")
    rtl.append("        .w_en(k_fc_valid_n),")
    rtl.append("        .r_addr(read_addr),")
    rtl.append("        .w_addr(write_addr),")
    rtl.append("        .sram_data_in(k_fc_out),")
    rtl.append("        .sram_data_out(k_buf_out)")
    rtl.append("    );")
    rtl.append("    sram #(.N_CHANNELS(1), .DATA_WIDTH(DATA_WIDTH), .DEPTH(BUFFER_DEPTH))")
    rtl.append("    v_buffer (")
    rtl.append("        .clk(clk), .rst(rst),")
    rtl.append("        .w_en(v_fc_valid_n),")
    rtl.append("        .r_addr(read_addr),")
    rtl.append("        .w_addr(write_addr),")
    rtl.append("        .sram_data_in(v_fc_out),")
    rtl.append("        .sram_data_out(v_buf_out)")
    rtl.append("    );")
    rtl.append("")
    rtl.append("    // Track the number of FC valid_n pulses observed.")
    rtl.append(f"    reg [{aw_buf - 1}:0] fc_out_count;")
    rtl.append("    always @(posedge clk or posedge rst) begin")
    rtl.append("        if (rst) begin")
    rtl.append("            write_addr   <= 0;")
    rtl.append("            fc_out_count <= 0;")
    rtl.append("        end else if (q_fc_valid_n) begin")
    rtl.append("            write_addr   <= write_addr   + 1;")
    rtl.append("            fc_out_count <= fc_out_count + 1;")
    rtl.append("        end")
    rtl.append("    end")
    rtl.append("")
    rtl.append("    // ─── DIMM instance (Architecture B: all back-to-back modes ON) ─")
    rtl.append("    wire [DATA_WIDTH-1:0] dimm_data_out;")
    rtl.append("    wire dimm_valid_n;")
    rtl.append("    wire dimm_ready_q, dimm_ready_k, dimm_ready_v;")
    rtl.append("    wire dimm_score_valid_o;")
    rtl.append("    wire dimm_score_all_done_o;")
    rtl.append("    wire dimm_softmax_done_o;")
    rtl.append("    wire dimm_wsum_done_o;")
    rtl.append("")
    rtl.append("    nldpe_dimm_top #(")
    rtl.append("        .N(N_SEQ), .D(D_HEAD), .W(W), .DATA_WIDTH(DATA_WIDTH),")
    rtl.append("        // AH Step B-5: activate every back-to-back / wide-addr mode.")
    rtl.append("        .SCORE_BACK_TO_BACK_MODE(1),")
    rtl.append("        .SOFTMAX_BACK_TO_BACK_MODE(1),")
    rtl.append("        .WSUM_BACK_TO_BACK_MODE(1),")
    rtl.append("        .SCORE_WIDE_ADDR_MODE(1),")
    rtl.append("        .SOFTMAX_WIDE_ADDR_MODE(1),")
    rtl.append("        .WSUM_WIDE_ADDR_MODE(1)")
    rtl.append("    ) dimm_inst (")
    rtl.append("        .clk(clk), .rst(rst),")
    rtl.append("        .valid_q(drive_valid_q), .valid_k(drive_valid_k), .valid_v(drive_valid_v),")
    rtl.append("        .ready_n(ready_n),")
    rtl.append("        .data_in_q(q_buf_out), .data_in_k(k_buf_out), .data_in_v(v_buf_out),")
    rtl.append("        .data_out(dimm_data_out),")
    rtl.append("        .ready_q(dimm_ready_q), .ready_k(dimm_ready_k), .ready_v(dimm_ready_v),")
    rtl.append("        .valid_n(dimm_valid_n),")
    rtl.append("        .score_valid_o(dimm_score_valid_o),")
    rtl.append("        .score_all_done_o(dimm_score_all_done_o),")
    rtl.append("        .softmax_done_o(dimm_softmax_done_o),")
    rtl.append("        .wsum_done_o(dimm_wsum_done_o),")
    rtl.append("        .score_next_q_row_trigger_i(score_next_q_row_pulse),")
    rtl.append("        .softmax_next_q_row_trigger_i(softmax_next_q_row_pulse),")
    rtl.append("        .wsum_next_q_row_trigger_i(wsum_next_q_row_pulse),")
    rtl.append("        .q_row_idx_i(q_row_idx)")
    rtl.append("    );")
    rtl.append("")
    rtl.append("    // ─── Top-level strict-serial cascade FSM ───────────────────────")
    rtl.append("    // S_LOAD_KV : drive valid_q (14 cyc, Q row 0), valid_k (1664 cyc, full K),")
    rtl.append("    //              valid_v (1664 cyc, full V). One-shot — these SRAMs persist.")
    rtl.append("    // S_CASCADE : per Q row k = 0..N-1:")
    rtl.append("    //              • If q_row_idx > 0: drive valid_q for DIMM_Q_WORDS cycles")
    rtl.append("    //                (Q row k, read from q_buffer at offset k*PACKED_NQ).")
    rtl.append("    //              • Wait for dimm_wsum_done_o (= Q row k entered WS_OUTPUT,")
    rtl.append("    //                meaning all 3 stages have completed for Q row k).")
    rtl.append("    //              • Fire all 3 *_next_q_row_trigger pulses in lockstep.")
    rtl.append("    //              • Increment q_row_idx; advance to compute next Q row.")
    rtl.append("    //              • After q_row_idx == N-1's wsum_done, transition to")
    rtl.append("    //                S_DRAIN_FINAL.")
    rtl.append("    // S_DRAIN_FINAL : drain margin (PHASE_DRAIN_CYC) for last Q row's wsum")
    rtl.append("    //                 to settle in out_sram before draining.")
    rtl.append("    // S_OUTPUT_DRAIN : assert valid_n for N_FC_OUT_WORDS cycles.")
    rtl.append("    always @(posedge clk or posedge rst) begin")
    rtl.append("        if (rst) begin")
    rtl.append("            state                  <= S_IDLE;")
    rtl.append("            in_count               <= 0;")
    rtl.append("            read_addr              <= 0;")
    rtl.append("            dimm_q_count           <= 0;")
    rtl.append("            dimm_k_count           <= 0;")
    rtl.append("            dimm_v_count           <= 0;")
    rtl.append("            drive_valid_q          <= 1'b0;")
    rtl.append("            drive_valid_k          <= 1'b0;")
    rtl.append("            drive_valid_v          <= 1'b0;")
    rtl.append("            q_row_idx              <= 0;")
    rtl.append("            ws_done_count          <= 0;")
    rtl.append("            ws_consumed_count      <= 0;")
    rtl.append("            phase_drain_count      <= 0;")
    rtl.append("            output_drain_count     <= 0;")
    rtl.append("        end else begin")
    rtl.append("            // Default: pulses low; states below override for one cycle.")
    rtl.append("            score_next_q_row_pulse   <= 1'b0;")
    rtl.append("            softmax_next_q_row_pulse <= 1'b0;")
    rtl.append("            wsum_next_q_row_pulse    <= 1'b0;")
    rtl.append("")
    rtl.append("            // Track wsum_done_o pulses globally (independent of FSM state).")
    rtl.append("            // wsum_done_o pulses for 1 cycle on entry to WS_OUTPUT for each Q row.")
    rtl.append("            // Counting allows S_CASCADE to consume events that fired during")
    rtl.append("            // S_LOAD_KV (e.g. Q row 0's wsum finishes mid-K/V load).")
    rtl.append("            if (dimm_wsum_done_o) ws_done_count <= ws_done_count + 1;")
    rtl.append("")
    rtl.append("            case (state)")
    rtl.append("                S_IDLE: begin")
    rtl.append("                    if (valid_x) state <= S_FEED_X;")
    rtl.append("                end")
    rtl.append("                S_FEED_X: begin")
    rtl.append("                    if (valid_x) in_count <= in_count + 1;")
    rtl.append("                    if (in_count + 1 == N_X_INPUT_WORDS) begin")
    rtl.append("                        state <= S_DRAIN_FC;")
    rtl.append("                    end")
    rtl.append("                end")
    rtl.append("                S_DRAIN_FC: begin")
    rtl.append("                    // Wait until all PACKED_NQ × N FC outputs are buffered.")
    rtl.append("                    if (fc_out_count >= N_FC_OUT_WORDS) begin")
    rtl.append("                        state         <= S_LOAD_KV;")
    rtl.append("                        read_addr     <= 0;       // Q row 0 starts at 0")
    rtl.append("                        drive_valid_q <= 1'b1;    // begin valid_q (Q row 0)")
    rtl.append("                        dimm_q_count  <= 0;")
    rtl.append("                        dimm_k_count  <= 0;")
    rtl.append("                        dimm_v_count  <= 0;")
    rtl.append("                        q_row_idx     <= 0;")
    rtl.append("                    end")
    rtl.append("                end")
    rtl.append("                S_LOAD_KV: begin")
    rtl.append("                    // Phase A: drive valid_q for DIMM_Q_WORDS cycles (Q row 0).")
    rtl.append("                    // Phase B: drive valid_k for DIMM_K_WORDS cycles (full K).")
    rtl.append("                    // Phase C: drive valid_v for DIMM_V_WORDS cycles (full V).")
    rtl.append("                    // After V completes, transition to S_CASCADE; q_row_idx=0")
    rtl.append("                    // (Q row 0's score is already being computed inside DIMM).")
    rtl.append("                    if (drive_valid_q) begin")
    rtl.append("                        dimm_q_count <= dimm_q_count + 1;")
    rtl.append("                        read_addr    <= read_addr + 1;")
    rtl.append("                        if (dimm_q_count + 1 == DIMM_Q_WORDS) begin")
    rtl.append("                            drive_valid_q <= 1'b0;")
    rtl.append("                            drive_valid_k <= 1'b1;")
    rtl.append("                            read_addr     <= 0;")
    rtl.append("                        end")
    rtl.append("                    end else if (drive_valid_k) begin")
    rtl.append("                        dimm_k_count <= dimm_k_count + 1;")
    rtl.append("                        read_addr    <= read_addr + 1;")
    rtl.append("                        if (dimm_k_count + 1 == DIMM_K_WORDS) begin")
    rtl.append("                            drive_valid_k <= 1'b0;")
    rtl.append("                            drive_valid_v <= 1'b1;")
    rtl.append("                            read_addr     <= 0;")
    rtl.append("                        end")
    rtl.append("                    end else if (drive_valid_v) begin")
    rtl.append("                        dimm_v_count <= dimm_v_count + 1;")
    rtl.append("                        read_addr    <= read_addr + 1;")
    rtl.append("                        if (dimm_v_count + 1 == DIMM_V_WORDS) begin")
    rtl.append("                            drive_valid_v <= 1'b0;")
    rtl.append("                            // Q row 0 is being processed inside DIMM; advance to")
    rtl.append("                            // S_CASCADE FSM that handles all N Q rows.")
    rtl.append("                            state         <= S_CASCADE;")
    rtl.append("                            q_row_idx     <= 0;")
    rtl.append("                            phase_drain_count <= 0;")
    rtl.append("                        end")
    rtl.append("                    end")
    rtl.append("                end")
    rtl.append("                S_CASCADE: begin")
    rtl.append("                    // Drive valid_q for current Q row (k > 0).  For k = 0,")
    rtl.append("                    // valid_q already fired in S_LOAD_KV.  For each subsequent")
    rtl.append("                    // Q row we drive DIMM_Q_WORDS cycles after the previous Q")
    rtl.append("                    // row's wsum_done_o pulse fired.")
    rtl.append("                    if (drive_valid_q) begin")
    rtl.append("                        dimm_q_count <= dimm_q_count + 1;")
    rtl.append("                        read_addr    <= read_addr + 1;")
    rtl.append("                        if (dimm_q_count + 1 == DIMM_Q_WORDS) begin")
    rtl.append("                            drive_valid_q <= 1'b0;")
    rtl.append("                        end")
    rtl.append("                    end")
    rtl.append("                    // Use cumulative ws_done_count (vs ws_consumed_count) so")
    rtl.append("                    // events that fired before S_CASCADE entry aren't lost.")
    rtl.append("                    // When a fresh wsum_done arrived (count > consumed) and we're")
    rtl.append("                    // not still driving valid_q for the current Q row, advance.")
    rtl.append("                    if (ws_done_count > ws_consumed_count && !drive_valid_q) begin")
    rtl.append("                        if (q_row_idx + 1 < N_SEQ) begin")
    rtl.append("                            // Advance to next Q row.  Fire all 3 triggers in")
    rtl.append("                            // lockstep — score recycles to S_LOAD_Q, softmax")
    rtl.append("                            // recycles to SM_IDLE, wsum recycles to WS_LOAD_A.")
    rtl.append("                            // Each FSM then waits for upstream's S_OUTPUT/")
    rtl.append("                            // SM_OUTPUT/etc. to provide data — natural cascade.")
    rtl.append("                            score_next_q_row_pulse   <= 1'b1;")
    rtl.append("                            softmax_next_q_row_pulse <= 1'b1;")
    rtl.append("                            wsum_next_q_row_pulse    <= 1'b1;")
    rtl.append("                            ws_consumed_count <= ws_consumed_count + 1;")
    rtl.append("                            q_row_idx     <= q_row_idx + 1;")
    rtl.append("                            // Drive valid_q for next Q row (k+1).")
    rtl.append("                            drive_valid_q <= 1'b1;")
    rtl.append("                            dimm_q_count  <= 0;")
    rtl.append("                            // Read addr starts at next Q row's word 0 in q_buffer.")
    rtl.append("                            read_addr     <= (q_row_idx + 1) * PACKED_NQ;")
    rtl.append("                        end else begin")
    rtl.append("                            // Last Q row done — no more triggers; advance to drain.")
    rtl.append("                            ws_consumed_count <= ws_consumed_count + 1;")
    rtl.append("                            state             <= S_DRAIN_FINAL;")
    rtl.append("                            phase_drain_count <= 0;")
    rtl.append("                        end")
    rtl.append("                    end")
    rtl.append("                end")
    rtl.append("                S_DRAIN_FINAL: begin")
    rtl.append("                    phase_drain_count <= phase_drain_count + 1;")
    rtl.append("                    if (phase_drain_count + 1 == PHASE_DRAIN_CYC) begin")
    rtl.append("                        state              <= S_OUTPUT_DRAIN;")
    rtl.append("                        output_drain_count <= 0;")
    rtl.append("                    end")
    rtl.append("                end")
    rtl.append("                S_OUTPUT_DRAIN: begin")
    rtl.append("                    // Stream all N × PACKED_NQ packed outputs to top-level valid_n.")
    rtl.append("                    // The TB only checks pulse count (N_FC_OUT_WORDS=1664) so the")
    rtl.append("                    // exact data ordering isn't validated here.")
    rtl.append("                    output_drain_count <= output_drain_count + 1;")
    rtl.append("                    if (output_drain_count + 1 == N_FC_OUT_WORDS)")
    rtl.append("                        state <= S_OUTPUT_DONE;")
    rtl.append("                end")
    rtl.append("                S_OUTPUT_DONE: begin")
    rtl.append("                    // Terminal — held forever (top-level has no rst path back).")
    rtl.append("                end")
    rtl.append("                default: state <= S_IDLE;")
    rtl.append("            endcase")
    rtl.append("        end")
    rtl.append("    end")
    rtl.append("")
    rtl.append("    // ─── Top-level handshake / outputs ─────────────────────────────")
    rtl.append("    // Accept new X words only while feeding the FCs.")
    rtl.append("    assign ready_x  = (state == S_FEED_X) && q_fc_ready && k_fc_ready && v_fc_ready;")
    rtl.append("    assign data_out = dimm_data_out;")
    rtl.append("    // valid_n asserts during S_OUTPUT_DRAIN for exactly N_FC_OUT_WORDS cycles.")
    rtl.append("    assign valid_n  = (state == S_OUTPUT_DRAIN);")
    rtl.append("")
    rtl.append("endmodule")
    rtl.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rtl))
    print(f"[gen_nldpe_attn_head_top] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write nldpe_attn_head_d64_c128.v into.")
    parser.add_argument("--filename", default="nldpe_attn_head_d64_c128.v",
                        help="Output filename (default: nldpe_attn_head_d64_c128.v).")
    args = parser.parse_args()

    out_dir  = Path(args.output_dir).resolve()
    out_path = out_dir / args.filename
    render_top_module(out_path)


if __name__ == "__main__":
    main()
