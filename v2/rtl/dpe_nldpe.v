// ============================================================================
// dpe_nldpe.v — NL-DPE primitive, v2 clean-room skeleton
//
// Ground truth: `v2/spec/dpe_nldpe.md` (**v2.0 integer dataflow, 2026-09-14**).
// Hand-written from the spec; legacy RTL (`rtl_flow/rtl/dpe_nldpe_faithful.v`)
// is a read-only witness — do not consult or copy it while implementing.
//
// INTERFACE FREEZE (2026-09-15): the port surface is identical to the legacy
// contract (`rtl_flow/vtr/dpe_blackbox.v`, arch XML `nl_dpe/*.xml`
// <model name="dpe">, `rtl_flow/rtl/dpe_nldpe*.v`): same 15 ports in the same
// order, same directions, widths `[DPE_BUF_WIDTH-1:0]`, outputs declared
// `reg`. Parity is machine-checked by `v2/smoke/check_interface.py`.
//
// Parameter surface = legacy superset:
//   KERNEL_WIDTH, NUM_COLS, DPE_BUF_WIDTH, PRECISION — drive the v2 datapath
//     (spec §1 names R/C/BUF/P; aliases defined below).
//   PIPELINE_DEPTH, ACAM_CYCLES, COMPUTE_CYCLES, ACAM_MODE — accepted for
//     instantiation compatibility with legacy wrappers/TBs but NOT used:
//     v2 timing is structural (COMPUTE_CYC = P+2 must emerge, P10) and the
//     ACAM mode comes from the `nl_dpe_control` port as workload
//     configuration (P27).
//
// Port semantics: spec §4. `w_buf_en` = ACT strobe; `load_input_reg` =
// WEIGHT strobe (one int8 weight per cycle on data_in[7:0], row-major
// row-outer, P23; untimed setup, WR_CYC excluded from pass cycles). The ACAM
// mode is latched from `nl_dpe_control` on the WEIGHT strobes into `mode_q`
// and held for the whole workload (P27) — it is not sampled per pass.
// Reserved inputs are tied 0 by the wrapper; `shift_add_bypass_ctrl` drives 0.
//
// Numeric contract: int8 weights/activations, exact integer MAC into a 32-bit
// accumulator (|y| <= R*2^14, safe for R <= 131072, P25), integer ACAM mode
// forms, then `trunc8` (clamp to int32, keep the low byte, P16). No rounding,
// no fp32 cores.
//
// TODO(you) blocks (spec anchors):
//   1 weight storage + WEIGHT strobe path            §3 / §4.2 / P23
//   2 input buffer: P banks x R bits + corner-turn   §3 / §4.3 / P1
//   3 crossbar integer slice partials s_b            §6 F2
//   4 partial-shift accumulate + MSB subtract        §6 F2 / P2 / P22
//   5 ACAM integer modes -> trunc8 (low byte)        §6 F3 / P16 / P24
//   6 output buffer + drain (5 bytes/cyc) + done     §4.5 / P11
//   7 timing FSM + readiness                         §5.1-5.3 / P1 / P10
//
// COMPUTE_CYC = P+2 must emerge from the structure (P10); Δ_impl is a single
// implementation-declared constant, invariant across M and modes (§5.3).
// The TB compares both the hierarchical int32 `y` and the 8-bit stream (P26).
//
// TB probe contract (verification-only internals, D8 — NOT ports; names are
// frozen so `v2/tb/tb_dpe_nldpe.v` can read them):
//   state       : FSM state, 0=IDLE/WEIGHT, 1=LOAD, 2=COMPUTE (P fires +
//                 MSB shift&acc), 3=ACAM, 4=OUTPUT
//   acc         : reg signed [31:0] acc [0:NUM_COLS-1] — pre-ACAM crossbar
//                 output (F2); valid until the ACAM fire consumes it
//   acam_fire   : 1-cycle pulse when ACAM converts `acc` into the output
//                 buffer (mode form -> trunc8); acc is stable in that cycle
//   drain_valid : 1-cycle pulse per drained 5-byte output word (OUTPUT_CYC
//                 pulses per pass; `data_out` carries that word in-cycle)
// ============================================================================

`timescale 1ns / 1ps

module dpe #(
    parameter KERNEL_WIDTH   = 256,   // spec §1 R
    parameter NUM_COLS       = 256,   // spec §1 C (reference: 512)
    parameter DPE_BUF_WIDTH  = 40,    // spec §1 BUF
    parameter PRECISION      = 8,     // spec §1 P
    parameter PIPELINE_DEPTH = 2,     // legacy-compat, unused by v2
    parameter ACAM_CYCLES    = 1,     // legacy-compat, unused by v2
    parameter COMPUTE_CYCLES = 10,    // legacy-compat, unused by v2
    parameter ACAM_MODE      = 0      // legacy-compat, unused (runtime port)
)(
    input  wire                       clk,
    input  wire                       reset,
    input  wire [DPE_BUF_WIDTH-1:0]   data_in,
    input  wire [1:0]                 nl_dpe_control,
    input  wire                       shift_add_control,
    input  wire                       w_buf_en,
    input  wire                       shift_add_bypass,
    input  wire                       load_output_reg,
    input  wire                       load_input_reg,
    output reg                        MSB_SA_Ready,
    output reg  [DPE_BUF_WIDTH-1:0]   data_out,
    output reg                        dpe_done,
    output reg                        reg_full,
    output reg                        shift_add_done,
    output reg                        shift_add_bypass_ctrl
);

    // ------------------------------------------------------------------
    // Spec §1 aliases (readability; frozen parameter names are legacy)
    // ------------------------------------------------------------------
    localparam R   = KERNEL_WIDTH;
    localparam C   = NUM_COLS;
    localparam P   = PRECISION;
    localparam BUF = DPE_BUF_WIDTH;

    // ------------------------------------------------------------------
    // Derived stage lengths (§5.1/§5.3) — no other constants allowed
    // ------------------------------------------------------------------
    localparam LOAD_CYC    = (R * 8 + BUF - 1) / BUF;   // §4.3
    localparam COMPUTE_CYC = P + 2;                     // §5.1
    localparam OUTPUT_CYC  = (C * 8 + BUF - 1) / BUF;   // §4.5
    localparam WR_CYC      = R * C;                     // P23, one-time

    // ------------------------------------------------------------------
    // TODO(you): 1 — weight storage + WEIGHT strobe path + mode_q
    //   §3: R*C int8 words, one per (r, c); §4.2: one byte per strobe
    //   cycle on data_in[7:0], order row-major row-outer (P23);
    //   stationary after programming (A5/A12). WR_CYC = R*C one-time,
    //   excluded from T_fill/T_steady (untimed setup), reported separately.
    //   P27: while load_input_reg is high, latch nl_dpe_control into
    //   `mode_q` (last strobe wins); `mode_q` is the workload's ACAM mode
    //   for every pass. Suggested: `if (load_input_reg) mode_q <= nl_dpe_control;`
    // ------------------------------------------------------------------
    reg signed [PRECISION-1:0] weights [0: KERNEL_WIDTH * NUM_COLS - 1];  // DPE internal weights array
    reg [1:0] mode_q;
    integer wr_ptr;
    wire weights_full = wr_ptr == R * C -1;
    always@(posedge clk) begin
        if (reset) begin
            wr_ptr <= 0;
            mode_q <= 0;
        end
        else if (load_input_reg) begin
            weights[wr_ptr] <= data_in[P - 1:0];
            wr_ptr  <= weights_full? 0: wr_ptr + 1;
            mode_q <= nl_dpe_control;
        end
    end


    // ------------------------------------------------------------------
    // TODO(you): 2 — input buffer (single, P1) + corner-turn (§4.3)
    //   P banks x R bits; byte-major writes: bank[b][j] = bit b of x[j].
    //   Burst = R bytes, 5 bytes/cycle. Refill permitted only from the
    //   cycle after MSB fire of the in-flight pass (MSB_SA_Ready rises).
    // ------------------------------------------------------------------
    reg[0: R-1] banks [0: P - 1];
    
    localparam INPUT_ELEMENTS= DPE_BUF_WIDTH / PRECISION;
    
    integer buf_ptr, k, j;
    wire burst_complete = (buf_ptr + INPUT_ELEMENTS) >= R;

    always @(posedge clk) begin
        if (reset) begin
            buf_ptr <= 0;
        end
        else if(w_buf_en & MSB_SA_Ready) begin
           for(k = 0; k < INPUT_ELEMENTS; k = k + 1) begin
                for(j = 0; j < PRECISION; j = j + 1) begin
                    if ((buf_ptr + k) < R)
                        banks[j][buf_ptr + k] <= data_in[P * k + j]; // store bit-slices
                end
           end 
            buf_ptr <= burst_complete? 0 : buf_ptr + INPUT_ELEMENTS; 
        end
    end


    // ------------------------------------------------------------------
    // TODO(you): 3+4 — crossbar integer slice partials (§6 F2)
    //   Per slice b (LSB -> MSB), per column c:
    //   s_b[c] = sum_r ( bank_b[r] ? W[r,c] : 0 )    — exact integer.
    //   One slice per cycle.
    //   partial-shift accumulate + MSB subtract
    //   y = sum_{b=0..P-2} (s_b << b)  -  (s_{P-1} << (P-1)).
    // ------------------------------------------------------------------
    reg signed [31:0] acc [0:C-1];        // probe name
    integer r, c, s_local;
    always @(posedge clk) begin
        if (fire_en) begin
            for (c = 0; c < C; c = c + 1) begin
                s_local = 0;
                for (r = 0; r < R; r = r + 1)
                    if (banks[slice_b][r]) s_local = s_local + weights[r*C + c];
                if (slice_b == 0)        acc[c] <= s_local;
                else if (slice_b == P-1) acc[c] <= acc[c] - (s_local << (P-1));
                else                     acc[c] <= acc[c] + (s_local << slice_b);
            end
        end
    end

    // ------------------------------------------------------------------
    // TODO(you): 5 — ACAM integer modes -> trunc8 (§6 F3 / P16 / P24)
    //   REGULAR: y;  ACTIVATION: relu(y);  EXP: 1 + y + floor(y^2/2)
    //   (wide intermediate; the int32 clamp in trunc8 is live); LOG: y - 1.
    //   Then trunc8: clamp to int32, keep the low byte as signed int8.
    //   C units parallel, 1 cycle, mode = `mode_q` (workload config, P27;
    //   no per-pass sampling).
    // ------------------------------------------------------------------
    localparam [1:0] STD = 0, ACT = 1, EXP = 2, LOG = 3;

    reg signed [7:0]  y8 [0:C-1];          // per-column int8 result (T2)
    reg signed [63:0] y64, e64;
    always @(*) begin
        for (c = 0; c < C; c = c + 1) begin
            case (mode_q)                  
                STD: y8[c] = acc[c][7:0];                          
                ACT: y8[c] = acc[c][31] ? 8'b0 : acc[c][7:0];    
                EXP: begin
                    y64 = acc[c];                                  
                    e64 = 1 + acc[c] + ((acc[c] * acc[c]) >>> 1);      
                    if      (e64 >  64'sd2147483647) e64 =  64'sd2147483647; // cap at int32
                    else if (e64 < -64'sd2147483648) e64 = -64'sd2147483648;
                    y8[c] = e64[7:0];                              
                end
                LOG: y8[c] = acc[c][7:0] - 8'd1;                 
                default: y8[c] = 8'sd0;
            endcase
        end
    end


    // ------------------------------------------------------------------
    // TODO(you): 6 — output buffer + drain (§4.5 / P11)
    //   C x 8 bits, written wholesale by ACAM; drained 5 bytes/cycle in
    //   column order, gapless; dpe_done 1-cycle pulse after the last byte;
    //   reg_full marks the buffer busy.
    // ------------------------------------------------------------------
    reg [7:0] obuf [0:C-1];
    integer c6;
    always @(posedge clk) begin
        if (acam_fire)                              // FSM pulse, cs+P+1
            for (c6 = 0; c6 < C; c6 = c6 + 1) obuf[c6] <= y8[c6];
    end

    // ------------------------------------------------------------------
    // TODO(you): 7 — timing FSM + readiness (§5.1-5.3 / P1 / P10)
    //   LOAD_CYC -> COMPUTE_CYC (= P+2, must emerge structurally) ->
    //   ACAM -> OUTPUT_CYC; next ACT burst may start the cycle after MSB
    //   fire; ACAM strictly after the previous drain (P11) and the
    //   accumulator is freed by the ACAM write (§5.2); reset synchronous
    //   active-high (A10); measured(M) = T_fill + (M-1)*T_steady + Δ_impl,
    //   Δ_impl invariant (I2).
    // ------------------------------------------------------------------

    // Placeholder drivers — replace as each TODO block lands (keeps this
    // file elaborating cleanly in the meantime). Outputs are `reg` to match
    // the legacy port declarations exactly.
    always @(data_in or nl_dpe_control or shift_add_control or w_buf_en
             or shift_add_bypass or load_output_reg or load_input_reg) begin
        MSB_SA_Ready          = 1'b0;
        data_out              = {DPE_BUF_WIDTH{1'b0}};
        dpe_done              = 1'b0;
        reg_full              = 1'b0;
        shift_add_done        = 1'b0;
        shift_add_bypass_ctrl = 1'b0;
    end

endmodule
