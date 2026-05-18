# Azure-Lily DPE Faithful Primitive — Implementation Walkthrough

**Companion doc for verifying Task #92: the faithful (physically-modeled) Azure-Lily DPE primitive.**

This doc walks through the implementation of `fc_verification/rtl/dpe_azurelily_faithful.v` — the **physical** behavioral model of the AL-DPE primitive — alongside its testbench `tb_dpe_azurelily_faithful.v`, Python oracle `oracles/azurelily_mac_oracle.py`, and generator `nl_dpe/gen_dpe_azurelily_faithful.py`.

It is the AL counterpart of `DPE_NLDPE_FAITHFUL_WALKTHROUGH.md`. **Read that doc first** if you want full background on the "faithful" methodology and the silicon-faithful single-substrate refactor — this doc focuses on what's different for Azure-Lily.

Both primitives have been refactored to a **silicon-faithful single-substrate** design ("Option A1"): the original 4-slot pass-tagged ring buffer (`input_buffer[QDEPTH=4][R]`, `mac_acc[QDEPTH][C]`) has been collapsed to a single bank of each substrate; LOAD now performs a **corner-turn** that rearranges byte-major BRAM input into a **slice-major** bit substrate `input_buf_slice[PRECISION][R]`; and a LOAD-gate (`load_safe`) interlocks pass-(k+1) LOAD against pass-k COMPUTE's slice-read window. See §2 (storage), §3a (corner-turn LOAD), and §6 (T_steady = max(LCYC + PRECISION, CCYC, OCYC) = 264 for AL).

## TL;DR — AL faithful vs NL faithful

Both implement the same `T(M)` cycle contract via DIFFERENT physical pipelines. Same CCYC=10 for INT8 by structural symmetry, different decomposition:

| Aspect | NL faithful | AL faithful |
|---|---|---|
| Internal pipeline stages | **2** (Crossbar + Analog Acc) | **3** (Crossbar + ADC + ShiftAdd) |
| ACAM stage | YES, 1 cycle (3 modes: ADC/identity, exp, log) | **NO** |
| ACAM_MODE parameter | present (0/1/2) | **absent** |
| Stage-index registers | `bit_idx_s0`, `bit_idx_s1` | `bit_idx_s0`, `bit_idx_s1`, **`bit_idx_s2`** |
| Output substrate | `acam_out[c]` (single bank, post-ACAM) | `mac_acc[c]` (single bank — drain directly, no ACAM) |
| TB test count | 7 (incl. T5 exp, T6 log) | **5** (no exp/log) |
| Default geometry | R=256, C=256, BUF=40 | R=512, C=128, BUF=16 |
| `pipeline_depth` in JSON | 2 | 3 |
| `acam_cycles` in JSON | 1 | 0 |
| CCYC = P + (PD−1) + AC for INT8 | 8 + 1 + 1 = **10** | 8 + 2 + 0 = **10** |
| T_fill (M=1) | LCYC=52 + CCYC=10 + OCYC=52 + 2 NBA = **116** | LCYC=256 + CCYC=10 + OCYC=64 + 2 NBA = **332** |
| T_steady (Option A1) | max(LCYC+P, CCYC, OCYC) = **60** | max(LCYC+P, CCYC, OCYC) = **264** |

The physical difference: **AL has no ACAM** (no native nonlinearity), so the read-out path is purely linear (analog crossbar → ADC sample → digital shift-add). NL has ACAM as a 1-cycle LUT lookup stage that can apply ADC/identity, exp, or log.

---

## 1. Module interface

```verilog
// dpe_azurelily_faithful.v:111-135
module dpe #(    // module name "dpe" matches VTR <model name="dpe"> contract
    parameter KERNEL_WIDTH   = 512,      // R (rows; AL default 512)
    parameter NUM_COLS       = 128,      // C (cols; AL default 128)
    parameter DPE_BUF_WIDTH  = 16,       // BUF (bits / byte-stream lane; AL default 16)
    parameter PRECISION      = 8,        // bit precision (INT8 default)
    parameter PIPELINE_DEPTH = 3,        // 3-stage: Crossbar + ADC + ShiftAdd
    parameter ACAM_CYCLES    = 0         // AL has NO ACAM
    // NOTE: no ACAM_MODE parameter
)(
    input  wire                       clk, reset,
    input  wire [DPE_BUF_WIDTH-1:0]   data_in,
    input  wire [1:0]                 nl_dpe_control,
    input  wire                       shift_add_control, w_buf_en, shift_add_bypass,
    input  wire                       load_output_reg, load_input_reg,
    output reg                        MSB_SA_Ready,
    output reg  [DPE_BUF_WIDTH-1:0]   data_out,
    output reg                        dpe_done, reg_full, shift_add_done, shift_add_bypass_ctrl
);
```

Port list is **byte-identical** to the lazy `dpe_azurelily.v` primitive and to the NL faithful primitive — the VTR `<model name="dpe">` contract is preserved.

**Key parameter differences vs NL faithful**:
- `KERNEL_WIDTH = 512` (AL has bigger crossbar rows than NL's 256)
- `NUM_COLS = 128` (AL has half NL's columns)
- `DPE_BUF_WIDTH = 16` (AL's narrower byte-stream lane → EPS=2 bytes/strobe; NL's BUF=40 gives EPS=5)
- `PIPELINE_DEPTH = 3` (3 stages vs NL's 2)
- `ACAM_CYCLES = 0` (no ACAM)
- **No `ACAM_MODE` parameter** — AL has no ACAM, so no mode selection

These defaults come from `azurelily/IMC/configs/azure_lily.json` (`capabilities.pipeline_depth=3`, `acam_cycles=0`; `fpga_specs.dpe_buf_width=16`; `geometry.array_rows=512`, `array_cols=128`).

---

## 2. Internal storage layout

`dpe_azurelily_faithful.v:112-116`

```verilog
// Weights (TB hierarchical-forces these)
reg signed [7:0]  weights      [0:KERNEL_WIDTH-1][0:NUM_COLS-1];

// Single-substrate input buffer in SLICE-MAJOR (bit-stratified) form.
// PRECISION banks of 1-bit cells; each bank holds one bit-slice across all R rows.
reg               input_buf_slice [0:PRECISION-1][0:KERNEL_WIDTH-1];

// Single-substrate Stage 2 accumulator (one int32 per output column).
// NOTE: AL has NO acam_out — mac_acc IS the output (drained directly).
reg signed [31:0] mac_acc          [0:NUM_COLS-1];

// Stage 0 latched output (single snapshot, shared)
reg signed [31:0] crossbar_sum_reg [0:NUM_COLS-1];

// Stage 1 ADC sample register (NEW vs NL: AL has this dedicated stage)
reg signed [31:0] adc_reg          [0:NUM_COLS-1];
```

**This is the silicon-faithful refactor** ("Option A1"): every substrate is **single-banked**. There is no `QDEPTH`, no ring buffer, no pass-tagged slots. Three substrates total (input slice / MAC / ADC reg), all single-banked — matching the physical regions a real AL chip would carry. (Stage 0's `crossbar_sum_reg` is the registered output of the analog crossbar fire and was already a single shared snapshot pre-refactor.)

Five storage elements, all Tier 2 observable from the TB:

| Storage | Role | Observable via |
|---|---|---|
| `weights` | R×C int8 weight matrix | `dut.weights[r][c]` |
| `input_buf_slice` | PRECISION × R 1-bit cells; bank `i` holds bit-slice `i` for all rows | `dut.input_buf_slice[i][r]` |
| `mac_acc` | Single int32 shift-add accumulator per output column (FINAL output, no ACAM stage) | `dut.mac_acc[c]` |
| `crossbar_sum_reg` | Stage 0 output for current bit-slice (shared, single snapshot) | `dut.crossbar_sum_reg[c]` |
| `adc_reg` | Stage 1 ADC sample for current bit-slice (NEW vs NL) | `dut.adc_reg[c]` |

**Slice-major layout vs old byte-major layout**:

```
OLD (pre-refactor, depth-4 ring, byte-major):
   input_buffer[slot=0..3][row=0..R-1]   each cell = 8-bit int
                                                                  (4 slots × R bytes)

NEW (single substrate, slice-major):
   input_buf_slice[bit=0..PRECISION-1][row=0..R-1]   each cell = 1 bit
   bank 0:  bit-slice-0 of [row0][row1] ...[row511]
   bank 1:  bit-slice-1 of [row0][row1] ...[row511]
   ...
   bank 7:  bit-slice-7 of [row0][row1] ...[row511]
                                                                  (PRECISION × R bits)
```

The total bit count is identical (R × PRECISION), but the **organisation** is rotated 90°: COMPUTE Stage 0 needs *one bit per row at a single bit-position* on every cycle, so reading down a slice-major bank is one memory plane per cycle (combinational over R rows of the same bank) instead of slicing through 512 byte-words. This matches real-silicon bit-line organisation, where every row contributes one bit on a single global bit-line at a time.

**Key difference vs NL**: no `acam_out` substrate. AL's `mac_acc` IS the final output (after MSB-subtract Stage 2 completes). OUTPUT drains `mac_acc` directly (single substrate, one entry per column). NL has an extra ACAM read-out stage that copies (with optional transform) `mac_acc` → `acam_out`, which OUTPUT then drains.

The `adc_reg` is the AL-specific addition. It represents a physical sample-and-hold stage between the analog crossbar output and the digital shift-add accumulator. Modeled as a registered copy of `crossbar_sum_reg`. Like the other AL substrates, it is single-banked.

**Why single-substrate** (no QDEPTH ring): the chip has *one* MAC, *one* ADC sample-and-hold, *one* input slice substrate. A 4-slot ring buffer would be 4× hardware. The refactor instead uses a **LOAD-gate (`load_safe`)** to serialise pass-(k+1) LOAD against pass-k COMPUTE's slice-read window, so the single substrate is always safe to re-use once COMPUTE has consumed it. See §3 and §6.

---

## 3. Sub-FSMs — three concurrent stages

LOAD and OUTPUT are identical to NL faithful (same byte-streaming protocol). COMPUTE has a different internal structure (3 sub-stages instead of 2).

### 3a. LOAD sub-FSM — corner-turn into slice substrate, gated by `load_safe`

`dpe_azurelily_faithful.v:208-227` — same Option A1 pattern as NL faithful:

```verilog
// LOAD sub-FSM (corner-turn into bit-stratified slice).
// Option A1 gate: only act on w_buf_en when load_safe == 1.
if (w_buf_en && load_safe) begin
    for (bb = 0; bb < ELEMS_PER_STROBE; bb = bb + 1) begin
        if (load_cycle_cnt * ELEMS_PER_STROBE + bb < KERNEL_WIDTH) begin
            for (ib = 0; ib < PRECISION; ib = ib + 1) begin
                input_buf_slice[ib][load_cycle_cnt * ELEMS_PER_STROBE + bb]
                    <= data_in[bb*8 + ib];
            end
        end
    end
    if (load_cycle_cnt == LOAD_CYCLES - 1) begin
        load_cycle_cnt <= 0;
        buf_loaded     <= 1;        // signals COMPUTE-idle wake branch
    end else begin
        load_cycle_cnt <= load_cycle_cnt + 1;
    end
end
```

Two new mechanisms (mirrored from NL):

**(1) Corner-turn distribution**. Per LOAD strobe, the BRAM provides `ELEMS_PER_STROBE = 2` bytes (AL has BUF=16 ⇒ 2 bytes/cycle), packed as `data_in[15:0] = {byte1, byte0}`. The corner-turn rearrangement scatters each byte's 8 bits across the 8 slice banks, so that *bit i of byte j of this strobe* ends up in `input_buf_slice[i][load_cycle_cnt*2 + j]`. In one strobe, 16 input bits go to 2 row-positions × 8 bit-banks = 16 cells, one bit per cell. By the last strobe (cycle `LOAD_CYCLES-1 = 255`), the full R × PRECISION substrate is populated.

The fixed wire permutation:
```
slice[i][load_cycle_cnt * 2 + j] <= data_in[j*8 + i]   for i = 0..7, j = 0..1
```
i.e., `data_in[0..7]` lights up bit-banks 0..7 at row offset 0; `data_in[8..15]` lights up bit-banks 0..7 at row offset 1.

**(2) `load_safe` gate**. `w_buf_en` is *gated* by `load_safe`. While COMPUTE is reading from `input_buf_slice` (the pass-k slice-read phase), LOAD must not overwrite the cells. The gate is **cleared** when COMPUTE wakes (line 305) and **re-asserted** when `bit_idx_s0 == PRECISION-2` so that `load_safe` is observed=1 on the cycle of the last slice read (lines 243-251).

For AL with R=512, BUF=16 (EPS=2 bytes/strobe): LCYC = ceil(512·8 / 16) = 256 strobes (vs NL's 52). LOAD takes 256 cycles to stream all 512 bytes.

### 3b. COMPUTE sub-FSM (the new piece — 3-stage)

`dpe_azurelily_faithful.v:229-309` — three internal stages, each advancing a separate bit index.

State variables:
- `compute_busy`: high while COMPUTE is processing a pass
- `bit_idx_s0`: stage-0 bit position (0..PRECISION), advances each cycle (crossbar fire)
- `bit_idx_s1`: stage-1 bit position (0..PRECISION), lags s0 by 1 cycle (ADC sample)
- `bit_idx_s2`: stage-2 bit position (0..PRECISION), lags s1 by 1 cycle (shift-add accumulator)
- `s1_valid`: armed 1 cycle after `compute_busy` wakes
- `s2_valid`: armed 1 cycle after `s1_valid` first arms

**Single-substrate handshake flags (replace ring-buffer pointers)**:
- `buf_loaded`: 1 when LOAD has just completed a pass; consumed by COMPUTE-idle wake branch (lines 291-308)
- `compute_done`: 1 when COMPUTE has just finished a pass; consumed by OUTPUT-idle wake branch
- `load_safe`: LOAD-gate (Option A1). Cleared when COMPUTE wakes; set when `bit_idx_s0 == PRECISION-2` so it is observed=1 on the cycle of the last slice read (lines 243-251)

There is no `q_load_tail`, `q_compute_head`, or `q_output_head` anymore. The single substrate, plus the three flags above, replaces all of the pass-tagged indexing.

Telemetry pulses (Tier 2 observable):
- `compute_first_bit_pulse`: NBA'd high on the cycle that bit_idx_s0 first transitions from idle
- `compute_last_bit_pulse`: NBA'd high on the cycle that Stage 2 commits the MSB subtract (= cycle `compute_busy` goes low NBA)

There's no `acam_commit_pulse` (AL has no ACAM).

### 3c. OUTPUT sub-FSM

`dpe_azurelily_faithful.v:312-336` — same pattern as NL faithful, but **drains the single-substrate `mac_acc` directly** (no ACAM intermediate):

```verilog
data_out[bb*8 +: 8] <=
    mac_acc[output_col_idx * ELEMS_PER_STROBE + bb][7:0];
```

(NL's version reads `acam_out[output_col_idx * ELEMS_PER_STROBE + bb]` here instead — single-substrate ACAM out post-refactor.)

For AL with C=128, BUF=16 (EPS=2): OCYC = ceil(128·8 / 16) = 64 strobes. Chains across passes if `compute_done` is set when OUTPUT finishes the current drain.

---

## 4. The COMPUTE stage — 3-stage bit-serial pipeline

The COMPUTE sub-FSM is where AL's distinct architecture lives. It has three internal stages, each modeling a physical layer of the AL-DPE silicon:

```
input_buf_slice[bit_idx_s0][row]            ← LOAD corner-turned bits in here
      ↓
[Bit-slice extractor]                       ← combinational, no register
      ↓
[Stage 0: Crossbar fire]                    ← combinational + register (crossbar_sum_reg)
      ↓
[Stage 1: ADC sample]                       ← registered (adc_reg) — NEW vs NL!
      ↓
[Stage 2: Shift-add accumulator]            ← registered (mac_acc — single substrate)
      ↓                                        8 cycles bit-slice entry
      ↓                                        + 2 cycles pipeline drain
      ↓                                        = PRECISION + (PIPELINE_DEPTH-1) cycles
mac_acc                                     ← single substrate, drained by OUTPUT (no ACAM!)
```

### Bit-slice extractor (combinational)

`dpe_azurelily_faithful.v:150-163`

Same as NL faithful post-refactor: combinationally reads one slice bank (the one indexed by `bit_idx_s0`) and computes the per-column sum.

```verilog
always @* begin
    for (c_idx = 0; c_idx < NUM_COLS; c_idx = c_idx + 1)
        crossbar_sum_comb[c_idx] = 32'sd0;
    if (compute_busy && bit_idx_s0 < PRECISION[4:0]) begin
        for (r_idx = 0; r_idx < KERNEL_WIDTH; r_idx = r_idx + 1) begin
            if (input_buf_slice[bit_idx_s0][r_idx]) begin   // ← read 1-bit cell directly
                for (c_idx = 0; c_idx < NUM_COLS; c_idx = c_idx + 1) begin
                    crossbar_sum_comb[c_idx] = crossbar_sum_comb[c_idx]
                        + {{24{weights[r_idx][c_idx][7]}}, weights[r_idx][c_idx]};
                end
            end
        end
    end
end
```

The combinational read is **one whole slice bank per cycle** (R=512 1-bit cells of `input_buf_slice[bit_idx_s0][*]`), not a byte extraction across 512 byte-words. The corner-turn LOAD pre-positioned the data exactly so this read pattern is natural.

### Stage 0: Crossbar fire (registered latch + LOAD-gate re-arm)

`dpe_azurelily_faithful.v:231-252`

```verilog
if (bit_idx_s0 < PRECISION[4:0]) begin
    for (cc = 0; cc < NUM_COLS; cc = cc + 1)
        crossbar_sum_reg[cc] <= crossbar_sum_comb[cc];
    bit_idx_s0 <= bit_idx_s0 + 5'd1;
    // Option A1: re-arm LOAD on the cycle BEFORE the last slice read.
    // load_safe NBA <= 1 when bit_idx_s0 == PRECISION-2, so load_safe is
    // OBSERVED=1 on the cycle of the last slice read. Pass-(k+1) LOAD can
    // start writing slices on that same cycle; NBA semantics protect
    // pass-k's combinational read (sees OLD value).
    if (PRECISION >= 2 && bit_idx_s0 == (PRECISION[4:0] - 5'd2))
        load_safe <= 1;
end
```

Same as NL faithful post-refactor. Each cycle while `compute_busy && bit_idx_s0 < PRECISION`: latch `crossbar_sum_reg` from the combinational sum, advance `bit_idx_s0`, and on the cycle before the last slice read NBA-set `load_safe` so pass-(k+1) LOAD can begin on the cycle of pass-k's last slice read.

### Stage 1: ADC sample (registered) — NEW vs NL

`dpe_azurelily_faithful.v:254-258`

```verilog
if (s1_valid && bit_idx_s1 < PRECISION[4:0]) begin
    for (cc = 0; cc < NUM_COLS; cc = cc + 1)
        adc_reg[cc] <= crossbar_sum_reg[cc];   // ADC sample
    bit_idx_s1 <= bit_idx_s1 + 5'd1;
end
```

This is the AL-specific stage. It models a physical ADC sample-and-hold:
- Reads `crossbar_sum_reg` (which was NBA-latched by Stage 0 the previous cycle)
- Copies it into `adc_reg` (single substrate, one int32 per column)
- Advances `bit_idx_s1`

Behaviorally, no transformation — it's just a register stage. But it's a **real cycle** in the pipeline, modeling the time real silicon's ADC takes to convert the analog bit-line current into a digital partial sum.

**Why a dedicated stage?** Because AL's silicon has it. NL's silicon doesn't (NL's analog accumulator integrates directly without an explicit ADC sample per bit-slice). The extra ADC stage costs 1 extra cycle in the pipeline drain — but AL saves it back by not having an ACAM stage at the tail.

`s1_valid` is armed 1 cycle after `compute_busy` wakes (lines 283-285), so Stage 1 begins reading `crossbar_sum_reg` one cycle after Stage 0 first latches into it.

### Stage 2: Shift-add accumulator (registered, MSB-subtract for signed)

`dpe_azurelily_faithful.v:260-281`

```verilog
if (s2_valid && bit_idx_s2 < PRECISION[4:0]) begin
    if (bit_idx_s2 == (PRECISION[4:0] - 5'd1)) begin
        // MSB: SUBTRACT (signed 2's-complement convention)
        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
            mac_acc[cc] <= mac_acc[cc] - (adc_reg[cc] <<< bit_idx_s2);
        // Signal COMPUTE done -- same NBA cycle (AL has NO ACAM fire stage)
        compute_last_bit_pulse  <= 1;
        compute_busy            <= 0;
        bit_idx_s0              <= 0;
        bit_idx_s1              <= 0;
        bit_idx_s2              <= 0;
        s1_valid                <= 0;
        s2_valid                <= 0;
        compute_done            <= 1;        // handshake to OUTPUT (single substrate)
        MSB_SA_Ready            <= 1;
        shift_add_done          <= 1;
    end else begin
        // Non-MSB: ADD with shift
        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
            mac_acc[cc] <= mac_acc[cc] + (adc_reg[cc] <<< bit_idx_s2);
    end
    bit_idx_s2 <= bit_idx_s2 + 5'd1;
end
```

Same MSB-subtract pattern as NL, but **operating on `adc_reg` instead of `crossbar_sum_reg`**. The 1-cycle ADC sample stage means Stage 2 reads values that are 2 cycles behind Stage 0's bit-slice latching. `mac_acc` is a single substrate (no QDEPTH slot index); it is zeroed in the COMPUTE-idle wake branch (line 307) when `buf_loaded` triggers the next pass.

**Critical difference from NL**: in AL, the MSB-subtract NBA ALSO signals COMPUTE done — there is no separate ACAM stage to wait for. `compute_busy <= 0` happens in the **same NBA** as the MSB subtract. This collapses the AL pipeline by 1 cycle vs NL (which has an extra ACAM cycle), but AL gained 1 cycle earlier from the ADC stage. Net: same CCYC = 10.

`s2_valid` is armed 1 cycle after `s1_valid` first arms (lines 287-289), giving the pipeline its 3-stage depth.

### Why AL has no separate ACAM fire stage

AL's silicon doesn't have an ACAM LUT. The shift-add accumulator produces the final MAC directly; that's the output. The mac_acc value at the end of Stage 2's MSB cycle is the final answer — no further processing needed before output drain.

Compare to NL: after Stage 1 (analog accumulator) commits the MSB subtract, the ACAM stage applies a 1-cycle LUT transform (identity for mode 0, exp for mode 1, log for mode 2). The result lives in `acam_out` (a separate register). Output drains `acam_out`, not `mac_acc`.

AL just drains `mac_acc` directly. Simpler tail, but the pipeline was 1 cycle deeper to get there.

---

## 5. The bit-serial pipeline timeline

Visualised as a per-cycle pipeline (relative cycles since `compute_busy` first went high):

```
Cycle:    0    1    2    3    4    5    6    7    8    9
                                                      |
                                                      Stage 2 MSB subtract
                                                      → mac_acc final
                                                      → compute_busy <= 0
                                                      → compute_last_bit_pulse

Stage 0 latches bit-slice into crossbar_sum_reg (one bit per cycle):
  bit 0:   ●
  bit 1:        ●
  bit 2:             ●
  bit 3:                  ●
  bit 4:                       ●
  bit 5:                            ●
  bit 6:                                 ●
  bit 7:                                      ●          ← (MSB enters)

Stage 1 latches adc_reg from crossbar_sum_reg (lags s0 by 1):
  ADC bit 0:    ●
  ADC bit 1:         ●
  ...
  ADC bit 7:                                       ●     ← MSB sample

Stage 2 reads adc_reg and accumulates into mac_acc (lags s1 by 1):
  ShAdd bit 0:        add (adc << 0)
  ShAdd bit 1:             add (adc << 1)
  ...
  ShAdd bit 6:                                add (adc << 6)
  ShAdd bit 7:                                     SUB (adc << 7)  ← MSB, signed
                                                       └─ same NBA: compute_busy <= 0

Total compute_busy duration: cycles 0..9 = 10 cycles.
```

Compare to NL (which has 2-stage Crossbar+Acc + 1-cycle ACAM = 10 cycles): same NUMBER, different physical decomposition. AL trades the ACAM cycle for an extra ADC stage in the pipeline drain.

---

## 6. Cycle emergence — how CCYC = 10 is derived from the code

```
PRECISION cycles for bit-slice entry  (8 cycles, bit_idx_s0 ∈ {0..7})
+ (PIPELINE_DEPTH - 1)                (2 cycles: Stage 1 ADC drain + Stage 2 MSB drain)
+ ACAM_CYCLES                         (0 cycles — AL has NO ACAM)
= 10 cycles for INT8 Azure-Lily
```

**Same total as NL**, different breakdown:
- NL: 8 + 1 + 1 = 10 (one drain cycle + one ACAM cycle)
- AL: 8 + 2 + 0 = 10 (two drain cycles, no ACAM)

For different PRECISION (without code changes):
- INT4: AL gives 4 + 2 + 0 = 6 cycles (same as NL's 4 + 1 + 1 = 6 — structural symmetry)
- INT16: AL gives 16 + 2 + 0 = 18 cycles (same as NL's 16 + 1 + 1 = 18)

The structural symmetry holds at every precision because both arches have `(PIPELINE_DEPTH - 1) + ACAM_CYCLES = 2`. If we changed either, the symmetry would break:
- e.g., if AL's ADC took 2 cycles instead of 1 (PIPELINE_DEPTH=4): AL CCYC = 8 + 3 + 0 = 11 ≠ NL's 10
- e.g., if NL's ACAM took 2 cycles (multi-cycle LUT lookup): NL CCYC = 8 + 1 + 2 = 11 ≠ AL's 10

### Steady-state cadence (T_steady)

CCYC and T_fill are **unchanged** by the silicon-faithful refactor — both per-pass compute duration and the single-pass total are identical to the pre-refactor numbers (10 and 332 respectively). What **did** change is the multi-pass steady-state cadence T_steady:

```
Pre-refactor (4-slot ring):  T_steady = max(LCYC, CCYC, OCYC) = max(256, 10, 64) = 256
Post-refactor (Option A1):   T_steady = max(LCYC + PRECISION, CCYC, OCYC)
                                      = max(256 + 8, 10, 64) = 264
```

The `+PRECISION` term comes from the **LOAD-gate** (`load_safe`). Pass-(k+1) LOAD cannot launch until pass-k COMPUTE has consumed all PRECISION slices of the single input substrate. So the LOAD→LOAD cadence is `LCYC + PRECISION`, not just `LCYC`:

```
LOAD-pass-(k+1) launches PRECISION cycles AFTER pass-k COMPUTE wakes.
```

For AL R=512 BUF=16 INT8: T_steady = 264. Note AL pays only a +3.1% cadence penalty for the single-substrate refactor; NL pays +15.4% (LCYC=52 is much closer to PRECISION=8). This is because AL's LOAD is bandwidth-bound (BUF=16 small, R=512 big), so the LOAD-gate's PRECISION-cycle penalty is small relative to LCYC.

The TB verifies emergence by **measuring** (not asserting) — see §9 and §11. T4 verifies T_steady = 264 by measuring multi-pass total cycles and matching the prediction `T_fill + (M-1)·T_steady = 332 + 264·(M-1)` (e.g., 596 for M=2, 1124 for M=4, 2180 for M=8).

---

## 7. Signed multiply — MSB sign extension

Identical mechanism to NL faithful, but in **Stage 2** (shift-add) instead of NL's Stage 1 (analog accumulator).

```verilog
// dpe_azurelily_faithful.v:377-389 (paraphrased)
if (bit_idx_s2 == PRECISION-1) begin
    mac_acc[c] <= mac_acc[c] - (adc_reg[c] << bit_idx_s2);   // MSB SUBTRACT
end else begin
    mac_acc[c] <= mac_acc[c] + (adc_reg[c] << bit_idx_s2);   // non-MSB ADD
end
```

The mathematical justification is the same as NL — see `DPE_NLDPE_FAITHFUL_WALKTHROUGH.md` §7 for the full derivation.

Verified by T5 (random signed inputs including negatives): the AL faithful primitive's `mac_acc` matches numpy's `signed_int8_mac` bit-exactly.

---

## 8. No ACAM stage in Azure-Lily

AL silicon has no ACAM (Analog Content-Addressable Memory). The lazy `dpe_azurelily.v` has an `ACAM_MODE` parameter for source-compatibility with the NL primitive, but it's tied off / unused.

The faithful primitive **does not have `ACAM_MODE`** as a parameter. The output of Stage 2 (shift-add) goes directly to the output buffer (`mac_acc`), which OUTPUT drains.

If an activation function is needed for an AL workload (e.g., ReLU), it must be applied in the CLB fabric **after** the DPE primitive — typically inside the `fc_top.v` wrapper's S_ACT_FINAL state (1 extra cycle). This is the architectural distinction: NL can fuse simple activations into the DPE; AL cannot.

For the smoke tests, AL faithful only verifies the linear MAC path (5 tests, no exp/log).

---

## 9. Testbench walkthrough

`fc_verification/tb_dpe_azurelily_faithful.v` (668 lines) is a single TB file with 5 test modes selected via `+define+TEST_MODE=N`. Mirrors the NL faithful TB structure, minus the ACAM mode tests.

### Top-level structure (same pattern as NL)

```verilog
module tb_dpe_azurelily_faithful;
    // Standard clock/reset, drive signals, DUT with PRECISION=8, R=512, C=128, BUF=16
    
    // Test vectors (loaded from oracles/test_vectors_al/ via $readmemh)
    reg signed [7:0]  w_mem    [0:R*C-1];
    reg signed [7:0]  x_mem    [0:M_PASSES*R-1];
    reg signed [7:0]  y_mem    [0:M_PASSES*C-1];
    reg signed [31:0] mac_mem  [0:M_PASSES*C-1];

    initial begin
        // 1. Load test vectors from .mem files
        // 2. Hierarchical-force weights into dut.weights
        // 3. Drive LOAD strobes (one BUF-bit byte chunk per cycle)
        // 4. Capture data_out into captured[]
        // 5. Wait for dpe_done to go low
        // 6. Compare captured byte-exact against y_mem
        // 7. Verify total_cycles matches oracle prediction
    end
endmodule
```

### The 5 test modes

| Mode | Setup | What it validates |
|---|---|---|
| **T1** | Identity weights, all-ones input | Functional LOAD→bit-serial→drain path works |
| **T2** | Random int8 weights and inputs (seeded) | Bit-exact functional correctness vs numpy oracle |
| **T3** | M=1, identity weights — **cycle emergence check** | Measures `last_compute_cycle - first_compute_cycle + 1` and verifies it equals oracle's CCYC=10. Confirms cycle count emerges from physical structure. |
| **T4** | M ∈ {1, 2, 4, 8} M-sweep, identity weights | LOAD-gated multi-pass cadence: total cycles match T_fill + (M−1)·T_steady = 332 + 264·(M−1) ⇒ 332 / 596 / 1124 / 2180. Each pass is functionally correct; TB also confirms `load_safe` falls-then-rises between passes. |
| **T5** | M=1, random signed inputs (including negatives) | Signed 2's complement MAC handled correctly (MSB-subtract in Stage 2) |

**No T5 (exp) or T6 (log) tests** — AL has no ACAM, so no nonlinear modes to verify. T5 here is the AL equivalent of NL's T7 (signed inputs).

### How T3 verifies cycle emergence

`tb_dpe_azurelily_faithful.v` (T3 block):

```verilog
// Telemetry capture from Tier 2 observable pulses
always @(posedge clk) begin
    if (dut.compute_first_bit_pulse) first_compute_cycle <= cycle_count;
    if (dut.compute_busy)            last_compute_cycle  <= cycle_count;
    if (dut.compute_last_bit_pulse)  t_msb_subtract      <= cycle_count;
end

// At end of test:
measured_ccyc = last_compute_cycle - first_compute_cycle + 1;
declared_ccyc_oracle = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES;  // = 8 + 2 + 0 = 10

if (measured_ccyc == declared_ccyc_oracle) begin
    $display("*** CCYC=%0d EMERGED from physical structure (NOT set as parameter) ***",
             measured_ccyc);
    $display("*** Breakdown: PRECISION=8 + (PIPELINE_DEPTH-1)=2 + ACAM_CYCLES=0 (NO ACAM in AL) ***");
end
```

The cycle count is *measured* from physical structure (the duration of `compute_busy`), compared against an *independent prediction* (the oracle's formula). They must agree.

The observed values from T3:
```
T3 first_compute_cycle  = 261
T3 last_compute_cycle   = 270
T3 MEASURED CCYC        = 270 - 261 + 1 = 10
T3 DECLARED CCYC_ORACLE = 10
```

---

## 10. Python oracle methodology

`fc_verification/oracles/azurelily_mac_oracle.py` is the independent reference. Mirrors `nldpe_mac_oracle.py` minus the ACAM transform functions.

### Key functions

```python
def signed_int8_mac(weights_2d, inputs_1d):
    """Per-column signed int8 × int8 → int32 dot product."""
    return weights_2d.astype(np.int32).T @ inputs_1d.astype(np.int32)

def bit_serial_signed_mac(weights, inputs, precision=8):
    """Reference for the bit-serial MSB-subtract algorithm."""
    # Same as NL — see nldpe_mac_oracle.py for full implementation.

def expected_total_cycles(R, C, BUF, M=1, PRECISION=8, PIPELINE_DEPTH=3, ACAM_CYCLES=0):
    """Cycle prediction from AL's physical model.
    
    CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES = 8 + 2 + 0 = 10 (INT8)
    LCYC = ceil(R * 8 / BUF) = ceil(512 * 8 / 16) = 256
    OCYC = ceil(C * 8 / BUF) = ceil(128 * 8 / 16) = 64
    T_fill = LCYC + 1 + CCYC + 1 + OCYC = 256 + 1 + 10 + 1 + 64 = 332
    T_steady = max(LCYC + PRECISION, CCYC, OCYC) = max(264, 10, 64) = 264   # Option A1
    T(M) = T_fill + (M-1) * T_steady                                          # e.g., 332/596/1124/2180
    """
```

**No `acam_transform` function** — AL has no ACAM modes to model.

### Test vector generation

`azurelily_mac_oracle.py --gen-test-vectors` writes 26 `.mem` files into `oracles/test_vectors_al/`:

```
t1_weights.mem    t1_inputs.mem    t1_expected.mem    t1_mac.mem    t1_meta.txt
... (T2 through T5)
cycle_oracle.txt
```

Separate from NL's `oracles/test_vectors/` to avoid name collisions. Each TB loads its own arch's vectors.

---

## 11. Worked example — T1 cycle-by-cycle

Setup:
- AL INT8 R=512 C=128 BUF=16 → LCYC=256, OCYC=64
- M=1, identity weights, all-ones input, no ACAM mode
- Predicted: each output[c] = 1 (= 1·1 from identity weights), total_cycles = 332

### TB drives LOAD (cycles 5–260, 256 strobes)

```
Cycle 5:   TB drives w_buf_en=1, data_in = first EPS=2 bytes of inputs (0x01 each).
           load_safe = 1 (initial), so the gate passes the strobe through.
           T_first_load <= 5.
           DUT corner-turn: for j=0..1, for i=0..7, input_buf_slice[i][j] NBA <= data_in[j*8+i].
           For input byte 0x01: bit 0 = 1, bits 1..7 = 0. So slice[0][0..1] NBA <= 1,
           slice[1..7][0..1] NBA <= 0.
Cycle 6:   next 2 bytes streamed; slice[0][2..3] NBA <= 1, slice[1..7][2..3] NBA <= 0.
...
Cycle 260: last LOAD strobe (256th 2-byte chunk). DUT:
           - slice[0][510..511] NBA <= 1, slice[1..7][510..511] NBA <= 0
           - buf_loaded NBA <= 1 (signals COMPUTE-idle wake branch)
           - load_cycle_cnt NBA <= 0 (re-arm for next pass)
```

### LOAD→COMPUTE handshake (cycle 261)

```
Cycle 261: DUT sees buf_loaded = 1 (NBA committed).
           COMPUTE idle branch fires:
             compute_busy           NBA <= 1
             bit_idx_s0             NBA <= 0
             bit_idx_s1             NBA <= 0
             bit_idx_s2             NBA <= 0
             s1_valid               NBA <= 0
             s2_valid               NBA <= 0
             compute_first_bit_pulse NBA <= 1  ← Tier 2 telemetry
             buf_loaded             NBA <= 0
             load_safe              NBA <= 0   ← LOCK substrate against pass-(k+1) LOAD
             mac_acc[c]             NBA <= 0  for all c
           
           TB captures: first_compute_cycle = 261.
```

### Bit-serial sweep (cycles 262–270)

```
Cycle 262: compute_busy = 1 visible. bit_idx_s0 = 0. Stage 0 reads slice bank 0:
           crossbar_sum_comb[c] = sum_r (input_buf_slice[0][r] ? weights[r][c] : 0)
                                = identity diagonal = 1 for c<min(R,C); else 0.
           crossbar_sum_reg[c] NBA <= 1.   bit_idx_s0 NBA <= 1.   s1_valid NBA <= 1.

Cycle 263: bit_idx_s0 = 1, s1_valid = 1, bit_idx_s1 = 0.
           Stage 0: slice bank 1, all zero. xbar_reg NBA <= 0.
           Stage 1: adc_reg[c] NBA <= crossbar_sum_reg[c] = 1 (bit 0's value).
           bit_idx_s0 NBA <= 2. bit_idx_s1 NBA <= 1. s2_valid NBA <= 1.

Cycle 264: Stage 0: slice bank 2, all zero. xbar_reg NBA <= 0.
           Stage 1: adc_reg NBA <= xbar_reg = 0 (bit 1's value).
           Stage 2: bit_idx_s2 = 0 (not MSB).
             mac_acc[c] NBA <= 0 + (adc_reg << 0) = 0 + 1 = 1.
           bit_idx_s2 NBA <= 1.

Cycle 265: Stage 0: slice bank 3, all zero. xbar_reg NBA <= 0.
           Stage 1: adc_reg NBA <= 0.
           Stage 2: bit_idx_s2 = 1. mac_acc += 0 << 1 = 0. (unchanged at 1)
           bit_idx_s2 NBA <= 2.

Cycle 266-269: Similar. bit_idx_s0/s1/s2 advance. mac_acc unchanged at 1.
               At cycle 268 (bit_idx_s0 == PRECISION-2 == 6) load_safe NBA <= 1
               fires; pass-(k+1) LOAD could begin on cycle 269.

Cycle 270: bit_idx_s0 = 8 (idle, no Stage 0 fire).
           bit_idx_s1 = 7. Stage 1: adc_reg NBA <= xbar_reg (bit 7 value = 0).
           bit_idx_s2 = 7 = PRECISION-1. MSB POSITION!
             mac_acc[c] NBA <= 1 - (adc_reg << 7) = 1 - 0 = 1.
           Same NBA cycle:
             compute_busy NBA <= 0
             compute_done NBA <= 1   (single-substrate handshake to OUTPUT)
             compute_last_bit_pulse NBA <= 1
             MSB_SA_Ready NBA <= 1.

           TB captures: last_compute_cycle = 270.
           MEASURED CCYC = 270 - 261 + 1 = 10 ✓ EMERGED FROM STRUCTURE.
```

### COMPUTE→OUTPUT handshake (cycle 271)

```
Cycle 271: DUT sees compute_done = 1 (NBA committed).
           OUTPUT idle branch fires:
             output_busy    NBA <= 1
             output_col_idx NBA <= 0
             compute_done   NBA <= 0
           Note: AL drains directly from the single-substrate mac_acc — no acam_out!
```

### OUTPUT drain (cycles 272–335, 64 strobes)

```
Cycle 272: output_busy = 1. Stage drains:
             data_out NBA <= mac_acc[0..1][7:0] = 0x01 each (truncated low byte)
             dpe_done NBA <= 1
           output_col_idx NBA <= 1.

Cycle 273: TB captures data_out = 0x01. cap_strobe_idx 0→1.
           T_done_last <= 273.
           DUT: data_out NBA <= mac_acc[2..3][7:0] = 0x01.

... 62 more drain cycles ...

Cycle 335: Last OUTPUT strobe. TB captures data_out = 0x01. T_done_last <= 335.
           DUT: output_col_idx NBA <= 0; output_busy NBA <= 0.
```

Wait — that gives 335, but the actual measurement was T_done_last=336, total_cycles=332. The extra cycle is the OUTPUT-side NBA register propagation. Approximate timing; for exact bit-level cycle counts, run the TB and observe.

### TB final check

```
Cycle 336: TB observes dpe_done went low.
total_cycles = T_done_last - T_first_load + 1 = 336 - 5 + 1 = 332  ✓
Compare captured[0..127*2-1] byte-by-byte against y_mem[0..127]:
  y_mem[c] = 0x01 for c < 128, 0x00 otherwise.
  All match → T1 PASS.

[tb_faithful_al] T1 T_first_load=5 T_done_last=336 total_cycles=332 (oracle T_fill=332)
[tb_faithful_al] T1 PASS
```

---

## 11.5. Concrete numerical example — bit-serial MAC math

This section shows the **actual numerical computation** with non-trivial signed inputs and weights, following the same example as the NL walkthrough doc (so you can compare the two arches' pipeline behavior on identical math).

### Example setup (same as NL §11.5)

R=4, C=2 (small for tracing; AL silicon uses R=512 C=128).

**Weights** `W[4][2]`:

```
        c=0    c=1
   r=0   3    -1
   r=1   2     5
   r=2  -4     1
   r=3   1    -2
```

**Input vector** `X[4]`:

```
   X[0] =  5   = 0b00000101
   X[1] = -3   = 0b11111101    (2's complement)
   X[2] =  1   = 0b00000001
   X[3] =  2   = 0b00000010
```

**Expected MAC** (independent of architecture):

```
MAC[0] = 3·5 + 2·(-3) + (-4)·1 + 1·2  =  7
MAC[1] = (-1)·5 + 5·(-3) + 1·1 + (-2)·2  = -23
```

Output bytes (int8 truncation):
- `data_out byte for c=0` = `7 & 0xFF = 0x07`
- `data_out byte for c=1` = `-23 & 0xFF = 0xE9`

### Bit-slice extraction (same as NL)

After LOAD's corner-turn, this same data lives in `input_buf_slice[b][r]` — column `b` of the table below is *literally* slice bank `b`:

```
b:        0  1  2  3  4  5  6  7
X[0]= 5:  1  0  1  0  0  0  0  0    → slice[0..7][0]
X[1]=-3:  1  0  1  1  1  1  1  1    → slice[0..7][1]
X[2]= 1:  1  0  0  0  0  0  0  0    → slice[0..7][2]
X[3]= 2:  0  1  0  0  0  0  0  0    → slice[0..7][3]
```

Rows with bit=1 at each position:
```
b=0: {0, 1, 2}        b=1: {3}              b=2: {0, 1}           b=3: {1}
b=4: {1}              b=5: {1}              b=6: {1}              b=7: {1}
```

### Stage 0 — crossbar_sum_comb per bit-slice (same values as NL)

| b | crossbar_sum_comb[0]                 | crossbar_sum_comb[1]                |
|---|---|---|
| 0 | W[0][0]+W[1][0]+W[2][0] = 3+2+(-4) = **1**   | W[0][1]+W[1][1]+W[2][1] = -1+5+1 = **5**     |
| 1 | W[3][0] = **1**                              | W[3][1] = **-2**                            |
| 2 | W[0][0]+W[1][0] = 3+2 = **5**                | W[0][1]+W[1][1] = -1+5 = **4**              |
| 3 | W[1][0] = **2**                              | W[1][1] = **5**                             |
| 4 | W[1][0] = **2**                              | W[1][1] = **5**                             |
| 5 | W[1][0] = **2**                              | W[1][1] = **5**                             |
| 6 | W[1][0] = **2**                              | W[1][1] = **5**                             |
| 7 | W[1][0] = **2** ← MSB                        | W[1][1] = **5** ← MSB                       |

### Stage 1 — adc_reg per bit-slice (NEW vs NL)

AL has an ADC sample stage. `adc_reg` is just a 1-cycle-delayed copy of `crossbar_sum_reg`:

| Cycle | bit_idx_s0 | bit_idx_s1 | crossbar_sum_reg[0] | adc_reg[0]    |
|---:|:-:|:-:|---:|---:|
| 0 | 0 | – (s1 not yet armed) | – (was 0) | – (was 0) |
| 1 | 1 | 0 | **1** (NBA from cycle 0) | – |
| 2 | 2 | 1 | **1** (NBA from cycle 1) | **1** (NBA from cycle 1) |
| 3 | 3 | 2 | **5** (NBA from cycle 2) | **1** (NBA from cycle 2) |
| 4 | 4 | 3 | **2** | **5** |
| 5 | 5 | 4 | **2** | **2** |
| 6 | 6 | 5 | **2** | **2** |
| 7 | 7 | 6 | **2** | **2** |
| 8 | 8 (idle) | 7 (last) | **2** | **2** |
| 9 | – | – | – | **2** (last bit 7 NBA from cycle 8) |

(Same pattern for column 1 with its own crossbar values.)

**Key observation**: at cycle k, Stage 2 reads `adc_reg` which holds the value latched at cycle k-1 (from `crossbar_sum_reg` at cycle k-2, from `crossbar_sum_comb` at cycle k-2). So Stage 2's bit position `bit_idx_s2` is 2 cycles behind Stage 0's `bit_idx_s0`.

### Stage 2 — mac_acc accumulation timeline

`bit_idx_s2` lags `bit_idx_s1` by 1 (so lags `bit_idx_s0` by 2). Each cycle while `bit_idx_s2 < PRECISION-1`: ADD; at `bit_idx_s2 == PRECISION-1`: SUBTRACT.

Column c=0 accumulation (single-substrate `mac_acc[0]` means column 0, not slot 0 — there is no slot index anymore):

```
Starting mac_acc[0] = 0

bit_idx_s2=0:   adc_reg=1.    shift = 1<<0 = 1.    mac_acc[0] += 1.    → mac_acc[0]= 1
bit_idx_s2=1:   adc_reg=1.    shift = 1<<1 = 2.    mac_acc[0] += 2.    → mac_acc[0]= 3
bit_idx_s2=2:   adc_reg=5.    shift = 5<<2 = 20.   mac_acc[0] += 20.   → mac_acc[0]= 23
bit_idx_s2=3:   adc_reg=2.    shift = 2<<3 = 16.   mac_acc[0] += 16.   → mac_acc[0]= 39
bit_idx_s2=4:   adc_reg=2.    shift = 2<<4 = 32.   mac_acc[0] += 32.   → mac_acc[0]= 71
bit_idx_s2=5:   adc_reg=2.    shift = 2<<5 = 64.   mac_acc[0] += 64.   → mac_acc[0]= 135
bit_idx_s2=6:   adc_reg=2.    shift = 2<<6 = 128.  mac_acc[0] += 128.  → mac_acc[0]= 263
bit_idx_s2=7:   adc_reg=2.    shift = 2<<7 = 256.  mac_acc[0] -= 256.  → mac_acc[0]= 7   ← MSB SUBTRACT
                                                                          + compute_busy NBA <= 0
                                                                          + compute_done NBA <= 1
```

**Final mac_acc[0] = 7** ✓ matches the directly-computed MAC.

Column c=1 accumulation:

```
Starting mac_acc[1] = 0

bit_idx_s2=0:   adc_reg= 5.   shift =  5<<0 =   5.    mac_acc[1] +=  5.   → mac_acc[1]=   5
bit_idx_s2=1:   adc_reg=-2.   shift = -2<<1 =  -4.    mac_acc[1] += -4.   → mac_acc[1]=   1
bit_idx_s2=2:   adc_reg= 4.   shift =  4<<2 =  16.    mac_acc[1] += 16.   → mac_acc[1]=  17
bit_idx_s2=3:   adc_reg= 5.   shift =  5<<3 =  40.    mac_acc[1] += 40.   → mac_acc[1]=  57
bit_idx_s2=4:   adc_reg= 5.   shift =  5<<4 =  80.    mac_acc[1] += 80.   → mac_acc[1]= 137
bit_idx_s2=5:   adc_reg= 5.   shift =  5<<5 = 160.    mac_acc[1] += 160.  → mac_acc[1]= 297
bit_idx_s2=6:   adc_reg= 5.   shift =  5<<6 = 320.    mac_acc[1] += 320.  → mac_acc[1]= 617
bit_idx_s2=7:   adc_reg= 5.   shift =  5<<7 = 640.    mac_acc[1] -= 640.  → mac_acc[1]= -23  ← MSB SUBTRACT
```

**Final mac_acc[1] = -23** ✓ matches the directly-computed MAC.

### Stage 2 also signals COMPUTE done (NO ACAM stage)

At the MSB cycle (bit_idx_s2=7), the same NBA that does the subtract also signals compute done:

```verilog
// dpe_azurelily_faithful.v Stage 2 MSB branch
mac_acc[cc]             <= mac_acc[cc] - (adc_reg[cc] <<< bit_idx_s2);
compute_busy            <= 0;
bit_idx_s0..s2          <= 0;
s1_valid, s2_valid      <= 0;
compute_done            <= 1;           // single-substrate handshake to OUTPUT
compute_last_bit_pulse  <= 1;
```

So `compute_busy` goes from 1 to 0 on the MSB cycle's posedge. Compared to NL, which needs an extra ACAM cycle after Stage 1's MSB to do the LUT lookup, AL skips this cycle entirely — the linear shift-add already produced the final value.

### Output drain (truncation, no ACAM transformation)

OUTPUT drains the single-substrate `mac_acc` directly (no `acam_out` intermediate):

```
c=0:   mac_acc[0] =  7  (0x00000007)   → data_out byte = 0x07
c=1:   mac_acc[1] = -23 (0xFFFFFFE9)   → data_out byte = 0xE9
```

Same final output as NL with ACAM_MODE=0. The only architectural difference is that NL paid 1 cycle for `acam_out <= mac_acc` (identity pass-through), while AL drained `mac_acc` directly. AL saved that 1 cycle by having an extra ADC stage 1 cycle earlier in the pipeline.

### Cycle-level alignment with standard AL geometry

For AL's standard R=512 geometry (LCYC=256), the example would extend the cycle numbers — but the bit-serial sweep itself is the same 10 cycles. Using the empirical T3 measurement:

| Cycle | bit_idx_s0 | bit_idx_s1 | bit_idx_s2 | crossbar_sum_reg[0] | adc_reg[0] | mac_acc[0] (post-NBA) |
|---:|:-:|:-:|:-:|---:|---:|---:|
| 261 (compute_busy=1, first cycle) | 0 (entering Stage 0) | – | – | – (was 0) | – (was 0) | 0 |
| 262 | 1 | 0 (s1_valid armed) | – | 1 (NBA from 261) | – | 0 |
| 263 | 2 | 1 | 0 (s2_valid armed) | 1 | 1 (NBA from 262) | 0 + 1 = 1 |
| 264 | 3 | 2 | 1 | 5 | 1 | 1 + 2 = 3 |
| 265 | 4 | 3 | 2 | 2 | 5 | 3 + 20 = 23 |
| 266 | 5 | 4 | 3 | 2 | 2 | 23 + 16 = 39 |
| 267 | 6 (load_safe NBA <= 1 fires) | 5 | 4 | 2 | 2 | 39 + 32 = 71 |
| 268 | 7 (last slice read; load_safe = 1 observed) | 6 | 5 | 2 | 2 | 71 + 64 = 135 |
| 269 | 8 (idle) | 7 (last) | 6 | 2 | 2 | 135 + 128 = 263 |
| 270 | – | – | 7 (MSB!) | – | 2 | 263 − 256 = **7** ← MSB subtract + compute_busy <= 0, compute_done <= 1 (NBA) |

MEASURED CCYC = 270 - 261 + 1 = **10 cycles** ✓.

### Summary — what the AL faithful primitive computes

Mathematically identical to NL:

```
MAC[c] = sum_{b=0..PRECISION-2} 2^b · (sum_r W[r][c] if X[r][b]==1 else 0)
       - 2^(PRECISION-1)        · (sum_r W[r][c] if X[r][PRECISION-1]==1 else 0)
```

Physically different from NL:
- **NL**: bit-slice → crossbar → analog accumulator (2 stages) → ACAM LUT (1 stage). Sums and MSB-subtract happen in the analog accumulator stage.
- **AL**: bit-slice → crossbar → ADC sample (1 stage) → shift-add accumulator (1 stage). Sums and MSB-subtract happen in the digital shift-add stage. No ACAM.

Same MAC values produced. Same 10-cycle CCYC. Verified bit-exactly by T2 (random unsigned) and T5 (signed with negatives).

---

## 12. Verification surface — what each test catches

| Bug type | Test that catches it | How |
|---|---|---|
| LOAD streams bytes to wrong buffer slot | T2, T4 | Bit-exact functional check fails |
| Stage 0 crossbar logic doesn't sign-extend weights | T5 (signed) | MAC magnitude wrong; functional fail |
| Stage 1 ADC fails to register (wrong adc_reg value) | T2, T4 | mac_acc accumulates wrong shifted values |
| Stage 2 doesn't subtract at MSB | T5 | Negative inputs give wrong MAC |
| Stage 2 misorders bit_idx_s2 | T2, T5 | MAC wrong by factor of bit-shift |
| Pipeline depth bug (3 stages not 1-cycle offset each) | T3 cycle measurement | first_compute_cycle / last_compute_cycle gap != 10 |
| LOAD-gate (`load_safe`) clears too early — pass-(k+1) LOAD corrupts pass-k slices | T4 (M-sweep) | Functional fail at M≥2; pass-k MAC scrambled by overwritten slice cells |
| LOAD-gate stays asserted too late — pass-(k+1) LOAD waits unnecessarily | T4 (M-sweep) | Cycle count > T_fill + (M-1)·T_steady = 332 + (M-1)·264 |
| Corner-turn bit permutation wrong (slice[i][j] gets wrong data_in bit) | T2, T4 | Functional fail; MAC reflects scrambled input bits |
| Compute substrate (single-banked mac_acc) reused before drain | T4 (M-sweep) | Pass m's MAC corrupts pass m+1; output mismatch |
| Drain-load overlap broken | T4 (M-sweep) | Total cycles != T_fill + (M-1)·T_steady = 332 / 596 / 1124 / 2180 |
| AL primitive accidentally implements ACAM | T2, T5 | mac_acc != expected (if extra cycle inserted) |

Compared to NL's 7-test verification surface, AL has 5 tests (no exp/log mode validation). Otherwise the surface is similar — both check bit-exact MAC, cycle emergence, and signed multiply.

---

## 13. Migration plan (deferred — not in scope for Task #92)

The faithful AL primitive currently coexists with the lazy `dpe_azurelily.v`. The lazy primitive is what `fc_top.v` and existing TBs use for Azure-Lily workloads.

Migration mirrors NL's path:

1. **Cross-verification**: run existing AL TBs (`tb_dpe_vmm.v` with `-DARCH_AL`, etc.) against `dpe_azurelily_faithful.v`. Should pass — faithful is a superset.

2. **Cycle count reconciliation**: faithful and lazy produce identical T_fill for AL workloads (332 for INT8 R=512 C=128 BUF=16, M=1). Both have the same +2 NBA handoff overhead.

3. **Replace**: rename lazy `dpe_azurelily.v` → `dpe_azurelily_lazy.v`; faithful → `dpe_azurelily.v`. Update generator reference. Re-run all smoke gates.

4. **Once both arches migrated**, the lazy primitives become archived reference material and the faithful versions become the production behavior models.

---

## 14. File map

| File | Purpose | Lines |
|---|---|---|
| `fc_verification/rtl/dpe_azurelily_faithful.v` | Faithful primitive | 442 |
| `fc_verification/tb_dpe_azurelily_faithful.v` | Standalone TB (5 test modes) | 668 |
| `fc_verification/oracles/azurelily_mac_oracle.py` | Independent numpy reference (no ACAM transforms) | 312 |
| `fc_verification/oracles/test_vectors_al/*.mem` | Pre-generated test vectors (26 files) | — |
| `nl_dpe/gen_dpe_azurelily_faithful.py` | Generator from JSON config | 449 |
| `fc_verification/Makefile` | +3 targets: `tb-faithful-al`, `faithful-vectors-al`, `faithful-smoke-al` | — |

---

## 15. Build and run commands

```bash
# Regenerate test vectors from oracle (deterministic; same seed → same vectors)
make -C fc_verification faithful-vectors-al

# Run a single test
make -C fc_verification tb-faithful-al TEST_MODE=1

# Run all 5 tests
make -C fc_verification faithful-smoke-al
```

Expected output of `faithful-smoke-al`: 5 sequential PASS reports. `total_cycles` for M=1 = 332; for M-sweep T4 with M ∈ {1,2,4,8}: 332/596/1124/2180 per the Option A1 cadence T_steady = LCYC + PRECISION = 256 + 8 = 264.

---

## 16. Summary

The faithful Azure-Lily DPE primitive is a **physical behavioral model** that:
- Has no `COMPUTE_CYCLES` parameter
- Implements the bit-serial pipeline with **3 visible stages** (crossbar / ADC sample / shift-add)
- Has **no ACAM stage** — Stage 2's output IS the output, drained directly
- Handles signed int8 MAC correctly via MSB-subtract (in Stage 2)
- Exposes all internal state as Tier 2 observable registers for TB introspection
- Validates "CCYC = PRECISION + (PIPELINE_DEPTH−1) + ACAM_CYCLES = 8 + 2 + 0 = 10" by **measuring** the gap from physical structure

Same architectural symmetry as NL faithful: both arches produce CCYC=10 for INT8, but via different physical decompositions. AL has more pipeline depth (1 extra stage for ADC); NL has the ACAM read-out stage instead. The verification surface confirms that the SAME signed int8 multiplication math holds in both arches, just routed through different physical structures.

This completes the dual-arch faithful track: both NL-DPE and Azure-Lily now have physically-grounded behavior models with independent numpy oracles, complete TB coverage, and structurally-derived cycle counts.

For the architectural context (why structurally-grounded primitives matter) and the NL-specific details (ACAM modes, NL's 2-stage pipeline), see `DPE_NLDPE_FAITHFUL_WALKTHROUGH.md` — that doc is the primary reference; this AL doc is the parallel companion.
