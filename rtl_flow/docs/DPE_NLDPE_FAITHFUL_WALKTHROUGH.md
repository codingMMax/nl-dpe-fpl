# NL-DPE Faithful Primitive — Implementation Walkthrough

**Companion doc for verifying Task #91: the faithful (physically-modeled) NL-DPE primitive.**

This doc walks through the implementation of `fc_verification/rtl/dpe_nldpe_faithful.v` — the **physical** behavioral model of the NL-DPE primitive — alongside its testbench `tb_dpe_nldpe_faithful.v`, Python oracle `oracles/nldpe_mac_oracle.py`, and generator `nl_dpe/gen_dpe_nldpe_faithful.py`.

The faithful primitive **inverts the validation relationship** with the §4 methodology formula:

| | Lazy primitive (`dpe_nldpe.v`) | Faithful primitive (`dpe_nldpe_faithful.v`) |
|---|---|---|
| `COMPUTE_CYCLES` parameter | Required: TB sets it to match the formula | **Does not exist** — there is no such parameter anywhere |
| Cycle count comes from… | TB/sim formula tells the RTL how many cycles to burn | RTL computes it by advancing `bit_idx` and ACAM fire registers; cycle count is a **structural consequence** |
| Cross-check value | "RTL agrees with the formula" (tautological — both encode the formula) | "Measured CCYC equals what the formula predicts" (independent — formula is a *prediction*, RTL is a *measurement*) |
| ACAM modes | Combinational transform on the full MAC, gated by `ACAM_MODE != 0` | **Physical 1-cycle stage**, fires regardless of mode (mode 0 = ADC identity passthrough) |
| Pipeline visibility | Single COMPUTE counter (opaque) | Per-stage registers `bit_idx_s0/s1`, `crossbar_sum_reg`, `mac_acc`, `acam_out` — **all observable** by TB hierarchical force |

Read this doc if you need to understand:
- How the bit-serial pipeline is implemented (§4, §5)
- Why CCYC = 10 for INT8 emerges from the code structure (§6)
- How signed int8 × int8 multiplication works bit-serially (§7)
- How ACAM modes 0/1/2 are implemented (§8)
- How the testbench validates emergence (§9, §11)
- How the Python oracle is independent of both sim and RTL formulas (§10)

For the lazy primitive design and the original 4-slot ring-buffer pattern, see `DPE_PRIMITIVE_WALKTHROUGH.md`. The faithful primitive has since been refactored to a **silicon-faithful single-substrate** design ("Option A1"): the 4-slot pass-tagged ring buffer (`input_buffer[QDEPTH=4][R]`, `mac_acc[QDEPTH][C]`, `acam_out[QDEPTH][C]`) has been collapsed to a single bank of each substrate, and a LOAD-gate (`load_safe`) interlocks pass-(k+1) LOAD against pass-k COMPUTE's slice-read window. LOAD now performs a **corner-turn** that rearranges byte-major BRAM input into a **slice-major** bit substrate `input_buf_slice[PRECISION][R]`. See §2 for the storage layout, §3a for the corner-turn LOAD, and §6 for the cycle implications (T_steady = max(LCYC + PRECISION, CCYC, OCYC) = 60).

---

## 1. Module interface

```verilog
// dpe_nldpe_faithful.v:95-119
module dpe #(    // module name "dpe" matches VTR <model name="dpe"> contract
    parameter KERNEL_WIDTH   = 256,   // R (rows)
    parameter NUM_COLS       = 256,   // C (cols)
    parameter DPE_BUF_WIDTH  = 40,    // BUF (bits per byte-stream lane)
    parameter PRECISION      = 8,     // bit precision (INT8 default)
    parameter PIPELINE_DEPTH = 2,     // (Crossbar, Acc) -> 2-stage internal pipeline
    parameter ACAM_CYCLES    = 1,     // ACAM read-out latency (always 1)
    parameter ACAM_MODE      = 0      // 0=ADC/identity, 1=exp, 2=log
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
```

**Port list is byte-identical to lazy** — the VTR arch XML `<model name="dpe">` contract is preserved (same 9 inputs + 6 outputs). The faithful primitive is a drop-in replacement at the port boundary.

**The new parameters** vs lazy:
- `PIPELINE_DEPTH = 2`: declared, used in code as the bit-serial pipeline depth (Stage 0 + Stage 1). Not a free knob — changing it would require code changes.
- `ACAM_CYCLES = 1`: declared, used as the ACAM stage latency. Always 1 for NL-DPE.
- `PRECISION = 8`: the bit width. **INT8 only** in this implementation; INT4/INT16 are deferred work.

**No `COMPUTE_CYCLES` parameter**. The lazy primitive's `COMPUTE_CYCLES` is gone — cycle count is not a free parameter anymore; it emerges from the code.

---

## 2. Internal storage layout

`dpe_nldpe_faithful.v:109-114`

```verilog
// Weights: pre-loaded by TB hierarchical force into dut.weights[r][c]
reg signed [7:0]  weights      [0:KERNEL_WIDTH-1][0:NUM_COLS-1];

// Single-substrate input buffer in SLICE-MAJOR (bit-stratified) form.
// PRECISION banks of 1-bit cells; each bank holds one bit-slice across all R rows.
reg               input_buf_slice [0:PRECISION-1][0:KERNEL_WIDTH-1];

// Single-substrate Stage 1 accumulator (one int32 per output column).
reg signed [31:0] mac_acc          [0:NUM_COLS-1];

// Stage 0 latched output (single snapshot, shared across bit-slices)
reg signed [31:0] crossbar_sum_reg [0:NUM_COLS-1];

// Single-substrate Stage 2 ACAM output, drained byte-wise by OUTPUT
reg signed [31:0] acam_out         [0:NUM_COLS-1];
```

**This is the silicon-faithful refactor** ("Option A1"): every substrate is **single-banked**. There is no `QDEPTH`, no ring buffer, no pass-tagged slots. Three substrates total (input slice / MAC / ACAM out), matching the three physical regions a real chip would carry.

Five storage elements:

| Storage | Role | Tier 2 observable? |
|---|---|---|
| `weights` | R×C weight matrix (TB hierarchical-forces these) | ✓ `dut.weights[r][c]` |
| `input_buf_slice` | PRECISION × R 1-bit cells; bank `i` holds bit-slice `i` for all rows | ✓ `dut.input_buf_slice[i][r]` |
| `mac_acc` | Single int32 signed accumulator per output column; Stage 1 builds the MAC bit by bit | ✓ `dut.mac_acc[c]` |
| `crossbar_sum_reg` | Stage 0 output for **current** bit-slice (one snapshot, shared) | ✓ `dut.crossbar_sum_reg[c]` |
| `acam_out` | Single int32 post-ACAM output per column, drained byte-wise | ✓ `dut.acam_out[c]` |

**Slice-major layout vs old byte-major layout**:

```
OLD (pre-refactor, depth-4 ring, byte-major):
   input_buffer[slot=0..3][row=0..R-1]   each cell = 8-bit int
   slot 0:  [byte_row0][byte_row1][byte_row2] ...[byte_row255]
   slot 1:  [byte_row0][byte_row1]            ...[byte_row255]
   ...                                                            (4 slots × R bytes)

NEW (single substrate, slice-major):
   input_buf_slice[bit=0..PRECISION-1][row=0..R-1]   each cell = 1 bit
   bank 0:  bit-slice-0 of [row0][row1][row2]            ...[row255]
   bank 1:  bit-slice-1 of [row0][row1]                  ...[row255]
   ...
   bank 7:  bit-slice-7 of [row0][row1]                  ...[row255]
                                                                  (PRECISION × R bits)
```

The total bit count is identical (R × PRECISION), but the **organisation** is rotated 90°: COMPUTE Stage 0 needs *one bit per row at a single bit-position* on every cycle, so reading down a slice-major bank is one memory plane per cycle (combinational over R rows of the same bank) instead of slicing through 256 byte-words. This matches real-silicon bit-line organisation, where every row contributes one bit on a single global bit-line at a time.

**Why `mac_acc` and `acam_out` are int32**: the user explicitly noted "we don't care about intermediate precision; analog signals fundamentally". int32 has plenty of headroom for int8 × int8 × 256 dot product (max magnitude ≈ 2²³, fits in 32 bits with sign). The truncation to int8 happens only at the OUTPUT drain step.

**Why single-substrate** (no QDEPTH ring): the chip has *one* MAC, *one* output register array, *one* input slice substrate. A 4-slot ring buffer would be 4× hardware. The refactor instead uses a **LOAD-gate (`load_safe`)** to serialise pass-(k+1) LOAD against pass-k COMPUTE's slice-read window, so the single substrate is always safe to re-use once COMPUTE has consumed it. See §3 and §6.

**Why `crossbar_sum_reg` is shared**: only ONE pass at a time is in the bit-serial sweep. The Stage 0 latch is a single register array, unchanged by the refactor.

---

## 3. Sub-FSMs — three concurrent stages

The faithful primitive has the same 3 sub-FSMs as the lazy version (LOAD/COMPUTE/OUTPUT), but COMPUTE is fundamentally different — it's a bit-serial pipeline instead of a count-down timer.

### 3a. LOAD sub-FSM — corner-turn into slice substrate, gated by `load_safe`

`dpe_nldpe_faithful.v:208-227`

```verilog
// LOAD sub-FSM (corner-turn into bit-stratified slice).
// Option A1 gate: only act on w_buf_en when load_safe == 1.
// When !load_safe, w_buf_en is IGNORED (TB must wait for
// load_safe to rise before resuming strobes).
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

Two new mechanisms compared to the byte-major byte-by-byte LOAD:

**(1) Corner-turn distribution**. Per LOAD strobe, the BRAM provides `ELEMS_PER_STROBE = 5` bytes (NL has BUF=40 ⇒ 5 bytes/cycle), packed as `data_in[39:0] = {byte4, byte3, byte2, byte1, byte0}`. The corner-turn rearrangement scatters each byte's 8 bits across the 8 slice banks, so that *bit i of byte j of this strobe* ends up in `input_buf_slice[i][load_cycle_cnt*5 + j]`. In one strobe, 40 input bits go to 5 row-positions × 8 bit-banks = 40 cells, one bit per cell. By the last strobe (cycle `LOAD_CYCLES-1 = 51`), the full R × PRECISION substrate is populated.

The fixed wire permutation:
```
slice[i][load_cycle_cnt * 5 + j] <= data_in[j*8 + i]   for i = 0..7, j = 0..4
```
i.e., `data_in[0..7]` lights up bit-banks 0..7 at row offset 0; `data_in[8..15]` lights up bit-banks 0..7 at row offset 1; etc.

**(2) `load_safe` gate**. `w_buf_en` is *gated* by `load_safe`. While COMPUTE is reading from `input_buf_slice` (the pass-k slice-read phase), LOAD must not overwrite the cells — otherwise pass-k's COMPUTE would mis-read mid-slice. The gate is **cleared** when COMPUTE wakes (line 310) and **re-asserted** when COMPUTE's last slice has been read combinationally (lines 243-251, set NBA on `bit_idx_s0 == PRECISION-2` so that load_safe=1 is observed on the cycle of the last slice read). Once `load_safe = 1`, pass-(k+1) LOAD may start writing — overlap with pass-k OUTPUT is fine because OUTPUT drains `acam_out`, a different substrate.

**Critical difference from lazy**: in the lazy primitive, the last LOAD strobe fires the **entire** VMM math combinationally (BLOCKING `=`) in one cycle. The faithful primitive does **NOT** fire any math in LOAD — it just streams bits via the corner-turn (all NBA writes). The math happens bit-serially in COMPUTE.

This means `input_buf_slice` is read by Stage 0 cycle after cycle in COMPUTE, NOT consumed in one tick.

### 3b. COMPUTE sub-FSM (the new piece)

`dpe_nldpe_faithful.v:229-314` — this is the main novelty. See §4-§6 below for the detailed walkthrough.

State variables:
- `compute_busy`: high while COMPUTE is processing a pass
- `bit_idx_s0`: stage-0 bit position (0..PRECISION), advances each cycle
- `bit_idx_s1`: stage-1 bit position (lags s0 by 1 cycle)
- `s1_valid`: armed one cycle after `compute_busy` wakes (so stage 1 reads a freshly-latched `crossbar_sum_reg`)
- `acam_fire`: 1-cycle pulse, armed when Stage 1 commits the MSB subtract

**Single-substrate handshake flags (replace ring-buffer pointers)**:
- `buf_loaded`: 1 when LOAD has just completed a pass; consumed by COMPUTE-idle wake branch (lines 297-313)
- `compute_done`: 1 when COMPUTE has just finished a pass; consumed by OUTPUT-idle wake branch
- `load_safe`: LOAD-gate (Option A1). Cleared when COMPUTE wakes; set when `bit_idx_s0 == PRECISION-2` so it is observed=1 on the cycle of the last slice read (lines 243-251)

There is no `q_load_tail`, `q_compute_head`, or `q_output_head` anymore. The single substrate, plus the three flags above, replaces all of the pass-tagged indexing.

Telemetry pulses (Tier 2 observable):
- `compute_first_bit_pulse`: NBA'd high on the cycle that bit_idx_s0 first transitions out of idle (= the cycle stage 0 latches bit 0)
- `acam_commit_pulse`: NBA'd high on the cycle that ACAM first commits to `acam_out`

### 3c. OUTPUT sub-FSM

`dpe_nldpe_faithful.v:317-341` — same pattern as before, but **drains `acam_out[c]` directly** (single substrate). Chains across passes if `compute_done` is set when OUTPUT finishes the current drain.

---

## 4. The COMPUTE stage — bit-serial pipeline

The COMPUTE sub-FSM is where the faithful primitive earns its name. It has three internal stages, each modeling a physical layer of the NL-DPE silicon:

```
input_buf_slice[bit_idx_s0][row]            ← LOAD corner-turned bits in here
      ↓
[Bit-slice extractor]                       ← combinational, no register
      ↓
[Stage 0: Crossbar fire]                    ← combinational + register (crossbar_sum_reg)
      ↓
[Stage 1: Analog accumulator]               ← registered (mac_acc — single substrate)
      ↓                                       8 cycles of bit-slice accumulation
      ↓                                       + 1 cycle for MSB to drain
      ↓                                       = PRECISION + (PIPELINE_DEPTH-1) cycles
[Stage 2: ACAM]                             ← registered, 1 cycle, mode-dependent
      ↓
acam_out                                    ← single substrate, drained by OUTPUT
```

### Bit-slice extractor (combinational)

`dpe_nldpe_faithful.v:151-164`

```verilog
always @* begin
    for (c_idx = 0; c_idx < NUM_COLS; c_idx = c_idx + 1)
        crossbar_sum_comb[c_idx] = 32'sd0;
    if (compute_busy && bit_idx_s0 < PRECISION[4:0]) begin
        for (r_idx = 0; r_idx < KERNEL_WIDTH; r_idx = r_idx + 1) begin
            if (input_buf_slice[bit_idx_s0][r_idx]) begin   // ← read 1-bit cell directly
                for (c_idx = 0; c_idx < NUM_COLS; c_idx = c_idx + 1) begin
                    // Sign-extend the int8 weight to int32 before adding
                    crossbar_sum_comb[c_idx] = crossbar_sum_comb[c_idx]
                        + {{24{weights[r_idx][c_idx][7]}}, weights[r_idx][c_idx]};
                end
            end
        end
    end
end
```

**What this does**: for the current `bit_idx_s0` value, walk down slice bank `bit_idx_s0` — `input_buf_slice[bit_idx_s0][0..R-1]` — which holds bit `bit_idx_s0` of every row's input. For each output column `c`, sum the weights of rows where that bit is 1. Sign-extend each weight from int8 to int32 before adding.

The combinational read is **one whole slice bank per cycle**, not a 1-row byte extraction. The corner-turn LOAD pre-positioned the data exactly so this read pattern is natural — every bit Stage 0 needs on cycle `bit_idx_s0` already lives in one slice bank.

This is purely combinational — `crossbar_sum_comb` is an `always @*` block. The synchronous block latches it into `crossbar_sum_reg` on the next posedge.

### Stage 0: Crossbar fire (registered latch + LOAD-gate re-arm)

`dpe_nldpe_faithful.v:231-252`

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

Each cycle while `compute_busy && bit_idx_s0 < PRECISION`:
- Latch `crossbar_sum_reg[c] <= crossbar_sum_comb[c]` (NBA — the value commits on next cycle's evaluation)
- Advance `bit_idx_s0`
- When `bit_idx_s0 == PRECISION-2` (cycle before the last slice read), NBA-set `load_safe`. This is the silicon-faithful interlock: pass-(k+1) LOAD can start on the cycle of pass-k's last slice read, because NBA ordering guarantees pass-k's combinational read sees the old slice value while pass-(k+1)'s NBA write commits at the *end* of the cycle.

**Cycle accounting**: `bit_idx_s0` advances 0→1→2→…→PRECISION over PRECISION+1 cycles (it sweeps through 0..PRECISION-1, then settles at PRECISION which means "done"). The 8 stage-0 work cycles for INT8 are bit_idx_s0 ∈ {0,1,2,3,4,5,6,7}.

### Stage 1: Analog accumulator (registered, MSB-subtract for signed)

`dpe_nldpe_faithful.v:254-266`

```verilog
if (s1_valid && bit_idx_s1 < PRECISION[4:0]) begin
    if (bit_idx_s1 == (PRECISION[4:0] - 5'd1)) begin
        // MSB: SUBTRACT (signed 2's-complement convention)
        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
            mac_acc[cc] <= mac_acc[cc] - (crossbar_sum_reg[cc] <<< bit_idx_s1);
        // Arm ACAM to fire NEXT cycle (1-cycle read-out)
        acam_fire <= 1;
    end else begin
        // Non-MSB bits: ADD with shift
        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
            mac_acc[cc] <= mac_acc[cc] + (crossbar_sum_reg[cc] <<< bit_idx_s1);
    end
    bit_idx_s1 <= bit_idx_s1 + 5'd1;
end
```

Each cycle while `s1_valid && bit_idx_s1 < PRECISION`:
- If `bit_idx_s1` is the MSB position (PRECISION-1=7 for INT8): **subtract** `crossbar_sum_reg << bit_idx_s1` from `mac_acc`. This implements 2's complement sign extension (see §7).
- Else: **add** `crossbar_sum_reg << bit_idx_s1` to `mac_acc`.
- Advance `bit_idx_s1`.

`mac_acc` is a single substrate now (`[0:NUM_COLS-1]`, no QDEPTH slot index). It is zeroed in the COMPUTE-idle wake branch (line 312) when `buf_loaded` triggers the next pass.

**`s1_valid` arming**: see `dpe_nldpe_faithful.v:268-270`. Armed one cycle after `compute_busy` wakes, so stage 1 begins reading from `crossbar_sum_reg` ONE cycle after stage 0 first latches into it. This is the pipeline depth: stage 0 fires first, stage 1 lags by 1 cycle.

**`acam_fire` arming**: see `dpe_nldpe_faithful.v:259`. Armed in the SAME NBA cycle as the MSB subtract. ACAM stage 2 fires on the next cycle (1 more cycle of latency).

### Stage 2: ACAM (registered, 3 modes)

`dpe_nldpe_faithful.v:272-295`

```verilog
if (acam_fire) begin
    if (ACAM_MODE == 0) begin
        // ADC / identity: no transform, just register the int32 MAC
        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
            acam_out[cc] <= mac_acc[cc];
    end else if (ACAM_MODE == 1) begin
        // Exp approximation: 1 + x + (x*x) >>> 1
        for (cc = 0; cc < NUM_COLS; cc = cc + 1) begin
            acam_out[cc] <= 32'sd1
                + mac_acc[cc]
                + ((mac_acc[cc] * mac_acc[cc]) >>> 1);
        end
    end else if (ACAM_MODE == 2) begin
        // Log approximation: x - 1
        for (cc = 0; cc < NUM_COLS; cc = cc + 1)
            acam_out[cc] <= mac_acc[cc] - 32'sd1;
    end
    acam_commit_pulse <= 1;
    acam_fire         <= 0;
    // COMPUTE done -- set compute_done NBA so OUTPUT picks it up next cycle.
    compute_busy   <= 0;
    bit_idx_s0     <= 0;
    bit_idx_s1     <= 0;
    s1_valid       <= 0;
    compute_done   <= 1;
    MSB_SA_Ready   <= 1;
    shift_add_done <= 1;
end
```

Reads `mac_acc` (which holds the fully-accumulated MAC after Stage 1's MSB subtract) and writes to the single-substrate `acam_out`. NBA'd transforms by mode:
- **Mode 0 (ADC / identity)**: `acam_out <= mac_acc` — just registers the value. The "ADC" interpretation: the analog accumulator's value is sampled into a digital register; later, OUTPUT truncates int32 → int8 (the actual quantisation).
- **Mode 1 (exp)**: `acam_out <= 1 + mac_acc + (mac_acc * mac_acc) >>> 1` — Taylor series approximation of e^x around x=0.
- **Mode 2 (log)**: `acam_out <= mac_acc - 1` — first-order log approximation log(1+x) ≈ x, shifted to log(x) = log(1 + (x-1)) ≈ x - 1.

ACAM always takes exactly 1 cycle, regardless of mode. Even mode 0 (identity) consumes a cycle — it's the ADC quantisation cycle.

---

## 5. The bit-serial pipeline timeline

Visualised as a per-cycle pipeline (relative cycles since `compute_busy` first went high; assumes LOAD finished and COMPUTE wake handshake completed):

```
Cycle:    0    1    2    3    4    5    6    7    8    9
                                                      |
                                                      acam_out
                                                      committed
                                                      (acam_commit_pulse)

Stage 0 latches bit-slice into crossbar_sum_reg (one bit per cycle):
  bit 0:   ●
  bit 1:        ●
  bit 2:             ●
  bit 3:                  ●
  bit 4:                       ●
  bit 5:                            ●
  bit 6:                                 ●
  bit 7:                                      ●          ← (MSB enters)

Stage 1 accumulates from crossbar_sum_reg (lags s0 by 1):
  s1 reads bit 0:      add (xbar_reg << 0)
  s1 reads bit 1:           add (xbar_reg << 1)
  s1 reads bit 2:                add (xbar_reg << 2)
  ...
  s1 reads bit 6:                                add (xbar_reg << 6)
  s1 reads bit 7:                                     SUB (xbar_reg << 7)   ← MSB, signed
                                                       └─ same NBA arms acam_fire <= 1

Stage 2 ACAM fires (1 cycle after MSB subtract):
                                                            ●         ← ACAM transform NBA'd

After ACAM commits: compute_busy <= 0; compute_done <= 1.

Total compute_busy duration: cycles 0..9 = 10 cycles.
```

The +2 on T_fill (NBA handoff for LOAD→COMPUTE and COMPUTE→OUTPUT via the `buf_loaded` / `compute_done` flags) is **already accounted for** in the wrapper FSM cycle count, NOT in CCYC itself. CCYC = the duration of `compute_busy` = 10 cycles. The wrapper-level handshakes add their +1 each, totalling T_fill = LCYC + 1 + CCYC + 1 + OCYC = 52 + 1 + 10 + 1 + 52 = 116 (for NL R=C=256 BUF=40, M=1) — unchanged by the refactor.

---

## 6. Cycle emergence — how CCYC = 10 is derived from the code

This is the **key claim** of the faithful primitive. No `COMPUTE_CYCLES` parameter exists. The 10 cycles arise from:

```
PRECISION cycles for bit-slice entry  (8 cycles, bit_idx_s0 ∈ {0..7})
+ (PIPELINE_DEPTH - 1)                (1 cycle for last bit to drain through Stage 1)
+ ACAM_CYCLES                         (1 cycle for ACAM read-out)
= 10 cycles for INT8 NL-DPE
```

For different PRECISION values (without changing the code structure):
- INT4: 4 + 1 + 1 = 6 cycles
- INT8: 8 + 1 + 1 = 10 cycles
- INT16: 16 + 1 + 1 = 18 cycles

These would emerge from the code automatically because:
- The `bit_idx_s0 < PRECISION` loop bounds scale with PRECISION
- The pipeline depth is set by code structure (2 stages: latch + accumulate)
- The ACAM stage is always 1 cycle

This is what we mean by "structural emergence" — the cycle count is a *consequence* of the code's pipeline structure, not a parameter to be matched.

### Steady-state cadence (T_steady)

CCYC and T_fill are **unchanged** by the silicon-faithful refactor — both per-pass compute duration and the single-pass total are identical to the pre-refactor numbers (10 and 116 respectively). What **did** change is the multi-pass steady-state cadence T_steady:

```
Pre-refactor (4-slot ring):  T_steady = max(LCYC, CCYC, OCYC) = max(52, 10, 52) = 52
Post-refactor (Option A1):   T_steady = max(LCYC + PRECISION, CCYC, OCYC)
                                      = max(52 + 8, 10, 52) = 60
```

The `+PRECISION` term comes from the **LOAD-gate** (`load_safe`). Pass-(k+1) LOAD cannot launch until pass-k COMPUTE has consumed all PRECISION slices of the single input substrate. So the LOAD→LOAD cadence is `LCYC + PRECISION`, not just `LCYC`. The cadence equation:

```
LOAD-pass-(k+1) launches PRECISION cycles AFTER pass-k COMPUTE wakes.
```

This is what real silicon would do with one input buffer: re-use is gated on consumption. The +PRECISION cost is the price of dropping the 4× ring-buffer hardware.

For typical configurations (LCYC + PRECISION ≥ OCYC ≥ CCYC), `T_steady = LCYC + PRECISION` dominates. For NL R=256 BUF=40 INT8: T_steady = 60.

**The testbench verifies emergence by MEASURING, not asserting**. See §9 and §11 for how T3 explicitly captures `first_compute_cycle` and `last_compute_cycle` from the Tier-2 observable pulses and verifies their difference equals 10. T4 verifies T_steady = 60 by measuring multi-pass total cycles and matching the prediction `T_fill + (M-1)·T_steady = 116 + 60·(M-1)` (e.g., 176 for M=2, 296 for M=4, 536 for M=8).

---

## 7. Signed multiply — MSB sign extension

NL-DPE operates on signed int8 inputs and weights. The bit-serial accumulator must handle 2's complement sign extension correctly.

**The trick**: at the MSB position (bit_idx_s1 = PRECISION-1 = 7 for INT8), **subtract** the weighted sum instead of adding.

```verilog
// dpe_nldpe_faithful.v:359-370 (paraphrased)
if (bit_idx_s1 == PRECISION-1) begin
    mac_acc[c] <= mac_acc[c] - (crossbar_sum_reg[c] << bit_idx_s1);
end else begin
    mac_acc[c] <= mac_acc[c] + (crossbar_sum_reg[c] << bit_idx_s1);
end
```

**Why this works**: for a signed int8 input `x` in 2's complement,
```
x = x[0]·2⁰ + x[1]·2¹ + ... + x[6]·2⁶ - x[7]·2⁷
```
The MSB bit `x[7]` represents a *negative* contribution of `2⁷`. So when summing the dot product:
```
MAC = sum_r w_r · x_r = sum_r w_r · (sum_{b=0..6} x_r[b]·2^b - x_r[7]·2⁷)
    = sum_{b=0..6} 2^b · sum_r (w_r if x_r[b] else 0)
    - 2⁷ · sum_r (w_r if x_r[7] else 0)
```
The bracketed `sum_r` at each bit position is exactly `crossbar_sum_reg` (after sign-extending the int8 weight). So we accumulate (+) for bit positions 0..6 and (−) for the MSB position 7.

**Verification (T7)**: the testbench uses random signed inputs including negatives (MSB=1 cases). The numpy oracle computes `signed_int8_mac` using standard numpy int8/int32 arithmetic. The faithful primitive's MAC must match bit-exactly. T7 passes ⇒ MSB-subtract logic is correct.

---

## 8. ACAM modes (Stage 2)

ACAM has 3 modes, all registered, all 1 cycle. The mode is set at compile time via the `ACAM_MODE` parameter.

### Mode 0 — ADC / identity

`acam_out <= mac_acc`

**Physical interpretation**: the analog accumulator's value is sampled into a digital register. There is no nonlinear transform; the only "work" done is the analog-to-digital quantisation. The cycle is still consumed because the sampling itself is a registered stage.

This is the most common mode for ordinary VMM operations (matrix multiply without fused activation).

### Mode 1 — Exp approximation

`acam_out <= 1 + mac_acc + (mac_acc * mac_acc) >>> 1`

**Physical interpretation**: the ACAM hardware uses a content-addressable LUT to approximate e^x. We approximate using the Taylor expansion truncated at second order:
```
e^x ≈ 1 + x + x²/2 + x³/6 + ...
    ≈ 1 + x + x²/2  (second-order truncation)
```
In int32 arithmetic, `(mac_acc * mac_acc) >>> 1` is the `x²/2` term (signed right shift = signed divide by 2 rounding toward minus infinity).

**Caveat**: for large `mac_acc` (say |mac_acc| > 2¹⁵), the `mac_acc * mac_acc` term will overflow int32. This is a known limitation; in real silicon the ACAM LUT would saturate. In our model, overflow wraps silently — which is fine for small test cases (T5 uses MAC=4, well within range).

### Mode 2 — Log approximation

`acam_out <= mac_acc - 1`

**Physical interpretation**: log(x) Taylor around x=1: log(1+u) ≈ u, so log(x) ≈ x - 1. First-order only.

---

## 9. Testbench walkthrough

`fc_verification/tb_dpe_nldpe_faithful.v` is a single TB file with 7 test modes selected via `+define+TEST_MODE=N`. Each test loads pre-generated `.mem` files (from the Python oracle), drives the DUT, captures outputs, and compares against oracle predictions.

### Top-level structure

```verilog
// tb_dpe_nldpe_faithful.v:1-200 (paraphrased)

`define VECT_DIR "/path/to/oracles/test_vectors/"

module tb_dpe_nldpe_faithful;
    // Standard clock/reset, drive signals (data_in, w_buf_en, nl_dpe_control),
    // DUT instantiation with PRECISION=8, etc.
    
    // Test vector storage (loaded from .mem files via $readmemh)
    reg signed [7:0]  w_mem    [0:R*C-1];          // weights
    reg signed [7:0]  x_mem    [0:M_PASSES*R-1];   // inputs (per pass)
    reg signed [7:0]  y_mem    [0:M_PASSES*C-1];   // expected outputs (oracle)
    reg signed [31:0] mac_mem  [0:M_PASSES*C-1];   // expected pre-truncation MAC

    initial begin
        // 1. Load test vectors from .mem files
        $readmemh({`VECT_DIR, "t<N>_weights.mem"},  w_mem);
        $readmemh({`VECT_DIR, "t<N>_inputs.mem"},   x_mem);
        $readmemh({`VECT_DIR, "t<N>_expected.mem"}, y_mem);
        $readmemh({`VECT_DIR, "t<N>_mac.mem"},      mac_mem);

        // 2. Hierarchical-force weights into dut.weights[r][c]
        for (r = 0; r < R; r = r + 1)
            for (c = 0; c < C; c = c + 1)
                dut.weights[r][c] = w_mem[r * C + c];

        // 3. Drive LOAD strobes (one byte per cycle, EPS bytes per strobe)
        //    Capture T_first_load when first w_buf_en is observed.

        // 4. Continue driving until M_PASSES * LCYC strobes; meanwhile,
        //    a parallel always block captures data_out into captured[].

        // 5. Wait for dpe_done to go low (signals OUTPUT phase complete).
        //    Capture T_done_last.

        // 6. Compare captured outputs byte-exact against y_mem (numpy oracle).

        // 7. Compute total_cycles = T_done_last - T_first_load + 1.
        //    Verify total_cycles matches oracle's expected_total_cycles.
    end
endmodule
```

### The 7 test modes

| Mode | Setup | What it validates |
|---|---|---|
| **T1** | Identity weights, all-ones input, ACAM_MODE=0 | Functional LOAD→bit-serial→drain path works for the trivial case |
| **T2** | Random int8 weights and inputs (seed-controlled), ACAM_MODE=0 | Bit-exact functional correctness vs numpy oracle |
| **T3** | M=1, identity weights, ACAM_MODE=0 — **cycle emergence check** | Measures `last_compute_cycle - first_compute_cycle + 1` and verifies it equals the oracle's predicted CCYC=10. Confirms cycle count is *measured*, not *asserted*. |
| **T4** | M ∈ {1, 2, 4, 8} M-sweep, identity weights, ACAM_MODE=0 | LOAD-gated multi-pass cadence: total cycles match T_fill + (M−1)·T_steady = 116 + 60·(M−1) ⇒ 116 / 176 / 296 / 536. Each pass is functionally correct; TB also confirms `load_safe` falls-then-rises between passes. |
| **T5** | M=1, weights/inputs designed to give MAC=4 per col, ACAM_MODE=1 | Exp transform: output = `1 + 4 + 16/2 = 13` |
| **T6** | M=1, MAC=4, ACAM_MODE=2 | Log transform: output = `4 − 1 = 3` |
| **T7** | M=1, random signed inputs (including negatives), ACAM_MODE=0 | Signed 2's complement MAC works — MSB-subtract logic correct |

### How T3 verifies cycle emergence

`tb_dpe_nldpe_faithful.v` (test 3 block) — paraphrased:

```verilog
// Telemetry capture from Tier 2 observable pulses
integer first_compute_cycle, last_compute_cycle, t_acam_commit;
integer cycle_count;
always @(posedge clk) cycle_count <= cycle_count + 1;

always @(posedge clk) begin
    if (dut.compute_first_bit_pulse) first_compute_cycle <= cycle_count;
    if (dut.compute_busy)            last_compute_cycle  <= cycle_count;
    if (dut.acam_commit_pulse)       t_acam_commit       <= cycle_count;
end

// At end of test:
measured_ccyc = last_compute_cycle - first_compute_cycle + 1;
declared_ccyc_oracle = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES;  // = 10

if (measured_ccyc == declared_ccyc_oracle) begin
    $display("*** CCYC=%0d EMERGED from physical structure (NOT set as parameter) ***",
             measured_ccyc);
end else begin
    $display("*** CCYC MISMATCH: measured=%0d declared=%0d ***",
             measured_ccyc, declared_ccyc_oracle);
    $error;
end
```

**The point**: the TB *measures* the duration of `compute_busy` directly from the DUT's observable register. It compares this measurement against an *independently computed* prediction (from the oracle's structural formula). If they match, the emergence claim is verified. If the RTL had a bug (off-by-one in bit_idx_s1, missing ACAM_CYCLES, etc.), the measurement would differ from the prediction.

This is fundamentally different from the lazy primitive's testbench, which sets `COMPUTE_CYCLES=10` as a parameter and then verifies that the RTL takes 10 cycles. That's tautological — the RTL takes 10 because we told it to.

---

## 10. Python oracle methodology

`fc_verification/oracles/nldpe_mac_oracle.py` is the **independent reference** that the TB compares against.

### Key functions

```python
def signed_int8_mac(weights_2d, inputs_1d):
    """Per-column signed int8 × int8 → int32 dot product.
    Uses numpy int32 cast then matrix multiply.
    """
    return weights_2d.astype(np.int32).T @ inputs_1d.astype(np.int32)

def bit_serial_signed_mac(weights, inputs, precision=8):
    """Reference implementation of the bit-serial MSB-subtract trick.
    Used to cross-check `signed_int8_mac` (which uses direct multiply).
    """
    # ... see oracle for full implementation
    # The key step: for bit position < precision-1, add; for MSB, subtract.

def acam_transform(mac_int32, mode):
    """Apply ACAM mode transform:
      mode 0 (identity): return mac
      mode 1 (exp):      return 1 + mac + (mac * mac) >> 1  (signed)
      mode 2 (log):      return mac - 1
    """

def expected_total_cycles(R, C, BUF, M=1, PRECISION=8, PIPELINE_DEPTH=2, ACAM_CYCLES=1):
    """Cycle prediction from physical model (NOT from RTL or sim formula).
    
    CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES   ← derived from physics
    LCYC = ceil(R * 8 / BUF)
    OCYC = ceil(C * 8 / BUF)
    T_fill = LCYC + 1 + CCYC + 1 + OCYC                              ← +2 = NBA wrapper handoffs
    T_steady = max(LCYC + PRECISION, CCYC, OCYC)                     ← Option A1 (LOAD-gated)
    T(M) = T_fill + (M-1) * T_steady                                  e.g. 116/176/296/536
    """
```

### Independence from RTL and sim formulas

The oracle:
- Does **not** read `dpe_nldpe_faithful.v`
- Does **not** read `imc_core.py`'s `run_gemm`
- Computes results purely from first principles (signed numpy MAC) and the **declared** physical model (PIPELINE_DEPTH=2, ACAM_CYCLES=1, etc.)

This makes it a genuine third-party reference. When TB outputs match the oracle, it means the RTL implements signed int8 MAC + ACAM transform correctly. When measured cycle count matches the oracle's prediction, it means the RTL's pipeline structure produces the cycle count the methodology *predicts*.

### Test vector generation

`nldpe_mac_oracle.py --gen-test-vectors` writes 36 `.mem` files into `oracles/test_vectors/`:

```
t1_weights.mem    t1_inputs.mem    t1_expected.mem    t1_mac.mem    t1_meta.txt
t2_weights.mem    t2_inputs.mem    t2_expected.mem    t2_mac.mem    t2_meta.txt
... (T3 through T7, plus cycle_oracle.txt)
```

Each `.mem` file is a flat hex dump (`$readmemh` format). The TB loads them at simulation start. Re-running the generator with different seeds would refresh the test vectors deterministically.

The `cycle_oracle.txt` records the predicted cycle counts per test (e.g., `T3: LCYC=52 CCYC=10 OCYC=52 T_fill=116`). The TB uses these as the cycle-check oracle.

---

## 11. Worked example — T1 cycle-by-cycle

Setup:
- NL-DPE INT8 R=C=256 BUF=40 → LCYC=OCYC=52
- M=1, identity weights, all-ones input, ACAM_MODE=0
- Predicted: each output[c] = 1 (= 1·1 from identity weights), total_cycles = 116

### TB drives LOAD (cycles 5–56, 52 strobes)

```
Cycle 5:  TB drives w_buf_en=1, data_in = first 5 bytes of inputs (0x01 each).
          load_safe = 1 (initial), so the gate passes the strobe through.
          T_first_load <= 5 (TB captures the cycle of first w_buf_en).
          DUT corner-turn: for j=0..4, for i=0..7, input_buf_slice[i][j] NBA <= data_in[j*8+i].
          For input byte 0x01: bit 0 = 1, bits 1..7 = 0. So slice[0][0..4] NBA <= 1,
          slice[1..7][0..4] NBA <= 0.
          
Cycle 6:  next 5 bytes streamed; slice[0][5..9] NBA <= 1, slice[1..7][5..9] NBA <= 0.
...
Cycle 56: last LOAD strobe (52nd byte chunk). DUT:
          - slice[0][255] NBA <= 1, slice[1..7][255] NBA <= 0
          - buf_loaded NBA <= 1 (signals COMPUTE-idle wake branch)
          - load_cycle_cnt NBA <= 0 (re-arm for next pass)
```

### LOAD→COMPUTE handshake (cycle 57)

```
Cycle 57: DUT sees buf_loaded = 1 (NBA committed).
          COMPUTE idle branch fires:
            compute_busy           NBA <= 1
            bit_idx_s0             NBA <= 0
            bit_idx_s1             NBA <= 0
            s1_valid               NBA <= 0
            compute_first_bit_pulse NBA <= 1  ← Tier 2 telemetry
            buf_loaded             NBA <= 0
            load_safe              NBA <= 0   ← LOCK substrate against pass-(k+1) LOAD
            mac_acc[c]             NBA <= 0  for all c
          
          TB captures: first_compute_cycle = 57 (when compute_first_bit_pulse fires).
```

### Bit-serial sweep (cycles 58–66)

```
Cycle 58: compute_busy = 1 visible. bit_idx_s0 = 0. Stage 0 reads slice bank 0:
          crossbar_sum_comb[c] = sum_r (input_buf_slice[0][r] ? weights[r][c] : 0)
                               = weights[c][c] = 1  (for c < min(R, C); else 0)
            (slice[0][r] = 1 for all r since input bytes are 0x01; identity weights
             pick out the diagonal so only weights[c][c] contributes)
          crossbar_sum_reg[c] NBA <= 1
          bit_idx_s0          NBA <= 1
          s1_valid            NBA <= 1   ← s1 will start next cycle

Cycle 59: bit_idx_s0 = 1, s1_valid = 1, bit_idx_s1 = 0.
          Stage 0: slice bank 1. slice[1][r] = 0 for all r → crossbar_sum_comb[c] = 0.
          crossbar_sum_reg[c] NBA <= 0
          Stage 1: bit_idx_s1 = 0 (not MSB).
            mac_acc[c] NBA <= mac_acc[c] + (crossbar_sum_reg[c] << 0)
                            = 0 + (1 << 0) = 1
          bit_idx_s0 NBA <= 2; bit_idx_s1 NBA <= 1.

Cycle 60-64: similar. bit_idx_s0 advances 2..6 (reads slice banks 2..6, all zero);
             bit_idx_s1 advances 1..5. mac_acc[c] remains 1 (added 0 each cycle).

Cycle 65: bit_idx_s0 = 7, bit_idx_s1 = 6.
          Stage 0: slice bank 7 (all zero). crossbar_sum_reg NBA <= 0.
            Note: bit_idx_s0 == PRECISION-2 == 6 was hit one cycle earlier, so
            load_safe NBA was set to 1 on cycle 64. (Pass-1 LOAD could now start
            on cycle 65; for this M=1 example there's no pass 1.)
          Stage 1: bit_idx_s1 = 6 (not MSB-1=7). mac_acc[c] += 0 << 6 = no change.

Cycle 66: bit_idx_s0 = 8 = PRECISION (stage 0 no-fire). bit_idx_s1 = 7 = PRECISION-1.
          Stage 1: MSB position!
            mac_acc[c] NBA <= mac_acc[c] - (crossbar_sum_reg[c] << 7)
                            = 1 - (0 << 7) = 1
          acam_fire NBA <= 1  ← arms ACAM for next cycle

          last_compute_cycle observed by TB = 66 (compute_busy still high).
```

### ACAM fire (cycle 67)

```
Cycle 67: acam_fire = 1. Stage 2 (mode 0):
            acam_out[c] NBA <= mac_acc[c] = 1
          acam_commit_pulse NBA <= 1  ← Tier 2 telemetry
          compute_busy     NBA <= 0
          compute_done     NBA <= 1
          bit_idx_s0, s1, s1_valid NBA cleared
          acam_fire NBA <= 0
          
          MEASURED CCYC = last_compute_cycle - first_compute_cycle + 1
                        = 66 - 57 + 1 = 10  ✓ EMERGED FROM STRUCTURE
```

### COMPUTE→OUTPUT handshake (cycle 68)

```
Cycle 68: DUT sees compute_done = 1 (NBA committed).
          OUTPUT idle branch fires:
            output_busy    NBA <= 1
            output_col_idx NBA <= 0
            compute_done   NBA <= 0
```

### OUTPUT drain (cycles 69–120, 52 strobes)

```
Cycle 69: output_busy = 1. Stage drains:
            data_out NBA <= acam_out[0..4][7:0] = 0x01 each (truncated low byte)
            dpe_done NBA <= 1
          output_col_idx NBA <= 1.

Cycle 70: TB captures data_out = 0x01 (per first OUTPUT byte). cap_strobe_idx 0→1.
          T_done_last <= 70 (start of T_done_last tracking).
          DUT: data_out NBA <= acam_out[5..9][7:0] = 0x01 each.

... 51 more drain cycles ...

Cycle 120: Last OUTPUT strobe. TB captures data_out = 0x01.
           T_done_last <= 120.
           DUT: output_col_idx NBA <= 0; output_busy NBA <= 0 (compute_done = 0).
```

### TB final check

```
Cycle 121: TB observes dpe_done went low. Final compare phase begins.

total_cycles = T_done_last - T_first_load + 1 = 120 - 5 + 1 = 116  ✓
Compare captured[0..255*5-1] byte-by-byte against y_mem[0..255]:
  y_mem[c] = 0x01 for c < 256, 0x00 otherwise.
  All match → T1 PASS.

Also: MEASURED CCYC = 10 from §11.5 above ✓ EMERGED FROM STRUCTURE.

[tb_faithful] T1 T_first_load=5 T_done_last=120 total_cycles=116 (oracle T_fill=116)
[tb_faithful] T1 PASS
```

---

## 11.5. Concrete numerical example — bit-serial MAC math

The §11 trace above shows the cycle-level FSM coordination for the trivial identity case. This section shows the **actual numerical computation** with non-trivial signed inputs and weights, so you can follow how each bit-slice contributes to the final MAC value.

### Example setup

For pedagogical clarity, use a **small** matrix (R=4, C=2). The RTL works the same way at R=256, C=256 — just with bigger summation loops. Same INT8, signed, ACAM_MODE=0 (ADC/identity passthrough).

**Weights matrix** `W[4][2]`:

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

**Expected MAC per column** (signed int8 × int8 → int32):

```
MAC[0] = 3·5 + 2·(-3) + (-4)·1 + 1·2
       = 15 - 6 - 4 + 2 = 7

MAC[1] = (-1)·5 + 5·(-3) + 1·1 + (-2)·2
       = -5 - 15 + 1 - 4 = -23
```

After ACAM mode 0 (identity passthrough): `acam_out[0] = 7`, `acam_out[1] = -23` (single-substrate `acam_out`, indexed by column).

After OUTPUT truncation int32 → int8 (low byte):
- `data_out byte for c=0` = `7 & 0xFF = 0x07`
- `data_out byte for c=1` = `-23 & 0xFF = 0xE9` (signed int8: -23 = 256-23 mod 256 = 233 = 0xE9)

### Bit-slice extraction table

For each row, list which bit is `1` at each bit position (b = 0..7). After LOAD's corner-turn, this same data lives in `input_buf_slice[b][r]` — column `b` of the table below is *literally* slice bank `b`:

```
b:        0  1  2  3  4  5  6  7
X[0]= 5:  1  0  1  0  0  0  0  0    (0b00000101)    → slice[0..7][0] = these bits
X[1]=-3:  1  0  1  1  1  1  1  1    (0b11111101, 2c) → slice[0..7][1] = these bits
X[2]= 1:  1  0  0  0  0  0  0  0    (0b00000001)    → slice[0..7][2] = these bits
X[3]= 2:  0  1  0  0  0  0  0  0    (0b00000010)    → slice[0..7][3] = these bits
```

So `input_buf_slice[0][0..3] = {1, 1, 1, 0}` (bit 0 of X[0..3]), `input_buf_slice[1][0..3] = {0, 0, 0, 1}` (bit 1), etc.

So at each bit position, the set of rows with bit=1 is:

```
b=0: rows {0, 1, 2}        (X[0], X[1], X[2] all have bit 0 = 1)
b=1: rows {3}              (only X[3] has bit 1 = 1)
b=2: rows {0, 1}           (X[0]=5 and X[1]=-3 both have bit 2 = 1)
b=3: rows {1}              (only X[1]=-3, sign extension bits start here)
b=4: rows {1}
b=5: rows {1}
b=6: rows {1}
b=7: rows {1}              (MSB — X[1] has bit 7 = 1 since it's negative)
```

### Stage 0 — crossbar_sum_comb per bit-slice

At each cycle while `compute_busy && bit_idx_s0 < PRECISION`, the combinational block (`dpe_nldpe_faithful.v:151-164`) reads slice bank `bit_idx_s0` and computes:

```
crossbar_sum_comb[c] = sum_{r : input_buf_slice[bit_idx_s0][r] == 1} W[r][c]
```

For our example:

| b | Rows with bit_b=1 | crossbar_sum_comb[0]       | crossbar_sum_comb[1]      |
|---|---|---|---|
| 0 | {0, 1, 2} | W[0][0]+W[1][0]+W[2][0] = 3+2+(-4) = **1** | W[0][1]+W[1][1]+W[2][1] = -1+5+1 = **5** |
| 1 | {3}       | W[3][0] = **1**                              | W[3][1] = **-2**                            |
| 2 | {0, 1}    | W[0][0]+W[1][0] = 3+2 = **5**                | W[0][1]+W[1][1] = -1+5 = **4**              |
| 3 | {1}       | W[1][0] = **2**                              | W[1][1] = **5**                             |
| 4 | {1}       | W[1][0] = **2**                              | W[1][1] = **5**                             |
| 5 | {1}       | W[1][0] = **2**                              | W[1][1] = **5**                             |
| 6 | {1}       | W[1][0] = **2**                              | W[1][1] = **5**                             |
| 7 | {1}       | W[1][0] = **2** ← MSB                       | W[1][1] = **5** ← MSB                      |

These values are latched into `crossbar_sum_reg[c]` each cycle via the NBA at line 233.

### Stage 1 — mac_acc accumulation timeline

`bit_idx_s1` lags `bit_idx_s0` by 1 cycle. While `bit_idx_s1 < PRECISION-1`: ADD `(crossbar_sum_reg << bit_idx_s1)` to `mac_acc`. At `bit_idx_s1 == PRECISION-1 = 7` (the MSB): SUBTRACT instead.

#### Column c=0 accumulation:

```
Starting mac_acc[0] = 0  (cleared in compute idle wake branch, line 312)

bit_idx_s1=0:   xbar_reg=1.   shift = 1<<0 = 1.    mac_acc[0] += 1.    → mac_acc[0]= 1
bit_idx_s1=1:   xbar_reg=1.   shift = 1<<1 = 2.    mac_acc[0] += 2.    → mac_acc[0]= 3
bit_idx_s1=2:   xbar_reg=5.   shift = 5<<2 = 20.   mac_acc[0] += 20.   → mac_acc[0]= 23
bit_idx_s1=3:   xbar_reg=2.   shift = 2<<3 = 16.   mac_acc[0] += 16.   → mac_acc[0]= 39
bit_idx_s1=4:   xbar_reg=2.   shift = 2<<4 = 32.   mac_acc[0] += 32.   → mac_acc[0]= 71
bit_idx_s1=5:   xbar_reg=2.   shift = 2<<5 = 64.   mac_acc[0] += 64.   → mac_acc[0]= 135
bit_idx_s1=6:   xbar_reg=2.   shift = 2<<6 = 128.  mac_acc[0] += 128.  → mac_acc[0]= 263
bit_idx_s1=7:   xbar_reg=2.   shift = 2<<7 = 256.  mac_acc[0] -= 256.  → mac_acc[0]= 7   ← MSB SUBTRACT
                                                                          (line 257)
```

**Final mac_acc[0] = 7** ✓ matches the directly-computed MAC. (Single-substrate `mac_acc[0]` here means column 0, not slot 0 — there is no slot index anymore.)

#### Column c=1 accumulation:

```
Starting mac_acc[1] = 0

bit_idx_s1=0:   xbar_reg= 5.   shift =  5<<0 =   5.    mac_acc[1] +=  5.   → mac_acc[1]=   5
bit_idx_s1=1:   xbar_reg=-2.   shift = -2<<1 = -4.    mac_acc[1] += -4.   → mac_acc[1]=   1
bit_idx_s1=2:   xbar_reg= 4.   shift =  4<<2 =  16.    mac_acc[1] += 16.   → mac_acc[1]=  17
bit_idx_s1=3:   xbar_reg= 5.   shift =  5<<3 =  40.    mac_acc[1] += 40.   → mac_acc[1]=  57
bit_idx_s1=4:   xbar_reg= 5.   shift =  5<<4 =  80.    mac_acc[1] += 80.   → mac_acc[1]= 137
bit_idx_s1=5:   xbar_reg= 5.   shift =  5<<5 = 160.    mac_acc[1] += 160.  → mac_acc[1]= 297
bit_idx_s1=6:   xbar_reg= 5.   shift =  5<<6 = 320.    mac_acc[1] += 320.  → mac_acc[1]= 617
bit_idx_s1=7:   xbar_reg= 5.   shift =  5<<7 = 640.    mac_acc[1] -= 640.  → mac_acc[1]= -23  ← MSB SUBTRACT
```

**Final mac_acc[1] = -23** ✓ matches the directly-computed MAC.

### Why MSB subtract is correct (mathematical justification)

For signed int8 in 2's complement:

```
X = bit_0·2⁰ + bit_1·2¹ + ... + bit_6·2⁶ - bit_7·2⁷
```

So the dot product `sum_r W[r][c]·X[r]` decomposes as:

```
sum_r W[r][c]·X[r]
  = sum_r W[r][c] · (sum_{b=0..6} X[r][b]·2^b - X[r][7]·2⁷)
  = sum_{b=0..6} 2^b · sum_r (W[r][c] if X[r][b]==1 else 0)
                - 2⁷  · sum_r (W[r][c] if X[r][7]==1 else 0)
                                ↑
                          this is exactly crossbar_sum_comb for b=7
```

Each bit-position term `sum_r (W[r][c] if X[r][b]==1 else 0)` is exactly what Stage 0 computes for `crossbar_sum_comb` at that bit position. Stage 1 multiplies by `2^b` (the `<<= bit_idx_s1` shift) and ADDS for b<7, SUBTRACTS for b=7 — matching the decomposition above.

### Stage 2 — ACAM (mode 0 / identity / ADC)

After Stage 1 commits the MSB subtract at end of `bit_idx_s1=7` cycle, `acam_fire` is armed for the next cycle.

On the ACAM cycle:

```
acam_out[0] <= mac_acc[0] =  7      (mode 0: identity passthrough)
acam_out[1] <= mac_acc[1] = -23
acam_commit_pulse NBA <= 1
compute_busy NBA <= 0
compute_done NBA <= 1   (single-substrate handshake to OUTPUT)
```

### Output drain — truncation

OUTPUT drains `acam_out` (single substrate, one entry per column) byte-by-byte. For each output column, the int32 value is truncated to its low 8 bits:

```
c=0:   acam_out[0] =  7  (0x00000007)   → data_out byte = 0x07
c=1:   acam_out[1] = -23 (0xFFFFFFE9)   → data_out byte = 0xE9
```

These are written to `data_out` 5 bytes per cycle (EPS=5 for BUF=40). The TB captures them into `captured[]` and compares against `y_mem[]` from the oracle.

For this example: `y_mem[0] = 0x07`, `y_mem[1] = 0xE9`, rest = 0 (only c=0 and c=1 are valid columns).

### Cycle-level alignment (numerical example mapped to FSM cycles)

Using the same cycle numbering as §11 (T_first_load=5, etc.) but for this example:

- LOAD takes LCYC cycles to stream R=4 bytes: with EPS=5, LCYC = ceil(4*8/40) = 1 strobe. (At R=4, the LOAD is trivial — single strobe.)
- For the standard NL geometry (R=256), LCYC=52 still applies.

Using the standard R=256 timing for direct cycle comparison:

| Cycle | bit_idx_s0 | bit_idx_s1 | crossbar_sum_reg[0] | mac_acc[0] (post-NBA) |
|---:|:-:|:-:|---:|---:|
| 58 (compute_busy=1, first cycle) | 0 (reading slice bank 0) | – | – (was 0) | 0 |
| 59 | 1 (slice bank 1) | 0 (s1_valid armed) | 1 (NBA from cycle 58) | 0 + 1 = 1 |
| 60 | 2 (slice bank 2) | 1 | 1 (NBA from cycle 59) | 1 + 2 = 3 |
| 61 | 3 (slice bank 3) | 2 | 5 (NBA from cycle 60) | 3 + 20 = 23 |
| 62 | 4 | 3 | 2 | 23 + 16 = 39 |
| 63 | 5 | 4 | 2 | 39 + 32 = 71 |
| 64 | 6 (load_safe NBA <= 1 fires here for next-pass overlap) | 5 | 2 | 71 + 64 = 135 |
| 65 | 7 (last slice read; load_safe observed = 1) | 6 | 2 | 135 + 128 = 263 |
| 66 | 8 (idle) | 7 (MSB!) | 2 | 263 − 256 = **7** ← acam_fire armed |
| 67 | – | – | – | 7 (read by Stage 2, NBA to acam_out[0]; compute_done <= 1) |

At cycle 67, `acam_out[0] = 7` (single substrate, column 0; identity passthrough for ACAM mode 0).

### Summary of the bit-serial computation

The faithful primitive computes:

```
MAC[c] = sum_{b=0..PRECISION-2} 2^b · (sum_r W[r][c] if X[r][b]==1 else 0)
       - 2^(PRECISION-1)        · (sum_r W[r][c] if X[r][PRECISION-1]==1 else 0)
```

This is the **algorithmic decomposition** of signed multiplication into a sum of (bit-slice × weight) terms. The RTL parallelizes this across columns (the for-loop over `c_idx` in §4 Stage 0) and time-multiplexes across bit positions (the `bit_idx_s0/s1` advance one per cycle).

Each cycle does **one bit-slice across all columns**. With PRECISION=8, exactly 8 bit-slices need to traverse Stages 0 and 1. With 2-stage pipeline depth, the last bit takes 1 extra cycle to drain. With 1 ACAM cycle, the total compute duration is 8 + 1 + 1 = 10 cycles.

The TB verifies this end-to-end:
- Functional: captured int8 bytes match the oracle's `signed_int8_mac` output (T2, T7)
- Cycle: measured CCYC = 10 emerges from the stage register propagation (T3)
- Numerical: the trace above (manually computed) gives the same MAC values that numpy computes directly via `weights.T @ inputs`

This is what makes the faithful primitive an **independent reference**: the cycle count comes from physically advancing bit indices through stages, and the MAC values come from physically accumulating signed shift-add terms. Both can be cross-checked against the oracle without referring to the simulator's analytical formula.

---

## 12. Verification surface — what each test catches

| Bug type | T# that catches it | How |
|---|---|---|
| LOAD streams wrong bytes to wrong buffer slot | T2, T4 | Bit-exact functional check fails |
| Stage 0 crossbar logic doesn't sign-extend weights | T7 (mixed signed) | MAC magnitude wrong; functional fail |
| Stage 1 doesn't subtract at MSB | T7 | Negative inputs produce wrong (positive) MAC |
| Stage 1 misorders bit_idx_s1 (e.g., starts from MSB instead of LSB) | T2, T7 | MAC wrong by factor of bit-shift |
| Pipeline depth bug (s0 and s1 not 1-cycle offset) | T3 cycle measurement | first_compute_cycle / last_compute_cycle gap != 10 |
| ACAM mode 1 misses x² term | T5 | output != 13 (would be e.g. 5 or 9) |
| ACAM mode 2 wrong constant | T6 | output != 3 |
| LOAD-gate (`load_safe`) clears too early — pass-(k+1) LOAD corrupts pass-k slices | T4 (M-sweep) | Functional fail at M≥2; pass-k MAC scrambled by overwritten slice cells |
| LOAD-gate stays asserted too late — pass-(k+1) LOAD waits unnecessarily | T4 (M-sweep) | Cycle count > T_fill + (M-1)·T_steady = 116 + (M-1)·60 |
| Corner-turn bit permutation wrong (slice[i][j] gets wrong data_in bit) | T2, T4 | Functional fail; MAC reflects scrambled input bits |
| Compute substrate (single-banked mac_acc / acam_out) reused before drain | T4 (M-sweep) | Pass m's MAC corrupts pass m+1; output mismatch |
| Drain-load overlap broken | T4 (M-sweep) | Total cycles != T_fill + (M-1)·T_steady = 116 / 176 / 296 / 536 |

This is a richer verification surface than the lazy primitive's TB. The lazy TB only checks "MAC math correct" and "cycle count == declared value". The faithful TB checks "MAC math correct", "per-stage pipeline geometry correct" (T3), "ACAM transforms correct" (T5, T6), and "signed multiply correct" (T7).

---

## 13. Migration plan (deferred — not in scope for Task #91)

The faithful primitive currently coexists with the lazy primitive. The lazy `dpe_nldpe.v` is still what `fc_top.v` and all the existing TBs use.

Migration to make faithful the default:

1. **Cross-verification**: run existing TBs (`tb_dpe_vmm.v`, `tb_dpe_vmm_msweep.v`, `tb_dpe_acam.v`) against `dpe_nldpe_faithful.v` (instead of the lazy `dpe_nldpe.v`). The faithful primitive should pass them all — it's a superset of the lazy primitive's verification surface.

2. **Cycle count reconciliation**: faithful and lazy produce identical cycle counts (T_fill = 116 for INT8 NL-DPE R=C=256 BUF=40, M=1). Both have the same +2 NBA handoff overhead. The fc_top.v wrapper and downstream smoke tests should observe no change.

3. **Replace**: rename `dpe_nldpe.v` → `dpe_nldpe_lazy.v` (archive); rename `dpe_nldpe_faithful.v` → `dpe_nldpe.v`. Update the generator script reference. Re-run all smoke gates.

4. **AL faithful**: once NL migration is complete, mirror the work for Azure-Lily (Task #92). AL's pipeline has 3 internal stages (MAC + ADC + shift-add) and no ACAM. CCYC emergence should give the same 10 cycles by structural symmetry (PRECISION + 2 + 0 = 10) — but via different code structure.

Migration is a separate task. For now, the faithful primitive lives as a parallel-track development artifact that can be inspected and reasoned about independently.

---

## 14. File map

| File | Purpose | Lines |
|---|---|---|
| `fc_verification/rtl/dpe_nldpe_faithful.v` | Faithful primitive | 474 |
| `fc_verification/tb_dpe_nldpe_faithful.v` | Standalone TB with 7 test modes | 735 |
| `fc_verification/oracles/nldpe_mac_oracle.py` | Independent numpy reference | 397 |
| `fc_verification/oracles/test_vectors/*.mem` | Pre-generated test vectors (36 files) | — |
| `nl_dpe/gen_dpe_nldpe_faithful.py` | Generator from JSON config | ~250 |
| `fc_verification/Makefile` | +3 targets: `tb-faithful-nldpe`, `faithful-vectors`, `faithful-smoke` | — |

## 15. Build and run commands

```bash
# Regenerate test vectors from oracle (deterministic; same seed → same vectors)
make -C fc_verification faithful-vectors

# Run a single test
make -C fc_verification tb-faithful-nldpe TEST_MODE=1

# Run all 7 tests
make -C fc_verification faithful-smoke
```

Expected output of `faithful-smoke`: 7 sequential PASS reports, each emitting cycle measurements and `total_cycles` (always 116 for M=1; for M-sweep T4: 116/176/296/536 for M ∈ {1,2,4,8} per the Option A1 cadence T_steady = LCYC + PRECISION = 60).

---

## 16. Summary

The faithful NL-DPE primitive is a **physical behavioral model** that:
- Has no `COMPUTE_CYCLES` parameter
- Implements the bit-serial pipeline with 3 visible stages (crossbar / accumulator / ACAM)
- Handles signed int8 MAC correctly via MSB-subtract
- Supports 3 ACAM modes (ADC/identity, exp, log) each in 1 cycle
- Exposes all internal state as Tier 2 observable registers for TB introspection
- Validates "CCYC = PRECISION + (PIPELINE_DEPTH−1) + ACAM_CYCLES" by **measuring** the gap from physical structure rather than asserting it via parameter

The verification surface includes 7 tests with bit-exact numpy oracle for functional correctness and explicit Tier 2 telemetry for cycle emergence verification. The Python oracle is independent of both the simulator's `run_gemm` formula and the RTL — it computes from first principles.

This is the third leg of the verification triangle: the lazy primitive implements the formula (with the COMPUTE_CYCLES parameter); the simulator emits the formula analytically; **the faithful primitive derives cycle count from the physical structure**, giving the formula a genuine cross-check rather than self-validation.
