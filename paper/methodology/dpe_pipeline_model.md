# Per-Pass DPE Pipeline Model — Dataflow, BRAM Layout, Transpose

**Date:** 2026-04-18

## 1. Purpose and scope

This document specifies the hardware model for a **single DPE pass** — how
input data flows from BRAM into the analog crossbar, how the crossbar fires,
and how the output is serialized back out. It covers:

- Two BRAM data-layout options (natural-packed vs bit-plane-transposed).
- The transpose-buffer logic needed to move data between natural-packed
  producers and bit-serial crossbar consumers.
- Multi-pass pipelined timing, where the load of pass `k+1` overlaps with
  the output of pass `k`.
- Analytical cycle formulas parameterised by `R, C, W_BRAM, W_DPE,
  N_in_bits`.

This model sits **below** `attention_dimm_mapping.md` (which describes
K-identity mapping, W=16 lane allocation, DPE/DSP count) and **above**
`fc_verification/VERIFICATION.md` (which documents measured cycle counts).

## 2. Analog IMC primer

A DPE contains an analog crossbar of `R` rows × `C` columns. Each cell
stores one signed weight. Row drivers (DACs) apply the input vector to
the word-lines, currents sum on the bit-lines, and `C` per-column ADCs
digitise the result.

For 8-bit integer precision, the crossbar operates **bit-serially on the
input**: each analog fire uses one bit-plane of the input (1 bit per row),
produces a partial-MAC result per column, and a digital shift-add unit
accumulates `N_in_bits = 8` partial results into the final 8-bit × W-bit
MAC output. Inputs and outputs are 8-bit; the crossbar itself fires
`N_in_bits` times per full MAC.

In our configuration the DPE is wrapped with fixed parameters:

| Parameter | Symbol | Typical value |
|---|---|---|
| Crossbar rows (input length) | `R` | 128, 256, 512, 1024 |
| Crossbar cols (output length) | `C` | 64, 128, 256 |
| Input precision | `N_in_bits` | 8 |
| DPE-to-fabric bus width | `W_DPE` | 40 bits (= 5 int8 / cycle) |
| Read-BRAM word width | `W_BRAM` | 40 bits (default) |
| ACAM pipeline depth | `COMPUTE_CYCLES` | 3 (DIMM exp/log) or 44 (FC ADC) |

**Important assumption used throughout:** the read-BRAM and write-BRAM are
separate memory units. Loading the next pass's input does **not** contend
with draining the current pass's output. This is the design assumption of
the W=16 DIMM top and is consistent with having per-stage BRAM buffers
between pipeline stages.

## 3. BRAM data layouts

For 8-bit inputs there are two distinct layouts with identical total
storage but very different access patterns.

### 3.1 Layout A — natural int8 packed (load-then-compute)

Each read-BRAM word holds `W_BRAM / 8` consecutive int8 inputs.

Toy example: `R = 4` inputs, each 8-bit, with `W_BRAM = 8` (one int8
per word):

```
        x[0] = 10100101 (0xA5)
        x[1] = 00111100 (0x3C)
        x[2] = 11111111 (0xFF)
        x[3] = 00010000 (0x10)

read-BRAM  (word width = 8 bits = one int8)
         ┌───────────────────┐
  addr 0 │ 10100101  = x[0]  │
  addr 1 │ 00111100  = x[1]  │
  addr 2 │ 11111111  = x[2]  │
  addr 3 │ 00010000  = x[3]  │
         └───────────────────┘
```

**Load flow.** The outer FSM reads the vector word-by-word into the
DPE's internal `input_buffer[0..R-1]`. With `W_DPE = 40` (5 int8 /
cycle) this takes `⌈R / 5⌉` cycles. Once the buffer is full, the DPE
bit-slices the stored int8 values **internally** and fires
`N_in_bits` analog cycles. From the outside, the load phase is
**one block**, independent of `N_in_bits`.

**Pros:** upstream writers can emit natural int8 with no transposition.
**Cons:** load takes `R / 5` cycles regardless of precision; no
bit-serial speedup.

### 3.2 Layout B — bit-plane transposed (streaming bit-serial)

Each read-BRAM word holds **one bit-plane of multiple inputs**, i.e.
the same bit-position across a group of input rows.

Same toy example, with `W_BRAM = R = 4` (one full bit-plane per
word):

```
read-BRAM  (word width = 4 bits = R; one bit-plane per word)
              x[3] x[2] x[1] x[0]
            ┌─────────────────────┐
  addr 0    │  0    1    0    1   │   ← bit 0 of every input
  addr 1    │  0    1    0    0   │   ← bit 1
  addr 2    │  0    1    1    1   │   ← bit 2
  addr 3    │  1    1    1    0   │   ← bit 3
  addr 4    │  0    1    1    0   │   ← bit 4
  addr 5    │  0    1    1    1   │   ← bit 5
  addr 6    │  0    1    0    0   │   ← bit 6
  addr 7    │  0    1    0    1   │   ← bit 7
            └─────────────────────┘
```

Each BRAM word **is** a bit-slice: one read fills all `R` word-lines,
the crossbar fires, the shift-add accumulates with weight `2^k`, and
the next bit-plane is fetched.

For `R > W_BRAM` one bit-plane spans `⌈R / W_BRAM⌉` BRAM words. With
`R = 512` and `W_BRAM = 40`:

```
read-BRAM (word width 40, one bit-plane spans 13 consecutive words)
           ┌───────────────────────────────────────┐
  addr  0  │ bit 0 of x[  0 .. 39]                 │ ┐
  addr  1  │ bit 0 of x[ 40 .. 79]                 │ │ bit-plane 0
  ...      │                                       │ │ (13 words × 40 = 520 ≥ 512)
  addr 12  │ bit 0 of x[480 ..519]  (8-bit pad)    │ ┘
  addr 13  │ bit 1 of x[  0 .. 39]                 │ ┐
  ...      │                                       │ │ bit-plane 1
  addr 25  │ bit 1 of x[480 ..519]                 │ ┘
  ...      │                                       │
  addr 91  │ bit 7 of x[  0 .. 39]                 │ ┐
  ...      │                                       │ │ bit-plane 7
  addr 103 │ bit 7 of x[480 ..519]                 │ ┘
           └───────────────────────────────────────┘
```

Total read-BRAM usage: `N_in_bits · ⌈R / W_BRAM⌉ = 8 × 13 = 104`
words (vs `⌈R × 8 / W_BRAM⌉ = 103` for Layout A at the same width).
Same capacity to within one pad word.

**Pros:** each BRAM read directly feeds a bit-plane to the crossbar.
When `W_BRAM ≥ R`, the entire load phase is exactly `N_in_bits`
cycles, independent of `R`.
**Cons:** upstream writers must emit bit-sliced output, or a
transpose block must sit between them and the read-BRAM.

### 3.3 Storage cost comparison

For `R` inputs at `N_in_bits` precision, both layouts store
`R · N_in_bits` bits. Whether that occupies `⌈R · N_in_bits / W_BRAM⌉`
words or `N_in_bits · ⌈R / W_BRAM⌉` words differs only by the
worst-case padding overhead — at most `N_in_bits − 1` words. Storage
is essentially equal; **the layouts trade compute latency for write-side
complexity**.

## 4. Transpose logic

Layout B requires the input to arrive in bit-plane form. Producers in
our pipeline (Q/K/V projection DPEs, previous DIMM stages, upstream FC
layers) naturally emit **bit-parallel int8** — one natural-ordered int8
per bus word. A transpose block is therefore required between producer
and Layout-B read-BRAM.

### 4.1 Corner-turn buffer

Conceptually a 2-D register array of `R × N_in_bits` single-bit cells:

```
Transpose buffer (R rows × N_in_bits cols):
              bit 7    bit 6   ...    bit 0
            ┌─────────────────────────────────┐
   row 0    │ b7(x0)  b6(x0)   ...   b0(x0)   │
   row 1    │ b7(x1)  b6(x1)   ...   b0(x1)   │
   ...      │                                 │
   row R-1  │ b7(x_{R-1})  ...      b0(x_{R-1})│
            └─────────────────────────────────┘
              ▲ column read  (N_in_bits columns)
              │ = R bits wide
```

**Write side** (from producer): row-major, `W_DPE / 8` int8 values per
cycle. Each incoming word fills `W_DPE / 8` consecutive rows with all
`N_in_bits` bits of each row populated in parallel. Fill time:
`⌈R / (W_DPE / 8)⌉` cycles.

**Read side** (to read-BRAM / crossbar): column-major, one column
(= one bit-plane = `R` bits) per cycle, `N_in_bits` reads total.

### 4.2 FPGA implementation — shift-register array

The natural FPGA implementation is a set of `R` horizontal
shift-registers of length `N_in_bits`. The producer drives a small
number (`W_DPE / 8`) of registers per cycle with their 8 bits latched
in parallel; the bit-plane read taps the MSB (or a muxed bit-position)
of every shift-register simultaneously to form the `R`-bit plane.

This implementation uses `R · N_in_bits` flip-flops (4 096 for
`R = 512, N_in_bits = 8`) and no BRAM. On a modern FPGA that is a
trivial fraction of total CLB resources.

### 4.3 Area and latency cost

| Resource | Per transpose buffer | For full 64-DPE W=16 DIMM (worst case) |
|---|---|---|
| Flip-flops | `R · N_in_bits` (= 4 096 for R=512) | `≈ 512 × 8 × 64 ≈ 260 k` FFs |
| Routing | `R`-bit bus from buffer to read-BRAM | dominant cost on FPGA |
| Latency | Hidden behind previous pass's compute | Effective 0 cycles in steady state |

The transpose fill time is `⌈R / (W_DPE/8)⌉` cycles. With `R = 512`
and `W_DPE = 40` (5 int8 / cycle) that is 103 fill cycles. A single
DPE pass itself takes `O(R + C)` cycles, so a properly pipelined
design can always overlap transpose-fill with the prior pass's
analog + output activity → **zero added latency** in steady state.

## 5. Multi-pass pipelined timing

A single DPE pass is `Load → Analog-fire (bit-serial) → Output`. When
multiple passes run back-to-back (e.g. `M` output rows of a GEMM), the
passes can overlap:

```
     |——— Load (L) ———|—— Output (O) ——|
                       |L_next starts here  (next pass's input buffer is free)
                       |——— L_next ———|—— O_next ——|
                                        …
```

- **Single pass standalone:** `T_single = L + O`.
- **Steady-state per additional pass:** `T_steady = max(L, O)` — the
  input-buffer fill of the next pass runs in parallel with the output
  drain of the current pass.
- **Total `M`-pass time:**
  ```
  T_total = L + max(L, O) · (M − 1) + O
  ```

### Two regimes

| Regime | Which phase bottlenecks | Layout A typical? | Layout B typical? |
|---|---|---|---|
| Load-bound | `L > O` | Yes for small `C`, large `R` | No, load is tiny |
| Output-bound | `O > L` | Rare | Yes when `W_BRAM ≥ R` |

## 6. Analytical cycle formulas

All formulas in cycles. Handshake and drain contribute an additional
O(constant) cycles per pass (typically ~4 cycles of `reg_full →
nl_dpe_control → VMM-trigger → drain`), omitted below for clarity.

### 6.1 Layout A — natural-packed load-then-compute

```
L_A =  ⌈R · 8 / W_DPE⌉              ← load all R int8 into internal buffer
      +  N_in_bits                  ← bit-serial analog fire drain (overlapped w. ADC)
O   =  ⌈C · 8 / W_DPE⌉              ← stream C int8 outputs
```

Single pass: `T_A = L_A + O`.

### 6.2 Layout B — bit-plane-transposed streaming

```
L_B =  N_in_bits · ⌈R / W_BRAM⌉     ← one bit-plane per ⌈R/W_BRAM⌉ BRAM reads,
                                        each fires one analog cycle
O   =  ⌈C · 8 / W_DPE⌉              ← same output cost
```

When the BRAM port width matches the crossbar row count (`W_BRAM ≥ R`),
`L_B = N_in_bits`. When narrower, `L_B` is a multiple of `N_in_bits`.

### 6.3 Worked example — 512×128 crossbar, 8-bit, W_BRAM = 40, W_DPE = 40

| Phase | Layout A | Layout B |
|---|---:|---:|
| Load (`L`) | `⌈512·8 / 40⌉ = 103` | `8 · ⌈512/40⌉ = 8 · 13 = 104` |
| Output (`O`) | `⌈128·8 / 40⌉ = 26` | `26` |
| **Single pass** | **129** | **130** |

With this BRAM width (40 = 5 × 8 bits, narrower than `R`) Layout B
gives **no speedup** over Layout A — because each bit-plane still needs
13 BRAM reads to assemble 512 bits. Layout B only wins when the BRAM
port is provisioned to return a full bit-plane per read:

| `W_BRAM` | Layout A load | Layout B load | Layout B / A |
|---:|---:|---:|---:|
| 40   | 103 | 104 | 1.01× |
| 64   | 64  | 64  | 1.00× |
| 128  | 32  | 32  | 1.00× |
| 256  | 16  | 16  | 1.00× |
| **512** | **103** *(wider BRAM does not help Layout A because the bus is still W_DPE=40)* | **`8 · 1 = 8`** | **≈ 13×** |

The last row is the operating point where Layout B pays off: a
dedicated `R`-wide read-BRAM port, one bit-plane per cycle, load phase
collapses from 103 → 8 cycles.

**Multi-pass steady state (M passes):**

```
Layout A @ W_BRAM=40 :  T = 103 + max(103, 26)·(M-1) + 26 = 103·M + 26
Layout B @ W_BRAM=40 :  T = 104 + max(104, 26)·(M-1) + 26 ≈ 104·M + 26
Layout B @ W_BRAM=512:  T =   8 + max(  8, 26)·(M-1) + 26 = 26·M + 8   ← now output-bound
```

At `W_BRAM = R = 512` the system becomes output-bound (each pass
takes 26 cycles to drain), and the next optimisation is widening the
output bus `W_DPE`.

## 7. Design-choice guidance

| Design goal | Preferred layout | BRAM port width | Transpose cost |
|---|---|---|---|
| Minimum HW complexity, modest cycles | Layout A | `W_BRAM = W_DPE` | None |
| Maximum per-pass throughput | Layout B | `W_BRAM ≥ R` | `R · N_in_bits` FFs per DPE input |
| Mixed (some layers load-bound, some output-bound) | Layout B with `W_BRAM = R/2` | Moderate | Same transpose, narrower BRAM |

A full DSE would sweep `(W_BRAM, W_DPE, layout)` against target
per-pass cycle and CLB/FF budget. Our current W=16 DIMM top sits at
`(Layout A, W_BRAM = W_DPE = 40)` — the minimum-complexity point.
`dse_experiment_plan.md` will be extended in a future pass to add a
layout/width sweep.

## 8. What is implemented today

- The RTL (generators in `nl_dpe/`) implements **Layout A**. The DPE
  behavioural stub (`fc_verification/dpe_stub.v`) has an
  `input_buffer[0..KERNEL_WIDTH-1]` register that is filled
  word-by-word from the upstream BRAM before any compute begins.
- The simulator (`azurelily/IMC/peripherals/fpga_fabric.py::gemm_log`)
  uses the same Layout A assumption in its per-pass cycle model
  (`feed = read_cycles + sram_read_lat · K_id`).
- Multi-pass pipeline overlap: the simulator's attention pipeline uses
  `fill + (S − 1) · steady + drain` with `steady = max(stage per-row)`,
  which corresponds to §5's steady-state formula at the **stage** level
  (between DIMM stages), not the **pass** level (within a stage).
  Within-stage pass overlap is the next modelling refinement.
- The per-stage cycle alignment measured in Phase I.2 + J
  (`fc_verification/results/dimm_top_w16_alignment_log.txt`) —
  score 260 / softmax 27 / wsum 274 cycles — is a Layout A number.
  Moving to Layout B would drop the score and wsum load portions
  substantially (see §6.3).

## 9. Terminology quick reference

| Term | Meaning |
|---|---|
| Bit-plane / bit-slice | The value of one bit position (e.g. bit 3) across all `R` input rows — exactly `R` bits |
| Load phase | BRAM → input buffer / crossbar input. Layout A: fill buffer. Layout B: one bit-plane per fire. |
| Fire | One analog crossbar activation = one bit-plane worth of MAC |
| Pass | One full `(R, C)` MAC producing `C` 8-bit outputs from `R` 8-bit inputs (= `N_in_bits` fires) |
| Multi-pass | Running `M` passes back-to-back (e.g. one per row of an M-row GEMM) |
| Corner-turn / transpose | Changing the storage order from natural-packed int8 to bit-planes |

## 10. Cross-references

- `paper/methodology/attention_dimm_mapping.md`: mapping of attention
  stages to primitives; §9 will reference this document for per-pass
  dataflow.
- `fc_verification/VERIFICATION.md`: measured cycle counts; Phase I.2
  and J results are Layout-A numbers.
- `fc_verification/results/dimm_top_w16_alignment_log.txt`: rolling
  cycle-alignment detail, chronological.
- `azurelily/IMC/peripherals/fpga_fabric.py`: sim's `gemm_log`
  implements the Layout A analytical model.
