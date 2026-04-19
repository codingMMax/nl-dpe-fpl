# Per-Pass DPE Pipeline Model — Dataflow, BRAM Layout, Transpose

**Date:** 2026-04-18 (initial), revised 2026-04-19 (multi-pass regime analysis)

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
`M` passes run back-to-back (e.g. `M` output rows of a GEMM), the passes
*can* overlap. Three regimes bracket the design space:

- **Regime A (today's sim)** — fully serial, `T = M·(L + O)`.
- **Regime B (today's RTL and the committed target for the simulator)** —
  single-buffered input + output with realistic overlap between the
  next pass's load and the current pass's drain.
- **Regime C (archived alternative, not chosen)** — double-buffered
  input *and* output. Retained in §5.4 as a design-space reference
  only; see §5.6 for the committed target.

The committed path for this project is **Layout A + Regime B**
(§5.3.1). The difference between regimes is substantial for Layout B
but collapses to a no-op for Layout A in steady state — see the
worked example in §5.5.

### 5.1 Within-pass pipeline (the building block)

Before looking at inter-pass overlap, fix the per-pass cycle cost.

**Layout B — bit-plane streaming (read/fire alternation).** Each
bit-plane is one BRAM read of `L_read_0 = ⌈R / W_BRAM⌉` cycles that
feeds one analog fire of 1 cycle. When `L_read_0 > 1`, fire `k` can be
issued during the *start* of read `k+1`, so the pipeline cost per
bit-plane is `max(L_read_0, 1) = L_read_0` cycles:

```
per bit-plane, pipelined:   ┌── read (L_read_0 cyc) ──┐
                                                     └─ fire (1 cyc, overlaps
                                                        with next read's start)
```

Running `N_in_bits = 8` bit-planes:

```
 read 0: ████████████▏
 fire 0:             █   (1 cyc, hidden in read 1)
 read 1:             ████████████▏
 fire 1:                         █
 ⋮
 read 7:                                     ████████████▏
 fire 7:                                                 █   ← no next read to
                                                             hide behind

L_B = (N_in_bits − 1)·L_read_0   +   L_read_0 + 1
       ↑ 7 iterations pipelined    ↑ last read + standalone fire
    = N_in_bits · L_read_0 + 1
```

For 512×128, W_BRAM = 40: `L_read_0 = ⌈512/40⌉ = 13`, so
`L_B = 8·13 + 1 = 105 cycles`. The trailing `+1` is the final fire
that has no subsequent read to overlap with. (§6.2 states the
approximation `L_B ≈ N_in_bits · L_read_0 = 104`; the `+1` correction
here is exact and material when the drain is comparable to one read
chunk — see §5.3.)

**Layout A — monolithic buffer fill, then bit-serial fires.** One long
BRAM stream loads all `R` int8 values into the on-die input buffer, then
`N_in_bits` bit-serial fires drive the crossbar from the buffer:

```
L_A = ⌈R · 8 / W_DPE⌉ + N_in_bits
```

For 512×128, W_DPE = 40: `L_A = ⌈4096/40⌉ + 8 = 103 + 8 = 111 cycles`.

**Output drain** is independent of layout:

```
O = ⌈C · 8 / W_DPE⌉
  = ⌈128·8 / 40⌉ = 26 cycles   (C=128, W_DPE=40)
```

### 5.2 Regime A — fully serial (no inter-pass overlap)

Each pass runs to completion before the next one starts:

```
T_serial(M) = M · (L + O)
```

Upper bound. This is what the simulator's `gemm_log` emits today
(it multiplies `cycles_per_pass` by `M`, see §8). For 512×128
Layout B: `T_serial(M) = 131·M`; at M=8, 1 048 cycles.

### 5.3 Regime B — single-buffered (today's RTL)

**Assumption:** one input buffer (holds the bit-plane currently being
fired) and one shift-add / output register. The next pass cannot fire
until the current pass's drain completes, because the shift-add is
still streaming pass `k`'s output. Read-BRAM is separate from
write-BRAM (§2), so pass `k+1`'s BRAM reads *can* start during pass
`k`'s drain — but only until the one input-buffer slot is full, at
which point further reads stall.

**Pass-boundary timeline (512×128, Layout B, W_BRAM = 40, cycle 0 = pass k's read 0 start):**

```
cycle:    92-104   105        106 … 117   118 … 130    131      144  …   222    223 … 248
                                                                      
pass k     ┌──read 7──┐
           │ (13 cyc) │
                    fire 7 ██
                    drain:  ══════════════════════════  (26 cyc, ends 130)

pass k+1               read 0 ══════════  ╳╳╳╳╳╳╳╳╳╳╳╳  ← input buf full,
                       (13 cyc, overlaps)  13-cyc stall   shift-add still draining
                                                       fire 0 ██   ← first free
                                                                      slot: cycle 131
                                                       read 1 ══════════
                                                              fire 1 ██
                                                       ⋮
                                                                   fire 7 ██ (cyc 222)
                                                                   drain: ═════════════
                                                                          (26 cyc, ends 248)
```

**What costs cycles at the boundary.** Exactly one BRAM read (read 0,
13 cycles) hides inside pass `k`'s drain. The remaining
`O − L_read_0 = 26 − 13 = 13` cycles of drain block:

- pass `k+1`'s fire 0 (needs the shift-add which is still draining),
  **and**
- all further reads (input buffer is occupied by bit-plane 0 waiting
  to be fired, nothing can take its place).

So pass `k+1`'s fire 0 only starts at drain-end + 1.

**Per-pass increment**, measured fire-7(`k`) → fire-7(`k+1`):

```
  (N_in_bits − 1)·L_read_0 + max(L_read_0, O) + 1            ← general form
= L_B − L_read_0 + max(L_read_0, O)
= L_B + max(0,  O − L_read_0)
```

For 512×128 Layout B: `steady_sb = 105 + max(0, 26−13) = 118 cycles`.

**Total M-pass latency:**

```
T_sb(M) = L_B + steady_sb · (M − 1) + O
        = 105 + 118·(M − 1) + 26
        = 118·M + 13                  (512×128, Layout B, W_BRAM=40)
```

#### 5.3.1 Layout A pass boundary (worked example)

Layout A differs structurally from Layout B: the load is **monolithic**
(all `R` int8 streamed into the input buffer through a single contiguous
BRAM burst) and the fires are a separate **contiguous 8-cycle block**
that happens only *after* the full buffer is loaded — the DPE bit-slices
the already-loaded buffer internally, so each fire needs only 1 cycle
of crossbar activation and no additional BRAM access.

Per-pass cost breakdown (512×128, W_DPE = 40):

```
  Fill : ⌈R · 8 / W_DPE⌉    = ⌈512·8/40⌉ = 103 cycles
  Fire : N_in_bits         = 8 cycles        (bit-slice internally)
  Drain:                     26 cycles

  L_A = 103 + 8 = 111 cycles                 (load + fires)
  T_single_A = 111 + 26 = 137 cycles
```

The fires can **not** overlap with the load — one analog fire needs
bit-`k` of **all** 512 inputs driven onto the word-lines in the same
cycle, so the buffer must be fully loaded before fire 0 can issue.

**Pass-boundary timeline (512×128, Layout A, W_DPE = 40, cycle 0 = pass k's load start):**

```
cycle:   0 ───────── 102   103 ─ 110    111 ─ 136    111 ─── 213   214 ─ 221    222 ─ 247
                                                                                
pass k load:  ═════════════════
              (103 cyc, BRAM → input_buffer)
pass k fires:                    ████████
                                 (8 cyc, internal bit-slice,
                                  no BRAM access)
pass k drain:                              ═══════════════
                                           (26 cyc, ends 136)

pass k+1 load:                             ╔════════════════════════════════════╗
                                           (103 cyc — starts at cyc 111,
                                            overlaps entire pass-k drain plus
                                            77 more cycles before fires begin)
pass k+1 fires:                                                        ████████
                                                                       (8 cyc, shift-add
                                                                        free since cyc 137)
pass k+1 drain:                                                                   ═══════════════
                                                                                  (26 cyc, ends 247)
```

**What costs cycles at the boundary.** Two resource claims to check:

1. *Input buffer*: pass k's fires (cycles 103–110) consume bit-planes
   0–7 of the buffer. Once fire 7 retires at cycle 110, the buffer is
   free and pass k+1 begins its monolithic fill at cycle 111.
2. *Shift-add / output register*: pass k drains until cycle 136. Pass
   k+1 needs the shift-add for its fires, which start at cycle 214.
   Since 136 < 214, the drain has fully completed long before the
   fires want the shift-add.

The drain never blocks anything because the monolithic load (103
cycles) is longer than the drain (26 cycles). **No inter-pass stall.**

**Per-pass increment**, measured fire-7(`k`) → fire-7(`k+1`): fire 7
of pass k at cycle 110, fire 7 of pass k+1 at cycle 110 + 111 = 221,
so `steady = 111 cycles = L_A exactly`.

**Why the formula simplifies for Layout A.** In the general Regime B
formula `steady = L + max(0, O − L_chunk)`, the "overlap chunk"
`L_chunk` is whatever portion of pass k+1's load can start running
while pass k is draining. For Layout B this was a single bit-plane
fetch (`L_read_0 = 13`); for Layout A the load is a single monolithic
burst that can overlap drain entirely, so `L_chunk = L_A`. Substituting:

```
steady_A_sb = L_A + max(0, O − L_A) = max(L_A, O)
```

which reduces to `L_A` whenever `L_A > O` — true in every realistic
config (`R·8/W_DPE + N_in_bits > C·8/W_DPE` iff `R > C − W_DPE · N_in_bits / 8`;
for our values this is `R > C − 40`, essentially always).

**Total M-pass latency (Regime B, Layout A):**

```
T_A_sb(M) = L_A + max(L_A, O) · (M − 1) + O
          = L_A · M + O
          = 111·M + 26                       (512×128, W_DPE=40)
```

| M | T_A(M) cycles |
|---:|---:|
| 1 | 137 |
| 2 | 248 |
| 4 | 470 |
| 8 | **914** |
| 64 | 7 130 |

**Contrast with Layout B at same W_BRAM=40 (see §5.3 main body):**

| | Layout A | Layout B |
|---|---:|---:|
| `L` (load + trailing fire) | 111 | 105 |
| Single pass | 137 | 131 |
| Inter-pass stall (single-buf) | **0 cyc** | 13 cyc |
| Regime B steady / pass | 111 | 118 |
| Regime B T(M=8) | **914** | 957 |

Layout B wins the single-pass comparison by pipelining reads and
fires, but Layout A wins the multi-pass comparison because its
monolithic load naturally hides the drain. (Layout B is archived as
a design-space reference in §§3.2, 4; see §5.7 for the committed
layout choice.)

### 5.4 Regime C — archived alternative (not chosen)

> **Archived.** The project has committed to Regime B under Layout A
> (§5.3.1). This section is retained as a design-space reference for
> the `max(L, O)` stall-free formula; it is NOT an implementation
> target. Ignore this section for the scope of Phase 1/2/3 work.

**Assumption (archived reference):** the input register is
double-buffered (archived; pass `k+1` fills buffer B while pass `k`
fires from buffer A), *and* the shift-add / output register is
double-buffered (archived; pass `k+1` accumulates into register B
while pass `k` drains register A). Pass `k+1`'s fire 0 no longer waits
for drain, so the next pass's load runs in full parallel with the
current pass's drain:

```
steady_db = max(L, O)
T_db(M)   = L + max(L, O) · (M − 1) + O
```

For 512×128 Layout B: `T_db(M) = 105·M + 26` (load-bound).

Wide-BRAM variant (`W_BRAM = R = 512`, `L_read_0 = 1`, `L_B = 9`):
`T_db(M) = max(9, 26)·M + 9 = 26·M + 9` — now output-bound, and the
system's next optimisation is widening `W_DPE`.

### 5.5 Summary table — 512×128, Layout A, W_DPE = 40, M = 8

| Regime | Formula | T(M=8) | Where it applies |
|---|---|---:|---|
| A — serial | `M · (L_A + O) = 137·M` | 1 096 | current sim `gemm_log` (pre-Phase 1) |
| B — single-buffered | `111·M + 26` | **914** | **current RTL + committed sim target (post-Phase 1)** |
| C — double-buffered | `111·M + 26` | 914 | archived reference only — for Layout A, B and C coincide because `L_A > O` |

For completeness, the Layout B numbers at the same per-pass operating
point are archived below as a design-space reference (Layout B is not
on the committed path; see §5.7):

| Regime | Layout B @ W_BRAM=40 | Layout B @ W_BRAM=512 |
|---|---:|---:|
| A — serial | `131·M` = 1 048 | `35·M` = 280 |
| B — single-buffered | `118·M + 13` = 957 | `34·M + 1` = 273 |
| C — archived reference | `105·M + 26` = 866 | `26·M + 9` = 217 |

For Layout A the two active regimes (A and B) are the only ones that
matter: Regime B is what today's RTL already does, and Phase 1 moves
the simulator from Regime A to Regime B so sim and RTL agree.

### 5.6 Which regime to target

**Committed target:** Regime B for Layout A.

- Sim: Regime A today (pre-Phase 1); Regime B after Phase 1.
- RTL: Regime B today (no change needed).
- Regime A is a historical pre-Phase-1 state (pure sim artefact).
- Regime C is archived as a design-space reference (§5.4); it is not
  an implementation target.

| Hardware state | Input buffer | Output register | Regime | Status |
|---|---|---|---|---|
| Today's DPE stub + generator RTL | single | single | **B** | **committed RTL** — `steady = max(L, O) = L_A` for Layout A |
| Hypothetical double-buffer variant | double | double | C | archived reference only (§5.4) |

Phase I.2 + J (260 / 27 / 274 / 561 cyc) were measured against the
simulator running Regime A. The RTL itself is executing Regime B, so
closing the sim–RTL gap (Phase 1) is a sim-only change; after Phase 1
both sides run Regime B under Layout A and the tolerance tightens from
the legacy ≤ 20 cycles to ≤ 3 per stage / ≤ 5 end-to-end.

### 5.7 Target regime and the layout choice

**Committed path: Layout A + Regime B** (§5.3.1). Transpose block and
Regime C double-buffering are retired from the active plan (archived
in §§3.2, 4, 5.4 as design-space reference only). Today's RTL already
satisfies the Regime B model; Phase 1 only updates the simulator to
match it. Phases 2 and 3 are RTL/TB bug-fix passes under the tighter
≤ 3 / ≤ 5 tolerance — not architecture changes.

**Layout B is archived as a design-space alternative** (see §3.2 for
the bit-plane storage layout, §4 for the transpose hardware, §6.3 for
the per-pass cost tables). It is retained in the paper to show the
full design-space reach but is NOT a separately-costed alternative in
the active plan. A future DSE pass could resurrect it if the project
ever needs the wide-BRAM throughput; until then the transpose block,
wider BRAM port, and Layout-B cycle tables are reference material only.

## 6. Analytical cycle formulas

All formulas in cycles. Handshake and drain contribute an additional
O(constant) cycles per pass (typically ~4 cycles of `reg_full →
nl_dpe_control → VMM-trigger → drain`), omitted below for clarity.

### 6.1 Layout A — natural-packed load-then-compute

```
L_A =  ⌈R · 8 / W_DPE⌉              ← load all R int8 into internal buffer
      +  N_in_bits                  ← bit-serial analog fires (N_in_bits cycles)
O   =  ⌈C · 8 / W_DPE⌉              ← stream C int8 outputs
```

Single pass: `T_A = L_A + O`. The `+ N_in_bits` term must be counted
explicitly — the 8 bit-serial fires run *after* the full buffer is
loaded and *before* the shift-add is ready to drain, so they add to
the critical path.

### 6.2 Layout B — bit-plane-transposed streaming

```
L_read_0 = ⌈R / W_BRAM⌉              ← cycles to fetch one bit-plane
L_B      = N_in_bits · L_read_0 + 1  ← 7 iters pipelined + last read + trailing fire
O        = ⌈C · 8 / W_DPE⌉           ← same output cost
```

When the BRAM port width matches the crossbar row count (`W_BRAM ≥ R`),
`L_read_0 = 1` and `L_B = N_in_bits + 1`; when narrower, `L_B` scales
as `N_in_bits · ⌈R / W_BRAM⌉ + 1`. See §5.1 for the per-bit-plane
derivation of the trailing `+1` term.

### 6.3 Worked example — 512×128 crossbar, 8-bit, W_BRAM = 40, W_DPE = 40

| Phase | Layout A | Layout B |
|---|---:|---:|
| Load (`L`) | `103 + 8 = 111` | `8 · 13 + 1 = 105` |
| Output (`O`) | `26` | `26` |
| **Single pass** (`L + O`) | **137** | **131** |

At this BRAM width (40 = 5 × 8 bits, narrower than `R`) Layout B
already has a *lower per-pass* cost than Layout A — the bit-plane
streaming pipeline beats the "load everything, then fire 8 times"
schedule by `L_A − L_B = 6` cycles. Layout B still only pays off
*dramatically* when the BRAM port is provisioned to return a full
bit-plane per read:

| `W_BRAM` | Layout A load `L_A` | Layout B load `L_B` | Layout B / A |
|---:|---:|---:|---:|
| 40   | 111 | 105 | 0.95× |
| 64   | `64 + 8 = 72`  | `8·8 + 1 = 65`  | 0.90× |
| 128  | `32 + 8 = 40`  | `8·4 + 1 = 33`  | 0.83× |
| 256  | `16 + 8 = 24`  | `8·2 + 1 = 17`  | 0.71× |
| **512** | `103 + 8 = 111` *(wider BRAM does not help Layout A — the bus is still W_DPE=40)* | **`8·1 + 1 = 9`** | **≈ 12×** |

The last row is the operating point where Layout B wins decisively: a
dedicated `R`-wide read-BRAM port, one bit-plane per cycle, load
collapses from 111 → 9 cycles.

**Multi-pass steady state (M passes)** — committed path is Layout A,
Regime B. The table below gives Layout A (committed) and lists the
archived Layout B entries for design-space reference:

| Layout | `W_BRAM` | Regime B (committed for Layout A) |
|---|---:|---|
| A | 40  | **`111·M + 26`** *(steady = max(L_A, O) = 111 because L_A > O)* |

*Archived Layout B reference rows (not on the committed path, see §5.7):*

| Layout | `W_BRAM` | Regime B | Regime C (archived reference only) |
|---|---:|---|---|
| B | 40  | `118·M + 13` | `105·M + 26` |
| B | 512 | `34·M + 1`   | `26·M + 9` |

**Observations.**

- For Layout A (committed), the monolithic buffer fill (103 cyc) is
  already much longer than the drain (26 cyc), so drain fully
  overlaps the next pass's load. The `111·M + 26` formula is what
  Phase 1 commits the simulator to emit.
- (Archived) For Layout B at W_BRAM = 40, the single-buffered regime
  pays a `O − L_read_0 = 13` cycle penalty per pass boundary because
  only the first read (13 cyc) can overlap the drain.
- (Archived) For Layout B at W_BRAM = R = 512, the system becomes
  output-bound in the stall-free limit.

Earlier drafts of this doc quoted `103·M + 26` (Layout A), `104·M + 26`
(Layout B @ 40), and `26·M + 8` (Layout B @ 512) — all of which
silently dropped the trailing `+N_in_bits` fire of Layout A and the
trailing `+1` fire of Layout B. The Layout A number above is the
corrected committed value.

## 7. Design-choice guidance

**Committed path:** Layout A, Regime B (single-buffered, today's RTL).
Our W=16 DIMM top sits at `(Layout A, W_BRAM = W_DPE = 40)` — the
minimum-complexity point, and the configuration all Phase 1/2/3 work
targets.

Layout B and wider BRAM variants are archived in §§3.2, 4, 5.7 for
paper design-space reference. They are not separately-costed
alternatives in the active plan; a future DSE pass could add a
`(W_BRAM, W_DPE, layout)` sweep if the project ever wants to explore
the wide-BRAM Layout B throughput win.

## 8. What is implemented today

- The RTL (generators in `nl_dpe/`) implements **Layout A** with
  single-buffered input and output registers → **Regime B** per §5.3
  (the committed path). The DPE behavioural stub
  (`fc_verification/dpe_stub.v`) has an
  `input_buffer[0..KERNEL_WIDTH-1]` register that is filled
  word-by-word from the upstream BRAM before any compute begins, and
  the shift-add / output register is a single instance.
- The simulator (`azurelily/IMC/peripherals/fpga_fabric.py::gemm_log`)
  computes `cycles_per_row = cycles_per_pass` and sums them — this is
  **Regime A (serial)** per §5.2, which is the pre-Phase-1 state.
  Phase 1 (§8.1) moves the sim to **Regime B** under Layout A so
  sim and RTL agree. For Layout A at 512×128 the pre-Phase-1
  sim–RTL gap is bounded by `(steady_B − L_A − O)·(M−1) = 0·(M−1) = 0`
  cycles (because Layout A's steady state is `L_A` in both Regime A
  and Regime B — Regime A differs only by counting `+O` on every
  pass rather than once at the end). This is small enough that the
  legacy ≤ 20 cycle tolerance in Phase I.2 + J held pre-Phase 1.
- Multi-pass pipeline overlap: the simulator's attention pipeline uses
  `fill + (S − 1) · steady + drain` with `steady = max(stage per-row)`,
  which applies §5's overlap formula at the **stage** level (between
  DIMM stages), not the **pass** level (within a stage). Within-stage
  pass overlap is handled by the Phase 1 sim change.
- The per-stage cycle alignment measured in Phase I.2 + J
  (`fc_verification/results/dimm_top_w16_alignment_log.txt`) —
  score 260 / softmax 27 / wsum 274 cycles — is a Layout A number
  measured against the Regime-A simulator. Post-Phase 1 it will be
  re-measured against the Regime B simulator under the tighter
  ≤ 3 per stage / ≤ 5 end-to-end tolerance.

### 8.1 Phases (opened 2026-04-18, revised 2026-04-19)

The P4 track is three phases, all under **Layout A + Regime B**. No
RTL architecture changes are needed — the current RTL already runs
Regime B. Phases 2 and 3 are RTL/TB bug-fix passes under the tighter
tolerance, not redesigns.

**Phase 1 — Simulator Regime B swap (Layout A)**
Change `gemm_log` in `azurelily/IMC/peripherals/fpga_fabric.py` from
`M · cycles_per_pass` (Regime A) to the **Regime B** Layout A
formula:

```
T(M) = L_A · M + O
```

For 512×128 Layout A, `L_A = 111` and `O = 26`, so `T(M) = 111·M + 26`.
Preserve the `M = 1` case as `L_A + O = 137` exactly (today's
behaviour is the `M = 1` special case). Analytically verify on GEMV
(same-input-across-passes) and GEMM (different-input-per-pass) test
cases using the §5.5 Layout A row as the expected value.

| Case | Expected T(M) |
|---|---|
| 512×128 Layout A, any M | `111·M + 26` |

**Phase 2 — FC RTL re-verify under ≤3/≤5 tolerance**
No RTL architecture changes. Re-run FC verification against the
Regime-B sim from Phase 1 and add a new GEMM RTL workload. Because
both sim and RTL are now Regime B under Layout A, tighten the
tolerance from the legacy ≤ 20 cycles to **≤ 3 per stage / ≤ 5
end-to-end**. Fix any TB / generator bugs that surface under the
tighter tolerance — e.g. the wsum-probe NBA race documented in
`fc_verification/VERIFICATION.md` Phase J.

**Phase 3 — DIMM RTL re-verify under ≤3/≤5 tolerance**
No RTL architecture changes. Re-align DIMM Phase I.2 + J end-to-end
(`mac_qk`, `softmax`, `mac_sv`) against the Regime-B sim under the
same ≤ 3 / ≤ 5 tolerance. Fix any residual RTL or TB bugs surfaced
by the tighter bound. The `260 / 27 / 274 / 561` baseline is what
the refreshed numbers should stay close to; the goal is closing each
delta to within tolerance, not changing the pipeline architecture.

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
