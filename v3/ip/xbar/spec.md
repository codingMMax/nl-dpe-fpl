# `dpe` — the NL-DPE crossbar, row-parallel interface (v3 charter)

Status: 2026-10-22, written for the bit-plane pass (user decisions of 2026-10-21/22, `LEDGER.md`). The simulator's
crossbar model (`simulator/core/crossbar.py`) implements it. 2026-10-23: both RTL files implement it
(`rtl/sim/dpe_nldpe.v`, the behavior model, and `rtl/vtr/dpe_blackbox.v`, the VTR port contract; forked from v2 on your
go, `tools/copies.py` FORKED).

This charter replaces v2's interface and timing (`git show v2-final:v2/sim/spec/dpe_nldpe.md` §4–§5). It keeps v2's
numeric contract unchanged (§2 and §6 there): int8 stationary weights, int8 inputs, an exact integer MAC into an
int32 accumulator, and the integer output-stage forms with the `trunc8` low-byte rule.

## 1 Function

One pass computes, for the input vector $x$ (int8, $R$ rows) and the weights $W$ (int8, $R \times C$):

$$
y = x\,W \ \text{(int32, exact)}, \qquad \text{out} = \mathrm{form}(y) \ \text{(one byte per column)}
$$

- **Forms:** REGULAR $\mathrm{trunc8}(y)$, ACTIVATION $\mathrm{trunc8}(\max(y, 0))$,
  EXP $\mathrm{trunc8}(1 + y + \lfloor y^2/2 \rfloor)$, LOG $\mathrm{trunc8}(y - 1)$.
- **Extent:** rows $\ge$ `ROWS_USED` read as zero. Columns $\ge$ `COLS_USED` drain as zero bytes; the first
  `COLS_USED` bytes are the pass's window.
- **Two rows per column:** the DIMM (`ip/dimm/spec.md` §4, 2026-10-23) preloads the identity stacked twice where
  $R \ge 2I$ (rows $r$ and $I + r$ to column $r$, `ROWS_USED` = $2I$), so the pass's row sum adds its two log
  operands before EXP; the EXP form has period 256 in $y$, so the byte equals the one from the truncated sum.
- **Values:** unchanged from v2. Only the interface and the timing are new.

## 2 Parameters

| parameter | model name | meaning |
|---|---|---|
| `KERNEL_WIDTH` | $R$ | rows |
| `NUM_COLS` | $C$ | columns |
| `PRECISION` | $P$ | bits per input and output element, 8 |
| `ROWS_USED`, `COLS_USED` | | the pass extent (P29), $1 \le$ `ROWS_USED` $\le R$, $1 \le$ `COLS_USED` $\le C$ |

`DPE_BUF_WIDTH` (40) is removed: the data ports are $R$ and $C$ bits wide.

## 3 Ports

| port | width | dir | meaning |
|---|---|---|---|
| `data_in` | $R$ | in | during a burst, bit $r$ = bit $b$ of $x_r$, plane $b$ in the burst's cycle $b$ ($b = 0 \ldots P - 1$, LSB first) |
| `w_buf_en` | 1 | in | the plane strobe: high in each of a burst's $P$ cycles |
| `MSB_SA_Ready` | 1 | out | a burst may start in this cycle |
| `data_out` | $C$ | out | during the drain, bit $c$ = bit $b$ of output byte $c$, plane $b$ in the drain's cycle $b$ |
| `reg_full` | 1 | out | high in each of the drain's $P$ cycles (the drain's word valid) |
| `dpe_done` | 1 | out | pulse one cycle after the drain's last plane |
| `shift_add_done` | 1 | out | pulse one cycle after the MSB plane fires |
| `nl_dpe_control` | 2 | in | the output form (REGULAR 0, ACTIVATION 1, EXP 2, LOG 3), latched with the weights |
| `shift_add_control`, `shift_add_bypass`, `load_output_reg` | 1 each | in | reserved, unused (v2) |
| `load_input_reg` | 1 | in | the weight strobe, simulation twin only; tied 0 in every VTR twin |
| `shift_add_bypass_ctrl` | 1 | out | tied 0 (v2) |
| `clk`, `reset` | 1 each | in | |

So a crossbar has $R + 7$ input pins and $C + 5$ output pins besides the clock and reset. These are the pins of the
row-parallel tile (`docs/experiment.md` §5.1).

**Weights** are preloaded, never strobed (`PLAN.md` R8). The testbench writes every `weights[r·C + c]` and `mode_q`
through the hierarchy after each reset. The strobe path (`load_input_reg`, weight byte on `data_in[7:0]`) exists only
in the simulation twin.

## 4 The pass

**Burst rule.**
- A burst is $P$ consecutive cycles with `w_buf_en` high. Its first cycle has `MSB_SA_Ready` high.
- Planes arrive LSB first, one per cycle, without gaps.
- A strobe while `MSB_SA_Ready` is low and no burst is in progress, a gap inside a burst, or a ninth strobe is a
  producer error. The behavior model sets a sticky simulation-only `err` flag, and the testbenches check it.

**Timing rule.** For a burst ready at $t_{rdy}$ (event names as in the simulator):

```text
load_start    = max(t_rdy, acam_prev + 1)        (the first pass: max(t_rdy, 0))
load_end      = load_start + P - 1               (the last plane arrives)
compute_start = load_start + 1                   (each plane fires the cycle after it arrives)
msb_fire      = compute_start + P - 1            (= load_start + P)
acam          = max(msb_fire + 1, drain_end_prev + 1)
drain_start   = acam + 1
drain_end     = acam + P
next ready    = acam + 1                         (MSB_SA_Ready rises)
```

- **No add cycle:** each plane's partial sum is accumulated as it fires (LSB first, the MSB subtracted), and the
  ACAM fires the cycle after the MSB plane.
- **One accumulator, one output register:** the next burst may start only after the ACAM has freed the accumulator
  (`acam + 1`). The ACAM waits for the output register's previous drain.
- **The output register never binds.** `load_start` $\ge$ `acam_prev + 1` gives
  `acam` = `msb_fire + 1` = `load_start + P + 1` on every pass, while the previous drain ended at `acam_prev + P`.
  The `max` stays in the rule and in the model, and a model test asserts that it never binds.
- **The extent does not change the time.** Every pass loads $P$ planes and drains $P$ planes. `ROWS_USED` and
  `COLS_USED` change only the values (and the energy, which charges the input buffer per row used, §6).

With $P = 8$, one pass alone takes 18 cycles, and passes start every 10 cycles back to back:

```text
cycle   0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 ...
load    b0 b1 b2 b3 b4 b5 b6 b7       b0'b1'b2'b3'b4'b5'b6'b7'
fire       b0 b1 b2 b3 b4 b5 b6 b7       b0'b1'b2'b3'b4'b5'b6'b7'
acam                             A                             A'
drain                               d0 d1 d2 d3 d4 d5 d6 d7       d0'...
ready   1  1  1  1  1  1  1  1  0  0  1 (the second burst starts at 10)
```

For $M$ passes back to back from cycle 0, the last drain plane is at $10(M - 1) + 17$, so $T(M) = 10M + 8$.

**Status semantics.**
- `MSB_SA_Ready`: high from reset, and during a burst. Low from `load_start + P` until `acam`. High again from
  `acam + 1`. A producer that counts its own strobes (`att_ctrl`, `top/spec.md` §10 item 1) needs no change.
- `reg_full`: high exactly in `drain_start` … `drain_end`. Every module uses it as the drain's word valid.
- `dpe_done`: a pulse at `drain_end + 1`.
- `shift_add_done`: a pulse at `msb_fire + 1`.

## 5 The model

`simulator/core/crossbar.py`: `Crossbar` and `CrossbarGroup` (n crossbars in lock-step from one leader). One call
gives a pass's values, its events (the timing rule above) and its counts.

The model's closed form, used by its tests, is:
- passes back to back: `load_start`$_m$ = $\max(t_m, \text{load\_start}_{m-1} + P + 2)$;
- every event at a fixed offset from `load_start`: `compute_start` +1, `msb_fire` +$P$, `acam` +$P$+1,
  `drain_start` +$P$+2, `drain_end` +$2P$+1.

## 6 Energy and area

- **Energy per pass:** v2's prices (`simulator/platform/prices.py`). The whole array fires; the input buffer is
  charged per row used (P29); the output buffer and the drain per column. **The port traffic** (2026-10-23, user):
  the port is $R$ bits in and $C$ bits out per cycle, so a pass moves $P$ planes of the rows in use in and $P$ planes
  of every column out, priced per bit (`nl_dpe.json` `port`: the block-RAM access price per bit of its 40-bit port,
  0.0495 / 40 pJ, so the term is continuous with the pre-bit-plane 40-bit words without their rounding).
- **Area:** the row-parallel tile (`docs/experiment.md` §5.1): the height from the $R + 7$ / $C + 5$ pins, the width
  from the logic plus COFFE's local input crossbar.

## 7 The RTL change (the RTL pass, on your go)

`rtl/sim/dpe_nldpe.v`:
- `data_in` becomes $R$ bits and `data_out` $C$ bits; `DPE_BUF_WIDTH` goes.
- One plane register replaces the 8 bit-plane banks and the 5-byte corner-turn: plane $b$ fires the cycle after it
  arrives.
- The ACAM moves one cycle earlier (`acc_done` → `acam_fire` without the extra stage).
- The drain sends the output register's bit planes for $P$ cycles; columns $\ge$ `COLS_USED` drain 0.
- `MSB_SA_Ready` follows §4; the sticky `err` flag is added.

`rtl/vtr/dpe_blackbox.v`:
- It gets `KERNEL_WIDTH` and `NUM_COLS` parameters and the same port expressions as the behavior model, so
  `tools/check_twins.py` compares the two headers verbatim.
- The VTR twins keep bare `dpe` instances (no parameters). Each connects exactly $R$ and $C$ bits, and Parmys maps
  them onto the arch XML's `nl_dpe` block (`vtr/arch_rowpar/`, `data_in` $R$ pins, `data_out` $C$ pins).

## 8 Gates

- **The model:** `simulator/tests/testCrossbar.py` and `testTrace.py`:
  - the closed form of §5, hand-written;
  - the values against v2's corpora (the oracle is unchanged);
  - the never-binding output register;
  - the extent leaving the time unchanged.
- **GATE 2:** the modules' 64×64 corpora exercise every crossbar. A few crossbar-only cases at 64×64 and 128×64
  are your call at the docs review.
- **VTR:** the head smoke at 128×64 with `vtr/arch_rowpar/` (the crossbar count, Fmax, CLB, memory; 256×128 until
  2026-10-23).
