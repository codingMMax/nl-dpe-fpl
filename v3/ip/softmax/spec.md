# `softmax_online_top` — online softmax (v3 module spec)

Status: 2026-10-14, F3 / phase B, **validated**: GATE 2 18/18 at 64×64, VTR smoke at 128×128 (`LEDGER.md`).

**2026-10-22, the bit-plane pass** (§10 item 6):
- every port carries bit planes (`ip/xbar/spec.md`, `docs/architecture.md` §3.1);
- a double corner-turn replaces the 5-byte packer;
- each EXP feed sends one gapless burst per window, and that burst is the E5 window;
- the row sums add per-plane counts;
- E6 and E7 are plane windows.

**2026-10-23, the soft-logic trims** (§10 item 8, your go): E4 carries value words, so the input stage reads the
core stream's words straight from its FIFOs and the input corner-turn goes.

The behavior model (`simulator/models/softmaxOnline.py`) implements this version. 2026-10-23: both RTL twins implement
the bit-plane version (GATE 2 at 64×64, 18/18 + 1/1 + 5/5); the trims follow through the same loop.

This is the module's contract: the behavior model and both RTL twins (`ip/softmax/rtl/{sim,vtr}/softmax_online_top.v`)
implement it, and GATE 2 pins every latency named here. Built from the verified v2 core (store, row max, EXP feeds,
row sums, carry; `v2/sim/rtl/softmax_online_top.v`, strict GATE 2 574/574) and `docs/architecture.md` §4.5.
Refinements of §4.5 made while writing this spec are listed in §10.

## 1 Function

One run (one `start`) processes $S$ query rows over $N$ keys in $B = N/B_{kv}$ blocks. Block $b$ of row $m$, keys
$j \in b$:

$$
m_{blk} = \max_j s_{mj}, \quad m_{new} = \begin{cases} m_{blk} & b = 0 \\ \max(m, m_{blk}) & b \ge 1 \end{cases},
\quad \log\alpha = m - m_{new}, \quad \log P_{mj} = \max(s_{mj} - m_{new}, -128)
$$

$$
sp = \sum_j \mathrm{EXP}(\log P_{mj}), \quad
l' = \begin{cases} sp & b = 0 \\ \big(\mathrm{EXP}(\mathrm{clamp8}(\log\alpha + \log l)) \ll LN\big) + sp & b \ge 1 \end{cases},
\quad \log l = \mathrm{LOG}(\min(l' \gg LN, 127))
$$

with $LN = \log_2 N$, then $m \leftarrow m_{new}$. Outputs: $\log P$ per block, $\log\alpha$ per block $b \ge 1$, and the
final $\log l$ per run. Value contract (GATE 1): `softmax_online_ref.logcarry_stage_values`, every stage. The values
are unchanged by the bit-plane pass.

## 2 Parameters

| group | parameter | meaning |
|---|---|---|
| shape | `S, BKV, N, R, C, P` | rows per run, keys per block, keys per run, crossbar, bit slices (`BUF` removed 2026-10-22) |
| crossbars | `N_EXP` | EXP crossbars |
| input | `S_CH` | input channels (= the QK DIMM's $NG$) |
| credits | `LP_CAP`, `LA_CAP`, `LL_CAP` | consumer buffers in units: $\log P$ blocks (2), $\alpha$ blocks (2), $\log l$ runs (1) |

Derived (no free numbers):

| name | value |
|---|---|
| $I$, $B$, $LN$ | $\min(R, C)$; $N/B_{kv}$; $\log_2 N$ |
| $BLKEL$ | $S B_{kv}$: elements per block (= per tile) |
| $NWB$ | $BLKEL/I$: windows per block |
| $PW$ | $B \cdot NWB$: windows per run |
| $Q_W$ | $I/8$: bytes per core stream word (a window is $P$ such words) |
| $STREAM$ | $P \cdot NWB$: core stream words per block |
| $T_W$ | $NWB/$`S_CH`: windows per channel per tile |
| $DF$ | $2^{\lceil \log_2(2 T_W P) \rceil}$: input FIFO depth (plane words, per channel; two tiles) |
| $NB$, $DW$, $BDEPTH$ | the store: $\max(2, N_{EXP})$ banks, $DW = \max(1, 2NWB/NB)$ windows per bank, $BDEPTH = DW \cdot P$ words of $Q_W$ bytes |

Until 2026-10-22: $LW = \lceil 8I/40 \rceil$ words per window, $L_{last}$, $STREAM = \lceil BLKEL/5 \rceil$, $WL =
\lceil S/5 \rceil$, and 5 byte lanes per store bank.

Guards (the RTL's elaboration guards; the model raises on the same sets):
- $S$, $B_{kv}$, $N$, $I$, $N_{EXP}$, `S_CH` powers of two; $8 \le S \le I$; $8 \le B_{kv} \le \min(N, I)$; $P = 8$
  (one plane per bit of an int8 element; until 2026-10-22 `BUF` = 40);
- $S B_{kv} \ge I$ (whole windows); `S_CH` $\le NWB$; $PW$ a multiple of $N_{EXP}$ (lanes keep their windows across
  runs);
- `LP_CAP`, `LA_CAP`, `LL_CAP` $\ge 1$.

## 3 Ports

| port | width | dir | discipline | unit | `CAP` |
|---|---|---|---|---|---|
| `s_v`, `s_d` / `s_cr` | `S_CH`, `S_CH`·$I$ / 1 | in / out | push + credit | a tile ($S \times B_{kv}$ int8, $NWB$ windows of $I$, 8 value words of $Q_W$ bytes each, `docs/architecture.md` §3.1) | 2 tiles |
| `start` / `busy` | 1 / 1 | in / out | burst on ready | a run | 1 |
| `lp_v`, `lp_d` / `lp_cr` | $N_{EXP}$, $N_{EXP} \cdot I$ / 1 | out / in | push + credit | a block's $\log P$ ($NWB$ windows of $I$) | `LP_CAP` |
| `la_v`, `la_d` / `la_cr` | 1, $2S$ / 1 | out / in | push + credit | a block $b \ge 1$: one window of $S$ rows (9 bits, 2 lanes) | `LA_CAP` |
| `ll_v`, `ll_d` / `ll_cr` | 1, $S$ / 1 | out / in | push + credit | a run: one window of $S$ rows | `LL_CAP` |
| `done` | 1 | out | | pulse after the run's last $\log l$ plane | |

Framing (`architecture.md` §3.1, plane framing since 2026-10-22): every window is 8 plane words on consecutive
cycles, LSB first.
- E4 (`s`): tile $j$ = the QK DIMM's job $j$. Window $w$ = elements $wI \ldots wI + I - 1$ of the tile, row-major, on
  channel $w \bmod$ `S_CH`.
- E5 (`lp`): block $j$ of the stream ($j = rB + b$). Window $w$ = elements $wI \ldots$ of the block's $\log P$, on lane
  $(b \cdot NWB + w) \bmod N_{EXP}$, the EXP crossbar that converts it. The window is that crossbar's input burst.
- E6 (`la`): block $j$ with $b \ge 1$ is unit $r(B - 1) + b - 1$: one window, element $m$ = $\log\alpha$ of row $m$
  (9 bits: lane 1 carries bit 8).
- E7 (`ll`): run $r$ is unit $r$: one window, element $m$ = $\log l$ of row $m$.

Until 2026-10-22: pre-bit-plane words of 5 bytes; E6 was $S$ windows of one 9-bit word, E7 one window of $S$ words.

## 4 Structure

```text
 s_v/s_d (S_CH) -> [FIFO per channel, DF x I bits, 1W1R]
                -> [READER: tile j opens on its gate; window w's words from FIFO w mod S_CH, one per cycle]
                -> core stream (each word Q_W bytes)
 core (v2's datapath, at Q_W bytes per word):
   store (2-block ring, bytes) | row max -> m_new, log a tables -> E6 planes
   N_EXP EXP feeds: s - m_new, clamp -128, at Q_W lanes -> [DOUBLE CORNER-TURN] -> one 8-plane burst = E5 window
   row sums sp: per-plane counts of each row's bits, accumulated over the 8 planes
   carry: planes of clamp8(log a + log l) -> carry-EXP -> e (plane by plane) -> lq planes -> carry-LOG
          -> the log l table (plane by plane) | retire count
 outputs: lp = the EXP bursts; la = one plane window per block b >= 1; ll = the log l table's planes
 gates (registered): tile open = armed, rb < B, slot (block rb - 2 retired), lp credit, (rb >= 1) la credit
```

**The bit-plane datapath** (2026-10-22):
- **Corner-turns.** A corner-turn buffer is $8 \times I$ bits. An EXP feed writes a window's bytes into it $Q_W$ per
  cycle and sends the 8 planes once the last byte is in; two buffers per feed let one window burst while the next one
  fills. The input needs none since E4 carries value words (the trims of 2026-10-23).
- **Row sums.** Each EXP drain plane $b$ carries bit $b$ of the window's EXP bytes. A window holds whole rows
  ($I/B_{kv}$ of them, as $B_{kv}$ divides $I$). For each of them, the count of its $B_{kv}$ set bits in the plane is
  added with weight $2^b$, so after plane 7 the row's sum is exact.
- **Tables** ($m$, $m_{new}$, $\log\alpha$, $sp$, $e$, $lq$, $\log l$) are flip-flops, $S$ entries each. A plane of a
  table is bit $b$ of every entry: the carry inputs, E6 and E7 read planes directly, and the carry-EXP and carry-LOG
  drains write $e$ and $\log l$ one plane per cycle.

Extents: the EXP passes $I \times I$; the carry-EXP and carry-LOG passes $S \times S$ (the rows and columns in use).
Since 2026-10-22 the extent changes only the values and the energy, not the time (`ip/xbar/spec.md`).

## 5 Timing rules

Cycle 0 is the first cycle after reset. A signal "at $t$" is high (or a register reads its value) in cycle $t$.

| name | value | rule |
|---|---|---|
| `FIFO_LAT` | 1 | an E4 word valid at $t$ can be read from $t + 1$ |
| `CR_IN_LAT` | 1 | a credit pulse (`lp_cr`, `la_cr`, `ll_cr`) at $t$ counts for the gates from $t + 1$ |
| `RET_LAT` | 1 | a block retired at $t$ (its carry-LOG's last drain plane) counts for the gates from $t + 1$ |
| `ARM_LAT` | 2 | `start` accepted at $t$: the run is armed from $t + 2$ (the core's run state is cleared at $t + 1$) |
| `OPEN_LAT` | 1 | the reader decides to open a tile at $t_{dec}$ and reads its first plane at $t_{dec} + 1$ |
| `RD_LAT` | 1 | a word read at $r$ is the core's word at $r + 1$ (the registered read). Until the trims: `CT_LAT` 2, a window's first core word 2 cycles after its last plane's read (the registered read, the corner-turn) |
| `SCR_LAT` | 1 | `s_cr` pulses one cycle after a tile's last read |
| `MAX_LAT` | $\log_2 \min(Q_W, B_{kv})$ + `MAX_REG` | the block's last core word at $a_{last}$: every row's $m_{new}$ from $a_{last}$ + `MAX_LAT`. The tree over a word's elements of one row has $\log_2 \min(Q_W, B_{kv})$ levels; `MAX_REG` = 2: the input register and the running max with $m$ |
| `FEED_LAT` | `MAX_LAT` $+ P +$ `FEED_REG` | the EXP feeds start reading the block at $a_{last}$ + `MAX_LAT`: a window's $P$ store words, then `FEED_REG` = 2 (the registered read, the registered $s - m_{new}$ and clamp), fill its corner-turn by $a_{last}$ + `FEED_LAT` |
| `SP_LAT` | 3 | a block's row sums are complete `SP_LAT` cycles after its last EXP drain plane (the count register, the weighted add, the window add) |
| `WAIT_LAT` | 1 | the carry decision is registered |
| `GO_LAT` | 2 | a carry input computed from the tables (registered), then its plane register |
| `E_LAT` | 1 | the carry-EXP's last drain plane at $x$: $e$ is whole at $x + 1$ (its plane register) |
| `LA_LAT` | 1 | the E6 window's first plane one cycle after $m_{new}$ (the registered $m - m_{new}$) |
| `E7_LAT` | 2 | the $\log l$ read-out decided at $t$: its first plane at $t + 2$ (the decision register, the plane register) |
| `DONE_LAT` | 1 | `done` one cycle after the last $\log l$ plane |
| `GO_CR_LAT` | 1 | `busy` falls (the start port's credit) one cycle after `done` |

Until 2026-10-22: `PACK_LAT` 2 (the packer), `MAX_LAT` 4, `FEED_LAT` 9, `SP_LAT` 2, `GO_LAT` 4, `LA_EV_LAT` 5 (a
row's $\log\alpha$ five cycles after the word that completed the row), `E7_LAT` 3.

**Tile open.** The reader is idle from cycle 0 and from the cycle after a tile's last read. Tile $j$ (run $r$, block
$b$, the run's $rb = b$-th tile) is decided at the first cycle $t_{dec}$ at which the reader is idle and:
- run $r$ is armed ($t_{dec} \ge t_{start}(r) +$ `ARM_LAT`);
- $b < 2$, or block $b - 2$ is retired ($t_{dec} \ge ret(b - 2) +$ `RET_LAT`);
- $j <$ `LP_CAP`, or $t_{dec} \ge t_{lp\_cr}(j -$ `LP_CAP`$) +$ `CR_IN_LAT`;
- $b = 0$, or with $u = r(B - 1) + b - 1$: $u <$ `LA_CAP`, or $t_{dec} \ge t_{la\_cr}(u -$ `LA_CAP`$) +$ `CR_IN_LAT`;
- the tile's first plane is in FIFO 0 ($t_{dec} \ge t_{arr}(0, 0) +$ `FIFO_LAT`).

Opening takes one $\log P$ credit, and one $\alpha$ credit for $b \ge 1$.

**Reads.** Word $i$ of the tile (window $w = \lfloor i/P \rfloor$, word $c = i \bmod P$) is read at
$r_0 = t_{dec} +$ `OPEN_LAT` and $r_i = \max(r_{i-1} + 1,\ t_{arr}(w, c) +$ `FIFO_LAT`$)$, where
$t_{arr}(w, c)$ = window $w$'s first word cycle $+ c$. `s_cr` at $r_{last} +$ `SCR_LAT`.

**Core words.** Word $c$ of window $w$ (bytes $cQ_W \ldots cQ_W + Q_W - 1$ of the window, i.e. block bytes
$wI + cQ_W \ldots$) reaches the core at $t_s(w, c) = r_{wP + c} +$ `RD_LAT`, one word per cycle at most.
$a_{last} = t_s(NWB - 1, P - 1) = r_{last} +$ `RD_LAT`. (Until the trims: $t_s(w, c) = r_{wP + P - 1} + 2 + c$.)

**Core** (v2's recurrence on the bit-plane pass, from the stream). Per block $b$ of the run:
- EXP window $g = b \cdot NWB + w$ on crossbar $g \bmod N_{EXP}$: one pass ($I \times I$) ready at
  $a_{last} +$ `FEED_LAT`, placed by the pass rule on the crossbar's own state (back to back, a pass every 10 cycles).
  Its `load_start` is the E5 window's time.
- $sp_{done} = \max_w (drain\_start + P - 1) +$ `SP_LAT`.
- $wait = \max(sp_{done},\ a_{last} +$ `MAX_LAT`$,\ ret(b - 1)) +$ `WAIT_LAT` (no $ret(b - 1)$ term at $b = 0$).
- $b = 0$: the carry-LOG pass ($S \times S$) ready at $wait +$ `GO_LAT`.
- $b \ge 1$: the carry-EXP pass ($S \times S$) ready at $wait +$ `GO_LAT`, with its last drain plane at
  $x = drain\_start + P - 1$; then the carry-LOG pass ready at $x +$ `E_LAT` $+$ `GO_LAT`.
- $ret(b) = drain\_start_{LOG} + P - 1$, the carry-LOG's last drain plane. After the last block,
  $t_{cd} = ret(B - 1) + 1$ (the core's done).

Until 2026-10-22: the carry walked $WL$ words ($b = 0$: carry-LOG at $wait + WL +$ `GO_LAT`; $ret$ and $x$ at
$drain\_start + WL - 1$).

The crossbars keep their pass state across runs; every other core register is cleared on the run's arm.

**$\log\alpha$.** Block $b \ge 1$'s E6 window at $a_{last} +$ `MAX_LAT` $+$ `LA_LAT`.

**$\log l$ and done.** The read-out of run $r$ is decided at $t_{d7} = \max(t_{cd},\ t_{ll\_cr}(r -$
`LL_CAP`$) +$ `CR_IN_LAT`$)$ (no credit term for $r <$ `LL_CAP`), and takes one $\log l$ credit. E7 window at
$t_{d7} +$ `E7_LAT`; `done` at $t_{d7} +$ `E7_LAT` $+ P - 1 +$ `DONE_LAT`; `busy` falls at `done` + `GO_CR_LAT`.

**Run start.** `start` is taken in a cycle with `busy` low; `busy` is high from the next cycle until `done` +
`GO_CR_LAT`.

Model checks: the FIFO of a channel never holds more than $DF$ words; a block's E6 window leaves before block $b + 2$
can overwrite its table (guaranteed by the slot rule).

## 6 Memories

| memory | organization | size |
|---|---|---|
| input FIFOs | `S_CH` × $DF$ words × $I$ bits, 1W1R, registered read | $2 \cdot NWB \cdot P$ value words in use |
| store | $NB$ banks × $BDEPTH$ words × $I$ bits ($Q_W$ bytes), 1W1R | |
| corner-turns | 2 per EXP feed: $8 \times I$ bits each, flip-flops (until the trims also 2 at the input) | |
| tables | $m$, $m_{new}$ and $\log\alpha$ (two each), $sp$ (two), $e$, $lq$, $\log l$: $S$ entries each, flip-flops (v2) | |

Until 2026-10-22: FIFOs of pre-bit-plane words, and the store's banks as 5 byte-lane memories of 8 bits.

## 7 Counts and resources

- Passes per block: $NWB$ EXP ($I \times I$); one carry-LOG ($S \times S$); one carry-EXP ($S \times S$) for $b \ge 1$.
- CLB per block: compares $S B_{kv}$ (row max) $+ S$ (running max); adds $2 S B_{kv}$ ($s - m_{new}$, row sums)
  $+ 2S$ ($\log\alpha + \log l$, $l'$); selects $S B_{kv} + 2S$ (the clamps).
- Memory per block: $S B_{kv}$ bytes written into the store and read by the feeds. FIFO words per tile: $NWB \cdot P$
  plane words, also counted as memory traffic: each word's $I/8$ bytes written once and read once (2026-10-15; until
  2026-10-22 each pre-bit-plane word's 5 bytes, so $5 \lceil I/5 \rceil$ bytes per window).
- Crossbars: $N_{EXP}$ EXP + 1 carry-EXP + 1 carry-LOG = $N_{EXP} + 2$.

## 8 Gates

- GATE 1 (case writer): every stage of the model == `softmax_online_ref.logcarry_stage_values`, every run.
- Closed form (the model's test, written by hand from §5): the core, fed a stream with every tile early, gives every
  block's EXP passes, $sp_{done}$, $wait$, carry passes and $ret$ by the recurrence of §5, and the run's total. (Until
  2026-10-22 the witness was `softmax_online_sim.softmax_online_cycle_model`, which times the pre-bit-plane pass.)
- GATE 2 (64×64): the RTL's values on E5, E6 and E7, every port event and the total cycles == the model's.
- VTR smoke (128×128): crossbar count == $N_{EXP} + 2$; Fmax, CLB, memory, path #1.

## 9 The instance in the head

| parameter | value |
|---|---|
| `S`, `BKV`, `N` | $\min(n, I)$, 16, $n$ |
| `N_EXP`, `S_CH` | 2, the QK DIMM's $NG = N_E/d$ |
| `LP_CAP`, `LA_CAP`, `LL_CAP` | 2 (the PV DIMM's A slots), 2 (the accumulator's $\alpha$ ring), 1 (its $\log l$ table) |

## 10 Refinements of `architecture.md` §4.5

1. **E6 carried one row per window** ($W = 1$, $S$ windows per block) until 2026-10-22: each row left as its maximum
   completed, as in v2, so no buffer was needed. Since the bit-plane pass E6 is one window of $S$ rows (item 6).
2. **The carry passes use the extent $S \times S$**, the rows and columns in use (the extent rule); the EXP passes
   stay $I \times I$. §1 listed $I \times I$ for all three. $S < I$ occurs for $n < I$, so the 64×64 corpus exercises it.
3. **The gates move to the reader.** The core's v2 `score_ready` is gone: the reader opens a tile only when the store
   slot, the run and the credits allow, and the stream cannot hold words. A tile opens one cycle after its decision,
   so one idle cycle separates tiles.
4. **New guards:** $S B_{kv} \ge I$ (windows are whole), `S_CH` $\le NWB$, $PW$ a multiple of $N_{EXP}$.
5. **Run control:** `done` follows the last E7 plane; `busy` falling is the start port's credit. The final $m$ is not
   an output (v2's `rd_m` is dropped).
6. **The bit-plane pass** (user, 2026-10-21/22; `ip/xbar/spec.md`).
   - **Ports:** every port carries 8 plane words per window: `s_d` and `lp_d` $I$ bits per channel, `la_d` $2S$ bits,
     `ll_d` $S$ bits. `BUF` is removed.
   - **Input:** the FIFOs hold plane words. A double corner-turn makes the core stream of $Q_W = I/8$ bytes per word,
     so the 5-byte packer goes; window-aligned words cost nothing now.
   - **EXP feeds:** $s - m_{new}$ and the clamp at $Q_W$ lanes into a double corner-turn, then one gapless burst per
     window. The burst is the E5 window, so the pop-when-ready feed FIFO goes.
   - **Row sums:** per-plane counts, weighted by $2^b$.
   - **Carry:** inputs and outputs as planes of the tables, with no walk; `GO_LAT` 4 → 2.
   - **E6:** one window of $S$ rows per block, after the block's last $m_{new}$ (`LA_LAT`). **E7:** one window of $S$
     rows; `E7_LAT` 3 → 2.
   - **Store:** $NB$ banks of $Q_W$-byte words (was 5 byte lanes).
   - **Values, counts (except the FIFO bytes, §7) and the gate rules are unchanged.**
7. **RTL details of the bit-plane pass** (2026-10-23; no timing rule changes): an EXP feed starts window $k$'s reads at
   $s_k = \max(a_{last} + $ `MAX_LAT`$,\ s_{k-1} + 10)$ and loads the window's rows' $m_{new}$ at $s_k$, so its
   burst at $s_k + 10$ is the pass rule's `load_start`; block $b$'s $m$ is the rotating ring of the row max (as in v2).
8. **The soft-logic trims** (2026-10-23, your go): E4 carries value words (`docs/architecture.md` §3.1). The reader's
   words go straight into the core, so the input's double corner-turn ($2 \times 8I$ flip-flop bits, about 2k at
   $I$ = 128) goes, and a word reaches the core at its read + `RD_LAT` (1) instead of after its window's last plane
   (+2 + its index): a block's last word reaches the core 8 cycles earlier. Everything else is unchanged.
