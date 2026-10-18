# `softmax_online_top` — online softmax (v3 module spec)

Status: 2026-10-14, F3 / phase B, **validated**: GATE 2 18/18 at 64×64, VTR smoke at 128×128 (`LEDGER.md`). This
is the module's contract: the behavior model `ip/softmax/model.py` and
both RTL twins (`ip/softmax/rtl/{sim,vtr}/softmax_online_top.v`) implement it, and GATE 2 pins every latency named
here. Built from the verified v2 core (store, row max, EXP feeds, row sums, carry; `v2/sim/rtl/softmax_online_top.v`,
strict GATE 2 574/574) and `docs/architecture.md` §4.5. Refinements of §4.5 made while writing this spec are listed in
§10.

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
final $\log l$ per run. Value contract (GATE 1): `softmax_online_ref.logcarry_stage_values`, every stage.

## 2 Parameters

| group | parameter | meaning |
|---|---|---|
| shape | `S, BKV, N, R, C, BUF, P` | rows per run, keys per block, keys per run, crossbar, port width, bit slices |
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
| $LW$ | $\lceil 8I/BUF \rceil$: words per window |
| $L_{last}$ | $I - 5(LW - 1)$: bytes in a window's last word |
| $STREAM$ | $\lceil BLKEL/5 \rceil$: core stream words per block |
| $T_W$ | $NWB/$`S_CH`: windows per channel per tile |
| $DF$ | $2^{\lceil \log_2(2 T_W LW) \rceil}$: input FIFO depth (words, per channel; two tiles) |
| $WL$ | $\lceil S/5 \rceil$: words of a carry pass |
| $NB$, $WPW$, $DW$, $BDEPTH$ | the v2 store: $\max(2, N_{EXP})$ banks, $LW$, $\max(1, 2NWB/NB)$, $DW \cdot WPW$ |

Guards (the RTL's elaboration guards; the model raises on the same sets):
- $S$, $B_{kv}$, $N$, $I$, $N_{EXP}$, `S_CH` powers of two; $8 \le S \le I$; $8 \le B_{kv} \le \min(N, I)$; `BUF` = 40
  (v2);
- $S B_{kv} \ge I$ (whole windows); `S_CH` $\le NWB$; $PW$ a multiple of $N_{EXP}$ (lanes keep their windows across
  runs);
- `LP_CAP`, `LA_CAP`, `LL_CAP` $\ge 1$.

## 3 Ports

| port | width | dir | discipline | unit | `CAP` |
|---|---|---|---|---|---|
| `s_v`, `s_d` / `s_cr` | `S_CH`, `S_CH`·40 / 1 | in / out | push + credit | a tile ($S \times B_{kv}$ int8, $NWB$ windows of $I$) | 2 tiles |
| `start` / `busy` | 1 / 1 | in / out | burst on ready | a run | 1 |
| `lp_v`, `lp_d` / `lp_cr` | $N_{EXP}$, $N_{EXP} \cdot 40$ / 1 | out / in | push + credit | a block's $\log P$ ($NWB$ windows of $I$) | `LP_CAP` |
| `la_v`, `la_d` / `la_cr` | 1, 9 / 1 | out / in | push + credit | a block $b \ge 1$: $S$ windows of one row | `LA_CAP` |
| `ll_v`, `ll_d` / `ll_cr` | 1, 8 / 1 | out / in | push + credit | a run: one window of $S$ rows | `LL_CAP` |
| `done` | 1 | out | | pulse after the run's last $\log l$ word | |

Framing (`architecture.md` §3.1):
- E4 (`s`): tile $j$ = the QK DIMM's job $j$. Window $w$ = elements $wI \ldots wI + I - 1$ of the tile, row-major; it
  is on channel $w \bmod$ `S_CH`, $LW$ words, the last word padded.
- E5 (`lp`): block $j$ of the stream ($j = rB + b$); window $w$ = elements $wI \ldots$ of the block's $\log P$, $LW$
  words, on lane $(b \cdot NWB + w) \bmod N_{EXP}$ = the EXP crossbar that converts it.
- E6 (`la`): block $j$ with $b \ge 1$ is unit $r(B - 1) + b - 1$; window $w$ = row $w$, one 9-bit word.
- E7 (`ll`): run $r$ is unit $r$; one window of $S$ words, word $m$ = $\log l$ of row $m$.

## 4 Structure

```text
 s_v/s_d (S_CH) -> [FIFO per channel, DF x 40, 1W1R]
                -> [READER: tile j opens on its gate; window w from FIFO w mod S_CH, one word per cycle]
                -> [PACKER: window words (5 or L_last bytes) -> STREAM words per block, each block padded] -> core
 core (v2, unchanged datapath; extents below):
   store (2-block ring) | row max -> m_new, log a tables, la events | N_EXP EXP feeds -> lp words (pops)
   row sums sp | carry: [carry-EXP of clamp8(log a + log l)] -> lq -> [carry-LOG] -> log l | retire count
 outputs: lp = the pops; la = the la events of blocks b >= 1; ll = the final log l, read row by row
 gates (registered): tile open = armed, rb < B, slot (block rb - 2 retired), lp credit, (rb >= 1) la credit
```

Extents: the EXP passes $I \times I$; the carry-EXP and carry-LOG passes $S \times S$ (the rows and columns in use;
since 2026-10-18 the simulator drains every column of the array on every pass, the extent setting the load and the
valid window, while the RTL still stops after the columns in use: `architecture.md` §1).

## 5 Timing rules

Cycle 0 is the first cycle after reset. A signal "at $t$" is high (or a register reads its value) in cycle $t$.

| name | value | rule |
|---|---|---|
| `FIFO_LAT` | 1 | an E4 word valid at $t$ can be read from $t + 1$ |
| `CR_IN_LAT` | 1 | a credit pulse (`lp_cr`, `la_cr`, `ll_cr`) at $t$ counts for the gates from $t + 1$ |
| `RET_LAT` | 1 | a block retired at $t$ (its carry-LOG drain word with row $S - 1$) counts for the gates from $t + 1$ |
| `ARM_LAT` | 2 | `start` accepted at $t$: the run is armed from $t + 2$ (the core's run state is cleared at $t + 1$) |
| `OPEN_LAT` | 1 | the reader decides to open a tile at $t_{dec}$ and reads its first word at $t_{dec} + 1$ |
| `PACK_LAT` | 2 | a core stream word completed by the read at $r$ reaches the core at $r + 2$; a second word completed by the same read, at $r + 3$ |
| `SCR_LAT` | 1 | `s_cr` pulses one cycle after a tile's last read |
| `MAX_LAT`, `FEED_LAT`, `SP_LAT`, `WAIT_LAT`, `GO_LAT` | 4, 9, 2, 1, 4 | the v2 core (`softmax_online_sim`), from the block's last stream word $a_{last}$ |
| `LA_EV_LAT` | 5 | a row's $\log\alpha$ is on `la_v` five cycles after the stream word that completes the row |
| `E7_LAT` | 3 | the $\log l$ read-out decided at $t$: its first word at $t + 3$ (address register + the 2-cycle registered read) |
| `DONE_LAT` | 1 | `done` one cycle after the last $\log l$ word |
| `GO_CR_LAT` | 1 | `busy` falls (the start port's credit) one cycle after `done` |

**Tile open.** The reader is idle from cycle 0 and from the cycle after a tile's last read. Tile $j$ (run $r$, block
$b$, the run's $rb = b$-th tile) is decided at the first cycle $t_{dec}$ at which the reader is idle and:
- run $r$ is armed ($t_{dec} \ge t_{start}(r) +$ `ARM_LAT`);
- $b < 2$, or block $b - 2$ is retired ($t_{dec} \ge ret(b - 2) +$ `RET_LAT`);
- $j <$ `LP_CAP`, or $t_{dec} \ge t_{lp\_cr}(j -$ `LP_CAP`$) +$ `CR_IN_LAT`;
- $b = 0$, or with $u = r(B - 1) + b - 1$: $u <$ `LA_CAP`, or $t_{dec} \ge t_{la\_cr}(u -$ `LA_CAP`$) +$ `CR_IN_LAT`;
- the tile's first word is in FIFO 0 ($t_{dec} \ge t_{arr}(0, 0) +$ `FIFO_LAT`).

Opening takes one $\log P$ credit, and one $\alpha$ credit for $b \ge 1$.

**Reads.** Input word $i$ of the tile (window $w = \lfloor i/LW \rfloor$, word $c = i \bmod LW$) is read at
$r_0 = t_{dec} +$ `OPEN_LAT` and $r_i = \max(r_{i-1} + 1,\ t_{arr}(w, c) +$ `FIFO_LAT`$)$, where
$t_{arr}(w, c)$ = window $w$'s first word cycle $+ c$. `s_cr` at $r_{last} +$ `SCR_LAT`.

**Packer.** Word $c$ carries bytes $5c \ldots$ of its window: 5 bytes, or $L_{last}$ for $c = LW - 1$; window $w$'s byte
$x$ is block byte $wI + x$. Core stream word $k$ (block bytes $5k \ldots 5k + 4$) is complete with the input word
$i(k)$ that holds block byte $\min(5k + 4, BLKEL - 1)$; it reaches the core at
$t_s(k) = \max(r_{i(k)} +$ `PACK_LAT`$,\ t_s(k - 1) + 1)$. Only a tile's last read can complete two words, and the
reader's idle cycle after it leaves the packer free for the second. $a_{last} = t_s(STREAM - 1)$.

**Core** (v2 `softmax_online_cycle_model`, from the stream instead of a fixed start). Per block $b$ of the run:
- EXP window $g = b \cdot NWB + w$ on crossbar $g \bmod N_{EXP}$: one pass ($I \times I$) ready at
  $a_{last} +$ `FEED_LAT`, placed by the pass grammar on the crossbar's own state; its `load_start` is the E5
  window's time (its $LW$ words are popped on consecutive cycles);
- $sp_{done} = \max_w (drain\_start + LW - 1) +$ `SP_LAT`;
- $wait = \max(sp_{done},\ a_{last} +$ `MAX_LAT`$,\ ret(b - 1)) +$ `WAIT_LAT` (no $ret(b - 1)$ term at $b = 0$);
- $b = 0$: the carry-LOG pass ($S \times S$) ready at $wait + WL +$ `GO_LAT`; $b \ge 1$: the carry-EXP pass
  ($S \times S$) ready at $wait +$ `GO_LAT`, $x = drain\_start + WL - 1$, then the carry-LOG pass ready at
  $x +$ `GO_LAT`;
- $ret(b) = drain\_start_{LOG} + WL - 1$; after the last block, $t_{cd} = ret(B - 1) + 1$ (the core's done).

The crossbars keep their pass state across runs; every other core register is cleared on the run's arm.

**$\log\alpha$.** Row $m$ of block $b \ge 1$ completes with the stream word $k = \lfloor ((m + 1) B_{kv} - 1)/5 \rfloor$
of the block: its E6 window at $t_s(k) +$ `LA_EV_LAT`.

**$\log l$ and done.** The read-out of run $r$ is decided at $t_{d7} = \max(t_{cd},\ t_{ll\_cr}(r -$
`LL_CAP`$) +$ `CR_IN_LAT`$)$ (no credit term for $r <$ `LL_CAP`), and takes one $\log l$ credit. E7 window at
$t_{d7} +$ `E7_LAT`; `done` at $t_{d7} +$ `E7_LAT` $+ S - 1 +$ `DONE_LAT`; `busy` falls at `done` + `GO_CR_LAT`.

**Run start.** `start` is taken in a cycle with `busy` low; `busy` is high from the next cycle until `done` +
`GO_CR_LAT`.

Model checks: the FIFO of a channel never holds more than $DF$ words; a block's E6 rows leave before block $b + 2$ can
overwrite their table (guaranteed by the slot rule).

## 6 Memories

| memory | organization | size |
|---|---|---|
| input FIFOs | `S_CH` × $DF$ words × 40 bits, 1W1R, registered read | $2 \cdot NWB \cdot LW \cdot 5$ bytes in use |
| store | $NB$ banks × 5 byte lanes × $BDEPTH$ bytes, 1W1R (v2) | |
| tables | $m$, $m_{new}$ and $\log\alpha$ (two each), $sp$ (two), $lq$, $\log l$: $S$ entries each, flip-flops (v2) | |

## 7 Counts and resources

- Passes per block: $NWB$ EXP ($I \times I$); one carry-LOG ($S \times S$); one carry-EXP ($S \times S$) for $b \ge 1$.
- CLB per block: compares $S B_{kv}$ (row max) $+ S$ (running max); adds $2 S B_{kv}$ ($s - m_{new}$, row sums)
  $+ 2S$ ($\log\alpha + \log l$, $l'$); selects $S B_{kv} + 2S$ (the clamps).
- Memory per block: $S B_{kv}$ bytes written into the store and read by the feeds. FIFO words per tile: $NWB \cdot LW$,
  also counted as memory traffic: each word's 5 bytes written once and read once (2026-10-15).
- Crossbars: $N_{EXP}$ EXP + 1 carry-EXP + 1 carry-LOG = $N_{EXP} + 2$.

## 8 Gates

- GATE 1 (case writer): every stage of the model == `softmax_online_ref.logcarry_stage_values`, every run.
- Closed form: the core, fed v2's stream schedule (word 0 at 2, back to back, block $b \ge 2$ at $ret(b - 2) + 1$)
  with the carry extent $I$, reproduces `softmax_online_cycle_model(pass_extent="used")`: every block's events and the
  total.
- GATE 2 (64×64): the RTL's values on E5, E6 and E7, every port event and the total cycles == the model's.
- VTR smoke (128×128): crossbar count == $N_{EXP} + 2$; Fmax, CLB, memory, path #1.

## 9 The instance in the head

| parameter | value |
|---|---|
| `S`, `BKV`, `N` | $\min(n, I)$, 16, $n$ |
| `N_EXP`, `S_CH` | 2, the QK DIMM's $NG = N_E/d$ |
| `LP_CAP`, `LA_CAP`, `LL_CAP` | 2 (the PV DIMM's A slots), 2 (the accumulator's $\alpha$ ring), 1 (its $\log l$ table) |

## 10 Refinements of `architecture.md` §4.5

1. **E6 carries one row per window** ($W = 1$, $S$ windows per block): each row leaves as its maximum completes, as in
   v2, so no buffer is needed. The edge table of §3.3 had $W = S$.
2. **The carry passes use the extent $S \times S$**, the rows and columns in use (the extent rule); the EXP passes
   stay $I \times I$. §1 listed $I \times I$ for all three. $S < I$ occurs for $n < I$, so the 64×64 corpus exercises it.
3. **The gates move to the reader.** The core's v2 `score_ready` is gone: the reader opens a tile only when the store
   slot, the run and the credits allow, and the packer cannot hold words. A tile opens one cycle after its decision,
   so one idle cycle separates tiles; the packer uses it for a block's last partial word.
4. **New guards:** $S B_{kv} \ge I$ (windows are whole), `S_CH` $\le NWB$, $PW$ a multiple of $N_{EXP}$.
5. **Run control:** `done` follows the last E7 word; `busy` falling is the start port's credit. The final $m$ is not
   an output (v2's `rd_m` is dropped).
