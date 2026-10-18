# `softmax_online_acc` — online softmax accumulator (v3 module spec)

Status: 2026-10-14, F3 / phase C, **validated**: GATE 2 19/19 (plus 5/5 bring-up) at 64×64, VTR smoke at 128×128
(156.3 MHz) (`LEDGER.md`). 2026-10-15: the O reader is removed and E9 carries the final $O_8$ values as they are
computed (§10 item 7); RTL and model changed, the RTL not simulated after it (you), the head VTR smoke in
`LEDGER.md`. This is the module's contract: the
behavior model `ip/softmax/acc_model.py` and both RTL twins (`ip/softmax/rtl/{sim,vtr}/softmax_online_acc.v`)
implement it, and GATE 2 pins every latency named here. Built from the verified v2 o-carry station
(`v2/sim/rtl/att_ocarry_top.v`: the two crossbar groups, the job engines, the slot memories, the FIFOs; strict GATE 2
in v2) and `docs/architecture.md` §4.6. Refinements of §4.6 made while writing this spec are listed in §10.

## 1 Function

The output half of the online softmax over key blocks. Per run (one block of $S$ query rows), block $b$ of the run
brings a weighted partial sum $o_{part}$ ($S \times D$) and, for $b \ge 1$, the rescale factor $\log\alpha$ (one per
row). Per element ($m$ = row, $c$ = column):

$$
o' = \begin{cases} o_{part} & b = 0 \\ (so\ ?\ {-1} : 1) \cdot \mathrm{EXP}(\mathrm{clamp8}(\log\alpha_m + lo)) \cdot 2^{SO} + o_{part} & b \ge 1 \end{cases},
\quad so = [o' < 0], \quad lo = \mathrm{LOG}(\min(\lvert o' \rvert \gg SO, 127))
$$

After the run's last block, with the run's final $\log l$ (one per row):

$$
e_E = \mathrm{EXP}(\mathrm{clamp8}(lo - \log l_m)) \cdot 2^{SO - LN}, \qquad
O_8 = so\ ?\ \max(-e_E, -128) : \min(e_E, 127)
$$

$\mathrm{EXP}$ is read as an unsigned byte, $\mathrm{LOG}$ as a signed byte (`architecture.md` §2.1). Output: $O_8$,
one token row of $D$ int8 per window, rows in order. Value contract (GATE 1): `attention_ref.ocarry_stage_values`
(every block stage) and `attention_ref.epilogue` ($t_E$, $e_E$, $O_8$), every run.

## 2 Parameters

| group | parameter | meaning |
|---|---|---|
| shape | `S, D, R, C, BUF, P` | rows per run, row width, crossbar, port width, bit slices |
| run | `NBLK, SO, LN` | blocks per run; the shifts of $o$ and $l$ ($2^{SO}$, $2^{LN}$) |
| crossbars | `N_OC` | crossbars per group (one LOG group and one EXP group) |
| input | `P_CH`, `VWP` | $o_{part}$ channels (= the PV DIMM's $NG$); $o_{part}$ lane width |

The buffers are fixed: $o_{part}$ FIFOs of 2 tiles, an $\alpha$ ring of 2 blocks, one $\log l$ table. They are the
`CAP` of the three input ports (§3).

Derived (no free numbers):

| name | value |
|---|---|
| $I$ | $\min(R, C)$ |
| $P_{oc}$ | $S D / I$: windows per run (= per block) |
| $RPW$ | $I / D$: rows per window |
| $W$ | $\lceil P_{oc}/N_{OC} \rceil$: waves; window $p = w N_{OC} + i$ is on crossbar $i$ in wave $w$ |
| $LWC$ | $P_{oc} - (W - 1) N_{OC}$: windows of the last wave |
| $LW$ | $\lceil I/5 \rceil$: words per window |
| $T_W$ | $P_{oc}/$`P_CH`: windows per channel per tile |
| $FD$ | $2^{\lceil \log_2(2 T_W LW) \rceil}$: FIFO depth (words, per channel; two tiles) |
| $NJ$ | $NBLK \cdot W$: jobs per group per run |
| $SHL$, $SHR$ | $\max(SO - LN, 0)$, $\max(LN - SO, 0)$ |

Guards (the RTL's elaboration guards; the model raises on the same sets):
- $S$, $D$, $I$, `P_CH` powers of two; $8 \le S \le I$; $D \le I$; $I \mid S D$; `BUF` = 40;
- $1 \le N_{OC} \le P_{oc}$; $N_{OC} \le$ `P_CH` (a job's windows are in different FIFOs); `P_CH` $\mid P_{oc}$;
- $NBLK \ge 1$; $SO, LN \ge 0$; $1 \le$ `VWP` $\le 32$.

## 3 Ports

| port | width | dir | discipline | unit | `CAP` |
|---|---|---|---|---|---|
| `p_v`, `p_d` / `p_cr` | `P_CH`, `P_CH` · 5 · `VWP` / 1 | in / out | push + credit | a tile: one block's $o_{part}$ ($P_{oc}$ windows of $I$) | 2 tiles |
| `la_v`, `la_d` / `la_cr` | 1, 9 / 1 | in / out | push + credit | a block $b \ge 1$: $S$ windows of one row | 2 blocks |
| `ll_v`, `ll_d` / `ll_cr` | 1, 8 / 1 | in / out | push + credit | a run: one window of $S$ rows | 1 run |
| `o_v`, `o_d` | 1, $N_{OC}$ · 40 | out | push | a run: $W$ windows, one per epilogue wave, of $LW$ words of $5 N_{OC}$ int8 | — |
| `run_done` | 1 | out | | pulse one cycle after a run's end ($t_{o8}$: its last $O_8$ word on E9) | |
| `weight_in`, `weight_en` | 8, 1 | in | | simulation twin only (the testbench preloads, R8) | |

Framing (`architecture.md` §3.1):
- E8 (`p`): tile $j = rNBLK + b$ = the PV DIMM's job $j$. Window $p$ = elements $pI \ldots pI + I - 1$ of the
  row-major $S \times D$ tile (element $e = mD + c$), on channel $p \bmod$ `P_CH`, $LW$ words of 5 lanes of `VWP` bits.
- E6 (`la`): block $j$ with $b \ge 1$ is unit $r(NBLK - 1) + b - 1$; window $m$ = row $m$, one 9-bit word.
- E7 (`ll`): run $r$ is unit $r$; one window of $S$ words, word $m$ = $\log l$ of row $m$.
- E9 (`o`): run $r$ is unit $r$; window $w$ = epilogue wave $w$, $LW$ words. Word $q$ carries word $q$ of window
  $p = wN_{OC} + i$ in lanes $5i \ldots 5i + 4$ (`o_d` bits $40i \ldots 40i + 39$). Lanes past a window's $I$
  elements, and crossbars without a window ($i \ge LWC$ in the last wave), are 0. Element $k$ of window $p$ is
  element $pI + k$ of the run's row-major $S \times D$ tile. The consumer (the O projection) reorders.

## 4 Structure

```text
 p_v/p_d (P_CH) -> [FIFO per channel, FD x 5 VWP, 1W1R, registered read] ----------------+
 la_v/la_d      -> [alpha ring: 2 slots x S x 9 bit, flip-flops; unit u in slot u mod 2] |
 ll_v/ll_d      -> [log l table: S x 8 bit, flip-flops]                                   |
                                                                                          v
 LOG group (N_OC identity crossbars, I x I, leader = crossbar 0; engine: jobs (b, w) in order)
     feed o' = (b ? (so ? -e : e) << SO : 0) + o_part -> oq = min(|o'| >> SO, 127), so -> so_mem
     drain lo -> lo_mem
 EXP group (N_OC identity crossbars, I x I; engine: rescale (b >= 1, w), then the epilogue (w), in order)
     rescale  feed t  = clamp8(log a + lo)     drain e  -> e_mem
     epilogue feed tE = clamp8(lo - log l)     drain e_E -> O8 = sat8(+-e_E) -> register -> E9 (o_v, o_d word i)
 per crossbar: lo_mem, e_mem (W 2^ceil(log2 LW) words x 40 bit), so_mem (same depth x 5 bit), address {w, word}
```

The two groups and their engines are v2's, unchanged. New: the stream inputs of the $\alpha$ ring and the $\log l$
table, the credits, the E9 output, the per-channel FIFO write enables.

## 5 Timing rules

Cycle 0 is the first cycle after reset. A signal "at $t$" is high (or a register reads its value) in cycle $t$.

| name | value | rule |
|---|---|---|
| `FIFO_LAT` | 1 | an E8 word valid at $t$ is counted in its FIFO from $t + 1$ |
| `LA_LAT`, `LL_LAT` | 1, 1 | an E6 / E7 unit whose last word is at $t$ is present from $t + 1$ |
| `PRE` | 2 | a job decided at $d$: its feed's first word reaches the crossbars at $d + 2$ (the pass's `load_start`) |
| `JOB_DONE_LAT` | 1 | a job whose last drain word is at $t$ counts for the other group's readiness from $t + 1$ |
| `P_CR_LAT` | 1 | `p_cr` one cycle after the read of the last word of a block's last LOG job, at $d + LW - 1$ |
| `LA_CR_LAT`, `LL_CR_LAT` | 2, 2 | `la_cr` / `ll_cr` two cycles after the read of the last word of the unit's last rescale / epilogue job, at $d + LW - 1$ |
| `O8_LAT` | 2 | the run's last epilogue drain word at $t$: its $O_8$ word is on E9 and the run ends ($t_{o8}$) at $t + 2$ |
| `RESTART_LAT` | 1 | the next run's LOG jobs may decide from $t_{o8} + 1$ (the run state restarts) |
| `E9_LAT` | 2 | an epilogue drain word at $t$ is on E9 at $t + 2$ (the drain stage, then the output register) |
| `RUN_DONE_LAT` | 1 | `run_done` one cycle after the run's end: at $t_{o8} + 1$ |

**Inputs.** Tile $j$'s window $p$ has its first word at $t_{arr}(j, p)$ and its last at $t_{arr}(j, p) + LW - 1$.
$t_{la}(u)$ = the last word of E6 unit $u$; $t_{ll}(r)$ = the last word of E7 unit $r$.

**Jobs.** Each group places its jobs in order. A job is decided at the first cycle $d \ge$ its ready time at which the
group's leader can take a burst ($d \ge$ the previous pass's `msb_fire` + 1). Its pass is the crossbar pass grammar
(`ip/xbar/model.py`) on the group's state with `load_start` $= d +$ `PRE`. The job's windows are $p = wN_{OC} + i$,
$i < N_{OC}$ (the last wave: $i < LWC$). Ready times, run $r$:
- LOG $(b, w)$: $\max_p t_{arr}(rNBLK + b, p) + LW - 1 +$ `FIFO_LAT`; for $b \ge 1$ also
  `drain_end`(rescale $(b, w)$) + `JOB_DONE_LAT`; for $b = 0$, $r \ge 1$ also $t_{o8}(r - 1) +$ `RESTART_LAT`.
- rescale $(b \ge 1, w)$: $t_{la}(u) +$ `LA_LAT` with $u = r(NBLK - 1) + b - 1$; `drain_end`(LOG $(b - 1, w)$) +
  `JOB_DONE_LAT`.
- epilogue $(w)$: $t_{ll}(r) +$ `LL_LAT`; `drain_end`(LOG $(NBLK - 1, w)$) + `JOB_DONE_LAT`.

The EXP group's order in a run is rescale $(1, 0 \ldots W-1)$, ..., rescale $(NBLK - 1, \ldots)$, epilogue
$(0 \ldots W-1)$; the LOG group's is LOG $(0, \ldots)$, ..., LOG $(NBLK - 1, \ldots)$.

**Credits.** `p_cr` (unit $j$) at $d$(LOG $(b, W - 1)$) $+ LW - 1 +$ `P_CR_LAT`; `la_cr` (unit $u$) at
$d$(rescale $(b, W - 1)$) $+ LW - 1 +$ `LA_CR_LAT`; `ll_cr` (unit $r$) at $d$(epilogue $(W - 1)$) $+ LW - 1 +$
`LL_CR_LAT`.

**E9.** $t_{o8}(r)$ = `drain_end`(epilogue $(W - 1)$) + `O8_LAT`. E9 window $w$ of run $r$ (epilogue wave $w$) has
its first word at `drain_start`(epilogue $w$) + `E9_LAT` and $LW$ words back to back, so the run's last word is at
$t_{o8}(r)$. `run_done` at $t_{o8}(r) +$ `RUN_DONE_LAT`.

Model checks: the $\alpha$ ring and the $\log l$ table are never written while a job still needs the unit (follows
from the credits).

## 6 Memories

| memory | organization | size |
|---|---|---|
| $o_{part}$ FIFOs | `P_CH` × $FD$ words × $5\,$`VWP` bits, 1W1R, registered read (`ip/lib` `att_fifo`) | $2 T_W LW$ words in use |
| lo_mem, e_mem | per crossbar $W \cdot 2^{\lceil \log_2 LW \rceil}$ words × 40 bits, 1W1R | |
| so_mem | per crossbar $W \cdot 2^{\lceil \log_2 LW \rceil}$ words × 5 bits | |
| $\alpha$ ring, $\log l$ table | $2 S$ × 9 and $S$ × 8 bits, flip-flops | |

## 7 Counts and resources

- Passes per block: $P_{oc}$ LOG ($I \times I$); for $b \ge 1$, $P_{oc}$ EXP ($I \times I$). Per run: $P_{oc}$ EXP
  (the epilogue).
- CLB per block: adds $S D$ ($o'$) $+ S D$ ($\log\alpha + lo$, $b \ge 1$); selects $S D$ ($oq$) $+ S D$ (the clamp
  of $t$, $b \ge 1$). Per run: adds $S D$ ($lo - \log l$), selects $2 S D$ ($t_E$, $O_8$).
- Memory per block: $P_{oc} \cdot 5 LW$ bytes written into lo_mem and read back; for $b \ge 1$ the same into e_mem.
  Per run: the epilogue reads lo_mem ($P_{oc} \cdot 5 LW$ bytes); its $O_8$ goes to E9, not to a memory. FIFO words
  per tile: $P_{oc} LW$, also counted as memory traffic: each word's $\lceil 5\,$`VWP`$/8 \rceil$ bytes written once
  and read once (2026-10-15).
- Crossbars: $N_{OC}$ LOG + $N_{OC}$ EXP = $2 N_{OC}$.

## 8 Gates

- GATE 1 (case writer): every stage of the model == `attention_ref.ocarry_stage_values` and `epilogue`, and the
  $O_8$ decoded from every sent E9 window == the oracle's (the other lanes 0), every run.
- Closed form: with the v2 station's inputs (o_part windows, $\log\alpha$ and $\log l$ ready times), the job events
  reproduce v2 `ocarry_cycle_model` (every LOG and EXP job's decide, `load_start`, `msb_fire`, drains; `op_free`,
  `la_free`, the run end), with v2's O-feed hold off.
- GATE 2 (64×64): the RTL's E9 words, every port event, `run_done` and the total cycles == the model's. (2026-10-15:
  the testbenches still check the old row format; they move to E9 windows before the next RTL run.)
- VTR smoke (128×128): crossbar count == $2 N_{OC}$; Fmax, CLB, memory, path #1.

## 9 The instance in the head

| parameter | value |
|---|---|
| `S`, `D`, `NBLK` | $\min(n, I)$, $d$, $n/B_{kv}$ |
| `SO`, `LN` | $\log_2 n$, $\log_2 n$ |
| `P_CH`, `VWP` | the PV DIMM's $NG = N_{EP}/B_{kv}$; its output width $9 + \log_2 B_{kv}$ = 13 |
| `N_OC` | the floor rule (`architecture.md` §5.2): the smallest $N_{OC}$ with $T(\lceil P_{oc}/N_{OC} \rceil) \le \max(\lceil S B_{kv}/5 \rceil, T(1))$ |

## 10 Refinements of `architecture.md` §4.6

1. **The $\alpha$ ring is in arrival order across runs:** unit $u$ goes to slot $u \bmod 2$, and the next rescale job
   uses the oldest unit not yet used up. The $\alpha$ ring and the $\log l$ table are not cleared at a run's end: with
   two $\alpha$ credits, the next run's first unit can arrive before the run ends.
2. (removed by item 7) The O reader started after the run's last $O_8$ word and read $\lceil D/5 \rceil + 1$ words
   per row from e_mem, sharing its read port with the LOG feed.
3. **The table credits keep v2's timing** (two cycles after the last word's read of the unit's last job), although the
   tables are read only at the job decisions.
4. (superseded by item 7) `run_done` one cycle after the run's last E9 word; e_mem free one cycle after the run's last
   read.
5. **One write enable per FIFO** (`p_v` per channel), as the port framing defines; v2 had one valid for all groups.
6. **FIFO sets as one-hot registers** (VTR quality; no cycle changes): the FIFOs a LOG job reads, and the FIFO of each
   crossbar in the compute stage, are registered one-hot vectors that rotate by $N_{OC}$ per wave. They replace v2's
   modular base arithmetic $(f + P_{CH} - base) \bmod P_{CH}$ and its priority select. The LOG feed's $o'$ adders use
   $\max(9 + SO, VWP) + 1$ bits (v2: 34). 128×128 smoke: 103.3 → 156.3 MHz.
7. **No O reader: E9 carries the final $O_8$ values as they are computed** (2026-10-15, user: the O read-out was set
   by the 40-bit port, not by the crossbars or the mapping; the O projection reads the results):
   - the epilogue's $O_8$ words go to E9, registered, two cycles after each drain word: `o_v` (1 bit) and `o_d`
     ($N_{OC}$ × 40 bits, word $i$ from crossbar $i$); no $O_8$ memory, no read-out, nothing waits for one;
   - e_mem holds only the rescale's $e$; `run_done` pulses one cycle after the run's end, so the head's `done` is at
     the last run's $t_{o8} + 2$;
   - the 72 DSE builds: the 42 with $R_b = 1$ keep their cycles ($T = t_{o8} + 2$); the 30 with $R_b \ge 2$ end
     0–3.0 % earlier (median 1.6 %), because the earlier read-outs no longer hold the next run;
   - an intermediate version the same day kept the reader on its own memory (`o8_mem`, 12 memory blocks at 128×128);
     it was replaced by this one.
