# `softmax_online_acc` — online softmax accumulator (v3 module spec)

Status: 2026-10-14, F3 / phase C, **validated**: GATE 2 19/19 (plus 5/5 bring-up) at 64×64, VTR smoke at 128×128
(156.3 MHz) (`LEDGER.md`). 2026-10-15: the O reader is removed and E9 carries the final $O_8$ values as they are
computed (§10 item 7); RTL and model changed, the RTL not simulated after it (you), the head VTR smoke in
`LEDGER.md`.

**2026-10-22, the bit-plane pass** (§10 item 8):
- every port carries bit planes (`ip/xbar/spec.md`, `docs/architecture.md` §3.1);
- the LOG feed corner-turns its inputs, computes in $I/8$ lanes, and corner-turns $oq$ back into planes;
- the EXP feeds add bit-serially and hold 8 planes for the clamp;
- lo_mem and e_mem hold plane words;
- E9 is one plane window per epilogue wave.

The behavior model (`simulator/models/softmaxAccumulator.py`) implements this version. Both RTL twins still have the
40-bit version and follow in the RTL pass, so GATE 2 pins this version's latencies only after that.

This is the module's contract: the behavior model and both RTL twins (`ip/softmax/rtl/{sim,vtr}/softmax_online_acc.v`)
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

$\mathrm{EXP}$ is read as an unsigned byte, $\mathrm{LOG}$ as a signed byte (`architecture.md` §2.1). Output: $O_8$.
Value contract (GATE 1): `attention_ref.ocarry_stage_values` (every block stage) and `attention_ref.epilogue` ($t_E$,
$e_E$, $O_8$), every run. The values are unchanged by the bit-plane pass.

## 2 Parameters

| group | parameter | meaning |
|---|---|---|
| shape | `S, D, R, C, P` | rows per run, row width, crossbar, bit slices (`BUF` removed 2026-10-22) |
| run | `NBLK, SO, LN` | blocks per run; the shifts of $o$ and $l$ ($2^{SO}$, $2^{LN}$) |
| crossbars | `N_OC` | crossbars per group (one LOG group and one EXP group) |
| input | `P_CH`, `VWP` | $o_{part}$ channels (= the PV DIMM's $NG$); $o_{part}$ element width |

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
| $L_P$ | $\lceil$`VWP`$/8 \rceil$: lanes of an E8 plane word (2 in the head) |
| $T_W$ | $P_{oc}/$`P_CH`: windows per channel per tile |
| $FD$ | $2^{\lceil \log_2(2 T_W P) \rceil}$: FIFO depth (plane words, per channel; two tiles) |
| $NJ$ | $NBLK \cdot W$: jobs per group per run |
| $SHL$, $SHR$ | $\max(SO - LN, 0)$, $\max(LN - SO, 0)$ |

Until 2026-10-22: $LW = \lceil I/5 \rceil$ words per window, and the FIFOs and slot memories in 40-bit words.

Guards (the RTL's elaboration guards; the model raises on the same sets):
- $S$, $D$, $I$, `P_CH` powers of two; $8 \le S \le I$; $D \le I$; $I \mid S D$; $P = 8$ (until 2026-10-22 `BUF` = 40);
- $1 \le N_{OC} \le P_{oc}$; $N_{OC} \le$ `P_CH` (a job's windows are in different FIFOs); `P_CH` $\mid P_{oc}$;
- $NBLK \ge 1$; $SO, LN \ge 0$; $1 \le$ `VWP` $\le 16$ (at most two lanes; until 2026-10-22 $\le 32$).

## 3 Ports

| port | width | dir | discipline | unit | `CAP` |
|---|---|---|---|---|---|
| `p_v`, `p_d` / `p_cr` | `P_CH`, `P_CH` · $L_P I$ / 1 | in / out | push + credit | a tile: one block's $o_{part}$ ($P_{oc}$ windows of $I$) | 2 tiles |
| `la_v`, `la_d` / `la_cr` | 1, $2S$ / 1 | in / out | push + credit | a block $b \ge 1$: one window of $S$ rows (9 bits) | 2 blocks |
| `ll_v`, `ll_d` / `ll_cr` | 1, $S$ / 1 | in / out | push + credit | a run: one window of $S$ rows | 1 run |
| `o_v`, `o_d` | 1, $N_{OC} I$ | out | push | a run: $W$ windows, one per epilogue wave, of $N_{OC} I$ int8 | — |
| `run_done` | 1 | out | | pulse one cycle after a run's end ($t_{o8}$: its last $O_8$ plane on E9) | |
| `weight_in`, `weight_en` | 8, 1 | in | | simulation twin only (the testbench preloads, R8) | |

Framing (`architecture.md` §3.1, plane framing since 2026-10-22): every window is 8 plane words on consecutive
cycles, LSB first; an element wider than 8 bits uses lane 1 for bits $8 + b$.
- E8 (`p`): tile $j = rNBLK + b$ = the PV DIMM's job $j$. Window $p$ = elements $pI \ldots pI + I - 1$ of the
  row-major $S \times D$ tile (element $e = mD + c$), on channel $p \bmod$ `P_CH`, $L_P$ lanes.
- E6 (`la`): block $j$ with $b \ge 1$ is unit $r(NBLK - 1) + b - 1$: one window, element $m$ = row $m$'s
  $\log\alpha$.
- E7 (`ll`): run $r$ is unit $r$: one window, element $m$ = $\log l$ of row $m$.
- E9 (`o`): run $r$ is unit $r$; window $w$ = epilogue wave $w$: elements $iI \ldots iI + I - 1$ are crossbar $i$'s
  window $p = wN_{OC} + i$ (`o_d` bits $iI \ldots iI + I - 1$). Crossbars without a window ($i \ge LWC$ in the last
  wave) send 0. Element $k$ of window $p$ is element $pI + k$ of the run's row-major $S \times D$ tile. The consumer
  (the O projection) reorders.

Until 2026-10-22: 40-bit words (E8 5 lanes of `VWP` bits), E6 one row per window, E7 one row per word, E9 $LW$
words of $5 N_{OC}$ bytes.

## 4 Structure

```text
 p_v/p_d (P_CH) -> [FIFO per channel, FD x L_P I, 1W1R, registered read] ------------------+
 la_v/la_d      -> [alpha ring: 2 slots x S x 9 bit, flip-flops, written plane by plane]    |
 ll_v/ll_d      -> [log l table: S x 8 bit, flip-flops, written plane by plane]             |
                                                                                            v
 LOG group (N_OC identity crossbars, I x I, leader = crossbar 0; engine: jobs (b, w) in order)
     feed: the o_part, e and so planes -> [CORNER-TURN A, x2] -> I/8 lanes:
           o' = (b ? (so ? -e : e) << SO : 0) + o_part, oq = min(|o'| >> SO, 127), so -> [CORNER-TURN B, x2]
           -> one 8-plane burst of oq; so -> so_mem
     drain lo (planes) -> lo_mem
 EXP group (N_OC identity crossbars, I x I; engine: rescale (b >= 1, w), then the epilogue (w), in order)
     feed: lo planes + the table's planes, bit-serial, LSB first -> [8-PLANE HOLD, x2] -> clamp8 -> burst
     rescale  t  = clamp8(log a + lo)     drain e (planes) -> e_mem
     epilogue tE = clamp8(lo - log l)     drain e_E -> [8-PLANE HOLD] -> O8 = sat8(+-e_E) -> register -> E9
 per crossbar: lo_mem, e_mem (W P words x I bits, plane words), so_mem (W words x I bits), address {w, plane}
```

The two groups and their engines are v2's. New in v3: the stream inputs of the $\alpha$ ring and the $\log l$ table,
the credits, the E9 output, the per-channel FIFO write enables. New with the bit-plane pass: the feeds below.

**The bit-plane feeds** (2026-10-22):
- **LOG feed.** The job's $o_{part}$ planes (FIFOs), $e$ planes (e_mem, $b \ge 1$) and $so$ plane (so_mem) are read
  one plane per cycle into corner-turn A. Once A is whole, its $I$ elements go through the arithmetic $Q_W = I/8$ per
  cycle into corner-turn B, and B leaves as the crossbar's burst. A and B are double-buffered, because the next job's
  feed starts while this one's burst runs.
- **EXP feeds.** The job's lo planes (lo_mem) and the table's planes ($\log\alpha$, or $-\log l$ as $\sim\log l + 1$)
  go through one bit-serial adder per element, LSB first: the carry is cleared at plane 0. The sum planes are held.
  After plane 7 the per-element clamp flags are set, and the held planes leave as the burst, clamped. The hold is
  double-buffered.
- **$O_8$.** The epilogue's drain planes ($e_E$) are held. After plane 7, the per-element saturation flags of
  $\pm e_E$ are set, and the planes of $O_8$ leave on E9 through the output register. The negation is bit-serial,
  LSB first.

## 5 Timing rules

Cycle 0 is the first cycle after reset. A signal "at $t$" is high (or a register reads its value) in cycle $t$.

| name | value | rule |
|---|---|---|
| `FIFO_LAT` | 1 | an E8 plane valid at $t$ is counted in its FIFO from $t + 1$ |
| `LA_LAT`, `LL_LAT` | 1, 1 | an E6 / E7 unit whose last plane is at $t$ is present from $t + 1$ |
| `PRE_LOG` | 17 | a LOG job decided at $d$ reads its planes at $d \ldots d + 7$. With the registered read, A is whole at $d + 9$; the 8 groups of $Q_W$ elements are registered into B at $d + 9 \ldots d + 16$; the burst starts at $d + 17$ ($2P + 1$) |
| `PRE_EXP` | 10 | an EXP job decided at $d$ reads its lo planes at $d \ldots d + 7$. The held sum planes are complete at $d + 9$, the clamp flags register at $d + 9$, and the burst starts at $d + 10$ ($P + 2$) |
| `JOB_DONE_LAT` | 1 | a job whose last drain plane is at $t$ counts for the other group's readiness from $t + 1$ |
| `P_CR_LAT` | 1 | `p_cr` one cycle after the last plane read of a block's last LOG job, at $d + P - 1$ |
| `LA_CR_LAT`, `LL_CR_LAT` | 2, 2 | `la_cr` / `ll_cr` two cycles after the last lo plane read of the unit's last rescale / epilogue job, at $d + P - 1$ |
| `RESTART_LAT` | 1 | the next run's LOG jobs may decide from $t_{o8} + 1$ (the run state restarts) |
| `E9_LAT` | 10 | an epilogue pass's drain plane $b$ at `drain_start` $+ b$: E9 plane $b$ at `drain_start` + `E9_LAT` $+ b$. The hold is complete at `drain_start` + 8, the flags register at + 9, and the output register at + 10 ($P + 2$) |
| `RUN_DONE_LAT` | 1 | `run_done` one cycle after the run's end: at $t_{o8} + 1$ |

Until 2026-10-22: `PRE` 2 for both groups, `E9_LAT` and `O8_LAT` 2 (the drain word to E9).

**Inputs.** Tile $j$'s window $p$ has its first plane at $t_{arr}(j, p)$ and its last at $t_{arr}(j, p) + P - 1$.
$t_{la}(u)$ = the last plane of E6 unit $u$; $t_{ll}(r)$ = the last plane of E7 unit $r$.

**Jobs.** Each group places its jobs in order. A job is decided at the first cycle $d \ge$ its ready time from which
its pass finds the group's leader ready: $d + $ `PRE` $\ge$ the previous pass's `acam_done` + 1, i.e.
$d = \max(ready,\ \text{leader ready} - \text{PRE})$, and the pass's `load_start` $= d +$ `PRE`. So a job's feed is
prepared while the previous pass runs, and back-to-back jobs decide one pass period (10 cycles) apart. The job's
windows are $p = wN_{OC} + i$, $i < N_{OC}$ (the last wave: $i < LWC$). Ready times, run $r$:
- LOG $(b, w)$: $\max_p t_{arr}(rNBLK + b, p) + P - 1 +$ `FIFO_LAT`; for $b \ge 1$ also
  `drain_end`(rescale $(b, w)$) + `JOB_DONE_LAT`; for $b = 0$, $r \ge 1$ also $t_{o8}(r - 1) +$ `RESTART_LAT`.
- rescale $(b \ge 1, w)$: $t_{la}(u) +$ `LA_LAT` with $u = r(NBLK - 1) + b - 1$; `drain_end`(LOG $(b - 1, w)$) +
  `JOB_DONE_LAT`.
- epilogue $(w)$: $t_{ll}(r) +$ `LL_LAT`; `drain_end`(LOG $(NBLK - 1, w)$) + `JOB_DONE_LAT`.

Until 2026-10-22 a job was decided only once the leader could take a burst (the previous pass's `msb_fire` + 1), with
`PRE` 2.

The EXP group's order in a run is rescale $(1, 0 \ldots W-1)$, ..., rescale $(NBLK - 1, \ldots)$, epilogue
$(0 \ldots W-1)$; the LOG group's is LOG $(0, \ldots)$, ..., LOG $(NBLK - 1, \ldots)$.

**Credits.** `p_cr` (unit $j$) at $d$(LOG $(b, W - 1)$) $+ P - 1 +$ `P_CR_LAT`; `la_cr` (unit $u$) at
$d$(rescale $(b, W - 1)$) $+ P - 1 +$ `LA_CR_LAT`; `ll_cr` (unit $r$) at $d$(epilogue $(W - 1)$) $+ P - 1 +$
`LL_CR_LAT`. A table unit is read until its job's clamp flags at $d + P + 1$, which is when its credit pulses, so the
producer's next unit arrives after it.

**E9.** E9 window $w$ of run $r$ (epilogue wave $w$) has its first plane at `drain_start`(epilogue $w$) + `E9_LAT`
and its 8 planes back to back. $t_{o8}(r)$ = the run's last E9 plane = `drain_start`(epilogue $(W - 1)$) + `E9_LAT`
$+ P - 1$. `run_done` at $t_{o8}(r) +$ `RUN_DONE_LAT`.

Model checks: the $\alpha$ ring and the $\log l$ table are never written while a job still needs the unit (follows
from the credits).

## 6 Memories

| memory | organization | size |
|---|---|---|
| $o_{part}$ FIFOs | `P_CH` × $FD$ words × $L_P I$ bits, 1W1R, registered read (`ip/lib` `att_fifo`) | $2 T_W P$ plane words in use |
| lo_mem, e_mem | per crossbar $W P$ words × $I$ bits (plane words), 1W1R, address {w, plane} | |
| so_mem | per crossbar $W$ words × $I$ bits (one plane per window) | |
| corner-turns, holds | LOG feed: 2 A ($I \times ($`VWP` $+ 9)$ bits: $o_{part}$, $e$, $so$) and 2 B ($I \times 8$) per crossbar; EXP feed: 2 holds ($I \times 8$) per crossbar; $O_8$: one hold ($I \times 8$) per crossbar; flip-flops | |
| $\alpha$ ring, $\log l$ table | $2 S$ × 9 and $S$ × 8 bits, flip-flops | |

The $N_{OC}$ copies of lo_mem, e_mem and so_mem share their addresses and write enables (lock-step groups). Until
2026-10-22: FIFOs of $5\,$`VWP` bits, lo_mem and e_mem of $W 2^{\lceil \log_2 LW \rceil}$ words × 40 bits, so_mem × 5
bits.

## 7 Counts and resources

- Passes per block: $P_{oc}$ LOG ($I \times I$); for $b \ge 1$, $P_{oc}$ EXP ($I \times I$). Per run: $P_{oc}$ EXP
  (the epilogue).
- CLB per block: adds $S D$ ($o'$) $+ S D$ ($\log\alpha + lo$, $b \ge 1$); selects $S D$ ($oq$) $+ S D$ (the clamp
  of $t$, $b \ge 1$). Per run: adds $S D$ ($lo - \log l$), selects $2 S D$ ($t_E$, $O_8$).
- Memory per block: $P_{oc} I$ bytes written into lo_mem ($P$ plane words of $I/8$ bytes per window) and read back;
  for $b \ge 1$ the same into e_mem. Per run: the epilogue reads lo_mem ($P_{oc} I$ bytes); its $O_8$ goes to E9,
  not to a memory. FIFO words per tile: $P_{oc} P$ plane words, also counted as memory traffic: each word's
  $L_P I/8$ bytes written once and read once (2026-10-15). Until 2026-10-22: $P_{oc} \cdot 5 LW$ bytes per window set,
  and $\lceil 5\,$`VWP`$/8 \rceil$ bytes per FIFO word.
- Crossbars: $N_{OC}$ LOG + $N_{OC}$ EXP = $2 N_{OC}$.

## 8 Gates

- GATE 1 (case writer): every stage of the model == `attention_ref.ocarry_stage_values` and `epilogue`, and the
  $O_8$ decoded from every sent E9 window == the oracle's (the other elements 0), every run.
- Closed form (the model's test, written by hand from §5): with every input early, every job's decision,
  `load_start` and drain follow the job rules of §5 group by group, and the run's end is the last epilogue's
  `drain_start` + `E9_LAT` + 7. (Until 2026-10-22 the witness was v2's `ocarry_cycle_model`, which times the 40-bit
  pass.)
- GATE 2 (64×64): the RTL's E9 planes, every port event, `run_done` and the total cycles == the model's.
- VTR smoke (128×128): crossbar count == $2 N_{OC}$; Fmax, CLB, memory, path #1.

## 9 The instance in the head

| parameter | value |
|---|---|
| `S`, `D`, `NBLK` | $\min(n, I)$, $d$, $n/B_{kv}$ |
| `SO`, `LN` | $\log_2 n$, $\log_2 n$ |
| `P_CH`, `VWP` | the PV DIMM's $NG = N_{EP}/B_{kv}$; its output width $9 + \log_2 B_{kv}$ = 13 |
| `N_OC` | the floor rule (`architecture.md` §5.2), held at its 40-bit form: the smallest $N_{OC}$ with $T(\lceil P_{oc}/N_{OC} \rceil) \le \max(\lceil S B_{kv}/5 \rceil, T(1))$, $T$ the 40-bit pass |

## 10 Refinements of `architecture.md` §4.6

1. **The $\alpha$ ring is in arrival order across runs:** unit $u$ goes to slot $u \bmod 2$, and the next rescale job
   uses the oldest unit not yet used up. The $\alpha$ ring and the $\log l$ table are not cleared at a run's end: with
   two $\alpha$ credits, the next run's first unit can arrive before the run ends.
2. (removed by item 7) The O reader started after the run's last $O_8$ word and read $\lceil D/5 \rceil + 1$ words
   per row from e_mem, sharing its read port with the LOG feed.
3. **The table credits keep v2's timing** (two cycles after the last plane read of the unit's last job).
4. (superseded by item 7) `run_done` one cycle after the run's last E9 word; e_mem free one cycle after the run's last
   read.
5. **One write enable per FIFO** (`p_v` per channel), as the port framing defines; v2 had one valid for all groups.
6. **FIFO sets as one-hot registers** (VTR quality; no cycle changes): the FIFOs a LOG job reads, and the FIFO of each
   crossbar in the compute stage, are registered one-hot vectors that rotate by $N_{OC}$ per wave. They replace v2's
   modular base arithmetic $(f + P_{CH} - base) \bmod P_{CH}$ and its priority select. The LOG feed's $o'$ adders use
   $\max(9 + SO, VWP) + 1$ bits (v2: 34). 128×128 smoke: 103.3 → 156.3 MHz.
7. **No O reader: E9 carries the final $O_8$ values as they are computed** (2026-10-15, user: the O read-out was set
   by the 40-bit port, not by the crossbars or the mapping; the O projection reads the results):
   - the epilogue's $O_8$ goes to E9 as it drains: no $O_8$ memory, no read-out, nothing waits for one;
   - e_mem holds only the rescale's $e$; `run_done` pulses one cycle after the run's end;
   - the 72 DSE builds: the 42 with $R_b = 1$ kept their cycles ($T = t_{o8} + 2$); the 30 with $R_b \ge 2$ ended
     0–3.0 % earlier (median 1.6 %), because the earlier read-outs no longer held the next run;
   - an intermediate version the same day kept the reader on its own memory (`o8_mem`, 12 memory blocks at 128×128);
     it was replaced by this one.
8. **The bit-plane pass** (user, 2026-10-21/22; `ip/xbar/spec.md`).
   - **Ports:** every port carries 8 plane words per window: `p_d` $L_P I$ bits per channel, `la_d` $2S$, `ll_d` $S$,
     `o_d` $N_{OC} I$. `BUF` is removed.
   - **LOG feed:** corner-turn A, the arithmetic in $I/8$ lanes, corner-turn B (`PRE_LOG` 17). A full-width
     arithmetic would give 10 cycles, but it needs $N_{OC} \cdot I$ lanes instead of $N_{OC} \cdot I/8$.
   - **EXP feeds:** bit-serial adds with an 8-plane hold for clamp8 (`PRE_EXP` 10).
   - **Decision ahead of the leader:** a job decides `PRE` cycles before its leader is ready, so the feeds keep the
     crossbars back to back; the feed buffers are doubled for it.
   - **lo_mem, e_mem:** plane words ($WP$ × $I$ bits); so_mem one plane per window.
   - **E9:** one window of $N_{OC} I$ elements per epilogue wave, through an 8-plane hold for the saturation
     (`E9_LAT` 10).
   - **Values, counts (except the FIFO and slot-memory bytes, §7) and the job rules' structure are unchanged.**
