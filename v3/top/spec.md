# `att_ctrl` and `att_head` — one attention head (v3 spec)

Status: 2026-10-15, F4 / phase D, **closed**: model ALL PASS; RTL bring-up 5/5 and gate H 6/6 at 64×64 (every
event, the cycles and the values equal the model's); VTR smoke at 128×128 (278 / 278 crossbars, 116.5 MHz); `att_ctrl`
rtl-pinned (`LEDGER.md`). This is the head's contract: the
controller's behavior model `top/model.py`, the head workload `sim/workloads/attention_head.py`, and both RTL twins
(`top/rtl/{sim,vtr}/{att_ctrl,att_head}.v`) implement it, and gate H pins every latency named here. Built on the
validated modules (`ip/gemm`, `ip/dimm/spec.md`, `ip/softmax/spec.md`, `ip/softmax/acc_spec.md`) and
`docs/architecture.md` §4.1, §4.7 and §5. Refinements of §4.7 made while writing this spec are listed in §10.

**2026-10-22, the bit-plane pass** (§10 item 6): X and $O_8$ move as bit planes, and a token burst is 8 plane words
(`ip/xbar/spec.md`, `docs/architecture.md` §3.1). The behavior models implement this version. 2026-10-23: both RTL
twins implement it (`LOADX` = 8, `BUF` removed); gate H 6/6 and the bring-up 5/5 at 64×64.

**2026-10-23, the soft-logic trims** (your go): E4 and E8 carry value words (`docs/architecture.md` §3.1), the DIMM
drain registers each count once, and the accumulator's corner-turns are single buffers
(`ip/dimm/spec.md` §10 item 10, `ip/softmax/spec.md` §10 item 8, `ip/softmax/acc_spec.md` §10 item 10). The wiring
and every port width are unchanged.

## 1 Function

One head of textbook attention on NL-DPE crossbars (`architecture.md` §1):

$$
O_8 = \mathrm{softmax}\big(Q K^\top \cdot 2^{-5}\big)\, V, \qquad Q, K, V = \mathrm{trunc8}(X W_Q),\ \mathrm{trunc8}(X W_K),\ \mathrm{trunc8}(X W_V)
$$

with the integer forms of `architecture.md` §2.2. Input $X$ ($n \times d_m$ int8, one token row per burst), output
$O_8$ ($n \times d$ int8, token rows in order). Value contract (GATE 1): `attention_ref.head_stages`, stage `O8`,
bit-exact. There is no output projection.

`att_head` is wiring only. `att_ctrl` holds counters only. It decides:
- when a GEMM copy may take its next token;
- which jobs the two DIMMs run, in which order;
- when the online softmax starts a run;
- when the head is done.

## 2 Parameters

`att_head` takes the row's parameters (`experiment.md` §4): `NSEQ, DM, D, BKV, R, C, P, N_EXP, N_E, N_EP, N_OC,
P_PROJ, X_S` (`BUF` removed 2026-10-22).

Derived (no free numbers):

| name | value |
|---|---|
| $I$, $S$ | $\min(R, C)$; $\min(n, I)$ query rows per run |
| $R_b$, $B$ | $n/S$ runs; $n/B_{kv}$ key blocks |
| $V$, $ROWS_X$, $LOADX$ | $\lceil d_m/R \rceil$ GEMM row tiles; $\min(d_m, R)$ rows in use (P29); $P$ = 8 plane words per token burst (until 2026-10-22 $\lceil 8\,ROWS_X/40 \rceil$ words) |
| $LN = SO$ | $\log_2 n$ |
| $NG_{QK}$, $NG_{PV}$ | $N_E/d$, $N_{EP}/B_{kv}$: the DIMMs' output channels (1 and $2d/B_{kv}$ at the fixed mapping: $N_E = d$ since 2026-10-23, $N_{EP} = 2d$) |
| $NB_A$ | $I/B_{kv}$: rows per QK A staging group |
| $T_X$ | $n/P_{proj}$ tokens per copy |

The modules' parameters:

| instance | module | parameters |
|---|---|---|
| `gq`$p$, `gk`$p$, `gv`$p$ ($p < P_{proj}$) | `gemm_top` | $K = d_m$, $N = d$, `R`, `C`, `ROWS_USED` = $ROWS_X$ ($V = 1$) or $R$, `COLS_USED` = $d$ |
| `qk` | `dimm_tile_top` | `ip/dimm/cases.py` `qk_params`$(S, d, B_{kv}, R, C, N_E, P_{proj}, R_b, B)$: A int8 row frame, `A_CH` = $P_{proj}$, `A_SLOTS` = $R_b$; B int8, `TRANS_B` = 1, `B_CH` = $P_{proj}$, `B_SLOTS` = $B$; `NARROW` = 5 |
| `sm` | `softmax_online_top` | $S$, $B_{kv}$, $N = n$, `R`, `C`, $N_{EXP}$, `S_CH` = $NG_{QK}$, `LP_CAP` = 2, `LA_CAP` = 2, `LL_CAP` = 1 |
| `pv` | `dimm_tile_top` | `pv_params`$(S, d, B_{kv}, R, C, N_{EP}, N_{EXP}, P_{proj}, B)$: A log, window frame, `A_CH` = $N_{EXP}$, `A_SLOTS` = 2, released; B int8, `TRANS_B` = 0, `B_CH` = $P_{proj}$, `B_SLOTS` = $B$ |
| `acc` | `softmax_online_acc` | $S$, $D = d$, `R`, `C`, $N_{OC}$, `NBLK` = $B$, `SO` = `LN` = $\log_2 n$, `P_CH` = $NG_{PV}$, `VWP` = 13; $N_{OC}$ by the floor rule held at its pre-bit-plane form (`acc_spec.md` §9) |
| `ctrl` | `att_ctrl` | `NSEQ`, `S`, `BKV`, `I`, `P_PROJ`, `LOADX` |

Guards: every module's guard holds, plus:
- $n$, $P_{proj}$ powers of two, $P_{proj} \le n$;
- $d \le C$ ($H = 1$ in every GEMM);
- $ROWS_X \ge d$ (a GEMM copy's row period covers the DIMM converter's period, `ip/dimm/spec.md` §3; since
  2026-10-22 both are one pass period, 10 cycles).

## 3 Ports

`att_head`:

| port | width | dir | discipline | unit |
|---|---|---|---|---|
| `x_rdy` | $P_{proj}$ | out | burst on ready | copy $p$ may start its next token burst in this cycle |
| `x_v`, `x_d` | $P_{proj}$, $P_{proj} V \cdot ROWS_X$ | in | | copy $p$'s token burst: $LOADX$ = 8 plane words, lane $v$ = rows $vR \ldots vR + ROWS_X - 1$ of the token ($ROWS_X$ bits); tokens $t \equiv p \pmod{P_{proj}}$ in order; across copies in token order (token $t$'s burst starts no earlier than token $t - 1$'s; bursts may share a cycle) |
| `o_v`, `o_d` | 1, $N_{OC} I$ | out | push | E9: the final $O_8$ values, one plane window of $N_{OC} I$ elements per epilogue wave (`acc_spec.md` §3); the O projection reads them |
| `start` / `busy`, `done` | 1 / 1, 1 | in / out | burst on ready | one run of the head per reset; `busy` falling is the start port's credit |
| `weight_in`, `weight_en` | 8, 1 | in | | simulation twin only (the testbench preloads every crossbar, `PLAN.md` R8) |

`att_ctrl`:

| port | width | dir | meaning |
|---|---|---|---|
| `start`, `busy`, `done` | 1 each | in, out, out | the head's run control |
| `x_v` / `x_rdy` | $P_{proj}$ / $P_{proj}$ | in / out | a copy's burst planes (a burst starts when `x_v` rises) / the copy may start a burst |
| `gq_rdy`, `gk_rdy`, `gv_rdy` | $P_{proj}$ each | in | the copies' GEMM `MSB_SA_Ready` |
| `qa_cr`, `qb_cr`, `pb_cr` | 1 each | in | the staging credits of the QK A, QK B and PV B ports |
| `qj_v`, `qj_a`, `qj_b`, `qj_ra` / `qj_rdy` | 1, ⌈log₂ $R_b$⌉, ⌈log₂ $B$⌉, 1 / 1 | out / in | the QK DIMM's job port |
| `pj_v`, `pj_a`, `pj_b`, `pj_ra` / `pj_rdy` | 1, 1, ⌈log₂ $B$⌉, 1 / 1 | out / in | the PV DIMM's job port |
| `sm_start` / `sm_busy` | 1 / 1 | out / in | the online softmax's run port |
| `run_done` | 1 | in | the accumulator's `run_done` |

(A field of width ⌈log₂ 1⌉ is 1 bit, as in the DIMM.)

## 4 Structure

`att_head` is the netlist of `architecture.md` §4.1. Every connection:

| from | to | edge |
|---|---|---|
| `x_v`$[p]$, `x_d`$[p]$ | `gq`$p$, `gk`$p$, `gv`$p$ `act_en`, `act_in`; `ctrl.x_v`$[p]$ | E0 (one burst into three GEMMs) |
| `gq`$p$/`gk`$p$/`gv`$p$ `MSB_SA_Ready` | `ctrl.gq_rdy`$[p]$/`gk_rdy`$[p]$/`gv_rdy`$[p]$ | |
| `ctrl.x_rdy` | `x_rdy` | |
| `gq`$p$ `data_out`, `reg_full` | `qk.a_d`, `a_v` channel $p$ | E1 |
| `gk`$p$ `data_out`, `reg_full` | `qk.b_d`, `b_v` channel $p$ | E2 |
| `gv`$p$ `data_out`, `reg_full` | `pv.b_d`, `b_v` channel $p$ | E3 |
| `qk.a_cr`, `qk.b_cr`, `pv.b_cr` | `ctrl.qa_cr`, `qb_cr`, `pb_cr` | E1–E3 credits |
| `ctrl.qj_*` / `qk.j_rdy` | `qk.j_*` / `ctrl.qj_rdy` | QK jobs |
| `ctrl.pj_*` / `pv.j_rdy` | `pv.j_*` / `ctrl.pj_rdy` | PV jobs |
| `qk.c_v`, `c_d` / `sm.s_cr` | `sm.s_v`, `s_d` / `qk.c_cr` | E4 (value words) |
| `ctrl.sm_start` / `sm.busy` | `sm.start` / `ctrl.sm_busy` | softmax runs |
| `sm.lp_v`, `lp_d` / `pv.a_cr` | `pv.a_v`, `a_d` / `sm.lp_cr` | E5 |
| `sm.la_*` / `acc.la_cr` | `acc.la_*` / `sm.la_cr` | E6 |
| `sm.ll_*` / `acc.ll_cr` | `acc.ll_*` / `sm.ll_cr` | E7 |
| `pv.c_v`, `c_d` / `acc.p_cr` | `acc.p_v`, `p_d` / `pv.c_cr` | E8 (value words) |
| `acc.o_v`, `o_d` | `o_v`, `o_d` | E9 |
| `acc.run_done` | `ctrl.run_done` | |
| `start` / `ctrl.busy`, `ctrl.done` | `ctrl.start` / `busy`, `done` | |

Unused module outputs (`busy`, `done` of the DIMMs and the softmax's `done`, `dpe_done`) are left open.

## 5 Timing rules (`att_ctrl`)

Cycle 0 is the first cycle after reset. A signal "at $t$" is high (or a register reads its value) in cycle $t$.

| name | value | rule |
|---|---|---|
| `RUN_LAT` | 1 | `start` taken at $t_s$: the head runs from $t_s + 1$ (admission, both job ports and the softmax start may act) |
| `CR_LAT` | 1 | a staging credit pulse (`qa_cr`, `qb_cr`, `pb_cr`) at $t$ counts for admission from $t + 1$ |
| `DONE_LAT` | 1 | `done` one cycle after the last run's `run_done`, i.e. two cycles after its last $O_8$ plane on E9 (`acc_spec.md` §10 item 7) |
| `BUSY_LAT` | 1 | `busy` falls one cycle after `done` (the start port's credit) |

**Admission.** Copy $p$'s $k$-th token is $t = p + k P_{proj}$. `x_rdy`$[p]$ is high in cycle $c$ when every condition
holds:
- the head runs ($c \ge t_s +$ `RUN_LAT`);
- no burst of the copy is in progress (its plane count, started by `x_v` rising, is back to 0);
- `gq_rdy`$[p]$, `gk_rdy`$[p]$ and `gv_rdy`$[p]$ are high (after the copy's previous burst: its GEMMs' `acam_done` + 1,
  the pass rule of `ip/xbar/spec.md`; until 2026-10-22 `msb_fire` + 1);
- token $t$ is left ($t < n$);
- QK A staging: $f_{qa}(c) + 2 > \lfloor t/NB_A \rfloor$; QK B staging: $f_{qb}(c) + 2 > \lfloor t/B_{kv} \rfloor$; PV B
  credit (edge E3, `PB_CAP` = 4 rows per copy): $f_{pb}(c) + 4 P_{proj} > t$. Here $f_x(c)$ counts the pulses of
  $x$`_cr` at cycles $\le c -$ `CR_LAT`.

A burst starts in a cycle with `x_rdy`$[p]$ high, and then lasts $LOADX$ = 8 planes (the environment's contract). For
$k \ge 1$, `x_rdy`$[p]$ is low right after burst $k - 1$, and rises at

$$
x_{rdy}(p, k) = \max\big(rdy_{gq}(k - 1),\ rdy_{gk}(k - 1),\ rdy_{gv}(k - 1),\ t_{qa}(\lfloor t/NB_A \rfloor - 2) + 1,\ t_{qb}(\lfloor t/B_{kv} \rfloor - 2) + 1,\ t_{pb}(t - 4P_{proj}) + 1,\ t_s + 1\big)
$$

with $t_x(i)$ the cycle of pulse $i$ (a term with $i < 0$ is dropped). For $k = 0$ it is high from $t_s + 1$.

**QK jobs.** Job $j = rB + b$ ($j < R_b B$) carries A slot $r$, B slot $b$, no release. It is offered from
$t_s +$ `RUN_LAT`, and is taken in the first cycle with `qj_rdy` high: $t_{qj}(j) = \max(t_s + 1,\ t_{go}^{qk}(j - 1))$.
`qj_rdy` is high after reset, and from the DIMM's $t_{go}$ of the previous job.

**PV jobs.** Job $j$ carries A slot $j \bmod 2$ (released), B slot $j \bmod B$:
$t_{pj}(j) = \max(t_s + 1,\ t_{go}^{pv}(j - 1))$.

**Softmax runs.** Run $r$ ($r < R_b$) starts in the first cycle with `sm_busy` low:
$t_{st}(r) = \max(t_s + 1,\ t_{busy\downarrow}(r - 1))$. The softmax's `busy` rises the next cycle, so `sm_start`
lasts one cycle.

**Done.** `done` at $run\_done(R_b - 1) +$ `DONE_LAT`; `busy` falls at `done` + `BUSY_LAT`. One start per reset.

## 6 The model

`att_ctrl`'s model (`top/model.py`, `AttCtrl`) has these processes:

| process | steps | needs | emits |
|---|---|---|---|
| `start` | 1 | the start window | $t_s$ |
| `x`$p$ | $T_X$ | burst $k$ of copy $p$; the GEMMs' ready pulses $k$; the staging pulses the next token needs | the credit of burst $k$ at $x_{rdy}(p, k + 1)$ (none after the last token); checks burst $k$ starts at $\ge x_{rdy}(p, k)$ |
| `qj`, `pj` | $R_b B$ each | the job port's credit $j - 1$ | job $j$'s window |
| `go` | $R_b$ | the softmax's credit $r - 1$ | run $r$'s start window |
| `done` | 1 | the accumulator's `run_done` pulse $R_b - 1$ | the start port's credit (`busy` falling) |

The head workload is the same netlist as `att_head` (§4), built with three engine features (`simulator.md` §4.5):
- **Fan-out:** E0 copy $p$'s source port feeds `gq`$p$, `gk`$p$, `gv`$p$ and `ctrl.x`$p$.
- **Channel placement:** `gq`$p$'s output goes to channel $p$ of `qk.a` (likewise for `qk.b` and `pv.b`).
- **Pulse ports and credit routes:** the GEMMs' `act` credits (`MSB_SA_Ready` rising) go to `ctrl.rq`$p$, `rk`$p$,
  `rv`$p$. The DIMMs' staging credits go to `ctrl.qa`, `qb`, `pb`. `ctrl.x`$p$'s credits (`x_rdy` rising) go to the
  E0 source. The accumulator's `run_done` becomes a pulse port, routed to `ctrl.rd`.

The other edges are one to one, with credits back to the producer.

## 7 Counts and resources

`att_ctrl` has counters only: per copy a token counter and a burst word counter, three staging counters, two job
counters, a run counter, a `run_done` counter. No crossbar, no memory. The head's crossbars are the modules' sum
(`architecture.md` §4.2):

$$
N_{xbar} = 3 P_{proj} V + 3 P_{proj} + N_E + N_{EP} + (N_{EXP} + 2) + 2 N_{OC}
$$

## 8 Gates

- **GATE 1** (the head's case writer): the model's $O_8$ rows == `attention_ref.head_stages` stage `O8`. The
  intermediate streams are checked too: $q_8$, $k_8$, $v_8$; the scores $s_8$; $\log P$; $o_{part}$.
- **Composition:** with a random step order, the Report is the same. With values off, the events and counts are the
  same.
- **Bring-up + gate H** (64×64, small heads; `top/gen_cases.py`): the 5 bring-up heads cover $V = 1$ and $V = 2$,
  $R_b > 1$, $B \ge 2$, $P_{proj} \in \{1, 2, 4\}$. Gate H adds what they do not: X later than `x_rdy`, slow X, a late
  start, one QK channel ($N_E = d$), $V = 3$, and $d = 64$ (BERT-Tiny's shape). Pass: the RTL's $O_8$ rows, every
  port event of every edge (§4), `done` and the total cycles equal the model's.
- **VTR smoke** (128×128): crossbar count == $N_{xbar}$, no logic optimized away; Fmax, CLB, memory, path #1.

## 9 The rows of the DSE

Every DSE row is one `att_head` build: shape ($n$, $d_m$, $d$), config ($R$, $C$) and mapping ($P_{proj}$, $N_E$),
with the fixed values of `experiment.md` §4 ($B_{kv} = 16$, $N_{EXP} = 2$, $N_{EP} = 2d$, $N_{OC}$ by the floor
rule, $X_S = 16$).

## 10 Refinements of `architecture.md` §4.7

1. **`x_rdy` waits for the copy's burst to end.** A `dpe` keeps `MSB_SA_Ready` high through its whole load burst. So
   `att_ctrl` counts a copy's burst words, starts a token at `x_v` rising, and holds `x_rdy` low until the burst's
   last word.
2. **`att_ctrl` reads `x_v`:** it must count the tokens a copy has taken.
3. **Jobs and runs are offered from the head start;** a job port's offer is taken in the first cycle with `j_rdy` high.
4. **One start per reset.** A second run per reset would also need the DIMMs' slots cleared.
5. **PV B credit: 4 rows per copy, with X in token order** (user, 2026-10-15; `architecture.md` §5.2). The credit is
   taken when the token enters the GEMM, so its loop includes the GEMM. With 2 rows it bound the X stream on 176 of
   the 432 DSE builds (up to 8.9 % of the cycles); with 4 it binds on none (the pre-bit-plane timing). The PV DIMM writes V
   rows in row order, so the copies' bursts must start in token order (§3); a starved copy would otherwise let the
   other copies' rows overrun their row registers. The model raises on a violation (the PV DIMM's B port check).
6. **The bit-plane pass** (user, 2026-10-21/22; `ip/xbar/spec.md`).
   - **X:** a token burst is 8 plane words of $V \cdot ROWS_X$ bits per copy (`x_d` $P_{proj} V \cdot ROWS_X$ bits;
     `ROWS_X` is a derived header parameter of `att_head`, since a port width cannot use a body localparam), so
     `LOADX` = $P$ for
     every array. `BUF` is removed.
   - **$O_8$:** E9 is one plane window of $N_{OC} I$ elements per epilogue wave (`o_d` $N_{OC} I$ bits).
   - **Admission:** the GEMMs' `MSB_SA_Ready` rises at `acam_done` + 1. A copy's row period is the pass period (10
     cycles) on every array.
   - **Held:** the mapping ($N_{OC}$ by the pre-bit-plane floor rule) and the credit sizes, including `PB_CAP` = 4. Its loop
     (the token's burst, the GEMM's pass and drain, the DIMM's row write and credit) is about 33 cycles against the
     4 rows × 10 cycles of the copy's row period.
