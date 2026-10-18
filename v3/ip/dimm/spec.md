# `dimm_tile_top` — tiled DIMM (v3 module spec)

Status: 2026-10-14, F3 / phase A, **closed**: GATE 2 16/16 at 64×64, VTR smoke at 128×128 (`LEDGER.md`);
2026-10-15: the `TRANS_B` = 0 credit is 4 `B_CH` rows (§10 item 6, no RTL change). This is the
module's contract: the behavior model `ip/dimm/model.py` and both RTL
twins (`ip/dimm/rtl/{sim,vtr}/dimm_tile_top.v`) implement it, and GATE 2 pins every latency named here. Built from the
verified v2 tile core (memories, window engine, farm, registered trees; `v2/sim/rtl/dimm_tile_top.v`) and
`docs/architecture.md` §4.4. Refinements of §4.4 made while writing this spec are listed in §10.

## 1 Function

One job computes one product in the log domain:

$$
C_{mn} = \sum_{k} (-1)^{s_A(m,k) \oplus s_B(k,n)}\ \mathrm{EXP}\big(\mathrm{trunc8}(L_A[m][k] + L_B[k][n])\big),
\qquad C = \mathrm{NARROW}\ ?\ \mathrm{sat8}(C \gg \mathrm{NARROW}) : C
$$

- $A$ is $M \times K$, $B$ is $K \times N$. With `TRANS_B` = 1 the B port carries the rows of $B^\top$.
- An int8 operand ($X$`_DOM` = 0) is converted on the port: $L = \mathrm{LOG}(x) = \mathrm{trunc8}(x - 1)$. A log
  operand ($X$`_DOM` = 1) is used as given.
- Sign $s(L) = L[7] \oplus \&L[6{:}0]$, or 0 when $X$`_SIGNED` = 0.
- Value contract (GATE 1): `attention_ref.signed_tile` on the converted operands, then `attention_ref.narrow`.

## 2 Parameters

| group | parameter | meaning |
|---|---|---|
| shape | `M, N, K, R, C, BUF, P, N_E` | job shape, crossbar, farm crossbars |
| slices | `X_S` | crossbars per slice, 16 |
| A port | `A_DOM`, `A_SIGNED`, `A_CH`, `A_SLOTS` | int8 / log; sign; input channels; matrices held |
| | `A_FRAME`, `A_ROWS` | 0 row frame (one row of $K$ per window; `A_ROWS` = 1), 1 window frame (`A_ROWS` rows per window) |
| B port | `B_DOM`, `B_SIGNED`, `B_CH`, `B_SLOTS`, `TRANS_B` | as A; one row per window: $K$ elements (`TRANS_B` = 1) or $N$ |
| output | `NARROW`, `C_CAP` | 0 full width, $s > 0$: $\mathrm{sat8}(C \gg s)$; output credits (jobs) |

Derived (no free numbers):

| name | value |
|---|---|
| $I$ | $\min(R, C)$ |
| $NB$ | $I / N$: rows per farm window |
| $W_K$ | $MN / I$: windows per column $k$ (per job) |
| $NG$ | $N_E / K$: groups, i.e. complete windows per wave |
| $T_W$ | $W_K / NG$: waves |
| $LW$ | $\lceil 8I/BUF \rceil$: words per window (feed and drain) |
| $X_S'$, $n_s$, $K_s$ | $\min(X_S, N_E)$ crossbars per slice; $N_E / X_S'$ slices; $X_S'/NG$ columns per slice |
| $A_{GRP}$ | the largest power of two $\le \min(LW, X_S')$ |
| $B_{GRP}$ | $\min(A_{GRP}, K_s)$ |
| $L_B$ | B row length: $K$ (`TRANS_B` = 1) or $N$ |
| $E_w$ | words a window-frame window gives a slice: $(A_{ROWS}/NB)\,K_s$ |
| $VW$, $E_C$ | $9 + \log_2 K$; output element bits $E_C$ = 8 (`NARROW` > 0) or $VW$ |

Guards (the RTL's elaboration guards; the model raises on the same sets):
- $M, N, K, N_E, X_S, I$ powers of two; $I \ge 8$; $N \le I$; $MN \ge I$; $2 \le K \le N_E$; $NG \le W_K$;
  $NG \le X_S'$; $K \le I$ and $L_B \le I$ (a converter holds one row); `BUF` = 40.
- Row frame: `A_ROWS` = 1, `A_CH` $\le NB$. Window frame: `A_ROWS` a multiple of $NB$, $M$ a multiple of `A_ROWS`.
- `A_CH`, `B_CH`, `A_SLOTS`, `B_SLOTS`, `C_CAP` $\ge 1$; $0 \le$ `NARROW` $< VW$.

## 3 Ports

| port | width | dir | discipline | unit | `CAP` |
|---|---|---|---|---|---|
| `a_v`, `a_d` / `a_cr` | `A_CH`, `A_CH`·40 / 1 | in / out | push + credit | row frame: a group of $NB$ rows; window frame: a matrix | 2 groups; `A_SLOTS` |
| `b_v`, `b_d` / `b_cr` | `B_CH`, `B_CH`·40 / 1 | in / out | push + credit | `TRANS_B` = 1: a matrix ($N$ rows); 0: a row | 2 matrices; 4 `B_CH` rows |
| `j_v`, `j_a`, `j_b`, `j_ra` / `j_rdy` | 1, ⌈log₂ `A_SLOTS`⌉, ⌈log₂ `B_SLOTS`⌉, 1 / 1 | in / out | burst on ready | a job | 1 |
| `c_v`, `c_d` / `c_cr` | $NG$, $NG \cdot 5E_C$ / 1 | out / in | push + credit | one job's $C$ | `C_CAP` |
| `busy` | 1 | out | | from start to `done` | |
| `weight_in`, `weight_en` | 8, 1 | in | | simulation twin only | |

Framing (`architecture.md` §3.1): window $w$ of a stream goes on channel $w \bmod L$, its words on consecutive
cycles, word $c$ = elements $5c \ldots 5c+4$, slots past the window are 0.
- A, row frame: window $t$ = row $t$ ($K$ elements); window frame: window $w$ = rows $wA_{ROWS} \ldots$, row-major.
- B: window $t$ = row $t$ of $B$ (`TRANS_B` = 0, $N$ elements) or of $B^\top$ (`TRANS_B` = 1, $K$ elements).
- Matrix $i$ of a port = its rows $iM \ldots$ (A), $iK \ldots$ (B, `TRANS_B` = 0) or $iN \ldots$ (`TRANS_B` = 1); it
  fills slot $i \bmod X$`_SLOTS`.
- C: window $w = g + NG\,t$ of job $j$ (wave $t$, group $g$) on channel $g$: $I$ elements = rows $wNB \ldots$ of $C$,
  $LW$ words. All $NG$ channels carry a wave's words in the same cycles.

Port contract for the producers:
- int8 ports: consecutive windows on one channel start at least one converter pass period apart,
  $T_{cv} = \max(\lceil 8L/BUF \rceil + P, P + 2, \lceil 8L/BUF \rceil + 1)$ ($L$ = the row length); the model raises
  on an overrun.
- Window frame: the slice writer must keep up (`architecture.md` §4.4 rate guard); the model raises on an overrun.
- Row-frame ports and the B port never wrap their slots: a port receives at most `X_SLOTS` matrices per reset.
- `TRANS_B` = 0: rows reach the port in row order (row $t$'s first word no earlier than row $t - 1$'s; rows may share
  a cycle). The row writer writes in row order, so a late row would hold the later rows in their registers. The
  model raises on a violation.

## 4 Structure

```text
 a_v/a_d (A_CH) -> [CONV per channel, A_DOM = 0] -> [DIST reg] --+
 b_v/b_d (B_CH) -> [CONV per channel, B_DOM = 0] -> [DIST reg] --+--> every slice (each takes its columns)
                                                                 |
 slice s (s = 0 .. n_s-1; K_s columns x NG groups = X_S' crossbars):
   A loader: row frame -> TRANSPOSER (G = NB rows)   |  window frame -> WINDOW STAGER
   B loader: TRANS_B = 1 -> TRANSPOSER (G = N rows)  |  TRANS_B = 0 -> the shared ROW WRITER's bus
   A memories (X_S'/A_GRP, blocked), B memories (K_s/B_GRP); window engine (leader = the slice's crossbar 0);
   crossbars (EXP, identity, I x I); the first log2 K_s levels of each group's tree
 shared: slot valid bits, release, credits; JOB register + decision; tree levels log2 n_s; NARROW; c_v / c_d
```

## 5 Timing rules

Cycle 0 is the first cycle after reset. A signal "at $t$" is high (or a register reads its value) in cycle $t$.

| name | value | rule |
|---|---|---|
| `CONV_IN_LAT` | 1 | a row's first word at port cycle $t$ starts its converter burst at $t + 1$ (the input register); the converter crossbar must be ready then (contract) |
| — | pass grammar | the converter drains $L$ log bytes: word $c$ at `drain_start` $+ c$ |
| `DIST_LAT` | 1 | a stream word (the converter's drain, or the port word for a log operand) reaches every slice one cycle later |
| `EMIT_LAT` | 1 | a stager's unit is complete when its last word reaches the slice at $t_c$; its first write is at $\ge t_c + 1$ |
| `CR_LAT` | 1 | a staging credit pulses one cycle after the unit's last write |
| `DEC_LAT` | 1 | the job decision is registered: decided in cycle $t_{dec}$, the slices start at $t_{go} = t_{dec} + 1$ |
| `START_LAT` | $A_{GRP} + 2$ | the slices start at $t_{go}$; every window of the job is ready at $t_{go}$ + `START_LAT` |
| `DRAIN_REG` | 1 | the registered signed drain value |
| `TREE_LEVELS` | $\log_2 K$ | $\log_2 K_s$ in each slice, then $\log_2 n_s$ shared, all registered |
| `DONE_LAT` | 2 | `done` two cycles after the job's last output word |
| `IDLE_LAT` | 2 | the slices read idle two cycles after the job's last drain word |
| `REL_LAT` | 1 | a released slot reads invalid, and its credit pulses, one cycle after `done` |

**Stream arrival.** Window $w$ of a port at port cycle $t_p$ (first word). Its stream time $t_s$ is the converter's
`drain_start` (int8) or $t_p$ (log); word $c$ reaches the slices at $a_c = t_s + c +$ `DIST_LAT`. The window is complete
at the slices at $a_{last} = t_s + \lceil W/5 \rceil - 1 +$ `DIST_LAT`.

**Transposer** (row frame A: $G = NB$, emits $K_s$ words of $NB$ bytes into the A memories; `TRANS_B` = 1 B: $G = N$,
emits $K_s$ words of $N$ bytes into the B memories). Group $q$ = rows $qG \ldots qG + G - 1$, staged in buffer
$q \bmod 2$, bank = row mod $G$.
- Complete at $t_c(q)$ = the latest $a_{last}$ of its rows.
- Writes: $t_e(q) = \max(t_c(q) + 1,\ t_e(q-1) + K_s)$; word $i$ (local column $i$, $k = sK_s + i$) at $t_e(q) + i$.
- Credit (unit = the group): `a_cr` / `b_cr` at $t_e(q) + K_s$.
- A group is window $w = q \bmod W_K$ of A matrix $\lfloor q/W_K \rfloor$; for B, group $q$ is B matrix $q$.

**Window stager** (window frame A). Window $w$ is staged per channel in entry $\lfloor w/A_{CH} \rfloor \bmod 2$.
- Complete at $t_c(w) = a_{last}(w)$; writes $t_e(w) = \max(t_c(w) + 1,\ t_e(w-1) + E_w)$, $E_w$ words (NB-row group
  $q$ outer, local column $i$ inner) at $t_e(w) \ldots t_e(w) + E_w - 1$.
- Overrun (model check): window $w + 2A_{CH}$'s first word must reach the slices after $t_e(w) + E_w - 1$.
- No staging credit: the window-frame A port credits slots (below).

**Row writer** (`TRANS_B` = 0 B, shared by the slices). Per channel two row registers (row $t$ in register
$\lfloor t/B_{CH} \rfloor \bmod 2$).
- Row $t$ ready at $a_{last}(t) + 1$; written in row order, one per cycle: $t_w(t) = \max(a_{last}(t) + 1,\ t_w(t-1) + 1)$,
  into the B memory of slice $\lfloor k/K_s \rfloor$, $k = t \bmod K$.
- Credit (unit = the row): `b_cr` at $t_w(t) + 1$.
- Four credits per channel on two registers: with rows in row order (§3), a row holds its register for at most
  $\lceil L_B/5 \rceil + B_{CH} - 1$ cycles, and a channel's rows reach the slices at least $T_{cv}$ apart (§3), with
  $T_{cv} \ge \lceil L_B/5 \rceil + P$. So at most one register per channel is full at a time, at any credit count;
  the model's register check raises otherwise.

**Slots.** A matrix is valid in cycle $t_v$ = its last write + 1 (every slice writes in lock-step). A window-frame A
slot released by a job is invalid from `done` + `REL_LAT`, when `a_cr` pulses (unit = the matrix).

**Jobs.** Job $j$ is taken at $t_{in}$ ($j\_v \wedge j\_rdy$), into a one-entry register; `j_rdy` = the register is
empty. The decision cycle is the first $t_{dec} \ge t_{in} + 1$ with:
- its A matrix and B matrix valid ($t_{dec} \ge t_v$ of each);
- every slice idle ($t_{dec} \ge t_{idle}$, initially 0);
- an output credit: $t_{dec} \ge t_{cr}(j - C_{CAP}) + 1$ for $j \ge$ `C_CAP` ($t_{cr}$ = that `c_cr` pulse).

Then $t_{go} = t_{dec} + 1$; the register frees, so `j_rdy` rises at $t_{go}$ (the job port's credit). The job's A
matrix is the one in slot `j_a`: for a released slot, the oldest not yet released.

**Core** (per slice, lock-step; the v2 tile from its start edge). Wave $t$ is one `XbarGroup` pass of the farm (EXP,
identity, $I \times I$) ready at $t_{go}$ + `START_LAT`, placed by the pass grammar on the farm's own state. Crossbar
$f = g + NG\,k$ (column $k$, group $g$) feeds window $(g + NG\,t, k)$: $u = \mathrm{trunc8}(L_A[m][k] + L_B[k][n])$ over
the window's $I$ elements.
- Output word $c$ of wave $t$, every channel: at `drain_start`$(t) + c +$ `DRAIN_REG` + `TREE_LEVELS`.
- `done` = the last output word + `DONE_LAT`; $t_{idle}$ = the last wave's `drain_end` + `IDLE_LAT`.

## 6 Memories

| memory | per slice | words × bytes |
|---|---|---|
| A | $X_S'/A_{GRP}$, blocked: crossbar $f$ in memory $\lfloor f/A_{GRP} \rfloor$, index $f \bmod A_{GRP}$ | `A_SLOTS` $\cdot T_W \cdot A_{GRP}$ × $NB$, address {slot, wave, index} |
| B | $K_s/B_{GRP}$: local column $i$ in memory $\lfloor i/B_{GRP} \rfloor$ | `B_SLOTS` $\cdot B_{GRP}$ × $N$, address {slot, $i \bmod B_{GRP}$} |
| transposer staging | 2 × $G$ × $K_s$ bytes, flip-flops | |
| window staging | `A_CH` × 2 × `A_ROWS` × $K_s$ bytes, flip-flops | |
| row writer | `B_CH` × 2 × $N$ bytes (shared) | |

A word (slot, $k$, $w$) = rows $wNB \ldots wNB + NB - 1$ of column $k$, for crossbar $f = (w \bmod NG) + NG\,(k - sK_s)$,
wave $\lfloor w/NG \rfloor$. B word (slot, $k$) = row $k$ of $B$.

## 7 Counts and resources

- Passes: LOG ($L$ rows × $L$ columns) per converted row; EXP ($I \times I$) $N_E T_W$ per job.
- CLB adds: $K I$ per output window (the pairing adds $L_A + L_B$, one per product) $+ (K - 1) I$ (the trees).
  Compares: $I$ per output window when `NARROW` > 0 (the narrowing). Memory: the bytes written into A / B, and read
  by the farm. (2026-10-15: the pairing adds and the narrowing compares were added, as in v2's attention ledger.)
- Crossbars: $N_E$ EXP + `A_CH` LOG (`A_DOM` = 0) + `B_CH` LOG (`B_DOM` = 0).

## 8 Gates

- GATE 1 (case writer): the model's $C$ == the oracle's, bit-exact, every job.
- Closed form: for back-to-back jobs, every output word relative to $t_{go}$ and `done` equal the v2 tile's
  (`dimm_tile_sim.tile_cycle_model`, with $A_{GRP}$ by the slice rule).
- GATE 2 (64×64): the RTL's values, every port event (windows, credits, job takes), the total cycles and every
  slice's memory image == the model's; every slice's engine state equal in every cycle (monitor).
- VTR smoke (128×128): crossbar count == $N_E$ + converters; Fmax, CLB, memory.

## 9 The two instances of the head

| | QK | PV |
|---|---|---|
| $M \times N \times K$ | $S \times B_{kv} \times d$ | $S \times d \times B_{kv}$ |
| A | int8 $q_8$, row frame, `A_CH` = $P_{proj}$, `A_SLOTS` = $R_b$ | log $\log P$, sign +, window frame (`A_ROWS` = $NWEL/B_{kv}$), `A_CH` = $N_{EXP}$, `A_SLOTS` = 2, released |
| B | int8 $k_8$, `TRANS_B` = 1, `B_CH` = $P_{proj}$, `B_SLOTS` = $B$ | int8 $v_8$, `TRANS_B` = 0, `B_CH` = $P_{proj}$, `B_SLOTS` = $B$ |
| C | `NARROW` = 5 | `NARROW` = 0 |

## 10 Refinements of `architecture.md` §4.4

1. **The job port** is burst on ready into a one-entry job register: `j_rdy` = the register is empty, so it does not
   depend on the offered job. The decision (slots valid, slices idle, output credit) is made on the register.
2. **No `j_rb`:** B slots are never released (they hold every B matrix of the head). Only a window-frame A slot is
   released, and only that port credits slots.
3. **The `TRANS_B` = 0 loader is shared:** a V row is one B word for one slice, at most `B_CH` rows per row period, so
   one writer for all slices keeps up. Credits are per row, in row order.
4. **$MN \ge I$** is required (no partial farm windows; the head always has $MN \ge I$).
5. **Staging credits** for the row-frame A port (per $NB$-row group) and the `TRANS_B` = 1 B port (per matrix) are one
   pulse per freed unit. Producers on several channels share them.
6. **The `TRANS_B` = 0 port's credit is 4 `B_CH` rows** (user, 2026-10-15; was 2): its producer in the head pays the
   credit before a GEMM, so the loop is longer than the port's own (`architecture.md` §5.2). It comes with the
   row-order contract (§3; user's choice over 4 registers per channel). No RTL change: the producer counts the
   credits, and with rows in order the registers hold at most one row per channel (§5). The GATE 2 corpus
   regenerates byte-identically.
7. **Read addresses from slice registers only** (Fmax, 2026-10-15; RTL only, no timing rule changes). The head's VTR
   smoke ran at 116.5 MHz: the job-start register crossed the chip to every slice and selected the first memory read
   address in the start cycle. Now each slice loads its A and B slots from the job register while it is idle (the
   decision needs every slice idle, and the job register holds the job through the start cycle), and its wave count
   and fill index are zero while idle. Every read address in a job's cycles is unchanged.
