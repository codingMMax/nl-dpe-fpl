# `dimm_tile_top` — tiled DIMM (v3 module spec)

Status: 2026-10-14, F3 / phase A, **closed**: GATE 2 16/16 at 64×64, VTR smoke at 128×128 (`LEDGER.md`);
2026-10-15: the `TRANS_B` = 0 credit is 4 `B_CH` rows (§10 item 6, no RTL change).

**2026-10-22, the bit-plane pass** (§10 item 8):
- every port carries bit planes (`ip/xbar/spec.md`, `docs/architecture.md` §3.1);
- the farm is fed bit-serially;
- the output is summed per plane.

The behavior model (`simulator/models/dimmTile.py`) implements this version. Both RTL twins still have the 40-bit
version (F3) and follow in the RTL pass, so GATE 2 pins this version's latencies only after that.

This is the module's contract: the behavior model and both RTL twins (`ip/dimm/rtl/{sim,vtr}/dimm_tile_top.v`)
implement it, and GATE 2 pins every latency named here. Built from the verified v2 tile core (memories, window engine,
farm, registered trees; `v2/sim/rtl/dimm_tile_top.v`) and `docs/architecture.md` §4.4. Refinements of §4.4 made while
writing this spec are listed in §10.

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
| shape | `M, N, K, R, C, P, N_E` | job shape, crossbar, farm crossbars (`BUF` removed 2026-10-22) |
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
| $P$ | 8: plane words per window (every port) and bit slices of a pass |
| $X_S'$, $n_s$, $K_s$ | $\min(X_S, N_E)$ crossbars per slice; $N_E / X_S'$ slices; $X_S'/NG$ columns per slice |
| $A_{GRP}$ | the largest power of two $\le \min(P, X_S')$ (each A memory serves $A_{GRP}$ crossbars in a wave's $P$ planes) |
| $N_{AB}$ | $X_S'/A_{GRP}$: A memories per slice, each with its own write port |
| $B_{GRP}$ | $\min(A_{GRP}, K_s)$ |
| $L_B$ | B row length: $K$ (`TRANS_B` = 1) or $N$ |
| $E_w$ | words a window-frame window gives a slice: $(A_{ROWS}/NB)\,K_s$ |
| $E_{wp}$ | the most of a window's $E_w$ words that any one A memory takes (over the windows; §6 layout) |
| $VW$, $E_C$ | $9 + \log_2 K$; output element bits $E_C$ = 8 (`NARROW` > 0) or $VW$ ($\lceil E_C/8 \rceil$ lanes) |
| $T_{cv}$ | $P + 2$: the converter's pass period |

Guards (the RTL's elaboration guards; the model raises on the same sets):
- $P = 8$: one plane per bit of an int8 element (the stagers, the bit-serial adders and the root's $2^b$ and $255$
  are byte-wide; until 2026-10-22 the guard was `BUF` = 40).
- $M, N, K, N_E, X_S, I$ powers of two; $I \ge 8$; $N \le I$; $MN \ge I$; $2 \le K \le N_E$; $NG \le W_K$;
  $NG \le X_S'$; $K \le I$ and $L_B \le I$ (a converter holds one row).
- Row frame: `A_ROWS` = 1, `A_CH` $\le NB$. Window frame: `A_ROWS` a multiple of $NB$, $M$ a multiple of `A_ROWS`,
  $A_{ROWS} K \le I$ (an input window is at most one EXP window).
- `A_CH`, `B_CH`, `A_SLOTS`, `B_SLOTS`, `C_CAP` $\ge 1$; $0 \le$ `NARROW` $< VW$.

## 3 Ports

| port | width | dir | discipline | unit | `CAP` |
|---|---|---|---|---|---|
| `a_v`, `a_d` / `a_cr` | `A_CH`, `A_CH`·$A_W$ / 1 | in / out | push + credit | row frame: a group of $NB$ rows; window frame: a matrix | 2 groups; `A_SLOTS` |
| `b_v`, `b_d` / `b_cr` | `B_CH`, `B_CH`·$L_B$ / 1 | in / out | push + credit | `TRANS_B` = 1: a matrix ($N$ rows); 0: a row | 2 matrices; 4 `B_CH` rows |
| `j_v`, `j_a`, `j_b`, `j_ra` / `j_rdy` | 1, ⌈log₂ `A_SLOTS`⌉, ⌈log₂ `B_SLOTS`⌉, 1 / 1 | in / out | burst on ready | a job | 1 |
| `c_v`, `c_d` / `c_cr` | $NG$, $NG \cdot I \lceil E_C/8 \rceil$ / 1 | out / in | push + credit | one job's $C$ | `C_CAP` |
| `busy` | 1 | out | | from start to `done` | |
| `weight_in`, `weight_en` | 8, 1 | in | | simulation twin only | |

Framing (`architecture.md` §3.1, plane framing since 2026-10-22): window $w$ of a stream goes on channel
$w \bmod L$ as 8 plane words on consecutive cycles, LSB first; word $b$ carries bit $b$ of every element (a 13-bit
$C$ in 2 lanes: lane 1 carries bit $8 + b$); bits past the window are 0.
- A, row frame: window $t$ = row $t$ ($K$ elements, $A_W = K$); window frame: window $w$ = rows $wA_{ROWS} \ldots$,
  row-major ($A_W = A_{ROWS} K$ elements).
- B: window $t$ = row $t$ of $B$ (`TRANS_B` = 0, $N$ elements) or of $B^\top$ (`TRANS_B` = 1, $K$ elements).
- Matrix $i$ of a port = its rows $iM \ldots$ (A), $iK \ldots$ (B, `TRANS_B` = 0) or $iN \ldots$ (`TRANS_B` = 1); it
  fills slot $i \bmod X$`_SLOTS`.
- C: window $w = g + NG\,t$ of job $j$ (wave $t$, group $g$) on channel $g$: $I$ elements = rows $wNB \ldots$ of $C$,
  8 words. All $NG$ channels carry a wave's words in the same cycles.

Port contract for the producers:
- int8 ports: consecutive windows on one channel start at least one converter pass period apart, $T_{cv} = P + 2$
  (10; until 2026-10-22 $\max(\lceil 8L/40 \rceil + P,\ P + 2,\ \lceil 8C/40 \rceil + 1)$). The model raises on an
  overrun.
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
   A loader: row frame -> TRANSPOSER (G = NB rows)   |  window frame -> WINDOW STAGER (one write port per A memory)
   B loader: TRANS_B = 1 -> TRANSPOSER (G = N rows)  |  TRANS_B = 0 -> the shared ROW WRITER's bus
   A memories (N_AB = X_S'/A_GRP, blocked), B memories (K_s/B_GRP); window engine (leader = the slice's crossbar 0);
   per crossbar: A and B shift registers, I bit-serial adders -> dpe (EXP, identity, I x I) -> sign XOR;
   the first log2 K_s levels of each group's per-plane popcount tree
 shared: slot valid bits, release, credits; JOB register + decision; tree levels log2 n_s; per group the ROOT
 accumulator, NARROW and the OUTPUT PLANE BUFFER; c_v / c_d
```

**The bit-plane datapath** (2026-10-22):
- **Converters** take the int8 row's planes as they come: the planes go straight into the converter crossbar.
- **The distribution register** carries planes to every slice.
- **The stagers** keep byte registers and write each arriving plane as one bit of every byte. A unit is complete
  with its last plane. The memories keep the byte layout of §6.
- **Farm feed.** Each crossbar holds its A word ($NB$ bytes) and its column's B word ($N$ bytes) in shift registers
  that shift one bit per plane. Row $r$ of the crossbar's input ($m = r / N$, $n = r \bmod N$ in the window) takes
  $\mathrm{trunc8}(L_A[m][k] + L_B[k][n])$ from a bit-serial full adder: the carry is cleared at plane 0, and the
  carry out of plane 7 is dropped.
- **Drain.** For an unsigned EXP byte $e$ and the term's sign $s$, $x = e \oplus (s\ ?\ \mathtt{0xFF} : 0)$, so
  $\pm e = x - 255 s$. Therefore
  $$
  C = \sum_k \pm e_k = \sum_b 2^b\, \mathrm{POP}_b - 255\, n_{neg}, \qquad
  \mathrm{POP}_b = \sum_k \mathrm{bit}_b(x_k), \qquad n_{neg} = \sum_k s_k .
  $$
  - Per plane, the group's tree adds the counts $\mathrm{POP}_b$ (registered levels).
  - $n_{neg}$ goes through the same tree in the cycle before the first drain plane, and loads the root accumulator
    with $-255\, n_{neg}$ (a shift and a subtract).
  - The root adds $2^b\, \mathrm{POP}_b$ plane by plane, LSB first.
  - After plane 7 the value is exact. NARROW runs on it, and the output plane buffer sends its 8 planes (2 lanes for
    $E_C > 8$).

## 5 Timing rules

Cycle 0 is the first cycle after reset. A signal "at $t$" is high (or a register reads its value) in cycle $t$.

| name | value | rule |
|---|---|---|
| `CONV_IN_LAT` | 1 | a row's first plane at port cycle $t$ starts its converter burst at $t + 1$ (the input register); the converter crossbar must be ready then (contract) |
| — | the pass rule | the converter drains its $L$ log bytes as 8 planes: plane $b$ at `drain_start` $+ b$ (`ip/xbar/spec.md`) |
| `DIST_LAT` | 1 | a stream plane (the converter's drain, or the port word for a log operand) reaches every slice one cycle later |
| `EMIT_LAT` | 1 | a stager's unit is complete when its last plane reaches the slice at $t_c$; its first write is at $\ge t_c + 1$ |
| `CR_LAT` | 1 | a staging credit pulses one cycle after the unit's last write |
| `DEC_LAT` | 1 | the job decision is registered: decided in cycle $t_{dec}$, the slices start at $t_{go} = t_{dec} + 1$ |
| `START_LAT` | $A_{GRP} + 2$ | the slices start at $t_{go}$; every window of the job is ready at $t_{go}$ + `START_LAT` |
| `DRAIN_REG` | 1 | the registered sign XOR of a drain plane |
| `TREE_LEVELS` | $\log_2 K$ | the per-plane popcount tree: $\log_2 K_s$ levels in each slice, then $\log_2 n_s$ shared, all registered |
| `ROOT_LAT` | 2 | after the last plane reaches the root: the root accumulator, then the output plane buffer |
| `OUT_LAT` | `DRAIN_REG` + `TREE_LEVELS` + $(P - 1)$ + `ROOT_LAT` | a wave's `drain_start` → its first output plane |
| `DONE_LAT` | 2 | `done` two cycles after the job's last output word |
| `IDLE_LAT` | 2 | the slices read idle two cycles after the job's last drain plane |
| `REL_LAT` | 1 | a released slot reads invalid, and its credit pulses, one cycle after `done` |

**Stream arrival.** Window $w$ of a port at port cycle $t_p$ (first plane). Its stream time $t_s$ is the converter's
`drain_start` (int8) or $t_p$ (log); plane $b$ reaches the slices at $a_b = t_s + b +$ `DIST_LAT`. The window is
complete at the slices at $a_{last} = t_s + P - 1 +$ `DIST_LAT`. (Until 2026-10-22: $\lceil W/5 \rceil$ words.)

**Transposer** (row frame A: $G = NB$, emits $K_s$ words of $NB$ bytes into the A memories; `TRANS_B` = 1 B: $G = N$,
emits $K_s$ words of $N$ bytes into the B memories). Group $q$ = rows $qG \ldots qG + G - 1$, staged in buffer
$q \bmod 2$, bank = row mod $G$.
- Complete at $t_c(q)$ = the latest $a_{last}$ of its rows.
- Writes: $t_e(q) = \max(t_c(q) + 1,\ t_e(q-1) + K_s)$; word $i$ (local column $i$, $k = sK_s + i$) at $t_e(q) + i$.
- Credit (unit = the group): `a_cr` / `b_cr` at $t_e(q) + K_s$.
- A group is window $w = q \bmod W_K$ of A matrix $\lfloor q/W_K \rfloor$; for B, group $q$ is B matrix $q$.

**Window stager** (window frame A). Window $w$ is staged per channel in entry $\lfloor w/A_{CH} \rfloor \bmod 2$.
- Complete at $t_c(w) = a_{last}(w)$; writes start at $t_e(w) = \max(t_c(w) + 1,\ t_e(w-1) + E_{wp})$.
- Each A memory's port writes its words of the window (NB-row group $q$ outer, local column $i$ inner) one per
  cycle, all ports from $t_e(w)$. So the window's last write is at $t_e(w) + E_{wp} - 1$.
- Until 2026-10-22 one port per slice wrote all $E_w$ words. At one window per 10 cycles per channel, that would be
  $N_{EXP} E_w / 10$ = 1.6 words per cycle on the PV instance.
- Overrun (model check): window $w + 2A_{CH}$'s first plane must reach the slices after $t_e(w) + E_{wp} - 1$.
- No staging credit: the window-frame A port credits slots (below).

**Row writer** (`TRANS_B` = 0 B, shared by the slices). Per channel two row registers (row $t$ in register
$\lfloor t/B_{CH} \rfloor \bmod 2$).
- Row $t$ ready at $a_{last}(t) + 1$; written in row order, one per cycle: $t_w(t) = \max(a_{last}(t) + 1,\ t_w(t-1) + 1)$,
  into the B memory of slice $\lfloor k/K_s \rfloor$, $k = t \bmod K$.
- Credit (unit = the row): `b_cr` at $t_w(t) + 1$.
- Four credits per channel on two registers:
  - with rows in row order (§3), a row holds its register for at most $P + B_{CH} - 1$ cycles (its 8 planes, then the
    rows ahead of it at the writer);
  - a channel's rows reach the slices at least $T_{cv} = P + 2$ apart (§3);
  - so two registers per channel hold every row in flight while $P + B_{CH} - 1 \le 2(P + 2)$, at any credit count;
  - the model's register check raises otherwise.

**Slots.** A matrix is valid in cycle $t_v$ = its last write + 1 (every slice writes in lock-step). A window-frame A
slot released by a job is invalid from `done` + `REL_LAT`, when `a_cr` pulses (unit = the matrix).

**Jobs.** Job $j$ is taken at $t_{in}$ ($j\_v \wedge j\_rdy$), into a one-entry register; `j_rdy` = the register is
empty. The decision cycle is the first $t_{dec} \ge t_{in} + 1$ with:
- its A matrix and B matrix valid ($t_{dec} \ge t_v$ of each);
- every slice idle ($t_{dec} \ge t_{idle}$, initially 0);
- an output credit: $t_{dec} \ge t_{cr}(j - C_{CAP}) + 1$ for $j \ge$ `C_CAP` ($t_{cr}$ = that `c_cr` pulse).

Then $t_{go} = t_{dec} + 1$; the register frees, so `j_rdy` rises at $t_{go}$ (the job port's credit). The job's A
matrix is the one in slot `j_a`: for a released slot, the oldest not yet released.

**Core** (per slice, lock-step; the v2 tile from its start edge). Wave $t$ is one `CrossbarGroup` pass of the farm
(EXP, identity, $I \times I$) ready at $t_{go}$ + `START_LAT`, placed by the pass rule on the farm's own state:
back to back, a wave every 10 cycles. Crossbar $f = g + NG\,k$ (column $k$, group $g$) feeds window $(g + NG\,t, k)$:
$u = \mathrm{trunc8}(L_A[m][k] + L_B[k][n])$ over the window's $I$ elements, one bit-serial plane per cycle.
- Output plane $b$ of wave $t$, every channel: at `drain_start`$(t)$ + `OUT_LAT` $+ b$. (Until 2026-10-22, output word
  $c$ at `drain_start` $+ c +$ `DRAIN_REG` + `TREE_LEVELS`: the 5-lane sums streamed as they drained.)
- The root and the output plane buffer serve one wave at a time. A wave's planes reach the root 10 cycles after the
  previous wave's, and leave 8 cycles later, so one buffer suffices.
- `done` = the last output plane + `DONE_LAT`; $t_{idle}$ = the last wave's `drain_end` + `IDLE_LAT`.

## 6 Memories

| memory | per slice | words × bytes |
|---|---|---|
| A | $N_{AB} = X_S'/A_{GRP}$, blocked: crossbar $f$ in memory $\lfloor f/A_{GRP} \rfloor$, index $f \bmod A_{GRP}$; each its own write port | `A_SLOTS` $\cdot T_W \cdot A_{GRP}$ × $NB$, address {slot, wave, index} |
| B | $K_s/B_{GRP}$: local column $i$ in memory $\lfloor i/B_{GRP} \rfloor$ | `B_SLOTS` $\cdot B_{GRP}$ × $N$, address {slot, $i \bmod B_{GRP}$} |
| transposer staging | 2 × $G$ × $K_s$ bytes, flip-flops (written one bit per plane) | |
| window staging | `A_CH` × 2 × `A_ROWS` × $K_s$ bytes, flip-flops (written one bit per plane) | |
| row writer | `B_CH` × 2 × $N$ bytes (shared, written one bit per plane) | |
| farm feed | per crossbar: the A word ($NB$ bytes) and the column's B word ($N$ bytes) as shift registers, $I$ carry flip-flops | |
| root and output | per group: $I$ accumulators of $VW + 1$ bits, the output plane buffer $I \times E_C$ bits | |

With $A_{GRP}$ capped at $P$ = 8 (2026-10-22; until then $\min(\lceil 8I/40 \rceil, X_S')$ = 8 / 16 / 16 at $I$ = 64 /
128 / 256), a 16-crossbar slice has 2 A memories at every $I$, twice the count at $I \ge 128$.

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
- Closed form (the model's test, written by hand from §5): for back-to-back jobs, wave $t$ starts at
  $t_{go}$ + `START_LAT` + $10t$; its output planes at `drain_start` + `OUT_LAT`; `done` and $t_{idle}$ as §5.
  (Until 2026-10-22 the witness was `dimm_tile_sim.tile_cycle_model`, which times the 40-bit pass.)
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
8. **The bit-plane pass** (user, 2026-10-21/22; `ip/xbar/spec.md`).
   - **Ports:** every port carries 8 plane words per window (`a_d` $A_W$ bits, `b_d` $L_B$ bits, `c_d`
     $I \lceil E_C/8 \rceil$ bits per channel); `BUF` is removed.
   - **Converters:** they take the int8 planes directly; their contract is $T_{cv}$ = 10.
   - **Stagers:** they write one bit per plane into their byte registers. A unit is complete with its last plane
     (`DIST_LAT` after the stream's plane 7).
   - **$A_{GRP}$:** the cap is $P$ = 8 instead of $\lceil 8I/40 \rceil$, because a wave's feed now lasts $P$ planes.
     `START_LAT` = 10, and a slice has 2 A memories at every $I$.
   - **Window stager:** it writes through each A memory's own port ($N_{AB}$ = 2), so the PV A port keeps up at one
     window per 10 cycles per channel.
   - **Farm feed:** bit-serial $\mathrm{trunc8}(L_A + L_B)$ from shift registers. The 5-lane geometry muxes and the
     lane rotation go.
   - **Drain:** per-plane popcount trees, the root accumulator with the $-255\, n_{neg}$ preload, NARROW on the
     exact value, and the output plane buffer. `OUT_LAT` = $10 + \log_2 K$ (was $1 + \log_2 K$).
   - **Guard:** the window frame's $A_{ROWS} K \le 5 \cdot 2^{16}$ becomes $A_{ROWS} K \le I$.
   - **Values, counts and the job and slot rules are unchanged.**
