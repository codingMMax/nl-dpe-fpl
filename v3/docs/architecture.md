# v3 architecture: one attention head on NL-DPE

Status: 2026-10-14, draft for review. User decisions of 2026-10-14:
- every pass uses only the rows and columns in use, at most $I = \min(R, C)$ for a conversion (§1); since 2026-10-18
  the extent sets the load and the valid window only: the simulator drains every column of the array on every pass
  (the RTL still stops after the columns in use);
- the tiled DIMM's farm is cut into slices of $X_S = 16$ crossbars; $X_S$ is fixed, and the DSE searches the farm
  size $N_E$ analytically.
Nothing here has been simulated or synthesized. Tags:
- **[derived]**: follows from the formulas here (computed with a small script, no run);
- **[estimate]**: not backed by any run;
- **[open]**: waits for your decision (§10).

The experiment that uses this head is `experiment.md`.

## 1 Overview

**What is computed.** Textbook attention for one head, on NL-DPE crossbars, in the integer log domain:

$$
O_8 = \mathrm{softmax}\big(Q K^\top \cdot 2^{-5}\big)\, V, \qquad Q, K, V = \mathrm{trunc8}(X W_Q),\ \mathrm{trunc8}(X W_K),\ \mathrm{trunc8}(X W_V)
$$

- Inputs: $X \in \mathbb{Z}_8^{n \times d_m}$; $W_Q, W_K, W_V \in \mathbb{Z}_8^{d_m \times d}$.
- Output: $O_8 \in \mathbb{Z}_8^{n \times d}$.
- There is no output projection.

The softmax is blocked: runs of $S$ query rows, key blocks of $B_{kv}$ keys, online over the blocks. The value
contract is `v2/sim/oracle/attention_ref.py::head_stages`, stage `O8`, bit-exact.

**Building blocks.** Three general modules, plus a small controller:

| module | role |
|---|---|
| GEMM (`gemm_top`) | $Y = \mathrm{trunc8}(X W)$, used for the projections |
| tiled DIMM (`dimm_tile_top`) | a matrix product in the log domain, used for $QK^\top$ and $PV$ |
| online softmax (`softmax_online_top` + `softmax_online_acc`) | scores → $\log P$, $\alpha$, $l$; then the output accumulator → $O_8$ |
| tiling controller (`att_ctrl`) | the only logic specific to attention: which tile jobs run, in which order, and when tokens and runs start |

**Target.** VTR (Parmys + VPR) on the NL-DPE arch XMLs. The `dpe` hard block is the crossbar. Every module has a
simulation twin (behavior `dpe`) and a VTR twin (bare `dpe`).

**The crossbar pass** (`v2/sim/spec/dpe_nldpe.md`). A pass has three phases:
- LOAD: $\lceil rows/5 \rceil$ words, 5 bytes per cycle;
- COMPUTE: $P + 2 = 10$ cycles;
- OUTPUT: $\lceil cols/5 \rceil$ words, one per cycle.

Its timing:
- $T_{fill} = LOAD + 10 + OUTPUT$ and $T_{steady} = \max(LOAD + P,\ 10,\ OUTPUT + 1)$;
- $T(p) = T_{fill} + (p - 1)T_{steady}$ for $p$ back-to-back passes.

A load burst starts only while `MSB_SA_Ready` is high, and once started neither the load nor the drain stops.

**Extent rule** (P29, user 2026-10-14): every pass loads only the rows in use and drains only the columns in use, which
saves load and drain cycles. A conversion pass never uses more than $I = \min(R, C)$. **Amended 2026-10-18 (user):**
the drain walks every column of the array, in the energy (the output buffer and the drain are charged for $C$
columns) and in the simulator's timing (`OUTPUT_CYC` = $\lceil 8C/BUF \rceil$); the columns in use are the valid
prefix of the drained window. The RTL keeps the early-stopping drain, so the simulator is not RTL-pinned for passes
with fewer columns in use than $C$ (`LEDGER.md`, Needs you 24).

| crossbars | extent (rows × columns) |
|---|---|
| DIMM farms, online softmax EXP, accumulator groups | $I \times I$ |
| online softmax carry-EXP and carry-LOG (one value per query row) | $S \times S$ |
| DIMM LOG converters (one row of $d$ per pass) | $d \times d$ |
| GEMM tiles | $\min(d_m, R) \times d$ |

**Requirements.**

| item | value |
|---|---|
| configs | $R \in \{128, 256, 512, 1024\}$ × $C \in \{64, 128, 256\}$; $I = \min(R, C) \in \{64, 128, 256\}$ |
| shapes | six (`experiment.md` §3.2): $n \in \{32, 128, 256\}$, $d \in \{16, 32, 64\}$, $d_m \in \{64, 128, 192, 768\}$ |
| parameters | shape, config and mapping are fixed at build time; one VTR circuit per row |
| mapping | fixed for every shape, no search (user, 2026-10-15; `experiment.md` §4.2): $P_{proj} = 4$, $N_E = d$, $N_{EP} = 2d$, $N_{EXP} = 2$, $N_{OC}$ by the floor rule (§5.2), $B_{kv} = 16$, DIMM slices of $X_S = 16$ crossbars |
| platform | `BUF` = 40 (5 bytes per cycle per crossbar port), `P` = 8 |
| RTL verification | on 64×64 crossbars, with small corpora; a full iverilog launch needs your approval |
| VTR | VTR twins only; no DSPs (no runtime products) |

**Design principle.** Each module's ports take a matrix the way the math defines it: row-major rows. Each module
converts its inputs into its own internal layout. Every crossbar output is a push stream, with credits back from the
consumer's buffer. No module knows about runs or key blocks. That knowledge lives only in the tiling controller.

The alternative would be modules that expose their internal memories, with a head-specific adapter that computes
their address maps and holds every credit. That alternative spreads the transposes and the flow control over logic
that no module's own tests cover. Here every transpose, conversion and buffer sits inside the module whose data it
serves, and is verified with that module.

## 2 The computation

### 2.1 Symbols

| symbol | RTL name | meaning |
|---|---|---|
| $n$, $d_m$, $d$ | `NSEQ`, `DM`, `D` | tokens (query rows = keys), model width, head width |
| $R$, $C$, $I$ | `R`, `C`, `I` | crossbar rows, columns, $I = \min(R, C)$ |
| $B_{kv}$ | `BKV` | keys per block |
| $S$ | `S` | query rows per run, $\min(n, I)$ |
| $R_b$, $B$ | `RB`, `NBLK` | runs $n/S$; key blocks $n/B_{kv}$ |
| $V$ | `VP` | GEMM row tiles, $\lceil d_m/R \rceil$ |
| $LN = SO$ | `LN`, `SO` | $\log_2 n$ (the mean shift of $l$ and of $o$) |
| $P_{proj}$ | `P_PROJ` | projection copies; token $t$ goes to copy $t \bmod P_{proj}$ |
| $N_E$, $N_{EP}$, $N_{EXP}$, $N_{OC}$ | same | QK farm, PV farm, softmax EXP crossbars, accumulator group size |
| $LW$ | `LW` | words per $I$-element window, $\lceil I/5 \rceil$ |

ACAM forms:
- $\mathrm{EXP}(y) = \mathrm{trunc8}(1 + y + \lfloor y^2/2 \rfloor)$, read as an unsigned byte;
- $\mathrm{LOG}(x) = \mathrm{trunc8}(x - 1)$;
- $\mathrm{trunc8}$ is the low byte; $\mathrm{sat8}$ and $\mathrm{clamp8}$ clamp to $[-128, 127]$.

The sign of an operand is recovered from its log byte: $s(L) = L[7] \oplus \&L[6{:}0]$.

### 2.2 Stages

Per run $r$ (query rows $rS \ldots rS + S - 1$) and key block $b$ (keys $bB_{kv} \ldots bB_{kv} + B_{kv} - 1$). Index
names: $m$ = query row, $j$ = key, $k$ = head feature, $c$ = column of $V$.

| # | stage | module | crossbars (mode) | consumes | produces | integer form | meaning in float |
|---|---|---|---|---|---|---|---|
| F0 | projection | GEMM ×3 (× $P_{proj}$) | $V$ tiles each (REGULAR) | $X[t]$: $d_m$ int8 | $q_8[t], k_8[t], v_8[t]$: $d$ int8 each | $x_8 = \mathrm{trunc8}(X[t]\, W)$ | $x W$ |
| F1 | log conversion | DIMM operand ports | 1 per input channel (LOG, identity, $d \times d$) | $q_8, k_8, v_8$ rows | $L_Q, L_K, L_V$: log bytes | $L = \mathrm{LOG}(x_8)$, sign $s(L)$ | $\log\lvert x \rvert$, sign |
| F2 | scores | QK DIMM | $N_E$ (EXP, identity, $I \times I$) | $L_Q$ of run $r$ ($S \times d$), $L_K$ of block $b$ ($B_{kv} \times d$) | $acc$: $S \times B_{kv}$, $9 + \log_2 d$ bits | $acc_{mj} = \sum_k (-1)^{s(L_{Q,mk}) \oplus s(L_{K,jk})}\, \mathrm{EXP}(\mathrm{trunc8}(L_{Q,mk} + L_{K,jk}))$ | $q_m \cdot k_j$ |
| F3 | narrowing | QK DIMM output | — | $acc$ | $s_8$: $S \times B_{kv}$ int8 | $s_8 = \mathrm{sat8}(acc \gg 5)$ | $q \cdot k / 32$, in place of $1/\sqrt{d}$ |
| F4 | block softmax | online softmax | $N_{EXP}$ EXP + carry-EXP + carry-LOG (identity) | $s_8$; $(m, \log l)$ of block $b - 1$ | $\log P$ ($S \times B_{kv}$), $\log\alpha$ ($S$), $\log l$ ($S$) | $m_{blk} = \max_j s_8$; $m_{new} = m_{blk}$ ($b = 0$), else $\max(m, m_{blk})$; $\log\alpha = m - m_{new}$; $\log P = \max(s_8 - m_{new}, -128)$; $sp = \sum_j \mathrm{EXP}(\log P)$; $l' = sp$ ($b = 0$), else $(\mathrm{EXP}(\mathrm{clamp8}(\log\alpha + \log l)) \ll LN) + sp$; $\log l = \mathrm{LOG}(\min(l' \gg LN, 127))$ | running max $m$; $P = e^{s - m}$; $\alpha = e^{m_{old} - m_{new}}$; $l \leftarrow \alpha l + \sum_j P$ |
| F5 | weighted sum | PV DIMM | $N_{EP}$ (EXP, identity) | $\log P$ of block $(r, b)$, $L_V$ of block $b$ ($B_{kv} \times d$) | $o_{part}$: $S \times d$, 13 bits | $o_{part,mc} = \sum_{j \in b} (-1)^{s(L_{V,jc})}\, \mathrm{EXP}(\mathrm{trunc8}(\log P_{mj} + L_{V,jc}))$ | $\sum_j P_{mj} v_{jc}$ |
| F6 | rescale and add | softmax accumulator | $N_{OC}$ EXP + $N_{OC}$ LOG (identity) | $o_{part}$, $\log\alpha$, $(lo, so)$ of block $b - 1$ | $(lo, so)$: $S \times d$ | $b = 0$: $o' = o_{part}$. $b \ge 1$: $o' = (so\ ?\ {-1} : 1) \cdot \mathrm{EXP}(\mathrm{clamp8}(\log\alpha + lo)) \ll SO + o_{part}$. Then $so = [o' < 0]$, $lo = \mathrm{LOG}(\min(\lvert o' \rvert \gg SO, 127))$ | $o \leftarrow \alpha o + \sum_j P v$, as log magnitude + sign |
| F7 | normalize | softmax accumulator (EXP group) | (EXP) | final $(lo, so)$, final $\log l$ | $O_8$: $S \times d$ int8 | $e_E = \mathrm{EXP}(\mathrm{clamp8}(lo - \log l))$; $O_8 = so\ ?\ \max(-e_E, -128) : \min(e_E, 127)$ | $O = o / l$ |

$SO = LN$, so the scale $2^{SO - LN}$ of F7 is 1. With the ACAM forms replaced by the real exp and log and the
saturations removed, this dataflow is textbook $\mathrm{softmax}(qk^\top 2^{-5})\,v$
(`attention_ref.head_mirror`, float ops).

### 2.3 Order and output

1. The GEMMs read $X$ once, tokens $t = 0 \ldots n - 1$. F1 writes $L_Q$, $L_K$, $L_V$ into the DIMMs' operand
   memories, which hold them for the whole head.
2. For each run $r = 0 \ldots R_b - 1$ and block $b = 0 \ldots B - 1$, the stages run in the order F2 → F3 → F4 → F5 →
   F6. Consecutive blocks overlap as a pipeline (§5).
3. After block $B - 1$ of run $r$, F7 gives the run's $S$ rows of $O_8$.
4. **Output:** $O_8$ rows in token order. The latency runs from `start` (weights preloaded) to the last $O_8$ word + 1.

**Carried state** (cleared per run):
- the online softmax keeps $(m, \log l)$ per query row;
- the accumulator keeps $(lo, so)$ per element.

The DIMMs keep no state between jobs.

## 3 Interfaces

### 3.1 Word and window

A port carries **units**. A unit is a matrix (a token row, a tile's result, a block's $\log P$, ...) or a per-row
vector. A unit is sent as follows:
- it is cut row-major into **windows** of $W$ elements;
- each window goes out as $\lceil W/Q \rceil$ **words** of $Q$ slots:
  - $Q = 5$ for matrices;
  - $Q = 1$ for the vectors $\log\alpha$ and $\log l$;
- windows are dealt to $L$ **channels**: window $w$ goes on channel $w \bmod L$;
- framing is by count: $W$, $Q$ and the unit size are build-time parameters of both ends.

Signals of channel `x` (an $L$-channel port packs channel $i$ into field $i$ of each signal):

| signal | width | direction | description |
|---|---|---|---|
| `x_v` | 1 | producer → consumer | a word is valid this cycle |
| `x_d` | $Q \cdot E$ | producer → consumer | slot $i$ in bits $[iE, (i+1)E)$ = element $Qc + i$ of the window, where $c$ is the word's index in the window |
| `x_cr` | 1 | consumer → producer | credit: one pulse per unit freed in the consumer's buffer |
| `x_rdy` | 1 | consumer → producer | burst ports only: the consumer's crossbar can take a burst now (`MSB_SA_Ready`) |

**Invariants** (each one checked by a port monitor):
- **W1 gapless:** a window's words are on consecutive cycles.
- **W2 padding:** slots past a window's last element are 0.
- **W3 order:** on a channel, windows are in increasing index, and units are in the producer's order.
- **W4 lock-step:** the channels of one DIMM output carry one wave's words in the same cycles.

### 3.2 Port disciplines

1. **Push + credit** (every module output inside the head):
   - The producer sends a unit at its own pace, and never waits inside it.
   - It starts a unit only with a credit in hand: its counter starts at `CAP`, drops by one when it decides to start a
     unit, and rises by one on each `x_cr` pulse.
   - The consumer's buffer holds `CAP` units. It takes every word (there is no ready), and pulses `x_cr` once per unit,
     after its last read of that unit.
2. **Burst on ready** (the GEMM input):
   - A row burst starts only in a cycle with `x_rdy` = 1, and then does not stop.
   - The head's X port uses this discipline (§4.7).
3. **Push without credit** (the head output $O_8$): the environment takes every word.

**Why push + credit?** A crossbar's drain cannot stall. So exactly one buffer is needed per edge, and the producer
must know before it starts that the buffer has room. Here the buffer sits in the consumer, which needs the data in
its own layout anyway.

### 3.3 Edges

$NG_{QK} = N_E/d$ and $NG_{PV} = N_{EP}/B_{kv}$ are the DIMMs' output channels. $NWEL = \min(S B_{kv}, I)$ is the
softmax window.

The example column is BERT-Tiny ($n$ 128, $d$ 64, $d_m$ 128) at 128×64, with $P_{proj} = 1$ and $N_E = 2d$
[derived].

| edge | data | from → to | unit | $E$, $Q$ | $W$ | $L$ | discipline, `CAP` | credit | example |
|---|---|---|---|---|---|---|---|---|---|
| E0 | $X$ | environment → GEMM ×3, copy $p$ | token row, $d_m$ | 8, 5 | $\min(d_m, R)$ (row tile) | $V$ | burst on ready, token order across copies | — | 26 words/token |
| E1 | $q_8$ | GEMM Q → QK DIMM, A port | token row, $d$ | 8, 5 | $d$ | $P_{proj}$ | push, 2 staging groups | `qa_cr` | 13 words/row |
| E2 | $k_8$ | GEMM K → QK DIMM, B port | token row, $d$ | 8, 5 | $d$ | $P_{proj}$ | push, 2 staging groups | `qb_cr` | 13 words/row |
| E3 | $v_8$ | GEMM V → PV DIMM, B port | token row, $d$ | 8, 5 | $d$ | $P_{proj}$ | push, $4P_{proj}$ rows | `pb_cr` | 13 words/row |
| E4 | $s_8$ | QK DIMM → online softmax | tile: $S \times B_{kv}$ | 8, 5 | $I$ | $NG_{QK}$ | push, 2 tiles | `s_cr` | 16 windows of 13 words, 2 channels |
| E5 | $\log P$ | online softmax → PV DIMM, A port | block: $S \times B_{kv}$ | 8, 5 | $NWEL$ | $N_{EXP}$ | push, 2 blocks | `pa_cr` | 16 windows of 4 rows × 16 keys |
| E6 | $\log\alpha$ | online softmax → accumulator | block $b \ge 1$: $S$ windows of one row | 9, 1 | 1 | 1 | push, 2 blocks | `la_cr` | 64 words |
| E7 | $\log l$ | online softmax → accumulator | run: $S$ | 8, 1 | $S$ | 1 | push, 1 run | `ll_cr` | 64 words |
| E8 | $o_{part}$ | PV DIMM → accumulator | tile: $S \times d$ | 13, 5 | $I$ | $NG_{PV}$ | push, 2 tiles | `op_cr` | 64 windows, 8 channels |
| E9 | $O_8$ | accumulator → head output (the O projection) | run: $S \times d$, one window per normalize wave | 8, $5 N_{OC}$ | $5 N_{OC} LW$ | 1 | push, no credit | — | 8 windows of 13 words of 320 bits per run |

The staging groups of E1–E2 are §4.4's $G$ rows: $I/B_{kv}$ for Q, $B_{kv}$ for K. A group is freed once every slice
of the DIMM has written it out. Since several GEMM copies share one of these credits, a copy compares instead of
decrementing: it may start token $t$ when the groups freed plus `CAP` exceed $t$'s group index.

## 4 Architecture

### 4.1 Top

```text
                     X [n x d_m] int8                     E0 burst on ready, P_proj x V lanes
                       |
   +-------------------v----------------------+
   | GEMM x 3 (W_V, W_Q, W_K), P_proj copies  |   F0  x8 = trunc8(X W)
   +--+--------------+--------------+---------+
      | E3 v8        | E1 q8        | E2 k8                      push; credits pb_cr, qa_cr, qb_cr
      |        +-----v--------------v-------+
      |        | tiled DIMM "QK"            |   F1  LOG on the A and B ports
      |        |  A: q8 rows   B: k8 rows^T |   F2  acc = sum_k +-EXP(L_Q + L_K)
      |        |  N_E EXP crossbars, trees  |   F3  s8 = sat8(acc >> 5)
      |        +-------------+--------------+
      |                      | E4 s8 [S x Bkv] per tile, NG_QK ch     push; credit s_cr, 2 tiles
      |        +-------------v--------------+
      |        | online softmax             |   F4  m, log a, log P, log l
      |        |  input FIFOs + packer      |
      |        |  N_EXP EXP, carry EXP/LOG  |
      |        +--+-------------------+--+--+
      |           | E5 log P          |  | E6 log a, E7 log l     E5 pa_cr; E6 la_cr; E7 ll_cr
      |           | N_EXP channels    |  |
   +--v-----------v-----+             |  |
   | tiled DIMM "PV"    |             |  |   F1  LOG on the B port
   |  A: log P  B: v8   |             |  |   F5  o_part = sum_j +-EXP(log P + L_V)
   |  N_EP EXP crossbars|             |  |
   +---------+----------+             |  |
             | E8 o_part [S x d] 13 bit, NG_PV ch                 push; credit op_cr, 2 tiles
   +---------v------------------------v--v--+
   | online softmax accumulator              |   F6  (lo, so) per element
   |  input FIFOs, log a slots, log l table  |   F7  O8 = EXP(lo - log l), signed
   |  EXP group + LOG group, e_mem           |
   +--------------------+--------------------+
                        | E9 O8 [n x d]: N_OC words per cycle    push (read by the O projection)
                        v

   att_ctrl: token admission (x_rdy), the job lists of both DIMMs, softmax run starts, done
```

E6 and E7 pass beside the PV DIMM; it does not touch them.

### 4.2 Modules

| module | folder | instances | crossbars | built from |
|---|---|---|---|---|
| GEMM `gemm_top` | `ip/gemm/` | $3 P_{proj}$ | $V$ each ($H = 1$) | the certified `v2/sim/rtl/gemm_top.v`, unchanged |
| tiled DIMM `dimm_tile_top` | `ip/dimm/` | 2 (QK, PV) | farm + one LOG converter per int8 input channel | the verified core of `v2/sim/rtl/dimm_tile_top.v` (memories, window engine, farm, trees), cut into slices of 16 crossbars, plus new operand ports, slots, job queue and output narrowing |
| online softmax `softmax_online_top` | `ip/softmax/` | 1 | $N_{EXP} + 2$ | `v2/sim/rtl/softmax_online_top.v`, plus a new input stage, credit gates, output framing, P29 |
| accumulator `softmax_online_acc` | `ip/softmax/` | 1 | $2 N_{OC}$ | the datapath of `v2/sim/rtl/att_ocarry_top.v`, plus stream inputs, credits and an O row reader |
| controller `att_ctrl` | `top/` | 1 | 0 | new |
| top `att_head` | `top/` | 1 | 0 | wiring |
| crossbar `dpe` | `ip/xbar/` | every crossbar | 1 | certified |
| FIFO (1W1R, any depth) | `ip/lib/` | inside the softmax and the accumulator | 0 | `v2/sim/rtl/att_fifo.v` |

Crossbars per head ($H = 1$ on every row) [derived]:

$$
N_{xbar} = \underbrace{3 P_{proj} V}_{\text{GEMMs}} + \underbrace{3 P_{proj}}_{\text{LOG converters}} + N_E + N_{EP} + (N_{EXP} + 2) + 2 N_{OC}
$$

The example: 3 + 3 + 128 + 128 + 4 + 16 = 282.

Each module's full spec (cycle formulas, port tables, case corpus) goes into `ip/<module>/spec.md` in its phase
(§8). This document fixes the boundaries, the ports and the data.

### 4.3 GEMM (`gemm_top`)

**Function.** $Y = \mathrm{trunc8}(X W)$, REGULAR. Used here with $K = d_m$, $N = d$:
- $V = \lceil d_m/R \rceil$ row tiles reduced by a byte tree;
- $H = 1$;
- P29 extent: rows $\min(d_m, R)$, columns $d$.

**Ports** (unchanged):

| port | width | description |
|---|---|---|
| `act_in`, `act_en` | $V \cdot 40$, 1 | E0: one row per pass, lane $v$ = row tile $v$, gapless burst |
| `MSB_SA_Ready` | 1 (out) | a burst may start (`x_rdy`) |
| `data_out`, `out_valid` | 40, 1 (out) | E1–E3: the row's $d$ results, $\lceil d/5 \rceil$ words, `out_valid` = word valid |
| `reg_full` | 1 (out) | a busy flag, one cycle behind the words (not a word valid) |
| `dpe_done` | 1 (out) | pulse after a pass's last word |

`out_valid` was an internal register (the v2 testbench read it by name). v3 makes it a port; nothing else in
`gemm_top` changed (2026-10-14, found at the head bring-up: `reg_full` had been listed here as the word valid).
| `weight_in`, `weight_en` | 8, 1 | simulation twin only |

**Consumes** an $X$ row. **Produces** one int8 row of $Q$, $K$ or $V$. **Holds** the weights only.

**Timing** (`gemm_sim.cycle_model`): one row per $T_p$, the $T_{steady}$ of its pass.
- $T_p = 34$ cycles in the example.
- The three GEMMs of a copy run in lock-step on the same X burst.

### 4.4 Tiled DIMM (`dimm_tile_top`)

**Function.** One job computes one product in the log domain:

$$
C_{mn} = \sum_{k} (-1)^{s_A(m,k) \oplus s_B(k,n)}\ \mathrm{EXP}\big(\mathrm{trunc8}(L_A[m][k] + L_B[k][n])\big)
$$

with $C = AB$, or $C = AB^\top$ when `TRANS_B` = 1.
- An int8 operand is converted inside the module: $L = \mathrm{LOG}(x)$, sign $s(L)$.
- A log-domain operand is used as given: sign $s(L)$, or + when `X_SIGNED` = 0.

**Parameters.**

| group | parameters | meaning |
|---|---|---|
| shape | `M, N, K, R, C, BUF, P, N_E` | job shape, crossbar, farm |
| slices | `X_S` | crossbars per slice: 16, fixed (not searched; the DSE searches $N_E$) |
| per operand X ∈ {A, B} | `X_DOM` | 0: int8 (LOG converters on the port), 1: log bytes |
| | `X_SIGNED` | sign from the log byte, or + |
| | `X_CH`, `X_W` | input channels; elements per input window (a whole number of rows) |
| | `X_SLOTS` | matrices held |
| B only | `TRANS_B` | 1: the B port carries $B^\top$ rows (columns of $B$) |
| output | `NARROW` | 0: full width ($9 + \log_2 K$ bits); $s > 0$: $\mathrm{sat8}(C \gg s)$ |
| | `C_CAP` | credits (consumer buffer, in jobs) |
| derived | $NB = I/N$, $W_K = \max(1, MN/I)$, $NG = N_E/K$, $T_W = K W_K / N_E$; slices $n_s = \max(1, N_E/X_S)$ of $K_s = \min(X_S, N_E)/NG$ columns; $A_{GRP}$, bank counts | no free numbers |

**Ports.**

| port | width | direction | description |
|---|---|---|---|
| `a_v`, `a_d` | `A_CH`, `A_CH`·40 | in | A rows (int8 or log bytes), row-major windows |
| `a_cr` | 1 | out | an A slot was freed |
| `b_v`, `b_d` | `B_CH`, `B_CH`·40 | in | B rows (or $B^\top$ rows) |
| `b_cr` | 1 | out | a B slot was freed |
| `j_v`, `j_a`, `j_b`, `j_ra` | 1, $\lceil \log_2$ `A_SLOTS` $\rceil$, $\lceil \log_2$ `B_SLOTS` $\rceil$, 1 | in | a job: the A slot, the B slot, release A after the job (B slots are never released) |
| `j_rdy` | 1 | out | the one-entry job register is empty: a job offered now is taken (burst on ready) |
| `c_v`, `c_d` | $NG$, $NG \cdot 5 E_C$ | out | $C$: window $w$ on channel $w \bmod NG$, $I$ elements per window |
| `c_cr` | 1 | in | the consumer freed one job's output |
| `busy` | 1 | out | |
| `weight_in`, `weight_en` | 8, 1 | in | simulation twin only (identity weights) |

The module's full contract (timing rules, named latencies, memories, gates) is `ip/dimm/spec.md`; its §10 lists
the refinements of this section made there.

**Slices.** The farm is cut into $n_s$ slices of $X_S = 16$ crossbars (one slice when $N_E \le X_S$).
- Crossbar $f$ works on column $k = \lfloor f/NG \rfloor$ of group $f \bmod NG$ for the whole tile. So slice $s$ is
  the columns $[sK_s, (s+1)K_s)$ of all $NG$ groups, i.e. crossbars $[sX_S, (s+1)X_S)$.
- **Per slice:**
  - the operand memories of its columns;
  - a loader slice with one A write port and one B write port, one word per cycle each;
  - its crossbars with their feed and drain registers;
  - its own copy of the window engine, led by the slice's crossbar 0;
  - the first $\log_2 K_s$ levels of each group's adder tree.
- **Shared:**
  - the LOG converters;
  - the distribution of each input word to the slice(s) that own its columns;
  - the job logic;
  - the last $\log_2 n_s$ levels of each group's tree, registered like the levels inside a slice, so the total depth
    stays $\log_2 K$;
  - the narrowing and the output channels.
- **Lock-step without signals between slices.** All slices start a job in the same cycle. Every crossbar has a window
  in every wave (in the power-of-two domain $T_W$ is an integer and every wave is full). All crossbars have the same
  pass timing. So each slice's engine, following its own crossbar 0, issues the same strobes in the same cycles as
  every other slice.

**Why slices?** With one engine per farm, every control bit drives the whole farm: up to 16,384 sinks per bit with a
128-crossbar farm [derived: RTL sinks per bit, before synthesis]. A slice bounds this to at most 2,688, whatever the
farm size. It also splits each operand's write traffic: a slice writes only its own columns. One write port per slice
then keeps up on every port, at all 432 candidates: the worst is 0.76 words per cycle, the Q port at $P_{proj} = 4$.

**Structure.**

```text
 A in (A_CH ch) -> [LOG converter per channel, if A_DOM = int8] --+
 B in (B_CH ch) -> [LOG converter per channel, if B_DOM = int8] --+--> distribution: each word to the slice(s)
                                                                  |    that own its columns
                                                                  v
 slice s = K_s columns x NG groups = X_S crossbars (s = 0 .. n_s-1):
   [A loader slice] -> [A memories: A_SLOTS slots]        [B loader slice] -> [B memories: B_SLOTS slots]
   [window engine copy, led by the slice's crossbar 0]
   crossbar f: feed u = trunc8(L_A[row][k] + L_B[k][col]), I elements -> dpe EXP (I x I) -> sign s_A ^ s_B -> +-e
   [per group: adder tree over the slice's K_s columns, log2 K_s levels]
                                                                  |
 jobs (j_*) -> [job logic: slots valid in every slice, every slice idle, output credit] --start--> every slice
 [per group: adder tree across the n_s slices, log2 n_s levels] -> [NARROW: sat8(. >> s)] -> C out (NG channels)
```

1. **LOG converter** (one per input channel of an int8 operand):
   - an identity crossbar in LOG mode, extent = the row length;
   - one row per pass, with a one-row input buffer;
   - it keeps up because its pass ($\lceil d/5 \rceil + 8$ cycles; 21 in the example) is never longer than the GEMM's
     row period: $d \le \min(d_m, R)$ on every shape [derived].
2. **Distribution.** A converted row, or a softmax window, is a sequence of 40-bit words. Each slice keeps the words
   that hold its columns; a word that straddles a slice boundary goes to both. It is registered, with at most $n_s$
   loads per bit.
3. **Loader slices.** They stage rows and write the farm's word layout, where A word (slot, $k$, window $w$) = rows
   $wNB \ldots wNB + NB - 1$ of column $k$.
   - **A, row-framed input** (Q):
     - $G = NB$ row banks hold the slice's columns of each row, 2 groups deep;
     - row $t$ goes to bank $t \bmod G$, so rows arriving together on several channels land in different banks
       ($P_{proj} \le G$);
     - a complete group is emitted as $K_s$ column words, one per cycle.
   - **A, window-framed input** ($\log P$):
     - per input channel, a window register holds the slice's $K_s$ keys of the window, 2 windows deep;
     - each window gives $K_s \times$ (rows per window / $NB$) words, one per cycle;
     - the channels take turns on the slice's write port.
   - **B, `TRANS_B` = 0** (V): the slice's rows (its keys) go one word each through a short queue.
   - **B, `TRANS_B` = 1** (K): $B_{kv}$ rows' slice columns are staged; then $K_s$ words are emitted, word $k$ = byte
     $k$ of each staged row.
4. **Slots.**
   - Matrix $i$ of a port fills slot $i \bmod$ `X_SLOTS`. A slot is valid once every slice has written its part.
   - A job whose release bit is set frees its slot at `done`.
   - Each port returns one kind of credit on `x_cr`:
     - a port whose slots are released credits per slot (the PV A port here);
     - a port whose slots hold the whole head credits per staging group, once every slice has written the group out
       (the Q, K and V ports here);
     - a window-framed port needs no staging credit, because its staging keeps up by the write-rate guard.
5. **Core of a slice** (the verified datapath, per slice):
   - A memories: $X_S/A_{GRP}$ banks of $NB$-byte words at {slot, wave, index}, blocked. One write port per slice
     means no interleaving is needed.
   - $A_{GRP}$ is the largest power of two $\le \min(\lceil I/5 \rceil, X_S)$: 8 / 16 / 16 at $I$ = 64 / 128 / 256.
   - B memories: k-slot memories of $N$-byte words for the slice's $K_s$ columns.
   - Per crossbar:
     - prefetch registers;
     - the feed word computed one cycle ahead;
     - `dpe` (EXP, identity weights, $I \times I$);
     - a 4-entry A sign ring;
     - a registered signed drain value.
   - The engine copy: take-now acceptance, depth-2 drain queue.
   - The slice's levels of the registered adder trees.
6. **Job logic.** The offered job is taken when all of these hold:
   - its A slot and its B slot are valid in every slice;
   - every slice is idle;
   - the output credit is ≥ 1.

   The decision is registered and starts all slices.

**The two instances** (example values [derived]):

| | QK | PV |
|---|---|---|
| $M \times N \times K$ | $S \times B_{kv} \times d$ | $S \times d \times B_{kv}$ |
| A port | E1: $q_8$ rows, int8, $P_{proj}$ channels, row frame; `A_SLOTS` = $R_b$ (all of Q), no release | E5: $\log P$, log bytes, sign +, $N_{EXP}$ channels, window frame ($NWEL/B_{kv}$ rows per window); `A_SLOTS` = 2, released per job |
| B port | E2: $k_8$ rows, int8, `TRANS_B` = 1; `B_SLOTS` = $B$ (all of K), no release | E3: $v_8$ rows, int8, `TRANS_B` = 0; `B_SLOTS` = $B$ (all of V), no release |
| farm | $N_E \in \{d, 2d\}$ | $N_{EP} = 2d$ |
| slices | $N_E/16$ slices of $16/NG$ columns | $2d/16$ slices of $128/d$ keys |
| output | E4: `NARROW` = 5, int8, $NG = N_E/d$ channels | E8: full width (13 bits), $NG = 2d/B_{kv}$ channels |
| crossbars | $N_E + 2P_{proj}$ (converters on A and B) | $N_{EP} + P_{proj}$ (converter on B) |
| example | $NB$ 4, $W_K$ 16, $NG$ 2, 8 waves, 1,024 windows; 8 slices of 8 columns; 130 crossbars | $NB$ 1, $W_K$ 64, $NG$ 8, 8 waves, 1,024 windows; 8 slices of 2 keys; 129 crossbars |

**Timing** (the core, `dimm_tile_sim.tile_cycle_model`, with $A_{GRP}$ by the slice rule above):
- $T_W$ waves;
- start latency $A_{GRP} + 2$;
- output word $c$ of wave $t$ at its drain start $+ c + 1 + \log_2 K$; the tree levels across slices are counted in
  $\log_2 K$;
- job total $T(T_W) + \log_2 K + A_{GRP} + 4$.

Two side effects of the $A_{GRP}$ cap at $I = 256$: 16 cycles less start latency per job, and twice the QK A memories
(8 instead of 4 at $N_E = 128$).

The new stages add named latencies, pinned at gate D2 (§7):
- the converter: one pass;
- `LD_LAT`: a group's last row → its last word written, including the distribution register;
- the registered job decision.

**Guards.**
- Powers of two for $M, N, K, N_E, X_S$; $N \le I$; $2 \le K \le N_E$; $N_E / K \le W_K$; $NG \le X_S$ (a slice holds
  whole columns).
- Write rate per slice port ≤ 1 word per cycle:
  - A, row frame: $K_s P_{proj} \le NB\, T_p$, with $T_p$ the producer's row period;
  - A, window frame: $N_{EXP} K_s \cdot$ (rows per window / $NB$) $\le T_{sm}$, with $T_{sm} = \lceil I/5 \rceil + 8$
    the softmax EXP pass period;
  - B: $K_s P_{proj} \le B_{kv} T_p$.

  At $X_S = 16$ the worst values over the 432 candidates are 0.76 (Q), 0.76 ($\log P$), 0.19 (K) and 0.06 (V) words
  per cycle [derived]. So $B_{kv} = 16$ runs on 64×64.
- For the record: with one write port per slice, the PV farm size is limited only by this rate guard. $N_{EP} = d$
  would pass at $I \ge 128$, and at $I = 64$ only with $X_S = 8$ [derived]. $N_{EP}$ stays $2d$ for now; searching
  $\{d, 2d\}$ at $I \ge 128$ is open (`experiment.md` §7).

**Why int8 operands with an internal LOG?** It makes the DIMM a general matrix product. Its producers then need
nothing special: the GEMM's output goes in directly. The price is one converter crossbar per int8 input channel
($3P_{proj}$ per head).

**Why row-major ports?** The column-major layout is the farm's own dataflow: one crossbar per $k$, outer product over
$k$. With row-major ports, no producer ever needs to know the window, group or bank map. `TRANS_B` covers $QK^\top$
without a transpose outside the module.

### 4.5 Online softmax (`softmax_online_top`)

**Function.** One run per `start`: $S$ query rows over $N$ keys in blocks of $B_{kv}$. F4 runs per block, and the
final $\log l$ is sent at the end of the run.

**Parameters:** `S, BKV, N, R, C, BUF, P, N_EXP`; `S_CH` (input channels = the QK DIMM's $NG$); credit capacities.

**Ports.**

| port | width | direction | description |
|---|---|---|---|
| `s_v`, `s_d` | `S_CH`, `S_CH`·40 | in | E4: score windows ($I$ elements, int8) |
| `s_cr` | 1 | out | a tile's windows have all been read |
| `lp_v`, `lp_d` | $N_{EXP}$, $N_{EXP} \cdot 40$ | out | E5: $\log P$ windows; window $g$ of the run on lane $g \bmod N_{EXP}$ |
| `lp_cr` | 1 | in | the consumer freed a $\log P$ slot |
| `la_v`, `la_d` | 1, 9 | out | E6: $\log\alpha$ of blocks $b \ge 1$, one row per word |
| `la_cr` | 1 | in | the consumer freed an $\alpha$ slot |
| `ll_v`, `ll_d` | 1, 8 | out | E7: the final $\log l$ of the run, one row per word |
| `ll_cr` | 1 | in | the consumer freed the $\log l$ slot |
| `start`, `busy`, `done` | 1 each | in, out, out | one run per `start`; `done` after the run's $\log l$ has been sent |

**Consumes** $s_8$ per tile. **Produces** $\log P$ and $\log\alpha$ per block, and $\log l$ per run. **Holds**
$(m, \log l)$ per row within a run.

**Structure.**

```text
  E4 (S_CH ch) -> [input stage: FIFO per channel (2 tiles), read in window order, 20-byte packer] -> block stream
     -> [store: 2-block ring]   [row max during the stream -> m_new, log a] -----------------> E6 log a
     -> [EXP feed g = 0 .. N_EXP-1: read a window, s - m_new, clamp -128 -> 4-deep FIFO] ---------> E5 lane g
              -> [EXP crossbar g] -> [row sums sp]
     -> [carry: b = 0: lq = min(sp >> LN, 127); b >= 1: carry-EXP(clamp8(log a + log l)); carry-LOG -> log l]
     -> [log l out: rows 0 .. S-1 after the run] ---------------------------------------------> E7 log l
```

1. **Input stage** (new):
   - one FIFO per channel, 40 bits wide, depth $2^{\lceil \log_2(2 T_W LW) \rceil}$ (two tiles);
   - the reader takes window $w$ from FIFO $w \bmod$ `S_CH`, one word per cycle, in window order;
   - a 20-byte packer forms the block stream: $S B_{kv}$ bytes, 5 per word, each block padded to whole words;
   - `s_cr` pulses after a tile's last read.

   A block's stream starts when all of these hold:
   - the run is armed;
   - the store has a slot (block $b - 2$ is retired);
   - a $\log P$ credit is available;
   - for $b \ge 1$, an $\alpha$ credit is available.

   **Why pack?** Window-aligned words would remove the packer. But they cost 1–12 extra words per block (up to
   1.46 %) [derived], on rows where the score stream sets the pace.
2. **Core** (the verified datapath):
   - store: $\max(2, N_{EXP})$ banks × 5 byte-lane RAMs, a 2-block ring, window-local layout;
   - row max during the stream: $m_{new}$ and $\log\alpha$ tables, ping-pong by block parity;
   - $N_{EXP}$ EXP feeds: each reads its windows, subtracts $m_{new}$ and clamps. The words its crossbar pops are the
     $\log P$ output, which is why E5 cannot stall and needs the credit;
   - row sums (2 stages, ping-pong);
   - the carry FSM with the carry-EXP and carry-LOG crossbars;
   - the final carry registers.

   The EXP crossbars use the extent $I \times I$; the carry-EXP and carry-LOG passes $S \times S$ (the rows in use).
3. **Outputs** (framing new, data unchanged):
   - E5 = the popped feed words;
   - E6 = the per-row $\log\alpha$ events of blocks $b \ge 1$;
   - E7 = the final $\log l$, read row by row through the 2-cycle registered read once `ll_cr` allows it.

   `done` follows the last E7 word.

**Timing** (`softmax_online_sim`): the certified recurrence, with per-block score availability. The stream floor is
$W_{st} = \lceil S B_{kv}/5 \rceil$ words per block (205 in the example).

The module's full contract is `ip/softmax/spec.md`. Its §10 lists the refinements made there: E6 carries one row per
window; the carry passes use $S \times S$; the gates sit in the input stage's reader, which opens a tile one cycle
after its decision; `done` follows the last E7 word.

### 4.6 Online softmax accumulator (`softmax_online_acc`)

**Function.** The output half of the online softmax: F6 per block and F7 per run. It applies to any weighted sum under
a blocked online softmax; here, the PV products.

**Parameters:** `S, D, R, C, BUF, P, N_OC, NBLK, SO, LN`; `P_CH` (input channels = the PV DIMM's $NG$); `VWP` (13).

**Ports.**

| port | width | direction | description |
|---|---|---|---|
| `p_v`, `p_d` | `P_CH`, `P_CH` · 65 | in | E8: $o_{part}$ windows (13-bit lanes) |
| `p_cr` | 1 | out | a tile's $o_{part}$ has all been read |
| `la_v`, `la_d`; `la_cr` | 1, 9; 1 | in; out | E6: $\log\alpha$; credit after an $\alpha$ slot's last use |
| `ll_v`, `ll_d`; `ll_cr` | 1, 8; 1 | in; out | E7: $\log l$; credit after the normalization's last read |
| `o_v`, `o_d` | 1, $N_{OC}$ · 40 | out | E9: the final $O_8$ values, one 40-bit word per crossbar per cycle, as the normalize jobs drain |
| `run_done` | 1 | out | pulse one cycle after a run's last $O_8$ word |

**Consumes** $o_{part}$ and $\log\alpha$ per block, and $\log l$ per run. **Produces** the final $O_8$ values.
**Holds** $(lo, so)$ per element within a run; the $O_8$ is not stored, it leaves on E9 as it is computed.

**Structure.**

```text
  E8 -> [FIFO per channel, 2 tiles] --+     E6 -> [log a: 2-entry ring]     E7 -> [log l table]
                                      v
  [LOG group, N_OC crossbars]  feed o' = (b ? +-e << SO : 0) + o_part -> oq, so    -> drain lo -> lo_mem, so_mem
  [EXP group, N_OC crossbars]  rescale feed   t  = clamp8(log a + lo)            -> drain e  -> e_mem
                               normalize feed tE = clamp8(lo - log l)            -> drain O8 -> E9 (o_v, o_d)
```

1. **Inputs:**
   - FIFOs of $5 \cdot 13$ bits, two tiles deep; window $p$ in FIFO $p \bmod$ `P_CH`;
   - $\log\alpha$: a 2-entry ring of $S \times 9$-bit tables, in arrival order;
   - $\log l$: an $S \times 8$-bit table.
2. **Engines** (the verified datapath):
   - Two lock-step groups of $N_{OC}$ identity crossbars ($I \times I$), each led by its crossbar 0.
   - Window $p = wN_{OC} + i$ runs on crossbar $i$ in wave $w$. There are $P_{oc} = S d/I$ windows per run, each of
     $RPW = I/d$ query rows.
   - Per crossbar: `lo_mem` and `e_mem` ($W_{oc}\, 2^{\lceil \log_2 LW \rceil}$ words of 40 bits), and `so_mem`.
   - Readiness:
     - **LOG** $(b, w)$: its windows are complete in their FIFOs, and the rescale $(b, w)$ is drained when $b \ge 1$.
     - **Rescale** $(b \ge 1, w)$: an $\alpha$ table is present and LOG$(b - 1, w)$ is drained.
     - **Normalize** $(w)$: $\log l$ is present and LOG$(B - 1, w)$ is drained.
   - A run's jobs begin after the previous run's last $O_8$ word.
3. **Output (E9):** the normalize jobs' $O_8$ words, registered, two cycles after each drain word, one per crossbar:
   - `o_d` word $i$ is word $q$ of crossbar $i$'s window $p = wN_{OC} + i$ (wave $w$); crossbars without a window in
     the last wave send zeros, as do the lanes past a window's $I$ elements;
   - element $k$ of window $p$ is element $pI + k$ of the run's row-major $S \times d$ tile;
   - there is no O reader and no read-out: the O projection reads E9 (user, 2026-10-15: the read-out is set by the
     40-bit port, not by the crossbars or the mapping). `run_done` pulses one cycle after the run's last $O_8$ word.

**Why in the softmax IP?** It uses only what the online softmax produces ($\alpha$, $l$) and the weighted partials.
Together with `softmax_online_top` it forms the whole online softmax over key blocks. Nothing in it is specific to
attention.

**Timing:** the engines follow the verified v2 recurrence (`ocarry_sim`). The module's full contract, with the stream
inputs and the output, is `ip/softmax/acc_spec.md`. Its §10 lists the refinements made there: the $\alpha$ ring in
arrival order across runs, not cleared at a run's end; the table credits; `run_done`; no O reader (2026-10-15).

### 4.7 Tiling controller (`att_ctrl`) and top (`att_head`)

`att_ctrl` is the only logic specific to attention. It holds counters only:

| task | rule |
|---|---|
| token admission | `x_rdy`[$p$] = the copy's three GEMMs have `MSB_SA_Ready`, and the QK A, QK B and PV B ports have staging credit for copy $p$'s next token $t$ |
| QK job list | $j = 0 \ldots R_b B - 1$: A slot $\lfloor j/B \rfloor$, B slot $j \bmod B$, no releases |
| PV job list | the same $j$: A slot $j \bmod 2$ (release), B slot $j \bmod B$ |
| softmax runs | run 0 at `start`; run $r + 1$ at the softmax's `done` of run $r$ |
| `done` | the accumulator's `run_done` of run $R_b - 1$ (one cycle after its last $O_8$ word on E9) |

The full contract is `top/spec.md`. Its §10 lists the refinements made there:
- `x_rdy` stays low until the copy's burst has ended, because a `dpe` keeps `MSB_SA_Ready` high through its load burst;
- `att_ctrl` reads `x_v` to count the tokens a copy has taken;
- jobs and runs are offered from the head start;
- one start per reset.

The `att_head` ports:

| port | width | direction | description |
|---|---|---|---|
| `x_rdy` | $P_{proj}$ | out | copy $p$ may take its next token |
| `x_v`, `x_d` | $P_{proj}$, $P_{proj} V \cdot 40$ | in | E0: copy $p$'s token burst, `LOADX` = $\lceil \min(d_m, R)/5 \rceil$ words, $V$ lanes; tokens $t \equiv p \pmod{P_{proj}}$ in order |
| `o_v`, `o_d` | 1, $N_{OC}$ · 40 | out | E9: the final $O_8$ values, $N_{OC}$ words per cycle (read by the O projection) |
| `start`, `busy`, `done` | 1 each | in, out, out | one head per `start` |
| weights | | in | simulation twin only; the VTR twin has the weights preloaded |

## 5 Flow control and schedule

### 5.1 Start rules

Every rule is local to its module and decided in a register:
- **GEMM copy $p$:** a token burst starts on `x_rdy`[$p$] (§4.7).
- **DIMM:** the offered job is taken when both its slots are valid, the farm is idle and an output credit is free
  (§4.4).
- **Online softmax:** a run starts on `start`. A block's stream starts under the four conditions of §4.5. A block's EXP
  windows issue once its row maxima are complete.
- **Accumulator:** the job readiness of §4.6.

### 5.2 Credits and fixed sizes

| edge | buffer (in the consumer) | `CAP` | freed when |
|---|---|---|---|
| E1 | QK A staging | 2 groups of $I/B_{kv}$ rows | every slice has written the group's column words |
| E2 | QK B staging | 2 groups of $B_{kv}$ rows | every slice has written the group's words |
| E3 | PV B row registers | $4 P_{proj}$ rows | the row's word is written (each row goes to one slice) |
| E4 | softmax input FIFOs | 2 tiles | the tile's last word is read |
| E5 | PV A slots | 2 blocks | the PV job of that block is done |
| E6 | accumulator $\alpha$ ring | 2 blocks | the block's last rescale job has read it |
| E7 | accumulator $\log l$ table | 1 run | the run's last normalize job has read it |
| E8 | accumulator input FIFOs | 2 tiles | the block's last LOG job has read it |

**Why E3 has 4 rows per copy** (user, 2026-10-15). The GEMM cannot stall, so `att_ctrl` takes the E3 credit when
the token enters the GEMM. The credit's loop is therefore X burst → V row out of the GEMM → LOG conversion → row
write → `pb_cr`: 65 cycles at BERT-Tiny on 64×64, against a GEMM row period of 21. A credit binds when its loop is
longer than `CAP` row periods. With 2 rows it bound the X stream on 176 of the 432 DSE builds (up to 8.9 % of the
cycles); with 4 it binds on none, and every build runs as fast as with no E3 credit at all (model,
`v3/docs/architecture_page/exp/pb_*.py`). The PV DIMM keeps two row registers per copy. It writes the V rows in row order, so
the X bursts must start in token order across the copies (E0; one token stream does). Then rows come at least one
conversion pass apart and stay at most $\lceil d/5 \rceil + P_{proj} - 1$ cycles, so at most one register per copy
is full at a time. Without the order, a starved copy would let the other copies' rows overrun their registers; the
model raises on it. (The alternative, 4 registers per copy, was not taken.)

Fixed sizes [derived from the floor rule]:
- $N_{OC}$ is the smallest group whose pass time reaches the stream floor:
  $T(\lceil P_{oc}/N_{OC} \rceil) \le \max(\lceil S B_{kv}/5 \rceil, T(1))$.
- That gives 6–8 at $d$ 64, 4 for JetFormer-150P and 2 for JetFormer-JetClass; 8 in the example.

### 5.3 No deadlock

Each module processes its units in one fixed order: tokens in order; tiles and blocks in $j = rB + b$; runs in order.

A unit waits only for:
- input units with an index ≤ its own;
- credits that come back after units with an index ≤ its own − `CAP` (`CAP` ≥ 1).

So every wait points to a strictly earlier unit, and no cycle can form.

One rule makes this hold: **ports fed by the X stream (E1–E3) hold the whole head** (`A_SLOTS` = $R_b$ for Q,
`B_SLOTS` = $B$ for K and V, no release). The three projections share one X stream. If Q slots were released and
reused, a full Q slot would stop X. With X stopped, the K blocks that the current run still needs would never be
produced.

The testbenches carry a no-progress watchdog as the empirical check.

### 5.4 Data per head

| edge | bytes [derived] | example |
|---|---|---|
| $X$ (read once) | $n d_m$ | 16,384 |
| $q_8$, $k_8$, $v_8$ | $n d$ each | 8,192 each |
| $s_8$, $\log P$ | $n^2$ each | 16,384 each |
| $\log\alpha$ | $R_b (B - 1) S \cdot 9/8$ | 1,008 |
| $\log l$ | $n$ | 128 |
| $o_{part}$ | $(n^2 d / B_{kv}) \cdot 13/8$ | 106,496 |
| $O_8$ | $n d$ | 8,192 |

$o_{part}$ is the largest stream: 13 bits per element per key block.

### 5.5 Memories

Sizes in bytes [derived]. VTR counts memory blocks, so this does not predict the VTR memory figure.

| memory | module | organization | size | example |
|---|---|---|---|---|
| $L_Q$ (A slots) | QK DIMM | per slice $X_S/A_{GRP}$ banks of $NB$-byte words ($N_E/A_{GRP}$ in all) | $n d$ | 8,192 |
| $L_K^\top$ (B slots) | QK DIMM | per slice $K_s$ k-slots of $B_{kv}$-byte words | $n d$ | 8,192 |
| $\log P$ (A slots) | PV DIMM | per slice $X_S/A_{GRP}$ blocked banks, 2 slots | $2 S B_{kv}$ | 2,048 |
| $L_V$ (B slots) | PV DIMM | per slice $K_s$ k-slots of $d$-byte words | $n d$ | 8,192 |
| staging Q / K / V | DIMM loader slices | row banks holding the slice's columns / row queues | | about 640 / 2,560 / 130 |
| staging $\log P$ | PV DIMM loader slices | 2 windows per channel (the slice's keys), flip-flops | $2 N_{EXP} \cdot 5 LW$ in all | 260 |
| score FIFOs + store | online softmax | `S_CH` × 2 tiles; 2-block ring | | 2,560 + 2,080 |
| $o_{part}$ FIFOs | accumulator | `P_CH` × 2 tiles × 65 bits | | 16,640 |
| lo / e / so | accumulator | per crossbar $W_{oc}\, 2^{\lceil \log_2 LW \rceil}$ words | | 10,880 |
| total | | | | about 62 KB, plus small register tables |

## 6 Model ↔ RTL contract

- **One model per module**, with the module's name, parameters and ports. The head model is the same graph as
  `att_head`, and `att_ctrl` is modeled by the same rules.
- **Event unit: the window.** Events are:
  - the cycle of each window's first word, per (port, channel, unit, window);
  - each credit pulse;
  - each job take;
  - each token burst.

  Windows are gapless (W1), push units never stall, and crossbar bursts never stop. So these events fix the cycle of
  every word: the event list is exact.
- **A module model** maps its input events and the credits arriving at its outputs to its output events and the
  credits it returns.
  - Where a datapath already has a certified model, that model is reused: `gemm_sim`, `dimm_tile_sim`,
    `softmax_online_sim`, `ocarry_sim`.
  - The new stages add their pinned latencies: converters, loaders, slots, the softmax input stage, the E9 output.
  - The DIMM's slices are internal and add no port events. The DIMM model uses the slice rule for $A_{GRP}$ (§4.4).
- **Composition** visits the units in the order of §5.3, so one pass computes every event.
- **Gate per module:** port monitors in the testbench log the same events. A module passes when its values are
  bit-exact, every event is equal and its total cycles are equal (`tools/rtl_gate2.py`).
- **Energy:** per-module counts (passes per mode, CLB operations, the traffic of §5.4) times the frozen per-operation
  prices.

## 7 Validation plan

All RTL runs are on 64×64 crossbars with small corpora. A full iverilog launch needs your approval. VTR runs only on
your go.

| gate | covers | oracle |
|---|---|---|
| D1 (DIMM memory images, in every D2 case) | distribution, loader slices and slots: each slice's A and B memory image at the end of the case; random arrival gaps, late credits, back-to-back jobs, slot release; the write-rate and overflow probes | numpy: each slice's operand memory image = its columns of the converted, transposed input rows |
| D2 (DIMM) | the QK and PV configurations at their ports, with several slices: int8 operands with converters, `TRANS_B`, window-framed log input on 2 channels, narrowing; credit starvation; a monitor that every slice's engine state is equal in every cycle | `attention_ref.signed_tile`, `dimm_tile_sim` |
| S (online softmax) | the online-softmax corpus driven through the windowed input; credit holds on $\log P$, $\alpha$, $\log l$; one $R \ne C$ case for the extent (your approval) | `softmax_online_ref`, `softmax_online_sim` |
| A (accumulator) | its corpus with stream inputs; the $\alpha$ ring; O rows and their order | `attention_ref.ocarry_stage_values`, `epilogue` |
| H (head) | small 64×64 heads: the bring-up ($V = 1$ and $V > 1$, $R_b > 1$, $B = 2$, $P_{proj} \in \{1, 2, 4\}$), then gate H (late and slow X, a late start, $N_E = d$, $V = 3$, $d = 64$) | values: `head_stages` stage `O8`; events: the composed model |
| V (VTR smoke) | each module alone, then the head: crossbar count == expected, no logic optimized away; Fmax, CLB, memory, path #1. For the DIMM: one 128-crossbar point | VTR twins |

In every gate:
- credits returned late;
- back-to-back units;
- the smallest sizes ($n$ 32, $d$ 16).

**Weight loading in every testbench** (`PLAN.md` R8):
- Weights are preloaded at zero simulated time, never strobed. The weight file is read with `$readmemh`; after each
  reset, every `dpe`'s `weights` and `mode_q` are written through the hierarchy. Identity crossbars get the
  identity the same way.
- No testbench drives the `weight_in` / `weight_en` strobes (R·C cycles per crossbar), not even in one case. The
  strobe path exists only in the sim twin (the VTR twin has bare `dpe` blocks), and a wrong preload fails the value
  checks.
- Programming is untimed setup, so every cycle contract and measured number is unchanged.

Not in the RTL gates (your decision, 2026-10-14, when the DIMM closed):
- **RTL mutants.** Each gate compares every value and every port event, cycle by cycle, so a planted bug in a credit,
  a value or a slice's state fails it. The module models keep their mutant tests (`simulator.md` §7).
- **A credit that never comes back.** A hang fails the case anyway: the testbench's watchdog, or the runner's timeout.
- **RTL runs at the 432 candidates' parameter sets.** The model checks that every build is inside the module's range
  (for the DIMM, also the write rate $\le 1$). In the DIMM's logic, $R$ and $C$ enter only through $I$; every size is
  a power of two with its widths computed by formula, so the 64×64 cases run the same formulas. The DSE's VTR runs
  build every pick.
- **A weight-strobe case** (above).

## 8 Implementation phases

| phase | work | accepted when | why this order |
|---|---|---|---|
| A | tiled DIMM: slices, converters, distribution, loader slices, slots, job queue, narrowing; both twins | gate D2 (with D1 in every case); VTR smoke of the DIMM | the largest change, and the Fmax and area risk |
| B | online softmax: input stage, credit gates, output framing, extent | gate S | small changes around a verified datapath |
| C | accumulator: stream inputs, $\alpha$ ring, credits, O reader | gate A | |
| D | `att_ctrl` + `att_head` | gate H; VTR smoke of the head | |
| E | module models at the RTL ports, then the composition | events equal per module, then per head | alongside A–D, module by module |

The GEMM needs no work.

## 9 Risks and decision gates

| risk | gate | response |
|---|---|---|
| the farm-wide fan-out limits Fmax (one engine per farm: up to 16,384 sinks per control bit [derived]) | the VTR smoke of gate V at a 128-crossbar point, seed 11: Fmax, CLB, memory and path #1 | slices of 16 are fixed; if path #1 is still a farm-wide net, report it for your decision |
| restructuring the verified core into slices breaks it | gate D2 with several slices, the per-cycle engine-equality monitor | fix the RTL; the core's datapath per crossbar is unchanged |
| the DIMM's new operand side costs Fmax or area | the same smoke: the loader and distribution paths in the timing report | register the loader → memory path |
| the O read-out holds the next run's rescale jobs (`e_mem` is the O buffer) | the head model reports the hold time per row of the DSE | resolved 2026-10-15: no O reader; the $O_8$ words leave on E9 as they are computed (`acc_spec.md` §10 item 7) |
| the softmax extent change is visible only at $R \ne C$ | one $R \ne C$ case (an exception to the 64×64 rule, your approval) | otherwise $R \ne C$ softmax timing stays model-only, flagged |
| memory growth (staging, FIFOs) | VTR memory count per module | |
| deadlock | §5.3, the watchdog, credit starvation cases | |

## 10 Open for your review

1. The module boundaries and port contracts (§3, §4).
2. (resolved 2026-10-15: no O reader; the head outputs the final $O_8$ values and the O projection reads them,
   `experiment.md` §5.)
3. One $R \ne C$ case for the softmax extent.
4. The VTR smoke of gate V at one 128-crossbar point, on your go.
5. (resolved 2026-10-15: $N_{EP} = 2d$ in the fixed mapping, `experiment.md` §4.2.)

`experiment.md` was updated to this document on 2026-10-14.
