# Where an attention head's time and energy go, stage by stage

> **Status note (2026-10-23):** pre-bit-plane analysis. It models the previous design (the 5-byte port, `BUF` = 40)
> at the $N_E = 2d$ mapping and describes the 2026-10-16 run; frozen until the port to the bit-plane results after
> stage 4 (`dse/stage_analysis.py`, `stages_*.py`, `pipelines.py`).

Status: 2026-10-19, an explainer for the sensitivity study (user: the energy and time breakdown by stage for the
smallest and the largest head, how the mapping sets the crossbars of each stage, and how the array size enters).
The numbers are the simulator's (`dse/config_table.py`, `dse/critical_path.json`, the priced tables); the equations are
the closed forms of `mapping_sensitivity.md` §2, checked against the simulator on the 72 builds.

Symbols: $n$ tokens, $d_m$ model width, $d$ head width; the array $R \times C$, $I = \min(R, C)$; $S = \min(n, I)$
queries per run; $B_{kv} = 16$ keys per block; the port moves 5 bytes per cycle (40 bits); $P = 8$ bit-slices.

## 1 Four stages, one kind of pass

Every stage works in passes of one crossbar. A pass loads its input rows through the port, runs the 8 bit-slices, and
drains the output columns. Its period is

$$
\tau = \max\!\Big(\Big\lceil \frac{8 \cdot \text{rows loaded}}{40} \Big\rceil + 8,\ \Big\lceil \frac{8C}{40} \Big\rceil + 1\Big)
\qquad\text{(21 cycles for 64 rows on 64 columns, 60 for 256 on 256)}
$$

and its energy fires the whole array whatever it uses:

$$
p(R, C) \approx 10.94 \cdot \frac{C}{64} + 1.31 \cdot \frac{RC}{8192} + 0.015 \cdot \text{rows loaded} + 0.01 \cdot C \quad \text{pJ}
\qquad\text{(13.8 pJ at 128×64, 23.0 at 1024×64, 92.0 at 1024×256)}
$$

The first term is the ACAM output stage, one per column; the second the cells, one per bit-slice per cell. The stages:

| stage | what the crossbars do | crossbars (the mapping) | passes per head (the work) |
|---|---|---:|---:|
| projections Q, K, V | a real matrix–vector product: a token's $d_m$ inputs in, $d$ outputs out; $V = \lceil d_m / R \rceil$ tiles per projection | $3 P_{proj} V$ (+ $3 P_{proj}$ converters feeding the DIMMs) | $3 V n$ |
| tiled QK | $Q K^\top$ in the log domain: the products are CLB adds, the crossbars convert back (EXP) $I$ values per pass with identity weights; CLB trees sum | $N_E = 2d$ | $n^2 d / I$ (+ $2n$ conversions of the rows) |
| online softmax | reads each $S \times B_{kv}$ score block through one port, EXP of the scores, running max and sum, carries | $N_{EXP} + 2 = 4$ | $n^2 / I$ (+ 2 carries per block) |
| tiled PV | $P V$: the same as QK | $N_{EP} = 2d$ | $n^2 d / I$ (+ $n$) |
| output rescale | the o-carry: rescale and LOG passes per block | $2 N_{OC}$ (4–16) | $2 n^2 d / (I B_{kv})$ |

Two facts carry the whole explanation. **The work does not depend on the array:** a conversion pass handles $I$
elements, so a larger $I$ means fewer passes of the same total, and the projections do $3 d_m d$ MACs per token
whatever $R$. **The crossbar count does not depend on the array either:** it is $4d$ for the two farms plus a few,
set by the rate rule below, not by $R \times C$.

## 2 Why the farms have $2d$ crossbars (how the mapping sets the counts)

A stage must keep up with the score stream, which delivers a block of $S B_{kv}$ scores in $S B_{kv} / 5$ cycles
through one port. A farm has $S B_{kv} d$ conversions per block, $I$ per pass of $\tau$ cycles, over $N$ crossbars:

$$
\frac{S B_{kv} d}{I} \cdot \frac{\tau}{N} \le \frac{S B_{kv}}{5}
\;\Longrightarrow\;
N \ge \frac{5 d\, \tau}{I} \approx d\Big(1 + \frac{40}{I}\Big) \;\to\; 2d \text{ (rounded to a power of two, } I \ge 40).
$$

The same rule gives $N_{EXP} \ge 1 + 40/I \to 2$ for the softmax's EXP lanes and the floor rule's 2–8 o-carry groups.
So the mapping parameters set the crossbars per stage as: projections $3 P_{proj} V$, QK $N_E$, softmax $N_{EXP} + 2$,
PV $N_{EP}$, accumulator $2 N_{OC}$; the DSE's values are the smallest that keep up (`mapping_sensitivity.md` §4–§5:
more buys nothing, fewer costs 8–61 % cycles). Only the projections' count follows the array, through $V = \lceil d_m / R \rceil$:
18 crossbars for ViT at $R = 128$, 3 at $R = 1024$.

## 3 Energy, by stage

$$
E \approx \underbrace{3 V n\, p}_{\text{projections}} + \underbrace{2 \cdot \frac{n^2 d}{I}\, p}_{\text{QK + PV passes}} + \underbrace{\frac{n^2}{I} p + \frac{2 n^2 d}{I B_{kv}} p}_{\text{softmax, accumulator}} + \underbrace{E_{CLB}}_{\text{adds, trees, traffic, memory: fixed}}
$$

The farms' passes are the bulk, so $E \propto n^2 d \cdot p(R, C) / I$: **every conversion fires $RC$ cells to use
$I$ of them.** Fired cells per useful element, $RC / I$: 128 at 128×64, 1,024 at 1024×64 and at 1024×256. In plain
terms:
- more rows at a fixed $C$: the ACAM term is unchanged, the cells term grows with $R$; the head costs +35 % from
  128×64 to 1024×64 for identical work;
- more columns at a fixed $R$: the passes halve ($I$ doubles) and the ACAM price doubles, so the energy is flat until
  $C > R$; then $I$ stays at $R$ while the ACAM price keeps growing (128×256: +55 %);
- the CLB part (the log-domain adds, the trees, the port traffic, the memories) is the same on every array: 813 nJ of
  QK's energy on ViT at both arrays, so it is 47 % of QK at 128×64 and 34 % at 1024×256.

| stage | JetFormer-150P ($n$ 32, $d_m$ 64, $d$ 32) 128×64 | 1024×256 | ViT-B/16 ($n$ 256, $d_m$ 768, $d$ 64) 128×64 | 1024×256 |
|---|---:|---:|---:|---:|
| projections | 1 nJ (5 %) | 9 (16 %) | 98 (3 %) | 85 (2 %) |
| tiled QK | 14 (45 %) | 24 (42 %) | 1,727 (46 %) | 2,367 (47 %) |
| online softmax | 1 (3 %) | 1 (2 %) | 50 (1 %) | 61 (1 %) |
| tiled PV | 14 (43 %) | 21 (36 %) | 1,687 (45 %) | 2,308 (45 %) |
| output rescale | 1 (5 %) | 2 (4 %) | 188 (5 %) | 263 (5 %) |
| **head** | **32 nJ** | **57 nJ** | **3,751 nJ** | **5,083 nJ** |

The two farms are 90 % of the energy of every head on every array; the array moves their price per pass (13.8 →
92.0 pJ) against their pass count (÷4 from $I$ 64 to 256), net +36 to +78 %.

## 4 Time, by stage

$$
T \approx \underbrace{\frac{n\, T_p}{P_{proj}}}_{\text{projection stream}} + \underbrace{R_b B \cdot \max\!\Big(\frac{S B_{kv}}{5},\ T_{QK},\ T_{PV},\ T_{EXP},\ T_{ACC}\Big)}_{\text{core}} - \text{overlap},
\qquad T_p = \max\!\Big(\Big\lceil \frac{8 \min(d_m, R)}{40} \Big\rceil + 8,\ \Big\lceil \frac{8C}{40} \Big\rceil + 1\Big)
$$

- **Projections:** a token's inputs enter at 5 bytes per cycle, so a pass takes as long as the rows it loads. With
  $R \ge d_m$ one tile loads all $d_m$ rows (154 cycles for ViT); with $R < d_m$ the $V$ tiles load $R$ rows each in
  parallel (26 cycles at $R = 128$) and a tree of $\log_2 V$ cycles sums them. A wide $C$ adds a drain of
  $\lceil 8C / 40 \rceil$ cycles that stops hiding behind the next load at $C = 256$. This stage is 36–71 % of the
  critical path, and it is where the whole $R$ penalty sits.
- **Tiled QK and PV:** busy for $n^2 d\, \tau / (I N)$ cycles each, $\approx \frac{n^2}{2}(\frac{1}{5} + \frac{8}{I})$
  with $N = 2d$: nearly the same on every array, because a crossbar converts $I / \tau \approx 3$–4 elements per cycle
  whatever its size (the port, not the array, sets the rate). They sit under the stream floor and are hidden; PV
  shows 6–20 % of the path as the tail of each run.
- **Online softmax:** the score stream through one port, $n^2 / 5$ cycles in all, is the core's floor on every
  array; 19–43 % of the path.
- **Output rescale:** hidden but for its tail (1–20 %).

| stage, cycles on the critical path | JetFormer-150P 128×64 | 1024×256 | ViT-B/16 128×64 | 1024×256 |
|---|---:|---:|---:|---:|
| projections | 675 (53 %) | 1,667 (65 %) | 8,710 (36 %) | 41,475 (71 %) |
| tiled QK | 185 (14 %) | 121 (5 %) | 87 (0 %) | 134 (0 %) |
| online softmax | 276 (22 %) | 116 (4 %) | 10,480 (43 %) | 11,299 (19 %) |
| tiled PV | 0 | 155 (6 %) | 4,851 (20 %) | 4,403 (8 %) |
| output rescale | 141 (11 %) | 518 (20 %) | 265 (1 %) | 994 (2 %) |
| **head** | **1,279** | **2,579** | **24,395** | **58,307** |

From the smallest array to the largest, JetFormer doubles (its 21-cycle projection pass becomes drain-bound at 53)
and ViT grows 2.4× (its projection pass goes from 34 to 162 cycles: 768 rows through one port instead of six tiles
of 128); the core of each changes little.

## 5 The model: cycles, energy and area per stage (the stage page)

The equations the stage page carries, with our mapping's constraints ($P_{proj} = 1$, $N_E = N_{EP} = 2d$,
$N_{EXP} = 2$, $N_{OC}$ by the floor rule, $B_{kv} = 16$, $S = \min(n, I)$, a reader of $K_{reader} = 5$ scores per
cycle). Symbols: $n$, $d_m$, $d$; $I = \min(R, C)$; $S = \min(n, I)$, $N_{run} = n/S$, $N_{block} = n/B_{kv}$;
$V = \lceil d_m/R \rceil$; $w = 5$ bytes per cycle per port, $P = 8$ bit-slices. A pass that loads $r$ rows takes
$\tau(r) = \max(\lceil r/w \rceil + P, \lceil C/w \rceil + 1)$ cycles and costs $e_{pass}(r) = 0.17C + 0.00016RC +
0.015r + 0.010C$ pJ plus its port words $e_{port}(r) = 0.0495(\lceil 8r/40 \rceil + \lceil 8C/40 \rceil)$ pJ.

**Cycles per stage** (one key block of $S \times B_{kv}$ scores):

$$
T_{projection\_token} = \tau(\min(d_m, R)), \quad T_{projection\_stream} = n\,T_{projection\_token}/P_{proj}
$$
$$
T_{QK} = \frac{S B_{kv} d}{I N_E}\tau(I), \quad T_{PV} = \frac{S d B_{kv}}{I N_{EP}}\tau(I), \quad
T_{reader} = \Big\lceil \frac{S B_{kv}}{K_{reader}} \Big\rceil, \quad T_{EXP} = \frac{S B_{kv}}{I N_{EXP}}\tau(I)
$$
$$
T_{softmax\_latency} = T_{reader} + T_{EXP} + 2\tau(S) + \lceil S/w \rceil, \quad
T_{softmax} = \max(T_{reader}, T_{softmax\_latency}/2)
$$
$$
W_{rescale} = \Big\lceil \frac{S d}{I N_{OC}} \Big\rceil, \quad
T_{output\_rescale} = \frac{T_{EXP} + T_{PV} + W_{rescale}\,\tau(I) + 2\tau(I)}{2}
$$

The output rescale's period is half its loop: a block's row maxima reach it through a ring of two blocks, and a block
goes through the EXP passes, the PV job and the rescale waves before the ring frees (`architecture.md` §5.2, E6).

**Latency:**

$$
T_{block} = \max(T_{QK}, T_{softmax}, T_{EXP}, T_{PV}, T_{output\_rescale})
$$
$$
T_{stream\_bound} = T_{projection\_stream} + \max(1, N_{run} - 1)\,N_{block}\,T_{block}, \quad
T_{core\_bound} = S\,T_{projection\_token}/P_{proj} + N_{run} N_{block}\,T_{block}
$$
$$
T_{head} = \max(T_{stream\_bound}, T_{core\_bound}) + T_{fill} + T_{first\_block} + T_{readout} + T_{tail}
$$

with $T_{fill} = \lceil \log_2 V \rceil + 28$, $T_{first\_block} = N_{run}(T_{softmax\_latency} - T_{block})$,
$T_{readout} = (N_{run} - 1)(S + 4)$, $T_{tail} = T_{PV} + W_{rescale}\tau(I)$.

**Energy per stage** ($e_{add} = 0.085$, $e_{compare} = 0.264$, $e_{memory} = 0.0495$ pJ per 5 bytes):

$$
E_{projection} = 3Vn\,[e_{pass}(\min(d_m, R)) + e_{port}] + 3nd(V-1)e_{add}, \quad
E_{QK} = \Big(\frac{n^2 d}{I} + 2n\Big)[e_{pass}(I) + e_{port}] + (2n^2 d - n^2)e_{add} + n^2 e_{compare}
$$
$$
E_{softmax} = \Big(\frac{n^2}{I} + N_{run}(2N_{block} - 1)\Big)[e_{pass}(I) + e_{port}] + \tfrac{17}{8}n^2 e_{add} + \tfrac{17}{16}n^2 e_{compare}, \quad
E_{PV} = \Big(\frac{n^2 d}{I} + n\Big)[e_{pass}(I) + e_{port}] + \Big(2n^2 d - \frac{n^2 d}{B_{kv}}\Big)e_{add}
$$
$$
E_{output\_rescale} = \frac{2n^2 d}{I B_{kv}}[e_{pass}(I) + e_{port}] + \frac{n^2 d}{8}e_{add}, \quad
E_{memory} \approx \frac{e_{memory}}{5}\Big(3nd + \frac{2n^2 d}{B_{kv}} + \frac{2n^2 d}{S} + 4n^2 + \frac{7n^2 d}{B_{kv}}\Big)
$$

**Area per stage** ($A_{tile}(R, C)$ = the tile's grid cells × 66,117 MWTA × 0.033864 µm²):

$$
A_{projection} = 3P_{proj}V A_{tile}, \ A_{QK} = (N_E + 2P_{proj})A_{tile}, \ A_{softmax} = (N_{EXP} + 2)A_{tile}, \
A_{PV} = (N_{EP} + P_{proj})A_{tile}, \ A_{output\_rescale} = 2N_{OC}A_{tile}
$$

plus $A_{CLB}(I)$ and $A_{memory}$ at head level from VTR. Against the simulator on the 12 arrays: cycles within
12 % (JetFormer-150P) and 17 % (ViT-B/16; 11 of 12 within 6 %), energy within 0.1 %, area exact (the crossbar tiles plus VTR's CLBs and memory).
With our constants the output rescale's loop is the largest core term on every array, so $T_{block} =
T_{output\_rescale}$; the array enters the core only through $S$ and $\tau(I)$, the stream through $\min(d_m, R)$,
the energy through $e_{pass}/I$, the area through $A_{tile}$ and $A_{CLB}(I)$.

## 6 In plain English

The array enters the energy as *cells fired per useful element*, $RC / I$, and the time as *rows to load and columns
to drain per token*. Nothing in the head gets faster with a larger array: the crossbars are used as port-fed
converters at 3–4 elements per cycle each, their number is set by $d$ and the rate rule, and the projections only
get slower as more rows have to come through the port. The smallest array that holds the window, $I = d$, does the
same work in the same time with the fewest cells fired and the narrowest soft logic. FC is the opposite case: its
work is weight-stationary MACs, and a bigger array really does more MACs per pass.

## 7 The pipelined diagram, the closed-form models and the sweep (the stage-analysis page)

`dse/stage_analysis.py` runs the two case-study heads (`space.py CASE_STUDY`: JetFormer-150P and ViT-B/16) at the
12 configs and writes `dse/results/stages/stages.json`; `dse/plot_stages.py` draws the time diagram and the sweep;
`dse/stages_page.py` writes the page `dse/results/stages/stages.html` (published 2026-10-19).

**The time diagram** is the simulator's steps, one lane per stage (projections, tiled QK, online softmax, tiled PV,
output rescale; a stage's units merged into its busy intervals), at 128×64 and at 1024×256. Reading it: the projection stream feeds everything; the first run's blocks pace themselves on
the stream (a QK job needs its block's 16 keys and its run's $S$ queries); later runs' blocks follow at the block
period, with the softmax's tile read and EXP passes alternating on the critical path and the PV job and the
accumulator pipelining one or two blocks behind; each run ends with the $S$-cycle $\log l$ read-out.

**The cycle model** is §5's $T_{head}$ (the stage page, section "The model", with a bullet explanation of every
term); the page's table lists it against the simulator on every array.

**The energy model** is §5's per-stage closed form (passes at the pass price, the CLB adds and compares, the
memory traffic): within
0.0 % and 0.4 % of the
simulator on every array.

**The brief** (`dse/stages_brief.py` → `dse/results/stages/brief.html`, with `stages_costs.png`: stacked bars of the
critical-path cycles, the energy and the area by stage on the 12 arrays) is the one-page version for discussion,
without the model: the five stages with their work and what sets their crossbar count (the projections by capacity,
every other stage by bandwidth: the ports needed to keep pace with the score stream, which the array does not change),
what one crossbar delivers (3.0 to 4.8 values per cycle from I = 64 to 1024) against what it costs ($RC$ cells, so a
unit of bandwidth costs about 23× more cells at 1024×256 than at 128×64), the cost of every stage at 128×64 against
1024×256 with what the larger array gains at each (fewer ports for the projections; 1.4× per crossbar in the farms that
the reader cannot use; nothing in the softmax), why the knob sweep cannot reverse it (`mapping_optimum.json`: the
per-build optimum cuts every ADP to 0.25–0.79 of the fixed mapping and swaps only ranks 4 and 5), and the FC contrast
(2048 × 256 is 64 tiles of 128×64 or 2 of 1024×256: capacity-set, so its area falls with the array while attention's
rises). The line: attention needs ports, not cells.

**The sweep** walks the DSE's own grid: one curve per $C \in \{64, 128, 256\}$ against $R \in \{128, \ldots, 1024\}$,
four panels per head (cycles, energy, area, cycles × area relative to 128×64); lines are the closed forms
$T_{head}$, $E_{head}$, $A_{head}$, markers the simulator (`stage_analysis.py` `grid`; the findings below are
asserted by its self-test). Cycles: JetFormer-150P ($d_m = 64$) never loads a row beyond 64, so its time does not
change with $R$ (1,279 / 1,532 / 2,579 cycles at $C$ = 64 / 128 / 256 for every $R \ge C$) and grows only with $C$
through the drain; ViT-B/16 ($d_m = 768$) loads all $R$ rows per token, so its time grows with $R$ (24.4k to 57.2k
at $C = 64$), and $C = 128$ is the fastest at every $R$ because the core left after the projection stream,
$\max(1, N_{run} - 1) N_{block} T_{block}$, is smallest at $S = 128$ (7.9k cycles, against 13.1k at $S = 64$ and
13.9k at $S = 256$). Energy: the CLB adds, compares and memory are the same at every array (12.3 nJ of 31.8 nJ at
JetFormer-150P 128×64, 1.54 µJ of 3.75 µJ at ViT-B/16); the passes fire all $RC$ cells and drain all $C$ columns
for $I$ converted values, so energy rises with $R$ at every $C$ once $R \ge C$, and 128×256 ($R < C$: twice the
passes, each draining 256 columns) costs more than 256×256 at both heads. Area: the tiles scale with $RC$ and the CLB
with $I$, so area rises with $R$ and with $C$. Cycles × area rises with $R$ and with $C$ at every point: 128×64 is
the best at both heads and 1024×256 costs 4.4× (JetFormer-150P) and 5.4× (ViT-B/16); ViT-B/16's 128×128 is 9 %
faster but 31 % larger (1.19×). The mechanism: a projection pass loads $\min(d_m, R)$ rows, a DIMM pass loads
$I = \min(R, C)$ rows, and every pass fires all $RC$ cells and drains all $C$ columns; rows and columns beyond
those cost area, energy and drain time and save no time.
