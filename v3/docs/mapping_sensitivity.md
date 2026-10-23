# Mapping parameters, constants, and what they do to latency, area and energy

Status: 2026-10-18, analysis for the sensitivity study (user: "list out all the mapping parameters / constants, and
analytically analyze their impact on these evaluated metrics"). The DSE does not search the mapping
(`experiment.md` §4.2: one fixed mapping for every shape); this document lists everything the three metrics depend on
besides the crossbar config $(R, C)$ and the workload, derives how each enters the metrics, and checks the derivation
with one-at-a-time runs of the simulator (`dse/mapping_sensitivity.py` → `dse/results/mapping_sensitivity.md`; §4).

Throughout: $I = \min(R, C)$, $S = \min(n, I)$ queries per run, $R_b = n / S$ runs, $B = n / B_{kv}$ key blocks,
$V = \lceil d_m / R \rceil$ projection row tiles, $EPS = BUF / 8 = 5$ elements per port word, and the pass period of a
crossbar whose load is $I$ rows

$$
\tau_I = \max\!\big(\lceil 8I / BUF \rceil + P,\ \lceil 8C / BUF \rceil + 1\big)
$$

(the load plus the bit-slices, or the drain of all $C$ columns plus the hand-over, whichever is longer; `architecture.md`
§1, `simulator.md` §4.2).

## 1 The inventory

### 1.1 The mapping knobs (the head top's free parameters, `experiment.md` §4.2)

| knob | fixed value | rule that set it | what it sizes | where it enters the metrics |
|---|---|---|---|---|
| $P_{proj}$ | 1 | the configuration the VTR smoke synthesized | copies of the Q, K, V projection GEMMs; the X stream's parallelism ($T_X = n / P_{proj}$ tokens per copy) | the projection stream $n\,T_p / P_{proj}$ (cycles); $3 P_{proj} (V + 1)$ crossbars; the staging credits ($4 P_{proj}$ PV B rows) |
| $N_E$ | $2d$ | the synthesized configuration | QK farm crossbars: $NG_{QK} = N_E / d$ output channels, $T_W = (S B_{kv} / I) / NG_{QK}$ waves per job | the QK stage period (cycles); $N_E$ crossbars; $N_E / X_S$ slices of soft logic |
| $N_{EP}$ | $2d$ | with $d$ the accumulator's $N_{OC}$ cap falls below the floor rule and the $\log P$ staging overruns at $I = 64$ | PV farm crossbars: $NG_{PV} = N_{EP} / B_{kv}$ channels, $T_W = (S d / I) / NG_{PV}$ waves | the PV stage period; $N_{EP}$ crossbars; the cap $N_{OC} \le N_{EP} / B_{kv}$ |
| $N_{EXP}$ | 2 | the smallest power of two whose EXP stage keeps up with the score stream | EXP crossbars of the online softmax: $NWB = S B_{kv} / I$ passes per block over $N_{EXP}$ lanes | the EXP stage period; $N_{EXP} + 2$ crossbars; the PV A port's channels (the $\log P$ staging, a fixed RTL buffer) |
| $N_{OC}$ | the floor rule, 2–8 | the smallest group whose o-carry keeps up with the score stream (`architecture.md` §5.2), capped at $N_{EP} / B_{kv}$ | o-carry crossbars: $\lceil P_{oc} / N_{OC} \rceil$ waves per block, $P_{oc} = S d / I$ | the accumulator's stage period; $2 N_{OC}$ crossbars |
| $B_{kv}$ | 16 | the only value legal on every row | keys per block: $B = n / B_{kv}$ blocks, the $S \times B_{kv}$ score window | the stream floor per block $\lceil S B_{kv} / EPS \rceil$; the carry and rescale pass counts ($\propto 1 / B_{kv}$); $N_{OC}$'s floor and cap |
| $X_S$ | 16 | user 2026-10-14, bounds the fan-out | crossbars per DIMM slice: $N_E / X_S$ slices, the staging group $A_{GRP}$ | the DIMM start latency ($A_{GRP} + 2$) and staging rates (cycles, $\le 2\,\%$); the slice count (soft logic, Fmax: VTR only) |
| $B_q = S$ | $\min(n, I)$ | one query block per run | $R_b = n / S$ runs; the first run's QK can start once $S$ tokens are projected | how much of the core overlaps the projection stream; the run re-arm costs |

Derived from them and fixed by the RTL's contracts (`architecture.md` §5.2): the credits `LP_CAP` 2, `LA_CAP` 2, `LL_CAP` 1,
`C_CAP` 2, the QK A staging of 2 groups, the PV B registers of $4 P_{proj}$ rows, the softmax FIFOs of 2 tiles, the
accumulator FIFOs of 2 tiles; `S_CH` $= N_E / d$, `P_CH` $= N_{EP} / B_{kv}$. They bound the stream, they are not
choices; a knob value that makes one of them overrun is "outside a contract" in §4.

### 1.2 The hardware constants

| constant | value | where | what it fixes |
|---|---|---|---|
| $BUF$ | 40 bits | every module's guard (`BUF = 40`); the `dpe` port | every rate: a load of $r$ rows takes $\lceil 8r / 40 \rceil$ cycles, a drain of $C$ columns $\lceil 8C / 40 \rceil$, the score stream $\lceil S B_{kv} / 5 \rceil$ per block; the port traffic energy per word |
| $P$ | 8 bit-slices | the crossbar's `PRECISION` (int8) | the compute part of a pass, $P + 2$ cycles; the cells' energy per pass ($P$ bit-slice events) |
| the pass grammar | `RDY_LAT`, `SHIFT_ACC_LAT`, `ACAM_LAT`, `DRAIN_LAT` = 1 | `simulator/core/crossbar.py`, pinned by GATE 2 | $\tau_I$ above; back-to-back passes at $\lceil 8r/40 \rceil + P$ |
| the pass extent | load $ROWS\_USED$ rows (P29), drain all $C$ columns (2026-10-18) | `dpe_nldpe.v`, the models | the load side of $\tau$ follows the rows in use; the drain never shrinks |
| the module latencies | named constants (`CONV_IN_LAT`, `DIST_LAT`, `EMIT_LAT`, …, 3–15 per module) | the specs and models, pinned by GATE 2 | the `logic` share of the critical path (2–4 % of a head) |
| the pass price | cells $1.31\,\mathrm{pJ} \times RC / 8192$; ACAM $10.94\,\mathrm{pJ} \times C / 64$; input buffer $0.0149\,\mathrm{pJ}$ per row in use; output buffer $0.0099\,\mathrm{pJ}$ per column | `simulator/platform/nl_dpe.json`, `prices.py` (the whole array fires, only the rows in use are read) | the energy of a pass: $13.8\,\mathrm{pJ}$ at 128×64 (79 % ACAM), $23.0$ at 1024×64, $27.7$ at 128×128, $92.0$ at 1024×256 |
| the fabric prices | block RAM $0.0495\,\mathrm{pJ}$ per 40-bit access; CLB add $0.085\,\mathrm{pJ}$, compare $0.264\,\mathrm{pJ}$ | the same | the port traffic ($\lceil 8r/40 \rceil + \lceil 8C/40 \rceil$ words per pass), the memory bytes, the CLB terms |
| the tile footprint | cells per crossbar: 1×5 (128×64), 2×3 (256×64), 1×11 (1024×64, 128×128), 4×6 (256×256), 5×8 (1024×256), … | `vtr/arch/nl_dpe_<R>x<C>_auto.xml` | the crossbar area: $N_{xbar} \times w h \times 66{,}117$ MWTA ($0.033864\,\mu\mathrm{m}^2$ per MWTA) |
| CLB and memory tiles | 1 and 2 grid cells; counts from VPR | `area.py` | the soft-logic and BRAM area: VTR only, no model |
| Fmax | VTR, the mean of seeds {11, 13, 17}, channel width 300, `clock.sdc` | `vtr_dse.py` | latency $= T / F_{max}$ |

### 1.3 The evaluation conventions

Batch 1; one head (no O-projection, no multi-head sharing); weights preloaded (programming energy reported apart,
never per inference); dense attention (no masks); $n$ a power of two; $T = t_{done} - t_{start}$ of `att_ctrl`; energy
per inference from the frozen prices; FC as one GEMV with $V = \lceil K / R \rceil$, $H = \lceil N / C \rceil$ tiles, one
copy, the used pass extent. FC has no mapping knob: its only choice is the extent rule.

## 2 How the metrics depend on them

### 2.1 Cycles

The critical path (`experiment.md` §9.3) is the projection stream followed by the core, with overlap on the heads
that have several runs:

$$
T \;\approx\; \frac{n\, T_p}{P_{proj}} + R_b B \cdot \max\!\big(\underbrace{\lceil S B_{kv} / EPS \rceil}_{\text{score stream}},\ T_{QK},\ T_{PV},\ T_{EXP},\ T_{ACC}\big) - \text{overlap} + \text{fill},
\qquad T_p = \max\!\Big(\Big\lceil \frac{8 \min(d_m, R)}{BUF} \Big\rceil + P,\ \Big\lceil \frac{8C}{BUF} \Big\rceil + 1\Big)
$$

and each core stage's period is its work per block divided by its crossbars, times the pass period:

$$
T_{QK} \approx \frac{S B_{kv} d}{I\, N_E}\,\tau_I, \qquad
T_{PV} \approx \frac{S d B_{kv}}{I\, N_{EP}}\,\tau_I, \qquad
T_{EXP} \approx \frac{S B_{kv}}{I\, N_{EXP}}\,\tau_I, \qquad
T_{ACC} \approx \Big\lceil \frac{S d}{I\, N_{OC}} \Big\rceil \tau_I .
$$

At the fixed mapping on BERT-Tiny at 128×64 ($S = I = 64$, $\tau_I = 21$): the stream floor is 205 cycles per block,
$T_{QK} = T_{PV} = T_{EXP} = 168$, $T_{ACC} = 183$ ($N_{OC} = 8$). Every stage sits just under the floor: that is what
the rate rules of §1.1 do. Consequences:

- **A stage's crossbars matter only while that stage is the maximum.** Halving $N_E$ doubles $T_{QK}$ to 336, above
  the floor, and the block period follows it; doubling $N_E$ to $4d$ changes nothing. The same for $N_{EP}$, $N_{EXP}$
  and $N_{OC}$: removing crossbars costs the ratio of the stage period to the floor, adding them buys nothing.
- **$P_{proj}$ is the one knob that moves the other half of $T$.** The projection stream is $n T_p / P_{proj}$ cycles
  and holds 39–63 % of the critical path at $P_{proj} = 1$ (§9.3 of `experiment.md`), more at large $R$ and
  $d_m > R$ (the load $\lceil 8 \min(d_m, R) / 40 \rceil$ grows with $R$). Two copies halve it where the core does
  not take over.
- **$B_{kv}$ moves the floor and the stages together** ($\lceil S B_{kv} / 5 \rceil$ against $\tau_I$-multiples of
  $B_{kv}$), so its cycle effect is the rounding and the per-block fixed costs: small either way.
- **$S = \min(n, I)$ is where $I$ enters the core:** the first run's QK waits for $S$ projected tokens, so at
  $n = S$ (one run) nothing of the core overlaps the stream, while at $n = 2S$ the first run's core runs under the
  second half of the stream. This is the +21 % cycles of the $n = 128$ heads from 128×64 to 128×128, not the drain.
- **$BUF$ and $P$ scale every term:** all rates are per 40-bit word; $P$ adds to every pass. They are fixed by the
  RTL (the `BUF = 40` guards), so only an RTL change could move them.

### 2.2 Crossbars and area

$$
N_{xbar} = 3 P_{proj} (V + 1) + N_E + N_{EP} + (N_{EXP} + 2) + 2 N_{OC},
\qquad A_{xbar} = N_{xbar} \cdot w h \cdot 66{,}117\ \mathrm{MWTA}
$$

(checked by `tools/check_head_params.py` on the 72 builds). At $d = 64$ and the fixed mapping: 272–297 crossbars, of
which the two farms are $4d = 256$. So $N_E$ and $N_{EP}$ set the crossbar count ($+d$ per step, +22 % of the
crossbars per farm doubling), $P_{proj}$ adds $3(V + 1)$ per copy (6 at $V = 1$, 21 at $V = 6$), $N_{EXP}$ and $N_{OC}$
one or two each. The crossbar area is 9–36 % of a head's area (`experiment.md` §9.2); the rest is soft logic, whose
size VTR alone gives: it grows with $I$ (10.4k, 12.9k, 17.5k CLBs at $I$ = 64, 128, 256, the fixed mapping) and
presumably with the slice count $N_E / X_S$, the EXP lanes and the o-carry groups, which no VTR run of a variant has
measured. A variant's area in this analysis is therefore its crossbar area only; its soft logic and Fmax are open.

### 2.3 Energy

The passes of a head are set by the work, not by how many crossbars share it (checked against the simulator's ledger
on all 72 builds, `dse/mapping_sensitivity.py --self-test`):

$$
\text{gemm}: 3 V n, \quad
\text{QK}: \frac{n^2 d}{I} + 2n, \quad
\text{PV}: \frac{n^2 d}{I} + n, \quad
\text{softmax}: \frac{n^2}{I} + R_b (2B - 1), \quad
\text{acc}: 2 R_b B \frac{S d}{I} = \frac{2 n^2 d}{I B_{kv}} .
$$

None of $P_{proj}$, $N_E$, $N_{EP}$, $N_{EXP}$, $N_{OC}$, $X_S$ appears, and the CLB operations, the port traffic and
the memory bytes of the ledger are work terms too. So the energy is invariant under every mapping knob but $B_{kv}$,
which changes the carry and rescale passes: $\pm 5\,\%$ between $B_{kv}$ = 8 and 32 (§4). What the energy does depend
on is the config, through the pass price and the pass count:

$$
E_{core} \;\approx\; \frac{n^2 d}{I}\,\Big[\underbrace{10.94 \tfrac{C}{64}}_{\text{ACAM}} + \underbrace{1.31 \tfrac{RC}{8192}}_{\text{cells}} + \underbrace{0.0149\, I + 0.0099\, C}_{\text{buffers}} + \text{traffic}\Big]\ \mathrm{pJ} + \text{CLB} + \text{memory}.
$$

At fixed $C$ a larger $R$ only adds the cells term (128×64 → 1024×64: the price 13.8 → 23.0 pJ, the head +34 %); at
fixed $R$ a larger $C$ halves the passes ($\propto 1/I$) and doubles the ACAM, so the energy is about flat (128×64 →
128×128: +3 %); 128×256 pays the ACAM of 256 columns with the passes of $I = 128$ (+60 %). These are the energy
differences of `experiment.md` §9.4, and no mapping choice can change them.

## 3 Per knob: the direction and the size

| knob | cycles | energy | crossbars (area) | soft logic, Fmax | limit |
|---|---|---|---|---|---|
| $P_{proj}$ 1 → 2 → 4 | the stream $\propto 1 / P_{proj}$: -43 to -10 % (median -26 %) at 2, -64 to -15 % (median -35 %) at 4; more where the stream's share is larger (large $R$, $d_m > R$) | none | $+3(V+1)$ per copy | more X writers and staging (unmeasured) | $P_{proj} \le n$, a power of two |
| $N_E$ $d$ → $2d$ → $4d$ | $d$: QK binds, +1 to +32 % (median +8 %); $4d$: nothing (-1 to -0 % (median -0 %)) | none | $\pm d$ | $N_E / X_S$ slices | $N_E \ge d$ (`K <= N_E`) |
| $N_{EP}$ $d$ → $2d$ → $4d$ | $4d$: -9 to +0 % (median -4 %) (PV was just under the floor) | none | $\pm d$ | slices | $d$ puts the $N_{OC}$ cap $N_{EP} / B_{kv}$ below the floor rule on most builds |
| $N_{EXP}$ 1 → 2 → 4 | 1: EXP binds, +2 to +28 % (median +13 %); 4: -9 to +0 % (median -4 %) | none | $\pm 1$ | EXP lanes | 4 overruns the PV $\log P$ staging at $I = 64$ (a contract: the RTL's fixed buffer) |
| $N_{OC}$ half / floor / cap | half: the accumulator binds, +2 to +61 % (median +12 %); cap $= N_{EP} / B_{kv}$ (8 at $d = 64$): -1 to +0 % (median +0 %), the floor rule already picks 6–8 | none | $\pm 2$ per group | o-carry groups | $N_{OC} \le N_{EP} / B_{kv}$ and $\le S d / I$ |
| $B_{kv}$ 8 / 16 / 32 | 8: -3 to +9 % (median +1 %); 32: -0 to +13 % (median +3 %) (rounding and per-block costs) | 8: +4.6 %; 32: -2.3 % (carry and rescale passes $\propto 1/B_{kv}$) | 8: +8 to +16 ($N_{OC}$'s floor rises); 32: −8 | the score window | $8 \le B_{kv} \le \min(n, I)$; 8 fails the $N_{OC}$ cap on the $d = 32$ head |
| $X_S$ 8 / 16 / 32 | 8: -2 to -0 % (median -1 %); 32: +0 to +2 % (median +1 %) | none | none | the slice count and fan-out: the real effect, VTR only | 32 overruns the PV staging at $I = 64$ |

(The ranges are the one-at-a-time runs of §4 over the 72 builds; the direction and the mechanism are §2.)

## 4 The one-at-a-time sweep

`python3 v3/dse/mapping_sensitivity.py` builds every attention build (6 heads × 12 configs) with each knob changed
alone and writes `dse/results/mapping_sensitivity.md` (the summary below) and `.json` (every variant). Cycles,
passes, energy and crossbars are exact; Fmax and the CLB area of a variant would need VTR.

**Summary** (72 builds; the full tables with the per-config medians and every variant: `dse/results/mapping_sensitivity.md`, `.json`; 1.5 min to regenerate):

| knob | value | cycles ÷ fixed (min / median / max over the builds) | energy ÷ fixed (median) | crossbars Δ | builds outside |
|---|---|---:|---:|---:|---|
| P_PROJ | 2 | 0.57 / 0.74 / 0.90 | 1.000 | +6 … +21 | none |
| P_PROJ | 4 | 0.36 / 0.65 / 0.85 | 1.000 | +18 … +63 | none |
| N_E | d | 1.01 / 1.08 / 1.32 | 1.000 | -64 … -16 | none |
| N_E | 4d | 0.99 / 1.00 / 1.00 | 1.000 | +32 … +128 | 3 (domain: 1024×256, 256×256, 512×256) |
| N_EP | d | — | — | — | all 72 outside (domain) |
| N_EP | 4d | 0.91 / 0.96 / 1.00 | 1.000 | +32 … +128 | 3 (domain: 1024×256, 256×256, 512×256) |
| N_EXP | 1 | 1.02 / 1.13 / 1.28 | 1.000 | -1 … -1 | none |
| N_EXP | 4 | 0.91 / 0.96 / 1.00 | 1.000 | +2 … +2 | 20 (contract: 1024×64, 128×64, 256×64, 512×64) |
| N_OC | half | 1.02 / 1.12 / 1.61 | 1.000 | -8 … -2 | none |
| N_OC | cap | 0.99 / 1.00 / 1.00 | 1.000 | +0 … +4 | none |
| BKV | 8 | 0.97 / 1.01 / 1.09 | 1.046 | +4 … +18 | 7 (domain: 1024×256, 1024×64, 128×64, 256×256 …) |
| BKV | 32 | 1.00 / 1.03 / 1.13 | 0.977 | -8 … -4 | 12 (domain: 1024×128, 1024×256, 1024×64, 128×128 …) |
| X_S | 8 | 0.98 / 0.99 / 1.00 | 1.000 | +0 … +0 | none |
| X_S | 32 | 1.00 / 1.01 / 1.02 | 1.000 | +0 … +0 | 24 (contract: 1024×64, 128×64, 256×64, 512×64) |

The energy column is 1.000 for every knob but $B_{kv}$, as §2.3 derives. The two knobs that fail are failures of the
RTL's fixed buffers, not of the model: $N_{EXP} = 4$ and $X_S = 32$ overrun the PV DIMM's $\log P$ staging on the
$C = 64$ configs (20 and 24 builds), and $N_{EP} = d$ puts the accumulator's cap $N_{OC} \le N_{EP} / B_{kv}$
under the floor rule's pick on every build.

**Does the mapping change the latency order of the configs?** $P_{proj}$ is the only knob whose effect differs
between configs by more than a few percent (per-config medians at 4 copies: 0.78 at 128×64, 0.70 at 1024×64, 0.60 at
128×256, 0.53 at 1024×256: the stream's share of the path is larger on the large arrays). Cycles relative to 128×64,
geomean over the six heads:

| config | cycles ÷ 128×64, fixed mapping | cycles ÷ 128×64, P_proj = 4 |
|---|---:|---:|
| 128×64 | 1.00 | 1.00 |
| 256×64 | 1.13 | 1.01 |
| 512×64 | 1.29 | 1.05 |
| 1024×64 | 1.42 | 1.10 |
| 128×128 | 1.10 | 0.99 |
| 256×128 | 1.20 | 1.02 |
| 512×128 | 1.37 | 1.07 |
| 1024×128 | 1.51 | 1.11 |
| 128×256 | 1.50 | 1.24 |
| 256×256 | 1.52 | 1.13 |
| 512×256 | 1.72 | 1.20 |
| 1024×256 | 1.88 | 1.26 |

At 4 copies the cycle order of the $C \le 128$ configs collapses: all eight lie within 11 % of 128×64, and 128×128
is 1 % faster than it (1024×64 goes from 1.42× to 1.10×); the $C = 256$ configs stay 13–26 % slower (1024×256 from
1.88× to 1.26×). So with more projection copies the cycles no longer separate the small and medium arrays, and the
latency ranking would be decided by Fmax and by the soft logic of the 4-copy builds, both unmeasured: that is the
VTR run §7 asks for. The area and energy orders are unchanged by construction (§2.2, §2.3).

## 5 Can the best mapping be derived analytically?

For the core, yes, and it is the mapping the DSE used. With the block period bound below by the score stream
$F = \lceil S B_{kv} / EPS \rceil$ and each stage's period $W_{stage}\, \tau_I / N_{stage}$ (§2.1), the smallest stage
that keeps up is

$$
N_E^\* = N_{EP}^\* = \Big\lceil \frac{S B_{kv} d\, \tau_I}{I F} \Big\rceil \approx d\Big(1 + \frac{40}{I}\Big), \qquad
N_{EXP}^\* = \Big\lceil \frac{S B_{kv} \tau_I}{I F} \Big\rceil \approx 1 + \frac{40}{I}, \qquad
N_{OC}^\*: \text{the floor rule},
$$

using $\tau_I \approx 8I/40 + P$ and $F \approx S B_{kv}/5$. Rounded up to powers of two these are $2d$, $2d$, 2 and 2–8
for every head and every config with $I \ge 40$: the rate rules of §1.1 are these formulas, and the sweep confirms the
point is the optimum of cycles against crossbars (more buys nothing, less costs 8–61 %). Energy does not enter (it is
mapping-invariant), and the crossbar area grows linearly past the point, so on ADP and EDAP the balanced point is the
best the core can do; only the soft logic, which no VTR run of a variant has measured, could move it.

For the projections the DSE's $P_{proj} = 1$ is not the balanced value. The core consumes a block of $B_{kv}$ keys per
$F$ cycles, so the stream keeps up when $T_p / P_{proj} \le F / B_{kv} \approx S / 5$:

$$
P_{proj}^\* = \min\!\Big(n,\ \frac{I}{B_{kv}},\ 2^{\lceil \log_2 (5\, T_p / S) \rceil}\Big)
$$

(the cap $I / B_{kv}$ is the QK A staging group, `A_CH` $\le NB$ in `ip/dimm/spec.md`: 4 at $I = 64$, 8 at 128, 16 at
256). It is 2–16 and differs per head and config. `dse/mapping_sensitivity.py --balanced` builds every head at
$P_{proj}^\*$ with the core at the rate rule's values (`dse/results/mapping_balanced.md`):

| config | P_proj* per head | cycles fixed (geomean) | cycles balanced | ÷ fixed | ÷ n²/5 | crossbar area ÷ fixed | cycles ÷ 128×64: fixed → balanced |
|---|---|---:|---:|---:|---:|---:|---:|
| 128×64 | 4 / 4 / 4 / 4 / 4 / 4 | 8,410 | 6,533 | 0.78 | 1.99 | 1.15 | 1.00 → 1.00 |
| 128×128 | 2 / 2 / 2 / 2 / 8 / 2 | 9,234 | 6,986 | 0.76 | 2.13 | 1.09 | 1.10 → 1.07 |
| 128×256 | 4 / 4 / 4 / 4 / 8 / 4 | 12,616 | 7,845 | 0.62 | 2.39 | 1.18 | 1.50 → 1.20 |
| 256×64 | 4 / 4 / 4 / 4 / 4 / 4 | 9,492 | 6,603 | 0.70 | 2.02 | 1.12 | 1.13 → 1.01 |
| 256×128 | 4 / 2 / 4 / 2 / 8 / 2 | 10,062 | 6,977 | 0.69 | 2.13 | 1.11 | 1.20 → 1.07 |
| 256×256 | 2 / 2 / 4 / 4 / 16 / 4 | 12,747 | 7,419 | 0.58 | 2.26 | 1.17 | 1.52 → 1.14 |
| 512×64 | 4 / 4 / 4 / 4 / 4 / 4 | 10,871 | 6,827 | 0.63 | 2.08 | 1.11 | 1.29 → 1.05 |
| 512×128 | 8 / 2 / 8 / 2 / 8 / 2 | 11,548 | 6,944 | 0.60 | 2.12 | 1.14 | 1.37 → 1.06 |
| 512×256 | 4 / 2 / 8 / 4 / 16 / 4 | 14,464 | 7,363 | 0.51 | 2.25 | 1.19 | 1.72 → 1.13 |
| 1024×64 | 4 / 4 / 4 / 4 / 4 / 4 | 11,963 | 7,192 | 0.60 | 2.19 | 1.10 | 1.42 → 1.10 |
| 1024×128 | 8 / 2 / 8 / 2 / 8 / 2 | 12,727 | 7,124 | 0.56 | 2.17 | 1.11 | 1.51 → 1.09 |
| 1024×256 | 4 / 2 / 8 / 4 / 16 / 4 | 15,846 | 7,668 | 0.48 | 2.34 | 1.18 | 1.88 → 1.17 |

Read: the balanced mapping cuts the cycles by 22–52 % per config for 9–19 % more crossbar area, and the configs'
cycle differences collapse from 1.00–1.88 to 1.00–1.20 of 128×64: the latency ranking of the attention family is a
property of the fixed mapping as much as of the arrays. What remains above the stream floor $n^2/5$ (the ÷ $n^2/5$
column, 2.0–2.4) is structural: the online softmax's block pipeline is two deep, so the stream of one block and the
EXP passes of the previous alternate on the critical path (a block period of 235 against the floor 205 at $I = 64$);
each run adds its PV and accumulator tail and the re-arm; and where the cap $I / B_{kv}$ holds $P_{proj}^\*$ below
$5 T_p / S$ (BERT-base and ViT at $R \ge 512$, every $d_m = 768$ head at $C = 64$), the stream still binds. Those
three are RTL limits (the staging group, the pipeline depth, the 40-bit port), not mapping choices.

What the balanced numbers do not say: the soft logic and Fmax of a build with 2–16 projection copies (each copy is
an X bus, a GEMM copy with its three converters, and its staging rows). VTR runs of the $P_{proj}^\*$ builds on a few
configs are the measurement the sensitivity study needs before the mapping is changed.

## 6 The mapping optimized per build: does the ranking move?

The test the question needs (user, 2026-10-18): for every build, the mapping that minimizes each metric, then the
per-config geometric means and the ranking at those optima against the fixed mapping. `dse/mapping_sensitivity.py
--optimize` builds every mapping of the grid $P_{proj} \in \{1, 2, 4, 8, 16\} \times N_E \in \{d, 2d, 4d\} \times
N_{EP} \in \{2d, 4d\} \times N_{EXP} \in \{1, 2, 4\} \times B_{kv} \in \{8, 16, 32\}$ ($N_{OC}$ by the floor rule,
$X_S = 16$; 270 points, the ones a guard or a contract rejects dropped) for each of the 72 builds (13 min), and keeps
per build and per objective the minimizing mapping. Cycles and energy are the variant's own; the area is the fixed
build's CLBs and memory blocks (VTR) plus the variant's crossbar tiles, the latency its cycles over the fixed build's
Fmax: the soft logic and Fmax of a variant are assumed unchanged (`dse/results/mapping_optimum.md`, `.json`).

**ADP, fixed mapping against the ADP-optimal mapping per build** ($G_{ADP}$ as in `rank.py`, lower is better):

| config | fixed: G_ADP (rank) | ADP-optimal mapping: G_ADP (rank) | Δ rank | ADP ÷ fixed (geomean over the heads) | the mapping the heads pick |
|---|---:|---:|---:|---:|---|
| 128×64 | 0.201 (1) | 0.314 (1) | +0 | 0.65 | P_PROJ 4/8, N_E 2d, N_EP 4d, N_EXP 4, BKV 16/8 |
| 256×64 | 0.230 (2) | 0.326 (2) | +0 | 0.58 | P_PROJ 4/8, N_E 2d, N_EP 4d, N_EXP 4, BKV 16/8 |
| 512×64 | 0.275 (3) | 0.364 (3) | +0 | 0.55 | P_PROJ 4/8, N_E 2d, N_EP 2d/4d, N_EXP 4, BKV 16/8 |
| 128×128 | 0.306 (4) | 0.465 (5) | +1 | 0.63 | P_PROJ 4/8, N_E 2d, N_EP 2d/4d, N_EXP 4, BKV 16 |
| 1024×64 | 0.318 (5) | 0.401 (4) | -1 | 0.52 | P_PROJ 4/8, N_E 2d, N_EP 2d/4d, N_EXP 4, BKV 16/8 |
| 256×128 | 0.332 (6) | 0.468 (6) | +0 | 0.58 | P_PROJ 8, N_E 2d, N_EP 2d/4d, N_EXP 4, BKV 16 |
| 512×128 | 0.415 (7) | 0.527 (7) | +0 | 0.52 | P_PROJ 16/8, N_E 2d, N_EP 2d, N_EXP 4, BKV 16/8 |
| 1024×128 | 0.475 (8) | 0.561 (8) | +0 | 0.49 | P_PROJ 16/8, N_E 2d, N_EP 2d, N_EXP 4, BKV 16/8 |
| 128×256 | 0.483 (9) | 0.669 (9) | +0 | 0.57 | P_PROJ 4/8, N_E 2d, N_EP 2d, N_EXP 4, BKV 16/8 |
| 256×256 | 0.673 (10) | 0.795 (10) | +0 | 0.49 | P_PROJ 16/8, N_E 1d/2d, N_EP 2d, N_EXP 4, BKV 16/32 |
| 512×256 | 0.814 (11) | 0.870 (11) | +0 | 0.44 | P_PROJ 16/8, N_E 1d/2d, N_EP 2d, N_EXP 4, BKV 16/32 |
| 1024×256 | 1.000 (12) | 1.000 (12) | +0 | 0.41 | P_PROJ 16/8, N_E 1d/2d, N_EP 2d, N_EXP 4, BKV 16/32 |

**Latency, area and EDAP:**

| config | fixed: G_D (front) | latency-optimal: G_D (front) | cycles ÷ fixed | fixed: G_A | area-optimal: G_A | area ÷ fixed | fixed: G_EDAP (rank) | EDAP-optimal: G_EDAP (rank) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128×64 | 0.440 (front) | 0.719 (front) | 0.60 | 0.449 | 0.478 | 0.98 | 0.137 (1) | 0.216 (1) |
| 256×64 | 0.497 | 0.729 | 0.54 | 0.457 | 0.484 | 0.97 | 0.164 (2) | 0.234 (2) |
| 512×64 | 0.572 | 0.778 | 0.50 | 0.473 | 0.497 | 0.97 | 0.214 (3) | 0.288 (3) |
| 128×128 | 0.507 | 0.773 | 0.56 | 0.593 | 0.622 | 0.96 | 0.215 (4) | 0.328 (4) |
| 1024×64 | 0.630 | 0.807 | 0.47 | 0.497 | 0.517 | 0.95 | 0.291 (6) | 0.376 (6) |
| 256×128 | 0.545 | 0.773 | 0.53 | 0.601 | 0.628 | 0.96 | 0.243 (5) | 0.344 (5) |
| 512×128 | 0.655 | 0.832 | 0.47 | 0.624 | 0.646 | 0.95 | 0.334 (7) | 0.426 (7) |
| 1024×128 | 0.694 | 0.815 | 0.43 | 0.673 | 0.687 | 0.94 | 0.448 (8) | 0.535 (8) |
| 128×256 | 0.702 | 0.938 | 0.49 | 0.677 | 0.691 | 0.94 | 0.525 (10) | 0.749 (11) |
| 256×256 | 0.760 | 0.875 | 0.43 | 0.872 | 0.896 | 0.94 | 0.521 (9) | 0.615 (9) |
| 512×256 | 0.870 | 0.903 | 0.38 | 0.920 | 0.935 | 0.93 | 0.691 (11) | 0.738 (10) |
| 1024×256 | 0.984 | 0.951 | 0.36 | 1.000 | 1.000 | 0.92 | 1.000 (12) | 1.000 (12) |

Read:
- **The ranking does not move.** The ADP order is the fixed mapping's with one swap, 128×128 and 1024×64 at places
  4 and 5 (0.465 against 0.401, both far from the front); the EDAP order has one swap at places 10 and 11. The
  latency-vs-area front is 128×64 alone in both. The cross-family conclusion (`experiment.md` §8) stands.
- **The numbers move a lot.** The ADP-optimal mapping cuts every config's ADP to 0.41–0.65 of the fixed one, the
  cycles to 0.36–0.60, for 2–8 % more area; the gain grows with the array (1024×256 0.41, 128×64 0.65), so the
  spread between configs compresses (G_ADP 0.20 → 0.31 at the best, 1.00 at the worst) without reordering.
- **What the optimum is:** $P_{proj}$ 4–16 (the balance rule of §5, higher where the staging cap allows), $N_E = 2d$
  ($d$ at $C = 256$ on some heads, where the stream, not QK, binds), $N_{EP} = 2d$ or $4d$, $N_{EXP} = 4$, $B_{kv} = 16$.
  The core sits at the rate rules; the projections are the change.
- **The area-optimal mapping saves 2–8 %** (the crossbar part only) and costs 15–30 % cycles: never worth it on ADP.

So within the modeled grid the mapping is not a DSE knob for the *ranking*: the DSE's conclusions about which array
wins are robust to it. It is a knob for the absolute latency and ADP of every config (×0.4–0.65) and for the gaps
between them. The one assumption to close is the soft logic and Fmax of a 4–16-copy build; if those grow alike on
every config the ranking holds, and only a VTR run of the optimal mapping on two or three configs can say.

## 7 What it means for the DSE's conclusions

1. **Energy and crossbar area are mapping-invariant** (energy to +4.6 / -2.3 % through $B_{kv}$ alone; the crossbar count
   linear in the farm sizes, +22 % per farm doubling, +2 % per projection copy). The energy ranking of the configs and
   the energy term of EDAP are properties of $(R, C)$ and the workload, not of the mapping.
2. **The fixed mapping is balanced at the stream floor.** Each core stage is within 20 % of the floor, so the mapping
   cannot be made much faster by adding crossbars to the core ($N_E$ $4d$: 1.00; $N_{EXP}$ 4: at most −5 %), and it is
   not over-provisioned: removing them costs +1 to +32 % (median +8 %) ($N_E = d$), +2 to +28 % (median +13 %) ($N_{EXP} = 1$), +2 to +61 % (median +12 %) (half the o-carry groups).
3. **The one large lever is $P_{proj}$**, which only the projection stream feels: -64 to -15 % (median -35 %) cycles at 4 copies for
   18–63 extra crossbars. It acts unequally on the configs, because the stream's share of the path grows with $R$;
   the per-config table in §4 says by how much the latency ranking at a fixed $C$ compresses. Its soft-logic cost
   (writers, staging) is the open question, and the candidate for the next VTR runs together with $N_{EXP} = 4$
   where the contracts allow it.
4. **Three constants dominate everything and are not mapping choices:** the 40-bit port (every rate), the ACAM price
   per column (the energy), and the drain of all $C$ columns (the latency at $C = 256$). A study that wants to move
   the attention front has to move one of them in the RTL.
