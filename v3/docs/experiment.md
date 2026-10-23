# v3 experiment: crossbar-size DSE on NL-DPE (FC + attention)

Status: 2026-10-14, draft for review. The user decided on 2026-10-14:
- the folder is `v3/`;
- $n$ is fixed at build time;
- an attention row is one head: the Q / K / V projections and the attention core. It is built from general modules
  (GEMM, tiled DIMM, online softmax; `architecture.md`) and has no output projection;
- `dse_v2` step 7 and `simnext` are frozen;
- the mapping search of §4 is adopted, including the $P_{proj}$ RTL work;
- every crossbar pass uses only the rows and columns in use, at most $I = \min(R, C)$ for a conversion;
- the tiled DIMM is cut into slices of 16 crossbars. This is fixed, not searched.

Writing this document needed no run. Every number is quoted from a named earlier run or derived from the formulas
here; model-only figures are marked as such. Module structure: `architecture.md`. Earlier parameter measurements,
made on an earlier layer model: `mapping_tiling.md`.

## 1 Question

Which crossbar size $(R, C)$ makes the best NL-DPE FPGA for FC layers and for textbook attention, when every
(config, workload) runs on its best mapping? The metric is $\mathrm{EDAP} = E \cdot D \cdot A$ per (config,
workload), ranked per family (§5).

## 2 Design space

$$
\text{point} = (R, C) \times \text{workload}, \qquad \text{row} = \text{point} \times \text{picked mapping}
$$

The config and the workload are given (§2.1, §3). The mapping is the only free choice inside a point (§4). Shape,
config and mapping are all build-time parameters, so each row is one VTR circuit.

### 2.1 Configs

- **8 configs (user, 2026-10-22):** $R \in \{128, 256, 512\}$, $C \in \{64, 128, 256\}$, $C \le R$, so
  $I = \min(R, C) = C$. The 1024-row arrays and 128×256 left with the row-parallel tile rule (§5.1): every config has
  $C \le R$, and the COFFE local-crossbar sweep stops at $R = 512$.
  - 64×64 is the size of the RTL's functional validation (GATE 2) only. Its tile is in the area model (§5.1), but it
    has no DSE row.
  - 256×128 is the VTR smoke point.
  - Until 2026-10-22: $R \in \{128, 256, 512, 1024\} \times C \in \{64, 128, 256\}$, 12 configs (the results of
    §8–§9 as of 2026-10-16).
- Arch XMLs: `vtr/arch/nl_dpe_{R}x{C}_auto.xml`, v2's, with the 40-bit ports and the legacy tiles; VTR uses them until
  the RTL gets the row-parallel ports. The row-parallel XMLs are staged in `vtr/arch_rowpar/` (`vtr/gen_arch.py`).
  The auto layout grows the grid to fit the circuit. The area is a resource sum (§5), so grid growth does not enter
  the metric.
- Platform constants: `BUF` = 40 bits, i.e. a crossbar loads 5 B per cycle; `P` = 8. The row-parallel interface
  ($R$ bits in and $C$ bits out per cycle) is priced in the area only so far (§5.1); the cycles still use `BUF` = 40.

### 2.2 Row unit

- **FC**: one GEMM $(K, N)$, batch $M = 1$.
- **Attention**: one head of textbook attention, $O_8 = \mathrm{softmax}(QK^\top 2^{-5})\,V$ (`architecture.md`
  §1–§2). A row contains:
  - the Q / K / V projections: three GEMMs ($K = d_m$, $N = d$), times $P_{proj}$ copies;
  - the attention core: the QK DIMM, the online softmax, the PV DIMM and the softmax accumulator;
  - the tiling controller.

  There is no output projection, residual add, LayerNorm or FFN. $V > 1$ shows up inside an attention row, in the
  projections.

## 3 Workloads

### 3.1 FC (snapshot from the 2026-10-10 run, not re-run)

| shape $(K, N)$ | provenance |
|---|---|
| (512, 128) | Round-1 `fc_512_128`, the BERT-Tiny FFN ($d_{ff}$ 512 × $d_m$ 128) |
| (2048, 256) | Round-1 `fc_2048_256`, a large FFN-scale projection |
| (256, 512) | Round-1 `fc_256_512`, dense output-projection scale |
| (512, 512) | Round-1 `fc_512_512`, a large dense projection |

- Results reused (user, 2026-10-13):
  - 48 VTR rows (seeds {11, 13, 17}, all ok) and 48 priced rows (used extent). Since 2026-10-22 the priced FC rows
    are the 32 on the 8 configs (§2.1);
  - files: v2's `vtr_results.csv`, `dse_v2.csv`, `ranking_fc.json`, kept as locked copies in `simulator/tests/ref/`
    (the regression reference of `testFcRows.py`); the FC results of this experiment are `dse/results/fc_vtr_results.csv`
    (the 48 FC rows of v2's VTR table) and `dse/results/fc_priced_results.csv` (the v3 model's rows on them,
    `dse/fc_rows.py`; 2026-10-17).
- The snapshot stays valid while the FC shapes, `gemm_top`, the prices and the area formula are unchanged. The v3
  model must reproduce its cycles and energy row by row (§6).
- Optional open item: these are the Round-1 FC shapes, not the FFNs of the six attention models.

### 3.2 Attention: six shapes (user, 2026-10-13)

The rules, the reason for each shape, primary and backup sources, rejected candidates and references are in
Appendix A.

| # | model | application | heads | $n$ | $d_m$ | $d$ | $n$ from |
|---|---|---|---|---|---|---|---|
| 1 | ViT-B/16 | image | 12 | 256 | 768 | 64 | ViTA [1]: 256 patches |
| 2 | DeiT-T | image | 3 | 256 | 192 | 64 | HG-PIPE [2], ViTA [1]: 196 patches + class token = 197 → 256 |
| 3 | BERT-base | text | 12 | 128 | 768 | 64 | FQ-BERT [5], multi-FPGA [6]: 128 |
| 4 | BERT-Tiny | text | 2 | 128 | 128 | 64 | the GLUE length of [5, 6] (by analogy) |
| 5 | JetFormer-150P | jet tagging | 2 | 32 | 64 | 32 | JetFormer [14], FPGA build [15]: 30 particles + class token = 31 → 32 |
| 6 | JetFormer-JetClass | jet tagging | 8 | 128 | 128 | 16 | JetFormer [14] code: 128 particles + class token = 129 → 128 (GPU only, flagged) |

A row computes one head; the "heads" column only describes the model.

Derived sizes per shape, from $n$, $d$, $d_m$ and the config:
- $S = \min(n, I)$, $R_b = n/S$, $B = n/B_{kv}$;
- $V = \lceil d_m/R \rceil$; $H = \lceil d/C \rceil = 1$ on every row.

| shape | $R_b$ at $I$ = 64 / 128 / 256 | $B$ | $V$ at $R$ = 128 / 256 / 512 / 1024 |
|---|---|---|---|
| ViT-B/16 | 4 / 2 / 1 | 16 | 6 / 3 / 2 / 1 |
| DeiT-T | 4 / 2 / 1 | 16 | 2 / 1 / 1 / 1 |
| BERT-base | 2 / 1 / 1 | 8 | 6 / 3 / 2 / 1 |
| BERT-Tiny | 2 / 1 / 1 | 8 | 1 |
| JetFormer-150P | 1 ($S$ = 32 on every config) | 2 | 1 |
| JetFormer-JetClass | 2 / 1 / 1 | 8 | 1 |

This gives 72 attention points, 21 of them with $V > 1$: ViT-B/16 and BERT-base 9 each, DeiT-T 3. The domain checks
all hold: $d \ge B_{kv} = 16$, $8 \le S \le I$, and $B_{kv} \le \min(n, I)$.

## 4 Mapping

**Definition.** A mapping is the set of values of the head top's parameters that are neither shape
(`NSEQ, DM, D`) nor config (`R, C, BUF, P`). The top `att_head` (`architecture.md` §4.7) has the parameters
`NSEQ, DM, D, BKV, R, C, BUF, P, N_EXP, N_E, N_EP, N_OC, P_PROJ, X_S`. Whatever the modules derive is not a choice.

**Decision (user, 2026-10-15): one fixed mapping for every attention shape, no search.** Each parameter is set by a
rate rule: a stage keeps up with its producer and its consumer, with as few resources as that needs. There is no
candidate sweep, no pick rule and no calibration VTR. The functional verification already done covers these builds:
the model is pinned to the RTL by the gates, and every build is inside every module's domain.

### 4.1 FC

There is no free parameter:
- $V = \lceil K/R \rceil$ row tiles and $H = \lceil N/C \rceil$ column tiles;
- one copy;
- used pass extent (P29).

So the FC mapping space has exactly one point, and that point is the snapshot.

### 4.2 Attention: the fixed mapping

| parameter | value | rule | the alternatives (head model, 72 rows, 2026-10-15) |
|---|---|---|---|
| $P_{proj}$ | 1 | The configuration the head's VTR smoke synthesized (user, 2026-10-15: no further VTR runs before the DSE). | 2 or 4 copies shorten the front phase (at $N_E = d$, 1 copy cost +41–48 % cycles against 4), but they were never synthesized |
| $N_E$ (QK farm) | $2d$ | The same: the synthesized configuration. | $d$: $d$ fewer crossbars and half the QK slices, for more cycles (1–15 % at $P_{proj} = 4$) |
| $N_{EP}$ (PV farm) | $2d$ | With $d$ the accumulator would need $N_{OC} \le N_{EP}/16$, below the floor rule, and the PV $\log P$ staging overruns at $I = 64$. | — |
| $N_{EXP}$ | 2 | The smallest power of two whose EXP stage keeps up with the score stream. | 4: −6 % cycles (median), but outside the RTL on 20 of the 72 rows (the PV $\log P$ write rate) |
| $N_{OC}$ | the floor rule, 2–8 | The smallest group that keeps up with the score stream (`architecture.md` §5.2). | — |
| $B_{kv}$ | 16 | The only value legal on every row. | 8 or 32: 0 to +8 % cycles |
| $X_S$ | 16 | User 2026-10-14; it bounds fan-out. | 8 or 32: cycles change by at most 1.5 %; the effect is on Fmax and CLB only |
| $B_q$ | $S$ | One query block per run. | — |

Crossbars per row ($H = 1$):

$$
N_{xbar} = \underbrace{3V + 3}_{\text{projections}} + \underbrace{2d + 2d + (2 + 2) + 2 N_{OC}}_{\text{QK, PV, softmax, accumulator}}
$$

For example BERT-Tiny at 128×128 ($V = 1$, $d = 64$, $N_{OC} = 6$): 6 + 128 + 128 + 4 + 12 = 278, the count VTR
placed. `tools/check_head_params.py` evaluates `att_head.v` for all 48 builds (72 until 2026-10-22): every instance's parameters and the
crossbar count equal the simulator's (2026-10-15, ALL PASS, with a negative control).

**How sensitive the metrics are to these knobs**, one at a time over the 72 builds of the 12 configs (2026-10-19; not yet re-run on the 8 configs and the row-parallel tiles), with the closed forms behind it:
`docs/mapping_sensitivity.md` (`dse/mapping_sensitivity.py`). In short: energy and crossbar area are
mapping-invariant ($B_{kv}$ apart, $\pm 5\,\%$), the fixed mapping sits at the stream floor, and $P_{proj}$ is the
one knob with a large effect on the cycles (−35 % median at 4 copies), larger on the large arrays.

**Outside the mapping (not built):** $S < I$ (each run would re-stream K and V); a row split $V' > V$ or Q / K / V
column packing; more than one head per row or an output projection; runtime $n$ (build-time $n$, 2026-10-14).

### 4.3 The rows

48 builds: the six shapes × the 8 configs (72 on the 12 configs until 2026-10-22), each one `att_head` build with
the mapping above. Every one is inside every module's domain (head model, 2026-10-15; rechecked 2026-10-22 on the 8
configs and 64×64).

### 4.4 Latency model (for reading the results)

**The bit-plane pass (2026-10-22, model only; §8 still carries the 40-bit cycles).** Every crossbar pass takes 10
cycles back to back on every array, and every stream moves a window in 8 cycles. Measured on the model's 48 builds
[model, 2026-10-22]:
- front: the X stream, one token per pass period per copy, so $10 n / P_{proj}$ cycles ($n = 128$: 1,279);
- core: $R_b B$ key blocks. A block needs at least the score stream, $8 S B_{kv} / I$ cycles (one plane per cycle into
  the online softmax; 128 at $S = I$). On most builds the E5 credit loop sets the block period instead: the PV DIMM
  holds two $\log P$ blocks, so a block's tile waits for the PV job of the block two before (171–175 cycles per
  block at $S = I$, $d = 64$). A third slot would cut 0–16.2 % of the cycles (median 8.4 %; `LEDGER.md` Needs you 26);
- the array's rows no longer change the time, except through the GEMM tree's depth (one cycle per level): $T$
  depends on $I = C$. Example: ViT-B/16 takes 12,481 / 7,347 / 5,708 cycles at $C$ = 64 / 128 / 256, whatever $R$.

**Until 2026-10-22 (the 40-bit pass).** This first-order model explained the cycles (checked on the 72 rows,
2026-10-15):

$$
T \approx \underbrace{\frac{S\, T_p}{P_{proj}}}_{\text{front}} + \underbrace{R_b B \cdot \max\!\Big(\Big\lceil \frac{S B_{kv}}{5} \Big\rceil,\ T_{QK},\ T_{PV},\ T_{EXP},\ T_{ACC}\Big)}_{\text{core}} + \text{fill}, \qquad T_p = \max\!\Big(\Big\lceil \frac{\min(d_m, R)}{5} \Big\rceil + 8,\ \Big\lceil \frac{8C}{40} \Big\rceil + 1\Big)
$$

- front: the first run's $Q$ through the projections; exact to within 80–140 cycles. $T_p$ is a projection copy's
  row period: the X load of $\min(d_m, R)$ rows, or the GEMM's drain of all $C$ columns (2026-10-18), whichever is
  longer; the drain wins where $d_m$ is small and $C$ large (JetFormer-150P at $C \ge 128$, every shape at
  $C = 256$ with $d_m \le 128$).
- core: $R_b B$ key blocks, each at least the score stream $\lceil S B_{kv}/5 \rceil$ (one 40-bit port into the online
  softmax). The mapping above keeps every stage at that floor on most rows: the measured block period is 1.02× the
  estimate (median). Summed, the core's floor is about $n^2/5$ cycles, whatever the crossbar size.
- With $P_{proj} = 4$ the front was 3–45 % of the cycles (median 15 %) and the core 30–92 % (median 67 %); at the
  fixed $P_{proj} = 1$ the front is about 4 times longer.

## 5 Metrics

| quantity | definition |
|---|---|
| cycles $T$ | The v3 model: the composition of the module models (`architecture.md` §6). Each module is pinned to its RTL at its gate, and the head's events equal the RTL on 64×64 (gate H, `architecture.md` §7). Compute cycles, cold, weights preloaded (user, 2026-10-15): $T = t_{done} - t_{start}$, the head's `done` in the model and in the RTL. `done` comes two cycles after the last run's last $O_8$ plane leaves the accumulator on E9 ($t_{o8} + 2$). There is no O read-out: the head outputs the final $O_8$ values as they are computed (`o_v`, `o_d`: one plane window of $N_{OC} I$ values per wave) and the O projection reads them (user, 2026-10-15: the old read-out was set by the 40-bit port, not by the crossbars or the mapping; `acc_spec.md` §10 item 7). **Since 2026-10-22** the models follow the bit-plane pass (`ip/xbar/spec.md`), model-only until the RTL pass pins them again; the results of §8 still carry the 40-bit cycles until they are regenerated (`LEDGER.md`). |
| energy $E$ | Frozen per-operation prices (`cost.py` + `Platform("nl_dpe", rows=R, cols=C)`), counted per module: crossbar passes, CLB adds and compares, memory bytes. **The pass price (user, 2026-10-17):** the whole array fires on every pass, so the $R \times C$ cells, the ACAM on every column, the output buffer and the output drain are charged in full; only the rows in use are read, so the input-buffer bytes and the input port words follow `ROWS_USED` (P29). Until that date every pass was priced as the full array on both sides, which overcharged the input side of every conversion pass on $R > C$ arrays and of every GEMM pass with $K < R$ (attention energy fell by 6 % at 128×64 and 41 % at 1024×64; `simulator/platform/prices.py`). The DIMM counts one pairing add per product and one compare per narrowed output; FIFO words count as memory bytes, written once and read once (v2's ledger, 2026-10-15). CLB selects are not priced. Weight programming is excluded (setup). |
| area $A$ | $(clb + \sum_{wc} W H + 2\, mem) \times 66{,}117$ MWTA. The counts come from VPR (the largest over the seeds); each used cell is priced at the routing-aware CLB tile (2239 µm²). $W H$ is the crossbar tile of §5.1. The simulator predicts $mem$ and the crossbar count; every build's VPR counts must equal them (a mismatch stops the build for a look). |
| Fmax | VTR, the mean over seeds {11, 13, 17}. Route channel width 300. SDC = `clock.sdc` (I/O-excluding). |
| latency | $D[\mathrm{ns}] = T \cdot 1000 / F_{max}[\mathrm{MHz}]$ |
| EDAP | $E[\mathrm{pJ}] \cdot D[\mathrm{ns}] \cdot A[\mathrm{mm}^2]$ per row |
| notation | One for every metric (user, 2026-10-17): per workload, $s_X(c) = X(c) / \max_{c'} X(c')$, so the worst config scores 1 and lower is better; $G_X$ = the geometric mean of $s_X$ over the family's workloads, for $X \in \{A, D, E, ADP = A \cdot D, EDAP = E \cdot D \cdot A\}$. Every config of a workload is divided by the same number, so the ranks and the Pareto fronts do not depend on the normalizer; the cross-family distance $d$ does. Until 2026-10-17 the EDAP view used best ÷ EDAP (higher is better) and measured $d$ from $(1 - G_{FC}, 1 - G_{ATT})$. |
| ranking | **Latency vs area (user, 2026-10-16), replacing the EDAP ranking:** per family, each config's point $(G_A, G_D)$ and the Pareto front (no other config lower on both). Across the families (user, the same day): the point $(G_{ADP}^{FC}, G_{ADP}^{ATT})$, the Pareto front of that plane and circles of equal distance $d = \sqrt{(G_{ADP}^{FC})^2 + (G_{ADP}^{ATT})^2}$ from the origin (`dse/rank.py`, `adp/latency_area.json`). The EDAP view stays beside it (`edap/ranking_*.json`): rank by $G_{EDAP}$; across the families the point $(G_{EDAP}^{FC}, G_{EDAP}^{ATT})$ and its $d$. |
| reported beside | $G_E$ (beside the fronts, not part of them); the crossbars; the share of the cycles spent streaming the projections (§9). |

### 5.1 The crossbar tile (user, 2026-10-22)

Each crossbar gets one data pin per row in and one per column out, so it can load $R$ bits and drain $C$ bits per
cycle. It also keeps today's 7 control inputs and 5 status outputs. Its tile is sized for those pins and for its area.
Every cell is charged at the routing-aware CLB tile, 2239 µm², which includes the cell's switch box and connection
boxes, so the block's logic fills whole cells:

$$
H = \Big\lceil \frac{\max(R + 7,\ C + 5)}{60} \Big\rceil, \qquad
W = \Big\lceil \frac{A_{logic} + A_{xbar}}{2239\,H} \Big\rceil, \qquad
A_{tile} = W H \times 66{,}117\ \text{MWTA}
$$

- **60 pins per tile row:** the CLB's 60 inputs per tile. In the staged XMLs, VPR spreads the pins evenly over the
  tile's sides (`pinlocations pattern="spread"`), with the CLB's Fc (in 0.15, out 0.10, clock 0) and switch blocks at
  every location.
- **$A_{logic}$:** the formula of `nl_dpe/area_power.py`: the cells, the ACAM with its XOR, two buffers of
  $\max(R, C)$ entries, and the DACs ($4C$).
- **$A_{xbar}$:** the local input crossbar for $R + 7$ inputs, from COFFE's row-parallel sweep
  (`coffe_flow_22nm/sweep_summary_full.csv`, column `total_um2`; COFFE `ecde3ef`, `-a 1 -d 2 -hi 100`, TensorSlices
  22 nm constants). It is 221.17 / 679.79 / 2,080.09 / 7,269.62 µm² at $R$ = 64 / 128 / 256 / 512, and depends on $R$
  only.
- **Known approximation:** COFFE's constants assume channel width 320; VTR routes at 300.
- **Constants:** `simulator/platform/nl_dpe_tile.json`; code: `simulator/platform/area.py` `footprint`.

| array | logic + crossbar (µm²) | $H$ | $W$ | cells | used | legacy cells |
|---|---|---|---|---|---|---|
| 64×64 (validation only) | 11,521 | 2 | 3 | 6 | 86 % | — |
| 128×64 | 12,878 | 3 | 2 | 6 | 96 % | 5 |
| 128×128 | 24,722 | 3 | 4 | 12 | 92 % | 11 |
| 256×64 | 16,074 | 5 | 2 | 10 | 72 % | 6 |
| 256×128 | 29,360 | 5 | 3 | 15 | 87 % | 12 |
| 256×256 | 55,931 | 5 | 5 | 25 | 99.9 % | 24 |
| 512×64 | 24,855 | 9 | 2 | 18 | 62 % | 8 |
| 512×128 | 41,025 | 9 | 3 | 27 | 68 % | 15 |
| 512×256 | 73,363 | 9 | 4 | 36 | 91 % | 30 |

**Why the tall, narrow arrays grow most.**
- The pins set the height.
- The area then needs just over a whole number of columns: 1.23 at 512×64, 2.04 at 512×128, 1.44 at 256×64.
- So the last column is mostly empty.
- At $R = 512$ the local crossbar adds 41 % to the 512×64 logic, because it grows about as $R^2$.

At 512×64, the local crossbar adds 4 of the 10 extra cells, and the empty column adds the other 6.

**Legacy rule** (until 2026-10-22: the tiles of v2's arch XMLs).
- The tile was the logic alone, rounded to the nearest cell, with no local crossbar. Some tiles were therefore smaller
  than their logic: 128×64 had 5 cells, 11,195 µm², for 12,198 µm² of logic.
- The regression tests against v2 keep this rule (`area.footprint(..., rule="legacy")`).

## 6 Procedure

The module work and its gates are in `architecture.md` §7–§8. The experiment's own steps:

1. **FC:** the v3 model reproduces the 48 snapshot rows (cycles and energy). `gemm_top` and the prices are unchanged.
2. **Attention, model ↔ RTL:**
   - the module gates and the head gate of `architecture.md` §7, on 64×64 with small corpora (a full iverilog launch
     needs your approval);
   - one $R \ne C$ case for the softmax extent (your approval);
   - values = `attention_ref.head_stages`, stage `O8`.
3. **The 72 builds (§4.3):**
   - the model's compute cycles, energy, crossbar count and memory-block prediction for each build (done 2026-10-15:
     `dse/attn_rows.py`, then `att_rows.py`);
   - VTR, on your go: the launch plan below;
   - pricing: area, latency and EDAP per build from the VTR counts (`dse/attn_rows.py --vtr`; the FC rows the same way
     with `dse/fc_rows.py --vtr`).
4. **Figures**, per family (FC: the 48 snapshot rows; attention: the 72 builds), as v2's `rank_dse.py`, `plot_dse.py`
   and `plot_pareto_family.py` (matplotlib, the miniconda Python):
   - **geomean ranking** (§5 notation, since 2026-10-17): rank by $G_{EDAP}$, lower is better. One figure with four
     panels: the R × C map of $G_{EDAP}$; the per-workload EDAP scores in rank order with $G_{EDAP}$; $G_{EDAP}$ per
     config; $G_E$, $G_D$ and $G_A$. The raw data only (you, 2026-10-16): each build's Fmax is the mean of its three
     seeds; no seed resampling;
   - **Pareto front:** x = $G_D$, y = $G_E \cdot G_A$, with iso-EDAP lines (slope −1 in log-log, so their order is
     the ranking), the 2-cost front and the 3-cost (E, D, A) front.
   - **Done 2026-10-16** (your go). First as the EDAP ranking above: `dse/rank.py` (ported from v2's `rank_dse.py`;
     the FC ranking reproduces v2's to $4 \times 10^{-16}$), `ranking_{fc,attention,cross}.json`,
     `figures/dse_{fc,attention,pareto,cross}.{png,pdf}` (attention #1 128×64, FC #1 512×128; $G_{EDAP}$ = 0.150 and
     0.152 in the §5 notation, $G$ = 1.000 and 0.829 in the old best ÷ EDAP one). Then, after your decisions the same day, as latency vs area (§5 ranking row): `latency_area.json`,
     `figures/dse_la.{png,pdf}` (three panels), `dse_la_{fc,attention}.png`, `dse_adp_cross.png`, and the results page
     (`dse/results_page.py`). The final results are in §8; why the two families differ, in §9 (`dse/breakdown.py`,
     `breakdown.json`). The 16 builds of the $d = 64$ shapes at $C = 64$ are priced from VPR's memory count (5–8
     blocks above the prediction; you).

**Launch plan (VTR of the 72 builds; nothing runs before your go):**

| item | plan |
|---|---|
| what | the 72 builds, the final RTL (fixed DIMM, no O reader), the fixed mapping from `head_params`; the smokes' VTR set-up (VTR twins + the smoke's parameter wrapper, the config's arch XML, channel width 300, `clock.sdc`); seeds {11, 13, 17} |
| driver | `dse/vtr_dse.py` (in `vtr/` until 2026-10-17; `--self-test`: fake tools, ALL PASS; 6 of 6 injected bugs caught) |
| per build | circuit (its sha256 names the run folder) → elaboration check (Yosys: the top, crossbars = the simulator's, memory list = the simulator's, legal memory shapes; a failure stops the build; Yosys's result is cached, the check is recomputed on every run) → one synthesis (Parmys + ABC, `-end abc`) → VPR per seed on that netlist (byte-identical to the full flow's VPR input, checked on the accumulator and head smokes) → checks per seed (flow OK, Fmax > 0, crossbars and memory blocks = the simulator's) |
| result | per build: Fmax = the mean of the passing seeds, each seed's Fmax kept in the CSV; CLB, memory, crossbars, I/O = the largest; notes |
| workers | 36, memory-guarded. The head smoke peaked at 0.68 GB (Yosys elaboration), 3.0 GB (Parmys, 2 min), 1.0 GB (ABC, 3.6 h) and 8.1 GB (VPR, 22 min); 125 GB in the machine. Each tool process reserves its expected peak: elaboration 2, Parmys 6, ABC 1.5, VPR 12 GiB (about 1.5× the smoke; v2's VPR peaks grew with the grid, 2.4–8.2 GiB), raised during the run to 1.25× the largest peak measured. A process starts when the reserves of the running ones plus its own fit in the memory available at the start less 16 GiB, and the machine has its reserve + 16 GiB free. Syntheses start at least 60 s apart and never while a VPR waits; at most 12 VPRs at once; per-process cap 48 GiB (v2's, via `prlimit`); no timeout |
| order | the 72 elaboration checks first; then the syntheses, largest first (the simulator's memory bits); a waiting VPR seed before a waiting synthesis |
| time | about 350 CPU-hours [estimate]. ABC grows about with the square of the netlist size (43 runs of v2 and v3; the driver test's 53 MB netlist: 52 min; the head smoke's 99 MB: 3.6 h), so the $I = 256$ builds may take 11–16 h each if their netlists are about 1.6× the smoke's, as in v2; 216 VPR runs of 7–25 min. Wall about 12–17 h, set by the largest build's ABC [estimate] |
| storage | run folders `work/vtr/dse/<build>_<hash>/` (circuit, elaboration, `synth/`, `s11/`, `s13/`, `s17/`); results `dse/results/attn_vtr_results.csv` (`vtr_att_72.csv` until 2026-10-17); log `work/dse/vtr_dse.log` (one timestamped line per event). About 2–6 GB per build, 0.3–0.5 TB in all [estimate], 1.2 TB free; checked at start |
| resume | the same command skips every finished step of the same circuit |
| driver test | the elaboration check of all 72 builds (Yosys only, about 5 min at 36); one small build (JetFormer-150P, 128×128) end to end with its 3 seeds; the same command again (every step skipped, the CSV row and folders in place). Elaboration result (2026-10-15): crossbars equal on all 72; the memory lists found two simulator bugs (the DIMM's two bank kinds of one shape counted once, and depths not rounded to the RTL's address fields; fixed, 18 predictions at $I = 256$ rose by 14–56 blocks) and one RTL issue: with $V > 1$ (21 builds) `gemm_top`'s reduction `tree` array is inferred as a many-port memory (24 × 32 bits with 5 read / 12 write ports at $V = 2$, up to 128 × 32 with 149 / 128 at $V = 6$), which the simulator does not have. Fixed the same day (you): the tree is one wide register in both twins; the 64×64 head check with $V = 3$ passed (every event) and the elaboration check passes on all 72 builds |
| then | pricing (step 3), figures (step 4) |

## 7 Open (for your review)

1. (resolved 2026-10-15: $N_{EP} = 2d$ in the fixed mapping, §4.2.)
2. (dropped 2026-10-15: the calibration VTR; there is no pick rule.)
3. (resolved 2026-10-15: VTR on the 72 builds.)
4. **FC shapes** (optional, carried): Round-1 shapes vs the six models' FFNs.
5. (done 2026-10-15: the $R \ne C$ softmax case, 1/1.)
6. **The simulator for sensitivity studies** (your idea, 2026-10-16): the assumptions of §9.6 are fixed in the code
   today, and CLB count and Fmax exist only from VTR. Proposal in LEDGER Needs you 19.

## 8 Results (2026-10-22: the row-parallel tiles, 8 configs)

48 attention builds (6 shapes × 8 configs, VTR seeds 11, 13, 17) and 32 FC rows (the v2 snapshot's 4 shapes × 8
configs), priced with the row-parallel crossbar tile of §5.1.

**The cycles still use the 40-bit ports** (`BUF` = 40). These rows pay for the row-parallel tile's area without its
bandwidth, so they are pessimistic for the larger arrays. They stay so until two things are done: the timing pass (the
10-cycle pass and the bit-plane streams), and a VTR run with the row-parallel XMLs. The CLB, memory and Fmax are those
of the 2026-10-15/16 VTR runs.

Every score is the config's value ÷ the worst config's value on the workload, geometric mean over the family's
workloads: lower is better, and the worst config is 1. The $G$ values are relative to the worst of this run's 8
configs, so they do not compare with the 2026-10-16 values. Fmax is the mean of the three seeds.

**Latency vs area per family** (the Pareto fronts; $G_E$ beside):

| family | front config | $G_A$ | $G_D$ | $G_E$ |
|---|---|---|---|---|
| FC | 512×256 | 0.464 | 0.992 | 0.366 |
| FC | 256×256 | 0.542 | 0.743 | 0.538 |
| FC | 256×128 | 0.560 | 0.630 | 0.500 |
| FC | 256×64 | 0.777 | 0.597 | 0.576 |
| FC | 128×128 | 0.858 | 0.507 | 0.924 |
| FC | 128×64 | 0.862 | 0.431 | 1.000 |
| attention | 128×64 | 0.472 | 0.497 | 0.802 |

- **FC:** 6 of the 8 configs are on the front, from the smallest area (512×256) to the lowest latency (128×64).
  - 512×128, the smallest area of the 2026-10-16 front, drops off it. Its tile grew from 15 to 27 cells and is 68 %
    used (§5.1), so 256×256 now beats it on both axes.
  - 512×64 (18 cells, 62 % used) is beaten by 256×64.
- **Attention:** 128×64 has both the smallest area and the lowest latency, so it alone is the front, as on 2026-10-16.

**Across the families: area-delay product** ($x$ = FC, $y$ = attention), ordered by the distance $d$ from the origin:

| # | config | FC $G_{ADP}$ | attention $G_{ADP}$ | $d$ | FC $G_E$ | attention $G_E$ | front |
|---|---|---|---|---|---|---|---|
| 1 | 128×64 | 0.483 | 0.239 | 0.539 | 1.000 | 0.802 | yes |
| 2 | 256×128 | 0.458 | 0.404 | 0.611 | 0.500 | 0.863 | yes |
| 3 | 256×64 | 0.602 | 0.288 | 0.667 | 0.576 | 0.838 | |
| 4 | 128×128 | 0.564 | 0.362 | 0.670 | 0.924 | 0.828 | |
| 5 | 512×128 | 0.663 | 0.561 | 0.869 | 0.348 | 0.946 | |
| 6 | 256×256 | 0.524 | 0.793 | 0.950 | 0.538 | 0.911 | |
| 7 | 512×64 | 0.975 | 0.376 | 1.045 | 0.426 | 0.918 | |
| 8 | 512×256 | 0.597 | 1.000 | 1.165 | 0.366 | 1.000 | |

- **The front** is 256×128 (the best in FC) and 128×64 (the best in attention).
- **256×64 leaves the front.** Its tile grew from 6 to 10 cells (+67 %), against 5 to 6 at 128×64 (+20 %). The cycles
  and Fmax are unchanged, so 128×64 now beats it in both families.
- **128×64 is the closest to the origin**, but it has the highest FC energy ($G_E$ 1.000). 256×128 halves that (0.500)
  at a similar attention energy (0.863 against 0.802).
- **The R = 512 configs** are 5th, 7th and 8th.

**Before** (2026-10-16: 12 configs, the legacy tiles; in git and in `work/snapshots/results_12cfg_2026-10-22/`):
- the FC front was 512×128 → 256×128 → 256×64 → 128×64;
- the attention front was 128×64 alone;
- the cross front was 256×128, 256×64 and 128×64, with 128×64 and 256×64 tied in $d$ and 256×64 the balanced choice.

Files: `dse/results/adp/latency_area.json`, `dse/results/adp/dse_la.pdf` (the three panels for a paper), and the
results page `dse/results/adp/dse_results.html`. The 2026-10-16 version of the page is
https://claude.ai/artifact/KiVh2rqfLT4ifXSiCMxbeR, not yet republished.

## 9 Why attention and FC differ

**In short.** FC rewards large crossbars: a larger array holds more weights per tile, so a layer needs fewer tiles and
less area, at the cost of longer passes. Attention rewards small crossbars: its crossbar count is set by the mapping,
not by the array size, and each crossbar works on an $I \times I$ window ($I = \min(R, C)$). A larger array adds
area, energy per pass and soft logic, but no useful work. The counts below are means over each family's workloads;
the area and energy shares are the splits of `dse/breakdown.py` (attention: the geometric mean of each share over
its shapes; FC: the share of the total summed over its shapes), all in `dse/results/breakdown.json`.

### 9.1 Architecture: what the crossbars do

- **FC** is a weight-stationary GEMV ($M = 1$): the crossbars store the weight matrix $W$ ($K \times N$), in
  $V \cdot H = \lceil K/R \rceil \lceil N/C \rceil$ tiles (1 to 64 per build: 30 on average at 128×64, 2 at
  512×256). A larger array stores more of $W$ per tile.
- **Attention** multiplies activations by activations ($QK^T$, then $PV$), so neither operand can stay in a
  crossbar. The tiled DIMM uses the crossbars as element-wise EXP / LOG converters with identity weights, one
  $I$-wide window per pass. The crossbar count follows the head dimension and the fixed mapping (§4.2: $N_E = N_{EP} =
  2d$, ...): 78 crossbars for $d = 16$, 146 for $d = 32$, 278–297 for $d = 64$, nearly the same at every array
  size. Of a conversion crossbar's cells,
  $I^2 / (R C)$ are in use: 100 % at 128×128 and 256×256, 50 % at 128×64, 12.5 % at 512×64.

### 9.2 Area breakdown (VPR counts, the row-parallel tiles)

| config | tile cells | FC: crossbars (mean) | FC: tile / CLB / memory share | attention: crossbars (mean) | attention: tile / CLB / memory share | attention CLBs (mean) |
|---|---|---|---|---|---|---|
| 128×64 | 6 | 30 | 96 / 3 / 1 % | 231 | 11 / 84 / 4 % | 10,441 |
| 128×128 | 12 | 15 | 97 / 2 / 1 % | 228 | 16 / 80 / 3 % | 12,887 |
| 256×64 | 10 | 15 | 92 / 7 / 1 % | 227 | 17 / 79 / 4 % | 10,425 |
| 256×128 | 15 | 8 | 95 / 4 / 1 % | 225 | 20 / 77 / 3 % | 12,892 |
| 256×256 | 25 | 4 | 95 / 4 / 1 % | 225 | 23 / 74 / 2 % | 17,521 |
| 512×64 | 18 | 8 | 88 / 11 / 0 % | 226 | 27 / 70 / 4 % | 10,424 |
| 512×128 | 27 | 4 | 94 / 5 / 0 % | 224 | 30 / 67 / 3 % | 12,869 |
| 512×256 | 36 | 2 | 94 / 5 / 1 % | 224 | 30 / 67 / 2 % | 17,532 |

- **FC area is crossbar tiles** (88–97 %). A tile takes 6 grid cells at 128×64 and 36 at 512×256: 6× the grid area
  for 16× the crossbar cells. So a larger array stores a weight in less area, and FC area falls from 128×64
  ($G_A$ 0.86) to 512×256 (0.46).
  - Not at every step. At $C = 64$ and 128, $R = 512$ takes more area than $R = 256$ ($G_A$ 0.78 → 0.85 and
    0.56 → 0.59).
  - Its 519 input pins make the tile 9 cells tall, and its local input crossbar adds 41 % and 22 % to the logic.
    The area then needs just over one or two columns (1.23 and 2.04), so the last column is mostly empty (§5.1).
  - Under the legacy tiles (2026-10-16), FC area fell at every step.
- **Attention area is soft logic** (CLB 67–84 %). The CLB count follows the datapath width $I$: about 10.4k at
  $C = 64$, 12.9k at $C = 128$, 17.5k at $C = 256$ (VPR's counts, unchanged). On top of it, the crossbar tiles grow
  with the array (11 % of the area at 128×64, 30 % at 512×256). So attention area rises with the array size.

### 9.3 Latency breakdown ($D = T / F_{max}$)

- **FC:** one pass per tile, 52–147 cycles, growing with $R$ and $C$: each pass streams $\min(R, K)$ input rows in and
  $\min(C, N)$ outputs out through the 40-bit port. Fmax 379–451 MHz: the designs are tiny (4–20 CLBs). Several
  builds report the same 450.6 MHz, which suggests the crossbar block's own timing sets it (an inference, not
  checked).
- **Attention:** 12k–20k cycles, in two parts measured in the simulator:
  1. **The projection stream.** Every token passes through the one GEMM copy ($P_{proj} = 1$) at $T_p$ cycles (§4.4:
     the X load of $\min(d_m, R)$ rows, or the drain of all $C$ columns since 2026-10-18), so the projections end at
     $n\,T_p$ (within 28 cycles of the simulator's last projection output on all 48 builds; with the load alone the
     formula was 7 cycles exact before the drain rule). That is a median 56 % of the cycles (36–81 %). It grows with
     $R$ while $d_m > R$: for BERT-base ($d_m = 768$), 4.4k cycles at $R = 128$ and 14.2k at $R = 512$; for
     BERT-Tiny ($d_m = 128$) it stays at 4.4k up to $C = 128$ and reaches 6.8k at $C = 256$, where the drain paces it.
  2. **The core.** Bound by the score stream into the online softmax (one 40-bit port, about $n^2/5$ cycles), which
     does not depend on the array size. A smaller $I$ means smaller query runs ($S = \min(n, I)$) that overlap the
     projection stream better: BERT-Tiny takes 7,752 cycles at 128×64 and 9,405 at 128×128.

  Fmax is 114–131 MHz (config means), set by the 10k–18k-CLB soft logic; it falls as $C$ grows (about 130 MHz at
  $C = 64$, 124 at 128, 114 at 256).

- **The config tables and the critical path** (`dse/config_table.py`, user 2026-10-18; output
  `dse/results/attn_config_table.md` and, per build, `dse/results/critical_path.json`). Every number is the
  geometric mean over the six heads unless said otherwise.
  1. *The chain:* each config relative to a reference (default the table's best ADP): area, cycles, Fmax, latency
     $=$ cycles $/$ Fmax, ADP $=$ area $\times$ latency, energy and EDAP $=$ energy $\times$ ADP. The columns multiply
     exactly, because the geometric mean of a product is the product of the geometric means; the ADP and EDAP columns
     are the ratios of the configs' $G_{ADP}$ / $G_{EDAP}$, so they order the configs as the rankings do.
  2. *What is inside:* per head the crossbar, CLB and BRAM area and their total (mm², VPR counts as in §9.2), the
     energy (nJ, the priced ledger), the cycles on the head's critical path by kind, the head's cycles $T$ and Fmax
     (MHz). The parts are geometric means too, so they come close to their total but do not add up to it exactly.
  3. *The critical path* (`simulator.md` §3.1, traced times): the simulator's times carry their history, every `max`
     keeps the winning input's chain and every added latency is a labeled segment, so a head's `done` carries the
     one chain of waits that set it, from the start to the last output word, and the segments' cycles add up to
     $T$ exactly. A segment is labeled by the module it is in (its role: gemm, qk, sm, pv, acc, ctrl) and by its
     kind: *DPE load* (a pass's input burst, $\lceil 8\,ROWS\_USED / BUF \rceil$ words into a crossbar), *DPE compute*
     (the $P$ bit-slices, the shift-accumulate, the ACAM and the hand-overs), *DPE drain* (the first output word, and
     the drain words a consumer takes), *stream* (the words of a port, or a FIFO read one word per cycle: the 40-bit
     paths between modules), *logic* (every other named latency: trees, registers, decisions, the writers, and the
     start's cycle). Table 3 gives the shares of $T$ by kind and by module (the mean over the heads of each share),
     the per-head tables the same for one config, and the FC table the one-pass rows. A cycle that two inputs set
     together is charged to the first the model names (a tie): the breakdown is exact for the path it follows. Where
     a stream feeds a crossbar directly (the GEMM's output into a DIMM converter), the chain records the crossbar's
     load, not the producer's words; the X bus's words are the projection crossbar's load.
     The results page shows table 3 as stacked bars (`dse/plot_dse.py` `dse_critical_path.png`: attention by kind and by
     module, FC by kind) with the tables beneath (`dse/results_page.py`).

### 9.4 Energy breakdown

- **The pass price (§5; 2026-10-17):** the whole array fires on every pass (the cells and the ACAM at the full
  $R \times C$, the output buffer and the drain), and only the rows in use are read. An attention pass therefore costs
  13.9 pJ at 128×64 and 70.7 pJ at 512×256 on average (the full array would be 14.8 and 74.9), and 17.8 pJ at
  512×64 (24.4 as the full array): a conversion on an $R > C$ array reads $I$ of its $R$ rows.
- **FC:** the DPE passes are 69–82 % of the energy, the CLB adds 6–23 % (the adder tree, so only with $V > 1$), the
  passes' port traffic 7–17 %. The pass count falls with the array size (30 to 2 per build) faster than the price
  rises, so FC energy falls as the array grows: the same direction as its area.
- **Attention:** the DPE passes are 54–63 %, the CLB adds and compares 32–40 %, the port traffic 4–5 %, the memory
  bytes under 1 %. The pass count depends on $I$ only (about 62k at $I = 64$, 31k at 128, 16k at 256), so at a fixed
  $I$ the energy rises with $R$: the cells and the ACAM grow with the array, the work does not.

### 9.5 What follows

The two families pull in opposite directions: FC towards large arrays (area and energy), attention towards small
ones (area, latency and Fmax). The cross-family front therefore sits in the middle: 256×128 and 128×64 since
2026-10-22 (256×128, 256×64 and 128×64 with the legacy tiles).

### 9.6 The modeling assumptions behind the difference (each could be revisited)

1. **The fixed attention mapping** (§4.2). The crossbar count does not follow the array size. A mapping that used the
   extra cells of a large array (several windows packed into one crossbar, or more GEMM copies) would change the
   attention ranking.
2. **The pass price:** the whole array fires on every pass (the cells and the ACAM at the full $R \times C$), and only
   the rows in use are read (2026-10-17; before, both sides were priced as the full array). The fixed part, cells plus
   ACAM, is what makes a large array expensive per pass whatever the window.
3. **One projection copy** ($P_{proj} = 1$). The X stream is serial, 5 bytes per cycle per copy, which makes the
   projections half of the attention cycles.
4. **The 40-bit ports.** They bound both the score stream (the core's $n^2/5$) and the projection load. Since
   2026-10-22 the area already pays for the row-parallel ports ($R$ pins in, $C$ out; §5.1) while the cycles do not
   yet use them: the timing pass will give the larger arrays their bandwidth.
5. **FC is a single GEMV** ($M = 1$, the v2 snapshot, from v2's RTL before the tree fix). Its latency is one pass;
   a batched FC would weigh the weight load differently.
6. **Area** prices every used grid cell at the routing-aware CLB tile; the crossbar tile follows the row-parallel
   rule (§5.1: whole columns of a pin-set height, so tall, narrow arrays leave a column mostly empty); CLB and memory
   counts come from VPR, and Fmax from VTR (the mean of three seeds), both from the runs with v2's arch XMLs.

---

## Appendix A Attention shapes: rules, sources, rejected candidates

Moved from `v2/experiment/dse/plan.md` §3.1, which stays as the source record. The jet-tagging sources [14–19] were
checked on 2026-10-13; [14], [16] and [17] were read in full text, and [14] and [15] also in their code.

### A.1 Selection rules

1. **Textbook attention only**: $\mathrm{softmax}(QK^\top/\sqrt{d})\,V$ over all $n$ tokens. No window, no score
   bias, no mask. User 2026-10-12: the heads go into a complete model, and an unusual attention scheme multiplies the
   implementation effort. A class token counts as an ordinary token.
2. **$n$ taken to a power of two** (user 2026-10-13): $n$ = the power of two nearest to the paper's token count
   (patches or particles, plus the class token where the model has one). The workload is modeled at that $n$
   directly, with no padding and so no key mask.
3. **$n$ = the length a cited FPGA paper runs the model at** (user 2026-10-12), never picked by us.
   - Where no FPGA paper runs the model (BERT-Tiny), $n$ is the GLUE length of the BERT FPGA papers [5, 6], marked
     "by analogy".
   - Where the length appears only in the paper's code (JetFormer-JetClass), it is marked so.
4. **Models run by FPGA transformer accelerators** [1–7, 14, 15]. Flagged exceptions:
   - BERT-Tiny: no FPGA paper.
   - JetFormer-JetClass: from an FPGA paper, but trained and evaluated on GPU only.
5. **One shape per model** (user 2026-10-12): a model listed twice would count twice in the geomean. Two sizes of
   one family count as two models.
6. **Both $V = 1$ and $V > 1$** appear; $V$ lives only in the projections.
7. **Two shapes per application** (user 2026-10-13): image, text, jet tagging at the LHC. Preferably each shape
   differs from the others.
8. **Inside the core's domain**: $d \ge B_{kv} = 16$, $8 \le S = \min(n, I) \le I$, $B_{kv} \le \min(n, I)$.

### A.2 Why each shape (primary source, then backups)

- **ViT-B/16, $n = 256$**
  - Primary: ViTA's own setting on an edge FPGA (Zynq ZC7020) [1]: "We consider an image dimension (H×W) of 256×256,
    thus giving the sequence length (N) as (H.W)/P² = 256." ViTA counts the patches only, no class token.
  - Model: 12 layers, $d_m$ 768, 12 heads, 16×16 patches [10].
- **DeiT-T, $n = 256$** (user 2026-10-13; DeiT-S was the alternative)
  - Primary: HG-PIPE [2] runs DeiT-tiny on ZCU102 (split into four parts) and on VCK190, with sequence length
    T = 196.
  - Backups: ViTA [1] runs DeiT-T at 224×224 ($N = HW/P^2 = 196$). HeatViT [3] runs DeiT-T on ZCU102. The survey's
    model table lists DeiT-tiny [4].
  - 196 patches + 1 class token [9] = 197 tokens → 256.
  - Model: 12 layers, $d_m$ 192, 3 heads, 16×16 patches [9].
- **BERT-base, $n = 128$**
  - Primary: FQ-BERT [5] (ZCU102 / ZCU111, batch 1): "the length of sentence is set to 128".
  - Backups: the multi-FPGA study [6] uses 128 as the GLUE typical size; the survey lists BERT-base designs [4].
  - COBRA [7] runs BERT-base at 512 on KV260 / ZCU102. With one shape per model, 128 wins (two papers).
- **BERT-Tiny, $n = 128$ (by analogy)**
  - Model: Google's compact BERT [12], $d_m$ 128, 2 heads (heads $= d_m/64$).
  - No FPGA paper on it was found, so $n$ is the GLUE length of [5, 6].
  - Kept for $V = 1$ on every config and for continuity with the earlier DSEs (Round-1 `attention_128_64`, the
    committed 2026-10-10 rows).
- **JetFormer-150P, $n = 32$**
  - Primary: JetFormer [14], an encoder-only transformer for LHC jet tagging (from offline analysis to FPGA
    triggers). §5.2.1: "The JetFormer model trained on the HLS4ML 150P dataset uses a compact architecture with 3
    transformer encoder layers, an embedding dimension of 64, and 2 attention heads". Table 2 runs it at 30 / 50 /
    100 / 150 particles.
  - The parameter counts fit one architecture over Table 2 exactly: $101{,}637 + 6\,(n_p+1)^2$ at $n_p$ = 30 / 50 /
    100 / 150. The $(n_p+1)^2$ term is a score BatchNorm that is defined in `transformer/src/layer.py` but commented
    out of `forward`.
  - Backups:
    - the FPGA build of the same configuration, Wojcicki et al. ICFPT 2022 [15]. Its repository's
      `hls/firmware/defines.h` gives `N_PARTICLES 30`, `N_FEATURES 16`, `N_TRANSFORMER_LAYERS 3`, `N_HEADS 2`,
      `N_EMBEDDED_DIM 64`, and the comment "N_TRANSFORMER uses N_PARTICLES + 1 to account for the hidden class
      tokens". The paper text itself was not read.
    - JetFormer's code default `--num_particles 30`.
    - Laatu et al. [16] run single-head MHA at 32 particles on an Alveo U250 (83 ns, Table 1).
  - 30 particles + 1 class token = 31 tokens → 32; $d = 64/2 = 32$.
- **JetFormer-JetClass, $n = 128$ (flagged)**
  - Primary: JetFormer [14] §5.2.1: "For the JetClass dataset, we scale up JetFormer to a larger configuration with
    10 transformer encoder layers, an embedding dimension of 128, and 8 attention heads", so $d = 16$.
  - The particle count is only in the code: `transformer/jetclass_dataset.py` pads every jet to `maxlen=128`
    (`jetclass_train.py`: `num_particles = 128`). The paper says only "zero-padding is applied as in the
    150-particle dataset".
  - 128 + 1 = 129 tokens → 128.
  - Flagged: this size is trained and evaluated on GPU only.
  - Backup for the head size: ParT [17] on JetClass, $d_m$ 128, 8 heads, "a query dimension of 16 for each head".
- **Textbook check of JetFormer** (`transformer/src/layer.py`, `SelfAttention.forward`):
  - $Q, K, V$ come from bias-free linear maps, followed by `softmax(energy / head_dim ** 0.5)`, with no bias and no
    mask;
  - the class token is prepended as an ordinary token (`transformer/src/net.py`).

### A.3 Rejected candidates

| candidate | why not |
|---|---|
| DeiT-S (256, 64, 384) | The second image model is DeiT-T (user 2026-10-13). |
| BERT-Small (128, 64, 512) | It was in the 2026-10-12 set; dropped for two shapes per application (rule 7). It was the weakest text shape (no FPGA paper). |
| Swin-T | Window attention [1], a relative position bias and a shifted-window mask [13] (rule 1). |
| BERT-base at 512 [7], BERT-Tiny at 512 | One shape per model (rule 5). |
| BERT-Mini ($d_m$ 256) | No FPGA paper; $V > 1$ only at $R$ = 128. |
| edge-LLM decode [8] | One query row against the cached keys, a different shape from the core's runs. |
| JetFormer-tiny ($d_m$ 8, 2 heads, $d$ 4) | The FPGA-deployed JetFormer [14] §4.4: $d = 4 < B_{kv} = 16$ (rule 8). |
| JetFormer-150P at 150 particles | One shape per model; 30 particles is the FPGA build [15]. |
| Laatu et al. MHA [16] | The paper text gives no $d_m$. |
| Jiang et al. b-tagging [18, 19] | Hidden size 64 and 2 heads, 9,135 parameters, but no head dimension. |
| ParT [17] | Interaction matrix $U$ added to the pre-softmax scores as a bias (rule 1); no FPGA build. |
| Linformer [16] | $K$, $V$ projected to $k < n$ rows before the softmax (rule 1). |

### A.4 References

1. ViTA: A Vision Transformer Inference Accelerator for Edge Applications (arXiv 2302.09108). Zynq ZC7020, int8,
   150 MHz. §III-A: ViT-B/16 at 256×256 → $N = 256$. Tables III–IV: ViT-B/16, DeiT-S, DeiT-T, Swin-T at 224×224.
   https://shashanknag.github.io/files/camera_ready_paper.pdf
2. HG-PIPE (arXiv 2407.17879). DeiT-tiny / small; ZCU102, VCK190; T = 196. https://arxiv.org/html/2407.17879v2
3. HeatViT (arXiv 2211.08110). DeiT-T / S / B on ZCU102. https://ar5iv.labs.arxiv.org/html/2211.08110
4. Recent Developments in Transformer Inference Deployment on FPGA Platforms: A Survey (arXiv 2609.01212).
   https://arxiv.org/html/2609.01212
5. FQ-BERT (arXiv 2103.02800). ZCU102 / ZCU111, batch 1, sentence length 128. https://arxiv.org/pdf/2103.02800
6. The Feasibility of Implementing Large-Scale Transformers on Multi-FPGA Platforms (arXiv 2404.16158). Length 128.
   https://arxiv.org/pdf/2404.16158
7. COBRA (arXiv 2504.16269). BERT-base at 512, KV260 / ZCU102. https://arxiv.org/pdf/2504.16269
8. TeLLMe v2 (arXiv 2510.15926). Ternary LLM on an edge FPGA. https://arxiv.org/abs/2510.15926
9. DeiT (arXiv 2012.12877). Ti / S / B = $d_m$ 192 / 384 / 768, 3 / 6 / 12 heads, 12 layers, class token.
   https://arxiv.org/abs/2012.12877
10. ViT (arXiv 2010.11929). Base: 12 layers, $d_m$ 768, 12 heads. https://arxiv.org/abs/2010.11929
11. BERT (arXiv 1810.04805). Base: L 12, $d_m$ 768, 12 heads. https://arxiv.org/abs/1810.04805
12. Well-Read Students Learn Better (arXiv 1908.08962). The compact BERTs, heads $= d_m/64$.
    https://arxiv.org/abs/1908.08962
13. Swin Transformer (arXiv 2103.14030). Window attention, relative position bias, shifted windows with masking.
    https://arxiv.org/abs/2103.14030
14. JetFormer (Zheng et al.; arXiv 2601.17215, January 2026). §3.1 class token, no positional encoding; §5.2.1 the
    150P and JetClass models; Table 2: 30 / 50 / 100 / 150 particles; §4.4 JetFormer-tiny on FPGA (Allo HLS).
    https://arxiv.org/abs/2601.17215. Code: https://github.com/walkieq/JetFormer
15. Accelerating Transformer Neural Networks on FPGAs for High Energy Physics Experiments (Wojcicki, Que, Tapper,
    Luk; ICFPT 2022). Repository: https://github.com/custom-computing-ic/Transformer_Neural_Network_HLS
16. Sub-microsecond Transformers for Jet Tagging on FPGAs (Laatu et al.; NeurIPS 2025 ML4PS, arXiv 2510.24784).
    Single-head MHA, 8–64 particles, U250, 44–140 ns. https://arxiv.org/abs/2510.24784
17. Particle Transformer (ParT; arXiv 2202.03772). JetClass; 8 heads × 16; pair bias $U$.
    https://arxiv.org/abs/2202.03772
18. Low Latency Transformer Inference on FPGAs for Physics Applications with hls4ml (Jiang et al.; JINST 20 P04014,
    arXiv 2409.05207). https://arxiv.org/abs/2409.05207
19. Ultra Fast Transformers on FPGAs for Particle Physics Experiments (Jiang et al.; arXiv 2402.01047). b-tagging,
    15 tracks, 2 heads, 9,135 parameters. https://arxiv.org/abs/2402.01047
