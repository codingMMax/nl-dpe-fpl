ngxiang# Mapping and tiling parameters of the attention layer

Status: 2026-10-12, from the DSE design discussion. Every number below comes from the V1 cycle model
(`v2/sim/simulator/kernels/attention_sim.py`, `pass_engine.py`; used pass extent, P29) unless marked otherwise. The
reference points of that model are RTL-pinned; values off the reference candidate are model-only; area is counted in
crossbars only (CLB and memory need VTR).

## 1 Design space

$$
\text{design point} = \text{crossbar config } (R, C) \times \text{attention shape } (n, d, d_m) \times \text{mapping / tiling parameters}
$$

- Configs: $R \in \{128, 256, 512, 1024\} \times C \in \{64, 128, 256\}$, $I = \min(R, C)$.
- Shapes $(n, d, d_m)$ (2026-10-13 set; sources in `v2/experiment/dse/plan.md` §3.1, to be rewritten into
  `docs/experiment.md`): ViT-B/16 $(256, 64, 768)$, DeiT-T $(256, 64, 192)$, BERT-base $(128, 64, 768)$, BERT-Tiny
  $(128, 64, 128)$, JetFormer-150P $(32, 32, 64)$, JetFormer-JetClass $(128, 16, 128)$. $V = \lceil d_m / R \rceil$.
  The measurements in §2–§5 were taken on the 2026-10-12 set (ViT-B/16, BERT-base, BERT-Tiny of it are in this set).
- Configs and shapes are given. The mapping / tiling parameters are the only free choice; this file lists them and
  their measured impact.

## 2 Two crossbar facts behind the attention results

**F-load: a crossbar loads its input at 5 bytes per cycle** ($BUF = 40$ bits), so a pass takes about
$\text{rows}/5$ cycles. Steady pass per token (`crossbar_total`, $P = 8$, $BUF = 40$, 64 columns):

| rows loaded | 128 | 256 | 512 | 768 |
|---|---|---|---|---|
| cycles per pass | 34 | 60 | 111 | 162 |

The $V$ row tiles of a projection load in parallel. For $d_m = 768$: at $R = 128$, six tiles take 34 cycles per
token; at $R = 1024$, one crossbar takes 162 cycles per token (4.8×). Projection time grows with $R$, and the
projections dominate the large-$R$ rows.

**F-conv: an element-wise conversion uses at most an $I \times I$ block.** A pass outputs one value per column and
each value needs its own input row, so no mapping converts more than $I$ values per pass (`pass_engine.identity_extent`:
the used extent loads and drains $I$). The engine (QK / PV farms, softmax, o-carry) is 88–99 % of a single-head
row's crossbars and is all conversions. Share of each crossbar the engine can use, $I^2/(RC)$:

| | C = 64 | C = 128 | C = 256 |
|---|---|---|---|
| R = 128 | 0.50 | 1.00 | 0.50 |
| R = 256 | 0.25 | 0.50 | 1.00 |
| R = 512 | 0.125 | 0.25 | 0.50 |
| R = 1024 | 0.0625 | 0.125 | 0.25 |

Conversion pass time grows with $I$: $T(1)$ = 36 / 62 / 114 cycles and $T(16)$ = 351 / 572 / 1,014 cycles at
$I$ = 64 / 128 / 256.

Both facts are properties of crossbars, not of the DSE method: tall crossbars pay off in GEMMs with long rows (FC),
not in attention's element-wise work.

## 3 Parameter inventory

RTL parameters today. `attention_top` (V1): `NSEQ, DM, D, NH, BKV, R, C, BUF, P, N_EXP, N_E, N_EP, N_OC`.
`attention_core` (dse_v2): `D, BKV, R, C, BUF, P, N_EXP, N_E, N_EP, N_OC, N_MAX` (+ `CWOFF`, test only).
Shape = `NSEQ, DM, D, NH`; config = `R, C`; platform constants = `BUF = 40`, `P = 8`.

### 3.1 Projections (X row in, Q / K / V row out; O-projection)

| parameter | free? | impact |
|---|---|---|
| $V = \lceil d_m/R \rceil$, $H = \lceil d/C \rceil$, $H_O = \lceil d_m/C \rceil$ | derived | — |
| $P_{proj}$ (projection copies, D10) | yes; **model only, in neither RTL** | cycles ×0.68–0.88 at 2 copies, ×0.52–0.86 at 4 (§4). One copy costs $3VH + H_O$ crossbars, so it is cheapest at large $R$: for $d_m$ 768, +6 crossbars at 1024×256 vs +33 at 128×64 |
| row split $V' > V$ ($d_m$ over more crossbars than needed) | not modeled, not in RTL | cuts the per-token load (F-load) like a small $R$; costs crossbars, a byte-tree reduction and convert-once pools (the $V > 1$ path) |
| Q / K / V column packing (one head's Q, K, V side by side, $3d \le C$) | not modeled, not in RTL | at 1024×256: one crossbar instead of three; 128 token rows take 20,751 → 20,777 cycles (rows 768; 64 → 192 columns; the pass is load-bound); X loaded once instead of three times. $C = 128$: 2 crossbars instead of 3; $C = 64$: none. In a single-head row: 2 of about 280 crossbars |

### 3.2 FlashAttention tiling

| parameter | free? | impact |
|---|---|---|
| $S$ (query rows per run) | derived, $S = \min(n, I)$, in both RTLs and the model; the softmax domain allows $8 \le S \le I$, nothing implements a choice | smaller $S$: more runs, each re-streams all K / V; conversions fill fewer of the $I$ lanes. Expected: more cycles, less Q / accumulator memory (reasoning, not measured) |
| $B_{kv}$ (keys per block) | yes; power of two, $8 \le B_{kv} \le d$, $16 B_{kv} \ge I$, corner turn $B_{kv} \le \lceil 8R/40 \rceil$ | cycles ×0.99–1.07 over {8, 32, 64}; 16 is the only value legal on all 12 configs at $d = 64$. Keep 16 |
| $B_q$ (query block) | decided, $B_q = S$ (D7) | — |
| QK accumulator buffers | decided, 2 (D12) | 1 buffer: cycles ×1.00–1.02 |

### 3.3 Engine mapping and sizing

| parameter | free? | impact |
|---|---|---|
| windows and groups ($P_{mm}$, $P_{sm}$, $A_{GRP}$, identity blocks per crossbar) | derived from $S$, $B_{kv}$, $d$, $I$ ("no free numbers") | — |
| $n_{exp}$ (softmax EXP crossbars; farms default to $d \cdot n_{exp}$) | yes (reference = smallest power of two reaching the stream / fill floor, `v2/sim/spec/attention.md` §4.1) | ref 2. 1: ×1.06–1.29, −129 crossbars. 4: ×0.88–0.98, +258. 8, 16: no further gain |
| $N_E$ (QK farm; in the model the PV farm follows it, `farm_sizes`: $N_{EP} = N_E$) | yes, $\ge d$ (tile guard $K \le N_E$) | ref 128. 64: ×1.03–1.13, −128 (both farms). 32: ×1.15–1.84. 256: ×0.94–0.99, +256 |
| $N_{EP}$ (PV farm; the o-carry has its own EXP group since 2026-10-08) | **RTL: $\ge N_{EXP}\,d$** (2026-10-14: the PV tile's corner-turn guard, $N_{EP}/B_{kv}$ a multiple of $A_{CH} A_{NW} = N_{EXP}\,d/B_{kv}$), so not below the reference at $N_{EXP} = 2$ | ref 128. 64: ×1.03–1.12, −64 (PV alone, model only, outside the RTL domain). 32: ×1.13–1.83. 256: ×0.94–0.99, +128 |
| $N_{OC}$ (o-carry LOG crossbars) | yes | ref 6–8. 4: ×1.01–1.16. 2: ×1.13–1.98. 1: ×1.37–3.69 (saves 10–14 crossbars). 16: ×0.97–0.99 |
| $N_{MAX}$ (dse_v2 only: KV buffer length) | yes, $\ge n$ | memory only, no cycles; 2048 today, 8× the largest $n$ (256) |

## 4 Measured impact (one parameter at a time)

Method (2026-10-12): `attention_cycle_model` + `attention_dpe_count` + `run_dse.attention_energy`, used extent, one
parameter changed from the reference candidate; ten (config, shape) points: BERT-Tiny $n$ 128 at 512×128 and
1024×256; BERT-base $n$ 128 and ViT-B/16 $n$ 256 at 128×64, 256×256, 512×128 and 1024×256. The ranges in §3 are the
min–max of the cycle ratio over the points.

- **Energy**: no parameter moves it by more than 1 % (ratio 0.99–1.01). Energy is set by the shape and the config;
  every parameter is a latency vs area trade.
- **Engine parameters are at their knee**: growing the engine buys ≤ 12 % for twice the crossbars; shrinking
  $N_{OC}$ below the reference is expensive. Halving a farm costs 3–13 % in cycles and saves 64–128 crossbars, so the
  farm size is an area knob. Estimate: BERT-Tiny at 1024×256 (crossbar tiles 39 % of the VTR area, 2026-10-10 run),
  half the QK farm ≈ −18 % area for +7 % cycles.
- **$P_{proj}$ is the largest lever and the one that favors large crossbars** (F-load): up to −48 % cycles, at the
  lowest crossbar cost where $V = 1$.
- **$B_{kv}$ and the accumulator buffers**: ≤ 7 %; fixed.

## 5 Proposed search per (config, shape)

| parameter | values |
|---|---|
| $P_{proj}$ | {1, 2, 4} |
| farm scale | {½, 1} × reference (2026-10-14: on $N_E$ only, $N_{EP}$ fixed; `experiment.md` §4.2) |
| everything else | fixed: $S = \min(n, I)$, $B_{kv} = 16$, $B_q = S$, $n_{exp}$ and $N_{OC}$ at the reference, 2 accumulator buffers |

6 candidates per point, model-evaluated in seconds; each config keeps its best candidate per metric. Prerequisites:
1. $P_{proj}$ in the RTL (V1: `attention_top`; with the projections outside the core: the projection side).
2. An area model for CLB and memory. In the 2026-10-10 VTR rows (BERT-Tiny $n$ 128), CLB and memory depend on $I$
   and hardly on $R$: CLB 10,587–10,591 / 13,025–13,077 / 16,777–16,789 and memory 272 / 218 / 227 at $I$ = 64 /
   128 / 256 — fit on the committed attention rows, confirm on the picks.
3. VTR on the picked candidates only.

Not proposed: row split and Q / K / V packing (extra RTL for little gain in single-head rows); $S < I$ (not
implemented anywhere; expected to lose cycles).

## 6 Status

- 2026-10-14 (user): the 6-candidate search is adopted, including the $P_{proj}$ RTL work. The mapping space as
  decided (every parameter by class, candidates, pick rule, RTL work) is `experiment.md` §4. There the farm scale is
  proposed on $N_E$ only, because $N_{EP}$ cannot be halved inside the RTL domain (§3.3 row above).
- Measure $S < I$ only if a memory-limited design needs it.
