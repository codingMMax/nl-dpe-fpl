# Safe-Softmax Study — RTL + VTR + Analytical Energy

**Date**: 2026-08-05
**Spec**: `docs/superpowers/specs/2026-08-05-softmax-rtl-vtr-study-design.md` (as-built)
**Plan**: `docs/superpowers/plans/2026-08-05-softmax-study-implementation.md`

Standalone study — no simulator involvement. Cycles are measured in RTL
simulation (iverilog) and checked against locked closed-form formulas;
Fmax and resources come from VTR (3 seeds, averaged); energy is analytical
(op counts × arch-JSON constants).

## 1. What is compared

One safe-softmax stage over an S×S attention-score matrix, S ∈ {128, 256},
identical workload spec on all three architecture points:

| Point | Crossbar R×C | Arch XML | Softmax mechanism |
|---|---|---|---|
| Proposed-1 | 1024×128 | `benchmarks/arch/proposed_auto.xml` | DPE(I\|exp) ×16n + shared DPE(I\|log) |
| Proposed-2 | 1024×256 | `benchmarks/arch/al_like_auto.xml` | same RTL, C=256 energy charging |
| Azure-Lily | 512×128 | `benchmarks/arch/azure_lily_auto.xml` | exp/recip LUT ROMs + CLB trees + 16×16 `mac_int_9x9` |

Shared mapping (all rows): 3-pass safe softmax (max → subtract+exp+sum →
normalize), W=16 row-parallel lockstep lanes, `rows_per_lane = S/16`,
no tiling (full row buffered), 16-wide CLB stages, pipelined rows
(row r+1's max/exp overlaps row r's scalar/normalize).

**Output-domain caveat (must accompany any use of this table):** NL blocks
emit **log-domain** values (`log p_i`, per `log_softmax_fusion` — downstream
`mac_sv` consumes log-domain directly); Azure-Lily emits **linear**
probabilities. The two outputs are not bit-comparable.

NL DPE count = 16·n + 1 with n = ceil(S/C) exp DPEs per lane (free split:
every pass full) + 1 shared log DPE that converts all 16 lane sums per pass.

## 2. Result table

Fmax = 3-seed average; resources are seed-invariant. Cycles are measured
(and formula-exact, §4). Latency = cycles / Fmax. Energy per §5.

| Arch | R×C | S | CLB | DSP | DPE (wc) | BRAM | Fmax (MHz) | Cycles | Latency (µs) | Matrices/s | Energy (pJ) | pJ/elem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed-1 | 1024×128 | 128 | 1049 | 1 | 17 | 220 | 74.81 | 290 | 3.877 | 257,954 | 21,358 | 1.304 |
| Proposed-1 | 1024×128 | 256 | 1473 | 1 | 33 | 223 | 61.89 | 514 | 8.305 | 120,414 | 84,673 | 1.292 |
| Proposed-2 | 1024×256 | 128 | 1049 | 1 | 17 | 220 | 76.90 | 290 | 3.771 | 265,166 | 24,343 | 1.486 |
| Proposed-2 | 1024×256 | 256 | 1428 | 1 | 17 | 212 | 63.54 | 956 | 15.045 | 66,466 | 77,057 | 1.176 |
| Azure-Lily | 512×128 | 128 | 2338 | 64 | 0 | 284 | 48.41 | 99 | 2.045 | 489,031 | 76,001 | 4.639 |
| Azure-Lily | 512×128 | 256 | 2625 | 64 | 0 | 284 | 43.38 | 323 | 7.445 | 134,313 | 303,419 | 4.630 |

Seed Fmax spreads: AL_s128 47.8–49.4, AL_s256 42.9–43.8, P1_s128 71.0–77.7,
P2_s128 74.7–80.2, P1_s256 60.4–63.3, P2_s256 60.9–65.5 MHz.
The P1/P2 rows at S=128 are the **same netlist** (n=1, E=128) placed on the
two different arch grids — their Fmax difference (74.8 vs 76.9) is within
seed spread; treat them as tied.

DSP=1 on NL rows is the loader's `row × LCYC` address multiply (LCYC=26/52,
not a power of two) — one `multiply` block, not part of the softmax datapath.
64 DSP on AL = 256 `mac_int_9x9` ÷ 4 per `dsp_top` tile.

## 3. Findings

**F1 — Cycles: AL leads 3.2×; the ratio is exactly port-width.**
AL moves 16 elements/cycle/lane through its LUT array; NL moves 5·n through
the 40-bit DPE port. Measured cycle ratios (S=128: 290/99 = 2.9; S=256 P2:
956/323 = 3.0; P1 n=2: 514/323 = 1.6) track `16/(5n)` with fill effects.
Crossbar width C never enters — P1 = P2 = 290 cycles at S=128 confirms
port-bound, not column-bound.

**F2 — Fmax reverses most of it: wall-clock nearly ties at S=256.**
The NL blocks close ~75/62 MHz; AL closes 48/43 MHz — its critical path runs
through exp-ROM LUT logic, 16-input trees, and DSP chains, while the DPE is
a hard block. Latency: AL 2.05 µs vs NL 3.8–3.9 µs at S=128 (AL 1.9× faster);
at S=256 AL 7.45 µs vs P1 8.31 µs — only 1.12×. All three run far below the
300 MHz `fpga_specs.freq` assumption used elsewhere; softmax logic, not the
global clock target, is the binding path.

**F3 — Energy: NL wins 3.1–3.9× per element.**
1.18–1.49 pJ/elem (NL) vs 4.63 pJ/elem (AL). AL's cost is dominated by the
exp-ROM lookups charged at ALPHA=1.0 (upper bound; 4 CLBs × 0.66 pJ per
lookup = 2.64 pJ/elem) plus 1.2 pJ/elem of DSP multiply. NL's DPE component
is 0.44–0.62 pJ/elem.

**F4 — P1 vs P2 is the predicted latency/energy crossover.**
- S=128: P2's 256-wide crossbar carries only 128 useful elements but fires
  fully → DPE energy 0.62 pJ/elem vs P1's 0.44. P1 wins energy 1.14× overall
  (21.4 vs 24.3 nJ); latency tied. **P1 dominates at S=128.**
- S=256: P2 fills up (0.31 vs 0.44 pJ/elem DPE; 77.1 vs 84.7 nJ total,
  1.10× better), but P1's free split (n=2, 33 DPEs) makes it 1.81× faster
  (8.3 vs 15.0 µs). **Latency → P1, energy → P2.**

**F5 — Resources.**
AL: 2338–2625 CLB + 64 DSP tiles + 284 BRAM. NL: 1049–1473 CLB + 17–33 DPE
tiles + 212–223 BRAM + 1 DSP. AL's 64 `dsp_top` far exceeds
`azure_lily.json`'s `total_dsp: 16` budget — the 16-lane × 16-wide softmax
would monopolize 4× the chip's DSP allocation; VTR quantifies exactly the
feasibility concern the config implies. BRAM: 7 of AL's 284 come from packing
32 exp banks; NL avoids exp storage entirely (log-domain recompute) but pays
for a third score copy — net BRAM is comparable (284 vs 212–223).

## 4. Cycle formulas (locked, exact at all 6 smoke points)

`softmax_study/run_softmax_smoke.py` verifies functional bit-exactness AND
these closed forms on every run (6/6 PASS):

```
AL: total = (RPL + 4)·WPR + 3                     WPR = RPL = S/16
    S=128: 99   S=256: 323

NL: LCYC = ceil((S/n)/5)
    fill  = (WPR+4) + (LCYC+10+LCYC+2) + 20 + WPR + 4
    total = fill + (RPL−1)·max(WPR, LCYC)
    S=128 (n=1): 290    S=256 (n=2): 514    S=256 (n=1): 956
```

The NL formula is the spec §4 stage model plus a constant +4 (four FSM pipe
registers: a_v, bs_v, d_v, done). Functional truth for NL is the DPE
behavior model's ACAM semantics (`exp ≈ 1+x+x²/2`, `log ≈ x−1`,
`fc_verification/rtl/dpe_nldpe.v:203-209`), not float exp/log; for AL it is
the generated ROM contents. Oracles in `run_softmax_smoke.py` replicate both
bit-exactly.

## 5. Energy model

`softmax_study/softmax_energy.py`; constants read from
`azurelily/IMC/configs/{nl_dpe,azure_lily}.json` and the two `ref_*`
calibration values (`azurelily/IMC/imc_core/config.py:130-131`).

```
E_pass(C) = 8·e_analoge + 8·e_conv + e_digital·C     (full crossbar fires)
          = 53.07 pJ (C=128) / 75.01 pJ (C=256)
CLB compare 0.2644 pJ, CLB add 0.0850 pJ, DSP MAC 1.2 pJ,
LUT-ROM access 4 CLB × 0.66 pJ × ALPHA (ALPHA = 1.0, upper bound),
BRAM 0.0495 pJ/element-access.

Common:  S(S−1) compares + S² subtracts + S(S−1) sum adds + 7S² BRAM accesses
AL only: S²+S ROM lookups, S² DSP multiplies
NL only: S² log-domain subtracts, S·ceil(S/C) exp passes + ceil(S/16) log
         passes, each at E_pass(C)
```

Both DPE fires are charged at the full physical crossbar C even when
partially filled (that is what produces F4's crossover). Port TIME, by
contrast, scales with elements moved — the port-honest convention; this
deliberately diverges from `imc_core.dimm_nonlinear`, which charges
`load_dim = cols` per pass. Do not cross-compare cycle counts without
accounting for that.

## 6. Reproduce

```
python3 softmax_study/gen_luts.py                      # regenerate ROM include
python3 softmax_study/run_softmax_smoke.py             # 6/6 functional+cycle
python3 softmax_study/run_vtr_softmax.py --all --jobs 6  # 18 VTR runs (~1 h)
python3 softmax_study/softmax_energy.py                # final table
```

Outputs: `results/smoke.json`, `results/vtr_softmax.json`,
`results/softmax_table.{md,csv,json}`.

VTR flags: `--route_chan_width 300`, seeds {1,2,3}, and
`--pack_high_fanout_threshold memory:100000` — without the last one, VPR's
default memory:128 threshold drops the per-lane wide-memory address nets
from clustering attraction and scatters BRAM slices 2–3 per block
(measured: AL_s128 mem = 1542 instead of 284).

## 7. Known limitations

- ALPHA = 1.0 charges every ROM CLB on every lookup — an upper bound on AL
  exp energy. NL's advantage in F3 shrinks if ALPHA < 1.
- The `dsp=1` loader multiply and the load path itself are outside the
  measured cycles (loading is upstream traffic, identical for both archs).
- NL functional truth is the behavior model's ACAM approximation; accuracy
  vs float softmax is out of scope (the model itself is the contract).
- Fmax at S=256 has ±4% seed spread; single-grid `auto_layout` per arch.
- BRAM counts reflect VTR's 512×40 single-port block and this flow's packing;
  a flow with LUTRAM would map the shallow 8-deep exp banks differently.
