# v3/dse — the DSE's numbers, rankings, figures and pages

Everything here lives in `results/`, one pair of tables per family: the VTR table (what VTR measured per build) and the
priced table (what the simulator computed per build, with the VTR columns joined in), `fc_vtr_results.csv`,
`fc_priced_results.csv`, `attn_vtr_results.csv`, `attn_priced_results.csv`; then the rankings, the breakdown, the
figures and the pages. From the repository root, with `export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$PWD/v3/work/tmp`:

```text
bash v3/dse/run_all.sh          every step below, in order (about a minute; the VTR results are inputs, never re-run)
```

| step | script | reads | writes |
|---|---|---|---|
| — | `space.py` | — | the 6 attention shapes and the 12 crossbar configs, defined once; imported by every script |
| 0 | `vtr_dse.py` (hours; on your go only; not part of `run_all.sh`) | the 72 builds' circuits from the VTR twins (`top/`, `ip/*/rtl/vtr`), `vtr/arch/`, `vtr/run_vtr.py`, `vtr/smoke.py`, VTR | the run folders `work/vtr/dse/<build>_<hash>/` and `results/attn_vtr_results.csv`: per build Fmax per seed and the mean, CLB, memory, crossbars, I/O, DSP, the elaboration check. `--aggregate` rebuilds the CSV from the folders with no tool; `--self-test` runs it on fake tools; `dse_watch.py --pid PID` reports on a running driver. The FC table `results/fc_vtr_results.csv` is v2's run (below), not regenerated |
| 1 | `fc_rows.py OUT.csv --vtr results/fc_vtr_results.csv` | the simulator (one GEMV per row, `simulator/platform/rows.py fcRow`; `space.py FC_SHAPES` × `CONFIGS`), the FC VTR table | `results/fc_priced_results.csv`: per row the cycles, the pass count, the energy per term, the crossbars, and with VTR the area, latency and EDAP |
| 1 | `attn_rows.py OUT.csv --vtr results/attn_vtr_results.csv` | the simulator (one head per build, `attentionRow`), the attention VTR table | `results/attn_priced_results.csv`: the same per build, plus the energy per role and the memory inventory |
| 1b | `check_energy.py` (4 s; part of `run_all.sh`) | both priced tables, the simulator's ledger of each row, the raw price table `simulator/platform/nl_dpe.json` | nothing: it prices every row from the ledger with the energy rule restated from the raw table (not from `prices.py`), compares the energy terms, the pass counts and the attention role columns with the row, and tries the pass price at its boundaries; exit 1 on any difference |
| — | `data.py` | the two priced tables | `fcRows()`, `attentionRows()` for the scripts below |
| 2 | `rank.py` | both tables | `results/adp/latency_area.json`: per family $G_A$, $G_D$, $G_E$ and the latency vs area front; $G_{ADP}$ and the cross-family front and distance. `results/edap/ranking_{fc,attention,cross}.json`: the EDAP view. `--self-test` checks the arithmetic |
| 3 | `breakdown.py` | both tables; the simulator for the projections' end | `results/breakdown.json`: per config the area split (DPE tiles / CLBs / memory blocks, VPR counts), the energy split (DPE passes / CLB / port traffic / memory bytes, the simulator's ledger), the means (crossbars, CLBs, cycles, Fmax, passes, the price paid per pass); prints the split tables |
| 4 | `plot_dse.py` (matplotlib: the miniconda Python) | the JSONs | `results/adp/dse_la_fc.png`, `dse_la_attention.png`: (a) the Pareto front, (b) the area split, (c) the energy split; `dse_adp_cross.png`; `dse_la.{png,pdf}` (the three ADP panels, for the paper); `results/edap/dse_{fc,attention,pareto,cross}.{png,pdf}` |
| 5 | `results_page.py` | the JSONs, the PNG sizes | `results/adp/dse_results.html` (the published page), `results/edap/dse_results_edap.html` |

## Conventions

- **Scores** (`rank.py`): per workload, score = the config's value ÷ the worst config's value, so the worst config is
  1 and lower is better; $G_X$ = the geometric mean of the scores over a family's workloads, $X$ = A (area), D
  (latency), E (energy), ADP (A × D), EDAP (E × D × A). The normalizer moves no rank and no Pareto front; it moves the
  cross-family distance $d$ from the origin.
- **Splits** (`breakdown.py`): attention takes the geometric mean of each share over its 6 shapes; FC takes the share
  of the total summed over its 4 shapes, because several FC builds have no memory block and no CLB add, and the
  geometric mean of a share that is 0 on one shape is 0.
- **Energy** (`simulator/platform/prices.py`, 2026-10-17): a crossbar pass fires the whole array (the cells, the ACAM
  on every column, the output buffer and the drain) and reads only the rows in use (the input-buffer bytes and the
  input port words follow `ROWS_USED`). Weight programming is setup, reported apart.
- **Fmax** is the mean of VTR seeds 11, 13 and 17; area counts CLBs, crossbar tiles and memory blocks from VPR.

## Inputs that are not regenerated here

- `results/attn_vtr_results.csv`: the VTR run of the 72 attention builds (`vtr_dse.py`, 2026-10-15/16, 19 h; the run
  folders are in `v3/work/vtr/dse/`, 276 GB, and `vtr_dse.py --aggregate` rebuilds the CSV from them).
- `results/fc_vtr_results.csv`: the VTR run of the 48 FC rows, v2's run of 2026-10-10 (user 2026-10-13: reused, not
  re-run): the 48 `family == gemm` rows of v2's `vtr_results.csv`, verbatim (`tools/copies.py` records the source).
  v2's own priced rows and ranking (`dse_v2.csv`, `ranking_fc.json`, with the 84-row `vtr_results.csv`) are the
  regression reference of `simulator/tests/testFcRows.py`, kept in `simulator/tests/ref/` as locked copies.

## Checks

```text
python3 v3/dse/check_energy.py                   every priced row against an independent pricing of its ledger
python3 v3/dse/rank.py --self-test               the score arithmetic on hand-made cases
python3 v3/simulator/tests/testFcRows.py         the 48 FC rows against v2's snapshot
python3 v3/tools/check_head_params.py            the simulator's head parameters == the RTL's, for the 72 builds
```
