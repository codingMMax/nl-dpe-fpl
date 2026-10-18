# v3/dse — the DSE's numbers, rankings, figures and pages

Everything here reads two tables (the 72 priced attention builds and the 48 FC rows of the v2 snapshot) and writes
`results/`. From the repository root, with `export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$PWD/v3/work/tmp`:

```text
bash v3/dse/run_all.sh          every step below, in order (about a minute; the VTR results are inputs, never re-run)
```

| step | script | reads | writes |
|---|---|---|---|
| — | `space.py` | — | the 6 attention shapes and the 12 crossbar configs, defined once; imported by every script |
| 1 | `att_rows.py OUT.csv --vtr results/vtr_att_72.csv` | the simulator (one head per build, `simulator/platform/rows.py attentionRow`), the VTR table | `results/att_priced_72.csv`: per build the cycles, the pass count, the energy per term and per role, the crossbars, the memory inventory, and with VTR the area, latency and EDAP |
| — | `data.py` | `results/att_priced_72.csv`; `fc_snapshot/` through the simulator (`fcRow`) | the two tables every script loads: `attentionRows()`, `fcRows()` |
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

- `results/vtr_att_72.csv`: the VTR run of the 72 builds (`v3/vtr/vtr_dse.py`, 2026-10-15/16, 19 h).
- `fc_snapshot/`: v2's FC rows (`dse_v2.csv`, `vtr_results.csv`, `ranking_fc.json`), locked copies
  (`tools/copies.py`); the v3 model rebuilds the rows from them (`simulator/tests/testFcRows.py` proves it).

## Checks

```text
python3 v3/dse/rank.py --self-test               the score arithmetic on hand-made cases
python3 v3/simulator/tests/testFcRows.py         the 48 FC rows against v2's snapshot
python3 v3/tools/check_head_params.py            the simulator's head parameters == the RTL's, for the 72 builds
```
