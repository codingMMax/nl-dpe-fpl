# simulator/tests/ref — v2's FC results, the regression reference

Locked copies of v2's FC DSE run of 2026-10-10 (`tools/copies.py` records each source's sha256; in `dse/fc_snapshot/`
until 2026-10-17):

- `dse_v2.csv`: v2's priced rows (84: the 48 `family == gemm` FC rows that matter, and 36 rows of the earlier attention
  DSE, which is not a valid experiment); energy at v2's price rule (every pass as the full array on both sides).
- `vtr_results.csv`: v2's VTR table for the same 84 rows; its 48 FC rows are also `dse/results/fc_vtr_results.csv`.
- `ranking_fc.json`: v2's own FC ranking (the v3 `dse/rank.py` reproduced it to 4e-16 when ported).

`testFcRows.py` rebuilds the 48 FC rows with the v3 model and checks them against `dse_v2.csv`: cycles, counts and
area exactly; the energy exactly with the full extent, and with the used extent equal to v2's minus the unread rows'
input cost. The current FC results are `dse/results/fc_priced_results.csv` (`dse/fc_rows.py`).
