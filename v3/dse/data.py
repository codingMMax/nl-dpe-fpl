"""data.py — the DSE's two priced tables, loaded the same way by every script in dse/:

  attentionRows()   the 48 priced attention builds (6 shapes x the 8 configs), dse/results/attn_priced_results.csv
                    (dse/attn_rows.py --vtr)
  fcRows()          the 32 priced FC rows (4 shapes x the 8 configs), dse/results/fc_priced_results.csv
                    (dse/fc_rows.py --vtr)

Both come back as the CSV's rows, one dict per row with every value a string; both carry config "RxC", R, C,
T_cycles, passes, the energy terms e_*_pj, clb_count, wc_vtr, mem_blocks, area_total_mwta, area_total_mm2, fmax_mhz,
latency_ns, edap. The workload of an attention row is its `shape`; of an FC row, fcWorkload(row) = "M x K x N".
"""
from __future__ import annotations

import csv
from pathlib import Path

from dse.space import CONFIGS, FC_SHAPES, SHAPES

V3 = Path(__file__).resolve().parents[1]
RESULTS = V3 / "dse" / "results"
ATT_CSV = RESULTS / "attn_priced_results.csv"
FC_CSV = RESULTS / "fc_priced_results.csv"
MM2_PER_MWTA = 0.033864e-6          # 1 MWTA = 0.033864 um^2


def _priced(path: Path, n: int, writer: str) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    priced = sum(bool(r.get("edap")) for r in rows)
    if len(rows) != n or priced != n:
        raise SystemExit(f"{path}: expected {n} priced rows ({writer}), got {priced} priced of {len(rows)}")
    return rows


def attentionRows(path: Path = ATT_CSV) -> list:
    return _priced(path, len(SHAPES) * len(CONFIGS), "dse/attn_rows.py --vtr")


def fcRows(path: Path = FC_CSV) -> list:
    return _priced(path, len(FC_SHAPES) * len(CONFIGS), "dse/fc_rows.py --vtr")


def fcWorkload(r: dict) -> str:
    """The FC workload name of a row, M x K x N."""
    return f"{r['M']}x{r['K']}x{r['N']}"
