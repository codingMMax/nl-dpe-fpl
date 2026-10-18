"""data.py — the DSE's two input tables, loaded the same way by every script in dse/:

  attentionRows()   the 72 priced attention builds, dse/results/att_priced_72.csv (dse/att_rows.py --vtr): one dict per
                    build with the CSV's columns as strings
  fcRows()          the 48 FC rows of the v2 snapshot (dse/fc_snapshot/: the FC rows of dse_v2.csv with their VTR
                    metrics), rebuilt by the v3 model (simulator/platform/rows.py fcRow, as simulator/tests/testFcRows.py
                    checks them): one dict per row with numbers as numbers

Both carry, per row: config "RxC", R, C, T_cycles, passes, the energy terms e_*_pj (the FC rows name the memory term
e_ser_pj, v2's column), clb_count, wc_vtr, mem_blocks, area_total_mwta, fmax_mhz, latency_ns.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
RESULTS = V3 / "dse" / "results"
SNAP = V3 / "dse" / "fc_snapshot"
ATT_CSV = RESULTS / "att_priced_72.csv"
MM2_PER_MWTA = 0.033864e-6          # 1 MWTA = 0.033864 um^2


def attentionRows(path: Path = ATT_CSV) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 72 or any(not r.get("edap") for r in rows):
        raise SystemExit(f"{path}: expected 72 priced builds (dse/att_rows.py --vtr), got "
                         f"{sum(bool(r.get('edap')) for r in rows)} of {len(rows)}")
    return rows


def fcRows() -> list:
    from simulator.platform.rows import fcRow
    from simulator.platform import area
    from simulator.platform.platform import Platform
    vtr = area.vtrTable(SNAP / "vtr_results.csv")
    with open(SNAP / "dse_v2.csv", newline="") as f:
        snap = [r for r in csv.DictReader(f) if r["family"] == "gemm"]
    rows = [fcRow(int(r["R"]), int(r["C"]), (int(r["M"]), int(r["K"]), int(r["N"])),
                  Platform("nl_dpe", rows=int(r["R"]), cols=int(r["C"])), vtr, passExtent=r["pass_extent"])
            for r in snap]
    if len(rows) != 48:
        raise SystemExit(f"FC: {len(rows)} rows, expected 48")
    return rows


def fcWorkload(r: dict) -> str:
    """The FC workload name of a row, M x K x N."""
    return f"{r['M']}x{r['K']}x{r['N']}"
