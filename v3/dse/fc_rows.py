#!/usr/bin/env python3
"""fc_rows.py — the FC DSE rows from the FC workload: one row per (shape, config) (docs/experiment.md §3.1, §4.1, §5).

The four FC shapes (dse/space.py FC_SHAPES, Round 1's GEMVs, M = 1) on the 8 crossbar configs, the P29 used extent.
Each row is simulator/platform/rows.py `fcRow`: the GEMM's cycles, pass count, energy from the Report's counts x the
platform's prices (per term; weight programming apart, as setup), the crossbar count and area; with the VTR metrics
(--vtr: the FC VTR table, v2's run of 2026-10-10 kept in dse/results/fc_vtr_results.csv) also the total area, latency
and EDAP. The v2 snapshot of these rows (its prices, its simulator) is the regression reference of
simulator/tests/testFcRows.py (simulator/tests/ref/dse_v2.csv); this file is the current result.
Usage: python3 v3/dse/fc_rows.py v3/dse/results/fc_priced_results.csv --vtr v3/dse/results/fc_vtr_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse.space import CONFIGS, FC_PASS_EXTENT, FC_SHAPES  # noqa: E402
from simulator.platform import area  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402
from simulator.platform.rows import fcRow  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--vtr", default=None, help="the FC VTR table (vtr_results.csv columns)")
    a = ap.parse_args()
    vtr = area.vtrTable(Path(a.vtr)) if a.vtr else None
    rows = [fcRow(R, C, shape, Platform("nl_dpe", rows=R, cols=C), vtr, passExtent=FC_PASS_EXTENT)
            for (R, C) in CONFIGS for shape in FC_SHAPES]
    missing = [r for r in rows if "fmax_mhz" not in r]
    if vtr is not None and missing:
        raise SystemExit(f"{len(missing)} rows have no VTR metrics, e.g. {missing[0]['config']} "
                         f"{missing[0]['M']}x{missing[0]['K']}x{missing[0]['N']}")
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"fc_rows: {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
