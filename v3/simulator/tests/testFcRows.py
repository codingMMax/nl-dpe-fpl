#!/usr/bin/env python3
"""testFcRows.py — the 48 FC rows of the v2 snapshot (docs/simulator.md §7, "FC regression").

Every FC row of dse/fc_snapshot/dse_v2.csv (family gemm: 12 crossbar configs x 4 shapes, P29 "used" extent) is rebuilt
from the FC workload: cycles from the run, energy from the Report's counts x the platform prices, area from the
crossbar count and the snapshot's VTR metrics. Every numeric column must be equal: integers exactly, floats to 1e-12
relative (the same operations in the same order give the same doubles).
Run: python3 v3/simulator/tests/testFcRows.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
from simulator.platform.rows import fcRow  # noqa: E402
from simulator.platform import area  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402

SNAP = V3 / "dse" / "fc_snapshot"
INT_COLS = ("R", "C", "M", "K", "N", "V", "H", "dpe_count", "T_cycles", "area_wc_mwta", "clb_count", "mem_blocks",
            "wc_vtr", "area_mem_mwta", "area_total_mwta")
FLOAT_COLS = ("e_pass_pj", "e_adds_pj", "e_traffic_pj", "e_ser_pj", "e_total_pj", "e_setup_pj", "fmax_mhz",
              "latency_ns")


def main() -> int:
    with open(SNAP / "dse_v2.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["family"] == "gemm"]
    assert len(rows) == 48, len(rows)
    vtr = area.vtrTable(SNAP / "vtr_results.csv")
    bad, worst = [], 0.0
    for r in rows:
        R, C = int(r["R"]), int(r["C"])
        got = fcRow(R, C, (int(r["M"]), int(r["K"]), int(r["N"])), Platform("nl_dpe", rows=R, cols=C), vtr,
                    passExtent=r["pass_extent"])
        for k in INT_COLS:
            if int(r[k]) != int(got[k]):
                bad.append(f"{r['config']} K{r['K']} N{r['N']} {k}: {r[k]} != {got[k]}")
        for k in FLOAT_COLS:
            a, b = float(r[k]), float(got[k])
            rel = abs(a - b) / max(abs(a), 1e-300)
            worst = max(worst, rel)
            if rel > 1e-12:
                bad.append(f"{r['config']} K{r['K']} N{r['N']} {k}: {a!r} != {b!r}")
    assert not bad, f"{len(bad)} differences, first {bad[:5]}"
    cycles = sorted(int(r["T_cycles"]) for r in rows)
    print(f"  fc rows  PASS  48/48 rows, {len(INT_COLS)} integer + {len(FLOAT_COLS)} float columns "
          f"(max relative float difference {worst:.1e}); cycles {cycles[0]}..{cycles[-1]}")
    print("fc rows: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
