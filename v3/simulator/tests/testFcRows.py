#!/usr/bin/env python3
"""testFcRows.py — the 48 FC rows of the v2 snapshot (docs/simulator.md §7, "FC regression").

Every FC row of v2's priced table (simulator/tests/ref/dse_v2.csv, family gemm: 12 crossbar configs x 4 shapes, P29
"used" extent) is rebuilt from the FC workload: cycles from the run, energy from the Report's counts x the platform
prices, area from the crossbar count and v2's VTR metrics (ref/vtr_results.csv). Integers exactly, floats to 1e-12
relative (the same operations in the same order give the same doubles). The v3 rows name the memory term e_mem_pj
where v2 wrote e_ser_pj.

The pass price changed on 2026-10-17 (user): the whole array still fires on every pass, but only the rows in use are
read (the input-buffer bytes and the input port words follow ROWS_USED); v2 priced both sides of every pass as the full
array. So the energy is checked in two steps:
  1. the full extent: every pass reads R rows, and e_pass, e_traffic and e_total must equal v2's exactly;
  2. the used extent (the snapshot's rows): e_pass and e_traffic must be v2's minus the unread rows' input cost,
     passes x [3 (R - rows used) x the input-buffer pJ/byte + (words(R) - words(rows used)) x the block-RAM pJ/access],
     computed here from the row's M, V, H, K and the raw price table, never from prices.py's pass functions. The rows
     with rows used == R (36 of 48) must then equal v2 exactly, and the 12 rows with K < R and V = 1 must be cheaper.
Run: python3 v3/simulator/tests/testFcRows.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
from simulator.platform.rows import fcRow  # noqa: E402
from simulator.platform import area  # noqa: E402
from simulator.platform.platform import BUFFER_ACCESSES, Platform  # noqa: E402

REF = V3 / "simulator" / "tests" / "ref"           # v2's priced FC rows and VTR table, locked copies (tools/copies.py)
INT_COLS = ("R", "C", "M", "K", "N", "V", "H", "dpe_count", "T_cycles", "area_wc_mwta", "clb_count", "mem_blocks",
            "wc_vtr", "area_mem_mwta", "area_total_mwta")
FLOAT_COLS = ("e_adds_pj", "e_ser_pj", "e_setup_pj", "fmax_mhz", "latency_ns")     # the same under both price rules
PASS_COLS = ("e_pass_pj", "e_traffic_pj", "e_total_pj")                              # the input side moved
V3_NAME = {"e_ser_pj": "e_mem_pj"}                                                   # v2's column -> the v3 row's


def close(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), 1e-300)


def main() -> int:
    with open(REF / "dse_v2.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["family"] == "gemm"]
    assert len(rows) == 48, len(rows)
    vtr = area.vtrTable(REF / "vtr_results.csv")
    bad, worst, cheaper, savings = [], 0.0, 0, []
    for r in rows:
        R, C, M, K, N = (int(r[k]) for k in ("R", "C", "M", "K", "N"))
        p = Platform("nl_dpe", rows=R, cols=C)
        shape = (M, K, N)
        got = fcRow(R, C, shape, p, vtr, passExtent=r["pass_extent"])
        full = fcRow(R, C, shape, p, vtr, passExtent="full")
        tag = f"{r['config']} K{K} N{N}"
        for k in INT_COLS:
            if int(r[k]) != int(got[k]):
                bad.append(f"{tag} {k}: {r[k]} != {got[k]}")
        for k in FLOAT_COLS:
            rel = close(float(r[k]), float(got[V3_NAME.get(k, k)]))
            worst = max(worst, rel)
            if rel > 1e-12:
                bad.append(f"{tag} {k}: {r[k]} != {got[V3_NAME.get(k, k)]!r}")
        for k in PASS_COLS:                                        # 1 the full extent prices every pass as v2 did
            rel = close(float(r[k]), float(full[k]))
            worst = max(worst, rel)
            if rel > 1e-12:
                bad.append(f"{tag} full-extent {k}: {r[k]} != {full[k]!r}")
        V, H = int(got["V"]), int(got["H"])
        rowsUsed = K if V == 1 else R                              # the P29 used extent of gemm_top
        passes = M * V * H
        if int(got["passes"]) != passes:
            bad.append(f"{tag} passes: {got['passes']} != {passes}")
        words = lambda n: math.ceil(n * 8 / p.portWidthBits)       # noqa: E731
        inBytePj = p.components["input_buffer"]["energy_pj_per_byte"]
        bramPj = p.fabric["block_ram"]["energy_pj_per_access"]
        savePass = passes * BUFFER_ACCESSES["input_byte"] * (R - rowsUsed) * inBytePj
        saveTraffic = passes * (words(R) - words(rowsUsed)) * bramPj
        want = dict(e_pass_pj=float(r["e_pass_pj"]) - savePass, e_traffic_pj=float(r["e_traffic_pj"]) - saveTraffic,
                    e_total_pj=float(r["e_total_pj"]) - savePass - saveTraffic)
        for k in PASS_COLS:                                        # 2 the used extent: v2 minus the unread rows
            rel = close(want[k], float(got[k]))
            worst = max(worst, rel)
            if rel > 1e-12:
                bad.append(f"{tag} used-extent {k}: {want[k]!r} != {got[k]!r}")
        if rowsUsed < R:
            cheaper += 1
            savings.append(1 - float(got["e_total_pj"]) / float(r["e_total_pj"]))
            if not float(got["e_total_pj"]) < float(r["e_total_pj"]):
                bad.append(f"{tag}: rows used {rowsUsed} < {R} but the energy did not fall")
        elif close(float(r["e_total_pj"]), float(got["e_total_pj"])) > 1e-12:
            bad.append(f"{tag}: rows used == R but e_total {r['e_total_pj']} != {got['e_total_pj']!r}")
    assert not bad, f"{len(bad)} differences, first {bad[:5]}"
    assert cheaper == 12, f"{cheaper} rows read fewer rows than R, expected 12 (K < R with V = 1)"
    cycles = sorted(int(r["T_cycles"]) for r in rows)
    print(f"  fc rows  PASS  48/48 rows: {len(INT_COLS)} integer + {len(FLOAT_COLS)} float columns equal to v2; the "
          f"full extent reproduces v2's {len(PASS_COLS)} pass-energy columns; the used extent equals v2 minus the "
          f"unread rows' input cost on every row (max relative float difference {worst:.1e}); {cheaper} rows read "
          f"fewer rows than R and are {100 * min(savings):.1f}-{100 * max(savings):.1f} % cheaper; cycles "
          f"{cycles[0]}..{cycles[-1]}")
    print("fc rows: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
