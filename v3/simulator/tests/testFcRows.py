#!/usr/bin/env python3
"""testFcRows.py — the 48 FC rows of the v2 snapshot (docs/simulator.md §7, "FC regression").

Every FC row of v2's priced table (simulator/tests/ref/dse_v2.csv, family gemm: 12 crossbar configs x 4 shapes, P29
"used" extent) is rebuilt from the FC workload: cycles from the run, energy from the Report's counts x the platform
prices, area from the crossbar count and v2's VTR metrics (ref/vtr_results.csv), with v2's crossbar tiles (the
legacy tile rule, area.footprint rule="legacy"; the row-parallel tile of 2026-10-22 is testArea.py's). Integers exactly, floats to 1e-12
relative (the same operations in the same order give the same doubles). The v3 rows name the memory term e_mem_pj
where v2 wrote e_ser_pj.

The cycles follow the bit-plane pass since 2026-10-22 (ip/gemm/spec.md): v2's T_cycles and latency_ns time the pre-bit-plane
pass, so they are checked against the closed form instead. One row (M = 1): the burst at 0, the drain from 10, the
first output plane L_W later and the last 7 after it, so T = 18 + L_W with L_W = ceil(log2 V) (V > 1) + 1; latency =
T / Fmax with v2's Fmax.

The pass price changed on 2026-10-17 (user): the whole array still fires on every pass, but only the rows in use are
read (the input-buffer bytes and the input port words follow ROWS_USED); v2 priced both sides of every pass as the full
array. So the energy is checked in two steps:
  1. the full extent: every pass reads R rows; e_pass must equal v2's exactly, and e_traffic / e_total v2's with the
     port traffic re-priced (2026-10-23, user: the row-parallel port, P planes of R rows in and C columns out per
     pass, per bit; v2 priced 40-bit words, so the difference is the word rounding), computed here from the raw table;
  2. the used extent (the snapshot's rows): the same minus the unread rows' input cost, passes x [3 (R - rows used) x
     the input-buffer pJ/byte + (R - rows used) x P x the port pJ/bit], never from prices.py's pass functions. The 12
     rows with K < R and V = 1 must be cheaper than v2; the other 36 equal v2 up to the traffic re-pricing.
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
INT_COLS = ("R", "C", "M", "K", "N", "V", "H", "dpe_count", "area_wc_mwta", "clb_count", "mem_blocks", "wc_vtr",
            "area_mem_mwta", "area_total_mwta")
FLOAT_COLS = ("e_adds_pj", "e_ser_pj", "e_setup_pj", "fmax_mhz")                   # the same under both price rules
PASS_COLS = ("e_pass_pj", "e_traffic_pj", "e_total_pj")                              # the input side moved
V3_NAME = {"e_ser_pj": "e_mem_pj"}                                                   # v2's column -> the v3 row's
V2_PORT_BITS = 40                     # v2's snapshot priced 40-bit port words; the row-parallel port is priced per bit


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
        got = fcRow(R, C, shape, p, vtr, passExtent=r["pass_extent"], tileRule="legacy")
        full = fcRow(R, C, shape, p, vtr, passExtent="full", tileRule="legacy")
        tag = f"{r['config']} K{K} N{N}"
        for k in INT_COLS:
            if int(r[k]) != int(got[k]):
                bad.append(f"{tag} {k}: {r[k]} != {got[k]}")
        for k in FLOAT_COLS:
            rel = close(float(r[k]), float(got[V3_NAME.get(k, k)]))
            worst = max(worst, rel)
            if rel > 1e-12:
                bad.append(f"{tag} {k}: {r[k]} != {got[V3_NAME.get(k, k)]!r}")
        V, H = int(got["V"]), int(got["H"])
        lw = (math.ceil(math.log2(V)) if V > 1 else 0) + 1          # the bit-serial tree's levels + the output register
        T = 10 * M + 8 + lw                                        # the bit-plane closed form (ip/gemm/spec.md)
        if int(got["T_cycles"]) != T:
            bad.append(f"{tag} T_cycles: {got['T_cycles']} != the closed form {T}")
        if abs(float(got["latency_ns"]) - T * 1000.0 / float(r["fmax_mhz"])) > 5e-4:
            bad.append(f"{tag} latency_ns: {got['latency_ns']} != {T} / {r['fmax_mhz']} MHz")
        rowsUsed = K if V == 1 else R                              # the P29 used extent of gemm_top
        passes = M * V * H
        if int(got["passes"]) != passes:
            bad.append(f"{tag} passes: {got['passes']} != {passes}")
        words40 = lambda n: math.ceil(n * 8 / V2_PORT_BITS)        # noqa: E731  v2's 40-bit port words
        inBytePj = p.components["input_buffer"]["energy_pj_per_byte"]
        bramPj = p.fabric["block_ram"]["energy_pj_per_access"]
        trafficV2 = passes * (words40(R) + words40(C)) * bramPj           # v2: 40-bit words at the full extent
        newTraffic = lambda rows: passes * (rows + C) * p.inputBitSlices * p.portEnergyPjPerBit   # noqa: E731
        wantFull = dict(e_pass_pj=float(r["e_pass_pj"]), e_traffic_pj=newTraffic(R),
                        e_total_pj=float(r["e_total_pj"]) - (trafficV2 - newTraffic(R)))
        for k in PASS_COLS:                                        # 1 the full extent: v2's pass energy, the traffic per bit
            rel = close(wantFull[k], float(full[k]))
            worst = max(worst, rel)
            if rel > 1e-12:
                bad.append(f"{tag} full-extent {k}: {wantFull[k]!r} != {full[k]!r}")
        savePass = passes * BUFFER_ACCESSES["input_byte"] * (R - rowsUsed) * inBytePj
        want = dict(e_pass_pj=float(r["e_pass_pj"]) - savePass, e_traffic_pj=newTraffic(rowsUsed),
                    e_total_pj=float(r["e_total_pj"]) - savePass - (trafficV2 - newTraffic(rowsUsed)))
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
        elif close(want["e_total_pj"], float(got["e_total_pj"])) > 1e-12:
            bad.append(f"{tag}: rows used == R but e_total {got['e_total_pj']!r} != v2's re-priced {want['e_total_pj']!r}")
    assert not bad, f"{len(bad)} differences, first {bad[:5]}"
    assert cheaper == 12, f"{cheaper} rows read fewer rows than R, expected 12 (K < R with V = 1)"
    v2Cycles = sorted(int(r["T_cycles"]) for r in rows)
    print(f"  fc rows  PASS  48/48 rows: {len(INT_COLS)} integer + {len(FLOAT_COLS)} float columns equal to v2; the "
          f"full extent reproduces v2's pass energy with the port traffic re-priced per bit (2026-10-23); the used extent "
          f"equals that minus the unread rows' input cost on every row (max relative float difference {worst:.1e}); {cheaper} rows read "
          f"fewer rows than R and are {100 * min(savings):.1f}-{100 * max(savings):.1f} % cheaper; cycles == the "
          f"bit-plane closed form 18 + L_W on every row (v2's pre-bit-plane cycles were {v2Cycles[0]}..{v2Cycles[-1]})")
    print("fc rows: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
