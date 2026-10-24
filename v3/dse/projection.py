#!/usr/bin/env python3
"""projection.py — the DSE projected onto the bit-plane design before its VTR run (user, 2026-10-23: "assuming the
Fmax does not change, CLB does not change, the BRAM and DPE count are predicted (estimated) from simulator"). Every
number it writes is an estimate: no bit-plane netlist has been placed and routed.

Each of the 48 attention builds and the 32 FC rows is priced by the code that prices the DSE (simulator/platform/
rows.py attentionRow, fcRow) on the bit-plane simulator (its cycles, energy and crossbar count), with the build's VTR
metrics substituted:
  held       fmax_mhz (and the seeds' Fmax) and clb_count: the build's previous (2026-10-16) VTR row (dse/results/
             attn_vtr_results.csv; FC: v2's run, dse/results/fc_vtr_results.csv)
  predicted  wc_vtr = the simulator's crossbar count; mem_blocks = area.predictedMemBlocks of the simulator's memories
             (FC: 0, the bit-plane GEMM has no memories; v2's FC rows carry the blocks of its old tree)
Output, with the priced tables' columns: dse/results/projection/{attn,fc}_priced_results.csv. The rankings and the
figures then run unchanged on that folder (dse/data.py DSE_RESULTS):
  DSE_RESULTS=v3/dse/results/projection python3 v3/dse/rank.py        (then breakdown.py, plot_dse.py)
and dse/compare_page.py sets the two folders side by side.
Usage: python3 v3/dse/projection.py
       python3 v3/dse/projection.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse import data  # noqa: E402
from dse.attn_rows import load_vtr  # noqa: E402
from dse.space import CONFIGS, FC_PASS_EXTENT, FC_SHAPES, SHAPES  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.models.fc import fc  # noqa: E402
from simulator.platform import area  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402
from simulator.platform.rows import UM2_PER_MWTA, attentionRow, fcRow  # noqa: E402

OLD = V3 / "dse" / "results"                      # the previous tables (2026-10-16 run): the held metrics and the old side
OUT = OLD / "projection"
ATT_VTR, FC_VTR = OLD / "attn_vtr_results.csv", OLD / "fc_vtr_results.csv"
ATT_OLD, FC_OLD = OLD / "attn_priced_results.csv", OLD / "fc_priced_results.csv"


def held(m: dict, wc: int, memory: int) -> dict:
    """A build's previous (2026-10-16) VTR metrics with its crossbar and memory counts replaced by the simulator's."""
    return dict(m, wc=wc, memory=memory)


def attentionProjected(vtr: dict) -> list:
    """The 48 attention rows (attn_rows.py's order): the simulator's row, then priced with the held metrics."""
    rows = []
    for name, n, dm, d in SHAPES:
        for R, C in CONFIGS:
            key = (name, R, C)
            if key not in vtr:
                raise SystemExit(f"{name}@{R}x{C}: no previous (2026-10-16) VTR row to hold")
            sim = attentionRow(name, n, dm, d, R, C)
            rows.append(attentionRow(name, n, dm, d, R, C,
                                     {key: held(vtr[key], sim["crossbars"], sim["mem_blocks_model"])}))
    return rows


def fcMemBlocks(R: int, C: int, shape: tuple, platform: Platform) -> int:
    """The memory blocks area.predictedMemBlocks gives the FC GEMM's memories."""
    M, K, N = shape
    rep = run(fc(K, N, R, C, M=M, platform=platform, passExtent=FC_PASS_EXTENT, withValues=False))
    return area.predictedMemBlocks(rep.resources.values(), R, C)


def fcProjected(vtr: dict) -> list:
    """The 32 FC rows (fc_rows.py's order), the same way."""
    rows = []
    for R, C in CONFIGS:
        p = Platform("nl_dpe", rows=R, cols=C)
        for shape in FC_SHAPES:
            key = (f"{R}x{C}", "gemm", *shape)
            if key not in vtr:
                raise SystemExit(f"FC {'x'.join(map(str, shape))}@{R}x{C}: no previous (2026-10-16) VTR row to hold")
            sim = fcRow(R, C, shape, p, passExtent=FC_PASS_EXTENT)
            rows.append(fcRow(R, C, shape, p, {key: held(vtr[key], sim["dpe_count"], fcMemBlocks(R, C, shape, p))},
                              passExtent=FC_PASS_EXTENT))
    return rows


def write(rows: list, old: Path, out: Path) -> None:
    """The rows under the old table's header (attn_rows.py's column order; the two must agree)."""
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(old, newline="") as f:
        header = next(csv.reader(f))
    if keys != header:
        raise SystemExit(f"{out.name}: columns {keys} differ from {old.name}'s {header}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def priced(T: int, eTotal: float, R: int, C: int, m: dict) -> dict:
    """The VTR columns of a row as rows.py prices them (area.withVtr, then mm^2 and EDAP), for the self-test."""
    out = area.withVtr(T, R, C, m)
    out["area_total_mm2"] = round(out["area_total_mwta"] * UM2_PER_MWTA / 1e6, 4)
    out["edap"] = round(eTotal * out["latency_ns"] * out["area_total_mm2"], 1)
    return out


def _self_test() -> None:
    ok = 0
    attVtr, fcVtr = load_vtr([ATT_VTR]), area.vtrTable(FC_VTR)
    # 1 today's priced tables follow from their inputs by this pricing (the old and the projected side share it):
    #   every VTR column exactly; EDAP within the rounding of e_total_pj (attention keeps 2 decimals) and its own 0.05
    tables = ((data.attentionRows(ATT_OLD), lambda r: attVtr[(r["shape"], int(r["R"]), int(r["C"]))]),
              (data.fcRows(FC_OLD), lambda r: fcVtr[(r["config"], "gemm", int(r["M"]), int(r["K"]), int(r["N"]))]))
    for rows, m_of in tables:
        for r in rows:
            e = float(r["e_total_pj"])
            p = priced(int(r["T_cycles"]), e, int(r["R"]), int(r["C"]), m_of(r))
            for k in ("fmax_mhz", "clb_count", "mem_blocks", "wc_vtr", "area_mem_mwta", "area_total_mwta",
                      "latency_ns", "area_total_mm2"):
                assert float(r[k]) == p[k], (r["config"], k, r[k], p[k])
            assert abs(float(r["edap"]) - p["edap"]) <= float(r["edap"]) * 0.005 / e + 0.05, (r["config"], r["edap"])
            ok += 1
    # 2 held(): the two counts replaced, the input untouched, every other metric kept
    m = dict(fmax_mhz=100.0, clb=5, memory=7, wc=9, seeds="11;13;17", fmax_seed={11: 99.0})
    h = held(m, 3, 0)
    assert (h["wc"], h["memory"]) == (3, 0) and (m["wc"], m["memory"]) == (9, 7)
    assert {k: v for k, v in h.items() if k not in ("wc", "memory")} == {k: v for k, v in m.items()
                                                                          if k not in ("wc", "memory")}
    ok += 1
    p = priced(1000, 1.0, 256, 128, h)                  # 3 a build with no memory block: no memory area
    w, hh = area.footprint(256, 128)
    assert p["area_mem_mwta"] == 0 and p["area_total_mwta"] == (5 + 3 * w * hh) * area.MWTA_PER_CELL
    assert p["latency_ns"] == 10000.0; ok += 1
    for fn in (attentionProjected, fcProjected):        # 4 a build without a previous (2026-10-16) VTR row stops the run
        try:
            fn({})
        except SystemExit:
            ok += 1
        else:
            raise AssertionError(f"{fn.__name__} ran without VTR rows")
    # 5 the projected rows: the simulator's columns untouched, the held metrics the old ones, the counts the
    #   simulator's, the area and the latency by hand (row-parallel tiles: clb + wc W H + 2 mem cells)
    at, fcr = attentionProjected(attVtr), fcProjected(fcVtr)
    assert len(at) == len(SHAPES) * len(CONFIGS) and len(fcr) == len(FC_SHAPES) * len(CONFIGS)
    for r in at:
        sim = attentionRow(r["shape"], r["n"], r["d_model"], r["d"], r["R"], r["C"])
        m = attVtr[(r["shape"], r["R"], r["C"])]
        assert all(r[k] == v for k, v in sim.items()), r["shape"]
        assert (r["wc_vtr"], r["mem_blocks"], r["mem_check"]) == (sim["crossbars"], sim["mem_blocks_model"], "equal")
        assert (r["clb_count"], r["fmax_mhz"], r["vtr_seeds"]) == (m["clb"], round(m["fmax_mhz"], 3), m["seeds"])
        assert all(r[f"fmax_s{s}"] == f for s, f in m["fmax_seed"].items())
        w, hh = area.footprint(r["R"], r["C"])
        assert r["area_total_mwta"] == (m["clb"] + r["crossbars"] * w * hh + 2 * r["mem_blocks"]) * 66117
        assert r["latency_ns"] == round(r["T_cycles"] * 1000 / m["fmax_mhz"], 3); ok += 1
    for r in fcr:
        m = fcVtr[(r["config"], "gemm", r["M"], r["K"], r["N"])]
        assert (r["wc_vtr"], r["mem_blocks"]) == (r["dpe_count"], 0), r["config"]   # no memory in the bit-plane GEMM
        assert (r["clb_count"], r["fmax_mhz"]) == (m["clb"], round(m["fmax_mhz"], 3))
        w, hh = area.footprint(r["R"], r["C"])
        assert r["area_total_mwta"] == (m["clb"] + r["dpe_count"] * w * hh) * 66117; ok += 1
    print(f"projection self-test: ALL PASS ({ok} checks)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return 0
    at, fcr = attentionProjected(load_vtr([ATT_VTR])), fcProjected(area.vtrTable(FC_VTR))
    write(at, ATT_OLD, OUT / ATT_OLD.name)
    write(fcr, FC_OLD, OUT / FC_OLD.name)
    print(f"projection: {len(at)} attention builds, {len(fcr)} FC rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
