#!/usr/bin/env python3
"""breakdown.py — why attention and FC rank the crossbar configs differently (docs/experiment.md §9): per config, the
mean over each family's workloads of the area split (CLB / crossbar tiles / memory, from the VPR counts), the cycles,
Fmax, the energy split (crossbar passes / CLB adds), the crossbar count and pass count, the full-array pass price; for
attention also the cycle at which the projections end (simulator/platform/breakdown.py, from the Report's events) and
the share of a conversion crossbar's cells in use (I x I of R x C, I = min(R, C)).
Inputs: dse/results/att_priced_72.csv (attention) and the 48 FC rows rebuilt by simulator/platform/rows.py from the v2
snapshot (dse/fc_snapshot/).
Usage: python3 v3/dse/breakdown.py          # dse/results/breakdown.json, and a table on stdout
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse.space import CONFIGS, SHAPES  # noqa: E402
from simulator.platform.breakdown import areaByComponent, projectionEnd  # noqa: E402
from simulator.platform.rows import fcRow  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.platform import area, prices  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402

RESULTS = V3 / "dse" / "results"
SNAP = V3 / "dse" / "fc_snapshot"


def fcRows() -> list:
    vtr = area.vtrTable(SNAP / "vtr_results.csv")
    with open(SNAP / "dse_v2.csv", newline="") as f:
        snap = [r for r in csv.DictReader(f) if r["family"] == "gemm"]
    return [fcRow(int(r["R"]), int(r["C"]), (int(r["M"]), int(r["K"]), int(r["N"])),
                  Platform("nl_dpe", rows=int(r["R"]), cols=int(r["C"])), vtr, passExtent=r["pass_extent"])
            for r in snap]


def projectionsEnd(n: int, dm: int, d: int, R: int, C: int) -> tuple:
    """(cycles, the cycle at which the last projection output leaves the GEMMs) of one attention build."""
    g, parts = attentionHead(headParams(n, dm, d, R, C), withValues=False)
    rep = run(g)
    return parts["ctrl"].done - parts["ctrl"].tStart, projectionEnd(rep)


def perConfig(rows: list, family: str) -> dict:
    out = {}
    for R, C in CONFIGS:
        cfg = f"{R}x{C}"
        rs = [r for r in rows if r["config"] == cfg]
        price = prices.passEnergyPj(Platform("nl_dpe", rows=R, cols=C))["total"]
        tileCells = areaByComponent(0, 1, 0, R, C)["tile_cells"]

        def mean(fn):
            return st.mean(fn(r) for r in rs)

        def share(r, k):
            return areaByComponent(int(r["clb_count"]), int(r["wc_vtr"]), int(r["mem_blocks"]), R, C)["shares"][k]
        o = dict(crossbars=mean(lambda r: int(r["wc_vtr"])), tile_cells=tileCells,
                 clb_count=mean(lambda r: int(r["clb_count"])),
                 area_clb=mean(lambda r: share(r, "clb")), area_xbar=mean(lambda r: share(r, "xbar")),
                 area_mem=mean(lambda r: share(r, "mem")), cycles=mean(lambda r: int(r["T_cycles"])),
                 fmax_mhz=mean(lambda r: float(r["fmax_mhz"])),
                 e_pass=mean(lambda r: float(r["e_pass_pj"]) / float(r["e_total_pj"])),
                 e_adds=mean(lambda r: float(r["e_adds_pj"]) / float(r["e_total_pj"])),
                 passes=mean(lambda r: float(r["e_pass_pj"]) / price), pass_price_pj=price)
        if family == "attention":
            I = min(R, C)
            o["used_cells"] = I * I / (R * C)
        out[cfg] = o
    return out


def main() -> int:
    with open(RESULTS / "att_priced_72.csv", newline="") as f:
        att = list(csv.DictReader(f))
    res = dict(fc=perConfig(fcRows(), "fc"), attention=perConfig(att, "attention"), projections=[])
    for name, n, dm, d in SHAPES:
        for R, C in CONFIGS:
            T, end = projectionsEnd(n, dm, d, R, C)
            res["projections"].append(dict(shape=name, n=n, d_model=dm, config=f"{R}x{C}", cycles=T, end=end,
                                           stream=n * (-(-min(dm, R) // 5) + 8), core_floor=n * n // 5))
    shares = [p["end"] / p["cycles"] for p in res["projections"]]
    res["projection_share"] = dict(median=st.median(shares), min=min(shares), max=max(shares),
                                   stream_max_diff=max(abs(p["end"] - p["stream"]) for p in res["projections"]))
    (RESULTS / "breakdown.json").write_text(json.dumps(res, indent=1) + "\n")
    for family in ("fc", "attention"):
        print(f"== {family}: mean over the workloads")
        print("  config    xbars cells  CLB%  xbar%  mem%    CLBs   cycles   Fmax  passE%  addsE%   passes  pJ/pass")
        for cfg, o in res[family].items():
            print(f"  {cfg:9s} {o['crossbars']:5.0f} {o['tile_cells']:5d} {100*o['area_clb']:5.0f} {100*o['area_xbar']:6.0f}"
                  f" {100*o['area_mem']:5.0f} {o['clb_count']:7.0f} {o['cycles']:8.0f} {o['fmax_mhz']:6.1f}"
                  f" {100*o['e_pass']:6.0f} {100*o['e_adds']:7.0f} {o['passes']:8.0f} {o['pass_price_pj']:8.1f}")
    ps = res["projection_share"]
    print(f"attention: projections end at {100*ps['median']:.0f} % of the cycles (median; {100*ps['min']:.0f}-"
          f"{100*ps['max']:.0f} %), = n x (ceil(min(d_m, R)/5) + 8) within {ps['stream_max_diff']} cycles -> "
          f"{RESULTS / 'breakdown.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
