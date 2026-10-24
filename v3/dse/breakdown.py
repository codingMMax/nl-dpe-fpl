#!/usr/bin/env python3
"""breakdown.py — what the area, the energy and the cycles of each crossbar config are made of (docs/experiment.md
§9; the (b) and (c) panels of the per-family figures). Per family and config:

  area_split     the DPE tiles, the CLBs and the memory blocks, from the VPR counts (grid cells: a crossbar tile is
                 the arch XML's wc footprint, a memory block 2 cells, a CLB 1)
  energy_split   the simulator's energy ledger priced by simulator/platform/prices.py: the DPE passes (the array, the
                 ACAM, the DPE's own buffers), the CLB adds and compares, the block-RAM port traffic of the passes, and
                 the memory bytes of the FIFOs and buffers
  the means over the family's workloads of the crossbar count, the CLB count, the cycles, Fmax, the pass count and
  the mean price paid per pass (the full-array price beside it), and for attention the share of a conversion
  crossbar's cells in use (I x I of R x C) and the cycle at which the projections end (from the Report's events)

How a split is summarized over a family's shapes (user, 2026-10-17): attention takes the geometric mean of each
share over its 6 shapes; FC takes the share of the total summed over its 4 shapes, because several FC builds have no
memory block and no CLB add, and a geometric mean of a share that is 0 on one shape is 0.
Inputs: dse/data.py (the 48 priced attention builds, the 32 FC rows).
Usage: python3 v3/dse/breakdown.py          # dse/results/breakdown.json; the split tables and the means on stdout
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse import data  # noqa: E402
from dse.space import CONFIGS, SHAPES  # noqa: E402


_ROW_NE: dict = {}


def rowNE() -> dict:
    """{(n, d_model, d, R, C): N_E} of the priced table's rows: each build is traced at the mapping its VTR line
    was synthesized with (the 2026-10-16 run's lines carry 2d, the sweep's d); a build without a row takes the default."""
    if not _ROW_NE:
        for r in data.attentionRows():
            _ROW_NE[(int(r["n"]), int(r["d_model"]), int(r["d"]), int(r["R"]), int(r["C"]))] = int(r["N_E"])
    return _ROW_NE
from simulator.platform.breakdown import areaByComponent, projectionEnd  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.platform import prices  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402

RESULTS = data.RESULTS
AREA_PARTS = ("dpe", "clb", "mem")
ENERGY_PARTS = ("dpe", "clb", "traffic", "mem")


def num(r: dict, key: str) -> float:
    """A row's number; a term a family does not have (FC has no compares, no e_mem_pj) is 0."""
    v = r.get(key, 0)
    return float(v) if v not in ("", None) else 0.0


def summarize(parts: list, family: str) -> dict:
    """parts: one {part: amount} per shape -> {part: share} over the family (geometric mean of the per-shape shares
    for attention; the share of the summed amounts for FC), plus how it was done."""
    keys = list(parts[0])
    if family == "attention":
        out = {}
        for k in keys:
            xs = [p[k] / sum(p.values()) for p in parts]
            if min(xs) <= 0:
                raise ValueError(f"a {k} share of 0, no geometric mean")
            out[k] = math.exp(sum(math.log(x) for x in xs) / len(xs))
        return dict(out, method="geometric mean of the per-shape shares")
    total = sum(sum(p.values()) for p in parts)
    return dict({k: sum(p[k] for p in parts) / total for k in keys}, method="share of the total summed over the shapes")


def areaSplit(rs: list, R: int, C: int, family: str) -> dict:
    """The DPE tiles, the CLBs and the memory blocks of each build, in grid cells (areaByComponent)."""
    cells = [areaByComponent(int(r["clb_count"]), int(r["wc_vtr"]), int(r["mem_blocks"]), R, C)["cells"] for r in rs]
    return summarize([dict(dpe=c["xbar"], clb=c["clb"], mem=c["mem"]) for c in cells], family)


def energySplit(rs: list, family: str) -> dict:
    """The DPE passes, the CLB adds + compares, the block-RAM port traffic of the passes, the memory bytes."""
    return summarize([dict(dpe=num(r, "e_pass_pj"), clb=num(r, "e_adds_pj") + num(r, "e_cmp_pj"),
                           traffic=num(r, "e_traffic_pj"), mem=num(r, "e_mem_pj") + num(r, "e_ser_pj")) for r in rs],
                     family)


def projectionsEnd(n: int, dm: int, d: int, R: int, C: int) -> tuple:
    """(cycles, the cycle at which the last projection output leaves the GEMMs) of one attention build."""
    g, parts = attentionHead(headParams(n, dm, d, R, C, N_E=rowNE().get((n, dm, d, R, C))), withValues=False)   # the row's own mapping
    rep = run(g)
    return parts["ctrl"].done - parts["ctrl"].tStart, projectionEnd(rep)


def perConfig(rows: list, family: str) -> dict:
    out = {}
    for R, C in CONFIGS:
        cfg = f"{R}x{C}"
        rs = [r for r in rows if r["config"] == cfg]
        if not rs:
            raise SystemExit(f"{family}: no rows for {cfg}")
        p = Platform("nl_dpe", rows=R, cols=C)
        mean = lambda fn: st.mean(fn(r) for r in rs)           # noqa: E731
        o = dict(crossbars=mean(lambda r: int(r["wc_vtr"])), tile_cells=areaByComponent(0, 1, 0, R, C)["tile_cells"],
                 clb_count=mean(lambda r: int(r["clb_count"])), cycles=mean(lambda r: int(r["T_cycles"])),
                 fmax_mhz=mean(lambda r: float(r["fmax_mhz"])), passes=mean(lambda r: int(r["passes"])),
                 pass_price_pj=mean(lambda r: num(r, "e_pass_pj") / int(r["passes"])),      # paid, the rows in use read
                 pass_price_full_pj=prices.passEnergyPj(p)["total"],                          # every row read
                 area_split=areaSplit(rs, R, C, family), energy_split=energySplit(rs, family))
        if family == "attention":
            I = min(R, C)
            o["used_cells"] = I * I / (R * C)
        out[cfg] = o
    return out


def printSplit(res: dict, family: str, key: str, parts: tuple) -> None:
    method = res[family]["128x64"][key]["method"]
    print(f"== {family} {key.replace('_', ' ')} per config, % ({method})")
    print("  " + f"{'R x C':9s}" + "".join(f"{p:>9s}" for p in parts))
    for cfg, o in res[family].items():
        print("  " + f"{cfg:9s}" + "".join(f"{100 * o[key][p]:9.1f}" for p in parts))


def main() -> int:
    res = dict(fc=perConfig(data.fcRows(), "fc"), attention=perConfig(data.attentionRows(), "attention"),
               projections=[])
    for name, n, dm, d in SHAPES:
        for R, C in CONFIGS:
            T, end = projectionsEnd(n, dm, d, R, C)
            res["projections"].append(dict(shape=name, n=n, d_model=dm, config=f"{R}x{C}", cycles=T, end=end,
                                           stream=n * max(-(-min(dm, R) // 5) + 8, -(-8 * C // 40) + 1),
                                           core_floor=n * n // 5))     # a copy's row period: the X load or the drain
            # (stream, core_floor: the pre-bit-plane closed forms of the 2026-10-16 run; stage 5 replaces them)
    shares = [p["end"] / p["cycles"] for p in res["projections"]]
    res["projection_share"] = dict(median=st.median(shares), min=min(shares), max=max(shares),
                                   stream_max_diff=max(abs(p["end"] - p["stream"]) for p in res["projections"]))
    (RESULTS / "breakdown.json").write_text(json.dumps(res, indent=1) + "\n")
    for family in ("fc", "attention"):
        printSplit(res, family, "area_split", AREA_PARTS)
        printSplit(res, family, "energy_split", ENERGY_PARTS)
        print(f"== {family}: means over the workloads")
        print("  config    xbars cells    CLBs   cycles   Fmax   passes  pJ/pass  (full array)")
        for cfg, o in res[family].items():
            print(f"  {cfg:9s} {o['crossbars']:5.0f} {o['tile_cells']:5d} {o['clb_count']:7.0f} {o['cycles']:8.0f} "
                  f"{o['fmax_mhz']:6.1f} {o['passes']:8.0f} {o['pass_price_pj']:8.2f}  ({o['pass_price_full_pj']:7.2f})")
    ps = res["projection_share"]
    print(f"attention: projections end at {100*ps['median']:.0f} % of the cycles (median; {100*ps['min']:.0f}-"
          f"{100*ps['max']:.0f} %), = n x max(ceil(min(d_m, R)/5) + 8, ceil(8C/40) + 1) within "
          f"{ps['stream_max_diff']} cycles -> "
          f"{RESULTS / 'breakdown.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
