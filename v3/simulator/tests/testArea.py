#!/usr/bin/env python3
"""testArea.py — the crossbar tile (simulator/platform/area.py; docs/experiment.md §5.1, user 2026-10-22).

  shape      tileShape at its edges: pins exactly on a multiple of 60 (H = k, not k + 1) and one over (k + 1); an area
             of exactly k columns (W = k) and one MWTA over (k + 1); the larger pin side sets H (the outputs when
             C > R); a 1-pin, 1-MWTA block is 1 x 1; no pins, no area, a negative, NaN or infinite area is refused
  logic      logicAreaUm2 on the 9 arrays equals nl_dpe/area_power.py's formula (written out here, 1e-9 relative) and
             COFFE's logic_um2 (coffe_flow_22nm/sweep_summary_full.csv, quoted below, to its 0.1 um^2)
  xbar       localCrossbarUm2 equals the CSV's total_um2 for R = 64, 128, 256, 512; any other R is refused
  pins       the tile's 7 control and 5 status pins are the dpe block's ports besides data, clock and reset (the arch
             XML's nl_dpe pb_type and its wc tile: I = 40 + 7, O = 40 + 5 at today's 40-bit data)
  footprint  the 9 row-parallel tiles worked out by hand, and again from the quoted CSV with integer pin arithmetic;
             C > R (the outputs set H); the 12 legacy tiles (v2's arch XMLs); an unknown rule or array is refused;
             areaWcMwta and withVtr price count x W x H x 66,117 MWTA under both rules
The legacy table is also proven by testFcRows.py and checkEquivalence.py, which reproduce v2's area columns with it.
Run: python3 v3/simulator/tests/testArea.py
"""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
from simulator.platform import area  # noqa: E402

CELL = 66117                                  # MWTA per grid cell (the CLB tile, 2239 um^2, routing included)
UM2_PER_MWTA = 0.033864
# coffe_flow_22nm/sweep_summary_full.csv (commit 0a750e4): R, C, logic_um2 (rounded to 0.1), total_um2 (the local
# input crossbar for R + 7 inputs)
COFFE = ((64, 64, 11300.0, 221.171477274), (128, 64, 12198.0, 679.785877064), (128, 128, 24041.8, 679.785877064),
         (256, 64, 13993.9, 2080.0886326), (256, 128, 27279.5, 2080.0886326), (256, 256, 53850.5, 2080.0886326),
         (512, 64, 17585.8, 7269.61884854), (512, 128, 33754.8, 7269.61884854), (512, 256, 66092.9, 7269.61884854))
# (W, H) by hand: H = ceil(max(R + 7, C + 5) / 60); W = ceil((logic + crossbar) / (H x 2239 um^2))
#   64x64    71 pins -> 2;  11,521 / 4,478  = 2.57 -> 3      256x256  263 -> 5;  55,931 / 11,195 = 4.996 -> 5
#   128x64  135 -> 3;  12,878 / 6,717  = 1.92 -> 2      512x64   519 -> 9;  24,855 / 20,151 = 1.23 -> 2
#   128x128 135 -> 3;  24,722 / 6,717  = 3.68 -> 4      512x128  519 -> 9;  41,025 / 20,151 = 2.04 -> 3
#   256x64  263 -> 5;  16,074 / 11,195 = 1.44 -> 2      512x256  519 -> 9;  73,363 / 20,151 = 3.64 -> 4
#   256x128 263 -> 5;  29,360 / 11,195 = 2.62 -> 3
ROW_PARALLEL = {(64, 64): (3, 2), (128, 64): (2, 3), (128, 128): (4, 3), (256, 64): (2, 5), (256, 128): (3, 5),
                (256, 256): (5, 5), (512, 64): (2, 9), (512, 128): (3, 9), (512, 256): (4, 9)}
# v2's arch XMLs (vtr/arch/nl_dpe_<R>x<C>_auto.xml, <tile name="wc" height width>), as (W, H)
LEGACY = {(128, 64): (1, 5), (128, 128): (1, 11), (128, 256): (3, 7), (256, 64): (2, 3), (256, 128): (3, 4),
          (256, 256): (4, 6), (512, 64): (2, 4), (512, 128): (3, 5), (512, 256): (5, 6), (1024, 64): (1, 11),
          (1024, 128): (3, 7), (1024, 256): (5, 8)}


def refused(f) -> bool:
    try:
        f()
    except (ValueError, OverflowError):
        return True
    return False


def testShape() -> int:
    n = 0
    for k in (1, 2, 9):                                        # pins on a multiple of 60, and one over
        assert area.tileShape(60 * k, 1, 1.0) == (1, k); n += 1
        assert area.tileShape(60 * k + 1, 1, 1.0) == (1, k + 1); n += 1
        assert area.tileShape(1, 60 * k, 1.0) == (1, k); n += 1           # the output side alone
    for h, k in ((1, 1), (1, 2), (3, 4), (9, 2)):              # an area of exactly k columns of height h, and 1 MWTA over
        pins = 60 * h
        assert area.tileShape(pins, 5, float(k * h * CELL)) == (k, h); n += 1
        assert area.tileShape(pins, 5, float(k * h * CELL + 1)) == (k + 1, h); n += 1
    assert area.tileShape(71, 133, 1.0) == (1, 3); n += 1     # C > R: the outputs set H
    assert area.tileShape(133, 71, 1.0) == (1, 3); n += 1
    assert area.tileShape(1, 1, 1.0) == (1, 1); n += 1
    for bad in ((0, 5, 1.0), (5, 0, 1.0), (-1, 5, 1.0), (5, 5, 0.0), (5, 5, -1.0), (5, 5, math.nan), (5, 5, math.inf)):
        assert refused(lambda: area.tileShape(*bad)), bad; n += 1
    return n


def areaPowerUm2(R: int, C: int) -> float:
    """nl_dpe/area_power.py's DPE logic area, written out (mm^2 there, um^2 here)."""
    mm2 = (0.011534 * (R / 256) * (C / 256) + 0.041431 / 256 * C + 2 * 0.0003542 / 256 * max(R, C)
           + 0.0000782 * 4 * C / 1024 + 7 * C / (7 * 256) * 0.0000989)
    return mm2 * 1e6


def testLogic() -> int:
    n = 0
    for R, C, logic, _ in COFFE:
        got = area.logicAreaUm2(R, C)
        assert abs(got / areaPowerUm2(R, C) - 1) < 1e-9, (R, C, got, areaPowerUm2(R, C)); n += 1
        assert abs(got - logic) <= 0.05 + 1e-6, (R, C, got, logic); n += 1
    # the buffers follow the larger side: one more column on a square array adds a cell column, an ACAM column, a DAC
    # column and two buffer entries
    step = area.logicAreaUm2(64, 65) - area.logicAreaUm2(64, 64)
    assert abs(step - areaPowerUm2(64, 65) + areaPowerUm2(64, 64)) < 1e-9; n += 1
    return n


def testCrossbar() -> int:
    n = 0
    for R, _, _, xbar in COFFE:
        assert area.localCrossbarUm2(R) == xbar, (R, area.localCrossbarUm2(R), xbar); n += 1
    for R in (0, 32, 65, 1024):
        assert refused(lambda: area.localCrossbarUm2(R)), R; n += 1
    return n


def testPins() -> int:
    root = ET.parse(V3 / "vtr" / "arch" / "nl_dpe_128x64_auto.xml").getroot()
    dpe = next(pb for pb in root.iter("pb_type") if pb.get("name") == "nl_dpe")
    ins = {x.get("name"): int(x.get("num_pins")) for x in dpe.findall("input")}
    outs = {x.get("name"): int(x.get("num_pins")) for x in dpe.findall("output")}
    control = sum(v for k, v in ins.items() if k not in ("data_in", "reset"))
    status = sum(v for k, v in outs.items() if k != "data_out")
    assert (control, status) == (area.TILE["control_pins"], area.TILE["status_pins"]) == (7, 5), (control, status)
    tile = next(x for x in root.iter("tile") if x.get("name") == "wc")
    sub = tile.find("sub_tile")
    pins = {x.get("name"): int(x.get("num_pins")) for x in sub if x.tag in ("input", "output")}
    assert (pins["I"], pins["O"]) == (ins["data_in"] + control, outs["data_out"] + status) == (47, 45), pins
    return 2


def testFootprint() -> int:
    n = 0
    for (R, C, logic, xbar) in COFFE:
        assert area.footprint(R, C) == ROW_PARALLEL[(R, C)], (R, C, area.footprint(R, C)); n += 1
        h = -(-max(R + 7, C + 5) // 60)                         # the same rule from the quoted CSV, pins in integers
        w = math.ceil((logic + xbar) / UM2_PER_MWTA / (h * CELL))
        assert (w, h) == ROW_PARALLEL[(R, C)], (R, C, w, h); n += 1
        used = (logic + xbar) / UM2_PER_MWTA / (w * h * CELL)
        assert used <= 1 and (w == 1 or (logic + xbar) / UM2_PER_MWTA > (w - 1) * h * CELL), (R, C, used); n += 1
    # C > R: 261 output pins set H = 5; 45,200 um^2 of logic + 221 of crossbar = 4.06 columns of 11,195 -> W = 5
    assert area.footprint(64, 256) == (5, 5); n += 1
    for (R, C), wh in LEGACY.items():
        assert area.footprint(R, C, rule="legacy") == wh, (R, C); n += 1
    for bad in (lambda: area.footprint(64, 64, rule="legacy"), lambda: area.footprint(128, 64, rule="xml"),
                lambda: area.footprint(1024, 64), lambda: area.footprint(96, 64)):
        assert refused(bad); n += 1
    m = dict(fmax_mhz=100.0, clb=1000, memory=10, wc=7)
    for rule, table in (("row_parallel", ROW_PARALLEL), ("legacy", LEGACY)):
        w, h = table[(256, 128)]
        assert area.areaWcMwta(7, 256, 128, rule) == 7 * w * h * CELL; n += 1
        assert area.areaWcMwta(0, 256, 128, rule) == 0; n += 1
        got = area.withVtr(5000, 256, 128, m, rule)
        assert got["area_total_mwta"] == (1000 + 7 * w * h + 2 * 10) * CELL and got["latency_ns"] == 50000.0; n += 1
    return n


def main() -> int:
    print(f"  shape     PASS  {testShape()} checks (pins and area on their boundaries, C > R, refused inputs)")
    print(f"  logic     PASS  {testLogic()} checks (area_power.py's formula and COFFE's logic on the 9 arrays)")
    print(f"  xbar      PASS  {testCrossbar()} checks (COFFE's local crossbar per R, other R refused)")
    print(f"  pins      PASS  {testPins()} checks (7 control + 5 status pins = the dpe block's other ports)")
    print(f"  footprint PASS  {testFootprint()} checks (9 row-parallel tiles by hand and from the CSV, 12 legacy, pricing)")
    print("testArea: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
