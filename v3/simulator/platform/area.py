"""area.py — area and Fmax pricing (docs/simulator.md §4.7, docs/experiment.md §5).

Crossbars are exact: count x the crossbar tile's footprint. The footprint follows the row-parallel rule of
experiment.md §5.1 (user, 2026-10-22; constants in nl_dpe_tile.json): the height from the pins (R + 7 in, C + 5 out,
60 per tile row), the width from the logic plus COFFE's local input crossbar, every cell charged as one CLB tile with
its routing. rule="legacy" is the tile of v2's arch XMLs (a frozen table), which the v2 regression tests keep. CLB and
memory come from VTR per row; the memory blocks are also predicted from each module's memory list (the arch XML's
memory modes). One grid cell = 66,117 MWTA (the routing-aware CLB tile, 2239 um^2); a memory tile is 1 x 2 cells.
Latency in ns = cycles x 1000 / Fmax.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from simulator.core.cycleMath import ceilDiv
from simulator.platform.platform import MWTA_UM2, Platform

V3 = Path(__file__).resolve().parents[2]
ARCH_DIR = V3 / "vtr" / "arch_rowpar"       # the live XMLs (the memory block is the same in v2's vtr/arch/)
TILE = json.loads((Path(__file__).resolve().parent / "nl_dpe_tile.json").read_text(encoding="utf-8"))
MWTA_PER_CELL = 66117
MEM_TILE_CELLS = 2
_DP_RE = re.compile(r'<mode name="mem_(\d+)x(\d+)_dp">')


def logicAreaUm2(R: int, C: int) -> float:
    """The crossbar block's logic (nl_dpe/area_power.py): the cells, the ACAM with its XOR and two buffers of
    max(R, C) entries at the platform's per-unit areas, plus the DACs."""
    return Platform("nl_dpe", rows=R, cols=C, bufferSize=max(R, C)).totalAreaUm2 + C * TILE["dac_um2_per_col"]


def localCrossbarUm2(R: int) -> float:
    """COFFE's local input crossbar for R + 7 inputs (the row-parallel sweep; it depends on R only)."""
    if str(R) not in TILE["local_crossbar_um2"]:
        raise ValueError(f"no COFFE local crossbar for R = {R} (nl_dpe_tile.json has R = "
                         f"{', '.join(TILE['local_crossbar_um2'])})")
    return TILE["local_crossbar_um2"][str(R)]


def blockAreaMwta(R: int, C: int) -> float:
    """The block a crossbar tile holds: its logic plus its local input crossbar, in MWTA (the XML tile's area)."""
    return (logicAreaUm2(R, C) + localCrossbarUm2(R)) / MWTA_UM2


def tileShape(pinsIn: int, pinsOut: int, areaMwta: float) -> tuple:
    """(width, height) in grid cells of a block with these pins and this area: a tile row carries 60 pins, so the
    height holds the larger pin side; the width holds the area in whole columns of that height."""
    if min(pinsIn, pinsOut) < 1 or not 0 < areaMwta < math.inf:
        raise ValueError(f"a tile needs pins and area (pins {pinsIn} in, {pinsOut} out, area {areaMwta} MWTA)")
    h = ceilDiv(max(pinsIn, pinsOut), TILE["pins_per_tile_row"])
    return math.ceil(areaMwta / (h * MWTA_PER_CELL)), h


def footprint(R: int, C: int, rule: str = "row_parallel") -> tuple:
    """(width, height) of an R x C crossbar's tile in grid cells: the row-parallel rule (R + 7 pins in, C + 5 out,
    experiment.md §5.1), or rule="legacy", the tile of v2's arch XMLs."""
    if rule == "legacy":
        key = f"{R}x{C}"
        if key not in TILE["legacy_footprint"]:
            raise ValueError(f"no legacy tile for {key}")
        w, h = TILE["legacy_footprint"][key]
        return w, h
    if rule != "row_parallel":
        raise ValueError(f"unknown tile rule {rule!r} (row_parallel, legacy)")
    return tileShape(R + TILE["control_pins"], C + TILE["status_pins"], blockAreaMwta(R, C))


def areaWcMwta(nCrossbars: int, R: int, C: int, rule: str = "row_parallel") -> int:
    w, h = footprint(R, C, rule)
    return nCrossbars * w * h * MWTA_PER_CELL


def memModes(R: int, C: int) -> tuple:
    """(words, bits) of the memory block's dual-port modes in the arch XML; a 1W1R memory maps onto one of them."""
    path = ARCH_DIR / f"nl_dpe_{R}x{C}_auto.xml"
    modes = tuple((int(a), int(b)) for a, b in _DP_RE.findall(path.read_text(encoding="utf-8")))
    if not modes:
        raise ValueError(f"no dual-port memory mode in {path}")
    return modes


def memBlocks(words: int, bits: int, modes: tuple) -> int:
    """Blocks of one memory in its cheapest dual-port mode, whole blocks deep and wide."""
    return min(ceilDiv(words, depth) * ceilDiv(bits, width) for depth, width in modes)


def predictedMemBlocks(resources, R: int, C: int) -> int:
    """The memory blocks VTR places for these Resources (an iterable): each memory in its cheapest dual-port mode, and
    a group of copies with the same addresses and write enables (Resources.memShared) packed by its summed width.
    Checked against VPR's packed netlist at the head smoke point (BERT-Tiny 128x128: 70 memories, 376 blocks; the
    pre-bit-plane RTL, 2026-10-15). The bit-plane model predicts 593 blocks there (2026-10-22), not yet checked by VPR."""
    modes, n = memModes(R, C), 0
    for r in resources:
        for (words, bits), k in r.memories.items():
            copies = r.memShared.get((words, bits), 1)
            if k % copies:
                raise ValueError(f"{k} memories of {words} x {bits} do not split into groups of {copies}")
            n += (k // copies) * memBlocks(words, bits * copies, modes)
    return n


def vtrTable(path: Path) -> dict:
    """{(config, family, M, K, N): averaged VTR metrics} from a vtr_results.csv (ok = 1, averaged lines only)."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("ok") != "1" or "seed" in r:
                continue
            out[(r["config"], r["family"], int(r["M"]), int(r["K"]), int(r["N"]))] = dict(
                fmax_mhz=float(r["fmax_mhz"]), clb=int(r["clb"]), memory=int(r["memory"]), wc=int(r["wc"]),
                seeds=len([x for x in r["seeds"].split(";") if x]), top=r["top"], run_dir=r["run_dir"])
    return out


def withVtr(cycles: int, R: int, C: int, m: dict, rule: str = "row_parallel") -> dict:
    """Area and latency of a row from its VTR metrics: area = CLB + wc tiles + memory tiles; latency = cycles / Fmax."""
    w, h = footprint(R, C, rule)
    return dict(fmax_mhz=round(m["fmax_mhz"], 3), clb_count=m["clb"], mem_blocks=m["memory"], wc_vtr=m["wc"],
                area_mem_mwta=m["memory"] * MEM_TILE_CELLS * MWTA_PER_CELL,
                area_total_mwta=(m["clb"] + m["wc"] * w * h + MEM_TILE_CELLS * m["memory"]) * MWTA_PER_CELL,
                latency_ns=round(cycles * 1000.0 / m["fmax_mhz"], 3))
