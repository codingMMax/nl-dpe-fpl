"""area.py — area and Fmax pricing (docs/simulator.md §4.7, docs/experiment.md §5).

Crossbars are exact: count x the arch-XML `wc` tile footprint. CLB and memory come from VTR per row; the memory blocks
are also predicted from each module's memory list. One grid cell = 66,117 MWTA (the routing-aware CLB tile, 2239 um^2);
a memory tile is 1 x 2 cells. Latency in ns = cycles x 1000 / Fmax.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from simulator.core.cycleMath import ceilDiv

V3 = Path(__file__).resolve().parents[2]
ARCH_DIR = V3 / "vtr" / "arch"
MWTA_PER_CELL = 66117
MEM_TILE_CELLS = 2
_WC_RE = re.compile(r'<tile name="wc"\s+height="(\d+)"\s+width="(\d+)"')
_DP_RE = re.compile(r'<mode name="mem_(\d+)x(\d+)_dp">')


def wcFootprint(R: int, C: int) -> tuple:
    """(width, height) of the wc tile in grid cells, from v3/vtr/arch/nl_dpe_<R>x<C>_auto.xml."""
    path = ARCH_DIR / f"nl_dpe_{R}x{C}_auto.xml"
    m = _WC_RE.search(path.read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"no wc tile in {path}")
    return int(m.group(2)), int(m.group(1))


def areaWcMwta(nCrossbars: int, R: int, C: int) -> int:
    w, h = wcFootprint(R, C)
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
    Checked against VPR's packed netlist at the head smoke point (BERT-Tiny 128x128: 70 memories, 376 blocks)."""
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


def withVtr(cycles: int, R: int, C: int, m: dict) -> dict:
    """Area and latency of a row from its VTR metrics: area = CLB + wc tiles + memory tiles; latency = cycles / Fmax."""
    w, h = wcFootprint(R, C)
    return dict(fmax_mhz=round(m["fmax_mhz"], 3), clb_count=m["clb"], mem_blocks=m["memory"], wc_vtr=m["wc"],
                area_mem_mwta=m["memory"] * MEM_TILE_CELLS * MWTA_PER_CELL,
                area_total_mwta=(m["clb"] + m["wc"] * w * h + MEM_TILE_CELLS * m["memory"]) * MWTA_PER_CELL,
                latency_ns=round(cycles * 1000.0 / m["fmax_mhz"], 3))
