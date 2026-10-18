"""breakdown.py — what a run's time, energy and area are made of (docs/experiment.md §9; LEDGER item 19 S-a).

  latencyByPhase    the head's cycles split at the end of the projection stream (the last output word of a gemm-role
                    instance): the projections' share and the core that follows
  energyByRole      pJ per role and per term (passes, adds, compares, traffic, memory), each instance priced on its own
                    counts, plus the role's crossbar passes
  areaByComponent   the VTR area split into CLB cells, crossbar tiles and memory tiles, with each share
The functions read a Report and the workload's parts; nothing here re-runs a model.
"""
from __future__ import annotations

from simulator.platform import area, prices
from simulator.platform.platform import Platform


def projectionEnd(rep) -> int | None:
    """The cycle at which the last projection output leaves the GEMMs (the last `out` window of a gemm-role instance),
    or None without projections."""
    times = [e.time for e in rep.events
             if e.kind == "win" and e.port == "out" and rep.roles.get(e.instance) == "gemm"]
    return max(times) if times else None


def latencyByPhase(rep, parts: dict) -> dict:
    """cycles = done - start of the head; the projection stream's end and share; the core after it."""
    ctrl = parts["ctrl"]
    cycles = ctrl.done - ctrl.tStart
    end = projectionEnd(rep)
    stream = None if end is None else end - ctrl.tStart
    return dict(cycles=cycles, projection_end=stream, projection_share=None if stream is None else stream / cycles,
                core_after_projections=None if stream is None else cycles - stream)


def energyByRole(rep, platform: Platform) -> dict:
    """role -> {e_pass_pj, e_adds_pj, e_cmp_pj, e_traffic_pj, e_mem_pj, e_total_pj, passes}."""
    out = {}
    for instance, counts in rep.counts.items():
        role = rep.roles[instance]
        e = prices.energy(counts, platform)
        total = out.setdefault(role, {k: 0.0 for k in e} | {"passes": 0})
        for k, v in e.items():
            total[k] += v
        total["passes"] += counts.nPasses
    return out


def areaByComponent(clbCount: int, crossbars: int, memBlocks: int, R: int, C: int) -> dict:
    """The VTR area in grid cells and shares: CLB cells, crossbar tiles (count x the wc footprint), memory tiles."""
    w, h = area.wcFootprint(R, C)
    cells = dict(clb=clbCount, xbar=crossbars * w * h, mem=area.MEM_TILE_CELLS * memBlocks)
    total = sum(cells.values())
    return dict(cells=cells, total_cells=total, total_mwta=total * area.MWTA_PER_CELL,
                shares={k: v / total for k, v in cells.items()}, tile_cells=w * h)
