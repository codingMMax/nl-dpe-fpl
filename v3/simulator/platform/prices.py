"""prices.py — energy = counts x the frozen per-operation prices of the platform (docs/simulator.md §4.7).

Counting rules of v2/sim/spec/simulator.md §4: a crossbar pass is priced as the full array (P29: the extent changes
time, not the pass energy), with its input / output port traffic through block RAM; CLB adds and compares; memory
bytes per block-RAM access of port_width / 8 bytes. Weight programming is setup energy, never per-inference energy.
"""
from __future__ import annotations

from simulator.core.cycleMath import ceilDiv
from simulator.platform.platform import BUFFER_ACCESSES, COMPONENTS, Platform


def passEnergyPj(p: Platform, activeColumns: int | None = None) -> dict:
    """pJ per component of one pass of the whole array, plus 'total' (the ADC conversions count the active columns)."""
    nActive = p.cols if activeColumns is None else activeColumns
    events = {"bit_slice": p.inputBitSlices,
              "input_byte": BUFFER_ACCESSES["input_byte"] * p.rows,
              "output_byte": BUFFER_ACCESSES["output_byte"] * p.cols,
              "pass": 1,
              "conversion": p.inputBitSlices * nActive}
    out = {name: events[COMPONENTS[name][3]] * p.eventEnergyPj[name] for name in p.components}
    out["total"] = sum(out.values())
    return out


def passPortAccesses(p: Platform) -> tuple:
    """(input, output) port words of one pass."""
    return ceilDiv(p.rows * 8, p.portWidthBits), ceilDiv(p.cols * 8, p.portWidthBits)


def passTrafficEnergyPj(p: Platform) -> float:
    return sum(passPortAccesses(p)) * p.fabric["block_ram"]["energy_pj_per_access"]


def memoryAccesses(fabric: dict, nBytes: int) -> int:
    return ceilDiv(nBytes * 8, fabric["block_ram"]["port_width_bits"])


def memoryEnergyPj(fabric: dict, nBytes: int) -> float:
    return memoryAccesses(fabric, nBytes) * fabric["block_ram"]["energy_pj_per_access"]


def addEnergyPj(fabric: dict, n: int) -> float:
    return n * fabric["clb"]["add_pj"]


def compareEnergyPj(fabric: dict, n: int) -> float:
    return n * fabric["clb"]["compare_pj"]


def setupEnergyPj(p: Platform, nCrossbars: int) -> float:
    """Weight programming of every crossbar: ceil(R C / 5) block-RAM accesses each (charter §4.5), reported apart."""
    return nCrossbars * memoryEnergyPj(p.fabric, p.rows * p.cols)


def energy(counts, p: Platform) -> dict:
    """pJ per term for one module's (or a workload's) Counts, keyed as the DSE rows name them."""
    f = p.fabric
    n = counts.nPasses
    e = dict(e_pass_pj=n * passEnergyPj(p)["total"],
             e_adds_pj=addEnergyPj(f, counts.clbAdd),
             e_cmp_pj=compareEnergyPj(f, counts.clbCompare),
             e_traffic_pj=n * passTrafficEnergyPj(p),
             e_mem_pj=memoryEnergyPj(f, counts.memReadBytes) + memoryEnergyPj(f, counts.memWriteBytes))
    e["e_total_pj"] = sum(e.values())
    return e
