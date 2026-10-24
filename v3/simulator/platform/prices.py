"""prices.py — energy = counts x the frozen per-operation prices of the platform (docs/simulator.md §4.7).

Counting rules (v2/sim/spec/simulator.md §4, amended 2026-10-17): a crossbar pass fires the whole array every time: the
R x C cells, the ACAM output stage on every column, the output buffer and the output drain are charged in full. Only
the input side follows the rows in use (ROWS_USED of the pass, P29): the input-buffer bytes and the input port words.
Until 2026-10-17 every pass was priced as the full array on both sides, which overcharged the input side of passes
with ROWS_USED < R (every conversion pass on R > C arrays, every GEMM pass with K < R). The output extent (COLS_USED)
changes time, not energy (user: outputs full). CLB adds and compares, memory bytes per block-RAM access of
port_width / 8 bytes. Weight programming is setup energy, never per-inference energy.
The port traffic (2026-10-23, user): the crossbar's port is R bits in and C bits out per cycle, so a pass moves P planes
of the rows in use in and P planes of every column out; it is priced per bit (nl_dpe.json `port`: the block-RAM access
price per bit of its port, continuous with the pre-bit-plane 40-bit words without their rounding).
"""
from __future__ import annotations

from simulator.core.cycleMath import ceilDiv
from simulator.platform.platform import BUFFER_ACCESSES, COMPONENTS, Platform


def passEnergyPj(p: Platform, rowsUsed: int | None = None) -> dict:
    """pJ per component of one pass, plus 'total': the whole array fires (cells, ACAM on every column, output buffer,
    the ADC conversions of every column on a platform that has them); the input buffer counts the rows in use."""
    rows = p.rows if rowsUsed is None else rowsUsed
    if not 1 <= rows <= p.rows:
        raise ValueError(f"rows used {rows} outside 1..{p.rows}")
    events = {"bit_slice": p.inputBitSlices,
              "input_byte": BUFFER_ACCESSES["input_byte"] * rows,
              "output_byte": BUFFER_ACCESSES["output_byte"] * p.cols,
              "pass": 1,
              "conversion": p.inputBitSlices * p.cols}
    out = {name: events[COMPONENTS[name][3]] * p.eventEnergyPj[name] for name in p.components}
    out["total"] = sum(out.values())
    return out


def passPortWords(p: Platform, rowsUsed: int | None = None) -> tuple:
    """(input, output) port words of one pass: on the row-parallel interface P planes each way (a plane carries the rows
    in use, a drain plane every column); on a fixed-width port the words of 8 bits per row in use / per column."""
    rows = p.rows if rowsUsed is None else rowsUsed
    if p.portFixedWidthBits is None:
        return p.portWordsPerPass, p.portWordsPerPass
    return ceilDiv(rows * 8, p.portFixedWidthBits), ceilDiv(p.cols * 8, p.portFixedWidthBits)


def passPortBits(p: Platform, rowsUsed: int | None = None) -> tuple:
    """(input, output) bits one pass moves through the port: P planes of the rows in use in, P planes of every column
    out (row-parallel); on a fixed-width port the words times their width."""
    rows = p.rows if rowsUsed is None else rowsUsed
    if p.portFixedWidthBits is None:
        return rows * p.portWordsPerPass, p.cols * p.portWordsPerPass
    wi, wo = passPortWords(p, rowsUsed)
    return wi * p.portFixedWidthBits, wo * p.portFixedWidthBits


def passTrafficEnergyPj(p: Platform, rowsUsed: int | None = None) -> float:
    """The port traffic of one pass: the bits moved times the platform's price per bit."""
    return sum(passPortBits(p, rowsUsed)) * p.portEnergyPjPerBit


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
    """pJ per term for one module's (or a workload's) Counts, keyed as the DSE rows name them. The passes are priced
    per (mode, rows used, columns used) key of the ledger: the rows in use set the input side of each price."""
    f = p.fabric
    ePass = eTraffic = 0.0
    for (_mode, rowsUsed, _colsUsed), n in counts.passes.items():
        ePass += n * passEnergyPj(p, rowsUsed)["total"]
        eTraffic += n * passTrafficEnergyPj(p, rowsUsed)
    e = dict(e_pass_pj=ePass,
             e_adds_pj=addEnergyPj(f, counts.clbAdd),
             e_cmp_pj=compareEnergyPj(f, counts.clbCompare),
             e_traffic_pj=eTraffic,
             e_mem_pj=memoryEnergyPj(f, counts.memReadBytes) + memoryEnergyPj(f, counts.memWriteBytes))
    e["e_total_pj"] = sum(e.values())
    return e
