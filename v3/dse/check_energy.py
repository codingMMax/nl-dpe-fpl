#!/usr/bin/env python3
"""check_energy.py — is every priced row priced by the energy rule? (docs/experiment.md §5, simulator/platform/prices.py)

The rule, restated here from the raw price table (simulator/platform/nl_dpe.json) and not taken from prices.py: a
crossbar pass fires the whole R x C array every time and reads only the rows in use (user, 2026-10-17):

  pass energy  = P x cell_pj x R x C + acam_pj x C + 3 x rows_used x in_byte_pj + 2 x C x out_byte_pj
  port traffic = (ceil(8 rows_used / port) + ceil(8 C / port)) x bram_pj          inputs by the rows in use, outputs full
  CLB adds / compares = count x add_pj / compare_pj;   memory = ceil(8 bytes / port) x bram_pj per direction

Checks (every failure is listed; the exit status is 1 if there is one):
  1  one pass at the boundaries: prices.py equals the formula for rows used 1, 2, 5, 6, R/2, R-1, R on all the configs;
     the cells, the ACAM and the output side do not move with the rows used; 0 and R+1 rows are rejected; an empty
     ledger costs nothing; a ledger with two keys costs the sum of its keys
  2  every row of results/fc_priced_results.csv and results/attn_priced_results.csv: the simulator's ledger for the row's
     workload, priced by the formula, equals the row's energy terms (the attention terms are stored to 0.01 pJ, the FC
     terms exactly) and its pass count; an attention row's role columns equal the formula per role
  3  how far the rule moved the rows: the same ledger priced with every pass reading all R rows
Usage: python3 v3/dse/check_energy.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse import data  # noqa: E402
from dse.space import CONFIGS  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.records import Counts  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402
from simulator.models.fc import fc  # noqa: E402
from simulator.platform import platform as platformModule  # noqa: E402
from simulator.platform import prices  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402

TABLE = json.loads((V3 / "simulator" / "platform" / "nl_dpe.json").read_text())
TILE, FABRIC = TABLE["compute_tile"], TABLE["fabric"]
P, PORT = TILE["input_bit_slices"], TILE["port_width_bits"]
CELL, ACAM = TILE["crossbar"]["energy_pj_per_cell"], TILE["acam"]["energy_pj_per_col"]
IN_BYTE, OUT_BYTE = TILE["input_buffer"]["energy_pj_per_byte"], TILE["output_buffer"]["energy_pj_per_byte"]
BRAM, BRAM_BITS = FABRIC["block_ram"]["energy_pj_per_access"], FABRIC["block_ram"]["port_width_bits"]
ADD, COMPARE = FABRIC["clb"]["add_pj"], FABRIC["clb"]["compare_pj"]
IN_ACCESSES, OUT_ACCESSES = 3, 2          # buffer accesses per input / output byte (platform.BUFFER_ACCESSES)
ROLES = ("gemm", "qk", "sm", "pv", "acc")  # the roles an attention row prices apart (simulator/platform/rows.py)
TERMS = ("e_pass_pj", "e_traffic_pj", "e_adds_pj", "e_cmp_pj", "e_mem_pj", "e_total_pj")
assert platformModule.BUFFER_ACCESSES == {"input_byte": IN_ACCESSES, "output_byte": OUT_ACCESSES}, \
    "the buffer accesses per byte changed in platform.py: review this check's formula"


def words(n_bytes: int, bits: int) -> int:
    return math.ceil(n_bytes * 8 / bits)


def pass_pj(R: int, C: int, rows: int) -> float:
    return P * CELL * R * C + ACAM * C + IN_ACCESSES * rows * IN_BYTE + OUT_ACCESSES * C * OUT_BYTE


def traffic_pj(R: int, C: int, rows: int) -> float:
    return (words(rows, PORT) + words(C, PORT)) * BRAM


def priced(counts: Counts, R: int, C: int, full: bool = False) -> dict:
    """The formula on one ledger; full = every pass reads all R rows (the rule before 2026-10-17)."""
    e = dict(e_pass_pj=0.0, e_traffic_pj=0.0)
    for (_mode, rows, _cols), n in counts.passes.items():
        r = R if full else rows
        e["e_pass_pj"] += n * pass_pj(R, C, r)
        e["e_traffic_pj"] += n * traffic_pj(R, C, r)
    e["e_adds_pj"] = counts.clbAdd * ADD
    e["e_cmp_pj"] = counts.clbCompare * COMPARE
    e["e_mem_pj"] = (words(counts.memReadBytes, BRAM_BITS) + words(counts.memWriteBytes, BRAM_BITS)) * BRAM
    e["e_total_pj"] = sum(e.values())
    return e


def check_boundaries(fails: list) -> int:
    n = 0
    for R, C in CONFIGS:
        p = Platform("nl_dpe", rows=R, cols=C)
        full = prices.passEnergyPj(p, R)
        for rows in sorted({1, 2, 5, 6, R // 2, R - 1, R}):
            got, tag = prices.passEnergyPj(p, rows), f"{R}x{C} rows {rows}"
            n += 1
            if not math.isclose(got["total"], pass_pj(R, C, rows), rel_tol=1e-12):
                fails.append(f"{tag}: pass {got['total']!r} != {pass_pj(R, C, rows)!r}")
            if not math.isclose(prices.passTrafficEnergyPj(p, rows), traffic_pj(R, C, rows), rel_tol=1e-12):
                fails.append(f"{tag}: traffic {prices.passTrafficEnergyPj(p, rows)!r} != {traffic_pj(R, C, rows)!r}")
            for k in ("crossbar", "acam", "output_buffer"):
                if got[k] != full[k]:
                    fails.append(f"{tag}: {k} moves with the rows used")
            if rows < R and not got["input_buffer"] < full["input_buffer"]:
                fails.append(f"{tag}: the input buffer does not get cheaper with fewer rows")
        for bad in (0, R + 1):
            try:
                prices.passEnergyPj(p, bad)
            except ValueError:
                continue
            fails.append(f"{R}x{C}: {bad} rows used was accepted")
    p = Platform("nl_dpe", rows=256, cols=128)
    if any(v != 0 for v in prices.energy(Counts(), p).values()):
        fails.append("an empty ledger costs energy")
    two = Counts(passes=Counter({("EXP", 64, 64): 3, ("REGULAR", 200, 128): 2}), clbAdd=7, clbCompare=5,
                 memReadBytes=11, memWriteBytes=3)
    got, want = prices.energy(two, p), priced(two, 256, 128)
    for k in TERMS:
        if not math.isclose(got[k], want[k], rel_tol=1e-12):
            fails.append(f"a two-key ledger, {k}: {got[k]!r} != {want[k]!r}")
    return n


def attention_run(r: dict):
    g, _ = attentionHead(headParams(int(r["n"]), int(r["d_model"]), int(r["d"]), int(r["R"]), int(r["C"])),
                         withValues=False)
    return run(g)


def fc_run(r: dict):
    R, C = int(r["R"]), int(r["C"])
    return run(fc(int(r["K"]), int(r["N"]), R, C, M=int(r["M"]), platform=Platform("nl_dpe", rows=R, cols=C),
                  passExtent=r["pass_extent"], withValues=False))


def check_rows(family: str, rows: list, make_run, tol_abs: float | None, fails: list) -> dict:
    """tol_abs: the stored precision of a term in pJ (attention 0.01 -> 0.0051), None = exact (relative 1e-12)."""
    worst, moved = dict.fromkeys(TERMS, 0.0), []
    for r in rows:
        R, C = int(r["R"]), int(r["C"])
        tag = f"{family} {r.get('shape', '')} {r['config']}" + ("" if family == "attention" else
                                                               f" {r['M']}x{r['K']}x{r['N']}")
        rep = make_run(r)
        want = priced(rep.totalCounts, R, C)
        for k in TERMS:
            got = float(r.get(k, 0) or 0)
            diff = abs(got - want[k])
            worst[k] = max(worst[k], diff)
            if (diff > tol_abs) if tol_abs is not None else not math.isclose(got, want[k], rel_tol=1e-12, abs_tol=1e-300):
                fails.append(f"{tag} {k}: row {got!r} != formula {want[k]!r}")
        if int(r["passes"]) != rep.totalCounts.nPasses:
            fails.append(f"{tag} passes: {r['passes']} != {rep.totalCounts.nPasses}")
        if family == "attention":
            byRole = {}
            for inst, counts in rep.counts.items():
                byRole[rep.roles[inst]] = byRole.get(rep.roles[inst], 0.0) + priced(counts, R, C)["e_total_pj"]
            for role in ROLES:
                if abs(float(r[f"e_{role}_pj"]) - byRole.get(role, 0.0)) > 0.0051:
                    fails.append(f"{tag} e_{role}_pj: row {r[f'e_{role}_pj']} != formula {byRole.get(role, 0.0):.4f}")
            if any(v != 0 for k, v in byRole.items() if k not in ROLES):
                fails.append(f"{tag}: energy in a role the row does not carry: "
                             f"{ {k: v for k, v in byRole.items() if k not in ROLES and v} }")
        moved.append((1 - want["e_total_pj"] / priced(rep.totalCounts, R, C, full=True)["e_total_pj"], r["config"]))
    return dict(worst=worst, moved=moved)


def report(family: str, res: dict, n: int) -> None:
    moved = res["moved"]
    changed = [m for m in moved if m[0] > 1e-12]
    lo, hi = min(changed), max(changed)
    print(f"  {family:9s} {n} rows: largest difference per term " +
          ", ".join(f"{k[2:-3]} {v:.1e}" for k, v in res["worst"].items()) + " pJ")
    print(f"  {'':9s} the rule moved {len(changed)} of {n} rows against reading all R rows on every pass: "
          f"{100 * lo[0]:.1f} % ({lo[1]}) to {100 * hi[0]:.1f} % ({hi[1]}) cheaper; "
          f"{n - len(changed)} unchanged")


def main() -> int:
    fails: list = []
    nb = check_boundaries(fails)
    fc_rows, att_rows = data.fcRows(), data.attentionRows()
    fcRes = check_rows("fc", fc_rows, fc_run, None, fails)
    attRes = check_rows("attention", att_rows, attention_run, 0.0051, fails)
    print(f"check_energy: {nb} boundary cases on {len(CONFIGS)} configs, {len(fc_rows)} FC rows, {len(att_rows)} "
          f"attention rows")
    report("fc", fcRes, len(fc_rows))
    report("attention", attRes, len(att_rows))
    if fails:
        print(f"check_energy: FAIL, {len(fails)} difference(s), first {fails[:6]}")
        return 1
    print("check_energy: ALL PASS (every priced row equals the formula on the simulator's ledger)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
