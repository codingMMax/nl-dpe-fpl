#!/usr/bin/env python3
"""config_table.py — the attention crossbar configs side by side (docs/experiment.md §9.3, "the config tables"), with
the critical path of every DSE build (the traced times of simulator/core/trace.py). Every number is the geometric
mean over the six heads unless said otherwise.

Table 1, the chain: every column relative to one reference config (default: the table's best ADP), and the columns
multiply exactly, because the geometric mean of a product is the product of the geometric means:
  area x (cycles / Fmax) = area x latency = ADP;  energy x ADP = EDAP

Table 2, what is inside: per head the crossbar, CLB and BRAM area and their total (mm², VPR counts,
simulator/platform/breakdown.py areaByComponent; the three add up to the row's area_total_mwta, checked on every row),
the energy (nJ, the priced ledger), the cycles on the head's critical path by kind, the head's cycles T, and Fmax.
The critical path: the head is run (timing only) and its last event, att_ctrl's done, carries the chain of waits that
set it; the chain's segments add up to T exactly and are labeled by the module (its role: gemm, qk, sm, pv, acc,
ctrl) and the kind of wait:
  DPE load     a pass's input burst (ceil(8 ROWS_USED / BUF) words into a crossbar)
  DPE compute  the P bit-slices, the shift-accumulate, the ACAM, the hand-overs to the next burst
  DPE drain    the first output word, and the drain words a consumer takes
  stream       the words of a port, or a FIFO read one word per cycle: the 40-bit paths between modules
  logic        every other named latency: trees, registers, decisions, the writers (and the start's cycle)
The parts are geometric means too, so they come close to their total but do not add up to it exactly.

Table 3, the critical path as shares of T: by kind, then by module (the mean over the six heads of each share).
Table 4, per head at two configs: the same shares, one row per head.
Table 5, FC: the critical path by kind (one pass per row: the load, the slices, the first drain word, the tree and
the serializer, the output words on the port); the mean over the four shapes of each share.

Outputs: dse/results/attn_config_table.md (the tables) and dse/results/critical_path.json (every build's critical
path: the cycles by kind, by module, by both). Inputs: dse/data.py (the priced tables) and the simulator.
Usage: python3 v3/dse/config_table.py [CONFIG ...] [--ref CONFIG] [--out FILE]     # default: the 12 configs,
         rows grouped by C with R inside; prints the tables
       python3 v3/dse/config_table.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse import data  # noqa: E402
from dse import rank  # noqa: E402
from dse.space import CONFIGS, FC_PASS_EXTENT, FC_SHAPES, SHAPES  # noqa: E402
from simulator.core.cycleMath import ceilDiv  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.trace import KINDS, breakdown, segments  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402
from simulator.models.fc import fc  # noqa: E402
from simulator.platform import area  # noqa: E402
from simulator.platform.breakdown import areaByComponent  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402

DEFAULT = tuple(f"{R}x{C}" for R, C in sorted(CONFIGS, key=lambda rc: (rc[1], rc[0])))     # by C, then R
OUT_MD = data.RESULTS / "attn_config_table.md"
OUT_JSON = data.RESULTS / "critical_path.json"
KIND_ORDER = ("dpe_load", "dpe_compute", "dpe_drain", "stream", "logic")
KIND_NAME = {"dpe_load": "DPE load", "dpe_compute": "DPE compute", "dpe_drain": "DPE drain", "stream": "stream",
             "logic": "logic"}
ROLE_ORDER = ("gemm", "qk", "sm", "pv", "acc", "ctrl")
GEO = ("xbar", "clb", "mem", "area", "energy", "cycles", "fmax", "latency", "adp", "edap") + KIND_ORDER


def geomean(xs: list) -> float:
    if not xs or min(xs) <= 0:
        raise ValueError(f"geomean needs positive values, got {xs}")
    return math.exp(st.mean(math.log(x) for x in xs))


# -- the critical path of one build ------------------------------------------------------------------------------
def pathSums(t, since: int = 0) -> dict:
    """The chain of `t`: its cycles by kind, by role and by (role, kind), and the checks (the sum, nothing untraced,
    no negative segment)."""
    bd = breakdown(t, since)
    byKind, byRole, byBoth = {}, {}, {}
    for (role, kind), c in bd.items():
        if kind not in KINDS:
            raise SystemExit(f"unknown trace kind {kind!r}")
        byKind[kind] = byKind.get(kind, 0) + c
        byRole[role] = byRole.get(role, 0) + c
        byBoth[f"{role}/{kind}"] = c
    segs = segments(t)
    return dict(total=sum(bd.values()), by_kind=byKind, by_role=byRole, by_role_kind=byBoth, segments=len(segs),
                untraced=byKind.get("untraced", 0), negative=sum(1 for _, c in segs if c < 0))


def checkPath(b: dict) -> dict:
    if b["total"] != b["T"] or b["untraced"] or b["negative"]:
        raise SystemExit(f"{b['family']} {b['workload']} {b['config']}: the chain's segments ({b['total']}) vs T "
                         f"({b['T']}), untraced {b['untraced']}, negative segments {b['negative']}")
    return b


def attentionPath(name: str, n: int, dm: int, d: int, R: int, C: int) -> dict:
    """One head's critical path (timing only)."""
    g, parts = attentionHead(headParams(n, dm, d, R, C), withValues=False)
    run(g)
    ctrl = parts["ctrl"]
    return checkPath(dict(family="attention", workload=name, config=f"{R}x{C}", n=n, d_model=dm, d=d,
                          T=ctrl.done - ctrl.tStart, **pathSums(ctrl.done, since=ctrl.tStart)))


def fcPath(M: int, K: int, N: int, R: int, C: int) -> dict:
    """One FC row's critical path: the Report's cycles (the sink's last word + 1)."""
    rep = run(fc(K, N, R, C, M=M, platform=Platform("nl_dpe", rows=R, cols=C), passExtent=FC_PASS_EXTENT,
                 withValues=False))
    return checkPath(dict(family="fc", workload=f"{M}x{K}x{N}", config=f"{R}x{C}", M=M, K=K, N=N,
                          T=int(rep.cycles), **pathSums(rep.cycles)))


def allPaths(configs: tuple = DEFAULT) -> list:
    """The critical paths of the attention builds of `configs` and of every FC row."""
    rc = [tuple(int(x) for x in c.split("x")) for c in configs]
    out = [attentionPath(name, n, dm, d, R, C) for name, n, dm, d in SHAPES for R, C in rc]
    out += [fcPath(M, K, N, R, C) for R, C in CONFIGS for M, K, N in FC_SHAPES]
    return out


# -- the rows ------------------------------------------------------------------------------------------------------
def headValues(r: dict, path: dict) -> dict:
    """One build's values: the priced row's area parts, energy, latency, ADP, EDAP; the critical path's cycles by
    kind."""
    R, C = int(r["R"]), int(r["C"])
    a = areaByComponent(int(r["clb_count"]), int(r["wc_vtr"]), int(r["mem_blocks"]), R, C)
    if a["total_mwta"] != int(r["area_total_mwta"]):
        raise SystemExit(f"{r['shape']} {r['config']}: the area parts ({a['total_mwta']} MWTA) do not add up to the "
                         f"row's area_total_mwta ({r['area_total_mwta']})")
    mm2 = {k: v * area.MWTA_PER_CELL * data.MM2_PER_MWTA for k, v in a["cells"].items()}
    A, D, E, T = float(r["area_total_mm2"]), float(r["latency_ns"]), float(r["e_total_pj"]), int(r["T_cycles"])
    if path["T"] != T:
        raise SystemExit(f"{r['shape']} {r['config']}: the simulator's cycles ({path['T']}) differ from the priced "
                         f"table's ({T}): re-run dse/attn_rows.py")
    kinds = {k: path["by_kind"].get(k, 0) for k in KIND_ORDER}
    kinds["logic"] += sum(v for k, v in path["by_kind"].items() if k not in KIND_ORDER)   # the origin: the start
    roles = {role: path["by_role"].get(role, 0) for role in ROLE_ORDER}
    roles["ctrl"] += sum(v for role, v in path["by_role"].items() if role not in ROLE_ORDER)   # the sources' start
    return dict(xbar=mm2["xbar"], clb=mm2["clb"], mem=mm2["mem"], area=A, energy=E / 1e3, cycles=T,
                fmax=float(r["fmax_mhz"]), latency=D, adp=A * D, edap=E * D * A, **kinds,
                shares={k: v / T for k, v in {**kinds, **roles}.items()})


def summarize(rows: list, paths: list, configs: tuple) -> list:
    """Per config the geomean over the six heads of every value; the shares' means."""
    byBuild = {(p["workload"], p["config"]): p for p in paths if p["family"] == "attention"}
    out = []
    for cfg in configs:
        rs = [r for r in rows if r["config"] == cfg]
        if len(rs) != len(SHAPES):
            raise SystemExit(f"{cfg}: {len(rs)} heads in the priced table, expected {len(SHAPES)}")
        hs = [headValues(r, byBuild[(r["shape"], cfg)]) for r in rs]
        s = {k: geomean([h[k] for h in hs]) for k in GEO}
        s["shares"] = {k: st.mean(h["shares"][k] for h in hs) for k in KIND_ORDER + ROLE_ORDER}
        s["config"] = cfg
        out.append(s)
    return out


def chain(summary: list, ref: str | None = None) -> list:
    """Table 1's ratios against the reference config (default: the lowest ADP)."""
    base = min(summary, key=lambda s: s["adp"]) if ref is None else next((s for s in summary if s["config"] == ref),
                                                                           None)
    if base is None:
        raise SystemExit(f"--ref {ref}: not among the table's configs")
    out = []
    for s in summary:
        x = {k: s[k] / base[k] for k in ("area", "cycles", "fmax", "energy")}
        x["latency"] = x["cycles"] / x["fmax"]
        x["adp"] = x["area"] * x["latency"]
        x["edap"] = x["energy"] * x["adp"]
        out.append(dict(x, config=s["config"], ref=base["config"]))
    return out


# -- the tables ----------------------------------------------------------------------------------------------------
def cfgName(c: str) -> str:
    return c.replace("x", "×")


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def table1(rows: list) -> str:
    lines = ["| config | area | cycles | Fmax | latency = cycles ÷ Fmax | ADP = area × latency | energy "
             "| EDAP = energy × ADP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        adp, edap = f"{x['adp']:.2f}", f"{x['edap']:.2f}"
        if x["config"] == x["ref"]:                                   # the reference in bold
            adp, edap = f"**{adp}**", f"**{edap}**"
        lines.append(f"| {cfgName(x['config'])} | {x['area']:.2f} | {x['cycles']:.2f} | {x['fmax']:.2f} | "
                     f"{x['latency']:.2f} | {adp} | {x['energy']:.2f} | {edap} |")
    return "\n".join(lines)


def table2(summary: list) -> str:
    lines = ["| config | crossbar mm² | CLB mm² | BRAM mm² | area mm² | energy nJ | "
             + " | ".join(f"{KIND_NAME[k]} cyc" for k in KIND_ORDER) + " | cycles T | Fmax MHz |",
             "|---|---:|---:|---:|---:|---:|" + "---:|" * len(KIND_ORDER) + "---:|---:|"]
    for s in summary:
        lines.append(f"| {cfgName(s['config'])} | {s['xbar']:.1f} | {s['clb']:.1f} | {s['mem']:.1f} | {s['area']:.1f} | "
                     f"{s['energy']:,.0f} | " + " | ".join(f"{s[k]:,.0f}" for k in KIND_ORDER)
                     + f" | {s['cycles']:,.0f} | {s['fmax']:.0f} |")
    return "\n".join(lines)


def sharesHeader(first: str) -> list:
    return [f"| {first} | T | " + " | ".join(KIND_NAME[k] for k in KIND_ORDER) + " | " + " | ".join(ROLE_ORDER) + " |",
            "|---|---:|" + "---:|" * (len(KIND_ORDER) + len(ROLE_ORDER))]


def table3(summary: list) -> str:
    lines = sharesHeader("config")
    for s in summary:
        lines.append(f"| {cfgName(s['config'])} | {s['cycles']:,.0f} | "
                     + " | ".join(pct(s["shares"][k]) for k in KIND_ORDER + ROLE_ORDER) + " |")
    return "\n".join(lines)


def table4(rows: list, paths: list, cfg: str) -> str:
    byBuild = {(p["workload"], p["config"]): p for p in paths if p["family"] == "attention"}
    lines = sharesHeader("head")
    for r in rows:
        if r["config"] != cfg:
            continue
        h = headValues(r, byBuild[(r["shape"], cfg)])
        lines.append(f"| {r['shape']} | {h['cycles']:,} | "
                     + " | ".join(pct(h["shares"][k]) for k in KIND_ORDER + ROLE_ORDER) + " |")
    return "\n".join(lines)


def table5(paths: list) -> str:
    lines = ["| config | T (geomean) | " + " | ".join(KIND_NAME[k] for k in KIND_ORDER) + " |",
             "|---|---:|" + "---:|" * len(KIND_ORDER)]
    for R, C in sorted(CONFIGS, key=lambda rc: (rc[1], rc[0])):
        cfg = f"{R}x{C}"
        ps = [p for p in paths if p["family"] == "fc" and p["config"] == cfg]
        lines.append(f"| {cfgName(cfg)} | {geomean([p['T'] for p in ps]):,.0f} | "
                     + " | ".join(pct(st.mean(p["by_kind"].get(k, 0) / p["T"] for p in ps)) for k in KIND_ORDER)
                     + " |")
    return "\n".join(lines)


def document(summary: list, rows: list, paths: list, ref: str | None, configs: tuple) -> str:
    ratios = chain(summary, ref)
    r = cfgName(ratios[0]["ref"])
    detail = [c for c in ("128x64", "128x128") if c in configs] or list(configs[:1])
    perHead = "\n\n".join(f"**Per head at {cfgName(c)}** (shares of T: by kind, then by module)\n\n"
                          + table4(rows, paths, c) for c in detail)
    return f"""# Attention: crossbar configs side by side

Per head, the geometric mean over the six heads (`dse/config_table.py`; definitions in `docs/experiment.md` §9.3).

## 1. The chain, relative to {r}

Every column is relative to {r} (the best ADP of the table unless `--ref` names another), and the columns multiply
exactly: latency = cycles ÷ Fmax, ADP = area × latency, EDAP = energy × ADP. Lower is better, except Fmax.

{table1(ratios)}

## 2. What is inside

Area: VPR counts (crossbar tiles, CLBs, BRAM blocks at 2 cells). Energy: the priced ledger. Cycles: the head's critical
path by kind (the chain of waits that set the head's done; its segments add up to T exactly): DPE load = a pass's
input burst, DPE compute = the bit-slices, the shift-accumulate, the ACAM and the hand-overs, DPE drain = the first
output word and the drain words a consumer takes, stream = a port's words or a FIFO read one word per cycle (the 40-bit
paths between modules), logic = every other named latency. Fmax: VTR, the mean of seeds {{11, 13, 17}}. The parts are
geometric means too, so they come close to their total but do not add up to it exactly.

{table2(summary)}

## 3. The critical path as shares of T

By kind, then by module (the module whose wait the cycle is: the projections gemm, the QK and PV DIMMs, the online
softmax sm, the accumulator acc, the controller); the mean over the six heads of each share. A cycle that two inputs
set together is charged to the first the model names (a tie).

{table3(summary)}

{perHead}

## 4. FC: the critical path by kind

One pass per row (the load of min(R, K) rows, the slices, the first drain word, the tree and the serializer, the
output words on the port); the mean over the four shapes of each share.

{table5(paths)}
"""


# -- the self-test -------------------------------------------------------------------------------------------------
def selfTest() -> int:
    n = 0
    assert abs(geomean([1, 4]) - 2) < 1e-12 and abs(geomean([3]) - 3) < 1e-12; n += 1
    for bad in ([], [0, 1], [-1]):
        try:
            geomean(bad)
        except ValueError:
            n += 1
        else:
            raise AssertionError(f"geomean accepted {bad}")
    rows = data.attentionRows()
    paths = allPaths()                                 # checkPath ran on every build: the sum, untraced, negative
    assert len(paths) == 72 + 48; n += 1
    # the attention T is the priced table's T_cycles, the FC T the FC table's: the trace changes no value
    priced = {(r["shape"], r["config"]): int(r["T_cycles"]) for r in rows}
    pricedFc = {(r["M"], r["K"], r["N"], r["config"]): int(r["T_cycles"]) for r in data.fcRows()}
    for p in paths:
        if p["family"] == "attention":
            assert priced[(p["workload"], p["config"])] == p["T"], (p["workload"], p["config"])
        else:
            assert pricedFc[(str(p["M"]), str(p["K"]), str(p["N"]), p["config"])] == p["T"], p["workload"]
        n += 1
    # FC: the closed form per kind (one pass, the tree, the serializer, the port)
    for p in paths:
        if p["family"] != "fc":
            continue
        R, C = (int(x) for x in p["config"].split("x"))
        V, H = ceilDiv(p["K"], R), ceilDiv(p["N"], C)
        rws, cols = (p["K"] if V == 1 else R), (p["N"] if H == 1 else C)
        want = {"origin": 0, "dpe_load": ceilDiv(8 * rws, 40), "dpe_compute": 9, "dpe_drain": 1,
                "logic": ((V - 1).bit_length() if V > 1 else 0) + 1, "stream": ceilDiv(cols, 5)}
        assert p["by_kind"] == want, (p["workload"], p["config"], p["by_kind"], want); n += 1
    # attention: the kinds and roles known, every kind on every head's path (so the geomeans hold), the shares sum
    # to 1, the first token's projection load opens every single-run head's path
    for p in paths:
        if p["family"] != "attention":
            continue
        assert set(p["by_kind"]) <= set(KIND_ORDER) | {"origin"} and set(p["by_role"]) <= set(ROLE_ORDER) | {"x0", "st"}
        assert all(p["by_kind"].get(k, 0) > 0 for k in KIND_ORDER), (p["workload"], p["config"], p["by_kind"])
        n += 1
    s = summarize(rows, paths, DEFAULT)
    assert len(DEFAULT) == 12 and DEFAULT[:4] == ("128x64", "256x64", "512x64", "1024x64"); n += 1
    assert [x["config"] for x in s] == list(DEFAULT); n += 1
    for x in s:
        assert abs(sum(x["shares"][k] for k in KIND_ORDER) - 1) < 1e-12, x["config"]
        assert abs(sum(x["shares"][r] for r in ROLE_ORDER) - 1) < 1e-12, x["config"]
        n += 1
    # table 1: the reference is the lowest ADP (all 1.00), the chain holds, ADP / EDAP are rank.py's G ratios (the
    # normalization by the worst config cancels); latency_ns is stored to 0.001 ns, so 1e-6 relative
    t = chain(s)
    ref = next(x for x in t if x["config"] == t[0]["ref"])
    assert all(abs(ref[k] - 1) < 1e-12 for k in ("area", "cycles", "fmax", "latency", "adp", "energy", "edap")); n += 1
    assert min(x["adp"] for x in t) == ref["adp"]; n += 1
    pts = [dict(workload=r["shape"], config=r["config"], E=float(r["e_total_pj"]), D=float(r["latency_ns"]),
                A=float(r["area_total_mm2"])) for r in rows]
    gAdp = rank.adp_scores(pts)["G_ADP"]
    gEdap = {c["config"]: c["G_EDAP"] for c in rank.family_ranking(pts)["configs"]}
    sums = {x["config"]: x for x in s}
    for x in t:
        c = x["config"]
        for got, want in ((x["latency"], sums[c]["latency"] / sums[ref["config"]]["latency"]),
                          (x["adp"], gAdp[c] / gAdp[ref["config"]]), (x["edap"], gEdap[c] / gEdap[ref["config"]])):
            assert abs(got / want - 1) < 1e-6, (c, got, want)
            n += 1
    t2 = chain(s, "1024x256")
    assert all(abs(x["adp"] - y["adp"] / t[-1]["adp"]) < 1e-9 for x, y in zip(t2, t)) and t2[-1]["adp"] == 1; n += 1
    for bad in (lambda: chain(s, "99x99"), lambda: summarize(rows, paths, ("128x64", "99x99"))):
        try:
            bad()
        except SystemExit:
            n += 1
        else:
            raise AssertionError("an unknown config was accepted")
    # the document renders every table for a subset of configs
    text = document(summarize(rows, paths, ("128x64", "256x128")), rows, paths, None, ("128x64", "256x128"))
    assert text.count("\n| 128×64 |") == 4 and "Per head at 128×64" in text and "## 4. FC" in text; n += 1
    print(f"config_table self-test: ALL PASS ({n} checks on {len(paths)} builds)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="*", default=list(DEFAULT))
    ap.add_argument("--ref", default=None, help="the reference config of table 1 (default: the lowest ADP)")
    ap.add_argument("--out", default=str(OUT_MD))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return selfTest()
    configs = tuple(a.configs)
    rows = data.attentionRows()
    paths = allPaths(configs)
    text = document(summarize(rows, paths, configs), rows, paths, a.ref, configs)
    print(text)
    Path(a.out).write_text(text)
    OUT_JSON.write_text(json.dumps(dict(kinds=KIND_ORDER, roles=ROLE_ORDER, builds=paths), indent=1) + "\n")
    print(f"-> {a.out}\n-> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
