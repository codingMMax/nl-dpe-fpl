#!/usr/bin/env python3
"""mapping_sensitivity.py — how the evaluated metrics respond to each mapping parameter, one at a time
(docs/mapping_sensitivity.md).

The attention mapping is fixed (experiment.md §4.2): P_PROJ = 1, N_E = N_EP = 2d, N_EXP = 2, N_OC by the floor rule,
BKV = 16, X_S = 16, S = min(n, I). For every attention build (6 heads × 8 configs) and every value of every knob below,
the head is built with that one parameter changed and run (timing only, the trace off) and the script records the
cycles T, the pass count, the energy (the frozen prices), the crossbar count and the crossbar area, or the reason the
build is outside the domain (a module guard) or breaks a port contract (the RTL's fixed buffers). Per knob value it
summarizes the ratio to the fixed mapping over the builds: min / median / max, and how many builds are outside.
Latency in ns and the CLB area need VTR and are not re-derived: a variant's cycles and crossbar count are exact,
its Fmax and soft logic unknown.

  P_PROJ  1, 2, 4            projection copies (the X stream's parallelism)
  N_E     d, 2d, 4d          QK farm crossbars
  N_EP    d, 2d, 4d          PV farm crossbars (d is below the accumulator's N_OC cap on most builds)
  N_EXP   1, 2, 4            EXP crossbars of the online softmax
  N_OC    half, floor, cap   o-carry crossbars: half the floor rule's pick, the pick, the cap P_CH = N_EP / BKV
  BKV     8, 16, 32          keys per block
  X_S     8, 16, 32          crossbars per DIMM slice

--optimize builds every mapping of a grid (P_proj {1..16} × N_E {d,2d,4d} × N_EP {2d,4d} × N_EXP {1,2,4} × B_kv
{8,16,32}; N_OC by the floor rule, X_S 16) for every build, keeps per build and per objective (cycles, area, latency,
ADP, EDAP) the mapping that minimizes it, and ranks the configs at those optima against the fixed mapping
(docs/mapping_sensitivity.md §6). The soft logic and Fmax of a variant are the fixed build's (VTR); the crossbar area
is the variant's. Outputs dse/results/mapping_optimum.{json,md}. About 25 minutes for the 72 builds of the 12 configs (2026-10-19).

--balanced instead builds every head at P_proj* (docs/mapping_sensitivity.md §6: the smallest power of two with the
projection stream as fast as the core consumes keys, 5 T_p / S, capped by n and I / B_kv) with the other knobs at the
rate rule's values, and compares cycles, the stream floor n²/5 and the crossbar area with the fixed mapping.

Outputs: dse/results/mapping_sensitivity.json (every variant) and dse/results/mapping_sensitivity.md (the summary);
--balanced: dse/results/mapping_balanced.{json,md}.
Usage: python3 v3/dse/mapping_sensitivity.py [--configs 128x64,256x128] [--shapes BERT-Tiny,ViT-B/16] [--balanced | --optimize]
       python3 v3/dse/mapping_sensitivity.py --self-test
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
import simulator.core.trace as trace  # noqa: E402
from dse import data  # noqa: E402
from dse import rank  # noqa: E402
from dse.space import CONFIGS, SHAPES  # noqa: E402
from simulator.core.cycleMath import ceilDiv  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams, nOcFloor  # noqa: E402
from simulator.platform import area, prices  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402

OUT_JSON = data.RESULTS / "mapping_sensitivity.json"
OUT_MD = data.RESULTS / "mapping_sensitivity.md"
OUT_BAL_JSON = data.RESULTS / "mapping_balanced.json"
OUT_BAL_MD = data.RESULTS / "mapping_balanced.md"
OUT_OPT_JSON = data.RESULTS / "mapping_optimum.json"
OUT_OPT_MD = data.RESULTS / "mapping_optimum.md"
KNOBS = ("P_PROJ", "N_E", "N_EP", "N_EXP", "N_OC", "BKV", "X_S")


def variants(name: str, n: int, dm: int, d: int, R: int, C: int) -> list:
    """(knob, label, kwargs) of every one-at-a-time variant of a build, the fixed mapping first."""
    I = min(R, C)
    S = min(n, I)
    floor = nOcFloor(S, d, R, C, 16)
    cap = (2 * d) // 16                                   # the accumulator's N_OC <= P_CH = N_EP / BKV at N_EP = 2d
    out = [("fixed", "fixed", {})]
    out += [("P_PROJ", str(v), dict(P_PROJ=v)) for v in (1, 2, 4)]
    out += [("N_E", lab, dict(N_E=v)) for lab, v in (("d", d), ("2d", 2 * d), ("4d", 4 * d))]
    out += [("N_EP", lab, dict(N_EP=v)) for lab, v in (("d", d), ("2d", 2 * d), ("4d", 4 * d))]
    out += [("N_EXP", str(v), dict(N_EXP=v)) for v in (1, 2, 4)]
    out += [("N_OC", lab, dict(N_OC=v)) for lab, v in (("half", max(1, floor // 2)), ("floor", floor), ("cap", cap))]
    out += [("BKV", str(v), dict(BKV=v)) for v in (8, 16, 32)]
    out += [("X_S", str(v), dict(X_S=v)) for v in (8, 16, 32)]
    return out


def build(n: int, dm: int, d: int, R: int, C: int, **kw) -> dict:
    """One head, timing only: cycles, passes, energy, crossbars, or the reason it cannot be built."""
    try:
        hp = headParams(n, dm, d, R, C, **kw)
        g, parts = attentionHead(hp, withValues=False)
        rep = run(g)
    except Exception as e:                                # a module guard (ValueError) or a port contract (ContractError)
        kind = "domain" if isinstance(e, ValueError) else "contract"
        return dict(status=kind, reason=f"{type(e).__name__}: {str(e)[:120]}")
    ctrl = parts["ctrl"]
    e = prices.energy(rep.totalCounts, Platform("nl_dpe", rows=R, cols=C))
    return dict(status="ok", T=ctrl.done - ctrl.tStart, passes=sum(rep.totalCounts.passes.values()),
                e_total_pj=e["e_total_pj"], e_pass_pj=e["e_pass_pj"], crossbars=hp.nXbar,
                area_wc_mwta=area.areaWcMwta(hp.nXbar, R, C), S=hp.S, N_OC=hp.N_OC, N_E=hp.N_E, N_EP=hp.N_EP,
                N_EXP=hp.N_EXP, P_PROJ=hp.P_PROJ, BKV=hp.BKV, X_S=hp.X_S)


def sweep(shapes: list, configs: list) -> list:
    """Every build × every variant, with the ratios to the build's fixed mapping."""
    trace.enable(False)
    rows = []
    for name, n, dm, d in shapes:
        for R, C in configs:
            base = None
            for knob, label, kw in variants(name, n, dm, d, R, C):
                r = dict(shape=name, config=f"{R}x{C}", knob=knob, value=label, **build(n, dm, d, R, C, **kw))
                if knob == "fixed":
                    base = r
                    if r["status"] != "ok":
                        raise SystemExit(f"{name} {R}x{C}: the fixed mapping does not build: {r.get('reason')}")
                if r["status"] == "ok":
                    r["T_ratio"] = r["T"] / base["T"]
                    r["E_ratio"] = r["e_total_pj"] / base["e_total_pj"]
                    r["xbar_delta"] = r["crossbars"] - base["crossbars"]
                    r["area_wc_ratio"] = r["area_wc_mwta"] / base["area_wc_mwta"]
                rows.append(r)
    return rows


def summarize(rows: list) -> list:
    """Per (knob, value): the ratios' min / median / max over the builds that build, the count outside."""
    out = []
    seen = []
    for r in rows:
        key = (r["knob"], r["value"])
        if r["knob"] != "fixed" and key not in seen:
            seen.append(key)
    for knob, value in seen:
        rs = [r for r in rows if (r["knob"], r["value"]) == (knob, value)]
        ok = [r for r in rs if r["status"] == "ok"]
        s = dict(knob=knob, value=value, builds=len(rs), ok=len(ok),
                 outside={k: sum(1 for r in rs if r["status"] == k) for k in ("domain", "contract")},
                 outside_configs=sorted({r["config"] for r in rs if r["status"] != "ok"}))
        for k in ("T_ratio", "E_ratio", "area_wc_ratio"):
            v = [r[k] for r in ok]
            s[k] = dict(min=min(v), median=st.median(v), max=max(v)) if v else None
        s["xbar_delta"] = dict(min=min(r["xbar_delta"] for r in ok), max=max(r["xbar_delta"] for r in ok)) if ok else None
        out.append(s)
    return out


def table(summary: list, nBuilds: int) -> str:
    lines = ["| knob | value | cycles ÷ fixed (min / median / max) | energy ÷ fixed (median; min–max) | crossbars Δ "
             "| crossbar area ÷ fixed (median) | builds outside the domain or a contract |",
             "|---|---|---:|---:|---:|---:|---|"]
    for s in summary:
        if s["ok"] == 0:
            lines.append(f"| {s['knob']} | {s['value']} | — | — | — | — | all {s['builds']}: "
                         f"{s['outside']['domain']} domain, {s['outside']['contract']} contract |")
            continue
        t, e, a = s["T_ratio"], s["E_ratio"], s["area_wc_ratio"]
        out = s["builds"] - s["ok"]
        where = (f"{out} of {s['builds']} ({s['outside']['domain']} domain, {s['outside']['contract']} contract: "
                 + ", ".join(c.replace("x", "×") for c in s["outside_configs"][:6])
                 + (" …" if len(s["outside_configs"]) > 6 else "") + ")") if out else "none"
        lines.append(f"| {s['knob']} | {s['value']} | {t['min']:.2f} / {t['median']:.2f} / {t['max']:.2f} | "
                     f"{e['median']:.3f}; {e['min']:.3f}–{e['max']:.3f} | {s['xbar_delta']['min']:+d} … "
                     f"{s['xbar_delta']['max']:+d} | {a['median']:.2f} | {where} |")
    return "\n".join(lines)


def perConfig(rows: list, configs: list, key: str = "T_ratio") -> str:
    """Per (knob, value) the median ratio over the heads, one column per config: does the knob move the configs
    together, or does it change their order?"""
    cfgs = [f"{R}x{C}" for R, C in configs]
    seen = []
    for r in rows:
        if r["knob"] != "fixed" and (r["knob"], r["value"]) not in seen:
            seen.append((r["knob"], r["value"]))
    lines = ["| knob | value | " + " | ".join(c.replace("x", "×") for c in cfgs) + " |", "|---|---|" + "---:|" * len(cfgs)]
    for knob, value in seen:
        cells = []
        for c in cfgs:
            v = [r[key] for r in rows if (r["knob"], r["value"], r["config"]) == (knob, value, c) and r["status"] == "ok"]
            nAll = sum(1 for r in rows if (r["knob"], r["value"], r["config"]) == (knob, value, c))
            cells.append("—" if not v else f"{st.median(v):.2f}" + (f" ({len(v)}/{nAll})" if len(v) < nAll else ""))
        lines.append(f"| {knob} | {value} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def document(summary: list, rows: list, shapes: list, configs: list) -> str:
    nB = len(shapes) * len(configs)
    return f"""# Mapping sensitivity, one knob at a time

{len(shapes)} heads × {len(configs)} configs = {nB} builds; every knob changed alone from the fixed mapping
(P_PROJ 1, N_E = N_EP = 2d, N_EXP 2, N_OC by the floor rule, BKV 16, X_S 16); the simulator's cycles, pass counts,
energy (frozen prices) and crossbar count, timing only (`dse/mapping_sensitivity.py`; the analysis in
`docs/mapping_sensitivity.md`). Ratios are per build against its own fixed mapping, summarized over the builds that
build. Fmax and the CLB area of a variant need VTR and are not in this table.

{table(summary, nB)}

A row's "outside" counts the builds a module guard rejects (domain) or whose run breaks a port contract (contract: a
fixed buffer of the RTL overruns), with the configs concerned.

## Cycles ÷ fixed per config (the median over the heads that build; "(k/m)" = k of the m heads build)

A knob that moves every config by the same ratio leaves the latency ranking as it is; one that moves them
differently changes it.

{perConfig(rows, configs)}
"""


# -- the balanced mapping (docs/mapping_sensitivity.md §6) ----------------------------------------------------------
def pow2up(x: float) -> int:
    return 1 << max(0, (int(math.ceil(x)) - 1).bit_length())


def balancedPproj(n: int, dm: int, R: int, C: int, BKV: int = 16, BUF: int = 40, P: int = 8) -> int:
    """The smallest power of two with the projection stream at least as fast as the core consumes keys,
    P_proj >= EPS T_p / S (one block of B_kv tokens per stream floor ceil(S B_kv / EPS)), capped by n and by the QK A
    staging group I / B_kv (A_CH <= NB)."""
    I = min(R, C)
    S = min(n, I)
    Tp = max(ceilDiv(8 * min(dm, R), BUF) + P, ceilDiv(8 * C, BUF) + 1)
    return min(n, I // BKV, pow2up((BUF // 8) * Tp / S))


def balanced(shapes: list, configs: list) -> list:
    """Every build at the fixed mapping and at P_proj*: cycles, crossbars, the stream floor."""
    trace.enable(False)
    out = []
    for name, n, dm, d in shapes:
        for R, C in configs:
            star = balancedPproj(n, dm, R, C)
            base = build(n, dm, d, R, C)
            bal = build(n, dm, d, R, C, P_PROJ=star)
            if base["status"] != "ok" or bal["status"] != "ok":
                raise SystemExit(f"{name} {R}x{C}: {base.get('reason')} / {bal.get('reason')}")
            out.append(dict(shape=name, config=f"{R}x{C}", n=n, P_star=star, T_fixed=base["T"], T_bal=bal["T"],
                            T_ratio=bal["T"] / base["T"], floor=n * n // 5, floor_ratio=bal["T"] / (n * n / 5),
                            xbar_fixed=base["crossbars"], xbar_bal=bal["crossbars"],
                            area_wc_ratio=bal["area_wc_mwta"] / base["area_wc_mwta"]))
    return out


def balancedTable(rows: list, configs: list) -> str:
    cfgs = [f"{R}x{C}" for R, C in configs]
    gm = lambda v: math.exp(st.mean(math.log(x) for x in v))     # noqa: E731
    ref = {k: gm([r[k] for r in rows if r["config"] == cfgs[0]]) for k in ("T_fixed", "T_bal")}
    lines = ["| config | P_proj* per head | cycles fixed (geomean) | cycles balanced | ÷ fixed | ÷ n²/5 | crossbar area ÷ fixed "
             "| cycles ÷ " + cfgs[0].replace("x", "×") + ": fixed → balanced |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in cfgs:
        rs = [r for r in rows if r["config"] == c]
        tf, tb = gm([r["T_fixed"] for r in rs]), gm([r["T_bal"] for r in rs])
        lines.append(f"| {c.replace('x', '×')} | {' / '.join(str(r['P_star']) for r in rs)} | {tf:,.0f} | {tb:,.0f} | "
                     f"{tb / tf:.2f} | {gm([r['floor_ratio'] for r in rs]):.2f} | {gm([r['area_wc_ratio'] for r in rs]):.2f} | "
                     f"{tf / ref['T_fixed']:.2f} → {tb / ref['T_bal']:.2f} |")
    return "\n".join(lines)


def balancedDocument(rows: list, shapes: list, configs: list) -> str:
    return f"""# The balanced mapping: P_proj* per build

Every other knob keeps the rate rule's value (N_E = N_EP = 2d, N_EXP = 2, N_OC by the floor rule, B_kv 16, X_S 16),
which is the analytic optimum of the core (`docs/mapping_sensitivity.md` §6); P_proj* is the smallest power of two
with the projection stream at least as fast as the core consumes keys, P_proj >= 5 T_p / S, capped by n and by the QK
staging group I / B_kv. Per head in the order {", ".join(s[0] for s in shapes)}. Cycles and crossbars are the
simulator's; Fmax and the soft logic of the P_proj* builds need VTR (`dse/mapping_sensitivity.py --balanced`).

{balancedTable(rows, configs)}
"""


# -- the optimized mapping (docs/mapping_sensitivity.md §6) ---------------------------------------------------------
GRID = dict(P_PROJ=(1, 2, 4, 8, 16), N_E=("d", "2d", "4d"), N_EP=("2d", "4d"), N_EXP=(1, 2, 4), BKV=(8, 16, 32))
OBJECTIVES = ("cycles", "area", "latency", "adp", "edap")


def gridPoints(d: int) -> list:
    """Every mapping of the grid as (label, kwargs); N_OC by the floor rule and X_S = 16 throughout (the sweep: the
    floor rule is the optimum, X_S moves the cycles by at most 2 %)."""
    import itertools
    out = []
    for P_PROJ, ne, nep, N_EXP, BKV in itertools.product(*GRID.values()):
        mult = {"d": 1, "2d": 2, "4d": 4}
        out.append((f"P{P_PROJ} N_E={ne} N_EP={nep} N_EXP={N_EXP} BKV={BKV}",
                    dict(P_PROJ=P_PROJ, N_E=mult[ne] * d, N_EP=mult[nep] * d, N_EXP=N_EXP, BKV=BKV)))
    return out


def metrics(b: dict, vtr: dict, R: int, C: int) -> dict:
    """A variant's metrics with the fixed build's VTR soft logic and Fmax: area = the fixed CLBs and memory blocks +
    the variant's crossbar tiles; latency = the variant's cycles / the fixed Fmax; energy the variant's own."""
    w, h = area.footprint(R, C)
    A = (vtr["clb"] + b["crossbars"] * w * h + area.MEM_TILE_CELLS * vtr["mem"]) * area.MWTA_PER_CELL * data.MM2_PER_MWTA
    D = b["T"] * 1000.0 / vtr["fmax"]
    E = b["e_total_pj"]
    return dict(cycles=b["T"], area=A, latency=D, energy=E, adp=A * D, edap=E * A * D, crossbars=b["crossbars"])


def optimize(shapes: list, configs: list, grid=None) -> dict:
    """Per build the grid's metrics and, per objective, the mapping that minimizes it (ties: fewer crossbars, then
    the grid order); the fixed mapping's metrics beside them."""
    trace.enable(False)
    rows = data.attentionRows()
    vtrOf = {(r["shape"], r["config"]): dict(clb=int(r["clb_count"]), mem=int(r["mem_blocks"]), fmax=float(r["fmax_mhz"]))
             for r in rows}
    builds = []
    for name, n, dm, d in shapes:
        for R, C in configs:
            cfg = f"{R}x{C}"
            vtr = vtrOf[(name, cfg)]
            fixed = build(n, dm, d, R, C)
            fixedM = metrics(fixed, vtr, R, C)
            points = []
            for label, kw in (grid or gridPoints)(d):
                b = build(n, dm, d, R, C, **kw)
                if b["status"] == "ok":
                    points.append(dict(mapping=label, **kw, **metrics(b, vtr, R, C)))
            best = {}
            for obj in OBJECTIVES:
                m = min(points, key=lambda q: (q[obj], q["crossbars"]))
                best[obj] = dict(m, ratio=m[obj] / fixedM[obj])
            builds.append(dict(shape=name, config=cfg, n=n, d=d, grid=len(points), fixed=fixedM, best=best))
    return dict(grid=GRID, builds=builds)


def rankings(res: dict) -> dict:
    """Per objective the attention ranking (rank.py's G over the shapes) at the per-build optimum, beside the fixed
    mapping's: G_ADP and G_EDAP (family_ranking), G_D and G_A."""
    builds = res["builds"]
    def pts(get):
        return [dict(workload=b["shape"], config=b["config"], E=get(b)["energy"], D=get(b)["latency"], A=get(b)["area"])
                for b in builds]
    out = {"fixed": rankTable(pts(lambda b: b["fixed"]))}
    for obj in OBJECTIVES:
        out[obj] = rankTable(pts(lambda b, o=obj: b["best"][o]))
    return out


def rankTable(points: list) -> dict:
    fr = rank.family_ranking(points)
    la = rank.la_scores(points)
    adp = rank.adp_scores(points)["G_ADP"]
    byG = {c: g for c, g in adp.items()}
    adpRank = {c: i + 1 for i, c in enumerate(sorted(byG, key=lambda c: (byG[c], c)))}
    return {c["config"]: dict(G_EDAP=c["G_EDAP"], rank_edap=c["rank"], G_ADP=byG[c["config"]], rank_adp=adpRank[c["config"]],
                              G_D=c["G_D"], G_A=c["G_A"], G_E=c["G_E"],
                              front=next(x["front"] for x in la["configs"] if x["config"] == c["config"]))
            for c in fr["configs"]}


def mappingSummary(builds: list, obj: str, cfg: str) -> str:
    """The mappings a config's heads pick for an objective, as 'knob values' with the per-head spread."""
    ms = [b["best"][obj] for b in builds if b["config"] == cfg]
    parts = []
    for knob in ("P_PROJ", "N_E", "N_EP", "N_EXP", "BKV"):
        vals = [m[knob] for m in ms]
        d = {b["d"] for b in builds if b["config"] == cfg}
        if knob in ("N_E", "N_EP"):
            vals = [f"{v // b['d']}d" for v, b in zip(vals, [b for b in builds if b["config"] == cfg])]
        uniq = sorted(set(vals), key=str)
        parts.append(f"{knob} {uniq[0] if len(uniq) == 1 else '/'.join(map(str, uniq))}")
    return ", ".join(parts)


def optimumDocument(res: dict, ranks: dict, configs: list) -> str:
    builds = res["builds"]
    cfgs = [f"{R}x{C}" for R, C in configs]
    def cx(c): return c.replace("x", "×")
    fx = ranks["fixed"]
    tabA = ["| config | fixed: G_ADP (rank) | ADP-optimal mapping: G_ADP (rank) | Δ rank | ADP ÷ fixed (geomean over the heads) | the mapping the heads pick |",
            "|---|---:|---:|---:|---:|---|"]
    for c in sorted(cfgs, key=lambda c: fx[c]["rank_adp"]):
        o = ranks["adp"][c]
        ratios = [b["best"]["adp"]["ratio"] for b in builds if b["config"] == c]
        gm = math.exp(st.mean(math.log(x) for x in ratios))
        tabA.append(f"| {cx(c)} | {fx[c]['G_ADP']:.3f} ({fx[c]['rank_adp']}) | {o['G_ADP']:.3f} ({o['rank_adp']}) | "
                    f"{o['rank_adp'] - fx[c]['rank_adp']:+d} | {gm:.2f} | {mappingSummary(builds, 'adp', c)} |")
    tabB = ["| config | fixed: G_D (front) | latency-optimal: G_D (front) | cycles ÷ fixed | fixed: G_A | area-optimal: G_A | area ÷ fixed | fixed: G_EDAP (rank) | EDAP-optimal: G_EDAP (rank) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(cfgs, key=lambda c: fx[c]["rank_adp"]):
        gmc = math.exp(st.mean(math.log(b["best"]["cycles"]["ratio"]) for b in builds if b["config"] == c))
        gma = math.exp(st.mean(math.log(b["best"]["area"]["ratio"]) for b in builds if b["config"] == c))
        f = lambda r: " (front)" if r["front"] else ""      # noqa: E731
        tabB.append(f"| {cx(c)} | {fx[c]['G_D']:.3f}{f(fx[c])} | {ranks['cycles'][c]['G_D']:.3f}{f(ranks['cycles'][c])} | {gmc:.2f} | "
                    f"{fx[c]['G_A']:.3f} | {ranks['area'][c]['G_A']:.3f} | {gma:.2f} | {fx[c]['G_EDAP']:.3f} ({fx[c]['rank_edap']}) | "
                    f"{ranks['edap'][c]['G_EDAP']:.3f} ({ranks['edap'][c]['rank_edap']}) |")
    tabC = ["| shape | config | ADP-optimal mapping | cycles ÷ fixed | area ÷ fixed | ADP ÷ fixed | energy ÷ fixed |", "|---|---|---|---:|---:|---:|---:|"]
    for b in builds:
        m = b["best"]["adp"]
        tabC.append(f"| {b['shape']} | {cx(b['config'])} | {m['mapping']} | {m['cycles'] / b['fixed']['cycles']:.2f} | "
                    f"{m['area'] / b['fixed']['area']:.2f} | {m['ratio']:.2f} | {m['energy'] / b['fixed']['energy']:.3f} |")
    g = res["grid"]
    return f"""# The mapping optimized per build, and what it does to the attention ranking

For every build (shape × config) every mapping of the grid P_proj {g['P_PROJ']} × N_E {g['N_E']} × N_EP {g['N_EP']}
× N_EXP {g['N_EXP']} × B_kv {g['BKV']} (N_OC by the floor rule, X_S 16) is built in the simulator; the ones a module
guard or a port contract rejects are dropped. Per build and per objective the mapping that minimizes it is kept.
Cycles and energy are the variant's own; area = the fixed build's CLBs and memory blocks (VTR) + the variant's
crossbar tiles; latency = cycles / the fixed build's Fmax. So the soft logic and Fmax of a variant are assumed to be
the fixed mapping's: the crossbar part of the area is exact, the rest is not re-synthesized
(`dse/mapping_sensitivity.py --optimize`; the reading in `docs/mapping_sensitivity.md` §6).

## A. The ADP ranking of the attention family: fixed mapping against the ADP-optimal mapping per build

G_ADP = the geometric mean over the heads of area × latency ÷ the worst config's (rank.py); lower is better.

{chr(10).join(tabA)}

## B. Latency, area and EDAP

{chr(10).join(tabB)}

## C. Per build: the ADP-optimal mapping

{chr(10).join(tabC)}
"""


# -- closed forms (docs/mapping_sensitivity.md §2) ------------------------------------------------------------------
def passFormulas(n: int, d: int, I: int, V: int, RB: int, B: int, S: int) -> dict:
    """Passes per module of one head at the fixed mapping: the work, not the resources."""
    pOc = S * d // I
    return {"gemm": 3 * V * n, "qk": n * n * d // I + 2 * n, "pv": n * n * d // I + n,
            "sm": n * n // I + RB * (2 * B - 1), "acc": 2 * RB * B * pOc}


def crossbarFormula(V: int, d: int, P_PROJ: int, N_E: int, N_EP: int, N_EXP: int, N_OC: int) -> int:
    return 3 * P_PROJ * V + 3 * P_PROJ + N_E + N_EP + (N_EXP + 2) + 2 * N_OC


def selfTest() -> int:
    n = 0
    trace.enable(False)
    for name, nseq, dm, d in SHAPES:
        for R, C in CONFIGS:
            hp = headParams(nseq, dm, d, R, C)
            g, parts = attentionHead(hp, withValues=False)
            rep = run(g)
            agg = {}
            for i, c in rep.counts.items():
                agg[rep.roles[i]] = agg.get(rep.roles[i], 0) + sum(c.passes.values())
            want = passFormulas(nseq, d, hp.I, hp.V, hp.RB, hp.B, hp.S)
            assert {k: agg.get(k, 0) for k in want} == want, (name, R, C, agg, want); n += 1
            assert crossbarFormula(hp.V, d, hp.P_PROJ, hp.N_E, hp.N_EP, hp.N_EXP, hp.N_OC) == hp.nXbar; n += 1
    # the sweep on one build: the fixed variant's ratios are 1, an outside variant is reported, not raised
    rows = sweep([("BERT-Tiny", 128, 128, 64)], [(128, 64)])
    fixed = next(r for r in rows if r["knob"] == "fixed")
    assert fixed["T_ratio"] == 1 and fixed["E_ratio"] == 1 and fixed["xbar_delta"] == 0; n += 1
    assert any(r["status"] != "ok" for r in rows) and all("reason" in r for r in rows if r["status"] != "ok"); n += 1
    assert {r["knob"] for r in rows} == set(KNOBS) | {"fixed"}; n += 1
    s = summarize(rows)
    assert len(s) == sum(len(v) for v in ((1, 2, 4), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)))
    n += 1
    pp = next(x for x in s if (x["knob"], x["value"]) == ("P_PROJ", "1"))
    assert pp["T_ratio"]["median"] == 1 and pp["E_ratio"]["median"] == 1; n += 1
    ne = next(x for x in s if (x["knob"], x["value"]) == ("N_E", "4d"))
    assert ne["xbar_delta"]["min"] == 2 * 64 and ne["E_ratio"]["median"] == 1, ne; n += 1   # more crossbars, same passes
    text = document(s, rows, [("BERT-Tiny", 128, 128, 64)], [(128, 64)])
    assert "| P_PROJ | 4 |" in text and "| N_OC | cap |" in text and "## Cycles ÷ fixed per config" in text; n += 1
    pc = perConfig(rows, [(128, 64)])
    assert "| P_PROJ | 1 | 1.00 |" in pc, pc; n += 1
    # the balanced P_proj*: hand-computed (BERT-Tiny 128x64: T_p 34, S 64 -> 2.66 -> 4, cap I/16 = 4; BERT-base 1024x64:
    # T_p 162 -> 12.7 -> 16, capped at 4; JetFormer-150P 128x128: T_p 34, S 32 -> 5.3 -> 8, cap 8), a power of two <= n
    assert balancedPproj(128, 128, 128, 64) == 4 and balancedPproj(128, 768, 1024, 64) == 4; n += 1
    assert balancedPproj(32, 64, 128, 128) == 8 and balancedPproj(32, 64, 128, 64) == 4; n += 1
    assert pow2up(2.66) == 4 and pow2up(4) == 4 and pow2up(1) == 1 and pow2up(0.5) == 1; n += 1
    for name, nseq, dm, d in SHAPES:
        for R, C in CONFIGS:
            s = balancedPproj(nseq, dm, R, C)
            assert s & (s - 1) == 0 and 1 <= s <= min(nseq, min(R, C) // 16); n += 1
    b = balanced([("BERT-Tiny", 128, 128, 64)], [(128, 64)])
    assert b[0]["P_star"] == 4 and b[0]["T_ratio"] < 1 and b[0]["xbar_bal"] - b[0]["xbar_fixed"] == 18; n += 1
    assert "| 128×64 | 4 |" in balancedDocument(b, [("BERT-Tiny", 128, 128, 64)], [(128, 64)]); n += 1
    # the optimizer on a 4-point grid and two configs: the fixed mapping is in the grid, every optimum is <= it, the
    # rankings cover both configs, the document renders
    tiny = lambda d: [("P1 N_E=2d N_EP=2d N_EXP=2 BKV=16", dict(P_PROJ=1, N_E=2 * d, N_EP=2 * d, N_EXP=2, BKV=16)),   # noqa: E731
                      ("P4 N_E=2d N_EP=2d N_EXP=2 BKV=16", dict(P_PROJ=4, N_E=2 * d, N_EP=2 * d, N_EXP=2, BKV=16)),
                      ("P1 N_E=d N_EP=2d N_EXP=1 BKV=16", dict(P_PROJ=1, N_E=d, N_EP=2 * d, N_EXP=1, BKV=16)),
                      ("P1 N_E=2d N_EP=d N_EXP=2 BKV=16", dict(P_PROJ=1, N_E=2 * d, N_EP=d, N_EXP=2, BKV=16))]
    res = optimize([("BERT-Tiny", 128, 128, 64), ("JetFormer-150P", 32, 64, 32)], [(128, 64), (256, 128)], grid=tiny)
    assert len(res["builds"]) == 4 and all(b["grid"] == 3 for b in res["builds"]), [b["grid"] for b in res["builds"]]
    n += 1                                                                      # N_EP = d is outside on every build
    for b in res["builds"]:
        for obj in OBJECTIVES:
            assert b["best"][obj]["ratio"] <= 1 + 1e-12, (b["shape"], b["config"], obj); n += 1
        assert b["best"]["cycles"]["P_PROJ"] == 4 and b["best"]["area"]["N_E"] == b["d"], b["shape"]; n += 1
    rk = rankings(res)
    assert set(rk) == {"fixed", *OBJECTIVES} and set(rk["adp"]) == {"128x64", "256x128"}; n += 1
    assert {rk["fixed"][c]["rank_adp"] for c in rk["fixed"]} == {1, 2}; n += 1
    doc = optimumDocument(res, rk, [(128, 64), (256, 128)])
    assert "## A." in doc and "| BERT-Tiny | 128×64 |" in doc; n += 1
    print(f"mapping_sensitivity self-test: ALL PASS ({n} checks)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", default=None, help="comma-separated RxC (default: the 12)")
    ap.add_argument("--shapes", default=None, help="comma-separated shape names (default: the 6)")
    ap.add_argument("--balanced", action="store_true", help="P_proj* per build instead of the one-at-a-time sweep")
    ap.add_argument("--optimize", action="store_true", help="the grid per build, the optimum per objective, the rankings")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return selfTest()
    configs = CONFIGS if a.configs is None else [tuple(int(x) for x in c.split("x")) for c in a.configs.split(",")]
    shapes = SHAPES if a.shapes is None else [s for s in SHAPES if s[0] in a.shapes.split(",")]
    if a.optimize:
        res = optimize(shapes, configs)
        ranks = rankings(res)
        text = optimumDocument(res, ranks, configs)
        print(text)
        OUT_OPT_MD.write_text(text)
        OUT_OPT_JSON.write_text(json.dumps(dict(res, rankings=ranks), indent=1) + "\n")
        print(f"-> {OUT_OPT_MD}\n-> {OUT_OPT_JSON}")
        return 0
    if a.balanced:
        rows = balanced(shapes, configs)
        text = balancedDocument(rows, shapes, configs)
        print(text)
        OUT_BAL_MD.write_text(text)
        OUT_BAL_JSON.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"-> {OUT_BAL_MD}\n-> {OUT_BAL_JSON}")
        return 0
    rows = sweep(shapes, configs)
    summary = summarize(rows)
    text = document(summary, rows, shapes, configs)
    print(text)
    OUT_MD.write_text(text)
    OUT_JSON.write_text(json.dumps(dict(knobs=KNOBS, summary=summary, rows=rows), indent=1) + "\n")
    print(f"-> {OUT_MD}\n-> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
