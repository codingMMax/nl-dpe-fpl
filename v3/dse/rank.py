#!/usr/bin/env python3
"""rank.py — the ranking of the crossbar configs per family and across the families (docs/experiment.md §5 ranking
row, §6 step 4), in two views: EDAP (energy x delay x area) and latency vs area with the area-delay product (ADP).

One notation for both views (user, 2026-10-17). Per family (FC, attention), for config c, workload w and quantity X:
  score_X(c, w) = X(c, w) / max_c' X(c', w)      (1.0 = the worst config on w; lower is better)
  G_X(c)        = geometric mean of score_X(c, w) over the family's workloads
with X = A (area, mm^2), D (latency, ns), E (energy, pJ), ADP = A * D, EDAP = E * D * A. Every config of a workload is
divided by the same number, so the ranks and the Pareto fronts do not depend on the normalizer; the cross-family
distance d does.
  EDAP view: rank by G_EDAP, ascending; G_E, G_D, G_A beside. Across the families: the point (G_FC, G_ATT) = the
             G_EDAP of each family, d = its distance from the origin (smaller is better).
  latency vs area view: per family the point (G_A, G_D) and its Pareto front, G_E beside; across the families the
             point (G_FC, G_ATT) = the G_ADP of each family, its front and d.
Ported from v2/experiment/dse/rank_dse.py (the inputs are v3's). The raw data only (user, 2026-10-16: no resampling):
every point uses its build's Fmax as measured, the mean of its three VTR seeds.
Inputs:
  attention  the 72 builds priced by dse/att_rows.py --vtr (dse/results/att_priced_72.csv): workloads = the six shapes
  FC         the 48 rows of the v2 snapshot (dse/fc_snapshot/) rebuilt by the FC workload
             (simulator/platform/rows.py fcRow, as simulator/tests/testFcRows.py checks them): workloads = the four
             M x K x N shapes
Usage: python3 v3/dse/rank.py              # dse/results/edap/ranking_{fc,attention,cross}.json, adp/latency_area.json
       python3 v3/dse/rank.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse import data  # noqa: E402

RESULTS = data.RESULTS
EDAP = RESULTS / "edap"            # the earlier view: energy x delay x area product ranking
ADP = RESULTS / "adp"              # the current view: latency vs area per family, area-delay product across them


def geomean(xs: list) -> float:
    if not xs:
        raise ValueError("geomean of nothing")
    for x in xs:
        if not (x > 0) or math.isinf(x):
            raise ValueError(f"geomean needs finite positive values, got {x}")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def normalized(vals: dict) -> dict:
    """vals: {workload: {config: value > 0}}. Every workload must cover the same configs (a missing point is never
    silently scored). Returns {config: (G, {workload: score})}, score = value / the largest value on the workload,
    G = their geometric mean; configs in the first workload's order (the JSON key order stays fixed)."""
    if not vals:
        raise ValueError("no workloads")
    configs = None
    for w, row in vals.items():
        if not row:
            raise ValueError(f"workload {w} has no configs")
        if configs is None:
            configs = list(row)
        elif set(row) != set(configs):
            raise ValueError(f"workload {w} covers {sorted(set(row) ^ set(configs))} differently")
    for w, row in vals.items():
        for c, v in row.items():
            if not (v > 0) or math.isinf(v):
                raise ValueError(f"value({c}, {w}) = {v}")
    worst = {w: max(row.values()) for w, row in vals.items()}
    out = {}
    for c in configs:
        scores = {w: row[c] / worst[w] for w, row in vals.items()}
        out[c] = (geomean(list(scores.values())), scores)
    return out


def G_of(vals: dict) -> dict:
    """{config: G}: the geometric mean of value / the largest value over the workloads."""
    return {c: g for c, (g, _) in normalized(vals).items()}


def order(ranked: dict) -> list:
    """Configs by G ascending (lower is better); ties broken by config name for a stable order."""
    return sorted(ranked, key=lambda c: (ranked[c][0], c))


def family_ranking(points: list) -> dict:
    """points: [dict(workload, config, E [pJ], D [ns], A [mm^2])]."""
    E, D, A, edap = {}, {}, {}, {}
    for p in points:
        w, c = p["workload"], p["config"]
        for tab, v in ((E, p["E"]), (D, p["D"]), (A, p["A"]), (edap, p["E"] * p["D"] * p["A"])):
            tab.setdefault(w, {})[c] = v
    ranked = normalized(edap)
    gE, gD, gA = G_of(E), G_of(D), G_of(A)
    configs = []
    for i, c in enumerate(order(ranked)):
        R, C = (int(x) for x in c.split("x"))
        configs.append(dict(config=c, R=R, C=C, rank=i + 1, G_EDAP=ranked[c][0], scores=ranked[c][1],
                            G_E=gE[c], G_D=gD[c], G_A=gA[c],
                            edap={w: edap[w][c] for w in edap}, energy_pj={w: E[w][c] for w in E},
                            latency_ns={w: D[w][c] for w in D}, area_mm2={w: A[w][c] for w in A}))
    return dict(workloads=list(edap), metric="EDAP = pJ * ns * mm^2; per workload, score = EDAP / the largest EDAP "
                "(1 = the worst config); G_EDAP = the geometric mean over the workloads, lower is better; G_E, G_D, "
                "G_A the same for energy, latency and area; Fmax = the mean of the 3 VTR seeds",
                best_per_workload={w: min(edap[w], key=edap[w].get) for w in edap}, configs=configs)


def cross_point(f: dict, t: dict, energy: dict | None = None) -> dict:
    """Across the families: f, t = {config: G} of FC and attention; each config is the point (G_FC, G_ATT); the front:
    configs no other config beats in both families (at or below in both, below in one); d = the point's distance from
    the origin (user: circles of equal distance), rows in order of d. energy: {config: (G_E in FC, G_E in attention)},
    reported beside."""
    if set(f) != set(t):
        raise ValueError(f"the families cover different configs: {sorted(set(f) ^ set(t))}")
    front = {c for c in f if not any(f[o] <= f[c] and t[o] <= t[c] and (f[o], t[o]) != (f[c], t[c]) for o in f)}
    d = {c: math.hypot(f[c], t[c]) for c in f}
    return dict(configs=[dict(config=c, rank=k + 1, G_FC=f[c], G_ATT=t[c], d=d[c], front=c in front,
                              **({} if energy is None else dict(G_E_FC=energy[c][0], G_E_ATT=energy[c][1])))
                         for k, c in enumerate(sorted(f, key=lambda c: (d[c], c)))])


def cross_ranking(fc: dict, at: dict) -> dict:
    """EDAP across the families: the point (G_EDAP in FC, G_EDAP in attention), its front and d."""
    return dict(metric="the point (G_FC, G_ATT) = (G_EDAP in FC, G_EDAP in attention); d = its distance from the "
                "origin; smaller is better",
                **cross_point({c["config"]: c["G_EDAP"] for c in fc["configs"]},
                              {c["config"]: c["G_EDAP"] for c in at["configs"]}))


def _front_rows(a: dict, lat: dict, en: dict) -> list:
    """Rows of (G_A, G_D, G_E) per config; front = no other config at or below on area and latency and below on one
    (energy is reported beside the front, not part of it)."""
    front = {c for c in a if not any(a[o] <= a[c] and lat[o] <= lat[c] and (a[o], lat[o]) != (a[c], lat[c]) for o in a)}
    return [dict(config=c, G_A=a[c], G_D=lat[c], G_E=en[c], front=c in front)
            for c in sorted(a, key=lambda c: (a[c], lat[c], c))]


def la_scores(points: list) -> dict:
    """Latency vs area (user, 2026-10-16): per family, each config's G_A (area) and G_D (latency); G_E (energy) is
    reported beside, outside the front."""
    D, A, E = {}, {}, {}
    for p in points:
        D.setdefault(p["workload"], {})[p["config"]] = p["D"]
        A.setdefault(p["workload"], {})[p["config"]] = p["A"]
        E.setdefault(p["workload"], {})[p["config"]] = p["E"]
    return dict(workloads=list(D), configs=_front_rows(G_of(A), G_of(D), G_of(E)))


def adp_scores(points: list) -> dict:
    """Area-delay product (user, 2026-10-16): per family, each config's G_ADP (area x latency / the largest on each
    workload, geometric mean)."""
    P = {}
    for p in points:
        P.setdefault(p["workload"], {})[p["config"]] = p["A"] * p["D"]
    return dict(workloads=list(P), G_ADP=G_of(P))


def adp_cross(fc: dict, at: dict, energy: dict | None = None) -> dict:
    """ADP across the families (user, 2026-10-16): the point (G_ADP in FC, G_ADP in attention), its front and d."""
    return cross_point(fc["G_ADP"], at["G_ADP"], energy)


def attention_points() -> list:
    """(workload, config, E [pJ], D [ns], A [mm^2]) per priced attention build (dse/data.py)."""
    return [dict(workload=r["shape"], config=r["config"], E=float(r["e_total_pj"]), D=float(r["latency_ns"]),
                 A=float(r["area_total_mm2"])) for r in data.attentionRows()]


def fc_points() -> list:
    """The same for the 48 FC rows of the v2 snapshot, rebuilt by the v3 model (dse/data.py)."""
    return [dict(workload=data.fcWorkload(r), config=r["config"], E=r["e_total_pj"], D=r["latency_ns"],
                 A=r["area_total_mwta"] * data.MM2_PER_MWTA) for r in data.fcRows()]


def _self_test() -> None:
    ok = 0
    r = normalized({"w": {"a": 2.0, "b": 4.0}})            # 1 one workload: G = score; the worst scores exactly 1
    assert r["a"][0] == 0.5 and r["b"][0] == 1.0; ok += 1
    base = {"w1": {"a": 1.0, "b": 3.0, "c": 2.0}, "w2": {"a": 5.0, "b": 1.0, "c": 2.0}}
    big = {"w1": {k: v * 1e9 for k, v in base["w1"].items()}, "w2": dict(base["w2"])}
    for c in "abc":                                       # 2 scale invariance per workload
        assert math.isclose(normalized(base)[c][0], normalized(big)[c][0], rel_tol=1e-12); ok += 1

    def by_ref(ed, ref):
        return {c: geomean([ed[w][c] / ed[w][ref] for w in ed]) for c in ed["w1"]}
    rb = normalized(base)
    for ref in "abc":                                     # 3 normalizing to any fixed config keeps the order
        br = by_ref(base, ref)
        assert sorted(br, key=lambda c: (br[c], c)) == order(rb); ok += 1
    t = normalized({"w": {"y": 1.0, "x": 1.0}})           # 4 ties: equal G, stable order by name
    assert t["x"][0] == t["y"][0] == 1.0 and order(t) == ["x", "y"]; ok += 1
    assert math.isclose(rb["a"][0], math.sqrt(1 / 3)) and math.isclose(rb["b"][0], math.sqrt(1 / 5))   # 5 by hand:
    assert math.isclose(rb["c"][0], math.sqrt(4 / 15)) and order(rb) == ["b", "c", "a"]; ok += 1         # worst 3, 5
    for bad in ({"w": {"a": 0.0, "b": 1.0}}, {"w": {"a": -1.0, "b": 1.0}}, {"w": {"a": float("nan"), "b": 1.0}},
                {"w": {"a": float("inf"), "b": 1.0}}, {"w1": {"a": 1.0, "b": 1.0}, "w2": {"a": 1.0}}, {}, {"w": {}}):
        try:                                              # 6 rejected: zero, negative, NaN, inf, a missing point, empty
            normalized(bad)
        except ValueError:
            ok += 1
        else:
            raise AssertionError(f"accepted {bad}")
    g = G_of({"w1": {"a": 1.0, "b": 2.0}, "w2": {"a": 3.0, "b": 6.0}})   # 7 G of one quantity
    assert math.isclose(g["a"], 0.5) and g["b"] == 1.0; ok += 1
    pts = [dict(workload=f"w{n}", config=c, E=e, D=dd, A=1.0)   # 8 family_ranking by hand: EDAP = E x D,
           for c, n, e, dd in (("1x1", 1, 1.0, 1.0), ("2x2", 1, 2.0, 1.0), ("1x1", 2, 3.0, 2.0), ("2x2", 2, 1.0, 2.0))]
    fr = family_ranking(pts)                              # the worst EDAP is 2 on w1 and 6 on w2
    g = {c["config"]: c["G_EDAP"] for c in fr["configs"]}
    assert math.isclose(g["1x1"], math.sqrt(0.5 * 1)) and math.isclose(g["2x2"], math.sqrt(1 * (2 / 6)))
    assert [c["config"] for c in fr["configs"]] == ["2x2", "1x1"]; ok += 1

    def fam(gs):                                          # 9 cross score: family results with fixed G_EDAP per config
        return dict(configs=[dict(config=c, G_EDAP=v) for c, v in gs.items()])
    cr = cross_ranking(fam({"a": 0.5, "b": 1.0}), fam({"a": 0.5, "b": 1.0}))
    assert cr["configs"][0]["config"] == "a" and math.isclose(cr["configs"][0]["d"], math.hypot(0.5, 0.5))
    assert cr["configs"][0]["front"] and not cr["configs"][1]["front"]; ok += 1      # a beats b in both families
    cr = cross_ranking(fam({"bal": 0.38, "att": 0.48, "far": 0.70}), fam({"bal": 0.43, "att": 0.25, "far": 0.70}))
    assert [c["config"] for c in cr["configs"]] == ["att", "bal", "far"] and max(0.38, 0.43) < max(0.48, 0.25)
    assert {c["config"] for c in cr["configs"] if c["front"]} == {"att", "bal"}; ok += 1   # distance, not max-min
    try:                                                  # 10 the families must cover the same configs
        cross_ranking(fam({"a": 1.0}), fam({"b": 1.0}))
    except ValueError:
        ok += 1
    else:
        raise AssertionError("cross_ranking accepted different config sets")
    one = [dict(workload="w", config=c, E=1.0, D=dd, A=aa) for c, aa, dd in (("s", 1.0, 4.0), ("f", 2.0, 2.0),
                                                                            ("x", 2.0, 4.0))]
    la = {c["config"]: c for c in la_scores(one)["configs"]}     # 11 latency vs area: value / max, the worst is 1
    assert (la["s"]["G_A"], la["s"]["G_D"]) == (0.5, 1.0)
    assert (la["f"]["G_A"], la["f"]["G_D"]) == (1.0, 0.5)
    assert (la["x"]["G_A"], la["x"]["G_D"]) == (1.0, 1.0)
    assert la["s"]["front"] and la["f"]["front"] and not la["x"]["front"]
    assert all(c["G_E"] == 1.0 for c in la.values())               # equal energies: all 1
    le = {c["config"]: c for c in la_scores([dict(p, E={"s": 1.0, "f": 4.0, "x": 2.0}[p["config"]]) for p in one])
          ["configs"]}                                               # energy / the highest, beside the front
    assert all(math.isclose(le[c]["G_E"], v) for c, v in (("s", .25), ("f", 1.0), ("x", .5)))
    assert [le[c]["front"] for c in "sfx"] == [la[c]["front"] for c in "sfx"]; ok += 1
    two = one + [dict(p, workload="v", A=p["A"] * 1e6, D=p["D"] * 1e-3) for p in one]   # 12 per-workload scale
    lb = {c["config"]: c for c in la_scores(two)["configs"]}
    assert all(math.isclose(lb[c]["G_A"], la[c]["G_A"]) and math.isclose(lb[c]["G_D"], la[c]["G_D"]) for c in la)
    ok += 1
    ad = adp_scores(one)["G_ADP"]                         # 13 ADP: area x latency / the largest, the worst is 1
    assert (ad["s"], ad["f"], ad["x"]) == (0.5, 0.5, 1.0); ok += 1
    att = adp_scores([dict(p, D={"s": 1.0, "f": 4.0, "x": 4.0}[p["config"]]) for p in one])   # s 1, f 8, x 8 -> /8
    cr = adp_cross(adp_scores(one), att, {"s": (.1, .2), "f": (.3, .4), "x": (.5, .6)})   # 14 x = FC, y = attention
    ac = {c["config"]: c for c in cr["configs"]}
    near = lambda p, q: all(math.isclose(u, v) for u, v in zip(p, q))  # noqa: E731  (exp(log(x)) is not exact)
    assert near((ac["s"]["G_FC"], ac["s"]["G_ATT"]), (0.5, 0.125)) and near((ac["f"]["G_FC"], ac["f"]["G_ATT"]),
                                                                            (0.5, 1.0))
    assert ac["s"]["front"] and not ac["f"]["front"] and not ac["x"]["front"]
    assert near([ac[c]["d"] for c in "sfx"], [math.hypot(.5, .125), math.hypot(.5, 1), math.hypot(1, 1)])  # distance
    assert [c["config"] for c in cr["configs"]] == ["s", "f", "x"] and [c["rank"] for c in cr["configs"]] == [1, 2, 3]
    assert (ac["f"]["G_E_FC"], ac["f"]["G_E_ATT"]) == (.3, .4); ok += 1
    try:                                                  # 15 the families must cover the same configs
        adp_cross(adp_scores(one), adp_scores([dict(p, config=p["config"] + "y") for p in one]))
    except ValueError:
        ok += 1
    else:
        raise AssertionError("adp_cross accepted different config sets")
    print(f"rank self-test: ALL PASS ({ok} checks)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return 0
    EDAP.mkdir(parents=True, exist_ok=True)
    ADP.mkdir(parents=True, exist_ok=True)
    out, la, adp = {}, {}, {}
    for name, pts in (("fc", fc_points()), ("attention", attention_points())):
        la[name] = la_scores(pts)
        adp[name] = adp_scores(pts)
        res = dict(family=name, **family_ranking(pts))
        out[name] = res
        path = EDAP / f"ranking_{name}.json"
        path.write_text(json.dumps(res, indent=1) + "\n")
        print(f"{name}: {len(res['configs'])} configs x {len(res['workloads'])} workloads -> {path}; top 3: "
              + ", ".join(f"{c['config']} G_EDAP={c['G_EDAP']:.3f}" for c in res["configs"][:3]))
    cr = cross_ranking(out["fc"], out["attention"])
    path = EDAP / "ranking_cross.json"
    path.write_text(json.dumps(cr, indent=1) + "\n")
    print(f"EDAP across the families (the point (G_FC, G_ATT), d from the origin) -> {path}; top 3: "
          + ", ".join(f"{c['config']} d={c['d']:.3f}" for c in cr["configs"][:3]))
    en = {c["config"]: [c["G_E"]] for c in la["fc"]["configs"]}
    for c in la["attention"]["configs"]:
        en[c["config"]].append(c["G_E"])
    path = ADP / "latency_area.json"
    path.write_text(json.dumps(dict(
        metric="per workload, score = value / the worst config's value (1 = the worst, lower is better); G_X = the "
               "geometric mean of the scores over a family's workloads, X = A (area), D (latency), E (energy), "
               "ADP (A x D). Per family: (G_A, G_D) and its Pareto front, G_E beside; across the families: the point "
               "(G_FC, G_ATT) = the G_ADP of each family, its front and d = the distance from the origin",
        **la, adp=dict(fc=adp["fc"], attention=adp["attention"], cross=adp_cross(adp["fc"], adp["attention"], en))),
        indent=1) + "\n")
    for name in ("fc", "attention"):
        print(f"latency vs area, {name}: front " + ", ".join(
            f"{c['config']} (G_A {c['G_A']:.3f}, G_D {c['G_D']:.3f})" for c in la[name]["configs"] if c["front"]))
    print("ADP across the families (x = FC, y = attention): front " + ", ".join(
        f"{c['config']} ({c['G_FC']:.3f}, {c['G_ATT']:.3f}, d {c['d']:.4f})"
        for c in adp_cross(adp["fc"], adp["attention"])["configs"] if c["front"]))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
