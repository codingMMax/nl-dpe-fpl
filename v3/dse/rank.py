#!/usr/bin/env python3
"""rank.py — per-family EDAP ranking of the crossbar configs and the FC x attention cross score (docs/experiment.md
§5 ranking row, §6 step 4).

Ported from v2/experiment/dse/rank_dse.py (the metric and the factor view unchanged; the inputs are v3's). The raw
data only (user, 2026-10-16: no resampling): every point uses its build's Fmax as measured, the mean of its three VTR
seeds. Per family (FC, attention), for config c and workload w:
  EDAP(c, w) = E[pJ] * D[ns] * A[mm^2]
  score(c, w) = min_c' EDAP(c', w) / EDAP(c, w)            (1.0 = the best config on w)
  G(c)        = geometric mean of score(c, w) over the family's workloads; rank by G, descending.
  factors     for E, D and A separately: value / best value of that factor on the workload, geometric mean over the
              family's workloads (1.0 = best on every workload).
Across the families (user, 2026-10-16): each config is the point (1 - G_FC, 1 - G_ATT); the origin is a config best
on every workload of both families, and a config is better the closer it lies: d = sqrt((1 - G_FC)^2 + (1 - G_ATT)^2).
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
RESULTS = V3 / "dse" / "results"
EDAP = RESULTS / "edap"            # the earlier view: energy x delay x area product ranking
ADP = RESULTS / "adp"              # the current view: latency vs area per family, area-delay product across them
ATT_CSV = RESULTS / "att_priced_72.csv"
SNAP = V3 / "dse" / "fc_snapshot"
MM2_PER_MWTA = 0.033864e-6          # 1 MWTA = 0.033864 um^2


def geomean(xs: list) -> float:
    if not xs:
        raise ValueError("geomean of nothing")
    for x in xs:
        if not (x > 0) or math.isinf(x):
            raise ValueError(f"geomean needs finite positive values, got {x}")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def rank_configs(edap: dict) -> dict:
    """edap: {workload: {config: EDAP > 0}}. Every workload must cover the same configs (a missing point is never
    silently scored). Returns {config: (G, {workload: score})}."""
    if not edap:
        raise ValueError("no workloads")
    configs = None
    for w, row in edap.items():
        if not row:
            raise ValueError(f"workload {w} has no configs")
        if configs is None:
            configs = set(row)
        elif set(row) != configs:
            raise ValueError(f"workload {w} covers {sorted(set(row) ^ configs)} differently")
    out = {}
    best = {w: min(row.values()) for w, row in edap.items()}
    for c in sorted(configs):
        scores = {}
        for w, row in edap.items():
            v = row[c]
            if not (v > 0) or math.isinf(v):
                raise ValueError(f"EDAP({c}, {w}) = {v}")
            scores[w] = best[w] / v
        out[c] = (geomean(list(scores.values())), scores)
    return out


def order(ranked: dict) -> list:
    """Configs by G descending; ties broken by config name for a stable order."""
    return sorted(ranked, key=lambda c: (-ranked[c][0], c))


def factor_view(vals: dict) -> dict:
    """vals: {workload: {config: value > 0}} -> {config: geomean_w(value / min value on w)}."""
    return {c: geomean([vals[w][c] / min(vals[w].values()) for w in vals]) for c in next(iter(vals.values()))}


def family_ranking(points: list) -> dict:
    """points: [dict(workload, config, E [pJ], D [ns], A [mm^2])]."""
    E, D, A, edap = {}, {}, {}, {}
    for p in points:
        w, c = p["workload"], p["config"]
        for tab, v in ((E, p["E"]), (D, p["D"]), (A, p["A"]), (edap, p["E"] * p["D"] * p["A"])):
            tab.setdefault(w, {})[c] = v
    ranked = rank_configs(edap)
    fE, fD, fA = factor_view(E), factor_view(D), factor_view(A)
    configs = []
    for i, c in enumerate(order(ranked)):
        R, C = (int(x) for x in c.split("x"))
        configs.append(dict(config=c, R=R, C=C, rank=i + 1, G=ranked[c][0], scores=ranked[c][1],
                            factor_E=fE[c], factor_D=fD[c], factor_A=fA[c],
                            edap={w: edap[w][c] for w in edap}, energy_pj={w: E[w][c] for w in E},
                            latency_ns={w: D[w][c] for w in D}, area_mm2={w: A[w][c] for w in A}))
    return dict(workloads=list(edap), metric="EDAP = pJ * ns * mm^2; score = best/EDAP per workload; G = geometric mean "
                "over workloads; Fmax = the mean of the 3 VTR seeds", best_per_workload={w: min(edap[w], key=edap[w].get)
                                                                                         for w in edap}, configs=configs)


def cross_ranking(fc: dict, at: dict) -> dict:
    """The FC x attention cross score: per config the point (1 - G_FC, 1 - G_ATT) and its distance d from the origin
    (smaller is better), and the front: configs no other config beats on both axes."""
    f = {c["config"]: c for c in fc["configs"]}
    a = {c["config"]: c for c in at["configs"]}
    if set(f) != set(a):
        raise ValueError(f"the families cover different configs: {sorted(set(f) ^ set(a))}")
    xy = {c: (1 - f[c]["G"], 1 - a[c]["G"]) for c in f}
    d = {c: math.hypot(*xy[c]) for c in f}
    front = {c for c in f if not any(xy[o][0] <= xy[c][0] and xy[o][1] <= xy[c][1] and xy[o] != xy[c] for o in f)}
    return dict(metric="d = sqrt((1 - G_FC)^2 + (1 - G_ATT)^2): distance from the origin, the config best on every "
                "workload of both families; smaller is better",
                configs=[dict(config=c, rank=k + 1, x=xy[c][0], y=xy[c][1], d=d[c], front=c in front, G_FC=f[c]["G"],
                              G_ATT=a[c]["G"]) for k, c in enumerate(sorted(f, key=lambda c: (d[c], c)))])


def _front_rows(a: dict, lat: dict, en: dict) -> list:
    """Rows of (area score, latency score, energy score) per config; front = no other config at or below on area and
    latency and below on one (energy is reported beside the front, not part of it)."""
    front = {c for c in a if not any(a[o] <= a[c] and lat[o] <= lat[c] and (a[o], lat[o]) != (a[c], lat[c]) for o in a)}
    return [dict(config=c, area_score=a[c], latency_score=lat[c], energy_score=en[c], front=c in front)
            for c in sorted(a, key=lambda c: (a[c], lat[c], c))]


def la_scores(points: list) -> dict:
    """Latency vs area (user, 2026-10-16): per workload, a config's area score = its area / the largest area of any
    config on that workload, its latency score = its latency / the highest latency (the worst config is 1; lower is
    better); per config, the geometric mean over the family's workloads. The energy score (energy / the highest energy,
    the same way; user 2026-10-16) is reported beside, outside the front."""
    D, A, E = {}, {}, {}
    for p in points:
        D.setdefault(p["workload"], {})[p["config"]] = p["D"]
        A.setdefault(p["workload"], {})[p["config"]] = p["A"]
        E.setdefault(p["workload"], {})[p["config"]] = p["E"]
    cfgs = list(next(iter(D.values())))                       # the points' order (a set would vary the JSON key order)
    for w in D:
        if set(D[w]) != set(cfgs) or set(A[w]) != set(cfgs) or set(E[w]) != set(cfgs):
            raise ValueError(f"workload {w} covers other configs")
    sc = lambda X: {c: geomean([X[w][c] / max(X[w].values()) for w in X]) for c in cfgs}  # noqa: E731
    return dict(workloads=list(D), configs=_front_rows(sc(A), sc(D), sc(E)))


def adp_scores(points: list) -> dict:
    """Area-delay product (user, 2026-10-16): per workload, a config's ADP score = its area x latency / the largest
    area x latency of any config on that workload (the worst config is 1; lower is better); per config, the geometric
    mean over the family's workloads."""
    P = {}
    for p in points:
        P.setdefault(p["workload"], {})[p["config"]] = p["A"] * p["D"]
    cfgs = list(next(iter(P.values())))                       # the points' order (a set would vary the JSON key order)
    if any(set(P[w]) != set(cfgs) for w in P):
        raise ValueError("the workloads cover different configs")
    return dict(workloads=list(P), scores={c: geomean([P[w][c] / max(P[w].values()) for w in P]) for c in cfgs})


def adp_cross(fc: dict, at: dict, energy: dict | None = None) -> dict:
    """Across the families (user, 2026-10-16): x = the config's FC ADP score, y = its attention ADP score; the front:
    configs no other config beats in both families (at or below in both, below in one); d = the point's distance from
    the origin, sqrt(x^2 + y^2) (user: circles of equal distance), rows in order of d. energy: {config: (FC energy
    score, attention energy score)}, reported beside."""
    f, t = fc["scores"], at["scores"]
    if set(f) != set(t):
        raise ValueError(f"the families cover different configs: {sorted(set(f) ^ set(t))}")
    front = {c for c in f if not any(f[o] <= f[c] and t[o] <= t[c] and (f[o], t[o]) != (f[c], t[c]) for o in f)}
    d = {c: math.hypot(f[c], t[c]) for c in f}
    return dict(configs=[dict(config=c, rank=k + 1, adp_fc=f[c], adp_att=t[c], d=d[c], front=c in front,
                              **({} if energy is None else dict(energy_fc=energy[c][0], energy_att=energy[c][1])))
                         for k, c in enumerate(sorted(f, key=lambda c: (d[c], c)))])


def attention_points(path: Path = ATT_CSV) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 72 or any(not r.get("edap") for r in rows):
        raise SystemExit(f"{path}: expected 72 priced builds (dse/att_rows.py --vtr), got "
                         f"{sum(bool(r.get('edap')) for r in rows)} of {len(rows)}")
    return [dict(workload=r["shape"], config=r["config"], E=float(r["e_total_pj"]), D=float(r["latency_ns"]),
                 A=float(r["area_total_mm2"])) for r in rows]


def fc_points() -> list:
    from simulator.platform.rows import fcRow
    from simulator.platform import area
    from simulator.platform.platform import Platform
    vtr = area.vtrTable(SNAP / "vtr_results.csv")
    with open(SNAP / "dse_v2.csv", newline="") as f:
        snap = [r for r in csv.DictReader(f) if r["family"] == "gemm"]
    pts = []
    for r in snap:
        R, C, M, K, N = (int(r[k]) for k in ("R", "C", "M", "K", "N"))
        row = fcRow(R, C, (M, K, N), Platform("nl_dpe", rows=R, cols=C), vtr, passExtent=r["pass_extent"])
        pts.append(dict(workload=f"{M}x{K}x{N}", config=row["config"], E=row["e_total_pj"], D=row["latency_ns"],
                        A=row["area_total_mwta"] * MM2_PER_MWTA))
    if len(pts) != 48:
        raise SystemExit(f"FC: {len(pts)} rows, expected 48")
    return pts


def _self_test() -> None:
    ok = 0
    r = rank_configs({"w": {"a": 2.0, "b": 4.0}})          # 1 one workload: G = score; the best scores exactly 1
    assert r["a"][0] == 1.0 and r["b"][0] == 0.5; ok += 1
    base = {"w1": {"a": 1.0, "b": 3.0, "c": 2.0}, "w2": {"a": 5.0, "b": 1.0, "c": 2.0}}
    big = {"w1": {k: v * 1e9 for k, v in base["w1"].items()}, "w2": dict(base["w2"])}
    for c in "abc":                                       # 2 scale invariance per workload
        assert math.isclose(rank_configs(base)[c][0], rank_configs(big)[c][0], rel_tol=1e-12); ok += 1

    def by_ref(ed, ref):
        return {c: geomean([ed[w][ref] / ed[w][c] for w in ed]) for c in ed["w1"]}
    rb = rank_configs(base)
    for ref in "abc":                                     # 3 normalizing to any fixed config keeps the order
        br = by_ref(base, ref)
        assert sorted(br, key=lambda c: (-br[c], c)) == order(rb); ok += 1
    t = rank_configs({"w": {"y": 1.0, "x": 1.0}})         # 4 ties: equal G, stable order by name
    assert t["x"][0] == t["y"][0] == 1.0 and order(t) == ["x", "y"]; ok += 1
    assert math.isclose(rb["a"][0], math.sqrt(1 / 5)) and math.isclose(rb["b"][0], math.sqrt(1 / 3))   # 5 by hand
    assert math.isclose(rb["c"][0], 0.5) and order(rb) == ["b", "c", "a"]; ok += 1
    for bad in ({"w": {"a": 0.0, "b": 1.0}}, {"w": {"a": -1.0, "b": 1.0}}, {"w": {"a": float("nan"), "b": 1.0}},
                {"w": {"a": float("inf"), "b": 1.0}}, {"w1": {"a": 1.0, "b": 1.0}, "w2": {"a": 1.0}}, {}, {"w": {}}):
        try:                                              # 6 rejected: zero, negative, NaN, inf, a missing point, empty
            rank_configs(bad)
        except ValueError:
            ok += 1
        else:
            raise AssertionError(f"accepted {bad}")
    f = factor_view({"w1": {"a": 1.0, "b": 2.0}, "w2": {"a": 3.0, "b": 6.0}})   # 7 factors
    assert f == {"a": 1.0, "b": 2.0}; ok += 1
    pts = [dict(workload=f"w{n}", config=c, E=e, D=dd, A=1.0)   # 8 family_ranking by hand: EDAP = E x D
           for c, n, e, dd in (("1x1", 1, 1.0, 1.0), ("2x2", 1, 2.0, 1.0), ("1x1", 2, 3.0, 2.0), ("2x2", 2, 1.0, 2.0))]
    fr = family_ranking(pts)
    g = {c["config"]: c["G"] for c in fr["configs"]}
    assert math.isclose(g["1x1"], math.sqrt(1 * (2 / 6))) and math.isclose(g["2x2"], math.sqrt(0.5 * 1))
    assert [c["config"] for c in fr["configs"]] == ["2x2", "1x1"]; ok += 1

    def fam(gs):                                          # 9 cross score: family results with fixed G per config
        return dict(configs=[dict(config=c, G=v) for c, v in gs.items()])
    cr = cross_ranking(fam({"a": 1.0, "b": 0.5}), fam({"a": 1.0, "b": 0.5}))
    assert cr["configs"][0]["config"] == "a" and cr["configs"][0]["d"] == 0.0     # best in both: the origin
    assert math.isclose(cr["configs"][1]["d"], math.hypot(0.5, 0.5)) and not cr["configs"][1]["front"]; ok += 1
    cr = cross_ranking(fam({"bal": 0.62, "att": 0.52, "far": 0.30}), fam({"bal": 0.57, "att": 0.75, "far": 0.30}))
    assert [c["config"] for c in cr["configs"]] == ["att", "bal", "far"] and min(0.62, 0.57) > min(0.52, 0.75)
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
    assert (la["s"]["area_score"], la["s"]["latency_score"]) == (0.5, 1.0)
    assert (la["f"]["area_score"], la["f"]["latency_score"]) == (1.0, 0.5)
    assert (la["x"]["area_score"], la["x"]["latency_score"]) == (1.0, 1.0)
    assert la["s"]["front"] and la["f"]["front"] and not la["x"]["front"]
    assert all(c["energy_score"] == 1.0 for c in la.values())       # equal energies: all 1
    le = {c["config"]: c for c in la_scores([dict(p, E={"s": 1.0, "f": 4.0, "x": 2.0}[p["config"]]) for p in one])
          ["configs"]}                                               # energy / the highest, beside the front
    assert all(math.isclose(le[c]["energy_score"], v) for c, v in (("s", .25), ("f", 1.0), ("x", .5)))
    assert [le[c]["front"] for c in "sfx"] == [la[c]["front"] for c in "sfx"]; ok += 1
    two = one + [dict(p, workload="v", A=p["A"] * 1e6, D=p["D"] * 1e-3) for p in one]   # 12 per-workload scale
    lb = {c["config"]: c for c in la_scores(two)["configs"]}
    assert all(math.isclose(lb[c]["area_score"], la[c]["area_score"]) and
               math.isclose(lb[c]["latency_score"], la[c]["latency_score"]) for c in la); ok += 1
    ad = adp_scores(one)["scores"]                        # 13 ADP: area x latency / the largest, the worst is 1
    assert (ad["s"], ad["f"], ad["x"]) == (0.5, 0.5, 1.0); ok += 1
    att = adp_scores([dict(p, D={"s": 1.0, "f": 4.0, "x": 4.0}[p["config"]]) for p in one])   # s 1, f 8, x 8 -> /8
    cr = adp_cross(adp_scores(one), att, {"s": (.1, .2), "f": (.3, .4), "x": (.5, .6)})   # 14 x = FC, y = attention
    ac = {c["config"]: c for c in cr["configs"]}
    near = lambda p, q: all(math.isclose(u, v) for u, v in zip(p, q))  # noqa: E731  (exp(log(x)) is not exact)
    assert near((ac["s"]["adp_fc"], ac["s"]["adp_att"]), (0.5, 0.125)) and near((ac["f"]["adp_fc"], ac["f"]["adp_att"]),
                                                                                (0.5, 1.0))
    assert ac["s"]["front"] and not ac["f"]["front"] and not ac["x"]["front"]
    assert near([ac[c]["d"] for c in "sfx"], [math.hypot(.5, .125), math.hypot(.5, 1), math.hypot(1, 1)])  # distance
    assert [c["config"] for c in cr["configs"]] == ["s", "f", "x"] and [c["rank"] for c in cr["configs"]] == [1, 2, 3]
    assert (ac["f"]["energy_fc"], ac["f"]["energy_att"]) == (.3, .4); ok += 1
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
              + ", ".join(f"{c['config']} G={c['G']:.3f}" for c in res["configs"][:3]))
    cr = cross_ranking(out["fc"], out["attention"])
    path = EDAP / "ranking_cross.json"
    path.write_text(json.dumps(cr, indent=1) + "\n")
    print(f"cross (distance of (1 - G_FC, 1 - G_ATT) from the origin) -> {path}; top 3: "
          + ", ".join(f"{c['config']} d={c['d']:.3f}" for c in cr["configs"][:3]))
    en = {c["config"]: [c["energy_score"]] for c in la["fc"]["configs"]}
    for c in la["attention"]["configs"]:
        en[c["config"]].append(c["energy_score"])
    path = ADP / "latency_area.json"
    path.write_text(json.dumps(dict(
        metric="per family, per workload: area / largest area and latency / highest latency (the worst config is 1, "
               "lower is better), geometric mean over the family's workloads; across the families: x = the FC area-"
               "delay product score, y = the attention one (ADP / the largest ADP per workload, geometric mean)",
        **la, adp=dict(fc=adp["fc"], attention=adp["attention"], cross=adp_cross(adp["fc"], adp["attention"], en))),
        indent=1) + "\n")
    for name in ("fc", "attention"):
        print(f"latency vs area, {name}: front " + ", ".join(
            f"{c['config']} (area {c['area_score']:.3f}, latency {c['latency_score']:.3f})"
            for c in la[name]["configs"] if c["front"]))
    print("ADP across the families (x = FC, y = attention): front " + ", ".join(
        f"{c['config']} ({c['adp_fc']:.3f}, {c['adp_att']:.3f}, d {c['d']:.4f})"
        for c in adp_cross(adp["fc"], adp["attention"])["configs"] if c["front"]))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
