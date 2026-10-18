#!/usr/bin/env python3
"""gen_softmax_online_cases.py — generate GATE-1-certified online softmax cases.

Every case is written through `NldpeSoftmaxOnline.dump_case`, which GATE-1-
certifies `sim == softmax_online_ref` (every stage) BEFORE any file is written,
and stamps the modelled cycles (`compute_cycles`) for the strict GATE-2 runner.

Shapes: scores are [S, N] (S rows per run, N keys streamed in N/Bkv blocks).
`--runs K` writes K back-to-back runs per case (r0/ .. r{K-1}/, each certified;
the TB re-arms the kernel between runs); run r uses class (i + r) mod #classes
with its own seed. Scores are seeded per (S, N, class) only, so the Bkv / n_exp
sweep of one shape sees identical input.

Presets (default `all` = every preset, one manifest):
  square  S = 16..128, N = S, Bkv = 8..64, n_exp 1/2/4, all classes
  s8      S = N = 8
  s256    S = N = 256
  rect    N != S: (16 x 64|256), (64 x 256|1024), (128 x 32|512), (8 x 64|512),
          (256 x 1024)
  runs    3 back-to-back runs per case
  rc      PRE-1 (2026-10-08): the 11 DSE geometries with R != C or I < 256 (R x C in
          {128,256,512,1024} x {64,128,256} minus 256 x 256), S = min(I, 128)
  nexp    PRE-1: n_exp 8 / 16 (R = C = 256 and R != C)
  gaps    PRE-1: `score_valid` gaps — producer offer edges per word (offer.mem, TB
          +OFFER=1): `inter` (late blocks, as the attention head's QK tiles deliver them),
          `intra` (bubbles inside a block), `both`; expected cycles from the availability
          model (`attention_sim.softmax_availability_model`, full extent = the RTL),
          which equals the certified model when no offer gates (its self-test)
Or give the axes directly (--Ss/--Ns/--BKV/--nExps/--classes/--runs).

The output dir accumulates across runs; the harness executes only the names
listed in `manifest.txt`.

Usage:
  python3 v2/sim/smoke/gen_softmax_online_cases.py                    # all presets
  python3 v2/sim/smoke/gen_softmax_online_cases.py --preset square,runs
  python3 v2/sim/smoke/gen_softmax_online_cases.py --Ss 64 --Ns S,256 --BKV 16 --nExps 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]  # v3 path (tools/copies.py)
sys.path.insert(0, str(REPO / "v3" / "oracle" / "v2_kernels"))  # v3 path (tools/copies.py)
sys.path.insert(0, str(REPO / "v3" / "oracle"))  # v3 path (tools/copies.py)
import softmax_online_ref as oref  # noqa: E402
import softmax_online_sim as osim  # noqa: E402
import attention_sim as asim  # noqa: E402

ALL = "random,narrow,extremes,uniform,ties,rising"
PRESETS = {
    "square": [dict(Ss="16,32,64,128", Ns="S", BKV="8,16,32,64", nExps="1,2,4", classes=ALL)],
    "s8": [dict(Ss="8", Ns="S", BKV="8", nExps="1,2,4", classes=ALL)],
    "s256": [dict(Ss="256", Ns="S", BKV="16,64", nExps="1,4", classes="random,narrow")],
    "rect": [dict(Ss="16", Ns="64,256", BKV="8,16,32", nExps="1,2,4", classes=ALL),
             dict(Ss="64", Ns="256,1024", BKV="16,64", nExps="1,4", classes="random,narrow,extremes"),
             dict(Ss="128", Ns="32,512", BKV="16,32", nExps="2,4", classes="random,narrow"),
             dict(Ss="8", Ns="64,512", BKV="8,32", nExps="1,2", classes="random,ties"),
             dict(Ss="256", Ns="1024", BKV="16", nExps="4", classes="random")],
    "runs": [dict(Ss="16,64", Ns="S,256", BKV="16", nExps="1,2", classes="random,narrow,extremes",
                  runs=3)],
    # PRE-1 (2026-10-08): geometries off R = C = 256, n_exp up to 16, producer gaps
    "rc": [dict(RC=rc, Ss=str(min(min(int(x) for x in rc.split("x")), 128)), Ns="S,512", BKV="16",
                nExps="1,2", classes="random,extremes")
           for rc in ("128x64", "128x128", "128x256", "256x64", "256x128", "512x64", "512x128",
                      "512x256", "1024x64", "1024x128", "1024x256")]
          + [dict(RC="128x64,1024x256", Ss="16", Ns="64", BKV="8", nExps="1", classes="narrow,ties")],
    "nexp": [dict(RC="256x256", Ss="128", Ns="512", BKV="16", nExps="8,16", classes="random,narrow"),
             dict(RC="256x256", Ss="256", Ns="256", BKV="16", nExps="16", classes="random"),
             dict(RC="512x128", Ss="128", Ns="256", BKV="16", nExps="8,16", classes="random,extremes"),
             dict(RC="128x64", Ss="64", Ns="128", BKV="16", nExps="8,16", classes="random")],
    "gaps": [dict(RC="256x256", Ss="64", Ns="256", BKV="16", nExps="1,2", classes="random,extremes",
                  gaps="inter,intra,both"),
             dict(RC="128x64", Ss="64", Ns="128", BKV="16", nExps="2", classes="random", gaps="inter,intra,both"),
             dict(RC="512x128", Ss="128", Ns="512", BKV="16", nExps="8", classes="random", gaps="inter,both"),
             dict(RC="256x256", Ss="16", Ns="64", BKV="8", nExps="1", classes="ties,rising", gaps="intra"),
             dict(RC="256x256", Ss="64", Ns="256", BKV="16", nExps="2", classes="random,narrow,extremes",
                  gaps="both", runs=3)],
}


def offer_edges(S: int, N: int, bkv: int, spec: str, seed: int) -> list:
    """Per-word earliest offer edges (relative to the start edge) for a gap pattern.
    inter: block b's first word not before 2 + b*G + jitter (G = 2*STREAM + 50, jitter < G/2);
    intra: inside each block a bubble of 1..3 edges after a word with probability 1/4;
    both: the two combined. Word w of a block is never offered before word w-1 (+1)."""
    rng = np.random.default_rng(seed + 31 * S + 5 * N + 3 * bkv)
    st = -(-S * bkv // 5)
    G = 2 * st + 50
    offs = []
    for b in range(N // bkv):
        t = 2 + b * G + int(rng.integers(0, G // 2)) if spec in ("inter", "both") else 0
        for w in range(st):
            if w and spec in ("intra", "both"):
                t += 1 + (int(rng.integers(1, 4)) if rng.random() < 0.25 else 0)
            elif w:
                t = 0
            offs.append(t)
    return offs


def _stamp_offers(case_dir: Path, offs: list, expected: int) -> None:
    (case_dir / "offer.mem").write_text("".join(f"{x:08x}\n" for x in offs))
    cj = json.loads((case_dir / "case.json").read_text())
    cj.update(OFFER=1, expected_cycles=expected, continuous_cycles=cj["compute_cycles"])
    (case_dir / "case.json").write_text(json.dumps(cj, indent=1) + "\n")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def scores_for(S: int, N: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 17 * S + 7 * N)
    return np.asarray(oref._stimuli(S, N, rng)[kind], dtype=np.int64).clip(-128, 127)


def gen(out: Path, ax: dict, R: int, seed: int, names: list, C: int | None = None) -> None:
    classes = ax["classes"].split(",")
    runs = int(ax.get("runs", 1))
    geoms = ([tuple(int(x) for x in rc.split("x")) for rc in ax["RC"].split(",")]
             if "RC" in ax else [(R, R if C is None else C)])
    gaps = ax["gaps"].split(",") if "gaps" in ax else [None]
    Ss = [int(x) for x in ax["Ss"].split(",")]
    combos = [(Rg, Cg, S, N, ci, kind, bkv, ne, gp)
              for (Rg, Cg) in geoms for S in Ss
              for N in [S if x == "S" else int(x) for x in ax["Ns"].split(",")]
              for ci, kind in enumerate(classes)
              for bkv in [int(x) for x in ax["BKV"].split(",")] if bkv <= N
              for ne in [int(x) for x in ax["nExps"].split(",")]
              for gp in gaps]
    for (Rg, Cg, S, N, ci, kind, bkv, ne, gp) in combos:
        tag = f"S{S}_N{N}_B{bkv}_nX{ne}" if N != S else f"S{S}_B{bkv}_nX{ne}"
        tag += ("" if (Rg, Cg) == (256, 256) else f"_{Rg}x{Cg}") + ("" if gp is None else f"_g{gp}")
        name = f"{kind}_{tag}" if runs <= 1 else f"runs{runs}_{kind}_{tag}"
        if name in names:
            continue
        sim = osim.NldpeSoftmaxOnline(S, R=Rg, C=Cg, n_exp=ne, Bkv=bkv, N=N)

        def one(case_dir: Path, sc: np.ndarray, k: str, r: int) -> int:
            sim.dump_case(case_dir, sc, cls=k)
            if gp is None:
                return json.loads((case_dir / "case.json").read_text())["compute_cycles"]
            offs = offer_edges(S, N, bkv, gp, seed + 1000 * r)
            exp = asim.softmax_availability_model(S, bkv, Rg, Cg, ne, N, pass_extent="full",
                                                  offers=offs)["total"]
            _stamp_offers(case_dir, offs, exp)
            return exp

        if runs <= 1:
            one(out / name, scores_for(S, N, kind, seed), kind, 0)
        else:
            kinds, exps = [], []
            for r in range(runs):
                k_r = classes[(ci + r) % len(classes)]
                exps.append(one(out / name / f"r{r}", scores_for(S, N, k_r, seed + 1000 * r), k_r, r))
                kinds.append(k_r)
            top = json.loads((out / name / "r0" / "case.json").read_text())
            top.update(RUNS=runs, cls=",".join(kinds), expected_cycles=exps)
            if gp is not None:
                top["OFFER"] = 1
            (out / name / "case.json").write_text(json.dumps(top, indent=1) + "\n")
        names.append(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "v3" / "work" / "softmax" / "softmax_online_stimuli"))  # v3 path (tools/copies.py)
    ap.add_argument("--preset", default=None, help="comma list of " + ",".join(PRESETS) + " (default: all)")
    ap.add_argument("--Ss", default=None)
    ap.add_argument("--Ns", default="S", help="key lengths; 'S' = square")
    ap.add_argument("--BKV", default="8,16,32")
    ap.add_argument("--nExps", default="1,2,4")
    ap.add_argument("--classes", default=ALL)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--R", type=int, default=256)
    ap.add_argument("--C", type=int, default=None, help="default: C = R")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true", help="append to an existing manifest")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.txt"
    names: list = manifest.read_text().split() if a.append and manifest.exists() else []
    if a.Ss is not None:
        axes = [dict(Ss=a.Ss, Ns=a.Ns, BKV=a.BKV, nExps=a.nExps, classes=a.classes, runs=a.runs)]
    else:
        keys = list(PRESETS) if a.preset in (None, "all") else a.preset.split(",")
        axes = [ax for k in keys for ax in PRESETS[k]]
    for ax in axes:
        gen(out, ax, a.R, a.seed, names, a.C)
    manifest.write_text("\n".join(names) + "\n")
    print(f"gen_softmax_online_cases: {len(names)} GATE-1-certified cases under {rel(out)} (manifest.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
