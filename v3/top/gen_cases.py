#!/usr/bin/env python3
"""gen_cases.py — the gate-H cases of `att_head` on 64x64 crossbars (PLAN R4; top/spec.md §8).

Every case goes through top/cases.py: model run, GATE 1 against the oracle, then the testbench's files and the model's
events.
  --preset bringup   5 small heads (the smallest head, P_PROJ 2 and 4, V = 2, two runs) for RTL bring-up
  --preset corpus    the gate-H corpus (6 heads, see corpus())

Usage: python3 v3/top/gen_cases.py [--preset bringup|corpus] [--out DIR] [--only SUBSTR]
(default out: v3/work/f4/bringup, or v3/work/corpus/head_64x64 for the corpus)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
sys.path.insert(0, str(V3))
from top.cases import randomCase, writeTbFiles  # noqa: E402

R = C = 64


def shape(n, dm, d, pp=1, **kw):
    return dict(NSEQ=n, DM=dm, D=d, R=R, C=C, P_PROJ=pp, **kw)


def bringup(rng) -> list:
    return [randomCase("bu_n32", shape(32, 64, 16), rng),
            randomCase("bu_p2", shape(64, 32, 32, pp=2), rng),
            randomCase("bu_v2", shape(32, 128, 32), rng),
            randomCase("bu_runs", shape(128, 64, 16), rng),
            randomCase("bu_p4", shape(32, 32, 16, pp=4), rng, cls="small")]


def corpus(rng) -> list:
    """Gate H (architecture.md §7): what the bring-up does not cover. X later than x_rdy and a late start (att_ctrl
    paths the bring-up never takes), one QK channel (N_E = d), V = 3, and d = 64 (BERT-Tiny's shape, 2 runs)."""
    return [randomCase("H_late_x", shape(64, 64, 16, pp=2), rng, xAvailableAt=[60 * t for t in range(64)]),
            randomCase("H_slow_x", shape(32, 64, 32), rng, xPeriod=90),
            randomCase("H_late_start", shape(32, 64, 16), rng, tStart=37),
            randomCase("H_n64_d16_ne1", shape(64, 64, 16, N_E=16), rng),            # N_E = d: one QK channel
            randomCase("H_v3_p2", shape(32, 192, 16, pp=2), rng, cls="small"),      # V = 3
            randomCase("H_n128_d64", shape(128, 128, 64), rng)]                     # BERT-Tiny's shape, 2 runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="bringup", choices=("bringup", "corpus"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else (V3 / "work" / "corpus" / "head_64x64" if a.preset == "corpus"
                                     else V3 / "work" / "f4" / "bringup")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    names = []
    for c in (corpus(rng) if a.preset == "corpus" else bringup(rng)):
        if a.only and a.only not in c.name:
            continue
        cj = writeTbFiles(c, out / c.name)
        names.append(c.name)
        print(f"  {c.name:14s} crossbars {cj['n_xbar']:4d} events {cj['events']:5d} cycles {cj['cycles']} "
              f"done {cj['done']}")
    (out / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"gen_cases: {len(names)} GATE-1-certified heads under {out} (manifest.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
