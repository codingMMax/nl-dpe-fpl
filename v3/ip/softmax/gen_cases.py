#!/usr/bin/env python3
"""gen_cases.py — the GATE-2 cases of `softmax_online_top` on 64x64 crossbars (PLAN R4; ip/softmax/spec.md §8).

Every case goes through ip/softmax/cases.py: model run, GATE 1 against the oracle, then the testbench's files and the
model's events.
  --preset bringup   5 small cases (one run of each mechanism) for RTL bring-up
  --preset corpus    the GATE-2 corpus: S 8 / 16 / 32 / 64 (S < I exercises the carry extent S x S), N up to 8 S,
                     BKV 16, N_EXP 2 (and 1, 4), S_CH 1 / 2 / 4, 1-3 runs; credit holds on log P, alpha, log l;
                     late tiles and a slow producer; the smallest head (n 32); classes random / zeros / extremes /
                     small / rising
  --preset rc        the one R != C case (128 x 64 crossbars; an exception to the 64x64 rule, experiment.md §7)

Usage: python3 v3/ip/softmax/gen_cases.py [--preset bringup|corpus|rc] [--out DIR] [--only SUBSTR]
(default out: v3/work/corpus/softmax_64x64, v3/work/f3/softmax/bringup, or v3/work/corpus/softmax_rc)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1]
sys.path.insert(0, str(V3))
from ip.softmax.cases import randomCase, writeTbFiles  # noqa: E402

R = C = 64


def bringup(rng) -> list:
    def p(S, N, sch=1, ne=2):
        return dict(S=S, BKV=16, N=N, R=R, C=C, N_EXP=ne, S_CH=sch)
    return [randomCase("bu_tiny", p(8, 32), 1, rng),
            randomCase("bu_2ch_2runs", p(32, 64, sch=2), 2, rng),
            randomCase("bu_holds", p(16, 64), 2, rng, lpCreditLatency=200, laCreditLatency=150, llCreditLatency=400),
            randomCase("bu_slow", p(64, 128), 1, rng, sPeriod=30),
            randomCase("bu_slot", p(8, 128), 1, rng)]


def corpus(rng) -> list:
    cases = []

    def add(name, S, N, runs=1, sch=1, ne=2, cls="random", **kw):
        p = dict(S=S, BKV=16, N=N, R=R, C=C, N_EXP=ne, S_CH=sch)
        cases.append(randomCase(name, p, runs, rng, cls=cls, **kw))
    add("sm_n32", 32, 32, runs=2)                                  # the smallest head: n 32 (S = n < I)
    add("sm_s8_n64", 8, 64)                                        # carry extent 8 x 8, slot-bound stream
    add("sm_s16_n128_2ch", 16, 128, sch=2)
    add("sm_s32_n256", 32, 256, runs=2)                            # 16 blocks, two runs
    add("sm_s64_n64", 64, 64, runs=3, sch=2)                       # S = I, three runs back to back
    add("sm_s64_n256_4ch", 64, 256, sch=4)
    add("sm_s64_n128_ne1", 64, 128, ne=1)
    add("sm_s64_n128_ne4", 64, 128, ne=4, sch=2)
    add("sm_s32_n128_lphold", 32, 128, runs=2, lpCreditLatency=400)       # the PV DIMM holds its log P slots
    add("sm_s32_n128_lahold", 32, 128, runs=2, laCreditLatency=600)       # the accumulator holds its alpha ring
    add("sm_s16_n32_llhold", 16, 32, runs=3, llCreditLatency=1200)        # the accumulator holds its log l table
    add("sm_s32_n128_late", 32, 128, sch=2, sAvailableAt=[700 * j for j in range(8)])
    add("sm_s64_n128_slow", 64, 128, sPeriod=30)                   # the reader waits for words
    add("sm_s32_n64_runsgap", 32, 64, runs=2, goAvailableAt=[0, 6000])    # the next start comes late
    add("sm_s32_n128_zeros", 32, 128, cls="zeros")
    add("sm_s32_n128_extremes", 32, 128, runs=2, cls="extremes")
    add("sm_s16_n128_small", 16, 128, cls="small")
    add("sm_s32_n256_rising", 32, 256, cls="rising")
    return cases


def rc(rng) -> list:
    """The one R != C case (experiment.md §7): 128 x 64 crossbars, I = 64, so every EXP pass uses 64 of the 128 rows
    (the extent I x I shows only at R != C); S = I, two channels, two runs."""
    return [randomCase("sm_rc_128x64", dict(S=64, BKV=16, N=128, R=128, C=64, N_EXP=2, S_CH=2), 2, rng)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="corpus", choices=("bringup", "corpus", "rc"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else {"corpus": V3 / "work" / "corpus" / "softmax_64x64",
                                     "bringup": V3 / "work" / "f3" / "softmax" / "bringup",
                                     "rc": V3 / "work" / "corpus" / "softmax_rc"}[a.preset]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    names = []
    for c in {"corpus": corpus, "bringup": bringup, "rc": rc}[a.preset](rng):
        if a.only and a.only not in c.name:
            continue
        cj = writeTbFiles(c, out / c.name)
        names.append(c.name)
        print(f"  {c.name:24s} tiles {cj['NBLK']:3d} runs {cj['NRUN']} la {cj['NLA']:5d} events {cj['events']:5d} "
              f"cycles {cj['cycles']}")
    (out / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"gen_cases: {len(names)} GATE-1-certified cases under {out} (manifest.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
