#!/usr/bin/env python3
"""gen_cases.py — the GATE-2 corpus of `dimm_tile_top` on 64x64 crossbars (PLAN R4; ip/dimm/spec.md §8).

Every case goes through ip/dimm/cases.py: model run, GATE 1 against the oracle, then the testbench's files and the
model's events. The corpus covers both instances of the head (spec §9): QK (int8 rows, row frame, TRANS_B = 1,
NARROW = 5) and PV (log windows, window frame, released slots, TRANS_B = 0, full width), with 1 / 2 / 4 / 8 slices,
1 / 2 / 4 channels, d 16 / 32 / 64, several runs, blocks and jobs, output credit starvation, late jobs and late
operands, and the random / zeros / extremes classes.

Usage: python3 v3/ip/dimm/gen_cases.py [--out DIR] [--only SUBSTR]   (default out: v3/work/corpus/dimm_64x64)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1]
sys.path.insert(0, str(V3))
from ip.dimm.cases import convPeriod, pvParams, qkParams, randomCase, writeTbFiles  # noqa: E402

R = C = 64


def qkJobs(runs: int, blocks: int) -> list:
    return [(r, b, 0) for r in range(runs) for b in range(blocks)]


def pvJobs(runs: int, blocks: int) -> list:
    return [(j % 2, j % blocks, 1) for j in range(runs * blocks)]


def corpus(rng) -> list:
    cases = []

    def qk(name, S, d, nE, pp=1, runs=1, blocks=1, cls="random", cCap=2, **kw):
        p = qkParams(S, d, 16, R, C, nE, pProj=pp, runs=runs, blocks=blocks, cCap=cCap)
        cases.append(randomCase(name, p, runs, blocks, qkJobs(runs, blocks), rng, cls=cls,
                                aPeriod=kw.pop("aPeriod", convPeriod()), bPeriod=kw.pop("bPeriod", convPeriod()),
                                **kw))

    def pv(name, S, d, nEp, pp=1, runs=1, blocks=2, cls="random", **kw):
        p = pvParams(S, d, 16, R, C, nEp, nExp=2, pProj=pp, blocks=blocks)
        cases.append(randomCase(name, p, runs * blocks, blocks, pvJobs(runs, blocks), rng, cls=cls,
                                aPeriod=kw.pop("aPeriod", convPeriod()),
                                bPeriod=kw.pop("bPeriod", convPeriod()), **kw))

    # QK: S x 16 x d, int8 rows (row frame), K rows transposed, NARROW 5
    qk("qk_d16_e16_1s", 32, 16, 16)                                   # 1 slice
    qk("qk_d16_e32_2s_2ch_2x2", 32, 16, 32, pp=2, runs=2, blocks=2)   # 2 slices, 2 channels, 4 jobs
    qk("qk_d16_e64_4s", 64, 16, 64, blocks=2)                         # 4 slices of 4 columns (NG 4)
    qk("qk_d32_e64_4s_2ch_2x2", 64, 32, 64, pp=2, runs=2, blocks=2)
    qk("qk_d64_e64_4s", 64, 64, 64)                                   # NG 1, 16 columns per slice
    qk("qk_d16_e32_4ch", 64, 16, 32, pp=4, blocks=2)                  # 4 channels (NB 4)
    qk("qk_d32_e32_starve", 64, 32, 32, runs=2, blocks=2, cCap=1, cCreditLatency=300)
    qk("qk_d16_e32_late", 32, 16, 32, runs=2, blocks=2, jAvailableAt=[0, 2500, 2600, 5000], bAvailableAt=[0, 1800])
    qk("qk_d16_e32_zeros", 32, 16, 32, runs=1, blocks=2, cls="zeros")
    qk("qk_d32_e64_extremes", 64, 32, 64, runs=1, blocks=2, cls="extremes")
    # PV: S x d x 16, log P windows (window frame, 2 EXP lanes), released slots, V rows, full width
    pv("pv_d16_e32_2s", 32, 16, 32)                                   # 2 slices, NB 4 = rows per window
    pv("pv_d32_e64_4s_2ch", 64, 32, 64, pp=2)                         # 4 slices, B on 2 channels
    pv("pv_d64_e128_8s", 32, 64, 128)                                 # 8 slices of 2 keys, NB 1
    pv("pv_d16_e32_2runs", 32, 16, 32, runs=2)                        # 4 jobs, slots released and refilled
    pv("pv_d32_e64_extremes", 64, 32, 64, cls="extremes")
    pv("pv_d16_e32_zeros", 32, 16, 32, cls="zeros")
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(V3 / "work" / "corpus" / "dimm_64x64"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    names = []
    for c in corpus(rng):
        if a.only and a.only not in c.name:
            continue
        cj = writeTbFiles(c, out / c.name)
        names.append(c.name)
        print(f"  {c.name:28s} A {cj['NA']:4d} B {cj['NBW']:4d} jobs {cj['NJ']:2d} mem {cj['NMEM']:5d} "
              f"events {cj['events']:4d} cycles {cj['cycles']}")
    (out / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"gen_cases: {len(names)} GATE-1-certified cases under {out} (manifest.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
