#!/usr/bin/env python3
"""gen_cases.py — the GATE-2 corpus of `gemm_top` on 64x64 crossbars (PLAN R4; ip/gemm/spec.md §7).

Every case goes through ip/gemm/cases.py: model run, GATE 1 against the oracle, then the testbench's files and the
model's events. The corpus covers the four FC shapes at 64x64 (V 8 / 32 / 4 / 8, H 2 / 4 / 8 / 8); V 1 / 2 / 3 / 5 /
16 and H 1 / 3 with K and N not multiples of 64 and both P29 extents; three or more rows back to back; late offers;
and the random / zeros / extremes / carry classes.

Usage: python3 v3/ip/gemm/gen_cases.py [--out DIR] [--only SUBSTR]   (default out: v3/work/corpus/bitplane/gemm_64x64)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1]
sys.path.insert(0, str(V3))
from ip.gemm.cases import randomCase, writeTbFiles  # noqa: E402

R = C = 64


def corpus(rng) -> list:
    cases = []

    def case(name, K, N, M=3, **kw):
        cases.append(randomCase(name, K, N, R, C, M, rng, **kw))

    # the four FC shapes (dse/space.py) at 64x64
    case("fc_512x128", 512, 128)                                  # V 8, H 2
    case("fc_2048x256", 2048, 256)                                # V 32, H 4
    case("fc_256x512", 256, 512)                                  # V 4, H 8
    case("fc_512x512", 512, 512)                                  # V 8, H 8
    # V 1 / 2 / 3 / 5 / 16, H 1 / 3; K and N not multiples of 64; the P29 extents
    case("v1_h1_used", 40, 24, M=4)                               # ROWS_USED 40, COLS_USED 24
    case("v1_h1_full", 40, 24, M=4, passExtent="full")            # the whole array: padding rows and columns
    case("v1_h3", 64, 129)                                        # K == R, the last lane holds 1 column
    case("v2_h3", 100, 150)
    case("v3_h1", 150, 50, M=5)                                   # 5 rows back to back
    case("v5_h3", 300, 170)                                       # an odd node passes through at levels 1 and 2
    case("v16_h1", 1000, 64)                                      # 4 levels
    case("v16_h3_full", 1000, 130, M=2, passExtent="full")
    # late offers: row 1 after its credit, row 2 before (it waits for MSB_SA_Ready), row 3 long after
    case("v3_h1_late", 150, 50, M=4, availableAt=[0, 30, 31, 200])
    case("v1_h1_late", 40, 24, M=3, availableAt=[5, 6, 100])
    # value classes
    case("v5_h3_extremes", 300, 170, cls="extremes")
    case("v16_h1_carry", 1000, 64, cls="carry")                   # tile bytes -1 / -128 / 127 / 1: full carry chains
    case("v3_h3_carry", 150, 150, cls="carry")
    case("v2_h1_zeros", 100, 50, cls="zeros")
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(V3 / "work" / "corpus" / "bitplane" / "gemm_64x64"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261023)
    names = []
    for c in corpus(rng):
        if a.only and a.only not in c.name:
            continue
        cj = writeTbFiles(c, out / c.name)
        names.append(c.name)
        ext = f"{cj['defines']['RU_TB']}x{cj['defines']['CU_TB']}"
        print(f"  {c.name:16s} K {c.K:5d} N {c.N:4d} V {cj['V']:2d} H {cj['H']} M {cj['M']} extent {ext} "
              f"events {cj['events']:3d} cycles {cj['cycles']}")
    (out / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"gen_cases: {len(names)} GATE-1-certified cases under {out} (manifest.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
