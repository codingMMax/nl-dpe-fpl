#!/usr/bin/env python3
"""acc_gen_cases.py — the GATE-2 cases of `softmax_online_acc` on 64x64 crossbars (PLAN R4; ip/softmax/acc_spec.md §8).

Every case goes through ip/softmax/acc_cases.py: model run, GATE 1 against the oracle, then the testbench's files and
the model's events.
  --preset bringup   5 small cases (one window, two blocks, a partial last wave, late inputs, D = I) for RTL bring-up
  --preset corpus    the GATE-2 corpus: the head's instances at 64x64 (n 32 .. 128, d 16 / 32 / 64, N_OC by the floor
                     rule, P_CH = 2d / 16), partial last waves (N_OC 3, 5, 6, 7), W = 1 (acc_w1), SO != LN, 1-3 runs;
                     the inputs
                     available early (every credit binds), late alpha / log l, a slow o_part producer; classes random /
                     zeros / allneg / extremes / small / alpha0

Usage: python3 v3/ip/softmax/acc_gen_cases.py [--preset bringup|corpus] [--out DIR] [--only SUBSTR]
(default out: v3/work/corpus/acc_64x64, or v3/work/f3/acc/bringup for the bring-up preset)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1]
sys.path.insert(0, str(V3))
from ip.softmax.acc_cases import randomCase, writeTbFiles  # noqa: E402

R = C = 64


def params(S, D, nOc, pch, nblk, so=None, ln=None):
    lg = (nblk * 16).bit_length() - 1              # log2 n for a head with n = NBLK * Bkv keys
    return dict(S=S, D=D, R=R, C=C, N_OC=nOc, NBLK=nblk, SO=lg if so is None else so, LN=lg if ln is None else ln,
                P_CH=pch)


def bringup(rng) -> list:
    return [randomCase("bu_tiny", params(8, 8, 1, 1, 1), 1, rng),
            randomCase("bu_2blk", params(16, 16, 2, 2, 2), 2, rng),
            randomCase("bu_partial", params(32, 16, 3, 4, 3), 2, rng),
            randomCase("bu_late", params(32, 32, 4, 4, 3), 2, rng, laAvailableAt=[800 * u for u in range(4)],
                       llAvailableAt=[0, 5000]),
            randomCase("bu_d64", params(64, 64, 8, 8, 2), 2, rng)]


def corpus(rng) -> list:
    cases = []

    def add(name, S, D, nOc, pch, nblk, runs=1, cls="random", so=None, ln=None, **kw):
        cases.append(randomCase(name, params(S, D, nOc, pch, nblk, so, ln), runs, rng, cls=cls, **kw))
    # the head's instances at 64x64 (N_OC by the floor rule, P_CH = N_EP / Bkv = 2d / 16)
    add("acc_n32_d16", 32, 16, 2, 2, 2, runs=1)                          # the smallest head
    add("acc_n32_d32", 32, 32, 4, 4, 2, runs=1)                          # JetFormer-150P
    add("acc_n128_d16", 64, 16, 2, 2, 8, runs=2)                         # JetFormer-JetClass
    add("acc_n64_d64", 64, 64, 8, 8, 4, runs=1)                          # D = I: one row per window
    add("acc_n128_d64", 64, 64, 8, 8, 8, runs=2)                         # BERT-Tiny's geometry
    # partial last waves, other group sizes
    add("acc_noc3", 32, 16, 3, 4, 3, runs=2)
    add("acc_noc5", 64, 32, 5, 8, 2, runs=2)
    add("acc_noc6", 64, 64, 6, 8, 2, runs=2)
    add("acc_noc7", 64, 64, 7, 8, 3, runs=1)
    add("acc_noc1_b1", 16, 16, 1, 1, 1, runs=3)                          # NBLK = 1: the epilogue right after LOG
    # epilogue scales, 2^(SO - LN) both ways
    add("acc_so_gt_ln", 32, 16, 2, 2, 3, runs=1, cls="extremes", so=6, ln=3)
    add("acc_so_lt_ln", 32, 16, 2, 2, 3, runs=1, cls="extremes", so=2, ln=6)
    # arrivals: late alpha / log l, a slow o_part producer, the credits binding (inputs available at 0, the default)
    add("acc_late_la", 32, 32, 4, 4, 4, runs=2, laAvailableAt=[900 * u for u in range(6)])
    add("acc_late_ll", 32, 16, 2, 2, 2, runs=3, llAvailableAt=[0, 3000, 6500])
    add("acc_slow_p", 64, 32, 4, 8, 3, runs=1, pPeriod=60, pAvailableAt=[400 * j for j in range(3)])
    # value classes
    for cls in ("zeros", "allneg", "small", "alpha0"):
        add(f"acc_{cls}", 32, 16, 3, 4, 3, runs=1, cls=cls)
    # W = 1 (P_OC = N_OC = 4): so is a register (spec §10 item 11, 2026-10-23); last, so the 19 cases above keep their
    # random draws
    add("acc_w1", 16, 16, 4, 4, 3, runs=2)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="corpus", choices=("bringup", "corpus"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else (V3 / "work" / "corpus" / "acc_64x64" if a.preset == "corpus"
                                     else V3 / "work" / "f3" / "acc" / "bringup")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    names = []
    for c in (corpus(rng) if a.preset == "corpus" else bringup(rng)):
        if a.only and a.only not in c.name:
            continue
        cj = writeTbFiles(c, out / c.name)
        names.append(c.name)
        print(f"  {c.name:18s} tiles {cj['params']['NBLK'] * cj['plusargs']['NRUN']:3d} runs {cj['plusargs']['NRUN']} "
              f"events {cj['events']:5d} cycles {cj['cycles']}")
    (out / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"acc_gen_cases: {len(names)} GATE-1-certified cases under {out} (manifest.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
