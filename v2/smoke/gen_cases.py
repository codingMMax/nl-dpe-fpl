#!/usr/bin/env python3
"""gen_cases.py — generate §9 stimulus cases for the v2 RTL cross-check.

Enumerates the spec's stimulus classes (`v2/spec/dpe_nldpe.md` §9) and writes
them via `NldpeDpe.dump_case`:

  identity  : W = eye(R, C)                      (I7 when R == C)
  random    : W random int8 (both signs); X uniform int8
  extremes  : as random, plus the signed extremes -128 / +127 and a zero row

The ACAM mode is workload configuration (P27): one mode per case, programmed
with the weights and held across all M passes. `--modes` sweeps it at the case
level; multi-pass runs (pipeline calibration) use one mode per run.

Every case directory (weights.mem, act.mem, expected_y.npz, expected_out.mem,
case.json) is consumed by the Stage 1.5 RTL harness. No NaN/Inf (A17).

Usage:
  python3 v2/smoke/gen_cases.py                  # both geoms, M in {1,2}, all modes
  python3 v2/smoke/gen_cases.py --ms 1,2,4,8     # full M sweep (slow)
  python3 v2/smoke/gen_cases.py --classes identity,random --ms 1,2,4,8 --modes 0
  python3 v2/smoke/gen_cases.py --geoms 256x256 --modes 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2" / "sim"))
sys.path.insert(0, str(REPO / "v2" / "oracle"))
import nldpe_sim as S  # noqa: E402


def parse_geoms(spec: str) -> list[tuple[int, int]]:
    geoms = []
    for tok in spec.split(","):
        r, c = tok.lower().split("x")
        geoms.append((int(r), int(c)))
    return geoms


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def make_weights(kind: str, R: int, C: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "identity":
        return np.eye(R, C, dtype=np.int8)
    return rng.integers(-128, 128, size=(R, C), dtype=np.int8)


def make_activations(kind: str, M: int, R: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.integers(-128, 128, size=(M, R), dtype=np.int8)
    if kind == "extremes":
        X[0] = -128
        if R >= 1:
            X[0, 0] = 127
        if R >= 2:
            X[0, 1] = 0
        if M >= 2:
            X[1] = 127
        if M >= 3:
            X[2] = 0
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "v2" / "smoke" / "stimuli"))
    ap.add_argument("--geoms", default="256x256,256x512")
    ap.add_argument("--ms", default="1,2")
    ap.add_argument("--modes", default="0,1,2,3")
    ap.add_argument("--classes", default="identity,random,extremes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    mode_tokens = [int(t) for t in args.modes.split(",")]
    n = 0
    for (R, C) in parse_geoms(args.geoms):
        for kind in args.classes.split(","):
            for M in [int(t) for t in args.ms.split(",")]:
                rng = np.random.default_rng(args.seed + 17 * R + C + 31 * M + len(kind))
                d = S.NldpeDpe(R=R, C=C)
                d.program_weights(make_weights(kind, R, C, rng))
                X = make_activations(kind, M, R, rng)
                for mode in mode_tokens:
                    case_dir = out_root / f"{kind}_{R}x{C}_M{M}_m{mode}"
                    d.dump_case(case_dir, X, mode)
                    n += 1
                    print(f"wrote {rel(case_dir)}")
    print(f"gen_cases: {n} cases under {rel(out_root)}")


if __name__ == "__main__":
    main()
