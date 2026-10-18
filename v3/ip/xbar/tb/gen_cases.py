#!/usr/bin/env python3
"""gen_cases.py — generate §9 stimulus cases for the v2 RTL cross-check.

Enumerates the spec's stimulus classes (`v2/sim/spec/dpe_nldpe.md` §9) and writes
them via `NldpeDpe.dump_case`. Every dumped case is certified by GATE 1 first:
`dump_case` refuses to write a case whose sim values disagree with the NumPy
oracle on int32 `y`, output bytes, or §5.3 cycles.

Stimulus classes:
  identity  : W = eye(R, C)                      (I7 when R == C)
  random    : W random int8 (both signs); X uniform int8
  extremes  : as random, plus the signed extremes -128 / +127 and a zero row

The ACAM mode is workload configuration (P27): one mode per case, programmed
with the weights and held across all M passes. `--modes` sweeps it at the case
level; multi-pass runs (pipeline calibration) use one mode per run.

Every case directory (weights.mem, act.mem, expected_y.npz, expected_out.mem,
case.json) is consumed by the Stage 1.5 RTL harness. No NaN/Inf (A17).

Usage:
  python3 v2/sim/smoke/gen_cases.py                  # both geoms, M in {1,2}, all modes
  python3 v2/sim/smoke/gen_cases.py --ms 1,2,4,8     # full M sweep (slow)
  python3 v2/sim/smoke/gen_cases.py --classes identity,random --ms 1,2,4,8 --modes 0
  python3 v2/sim/smoke/gen_cases.py --geoms 256x256 --modes 0
  python3 v2/sim/smoke/gen_cases.py --clean          # drop the accumulated corpus first
  python3 v2/sim/smoke/gen_cases.py --p29            # P29 pass-extent corpus (rows / columns in use)

P29 corpus (spec v2.0.3): each geometry declares the rows streamed in and the columns drained per
pass. Weights are random over the WHOLE array (unused rows hold non-zero weights, so a leak of an
unloaded row would show), activations are random over all R rows (the dump keeps only the rows in
use). Per geometry and class: M = 1 and M = 4 in mode 0 (T_steady calibration) and M = 2 in every
mode. Corners: 1 row x 1 column, rows / columns not multiples of 5, the compute-bound 8x8 corner,
a tall 1024x64 identity-width extent and a wide 64x256 one.

The output dir accumulates across runs (old cases stay inspectable via
`v2/sim/smoke/test_dpe_primitive.py`); `manifest.txt` lists the cases of this
generation, and the RTL harness executes only those.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]  # v3 path (tools/copies.py)
sys.path.insert(0, str(REPO / "v3" / "oracle" / "v2_kernels"))  # v3 path (tools/copies.py)
sys.path.insert(0, str(REPO / "v3" / "oracle"))  # v3 path (tools/copies.py)
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


P29_GEOMS = [  # (R, C, rows in use, columns in use)
    (8, 8, 5, 3), (40, 40, 1, 1), (40, 40, 17, 23), (128, 64, 64, 64), (256, 256, 128, 64),
    (256, 256, 96, 40), (256, 512, 256, 129), (1024, 64, 64, 64), (64, 256, 64, 64),
]


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
    ap.add_argument("--out", default=str(REPO / "v3" / "work" / "xbar" / "dpe_stimuli"))  # v3 path (tools/copies.py)
    ap.add_argument("--geoms", default="256x256,256x512")
    ap.add_argument("--ms", default="1,2, 4, 8")
    ap.add_argument("--modes", default="0,1,2,3")
    ap.add_argument("--classes", default="identity,random,extremes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="delete existing cases under --out first")
    ap.add_argument("--p29", action="store_true",
                    help="P29 pass-extent corpus instead of the --geoms/--ms/--modes sweep")
    args = ap.parse_args()

    out_root = Path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    mode_tokens = [int(t) for t in args.modes.split(",")]
    names: list[str] = []
    if args.p29:
        for (R, C, rows, cols) in P29_GEOMS:
            for kind in ("identity", "random", "extremes"):
                for M, modes in ((1, (0,)), (4, (0,)), (2, (0, 1, 2, 3))):
                    rng = np.random.default_rng(args.seed + 17 * R + C + 31 * M + len(kind) + 7 * rows + cols)
                    d = S.NldpeDpe(R=R, C=C, rows_used=rows, cols_used=cols)
                    d.program_weights(make_weights(kind, R, C, rng))
                    X = make_activations(kind, M, R, rng)
                    for mode in modes:
                        case_dir = out_root / f"p29_{kind}_{R}x{C}_r{rows}c{cols}_M{M}_m{mode}"
                        d.dump_case(case_dir, X, mode)
                        names.append(case_dir.name)
                        print(f"wrote {rel(case_dir)}")
        (out_root / "manifest.txt").write_text("\n".join(names) + "\n")
        print(f"gen_cases: {len(names)} P29 cases under {rel(out_root)} (manifest.txt)")
        return
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
                    names.append(case_dir.name)
                    print(f"wrote {rel(case_dir)}")
    (out_root / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"gen_cases: {len(names)} cases under {rel(out_root)} (manifest.txt)")


if __name__ == "__main__":
    main()
