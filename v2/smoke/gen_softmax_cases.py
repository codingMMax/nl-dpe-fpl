#!/usr/bin/env python3
"""gen_softmax_cases.py — generate Stage-3 softmax cases for the RTL cross-check.

Every case is written through `NldpeSoftmax.dump_case`, which GATE-1-certifies
`sim ≡ softmax_ref` (all eight stages) and the packed pass counts BEFORE any
file is written.

Axes:
  --Ss      : score dimension S        (default 128,256)
  --nExps   : EXP crossbar count       (default 1,2,4)
  --nLogs   : LOG crossbar count       (default 1,2)
  --classes : random,extremes,uniform,ties,single-max
              (default all)

Scores are seeded WITHOUT `n_exp`/`n_log`, so the same (S, class) has identical
input across the count sweep — the harness gates value invariance.

The output dir accumulates across runs; the harness executes only the names
listed in `manifest.txt`.

Usage:
  python3 v2/smoke/gen_softmax_cases.py                    # 60 default cases
  python3 v2/smoke/gen_softmax_cases.py --nExps 1,4        # subset sweep
  python3 v2/smoke/gen_softmax_cases.py --clean            # drop corpus first
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2" / "sim"))
sys.path.insert(0, str(REPO / "v2" / "oracle"))
import softmax_sim as S  # noqa: E402


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def make_scores(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "random":
        return rng.integers(-128, 128, size=(size, size), dtype=np.int8)
    if kind == "extremes":
        return np.array([[-128, 127] * (size // 2),
                         [127, -128] * (size // 2)] * (size // 2),
                        dtype=np.int8)
    if kind == "uniform":
        return np.zeros((size, size), dtype=np.int8)
    if kind == "ties":
        return np.full((size, size), 5, dtype=np.int8)
    if kind == "single-max":
        scores = np.zeros((size, size), dtype=np.int8)
        scores[:, size // 2] = 127
        return scores
    raise ValueError(f"unknown class {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(REPO / "v2" / "smoke" / "softmax_stimuli"))
    ap.add_argument("--Ss", default="128,256")
    ap.add_argument("--RCs", default="256",
                    help="crossbar geometries R=C (default 256)")
    ap.add_argument("--nExps", default="1,2,4")
    ap.add_argument("--nLogs", default="1,2")
    ap.add_argument("--classes", default="random,extremes,uniform,ties,"
                                        "single-max")
    ap.add_argument("--only", default=None,
                    help="comma-separated class tags, e.g. random,ties")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="delete existing cases under --out first")
    ap.add_argument("--append", action="store_true",
                    help="append to an existing manifest instead of rewriting")
    args = ap.parse_args()

    out_root = Path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    only = args.only.split(",") if args.only else None
    manifest_path = out_root / "manifest.txt"
    names: list[str] = []
    if args.append and manifest_path.exists():
        names = [n for n in manifest_path.read_text().split() if n]
    for size in [int(s) for s in args.Ss.split(",")]:
        for rc in [int(t) for t in args.RCs.split(",")]:
            for kind in args.classes.split(","):
                if only is not None and kind not in only:
                    continue
                rng = np.random.default_rng(
                    args.seed + 17 * size + 31 * rc + len(kind))
                scores = make_scores(kind, size, rng)
                geom = "" if rc == 256 else f"_R{rc}C{rc}"
                for n_exp in [int(t) for t in args.nExps.split(",")]:
                    for n_log in [int(t) for t in args.nLogs.split(",")]:
                        unit = S.NldpeSoftmax(S=size, R=rc, C=rc,
                                              n_exp=n_exp, n_log=n_log)
                        case_dir = (out_root /
                                    f"{kind}_S{size}{geom}_nX{n_exp}_nL{n_log}")
                        unit.dump_case(case_dir, scores)
                        names.append(case_dir.name)
                        print(f"wrote {rel(case_dir)}")

    manifest_path.write_text("\n".join(names) + "\n")
    print(f"gen_softmax_cases: {len(names)} cases under {rel(out_root)} "
          f"(Ss={args.Ss}, nExps={args.nExps}, nLogs={args.nLogs}, "
          f"classes={args.classes}, manifest.txt)")


if __name__ == "__main__":
    main()
