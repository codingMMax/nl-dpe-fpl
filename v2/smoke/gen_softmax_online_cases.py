#!/usr/bin/env python3
"""gen_softmax_online_cases.py — generate GATE-1-certified online softmax cases.

Every case is written through `NldpeSoftmaxOnline.dump_case`, which GATE-1-
certifies `sim ≡ softmax_online_ref.softmax_online_model` (all probes) and the
packed pass counts BEFORE any file is written.

Axes: --Ss (64,128,256), --BKV (block size, must divide S), --nExps, --classes.

Usage:
  python3 v2/smoke/gen_softmax_online_cases.py
  python3 v2/smoke/gen_softmax_online_cases.py --Ss 64 --BKV 16 --nExps 1
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2" / "sim" / "simulator" / "kernels"))
sys.path.insert(0, str(REPO / "v2" / "oracle"))
import softmax_online_sim as S  # noqa: E402


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
                    default=str(REPO / "v2" / "smoke" / "softmax_online_stimuli"))
    ap.add_argument("--Ss", default="64,128")
    ap.add_argument("--BKV", default="16")
    ap.add_argument("--R", type=int, default=256)
    ap.add_argument("--nExps", default="1")
    ap.add_argument("--nLogs", type=int, default=1)
    ap.add_argument("--classes", default="random,extremes,uniform,ties")
    ap.add_argument("--only", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--append", action="store_true")
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
        for kind in args.classes.split(","):
            if only is not None and kind not in only:
                continue
            rng = np.random.default_rng(
                args.seed + 17 * size + 31 * args.R + len(kind))
            scores = make_scores(kind, size, rng)
            for bkv in [int(t) for t in args.BKV.split(",")]:
                if size % bkv:
                    continue
                for n_exp in [int(t) for t in args.nExps.split(",")]:
                    unit = S.NldpeSoftmaxOnline(
                        S=size, R=args.R, C=args.R, Bkv=bkv,
                        n_exp=n_exp, n_log=args.nLogs)
                    case_dir = (out_root /
                                f"{kind}_S{size}_B{bkv}_nX{n_exp}")
                    unit.dump_case(case_dir, scores)
                    names.append(case_dir.name)
                    print(f"wrote {rel(case_dir)}")

    manifest_path.write_text("\n".join(names) + "\n")
    print(f"gen_softmax_online_cases: {len(names)} cases under {rel(out_root)} "
          f"(Ss={args.Ss}, BKV={args.BKV}, nExps={args.nExps})")


if __name__ == "__main__":
    main()
