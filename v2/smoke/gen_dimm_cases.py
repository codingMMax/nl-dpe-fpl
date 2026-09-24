#!/usr/bin/env python3
"""gen_dimm_cases.py — generate Stage-4 DIMM cases for the RTL cross-check.

Sweeps the pool/farm shape matrix over the farm-crossbar count `n_E`
(the RTL `N_E` parameter; `n_A`/`n_B` are derived by the balance law and
emitted per case). Every case is written through `NldpeDimm.dump_case`, which
GATE-1-certifies `sim ≡ dimm_ref` (logA/logB/exp_u/exp_bytes/acc/C), the
issued pass counts, and the schedule cycle total BEFORE any file is written.

Axes:
  shapes  : 6 fixed (M, N, K, R, C) shapes, small -> large:
              (4,5,3)@8x8          small crossbar, multi-pass everywhere
              (8,10,6)@256x256     baseline; farm unpacked 6 vs ideal 2
              (16,16,4)@64x32      R != C (I = 32)
              (32,32,8)@128x64     R != C, larger
              (64,128,64)@256x256  rectangular SxV-like
              (128,64,64)@256x256  mirrored
  --nEs   : farm crossbar counts          (default 1,2,4,8,16)
  --classes identity,random,extremes      (default random,extremes)

A/B stimuli are seeded WITHOUT `n_E`, so the same (shape, class) has identical
operands across the sweep — the harness gates value invariance across `n_E`.

The output dir accumulates across runs (old cases stay inspectable); the
harness executes only the names listed in `manifest.txt`.

Usage:
  python3 v2/smoke/gen_dimm_cases.py                 # 60 default cases
  python3 v2/smoke/gen_dimm_cases.py --nEs 1,4,16    # subset sweep
  python3 v2/smoke/gen_dimm_cases.py --clean         # drop the corpus first
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
import dimm_sim as D  # noqa: E402

SHAPES = [
    #  M   N   K   R    C
    (2, 3, 1, 8, 8),        # single window everywhere: T_fill calibration
    (4, 5, 3, 8, 8),
    (8, 10, 6, 256, 256),
    (16, 16, 4, 64, 32),
    (32, 32, 8, 128, 64),
    (64, 128, 64, 256, 256),
    (128, 64, 64, 256, 256),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def make_operands(kind: str, M: int, N: int, K: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    if kind == "extremes":
        A[0, :] = -128
        if M > 1:
            A[1, :] = 127
        if K > 1:
            A[0, K - 1] = 0
        B[0, :] = 127
        if K > 1:
            B[K - 1, :] = -128
        if N > 1:
            B[:, N - 1] = 0
    return A, B


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "v2" / "smoke" / "dimm_stimuli"))
    ap.add_argument("--nEs", default="1,2,4,8,16")
    ap.add_argument("--classes", default="random,extremes")
    ap.add_argument("--only", default=None,
                    help="comma-separated shape tags, e.g. 4x5x3,64x128x64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="delete existing cases under --out first")
    args = ap.parse_args()

    out_root = Path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    only = args.only.split(",") if args.only else None
    names: list[str] = []
    for (M, N, K, R, C) in SHAPES:
        shape_tag = f"{M}x{N}x{K}"
        if only is not None and not any(tok in shape_tag for tok in only):
            continue
        for kind in args.classes.split(","):
            rng = np.random.default_rng(
                args.seed + 17 * M + N + 31 * K + R + C + len(kind))
            A, B = make_operands(kind, M, N, K, rng)
            for n_E in [int(t) for t in args.nEs.split(",")]:
                d = D.NldpeDimm(R=R, C=C, n_E=n_E)
                case_dir = out_root / f"{kind}_{M}x{N}x{K}_{R}x{C}_nE{n_E}"
                d.dump_case(case_dir, A, B)
                names.append(case_dir.name)
                print(f"wrote {rel(case_dir)}")

    (out_root / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"gen_dimm_cases: {len(names)} cases under {rel(out_root)} "
          f"(nEs={args.nEs}, classes={args.classes}, manifest.txt)")


if __name__ == "__main__":
    main()
