#!/usr/bin/env python3
"""gen_gemm_cases.py — generate Stage-2 GEMM array cases for the RTL cross-check.

Enumerates shapes over the V×H tiling space and writes each case through
`NldpeGemm.dump_case`, which GATE-1-certifies `sim ≡ gemm_ref` (S int32 +
output bytes) and the §5.3 cycle total BEFORE any file is written.

Axes:
  --geoms   R x C        crossbar sizes          (default 256x256)
  --ks      K list       reduction lengths       (default 128,400)
  --ns      N list       output widths           (default 256,512)
  --ms      M list       pass counts             (default 1,2)
  --classes identity,random,extremes
  --stage   1A | 1B | 1C | 1D | all   filter on (V, H):
              1A  V=1 H=1     1B  V>1 H=1
              1C  V=1 H>1     1D  V>1 H>1

The output dir accumulates across runs (old cases stay inspectable); the
harness executes only the names listed in `manifest.txt`.

Usage:
  python3 v2/smoke/gen_gemm_cases.py                       # 16 default cases
  python3 v2/smoke/gen_gemm_cases.py --stage 1A            # V=1,H=1 only
  python3 v2/smoke/gen_gemm_cases.py --ks 128,400,1024 --ns 120,512 --ms 1,2,4
  python3 v2/smoke/gen_gemm_cases.py --clean               # drop the corpus first
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
import gemm_sim as G  # noqa: E402


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def parse_geoms(spec: str) -> list[tuple[int, int]]:
    geoms = []
    for tok in spec.split(","):
        r, c = tok.lower().split("x")
        geoms.append((int(r), int(c)))
    return geoms


def stage_of(V: int, H: int) -> str:
    if V == 1 and H == 1:
        return "1A"
    if V > 1 and H == 1:
        return "1B"
    if V == 1 and H > 1:
        return "1C"
    return "1D"


def make_weights(kind: str, K: int, N: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "identity":
        return np.eye(K, N, dtype=np.int8)
    W = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    if kind == "extremes":
        W[0, :] = -128
        if K > 1:
            W[1, :] = 127
        if N > 1:
            W[:, N - 1] = 0
    return W


def make_activations(kind: str, M: int, K: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    if kind == "extremes":
        X[0] = -128
        if M > 1:
            X[1] = 127
        if K > 1:
            X[0, K - 1] = 0
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "v2" / "smoke" / "gemm_stimuli"))
    ap.add_argument("--geoms", default="256x256")
    ap.add_argument("--ks", default="128,400")
    ap.add_argument("--ns", default="256,512")
    ap.add_argument("--ms", default="1,2")
    ap.add_argument("--classes", default="identity,random")
    ap.add_argument("--stage", choices=("1A", "1B", "1C", "1D", "all"), default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="delete existing cases under --out first")
    args = ap.parse_args()

    out_root = Path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    for (R, C) in parse_geoms(args.geoms):
        for K in [int(t) for t in args.ks.split(",")]:
            for N in [int(t) for t in args.ns.split(",")]:
                V, H = G.derive_vh(K, N, R, C)
                if args.stage != "all" and stage_of(V, H) != args.stage:
                    continue
                for M in [int(t) for t in args.ms.split(",")]:
                    for kind in args.classes.split(","):
                        rng = np.random.default_rng(
                            args.seed + 17 * R + C + 31 * K + N + 7 * M + len(kind))
                        d = G.NldpeGemm(M, K, N, R, C)
                        d.program_weights(make_weights(kind, K, N, rng))
                        X = make_activations(kind, M, K, rng)
                        case_dir = out_root / f"{kind}_{R}x{C}_{K}x{N}_M{M}"
                        d.dump_case(case_dir, X)
                        names.append(case_dir.name)
                        print(f"wrote {rel(case_dir)}  (V={V} H={H} {stage_of(V, H)})")

    (out_root / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"gen_gemm_cases: {len(names)} cases under {rel(out_root)} "
          f"(stage={args.stage}, manifest.txt)")


if __name__ == "__main__":
    main()
