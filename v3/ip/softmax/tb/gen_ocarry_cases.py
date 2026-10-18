#!/usr/bin/env python3
"""gen_ocarry_cases.py — GATE-1-certified cases for the o-carry station (`att_ocarry_top`).

Every case is written by `ocarry_sim.dump_case` (GATE 1: values through the certified EXP / LOG passes ==
`attention_ref` o-carry + epilogue, per run) with the model's event cycles for the strict runner. Input pulses come
from `ocarry_sim.feasible_schedule` (the head's buffer-reuse contract).

o_part arrives as the window-major PV tile's output stream (NGP groups = FIFOs, waves WAVE_PACE cycles apart).
Corpus (default = all presets except `anchor`):
  shape   S=16 D=8 @ 64x64 n_oc 1 (P_oc 2, NGP 2), S=16 D=16 n_oc 3 (partial last wave, NGP 4), S=32 D=8 n_oc 4
          (W 1), S=8 D=8 @ 32x32 NBLK 1 (block 0 + epilogue only), S=16 D=16 @ 64x32 n_oc 8 (R != C, NGP 8),
          S=16 D=64 @ 64x64 (D = I: one query row per window, NGP 8 > n_oc 4)
  sched   S=16 D=16 n_oc 2 NBLK 4: log_a early / late / at the stream start, a tight block pace (op_free-bound),
          gapless waves (WAVE_PACE = LW), NGP 2 and 4
  runs    two back-to-back runs (NRUN 2), SO > LN and SO < LN epilogue scaling; the next run's o_part arriving
          before the previous run ends (the station's run restart binds)
  reset   reset in the middle of a run (block 1 of 3 / the epilogue), then the whole schedule again
  feed    the O buffer = e_mem (rework M2): a long O feed (o_fed late) binds the next run's EXP hold — three runs,
          NBLK 3 (the first rescale waits) and NBLK 1 (the epilogue is the first EXP job), with an early next run
  anchor  BERT-Tiny @ 256x256: S=128 D=64 n_oc 7 NBLK 8 (P_oc 32, W 5, NGP 8)
Classes: random, zeros, allneg, extremes, alpha0 (log_a = 0).

Usage: python3 v2/sim/smoke/gen_ocarry_cases.py [--preset shape,sched,...] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]  # v3 path (tools/copies.py)
sys.path.insert(0, str(REPO / "v3" / "oracle" / "v2_kernels"))  # v3 path (tools/copies.py)
import ocarry_sim as oc  # noqa: E402

CLASSES = ("random", "zeros", "allneg", "extremes", "alpha0")


def presets() -> dict:
    P = {}
    P["shape"] = [dict(S=16, D=8, R=64, C=64, n_oc=1, NBLK=3, ngp=2), dict(S=16, D=16, R=64, C=64, n_oc=3, NBLK=2),
                  dict(S=32, D=8, R=64, C=64, n_oc=4, NBLK=3), dict(S=8, D=8, R=32, C=32, n_oc=2, NBLK=1),
                  dict(S=16, D=16, R=64, C=32, n_oc=8, NBLK=2), dict(S=16, D=64, R=64, C=64, n_oc=4, NBLK=2, ngp=8)]
    P["sched"] = [dict(S=16, D=16, R=64, C=64, n_oc=2, NBLK=4, lead=ld, pace=pc, ngp=ng, wpace=wp, classes=("random",))
                  for ld, pc, ng, wp in ((300, 150, 2, None), (-120, 150, 4, None), (0, 150, 2, 13), (0, 1, 2, None),
                                         (0, 1, 4, 13))]
    P["runs"] = [dict(S=16, D=8, R=64, C=64, n_oc=2, NBLK=3, NRUN=2, classes=("random", "allneg")),
                 dict(S=16, D=8, R=64, C=64, n_oc=2, NBLK=3, NRUN=2, early=300, classes=("random",)),
                 dict(S=16, D=16, R=64, C=64, n_oc=3, NBLK=2, NRUN=2, early=500, classes=("random",)),
                 dict(S=16, D=8, R=64, C=64, n_oc=1, NBLK=2, SO=5, LN=3, classes=("random", "extremes")),
                 dict(S=16, D=8, R=64, C=64, n_oc=1, NBLK=2, SO=3, LN=5, classes=("random", "extremes"))]
    P["reset"] = [dict(S=16, D=16, R=64, C=64, n_oc=2, NBLK=3, NRUN=2, midreset=f, classes=("random",))
                  for f in (0.3, 0.45)]
    P["feed"] = [dict(S=16, D=16, R=64, C=64, n_oc=2, NBLK=3, NRUN=3, feed=900, classes=("random", "extremes")),
                 dict(S=16, D=8, R=64, C=64, n_oc=2, NBLK=1, NRUN=3, feed=700, classes=("random",)),
                 dict(S=16, D=16, R=64, C=64, n_oc=3, NBLK=2, NRUN=3, feed=1200, early=500, classes=("random",)),
                 dict(S=32, D=8, R=64, C=64, n_oc=4, NBLK=3, NRUN=2, feed=0, early=300, classes=("random",))]
    P["anchor"] = [dict(S=128, D=64, R=256, C=256, n_oc=7, NBLK=8, ngp=8, classes=("random",))]
    return P


def gen_case(out: Path, spec: dict, cls: str, seed: int) -> str:
    S, D, R, C, n, B = (spec[k] for k in ("S", "D", "R", "C", "n_oc", "NBLK"))
    SO, LN, NRUN = spec.get("SO", 4), spec.get("LN", 4), spec.get("NRUN", 1)
    st = oc.NldpeOcarry(S, D, R, C, n, B, SO=SO, LN=LN, ngp=spec.get("ngp"))
    rng = np.random.default_rng(seed + 1009 * S + 31 * D + 7 * n + 3 * B + R + C + 11 * SO + LN)
    data = [oc._random_inputs(st.g, rng, cls) for _ in range(NRUN)]
    sched = oc.feasible_schedule(st.g, NRUN, 5, op_pace=spec.get("pace", 3 * st.g.LOAD),
                                 la_lead=spec.get("lead", 100), ll_lead=60, wave_pace=spec.get("wpace"),
                                 early=spec.get("early", 0), feed=spec.get("feed", 0))
    name = f"{cls}_S{S}_D{D}_{R}x{C}_n{n}_g{st.g.NGP}_B{B}" + (f"_SO{SO}_LN{LN}" if (SO, LN) != (4, 4) else "") \
        + (f"_runs{NRUN}" if NRUN > 1 else "") + (f"_early{spec['early']}" if spec.get("early") else "") \
        + (f"_lead{spec['lead']}_pace{spec['pace']}" if "lead" in spec else "") \
        + (f"_wp{spec['wpace']}" if spec.get("wpace") else "") \
        + (f"_feed{spec['feed']}" if "feed" in spec else "") \
        + (f"_midreset{int(100 * spec['midreset'])}" if "midreset" in spec else "")
    case = oc.dump_case(out / name, st, data, sched, cls=cls)
    if "midreset" in spec:                     # a cycle inside the first run (fraction of its length)
        import json
        case["MIDRESET"] = max(1, int(spec["midreset"] * case["exp_run_done"][0]))
        (out / name / "case.json").write_text(json.dumps(case, indent=1) + "\n")
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "v3" / "work" / "softmax" / "ocarry_stimuli"))  # v3 path (tools/copies.py)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rc", default=None, help="only the cases on this crossbar, e.g. 64x64 (RTL verification size)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = presets()
    keys = [k for k in P if k != "anchor"] if a.preset in (None, "all") else a.preset.split(",")
    names = []
    for k in keys:
        for spec in P[k]:
            if a.rc and f"{spec['R']}x{spec['C']}" != a.rc:
                continue
            for cls in spec.get("classes", CLASSES):
                names.append(gen_case(out, spec, cls, a.seed))
    if not names:
        raise SystemExit("no case matches")
    (out / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"gen_ocarry_cases: {len(names)} GATE-1-certified cases under {out} (manifest.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
