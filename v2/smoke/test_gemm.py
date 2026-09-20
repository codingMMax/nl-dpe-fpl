#!/usr/bin/env python3
"""test_gemm.py — verify/report Stage-2 GEMM array cases.

Layout mirrors `test_dpe_primitive.py` (same table / --list / per-case UX) and
reuses its decoders and formatters. Reads case dirs under
`v2/smoke/gemm_stimuli/<case>/` produced by `gen_gemm_cases.py` /
`run_gemm_rtl.py`:

  sim_cyc     case.json `used_cycles` (simulator schedule)
  rtl_cyc     TB `measured` cycles (from `observed/`)
  delta_cyc   rtl_cyc - (T_fill_array + (M-1)*T_steady) = Δ_impl
  psum_match  expected vs observed wide reduced partial S (int32, H*C cols)
  out8_match  expected vs observed lane bytes (H*C per pass)
  verdict     TB counters (errors/ready/cycle) + independent recompute

Modes:
  (no args)              verification table over the corpus (default)
  --list                 compact index of the corpus ('*' = latest run)
  <case> [<case> ...]    full per-case view

Usage:
  python3 v2/smoke/test_gemm.py
  python3 v2/smoke/test_gemm.py identity_256x256_128x256_M1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_dpe_primitive as tp  # noqa: E402  (reuse decoders + formatters)

REPO = HERE.parents[1]
STIMULI = REPO / "v2" / "smoke" / "gemm_stimuli"


# -- GEMM-specific decoders ---------------------------------------------------
def decode_weights(path: Path, count: int) -> np.ndarray:
    words = [int(ln, 16) & 0xFF for ln in path.read_text().split()]
    assert len(words) == count, f"{path}: {len(words)} != {count}"
    return np.array(words, dtype=np.uint8).view(np.int8)


def decode_expected_psum(path: Path) -> np.ndarray:
    return np.load(path)["S"].astype(np.int64)


def decode_act_lanes(path: Path, M: int, V: int, LCYC: int, R: int,
                     EPS: int) -> np.ndarray:
    """act.mem (M bursts x V lane blocks of LCYC words) -> int8 [M, V, R]."""
    words = [int(ln, 16) for ln in path.read_text().split()]
    assert len(words) == M * V * LCYC, f"{path}: {len(words)} != {M*V*LCYC}"
    out = np.zeros((M, V, R), dtype=np.int8)
    idx = 0
    for m in range(M):
        for v in range(V):
            lane: list[int] = []
            for _ in range(LCYC):
                lane.extend((words[idx] >> (8 * i)) & 0xFF for i in range(EPS))
                idx += 1
            out[m, v] = np.array(lane[:R], dtype=np.uint8).view(np.int8)
    return out


def parse_observed(case_dir: Path, meta: dict) -> dict | None:
    """TB-persisted observed run: counters/verdict + independent match check."""
    obs = case_dir / "observed"
    cyc = obs / "observed_cycles.txt"
    if not cyc.exists():
        return None
    kv: dict[str, str] = {}
    passes: list[str] = []
    for ln in cyc.read_text().splitlines():
        if ln.startswith("pass="):
            passes.append(ln)
            continue
        for tok in ln.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                kv[k] = v
    M, K, N = (int(meta[k]) for k in ("M", "K", "N"))
    R, C = int(meta["R"]), int(meta["C"])
    V, H = -(-K // R), -(-N // C)
    exp_S = decode_expected_psum(case_dir / "expected_psum.npz")
    Sp = np.zeros((M, H * C), dtype=np.int64)
    Sp[:, :N] = exp_S
    exp_o = tp.decode_expected_out(case_dir / "expected_out.mem").reshape(-1)
    obs_S = tp.decode_observed_y(obs / "observed_psum.hex")
    obs_o = tp.decode_observed_out(obs / "observed_out.hex")
    psum_ok = len(obs_S) == Sp.size and bool((obs_S == Sp.reshape(-1)).all())
    out_ok = len(obs_o) == exp_o.size and bool((obs_o == exp_o).all())
    return {"kv": kv, "passes": passes, "psum_ok": psum_ok, "out_ok": out_ok,
            "obs_S": obs_S, "obs_o": obs_o, "exp_S": Sp, "exp_o": exp_o,
            "M": M, "K": K, "N": N, "R": R, "C": C, "V": V, "H": H}


def status_table(root: Path) -> int:
    if not root.is_dir():
        sys.exit(f"no stimuli at {root} — run v2/smoke/run_gemm_rtl.py first")
    manifest = root / "manifest.txt"
    latest = set(manifest.read_text().split()) if manifest.exists() else set()
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    hdr = (f"{'case':<40} {'M':>2} {'sim_cyc':>8} {'rtl_cyc':>8} "
           f"{'delta_cyc':>10} {'psum_match':>10} {'out8_match':>10} {'verdict':>7}")
    print(hdr)
    print("-" * len(hdr))
    n_bad = 0
    for d in dirs:
        if not (d / "case.json").exists():
            continue        # incomplete case (interrupted generation)
        meta = json.loads((d / "case.json").read_text())
        mark = "*" if d.name in latest else " "
        sim = meta["used_cycles"]
        st = parse_observed(d, meta)
        if st is None:
            print(f"{mark} {d.name:<38} {meta['M']:>2} {sim:>8} "
                  f"{'-':>8} {'-':>10} {'-':>10} {'-':>10} {'-':>7}")
            continue
        kv = st["kv"]
        rtl = int(kv.get("measured", -1))
        delta = int(kv.get("delta", 0))
        ok = st["psum_ok"] and st["out_ok"]
        verdict = kv.get("verdict", "PASS" if ok else "FAIL")
        if not ok:
            verdict = "FAIL"
        ps = f"{st['exp_S'].size}/{st['exp_S'].size}" if st["psum_ok"] else "MISMATCH"
        os_ = f"{st['exp_o'].size}/{st['exp_o'].size}" if st["out_ok"] else "MISMATCH"
        print(f"{mark} {d.name:<38} {meta['M']:>2} {sim:>8} {rtl:>8} "
              f"{delta:>10} {ps:>10} {os_:>10} {verdict:>7}")
        n_bad += verdict != "PASS"
    print(f"\n{len(dirs)} case(s); '*' = latest generation run")
    print("  sim_cyc    = case.json used_cycles (simulator schedule)")
    print("  rtl_cyc    = TB measured cycles (observed/)")
    print("  delta_cyc  = rtl_cyc - (T_fill_array + (M-1)*T_steady) = Δ_impl")
    print("  psum_match = expected vs observed wide S at dpe_done (int32, H*C cols)")
    print("  out8_match = expected vs observed lane bytes (H*C per pass)")
    return 1 if n_bad else 0


def index_list(root: Path) -> int:
    if not root.is_dir():
        sys.exit(f"no stimuli at {root} — run v2/smoke/run_gemm_rtl.py first")
    manifest = root / "manifest.txt"
    latest = set(manifest.read_text().split()) if manifest.exists() else set()
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    n_complete = 0
    for d in dirs:
        if not (d / "case.json").exists():
            continue        # incomplete case (interrupted generation)
        n_complete += 1
        meta = json.loads((d / "case.json").read_text())
        obs = "observed" if (d / "observed").is_dir() else "        "
        mark = "*" if d.name in latest else " "
        print(f"{mark} {d.name:<40} M={meta['M']} K={meta['K']} N={meta['N']} "
              f"V={meta['V']} H={meta['H']} used_cycles={meta['used_cycles']:>5} {obs}")
    print(f"\n{n_complete} case(s); '*' = latest generation run "
          f"(corpus accumulates across runs; regenerate with run_gemm_rtl.py)")
    return 0


def show(case_dir: Path, rows: int, cols: int, diff: bool) -> bool:
    meta = json.loads((case_dir / "case.json").read_text())
    M, K, N = int(meta["M"]), int(meta["K"]), int(meta["N"])
    R, C, BUF, P = (int(meta[k]) for k in ("R", "C", "BUF", "P"))
    V, H = int(meta["V"]), int(meta["H"])
    LCYC = -(-R * 8 // BUF)
    CCYC = P + 2
    OCYC = -(-C * 8 // BUF)
    TREE = (V - 1).bit_length() if V > 1 else 0
    L_W = TREE + 1
    t_fill = LCYC + CCYC + OCYC + L_W
    t_steady = max(LCYC + P, CCYC, OCYC + 1, 1)
    total = t_fill + (M - 1) * t_steady
    st = parse_observed(case_dir, meta) if diff else None

    print("=" * 78)
    print(f"case {case_dir.name}")
    print(f"  K={K} N={N} R={R} C={C} BUF={BUF} P={P} M={M} "
          f"-> V={V} H={H} (tiles: {V*H})")

    print("\ncycles")
    print(f"  sim_cyc  (case.json used_cycles)      : {meta['used_cycles']}")
    print(f"  formula  (T_fill_array+(M-1)*T_steady): {total}   "
          f"{'match' if total == meta['used_cycles'] else 'MISMATCH'}"
          f"  [LOAD={LCYC} COMPUTE={CCYC} OUTPUT={OCYC} TREE_PIPE={TREE} "
          f"L_w={L_W} T_fill={t_fill} T_steady={t_steady}]")
    if st is not None:
        kv = st["kv"]
        print(f"  rtl_cyc  (TB measured)                : {kv.get('measured', '?')}   "
              f"delta_cyc={kv.get('delta', '?')} (Δ_impl)")
    else:
        print("  rtl_cyc  (TB measured)                : -  "
              "(no observed dump — run v2/smoke/run_gemm_rtl.py)")

    W = decode_weights(case_dir / "weights.mem", V * H * R * C)
    X = decode_act_lanes(case_dir / "act.mem", M, V, LCYC, R, BUF // 8)
    exp_S = decode_expected_psum(case_dir / "expected_psum.npz")
    exp_o = tp.decode_expected_out(case_dir / "expected_out.mem")

    print(f"\nW [{K},{N}] min={W.min()} max={W.max()} zeros={(W == 0).sum()}"
          + ("  IDENTITY" if K == N and (W == np.eye(K, dtype=np.int8)).all() else ""))
    print(f"X [{M},{K}] per pass (lane slices):")
    for m in range(M):
        for v in range(V):
            print(f"  pass {m} lane {v}: {tp.fmt_vec(X[m, v], cols)}")
    print(f"\nexpected psum S [{M},{N}] (int32, pre-serializer): "
          f"{tp.fmt_vec(exp_S.reshape(-1), cols)}")
    print("\nexpected bytes (lane-major, H*C per pass; padding cols zero):")
    for m in range(M):
        print(f"  pass {m}: {tp.fmt_bytes(exp_o[m], cols)}")

    print("\nvalues / GATE 2")
    if st is None:
        print("  (no observed dump — run v2/smoke/run_gemm_rtl.py)")
        return True
    obs_S, obs_o = st["obs_S"], st["obs_o"]
    print(f"  observed psum S : {tp.fmt_vec(obs_S, cols)}")
    print(f"  observed bytes  : {tp.fmt_bytes(obs_o, cols)}")
    ps_line = (f"{st['exp_S'].size}/{st['exp_S'].size} bit-exact"
               if st["psum_ok"] else "MISMATCH")
    o_line = (f"{st['exp_o'].size}/{st['exp_o'].size} bit-exact"
              if st["out_ok"] else "MISMATCH")
    print(f"  psum_match      : {ps_line}   (wide S, H*C columns)")
    print(f"  out8_match      : {o_line}   (lane bytes)")
    kv = st["kv"]
    if "verdict" in kv:
        print(f"  tb verdict: {kv['verdict']} (errors={kv.get('errors', '?')} "
              f"ready={kv.get('ready_err', '?')} cycle={kv.get('cycle_err', '?')})")
    if st["passes"]:
        print("  pass cycles: " + "; ".join(st["passes"]))

    ok = st["psum_ok"] and st["out_ok"]
    for name, got, exp, width in (("S", obs_S, st["exp_S"].reshape(-1), N),
                                  ("out", obs_o, st["exp_o"], H * C)):
        if len(got) != exp.size:
            print(f"  {name} size mismatch: observed {len(got)} vs expected {exp.size}")
            continue
        bad = np.nonzero(got != exp)[0]
        if len(bad):
            print(f"  {name} MISMATCH: {len(bad)}/{exp.size} (first {min(5, len(bad))}):")
            for i in bad[:5]:
                print(f"    {name}[{i // width}, {i % width}] "
                      f"exp={int(exp[i])} got={int(got[i])}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="case name (or unique substring)")
    ap.add_argument("--stimuli", default=str(STIMULI))
    ap.add_argument("--list", action="store_true", help="compact corpus index")
    ap.add_argument("--status", action="store_true",
                    help="verification table (default when no case is given)")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--full", action="store_true", help="no window limits")
    ap.add_argument("--no-diff", action="store_true", help="skip observed diff")
    args = ap.parse_args()

    root = Path(args.stimuli)
    rows = 10**9 if args.full else args.rows
    cols = 10**9 if args.full else args.cols

    if not args.cases:
        return index_list(root) if args.list else status_table(root)
    if args.list:
        return index_list(root)

    ok = True
    for i, token in enumerate(args.cases):
        if i:
            print()
        ok &= show(tp.find_case(root, token), rows, cols, not args.no_diff)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
