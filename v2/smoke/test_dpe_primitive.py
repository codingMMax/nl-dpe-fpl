#!/usr/bin/env python3
"""test_dpe_primitive.py — verify/report v2 DPE-primitive test cases.

Reads case directories produced by `gen_cases.py` / `run_dpe_rtl.py` under
`v2/smoke/stimuli/<case>/` and reports, per case:

  sim_cyc     case.json `used_cycles` (the simulator's schedule)
  rtl_cyc     TB `measured` cycles (from `observed/`)
  delta_cyc   rtl_cyc - (T_fill + (M-1)*T_steady) = Δ_impl
  y32_match   expected vs observed int32 `y` at acam_fire
              (pre-ACAM accumulator, P26 dual compare)
  out8_match  expected vs observed drained 8-bit stream
              (post-ACAM mode form + `trunc8`, §4.5)
  verdict     TB counters (errors/ready/span/cycle) + independent recompute

Modes:
  (no args)              verification table over the whole corpus (default)
  --list                 compact index of the corpus ('*' = latest run)
  <case> [<case> ...]    full per-case view: vectors W/X, expected and
                         observed values/bytes, cycle breakdown, match counts

Usage:
  python3 v2/smoke/test_dpe_primitive.py
  python3 v2/smoke/test_dpe_primitive.py identity_256x256_M1_m0
  python3 v2/smoke/test_dpe_primitive.py 8x8 --full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STIMULI = REPO / "v2" / "smoke" / "stimuli"


def decode_weights(path: Path, R: int, C: int) -> np.ndarray:
    words = [int(ln, 16) & 0xFF for ln in path.read_text().split()]
    assert len(words) == R * C, f"{path}: {len(words)} != {R*C}"
    return np.array(words, dtype=np.uint8).view(np.int8).reshape(R, C)


def decode_act(path: Path, M: int, R: int, LCYC: int, EPS: int) -> np.ndarray:
    words = [int(ln, 16) for ln in path.read_text().split()]
    assert len(words) == M * LCYC, f"{path}: {len(words)} != {M*LCYC}"
    out: list[int] = []
    for w in words:
        out.extend((w >> (8 * i)) & 0xFF for i in range(EPS))
    return np.array(out[: R * M], dtype=np.uint8).view(np.int8).reshape(M, R)


def decode_expected_out(path: Path) -> np.ndarray:
    return np.array([[int(ln[i:i+2], 16) for i in range(0, len(ln), 2)]
                     for ln in path.read_text().splitlines() if ln.strip()],
                    dtype=np.uint8)


def decode_observed_y(path: Path) -> np.ndarray:
    vals = []
    for ln in path.read_text().split():
        v = int(ln, 16) & 0xFFFFFFFF
        vals.append(v - (1 << 32) if v >= (1 << 31) else v)
    return np.asarray(vals, dtype=np.int64)


def decode_observed_out(path: Path) -> np.ndarray:
    return np.array([int(ln, 16) for ln in path.read_text().split()], dtype=np.uint8)


def fmt_vec(v: np.ndarray, cols: int) -> str:
    head = ", ".join(str(int(x)) for x in v[:cols])
    return f"[{head}{', ...' if len(v) > cols else ''}]  ({len(v)} values)"


def fmt_bytes(v: np.ndarray, cols: int) -> str:
    return " ".join(f"{b:02x}" for b in v[:cols]) + (" ..." if len(v) > cols else "")


def window(arr: np.ndarray, rows: int, cols: int) -> list[str]:
    lines = []
    for r in range(min(rows, arr.shape[0])):
        row = arr[r]
        head = ", ".join(f"{int(x):4d}" for x in row[:cols])
        lines.append(f"  [{head}{', ...' if len(row) > cols else ''}]")
    if arr.shape[0] > rows:
        lines.append(f"  ... ({arr.shape[0]} rows total)")
    return lines


def find_case(root: Path, token: str) -> Path:
    exact = root / token
    if exact.is_dir():
        return exact
    hits = sorted(p for p in root.iterdir()
                  if p.is_dir() and token.lower() in p.name.lower())
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"no case matching '{token}' under {root.relative_to(REPO)}")
    sys.exit("ambiguous; matches:\n  " + "\n  ".join(p.name for p in hits))


def parse_observed(case_dir: Path, meta: dict) -> dict | None:
    """TB-persisted observed run: counters/verdict + independent bit-exact check."""
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
    exp_y = np.load(case_dir / "expected_y.npz")["y"].astype(np.int64).reshape(-1)
    exp_o = decode_expected_out(case_dir / "expected_out.mem").reshape(-1)
    oy = decode_observed_y(obs / "observed_y.hex")
    oo = decode_observed_out(obs / "observed_out.hex")
    y_ok = len(oy) == exp_y.size and bool((oy == exp_y).all())
    o_ok = len(oo) == exp_o.size and bool((oo == exp_o).all())
    return {"kv": kv, "passes": passes, "y_ok": y_ok, "o_ok": o_ok,
            "y_n": exp_y.size, "o_n": exp_o.size,
            "oy": oy, "oo": oo, "exp_y": exp_y, "exp_o": exp_o}


def formula(meta: dict) -> tuple[int, int, int, int, int, str]:
    R, C, BUF, P, M = (meta[k] for k in ("R", "C", "BUF", "P", "M"))
    LCYC = -(-R * 8 // BUF)
    CCYC = P + 2
    OCYC = -(-C * 8 // BUF)
    t_fill = LCYC + CCYC + OCYC
    t_steady = max(LCYC + P, CCYC, OCYC + 1)
    bound = "input" if t_steady == LCYC + P else "compute" if t_steady == CCYC else "output"
    return LCYC, CCYC, OCYC, t_fill, t_steady, bound


def status_table(root: Path) -> int:
    if not root.is_dir():
        sys.exit(f"no stimuli at {root} — run v2/smoke/run_dpe_rtl.py first")
    manifest = root / "manifest.txt"
    latest = set(manifest.read_text().split()) if manifest.exists() else set()
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    hdr = (f"{'case':<28} {'M':>2} {'mode':>4} {'sim_cyc':>8} {'rtl_cyc':>8} "
           f"{'delta_cyc':>10} {'y32_match':>9} {'out8_match':>10} "
           f"{'verdict':>7}")
    print(hdr)
    print("-" * len(hdr))
    n_bad = 0
    for d in dirs:
        meta = json.loads((d / "case.json").read_text())
        mark = "*" if d.name in latest else " "
        sim = meta["used_cycles"]
        st = parse_observed(d, meta)
        if st is None:
            print(f"{mark} {d.name:<26} {meta['M']:>2} {meta['mode']:>4} {sim:>8} "
                  f"{'-':>8} {'-':>10} {'-':>9} {'-':>10} {'-':>7}")
            continue
        kv = st["kv"]
        rtl = int(kv.get("measured", -1))
        delta = int(kv.get("delta", 0))
        ok = st["y_ok"] and st["o_ok"]
        verdict = kv.get("verdict", "PASS" if ok else "FAIL")
        if not ok:
            verdict = "FAIL"
        ys = f"{st['y_n']}/{st['y_n']}" if st["y_ok"] else "MISMATCH"
        os_ = f"{st['o_n']}/{st['o_n']}" if st["o_ok"] else "MISMATCH"
        print(f"{mark} {d.name:<26} {meta['M']:>2} {meta['mode']:>4} {sim:>8} "
              f"{rtl:>8} {delta:>10} {ys:>9} {os_:>10} {verdict:>7}")
        n_bad += verdict != "PASS"
    print(f"\n{len(dirs)} case(s); '*' = latest generation run")
    print("  sim_cyc    = case.json used_cycles (simulator schedule)")
    print("  rtl_cyc    = TB measured cycles (observed/)")
    print("  delta_cyc  = rtl_cyc - (T_fill + (M-1)*T_steady) = Δ_impl")
    print("  y32_match  = expected vs observed int32 y at acam_fire (pre-ACAM, P26)")
    print("  out8_match = expected vs observed drained 8-bit stream (post-trunc8, §4.5)")
    return 1 if n_bad else 0


def index_list(root: Path) -> int:
    if not root.is_dir():
        sys.exit(f"no stimuli at {root} — run v2/smoke/run_dpe_rtl.py first")
    manifest = root / "manifest.txt"
    latest = set(manifest.read_text().split()) if manifest.exists() else set()
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        meta = json.loads((d / "case.json").read_text())
        obs = "observed" if (d / "observed").is_dir() else "        "
        mark = "*" if d.name in latest else " "
        print(f"{mark} {d.name:<32} M={meta['M']} mode={meta['mode']} "
              f"used_cycles={meta['used_cycles']:>5} {obs}")
    print(f"\n{len(dirs)} case(s); '*' = latest generation run "
          f"(corpus accumulates across runs; regenerate with run_dpe_rtl.py)")
    return 0


def show(case_dir: Path, rows: int, cols: int, diff: bool) -> bool:
    meta = json.loads((case_dir / "case.json").read_text())
    R, C, BUF, P, M, mode = (meta[k] for k in ("R", "C", "BUF", "P", "M", "mode"))
    EPS = BUF // 8
    LCYC, CCYC, OCYC, t_fill, t_steady, bound = formula(meta)
    total = t_fill + (M - 1) * t_steady
    st = parse_observed(case_dir, meta) if diff else None

    print("=" * 78)
    print(f"case {case_dir.name}")
    print(f"  R={R} C={C} BUF={BUF} P={P} M={M} mode={mode} "
          f"(0=REGULAR 1=ACTIVATION 2=EXP 3=LOG)")

    print("\ncycles")
    print(f"  sim_cyc  (case.json used_cycles)     : {meta['used_cycles']}")
    print(f"  formula  (§5.3 T_fill+(M-1)*T_steady): {total}   "
          f"{'match' if total == meta['used_cycles'] else 'MISMATCH'}"
          f"  [LOAD={LCYC} COMPUTE={CCYC} OUTPUT={OCYC} T_fill={t_fill} "
          f"T_steady={t_steady} bound={bound}]")
    if st is not None:
        kv = st["kv"]
        print(f"  rtl_cyc  (TB measured)               : {kv.get('measured', '?')}   "
              f"delta_cyc={kv.get('delta', '?')} (Δ_impl)")
    else:
        print("  rtl_cyc  (TB measured)               : -  "
              "(no observed dump — run v2/smoke/run_dpe_rtl.py)")

    W = decode_weights(case_dir / "weights.mem", R, C)
    X = decode_act(case_dir / "act.mem", M, R, LCYC, EPS)
    exp_y = np.load(case_dir / "expected_y.npz")["y"].astype(np.int64)
    exp_o = decode_expected_out(case_dir / "expected_out.mem")

    print(f"\nW [{R},{C}] min={W.min()} max={W.max()} zeros={(W == 0).sum()}"
          + ("  IDENTITY" if R == C and (W == np.eye(R, dtype=np.int8)).all() else ""))
    for ln in window(W.astype(np.int64), rows, cols):
        print(ln)
    print(f"\nX [{M},{R}] per pass:")
    for m in range(M):
        print(f"  pass {m}: {fmt_vec(X[m], cols)}")
    print(f"\nexpected y (int32, pre-ACAM): {fmt_vec(exp_y.reshape(-1), cols)}")
    print("\nexpected bytes (stream, column order):")
    for m in range(M):
        print(f"  pass {m}: {fmt_bytes(exp_o[m], cols)}")

    print("\nvalues / GATE 2")
    if st is None:
        print("  (no observed dump — run v2/smoke/run_dpe_rtl.py)")
        return True

    oy, oo = st["oy"], st["oo"]
    print(f"  observed int32 y     : {fmt_vec(oy, cols)}")
    print(f"  observed bytes       : {fmt_bytes(oo, cols)}")
    y_line = (f"{st['y_n']}/{st['y_n']} bit-exact" if st["y_ok"] else "MISMATCH")
    o_line = (f"{st['o_n']}/{st['o_n']} bit-exact" if st["o_ok"] else "MISMATCH")
    print(f"  y32_match            : {y_line}   (expected int32 y at acam_fire)")
    print(f"  out8_match           : {o_line}   (expected drained 8-bit stream)")
    kv = st["kv"]
    if "verdict" in kv:
        print(f"  tb verdict: {kv['verdict']} (errors={kv.get('errors', '?')} "
              f"ready={kv.get('ready_err', '?')} span={kv.get('span_err', '?')} "
              f"cycle={kv.get('cycle_err', '?')})")
    if st["passes"]:
        print("  pass cycles: " + "; ".join(st["passes"]))

    ok = st["y_ok"] and st["o_ok"]
    for name, got, exp in (("y", oy, st["exp_y"]), ("out", oo, st["exp_o"])):
        if len(got) != exp.size:
            print(f"  {name} size mismatch: observed {len(got)} vs expected {exp.size}")
            continue
        bad = np.nonzero(got != exp)[0]
        if len(bad):
            print(f"  {name} MISMATCH: {len(bad)}/{exp.size} "
                  f"(first {min(5, len(bad))}):")
            for i in bad[:5]:
                if name == "y":
                    print(f"    y[{i // C},{i % C}] exp={int(exp[i])} got={int(got[i])}")
                else:
                    print(f"    out[{i // C},{i % C}] exp={exp[i]:02x} got={got[i]:02x}")
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
        ok &= show(find_case(root, token), rows, cols, not args.no_diff)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
