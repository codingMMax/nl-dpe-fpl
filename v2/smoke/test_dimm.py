#!/usr/bin/env python3
"""test_dimm.py — verify/report Stage-4 DIMM cases.

Layout mirrors `test_gemm.py` (same table / --list / per-case UX) and reuses
`test_dpe_primitive.py` formatters. Reads case dirs under
`v2/smoke/dimm_stimuli/<case>/` produced by `gen_dimm_cases.py` /
`run_dimm_rtl.py`:

  model_cyc   case.json `used_cycles` (overlapped pool/farm model)
  ser_cyc     case.json `serialize_cycles` (M*N, C serializer)
  rtl_cyc     TB `measured` cycles (start -> done)
  delta_cyc   rtl_cyc - (model_cyc + ser_cyc) = Δ_impl
  la_match    expected vs observed LA_T (K*M int8, transposed probe order)
  lb_match    expected vs observed LB (K*N int8)
  acc_match   expected vs observed acc_q (M*N int32, valid at done)
  c_match     expected vs observed C stream (M*N int32, row-major)
  verdict     TB counters (errors/ready/cycle)

Modes:
  (no args)              verification table over the corpus (default)
  --list                 compact index of the corpus ('*' = latest run)
  <case> [<case> ...]    full per-case view

Usage:
  python3 v2/smoke/test_dimm.py
  python3 v2/smoke/test_dimm.py random_8x10x6_256x256_nE4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_dpe_primitive as tp  # noqa: E402  (reuse find_case + formatters)

REPO = HERE.parents[1]
STIMULI = REPO / "v2" / "smoke" / "dimm_stimuli"


# -- DIMM-specific decoders ---------------------------------------------------
def decode_bytes(path: Path) -> np.ndarray | None:
    """2-hex lines -> int8; None when the dump contains X/Z (undriven RTL)."""
    toks = path.read_text().split()
    if any(("x" in t.lower() or "z" in t.lower()) for t in toks):
        return None
    return np.array([int(t, 16) & 0xFF for t in toks],
                    dtype=np.uint8).view(np.int8)


def decode_words(path: Path) -> np.ndarray | None:
    """8-hex lines -> int32; None when the dump contains X/Z."""
    toks = path.read_text().split()
    if any(("x" in t.lower() or "z" in t.lower()) for t in toks):
        return None
    return np.array([int(t, 16) & 0xFFFFFFFF for t in toks],
                    dtype=np.uint32).view(np.int32)


def expected_bits(case_dir: Path, meta: dict) -> dict:
    M, N, K = int(meta["M"]), int(meta["N"]), int(meta["K"])
    la = np.load(case_dir / "expected_la.npz")["la"].astype(np.int8)
    lb = np.load(case_dir / "expected_lb.npz")["lb"].astype(np.int8)
    acc = np.load(case_dir / "expected_acc.npz")["acc"].astype(np.int32)
    c = decode_words(case_dir / "expected_c.mem")
    assert la.shape == (K, M) and lb.shape == (K, N)
    assert acc.shape == (M, N) and c.shape == (M * N,)
    return {"la": la.reshape(-1), "lb": lb.reshape(-1),
            "acc": acc.reshape(-1), "c": c}


def parse_observed(case_dir: Path, meta: dict) -> dict | None:
    """TB-persisted observed run: counters/verdict + independent match checks."""
    obs = case_dir / "observed"
    cyc = obs / "observed_cycles.txt"
    if not cyc.exists():
        return None
    kv: dict[str, str] = {}
    for ln in cyc.read_text().splitlines():
        for tok in ln.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                kv[k] = v
    exp = expected_bits(case_dir, meta)
    obs_la = decode_bytes(obs / "observed_la.hex")
    obs_lb = decode_bytes(obs / "observed_lb.hex")
    obs_acc = decode_words(obs / "observed_acc.hex")
    obs_c = decode_words(obs / "observed_c.hex")
    got = {"la": obs_la, "lb": obs_lb, "acc": obs_acc, "c": obs_c}
    match = {k: (got[k] is not None and got[k].shape == exp[k].shape
                 and bool((got[k] == exp[k]).all())) for k in exp}
    return {"kv": kv, "exp": exp, "obs": got, "match": match}


def status_table(root: Path) -> int:
    if not root.is_dir():
        sys.exit(f"no stimuli at {root} — run v2/smoke/run_dimm_rtl.py first")
    manifest = root / "manifest.txt"
    latest = set(manifest.read_text().split()) if manifest.exists() else set()
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    hdr = (f"{'case':<38} {'nE':>3} {'model':>7} {'ser':>5} {'rtl':>8} "
           f"{'delta':>6} {'stages':>7} {'cyc':>5} {'verdict':>7}")
    print(hdr)
    print("-" * len(hdr))
    n_bad = 0
    for d in dirs:
        if not (d / "case.json").exists():
            continue        # incomplete case (interrupted generation)
        meta = json.loads((d / "case.json").read_text())
        mark = "*" if d.name in latest else " "
        st = parse_observed(d, meta)
        if st is None:
            print(f"{mark} {d.name:<36} {meta['n_E']:>3} "
                  f"{meta['used_cycles']:>7} {meta['serialize_cycles']:>5} "
                  f"{'-':>8} {'-':>6} {'-':>7} {'-':>7}")
            continue
        kv = st["kv"]
        rtl = int(kv.get("measured", -1))
        delta = int(kv.get("delta", 0))
        stages_ok = all(st["match"].values())
        verdict = kv.get("verdict", "PASS" if stages_ok else "FAIL")
        if not stages_ok:
            verdict = "FAIL"
        sm = "all-ok" if stages_ok else "MISMATCH"
        cyc_ok = (int(kv.get("span_A", -1)) == int(meta["T_A"])
                  and int(kv.get("span_B", -1)) == int(meta["T_B"])
                  and int(kv.get("span_F", -1)) == int(meta["T_E"])
                  and int(kv.get("fill", -1)) == int(meta["T_start"])
                  and int(kv.get("ser", -1)) == int(meta["serialize_cycles"]))
        cy = "exact" if cyc_ok else "BAD"
        print(f"{mark} {d.name:<36} {meta['n_E']:>3} "
              f"{meta['used_cycles']:>7} {meta['serialize_cycles']:>5} "
              f"{rtl:>8} {delta:>6} {sm:>7} {cy:>5} {verdict:>7}")
        n_bad += verdict != "PASS" or not cyc_ok
    print(f"\n{len(dirs)} case(s); '*' = latest generation run")
    print("  model_cyc  = case.json used_cycles (overlapped pool/farm)")
    print("  ser_cyc    = M*N C-serializer words (RTL-only, reported)")
    print("  rtl_cyc    = TB measured cycles (start -> done)")
    print("  delta_cyc  = rtl_cyc - (model_cyc + ser_cyc) = Δ_impl")
    print("  stages     = la / lb / acc / C all bit-exact")
    return 1 if n_bad else 0


def index_list(root: Path) -> int:
    if not root.is_dir():
        sys.exit(f"no stimuli at {root} — run v2/smoke/run_dimm_rtl.py first")
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
        print(f"{mark} {d.name:<40} n_A={meta['n_A']} n_B={meta['n_B']} "
              f"n_E={meta['n_E']} used_cycles={meta['used_cycles']:>6} {obs}")
    print(f"\n{n_complete} case(s); '*' = latest generation run "
          f"(corpus accumulates across runs; regenerate with run_dimm_rtl.py)")
    return 0


def show(case_dir: Path, rows: int, cols: int, diff: bool) -> bool:
    meta = json.loads((case_dir / "case.json").read_text())
    M, N, K = int(meta["M"]), int(meta["N"]), int(meta["K"])
    R, C, BUF, P = (int(meta[k]) for k in ("R", "C", "BUF", "P"))
    st = parse_observed(case_dir, meta) if diff else None

    print("=" * 78)
    print(f"case {case_dir.name}")
    print(f"  M={M} N={N} K={K} R={R} C={C} BUF={BUF} P={P} "
          f"n_A={meta['n_A']} n_B={meta['n_B']} n_E={meta['n_E']}")

    print("\ncycles")
    print(f"  model    (case.json used_cycles)   : {meta['used_cycles']}   "
          f"[T_A={meta['T_A']} T_B={meta['T_B']} T_E={meta['T_E']} "
          f"passes=({meta['passes_A']},{meta['passes_B']},{meta['passes_E']})]")
    print(f"  serialize (M*N, RTL C stream)      : {meta['serialize_cycles']}")
    exp = meta["used_cycles"] + meta["serialize_cycles"]
    if st is not None:
        kv = st["kv"]
        print(f"  rtl      (TB measured)             : {kv.get('measured', '?')}   "
              f"delta_cyc={kv.get('delta', '?')} (Δ_impl)")
        print(f"  expected (model + serialize)       : {exp}")
        print(f"  load_cycles={kv.get('load_cycles', '?')} "
              f"weight_cycles={kv.get('weight_cycles', '?')} "
              f"(setup, excluded)")
        print(f"  spans    : A {kv.get('span_A')}/{meta['T_A']}  "
              f"B {kv.get('span_B')}/{meta['T_B']}  "
              f"F {kv.get('span_F')}/{meta['T_E']}  "
              f"fill {kv.get('fill')}/{meta['T_start']}  "
              f"ser {kv.get('ser')}/{meta['serialize_cycles']}   "
              f"(observed/expected)")
        print(f"  windows  : A {kv.get('wins_A')}/{meta['passes_A']}  "
              f"B {kv.get('wins_B')}/{meta['passes_B']}  "
              f"F {kv.get('wins_F')}/{meta['passes_E']}")
    else:
        print("  rtl      (TB measured)             : -  "
              "(no observed dump — run v2/smoke/run_dimm_rtl.py)")

    exp_bits = expected_bits(case_dir, meta)
    print(f"\nexpected LA_T [{K},{M}] (probe order k*M+m): "
          f"{tp.fmt_bytes(exp_bits['la'], cols)}")
    print(f"expected LB   [{K},{N}] (index k*N+n): "
          f"{tp.fmt_bytes(exp_bits['lb'], cols)}")
    print(f"expected acc  [{M},{N}] int32: "
          f"{tp.fmt_vec(exp_bits['acc'], cols)}")
    print(f"expected C    [{M},{N}] int32: "
          f"{tp.fmt_vec(exp_bits['c'], cols)}")

    print("\nvalues / GATE 2")
    if st is None:
        print("  (no observed dump — run v2/smoke/run_dimm_rtl.py)")
        return True
    for key, label in (("la", "LA_T"), ("lb", "LB"), ("acc", "acc_q"),
                       ("c", "C stream")):
        got, want = st["obs"][key], st["exp"][key]
        ok = st["match"][key]
        if ok:
            line = f"{want.size}/{want.size} bit-exact"
        elif got is None:
            line = "MISMATCH (dump has X/Z — RTL outputs undriven)"
        else:
            line = "MISMATCH"
        print(f"  {label:<9}: {line}")
        if not ok and got is not None and got.size == want.size:
            bad = np.nonzero(got != want)[0]
            for i in bad[:5]:
                print(f"    [{i}] exp={int(want[i])} got={int(got[i])}")
    kv = st["kv"]
    if "verdict" in kv:
        print(f"  tb verdict: {kv['verdict']} (errors={kv.get('errors', '?')} "
              f"ready={kv.get('ready_err', '?')} cycle={kv.get('cycle_err', '?')})")
    return all(st["match"].values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="case name (or unique substring)")
    ap.add_argument("--stimuli", default=str(STIMULI))
    ap.add_argument("--list", action="store_true", help="compact corpus index")
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
