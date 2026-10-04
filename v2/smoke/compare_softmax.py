#!/usr/bin/env python3
"""compare_softmax.py — conventional vs online softmax (values + cycles + storage).

Runs both v2 softmax machines on identical scores and reports:

  A. values   — deviation from the exact dense softmax
                (conventional = ACAM surrogate + global max, approximate;
                 online = full-precision block rescale, exact by construction)
  B. cycles   — used / compute / emit for both machines, and the delta
  C. storage  — score-side buffer elements (the online win)

Run:  python3 v2/smoke/compare_softmax.py [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
sys.path.insert(0, str(HERE.parent / "oracle"))
import softmax_ref as sref  # noqa: E402
import softmax_online_ref as oref  # noqa: E402
from softmax_sim import NldpeSoftmax  # noqa: E402
from softmax_online_sim import NldpeSoftmaxOnline  # noqa: E402

R = C = 256
CLASSES = {"small": (-16, 16), "full": (-128, 127)}
BKV = lambda S: sorted({16, 32, S})


def storage_bytes(S: int, Bkv: int) -> dict:
    """Score-side storage (bytes): 1 B/score, int8 max, 4 B partial."""
    B = S // Bkv
    return {
        "conv": S * S + 3 * S,
        "online": S * Bkv + B * S * 1 + B * S * 4,
    }


def _decompose(err: np.ndarray) -> tuple[int, int]:
    """Split a per-row error matrix into (offset, relative) maxima.

    In log-domain softmax the per-row additive constant is arbitrary: only the
    within-row differences shape the distribution.  `offset` = max |row median|,
    `rel` = max |err - row median| (distribution-relevant).
    """
    off = np.median(err, axis=1, keepdims=True)
    return int(np.abs(off).max()), int(np.abs(err - off).max())


def run_corpus(seed: int = 3):
    rows = []
    for S in (128, 256):
        for cname, (lo, hi) in CLASSES.items():
            rng = np.random.default_rng(seed + S + (0 if cname == "small" else 1))
            scores = rng.integers(lo, hi + 1, size=(S, S), dtype=np.int8)
            conv_ref = sref.softmax_stage_values(scores.astype(np.int32))
            reg = oref.regular_exact(scores.astype(np.int32))
            for Bkv in BKV(S):
                exact = oref.softmax_online_exact(scores.astype(np.int32), Bkv)
                for n_exp, n_log in ((1, 1), (2, 1), (4, 2)):
                    conv = NldpeSoftmax(S=S, R=R, C=C, n_exp=n_exp,
                                        n_log=n_log).run(scores)
                    onl = NldpeSoftmaxOnline(S=S, R=R, C=C, Bkv=Bkv,
                                             n_exp=n_exp, n_log=n_log).run(scores)
                    assert np.array_equal(conv.softmax_out,
                                          conv_ref["softmax_out"]), "conv != ref"
                    onl_model = oref.softmax_online_model(
                        scores.astype(np.int32), Bkv)
                    assert np.array_equal(onl.softmax_out,
                                          onl_model["out8"]), "online != oracle"
                    e = exact["out8_exact"].astype(np.int32)
                    c_off, c_rel = _decompose(
                        conv.softmax_out.astype(np.int32) - e)
                    o_off, o_rel = _decompose(
                        onl.softmax_out.astype(np.int32) - e)
                    st = storage_bytes(S, Bkv)
                    rows.append(dict(
                        S=S, cls=cname, Bkv=Bkv, n_exp=n_exp, n_log=n_log,
                        conv_off=c_off, conv_rel=c_rel,
                        onl_off=o_off, onl_rel=o_rel,
                        conv_L1=_l1(conv.softmax_out, exact["p"]),
                        onl_L1=_l1(onl.softmax_out, exact["p"]),
                        reg_L1=_l1(reg["out8"], exact["p"]),
                        conv_clamp=oref.clamped_fraction(conv.softmax_out),
                        onl_clamp=oref.clamped_fraction(onl.softmax_out),
                        reg_clamp=oref.clamped_fraction(reg["out8"]),
                        conv_used=conv.used_cycles, conv_comp=conv.compute_cycles,
                        onl_used=onl.used_cycles, onl_comp=onl.compute_cycles,
                        onl_emit=onl.emit_start,
                        d_used=onl.used_cycles - conv.used_cycles,
                        conv_pexp=conv.passes_exp, onl_pexp=onl.passes_exp,
                        conv_plog=conv.passes_log, onl_plog=onl.passes_log,
                        conv_store=st["conv"], onl_store=st["online"]))
    return rows


def _l1(out8: np.ndarray, p: np.ndarray) -> float:
    """Max per-row L1 distance between a reconstructed and the exact dist."""
    return float(np.abs(oref.reconstruct_p(out8) - p).sum(1).max())


def print_tables(rows) -> None:
    print("## A. Values — deviation from the exact dense softmax\n")
    print("`offset` = per-row additive constant (arbitrary in log-domain softmax); "
          "`rel` = within-row error; `L1` = max per-row L1 of the reconstructed "
          "distribution vs exact `p`; `clamp%` = outputs at the ±128/127 rails. "
          "Regular-exact is 0 for all columns by definition.\n")
    print("| S | class | score range | conv offset | conv rel | conv L1 | "
          "conv clamp% | online offset | online rel | online L1 | online clamp% |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    seen = set()
    for r in rows:
        key = (r["S"], r["cls"])
        if key in seen:
            continue
        seen.add(key)
        lo, hi = CLASSES[r["cls"]]
        print(f"| {r['S']} | {r['cls']} | [{lo},{hi}] | {r['conv_off']} | "
              f"{r['conv_rel']} | {r['conv_L1']:.4f} | {r['conv_clamp']:.4f} | "
              f"{r['onl_off']} | {r['onl_rel']} | {r['onl_L1']:.4f} | "
              f"{r['onl_clamp']:.4f} |")

    print("\n## B. Cycles — identical scores/geometry (conv = full-row machine)\n")
    print("| S | Bkv | n_exp | n_log | conv used | conv comp | online used | "
          "online comp | online emit | Δused |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    seen = set()
    for r in rows:
        key = (r["S"], r["Bkv"], r["n_exp"], r["n_log"])
        if key in seen:
            continue
        seen.add(key)
        print(f"| {r['S']} | {r['Bkv']} | {r['n_exp']} | {r['n_log']} | "
              f"{r['conv_used']} | {r['conv_comp']} | {r['onl_used']} | "
              f"{r['onl_comp']} | {r['onl_emit']} | {r['d_used']:+d} |")

    print("\n## C. Pass counts + score-side storage\n")
    print("| S | Bkv | blocks | conv passes e/l | online passes e/l | "
          "conv store (B) | online store (B) | ratio |")
    print("|---|---:|---:|---|---|---:|---:|---:|")
    seen = set()
    for r in rows:
        key = (r["S"], r["Bkv"])
        if key in seen:
            continue
        seen.add(key)
        print(f"| {r['S']} | {r['Bkv']} | {r['S'] // r['Bkv']} | "
              f"{r['conv_pexp']}/{r['conv_plog']} | "
              f"{r['onl_pexp']}/{r['onl_plog']} | {r['conv_store']} | "
              f"{r['onl_store']} | {r['conv_store'] / r['onl_store']:.2f}× |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    rows = run_corpus()
    print_tables(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
