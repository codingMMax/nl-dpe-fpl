#!/usr/bin/env python3
"""test_softmax_online.py — inspect the online-softmax RTL corpus (GATE-2 logs).

Reads the per-case TB logs written by `v2/smoke/run_softmax_online_rtl.py`
(`v2/smoke/logs/smol_<case>.log`) and prints the verification table.

Usage:
  python3 v2/smoke/test_softmax_online.py          # table over all logged cases
  python3 v2/smoke/test_softmax_online.py --list   # case names only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "v2" / "smoke" / "logs"
STIMULI = REPO / "v2" / "smoke" / "softmax_online_stimuli"

RESULT_RE = re.compile(r"\[tb_softmax_online\] (PASS|FAIL) (\S+) errors=(\d+) "
                       r"measured=(-?\d+) expected=(-?\d+) delta=(-?\d+) "
                       r"words=(\d+)")
HEADER_RE = re.compile(r"\[tb_softmax_online\] case=\S+ S=(\d+) .* BKV=(\d+) "
                       r"N_EXP=(\d+)")


def case_meta(name: str) -> dict:
    path = STIMULI / name / "case.json"
    return json.loads(path.read_text()) if path.exists() else {}


def scan() -> list[dict]:
    rows = []
    for log in sorted(LOGS.glob("smol_*.log"), key=lambda p: p.stat().st_mtime):
        m = RESULT_RE.search(log.read_text())
        if not m:
            continue
        v, name, errs, meas, exp, delta, words = m.groups()
        meta = case_meta(name)
        hm = HEADER_RE.search(log.read_text())
        S = meta.get("S") or (int(hm.group(1)) if hm else 0)
        Bkv = meta.get("Bkv") or (int(hm.group(2)) if hm else 0)
        nX = meta.get("n_exp") or (int(hm.group(3)) if hm else 0)
        rows.append({"case": name, "verdict": v, "errors": int(errs),
                     "measured": int(meas), "expected": int(exp),
                     "delta": int(delta), "words": int(words),
                     "S": S, "Bkv": Bkv, "n_exp": nX})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    rows = scan()
    if args.list:
        for r in rows:
            print(r["case"])
        return
    if not rows:
        print("no smol_*.log found — run v2/smoke/run_softmax_online_rtl.py")
        return
    print(f"{'case':40s} {'S':>4} {'Bkv':>4} {'nX':>3} {'errs':>6} "
          f"{'meas':>7} {'exp':>7} {'delta':>6} {'words':>6} verdict")
    n_pass = 0
    for r in rows:
        ok = (r["verdict"] == "PASS" and r["errors"] == 0 and r["delta"] == 0
              and r["words"] == r["S"] * r["S"])
        n_pass += ok
        print(f"{r['case']:40s} {r['S']:>4} {r['Bkv']:>4} {r['n_exp']:>3} "
              f"{r['errors']:>6} {r['measured']:>7} {r['expected']:>7} "
              f"{r['delta']:>+6} {r['words']:>6} "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"test_softmax_online: {n_pass}/{len(rows)} PASS")


if __name__ == "__main__":
    main()
