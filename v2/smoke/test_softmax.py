#!/usr/bin/env python3
"""test_softmax.py — inspect the Stage-3 softmax RTL corpus (GATE-2 logs).

Reads the per-case TB logs written by `v2/smoke/run_softmax_rtl.py`
(`v2/smoke/logs/softmax_<case>.log`) and the case metadata from the stimuli
dirs, then prints the verification table.

Usage:
  python3 v2/smoke/test_softmax.py            # table over all logged cases
  python3 v2/smoke/test_softmax.py --list     # case names only
  python3 v2/smoke/test_softmax.py <case>     # full detail for one case
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "v2" / "smoke" / "logs"
STIMULI_DIRS = [REPO / "v2" / "smoke" / "softmax_stimuli",
                REPO / "v2" / "smoke" / "softmax_x2_stimuli"]

RESULT_RE = re.compile(r"\[tb_softmax_top\] (PASS|FAIL) (\S+) errors=(\d+) "
                       r"measured=(-?\d+) expected=(-?\d+) delta=(-?\d+) "
                       r"words=(\d+)")


def case_meta(name: str) -> dict:
    for root in STIMULI_DIRS:
        path = root / name / "case.json"
        if path.exists():
            return json.loads(path.read_text())
    return {}


SWEEP_LINE = re.compile(r"\s+(PASS|FAIL)\s+(\S+?):")
SWEEP_NUM = re.compile(r"(errors|measured|expected|delta)=(-?\d+)")


def scan_sweeps(rows: list[dict], seen: set) -> None:
    """Also fold in aggregate sweep logs (`sweep_*.log`) for the record."""
    for sweep in sorted(LOGS.glob("sweep_*.log")):
        for line in sweep.read_text().splitlines():
            m = SWEEP_LINE.match(line)
            if not m:
                continue
            name = m.group(2)
            if name in seen:
                continue
            nums = dict(SWEEP_NUM.findall(line))
            rows.append({
                "case": name, "log": sweep, "meta": case_meta(name),
                "verdict": m.group(1),
                "errors": int(nums.get("errors", 0)),
                "measured": int(nums.get("measured", -1)),
                "expected": int(nums.get("expected", -1)),
                "delta": int(nums.get("delta", -999999)),
                "words": case_meta(name).get("S", 0) ** 2,
                "sweep": True,
            })
            seen.add(name)


def scan() -> list[dict]:
    rows = []
    for log in sorted(LOGS.glob("softmax_*.log")):
        text = log.read_text()
        m = RESULT_RE.search(text)
        name = log.stem.replace("softmax_", "")
        meta = case_meta(name)
        row = {"case": name, "log": log, "meta": meta}
        if m:
            row.update(verdict=m.group(1), errors=int(m.group(3)),
                       measured=int(m.group(4)), expected=int(m.group(5)),
                       delta=int(m.group(6)), words=int(m.group(7)))
        rows.append(row)
    seen = {r["case"] for r in rows}
    scan_sweeps(rows, seen)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    rows = scan()
    if args.case:
        sel = [r for r in rows if r["case"] == args.case]
        if not sel:
            print(f"no log for case {args.case!r}")
            return
        r = sel[0]
        meta = r["meta"]
        print(f"case      : {r['case']}")
        print(f"geometry  : S={meta.get('S')} R={meta.get('R')} C={meta.get('C')} "
              f"n_exp={meta.get('n_exp')} n_log={meta.get('n_log')} "
              f"clb_width={meta.get('clb_width')}")
        print(f"passes    : exp={meta.get('passes_exp')} log={meta.get('passes_log')}")
        print(f"model cyc : {meta.get('used_cycles')} "
              f"(compute {meta.get('compute_cycles')}, drain {meta.get('drain_cycles')})")
        print(f"rtl cyc   : {r.get('measured')}  delta={r.get('delta')}  "
              f"errors={r.get('errors')}  words={r.get('words')}  "
              f"verdict={r.get('verdict')}")
        print("--- tb log ---")
        print(r["log"].read_text())
        return

    if args.list:
        for r in rows:
            print(r["case"])
        return

    if not rows:
        print("no logs found — run v2/smoke/run_softmax_rtl.py first")
        return
    print(f"{'case':38s} {'S':>4s} {'R x C':>9s} {'nX':>2s} {'nL':>2s} "
          f"{'model':>8s} {'rtl':>8s} {'delta':>5s} {'err':>4s} verdict")
    n_ok = 0
    for r in rows:
        m = r["meta"]
        ok = (r.get("verdict") == "PASS" and r.get("errors") == 0
              and r.get("delta") == 0)
        n_ok += 1 if ok else 0
        print(f"{r['case']:38s} {m.get('S', '?'):>4} "
              f"{str(m.get('R'))+'x'+str(m.get('C')):>9s} "
              f"{m.get('n_exp', '?'):>2} {m.get('n_log', '?'):>2} "
              f"{r.get('expected', -1):>8} {r.get('measured', -1):>8} "
              f"{r.get('delta', '?'):>5} {r.get('errors', '?'):>4} "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"test_softmax: {n_ok}/{len(rows)} PASS")


if __name__ == "__main__":
    main()
