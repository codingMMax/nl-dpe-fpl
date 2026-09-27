#!/usr/bin/env python3
"""run_softmax_rtl.py — Stage-3 GATE-2 harness for the v2 softmax RTL.

Pipeline: case (certified by `gen_softmax_cases.py`) -> `softmax_top` under
`tb_softmax_top.v` -> dual compare + cycle delta.

Strict gates per case:
  * all six probe stages and the output stream bit-exact (errors == 0);
  * measured cycles (start -> done) == case.json `used_cycles` (delta == 0);
  * the output stream carries exactly S*S words.

Usage:
  python3 v2/smoke/run_softmax_rtl.py [--only <substr>] [--quick]
        [--stimuli DIR] [--work DIR] [--timeout SEC]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RTL_FILES = [REPO / "v2" / "rtl" / "dpe_nldpe.v",
             REPO / "v2" / "rtl" / "softmax_top.v"]
TB_FILE = REPO / "v2" / "tb" / "tb_softmax_top.v"
LOCK = REPO / "v2" / "smoke" / ".softmax_rtl.lock"

VECTOR_MAP = {
    "scores.mem": "scores.hex",
    "expected_rowmax.mem": "exrm.hex",
    "expected_expin.mem": "exin.hex",
    "expected_expout.mem": "exout.hex",
    "expected_sum.mem": "exsum.hex",
    "expected_lq.mem": "exlq.hex",
    "expected_logout.mem": "exlg.hex",
    "expected_out.mem": "expout.hex",
}

RESULT_RE = re.compile(r"\[tb_softmax_top\] (PASS|FAIL) (\S+) errors=(\d+) "
                       r"measured=(-?\d+) expected=(-?\d+) delta=(-?\d+) "
                       r"words=(\d+)")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def expand_vectors(case_dir: Path, vec_dir: Path) -> None:
    vec_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in VECTOR_MAP.items():
        shutil.copyfile(case_dir / src, vec_dir / dst)


def compile_case(case: dict, work: Path) -> Path:
    tag = (f"softmax_S{case['S']}_nX{case['n_exp']}_nL{case['n_log']}"
           f"_{case['R']}x{case['C']}")
    vvp = work / f"{tag}.vvp"
    if vvp.exists():
        return vvp
    cmd = ["iverilog", "-g2005", "-o", str(vvp),
           f"-DS_TB={case['S']}", f"-DR_TB={case['R']}",
           f"-DC_TB={case['C']}", f"-DBUF_TB={case['BUF']}",
           f"-DP_TB={case['P']}", f"-DNE_TB={case['n_exp']}",
           f"-DNL_TB={case['n_log']}",
           str(TB_FILE), *[str(f) for f in RTL_FILES]]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"compile failed for {tag}")
    return vvp


def run_case(case_dir: Path, case: dict, vvp: Path, work: Path,
             timeout: int) -> dict:
    vec_dir = work / case_dir.name / "vectors"
    expand_vectors(case_dir, vec_dir)
    cmd = ["vvp", str(vvp), f"+VDIR={vec_dir}", f"+CASE={case_dir.name}",
           f"+EXPECTED={case['used_cycles']}"]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=timeout, cwd=str(work))
    out = res.stdout + res.stderr
    m = RESULT_RE.search(out)
    if not m:
        return {"case": case_dir.name, "ok": False,
                "detail": "no TB result line", "log": out}
    verdict, name, errors, measured, expected, delta, words = m.groups()
    ok = (verdict == "PASS" and int(errors) == 0 and int(delta) == 0
          and int(words) == case["S"] * case["S"])
    return {"case": case_dir.name, "ok": ok, "errors": int(errors),
            "measured": int(measured), "expected": int(expected),
            "delta": int(delta), "words": int(words), "log": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli",
                    default=str(REPO / "v2" / "smoke" / "softmax_stimuli"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="run only the first case")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    if LOCK.exists():
        print(f"refusing: lock exists ({rel(LOCK)}) — another run in flight")
        sys.exit(2)
    stimuli = Path(args.stimuli)
    manifest = stimuli / "manifest.txt"
    if not manifest.exists():
        print(f"missing manifest: {rel(manifest)} — run gen_softmax_cases.py")
        sys.exit(2)
    names = [n for n in manifest.read_text().split() if n]
    if args.only:
        names = [n for n in names if args.only in n]
    if args.quick:
        names = names[:1]
    if not names:
        print("no cases selected")
        sys.exit(2)

    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(
        prefix="softmax_rtl_"))
    work.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(work))
    results = []
    t0 = time.time()
    try:
        for name in names:
            case_dir = stimuli / name
            case = json.loads((case_dir / "case.json").read_text())
            vvp = compile_case(case, work)
            res = run_case(case_dir, case, vvp, work, args.timeout)
            results.append(res)
            status = "PASS" if res["ok"] else "FAIL"
            if res["ok"]:
                print(f"  {status} {name}: measured={res['measured']} "
                      f"expected={res['expected']} delta={res['delta']}")
            else:
                print(f"  {status} {name}: {res.get('detail', '')} "
                      f"errors={res.get('errors')} measured="
                      f"{res.get('measured')} expected={res.get('expected')} "
                      f"delta={res.get('delta')}")
                print(res.get("log", ""))
    finally:
        LOCK.unlink(missing_ok=True)

    n_pass = sum(1 for r in results if r["ok"])
    print(f"run_softmax_rtl: {n_pass}/{len(results)} PASS "
          f"({time.time() - t0:.1f}s, work={rel(work)})")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
