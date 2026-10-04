#!/usr/bin/env python3
"""run_softmax_online_rtl.py — GATE-2 harness for the v2 online softmax RTL.

Pipeline: case (certified by `gen_softmax_online_cases.py`) ->
`softmax_online_top` under `tb_softmax_online_top.v` -> dual compare + cycle
delta. Gates: all probes + output stream bit-exact; measured (start->done) ==
case.json `compute_cycles` (delta == 0); S*S output words.

Usage:
  python3 v2/smoke/run_softmax_online_rtl.py [--only <substr>] [--quick]
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
             REPO / "v2" / "rtl" / "softmax_top.v",
             REPO / "v2" / "rtl" / "softmax_online_top.v"]
TB_FILE = REPO / "v2" / "tb" / "tb_softmax_online_top.v"
LOCK = REPO / "v2" / "smoke" / ".softmax_online_rtl.lock"

VECTOR_MAP = {
    "scores.mem": "scores.hex",
    "expected_blkmax.mem": "exblkmax.hex",
    "expected_blksp.mem": "exblksp.hex",
    "expected_factor.mem": "exfactor.hex",
    "expected_L.mem": "exL.hex",
    "expected_lq.mem": "exlq.hex",
    "expected_ls.mem": "exls.hex",
    "expected_out.mem": "exout.hex",
}

RESULT_RE = re.compile(r"\[tb_softmax_online\] (PASS|FAIL) (\S+) errors=(\d+) "
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
    tag = (f"smol_S{case['S']}_B{case['Bkv']}_nX{case['n_exp']}"
           f"_{case['R']}x{case['C']}")
    vvp = work / f"{tag}.vvp"
    if vvp.exists():
        vvp.unlink()  # geometry never recompiles from cache across edits
    cmd = ["iverilog", "-g2005", "-o", str(vvp),
           f"-DS_TB={case['S']}", f"-DR_TB={case['R']}",
           f"-DC_TB={case['C']}", f"-DBUF_TB={case['BUF']}",
           f"-DP_TB={case['P']}", f"-DBKV_TB={case['Bkv']}",
           f"-DNE_TB={case['n_exp']}", f"-DNL_TB={case['n_log']}",
           f"-DNF_TB={case.get('n_fac', case['n_exp'])}",
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
    expected = case.get("compute_cycles", case["used_cycles"])
    cmd = ["vvp", str(vvp), f"+VDIR={vec_dir}", f"+CASE={case_dir.name}",
           f"+EXPECTED={expected}"]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=timeout, cwd=str(work))
    out = res.stdout + res.stderr
    m = RESULT_RE.search(out)
    if not m:
        return {"case": case_dir.name, "ok": False,
                "detail": "no TB result line", "log": out}
    verdict, name, errors, measured, expected_s, delta, words = m.groups()
    ok = (verdict == "PASS" and int(errors) == 0 and int(delta) == 0
          and int(words) == case["S"] * case["S"])
    return {"case": case_dir.name, "ok": ok, "errors": int(errors),
            "measured": int(measured), "expected": int(expected_s),
            "delta": int(delta), "words": int(words), "log": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli",
                    default=str(REPO / "v2" / "smoke" /
                                "softmax_online_stimuli"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    if LOCK.exists():
        print(f"refusing: lock exists ({rel(LOCK)})")
        sys.exit(2)
    stimuli = Path(args.stimuli)
    manifest = stimuli / "manifest.txt"
    if not manifest.exists():
        print(f"missing manifest: {rel(manifest)}")
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
        prefix="smol_rtl_"))
    work.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(work))
    results = []
    t0 = time.time()
    logs_dir = REPO / "v2" / "smoke" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in names:
            case_dir = stimuli / name
            case = json.loads((case_dir / "case.json").read_text())
            vvp = compile_case(case, work)
            res = run_case(case_dir, case, vvp, work, args.timeout)
            results.append(res)
            (logs_dir / f"smol_{name}.log").write_text(res.get("log", ""))
            status = "PASS" if res["ok"] else "FAIL"
            if res["ok"]:
                print(f"  {status} {name}: measured={res['measured']} "
                      f"expected={res['expected']} delta={res['delta']}")
            else:
                print(f"  {status} {name}: {res.get('detail', '')} "
                      f"errors={res.get('errors')} measured="
                      f"{res.get('measured')} expected={res.get('expected')} "
                      f"delta={res.get('delta')}")
    finally:
        LOCK.unlink(missing_ok=True)

    n_pass = sum(1 for r in results if r["ok"])
    print(f"run_softmax_online_rtl: {n_pass}/{len(results)} PASS "
          f"({time.time() - t0:.1f}s, work={rel(work)})")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
