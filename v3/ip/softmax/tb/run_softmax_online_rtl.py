#!/usr/bin/env python3
"""run_softmax_online_rtl.py — GATE-2 harness for the v2 online softmax RTL.

Pipeline: case (certified by `gen_softmax_online_cases.py`) ->
`softmax_online_top` under `tb_softmax_online_top.v`. Gates per case: every
output-port event (log|P|, log_a, log_l), the final carry and the stage probes
(block max, sp, carry EXP byte, lq) bit-exact, and measured (start -> done) ==
case.json `expected_cycles` (strict, Delta_impl = 0; one value per run for
back-to-back cases). Gap cases (case.json OFFER = 1, PRE-1) run with +OFFER=1: the TB
drives each score word from its offer edge on; their expected cycles come from the
availability model.

Usage:
  python3 v2/sim/smoke/run_softmax_online_rtl.py [--only <substr>] [--quick]
        [--stimuli DIR] [--work DIR] [--jobs N] [--timeout SEC]
Logs: v2/sim/smoke/logs/smol_<case>.log (per case) + summary.json in the work dir.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # v3 path (tools/copies.py)
RTL_FILES = [REPO / "v3" / "ip" / "xbar" / "rtl" / "sim" / "dpe_nldpe.v",
             REPO / "v3" / "ip" / "softmax" / "rtl" / "sim" / "softmax_online_top.v"]  # v3 path (tools/copies.py)
TB_FILE = REPO / "v3" / "ip" / "softmax" / "tb" / "tb_softmax_online_top.v"  # v3 path (tools/copies.py)
LOGS = REPO / "v3" / "work" / "softmax" / "logs"  # v3 path (tools/copies.py)
LOCK = REPO / "v3" / "work" / "softmax" / ".softmax_online_rtl.lock"  # v3 path (tools/copies.py)

RESULT_RE = re.compile(r"\[tb_softmax_online\] (PASS|FAIL) \S+ errors=(\d+) measured=(-?\d+) stall=(\d+)")
RUN_RE = re.compile(r"\[tb_softmax_online\] RUN \d+ errors=\d+ measured=(-?\d+)")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def compile_cfg(cfg: tuple, work: Path) -> Path:
    S, BKV, NE, R, C, N = cfg
    vvp = work / f"smol_S{S}_N{N}_B{BKV}_E{NE}_{R}x{C}.vvp"
    if not vvp.exists():
        cmd = ["iverilog", "-g2005", "-o", str(vvp),
               f"-DS_TB={S}", f"-DBKV_TB={BKV}", f"-DNE_TB={NE}", f"-DR_TB={R}", f"-DC_TB={C}",
               f"-DN_TB={N}", str(TB_FILE), *[str(f) for f in RTL_FILES]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compile failed {cfg}:\n{r.stdout}{r.stderr}")
    return vvp


def run_case(case_dir: Path, vvp: Path, work: Path, timeout: float) -> dict:
    case = json.loads((case_dir / "case.json").read_text())
    # short relative path for $readmemh (long absolute paths overflow the TB string)
    vdir = os.path.relpath(case_dir, work)
    runs = case.get("RUNS", 0)
    t0 = time.time()
    try:
        r = subprocess.run(["vvp", "-n", str(vvp), f"+VDIR={vdir}", f"+CASE={case_dir.name}",
                            *([f"+RUNS={runs}"] if runs else []),
                            *(["+OFFER=1"] if case.get("OFFER") else [])],
                           cwd=work, capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = "TIMEOUT"
    (LOGS / f"smol_{case_dir.name}.log").write_text(out)
    m = RESULT_RE.search(out)
    expected = case.get("expected_cycles")
    res = dict(case=case_dir.name, wall=round(time.time() - t0, 1),
               verdict=m.group(1) if m else "ERROR",
               errors=int(m.group(2)) if m else -1,
               measured=int(m.group(3)) if m else -1,
               stall=int(m.group(4)) if m else -1,
               runs_measured=[int(x) for x in RUN_RE.findall(out)], expected=expected)
    mm = re.search(r"mismatches: (.*)", out)
    res["detail"] = mm.group(1) if mm else ("" if m else out[-400:])
    ok = res["verdict"] == "PASS" and res["errors"] == 0
    if isinstance(expected, list):
        ok = ok and res["runs_measured"] == expected
    else:
        ok = ok and expected is not None and res["measured"] == expected
    res["ok"] = ok
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stimuli", default=str(REPO / "v3" / "work" / "softmax" / "softmax_online_stimuli"))  # v3 path (tools/copies.py)
    ap.add_argument("--work", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--quick", action="store_true", help="first case only")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    if LOCK.exists():
        print(f"refusing: lock exists ({rel(LOCK)}) — another run in flight")
        return 2
    stim = Path(a.stimuli).resolve()
    names = [n for n in (stim / "manifest.txt").read_text().split() if n]
    if a.only:
        names = [n for n in names if a.only in n]
    if a.quick:
        names = names[:1]
    if not names:
        print("no cases selected")
        return 2
    work = Path(a.work).resolve() if a.work else Path(tempfile.mkdtemp(prefix="smol_rtl_"))
    work.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(work))
    t0 = time.time()
    try:
        cfgs = {}
        for n in names:
            c = json.loads((stim / n / "case.json").read_text())
            cfgs[n] = (c["S"], c["BKV"], c["N_EXP"], c["R"], c["C"], c.get("N", c["S"]))
        vvps = {cfg: compile_cfg(cfg, work) for cfg in sorted(set(cfgs.values()))}
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            res = list(ex.map(lambda n: run_case(stim / n, vvps[cfgs[n]], work, a.timeout), names))
    finally:
        LOCK.unlink(missing_ok=True)
    for r in res:
        e = r["expected"]
        exp = (f" runs={r['runs_measured']} expected={e}" if isinstance(e, list)
               else f" expected={e} delta={r['measured'] - e}" if e is not None else "")
        print(f"  {'PASS' if r['ok'] else 'FAIL'} {r['case']}: measured={r['measured']}{exp} "
              f"stall={r['stall']}" + ("" if r["ok"] else f"  [{r['verdict']} errors={r['errors']}] {r['detail']}"))
    (work / "summary.json").write_text(json.dumps(res, indent=1))
    npass = sum(r["ok"] for r in res)
    print(f"run_softmax_online_rtl: {npass}/{len(res)} PASS ({time.time() - t0:.1f}s, work={rel(work)})")
    return 0 if npass == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
