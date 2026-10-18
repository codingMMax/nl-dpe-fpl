#!/usr/bin/env python3
"""run_ocarry_rtl.py — strict GATE-2 harness for the o-carry station (`att_ocarry_top`).

Per case (certified by `gen_ocarry_cases.py`): every LOG drain word (lo) and every O8 word bit-exact (the TB), the
TB's invariants clear, and every event cycle == the model in case.json (Δ_impl = 0): op_free and la_free per block,
the first / last O8 word per wave, run_done per run, the drain end of every LOG job.

Usage: python3 v2/sim/smoke/run_ocarry_rtl.py [--only SUBSTR] [--stimuli DIR] [--jobs N] [--timeout SEC]
Logs: v2/sim/smoke/logs/oc_<case>.log; summary.json in the work dir.
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
RTL = [REPO / "v3" / "ip" / "xbar" / "rtl" / "sim" / "dpe_nldpe.v", REPO / "v3" / "ip" / "lib" / "rtl" / "sim" / "att_fifo.v",
       REPO / "v3" / "ip" / "softmax" / "rtl" / "sim" / "att_ocarry_top.v"]  # v3 path (tools/copies.py)
TB = REPO / "v3" / "ip" / "softmax" / "tb" / "tb_att_ocarry_top.v"  # v3 path (tools/copies.py)
LOGS = REPO / "v3" / "work" / "softmax" / "logs"  # v3 path (tools/copies.py)
EV_RE = re.compile(r"^EV (\w+) ((?:-?\d+ ?)+)$", re.M)
RES_RE = re.compile(r"\[tb_ocarry\] (PASS|FAIL) \S+ errors=(\d+)")


def cfg_of(c: dict) -> tuple:
    return (c["S"], c["d"], c["R"], c["C"], c["N_OC"], c["NBLK"], c["SO"], c["LN"], c["NRUN"], c["NGP"], c["VWP"])


def compile_cfg(cfg: tuple, work: Path) -> Path:
    keys = ("S", "D", "R", "C", "NOC", "NBLK", "SO", "LN", "NRUN", "NGP", "VWP")
    vvp = work / ("oc_" + "_".join(str(x) for x in cfg) + ".vvp")
    if not vvp.exists():
        cmd = ["iverilog", "-g2005", "-o", str(vvp), *[f"-D{k}_TB={v}" for k, v in zip(keys, cfg)],
               str(TB), *map(str, RTL)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compile failed {cfg}:\n{r.stdout}{r.stderr}")
    return vvp


def expected_events(c: dict) -> dict:
    ev = {}
    for r in range(c["NRUN"]):
        for b, x in enumerate(c["exp_op_free"][r]):
            ev[("opfree", r, b)] = x
        for b, x in enumerate(c["exp_la_free"][r]):
            if b >= 1:
                ev[("lafree", r, b)] = x
        for w, x in enumerate(c["exp_o8_first"][r]):
            ev[("o8first", r, w)] = x
        for w, x in enumerate(c["exp_o8_last"][r]):
            ev[("o8last", r, w)] = x
        ev[("rundone", r)] = c["exp_run_done"][r]
        ev[("ofed", r)] = c["o_fed"][r]                # the TB's O feed release (sched.txt), echoed
        for j, x in enumerate(c["exp_log_drain_end"][r]):
            ev[("logend", r, j)] = x
    return ev


def run_case(d: Path, vvp: Path, work: Path, timeout: float) -> dict:
    c = json.loads((d / "case.json").read_text())
    args = ["vvp", "-n", str(vvp), f"+VDIR={os.path.relpath(d, work)}", f"+CASE={d.name}"]
    if c.get("MIDRESET"):
        args.append(f"+MIDRESET={c['MIDRESET']}")
    args.append(f"+PACE={c['WAVE_PACE']}")
    t0 = time.time()
    try:
        r = subprocess.run(args, cwd=work, capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = "TIMEOUT"
    (LOGS / f"oc_{d.name}.log").write_text(out)
    got = {}
    for name, nums in EV_RE.findall(out):
        v = [int(x) for x in nums.split()]
        got[(name, *v[:-1])] = v[-1]
    exp = expected_events(c)
    diff = {k: (got.get(k), x) for k, x in exp.items() if got.get(k) != x}
    extra = [k for k in got if k not in exp]
    m = RES_RE.search(out)
    ok = bool(m) and m.group(1) == "PASS" and not diff and not extra
    delta = sorted({g - x for (g, x) in diff.values() if g is not None})
    return dict(case=d.name, ok=ok, verdict=m.group(1) if m else "ERROR", n_events=len(exp),
                mismatches=len(diff), extra=len(extra), deltas=delta[:6],
                first=sorted(diff.items())[:4] if diff else [], wall=round(time.time() - t0, 1),
                tail="" if ok else out[-800:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stimuli", default=str(REPO / "v3" / "work" / "softmax" / "ocarry_stimuli"))  # v3 path (tools/copies.py)
    ap.add_argument("--only", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    stim = Path(a.stimuli).resolve()
    names = [n for n in (stim / "manifest.txt").read_text().split() if n and (not a.only or a.only in n)]
    if not names:
        print("no cases selected")
        return 2
    work = Path(a.work).resolve() if a.work else Path(tempfile.mkdtemp(prefix="oc_rtl_"))
    work.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfgs = {n: cfg_of(json.loads((stim / n / "case.json").read_text())) for n in names}
    vvps = {cfg: compile_cfg(cfg, work) for cfg in sorted(set(cfgs.values()))}
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(lambda n: run_case(stim / n, vvps[cfgs[n]], work, a.timeout), names))
    for r in res:
        print(f"  {'PASS' if r['ok'] else 'FAIL'} {r['case']}: events {r['n_events']}, mismatches {r['mismatches']}"
              + (f", extra {r['extra']}" if r["extra"] else "")
              + ("" if r["ok"] else f" [{r['verdict']}] deltas={r['deltas']} first={r['first']}\n{r['tail']}"))
    (work / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    npass = sum(r["ok"] for r in res)
    print(f"run_ocarry_rtl: {npass}/{len(res)} PASS ({time.time() - t0:.1f}s, work={work})")
    return 0 if npass == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
