#!/usr/bin/env python3
"""rtl_gate2.py — the GATE-2 runner of every module (docs/architecture.md §6, §7).

A corpus is a folder of case folders written by ip/<module>/cases.py. A case folder holds the testbench's files,
events.txt (the model's port events) and case.json (the iverilog defines, the testbench's plusargs, the model's total
cycles, and the words per window of every port). Each case runs the module's testbench on the simulation twin
(iverilog) and passes when:
  1 the testbench prints its PASS line (its value, protocol and count checks),
  2 every event line equals the model's (sorted by cycle; the event lines fix the cycle of every word), and
  3 the total cycles, last word of any window + 1, are equal (computed from the RTL's own event lines).
Each module's ip/<module>/tb/run_*_rtl.py names its testbench, its RTL files and its PASS mark, and calls main().

Usage (through a module's runner):
  run_*_rtl.py --stimuli DIR [--only SUBSTR] [--jobs N] [--work DIR]   run the cases; logs <work>/<case>.log
  run_*_rtl.py --stimuli DIR --recheck --work DIR                      re-check existing logs, no simulation
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EV_RE = re.compile(r"^EV (\d+ .*)$", re.M)


def _key(line: str) -> tuple:
    t, rest = line.split(" ", 1)
    return int(t), rest


def cycles_of(lines: list, port_words: dict) -> int:
    """The last word of any window + 1, from event lines `<t> <inst.port> win ...`."""
    end = -1
    for ln in lines:
        t, key, kind = ln.split()[:3]
        if kind == "win":
            end = max(end, int(t) + port_words[key] - 1)
    return end + 1


def check(case_dir: Path, out: str, pass_mark: str) -> dict:
    """A testbench output against the case's model results."""
    cj = json.loads((case_dir / "case.json").read_text())
    got = sorted(EV_RE.findall(out), key=_key)
    want = sorted((ln for ln in (case_dir / "events.txt").read_text().splitlines() if ln), key=_key)
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    if first is None and len(got) != len(want):
        first = min(len(got), len(want))
    res = dict(case=case_dir.name, tb_pass=pass_mark in out, events=len(got), events_model=len(want),
               cycles=cycles_of(got, cj["port_words"]) if got else None, cycles_model=cj["cycles"])
    res["ok"] = res["tb_pass"] and first is None and res["cycles"] == res["cycles_model"]
    if first is not None:
        res["first_diff"] = dict(index=first, rtl=got[first] if first < len(got) else None,
                                 model=want[first] if first < len(want) else None)
    if not res["tb_pass"]:
        res["tail"] = "\n".join(ln for ln in out.splitlines() if "ERROR" in ln or "FAIL" in ln)[:1500]
    return res


def self_test(tmp: Path) -> None:
    """check() on a made-up case: equal output passes; a moved event, a missing PASS line, a wrong cycle count fail."""
    d = tmp / "rtl_gate2_selftest"
    d.mkdir(parents=True, exist_ok=True)
    ev = ["0 src.a win ch0 u0 w0", "5 m.a cr u0", "9 m.c win ch0 u0 w0"]
    (d / "events.txt").write_text("".join(x + "\n" for x in ev))
    (d / "case.json").write_text(json.dumps(dict(cycles=12, port_words={"src.a": 3, "m.c": 3})))
    out = "".join(f"EV {x}\n" for x in ev) + "[tb] PASS\n"
    assert check(d, out, "[tb] PASS")["ok"]
    moved = check(d, out.replace("EV 5 m.a", "EV 6 m.a"), "[tb] PASS")
    assert not moved["ok"] and moved["first_diff"]["index"] == 1
    assert not check(d, out.replace("PASS", "FAIL"), "[tb] PASS")["ok"]
    (d / "case.json").write_text(json.dumps(dict(cycles=12, port_words={"src.a": 3, "m.c": 4})))   # a longer window
    wrong = check(d, out, "[tb] PASS")
    assert not wrong["ok"] and wrong["cycles"] == 13


def compile_tb(tb: Path, rtl: list, top: str, defines: dict, work: Path) -> Path:
    """One iverilog build per set of defines (cached in the work folder)."""
    vvp = work / ("tb_" + "_".join(f"{k[:-3]}{v}" for k, v in sorted(defines.items())) + ".vvp")
    if not vvp.exists():
        cmd = ["iverilog", "-g2005", "-o", str(vvp), *[f"-D{k}={v}" for k, v in defines.items()], "-s", top,
               str(tb), *map(str, rtl)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compile failed:\n{r.stdout}{r.stderr}")
    return vvp


def run_one(case_dir: Path, vvp: Path, work: Path, timeout: float, pass_mark: str) -> dict:
    cj = json.loads((case_dir / "case.json").read_text())
    args = ["vvp", "-n", str(vvp), f"+VDIR={case_dir}", f"+CASE={case_dir.name}",
            *[f"+{k}={v}" for k, v in cj["plusargs"].items()]]
    t0 = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = "TIMEOUT"
    (work / f"{case_dir.name}.log").write_text(out)
    return dict(check(case_dir, out, pass_mark), wall=round(time.time() - t0, 1))


def main(tb: Path, rtl: list, top: str, pass_mark: str, name: str) -> int:
    ap = argparse.ArgumentParser(description=f"GATE 2 of {name}")
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--work", default=None)
    ap.add_argument("--recheck", action="store_true", help="check the logs already in --work, no simulation")
    a = ap.parse_args()
    stim = Path(a.stimuli).resolve()
    names = [n for n in (stim / "manifest.txt").read_text().split() if n and (not a.only or a.only in n)]
    if not names:
        print("no cases selected")
        return 2
    work = Path(a.work).resolve() if a.work else stim.parent / (stim.name + "_rtl")
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.recheck:
        res = [dict(check(stim / n, (work / f"{n}.log").read_text(), pass_mark), wall=0.0) for n in names]
    else:
        vvps = {}
        for n in names:
            defs = json.loads((stim / n / "case.json").read_text())["defines"]
            key = tuple(sorted(defs.items()))
            if key not in vvps:
                vvps[key] = compile_tb(tb, rtl, top, defs, work)
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            res = list(ex.map(lambda n: run_one(stim / n, vvps[tuple(sorted(json.loads(
                (stim / n / "case.json").read_text())["defines"].items()))], work, a.timeout, pass_mark), names))
    for r in res:
        line = (f"  {'PASS' if r['ok'] else 'FAIL'} {r['case']}: {r['events']} events (model {r['events_model']}), "
                f"cycles {r['cycles']} (model {r['cycles_model']}), {r['wall']}s")
        if "first_diff" in r:
            line += (f"\n      first difference #{r['first_diff']['index']}: rtl {r['first_diff']['rtl']} | "
                     f"model {r['first_diff']['model']}")
        if "tail" in r:
            line += "\n      " + r["tail"].replace("\n", "\n      ")
        print(line)
    (work / ("recheck.json" if a.recheck else "summary.json")).write_text(json.dumps(res, indent=1))
    npass = sum(r["ok"] for r in res)
    print(f"{name}: {npass}/{len(res)} PASS{' (recheck of existing logs)' if a.recheck else ''} "
          f"({time.time() - t0:.0f}s, work {work})")
    return 0 if npass == len(res) else 1


if __name__ == "__main__":
    self_test(Path(__file__).resolve().parents[1] / "work" / "tmp")
    print("rtl_gate2 self-test: ALL PASS")
