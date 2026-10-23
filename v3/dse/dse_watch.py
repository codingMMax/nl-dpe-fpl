#!/usr/bin/env python3
"""dse_watch.py — the background watcher of the VTR DSE (vtr_dse.py): every --every minutes (default 90) it appends a
status report to work/dse/vtr_dse_status.log (follow it with `tail -f v3/work/dse/vtr_dse_status.log`), and a final
report with every build's result when the driver exits. It only reads; it never starts, stops or changes a run.

A report has: what finished (builds with all three seeds counted, syntheses, VPR seeds; the latest builds' Fmax, CLB,
memory, crossbars), what runs now (syntheses in Parmys or in ABC, VPRs: from the driver's process tree), what waits,
the failures (from the run log since the driver started), memory-gate waits, the machine (memory available, our
processes' memory, disk), and an end estimate for each running ABC from its netlist size [estimate]:
ABC hours = 3.58 x (netlist MB / 99.4)^2.24 (fitted on the head smoke, 99.4 MB -> 3.58 h, and the driver test's
JetFormer build, 53.0 MB -> 0.875 h), with the finished syntheses' actual / predicted ratio beside it.
Usage: python3 v3/dse/dse_watch.py --pid DRIVER_PID [--every 90] [--status PATH] [--once]
       python3 v3/dse/dse_watch.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(HERE))
import vtr_dse as D  # noqa: E402

STATUS = V3 / "work" / "dse" / "vtr_dse_status.log"
ABC_REF_MB, ABC_REF_H, ABC_EXP = 99.4, 3.58, 2.24
STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
ELAPSED = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)")


def abc_hours(mb: float) -> float:
    return ABC_REF_H * (mb / ABC_REF_MB) ** ABC_EXP


def _hms(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min" if sec >= 3600 else f"{sec // 60} min"


def _wall(text: str):
    m = ELAPSED.search(text)
    if not m:
        return None
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(m.group(1).split(":"))))


def start_time(pid: int):
    """The process's start as a Unix time (from /proc), None if it is gone."""
    try:
        ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        boot = next(float(x.split()[1]) for x in open("/proc/stat") if x.startswith("btime"))
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, StopIteration, ValueError):
        return None


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def process_tree(root: int) -> list:
    """[(pid, comm, rss GiB, cwd, age s)] of every descendant of root."""
    procs, now = {}, time.time()
    boot = next(float(x.split()[1]) for x in open("/proc/stat") if x.startswith("btime"))
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            head, tail = (d / "stat").read_text().rsplit(")", 1)
            f = tail.split()
            rss = next((int(x.split()[1]) for x in (d / "status").read_text().splitlines()
                        if x.startswith("VmRSS:")), 0) / 2 ** 20
            try:
                cwd = os.readlink(d / "cwd")
            except OSError:
                cwd = ""
            procs[int(d.name)] = (int(f[1]), head.split("(", 1)[1], rss, cwd,
                                  now - (boot + int(f[19]) / os.sysconf("SC_CLK_TCK")))
        except (OSError, IndexError, ValueError):
            continue
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        for q, (pp, comm, rss, cwd, age) in procs.items():
            if pp == p:
                out.append((q, comm, rss, cwd, age))
                todo.append(q)
    return out


def run_events(log_lines: list) -> dict:
    """Counts and lists from the driver's run-log lines (one run): synthesis starts / OK / failed / cached, VPR starts /
    OK / failed / skipped / not counted, errors, elaboration failures, memory-gate waits, the exit line."""
    ev = dict(synth_start=[], synth_ok=[], synth_fail=[], synth_cached=[], vpr_start=[], vpr_ok=[], vpr_fail=[],
              vpr_skip=[], not_counted=[], errors=[], elab_fail=[], gate=[], exit=None)
    for ln in log_lines:
        m = STAMP.match(ln)
        if not m:
            continue
        msg = m.group(2)
        if " synth start (pid " in msg:
            ev["synth_start"].append(msg.split(" synth start")[0])
        elif " synth: OK (" in msg:
            ev["synth_ok"].append((msg.split(" synth:")[0], float(msg.split(" synth: OK (")[1].split(" s")[0])))
        elif " synth: FAILED" in msg:
            ev["synth_fail"].append(msg)
        elif " synth: cached" in msg:
            ev["synth_cached"].append(msg.split(" synth:")[0])
        elif ": VPR start (pid " in msg:
            ev["vpr_start"].append(msg.split(": VPR start")[0])
        elif ": VPR OK (" in msg:
            ev["vpr_ok"].append(msg)
            if "NOT COUNTED" in msg:
                ev["not_counted"].append(msg)
        elif ": VPR failed" in msg:
            ev["vpr_fail"].append(msg)
        elif ": finished, skipped" in msg:
            ev["vpr_skip"].append(msg)
        elif msg.startswith("ERROR "):
            ev["errors"].append(msg)
        elif " elab: FAILED" in msg or ("checked again: FAILED" in msg):
            ev["elab_fail"].append(msg)
        elif "at the memory gate" in msg:
            ev["gate"].append(msg)
        elif msg.startswith("=== wrote "):
            ev["exit"] = msg
    return ev


def _since(path: Path, t0: float) -> list:
    """The run-log lines stamped at or after t0 (the driver's start)."""
    lim = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0 - 1))
    return [ln for ln in D._read(path).splitlines() if STAMP.match(ln) and ln[:19] >= lim]


def _size_gb(dirs) -> float:
    tot = 0
    for d in dirs:
        for r, _, fs in os.walk(d):
            for f in fs:
                try:
                    tot += os.lstat(os.path.join(r, f)).st_size
                except OSError:
                    pass
    return tot / 1e9


def report(pid: int, t0: float, n: int, final: bool, csv_path: Path = D.CSV_PATH, run_log: Path = D.RUN_LOG) -> str:
    builds = D.all_builds()
    by_dir = {b.dir.name: b for b in builds}
    ev = run_events(_since(run_log, t0))
    L = [f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} | VTR DSE status, "
         + ("FINAL report" if final else f"report {n}") + f" ({_hms(time.time() - t0)} after launch) ===="]
    L.append(f"driver: {'running' if alive(pid) else 'ended'}, pid {pid}; run log {D._rel(run_log)}; "
             f"every event: tail -f {D._rel(run_log)}")
    rows = {}
    if csv_path.is_file():
        with open(csv_path, newline="") as f:
            rows = {r["build"]: r for r in csv.DictReader(f)}
    cur = [rows.get(b.name, {}) for b in builds]
    cur = [r for r, b in zip(cur, builds) if r.get("hash") == b.digest[:12]]       # lines of the current circuit
    complete = [r for r in cur if r.get("ok") == "1" and r.get("seeds") == "11;13;17"]
    synth_done = len(ev["synth_ok"]) + len(ev["synth_cached"])
    vpr_done = len(ev["vpr_ok"]) + len(ev["vpr_fail"])
    L.append(f"done: {len(complete)} / {len(builds)} builds complete (3 seeds counted) | syntheses {synth_done} / "
             f"{len(builds)} | VPR seeds {vpr_done + len(ev['vpr_skip'])} / {3 * len(builds)} (this run {vpr_done}, "
             f"skipped as finished "
             f"{len(ev['vpr_skip'])}, not counted {len(ev['not_counted'])})")
    tree = process_tree(pid) if alive(pid) else []

    def build_of(cwd):
        parts = Path(cwd).parts
        return next((by_dir[p] for p in parts if p in by_dir), None)
    parmys = [(build_of(c), age) for (_, comm, _, c, age) in tree if comm == "yosys" and c.endswith("/synth")]
    abc = [(build_of(c), age) for (_, comm, _, c, age) in tree if comm == "abc"]
    vpr = [(build_of(c), age) for (_, comm, _, c, age) in tree if comm == "vpr"]
    ours = sum(rss for (_, _, rss, _, _) in tree)
    wait_syn = len(builds) - len(ev["synth_start"]) - len(ev["synth_cached"])
    queued = 3 * synth_done - len(ev["vpr_skip"])
    L.append(f"running: {len(parmys) + len(abc)} syntheses ({len(parmys)} in Parmys, {len(abc)} in ABC), {len(vpr)} VPR"
             f" | waiting: {max(0, wait_syn)} syntheses, {max(0, queued - len(ev['vpr_start']))} VPR seeds")
    fails = ev["synth_fail"] + ev["vpr_fail"] + ev["not_counted"] + ev["errors"] + ev["elab_fail"]
    L.append("failures: " + ("none" if not fails else f"{len(fails)}"))
    L += [f"  {x[:200]}" for x in fails[-10:]]
    if ev["gate"]:
        L.append(f"memory-gate waits over 60 s so far: {len(ev['gate'])}; latest: {ev['gate'][-1][:160]}")
    if complete and not final:
        L.append("latest complete builds (CSV order):")
        for r in complete[-5:]:
            L.append(f"  {r['shape']}@{r['config']:<9} Fmax {r['fmax_mhz']} MHz ({r['fmax_s11']} / {r['fmax_s13']} / "
                     f"{r['fmax_s17']})  clb {r['clb']}  memory {r['memory']}/{r['mem_blocks_model']}  "
                     f"wc {r['wc']}/{r['expected_wc']}")
    mem = D.mem_available_gib()
    disk = shutil.disk_usage(D.OUT).free / 1e12
    L.append(f"machine: memory available {mem:.0f} GiB, our processes {ours:.1f} GiB; disk free {disk:.2f} TB, "
             f"this run's build folders {_size_gb([b.dir for b in builds if b.dir.exists()]):.0f} GB")
    # timing: finished syntheses (actual ABC vs predicted), running ABCs' predicted ends
    ratios, dur = [], [s for _, s in ev["synth_ok"]]
    for lab, _ in ev["synth_ok"]:
        b = next((x for x in builds if x.label == lab), None)
        net = next(iter((b.dir / "synth").glob("0_*.parmys.blif")), None) if b else None
        w = _wall(D._read(b.dir / "synth" / "abc0.out")) if b else None
        if net is not None and w:
            ratios.append(w / 3600 / abc_hours(net.stat().st_size / 1e6))
    if dur:
        L.append(f"syntheses finished: {len(dur)}, median {_hms(statistics.median(dur))}, max {_hms(max(dur))}"
                 + (f"; ABC actual / predicted: median x{statistics.median(ratios):.2f}" if ratios else ""))
    vdur = [float(m.group(1)) for m in (re.search(r"VPR OK \((\d+) s", x) for x in ev["vpr_ok"]) if m]
    if vdur:
        L.append(f"VPR seeds finished: {len(vdur)}, median {_hms(statistics.median(vdur))}, max {_hms(max(vdur))}")
    ends = []
    for b, age in abc:
        net = next(iter((b.dir / "synth").glob("0_*.parmys.blif")), None) if b else None
        if net is not None:
            mb = net.stat().st_size / 1e6
            ends.append((net.stat().st_mtime + abc_hours(mb) * 3600 * (statistics.median(ratios) if ratios else 1),
                         b.label, mb))
    if ends:
        ends.sort()
        f = lambda e: f"{time.strftime('%m-%d %H:%M', time.localtime(e[0]))} {e[1]} ({e[2]:.0f} MB)"  # noqa: E731
        L.append(f"running ABCs, predicted ends [estimate]: first {f(ends[0])}, last {f(ends[-1])}"
                 + (f"; then {max(0, wait_syn)} waiting syntheses and their VPRs" if wait_syn > 0 else "; then VPR"))
    if final:
        L.append(f"driver's last line: {ev['exit'] or 'none (ended without its summary line)'}")
        L.append("every build (CSV, current circuit):")
        for r in cur:
            L.append(f"  {r['shape']}@{r['config']:<9} ok {r['ok']}  Fmax {r['fmax_mhz'] or '-':>8} MHz  clb "
                     f"{r['clb'] or '-':>6}  memory {r['memory'] or '-'}/{r['mem_blocks_model']}  wc {r['wc'] or '-'}/"
                     f"{r['expected_wc']}  seeds {r['seeds'] or '-'}" + (f"  note: {r['note']}" if r['note'] else ""))
    return "\n".join(x for x in L if x) + "\n"


def emit(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(text + "\n")
    print(text, flush=True)


def self_test() -> int:
    """The run-log parser on a synthetic log (every kind of line, one of each failure), the ABC law at its two
    calibration points, the time formatter, and a report against a finished (absent) driver without errors."""
    log = ["2026-10-15 20:00:00 === VTR DSE: 72 builds, seeds 11,13,17",
           "2026-10-15 20:00:01 A@1x1 synth start (pid 5)", "2026-10-15 20:00:02 B@1x1 synth: cached",
           "2026-10-15 21:00:00 A@1x1 synth: OK (3600 s; peaks Parmys 1.5 GiB, ABC 0.5 GiB)",
           "2026-10-15 21:00:00 C@1x1 synth: FAILED (exit 1) (10 s; peaks Parmys None GiB, ABC None GiB)",
           "2026-10-15 21:00:01 A@1x1 seed 11: VPR start (pid 6)",
           "2026-10-15 21:10:00 A@1x1 seed 11: VPR OK (599 s, peak 4.0 GiB) -> Fmax 130.0 MHz, clb 1",
           "2026-10-15 21:10:00 A@1x1 seed 13: VPR OK (601 s, peak 4.0 GiB) -> Fmax 1 MHz; NOT COUNTED: wc 1 != 2",
           "2026-10-15 21:10:00 A@1x1 seed 17: VPR failed: VPR exit 1 (5 s, peak None GiB)",
           "2026-10-15 21:10:00 B@1x1 seed 11: finished, skipped", "2026-10-15 21:10:00 ERROR B@1x1 seed 13: boom",
           "2026-10-15 21:10:00 D@1x1 seed 11: started after 75 s at the memory gate (last wait: budget)",
           "not a stamped line", "2026-10-15 22:00:00 === wrote x.csv (72 builds); 1 error(s); exit 1"]
    ev = run_events(log)
    fails = []
    want = dict(synth_start=1, synth_ok=1, synth_fail=1, synth_cached=1, vpr_start=1, vpr_ok=2, vpr_fail=1,
                vpr_skip=1, not_counted=1, errors=1, elab_fail=0, gate=1)
    got = {k: len(ev[k]) for k in want}
    if got != want or ev["exit"] is None or ev["synth_ok"][0] != ("A@1x1", 3600.0):
        fails.append(f"run_events: {got} (want {want}), exit {ev['exit']}, {ev['synth_ok']}")
    if not (abs(abc_hours(99.4) - 3.58) < 1e-9 and abs(abc_hours(53.0) - 0.875) < 0.03):
        fails.append(f"abc_hours: {abc_hours(99.4)}, {abc_hours(53.0)}")
    if (_hms(59), _hms(3600), _hms(5400 + 59)) != ("0 min", "1 h 00 min", "1 h 30 min"):
        fails.append(f"_hms: {_hms(59)}, {_hms(3600)}, {_hms(5459)}")
    if _wall("\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03\n") != 3723:
        fails.append("_wall h:mm:ss")
    td = Path(tempfile.mkdtemp(prefix="watch_", dir=V3 / "work" / "tmp"))
    (td / "run.log").write_text("\n".join(log) + "\n")
    try:
        rep = report(2 ** 22 + 12345, time.mktime(time.strptime("2026-10-15 19:59:59", "%Y-%m-%d %H:%M:%S")), 1,
                     True, csv_path=td / "none.csv", run_log=td / "run.log")
        if not ("FINAL report" in rep and "driver: ended" in rep and "failures: 4" in rep and "exit 1" in rep
                and "syntheses finished: 1, median 1 h 00 min" in rep and "VPR seeds finished: 2" in rep):
            fails.append(f"report on a finished driver:\n{rep}")
    except Exception as e:  # noqa: BLE001
        fails.append(f"report raised {e!r}")
    for f in fails:
        print("  FAIL", f)
    print("dse_watch self-test:", "ALL PASS" if not fails else f"{len(fails)} FAILURE(S)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pid", type=int)
    ap.add_argument("--every", type=float, default=90, help="minutes between reports")
    ap.add_argument("--status", default=None, help=f"status log, appended (default {D._rel(STATUS)})")
    ap.add_argument("--once", action="store_true", help="one report now, then exit")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.pid is None:
        raise SystemExit("--pid DRIVER_PID required")
    path = Path(a.status).resolve() if a.status else STATUS
    t0 = start_time(a.pid) or time.time()
    if a.once:
        emit(report(a.pid, t0, 0, not alive(a.pid)), path)
        return 0
    n, nxt = 0, time.time() + 120                        # the first report 2 min after the start, then every --every
    while alive(a.pid):
        if time.time() >= nxt:
            try:
                emit(report(a.pid, t0, n, False), path)
            except Exception as e:  # noqa: BLE001
                emit(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} | report {n} failed: {e!r} ====", path)
            n, nxt = n + 1, nxt + a.every * 60
        time.sleep(30)
    time.sleep(5)                                        # the driver's last CSV and log lines
    emit(report(a.pid, t0, n, True), path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
