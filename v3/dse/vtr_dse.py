#!/usr/bin/env python3
"""vtr_dse.py — VTR of the attention DSE builds (docs/experiment.md §6 step 3, its launch plan) -> dse/results/attn_vtr_results.csv.

A build is one attention head (`att_head`) at the fixed mapping (the `head_params` defaults, experiment.md §4.2) for one
shape (§3.2) on one crossbar config: 6 shapes x 12 configs = 72 builds. Its circuit is the smoke's (smoke.circuit: the
VTR twins + the parameter wrapper; top att_head_<shape>_<R>x<C>). The first 12 hex digits of the circuit's sha256 name
the build's folder work/vtr/dse/<shape>_<R>x<C>_<hash>/, so another RTL gets another folder and nothing is overwritten:
  <top>.v            the circuit
  elab/              the elaboration check (smoke.elab: Parmys's Yosys front end), its verdict in elab/elab.json: the top,
                     crossbars == the simulator's, the memory list (words x bits, with counts) == the simulator's, every
                     memory shape one Parmys maps. A failed check stops the build.
  synth/, synth.log  Parmys + ABC, once (run_vtr_flow.py -end abc). synth/<top>.abc.blif is VPR's input: byte-identical
                     to the full flow's pre-vpr.blif (checked on the accumulator smoke and the head smoke).
  s<seed>/           VPR on that netlist with the full flow's VPR command line (the config's arch XML, route channel
                     width 300, clock.sdc, --seed), under `time -v` as in the full flow; s<seed>/flow.log gets the
                     success line when VPR exits 0.
A seed counts when VPR ended OK, Fmax > 0 and wc == the simulator's crossbars. Its memory blocks are compared with the
simulator's prediction (sim/platform/area.py) and the result is reported in mem_check, not counted against the seed
(2026-10-16, user: a build whose memory differs is priced from VPR's own count). The CSV has one line per build with a
folder: the build's parameters, the elaboration verdict, fmax_mhz = the mean over the counted seeds and each seed's
Fmax, CLB / memory / wc / io / DSP = the largest over the counted seeds, the simulator's counts, mem_check, wall times,
memory peaks, and a note naming each seed that did not count.

Scheduling (the launch plan): --jobs workers over one priority queue. First the elaboration checks of every selected
build; then the syntheses, largest first (the simulator's memory bits); a waiting VPR seed goes before a waiting
synthesis. Every tool process passes the memory gate (Gate) and runs under a per-process address-space cap (--mem-gib,
via prlimit, inherited by the tool's children); there is no timeout.
Resume: the same command skips every finished step of the same circuit: an elaboration that ran (Yosys exit 0, its
verdict stands), a synthesis with its success line and netlist, a seed with its success line. An unfinished step runs
again in its folder.
Run log: one timestamped line per event, appended to work/dse/vtr_dse.log and echoed to stdout (do not redirect stdout
into the same file); each tool's PID is logged at its start.

Usage:
  python3 v3/dse/vtr_dse.py --elab-only                        the elaboration checks of the 72 builds
  python3 v3/dse/vtr_dse.py [--build JetFormer-150P@128x128]   the builds end to end (default all 72), seeds 11,13,17
  python3 v3/dse/vtr_dse.py --aggregate                        rebuild the CSV from the folders (no tool)
  python3 v3/dse/vtr_dse.py --self-test                        fake tools, no VTR
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "vtr"))          # run_vtr.py and smoke.py: the generic VTR flow
import smoke  # noqa: E402
from dse.space import CONFIGS, SHAPES  # noqa: E402
from run_vtr import build_vtr_command, find_vpr_log, parse_metrics, parse_resources  # noqa: E402
from simulator.platform import area  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402

OUT = V3 / "work" / "vtr" / "dse"
CSV_PATH = V3 / "dse" / "results" / "attn_vtr_results.csv"     # vtr_att_72.csv until 2026-10-17
RUN_LOG = V3 / "work" / "dse" / "vtr_dse.log"
SEEDS = (11, 13, 17)
TOOLS = dict(flow=smoke.VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py", python=smoke.VTR_ROOT / ".venv" / "bin" /
             "python", vpr=smoke.VTR_ROOT / "vpr" / "vpr")
FLOW_OK = re.compile(r"\sOK \(took ([\d.]+) seconds")   # run_vtr_flow.py's success line ("<arch>/<top>\t\tOK (took ..")
MAXRSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")     # GNU time -v
YOSYS_PEAK = re.compile(r"MEM: ([\d.]+) MB peak")
LEGAL_PORTS = ((1, 1), (1, 2), (2, 1), (2, 2))             # (read, write) ports Parmys's RAM mapper takes; ROMs too
MAP_KEYS = ("P_PROJ", "N_E", "N_EP", "N_EXP", "N_OC", "BKV", "X_S")
DISK_GB_PER_BUILD = 8      # the head smoke's folder: synthesis ~0.5 GB + 1.4 GB per VPR seed, x 1.7 for larger builds
ELAB, VPR_FIRST, SYNTH, STOP = 0, 1, 2, 9                 # task priorities (lower first)


def slug(shape: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", shape.lower()).strip("_")


def build_dir(out: Path, name: str, text: str) -> Path:
    return out / f"{name}_{hashlib.sha256(text.encode()).hexdigest()[:12]}"


class Build:
    """One DSE build: its circuit and the simulator's counts (crossbars, memory list, predicted memory blocks)."""

    def __init__(self, shape: str, n: int, dm: int, d: int, R: int, C: int, out: Path = OUT):
        self.shape, self.n, self.dm, self.d, self.R, self.C = shape, n, dm, d, R, C
        self.name, self.label = f"{slug(shape)}_{R}x{C}", f"{shape}@{R}x{C}"
        self.lp = headParams(n, dm, d, R, C)
        self.params = smoke.head_point(self.lp)
        self.top, self.text, self.n_xbar, _ = smoke.circuit("head", self.name, params=self.params)
        self.digest = hashlib.sha256(self.text.encode()).hexdigest()
        self.dir = build_dir(out, self.name, self.text)
        g, _ = attentionHead(self.lp, withValues=False)
        res = [m.resources for m in g.modules]
        if sum(r.nCrossbars for r in res) != self.n_xbar:
            raise ValueError(f"{self.label}: headParams nXbar {self.n_xbar} != the module models' crossbars")
        self.mems = Counter()
        for r in res:
            self.mems.update(r.memories)
        self.mem_blocks = area.predictedMemBlocks(res, R, C)
        self.mem_bits = sum(w * b * k for (w, b), k in self.mems.items())
        self.arch = smoke.arch_for(self.params)

    @property
    def circuit_path(self) -> Path:
        return self.dir / f"{self.top}.v"

    @property
    def netlist(self) -> Path:
        return self.dir / "synth" / f"{self.top}.abc.blif"


def all_builds(out: Path = OUT) -> list:
    return [Build(s, n, dm, d, R, C, out) for (s, n, dm, d) in SHAPES for (R, C) in CONFIGS]


def _read(p: Path) -> str:
    return p.read_text(errors="replace") if p.is_file() else ""


def _json(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def _rc_text(rc: int) -> str:
    return f"killed (signal {-rc})" if rc < 0 else f"exit {rc}"


def _gib(path: Path, rx=MAXRSS, scale: float = 2 ** 20):
    """The last peak in a tool's output (GNU time -v: kB; Yosys's MEM line: MB, scale 1024), in GiB; None: none."""
    m = rx.findall(_read(path))
    return round(float(m[-1]) / scale, 2) if m else None


def _rel(p: Path) -> str:
    return str(p.relative_to(V3.parent)) if p.is_relative_to(V3.parent) else str(p)


def run_tool(cmd: list, log: Path, cwd: Path, mem_gib: float, on_start=None) -> int:
    """Run one tool command to its end (no timeout) with its output in `log`, under an address-space cap of mem_gib
    (prlimit execs the tool in place, its children inherit the cap): the exit code (< 0: killed by that signal)."""
    capped = ["prlimit", f"--as={int(mem_gib * 2 ** 30)}", "--", *map(str, cmd)]
    with open(log, "w") as fh:
        p = subprocess.Popen(capped, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd)
        if on_start is not None:
            on_start(p.pid)
        return p.wait()


def mem_available_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    raise RuntimeError("no MemAvailable in /proc/meminfo")


class RunLog:
    """The run log: each line timestamped and appended as it happens (opened per line, so `tail -f` and an interrupted
    run see every line), echoed to stdout; safe from the worker threads."""

    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self.lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
            print(line, flush=True)


class Gate:
    """Memory admission of the tool processes (the launch plan). A process starts when
      - the reserves of the running processes plus its own fit in `budget` (GiB),
      - the machine's available memory is at least its reserve plus `free` (GiB; this also covers other users),
      - a VPR: fewer than `vpr_max` VPRs run; a synthesis: no VPR waits, and the last synthesis started `gap_s` ago or
        more (the Parmys peaks do not stack).
    With none of ours running, the budget does not apply (a reserve above the budget still starts). Reserves (GiB) start
    near 1.5x the head smoke's peaks (Yosys elaboration 0.68, Parmys 3.0, ABC 1.0, VPR 8.1 GB) and rise to 1.25x the
    largest peak measured in the run (`measured`); a synthesis holds the Parmys reserve until its ABC starts
    (synth/abc0.out exists), then the ABC reserve."""
    RESERVE = dict(elab=2.0, parmys=6.0, abc=1.5, vpr=12.0)

    def __init__(self, budget: float, free: float = 16.0, vpr_max: int = 12, gap_s: float = 60.0,
                 meminfo=mem_available_gib, clock=time.monotonic, poll_s: float = 5.0, log=None):
        self.budget, self.free, self.vpr_max, self.gap_s = budget, free, vpr_max, gap_s
        self.meminfo, self.clock, self.poll_s, self.log = meminfo, clock, poll_s, log
        self.reserve = dict(self.RESERVE)
        self.cv = threading.Condition()
        self.held = {}                                   # token -> (kind, reserve function)
        self.vpr_waiting = 0
        self.last_synth = None

    def reserve_fn(self, kind: str, synth_dir: Path | None = None):
        if kind == "synth":
            return lambda: self.reserve["abc"] if (synth_dir / "abc0.out").exists() else self.reserve["parmys"]
        return lambda: self.reserve[kind]

    def wait_reason(self, kind: str, need: float) -> str:
        """Why a process of `kind` reserving `need` GiB may not start now ("" = it may). Call with the lock held."""
        running = list(self.held.values())
        if running and sum(fn() for _, fn in running) + need > self.budget:
            return "budget"
        if self.meminfo() < need + self.free:
            return "free memory"
        if kind == "vpr" and sum(k == "vpr" for k, _ in running) >= self.vpr_max:
            return "VPR limit"
        if kind == "synth" and self.vpr_waiting:
            return "a VPR waits"
        if kind == "synth" and self.last_synth is not None and self.clock() - self.last_synth < self.gap_s:
            return "synthesis spacing"
        return ""

    @contextlib.contextmanager
    def hold(self, kind: str, tag: str = "", synth_dir: Path | None = None):
        fn = self.reserve_fn(kind, synth_dir)
        t0, why = time.monotonic(), ""
        with self.cv:
            self.vpr_waiting += kind == "vpr"
            try:
                while True:
                    reason = self.wait_reason(kind, fn())
                    if not reason:
                        break
                    why = reason
                    self.cv.wait(timeout=self.poll_s)
            finally:
                self.vpr_waiting -= kind == "vpr"
            token = object()
            self.held[token] = (kind, fn)
            if kind == "synth":
                self.last_synth = self.clock()
            self.cv.notify_all()                         # a synthesis may wait on vpr_waiting
        waited = time.monotonic() - t0
        if self.log is not None and waited >= 60:
            self.log(f"{tag}: started after {waited:.0f} s at the memory gate (last wait: {why})")
        try:
            yield
        finally:
            with self.cv:
                del self.held[token]
                self.cv.notify_all()

    def measured(self, kind: str, gib) -> None:
        if gib:
            with self.cv:
                self.reserve[kind] = max(self.reserve[kind], round(1.25 * gib, 2))


class Pool:
    """`jobs` worker threads over one priority queue (lower priority value first; first in, first out within a
    priority). A task may submit more tasks; run() returns when every task, those included, has ended. A task that
    raises is recorded in `errors` and the pool goes on. run() may be called again (a new set of workers)."""

    def __init__(self, jobs: int):
        self.jobs = max(1, jobs)
        self.q = queue.PriorityQueue()
        self.seq = itertools.count()
        self.errors = []

    def submit(self, prio: int, fn, *args) -> None:
        self.q.put((prio, next(self.seq), fn, args))

    def run(self) -> None:
        def worker():
            while True:
                _, _, fn, args = self.q.get()
                try:
                    if fn is None:                       # stop marker (queued after the last task ended)
                        return
                    fn(*args)
                except Exception as e:  # noqa: BLE001
                    self.errors.append(e)
                finally:
                    self.q.task_done()
        ts = [threading.Thread(target=worker, daemon=True) for _ in range(self.jobs)]
        for t in ts:
            t.start()
        self.q.join()                                    # every task, including the ones tasks submitted
        for _ in ts:
            self.q.put((STOP, next(self.seq), None, ()))
        for t in ts:
            t.join()


def elab_check(e: dict, top: str, n_xbar: int, mems: Counter) -> list:
    """The elaboration check's problems ([] = OK): the top, the crossbar count and the memory list (words x bits, with
    counts) against the simulator's, and every memory's port shape (1R1W, 1R2W, 2R1W, 2R2W or a ROM)."""
    bad = []
    if e.get("top") != top:
        bad.append(f"top {e.get('top')} (expected {top})")
    if e.get("dpe") != n_xbar:
        bad.append(f"crossbars {e.get('dpe')} (simulator {n_xbar})")
    got = Counter((m["size"], m["width"]) for m in e.get("mems", []))
    if got != mems:
        txt = lambda c: ", ".join(f"{k} x {w}x{b}" for (w, b), k in sorted(c.items())) or "none"  # noqa: E731
        bad.append(f"memories: Yosys has more {txt(got - mems)}; fewer {txt(mems - got)} (words x bits)")
    odd = [m for m in e.get("mems", []) if not (m["wr"] == 0 or (m["rd"], m["wr"]) in LEGAL_PORTS)]
    if odd:
        bad.append(f"{len(odd)} memories with port shapes Parmys does not map: "
                   + ", ".join(f"{m['name']} {m['rd']}R{m['wr']}W" for m in odd[:4]))
    return bad


def elab_record(b: Build):
    """The build's Yosys result when it ran to its end (exit 0), else None. Yosys's result is fixed by the circuit
    (the folder); the verdict also depends on the simulator's counts, so elab_verdict recomputes it every time."""
    e = _json(b.dir / "elab" / "elab.json")
    return e if e is not None and e.get("rc") == 0 else None


def elab_verdict(b: Build):
    """The elaboration check's problems ([] = OK) against the current simulator; None: Yosys has not run to its end."""
    e = elab_record(b)
    return None if e is None else elab_check(e, b.top, b.n_xbar, b.mems)


def synth_done(b: Build) -> bool:
    return bool(FLOW_OK.search(_read(b.dir / "synth.log"))) and b.netlist.is_file()


def seed_done(b: Build, seed: int) -> bool:
    """A seed is finished when VPR exited 0 on this build's netlist; its result stands (a count mismatch included)."""
    return bool(FLOW_OK.search(_read(b.dir / f"s{seed}" / "flow.log")))


def seed_result(b: Build, seed: int) -> tuple:
    """(result dict or None, why not counted) of one seed."""
    sd = b.dir / f"s{seed}"
    m = FLOW_OK.search(_read(sd / "flow.log"))
    if not m:
        return None, "not run" if not sd.exists() else "VPR not finished OK"
    try:
        log = find_vpr_log(sd)
        _, fmax = parse_metrics(log)
        res = parse_resources(log)
    except Exception as e:  # noqa: BLE001
        return None, f"VPR log: {e}"
    r = dict(fmax=fmax, clb=res.get("clb", 0), memory=res.get("memory", 0), wc=res.get("wc", 0), io=res.get("io", 0),
             dsp=res.get("dsp_top", 0), vpr_s=float(m.group(1)), peak=_gib(sd / "vpr.out"))
    if not fmax or fmax <= 0:
        return None, "no Fmax in the VPR log"
    if r["wc"] != b.n_xbar:
        return r, f"wc {r['wc']} != the simulator's {b.n_xbar}"
    return r, ""                                         # memory vs the prediction: build_row's mem_check


FIELDS = (["shape", "n", "d_model", "d", "config", "R", "C", *MAP_KEYS, "build", "hash", "top", "elab", "ok",
           "fmax_mhz", *[f"fmax_s{s}" for s in SEEDS], "clb", "memory", "wc", "io", "dsp", "expected_wc",
           "mem_blocks_model", "mem_bits_model", "mem_check", "seeds", "synth_s", "vpr_s", "peak_gib_synth",
           "peak_gib_vpr", "dir", "note"])


def build_row(b: Build) -> dict:
    """The build's CSV line from its folder."""
    row = dict.fromkeys(FIELDS, "")
    row.update(shape=b.shape, n=b.n, d_model=b.dm, d=b.d, config=f"{b.R}x{b.C}", R=b.R, C=b.C,
               **{k: b.params[k] for k in MAP_KEYS}, build=b.name, hash=b.digest[:12], top=b.top, ok=0,
               expected_wc=b.n_xbar, mem_blocks_model=b.mem_blocks, mem_bits_model=b.mem_bits, dir=_rel(b.dir))
    e, bad = _json(b.dir / "elab" / "elab.json"), elab_verdict(b)
    if e is not None and e.get("rc") != 0:
        bad = [f"Yosys {_rc_text(e['rc'])}"]
    row["elab"] = "" if e is None else ("OK" if not bad else "; ".join(bad))
    if e is None or bad:
        row["note"] = ("elaboration not run" if e is None else "elaboration did not finish" if e["rc"] != 0
                       else "stopped at the elaboration check")
        return row
    sl = _read(b.dir / "synth.log")
    m = FLOW_OK.search(sl)
    peaks = [g for g in (_gib(b.dir / "synth" / "parmys.out"), _gib(b.dir / "synth" / "abc0.out")) if g]
    row.update(synth_s=float(m.group(1)) if m else "", peak_gib_synth=max(peaks) if peaks else "")
    if not synth_done(b):
        row["note"] = "not synthesized" if not sl else "synthesis not finished OK"
        return row
    good, notes, ran = [], [], []
    for s in SEEDS:
        r, why = seed_result(b, s)
        if r is not None:
            ran.append(r)
            row[f"fmax_s{s}"] = r["fmax"]
        if r is not None and not why:
            good.append((s, r))
        else:
            notes.append(f"s{s}: {why}")
    row.update(seeds=";".join(str(s) for s, _ in good), note="; ".join(notes),
               vpr_s=max(r["vpr_s"] for r in ran) if ran else "",
               peak_gib_vpr=max((r["peak"] for r in ran if r["peak"]), default=""))
    if good:
        row.update(ok=1, fmax_mhz=round(sum(r["fmax"] for _, r in good) / len(good), 3),
                   **{k: max(r[k] for _, r in good) for k in ("clb", "memory", "wc", "io", "dsp")})
        row["mem_check"] = ("equal" if all(r["memory"] == b.mem_blocks for _, r in good)
                            else f"VPR {row['memory']} vs predicted {b.mem_blocks}")
    return row


def write_csv(rows: list, path: Path, order: list) -> None:
    """Merge build lines into the CSV (one line per build; a build's new line replaces its old one), in build order."""
    old = {}
    if path.is_file():
        with open(path, newline="") as f:
            old = {r["build"]: r for r in csv.DictReader(f)}
    old.update({r["build"]: r for r in rows})
    rank = {name: i for i, name in enumerate(order)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(sorted(old.values(), key=lambda r: (rank.get(r["build"], len(rank)), r["build"])))
    tmp.replace(path)                                    # readers never see a half-written CSV


class Run:
    """One invocation: the selected builds through the elaboration checks, then synthesis and VPR."""

    def __init__(self, a, builds: list, order: list, csv_path: Path, lg: RunLog, gate: Gate):
        self.a, self.builds, self.order, self.csv_path, self.lg, self.gate = a, builds, order, csv_path, lg, gate
        self.seeds = [int(s) for s in a.seeds.split(",")]
        self.pool = Pool(a.jobs)
        self.errors, self.csv_lock = [], threading.Lock()

    def error(self, what: str, e: Exception) -> None:
        self.errors.append(e)
        self.lg(f"ERROR {what}: {e!r}")

    def save_csv(self) -> None:
        with self.csv_lock:
            write_csv([build_row(b) for b in self.builds if b.dir.exists()], self.csv_path, self.order)

    def tool(self, cmd: list, log: Path, cwd: Path, tag: str) -> int:
        return run_tool(cmd, log, cwd, self.a.mem_gib, on_start=lambda pid: self.lg(f"{tag} start (pid {pid})"))

    def elab(self, b: Build) -> None:
        try:
            bad = elab_verdict(b)
            if bad is not None:
                self.lg(f"{b.label} elab: cached (Yosys), checked again: {'OK' if not bad else 'FAILED: ' + '; '.join(bad)}")
                return
            work = b.dir / "elab"
            work.mkdir(parents=True, exist_ok=True)
            if _read(b.circuit_path) != b.text:
                b.circuit_path.write_text(b.text)
            with self.gate.hold("elab", f"{b.label} elab"):
                e = smoke.elab(b.circuit_path, b.arch, work,
                               run=lambda cmd, cwd: self.tool(cmd, cwd / "yosys.out", cwd, f"{b.label} elab"))
            e["peak_gib"] = _gib(work / "elab.log", YOSYS_PEAK, 1024)
            self.gate.measured("elab", e["peak_gib"])
            e["problems"] = elab_check(e, b.top, b.n_xbar, b.mems) if e["rc"] == 0 else [f"Yosys {_rc_text(e['rc'])}"]
            (work / "elab.json").write_text(json.dumps(e, indent=1) + "\n")
            self.lg(f"{b.label} elab: {'OK' if not e['problems'] else 'FAILED: ' + '; '.join(e['problems'])} "
                    f"({e['wall_s']} s, peak {e['peak_gib']} GiB; crossbars {e.get('dpe')}/{b.n_xbar}, "
                    f"{len(e.get('mems', []))} memories vs the simulator's {sum(b.mems.values())})")
        except Exception as ex:  # noqa: BLE001
            self.error(f"{b.label} elab", ex)

    def queue_seeds(self, b: Build) -> None:
        for s in self.seeds:
            if seed_done(b, s):
                self.lg(f"{b.label} seed {s}: finished, skipped")
            else:
                self.pool.submit(VPR_FIRST, self.vpr, b, s)

    def synth(self, b: Build) -> None:
        try:
            sd = b.dir / "synth"
            sd.mkdir(parents=True, exist_ok=True)
            py = TOOLS["python"] if TOOLS["python"].is_file() else None
            cmd = build_vtr_command(TOOLS["flow"], py, b.circuit_path, b.arch, smoke.ROUTE_CHAN_WIDTH, smoke.SDC,
                                    sd) + ["-end", "abc"]
            with self.gate.hold("synth", f"{b.label} synth", synth_dir=sd):
                t0 = time.time()
                rc = self.tool(cmd, b.dir / "synth.log", sd, f"{b.label} synth")
            pk = (_gib(sd / "parmys.out"), _gib(sd / "abc0.out"))
            self.gate.measured("parmys", pk[0])
            self.gate.measured("abc", pk[1])
            ok = synth_done(b)
            self.lg(f"{b.label} synth: {'OK' if ok else 'FAILED (' + _rc_text(rc) + ')'} ({time.time() - t0:.0f} s; "
                    f"peaks Parmys {pk[0]} GiB, ABC {pk[1]} GiB)")
            if ok:
                self.queue_seeds(b)
            else:
                self.save_csv()
        except Exception as ex:  # noqa: BLE001
            self.error(f"{b.label} synth", ex)

    def vpr(self, b: Build, seed: int) -> None:
        try:
            sd = b.dir / f"s{seed}"
            sd.mkdir(parents=True, exist_ok=True)
            net = b.netlist
            cmd = ["/usr/bin/env", "time", "-v", TOOLS["vpr"], b.arch, b.top, "--circuit_file", net,
                   "--route_chan_width", smoke.ROUTE_CHAN_WIDTH, "--sdc_file", smoke.SDC, "--seed", seed]
            (sd / "flow.log").write_text("")                 # an earlier attempt's line no longer stands
            with self.gate.hold("vpr", f"{b.label} seed {seed}"):
                t0 = time.time()
                rc = self.tool(cmd, sd / "vpr.out", sd, f"{b.label} seed {seed}: VPR")
                dt = time.time() - t0
            status = "OK" if rc == 0 else f"failed: VPR {_rc_text(rc)}"
            (sd / "flow.log").write_text(
                f"VPR only, netlist {net} (sha256 {hashlib.sha256(net.read_bytes()).hexdigest()}), "
                f"circuit sha256 {b.digest}\n{b.arch.stem}/{b.top}\t\t{status} (took {dt:.2f} seconds)\n")
            self.gate.measured("vpr", _gib(sd / "vpr.out"))
            r, why = seed_result(b, seed)
            res = "" if r is None else (f" -> Fmax {r['fmax']} MHz, clb {r['clb']}, memory {r['memory']}/{b.mem_blocks}, "
                                        f"wc {r['wc']}/{b.n_xbar}, io {r['io']}, dsp {r['dsp']}")
            self.lg(f"{b.label} seed {seed}: VPR {status} ({dt:.0f} s, peak {_gib(sd / 'vpr.out')} GiB){res}"
                    + (f"; NOT COUNTED: {why}" if why else ""))
            self.save_csv()
        except Exception as ex:  # noqa: BLE001
            self.error(f"{b.label} seed {seed}", ex)

    def go(self) -> int:
        for b in self.builds:
            self.pool.submit(ELAB, self.elab, b)
        self.pool.run()                                  # every elaboration check first
        passed = [b for b in self.builds if elab_verdict(b) == []]
        self.lg(f"elaboration: {len(passed)}/{len(self.builds)} builds OK"
                + "".join(f"; FAILED {b.label}" for b in self.builds if b not in passed))
        if not self.a.elab_only:
            for b in sorted(passed, key=lambda x: (-x.mem_bits, -x.n_xbar, x.name)):   # largest first
                if synth_done(b):
                    self.lg(f"{b.label} synth: cached")
                    self.queue_seeds(b)                  # VPR priority: ahead of every waiting synthesis
                else:
                    self.pool.submit(SYNTH, self.synth, b)
            self.pool.run()
        self.errors += self.pool.errors
        self.save_csv()
        rows = [build_row(b) for b in self.builds]
        for r in rows:
            if self.a.elab_only:
                continue
            self.lg(f"{r['shape']}@{r['config']}: ok={r['ok']} Fmax {r['fmax_mhz'] or '-'} MHz (seeds {r['seeds'] or '-'}) "
                    f"clb {r.get('clb', '')} memory {r.get('memory', '')}/{r['mem_blocks_model']} wc {r.get('wc', '')}/"
                    f"{r['expected_wc']}" + (f"; {r['note']}" if r.get("note") else ""))
        want = {f"s{s}" for s in self.seeds}
        complete = all(r["elab"] == "OK" and (self.a.elab_only or not any(n.split(":")[0] in want for n in
                       r.get("note", "").split("; ") if n) and r["ok"] == 1) for r in rows)
        rc = 0 if complete and not self.errors else 1
        self.lg(f"=== wrote {self.csv_path} ({len(rows)} builds); {len(self.errors)} error(s); exit {rc}")
        return rc


def select(builds: list, names: list) -> list:
    if not names:
        return builds
    by = {**{b.label: b for b in builds}, **{b.name: b for b in builds}}
    bad = [n for n in names if n not in by]
    if bad:
        raise SystemExit(f"unknown build(s) {bad}; e.g. {builds[0].label} or {builds[0].name}")
    return [by[n] for n in dict.fromkeys(names)]


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="append", default=[], help="SHAPE@RxC or the build name (repeatable; default all)")
    ap.add_argument("--elab-only", action="store_true", help="the elaboration checks only")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--jobs", type=int, default=36)
    ap.add_argument("--mem-gib", type=float, default=48, help="address-space cap per tool process")
    ap.add_argument("--vpr-max", type=int, default=12)
    ap.add_argument("--synth-gap", type=float, default=60, help="seconds between synthesis starts")
    ap.add_argument("--free-gib", type=float, default=16, help="memory kept free besides a new process's reserve")
    ap.add_argument("--aggregate", action="store_true", help="rebuild the CSV from the folders (no tool)")
    ap.add_argument("--out", default=None, help=f"build folders (default {_rel(OUT)})")
    ap.add_argument("--csv", default=None, help=f"default {_rel(CSV_PATH)}")
    ap.add_argument("--log", default=None, help=f"run log, appended (default {_rel(RUN_LOG)})")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if any(int(s) not in SEEDS for s in a.seeds.split(",")):
        raise SystemExit(f"--seeds must be among {SEEDS}")
    out = Path(a.out).resolve() if a.out else OUT
    csv_path = Path(a.csv).resolve() if a.csv else CSV_PATH
    every = all_builds(out)
    builds = select(every, a.build)
    order = [b.name for b in every]
    if a.aggregate:
        write_csv([build_row(b) for b in builds if b.dir.exists()], csv_path, order)
        print(f"wrote {csv_path}")
        return 0
    lg = RunLog(Path(a.log).resolve() if a.log else RUN_LOG)
    out.mkdir(parents=True, exist_ok=True)
    budget = round(mem_available_gib() - a.free_gib, 1)
    gate = Gate(budget, a.free_gib, a.vpr_max, a.synth_gap, log=lg)
    disk = shutil.disk_usage(out).free / 1e9
    need = 0 if a.elab_only else DISK_GB_PER_BUILD * sum(not synth_done(b) for b in builds)
    lg(f"=== VTR DSE: {len(builds)} builds, seeds {a.seeds}, jobs {a.jobs}, cap {a.mem_gib} GiB/process, no timeout; "
       f"gate: budget {budget} GiB, free >= {a.free_gib} GiB + reserve, VPR <= {a.vpr_max}, syntheses >= {a.synth_gap} s "
       f"apart, reserves {gate.reserve} GiB; disk free {disk:.0f} GB (need {need}); out {_rel(out)}"
       + ("; elaboration only" if a.elab_only else "") + ": " + ", ".join(b.label for b in builds))
    if disk < need:
        lg(f"=== not started: {disk:.0f} GB free on the disk, {need} GB needed; exit 1")
        return 1
    return Run(a, builds, order, csv_path, lg, gate).go()


# ---------------------------------------------------------------------------------------------------------------- tests

_FAKE_YOSYS = r'''#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path
a = sys.argv[1:]
script = Path(a[a.index("-s") + 1]).read_text()
top = Path(re.search(r"read_verilog -sv -nolatches (\S+)", script).group(1)).stem
spec = json.loads(Path(os.environ["FAKE_SPEC"]).read_text())[top]["elab"]
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(f"yosys {top}\n")
if spec.get("kill"):
    os.kill(os.getpid(), 9)
for path, what in re.findall(r"tee -q -o (\S+) (select -list A:top|select -count t:dpe|dump t:\$mem_v2)", script):
    if what.startswith("select -list"):
        Path(path).write_text("".join(t + "\n" for t in spec.get("tops", [top])))
    elif what.startswith("select -count"):
        Path(path).write_text(f"{spec['dpe']}\n")
    else:
        Path(path).write_text("".join(
            f"  cell $mem_v2 $mem{i}\n    parameter \\SIZE {s}\n    parameter \\WIDTH {w}\n"
            f"    parameter \\RD_PORTS {r}\n    parameter \\WR_PORTS {p}\n  end\n" for i, (s, w, r, p) in
            enumerate(spec["mems"])))
Path(a[a.index("-l") + 1]).write_text("End of script. Logfile hash: 0, CPU: user 0.01s, MEM: 700.00 MB peak\n")
sys.exit(spec.get("rc", 0))
'''

_FAKE_FLOW = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
circ, arch, temp = Path(a[0]), Path(a[1]), Path(a[a.index("-temp_dir") + 1])
top = circ.stem
spec = json.loads(Path(os.environ["FAKE_SPEC"]).read_text())[top]
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(f"synth {top} {' '.join(a)}\n")
(temp / "parmys.out").write_text("\tMaximum resident set size (kbytes): 3145728\n")
(temp / "abc0.out").write_text("\tMaximum resident set size (kbytes): 1048576\n")
if spec.get("synth_rc", 0) == 0:
    (temp / f"{top}.abc.blif").write_text(f".model {top}\n.end\n")
    print(f"{arch.stem}/{top}\t\tOK (took 1.50 seconds, overall memory peak 3.00 GiB consumed by parmys run)")
else:
    print(f"{arch.stem}/{top}\t\tfailed: parmys (took 1.00 seconds)")
sys.exit(spec.get("synth_rc", 0))
'''

_FAKE_VPR = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
top, seed = a[1], a[a.index("--seed") + 1]
v = json.loads(Path(os.environ["FAKE_SPEC"]).read_text())[top]["vpr"]
v = dict(v["all"], **v.get(seed, {}))
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(f"vpr {top} {seed} {chr(31).join(a)}\n")
fm = v["fmax"]
Path("vpr_stdout.log").write_text(
    f"Final critical path delay (least slack): {1000 / fm:.3f} ns, Fmax: {fm} MHz\nclk to clk CPD: {1000 / fm:.3f} ns "
    f"({fm} MHz)\nResource usage...\n" + "".join(f"\tNetlist\n\t\t{n}\tblocks of type: {t}\n\tArchitecture\n"
                                                 f"\t\t99999\tblocks of type: {t}\n" for t, n in v["res"].items()) + "\n")
sys.exit(v.get("rc", 0))
'''


def _gate_tests() -> list:
    """Gate: the budget (with something of ours running), the free memory (with or without), the VPR limit, a waiting
    VPR and the start spacing holding a synthesis back, the Parmys -> ABC reserve switch on abc0.out, measured peaks
    raising (never lowering) a reserve, a reserve above the budget starting alone, and a waiter released by a release."""
    fails = []
    mem, now = [100.0], [1000.0]
    g = Gate(budget=50, free=10, vpr_max=2, gap_s=60, meminfo=lambda: mem[0], clock=lambda: now[0], poll_s=0.05)
    hold = lambda kind, fn: g.held.__setitem__(object(), (kind, fn))  # noqa: E731
    if g.wait_reason("vpr", 12):
        fails.append(f"gate: an idle machine must admit a VPR ({g.wait_reason('vpr', 12)})")
    hold("vpr", lambda: 12.0)
    hold("vpr", lambda: 12.0)
    if g.wait_reason("vpr", 12) != "VPR limit":
        fails.append(f"gate: the third VPR at vpr_max 2: {g.wait_reason('vpr', 12)!r}")
    hold("abc", lambda: 20.0)                            # 44 of 50 held
    if g.wait_reason("elab", 6) or g.wait_reason("elab", 6.5) != "budget":
        fails.append(f"gate: budget edge (44 + 6 fits 50, 44 + 6.5 not): {g.wait_reason('elab', 6)!r}, "
                     f"{g.wait_reason('elab', 6.5)!r}")
    g.held.clear()
    mem[0] = 15.9
    if g.wait_reason("elab", 6) != "free memory":         # 15.9 < 6 + 10, nothing of ours running
        fails.append(f"gate: free memory below reserve + free: {g.wait_reason('elab', 6)!r}")
    mem[0] = 16.0
    if g.wait_reason("elab", 6):
        fails.append("gate: free memory == reserve + free admits")
    mem[0] = 200.0
    if g.wait_reason("vpr", 80):                          # above the budget, nothing of ours running: starts
        fails.append(f"gate: a reserve above the budget must start when nothing runs ({g.wait_reason('vpr', 80)!r})")
    g.last_synth = now[0] - 59.5                          # (binary-exact steps)
    if g.wait_reason("synth", 6) != "synthesis spacing":
        fails.append("gate: a synthesis 59.5 s after the last one must wait")
    now[0] += 0.5
    if g.wait_reason("synth", 6):
        fails.append("gate: a synthesis 60 s after the last one starts")
    g.vpr_waiting = 1
    if g.wait_reason("synth", 6) != "a VPR waits" or g.wait_reason("elab", 1):
        fails.append("gate: a waiting VPR holds back a synthesis (only)")
    g.vpr_waiting = 0
    td = Path(tempfile.mkdtemp(prefix="gate_", dir=V3 / "work" / "tmp"))
    fn = g.reserve_fn("synth", td)
    a0 = fn()
    (td / "abc0.out").write_text("")
    if (a0, fn()) != (6.0, 1.5):
        fails.append(f"gate: synthesis reserve Parmys -> ABC on abc0.out: {(a0, fn())}")
    g.measured("vpr", 4.0)
    g.measured("vpr", None)
    r1 = g.reserve["vpr"]
    g.measured("vpr", 16.0)
    if (r1, g.reserve["vpr"]) != (12.0, 20.0):
        fails.append(f"gate: measured peaks raise a reserve to 1.25x, never lower it: {r1}, {g.reserve['vpr']}")
    g = Gate(budget=20, free=0, meminfo=lambda: 100.0, poll_s=0.02)   # a waiter released by a release
    order = []

    def second():
        with g.hold("vpr"):
            order.append("second")
    with g.hold("vpr"):
        t = threading.Thread(target=second)
        t.start()
        time.sleep(0.2)
        order.append("first ends")
    t.join(2)
    if order != ["first ends", "second"]:
        fails.append(f"gate: 12 + 12 > 20 must wait for the release: {order}")
    return fails


def _pool_tests() -> list:
    """Pool (v2's tests): a VPR seed submitted by a running task runs before the syntheses already waiting (one worker:
    deterministic order), FIFO within a priority, elaboration first; a raising task recorded and the rest still run;
    nested submissions awaited; no tasks -> returns; `jobs` tasks at once and never more; run() again works."""
    fails, order = [], []
    p = Pool(1)

    def row(n, kids):
        order.append(n)
        for k in kids:
            p.submit(VPR_FIRST, order.append, k)
    p.submit(SYNTH, row, "A", ["a11", "a13"])
    p.submit(SYNTH, row, "B", ["b11"])
    p.submit(ELAB, order.append, "E")
    p.submit(SYNTH, row, "C", [])
    p.run()
    if order != ["E", "A", "a11", "a13", "B", "b11", "C"]:
        fails.append(f"pool order: {order}")
    p, ran = Pool(2), []

    def bad():
        raise RuntimeError("task error")

    def nest(d):
        ran.append(d)
        if d < 3:
            p.submit(SYNTH, nest, d + 1)
    p.submit(SYNTH, bad)
    p.submit(SYNTH, nest, 0)
    p.run()
    if not (len(p.errors) == 1 and ran == [0, 1, 2, 3]):
        fails.append(f"pool error / nested tasks: errors {p.errors}, ran {ran}")
    p = Pool(3)
    p.run()                                              # no tasks: returns at once
    live, peak, lock = [0], [0], threading.Lock()

    def busy():
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.3)
        with lock:
            live[0] -= 1
    t0 = time.time()
    for _ in range(6):
        p.submit(SYNTH, busy)
    p.run()
    dt = time.time() - t0
    p.submit(SYNTH, busy)                                # the same pool again
    p.run()
    if not (peak[0] == 3 and dt < 1.2):
        fails.append(f"pool parallelism: peak {peak[0]} (want 3), 6 x 0.3 s took {dt:.2f} s (want ~0.6)")
    return fails


def _elab_check_tests(b: Build) -> list:
    """elab_check on a passing result and one change at a time: the top (another, two), a crossbar off by one, a memory
    missing / extra / one bit wider, an illegal port shape (3R1W, 1R3W), a ROM (legal), Yosys's memories absent."""
    fails = []
    mems = [dict(name=f"m{i}", size=w, width=bits, rd=1, wr=1) for i, ((w, bits), k) in
            enumerate(sorted(b.mems.items())) for _ in range(k)]
    ok = dict(rc=0, top=b.top, dpe=b.n_xbar, mems=mems)
    if elab_check(ok, b.top, b.n_xbar, b.mems):
        fails.append(f"elab_check on a passing result: {elab_check(ok, b.top, b.n_xbar, b.mems)}")
    m0 = mems[0]
    cases = [("top", dict(ok, top="other"), "top other"), ("two tops", dict(ok, top=[b.top, "x"]), "top ["),
             ("crossbar -1", dict(ok, dpe=b.n_xbar - 1), f"crossbars {b.n_xbar - 1}"),
             ("crossbar +1", dict(ok, dpe=b.n_xbar + 1), f"crossbars {b.n_xbar + 1}"),
             ("memory missing", dict(ok, mems=mems[1:]), f"fewer 1 x {m0['size']}x{m0['width']}"),
             ("memory extra", dict(ok, mems=mems + [m0]), f"more 1 x {m0['size']}x{m0['width']}"),
             ("one bit wider", dict(ok, mems=[dict(m0, width=m0["width"] + 1)] + mems[1:]),
              f"more 1 x {m0['size']}x{m0['width'] + 1}"),
             ("3R1W", dict(ok, mems=[dict(m0, rd=3)] + mems[1:]), "3R1W"),
             ("1R3W", dict(ok, mems=[dict(m0, wr=3)] + mems[1:]), "1R3W"),
             ("no memories", {k: v for k, v in ok.items() if k != "mems"}, "Yosys has more none")]
    for what, e, want in cases:
        got = "; ".join(elab_check(e, b.top, b.n_xbar, b.mems))
        if want not in got:
            fails.append(f"elab_check {what}: want {want!r} in {got!r}")
    if elab_check(dict(ok, mems=[dict(m0, rd=1, wr=0)] + mems[1:]), b.top, b.n_xbar, b.mems):
        fails.append("elab_check: a ROM (no write port) is legal")
    return fails


def _flow_tests(td: Path) -> list:
    """main() end to end on fake Yosys / run_vtr_flow.py / VPR (no VTR): build A passes everything (3 seeds: the VPR
    command line, the CSV line: mean Fmax, per-seed Fmax, largest counts, the run log); build B fails its elaboration
    (no synthesis, exit 1); a rerun calls no tool and keeps the CSV byte-identical; an unfinished seed runs alone
    again; a memory count above the prediction is counted and reported (mem_check); a VPR exit 1 is not counted and
    stays pending; a failed
    synthesis queues no VPR and runs again next time; another circuit is another folder; the real log and CSV are not
    touched; --timeout is rejected."""
    fails = []
    out, csvp, logp, calls, spec_p = td / "out", td / "vtr.csv", td / "run.log", td / "calls.txt", td / "spec.json"
    tools = td / "tools"
    tools.mkdir()
    for name, text in (("yosys", _FAKE_YOSYS), ("run_vtr_flow.py", _FAKE_FLOW), ("vpr", _FAKE_VPR)):
        (tools / name).write_text(text)
        (tools / name).chmod(0o755)
    every = {b.label: b for b in all_builds(out)}
    A, B, F = every["JetFormer-150P@128x128"], every["JetFormer-150P@128x64"], every["JetFormer-150P@256x64"]
    K = every["JetFormer-150P@512x64"]
    mlist = lambda b: [[w, bits, 1, 1] for (w, bits), k in sorted(b.mems.items()) for _ in range(k)]  # noqa: E731
    res = lambda b, **kw: dict(dict(io=100, clb=500, wc=b.n_xbar, memory=b.mem_blocks, dsp_top=0), **kw)  # noqa: E731
    spec = {A.top: dict(elab=dict(dpe=A.n_xbar, mems=mlist(A)),
                        vpr={"all": dict(fmax=110.0, res=res(A)), "13": dict(fmax=113.0, res=res(A, clb=517)),
                             "17": dict(fmax=120.0, res=res(A, clb=505))}),
            B.top: dict(elab=dict(dpe=B.n_xbar - 1, mems=mlist(B)), vpr={"all": dict(fmax=1.0, res=res(B))}),
            F.top: dict(elab=dict(dpe=F.n_xbar, mems=mlist(F)), synth_rc=1, vpr={"all": dict(fmax=1.0, res=res(F))}),
            K.top: dict(elab=dict(kill=True, dpe=0, mems=[]), vpr={"all": dict(fmax=1.0, res=res(K))})}
    spec_p.write_text(json.dumps(spec))
    os.environ.update(FAKE_SPEC=str(spec_p), FAKE_CALLS=str(calls))
    real = (dict(TOOLS), smoke.YOSYS)
    stamp = lambda p: (p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else None  # noqa: E731
    before = (stamp(CSV_PATH), stamp(RUN_LOG))
    TOOLS.update(flow=tools / "run_vtr_flow.py", python=tools / "no_python", vpr=tools / "vpr")
    smoke.YOSYS = tools / "yosys"
    base = ["--out", str(out), "--csv", str(csvp), "--log", str(logp), "--jobs", "3", "--synth-gap", "0", "--free-gib",
            "0"]
    ncalls = lambda: len(_read(calls).splitlines())  # noqa: E731

    def rows():
        with open(csvp, newline="") as f:
            return {r["build"]: r for r in csv.DictReader(f)}

    def stable():                                        # a rerun seed changes only its own wall time and peak
        return {k: {f: v for f, v in r.items() if f not in ("vpr_s", "peak_gib_vpr")} for k, r in rows().items()}
    try:
        rc = main(base + ["--build", A.label, "--build", B.name])
        cl = _read(calls).splitlines()
        want = sorted([f"yosys {A.top}", f"yosys {B.top}", f"synth {A.top}"] + [f"vpr {A.top} {s}" for s in SEEDS])
        got = sorted(" ".join(x.split(" ")[:3]) if x.startswith("vpr") else " ".join(x.split(" ")[:2]) for x in cl)
        if rc != 1 or got != want:
            fails.append(f"main, A + B: rc {rc} (want 1, B fails its elaboration), calls {got} (want {want})")
        vpr11 = next((x for x in cl if x.startswith(f"vpr {A.top} 11 ")), "").split(" ", 3)[-1].split(chr(31))
        wantv = [str(A.arch), A.top, "--circuit_file", str(A.netlist), "--route_chan_width", "300", "--sdc_file",
                 str(smoke.SDC), "--seed", "11"]
        if vpr11 != wantv:
            fails.append(f"VPR command line: {vpr11} != {wantv}")
        syn = next((x for x in cl if x.startswith("synth")), "").split(" ")[2:]
        if syn[-2:] != ["-end", "abc"] or syn[:2] != [str(A.circuit_path), str(A.arch)] or "-temp_dir" not in syn:
            fails.append(f"synthesis command line: {syn}")
        r = rows()
        ra, rb = r.get(A.name, {}), r.get(B.name, {})
        if not (ra.get("ok") == "1" and ra.get("fmax_mhz") == "114.333" and ra.get("seeds") == "11;13;17"
                and (ra.get("fmax_s11"), ra.get("fmax_s13"), ra.get("fmax_s17")) == ("110.0", "113.0", "120.0")
                and (ra.get("clb"), ra.get("memory"), ra.get("wc"), ra.get("io"), ra.get("dsp")) ==
                (str(517), str(A.mem_blocks), str(A.n_xbar), "100", "0") and ra.get("note") == ""
                and ra.get("elab") == "OK" and ra.get("synth_s") == "1.5" and ra.get("peak_gib_synth") == "3.0"
                and ra.get("mem_check") == "equal"
                and ra.get("hash") == A.digest[:12] and ra.get("N_E") == str(2 * A.d)):
            fails.append(f"CSV line of A: {ra}")
        if not (rb.get("ok") == "0" and "crossbars" in rb.get("elab", "") and "elaboration" in rb.get("note", "")):
            fails.append(f"CSV line of B (stopped at its elaboration check): {rb}")
        ll = _read(logp).splitlines()
        tm = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ")
        at = lambda t: next((i for i, x in enumerate(ll) if t in x), -1)  # noqa: E731
        need = [f"{A.label} elab start (pid ", f"{A.label} elab: OK", f"{B.label} elab: FAILED: crossbars",
                "elaboration: 1/2 builds OK; FAILED " + B.label, f"{A.label} synth start (pid ", f"{A.label} synth: OK",
                f"{A.label} seed 11: VPR start (pid ", f"{A.label} seed 11: VPR OK", "Fmax 110.0 MHz, clb 500",
                f"memory {A.mem_blocks}/{A.mem_blocks}", "; 0 error(s); exit 1"]
        if not (all(at(t) >= 0 for t in need) and all(tm.match(x) for x in ll) and at("synth start") > at("elaboration:")
                and ll[0].split(" ", 2)[2].startswith("=== VTR DSE: 2 builds")):
            fails.append(f"run log: missing {[t for t in need if at(t) < 0]} or order / timestamps: {ll}")
        text0, n0, st0 = csvp.read_text(), ncalls(), stable()
        rc = main(base + ["--build", A.label, "--build", B.name])     # rerun: every step skipped
        if ncalls() != n0 or csvp.read_text() != text0 or rc != 1 or f"{A.label} seed 17: finished, skipped" not in \
                _read(logp) or f"{A.label} elab: cached (Yosys), checked again: OK" not in _read(logp):
            fails.append(f"rerun: {ncalls() - n0} tool calls (want 0), CSV identical {csvp.read_text() == text0}, rc {rc}")
        ej = A.dir / "elab" / "elab.json"                              # the verdict is recomputed, never read back:
        ej.write_text(json.dumps(dict(json.loads(ej.read_text()), problems=["stale verdict"])))    # a stale one is
        if elab_verdict(A) != [] or main(base + ["--build", A.label, "--elab-only"]) != 0 or ncalls() != n0:  # ignored
            fails.append(f"a stored verdict must not stand: {elab_verdict(A)}, {ncalls() - n0} tool calls")
        (A.dir / "s13" / "flow.log").write_text("")                    # an unfinished seed runs alone again
        rc = main(base + ["--build", A.label])
        new = _read(calls).splitlines()[n0:]
        if rc != 0 or len(new) != 1 or not new[0].startswith(f"vpr {A.top} 13 ") or stable() != st0:
            fails.append(f"an unfinished seed: rc {rc}, calls {new}, CSV equal {stable() == st0}")
        spec[A.top]["vpr"]["17"]["res"]["memory"] = A.mem_blocks + 1   # memory above the prediction: counted,
        spec_p.write_text(json.dumps(spec))                             # reported in mem_check
        (A.dir / "s17" / "flow.log").write_text("")
        rc = main(base + ["--build", A.label])
        ra = rows()[A.name]
        if not (rc == 0 and ra["seeds"] == "11;13;17" and ra["fmax_mhz"] == "114.333" and ra["note"] == ""
                and ra["memory"] == str(A.mem_blocks + 1)
                and ra["mem_check"] == f"VPR {A.mem_blocks + 1} vs predicted {A.mem_blocks}") or not seed_done(A, 17):
            fails.append(f"memory above the prediction on s17 (counted, in mem_check): rc {rc}, {ra}")
        spec[A.top]["vpr"]["17"]["rc"] = 1                              # VPR exit 1: not counted, pending
        spec_p.write_text(json.dumps(spec))
        (A.dir / "s17" / "flow.log").write_text("")
        rc = main(base + ["--build", A.label])
        ra = rows()[A.name]
        if not (rc == 1 and "s17: VPR not finished OK" in ra["note"] and ra["fmax_s17"] == "" and not seed_done(A, 17)
                and "seed 17: VPR failed: VPR exit 1" in _read(logp)):
            fails.append(f"VPR exit 1 on s17: rc {rc}, {ra}")
        n1 = ncalls()                                                   # a failed synthesis: no VPR, runs again
        main(base + ["--build", F.label])
        main(base + ["--build", F.label])
        new = [x.split(" ")[0] for x in _read(calls).splitlines()[n1:]]
        rf = rows().get(F.name, {})
        if new != ["yosys", "synth", "synth"] or rf.get("note") != "synthesis not finished OK" or \
                f"{F.label} synth: FAILED (exit 1)" not in _read(logp):
            fails.append(f"failed synthesis: calls {new} (want yosys, synth, synth), {rf}")
        n2 = ncalls()                                                   # a killed elaboration: runs again
        main(base + ["--build", K.label, "--elab-only"])
        rc = main(base + ["--build", K.label, "--elab-only"])
        new = [x.split(" ")[0] for x in _read(calls).splitlines()[n2:]]
        rk = rows().get(K.name, {})
        if not (new == ["yosys", "yosys"] and rc == 1 and rk.get("elab") == "Yosys killed (signal 9)"
                and rk.get("note") == "elaboration did not finish"):
            fails.append(f"killed elaboration: calls {new} (want yosys twice), rc {rc}, {rk}")
        d2 = build_dir(out, A.name, A.text + "// an RTL edit\n")
        if d2 == A.dir or d2.parent != A.dir.parent or build_dir(out, A.name, A.text) != A.dir:
            fails.append("another circuit must get another folder, the same circuit the same")
        try:
            main(base + ["--timeout", "5"])
            fails.append("--timeout must be rejected (no timeout)")
        except SystemExit as e:
            if e.code != 2:
                fails.append(f"--timeout: exit {e.code}, want the argparse error 2")
        try:
            main(base + ["--build", "NoSuchShape@1x1"])
            fails.append("an unknown build must be rejected")
        except SystemExit:
            pass
    finally:
        TOOLS.clear()
        TOOLS.update(real[0])
        smoke.YOSYS = real[1]
    if (stamp(CSV_PATH), stamp(RUN_LOG)) != before:
        fails.append("the self-test touched the DSE's CSV or run log")
    return fails


def self_test() -> int:
    """The builds (72, unique names / tops / folders, the smoke point's counts), the elaboration check, the gate, the
    pool, and main() end to end on fake tools. Temporary files under work/tmp (left in place)."""
    fails = []
    (V3 / "work" / "tmp").mkdir(parents=True, exist_ok=True)
    td = Path(tempfile.mkdtemp(prefix="vtr_dse_test_", dir=V3 / "work" / "tmp"))
    every = all_builds(td / "out")
    if not (len(every) == 72 == len({b.name for b in every}) == len({b.top for b in every})
            == len({b.dir for b in every})):
        fails.append("72 builds with unique names, tops and folders")
    bt = next(b for b in every if b.label == "BERT-Tiny@128x128")
    if (bt.n_xbar, bt.mem_blocks, bt.params["N_OC"], bt.params["N_E"], bt.params["P_PROJ"]) != (278, 376, 6, 128, 1):
        fails.append(f"BERT-Tiny@128x128 (the smoke point): {bt.n_xbar} crossbars, {bt.mem_blocks} blocks, {bt.params}")
    if all_builds(td / "out")[5].digest != every[5].digest:
        fails.append("the circuit hash must be stable")
    fails += _elab_check_tests(bt)
    fails += _gate_tests()
    fails += _pool_tests()
    fails += _flow_tests(td)
    for f in fails:
        print("  FAIL", f)
    print(f"vtr_dse self-test ({_rel(td)}):", "ALL PASS" if not fails else f"{len(fails)} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
