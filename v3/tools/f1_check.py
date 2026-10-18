#!/usr/bin/env python3
"""f1_check.py — the F1 gate (PLAN phase F1; the check agreed with the user on 2026-10-14).

No full iverilog launch: the RTL copies are byte-identical to v2 (sha256 lock), so their v2 GATE 2 results carry over;
one 64x64 case per harness checks that the copied harnesses (edited path lines only) still run. Steps, in order:

  probe      the isolation probe flags a file open, an import and a path in a subprocess argument that lie in the
             repository outside v3/, and nothing on a clean command
  lock       tools/copies.py: every copy == its v2 source (read from the git tag v2-final) with its declared path lines
  twins      tools/check_twins.py --self-test, then the gate
  selftests  every copied self-test, unchanged (6 oracles, 10 kernels + attention_sim --values, the platform files)
  regen      every copied case generator writes its default corpus (GATE 1 runs inside each dump_case). The v2
             generators went with v2/; the generators and what they import are locked copies, and their corpora were
             compared with v2's byte for byte while v2 existed
  rtl        one 64x64 case per copied RTL harness (dpe, gemm, tiled DIMM, online softmax, o-carry) and the FIFO
             testbench

Every v3 command runs under the isolation probe (tools/probe_site): no v3 process may open, import or pass on a path
in the repository outside v3/. copies.py (which reads the sources from the git tag, through git) runs without it.

Outputs under v3/work/f1/<tag>/ (nothing is deleted; a new tag starts clean): logs/<name>.log, probe/<name>.jsonl,
regen/<corpus>/, rtl/, summary.txt.

Usage:
  python3 v3/tools/f1_check.py [--tag T] [--steps probe,lock,twins,selftests,regen,rtl] [--jobs 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
REPO = V3.parent
WORK = V3 / "work"
PY = sys.executable
STEPS = ("probe", "lock", "twins", "selftests", "regen", "rtl")

ORACLES = ("nldpe_ref", "gemm_ref", "dimm_ref", "softmax_ref", "softmax_online_ref", "attention_ref")
KERNELS = ("nldpe_sim", "pass_engine", "gemm_sim", "dimm_sim", "softmax_sim", "softmax_online_sim", "dimm_tile_sim",
           "ocarry_sim", "proj_sim", "attention_sim")
# corpus -> (generator, arguments besides --out)
CORPORA = {
    "dpe": ("v3/ip/xbar/tb/gen_cases.py", ()),
    "dpe_p29": ("v3/ip/xbar/tb/gen_cases.py", ("--p29",)),
    "gemm": ("v3/ip/gemm/tb/gen_gemm_cases.py", ()),
    "gemm_p29": ("v3/ip/gemm/tb/gen_gemm_cases.py", ("--p29",)),
    # no "dimm_tile": F3 rewrote the DIMM (its corpora come from ip/dimm/gen_cases.py and are checked by GATE 2)
    "softmax_online": ("v3/ip/softmax/tb/gen_softmax_online_cases.py", ()),
    "ocarry": ("v3/ip/softmax/tb/gen_ocarry_cases.py", ()),
}


class Ctx:
    def __init__(self, tag: str, jobs: int, phase: str = "f1"):
        self.root = WORK / phase / tag
        self.logs, self.probe = self.root / "logs", self.root / "probe"
        for d in (self.logs, self.probe, WORK / "tmp"):
            d.mkdir(parents=True, exist_ok=True)
        self.jobs = jobs
        self.lines: list[str] = []
        self.fails: list[str] = []
        self.probed: list[str] = []

    def say(self, s: str = "") -> None:
        print(s, flush=True)
        self.lines.append(s)

    def env(self, name: str | None) -> dict:
        e = {k: v for k, v in os.environ.items() if k not in ("V3_PROBE_LOG", "PYTHONPATH")}
        e["PYTHONDONTWRITEBYTECODE"] = "1"
        e["TMPDIR"] = str(WORK / "tmp")
        if name is not None:                      # probed
            e["PYTHONPATH"] = str(HERE / "probe_site")
            e["V3_PROBE_LOG"] = str(self.probe / f"{name}.jsonl")
            self.probed.append(name)
        return e

    def run(self, name: str, cmd: list, probe: bool = True, timeout: float = 3600, cwd: Path = REPO) -> tuple:
        log = self.logs / f"{name}.log"
        t0 = time.time()
        with open(log, "w") as fh:
            fh.write("$ " + " ".join(map(str, cmd)) + "\n")
            fh.flush()
            try:
                rc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=self.env(name if probe else None), stdout=fh,
                                    stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        text = log.read_text(errors="replace")
        last = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")
        return rc, time.time() - t0, log, last, text

    def rel(self, p: Path) -> str:
        return str(p.relative_to(REPO))


def probe_records(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------------------------------------------------
def step_probe(c: Ctx) -> None:
    out = REPO / "softmax_study"          # a repository folder outside v3/ whose gen_luts.py is safe to import
    # dirty: an open, an import, a path as an absolute plusarg and as a plusarg relative to a child cwd, all outside v3/
    dirty = ("import subprocess, sys\n"
             f"open({str(out / 'gen_luts.py')!r}).read()\n"
             f"sys.path.insert(0, {str(out)!r})\nimport gen_luts\n"
             f"subprocess.run(['true', '+VDIR={out}'])\n"
             f"subprocess.run(['true', '+VDIR=../paper'], cwd={str(out)!r})\n")
    # clean: v3 paths only, including the runners' forms (an absolute plusarg; one relative to a v3 child cwd)
    clean = ("import subprocess\n"
             f"open({str(V3 / 'README.md')!r}).read()\n"
             f"subprocess.run(['true', '+VDIR={V3 / 'tools'}', '+CASE=x'])\n"
             f"subprocess.run(['true', '+VDIR=../tools'], cwd={str(V3 / 'work')!r})\n")
    c.run("probe_dirty", [PY, "-c", dirty])
    c.run("probe_clean", [PY, "-c", clean])
    viol = [v for r in probe_records(c.probe / "probe_dirty.jsonl") for v in r["violations"]]
    kinds = {k for k in ("open", "import", "exec arg") if any(v.startswith(k) for v in viol)}
    n_exec = sum(v.startswith("exec arg") for v in viol)
    rec_clean = probe_records(c.probe / "probe_clean.jsonl")
    bad_clean = [v for r in rec_clean for v in r["violations"]]
    ok = kinds == {"open", "import", "exec arg"} and n_exec == 2 and rec_clean and not bad_clean
    c.probed.remove("probe_dirty")                # its violations are the test, not a failure
    c.say(f"  probe      {'PASS' if ok else 'FAIL'}  dirty command flagged {sorted(kinds)} ({n_exec} of 2 outside "
          f"plusargs); clean command (v3 plusargs, absolute and relative) {len(bad_clean)} violations")
    if not ok:
        c.fails.append("probe")


def step_lock(c: Ctx) -> None:
    rc, dt, log, last, _ = c.run("lock", [PY, HERE / "copies.py"], probe=False, timeout=600)
    c.say(f"  lock       {'PASS' if rc == 0 else 'FAIL'}  {last}  [{c.rel(log)}]")
    if rc != 0:
        c.fails.append("lock")


def step_twins(c: Ctx) -> None:
    for name, args in (("twins_selftest", ["--self-test"]), ("twins", [])):
        rc, dt, log, last, _ = c.run(name, [PY, HERE / "check_twins.py", *args], timeout=600)
        c.say(f"  {name:14s} {'PASS' if rc == 0 else 'FAIL'}  {last}  [{c.rel(log)}]")
        if rc != 0:
            c.fails.append(name)


def step_selftests(c: Ctx) -> None:
    tests = [(f"oracle_{o}", [V3 / "oracle" / f"{o}.py"]) for o in ORACLES]
    tests += [(f"kernel_{k}", [V3 / "oracle" / "v2_kernels" / f"{k}.py"]) for k in KERNELS]
    tests += [("kernel_attention_sim_values", [V3 / "oracle" / "v2_kernels" / "attention_sim.py", "--values"])]
    # the platform (simulator/platform/) is no longer a v2 copy: its prices are checked by the FC rows (F2)

    def one(t):
        name, argv = t
        return (name,) + c.run(name, [PY, *argv], timeout=7200)[:4]

    with ThreadPoolExecutor(max_workers=c.jobs) as ex:
        res = list(ex.map(one, tests))
    for name, rc, dt, log, last in res:
        c.say(f"  {name:30s} {'PASS' if rc == 0 else 'FAIL'} {dt:7.1f}s  {last[:90]}")
        if rc != 0:
            c.fails.append(name)
    c.say(f"  selftests: {sum(r[1] == 0 for r in res)}/{len(res)} PASS")


def _size(d: Path) -> tuple:
    """(number of files, bytes) under d."""
    files = [p for p in d.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def step_regen(c: Ctx) -> None:
    jobs = []
    for corpus, (gen, args) in CORPORA.items():
        out = c.root / "regen" / corpus
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{c.rel(out)} is not empty; use a new --tag")
        jobs.append((corpus, [PY, REPO / gen, "--out", out, *args]))

    def one(j):
        corpus, cmd = j
        return (corpus,) + c.run(f"regen_{corpus}", cmd, timeout=7200)[:4]

    with ThreadPoolExecutor(max_workers=c.jobs) as ex:
        res = list(ex.map(one, jobs))
    for corpus, rc, dt, log, last in res:
        n_files, n_bytes = _size(c.root / "regen" / corpus)
        man = c.root / "regen" / corpus / "manifest.txt"
        n_cases = len(man.read_text().split()) if man.exists() else 0
        ok = rc == 0 and n_files > 0
        c.say(f"  {corpus:15s} {'PASS' if ok else 'FAIL'}  {n_cases} cases, {n_files} files, {n_bytes / 1e6:.1f} MB; "
              f"rc {rc} ({dt:.0f}s)  [{c.rel(log)}]")
        if not ok:
            c.fails.append(f"regen_{corpus}")


def _one_case_dir(src_corpus: Path, dst: Path, pick) -> str:
    """Copy one case of a certified corpus into its own stimuli dir (manifest of one)."""
    names = [n for n in (src_corpus / "manifest.txt").read_text().split() if n]
    name = next(n for n in names if pick(n, json.loads((src_corpus / n / "case.json").read_text())))
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_corpus / name, dst / name, dirs_exist_ok=True)
    (dst / "manifest.txt").write_text(name + "\n")
    return name


def step_rtl(c: Ctx) -> None:
    rtl = c.root / "rtl"
    for m in ("xbar", "gemm", "dimm", "softmax", "lib"):
        (WORK / m).mkdir(parents=True, exist_ok=True)       # the runners' lock files live there
    regen = c.root / "regen"
    is64 = lambda cj: (cj.get("R"), cj.get("C")) == (64, 64)    # noqa: E731
    runs = [("rtl_dpe", [PY, V3 / "ip/xbar/tb/run_dpe_rtl.py", "--geoms", "64x64", "--ms", "2", "--modes", "0",
                         "--classes", "random", "--jobs", "1"]),
            ("rtl_gemm", [PY, V3 / "ip/gemm/tb/run_gemm_rtl.py", "--geoms", "64x64", "--ks", "64", "--ns", "64",
                          "--ms", "2", "--classes", "random", "--jobs", "1"])]
    rc, _, log, last, _ = c.run("rtl_softmax_gen", [PY, V3 / "ip/softmax/tb/gen_softmax_online_cases.py", "--out",
                                                    rtl / "softmax_online_stimuli", "--Ss", "32", "--Ns", "S",
                                                    "--BKV", "16", "--nExps", "2", "--classes", "random",
                                                    "--R", "64"])
    if rc != 0:
        c.fails.append("rtl_softmax_gen")
        c.say(f"  rtl_softmax_gen FAIL  {last}  [{c.rel(log)}]")
    runs.append(("rtl_softmax_online", [PY, V3 / "ip/softmax/tb/run_softmax_online_rtl.py", "--stimuli",
                                        rtl / "softmax_online_stimuli", "--jobs", "1", "--work",
                                        rtl / "softmax_online_work"]))
    o = _one_case_dir(regen / "ocarry", rtl / "ocarry_stimuli", lambda n, cj: n.startswith("random_") and is64(cj))
    runs.append(("rtl_ocarry", [PY, V3 / "ip/softmax/tb/run_ocarry_rtl.py", "--stimuli", rtl / "ocarry_stimuli",
                                "--jobs", "1", "--work", rtl / "ocarry_work"]))
    c.say(f"  cases: o-carry {o}")
    for name, cmd in runs:
        rc, dt, log, last, _ = c.run(name, cmd, timeout=3600)
        c.say(f"  {name:20s} {'PASS' if rc == 0 else 'FAIL'} {dt:6.1f}s  {last[:100]}  [{c.rel(log)}]")
        if rc != 0:
            c.fails.append(name)
    vvp = rtl / "tb_att_fifo.vvp"
    rc, dt, log, last, text = c.run("rtl_fifo_build", ["iverilog", "-g2005", "-o", vvp, V3 / "ip/lib/tb/tb_att_fifo.v",
                                                       V3 / "ip/lib/rtl/sim/att_fifo.v"], timeout=600)
    if rc == 0:
        rc, dt, log, last, text = c.run("rtl_fifo", ["vvp", "-n", vvp], timeout=600)
    ok = rc == 0 and "[tb_fifo] PASS" in text
    c.say(f"  {'rtl_fifo':20s} {'PASS' if ok else 'FAIL'} {dt:6.1f}s  {last[:100]}  [{c.rel(log)}]")
    if not ok:
        c.fails.append("rtl_fifo")


def isolation(c: Ctx) -> None:
    n_proc, bad = 0, []
    for name in c.probed:
        recs = probe_records(c.probe / f"{name}.jsonl")
        n_proc += len(recs)
        bad += [f"{name}: {v}" for r in recs for v in r["violations"]]
    c.say(f"  isolation  {'PASS' if not bad else 'FAIL'}  {len(c.probed)} probed commands, {n_proc} Python processes, "
          f"{len(bad)} accesses outside v3/")
    for b in bad[:10]:
        c.say(f"      {b}")
    if bad:
        c.fails.append("isolation")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    steps = a.steps.split(",")
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}")
    if "rtl" in steps and "regen" not in steps and not (WORK / "f1" / a.tag / "regen").exists():
        raise SystemExit("the rtl step takes its tile and o-carry cases from this tag's regen corpus: run regen first")
    c = Ctx(a.tag, a.jobs)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout
    iv = subprocess.run(["iverilog", "-V"], capture_output=True, text=True).stdout.splitlines()[:1]
    lock_sha = hashlib.sha256((HERE / "copies.lock").read_bytes()).hexdigest()[:16]
    c.say(f"F1 check {a.tag}: HEAD {head.strip()}, {iv[0] if iv else 'iverilog ?'}, python {sys.version.split()[0]}, "
          f"copies.lock {lock_sha}")
    t0 = time.time()
    for s in steps:
        c.say(f"[{s}]")
        globals()[f"step_{s}"](c)
    c.say("[isolation]")
    isolation(c)
    c.say(f"F1 check: {'ALL PASS' if not c.fails else 'FAIL: ' + ', '.join(c.fails)} ({time.time() - t0:.0f}s, "
          f"logs {c.rel(c.logs)})")
    (c.root / "summary.txt").write_text("\n".join(c.lines) + "\n")
    return 0 if not c.fails else 1


if __name__ == "__main__":
    sys.exit(main())
