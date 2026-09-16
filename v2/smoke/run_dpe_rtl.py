#!/usr/bin/env python3
"""run_dpe_rtl.py — Stage 1.5 RTL cross-check harness for the v2 NL-DPE.

Flow:
  1. `v2/smoke/check_interface.py` — interface freeze gate (hard fail)
  2. `v2/smoke/gen_cases.py` — spec §9 stimulus classes into `stimuli/`
  3. expand each case into `$readmemh` vectors (`<case>/vectors/*.hex`)
  4. compile `v2/tb/tb_dpe_nldpe.v` + `v2/rtl/dpe_nldpe.v` once per geometry
  5. run vvp per case; dual-compare (hierarchical int32 y + drained stream),
     readiness, structural COMPUTE span, and cycle formula are gated in-TB
  6. harness gate: every case PASSes and Δ_impl is constant within a geometry
     (`measured − (T_fill + (M−1)·T_steady)`); report table + log

Usage:
  python3 v2/smoke/run_dpe_rtl.py                     # both geoms, M∈{1,2}, all modes
  python3 v2/smoke/run_dpe_rtl.py --quick             # identity+random, M∈{1,2}, mode 0
  python3 v2/smoke/run_dpe_rtl.py --ms 1,2,4,8 --classes identity,random --modes 0
  python3 v2/smoke/run_dpe_rtl.py --no-gen            # reuse existing stimuli/
  python3 v2/smoke/run_dpe_rtl.py --rtl /path/to/dpe_nldpe.v
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "v2" / "smoke"
STIMULI = SMOKE / "stimuli"
RESULTS = SMOKE / "results"
TB = REPO / "v2" / "tb" / "tb_dpe_nldpe.v"
DEFAULT_RTL = REPO / "v2" / "rtl" / "dpe_nldpe.v"
CHECK_IF = SMOKE / "check_interface.py"
GEN_CASES = SMOKE / "gen_cases.py"

RESULT_RE = re.compile(
    r"\[tb_dpe_nldpe\] (PASS|FAIL).*?"
    r"measured=(-?\d+) expected=(-?\d+) delta=(-?\d+)")
ERRCOUNT_RE = re.compile(r"errors=(\d+) ready_err=(\d+) span_err=(\d+) cycle_err=(\d+)")

PROBES = ("state", "acc", "acam_fire", "drain_valid")


@dataclass
class Case:
    name: str
    path: Path
    r: int
    c: int
    buf: int
    p: int
    m: int
    mode: int
    expected: int
    status: str = "?"
    measured: int = -1
    delta: int = 0
    errors: int = -1
    detail: str = ""

    @property
    def geom(self) -> tuple[int, int, int, int]:
        return (self.r, self.c, self.buf, self.p)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def t_steady(r: int, c: int, buf: int, p: int) -> int:
    """§5.3 T_steady = max(LOAD_CYC + P, COMPUTE_CYC, OUTPUT_CYC + 1)."""
    load = -(-r * 8 // buf)
    out = -(-c * 8 // buf)
    return max(load + p, p + 2, out + 1)


def expand_vectors(case_dir: Path, meta: dict) -> Path:
    """Convert dump_case outputs into $readmemh files under <case>/vectors."""
    vdir = case_dir / "vectors"
    vdir.mkdir(exist_ok=True)
    R, C, M = int(meta["R"]), int(meta["C"]), int(meta["M"])

    w = [int(line, 16) & 0xFF
         for line in (case_dir / "weights.mem").read_text().split()]
    assert len(w) == R * C, f"{case_dir}: weights {len(w)} != {R*C}"
    (vdir / "w.hex").write_text("".join(f"{b:02x}\n" for b in w))

    act = [line.strip() for line in (case_dir / "act.mem").read_text().splitlines()
           if line.strip()]
    (vdir / "act.hex").write_text("".join(f"{int(x, 16):010x}\n" for x in act))

    y = np.load(case_dir / "expected_y.npz")["y"].astype(np.int32)
    assert y.shape == (M, C), f"{case_dir}: expected_y {y.shape} != {(M, C)}"
    (vdir / "expy.hex").write_text(
        "".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in y.reshape(-1)))

    out_lines = [ln.strip() for ln in
                 (case_dir / "expected_out.mem").read_text().splitlines()
                 if ln.strip()]
    assert len(out_lines) == M, f"{case_dir}: expected_out passes {len(out_lines)} != {M}"
    bytes_out: list[int] = []
    for ln in out_lines:
        assert len(ln) == 2 * C, f"{case_dir}: expected_out line len {len(ln)} != {2*C}"
        bytes_out.extend(int(ln[i:i+2], 16) for i in range(0, len(ln), 2))
    (vdir / "expo.hex").write_text("".join(f"{b:02x}\n" for b in bytes_out))
    return vdir


def load_cases(stimuli: Path) -> list[Case]:
    cases = []
    for cj in sorted(stimuli.glob("*/case.json")):
        meta = json.loads(cj.read_text())
        cases.append(Case(
            name=cj.parent.name, path=cj.parent,
            r=int(meta["R"]), c=int(meta["C"]), buf=int(meta["BUF"]),
            p=int(meta["P"]), m=int(meta["M"]),
            mode=int(meta["mode"]),
            expected=int(meta["used_cycles"]),
        ))
    return cases


def compile_for(case: Case, rtl: Path, out_dir: Path) -> tuple[Path | None, str]:
    bin_path = out_dir / f"dpe_{case.r}x{case.c}_{case.p}.vvp"
    if bin_path.exists():
        return bin_path, ""
    cmd = [
        "iverilog", "-g2005", "-o", str(bin_path),
        f"-DR_TB={case.r}", f"-DC_TB={case.c}",
        f"-DBUF_TB={case.buf}", f"-DP_TB={case.p}",
        str(TB), str(rtl),
    ]
    p = run(cmd, cwd=REPO)
    if p.returncode != 0:
        missing = [pr for pr in PROBES if pr in p.stderr]
        hint = (" (RTL probe contract missing: " + ", ".join(missing) + ")"
                if missing else "")
        return None, p.stderr.strip() + hint
    return bin_path, ""


def run_case(bin_path: Path, case: Case, timeout: int) -> None:
    vdir = case.path / "vectors"
    p = run(["vvp", str(bin_path), f"+M={case.m}", f"+MODE={case.mode}",
             f"+VDIR={vdir.resolve()}", f"+CASE={case.name}"],
            cwd=REPO, timeout=timeout)
    m = RESULT_RE.search(p.stdout)
    if not m:
        case.status = "ERROR"
        case.detail = (p.stdout + p.stderr).strip()[-400:]
        return
    case.status = m.group(1)
    case.measured = int(m.group(2))
    case.delta = int(m.group(4))
    em = ERRCOUNT_RE.search(p.stdout)
    case.errors = sum(int(g) for g in em.groups()) if em else 0
    if case.status == "FAIL" and not case.detail:
        fails = [ln for ln in p.stdout.splitlines() if "MISMATCH" in ln or "ERROR" in ln]
        case.detail = " | ".join(fails[:3])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geoms", default=None)
    ap.add_argument("--ms", default=None)
    ap.add_argument("--modes", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--rtl", default=str(DEFAULT_RTL))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if args.quick:
        geoms = args.geoms or "256x256"
        ms = args.ms or "1,2"
        modes = args.modes or "0"
        classes = args.classes or "identity,random"
    else:
        geoms = args.geoms or "256x256,256x512"
        ms = args.ms or "1,2"
        modes = args.modes or "0,1,2,3"
        classes = args.classes or "identity,random,extremes"

    print("=" * 78)
    print("v2 NL-DPE RTL cross-check (Stage 1.5)")
    print(f"  rtl     : {rel(Path(args.rtl))}")
    print(f"  geoms   : {geoms}")
    print(f"  M       : {ms}")
    print(f"  modes   : {modes}")
    print(f"  classes : {classes}")
    print("=" * 78)

    p = run([sys.executable, str(CHECK_IF)], cwd=REPO)
    print(p.stdout.strip())
    if p.returncode != 0:
        print("check_interface FAILED — refusing to run the cross-check.")
        return 2

    lock_fd = None
    lock_path = SMOKE / ".dpe_rtl.lock"
    if not args.no_gen:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print(f"another run is active ({rel(lock_path)}); "
                  f"remove the lock file if stale")
            return 2
    try:
        return _execute(args, geoms, ms, modes, classes)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)


def _execute(args: argparse.Namespace, geoms: str, ms: str,
             modes: str, classes: str) -> int:
    if not args.no_gen:
        if STIMULI.exists():
            shutil.rmtree(STIMULI)
        cmd = [sys.executable, str(GEN_CASES), "--out", str(STIMULI),
               "--geoms", geoms, "--ms", ms, "--modes", modes,
               "--classes", classes, "--seed", str(args.seed)]
        p = run(cmd, cwd=REPO)
        if p.returncode != 0:
            print(p.stdout[-2000:])
            print(p.stderr[-2000:])
            return 2
    cases = load_cases(STIMULI)
    if not cases:
        print(f"no cases under {rel(STIMULI)}")
        return 2
    print(f"cases: {len(cases)}")

    for case in cases:
        expand_vectors(case.path, {
            "R": case.r, "C": case.c, "BUF": case.buf, "P": case.p,
            "M": case.m,
        })

    rtl = Path(args.rtl).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="v2dpe_", dir="/tmp/opencode"))
    bins: dict[tuple[int, int, int, int], Path] = {}
    for case in cases:
        if case.geom in bins:
            continue
        bin_path, err = compile_for(case, rtl, tmp)
        if bin_path is None:
            print(f"\nCOMPILE FAILED ({case.r}x{case.c}):\n{err}")
            print("\nThe RTL TODO blocks (probe contract) are likely not implemented yet.")
            return 3
        bins[case.geom] = bin_path
    print(f"compiled {len(bins)} geometry binaries")

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(lambda c: run_case(bins[c.geom], c, args.timeout), cases))

    # -- report ---------------------------------------------------------------
    print()
    hdr = f"{'case':<44} {'P':>3} {'status':>6} {'measured':>8} {'expected':>8} {'delta':>5} {'errs':>5}"
    print(hdr)
    print("-" * len(hdr))
    for case in cases:
        print(f"{case.name:<44} {case.m:>3} {case.status:>6} "
              f"{case.measured:>8} {case.expected:>8} {case.delta:>5} {case.errors:>5}"
              + (f"  {case.detail[:90]}" if case.detail else ""))

    failures = [c for c in cases if c.status != "PASS"]
    deltas: dict[tuple[int, int], set[int]] = {}
    for case in cases:
        if case.status == "PASS":
            deltas.setdefault((case.r, case.c), set()).add(case.delta)

    print()
    ok = True
    if failures:
        ok = False
        print(f"GATE FAIL: {len(failures)} case(s) not PASS")
    for geom, ds in sorted(deltas.items()):
        if len(ds) == 1:
            print(f"GATE OK  : {geom[0]}x{geom[1]} delta_impl = {ds.pop()} (constant)")
        else:
            ok = False
            print(f"GATE FAIL: {geom[0]}x{geom[1]} delta_impl varies: {sorted(ds)}")
    if not deltas:
        ok = False
        print("GATE FAIL: no PASSing cases")

    # -- T_steady calibration gate (the multi-pass pipeline model) -----------
    # measured(M) = T_fill + (M-1)*T_steady + delta, so per geometry:
    #   (a) all PASSing cases with the same M must measure the same, and
    #   (b) measured(m2) - measured(m1) == (m2 - m1) * T_steady.
    by_geom: dict[tuple[int, int, int, int], dict[int, set[int]]] = {}
    for case in cases:
        if case.status == "PASS":
            by_geom.setdefault(case.geom, {}).setdefault(case.m, set()).add(case.measured)
    if not by_geom:
        ok = False
    for geom, by_m in sorted(by_geom.items()):
        r, c, buf, p = geom
        ts = t_steady(r, c, buf, p)
        ms = sorted(by_m)
        for m, vals in sorted(by_m.items()):
            if len(vals) > 1:
                ok = False
                print(f"GATE FAIL: {r}x{c} measured varies within M={m}: {sorted(vals)}")
        for m1, m2 in zip(ms, ms[1:]):
            v1, v2 = next(iter(by_m[m1])), next(iter(by_m[m2]))
            step, want = v2 - v1, (m2 - m1) * ts
            if step == want:
                print(f"GATE OK  : {r}x{c} step M={m1}->{m2} = {step} "
                      f"(T_steady={ts} x {m2 - m1})")
            else:
                ok = False
                print(f"GATE FAIL: {r}x{c} step M={m1}->{m2} = {step}, "
                      f"expected {want} (T_steady={ts} x {m2 - m1})")
        if len(ms) == 1:
            print(f"GATE NOTE: {r}x{c} only M={ms[0]} present; no step to calibrate")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "dpe_nldpe_rtl_smoke.log", "w") as f:
        f.write("\n".join(
            f"{c.name} M={c.m} mode={c.mode} {c.status} "
            f"measured={c.measured} expected={c.expected} delta={c.delta}"
            for c in cases) + "\n")
    print(f"\nlog: {rel(RESULTS / 'dpe_nldpe_rtl_smoke.log')}")
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
