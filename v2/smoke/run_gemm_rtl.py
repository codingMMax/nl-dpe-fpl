#!/usr/bin/env python3
"""run_gemm_rtl.py — Stage-2 GATE-2 harness for the v2 GEMM array.

Flow:
  1. `v2/smoke/gen_gemm_cases.py` — GATE-1-certified cases into
     `gemm_stimuli/` (sim == oracle values + §5.3 cycles, certified)
  2. expand each case into `$readmemh` vectors (`<case>/vectors/*.hex`)
  3. compile `v2/tb/tb_gemm_top.v` + `v2/rtl/gemm_top.v` once per geometry
  4. run vvp per case; the TB gates the dual compare (wide `S_col` + lane
     bytes), readiness and drain integrity in-TB
  5. harness gate: every case PASSes, Δ_impl is constant per geometry
     (across M), and T_steady steps match (M2−M1)·T_steady; report + log

Usage:
  python3 v2/smoke/run_gemm_rtl.py                        # all generated cases
  python3 v2/smoke/run_gemm_rtl.py --stage 1A             # V=1,H=1 only
  python3 v2/smoke/run_gemm_rtl.py --no-gen               # reuse manifest
  python3 v2/smoke/run_gemm_rtl.py --rtl /path/to/gemm_top.v
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
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "v2" / "smoke"
STIMULI = SMOKE / "gemm_stimuli"
LOGS = SMOKE / "logs"
TB = REPO / "v2" / "tb" / "tb_gemm_top.v"
DEFAULT_RTL = REPO / "v2" / "rtl" / "gemm_top.v"
# gemm_top instantiates the certified primitive; it must be compiled together
PRIM_RTL = REPO / "v2" / "rtl" / "dpe_nldpe.v"
GEN_CASES = SMOKE / "gen_gemm_cases.py"

RESULT_RE = re.compile(
    r"\[tb_gemm_top\] (PASS|FAIL).*?"
    r"measured=(-?\d+) expected=(-?\d+) delta=(-?\d+)")
ERRCOUNT_RE = re.compile(r"errors=(\d+) ready_err=(\d+) cycle_err=(\d+)")

PROBES = ("S_col", "out_valid")


@dataclass
class Case:
    name: str
    path: Path
    k: int
    n: int
    r: int
    c: int
    buf: int
    p: int
    m: int
    expected: int
    status: str = "?"
    measured: int = -1
    delta: int = 0
    errors: int = -1
    detail: str = ""

    @property
    def geom(self) -> tuple[int, int, int, int]:
        return (self.k, self.n, self.r, self.c)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def derive_vh(k: int, n: int, r: int, c: int) -> tuple[int, int]:
    return -(-k // r), -(-n // c)


def t_steady(k: int, n: int, r: int, c: int, buf: int, p: int) -> int:
    """§5.3: max(LOAD+P, COMPUTE, OUTPUT+1, RED_PERIOD=1); L_w is fill only."""
    load = -(-r * 8 // buf)
    out = -(-c * 8 // buf)
    return max(load + p, p + 2, out + 1, 1)


def expand_vectors(case_dir: Path, meta: dict) -> Path:
    """Convert dump_case outputs into $readmemh files under <case>/vectors."""
    vdir = case_dir / "vectors"
    vdir.mkdir(exist_ok=True)
    M, K, N = int(meta["M"]), int(meta["K"]), int(meta["N"])
    R, C = int(meta["R"]), int(meta["C"])
    V, H = derive_vh(K, N, R, C)

    w = [int(ln, 16) & 0xFF
         for ln in (case_dir / "weights.mem").read_text().split()]
    assert len(w) == V * H * R * C, f"{case_dir}: weights {len(w)} != {V*H*R*C}"
    (vdir / "w.hex").write_text("".join(f"{b:02x}\n" for b in w))

    act = [ln.strip() for ln in (case_dir / "act.mem").read_text().splitlines()
           if ln.strip()]
    (vdir / "act.hex").write_text("".join(f"{int(x, 16):010x}\n" for x in act))

    # wide partial S: [M, N] int32 -> [M, H*C] with zero padding columns (A5)
    S = np.load(case_dir / "expected_psum.npz")["S"].astype(np.int32)
    assert S.shape == (M, N), f"{case_dir}: expected_psum {S.shape} != {(M, N)}"
    Sp = np.zeros((M, H * C), dtype=np.int32)
    Sp[:, :N] = S
    (vdir / "expS.hex").write_text(
        "".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in Sp.reshape(-1)))

    # lane bytes: M lines of H*C hex pairs -> one 2-hex byte per line
    out_lines = [ln.strip() for ln in
                 (case_dir / "expected_out.mem").read_text().splitlines()
                 if ln.strip()]
    assert len(out_lines) == M, f"{case_dir}: expected_out passes {len(out_lines)} != {M}"
    bytes_out: list[int] = []
    for ln in out_lines:
        assert len(ln) == 2 * H * C, f"{case_dir}: expected_out line len {len(ln)} != {2*H*C}"
        bytes_out.extend(int(ln[i:i+2], 16) for i in range(0, len(ln), 2))
    (vdir / "expo.hex").write_text("".join(f"{b:02x}\n" for b in bytes_out))
    return vdir


def load_cases(stimuli: Path) -> list[Case]:
    """Cases of the latest generation run (manifest.txt), not the whole corpus."""
    manifest = stimuli / "manifest.txt"
    if manifest.exists():
        dirs = [stimuli / n for n in manifest.read_text().split()]
    else:
        dirs = sorted(p.parent for p in stimuli.glob("*/case.json"))
    cases = []
    for d in dirs:
        cj = d / "case.json"
        if not cj.exists():
            continue
        meta = json.loads(cj.read_text())
        cases.append(Case(
            name=d.name, path=d,
            k=int(meta["K"]), n=int(meta["N"]), r=int(meta["R"]),
            c=int(meta["C"]), buf=int(meta["BUF"]), p=int(meta["P"]),
            m=int(meta["M"]), expected=int(meta["used_cycles"]),
        ))
    return cases


def compile_for(case: Case, rtl: Path, out_dir: Path) -> tuple[Path | None, str]:
    bin_path = out_dir / f"gemm_{case.k}x{case.n}_{case.r}x{case.c}_{case.p}.vvp"
    if bin_path.exists():
        return bin_path, ""
    cmd = [
        "iverilog", "-g2005", "-o", str(bin_path),
        f"-DK_TB={case.k}", f"-DN_TB={case.n}",
        f"-DR_TB={case.r}", f"-DC_TB={case.c}",
        f"-DBUF_TB={case.buf}", f"-DP_TB={case.p}",
        str(TB), str(rtl), str(PRIM_RTL),
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
    odir = case.path / "observed"
    odir.mkdir(parents=True, exist_ok=True)
    p = run(["vvp", str(bin_path), f"+M={case.m}",
             f"+VDIR={vdir.resolve()}", f"+ODIR={odir.resolve()}",
             f"+CASE={case.name}"],
            cwd=REPO, timeout=timeout)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"gemm_{case.name}.log").write_text(
        f"# {case.name} M={case.m} K={case.k} N={case.n} "
        f"{case.r}x{case.c} BUF={case.buf} P={case.p}\n" + p.stdout
        + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else ""))
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
    ap.add_argument("--ks", default=None, help="K list for generation")
    ap.add_argument("--ns", default=None, help="N list for generation")
    ap.add_argument("--geoms", default=None, help="R x C list for generation")
    ap.add_argument("--ms", default=None, help="M list for generation")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--stage", choices=("1A", "1B", "1C", "1D", "all"), default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--rtl", default=str(DEFAULT_RTL))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("v2 GEMM array RTL cross-check (Stage 2)")
    print(f"  rtl     : {rel(Path(args.rtl))}")
    print(f"  stage   : {args.stage}")
    print("=" * 78)

    lock_fd = None
    lock_path = SMOKE / ".gemm_rtl.lock"
    if not args.no_gen:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print(f"another run is active ({rel(lock_path)}); "
                  f"remove the lock file if stale")
            return 2
    try:
        return _execute(args)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)


def _execute(args: argparse.Namespace) -> int:
    if not args.no_gen:
        cmd = [sys.executable, str(GEN_CASES), "--out", str(STIMULI),
               "--stage", args.stage, "--seed", str(args.seed)]
        for flag, val in (("--ks", args.ks), ("--ns", args.ns),
                          ("--geoms", args.geoms), ("--ms", args.ms),
                          ("--classes", args.classes)):
            if val is not None:
                cmd += [flag, val]
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
            "M": case.m, "K": case.k, "N": case.n, "R": case.r, "C": case.c,
        })

    rtl = Path(args.rtl).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="v2gemm_"))
    bins: dict[tuple[int, int, int, int], Path] = {}
    for case in cases:
        if case.geom in bins:
            continue
        bin_path, err = compile_for(case, rtl, tmp)
        if bin_path is None:
            print(f"\nCOMPILE FAILED ({case.k}x{case.n} rlt={case.r}x{case.c}):\n{err}")
            print("\nThe RTL probe contract (`S_col`, `out_valid`) or the port "
                  "surface is likely not implemented yet.")
            return 3
        bins[case.geom] = bin_path
    print(f"compiled {len(bins)} geometry binaries")

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(lambda c: run_case(bins[c.geom], c, args.timeout), cases))

    # -- report ---------------------------------------------------------------
    print()
    hdr = (f"{'case':<44} {'P':>3} {'status':>6} {'measured':>8} "
           f"{'expected':>8} {'delta':>5} {'errs':>5}")
    print(hdr)
    print("-" * len(hdr))
    for case in cases:
        print(f"{case.name:<44} {case.m:>3} {case.status:>6} "
              f"{case.measured:>8} {case.expected:>8} {case.delta:>5} {case.errors:>5}"
              + (f"  {case.detail[:90]}" if case.detail else ""))

    failures = [c for c in cases if c.status != "PASS"]
    deltas: dict[tuple[int, int, int, int], set[int]] = {}
    for case in cases:
        if case.status == "PASS":
            deltas.setdefault(case.geom, set()).add(case.delta)

    print()
    ok = True
    if failures:
        ok = False
        print(f"GATE FAIL: {len(failures)} case(s) not PASS")
    for geom, ds in sorted(deltas.items()):
        k, n, r, c = geom
        if len(ds) == 1:
            print(f"GATE OK  : K={k} N={n} {r}x{c} delta_impl = {ds.pop()} (constant)")
        else:
            ok = False
            print(f"GATE FAIL: K={k} N={n} {r}x{c} delta_impl varies: {sorted(ds)}")
    if not deltas:
        ok = False
        print("GATE FAIL: no PASSing cases")

    # -- T_steady calibration gate -------------------------------------------
    by_geom: dict[tuple[int, int, int, int], dict[int, set[int]]] = {}
    for case in cases:
        if case.status == "PASS":
            by_geom.setdefault(case.geom, {}).setdefault(case.m, set()).add(case.measured)
    if not by_geom:
        ok = False
    for geom, by_m in sorted(by_geom.items()):
        k, n, r, c = geom
        ts = t_steady(k, n, r, c, cases[0].buf, cases[0].p)
        ms = sorted(by_m)
        for m, vals in sorted(by_m.items()):
            if len(vals) > 1:
                ok = False
                print(f"GATE FAIL: K={k} N={n} {r}x{c} measured varies within M={m}: {sorted(vals)}")
        for m1, m2 in zip(ms, ms[1:]):
            v1, v2 = next(iter(by_m[m1])), next(iter(by_m[m2]))
            step, want = v2 - v1, (m2 - m1) * ts
            if step == want:
                print(f"GATE OK  : K={k} N={n} {r}x{c} step M={m1}->{m2} = {step} "
                      f"(T_steady={ts} x {m2 - m1})")
            else:
                ok = False
                print(f"GATE FAIL: K={k} N={n} {r}x{c} step M={m1}->{m2} = {step}, "
                      f"expected {want} (T_steady={ts} x {m2 - m1})")
        if len(ms) == 1:
            print(f"GATE NOTE: K={k} N={n} {r}x{c} only M={ms[0]} present; no step to calibrate")

    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = "\n".join(
        f"{c.name} M={c.m} {c.status} measured={c.measured} "
        f"expected={c.expected} delta={c.delta}" for c in cases) + "\n"
    run_log = LOGS / f"rtl_smoke_gemm_{stamp}.log"
    run_log.write_text(summary)
    (LOGS / "gemm_latest.log").write_text(summary)
    print(f"\nlogs: {rel(run_log)}  (per-case TB logs in {rel(LOGS)}/)")
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
