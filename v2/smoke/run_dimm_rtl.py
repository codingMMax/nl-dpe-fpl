#!/usr/bin/env python3
"""run_dimm_rtl.py — Stage-4 GATE-2 harness for the v2 DIMM (`dimm_top`).

Flow:
  1. `v2/smoke/gen_dimm_cases.py` — GATE-1-certified cases into
     `dimm_stimuli/` (sim == oracle values, issued pass counts, cycle model)
  2. expand each case into `$readmemh` vectors (`<case>/vectors/*.hex`)
  3. compile `v2/tb/tb_dimm_top.v` + `v2/rtl/dimm_top.v` once per full
     parameter set (M,N,K,R,C,BUF,P,N_A,N_B,N_E)
  4. run vvp per case; the TB gates the staged compare (la_q / lb_q / acc_q /
     C stream), readiness and drain integrity in-TB
  5. harness gate: every case PASSes and Δ_impl is constant per
     (shape, n_E) across classes; report + log

Δ_impl = measured − (model.total + serialize_cycles): the true-overlap skew
(producer→farm handoff + last drain + serializer start), reported, not hidden.

Usage:
  python3 v2/smoke/run_dimm_rtl.py                     # full 60-case sweep
  python3 v2/smoke/run_dimm_rtl.py --nEs 4             # one n_E slice
  python3 v2/smoke/run_dimm_rtl.py --no-gen            # reuse manifest
  python3 v2/smoke/run_dimm_rtl.py --rtl /path/to/dimm_top.v
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
STIMULI = SMOKE / "dimm_stimuli"
LOGS = SMOKE / "logs"
TB = REPO / "v2" / "tb" / "tb_dimm_top.v"
DEFAULT_RTL = REPO / "v2" / "rtl" / "dimm_top.v"
PRIM_RTL = REPO / "v2" / "rtl" / "dpe_nldpe.v"
GEN_CASES = SMOKE / "gen_dimm_cases.py"

RESULT_RE = re.compile(
    r"\[tb_dimm_top\] (PASS|FAIL).*?"
    r"measured=(-?\d+) expected=(-?\d+) delta=(-?\d+)"
    r" span_A=(-?\d+) span_B=(-?\d+) span_F=(-?\d+) fill=(-?\d+) ser=(-?\d+)"
    r" wins_A=(-?\d+) wins_B=(-?\d+) wins_F=(-?\d+)")
ERRCOUNT_RE = re.compile(
    r"errors=(\d+) ready_err=(\d+) cycle_err=(\d+)")

PROBES = ("la_q", "lb_q", "acc_q")


@dataclass
class Case:
    name: str
    path: Path
    m: int
    n: int
    k: int
    r: int
    c: int
    buf: int
    p: int
    na: int
    nb: int
    ne: int
    model: int          # used_cycles (overlapped pool/farm + fill)
    serialize: int      # M*N C-serializer words
    t_start: int = 0    # case.json producer->farm fill
    t_a: int = 0
    t_b: int = 0
    t_e: int = 0
    status: str = "?"
    measured: int = -1
    delta: int = 0
    errors: int = -1
    # observed per-stage spans (TB)
    span_a: int = -1
    span_b: int = -1
    span_f: int = -1
    fill: int = -1
    ser: int = -1
    wins_a: int = -1
    wins_b: int = -1
    wins_f: int = -1
    # expected issued-window counts (case.json plan)
    passes_a: int = 0
    passes_b: int = 0
    passes_e: int = 0
    detail: str = ""

    @property
    def geom(self) -> tuple[int, ...]:
        return (self.m, self.n, self.k, self.r, self.c,
                self.buf, self.p, self.na, self.nb, self.ne)

    @property
    def expected(self) -> int:
        return self.model + self.serialize

    @property
    def shape_ne(self) -> tuple[int, ...]:
        return (self.m, self.n, self.k, self.r, self.c, self.ne)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def expand_vectors(case_dir: Path, meta: dict) -> Path:
    """Convert dump_case outputs into $readmemh files under <case>/vectors."""
    vdir = case_dir / "vectors"
    vdir.mkdir(exist_ok=True)
    M, N, K = int(meta["M"]), int(meta["N"]), int(meta["K"])

    a_words = [ln.strip() for ln in (case_dir / "a.mem").read_text().splitlines()
               if ln.strip()]
    (vdir / "a.hex").write_text("".join(f"{int(x, 16):010x}\n" for x in a_words))
    b_words = [ln.strip() for ln in (case_dir / "b.mem").read_text().splitlines()
               if ln.strip()]
    (vdir / "b.hex").write_text("".join(f"{int(x, 16):010x}\n" for x in b_words))

    la = np.load(case_dir / "expected_la.npz")["la"].astype(np.int8)
    assert la.shape == (K, M), f"{case_dir}: expected_la {la.shape} != {(K, M)}"
    (vdir / "expla.hex").write_text(
        "".join(f"{int(b) & 0xFF:02x}\n" for b in la.reshape(-1)))
    lb = np.load(case_dir / "expected_lb.npz")["lb"].astype(np.int8)
    assert lb.shape == (K, N), f"{case_dir}: expected_lb {lb.shape} != {(K, N)}"
    (vdir / "explb.hex").write_text(
        "".join(f"{int(b) & 0xFF:02x}\n" for b in lb.reshape(-1)))

    acc = np.load(case_dir / "expected_acc.npz")["acc"].astype(np.int32)
    assert acc.shape == (M, N), f"{case_dir}: expected_acc {acc.shape} != {(M, N)}"
    (vdir / "expacc.hex").write_text(
        "".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in acc.reshape(-1)))

    c_lines = [ln.strip() for ln in
               (case_dir / "expected_c.mem").read_text().splitlines()
               if ln.strip()]
    assert len(c_lines) == M * N, f"{case_dir}: expected_c {len(c_lines)} != {M*N}"
    (vdir / "expc.hex").write_text(
        "".join(f"{int(x, 16):08x}\n" for x in c_lines))
    return vdir


def load_cases(stimuli: Path) -> list[Case]:
    """Cases of the latest generation run (manifest.txt)."""
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
            m=int(meta["M"]), n=int(meta["N"]), k=int(meta["K"]),
            r=int(meta["R"]), c=int(meta["C"]), buf=int(meta["BUF"]),
            p=int(meta["P"]), na=int(meta["n_A"]), nb=int(meta["n_B"]),
            ne=int(meta["n_E"]), model=int(meta["used_cycles"]),
            serialize=int(meta["serialize_cycles"]),
            t_start=int(meta["T_start"]),
            t_a=int(meta["T_A"]), t_b=int(meta["T_B"]), t_e=int(meta["T_E"]),
            passes_a=int(meta["passes_A"]), passes_b=int(meta["passes_B"]),
            passes_e=int(meta["passes_E"]),
        ))
    return cases


def compile_for(case: Case, rtl: Path, out_dir: Path) -> tuple[Path | None, str]:
    tag = (f"dimm_{case.m}x{case.n}x{case.k}_{case.r}x{case.c}"
           f"_a{case.na}b{case.nb}e{case.ne}")
    bin_path = out_dir / f"{tag}.vvp"
    if bin_path.exists():
        return bin_path, ""
    cmd = [
        "iverilog", "-g2005", "-o", str(bin_path),
        f"-DM_TB={case.m}", f"-DN_TB={case.n}", f"-DK_TB={case.k}",
        f"-DR_TB={case.r}", f"-DC_TB={case.c}",
        f"-DBUF_TB={case.buf}", f"-DP_TB={case.p}",
        f"-DNA_TB={case.na}", f"-DNB_TB={case.nb}", f"-DNE_TB={case.ne}",
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
    p = run(["vvp", str(bin_path), f"+CASE={case.name}",
             f"+VDIR={vdir.resolve()}", f"+ODIR={odir.resolve()}",
             f"+EXPECTED={case.expected}"],
            cwd=REPO, timeout=timeout)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"dimm_{case.name}.log").write_text(
        f"# {case.name} M={case.m} N={case.n} K={case.k} "
        f"{case.r}x{case.c} BUF={case.buf} P={case.p} "
        f"N_A={case.na} N_B={case.nb} N_E={case.ne}\n" + p.stdout
        + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else ""))
    m = RESULT_RE.search(p.stdout)
    if not m:
        case.status = "ERROR"
        case.detail = (p.stdout + p.stderr).strip()[-400:]
        return
    case.status = m.group(1)
    case.measured = int(m.group(2))
    case.delta = int(m.group(4))
    case.span_a = int(m.group(5))
    case.span_b = int(m.group(6))
    case.span_f = int(m.group(7))
    case.fill = int(m.group(8))
    case.ser = int(m.group(9))
    case.wins_a = int(m.group(10))
    case.wins_b = int(m.group(11))
    case.wins_f = int(m.group(12))
    em = ERRCOUNT_RE.search(p.stdout)
    case.errors = sum(int(g) for g in em.groups()) if em else 0
    if case.status == "FAIL" and not case.detail:
        fails = [ln for ln in p.stdout.splitlines()
                 if "MISMATCH" in ln or "ERROR" in ln]
        case.detail = " | ".join(fails[:3])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nEs", default=None, help="n_E list for generation")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--only", default=None, help="shape tags for generation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--rtl", default=str(DEFAULT_RTL))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("v2 DIMM RTL cross-check (Stage 4)")
    print(f"  rtl     : {rel(Path(args.rtl))}")
    print(f"  nEs     : {args.nEs or '1,2,4,8,16'}")
    print("=" * 78)

    lock_fd = None
    lock_path = SMOKE / ".dimm_rtl.lock"
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
               "--seed", str(args.seed)]
        if args.nEs is not None:
            cmd += ["--nEs", args.nEs]
        if args.classes is not None:
            cmd += ["--classes", args.classes]
        if args.only is not None:
            cmd += ["--only", args.only]
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
            "M": case.m, "N": case.n, "K": case.k,
        })

    rtl = Path(args.rtl).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="v2dimm_"))
    bins: dict[tuple[int, ...], Path] = {}
    for case in cases:
        if case.geom in bins:
            continue
        bin_path, err = compile_for(case, rtl, tmp)
        if bin_path is None:
            print(f"\nCOMPILE FAILED ({case.name}):\n{err}")
            print("\nThe RTL probe contract (`la_q`, `lb_q`, `acc_q`) or the "
                  "port surface is likely not implemented yet.")
            return 3
        bins[case.geom] = bin_path
    print(f"compiled {len(bins)} parameter-set binaries")

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(lambda c: run_case(bins[c.geom], c, args.timeout), cases))

    # -- report ---------------------------------------------------------------
    def stage_exact(c: Case) -> bool:
        return (c.span_a == c.t_a and c.span_b == c.t_b
                and c.span_f == c.t_e and c.ser == c.m * c.n
                and c.fill == c.t_start
                and c.wins_a == c.passes_a and c.wins_b == c.passes_b
                and c.wins_f == c.passes_e)

    print()
    hdr = (f"{'case':<40} {'nA':>2} {'nB':>2} {'nE':>3} {'model':>7} "
           f"{'ser':>5} {'meas':>8} {'delta':>6} {'spans':>5} "
           f"{'status':>6} {'errs':>5}")
    print(hdr)
    print("-" * len(hdr))
    for case in cases:
        spans = ("exact" if stage_exact(case) else "BAD") \
            if case.status == "PASS" else "-"
        print(f"{case.name:<40} {case.na:>2} {case.nb:>2} {case.ne:>3} "
              f"{case.model:>7} {case.serialize:>5} {case.measured:>8} "
              f"{case.delta:>6} {spans:>5} {case.status:>6} {case.errors:>5}"
              + (f"  {case.detail[:70]}" if case.detail else ""))

    # -- strict gates: delta_impl == 0 and every stage span exactly matches
    #    the model (span_A == T_A, span_B == T_B, span_F == T_E,
    #    ser == M*N, fill == T_start) --------------------------------------
    failures = [c for c in cases if c.status != "PASS"]
    print()
    ok = True
    if failures:
        ok = False
        print(f"GATE FAIL: {len(failures)} case(s) not PASS")
    n_checked = 0
    for case in cases:
        if case.status != "PASS":
            continue
        n_checked += 1
        bad = []
        if case.delta != 0:
            bad.append(f"delta_impl={case.delta}")
        if case.span_a != case.t_a:
            bad.append(f"span_A={case.span_a}!=T_A={case.t_a}")
        if case.span_b != case.t_b:
            bad.append(f"span_B={case.span_b}!=T_B={case.t_b}")
        if case.span_f != case.t_e:
            bad.append(f"span_F={case.span_f}!=T_E={case.t_e}")
        if case.ser != case.m * case.n:
            bad.append(f"ser={case.ser}!=M*N={case.m * case.n}")
        if case.fill != case.t_start:
            bad.append(f"fill={case.fill}!=T_start={case.t_start}")
        if case.wins_a != case.passes_a:
            bad.append(f"wins_A={case.wins_a}!=passes_A={case.passes_a}")
        if case.wins_b != case.passes_b:
            bad.append(f"wins_B={case.wins_b}!=passes_B={case.passes_b}")
        if case.wins_f != case.passes_e:
            bad.append(f"wins_F={case.wins_f}!=passes_E={case.passes_e}")
        if bad:
            ok = False
            print(f"GATE FAIL: {case.name}: " + ", ".join(bad))
    if not n_checked:
        ok = False
        print("GATE FAIL: no PASSing cases")
    else:
        print(f"GATE OK  : {n_checked}/{len(cases)} cases exact "
              f"(delta_impl = 0; spans == T_A/T_B/T_E; ser == M*N; "
              f"fill == T_start; window counts == P_A/P_B/P_E)")

    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = "\n".join(
        f"{c.name} n_E={c.ne} {c.status} measured={c.measured} "
        f"expected={c.expected} delta={c.delta}" for c in cases) + "\n"
    run_log = LOGS / f"rtl_smoke_dimm_{stamp}.log"
    run_log.write_text(summary)
    (LOGS / "dimm_latest.log").write_text(summary)
    print(f"\nlogs: {rel(run_log)}  (per-case TB logs in {rel(LOGS)}/)")
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
