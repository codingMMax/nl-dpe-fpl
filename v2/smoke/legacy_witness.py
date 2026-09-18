#!/usr/bin/env python3
"""legacy_witness.py — third-witness cross-check of the v2 corpus (spec §9).

Purpose
-------
The v2 chain is `spec -> oracle -> (GATE 1) -> sim -> (GATE 2) -> RTL`. That
chain is internally consistent but all of its witnesses share one lineage
(the v2 tree). This script adds the *independent* witness the spec promises:
the frozen legacy material under `rtl_flow/` (different sessions, different
code), run against the **same v2 stimulus**.

Two witnesses are produced per case, both for v2 mode 0 (REGULAR) at the
legacy geometry 256x256:

  W1 (numerical) legacy oracle : rtl_flow/smoke/oracles/nldpe_mac_oracle.py
     - direct int8 matmul and bit-serial MAC (it cross-checks itself)
     - compared bit-exactly against `v2/oracle/nldpe_ref.compute_y`
     - low-byte output compared against `acam_transform(y, REGULAR)`

  W2 (RTL) legacy primitive  : rtl_flow/rtl/dpe_nldpe_faithful.v
     - driven through TEST_MODE=2 of its own frozen TB, but with the
       `.mem` vectors written from a v2 case: t2_expected.mem is the v2
       expected byte stream, so the legacy RTL is compared against the
       GATE-1-certified v2 expected bits on identical stimulus
     - functional verdict is the witness; cycle count is reported only
       (legacy is double-buffered: T_steady 52 vs v2 60 — a documented
       differing witness, spec §10 P1)

Scope limits (deliberate, reported not hidden):
  * v2 mode 0 only (legacy identity ACAM); ACTIVATION has no legacy form;
    legacy EXP diverges once the wide intermediate exceeds int32 (wrap vs
    v2 clamp).
  * 256x256 only: legacy was never validated at C=512, where its output
    buffer has the documented overwrite hazard (legacy brief §4.1).
  * M=1 only: the legacy T2 body is a single-pass test.

Usage:
  python3 v2/smoke/legacy_witness.py [--keep]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LEGACY_ORACLE = REPO / "rtl_flow" / "smoke" / "oracles" / "nldpe_mac_oracle.py"
LEGACY_RTL = REPO / "rtl_flow" / "rtl" / "dpe_nldpe_faithful.v"
LEGACY_TB = REPO / "rtl_flow" / "tb" / "tb_dpe_nldpe_faithful.v"
GEN_CASES = REPO / "v2" / "smoke" / "gen_cases.py"

sys.path.insert(0, str(REPO / "v2" / "oracle"))
import nldpe_ref as ref  # noqa: E402

R, C, BUF, P, M = 256, 256, 40, 8, 1
LCYC = (R * 8 + BUF - 1) // BUF
RTLL_CYC_RE = re.compile(r"total_cycles=(-?\d+)")
VERDICT_RE = re.compile(r"\[tb_faithful\] T2 (PASS|FAIL)(?: \((\d+) errors\))?")


def load_legacy_oracle():
    spec = importlib.util.spec_from_file_location("legacy_nldpe_oracle", LEGACY_ORACLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_weight_mem(path: Path) -> np.ndarray:
    """v2 `weights.mem`: one 10-hex-digit word per line, byte in [7:0]."""
    words = [int(ln, 16) & 0xFF for ln in path.read_text().split()]
    assert len(words) == R * C, f"{path}: {len(words)} weights != {R*C}"
    return np.array(words, dtype=np.uint8).view(np.int8).reshape(R, C)


def decode_act_mem(path: Path, m: int) -> np.ndarray:
    """v2 `act.mem`: M bursts of LCYC words, 5 bytes/word (byte i in 8i+7:8i)."""
    words = [int(ln, 16) for ln in path.read_text().split()]
    assert len(words) == m * LCYC, f"{path}: {len(words)} act words != {m*LCYC}"
    out = []
    for w in words:
        out.extend((w >> (8 * i)) & 0xFF for i in range(BUF // 8))
    return np.array(out[: R * m], dtype=np.uint8).view(np.int8).reshape(m, R)


def decode_expected_out(path: Path, m: int) -> np.ndarray:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == m, f"{path}: {len(lines)} passes != {m}"
    out = [[int(ln[i:i+2], 16) for i in range(0, len(ln), 2)] for ln in lines]
    return np.array(out, dtype=np.uint8)


def write_legacy_vectors(vdir: Path, W: np.ndarray, x0: np.ndarray, exp0: np.ndarray) -> None:
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "t2_weights.mem").write_text(
        "".join(f"{b:02x}\n" for b in np.ascontiguousarray(W).view(np.uint8).reshape(-1)))
    (vdir / "t2_inputs.mem").write_text(
        "".join(f"{b:02x}\n" for b in np.ascontiguousarray(x0).view(np.uint8)))
    (vdir / "t2_expected.mem").write_text(
        "".join(f"{b:02x}\n" for b in exp0))


def run_legacy_rtl(vdir: Path, bin_path: Path) -> tuple[str, int, int, str]:
    cc = subprocess.run(
        ["iverilog", "-g2005", "-o", str(bin_path),
         "-DTEST_MODE=2", f"-DR_TB={R}", f"-DC_TB={C}", f"-DBUF_TB={BUF}",
         f"-DPRECISION_TB={P}", f'-DVECT_DIR="{vdir}/"',
         str(LEGACY_TB), str(LEGACY_RTL)],
        capture_output=True, text=True, cwd=REPO)
    if cc.returncode != 0:
        return "COMPILE_FAIL", -1, -1, cc.stderr.strip()[-300:]
    rv = subprocess.run(["vvp", str(bin_path)], capture_output=True, text=True, cwd=REPO)
    out = rv.stdout
    vm = VERDICT_RE.search(out)
    status = f"{vm.group(1)}" + (f" ({vm.group(2)} errs)" if vm and vm.group(2) else "") \
        if vm else "NO_VERDICT"
    cm = RTLL_CYC_RE.search(out)
    leg_cycles = int(cm.group(1)) if cm else -1
    mism = [ln for ln in out.splitlines() if "MISMATCH" in ln]
    detail = " | ".join(mism[:3])
    return status, leg_cycles, len(mism), detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="v2legacy_"))
    try:
        stim = tmp / "stimuli"
        subprocess.run(
            [sys.executable, str(GEN_CASES), "--out", str(stim), "--geoms", "256x256",
             "--ms", "1", "--modes", "0", "--classes", "identity,random,extremes"],
            check=True, cwd=REPO, stdout=subprocess.DEVNULL)

        legacy = load_legacy_oracle()
        print(f"{'case':<34} {'W1 mac':>6} {'W1 bytes':>8} {'W2 rtl':>12} "
              f"{'leg-cyc':>8} {'v2-cyc':>7}")
        print("-" * 82)
        ok = True
        for case_dir in sorted(p for p in stim.iterdir() if p.is_dir()):
            meta = json.loads((case_dir / "case.json").read_text())
            assert (meta["R"], meta["C"], meta["M"], meta["mode"]) == (R, C, M, 0), meta

            W = decode_weight_mem(case_dir / "weights.mem")
            X = decode_act_mem(case_dir / "act.mem", M)
            exp_out = decode_expected_out(case_dir / "expected_out.mem", M)

            # W1: independent legacy oracle (direct + bit-serial, self-checking)
            y_v2 = ref.compute_y(W, X)
            out_v2 = ref.acam_transform(y_v2, ref.MODE_REGULAR).view(np.uint8)
            assert (out_v2 == exp_out).all(), f"{case_dir.name}: v2 expected drift"
            mac, _acam, byte = legacy.expected_outputs(W, X[0], acam_mode=0, precision=8)
            mac_ok = bool((mac == y_v2[0]).all())
            byte_ok = bool((byte == out_v2[0]).all())

            # W2: legacy RTL on the same stimulus, v2 expected bytes
            vdir = tmp / "vect" / case_dir.name
            write_legacy_vectors(vdir, W, X[0], exp_out[0])
            bin_path = tmp / f"{case_dir.name}.vvp"
            status, leg_cycles, n_mism, detail = run_legacy_rtl(vdir, bin_path)

            rtl_ok = status.startswith("PASS")
            ok &= mac_ok and byte_ok and rtl_ok
            print(f"{case_dir.name:<34} {'OK' if mac_ok else 'MISMATCH':>6} "
                  f"{'OK' if byte_ok else 'MISMATCH':>8} {status:>12} "
                  f"{leg_cycles:>8} {meta['used_cycles']:>7}"
                  + (f"  {detail[:40]}" if detail else ""))

        print()
        print("W1 = legacy oracle vs v2 oracle (int32 MAC + REGULAR low byte)")
        print("W2 = legacy RTL (TEST_MODE=2) vs v2 expected bytes on identical stimulus;")
        print("     cycles are report-only: legacy is double-buffered (T_steady 52 vs v2 60)")
        print("RESULT: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
