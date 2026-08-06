#!/usr/bin/env python3
"""Safe-softmax RTL smoke harness: bit-exact oracles + iverilog sweep + cycle check.

Standalone study (no simulator import). Conventions locked in
docs/superpowers/plans/2026-08-05-softmax-study-implementation.md:

  AL : u = max-x; E8[u] = min(255, round(255*exp(-u/16)));   sum = sum(E8)
       idx = min(255, sum >> log2(S));  R8 = recip ROM;
       p = min(255, (E8*R8[idx]) >> log2(S))                    [linear domain]
  NL : d = max(x-max, -128); exp byte = (1 + d + (d*d)//2) & 0xFF  [ACAM_MODE=1
       semantics, fc_verification/rtl/dpe_nldpe.v:203-206]; sum unsigned;
       lq = min(sum >> log2(S), 127); ls = lq - 1              [ACAM_MODE=2];
       out = clamp(x - max - ls, -128, 127)                     [log domain]

Row->lane mapping (lockstep, W=16): lane k owns global rows {k + 16*i}.
Hex files are lane-major: entry ((k*RPL + i)*S + j) = matrix[k + 16*i][j].

Usage:
    python3 softmax_study/run_softmax_smoke.py                # all cases
    python3 softmax_study/run_softmax_smoke.py --case al_s128
    python3 softmax_study/run_softmax_smoke.py --no-cycle-gate
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RTL = HERE / "rtl"
TB = HERE / "tb"
RESULTS = HERE / "results"
DPE_MODEL = REPO / "fc_verification" / "rtl" / "dpe_nldpe.v"

W = 16          # lanes
SEED = 7

# ── ROM tables (must match gen_luts.py exactly) ─────────────────────────
E8 = [min(255, int(255 * math.exp(-u / 16.0) + 0.5)) for u in range(256)]
R8 = [255] + [min(255, (2 * 256 + i) // (2 * i)) for i in range(1, 256)]


# ── Oracles ─────────────────────────────────────────────────────────────
def oracle_al(scores: np.ndarray, S: int) -> np.ndarray:
    """Linear-domain safe softmax, AL fixed-point path. Returns uint8."""
    r_shift = S.bit_length() - 1
    out = np.zeros_like(scores, dtype=np.uint8)
    for r in range(scores.shape[0]):
        row = scores[r].astype(int)
        m = int(row.max())
        e = np.array([E8[m - x] for x in row], dtype=int)
        s = int(e.sum())
        idx = min(255, s >> r_shift)
        rec = R8[idx]
        out[r] = np.minimum(255, (e * rec) >> r_shift).astype(np.uint8)
    return out


def oracle_nl(scores: np.ndarray, S: int) -> np.ndarray:
    """Log-domain safe softmax, NL ACAM-model semantics. Returns int8."""
    log_shift = S.bit_length() - 1
    out = np.zeros_like(scores, dtype=np.int8)
    for r in range(scores.shape[0]):
        row = scores[r].astype(int)
        m = int(row.max())
        d = np.maximum(row - m, -128)
        eb = np.array([(1 + v + (v * v) // 2) & 0xFF for v in d], dtype=int)
        s = int(eb.sum())
        lq = min(s >> log_shift, 127)
        ls = lq - 1
        out[r] = np.clip(row - m - ls, -128, 127).astype(np.int8)
    return out


# ── Cycle model ─────────────────────────────────────────────────────────
# Measured-locked closed forms, valid at the swept S values. The table's
# throughput column uses MEASURED cycles; these formulas are the checksum
# that the RTL's schedule matches the documented stage model.
AL_LOCKED = True    # Task 3 Step 5: anchors S=128 -> 99, S=256 -> 323
NL_LOCKED = False   # Task 4 Step 5 pending


def predict_cycles(kind: str, S: int, C: int | None = None,
                   n_exp: int | None = None) -> tuple[int | None, str]:
    """Return (predicted_total_cycles or None, explanation)."""
    rpl = S // W
    wpr = S // 16                     # 16-wide words per row
    if kind == "al":
        # Row-granular pipeline: stages A/B/D each stream wpr words/row at
        # 1 word/cycle with no inter-row bubble (steady = wpr); the fill
        # chain A(0)->B(0)->Cs(0)->D(0) plus pipe registers measures
        # 5*wpr + 3 - wpr*1 ... locked closed form over both anchors:
        #   total = (rpl + 4) * wpr + 3
        # (S=128: (8+4)*8+3 = 99;  S=256: (16+4)*16+3 = 323)
        if not AL_LOCKED:
            return None, f"AL steady={wpr} (unlocked)"
        return (rpl + 4) * wpr + 3, f"AL locked: (rpl+4)*wpr+3, steady={wpr}"
    else:
        E = S // n_exp
        lcyc = (E + 4) // 5           # = ceil(E/5); also OCYC
        # B occupancy = lcyc (strobe rate; drain overlaps next row's strobes).
        steady = max(wpr, lcyc, 10)
        if not NL_LOCKED:
            return None, f"NL steady={steady} lcyc={lcyc} (unlocked)"
        return None, "NL formula locked in Task 4"


# ── Cases ───────────────────────────────────────────────────────────────
@dataclass
class Case:
    label: str
    kind: str            # "al" | "nl"
    S: int
    C: int | None = None
    n_exp: int | None = None

    @property
    def rpl(self) -> int:
        return self.S // W


CASES = [
    Case("al_s128", "al", 128),
    Case("al_s256", "al", 256),
    Case("nl_p1_s128", "nl", 128, C=128, n_exp=1),
    Case("nl_p2_s128", "nl", 128, C=256, n_exp=1),   # E=S=128 < C: same elab as p1
    Case("nl_p1_s256", "nl", 256, C=128, n_exp=2),   # free split: 2 exp DPEs/lane
    Case("nl_p2_s256", "nl", 256, C=256, n_exp=1),
]


# ── Hex emit ────────────────────────────────────────────────────────────
def lane_major(mat: np.ndarray, S: int) -> np.ndarray:
    """Reorder (S,S) matrix rows to lane-major flat bytes."""
    rpl = S // W
    rows = [mat[k + W * i] for k in range(W) for i in range(rpl)]
    return np.concatenate(rows)


def write_hex(path: Path, flat: np.ndarray) -> None:
    with path.open("w") as f:
        for b in flat.astype(np.uint8):
            f.write(f"{int(b):02x}\n")


# ── Runner ──────────────────────────────────────────────────────────────
def run_case(case: Case, keep: bool, cycle_gate: bool) -> dict:
    work = RESULTS / f"smoke_{case.label}"
    work.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(SEED)
    scores = rng.randint(-128, 128, (case.S, case.S)).astype(np.int8)
    expected = (oracle_al if case.kind == "al" else oracle_nl)(scores, case.S)

    write_hex(work / "scores.hex", lane_major(scores.view(np.uint8), case.S))
    write_hex(work / "expected.hex", lane_major(expected.view(np.uint8), case.S))

    sim = work / "sim.out"
    if case.kind == "al":
        srcs = [RTL / "softmax_al.v", TB / "sim_models.v", TB / "tb_softmax_al.v"]
        defines = [f"-DS_TB={case.S}"]
        tb_top = "tb_softmax_al"
    else:
        srcs = [DPE_MODEL, RTL / "softmax_nldpe.v", TB / "tb_softmax_nldpe.v"]
        defines = [f"-DS_TB={case.S}", f"-DC_TB={case.C}",
                   f"-DNEXP_TB={case.n_exp}"]
        tb_top = "tb_softmax_nldpe"

    cmd = ["iverilog", "-g2005", "-o", str(sim), "-s", tb_top,
           f"-I{RTL}", *defines,
           f"-DSCORES_HEX=\"{work / 'scores.hex'}\"",
           f"-DEXPECTED_HEX=\"{work / 'expected.hex'}\"",
           *[str(s) for s in srcs]]
    comp = subprocess.run(cmd, capture_output=True, text=True)
    if comp.returncode != 0:
        return dict(label=case.label, status="COMPILE_FAIL",
                    detail=comp.stderr.strip().splitlines()[-8:])

    run = subprocess.run(["vvp", str(sim)], capture_output=True, text=True,
                         timeout=1800)
    out = run.stdout
    (work / "vvp.log").write_text(out + "\n--- stderr ---\n" + run.stderr)

    cycles = None
    for line in out.splitlines():
        if line.startswith("CYCLES="):
            cycles = int(line.split("=")[1])
    functional = "TB_PASS" in out
    mism = [ln for ln in out.splitlines() if ln.startswith("MISMATCH")][:10]

    pred, pred_note = predict_cycles(case.kind, case.S, case.C, case.n_exp)
    cycle_ok = (pred is None) or (cycles == pred)

    status = "PASS" if functional and (cycle_ok or not cycle_gate) else "FAIL"
    if not keep:
        sim.unlink(missing_ok=True)
    return dict(label=case.label, status=status, functional=functional,
                cycles=cycles, predicted=pred, pred_note=pred_note,
                cycle_exact=(pred is not None and cycles == pred),
                mismatches=mism)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="append", help="run only these labels")
    ap.add_argument("--no-cycle-gate", action="store_true",
                    help="functional check only (bring-up)")
    ap.add_argument("--keep", action="store_true", help="keep sim binaries")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.case or c.label in args.case]
    results = []
    for c in cases:
        r = run_case(c, keep=args.keep, cycle_gate=not args.no_cycle_gate)
        results.append(r)
        tag = r["status"]
        cyc = f"cycles={r.get('cycles')} pred={r.get('predicted')}" \
            if r.get("cycles") is not None else r.get("detail", "")
        print(f"[{c.label:12s}] {tag:12s} {cyc}")
        for m in r.get("mismatches", []):
            print(f"    {m}")

    summary = RESULTS / "smoke.json"
    summary.write_text(json.dumps(results, indent=2, default=str))
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{n_pass}/{len(results)} PASS  (details: {summary})")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
