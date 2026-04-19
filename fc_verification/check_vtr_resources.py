#!/usr/bin/env python3
"""VTR resource check for Phase 2 FC verification.

Consumes the existing `block_comp_apr_11/results/block_comparison_results.csv`
(3-seed-averaged VTR outputs for all 12 FC configs). Asserts for each row:
  * dpe_count == V * H (exact — the load-bearing architectural claim)
  * wc (tile-count instantiated via VTR WC hard block) == dpe_count
  * BRAM, CLB reported counts are printed for audit; no hard envelope
    enforced (the VTR packer's DSP/CLB minor drift is architecturally
    benign and does not change the per-stage cycle numbers verified by
    run_fc_phase2.py).

Exits 0 iff all 12 rows have matching dpe_count and wc.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "block_comp_apr_11" / "results" / "block_comparison_results.csv"


def main() -> int:
    if not CSV_PATH.exists():
        print(f"[vtr-check] missing {CSV_PATH}", file=sys.stderr)
        return 1
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 12:
        print(f"[vtr-check] expected 12 rows, got {len(rows)}", file=sys.stderr)
        return 1
    header = f"{'config':32s} {'V*H':>4s} {'DPE':>4s} {'WC':>3s} {'CLB':>4s} {'BRAM':>5s} {'grid':>9s} {'Fmax':>7s}  verdict"
    print(header)
    print("-" * len(header))
    fail = 0
    for r in rows:
        cfg = f"{r['setup']}/{r['workload']}"
        vh = int(r["V"]) * int(r["H"])
        dpe = int(r["dpe_count"])
        wc = int(r["wc"])
        ok = dpe == vh and wc == vh
        if not ok:
            fail += 1
        verdict = "PASS" if ok else "FAIL"
        print(f"{cfg:32s} {vh:4d} {dpe:4d} {wc:3d} {r['clb']:>4s} {r['bram']:>5s} "
              f"{r['grid_w']:>3s}x{r['grid_h']:<5s} {r['fmax_avg_mhz']:>7s}  {verdict}")
    print()
    if fail:
        print(f"[vtr-check] FAIL: {fail}/12 configs have DPE/WC count mismatch vs V*H")
        return 1
    print(f"[vtr-check] PASS: 12/12 configs have DPE count == WC == V*H exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
