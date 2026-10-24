#!/usr/bin/env python3
"""attn_rows.py — the attention DSE rows from the head workload: one row per build (docs/experiment.md §4.3, §5).

The mapping of the rows it prices (the 2026-10-16 run): P_PROJ 1, N_E = N_EP = 2d, N_EXP 2, N_OC by the floor rule, B_kv 16, X_S 16,
(the fixed mapping is N_E = d since 2026-10-23; this script passes N_E = 2d explicitly until stage 5),
B_q = S (the headParams defaults; tools/check_head_params.py proves the RTL builds the same head for every build).
Each row is simulator/platform/rows.py `attentionRow`: the head's compute cycles (done - start, no O read-out),
energy from the Report's counts x the platform's frozen prices (in total, per term and per role; weight programming
apart, as setup), the crossbar area (count x the arch XML's wc footprint) and the memory inventory with its predicted
blocks. CLB and memory blocks come from VTR; a build with VTR metrics (--vtr: a smoke's result.json, or the DSE's
dse/results/attn_vtr_results.csv from v3/dse/vtr_dse.py) also gets its total area, latency and EDAP (pJ x ns x mm^2).
Usage: python3 v3/dse/attn_rows.py v3/dse/results/attn_priced_results.csv --vtr v3/dse/results/attn_vtr_results.csv
       python3 v3/dse/attn_rows.py OUT.csv [--vtr RESULT.json | VTR_ATT_72.csv ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse.space import CONFIGS, SHAPES  # noqa: E402
from simulator.platform.rows import attentionRow  # noqa: E402



def load_vtr(paths: list) -> dict:
    """{(shape, R, C): VTR metrics} from smoke result.json files or the DSE's CSV (v3/dse/vtr_dse.py, dse/results/
    attn_vtr_results.csv: its ok lines, Fmax = the seeds' mean)."""
    out = {}
    for f in paths:
        if str(f).endswith(".csv"):
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r["ok"] != "1":
                        continue
                    if int(r["P_PROJ"]) != 1 or int(r["N_E"]) != 2 * int(r["d"]):
                        raise ValueError(f"{f} {r['build']}: not the fixed mapping (P_PROJ {r['P_PROJ']}, N_E {r['N_E']})")
                    out[(r["shape"], int(r["R"]), int(r["C"]))] = dict(
                        fmax_mhz=float(r["fmax_mhz"]), clb=int(r["clb"]), memory=int(r["memory"]), wc=int(r["wc"]),
                        seeds=r["seeds"], fmax_seed={s: float(r[f"fmax_s{s}"]) for s in (11, 13, 17)
                                                     if r.get(f"fmax_s{s}")})
            continue
        r = json.loads(Path(f).read_text())
        p, fl = r["params"], r["flow"]
        shape = next(s for (s, n, dm, d) in SHAPES if (n, dm, d) == (p["NSEQ"], p["DM"], p["D"]))
        if p.get("P_PROJ") != 1 or p.get("N_E") != 2 * p["D"]:
            raise ValueError(f"{f}: not the fixed mapping (P_PROJ {p.get('P_PROJ')}, N_E {p.get('N_E')})")
        res = fl["resources"]
        out[(shape, p["R"], p["C"])] = dict(fmax_mhz=fl["fmax_mhz"], clb=res["clb"], memory=res["memory"], wc=res["wc"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--vtr", nargs="*", default=[])
    a = ap.parse_args()
    vtr = load_vtr(a.vtr)
    # N_E = 2d: the mapping of the VTR rows this prices (the 2026-10-16 run); the fixed mapping is N_E = d since
    # 2026-10-23 (headParams's default). Stage 5 prices the bit-plane run at the default.
    rows = [attentionRow(name, n, dm, d, R, C, vtr, nE=2 * d) for (name, n, dm, d) in SHAPES for (R, C) in CONFIGS]
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"attn_rows: {len(rows)} builds -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
