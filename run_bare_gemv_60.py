"""Quick bare GEMV sweep on 60×60 grid for FlexScore Pareto.

Uses existing arch XMLs + gen_gemm_wrapper RTL generator.
3 configs × 3 workloads × 20 (d,c) points × 3 seeds.
Only runs feasible points where P >= 1.
"""

import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "nl_dpe"))

ARCH_DIR = PROJECT_DIR / "dse" / "configs" / "arch_60"
OUTPUT_BASE = PROJECT_DIR / "dse" / "flexscore_60" / "dl_gemv"
RESULTS_DIR = PROJECT_DIR / "dse" / "results" / "plots" / "round2_flexscore"
RTL_DIR = OUTPUT_BASE / "rtl"

VTR_ROOT = Path(os.environ.get("VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"))
VTR_FLOW = VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py"

GRID_W, GRID_H = 60, 60
BASELINE_CLBS = 2908
BRAM_STARTX, BRAM_REPEATX, BRAM_TILE_H = 2, 16, 2
DSP_STARTX, DSP_REPEATX, DSP_TILE_H = 6, 16, 4
ROUTE_CHAN_WIDTH = 300

SEEDS = [41906, 7297, 1640]

D_PCTS = [0, 20, 40, 60, 80]
C_PCTS = [0, 20, 40, 60]

CONFIGS = [
    ("512x128", 512, 128),
    ("1024x128", 1024, 128),
    ("1024x64", 1024, 64),
    ("1024x256", 1024, 256),
    ("512x256", 512, 256),
]

WORKLOADS = [
    ("fc_512_128", 512, 128),
    ("fc_512_512", 512, 512),
    ("fc_2048_256", 2048, 256),
]

# Bare GEMV: ~25 CLBs/rep, 4 BRAMs/rep, 0 DSPs, V*H DPEs/rep
CLBS_PER_REP = 25
CLB_OVERHEAD = 30
BRAMS_PER_REP = 4


def _col_positions(startx, repeatx):
    interior = GRID_W - 2
    pos = []
    x = startx
    while x <= interior:
        pos.append(x)
        x += repeatx
    return pos


def get_tile_dims(R, C):
    from area_power import dpe_specs
    s = dpe_specs(R, C)
    return s['tile_width'], s['tile_height']


def count_dpes(tile_w, tile_h, d_ratio, c_ratio):
    interior_w = GRID_W - 2
    interior_h = GRID_H - 2
    tiles_per_col = interior_h // tile_h
    dsp_pos = _col_positions(DSP_STARTX, DSP_REPEATX)
    bram_pos = _col_positions(BRAM_STARTX, BRAM_REPEATX)
    n_rm = min(len(dsp_pos), max(0, round(d_ratio * len(dsp_pos))))
    dsp_removed = sorted(dsp_pos[-n_rm:]) if n_rm > 0 else []
    dsp_kept = sorted(set(dsp_pos) - set(dsp_removed))
    occupied = set(bram_pos) | set(dsp_kept)
    wc = 0
    for pos in dsp_removed:
        if all(1 <= pos+dx <= interior_w and pos+dx not in occupied for dx in range(tile_w)):
            wc += tiles_per_col
            for dx in range(tile_w): occupied.add(pos+dx)
    if c_ratio > 0:
        n_clb = max(1, round(c_ratio * interior_w / tile_w))
        repeatx = max(tile_w+1, interior_w // n_clb)
        x = DSP_STARTX
        while x <= interior_w:
            if all(1 <= x+dx <= interior_w and x+dx not in occupied for dx in range(tile_w)):
                wc += tiles_per_col
                for dx in range(tile_w): occupied.add(x+dx)
            x += repeatx
    return wc


def count_brams():
    interior_w = GRID_W - 2
    interior_h = GRID_H - 2
    n = len(range(BRAM_STARTX, interior_w+1, BRAM_REPEATX))
    return n * (interior_h // BRAM_TILE_H)


def feasibility(R, C, K, N, d, c):
    tw, th = get_tile_dims(R, C)
    dpes = count_dpes(tw, th, d/100, c/100)
    V = math.ceil(K/R)
    H = math.ceil(N/C)
    dpe_per_rep = V * H
    p_dpe = dpes // dpe_per_rep if dpe_per_rep > 0 else 0
    p_clb = max(0, (int(BASELINE_CLBS*(1-c/100)) - CLB_OVERHEAD) // CLBS_PER_REP)
    p_bram = count_brams() // BRAMS_PER_REP
    P = min(p_dpe, p_clb, p_bram)
    area = dpes * tw * th / (GRID_W * GRID_H) * 100
    return P, area, p_dpe, p_clb, p_bram


def parse_fmax(log_path):
    text = log_path.read_text()
    m = re.search(r"Final critical path.*?Fmax:\s+([\d.]+)\s+MHz", text)
    return float(m.group(1)) if m else float("nan")


def run_one(cfg_name, R, C, K, N, P, d, c, seeds):
    from gen_gemm_wrapper import gen_gemm_wrapper
    tw, th = get_tile_dims(R, C)
    arch = ARCH_DIR / f"nl_dpe_tw{tw}_d{d}_c{c}_fixed.xml"
    wl_name = f"fc_{K}_{N}"

    RTL_DIR.mkdir(parents=True, exist_ok=True)
    rtl_path = gen_gemm_wrapper(K, N, R, C, P, str(RTL_DIR))

    base_dir = OUTPUT_BASE / cfg_name / f"d{d}_c{c}" / wl_name
    seed_fmax = []
    for seed in seeds:
        out_dir = base_dir / f"seed{seed}"
        log = out_dir / "vpr_stdout.log"
        if log.exists():
            f = parse_fmax(log)
            if f == f:
                seed_fmax.append(f)
                continue
        cmd = [sys.executable, str(VTR_FLOW), str(rtl_path), str(arch),
               "--route_chan_width", str(ROUTE_CHAN_WIDTH),
               "-temp_dir", str(out_dir), "--seed", str(seed)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if log.exists():
                f = parse_fmax(log)
                seed_fmax.append(f if f == f else float("nan"))
            else:
                seed_fmax.append(float("nan"))
        except:
            seed_fmax.append(float("nan"))

    valid = [f for f in seed_fmax if f == f]
    avg = sum(valid)/len(valid) if valid else float("nan")
    area = count_dpes(tw, th, d/100, c/100) * tw * th / (GRID_W*GRID_H) * 100
    lat = 1e3 / (P * avg) if P > 0 and avg == avg and avg > 0 else float("inf")

    return {
        "config": cfg_name, "d_pct": d, "c_pct": c,
        "workload": wl_name, "P": P, "fmax_mhz": avg,
        "eff_latency_ns": lat, "dpe_area_pct": area,
        "status": "OK" if valid else "FAILED",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    jobs = []
    for cfg_name, R, C in CONFIGS:
        for wl_name, K, N in WORKLOADS:
            for d in D_PCTS:
                for c in C_PCTS:
                    P, area, p_dpe, p_clb, p_bram = feasibility(R, C, K, N, d, c)
                    if P < 1:
                        continue
                    jobs.append((cfg_name, R, C, K, N, P, d, c))

    print(f"Bare GEMV sweep: {len(jobs)} feasible × 3 seeds = {len(jobs)*3} VTR runs")

    if args.dry_run:
        print(f"\n{'Config':>12} {'WL':>14} {'d%':>4} {'c%':>4} {'P':>4} {'P_dpe':>6} {'P_clb':>6} {'P_brm':>6}")
        for j in jobs:
            cfg, R, C, K, N, P, d, c = j
            _, _, pd, pc, pb = feasibility(R, C, K, N, d, c)
            print(f"{cfg:>12} fc_{K}_{N:>14} {d:>4} {c:>4} {P:>4} {pd:>6} {pc:>6} {pb:>6}")
        return

    print(f"Running with --jobs {args.jobs}...")
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {}
        for j in jobs:
            f = executor.submit(run_one, *j, SEEDS)
            futures[f] = j
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)
            j = futures[f]
            print(f"  [{i}/{len(jobs)}] {r['config']} {r['workload']} "
                  f"d={r['d_pct']}% c={r['c_pct']}% P={r['P']}: "
                  f"Fmax={r['fmax_mhz']:.1f} lat={r['eff_latency_ns']:.2f} ({r['status']})")

    # Write CSV
    csv_path = RESULTS_DIR / "flexscore_dl_gemv_results.csv"
    fields = ["config", "d_pct", "c_pct", "workload", "P", "fmax_mhz",
              "eff_latency_ns", "dpe_area_pct", "status"]
    rows = sorted(results, key=lambda r: (r["config"], r["d_pct"], r["c_pct"], r["workload"]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults → {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
