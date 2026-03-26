"""
FlexScore DSE v3 (Round 2 Part 4): Proportional budget sweep on 60×60 grid.

Single "area budget %" parameter — each resource type (CLB, DSP, BRAM)
loses the same percentage of itself. DPE tiles placed in freed CLB columns.
Non-DL: FlexScore (4 benchmarks). DL: bare GEMV (3 workloads, 5 configs).

Grid: 60×60 (2908 CLBs, 56 DSPs, 116 BRAMs)
Budget levels: {0, 10, 20, 30, 40, 50}%
Configs: 512×128, 1024×128, 1024×64, 1024×256, 512×256
Non-DL benchmarks: bgm, LU8PEEng, stereovision1, arm_core
DL workloads: fc_512_128, fc_512_512, fc_2048_256 (bare GEMV)

Usage:
  python flexscore_dse.py --gen-arch         # generate 60×60 arch XMLs
  python flexscore_dse.py --nondl            # run non-DL benchmarks
  python flexscore_dse.py --dl               # run DL bare GEMV
  python flexscore_dse.py --sanity           # sanity check
  python flexscore_dse.py --nondl --dl       # run both
  python flexscore_dse.py --skip-existing    # resume
  python flexscore_dse.py --dry-run          # list runs
"""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
ARCH_DIR = PROJECT_DIR / "dse" / "configs" / "arch_60"
RESULTS_DIR = PROJECT_DIR / "dse" / "results" / "plots" / "round2_flexscore"
OUTPUT_BASE = PROJECT_DIR / "dse" / "flexscore_60"

VTR_ROOT = Path(os.environ.get(
    "VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"))
VTR_FLOW = VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py"
BENCH_DIR = VTR_ROOT / "vtr_flow" / "benchmarks" / "verilog"

sys.path.insert(0, str(PROJECT_DIR / "nl_dpe"))

ROUTE_CHAN_WIDTH = 300

# ── Grid ──────────────────────────────────────────────────────────────────
GRID_W, GRID_H = 60, 60
BASELINE_CLBS_60 = 2908   # measured from VTR on 60×60 baseline

# ── Multi-seed ────────────────────────────────────────────────────────────
NUM_SEEDS = 3
SEED_FILE = OUTPUT_BASE / "seeds.json"

# ── Sweep ─────────────────────────────────────────────────────────────────
BUDGET_LEVELS = [0, 10, 20, 30, 35, 40, 50]

# ── Configs ───────────────────────────────────────────────────────────────
CONFIGS = [
    {"name": "512x128",  "R": 512,  "C": 128},
    {"name": "1024x128", "R": 1024, "C": 128},
    {"name": "1024x64",  "R": 1024, "C": 64},
    {"name": "1024x256", "R": 1024, "C": 256},
    {"name": "512x256",  "R": 512,  "C": 256},
]

# ── Non-DL Benchmarks ────────────────────────────────────────────────────
BENCHMARKS = {
    "bgm":            "bgm.v",
    "LU8PEEng":       "LU8PEEng.v",
    "stereovision1":  "stereovision1.v",
    "arm_core":       "arm_core.v",
}

SANITY_POINTS = [
    (0, "bgm"), (20, "bgm"), (40, "bgm"),
    (0, "stereovision1"), (20, "stereovision1"), (40, "stereovision1"),
]

# ── DL Workloads (bare GEMV) ────────────────────────────────────────────
DL_WORKLOADS = [
    ("fc_512_128", 512, 128),
    ("fc_512_512", 512, 512),
    ("fc_2048_256", 2048, 256),
]

# Attention DL workloads (n_seq, d_head)
DL_ATTENTION_WORKLOADS = [
    ("attention_128_64",  128, 64),
    ("attention_256_64",  256, 64),
    ("attention_128_128", 128, 128),
]

# Attention-optimal configs (from Round 1 attention ranking top-3)
ATTN_CONFIGS = [
    {"name": "128x64",  "R": 128, "C": 64},
    {"name": "512x64",  "R": 512, "C": 64},
    {"name": "128x128", "R": 128, "C": 128},
]

# Bare GEMV per-replica resources (calibrated from VTR)
GEMV_CLBS_PER_REP = 25
GEMV_CLB_OVERHEAD = 30
GEMV_BRAMS_PER_REP = 4

# Attention per-replica resources (from 120×120 calibration — will re-calibrate on 60×60)
ATTN_CLBS_PER_REP = 145
ATTN_CLB_OVERHEAD = 15
ATTN_BRAMS_PER_REP = 64
ATTN_DSPS_PER_REP = 2

# Layout constants
BRAM_TILE_HEIGHT = 2
BRAM_STARTX = 2
BRAM_REPEATX = 16
DSP_TILE_HEIGHT = 4
DSP_STARTX = 6
DSP_REPEATX = 16


# ── Budget Mapping ───────────────────────────────────────────────────────

def budget_to_ratios(budget_pct):
    """Convert budget % to (clb_ratio, dsp_ratio, bram_ratio).

    All resource types lose the same proportional fraction.
    """
    ratio = budget_pct / 100.0
    return ratio, ratio, ratio


# ── Seed Management ───────────────────────────────────────────────────────

def get_or_create_seeds():
    if SEED_FILE.exists():
        with open(SEED_FILE) as f:
            seeds = json.load(f)
        print(f"Loaded seeds: {seeds}")
        return seeds
    seeds = random.sample(range(1, 100000), NUM_SEEDS)
    SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEED_FILE, "w") as f:
        json.dump(seeds, f)
    print(f"Generated seeds: {seeds}")
    return seeds


# ── Resource Counting ─────────────────────────────────────────────────────

def _baseline_col_positions(startx, repeatx, grid_w):
    interior = grid_w - 2
    positions = []
    x = startx
    while x <= interior:
        positions.append(x)
        x += repeatx
    return positions


def get_tile_dims(R, C):
    from area_power import dpe_specs
    s = dpe_specs(R, C)
    return s['tile_width'], s['tile_height']


def count_available_wc(tile_w, tile_h, clb_ratio, dsp_ratio, bram_ratio=0.0):
    """Count DPE tiles on 60×60 grid after proportional replacement."""
    interior_w = GRID_W - 2
    interior_h = GRID_H - 2
    tiles_per_col = interior_h // tile_h

    dsp_positions = _baseline_col_positions(DSP_STARTX, DSP_REPEATX, GRID_W)
    bram_positions = _baseline_col_positions(BRAM_STARTX, BRAM_REPEATX, GRID_W)

    n_dsp_remove = min(len(dsp_positions),
                       max(0, round(dsp_ratio * len(dsp_positions))))
    dsp_removed = sorted(dsp_positions[-n_dsp_remove:]) if n_dsp_remove > 0 else []
    dsp_kept = sorted(set(dsp_positions) - set(dsp_removed))

    n_bram_remove = min(len(bram_positions),
                        max(0, round(bram_ratio * len(bram_positions))))
    bram_kept = sorted(set(bram_positions) -
                       set(sorted(bram_positions[-n_bram_remove:])
                           if n_bram_remove > 0 else []))

    if clb_ratio > 0:
        n_clb_dpe_cols = max(1, round(clb_ratio * interior_w / tile_w))
        wc_clb_repeatx = max(tile_w + 1, interior_w // n_clb_dpe_cols)
    else:
        wc_clb_repeatx = None

    occupied = set()
    for pos in bram_kept:
        occupied.add(pos)
    for pos in dsp_kept:
        occupied.add(pos)

    wc_from_dsp = 0
    for pos in dsp_removed:
        can_place = all(1 <= pos + dx <= interior_w and (pos + dx) not in occupied
                        for dx in range(tile_w))
        if can_place:
            wc_from_dsp += tiles_per_col
            for dx in range(tile_w):
                occupied.add(pos + dx)

    wc_from_clb = 0
    if wc_clb_repeatx:
        x = DSP_STARTX
        while x <= interior_w:
            can_place = all(1 <= x + dx <= interior_w and (x + dx) not in occupied
                            for dx in range(tile_w))
            if can_place:
                wc_from_clb += tiles_per_col
                for dx in range(tile_w):
                    occupied.add(x + dx)
            x += wc_clb_repeatx

    return wc_from_dsp + wc_from_clb


def count_available_brams(bram_ratio=0.0):
    """Count BRAM tiles remaining after proportional removal."""
    interior_h = GRID_H - 2
    bram_positions = _baseline_col_positions(BRAM_STARTX, BRAM_REPEATX, GRID_W)
    n_remove = min(len(bram_positions),
                   max(0, round(bram_ratio * len(bram_positions))))
    n_kept = len(bram_positions) - n_remove
    return n_kept * (interior_h // BRAM_TILE_HEIGHT)


def count_available_dsps(dsp_ratio):
    interior_h = GRID_H - 2
    dsps_per_col = interior_h // DSP_TILE_HEIGHT
    dsp_positions = _baseline_col_positions(DSP_STARTX, DSP_REPEATX, GRID_W)
    n_remove = min(len(dsp_positions),
                   max(0, round(dsp_ratio * len(dsp_positions))))
    return (len(dsp_positions) - n_remove) * dsps_per_col


def avail_clbs(clb_ratio):
    return int((1 - clb_ratio) * BASELINE_CLBS_60)


def avail_dsps(dsp_ratio):
    """Remaining DSP tiles after proportional removal on 60×60."""
    dsp_positions = _baseline_col_positions(DSP_STARTX, DSP_REPEATX, GRID_W)
    n_dsp_cols = len(dsp_positions)
    n_removed = min(n_dsp_cols, max(0, round(dsp_ratio * n_dsp_cols)))
    remaining_cols = n_dsp_cols - n_removed
    dsps_per_col = (GRID_W - 2) // 4  # DSP tile height = 4
    return remaining_cols * dsps_per_col


def dpe_area_pct(tile_w, tile_h, clb_ratio, dsp_ratio, bram_ratio=0.0):
    n_dpes = count_available_wc(tile_w, tile_h, clb_ratio, dsp_ratio, bram_ratio)
    return n_dpes * tile_w * tile_h / (GRID_W * GRID_H) * 100


def compute_dl_feasibility(R, C, K, N, budget_pct):
    """3-resource feasibility for bare GEMV on 60×60."""
    clb_ratio, dsp_ratio, bram_ratio = budget_to_ratios(budget_pct)
    tile_w, tile_h = get_tile_dims(R, C)
    total_dpes = count_available_wc(tile_w, tile_h, clb_ratio, dsp_ratio, bram_ratio)

    V = math.ceil(K / R)
    H = math.ceil(N / C)
    dpes_per_rep = V * H
    p_dpe = total_dpes // dpes_per_rep if dpes_per_rep > 0 else 0

    clbs = avail_clbs(clb_ratio)
    p_clb = max(0, (clbs - GEMV_CLB_OVERHEAD) // GEMV_CLBS_PER_REP)

    brams = count_available_brams(bram_ratio)
    p_bram = brams // GEMV_BRAMS_PER_REP

    P = min(p_dpe, p_clb, p_bram)

    limits = {'dpe': p_dpe, 'clb': p_clb, 'bram': p_bram}
    limit = min(limits, key=limits.get) if P >= 1 else 'none'

    return {
        'feasible': P >= 1, 'P': P,
        'p_dpe': p_dpe, 'p_clb': p_clb, 'p_bram': p_bram,
        'limit': limit, 'total_dpes': total_dpes,
        'V': V, 'H': H, 'dpes_per_rep': dpes_per_rep,
        'area_pct': dpe_area_pct(tile_w, tile_h, clb_ratio, dsp_ratio, bram_ratio),
    }


def compute_dl_attention_feasibility(R, C, n_seq, d_head, budget_pct):
    """4-resource feasibility for attention head on 60×60."""
    clb_ratio, dsp_ratio, bram_ratio = budget_to_ratios(budget_pct)
    tile_w, tile_h = get_tile_dims(R, C)
    total_dpes = count_available_wc(tile_w, tile_h, clb_ratio, dsp_ratio, bram_ratio)

    V = math.ceil(d_head / R)
    H = math.ceil(d_head / C)
    h_dimm = math.ceil(max(d_head, n_seq) / C)
    dpes_per_rep = 3 * V * H + 4 * h_dimm  # projections + DIMM

    p_dpe = total_dpes // dpes_per_rep if dpes_per_rep > 0 else 0

    clbs = avail_clbs(clb_ratio)
    p_clb = max(0, (clbs - ATTN_CLB_OVERHEAD) // ATTN_CLBS_PER_REP)

    brams = count_available_brams(bram_ratio)
    p_bram = brams // ATTN_BRAMS_PER_REP

    dsps = avail_dsps(dsp_ratio)
    p_dsp = dsps // ATTN_DSPS_PER_REP if ATTN_DSPS_PER_REP > 0 else 999

    P = min(p_dpe, p_clb, p_bram, p_dsp)

    limits = {'dpe': p_dpe, 'clb': p_clb, 'bram': p_bram, 'dsp': p_dsp}
    limit = min(limits, key=limits.get) if P >= 1 else 'none'

    return {
        'feasible': P >= 1, 'P': P,
        'p_dpe': p_dpe, 'p_clb': p_clb, 'p_bram': p_bram, 'p_dsp': p_dsp,
        'limit': limit, 'total_dpes': total_dpes,
        'V': V, 'H': H, 'h_dimm': h_dimm, 'dpes_per_rep': dpes_per_rep,
        'area_pct': dpe_area_pct(tile_w, tile_h, clb_ratio, dsp_ratio, bram_ratio),
    }


# ── Arch XML Generation ──────────────────────────────────────────────────

def gen_arch_xmls():
    """Generate 60×60 arch XMLs for all tile groups × budget levels."""
    from gen_arch_xml import gen_arch_xml

    tile_groups = {}
    for cfg in list(CONFIGS) + list(ATTN_CONFIGS):
        tw, th = get_tile_dims(cfg["R"], cfg["C"])
        key = f"tw{tw}"
        if key not in tile_groups:
            tile_groups[key] = (cfg["R"], cfg["C"], tw, th)

    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for tg_name, (R, C, tw, th) in tile_groups.items():
        for budget in BUDGET_LEVELS:
            out_name = f"nl_dpe_{tg_name}_b{budget}_fixed.xml"
            out_path = ARCH_DIR / out_name
            if out_path.exists():
                continue
            clb_ratio, dsp_ratio, bram_ratio = budget_to_ratios(budget)
            gen_arch_xml(
                R, C, mode="fixed_dsp_clb_replace",
                output_dir=ARCH_DIR,
                fixed_grid_w=GRID_W, fixed_grid_h=GRID_H,
                dsp_ratio=dsp_ratio, clb_ratio=clb_ratio,
                bram_ratio=bram_ratio,
            )
            # Rename from default name to tile-group + budget name
            dsp_pct = int(round(dsp_ratio * 100))
            clb_pct = int(round(clb_ratio * 100))
            bram_pct = int(round(bram_ratio * 100))
            default_name = f"nl_dpe_{R}x{C}_d{dsp_pct}_c{clb_pct}_b{bram_pct}_fixed.xml"
            src = ARCH_DIR / default_name
            if src.exists() and src != out_path:
                src.rename(out_path)
            count += 1
    print(f"Generated {count} arch XMLs in {ARCH_DIR}")
    return tile_groups


def arch_xml_path(tg_name, budget_pct):
    return ARCH_DIR / f"nl_dpe_{tg_name}_b{budget_pct}_fixed.xml"


# ── VTR Helpers ───────────────────────────────────────────────────────────

def parse_fmax(vpr_log: Path) -> float:
    text = vpr_log.read_text()
    m = re.search(r"Final critical path.*?Fmax:\s+([\d.]+)\s+MHz", text)
    return float(m.group(1)) if m else float("nan")


def parse_resources(vpr_log: Path) -> dict:
    text = vpr_log.read_text()
    resources = {"clb": 0, "dsp_top": 0, "memory": 0, "wc": 0}
    for bt in resources:
        m = re.search(rf"^\s+{bt}\s+(\d+)\s", text, re.MULTILINE)
        if m:
            resources[bt] = int(m.group(1))
    return resources


def run_vtr(design: Path, arch: Path, out_dir: Path, seed: int, timeout=1200):
    """Run single VTR flow. Returns (fmax, resources, status)."""
    design = Path(design).resolve()
    arch = Path(arch).resolve()
    out_dir = Path(out_dir).resolve()
    cmd = [
        sys.executable, str(VTR_FLOW),
        str(design), str(arch),
        "--route_chan_width", str(ROUTE_CHAN_WIDTH),
        "-temp_dir", str(out_dir),
        "--seed", str(seed),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        vpr_log = out_dir / "vpr_stdout.log"
        if vpr_log.exists():
            fmax = parse_fmax(vpr_log)
            res = parse_resources(vpr_log)
            return fmax, res, "OK" if fmax == fmax else "NO_FMAX"
        return float("nan"), {}, "NO_LOG"
    except subprocess.TimeoutExpired:
        return float("nan"), {}, "TIMEOUT"
    except Exception as e:
        return float("nan"), {}, f"ERROR: {e}"


def run_multi_seed(design, arch, base_dir, seeds, timeout=1200):
    """Run VTR for all seeds, average Fmax."""
    seed_fmax = []
    last_res = {}
    for seed in seeds:
        out_dir = base_dir / f"seed{seed}"
        fmax, res, status = run_vtr(design, arch, out_dir, seed, timeout)
        seed_fmax.append(fmax if status == "OK" else float("nan"))
        if status == "OK":
            last_res = res
    valid = [f for f in seed_fmax if f == f]
    avg = sum(valid) / len(valid) if valid else float("nan")
    return avg, seed_fmax, last_res, "OK" if valid else "FAILED"


def check_existing_seeds(base_dir, seeds):
    for seed in seeds:
        log = base_dir / f"seed{seed}" / "vpr_stdout.log"
        if not log.exists():
            return False
        if parse_fmax(log) != parse_fmax(log):  # NaN check
            return False
    return True


def load_existing_seeds(base_dir, seeds):
    seed_fmax = []
    last_res = {}
    for seed in seeds:
        log = base_dir / f"seed{seed}" / "vpr_stdout.log"
        fmax = parse_fmax(log)
        seed_fmax.append(fmax)
        if fmax == fmax:
            last_res = parse_resources(log)
    valid = [f for f in seed_fmax if f == f]
    avg = sum(valid) / len(valid) if valid else float("nan")
    return avg, seed_fmax, last_res


# ── Non-DL Sweep ─────────────────────────────────────────────────────────

def run_nondl_single(tg_name, R, C, budget, bname, bfile, seeds):
    """Run one non-DL benchmark point (all seeds)."""
    arch = arch_xml_path(tg_name, budget)
    bench = BENCH_DIR / bfile
    base_dir = OUTPUT_BASE / "nondl" / tg_name / f"b{budget}" / bname

    if not arch.exists():
        return {"tg": tg_name, "budget_pct": budget, "benchmark": bname,
                "fmax_mhz": float("nan"), "status": "MISSING_ARCH"}

    timeout = 900 if bname == "LU8PEEng" else 600
    avg, seeds_f, res, status = run_multi_seed(bench, arch, base_dir, seeds, timeout)

    result = {"tg": tg_name, "budget_pct": budget, "benchmark": bname,
              "fmax_mhz": avg, "status": status,
              "clb": res.get("clb", 0), "dsp_top": res.get("dsp_top", 0),
              "memory": res.get("memory", 0)}
    for i, sf in enumerate(seeds_f):
        result[f"fmax_seed{i+1}"] = sf
    return result


def run_nondl_sweep(args, seeds, sanity=False):
    """Run non-DL FlexScore sweep."""
    tile_groups = {}
    for cfg in list(CONFIGS) + list(ATTN_CONFIGS):
        tw, th = get_tile_dims(cfg["R"], cfg["C"])
        key = f"tw{tw}"
        if key not in tile_groups:
            tile_groups[key] = (cfg["R"], cfg["C"])

    jobs = []
    if sanity:
        for budget, bname in SANITY_POINTS:
            for tg_name in tile_groups:
                R, C = tile_groups[tg_name]
                jobs.append((tg_name, R, C, budget, bname, BENCHMARKS[bname]))
    else:
        for tg_name, (R, C) in tile_groups.items():
            for budget in BUDGET_LEVELS:
                for bname, bfile in BENCHMARKS.items():
                    jobs.append((tg_name, R, C, budget, bname, bfile))

    if args.skip_existing:
        pending, cached = [], []
        for job in jobs:
            tg, R, C, budget, bname, bfile = job
            base_dir = OUTPUT_BASE / "nondl" / tg / f"b{budget}" / bname
            if check_existing_seeds(base_dir, seeds):
                cached.append(job)
            else:
                pending.append(job)
        print(f"Non-DL: {len(jobs)} total | {len(cached)} cached | "
              f"{len(pending)} pending ({len(pending)*NUM_SEEDS} VTR runs)")
    else:
        pending, cached = jobs, []
        print(f"Non-DL: {len(jobs)} points × {NUM_SEEDS} seeds "
              f"= {len(jobs)*NUM_SEEDS} VTR runs")

    if args.dry_run:
        for j in pending:
            print(f"  {j[0]} budget={j[3]}% {j[4]}")
        return []

    results = []
    for job in cached:
        tg, R, C, budget, bname, bfile = job
        base_dir = OUTPUT_BASE / "nondl" / tg / f"b{budget}" / bname
        avg, sf, res = load_existing_seeds(base_dir, seeds)
        r = {"tg": tg, "budget_pct": budget, "benchmark": bname,
             "fmax_mhz": avg, "status": "OK (cached)",
             "clb": res.get("clb", 0), "dsp_top": res.get("dsp_top", 0),
             "memory": res.get("memory", 0)}
        for i, f in enumerate(sf):
            r[f"fmax_seed{i+1}"] = f
        results.append(r)

    if pending:
        print(f"Running {len(pending)} non-DL points (--jobs {args.jobs})...")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {}
            for job in pending:
                f = executor.submit(run_nondl_single, *job, seeds)
                futures[f] = job
            for i, f in enumerate(as_completed(futures), 1):
                r = f.result()
                results.append(r)
                print(f"  [{i}/{len(pending)}] {r['tg']} budget={r['budget_pct']}% "
                      f"{r['benchmark']}: avg={r['fmax_mhz']:.1f} MHz ({r['status']})")

    return results


# ── DL Sweep ──────────────────────────────────────────────────────────────

def run_dl_single(cfg_name, R, C, budget, wl_name, K, N, P, seeds):
    """Run one DL bare GEMV point (all seeds)."""
    from gen_gemm_wrapper import gen_gemm_wrapper

    tw, th = get_tile_dims(R, C)
    arch = arch_xml_path(f"tw{tw}", budget)
    rtl_dir = OUTPUT_BASE / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)

    rtl_path = gen_gemm_wrapper(K, N, R, C, P, str(rtl_dir))
    base_dir = OUTPUT_BASE / "dl" / cfg_name / f"b{budget}" / wl_name

    avg, seeds_f, res, status = run_multi_seed(
        Path(rtl_path), arch, base_dir, seeds, timeout=900)

    clb_ratio, dsp_ratio, bram_ratio = budget_to_ratios(budget)
    area = dpe_area_pct(tw, th, clb_ratio, dsp_ratio, bram_ratio)
    eff_lat = 1e3 / (P * avg) if P > 0 and avg == avg and avg > 0 else float("inf")

    result = {"config": cfg_name, "budget_pct": budget,
              "workload": wl_name, "P": P, "fmax_mhz": avg,
              "eff_latency_ns": eff_lat, "dpe_area_pct": area,
              "status": status}
    for i, sf in enumerate(seeds_f):
        result[f"fmax_seed{i+1}"] = sf
    return result


def run_dl_sweep(args, seeds):
    """Run DL bare GEMV sweep on 60×60."""
    jobs = []
    for cfg in CONFIGS:
        R, C = cfg["R"], cfg["C"]
        cfg_name = cfg["name"]
        for wl_name, K, N in DL_WORKLOADS:
            for budget in BUDGET_LEVELS:
                if budget == 0:
                    continue  # no DPEs at 0% budget
                feas = compute_dl_feasibility(R, C, K, N, budget)
                if not feas['feasible']:
                    continue
                jobs.append((cfg_name, R, C, budget, wl_name, K, N, feas['P']))

    print(f"DL: {len(jobs)} feasible points × {NUM_SEEDS} seeds "
          f"= {len(jobs)*NUM_SEEDS} VTR runs")

    if args.skip_existing:
        pending, cached = [], []
        for job in jobs:
            cfg_name, R, C, budget, wl_name, K, N, P = job
            base_dir = OUTPUT_BASE / "dl" / cfg_name / f"b{budget}" / wl_name
            if check_existing_seeds(base_dir, seeds):
                cached.append(job)
            else:
                pending.append(job)
        print(f"  Cached: {len(cached)} | Pending: {len(pending)}")
    else:
        pending, cached = jobs, []

    if args.dry_run:
        print(f"\n{'Config':>12} {'Workload':>14} {'Budget':>7} {'P':>4} {'Area%':>6}")
        for j in pending:
            cfg_name, R, C, budget, wl_name, K, N, P = j
            tw, th = get_tile_dims(R, C)
            clb_r, dsp_r, bram_r = budget_to_ratios(budget)
            area = dpe_area_pct(tw, th, clb_r, dsp_r, bram_r)
            print(f"{cfg_name:>12} {wl_name:>14} {budget:>6}% {P:>4} {area:>6.1f}")
        return []

    results = []
    for job in cached:
        cfg_name, R, C, budget, wl_name, K, N, P = job
        base_dir = OUTPUT_BASE / "dl" / cfg_name / f"b{budget}" / wl_name
        avg, sf, res = load_existing_seeds(base_dir, seeds)
        tw, th = get_tile_dims(R, C)
        clb_r, dsp_r, bram_r = budget_to_ratios(budget)
        area = dpe_area_pct(tw, th, clb_r, dsp_r, bram_r)
        eff_lat = 1e3 / (P * avg) if P > 0 and avg == avg and avg > 0 else float("inf")
        r = {"config": cfg_name, "budget_pct": budget,
             "workload": wl_name, "P": P, "fmax_mhz": avg,
             "eff_latency_ns": eff_lat, "dpe_area_pct": area,
             "status": "OK (cached)"}
        for i, f in enumerate(sf):
            r[f"fmax_seed{i+1}"] = f
        results.append(r)

    if pending:
        print(f"Running {len(pending)} DL points (--jobs {args.jobs})...")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {}
            for job in pending:
                f = executor.submit(run_dl_single, *job, seeds)
                futures[f] = job
            for i, f in enumerate(as_completed(futures), 1):
                r = f.result()
                results.append(r)
                cfg_name, _, _, budget, wl_name, _, _, P = futures[f]
                print(f"  [{i}/{len(pending)}] {cfg_name} {wl_name} "
                      f"budget={budget}% P={P}: "
                      f"Fmax={r['fmax_mhz']:.1f} lat={r['eff_latency_ns']:.2f} "
                      f"({r['status']})")

    return results


# ── Attention DL Sweep ───────────────────────────────────────────────────

def run_dl_attention_single(cfg_name, R, C, budget, wl_name, n_seq, d_head, P, seeds):
    """Run one attention DL point (all seeds)."""
    from gen_attention_gemm_wrapper import gen_attention_gemm_wrapper

    tw, th = get_tile_dims(R, C)
    arch = arch_xml_path(f"tw{tw}", budget)
    rtl_dir = OUTPUT_BASE / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)

    rtl_path = gen_attention_gemm_wrapper(n_seq, d_head, R, C, P, str(rtl_dir))
    base_dir = OUTPUT_BASE / "dl_attention" / cfg_name / f"b{budget}" / wl_name

    avg, seeds_f, res, status = run_multi_seed(
        Path(rtl_path), arch, base_dir, seeds, timeout=900)

    clb_ratio, dsp_ratio, bram_ratio = budget_to_ratios(budget)
    area = dpe_area_pct(tw, th, clb_ratio, dsp_ratio, bram_ratio)
    eff_lat = 1e3 / (P * avg) if P > 0 and avg == avg and avg > 0 else float("inf")

    result = {"config": cfg_name, "budget_pct": budget,
              "workload": wl_name, "wl_type": "attention",
              "n_seq": n_seq, "d_head": d_head,
              "P": P, "fmax_mhz": avg,
              "eff_latency_ns": eff_lat, "dpe_area_pct": area,
              "status": status}
    for i, sf in enumerate(seeds_f):
        result[f"fmax_seed{i+1}"] = sf
    return result


def run_dl_attention_sweep(args, seeds):
    """Run attention DL sweep on 60×60 for attention-optimal configs."""
    jobs = []
    for cfg in ATTN_CONFIGS:
        R, C = cfg["R"], cfg["C"]
        cfg_name = cfg["name"]
        for wl_name, n_seq, d_head in DL_ATTENTION_WORKLOADS:
            for budget in BUDGET_LEVELS:
                if budget == 0:
                    continue
                feas = compute_dl_attention_feasibility(R, C, n_seq, d_head, budget)
                if not feas['feasible']:
                    continue
                jobs.append((cfg_name, R, C, budget, wl_name, n_seq, d_head, feas['P']))

    print(f"DL Attention: {len(jobs)} feasible points × {NUM_SEEDS} seeds "
          f"= {len(jobs)*NUM_SEEDS} VTR runs")

    if args.skip_existing:
        pending, cached = [], []
        for job in jobs:
            cfg_name, R, C, budget, wl_name, n_seq, d_head, P = job
            base_dir = OUTPUT_BASE / "dl_attention" / cfg_name / f"b{budget}" / wl_name
            if check_existing_seeds(base_dir, seeds):
                cached.append(job)
            else:
                pending.append(job)
        print(f"  Cached: {len(cached)} | Pending: {len(pending)}")
    else:
        pending, cached = jobs, []

    if args.dry_run:
        print(f"\n{'Config':>12} {'Workload':>20} {'Budget':>7} {'P':>4} {'Area%':>6}")
        for j in pending:
            cfg_name, R, C, budget, wl_name, n_seq, d_head, P = j
            tw, th = get_tile_dims(R, C)
            clb_r, dsp_r, bram_r = budget_to_ratios(budget)
            area = dpe_area_pct(tw, th, clb_r, dsp_r, bram_r)
            print(f"{cfg_name:>12} {wl_name:>20} {budget:>6}% {P:>4} {area:>6.1f}")
        return []

    results = []
    for job in cached:
        cfg_name, R, C, budget, wl_name, n_seq, d_head, P = job
        base_dir = OUTPUT_BASE / "dl_attention" / cfg_name / f"b{budget}" / wl_name
        avg, sf, res = load_existing_seeds(base_dir, seeds)
        tw, th = get_tile_dims(R, C)
        clb_r, dsp_r, bram_r = budget_to_ratios(budget)
        area = dpe_area_pct(tw, th, clb_r, dsp_r, bram_r)
        eff_lat = 1e3 / (P * avg) if P > 0 and avg == avg and avg > 0 else float("inf")
        r = {"config": cfg_name, "budget_pct": budget,
             "workload": wl_name, "wl_type": "attention",
             "n_seq": n_seq, "d_head": d_head,
             "P": P, "fmax_mhz": avg,
             "eff_latency_ns": eff_lat, "dpe_area_pct": area,
             "status": "OK (cached)"}
        for i, f in enumerate(sf):
            r[f"fmax_seed{i+1}"] = f
        results.append(r)

    if pending:
        print(f"Running {len(pending)} attention DL points (--jobs {args.jobs})...")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {}
            for job in pending:
                f = executor.submit(run_dl_attention_single, *job, seeds)
                futures[f] = job
            for i, f in enumerate(as_completed(futures), 1):
                r = f.result()
                results.append(r)
                cfg_name, _, _, budget, wl_name, _, _, P = futures[f]
                print(f"  [{i}/{len(pending)}] {cfg_name} {wl_name} "
                      f"budget={budget}% P={P}: "
                      f"Fmax={r['fmax_mhz']:.1f} lat={r['eff_latency_ns']:.2f} "
                      f"({r['status']})")

    return results


# ── FlexScore Computation ────────────────────────────────────────────────

def compute_flexscore(nondl_results):
    """FlexScore per (tile_group, budget) = mean(Fmax/baseline) across benchmarks."""
    baselines = {}
    for r in nondl_results:
        if r["budget_pct"] == 0 and r["status"].startswith("OK"):
            baselines[(r["tg"], r["benchmark"])] = r["fmax_mhz"]

    by_arch = defaultdict(dict)
    for r in nondl_results:
        if r["status"].startswith("OK"):
            by_arch[(r["tg"], r["budget_pct"])][r["benchmark"]] = r["fmax_mhz"]

    flexscores = []
    for (tg, budget), bench_fmax in sorted(by_arch.items()):
        ratios = []
        for bname, fmax in bench_fmax.items():
            bl = baselines.get((tg, bname), 0)
            if bl > 0:
                ratios.append(fmax / bl)
            else:
                ratios.append(0.0)
        for bname in BENCHMARKS:
            if bname not in bench_fmax:
                if (tg, bname) in baselines:
                    ratios.append(0.0)

        fs = sum(ratios) / len(ratios) if ratios else 0.0
        clb_ratio, dsp_ratio, bram_ratio = budget_to_ratios(budget)
        flexscores.append({
            "tg": tg, "budget_pct": budget,
            "flexscore": fs, "n_benchmarks": len(ratios),
            "min_ratio": min(ratios) if ratios else 0,
            "max_ratio": max(ratios) if ratios else 0,
            "clb_remaining": avail_clbs(clb_ratio),
            "dsp_remaining": count_available_dsps(dsp_ratio),
            "bram_remaining": count_available_brams(bram_ratio),
        })
    return flexscores


# ── CSV Output ───────────────────────────────────────────────────────────

def write_nondl_csv(results):
    path = RESULTS_DIR / "flexscore_raw_results.csv"
    seed_cols = [f"fmax_seed{i+1}" for i in range(NUM_SEEDS)]
    fields = ["tg", "budget_pct", "benchmark", "fmax_mhz"] + seed_cols + [
        "clb", "dsp_top", "memory", "status"]
    rows = sorted(results, key=lambda r: (r["tg"], r["budget_pct"], r["benchmark"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Non-DL results -> {path} ({len(rows)} rows)")


def write_dl_csv(results):
    path = RESULTS_DIR / "flexscore_dl_gemv_results.csv"
    seed_cols = [f"fmax_seed{i+1}" for i in range(NUM_SEEDS)]
    fields = ["config", "budget_pct", "workload", "P", "fmax_mhz",
              "eff_latency_ns", "dpe_area_pct"] + seed_cols + ["status"]
    rows = sorted(results, key=lambda r: (r["config"], r["budget_pct"], r["workload"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"DL results -> {path} ({len(rows)} rows)")


def write_flexscore_csv(flexscores):
    path = RESULTS_DIR / "flexscore_summary.csv"
    fields = ["tg", "budget_pct", "flexscore", "n_benchmarks",
              "min_ratio", "max_ratio",
              "clb_remaining", "dsp_remaining", "bram_remaining"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flexscores)
    print(f"FlexScore summary -> {path}")

    print("\n-- FlexScore Summary --")
    print(f"{'TG':>5} {'Budget':>7} {'FlexScore':>10} {'Min':>6} {'Max':>6} "
          f"{'CLBs':>6} {'DSPs':>5} {'BRAMs':>6}")
    for fs in flexscores:
        print(f"{fs['tg']:>5} {fs['budget_pct']:>6}% "
              f"{fs['flexscore']:10.4f} {fs['min_ratio']:6.3f} {fs['max_ratio']:6.3f} "
              f"{fs['clb_remaining']:>6} {fs['dsp_remaining']:>5} {fs['bram_remaining']:>6}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FlexScore DSE v3: proportional budget sweep on 60x60 grid")
    parser.add_argument("--gen-arch", action="store_true",
                        help="Generate 60x60 arch XMLs")
    parser.add_argument("--nondl", action="store_true",
                        help="Run non-DL FlexScore sweep")
    parser.add_argument("--dl", action="store_true",
                        help="Run DL bare GEMV sweep")
    parser.add_argument("--dl-attention", action="store_true",
                        help="Run DL attention sweep (attention-optimal configs)")
    parser.add_argument("--sanity", action="store_true",
                        help="Run sanity check only")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if args.gen_arch:
        gen_arch_xmls()
        return

    seeds = get_or_create_seeds()

    if args.sanity:
        print("=== Sanity Check (non-DL) ===")
        results = run_nondl_sweep(args, seeds, sanity=True)
        if results:
            write_nondl_csv(results)
            flexscores = compute_flexscore(results)
            write_flexscore_csv(flexscores)
        return

    if args.nondl:
        print("=== Non-DL FlexScore Sweep ===")
        nondl_results = run_nondl_sweep(args, seeds)
        if nondl_results:
            write_nondl_csv(nondl_results)
            flexscores = compute_flexscore(nondl_results)
            write_flexscore_csv(flexscores)

    if args.dl:
        print("\n=== DL Bare GEMV Sweep ===")
        dl_results = run_dl_sweep(args, seeds)
        if dl_results:
            write_dl_csv(dl_results)

    if args.dl_attention:
        print("\n=== DL Attention Sweep ===")
        # Generate arch XMLs for attention configs (tw3 — all have tile_width=3)
        # Check if tw3 arch XMLs already exist, generate if needed
        for budget in BUDGET_LEVELS:
            arch_path = arch_xml_path("tw3", budget)
            if not arch_path.exists():
                print(f"  Missing arch XML: {arch_path} — run --gen-arch first")
                return
        dl_attn_results = run_dl_attention_sweep(args, seeds)
        if dl_attn_results:
            write_dl_attention_csv(dl_attn_results)


def write_dl_attention_csv(results):
    """Write attention DL results to CSV."""
    path = RESULTS_DIR / "flexscore_dl_attention_results.csv"
    seed_cols = [f"fmax_seed{i+1}" for i in range(NUM_SEEDS)]
    fields = ["config", "budget_pct", "workload", "wl_type",
              "n_seq", "d_head", "P", "fmax_mhz",
              "eff_latency_ns", "dpe_area_pct"] + seed_cols + ["status"]
    rows = sorted(results, key=lambda r: (r["config"], r["budget_pct"], r["workload"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"DL Attention results -> {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
