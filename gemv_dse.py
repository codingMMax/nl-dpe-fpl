#!/usr/bin/env python3
"""
GEMV/FC crossbar-size DSE orchestrator.

Round 1: sweep DPE_CONFIGS × FC_WORKLOADS with auto_layout, rank by throughput/mm².
Round 2: (placeholder) fixed-layout comparison for top-K configs.

Usage:
    python gemv_dse.py --round 1
    python gemv_dse.py --round 1 --configs 256x256,128x64 --workloads fc_64_64,fc_512_128
    python gemv_dse.py --round 1 --dry-run
    python gemv_dse.py --round 2 --r1-results dse/results/round1_results.csv
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT / "nl_dpe"))
from run_vtr import run_single, find_vpr_log, parse_metrics, parse_resources
from gen_arch_xml import gen_arch_xml
from gen_gemv_wrappers import gen_wrapper
from area_power import dpe_specs

# ── VTR paths ─────────────────────────────────────────────────────────────
VTR_ROOT = Path(os.environ.get("VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"))
VTR_FLOW = VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py"
VTR_PYTHON = VTR_ROOT / ".venv" / "bin" / "python"
if not VTR_PYTHON.is_file():
    VTR_PYTHON = None

# ── IMC paths ─────────────────────────────────────────────────────────────
AZURELILY_ROOT = PROJECT_ROOT / "azurelily"
IMC_TEST = AZURELILY_ROOT / "IMC" / "test.py"
BASE_IMC_CONFIG = AZURELILY_ROOT / "IMC" / "configs" / "nl_dpe.json"

# ── constants ─────────────────────────────────────────────────────────────
DPE_CONFIGS = [(r, c) for r in [128, 256, 512] for c in [64, 128, 256]]

FC_WORKLOADS = {
    "fc_64_64":    (64,   64),
    "fc_128_128":  (128,  128),
    "fc_512_128":  (512,  128),
    "fc_2048_256": (2048, 256),
    "fc_256_512":  (256,  512),
    "fc_512_512":  (512,  512),
}

CLB_TILE_UM2 = 2239

DEFAULT_ROUTE_CHAN_WIDTH = 300
DEFAULT_SEED = 42

CSV_COLUMNS = [
    'config', 'rows', 'cols',
    'workload', 'K', 'N',
    'V', 'H', 'dpe_count', 'acam_eligible',
    'fmax_mhz', 'grid_w', 'grid_h', 'fpga_area_mm2',
    'latency_ns', 'energy_pj',
    'throughput_per_mm2', 'throughput_per_J',
    'e_imc_vmm', 'e_imc_digital', 'e_clb_reduction', 'e_clb_activation', 'e_sram',
    'clb_count', 'dsp_count', 'mem_count', 'wc_count',
    'run_dir',
]

# ── IMC output parsing regexes ────────────────────────────────────────────
RE_ENERGY_LAYER = re.compile(r"Energy total \(by layer\):\s+([\d.]+)\s+pJ")
RE_LAT_CRIT = re.compile(r"Latency total \(critical path\):\s+([\d.]+)\s+ns")
RE_ENERGY_GROUPED = re.compile(
    r"Energy grouped:\s+DPE=([\d.]+)\s+pJ,\s+Memory=([\d.]+)\s+pJ,\s+FPGA=([\d.]+)\s+pJ"
)
# Component breakdown lines, e.g. "  imc_vmm: 75.0100 pJ"
RE_BREAKDOWN_LINE = re.compile(r"^\s+(\S+):\s+([\d.]+)\s+pJ", re.MULTILINE)

# ── VPR grid size parsing ────────────────────────────────────────────────
GRID_SIZE_RE = re.compile(r"FPGA sized to (\d+) x (\d+)")


def parse_grid_size(vpr_log_path: Path):
    """Parse FPGA grid dimensions from VPR log."""
    content = vpr_log_path.read_text(errors="replace")
    match = GRID_SIZE_RE.search(content)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Could not parse grid size from {vpr_log_path}")


# ── IMC config patching ──────────────────────────────────────────────────
def patch_imc_config(base_cfg: Path, rows: int, cols: int, fmax_mhz: float,
                     output_path: Path) -> Path:
    """Patch IMC config with DPE geometry and energy params at the given freq."""
    specs = dpe_specs(rows, cols, freq_ghz=fmax_mhz / 1000.0)

    with open(base_cfg) as f:
        cfg = json.load(f)

    cfg['geometry']['array_rows'] = rows
    cfg['geometry']['array_cols'] = cols
    cfg['params']['e_analoge_pj'] = specs['e_analogue_pj']
    cfg['params']['e_digital_pj'] = specs['e_digital_pj']
    cfg['fpga_specs']['freq'] = fmax_mhz

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cfg, f, indent=4)

    return output_path


# ── IMC runner ────────────────────────────────────────────────────────────
def run_imc_fc(config_path: Path, K: int, N: int):
    """Run the IMC simulator for an FC workload and parse results.

    Returns: (energy_pj, latency_ns, breakdown_dict)
    """
    cmd = [
        sys.executable, str(IMC_TEST),
        "--model", "fc",
        "--imc_file", str(config_path),
        "--seq_length", str(K),
        "--head_dim", str(N),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(AZURELILY_ROOT)
    )
    out = result.stdout + result.stderr

    # Parse energy total
    m = RE_ENERGY_LAYER.search(out)
    energy_pj = float(m.group(1)) if m else None

    # Parse latency
    m = RE_LAT_CRIT.search(out)
    latency_ns = float(m.group(1)) if m else None

    # Parse component breakdown
    breakdown = {}
    for m in RE_BREAKDOWN_LINE.finditer(out):
        breakdown[m.group(1)] = float(m.group(2))

    # Parse grouped breakdown
    m = RE_ENERGY_GROUPED.search(out)
    if m:
        breakdown['_dpe_grouped'] = float(m.group(1))
        breakdown['_mem_grouped'] = float(m.group(2))
        breakdown['_fpga_grouped'] = float(m.group(3))

    if result.returncode != 0 and energy_pj is None:
        print(f"    [IMC] WARNING: non-zero exit ({result.returncode}), "
              f"last output:\n{out[-500:]}", file=sys.stderr)

    return energy_pj, latency_ns, breakdown


# ── FC wrapper generation ─────────────────────────────────────────────────
def gen_fc_wrapper_for_dse(K: int, N: int, rows: int, cols: int,
                           output_dir: Path) -> Path:
    """Generate an FC wrapper Verilog file for DSE.

    Tries to import gen_fc_wrapper from gen_gemv_wrappers. If not available,
    falls back to gen_wrapper (GEMV wrapper with appropriate naming).
    """
    try:
        from gen_gemv_wrappers import gen_fc_wrapper
        return gen_fc_wrapper(K, N, rows, cols, output_dir=output_dir)
    except ImportError:
        pass

    # Fallback: use gen_wrapper (GEMV) and rename to fc_ convention
    # gen_wrapper writes to SCRIPT_DIR / f"gemv_{name}.v", so we need to
    # generate and then move/copy.
    name = f"1_{K}_{N}"
    src_path = gen_wrapper(name, K, N, rows=rows, cols=cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    dst_path = output_dir / f"fc_{K}_{N}_{rows}x{cols}.v"

    # Read, patch header comment, write to destination
    content = src_path.read_text()
    content = content.replace(
        f"// Auto-generated GEMV wrapper: {name}",
        f"// Auto-generated FC wrapper: fc_{K}_{N} on {rows}x{cols}",
    )
    dst_path.write_text(content)

    # Clean up the temporary file in SCRIPT_DIR
    if src_path.exists() and src_path != dst_path:
        src_path.unlink(missing_ok=True)

    v = math.ceil(K / rows)
    h = math.ceil(N / cols)
    print(f"  Generated {dst_path.name}  (V={v}, H={h})")
    return dst_path


# ── VTR DSE wrapper ───────────────────────────────────────────────────────
def run_vtr_dse(rtl: Path, arch: Path, run_dir: Path,
                rows: int, cols: int, K: int, N: int,
                wl_name: str) -> dict:
    """Run VTR for one DSE point and return parsed results."""
    config_name = f"{rows}x{cols}"
    design_name = f"fc_{K}_{N}_{config_name}"

    result = run_single(
        vtr_flow=VTR_FLOW,
        vtr_python=VTR_PYTHON,
        design=rtl,
        arch=arch,
        route_chan_width=DEFAULT_ROUTE_CHAN_WIDTH,
        sdc_file=None,
        run_dir=run_dir,
        seed=DEFAULT_SEED,
        run_index=0,
        total_runs=1,
        design_name=design_name,
    )

    # Parse grid size
    log_path = find_vpr_log(run_dir)
    try:
        grid_w, grid_h = parse_grid_size(log_path)
    except ValueError:
        grid_w, grid_h = 0, 0

    return {
        'fmax_mhz': result.fmax_mhz,
        'grid_w': grid_w,
        'grid_h': grid_h,
        'resources': result.resources,
        'run_dir': str(run_dir),
        'wirelength': result.wirelength,
        'elapsed_s': result.elapsed_s,
    }


# ── Top-K selection ───────────────────────────────────────────────────────
def compute_normalized_geomean(results: list) -> dict:
    """Compute per-config normalized geometric mean scores across workloads.

    For each workload, normalize each config's metric to the best config for
    that workload (best = 1.0, others ≤ 1.0).  Then take the geometric mean
    of the normalized values across all workloads.

    Returns a dict: config -> {'geomean_tput_mm2': float,
                                'geomean_tput_J':   float,
                                'geomean_combined': float,
                                'per_workload': {wl: {'norm_tput_mm2', 'norm_tput_J'}}}
    """
    # Collect metrics per workload across all configs
    workloads = sorted({r['workload'] for r in results})
    configs   = sorted({r['config']   for r in results})

    # index: (config, workload) -> row
    idx = {(r['config'], r['workload']): r for r in results}

    # Per-workload best values
    best_tput_mm2 = {}
    best_tput_J   = {}
    for wl in workloads:
        vals_mm2 = [idx[(c, wl)]['throughput_per_mm2']
                    for c in configs if (c, wl) in idx]
        vals_J   = [idx[(c, wl)]['throughput_per_J']
                    for c in configs if (c, wl) in idx]
        best_tput_mm2[wl] = max(vals_mm2) if vals_mm2 else 1.0
        best_tput_J[wl]   = max(vals_J)   if vals_J   else 1.0

    scores = {}
    for cfg in configs:
        norm_mm2_vals = []
        norm_J_vals   = []
        per_wl = {}
        for wl in workloads:
            if (cfg, wl) not in idx:
                continue
            r = idx[(cfg, wl)]
            n_mm2 = r['throughput_per_mm2'] / best_tput_mm2[wl] if best_tput_mm2[wl] > 0 else 0
            n_J   = r['throughput_per_J']   / best_tput_J[wl]   if best_tput_J[wl]   > 0 else 0
            norm_mm2_vals.append(n_mm2)
            norm_J_vals.append(n_J)
            per_wl[wl] = {'norm_tput_mm2': n_mm2, 'norm_tput_J': n_J}

        if not norm_mm2_vals:
            continue
        # Geometric mean: exp(mean(log(values)))
        gm_mm2 = math.exp(sum(math.log(v) for v in norm_mm2_vals) / len(norm_mm2_vals))
        gm_J   = math.exp(sum(math.log(v) for v in norm_J_vals)   / len(norm_J_vals))
        gm_combined = math.exp((math.log(gm_mm2) + math.log(gm_J)) / 2)

        scores[cfg] = {
            'geomean_tput_mm2': gm_mm2,
            'geomean_tput_J':   gm_J,
            'geomean_combined': gm_combined,
            'per_workload':     per_wl,
        }
    return scores


def select_top_configs(results: list, k: int = 3) -> list:
    """Select top-K configs by normalized geometric mean across workloads."""
    scores = compute_normalized_geomean(results)
    sorted_cfgs = sorted(scores.items(), key=lambda x: -x[1]['geomean_combined'])
    return [cfg for cfg, _ in sorted_cfgs[:k]]


# ── Round 1 ───────────────────────────────────────────────────────────────
def run_round1(args):
    """Execute Round 1 DSE: sweep configs × workloads, rank by throughput/mm²."""
    dse_dir = args.dse_dir
    arch_dir = dse_dir / "configs" / "arch"
    imc_dir = dse_dir / "configs" / "imc"
    rtl_dir = dse_dir / "rtl"
    r1_dir = dse_dir / "round1"
    results_dir = dse_dir / "results"

    for d in [arch_dir, imc_dir, rtl_dir, r1_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Parse config/workload filters
    if args.configs:
        configs = []
        for s in args.configs.split(","):
            s = s.strip()
            r, c = s.split("x")
            configs.append((int(r), int(c)))
    else:
        configs = DPE_CONFIGS

    if args.workloads:
        workload_names = [w.strip() for w in args.workloads.split(",")]
        workloads = {k: v for k, v in FC_WORKLOADS.items() if k in workload_names}
        missing = set(workload_names) - set(workloads.keys())
        if missing:
            print(f"WARNING: unknown workloads: {missing}", file=sys.stderr)
    else:
        workloads = FC_WORKLOADS

    # ── Phase 1: Generate artifacts ──────────────────────────────────────
    print("=" * 70)
    print("Phase 1: Generating architecture XMLs and RTL")
    print("=" * 70)

    for (R, C) in configs:
        gen_arch_xml(R, C, mode="auto", output_dir=arch_dir)
        for wl_name, (K, N) in workloads.items():
            gen_fc_wrapper_for_dse(K, N, R, C, output_dir=rtl_dir)

    # ── Phase 2: Build VTR job list ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: Building job list")
    print("=" * 70)

    jobs = []
    skipped_jobs = []
    for (R, C) in configs:
        arch_path = arch_dir / f"nl_dpe_{R}x{C}_auto.xml"
        if not arch_path.is_file():
            print(f"  WARNING: arch XML not found: {arch_path}", file=sys.stderr)
            continue

        for wl_name, (K, N) in workloads.items():
            rtl_path = rtl_dir / f"fc_{K}_{N}_{R}x{C}.v"
            if not rtl_path.is_file():
                print(f"  WARNING: RTL not found: {rtl_path}", file=sys.stderr)
                continue

            run_dir = r1_dir / f"{R}x{C}" / wl_name
            v = math.ceil(K / R)
            h = math.ceil(N / C)
            acam = (v == 1)
            job = {
                'rtl': rtl_path,
                'arch': arch_path,
                'run_dir': run_dir,
                'R': R, 'C': C, 'K': K, 'N': N,
                'wl_name': wl_name,
                'V': v, 'H': h,
                'dpe_count': v * h,
                'acam_eligible': acam,
            }
            if args.skip_existing and (run_dir / "imc_result.json").exists():
                print(f"  SKIP (existing): {R}x{C} / {wl_name}")
                skipped_jobs.append(job)
                continue
            jobs.append(job)

    print(f"\n  Total jobs: {len(jobs)}")

    # ── Dry-run: print table and exit ────────────────────────────────────
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN — job table")
        print("=" * 70)
        hdr = (f"{'Config':<10} {'Workload':<14} {'V':>3} {'H':>3} {'DPEs':>5} "
               f"{'acam':>5} {'RTL file':<35} {'Arch XML'}")
        print(hdr)
        print("-" * len(hdr))
        for j in jobs:
            rtl_name = j['rtl'].name
            arch_name = j['arch'].name
            acam_str = "yes" if j['acam_eligible'] else "no"
            print(f"{j['R']}x{j['C']:<6} {j['wl_name']:<14} "
                  f"{j['V']:>3} {j['H']:>3} {j['dpe_count']:>5} "
                  f"{acam_str:>5} {rtl_name:<35} {arch_name}")
        print(f"\n  {len(jobs)} jobs would be run.")
        return

    if not jobs and not skipped_jobs:
        print("  No jobs to run (all skipped or no valid configs).")
        return
    if not jobs:
        print("  No new jobs to run; will reload pre-existing results.")

    # ── Phase 3: Run VTR (parallel) ──────────────────────────────────────
    vtr_results = {}  # key: (R, C, wl_name) -> vtr result dict
    if jobs:
        print("\n" + "=" * 70)
        print("Phase 3: Running VTR synthesis & P&R")
        print("=" * 70)

        cpu_count = os.cpu_count() or 1
        max_workers = args.jobs if args.jobs > 0 else min(len(jobs), cpu_count)
        print(f"  Workers: {max_workers}")

        def _run_one_vtr(job):
            try:
                r = run_vtr_dse(
                    rtl=job['rtl'],
                    arch=job['arch'],
                    run_dir=job['run_dir'],
                    rows=job['R'], cols=job['C'],
                    K=job['K'], N=job['N'],
                    wl_name=job['wl_name'],
                )
                return job, r, None
            except Exception as exc:
                return job, None, str(exc)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_vtr, j): j for j in jobs}
            for fut in as_completed(futures):
                job, result, error = fut.result()
                key = (job['R'], job['C'], job['wl_name'])
                if error:
                    print(f"  FAILED: {key}: {error}", file=sys.stderr)
                else:
                    vtr_results[key] = result
                    print(f"  OK: {job['R']}x{job['C']} / {job['wl_name']}  "
                          f"fmax={result['fmax_mhz']:.1f} MHz  "
                          f"grid={result['grid_w']}x{result['grid_h']}")

        print(f"\n  Completed: {len(vtr_results)}/{len(jobs)}")

    # ── Phase 4: Run IMC simulator (sequential) ─────────────────────────
    all_results = []
    if jobs:
        print("\n" + "=" * 70)
        print("Phase 4: Running IMC energy/latency simulation")
        print("=" * 70)

    for job in jobs:
        key = (job['R'], job['C'], job['wl_name'])
        if key not in vtr_results:
            continue

        vtr = vtr_results[key]
        R, C = job['R'], job['C']
        K, N = job['K'], job['N']
        wl_name = job['wl_name']
        config_name = f"{R}x{C}"
        run_dir = Path(vtr['run_dir'])

        # Patch IMC config
        imc_cfg_path = imc_dir / f"nl_dpe_{config_name}_{wl_name}.json"
        patch_imc_config(BASE_IMC_CONFIG, R, C, vtr['fmax_mhz'], imc_cfg_path)

        # Run IMC
        print(f"  IMC: {config_name} / {wl_name} @ {vtr['fmax_mhz']:.1f} MHz ... ",
              end="", flush=True)
        energy_pj, latency_ns, breakdown = run_imc_fc(imc_cfg_path, K, N)

        if energy_pj is not None and latency_ns is not None:
            print(f"energy={energy_pj:.2f} pJ  latency={latency_ns:.2f} ns")
        else:
            print("PARSE FAILED")
            continue

        # Save imc_result.json
        imc_result = {
            'config': config_name,
            'workload': wl_name,
            'fmax_mhz': vtr['fmax_mhz'],
            'energy_pj': energy_pj,
            'latency_ns': latency_ns,
            'breakdown': breakdown,
        }
        imc_result_path = run_dir / "imc_result.json"
        imc_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(imc_result_path, 'w') as f:
            json.dump(imc_result, f, indent=2)

        # Compute derived metrics
        grid_w = vtr['grid_w']
        grid_h = vtr['grid_h']
        fpga_area_mm2 = grid_w * grid_h * CLB_TILE_UM2 / 1e6

        throughput_per_s = 1e9 / latency_ns if latency_ns > 0 else 0
        throughput_per_mm2 = throughput_per_s / fpga_area_mm2 if fpga_area_mm2 > 0 else 0
        throughput_per_J = 1e12 / energy_pj if energy_pj > 0 else 0  # energy in pJ

        resources = vtr.get('resources', {})

        row = {
            'config': config_name,
            'rows': R,
            'cols': C,
            'workload': wl_name,
            'K': K,
            'N': N,
            'V': job['V'],
            'H': job['H'],
            'dpe_count': job['dpe_count'],
            'acam_eligible': job['acam_eligible'],
            'fmax_mhz': vtr['fmax_mhz'],
            'grid_w': grid_w,
            'grid_h': grid_h,
            'fpga_area_mm2': fpga_area_mm2,
            'latency_ns': latency_ns,
            'energy_pj': energy_pj,
            'throughput_per_mm2': throughput_per_mm2,
            'throughput_per_J': throughput_per_J,
            'e_imc_vmm': breakdown.get('imc_vmm', 0),
            'e_imc_digital': breakdown.get('imc_digital_post', 0),
            'e_clb_reduction': breakdown.get('clb_reduction', 0),
            'e_clb_activation': breakdown.get('fpga_activation', 0),
            'e_sram': breakdown.get('sram_read', 0) + breakdown.get('sram_write', 0),
            'clb_count': resources.get('clb', 0),
            'dsp_count': resources.get('dsp_top', 0),
            'mem_count': resources.get('memory', 0),
            'wc_count': resources.get('wc', 0),
            'run_dir': str(run_dir),
        }
        all_results.append(row)

    # ── Phase 4b: Reload skipped (pre-existing) results ─────────────────
    if skipped_jobs:
        print(f"\n  Reloading {len(skipped_jobs)} pre-existing results...")
        for job in skipped_jobs:
            run_dir = Path(job['run_dir'])
            imc_path = run_dir / "imc_result.json"
            try:
                with open(imc_path) as f:
                    imc = json.load(f)
                log_path = find_vpr_log(run_dir)
                grid_w, grid_h = parse_grid_size(log_path)
                resources = parse_resources(log_path)
                R, C = job['R'], job['C']
                K, N = job['K'], job['N']
                fpga_area_mm2 = grid_w * grid_h * CLB_TILE_UM2 / 1e6
                latency_ns = imc['latency_ns']
                energy_pj = imc['energy_pj']
                breakdown = imc.get('breakdown', {})
                throughput_per_s = 1e9 / latency_ns if latency_ns > 0 else 0
                throughput_per_mm2 = throughput_per_s / fpga_area_mm2 if fpga_area_mm2 > 0 else 0
                throughput_per_J = 1e12 / energy_pj if energy_pj > 0 else 0
                all_results.append({
                    'config': f"{R}x{C}",
                    'rows': R, 'cols': C,
                    'workload': job['wl_name'],
                    'K': K, 'N': N,
                    'V': job['V'], 'H': job['H'],
                    'dpe_count': job['dpe_count'],
                    'acam_eligible': job['acam_eligible'],
                    'fmax_mhz': imc['fmax_mhz'],
                    'grid_w': grid_w, 'grid_h': grid_h,
                    'fpga_area_mm2': fpga_area_mm2,
                    'latency_ns': latency_ns,
                    'energy_pj': energy_pj,
                    'throughput_per_mm2': throughput_per_mm2,
                    'throughput_per_J': throughput_per_J,
                    'e_imc_vmm': breakdown.get('imc_vmm', 0),
                    'e_imc_digital': breakdown.get('imc_digital_post', 0),
                    'e_clb_reduction': breakdown.get('clb_reduction', 0),
                    'e_clb_activation': breakdown.get('fpga_activation', 0),
                    'e_sram': breakdown.get('sram_read', 0) + breakdown.get('sram_write', 0),
                    'clb_count': resources.get('clb', 0),
                    'dsp_count': resources.get('dsp_top', 0),
                    'mem_count': resources.get('memory', 0),
                    'wc_count': resources.get('wc', 0),
                    'run_dir': str(run_dir),
                })
                print(f"  LOADED: {R}x{C} / {job['wl_name']}  "
                      f"fmax={imc['fmax_mhz']:.1f} MHz  grid={grid_w}x{grid_h}")
            except Exception as e:
                print(f"  WARNING: could not reload {run_dir}: {e}", file=sys.stderr)

    # ── Phase 5: Write CSV ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 5: Writing results CSV")
    print("=" * 70)

    csv_path = results_dir / "round1_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in sorted(all_results, key=lambda r: (-r['throughput_per_mm2'])):
            writer.writerow(row)

    print(f"  Written: {csv_path}  ({len(all_results)} rows)")

    # ── Phase 6: Select top-K ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Phase 6: Selecting top-{args.top_k} configs")
    print("=" * 70)

    geomean_scores = compute_normalized_geomean(all_results)
    top_configs = select_top_configs(all_results, k=args.top_k)

    top_json = results_dir / "top3_configs.json"
    with open(top_json, 'w') as f:
        json.dump({
            'top_k': args.top_k,
            'metric': 'geomean_normalized_tput_mm2_and_J',
            'ranking': [
                {
                    'config': cfg,
                    'geomean_combined': geomean_scores[cfg]['geomean_combined'],
                    'geomean_tput_mm2': geomean_scores[cfg]['geomean_tput_mm2'],
                    'geomean_tput_J':   geomean_scores[cfg]['geomean_tput_J'],
                    'per_workload':     geomean_scores[cfg]['per_workload'],
                }
                for cfg, _ in sorted(geomean_scores.items(),
                                     key=lambda x: -x[1]['geomean_combined'])
            ],
            'top_configs': top_configs,
        }, f, indent=2)

    print(f"  Top-{args.top_k} configs: {top_configs}")
    print(f"  Written: {top_json}")

    # Print per-config normalized geomean ranking table
    print("\n" + "=" * 70)
    print("Config ranking (normalized geomean across workloads)")
    print("  Each metric normalized to best config per workload, then geomean'd")
    print("=" * 70)
    hdr2 = (f"{'Rank':>4} {'Config':<10} {'GM_Tput/mm2':>12} "
            f"{'GM_Tput/J':>11} {'GM_combined':>12}")
    print(hdr2)
    print("-" * len(hdr2))
    for rank, (cfg, sc) in enumerate(
            sorted(geomean_scores.items(), key=lambda x: -x[1]['geomean_combined']), 1):
        marker = " ◀ top" if cfg in top_configs else ""
        print(f"{rank:>4} {cfg:<10} {sc['geomean_tput_mm2']:>12.4f} "
              f"{sc['geomean_tput_J']:>11.4f} {sc['geomean_combined']:>12.4f}{marker}")

    # Print per-workload raw results table
    print("\n" + "=" * 70)
    print("Per-workload raw results")
    print("=" * 70)
    hdr = (f"{'Config':<10} {'Workload':<14} {'V':>2} {'H':>2} {'acam':>5} "
           f"{'Fmax':>7} {'Lat ns':>9} {'E pJ':>9} "
           f"{'Tput/mm2':>11} {'Tput/J':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(all_results, key=lambda x: (x['workload'], -x['throughput_per_mm2'])):
        acam_str = "yes" if r['acam_eligible'] else "no"
        print(f"{r['config']:<10} {r['workload']:<14} {r['V']:>2} {r['H']:>2} {acam_str:>5} "
              f"{r['fmax_mhz']:>7.1f} {r['latency_ns']:>9.2f} {r['energy_pj']:>9.2f} "
              f"{r['throughput_per_mm2']:>11.0f} {r['throughput_per_J']:>11.0f}")


# ── Round 2 (placeholder) ────────────────────────────────────────────────
def run_round2(args):
    """Round 2 not yet implemented."""
    print("Round 2 not yet implemented")
    return


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="GEMV/FC crossbar-size DSE orchestrator"
    )
    parser.add_argument(
        "--round", type=int, choices=[1, 2], required=True,
        help="DSE round (1=auto-layout sweep, 2=fixed-layout comparison)",
    )
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Comma-separated RxC configs, e.g. '256x256,128x64'",
    )
    parser.add_argument(
        "--workloads", type=str, default=None,
        help="Comma-separated workload names",
    )
    parser.add_argument(
        "--jobs", type=int, default=0,
        help="Parallel VTR workers (0=auto)",
    )
    parser.add_argument(
        "--dse-dir", type=Path, default=Path("dse"),
        help="Root directory for DSE outputs (default: dse/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print job table and exit without running VTR/IMC",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip configs that already have imc_result.json (default: True)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top configs to select (default: 3)",
    )
    parser.add_argument(
        "--r1-results", type=Path, default=None,
        help="Path to round1_results.csv (for Round 2 input)",
    )
    parser.add_argument(
        "--template-scale", type=float, default=1.2,
        help="Grid scale factor for Round 2 fixed layout (default: 1.2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.dse_dir = args.dse_dir.resolve()  # ensure absolute path for VTR subprocess

    print(f"GEMV/FC DSE — Round {args.round}")
    print(f"  DSE dir: {args.dse_dir}")
    print()

    if args.round == 1:
        run_round1(args)
    elif args.round == 2:
        run_round2(args)


if __name__ == "__main__":
    main()
