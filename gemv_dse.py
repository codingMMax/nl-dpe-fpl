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
from gen_attention_wrapper import gen_attention_wrapper
from gen_dsp_gemv_wrapper import gen_dsp_fc_wrapper
from gen_gemm_wrapper import gen_gemm_wrapper
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

ATTENTION_N_SEQ = 128
ATTENTION_D_HEAD = 128

DEFAULT_ROUTE_CHAN_WIDTH = 300
DEFAULT_SEED = 42
MULTI_SEEDS = [1, 2, 3]   # 3 seeds for stable Fmax averaging in Round 2

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


# ── IMC attention runner ─────────────────────────────────────────────────
ATTENTION_CSV_COLUMNS = [
    'config', 'rows', 'cols',
    'V_proj', 'H_proj', 'dpes_per_proj', 'total_dpes', 'acam_eligible',
    'fmax_mhz', 'grid_w', 'grid_h', 'fpga_area_mm2',
    'latency_ns', 'energy_pj',
    'throughput_per_mm2', 'throughput_per_J',
    'e_dpe_grouped', 'e_mem_grouped', 'e_fpga_grouped',
    'e_imc_vmm', 'e_imc_digital', 'e_sram',
    'clb_count', 'dsp_count', 'mem_count', 'wc_count',
    'run_dir',
]


def run_imc_attention(config_path: Path, n_seq: int, d_head: int):
    """Run the IMC simulator for an attention workload and parse results.

    Returns: (energy_pj, latency_ns, breakdown_dict)
    """
    cmd = [
        sys.executable, str(IMC_TEST),
        "--model", "attention",
        "--imc_file", str(config_path),
        "--seq_length", str(n_seq),
        "--head_dim", str(d_head),
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
        print(f"    [IMC-attn] WARNING: non-zero exit ({result.returncode}), "
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
            if args.skip_existing and not args.force and (run_dir / "imc_result.json").exists():
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


# ── Attention DSE ────────────────────────────────────────────────────────
def run_attention_dse(args):
    """Execute attention head DSE: 9 configs × 1 workload (N=128, d=128).

    Separate from Round 1/2 FC workloads.  Uses gen_attention_wrapper.py
    for RTL generation and --model attention for IMC energy/latency.
    """
    dse_dir = args.dse_dir
    arch_dir = dse_dir / "configs" / "arch"
    imc_dir = dse_dir / "configs" / "imc"
    rtl_dir = dse_dir / "rtl"
    attn_dir = dse_dir / "attention"
    results_dir = dse_dir / "results"

    for d in [arch_dir, imc_dir, rtl_dir, attn_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Parse config filter
    if args.configs:
        configs = []
        for s in args.configs.split(","):
            s = s.strip()
            r, c = s.split("x")
            configs.append((int(r), int(c)))
    else:
        configs = DPE_CONFIGS

    n_seq = ATTENTION_N_SEQ
    d_head = ATTENTION_D_HEAD

    # ── Phase 1: Generate artifacts ──────────────────────────────────────
    print("=" * 70)
    print("Phase 1: Generating architecture XMLs and attention RTL")
    print("=" * 70)

    for (R, C) in configs:
        gen_arch_xml(R, C, mode="auto", output_dir=arch_dir)
        gen_attention_wrapper(n_seq, d_head, R, C, output_dir=str(rtl_dir))

    # ── Phase 2: Build job list ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: Building attention job list")
    print("=" * 70)

    jobs = []
    skipped_jobs = []
    for (R, C) in configs:
        arch_path = arch_dir / f"nl_dpe_{R}x{C}_auto.xml"
        if not arch_path.is_file():
            print(f"  WARNING: arch XML not found: {arch_path}", file=sys.stderr)
            continue

        rtl_path = rtl_dir / f"attention_{R}x{C}.v"
        if not rtl_path.is_file():
            print(f"  WARNING: attention RTL not found: {rtl_path}", file=sys.stderr)
            continue

        run_dir = attn_dir / f"{R}x{C}"
        v = math.ceil(d_head / R)
        h = math.ceil(d_head / C)
        job = {
            'rtl': rtl_path,
            'arch': arch_path,
            'run_dir': run_dir,
            'R': R, 'C': C,
            'V': v, 'H': h,
            'dpes_per_proj': v * h,
            'total_dpes': 3 * v * h,
            'acam_eligible': (v == 1),
        }
        if args.skip_existing and not args.force and (run_dir / "imc_result.json").exists():
            print(f"  SKIP (existing): {R}x{C}")
            skipped_jobs.append(job)
            continue
        jobs.append(job)

    print(f"\n  Total jobs: {len(jobs)}, skipped: {len(skipped_jobs)}")

    # ── Dry-run ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n  DRY RUN — attention job table:")
        for j in jobs:
            print(f"    {j['R']}x{j['C']}  V={j['V']} H={j['H']}  "
                  f"DPEs={j['total_dpes']}  ACAM={'yes' if j['acam_eligible'] else 'no'}")
        return

    if not jobs and not skipped_jobs:
        print("  No jobs to run.")
        return

    # ── Phase 3: Run VTR (parallel) ──────────────────────────────────────
    vtr_results = {}
    if jobs:
        print("\n" + "=" * 70)
        print("Phase 3: Running VTR synthesis & P&R (attention)")
        print("=" * 70)

        cpu_count = os.cpu_count() or 1
        max_workers = args.jobs if args.jobs > 0 else min(len(jobs), cpu_count)
        print(f"  Workers: {max_workers}")

        def _run_one_vtr_attn(job):
            try:
                R, C = job['R'], job['C']
                config_name = f"{R}x{C}"
                design_name = f"attention_{config_name}"

                result = run_single(
                    vtr_flow=VTR_FLOW,
                    vtr_python=VTR_PYTHON,
                    design=job['rtl'],
                    arch=job['arch'],
                    route_chan_width=DEFAULT_ROUTE_CHAN_WIDTH,
                    sdc_file=None,
                    run_dir=job['run_dir'],
                    seed=DEFAULT_SEED,
                    run_index=0,
                    total_runs=1,
                    design_name=design_name,
                )

                log_path = find_vpr_log(job['run_dir'])
                try:
                    grid_w, grid_h = parse_grid_size(log_path)
                except ValueError:
                    grid_w, grid_h = 0, 0

                return job, {
                    'fmax_mhz': result.fmax_mhz,
                    'grid_w': grid_w,
                    'grid_h': grid_h,
                    'resources': result.resources,
                    'run_dir': str(job['run_dir']),
                    'wirelength': result.wirelength,
                    'elapsed_s': result.elapsed_s,
                }, None
            except Exception as exc:
                return job, None, str(exc)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_vtr_attn, j): j for j in jobs}
            for fut in as_completed(futures):
                job, result, error = fut.result()
                key = (job['R'], job['C'])
                if error:
                    print(f"  FAILED: {key}: {error}", file=sys.stderr)
                else:
                    vtr_results[key] = result
                    print(f"  OK: {job['R']}x{job['C']}  "
                          f"fmax={result['fmax_mhz']:.1f} MHz  "
                          f"grid={result['grid_w']}x{result['grid_h']}")

        print(f"\n  Completed: {len(vtr_results)}/{len(jobs)}")

    # ── Phase 4: Run IMC attention model ─────────────────────────────────
    all_results = []
    if jobs:
        print("\n" + "=" * 70)
        print("Phase 4: Running IMC attention energy/latency simulation")
        print("=" * 70)

    for job in jobs:
        key = (job['R'], job['C'])
        if key not in vtr_results:
            continue

        vtr = vtr_results[key]
        R, C = job['R'], job['C']
        config_name = f"{R}x{C}"
        run_dir = Path(vtr['run_dir'])

        # Patch IMC config with DPE geometry + measured Fmax
        imc_cfg_path = imc_dir / f"nl_dpe_{config_name}_attention.json"
        patch_imc_config(BASE_IMC_CONFIG, R, C, vtr['fmax_mhz'], imc_cfg_path)

        # Run IMC
        print(f"  IMC: {config_name} / attention @ {vtr['fmax_mhz']:.1f} MHz ... ",
              end="", flush=True)
        energy_pj, latency_ns, breakdown = run_imc_attention(
            imc_cfg_path, n_seq, d_head)

        if energy_pj is not None and latency_ns is not None:
            print(f"energy={energy_pj:.2f} pJ  latency={latency_ns:.2f} ns")
        else:
            print("PARSE FAILED")
            continue

        # Save imc_result.json
        imc_result = {
            'config': config_name,
            'workload': 'attention',
            'n_seq': n_seq,
            'd_head': d_head,
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
        throughput_per_J = 1e12 / energy_pj if energy_pj > 0 else 0

        resources = vtr.get('resources', {})

        row = {
            'config': config_name,
            'rows': R,
            'cols': C,
            'V_proj': job['V'],
            'H_proj': job['H'],
            'dpes_per_proj': job['dpes_per_proj'],
            'total_dpes': job['total_dpes'],
            'acam_eligible': job['acam_eligible'],
            'fmax_mhz': vtr['fmax_mhz'],
            'grid_w': grid_w,
            'grid_h': grid_h,
            'fpga_area_mm2': fpga_area_mm2,
            'latency_ns': latency_ns,
            'energy_pj': energy_pj,
            'throughput_per_mm2': throughput_per_mm2,
            'throughput_per_J': throughput_per_J,
            'e_dpe_grouped': breakdown.get('_dpe_grouped', 0),
            'e_mem_grouped': breakdown.get('_mem_grouped', 0),
            'e_fpga_grouped': breakdown.get('_fpga_grouped', 0),
            'e_imc_vmm': breakdown.get('imc_vmm', 0),
            'e_imc_digital': breakdown.get('imc_digital_post', 0),
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
                    'V_proj': job['V'], 'H_proj': job['H'],
                    'dpes_per_proj': job['dpes_per_proj'],
                    'total_dpes': job['total_dpes'],
                    'acam_eligible': job['acam_eligible'],
                    'fmax_mhz': imc['fmax_mhz'],
                    'grid_w': grid_w, 'grid_h': grid_h,
                    'fpga_area_mm2': fpga_area_mm2,
                    'latency_ns': latency_ns,
                    'energy_pj': energy_pj,
                    'throughput_per_mm2': throughput_per_mm2,
                    'throughput_per_J': throughput_per_J,
                    'e_dpe_grouped': breakdown.get('_dpe_grouped', 0),
                    'e_mem_grouped': breakdown.get('_mem_grouped', 0),
                    'e_fpga_grouped': breakdown.get('_fpga_grouped', 0),
                    'e_imc_vmm': breakdown.get('imc_vmm', 0),
                    'e_imc_digital': breakdown.get('imc_digital_post', 0),
                    'e_sram': breakdown.get('sram_read', 0) + breakdown.get('sram_write', 0),
                    'clb_count': resources.get('clb', 0),
                    'dsp_count': resources.get('dsp_top', 0),
                    'mem_count': resources.get('memory', 0),
                    'wc_count': resources.get('wc', 0),
                    'run_dir': str(run_dir),
                })
                print(f"  LOADED: {R}x{C}  "
                      f"fmax={imc['fmax_mhz']:.1f} MHz  grid={grid_w}x{grid_h}")
            except Exception as e:
                print(f"  WARNING: could not reload {run_dir}: {e}", file=sys.stderr)

    # ── Phase 5: Write attention_results.csv ─────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 5: Writing attention results CSV")
    print("=" * 70)

    csv_path = results_dir / "attention_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ATTENTION_CSV_COLUMNS)
        writer.writeheader()
        for row in sorted(all_results, key=lambda r: -r['throughput_per_mm2']):
            writer.writerow(row)

    print(f"  Written: {csv_path}  ({len(all_results)} rows)")

    # ── Phase 6: Print summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Attention DSE Results (N={n_seq}, d={d_head})")
    print("=" * 70)
    hdr = (f"{'Config':<10} {'V':>2} {'H':>2} {'DPEs':>5} {'acam':>5} "
           f"{'Fmax':>7} {'Grid':>8} {'Area mm2':>9} "
           f"{'Lat ns':>9} {'E pJ':>9} "
           f"{'Tput/mm2':>11} {'Tput/J':>11} "
           f"{'E_DPE':>8} {'E_FPGA':>8} {'E_Mem':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(all_results, key=lambda x: -x['throughput_per_mm2']):
        acam_str = "yes" if r['acam_eligible'] else "no"
        grid_str = f"{r['grid_w']}x{r['grid_h']}"
        print(f"{r['config']:<10} {r['V_proj']:>2} {r['H_proj']:>2} "
              f"{r['total_dpes']:>5} {acam_str:>5} "
              f"{r['fmax_mhz']:>7.1f} {grid_str:>8} {r['fpga_area_mm2']:>9.3f} "
              f"{r['latency_ns']:>9.2f} {r['energy_pj']:>9.2f} "
              f"{r['throughput_per_mm2']:>11.0f} {r['throughput_per_J']:>11.0f} "
              f"{r['e_dpe_grouped']:>8.2f} {r['e_fpga_grouped']:>8.2f} "
              f"{r['e_mem_grouped']:>8.2f}")


# ── Round 2 ──────────────────────────────────────────────────────────────

CLB_REPLACE_RATIOS = [0.05, 0.08, 0.12, 0.15]

# DSP+BRAM comparison: # DSP/BRAM columns to place in fixed grid
DSP_BRAM_EXTRA_DSPS = 4   # DSP columns (each column ~26 DSP tiles at height 4)
DSP_BRAM_EXTRA_BRAMS = 8  # BRAM columns (each column ~52 memory tiles at height 2)
DSP_BRAM_MAX_PARALLEL = 32  # parallel MAC channels per DSP design

ROUND2_CSV_COLUMNS = [
    'mode', 'clb_replace_ratio', 'pair_name',
    'config', 'rows', 'cols',
    'workload', 'wl_type', 'K', 'N',
    'V', 'H', 'dpe_count', 'acam_eligible',
    'fmax_mhz', 'grid_w', 'grid_h', 'fpga_area_mm2',
    'latency_ns', 'energy_pj',
    'throughput_per_mm2', 'throughput_per_J',
    'e_imc_vmm', 'e_imc_digital', 'e_clb_reduction', 'e_clb_activation', 'e_sram',
    'e_dpe_grouped', 'e_mem_grouped', 'e_fpga_grouped',
    'clb_count', 'dsp_count', 'mem_count', 'wc_count',
    'run_dir',
]


def _run_round2_dsp_bram(args, fixed_w, fixed_h, configs, workloads,
                          arch_dir, rtl_dir, r2_dir, results_dir,
                          clb_replace_results):
    """Round 2 Part 2: DSP+BRAM equivalence — VTR synthesis of DSP-based FC designs.

    For each FC workload, generates a DSP-based GEMV design (no DPE hard blocks),
    runs VTR on a fixed grid with DSP+BRAM resources, and compares throughput/area
    against the DPE-based results from Part 1.
    """
    # Only FC workloads — DSP can't implement attention DIMM/softmax
    fc_workloads = [w for w in workloads if w['type'] == 'fc']

    print("\n" + "=" * 70)
    print("Round 2 Part 2 — DSP+BRAM Equivalence Comparison")
    print("=" * 70)
    print(f"  Fixed grid: {fixed_w}x{fixed_h}")
    print(f"  DSP columns: {DSP_BRAM_EXTRA_DSPS}, BRAM columns: {DSP_BRAM_EXTRA_BRAMS}")
    print(f"  Parallel MACs: {DSP_BRAM_MAX_PARALLEL}")
    print(f"  FC workloads: {[w['name'] for w in fc_workloads]}")

    # ── Generate arch XML (one shared, no DPE tiles) ─────────────────────
    # Use a dummy R,C — the arch XML's DPE tile is irrelevant since we don't
    # place any wc columns, but gen_arch_xml requires R,C for the template.
    dummy_R, dummy_C = configs[0]
    arch_path = gen_arch_xml(
        dummy_R, dummy_C, mode="fixed_dsp_bram", output_dir=str(arch_dir),
        fixed_grid_w=fixed_w, fixed_grid_h=fixed_h,
        extra_dsps=DSP_BRAM_EXTRA_DSPS, extra_brams=DSP_BRAM_EXTRA_BRAMS,
        pair_name="dsp_bram",
    )

    # ── Generate DSP RTL for each workload ───────────────────────────────
    dsp_rtl_dir = rtl_dir / "dsp"
    dsp_rtl_dir.mkdir(parents=True, exist_ok=True)
    for wl in fc_workloads:
        gen_dsp_fc_wrapper(wl['K'], wl['N'],
                           output_dir=str(dsp_rtl_dir),
                           max_parallel=DSP_BRAM_MAX_PARALLEL)

    # ── Build DSP job list ───────────────────────────────────────────────
    dsp_jobs = []
    dsp_skipped = []
    for wl in fc_workloads:
        K, N = wl['K'], wl['N']
        wl_name = wl['name']
        rtl_path = dsp_rtl_dir / f"dsp_fc_{K}_{N}.v"
        if not rtl_path.is_file():
            print(f"  WARNING: DSP RTL not found: {rtl_path}", file=sys.stderr)
            continue
        run_dir = r2_dir / "dsp_bram" / wl_name
        job = {
            'rtl': rtl_path,
            'arch': arch_path,
            'run_dir': run_dir,
            'K': K, 'N': N,
            'wl_name': wl_name,
        }
        if args.skip_existing and not args.force and \
           (run_dir / "dsp_result.json").exists():
            print(f"  SKIP (existing): dsp_bram/{wl_name}")
            dsp_skipped.append(job)
            continue
        dsp_jobs.append(job)

    print(f"  DSP jobs: {len(dsp_jobs)}, skipped: {len(dsp_skipped)}")

    if args.dry_run:
        return

    # ── Run VTR (parallel) ───────────────────────────────────────────────
    dsp_vtr_results = {}
    if dsp_jobs:
        cpu_count = os.cpu_count() or 1
        max_workers = args.jobs if args.jobs > 0 else min(len(dsp_jobs), cpu_count)

        def _run_one_dsp_vtr(job):
            try:
                design_name = f"dsp_fc_{job['K']}_{job['N']}"
                r = run_single(
                    vtr_flow=VTR_FLOW,
                    vtr_python=VTR_PYTHON,
                    design=job['rtl'],
                    arch=job['arch'],
                    route_chan_width=DEFAULT_ROUTE_CHAN_WIDTH,
                    sdc_file=None,
                    run_dir=job['run_dir'],
                    seed=DEFAULT_SEED,
                    run_index=0,
                    total_runs=1,
                    design_name=design_name,
                )
                log_path = find_vpr_log(job['run_dir'])
                try:
                    grid_w, grid_h = parse_grid_size(log_path)
                except ValueError:
                    grid_w, grid_h = fixed_w, fixed_h
                return job, {
                    'fmax_mhz': r.fmax_mhz,
                    'grid_w': grid_w,
                    'grid_h': grid_h,
                    'resources': r.resources,
                    'run_dir': str(job['run_dir']),
                }, None
            except Exception as exc:
                return job, None, str(exc)

        print(f"  Running {len(dsp_jobs)} DSP VTR jobs ({max_workers} workers)...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_dsp_vtr, j): j for j in dsp_jobs}
            for fut in as_completed(futures):
                job, result, error = fut.result()
                if error:
                    print(f"  FAILED: dsp_bram/{job['wl_name']}: {error}",
                          file=sys.stderr)
                else:
                    dsp_vtr_results[job['wl_name']] = result
                    print(f"  OK: dsp_bram/{job['wl_name']}  "
                          f"fmax={result['fmax_mhz']:.1f} MHz  "
                          f"dsp={result['resources'].get('dsp_top', 0)}  "
                          f"mem={result['resources'].get('memory', 0)}")

    # ── Compute DSP results ──────────────────────────────────────────────
    dsp_results = []
    for job in dsp_jobs + dsp_skipped:
        wl_name = job['wl_name']
        K, N = job['K'], job['N']
        run_dir = Path(job['run_dir'])

        if wl_name in dsp_vtr_results:
            vtr = dsp_vtr_results[wl_name]
        elif (run_dir / "dsp_result.json").exists():
            # Reload from saved result
            try:
                with open(run_dir / "dsp_result.json") as f:
                    saved = json.load(f)
                log_path = find_vpr_log(run_dir)
                resources = parse_resources(log_path)
                try:
                    grid_w, grid_h = parse_grid_size(log_path)
                except ValueError:
                    grid_w, grid_h = fixed_w, fixed_h
                vtr = {
                    'fmax_mhz': saved['fmax_mhz'],
                    'grid_w': grid_w,
                    'grid_h': grid_h,
                    'resources': resources,
                }
            except Exception as e:
                print(f"  WARNING: reload failed for dsp_bram/{wl_name}: {e}",
                      file=sys.stderr)
                continue
        else:
            continue

        P = min(N, DSP_BRAM_MAX_PARALLEL)
        batches = math.ceil(N / P)
        latency_cycles = K * batches + P
        fmax_hz = vtr['fmax_mhz'] * 1e6
        latency_ns = latency_cycles / fmax_hz * 1e9 if fmax_hz > 0 else 0

        grid_w, grid_h = vtr['grid_w'], vtr['grid_h']
        fpga_area_mm2 = grid_w * grid_h * CLB_TILE_UM2 / 1e6
        throughput_per_s = 1e9 / latency_ns if latency_ns > 0 else 0
        throughput_per_mm2 = throughput_per_s / fpga_area_mm2 if fpga_area_mm2 > 0 else 0

        resources = vtr.get('resources', {})

        # Save result JSON
        dsp_result = {
            'workload': wl_name,
            'fmax_mhz': vtr['fmax_mhz'],
            'latency_cycles': latency_cycles,
            'latency_ns': latency_ns,
            'P': P,
            'batches': batches,
        }
        result_path = run_dir / "dsp_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(dsp_result, f, indent=2)

        row = {
            'mode': 'dsp_bram',
            'clb_replace_ratio': 0,
            'pair_name': 'dsp_bram',
            'config': 'dsp_bram',
            'rows': 0, 'cols': 0,
            'workload': wl_name,
            'wl_type': 'fc',
            'K': K, 'N': N,
            'V': 0, 'H': 0,
            'dpe_count': 0,
            'acam_eligible': False,
            'fmax_mhz': vtr['fmax_mhz'],
            'grid_w': grid_w, 'grid_h': grid_h,
            'fpga_area_mm2': fpga_area_mm2,
            'latency_ns': latency_ns,
            'energy_pj': 0,  # no IMC model for DSP
            'throughput_per_mm2': throughput_per_mm2,
            'throughput_per_J': 0,  # no energy model
            'e_imc_vmm': 0, 'e_imc_digital': 0,
            'e_clb_reduction': 0, 'e_clb_activation': 0, 'e_sram': 0,
            'e_dpe_grouped': 0, 'e_mem_grouped': 0, 'e_fpga_grouped': 0,
            'clb_count': resources.get('clb', 0),
            'dsp_count': resources.get('dsp_top', 0),
            'mem_count': resources.get('memory', 0),
            'wc_count': resources.get('wc', 0),
            'run_dir': str(run_dir),
        }
        dsp_results.append(row)

    # ── Write DSP comparison CSV ─────────────────────────────────────────
    if dsp_results:
        csv_path = results_dir / "round2_dsp_comparison.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ROUND2_CSV_COLUMNS,
                                     extrasaction='ignore')
            writer.writeheader()
            for row in sorted(dsp_results, key=lambda r: r['workload']):
                writer.writerow(row)
        print(f"\n  Written: {csv_path}  ({len(dsp_results)} rows)")

        # Summary
        print("\n  DSP+BRAM Results:")
        print(f"  {'Workload':<14} {'Fmax':>7} {'DSP':>4} {'MEM':>4} {'CLB':>5} "
              f"{'Lat ns':>10} {'Tput/mm2':>11}")
        print("  " + "-" * 60)
        for r in sorted(dsp_results, key=lambda x: x['workload']):
            print(f"  {r['workload']:<14} {r['fmax_mhz']:>7.1f} "
                  f"{r['dsp_count']:>4} {r['mem_count']:>4} {r['clb_count']:>5} "
                  f"{r['latency_ns']:>10.1f} {r['throughput_per_mm2']:>11.0f}")

        # Add DSP results to the main Round 2 CSV
        clb_replace_results.extend(dsp_results)


def run_round2(args):
    """Execute Round 2 DSE: fixed-layout CLB replacement sweep on top-3 configs.

    Part 1: CLB replacement sweep (4 ratios × 3 configs × 7 workloads = 84 runs)
    Part 2: DSP+BRAM equivalence (blocked — needs DSP-based RTL generator)
    """
    dse_dir = args.dse_dir
    results_dir = dse_dir / "results"
    arch_dir = dse_dir / "configs" / "arch"
    imc_dir = dse_dir / "configs" / "imc"
    rtl_dir = dse_dir / "rtl"
    r2_dir = dse_dir / "round2"

    for d in [arch_dir, imc_dir, rtl_dir, r2_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────
    template_path = results_dir / "round2_template.json"
    top3_path = results_dir / "top3_configs.json"

    if not template_path.exists():
        print("ERROR: round2_template.json not found. Run Step 4 first.",
              file=sys.stderr)
        return
    if not top3_path.exists():
        print("ERROR: top3_configs.json not found. Run Round 1 first.",
              file=sys.stderr)
        return

    with open(template_path) as f:
        tmpl = json.load(f)
    fixed_w = tmpl['fixed_grid_w']
    fixed_h = tmpl['fixed_grid_h']

    with open(top3_path) as f:
        top3_data = json.load(f)
    top3_configs = top3_data['top_configs']

    configs = []
    for s in top3_configs:
        r, c = s.split("x")
        configs.append((int(r), int(c)))

    # FC + attention workloads
    workloads = []
    for wl_name, (K, N) in FC_WORKLOADS.items():
        workloads.append({
            'name': wl_name, 'type': 'fc', 'K': K, 'N': N,
        })
    workloads.append({
        'name': 'attention', 'type': 'attention',
        'K': ATTENTION_D_HEAD, 'N': ATTENTION_D_HEAD,
    })

    print(f"Round 2 DSE — Part 1: CLB Replacement Sweep")
    print(f"  Fixed grid: {fixed_w}x{fixed_h}")
    print(f"  Top-3 configs: {top3_configs}")
    print(f"  Ratios: {CLB_REPLACE_RATIOS}")
    print(f"  Workloads: {[w['name'] for w in workloads]}")
    print(f"  DSE dir: {dse_dir}")

    # ── Phase 1: Generate arch XMLs and RTL ──────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 1: Generating architecture XMLs and RTL")
    print("=" * 70)

    for (R, C) in configs:
        for ratio in CLB_REPLACE_RATIOS:
            gen_arch_xml(
                R, C, mode="fixed_clb_replace", output_dir=arch_dir,
                fixed_grid_w=fixed_w, fixed_grid_h=fixed_h,
                clb_replace_ratio=ratio,
            )
        # Generate RTL for all workloads (reuse if already exists)
        for wl in workloads:
            if wl['type'] == 'fc':
                gen_fc_wrapper_for_dse(wl['K'], wl['N'], R, C,
                                       output_dir=rtl_dir)
            else:
                gen_attention_wrapper(
                    ATTENTION_N_SEQ, ATTENTION_D_HEAD, R, C,
                    output_dir=str(rtl_dir),
                )

    # ── Phase 2: Build job list ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: Building job list")
    print("=" * 70)

    jobs = []
    skipped_jobs = []

    for (R, C) in configs:
        config_name = f"{R}x{C}"
        for ratio in CLB_REPLACE_RATIOS:
            pct = int(round(ratio * 100))
            arch_name = f"nl_dpe_{R}x{C}_clb{pct}_fixed.xml"
            arch_path = arch_dir / arch_name
            if not arch_path.is_file():
                print(f"  WARNING: arch XML not found: {arch_path}",
                      file=sys.stderr)
                continue

            for wl in workloads:
                wl_name = wl['name']
                K, N = wl['K'], wl['N']

                if wl['type'] == 'fc':
                    rtl_path = rtl_dir / f"fc_{K}_{N}_{R}x{C}.v"
                    v = math.ceil(K / R)
                    h = math.ceil(N / C)
                else:
                    rtl_path = rtl_dir / f"attention_{R}x{C}.v"
                    v = math.ceil(ATTENTION_D_HEAD / R)
                    h = math.ceil(ATTENTION_D_HEAD / C)

                if not rtl_path.is_file():
                    print(f"  WARNING: RTL not found: {rtl_path}",
                          file=sys.stderr)
                    continue

                run_dir = r2_dir / f"{config_name}" / f"clb{pct}" / wl_name
                dpe_count = v * h if wl['type'] == 'fc' else 3 * v * h

                job = {
                    'rtl': rtl_path,
                    'arch': arch_path,
                    'run_dir': run_dir,
                    'R': R, 'C': C, 'K': K, 'N': N,
                    'wl_name': wl_name,
                    'wl_type': wl['type'],
                    'V': v, 'H': h,
                    'dpe_count': dpe_count,
                    'acam_eligible': (v == 1),
                    'ratio': ratio,
                    'pct': pct,
                }
                if args.skip_existing and not args.force and \
                   (run_dir / "imc_result.json").exists():
                    print(f"  SKIP (existing): {config_name}/clb{pct}/{wl_name}")
                    skipped_jobs.append(job)
                    continue
                jobs.append(job)

    print(f"\n  Total jobs: {len(jobs)}, skipped: {len(skipped_jobs)}")

    if args.dry_run:
        print("\n  DRY RUN — job list printed above.")
        return

    if not jobs and not skipped_jobs:
        print("  No jobs to run.")
        return

    # ── Phase 3: Run VTR (parallel) ──────────────────────────────────────
    vtr_results = {}
    if jobs:
        print("\n" + "=" * 70)
        print("Phase 3: Running VTR synthesis & P&R (fixed layout)")
        print("=" * 70)

        cpu_count = os.cpu_count() or 1
        max_workers = args.jobs if args.jobs > 0 else min(len(jobs), cpu_count)
        print(f"  Workers: {max_workers}")

        def _run_one_vtr_r2(job):
            try:
                wl = job['wl_name']
                design_name = f"attention_{job['R']}x{job['C']}" \
                    if job['wl_type'] == 'attention' \
                    else f"fc_{job['K']}_{job['N']}_{job['R']}x{job['C']}"
                r = run_single(
                    vtr_flow=VTR_FLOW,
                    vtr_python=VTR_PYTHON,
                    design=job['rtl'],
                    arch=job['arch'],
                    route_chan_width=DEFAULT_ROUTE_CHAN_WIDTH,
                    sdc_file=None,
                    run_dir=job['run_dir'],
                    seed=DEFAULT_SEED,
                    run_index=0,
                    total_runs=1,
                    design_name=design_name,
                )
                log_path = find_vpr_log(job['run_dir'])
                try:
                    grid_w, grid_h = parse_grid_size(log_path)
                except ValueError:
                    grid_w, grid_h = fixed_w, fixed_h
                return job, {
                    'fmax_mhz': r.fmax_mhz,
                    'grid_w': grid_w,
                    'grid_h': grid_h,
                    'resources': r.resources,
                    'run_dir': str(job['run_dir']),
                }, None
            except Exception as exc:
                return job, None, str(exc)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_vtr_r2, j): j for j in jobs}
            for fut in as_completed(futures):
                job, result, error = fut.result()
                key = (job['R'], job['C'], job['pct'], job['wl_name'])
                if error:
                    print(f"  FAILED: {key}: {error}", file=sys.stderr)
                else:
                    vtr_results[key] = result
                    print(f"  OK: {job['R']}x{job['C']}/clb{job['pct']}/"
                          f"{job['wl_name']}  fmax={result['fmax_mhz']:.1f} MHz")

        print(f"\n  Completed: {len(vtr_results)}/{len(jobs)}")

    # ── Phase 4: Run IMC simulator (sequential) ─────────────────────────
    all_results = []
    if jobs:
        print("\n" + "=" * 70)
        print("Phase 4: Running IMC energy/latency simulation")
        print("=" * 70)

    for job in jobs:
        key = (job['R'], job['C'], job['pct'], job['wl_name'])
        if key not in vtr_results:
            continue

        vtr = vtr_results[key]
        R, C = job['R'], job['C']
        K, N = job['K'], job['N']
        wl_name = job['wl_name']
        config_name = f"{R}x{C}"
        run_dir = Path(vtr['run_dir'])

        # Patch IMC config
        imc_cfg_path = imc_dir / f"nl_dpe_{config_name}_clb{job['pct']}_{wl_name}.json"
        patch_imc_config(BASE_IMC_CONFIG, R, C, vtr['fmax_mhz'], imc_cfg_path)

        # Run IMC
        label = f"{config_name}/clb{job['pct']}/{wl_name}"
        print(f"  IMC: {label} @ {vtr['fmax_mhz']:.1f} MHz ... ",
              end="", flush=True)

        if job['wl_type'] == 'attention':
            energy_pj, latency_ns, breakdown = run_imc_attention(
                imc_cfg_path, ATTENTION_N_SEQ, ATTENTION_D_HEAD)
        else:
            energy_pj, latency_ns, breakdown = run_imc_fc(
                imc_cfg_path, K, N)

        if energy_pj is not None and latency_ns is not None:
            print(f"energy={energy_pj:.2f} pJ  latency={latency_ns:.2f} ns")
        else:
            print("PARSE FAILED")
            continue

        # Save imc_result.json
        imc_result = {
            'config': config_name,
            'workload': wl_name,
            'wl_type': job['wl_type'],
            'ratio': job['ratio'],
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
        grid_w, grid_h = vtr['grid_w'], vtr['grid_h']
        fpga_area_mm2 = grid_w * grid_h * CLB_TILE_UM2 / 1e6
        throughput_per_s = 1e9 / latency_ns if latency_ns > 0 else 0
        throughput_per_mm2 = throughput_per_s / fpga_area_mm2 if fpga_area_mm2 > 0 else 0
        throughput_per_J = 1e12 / energy_pj if energy_pj > 0 else 0
        resources = vtr.get('resources', {})

        row = {
            'mode': 'clb_replace',
            'clb_replace_ratio': job['ratio'],
            'pair_name': '',
            'config': config_name,
            'rows': R, 'cols': C,
            'workload': wl_name,
            'wl_type': job['wl_type'],
            'K': K, 'N': N,
            'V': job['V'], 'H': job['H'],
            'dpe_count': job['dpe_count'],
            'acam_eligible': job['acam_eligible'],
            'fmax_mhz': vtr['fmax_mhz'],
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
            'e_dpe_grouped': breakdown.get('_dpe_grouped', 0),
            'e_mem_grouped': breakdown.get('_mem_grouped', 0),
            'e_fpga_grouped': breakdown.get('_fpga_grouped', 0),
            'clb_count': resources.get('clb', 0),
            'dsp_count': resources.get('dsp_top', 0),
            'mem_count': resources.get('memory', 0),
            'wc_count': resources.get('wc', 0),
            'run_dir': str(run_dir),
        }
        all_results.append(row)

    # ── Phase 4b: Reload skipped results ─────────────────────────────────
    if skipped_jobs:
        print(f"\n  Reloading {len(skipped_jobs)} pre-existing results...")
        for job in skipped_jobs:
            run_dir = Path(job['run_dir'])
            imc_path = run_dir / "imc_result.json"
            try:
                with open(imc_path) as f:
                    imc = json.load(f)
                log_path = find_vpr_log(run_dir)
                try:
                    grid_w, grid_h = parse_grid_size(log_path)
                except ValueError:
                    grid_w, grid_h = fixed_w, fixed_h
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
                    'mode': 'clb_replace',
                    'clb_replace_ratio': job['ratio'],
                    'pair_name': '',
                    'config': f"{R}x{C}",
                    'rows': R, 'cols': C,
                    'workload': job['wl_name'],
                    'wl_type': job['wl_type'],
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
                    'e_dpe_grouped': breakdown.get('_dpe_grouped', 0),
                    'e_mem_grouped': breakdown.get('_mem_grouped', 0),
                    'e_fpga_grouped': breakdown.get('_fpga_grouped', 0),
                    'clb_count': resources.get('clb', 0),
                    'dsp_count': resources.get('dsp_top', 0),
                    'mem_count': resources.get('memory', 0),
                    'wc_count': resources.get('wc', 0),
                    'run_dir': str(run_dir),
                })
            except Exception as e:
                print(f"  WARNING: could not reload {run_dir}: {e}",
                      file=sys.stderr)

    # ── Phase 5: Write CSV ───────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 70)
        print("Phase 5: Writing Round 2 results CSV")
        print("=" * 70)
        csv_path = results_dir / "round2_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ROUND2_CSV_COLUMNS,
                                     extrasaction='ignore')
            writer.writeheader()
            for row in sorted(all_results,
                              key=lambda r: (r['config'], r['clb_replace_ratio'],
                                             r['workload'])):
                writer.writerow(row)
        print(f"  Written: {csv_path}  ({len(all_results)} rows)")

    # ── Phase 6: Summary ─────────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 70)
        print("Round 2 Part 1 — CLB Replacement Summary")
        print("=" * 70)
        hdr = (f"{'Config':<10} {'Ratio':>5} {'Workload':<14} "
               f"{'Fmax':>7} {'wc':>3} {'clb':>5} "
               f"{'E pJ':>12} {'Lat ns':>12} {'Tput/mm2':>11} {'Tput/J':>11}")
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(all_results,
                         key=lambda x: (x['config'], x['clb_replace_ratio'],
                                        x['workload'])):
            print(f"{r['config']:<10} {r['clb_replace_ratio']:>5.0%} "
                  f"{r['workload']:<14} "
                  f"{r['fmax_mhz']:>7.1f} {r['wc_count']:>3} "
                  f"{r['clb_count']:>5} "
                  f"{r['energy_pj']:>12.2f} {r['latency_ns']:>12.2f} "
                  f"{r['throughput_per_mm2']:>11.0f} "
                  f"{r['throughput_per_J']:>11.0f}")

    # ── Part 2: DSP+BRAM equivalence comparison ────────────────────────
    _run_round2_dsp_bram(args, fixed_w, fixed_h, configs, workloads,
                         arch_dir, rtl_dir, r2_dir, results_dir, all_results)


# ── Round 2 Prototype — DSP+CLB Replacement Sweep ─────────────────────────

DSP_RATIOS = [0.20, 0.40, 0.60, 0.80, 1.00]
CLB_RATIOS = [0.00, 0.20, 0.40, 0.60, 0.80]
BASELINE_CLBS = 10978  # for 120x120 grid (scaled from 106x106: 8528 * (118/104)^2)
BASELINE_DSP_AREA_UM2 = 182 * 8593     # 1.564 mm²
BASELINE_CLB_AREA_UM2 = 10978 * 2239   # 24.58 mm²
MWTA_TO_UM2 = 0.033864
BRAM_TILE_HEIGHT = 2   # from arch XML: <tile name="memory" height="2">
BRAM_STARTX = 2
BRAM_REPEATX = 16

PROTO_CSV_COLUMNS = [
    'dsp_ratio', 'clb_ratio', 'config', 'workload',
    'P', 'total_dpes', 'V', 'H', 'dpes_per_replica',
    'clbs_need', 'clbs_avail', 'clb_util_pct',
    'fmax_mhz', 'fmax_peak_mhz', 'utilization',
    'grid_w', 'grid_h',
    'clb_count', 'dsp_count', 'mem_count', 'wc_count',
    'run_dir',
]


def count_available_wc(grid_w, grid_h, tile_w, tile_h,
                       dsp_ratio, clb_ratio):
    """Count actual wc (DPE) tile slots in the VTR grid layout.

    Simulates VTR's priority-based column placement to determine how many
    wc tiles can physically fit. Must match the layout logic in gen_arch_xml.py.
    """
    from gen_arch_xml import (BASELINE_DSP_STARTX, BASELINE_DSP_REPEATX,
                              BASELINE_BRAM_STARTX, BASELINE_BRAM_REPEATX,
                              _baseline_col_positions)

    interior_w = grid_w - 2
    interior_h = grid_h - 2
    tiles_per_col = interior_h // tile_h

    # Baseline DSP and BRAM positions
    dsp_positions = _baseline_col_positions(
        BASELINE_DSP_STARTX, BASELINE_DSP_REPEATX, grid_w)
    bram_positions = _baseline_col_positions(
        BASELINE_BRAM_STARTX, BASELINE_BRAM_REPEATX, grid_w)

    # DSP columns to remove (rightmost first, matching gen_arch_xml)
    n_dsp_remove = min(len(dsp_positions),
                       max(0, round(dsp_ratio * len(dsp_positions))))
    dsp_removed = sorted(dsp_positions[-n_dsp_remove:]) if n_dsp_remove > 0 else []
    dsp_kept = sorted(set(dsp_positions) - set(dsp_removed))

    # CLB replacement repeatx (matching gen_arch_xml)
    if clb_ratio > 0:
        n_clb_dpe_cols = max(1, round(clb_ratio * interior_w / tile_w))
        wc_clb_repeatx = max(tile_w + 1, interior_w // n_clb_dpe_cols)
    else:
        wc_clb_repeatx = None

    # Track occupied columns (by type at priority >= 15)
    occupied = set()

    # BRAM columns (priority 20, width 1)
    for pos in bram_positions:
        occupied.add(pos)

    # Remaining DSP columns (priority 20, width 1)
    for pos in dsp_kept:
        occupied.add(pos)

    # wc from DSP replacement (priority 20, width tile_w)
    wc_from_dsp = 0
    for pos in dsp_removed:
        can_place = all(1 <= pos + dx <= interior_w and (pos + dx) not in occupied
                        for dx in range(tile_w))
        if can_place:
            wc_from_dsp += tiles_per_col
            for dx in range(tile_w):
                occupied.add(pos + dx)

    # wc from CLB replacement (priority 15, width tile_w)
    wc_from_clb = 0
    if wc_clb_repeatx:
        x = BASELINE_DSP_STARTX  # startx matches gen_arch_xml
        while x <= interior_w:
            can_place = all(1 <= x + dx <= interior_w and (x + dx) not in occupied
                           for dx in range(tile_w))
            if can_place:
                wc_from_clb += tiles_per_col
                for dx in range(tile_w):
                    occupied.add(x + dx)
            x += wc_clb_repeatx

    return wc_from_dsp + wc_from_clb


def count_available_brams(grid_w, grid_h):
    """Count physical BRAM tiles on the VTR grid.

    BRAM columns are unaffected by DSP/CLB replacement (different positions,
    same priority 20). Count is constant for a given grid size.
    """
    interior_w = grid_w - 2
    interior_h = grid_h - 2
    n_bram_cols = len(range(BRAM_STARTX, interior_w + 1, BRAM_REPEATX))
    brams_per_col = interior_h // BRAM_TILE_HEIGHT
    return n_bram_cols * brams_per_col


def estimate_brams_per_replica(k, data_width=40):
    """Estimate physical BRAM blocks per GEMM replica.

    Each replica has SRAMs (input buffer, controller state) that VTR packs
    into physical memory blocks. Empirically validated:
      K <= 1024: 4 BRAMs/rep  (e.g. fc_512_128, fc_512_512)
      K >  1024: 6 BRAMs/rep  (e.g. fc_2048_256)
    """
    if k <= 1024:
        return 4
    else:
        return 6


def compute_feasibility(dsp_ratio, clb_ratio, rows, cols, k, n,
                        grid_w=120, grid_h=120, clbs_per_replica=25):
    """Grid-based feasibility check for a (d, c) sweep point.

    Checks three resource limits and caps P to the minimum:
      1. DPE tiles available (from grid layout)
      2. CLB capacity (remaining CLBs after replacement)
      3. BRAM capacity (constant per grid size)

    Returns dict with 'feasible', 'p', resource details, and limiting factor.
    """
    specs = dpe_specs(rows, cols)
    tile_w = specs['tile_width']
    tile_h = specs['tile_height']

    # Count actual wc tiles available
    total_dpes = count_available_wc(
        grid_w, grid_h, tile_w, tile_h, dsp_ratio, clb_ratio)

    v = math.ceil(k / rows)
    h = math.ceil(n / cols)
    dpes_per_replica = v * h
    p_dpe = total_dpes // dpes_per_replica if dpes_per_replica > 0 else 0

    # CLB limit
    clbs_avail = int((1 - clb_ratio) * BASELINE_CLBS)
    p_clb = max(0, (clbs_avail - 30) // (clbs_per_replica + 5)) if clbs_avail > 30 else 0

    # BRAM limit
    brams_avail = count_available_brams(grid_w, grid_h)
    brams_per_rep = estimate_brams_per_replica(k)
    p_bram = brams_avail // brams_per_rep if brams_per_rep > 0 else 0

    # P = min of all three limits
    p = min(p_dpe, p_clb, p_bram)

    # Determine limiting factor
    if p < 1:
        limit = 'none'
    elif p == p_bram and p_bram <= p_dpe and p_bram <= p_clb:
        limit = 'bram'
    elif p == p_clb and p_clb <= p_dpe:
        limit = 'clb'
    else:
        limit = 'dpe'

    clbs_need = p * clbs_per_replica + 5 * p + 30 if p > 0 else 30
    clb_util = clbs_need / clbs_avail if clbs_avail > 0 else float('inf')

    feasible = p >= 1

    return {
        'feasible': feasible,
        'p': p,
        'p_dpe': p_dpe,
        'p_clb': p_clb,
        'p_bram': p_bram,
        'limit': limit,
        'total_dpes': total_dpes,
        'v': v, 'h': h,
        'dpes_per_replica': dpes_per_replica,
        'clbs_need': clbs_need,
        'clbs_avail': clbs_avail,
        'clb_util': clb_util,
        'brams_avail': brams_avail,
        'brams_need': p * brams_per_rep,
        'brams_per_rep': brams_per_rep,
    }


def run_round2_prototype(args):
    """Round 2 Prototype: DSP+CLB replacement sweep on a single workload.

    Sweeps 5 DSP ratios × 5 CLB ratios = 25 points.
    Generates arch XML + P-replica RTL, runs VTR, computes throughput utilization.
    """
    dse_dir = args.dse_dir
    results_dir = dse_dir / "results"
    arch_dir = dse_dir / "configs" / "arch"
    rtl_dir = dse_dir / "rtl"
    r2_dir = dse_dir / "round2_proto"

    for d in [arch_dir, rtl_dir, r2_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load fixed grid from template
    template_path = results_dir / "round2_template.json"
    if not template_path.exists():
        print("ERROR: round2_template.json not found. Run Round 1 first.",
              file=sys.stderr)
        return
    with open(template_path) as f:
        tmpl = json.load(f)
    fixed_w = tmpl['fixed_grid_w']
    fixed_h = tmpl['fixed_grid_h']

    # Prototype config and workload
    R, C = 512, 128
    K, N = 2048, 256
    wl_name = "fc_2048_256"
    config_name = f"{R}x{C}"

    print(f"Round 2 Prototype — DSP+CLB Replacement Sweep")
    print(f"  Config: {config_name}")
    print(f"  Workload: {wl_name} (K={K}, N={N})")
    print(f"  Fixed grid: {fixed_w}x{fixed_h}")
    print(f"  DSP ratios: {DSP_RATIOS}")
    print(f"  CLB ratios: {CLB_RATIOS}")
    print()

    # ── Phase 1: Analytical feasibility ──────────────────────────────────
    print("=" * 70)
    print("Phase 1: Analytical Feasibility Check")
    print("=" * 70)

    jobs = []
    for dsp_r in DSP_RATIOS:
        for clb_r in CLB_RATIOS:
            f = compute_feasibility(dsp_r, clb_r, R, C, K, N,
                                    grid_w=fixed_w, grid_h=fixed_h)
            dsp_pct = int(round(dsp_r * 100))
            clb_pct = int(round(clb_r * 100))
            status = "OK" if f['feasible'] else "SKIP"
            print(f"  d={dsp_pct:>3}% c={clb_pct:>3}%  P={f['p']:>3}  "
                  f"(dpe={f['p_dpe']:>3} clb={f['p_clb']:>3} "
                  f"bram={f['p_bram']:>3})  "
                  f"BRAMs {f['brams_need']:>4}/{f['brams_avail']:>4}  "
                  f"[{status}]")

            if not f['feasible']:
                continue

            run_dir = r2_dir / f"d{dsp_pct}_c{clb_pct}" / wl_name
            jobs.append({
                'dsp_ratio': dsp_r, 'clb_ratio': clb_r,
                'dsp_pct': dsp_pct, 'clb_pct': clb_pct,
                'P': f['p'],
                'total_dpes': f['total_dpes'],
                'V': f['v'], 'H': f['h'],
                'dpes_per_replica': f['dpes_per_replica'],
                'clbs_need': f['clbs_need'],
                'clbs_avail': f['clbs_avail'],
                'clb_util': f['clb_util'],
                'brams_avail': f['brams_avail'],
                'brams_need': f['brams_need'],
                'limit': f['limit'],
                'run_dir': run_dir,
            })

    print(f"\n  Feasible points: {len(jobs)}/{len(DSP_RATIOS) * len(CLB_RATIOS)}")

    if args.dry_run:
        print("\n  DRY RUN — stopping here.")
        return

    # ── Phase 2: Generate arch XMLs and RTL ──────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: Generating architecture XMLs and RTL")
    print("=" * 70)

    skipped_jobs = []
    active_jobs = []
    for job in jobs:
        # Arch XML
        arch_path = gen_arch_xml(
            R, C, mode="fixed_dsp_clb_replace", output_dir=arch_dir,
            fixed_grid_w=fixed_w, fixed_grid_h=fixed_h,
            dsp_ratio=job['dsp_ratio'], clb_ratio=job['clb_ratio'],
        )
        job['arch'] = arch_path

        # RTL (P-replica wrapper)
        rtl_path = gen_gemm_wrapper(
            K, N, R, C, job['P'], output_dir=str(rtl_dir),
        )
        job['rtl'] = rtl_path

        # Skip existing?
        if args.skip_existing and not args.force and \
           (job['run_dir'] / "vtr_done.json").exists():
            print(f"  SKIP (existing): d{job['dsp_pct']}_c{job['clb_pct']}")
            skipped_jobs.append(job)
            continue
        active_jobs.append(job)

    print(f"\n  Active VTR jobs: {len(active_jobs)}, skipped: {len(skipped_jobs)}")

    # ── Phase 3: Run VTR ─────────────────────────────────────────────────
    vtr_results = {}
    if active_jobs:
        print("\n" + "=" * 70)
        print("Phase 3: Running VTR synthesis & P&R")
        print("=" * 70)

        cpu_count = os.cpu_count() or 1
        max_workers = args.jobs if args.jobs > 0 else min(len(active_jobs), cpu_count)
        print(f"  Workers: {max_workers}")

        def _run_one_proto(job):
            try:
                design_name = f"gemm_{K}_{N}_{R}x{C}_P{job['P']}"
                r = run_single(
                    vtr_flow=VTR_FLOW,
                    vtr_python=VTR_PYTHON,
                    design=job['rtl'],
                    arch=job['arch'],
                    route_chan_width=DEFAULT_ROUTE_CHAN_WIDTH,
                    sdc_file=None,
                    run_dir=job['run_dir'],
                    seed=DEFAULT_SEED,
                    run_index=0,
                    total_runs=1,
                    design_name=design_name,
                )
                log_path = find_vpr_log(job['run_dir'])
                try:
                    grid_w, grid_h = parse_grid_size(log_path)
                except ValueError:
                    grid_w, grid_h = fixed_w, fixed_h
                resources = r.resources
                return job, {
                    'fmax_mhz': r.fmax_mhz,
                    'grid_w': grid_w,
                    'grid_h': grid_h,
                    'resources': resources,
                }, None
            except Exception as exc:
                return job, None, str(exc)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_proto, j): j for j in active_jobs}
            for fut in as_completed(futures):
                job, result, error = fut.result()
                key = (job['dsp_pct'], job['clb_pct'])
                if error:
                    print(f"  FAILED d{key[0]}_c{key[1]}: {error}",
                          file=sys.stderr)
                else:
                    vtr_results[key] = result
                    # Save marker
                    done_path = job['run_dir'] / "vtr_done.json"
                    done_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(done_path, 'w') as f:
                        json.dump({
                            'fmax_mhz': result['fmax_mhz'],
                            'grid_w': result['grid_w'],
                            'grid_h': result['grid_h'],
                        }, f, indent=2)
                    print(f"  OK d{key[0]}_c{key[1]}:  "
                          f"P={job['P']}  fmax={result['fmax_mhz']:.1f} MHz  "
                          f"wc={result['resources'].get('wc', 0)}  "
                          f"clb={result['resources'].get('clb', 0)}")

        print(f"\n  Completed: {len(vtr_results)}/{len(active_jobs)}")

    # ── Phase 3b: Reload skipped results ─────────────────────────────────
    for job in skipped_jobs:
        key = (job['dsp_pct'], job['clb_pct'])
        done_path = job['run_dir'] / "vtr_done.json"
        try:
            with open(done_path) as f:
                saved = json.load(f)
            log_path = find_vpr_log(job['run_dir'])
            resources = parse_resources(log_path)
            vtr_results[key] = {
                'fmax_mhz': saved['fmax_mhz'],
                'grid_w': saved['grid_w'],
                'grid_h': saved['grid_h'],
                'resources': resources,
            }
        except Exception as e:
            print(f"  WARNING: reload failed for d{key[0]}_c{key[1]}: {e}",
                  file=sys.stderr)

    # ── Phase 4: Compute metrics ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 4: Computing throughput utilization")
    print("=" * 70)

    # Find Fmax_peak for each DSP ratio (Fmax at CLB=0%)
    fmax_peak = {}
    for dsp_r in DSP_RATIOS:
        dsp_pct = int(round(dsp_r * 100))
        key = (dsp_pct, 0)
        if key in vtr_results:
            fmax_peak[dsp_pct] = vtr_results[key]['fmax_mhz']
        else:
            print(f"  WARNING: missing baseline d={dsp_pct}% c=0% — "
                  f"cannot compute utilization for this DSP ratio")

    all_rows = []
    for job in jobs:
        key = (job['dsp_pct'], job['clb_pct'])
        if key not in vtr_results:
            continue
        vtr = vtr_results[key]
        peak = fmax_peak.get(job['dsp_pct'])
        util = vtr['fmax_mhz'] / peak if peak and peak > 0 else None

        resources = vtr.get('resources', {})
        row = {
            'dsp_ratio': job['dsp_ratio'],
            'clb_ratio': job['clb_ratio'],
            'config': config_name,
            'workload': wl_name,
            'P': job['P'],
            'total_dpes': job['total_dpes'],
            'V': job['V'], 'H': job['H'],
            'dpes_per_replica': job['dpes_per_replica'],
            'clbs_need': job['clbs_need'],
            'clbs_avail': job['clbs_avail'],
            'clb_util_pct': round(job['clb_util'] * 100, 1),
            'fmax_mhz': vtr['fmax_mhz'],
            'fmax_peak_mhz': peak if peak else 0,
            'utilization': round(util, 4) if util else 0,
            'grid_w': vtr['grid_w'],
            'grid_h': vtr['grid_h'],
            'clb_count': resources.get('clb', 0),
            'dsp_count': resources.get('dsp_top', 0),
            'mem_count': resources.get('memory', 0),
            'wc_count': resources.get('wc', 0),
            'run_dir': str(job['run_dir']),
        }
        all_rows.append(row)

        util_str = f"{util:.3f}" if util else "N/A"
        print(f"  d={job['dsp_pct']:>3}% c={job['clb_pct']:>3}%  "
              f"P={job['P']:>3}  Fmax={vtr['fmax_mhz']:>7.1f}  "
              f"peak={peak if peak else 0:>7.1f}  util={util_str}")

    # ── Phase 5: Write CSV ───────────────────────────────────────────────
    if all_rows:
        csv_path = results_dir / "round2_proto_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=PROTO_CSV_COLUMNS)
            writer.writeheader()
            for row in sorted(all_rows,
                              key=lambda r: (r['dsp_ratio'], r['clb_ratio'])):
                writer.writerow(row)
        print(f"\n  Written: {csv_path}  ({len(all_rows)} rows)")

    print("\nDone. Run plot_throughput_utilization.py to generate the figure.")


def run_round2_full(args):
    """Round 2 Full Sweep: DSP+CLB replacement across top-3 configs and FC workloads.

    Sweeps 3 configs × 3 workloads × 20 (d,c) points = 180 VTR runs.
    """
    dse_dir = args.dse_dir
    results_dir = dse_dir / "results"
    arch_dir = dse_dir / "configs" / "arch"
    rtl_dir = dse_dir / "rtl"
    r2_dir = dse_dir / "round2_full"

    for d in [arch_dir, rtl_dir, r2_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load fixed grid from template
    template_path = results_dir / "round2_template.json"
    if not template_path.exists():
        print("ERROR: round2_template.json not found. Run Round 1 first.",
              file=sys.stderr)
        return
    with open(template_path) as f:
        tmpl = json.load(f)
    fixed_w = tmpl['fixed_grid_w']
    fixed_h = tmpl['fixed_grid_h']

    # Top-3 from Round 1 + area-equivalent to Azure-Lily
    TOP3_CONFIGS = [(512, 128), (1024, 128), (1024, 64), (1024, 256), (512, 256)]
    if args.configs:
        configs = []
        for s in args.configs.split(","):
            r, c = s.strip().split("x")
            configs.append((int(r), int(c)))
    else:
        configs = TOP3_CONFIGS

    # Default workloads: medium, large, super-large
    R2_DEFAULT_WORKLOADS = {
        "fc_512_128":  (512,  128),
        "fc_512_512":  (512,  512),
        "fc_2048_256": (2048, 256),
    }
    if args.workloads:
        workload_names = [w.strip() for w in args.workloads.split(",")]
        workloads = {k: v for k, v in FC_WORKLOADS.items() if k in workload_names}
    else:
        workloads = R2_DEFAULT_WORKLOADS

    # Drop c=80% (identical to c=60% due to column saturation)
    clb_ratios = [c for c in CLB_RATIOS if c <= 0.6]

    print(f"Round 2 Full Sweep — DSP+CLB Replacement, Multi-Config")
    print(f"  Configs: {[f'{R}x{C}' for R, C in configs]}")
    print(f"  Workloads: {list(workloads.keys())}")
    print(f"  Fixed grid: {fixed_w}x{fixed_h}")
    print(f"  DSP ratios: {DSP_RATIOS}")
    print(f"  CLB ratios: {clb_ratios}")
    print(f"  Points per (config, workload): {len(DSP_RATIOS) * len(clb_ratios)}")
    print(f"  Total sweep: {len(configs)} configs × {len(workloads)} workloads × "
          f"{len(DSP_RATIOS) * len(clb_ratios)} points = "
          f"{len(configs) * len(workloads) * len(DSP_RATIOS) * len(clb_ratios)} runs")
    print()

    # ── Phase 1: Feasibility for all configs × workloads ──────────────────
    print("=" * 70)
    print("Phase 1: Analytical Feasibility Check")
    print("=" * 70)

    all_jobs = []
    for R, C in configs:
        config_name = f"{R}x{C}"
        print(f"\n{'─' * 60}")
        print(f"  Config: {config_name}")
        print(f"{'─' * 60}")
        for wl_name, (K, N) in workloads.items():
            print(f"\n  --- {config_name} / {wl_name} (K={K}, N={N}) ---")
            wl_jobs = 0
            for dsp_r in DSP_RATIOS:
                for clb_r in clb_ratios:
                    f = compute_feasibility(dsp_r, clb_r, R, C, K, N,
                                            grid_w=fixed_w, grid_h=fixed_h)
                    dsp_pct = int(round(dsp_r * 100))
                    clb_pct = int(round(clb_r * 100))
                    status = "OK" if f['feasible'] else "SKIP"
                    limit_tag = f"[{f['limit']}]" if f['feasible'] else ""
                    print(f"    d={dsp_pct:>3}% c={clb_pct:>3}%  P={f['p']:>3}  "
                          f"(dpe={f['p_dpe']:>3} clb={f['p_clb']:>3} "
                          f"bram={f['p_bram']:>3})  "
                          f"CLBs {f['clbs_need']:>5}/{f['clbs_avail']:>5}  "
                          f"BRAMs {f['brams_need']:>4}/{f['brams_avail']:>4}  "
                          f"[{status}] {limit_tag}")

                    if not f['feasible']:
                        continue

                    run_dir = r2_dir / config_name / f"d{dsp_pct}_c{clb_pct}" / wl_name
                    all_jobs.append({
                        'dsp_ratio': dsp_r, 'clb_ratio': clb_r,
                        'dsp_pct': dsp_pct, 'clb_pct': clb_pct,
                        'R': R, 'C': C, 'config': config_name,
                        'wl_name': wl_name, 'K': K, 'N': N,
                        'P': f['p'],
                        'total_dpes': f['total_dpes'],
                        'V': f['v'], 'H': f['h'],
                        'dpes_per_replica': f['dpes_per_replica'],
                        'clbs_need': f['clbs_need'],
                        'clbs_avail': f['clbs_avail'],
                        'clb_util': f['clb_util'],
                        'brams_avail': f['brams_avail'],
                        'brams_need': f['brams_need'],
                        'limit': f['limit'],
                        'run_dir': run_dir,
                    })
                    wl_jobs += 1
            print(f"    Feasible: {wl_jobs}/{len(DSP_RATIOS) * len(clb_ratios)}")

    total_points = len(DSP_RATIOS) * len(clb_ratios) * len(workloads) * len(configs)
    print(f"\n  Total feasible: {len(all_jobs)}/{total_points}")

    if args.dry_run:
        print("\n  DRY RUN — stopping here.")
        return

    # ── Phase 2: Generate arch XMLs and RTL ───────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: Generating architecture XMLs and RTL")
    print("=" * 70)

    # Arch XMLs are shared across workloads (same grid layout per (config, d, c))
    arch_cache = {}
    skipped_jobs = []
    active_jobs = []

    for job in all_jobs:
        # Arch XML (cached per (config, d, c))
        dc_key = (job['R'], job['C'], job['dsp_pct'], job['clb_pct'])
        if dc_key not in arch_cache:
            arch_path = gen_arch_xml(
                job['R'], job['C'], mode="fixed_dsp_clb_replace",
                output_dir=arch_dir,
                fixed_grid_w=fixed_w, fixed_grid_h=fixed_h,
                dsp_ratio=job['dsp_ratio'], clb_ratio=job['clb_ratio'],
            )
            arch_cache[dc_key] = arch_path
        job['arch'] = arch_cache[dc_key]

        # RTL (P-replica wrapper, unique per config × workload × P)
        rtl_path = gen_gemm_wrapper(
            job['K'], job['N'], job['R'], job['C'], job['P'],
            output_dir=str(rtl_dir),
        )
        job['rtl'] = rtl_path

        # Skip existing?
        if args.skip_existing and not args.force and \
           (job['run_dir'] / "vtr_done.json").exists():
            print(f"  SKIP: d{job['dsp_pct']}_c{job['clb_pct']}/{job['wl_name']}")
            skipped_jobs.append(job)
            continue
        active_jobs.append(job)

    print(f"\n  Active VTR jobs: {len(active_jobs)}, skipped: {len(skipped_jobs)}")

    # ── Phase 3: Run VTR ──────────────────────────────────────────────────
    vtr_results = {}
    if active_jobs:
        print("\n" + "=" * 70)
        print(f"Phase 3: Running VTR ({len(active_jobs)} jobs)")
        print("=" * 70)

        cpu_count = os.cpu_count() or 1
        max_workers = args.jobs if args.jobs > 0 else min(len(active_jobs), cpu_count)
        print(f"  Workers: {max_workers}")

        def _run_one_full(job):
            """Run VTR with multiple seeds, return averaged Fmax."""
            try:
                design_name = (f"gemm_{job['K']}_{job['N']}_"
                               f"{job['R']}x{job['C']}_P{job['P']}")
                seed_fmax = []
                last_resources = None
                last_grid_w, last_grid_h = fixed_w, fixed_h

                for seed in MULTI_SEEDS:
                    seed_dir = job['run_dir'] / f"seed{seed}"
                    r = run_single(
                        vtr_flow=VTR_FLOW,
                        vtr_python=VTR_PYTHON,
                        design=job['rtl'],
                        arch=job['arch'],
                        route_chan_width=DEFAULT_ROUTE_CHAN_WIDTH,
                        sdc_file=None,
                        run_dir=seed_dir,
                        seed=seed,
                        run_index=0,
                        total_runs=1,
                        design_name=design_name,
                    )
                    seed_fmax.append(r.fmax_mhz)
                    last_resources = r.resources
                    log_path = find_vpr_log(seed_dir)
                    try:
                        last_grid_w, last_grid_h = parse_grid_size(log_path)
                    except ValueError:
                        pass

                avg_fmax = sum(seed_fmax) / len(seed_fmax)
                return job, {
                    'fmax_mhz': avg_fmax,
                    'fmax_per_seed': seed_fmax,
                    'grid_w': last_grid_w,
                    'grid_h': last_grid_h,
                    'resources': last_resources,
                }, None
            except Exception as exc:
                return job, None, str(exc)

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_full, j): j for j in active_jobs}
            for fut in as_completed(futures):
                job, result, error = fut.result()
                key = (job['config'], job['dsp_pct'], job['clb_pct'], job['wl_name'])
                if error:
                    print(f"  FAILED {job['config']}/{job['wl_name']} "
                          f"d{job['dsp_pct']}_c{job['clb_pct']}: "
                          f"{error}", file=sys.stderr)
                else:
                    vtr_results[key] = result
                    done_path = job['run_dir'] / "vtr_done.json"
                    done_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(done_path, 'w') as f:
                        json.dump({
                            'fmax_mhz': result['fmax_mhz'],
                            'fmax_per_seed': result.get('fmax_per_seed', []),
                            'grid_w': result['grid_w'],
                            'grid_h': result['grid_h'],
                        }, f, indent=2)
                    completed += 1
                    seeds_str = ", ".join(f"{f:.1f}" for f in result.get('fmax_per_seed', []))
                    print(f"  [{completed}/{len(active_jobs)}] "
                          f"{job['config']}/{job['wl_name']} "
                          f"d{job['dsp_pct']}_c{job['clb_pct']}:  "
                          f"P={job['P']}  fmax={result['fmax_mhz']:.1f} MHz "
                          f"[{seeds_str}]  "
                          f"wc={result['resources'].get('wc', 0)}  "
                          f"clb={result['resources'].get('clb', 0)}")

        print(f"\n  Completed: {len(vtr_results)}/{len(active_jobs)}")

    # ── Phase 3b: Reload skipped results ──────────────────────────────────
    for job in skipped_jobs:
        key = (job['config'], job['dsp_pct'], job['clb_pct'], job['wl_name'])
        done_path = job['run_dir'] / "vtr_done.json"
        try:
            with open(done_path) as f:
                saved = json.load(f)
            # Try to load resources from a seed subdirectory first, then fall back
            resources = {}
            for seed in MULTI_SEEDS:
                seed_dir = job['run_dir'] / f"seed{seed}"
                try:
                    log_path = find_vpr_log(seed_dir)
                    resources = parse_resources(log_path)
                    break
                except Exception:
                    continue
            if not resources:
                # Fall back to legacy single-seed layout
                try:
                    log_path = find_vpr_log(job['run_dir'])
                    resources = parse_resources(log_path)
                except Exception:
                    pass
            vtr_results[key] = {
                'fmax_mhz': saved['fmax_mhz'],
                'fmax_per_seed': saved.get('fmax_per_seed', []),
                'grid_w': saved['grid_w'],
                'grid_h': saved['grid_h'],
                'resources': resources,
            }
        except Exception as e:
            print(f"  WARNING: reload failed for {job['config']}/{job['wl_name']} "
                  f"d{job['dsp_pct']}_c{job['clb_pct']}: {e}",
                  file=sys.stderr)

    # ── Phase 4: IMC energy per inference ─────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 4: Computing IMC energy per inference")
    print("=" * 70)

    imc_dir = dse_dir / "configs" / "imc"
    imc_dir.mkdir(parents=True, exist_ok=True)

    imc_results = {}  # key -> (energy_pj, latency_ns, breakdown)
    for job in all_jobs:
        key = (job['config'], job['dsp_pct'], job['clb_pct'], job['wl_name'])
        if key not in vtr_results:
            continue
        vtr = vtr_results[key]
        R, C = job['R'], job['C']
        fn = vtr['fmax_mhz']

        # Check cached imc_result.json in run_dir
        imc_cache_path = job['run_dir'] / "imc_result.json"
        if imc_cache_path.exists():
            try:
                with open(imc_cache_path) as f:
                    cached = json.load(f)
                imc_results[key] = (cached['energy_pj'], cached['latency_ns'],
                                    cached.get('breakdown', {}))
                continue
            except Exception:
                pass

        # Patch IMC config and run
        imc_cfg_name = f"nl_dpe_{R}x{C}_{job['wl_name']}_d{job['dsp_pct']}_c{job['clb_pct']}.json"
        imc_cfg_path = imc_dir / imc_cfg_name
        patch_imc_config(BASE_IMC_CONFIG, R, C, fn, imc_cfg_path)

        energy_pj, latency_ns, breakdown = run_imc_fc(imc_cfg_path, job['K'], job['N'])
        if energy_pj is not None and latency_ns is not None:
            imc_results[key] = (energy_pj, latency_ns, breakdown)
            # Cache
            imc_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(imc_cache_path, 'w') as f:
                json.dump({
                    'energy_pj': energy_pj, 'latency_ns': latency_ns,
                    'breakdown': breakdown, 'fmax_mhz': fn,
                }, f, indent=2)
            print(f"  {job['config']}/{job['wl_name']} d{job['dsp_pct']}_c{job['clb_pct']}: "
                  f"E={energy_pj:.1f} pJ  lat={latency_ns:.1f} ns")
        else:
            print(f"  WARNING: IMC failed for {job['config']}/{job['wl_name']} "
                  f"d{job['dsp_pct']}_c{job['clb_pct']}")

    print(f"  IMC results: {len(imc_results)} points")

    # ── Phase 5: Compute metrics ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 5: Computing throughput scalability + energy efficiency")
    print("=" * 70)

    # Baseline f0 and E0 per (config, workload): smallest (d, c) point available
    baseline_fmax = {}
    baseline_energy = {}
    for R, C in configs:
        cfg = f"{R}x{C}"
        for wl_name in workloads:
            key = (cfg, 20, 0, wl_name)
            if key in vtr_results and key in imc_results:
                baseline_fmax[(cfg, wl_name)] = vtr_results[key]['fmax_mhz']
                baseline_energy[(cfg, wl_name)] = imc_results[key][0]
                print(f"  Baseline {cfg}/{wl_name}: f0 = {baseline_fmax[(cfg, wl_name)]:.1f} MHz, "
                      f"E0 = {baseline_energy[(cfg, wl_name)]:.1f} pJ (d=20% c=0%)")
            else:
                # Find smallest-P point for this (config, workload)
                best_key = None
                best_p = float('inf')
                for job in all_jobs:
                    jk = (job['config'], job['dsp_pct'], job['clb_pct'], job['wl_name'])
                    if jk[0] == cfg and jk[3] == wl_name and \
                       jk in vtr_results and jk in imc_results:
                        if job['P'] < best_p:
                            best_p = job['P']
                            best_key = jk
                if best_key:
                    baseline_fmax[(cfg, wl_name)] = vtr_results[best_key]['fmax_mhz']
                    baseline_energy[(cfg, wl_name)] = imc_results[best_key][0]
                    print(f"  Baseline {cfg}/{wl_name}: f0 = "
                          f"{baseline_fmax[(cfg, wl_name)]:.1f} MHz, "
                          f"E0 = {baseline_energy[(cfg, wl_name)]:.1f} pJ "
                          f"(d={best_key[1]}% c={best_key[2]}%, P={best_p})")
                else:
                    print(f"  WARNING: no baseline for {cfg}/{wl_name}")

    all_rows = []
    for job in all_jobs:
        key = (job['config'], job['dsp_pct'], job['clb_pct'], job['wl_name'])
        if key not in vtr_results:
            continue
        vtr = vtr_results[key]
        f0 = baseline_fmax.get((job['config'], job['wl_name']))
        e0 = baseline_energy.get((job['config'], job['wl_name']))
        fn = vtr['fmax_mhz']
        n = job['P']
        gain = n * fn / f0 if f0 and f0 > 0 else None
        util = fn / f0 if f0 and f0 > 0 else None

        # Energy metrics
        en = imc_results[key][0] if key in imc_results else None
        # energy_ratio = E0/En (>1 means per-inference energy improved)
        energy_ratio = e0 / en if e0 and en and en > 0 else None
        # tput_per_J scaling = (fn/En) / (f0/E0)
        tpj_scaling = (fn / en) / (f0 / e0) if (f0 and e0 and fn and en
                                                  and e0 > 0 and en > 0
                                                  and f0 > 0) else None

        resources = vtr.get('resources', {})
        row = {
            'dsp_ratio': job['dsp_ratio'],
            'clb_ratio': job['clb_ratio'],
            'config': job['config'],
            'workload': job['wl_name'],
            'P': n,
            'total_dpes': job['total_dpes'],
            'V': job['V'], 'H': job['H'],
            'dpes_per_replica': job['dpes_per_replica'],
            'clbs_need': job['clbs_need'],
            'clbs_avail': job['clbs_avail'],
            'clb_util_pct': round(job['clb_util'] * 100, 1),
            'fmax_mhz': fn,
            'fmax_baseline_mhz': f0 if f0 else 0,
            'gain': round(gain, 4) if gain else 0,
            'utilization': round(util, 4) if util else 0,
            'energy_pj': round(en, 2) if en else 0,
            'energy_baseline_pj': round(e0, 2) if e0 else 0,
            'energy_ratio': round(energy_ratio, 4) if energy_ratio else 0,
            'tpj_scaling': round(tpj_scaling, 4) if tpj_scaling else 0,
            'grid_w': vtr['grid_w'],
            'grid_h': vtr['grid_h'],
            'clb_count': resources.get('clb', 0),
            'dsp_count': resources.get('dsp_top', 0),
            'mem_count': resources.get('memory', 0),
            'wc_count': resources.get('wc', 0),
            'run_dir': str(job['run_dir']),
        }
        all_rows.append(row)

    # ── Phase 6: Write CSV ────────────────────────────────────────────────
    if all_rows:
        csv_columns = [
            'dsp_ratio', 'clb_ratio', 'config', 'workload',
            'P', 'total_dpes', 'V', 'H', 'dpes_per_replica',
            'clbs_need', 'clbs_avail', 'clb_util_pct',
            'fmax_mhz', 'fmax_baseline_mhz', 'gain', 'utilization',
            'energy_pj', 'energy_baseline_pj', 'energy_ratio', 'tpj_scaling',
            'grid_w', 'grid_h',
            'clb_count', 'dsp_count', 'mem_count', 'wc_count', 'run_dir',
        ]
        csv_path = results_dir / "round2_full_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for row in sorted(all_rows,
                              key=lambda r: (r['config'], r['workload'],
                                             r['dsp_ratio'], r['clb_ratio'])):
                writer.writerow(row)
        print(f"\n  Written: {csv_path}  ({len(all_rows)} rows)")

        # Per-config/workload summary
        print("\n  Per-config/workload summary:")
        for R, C in configs:
            cfg = f"{R}x{C}"
            for wl_name in workloads:
                wl_rows = [r for r in all_rows
                           if r['config'] == cfg and r['workload'] == wl_name]
                if wl_rows:
                    best = max(wl_rows, key=lambda r: r['gain'])
                    pos_utils = [r['utilization'] for r in wl_rows
                                 if r['utilization'] > 0]
                    worst_util = min(pos_utils) if pos_utils else 0
                    print(f"    {cfg}/{wl_name}: {len(wl_rows)} points, "
                          f"best gain={best['gain']:.1f}x (P={best['P']}), "
                          f"worst util={worst_util:.1%}")

    print("\nDone. Results in round2_full_results.csv.")


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="GEMV/FC crossbar-size DSE orchestrator"
    )
    parser.add_argument(
        "--round", type=int, choices=[1, 2], default=None,
        help="FC DSE round (1=auto-layout sweep, 2=fixed-layout comparison)",
    )
    parser.add_argument(
        "--attention", action="store_true",
        help="Run attention head DSE (9 configs x 1 workload, N=128 d=128)",
    )
    parser.add_argument(
        "--round2-proto", action="store_true",
        help="Run Round 2 prototype: DSP+CLB sweep on fc_2048_256 (512x128)",
    )
    parser.add_argument(
        "--round2-full", action="store_true",
        help="Run Round 2 full sweep: DSP+CLB sweep on all FC workloads (512x128)",
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
        "--force", action="store_true", default=False,
        help="Force re-run even if results exist (overrides --skip-existing)",
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

    if args.attention:
        print(f"Attention Head DSE (N={ATTENTION_N_SEQ}, d={ATTENTION_D_HEAD})")
    elif args.round2_proto:
        print(f"Round 2 Prototype — DSP+CLB Replacement Sweep")
    elif args.round2_full:
        print(f"Round 2 Full Sweep — All Workloads")
    elif args.round:
        print(f"GEMV/FC DSE — Round {args.round}")
    else:
        print("ERROR: specify --round, --attention, --round2-proto, or --round2-full")
        sys.exit(1)
    print(f"  DSE dir: {args.dse_dir}")
    print()

    if args.attention:
        run_attention_dse(args)
    elif args.round2_proto:
        run_round2_prototype(args)
    elif args.round2_full:
        run_round2_full(args)
    elif args.round == 1:
        run_round1(args)
    elif args.round == 2:
        run_round2(args)


if __name__ == "__main__":
    main()
