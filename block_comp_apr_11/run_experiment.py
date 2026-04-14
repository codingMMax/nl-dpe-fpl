#!/usr/bin/env python3
"""
DPE Block Comparison — Experiment Runner

Runs VTR (6 setups × 2 workloads × 3 seeds = 36 runs), then IMC simulator
(12 runs) to produce a results CSV and pairwise comparison table.

Usage:
    python block_comp_apr_11/run_experiment.py                # full run
    python block_comp_apr_11/run_experiment.py --skip-vtr     # IMC only (reuse VTR results)
    python block_comp_apr_11/run_experiment.py --jobs 6       # limit parallelism
    python block_comp_apr_11/run_experiment.py --skip-existing # resume interrupted VTR
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLOCK_COMP_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT / "nl_dpe"))
from run_vtr import run_single, find_vpr_log, parse_metrics, parse_resources
from area_power import dpe_specs

# ── VTR paths ─────────────────────────────────────────────────────────────
VTR_ROOT = Path(os.environ.get(
    "VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"))
VTR_FLOW = VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py"
VTR_PYTHON = VTR_ROOT / ".venv" / "bin" / "python"
if not VTR_PYTHON.is_file():
    VTR_PYTHON = None

ROUTE_CHAN_WIDTH = 300

# ── IMC paths ─────────────────────────────────────────────────────────────
AZURELILY_ROOT = PROJECT_ROOT / "azurelily"
IMC_TEST = AZURELILY_ROOT / "IMC" / "test.py"
IMC_CONFIGS_DIR = AZURELILY_ROOT / "IMC" / "configs"

# ── Regex for IMC output parsing ──────────────────────────────────────────
RE_ENERGY_LAYER = re.compile(r"Energy total \(by layer\):\s+([\d.]+)\s+pJ")
RE_LAT_CRIT = re.compile(r"Latency total \(critical path\):\s+([\d.]+)\s+ns")
RE_ENERGY_GROUPED = re.compile(
    r"Energy grouped:\s+DPE=([\d.]+)\s+pJ,\s+Memory=([\d.]+)\s+pJ,\s+FPGA=([\d.-]+)\s+pJ")
RE_BREAKDOWN_LINE = re.compile(r"^\s+(\S+):\s+([\d.]+)\s+pJ", re.MULTILINE)
GRID_SIZE_RE = re.compile(r"FPGA sized to (\d+) x (\d+)")

# ── Experiment configuration ──────────────────────────────────────────────
SETUPS = [
    {"name": "setup0", "rows": 512,  "cols": 128, "conv": "adc",  "dpe_bw": 16,
     "base_cfg": "azure_lily.json"},
    {"name": "setup1", "rows": 512,  "cols": 128, "conv": "adc",  "dpe_bw": 40,
     "base_cfg": "azure_lily.json"},
    {"name": "setup2", "rows": 512,  "cols": 128, "conv": "acam", "dpe_bw": 40,
     "base_cfg": "nl_dpe.json"},
    {"name": "setup3", "rows": 1024, "cols": 128, "conv": "adc",  "dpe_bw": 16,
     "base_cfg": "azure_lily.json"},
    {"name": "setup4", "rows": 1024, "cols": 128, "conv": "adc",  "dpe_bw": 40,
     "base_cfg": "azure_lily.json"},
    {"name": "setup5", "rows": 1024, "cols": 128, "conv": "acam", "dpe_bw": 40,
     "base_cfg": "nl_dpe.json"},
]

WORKLOADS = [
    ("fc_512_128", 512, 128),
    ("fc_2048_256", 2048, 256),
]

SEEDS = [1, 2, 3]

# ── Arch XML mapping ─────────────────────────────────────────────────────
ARCH_XMLS = {
    512:  PROJECT_ROOT / "dse" / "configs" / "arch" / "nl_dpe_512x128_auto.xml",
    1024: PROJECT_ROOT / "dse" / "configs" / "arch" / "nl_dpe_1024x128_auto.xml",
}


def parse_grid_size(vpr_log_path: Path):
    content = vpr_log_path.read_text(errors="replace")
    match = GRID_SIZE_RE.search(content)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Could not parse grid size from {vpr_log_path}")


def get_rtl_path(setup, wl_name, K, N):
    """Locate the pre-generated RTL file for a setup × workload."""
    R, C = setup["rows"], setup["cols"]
    conv, bw = setup["conv"], setup["dpe_bw"]
    fname = f"fc_{K}_{N}_{R}x{C}_{conv}_dw{bw}.v"
    path = BLOCK_COMP_DIR / "rtl" / setup["name"] / fname
    if not path.is_file():
        raise FileNotFoundError(f"RTL not found: {path}")
    return path


def patch_imc_config(base_cfg: Path, rows, cols, fmax_mhz, dpe_bw,
                     conversion, output_path):
    """Patch IMC config with geometry, energy, freq, and dpe_buf_width.

    Energy model depends on conversion type:
      - ACAM (NL-DPE): use dpe_specs() which computes crossbar+ACAM power
        scaled by (rows, cols, freq). e_conv_pj = 0.
      - ADC (Azure-Lily): use paper-calibrated values scaled for crossbar size.
        Azure-Lily paper: 20 mW total at 1.2 GHz for 512×128.
          crossbar+DAC+other: 5.6 mW → scales linearly with rows
          ADC: 14.4 mW for 8 ADCs (cols_per_adc=16) → e_conv_pj per col per
            bit-slice = 2.33 pJ (calibrated ground truth, independent of rows)
        When rows change (e.g. 1024), crossbar power doubles but ADC stays same.
    """
    with open(base_cfg) as f:
        cfg = json.load(f)

    cfg['geometry']['array_rows'] = rows
    cfg['geometry']['array_cols'] = cols
    cfg['fpga_specs']['freq'] = fmax_mhz
    cfg['fpga_specs']['dpe_buf_width'] = dpe_bw

    if conversion == "acam":
        # NL-DPE model: dpe_specs computes crossbar + ACAM energy
        # scale_with_geometry=false (NL-DPE default from nl_dpe.json)
        #   energy_per_vmm = k_slicing * e_analogue_pj  (total, not per-col)
        #   energy_per_digital = k_accum * e_digital_pj * cols
        specs = dpe_specs(rows, cols, freq_ghz=fmax_mhz / 1000.0)
        cfg['params']['e_analoge_pj'] = specs['e_analogue_pj']
        cfg['params']['e_digital_pj'] = specs['e_digital_pj']
        cfg['params']['e_conv_pj'] = 0.0
    else:
        # ADC model: Azure-Lily paper-calibrated values, scaled for crossbar size.
        # scale_with_geometry=true (Azure-Lily default from azure_lily.json)
        #   e_analoge_pj, e_conv_pj, e_digital_pj are all per-column values;
        #   config.py multiplies them by cols at load time.
        #
        # Reference: Azure-Lily paper, 512×128 @ 1.2 GHz, total 20 mW:
        #   crossbar+DAC+other: 5.6 mW → per-pass = 5.6/1.2 * 8 = 37.33 pJ
        #   ADC: 14.4 mW for 8 ADCs (cols_per_adc=16)
        #     → per-col per-bit-slice = e_conv_pj = 2.33 pJ (calibrated)
        #
        # Scaling rules:
        #   crossbar power ∝ rows × cols (more cells = more current)
        #   ADC count = cols / cols_per_adc → independent of rows
        #   e_digital_pj = 0 (no ACAM in ADC architecture)
        freq_ghz = fmax_mhz / 1000.0
        ref_rows, ref_cols = 512, 128
        ref_freq_ghz = 1.2

        # Crossbar energy per-col per-bit-slice (JSON value, before scaling)
        # Config loader does: e_analoge_pj *= cols, then energy = k * e_analoge_pj
        # So JSON value = power_per_col / freq / k_slicing
        # p_crossbar_per_col = 5.6 mW * (rows/512) / 128 cols
        p_crossbar_per_col_mw = 5.6 * (rows / ref_rows) / ref_cols
        e_analoge_per_col = p_crossbar_per_col_mw / freq_ghz  # pJ per bit-slice
        cfg['params']['e_analoge_pj'] = e_analoge_per_col

        # ADC energy: independent of rows, calibrated at 2.33 pJ/col/bit-slice
        cfg['params']['e_conv_pj'] = 2.33

        # No ACAM in ADC architecture
        cfg['params']['e_digital_pj'] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    return output_path


def run_imc_fc(config_path: Path, K: int, N: int):
    """Run IMC simulator for FC workload, return (energy_pj, latency_ns, breakdown)."""
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

    m = RE_ENERGY_LAYER.search(out)
    energy_pj = float(m.group(1)) if m else None

    m = RE_LAT_CRIT.search(out)
    latency_ns = float(m.group(1)) if m else None

    breakdown = {}
    for m in RE_BREAKDOWN_LINE.finditer(out):
        breakdown[m.group(1)] = float(m.group(2))

    m = RE_ENERGY_GROUPED.search(out)
    if m:
        breakdown['_dpe_grouped'] = float(m.group(1))
        breakdown['_mem_grouped'] = float(m.group(2))
        breakdown['_fpga_grouped'] = float(m.group(3))

    if result.returncode != 0 and energy_pj is None:
        print(f"  [IMC] WARNING: exit {result.returncode}\n{out[-500:]}")

    return energy_pj, latency_ns, breakdown


def get_io_latency_breakdown(patched_cfg_path, rows, cols, fmax_mhz, dpe_bw, K, N):
    """Get block-level IO latency breakdown from the IMC simulator directly.

    Uses gemm_pipeline_profile(K, N, kernel_size=1, in_channels=K) to match
    the scheduler's _run_linear() call path.

    The pipeline has 4 stages:
      read:      BRAM → DPE via _streaming_read_latency_row (uses bram_width=40)
      core_row:  8 bit-slices + ADC/ACAM + output serialize (uses dpe_buf_width)
      reduction: CLB adder tree (V>1 only)
      write:     DPE → BRAM via dpe_buf_width
    """
    sys.path.insert(0, str(AZURELILY_ROOT / "IMC"))
    from imc_core.config import Config as IMCConfig
    from imc_core.imc_core import IMCCore as IMCCoreClass
    from peripherals.memory import MemoryModel
    from scheduler_stats.stats import Stats

    cfg = IMCConfig(str(patched_cfg_path))
    stats = Stats()
    mem = MemoryModel(cfg, stats)
    core = IMCCoreClass(cfg, mem, stats)

    # Match the scheduler's call: kernel_size=1, in_channels=K for FC layers
    profile = core.gemm_pipeline_profile(K, N, kernel_size=1, in_channels=K)

    # Output serialization (portion of core_row that depends on dpe_buf_width)
    dpe_bw_val = getattr(cfg, 'dpe_buf_width', cfg.bram_width)
    t_clk = 1e3 / fmax_mhz
    n_access_out = math.ceil(cols * 8 / dpe_bw_val)
    output_serialize_ns = n_access_out * t_clk

    # Identify bottleneck
    stages = {
        "read": profile["read_row"],
        "core_row": profile["core_row"],
        "reduction": profile["reduction_row"],
        "write": profile["write_row"],
    }
    bottleneck = max(stages, key=stages.get)

    return {
        "read_ns": round(profile["read_row"], 2),
        "core_row_ns": round(profile["core_row"], 2),
        "output_serialize_ns": round(output_serialize_ns, 2),
        "reduction_ns": round(profile["reduction_row"], 2),
        "write_ns": round(profile["write_row"], 2),
        "first_output_ns": round(profile["first_output_ns"], 2),
        "steady_ns": round(profile["steady"], 2),
        "io_bottleneck": bottleneck,
    }


# ── VTR job runner ────────────────────────────────────────────────────────
def run_one_vtr(setup, wl_name, K, N, seed, skip_existing=False):
    """Run a single VTR job. Returns dict with results or None on skip."""
    rtl = get_rtl_path(setup, wl_name, K, N)
    arch = ARCH_XMLS[setup["rows"]]
    run_dir = BLOCK_COMP_DIR / "vtr_runs" / setup["name"] / wl_name / f"seed{seed}"

    done_flag = run_dir / "vtr_done.flag"
    if skip_existing and done_flag.is_file():
        # Parse existing results
        try:
            log_path = find_vpr_log(run_dir)
            wirelength, fmax_mhz = parse_metrics(log_path)
            resources = parse_resources(log_path)
            grid_w, grid_h = parse_grid_size(log_path)
            return {
                "setup": setup["name"], "workload": wl_name, "seed": seed,
                "fmax_mhz": fmax_mhz, "resources": resources,
                "grid_w": grid_w, "grid_h": grid_h,
                "wirelength": wirelength, "skipped": True,
            }
        except Exception as e:
            print(f"  [VTR] Could not parse existing {run_dir}: {e}, re-running")

    run_dir.mkdir(parents=True, exist_ok=True)

    design_name = rtl.stem
    R, C = setup["rows"], setup["cols"]

    result = run_single(
        vtr_flow=VTR_FLOW,
        vtr_python=VTR_PYTHON,
        design=rtl,
        arch=arch,
        route_chan_width=ROUTE_CHAN_WIDTH,
        sdc_file=None,
        run_dir=run_dir,
        seed=seed,
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

    # Write done flag
    done_flag.write_text(f"fmax={result.fmax_mhz:.2f}\n")

    return {
        "setup": setup["name"], "workload": wl_name, "seed": seed,
        "fmax_mhz": result.fmax_mhz, "resources": result.resources,
        "grid_w": grid_w, "grid_h": grid_h,
        "wirelength": result.wirelength, "skipped": False,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DPE Block Comparison Experiment")
    parser.add_argument("--skip-vtr", action="store_true",
                        help="Skip VTR runs, reuse existing results")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip VTR runs that already completed")
    parser.add_argument("--jobs", type=int, default=12,
                        help="Max parallel VTR workers (default: 12)")
    args = parser.parse_args()

    results_dir = BLOCK_COMP_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = BLOCK_COMP_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # Phase 1: VTR runs
    # ────────────────────────────────────────────────────────────────────
    vtr_results = []  # list of dicts from run_one_vtr

    if not args.skip_vtr:
        # Build job list
        jobs = []
        for setup in SETUPS:
            for wl_name, K, N in WORKLOADS:
                for seed in SEEDS:
                    jobs.append((setup, wl_name, K, N, seed))

        print(f"Phase 1: VTR flow ({len(jobs)} runs, {args.jobs} workers)")
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {}
            for setup, wl_name, K, N, seed in jobs:
                tag = f"{setup['name']}/{wl_name}/seed{seed}"
                fut = executor.submit(
                    run_one_vtr, setup, wl_name, K, N, seed, args.skip_existing)
                futures[fut] = tag

            for fut in as_completed(futures):
                tag = futures[fut]
                try:
                    r = fut.result()
                    status = "skip" if r.get("skipped") else "done"
                    print(f"  [{status}] {tag}: Fmax={r['fmax_mhz']:.1f} MHz")
                    vtr_results.append(r)
                except Exception as e:
                    print(f"  [FAIL] {tag}: {e}")

        elapsed = time.time() - t0
        print(f"Phase 1 complete: {len(vtr_results)}/{len(jobs)} runs in {elapsed:.0f}s\n")
    else:
        # Load existing VTR results from vtr_runs/
        print("Phase 1: Skipping VTR (--skip-vtr), loading existing results...")
        for setup in SETUPS:
            for wl_name, K, N in WORKLOADS:
                for seed in SEEDS:
                    run_dir = (BLOCK_COMP_DIR / "vtr_runs" / setup["name"]
                               / wl_name / f"seed{seed}")
                    try:
                        log_path = find_vpr_log(run_dir)
                        wirelength, fmax_mhz = parse_metrics(log_path)
                        resources = parse_resources(log_path)
                        grid_w, grid_h = parse_grid_size(log_path)
                        vtr_results.append({
                            "setup": setup["name"], "workload": wl_name,
                            "seed": seed, "fmax_mhz": fmax_mhz,
                            "resources": resources,
                            "grid_w": grid_w, "grid_h": grid_h,
                            "wirelength": wirelength, "skipped": True,
                        })
                    except Exception as e:
                        print(f"  [MISS] {setup['name']}/{wl_name}/seed{seed}: {e}")
        print(f"  Loaded {len(vtr_results)} existing results\n")

    # ────────────────────────────────────────────────────────────────────
    # Phase 2: Aggregate VTR results (average Fmax across seeds)
    # ────────────────────────────────────────────────────────────────────
    print("Phase 2: Aggregating VTR results...")

    # Group by (setup, workload)
    grouped = defaultdict(list)
    for r in vtr_results:
        grouped[(r["setup"], r["workload"])].append(r)

    aggregated = {}  # (setup_name, wl_name) -> dict
    for (setup_name, wl_name), seed_results in sorted(grouped.items()):
        fmax_values = [r["fmax_mhz"] for r in seed_results]
        avg_fmax = sum(fmax_values) / len(fmax_values)

        # Take resources from first seed (should be identical across seeds)
        resources = seed_results[0].get("resources", {})
        grid_w = seed_results[0].get("grid_w", 0)
        grid_h = seed_results[0].get("grid_h", 0)

        aggregated[(setup_name, wl_name)] = {
            "fmax_avg": avg_fmax,
            "fmax_seeds": fmax_values,
            "resources": resources,
            "grid_w": grid_w,
            "grid_h": grid_h,
        }
        seeds_str = ", ".join(f"{f:.1f}" for f in fmax_values)
        print(f"  {setup_name}/{wl_name}: Fmax avg={avg_fmax:.1f} MHz "
              f"[{seeds_str}], CLB={resources.get('clb', '?')}, "
              f"wc={resources.get('wc', '?')}")

    print()

    # ────────────────────────────────────────────────────────────────────
    # Phase 3: IMC Simulator
    # ────────────────────────────────────────────────────────────────────
    print("Phase 3: Running IMC simulator...")

    all_results = []

    for setup in SETUPS:
        sname = setup["name"]
        R, C = setup["rows"], setup["cols"]
        base_cfg_path = IMC_CONFIGS_DIR / setup["base_cfg"]

        for wl_name, K, N in WORKLOADS:
            key = (sname, wl_name)
            if key not in aggregated:
                print(f"  [SKIP] {sname}/{wl_name}: no VTR results")
                continue

            agg = aggregated[key]
            fmax = agg["fmax_avg"]

            # Patch IMC config
            cfg_path = configs_dir / f"{sname}_{wl_name}.json"
            patch_imc_config(base_cfg_path, R, C, fmax, setup["dpe_bw"],
                           setup["conv"], cfg_path)

            # Run IMC
            energy_pj, latency_ns, breakdown = run_imc_fc(cfg_path, K, N)
            print(f"  {sname}/{wl_name}: E={energy_pj:.1f} pJ, "
                  f"Lat={latency_ns:.1f} ns @ {fmax:.1f} MHz")

            # Block-level IO breakdown
            io = get_io_latency_breakdown(
                cfg_path, R, C, fmax, setup["dpe_bw"], K, N)

            # Tiling
            V = math.ceil(K / R)
            H = math.ceil(N / C)
            dpe_count = V * H

            row = {
                "setup": sname,
                "rows": R, "cols": C,
                "conv": setup["conv"],
                "dpe_bw": setup["dpe_bw"],
                "workload": wl_name,
                "K": K, "N": N,
                "V": V, "H": H,
                "dpe_count": dpe_count,
                "fmax_avg_mhz": round(fmax, 2),
                "fmax_seeds": ";".join(f"{f:.2f}" for f in agg["fmax_seeds"]),
                "clb": agg["resources"].get("clb", 0),
                "bram": agg["resources"].get("memory", 0),
                "wc": agg["resources"].get("wc", 0),
                "dsp": agg["resources"].get("dsp_top", 0),
                "grid_w": agg["grid_w"],
                "grid_h": agg["grid_h"],
                "energy_pj": round(energy_pj, 2) if energy_pj else None,
                "latency_ns": round(latency_ns, 2) if latency_ns else None,
                "read_ns": round(io["read_ns"], 2),
                "core_row_ns": round(io["core_row_ns"], 2),
                "output_serialize_ns": round(io["output_serialize_ns"], 2),
                "reduction_ns": round(io["reduction_ns"], 2),
                "write_ns": round(io["write_ns"], 2),
                "first_output_ns": round(io["first_output_ns"], 2),
                "steady_ns": round(io["steady_ns"], 2),
                "e_dpe_pj": round(breakdown.get("_dpe_grouped", 0), 2),
                "e_crossbar_pj": round(breakdown.get("imc_vmm", 0), 2),
                "e_adc_pj": round(breakdown.get("imc_conversion", 0), 2),
                "e_acam_pj": round(breakdown.get("imc_digital_post", 0), 2),
                "e_mem_pj": round(breakdown.get("_mem_grouped", 0), 2),
                "e_fpga_pj": round(breakdown.get("_fpga_grouped", 0), 2),
                "io_bottleneck": io["io_bottleneck"],
            }
            all_results.append(row)

    print()

    # ────────────────────────────────────────────────────────────────────
    # Phase 4: Write results CSV
    # ────────────────────────────────────────────────────────────────────
    csv_path = results_dir / "block_comparison_results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Phase 4: Results written to {csv_path}")
    else:
        print("Phase 4: No results to write!")
        return

    # ────────────────────────────────────────────────────────────────────
    # Phase 5: Pairwise comparison
    # ────────────────────────────────────────────────────────────────────
    print("\nPhase 5: Pairwise comparison\n")

    # Index results by (setup, workload)
    by_key = {(r["setup"], r["workload"]): r for r in all_results}

    comparisons = [
        ("setup1", "setup0", "Bus width (16→40)"),
        ("setup2", "setup1", "ACAM vs ADC"),
        ("setup3", "setup0", "Crossbar (512→1024)"),
        ("setup4", "setup0", "Crossbar + Bus"),
        ("setup5", "setup0", "All three (NL-DPE)"),
    ]

    lines = []
    header = (f"{'Comparison':<30} | {'fc_512_128':^30} | {'fc_2048_256':^30} "
              f"| {'CLB Δ':^10} | {'DPE Δ':^10}")
    sep = "-" * len(header)
    sub_header = (f"{'':30} | {'Lat ratio':>10} {'E ratio':>10} {'Fmax ratio':>8}"
                  f" | {'Lat ratio':>10} {'E ratio':>10} {'Fmax ratio':>8}"
                  f" | {'':^10} | {'':^10}")
    lines.append(header)
    lines.append(sep)
    lines.append(sub_header)
    lines.append(sep)

    for test_setup, ref_setup, desc in comparisons:
        parts = []
        clb_delta = "?"
        dpe_delta = "?"
        for wl_name, _, _ in WORKLOADS:
            ref = by_key.get((ref_setup, wl_name))
            test = by_key.get((test_setup, wl_name))
            if ref and test and ref["latency_ns"] and test["latency_ns"]:
                lat_ratio = test["latency_ns"] / ref["latency_ns"]
                e_ratio = (test["energy_pj"] / ref["energy_pj"]
                           if ref["energy_pj"] and test["energy_pj"] else 0)
                fmax_ratio = (test["fmax_avg_mhz"] / ref["fmax_avg_mhz"]
                              if ref["fmax_avg_mhz"] else 0)
                parts.append(f"{lat_ratio:>10.3f} {e_ratio:>10.3f} {fmax_ratio:>8.3f}")

                if wl_name == "fc_2048_256":
                    clb_delta = f"{test['clb'] - ref['clb']:+d}"
                    dpe_delta = f"{test['dpe_count'] - ref['dpe_count']:+d}"
            else:
                parts.append(f"{'N/A':>10} {'N/A':>10} {'N/A':>8}")

        label = f"{test_setup} vs {ref_setup} ({desc})"
        line = f"{label:<30} | {parts[0]} | {parts[1]} | {clb_delta:^10} | {dpe_delta:^10}"
        lines.append(line)

    comparison_text = "\n".join(lines)
    print(comparison_text)

    comp_path = results_dir / "pairwise_comparison.txt"
    with open(comp_path, 'w') as f:
        f.write(comparison_text + "\n")
    print(f"\nPairwise comparison written to {comp_path}")

    # ────────────────────────────────────────────────────────────────────
    # Phase 6: Sanity checks
    # ────────────────────────────────────────────────────────────────────
    print("\nPhase 6: Sanity checks")
    errors = 0

    for r in all_results:
        # Check Fmax range
        if r["fmax_avg_mhz"] < 10 or r["fmax_avg_mhz"] > 800:
            print(f"  [WARN] {r['setup']}/{r['workload']}: "
                  f"Fmax={r['fmax_avg_mhz']:.1f} MHz out of expected range")
            errors += 1

    # Check fc_512_128: setup0 and setup3 should have same buf_fill
    # (K=512 <= R for both 512 and 1024)
    s0 = by_key.get(("setup0", "fc_512_128"))
    s3 = by_key.get(("setup3", "fc_512_128"))
    if s0 and s3:
        # Both have dpe_bw=16, K=512, so buf_fill depends on min(K,R) and dpe_bw
        # setup0: min(512,512)=512, setup3: min(512,1024)=512 → same
        if abs(s0["read_ns"] - s3["read_ns"]) > 0.1:
            # At different Fmax, buf_fill will differ. Check normalized.
            norm_0 = s0["read_ns"] * s0["fmax_avg_mhz"]
            norm_3 = s3["read_ns"] * s3["fmax_avg_mhz"]
            if abs(norm_0 - norm_3) / max(norm_0, 1) > 0.01:
                print(f"  [WARN] fc_512_128 buf_fill mismatch: "
                      f"setup0={s0['read_ns']:.1f}ns, "
                      f"setup3={s3['read_ns']:.1f}ns "
                      f"(normalized: {norm_0:.0f} vs {norm_3:.0f})")
                errors += 1

    # Latency ordering for fc_512_128: setup5 <= setup2 <= setup1 <= setup0
    fc_small = [(s, by_key.get((s, "fc_512_128"))) for s in
                ["setup5", "setup2", "setup1", "setup0"]]
    fc_small = [(s, r) for s, r in fc_small if r and r["latency_ns"]]
    for i in range(len(fc_small) - 1):
        s1, r1 = fc_small[i]
        s2, r2 = fc_small[i + 1]
        if r1["latency_ns"] > r2["latency_ns"] * 1.05:  # 5% tolerance
            print(f"  [WARN] Latency ordering: {s1}={r1['latency_ns']:.1f}ns > "
                  f"{s2}={r2['latency_ns']:.1f}ns (expected <=)")
            errors += 1

    if errors == 0:
        print("  All sanity checks passed!")
    else:
        print(f"  {errors} warnings")

    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()
