#!/usr/bin/env python3
"""
VTR + IMC experiment for Azure-Lily full DIMM top (W=16).

Reads fc_verification/rtl/azurelily_dimm_top_d64_c128.v (generated separately
by nl_dpe/gen_dimm_azurelily_top.py). Runs VTR 3 seeds, parses Fmax/CLB/BRAM/
DSP, runs IMC at VTR-measured Fmax, saves JSON summary.

Expected metrics (Phase L targets):
  DSP count: ~32 (2 matmul stages × 16 lanes × 1 int_sop_4 each) ± packing drift
  DPE count: 0 (AL uses DSPs, not DPEs)
  Fmax/CLB/BRAM: accept whatever VTR reports

Usage:
    python3 fc_verification/gen_azurelily_dimm_top_vtr.py
    python3 fc_verification/gen_azurelily_dimm_top_vtr.py --skip-vtr  # IMC only
"""
import sys
import os
import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "nl_dpe"))

from run_vtr import run_single, find_vpr_log, parse_metrics, parse_resources

# ── Config ──────────────────────────────────────────────────────────
D_HEAD = 64
N_SEQ = 128
W = 16
DPE_BW = 40

SEEDS = [1, 2, 3]

VTR_ROOT = Path(os.environ.get("VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"))
VTR_FLOW = VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py"
VTR_PYTHON = VTR_ROOT / ".venv" / "bin" / "python"
if not VTR_PYTHON.is_file():
    VTR_PYTHON = None

ARCH_XML = PROJECT_ROOT / "benchmarks" / "arch" / "azure_lily_auto.xml"
AZURELILY_ROOT = PROJECT_ROOT / "azurelily"
IMC_TEST = AZURELILY_ROOT / "IMC" / "test.py"
IMC_CONFIG = AZURELILY_ROOT / "IMC" / "configs" / "azure_lily.json"

RTL_PATH = PROJECT_ROOT / "fc_verification" / "rtl" / "azurelily_dimm_top_d64_c128.v"
VTR_DIR = PROJECT_ROOT / "fc_verification" / "vtr_runs" / "azurelily_dimm_top_d64_c128"
RESULTS_DIR = PROJECT_ROOT / "fc_verification" / "results"

RE_ENERGY = re.compile(r"Energy total \(by layer\):\s+([\d.]+)\s+pJ")
RE_LAT = re.compile(r"Latency total \(critical path\):\s+([\d.]+)\s+ns")
RE_GROUPED = re.compile(r"Energy grouped:\s+DPE=([\d.]+)\s+pJ,\s+Memory=([\d.]+)\s+pJ,\s+FPGA=([\d.-]+)\s+pJ")
RE_BREAKDOWN = re.compile(r"^\s+(\S+):\s+([\d.]+)\s+pJ", re.MULTILINE)


def regenerate_rtl():
    """Regenerate the Azure-Lily full DIMM top RTL from the generator."""
    gen = PROJECT_ROOT / "nl_dpe" / "gen_dimm_azurelily_top.py"
    subprocess.check_call([
        sys.executable, str(gen),
        "--N", str(N_SEQ), "--d", str(D_HEAD),
        "--C", str(128), "--W", str(W),
    ])
    assert RTL_PATH.is_file(), f"RTL missing: {RTL_PATH}"
    print(f"RTL regenerated: {RTL_PATH}")


def run_vtr_seeds():
    """Run VTR 3 seeds, return average Fmax + per-seed list + parsed resources."""
    fmax_list = []
    resources = None

    for seed in SEEDS:
        seed_dir = VTR_DIR / f"seed{seed}"
        done_flag = seed_dir / "vtr_done.flag"

        if done_flag.is_file():
            log = find_vpr_log(seed_dir)
            _, fmax = parse_metrics(log)
            if resources is None:
                resources = parse_resources(log)
            fmax_list.append(fmax)
            print(f"  [skip] seed {seed}: Fmax={fmax:.1f} MHz")
            continue

        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Running VTR seed {seed}...")
        result = run_single(
            vtr_flow=VTR_FLOW, vtr_python=VTR_PYTHON,
            design=RTL_PATH, arch=ARCH_XML,
            route_chan_width=300, sdc_file=None,
            run_dir=seed_dir, seed=seed,
            run_index=0, total_runs=1,
            design_name="azurelily_dimm_top_d64_c128",
        )
        fmax_list.append(result.fmax_mhz)
        if resources is None:
            resources = result.resources
        done_flag.write_text(f"fmax={result.fmax_mhz:.2f}\n")
        print(f"  [done] seed {seed}: Fmax={result.fmax_mhz:.1f} MHz")

    avg_fmax = sum(fmax_list) / len(fmax_list)
    return avg_fmax, fmax_list, resources


def run_imc(fmax_mhz):
    """Run IMC simulator for the AL attention workload at given Fmax."""
    with open(IMC_CONFIG) as f:
        cfg = json.load(f)
    cfg['fpga_specs']['freq'] = fmax_mhz
    cfg['fpga_specs']['dpe_buf_width'] = DPE_BW
    cfg['fpga_specs']['total_dsp'] = W

    patched_cfg = RESULTS_DIR / "azurelily_dimm_top_imc_config.json"
    with open(patched_cfg, 'w') as f:
        json.dump(cfg, f, indent=4)

    cmd = [
        sys.executable, str(IMC_TEST),
        "--model", "attention",
        "--imc_file", str(patched_cfg),
        "--seq_length", str(N_SEQ),
        "--head_dim", str(D_HEAD),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(AZURELILY_ROOT))
    out = result.stdout + result.stderr

    m = RE_ENERGY.search(out)
    energy_pj = float(m.group(1)) if m else None
    m = RE_LAT.search(out)
    latency_ns = float(m.group(1)) if m else None

    breakdown = {}
    for mm in RE_BREAKDOWN.finditer(out):
        breakdown[mm.group(1)] = float(mm.group(2))
    m = RE_GROUPED.search(out)
    if m:
        breakdown['_dpe'] = float(m.group(1))
        breakdown['_mem'] = float(m.group(2))
        breakdown['_fpga'] = float(m.group(3))

    if energy_pj is None:
        print(f"  WARNING: IMC parse failed\n{out[-800:]}")

    return energy_pj, latency_ns, breakdown, out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vtr", action="store_true")
    ap.add_argument("--skip-imc", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VTR_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"  Azure-Lily Full DIMM Top VTR + IMC (W={W}, N={N_SEQ}, d={D_HEAD})")
    print("=" * 68)

    print("\n[1] Regenerating RTL...")
    regenerate_rtl()

    if not args.skip_vtr:
        print("\n[2] Running VTR 3 seeds (may take a while)...")
        avg_fmax, fmax_seeds, resources = run_vtr_seeds()
    else:
        print("\n[2] Skipping VTR, loading existing runs...")
        fmax_list = []
        resources = None
        for seed in SEEDS:
            seed_dir = VTR_DIR / f"seed{seed}"
            try:
                log = find_vpr_log(seed_dir)
                _, fmax = parse_metrics(log)
                fmax_list.append(fmax)
                if resources is None:
                    resources = parse_resources(log)
            except Exception as e:
                print(f"  seed {seed}: parse failed: {e}")
        avg_fmax = sum(fmax_list) / len(fmax_list) if fmax_list else 300.0
        fmax_seeds = fmax_list

    print(f"\n  Fmax avg: {avg_fmax:.1f} MHz  (seeds: {', '.join(f'{f:.1f}' for f in fmax_seeds)})")
    if resources:
        print(f"  Resources: CLB={resources.get('clb', '?')}, "
              f"BRAM={resources.get('memory', '?')}, "
              f"DPE(wc)={resources.get('wc', '?')}, "
              f"DSP={resources.get('dsp_top', '?')}")

    imc_result = None
    if not args.skip_imc:
        print(f"\n[3] Running IMC @ {avg_fmax:.1f} MHz...")
        energy_pj, latency_ns, breakdown, imc_out = run_imc(avg_fmax)
        print(f"  Energy: {energy_pj:.1f} pJ" if energy_pj else "  Energy: FAILED")
        print(f"  Latency: {latency_ns:.1f} ns" if latency_ns else "  Latency: FAILED")
        imc_result = (energy_pj, latency_ns, breakdown, imc_out)

    print("\n[4] Saving results...")
    results = {
        "config": f"AzureLily_W={W}",
        "d_head": D_HEAD, "seq_len": N_SEQ, "W": W,
        "fmax_avg_mhz": round(avg_fmax, 2),
        "fmax_seeds": [round(f, 2) for f in fmax_seeds],
        "clb": resources.get('clb', 0) if resources else 0,
        "bram": resources.get('memory', 0) if resources else 0,
        "wc": resources.get('wc', 0) if resources else 0,  # DPE count (should be 0)
        "dsp": resources.get('dsp_top', 0) if resources else 0,
        "imc_energy_pj": round(imc_result[0], 2) if (imc_result and imc_result[0]) else None,
        "imc_latency_ns": round(imc_result[1], 2) if (imc_result and imc_result[1]) else None,
        "imc_breakdown": ({k: round(v, 4) for k, v in imc_result[2].items()}
                          if imc_result else {}),
    }
    results_path = RESULTS_DIR / "azurelily_dimm_top_vtr_imc_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {results_path}")

    if imc_result:
        with open(RESULTS_DIR / "azurelily_dimm_top_imc_output.txt", 'w') as f:
            f.write(imc_result[3])

    print("\nDone.")


if __name__ == "__main__":
    main()
