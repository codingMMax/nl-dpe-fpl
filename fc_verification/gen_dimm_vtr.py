#!/usr/bin/env python3
"""
Generate DIMM RTL for VTR synthesis and run VTR + IMC experiment.

Config: Proposed NL-DPE (R=1024, C=128, dpe_buf_width=40)
Workload: DIMM score matrix (QK^T) for BERT-Tiny (d=64, S=128, K_id=2)

Usage:
    python3 fc_verification/gen_dimm_vtr.py              # generate + VTR + IMC
    python3 fc_verification/gen_dimm_vtr.py --skip-vtr   # IMC only (reuse VTR)
"""
import sys
import math
import os
import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "nl_dpe"))

from gen_attention_wrapper import _gen_dimm_score_matrix
from gen_gemv_wrappers import _get_supporting_modules
from run_vtr import run_single, find_vpr_log, parse_metrics, parse_resources
from area_power import dpe_specs

# ── Config ──────────────────────────────────────────────────────────
R, C = 1024, 128
DPE_BW = 40
DW = DPE_BW
EPW = DW // 8
D_HEAD = 64
N_SEQ = 128
K_ID = C // D_HEAD  # 2

PACKED_D = math.ceil(D_HEAD / EPW)
PACKED_ND = N_SEQ * PACKED_D

DEPTH_Q = PACKED_D + 1
DEPTH_K = PACKED_ND + 1
DEPTH_SCORE = N_SEQ + 1
H_DIMM = 1

SEEDS = [1, 2, 3]

# Paths
VTR_ROOT = Path(os.environ.get("VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"))
VTR_FLOW = VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py"
VTR_PYTHON = VTR_ROOT / ".venv" / "bin" / "python"
if not VTR_PYTHON.is_file():
    VTR_PYTHON = None

ARCH_XML = PROJECT_ROOT / "dse" / "configs" / "arch" / "nl_dpe_1024x128_auto.xml"
AZURELILY_ROOT = PROJECT_ROOT / "azurelily"
IMC_TEST = AZURELILY_ROOT / "IMC" / "test.py"
IMC_CONFIG = AZURELILY_ROOT / "IMC" / "configs" / "nl_dpe.json"

OUT_DIR = PROJECT_ROOT / "fc_verification"
RTL_DIR = OUT_DIR / "rtl"
VTR_DIR = OUT_DIR / "vtr_runs" / "dimm_d64_s128"
RESULTS_DIR = OUT_DIR / "results"

# Regex
RE_ENERGY = re.compile(r"Energy total \(by layer\):\s+([\d.]+)\s+pJ")
RE_LAT = re.compile(r"Latency total \(critical path\):\s+([\d.]+)\s+ns")
RE_GROUPED = re.compile(r"Energy grouped:\s+DPE=([\d.]+)\s+pJ,\s+Memory=([\d.]+)\s+pJ,\s+FPGA=([\d.-]+)\s+pJ")
RE_BREAKDOWN = re.compile(r"^\s+(\S+):\s+([\d.]+)\s+pJ", re.MULTILINE)
GRID_RE = re.compile(r"FPGA sized to (\d+) x (\d+)")


def generate_rtl():
    """Generate DIMM RTL for VTR."""
    rtl = _gen_dimm_score_matrix(
        n_seq=N_SEQ, d_head=D_HEAD, h_dimm=H_DIMM,
        depth_q=DEPTH_Q, depth_k=DEPTH_K, depth_score=DEPTH_SCORE,
        data_width=DW, dual_identity=(K_ID >= 2), uid=0,
    )
    # Only include sram module (needed by DIMM score matrix)
    # Don't include conv_layer_single_dpe etc. (unused by direct DPE)
    supporting = _get_supporting_modules()

    out_path = RTL_DIR / "dimm_pipeline_d64_c128.v"
    with open(out_path, "w") as f:
        f.write(f"// DIMM for VTR: R={R}, C={C}, d={D_HEAD}, S={N_SEQ}, K_id={K_ID}\n\n")
        f.write(rtl)
        f.write("\n\n")
        f.write(supporting)

    print(f"RTL: {out_path}")
    return out_path


def run_vtr_seeds(rtl_path):
    """Run VTR with 3 seeds, return average Fmax and resources."""
    fmax_list = []
    resources = None

    for seed in SEEDS:
        seed_dir = VTR_DIR / f"seed{seed}"
        done_flag = seed_dir / "vtr_done.flag"

        if done_flag.is_file():
            # Parse existing
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
            design=rtl_path, arch=ARCH_XML,
            route_chan_width=300, sdc_file=None,
            run_dir=seed_dir, seed=seed,
            run_index=0, total_runs=1,
            design_name="dimm_d64_s128",
        )
        fmax_list.append(result.fmax_mhz)
        if resources is None:
            resources = result.resources
        done_flag.write_text(f"fmax={result.fmax_mhz:.2f}\n")
        print(f"  [done] seed {seed}: Fmax={result.fmax_mhz:.1f} MHz")

    avg_fmax = sum(fmax_list) / len(fmax_list)
    return avg_fmax, fmax_list, resources


def run_imc(fmax_mhz):
    """Run IMC simulator for the DIMM workload at given Fmax."""
    # Patch config
    specs = dpe_specs(R, C, freq_ghz=fmax_mhz / 1000.0)
    with open(IMC_CONFIG) as f:
        cfg = json.load(f)
    cfg['geometry']['array_rows'] = R
    cfg['geometry']['array_cols'] = C
    cfg['params']['e_analoge_pj'] = specs['e_analogue_pj']
    cfg['params']['e_digital_pj'] = specs['e_digital_pj']
    cfg['fpga_specs']['freq'] = fmax_mhz
    cfg['fpga_specs']['dpe_buf_width'] = DPE_BW

    patched_cfg = RESULTS_DIR / "dimm_imc_config.json"
    with open(patched_cfg, 'w') as f:
        json.dump(cfg, f, indent=4)

    # Run: gemm_log(M=S, K=d, N=S) for QK^T
    # Use fc model with K=d_head, N=seq_len to exercise dimm_nonlinear
    # Actually, the IMC test.py doesn't have a standalone DIMM workload.
    # Use --model attention with seq_length and head_dim
    cmd = [
        sys.executable, str(IMC_TEST),
        "--model", "attention",
        "--imc_file", str(patched_cfg),
        "--seq_length", str(N_SEQ),
        "--head_dim", str(D_HEAD),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(AZURELILY_ROOT))
    out = result.stdout + result.stderr

    # Parse
    m = RE_ENERGY.search(out)
    energy_pj = float(m.group(1)) if m else None
    m = RE_LAT.search(out)
    latency_ns = float(m.group(1)) if m else None

    breakdown = {}
    for m in RE_BREAKDOWN.finditer(out):
        breakdown[m.group(1)] = float(m.group(2))

    m = RE_GROUPED.search(out)
    if m:
        breakdown['_dpe'] = float(m.group(1))
        breakdown['_mem'] = float(m.group(2))
        breakdown['_fpga'] = float(m.group(3))

    if energy_pj is None:
        print(f"  WARNING: IMC failed\n{out[-500:]}")

    return energy_pj, latency_ns, breakdown, out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vtr", action="store_true")
    args = parser.parse_args()

    RTL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DIMM VTR + IMC Experiment")
    print(f"  Config: R={R}, C={C}, d={D_HEAD}, S={N_SEQ}, K_id={K_ID}")
    print("=" * 60)

    # Step 1: Generate RTL
    print("\n[1] Generating RTL...")
    rtl_path = generate_rtl()

    # Step 2: Run VTR
    if not args.skip_vtr:
        print("\n[2] Running VTR (3 seeds)...")
        avg_fmax, fmax_seeds, resources = run_vtr_seeds(rtl_path)
    else:
        print("\n[2] Skipping VTR, loading existing...")
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
            except:
                pass
        avg_fmax = sum(fmax_list) / len(fmax_list) if fmax_list else 300.0
        fmax_seeds = fmax_list

    print(f"\n  Fmax avg: {avg_fmax:.1f} MHz ({', '.join(f'{f:.1f}' for f in fmax_seeds)})")
    if resources:
        print(f"  Resources: CLB={resources.get('clb', '?')}, "
              f"BRAM={resources.get('memory', '?')}, "
              f"DPE(wc)={resources.get('wc', '?')}, "
              f"DSP={resources.get('dsp_top', '?')}")

    # Step 3: Run IMC
    print(f"\n[3] Running IMC simulator @ {avg_fmax:.1f} MHz...")
    energy_pj, latency_ns, breakdown, imc_out = run_imc(avg_fmax)
    print(f"  Energy: {energy_pj:.1f} pJ")
    print(f"  Latency: {latency_ns:.1f} ns")

    # Step 4: Save results
    results = {
        "config": f"Proposed_{R}x{C}",
        "d_head": D_HEAD, "seq_len": N_SEQ, "K_id": K_ID,
        "fmax_avg_mhz": round(avg_fmax, 2),
        "fmax_seeds": [round(f, 2) for f in fmax_seeds],
        "clb": resources.get('clb', 0) if resources else 0,
        "bram": resources.get('memory', 0) if resources else 0,
        "wc": resources.get('wc', 0) if resources else 0,
        "dsp": resources.get('dsp_top', 0) if resources else 0,
        "energy_pj": round(energy_pj, 2) if energy_pj else None,
        "latency_ns": round(latency_ns, 2) if latency_ns else None,
        "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
    }

    results_path = RESULTS_DIR / "dimm_vtr_imc_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {results_path}")

    # Save IMC output
    with open(RESULTS_DIR / "dimm_imc_output.txt", 'w') as f:
        f.write(imc_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
