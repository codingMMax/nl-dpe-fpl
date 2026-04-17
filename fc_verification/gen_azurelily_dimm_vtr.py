#!/usr/bin/env python3
"""Azure-Lily DIMM block-level comparison: IMC simulator run.

Mirrors gen_dimm_vtr.py but uses Azure-Lily config (DSP + CLB for attention).
VTR synthesis run is optional — by default we just run the IMC simulator to
produce an energy/latency breakdown for comparison with NL-DPE.

Usage:
    python3 fc_verification/gen_azurelily_dimm_vtr.py            # IMC only
    python3 fc_verification/gen_azurelily_dimm_vtr.py --with-vtr # + VTR (slower)
"""
import sys
import os
import json
import re
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Config
D_HEAD = 64
N_SEQ = 128
AZURELILY_ROOT = PROJECT_ROOT / "azurelily"
IMC_TEST = AZURELILY_ROOT / "IMC" / "test.py"
IMC_CONFIG = AZURELILY_ROOT / "IMC" / "configs" / "azure_lily.json"

RESULTS_DIR = PROJECT_ROOT / "fc_verification" / "results"
OUT_CFG = RESULTS_DIR / "azurelily_dimm_imc_config.json"
OUT_JSON = RESULTS_DIR / "azurelily_dimm_vtr_imc_results.json"
OUT_TXT = RESULTS_DIR / "azurelily_dimm_imc_output.txt"

RE_ENERGY = re.compile(r"Energy total \(by layer\):\s+([\d.]+)\s+pJ")
RE_LAT = re.compile(r"Latency total \(critical path\):\s+([\d.]+)\s+ns")
RE_GROUPED = re.compile(
    r"Energy grouped:\s+DPE=([\d.]+)\s+pJ,\s+Memory=([\d.]+)\s+pJ,\s+FPGA=([\d.-]+)\s+pJ"
)
RE_BREAKDOWN = re.compile(r"^\s+(\S+):\s+([\d.]+)\s+pJ", re.MULTILINE)


def run_imc():
    """Patch config (write out a copy) and run the IMC simulator."""
    with open(IMC_CONFIG) as f:
        cfg = json.load(f)

    # Keep Azure-Lily defaults (freq=300 MHz, dpe_buf_width=16 for DPE projections).
    # But for apples-to-apples comparison, use 40-bit bus (matches NL-DPE test config).
    cfg["fpga_specs"]["dpe_buf_width"] = 40  # match NL-DPE bus for DIMM DSP path

    with open(OUT_CFG, "w") as f:
        json.dump(cfg, f, indent=4)

    cmd = [
        sys.executable, str(IMC_TEST),
        "--model", "attention",
        "--imc_file", str(OUT_CFG),
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
    for m in RE_BREAKDOWN.finditer(out):
        breakdown[m.group(1)] = float(m.group(2))

    m = RE_GROUPED.search(out)
    if m:
        breakdown["_dpe"] = float(m.group(1))
        breakdown["_mem"] = float(m.group(2))
        breakdown["_fpga"] = float(m.group(3))

    if energy_pj is None:
        print(f"WARNING: IMC parse failed. Raw output tail:\n{out[-800:]}")

    return energy_pj, latency_ns, breakdown, out, cfg["fpga_specs"]["freq"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-vtr", action="store_true",
                        help="Also run VTR synthesis (slow, optional)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Azure-Lily DIMM Block-Level Comparison")
    print(f"  Config: Azure-Lily (DSP + CLB path, d={D_HEAD}, S={N_SEQ})")
    print("=" * 60)

    if args.with_vtr:
        print("\n[VTR synthesis not implemented in this orchestrator.")
        print(" For now, IMC results use azure_lily.json default Fmax (300 MHz).]")

    print("\n[IMC] Running attention model on Azure-Lily config...")
    energy_pj, latency_ns, breakdown, raw_out, fmax_mhz = run_imc()
    if energy_pj is None:
        sys.exit(1)
    print(f"  Energy : {energy_pj:.1f} pJ")
    print(f"  Latency: {latency_ns:.1f} ns")

    results = {
        "config": "Azure-Lily",
        "d_head": D_HEAD,
        "seq_len": N_SEQ,
        "fmax_mhz": fmax_mhz,
        "energy_pj": round(energy_pj, 2),
        "latency_ns": round(latency_ns, 2),
        "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_TXT, "w") as f:
        f.write(raw_out)

    print(f"\n  Results: {OUT_JSON}")
    print(f"  Raw out: {OUT_TXT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
