#!/usr/bin/env python3
"""BERT-Tiny End-to-End Benchmark: VTR results → IMC simulator → final metrics.

Reads VTR Fmax + resource usage, patches IMC simulator config, runs BERT-Tiny
model, computes energy/latency/throughput/area metrics for all 4 architectures.

Usage:
    python benchmarks/run_bert_tiny_benchmark.py
    python benchmarks/run_bert_tiny_benchmark.py --vtr-dir benchmarks/vtr_runs
"""
import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "azurelily"))
sys.path.insert(0, str(PROJECT_DIR / "azurelily" / "IMC"))
sys.path.insert(0, str(PROJECT_DIR / "nl_dpe"))

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
from models.bert_tiny import bert_tiny_model

CLB_TILE_UM2 = 2239  # includes routing

# Architecture configs: (config_json, rows, cols, dpe_tile_w, dpe_tile_h)
ARCH_CONFIGS = {
    "baseline":   ("baseline.json",    1,    1,   0, 0),
    "proposed":   ("nl_dpe.json",      1024, 128, 3, 7),
    "al_like":    ("nl_dpe.json",      1024, 256, 5, 8),
    "azurelily":  ("azure_lily.json",  512,  128, 6, 5),
}


def _functional_dpes(rows, cols):
    """Non-DIMM DPEs: Q/K/V/O projections + FFN1 + FFN2, × 2 blocks."""
    D_MODEL, D_FF, NUM_BLOCKS = 128, 512, 2
    V_proj = math.ceil(D_MODEL / rows)
    H_proj = math.ceil(D_MODEL / cols)
    V_ffn1 = math.ceil(D_MODEL / rows)
    H_ffn1 = math.ceil(D_FF / cols)
    V_ffn2 = math.ceil(D_FF / rows)
    H_ffn2 = math.ceil(D_MODEL / cols)
    per_block = 4 * V_proj * H_proj + V_ffn1 * H_ffn1 + V_ffn2 * H_ffn2
    return per_block * NUM_BLOCKS

# VTR ground truth resources (available, from count_resources.py)
VTR_AVAILABLE = {
    "baseline":  {"DPEs": 0,   "DSPs": 333, "BRAMs": 740, "CLBs": 19092},
    "proposed":  {"DPEs": 294, "DSPs": 222, "BRAMs": 518, "CLBs": 13806},
    "al_like":   {"DPEs": 90,  "DSPs": 222, "BRAMs": 444, "CLBs": 16528},
    "azurelily": {"DPEs": 261, "DSPs": 333, "BRAMs": 740, "CLBs": 11262},
}


def parse_vtr_results(vtr_dir, arch_name):
    """Parse Fmax and resource usage from VTR run logs (3 seeds)."""
    base = Path(vtr_dir) / f"bert_tiny_{arch_name}"
    fmax_values = []
    resources = None

    for seed in [1, 2, 3]:
        seed_dir = base / f"seed_{seed}"
        # Find vpr_stdout.log
        log_files = list(seed_dir.rglob("vpr_stdout.log"))
        if not log_files:
            print(f"  WARNING: No vpr_stdout.log in {seed_dir}")
            continue

        with open(log_files[0]) as f:
            txt = f.read()

        # Extract Fmax
        fmax_matches = re.findall(r'Fmax:\s*([\d.]+)\s*MHz', txt)
        if fmax_matches:
            fmax_values.append(float(fmax_matches[-1]))

        # Extract resource usage (first occurrence = used)
        if resources is None:
            resources = {}
            for btype, label in [("wc", "DPEs"), ("clb", "CLBs"),
                                  ("dsp_top", "DSPs"), ("memory", "BRAMs")]:
                matches = re.findall(rf'(\d+)\s+blocks of type: {btype}', txt)
                if matches:
                    resources[label] = int(matches[0])  # first = used
                else:
                    resources[label] = 0

    if not fmax_values:
        return None

    avg_fmax = sum(fmax_values) / len(fmax_values)
    return {
        "fmax_avg": avg_fmax,
        "fmax_seeds": fmax_values,
        "resources_used": resources,
    }


def run_bert_sim(arch_name, fmax, resources_used):
    """Run IMC simulator for BERT-Tiny and return energy/latency."""
    cfg_file, R, C, tw, th = ARCH_CONFIGS[arch_name]
    cfg_path = str(PROJECT_DIR / "azurelily" / "IMC" / "configs" / cfg_file)

    cfg = Config(cfg_path)
    cfg.rows = R
    cfg.cols = C
    cfg.freq = fmax

    # Use EXACT resources reported by VTR (used, not available)
    cfg.total_dsp = resources_used.get("DSPs", 0)
    cfg.total_clb = resources_used.get("CLBs", 0)
    cfg.total_mem = resources_used.get("BRAMs", 0)

    # DIMM DPE pool = VTR used DPEs - functional (projection/FFN) DPEs
    dpe_used = resources_used.get("DPEs", 0)
    cfg.total_dimm_dpes = max(0, dpe_used - _functional_dpes(R, C))

    stats = Stats()
    mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)

    md, _ = bert_tiny_model(1, 1, 128, 128, False, False)
    for layer in md["embedding"]:
        sched.run_layer(layer)
    for block in md["blocks"]:
        for layer in block["qkv_proj"]:
            sched.run_layer(layer)
        for hi in range(block["num_heads"]):
            for layer in block["head_attention"]:
                sched.run_layer(layer)
        for layer in block["post_attn"]:
            sched.run_layer(layer)
        for layer in block["ffn"]:
            sched.run_layer(layer)

    total_energy = sum(stats.energy_breakdown.values())
    total_latency = sum(stats.latency_breakdown.values())
    return total_energy, total_latency, dict(stats.energy_breakdown)


def compute_used_area(arch_name, resources_used):
    """Compute area from USED resources (not total FPGA).

    Each resource tile occupies grid cells × CLB_TILE_UM2 (includes routing).
    """
    _, _, _, tw, th = ARCH_CONFIGS[arch_name]

    dpe_cells = resources_used.get("DPEs", 0) * tw * th
    dsp_cells = resources_used.get("DSPs", 0) * 1 * 4   # DSP tile: 1×4
    bram_cells = resources_used.get("BRAMs", 0) * 1 * 2  # BRAM tile: 1×2
    clb_cells = resources_used.get("CLBs", 0) * 1 * 1    # CLB tile: 1×1

    total_cells = dpe_cells + dsp_cells + bram_cells + clb_cells
    return total_cells * CLB_TILE_UM2 / 1e6  # mm²


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtr-dir", default="benchmarks/vtr_runs",
                        help="Directory with VTR run results")
    args = parser.parse_args()

    vtr_dir = PROJECT_DIR / args.vtr_dir
    results = []

    print("BERT-Tiny End-to-End Benchmark (150×150 FPGA)")
    print("=" * 80)

    for arch_name in ["baseline", "proposed", "al_like", "azurelily"]:
        print(f"\n--- {arch_name} ---")

        # Parse VTR results
        vtr = parse_vtr_results(vtr_dir, arch_name)
        if vtr is None:
            print(f"  SKIPPED: No VTR results found")
            continue

        fmax = vtr["fmax_avg"]
        res_used = vtr["resources_used"]
        print(f"  Fmax: {fmax:.1f} MHz (seeds: {vtr['fmax_seeds']})")
        print(f"  Used: DPEs={res_used.get('DPEs',0)}, DSPs={res_used.get('DSPs',0)}, "
              f"BRAMs={res_used.get('BRAMs',0)}, CLBs={res_used.get('CLBs',0)}")

        # Run simulator
        energy_pj, latency_ns, breakdown = run_bert_sim(arch_name, fmax, res_used)
        print(f"  Energy: {energy_pj/1e6:.3f} M pJ")
        print(f"  Latency: {latency_ns/1e3:.1f} µs")

        # Compute metrics
        throughput = 1e9 / latency_ns  # inferences/s
        used_area = compute_used_area(arch_name, res_used)
        throughput_per_mm2 = throughput / used_area if used_area > 0 else 0
        throughput_per_j = 1e12 / energy_pj if energy_pj > 0 else 0

        print(f"  Throughput: {throughput:.0f} inf/s")
        print(f"  Used area: {used_area:.2f} mm²")
        print(f"  Throughput/mm²: {throughput_per_mm2:.0f} inf/s/mm²")
        print(f"  Throughput/J: {throughput_per_j:.0f} inf/J")

        # Top energy breakdown
        top_key = max(breakdown, key=breakdown.get)
        top_pct = breakdown[top_key] / energy_pj * 100
        print(f"  Top energy: {top_key} ({top_pct:.0f}%)")

        results.append({
            "arch": arch_name,
            "fmax_avg": round(fmax, 1),
            "fmax_s1": vtr["fmax_seeds"][0] if len(vtr["fmax_seeds"]) > 0 else 0,
            "fmax_s2": vtr["fmax_seeds"][1] if len(vtr["fmax_seeds"]) > 1 else 0,
            "fmax_s3": vtr["fmax_seeds"][2] if len(vtr["fmax_seeds"]) > 2 else 0,
            "dpe_used": res_used.get("DPEs", 0),
            "dsp_used": res_used.get("DSPs", 0),
            "bram_used": res_used.get("BRAMs", 0),
            "clb_used": res_used.get("CLBs", 0),
            "energy_pj": round(energy_pj, 2),
            "latency_ns": round(latency_ns, 2),
            "throughput_infs": round(throughput, 2),
            "used_area_mm2": round(used_area, 4),
            "throughput_per_mm2": round(throughput_per_mm2, 2),
            "throughput_per_j": round(throughput_per_j, 2),
            "top_breakdown": top_key,
            "top_breakdown_pct": round(top_pct, 1),
        })

    # Write CSV
    if results:
        out_dir = PROJECT_DIR / "benchmarks" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "bert_tiny_final_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {csv_path}")

        # Summary table
        print(f"\n{'='*80}")
        print(f"{'Arch':>12} {'Energy(MpJ)':>12} {'Lat(µs)':>10} {'Tput(inf/s)':>12} "
              f"{'Area(mm²)':>10} {'Tput/mm²':>10} {'Tput/J':>10}")
        print(f"{'-'*80}")
        for r in results:
            print(f"{r['arch']:>12} {r['energy_pj']/1e6:>11.3f} "
                  f"{r['latency_ns']/1e3:>9.1f} {r['throughput_infs']:>11.0f} "
                  f"{r['used_area_mm2']:>9.2f} {r['throughput_per_mm2']:>9.0f} "
                  f"{r['throughput_per_j']:>9.0f}")


if __name__ == "__main__":
    main()
