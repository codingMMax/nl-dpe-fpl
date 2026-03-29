#!/usr/bin/env python3
"""CNN Benchmark: VTR results → IMC simulator → final metrics.

Reads VTR Fmax + resource usage for ResNet-9 and VGG-11, patches IMC simulator,
computes energy/latency/throughput/area metrics for all 3 architectures.

Usage:
    python benchmarks/run_cnn_benchmark.py
    python benchmarks/run_cnn_benchmark.py --vtr-dir benchmarks/vtr_runs
"""
import argparse
import csv
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

CLB_TILE_UM2 = 2239

# Architecture configs: (config_json, rows, cols, dpe_tile_w, dpe_tile_h)
ARCH_CONFIGS = {
    "proposed":  ("nl_dpe.json",      1024, 128, 3, 7),
    "al_like":   ("nl_dpe.json",      1024, 256, 5, 8),
    "azurelily": ("azure_lily.json",   512, 128, 6, 5),
}

# VTR ground truth: AVAILABLE resources per arch
VTR_AVAILABLE = {
    "proposed":  {"DSPs": 222, "CLBs": 13806, "BRAMs": 518},
    "al_like":   {"DSPs": 222, "CLBs": 16528, "BRAMs": 444},
    "azurelily": {"DSPs": 333, "CLBs": 11262, "BRAMs": 740},
}

# RTL → arch mapping
RTL_ARCH_MAP = {
    "resnet9_proposed":     "proposed",
    "resnet9_al_matched":   "al_like",
    "resnet_1_azurelily":   "azurelily",
    "vgg11_proposed":       "proposed",
    "vgg11_al_matched":     "al_like",
    "vgg11_1_azurelily":    "azurelily",
}

# RTL → model mapping
RTL_MODEL_MAP = {
    "resnet9_proposed":     "resnet",
    "resnet9_al_matched":   "resnet",
    "resnet_1_azurelily":   "resnet",
    "vgg11_proposed":       "vgg",
    "vgg11_al_matched":     "vgg",
    "vgg11_1_azurelily":    "vgg",
}

# Display names
RTL_DISPLAY = {
    "resnet9_proposed":     "ResNet-9 Proposed",
    "resnet9_al_matched":   "ResNet-9 AL-like",
    "resnet_1_azurelily":   "ResNet-9 Azure-Lily",
    "vgg11_proposed":       "VGG-11 Proposed",
    "vgg11_al_matched":     "VGG-11 AL-like",
    "vgg11_1_azurelily":    "VGG-11 Azure-Lily",
}


def parse_vtr_results(vtr_dir, rtl_name):
    """Parse Fmax and resource usage from VTR run logs (3 seeds)."""
    base = Path(vtr_dir) / rtl_name
    fmax_values = []
    resources = None

    for seed_idx in [1, 2, 3]:
        seed_dir = base / f"seed_{seed_idx}"
        log_files = list(seed_dir.rglob("vpr_stdout.log"))
        if not log_files:
            continue

        with open(log_files[0]) as f:
            txt = f.read()

        fmax_matches = re.findall(r'Fmax:\s*([\d.]+)\s*MHz', txt)
        if fmax_matches:
            fmax_values.append(float(fmax_matches[-1]))

        if resources is None:
            resources = {}
            for btype, label in [("wc", "DPEs"), ("clb", "CLBs"),
                                  ("dsp_top", "DSPs"), ("memory", "BRAMs")]:
                matches = re.findall(rf'(\d+)\s+blocks of type: {btype}', txt)
                if matches:
                    resources[label] = int(matches[0])
                else:
                    resources[label] = 0

    if not fmax_values:
        return None
    return {
        "fmax_avg": sum(fmax_values) / len(fmax_values),
        "fmax_seeds": fmax_values,
        "resources_used": resources or {},
    }


def run_model_sim(arch_name, model_name, fmax):
    """Run IMC simulator for a CNN model."""
    cfg_file, R, C, tw, th = ARCH_CONFIGS[arch_name]
    cfg = Config(str(PROJECT_DIR / "azurelily" / "IMC" / "configs" / cfg_file))
    cfg.rows = R; cfg.cols = C; cfg.freq = fmax

    avail = VTR_AVAILABLE[arch_name]
    cfg.total_dsp = avail["DSPs"]
    cfg.total_clb = avail["CLBs"]
    cfg.total_mem = avail["BRAMs"]

    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)

    if model_name == "resnet":
        from models.resnet import resnet_model
        layers, _ = resnet_model(1, 1, 1, 1, False, False)
    elif model_name == "vgg":
        from models.vggnet import vgg_model
        layers, _ = vgg_model(1, 1, 1, 1, False, False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    for layer in layers:
        sched.run_layer(layer)

    total_energy = sum(stats.energy_breakdown.values())
    total_latency = sum(stats.latency_breakdown.values())
    return total_energy, total_latency


def compute_used_area(arch_name, resources_used):
    """Compute area from USED resources."""
    _, _, _, tw, th = ARCH_CONFIGS[arch_name]
    dpe_cells = resources_used.get("DPEs", 0) * tw * th
    dsp_cells = resources_used.get("DSPs", 0) * 1 * 4
    bram_cells = resources_used.get("BRAMs", 0) * 1 * 2
    clb_cells = resources_used.get("CLBs", 0) * 1 * 1
    return (dpe_cells + dsp_cells + bram_cells + clb_cells) * CLB_TILE_UM2 / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtr-dir", default="benchmarks/vtr_runs")
    args = parser.parse_args()
    vtr_dir = PROJECT_DIR / args.vtr_dir

    results = []
    print("CNN Benchmark (150×150 FPGA)")
    print("=" * 80)

    for rtl_name in ["resnet9_proposed", "resnet9_al_matched", "resnet_1_azurelily",
                      "vgg11_proposed", "vgg11_al_matched", "vgg11_1_azurelily"]:
        arch_name = RTL_ARCH_MAP[rtl_name]
        model_name = RTL_MODEL_MAP[rtl_name]
        display = RTL_DISPLAY[rtl_name]

        print(f"\n--- {display} ---")

        vtr = parse_vtr_results(vtr_dir, rtl_name)
        if vtr is None:
            print(f"  SKIPPED: No VTR results")
            continue

        fmax = vtr["fmax_avg"]
        res_used = vtr["resources_used"]
        print(f"  Fmax: {fmax:.1f} MHz (seeds: {vtr['fmax_seeds']})")
        print(f"  Used: DPEs={res_used.get('DPEs',0)}, DSPs={res_used.get('DSPs',0)}, "
              f"BRAMs={res_used.get('BRAMs',0)}, CLBs={res_used.get('CLBs',0)}")

        energy, latency = run_model_sim(arch_name, model_name, fmax)
        throughput = 1e9 / latency
        used_area = compute_used_area(arch_name, res_used)
        tput_mm2 = throughput / used_area if used_area > 0 else 0
        tput_j = 1e12 / energy if energy > 0 else 0

        print(f"  Energy: {energy:.2f} pJ, Latency: {latency:.2f} ns")
        print(f"  Throughput: {throughput:.0f} inf/s, Area: {used_area:.4f} mm²")

        model_display = "ResNet-9" if model_name == "resnet" else "VGG-11"
        results.append({
            "model": model_display,
            "arch": arch_name,
            "fmax_avg": round(fmax, 1),
            "fmax_s1": vtr["fmax_seeds"][0] if len(vtr["fmax_seeds"]) > 0 else 0,
            "fmax_s2": vtr["fmax_seeds"][1] if len(vtr["fmax_seeds"]) > 1 else 0,
            "fmax_s3": vtr["fmax_seeds"][2] if len(vtr["fmax_seeds"]) > 2 else 0,
            "dpe_used": res_used.get("DPEs", 0),
            "dsp_used": res_used.get("DSPs", 0),
            "bram_used": res_used.get("BRAMs", 0),
            "clb_used": res_used.get("CLBs", 0),
            "energy_pj": round(energy, 2),
            "latency_ns": round(latency, 2),
            "throughput_infs": round(throughput, 2),
            "used_area_mm2": round(used_area, 4),
            "throughput_per_mm2": round(tput_mm2, 2),
            "throughput_per_j": round(tput_j, 2),
        })

    if results:
        out_dir = PROJECT_DIR / "benchmarks" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "cnn_benchmark_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {csv_path}")

        # Also generate imc_benchmark_results.csv for plot_cnn_efficiency.py compatibility
        compat_path = out_dir / "imc_benchmark_results.csv"
        with open(compat_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "label", "model", "imc_config", "rows", "cols", "fmax_mhz",
                "total_energy_pj", "total_latency_ns"])
            writer.writeheader()
            for r in results:
                cfg_file, R, C, _, _ = ARCH_CONFIGS[r["arch"]]
                arch_label = {"proposed": "NL-DPE Proposed",
                              "al_like": "NL-DPE AL-Matched",
                              "azurelily": "Azure-Lily"}[r["arch"]]
                writer.writerow({
                    "label": f"{r['model']} {arch_label}",
                    "model": r["model"].lower().replace("-", ""),
                    "imc_config": cfg_file.replace(".json", ""),
                    "rows": R, "cols": C,
                    "fmax_mhz": r["fmax_avg"],
                    "total_energy_pj": r["energy_pj"],
                    "total_latency_ns": r["latency_ns"],
                })
        print(f"Saved: {compat_path}")

        # Summary
        print(f"\n{'Model':>10} {'Arch':>12} {'Fmax':>7} {'Energy(pJ)':>12} {'Lat(ns)':>10} "
              f"{'Tput':>8} {'Area':>8} {'T/mm²':>8} {'T/J':>10}")
        print("-" * 90)
        for r in results:
            print(f"{r['model']:>10} {r['arch']:>12} {r['fmax_avg']:>6.1f} "
                  f"{r['energy_pj']:>11.0f} {r['latency_ns']:>9.0f} "
                  f"{r['throughput_infs']:>7.0f} {r['used_area_mm2']:>7.3f} "
                  f"{r['throughput_per_mm2']:>7.0f} {r['throughput_per_j']:>9.0f}")


if __name__ == "__main__":
    main()
