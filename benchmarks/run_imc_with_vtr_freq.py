#!/usr/bin/env python3
"""Run IMC simulator with VTR-reported Fmax and specified DPE config.

Usage:
    python benchmarks/run_imc_with_vtr_freq.py \
        --model resnet --imc_config nl_dpe \
        --rows 1024 --cols 128 --fmax 181.6

    # Or auto-extract Fmax from VTR run.log:
    python benchmarks/run_imc_with_vtr_freq.py \
        --model resnet --imc_config nl_dpe \
        --rows 1024 --cols 128 \
        --vtr_run_log benchmarks/rtl/resnet9_proposed/run.log

    # Batch mode: run all configs for a model
    python benchmarks/run_imc_with_vtr_freq.py --batch
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "azurelily"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "azurelily" / "IMC"))

from simulator import IMC


def extract_fmax_from_run_log(run_log_path):
    """Extract averaged Fmax from VTR run.log."""
    with open(run_log_path) as f:
        text = f.read()
    # Try format: "avg_fmax=181.6 MHz"
    match = re.search(r'avg_fmax=([0-9.]+)', text)
    if match:
        return float(match.group(1))
    # Try format: "  fmax       = 181.627 MHz"
    match = re.search(r'fmax\s+=\s+([0-9.]+)\s+MHz', text)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not find fmax in {run_log_path}")


def run_imc(model_name, imc_config_name, rows, cols, fmax_mhz, seq_length=128, head_dim=128):
    """Run IMC simulator with specified config, return results dict."""
    cfg_path = str(Path(__file__).resolve().parent.parent / "azurelily" / "IMC" / "configs" / f"{imc_config_name}.json")

    # Create IMC simulator with config
    imc = IMC(cfg_path)

    # Override crossbar size and frequency
    imc.cfg.rows = rows
    imc.cfg.cols = cols
    imc.cfg.freq = fmax_mhz

    # Reinitialize components with updated config
    from imc_core.imc_core import IMCCore
    from peripherals.fpga_fabric import FPGAFabric
    from peripherals.memory import MemoryModel
    from scheduler_stats.scheduler import Scheduler

    imc.memory = MemoryModel(imc.cfg, imc.stats)
    imc.imc_core = IMCCore(imc.cfg, imc.memory, imc.stats)
    imc.fpga = FPGAFabric(imc.cfg, imc.memory, imc.stats, imc_core=imc.imc_core)
    imc.scheduler = Scheduler(imc.cfg, imc.stats, imc.imc_core, imc.fpga)

    # Load model
    if model_name == "resnet":
        from models.resnet import resnet_model
        all_layers, _ = resnet_model(1, 1, seq_length, head_dim, False, False)
    elif model_name == "vgg":
        from models.vggnet import vgg_model
        all_layers, _ = vgg_model(1, 1, seq_length, head_dim, False, False)
    elif model_name == "attention":
        from models.attention import attention_model
        all_layers, _ = attention_model(1, 1, seq_length, head_dim, False, False)
    elif model_name == "bert_tiny":
        from models.bert_tiny import bert_tiny_model
        all_layers, _ = bert_tiny_model(1, 1, seq_length, head_dim, False, False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Run through scheduler
    if model_name == "bert_tiny":
        # BERT-Tiny returns dict with embedding + blocks structure
        # Use run_bert_model traversal from test.py
        for layer in all_layers["embedding"]:
            imc.scheduler.run_layer(layer)
        for block in all_layers["blocks"]:
            num_heads = block["num_heads"]
            for layer in block["qkv_proj"]:
                imc.scheduler.run_layer(layer)
            head_latencies = {}
            for head_idx in range(num_heads):
                for layer in block["head_attention"]:
                    lat, energy = imc.scheduler.run_layer(layer)
                    if head_idx == 0:
                        head_latencies[layer.name] = lat
            # Latency correction for parallel heads
            if num_heads > 1:
                for layer_name, single_lat in head_latencies.items():
                    subtract = single_lat * (num_heads - 1)
                    imc.stats.latency_raw[layer_name] = imc.stats.latency_raw.get(layer_name, 0) - subtract
                    imc.stats.latency_stats[layer_name] = imc.stats.latency_stats.get(layer_name, 0) - subtract
            for layer in block["post_attn"]:
                imc.scheduler.run_layer(layer)
            for layer in block["ffn"]:
                imc.scheduler.run_layer(layer)
    else:
        for layer in all_layers:
            imc.scheduler.run_layer(layer)
    imc.finalize_latency_stats()

    # Collect results
    total_energy = sum(imc.energy_stats.values())
    total_latency = sum(imc.latency_stats.values())

    # Grouped breakdown
    bd = dict(imc.energy_breakdown)
    e_dpe = sum(v for k, v in bd.items() if "imc" in k and "tile" not in k)
    e_fpga = sum(v for k, v in bd.items()
                 if any(x in k for x in ["clb", "dsp", "maxpool", "residual", "activation"]))
    e_mem = sum(v for k, v in bd.items() if "sram" in k or "bram" in k)

    return {
        "model": model_name,
        "imc_config": imc_config_name,
        "rows": rows,
        "cols": cols,
        "fmax_mhz": fmax_mhz,
        "total_energy_pj": total_energy,
        "total_latency_ns": total_latency,
        "dpe_energy_pj": e_dpe,
        "fpga_energy_pj": e_fpga,
        "mem_energy_pj": e_mem,
        "dpe_pct": e_dpe / total_energy * 100 if total_energy > 0 else 0,
        "fpga_pct": e_fpga / total_energy * 100 if total_energy > 0 else 0,
        "mem_pct": e_mem / total_energy * 100 if total_energy > 0 else 0,
        "breakdown": bd,
    }


# ═══════════════════════════════════════════════════════════════════════
# Batch mode: all benchmark configs
# ═══════════════════════════════════════════════════════════════════════

BENCHMARK_CONFIGS = [
    # (label, model, imc_config, rows, cols, fmax_override)
    # Using fixed 150×150 layout VTR Fmax values directly
    ("ResNet-9 NL-DPE Proposed",    "resnet", "nl_dpe",      1024, 128, 172.4),
    ("ResNet-9 NL-DPE AL-Matched",  "resnet", "nl_dpe",      1024, 256, 160.4),
    ("ResNet-9 Azure-Lily",         "resnet", "azure_lily",    512, 128, 198.9),
    ("VGG-11 NL-DPE Proposed",      "vgg",    "nl_dpe",      1024, 128, 111.5),
    ("VGG-11 NL-DPE AL-Matched",    "vgg",    "nl_dpe",      1024, 256, 136.9),
    ("VGG-11 Azure-Lily",           "vgg",    "azure_lily",    512, 128, 146.3),
    ("BERT-Tiny NL-DPE Proposed",   "bert_tiny", "nl_dpe",    1024, 128, 141.5),
    ("BERT-Tiny NL-DPE AL-Matched", "bert_tiny", "nl_dpe",    1024, 256, 140.1),
    ("BERT-Tiny Azure-Lily",        "bert_tiny", "azure_lily",  512, 128, 124.9),
]


def run_batch():
    """Run all benchmark configs, print comparison table."""
    import csv

    results = []
    print(f"{'Label':<35} {'Config':>10} {'Fmax':>8} {'Energy(pJ)':>12} {'Lat(ns)':>14} "
          f"{'DPE%':>6} {'FPGA%':>6} {'Mem%':>6}")
    print("-" * 105)

    for label, model, imc_cfg, rows, cols, fmax_or_log in BENCHMARK_CONFIGS:
        if isinstance(fmax_or_log, (int, float)):
            fmax = float(fmax_or_log)
        else:
            fmax = extract_fmax_from_run_log(fmax_or_log)
        r = run_imc(model, imc_cfg, rows, cols, fmax)
        results.append((label, r))
        cfg_str = f"{rows}×{cols}"
        print(f"{label:<35} {cfg_str:>10} {fmax:>7.1f} {r['total_energy_pj']:>11,.0f} "
              f"{r['total_latency_ns']:>13,.0f} {r['dpe_pct']:>5.1f}% {r['fpga_pct']:>5.1f}% "
              f"{r['mem_pct']:>5.1f}%")

    # Comparison ratios
    print(f"\n{'='*80}")
    print("COMPARISON (NL-DPE vs Azure-Lily)")
    print(f"{'='*80}")

    for model in ["resnet", "vgg"]:
        model_results = {l: r for l, r in results if r["model"] == model}
        al_key = [l for l in model_results if "Azure-Lily" in l][0]
        al = model_results[al_key]

        print(f"\n  {model.upper()}:")
        for label, r in model_results.items():
            if "Azure-Lily" in label:
                continue
            e_ratio = al["total_energy_pj"] / r["total_energy_pj"]
            l_ratio = al["total_latency_ns"] / r["total_latency_ns"]
            inf_per_j_r = 1e12 / r["total_energy_pj"]
            inf_per_j_al = 1e12 / al["total_energy_pj"]
            eff_ratio = inf_per_j_r / inf_per_j_al
            print(f"    {label}:")
            print(f"      Energy: {e_ratio:.2f}× better")
            print(f"      Latency: {l_ratio:.2f}× {'faster' if l_ratio > 1 else 'slower'}")
            print(f"      Inf/J: {eff_ratio:.2f}× better")

    # Save CSV
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "imc_benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "model", "imc_config", "rows", "cols", "fmax_mhz",
                     "total_energy_pj", "total_latency_ns",
                     "dpe_energy_pj", "fpga_energy_pj", "mem_energy_pj",
                     "dpe_pct", "fpga_pct", "mem_pct"])
        for label, r in results:
            w.writerow([label, r["model"], r["imc_config"], r["rows"], r["cols"],
                        r["fmax_mhz"], r["total_energy_pj"], r["total_latency_ns"],
                        r["dpe_energy_pj"], r["fpga_energy_pj"], r["mem_energy_pj"],
                        f"{r['dpe_pct']:.1f}", f"{r['fpga_pct']:.1f}", f"{r['mem_pct']:.1f}"])
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IMC simulator with VTR Fmax")
    parser.add_argument("--batch", action="store_true", help="Run all benchmark configs")
    parser.add_argument("--model", type=str, help="Model name (resnet, vgg, attention, bert_tiny)")
    parser.add_argument("--imc_config", type=str, help="IMC config name (nl_dpe, azure_lily)")
    parser.add_argument("--rows", type=int, help="Crossbar rows")
    parser.add_argument("--cols", type=int, help="Crossbar cols")
    parser.add_argument("--fmax", type=float, help="VTR Fmax in MHz")
    parser.add_argument("--vtr_run_log", type=str, help="Path to VTR run.log (extracts avg_fmax)")
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--head_dim", type=int, default=128)
    args = parser.parse_args()

    if args.batch:
        run_batch()
    else:
        if not all([args.model, args.imc_config, args.rows, args.cols]):
            parser.error("Need --model, --imc_config, --rows, --cols (and --fmax or --vtr_run_log)")
        if args.vtr_run_log:
            fmax = extract_fmax_from_run_log(args.vtr_run_log)
        elif args.fmax:
            fmax = args.fmax
        else:
            parser.error("Need either --fmax or --vtr_run_log")

        r = run_imc(args.model, args.imc_config, args.rows, args.cols, fmax,
                     args.seq_length, args.head_dim)
        print(json.dumps(r, indent=2, default=str))
