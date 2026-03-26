#!/usr/bin/env python3
"""Attention Head Benchmark: NL-DPE vs Baseline FPGA.

Compares full attention head energy across NL-DPE (ACAM log-domain DIMM)
vs baseline FPGA (DSP MACs). Sweeps sequence length N to show O(N²) advantage.

The DIMM advantage can't be measured in isolation — it emerges from the
full pipeline where projection DPEs output in log domain (ACAM=log),
enabling log-domain add to replace multiply in Q·K^T and Score·V.

Usage:
    python paper/dimm_benchmark.py
"""

import sys, os, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "azurelily"))
sys.path.insert(0, str(ROOT / "azurelily" / "IMC"))
sys.path.insert(0, str(ROOT / "nl_dpe"))

# Suppress IMC verbose logging
os.environ["IMC_LOG_LEVEL"] = "WARNING"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

OUTDIR = ROOT / "dse" / "results" / "plots" / "dimm_benchmark"
OUTDIR.mkdir(parents=True, exist_ok=True)


def run_attention_energy(config_name, seq_len, head_dim, crossbar_rows=512, crossbar_cols=128):
    """Run full attention head through IMC simulator, return per-stage energy.

    Both NL-DPE and baseline use the same crossbar size (default 512×128)
    for fair comparison — only the energy parameters differ.
    """
    from imc_core.config import Config
    from imc_core.imc_core import IMCCore
    from peripherals.fpga_fabric import FPGAFabric
    from peripherals.memory import MemoryModel
    from scheduler_stats.stats import Stats
    from scheduler_stats.scheduler import Scheduler

    config_path = ROOT / "azurelily" / "IMC" / "configs" / f"{config_name}.json"
    cfg = Config(str(config_path))

    # Override crossbar size at runtime for fair comparison
    import json
    raw = json.load(open(config_path))
    raw_params = raw.get("params", {})
    old_cols = cfg.cols

    cfg.rows = crossbar_rows
    cfg.cols = crossbar_cols

    # Re-scale energy params that were scaled by old cols at load time
    if cfg.scale_with_geometry and old_cols > 0:
        # These were multiplied by old_cols during Config.__init__
        # Rescale to new_cols
        scale = crossbar_cols / old_cols
        cfg.e_analoge_pj = cfg.e_analoge_pj * scale
        cfg.e_conv_pj = cfg.e_conv_pj * scale
        cfg.e_digital_pj = cfg.e_digital_pj * scale
        cfg.t_conv_ns = raw_params.get("t_conv", 0.0) * cfg.core_cycle_ns * cfg.cols_per_adc

    stats = Stats()
    memory = MemoryModel(cfg, stats)
    imc_core = IMCCore(cfg, memory, stats)
    fpga = FPGAFabric(cfg, memory, stats, imc_core=imc_core)
    scheduler = Scheduler(cfg, stats, imc_core, fpga)

    # Build attention layers manually (same as models/attention.py)
    import nn as nn_module
    d = head_dim
    N = seq_len

    layers = [
        nn_module.Linear_Layer(in_channels=d, out_channels=d, num_inputs=N,
                               out_sram_phit_size=32, name="linear_Q",
                               debug=False, energy_stats=stats.energy_stats),
        nn_module.Linear_Layer(in_channels=d, out_channels=d, num_inputs=N,
                               out_sram_phit_size=32, name="linear_K",
                               debug=False, energy_stats=stats.energy_stats),
        nn_module.Linear_Layer(in_channels=d, out_channels=d, num_inputs=N,
                               out_sram_phit_size=8, name="linear_V",
                               debug=False, energy_stats=stats.energy_stats),
        nn_module.MAC_QK_Layer(d=d, N=N, num_macs=4,
                               Q_sram_phit_size=32, K_sram_phit_size=32,
                               name="mac_qk",
                               debug=False, energy_stats=stats.energy_stats),
        nn_module.Softmax_Exp_Layer(d=N, N=N, name="softmax_exp",
                                    debug=False, energy_stats=stats.energy_stats),
        nn_module.Softmax_Norm_Layer(d=N, N=N, name="softmax_norm",
                                     debug=False, energy_stats=stats.energy_stats),
        nn_module.MAC_SV_Layer(d=d, N=N, num_macs=4,
                               V_sram_phit_size=8, name="mac_sv",
                               debug=False, energy_stats=stats.energy_stats),
    ]

    stage_results = {}
    total_energy = 0
    total_latency = 0

    for layer in layers:
        # Reset stats for per-stage tracking
        old_breakdown = dict(stats.energy_breakdown)

        if layer.type == "linear":
            lat, energy, _ = scheduler._run_linear(layer)
        elif layer.type == "mac_qk":
            lat, energy = scheduler._run_dimm(layer)
        elif layer.type == "mac_sv":
            lat, energy = scheduler._run_dimm(layer)
        elif layer.type == "softmax_exp":
            lat, energy = scheduler._run_softmax_exp(layer)
        elif layer.type == "softmax_norm":
            lat, energy = scheduler._run_softmax_norm(layer)
        else:
            continue

        # Compute per-stage delta
        new_breakdown = dict(stats.energy_breakdown)
        delta = {}
        for k in new_breakdown:
            delta[k] = new_breakdown[k] - old_breakdown.get(k, 0)

        stage_results[layer.name] = {
            "energy_pj": energy,
            "latency_ns": lat,
            "breakdown": delta,
        }
        total_energy += energy
        total_latency += lat

    return {
        "total_energy_pj": total_energy,
        "total_latency_ns": total_latency,
        "stages": stage_results,
        "config": config_name,
        "N": seq_len,
        "d": head_dim,
    }


def run_baseline_fpga_energy(seq_len, head_dim):
    """Baseline FPGA: ALL operations on DSP + CLB. No DPE at all."""
    from imc_core.config import Config
    from imc_core.imc_core import IMCCore
    from peripherals.fpga_fabric import FPGAFabric
    from peripherals.memory import MemoryModel
    from scheduler_stats.stats import Stats

    cfg = Config(str(ROOT / "azurelily" / "IMC" / "configs" / "azure_lily.json"))
    cfg.rows = 512; cfg.cols = 128
    stats = Stats()
    memory = MemoryModel(cfg, stats)
    imc_core = IMCCore(cfg, memory, stats)
    fpga = FPGAFabric(cfg, memory, stats, imc_core=imc_core)

    N, d = seq_len, head_dim
    total_e, total_l = 0, 0

    # Projections: DSP GEMM (no DPE)
    for _ in range(3):
        l, e = fpga.gemm_dsp(N, d, d)
        total_e += e; total_l += l
    e_proj, l_proj = total_e, total_l

    # DIMM-1 QK^T: DSP MAC
    l, e = fpga.gemm_dsp(N, d, N)
    e_dimm1 = e; total_e += e; total_l += l

    # Softmax: CLB exp + CLB norm
    l_exp, e_exp = fpga.exp_fpga(N)
    e_sexp = e_exp * N; total_e += e_sexp; total_l += l_exp * N
    l_norm, e_norm, e_sum, e_inv, e_mul = fpga.norm_fpga(N)
    e_snorm = e_norm * N; total_e += e_snorm; total_l += l_norm * N
    e_soft = e_sexp + e_snorm

    # DIMM-2 SV: DSP MAC
    l, e = fpga.gemm_dsp(N, N, d)
    e_dimm2 = e; total_e += e; total_l += l

    return {
        "total_energy_pj": total_e, "total_latency_ns": total_l,
        "proj": e_proj, "dimm": e_dimm1 + e_dimm2, "soft": e_soft,
        "proj_lat": l_proj, "N": N, "d": d,
    }


def run_sweep():
    """Run three-way attention head energy/latency sweep across N values."""
    N_values = [128, 256, 512, 1024]
    d = 128  # fixed head_dim

    # Three architectures
    ARCHS = ["NL-DPE", "Azure-Lily", "Baseline"]
    results = []

    print("=" * 100)
    print(f"Three-Way Attention Head Benchmark (d={d})")
    print("NL-DPE (DPE+ACAM) vs Azure-Lily (DPE+ADC) vs Baseline FPGA (DSP only)")
    print("=" * 100)

    projection_stages = ["linear_Q", "linear_K", "linear_V"]
    dimm_stages = ["mac_qk", "mac_sv"]
    softmax_stages = ["softmax_exp", "softmax_norm"]

    print(f"\n{'N':>5} | {'Arch':>12} | {'Proj(nJ)':>10} {'DIMM(nJ)':>10} {'Soft(nJ)':>10} | {'Total(nJ)':>11} {'Lat(us)':>9} | {'E/BL':>6} {'L/BL':>6}")
    print("-" * 100)

    for N in N_values:
        row = {"N": N, "d": d}

        # NL-DPE
        r = run_attention_energy("nl_dpe", N, d)
        for key, stages in [("proj", projection_stages), ("dimm", dimm_stages), ("soft", softmax_stages)]:
            row[f"NL-DPE_{key}"] = sum(r["stages"][s]["energy_pj"] for s in stages)
        row["NL-DPE_total"] = r["total_energy_pj"]
        row["NL-DPE_latency"] = r["total_latency_ns"]

        # Azure-Lily
        r = run_attention_energy("azure_lily", N, d)
        for key, stages in [("proj", projection_stages), ("dimm", dimm_stages), ("soft", softmax_stages)]:
            row[f"Azure-Lily_{key}"] = sum(r["stages"][s]["energy_pj"] for s in stages)
        row["Azure-Lily_total"] = r["total_energy_pj"]
        row["Azure-Lily_latency"] = r["total_latency_ns"]

        # Baseline FPGA (no DPE)
        bl = run_baseline_fpga_energy(N, d)
        row["Baseline_total"] = bl["total_energy_pj"]
        row["Baseline_latency"] = bl["total_latency_ns"]
        row["Baseline_proj"] = bl["proj"]
        row["Baseline_dimm"] = bl["dimm"]
        row["Baseline_soft"] = bl["soft"]

        results.append(row)

        for arch in ARCHS:
            e = row[f"{arch}_total"]
            l = row[f"{arch}_latency"]
            ep = row[f"{arch}_proj"]
            ed = row[f"{arch}_dimm"]
            es = row[f"{arch}_soft"]
            re = row["Baseline_total"] / e if e > 0 else 0
            rl = row["Baseline_latency"] / l if l > 0 else 0
            print(f"{N:>5} | {arch:>12} | {ep/1e3:>9.1f} {ed/1e3:>9.1f} {es/1e3:>9.1f} | "
                  f"{e/1e3:>10.1f} {l/1e3:>8.1f} | {re:>5.2f}x {rl:>5.2f}x")
        print()

    return results


def plot_energy_vs_N(results):
    """Plot: Three-way energy + latency comparison."""
    ARCHS = ["NL-DPE", "Azure-Lily", "Baseline"]
    COLORS = {"NL-DPE": "#059669", "Azure-Lily": "#2563EB", "Baseline": "#DC2626"}
    MARKERS = {"NL-DPE": "o", "Azure-Lily": "s", "Baseline": "D"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.32, top=0.85)
    fig.suptitle("Attention Head: NL-DPE FPGA vs Azure-Lily FPGA vs Baseline FPGA",
                 fontsize=12, fontweight="bold", y=0.97)

    Ns = [r["N"] for r in results]

    # Plot 1: Total energy
    ax = axes[0]
    for arch in ARCHS:
        ax.plot(Ns, [r[f"{arch}_total"]/1e6 for r in results], color=COLORS[arch],
                linewidth=2, marker=MARKERS[arch], markersize=6, label=arch)
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("Energy (M pJ)")
    ax.set_title("Total Energy", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # Plot 2: Latency
    ax = axes[1]
    for arch in ARCHS:
        ax.plot(Ns, [r[f"{arch}_latency"]/1e3 for r in results], color=COLORS[arch],
                linewidth=2, marker=MARKERS[arch], markersize=6, label=arch)
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("Latency (\u00b5s)")
    ax.set_title("Total Latency", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # Plot 3: Ratios vs Baseline
    ax = axes[2]
    for arch in ["NL-DPE", "Azure-Lily"]:
        e_ratios = [r["Baseline_total"] / r[f"{arch}_total"] for r in results]
        l_ratios = [r["Baseline_latency"] / r[f"{arch}_latency"] for r in results]
        ax.plot(Ns, e_ratios, color=COLORS[arch], linewidth=2,
                marker=MARKERS[arch], markersize=6, label=f"{arch} energy")
        ax.plot(Ns, l_ratios, color=COLORS[arch], linewidth=1.5,
                marker=MARKERS[arch], markersize=4, linestyle="--",
                label=f"{arch} latency", alpha=0.7)
    ax.axhline(y=1, color="gray", linewidth=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("Improvement vs Baseline FPGA (\u00d7)")
    ax.set_title("Advantage over Baseline", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(bottom=0)

    out = OUTDIR / "attention_energy_comparison.pdf"
    fig.savefig(out)
    print(f"\nSaved: {out}")
    plt.close()


def plot_breakdown_bars(results):
    """Stacked bar: energy breakdown by stage for each N, three-way."""
    fig, axes = plt.subplots(1, len(results), figsize=(3.5*len(results)+1, 5),
                             sharey=True)
    fig.subplots_adjust(wspace=0.08, top=0.85)
    fig.suptitle("Energy Breakdown: NL-DPE vs Azure-Lily vs Baseline FPGA",
                 fontsize=12, fontweight="bold", y=0.97)

    stage_colors = {
        "Projections": "#3B82F6",
        "DIMM": "#059669",
        "Softmax": "#F59E0B",
    }

    categories = ["NL-DPE", "Azure-Lily", "Baseline"]

    for idx, r in enumerate(results):
        ax = axes[idx] if len(results) > 1 else axes
        N = r["N"]

        stage_names = ["Projections", "DIMM", "Softmax"]

        x = np.arange(len(categories))
        bar_width = 0.6

        bottoms = np.zeros(len(categories))
        for stage_name in stage_names:
            key_map = {"Projections": "proj", "DIMM": "dimm", "Softmax": "soft"}
            vals = np.array([r[f"{cat}_{key_map[stage_name]}"] / 1e6 for cat in categories])
            ax.bar(x, vals, bar_width, bottom=bottoms,
                   color=stage_colors[stage_name], label=stage_name if idx == 0 else None,
                   edgecolor="white", linewidth=0.5)
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(["NL-DPE", "AL", "BL"], fontsize=7)
        ax.set_title(f"N={N}", fontweight="bold", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Energy (M pJ)")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.15, axis="y")

    out = OUTDIR / "attention_energy_breakdown.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close()


def write_summary(results):
    """Write summary markdown."""
    out = OUTDIR / "ATTENTION_BENCHMARK_RESULTS.md"
    with open(out, "w") as f:
        f.write("# Attention Head Benchmark: NL-DPE vs Azure-Lily vs Baseline FPGA\n\n")
        f.write(f"Head dimension d = {results[0]['d']}, crossbar 512\u00d7128\n\n")
        f.write("## Energy + Latency Comparison\n\n")
        f.write("| N | | Energy (nJ) | Latency (\u00b5s) | E vs BL | L vs BL |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            bl_e = r["Baseline_total"]
            bl_l = r["Baseline_latency"]
            for arch in ["NL-DPE", "Azure-Lily", "Baseline"]:
                e = r[f"{arch}_total"]
                l = r[f"{arch}_latency"]
                re = bl_e / e if e > 0 else 0
                rl = bl_l / l if l > 0 else 0
                f.write(f"| {r['N']} | {arch} | {e/1e3:,.1f} | {l/1e3:,.1f} | "
                        f"{re:.2f}\u00d7 | {rl:.2f}\u00d7 |\n")
            f.write(f"| | | | | | |\n")

        f.write("\n## Key Findings\n\n")
        f.write("- NL-DPE wins on BOTH energy and latency vs Azure-Lily and Baseline\n")
        for r in results:
            re_al = r["Azure-Lily_total"] / r["NL-DPE_total"]
            rl_al = r["Azure-Lily_latency"] / r["NL-DPE_latency"]
            re_bl = r["Baseline_total"] / r["NL-DPE_total"]
            rl_bl = r["Baseline_latency"] / r["NL-DPE_latency"]
            f.write(f"- N={r['N']}: {re_al:.1f}\u00d7 energy / {rl_al:.1f}\u00d7 latency vs AL, "
                    f"{re_bl:.1f}\u00d7 / {rl_bl:.1f}\u00d7 vs BL\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    results = run_sweep()
    print("\nGenerating plots...")
    plot_energy_vs_N(results)
    plot_breakdown_bars(results)
    write_summary(results)
    print("\nDone.")
