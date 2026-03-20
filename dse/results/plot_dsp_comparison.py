#!/usr/bin/env python3
"""DPE hard blocks vs DSP+BRAM comparison using actual VTR synthesis results.

Left panel:  Throughput/mm² comparison (DPE best vs DSP VTR results)
Right panel: Per-DPE-tile area equivalence (analytical — how many DSP+BRAM
             tiles needed to match one DPE tile's MAC parallelism)

Inputs:
    dse/results/round2_results.csv       — DPE Round 2 results (Part 1)
    dse/results/round2_dsp_comparison.csv — DSP VTR synthesis results (Part 2)

Outputs:
    dse/results/round2_dsp_comparison.pdf — 2-panel figure
"""

import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
ROUND2_CSV = RESULTS_DIR / "round2_results.csv"
DSP_CSV = RESULTS_DIR / "round2_dsp_comparison.csv"

# ── Architecture constants ─────────────────────────────────────────────
MWTA_UM2 = 0.033864
DSP_AREA_MWTA = 253_779
DSP_MACS_PER_TILE = 4
MEM_AREA_MWTA = 137_668
MEM_BYTES_PER_TILE = 2560
CLB_TILE_UM2 = 2239
FIXED_GRID = 106

# DPE configs (top-3)
DPE_CONFIGS = {
    "512x128": {"rows": 512, "cols": 128, "area_mwta": 996_777},
    "512x64":  {"rows": 512, "cols": 64,  "area_mwta": 519_307},
    "512x256": {"rows": 512, "cols": 256, "area_mwta": 1_951_716},
}

FC_WORKLOADS = {
    "fc_64_64":    (64, 64),
    "fc_128_128":  (128, 128),
    "fc_512_128":  (512, 128),
    "fc_256_512":  (256, 512),
    "fc_512_512":  (512, 512),
    "fc_2048_256": (2048, 256),
}


def compute_dpe_equivalence(config_name: str):
    """How many DSP+BRAM tiles to match one DPE tile's MAC parallelism."""
    cfg = DPE_CONFIGS[config_name]
    R, C = cfg["rows"], cfg["cols"]
    dpe_macs = R * C
    dsps_needed = math.ceil(dpe_macs / DSP_MACS_PER_TILE)
    dsp_area = dsps_needed * DSP_AREA_MWTA
    weight_bytes = R * C
    brams_needed = math.ceil(weight_bytes / MEM_BYTES_PER_TILE)
    bram_area = brams_needed * MEM_AREA_MWTA
    total_area = dsp_area + bram_area
    dpe_area = cfg["area_mwta"]
    return {
        "config": config_name,
        "dpe_area_mm2": dpe_area * MWTA_UM2 / 1e6,
        "total_dsp_bram_area_mm2": total_area * MWTA_UM2 / 1e6,
        "area_ratio": total_area / dpe_area,
        "mac_per_mwta_dpe": dpe_macs / dpe_area,
        "mac_per_mwta_dsp": dpe_macs / total_area,
    }


def main():
    # ── Load DPE Round 2 results ──────────────────────────────────────
    with open(ROUND2_CSV) as f:
        r2_rows = list(csv.DictReader(f))
    # Filter to clb_replace mode only
    dpe_rows = [r for r in r2_rows if r.get("mode", "clb_replace") == "clb_replace"]

    # ── Load DSP VTR results ──────────────────────────────────────────
    with open(DSP_CSV) as f:
        dsp_rows = list(csv.DictReader(f))

    area_mm2 = FIXED_GRID ** 2 * CLB_TILE_UM2 / 1e6

    # ── Build comparison data ────────────────────────────────────────
    comparison_rows = []
    for wl_name, (K, N) in FC_WORKLOADS.items():
        # Best DPE result for this workload
        dpe_wl = [r for r in dpe_rows if r["workload"] == wl_name]
        if not dpe_wl:
            continue
        best_dpe = max(dpe_wl, key=lambda r: float(r["throughput_per_mm2"]))

        # DSP VTR result for this workload
        dsp_wl = [r for r in dsp_rows if r["workload"] == wl_name]
        if not dsp_wl:
            continue
        dsp = dsp_wl[0]

        dpe_tput = float(best_dpe["throughput_per_mm2"])
        dsp_tput = float(dsp["throughput_per_mm2"])
        speedup = dpe_tput / dsp_tput if dsp_tput > 0 else float("inf")

        comparison_rows.append({
            "workload": wl_name,
            "K": K, "N": N,
            "dpe_config": best_dpe["config"],
            "dpe_ratio": best_dpe.get("clb_replace_ratio", ""),
            "dpe_wc_count": int(best_dpe.get("wc_count", 0)),
            "dpe_fmax_mhz": float(best_dpe["fmax_mhz"]),
            "dpe_latency_ns": float(best_dpe["latency_ns"]),
            "dpe_tput_per_mm2": dpe_tput,
            "dsp_fmax_mhz": float(dsp["fmax_mhz"]),
            "dsp_latency_ns": float(dsp["latency_ns"]),
            "dsp_tput_per_mm2": dsp_tput,
            "dsp_dsp_count": int(dsp.get("dsp_count", 0)),
            "dsp_mem_count": int(dsp.get("mem_count", 0)),
            "dsp_clb_count": int(dsp.get("clb_count", 0)),
            "dpe_speedup": speedup,
        })

    # ── Per-tile equivalence (analytical) ─────────────────────────────
    equiv_data = [compute_dpe_equivalence(c) for c in ["512x128", "512x64", "512x256"]]

    # ── Print results ────────────────────────────────────────────────
    print("=" * 70)
    print("DPE vs DSP+BRAM: VTR Synthesis Comparison")
    print("=" * 70)
    for r in comparison_rows:
        print(f"\n  {r['workload']} (K={r['K']}, N={r['N']}):")
        print(f"    DPE ({r['dpe_config']}, {r['dpe_wc_count']} tiles): "
              f"Fmax={r['dpe_fmax_mhz']:.1f} MHz, lat={r['dpe_latency_ns']:.1f} ns, "
              f"tput/mm²={r['dpe_tput_per_mm2']:.0f}")
        print(f"    DSP ({r['dsp_dsp_count']} DSP, {r['dsp_mem_count']} MEM, "
              f"{r['dsp_clb_count']} CLB): "
              f"Fmax={r['dsp_fmax_mhz']:.1f} MHz, lat={r['dsp_latency_ns']:.1f} ns, "
              f"tput/mm²={r['dsp_tput_per_mm2']:.0f}")
        print(f"    DPE speedup: {r['dpe_speedup']:.1f}×")

    geomean_speedup = math.exp(
        sum(math.log(r["dpe_speedup"]) for r in comparison_rows) / len(comparison_rows)
    )
    print(f"\nGeomean DPE speedup: {geomean_speedup:.1f}×")

    # ── Plot ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: Throughput/mm² comparison (actual VTR data) ─────────────
    wl_names = [r["workload"] for r in comparison_rows]
    wl_labels = [w.replace("fc_", "FC ").replace("_", "\u00D7") for w in wl_names]
    dpe_tputs = [r["dpe_tput_per_mm2"] for r in comparison_rows]
    dsp_tputs = [r["dsp_tput_per_mm2"] for r in comparison_rows]

    x = np.arange(len(wl_names))
    width = 0.35

    ax1.bar(x - width/2, dpe_tputs, width, label="DPE (best config)",
            color="#2563EB", edgecolor="white", linewidth=0.5)
    ax1.bar(x + width/2, dsp_tputs, width, label="DSP+BRAM (VTR)",
            color="#F59E0B", edgecolor="white", linewidth=0.5)

    ax1.set_yscale("log")
    ax1.set_ylabel("Throughput / mm\u00B2 (log scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(wl_labels, rotation=35, ha="right")
    ax1.set_title("DPE vs DSP+BRAM: Throughput/mm\u00B2", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    for i, row in enumerate(comparison_rows):
        ax1.annotate(f"{row['dpe_speedup']:.0f}\u00D7",
                     xy=(x[i], max(dpe_tputs[i], dsp_tputs[i])),
                     ha="center", va="bottom", fontsize=8, fontweight="bold",
                     color="#DC2626")

    # ── Right: Per-tile area equivalence (analytical) ─────────────────
    configs = [e["config"] for e in equiv_data]
    dpe_areas = [e["dpe_area_mm2"] * 1e3 for e in equiv_data]
    dsp_bram_areas = [e["total_dsp_bram_area_mm2"] * 1e3 for e in equiv_data]

    x2 = np.arange(len(configs))
    ax2.bar(x2 - width/2, dpe_areas, width, label="1 DPE tile",
            color="#2563EB", edgecolor="white", linewidth=0.5)
    ax2.bar(x2 + width/2, dsp_bram_areas, width,
            label="Equiv. DSP+BRAM", color="#F59E0B",
            edgecolor="white", linewidth=0.5)

    ax2.set_yscale("log")
    ax2.set_ylabel("Area (10\u207B\u00B3 mm\u00B2, log scale)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(configs)
    ax2.set_xlabel("DPE Configuration (R\u00D7C)")
    ax2.set_title("Area for Same MAC Parallelism", fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    for i, eq in enumerate(equiv_data):
        ax2.annotate(f"{eq['area_ratio']:.0f}\u00D7",
                     xy=(x2[i], dsp_bram_areas[i]),
                     ha="center", va="bottom", fontsize=8, fontweight="bold",
                     color="#DC2626")

    fig.suptitle(f"Round 2 Part 2: DPE Hard Block vs DSP+BRAM\n"
                 f"(Fixed {FIXED_GRID}\u00D7{FIXED_GRID} grid, {area_mm2:.1f} mm\u00B2, "
                 f"geomean {geomean_speedup:.0f}\u00D7 DPE advantage)",
                 fontweight="bold", fontsize=12, y=1.03)
    fig.tight_layout()

    pdf_path = RESULTS_DIR / "round2_dsp_comparison.pdf"
    fig.savefig(pdf_path)
    print(f"\nSaved: {pdf_path}")


if __name__ == "__main__":
    main()
