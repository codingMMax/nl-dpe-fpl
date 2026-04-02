#!/usr/bin/env python3
"""Run IMC simulator using VTR results for BERT-Tiny seq_len sweep.

Produces two CSV files:
  1. Variable Fmax: each (arch, seq_len) uses its own VTR Fmax
  2. Fixed Fmax: all seq_lens use the max Fmax from seq_len=128

Usage:
    python benchmarks/run_seqlen_imc.py
"""
import csv
import math
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

CLB_TILE_UM2 = 2239

ARCH_CONFIGS = {
    "proposed":  ("nl_dpe.json",      1024, 128, 3, 7),
    "al_like":   ("nl_dpe.json",      1024, 256, 5, 8),
    "azurelily": ("azure_lily.json",   512, 128, 6, 5),
}

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]

DIMM_LAYER_KEYWORDS = ["mac_qk", "mac_sv", "softmax_exp", "softmax_norm"]

VTR_DIR = PROJECT_DIR / "benchmarks" / "vtr_runs_seqlen"
RESULTS_DIR = PROJECT_DIR / "benchmarks" / "results"


def _functional_dpes(rows, cols):
    D_MODEL, D_FF = 128, 512
    per_block = (4 * math.ceil(D_MODEL / rows) * math.ceil(D_MODEL / cols)
                 + math.ceil(D_MODEL / rows) * math.ceil(D_FF / cols)
                 + math.ceil(D_FF / rows) * math.ceil(D_MODEL / cols))
    return per_block * 2


def parse_vtr(arch_name, seq_len):
    """Parse VTR results for one (arch, seq_len) point."""
    design_dir = VTR_DIR / f"bert_tiny_{arch_name}_s{seq_len}"
    if not design_dir.is_dir():
        return None

    fmaxes = []
    resources = None
    grid = (0, 0)

    for lp in sorted(design_dir.rglob("vpr_stdout.log")):
        txt = lp.read_text(errors="replace")
        fm = re.findall(r'Fmax:\s*([\d.]+)\s*MHz', txt)
        if not fm:
            continue
        fmaxes.append(float(fm[-1]))
        gm = re.search(r'FPGA sized to (\d+) x (\d+)', txt)
        if gm:
            grid = (int(gm.group(1)), int(gm.group(2)))
        if resources is None:
            res = {}
            ctx = None
            for line in txt.splitlines():
                if "Netlist" in line and "block" not in line:
                    ctx = "net"
                elif "Architecture" in line and "block" not in line:
                    ctx = "arch"
                elif ctx == "net":
                    m = re.match(r'\s+(\d+)\s+blocks of type:\s+(\w+)', line)
                    if m:
                        res[m.group(2)] = int(m.group(1))
            if res:
                resources = res

    if not fmaxes or not resources:
        return None

    return {
        "fmax_avg": sum(fmaxes) / len(fmaxes),
        "fmax_seeds": fmaxes,
        "grid_w": grid[0],
        "grid_h": grid[1],
        "dpe_used": resources.get("wc", 0),
        "dsp_used": resources.get("dsp_top", 0),
        "bram_used": resources.get("memory", 0),
        "clb_used": resources.get("clb", 0),
    }


def compute_used_area(arch_name, vtr):
    """Area from used resources only."""
    _, _, _, tw, th = ARCH_CONFIGS[arch_name]
    dpe_cells = vtr["dpe_used"] * tw * th
    dsp_cells = vtr["dsp_used"] * 1 * 4
    bram_cells = vtr["bram_used"] * 1 * 2
    clb_cells = vtr["clb_used"] * 1 * 1
    return (dpe_cells + dsp_cells + bram_cells + clb_cells) * CLB_TILE_UM2 / 1e6


def run_bert_sim(arch_name, fmax, vtr_resources, seq_len):
    """Run IMC simulator for one (arch, seq_len, fmax) point."""
    cfg_file, R, C, tw, th = ARCH_CONFIGS[arch_name]
    cfg_path = str(PROJECT_DIR / "azurelily" / "IMC" / "configs" / cfg_file)

    cfg = Config(cfg_path)
    cfg.rows = R
    cfg.cols = C
    cfg.freq = fmax
    cfg.total_dsp = vtr_resources["dsp_used"]
    cfg.total_clb = vtr_resources["clb_used"]
    cfg.total_mem = vtr_resources["bram_used"]
    cfg.total_dimm_dpes = max(0, vtr_resources["dpe_used"] - _functional_dpes(R, C))

    stats = Stats()
    mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)

    md, _ = bert_tiny_model(1, 1, seq_len, 128, False, False)

    # Track energy per operation group using snapshot deltas
    def _snap():
        return sum(stats.energy_breakdown.values())

    def _run_group(layers):
        e0 = _snap()
        for layer in layers:
            sched.run_layer(layer)
        return _snap() - e0

    group_energy = {"dimm": 0, "proj_ffn": 0, "other": 0}

    # Embedding + LN → Other
    group_energy["other"] += _run_group(md["embedding"])

    for block in md["blocks"]:
        # Q/K/V projections → Proj
        group_energy["proj_ffn"] += _run_group(block["qkv_proj"])

        # Per-head attention: mac_qk, softmax_exp, softmax_norm, mac_sv → DIMM
        for hi in range(block["num_heads"]):
            group_energy["dimm"] += _run_group(block["head_attention"])

        # O projection → Proj, residual+LN → Other
        group_energy["proj_ffn"] += _run_group([block["post_attn"][0]])
        group_energy["other"] += _run_group(block["post_attn"][1:])

        # FFN1+FFN2 → Proj+FFN, residual+LN → Other
        group_energy["proj_ffn"] += _run_group(block["ffn"][:2])
        group_energy["other"] += _run_group(block["ffn"][2:])

    total_energy = sum(stats.energy_breakdown.values())
    total_latency = sum(stats.latency_breakdown.values())

    # DIMM vs non-DIMM latency split
    dimm_latency = 0
    non_dimm_latency = 0
    for layer_name, lat in stats.latency_raw.items():
        if any(kw in layer_name for kw in DIMM_LAYER_KEYWORDS):
            dimm_latency += lat
        else:
            non_dimm_latency += lat

    # Per-resource energy breakdown
    DPE_KEYS = {"imc_vmm", "imc_conversion", "imc_digital_post",
                "imc_dimm_exp", "imc_dimm_log"}
    dpe_energy = sum(stats.energy_breakdown.get(k, 0) for k in DPE_KEYS)
    dsp_energy = stats.energy_breakdown.get("dsp_gemm", 0) + stats.energy_breakdown.get("dsp_add", 0)
    mem_energy = stats.energy_breakdown.get("sram_read", 0) + stats.energy_breakdown.get("sram_write", 0)
    clb_energy = total_energy - dpe_energy - dsp_energy - mem_energy

    return {
        "energy_pj": total_energy,
        "latency_ns": total_latency,
        "latency_dimm_ns": dimm_latency,
        "latency_non_dimm_ns": non_dimm_latency,
        # Per-resource breakdown
        "energy_dpe_pj": dpe_energy,
        "energy_dsp_pj": dsp_energy,
        "energy_mem_pj": mem_energy,
        "energy_clb_pj": clb_energy,
        # Per-operation breakdown
        "energy_dimm_pj": group_energy["dimm"],
        "energy_proj_ffn_pj": group_energy["proj_ffn"],
        "energy_other_pj": group_energy["other"],
    }


def run_sweep(use_fixed_fmax=False):
    """Run full sweep. Returns list of result rows."""
    # First pass: collect all VTR results and find max Fmax per arch
    vtr_cache = {}
    max_fmax = {}

    for arch_name in ARCH_CONFIGS:
        for S in SEQ_LENS:
            vtr = parse_vtr(arch_name, S)
            if vtr is None:
                print(f"  SKIP {arch_name} s{S}: no VTR results")
                continue
            vtr_cache[(arch_name, S)] = vtr
            if arch_name not in max_fmax or vtr["fmax_avg"] > max_fmax[arch_name]:
                max_fmax[arch_name] = vtr["fmax_avg"]

    if use_fixed_fmax:
        print(f"Fixed Fmax mode: {', '.join(f'{k}={v:.1f}MHz' for k, v in max_fmax.items())}")

    rows = []
    for arch_name in ["proposed", "al_like", "azurelily"]:
        for S in SEQ_LENS:
            if (arch_name, S) not in vtr_cache:
                continue

            vtr = vtr_cache[(arch_name, S)]
            fmax = max_fmax[arch_name] if use_fixed_fmax else vtr["fmax_avg"]

            print(f"  {arch_name} s{S}: Fmax={fmax:.1f} MHz, "
                  f"DPE={vtr['dpe_used']}, DSP={vtr['dsp_used']}, "
                  f"BRAM={vtr['bram_used']}...")

            sim = run_bert_sim(arch_name, fmax, vtr, S)
            area = compute_used_area(arch_name, vtr)
            grid_area = vtr["grid_w"] * vtr["grid_h"] * CLB_TILE_UM2 / 1e6

            throughput = 1e9 / sim["latency_ns"] if sim["latency_ns"] > 0 else 0
            tput_mm2 = throughput / area if area > 0 else 0
            tput_j = 1e12 / sim["energy_pj"] if sim["energy_pj"] > 0 else 0
            dimm_pct = sim["latency_dimm_ns"] / sim["latency_ns"] * 100 if sim["latency_ns"] > 0 else 0

            rows.append({
                "arch": arch_name,
                "seq_len": S,
                "fmax_used": round(fmax, 1),
                "fmax_vtr": round(vtr["fmax_avg"], 1),
                "dpe_used": vtr["dpe_used"],
                "dsp_used": vtr["dsp_used"],
                "bram_used": vtr["bram_used"],
                "clb_used": vtr["clb_used"],
                "grid_w": vtr["grid_w"],
                "grid_h": vtr["grid_h"],
                "grid_area_mm2": round(grid_area, 2),
                "used_area_mm2": round(area, 4),
                "energy_pj": round(sim["energy_pj"], 2),
                "latency_ns": round(sim["latency_ns"], 2),
                "throughput_infs": round(throughput, 2),
                "throughput_per_mm2": round(tput_mm2, 2),
                "throughput_per_j": round(tput_j, 2),
                "latency_dimm_ns": round(sim["latency_dimm_ns"], 2),
                "latency_non_dimm_ns": round(sim["latency_non_dimm_ns"], 2),
                "dimm_pct": round(dimm_pct, 1),
                "energy_dpe_pj": round(sim["energy_dpe_pj"], 2),
                "energy_dsp_pj": round(sim["energy_dsp_pj"], 2),
                "energy_mem_pj": round(sim["energy_mem_pj"], 2),
                "energy_clb_pj": round(sim["energy_clb_pj"], 2),
                "energy_dimm_pj": round(sim["energy_dimm_pj"], 2),
                "energy_proj_ffn_pj": round(sim["energy_proj_ffn_pj"], 2),
                "energy_other_pj": round(sim["energy_other_pj"], 2),
            })

            print(f"    Energy={sim['energy_pj']/1e6:.3f} MpJ, "
                  f"Latency={sim['latency_ns']/1e3:.1f} µs, "
                  f"DIMM={dimm_pct:.0f}%, "
                  f"Tput/mm²={tput_mm2:.0f}")

    return rows


def write_csv(rows, path):
    if not rows:
        print("No results to write.")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


def print_summary(rows, label):
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")
    print(f"{'Arch':>10} {'S':>5} {'Fmax':>6} {'DPE':>5} {'DSP':>5} {'Area':>8} "
          f"{'E(MpJ)':>9} {'Lat(µs)':>9} {'DIMM%':>6} {'Tput/mm²':>10} {'Tput/J':>10}")
    print("-" * 100)
    prev = None
    for r in rows:
        if prev and prev != r["arch"]:
            print()
        prev = r["arch"]
        print(f"{r['arch']:>10} {r['seq_len']:>5} {r['fmax_used']:>5.0f} "
              f"{r['dpe_used']:>5} {r['dsp_used']:>5} {r['used_area_mm2']:>7.2f} "
              f"{r['energy_pj']/1e6:>8.3f} {r['latency_ns']/1e3:>8.1f} "
              f"{r['dimm_pct']:>5.0f}% {r['throughput_per_mm2']:>9.0f} "
              f"{r['throughput_per_j']:>9.0f}")


if __name__ == "__main__":
    print("=== Run 1: Variable Fmax (VTR-reported per design) ===\n")
    rows_var = run_sweep(use_fixed_fmax=False)
    write_csv(rows_var, RESULTS_DIR / "bert_tiny_seqlen_variable_fmax.csv")
    print_summary(rows_var, "Variable Fmax Results")

    print("\n\n=== Run 2: Fixed Fmax (max from seq_len=128) ===\n")
    rows_fix = run_sweep(use_fixed_fmax=True)
    write_csv(rows_fix, RESULTS_DIR / "bert_tiny_seqlen_fixed_fmax.csv")
    print_summary(rows_fix, "Fixed Fmax Results")
