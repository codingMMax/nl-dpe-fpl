#!/usr/bin/env python3
"""Collect BERT-Tiny energy breakdown data across sequence lengths.

Outputs a CSV with per-N, per-arch energy for:
  - High-level groups: DIMM, Proj, FFN, Other
  - DIMM-internal split: DPE crossbar vs FPGA fabric

Saved to: paper/scripts/bert_energy_plots/bert_energy_sweep.csv
"""
import math
import sys, csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "nl_dpe"))
sys.path.insert(0, str(ROOT / "azurelily"))
sys.path.insert(0, str(ROOT / "azurelily" / "IMC"))

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
from models.bert_tiny import bert_tiny_model

OUT_CSV = Path(__file__).resolve().parent / "bert_energy_sweep.csv"

# Keys that represent DPE/IMC crossbar energy
DPE_KEYS = {"imc_vmm", "imc_conversion", "imc_digital_post",
            "imc_dimm_exp", "imc_dimm_log"}

VTR_USED = {
    "proposed":  {"DSPs": 4,   "CLBs": 453, "BRAMs": 172, "DPEs": 274},
    "azurelily": {"DSPs": 326, "CLBs": 188, "BRAMs": 16,  "DPEs": 18},
}


def _functional_dpes(rows, cols):
    """Non-DIMM DPEs: Q/K/V/O + FFN1 + FFN2, × 2 blocks."""
    D_MODEL, D_FF = 128, 512
    per_block = (4 * math.ceil(D_MODEL/rows) * math.ceil(D_MODEL/cols)
                 + math.ceil(D_MODEL/rows) * math.ceil(D_FF/cols)
                 + math.ceil(D_FF/rows) * math.ceil(D_MODEL/cols))
    return per_block * 2

CONFIGS = [
    ("Proposed",   "nl_dpe.json",     1024, 128, 135.7, "proposed"),
    ("Azure-Lily", "azure_lily.json",  512, 128,  45.3, "azurelily"),
]

SEQ_LENS = [1024, 2048, 4096, 6144, 8192]


def run_breakdown(cfg_file, R, C, fmax, N, avail_key):
    cfg = Config(str(ROOT / "azurelily" / "IMC" / "configs" / f"{cfg_file}"))
    cfg.rows = R; cfg.cols = C; cfg.freq = fmax
    avail = VTR_USED.get(avail_key, {})
    if avail.get("DSPs") is not None: cfg.total_dsp = avail["DSPs"]
    if avail.get("CLBs") is not None: cfg.total_clb = avail["CLBs"]
    if avail.get("BRAMs") is not None: cfg.total_mem = avail["BRAMs"]
    dpe_used = avail.get("DPEs", 0)
    cfg.total_dimm_dpes = max(0, dpe_used - _functional_dpes(R, C))

    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)

    md, _ = bert_tiny_model(1, 1, N, 128, False, False)
    e_groups = defaultdict(float)
    l_groups = defaultdict(float)
    dimm_dpe = 0.0
    dimm_fabric = 0.0

    def snap_e():
        return dict(stats.energy_breakdown)

    def snap_l():
        return dict(stats.latency_raw)

    def run_group(layers, group_name):
        nonlocal dimm_dpe, dimm_fabric
        eb = snap_e(); lb = snap_l()
        for layer in layers:
            sched.run_layer(layer)
        ea = snap_e(); la = snap_l()
        e_delta = {k: ea.get(k, 0) - eb.get(k, 0) for k in set(eb) | set(ea)}
        l_delta = {k: la.get(k, 0) - lb.get(k, 0) for k in set(lb) | set(la)}
        e_total = sum(e_delta.values())
        l_total = sum(l_delta.values())
        e_groups[group_name] += e_total
        l_groups[group_name] += l_total
        if group_name == "DIMM":
            dpe_e = sum(e_delta.get(k, 0) for k in DPE_KEYS)
            dimm_dpe += dpe_e
            dimm_fabric += (e_total - dpe_e)

    run_group(md["embedding"], "Other")
    for block in md["blocks"]:
        run_group(block["qkv_proj"], "Proj")
        for hi in range(block["num_heads"]):
            run_group(block["head_attention"], "DIMM")
        run_group([block["post_attn"][0]], "Proj")
        run_group(block["post_attn"][1:], "Other")
        run_group(block["ffn"][:2], "FFN")
        run_group(block["ffn"][2:], "Other")

    return {
        "DIMM": e_groups["DIMM"],
        "Proj": e_groups["Proj"],
        "FFN": e_groups["FFN"],
        "Other": e_groups["Other"],
        "DIMM_DPE": dimm_dpe,
        "DIMM_Fabric": dimm_fabric,
        "Total": sum(e_groups.values()),
        "lat_DIMM": l_groups["DIMM"],
        "lat_Proj": l_groups["Proj"],
        "lat_FFN": l_groups["FFN"],
        "lat_Other": l_groups["Other"],
        "lat_Total": sum(l_groups.values()),
    }


def main():
    rows = []
    for label, cfg_file, R, C, fmax, avail_key in CONFIGS:
        for N in SEQ_LENS:
            print(f"  Running {label} N={N}...", flush=True)
            d = run_breakdown(cfg_file, R, C, fmax, N, avail_key)
            rows.append({
                "arch": label,
                "seq_len": N,
                "total_pj": d["Total"],
                "dimm_pj": d["DIMM"],
                "proj_pj": d["Proj"],
                "ffn_pj": d["FFN"],
                "other_pj": d["Other"],
                "dimm_dpe_pj": d["DIMM_DPE"],
                "dimm_fabric_pj": d["DIMM_Fabric"],
                "total_ns": d["lat_Total"],
                "dimm_ns": d["lat_DIMM"],
                "proj_ns": d["lat_Proj"],
                "ffn_ns": d["lat_FFN"],
                "other_ns": d["lat_Other"],
            })

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
