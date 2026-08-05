#!/usr/bin/env python3
"""Operator-isolated softmax energy/throughput harness.

Runs ONLY the two softmax layers (Softmax_Exp + Softmax_Norm) through the
existing scheduler, so energy/latency are attributable to softmax alone
rather than lumped into the DIMM aggregate.

Unit of work: one N x N attention score matrix (softmax over N elements per
row, N rows) -- matching models/bert_tiny.py (d=S, N=S).
"""
import math
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "azurelily"))
sys.path.insert(0, str(PROJ / "azurelily" / "IMC"))
sys.path.insert(0, str(PROJ / "nl_dpe"))

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
import nn

from area_power import dpe_specs

CFG_DIR = PROJ / "azurelily" / "IMC" / "configs"


# arch -> (json, R, C, fmax_MHz, clb_used, dsp_used)  fmax/resources from
# benchmarks/results/bert_tiny_seqlen_fixed_fmax.csv (fixed-Fmax sweep, N=1024 row)
ARCHS = {
    "Proposed-1 NL-DPE 1024x128": ("nl_dpe.json",    1024, 128, 136.1, 708, 10),
    "Proposed-2 NL-DPE 1024x256": ("nl_dpe.json",    1024, 256, 136.6, 713, 10),
    "Azure-Lily 512x128 (ADC)":   ("azure_lily.json", 512, 128, 137.3, 533, 86),
}

SEQ = [128, 256, 512, 1024, 2048]

SOFTMAX_KEYS = ["imc_dimm_exp", "imc_dimm_log", "clb_exp",
                "clb_norm_sum", "clb_norm_inv", "clb_subtract", "mul"]


def build(arch, rescale_energy=True):
    js, R, C, fmax, clb, dsp = ARCHS[arch]
    cfg = Config(str(CFG_DIR / js))
    cfg.rows, cfg.cols = R, C
    cfg.freq = fmax
    cfg.total_clb, cfg.total_dsp = clb, dsp
    if rescale_energy and cfg.analoge_nonlinear_support:
        # gemv_dse.patch_imc_config() does this; run_seqlen_imc.py does NOT.
        s = dpe_specs(R, C, freq_ghz=cfg.core_freq_MHz / 1000.0)
        cfg.e_analoge_pj = s["e_analogue_pj"]
        cfg.e_digital_pj = s["e_digital_pj"]
    stats = Stats()
    mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    return cfg, stats, Scheduler(cfg, stats, imc, fpga)


def softmax_once(arch, N, rescale_energy=True):
    cfg, stats, sched = build(arch, rescale_energy)
    es = {}
    exp = nn.Softmax_Exp_Layer(d=N, N=N, name="sm_exp", debug=False, energy_stats=es)
    nrm = nn.Softmax_Norm_Layer(d=N, N=N, name="sm_norm", debug=False, energy_stats=es)

    e_lat, e_eng = sched.run_layer(exp)
    n_lat, n_eng = sched.run_layer(nrm)

    bd = {k: v for k, v in stats.energy_breakdown.items()
          if k in SOFTMAX_KEYS and v > 0}
    return {
        "lat_ns": e_lat + n_lat,
        "lat_exp_ns": e_lat, "lat_norm_ns": n_lat,
        "e_pj": e_eng + n_eng,
        "e_exp_pj": e_eng, "e_norm_pj": n_eng,
        "elems": N * N,
        "breakdown": bd,
    }


def main():
    rescale = "--no-rescale" not in sys.argv
    print(f"Softmax-only operator comparison  (energy rescaling: {rescale})")
    print(f"Unit of work = one N x N softmax  |  freq = per-arch VTR Fmax\n")
    hdr = (f"{'arch':<28}{'N':>6}{'E (nJ)':>11}{'lat (us)':>11}"
           f"{'pJ/elem':>10}{'Gelem/s':>10}")
    print(hdr); print("-" * len(hdr))
    store = {}
    for arch in ARCHS:
        for N in SEQ:
            r = softmax_once(arch, N, rescale)
            store[(arch, N)] = r
            print(f"{arch:<28}{N:>6}{r['e_pj']/1e3:>11.1f}{r['lat_ns']/1e3:>11.2f}"
                  f"{r['e_pj']/r['elems']:>10.4f}{r['elems']/r['lat_ns']:>10.3f}")
        print()

    print("Ratio vs Azure-Lily (>1 = NL-DPE better)")
    print(f"{'arch':<28}{'N':>6}{'energy x':>11}{'speed x':>11}")
    print("-" * 56)
    for arch in list(ARCHS)[:2]:
        for N in SEQ:
            a = store[("Azure-Lily 512x128 (ADC)", N)]
            p = store[(arch, N)]
            print(f"{arch:<28}{N:>6}{a['e_pj']/p['e_pj']:>11.2f}"
                  f"{a['lat_ns']/p['lat_ns']:>11.2f}")
        print()

    print("Energy breakdown, N=1024 (pJ):")
    for arch in ARCHS:
        r = store[(arch, 1024)]
        tot = r["e_pj"]
        parts = "  ".join(f"{k}={v/1e3:.1f}nJ({100*v/tot:.0f}%)"
                          for k, v in sorted(r["breakdown"].items(),
                                             key=lambda x: -x[1]))
        print(f"  {arch:<28} {parts}")
        print(f"  {'':<28} exp={r['e_exp_pj']/1e3:.1f}nJ / "
              f"norm={r['e_norm_pj']/1e3:.1f}nJ | "
              f"lat exp={r['lat_exp_ns']/1e3:.2f}us norm={r['lat_norm_ns']/1e3:.2f}us")


if __name__ == "__main__":
    main()
