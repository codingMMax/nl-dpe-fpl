#!/usr/bin/env python3
"""Analytical energy + throughput for the safe-softmax study.

Standalone: reads constants from the arch config JSONs directly (no
simulator import), measured cycles from results/smoke.json, and VTR
Fmax/resources from results/vtr_softmax.json. Emits the final table
(markdown + CSV).

Energy model (spec §6, docs/superpowers/specs/2026-08-05-...-design.md):

  DPE pass (full crossbar fires regardless of fill):
      E_pass(C) = 8*e_analoge + 8*e_conv + 1*e_digital*C          [pJ]
  CLB ops: compare 0.26439 pJ, add 0.08498 pJ
      (ref_compare_pj = (793.1801e-6/3)*1e3, ref_sum_pj = 84.98358e-6*1e3,
       archive/azurelily_simulator/IMC/imc_core/config.py:130-131)
  LUT ROM access: n_clb_rom * clb_pj_per_mac * ALPHA per lookup
      (256 x 8b ROM = 2048 bit / 64 bit-per-LUT6 = 32 LUT / 8 per CLB = 4 CLB)
  DSP MAC: dsp_pj_per_mac. BRAM: bram_pj_per_access per element access.

Op counts per S x S safe softmax:
  common : S*(S-1) compares, S^2 subtract adds, S*(S-1) sum adds,
           7*S^2 BRAM element accesses (4 writes + 3 reads per element)
  AL only: S^2 exp-ROM lookups, S recip lookups, S^2 DSP multiplies
  NL only: S^2 log-domain subtract adds,
           S*ceil(S/C) exp DPE passes, ceil(S/16) log DPE passes
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
CFG_DIR = REPO / "archive" / "azurelily_simulator" / "IMC" / "configs"

ALPHA = 1.0          # LUT ROM activity factor (upper bound, reported)
N_CLB_ROM = 4        # 256x8b ROM in CLB LUTs
W = 16               # lanes

# ── area model ──────────────────────────────────────────────────────────
# Tile areas in minimum-width-transistor areas (MWTA), read from the arch
# XMLs (benchmarks/arch/*.xml <tile area=...>). Two area conventions are
# reported because they can disagree on the S=256 verdict:
#   used  : sum(count x tile MWTA), scaled to um^2 by the project's CLB-tile
#           convention (CLB tile = 2239 um^2 incl. routing, CLAUDE.md;
#           other tiles proportional by area/CLB_TILE_MWTA -- tile sizing v2)
#   grid  : device grid from VPR auto_layout x 2239 um^2 (the DSE convention,
#           CLAUDE.md "Area = grid_W x grid_H x 2239 / 1e6 [mm^2]")
TILE_MWTA = {"clb": 27905, "dsp_top": 253779, "memory": 137668}
WC_MWTA = {"proposed_auto.xml": 1379212,      # P1 1024x128, tile 3w x 7h
           "al_like_auto.xml": 2674749,       # P2 1024x256, tile 5w x 8h
           "azure_lily_auto.xml": 2320000}    # AL 512x128,  tile 6w x 5h
CLB_TILE_UM2 = 2239.0
UM2_PER_MWTA = CLB_TILE_UM2 / TILE_MWTA["clb"]


def area_mm2(resources: dict, arch_xml: str, grid: list) -> tuple[float, float]:
    """Return (used-block area, device-grid area) in mm^2."""
    mwta = (resources.get("clb", 0) * TILE_MWTA["clb"]
            + resources.get("dsp_top", 0) * TILE_MWTA["dsp_top"]
            + resources.get("memory", 0) * TILE_MWTA["memory"]
            + resources.get("wc", 0) * WC_MWTA[arch_xml])
    used = mwta * UM2_PER_MWTA / 1e6
    gw, gh = (grid + [0, 0])[:2]
    return used, gw * gh * CLB_TILE_UM2 / 1e6

# ── constants from configs ──────────────────────────────────────────────
nl = json.loads((CFG_DIR / "nl_dpe.json").read_text())
al = json.loads((CFG_DIR / "azure_lily.json").read_text())

E_ANALOGE = nl["params"]["e_analoge_pj"]          # 3.89
E_CONV = nl["params"]["e_conv_pj"]                # 0
E_DIGITAL = nl["params"]["e_digital_pj"]          # 0.171445313 per column
E_DSP = al["fpga_specs"]["dsp_pj_per_mac"]        # 1.2
E_CLB_MAC = al["fpga_specs"]["clb_pj_per_mac"]    # 0.660
E_BRAM = al["fpga_specs"]["bram_pj_per_access"]   # 0.0495
E_CLB_CMP = (793.1801e-6 / 3.0) * 1e3             # 0.26439 pJ
E_CLB_ADD = 84.98358e-6 * 1e3                     # 0.08498 pJ


def e_pass(C: int) -> float:
    return 8 * E_ANALOGE + 8 * E_CONV + 1 * E_DIGITAL * C


def energy(kind: str, S: int, C: int | None) -> dict:
    """Total pJ for one S x S safe softmax. Returns component breakdown."""
    common_cmp = S * (S - 1) * E_CLB_CMP
    common_add = (S * S + S * (S - 1)) * E_CLB_ADD     # subtract + sum tree
    bram = 7 * S * S * E_BRAM
    out = dict(clb_compare=common_cmp, clb_add=common_add, bram=bram)
    if kind == "al":
        out["lut_rom"] = (S * S + S) * N_CLB_ROM * E_CLB_MAC * ALPHA
        out["dsp"] = S * S * E_DSP
        out["dpe"] = 0.0
    else:
        out["clb_add"] += S * S * E_CLB_ADD             # log-domain subtract
        n_exp_pass = S * math.ceil(S / C)
        n_log_pass = math.ceil(S / W)
        out["dpe"] = (n_exp_pass + n_log_pass) * e_pass(C)
        out["lut_rom"] = 0.0
        out["dsp"] = 0.0
    out["total"] = sum(v for k, v in out.items())
    return out


# ── table assembly ──────────────────────────────────────────────────────
ROWS = [
    # vtr_label, smoke_label, display, RxC, kind, C (energy-charged)
    ("P1_s128",  "nl_p1_s128", "Proposed-1",  "1024x128", "nl", 128),
    ("P1_s256",  "nl_p1_s256", "Proposed-1",  "1024x128", "nl", 128),
    ("P2_s128",  "nl_p2_s128", "Proposed-2",  "1024x256", "nl", 256),
    ("P2_s256",  "nl_p2_s256", "Proposed-2",  "1024x256", "nl", 256),
    ("AL5_s128", "al5_s128",   "AL (E=5)",    "512x128",  "al", None),
    ("AL5_s256", "al5_s256",   "AL (E=5)",    "512x128",  "al", None),
    ("AL_s128",  "al_s128",    "AL (E=16)",   "512x128",  "al", None),
    ("AL_s256",  "al_s256",    "AL (E=16)",   "512x128",  "al", None),
]
# Normalization baseline: the supply-matched AL (E=5, 40 bit/cycle operand
# feed = one NL-DPE port). AL's *energy* is width-invariant, so the energy
# baseline is the same whichever AL is chosen; only area/throughput differ.
BASE_ARCH = "AL (E=5)"


def grid_from_log(vtr_label: str) -> list:
    """Fallback: read 'FPGA sized to W x H' from a kept VPR log."""
    import re
    for seed in (1, 2, 3):
        log = RESULTS / f"vtr_{vtr_label}_seed{seed}" / "vpr_stdout.log"
        if log.is_file():
            m = re.findall(r"FPGA sized to (\d+) x (\d+)",
                           log.read_text(errors="replace"))
            if m:
                return [int(m[-1][0]), int(m[-1][1])]
    return [0, 0]


def main() -> int:
    smoke = {r["label"]: r for r in
             json.loads((RESULTS / "smoke.json").read_text())}
    vtr_path = RESULTS / "vtr_softmax.json"
    vtr = {r["label"]: r for r in json.loads(vtr_path.read_text())} \
        if vtr_path.is_file() else {}

    table = []
    for vlbl, slbl, disp, rxc, kind, C in ROWS:
        S = int(vlbl.split("_s")[1])
        cyc = smoke.get(slbl, {}).get("cycles")
        v = vtr.get(vlbl, {})
        res = v.get("resources", {})
        fmax = v.get("fmax_avg_mhz")
        e = energy(kind, S, C)
        lat_us = cyc / fmax if (cyc and fmax) else None
        grid = v.get("grid") or grid_from_log(vlbl)
        a_used, a_grid = area_mm2(res, v.get("arch", ""), grid) if res else (None, None)
        thr = (1e6 / lat_us) if lat_us else None
        row = dict(
            arch=disp, rxc=rxc, S=S,
            clb=res.get("clb"), dsp=res.get("dsp_top"),
            wc=res.get("wc"), bram=res.get("memory"),
            grid=grid,
            area_used_mm2=round(a_used, 3) if a_used else None,
            area_grid_mm2=round(a_grid, 3) if a_grid else None,
            fmax_mhz=round(fmax, 2) if fmax else None,
            fmax_seeds=[round(x, 2) for x in v.get("fmax_seeds", [])],
            cycles=cyc,
            latency_us=round(lat_us, 3) if lat_us else None,
            matrices_per_s=round(thr) if thr else None,
            rows_per_s=round(S * thr) if thr else None,
            energy_pj=round(e["total"], 1),
            pj_per_element=round(e["total"] / (S * S), 4),
            dpe_pj=round(e["dpe"], 1),
            dpe_pj_per_element=round(e["dpe"] / (S * S), 4),
            # area-normalized (per spec: throughput/mm^2 and energy/mm^2)
            thr_per_mm2_used=round(thr / a_used, 1) if (thr and a_used) else None,
            thr_per_mm2_grid=round(thr / a_grid, 1) if (thr and a_grid) else None,
            energy_per_mm2_used=round(e["total"] / a_used, 1) if a_used else None,
            energy_per_mm2_grid=round(e["total"] / a_grid, 1) if a_grid else None,
            breakdown={k: round(v2, 1) for k, v2 in e.items()},
        )
        table.append(row)

    # ── normalize to the supply-matched AL at the same S ──
    for r in table:
        base = next(b for b in table
                    if b["arch"] == BASE_ARCH and b["S"] == r["S"])
        for key, norm in (("thr_per_mm2_used", "n_thr_used"),
                          ("thr_per_mm2_grid", "n_thr_grid"),
                          ("energy_per_mm2_used", "n_energy_used"),
                          ("energy_per_mm2_grid", "n_energy_grid"),
                          ("energy_pj", "n_energy_total"),
                          ("matrices_per_s", "n_throughput")):
            r[norm] = (round(r[key] / base[key], 3)
                       if (r.get(key) and base.get(key)) else None)

    (RESULTS / "softmax_table.json").write_text(json.dumps(table, indent=2))
    with (RESULTS / "softmax_table.csv").open("w", newline="") as f:
        cols = [k for k in table[0] if k not in ("breakdown", "fmax_seeds")]
        wcsv = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wcsv.writeheader()
        wcsv.writerows(table)

    # ── table 1: raw measurements ──
    lines = [
        "### Measured",
        "",
        ("| Arch | R×C | S | CLB | DSP | DPE (wc) | BRAM | Grid | "
         "Area used (mm²) | Area grid (mm²) | Fmax (MHz) | Cycles | "
         "Latency (µs) | Matrices/s | Energy (pJ) |"),
        "|" + "---|" * 15,
    ]
    for r in table:
        g = f"{r['grid'][0]}×{r['grid'][1]}" if r.get("grid") else "-"
        lines.append(
            f"| {r['arch']} | {r['rxc']} | {r['S']} | {r['clb']} | {r['dsp']} "
            f"| {r['wc']} | {r['bram']} | {g} | {r['area_used_mm2']} "
            f"| {r['area_grid_mm2']} | {r['fmax_mhz']} | {r['cycles']} "
            f"| {r['latency_us']} | {r['matrices_per_s']} | {r['energy_pj']} |")

    # ── table 2: normalized to the supply-matched AL at the same S ──
    lines += [
        "",
        f"### Normalized to {BASE_ARCH} (= 1.00 at each S)",
        "",
        "Throughput and throughput/mm²: higher is better. Energy: lower is",
        "better. AL's total energy is width-invariant (op counts don't depend",
        "on datapath width), so the energy baseline is the same for E=5 and",
        "E=16 — only area and throughput move.",
        "",
        ("| Arch | S | Throughput | Tput/mm² used | Tput/mm² grid | "
         "Total energy | Energy/mm² used | Energy/mm² grid |"),
        "|" + "---|" * 8,
    ]
    for r in table:
        lines.append(
            f"| {r['arch']} | {r['S']} | {r['n_throughput']} "
            f"| {r['n_thr_used']} | {r['n_thr_grid']} "
            f"| {r['n_energy_total']} | {r['n_energy_used']} "
            f"| {r['n_energy_grid']} |")

    md = "\n".join(lines)
    (RESULTS / "softmax_table.md").write_text(md + "\n")
    print(md)
    print(f"\nALPHA (LUT activity) = {ALPHA}; "
          f"E_pass(128) = {e_pass(128):.2f} pJ, E_pass(256) = {e_pass(256):.2f} pJ")
    print(f"Area: CLB tile = {CLB_TILE_UM2} µm², other tiles ∝ MWTA "
          f"(tile-sizing v2); grid area = grid_W × grid_H × CLB tile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
