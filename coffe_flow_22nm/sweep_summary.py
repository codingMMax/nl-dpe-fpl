#!/usr/bin/env python3
"""sweep_summary.py [fullr|bw] — aggregate a dpe sweep into sweep_summary[_full].csv/.md.

Models:
  fullr (default) — row-parallel interface: one data pin per crossbar row
      n_in = R + 7 controls; n_out = C + 5 (status; inert); load/drain = 8/8 cycles (int8 bit-serial)
      runs: work/dpe_full_R<R>_C64   ->   sweep_summary_full.csv / sweep_summary_full.md
  bw — proportional-bandwidth family
      B_in = 40*R/128 (+7); B_out = 40*C/64 (+5); load/drain = 26/13 cycles
      runs: work/dpe_R<R>_C64        ->   sweep_summary.csv / sweep_summary.md

One measured run per R (representative C=64) covers every C because num_gen_outputs is never
read by COFFE. Unit costs are derived from each run's own sizing_results_final.txt (no assumed
transistor sizes); geometry and area closures are asserted per run. Logic area: nl_dpe/area_power.py.
"""

import csv
import math
import os
import re
import sys

FLOW = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(FLOW)
sys.path.insert(0, os.path.join(REPO, "nl_dpe"))
from area_power import dpe_specs  # noqa: E402

MWTA_NM2 = 33864.0


def A_p(w):
    return 0.447 + 0.128 * w + 0.391 * math.sqrt(w)     # pass/rest [MWTA]


def A_i(w):
    return 0.518 + 0.127 * w + 0.428 * math.sqrt(w)     # inverter [MWTA]


MODEL = sys.argv[1] if len(sys.argv) > 1 else "fullr"
if MODEL == "fullr":
    R_LIST = [64, 128, 256, 512]
    C_LIST = [64, 128, 256]
    CTRL, OUT_FIX = 7, 5
    N_IN = lambda R: R + CTRL
    N_OUT = lambda C: C + OUT_FIX
    LOAD = lambda R: int(math.ceil(8.0 * R / R))        # 8 (int8 bit-serial)
    DRAIN = lambda C: int(math.ceil(8.0 * C / C))       # 8
    RUNDIR = lambda R: os.path.join(FLOW, "work", "dpe_full_R%d_C64" % R)
    MUX = lambda R: "dpe_full_r%d_local_mux" % R
    OUT_BASE = "sweep_summary_full"
    MODEL_DESC = ("Row-parallel interface (one data pin per crossbar row): "
                  "`num_gen_inputs = R + 7` (7 fixed controls), `num_gen_outputs = C + 5` "
                  "(status; inert — COFFE never reads it, so one measured run per R covers every C). "
                  "Load/drain = `ceil(8R/R)`/`ceil(8C/C)` = **8/8 cycles** (int8 bit-serial). "
                  "Runs: `work/dpe_full_R{64,128,256,512}_C64` (`run_dpe.sh <R>`).")
elif MODEL == "bw":
    R_LIST = [128, 256, 512, 1024]
    C_LIST = [64, 128, 256]
    CTRL, OUT_FIX = 7, 5
    N_IN = lambda R: 40 * R // 128 + CTRL
    N_OUT = lambda C: 40 * C // 64 + OUT_FIX
    LOAD = lambda R: int(math.ceil(8.0 * R / (40 * R // 128)))     # 26
    DRAIN = lambda C: int(math.ceil(8.0 * C / (40 * C // 64)))     # 13
    RUNDIR = lambda R: os.path.join(FLOW, "work", "dpe_R%d_C64" % R)
    MUX = lambda R: "dpe_r%d_local_mux" % R
    OUT_BASE = "sweep_summary"
    MODEL_DESC = ("Proportional-bandwidth family: `B_in = 40·R/128` (+7 controls), "
                  "`B_out = 40·C/64` (+5 status; inert), load/drain = 26/13 cycles. "
                  "Runs: `work/dpe_R{128,256,512,1024}_C64` (`run_dpe.sh <R> bw`).")
else:
    sys.exit("usage: sweep_summary.py [fullr|bw]")


def parse_report(path, R):
    text = open(path).read()
    det = text[text.find("DETAILS OF HARD BLOCK"):]
    det = det[:det.find("\n|----")] if "\n|----" in det else det
    hbi = text[text.find("HARDBLOCK INFORMATION"):]
    hbi = hbi[:hbi.find("VPR DELAYS")]

    def grab(p):
        m = re.search(p, det)
        return int(m.group(1)) if m else None

    r = {
        "required": grab(r"Required MUX size:\s*(\d+):1"),
        "implemented": grab(r"Implemented MUX size:\s*(\d+):1"),
        "L1": grab(r"Level 1 size =\s*(\d+)"),
        "L2": grab(r"Level 2 size =\s*(\d+)"),
        "muxes": grab(r"Number of MUXes per tile:\s*(\d+)"),
        "sram_per_mux": grab(r"Number of SRAM cells per MUX:\s*(\d+)"),
        "bare": float(re.search(r"Local mux area:\s*([0-9.eE+-]+)", hbi).group(1)),
        "total": float(re.search(r"Local mux area with sram:\s*([0-9.eE+-]+)", hbi).group(1)),
    }

    # gates: geometry formulas + SRAM closure (model-independent)
    n = r["muxes"]
    assert n == N_IN(R), (path, n, N_IN(R))
    assert r["required"] == math.ceil(n * 0.5), path
    assert r["L2"] == int(math.sqrt(r["required"])), path
    assert r["L1"] == math.ceil(r["required"] / r["L2"]), path
    assert r["implemented"] == r["L1"] * r["L2"], path
    assert r["sram_per_mux"] == r["L1"] + r["L2"], path
    cell = 4 * MWTA_NM2 / 1e6
    assert abs((r["total"] - r["bare"]) - n * r["sram_per_mux"] * cell) < 1e-6, path
    return r


def read_sizes(path):
    d = {}
    for line in open(path):
        line = line.split("#")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                d[k.strip()] = float(v)
            except ValueError:
                pass
    return d


def unit_costs(R, r):
    """Derive the area-counted transistor sizes/unit costs from the run's own sizing results."""
    s = read_sizes(os.path.join(RUNDIR(R), "sizing_results_final.txt"))
    mux = MUX(R)
    w_l1 = s["ptran_%s_L1_nmos" % mux]
    w_l2 = s["ptran_%s_L2_nmos" % mux]
    w_rest = s["rest_%s_pmos" % mux]
    w_i1n = s["inv_%s_1_nmos" % mux]
    w_i1p = s["inv_%s_1_pmos" % mux]
    per_mux = (r["implemented"] * A_p(w_l1) + r["L2"] * A_p(w_l2)
               + A_p(w_rest) + A_i(w_i1n) + A_i(w_i1p))
    assert abs(r["bare"] - r["muxes"] * per_mux * MWTA_NM2 / 1e6) < 1e-6, (R, r["bare"], per_mux)
    return {
        "l1_w": w_l1, "l2_w": w_l2, "rest_w": w_rest, "inv1_n_w": w_i1n, "inv1_p_w": w_i1p,
        "pass_unit_um2": round(A_p(w_l1) * MWTA_NM2 / 1e6, 6),
        "inv1_unit_um2": round((A_i(w_i1n) + A_i(w_i1p)) * MWTA_NM2 / 1e6, 6),
        "per_mux_um2": round(per_mux * MWTA_NM2 / 1e6, 6),
    }


def main():
    runs, costs = {}, {}
    for R in R_LIST:
        p = os.path.join(RUNDIR(R), "report.txt")
        runs[R] = parse_report(p, R)
        costs[R] = unit_costs(R, runs[R])
        print("parsed R=%-4d  bare=%9.4f  total=%9.4f um2  (gates OK; sizes L1=%g L2=%g rest=%g inv1=%g/%g)"
              % (R, runs[R]["bare"], runs[R]["total"], costs[R]["l1_w"], costs[R]["l2_w"],
                 costs[R]["rest_w"], costs[R]["inv1_n_w"], costs[R]["inv1_p_w"]))

    rows = []
    for R in R_LIST:
        r, u = runs[R], costs[R]
        sram = r["total"] - r["bare"]
        cp = r["muxes"] * r["implemented"]
        for C in [c for c in C_LIST if c <= R]:      # DSE constraint: C <= R
            s = dpe_specs(R, C)
            logic = s["area_um2"]
            rows.append({
                "R": R, "C": C,
                "n_in": N_IN(R), "n_out": N_OUT(C),
                "load_cycles": LOAD(R), "drain_cycles": DRAIN(C),
                "mux_required": r["required"], "mux_implemented": r["implemented"],
                "L1": r["L1"], "L2": r["L2"], "sram_per_mux": r["sram_per_mux"],
                "muxes": r["muxes"], "crosspoints": cp,
                "bare_um2": round(r["bare"], 4), "sram_um2": round(sram, 4),
                "total_um2": round(r["total"], 4),
                "total_mwta": round(r["total"] * 1e6 / MWTA_NM2, 1),
                "per_pin_um2": round(r["total"] / N_IN(R), 4),
                "per_crosspoint_um2": round(r["total"] / cp, 5),
                "logic_um2": round(logic, 1),
                "routing_um2": round(s["routing_um2"], 1),
                "tile_total_um2": round(s["tile_total_um2"], 1),
                "logic_plus_xbar_um2": round(logic + r["total"], 1),
                "xbar_vs_logic_pct": round(100.0 * r["total"] / logic, 3),
                "l1_w": u["l1_w"], "l2_w": u["l2_w"], "rest_w": u["rest_w"],
                "inv1_n_w": u["inv1_n_w"], "inv1_p_w": u["inv1_p_w"],
                "pass_unit_um2": u["pass_unit_um2"], "inv1_unit_um2": u["inv1_unit_um2"],
                "per_mux_um2": u["per_mux_um2"],
                "run": "work/%s_C64" % ("dpe_full_R%d" % R if MODEL == "fullr" else "dpe_R%d" % R),
            })

    csv_path = os.path.join(FLOW, OUT_BASE + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (csv_path, len(rows)))

    md = []
    md.append("# dpe local-input-crossbar — %s sweep (COFFE `-hi`, TensorSlices-calibrated constants)\n"
              % ("row-parallel" if MODEL == "fullr" else "proportional-bandwidth"))
    md.append(MODEL_DESC + " Logic area from `nl_dpe/area_power.py`. Calibration anchor: "
              "`work/tensorslice` reproduces the published 2,771 µm² to +0.0117%%. "
              "Configs are restricted to `C <= R` (DSE constraint): **%d valid rows**.\n" % len(rows))
    md.append("| R×C | n_in | n_out | cycles | mux | bits | bare µm² | SRAM µm² | **total µm²** | "
              "per-pin | logic µm² | **xbar/logic** |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        md.append("| %d×%d | %d | %d | %d/%d | %d:1 | %d | %.1f | %.1f | **%.1f** | %.2f | %.0f | "
                  "**%.2f%%** |" % (
                      x["R"], x["C"], x["n_in"], x["n_out"], x["load_cycles"], x["drain_cycles"],
                      x["mux_implemented"], x["sram_per_mux"], x["bare_um2"], x["sram_um2"],
                      x["total_um2"], x["per_pin_um2"], x["logic_um2"], x["xbar_vs_logic_pct"]))
    md.append("")
    uniq = sorted(set((u["l1_w"], u["l2_w"], u["rest_w"], u["inv1_n_w"], u["inv1_p_w"])
                      for u in costs.values()))
    if len(uniq) == 1:
        u = costs[R_LIST[0]]
        md.append(("All %d measured runs converged to identical area-counted widths "
                   "(L1/L2 pass %g/%g, restorer %g, inv_1 %g/%g); unit costs: pass %.6f µm², "
                   "inv_1 pair %.6f µm²; bare = `n·[(impl+L2+1)·A_p + A_inv1]` MWTA exactly "
                   "(asserted per run).\n")
                  % (len(R_LIST), u["l1_w"], u["l2_w"], u["rest_w"], u["inv1_n_w"],
                     u["inv1_p_w"], u["pass_unit_um2"], u["inv1_unit_um2"]))
    else:
        md.append("Area-counted widths differ across runs (see CSV): %s. The summary derives "
                  "per-run unit costs from each `sizing_results_final.txt` (no assumption).\n"
                  % ", ".join("R%d: %s" % (R, costs[R]) for R in R_LIST))
    md.append("Note: block names are lowercase (`%s`) because HSPICE folds netlist/measure names "
              "to lowercase in `.lis` output while COFFE looks them up with the configured case.\n"
              % ("dpe_full_r<R>" if MODEL == "fullr" else "dpe_r<R>"))
    if MODEL == "fullr":
        md.append("Comparison context: the real fixed 40-bit-port block (`work/dpe_R128_C64`, 47 pins) "
                  "is 112.15 µm² = 0.92% of 128×64 logic; the proportional-bandwidth family is in "
                  "`sweep_summary.csv`/`sweep_summary.md` (0.23–12.58%).\n")
    md_path = os.path.join(FLOW, OUT_BASE + ".md")
    open(md_path, "w").write("\n".join(md))
    print("wrote %s" % md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
