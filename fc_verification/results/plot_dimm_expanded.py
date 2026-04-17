#!/usr/bin/env python3
"""Plot expanded DIMM breakdown — NL-DPE vs Azure-Lily, per stage.

Side-by-side stacked bars:
  Left  : NL-DPE (Crossbar-based DIMM via gemm_log + dimm_nonlinear)
  Right : Azure-Lily (DSP + CLB-based DIMM)

Reads:
  dimm_vtr_imc_results.json               — NL-DPE (existing)
  azurelily_dimm_vtr_imc_results.json     — Azure-Lily (new)

Outputs:
  dimm_architecture_comparison.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
try:
    from style_constants import apply_style
    apply_style()
except Exception:
    pass

def load(p):
    with open(p) as f:
        return json.load(f)

nl = load(SCRIPT_DIR / "dimm_vtr_imc_results.json")
al = load(SCRIPT_DIR / "azurelily_dimm_vtr_imc_results.json")

# NL-DPE grouping (matches existing plot_dimm_breakdown.py categories):
def nl_group(bd):
    return {
        "Crossbar (VMM)":   sum(v for k, v in bd.items() if "vmm" in k and not k.startswith("_")),
        "ACAM (exp/log)":   sum(v for k, v in bd.items() if "digital" in k and not k.startswith("_")),
        "CLB (add+reduce)": sum(v for k, v in bd.items() if "clb" in k),
        "DSP (MAC)":        bd.get("dsp_gemm", 0) + bd.get("mul", 0),
        "BRAM R/W":         bd.get("sram_read", 0) + bd.get("sram_write", 0),
    }

def al_group(bd):
    # Azure-Lily uses dsp_gemm (QK^T, SV) + CLB (softmax) + ADC conversions
    return {
        "Crossbar (VMM)":   0,  # not used
        "ACAM (exp/log)":   0,  # no ACAM
        "CLB (add+reduce)": sum(v for k, v in bd.items() if "clb" in k),
        "DSP (MAC)":        bd.get("dsp_gemm", 0) + bd.get("mul", 0)
                            + bd.get("imc_conversion", 0),  # ADC cost on crossbar ops
        "BRAM R/W":         bd.get("sram_read", 0) + bd.get("sram_write", 0),
    }

nl_cats = nl_group(nl["breakdown"])
al_cats = al_group(al["breakdown"])

# Keep only non-zero categories in display order
cat_order = ["Crossbar (VMM)", "ACAM (exp/log)", "CLB (add+reduce)", "DSP (MAC)", "BRAM R/W"]
cat_colors = {
    "Crossbar (VMM)":   "#7EC8E3",
    "ACAM (exp/log)":   "#FFB3BA",
    "CLB (add+reduce)": "#E9C46A",
    "DSP (MAC)":        "#A0C98A",
    "BRAM R/W":         "#C7CEEA",
}

fig, (ax_e, ax_l) = plt.subplots(1, 2, figsize=(7.5, 3.2),
    gridspec_kw={"width_ratios": [1.25, 1]})

# ─── Energy stacked bars ───
arch_data = [("Proposed\n(NL-DPE)", nl_cats, nl),
             ("Azure-Lily\n(DSP+CLB)", al_cats, al)]

xpos = [0, 1]
# Track which categories we've already put in the legend
seen_in_legend = set()
for x, (label, cats, meta) in zip(xpos, arch_data):
    bottom = 0
    total = sum(cats.values())
    for cat in cat_order:
        val = cats[cat]
        if val <= 0:
            continue
        color = cat_colors[cat]
        label_arg = cat if cat not in seen_in_legend else None
        if cat not in seen_in_legend:
            seen_in_legend.add(cat)
        ax_e.bar(x, val / 1e3, 0.55, bottom=bottom / 1e3, color=color,
                 edgecolor='white', linewidth=0.5, label=label_arg)
        pct = val / total * 100
        if pct > 6:
            ax_e.text(x, (bottom + val / 2) / 1e3, f"{pct:.0f}%",
                      ha='center', va='center', fontsize=7)
        bottom += val
    ax_e.text(x, total / 1e3 * 1.03, f"{total/1e3:.0f} nJ",
              ha='center', va='bottom', fontsize=8, fontweight='bold')

ax_e.set_ylabel("Energy (nJ)")
ax_e.set_title("DIMM Attention Energy (S=128, d=64)", fontsize=9, fontweight='bold')
ax_e.set_xticks(xpos)
ax_e.set_xticklabels([label for label, _, _ in arch_data], fontsize=7)
ax_e.legend(loc='upper right', fontsize=6, framealpha=0.9)
ax_e.set_ylim(0, max(sum(nl_cats.values()), sum(al_cats.values())) / 1e3 * 1.15)

# ─── Latency bar ───
latencies_us = [nl["latency_ns"] / 1e3, al["latency_ns"] / 1e3]
ax_l.bar(xpos, latencies_us, 0.55, color=["#9BB4D8", "#E8A87C"],
         edgecolor='white', linewidth=0.5)
for x, lat in zip(xpos, latencies_us):
    ax_l.text(x, lat, f"{lat:.0f} µs", ha='center', va='bottom', fontsize=8)
ax_l.set_ylabel("Latency (µs)")
ax_l.set_title("DIMM Attention Latency (W=16 lanes)", fontsize=9, fontweight='bold')
ax_l.set_xticks(xpos)
ax_l.set_xticklabels(["NL-DPE", "Azure-Lily"], fontsize=7)
ax_l.set_ylim(0, max(latencies_us) * 1.15)

plt.tight_layout()
out_path = SCRIPT_DIR / "dimm_architecture_comparison.pdf"
fig.savefig(out_path)
print(f"Saved: {out_path}")
print(f"  NL-DPE     : energy={nl['energy_pj']/1e3:.0f} nJ, latency={nl['latency_ns']/1e3:.0f} µs")
print(f"  Azure-Lily : energy={al['energy_pj']/1e3:.0f} nJ, latency={al['latency_ns']/1e3:.0f} µs")
