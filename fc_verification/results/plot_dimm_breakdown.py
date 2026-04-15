#!/usr/bin/env python3
"""Plot DIMM attention energy breakdown: stacked bar by component."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from style_constants import apply_style
apply_style()

with open(SCRIPT_DIR / "dimm_vtr_imc_results.json") as f:
    r = json.load(f)

bd = r["breakdown"]

# Group into categories
categories = {
    "Crossbar (VMM)": sum(v for k, v in bd.items() if "vmm" in k and not k.startswith("_")),
    "ACAM (exp/log)": sum(v for k, v in bd.items() if "digital" in k and not k.startswith("_")),
    "CLB (add+reduce)": sum(v for k, v in bd.items() if "clb" in k),
    "BRAM R/W": bd.get("sram_read", 0) + bd.get("sram_write", 0),
}

total = r["energy_pj"]
lat = r["latency_ns"]
fmax = r["fmax_avg_mhz"]

fig, (ax_e, ax_info) = plt.subplots(1, 2, figsize=(7.16, 2.5),
    gridspec_kw={"width_ratios": [2, 1.2]})

# Energy stacked bar
colors = ['#7EC8E3', '#FFB3BA', '#E9C46A', '#C7CEEA']
labels = list(categories.keys())
values = list(categories.values())
pcts = [v / total * 100 for v in values]

bottom = 0
bars = []
for val, color, label, pct in zip(values, colors, labels, pcts):
    b = ax_e.bar(0, val / 1e3, 0.5, bottom=bottom / 1e3, color=color,
                 edgecolor='white', linewidth=0.5, label=f"{label} ({pct:.0f}%)")
    bars.append(b)
    if pct > 5:
        ax_e.text(0, (bottom + val / 2) / 1e3, f"{pct:.0f}%",
                  ha='center', va='center', fontsize=7)
    bottom += val

ax_e.set_ylabel("Energy (nJ)")
ax_e.set_title(f"DIMM Attention Energy\n(S={r['seq_len']}, d={r['d_head']}, C=128)", fontsize=8, fontweight='bold')
ax_e.set_xticks([0])
ax_e.set_xticklabels([f"Proposed\n({fmax:.0f} MHz)"], fontsize=7)
ax_e.legend(loc='upper right', fontsize=6, framealpha=0.9)
ax_e.set_ylim(0, total / 1e3 * 1.15)

# Info table
ax_info.axis('off')
info_text = (
    f"Configuration\n"
    f"  Crossbar: 1024×128\n"
    f"  dpe_buf_width: 40b\n"
    f"  d_head: {r['d_head']}, S: {r['seq_len']}\n"
    f"  K_id: {r['K_id']} (dual-identity)\n\n"
    f"VTR Results\n"
    f"  Fmax: {fmax:.1f} MHz\n"
    f"  CLB: {r['clb']}, BRAM: {r['bram']}\n"
    f"  DPE: {r['wc']}, DSP: {r['dsp']}\n\n"
    f"IMC Results\n"
    f"  Energy: {total/1e3:.1f} nJ\n"
    f"  Latency: {lat/1e3:.1f} µs\n"
    f"  Throughput: {1e9/lat:.0f} inf/s"
)
ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
             fontsize=7, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
out_path = SCRIPT_DIR / "dimm_attention_breakdown.pdf"
fig.savefig(out_path)
print(f"Saved: {out_path}")
