#!/usr/bin/env python3
"""
Block comparison: stacked bar plots for latency and energy.

X-axis: baseline → +crossbar → +bus width → +ACAM (full proposed)
         Setup 0    Setup 3     Setup 4      Setup 5

Two panels:
  Left:  Latency (normalized to baseline, linear Y)
  Right: Energy  (normalized to baseline, broken Y-axis)

Workload: fc_2048_256 (exercises all three factors).
"""
import csv
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

CSV_PATH = SCRIPT_DIR / "block_comparison_results.csv"

# ── Workload selection (default: fc_2048_256, or pass as argv[1]) ────────
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("workload", nargs="?", default="fc_2048_256",
                     choices=["fc_512_128", "fc_2048_256"])
_args = _parser.parse_args()
WL = _args.workload
OUT_PATH = SCRIPT_DIR / f"block_comparison_stacked_{WL}.pdf"

# ── Load data ────────────────────────────────────────────────────────────
data = {}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        data[(r['setup'], r['workload'])] = r

# Accumulation path: baseline → +crossbar → +bus width → +ACAM
SETUPS = ["setup0", "setup3", "setup4", "setup5"]
LABELS = [
    "Baseline\n(512/adc/16b)",
    "+ Crossbar\n(1024/adc/16b)",
    "+ Bus Width\n(1024/adc/40b)",
    "+ ACAM\n(1024/acam/40b)",
]

# ── Extract latency components ───────────────────────────────────────────
lat_read = []
lat_compute = []
lat_outser = []
lat_reduc = []
lat_write = []

for s in SETUPS:
    r = data[(s, WL)]
    core = float(r['core_row_ns'])
    out_s = float(r['output_serialize_ns'])
    lat_read.append(float(r['read_ns']))
    lat_compute.append(core - out_s)
    lat_outser.append(out_s)
    lat_reduc.append(float(r['reduction_ns']))
    lat_write.append(float(r['write_ns']))

lat_total = [sum(x) for x in zip(lat_read, lat_compute, lat_outser, lat_reduc, lat_write)]
lat_base = lat_total[0]

# Normalize to baseline
lat_read_n    = [v / lat_base for v in lat_read]
lat_compute_n = [v / lat_base for v in lat_compute]
lat_outser_n  = [v / lat_base for v in lat_outser]
lat_reduc_n   = [v / lat_base for v in lat_reduc]
lat_write_n   = [v / lat_base for v in lat_write]
lat_total_n   = [v / lat_base for v in lat_total]

# ── Extract energy components ────────────────────────────────────────────
e_crossbar = []
e_adc = []
e_acam = []
e_mem = []
e_clb = []

for s in SETUPS:
    r = data[(s, WL)]
    e_crossbar.append(float(r['e_crossbar_pj']))
    e_adc.append(float(r['e_adc_pj']))
    e_acam.append(float(r['e_acam_pj']))
    e_mem.append(float(r['e_mem_pj']))
    e_clb.append(float(r['e_fpga_pj']))

e_total = [sum(x) for x in zip(e_crossbar, e_adc, e_acam, e_mem, e_clb)]
e_base = e_total[0]

# Normalize to baseline
e_crossbar_n = [v / e_base for v in e_crossbar]
e_adc_n      = [v / e_base for v in e_adc]
e_acam_n     = [v / e_base for v in e_acam]
e_mem_n      = [v / e_base for v in e_mem]
e_clb_n      = [v / e_base for v in e_clb]
e_total_n    = [v / e_base for v in e_total]

# ── Colors ───────────────────────────────────────────────────────────────
C_READ    = '#7EC8E3'   # light blue
C_COMPUTE = '#2D6A4F'   # dark green
C_OUTSER  = '#95D5B2'   # light green
C_REDUC   = '#E9C46A'   # gold
C_WRITE   = '#F4A261'   # orange

C_XBAR    = '#7EC8E3'   # light blue (crossbar)
C_ADC     = '#FFB3BA'   # pink (ADC/ACAM — same conversion stage)
C_ACAM    = '#FFB3BA'   # pink (ADC/ACAM — same conversion stage)
C_MEM     = '#C7CEEA'   # lavender (memory)
C_CLB     = '#E9C46A'   # gold (CLB)

# ── Figure: manual layout with fig.add_axes ─────────────────────────────
fig = plt.figure(figsize=(7.16, 2.8))

x = np.arange(len(SETUPS))
bar_w = 0.55

# Layout: left half = latency, right half = energy (broken Y: bottom + top)
# Margins: left=0.07, gap=0.06, right=0.01, bottom=0.22, top=0.88
L_left   = 0.07;  L_right  = 0.48;  L_w = L_right - L_left
E_left   = 0.57;  E_right  = 0.99;  E_w = E_right - E_left
bot_y    = 0.22;  top_y    = 0.88

# Energy broken axis split: bottom 40% for proposed bar, top 60% for ADC bars
e_bot_h  = (top_y - bot_y) * 0.35
e_gap    = (top_y - bot_y) * 0.06
e_top_h  = (top_y - bot_y) - e_bot_h - e_gap

ax_lat   = fig.add_axes([L_left, bot_y, L_w, top_y - bot_y])
ax_e_bot = fig.add_axes([E_left, bot_y, E_w, e_bot_h])
ax_e_top = fig.add_axes([E_left, bot_y + e_bot_h + e_gap, E_w, e_top_h])

# ── Panel (a): Latency — linear scale ───────────────────────────────────
bottoms = np.zeros(len(SETUPS))
for vals, color, label in [
    (lat_read_n,    C_READ,    'SRAM→DPE read'),
    (lat_compute_n, C_COMPUTE, 'DPE compute'),
    (lat_outser_n,  C_OUTSER,  'DPE output ser.'),
    (lat_reduc_n,   C_REDUC,   'CLB reduction'),
    (lat_write_n,   C_WRITE,   'DPE→BRAM write'),
]:
    ax_lat.bar(x, vals, bar_w, bottom=bottoms, color=color,
               edgecolor='white', linewidth=0.5, label=label)
    bottoms += np.array(vals)

# Annotate segments with > 5% share
lat_components = [
    (lat_read_n,    lat_read,    'Read'),
    (lat_compute_n, lat_compute, 'Compute'),
    (lat_outser_n,  lat_outser,  'Out ser.'),
    (lat_reduc_n,   lat_reduc,   'Reduc.'),
    (lat_write_n,   lat_write,   'Write'),
]
lat_bottoms_ann = np.zeros(len(SETUPS))
for vals_n, vals_abs, seg_label in lat_components:
    for i in range(len(SETUPS)):
        share = vals_n[i] / lat_total_n[i] if lat_total_n[i] > 0 else 0
        if share > 0.05:
            mid_y = lat_bottoms_ann[i] + vals_n[i] / 2
            pct = share * 100
            ax_lat.text(x[i], mid_y, f'{pct:.0f}%',
                        ha='center', va='center', fontsize=5.5, color='black')
    lat_bottoms_ann += np.array(vals_n)

ax_lat.set_xticks(x)
ax_lat.set_xticklabels(LABELS, fontsize=6.5)
ax_lat.set_ylabel('Latency')
ax_lat.set_title(f'(a) Latency Breakdown — {WL}', fontsize=8, fontweight='bold')
ax_lat.set_ylim(0, max(lat_total_n) * 1.18)
ax_lat.axhline(1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax_lat.legend(loc='upper right', fontsize=6, framealpha=0.9, ncol=1)
ax_lat.grid(True, alpha=0.08, axis='y')

# ── Panel (b): Energy — broken Y-axis ───────────────────────────────────
# Adaptive break points: gap between the proposed bar and the next-smallest
e_sorted = sorted(e_total_n)
e_smallest = e_sorted[0]          # proposed bar
e_second = e_sorted[1]            # next bar
break_lo = e_smallest * 2.5       # top of bottom panel: enough room for proposed
break_hi = e_second * 0.85        # bottom of top panel: just below the next bar

ax_e_bot.set_ylim(0, break_lo)
ax_e_top.set_ylim(break_hi, max(e_total_n) * 1.10)

# Draw stacked bars on both axes
for ax in [ax_e_bot, ax_e_top]:
    bottoms_e = np.zeros(len(SETUPS))
    # Merge ADC + ACAM into one "Conversion" segment (same color)
    e_conv_n = [a + b for a, b in zip(e_adc_n, e_acam_n)]
    for vals, color, label in [
        (e_crossbar_n, C_XBAR, 'Crossbar'),
        (e_conv_n,     C_ADC,  'ADC / ACAM'),
        (e_mem_n,      C_MEM,  'BRAM R/W'),
        (e_clb_n,      C_CLB,  'CLB activation'),
    ]:
        ax.bar(x, vals, bar_w, bottom=bottoms_e, color=color,
               edgecolor='white', linewidth=0.5, label=label)
        bottoms_e += np.array(vals)

# 
d = 0.02
for sp in ['left', 'right']:
    kwargs = dict(transform=ax_e_top.transAxes, color='k', clip_on=False, linewidth=0.8)
    xpos = 0.0 if sp == 'left' else 1.0
    ax_e_top.plot((xpos - d, xpos + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_e_bot.transAxes)
    ax_e_bot.plot((xpos - d, xpos + d), (1 - d, 1 + d), **kwargs)

# X labels only on bottom
ax_e_bot.set_xticks(x)
ax_e_bot.set_xticklabels(LABELS, fontsize=6.5)

# Y label: use fig.text centered between top and bottom energy axes
# (ax_e_bot.set_ylabel would be off-center due to broken axis)
e_label_y = bot_y + (e_bot_h + e_gap + e_top_h) / 2
fig.text(E_left - 0.04, e_label_y, 'Energy', va='center', ha='center',
         rotation='vertical', fontsize=8)
ax_e_bot.set_ylabel('')  # clear default
ax_e_top.set_title(f'(b) Energy Breakdown — {WL}', fontsize=8, fontweight='bold')

# Annotate energy segments with > 5% share
e_conv_n = [a + b for a, b in zip(e_adc_n, e_acam_n)]
e_conv_abs = [a + b for a, b in zip(e_adc, e_acam)]
e_components = [
    (e_crossbar_n, e_crossbar, 'Crossbar'),
    (e_conv_n,     e_conv_abs, 'ADC/ACAM'),
    (e_mem_n,      e_mem,      'Mem'),
    (e_clb_n,      e_clb,      'CLB'),
]
e_bottoms_ann = np.zeros(len(SETUPS))
for vals_n, vals_abs, seg_label in e_components:
    for i in range(len(SETUPS)):
        share = vals_n[i] / e_total_n[i] if e_total_n[i] > 0 else 0
        if share > 0.05:
            mid_y = e_bottoms_ann[i] + vals_n[i] / 2
            pct = share * 100
            # Place on whichever axis the midpoint falls in
            if mid_y <= break_lo:
                ax_e_bot.text(x[i], mid_y, f'{pct:.0f}%',
                              ha='center', va='center', fontsize=5.5, color='black')
            elif mid_y >= break_hi:
                ax_e_top.text(x[i], mid_y, f'{pct:.0f}%',
                              ha='center', va='center', fontsize=5.5, color='black')
    e_bottoms_ann += np.array(vals_n)

ax_e_top.legend(loc='upper right', fontsize=6, framealpha=0.9, ncol=1)
ax_e_bot.grid(True, alpha=0.08, axis='y')
ax_e_top.grid(True, alpha=0.08, axis='y')

fig.savefig(OUT_PATH)
print(f"Saved to {OUT_PATH}")

# Print summary
print(f"\n{WL} — Accumulation path:")
print(f"{'Setup':30s} {'Latency':>10s} {'norm':>6s} {'Energy':>10s} {'norm':>6s}")
for i, (s, l) in enumerate(zip(SETUPS, LABELS)):
    lb = l.replace('\n', ' ')
    print(f"{lb:30s} {lat_total[i]:10.1f} {lat_total_n[i]:6.2f} {e_total[i]:10.1f} {e_total_n[i]:6.3f}")
