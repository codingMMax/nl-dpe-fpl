#!/usr/bin/env python3
"""Count FPGA resources from a VTR arch XML with fixed_layout.

Usage:
    python count_resources.py                    # parse all XMLs in this dir
    python count_resources.py proposed_150x150.xml
    python count_resources.py *.xml

Reports VTR-verified ground truth when available, otherwise estimates from XML.
"""
import re
import sys
from pathlib import Path

# ── VTR-verified ground truth ──────────────────────────────────────────────
# From ResNet-9 test runs on 150x150 fixed grid, 2025-03-27.
# These are the "Architecture" block counts reported by VPR in vpr_stdout.log.
VTR_GROUND_TRUTH = {
    "proposed_150x150.xml": {
        "grid": "150x150", "tile": "3x7",
        "DPEs": 294, "DSPs": 222, "BRAMs": 518, "CLBs": 13806,
    },
    "al_like_150x150.xml": {
        "grid": "150x150", "tile": "5x8",
        "DPEs": 90, "DSPs": 222, "BRAMs": 444, "CLBs": 16528,
    },
    "azure_lily_150x150.xml": {
        "grid": "150x150", "tile": "6x5",
        "DPEs": 261, "DSPs": 333, "BRAMs": 740, "CLBs": 11262,
    },
    "baseline_150x150.xml": {
        "grid": "150x150", "tile": "0x0",
        "DPEs": 0, "DSPs": 333, "BRAMs": 740, "CLBs": 19092,
    },
}


def parse_grid_and_tile(arch_path):
    """Extract grid size and DPE tile dims from arch XML."""
    with open(arch_path) as f:
        txt = f.read()

    m = re.search(r'<fixed_layout[^>]*width="(\d+)"[^>]*height="(\d+)"', txt)
    if not m:
        return None
    grid_w, grid_h = int(m.group(1)), int(m.group(2))

    # DPE tile
    m = re.search(r'<tile name="wc"[^>]*height="(\d+)"[^>]*width="(\d+)"', txt)
    if m:
        tile_w, tile_h = int(m.group(2)), int(m.group(1))
    else:
        tile_w, tile_h = 0, 0

    return {
        'grid': f'{grid_w}x{grid_h}',
        'grid_w': grid_w, 'grid_h': grid_h,
        'total_cells': grid_w * grid_h,
        'tile_w': tile_w, 'tile_h': tile_h,
    }


def print_resources(name, grid_info, counts):
    """Print resource counts with FPGA area percentages."""
    total = counts['DPEs'] + counts['DSPs'] + counts['BRAMs'] + counts['CLBs']
    grid = grid_info['grid']
    total_cells = grid_info['total_cells']
    tw, th = grid_info['tile_w'], grid_info['tile_h']

    # Compute area in cells for each resource type
    dpe_cells = counts['DPEs'] * tw * th
    dsp_cells = counts['DSPs'] * 1 * 4   # DSP tile: 1x4
    bram_cells = counts['BRAMs'] * 1 * 2  # BRAM tile: 1x2
    clb_cells = counts['CLBs'] * 1 * 1    # CLB tile: 1x1
    accounted = dpe_cells + dsp_cells + bram_cells + clb_cells
    # Remaining cells are perimeter I/O + corners
    io_cells = total_cells - accounted

    print(f"=== {name} ({grid}, DPE tile {tw}x{th}) ===")
    print(f"  DPEs:  {counts['DPEs']:>5}  [{dpe_cells/total_cells*100:>5.1f}% area]")
    print(f"  DSPs:  {counts['DSPs']:>5}  [{dsp_cells/total_cells*100:>5.1f}% area]")
    print(f"  BRAMs: {counts['BRAMs']:>5}  [{bram_cells/total_cells*100:>5.1f}% area]")
    print(f"  CLBs:  {counts['CLBs']:>5}  [{clb_cells/total_cells*100:>5.1f}% area]")
    print(f"  I/O:   {io_cells:>5}  [{io_cells/total_cells*100:>5.1f}% area]")
    print()


def main():
    if len(sys.argv) < 2:
        script_dir = Path(__file__).resolve().parent
        files = sorted(script_dir.glob("*.xml"))
        if not files:
            print(f"No XML files found in {script_dir}")
            sys.exit(1)
    else:
        files = [Path(p) for p in sys.argv[1:]]

    for p in files:
        if not p.exists():
            print(f"File not found: {p}")
            continue

        grid_info = parse_grid_and_tile(p)
        if grid_info is None:
            print(f"{p.name}: no fixed_layout found")
            continue

        gt = VTR_GROUND_TRUTH.get(p.name)
        if gt:
            # Use ground truth tile info if available
            grid_info['tile_w'] = int(gt['tile'].split('x')[0])
            grid_info['tile_h'] = int(gt['tile'].split('x')[1])
            print_resources(p.name, grid_info, gt)
        else:
            print(f"=== {p.name} ({grid_info['grid']}) ===")
            print(f"  No VTR ground truth available. Run VTR to get accurate counts.")
            print(f"  DPE tile: {grid_info['tile_w']}x{grid_info['tile_h']}")
            print()


if __name__ == "__main__":
    main()
