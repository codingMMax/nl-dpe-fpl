"""
NL-DPE area, power, and energy model.

Provides:
  - dpe_specs(row, col): returns a dict with all DPE physical specs
  - VTR tile sizing via routing-aware formula
  - Energy parameters matching IMC simulator conventions

Usage:
  from area_power import dpe_specs
  specs = dpe_specs(256, 256)
  print(specs['tile_width'], specs['tile_height'])  # VTR tile dimensions
  print(specs['area_tag_mwta'])                      # for <tile area="...">
"""

import math


# ── VTR routing constants (from COFFE 22nm, collaborator-provided) ───────
# These are per-tile routing resource areas used to compute DPE tile dimensions.
# See: Estimating Routing resource area for DPE blocks (Archit Gajjar)
SB_AREA_UM2 = 688       # switch box area (1 per tile, retained inside hard block)
CB_AREA_UM2 = 303       # connection box area (only M+N on perimeter for hard block)
CLB_LOGIC_UM2 = 945     # CLB core logic area (= 27905 MWTA)
CLB_TILE_UM2 = CLB_LOGIC_UM2 + SB_AREA_UM2 + 2 * CB_AREA_UM2  # = 2239 um^2

MWTA_UM2 = 0.033864     # 1 MWTA = 0.033864 um^2
CLB_AREA_MWTA = 27905   # CLB core logic area in MWTA (from arch XML)


def dpe_specs(row=256, col=256, freq_ghz=1.0):
    """
    Compute all DPE physical specs for a given crossbar configuration.

    Args:
        row: number of crossbar rows
        col: number of crossbar columns
        freq_ghz: operating frequency in GHz (for energy calculation)

    Returns:
        dict with keys:
          Area:
            area_mm2          - DPE logic area in mm^2
            area_um2          - DPE logic area in um^2
            area_tag_mwta     - DPE logic area in MWTA (for XML <tile area="...">)
            area_breakdown    - dict of component areas in mm^2

          VTR tile sizing (routing-aware):
            tile_width        - VTR tile width (grid columns)
            tile_height       - VTR tile height (grid rows)
            tile_cells        - tile_width * tile_height
            routing_um2       - routing area (SBs + CBs) in um^2
            tile_total_um2    - DPE logic + routing total in um^2
            tile_overshoot    - fractional overshoot vs CLB-equivalent area

          Power:
            p_analogue_mw     - analog power (crossbar + DAC + input buffer)
            p_digital_mw      - digital power (ACAM + output buffer + XOR)
            power_total_mw    - total DPE power

          Energy (IMC simulator conventions):
            e_analogue_pj     - energy per VMM row activation
            e_digital_pj      - ACAM energy per column per cycle
            e_conv_pj         - ADC conversion energy (0 for NL-DPE)
    """
    # ── Area components (mm^2) ───────────────────────────────────────────
    crossbar_area = 0.011534 * (row / 256) * (col / 256)
    acam_area = 0.041431 / 256 * col
    buffer_size = max(row, col)  # buffer scales with larger dimension
    inbuf_area = 0.0003542 / 256 * buffer_size
    outbuf_area = 0.0003542 / 256 * buffer_size
    total_dac = col * 4
    dac_area = 0.0000782 * total_dac / 1024
    total_xors = 7 * col
    xor_area = total_xors / (7 * 256) * 0.0000989

    area_mm2 = crossbar_area + acam_area + inbuf_area + outbuf_area + dac_area + xor_area
    area_um2 = area_mm2 * 1e6
    area_tag_mwta = area_um2 / MWTA_UM2

    # ── VTR tile sizing ─────────────────────────────────────────────────
    # Hardcoded tiles for the 3 evaluation configs (from DSE + Azure-Lily paper).
    # All other configs: proportional scaling (area_MWTA / CLB_TILE_MWTA).
    FIXED_TILES = {
        (1024, 128): (3, 7),   # Proposed — from proportional scaling, DSE-validated
        (1024, 256): (5, 8),   # AL-like — from proportional scaling, DSE-validated
        (512,  128): (6, 5),   # Azure-Lily — from their published arch XML
    }

    if (row, col) in FIXED_TILES:
        tile_w, tile_h = FIXED_TILES[(row, col)]
    else:
        CLB_TILE_MWTA = CLB_TILE_UM2 / MWTA_UM2  # ~66117
        target_cells = max(1, round(area_tag_mwta / CLB_TILE_MWTA))
        best_tile = None
        for w in range(1, target_cells + 1):
            h = math.ceil(target_cells / w)
            wh = w * h
            diff = abs(w - h)
            if best_tile is None or wh < best_tile[2] or \
               (wh == best_tile[2] and diff < best_tile[3]):
                best_tile = (w, h, wh, diff)
        tile_w, tile_h = best_tile[0], best_tile[1]
    # Compute routing area (informational only — not used for tile sizing)
    routing_um2 = tile_w * tile_h * SB_AREA_UM2 + (tile_w + tile_h) * CB_AREA_UM2
    tile_total_um2 = area_um2 + routing_um2
    clb_equiv = tile_w * tile_h * CLB_TILE_UM2
    tile_overshoot = (tile_total_um2 / clb_equiv) - 1.0 if clb_equiv > 0 else 0.0

    # ── Power components (mW) ────────────────────────────────────────────
    crossbar_power = 1.31 * (row / 256) * (col / 256)
    acam_power = 43.52 / 256 * col
    inbuf_power = 0.1406 / 256 * buffer_size
    outbuf_power = 0.1406 / 256 * buffer_size
    dac_power = 2.44 * total_dac / 1024
    xor_power = total_xors / (7 * 256) * 0.235

    p_analogue = crossbar_power + dac_power + inbuf_power
    p_digital = acam_power + outbuf_power + xor_power
    power_total = p_analogue + p_digital

    # ── Energy parameters (IMC simulator conventions) ────────────────────
    # e_analogue_pj: energy for one full VMM row activation (all cols fire)
    #   Simulator: energy_per_vmm = k_slicing * e_analogue_pj
    #   → e_analogue_pj = p_analogue [mW] / freq [GHz]  (mW/GHz = pJ/cycle)
    #
    # e_digital_pj: ACAM energy *per column* per digital-post cycle
    #   Simulator (scale_with_geometry=False): energy = k_accum * e_digital_pj * cols
    #   → e_digital_pj = (p_digital [mW] / freq [GHz]) / col
    #
    # e_conv_pj: 0 for NL-DPE (ACAM absorbs ADC conversion)
    e_analogue_pj = p_analogue / freq_ghz
    e_digital_pj = (p_digital / freq_ghz) / col
    e_conv_pj = 0.0

    return {
        # Area
        "area_mm2": area_mm2,
        "area_um2": area_um2,
        "area_tag_mwta": area_tag_mwta,
        "area_breakdown": {
            "crossbar_mm2": crossbar_area,
            "acam_mm2": acam_area,
            "inbuf_mm2": inbuf_area,
            "outbuf_mm2": outbuf_area,
            "dac_mm2": dac_area,
            "xor_mm2": xor_area,
        },
        # VTR tile sizing
        "tile_width": tile_w,
        "tile_height": tile_h,
        "tile_cells": tile_w * tile_h,
        "routing_um2": routing_um2,
        "tile_total_um2": tile_total_um2,
        "tile_overshoot": tile_overshoot,
        # Power
        "p_analogue_mw": p_analogue,
        "p_digital_mw": p_digital,
        "power_total_mw": power_total,
        # Energy
        "e_analogue_pj": e_analogue_pj,
        "e_digital_pj": e_digital_pj,
        "e_conv_pj": e_conv_pj,
    }


# ── CLI: print specs for all DSE configs when run directly ───────────────
if __name__ == "__main__":
    import csv as _csv
    from pathlib import Path as _Path

    configs = [(r, c) for r in [128, 256, 512, 1024] for c in [64, 128, 256]]

    # Azure-Lily reference for area ratio
    AL_REF_CELLS = 30  # 6×5 tile

    print("NL-DPE Area, Power, and VTR Tile Sizing (proportional to Azure-Lily)")
    print("=" * 105)
    print(f"{'Config':>10} {'logic_um2':>10} {'MWTA':>10} {'tile WxH':>8} {'cells':>5} "
          f"{'tile_um2':>10} {'AL_ratio':>8} {'power_mW':>9}")
    print("-" * 105)

    csv_rows = []
    for r, c in configs:
        s = dpe_specs(r, c)
        tile_area_um2 = s['tile_cells'] * CLB_TILE_UM2
        al_ratio = s['tile_cells'] / AL_REF_CELLS
        print(f"{r}x{c:>3}: {s['area_um2']:>10.0f} {s['area_tag_mwta']:>10.0f} "
              f"{s['tile_width']}x{s['tile_height']:>2}   {s['tile_cells']:>5} "
              f"{tile_area_um2:>10.0f} {al_ratio:>7.2f}x "
              f"{s['power_total_mw']:>9.2f}")
        csv_rows.append({
            'config': f"{r}x{c}",
            'rows': r, 'cols': c,
            'logic_area_mwta': int(s['area_tag_mwta']),
            'logic_area_um2': int(s['area_um2']),
            'tile_w': s['tile_width'],
            'tile_h': s['tile_height'],
            'tile_cells': s['tile_cells'],
            'tile_area_um2': int(tile_area_um2),
            'al_area_ratio': round(al_ratio, 3),
            'power_mw': round(s['power_total_mw'], 2),
        })

    print()
    print(f"Azure-Lily reference: 512x128, tile 6x5={AL_REF_CELLS} cells, "
          f"area=2,320,000 MWTA")
    print(f"CLB_TILE = {CLB_TILE_UM2} um^2 = {CLB_TILE_UM2/MWTA_UM2:.0f} MWTA")
    print()

    # Save CSV reference table
    csv_path = _Path(__file__).resolve().parent.parent / "dse" / "results" / "config_tile_reference.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = _csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {csv_path}")

    # Print energy params for baseline 256x256
    s256 = dpe_specs(256, 256)
    print(f"\nEnergy params (256x256 baseline, freq=1GHz):")
    print(f"  e_analogue_pj = {s256['e_analogue_pj']:.6f}")
    print(f"  e_digital_pj  = {s256['e_digital_pj']:.9f}")
    print(f"  e_conv_pj     = {s256['e_conv_pj']}")
