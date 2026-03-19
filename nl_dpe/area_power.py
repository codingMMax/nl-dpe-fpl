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

    # ── VTR tile sizing (routing-aware) ──────────────────────────────────
    # Formula: DPE_logic + M*N*SB + (M+N)*CB ≈ M*N * CLB_tile
    # Rearranged: DPE_logic ≈ M*N*(CLB_tile - SB) - (M+N)*CB
    #           = M*N*1551 - (M+N)*303
    #
    # Find minimum M*N (grid cells) with overshoot < 10%.
    best_tile = None
    for w in range(1, 15):
        for h in range(w, 15):  # h >= w avoids duplicates
            capacity = w * h * (CLB_TILE_UM2 - SB_AREA_UM2) - (w + h) * CB_AREA_UM2
            if capacity >= area_um2:
                overshoot = (area_um2 - capacity) / (w * h * CLB_TILE_UM2)
                # Check total area: DPE + routing vs M*N CLB tiles
                routing = w * h * SB_AREA_UM2 + (w + h) * CB_AREA_UM2
                total_tile = area_um2 + routing
                clb_equiv = w * h * CLB_TILE_UM2
                tile_overshoot = (total_tile / clb_equiv) - 1.0
                if tile_overshoot < 0.10:  # accept up to 10% overshoot
                    if best_tile is None or w * h < best_tile[0] * best_tile[1]:
                        best_tile = (w, h, routing, total_tile, tile_overshoot)
                    break  # h is increasing, first valid h is optimal for this w

    if best_tile is None:
        # Fallback: use rough estimate
        cells_needed = math.ceil(area_um2 / (CLB_TILE_UM2 - SB_AREA_UM2))
        side = math.ceil(math.sqrt(cells_needed))
        routing = side * side * SB_AREA_UM2 + 2 * side * CB_AREA_UM2
        total_tile = area_um2 + routing
        best_tile = (side, side, routing, total_tile, 0.0)

    tile_w, tile_h = best_tile[0], best_tile[1]
    routing_um2 = best_tile[2]
    tile_total_um2 = best_tile[3]
    tile_overshoot = best_tile[4]

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
    configs = [(r, c) for r in [128, 256, 512] for c in [64, 128, 256]]

    print("NL-DPE Area, Power, and VTR Tile Sizing")
    print("=" * 95)
    print(f"{'Config':>10} {'logic_um2':>10} {'tile WxH':>8} {'cells':>5} "
          f"{'route_um2':>10} {'total_um2':>10} {'over%':>6} "
          f"{'area_MWTA':>10} {'power_mW':>9}")
    print("-" * 95)

    for r, c in configs:
        s = dpe_specs(r, c)
        print(f"{r}x{c:>3}: {s['area_um2']:>10.0f} "
              f"{s['tile_width']}x{s['tile_height']:>2}   {s['tile_cells']:>5} "
              f"{s['routing_um2']:>10.0f} {s['tile_total_um2']:>10.0f} "
              f"{s['tile_overshoot']*100:>5.1f}% "
              f"{s['area_tag_mwta']:>10.0f} {s['power_total_mw']:>9.2f}")

    print()
    print("VTR Routing Constants:")
    print(f"  SB = {SB_AREA_UM2} um^2, CB = {CB_AREA_UM2} um^2")
    print(f"  CLB_logic = {CLB_LOGIC_UM2} um^2, CLB_tile = {CLB_TILE_UM2} um^2")
    print(f"  1 MWTA = {MWTA_UM2} um^2")
    print()

    # Print energy params for baseline 256x256
    s256 = dpe_specs(256, 256)
    print("Energy params (256x256 baseline, freq=1GHz):")
    print(f"  e_analogue_pj = {s256['e_analogue_pj']:.6f}")
    print(f"  e_digital_pj  = {s256['e_digital_pj']:.9f}")
    print(f"  e_conv_pj     = {s256['e_conv_pj']}")
