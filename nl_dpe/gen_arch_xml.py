"""
Generate per-(R,C) VTR architecture XMLs by patching the template XML.

Four modes:
  auto                  - patch tile dims only, keep <auto_layout>
  fixed_clb_replace     - patch tile dims + replace <auto_layout> with <fixed_layout>,
                          increasing wc column density via clb_replace_ratio
  fixed_dsp_bram        - patch tile dims + replace <auto_layout> with <fixed_layout>,
                          no wc column, tune DSP/BRAM column density
  fixed_dsp_clb_replace - patch tile dims + replace <auto_layout> with <fixed_layout>,
                          simultaneously replace DSP and CLB columns with wc (DPE)

Usage:
  python gen_arch_xml.py --rows 256 --cols 128
  python gen_arch_xml.py --rows 256 --cols 128 --mode fixed_clb_replace \
      --fixed-grid 80 80 --clb-replace-ratio 0.08
  python gen_arch_xml.py --rows 256 --cols 128 --mode fixed_dsp_bram \
      --fixed-grid 80 80 --extra-dsps 4 --extra-brams 8 --pair-name all_dsp
  python gen_arch_xml.py --rows 256 --cols 128 --mode fixed_dsp_clb_replace \
      --fixed-grid 106 106 --dsp-ratio 0.6 --clb-ratio 0.4
"""

import argparse
import math
import re
from pathlib import Path

from area_power import dpe_specs

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Baseline layout constants (106×106 grid) ─────────────────────────────
# DSP columns: startx=6, repeatx=16 → positions in interior (cols 1..W-2)
BASELINE_DSP_STARTX = 6
BASELINE_DSP_REPEATX = 16
# BRAM columns: startx=2, repeatx=16
BASELINE_BRAM_STARTX = 2
BASELINE_BRAM_REPEATX = 16


def _baseline_col_positions(startx, repeatx, grid_w):
    """Enumerate column positions for a startx/repeatx pattern within grid."""
    interior = grid_w - 2  # exclude perimeter IO columns
    positions = []
    x = startx
    while x <= interior:
        positions.append(x)
        x += repeatx
    return positions


# ── Regex patterns ────────────────────────────────────────────────────────
TILE_WC_RE = re.compile(r'<tile name="wc" height="\d+" width="\d+" area="\d+">')
LAYOUT_RE = re.compile(r'<auto_layout[^>]*>.*?</auto_layout>', re.DOTALL)
# Match the entire <layout>...</layout> section (used in fixed modes to remove all
# existing layouts — auto_layout, fixed_layout, comments — and insert only the target)
FULL_LAYOUT_RE = re.compile(r'<layout>.*?</layout>', re.DOTALL)


def gen_arch_xml(
    rows: int,
    cols: int,
    template_xml: Path = SCRIPT_DIR / "nl_dpe_22nm_auto.xml",
    output_dir: Path = Path("."),
    mode: str = "auto",
    fixed_grid_w: int = None,
    fixed_grid_h: int = None,
    clb_replace_ratio: float = None,
    extra_dsps: int = None,
    extra_brams: int = None,
    pair_name: str = None,
    dsp_ratio: float = None,
    clb_ratio: float = None,
) -> Path:
    """Generate a patched architecture XML for VTR.

    Returns the Path of the generated file.
    """
    # ── Get DPE physical specs ────────────────────────────────────────────
    specs = dpe_specs(rows, cols)
    tile_w = specs["tile_width"]
    tile_h = specs["tile_height"]
    area_mwta = int(round(specs["area_tag_mwta"]))

    # ── Read template ─────────────────────────────────────────────────────
    text = template_xml.read_text()

    # ── Patch <tile name="wc" ...> line ───────────────────────────────────
    new_tile_line = (
        f'<tile name="wc" height="{tile_h}" width="{tile_w}" area="{area_mwta}">'
    )
    text = TILE_WC_RE.sub(new_tile_line, text)

    # ── Mode-specific layout patching ─────────────────────────────────────
    if mode == "auto":
        # Keep <auto_layout> unchanged
        out_name = f"nl_dpe_{rows}x{cols}_auto.xml"

    elif mode == "fixed_clb_replace":
        assert fixed_grid_w is not None and fixed_grid_h is not None, \
            "fixed_clb_replace mode requires --fixed-grid W H"
        assert clb_replace_ratio is not None, \
            "fixed_clb_replace mode requires --clb-replace-ratio"

        interior_cols = fixed_grid_w - 2
        n_dpe_cols = max(1, round(clb_replace_ratio * interior_cols / tile_w))
        wc_repeatx = max(tile_w + 1, interior_cols // n_dpe_cols)

        layout_section = (
            f'<layout>\n'
            f'    <fixed_layout name="nl_dpe_{rows}x{cols}" '
            f'width="{fixed_grid_w}" height="{fixed_grid_h}">\n'
            f'      <perimeter type="io" priority="101"/>\n'
            f'      <corners type="EMPTY" priority="102"/>\n'
            f'      <fill type="clb" priority="10"/>\n'
            f'      <col type="dsp_top" startx="6" starty="1" repeatx="16" priority="20"/>\n'
            f'      <col type="wc" startx="6" starty="1" repeatx="{wc_repeatx}" priority="15"/>\n'
            f'      <col type="memory" startx="2" starty="1" repeatx="16" priority="20"/>\n'
            f'    </fixed_layout>\n'
            f'  </layout>'
        )
        text = FULL_LAYOUT_RE.sub(layout_section, text)

        pct = int(round(clb_replace_ratio * 100))
        out_name = f"nl_dpe_{rows}x{cols}_clb{pct}_fixed.xml"

    elif mode == "fixed_dsp_bram":
        assert fixed_grid_w is not None and fixed_grid_h is not None, \
            "fixed_dsp_bram mode requires --fixed-grid W H"
        assert extra_dsps is not None and extra_brams is not None, \
            "fixed_dsp_bram mode requires --extra-dsps and --extra-brams"
        assert pair_name is not None, \
            "fixed_dsp_bram mode requires --pair-name"

        interior = fixed_grid_w - 2
        dsp_repeatx = max(1, interior // max(1, extra_dsps))
        bram_repeatx = max(1, interior // max(1, extra_brams))

        layout_section = (
            f'<layout>\n'
            f'    <fixed_layout name="nl_dpe_{rows}x{cols}" '
            f'width="{fixed_grid_w}" height="{fixed_grid_h}">\n'
            f'      <perimeter type="io" priority="101"/>\n'
            f'      <corners type="EMPTY" priority="102"/>\n'
            f'      <fill type="clb" priority="10"/>\n'
            f'      <col type="dsp_top" startx="6" starty="1" repeatx="{dsp_repeatx}" priority="20"/>\n'
            f'      <col type="memory" startx="2" starty="1" repeatx="{bram_repeatx}" priority="20"/>\n'
            f'    </fixed_layout>\n'
            f'  </layout>'
        )
        text = FULL_LAYOUT_RE.sub(layout_section, text)

        out_name = f"nl_dpe_{rows}x{cols}_{pair_name}_fixed.xml"

    elif mode == "fixed_dsp_clb_replace":
        assert fixed_grid_w is not None and fixed_grid_h is not None, \
            "fixed_dsp_clb_replace mode requires --fixed-grid W H"
        assert dsp_ratio is not None and clb_ratio is not None, \
            "fixed_dsp_clb_replace mode requires --dsp-ratio and --clb-ratio"

        # Enumerate baseline DSP column positions
        dsp_positions = _baseline_col_positions(
            BASELINE_DSP_STARTX, BASELINE_DSP_REPEATX, fixed_grid_w)
        n_dsp_cols = len(dsp_positions)

        # How many DSP columns to replace with wc
        n_dsp_remove = min(n_dsp_cols, max(0, round(dsp_ratio * n_dsp_cols)))
        # Remove rightmost first (deterministic)
        dsp_removed = sorted(dsp_positions[-n_dsp_remove:]) if n_dsp_remove > 0 else []
        dsp_kept = sorted(set(dsp_positions) - set(dsp_removed))

        # CLB replacement: wc columns in CLB space (same approach as fixed_clb_replace)
        interior_cols = fixed_grid_w - 2
        if clb_ratio > 0:
            n_clb_dpe_cols = max(1, round(clb_ratio * interior_cols / tile_w))
            wc_clb_repeatx = max(tile_w + 1, interior_cols // n_clb_dpe_cols)
        else:
            wc_clb_repeatx = None  # no CLB replacement

        # Build layout directives
        lines = []
        lines.append(f'<layout>')
        lines.append(f'    <fixed_layout name="nl_dpe_{rows}x{cols}" '
                     f'width="{fixed_grid_w}" height="{fixed_grid_h}">')
        lines.append(f'      <perimeter type="io" priority="101"/>')
        lines.append(f'      <corners type="EMPTY" priority="102"/>')
        lines.append(f'      <fill type="clb" priority="10"/>')

        # Remaining DSP columns (individual directives, repeatx > grid to place once)
        for pos in dsp_kept:
            lines.append(f'      <col type="dsp_top" startx="{pos}" starty="1" '
                         f'repeatx="{fixed_grid_w + 1}" priority="20"/>')

        # BRAM columns (unchanged)
        lines.append(f'      <col type="memory" startx="{BASELINE_BRAM_STARTX}" '
                     f'starty="1" repeatx="{BASELINE_BRAM_REPEATX}" priority="20"/>')

        # wc columns at removed DSP positions (priority 20 to claim DSP slots)
        for pos in dsp_removed:
            lines.append(f'      <col type="wc" startx="{pos}" starty="1" '
                         f'repeatx="{fixed_grid_w + 1}" priority="20"/>')

        # wc columns from CLB replacement (priority 15, uniform distribution)
        if wc_clb_repeatx is not None:
            lines.append(f'      <col type="wc" startx="{BASELINE_DSP_STARTX}" '
                         f'starty="1" repeatx="{wc_clb_repeatx}" priority="15"/>')

        lines.append(f'    </fixed_layout>')
        lines.append(f'  </layout>')
        layout_section = "\n".join(lines)
        text = FULL_LAYOUT_RE.sub(layout_section, text)

        dsp_pct = int(round(dsp_ratio * 100))
        clb_pct = int(round(clb_ratio * 100))
        out_name = f"nl_dpe_{rows}x{cols}_d{dsp_pct}_c{clb_pct}_fixed.xml"

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # ── Write output ──────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / out_name
    out_path.write_text(text)
    print(f"Generated: {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-(R,C) VTR architecture XMLs"
    )
    parser.add_argument("--rows", type=int, required=True,
                        help="Crossbar rows")
    parser.add_argument("--cols", type=int, required=True,
                        help="Crossbar columns")
    parser.add_argument("--template", type=Path,
                        default=SCRIPT_DIR / "nl_dpe_22nm_auto.xml",
                        help="Template XML path")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Output directory")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "fixed_clb_replace", "fixed_dsp_bram",
                                 "fixed_dsp_clb_replace"],
                        help="Generation mode")
    parser.add_argument("--fixed-grid", type=int, nargs=2, metavar=("W", "H"),
                        help="Fixed layout grid dimensions (width height)")
    parser.add_argument("--clb-replace-ratio", type=float,
                        help="Fraction of CLB columns to replace with wc "
                             "(for fixed_clb_replace mode)")
    parser.add_argument("--extra-dsps", type=int,
                        help="Number of DSP columns (for fixed_dsp_bram mode)")
    parser.add_argument("--extra-brams", type=int,
                        help="Number of BRAM columns (for fixed_dsp_bram mode)")
    parser.add_argument("--pair-name", type=str,
                        help="Name suffix for DSP+BRAM output file "
                             "(for fixed_dsp_bram mode)")
    parser.add_argument("--dsp-ratio", type=float,
                        help="Fraction of DSP columns to replace with wc "
                             "(for fixed_dsp_clb_replace mode)")
    parser.add_argument("--clb-ratio", type=float,
                        help="Fraction of CLB area to replace with wc "
                             "(for fixed_dsp_clb_replace mode)")

    args = parser.parse_args()

    grid_w = args.fixed_grid[0] if args.fixed_grid else None
    grid_h = args.fixed_grid[1] if args.fixed_grid else None

    gen_arch_xml(
        rows=args.rows,
        cols=args.cols,
        template_xml=args.template,
        output_dir=args.output_dir,
        mode=args.mode,
        fixed_grid_w=grid_w,
        fixed_grid_h=grid_h,
        clb_replace_ratio=args.clb_replace_ratio,
        extra_dsps=args.extra_dsps,
        extra_brams=args.extra_brams,
        pair_name=args.pair_name,
        dsp_ratio=args.dsp_ratio,
        clb_ratio=args.clb_ratio,
    )
