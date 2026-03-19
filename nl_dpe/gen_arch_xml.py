"""
Generate per-(R,C) VTR architecture XMLs by patching the template XML.

Three modes:
  auto             - patch tile dims only, keep <auto_layout>
  fixed_clb_replace - patch tile dims + replace <auto_layout> with <fixed_layout>,
                      increasing wc column density via clb_replace_ratio
  fixed_dsp_bram   - patch tile dims + replace <auto_layout> with <fixed_layout>,
                      no wc column, tune DSP/BRAM column density

Usage:
  python gen_arch_xml.py --rows 256 --cols 128
  python gen_arch_xml.py --rows 256 --cols 128 --mode fixed_clb_replace \
      --fixed-grid 80 80 --clb-replace-ratio 0.08
  python gen_arch_xml.py --rows 256 --cols 128 --mode fixed_dsp_bram \
      --fixed-grid 80 80 --extra-dsps 4 --extra-brams 8 --pair-name all_dsp
"""

import argparse
import re
from pathlib import Path

from area_power import dpe_specs

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Regex patterns ────────────────────────────────────────────────────────
TILE_WC_RE = re.compile(r'<tile name="wc" height="\d+" width="\d+" area="\d+">')
LAYOUT_RE = re.compile(r'<auto_layout[^>]*>.*?</auto_layout>', re.DOTALL)


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

        fixed_layout = (
            f'<fixed_layout name="nl_dpe_{rows}x{cols}" '
            f'width="{fixed_grid_w}" height="{fixed_grid_h}">\n'
            f'      <perimeter type="io" priority="101"/>\n'
            f'      <corners type="EMPTY" priority="102"/>\n'
            f'      <fill type="clb" priority="10"/>\n'
            f'      <col type="dsp_top" startx="6" starty="1" repeatx="16" priority="20"/>\n'
            f'      <col type="wc" startx="6" starty="1" repeatx="{wc_repeatx}" priority="22"/>\n'
            f'      <col type="memory" startx="2" starty="1" repeatx="16" priority="20"/>\n'
            f'    </fixed_layout>'
        )
        text = LAYOUT_RE.sub(fixed_layout, text)

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

        fixed_layout = (
            f'<fixed_layout name="nl_dpe_{rows}x{cols}" '
            f'width="{fixed_grid_w}" height="{fixed_grid_h}">\n'
            f'      <perimeter type="io" priority="101"/>\n'
            f'      <corners type="EMPTY" priority="102"/>\n'
            f'      <fill type="clb" priority="10"/>\n'
            f'      <col type="dsp_top" startx="6" starty="1" repeatx="{dsp_repeatx}" priority="20"/>\n'
            f'      <col type="memory" startx="2" starty="1" repeatx="{bram_repeatx}" priority="20"/>\n'
            f'    </fixed_layout>'
        )
        text = LAYOUT_RE.sub(fixed_layout, text)

        out_name = f"nl_dpe_{rows}x{cols}_{pair_name}_fixed.xml"

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
                        choices=["auto", "fixed_clb_replace", "fixed_dsp_bram"],
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
    )
