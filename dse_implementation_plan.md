# NL-DPE Crossbar Size DSE — Detailed Implementation Plan

This document describes the interface, logic, and dependencies for each TODO module listed in `dse_experiment_plan.md` §9.

## Modules

| # | File | Purpose | Effort |
|---|------|---------|--------|
| 1 | `azurelily/models/fc.py` | FC+activation model for IMC simulator | ~30 lines |
| 2 | `azurelily/IMC/test.py` | Register `fc_model` (2-line change) | 2 lines |
| 3 | `nl_dpe/gen_arch_xml.py` | Generate per-(R,C) arch XMLs | ~120 lines |
| 4 | `nl_dpe/gen_gemv_wrappers.py` | Extend for FC+activation RTL | ~200 lines |
| 5 | `gemv_dse.py` | DSE orchestrator (Round 1 + Round 2) | ~350 lines |

---

## Module 1: `azurelily/models/fc.py`

### Purpose

Single FC+activation layer for the IMC simulator. This is a minimal clone of `gemv_model` (`azurelily/models/gemv.py`) with exactly one change: `has_act=True`. The simulator's `_run_linear` in `scheduler.py` already handles the V=1 vs V>1 activation branching — no new simulation logic is needed here.

### Interface

Identical signature to `gemv_model` (required by `test.py`'s `model_list` dispatch):

```python
fc_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats)
    -> (all_layers, num_finishes)
```

CLI: `--seq_length K  --head_dim N` (same convention as gemv).

### Implementation

```python
import nn
import nn.constant as C


def fc_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats):
    """Single FC+activation layer model.

    Identical to gemv_model except has_act=True. The simulator's _run_linear
    checks analoge_nonlinear_check():
      V=1 (K <= crossbar rows): ACAM absorbs activation, no CLB energy added.
      V>1 (K > crossbar rows): fpga.activation(M, N) is called, adding CLB cost.
    """
    K = seq_length
    N = head_dim

    layer = nn.Layer(
        in_channels=K,
        out_channels=N,
        kernel_size=1,
        stride=1,
        padding=0,
        name=f"fc_{K}_{N}",
        type="linear",
        has_act=True,          # only difference from gemv_model
        num_computes=num_computes,
        num_inputs=num_inputs,
        debug=debug,
        energy_stats=energy_stats,
    )

    layer.set_input(1, 1, K)   # match gemv_model: no is_first kwarg
    all_layers = [layer]
    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)
    return all_layers, num_inputs * len(all_layers)
```

### What `has_act=True` changes in the simulator

From `scheduler.py:_run_linear` (lines 112–151):

1. `imc_core.run_gemm(M=1, K, N)` → VMM + ACAM + digital post-processing
2. `analoge_nonlinear_check(M, K, N)`:
   - **V=1** (K ≤ crossbar rows): `True` → activation absorbed by ACAM peripheral, no extra CLB energy or latency
   - **V>1**: `False` → `fpga.activation(M, N)` called, adding `M×N × act_energy_pj_per_op` energy and merging latency into pipeline

The DSE exploits this: configs where all workloads land on V=1 get activation for free. This is the main metric signal — ACAM eligibility.

### Notes

- No `FC_SHAPES` dict needed — the orchestrator (`gemv_dse.py`) owns the shape sweep, not the model file
- `set_input(1, 1, K)` not `set_input(1, 1, K, is_first=True)` — matches gemv_model exactly; `is_first` is set separately if needed by multi-layer models

### Testing

```bash
cd azurelily/IMC
python test.py --model fc --imc_file configs/nl_dpe.json --seq_length 64 --head_dim 64
# V=1: expect same energy as gemv (no activation CLB cost)
python test.py --model fc --imc_file configs/nl_dpe.json --seq_length 512 --head_dim 128
# V>1 for 256-row config: expect higher energy (activation CLB cost added)
```

---

## Module 2: `azurelily/IMC/test.py` — register fc_model

### Changes

Two lines only:

```python
# Line 16 — after the gemv import:
from models.fc import fc_model

# Line 101 — inside model_list dict, after "gemv" entry:
"fc": fc_model,
```

### What does NOT change

- The `run_regular_model` path handles fc correctly (single-layer, no transformer branching)
- All CLI args (`--seq_length`, `--head_dim`) are already defined for gemv and work identically for fc
- The energy/latency printing at the end of `__main__` works with any model

### Verification

After adding the two lines:

```bash
python azurelily/IMC/test.py --model fc --imc_file IMC/configs/nl_dpe.json \
    --seq_length 64 --head_dim 64
# Should produce non-zero energy output, same format as gemv
```

---

## Module 3: `nl_dpe/gen_arch_xml.py`

### Purpose

Generate per-(R,C) architecture XMLs for VTR by patching `nl_dpe_22nm_auto.xml`. There are exactly **two locations** in the XML that need changing:
1. The `<tile name="wc">` definition (DPE tile size and area)
2. The `<layout>` section (`auto_layout` for Round 1, `fixed_layout` for Round 2)

### How the arch XML controls the FPGA

Understanding the XML structure is essential for implementing this module correctly.

**Tile definitions** (`<tiles>` section, lines 635–726):

Each tile type is defined once with fixed height, width, and MWTA area:

```xml
<tile name="clb"    height="1" width="1" area="27905">   <!-- 1×1 grid cell -->
<tile name="wc"     height="5" width="6" area="1595429"> <!-- DPE: 5×6 cells -->
<tile name="dsp_top" height="4" width="1" area="253779"> <!-- DSP: 4×1 cells -->
<tile name="memory"  height="2" width="1" area="137668"> <!-- BRAM: 2×1 cells -->
```

The `area` tag is the **MWTA logic area** of the tile's core logic (routing area added by VTR internally). This is what `area_power.py:dpe_specs()` computes: `area_tag_mwta`. The `height`/`width` determine how many grid cells the tile occupies — computed via the routing-aware formula in `area_power.py`.

**Layout section** (`<layout>`, lines 728–737):

```xml
<auto_layout aspect_ratio="1.0">
  <perimeter type="io" priority="101"/>
  <corners type="EMPTY" priority="102"/>
  <fill type="clb" priority="10"/>
  <col type="dsp_top" startx="6" starty="1" repeatx="16" priority="20"/>
  <col type="wc"      startx="6" starty="1" repeatx="16" priority="22"/>
  <col type="memory"  startx="2" starty="1" repeatx="16" priority="20"/>
</auto_layout>
```

**Priority rules** (higher = overrides lower):
- `fill type="clb" priority="10"` — baseline: every tile is CLB
- `col type="dsp_top" ... priority="20"` — DSP columns at x=6,22,38,... override CLBs
- `col type="wc"      ... priority="22"` — **DPE columns at same x=6,22,38,..., priority 22 > 20, so DPE beats DSP at those positions**
- `col type="memory"  ... priority="20"` — BRAM columns at x=2,18,34,...

So the current layout puts one DPE column + one BRAM column every 16 cells. The `wc` tile is 6 columns wide (`width="6"`), so a single DPE column consumes 6 grid columns worth of area.

**What `auto_layout` does**: VTR sizes the FPGA grid to be the smallest square that fits the design's netlist. This is why it's used for Round 1 — we let VTR find the minimum footprint.

**What `fixed_layout` does**: explicitly specifies `width` and `height` of the entire chip. Used for Round 2 so all configurations are compared on the same chip area.

### Approach: regex text patching

The template XML uses custom VTR DTD entities that break `xml.etree`. Use regex-based text replacement — the same approach used in `run_imc_with_vtr_freq.py` for the IMC config JSON.

Only two regex replacements are needed:

```python
TILE_WC_RE = re.compile(
    r'<tile name="wc" height="\d+" width="\d+" area="\d+">'
)
LAYOUT_RE = re.compile(
    r'<auto_layout[^>]*>.*?</auto_layout>', re.DOTALL
)
```

### Interface

```python
gen_arch_xml(
    rows: int, cols: int,
    template_xml: Path = SCRIPT_DIR / "nl_dpe_22nm_auto.xml",
    output_dir: Path = Path("."),
    mode: str = "auto",              # "auto" | "fixed_clb_replace" | "fixed_dsp_bram"
    # Round 2 only:
    fixed_grid_w: int = None,
    fixed_grid_h: int = None,
    clb_replace_ratio: float = None, # 0.05, 0.08, 0.12, 0.15
    extra_dsps: int = None,          # for DSP+BRAM equivalence
    extra_brams: int = None,
) -> Path
```

### Mode = "auto" (Round 1 — 9 XMLs)

Only one patch needed: update the `<tile name="wc">` dimensions from `dpe_specs(rows, cols)`.

```python
specs = dpe_specs(rows, cols)
w, h = specs['tile_width'], specs['tile_height']
area = int(specs['area_tag_mwta'])

xml = template_xml.read_text()
xml = TILE_WC_RE.sub(
    f'<tile name="wc" height="{h}" width="{w}" area="{area}">',
    xml
)
# Layout stays <auto_layout aspect_ratio="1.0"> — unchanged
out = output_dir / f"nl_dpe_{rows}x{cols}_auto.xml"
out.write_text(xml)
```

The `<col type="wc">` placement directive stays `repeatx="16"` — this is fine for auto_layout because VTR sizes the grid to fit the design, regardless of how many DPE columns the template says to place.

### Mode = "fixed_clb_replace" (Round 2 Part 1 — 12 XMLs)

Goal: on a **fixed-size grid**, replace a fraction of CLB area with more DPEs by increasing DPE column density (smaller `repeatx`).

```python
specs = dpe_specs(rows, cols)
w, h = specs['tile_width'], specs['tile_height']
area = int(specs['area_tag_mwta'])

# How many DPE columns fit in the grid?
# Each DPE column occupies `w` horizontal grid cells and spans full height.
# Total interior grid columns = fixed_grid_w - 2 (excluding IO perimeter)
interior_cols = fixed_grid_w - 2
# Reserved: 1 BRAM col per 16 (memory repeatx=16), 1 IO perimeter
bram_cols = interior_cols // 16
# Remaining columns available for CLB or DPE
avail_cols = interior_cols - bram_cols

# How many DPE columns for this ratio?
# ratio = (n_dpe_cols × w) / avail_cols
n_dpe_cols = max(1, round(clb_replace_ratio * avail_cols / w))
dpe_repeatx = max(w + 1, avail_cols // n_dpe_cols)  # spacing between DPE columns

new_layout = f"""<fixed_layout name="dse" width="{fixed_grid_w}" height="{fixed_grid_h}">
  <perimeter type="io" priority="101"/>
  <corners type="EMPTY" priority="102"/>
  <fill type="clb" priority="10"/>
  <col type="dsp_top" startx="6" starty="1" repeatx="16" priority="20"/>
  <col type="wc"      startx="6" starty="1" repeatx="{dpe_repeatx}" priority="22"/>
  <col type="memory"  startx="2" starty="1" repeatx="16" priority="20"/>
</fixed_layout>"""

xml = template_xml.read_text()
xml = TILE_WC_RE.sub(f'<tile name="wc" height="{h}" width="{w}" area="{area}">', xml)
xml = LAYOUT_RE.sub(new_layout, xml)
ratio_pct = int(clb_replace_ratio * 100)
out = output_dir / f"nl_dpe_{rows}x{cols}_clb{ratio_pct}_fixed.xml"
out.write_text(xml)
```

The actual number of DPEs placed is determined by VTR after P&R — read from `wc_count` in the VPR log. The `repeatx` controls maximum available DPE slots; VTR fills them only as needed by the netlist.

### Mode = "fixed_dsp_bram" (Round 2 Part 2 — 12 XMLs)

Goal: replace the DPE entirely with DSPs and BRAMs of equivalent area, on the same fixed grid.

Strategy: remove `wc` from the layout, adjust `dsp_top` and `memory` `repeatx` to match the requested `extra_dsps` and `extra_brams` counts.

```python
# Compute repeatx for DSPs to achieve extra_dsps columns
# DSP column height=4, width=1 — one col per repeatx interval
dsp_repeatx = max(1, (fixed_grid_w - 2) // max(1, extra_dsps))
bram_repeatx = max(1, (fixed_grid_w - 2) // max(1, extra_brams))

new_layout = f"""<fixed_layout name="dse" width="{fixed_grid_w}" height="{fixed_grid_h}">
  <perimeter type="io" priority="101"/>
  <corners type="EMPTY" priority="102"/>
  <fill type="clb" priority="10"/>
  <col type="dsp_top" startx="3" starty="1" repeatx="{dsp_repeatx}" priority="20"/>
  <col type="memory"  startx="6" starty="1" repeatx="{bram_repeatx}" priority="20"/>
</fixed_layout>"""
# Note: no <col type="wc"> — DPE tile is still defined in <tiles> but never placed
```

### CLI

```bash
# Round 1: generate all 9 auto-layout XMLs
python gen_arch_xml.py --rows 128 --cols 64
python gen_arch_xml.py --rows 256 --cols 128

# Round 2 Part 1: CLB replacement
python gen_arch_xml.py --rows 256 --cols 128 --mode fixed_clb_replace \
    --fixed-grid 80x80 --clb-replace-ratio 0.08

# Round 2 Part 2: DSP+BRAM equivalence
python gen_arch_xml.py --rows 256 --cols 128 --mode fixed_dsp_bram \
    --fixed-grid 80x80 --extra-dsps 4 --extra-brams 8
```

### Reference Data from Template

| Line | Tag | Key values |
|------|-----|-----------|
| 635 | `<tile name="clb">` | height=1, width=1, area=27905 MWTA |
| 658 | `<tile name="wc">` | height=5, width=6, area=1595429 ← **patch this** |
| 672 | `<tile name="dsp_top">` | height=4, width=1, area=253779 |
| 708 | `<tile name="memory">` | height=2, width=1, area=137668 |
| 730 | `<auto_layout>` ... `</auto_layout>` | **replace for Round 2** |

### Dependencies

`area_power.py` (dpe_specs), template XML (`nl_dpe_22nm_auto.xml`)

---

## Module 4: `nl_dpe/gen_gemv_wrappers.py` — extend for FC+activation RTL

### Purpose

Extend `gen_gemv_wrappers.py` to generate FC+activation Verilog for the DSE. The key design decision is **Python-generates-Verilog** (a hardware generator pattern), not a Verilog template with generic `generate` blocks.

### Why not a Verilog `generate` approach

The existing `gemv_1_channel.v` already uses hardcoded `generate/if` branches for specific (V,H) pairs: V1H1, V2H1, V≤4H1, V≤8H1, V1H2, V2H2. For the DSE across {128,256,512}×{64,128,256} configs, the (V,H) range grows significantly:

| Config | fc_2048_256 | fc_256_512 | fc_512_512 |
|--------|-------------|------------|------------|
| 128×64 | V=16, H=4 | V=2, H=8 | V=4, H=8 |
| 128×128 | V=16, H=2 | V=2, H=4 | V=4, H=4 |
| 256×64 | V=8, H=4 | V=1, H=8 | V=2, H=8 |

A generic Verilog `generate` for the adder tree (nested loops with intermediate wire arrays) is fragile in Yosys and hard to debug. Python can compute the exact structure for any (V,H) and emit clean, readable Verilog — this is exactly the hardware-generator pattern (analogous to Chisel/HLS).

No new Verilog template file is needed. The supporting modules (`sram`, `conv_controller`, `global_controller`, `dpe`) are copied verbatim from `gemv_1_channel.v` into each generated file.

### Changes to `gen_gemv_wrappers.py`

#### 1. Add `--fc` flag to CLI

```python
parser.add_argument("--fc", action="store_true",
                    help="Generate FC+activation wrappers (instead of bare GEMV)")
parser.add_argument("--output-dir", default=str(SCRIPT_DIR),
                    help="Output directory for generated .v files")
```

When `--fc` is not set, existing `gen_wrapper()` behavior is unchanged.

#### 2. New top-level function: `gen_fc_wrapper(k, n, rows, cols, output_dir)`

```python
def gen_fc_wrapper(k: int, n: int, rows: int, cols: int,
                   output_dir: Path) -> Path:
    """Generate a self-contained FC+activation .v file for VTR.

    Computes V = ceil(k/rows), H = ceil(n/cols) and emits concrete Verilog:
      - V×H dpe instantiations with named wires
      - Explicit binary adder tree (Python-generated, not Verilog generate)
      - If V>1: activation_lut module (CLB LUT approximation of tanh)
      - If V=1: no activation module (ACAM absorbs it in hardware)

    Output: fc_{k}_{n}_{rows}x{cols}.v
    """
    v = math.ceil(k / rows)
    h = math.ceil(n / cols)
    acam_eligible = (v == 1)

    lines = []
    lines += _emit_file_header(k, n, rows, cols, v, h)

    # Top-level FC module
    lines += _emit_fc_top(k, n, rows, cols, v, h, acam_eligible)

    # Supporting modules (copied verbatim from gemv_1_channel.v)
    lines += _read_supporting_modules()   # sram, conv_controller, global_controller, dpe

    out_path = Path(output_dir) / f"fc_{k}_{n}_{rows}x{cols}.v"
    out_path.write_text("\n".join(lines))
    print(f"  Generated {out_path.name}  (V={v}, H={h}, acam_eligible={acam_eligible})")
    return out_path
```

#### 3. `_emit_fc_top(k, n, rows, cols, v, h, acam_eligible)` — emits the FC module

Structure mirrors `conv_layer_single_dpe` from `gemv_1_channel.v` but with V×H DPEs and an adder tree:

```python
def _emit_fc_top(k, n, rows, cols, v, h, acam_eligible) -> list[str]:
    lines = []

    # Module header + parameters (DATA_WIDTH=16, DEPTH, ADDR_WIDTH)
    lines += _emit_module_header(k, n, rows, cols, v, h)

    # SRAM + controller signals (identical to gemv_1_channel.v single-DPE section)
    lines += _emit_sram_and_ctrl_signals(k, n)

    # DPE array: V×H instances with named output wires
    # dpe_out_r{i}_h{j} for i in range(v), j in range(h)
    lines += _emit_dpe_array(v, h, n, cols)

    # Vertical reduction (adder tree) — only if V > 1
    if v > 1:
        lines += _emit_adder_tree(v, h, n, cols)
        # activation_lut instance after adder tree
        lines += _emit_activation_lut_inst(h, n, cols)
    else:
        # V=1: adder tree is trivial (direct wire), no activation
        lines += _emit_v1_output_wires(h, n, cols)

    # H-way horizontal concatenation of column outputs
    if h > 1:
        lines += _emit_horizontal_concat(h, n, cols)

    # SRAM output buffer + global controller
    lines += _emit_output_stage(k, n)

    lines += ["endmodule", ""]
    return lines
```

#### 4. `_emit_adder_tree(v, h, n, cols)` — the core generator function

Python computes the binary reduction tree for V inputs and emits explicit `wire` assignments. For V=4, H=1:

```python
def _emit_adder_tree(v, h, n, cols) -> list[str]:
    """Emit a binary adder tree reducing V partial sums to 1 per column.

    Each DPE output is DATA_WIDTH (16) bits wide × COLS elements = 16*cols bits.
    Each adder level increases bit width by 1 (to hold the carry).

    Example V=4, H=1, cols=64:
      level 0 input width = 16 bits per element
      sum_l0_r0 = dpe_out_r0_h0 + dpe_out_r1_h0  // 17-bit
      sum_l0_r1 = dpe_out_r2_h0 + dpe_out_r3_h0  // 17-bit
      sum_l1_r0 = sum_l0_r0 + sum_l0_r1           // 18-bit
      col_sum_h0 = sum_l1_r0                       // final output for column 0
    """
    lines = []
    for hj in range(h):
        # Start with dpe outputs for this column
        current_level = [f"dpe_out_r{i}_h{hj}" for i in range(v)]
        current_width = 16  # DATA_WIDTH

        level = 0
        while len(current_level) > 1:
            next_level = []
            current_width += 1  # each addition adds 1 bit
            for pair in range(len(current_level) // 2):
                a = current_level[2 * pair]
                b = current_level[2 * pair + 1]
                wire_name = f"sum_l{level}_r{pair}_h{hj}"
                # Emit the wire declaration and assignment
                lines.append(
                    f"wire [{current_width * cols - 1}:0] {wire_name} = "
                    f"{a} + {b};"
                )
                next_level.append(wire_name)
            if len(current_level) % 2 == 1:
                # Odd number: pass through the last one
                next_level.append(current_level[-1])
            current_level = next_level
            level += 1

        # Name the column's final sum
        lines.append(f"wire [{current_width * cols - 1}:0] col_sum_h{hj} = {current_level[0]};")

    return lines
```

#### 5. `_emit_activation_lut_inst(h, n, cols)` — CLB activation for V>1

Emits an inline piecewise-linear tanh module (per column output), then instantiates it:

```python
def _emit_activation_lut_module() -> list[str]:
    """Piecewise-linear tanh approximation — synthesizes to CLB LUTs.

    Only included in generated files where V > 1.
    This is what the DSE measures: CLB count difference between V=1 and V>1 designs.

    Input: col_sum_h{j} (wider adder tree output, truncated to DATA_WIDTH)
    Output: act_out_h{j} (DATA_WIDTH * COLS wide)
    """
    return [
        "module activation_lut #(",
        "    parameter DATA_WIDTH = 16,",
        "    parameter N = 64",   # number of parallel elements
        ")(input wire clk,",
        "  input wire [DATA_WIDTH*N-1:0] data_in,",
        "  output reg [DATA_WIDTH*N-1:0] data_out);",
        "    integer i;",
        "    always @(posedge clk) begin",
        "        for (i = 0; i < N; i = i + 1) begin",
        # Piecewise-linear tanh: saturate at ±0.5 (Q1.15 scaled)
        "            if ($signed(data_in[i*DATA_WIDTH +: DATA_WIDTH]) > $signed(16'sd16384))",
        "                data_out[i*DATA_WIDTH +: DATA_WIDTH] <= 16'sd16384;",
        "            else if ($signed(data_in[i*DATA_WIDTH +: DATA_WIDTH]) < $signed(-16'sd16384))",
        "                data_out[i*DATA_WIDTH +: DATA_WIDTH] <= -16'sd16384;",
        "            else",
        "                data_out[i*DATA_WIDTH +: DATA_WIDTH] <= data_in[i*DATA_WIDTH +: DATA_WIDTH];",
        "        end",
        "    end",
        "endmodule",
    ]
```

#### 6. `_read_supporting_modules()` — copy from `gemv_1_channel.v`

Supporting modules (`sram`, `conv_controller`, `global_controller`, `dpe`) are extracted from `gemv_1_channel.v` by scanning for `module` / `endmodule` boundaries:

```python
SUPPORTING_MODULES = [
    "sram", "conv_controller", "global_controller", "dpe",
]

def _read_supporting_modules() -> list[str]:
    """Extract supporting module definitions from gemv_1_channel.v."""
    src = GEMV_SRC.read_text()
    # Split on 'module ' boundaries, keep only the listed modules
    ...
```

This avoids duplicating hundreds of lines of Verilog and ensures the generated FC file always uses the same tested primitives.

### Output

For each (K, N, R, C) point, `gen_fc_wrapper` writes a self-contained `.v` file:

```
dse/rtl/fc_64_64_256x256.v      # V=1, H=1 — no activation module
dse/rtl/fc_512_128_128x64.v     # V=4, H=2 — includes activation_lut
dse/rtl/fc_2048_256_128x256.v   # V=16, H=1 — includes activation_lut
```

Each file is fully inspectable. The adder tree is explicit — no generic Verilog loops to debug.

### Dependencies

`gemv_1_channel.v` (source of supporting modules), `area_power.py` (not needed here — V/H computed from K/N/R/C directly)

### Testing

After generating a file, check two things:
1. VTR synthesis completes (DPE count = V×H from VPR resource report)
2. CLB count is higher for V>1 files than V=1 files for the same workload (activation_lut effect)

---

## DSE Folder Structure

All DSE artifacts live under `dse/` at the project root. This keeps generated files, VTR runs, and results separate from source code.

```
dse/
├── configs/
│   ├── arch/                          # Generated architecture XMLs
│   │   ├── nl_dpe_128x64_auto.xml
│   │   ├── nl_dpe_256x128_auto.xml
│   │   ├── ...                        # 9 XMLs for Round 1
│   │   ├── nl_dpe_256x128_clb5_fixed.xml
│   │   └── ...                        # Round 2 fixed-layout XMLs
│   └── imc/                           # Patched IMC config JSONs
│       ├── nl_dpe_128x64.json
│       ├── nl_dpe_256x128.json
│       └── ...                        # one per (R,C) config
│
├── rtl/                               # Generated FC+activation RTL
│   ├── fc_64_64_128x64.v
│   ├── fc_512_128_128x64.v
│   ├── ...                            # one per (config × workload)
│   └── fc_512_512_512x256.v
│
├── round1/                            # Round 1 VTR runs (auto_layout)
│   ├── 128x64/                        # grouped by DPE config
│   │   ├── fc_64_64/                  # each workload is a VTR run dir
│   │   │   ├── vpr_stdout.log
│   │   │   ├── *.net, *.place, *.route
│   │   │   └── imc_result.json        # IMC simulator output
│   │   ├── fc_512_128/
│   │   ├── fc_2048_256/
│   │   ├── fc_256_512/
│   │   ├── fc_512_512/
│   │   └── attention/
│   ├── 128x128/
│   ├── ...
│   └── 512x256/
│
├── round2/
│   ├── part1_clb/                     # CLB replacement sweep
│   │   ├── 256x128_clb5/             # {config}_clb{ratio_pct}
│   │   │   ├── fc_64_64/
│   │   │   │   ├── vpr_stdout.log
│   │   │   │   └── imc_result.json
│   │   │   └── ...
│   │   ├── 256x128_clb8/
│   │   ├── 256x128_clb12/
│   │   ├── 256x128_clb15/
│   │   └── ...                        # 3 configs × 4 ratios = 12 dirs
│   │
│   └── part2_dsp_bram/                # DSP+BRAM equivalence
│       ├── 256x128_all_dsp/           # {config}_{pair_name}
│       │   ├── fc_64_64/
│       │   │   ├── vpr_stdout.log
│       │   │   └── imc_result.json
│       │   └── ...
│       ├── 256x128_balanced/
│       ├── 256x128_equal_area/
│       ├── 256x128_storage_first/
│       └── ...                        # 3 configs × 4 pairs = 12 dirs
│
└── results/                           # Aggregated CSVs + plots
    ├── round1_results.csv
    ├── round2_part1_results.csv
    ├── round2_part2_results.csv
    ├── top3_configs.json              # Selected top-3 from Round 1
    ├── round1_throughput_per_mm2.png
    ├── round1_throughput_per_J.png
    └── round2_comparison.png
```

### Naming Conventions

| Artifact | Pattern | Example |
|----------|---------|---------|
| Arch XML (Round 1) | `nl_dpe_{R}x{C}_auto.xml` | `nl_dpe_256x128_auto.xml` |
| Arch XML (CLB replace) | `nl_dpe_{R}x{C}_clb{pct}_fixed.xml` | `nl_dpe_256x128_clb5_fixed.xml` |
| Arch XML (DSP+BRAM) | `nl_dpe_{R}x{C}_{pair}_fixed.xml` | `nl_dpe_256x128_all_dsp_fixed.xml` |
| IMC config | `nl_dpe_{R}x{C}.json` | `nl_dpe_256x128.json` |
| FC RTL | `fc_{K}_{N}_{R}x{C}.v` | `fc_512_128_256x128.v` |
| VTR run dir (R1) | `round1/{R}x{C}/{workload}/` | `round1/256x128/fc_512_128/` |
| VTR run dir (R2 P1) | `round2/part1_clb/{R}x{C}_clb{pct}/{workload}/` | `round2/part1_clb/256x128_clb5/fc_512_128/` |
| VTR run dir (R2 P2) | `round2/part2_dsp_bram/{R}x{C}_{pair}/{workload}/` | `round2/part2_dsp_bram/256x128_all_dsp/fc_512_128/` |
| IMC result | `{vtr_run_dir}/imc_result.json` | (alongside VTR logs) |

### `imc_result.json` Format

Stored alongside VTR outputs in each run directory:

```json
{
    "config": "256x128",
    "workload": "fc_512_128",
    "K": 512, "N": 128,
    "V": 2, "H": 1,
    "fmax_mhz": 42.5,
    "latency_ns": 1250.0,
    "energy_pj": 85.3,
    "energy_breakdown": {
        "imc_vmm": 31.12,
        "imc_digital_post": 5.47,
        "clb_reduction": 12.8,
        "fpga_activation": 3.6,
        "sram_read": 18.2,
        "sram_write": 14.1
    }
}
```

---

## VTR Integration: `run_vtr_dse()` wrapper

### Approach

Reuse the existing `nl_dpe/run_vtr.py` rather than create a new VTR runner. The DSE orchestrator imports `run_single`, `parse_metrics`, `parse_resources`, and `find_vpr_log` from `run_vtr.py`, and wraps them in a thin `run_vtr_dse()` function that handles DSE-specific directory layout.

### Why wrap instead of patch

- `run_vtr.py` is already well-tested and handles VTR path resolution, SDC files, error reporting, and metric parsing
- The DSE only needs control over **output directory** and **arch XML path** — both are already parameters of `run_single()`
- Patching `run_vtr.py` would risk breaking the existing multi-seed CLI workflow

### Implementation

```python
# In gemv_dse.py
import sys
sys.path.insert(0, str(Path(__file__).parent / "nl_dpe"))
from run_vtr import run_single, find_vpr_log, parse_metrics, parse_resources

# VTR setup (once at module level)
VTR_ROOT = Path(os.environ.get("VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"))
VTR_FLOW = VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py"
VTR_PYTHON = VTR_ROOT / ".venv" / "bin" / "python"
if not VTR_PYTHON.is_file():
    VTR_PYTHON = None


def run_vtr_dse(rtl_path: Path, arch_path: Path, run_dir: Path,
                route_chan_width: int = 300) -> dict:
    """Run a single VTR flow for a DSE point.

    Args:
        rtl_path: Path to FC+activation Verilog file
        arch_path: Path to per-(R,C) architecture XML
        run_dir: DSE output directory (e.g., dse/round1/256x128/fc_512_128/)

    Returns:
        dict with keys: fmax_mhz, grid_w, grid_h, resources, run_dir, wirelength
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reuse run_vtr.py's run_single — it handles:
    #   - VTR command construction
    #   - subprocess execution with error reporting
    #   - metric parsing (wirelength, fmax)
    #   - resource parsing (clb, wc, dsp_top, memory counts)
    result = run_single(
        vtr_flow=VTR_FLOW,
        vtr_python=VTR_PYTHON,
        design=rtl_path,
        arch=arch_path,
        route_chan_width=route_chan_width,
        sdc_file=None,         # FC designs don't need SDC constraints
        run_dir=run_dir,
        seed=42,               # fixed seed for DSE reproducibility
        run_index=0,
        total_runs=1,
        design_name=rtl_path.stem,
    )

    # Parse grid size from VPR log (not available in RunResult)
    vpr_log = find_vpr_log(run_dir)
    grid_w, grid_h = parse_grid_size(vpr_log)

    return {
        'fmax_mhz': result.fmax_mhz,
        'wirelength': result.wirelength,
        'grid_w': grid_w,
        'grid_h': grid_h,
        'resources': result.resources,
        'run_dir': run_dir,
    }


def run_vtr_batch(jobs: list, jobs_parallel: int = None) -> list:
    """Run multiple VTR jobs in parallel.

    Args:
        jobs: list of (rtl_path, arch_path, run_dir) tuples
        jobs_parallel: max parallel workers (default: cpu_count)

    Returns:
        list of result dicts (same order as input jobs)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if jobs_parallel is None:
        jobs_parallel = os.cpu_count() or 1

    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=jobs_parallel) as executor:
        futures = {}
        for i, (rtl, arch, run_dir) in enumerate(jobs):
            fut = executor.submit(run_vtr_dse, rtl, arch, run_dir)
            futures[fut] = i

        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()   # raises if VTR failed

    return results
```

### Grid Size Parsing

`run_vtr.py` parses wirelength, fmax, and resources, but NOT grid dimensions. The DSE needs grid size for the area metric (`grid_W × grid_H × 2239 / 1e6`). Add `parse_grid_size()` in `gemv_dse.py`:

```python
GRID_SIZE_RE = re.compile(r"FPGA sized to (\d+) x (\d+)")

def parse_grid_size(vpr_log_path: Path) -> tuple:
    """Parse FPGA grid dimensions from VPR log.

    VTR prints: 'FPGA sized to 40 x 40: 1600 grid tiles (auto)'
    Returns: (grid_w, grid_h)
    """
    content = vpr_log_path.read_text(errors="replace")
    match = GRID_SIZE_RE.search(content)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Could not parse grid size from {vpr_log_path}")
```

---

## Module 5: `gemv_dse.py` — DSE orchestrator

### Purpose

Single entry point for the full DSE. Controls what gets generated, what VTR runs, and how metrics are aggregated. Two rounds with different goals:

- **Round 1**: vary DPE crossbar geometry across 9 configs, auto-size FPGA, measure raw efficiency (throughput/mm², throughput/J, ACAM eligibility)
- **Round 2**: fix the FPGA grid at a template size, compare DPE density vs DSP+BRAM on a level playing field

### CLI

```bash
python gemv_dse.py --round 1                        # all 9 configs × 5 workloads
python gemv_dse.py --round 1 --configs 256x128      # single config, 5 workloads
python gemv_dse.py --round 1 --jobs 8               # 8 parallel VTR processes
python gemv_dse.py --round 1 --dry-run              # print plan, no execution
python gemv_dse.py --round 2                        # reads top3_configs.json from Round 1
python gemv_dse.py --round 2 --r1-results dse/results/round1_results.csv --top-k 3
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--round {1,2}` | required | Which round to run |
| `--configs` | all 9 | Comma-separated `RxC` configs to include |
| `--workloads` | all 5 | Comma-separated workload names |
| `--jobs` | `cpu_count` | Parallel VTR workers |
| `--dse-dir` | `dse/` | Root output directory |
| `--dry-run` | False | Print what would run, then exit |
| `--skip-existing` | True | Skip runs with existing `imc_result.json` |
| `--top-k` | 3 | Round 2: configs to carry forward |
| `--r1-results` | `dse/results/round1_results.csv` | Round 2: Round 1 CSV |
| `--template-scale` | 1.2 | Round 2: multiply template grid by this |

### Configuration constants

```python
DPE_CONFIGS = [(r, c) for r in [128, 256, 512] for c in [64, 128, 256]]  # 9 configs

FC_WORKLOADS = {
    "fc_64_64":    (64,   64),
    "fc_512_128":  (512,  128),
    "fc_2048_256": (2048, 256),
    "fc_256_512":  (256,  512),
    "fc_512_512":  (512,  512),
}

CLB_RATIOS   = [0.05, 0.08, 0.12, 0.15]   # Round 2 Part 1
DSP_BRAM_PAIRS = ["all_dsp", "balanced", "equal_area", "storage_first"]  # Round 2 Part 2

CLB_TILE_UM2  = 2239    # physical area per grid cell (µm²) — from area_power.py
DSP_AREA_MWTA = 253779  # from arch XML
BRAM_AREA_MWTA = 137668
```

### Round 1 control flow

```
main()
 └─ run_round1()
     ├─ Phase 1: Generate artifacts (fast, sequential)
     │    for (R, C) in DPE_CONFIGS:
     │        gen_arch_xml(R, C, mode="auto")          → dse/configs/arch/nl_dpe_{R}x{C}_auto.xml
     │        for (K, N) in FC_WORKLOADS:
     │            gen_fc_wrapper(K, N, R, C)            → dse/rtl/fc_{K}_{N}_{R}x{C}.v
     │
     ├─ Phase 2: Build VTR job list (45 runs for 5 workloads × 9 configs)
     │    jobs = [(rtl_path, arch_path, run_dir), ...]
     │    Skip any job where run_dir/imc_result.json already exists (resumable)
     │
     ├─ Phase 3: Run VTR in parallel
     │    run_vtr_batch(jobs, jobs_parallel=args.jobs)
     │    Each job runs VTR in its run_dir:
     │        run_dir = dse/round1/{R}x{C}/{workload}/
     │        writes: vpr_stdout.log, *.net, *.place, *.route
     │    Returns list of {fmax_mhz, grid_w, grid_h, resources, run_dir}
     │
     ├─ Phase 4: IMC simulator (sequential, fast — no VTR)
     │    for each (R, C, K, N, wl_name), vtr_res:
     │        imc_cfg = patch_imc_config(R, C, fmax=vtr_res['fmax_mhz'])
     │        energy_pj, latency_ns, breakdown = run_imc_fc(imc_cfg, K, N)
     │        save → run_dir/imc_result.json
     │        compute metrics (throughput/mm², throughput/J)
     │        append row to results list
     │
     ├─ Phase 5: Write CSV → dse/results/round1_results.csv
     │
     └─ Phase 6: Select top-3 → dse/results/top3_configs.json
          rank by mean(throughput_per_mm2) across all workloads per config
          break ties by mean(throughput_per_J)
```

### How `run_imc_fc` works

This is the IMC invocation in Phase 4. It calls `run_imc_with_vtr_freq.py`'s approach but in-process:

```python
def run_imc_fc(imc_cfg_path: str, K: int, N: int) -> tuple:
    """Run IMC simulator for one FC+activation point. Returns (energy_pj, latency_ns, breakdown)."""
    # Option A: subprocess (matches existing run_imc_with_vtr_freq.py pattern)
    cmd = [sys.executable, str(IMC_TEST),
           "--model", "fc",
           "--imc_file", imc_cfg_path,
           "--seq_length", str(K),
           "--head_dim", str(N)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(AZURELILY_ROOT))
    out = result.stdout + result.stderr

    energy_pj  = float(RE_ENERGY_LAYER.search(out).group(1))
    latency_ns = float(RE_LAT_CRIT.search(out).group(1))
    breakdown  = parse_energy_breakdown(out)   # parse per-component lines
    return energy_pj, latency_ns, breakdown
```

**Why subprocess over in-process IMC**: `run_imc_with_vtr_freq.py` already uses subprocess successfully. The IMC simulator has global state that makes repeated in-process calls unreliable. Subprocess isolates each call cleanly. The 45 IMC calls in Round 1 each take < 1 second — subprocess overhead is negligible.

### `patch_imc_config` details

The IMC config needs three things patched per DSE point:
1. `geometry.array_rows` and `geometry.array_cols` — crossbar size (varies per config)
2. `params.e_analoge_pj` and `params.e_digital_pj` — energy params (from `dpe_specs`, vary per config)
3. `fpga_specs.freq` — FPGA clock (from VTR result, varies per run)

```python
def patch_imc_config(base_cfg: Path, rows: int, cols: int,
                     fmax_mhz: float, output_path: Path):
    with open(base_cfg) as f:
        cfg = json.load(f)
    specs = dpe_specs(rows, cols, freq_ghz=fmax_mhz / 1000.0)
    cfg['geometry']['array_rows'] = rows
    cfg['geometry']['array_cols'] = cols
    cfg['params']['e_analoge_pj'] = specs['e_analogue_pj']
    cfg['params']['e_digital_pj'] = specs['e_digital_pj']
    cfg['fpga_specs']['freq'] = fmax_mhz
    with open(output_path, 'w') as f:
        json.dump(cfg, f, indent=4)
```

Note: `e_analogue_pj` and `e_digital_pj` depend on `freq_ghz` (power/freq = energy per cycle), so the freq must be passed to `dpe_specs` too — not just written to the JSON separately.

### Round 2 control flow

```
run_round2()
 ├─ Step 1: Load top-3 configs from dse/results/top3_configs.json
 │
 ├─ Step 2: Derive template grid from Round 1 results
 │    template = find result where config="128x64" AND workload="fc_2048_256"
 │    grid_w = ceil(template['grid_w'] × template_scale / 2) × 2  (round up to even)
 │    grid_h = ceil(template['grid_h'] × template_scale / 2) × 2
 │    # Why 128x64 + fc_2048_256? Smallest DPE config + largest workload = tightest fit
 │    # × 1.2 gives 20% headroom for Round 2 fixed-layout runs
 │
 ├─ Part 1: CLB replacement sweep (3 configs × 4 ratios × 5 workloads = 60 runs)
 │    for (R, C) in top3_configs:
 │        for ratio in [0.05, 0.08, 0.12, 0.15]:
 │            arch = gen_arch_xml(R, C, mode="fixed_clb_replace",
 │                                fixed_grid_w=grid_w, fixed_grid_h=grid_h,
 │                                clb_replace_ratio=ratio)
 │            for (K, N) in FC_WORKLOADS:
 │                run_dir = dse/round2/part1_clb/{R}x{C}_clb{pct}/{workload}/
 │                run VTR → parse resources (wc_count = actual DPEs placed)
 │                run IMC → energy, latency
 │                record: + clb_replace_ratio, n_dpes_placed=wc_count, dpe_utilization
 │
 └─ Part 2: DSP+BRAM equivalence (3 configs × 4 pairs × 5 workloads = 60 runs)
      for (R, C) in top3_configs:
          n_dpes = max wc_count seen for this config in Round 1
          dpe_area = n_dpes × dpe_specs(R,C)['area_tag_mwta']
          for pair in ["all_dsp", "balanced", "equal_area", "storage_first"]:
              x_dsps, y_brams = compute_pair(pair, dpe_area)
              arch = gen_arch_xml(R, C, mode="fixed_dsp_bram",
                                  fixed_grid_w=grid_w, fixed_grid_h=grid_h,
                                  extra_dsps=x_dsps, extra_brams=y_brams)
              RTL note: same fc_*.v files work — they use `dpe` black-box
                        VTR will fail to map DPEs onto DSP tiles (no match)
                        → need DSP-specific RTL (explicitly use DSP primitives)
                        OR treat as CLB-only (DSP tiles are present but unused)
              run VTR → parse dsp_count, mem_count (actual hardware used)
              run IMC (config with DSP energy model, not DPE)
              record: + pair_name, x_prime_dsps, y_prime_brams
```

### DSP+BRAM pair computation

```python
DSP_AREA_MWTA  = 253779   # from arch XML tile definition
BRAM_AREA_MWTA = 137668

def compute_dsp_bram_pair(pair_name: str, total_dpe_area_mwta: float) -> tuple:
    """Compute (n_dsps, n_brams) for a given equivalence strategy."""
    A = total_dpe_area_mwta
    if pair_name == "all_dsp":
        return (int(A / DSP_AREA_MWTA), 0)
    elif pair_name == "equal_area":
        return (int(A/2 / DSP_AREA_MWTA), int(A/2 / BRAM_AREA_MWTA))
    elif pair_name == "storage_first":
        # Minimum 1 DSP, rest BRAMs
        return (1, int((A - DSP_AREA_MWTA) / BRAM_AREA_MWTA))
    elif pair_name == "balanced":
        # Equal compute:storage ratio — solve for integer counts
        # DSP:BRAM count ratio matches 1 MAC : 1 KB weight storage
        # 1 DSP = 4 int8 MACs → compute bandwidth = 4N ops/cycle
        # 1 BRAM (36Kbit) → stores 36K/8 = 4500 int8 weights
        # balanced: BRAMs store enough weights for 1 full VMM pass
        # n_brams × 4500 = rows × cols  →  n_brams = ceil(R×C / 4500)
        # fill remaining area with DSPs
        n_brams = math.ceil(rows * cols / 4500)
        bram_area = n_brams * BRAM_AREA_MWTA
        n_dsps = max(1, int((A - bram_area) / DSP_AREA_MWTA))
        return (n_dsps, n_brams)
```

### Dry-run output (example)

```
$ python gemv_dse.py --round 1 --dry-run
Round 1 DSE plan: 9 configs × 5 workloads = 45 VTR runs
DSE dir: dse/

Config     Workload       V   H   DPEs  acam  RTL file                      Arch XML
---------- -------------- --- --- ----- ----- ----------------------------- ---------------------------
128x64     fc_64_64        1   2     2   yes  dse/rtl/fc_64_64_128x64.v    dse/configs/arch/nl_dpe_128x64_auto.xml
128x64     fc_512_128      4   2     8    no  dse/rtl/fc_512_128_128x64.v  dse/configs/arch/nl_dpe_128x64_auto.xml
128x64     fc_2048_256    16   4    64    no  dse/rtl/fc_2048_256_128x64.v dse/configs/arch/nl_dpe_128x64_auto.xml
...
512x256    fc_512_512      1   2     2   yes  dse/rtl/fc_512_512_512x256.v dse/configs/arch/nl_dpe_512x256_auto.xml

Run dirs: dse/round1/{config}/{workload}/
```

### Round 1 Pipeline

```python
def run_round1(configs, workloads, dse_dir, jobs):
    results = []
    arch_dir   = dse_dir / "configs" / "arch"
    imc_dir    = dse_dir / "configs" / "imc"
    rtl_dir    = dse_dir / "rtl"
    r1_dir     = dse_dir / "round1"
    result_dir = dse_dir / "results"

    for d in [arch_dir, imc_dir, rtl_dir, r1_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate arch XMLs (one per config) → dse/configs/arch/
    for (R, C) in configs:
        gen_arch_xml(R, C, mode="auto",
                     output_dir=str(arch_dir))
        # → dse/configs/arch/nl_dpe_{R}x{C}_auto.xml

    # Step 2: Generate IMC configs (one per config) → dse/configs/imc/
    for (R, C) in configs:
        specs = dpe_specs(R, C)
        patch_imc_config(
            base_config="azurelily/IMC/configs/nl_dpe.json",
            rows=R, cols=C,
            e_analoge_pj=specs['e_analogue_pj'],
            e_digital_pj=specs['e_digital_pj'],
            freq_mhz=0,  # placeholder, patched after VTR
            output_path=str(imc_dir / f"nl_dpe_{R}x{C}.json"),
        )

    # Step 3: Generate RTL (one per config × workload) → dse/rtl/
    for (R, C) in configs:
        for wl_name, (K, N) in workloads.items():
            gen_fc_wrapper(wl_name, K, N, R, C,
                          output_dir=str(rtl_dir))
            # → dse/rtl/fc_{K}_{N}_{R}x{C}.v

    # Step 4: Run VTR (parallelized, 54 total) → dse/round1/{config}/{workload}/
    vtr_jobs = []
    job_meta = []  # parallel list of (R, C, K, N, wl_name)
    for (R, C) in configs:
        arch = arch_dir / f"nl_dpe_{R}x{C}_auto.xml"
        for wl_name, (K, N) in workloads.items():
            rtl = rtl_dir / f"fc_{K}_{N}_{R}x{C}.v"
            run_dir = r1_dir / f"{R}x{C}" / wl_name
            vtr_jobs.append((rtl, arch, run_dir))
            job_meta.append((R, C, K, N, wl_name))

    vtr_results = run_vtr_batch(vtr_jobs, jobs_parallel=jobs)

    # Step 5: Run IMC simulator + save imc_result.json per run dir
    for (R, C, K, N, wl_name), vtr_res in zip(job_meta, vtr_results):
        # Patch freq into IMC config
        imc_cfg_path = imc_dir / f"nl_dpe_{R}x{C}.json"
        imc_cfg_patched = str(vtr_res['run_dir'] / f"nl_dpe_{R}x{C}_{wl_name}.json")
        patch_imc_config(
            base_config=str(imc_cfg_path),
            rows=R, cols=C,
            e_analoge_pj=dpe_specs(R,C)['e_analogue_pj'],
            e_digital_pj=dpe_specs(R,C)['e_digital_pj'],
            freq_mhz=vtr_res['fmax_mhz'],
            output_path=imc_cfg_patched,
        )

        energy_pj, latency_ns, breakdown = run_imc_fc(imc_cfg_patched, K, N)

        # Save imc_result.json alongside VTR outputs
        V = math.ceil(K / R)
        H = math.ceil(N / C)
        imc_result = {
            'config': f"{R}x{C}",
            'workload': wl_name,
            'K': K, 'N': N,
            'V': V, 'H': H,
            'fmax_mhz': vtr_res['fmax_mhz'],
            'latency_ns': latency_ns,
            'energy_pj': energy_pj,
            'energy_breakdown': breakdown,
        }
        with open(vtr_res['run_dir'] / "imc_result.json", 'w') as f:
            json.dump(imc_result, f, indent=4)

        # Compute metrics for CSV
        grid_area_mm2 = vtr_res['grid_w'] * vtr_res['grid_h'] * 2239 / 1e6
        throughput = 1e9 / latency_ns

        results.append({
            'config': f"{R}x{C}",
            'rows': R, 'cols': C,
            'workload': wl_name, 'K': K, 'N': N,
            'V': V, 'H': H, 'dpe_count': V * H,
            'acam_eligible': V == 1,
            'fmax_mhz': vtr_res['fmax_mhz'],
            'grid_w': vtr_res['grid_w'],
            'grid_h': vtr_res['grid_h'],
            'fpga_area_mm2': grid_area_mm2,
            'latency_ns': latency_ns,
            'energy_pj': energy_pj,
            'throughput_per_mm2': throughput / grid_area_mm2,
            'throughput_per_J': 1e12 / energy_pj,
            'e_imc_vmm': breakdown.get('imc_vmm', 0),
            'e_imc_digital': breakdown.get('imc_digital_post', 0),
            'e_clb_reduction': breakdown.get('clb_reduction', 0),
            'e_clb_activation': breakdown.get('fpga_activation', 0),
            'e_sram': breakdown.get('sram_read', 0) + breakdown.get('sram_write', 0),
            'clb_count': vtr_res['resources'].get('clb', 0),
            'dsp_count': vtr_res['resources'].get('dsp_top', 0),
            'mem_count': vtr_res['resources'].get('memory', 0),
            'wc_count': vtr_res['resources'].get('wc', 0),
            'run_dir': str(vtr_res['run_dir']),
        })

    # Step 6: Write CSV → dse/results/round1_results.csv
    csv_path = result_dir / "round1_results.csv"
    write_csv(results, csv_path)
    print(f"Round 1 results: {csv_path} ({len(results)} rows)")

    # Step 7: Select top-K configs → dse/results/top3_configs.json
    top_k = select_top_configs(results, k=3, metric='throughput_per_mm2')
    with open(result_dir / "top3_configs.json", 'w') as f:
        json.dump(top_k, f, indent=2)
    print(f"Top-{len(top_k)} configs: {top_k}")

    return results, top_k
```

### Helper Functions

#### IMC Config Patching

```python
def patch_imc_config(base_config, rows, cols, e_analoge_pj, e_digital_pj,
                     freq_mhz, output_path=None):
    """Create a patched nl_dpe.json for a specific (R, C, freq) config.

    Patches:
      geometry.array_rows = rows
      geometry.array_cols = cols
      params.e_analoge_pj = e_analoge_pj
      params.e_digital_pj = e_digital_pj
      fpga_specs.freq = freq_mhz

    Output path defaults to dse/configs/imc/nl_dpe_{R}x{C}.json
    """
    with open(base_config) as f:
        cfg = json.load(f)
    cfg['geometry']['array_rows'] = rows
    cfg['geometry']['array_cols'] = cols
    cfg['params']['e_analoge_pj'] = e_analoge_pj
    cfg['params']['e_digital_pj'] = e_digital_pj
    cfg['fpga_specs']['freq'] = freq_mhz
    if output_path is None:
        output_path = f"dse/configs/imc/nl_dpe_{rows}x{cols}.json"
    with open(output_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    return output_path
```

#### IMC Simulator Invocation

```python
def run_imc_fc(config_path, K, N):
    """Run IMC simulator for a single FC+activation layer.

    Option A (subprocess — simpler, uses existing test.py CLI):
        cmd = f"python azurelily/IMC/test.py --model fc --imc_file {config_path}"
              f" --seq_length {K} --head_dim {N}"
        # Parse stdout for energy/latency

    Option B (in-process — faster, no subprocess overhead):
        imc = IMC(config_path)
        all_layers, _ = fc_model(1, 1, K, N, False, energy_stats_template())
        for layer in all_layers:
            imc.run_layer(layer)
        imc.finalize_latency_stats()
        energy_pj = sum(imc.energy_breakdown.values())
        latency_ns = sum(imc.latency_stats.values())
        return energy_pj, latency_ns, dict(imc.energy_breakdown)

    Recommendation: Option B for Round 1 (54 calls, in-process is faster).
                    Option A as fallback / for debugging.
    """
```

#### Top-K Selection

```python
def select_top_configs(results, k=3, metric='throughput_per_mm2'):
    """Select top-K configs by average metric across workloads.

    For each config, compute mean(metric) over all workloads.
    Return top-K config tuples sorted descending.
    """
    from collections import defaultdict
    config_scores = defaultdict(list)
    for r in results:
        config_scores[r['config']].append(r[metric])
    avg_scores = {cfg: sum(v)/len(v) for cfg, v in config_scores.items()}
    sorted_configs = sorted(avg_scores.items(), key=lambda x: -x[1])
    return [cfg for cfg, _ in sorted_configs[:k]]
```

#### CSV Writer

```python
def write_csv(results: list, csv_path: Path):
    """Write results list-of-dicts to CSV.

    Uses the keys from the first result as column headers.
    Preserves column order as defined in the results dict.
    """
    import csv
    if not results:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
```

### Round 2 Pipeline

```python
def run_round2(top_configs, r1_csv_path, dse_dir, jobs, template_scale=1.2):
    """
    Round 2 has two parts:
      Part 1: CLB replacement sweep (72 runs)
      Part 2: DSP+BRAM equivalence (72 runs)

    Template generation:
      1. From Round 1, find (128×64, fc_2048_256) auto_layout result
      2. grid_template = (grid_w × template_scale, grid_h × template_scale)
      3. Use grid_template as fixed_layout for all Round 2 runs
    """
    arch_dir   = dse_dir / "configs" / "arch"
    rtl_dir    = dse_dir / "rtl"
    result_dir = dse_dir / "results"

    # Load Round 1 results from CSV
    r1_results = load_csv(r1_csv_path)

    # ── Template ──
    template_result = find_result(r1_results, config="128x64", workload="fc_2048_256")
    grid_w = int(int(template_result['grid_w']) * template_scale)
    grid_h = int(int(template_result['grid_h']) * template_scale)
    grid_w += grid_w % 2  # round up to even
    grid_h += grid_h % 2

    # ── Part 1: CLB replacement ──
    CLB_RATIOS = [0.05, 0.08, 0.12, 0.15]
    part1_jobs = []
    part1_meta = []
    for (R, C) in top_configs:
        specs = dpe_specs(R, C)
        for ratio in CLB_RATIOS:
            ratio_pct = int(ratio * 100)
            # Generate arch XML → dse/configs/arch/
            arch_path = gen_arch_xml(R, C, mode="fixed_clb_replace",
                        fixed_grid_w=grid_w, fixed_grid_h=grid_h,
                        clb_replace_ratio=ratio, output_dir=str(arch_dir))

            for wl_name, (K, N) in FC_WORKLOADS.items():
                rtl = rtl_dir / f"fc_{K}_{N}_{R}x{C}.v"
                run_dir = dse_dir / "round2" / "part1_clb" / f"{R}x{C}_clb{ratio_pct}" / wl_name
                part1_jobs.append((rtl, arch_path, run_dir))
                part1_meta.append((R, C, K, N, wl_name, ratio))

    part1_vtr = run_vtr_batch(part1_jobs, jobs_parallel=jobs)

    # Run IMC + collect results (same pattern as Round 1 Step 5)
    part1_results = []
    for (R, C, K, N, wl_name, ratio), vtr_res in zip(part1_meta, part1_vtr):
        # ... IMC run, metric computation ...
        # Additional columns:
        # 'clb_replace_ratio': ratio,
        # 'n_dpes_placed': n_dpes,
        # 'dpe_utilization': wc_count / n_dpes,
        pass

    write_csv(part1_results, result_dir / "round2_part1_results.csv")

    # ── Part 2: DSP+BRAM equivalence ──
    DSP_BRAM_PAIRS = [
        ("all_dsp",       lambda area: (int(area / 253779), 0)),
        ("balanced",      lambda area: solve_balanced(area)),
        ("equal_area",    lambda area: (int(area/2 / 253779), int(area/2 / 137668))),
        ("storage_first", lambda area: (1, int((area - 253779) / 137668))),
    ]
    part2_jobs = []
    part2_meta = []
    for (R, C) in top_configs:
        specs = dpe_specs(R, C)
        n_dpes = get_max_dpes_for_config(R, C, r1_results)
        total_dpe_area = n_dpes * specs['area_tag_mwta']
        for pair_name, pair_fn in DSP_BRAM_PAIRS:
            x_prime, y_prime = pair_fn(total_dpe_area)
            arch_path = gen_arch_xml(R, C, mode="fixed_dsp_bram",
                        fixed_grid_w=grid_w, fixed_grid_h=grid_h,
                        extra_dsps=x_prime, extra_brams=y_prime,
                        output_dir=str(arch_dir))

            for wl_name, (K, N) in FC_WORKLOADS.items():
                rtl = rtl_dir / f"fc_{K}_{N}_{R}x{C}.v"
                run_dir = dse_dir / "round2" / "part2_dsp_bram" / f"{R}x{C}_{pair_name}" / wl_name
                part2_jobs.append((rtl, arch_path, run_dir))
                part2_meta.append((R, C, K, N, wl_name, pair_name, x_prime, y_prime))

    part2_vtr = run_vtr_batch(part2_jobs, jobs_parallel=jobs)

    # Run IMC + collect results
    part2_results = []
    for (R, C, K, N, wl_name, pair_name, xp, yp), vtr_res in zip(part2_meta, part2_vtr):
        # ... IMC run, metric computation ...
        # Additional columns:
        # 'pair_name': pair_name,
        # 'x_prime_dsps': xp,
        # 'y_prime_brams': yp,
        pass

    write_csv(part2_results, result_dir / "round2_part2_results.csv")
```

### Output CSV Schema

**Round 1 columns** (`dse/results/round1_results.csv`):

```
config, rows, cols,
workload, K, N,
V, H, dpe_count, acam_eligible,
fmax_mhz, grid_w, grid_h, fpga_area_mm2,
latency_ns, energy_pj,
throughput_per_mm2, throughput_per_J,
e_imc_vmm, e_imc_digital, e_clb_reduction, e_clb_activation, e_sram,
clb_count, dsp_count, mem_count, wc_count,
run_dir
```

**Round 2 Part 1** (`dse/results/round2_part1_results.csv`) adds:

```
clb_replace_ratio, n_dpes_placed, dpe_utilization
```

**Round 2 Part 2** (`dse/results/round2_part2_results.csv`) adds:

```
pair_name, x_prime_dsps, y_prime_brams
```

### VTR Result Preservation

Each VTR run directory (e.g., `dse/round1/256x128/fc_512_128/`) preserves the full VTR output:
- `vpr_stdout.log` — VTR log with grid size, Fmax, wirelength, resource counts
- `*.net`, `*.place`, `*.route` — VTR intermediate files (useful for debugging)
- `imc_result.json` — IMC simulator output (energy, latency, breakdown)
- `nl_dpe_{R}x{C}_{wl}.json` — IMC config used for this run (with freq patched)

This means any individual DSE point can be re-analyzed without re-running VTR.

### Resumability

The orchestrator checks for existing `imc_result.json` before launching a VTR run:

```python
def is_run_complete(run_dir: Path) -> bool:
    """Check if a DSE run already completed successfully."""
    return (run_dir / "imc_result.json").is_file()
```

In the VTR batch loop, skip completed runs:

```python
for rtl, arch, run_dir in vtr_jobs:
    if is_run_complete(run_dir):
        print(f"  Skipping {run_dir} (already complete)")
        # Load existing imc_result.json instead
        continue
    # ... run VTR + IMC ...
```

This allows re-running `gemv_dse.py` after a crash without re-doing completed work.

### Dependencies

- `nl_dpe/area_power.py` (dpe_specs)
- `nl_dpe/gen_arch_xml.py` (gen_arch_xml)
- `nl_dpe/gen_gemv_wrappers.py` (gen_fc_wrapper)
- `nl_dpe/run_vtr.py` (run_single, find_vpr_log, parse_metrics, parse_resources)
- `azurelily/IMC/simulator.py` (IMC class)
- `azurelily/models/fc.py` (fc_model)

---

## Dependency Graph

```
area_power.py (DONE)
     │
     ├──→ gen_arch_xml.py ──→ per-(R,C) arch XML files
     │                              │
     └──→ gen_gemv_wrappers.py ──→ per-(R,C,K,N) RTL files
                                    │
                                    ▼
                            run_vtr.py (existing)
                                    │
                                    ▼
                           VTR output: Fmax, grid_W×H, resources
                                    │
     area_power.py ─────────────────┤
     (e_analogue, e_digital)        │
                                    ▼
                           patch_imc_config() → patched nl_dpe.json
                                    │
     fc.py (new) ───────────────────┤
                                    ▼
                           IMC simulator → energy_pj, latency_ns
                                    │
                                    ▼
                           gemv_dse.py → CSV + plots
```

---

## Execution Order

1. Implement `fc.py` + register in `test.py`
2. Implement `gen_arch_xml.py`
3. Extend `gen_gemv_wrappers.py` with generic stacking + activation
4. Implement `gemv_dse.py` Round 1 pipeline
5. Test Round 1 end-to-end on 1 config × 1 workload
6. Run full Round 1 (54 VTR runs)
7. Analyze results, select top-3
8. Implement `gemv_dse.py` Round 2 pipeline
9. Run Round 2 (144 VTR runs)
