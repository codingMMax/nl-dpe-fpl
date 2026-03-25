# NL-DPE Enhanced FPGA Architecture Figure

## Figure Description

Generate a **two-panel architecture diagram** showing the proposed NL-DPE enhanced FPGA. Left panel = FPGA floorplan with resource columns. Right panel = zoomed-in DPE tile internals. Top-tier architecture conference quality (ISCA/MICRO/ISFPGA).

**Style**: Clean, professional. Two-column figure width (~7 inches). Muted professional colors. No gradients or 3D effects. Crisp lines, consistent fonts. White background.

---

## Panel A (Left): FPGA Floorplan — "NL-DPE Enhanced FPGA"

Show a simplified FPGA grid with **column-based resource layout**. The grid represents the heterogeneous FPGA fabric after DPE integration.

### Column Types (left to right, repeating pattern):

```
| CLB | CLB | CLB | DPE | CLB | CLB | BRAM | CLB | CLB | CLB | DSP | CLB | CLB | DPE | CLB | ...
```

### Visual Elements:

- **CLB columns** (majority): Light gray fill (#E5E7EB). Label a few "CLB".
- **DPE columns** (replacing some DSP/CLB): Green fill (#059669). Label "DPE". These are the NL-DPE hard blocks. Show 2-3 DPE columns interspersed.
- **BRAM columns**: Blue fill (#3B82F6). Label "BRAM". Show 1-2 columns.
- **DSP columns** (remaining): Amber fill (#F59E0B). Label "DSP". Show 1-2 columns.
- **I/O ring**: Thin border around the grid labeled "I/O".

### Key annotations:
- Arrow from one DPE column to Panel B with label "Zoom-in"
- Small text: "DPE tiles replace DSP/CLB columns"
- Grid label: e.g., "120×120 VTR Grid" or generic "FPGA Fabric"

### DPE tile detail in the column:
- Each DPE column shows stacked DPE tiles (3 wide × 8 tall for 512×128 config)
- Tiles are visually distinct cells within the DPE column
- One tile highlighted (darker green outline) → zoom arrow to Panel B

### Proportions:
- CLB columns: ~78% of width (many columns)
- DPE columns: ~10-15% of width (a few columns)
- BRAM columns: ~5% (1-2 columns)
- DSP columns: ~2-5% (1-2 columns, some were replaced by DPE)

---

## Panel B (Right): DPE Tile Internals — "NL-DPE Hard Block"

Zoom into a single DPE tile showing the internal architecture. This is what differentiates NL-DPE from Azure-Lily.

### Block Diagram (top to bottom):

```
┌─────────────────────────────────────────┐
│              Input Buffer               │
│         (serial data_in, 40-bit)        │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│                                         │
│         ReRAM Crossbar Array            │
│         R rows × C columns              │
│                                         │
│   Voltage in (rows) → Current out (cols)│
│   Analog multiply-accumulate            │
│                                         │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│              ACAM Array                 │  ← THIS IS THE KEY
│        (Analog CAM, ReRAM-based)        │     DIFFERENTIATOR
│                                         │
│   Programmable nonlinear function:      │
│   • log(x) — for projection output     │
│   • exp(x) — for DIMM computation      │
│   • ReLU/sigmoid — for activation       │
│   • identity — for pass-through         │
│                                         │
│   One ACAM unit per crossbar column     │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│         Shift-Add Pipeline              │
│    (bit-serial readout, 8-bit input)    │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│             Output Buffer               │
│        (serial data_out, 40-bit)        │
└─────────────────────────────────────────┘
```

### Color coding:
- **Input/Output buffers**: Light gray (#E5E7EB)
- **ReRAM Crossbar**: Green (#059669) — analog compute core
- **ACAM Array**: Red/coral (#DC2626 or #EF4444) — the key innovation, make it visually prominent
- **Shift-Add Pipeline**: Light blue (#93C5FD)

### Key annotations on Panel B:
- Arrow pointing to ACAM with call-out: **"Replaces ADC — 45× lower energy"**
- Small text on crossbar: "Weight-persistent (projections) or Identity (DIMM)"
- Small text on ACAM: "Programmable: log, exp, ReLU, identity"
- Label crossbar dimensions: "512 rows × 128 cols" (or generic R×C)

### Comparison inset (optional, small):
A tiny side-by-side showing:
```
NL-DPE:       Azure-Lily:
Crossbar      Crossbar
   ↓             ↓
 [ACAM]        [ADC]     ← 53 pJ vs 2,386 pJ
   ↓             ↓
Output        Output
```

---

## Connecting Elements

- **Zoom arrow**: Dashed arrow from highlighted DPE tile in Panel A to Panel B border
- **Panel labels**: "(a) NL-DPE Enhanced FPGA" and "(b) DPE Tile Architecture"
- **Figure caption area**: Leave space below for caption text

---

## Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| CLB | Light gray | #E5E7EB |
| DPE / Crossbar | Green | #059669 |
| BRAM | Blue | #3B82F6 |
| DSP | Amber | #F59E0B |
| ACAM | Red/coral | #DC2626 |
| Shift-Add | Light blue | #93C5FD |
| I/O ring | Dark gray | #6B7280 |
| Buffers | Light gray | #F3F4F6 |
| Background | White | #FFFFFF |
| Borders/arrows | Dark gray | #374151 |

---

## Typography

- Panel labels: Bold, 11pt, serif
- Block labels (inside boxes): Bold, 9pt
- Annotation text: Regular, 8pt
- Small descriptions: Regular, 7pt, italic for emphasis
- All text: serif font (Times New Roman or similar)

---

## Key Design Principles

1. **Panel A** shows WHERE DPEs go in the FPGA (replacing DSP/CLB columns)
2. **Panel B** shows WHAT's inside a DPE (crossbar + ACAM)
3. **ACAM is visually prominent** — it's the key contribution (red color, call-out annotation)
4. **Clean data flow** — top to bottom in Panel B (input → crossbar → ACAM → output)
5. **Comparison with Azure-Lily** implied by the ACAM annotation ("replaces ADC")
6. **No clutter** — only essential blocks. White space between elements.

---

## Reference Data

- DPE crossbar: 512 rows × 128 columns (DSE-optimal config)
- DPE tile: 3×8 grid cells = 24 cells per tile
- ACAM: 130 rows of ReRAM-based analog CAM
- Data width: 40-bit bus
- Bit-serial pipeline: 8-bit input precision, 10 cycles per pass
- ACAM energy: 21.9 pJ per pass (vs Azure-Lily ADC: 2,386 pJ per pass)
- Total DPE pass energy: NL-DPE 53 pJ vs Azure-Lily 2,386 pJ (45× advantage)
- FPGA grid: 120×120 (DSE) or 550×550 (main results)
- Resource mix: ~78% CLB, ~10% DPE, ~7% BRAM, ~5% DSP
