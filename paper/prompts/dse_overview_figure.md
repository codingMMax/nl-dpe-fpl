# DSE Overview Figure — Top-Conference Style (ISCA/MICRO/ISFPGA)

## Figure Description

Generate a **methodology overview diagram** showing the two-round Design Space Exploration (DSE) for integrating DPE (Digital Processing Element) hard blocks into FPGA fabric.

**Style**: Clean, professional, top-tier architecture conference quality. Two-column figure width (~7 inches). Use muted professional colors (blues, greens, grays). No gradients or 3D effects. Crisp lines, consistent font sizes. Similar style to methodology figures in ISCA/MICRO/ISFPGA papers.

## Layout: Two-Row Flow Diagram

### Row 1: Round 1 — Crossbar Size Selection

**Left block**: Design Space
- Grid of crossbar configurations: R ∈ {128, 256, 512, 1024} × C ∈ {64, 128, 256}
- Show as a 4×3 matrix with cells labeled (e.g., "512×128")
- Highlight the top-3 configs (512×128, 1024×128, 1024×64) in green

**Arrow** → pointing right

**Middle block**: Evaluation
- Icon/label: "VTR Place & Route"
- Below: "Auto-layout grid (sized per design)"
- Below: "12 configs × 6 FC workloads = 72 VTR runs"

**Arrow** → pointing right

**Right block**: Result
- Ranking bar chart (simplified, 3-4 bars)
- Label: "#1: 512×128"
- Text: "EDAP geomean ranking"
- Arrow down: "Top-5 configs → Round 2"

### Row 2: Round 2 — Area Budget Optimization

**Left block**: Design Space
- Single parameter: "Area Budget: 0%, 10%, 20%, 30%, 40%, 50%"
- Show as a horizontal slider or 6 discrete points
- Below: "Proportional: CLBs, DSPs, BRAMs all reduced equally"
- Below: "5 DPE configs × 6 budgets"

**Arrow** → pointing right

**Middle block**: Dual Evaluation (two parallel paths)

Path A (top):
- Label: "DL Performance"
- Icon: DPE tile
- "Bare GEMV replicas (P × Fmax)"
- "3 FC workloads, geomean"

Path B (bottom):
- Label: "Non-DL Performance"
- Icon: FPGA fabric
- "FlexScore: Σ Fmax(budget)/Fmax(baseline)"
- "4 VTR benchmarks (bgm, LU8, stereo, arm)"

Both paths merge →

**Right block**: Result
- Pareto front sketch (simplified)
- X-axis: "Non-DL Degradation (1 − FlexScore)"
- Y-axis: "DL Latency"
- A few dots with a Pareto curve
- Star marking "Balanced Config"
- Label: "~400 VTR runs total"

## Connecting Elements

- **Round 1 → Round 2**: Arrow labeled "Top-5 configs"
- **Grid sizes**: Round 1 = "Auto" (variable), Round 2 = "60×60 Fixed"

## Annotations / Call-outs

- Round 1 box header: "**Round 1: Which crossbar size?**"
- Round 2 box header: "**Round 2: How much FPGA area for DPE?**"
- Small text showing VTR run counts: "72 runs" and "~400 runs"

## Color Scheme

| Element | Color |
|---------|-------|
| DPE-related blocks | Green (#059669) |
| FPGA fabric blocks | Blue (#2563EB) |
| Result/output blocks | Dark gray (#333) |
| Arrows | Medium gray (#888) |
| Highlighted configs | Light green fill |
| Background | White |
| Borders | Light gray (#DDD) |

## Typography

- Headers: Bold, 10pt
- Body text: Regular, 8-9pt
- Axis labels: 7pt
- All text: serif font (Times New Roman or similar)

## Key Design Principles

1. **Information flow**: Left to right, top to bottom
2. **Two clear phases**: Round 1 (top) and Round 2 (bottom)
3. **Dual metrics in Round 2**: DL and non-DL evaluated in parallel, merged at Pareto
4. **Progressive narrowing**: 12 configs → 5 configs → Balanced recommendation
5. **No clutter**: Only essential information. White space between blocks.

## Reference Data

- Round 1: 12 configs, 6 workloads, 72 VTR runs, EDAP metric, result: 512×128 wins
- Round 2: 5 configs, 6 budget levels, ~400 VTR runs, FlexScore + DL latency, Pareto front
- Grid: Round 1 auto-layout, Round 2 60×60 fixed
- Proportional budget: CLB/DSP/BRAM all reduced by same %, DPE tiles inserted
- FlexScore from Tan et al., IEEE CAL 2021
