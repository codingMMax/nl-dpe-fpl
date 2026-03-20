# Round 1 DSE: Crossbar Configuration Selection (Q1)

## Goal

Identify the FPGA-optimal NL-DPE crossbar configuration (R rows x C columns) for fully-connected / GEMV workloads. This answers paper **Q1: What crossbar size makes the best NL-DPE FPGA hard block?**

## Design Space

**Configurations** — 3 row counts x 3 column counts = 9 configs:

| Config  | DPE Logic Area (um^2) | VTR Tile (WxH) | Tile Total (um^2) | Power (mW) |
|---------|-----------------------|-----------------|--------------------|------------|
| 128x64  | 12,198                | 2x5             | 21,199             | 11.85      |
| 128x128 | 24,042                | 2x9             | 39,759             | 23.57      |
| 128x256 | 48,084                | 5x7             | 75,800             | 47.13      |
| 256x64  | 13,994                | 1x12            | 26,189             | 12.16      |
| 256x128 | 27,279                | 2x10            | 44,675             | 24.03      |
| 256x256 | 53,850                | 3x13            | 85,530             | 47.79      |
| 512x64  | 17,586                | 2x7             | 29,945             | 12.77      |
| 512x128 | 33,755                | 3x8             | 53,600             | 24.97      |
| 512x256 | 66,093                | 4x12            | 103,965            | 49.38      |

DPE area and power are computed analytically by `nl_dpe/area_power.py` from 22nm component models (crossbar, ACAM, DAC, buffers, XOR). VTR tile dimensions are determined by a routing-aware formula: each tile must accommodate DPE logic plus switch-box (SB = 688 um^2) and connection-box (CB = 303 um^2) routing overhead, with <10% area overshoot relative to the CLB tiles it displaces.

**Workloads** — 6 FC layers spanning small to large GEMV dimensions:

| Workload    | K (input) | N (output) | Rationale |
|-------------|-----------|------------|-----------|
| FC 64x64    | 64        | 64         | Tiny layer; all configs single-tile |
| FC 128x128  | 128       | 128        | Attention projection proxy (Q/K/V, d=128); all configs V=1 |
| FC 512x128  | 512       | 128        | Medium; only R=512 achieves V=1 |
| FC 256x512  | 256       | 512        | Wide output; R>=256 achieves V=1 |
| FC 512x512  | 512       | 512        | Large square; only R=512 achieves V=1 |
| FC 2048x256 | 2048      | 256        | Very deep; no config achieves V=1 |

These workloads are representative of FC layers in CNNs (LeNet through ResNet) and transformer attention projections. FC 128x128 specifically represents the Q/K/V linear projections in a transformer attention head with head dimension d=128. They are chosen to span the ACAM eligibility boundary: some fit a single tile (V=1, ACAM-eligible), others force multi-tile vertical stacking (V>1, CLB activation required).

## Assumptions and Implementation

### Tiling model
Each (config, workload) pair maps to V = ceil(K/R) vertical tiles and H = ceil(N/C) horizontal tiles, instantiating V*H DPE hard blocks. When V > 1, partial sums from vertically-stacked DPEs are reduced by a CLB-based adder tree. When V = 1, the DPE output is a final result and ACAM can perform in-DPE activation.

### RTL generation
`nl_dpe/gen_gemv_wrappers.py` generates concrete Verilog for each (config, workload): explicit DPE instantiations (`dpe` black-box primitives with `MSB_SA_Ready` output), column-prefixed adder trees (to avoid multi-driver collisions when H > 1), and a tanh activation LUT for V > 1 cases. The generated RTL is self-contained and synthesizable by VTR.

### Architecture XML generation
`nl_dpe/gen_arch_xml.py` patches a 22nm VTR architecture template with per-config DPE tile width, height, and area (in MWTA units). In Round 1, we use `auto` mode: the `<auto_layout>` directive lets VTR size the FPGA grid to just-fit each design. This means each (config, workload) pair gets a different FPGA grid — the minimum grid that can place and route the design.

### Area model
FPGA area is computed post-VTR as:

    FPGA_area_mm^2 = grid_W * grid_H * CLB_tile_um^2 / 1e6

where CLB_tile = 2239 um^2 (= CLB_logic 945 + SB 688 + 2*CB 303). This treats every grid cell as occupying one CLB-tile's worth of silicon, whether it contains a CLB, DPE, DSP, BRAM, or I/O. The auto-layout approach means area reflects the inherent resource demand of each design, not a fixed FPGA budget.

**Key assumption:** Round 1 does not constrain all configs to the same FPGA area. Each design gets a just-fit grid. This is deliberate — it measures the inherent area efficiency of each (config, workload) pair without penalizing small designs with wasted silicon. The fixed-area comparison is deferred to Round 2.

### Energy and latency model
After VTR reports Fmax and grid dimensions, `gemv_dse.py` patches an IMC simulator config (`azurelily/IMC/configs/nl_dpe.json`) with the actual crossbar geometry, per-config energy parameters from `area_power.py`, and the VTR-reported Fmax. The IMC simulator (`azurelily/IMC/test.py`) then computes per-inference energy (pJ) and latency (ns) including:

- **e_imc_vmm**: analog VMM energy (scales with V and Fmax)
- **e_imc_digital**: ACAM/digital-post energy (scales with H and cols)
- **e_clb_reduction**: adder tree energy in CLB (only when V > 1)
- **e_clb_activation**: fabric-based activation energy (only when V > 1; zero when ACAM activates)
- **e_sram**: weight SRAM read/write energy

### Ranking methodology
We use a SPEC-style normalized geometric mean to rank configs:

1. For each workload, normalize each config's metric to the best config for that workload (best = 1.0, others <= 1.0).
2. Compute the geometric mean of normalized values across all 6 workloads.
3. Combine throughput/mm^2 and throughput/J geomeans via their geometric mean to get a single combined score.

This ensures every workload contributes equally regardless of absolute magnitudes (small workloads have ~100x higher raw throughput/mm^2 than large ones). It also avoids bias toward configs that dominate one extreme workload.

**Total runs:** 9 configs x 6 workloads = 54 VTR + IMC runs. VTR used 12 parallel workers with route_chan_width = 300 and seed = 42. All 54 completed successfully.

---

## Key Finding

**512-row configurations dominate the ranking** because they maximize the number of workloads where vertical tiling V = 1, making those workloads eligible for ACAM in-DPE activation.

ACAM eligibility requires V = 1 (i.e., K <= R). With R = 512, five of six workloads achieve V = 1 (all except FC 2048x256, where K = 2048 forces V = 4). With R = 256, three workloads achieve V = 1 (fc_64_64, fc_128_128, fc_256_512). With R = 128, two (fc_64_64 and fc_128_128, where K <= 128).

ACAM-eligible runs consistently show lower energy (no CLB activation cost) and often smaller grid area (fewer reduction adder trees needed). For example, on FC 512x128, ACAM-eligible configs (V=1) use 159-275 pJ per inference, while V=2 configs use 328-573 pJ — a 2-4x energy penalty from CLB-based reduction and activation.

---

## Configuration Ranking

| Rank | Config  | GM Tput/mm^2 | GM Tput/J | GM Combined | ACAM-eligible workloads |
|------|---------|-------------|----------|-------------|------------------------|
| #1   | 512x128 | 0.883       | 0.822    | **0.852**   | 5/6                    |
| #2   | 512x256 | 0.673       | 0.599    | **0.635**   | 5/6                    |
| #3   | 512x64  | 0.505       | 0.792    | **0.632**   | 5/6                    |
| #4   | 256x128 | 0.405       | 0.598    | 0.492       | 3/6                    |
| #5   | 256x256 | 0.402       | 0.433    | 0.417       | 3/6                    |
| #6   | 128x256 | 0.401       | 0.276    | 0.333       | 2/6                    |
| #7   | 256x64  | 0.176       | 0.593    | 0.323       | 3/6                    |
| #8   | 128x64  | 0.245       | 0.379    | 0.305       | 2/6                    |
| #9   | 128x128 | 0.246       | 0.348    | 0.293       | 2/6                    |

Top-3 configs (512x128, 512x256, 512x64) all have R = 512 and achieve ACAM eligibility on 5/6 workloads. They advance to Round 2 (fixed-area fabric composition study).

---

## Per-Workload Observations

### FC 64x64 (K=64, N=64)
All 9 configs achieve V=1, H=1 (single DPE). Performance differences arise from tile area overhead — smaller crossbars (128x64) pack into 9x9 grids while larger ones (512x256) need 14x14. Best: 512x64 (smallest tile that still fits). Fmax range: 313-389 MHz.

### FC 128x128 (K=128, N=128) — Attention Projection Proxy
All 9 configs achieve V=1 (K=128 <= min R=128), so all are ACAM-eligible. Configs with C >= 128 need only H=1 (single DPE); C=64 configs need H=2. Differentiation is purely from tile area: 512x128 is best (10x10 grid, Fmax=359.5 MHz), while 256x64 is worst (24x24 grid due to H=2 with large tiles). This workload represents the Q/K/V linear projections in a transformer attention head (d=128) and confirms that even at the ACAM eligibility boundary (K = R = 128), area efficiency still favors compact tiles. Energy ranges from 111 pJ (128x128) to 266 pJ (512x256), driven entirely by analog crossbar size since all configs use ACAM activation.

### FC 512x128 (K=512, N=128)
Only R=512 configs achieve V=1 (ACAM-eligible). R=256 forces V=2; R=128 forces V=4. The performance gap is 10x between best (512x128) and worst (128x256), driven by both area efficiency and energy savings from ACAM. This workload most strongly separates the row-size tiers.

### FC 256x512 (K=256, N=512)
All R>=256 configs achieve V=1. Column count determines H: C=64 gives H=8 (8 DPEs), C=256 gives H=2. Despite having the same V=1 ACAM status, wider-column configs win on area because fewer DPE columns means a more compact grid. Best: 256x128 (V=1, H=4).

### FC 512x512 (K=512, N=512)
Only R=512 achieves V=1. Similar pattern to FC 512x128: ACAM-eligible configs dominate, with 512x256 (V=1, H=2) and 512x128 (V=1, H=4) at the top.

### FC 2048x256 (K=2048, N=256)
No config achieves V=1 (K=2048 >> max R=512). All configs require V=4-16 vertical tiles, with extensive CLB-based reduction. Best: 512x256 (V=4, H=1) which minimizes both tile dimensions. This workload tests raw tiling efficiency without ACAM benefit. The best/worst performance ratio is 14.3x — the widest spread, showing that config choice matters most for large workloads.

---

## Why 512x128 Wins

512x128 balances three factors:

1. **ACAM opportunity:** R=512 gives V=1 on 5/6 workloads — same as 512x256 and 512x64.
2. **Area efficiency:** C=128 is a middle ground. C=64 forces many horizontal tiles (H=8 for N=512), inflating grid area. C=256 makes each DPE tile large, inflating per-tile area even when fewer tiles are needed.
3. **Energy efficiency:** 512x128 has the best Tput/J on fc_512_128 (the workload where ACAM matters most) and competitive Tput/J across all other workloads.

512x256 ranks #2 due to higher per-tile area cost; 512x64 ranks #3 due to excessive horizontal tiling on wide workloads.

---

## Connection to Paper Q1

This Round 1 DSE directly answers **Q1: What crossbar size makes the best NL-DPE FPGA hard block?**

**Answer:** 512x128. The 512-row dimension is critical — it determines how many FC/GEMV workloads can exploit ACAM in-DPE activation (V=1 eligibility). The 128-column dimension optimizes the area-energy tradeoff across the workload mix.

The top-3 configs (512x128, 512x256, 512x64) advance to Round 2, where we fix the FPGA area and sweep CLB-to-DPE replacement ratios to answer Q2 (optimal DPE density).

---

## Figures

- **Figure: Config Ranking** (`round1_ranking.pdf`) — Horizontal bar chart showing geomean Tput/mm^2 and Tput/J for each config, sorted by combined score.
- **Figure: Config x Workload Heatmap** (`round1_heatmap.pdf`) — Normalized Tput/mm^2 per (config, workload) cell (9 configs x 6 workloads), annotated with V (vertical tile count) and ACAM eligibility. Reveals the structural pattern: 512-row configs have more dark (high-performance) cells because more workloads achieve V=1.
