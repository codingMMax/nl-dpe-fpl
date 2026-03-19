# NL-DPE Crossbar Size DSE Experiment Plan

## 1. Objective

Determine the optimal NL-DPE crossbar configuration (rows × cols) for a heterogeneous FPGA by evaluating **FC layers with activation** (VMM + reduction + activation) across 5 FC workloads and 1 attention head. Metrics: throughput/mm² and throughput/J.

The key insight: **rows determine ACAM activation eligibility** (V=1 needed), while **cols determine ACAM area/power cost**. These create a non-trivial Pareto front. When V=1, ACAM handles activation for free — no CLB logic needed, smaller FPGA, better energy.

**Two rounds**:
- **Round 1** (54 VTR runs): auto_layout, pure DPE efficiency study — which (R,C) shape is most efficient?
- **Round 2** (144 VTR runs): fixed-template FPGA, resource trade-off study — CLB replacement + DSP/BRAM equivalence on top-3 configs.

**Area methodology**: VTR-reported total area (logic + routing) from `vpr_stdout.log`, converted to mm² via MWTA.

---

## 2. Round 1 — DPE Config Sweep (54 VTR runs, auto_layout)

### 2.1 Approach

Each VTR run uses `auto_layout` — VTR determines the minimum FPGA grid needed. No fixed template, no DSP replacement. We simply **add DPEs** to the baseline arch XML and let VTR size the FPGA.

This is a pure DPE efficiency study: which (R, C) shape achieves best throughput/mm² and throughput/J when VTR gives each design exactly the resources it needs?

### 2.2 Per-Config Arch XML

For each of 9 crossbar configs, generate one arch XML:
1. Start from `nl_dpe_22nm_auto.xml` (baseline with `auto_layout`)
2. Update DPE tile dimensions and area from `dpe_specs(R, C)`:
   ```xml
   <tile name="wc" height="{tile_h}" width="{tile_w}" area="{area_tag_mwta}">
   ```
3. Keep CLBs, DSPs, BRAMs unchanged — VTR `auto_layout` sizes the grid to fit

### 2.3 Workloads

6 FC workloads **with activation** (not bare GEMV). The attention head is evaluated separately (see §12).

FC layers include activation (ReLU/tanh via CLB LUT) because:
- When V=1, ACAM handles activation for free → 0 CLBs
- When V>1, CLB reduction + activation logic is needed → measurable CLB cost
- This captures the key V=1 vs V>1 discontinuity

| Workload | K | N | Description |
|----------|------|-----|-------------|
| fc_64_64 | 64 | 64 | Small FC |
| fc_128_128 | 128 | 128 | Attention projection proxy (Q/K/V are d×d=128×128) |
| fc_512_128 | 512 | 128 | Medium FC |
| fc_2048_256 | 2048 | 256 | Large FC (worst-case DPE count) |
| fc_256_512 | 256 | 512 | Wide FC |
| fc_512_512 | 512 | 512 | Large-wide FC |

### 2.4 Per-Run Flow

For each (R, C, workload):
1. Generate arch XML with DPE tile (W, H, area_tag) from `dpe_specs(R, C)`
2. Generate RTL with correct V×H DPEs + reduction (if V>1) + activation (if V>1)
3. Run VTR with `auto_layout` → Fmax, grid_W, grid_H
4. Compute FPGA area: `grid_W × grid_H × 2239 µm²` → mm²
5. Patch IMC config: geometry (R, C), energy params from `dpe_specs(R, C)`, freq = VTR Fmax
6. Run IMC simulator with `fc_model(K, N, has_act=True)` → energy_pj, latency_ns
7. Compute metrics (see §8): throughput/mm², throughput/J

**54 total VTR runs**: 9 configs × 6 workloads.
**9 arch XMLs**: one per (R, C), shared across its 6 workloads.

---

## 3. DPE Physical Specs

### 3.1 Area and Power Model (`area_power.py`)

```python
from area_power import dpe_specs
specs = dpe_specs(row=256, col=256)  # returns full dict
```

Area components (DPE logic area, mm²):
```
crossbar_area = 0.011534 * (R/256) * (C/256)     ← 21% of baseline
acam_area     = 0.041431 / 256 * C               ← 77% (dominates, scales with C only)
inbuf_area    = 0.0003542 / 256 * max(R,C)
outbuf_area   = 0.0003542 / 256 * max(R,C)
dac_area      = 0.0000782 * (C*4) / 1024
xor_area      = (7*C) / (7*256) * 0.0000989
```

Power components (mW):
```
p_analogue = crossbar_power + dac_power + inbuf_power
p_digital  = acam_power + outbuf_power + xor_power   ← ACAM dominates (91%)
```

Energy (IMC simulator conventions):
```
e_analogue_pj = p_analogue / freq_GHz      # pJ per VMM row activation
e_digital_pj  = (p_digital / freq_GHz) / C # pJ per ACAM col per cycle
e_conv_pj     = 0                           # ACAM absorbs ADC cost
```

### 3.2 VTR Tile Sizing (Routing-Aware)

Each FPGA tile = core logic + 1 SB + 2 CBs. For a hard block spanning W×H tiles:
- Switch boxes: W×H (all retained inside hard block)
- Connection boxes: W+H only (perimeter, not 2×W×H)

**Formula** — find min (W, H) such that:
```
DPE_logic + W×H × SB + (W+H) × CB  ≈  W×H × CLB_tile
→ DPE_logic ≤ W×H × 1551 − (W+H) × 303
```

Constants (COFFE 22nm):
- SB = 688 µm², CB = 303 µm², CLB_logic = 945 µm², CLB_tile = 2239 µm²
- 1 MWTA = 0.033864 µm²

**The `area` tag in XML = DPE core logic area only** (consistent with CLB area=27905 MWTA = 945 µm²). Per-config (W, H) ensures fair comparison — small DPEs not penalized by oversized tiles.

### 3.3 Reference Table (all 9 configs)

| Config | Logic (µm²) | Tile W×H | area_tag (MWTA) | Power (mW) |
|--------|------------|----------|----------------|------------|
| 128×64  | 12,198  | 2×5  | 360,205   | 11.85 |
| 128×128 | 24,042  | 2×9  | 709,950   | 23.57 |
| 128×256 | 48,084  | 5×7  | 1,419,900 | 47.13 |
| 256×64  | 13,994  | 1×12 | 413,239   | 12.16 |
| 256×128 | 27,279  | 2×10 | 805,559   | 24.03 |
| 256×256 | 53,850  | 3×13 | 1,590,199 | 47.79 |
| 512×64  | 17,586  | 2×7  | 519,307   | 12.77 |
| 512×128 | 33,755  | 3×8  | 996,777   | 24.97 |
| 512×256 | 66,093  | 4×12 | 1,951,716 | 49.38 |

---

## 4. Tiling and DPE Count

```
V = ceil(K / R)       vertical tiles
H = ceil(N / C)       horizontal tiles
X = V × H            DPEs needed per inference
acam_eligible = (V == 1)
```

**X per config per workload** (bold = V=1, ACAM eligible):

| Workload \ Config | 128×64 | 128×128 | 128×256 | 256×64 | 256×128 | 256×256 | 512×64 | 512×128 | 512×256 |
|-------------------|--------|---------|---------|--------|---------|---------|--------|---------|---------|
| fc_64_64      | **1**  | **1**   | **1**   | **1**  | **1**   | **1**   | **1**  | **1**   | **1**   |
| fc_128_128    | **1**  | **1**   | **1**   | **1**  | **1**   | **1**   | **1**  | **1**   | **1**   |
| fc_512_128    | 8      | 4       | 4       | 4      | 2       | 2       | **2**  | **1**   | **1**   |
| fc_2048_256   | 64     | 32      | 16      | 32     | 16      | 8       | 16     | 8       | 4       |
| fc_256_512    | 16     | 8       | 4       | **8**  | **4**   | **2**   | **8**  | **4**   | **2**   |
| fc_512_512    | 32     | 16      | 8       | 16     | 8       | 4       | **8**  | **4**   | **2**   |

**X_max** (= max DPEs needed per inference, always from `fc_2048_256`):

| Config  | 128×64 | 128×128 | 128×256 | 256×64 | 256×128 | 256×256 | 512×64 | 512×128 | 512×256 |
|---------|--------|---------|---------|--------|---------|---------|--------|---------|---------|
| X_max   | 64     | 32      | 16      | 32     | 16      | 8       | 16     | 8       | 4       |

In Round 1, VTR `auto_layout` sizes the grid to fit X_max DPEs — no DSP removal needed.

---

## 5. Arch XML Generation

### Round 1 (per config, auto_layout)

Starting from `nl_dpe_22nm_auto.xml`, modify only the DPE tile definition per (R, C):

1. **Update DPE tile dimensions and area**:
   ```xml
   <tile name="wc" height="{tile_h}" width="{tile_w}" area="{area_tag_mwta}">
     <sub_tile name="wc">
       <site pb_type="wc" pin_mapping="direct"/>
     </sub_tile>
   </tile>
   ```
   Values from `dpe_specs(R, C)`.

2. **Keep `auto_layout`**: VTR sizes the FPGA grid automatically. No DSP removal, no CLB replacement — all resources are available.

3. **9 arch XMLs total**, one per (R, C) config.

### Round 2 (per config, fixed_layout)

Starting from the Round 2 template FPGA (see §11):

1. **Part 1 (CLB replacement)**: remove CLB grid positions, place DPE columns in freed space
2. **Part 2 (DSP+BRAM equivalence)**: replace DPE tiles with equivalent DSP+BRAM columns

---

## 6. RTL Generation (per config × workload)

Extend `gen_gemv_wrappers.py` to accept (R, C) parameters. Each RTL file implements a **FC layer with activation** (not bare GEMV):

- **X = V×H `dpe` black box instances** (port interface constant across all configs)
- **Controller FSM**: MEM_IDLE → MEM_FILL → MEM_EXECUTE
- **SRAM instances**: weight/input buffering
- **If V > 1**: reduction adder tree (CLB, ceil(log2(V)) levels, N-wide)
- **If V > 1**: activation module (CLB LUT — tanh/sigmoid, N-wide outputs)
- **If V = 1**: no reduction/activation CLB (ACAM handles activation for free)

The activation module is key: it consumes CLBs when V>1 and is free when V=1. This creates the measurable CLB cost difference between DPE configs.

`dpe` black box ports (constant, from arch XML pb_type):
- Inputs: `data_in[15:0], reset, nl_dpe_control[1:0], shift_add_control, w_buf_en, shift_add_bypass, load_output_reg, load_input_reg, clk`
- Outputs: `data_out[15:0], MSB_SA_Ready, dpe_done, reg_full, shift_add_done, shift_add_bypass_ctrl`

---

## 7. Energy & Latency Model (IMC Simulator)

Energy and latency are computed by the existing **IMC simulator** (`azurelily/IMC/`), not by standalone analytical formulas. This ensures consistency with the full-network evaluation used later in the paper.

### 7.1 FC+Activation Model (`azurelily/models/fc.py` — TODO)

New model function, modeled after `gemv_model` in `azurelily/models/gemv.py`:

```python
def fc_model(num_computes, num_inputs, seq_length, head_dim, debug, energy_stats,
             K=64, N=64):
    K = seq_length   # reuse CLI args: --seq_length=K --head_dim=N
    N = head_dim
    layer = nn.Layer(
        in_channels=K, out_channels=N, kernel_size=1, stride=1, padding=0,
        name=f"fc_{K}_{N}", type="linear",
        has_act=True,          # ← key difference from gemv_model
        num_computes=num_computes, num_inputs=num_inputs,
        debug=debug, energy_stats=energy_stats,
    )
    layer.set_input(1, 1, K, is_first=True)
    all_layers = [layer]
    all_layers[0].add_event(C.EVENT_NEW_DATA, 0)
    return all_layers, num_inputs * len(all_layers)
```

### 7.2 Per-Config IMC Config (`nl_dpe_{R}x{C}.json` — generated)

For each (R, C), generate a patched `nl_dpe.json` with values from `dpe_specs(R, C)`:

```json
{
  "geometry": {
    "array_rows": R,       // ← from DSE config
    "array_cols": C        // ← from DSE config
  },
  "params": {
    "e_analoge_pj": ...,   // ← dpe_specs(R, C).e_analogue_pj
    "e_digital_pj": ...,   // ← dpe_specs(R, C).e_digital_pj
    "e_conv_pj": 0         // unchanged (ACAM absorbs ADC)
  },
  "fpga_specs": {
    "freq": ...,           // ← VTR Fmax (MHz) from this run
  }
}
```

All other fields (capabilities, timing, FPGA energy constants) remain unchanged from baseline `nl_dpe.json`.

### 7.3 How the Simulator Computes Energy & Latency

The scheduler (`scheduler.py:_run_linear`) drives:

1. **IMC core** (`imc_core.run_gemm(M=1, K, N)`):
   - `e_vmm = M × (V×H) × k_vmm × e_analoge_pj` — VMM energy across all DPE tiles
   - `e_conv = M × (V×H) × k_adc × e_conv_pj` — ADC conversion (0 for NL-DPE)
   - `e_digital = M × (V×H) × k_dig × e_digital_pj × C` — ACAM energy (per-col, all cols fire)
   - `e_reduction` — CLB tree reduction when V>1: `(V-1) × N × e_clb_pj_per_mac × clb_coeff_add`

2. **BRAM memory** (`memory.py`):
   - `e_read = ceil(K / bytes_per_access) × bram_pj_per_access`
   - `e_write = ceil(N / bytes_per_access) × bram_pj_per_access`

3. **Activation** — handled differently based on V:
   - **V=1** (`analoge_nonlinear_check` returns True): ACAM handles activation → **no CLB cost**
   - **V>1** (`analoge_nonlinear_check` returns False): `fpga.activation(M, N)` → CLB LUT energy = `M × N × act_energy_pj_per_op`

4. **Latency** — pipelined model:
   - `t_fill = t_read + t_core + t_reduction + t_write`
   - `t_steady = max(t_read, t_core, t_reduction, t_write)`
   - `latency = t_fill + (M-1) × t_steady` (M=1 for GEMV → latency = t_fill)
   - If activation needed: merged into pipeline via `_merge_pipelined_activation_latency`

### 7.4 Key Constants (from `nl_dpe.json`)

```
k_vmm = 8, k_digital = 1, k_adc = 8
e_clb_pj_per_mac = 0.660,  clb_coeff_add ≈ 0.5
act_energy_pj_per_op = 0.45 pJ
bram_pj_per_access = 0.0495 pJ
bram_width = 16 bits (2 bytes per access)
```

### 7.5 Output per DSE Point

```python
imc = IMC(patched_config_path)
all_layers, _ = fc_model(..., K=K, N=N)
run_regular_model(all_layers, imc)
imc.finalize_latency_stats()

total_energy_pj = sum(imc.energy_breakdown.values())
total_latency_ns = sum(imc.latency_stats.values())
# Component breakdown available via imc.energy_breakdown:
#   imc_vmm, imc_conversion, imc_digital_post, clb_reduction,
#   sram_read, sram_write, fpga_activation
```

---

## 8. Metrics

### 8.1 Area Calculation

**FPGA physical area = grid dimensions × CLB tile area**:
```
fpga_area_mm2 = grid_W × grid_H × CLB_tile_um2 / 1e6
              = grid_W × grid_H × 2239 / 1e6    [mm²]
```

Each grid cell occupies one CLB tile's worth of physical silicon (2239 µm²), regardless of what's placed there (CLB, DPE, DSP, BRAM). This is the actual chip area.

| Round | Layout | grid_W × grid_H | Area |
|-------|--------|-----------------|------|
| Round 1 | auto_layout | varies per run (VTR decides minimum grid) | varies — captures min-area efficiency |
| Round 2 | fixed_layout | constant (template from §11.1) | constant — fair fixed-budget comparison |

**Grid dimensions**: parsed from VTR output or from the `fixed_layout` XML definition.

**Important**: VTR also reports `Total logic block area` and `Total routing area` in `vpr_stdout.log`. These are internal accounting (sum of `area` tags of placed tiles + routing estimates) — they change when CLBs are swapped for DPEs even though the physical chip is the same size. **Do NOT use these for throughput/mm².**

**Two different formulas, two different purposes**:
- §3.2 SB+CB formula → DPE tile (W, H) in grid cells → **input** to arch XML
- §8.1 grid × CLB_tile → total FPGA physical area → **output** metric denominator

### 8.2 Performance Metrics

```
# From VTR
Fmax_MHz       ← VTR critical path (per run)
fpga_area_mm2  = grid_W × grid_H × 2239 / 1e6       [mm²]

# From IMC simulator (with VTR Fmax patched into config)
total_energy_pj ← sum(imc.energy_breakdown.values())  [pJ]
latency_ns      ← sum(imc.latency_stats.values())     [ns]

# Derived
throughput      = 1e9 / latency_ns                    [inf/s]
throughput_per_mm2 = throughput / fpga_area_mm2        [inf/s/mm²]
throughput_per_J   = 1e12 / total_energy_pj            [inf/J]

# Annotations
acam_eligible   = (V == 1)       # True when rows ≥ K
dpe_count       = V × H          # DPEs used by this workload
```

---

## 9. Implementation Plan

| File | Status | Role |
|------|--------|------|
| `nl_dpe/area_power.py` | **Done** | `dpe_specs(R, C)` → area, power, energy, tile W×H |
| `azurelily/models/fc.py` | **Done** | FC+activation model for IMC simulator (`has_act=True`) |
| `azurelily/IMC/test.py` | **Done** | `fc` and `attention` models registered |
| `nl_dpe/gen_arch_xml.py` | **Done** | All 3 modes: auto, fixed_clb_replace, fixed_dsp_bram |
| `nl_dpe/gen_gemv_wrappers.py` | **Done** | FC mode with V×H DPE tiling, adder tree, activation_lut |
| `gemv_dse.py` | **Done** | Round 1 orchestrator with SPEC-style geomean ranking |
| `dse/results/` | **Done** | Round 1 CSV, plots, analysis |
| `nl_dpe/gen_attention_wrapper.py` | **TODO** | Parameterized attention head RTL generator (Q/K/V + DIMM + softmax) |
| `gemv_dse.py` (attention) | **TODO** | Wire `--model attention` + `run_transformer_model` for attention DSE |

### 9.1 Per-DSE-Point Pipeline

```
For each (R, C, K, N):
  1. dpe_specs(R, C)          → area_tag, tile_W, tile_H, e_analogue, e_digital
  2. gen_arch_xml(R, C)       → nl_dpe_{R}x{C}_auto.xml (DPE tile updated)
  3. gen_gemv_wrappers(R,C,K,N) → fc_{K}_{N}_{R}x{C}.v (VTR-ready RTL)
  4. run_vtr(rtl, arch_xml)   → Fmax, grid_W, grid_H
  5. patch nl_dpe.json        → nl_dpe_{R}x{C}.json (geometry + energy + freq)
  6. run IMC simulator        → total_energy_pj, latency_ns, breakdown
  7. compute metrics          → throughput/mm², throughput/J, ACAM eligible
```

---

## 10. Expected Outcomes

**Finding 1 — V=1 eliminates CLB logic**
Configs where rows ≥ K use only DPEs, no CLB reduction/activation. This reduces active power and energy. Area is fixed (same FPGA), so throughput/mm² improvement is purely from lower latency and energy.

**Finding 2 — ACAM activation = step-function energy discontinuity**
At the V=1 boundary (rows = K), activation energy drops from N×0.45 pJ to 0, and reduction energy drops from (V−1)×N×0.33 pJ to 0. For K=512: 512-row gets V=1, 256-row gets V=2 → 128×0.45+127×N×0.33 pJ penalty.

**Finding 3 — Cols drive area cost, rows drive compute cost**
ACAM is 77% of DPE area, scales linearly with C. More cols → larger DPE → fewer DPEs fit for same X_max → wider FPGA footprint. But fewer DPEs means fewer SRAMs too. Pareto front emerges across (R, C) axis.

**Finding 4 — Optimal config is workload-dependent**
- K=64: all V=1, smallest DPE (128×64) wins (least area, fewest DSPs removed)
- K=512: 512-row gets V=1, 128-row pays full CLB penalty → clear winner at rows=512
- K=2048: no config gets V=1, but larger rows reduce tiling overhead → 512-row still wins on latency

**Finding 5 (Phase 2, Attention) — Shifts optimal toward wider crossbars**
GEMV prefers tall-narrow (high R, low C). Attention DIMM uses ACAM as log across all columns — more columns = more parallel log operations per DPE. Joint optimal expected around 256×128.

---

## 11. Round 2 — FPGA-Aware DSE (on Top-3 Configs)

Round 1 uses auto_layout (VTR sizes each design freely). Round 2 constrains to a **fixed FPGA template** to evaluate resource trade-offs.

### 11.1 Template Generation

1. Take the **smallest DPE config** (128×64) with the **largest workload** (fc_2048_256)
2. Run VTR with auto_layout → get minimum FPGA grid size
3. **Scale by 1.2×** → fixed_layout template (provides headroom for CLB replacement)
4. All Round 2 runs use this template — fixed FPGA area

### 11.2 Input
Top-3 DPE configs selected from Round 1 results (e.g., 256×128, 512×128, 512×256 — TBD from data).

### 11.3 Part 1 — CLB Replacement Sweep

**Sweep**: 4 CLB replacement ratios × 3 configs × 6 workloads = **72 VTR runs**

For each config C and ratio r ∈ {5%, 8%, 12%, 15%}:
1. Compute DPEs that fit in r% of CLB area: `n_dpes = floor(r × total_CLB_area / dpe_area)`
2. Remove `n_dpes × tile_cells` CLB grid positions, place DPE columns
3. Keep all DSPs and BRAMs — do NOT remove them
4. Run VTR for 6 workloads → Fmax, resource utilization
5. Compute metrics: inf/J, inf/mm², DPE utilization

**Key question**: at what CLB replacement ratio does the CLB become the bottleneck (not enough CLBs for reduction/activation when V>1)? This is the "balance point" between DPE compute and CLB support logic.

DPE utilization per workload:
```
dpe_util = tiles_needed / n_dpes_placed    (tiles_needed = V × H)
```

### 11.4 Part 2 — DSP+BRAM Equivalence (DPE Value Proposition)

**Goal**: prove that a DPE hard block is more efficient than the equivalent DSPs+BRAMs occupying the same area.

**DPE functional equivalence** for config (R, C) at int8 precision:
- **Compute**: R×C int8 MACs per VMM pass → X = R×C / DSP_PEAK_MAC DSPs equivalent
  (DSP in 9×9 SOP-4 mode: DSP_PEAK_MAC = 4 int8 MACs/cycle)
- **Storage**: R×C×8 bits of weight data → Y = R×C×8 / BRAM_CAPACITY BRAMs equivalent
  (BRAM_CAPACITY = 36 Kbit = 36,864 bits)

**Area constraint**: for each workload's n_dpes DPEs placed:
```
X' × DSP_area + Y' × BRAM_area = n_dpes × DPE_area      (same total area)
X' / Y' = X / Y                                            (maintain compute/storage ratio)
```

Since X:Y ≈ 1152:1 (DPE is compute-dense), the "balanced" pair is essentially all DSPs. So we explore **4 meaningful pairs** along the area constraint line:

| Pair | Strategy | X' (DSPs) | Y' (BRAMs) | Purpose |
|------|----------|-----------|------------|---------|
| 1 | All DSP | area / DSP_area | 0 | Max compute baseline |
| 2 | Balanced ratio | solve X'/Y' = X/Y | solve | DPE-equivalent split |
| 3 | Equal area | area/2 / DSP_area | area/2 / BRAM_area | 50-50 split |
| 4 | Storage-first | min feasible | remainder / BRAM_area | Max weight capacity |

**VTR runs**: 3 configs × 4 pairs × 6 workloads = **72 VTR runs**

For each pair, replace the DPE tiles in the arch XML with (X' DSPs + Y' BRAMs). Run the same workload RTL but implemented using DSPs (MAC tree) + BRAMs (weight storage) instead of `dpe` black boxes. Compare throughput/J and throughput/mm² against the DPE version.

### 11.5 Round 2 Totals

| Sub-experiment | Configs | Variants | Workloads | VTR Runs |
|----------------|---------|----------|-----------|----------|
| Part 1: CLB replacement | 3 | 4 ratios | 6 | 72 |
| Part 2: DSP+BRAM equiv | 3 | 4 pairs | 6 | 72 |
| **Round 2 Total** | | | | **144** |

Combined with Round 1: 54 + 144 = **198 VTR runs total** (~1-2 hours).

### 11.6 Expected Outcomes

**Finding 6 — CLB saturation point exists at high replacement ratios**
At 15% CLB replacement, workloads with V>1 may lack CLBs for reduction/activation, causing Fmax degradation or routing failure. The optimal CLB ratio depends on the workload's CLB demand.

**Finding 7 — DPE is more area-efficient than DSP+BRAM equivalent**
The analog VMM in a DPE performs R×C MACs in ~8 cycles within a compact crossbar. The equivalent X'≈51 DSPs would need 51 separate DSP tiles + routing + accumulation logic. The DPE's dense analog compute should deliver higher throughput/mm² despite the ACAM overhead.

**Finding 8 — Weight storage advantage**
A 256×128 DPE stores 32K×8 = 256 Kbit of weights internally (in the crossbar), equivalent to ~7 BRAMs. But the DPE's crossbar storage is co-located with compute — no memory bandwidth bottleneck. The DSP+BRAM alternative requires explicit data movement between BRAM and DSP.

---

## 12. Attention Head Experiment (Separate from Round 1/2)

The attention head workload is structurally different from FC layers and is evaluated as a separate experiment rather than as a 7th Round 1 workload. The FC proxy `fc_128_128` captures the DPE projection component in the Round 1 ranking; this experiment captures the full attention pipeline including CLB-intensive DIMM stages.

### 12.1 Why Separate

1. **RTL is not parameterized**: The hand-written `attention_head_1_channel.v` hardcodes `DEPTH=256` and instantiates 3 fixed `dpe` blocks. Sweeping 9 configs requires a parameterized generator.
2. **Different ACAM mode**: Attention uses ACAM as **log function** (for DIMM), not activation. Different energy characteristics.
3. **Different energy model**: The IMC simulator's `attention` model uses `run_transformer_model()` with a multi-stage pipeline (linear_Q/K/V → mac_qk → exp → norm → mac_sv), not the single-layer `fc` model.
4. **Q/K/V projections are small**: d=128 → V=1 for all R≥128. The projections don't differentiate configs; the CLB DIMM/softmax overhead is the interesting variable.

### 12.2 Implementation Plan

1. **`nl_dpe/gen_attention_wrapper.py`** — Parameterized generator:
   - Q/K/V projections: call `gen_fc_wrapper(K=d, N=d, rows=R, cols=C)` × 3 — handles tiling for H>1 (e.g., C=64 → H=2)
   - DIMM score matrix, softmax, weighted sum: extract fixed CLB modules from `attention_head_1_channel.v` (unchanged across configs)
   - Top-level: stitch projections + fixed CLB stages with correct data flow wiring

2. **Attention IMC runner in `gemv_dse.py`**:
   - Detect `--model attention` workload type
   - Call `run_transformer_model()` path in `azurelily/IMC/test.py`
   - Parse attention-specific energy breakdown: linear_Q/K/V (DPE), mac_qk (CLB), exp (CLB), norm (CLB), mac_sv (CLB)

3. **9 VTR runs** (one per config): fixed workload (N=128, d=128)
   - Output: `dse/results/attention_results.csv`
   - Key metrics: Fmax, grid size, DPE energy vs CLB energy breakdown

### 12.3 Expected Findings

**Finding 9 — ACAM-as-log benefits are uniform across configs**
All configs achieve V=1 for d=128 projections, so all can use ACAM-as-log. The performance differentiator is tile area (smaller DPE → smaller grid → better tput/mm²) and CLB overhead from DIMM stages.

**Finding 10 — CLB DIMM/softmax dominates attention energy**
The 3 DPE projections are a small fraction of total attention energy. The CLB-based score matrix (N²×d MACs), softmax (exp+norm), and weighted sum (N²×d MACs) dominate. This motivates future work on integrating reduction/DIMM inside the hard block.
