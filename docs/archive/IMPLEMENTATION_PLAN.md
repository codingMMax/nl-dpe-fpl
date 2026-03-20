# NL-DPE Implementation Plan
_Updated: 2026-03-14_

## Part A: NL-DPE RTL Activation Removal (Completed)

### Summary
Since all layers in both 1-channel designs use k_tile=1 (K ≤ 25 for LeNet, K ≤ 9 for ResNet), ACAM is always active. External activation modules were removed and pipelines rewired.

- **LeNet**: Removed 4 activation layers (act1-act4), pipeline: `conv1 → pool1 → conv2 → pool2 → conv3 → conv4 → conv5`
- **ResNet**: Removed 8 activation layers (act1-act8), pipeline: `conv1 → conv2 → pool1 → conv3 → conv4 → res1 → conv5 → pool2 → conv6 → pool3 → conv7 → conv8 → res2 → pool4 → conv9`

See `nl_dpe_rtl/NL_DPE_VS_AZURELILY.md` for detailed differences.

---

## Part B: Physical-Aware DPE Grid in IMC Simulator

### B.1 Problem Statement

The IMC simulator currently computes DPE resources as:
```
k_tile = ceil(K / rows)       # logical vertical tiles
n_tile = ceil(N / cols)       # logical horizontal tiles
total_imc = k_tile * n_tile   # assumes 1 physical DPE per logical tile
```

This does NOT match the RTL, which uses fewer physical DPEs and time-multiplexes them. For Azurelily ResNet 1-channel:
- **IMC simulator**: 68 total DPEs (peak 16)
- **VTR/RTL**: 35 total DPEs (from actual Verilog instantiations)

### B.2 Root Cause: RTL DPE Stacking vs IMC Formula

The RTL provisions physical DPEs with a **k_reuse=2** strategy — each physical DPE handles 2 logical K-tiles via sequential passes:

| Layer | K (full channels) | k_tile=ceil(K/256) | RTL phys_V | RTL phys_H | RTL DPEs | IMC DPEs |
|-------|-------------------|-------------------|-----------|-----------|---------|---------|
| conv1 | 1×9=9 | 1 | 1 | 1 | 1 | 1 |
| conv2 | 56×9=504 | 2 | 1 | 1 | 1 | 2 |
| conv3 | 112×9=1008 | 4 | 2 | 1 | 2 | 4 |
| conv4 | 112×9=1008 | 4 | 2 | 1 | 2 | 4 |
| conv5 | 112×9=1008 | 4 | 2 | 2 | 4 | 8 |
| conv6 | 224×9=2016 | 8 | 4 | 2 | 8 | 16 |
| conv7 | 224×9=2016 | 8 | 4 | 2 | 8 | 16 |
| conv8 | 224×9=2016 | 8 | 4 | 2 | 8 | 16 |
| conv9 | 224×1=224 | 1 | 1 | 1 | 1 | 1 |
| **Total** | | | | | **35** | **68** |

Pattern: `phys_V = ceil(k_tile / k_reuse)` where `k_reuse = 2`. H dimension matches exactly.

### B.3 Physical FPGA Layout (from Architecture XML)

Both `azure_lily_22nm_with_dpe_550x550.xml` and `nl_dpe_22nm_550x550.xml` define:

- **Grid**: 550×550
- **DPE columns**: 8 active columns at x = {68, 108, 248, 288, 348, 388, 448, 488}
- **DPE groups**: 4 pairs of 2 columns each (40-unit spacing within pair)
- **DPE tile size**: 5 rows × 6 cols
- **DPEs per column**: floor((550-2)/5) ≈ 109
- **Total DPE capacity**: ~872 DPEs

Other resources:
- **Memory**: every 10 columns from x=6 (tile: 2×1)
- **DSP**: every 10 columns from x=4 (tile: 4×1)
- **CLB**: fills remaining space (tile: 1×1)
- **IO**: perimeter

### B.4 Proposed Config Changes

Add minimal parameters to `fpga_specs` in the JSON config:

```json
"fpga_specs": {
    "dpe_cols_per_group": 2,
    "k_reuse": 2,
    "total_imc": 896,
    ... (existing fields unchanged)
}
```

- `dpe_cols_per_group`: max H per layer (can't span groups) — constrains n_tile
- `k_reuse`: DPE time-multiplexing factor — determines phys_V from k_tile

### B.5 Code Changes in IMC Simulator

#### B.5.1 `imc_core/imc_core.py` — `run_gemm()`

**Resource counting** (line 142-145):
```python
# Before:
k_tile = math.ceil(K / self.cfg.rows)
n_tile = math.ceil(N / self.cfg.cols)
total_imc = k_tile * n_tile

# After:
k_tile = math.ceil(K / self.cfg.rows)
n_tile = math.ceil(N / self.cfg.cols)
k_reuse = getattr(self.cfg, 'k_reuse', 1)
phys_k = math.ceil(k_tile / k_reuse)
phys_imc = phys_k * n_tile               # physical DPEs for this layer
k_passes = math.ceil(k_tile / phys_k)    # sequential passes
total_imc = k_tile * n_tile              # logical tiles (for energy, unchanged)
self.stats.record_resource("imc_tiles", phys_imc, peak=True)  # physical count
```

**Energy** — unchanged. Same total VMMs, ADC conversions, reductions. Energy is work-based, not resource-based.

**Latency** — core compute per row scales by k_passes:

#### B.5.2 `imc_core/imc_core.py` — `gemm_pipeline_profile()`

```python
# Before:
t_core_row = self._core_bit_pipeline_row_latency()

# After:
k_tile = math.ceil(K / self.cfg.rows)
k_reuse = getattr(self.cfg, 'k_reuse', 1)
phys_k = math.ceil(k_tile / k_reuse)
k_passes = math.ceil(k_tile / phys_k)
t_core_row = k_passes * self._core_bit_pipeline_row_latency()
```

Reduction latency also changes — two-stage reduction:
```python
# Before:
t_reduc_row = self.clb_reduction_energy_latency(k_tile, active_cols=N)[1]

# After: intra-pass reduction + cross-pass reduction
t_reduc_intra = self.clb_reduction_energy_latency(phys_k, active_cols=N)[1]
t_reduc_cross = self.clb_reduction_energy_latency(k_passes, active_cols=N)[1]
t_reduc_row = k_passes * t_reduc_intra + t_reduc_cross
```

#### B.5.3 `imc_core/config.py`

Read new fields from JSON:
```python
self.k_reuse = data['fpga_specs'].get("k_reuse", 1)
self.dpe_cols_per_group = data['fpga_specs'].get("dpe_cols_per_group", 2)
```

#### B.5.4 Config JSON files

Add `k_reuse` and `dpe_cols_per_group` to:
- `IMC/configs/azure_lily.json`
- `IMC/configs/nl_dpe.json`

### B.6 DNN Mapping Example: ResNet conv6 on Physical DPE Grid

**Layer parameters**: conv6 has in_channels=224, kernel=3×3, out_channels=224

**GEMM dimensions**: M=64 (8×8 output positions), K=2016 (224×9), N=224

**Tiling**:
```
k_tile = ceil(2016 / 256) = 8      # 8 logical vertical tiles
n_tile = ceil(224 / 128)  = 2      # 2 logical horizontal tiles
k_reuse = 2
phys_V = ceil(8 / 2) = 4           # 4 physical DPE rows
phys_H = 2                         # 2 physical DPE columns
physical DPEs = 4 × 2 = 8          # matches RTL conv_layer_stacked_dpes_V4_H2
```

**Physical placement on FPGA**:
```
          Column Group 1
     col 68        col 108
   ┌─────────┐  ┌─────────┐
   │ DPE(1,1)│  │ DPE(1,2)│  ← phys_H=2 uses both columns in a group
   │ K[0:255]│  │ K[0:255]│     N[0:127]  N[128:223]
   ├─────────┤  ├─────────┤
   │ DPE(2,1)│  │ DPE(2,2)│
   │K[256:511]│ │K[256:511]│
   ├─────────┤  ├─────────┤
   │ DPE(3,1)│  │ DPE(3,2)│
   │K[512:767]│ │K[512:767]│
   ├─────────┤  ├─────────┤
   │ DPE(4,1)│  │ DPE(4,2)│  ← phys_V=4 rows
   │K[768:1023]││K[768:1023]│
   └─────────┘  └─────────┘
```

**Execution timeline (k_passes=2)**:
```
Pass 1: Load K[0:1023] across 4 rows → VMM → ADC → reduce 4 partials → intermediate result
Pass 2: Load K[1024:2015] across 4 rows → VMM → ADC → reduce 4 partials → intermediate result
Cross-pass: Reduce 2 intermediate results → final output
```

**Per-row latency**: `2 × t_core + 2 × t_reduc_intra + t_reduc_cross`
**Energy**: unchanged from k_reuse=1 (same total VMM/ADC/reduction operations)
**Resources**: 8 DPEs (not 16 as before)

### B.7 Impact Summary

| Metric | Effect of k_reuse | Reason |
|--------|------------------|--------|
| DPE count (resource) | **Reduced by ~2×** | Physical DPEs = logical / k_reuse |
| VMM energy | Unchanged | Same total MAC operations |
| ADC energy | Unchanged | Same total conversions |
| Reduction energy | Unchanged | Same total additions |
| Memory energy | Unchanged | Same data volume |
| Core latency per row | **Increased by k_passes×** | Sequential DPE reuse |
| Reduction latency | **Two-stage** | Intra-pass + cross-pass |
| Pipeline steady-state | **Potentially slower** | Core stage bottleneck |

---

## Part C: Baseline Metrics (Pre-Change Snapshot)

### C.1 IMC Simulator — Azurelily, 1-channel

**LeNet**:
- Energy: 128,412.31 pJ
- Latency (critical path): 110,544.87 ns
- Resources: imc_tiles total=6 peak=2, clb total=398 peak=120, memory=12, act_units=64

**ResNet**:
- Energy: 18,372,299.85 pJ
- Latency (critical path): 423,125.17 ns
- Resources: imc_tiles total=68 peak=16, clb total=7792 peak=1568, memory=26, act_units=128

### C.2 NN Simulator — Azurelily, 1-channel

**LeNet**:
- Energy: 128.90 nJ (= 128,901.30 pJ)
- Latency: 114,206.40 ns

**ResNet**:
- Energy: 17,371.92 nJ (= 17,371,919.74 pJ)
- Latency: 992,527.80 ns

### C.3 VTR Flow — Azurelily, 1-channel (from prior runs)

**LeNet**: wc=5, clb=37, memory=13, dsp=0, io=21
**ResNet**: wc=35, clb=110, memory=15, dsp=0, io=21
