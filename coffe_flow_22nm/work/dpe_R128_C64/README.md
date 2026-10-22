# Run — `dpe_R128_C64`

- **Result** (`report.txt` → `HARDBLOCK INFORMATION`): pass network **48.4871971812 µm²**, with select SRAM **112.151517181 µm²** (core 0.0) — the local input crossbar footprint of the NL-DPE `dpe` block at **R=128** (B_in = 40×128/128 = 40 bits/cycle).
- **Block identity** (bandwidth model): `B_in = 40×R/128 = 40` → `num_gen_inputs = 40 + 7 controls = 47`; `B_out = 40×C/64 = 40` (+5 status) → `num_gen_outputs = 45` (**inert** — COFFE never reads it). `crossbar_population = 0.50`, `num_crossbars = 1`, `num_dedicated_outputs = 0`. Load/drain cycles: 26/13.
- **Input config**: `config.yaml` (this folder, `name: dpe_r128`). Block names are **lowercase** because HSPICE folds netlist/measure names to lowercase in `.lis` output while COFFE looks them up with the configured case (an uppercase name breaks the sizing). Constants = the published TensorSlices COFFE artifact — identical across the whole sweep (see the flow `README.md`).
- **Launch**: `bash ../../run_dpe.sh 128` — config already present; runs `python2 coffe.py -a 1 -d 2 -hi 100 config.yaml` in the pristine COFFE clone (`launch.log` in this folder).
- **History**: run 2026-09-27 as tag `dpe22_p050` in `work/p050` (47 in / 45 out from `benchmarks/arch/proposed_auto.xml` `wc` tile) → renamed `work/dpe` (2026-09-28) → re-run 2026-09-30 after inert-key alignment (bit-identical numbers; raw pre-alignment evidence in `logs/sep_30/dpe/`) → renamed `work/dpe_R128_C64` with block name `dpe_r128` (2026-10-21, cosmetic — numbers/geometry unchanged). COFFE `/mnt/vault0/jiajunh5/coffe_pristine` @ `ecde3effd352e5b2ba89d17c1e4cbd620b369eca`, `ERF_MAX_ITERATIONS = 4`; original run metadata: `logs/06_main/dpe22_p050_run_meta.log`.

## Result breakdown (from `report.txt`)

| Quantity | Value |
|---|---|
| required / implemented mux | 24:1 (L1 = 6, L2 = 4) |
| muxes per tile | 47 |
| SRAM cells per mux | 10 |
| core area (`-hi`) | 0.0 µm² |
| pass network | 48.4871971812 µm² |
| select SRAM | 63.66432 µm² |
| **total (with select SRAM)** | **112.151517181 µm²** |

MWTA = `min_width_tran_area = 33864 nm²`; total = 3,311.821 MWTA.

## Notes

- `height: 16`, `clock_period: [3.0]` and `area_cost_exp: 1.0` are **inert** under `-hi` (COFFE never reads `height`; the clock and cost exponents are ASIC-flow only). They carry the TensorSlices artifact values so every sweep config and `work/tensorslice` differ **only in block identity** (`name`, `num_gen_inputs`) and output paths.
- The crossbar area depends only on `num_gen_inputs` (here set by the bandwidth rule from R) plus `crossbar_population`, `num_crossbars` and the process constants — it is independent of C and of the VTR grid footprint (COFFE never reads the VTR XML).
- This is the R=128 point of the sweep; the full 12-config table (all R×C, ratios vs `nl_dpe/area_power.py` logic area) is in `../sweep_summary.csv` / `../sweep_summary.md`.
