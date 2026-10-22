# Run — `dpe_R1024_C64`

- **Result** (`report.txt` → `HARDBLOCK INFORMATION`): pass network **1963.29535858 µm²**, with select SRAM **1151.646912 µm²** — total **3114.94227058 µm²** (core 0.0) — the local input crossbar footprint at **R=1024** (B_in = 40×1024/128 = 320 bits/cycle).
- **Block identity** (bandwidth model): `B_in = 40×R/128 = 320` → `num_gen_inputs = 327`; `B_out = 40×C/64 = 40` (+5 status) → `num_gen_outputs = 45` (**inert**). `crossbar_population = 0.50`, `num_crossbars = 1`, `num_dedicated_outputs = 0`. Load/drain cycles: 26/13.
- **Input config**: `config.yaml` (`name: dpe_r1024`, lowercase per the HSPICE case-folding constraint); generated from the R128 seed by `run_dpe.sh` — constants identical across the sweep.
- **Launch**: `bash ../../run_dpe.sh 1024`.
- **Run**: 2026-10-21 (parallel with R512), COFFE `/mnt/vault0/jiajunh5/coffe_pristine` @ `ecde3effd352e5b2ba89d17c1e4cbd620b369eca`, `ERF_MAX_ITERATIONS = 4`, `-a 1 -d 2 -hi 100`; `launch.log` in this folder.

## Result breakdown (from `report.txt`)

| Quantity | Value |
|---|---|
| required / implemented mux | 164:1 → 168:1 (L1 = 14, L2 = 12) |
| muxes per tile | 327 |
| SRAM cells per mux | 26 |
| core area (`-hi`) | 0.0 µm² |
| pass network | 1963.29535858 µm² |
| select SRAM | 1151.646912 µm² |
| **total (with select SRAM)** | **3114.94227058 µm²** |

## Notes

- Sized transistor widths are identical to the other sweep runs (pass W=1, restorer W=1, inv_1 nmos 2 / pmos 1), so the per-device unit costs match: bare = `327·[(168+12+1)·0.966 + 2.4503]` MWTA exactly; SRAM = `327·26·4·MWTA`.
- This is the measured R=1024 point; ratios for every (R,C) pairing are in `../sweep_summary.csv` / `../sweep_summary.md`.
