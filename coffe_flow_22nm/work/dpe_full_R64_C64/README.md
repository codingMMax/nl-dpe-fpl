# Run — `dpe_full_R64_C64`

- **Result** (`report.txt` → `HARDBLOCK INFORMATION`): pass network **105.762965274 µm²**, with select SRAM **115.408512 µm²** — total **221.171477274 µm²** (core 0.0) — the local input crossbar footprint at **R=64** under the row-parallel interface (64 data pins + 7 controls).
- **Block identity** (row-parallel model): `num_gen_inputs = R + 7 = 71`; `num_gen_outputs = C + 5 = 69` (**inert** — COFFE never reads it). `crossbar_population = 0.50`, `num_crossbars = 1`, `num_dedicated_outputs = 0`. Load/drain = 8/8 cycles (int8 bit-serial).
- **Input config**: `config.yaml` (`name: dpe_full_r64`, lowercase per the HSPICE case-folding constraint); generated from the 47-pin seed by `run_dpe.sh` — constants identical across the sweep.
- **Launch**: `bash ../../run_dpe.sh 64`.
- **Run**: 2026-10-22, COFFE `/mnt/vault0/jiajunh5/coffe_pristine` @ `ecde3effd352e5b2ba89d17c1e4cbd620b369eca`, `ERF_MAX_ITERATIONS = 4`, `-a 1 -d 2 -hi 100`; `launch.log` in this folder.

## Result breakdown (from `report.txt`)

| Quantity | Value |
|---|---|
| required / implemented mux | 36:1 (L1 = 6, L2 = 6) |
| muxes per tile | 71 |
| SRAM cells per mux | 12 |
| core area (`-hi`) | 0.0 µm² |
| pass network | 105.762965274 µm² |
| select SRAM | 115.408512 µm² |
| **total (with select SRAM)** | **221.171477274 µm²** |

## Notes

- Sized widths are identical to the other full-R runs (pass W=1, restorer W=1, inv_1 nmos 2 / pmos 1), so bare = `71·[(36+6+1)·0.966 + 2.4503]` MWTA exactly; SRAM = `71·12·4·MWTA`.
- Ratios vs the DPE logic area for every (R,C) pairing: `../sweep_summary_full.csv` / `../sweep_summary_full.md`.
