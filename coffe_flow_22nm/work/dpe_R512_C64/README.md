# Run — `dpe_R512_C64`

- **Result** (`report.txt` → `HARDBLOCK INFORMATION`): pass network **560.157879135 µm²**, with select SRAM **429.801888 µm²** — total **989.959767135 µm²** (core 0.0) — the local input crossbar footprint at **R=512** (B_in = 40×512/128 = 160 bits/cycle).
- **Block identity** (bandwidth model): `B_in = 40×R/128 = 160` → `num_gen_inputs = 167`; `B_out = 40×C/64 = 40` (+5 status) → `num_gen_outputs = 45` (**inert**). `crossbar_population = 0.50`, `num_crossbars = 1`, `num_dedicated_outputs = 0`. Load/drain cycles: 26/13.
- **Input config**: `config.yaml` (`name: dpe_r512`, lowercase per the HSPICE case-folding constraint); generated from the R128 seed by `run_dpe.sh` — constants identical across the sweep.
- **Launch**: `bash ../../run_dpe.sh 512`.
- **Run**: 2026-10-21 (parallel with R1024), COFFE `/mnt/vault0/jiajunh5/coffe_pristine` @ `ecde3effd352e5b2ba89d17c1e4cbd620b369eca`, `ERF_MAX_ITERATIONS = 4`, `-a 1 -d 2 -hi 100`; `launch.log` in this folder.

## Result breakdown (from `report.txt`)

| Quantity | Value |
|---|---|
| required / implemented mux | 84:1 → 90:1 (L1 = 10, L2 = 9) |
| muxes per tile | 167 |
| SRAM cells per mux | 19 |
| core area (`-hi`) | 0.0 µm² |
| pass network | 560.157879135 µm² |
| select SRAM | 429.801888 µm² |
| **total (with select SRAM)** | **989.959767135 µm²** |

## Notes

- Sized transistor widths are identical to the other sweep runs (pass W=1, restorer W=1, inv_1 nmos 2 / pmos 1), so the per-device unit costs match: bare = `167·[(90+9+1)·0.966 + 2.4503]` MWTA exactly; SRAM = `167·19·4·MWTA`.
- This is the measured R=512 point; ratios for every (R,C) pairing are in `../sweep_summary.csv` / `../sweep_summary.md`.
