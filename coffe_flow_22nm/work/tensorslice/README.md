# Run — `tensorslice`

- **Result** (`report.txt` → `HARDBLOCK INFORMATION`): pass network **1728.31423479 µm²**, with select SRAM **2771.32543479 µm²** (core 0.0) — reproduces the published TensorSlices Tensor local-crossbar area (**2,771 µm²**, TRETS'22) to **+0.0117%**. This is the calibration witness of the flow.
- **Block**: TensorSlices Tensor block, **308 in / 298 out** (published Tensor slice); `crossbar_population = 0.50`, `num_crossbars = 1`, `num_dedicated_outputs = 0`. Same constants as `work/dpe_R128_C64` — only the block identity differs.
- **Input config**: `config.yaml` (this folder). Constants are the published TensorSlices COFFE artifact — see the flow `README.md`.
- **Launch**: `bash ../../run_tensorslice.sh` — runs `python2 coffe.py -a 1 -d 2 -hi 100 config.yaml` in the pristine COFFE clone and appends to `launch.log`.
- **Original run**: tag `tsmatch_cal`, 2026-09-27, executed in `validation/work/tsmatch_cal` before the 2026-09-28 layout rename; COFFE `/mnt/vault0/jiajunh5/coffe_pristine` @ `ecde3effd352e5b2ba89d17c1e4cbd620b369eca`, `ERF_MAX_ITERATIONS = 4`. Run metadata (original tag): `logs/05_tensorslice/tsmatch_cal_run_meta.log`. On 2026-09-30 the report header, the subcircuit-area row and all internal SPICE/sizing identifiers (plus the mux netlist filenames) were cosmetically normalized `tsmatch_cal` → `tensorslice`; numbers and geometry are unchanged, and the unmodified raw report is preserved in `logs/sep_30/tensorslice/`.

## Result breakdown (from `report.txt`)

| Quantity | Value |
|---|---|
| required / implemented mux | 154:1 (L1 = 13, L2 = 12) |
| muxes per tile | 308 |
| SRAM cells per mux | 25 |
| core area (`-hi`) | 0.0 µm² |
| pass network | 1728.31423479 µm² |
| select SRAM | 1043.0112 µm² |
| **total (with select SRAM)** | **2771.32543479 µm²** |

MWTA = `min_width_tran_area = 33864 nm²`.
