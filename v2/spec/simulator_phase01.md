# Implementation brief — backbone simulator phases 0–1

> **Superseded (2026-10-03) — historical record.** Phases 0–1 are built,
> but not as specified here: the files live in `v2/sim/simulator/`
> (`configs/{nl_dpe,azure_lily}.json`, `platforms.py` — renamed to avoid
> shadowing the standard `platform` module — and `cost.py`; layout pinned in
> `v2/spec/simulator.md` §13.0), and the config format and energy model were
> redesigned during implementation (shared crossbar + buffers, one output
> stage, per-unit costs, `nn/` granularity, no DAC). The as-built docstrings
> are the reference. Arithmetic slip noted while building: §4-B's
> `8 × 2.33 × 128` is 2385.92, not 2385.28.

**Status**: v1.0, 2026-10-03. Companion to `v2/spec/simulator.md` (the
charter, v0.2): this brief expands charter §13 phases 0–1 into
transcription-level detail so it can be implemented directly. Where this
brief and the charter conflict, **the charter wins**. Do not implement any
later phase from this brief; when phases 0–1 pass their gates, phase 2+
will get its own brief.

**What is being built.** Four new files, nothing else:

```
v2/spec/platform/nl_dpe.json       Deliverable 1
v2/spec/platform/azure_lily.json   Deliverable 2
v2/sim/platform.py                 Deliverable 3
v2/sim/cost.py                     Deliverable 4
```

They add the **energy axis** to v2 and make the **cycle law callable for
any tile** from one home. No workload, no adapter, no composition exists
yet — phases 0–1 are pure data + pure functions.

---

## §0 How to implement this

Style rules (match the existing v2 modules):

- Python 3, stdlib + numpy only; `from __future__ import annotations`;
  module docstring stating what the module is and is not.
- Sibling imports inside `v2/sim/` use
  `sys.path.insert(0, str(Path(__file__).resolve().parent))` then
  `import sibling_module as ...  # noqa: E402` — exactly as
  `v2/sim/simulator/kernels/dimm_sim.py:54-56` does.
- Every module ends with a self-test run under
  `if __name__ == "__main__":` whose last line prints
  `<name> self-test: ALL PASS` and exits non-zero on any failure. No
  `assert` inside a try/except that swallows it; the module aborts loudly.
- Floats: compute in float, **round only in the pin tests** (§5). Never
  round inside the model.
- No comments beyond the module docstring and short provenance pointers;
  the reasoning lives in the charter, not in code.

Hard prohibitions (SIM2):

- Do not edit, move, or reformat anything under `v2/oracle/`, `v2/tb/`,
  `v2/rtl/`, `v2/smoke/`, or `v2/sim/simulator/kernels/{nldpe,gemm,dimm,softmax,softmax_online,pass_engine}.py`.
  They are certified; read-only imports are allowed and encouraged.
- Do not touch anything related to the online softmax (its sim is still
  being debugged as of 2026-10-03).
- Do not commit anything; leave the new files in the working tree for
  review.
- Do not carry over any v1 constant not listed here (in particular the
  archived `capabilities.pipeline_depth` of 2 or 3 is v1-only: the cycle
  law's `P` comes from `input_bit_slices`, §4 below).

---

## §1 Deliverable 1: `v2/spec/platform/nl_dpe.json`

Strict JSON — no comments, no trailing commas, exact content:

```json
{
  "platform_name": "nl_dpe",
  "schema_version": 1,
  "fabric": {
    "clock_mhz": 300,
    "block_ram": {"port_width_bits": 40, "energy_pj_per_access": 0.0495},
    "clb": {
      "energy_pj_per_op_unit": 0.66,
      "add_pj": 0.08498,
      "compare_pj": 0.26439,
      "activation_unit_pj": 0.45,
      "lookup_table_read_pj": 2.64,
      "lookup_activity_factor": 1.0
    },
    "dsp": {"multiply_accumulate_pj": 1.2}
  },
  "compute_tile": {
    "crossbar": {"rows": 256, "columns": 256},
    "input_bit_slices": 8,
    "port_width_bits": 40,
    "energy_derivation_clock_mhz": 1000,
    "pass_pipeline": [
      {"stage": "analog_activation", "fires": "per_bit_slice",
       "energy_pj_per_fire": 3.89, "charge_scope": "whole_array"},
      {"stage": "conversion", "energy_pj": 0},
      {"stage": "output_stage", "kind": "acam", "fires": "per_pass",
       "energy_pj_per_column": 0.171445313, "charge_scope": "all_columns",
       "modes": ["identity", "relu", "exp", "log"]}
    ],
    "accumulator": {"kind": "in_tile", "fabric_charge_per_pass": 0}
  },
  "operator_realizations": {
    "linear_matmul": {"engine": "gemm_array"},
    "log_domain_matmul": {"engine": "dimm"},
    "exp": {"engine": "tile_pass", "mode": "exp"},
    "log": {"engine": "tile_pass", "mode": "log"},
    "softmax_normalize": {"engine": "log_domain"}
  },
  "resource_budgets": {
    "total_io": 2630,
    "total_clb": 6688,
    "total_dsp": 132,
    "total_mem": 264,
    "total_imc": 896,
    "total_softmax_lanes": 16
  }
}
```

Notes:
- `fabric.clock_mhz` (300) is the **system clock** of charter §3.6 — the
  only clock used for time conversion; `energy_derivation_clock_mhz`
  (1000, tile section) is the clock the energy constants were *derived*
  at, used only by the provenance check (§3).
- `pipeline_depth` from the archived file is intentionally absent:
  COMPUTE_CYC = `input_bit_slices` + 2 = 10 (charter §5.3, SIM16).
- The ACAM `modes` list is v2's certified mode set
  (`dpe_nldpe.md` §6). `energy_pj_per_column` is the frozen archived
  `params.e_digital_pj` byte-exact.

## §2 Deliverable 2: `v2/spec/platform/azure_lily.json`

Identical `fabric` section, byte for byte (the loader enforces this). Full
content:

```json
{
  "platform_name": "azure_lily",
  "schema_version": 1,
  "fabric": {
    "clock_mhz": 300,
    "block_ram": {"port_width_bits": 40, "energy_pj_per_access": 0.0495},
    "clb": {
      "energy_pj_per_op_unit": 0.66,
      "add_pj": 0.08498,
      "compare_pj": 0.26439,
      "activation_unit_pj": 0.45,
      "lookup_table_read_pj": 2.64,
      "lookup_activity_factor": 1.0
    },
    "dsp": {"multiply_accumulate_pj": 1.2}
  },
  "compute_tile": {
    "crossbar": {"rows": 512, "columns": 128},
    "input_bit_slices": 8,
    "port_width_bits": 16,
    "pass_pipeline": [
      {"stage": "analog_activation", "energy_pj": 0},
      {"stage": "conversion", "kind": "adc",
       "conversions": "per_bit_slice_per_column",
       "energy_pj_per_column": 2.33, "charge_scope": "active_columns"},
      {"stage": "output_stage", "kind": "none"}
    ],
    "accumulator": {"kind": "fabric_shift_add", "fabric_charge_per_pass": 0}
  },
  "operator_realizations": {
    "linear_matmul": {"engine": "gemm_array"},
    "log_domain_matmul": {"engine": "unavailable"},
    "exp": {"engine": "fabric", "lookup_table_reads": 1},
    "log": {"engine": "fabric", "lookup_table_reads": 1},
    "softmax_normalize": {"engine": "fabric", "lookup_table_reads": 1,
                          "dsp_multiplies": 1},
    "attention_score_matmul": {"engine": "dsp_mac_lanes"}
  },
  "resource_budgets": {
    "total_io": 2630,
    "total_clb": 6688,
    "total_dsp": 16,
    "total_mem": 264,
    "total_imc": 896,
    "total_softmax_lanes": 16
  }
}
```

Notes:
- No `energy_derivation_clock_mhz`: the 2.33 pJ/column conversion is a
  frozen archived value, not derivable from any power model in this
  repo — there is **no Azure-Lily derivation check** (charter §3.4).
- The archived `scale_with_geometry` flag is gone: 2.33 states its
  per-column meaning directly (charter §3.2 row 1).
- `accumulator.fabric_charge_per_pass = 0` is the archived convention;
  raising it is OPEN decision SIM12 — do not "fix" it here.

## §2b Where every constant came from (provenance appendix)

| v2 constant | archived source |
|---|---|
| `fabric.clock_mhz` 300 | `fpga_specs.freq`, both files |
| block RAM port 40 | `fpga_specs.bram_width` (40 in both) |
| 0.0495 pJ/access | `fpga_specs.bram_pj_per_access` |
| CLB unit 0.66 | `fpga_specs.clb_pj_per_mac` |
| DSP 1.2 | `fpga_specs.dsp_pj_per_mac` |
| activation 0.45 | `fpga_specs.act_energy_pj_per_op` |
| add 0.08498 | archived `nn/constant.py` measured 84.98358e-6 nJ |
| compare 0.26439 | same source, 793.1801e-6 nJ / 3 |
| LUT read 2.64 | softmax study: 256×8-bit table = 32 LUT6 = 4 CLB units |
| NL `e_analoge` 3.89 | archived `params.e_analoge_pj` |
| NL `e_digital` 0.171445313 | archived `params.e_digital_pj` |
| NL conversion 0 | archived `params.e_conv_pj` |
| AL conversion 2.33 | archived `params.e_conv_pj` |
| AL activation 0 | archived `params.e_analoge_pj` |
| AL output stage none | archived `acam_rows: 0` |
| ACAM modes | `dpe_nldpe.md` §6 |
| budgets | `fpga_specs.total_*`; **only `total_dsp` differs** (NL 132, AL 16) |

Archived files (read-only provenance):
`archive/azurelily_simulator/IMC/configs/{nl_dpe,azure_lily}.json`.

## §3 Deliverable 3: `v2/sim/platform.py`

Loader + validation + the derivation check. No imports beyond stdlib.

### Types

```python
@dataclass
class Platform:
    name: str                       # "nl_dpe" | "azure_lily"
    path: Path
    fabric: dict                    # §1/§2 fabric section
    compute_tile: dict              # §1/§2 compute_tile section
    operator_realizations: dict
    resource_budgets: dict          # informational (SIM8); the cost model
                                    # never reads it
    @property
    def rows(self) -> int ...          # compute_tile.crossbar.rows
    @property
    def columns(self) -> int ...
    @property
    def port_width_bits(self) -> int ...
    @property
    def input_bit_slices(self) -> int ...
    @property
    def system_clock_mhz(self) -> int ...   # fabric.clock_mhz
    def tile(self) -> dict ...         # convenience: compute_tile
```

Dicts, not nested dataclasses: validation guarantees their shape, and the
pass-pipeline walk (`cost.tile_pass_energy`) is the only structural
consumer. Keep the platform loader thin.

### Functions

- `load_platform(name: str) -> Platform` — reads
  `v2/spec/platform/<name>.json` (path anchored:
  `Path(__file__).resolve().parents[1] / "spec" / "platform"`); strict
  `json.load` (a comment or NaN rejects the file by itself); validates
  (below); returns the `Platform`.
- `load_all() -> dict[str, Platform]` — loads every `*.json` in that
  directory, then asserts the **fabric sections are equal across all
  files** (plain `==` on the parsed dicts). This check also guarantees
  both platform files always exist and parse.
- `_check_platform(p: Platform) -> None` — the validation rules:
  1. top-level keys are exactly `{platform_name, schema_version, fabric,
     compute_tile, operator_realizations, resource_budgets}`;
     `schema_version == 1`; `platform_name` matches the filename stem.
  2. fabric keys are exactly `{clock_mhz, block_ram, clb, dsp}`;
     `block_ram` = `{port_width_bits, energy_pj_per_access}`; `clb` = the
     six keys shown in §1; `dsp` = one key; every energy value a positive
     number.
  3. compute_tile keys are exactly `{crossbar, input_bit_slices,
     port_width_bits, pass_pipeline, accumulator}` plus
     `energy_derivation_clock_mhz` **iff** platform is `nl_dpe`;
     `crossbar` = `{rows, columns}`; pass_pipeline has exactly three
     stages in order `analog_activation, conversion, output_stage`.
  4. stage vocabulary (closed):
     - `analog_activation`: either the zero form `{"stage": ...,
       "energy_pj": 0}` (Azure-Lily) or `{fires: "per_bit_slice",
       energy_pj_per_fire, charge_scope: "whole_array"}` (NL).
     - `conversion`: zero form (NL) or `{kind: "adc",
       conversions: "per_bit_slice_per_column", energy_pj_per_column,
       charge_scope: "active_columns"}` (AL).
     - `output_stage`: `{kind: "none"}` (AL) or `{kind: "acam",
       fires: "per_pass", energy_pj_per_column,
       charge_scope: "all_columns", modes}` (NL) with
       `modes == ["identity", "relu", "exp", "log"]`.
     `charge_scope` always present where energy is charged, drawn from
     `{whole_array, all_columns, active_columns}`.
  5. `accumulator`: `{kind: "in_tile" | "fabric_shift_add",
     fabric_charge_per_pass: 0}`.
  6. `operator_realizations`: required operator set —
     `nl_dpe`: `{linear_matmul, log_domain_matmul, exp, log,
     softmax_normalize}`; `azure_lily`: those plus
     `attention_score_matmul`. Closed engine vocabulary
     `{gemm_array, dimm, tile_pass, log_domain, fabric, dsp_mac_lanes,
     unavailable}`; `tile_pass` requires `mode ∈ {exp, log, identity,
     relu}`; `fabric` entries carry `lookup_table_reads ≥ 1`.
  7. One entry in `pass_pipeline` carries the frozen value
     `0.171445313` byte-exact on `nl_dpe` (guards against transcription
     drift).
- `derived_energy_constants(row: int, col: int, freq_ghz: float) -> dict`
  — the NL power model, transcribed from `nl_dpe/area_power.py:111-133`
  (cite that path in a short comment; do **not** import it — the DSE-era
  module stays untouched and v2 stays self-contained):

  ```python
  buffer_size  = max(row, col)
  crossbar_p   = 1.31 * (row / 256) * (col / 256)          # mW
  acam_p       = 43.52 / 256 * col
  inbuf_p      = 0.1406 / 256 * buffer_size
  outbuf_p     = 0.1406 / 256 * buffer_size
  dac_p        = 2.44 * (col * 4) / 1024
  xor_p        = (7 * col) / (7 * 256) * 0.235
  p_analogue_mw = crossbar_p + dac_p + inbuf_p
  p_digital_mw  = acam_p + outbuf_p + xor_p
  e_analogue_pj = p_analogue_mw / freq_ghz                 # per VMM row activation
  e_digital_pj  = (p_digital_mw / freq_ghz) / col          # per ACAM column per pass
  ```

  Return dict with both e_* values (and the two mW values for the print).
  This is the **provenance check only** — no cost function ever calls it
  (charter §3.4: the frozen file value is the truth).

### Self-test (`python3 v2/sim/platform.py`)

1. `load_all()` succeeds; fabric sections equal across the two files.
2. Derivation check at row=256, col=256, freq_ghz=1.0 (the tile's
   `energy_derivation_clock_mhz` / 1000):
   - frozen `e_analogue` 3.89:
     derived = **3.8906**; drift **0.0154%**; assert drift ≤ 0.02% and
     `round(derived, 2) == 3.89`.
   - frozen `e_digital` 0.171445313:
     derived = **0.171467188**; drift **0.0128%**; assert drift ≤ 0.02%
     (charter §3.4 logs the 0.013% figure; the drift lands within
     tolerance either way).
   - Print both derived values and drifts — the reviewer reads them.
3. Both files validate; NL's frozen 0.171445313 present; AL's
   `log_domain_matmul.engine == "unavailable"`; AL carries no derivation
   clock; NL budgets differ from AL in `total_dsp` only (132 vs 16).

Success line: `platform self-test: ALL PASS`.

## §4 Deliverable 4: `v2/sim/cost.py`

Pure cost functions of (platform/fabric data + counts). Imports: stdlib
(`math`) + the sibling cycle law:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_sim  # noqa: E402  (read-only, certified)
```

`p` (pipeline depth) for the cycle law is the tile's
`input_bit_slices` — **not** the archived `pipeline_depth` (§0).

### Functions (signatures exact)

```python
def tile_pass_energy(tile: dict, active_columns: int | None = None) -> dict
```
Walks `tile["pass_pipeline"]` in order; returns
`{"analog_activation": float, "conversion": float, "output_stage": float,
"crossbar_total": float}`. Stage rules — each stage charges only what its
object says (no hidden scaling, SIM3):
- `analog_activation` zero form → 0.0; otherwise
  `fires × energy_pj_per_fire` where `fires == "per_bit_slice"`
  means count = `tile["input_bit_slices"]`; `charge_scope =
  "whole_array"` → no column scaling ever.
- `conversion` zero form → 0.0; `kind == "adc"` with
  `conversions == "per_bit_slice_per_column"` →
  `input_bit_slices × energy_pj_per_column × n_active` where
  `n_active = active_columns if given else columns` (`charge_scope =
  "active_columns"`: only producing columns are charged).
- `output_stage` `kind == "none"` → 0.0; `kind == "acam"` with
  `fires == "per_pass"` →
  `1 × energy_pj_per_column × columns` — **all columns, never
  `active_columns`** (`charge_scope = "all_columns"`; the whole-array
  policy, charter §4.2: this is why identity packing divides per-element
  cost).
`crossbar_total` = sum of the three stage values.

```python
def pass_port_accesses(tile: dict) -> tuple[int, int]
```
`(in, out) = (ceil(rows*8/port), ceil(columns*8/port))` — one access per
port cycle; independent of `active_columns` (the input burst is always a
full row).

```python
def pass_traffic_energy(fabric: dict, tile: dict) -> float
```
`(in + out) × fabric.block_ram.energy_pj_per_access`.

```python
def bram_accesses(fabric: dict, n_bytes: int) -> int        # ceil(n_bytes / (port_width/8))
def bram_energy(fabric: dict, n_bytes: int) -> float        # accesses × per-access pJ
def fabric_add(fabric: dict, n: int) -> float               # n × 0.08498
def fabric_compare(fabric: dict, n: int) -> float           # n × 0.26439
def fabric_activation(fabric: dict, n: int) -> float        # n × 0.45
def dsp_mac(fabric: dict, n: int) -> float                  # n × 1.2
def lookup_table_read(fabric: dict, n: int) -> float        # n × 2.64 × activity factor
def weight_setup_energy(fabric: dict, tile: dict, n_crossbars: int) -> float
                                                            # bram_energy(rows*cols*n_crossbars) — setup, §4.5
```

```python
def tile_cycle_law(tile: dict, passes: int) -> nldpe_sim.CycleModel
```
Exactly `nldpe_sim.cycle_model(passes, tile["crossbar"]["rows"],
tile["crossbar"]["columns"], tile["input_bit_slices"],
tile["port_width_bits"])`; assert `passes >= 1`. One law, no duplication
(SIM16): every tile — NL or AL — obeys the certified v2 law, with
`LOAD = ceil(R·8/port)`, `COMPUTE = P+2`, `OUTPUT = ceil(C·8/port)`,
`T_fill`, `T_steady = max(LOAD+P, COMPUTE, OUTPUT+1)`.

Expected results of that delegation (the §5.3 table):

| tile | LOAD | COMPUTE | OUTPUT | T_fill | T_steady |
|---|---:|---:|---:|---:|---:|
| NL 256×256, port 40 | 52 | 10 | 52 | 114 | 60 |
| AL 512×128, port 16 | 256 | 10 | 64 | 330 | 264 |

### Self-test (`python3 v2/sim/cost.py`)

All pins are `abs(computed − frozen) ≤ tol` on 2-dp rounded values
(charter §9.1 convention). Pins:

**A. per-pass, NL tile.** `tile_pass_energy(nl_tile)` (active_columns
None ⇒ all columns):
- stages: `analog_activation == 31.12` (8 × 3.89) exactly;
  `conversion == 0.0`; `output_stage == 43.89` (tol 0.01 —
  0.171445313 × 256 = 43.890000128); `crossbar_total == 75.01` (tol
  0.01).
- `pass_port_accesses == (52, 52)`; `pass_traffic_energy == 5.15`
  (tol 0.01; exact 5.148); per-pass total `80.16` (tol 0.01; exact
  80.158000128).
- At `active_columns = 128` the **output stage is unchanged** (43.89 —
  whole-array policy) and the conversion stage is still 0: a partial
  pass costs the same. Assert it.
- 128-column check: build `{"rows": 256, "columns": 128, ...}` and
  assert `crossbar_total == 53.07` (tol 0.01) and
  `pass_port_accesses == (52, 26)`, traffic `3.86` (tol 0.01),
  per-pass total `56.93` (tol 0.01). (These reproduce the softmax
  study's printed numbers — charter §10.1.)

**B. per-pass, Azure-Lily tile.** `tile_pass_energy(al_tile)` with
`active_columns = 128`: conversion `2385.28` (tol 0.01), analog 0,
output 0, crossbar total `2385.28`; `pass_port_accesses == (256, 64)`;
traffic `15.84` (tol 0.01); per-pass total `2401.12` (tol 0.01;
charter §10.1 says ≈ 2401.1). One line, the whole story: the ADC
dominates the AL pass (`≈ 2.4 nJ`) vs the ACAM 75.01 pJ — at
**identical cycles** (check D).

**C. the DIMM worked example (charter §10.2)** — the §10.2 row
recomputes exactly from `cost` functions; sizes
`M = N = 128, K = 64`; the pass count is **schedule-injected**
(4,160 = 32 logA + 32 logB + 4,096 exp farm; an input to this test,
never recomputed from idealized packing):

```python
crossbar = 4160 * tile_pass_energy(nl_tile)["crossbar_total"]   # pin 312,041.6
adds     = fabric_add(fabric, 128 * 128 * 64)                   # pin 89,108.0
in_acc   = 4160 * 52                       # 216,320  (pass input traffic)
out_acc  = 4160 * 52                       # 216,320  (pass output traffic)
parked   = bram_accesses(fabric, 64 * (128 + 128))    # 3,277
ser      = bram_accesses(fabric, 128 * 128 * 4)       # 13,108  (int32 words)
bram     = (in_acc + out_acc + parked + ser) * 0.0495 # pin 22,226.7  (449,025 accesses)
total    = crossbar + adds + bram                     # pin 423,376.3
```
Pins (tol 0.05 each): crossbar 312,041.6; adds 89,108.0; bram
22,226.7; **total 423,376.3 ≈ 423.4 nJ**; and
`in_acc + out_acc + parked + ser == 449025` exactly.

**D. cycle-law delegation.** `tile_cycle_law` on both platform tiles:
NL gives `load_cyc 52, compute_cyc 10, output_cyc 52, t_fill 114,
t_steady 60`; AL gives `256, 10, 64, 330, 264` — exact equality. This
is the check that the v2 law covers both platforms through ONE code
path (SIM16).

**E. dual-compute guard** (house style): recompute each pin with
standalone formulas written directly in the self-test (e.g.
`8*3.89 + 0.171445313*256`, `8*2.33*128`, `ceil` traffic arithmetic)
and assert equality against the file-walk results to 1e-9 — the file
walk and the charter's tabulation must agree.

Success line: `cost self-test: ALL PASS`.

## §5 Gain/loss ledger (what this design deliberately costs)

- Frozen copies duplicate the archive's values: if the archive changes,
  v2 does not follow (provenance lives here and in the charter). The
  derivation check (§3) is the alarm against transcription drift, but
  only for NL's two tile constants.
- Zero-charge items stay zero (SIM12's shift-add, write pulses, static):
  the ledger in charter §4.6 is authoritative; phases 0–1 implement the
  zeros as data, and only a charter decision may change them.
- The conversion stage charges `charge_scope = "active_columns"` while
  the ACAM charges "all_columns": that asymmetry is the archived
  semantics, faithfully transcribed — a partial AL pass is cheaper by
  columns, a partial NL pass is not. Do not "unify" them here.

## §6 Gates

| gate | check | pass |
|---|---|---|
| A | Deliverables 1–2 byte-equal the §1/§2 blocks | reviewer eyeball |
| B | `python3 v2/sim/platform.py` | ends `platform self-test: ALL PASS`; drift prints ≤ 0.02% |
| C | `python3 v2/sim/cost.py` | ends `cost self-test: ALL PASS`; pin A–E green |
| — | forbid-list | no file outside the four deliverables modified (verify with `git status --short`) |

After gate C, stop and hand back for review. Next phase (adapters) gets
its own brief.
