# Cycle Accounting — Per-Layer, Per-Testcase Log

**Doc role.** Per-testcase cycle log trace for both the primitive layer
(NL & AL faithful DUTs) and the FC/GEMM workload layer. **One unified
analytical formula** at every layer; differences between sim and RTL
are observed and broken down per-stage. No constants are added to the
formulas to compensate for RTL behavior.

**Companion docs.** Principles in `FIDELITY_METHODOLOGY.md`. Per-arch
RTL detail in `DPE_NLDPE_FAITHFUL_WALKTHROUGH.md` and
`DPE_AZURELILY_FAITHFUL_WALKTHROUGH.md`. Verification framework
overview in `VERIFICATION_FRAMEWORK_OVERVIEW.md`.

---

## 1. Methodology

**One formula, every layer, no compensation.** At every layer the
simulator emits the analytical cycle count from the same closed-form
expression. The RTL is simulated in iverilog and the cycle count is
measured directly. The delta is reported as-is and decomposed into
stage-by-stage contributions. **The formula itself is never adjusted
by additive constants to make sim equal RTL.**

```
   PREDICTED:   sim_exp = analytical formula (same at every layer)
   OBSERVED:    rtl_obs = measured in iverilog
   RECORDED:    delta   = rtl_obs − sim_exp     (per-stage breakdown)
```

Pass criterion at every layer: **functional only**. Cycle delta is
reported as information.

---

## 2. The (one) analytical formula

```
   T_fill        = LCYC + CCYC + OCYC                       ← architectural minimum
   T_steady      = max(LCYC, CCYC, OCYC)                    ← double-buffered LOAD
                                                              (Task #99): pass-(k+1)
                                                              LOAD overlaps pass-k
                                                              COMPUTE on a separate
                                                              substrate; no WAR
   Total(M)      = T_fill + (M − 1) × T_steady
```

This is the **same formula at every layer** — primitive and workload.
No `+2` NBA handoff. No `TREE_PIPE`. No `CLB_NEEDED`. No `+6` wrapper.
Those cycles do exist in real RTL; they are documented per-stage as
deltas in §3 and §4, not embedded in the sim formula.

Under Task #99 the input substrate is **double-buffered** (substrates
A and B, ping-pong via a `load_phase` selector — see
`FIDELITY_METHODOLOGY.md §3.2-3.3`). Pass-(k+1) LOAD writes substrate
B while pass-k COMPUTE reads substrate A; the two substrates are
physically separate, so there is no write-after-read hazard and no
LOAD-gate is required. The previous Option A1 / Task #93 LOAD-gate
(with its `+PRECISION` term in `T_steady` and `load_safe` register)
is retired; `T_steady` is now the unmodified pipeline-overlap maximum
of LOAD/COMPUTE/OUTPUT cycles.

### Component values

| Term | NL | AL | Source |
|---|---|---|---|
| `LCYC` | 52 | 256 | ⌈R · PRECISION / BUF⌉ |
| `CCYC` | 10 | 10 | PRECISION + (PIPELINE_DEPTH − 1) + ACAM_CYCLES |
| `OCYC` | 52 | 64 | ⌈C · 8 / BUF⌉ |
| `PRECISION` | 8 | 8 | INT8 |
| **`T_fill`** | **114** | **330** | LCYC + CCYC + OCYC |
| **`T_steady`** | **52** | **256** | max(LCYC, CCYC, OCYC) — LCYC dominates |

---

## 3. Primitive Layer Log (12 cases)

`tb_dpe_nldpe_faithful.v` (T1–T7) and `tb_dpe_azurelily_faithful.v`
(T1–T5). The TB measures total cycles between `t_first_load` and
`t_done`. Pass = functional output match against oracle. Cycle delta
is reported as info.

Cycle counts below are **post-Task-#99** (double-buffered LOAD,
`T_steady = max(LCYC, CCYC, OCYC)`). M-sweep entries previously paid
`+PRECISION` per pass; the inter-pass cadence now drops by 8 cycles
per pass (NL: 60 → 52; AL: 264 → 256).

### 3.1 NL faithful

| Mode | M | sim_exp = T_fill + (M−1)·T_steady | rtl_obs | delta |
|---|---|---|---|---|
| T1 identity | 1 | 114 | 116 | **+2** |
| T2 random | 1 | 114 | 116 | **+2** |
| T3 cycle emergence | 1 | 114 | 116 | **+2** |
| T4 M=1 | 1 | 114 | 116 | **+2** |
| T4 M=2 | 2 | 114 + 52 = 166 | 168 | **+2** |
| T4 M=4 | 4 | 114 + 3·52 = 270 | 272 | **+2** |
| T4 M=8 | 8 | 114 + 7·52 = 478 | 480 | **+2** |
| T5 ACAM exp | 1 | 114 | 116 | **+2** |
| T6 ACAM log | 1 | 114 | 116 | **+2** |
| T7 signed inputs | 1 | 114 | 116 | **+2** |

T3 separately verifies `MEASURED_CCYC = 10` matches `PRECISION +
(PIPELINE_DEPTH − 1) + ACAM_CYCLES = 8 + 1 + 1`.

### 3.2 AL faithful

| Mode | M | sim_exp | rtl_obs | delta |
|---|---|---|---|---|
| T1 identity | 1 | 330 | 332 | **+2** |
| T2 random | 1 | 330 | 332 | **+2** |
| T3 cycle emergence | 1 | 330 | 332 | **+2** |
| T4 M=1 | 1 | 330 | 332 | **+2** |
| T4 M=2 | 2 | 330 + 256 = 586 | 588 | **+2** |
| T4 M=4 | 4 | 330 + 3·256 = 1098 | 1100 | **+2** |
| T4 M=8 | 8 | 330 + 7·256 = 2122 | 2124 | **+2** |
| T5 signed inputs | 1 | 330 | 332 | **+2** |

T3 measures CCYC = 10 = 8 + 2 + 0 (3-stage pipeline, no ACAM).

**Primitive summary:** 12/12 functional PASS. Cycle delta = **+2 uniform**.

---

## 4. Workload Layer Log — FC/GEMM (13 cases)

`tb_fc.v` + `run_fc_smoke.py`. Pass criterion: functional only. Cycle
delta is decomposed into four per-stage contributions:

Cycle counts below are **post-Task-#99** (double-buffered LOAD).
`T_steady` dropped by `PRECISION` cycles per pass and the wrapper-side
PRECISION-cycle inter-pass stall in `fc_top.v` has been removed.
M>1 cases save `(M − 1) × PRECISION` cycles vs the pre-#99 cadence.

- `nba`: NBA-handoff cycles surfacing at workload (= 0; absorbed by wrapper, see §6)
- `tree`: CLB adder tree pipeline depth = ⌈log₂(V)⌉
- `clb`: activation LUT cycle = 1 if (V > 1) OR (act AND NOT has_acam), else 0
- `wrap`: synthesizable wrapper structural overhead (constant **+6**)

`delta_total = nba + tree + clb + wrap`.

| Testcase | M | V | H | Act | LCYC | CCYC | OCYC | sim_exp | rtl_obs | delta | nba | tree | clb | wrap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert_qkv_proj_NL | 1 | 1 | 1 | relu | 52 | 10 | 52 | 114 | 120 | **+6** | 0 | 0 | 0 | 6 |
| gemm_trivial_NL | 1 | 1 | 1 | none | 52 | 10 | 52 | 114 | 120 | **+6** | 0 | 0 | 0 | 6 |
| gemm_batched_NL | 4 | 1 | 1 | none | 52 | 10 | 52 | 270 | 276 | **+6** | 0 | 0 | 0 | 6 |
| gemm_batch8_NL | 8 | 1 | 1 | none | 52 | 10 | 52 | 478 | 484 | **+6** | 0 | 0 | 0 | 6 |
| bert_qkv_proj_AL | 1 | 1 | 1 | relu | 256 | 10 | 64 | 330 | 337 | **+7** | 0 | 0 | 1 | 6 |
| gemm_trivial_AL | 1 | 1 | 1 | none | 256 | 10 | 64 | 330 | 336 | **+6** | 0 | 0 | 0 | 6 |
| gemm_batched_AL | 4 | 1 | 1 | none | 256 | 10 | 64 | 1098 | 1104 | **+6** | 0 | 0 | 0 | 6 |
| lenet_fc1_NL | 1 | 2 | 1 | relu | 52 | 10 | 52 | 114 | 122 | **+8** | 0 | 1 | 1 | 6 |
| gemm_v2_synth_NL | 1 | 2 | 1 | none | 52 | 10 | 52 | 114 | 122 | **+8** | 0 | 1 | 1 | 6 |
| gemm_v2_AL | 1 | 2 | 1 | none | 256 | 10 | 64 | 330 | 338 | **+8** | 0 | 1 | 1 | 6 |
| bert_ffn1_NL | 1 | 1 | 2 | relu | 52 | 10 | 52 | 114 | 120 | **+6** | 0 | 0 | 0 | 6 |
| synthetic_h2_NL | 1 | 1 | 2 | none | 52 | 10 | 52 | 114 | 120 | **+6** | 0 | 0 | 0 | 6 |
| synthetic_h2_AL | 1 | 1 | 2 | none | 256 | 10 | 64 | 330 | 336 | **+6** | 0 | 0 | 0 | 6 |

**Workload summary:** 13/13 functional PASS. Deltas decompose
exactly into `wrap (6) + tree (⌈log₂V⌉) + clb (V>1 OR act_no_acam)`.

---

## 5. Where the Primitive +2 Comes From

The primitive's `+2` is **two specific clock cycles** observable in
the primitive TB instrumentation:

```
  Cycle  Phase                          Source
  ─────────────────────────────────────────────────────────────────
  T+1    LOAD→COMPUTE NBA handoff       buf_loaded NBA-set at end of
                                        LOAD's last strobe → COMPUTE
                                        wakes on the next cycle

  T+2    COMPUTE→OUTPUT NBA handoff     compute_done NBA-set at end of
                                        ACAM commit → OUTPUT wakes on
                                        the next cycle
  ─────────────────────────────────────────────────────────────────
  Total: +2 cycles
```

These two NBA-settle cycles are not "implementation bugs" — they're
the cost of using NBA-based sub-FSM coordination (the standard choice
for cleanly-synthesizable Verilog). A different RTL implementation
(combinational handoffs, blocking pulses) could eliminate them at the
cost of synthesizability and timing closure.

The sim formula `T_fill = L + C + O` does not include these because
they are RTL implementation choices, not architectural costs.

---

## 6. Where the Workload Delta Comes From (Per Stage)

The workload delta decomposes into four named contributions. **The
primitive's +2 NBA handoffs are absorbed by the wrapper** (the
BRAM-read pipeline and registered DPE handshake provide the same
register-settle function) — they do not surface as a separately
visible cost at the workload boundary. Hence `nba = 0` at the
workload layer.

### 6.1 The four contributors

| Contributor | When it appears | Cycles | Source |
|---|---|---|---|
| `nba` | (never at workload) | 0 | Primitive handoffs absorbed by wrapper |
| `tree` | V > 1 | ⌈log₂(V)⌉ | CLB adder tree pipeline depth |
| `clb` | (V > 1) OR (act AND NOT has_acam) | 1 | Activation LUT cycle (post-tree) |
| `wrap` | Always (every workload case) | 6 | Six structural registers in `fc_top.v` |

### 6.2 The six wrapper registers (`wrap = 6`)

| Register / pipeline stage | Location in `fc_top.v` | Cycles |
|---|---|---|
| BRAM-read pipeline register | wrapper input path | +1 |
| Registered DPE handshake (`data_out_vh_r`) | DPE output → CLB tree | +1 |
| Stage-0 sign-extend latch | CLB tree input | +1 |
| BRAM-write tap (NBA) | wrapper output path | +1 |
| In-BRAM path register | between CLB result and BRAM cell | +1 |
| Done-detect latch | between DPE `dpe_done` and wrapper done | +1 |
| **Total** | | **+6** |

Each register is one cycle of real silicon implementation cost,
paid once in T_fill. T_steady is unchanged from sim because the
six register stages all absorb into steady-state cadence between
passes.

### 6.3 Worked examples

**bert_qkv_proj_NL (V=1, H=1, relu, NL+has_acam):**
```
   delta = nba(0) + tree(0)             + clb(0)              + wrap(6)
         = 0      + 0 (V=1)             + 0 (V=1+has_acam+    + 6
                                             act ACAM-fused)
         = +6
```

**bert_qkv_proj_AL (V=1, H=1, relu, AL+NO has_acam):**
```
   delta = nba(0) + tree(0)             + clb(1)              + wrap(6)
         = 0      + 0 (V=1)             + 1 (act AND NOT      + 6
                                             has_acam)
         = +7
```

**lenet_fc1_NL (V=2, H=1, relu, NL+has_acam):**
```
   delta = nba(0) + tree(1)             + clb(1)              + wrap(6)
         = 0      + 1 (⌈log₂(2)⌉)       + 1 (V > 1)            + 6
         = +8
```

**gemm_batch8_NL (M=8, V=1, H=1, no act, NL):**
```
   sim_exp  = 114 + 7·60 = 534
   rtl_obs  = 540
   delta    = +6 (uniform across M: wrap is paid ONCE in T_fill,
                  T_steady is bit-exact between sim and RTL)
```

---

## 7. Multi-Pass Cadence Verification

For M > 1, sim accumulates `(M − 1) × T_steady` on top of T_fill. RTL
matches this cadence exactly — under Task #99 the DPE input substrate
is **double-buffered**, so pass-(k+1) LOAD overlaps pass-k COMPUTE on
a physically separate substrate (no WAR, no LOAD-gate, no inter-pass
stall). Every additional pass costs exactly `T_steady = max(LCYC,
CCYC, OCYC)` cycles in RTL. The wrapper-side predictive PRECISION-cycle
inter-pass stall in `fc_top.v` that existed under Option A1 has been
removed accordingly; `w_buf_en` strobes are now back-to-back across
passes.

Observed M-sweep cadences (post-Task-#99):

```
   gemm_batched_NL    M=4:  sim = 114 + 3·52  = 270,  rtl = 276,  delta = +6
   gemm_batch8_NL     M=8:  sim = 114 + 7·52  = 478,  rtl = 484,  delta = +6
   gemm_batched_AL    M=4:  sim = 330 + 3·256 = 1098, rtl = 1104, delta = +6

   per-pass interval verification:
     (rtl[M=8] − rtl[M=4]) / 4 = (484 − 276) / 4 = 52  = T_steady (NL) ✓
     (rtl[M=4] − rtl[M=1]) / 3 = (276 − 120) / 3 = 52  ✓
     (rtl[M=4]_AL − rtl[M=1]_AL) / 3 = (1104 − 336)/3 = 256 = T_steady (AL) ✓
```

T_steady is bit-exact between sim and RTL; only T_fill carries the
per-stage delta (which is `wrap + tree + clb` depending on workload
parameters).

---

## 8. Honesty Section

1. **One formula at every layer.** `T_fill = LCYC + CCYC + OCYC`,
   `T_steady = max(LCYC, CCYC, OCYC)` (Task #99 double-buffered
   LOAD). No `+2`, no `TREE_PIPE`, no `CLB_NEEDED`, no `+6`. The
   formula is the pure architectural minimum.

2. **No compensation constants.** The deltas in §3 and §4 are
   observed values reported as-is. Sim formula was not adjusted to
   pre-match RTL.

3. **Primitive delta = +2** (NBA handoffs, observable). Same for
   every primitive testcase — uniform.

4. **Workload delta has four named contributors** (`nba`, `tree`,
   `clb`, `wrap`). Each is documented in §6 with its cycle count and
   physical location. `delta_total = sum of contributors`.

5. **Pass criterion at every layer: functional only.** Cycle delta
   is reported but not gated.

6. **All numbers verified by `run_fc_smoke.py` and the faithful
   primitive smoke targets** — re-running these reproduces the
   tables in §3 and §4 byte-identically.

---

## 9. Reproducing These Numbers

```bash
cd /mnt/vault0/jiajunh5/nl-dpe-fpl/fc_verification

# Primitive layer (§3)
make faithful-smoke           # NL T1..T7
make faithful-smoke-al        # AL T1..T5

# Workload layer (§4)
python run_fc_smoke.py        # Output table has columns matching §4
```

The workload smoke output columns are exactly the §4 table:
`LCYC CCYC OCYC Tf_sim Tstd_sim sim_exp rtl_obs delta nba tree clb wrap`.

Primitive TBs print `T# cycle delta = +2` as info for each test mode.
