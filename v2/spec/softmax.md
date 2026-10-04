# Softmax — NL-DPE full-row operator (packed-window model)

**Status**: v0.1, 2026-09-27. Normative for `v2/sim/simulator/kernels/softmax_sim.py` and
`v2/rtl/softmax_top.v`. Supersedes the §6 streaming row-pipeline sketch that
lived in `pool_farm_model.md` (now `dimm_throughput_model.md`).

**Online schedule**: `softmax_online.md` (this doc's value contract on a blocked /
deferred-α machine) and `flash_attention.md` (FA composition). The value contract
here is unchanged; **S6 below is superseded**.

**Contracts above this doc**:

- Values: `v2/oracle/softmax_ref.py` (bit-exact; staged oracle
  `softmax_stage_values`, pass-count formula `packed_pass_counts`).
- Primitive timing: `v2/spec/dpe_nldpe.md` §5.3 —
  `T(p) = T_fill + (p−1)·T_steady` per crossbar pass.
- Primitive geometry: `I = min(R, C)` elements per identity pass (F5–F8),
  `BUF = 40` bit port (5 bytes/cycle), `P = 8`.

---

## §1 Operator & value contract

Per row (S a power of two, rows independent):

```
row_max          = max_j scores[r,j]
exp_input        = max(scores[r,j] - row_max, -128)      lower clamp only
exp_acam_output  = ACAM_EXP(exp_input)                   MODE_EXP identity pass
output_sum       = sum_j unsigned(exp_acam_output)
log_input        = min(output_sum >> log2 S, 127)
log_output       = ACAM_LOG(log_input)                   MODE_LOG identity pass
softmax_out[r,j] = clamp(scores[r,j] - row_max - log_output, -128, 127)
```

- Output is **log-domain** (`log p_i` approximation), consumed by the S×V DIMM
  input path; it is not comparable with a linear-domain softmax.
- Byte sums use the **unsigned** reading of the ACAM output (values are
  magnitudes in an 8-bit field).
- Values are invariant to `n_exp` / `n_log` and to window packing (F8): the
  oracles are the only value criterion.

## §2 Machine model (packed windows)

- Certified `dpe(R, C)` instances with identity eye; capacity
  `I = min(R, C)` elements per pass.
- **EXP**: `n_exp` crossbars; the stream is the `S×S` `exp_input` matrix read
  row-major (`S²` elements); packed stride-`I` windows, assigned round-robin
  (`window j → crossbar j % n_exp`); the tail window is zero-padded and its
  padding discarded.
- **LOG**: `n_log` crossbars; the stream is the `S` `lq` values (one per row),
  same packing and assignment.
- **CLB stages** are structural pipelines (throughput assumed replicated —
  they add latency, not a bottleneck):

| stage | latency/behavior |
|---|---|
| row max | streaming fold at `CLB_WIDTH = 32` B/cycle; row `r` usable at `L_max + ceil((r+1)·S/CLB_WIDTH)`, `L_max = log2 S + clb_pipe` |
| row sum | incremental over the EXP drain; `lq` usable `L_sum = log2 S + clb_pipe` after the row's last element drains |
| clamp | `1 + clb_pipe` after `ls` is readable |

- **Full in-flight overlap** (no artificial buffering). The RTL provisions the
  feed bandwidth (banked score buffers, one read port per EXP crossbar) so the
  windows are never stalled by the feeder.
- **Final result**: `softmax_out[r,j] = clamp(score − row_max[r] − ls[r])`,
  computed combinationally. The top exposes it as `out_addr → data_out` and
  does **not** drain/serialize it (no output cycles).

## §3 Pass counts & scheduling

```
passes_exp = ceil(S² / I)        passes_log = ceil(S / I)
xbar_passes_x = ceil(passes_x / n_x)      (round-robin)
```

- EXP window `g` issues once `rows_done >= ceil(min((g+1)·I, S²) / S)`.
- LOG window `w` issues once `lq_done_count >= min((w+1)·I, S)`.
- **`n_log` is a real split only when `passes_log > 1`** (i.e. `S > I`); for
  `S ≤ I` the single LOG window always lands on crossbar 0 and `n_log > 1` is
  an idle-resource axis.
- Schedule-injected counts: the simulator computes the counts it actually
  issues; the model never guesses them.

## §4 Cycle contract

- **Measured (contract)**: the fused simulator schedules the units above with
  the certified primitive event grammar and measures `compute_cycles` = the last
  normalizer row ready = **results ready (whole result computed)**. `dump_case`
  writes it; GATE 2 gates the RTL `t_done − t_start == compute_cycles` with
  `Δ_impl = 0`. The final result is read combinationally (`out_addr`/`data_out`)
  and is **not drained or timed** (`serialize_cycles = drain_cycles = 0`).
  `e2e_cycles = load_cycles + compute_cycles`
  (`load_cycles = ceil(S²/(BUF/8))`, the preload the later stages require).
- **Check model (corridor, not the contract)**: `softmax_cycle_model` is the
  phase envelope with CLB priced 0:

```
T_exp  = T(xbar_passes_exp)        T_log = T(xbar_passes_log)
T_start = completion of the EXP windows feeding LOG window 0
total  = max(T_exp, T_start + T_log)   (overlap)
       = T_exp + T_log                 (serial sanity check)
```

The self-test asserts the corridor
`compute_cycles ∈ [T_exp + L_max, serial + slack]` (per-word readiness can
beat the envelope; CLB latencies land above it).

## §5 RTL mapping (`v2/rtl/softmax_top.v`)

| module | role |
|---|---|
| `softmax_wprog` | identity-eye broadcast (`R·C` strobes) + `MODE_EXP`/`MODE_LOG` pins |
| `softmax_max_unit` | streaming row-max fold (`CLB_WIDTH`, `MAX_LAT = log2S+1`) + `rows_done` |
| `softmax_exp_feed` | one EXP crossbar: two-context window engine, fused `exp_input = score − row_max` with the −128 clamp |
| `softmax_sum_unit` | per-crossbar partial banks + combine pipeline → `lq` |
| `softmax_log_unit` | one LOG crossbar: window engine over the `lq` stream |
| `softmax_out_unit` | combinational final result `clamp(score − row_max − log)`; read via `out_addr`/`data_out`, no drain |
| `softmax_top` | buffers, instances, window issue, `done`, probes |

**Frozen probes** (verification only): `row_max_q[S]`, `exp_in_q[S²]`,
`exp_out_q[S²]`, `sum_q[S]`, `lq_q[S]`, `log_out_q[S]`, the `out_addr`/`data_out`
result read, and `u_max.rows_done`.

## §6 Decisions

| # | decision |
|---|---|
| S1 | Pass counts are schedule-injected (`SoftmaxPassPlan`); the ref exposes the formula (`packed_pass_counts`) |
| S2 | CLB stages are priced as **structural latency only** (max/sum `log2S + clb_pipe`, clamp `1 + clb_pipe`); throughput is assumed replicated |
| S3 | `done` = **results ready** (whole result computed: last normalizer row ready); the result is read combinationally and is **not drained or counted** (no `serialize` cycles); the load is reported for the end-to-end contract |
| S4 | `n_log` splits LOG windows only for `passes_log > 1`; `S ≤ I` makes it an idle axis |
| S5 | Crossbar geometry `(R, C)` is **not finalized**; the corpus sweeps `I` via a geometry axis (`R=C ∈ {64,128,256}`) |
| S6 | ~~Online/FlashAttention softmax out of scope~~ **superseded** by `softmax_online.md` (blocked schedule) and `flash_attention.md` (composition); value contract unchanged |

## §7 Output consumption (attention)

Softmax emits log-domain values and the S×V DIMM's attenuation-side producer is
softmax's output, so the S×V logB pool is not needed (log-domain fusion).
The `T_start`/producer model of `dimm.md` applies unchanged.

## §8 Verification status

- **GATE 1** (`dump_case`): per case, all eight staged values ≡
  `softmax_ref`, issued pass counts ≡ `packed_pass_counts`, cycles certified —
  before any expected file is written.
- **GATE 2** (`tb_softmax_top.v` + `run_softmax_rtl.py`): all seven probes
  bit-exact, `Δ_impl = 0`, `S²` result elements checked (combinational read).
- **Corpus**: 60 cases (`S ∈ {128,256}`, 5 classes, `n_exp ∈ {1,2,4}`,
  `n_log ∈ {1,2}`, `R=C=256`) + X2 corpus (`R=C ∈ {64,128}`,
  `n_log ∈ {1,2,4}`, `P_LOG > 1` split witness) = **67/67 PASS**
  (`test_softmax.py`).
- **2026-10-03 re-run**: main corpus **60/60 PASS, Δ_impl = 0** after the
  shared `softmax_sum_unit` gained `DIRECT_COMMIT` (0 for conventional — the
  direct-commit path is parameter-gated off here; inertness by re-run). The
  X2 corpus was **not** re-run after that edit: the direct-commit path is
  inert by gating and the touched logic (`softmax_sum_unit` banks) is
  `n_exp`-covered by the main corpus; the 2026-09-27 X2 66/66 certification
  stands.
- Sim self-test includes the `P_LOG > 1` invariants (values invariant,
  `T_log` splits with `n_log`).
