#!/usr/bin/env python3
"""softmax_sim.py — NL conventional softmax behavior + cycle simulator.

Models `v2/sim/rtl/softmax_top.v` (charter `v2/sim/spec/softmax.md`). The
simulator is the golden model of the RTL: values bit-exact against the oracle
`v2/sim/oracle/softmax_ref.py` (GATE 1), measured cycles equal to the RTL's
(GATE 2, Delta_impl = 0).

Machine model (certified (R, C) identity crossbars; capacity I = min(R, C)
elements per identity pass; port width BUF, EPS = BUF/8 bytes/cycle):

    A  row max, computed while the scores stream in (softmax_loadmax)
    B  EXP: the exp_input matrix, S^2 elements, converted by n_exp crossbars
       over packed stride-I windows (window j -> crossbar j mod n_exp)
    Cs unsigned output sum -> lq = min(sum >> log2 S, 127)  (softmax_row_acc)
    C  LOG: the S row sums converted by one crossbar window (S <= I)
    D  softmax_out = clamp(scores - row_max - log_output), read through a
       registered result port (not timed: `done` = whole result computed)

Cycles are measured from the TB's `start` edge (0) to the `done` edge; the score
load is setup, the TB raising `start` one edge after the last load word
(`load_gap = 1`). Crossbar passes are placed with the certified pass grammar
(`pass_engine.schedule_pass_sequence`); the CLB stages add structural offsets,
each named after the RTL register stages it counts:

  MAX_LAT   = 3  last load word -> row max of the last row visible
                 (softmax_loadmax: lane-max stage, running max, table write)
  ISSUE_LAT = 3  go visible -> producer issue -> FIFO -> first dpe pop
  LOG_LAT   = 5  last row-sum drain word -> first LOG pop (row accumulator
                 stage A + event, lq write + count, lq_all, issue + FIFO, pop)
  DONE_LAT  = 2  last LOG drain word -> done (last_row register, done register)

  EXP_READY = max(start, last_word + MAX_LAT) + ISSUE_LAT   (all windows ready at
              one edge); the last row of a window sits in drain word WPW - 1,
              WPW = ceil(min(S^2, I) / EPS)
  LOG window ready = last row word + LOG_LAT; done = LOG drain_start +
              WL - 1 + DONE_LAT, WL = ceil(S / EPS)

Domain (the RTL's elaboration guards, enforced here too): S, I = min(R, C), n_exp
powers of two; 8 <= S <= I <= 1024; BUF = 40. (S > I — several LOG windows — is
outside the RTL's domain.)

Value contract : `v2/sim/oracle/softmax_ref.py` (bit-exact; imported as ref).
Run:  python3 v2/sim/simulator/kernels/softmax_sim.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make sibling modules importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "oracle"))
import nldpe_sim as prim  # noqa: E402
import nldpe_ref as nref  # noqa: E402
import softmax_ref as ref  # noqa: E402
from dimm_sim import identity_pass  # noqa: E402
from pass_engine import identity_extent, packed_windows, schedule_pass_sequence  # noqa: E402

# Structural offsets of softmax_top.v (see the module docstring).
MAX_LAT, ISSUE_LAT, LOG_LAT, DONE_LAT = 3, 3, 5, 2
LOAD_GAP = 1          # TB raises `start` this many edges after the last load word


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def check_domain(S: int, R: int, C: int, BUF: int, n_exp: int) -> None:
    """The RTL's elaboration-time guards."""
    I = min(R, C)
    pow2 = lambda x: x > 0 and x & (x - 1) == 0  # noqa: E731
    if not (pow2(S) and pow2(I) and pow2(n_exp)):
        raise ValueError("S, I = min(R, C), n_exp must be powers of two")
    if not 8 <= S <= I or I > 1024:
        raise ValueError(f"need 8 <= S <= I <= 1024 (S={S}, I={I})")
    if BUF != 40:
        raise ValueError("BUF must be 40 (EPS = 5)")


# ---------------------------------------------------------------------------
# Cycle model
# ---------------------------------------------------------------------------
@dataclass
class SoftmaxPassPlan:
    """Pass counts the schedule issues (checked against `ref.packed_pass_counts`)."""

    passes_exp: int     # packed EXP windows = ceil(S^2 / I)
    passes_log: int     # packed LOG windows = ceil(S / I) = 1


@dataclass
class SoftmaxCycleModel:
    """Measured-cycle model of softmax_top (start edge = 0 -> done edge)."""

    S: int
    R: int
    C: int
    BUF: int
    P: int
    I: int                  # min(R, C): elements per identity pass
    n_exp: int              # EXP crossbars (windows round-robin)
    n_log: int              # LOG crossbars (only the first carries the window)
    passes_exp: int         # packed EXP windows
    passes_log: int         # packed LOG windows (1)
    exp_passes_per_xbar: list
    exp_ready: int          # edge at which every EXP window is ready
    last_row_word: int      # edge of the last row-sum drain word
    log_ready: int          # edge at which the LOG window is ready
    total: int              # `done` edge = compute cycles
    events: list = field(default_factory=list)   # (unit, crossbar, passes, load_start, drain_end)
    pass_extent: str = "full"   # "full" (certified, RTL-witnessed) | "used" (P29: I x I passes, RTL revision pending)


def softmax_cycle_model(S: int, R: int = 256, C: int = 256, BUF: int = 40, P: int = 8,
                        n_exp: int = 1, n_log: int = 1,
                        load_gap: int = LOAD_GAP, *,
                        pass_extent: str = "full") -> SoftmaxCycleModel:
    """Cycles of the softmax_top RTL for one run (data-independent).

    `pass_extent` (keyword-only, dpe_nldpe.md P29): "full" (default) = the certified
    RTL schedule; "used" = every EXP / LOG identity pass loads and drains only
    I = min(R, C) (model-level only until the RTL revision).
    """
    check_domain(S, R, C, BUF, n_exp)
    rows, cols = identity_extent(R, C, pass_extent)
    I = min(R, C)
    SQ = S * S
    NWEL = min(SQ, I)
    PEXP = _cdiv(SQ, I)
    WPW, WL = _cdiv(NWEL, BUF // 8), _cdiv(S, BUF // 8)
    last_word = -load_gap
    exp_ready = max(0, last_word + MAX_LAT) + ISSUE_LAT
    last_row_word = 0
    passes, events = [], []
    for c in range(n_exp):
        n = len(range(c, PEXP, n_exp))
        passes.append(n)
        if n:
            ev = schedule_pass_sequence([exp_ready] * n, R, C, BUF, P, rows=rows, cols=cols)
            last_row_word = max(last_row_word, ev[-1]["drain_start"] + WPW - 1)
            events.append(("exp", c, n, ev[0]["load_start"], ev[-1]["drain_end"]))
    log_ready = last_row_word + LOG_LAT
    log_ev = schedule_pass_sequence([log_ready], R, C, BUF, P, rows=rows, cols=cols)[-1]
    events.append(("log", 0, 1, log_ev["load_start"], log_ev["drain_end"]))
    total = log_ev["drain_start"] + WL - 1 + DONE_LAT
    return SoftmaxCycleModel(S=S, R=R, C=C, BUF=BUF, P=P, I=I, n_exp=n_exp, n_log=n_log,
                             passes_exp=PEXP, passes_log=1, exp_passes_per_xbar=passes,
                             exp_ready=exp_ready, last_row_word=last_row_word,
                             log_ready=log_ready, total=total, events=events,
                             pass_extent=pass_extent)


# ---------------------------------------------------------------------------
# Packed-window value conversion (packing + timing live in `pass_engine`).
# ---------------------------------------------------------------------------
def _convert_packed_stream(flat: np.ndarray, mode: int, R: int, C: int,
                           BUF: int, P: int, crossbars: list,
                           return_y: bool = False) -> tuple:
    """Convert a flat stream through identity crossbars, packed windows.

    Values only (timing lives in the cycle model). Returns
    (values, passes_per_crossbar) — plus the pre-ACAM wide output `y` when
    `return_y=True` — with values in original element order.
    """
    flat = np.asarray(flat, dtype=np.int8).reshape(-1)
    I = min(R, C)
    _, window_lo, window_len, per_crossbar = packed_windows(
        flat.size, I, len(crossbars))

    values = np.empty(flat.size, dtype=np.int8)
    y_values = np.empty(flat.size, dtype=np.int32) if return_y else None
    passes_per_crossbar = []
    for c, window_ids in enumerate(per_crossbar):
        if not window_ids:
            passes_per_crossbar.append(0)
            continue
        chunk = np.concatenate([flat[window_lo[j]:window_lo[j] + window_len[j]]
                                for j in window_ids])
        if return_y:
            converted, pass_count, y = identity_pass(
                chunk, mode, R, C, BUF, P, dpe=crossbars[c], return_y=True)
        else:
            converted, pass_count = identity_pass(
                chunk, mode, R, C, BUF, P, dpe=crossbars[c])
        passes_per_crossbar.append(pass_count)
        offset = 0
        for j in window_ids:
            length = int(window_len[j])
            values[window_lo[j]:window_lo[j] + length] = \
                converted[offset:offset + length]
            if return_y:
                y_values[window_lo[j]:window_lo[j] + length] = \
                    y[offset:offset + length]
            offset += length
    if return_y:
        return values, passes_per_crossbar, y_values
    return values, passes_per_crossbar


@dataclass
class SoftmaxStages:
    """Staged diagnostics for one softmax run (`collect_stages=True`).

    Stage order mirrors the datapath: row max -> subtract/clamp -> EXP crossbar
    y (int32, pre-ACAM) -> ACAM output -> output sum / log input -> LOG ACAM
    output -> final softmax output.
    """

    row_max: np.ndarray          # int8  [S]     row maximum
    exp_input: np.ndarray        # int8  [S, S]  max(scores - row_max, -128)
    exp_crossbar_y: np.ndarray   # int32 [S, S]  EXP crossbar y (pre-ACAM; == exp_input)
    exp_acam_output: np.ndarray  # int8  [S, S]  ACAM_EXP output
    output_sum: np.ndarray       # int64 [S]     unsigned exp-output row sums
    log_input: np.ndarray        # int8  [S]     min(output_sum >> log2 S, 127)
    log_output: np.ndarray       # int8  [S]     ACAM_LOG(log_input)
    softmax_out: np.ndarray      # int8  [S, S]  log-domain output


@dataclass
class SoftmaxResult:
    """Everything the self-test / later cross-checks consume."""

    softmax_out: np.ndarray         # int8 [S, S] log-domain output
    used_cycles: int                # == compute_cycles (the output read is not timed)
    compute_cycles: int             # `done` edge (whole result computed)
    drain_cycles: int               # 0: no output drain is timed
    passes_exp: int                 # issued EXP window passes (all crossbars)
    passes_log: int                 # issued LOG window passes
    cycle_model: SoftmaxCycleModel
    stages: SoftmaxStages | None = None
    timeline: list = field(default_factory=list)  # (unit, crossbar, passes, start, end)


class NldpeSoftmax:
    """NL conventional softmax on NL-DPE primitives (packed-window simulator).

    Values are produced through the certified primitive (identity crossbars,
    packed windows) and checked per stage against `softmax_ref` by the
    self-test and `dump_case`; cycles come from `softmax_cycle_model`.
    """

    def __init__(self, S: int, R: int = 256, C: int = 256, BUF: int = 40,
                 P: int = 8, n_exp: int = 1, n_log: int = 1) -> None:
        assert n_exp >= 1 and n_log >= 1
        check_domain(S, R, C, BUF, n_exp)
        self.S, self.R, self.C, self.BUF, self.P = S, R, C, BUF, P
        self.I = min(R, C)
        self.n_exp, self.n_log = n_exp, n_log

        # Resources: n_exp EXP + n_log LOG identity crossbars (certified `dpe`).
        identity_eye = np.eye(R, C, dtype=np.int8)
        self._exp_crossbars = []
        for _ in range(n_exp):
            crossbar = prim.NldpeDpe(R, C, BUF, P)
            crossbar.program_weights(identity_eye)
            self._exp_crossbars.append(crossbar)
        self._log_crossbars = []
        for _ in range(n_log):
            crossbar = prim.NldpeDpe(R, C, BUF, P)
            crossbar.program_weights(identity_eye)
            self._log_crossbars.append(crossbar)

    def cycle_model(self) -> SoftmaxCycleModel:
        return softmax_cycle_model(self.S, self.R, self.C, self.BUF, self.P,
                                   self.n_exp, self.n_log)

    def run(self, scores: np.ndarray,
            collect_stages: bool = False) -> SoftmaxResult:
        """Run softmax on int8 scores [S, S] -> SoftmaxResult.

        Values must equal `softmax_ref.softmax_pass_view(scores)`; the staged
        dumps (when collected) must equal `softmax_ref.softmax_stage_values`
        bit-exactly.
        """
        scores = np.asarray(scores, dtype=np.int8)
        assert scores.shape == (self.S, self.S), \
            f"scores must be [{self.S}, {self.S}]"
        S = self.S
        mean_shift = S.bit_length() - 1
        scores_int16 = scores.astype(np.int16)

        # A — row max.
        row_max = scores.max(axis=1)
        exp_input = np.maximum(scores_int16 - row_max[:, None].astype(np.int16),
                               -128).astype(np.int8)

        # B — packed EXP conversion.
        exp_acam_flat, exp_passes, exp_y_flat = _convert_packed_stream(
            exp_input.reshape(-1), prim.MODE_EXP, self.R, self.C, self.BUF,
            self.P, self._exp_crossbars, return_y=True)
        exp_acam_output = exp_acam_flat.reshape(S, S)
        exp_crossbar_y = exp_y_flat.reshape(S, S)

        # Cs — unsigned output sum -> log input.
        output_sum = exp_acam_output.view(np.uint8).astype(np.int64).sum(axis=1)
        log_input = np.minimum(output_sum >> mean_shift, 127).astype(np.int8)

        # C — packed LOG conversion.
        log_output, log_passes = _convert_packed_stream(
            log_input, prim.MODE_LOG, self.R, self.C, self.BUF, self.P,
            self._log_crossbars)

        # D — log-domain output.
        softmax_out = np.clip(
            scores_int16 - row_max[:, None].astype(np.int16)
            - log_output[:, None].astype(np.int16),
            -128, 127).astype(np.int8)

        cm = self.cycle_model()
        plan = SoftmaxPassPlan(passes_exp=int(sum(exp_passes)),
                               passes_log=int(sum(log_passes)))
        stages = (SoftmaxStages(row_max=row_max, exp_input=exp_input,
                                exp_crossbar_y=exp_crossbar_y,
                                exp_acam_output=exp_acam_output,
                                output_sum=output_sum, log_input=log_input,
                                log_output=log_output, softmax_out=softmax_out)
                  if collect_stages else None)
        return SoftmaxResult(softmax_out=softmax_out,
                             used_cycles=cm.total, compute_cycles=cm.total,
                             drain_cycles=0,
                             passes_exp=plan.passes_exp, passes_log=plan.passes_log,
                             cycle_model=cm, stages=stages, timeline=cm.events)

    # -- stimulus dump (GATE 1 + GATE 2 expected files) ---------------------
    def dump_case(self, case_dir: Path, scores: np.ndarray) -> None:
        """Write stimulus + expected files for the softmax RTL cross-check.

        GATE 1: certify this exact case against `softmax_ref` (all eight
        stages) and the packed pass counts BEFORE any file is written; an
        uncertified case is never dumped.

        File contract (consumed by `v2/sim/tb/tb_softmax_top.v`, expanded by
        `v2/sim/smoke/run_softmax_rtl.py`):
          scores.mem            — S*S int8 row-major, packed BUF-wide (10-hex)
          expected_rowmax.mem   — S int8 (2-hex/line)
          expected_expin.mem    — S*S int8  (crossbar feed input)
          expected_expout.mem   — S*S int8  (ACAM_EXP output)
          expected_sum.mem      — S int32 (8-hex/line)
          expected_lq.mem       — S int8
          expected_logout.mem   — S int8   (ACAM_LOG output)
          expected_out.mem      — S*S int8 (softmax_out stream order)
          case.json             — params + pass counts + measured cycles
        """
        scores = np.asarray(scores, dtype=np.int8)
        assert scores.shape == (self.S, self.S), \
            f"scores must be [{self.S}, {self.S}]"
        result = self.run(scores, collect_stages=True)
        stages = result.stages
        expected = ref.softmax_stage_values(scores)

        # -- GATE 1: per-case oracle certification -------------------------
        for key in _STAGE_KEYS:
            got, want = getattr(stages, key), expected[key]
            if got.shape != want.shape or not np.array_equal(got, want):
                raise RuntimeError(
                    f"GATE 1: sim/oracle stage '{key}' mismatch "
                    f"— case not dumped")
        expected_passes = ref.packed_pass_counts(self.S, self.R, self.C)
        issued_passes = (result.passes_exp, result.passes_log)
        if issued_passes != expected_passes:
            raise RuntimeError(
                f"GATE 1: issued passes {issued_passes} != schedule formula "
                f"{expected_passes} — case not dumped")

        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        # scores.mem — packed BUF-wide stream words
        with open(case_dir / "scores.mem", "w") as f:
            for word in nref.pack_act_stream(scores.reshape(-1)):
                f.write(f"{word:010x}\n")

        def write_int8(name: str, values: np.ndarray) -> None:
            with open(case_dir / name, "w") as f:
                for v in np.asarray(values).reshape(-1):
                    f.write(f"{int(v) & 0xFF:02x}\n")

        def write_int32(name: str, values: np.ndarray) -> None:
            with open(case_dir / name, "w") as f:
                for v in np.asarray(values).reshape(-1):
                    f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")

        write_int8("expected_rowmax.mem", stages.row_max)
        write_int8("expected_expin.mem", stages.exp_input)
        write_int8("expected_expout.mem", stages.exp_acam_output)
        write_int32("expected_sum.mem", stages.output_sum)
        write_int8("expected_lq.mem", stages.log_input)
        write_int8("expected_logout.mem", stages.log_output)
        write_int8("expected_out.mem", stages.softmax_out)

        words = len(nref.pack_act_stream(scores.reshape(-1)))
        case = {
            "S": self.S, "R": self.R, "C": self.C, "BUF": self.BUF,
            "P": self.P, "n_exp": self.n_exp, "n_log": self.n_log,
            "passes_exp": int(result.passes_exp),
            "passes_log": int(result.passes_log),
            "used_cycles": int(result.compute_cycles),   # done = results ready
            "compute_cycles": int(result.compute_cycles),
            "drain_cycles": 0,                           # output not counted
            "load_cycles": int(words),
            "serialize_cycles": 0,                       # output not counted
            "e2e_cycles": int(words + result.compute_cycles),
            "load_words": int(words),
            "weight_cycles": int(self.R * self.C),
        }
        with open(case_dir / "case.json", "w") as f:
            json.dump(case, f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/simulator/kernels/softmax_sim.py
# Gates: per-stage values bit-exact vs the staged oracle, pass counts vs the
# schedule formula, count-invariance, RTL-measured cycle pins, structural
# sanity of the cycle model, edge classes, dual witnesses, domain guards.
# ---------------------------------------------------------------------------
_STAGE_KEYS = ("row_max", "exp_input", "exp_crossbar_y", "exp_acam_output",
               "output_sum", "log_input", "log_output", "softmax_out")

# Cycles measured on the RTL (strict GATE 2, Delta_impl = 0), data-independent:
# (S, n_exp) -> done edge at R = C = 256.
_RTL_MEASURED_PINS = {
    (8, 1): 149,
    (32, 1): 373, (32, 2): 253, (32, 4): 193, (64, 1): 1099, (64, 2): 619, (64, 4): 379,
    (128, 1): 3992, (128, 2): 2072, (128, 4): 1112,
    (256, 1): 15538, (256, 2): 7858, (256, 4): 4018,
}


def _check_stages(result: SoftmaxResult, expected_stages: dict,
                  label: str) -> None:
    stages = result.stages
    assert stages is not None, f"{label}: stages not collected"
    for key in _STAGE_KEYS:
        got, want = getattr(stages, key), expected_stages[key]
        if got.shape != want.shape or not np.array_equal(got, want):
            raise AssertionError(f"{label}: stage '{key}' mismatch")


def _self_test() -> None:
    rng = np.random.default_rng(6)

    # ── Stage gates + count-invariance (same scores per S) ─────────────────
    for S, count_configs in ((8, [(1, 1), (2, 1)]), (16, [(1, 1), (4, 1)]),
                             (32, [(1, 1), (2, 1), (4, 2)]),
                             (128, [(1, 1), (2, 1), (4, 2), (2, 3)]),
                             (256, [(1, 1), (2, 2)])):
        scores = rng.integers(-128, 128, size=(S, S), dtype=np.int8)
        expected_stages = ref.softmax_stage_values(scores)
        expected_passes = ref.packed_pass_counts(S, 256, 256)
        outputs = []
        for n_exp, n_log in count_configs:
            unit = NldpeSoftmax(S=S, n_exp=n_exp, n_log=n_log)
            result = unit.run(scores, collect_stages=True)
            label = f"S={S} n_exp={n_exp} n_log={n_log}"
            _check_stages(result, expected_stages, label)
            assert np.array_equal(result.softmax_out, ref.softmax_pass_view(scores)), label
            assert (result.passes_exp, result.passes_log) == expected_passes, label
            assert result.used_cycles == result.compute_cycles and result.drain_cycles == 0
            assert result.cycle_model.total == result.compute_cycles
            outputs.append(result.softmax_out)
        for other in outputs[1:]:
            assert np.array_equal(other, outputs[0]), f"S={S}: crossbar-count invariance"
        print(f"  [stages] S={S}: 8 stages vs oracle, invariance over {count_configs}: OK")
    for (S, n_exp), pin in _RTL_MEASURED_PINS.items():
        got = softmax_cycle_model(S, n_exp=n_exp).total
        assert got == pin, f"S={S} n_exp={n_exp}: model {got} != RTL {pin}"
    print(f"  [pins] RTL-measured cycles matched for {len(_RTL_MEASURED_PINS)} (S, n_exp) points: OK")

    # ── Cycle-model structure ───────────────────────────────────────────────
    for S in (8, 16, 32, 64, 128, 256):
        prev = None
        for ne in (1, 2, 4, 8):
            cm = softmax_cycle_model(S, n_exp=ne)
            assert cm.total > cm.log_ready > cm.last_row_word > cm.exp_ready, (S, ne)
            assert sum(cm.exp_passes_per_xbar) == cm.passes_exp == _cdiv(S * S, min(256, 256))
            if prev is not None:
                assert cm.total <= prev, f"S={S}: more EXP crossbars must not slow the run"
            prev = cm.total
        assert softmax_cycle_model(S, n_exp=2, n_log=3).total == softmax_cycle_model(S, n_exp=2).total, \
            "idle LOG crossbars cannot change the timing"
    print("  [cycle model] ordering of the stage edges, monotone in n_exp, idle n_log inert: OK")

    # ── Edge classes (routing exercised: n_exp=2, n_log=2) ─────────────────
    tie_row = np.full(128, 5, dtype=np.int8)
    single_max = np.zeros((128, 128), dtype=np.int8)
    single_max[:, 7] = 127
    edges = {
        "extremes": np.array([[-128, 127] * 64, [127, -128] * 64] * 64, dtype=np.int8),
        "uniform": np.zeros((128, 128), dtype=np.int8),
        "ties": np.tile(tie_row, (128, 1)),
        "single-max": single_max,
    }
    for name, edge_scores in edges.items():
        result = NldpeSoftmax(S=128, n_exp=2, n_log=2).run(edge_scores, collect_stages=True)
        _check_stages(result, ref.softmax_stage_values(edge_scores), name)
    print("  [edges] extremes uniform ties single-max: stages OK")

    # ── Dual witnesses ──────────────────────────────────────────────────────
    witness_scores = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)
    result = NldpeSoftmax(S=128).run(witness_scores)
    assert np.array_equal(result.softmax_out, ref.softmax_pass_view(witness_scores)), "pass view"
    assert np.array_equal(result.softmax_out, ref.softmax_elementwise_view(witness_scores)), \
        "elementwise dual view"
    print("  [witnesses] softmax_out == softmax_pass_view == softmax_elementwise_view: OK")

    # ── Domain guards (the RTL cannot run these) ────────────────────────────
    for bad in (dict(S=4), dict(S=512), dict(S=24), dict(S=128, R=64, C=64),
                dict(S=64, n_exp=3), dict(S=64, BUF=64)):
        try:
            NldpeSoftmax(**bad)
        except ValueError:
            continue
        raise AssertionError(f"domain guard missed {bad}")
    print("  [domain] S < 8, S > 1024, non-power-of-two S / n_exp, S > I, BUF != 40 rejected: OK")

    _test_pass_extent()
    print("softmax_sim self-test: ALL PASS")


def _test_pass_extent() -> None:
    """P29 extent: default == certified; 'used' == the same run on an I x I array; never slower."""
    for R, C in [(64, 64), (128, 64), (128, 256), (256, 256), (512, 128), (1024, 64)]:
        I = min(R, C)
        S = 8
        while S <= I:
            for n_exp in (1, 2, 4):
                full = softmax_cycle_model(S, R, C, n_exp=n_exp)
                assert full == softmax_cycle_model(S, R, C, n_exp=n_exp, pass_extent="full")
                used = softmax_cycle_model(S, R, C, n_exp=n_exp, pass_extent="used")
                small = softmax_cycle_model(S, I, I, n_exp=n_exp)
                assert used.total == small.total and used.total <= full.total
                assert (used.passes_exp, used.exp_passes_per_xbar) == (full.passes_exp, full.exp_passes_per_xbar)
                if R == C:
                    assert used.total == full.total
            S *= 2
    try:
        softmax_cycle_model(16, pass_extent="half")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown pass_extent must be rejected")
    print("  [extent] P29 'used' == I x I array, default unchanged, never slower: OK")


if __name__ == "__main__":
    _self_test()
