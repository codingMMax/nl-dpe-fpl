#!/usr/bin/env python3
"""softmax_sim.py — NL safe-softmax behavior simulator (SCAFFOLD — behavior TODO(you)).

Machine model: **streaming row-pipeline** (distinct from the DIMM pool/farm
model; see `v2/spec/pool_farm_model.md` §6). Rows are fed continuously through
dedicated port-rate converters and overlap, so the binding rate is the
5 elem/cycle/port feed — not the pass-gated primitive interval.

```
16 row-parallel lockstep lanes, lane k owns rows {k, k+16, ...} (RPL = S/16)
  A : max tree (WPR)
  B : (x-max) clamp, ACAM EXP via n_exp DPEs, 5 elem/cycle/port (LCYC)
  Cs: unsigned byte sum (CLB) + shared log DPE(s) (n_log) -> ls = trunc8(lq-1)
  D : clamp(x-max-ls) -> log-domain int8 (WPR)
  steady = max(WPR, LCYC)
  fill   = (WPR+4) + (LCYC+10+LCYC+2) + 20 + WPR + 4
  total  = fill + (RPL-1) * steady
```

Locked anchors (SOFTMAX_STUDY §4, measured RTL): 290 (S=128, n_exp=1),
514 (S=256, n_exp=2), 956 (S=256, n_exp=1).

Value contract : `v2/oracle/softmax_ref.py` (bit-exact; imported as ref)
Cycle contract : `SoftmaxCycleModel` / `softmax_cycle_model()` in this file

Your behavior implementation must make the gated self-test pass: values equal
the oracle, measured cycles equal the shadow.

Run:  python3 v2/sim/softmax_sim.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make the sibling oracle module importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
import softmax_ref as ref  # noqa: E402


@dataclass
class SoftmaxCycleModel:
    """Locked NL streaming cycle form (spec §6 / SOFTMAX_STUDY §4).

    Anchors: S=128 -> 290, S=256 (n_exp=2) -> 514, S=256 (n_exp=1) -> 956.
    """

    lanes: int      # W = 16
    rpl: int        # rows per lane = S / lanes
    wpr: int        # words per row (lane-wide CLB stage width)
    n_exp: int      # exp DPEs per lane = ceil(S / C)
    n_log: int      # shared log DPEs converting all lane sums
    lcyc: int       # ceil((S/n_exp)/5) — exp DPE pass latency at port rate
    fill: int       # (wpr+4) + (lcyc+10+lcyc+2) + 20 + wpr+4
    steady: int     # max(wpr, lcyc)
    total: int      # fill + (rpl - 1) * steady


def softmax_cycle_model(S: int, n_exp: int = 1, lanes: int = 16,
                        C: int | None = None, n_log: int = 1) -> SoftmaxCycleModel:
    """NL locked streaming closed form (spec §6 / SOFTMAX_STUDY §4)."""
    assert S % lanes == 0, "S must divide evenly across lanes"
    if C is not None and n_exp != -(-S // C):
        raise ValueError(f"n_exp={n_exp} != ceil(S/C)={-(-S // C)}")
    rpl = S // lanes
    wpr = S // lanes
    e_per_dpe = -(-S // n_exp)
    lcyc = -(-e_per_dpe // 5)
    steady = max(wpr, lcyc)
    fill = (wpr + 4) + (lcyc + 10 + lcyc + 2) + 20 + wpr + 4
    return SoftmaxCycleModel(
        lanes=lanes, rpl=rpl, wpr=wpr, n_exp=n_exp, n_log=n_log, lcyc=lcyc,
        fill=fill, steady=steady, total=fill + (rpl - 1) * steady,
    )


@dataclass
class SoftmaxResult:
    """Everything the self-test / later cross-checks consume."""

    out: np.ndarray                # int8 [S, S] log-domain output
    used_cycles: int               # measured; must equal SoftmaxCycleModel.total
    fill: int
    steady: int
    timeline: list = field(default_factory=list)  # stage events (free-form)


class NldpeSoftmax:
    """NL safe-softmax on NL-DPE primitives (streaming W-lane row pipeline)."""

    def __init__(self, S: int, C: int = 128, lanes: int = 16,
                 n_exp: int | None = None, n_log: int = 1) -> None:
        assert S % lanes == 0, "S must divide evenly across lanes"
        self.S, self.C, self.lanes = S, C, lanes
        self.n_exp = n_exp if n_exp is not None else -(-S // C)
        # n_log: shared log DPEs converting all lane sums. Pool/farm reading:
        # exp:log work = S:1, so the study's n_log=1 is over-provisioned by
        # S/(16*n_exp) — it never binds (spec §6).
        self.n_log = n_log
        # TODO(you): instantiate the streaming pipeline resources:
        #   - per lane: n_exp identity-programmed DPEs (MODE_EXP)
        #   - shared  : n_log identity-programmed DPEs (MODE_LOG)
        #   - per-lane score buffers and the CLB max/sum/clamp trees

    def shadow_cycles(self) -> SoftmaxCycleModel:
        """Cycle contract for this instance's geometry (scaffold-owned)."""
        return softmax_cycle_model(self.S, self.n_exp, self.lanes,
                                   n_log=self.n_log)

    def run(self, scores: np.ndarray) -> SoftmaxResult:
        """Run safe softmax on int8 scores [S, S] -> SoftmaxResult.

        Values must equal `softmax_ref.safe_softmax_nl(scores)`; used_cycles
        must equal `self.shadow_cycles().total`.
        """
        x = np.asarray(scores, dtype=np.int8)
        assert x.shape == (self.S, self.S), f"scores must be [{self.S}, {self.S}]"

        # ------------------------------------------------------------------
        # TODO(you) 1 — row max (stage A)
        #   16-wide CLB max tree per lane; lane k owns rows {k+16*i}.
        # ------------------------------------------------------------------
        # TODO(you) 2 — subtract + ACAM EXP (stage B)
        #   d = max(x - max, -128) into the n_exp streaming DPEs (MODE_EXP);
        #   port rate 5 elem/cycle -> LCYC = ceil((S/n_exp)/5) per row.
        # ------------------------------------------------------------------
        # TODO(you) 3 — unsigned byte sum (stage Cs)
        #   CLB reduction tree per lane; row sums feed the shared log DPE(s).
        # ------------------------------------------------------------------
        # TODO(you) 4 — ACAM LOG (shared DPE)
        #   lq = min(sum >> log2(S), 127); ls = trunc8(lq - 1) (MODE_LOG).
        # ------------------------------------------------------------------
        # TODO(you) 5 — log-domain output (stage D)
        #   D = clamp(x - max - ls, -128, 127) int8. Rows are pipelined:
        #   row r+1's A/B overlaps row r's Cs/D (streaming model, spec §6).
        # ------------------------------------------------------------------
        # TODO(you) 6 — cycle engine
        #   Reproduce the stage model: fill and steady from
        #   `self.shadow_cycles()`; measured total must equal the shadow.
        # ------------------------------------------------------------------
        raise NotImplementedError(
            "TODO(you): implement softmax behavior blocks 1-6 in run()"
        )


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/softmax_sim.py
# Gated: reports shadow-only until the behavior TODO blocks are implemented,
# so it can never print a false ALL PASS.
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # ── Locked streaming anchors (SOFTMAX_STUDY §4) ────────────────────────
    assert softmax_cycle_model(128, n_exp=1).total == 290
    assert softmax_cycle_model(256, n_exp=2).total == 514
    assert softmax_cycle_model(256, n_exp=1).total == 956

    rng = np.random.default_rng(6)
    x = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)
    expected = ref.safe_softmax_nl(x)

    sm = NldpeSoftmax(S=128, C=128)
    try:
        res = sm.run(x)
    except NotImplementedError as err:
        print(f"softmax_sim self-test: shadow PASS; behavior TODO — {err}")
        return

    # Values ≡ oracle.
    assert res.out.dtype == np.int8 and res.out.shape == (128, 128)
    assert np.array_equal(res.out, expected), "softmax values != softmax_ref"

    # Cycles ≡ shadow.
    shadow = sm.shadow_cycles()
    assert res.used_cycles == shadow.total, (
        f"measured {res.used_cycles} != shadow {shadow.total}"
    )

    print("softmax_sim self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
