#!/usr/bin/env python3
"""pass_engine.py — shared timing primitives for the v2 kernel simulators.

Provisional API (revisit after the softmax RTL flow). Three primitives, so
every kernel packs work and schedules crossbar passes identically:

  packed_windows          stride-I packed windows, round-robin over crossbars
  schedule_pass_sequence  certified DPE pass event grammar, with an optional
                          input-readiness gate per pass
  crossbar_total          T(p) = T_fill + (p-1)*T_steady (check-model helper)

Witness: the event grammar is what `nldpe_sim.NldpeDpe.run_workload` measures,
gated against the primitive RTL at Stage 1 (GATE 2, Δ_impl = 0); window packing
is the operator pass layer (F5–F8). No values live here — kernels keep their
own value paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the sibling primitive simulator importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_sim as prim  # noqa: E402


def crossbar_total(passes: int, R: int, C: int, P: int, BUF: int, *,
                   rows: int | None = None, cols: int | None = None) -> int:
    """Primitive-layer T(p) for one crossbar (check-model helper).

    T(p) = T_fill + (p - 1) * T_steady, read from the certified primitive
    cycle model; 0 passes -> 0 cycles. `rows` / `cols`: the P29 pass extent
    (default R / C = the certified full load; see `nldpe_sim.cycle_model`).
    """
    if passes <= 0:
        return 0
    return int(prim.cycle_model(passes, R, C, P, BUF, rows=rows, cols=cols).total)


def identity_extent(R: int, C: int, pass_extent: str) -> tuple:
    """(rows, cols) streamed by an identity (conversion) pass under a kernel's pass extent.

    "full" -> (None, None): the certified full load (R rows in, C columns out).
    "used" -> (I, I), I = min(R, C): only the identity block is loaded and drained (P29).
    """
    if pass_extent == "full":
        return None, None
    if pass_extent == "used":
        I = min(R, C)
        return I, I
    raise ValueError(f"pass_extent must be 'full' or 'used', got {pass_extent!r}")


def packed_windows(n_elements: int, I: int, n_crossbars: int) -> tuple:
    """Packed stride-I windows, round-robin over n_crossbars.

    Window j covers flat elements [j*I, min((j+1)*I, n_elements)); the tail
    window may be partial (zero-padded by the crossbar, padding discarded).
    Returns (n_windows, window_lo, window_len, per_crossbar_window_ids).
    """
    n_windows = -(-n_elements // I) if n_elements else 0
    window_lo = np.zeros(n_windows, dtype=np.int64)
    window_len = np.zeros(n_windows, dtype=np.int64)
    per_crossbar = [[] for _ in range(n_crossbars)]
    for j in range(n_windows):
        lo = j * I
        window_lo[j] = lo
        window_len[j] = min(I, n_elements - lo)
        per_crossbar[j % n_crossbars].append(j)
    return n_windows, window_lo, window_len, per_crossbar


def schedule_pass_sequence(ready_cycles: list, R: int, C: int, BUF: int,
                           P: int, *, rows: int | None = None,
                           cols: int | None = None) -> list:
    """Place a pass sequence back-to-back on one crossbar, gated by readiness.

    Event grammar (certified primitive model, `nldpe_sim.NldpeDpe.run_workload`;
    P1/A9 next-load, P11 output-buffer bound):

        load_start(m=0) = max(ready_0, 0)
        load_start(m>0) = max(ready_m, msb_fire_{m-1} + 1)
        compute_start   = max(load_start + ceil(R*P/BUF), acc_free)
        msb_fire        = compute_start + P - 1
        shift_acc_done  = msb_fire + 1
        acam_done       = max(shift_acc_done + 1, out_free)
        drain_start     = acam_done + 1
        drain_end       = acam_done + ceil(C*P/BUF)
        acc_free = acam_done + 1;  out_free = drain_end + 1

    `ready_cycles[m]` is the cycle the m-th window's input can be consumed.
    Events are relative to this crossbar's cycle 0 (shift by its start offset).
    Returns one event dict per window, in order. `rows` / `cols`: the P29 pass
    extent (default R / C, the certified full load), replacing R / C in the two
    ceil(...) terms above.
    """
    cm = prim.cycle_model(1, R, C, P, BUF, rows=rows, cols=cols)
    load_cycles = int(cm.load_cyc)
    output_cycles = int(cm.output_cyc)

    events = []
    load_start = 0
    acc_free = 0
    out_free = 0
    prev_msb_fire = -1
    for m, window_ready in enumerate(ready_cycles):
        load_start = (max(window_ready, prev_msb_fire + 1) if m
                      else max(window_ready, 0))
        compute_start = max(load_start + load_cycles, acc_free)
        msb_fire = compute_start + P - 1
        shift_acc_done = msb_fire + 1
        acam_done = max(shift_acc_done + 1, out_free)
        drain_start = acam_done + 1
        drain_end = acam_done + output_cycles
        events.append({"load_start": load_start, "msb_fire": msb_fire,
                       "drain_start": drain_start, "drain_end": drain_end})
        prev_msb_fire = msb_fire
        acc_free = acam_done + 1
        out_free = drain_end + 1
    return events


# ---------------------------------------------------------------------------
# Self-test — run:  python3 pass_engine.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    shapes = [(8, 8), (40, 40), (128, 64), (128, 256), (256, 256), (512, 128), (1024, 64)]
    for R, C in shapes:
        I = min(R, C)
        # default == explicit full extent == the certified primitive totals
        for p in (0, 1, 2, 9):
            full = crossbar_total(p, R, C, 8, 40)
            assert full == crossbar_total(p, R, C, 8, 40, rows=R, cols=C)
            assert full == (int(prim.cycle_model(p, R, C, 8, 40).total) if p else 0)
            used = crossbar_total(p, R, C, 8, 40, rows=I, cols=I)
            assert used == crossbar_total(p, I, I, 8, 40) and used <= full
        assert identity_extent(R, C, "full") == (None, None)
        assert identity_extent(R, C, "used") == (I, I)
        # pass grammar: default unchanged; used extent == the same grammar on an I x I array,
        # including readiness gaps and back-to-back passes
        ready = [0, 0, 5, 400, 401]
        assert schedule_pass_sequence(ready, R, C, 40, 8) == \
            schedule_pass_sequence(ready, R, C, 40, 8, rows=R, cols=C)
        assert schedule_pass_sequence(ready, R, C, 40, 8, rows=I, cols=I) == \
            schedule_pass_sequence(ready, I, I, 40, 8)
    try:
        identity_extent(256, 256, "half")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown pass_extent must be rejected")
    # windows: unchanged helper, spot-check the tail window and round-robin
    n, lo, ln, per = packed_windows(10, 4, 2)
    assert n == 3 and list(lo) == [0, 4, 8] and list(ln) == [4, 4, 2] and per == [[0, 2], [1]]
    print("pass_engine self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
