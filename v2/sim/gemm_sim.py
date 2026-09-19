#!/usr/bin/env python3
"""gemm_sim.py — OOP behavior model for the GEMM array (v2 clean-room).

Role
----
Stage-2 golden model, transcribed from the FROZEN charter
`v2/spec/gemm.md` (**v0.3, 2026-09-17**). It composes the certified primitive
simulator: `V*H` `NldpeDpe` instances (imported unmodified) + byte-tree
reduce + lane serializer. No MAC/ACAM math is re-implemented here.

Verification chain (same discipline as Stage 1):

    charter v2/spec/gemm.md
       │ transcription
       ▼
    v2/oracle/gemm_ref.py          exact composition reference (values)
       │ GATE 1 (per case, inside `dump_case`): sim ≡ oracle bit-exact on
       │ int32 S and output bytes, and cycles ≡ §5.3 — enforced BEFORE any
       │ expected file is written; uncertified cases are never dumped
       ▼
    this file                      golden model; owns the schedule and the
       │                           expected-bit dumps for the TB
       │ GATE 2 (harness): RTL ≡ dumped expected bits (S probe + H lanes),
       │ cycles = §5.3 total + invariant Δ_impl
       ▼
    v2/rtl/gemm_top.v              DUT (instantiates V×H `dpe`)

The array schedule is **copied from the primitive timelines** (lockstep
instances): `NldpeDpe.run_workload` gives per-pass events; the array adds the
reduce/output fill `L_w = TREE_PIPE + 1` (charter §5.2/§5.3). Do not re-derive
cycle math here.

Conventions
-----------
* W : np.int8 [K, N];  X : np.int8 [M, K].
* S   : np.int32 [M, N] — wide partial after the byte-tree reduce (G5)
* out8 : np.uint8 [M, N] — array output bytes (serializer: low byte of S)
* Charter v0.3: every tile runs REGULAR (its ACAM is the identity form,
  `trunc8(y_v)`) and there is **no ACAM after the reduction** — the lane
  serializer takes the low byte. Nonlinear forms are out of scope here.

GATE-1 dump contract (`dump_case`, consumed by the Stage-2 harness/TB):
  weights.mem       — V*H*R*C int8 words, §4.1 order (v-major, then h,
                      then row-major row-outer), 10-hex-digit word per line
  act.mem           — M bursts × V*LCYC 40-bit words; burst m = V lane
                      blocks of LCYC words (lane v first), §4.2 packing
  expected_psum.npz — S : int32 [M, N]   (wide P26-analogue probe)
  expected_out.mem  — one line per pass: H*C bytes hex, lane h = columns
                      h*C .. h*C+C-1 in column order (§4.4)
  case.json         — {M,K,N,R,C,BUF,P,V,H,used_cycles,weight_cycles,
                      t_fill,t_steady,l_w}

Self-test — run:  python3 gemm_sim.py   (green = sim ≡ oracle on its own
fixed cases + cycle formula checks; GATE 1 additionally certifies every
generated case before dumping).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make the sibling modules importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
import nldpe_sim as dpe_sim  # noqa: E402
import gemm_ref as ref  # noqa: E402
import nldpe_ref as nref  # noqa: E402  (self-test per-tile expectations only)

MODE_REGULAR = 0  # v0.3: the only tile mode at this level


@dataclass
class GemmCycleModel:
    """Charter §5.3 closed form (analytical shadow of the array schedule)."""

    load_cyc: int        # LOAD_CYC = ceil(R*8/BUF)
    compute_cyc: int     # COMPUTE_CYC = P + 2
    output_cyc: int      # OUTPUT_CYC_prim = ceil(C*8/BUF)   (per lane)
    tree_pipe: int       # TREE_PIPE = ceil(log2 V) for V > 1, else 0
    l_w: int             # L_w = TREE_PIPE + 1               (fill latency)
    t_fill_tile: int     # LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC_prim
    t_fill: int          # T_fill_tile + L_w
    # max(LOAD_CYC + P, COMPUTE_CYC, OUTPUT_CYC_prim + 1, 1)
    t_steady: int
    total: int           # T_fill + (M-1) * T_steady
    wr_cyc: int          # WR_CYC = V*H*R*C (one-time, excluded)


@dataclass
class GemmPassTimeline:
    """Array events for one pass (absolute cycles; primitive timeline + L_w)."""

    m: int
    load_start: int
    load_end: int          # inclusive
    compute_start: int
    msb_fire: int
    shift_acc_done: int
    acam_done: int
    drain_start: int       # first tile drain word (lanes in parallel)
    drain_end: int         # inclusive, primitive-side (per lane)
    array_drain_start: int  # first reduced lane word  = drain_start + L_w
    array_drain_end: int   # last reduced lane word   = drain_end + L_w
    done: int              # dpe_done pulse (after last lane byte)


@dataclass
class GemmResult:
    """Everything the GATE-2 harness consumes (Stage 2)."""

    S: np.ndarray            # int32 [M, N]  — byte-tree partial (probe)
    out8: np.ndarray         # uint8 [M, N]  — final bytes, n = h*C + c
    used_cycles: int         # measured Total(M): last lane word + 1
    weight_cycles: int       # WR_CYC, reported separately
    cycle: GemmCycleModel    # t_fill / t_steady / l_w / total together
    timeline: list[GemmPassTimeline] = field(default_factory=list)
    # per-tile SimResult references when run(..., collect_tiles=True)
    tile_results: list[list[dpe_sim.SimResult]] | None = None


def derive_vh(K: int, N: int, R: int, C: int) -> tuple[int, int]:
    """V = ceil(K/R), H = ceil(N/C) (charter §1). Kept in one place."""
    return -(-K // R), -(-N // C)


def cycle_model(M: int, K: int, N: int, R: int = 256, C: int = 256,
                BUF: int = 40, P: int = 8) -> GemmCycleModel:
    """Charter §5.3 closed-form cycle contract.

    TODO(you): implement exactly the §5.3 formulas (no other constants):
      LOAD_CYC      = ceil(R*8/BUF)
      COMPUTE_CYC   = P + 2
      OUTPUT_CYC    = ceil(C*8/BUF)
      TREE_PIPE     = ceil(log2 V) if V > 1 else 0
      L_w           = TREE_PIPE + 1
      T_fill_tile   = LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC
      T_fill        = T_fill_tile + L_w
      T_steady      = max(LOAD_CYC + P, COMPUTE_CYC, OUTPUT_CYC + 1, 1)
      total         = T_fill + (M-1) * T_steady
      WR_CYC        = V*H*R*C
    """

    V, H = derive_vh(K, N, R, C)

    load_cyc = -(-R * 8 // BUF)                 # int, not float
    compute_cyc = P + 2
    output_cyc = -(-C * 8 // BUF)
    tree_pipe = (V - 1).bit_length() if V > 1 else 0   # ceil(log2 V), int
    l_w = tree_pipe + 1
    t_fill_tile = load_cyc + compute_cyc + output_cyc
    t_fill = t_fill_tile + l_w
    t_steady = max(load_cyc + P, compute_cyc, output_cyc + 1, 1)
    total = t_fill + (M - 1) * t_steady

    return GemmCycleModel(
        load_cyc=load_cyc, compute_cyc=compute_cyc, output_cyc=output_cyc,
        tree_pipe=tree_pipe, l_w=l_w, t_fill_tile=t_fill_tile,
        t_fill=t_fill, t_steady=t_steady, total=total,
        wr_cyc=V * H * R * C)

class NldpeGemm:
    """Charter §1-§6: V×H `NldpeDpe` instances + reduce tree + serializer.

    Structure:
      __init__          : derive V,H; build self.tiles[v][h] = NldpeDpe(R,C)
      program_weights   : zero-pad W to (V*R,H*C); per tile
                          `tile.program_weights(Wt)` (charter §4.1)
      run               : zero-pad X to (M,V*R); per tile (v,h)
                          `tile.run_workload(Xv, REGULAR)`; accumulate
                          `S = sum_v sign_extend(out8)`; serialize the low
                          byte (no ACAM after reduction, v0.3); timeline
                          from one tile (lockstep) + L_w; cross-check
                          `used_cycles` against `cycle_model` (the measured
                          schedule IS the theorem — same rule as the
                          primitive sim)
      dump_case         : GATE 1 certification then the file contract above
    """

    def __init__(self, M: int, K: int, N: int, R: int = 256, C: int = 256,
                 BUF: int = 40, P: int = 8) -> None:
        self.M, self.K, self.N = M, K, N
        self.R, self.C, self.BUF, self.P = R, C, BUF, P
        self.V, self.H = derive_vh(K, N, R, C)
        self.tiles = [[dpe_sim.NldpeDpe(self.R, self.C) for _ in range(self.H)]
                      for _ in range(self.V)]
        self.W = None
        self._programmed = False

    # -- weight programming (§4.1) ------------------------------------------

    def program_weights(self, W: np.ndarray) -> int:
        """Store int8 [K,N]; return WR_CYC = V*H*R*C (charter §4.1).

        TODO(you): validate shape/dtype, zero-pad to (V*R, H*C), slice tiles
        row-major row-outer, call `NldpeDpe.program_weights` per tile.
        """
        W = np.asarray(W)
        assert W.ndim == 2 and W.shape == (self.K, self.N), \
            f"need W [{self.K},{self.N}], got {W.shape}"
        Wp = np.zeros((self.V * self.R, self.H * self.C), dtype=np.int8)
        Wp[:self.K, :self.N] = W.astype(np.int8, copy=False)
        cycles = 0
        for v in range(self.V):
            for h in range(self.H):
                cycles += self.tiles[v][h].program_weights(
                    Wp[v*self.R:(v+1)*self.R, h*self.C:(h+1)*self.C])
        self.W = W.astype(np.int8, copy=True)   # unpadded (oracle/GATE 1)
        self.Wp = Wp                            # padded (§4.1 dump order)
        self._programmed = True
        return cycles
    # -- datapath (values + schedule) ---------------------------------------

    def run(self, X: np.ndarray, collect_tiles: bool = False) -> GemmResult:
        assert self._programmed, "program_weights() first"
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == self.K, \
            f"need X [M,{self.K}], got {X.shape}"

        M = X.shape[0]
        cyc = cycle_model(M, self.K, self.N, self.R, self.C, self.BUF, self.P)
        tile_mode = MODE_REGULAR     # v0.3: all tiles REGULAR

        Xp = np.zeros((M, self.V * self.R), dtype=np.int8)
        Xp[:, :self.K] = X

        S = np.zeros((M, self.H * self.C), dtype=np.int64)
        tile_results = ([[None] * self.H for _ in range(self.V)]
                        if collect_tiles else None)
        timing_ref = None
        for v in range(self.V):
            Xv = Xp[:, v*self.R:(v+1)*self.R]
            for h in range(self.H):
                res = self.tiles[v][h].run_workload(Xv, tile_mode)
                S[:, h*self.C:(h+1)*self.C] += \
                    res.out_stream.view(np.int8).astype(np.int64)   # sign-extend
                if tile_results is not None:
                    tile_results[v][h] = res
                if timing_ref is None:    # timing only; values are in S above
                    timing_ref = res
                else:                     # lockstep: one timeline suffices
                    assert res.used_cycles == timing_ref.used_cycles, \
                        "tile schedules diverged"
        assert int(np.abs(S).max()) <= 2**31 - 1, "S exceeds int32 (B8)"
        S32 = S.astype(np.int32)[:, :self.N]
        # F4 serializer: low byte of S; no ACAM after reduction (charter v0.3)
        out8 = (S32 & 0xFF).astype(np.uint8).view(np.int8)

        timeline = []
        for t in timing_ref.timeline:
            timeline.append(GemmPassTimeline(
                m=t.m, load_start=t.load_start, load_end=t.load_end,
                compute_start=t.compute_start, msb_fire=t.msb_fire,
                shift_acc_done=t.shift_acc_done, acam_done=t.acam_done,
                drain_start=t.drain_start, drain_end=t.drain_end,
                array_drain_start=t.drain_start + cyc.l_w,
                array_drain_end=t.drain_end + cyc.l_w,
                done=t.drain_end + cyc.l_w + 1))
        used = timeline[-1].done
        assert used == cyc.total, f"measured {used} != formula {cyc.total}"
        return GemmResult(S=S32, out8=out8, used_cycles=used,
                          weight_cycles=self.V * self.H * self.R * self.C,
                          cycle=cyc, timeline=timeline,
                          tile_results=tile_results)
            
            
    # -- stimulus dump (GATE 1 + GATE 2 expected files) ----------------------
    def dump_case(self, case_dir: Path, X: np.ndarray) -> None:
        """Write stimulus + expected files (contract in the module docstring).

        GATE 1: certify this exact case against `gemm_ref` (S int32 and the
        output bytes) and against `cycle_model` (used_cycles == total) BEFORE
        any file is written; an uncertified case is never dumped.
        """
        assert self._programmed, "program_weights() first"
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == self.K, \
            f"need X [M,{self.K}], got {X.shape}"
        M = X.shape[0]

        res = self.run(X)

        # -- GATE 1: per-case oracle certification -------------------------
        S_exp, out_exp = ref.compute_gemm(self.W, X, self.R, self.C)
        if not np.array_equal(res.S, S_exp):
            raise RuntimeError("GATE 1: sim/oracle S mismatch — case not dumped")
        if not np.array_equal(res.out8.view(np.uint8), out_exp.view(np.uint8)):
            raise RuntimeError("GATE 1: sim/oracle out8 mismatch — case not dumped")
        if res.used_cycles != res.cycle.total:
            raise RuntimeError(
                f"GATE 1: sim schedule {res.used_cycles} != §5.3 total "
                f"{res.cycle.total} — case not dumped")

        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        # weights.mem — §4.1 order: v-major, then h, then row-major row-outer
        with open(case_dir / "weights.mem", "w") as f:
            for v in range(self.V):
                for h in range(self.H):
                    tile = self.Wp[v*self.R:(v+1)*self.R, h*self.C:(h+1)*self.C]
                    for b in np.ascontiguousarray(tile).view(np.uint8).reshape(-1):
                        f.write(f"{int(b):010x}\n")

        # act.mem — M bursts; burst m = V lane blocks of LCYC words (§4.2)
        Xp = np.zeros((M, self.V * self.R), dtype=np.int8)
        Xp[:, :self.K] = X
        with open(case_dir / "act.mem", "w") as f:
            for m in range(M):
                for v in range(self.V):
                    for w in nref.pack_act_stream(Xp[m, v*self.R:(v+1)*self.R]):
                        f.write(f"{w:010x}\n")

        # expected_psum.npz — wide byte-tree partial S, int32 [M, N]
        np.savez(case_dir / "expected_psum.npz", S=res.S)

        # expected_out.mem — one line/pass: H*C bytes; lane h = cols hC..hC+C-1.
        # Padding columns (n >= N) carry zero weights (A5) -> byte 0.
        padded = np.zeros((M, self.H * self.C), dtype=np.int8)
        padded[:, :self.N] = res.out8
        with open(case_dir / "expected_out.mem", "w") as f:
            for m in range(M):
                f.write("".join(f"{int(b) & 0xFF:02x}" for b in padded[m]) + "\n")

        case = {
            "M": M, "K": self.K, "N": self.N, "R": self.R, "C": self.C,
            "BUF": self.BUF, "P": self.P, "V": self.V, "H": self.H,
            "used_cycles": int(res.used_cycles),
            "weight_cycles": int(res.weight_cycles),
            "t_fill": int(res.cycle.t_fill),
            "t_steady": int(res.cycle.t_steady),
            "l_w": int(res.cycle.l_w),
        }
        with open(case_dir / "case.json", "w") as f:
            json.dump(case, f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Self-test — run:  python3 gemm_sim.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Gates the golden model on fixed + randomized cases (charter v0.3).

    Coverage: cycle anchors; a crossbar-size matrix R ∈ {64,128,256,512},
    C ∈ {64,128,256} with random prime M/K/N (V,H ∈ 1..3); padding and
    zero/extreme corners; I6 (V=1 ≡ primitive REGULAR); serializer
    sign-extension micro-tests; stationarity + determinism; T_steady step.
    Every case checks S, out8, used_cycles, and each tile's y/out8 against
    `nldpe_ref` / `gemm_ref`.
    """
    rng = np.random.default_rng(20260917)

    # primes up to 3*max(R) = 1536, for K/N/M sweeps
    sieve = np.ones(1600, dtype=bool)
    sieve[:2] = False
    for p in range(2, 41):
        if sieve[p]:
            sieve[p * p::p] = False
    PRIMES = np.nonzero(sieve)[0]

    # ── cycle_model anchors (charter §5.3) ──────────────────────────────────
    cm = cycle_model(1, 256, 256, 256, 256)
    assert (cm.t_fill, cm.t_steady, cm.l_w) == (115, 60, 1), cm
    cm2 = cycle_model(1, 512, 256, 256, 256)                    # V=2
    assert (cm2.t_fill, cm2.t_steady, cm2.tree_pipe, cm2.l_w) == (116, 60, 1, 2)
    cm512 = cycle_model(1, 256, 512, 256, 512)
    assert (cm512.t_fill, cm512.t_steady) == (166, 104)
    cmsmall = cycle_model(1, 8, 8, 8, 8)                        # compute-bound
    assert (cmsmall.t_fill, cmsmall.t_steady) == (15, 10)
    cmlarge = cycle_model(1, 57, 8, 8, 8)                       # V=8
    assert (cmlarge.tree_pipe, cmlarge.l_w, cmlarge.wr_cyc) == (3, 4, 512)

    # ── helpers ─────────────────────────────────────────────────────────────
    def assert_tiles(W, X, R, C, res):
        """Per-tile y/out8 bit-exact vs the primitive oracle (padded slices)."""
        K, N = W.shape
        M = X.shape[0]
        V, H = derive_vh(K, N, R, C)
        Wp = np.zeros((V * R, H * C), dtype=np.int8)
        Wp[:K, :N] = W
        Xp = np.zeros((M, V * R), dtype=np.int8)
        Xp[:, :K] = X
        for v in range(V):
            Xv = Xp[:, v * R:(v + 1) * R]
            for h in range(H):
                y_exp = nref.compute_y(Wp[v*R:(v+1)*R, h*C:(h+1)*C], Xv)
                o_exp = nref.acam_transform(y_exp, nref.MODE_REGULAR)
                tr = res.tile_results[v][h]
                assert np.array_equal(tr.y, y_exp), f"tile y mismatch ({v},{h})"
                assert np.array_equal(tr.out_stream.view(np.int8), o_exp), \
                    f"tile out8 mismatch ({v},{h})"

    def check_case(M, K, N, R, C, W=None, X=None, extremes=False):
        if W is None:
            W = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
        if X is None:
            X = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
        if extremes:
            W[0, :] = -128
            if K > 1:
                W[1, :] = 127
            X[0, :] = -128
            if M > 1:
                X[1, :] = 127
        V, H = derive_vh(K, N, R, C)
        g = NldpeGemm(M, K, N, R, C)
        wc = g.program_weights(W)
        assert wc == V * H * R * C, (wc, V * H * R * C)
        res = g.run(X, collect_tiles=True)
        S_exp, o_exp = ref.compute_gemm(W, X, R, C)
        tag = f"M={M} K={K} N={N} R={R} C={C}"
        assert np.array_equal(res.S, S_exp), f"S mismatch {tag}"
        assert np.array_equal(res.out8.view(np.uint8), o_exp.view(np.uint8)), \
            f"out8 mismatch {tag}"
        assert res.used_cycles == cycle_model(M, K, N, R, C).total, \
            f"cycles mismatch {tag}"
        assert_tiles(W, X, R, C, res)
        return res

    # ── crossbar-size matrix: one case per (R, C) ────────────────────────────
    R_CHOICES = [64, 128, 256, 512]
    C_CHOICES = [64, 128, 256]
    for R in R_CHOICES:
        for C in C_CHOICES:
            K = int(rng.choice(PRIMES[PRIMES <= 2 * R]))
            N = int(rng.choice(PRIMES[PRIMES <= 2 * C]))
            check_case(int(rng.choice([1, 2, 3, 5])), K, N, R, C)

    # ── random sweep: random crossbars, prime M/K/N, mixed extremes ──────────
    for _ in range(8):
        R = int(rng.choice(R_CHOICES))
        C = int(rng.choice(C_CHOICES))
        K = int(rng.choice(PRIMES[PRIMES <= 3 * R]))
        N = int(rng.choice(PRIMES[PRIMES <= 3 * C]))
        M = int(rng.choice([1, 2, 3, 5, 7]))
        check_case(M, K, N, R, C, extremes=bool(rng.integers(0, 2)))

    # ── boundary / padding corners ───────────────────────────────────────────
    check_case(1, 1, 1, 512, 256)              # degenerate single element
    check_case(2, 512, 256, 512, 256)          # exact multiples (V=1, H=1)
    check_case(2, 513, 257, 512, 256)          # +1 over both -> V=2, H=2
    R, C, K, N = 64, 64, 97, 59
    check_case(2, K, N, R, C, W=np.zeros((K, N), dtype=np.int8))
    check_case(2, K, N, R, C, X=np.zeros((2, K), dtype=np.int8))
    check_case(3, K, N, R, C, extremes=True)

    # ── I6: V=1 reduces to the primitive REGULAR exactly ────────────────────
    R, C, K, N, M = 256, 128, 251, 127, 3
    W = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    X = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    g = NldpeGemm(M, K, N, R, C)
    g.program_weights(W)
    res = g.run(X, collect_tiles=True)
    # bare primitive on the same (padded) tile slice
    Wt = np.zeros((R, C), dtype=np.int8)
    Wt[:K, :N] = W
    Xt = np.zeros((M, R), dtype=np.int8)
    Xt[:, :K] = X
    t = dpe_sim.NldpeDpe(R, C)
    t.program_weights(Wt)
    pres = t.run_workload(Xt, MODE_REGULAR)
    assert np.array_equal(res.out8.view(np.uint8), pres.out_stream[:, :N]), "I6 out8"
    assert np.array_equal(res.S,
                          pres.out_stream[:, :N].view(np.int8).astype(np.int64)), \
        "I6 S"
    assert res.used_cycles == pres.used_cycles + cycle_model(M, K, N, R, C).l_w, \
        "I6 cycles"

    # ── serializer micro-tests (v0.3 exactness) ─────────────────────────────
    # negative leaf: y0 = 127*1 + 3*1 = 130 -> out8 = -126 (0x82), not 130
    g = NldpeGemm(1, 2, 2, 2, 2)
    g.program_weights(np.array([[127, 0], [3, 0]], dtype=np.int8))
    res = g.run(np.array([[1, 1]], dtype=np.int8))
    assert res.S.tolist() == [[-126, 0]], res.S
    assert res.out8.view(np.int8).tolist() == [[-126, 0]]
    assert res.out8.view(np.uint8)[0, 0] == 0x82
    # sign-crossing: tiles 128 and -2 -> S = -130, out8 = trunc8(+126) = 126
    g = NldpeGemm(1, 4, 1, 2, 1)
    g.program_weights(np.array([[127], [1], [-2], [0]], dtype=np.int8))
    res = g.run(np.array([[1, 1, 1, 0]], dtype=np.int8))
    assert res.S.tolist() == [[-130]] and res.out8.view(np.int8).tolist() == [[126]]

    # ── stationarity (I5) + determinism ─────────────────────────────────────
    R, C, K, N = 64, 64, 97, 59
    W = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    Xa = rng.integers(-128, 128, size=(2, K), dtype=np.int8)
    Xb = rng.integers(-128, 128, size=(3, K), dtype=np.int8)
    g = NldpeGemm(2, K, N, R, C)
    g.program_weights(W)
    ra1 = g.run(Xa, collect_tiles=True)
    Sa, Oa = ref.compute_gemm(W, Xa, R, C)
    assert np.array_equal(ra1.S, Sa)
    assert np.array_equal(ra1.out8.view(np.uint8), Oa.view(np.uint8))
    assert_tiles(W, Xa, R, C, ra1)
    rb = g.run(Xb, collect_tiles=True)
    Sb, Ob = ref.compute_gemm(W, Xb, R, C)
    assert np.array_equal(rb.S, Sb)
    assert np.array_equal(rb.out8.view(np.uint8), Ob.view(np.uint8))
    assert_tiles(W, Xb, R, C, rb)
    ra2 = g.run(Xa)
    assert np.array_equal(ra2.S, ra1.S) and np.array_equal(ra2.out8, ra1.out8)
    assert [t.done for t in ra2.timeline] == [t.done for t in ra1.timeline]

    # ── T_steady step ───────────────────────────────────────────────────────
    R, C, K, N = 256, 256, 401, 251            # V=2, H=1
    W = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    X = rng.integers(-128, 128, size=(2, K), dtype=np.int8)
    g = NldpeGemm(1, K, N, R, C)
    g.program_weights(W)
    step = g.run(X).used_cycles - g.run(X[:1]).used_cycles
    assert step == cycle_model(1, K, N, R, C).t_steady, step

    print("gemm_sim self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
