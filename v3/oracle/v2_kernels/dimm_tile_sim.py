#!/usr/bin/env python3
"""dimm_tile_sim.py — behavior model + cycle model of the attention tile DIMM (`dimm_tile_top`).

Role (ATT-2a, charter `v2/sim/spec/attention.md` §4.3 / §5.3; implementation plan ATT-2a)
------------------------------------------------------------------------------------------
The attention head's two matmuls are E-only signed tiles of the certified DIMM's E stage:

    acc[m, n] = sum_k (-1)^(sA[m,k] ^ sB[k,n]) * EXP_u8(trunc8(LA[m,k] + LB[k,n]))

with LA [M, K], LB [K, N] pre-converted log bytes (no pools), signs recovered from the log bytes by the
D3 rule (`attention_ref.sign_from_log`) or hard-wired + (`a_signed` / `b_signed` = False: the PV tile's
log|P| operand). Value contract = `attention_ref.signed_tile` (GATE 1 in `dump_case`).

Schedule (normative, = the RTL; window-major, user 2026-10-08): the element space M*N is cut row-major into
W_K = ceil(M*N / I) windows of I = min(R, C) elements (window w = rows [w*NB, (w+1)*NB), NB = I / N). With
NG = n_E / K groups, window (w, k) runs on crossbar (w mod NG) + NG*k in wave t = w div NG: every crossbar keeps
one k for the whole tile, and group gi = crossbars {gi + NG*k, k = 0..K-1} holds all K terms of window gi + NG*t.
All crossbars advance in lock-step, T_W = W_K / NG waves (P29 pass extent I x I). Each wave's registered adder
tree (log2 K levels) adds the K drains of a group: a window leaves the tile complete, as an output stream word
per drain word (5 lanes, VW = 9 + log2 K bits each) — no accumulator, no read-modify-write.
RTL domain (elaboration guards): M, N, K, n_E powers of two, N <= I, I >= 8, K <= n_E, n_E / K <= W_K.
Outside it (n_E < K: the DSE candidate grid only) the model accumulates a window over K / n_E consecutive waves
(tree log2 n_E, + RMW_LAT) — no RTL implements that regime (`TileGeometry.rtl`).

Cycles (start edge -> `done`, data-independent; pinned by the RTL, strict GATE 2, Δ_impl = 0):
    output word c of wave t at   drain_start(t) + c + OUT_LAT,   OUT_LAT = DRAIN_REG + TREE_LEVELS
    total = START_LAT + (T(T_W) - 1) + OUT_LAT + DONE_LAT
    occ   = START_LAT + T(T_W)          (start -> the next tile may start: every window drained, edge X1)
T(p) = the certified primitive crossbar time with the I x I extent, drain_start(t) from the certified pass grammar.
START_LAT = 2: the operand memories are read in the start cycle, the first feed word is registered in the next
(E2a registered feed), the first window is fed two edges after `start`; with A memories banked by A_GRP > 1
crossbars (E2b-2) START_LAT = A_GRP + 2. DRAIN_REG = 1 (the certified registered farm drain), TREE_LEVELS
= log2 K registered adder levels, DONE_LAT = 2 (the last-word flag, then the `done` register).

RTL file contract (`dump_case`, the TB `tb_dimm_tile_top.v`): operands are written as whole words —
A word (sel, k, w) = LA[w*NB + i, k], i = 0..NB-1 (rows beyond M zero); B word (sel, k) = LB[k, n], n = 0..N-1;
an A write of several words (A_NW) covers consecutive windows inside one NG-aligned group of windows.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "oracle"))
import nldpe_sim as prim  # noqa: E402
from pass_engine import crossbar_total, schedule_pass_sequence  # noqa: E402
import nldpe_ref as nref  # noqa: E402
import attention_ref as aref  # noqa: E402

BUF, P, EPS = 40, 8, 5
START_LAT = 2          # operand words read in the start cycle, feed word registered, first window fed at edge 2


def start_lat(a_grp: int = 1) -> int:
    """Start edge -> first feed: 2 (A_GRP = 1), A_GRP + 2 (bank fill: one word per cycle, then f0 registered)."""
    return START_LAT if a_grp == 1 else a_grp + 2


def a_grp_rule(n_E: int, R: int, C: int, a_nw: int = 1, a_ch: int = 1) -> int:
    """Crossbars per A memory (RTL localparam A_GRP, derived — not a parameter; memory rework item 1, user 2026-10-09).

    A memory's read port refills its crossbars one per cycle, so a memory can serve at most LCYC = ceil(I * 8 / BUF)
    crossbars (every crossbar has its next A word before its next window starts), and the A_NW-word writes of A_CH
    channels need n_E / A_GRP >= A_NW * A_CH distinct memories. A_GRP = the largest power of two under both bounds:
    the fewest A memories that keep up (I = 64 / 128 / 256: 8 / 16 / 32)."""
    I = min(R, C)
    cap = min(-(-I * 8 // BUF), max(1, n_E // (a_nw * a_ch)))
    return 1 << (cap.bit_length() - 1)
DRAIN_REG = 1          # certified registered farm -> tree drain register (dimm W2)
DONE_LAT = 2           # last output word -> `done` (the tree's last-word flag, then the done register)
RMW_LAT = 2            # model only (n_E < K): accumulator read (synchronous) + write per partial wave


def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def _clog2(x: int) -> int:
    return max(0, (x - 1).bit_length())


def check_domain(M: int, N: int, K: int, R: int, C: int, n_E: int, a_nw: int = 1, a_ch: int = 1) -> None:
    I = min(R, C)
    lcyc = -(-I * 8 // BUF)
    if a_nw < 1 or a_ch < 1:
        raise ValueError(f"A_NW={a_nw}, A_CH={a_ch}: must be >= 1")
    a_grp = a_grp_rule(n_E, R, C, a_nw, a_ch)
    if not (_pow2(a_grp) and a_grp <= n_E and a_grp <= lcyc):
        raise ValueError(f"A_GRP={a_grp}: must be a power of two <= n_E={n_E} and <= LCYC={lcyc}")
    if not 1 <= a_nw <= n_E // a_grp:
        raise ValueError(f"A_NW={a_nw}: 1 .. n_E / A_GRP = {n_E // a_grp} (one write per interleaved memory)")
    if a_ch < 1 or (a_ch > 1 and (n_E // a_grp) % (a_ch * a_nw)):
        raise ValueError(f"A_CH={a_ch}: n_E / A_GRP = {n_E // a_grp} must be a multiple of A_CH * A_NW "
                         f"(each channel writes its own memory slice)")
    if not (_pow2(M) and _pow2(N) and _pow2(n_E) and _pow2(I)):
        raise ValueError(f"M={M}, N={N}, n_E={n_E}, I={I}: must be powers of two")
    if N > I or I < 8:
        raise ValueError(f"need N <= I and I >= 8 (N={N}, I={I})")
    if K < 1:
        raise ValueError("K must be >= 1")


def check_rtl_domain(M: int, N: int, K: int, R: int, C: int, n_E: int, a_nw: int = 1, a_ch: int = 1) -> None:
    """The window-major RTL's elaboration guards (on top of `check_domain`)."""
    check_domain(M, N, K, R, C, n_E, a_nw, a_ch)
    I = min(R, C)
    W_K = max(1, (M * N) // I)
    if not _pow2(K) or K < 2 or K > n_E or n_E // K > W_K:
        raise ValueError(f"window-major tile: K={K} must be a power of two, 2 <= K <= n_E={n_E}, "
                         f"n_E / K <= W_K={W_K}")
    NG = n_E // K
    if a_nw > NG:
        raise ValueError(f"A_NW={a_nw}: a multi-word A write stays inside one group of NG={NG} windows")
    if a_ch > 1 and NG % (a_ch * a_nw):
        raise ValueError(f"A_CH={a_ch}: NG={NG} must be a multiple of A_CH * A_NW (channel slices inside a group)")


@dataclass
class TileGeometry:
    M: int
    N: int
    K: int
    R: int
    C: int
    n_E: int
    I: int
    NB: int            # rows per window
    W_K: int           # windows per k
    PT: int            # windows per tile
    T_W: int           # windows per crossbar (waves)
    NG: int            # groups = complete windows per wave (n_E / K); model-only regime: 1
    D_CONC: int        # crossbars per group (K; model-only regime: n_E)
    d_conc: int        # members that hold a term of the window: D_CONC
    tree_levels: int
    acc_waves: int = 1 # waves that add into one window (model-only regime n_E < K: K / n_E)
    rtl: bool = True   # inside the window-major RTL domain


def geometry(M: int, N: int, K: int, R: int, C: int, n_E: int) -> TileGeometry:
    check_domain(M, N, K, R, C, n_E)
    I = min(R, C)
    NB = I // N
    W_K = max(1, (M * N) // I)
    PT = K * W_K
    if n_E >= K:                                     # window-major: NG complete windows per wave
        NG = min(n_E // K, W_K)
        D, acc = K, 1
        T_W = -(-W_K // NG)
    else:                                            # model only: a window over K / n_E consecutive waves
        NG, D, acc = 1, n_E, -(-K // n_E)
        T_W = W_K * acc
    rtl = _pow2(K) and n_E >= K and n_E // K <= W_K
    return TileGeometry(M=M, N=N, K=K, R=R, C=C, n_E=n_E, I=I, NB=NB, W_K=W_K, PT=PT, T_W=T_W, NG=NG,
                        D_CONC=D, d_conc=D, tree_levels=_clog2(D), acc_waves=acc, rtl=rtl)


@dataclass
class TileCycleModel:
    geometry: TileGeometry
    T_E: int           # crossbar time of T_W back-to-back windows (I x I extent)
    occ: int           # start edge -> start_ready (the farm is free for the next tile, edge X1)
    total: int         # start edge -> done
    passes: int        # windows issued (= PT)
    out_t0: list       # per wave t: the cycle (from the start edge) of its first output word; word c at + c
    out_lat: int       # drain word -> output word (DRAIN_REG + tree levels)


def tile_cycle_model(M: int, N: int, K: int, R: int, C: int, n_E: int,
                     pass_extent: str = "used", a_nw: int = 1, a_ch: int = 1) -> TileCycleModel:
    """`pass_extent` "used" (default) = the RTL (every pass loads / drains I = min(R, C)); "full" = the certified
    full load (R rows in, C columns out), for full-extent comparisons only — no RTL implements it. The A memory
    grouping (start latency A_GRP + 2) is derived from the A port (`a_grp_rule`)."""
    g = geometry(M, N, K, R, C, n_E)
    check_domain(M, N, K, R, C, n_E, a_nw, a_ch)
    a_grp = a_grp_rule(n_E, R, C, a_nw, a_ch)
    if pass_extent not in ("used", "full"):
        raise ValueError(f"pass_extent must be 'used' or 'full', got {pass_extent!r}")
    rows, cols = (g.I, g.I) if pass_extent == "used" else (None, None)
    T_E = crossbar_total(g.T_W, R, C, P, BUF, rows=rows, cols=cols)
    sl = start_lat(a_grp)
    occ = sl + T_E
    out_lat = DRAIN_REG + g.tree_levels
    ev = schedule_pass_sequence([sl] * g.T_W, R, C, BUF, P, rows=rows, cols=cols)
    assert ev[-1]["drain_end"] == sl + T_E - 1, "pass grammar and T(p) disagree"
    out_t0 = [e["drain_start"] + out_lat for e in ev]
    total = sl + (T_E - 1) + out_lat + (DONE_LAT if g.acc_waves == 1 else RMW_LAT + 1)
    return TileCycleModel(geometry=g, T_E=T_E, occ=occ, total=total, passes=g.PT, out_t0=out_t0, out_lat=out_lat)


class NldpeDimmTile:
    """Values through the certified primitive (one identity-EXP pass per window), cycles via the model."""

    def __init__(self, M: int, N: int, K: int, R: int, C: int, n_E: int,
                 a_signed: bool = True, b_signed: bool = True, a_nw: int = 1, a_ch: int = 1) -> None:
        self.g = geometry(M, N, K, R, C, n_E)
        check_rtl_domain(M, N, K, R, C, n_E, a_nw, a_ch)
        self.a_grp, self.a_nw, self.a_ch = a_grp_rule(n_E, R, C, a_nw, a_ch), a_nw, a_ch
        self.a_signed, self.b_signed = a_signed, b_signed
        I = self.g.I
        self.dpe = prim.NldpeDpe(R, C, BUF, P, rows_used=I, cols_used=I)
        self.dpe.program_weights(np.eye(R, C, dtype=np.int8))

    def run(self, LA: np.ndarray, LB: np.ndarray) -> dict:
        """LA int8 [M, K], LB int8 [K, N] log bytes -> acc int64 [M, N]; `windows` = the output stream order:
        per wave t, per group gi, window gi + NG*t (complete, I lanes; lanes past M*N are 0)."""
        g = self.g
        LA = np.asarray(LA, dtype=np.int64)
        LB = np.asarray(LB, dtype=np.int64)
        assert LA.shape == (g.M, g.K) and LB.shape == (g.K, g.N)
        sA = aref.sign_from_log(LA) if self.a_signed else np.zeros_like(LA)
        sB = aref.sign_from_log(LB) if self.b_signed else np.zeros_like(LB)
        acc = np.zeros(g.M * g.N, dtype=np.int64)
        touched = np.zeros(g.M * g.N, dtype=np.int64)
        X = np.zeros((g.PT, g.R), dtype=np.int8)
        sgn = np.zeros((g.PT, g.I), dtype=bool)
        idx = np.full((g.PT, g.I), -1, dtype=np.int64)
        for gw in range(g.PT):                          # window-major: window (w, k) on crossbar (w mod NG) + NG*k
            w, k = divmod(gw, g.K)
            e = w * g.I + np.arange(g.I)
            ok = e < g.M * g.N
            m, n = e[ok] // g.N, e[ok] % g.N
            X[gw, :ok.sum()] = nref.trunc8(LA[m, k] + LB[k, n])        # F7 int8 feed
            sgn[gw, :ok.sum()] = (sA[m, k] ^ sB[k, n]).astype(bool)
            idx[gw, :ok.sum()] = e[ok]
        res = self.dpe.run_workload(X, nref.MODE_EXP)
        out = res.out_stream[:, :g.I].astype(np.int64)                 # unsigned EXP bytes
        for gw in range(g.PT):
            ok = idx[gw] >= 0
            v = np.where(sgn[gw, ok], -out[gw, ok], out[gw, ok])
            np.add.at(acc, idx[gw, ok], v)
            np.add.at(touched, idx[gw, ok], 1)
        assert (touched == g.K).all(), "every element must be reduced over all K"
        flat = np.concatenate([acc, np.zeros(g.W_K * g.I - g.M * g.N, dtype=np.int64)]) \
            if g.M * g.N < g.W_K * g.I else acc
        windows = [[flat[(gi + g.NG * t) * g.I:(gi + g.NG * t + 1) * g.I] for gi in range(g.NG)]
                   for t in range(g.T_W)]
        return {"acc": acc.reshape(g.M, g.N), "sA": sA, "sB": sB, "passes": g.PT, "windows": windows}

    def dump_case(self, case_dir: Path, tiles: list, write_order: str = "kmajor",
                  late_writes: bool = False, cls: str = "") -> dict:
        """GATE 1 + files for `tb_dimm_tile_top.v`.

        tiles: list of dicts {a_sel, b_sel, LA [M,K], LB [K,N]} run back to back (each tile's windows leave on the
        output stream). Operand selections are written before the first start, or — `late_writes` — each
        tile's operands right after the previous tile's start (they must then use selections the running tile
        does not read). Words are written k-major (`kmajor`: k outer, window inner) or window-major (`wmajor`).
        Files:
          a_wr.txt         one A word per line: sel k w data (data hex, byte i = row w*NB + i at bits 8i)
          b_wr.txt         one B word per line: sel k data   (byte n = column n at bits 8n)
          tiles.txt        per tile: a_sel b_sel a0 a1 b0 b1 (word ranges written for this tile)
          exp_acc_<i>.mem  per tile: M*N int32 (row-major, 8-hex); the TB maps output word (t, c), group gi, lane j
                           to element (gi + NG*t)*I + 5c + j
          case.json        params, model, expected cycles per tile, out_t0 (first output word of each wave)
        """
        g = self.g
        case_dir = Path(case_dir)
        exp_accs = []
        for i, t in enumerate(tiles):
            r = self.run(t["LA"], t["LB"])
            want = aref.signed_tile(np.asarray(t["LA"]), r["sA"], np.asarray(t["LB"]), r["sB"])[0]
            if not np.array_equal(r["acc"], want):                      # GATE 1
                raise RuntimeError(f"GATE 1: tile {i}: sim acc != attention_ref.signed_tile — not dumped")
            exp_accs.append(r["acc"])
        cm = tile_cycle_model(g.M, g.N, g.K, g.R, g.C, g.n_E, a_nw=self.a_nw, a_ch=self.a_ch)
        case_dir.mkdir(parents=True, exist_ok=True)

        def hexword(vals) -> str:
            v = 0
            for i, b in enumerate(vals):
                v |= (int(b) & 0xFF) << (8 * i)
            return f"{v:x}"

        def a_words(sel: int, LA: np.ndarray) -> list:
            LA = np.asarray(LA, dtype=np.int64)
            pairs = [(k, w) for k in range(g.K) for w in range(g.W_K)]
            if write_order == "wmajor":
                pairs = [(k, w) for w in range(g.W_K) for k in range(g.K)]
            out = []
            for k, w in pairs:
                rows = [LA[w * g.NB + i, k] if w * g.NB + i < g.M else 0 for i in range(g.NB)]
                out.append(f"{sel:x} {k:x} {w:x} {hexword(rows)}")
            return out

        def b_words(sel: int, LB: np.ndarray) -> list:
            LB = np.asarray(LB, dtype=np.int64)
            ks = range(g.K) if write_order == "kmajor" else reversed(range(g.K))
            return [f"{sel:x} {k:x} {hexword(LB[k])}" for k in ks]

        aw, bw, rows = [], [], []
        seen_a, seen_b = {}, {}
        for t in tiles:
            a0, b0 = len(aw), len(bw)
            if t["a_sel"] not in seen_a or not np.array_equal(seen_a[t["a_sel"]], t["LA"]):
                aw += a_words(t["a_sel"], t["LA"])
                seen_a[t["a_sel"]] = np.asarray(t["LA"])
            if t["b_sel"] not in seen_b or not np.array_equal(seen_b[t["b_sel"]], t["LB"]):
                bw += b_words(t["b_sel"], t["LB"])
                seen_b[t["b_sel"]] = np.asarray(t["LB"])
            rows.append((t["a_sel"], t["b_sel"], a0, len(aw), b0, len(bw)))
        (case_dir / "a_wr.txt").write_text("".join(x + "\n" for x in aw))
        (case_dir / "b_wr.txt").write_text("".join(x + "\n" for x in bw))
        (case_dir / "tiles.txt").write_text("".join(" ".join(str(x) for x in r) + "\n" for r in rows))
        for i, acc in enumerate(exp_accs):
            (case_dir / f"exp_acc_{i}.mem").write_text(
                "".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in acc.reshape(-1)))
        case = dict(M=g.M, N=g.N, K=g.K, R=g.R, C=g.C, N_E=g.n_E, I=g.I, NB=g.NB, W_K=g.W_K, PT=g.PT,
                    T_W=g.T_W, NG=g.NG, D_CONC=g.D_CONC, tree_levels=g.tree_levels, VW=9 + g.tree_levels,
                    LW=-(-g.I // EPS),
                    A_SIGNED=int(self.a_signed), B_SIGNED=int(self.b_signed), A_GRP=self.a_grp, A_NW=self.a_nw, A_CH=self.a_ch,
                    NA_SEL=max(t["a_sel"] for t in tiles) + 1, NB_SEL=max(t["b_sel"] for t in tiles) + 1,
                    NTILES=len(tiles), A_WORDS=len(aw), B_WORDS=len(bw),
                    write_order=write_order, late_writes=int(late_writes), cls=cls,
                    T_E=cm.T_E, occ=cm.occ, expected_cycles=[cm.total] * len(tiles), out_t0=cm.out_t0,
                    out_lat=cm.out_lat, passes=cm.passes)
        (case_dir / "case.json").write_text(json.dumps(case, indent=1) + "\n")
        return case


# ---------------------------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/simulator/kernels/dimm_tile_sim.py
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(2026_10_08)
    n = 0
    # values: behavior == oracle on both sign modes, window-major RTL domain (NG = n_E / K <= W_K), partial windows,
    # R != C; the output stream holds every window once, complete, in wave / group order
    for (M, N, K, R, C, nE) in ((16, 8, 16, 64, 64, 16), (16, 8, 16, 64, 64, 32), (32, 16, 8, 64, 64, 32),
                                (8, 4, 8, 64, 64, 8), (64, 16, 4, 256, 128, 16), (4, 8, 8, 64, 32, 8),
                                (32, 32, 4, 128, 64, 64), (128, 16, 64, 256, 256, 128), (128, 64, 16, 256, 256, 128)):
        for a_s, b_s in ((True, True), (False, True)):
            LA = rng.integers(-128, 128, size=(M, K))
            LB = rng.integers(-128, 128, size=(K, N))
            t = NldpeDimmTile(M, N, K, R, C, nE, a_s, b_s)
            r = t.run(LA, LB)
            sA = aref.sign_from_log(LA) if a_s else np.zeros_like(LA)
            sB = aref.sign_from_log(LB) if b_s else np.zeros_like(LB)
            want = aref.signed_tile(LA, sA, LB, sB)[0]
            assert np.array_equal(r["acc"], want), (M, N, K, nE, a_s)
            g = t.g
            assert g.rtl and g.PT == K * g.W_K and g.T_W * g.NG == g.W_K and g.NG * g.D_CONC == nE
            flat = np.concatenate([want.reshape(-1), np.zeros(g.W_K * g.I - M * N, dtype=np.int64)])
            got = np.concatenate([r["windows"][tt][gi] for tt in range(g.T_W) for gi in range(g.NG)])
            order = np.concatenate([np.arange((gi + g.NG * tt) * g.I, (gi + g.NG * tt + 1) * g.I)
                                    for tt in range(g.T_W) for gi in range(g.NG)])
            assert np.array_equal(got, flat[order]) and sorted(order) == list(range(g.W_K * g.I))
            n += 1
    # cycle model: output words from the certified pass grammar; more crossbars never slower; every word of every
    # wave comes after its drain; model-only regime (n_E < K) still monotone
    for (M, N, K, R, C) in ((128, 16, 64, 256, 256), (128, 64, 16, 256, 256), (16, 8, 16, 64, 64)):
        prev = None
        for nE in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            g = geometry(M, N, K, R, C, nE)
            if nE >= K and nE // K > g.W_K:
                continue
            cm = tile_cycle_model(M, N, K, R, C, nE)
            assert cm.T_E >= crossbar_total(1, R, C, P, BUF, rows=min(R, C), cols=min(R, C))
            assert len(cm.out_t0) == g.T_W and all(b > a for a, b in zip(cm.out_t0, cm.out_t0[1:]))
            lw = -(-g.I // EPS)
            assert cm.out_t0[-1] + lw - 1 + DONE_LAT == cm.total or not g.rtl
            if prev is not None:
                assert cm.total <= prev.total or prev.geometry.rtl != g.rtl, (M, N, K, nE)
            prev = cm
            n += 1
    # the BERT-Tiny QK tile: 4 waves of 2 complete windows, a 6-level tree, A_GRP = 32 (LCYC = 52)
    cm = tile_cycle_model(128, 16, 64, 256, 256, 128)
    assert (cm.geometry.NG, cm.geometry.T_W, cm.geometry.tree_levels) == (2, 4, 6)
    assert cm.out_t0[0] == tile_cycle_model(128, 16, 64, 256, 256, 128, a_nw=4, a_ch=2).out_t0[0] + 32 - 16
    n += 2
    # A_GRP rule: pinned values; over a sweep it is a power of two, <= LCYC, leaves >= A_NW * A_CH memories, and is
    # the largest such (doubling it breaks a bound) — adversarial corners: n_E = A_NW * A_CH (one crossbar each),
    # A_NW not a power of two, LCYC = 2 (I = 8), LCYC a power of two (I = 16: 4)
    for args, want in (((128, 256, 256), 32), ((128, 128, 128), 16), ((128, 64, 64), 8), ((16, 64, 64), 8),
                       ((8, 8, 8), 2), ((64, 16, 16), 4), ((32, 32, 32), 4), ((64, 256, 128), 16),
                       ((64, 64, 64, 5), 8), ((32, 64, 64, 3), 8), ((16, 64, 64, 2, 4), 2), ((8, 64, 64, 2, 4), 1),
                       ((128, 256, 256, 4, 2), 16), ((4, 64, 64), 4), ((1, 64, 64), 1)):
        assert a_grp_rule(*args) == want, (args, a_grp_rule(*args), want)
        n += 1
    for nE in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        for I in (8, 16, 32, 64, 128, 256):
            for anw in (1, 2, 3, 4, 5, 8):
                for ach in (1, 2, 4):
                    g = a_grp_rule(nE, I, I, anw, ach)
                    lcyc, cap = -(-I * 8 // BUF), max(1, nE // (anw * ach))
                    assert _pow2(g) and g <= lcyc and g <= cap and (2 * g > lcyc or 2 * g > cap)
                    assert nE < anw * ach or nE // g >= anw * ach
                    n += 1
    # domain guards (RTL): K not a power of two, K > n_E, n_E / K > W_K, K = 1, A_NW > NG, A_CH slices outside a group
    for bad in (dict(a=(16, 8, 12, 64, 64, 16)), dict(a=(16, 8, 16, 64, 64, 8)), dict(a=(16, 8, 4, 64, 64, 16)),
                dict(a=(16, 8, 1, 64, 64, 2)),
                dict(a=(32, 16, 8, 64, 64, 16), a_nw=4), dict(a=(32, 16, 8, 64, 64, 32), a_nw=2, a_ch=4)):
        try:
            check_rtl_domain(*bad["a"], a_nw=bad.get("a_nw", 1), a_ch=bad.get("a_ch", 1))
        except ValueError:
            n += 1
            continue
        raise AssertionError(f"RTL domain guard missed {bad}")
    for bad in ((12, 8, 4, 64, 64, 4), (16, 128, 4, 64, 64, 4), (16, 8, 0, 64, 64, 4), (16, 8, 4, 64, 64, 3)):
        try:
            geometry(*bad)
        except ValueError:
            n += 1
            continue
        raise AssertionError(f"domain guard missed {bad}")
    print(f"dimm_tile_sim: ALL PASS ({n} checks: window-major values == attention_ref.signed_tile through the "
          "certified primitive (both sign modes, partial windows, R != C), output stream order, cycle model from the "
          "pass grammar, monotonicity, RTL-domain guards)")


if __name__ == "__main__":
    _self_test()
