#!/usr/bin/env python3
"""attention_ref.py — NumPy value reference for the blocked FlashAttention head (NL-DPE).

Role
----
Value contract of one attention head (charter `v2/sim/spec/attention.md` §5; FA dataflow
`v2/sim/spec/flash_attention.md` §1). Plain math, no state, no time. The behavior simulator
(`attention_sim.py`, GATE 1) and the RTL (GATE 2) must agree with it bit-exactly on every
stage below. Cycles are not modelled here.

Every stage is a certified oracle function except the three new contract pieces (the sign
rule, the score narrowing and the o-carry, charter §5.2–§5.4):

    projections  x8 = trunc8(X @ W)                    gemm_ref.compute_gemm (any V x H tiling)
    log, sign    L  = ACAM_LOG(x8) = trunc8(x8 - 1)    nldpe_ref (V=1: LOG in the projection
                                                       pass, LOG(y) == LOG(trunc8(y)); V>1: pool)
                 s  = L[7] ^ &L[6:0]                   == (x8 < 0), exact (D3)
    QK tile      acc[m,n] = sum_k (-1)^(sQ[m,k]^sK[n,k]) * EXP_u8(trunc8(LQ[m,k] + LK[n,k]))
                                                       (the certified DIMM E stage + signs)
    narrow (D1)  s8 = sat8(acc >> SCORE_SHIFT)         arithmetic shift, SCORE_SHIFT = 5
    softmax      softmax_online_ref.logcarry_stage_values(s8, Bkv): logp, log_a, log_l_final
    PV tile      o_part[b][m,c] = sum_{j in b} (-1)^sV[j,c] * EXP_u8(trunc8(logp[b][m,j] + LV[j,c]))
                                                       (sign of P hard-wired +)
    o-carry (D6) b = 0 : o' = o_part[0]
                 b >= 1: t = clamp8(log_a[b] + lo);  e = EXP_u8(t)
                         o' = (so ? -e : +e) << SO  +  o_part[b]
                 every b: oq = min(|o'| >> SO, 127);  so = (o' < 0);  lo = ACAM_LOG(oq)
    epilogue     tE = clamp8(lo - log_l_final);  eE = EXP_u8(tE) * 2^(SO - LN)
                 O8 = so ? max(-eE, -128) : min(eE, 127)
    O-projection y8 = trunc8(O8 @ WO)                  gemm_ref.compute_gemm (V_O x H_O tiles)
    layer (M5)   H heads, each its own projections and the head above up to O8; one O-projection over the
                 concatenated head outputs: y8 = trunc8(concat_h(O8_h) @ WO), WO int8 [H*d, dm]

with LN = log2 n_seq (the softmax's compaction shift: lq = min(l' >> LN, 127)) and the o-carry
compaction shift SO (default SO = LN: `o` is carried, like `l`, as its mean over the n_seq keys,
so the epilogue's EXP(log|o| - log l) is o/l with no extra shift). EXP_u8 = the ACAM EXP byte read
unsigned (P24: trunc8(1 + y + floor(y^2/2))); clamp8 = clamp to [-128, 127].

Properties (self-test)
----------------------
* Values are independent of the crossbar geometry (R, C): projections are exact (V=1 == V>1 ==
  H>1), conversions are elementwise (F5-F8); only pass counts depend on (R, C).
* Rows are independent: a run of S = min(n_seq, I) query rows gives the same values as all rows.
* Bkv = n_seq (one key block) is the conventional softmax + one signed PV DIMM + one compaction.
* With the P24 surrogates replaced by the real exp / log and the saturations removed, the same
  dataflow (`head_mirror` with float ops) is textbook attention softmax(q k^T 2^-SCORE_SHIFT) v.
* As for the softmax (decision A), the integer values are the declared ACAM approximations, not
  numerically meaningful attention; Bkv < n_seq drifts from Bkv = n_seq (reported, not gated).

Functions
---------
  sign_from_log(L)                  operand sign from its log byte (D3)
  project(X, W, R, C)               projection bytes, log bytes, signs, tiling
  signed_tile(LA, sA, LB, sB)       signed DIMM E stage + exact reduce -> (acc, wraps)
  narrow(acc, shift)                D1 score narrowing
  ocarry_stage_values(o_part, log_a, LN, SO)
  epilogue(lo, so, log_l, LN, SO)
  head_from_qkv(q8, k8, v8, Bkv)    the head between projections and O-projection (contract)
  head_stages(X, WQ, WK, WV, WO, Bkv, R, C)    full head (contract)
  layer_stages(X, WQ, WK, WV, WO, Bkv, R, C)   multi-head layer (contract, M5): WQ / WK / WV [H, dm, d]
  head_pass_view(X, WQ, WK, WV, WO, Bkv, R, C) dual witness: per run / tile / pass, with counts
  head_pass_view_qkv(q8, k8, v8, WO, Bkv, R, C) the same from given operands (LOG pools)
  head_mirror(q, k, v, Bkv, ops)    topology witness (INT_OPS == contract, FLOAT_OPS == attention)
  value_stats(st)                   saturation / clamp / wrap fractions (R2 bars)
  drift_report(q8, k8, v8, Bkv)     blocked vs single-block drift
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dimm_ref  # noqa: E402
import gemm_ref  # noqa: E402
import nldpe_ref as ref  # noqa: E402
import softmax_online_ref as sor  # noqa: E402

SCORE_SHIFT = 5          # D1 (user, 2026-10-07)
NOTEBOOK = Path(__file__).resolve().parent / "flash-attention-log-domain.ipynb"


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def check_domain(n_seq: int, d: int, dm: int, Bkv: int, R: int | None = None,
                 C: int | None = None) -> None:
    """Value domain: n_seq, Bkv powers of two, 8 <= Bkv <= n_seq, d, dm >= 1. With (R, C) the
    kernel domain is added: I = min(R, C) >= 8 (S = min(n_seq, I) >= 8) and Bkv <= I."""
    if not (_pow2(n_seq) and _pow2(Bkv)):
        raise ValueError(f"n_seq={n_seq}, Bkv={Bkv}: both must be powers of two")
    if not 8 <= Bkv <= n_seq:
        raise ValueError(f"Bkv={Bkv} outside 8 <= Bkv <= n_seq={n_seq}")
    if d < 1 or dm < 1:
        raise ValueError(f"d={d}, d_model={dm}: need >= 1")
    if R is not None or C is not None:
        I = min(R, C)
        if I < 8:
            raise ValueError(f"I = min(R, C) = {I} < 8")
        if Bkv > I:
            raise ValueError(f"Bkv={Bkv} > I={I} (online softmax kernel domain)")


def geometry(n_seq: int, d: int, dm: int, Bkv: int, R: int, C: int) -> dict:
    I = min(R, C)
    S = min(n_seq, I)
    return {"I": I, "S": S, "Rb": n_seq // S, "B": n_seq // Bkv,
            "V": -(-dm // R), "H": -(-d // C), "V_O": -(-d // R), "H_O": -(-dm // C)}


# ---------------------------------------------------------------------------
# Elementwise pieces
# ---------------------------------------------------------------------------
def _log_i8(x: np.ndarray) -> np.ndarray:
    """ACAM LOG of an integer value as the signed log byte (int64)."""
    return ref.acam_transform(np.asarray(x, dtype=np.int64), ref.MODE_LOG).astype(np.int64)


def _exp_u8(y: np.ndarray) -> np.ndarray:
    """ACAM EXP of an int8 feed as the unsigned byte (int64)."""
    return ref.acam_transform(np.asarray(y, dtype=np.int64), ref.MODE_EXP) \
        .view(np.uint8).astype(np.int64)


def _feed8(u: np.ndarray) -> np.ndarray:
    """F7: the wide CLB argument is truncated to the int8 crossbar feed."""
    return ref.trunc8(np.asarray(u, dtype=np.int64)).astype(np.int64)


def sign_from_log(L: np.ndarray) -> np.ndarray:
    """D3: sign bit of x from its log byte L = trunc8(x - 1): MSB(x) = L[7] ^ &L[6:0]."""
    b = np.asarray(L, dtype=np.int64) & 0xFF
    return ((b >> 7) & 1) ^ ((b & 0x7F) == 0x7F).astype(np.int64)


def narrow(acc: np.ndarray, shift: int = SCORE_SHIFT) -> np.ndarray:
    """D1: s8 = sat8(acc >> shift) (arithmetic shift = floor)."""
    return np.clip(np.asarray(acc, dtype=np.int64) >> shift, -128, 127)


def _scale_pow2(e: np.ndarray, k: int) -> np.ndarray:
    return e << k if k >= 0 else e >> (-k)


def _sat8_signed(neg: np.ndarray, mag: np.ndarray) -> np.ndarray:
    return np.where(neg.astype(bool), np.maximum(-mag, -128), np.minimum(mag, 127))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def project(X: np.ndarray, W: np.ndarray, R: int = 256, C: int = 256) -> dict:
    """Projection x8 = trunc8(X @ W) through the certified GEMM (V x H tiles), its log byte and
    sign. X int8 [n, dm]; W int8 [dm, d]."""
    X = np.asarray(X, dtype=np.int8)
    W = np.asarray(W, dtype=np.int8)
    S_wide, x8 = gemm_ref.compute_gemm(W, X, R, C)
    L = _log_i8(x8)
    V, H = gemm_ref.derive_vh(W.shape[0], W.shape[1], R, C)
    return {"x8": x8.astype(np.int64), "L": L, "s": sign_from_log(L), "S_wide": S_wide,
            "V": V, "H": H}


def project_log_in_pass(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """V = 1 producer: ACAM LOG applied to the crossbar's int32 y in the projection pass."""
    y = ref.compute_y(np.asarray(W, dtype=np.int8), np.asarray(X, dtype=np.int8))
    return _log_i8(y)


def signed_tile(LA: np.ndarray, sA: np.ndarray, LB: np.ndarray, sB: np.ndarray
                ) -> tuple[np.ndarray, int]:
    """Signed DIMM E stage + exact reduce over the inner dimension k.

    LA, sA [M, K]; LB, sB [K, N] (log bytes, sign bits). Returns (acc int64 [M, N], wraps) where
    wraps counts the feeds u = LA + LB outside int8 (truncated by F7, as in the certified DIMM).
    """
    LA = np.asarray(LA, dtype=np.int64)
    LB = np.asarray(LB, dtype=np.int64)
    sA = np.asarray(sA, dtype=np.int64)
    sB = np.asarray(sB, dtype=np.int64)
    M, K = LA.shape
    N = LB.shape[1]
    acc = np.zeros((M, N), dtype=np.int64)
    wraps = 0
    for k in range(K):
        u = LA[:, k][:, None] + LB[k][None, :]
        wraps += int(((u < -128) | (u > 127)).sum())
        e = _exp_u8(_feed8(u))
        neg = (sA[:, k][:, None] ^ sB[k][None, :]).astype(bool)
        acc += np.where(neg, -e, e)
    assert np.abs(acc).max(initial=0) < 2 ** 31, "tile accumulator exceeds int32"
    return acc, wraps


def ocarry_stage_values(o_part: np.ndarray, log_a: np.ndarray, LN: int, SO: int) -> dict:
    """D6 o-carry per block. o_part int [B, S, d]; log_a int [B, S] (0 at b = 0).

    Returns per block b: t, e (0 at b = 0), o_new (o'), oq, so, lo — all [B, S, d] — and the
    final (lo, so) [S, d]."""
    op = np.asarray(o_part, dtype=np.int64)
    la = np.asarray(log_a, dtype=np.int64)
    B = op.shape[0]
    keys = ("t", "e", "o_new", "oq", "so", "lo")
    st = {k: np.zeros(op.shape, dtype=np.int64) for k in keys}
    lo = so = None
    for b in range(B):
        if b == 0:
            t = np.zeros(op.shape[1:], dtype=np.int64)
            e = np.zeros(op.shape[1:], dtype=np.int64)
            o_new = op[0].copy()
        else:
            t = np.clip(la[b][:, None] + lo, -128, 127)
            e = _exp_u8(t)
            o_new = (np.where(so.astype(bool), -e, e) << SO) + op[b]
        assert np.abs(o_new).max(initial=0) < 2 ** 31, "o-carry exceeds int32"
        so = (o_new < 0).astype(np.int64)
        oq = np.minimum(np.abs(o_new) >> SO, 127)
        lo = _log_i8(oq)
        for k, v in zip(keys, (t, e, o_new, oq, so, lo)):
            st[k][b] = v
    return {**st, "lo_final": lo, "so_final": so}


def epilogue(lo: np.ndarray, so: np.ndarray, log_l: np.ndarray, LN: int, SO: int) -> dict:
    """ES + EE: O8 = sign(o) * EXP_u8(clamp8(lo - log_l)) * 2^(SO - LN), saturated to int8."""
    lo = np.asarray(lo, dtype=np.int64)
    tE = np.clip(lo - np.asarray(log_l, dtype=np.int64)[:, None], -128, 127)
    eE = _scale_pow2(_exp_u8(tE), SO - LN)
    return {"tE": tE, "eE": eE, "O8": _sat8_signed(np.asarray(so), eE)}


def _operands(x8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = _log_i8(x8)
    return L, sign_from_log(L)


def head_from_qkv(q8: np.ndarray, k8: np.ndarray, v8: np.ndarray, Bkv: int,
                  score_shift: int = SCORE_SHIFT, o_shift: int | None = None) -> dict:
    """The head from the projected int8 operands q8, k8, v8 [n, d] to O8 [n, d] (contract).
    All n query rows are processed at once (rows are independent; runs of S rows give the same
    values — self-test)."""
    q8, k8, v8 = (np.asarray(a, dtype=np.int64) for a in (q8, k8, v8))
    n, d = q8.shape
    assert k8.shape == (n, d) and v8.shape == (n, d), "q, k, v must be [n_seq, d]"
    check_domain(n, d, 1, Bkv)
    LN = n.bit_length() - 1
    SO = LN if o_shift is None else o_shift
    B = n // Bkv
    LQ, sQ = _operands(q8)
    LK, sK = _operands(k8)
    LV, sV = _operands(v8)
    acc, qk_wraps = signed_tile(LQ, sQ, LK.T, sK.T)
    s8 = narrow(acc, score_shift)
    sm = sor.logcarry_stage_values(s8, Bkv)
    o_part = np.zeros((B, n, d), dtype=np.int64)
    pv_wraps = 0
    zeros = np.zeros((n, Bkv), dtype=np.int64)
    for b in range(B):
        kb = slice(b * Bkv, (b + 1) * Bkv)
        o_part[b], w = signed_tile(sm["logp"][b], zeros, LV[kb], sV[kb])
        pv_wraps += w
    oc = ocarry_stage_values(o_part, sm["log_a"], LN, SO)
    ep = epilogue(oc["lo_final"], oc["so_final"], sm["log_l_final"], LN, SO)
    return {"n_seq": n, "d": d, "Bkv": Bkv, "B": B, "LN": LN, "SO": SO,
            "score_shift": score_shift,
            "LQ": LQ, "sQ": sQ, "LK": LK, "sK": sK, "LV": LV, "sV": sV,
            "acc": acc, "s8": s8, "softmax": sm, "o_part": o_part, "ocarry": oc, **ep,
            "wraps": {"qk": qk_wraps, "pv": pv_wraps}}


def head_stages(X: np.ndarray, WQ: np.ndarray, WK: np.ndarray, WV: np.ndarray, WO: np.ndarray,
                Bkv: int, R: int = 256, C: int = 256, score_shift: int = SCORE_SHIFT,
                o_shift: int | None = None) -> dict:
    """Full head (contract): X int8 [n, dm]; WQ/WK/WV int8 [dm, d]; WO int8 [d, dm].
    (R, C) set the tiling only; values are geometry-independent."""
    X = np.asarray(X, dtype=np.int8)
    n, dm = X.shape
    d = np.shape(WQ)[1]
    check_domain(n, d, dm, Bkv)
    pq, pk, pv = (project(X, W, R, C) for W in (WQ, WK, WV))
    hd = head_from_qkv(pq["x8"], pk["x8"], pv["x8"], Bkv, score_shift, o_shift)
    S_wide, y8 = gemm_ref.compute_gemm(np.asarray(WO, dtype=np.int8),
                                       hd["O8"].astype(np.int8), R, C)
    for name, p in (("Q", pq), ("K", pk), ("V", pv)):
        assert np.array_equal(p["L"], hd["L" + name]), "projection log byte mismatch"
    return {**hd, "q8": pq["x8"], "k8": pk["x8"], "v8": pv["x8"],
            "oproj_wide": S_wide, "y8": y8.astype(np.int64),
            "geometry": geometry(n, d, dm, Bkv, R, C), "R": R, "C": C}


def layer_stages(X: np.ndarray, WQ: np.ndarray, WK: np.ndarray, WV: np.ndarray, WO: np.ndarray,
                 Bkv: int, R: int = 256, C: int = 256, score_shift: int = SCORE_SHIFT,
                 o_shift: int | None = None) -> dict:
    """Multi-head attention layer (contract, M5): X int8 [n, dm]; WQ / WK / WV int8 [H, dm, d] (one set per head);
    WO int8 [H*d, dm]. Every head is the head contract up to O8 with its own projections (heads independent); the
    O-projection takes the concatenated head outputs: y8 = trunc8(concat_h(O8_h) @ WO). (R, C) set the tiling only."""
    X = np.asarray(X, dtype=np.int8)
    WQ, WK, WV = (np.asarray(w, dtype=np.int8) for w in (WQ, WK, WV))
    H, dm, d = WQ.shape
    assert WK.shape == WQ.shape == WV.shape and X.shape[1] == dm, "per-head weights [H, dm, d]"
    assert np.shape(WO) == (H * d, np.shape(WO)[1]), "WO must be [H*d, dm_out]"
    n = X.shape[0]
    check_domain(n, d, dm, Bkv)
    heads = []
    for h in range(H):
        pq, pk, pv = (project(X, W[h], R, C) for W in (WQ, WK, WV))
        hd = head_from_qkv(pq["x8"], pk["x8"], pv["x8"], Bkv, score_shift, o_shift)
        heads.append({**hd, "q8": pq["x8"], "k8": pk["x8"], "v8": pv["x8"]})
    O8cat = np.concatenate([hd["O8"] for hd in heads], axis=1)
    S_wide, y8 = gemm_ref.compute_gemm(np.asarray(WO, dtype=np.int8), O8cat.astype(np.int8), R, C)
    return {"H": H, "n_seq": n, "d": d, "dm": dm, "Bkv": Bkv, "heads": heads, "O8cat": O8cat,
            "oproj_wide": S_wide, "y8": y8.astype(np.int64), "R": R, "C": C,
            "SO": heads[0]["SO"], "LN": heads[0]["LN"], "score_shift": score_shift}


# ---------------------------------------------------------------------------
# Dual witness: per run, per tile, every ACAM through identity crossbar passes
# ---------------------------------------------------------------------------
def _xbar_tile_pass(Wt: np.ndarray, x: np.ndarray, mode: int, R: int, C: int) -> np.ndarray:
    """One weight-tile pass: R-row window x (zero-padded), C-column tile Wt (zero-padded)."""
    win = np.zeros((1, R), dtype=np.int8)
    win[0, :x.size] = x
    Wp = np.zeros((R, C), dtype=np.int8)
    Wp[:Wt.shape[0], :Wt.shape[1]] = Wt
    return ref.acam_transform(ref.compute_y(Wp, win), mode)[0].astype(np.int64)


def _proj_pass_view(X: np.ndarray, W: np.ndarray, R: int, C: int, passes: dict
                    ) -> np.ndarray:
    """Log bytes of X @ W the way the stations produce them (charter §3.3)."""
    n, dm = X.shape
    d = W.shape[1]
    V, H = -(-dm // R), -(-d // C)
    if V == 1:                                  # LOG in the projection pass, one pass per tile
        L = np.zeros((n, d), dtype=np.int64)
        for t in range(n):
            for h in range(H):
                cols = slice(h * C, min((h + 1) * C, d))
                L[t, cols] = _xbar_tile_pass(W[:, cols], X[t], ref.MODE_LOG, R, C)[:cols.stop - cols.start]
                passes["proj"] += 1
        return L
    x8 = np.zeros((n, d), dtype=np.int64)       # V > 1: REGULAR tiles, byte tree, then a pool
    for t in range(n):
        for h in range(H):
            cols = slice(h * C, min((h + 1) * C, d))
            tree = np.zeros(cols.stop - cols.start, dtype=np.int64)
            for v in range(V):
                rows = slice(v * R, min((v + 1) * R, dm))
                tree += _xbar_tile_pass(W[rows, cols], X[t, rows], ref.MODE_REGULAR, R, C
                                        )[:cols.stop - cols.start]
                passes["proj"] += 1
            x8[t, cols] = ref.trunc8(tree)
    L, p = ref.convert_stream(x8.reshape(-1).astype(np.int8), ref.MODE_LOG, R, C)
    passes["pool"] += p
    return L.astype(np.int64).reshape(n, d)


def _tile_pass_view(LA: np.ndarray, sA: np.ndarray, LB: np.ndarray, sB: np.ndarray,
                    R: int, C: int) -> tuple[np.ndarray, int]:
    """One tile's E farm: the rank-1 stream over k (k-major, then m, then n; dimm.md §2.1),
    packed in windows of I, then the signed reduce by index."""
    M, K = LA.shape
    N = LB.shape[1]
    u = LA.T[:, :, None] + LB[:, None, :]                      # [K, M, N]
    neg = (sA.T[:, :, None] ^ sB[:, None, :]).astype(bool)
    e, p = ref.convert_stream(ref.trunc8(u.reshape(-1)), ref.MODE_EXP, R, C)
    eu = e.view(np.uint8).astype(np.int64).reshape(K, M, N)
    return np.where(neg, -eu, eu).sum(axis=0), p


def head_pass_view(X: np.ndarray, WQ: np.ndarray, WK: np.ndarray, WV: np.ndarray,
                   WO: np.ndarray, Bkv: int, R: int, C: int, score_shift: int = SCORE_SHIFT,
                   o_shift: int | None = None) -> dict:
    """Dual witness of `head_stages`: runs of S rows, (run, block) tiles, every conversion an
    identity pass of capacity I (convert_stream), the softmax through its certified pass view.
    Returns the comparable stages and the pass counts per station."""
    X = np.asarray(X, dtype=np.int8)
    WQ, WK, WV, WO = (np.asarray(w, dtype=np.int8) for w in (WQ, WK, WV, WO))
    n, dm = X.shape
    check_domain(n, WQ.shape[1], dm, Bkv, R, C)
    passes = {k: 0 for k in ("proj", "pool", "qk_E", "pv_E", "oc_exp", "oc_log", "oproj")}
    LQ, LK, LV = (_proj_pass_view(X, W, R, C, passes) for W in (WQ, WK, WV))
    return _head_pass_core(LQ, LK, LV, WO, Bkv, R, C, score_shift, o_shift, passes)


def head_pass_view_qkv(q8: np.ndarray, k8: np.ndarray, v8: np.ndarray, WO: np.ndarray,
                       Bkv: int, R: int, C: int, score_shift: int = SCORE_SHIFT,
                       o_shift: int | None = None) -> dict:
    """Pass view from given int8 operands (adversarial classes): each operand stream is
    converted by a LOG pool (convert_stream), then the same per-run / per-tile core."""
    n, d = np.shape(q8)
    WO = np.asarray(WO, dtype=np.int8)
    check_domain(n, d, WO.shape[1], Bkv, R, C)
    passes = {k: 0 for k in ("proj", "pool", "qk_E", "pv_E", "oc_exp", "oc_log", "oproj")}
    Ls = []
    for x in (q8, k8, v8):
        L, p = ref.convert_stream(np.asarray(x, dtype=np.int8).reshape(-1), ref.MODE_LOG, R, C)
        passes["pool"] += p
        Ls.append(L.astype(np.int64).reshape(n, d))
    return _head_pass_core(*Ls, WO, Bkv, R, C, score_shift, o_shift, passes)


def _head_pass_core(LQ: np.ndarray, LK: np.ndarray, LV: np.ndarray, WO: np.ndarray, Bkv: int,
                    R: int, C: int, score_shift: int, o_shift: int | None, passes: dict) -> dict:
    n, d = LQ.shape
    dm = WO.shape[1]
    g = geometry(n, d, dm, Bkv, R, C)
    I, S, Rb, B = g["I"], g["S"], g["Rb"], g["B"]
    LN = n.bit_length() - 1
    SO = LN if o_shift is None else o_shift
    sQ, sK, sV = (sign_from_log(L) for L in (LQ, LK, LV))

    acc = np.zeros((n, n), dtype=np.int64)
    s8 = np.zeros((n, n), dtype=np.int64)
    logp = np.zeros((B, n, Bkv), dtype=np.int64)
    log_l = np.zeros(n, dtype=np.int64)
    o_part = np.zeros((B, n, d), dtype=np.int64)
    oq_all = np.zeros((B, n, d), dtype=np.int64)
    lo_fin = np.zeros((n, d), dtype=np.int64)
    so_fin = np.zeros((n, d), dtype=np.int64)
    O8 = np.zeros((n, d), dtype=np.int64)
    for r in range(Rb):
        rows = slice(r * S, (r + 1) * S)
        for b in range(B):                                     # QK tiles of the run
            kb = slice(b * Bkv, (b + 1) * Bkv)
            a, p = _tile_pass_view(LQ[rows], sQ[rows], LK[kb].T, sK[kb].T, R, C)
            acc[rows, kb] = a
            passes["qk_E"] += p
        s8[rows] = narrow(acc[rows], score_shift)
        sm = sor.logcarry_pass_view(s8[rows], Bkv, R, C)       # certified pass witness
        logp[:, rows] = sm["logp"]
        log_l[rows] = sm["log_l_final"]
        m = None                                               # RM / NM / LS (peripheral)
        lo = so = None
        for b in range(B):
            kb = slice(b * Bkv, (b + 1) * Bkv)
            m_blk = s8[rows, kb].max(axis=1)
            m_new = m_blk if b == 0 else np.maximum(m, m_blk)
            la = np.zeros(S, dtype=np.int64) if b == 0 else m - m_new
            assert np.array_equal(sm["logp"][b], np.maximum(s8[rows, kb] - m_new[:, None], -128))
            op, p = _tile_pass_view(sm["logp"][b], np.zeros((S, Bkv), dtype=np.int64),
                                    LV[kb], sV[kb], R, C)
            o_part[b, rows] = op
            passes["pv_E"] += p
            if b == 0:
                o_new = op
            else:
                t = np.clip(la[:, None] + lo, -128, 127)
                e, p = ref.convert_stream(t.reshape(-1).astype(np.int8), ref.MODE_EXP, R, C)
                passes["oc_exp"] += p
                eu = e.view(np.uint8).astype(np.int64).reshape(S, d)
                o_new = (np.where(so.astype(bool), -eu, eu) << SO) + op
            so = (o_new < 0).astype(np.int64)
            oq = np.minimum(np.abs(o_new) >> SO, 127)
            lg, p = ref.convert_stream(oq.reshape(-1).astype(np.int8), ref.MODE_LOG, R, C)
            passes["oc_log"] += p
            lo = lg.astype(np.int64).reshape(S, d)
            oq_all[b, rows] = oq
            m = m_new
        tE = np.clip(lo - sm["log_l_final"][:, None], -128, 127)   # epilogue EXP set
        e, p = ref.convert_stream(tE.reshape(-1).astype(np.int8), ref.MODE_EXP, R, C)
        passes["oc_exp"] += p
        eE = _scale_pow2(e.view(np.uint8).astype(np.int64).reshape(S, d), SO - LN)
        O8[rows] = _sat8_signed(so, eE)
        lo_fin[rows], so_fin[rows] = lo, so

    V_O, H_O = g["V_O"], g["H_O"]                               # O-projection, per token
    y8 = np.zeros((n, dm), dtype=np.int64)
    for t in range(n):
        for h in range(H_O):
            cols = slice(h * C, min((h + 1) * C, dm))
            tree = np.zeros(cols.stop - cols.start, dtype=np.int64)
            for v in range(V_O):
                rws = slice(v * R, min((v + 1) * R, d))
                tree += _xbar_tile_pass(WO[rws, cols], O8[t, rws].astype(np.int8),
                                        ref.MODE_REGULAR, R, C)[:cols.stop - cols.start]
                passes["oproj"] += 1
            y8[t, cols] = ref.trunc8(tree)
    return {"LQ": LQ, "sQ": sQ, "LK": LK, "sK": sK, "LV": LV, "sV": sV, "acc": acc, "s8": s8,
            "logp": logp, "log_l_final": log_l, "o_part": o_part, "oq": oq_all,
            "lo_final": lo_fin, "so_final": so_fin, "O8": O8, "y8": y8, "passes": passes,
            "geometry": g}


def expected_passes(n_seq: int, d: int, dm: int, Bkv: int, R: int, C: int) -> dict:
    """Closed-form pass counts per station (charter §6), for the pass view's counters."""
    g = geometry(n_seq, d, dm, Bkv, R, C)
    I, S, Rb, B, V, H = g["I"], g["S"], g["Rb"], g["B"], g["V"], g["H"]
    P_mm = -(-S * Bkv * d // I)
    P_oc = -(-S * d // I)
    return {"proj": 3 * n_seq * V * H,
            "pool": 3 * -(-n_seq * d // I) if V > 1 else 0,
            "qk_E": Rb * B * P_mm, "pv_E": Rb * B * P_mm,
            "oc_exp": Rb * B * P_oc, "oc_log": Rb * B * P_oc,
            "oproj": n_seq * g["V_O"] * g["H_O"]}


# ---------------------------------------------------------------------------
# Topology witness: the same dataflow with pluggable arithmetic
# ---------------------------------------------------------------------------
class _IntOps:
    """The integer contract (P24 ACAM forms, trunc8 feeds, saturations, clamps)."""
    name = "int"

    @staticmethod
    def log_sign(x):
        L = _log_i8(x)
        return L, sign_from_log(L).astype(bool)

    tile_exp = staticmethod(lambda u: _exp_u8(_feed8(u)))
    narrow = staticmethod(lambda acc, sh: np.clip(acc >> sh, -128, 127))
    logp = staticmethod(lambda s, m: np.maximum(s - m, -128))
    exp = staticmethod(_exp_u8)
    clamp8 = staticmethod(lambda x: np.clip(x, -128, 127))
    compact = staticmethod(lambda x, sh: np.minimum(x >> sh, 127))
    expand = staticmethod(lambda e, sh: e << sh)
    log_c = staticmethod(_log_i8)
    scale = staticmethod(_scale_pow2)
    out = staticmethod(lambda neg, mag: _sat8_signed(neg, mag))


class _FloatOps:
    """Real exp / log, no saturation: the dataflow must then be textbook attention."""
    name = "float"

    @staticmethod
    def log_sign(x):
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.log(np.abs(x)), x < 0

    tile_exp = staticmethod(np.exp)
    narrow = staticmethod(lambda acc, sh: acc * 2.0 ** -sh)
    logp = staticmethod(lambda s, m: s - m)
    exp = staticmethod(np.exp)
    clamp8 = staticmethod(lambda x: x)
    compact = staticmethod(lambda x, sh: x * 2.0 ** -sh)
    expand = staticmethod(lambda e, sh: e * 2.0 ** sh)

    @staticmethod
    def log_c(x):
        with np.errstate(divide="ignore"):
            return np.log(x)

    scale = staticmethod(lambda e, k: e * 2.0 ** k)
    out = staticmethod(lambda neg, mag: np.where(neg, -mag, mag))


INT_OPS, FLOAT_OPS = _IntOps(), _FloatOps()


def head_mirror(q, k, v, Bkv: int, ops=INT_OPS, score_shift: int = SCORE_SHIFT,
                o_shift: int | None = None) -> dict:
    """The FA log-carry dataflow (flash_attention.md §1 node order), written once, per key block,
    with pluggable arithmetic. INT_OPS must reproduce `head_from_qkv` bit-exactly; FLOAT_OPS must
    reproduce softmax(q k^T 2^-score_shift) v. Independent code from the contract path."""
    n, d = np.shape(q)
    LN = n.bit_length() - 1
    SO = LN if o_shift is None else o_shift
    LQ, sQ = ops.log_sign(q)
    LK, sK = ops.log_sign(k)
    LV, sV = ops.log_sign(v)
    m = ll = lo = so = None
    for b in range(n // Bkv):
        kb = slice(b * Bkv, (b + 1) * Bkv)
        mag = ops.tile_exp(LQ[:, None, :] + LK[kb][None, :, :])            # PA1 + PE1
        s = ops.narrow(np.where(sQ[:, None, :] ^ sK[kb][None, :, :], -mag, mag)
                       .sum(axis=2), score_shift)                            # PX1 + PS1 + D1
        m_new = s.max(axis=1) if b == 0 else np.maximum(m, s.max(axis=1))   # RM + NM
        la = None if b == 0 else m - m_new                                   # LS
        lp = ops.logp(s, m_new[:, None])                                     # LP
        sp = ops.exp(lp).sum(axis=1)                                         # EP + PS2
        l1 = sp if b == 0 else ops.expand(ops.exp(ops.clamp8(la + ll)), LN) + sp   # LA EA LB
        ll = ops.log_c(ops.compact(l1, LN))                                  # LG
        mag = ops.tile_exp(lp[:, :, None] + LV[kb][None, :, :])            # PV1 + PV2
        op = np.where(sV[kb][None, :, :], -mag, mag).sum(axis=1)            # PV3 + PV4
        if b == 0:
            o1 = op
        else:
            e = ops.expand(ops.exp(ops.clamp8(la[:, None] + lo)), SO)       # OA + OE
            o1 = np.where(so, -e, e) + op                                    # OB
        so = o1 < 0                                                          # SN
        lo = ops.log_c(ops.compact(np.abs(o1), SO))                         # OG
        m = m_new
    eE = ops.scale(ops.exp(ops.clamp8(lo - ll[:, None])), SO - LN)          # ES + EE
    return {"O": ops.out(so, eE), "m": m, "log_l": ll, "lo": lo, "so": so}


def textbook_attention(q, k, v, score_shift: int = SCORE_SHIFT) -> np.ndarray:
    """softmax((q k^T) 2^-score_shift) v in float64 (golden witness, not contract)."""
    s = (np.asarray(q, dtype=np.float64) @ np.asarray(k, dtype=np.float64).T) * 2.0 ** -score_shift
    p = np.exp(s - s.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)) @ np.asarray(v, dtype=np.float64)


def notebook_functions() -> dict:
    """The float golden functions of the FA notebook (function definitions only)."""
    nb = json.loads(NOTEBOOK.read_text())
    ns: dict = {"np": np}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if defs:
            exec(compile(ast.Module(body=defs, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
    return ns


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def value_stats(st: dict) -> dict:
    """Fractions of values touched by a saturation, clamp or feed wrap (from `head_from_qkv`)."""
    n, d, B, LN, SO = st["n_seq"], st["d"], st["B"], st["LN"], st["SO"]
    sm, oc = st["softmax"], st["ocarry"]
    raw = st["acc"] >> st["score_shift"]
    s8 = st["s8"]
    blocks = s8.reshape(n, B, st["Bkv"]).transpose(1, 0, 2)
    lp_raw = blocks - sm["m_new"][:, :, None]
    l_t_raw = sm["log_a"][1:] + sm["log_l"][:-1]
    o_t_raw = sm["log_a"][1:][:, :, None] + oc["lo"][:-1]
    tE_raw = oc["lo_final"] - sm["log_l_final"][:, None]
    feeds = n * n * d + B * n * st["Bkv"] * d
    nb1 = max(B - 1, 1)
    return {
        "score_clip": float(((raw < -128) | (raw > 127)).mean()),
        "feed_wrap": (st["wraps"]["qk"] + st["wraps"]["pv"]) / feeds,
        "logp_clamp": float((lp_raw < -128).mean()),
        "l_t_clamp": float(((l_t_raw < -128) | (l_t_raw > 127)).sum() / (nb1 * n)),
        "lq_sat": float(((sm["lprime"] >> LN) > 127).mean()),
        "o_t_clamp": float(((o_t_raw < -128) | (o_t_raw > 127)).sum() / (nb1 * n * d)),
        "oq_sat": float(((np.abs(oc["o_new"]) >> SO) > 127).mean()),
        "epi_clamp": float(((tE_raw < -128) | (tE_raw > 127)).mean()),
        "O_sat": float(((st["eE"] > 127) & (st["ocarry"]["so_final"] == 0)).mean()
                       + ((st["eE"] > 128) & (st["ocarry"]["so_final"] == 1)).mean()),
    }


def drift_report(q8, k8, v8, Bkv: int, **kw) -> dict:
    """Blocked (Bkv) vs single-block (Bkv = n_seq, wide int32 o, one compaction) on O8."""
    a = head_from_qkv(q8, k8, v8, Bkv, **kw)["O8"]
    g = head_from_qkv(q8, k8, v8, np.shape(q8)[0], **kw)["O8"]
    dd = np.abs(a - g)
    return {"equal_frac": float((dd == 0).mean()), "mean_abs": float(dd.mean()),
            "max_abs": int(dd.max())}


# ---------------------------------------------------------------------------
# Stimuli
# ---------------------------------------------------------------------------
def _jump_operands(n: int, d: int, Bkv: int) -> tuple[np.ndarray, np.ndarray]:
    """q, k with scores that start at the most negative product and rise block by block
    (large negative log_a at every max change). q = 2 everywhere; each key block uses one
    k value, ordered by the signed product (-1)^(k<0) * EXP_u8(trunc8(LOG(2) + LOG(k)))."""
    q = np.full((n, d), 2, dtype=np.int64)
    kv = np.arange(-128, 128, dtype=np.int64)
    prod = np.where(kv < 0, -1, 1) * _exp_u8(_feed8(_log_i8(np.array([2])) + _log_i8(kv)))
    order = kv[np.argsort(prod, kind="stable")]
    B = n // Bkv
    k = np.zeros((n, d), dtype=np.int64)
    for b in range(B):
        k[b * Bkv:(b + 1) * Bkv] = order[(b * (order.size - 1)) // max(B - 1, 1)]
    return q, k


def classes(n: int, d: int, Bkv: int, rng: np.random.Generator) -> dict:
    """Operand classes (q8, k8, v8) for the head tests."""
    r = lambda lo, hi: rng.integers(lo, hi + 1, size=(n, d))          # noqa: E731
    ext = np.tile(np.array([[127, -128], [-128, 127]]), (-(-n // 2), -(-d // 2)))[:n, :d]
    qj, kj = _jump_operands(n, d, Bkv)
    return {
        "random": (r(-128, 127), r(-128, 127), r(-128, 127)),
        "narrow16": (r(-16, 16), r(-16, 16), r(-16, 16)),
        "zeros": (np.zeros((n, d), np.int64),) * 3,
        "extremes": (ext, ext[::-1].copy(), ext),
        "ties": (np.full((n, d), 5), np.full((n, d), 5), r(-128, 127)),
        "negV": (r(-128, 127), r(-128, 127), r(-128, -1)),
        "jumps": (qj, kj, r(-128, 127)),
    }


# ---------------------------------------------------------------------------
# Self-test — run:  python3 attention_ref.py [--strict-r2]  (the R2 saturation bar is reported;
# --strict-r2 makes it fatal)
# ---------------------------------------------------------------------------
def _assert_same(a: dict, b: dict, keys, tag) -> None:
    for k in keys:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), (tag, k)


def _self_test(strict_r2: bool = False) -> None:
    rng = np.random.default_rng(2026_10_08)
    counts = {}

    # 1) D3 sign rule over all 256 bytes; the "bit 7 only" shortcut fails at exactly 0 and -128
    x = np.arange(-128, 128)
    L = _log_i8(x)
    assert np.array_equal(sign_from_log(L), (x < 0).astype(np.int64))
    wrong = x[((L & 0xFF) >> 7) != (x < 0)]
    assert sorted(wrong.tolist()) == [-128, 0], wrong
    counts["sign bytes"] = 256

    # 2) projections: geometry invariance (V=1 / V>1 / H>1) and LOG(y) == LOG(trunc8(y))
    n = 0
    for dm, d in ((32, 16), (96, 32), (128, 64), (192, 64), (64, 128), (300, 70)):
        X = rng.integers(-128, 128, size=(16, dm))
        W = rng.integers(-128, 128, size=(dm, d))
        base = project(X, W, 4096, 4096)
        assert np.array_equal(project_log_in_pass(X, W), base["L"]), (dm, d)
        for R, C in ((64, 32), (64, 64), (128, 64), (128, 128), (256, 64), (256, 256), (512, 128)):
            p = project(X, W, R, C)
            assert np.array_equal(p["x8"], base["x8"]) and np.array_equal(p["L"], base["L"])
            assert np.array_equal(p["s"], (p["x8"] < 0).astype(np.int64))
            n += 1
    counts["projection geometries"] = n

    # 3) signed tile: all-positive == certified DIMM; vs a scalar loop with signs
    n = 0
    for M, K, N in ((8, 16, 8), (16, 64, 16), (32, 32, 64), (5, 7, 3)):
        A = rng.integers(-128, 128, size=(M, K))
        Bm = rng.integers(-128, 128, size=(K, N))
        LA, LB = _log_i8(A), _log_i8(Bm)
        z = lambda s: np.zeros(s, dtype=np.int64)           # noqa: E731
        acc, _ = signed_tile(LA, z((M, K)), LB, z((K, N)))
        assert np.array_equal(acc, dimm_ref.dimm_matmul(A.astype(np.int8), Bm.astype(np.int8)))
        assert np.array_equal(acc, dimm_ref._reduce(LA.astype(np.int8), LB.astype(np.int8))[0])
        sA, sB = sign_from_log(LA), sign_from_log(LB)
        acc, _ = signed_tile(LA, sA, LB, sB)
        for i in range(M):
            for j in range(N):
                ref_ij = 0
                for kk in range(K):
                    u = int(LA[i, kk]) + int(LB[kk, j])
                    u8 = ((u + 128) & 0xFF) - 128
                    e = (1 + u8 + (u8 * u8) // 2) & 0xFF
                    ref_ij += -e if (A[i, kk] < 0) != (Bm[kk, j] < 0) else e
                assert acc[i, j] == ref_ij, (M, K, N, i, j)
        n += 1
    counts["signed tiles"] = n

    # 4) contract == dual pass view (incl. pass counts) on micro geometries: V=1, V>1, H>1, V_O>1,
    #    R_b > 1, I < n_seq, every class
    n = 0
    geos = [  # (n_seq, d, dm, Bkv, R, C)
        (16, 8, 16, 8, 64, 64), (32, 16, 48, 8, 64, 64), (32, 16, 96, 16, 64, 64),   # V = 2
        (64, 16, 32, 8, 32, 32),                         # I = 32 < n_seq: R_b = 2
        (32, 40, 32, 8, 64, 32),                         # H = 2 (d > C), d > I
        (16, 72, 24, 8, 64, 64),                         # V_O = 2 (d > R), H = 2
        (32, 16, 80, 8, 64, 32),                         # V = 2, H_O = 3
    ]
    for (ns, d, dm, Bkv, R, C) in geos:
        for name, (q8, k8, v8) in classes(ns, d, Bkv, rng).items():
            X = rng.integers(-128, 128, size=(ns, dm))
            Ws = [rng.integers(-128, 128, size=(dm, d)) for _ in range(3)]
            WO = rng.integers(-128, 128, size=(d, dm))
            a = head_stages(X, *Ws, WO, Bkv, R, C)
            b = head_pass_view(X, *Ws, WO, Bkv, R, C)
            _assert_same(a, b, ("LQ", "sQ", "LK", "sK", "LV", "sV", "acc", "s8", "o_part",
                                "O8", "y8"), (ns, d, dm, Bkv, R, C, name))
            _assert_same(a["softmax"], b, ("logp", "log_l_final"), (ns, d, dm, name))
            _assert_same(a["ocarry"], b, ("oq", "lo_final", "so_final"), (ns, d, dm, name))
            assert b["passes"] == expected_passes(ns, d, dm, Bkv, R, C), \
                (ns, d, dm, Bkv, R, C, b["passes"], expected_passes(ns, d, dm, Bkv, R, C))
            n += 1
            break                                    # one projected run per geometry here
        # every operand class through the contract, the pass view and the integer mirror
        for name, (q8, k8, v8) in classes(ns, d, Bkv, rng).items():
            WO = rng.integers(-128, 128, size=(d, dm))
            st = head_from_qkv(q8, k8, v8, Bkv)
            pv = head_pass_view_qkv(q8, k8, v8, WO, Bkv, R, C)
            _assert_same(st, pv, ("LQ", "sQ", "LK", "sK", "LV", "sV", "acc", "s8", "o_part",
                                  "O8"), (ns, d, Bkv, R, C, name))
            _assert_same(st["ocarry"], pv, ("oq", "lo_final", "so_final"), (ns, d, name))
            _, y8 = gemm_ref.compute_gemm(WO.astype(np.int8), st["O8"].astype(np.int8), R, C)
            assert np.array_equal(pv["y8"], y8), (ns, d, name, "y8")
            mi = head_mirror(q8, k8, v8, Bkv, INT_OPS)
            assert np.array_equal(mi["O"], st["O8"]), (ns, d, Bkv, name)
            assert np.array_equal(mi["lo"], st["ocarry"]["lo_final"]), (ns, d, Bkv, name)
            n += 1
    counts["dual-view + mirror micro cases"] = n

    # 5) invariants and closed forms on every class, d in {32, 64, 128}
    n = 0
    for ns, d, Bkv in ((64, 32, 16), (128, 64, 16), (64, 128, 8), (256, 64, 16)):
        for name, (q8, k8, v8) in classes(ns, d, Bkv, rng).items():
            st = head_from_qkv(q8, k8, v8, Bkv)
            sm, oc = st["softmax"], st["ocarry"]
            assert (sm["logp"] <= 0).all() and (sm["logp"] >= -128).all()
            assert (sm["log_a"] <= 0).all() and (sm["log_a"][0] == 0).all()
            assert (oc["oq"] >= 0).all() and (oc["oq"] <= 127).all()
            assert (oc["lo"] >= -1).all() and (oc["lo"] <= 126).all()
            assert (oc["e"][0] == 0).all() and np.array_equal(oc["o_new"][0], st["o_part"][0])
            assert (st["O8"] >= -128).all() and (st["O8"] <= 127).all()
            assert np.array_equal(st["sV"], (np.asarray(v8) < 0).astype(np.int64))
            if name == "zeros":                      # closed form (module docstring arithmetic)
                assert (st["acc"] == d).all() and (st["s8"] == min(d >> SCORE_SHIFT, 127)).all()
                assert (st["o_part"] == 0).all() and (st["O8"] == 1).all()
            if name == "negV":
                assert (st["sV"] == 1).all()
            if name == "ties":
                assert (st["s8"] == st["s8"][0, 0]).all() and (sm["log_a"] == 0).all()
            if name == "jumps":
                assert sm["log_a"].min() <= -64, (ns, d, sm["log_a"].min())
            mi = head_mirror(q8, k8, v8, Bkv, INT_OPS)
            assert np.array_equal(mi["O"], st["O8"]), (ns, d, Bkv, name)
            h = head_from_qkv(q8, k8, v8, Bkv)       # determinism
            assert np.array_equal(h["O8"], st["O8"])
            n += 1
    counts["invariant classes"] = n

    # 6) rows are independent (runs of S rows); Bkv = n_seq is softmax_rect + one PV DIMM
    n = 0
    for ns, d in ((64, 32), (128, 16), (256, 64)):
        for name, (q8, k8, v8) in classes(ns, d, 16, rng).items():
            full = head_from_qkv(q8, k8, v8, 16)
            S = ns // 4
            # rows of one run against all keys: recompute with the run's queries only
            LQ, sQ = _operands(np.asarray(q8)[S:2 * S])
            acc, _ = signed_tile(LQ, sQ, full["LK"].T, full["sK"].T)
            assert np.array_equal(acc, full["acc"][S:2 * S])
            sm = sor.logcarry_stage_values(narrow(acc), 16)
            assert np.array_equal(sm["log_l_final"], full["softmax"]["log_l_final"][S:2 * S])
            one = head_from_qkv(q8, k8, v8, ns)
            rect = sor.softmax_rect(full["s8"])
            op, _ = signed_tile(rect["exp_input"], np.zeros((ns, ns), np.int64),
                                full["LV"], full["sV"])
            LN = ns.bit_length() - 1
            assert np.array_equal(one["o_part"][0], op)
            oq = np.minimum(np.abs(op) >> LN, 127)
            so = (op < 0).astype(np.int64)
            ep = epilogue(_log_i8(oq), so, rect["log_output"], LN, LN)
            assert np.array_equal(one["O8"], ep["O8"]), (ns, d, name)
            n += 1
    counts["row-independence + Bkv=N reductions"] = n

    # 7) float topology witness: the mirror with real exp/log == textbook == notebook
    nb = notebook_functions()
    worst = 0.0
    n = 0
    for _ in range(120):
        ns = int(2 ** rng.integers(3, 8))
        d = int(rng.integers(1, 33))
        Bkv = int(2 ** rng.integers(3, ns.bit_length()))
        so_ = None if rng.random() < 0.5 else int(rng.integers(0, 12))
        q, k, v = (rng.standard_normal((ns, d)) * 3 for _ in range(3))
        got = head_mirror(q, k, v, Bkv, FLOAT_OPS, o_shift=so_)["O"]
        gold = textbook_attention(q, k, v)
        nbk = nb["attention_flash_logcarry"](q * 2.0 ** -SCORE_SHIFT, k, v, block=Bkv)
        nbr = nb["attention_reference"](q * 2.0 ** -SCORE_SHIFT, k, v)
        assert np.allclose(nbr, gold, rtol=1e-12, atol=1e-12)
        worst = max(worst, float(np.abs(got - gold).max()), float(np.abs(nbk - got).max()))
        n += 1
    assert worst < 1e-9, worst
    counts["float topology cases"] = n

    # 8a) layer (M5): H = 1 is the head; y8 == the low byte of the per-head partial O-projections' sum (exact
    #     integer, trunc8 a ring map); geometry independence; each head == head_stages with its own weights
    n = 0
    for (ns, d, dm, Bkv, H) in ((16, 8, 16, 8, 1), (32, 16, 32, 8, 2), (32, 16, 48, 8, 3), (64, 32, 96, 16, 3),
                                (32, 16, 24, 8, 2)):
        X = rng.integers(-128, 128, size=(ns, dm))
        WQ, WK, WV = (rng.integers(-128, 128, size=(H, dm, d)) for _ in range(3))
        WO = rng.integers(-128, 128, size=(H * d, dm))
        base = layer_stages(X, WQ, WK, WV, WO, Bkv, 4096, 4096)
        part = np.zeros((ns, dm), dtype=np.int64)
        for h in range(H):
            hs = head_stages(X, WQ[h], WK[h], WV[h], WO[h * d:(h + 1) * d], Bkv, 4096, 4096)
            assert np.array_equal(hs["O8"], base["heads"][h]["O8"]), (ns, d, H, h)
            assert np.array_equal(hs["s8"], base["heads"][h]["s8"]), (ns, d, H, h)
            part += hs["y8"]
        assert np.array_equal(((part + 128) & 0xFF) - 128, base["y8"]), (ns, d, dm, H)
        if H == 1:
            hs = head_stages(X, WQ[0], WK[0], WV[0], WO, Bkv, 4096, 4096)
            assert np.array_equal(hs["y8"], base["y8"])
        for R, C in ((64, 64), (32, 64), (64, 32), (128, 128)):
            if d > min(R, C):
                continue
            g = layer_stages(X, WQ, WK, WV, WO, Bkv, R, C)
            assert np.array_equal(g["y8"], base["y8"]) and np.array_equal(g["O8cat"], base["O8cat"])
        n += 1
    counts["layer cases"] = n

    # 8) domain guards
    for args in ((24, 8, 8, 8), (16, 8, 8, 4), (16, 8, 8, 32), (16, 0, 8, 8), (16, 8, 8, 12)):
        try:
            check_domain(*args)
        except ValueError:
            continue
        raise AssertionError(f"domain guard missed {args}")
    for R, C, Bkv in ((4, 64, 8), (64, 16, 32)):
        try:
            check_domain(64, 8, 8, Bkv, R, C)
        except ValueError:
            continue
        raise AssertionError(f"kernel domain guard missed R={R} C={C} Bkv={Bkv}")

    print("attention_ref: ALL PASS — contract checks (" + ", ".join(f"{v} {k}" for k, v in counts.items())
          + f"; float witness max |diff| {worst:.1e})")

    # 9) R2 bars: saturation / clamp / wrap rates (o-carry gated <= 1 % on random and ±16)
    print("  value stats, Bkv = 16 (fractions):")
    hdr = ("score_clip", "feed_wrap", "logp_clamp", "l_t_clamp", "lq_sat", "o_t_clamp",
           "oq_sat", "epi_clamp", "O_sat")
    print("    " + f"{'class':10s} {'n':>5s} {'d':>4s} " + " ".join(f"{h:>10s}" for h in hdr))
    bar_fail = []
    for ns in (64, 256, 1024):
        for d in (32, 64):
            cl = classes(ns, d, 16, rng)
            for name in ("random", "narrow16", "extremes", "negV", "jumps"):
                st = head_from_qkv(*cl[name], 16)
                vs = value_stats(st)
                print("    " + f"{name:10s} {ns:5d} {d:4d} "
                      + " ".join(f"{vs[h]:10.4f}" for h in hdr))
                if name in ("random", "narrow16") and max(vs["o_t_clamp"], vs["oq_sat"]) > 0.01:
                    bar_fail.append((name, ns, d, vs["o_t_clamp"], vs["oq_sat"]))
    if bar_fail:
        worst_sat = max(f[4] for f in bar_fail)
        print(f"  R2 o-carry saturation (reported; accepted as decision A, user 2026-10-08): above 1 % in "
              f"{len(bar_fail)} of 12 points (oq_sat up to {worst_sat:.3f}; o_t_clamp 0) — the P24 "
              "round trip EXP(LOG(x)) is the identity only for x <= 2; see "
              "v2/experiment/rtl/attention/review_R2.md")
        if strict_r2:
            raise AssertionError(f"R2 o-carry saturation bar failed: {bar_fail}")
    else:
        print("  R2 o-carry bar (o_t_clamp, oq_sat <= 1 % on random / narrow16): PASS")

    # 10) drift: blocked (Bkv = 16) vs single block (Bkv = n_seq), reported
    print("  drift of O8, Bkv = 16 vs Bkv = n_seq (equal fraction / mean |d| / max |d|):")
    for ns in (64, 256, 1024):
        for name in ("random", "narrow16"):
            q8, k8, v8 = classes(ns, 64, 16, rng)[name]
            dr = drift_report(q8, k8, v8, 16)
            print(f"    {name:9s} n_seq={ns:5d} d=64: equal {dr['equal_frac']:.3f}  "
                  f"mean|d| {dr['mean_abs']:.2f}  max|d| {dr['max_abs']}")


if __name__ == "__main__":
    _self_test(strict_r2="--strict-r2" in sys.argv[1:])
