#!/usr/bin/env python3
"""softmax_regular_vs_online.py — regular vs online softmax sweep + log.

Dedicated comparison sweep over arbitrary configurations: softmax size (S),
crossbar geometry (R, C), online block size (Bkv), EXP/LOG crossbar counts
(n_exp, n_log), factor-bank width (n_fac), and score classes.  For every
configuration both machines run on identical scores and the values are
gate-checked against the certified oracles before the row is recorded:

  gate 1: regular  == softmax_ref (ACAM surrogate, bit-exact)
  gate 2: online   == softmax_online_model (ACAM contract, bit-exact)

Cycle convention (no-drain, S3/O6): the output stage is untimed — regular
`e2e = load + compute` (stream before compute), online `e2e = compute`
(block stream overlapped with the run).  n_log > 1 only splits when
S > I = min(R, C); n_fac defaults to n_exp (sim sweep knob).

Run:
  python3 v2/smoke/softmax_regular_vs_online.py \
      [--Ss 128,256] [--RCs 256x256] [--Bkvs 16,32] [--nExps 1,2,4]
      [--nLogs 1,2] [--nFacs 1,4] [--classes small,full] [--seed 3]
      [--out softmax-regular-vs-online.log] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim" / "simulator" / "kernels"))
sys.path.insert(0, str(HERE.parent / "oracle"))
import softmax_ref as sref  # noqa: E402
import softmax_online_ref as oref  # noqa: E402
import nldpe_ref as nref  # noqa: E402
from softmax_sim import NldpeSoftmax  # noqa: E402
from softmax_online_sim import NldpeSoftmaxOnline  # noqa: E402

CLASSES = {"small": (-16, 16), "full": (-128, 127)}


def parse_rcs(spec: str) -> list:
    out = []
    for tok in spec.split(","):
        r, c = tok.lower().split("x")
        out.append((int(r), int(c)))
    return out


def ints(spec: str | None) -> list | None:
    if spec is None:
        return None
    return [int(t) for t in spec.split(",")]


def storage_bytes(S: int, Bkv: int) -> dict:
    """Score-side storage (bytes): 1 B/score, int8 max, 4 B partial."""
    B = S // Bkv
    return {
        "conv": S * S + 3 * S,
        "online": S * Bkv + B * S * 1 + B * S * 4,
    }


def _decompose(err: np.ndarray) -> tuple[int, int]:
    """(offset, relative) maxima; the row offset is arbitrary in log-domain."""
    off = np.median(err, axis=1, keepdims=True)
    return int(np.abs(off).max()), int(np.abs(err - off).max())


def _l1(out8: np.ndarray, p: np.ndarray) -> float:
    """Max per-row L1 distance between a reconstructed and the exact dist."""
    return float(np.abs(oref.reconstruct_p(out8) - p).sum(1).max())


def stream_words(n_bytes: int, buf: int) -> int:
    return -(-n_bytes // (buf // 8))


def run_point(S: int, R: int, C: int, Bkv: int, n_exp: int, n_log: int,
              n_fac: int | None, buf: int, scores: np.ndarray,
              seed_class: str) -> dict:
    assert S % Bkv == 0, f"Bkv={Bkv} must divide S={S}"
    assert 1 <= Bkv <= S
    conv = NldpeSoftmax(S=S, R=R, C=C, BUF=buf, n_exp=n_exp,
                        n_log=n_log).run(scores)
    onl = NldpeSoftmaxOnline(S=S, R=R, C=C, BUF=buf, Bkv=Bkv, n_exp=n_exp,
                             n_log=n_log, n_fac=n_fac).run(scores)
    conv_ref = sref.softmax_stage_values(scores.astype(np.int32))
    assert np.array_equal(conv.softmax_out, conv_ref["softmax_out"]), \
        f"regular != softmax_ref ({seed_class})"
    onl_model = oref.softmax_online_model(scores.astype(np.int32), Bkv)
    assert np.array_equal(onl.softmax_out, onl_model["out8"]), \
        f"online != model ({seed_class})"

    conv_load = len(nref.pack_act_stream(scores.reshape(-1)))
    B = S // Bkv
    onl_load = B * -(-(S * Bkv) // (buf // 8))  # block-major, padded per block
    st = storage_bytes(S, Bkv)
    exact = oref.softmax_online_exact(scores.astype(np.int32), Bkv)
    e = exact["out8_exact"].astype(np.int32)
    c_off, c_rel = _decompose(conv.softmax_out.astype(np.int32) - e)
    o_off, o_rel = _decompose(onl.softmax_out.astype(np.int32) - e)
    return dict(
        S=S, R=R, C=C, BUF=buf, Bkv=Bkv, blocks=B, n_exp=n_exp, n_log=n_log,
        n_fac=n_fac if n_fac is not None else n_exp,
        conv_load=conv_load, conv_comp=conv.compute_cycles,
        conv_e2e=conv_load + conv.compute_cycles,
        onl_load=onl_load, onl_comp=onl.compute_cycles,
        onl_e2e=onl.compute_cycles,
        d_e2e=onl.compute_cycles - (conv_load + conv.compute_cycles),
        onl_emit=onl.emit_start,
        conv_pexp=conv.passes_exp, onl_pexp=onl.passes_exp,
        conv_plog=conv.passes_log, onl_plog=onl.passes_log,
        conv_off=c_off, conv_rel=c_rel, onl_off=o_off, onl_rel=o_rel,
        conv_L1=_l1(conv.softmax_out, exact["p"]),
        onl_L1=_l1(onl.softmax_out, exact["p"]),
        conv_clamp=oref.clamped_fraction(conv.softmax_out),
        onl_clamp=oref.clamped_fraction(onl.softmax_out),
        conv_store=st["conv"], onl_store=st["online"])


def write_log(path: Path, rows: list, cmdline: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w") as f:
        f.write("# Softmax — regular vs online comparison log\n")
        f.write(f"# generated: {now}\n")
        f.write(f"# command:   {cmdline}\n")
        f.write("# convention: output stage untimed (no drain; S3/O6) — "
                "regular e2e = load + compute; online e2e = compute\n")
        f.write("#            (block stream overlapped with the run)\n")
        f.write("# gates: regular == softmax_ref; online == "
                "softmax_online_model (bit-exact, per row)\n")
        f.write("# RTL status: GATE-2 delta_impl = 0 (regular 60/60 corpus; "
                "online 64/64 corpus; sim == oracle == RTL)\n")
        f.write("# n_log > 1 splits only when S > I = min(R,C); n_fac "
                "defaults to n_exp\n")
        f.write(f"# port width BUF={rows[0]['BUF']} "
                f"(stream_bytes = BUF/8 = {rows[0]['BUF'] // 8})\n\n")

        f.write("## Values — deviation from the exact dense softmax\n\n")
        f.write("`offset` = arbitrary row constant; `rel` = within-row error; "
                "`L1` = max per-row L1 vs exact p; `clamp%` = outputs on the "
                "int8 rails.\n\n")
        f.write("| S | class | range | conv off | conv rel | conv L1 | conv clamp% "
                "| onl off | onl rel | onl L1 | onl clamp% |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        seen = set()
        for r in rows:
            key = (r["S"], r["cls"], r["Bkv"])
            if key in seen:
                continue
            seen.add(key)
            lo, hi = CLASSES[r["cls"]]
            f.write(f"| {r['S']} | {r['cls']} | [{lo},{hi}] | "
                    f"{r['conv_off']} | {r['conv_rel']} | "
                    f"{r['conv_L1']:.4f} | {r['conv_clamp']:.4f} | "
                    f"{r['onl_off']} | {r['onl_rel']} | {r['onl_L1']:.4f} | "
                    f"{r['onl_clamp']:.4f} |\n")

        f.write("\n## Cycles (compute / load, no-drain convention)\n\n")
        f.write("| S | R=C | Bkv | n_exp | n_log | n_fac | conv load | conv comp "
                "| conv e2e | onl comp(e2e) | d e2e | onl in-stream load |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        seen = set()
        for r in rows:
            key = (r["S"], r["R"], r["C"], r["Bkv"], r["n_exp"],
                   r["n_log"], r["n_fac"])
            if key in seen:
                continue
            seen.add(key)
            f.write(f"| {r['S']} | {r['R']}x{r['C']} | {r['Bkv']} | "
                    f"{r['n_exp']} | {r['n_log']} | {r['n_fac']} | "
                    f"{r['conv_load']} | {r['conv_comp']} | {r['conv_e2e']} | "
                    f"{r['onl_comp']} | {r['d_e2e']:+d} | {r['onl_load']} |\n")

        f.write("\n## Pass counts + score-side storage\n\n")
        f.write("| S | R=C | Bkv | blocks | conv passes e/l | onl passes e/l "
                "| conv store (B) | onl store (B) | ratio |\n")
        f.write("|---|---|---:|---:|---|---|---:|---:|---:|\n")
        seen = set()
        for r in rows:
            key = (r["S"], r["R"], r["C"], r["Bkv"])
            if key in seen:
                continue
            seen.add(key)
            f.write(f"| {r['S']} | {r['R']}x{r['C']} | {r['Bkv']} | "
                    f"{r['blocks']} | {r['conv_pexp']}/{r['conv_plog']} | "
                    f"{r['onl_pexp']}/{r['onl_plog']} | {r['conv_store']} | "
                    f"{r['onl_store']} | "
                    f"{r['conv_store'] / r['onl_store']:.2f}x |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ss", default="128,256")
    ap.add_argument("--RCs", default="256x256")
    ap.add_argument("--Bkvs", default=None, help="comma list; default 16,32,S")
    ap.add_argument("--nExps", default="1,2,4")
    ap.add_argument("--nLogs", default="1,2")
    ap.add_argument("--nFacs", default=None,
                    help="factor bank width sweep; default = n_exp")
    ap.add_argument("--buf", type=int, default=40)
    ap.add_argument("--classes", default="small,full")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("softmax-regular-vs-online.log"))
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    cmdline = " ".join(sys.argv[1:]) or "(defaults)"

    Ss = ints(args.Ss)
    RCs = parse_rcs(args.RCs)
    nExps = ints(args.nExps)
    nLogs = ints(args.nLogs)
    nFacs = ints(args.nFacs)
    classes = [t.strip() for t in args.classes.split(",") if t.strip()]
    rows = []
    for S in Ss:
        for (R, C) in RCs:
            bkvs = ints(args.Bkvs) or sorted({16, 32, S})
            for cname in classes:
                lo, hi = CLASSES[cname]
                rng = np.random.default_rng(
                    args.seed + S + (0 if cname == "small" else 1))
                scores = rng.integers(lo, hi + 1, size=(S, S), dtype=np.int8)
                for Bkv in bkvs:
                    for n_exp in nExps:
                        for n_log in nLogs:
                            for n_fac in (nFacs or [None]):
                                r = run_point(S, R, C, Bkv, n_exp, n_log,
                                              n_fac, args.buf, scores, cname)
                                r["cls"] = cname
                                rows.append(r)
                                print(f"  S={S} R={R} C={C} Bkv={Bkv} "
                                      f"nX={n_exp} nL={n_log} "
                                      f"nF={r['n_fac']} {cname}: "
                                      f"conv {r['conv_e2e']} onl "
                                      f"{r['onl_e2e']} ({r['d_e2e']:+d}) "
                                      "GATES OK", flush=True)
    write_log(args.out, rows, cmdline)
    print(f"wrote {args.out} ({len(rows)} configs)")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
