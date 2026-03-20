#!/usr/bin/env python3
"""
workload_mapping.py

Layer-wise DPE tiling and ACAM mode comparison: NL-DPE vs Azure-Lily.

ACAM mode rules (NL-DPE only):
  - ACT mode : V == 1 AND H == 1  (single tile → output is final, ACAM can activate)
  - ADC mode : V > 1  OR  H > 1   (multi-tile → output is partial sum, ACAM bypassed)

Azure-Lily always uses ADC-only peripheral regardless of tiling.

Usage:
    python workload_mapping.py
    python workload_mapping.py --workload ResNet
    python workload_mapping.py --nl_rows 128 --nl_cols 128   # Q1 block sizing sweep
    python workload_mapping.py --al_rows 256 --al_cols 256   # hypothetical Azure-Lily
"""

import argparse
import math

# ---------------------------------------------------------------------------
# Workload definitions: (layer_name, K, N)
#   K = kernel_h * kernel_w * in_channels  (rows consumed per tile)
#   N = out_channels                        (cols consumed per tile)
# Values verified against RTL comments in nl_dpe/vgg11_1_channel.v and
# experiments.md tiling tables.
# ---------------------------------------------------------------------------
WORKLOADS = {
    "LeNet": [
        ("conv1",  25,    6),
        ("conv2",  150,   16),
        ("full1",  400,   120),
        ("full2",  120,   84),
        ("full3",  84,    10),
    ],
    "ResNet": [
        ("conv1",  9,     56),
        ("conv2",  504,   112),
        ("conv3",  1008,  112),
        ("conv4",  1008,  112),
        ("conv5",  1008,  224),
        ("conv6",  2016,  224),
        ("conv7",  2016,  224),
        ("conv8",  2016,  224),
        ("conv9",  224,   10),
    ],
    "VGG11": [
        ("conv1",  9,     64),    # 1→64,   K=3×3×1
        ("conv2",  576,   128),   # 64→128, K=3×3×64
        ("conv3",  1152,  256),   # 128→256, K=3×3×128
        ("conv4",  2304,  256),   # 256→256, K=3×3×256
        ("conv5",  2304,  512),   # 256→512, K=3×3×256
        ("conv6",  4608,  512),   # 512→512, K=3×3×512
        ("conv7",  4608,  512),
        ("conv8",  4608,  512),
        ("conv9",  512,   10),    # FC/avg-pool head
    ],
}


def tile(K, N, rows, cols):
    V = math.ceil(K / rows)
    H = math.ceil(N / cols)
    return V, H


def acam_mode(V, H):
    """NL-DPE: ACT only when single-tile. Azure-Lily: always ADC."""
    return "ACT" if (V == 1 and H == 1) else "ADC"


def map_layers(layers, rows, cols):
    result = []
    for name, K, N in layers:
        V, H = tile(K, N, rows, cols)
        dpes = V * H
        mode = acam_mode(V, H)
        result.append((name, K, N, V, H, dpes, mode))
    return result


def opportunity_ratio(mapped):
    act = sum(1 for r in mapped if r[6] == "ACT")
    return act, len(mapped)


def fmt_nl_mode(mode):
    return "[ACT]" if mode == "ACT" else " ADC "


def print_workload(name, layers, nl_rows, nl_cols, al_rows, al_cols):
    nl = map_layers(layers, nl_rows, nl_cols)
    al = map_layers(layers, al_rows, al_cols)

    nl_total = sum(r[5] for r in nl)
    al_total = sum(r[5] for r in al)
    nl_act, nl_n = opportunity_ratio(nl)

    nl_label = f"NL-DPE ({nl_rows}×{nl_cols})"
    al_label = f"Azure-Lily ({al_rows}×{al_cols})"

    W = 76
    print(f"\n{'='*W}")
    print(f"  Workload: {name}")
    print(f"{'='*W}")
    hdr_nl = f"{'── ' + nl_label + ' ──':^26}"
    hdr_al = f"{'── ' + al_label + ' ──':^22}"
    print(f"  {'Layer':<8}  {'K':>6}  {'N':>5}   {hdr_nl}   {hdr_al}")
    print(f"  {'':8}  {'':>6}  {'':>5}   {'V×H':>5}  {'DPEs':>4}  {'ACAM':>5}   {'V×H':>5}  {'DPEs':>4}")
    sep = f"  {'-'*8}  {'-'*6}  {'-'*5}   {'-'*5}  {'-'*4}  {'-'*5}   {'-'*5}  {'-'*4}"
    print(sep)

    for (nm, K, N, nv, nh, nd, nm_mode), (_, _, _, av, ah, ad, _) in zip(nl, al):
        print(f"  {nm:<8}  {K:>6}  {N:>5}   {nv}×{nh:<3}  {nd:>4}  {fmt_nl_mode(nm_mode):>5}   "
              f"{av}×{ah:<3}  {ad:>4}")

    print(sep)
    print(f"  {'TOTAL':<8}  {'':>6}  {'':>5}   {'':>5}  {nl_total:>4}  {'':>5}   "
          f"{'':>5}  {al_total:>4}")

    nl_pct = nl_act / nl_n * 100
    print()
    print(f"  NL-DPE ACAM activation opportunity: {nl_act}/{nl_n} layers = {nl_pct:.0f}%")
    print(f"  (Azure-Lily: ADC-only peripheral, no activation mode)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise DPE tiling and ACAM mode: NL-DPE vs Azure-Lily"
    )
    parser.add_argument("--nl_rows", type=int, default=256,
                        help="NL-DPE crossbar rows (default: 256)")
    parser.add_argument("--nl_cols", type=int, default=256,
                        help="NL-DPE crossbar cols (default: 256)")
    parser.add_argument("--al_rows", type=int, default=512,
                        help="Azure-Lily crossbar rows (default: 512)")
    parser.add_argument("--al_cols", type=int, default=128,
                        help="Azure-Lily crossbar cols (default: 128)")
    parser.add_argument("--workload", type=str, default="all",
                        choices=["all", "LeNet", "ResNet", "VGG11"],
                        help="Which workload to show (default: all)")
    args = parser.parse_args()

    to_run = WORKLOADS if args.workload == "all" else {args.workload: WORKLOADS[args.workload]}

    print(f"\n  NL-DPE crossbar : {args.nl_rows} rows × {args.nl_cols} cols")
    print(f"  Azure-Lily      : {args.al_rows} rows × {args.al_cols} cols")
    print(f"  NL-DPE ACAM ACT : V == 1 AND H == 1  (single-tile → output is final)")

    for name, layers in to_run.items():
        print_workload(name, layers, args.nl_rows, args.nl_cols, args.al_rows, args.al_cols)


if __name__ == "__main__":
    main()
