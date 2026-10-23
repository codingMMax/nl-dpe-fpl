# The mapping optimized per build, and what it does to the attention ranking

For every build (shape × config) every mapping of the grid P_proj (1, 2, 4, 8, 16) × N_E ('d', '2d', '4d') × N_EP ('2d', '4d')
× N_EXP (1, 2, 4) × B_kv (8, 16, 32) (N_OC by the floor rule, X_S 16) is built in the simulator; the ones a module
guard or a port contract rejects are dropped. Per build and per objective the mapping that minimizes it is kept.
Cycles and energy are the variant's own; area = the fixed build's CLBs and memory blocks (VTR) + the variant's
crossbar tiles; latency = cycles / the fixed build's Fmax. So the soft logic and Fmax of a variant are assumed to be
the fixed mapping's: the crossbar part of the area is exact, the rest is not re-synthesized
(`dse/mapping_sensitivity.py --optimize`; the reading in `docs/mapping_sensitivity.md` §6).

## A. The ADP ranking of the attention family: fixed mapping against the ADP-optimal mapping per build

G_ADP = the geometric mean over the heads of area × latency ÷ the worst config's (rank.py); lower is better.

| config | fixed: G_ADP (rank) | ADP-optimal mapping: G_ADP (rank) | Δ rank | ADP ÷ fixed (geomean over the heads) | the mapping the heads pick |
|---|---:|---:|---:|---:|---|
| 128×64 | 0.201 (1) | 0.314 (1) | +0 | 0.65 | P_PROJ 4/8, N_E 2d, N_EP 4d, N_EXP 4, BKV 16/8 |
| 256×64 | 0.230 (2) | 0.326 (2) | +0 | 0.58 | P_PROJ 4/8, N_E 2d, N_EP 4d, N_EXP 4, BKV 16/8 |
| 512×64 | 0.275 (3) | 0.364 (3) | +0 | 0.55 | P_PROJ 4/8, N_E 2d, N_EP 2d/4d, N_EXP 4, BKV 16/8 |
| 128×128 | 0.306 (4) | 0.465 (5) | +1 | 0.63 | P_PROJ 4/8, N_E 2d, N_EP 2d/4d, N_EXP 4, BKV 16 |
| 1024×64 | 0.318 (5) | 0.401 (4) | -1 | 0.52 | P_PROJ 4/8, N_E 2d, N_EP 2d/4d, N_EXP 4, BKV 16/8 |
| 256×128 | 0.332 (6) | 0.468 (6) | +0 | 0.58 | P_PROJ 8, N_E 2d, N_EP 2d/4d, N_EXP 4, BKV 16 |
| 512×128 | 0.415 (7) | 0.527 (7) | +0 | 0.52 | P_PROJ 16/8, N_E 2d, N_EP 2d, N_EXP 4, BKV 16/8 |
| 1024×128 | 0.475 (8) | 0.561 (8) | +0 | 0.49 | P_PROJ 16/8, N_E 2d, N_EP 2d, N_EXP 4, BKV 16/8 |
| 128×256 | 0.483 (9) | 0.669 (9) | +0 | 0.57 | P_PROJ 4/8, N_E 2d, N_EP 2d, N_EXP 4, BKV 16/8 |
| 256×256 | 0.673 (10) | 0.795 (10) | +0 | 0.49 | P_PROJ 16/8, N_E 1d/2d, N_EP 2d, N_EXP 4, BKV 16/32 |
| 512×256 | 0.814 (11) | 0.870 (11) | +0 | 0.44 | P_PROJ 16/8, N_E 1d/2d, N_EP 2d, N_EXP 4, BKV 16/32 |
| 1024×256 | 1.000 (12) | 1.000 (12) | +0 | 0.41 | P_PROJ 16/8, N_E 1d/2d, N_EP 2d, N_EXP 4, BKV 16/32 |

## B. Latency, area and EDAP

| config | fixed: G_D (front) | latency-optimal: G_D (front) | cycles ÷ fixed | fixed: G_A | area-optimal: G_A | area ÷ fixed | fixed: G_EDAP (rank) | EDAP-optimal: G_EDAP (rank) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128×64 | 0.440 (front) | 0.719 (front) | 0.60 | 0.449 | 0.478 | 0.98 | 0.137 (1) | 0.216 (1) |
| 256×64 | 0.497 | 0.729 | 0.54 | 0.457 | 0.484 | 0.97 | 0.164 (2) | 0.234 (2) |
| 512×64 | 0.572 | 0.778 | 0.50 | 0.473 | 0.497 | 0.97 | 0.214 (3) | 0.288 (3) |
| 128×128 | 0.507 | 0.773 | 0.56 | 0.593 | 0.622 | 0.96 | 0.215 (4) | 0.328 (4) |
| 1024×64 | 0.630 | 0.807 | 0.47 | 0.497 | 0.517 | 0.95 | 0.291 (6) | 0.376 (6) |
| 256×128 | 0.545 | 0.773 | 0.53 | 0.601 | 0.628 | 0.96 | 0.243 (5) | 0.344 (5) |
| 512×128 | 0.655 | 0.832 | 0.47 | 0.624 | 0.646 | 0.95 | 0.334 (7) | 0.426 (7) |
| 1024×128 | 0.694 | 0.815 | 0.43 | 0.673 | 0.687 | 0.94 | 0.448 (8) | 0.535 (8) |
| 128×256 | 0.702 | 0.938 | 0.49 | 0.677 | 0.691 | 0.94 | 0.525 (10) | 0.749 (11) |
| 256×256 | 0.760 | 0.875 | 0.43 | 0.872 | 0.896 | 0.94 | 0.521 (9) | 0.615 (9) |
| 512×256 | 0.870 | 0.903 | 0.38 | 0.920 | 0.935 | 0.93 | 0.691 (11) | 0.738 (10) |
| 1024×256 | 0.984 | 0.951 | 0.36 | 1.000 | 1.000 | 0.92 | 1.000 (12) | 1.000 (12) |

## C. Per build: the ADP-optimal mapping

| shape | config | ADP-optimal mapping | cycles ÷ fixed | area ÷ fixed | ADP ÷ fixed | energy ÷ fixed |
|---|---|---|---:|---:|---:|---:|
| ViT-B/16 | 128×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.64 | 1.06 | 0.69 | 1.000 |
| ViT-B/16 | 128×128 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.71 | 1.11 | 0.79 | 1.000 |
| ViT-B/16 | 128×256 | P4 N_E=2d N_EP=2d N_EXP=4 BKV=8 | 0.73 | 1.07 | 0.79 | 1.048 |
| ViT-B/16 | 256×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.52 | 1.06 | 0.55 | 1.000 |
| ViT-B/16 | 256×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.66 | 1.05 | 0.69 | 1.000 |
| ViT-B/16 | 256×256 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.53 | 1.07 | 0.56 | 1.000 |
| ViT-B/16 | 512×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.43 | 1.08 | 0.46 | 1.000 |
| ViT-B/16 | 512×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.46 | 1.05 | 0.48 | 1.000 |
| ViT-B/16 | 512×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.37 | 1.13 | 0.42 | 1.000 |
| ViT-B/16 | 1024×64 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=8 | 0.38 | 1.04 | 0.40 | 1.050 |
| ViT-B/16 | 1024×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.36 | 1.04 | 0.37 | 1.000 |
| ViT-B/16 | 1024×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.30 | 1.11 | 0.33 | 1.000 |
| DeiT-T | 128×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.64 | 1.05 | 0.68 | 1.000 |
| DeiT-T | 128×128 | P8 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.69 | 1.11 | 0.77 | 1.000 |
| DeiT-T | 128×256 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=8 | 0.70 | 1.07 | 0.76 | 1.050 |
| DeiT-T | 256×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.57 | 1.06 | 0.60 | 1.000 |
| DeiT-T | 256×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.70 | 1.03 | 0.72 | 1.000 |
| DeiT-T | 256×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.52 | 1.07 | 0.56 | 1.000 |
| DeiT-T | 512×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.57 | 1.07 | 0.61 | 1.000 |
| DeiT-T | 512×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.70 | 1.03 | 0.72 | 1.000 |
| DeiT-T | 512×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.52 | 1.09 | 0.57 | 1.000 |
| DeiT-T | 1024×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.57 | 1.10 | 0.63 | 1.000 |
| DeiT-T | 1024×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.70 | 1.04 | 0.73 | 1.000 |
| DeiT-T | 1024×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.52 | 1.11 | 0.58 | 1.000 |
| BERT-base | 128×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.63 | 1.06 | 0.67 | 1.000 |
| BERT-base | 128×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.56 | 1.08 | 0.60 | 1.000 |
| BERT-base | 128×256 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=8 | 0.47 | 1.15 | 0.54 | 1.046 |
| BERT-base | 256×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.47 | 1.06 | 0.50 | 1.000 |
| BERT-base | 256×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.43 | 1.05 | 0.46 | 1.000 |
| BERT-base | 256×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.39 | 1.15 | 0.45 | 1.000 |
| BERT-base | 512×64 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=8 | 0.36 | 1.04 | 0.37 | 1.047 |
| BERT-base | 512×128 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=8 | 0.31 | 1.11 | 0.34 | 1.046 |
| BERT-base | 512×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.28 | 1.14 | 0.32 | 1.000 |
| BERT-base | 1024×64 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=8 | 0.28 | 1.04 | 0.29 | 1.049 |
| BERT-base | 1024×128 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=8 | 0.24 | 1.10 | 0.27 | 1.048 |
| BERT-base | 1024×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.22 | 1.11 | 0.25 | 1.000 |
| BERT-Tiny | 128×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.63 | 1.05 | 0.66 | 1.000 |
| BERT-Tiny | 128×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.55 | 1.02 | 0.57 | 1.000 |
| BERT-Tiny | 128×256 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=8 | 0.47 | 1.05 | 0.50 | 1.050 |
| BERT-Tiny | 256×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.63 | 1.06 | 0.67 | 1.000 |
| BERT-Tiny | 256×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.55 | 1.03 | 0.57 | 1.000 |
| BERT-Tiny | 256×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.42 | 1.08 | 0.45 | 1.000 |
| BERT-Tiny | 512×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.63 | 1.07 | 0.68 | 1.000 |
| BERT-Tiny | 512×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.55 | 1.03 | 0.57 | 1.000 |
| BERT-Tiny | 512×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.42 | 1.09 | 0.46 | 1.000 |
| BERT-Tiny | 1024×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.63 | 1.10 | 0.69 | 1.000 |
| BERT-Tiny | 1024×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.55 | 1.04 | 0.58 | 1.000 |
| BERT-Tiny | 1024×256 | P16 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.42 | 1.11 | 0.46 | 1.000 |
| JetFormer-150P | 128×64 | P8 N_E=2d N_EP=4d N_EXP=4 BKV=8 | 0.50 | 1.08 | 0.54 | 1.044 |
| JetFormer-150P | 128×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.50 | 1.05 | 0.52 | 1.000 |
| JetFormer-150P | 128×256 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.39 | 1.08 | 0.42 | 1.000 |
| JetFormer-150P | 256×64 | P8 N_E=2d N_EP=4d N_EXP=4 BKV=8 | 0.50 | 1.10 | 0.55 | 1.044 |
| JetFormer-150P | 256×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.50 | 1.05 | 0.52 | 1.000 |
| JetFormer-150P | 256×256 | P8 N_E=d N_EP=2d N_EXP=4 BKV=32 | 0.42 | 1.01 | 0.42 | 0.980 |
| JetFormer-150P | 512×64 | P8 N_E=2d N_EP=4d N_EXP=4 BKV=8 | 0.50 | 1.13 | 0.56 | 1.045 |
| JetFormer-150P | 512×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.50 | 1.06 | 0.53 | 1.000 |
| JetFormer-150P | 512×256 | P8 N_E=d N_EP=2d N_EXP=4 BKV=32 | 0.42 | 1.02 | 0.42 | 0.980 |
| JetFormer-150P | 1024×64 | P8 N_E=2d N_EP=4d N_EXP=4 BKV=8 | 0.50 | 1.17 | 0.58 | 1.046 |
| JetFormer-150P | 1024×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.50 | 1.08 | 0.54 | 1.000 |
| JetFormer-150P | 1024×256 | P8 N_E=d N_EP=2d N_EXP=4 BKV=32 | 0.42 | 1.02 | 0.43 | 0.980 |
| JetFormer-JetClass | 128×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.61 | 1.05 | 0.65 | 1.000 |
| JetFormer-JetClass | 128×128 | P8 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.51 | 1.12 | 0.57 | 1.000 |
| JetFormer-JetClass | 128×256 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.47 | 1.12 | 0.52 | 1.000 |
| JetFormer-JetClass | 256×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.61 | 1.06 | 0.65 | 1.000 |
| JetFormer-JetClass | 256×128 | P8 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.51 | 1.13 | 0.58 | 1.000 |
| JetFormer-JetClass | 256×256 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.45 | 1.11 | 0.50 | 1.000 |
| JetFormer-JetClass | 512×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.61 | 1.08 | 0.66 | 1.000 |
| JetFormer-JetClass | 512×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.54 | 1.09 | 0.58 | 1.000 |
| JetFormer-JetClass | 512×256 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.45 | 1.13 | 0.51 | 1.000 |
| JetFormer-JetClass | 1024×64 | P4 N_E=2d N_EP=4d N_EXP=4 BKV=16 | 0.61 | 1.11 | 0.68 | 1.000 |
| JetFormer-JetClass | 1024×128 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.54 | 1.12 | 0.60 | 1.000 |
| JetFormer-JetClass | 1024×256 | P8 N_E=2d N_EP=2d N_EXP=4 BKV=16 | 0.45 | 1.16 | 0.52 | 1.000 |
