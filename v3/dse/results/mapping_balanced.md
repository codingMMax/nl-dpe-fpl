# The balanced mapping: P_proj* per build

Every other knob keeps the rate rule's value (N_E = N_EP = 2d, N_EXP = 2, N_OC by the floor rule, B_kv 16, X_S 16),
which is the analytic optimum of the core (`docs/mapping_sensitivity.md` §6); P_proj* is the smallest power of two
with the projection stream at least as fast as the core consumes keys, P_proj >= 5 T_p / S, capped by n and by the QK
staging group I / B_kv. Per head in the order ViT-B/16, DeiT-T, BERT-base, BERT-Tiny, JetFormer-150P, JetFormer-JetClass. Cycles and crossbars are the
simulator's; Fmax and the soft logic of the P_proj* builds need VTR (`dse/mapping_sensitivity.py --balanced`).

| config | P_proj* per head | cycles fixed (geomean) | cycles balanced | ÷ fixed | ÷ n²/5 | crossbar area ÷ fixed | cycles ÷ 128×64: fixed → balanced |
|---|---|---:|---:|---:|---:|---:|---:|
| 128×64 | 4 / 4 / 4 / 4 / 4 / 4 | 8,410 | 6,533 | 0.78 | 1.99 | 1.15 | 1.00 → 1.00 |
| 128×128 | 2 / 2 / 2 / 2 / 8 / 2 | 9,234 | 6,986 | 0.76 | 2.13 | 1.09 | 1.10 → 1.07 |
| 128×256 | 4 / 4 / 4 / 4 / 8 / 4 | 12,616 | 7,845 | 0.62 | 2.39 | 1.18 | 1.50 → 1.20 |
| 256×64 | 4 / 4 / 4 / 4 / 4 / 4 | 9,492 | 6,603 | 0.70 | 2.02 | 1.12 | 1.13 → 1.01 |
| 256×128 | 4 / 2 / 4 / 2 / 8 / 2 | 10,062 | 6,977 | 0.69 | 2.13 | 1.11 | 1.20 → 1.07 |
| 256×256 | 2 / 2 / 4 / 4 / 16 / 4 | 12,747 | 7,419 | 0.58 | 2.26 | 1.17 | 1.52 → 1.14 |
| 512×64 | 4 / 4 / 4 / 4 / 4 / 4 | 10,871 | 6,827 | 0.63 | 2.08 | 1.11 | 1.29 → 1.05 |
| 512×128 | 8 / 2 / 8 / 2 / 8 / 2 | 11,548 | 6,944 | 0.60 | 2.12 | 1.14 | 1.37 → 1.06 |
| 512×256 | 4 / 2 / 8 / 4 / 16 / 4 | 14,464 | 7,363 | 0.51 | 2.25 | 1.19 | 1.72 → 1.13 |
| 1024×64 | 4 / 4 / 4 / 4 / 4 / 4 | 11,963 | 7,192 | 0.60 | 2.19 | 1.10 | 1.42 → 1.10 |
| 1024×128 | 8 / 2 / 8 / 2 / 8 / 2 | 12,727 | 7,124 | 0.56 | 2.17 | 1.11 | 1.51 → 1.09 |
| 1024×256 | 4 / 2 / 8 / 4 / 16 / 4 | 15,846 | 7,668 | 0.48 | 2.34 | 1.18 | 1.88 → 1.17 |
