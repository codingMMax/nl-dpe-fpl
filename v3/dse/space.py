"""space.py — the DSE's design space (docs/experiment.md §2–§3, App. A), defined once: the nine crossbar configs
(R, C), the six attention shapes (name, n tokens, d_model, d per head) and the four FC shapes (M, K, N). The row
builders, the VTR driver, the breakdown, the head-parameter gate, the architecture page and the simulator tests import
it from here.

The configs (user): C <= R, R from 64 to 512 (the row-parallel tile rule, experiment.md §5.1). 64x64 joined on
2026-10-23 (it is also the RTL's functional-validation size); 2026-10-22 to 2026-10-23 the eight with R >= 128; the
twelve with R = 1024 and 128x256 before."""

CONFIGS = [(r, c) for r in (64, 128, 256, 512) for c in (64, 128, 256) if c <= r]
SHAPES = [("ViT-B/16", 256, 768, 64), ("DeiT-T", 256, 192, 64), ("BERT-base", 128, 768, 64),
          ("BERT-Tiny", 128, 128, 64), ("JetFormer-150P", 32, 64, 32), ("JetFormer-JetClass", 128, 128, 16)]
# FC: Round 1's GEMVs (experiment.md §3.1), batch M = 1; one row of X[K] through W[K, N]
FC_SHAPES = [(1, 512, 128), (1, 2048, 256), (1, 256, 512), (1, 512, 512)]
FC_PASS_EXTENT = "used"             # P29: a pass loads min(R, K) rows and drains min(C, N) columns
# the case-study pair of the stage analysis (user 2026-10-19): the smallest and the largest attention shape
CASE_STUDY = ("JetFormer-150P", "ViT-B/16")