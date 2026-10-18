"""space.py — the DSE's design space (docs/experiment.md §2–§3, App. A), defined once: the twelve crossbar configs
(R, C), the six attention shapes (name, n tokens, d_model, d per head) and the four FC shapes (M, K, N). The row
builders, the VTR driver, the breakdown, the head-parameter gate, the architecture page and the simulator tests import
it from here."""

CONFIGS = [(r, c) for r in (128, 256, 512, 1024) for c in (64, 128, 256)]
SHAPES = [("ViT-B/16", 256, 768, 64), ("DeiT-T", 256, 192, 64), ("BERT-base", 128, 768, 64),
          ("BERT-Tiny", 128, 128, 64), ("JetFormer-150P", 32, 64, 32), ("JetFormer-JetClass", 128, 128, 16)]
# FC: Round 1's GEMVs (experiment.md §3.1), batch M = 1; one row of X[K] through W[K, N]
FC_SHAPES = [(1, 512, 128), (1, 2048, 256), (1, 256, 512), (1, 512, 512)]
FC_PASS_EXTENT = "used"             # P29: a pass loads min(R, K) rows and drains min(C, N) columns
