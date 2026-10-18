"""space.py — the attention DSE's design space (docs/experiment.md §3, App. A), defined once: the six workload shapes
(name, n tokens, d_model, d per head) and the twelve crossbar configs (R, C). The row builder, the VTR driver, the
breakdown, the head-parameter gate, the architecture page and the simulator tests import it from here."""

SHAPES = [("ViT-B/16", 256, 768, 64), ("DeiT-T", 256, 192, 64), ("BERT-base", 128, 768, 64),
          ("BERT-Tiny", 128, 128, 64), ("JetFormer-150P", 32, 64, 32), ("JetFormer-JetClass", 128, 128, 16)]
CONFIGS = [(r, c) for r in (128, 256, 512, 1024) for c in (64, 128, 256)]
