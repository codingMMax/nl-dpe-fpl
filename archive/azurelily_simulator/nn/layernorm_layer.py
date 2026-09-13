"""LayerNorm layer for the analytical scheduler path.

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
All operations map to CLB fabric:
  - mean: reduction tree (sum) + divide
  - variance: element-wise subtract + square + reduction tree
  - normalize: element-wise subtract, multiply (rsqrt), scale (gamma), shift (beta)
  - rsqrt(var+eps): CLB LUT (same cost model as inverse in softmax norm)
"""

import numpy as np


class LayerNorm_Layer:
    """Thin parameter container dispatched by Scheduler._run_layernorm()."""

    def __init__(
        self,
        normalized_shape: int,   # dimension to normalize over (d_model)
        seq_len: int,            # number of tokens (each normalized independently)
        name: str,
        debug: bool,
        energy_stats,
    ):
        self.normalized_shape = normalized_shape
        self.seq_len = seq_len
        self.name = name
        self.debug = debug
        self.energy_stats = energy_stats
        self.type = "layernorm"

        self.next_layer = None
        self.done = False
        self.event_queue = []

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def add_event(self, event_type, event_time):
        event = {"EVENT_TYPE": event_type, "EVENT_TIME": event_time}
        self.event_queue.append(event)
        self.event_queue = sorted(self.event_queue, key=lambda d: d["EVENT_TIME"])
