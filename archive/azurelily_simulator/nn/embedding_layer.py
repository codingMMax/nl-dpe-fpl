"""Embedding layer for the analytical scheduler path.

BERT-style embedding = token_embed + position_embed + segment_embed.
All three are table lookups (memory reads) followed by element-wise add (CLB).
"""

import numpy as np


class Embedding_Layer:
    """Thin parameter container dispatched by Scheduler._run_embedding()."""

    def __init__(
        self,
        vocab_size: int,         # token embedding table rows (30522 for BERT)
        d_model: int,            # embedding dimension
        max_seq_len: int,        # position embedding table rows (512 for BERT)
        seq_len: int,            # actual sequence length
        name: str,
        debug: bool,
        energy_stats,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.seq_len = seq_len
        self.name = name
        self.debug = debug
        self.energy_stats = energy_stats
        self.type = "embedding"

        self.next_layer = None
        self.done = False
        self.event_queue = []

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def add_event(self, event_type, event_time):
        event = {"EVENT_TYPE": event_type, "EVENT_TIME": event_time}
        self.event_queue.append(event)
        self.event_queue = sorted(self.event_queue, key=lambda d: d["EVENT_TIME"])
