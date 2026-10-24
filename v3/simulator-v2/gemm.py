from __future__ import annotations

import numpy as np
import math
from core import Core, CorePass


class GEMM:
    def __init__(self, core: Core, M, K, N):
        self.core = core
        self.M = M
        self.K = K
        self.N = N
    
    def _tile_to_core(self):
        