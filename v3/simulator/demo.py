from simulator.core.engine import run
from simulator.platform import area, prices
from simulator.platform.platform import Platform
from simulator.models.fc import fc, result
from simulator.models.attentionHead import attentionHead, headParams
import numpy as np
import time

M, K, N = 512, 512, 128
R,C  = 256, 128
random_gen = np.random(time.time())
X = random_gen.rand((M, K))
W = random_gen.rand((K, N))
nl_dpe = Platform("nl_dpe", rows=R, cols=C)

fc_workload = fc(K, N, R, C, M, platform=nl_dpe, X=X, W=W, withValues=True)
report = run(fc_workload)

print(f"{fc_workload.name} on crossbar ({R, C}): take {report.cycles}, {report.outputs}")