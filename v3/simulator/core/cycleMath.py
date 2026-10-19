"""cycleMath.py — the small integer arithmetic every model uses, defined once."""
from __future__ import annotations


def ceilDiv(a: int, b: int) -> int:
    """ceil(a / b) for integers."""
    return -(-int(a) // int(b))


def clog2(x: int) -> int:
    """ceil(log2 x) for x >= 1, as the RTL's clog2 functions compute it (0 for x <= 1)."""
    return max(0, (int(x) - 1).bit_length())


def isPowerOfTwo(x) -> bool:
    x = int(x)
    return x > 0 and (x & (x - 1)) == 0


def readyAt(*times) -> int:
    """A decision: the first cycle at which every condition holds, i.e. the latest of the given times. None is no
    condition (for example a credit not needed yet); with no condition at all, cycle 0."""
    known = [t for t in times if t is not None]                  # the chosen object is kept (a traced Time)
    return max(known) if known else 0
