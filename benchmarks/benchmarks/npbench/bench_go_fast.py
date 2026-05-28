"""npbench wrapper: GoFast — mkl_umath ops: tanh.

Preset sizes from npbench bench_info/go_fast.json:
  M: N=6_000
  L: N=20_000

Note: the npbench ``go_fast`` kernel iterates diagonals in a Python loop
(go_fast_loop).  A vectorized variant (go_fast_vec) using np.tanh on the
full diagonal is included for direct MKL VM throughput measurement.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/go_fast/go_fast.py
def _initialize(N):
    from numpy.random import default_rng

    rng = default_rng(42)
    a = rng.random((N, N))
    return (a,)


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/go_fast/go_fast_numpy.py
def _go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


_PRESETS = {
    "M": {"N": 6_000},
    "L": {"N": 20_000},
}


class BenchGoFastLoop:
    """Original npbench kernel — Python loop calling np.tanh per element."""

    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        (self.a,) = cache[preset]
        np.tanh(self.a[0, 0])

    def time_go_fast_loop(self, cache, preset):
        _go_fast(self.a)


class BenchGoFastVec:
    """Vectorized variant — np.tanh on the full diagonal array at once."""

    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        (self.a,) = cache[preset]
        self.diag = np.copy(np.diag(self.a))
        np.tanh(self.diag)

    def time_go_fast_vec(self, cache, preset):
        np.tanh(self.diag)
