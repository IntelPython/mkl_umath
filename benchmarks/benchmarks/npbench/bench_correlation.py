"""npbench wrapper: Correlation — mkl_umath ops: sqrt, std, mean.

Preset sizes from npbench bench_info/correlation.json:
  M: M=1400, N=1800
  L: M=3200, N=4000

The kernel mutates ``data`` in-place (data -= mean; data /= ...), so
setup() copies from the cache before each timing round.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/correlation/correlation.py
def _initialize(M, N, datatype=np.float64):
    float_n = datatype(N)
    data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=datatype)
    return float_n, data


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/correlation/correlation_numpy.py
def _kernel(M, float_n, data):
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]
    return corr


_PRESETS = {
    "M": {"M": 1400, "N": 1800},
    "L": {"M": 3200, "N": 4000},
}


class BenchCorrelation:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        float_n, data = cache[preset]
        self.M = _PRESETS[preset]["M"]
        self.float_n = float_n
        self.data = data.copy()  # kernel mutates data in-place

    def time_correlation(self, cache, preset):
        _kernel(self.M, self.float_n, self.data)
