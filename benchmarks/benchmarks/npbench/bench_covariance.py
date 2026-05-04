"""npbench wrapper: Covariance — mkl_umath ops: mean.

Preset sizes from npbench bench_info/covariance.json:
  S: M=500,  N=600
  L: M=3200, N=4000

The kernel mutates ``data`` in-place (data -= mean), so setup() copies
from the cache before each timing round.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/covariance/covariance.py
def _initialize(M, N, datatype=np.float64):
    float_n = datatype(N)
    data = np.fromfunction(lambda i, j: (i * j) / M, (N, M), dtype=datatype)
    return float_n, data


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/covariance/covariance_numpy.py
def _kernel(M, float_n, data):
    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)
    return cov


_PRESETS = {
    "S": {"M": 500,  "N": 600},
    "L": {"M": 3200, "N": 4000},
}


class BenchCovariance:
    params = (["S", "L"],)
    param_names = ["preset"]

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        float_n, data = cache[preset]
        self.M = _PRESETS[preset]["M"]
        self.float_n = float_n
        self.data = data.copy()  # kernel mutates data in-place

    def time_covariance(self, cache, preset):
        _kernel(self.M, self.float_n, self.data)
