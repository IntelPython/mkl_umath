"""npbench wrapper: GESUMMV (scalar, vector and matrix multiplication) — mkl_umath ops: matmul.

Preset sizes from npbench bench_info/gesummv.json:
  M: N=4_000
  L: N=14_000
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/gesummv/gesummv.py
def _initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % N) / N, (N, N), dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, (N,), dtype=datatype)
    return alpha, beta, A, B, x


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/gesummv/gesummv_numpy.py
def _kernel(alpha, beta, A, B, x):
    return alpha * A @ x + beta * B @ x


_PRESETS = {
    "M": {"N": 4_000},
    "L": {"N": 14_000},
}


class BenchGesummv:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        self.alpha, self.beta, self.A, self.B, self.x = cache[preset]

    def time_gesummv(self, cache, preset):
        _kernel(self.alpha, self.beta, self.A, self.B, self.x)
