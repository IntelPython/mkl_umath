"""npbench wrapper: GEMVER (vector multiplication and matrix addition) — mkl_umath ops: outer.

Preset sizes from npbench bench_info/gemver.json:
  M: N=3_000
  L: N=10_000

The kernel mutates A, x, and w in-place, so setup() copies those from cache.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/gemver/gemver.py
def _initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A  = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)
    u1 = np.fromfunction(lambda i: i, (N,), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, (N,), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, (N,), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, (N,), dtype=datatype)
    w  = np.zeros((N,), dtype=datatype)
    x  = np.zeros((N,), dtype=datatype)
    y  = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, (N,), dtype=datatype)
    z  = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, (N,), dtype=datatype)
    return alpha, beta, A, u1, v1, u2, v2, w, x, y, z


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/gemver/gemver_numpy.py
def _kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x


_PRESETS = {
    "M": {"N": 3_000},
    "L": {"N": 10_000},
}


class BenchGemver:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        alpha, beta, A, u1, v1, u2, v2, w, x, y, z = cache[preset]
        self.alpha = alpha
        self.beta  = beta
        self.A  = A.copy()   # mutated: A += outer(u1,v1) + outer(u2,v2)
        self.u1 = u1
        self.v1 = v1
        self.u2 = u2
        self.v2 = v2
        self.w  = w.copy()   # mutated: w += alpha * A @ x
        self.x  = x.copy()   # mutated: x += beta * y @ A + z
        self.y  = y
        self.z  = z

    def time_gemver(self, cache, preset):
        _kernel(
            self.alpha, self.beta,
            self.A, self.u1, self.v1, self.u2, self.v2,
            self.w, self.x, self.y, self.z,
        )
