"""npbench wrapper: Cholesky decomposition v2 — mkl_umath ops: linalg.cholesky.

Preset sizes from npbench bench_info/cholesky2.json:
  M: N=2200
  L: N=8000

The kernel mutates A in-place (A[:] = cholesky(A) + triu(A, k=1)), so
setup() copies A from cache before each timing round.

The initialization constructs a symmetric positive-definite matrix via A @ A^T,
which is expensive at N=8000.  setup_cache() runs this once per commit.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/cholesky2/cholesky2.py
def _initialize(N, datatype=np.float64):
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, : i + 1] = np.fromfunction(
            lambda j: (-j % N) / N + 1, (i + 1,), dtype=datatype
        )
        A[i, i + 1 :] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)
    return A


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/cholesky2/cholesky2_numpy.py
def _kernel(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)


_PRESETS = {
    "M": {"N": 2200},
    "L": {"N": 8000},
}


class BenchCholesky2:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        self.A = cache[preset].copy()  # kernel mutates A in-place

    def time_cholesky2(self, cache, preset):
        _kernel(self.A)
