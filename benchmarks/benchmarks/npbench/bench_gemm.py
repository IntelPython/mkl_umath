"""npbench wrapper: GEMM (general matrix-matrix multiply).

mkl_umath ops: matmul.

Preset sizes from npbench bench_info/gemm.json:
  M: NI=2500, NJ=2750, NK=3000
  L: NI=7000, NJ=7500, NK=8000

The kernel mutates C in-place (C[:] = alpha * A @ B + beta * C), so
setup() copies C from cache before each timing round.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/gemm/gemm.py
def _initialize(NI, NJ, NK, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(
        lambda i, j: ((i * j + 1) % NI) / NI, (NI, NJ), dtype=datatype
    )
    A = np.fromfunction(
        lambda i, k: (i * (k + 1) % NK) / NK, (NI, NK), dtype=datatype
    )
    B = np.fromfunction(
        lambda k, j: (k * (j + 2) % NJ) / NJ, (NK, NJ), dtype=datatype
    )
    return alpha, beta, C, A, B


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/gemm/gemm_numpy.py
def _kernel(alpha, beta, C, A, B):
    C[:] = alpha * A @ B + beta * C


_PRESETS = {
    "M": {"NI": 2500, "NJ": 2750, "NK": 3000},
    "L": {"NI": 7000, "NJ": 7500, "NK": 8000},
}


class BenchGemm:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        alpha, beta, C, A, B = cache[preset]
        self.alpha = alpha
        self.beta = beta
        self.C = C.copy()  # mutated in-place
        self.A = A
        self.B = B

    def time_gemm(self, cache, preset):
        _kernel(self.alpha, self.beta, self.C, self.A, self.B)
