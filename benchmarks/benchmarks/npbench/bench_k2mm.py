"""npbench wrapper: 2MM (two matrix multiplications) — mkl_umath ops: matmul.

Preset sizes from npbench bench_info/k2mm.json:
  M: NI=2000, NJ=2250, NK=2500, NL=2750
  L: NI=6000, NJ=6500, NK=7000, NL=7500

The kernel mutates D in-place (D[:] = alpha * A @ B @ C + beta * D), so
setup() copies D from cache before each timing round.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/k2mm/k2mm.py
def _initialize(NI, NJ, NK, NL, datatype=np.float64):
    alpha = datatype(1.5)
    beta  = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NK), dtype=datatype)
    B = np.fromfunction(lambda i, j: (i * (j + 1) % NJ) / NJ, (NK, NJ), dtype=datatype)
    C = np.fromfunction(lambda i, j: ((i * (j + 3) + 1) % NL) / NL, (NJ, NL), dtype=datatype)
    D = np.fromfunction(lambda i, j: (i * (j + 2) % NK) / NK, (NI, NL), dtype=datatype)
    return alpha, beta, A, B, C, D


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/k2mm/k2mm_numpy.py
def _kernel(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D


_PRESETS = {
    "M": {"NI": 2000, "NJ": 2250, "NK": 2500, "NL": 2750},
    "L": {"NI": 6000, "NJ": 6500, "NK": 7000, "NL": 7500},
}


class BenchK2mm:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        alpha, beta, A, B, C, D = cache[preset]
        self.alpha = alpha
        self.beta  = beta
        self.A = A
        self.B = B
        self.C = C
        self.D = D.copy()  # mutated in-place

    def time_k2mm(self, cache, preset):
        _kernel(self.alpha, self.beta, self.A, self.B, self.C, self.D)
