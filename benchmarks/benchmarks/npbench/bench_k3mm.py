"""npbench wrapper: 3MM (three matrix multiplications) — mkl_umath ops: matmul.

Preset sizes from npbench bench_info/k3mm.json:
  M: NI=2000, NJ=2200, NK=2400, NL=2600, NM=2800
  L: NI=5500, NJ=6000, NK=6500, NL=7000, NM=7500
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/k3mm/k3mm.py
def _initialize(NI, NJ, NK, NL, NM, datatype=np.float64):
    A = np.fromfunction(
        lambda i, j: ((i * j + 1) % NI) / (5 * NI), (NI, NK), dtype=datatype
    )
    B = np.fromfunction(
        lambda i, j: ((i * (j + 1) + 2) % NJ) / (5 * NJ),
        (NK, NJ),
        dtype=datatype,
    )
    C = np.fromfunction(
        lambda i, j: (i * (j + 3) % NL) / (5 * NL), (NJ, NM), dtype=datatype
    )
    D = np.fromfunction(
        lambda i, j: ((i * (j + 2) + 2) % NK) / (5 * NK),
        (NM, NL),
        dtype=datatype,
    )
    return A, B, C, D


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/k3mm/k3mm_numpy.py
def _kernel(A, B, C, D):
    return A @ B @ C @ D


_PRESETS = {
    "M": {"NI": 2000, "NJ": 2200, "NK": 2400, "NL": 2600, "NM": 2800},
    "L": {"NI": 5500, "NJ": 6000, "NK": 6500, "NL": 7000, "NM": 7500},
}


class BenchK3mm:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        self.A, self.B, self.C, self.D = cache[preset]

    def time_k3mm(self, cache, preset):
        _kernel(self.A, self.B, self.C, self.D)
