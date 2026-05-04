"""npbench wrapper: Doitgen (multiresolution analysis) — mkl_umath ops: matmul.

Preset sizes from npbench bench_info/doitgen.json:
  S: NR=60,  NQ=60,  NP=128
  L: NR=220, NQ=250, NP=512

The kernel mutates ``A`` in-place (A[:] = ...), so setup() copies from cache.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/doitgen/doitgen.py
def _initialize(NR, NQ, NP, datatype=np.float64):
    A = np.fromfunction(
        lambda i, j, k: ((i * j + k) % NP) / NP, (NR, NQ, NP), dtype=datatype
    )
    C4 = np.fromfunction(
        lambda i, j: (i * j % NP) / NP, (NP, NP), dtype=datatype
    )
    return A, C4


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/doitgen/doitgen_numpy.py
def _kernel(NR, NQ, NP, A, C4):
    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


_PRESETS = {
    "S": {"NR": 60,  "NQ": 60,  "NP": 128},
    "L": {"NR": 220, "NQ": 250, "NP": 512},
}


class BenchDoitgen:
    params = (["S", "L"],)
    param_names = ["preset"]

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        A, C4 = cache[preset]
        p = _PRESETS[preset]
        self.NR, self.NQ, self.NP = p["NR"], p["NQ"], p["NP"]
        self.A = A.copy()  # kernel mutates A in-place
        self.C4 = C4

    def time_doitgen(self, cache, preset):
        _kernel(self.NR, self.NQ, self.NP, self.A, self.C4)
