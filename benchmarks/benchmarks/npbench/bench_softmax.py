"""npbench wrapper: Softmax — mkl_umath ops: exp, max, sum.

Preset sizes from npbench bench_info/softmax.json:
  S: N=16,  H=16,  SM=128   (float32)
  L: N=64,  H=16,  SM=448   (float32)

npbench initializes this benchmark with float32 explicitly.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/deep_learning/softmax/softmax.py
def _initialize(N, H, SM):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return (x,)


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/deep_learning/softmax/softmax_numpy.py
def _softmax(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


_PRESETS = {
    "S": {"N": 16,  "H": 16, "SM": 128},
    "L": {"N": 64,  "H": 16, "SM": 448},
}


class BenchSoftmax:
    params = (["S", "L"],)
    param_names = ["preset"]

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        (self.x,) = cache[preset]

    def time_softmax(self, cache, preset):
        _softmax(self.x)
