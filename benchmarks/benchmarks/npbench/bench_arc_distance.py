"""npbench wrapper: Arc Distance — mkl_umath ops: sin, cos, arctan2, sqrt.

Preset sizes from npbench bench_info/arc_distance.json:
  M: N=1_000_000
  L: N=10_000_000
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/pythran/arc_distance/arc_distance.py
def _initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    t0 = rng.random((N,))
    p0 = rng.random((N,))
    t1 = rng.random((N,))
    p1 = rng.random((N,))
    return t0, p0, t1, p1


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/pythran/arc_distance/arc_distance_numpy.py
def _arc_distance(theta_1, phi_1, theta_2, phi_2):
    temp = (
        np.sin((theta_2 - theta_1) / 2) ** 2
        + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
    )
    return 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))


_PRESETS = {
    "M": {"N": 1_000_000},
    "L": {"N": 10_000_000},
}


class BenchArcDistance:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        self.theta_1, self.phi_1, self.theta_2, self.phi_2 = cache[preset]

    def time_arc_distance(self, cache, preset):
        _arc_distance(self.theta_1, self.phi_1, self.theta_2, self.phi_2)
