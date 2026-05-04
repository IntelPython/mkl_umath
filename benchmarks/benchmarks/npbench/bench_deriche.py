"""npbench wrapper: Deriche Edge Detector — mkl_umath ops: exp.

Preset sizes from npbench bench_info/deriche.json:
  S: W=400,  H=200
  L: W=6000, H=3000

Warning: this kernel contains Python for-loops over rows/columns.
At the L preset the Python loops dominate runtime; exp() calls on scalar
floats are measured, not vectorised MKL VM throughput.  The L preset is
included for historical comparability with npbench runs.
"""

import numpy as np


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/deriche/deriche.py
def _initialize(W, H, datatype=np.float64):
    alpha = datatype(0.25)
    imgIn = np.fromfunction(
        lambda i, j: ((313 * i + 991 * j) % 65536) / 65535.0,
        (W, H),
        dtype=datatype,
    )
    return alpha, imgIn


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/polybench/deriche/deriche_numpy.py
def _kernel(alpha, imgIn):
    k = (
        (1.0 - np.exp(-alpha))
        * (1.0 - np.exp(-alpha))
        / (1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    )
    a1 = a5 = k
    a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = a8 = -k * np.exp(-2.0 * alpha)
    b1 = 2.0 ** (-alpha)
    b2 = -np.exp(-2.0 * alpha)
    c1 = c2 = 1

    y1 = np.empty_like(imgIn)
    y1[:, 0] = a1 * imgIn[:, 0]
    y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    for j in range(2, imgIn.shape[1]):
        y1[:, j] = (
            a1 * imgIn[:, j]
            + a2 * imgIn[:, j - 1]
            + b1 * y1[:, j - 1]
            + b2 * y1[:, j - 2]
        )

    y2 = np.empty_like(imgIn)
    y2[:, -1] = 0.0
    y2[:, -2] = a3 * imgIn[:, -1]
    for j in range(imgIn.shape[1] - 3, -1, -1):
        y2[:, j] = (
            a3 * imgIn[:, j + 1]
            + a4 * imgIn[:, j + 2]
            + b1 * y2[:, j + 1]
            + b2 * y2[:, j + 2]
        )

    imgOut = c1 * (y1 + y2)

    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, imgIn.shape[0]):
        y1[i, :] = (
            a5 * imgOut[i, :]
            + a6 * imgOut[i - 1, :]
            + b1 * y1[i - 1, :]
            + b2 * y1[i - 2, :]
        )

    y2[-1, :] = 0.0
    y2[-2, :] = a7 * imgOut[-1, :]
    for i in range(imgIn.shape[0] - 3, -1, -1):
        y2[i, :] = (
            a7 * imgOut[i + 1, :]
            + a8 * imgOut[i + 2, :]
            + b1 * y2[i + 1, :]
            + b2 * y2[i + 2, :]
        )

    return c2 * (y1 + y2)


_PRESETS = {
    "S": {"W": 400,  "H": 200},
    "L": {"W": 6000, "H": 3000},
}


class BenchDeriche:
    # L preset has Python loops over 6000 rows — allow extra time
    timeout = 600

    params = (["S", "L"],)
    param_names = ["preset"]

    def setup_cache(self):
        return {p: _initialize(**kw) for p, kw in _PRESETS.items()}

    def setup(self, cache, preset):
        self.alpha, self.imgIn = cache[preset]

    def time_deriche(self, cache, preset):
        _kernel(self.alpha, self.imgIn)
