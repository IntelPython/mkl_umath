"""npbench wrapper: Mandelbrot set (two variants).

mkl_umath ops: abs, multiply, add.

Preset sizes from npbench bench_info/mandelbrot1.json and mandelbrot2.json:
  M: XN=YN=250/500, maxiter=150/80
  L: XN=YN=833/1000, maxiter=200/100

mandelbrot1 (slow): uses np.less mask + index-based update loop.
mandelbrot2 (fast): uses dynamic array compaction; more cache-friendly.

Both kernels operate on complex128 arrays.  The dominant mkl_umath op is
np.abs() on complex arrays at each iteration step.
"""

import numpy as np

# --- mandelbrot1 ---


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/mandelbrot1/mandelbrot1_numpy.py
def _mandelbrot1(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = np.linspace(xmin, xmax, xn, dtype=np.float64)
    Y = np.linspace(ymin, ymax, yn, dtype=np.float64)
    C = X + Y[:, None] * 1j
    N = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, dtype=np.complex128)
    for n in range(maxiter):
        mask = np.less(abs(Z), horizon)
        N[mask] = n
        Z[mask] = Z[mask] ** 2 + C[mask]
    N[N == maxiter - 1] = 0
    return Z, N


# --- mandelbrot2 ---


# Inlined from spcl/npbench @ main
# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/mandelbrot2/mandelbrot2_numpy.py
def _mandelbrot2(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
    Xi, Yi = np.mgrid[0:xn, 0:yn]
    X = np.linspace(xmin, xmax, xn, dtype=np.float64)[Xi]
    Y = np.linspace(ymin, ymax, yn, dtype=np.float64)[Yi]
    C = X + Y * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    Xi.shape = Yi.shape = C.shape = xn * yn

    Z = np.zeros(C.shape, np.complex128)
    for i in range(itermax):
        if not len(Z):
            break
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)
        rem = np.abs(Z) > horizon
        Z_[Xi[rem], Yi[rem]] = Z[rem]
        N_[Xi[rem], Yi[rem]] = i + 1
        ind = ~rem
        Z = Z[ind]
        C = C[ind]
        Xi = Xi[ind]
        Yi = Yi[ind]
    return Z_, N_


_PRESETS_M1 = {
    "M": {
        "xmin": -1.75,
        "xmax": 0.25,
        "ymin": -1.00,
        "ymax": 1.00,
        "xn": 250,
        "yn": 250,
        "maxiter": 150,
        "horizon": 2.0,
    },
    "L": {
        "xmin": -2.00,
        "xmax": 0.50,
        "ymin": -1.25,
        "ymax": 1.25,
        "xn": 833,
        "yn": 833,
        "maxiter": 200,
        "horizon": 2.0,
    },
}

_PRESETS_M2 = {
    "M": {
        "xmin": -2.00,
        "xmax": 0.50,
        "ymin": -1.25,
        "ymax": 1.25,
        "xn": 500,
        "yn": 500,
        "itermax": 80,
        "horizon": 2.0,
    },
    "L": {
        "xmin": -2.25,
        "xmax": 0.75,
        "ymin": -1.50,
        "ymax": 1.50,
        "xn": 1000,
        "yn": 1000,
        "itermax": 100,
        "horizon": 2.0,
    },
}


class BenchMandelbrot1:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup(self, preset):
        _mandelbrot1(**_PRESETS_M1[preset])

    def time_mandelbrot1(self, preset):
        _mandelbrot1(**_PRESETS_M1[preset])


class BenchMandelbrot2:
    params = (["M", "L"],)
    param_names = ["preset"]
    number = 1
    repeat = 20

    def setup(self, preset):
        _mandelbrot2(**_PRESETS_M2[preset])

    def time_mandelbrot2(self, preset):
        _mandelbrot2(**_PRESETS_M2[preset])
