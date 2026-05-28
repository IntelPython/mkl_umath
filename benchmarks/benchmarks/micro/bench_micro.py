"""Micro-benchmarks for mkl_umath unary ufuncs.

Times each ufunc over a Cartesian product of
  dtype  in [float32, float64]
  size   in [10_000, 100_000, 1_000_000]

Arrays are pre-allocated in setup() and reused across timing calls.
Patching is applied once at package import via benchmarks._patch_setup.
"""

import numpy as np

_UFUNC_CONFIGS = {
    "exp": {"func": np.exp, "low": -10.0, "high": 10.0},
    "exp2": {"func": np.exp2, "low": -10.0, "high": 10.0},
    "expm1": {"func": np.expm1, "low": -10.0, "high": 10.0},
    "log": {"func": np.log, "low": 1e-3, "high": 1e3},
    "log2": {"func": np.log2, "low": 1e-3, "high": 1e3},
    "log10": {"func": np.log10, "low": 1e-3, "high": 1e3},
    "log1p": {"func": np.log1p, "low": 0.0, "high": 10.0},
    "sin": {"func": np.sin, "low": -np.pi, "high": np.pi},
    "cos": {"func": np.cos, "low": -np.pi, "high": np.pi},
    "tan": {"func": np.tan, "low": -1.4, "high": 1.4},
    "arcsin": {"func": np.arcsin, "low": -1.0, "high": 1.0},
    "arccos": {"func": np.arccos, "low": -1.0, "high": 1.0},
    "arctan": {"func": np.arctan, "low": -10.0, "high": 10.0},
    "sinh": {"func": np.sinh, "low": -5.0, "high": 5.0},
    "cosh": {"func": np.cosh, "low": -5.0, "high": 5.0},
    "tanh": {"func": np.tanh, "low": -5.0, "high": 5.0},
    "arcsinh": {"func": np.arcsinh, "low": -10.0, "high": 10.0},
    "arccosh": {"func": np.arccosh, "low": 1.0, "high": 100.0},
    "arctanh": {"func": np.arctanh, "low": -0.99, "high": 0.99},
    "sqrt": {"func": np.sqrt, "low": 0.0, "high": 100.0},
    "cbrt": {"func": np.cbrt, "low": -100.0, "high": 100.0},
    "square": {"func": np.square, "low": -10.0, "high": 10.0},
    "fabs": {"func": np.fabs, "low": -100.0, "high": 100.0},
    "absolute": {"func": np.absolute, "low": -100.0, "high": 100.0},
    "reciprocal": {"func": np.reciprocal, "low": 0.01, "high": 100.0},
}


class BenchMicro:
    params = (
        sorted(_UFUNC_CONFIGS.keys()),
        ["float32", "float64"],
        [10_000, 100_000, 1_000_000],
    )
    param_names = ["ufunc", "dtype", "size"]

    def setup(self, ufunc, dtype, size):
        cfg = _UFUNC_CONFIGS[ufunc]
        rng = np.random.default_rng(42)
        self.x = rng.uniform(cfg["low"], cfg["high"], size).astype(dtype)
        self._func = cfg["func"]
        self._func(self.x)

    def time_micro(self, ufunc, dtype, size):
        self._func(self.x)


class BenchArctan2:
    """Binary ufunc arctan2"""

    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.y = rng.uniform(-1.0, 1.0, size).astype(dtype)
        self.x = rng.uniform(-1.0, 1.0, size).astype(dtype)
        np.arctan2(self.y, self.x)

    def time_arctan2(self, dtype, size):
        np.arctan2(self.y, self.x)


class BenchPower:
    """Binary ufunc power (arbitrary exponent via MKL vdPow)"""

    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.base = rng.uniform(0.1, 10.0, size).astype(dtype)
        self.exp = rng.uniform(0.5, 3.0, size).astype(dtype)
        np.power(self.base, self.exp)

    def time_power(self, dtype, size):
        np.power(self.base, self.exp)
