"""Micro-benchmarks for mkl_umath trigonometric ufuncs.

Each class times a single ufunc over a Cartesian product of
  dtype  ∈ [float32, float64]
  size   ∈ [10_000, 100_000, 1_000_000]

Arrays are pre-allocated in setup() and reused across timing calls.
Patching is applied once at package import via benchmarks._patch_setup.
"""

import numpy as np


class BenchSin:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-np.pi, np.pi, size).astype(dtype)

    def time_sin(self, dtype, size):
        np.sin(self.x)


class BenchCos:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-np.pi, np.pi, size).astype(dtype)

    def time_cos(self, dtype, size):
        np.cos(self.x)


class BenchTan:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        # Avoid values near ±π/2 where tan diverges
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-1.4, 1.4, size).astype(dtype)

    def time_tan(self, dtype, size):
        np.tan(self.x)


class BenchArcsin:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-1.0, 1.0, size).astype(dtype)

    def time_arcsin(self, dtype, size):
        np.arcsin(self.x)


class BenchArccos:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-1.0, 1.0, size).astype(dtype)

    def time_arccos(self, dtype, size):
        np.arccos(self.x)


class BenchArctan:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-10.0, 10.0, size).astype(dtype)

    def time_arctan(self, dtype, size):
        np.arctan(self.x)


class BenchArctan2:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.y = rng.uniform(-1.0, 1.0, size).astype(dtype)
        self.x = rng.uniform(-1.0, 1.0, size).astype(dtype)

    def time_arctan2(self, dtype, size):
        np.arctan2(self.y, self.x)


class BenchSinh:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        # float32 overflows sinh around ±89; keep well inside that
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-5.0, 5.0, size).astype(dtype)

    def time_sinh(self, dtype, size):
        np.sinh(self.x)


class BenchCosh:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-5.0, 5.0, size).astype(dtype)

    def time_cosh(self, dtype, size):
        np.cosh(self.x)


class BenchTanh:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-5.0, 5.0, size).astype(dtype)

    def time_tanh(self, dtype, size):
        np.tanh(self.x)
