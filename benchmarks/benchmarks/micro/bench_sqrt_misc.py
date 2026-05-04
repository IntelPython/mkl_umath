"""Micro-benchmarks for mkl_umath sqrt, cbrt, and miscellaneous ufuncs.

Each class times a single ufunc over a Cartesian product of
  dtype  ∈ [float32, float64]
  size   ∈ [10_000, 100_000, 1_000_000]

Arrays are pre-allocated in setup() and reused across timing calls.
Patching is applied once at package import via benchmarks._patch_setup.
"""

import numpy as np


class BenchSqrt:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(0.0, 100.0, size).astype(dtype)

    def time_sqrt(self, dtype, size):
        np.sqrt(self.x)


class BenchCbrt:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-100.0, 100.0, size).astype(dtype)

    def time_cbrt(self, dtype, size):
        np.cbrt(self.x)


class BenchSquare:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-10.0, 10.0, size).astype(dtype)

    def time_square(self, dtype, size):
        np.square(self.x)


class BenchFabs:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-100.0, 100.0, size).astype(dtype)

    def time_fabs(self, dtype, size):
        np.fabs(self.x)


class BenchAbsolute:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-100.0, 100.0, size).astype(dtype)

    def time_absolute(self, dtype, size):
        np.absolute(self.x)


class BenchReciprocal:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        # Avoid values near zero to prevent inf results dominating timing
        rng = np.random.default_rng(42)
        self.x = rng.uniform(0.01, 100.0, size).astype(dtype)

    def time_reciprocal(self, dtype, size):
        np.reciprocal(self.x)
