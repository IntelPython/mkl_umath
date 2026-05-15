"""Micro-benchmarks for mkl_umath exponential and logarithm ufuncs.

Each class times a single ufunc over a Cartesian product of
  dtype  ∈ [float32, float64]
  size   ∈ [10_000, 100_000, 1_000_000]

Arrays are pre-allocated in setup() and reused across timing calls.
Patching is applied once at package import via benchmarks._patch_setup.
"""

import numpy as np


class BenchExp:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        # float32 overflows exp around 88.7; use [-10, 10] safe for both dtypes
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-10.0, 10.0, size).astype(dtype)

    def time_exp(self, dtype, size):
        np.exp(self.x)


class BenchExp2:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        # float32 overflows exp2 around 127
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-10.0, 10.0, size).astype(dtype)

    def time_exp2(self, dtype, size):
        np.exp2(self.x)


class BenchExpm1:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-10.0, 10.0, size).astype(dtype)

    def time_expm1(self, dtype, size):
        np.expm1(self.x)


class BenchLog:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(1e-3, 1e3, size).astype(dtype)

    def time_log(self, dtype, size):
        np.log(self.x)


class BenchLog2:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(1e-3, 1e3, size).astype(dtype)

    def time_log2(self, dtype, size):
        np.log2(self.x)


class BenchLog10:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        rng = np.random.default_rng(42)
        self.x = rng.uniform(1e-3, 1e3, size).astype(dtype)

    def time_log10(self, dtype, size):
        np.log10(self.x)


class BenchLog1p:
    params = (["float32", "float64"], [10_000, 100_000, 1_000_000])
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        # log1p(x) is defined for x > -1; use [0, 10] which is always safe
        rng = np.random.default_rng(42)
        self.x = rng.uniform(0.0, 10.0, size).astype(dtype)

    def time_log1p(self, dtype, size):
        np.log1p(self.x)
