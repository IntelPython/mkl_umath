"""Micro-benchmarks for in-place complex add and subtract.

Up to VML_ASM_THRESHOLD (100000 elements) mkl_umath runs its own C loops
instead of oneMKL VM, with separate loops for an output that is one of the
inputs: out=a, out=b and z += z (both inputs and the output are one array).

Arrays are pre-allocated in setup() and reused across timing calls.
Patching is applied once at package import via benchmarks._patch_setup.
"""

import numpy as np

from .bench_micro import _uniform


class _Inplace:
    params = (
        ["complex64", "complex128"],
        [10_000, 100_000],
        ["out=a", "out=b", "z += z"],
    )
    param_names = ["dtype", "size", "mode"]
    ufunc = ""

    def setup(self, dtype, size, mode):
        self._func = getattr(np, self.ufunc)
        rng = np.random.default_rng(42)
        a = _uniform(rng, size, dtype)
        b = _uniform(rng, size, dtype)
        if mode == "out=a":
            self.args, self.out = (a, b), a
        elif mode == "out=b":
            self.args, self.out = (a, b), b
        else:
            # repeated z += z doubles z and overflows to inf; zeros keep it
            # finite, and add/subtract run equally fast on any finite value
            z = np.zeros(size, dtype=dtype)
            self.args, self.out = (z, z), z
        self._func(*self.args, out=self.out)

    def time_inplace(self, dtype, size, mode):
        self._func(*self.args, out=self.out)


class BenchAddInplace(_Inplace):
    """In-place add"""

    ufunc = "add"


class BenchSubtractInplace(_Inplace):
    """In-place subtract"""

    ufunc = "subtract"
