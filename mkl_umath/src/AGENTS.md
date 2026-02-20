# AGENTS.md — mkl_umath/src/

C/Cython implementation layer: MKL VM integration, ufunc loops, and NumPy patching.

## Core files
- **ufuncsmodule.c** — NumPy ufunc registration and module init
- **ufuncsmodule.h** — ufunc module public headers
- **mkl_umath_loops.c.src** — MKL VM loop implementations (template, ~60k LOC)
- **mkl_umath_loops.h.src** — loop function declarations (template)
- **_patch.pyx** — Cython patching layer (runtime NumPy loop replacement)
- **fast_loop_macros.h** — loop generation macros
- **blocking_utils.h** — blocking/chunking utilities for large arrays

## Template system
- `.src` files are processed by `_vendored/conv_template.py` at build time
- Generates type-specialized loops for float32, float64, complex64, complex128
- Pattern: `/**begin repeat ... end repeat**/` blocks

## MKL VM integration
- Calls `vdSin`, `vsExp`, etc. from Intel MKL Vector Math (VM)
- Blocking strategy: chunk large arrays for cache efficiency
- Error handling: MKL VM status → NumPy error state

## Patching mechanism (_patch.pyx)
- Cython extension exposing `use_in_numpy()`, `restore()`, `is_patched()`
- Replaces function pointers in NumPy's ufunc loop tables
- Thread-safe: guards against concurrent patching
- Reversible: stores original pointers for restoration

## Build output
- `mkl_umath_loops.c` → shared library (libmkl_umath_loops.so/.dll)
- `_patch.pyx` → Python extension (_patch.*.so)
- `ufuncsmodule.c` + `__umath_generated.c` → `_ufuncs` extension

## Development notes
- **Precision flags:** fp:precise, fimf-precision=high enforced in CMake
- **Security:** Stack protections, FORTIFY_SOURCE enabled
- **Vectorization:** `-fveclib=SVML -fvectorize` for SIMD
- **Optimization reports:** `cmake -DOPTIMIZATION_REPORT=ON` for `-qopt-report=3`
