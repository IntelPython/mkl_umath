# AGENTS.md

Entry point for agent context in this repo.

## What this repository is
`mkl_umath` exposes Intel® OneMKL-powered universal function loops for NumPy, originally part of Intel® Distribution for Python* and factored out per NEP-36 (Fair Play).

It provides:
- `mkl_umath._ufuncs` — OneMKL-backed NumPy ufunc loops
- `mkl_umath._patch_numpy` — runtime patching interface (`patch_numpy_umath` `restore_numpy_umath`, `is_patched()`)
- Performance-optimized math operations (sin, cos, exp, log, etc.) using Intel MKL VM

## Key components
- **Python interface:** `mkl_umath/__init__.py`, `_init_helper.py`
- **Core C implementation:** `mkl_umath/src/` (ufuncsmodule.c, mkl_umath_loops.c.src)
- **Cython patch layer:** `mkl_umath/src/_patch_numpy.pyx`
- **Code generation:** `generate_umath.py`, `generate_umath_doc.py`
- **Build system:** meson-python + Cython

## Build dependencies
**Required:**
- Compiler toolchain: Intel `icx` or `clang` (with Intel-only flags gated when using clang)
- Intel® oneMKL (`mkl-devel`)
- meson-python, CMake, Ninja, Cython, NumPy

**Build against an existing `mkl` installation:**

Install the build dependencies via Conda:
```bash
conda install -c https://software.repos.intel.com/python/conda \
  mkl-devel dpcpp_linux-64 cython meson-python cmake ninja numpy
```
or via pip:
```bash
pip install mkl-devel cython meson-python cmake ninja numpy
```
then build:
```bash
CC=icx pip install --no-deps --no-build-isolation .  # clang is also supported in CI
```

## CI/CD
- **Platforms:** Linux, Windows
- **Python versions:** 3.10, 3.11, 3.12, 3.13, 3.14
- **Workflows:** `.github/workflows/`
  - `conda-package.yml` — main conda build/test pipeline
  - `conda-package-cf.yml` — conda-forge-oriented build/test pipeline
  - `build_pip.yml` — validates pip build with pre-release NumPy
  - `build-with-clang.yml` — Intel clang compatibility check
  - `build-with-standard-clang.yml` — standard clang compatibility check
  - `openssf-scorecard.yml` — security scorecard

## Distribution
- **Conda:** `https://software.repos.intel.com/python/conda`
- **PyPI:** `https://software.repos.intel.com/python/pypi`
- Requires Intel-optimized NumPy from Intel channels

## Usage
```python
import mkl_umath
mkl_umath.patch_numpy_umath()    # Patch NumPy to use MKL loops
# ... perform NumPy operations (now accelerated) ...
mkl_umath.restore_numpy_umath()  # Restore original NumPy loops
```

## How to work in this repo
- **Performance:** Changes should maintain or improve MKL VM utilization
- **Compatibility:** Must work with upstream NumPy APIs (NEP-36 compliance)
- **Testing:** Add tests to `mkl_umath/tests/test_basic.py`
- **Build hygiene:** `meson.build` is the source of truth for build config — verify Linux + Windows
- **Docs:** Update docstrings via `ufunc_docstrings_numpy{1,2}.py`

## Code structure
- **Generated code:** `*.src` files are templates (conv_template.py processes them)
- **Precision flags:** fp:precise, fimf-precision=high, fprotect-parens (non-negotiable)
- **Security:** Stack protection, FORTIFY_SOURCE, NX/DEP enforced in CMake


## Common pitfalls
- **NumPy source:** Requires Intel-optimized NumPy from Intel channel (`software.repos.intel.com/python/conda`). PyPI NumPy may cause runtime failures or incorrect results.
- **Precision flags:** `fp:precise`, `fimf-precision=high` enforce IEEE 754 compliance. Removing them breaks numerical correctness in scientific computing.
- **Patching order:** If using multiple Intel patches (e.g., `mkl_random` + `mkl_umath`), apply `mkl_umath` last. Verify with `is_patched()` after each.
- **Compiler/toolchain:** `icx` and `clang` are both supported; when using clang, keep Intel-only flags behind compiler guards.
- **Build validation:**
  - After setup: `which ${CC:-icx}` → should resolve to the intended compiler toolchain
  - Check: `python -c "import numpy; print(numpy.__version__)"` → confirm NumPy is available

## Notes
- `_vendored/` contains vendored NumPy code generation utilities
- Version in `mkl_umath/_version.py` (read dynamically by `meson.build`)
- Patching is runtime-only; no NumPy source modification

## Directory map
Below directories have local `AGENTS.md` for deeper context:
- `.github/AGENTS.md` — CI/CD workflows and automation
- `mkl_umath/AGENTS.md` — Python API and code generation
- `mkl_umath/src/AGENTS.md` — C/Cython implementation layer
- `mkl_umath/tests/AGENTS.md` — unit tests and validation
- `conda-recipe/AGENTS.md` — Intel channel conda packaging
- `conda-recipe-cf/AGENTS.md` — conda-forge compatible recipe
- `_vendored/AGENTS.md` — vendored NumPy utilities

---

For broader IntelPython ecosystem context, see:
- `dpnp` (Data Parallel NumPy)
- `mkl_random` (MKL-based random number generation)
- `numba-dpex` (Numba + SYCL)
