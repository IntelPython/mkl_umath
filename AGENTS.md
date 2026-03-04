# AGENTS.md

Entry point for agent context in this repo.

## What this repository is
`mkl_umath` exposes Intel® OneMKL-powered universal function loops for NumPy, originally part of Intel® Distribution for Python* and factored out per NEP-36 (Fair Play).

It provides:
- `mkl_umath._ufuncs` — OneMKL-backed NumPy ufunc loops
- `mkl_umath._patch` — runtime patching interface (`use_in_numpy()`, `restore()`, `is_patched()`)
- Performance-optimized math operations (sin, cos, exp, log, etc.) using Intel MKL VM

## Key components
- **Python interface:** `mkl_umath/__init__.py`, `_init_helper.py`
- **Core C implementation:** `mkl_umath/src/` (ufuncsmodule.c, mkl_umath_loops.c.src)
- **Cython patch layer:** `mkl_umath/src/_patch.pyx`
- **Code generation:** `generate_umath.py`, `generate_umath_doc.py`
- **Build system:** CMake (CMakeLists.txt) + scikit-build

## Build dependencies
**Required:**
- Intel® C Compiler (icx)
- Intel® OneMKL (mkl-devel)
- Intel® TBB (tbb-devel)
- NumPy, Cython, scikit-build, cmake, ninja

**Conda environment:**
```bash
conda install -c https://software.repos.intel.com/python/conda \
  mkl-devel tbb-devel dpcpp_linux-64 numpy-base \
  cmake ninja cython scikit-build
export MKLROOT=$CONDA_PREFIX
CC=icx pip install --no-build-isolation --no-deps .
```

## CI/CD
- **Platforms:** Linux, Windows
- **Python versions:** 3.10, 3.11, 3.12, 3.13, 3.14
- **Workflows:** `.github/workflows/`
  - `conda-package.yml` — main build/test pipeline
  - `build_pip.yaml` — PyPI wheel builds
  - `build-with-clang.yml` — Clang compatibility check
  - `openssf-scorecard.yml` — security scorecard

## Distribution
- **Conda:** `https://software.repos.intel.com/python/conda`
- **PyPI:** `https://software.repos.intel.com/python/pypi`
- Requires Intel-optimized NumPy from Intel channels

## Usage
```python
import mkl_umath
mkl_umath.use_in_numpy()  # Patch NumPy to use MKL loops
# ... perform NumPy operations (now accelerated) ...
mkl_umath.restore()       # Restore original NumPy loops
```

## How to work in this repo
- **Performance:** Changes should maintain or improve MKL VM utilization
- **Compatibility:** Must work with upstream NumPy APIs (NEP-36 compliance)
- **Testing:** Add tests to `mkl_umath/tests/test_basic.py`
- **Build hygiene:** CMake changes → verify Linux + Windows
- **Docs:** Update docstrings via `ufunc_docstrings_numpy{1,2}.py`

## Code structure
- **Generated code:** `*.src` files are templates (conv_template.py processes them)
- **Precision flags:** fp:precise, fimf-precision=high, fprotect-parens (non-negotiable)
- **Security:** Stack protection, FORTIFY_SOURCE, NX/DEP enforced in CMake


## Common pitfalls
- **NumPy source:** Requires Intel-optimized NumPy from Intel channel (`software.repos.intel.com/python/conda`). PyPI NumPy may cause runtime failures or incorrect results.
- **Precision flags:** `fp:precise`, `fimf-precision=high` enforce IEEE 754 compliance. Removing them breaks numerical correctness in scientific computing.
- **Patching order:** If using multiple Intel patches (e.g., `mkl_random` + `mkl_umath`), apply `mkl_umath` last. Verify with `is_patched()` after each.
- **Compiler:** Intel `icx` is the supported compiler. `build-with-clang.yml` validates compatibility, but icx is recommended for production.
- **Build validation:**
  - After setup: `which icx` → should point to conda env or oneAPI location
  - Verify: `echo $MKLROOT` → should be set
  - Check: `python -c "import numpy; print(numpy.__version__)"` → confirm Intel NumPy

## Notes
- `_vendored/` contains vendored NumPy code generation utilities
- Version in `mkl_umath/_version.py` (dynamic via setuptools)
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
