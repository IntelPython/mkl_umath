# AGENTS.md — mkl_umath

Entry point for agent context in this repo.

## What this project is
`mkl_umath` provides NumPy ufunc loops backed by Intel® oneMKL Vector Math, and
can patch them into NumPy at runtime. It was factored out of Intel® Distribution
for Python* per NEP-36 (Fair Play).

## Key components
- **Package and public API:** `mkl_umath/`, `mkl_umath/__init__.py`
- **Ufunc extension:** `mkl_umath/src/ufuncsmodule.c`, plus `__umath_generated.c`
  from `mkl_umath/generate_umath.py`
- **Loop templates:** `mkl_umath/src/mkl_umath_loops.{c,h}.src`
- **Patching:** `mkl_umath/src/_patch_numpy.pyx`; persistent and one-shot
  patching in `patch.py`, `with_patch.py`, `_patch_startup.py`, and the
  `__main__.py` CLI
- **Tests:** `mkl_umath/tests/`
- **Vendored helpers:** `_vendored/`
- **Packaging:** `conda-recipe/`, `conda-recipe-cf/`
- **Benchmarks:** `benchmarks/`

## Build/runtime basics
- Build system: `pyproject.toml` + `meson.build`, with options in `meson.options`
- Build deps: `mkl-devel`, `numpy`, `meson-python`, `cmake`, `ninja`, `cython`,
  and a C compiler (CI uses `icx` and `clang`)
- Runtime deps: `numpy`; the conda recipes add the MKL and compiler runtimes
- Setup, checks, and style: `CONTRIBUTING.md`
- Single test: `pytest mkl_umath/tests/<file>::<test>`
- Single-file lint: `pre-commit run --files <path>`

## Development guardrails
- Preserve NumPy ufunc behavior; patched loops stand in for NumPy's own.
- Edit the `*.src` templates and `generate_umath.py`, not generated C.
- Keep the floating-point precision flags in `meson.build`; Intel-only flags stay
  behind its compiler checks.
- Keep patching reversible, with `is_patched()` reporting the truth.
- Keep both extensions free-threading compatible.
- Pair behavior changes with tests and keep diffs minimal.
- Avoid hardcoding mutable versions/matrices/channels in docs.

## Where truth lives
- Build/config: `pyproject.toml`, `meson.build`, `meson.options`
- Dependencies: `pyproject.toml`, `conda-recipe*/meta.yaml`
- CI/workflows: `.github/workflows/*.yml`
- Public API: `mkl_umath/__init__.py`, `mkl_umath/src/_patch_numpy.pyx`
- Tests: `mkl_umath/tests/`

For behavior policy, see `.github/copilot-instructions.md`.

## Directory map
Use nearest local `AGENTS.md` when present:
- `.github/AGENTS.md` — CI workflows and automation policy
- `mkl_umath/AGENTS.md` — package modules, API, and code generation
- `mkl_umath/src/AGENTS.md` — loop templates and the two extensions
- `mkl_umath/tests/AGENTS.md` — test scope and conventions
- `conda-recipe/AGENTS.md` — Intel-channel conda packaging
- `conda-recipe-cf/AGENTS.md` — conda-forge recipe
- `_vendored/AGENTS.md` — vendored NumPy template tooling
- `benchmarks/AGENTS.md` — ASV performance suite
