# AGENTS.md — mkl_umath

Entry point for agent context in this repo.

## What this project is
`mkl_umath` provides NumPy ufunc loops backed by Intel® oneMKL Vector Math, and
can patch them into NumPy at runtime. It was factored out of Intel® Distribution
for Python* per NEP-36 (Fair Play).

## Key components
- **Package and public API:** `mkl_umath/`, `mkl_umath/__init__.py`
- **Ufunc extension:** `mkl_umath/src/ufuncsmodule.c`, plus `__umath_generated.c`
  from `mkl_umath/generate_umath.py` and `_umath_doc_generated.h` from
  `mkl_umath/ufunc_docstrings_numpy2.py`
- **Loop templates:** `mkl_umath/src/mkl_umath_loops.{c,h}.src`
- **Patching:** `mkl_umath/src/_patch_numpy.pyx`; persistent and one-shot
  patching in `mkl_umath/patch.py`, `mkl_umath/with_patch.py`,
  `mkl_umath/_patch_startup.py`, and the `mkl_umath/__main__.py` CLI
- **Tests:** `mkl_umath/tests/`
- **Vendored helpers:** `_vendored/`
- **Packaging:** `conda-recipe/`, `conda-recipe-cf/`
- **Benchmarks:** `benchmarks/`

## Build/runtime basics
- Build system: `pyproject.toml` + `meson.build`, with options in `meson.options`
- Build deps: `mkl-devel`, `numpy`, `meson-python`, `cmake`, `ninja`, `cython`,
  and a C compiler (CI uses `icx` and `clang`)
- Runtime deps: `numpy`, NumPy 2 only (see `pyproject.toml`); the conda recipes
  add the MKL and compiler runtimes
- Platforms: Linux and Windows; `.github/workflows/` is canonical for the matrix
- Setup, checks, and style: `CONTRIBUTING.md`
- Single test: `pytest mkl_umath/tests/<file>.py::<test>`, from a checkout with the
  editable install in `CONTRIBUTING.md`
- Single-file lint: `pre-commit run --files <path>`, from a branch (see
  `CONTRIBUTING.md`)

## Development guardrails
- Preserve NumPy ufunc behavior; patched loops stand in for NumPy's own.
- Edit the `*.src` templates, `generate_umath.py`, and
  `ufunc_docstrings_numpy2.py`, not generated C or headers.
- Keep the floating-point precision flags in `meson.build`: `-fno-fast-math`,
  `-fimf-precision=high`, `-fprotect-parens` (GCC syntax) and `/fp:precise`,
  `/Qimf-precision=high`, `/Qprotect-parens` (MSVC syntax). Both recipes'
  `conda_build_config.yaml` also set `/fp:precise` on Windows. Intel-only flags
  stay behind the compiler checks in `meson.build`.
- Keep the hardening flags in `meson.build`: `-fstack-protector-all`,
  `-D_FORTIFY_SOURCE=2`, `-z relro`/`now`/`noexecstack`, `/GS`, `/DynamicBase`,
  and `/NXCompat`.
- Performance changes must not compromise numerical correctness.
- Avoid hardcoded ISA assumptions beyond what the build configuration sets.
- `meson.build` changes must keep both the GCC-syntax (Linux) and MSVC-syntax
  (Windows) branches working; CI covers both.
- Preserve the patching API (`patch_numpy_umath()`, `restore_numpy_umath()`,
  `is_patched()`, and the `mkl_umath()` context manager) unless asked to change
  it. The deprecated `use_in_numpy()` and `restore()` aliases stay until a
  release removes them.
- Keep patching reversible, with `is_patched()` reporting the truth.
- Keep both extensions free-threading compatible.
- Pair behavior changes with tests and keep diffs minimal.
- Avoid hardcoding mutable versions/matrices/channels in docs, and never invent
  versions, compilers, or CI matrix values; take them from the files below.
- Never put secrets or credentials in code or docs.

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
