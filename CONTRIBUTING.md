# Contributing to `mkl_umath`

This document covers the development workflow: how to get a working build, how
to run the checks, and what to include in a pull request.

For end-user installation and usage, see [README.md](README.md). For a map of
the source tree, see [`AGENTS.md`](AGENTS.md), which links to the local
`AGENTS.md` files in directories that have their own rules. Security
vulnerabilities go through the process in [SECURITY.md](SECURITY.md).

---

## Development setup

Building requires a C compiler, oneMKL headers and libraries (`mkl-devel`), and
NumPy. CI builds with the Intel `icx` compiler and with upstream `clang`. On
Linux, a conda-forge environment provides all of it, including `icx`:

```sh
# add python=X.Y to target a specific interpreter
conda create -n mkl_umath-dev -c conda-forge --override-channels python pip \
    mkl-devel dpcpp_linux-64 numpy meson-python ninja cmake "cython>=3.1.0" pytest
conda activate mkl_umath-dev
```

Then build in place, which reuses the environment's MKL and NumPy:

```sh
CC=icx pip install -e . --no-build-isolation --no-deps --verbose \
    -Csetup-args=-Dmkl_threading=gnu_thread
```

`gnu_thread` matches the conda-forge recipe. The default threading layer,
`intel_thread`, needs Intel's OpenMP runtime; against conda-forge's MKL,
`import mkl_umath` then fails with `undefined symbol: __atomic_compare_exchange`.
`meson.options` lists the other layers.

`pyproject.toml` defines the supported Python range, and `.github/workflows/` is
canonical for the versions CI covers. `README.md` documents the non-editable
install paths, including the isolated build that resolves its own `mkl-devel`
and `numpy`.

### Rebuilding

`meson-python` rebuilds the extensions on import for editable installs, so
editing `.pyx`, `.c.src`, `generate_umath.py`, or `meson.build` and rerunning
`pytest` is usually enough. Generated sources and the compiled extensions live
under `build/<tag>/` rather than in the source tree. If a build gets into a bad
state, `rm -rf build` and reinstall.

## Running the checks

```sh
pytest mkl_umath/tests          # test suite
pre-commit run --all-files      # lint and format hooks
```

To run a single test, use `pytest mkl_umath/tests/<file>::<test>`; to lint one
file, `pre-commit run --files <path>`.

Install the hooks once with `pre-commit install` and they run on each commit.
`.pre-commit-config.yaml` is the source of truth for the tooling.

Opening a pull request also runs CI, which builds and tests the package across
platforms and Python versions and runs various lint and static-analysis checks.

## Code style

Style is loose, and the pre-commit hooks enforce most of it:

- Python is formatted with `black` and `isort`, with a line length of 80.
- Cython is not touched by `black`. `isort` sorts its imports, `cython-lint`
  checks it against the same 80-column limit, and string literals use double
  quotes.
- C sources follow the repository's `.clang-format`.
- Otherwise, match the surrounding code.

## Dos and don'ts

**Do**

- Keep changes atomic and single-purpose.
- Preserve NumPy behavior. Patching swaps these loops in for NumPy's own, so a
  result that differs from stock NumPy, including NaN and signed-zero handling,
  is a bug. Call out an intentional difference in the PR.
- Add tests in `mkl_umath/tests/` alongside behavior changes, and a regression
  test with every bug fix.
- Keep tests deterministic.
- Edit the `*.src` templates in `mkl_umath/src/` and `generate_umath.py` for
  loop changes. The C they produce is regenerated on every build.
- Keep patching reversible and observable: anything installed can be
  uninstalled, and `is_patched()` reports the truth.
- Keep both extensions free-threading compatible.
- Cite the source-of-truth file for mutable details: `pyproject.toml`,
  `meson.build`, `meson.options`, `conda-recipe*/meta.yaml`,
  `.github/workflows/`.
- Give benchmark numbers reproducible context — hardware, versions, and the
  command you ran.

**Don't**

- Commit generated artifacts, or hand-edit generated C.
- Remove or weaken the floating-point precision flags in `meson.build`.
- Hardcode versions, build flags, CI matrices, or channel URLs in documentation.
- Assert on timing or throughput in the test suite.
- Refactor `_vendored/` opportunistically. Keep local diffs minimal and send
  fixes upstream where you can.
- Introduce ISA-specific assumptions outside explicit build configuration.

## Submitting a change

Work on a branch: the `no-commit-to-branch` hook blocks direct commits to
`main` and `maintenance/*`.

If the change is user-visible — behavior, API, packaging, or build output — add
a `CHANGELOG.md` entry under `## [dev]` in the matching section, with a
`[gh-NNN](https://github.com/IntelPython/mkl_umath/pull/NNN)` link. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Docs,
tooling, and CI-only changes are usually left out.

Then open the PR and fill in the template, including what you verified locally
and what you left to CI.

By contributing you agree that your contributions are licensed under the
BSD-3-Clause terms in [LICENSE.txt](LICENSE.txt).
