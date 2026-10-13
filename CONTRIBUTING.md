# Contributing to `mkl_umath`

See [README.md](README.md) for usage, [AGENTS.md](AGENTS.md) for a map of the
source tree, and [SECURITY.md](SECURITY.md) to report a vulnerability.

## Setup

On Linux:

```sh
conda create -n mkl_umath-dev --override-channels -c conda-forge python pip \
    mkl-devel dpcpp_linux-64 numpy meson-python ninja cmake "cython>=3.1.0" \
    pytest pre-commit pylint shellcheck
conda activate mkl_umath-dev
CC=icx pip install -e . --no-build-isolation --no-deps \
    -Csetup-args=-Dmkl_threading=gnu_thread
```

`--override-channels` keeps MKL and its OpenMP runtime from mixing with your
configured channels. `gnu_thread` is required with conda-forge's MKL; the
default threading layer fails at import.

Windows is also supported, and the conda workflows build and test it in CI.
`meson.build` has separate GCC-syntax and MSVC-syntax branches, so keep both
working when you change it.

## Checks

From the editable install above:

```sh
pytest mkl_umath/tests
pre-commit run --all-files
```

The `pylint` and `shellcheck` hooks run the copies installed in the
environment. Run `pre-commit install` once to run the hooks on each commit.

The pre-commit hooks enforce formatting; otherwise, match the surrounding code.

## Guidelines

- Keep changes small and focused.
- Match stock NumPy results, and call out any intentional difference.
- Add tests with behavior changes, and a regression test with bug fixes.
- Edit the `*.src` templates and `generate_umath.py`, not generated C.
- Keep the floating-point precision flags in `meson.build`.
- Keep patching reversible.

## Pull requests

Work on a branch and fill in the PR template. For user-visible changes, add a
`CHANGELOG.md` entry under `## [dev]`, in the matching section, with a
`[gh-NNN](https://github.com/IntelPython/mkl_umath/pull/NNN)` link.

Contributions are licensed under the terms in [LICENSE.txt](LICENSE.txt).
