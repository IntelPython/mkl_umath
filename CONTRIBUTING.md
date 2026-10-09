# Contributing to `mkl_umath`

See [README.md](README.md) for usage, [AGENTS.md](AGENTS.md) for a map of the
source tree, and [SECURITY.md](SECURITY.md) to report a vulnerability.

## Setup

On Linux:

```sh
conda create -n mkl_umath-dev -c conda-forge python pip mkl-devel \
    dpcpp_linux-64 numpy meson-python ninja cmake "cython>=3.1.0" pytest
conda activate mkl_umath-dev
CC=icx pip install -e . --no-build-isolation --no-deps \
    -Csetup-args=-Dmkl_threading=gnu_thread
```

`gnu_thread` is required with conda-forge's MKL; the default threading layer
fails at import.

## Checks

```sh
pytest mkl_umath/tests
pre-commit run --all-files
```

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
`CHANGELOG.md` entry under `[dev]` with a `gh-NNN` link.

Contributions are licensed under the terms in [LICENSE.txt](LICENSE.txt).
