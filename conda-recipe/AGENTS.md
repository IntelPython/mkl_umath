# AGENTS.md — conda-recipe/

The conda recipe, built against conda-forge only and kept aligned with
conda-forge/mkl_umath-feedstock.

## Files
- `meta.yaml` — package metadata, dependencies, and the package test
- `build.sh` / `bld.bat` — `pip install` with `icx`. On Linux, `build.sh` links
  MKL's GNU threading layer (`-Dmkl_threading=gnu_thread`); `bld.bat` keeps the
  `intel_thread` default on Windows
- `conda_build_config.yaml` — the Python versions CI builds, NumPy 2 and the
  compiler pins. `ci/matrix.py` reads its `python:` list: one version per
  line, no selectors

## Guardrails
- Treat recipe files as canonical for packaging intent and dependency pins.
- Run dependencies come from `run_exports`, including `llvm-openmp` and
  NumPy's upper bound.
- The version comes from the git tag (`GIT_DESCRIBE_TAG`).
- `intel-ci.yml` builds this recipe with `ci/build.py`, which runs the package
  test in `meta.yaml` (`pip check` and `pytest --pyargs mkl_umath`), and then
  tests the built package in a separate job.
- Wheels don't come from this recipe: `ci/build.py --system wheel` builds them
  from `pyproject.toml`.
