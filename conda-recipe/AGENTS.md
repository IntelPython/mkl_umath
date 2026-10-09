# AGENTS.md — conda-recipe/

Intel-channel conda packaging.

## Files
- `meta.yaml` — package metadata, dependencies, and the package test
- `build.sh` / `bld.bat` — build a wheel with `python -m build` using `icx`,
  then install it; `build.sh` also retags the wheel's platform
- `conda_build_config.yaml` — NumPy and compiler pins
- `run_tests.sh` / `run_tests.bat` — not used; conda-build runs the test
  commands in `meta.yaml`

## Guardrails
- Treat recipe files as canonical for packaging intent and dependency pins.
- Keep recipe changes in step with `.github/workflows/conda-package.yml`, which
  builds this recipe and runs the full test suite against the result.
- The package test in `meta.yaml` runs only `test_basic.py`.
