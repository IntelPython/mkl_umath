# AGENTS.md — conda-recipe-cf/

conda-forge variant of the conda recipe, kept aligned with
conda-forge/mkl_umath-feedstock.

## Differences from `conda-recipe/`
- Resolves dependencies from conda-forge only
- Installs with `pip install` directly; no wheel build or retag
- On Linux, links MKL's GNU threading layer (`-Dmkl_threading=gnu_thread` in
  `build.sh`); `bld.bat` keeps the `intel_thread` default on Windows
- Takes run dependencies from `run_exports`, including `llvm-openmp` and
  NumPy's upper bound; the Intel-channel recipe uses `pin_compatible` for
  `mkl`, `intel-openmp` and `intel-cmplr-lib-rt`, and ignores NumPy's
- Pins the Linux compiler to conda-forge's global pinning (see
  `conda_build_config.yaml`)
- Sets its version by hand in `meta.yaml` instead of reading git tags

Both recipes build with `icx`. Each has its own copy of
`conda_build_config.yaml`; keep the two in step unless a difference is
intentional.

## Guardrails
- Keep conda-forge recipe semantics separate from the Intel-channel recipe.
- Keep the `meta.yaml` version equal to `mkl_umath/_version.py`.
- Keep changes in step with `.github/workflows/conda-package-cf.yml`, which
  builds this recipe with `--error-overlinking` and runs the full test suite
  against the result.
- GitHub CI builds with `--no-test`, so the package test in `meta.yaml` does not
  run there.
