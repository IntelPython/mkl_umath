# AGENTS.md — conda-recipe-cf/

conda-forge variant of the conda recipe.

## Differences from `conda-recipe/`
- Resolves dependencies from conda-forge only
- Installs with `pip install` directly; no wheel build or retag
- On Linux, links MKL's GNU threading layer (`-Dmkl_threading=gnu_thread` in
  `build.sh`); `bld.bat` keeps the `intel_thread` default on Windows
- Uses `llvm-openmp` as a host dependency only, with no run pin; the
  Intel-channel recipe pins `intel-openmp` at run time
- Sets its version by hand in `meta.yaml` instead of reading git tags

Both recipes build with `icx`. Each has its own copy of
`conda_build_config.yaml`; keep the two in step unless a difference is
intentional.

## Guardrails
- Keep conda-forge recipe semantics separate from the Intel-channel recipe.
- Keep the `meta.yaml` version equal to `mkl_umath/_version.py`.
- Keep changes in step with `.github/workflows/conda-package-cf.yml`, which
  builds this recipe and runs the full test suite against the result.
