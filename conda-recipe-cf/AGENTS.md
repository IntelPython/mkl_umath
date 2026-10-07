# AGENTS.md — conda-recipe-cf/

conda-forge variant of the conda recipe.

## Differences from `conda-recipe/`
- Resolves dependencies from conda-forge only
- Installs with `pip install` directly; no wheel build or retag
- Links MKL's GNU threading layer (`-Dmkl_threading=gnu_thread` in `build.sh`)
  and uses `llvm-openmp` instead of `intel-openmp`
- Sets its version by hand in `meta.yaml` instead of reading git tags

Both recipes build with `icx` and use the same `conda_build_config.yaml`.

## Guardrails
- Keep conda-forge recipe semantics separate from the Intel-channel recipe.
- Keep the `meta.yaml` version equal to `mkl_umath/_version.py`.
- Keep changes in step with `.github/workflows/conda-package-cf.yml`, which
  builds this recipe and runs the full test suite against the result.
