# AGENTS.md — conda-recipe-cf/

Conda-forge compatible build recipe (alternative to Intel channel recipe).

## Difference from conda-recipe/
- `conda_build_config.yaml` lists the Python versions Intel Python CI builds, NumPy 2 and the compilers;
  its `python:` list is read by `ci/matrix.py`: plain versions, no selectors
- May use different compiler toolchain
- For conda-forge feedstock integration (if upstreamed)

## Files
- **meta.yaml** — conda-forge compatible metadata
- **build.sh** / **bld.bat** — platform build scripts
- **run_tests.{sh,bat}** — test invocation

## Status
- Built and tested by `conda-package-cf.yml` and by Intel Python CI (`intel-ci.yml`)
- Maintained for potential conda-forge submission
- Use `conda-recipe/` for Intel channel builds

## Notes
- If upstreaming to conda-forge, this recipe should be preferred
- Compiler requirements may differ (Clang/GCC vs Intel icx)
