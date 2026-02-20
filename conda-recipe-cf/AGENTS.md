# AGENTS.md — conda-recipe-cf/

Conda-forge compatible build recipe (alternative to Intel channel recipe).

## Difference from conda-recipe/
- No `conda_build_config.yaml` (uses conda-forge defaults)
- May use different compiler toolchain
- For conda-forge feedstock integration (if upstreamed)

## Files
- **meta.yaml** — conda-forge compatible metadata
- **build.sh** / **bld.bat** — platform build scripts
- **run_tests.{sh,bat}** — test invocation

## Status
- Not currently used in main CI workflows
- Maintained for potential conda-forge submission
- Use `conda-recipe/` for Intel channel builds

## Notes
- If upstreaming to conda-forge, this recipe should be preferred
- Compiler requirements may differ (Clang/GCC vs Intel icx)
