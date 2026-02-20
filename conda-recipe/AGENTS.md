# AGENTS.md — conda-recipe/

Conda package build recipe for Intel channel distribution.

## Files
- **meta.yaml** — package metadata, dependencies, build requirements
- **build.sh** — Linux build script
- **bld.bat** — Windows build script
- **conda_build_config.yaml** — build matrix (Python versions, numpy pins)
- **run_tests.{sh,bat}** — post-build test invocation

## Build configuration
- **Channels:** `https://software.repos.intel.com/python/conda`, `conda-forge`
- **Python versions:** 3.10, 3.11, 3.12, 3.13
- **Compilers:** Intel C compiler (icx/icl)
- **Dependencies:** mkl-devel, tbb-devel, dpcpp_{linux,win}-64, numpy-base

## Build outputs
- Conda package: `mkl_umath-<version>-<build>.conda`
- Platform-specific: `linux-64/`, `win-64/`

## CI usage
- Built in `.github/workflows/conda-package.yml`
- Artifacts uploaded per Python version
- Test stage uses built artifacts from channel

## Maintenance
- Keep `conda_build_config.yaml` in sync with CI matrix
- NumPy pin: must match Intel channel NumPy versions
- MKL/TBB versions: track oneAPI releases
