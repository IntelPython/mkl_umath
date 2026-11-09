# AGENTS.md — conda-recipe/

The conda recipe, built against conda-forge only.

## Files
- **meta.yaml** — metadata, dependencies, test commands (`pytest --pyargs mkl_umath`)
- **build.sh** / **bld.bat** — platform build scripts
- **conda_build_config.yaml** — the Python versions Intel CI builds, NumPy 2 and the compilers;
  its `python:` list is read by `ci/matrix.py`: one version per line, no selectors

## CI
- Built and tested by Intel CI (`intel-ci.yml`)
- Wheels don't come from this recipe: `intel-ci.yml` builds them from `pyproject.toml` with `ci/build.py`
