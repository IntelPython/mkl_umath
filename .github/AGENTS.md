# AGENTS.md — .github/

CI/CD workflows and repo automation.

## Workflows (source of truth)
- `intel-ci.yml` — builds and tests the conda packages of `conda-recipe/` and
  the wheels, for the platforms and Python versions the recipe and
  `pyproject.toml` declare (`ci/matrix.py`); see `ci/README.md`
- `build_pip.yml` — editable pip build and test with `icx`, including
  pre-release NumPy
- `build-with-clang.yml` — build and test with `icx` from the oneAPI apt
  repository
- `build-with-standard-clang.yml` — build and test with upstream clang
- `pre-commit.yml` — lint/format checks
- `coverity.yml` — Coverity static analysis (see `coverity/README.md`)
- `openssf-scorecard.yml` — OpenSSF Scorecard
- `zizmor.yml` — GitHub Actions security lint

The three pip workflows run `pytest --pyargs mkl_umath` against the editable
install. `intel-ci.yml` installs each built conda package and wheel in a test
job of its own and runs the suite with `ci/test.py`.

## Templates and config
- `pull_request_template.md` — PR description, verification, and checklist
- `ISSUE_TEMPLATE/` — bug report and feature request forms; `config.yml` adds
  contact links that send security reports to Intel's vulnerability process
  and usage questions to the README
- `dependabot.yml` — update schedule for GitHub Actions and pre-commit hooks
- `CODEOWNERS` — default reviewers
- `copilot-instructions.md` — behavior policy for coding agents

## Policy
- Treat workflow YAML as canonical for platform/Python matrices, except `intel-ci.yml`, whose conda matrix is `conda-recipe/conda_build_config.yaml` and whose wheel matrix is the Python classifiers in `pyproject.toml`.
- Avoid doc claims about CI coverage unless present in workflow config.
- Follow the conventions every workflow here uses and zizmor checks: pin each
  action to a full commit SHA with a version comment, set least-privilege
  `permissions`, and set `persist-credentials: false` on checkouts.
