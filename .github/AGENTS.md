# AGENTS.md — .github/

CI/CD workflows and repo automation.

## Workflows (source of truth)
- `conda-package.yml` — Intel-channel conda build and test
- `conda-package-cf.yml` — conda-forge conda build and test
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
install. The conda workflows build with `conda build --no-test` and run the
same command in separate test jobs on the built package.

## Templates and config
- `pull_request_template.md` — PR description, verification, and checklist
- `ISSUE_TEMPLATE/` — bug report and feature request forms; `config.yml` adds
  contact links that send security reports to Intel's vulnerability process
  and usage questions to the README
- `dependabot.yml` — update schedule for GitHub Actions and pre-commit hooks
- `CODEOWNERS` — default reviewers
- `copilot-instructions.md` — behavior policy for coding agents

## Policy
- Treat workflow YAML as canonical for platform/Python matrices.
- Avoid doc claims about CI coverage unless present in workflow config.
- Follow the conventions every workflow here uses and zizmor checks: pin each
  action to a full commit SHA with a version comment, set least-privilege
  `permissions`, and set `persist-credentials: false` on checkouts.
