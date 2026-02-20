# AGENTS.md — .github/

CI/CD workflows, automation, security scanning, and package distribution.

## Workflows
- **conda-package.yml** — main build/test pipeline (Linux/Windows, Python 3.10-3.13)
- **build_pip.yaml** — PyPI wheel builds for Intel channel
- **build-with-clang.yml** — Clang compatibility validation
- **openssf-scorecard.yml** — security posture scanning

## CI/CD policy
- Keep build matrix (Python versions, platforms) in workflow files only
- Required checks: conda build + test on all supported Python versions
- Artifact naming: `$PACKAGE_NAME $OS Python $VERSION`
- Conda channel: `https://software.repos.intel.com/python/conda`

## Security
- OpenSSF Scorecard runs automatically
- CODEOWNERS enforces review policy
- Dependabot monitors dependencies (`.github/dependabot.yml`)

## Notes
- Workflow/job renames are breaking for downstream tooling
- Cache key includes `meta.yaml` hash for conda packages
- Artifacts uploaded per-Python-version for parallel testing
