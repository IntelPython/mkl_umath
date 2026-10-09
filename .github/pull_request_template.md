# Description

<!-- What changed and why. Link any related issues. -->

## Verification

<!-- The commands you ran, and the platform and versions you ran them on. -->

- Tests: <!-- e.g. `pytest mkl_umath/tests`, Python 3.12 / NumPy 2.x, Linux -->
- Lint: <!-- `pre-commit run --all-files` -->

## Not verified

<!--
Anything skipped or left to CI, and why. Examples: Windows, the Intel-channel
conda build, the benchmarks. Write "none" if you ran everything relevant.
-->

## Checklist

- [ ] Results match stock NumPy, or the difference is intentional and called out above.
- [ ] Behavior changes have tests in `mkl_umath/tests/`; bug fixes have a regression test.
- [ ] `CHANGELOG.md` updated under `## [dev]` with a `[gh-NNN]` link, or the change isn't user-visible.

<!-- See CONTRIBUTING.md for the build and test workflow, and AGENTS.md for the module map. -->
