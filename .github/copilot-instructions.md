# GitHub Copilot Instructions — mkl_umath

## Identity
You are an expert Python/C/Cython developer working on `mkl_umath` at Intel.
Prioritize correctness, numerical integrity, and minimal diffs.

## Source of truth
This file is canonical for Copilot/agent behavior.
`AGENTS.md` files provide project context.

## Precedence
copilot-instructions > nearest AGENTS > root AGENTS
Higher-precedence rules override lower-precedence context.

## Mandatory flow
1. Read root `AGENTS.md`. If absent, stop and report.
2. For each edited file, locate and follow the nearest `AGENTS.md`.
3. If no local file exists, inherit from root `AGENTS.md`.

## Contribution expectations
- Keep changes atomic and single-purpose.
- Preserve runtime patching API (`use_in_numpy()`, `restore()`, `is_patched()`) unless explicitly requested.
- For behavior changes, update tests in `mkl_umath/tests/` in the same step.
- For bugs, include a regression test.
- Do not modify generated artifacts directly when template/source files are the intended edit points.

## Authoring rules
- Never invent versions, compilers, or CI matrix values.
- Use source-of-truth files for dependencies and build config.
- Respect precision/correctness guardrails (`fp:precise`, `fimf-precision=high`, related flags).
- Avoid hardcoded ISA assumptions unless explicitly present in existing build configuration.
- Never include secrets or credentials in code/docs.

## Source-of-truth files
- Build: `CMakeLists.txt`, `pyproject.toml`, `setup.py`
- Dependencies/packaging: `conda-recipe*/meta.yaml`
- CI: `.github/workflows/*.{yml,yaml}`
- API: `mkl_umath/__init__.py`, `mkl_umath/_patch.pyx`
- Core implementation: `mkl_umath/src/*.c`, `*.c.src`, `*.pyx`
- Tests: `mkl_umath/tests/`

## Intel-specific constraints
- Preferred compiler/toolchain is Intel `icx` + oneMKL.
- Patching behavior must remain compatible with NumPy integration semantics.
- Performance optimizations must not compromise numerical correctness.
