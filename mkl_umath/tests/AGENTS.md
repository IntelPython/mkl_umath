# AGENTS.md — mkl_umath/tests/

Test suite for the loops and patching. `meson.build` installs it with the
package.

## Files
- `test_basic.py` — loop results compared against NumPy
- `test_patching.py` — patch and restore state, and the `mkl_umath()` context
  manager
- `test_cli.py` — persistent patch install, uninstall, and status
- `test_freethreading.py` — concurrent ufunc use and patching; the GIL check
  runs only on a free-threaded build

## Expectations
- Behavior changes include test updates in the same PR; bug fixes include a
  regression test.
- Keep tests deterministic and free of timing assertions.

## Entry points
- `pytest mkl_umath/tests` from a checkout
- `pytest --pyargs mkl_umath` against an installed package
