# AGENTS.md — mkl_umath/tests/

Unit tests for MKL-backed ufuncs and NumPy patching.

## Test files
- **test_basic.py** — core functionality, patching API, numerical correctness

## Test coverage
- Ufunc correctness: compare MKL loops vs NumPy reference
- Patching: `patch_numpy_umath()`, `restore_numpy_umath()`, `is_patched()` state transitions
- Edge cases: NaN, Inf, empty arrays, large arrays
- Dtype coverage: float32, float64, complex64, complex128

## Running tests
```bash
pytest mkl_umath/tests/
```

## CI integration
- Tests run in conda-package.yml workflow
- Separate test jobs per Python version (3.10-3.14)
- Linux + Windows platforms

## Adding tests
- New ufuncs → add to `test_basic.py` with NumPy reference comparison
- Patching behavior → test state transitions and thread safety
- Use `numpy.testing.assert_allclose` for floating-point comparisons
