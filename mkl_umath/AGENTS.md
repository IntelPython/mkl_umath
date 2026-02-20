# AGENTS.md — mkl_umath/

Core MKL-backed ufunc implementation: Python interface, Cython patching, and C/MKL integration.

## Structure
- `__init__.py` — public API surface (`_ufuncs`, `_patch`, version)
- `_init_helper.py` — module initialization helpers
- `_version.py` — version string (dynamic via setuptools)
- `src/` — C implementation and Cython patch layer
- `tests/` — basic functionality and patching tests
- `generate_umath.py` — code generation for ufunc loops
- `generate_umath_doc.py` — docstring generation
- `ufunc_docstrings_numpy{1,2}.py` — NumPy version-specific docstrings

## Patching API
```python
mkl_umath.use_in_numpy()  # Replace NumPy loops with MKL
mkl_umath.restore()       # Restore original NumPy loops
mkl_umath.is_patched()    # Check patch status
```

## Development guardrails
- **API stability:** Patching must be runtime-only, no NumPy source modification
- **Precision:** fp:precise, fimf-precision=high, fprotect-parens are non-negotiable
- **Compatibility:** Must work with upstream NumPy (NEP-36 compliance)
- **Testing:** Add tests to `tests/test_basic.py` for new ufuncs or patch behavior

## Code generation
- `*.src` files are templates processed by `_vendored/conv_template.py`
- Generated files: `src/__umath_generated.c`, loop implementations
- Docstrings: dual NumPy 1.x/2.x support via separate docstring modules

## Notes
- `_patch.pyx` is Cython; changes require Cython rebuild
- MKL VM loops in `src/mkl_umath_loops.c.src`
- `src/ufuncsmodule.c` — NumPy ufunc registration and dispatch
