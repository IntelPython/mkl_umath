# AGENTS.md — mkl_umath/

Package sources: public API, patching entry points, and ufunc code generation.

## Key files
- `__init__.py` — public API: the ufuncs from `_ufuncs` and the patching
  functions from `_patch_numpy`
- `_init_helper.py` — imported first by `__init__.py`; in a Windows venv, adds
  `Library/bin` to the DLL search path so the extensions can load
- `generate_umath.py` — generates `__umath_generated.c` in the build directory
- `patch.py`, `_patch_startup.py`, `with_patch.py`, `__main__.py` — persistent
  (`.pth`) and one-shot patching behind `python -m mkl_umath`
- `_version.py` — the version; `meson.build` reads it
- `generate_umath_doc.py`, `ufunc_docstrings_numpy{1,2}.py` — docstring sources
  adapted from NumPy; the build does not run them
- `src/` — loop templates and the two extensions
- `tests/` — test suite

## Guardrails
- Use `patch_numpy_umath()` / `restore_numpy_umath()` in new code and docs;
  `use_in_numpy()` and `restore()` are deprecated aliases.
- Keep patching reversible: anything installed can be uninstalled, and
  `is_patched()` reports the truth.
- New top-level `mkl_umath/*.py` modules must be listed in `py.install_sources`
  in `meson.build`, or they are not installed. `tests/` is installed whole by
  `install_subdir`, and a new extension needs its own `py.extension_module`.
