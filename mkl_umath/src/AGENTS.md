# AGENTS.md — mkl_umath/src/

C and Cython sources for the ufunc loops and the patching extension.

## Key files
- `mkl_umath_loops.c.src`, `mkl_umath_loops.h.src` — loop templates, expanded
  at build time by `_vendored/process_src_template.py` and compiled into the
  `libmkl_umath_loops` shared library
- `ufuncsmodule.c` — the `_ufuncs` extension, built with the generated
  `__umath_generated.c`
- `_patch_numpy.pyx` — the `_patch_numpy` extension; swaps loops into NumPy's
  ufuncs with `PyUFunc_ReplaceLoopBySignature` and keeps the originals for
  restore
- `fast_loop_macros.h`, `blocking_utils.h` — loop helpers

## Guardrails
- Edit sources, not generated files; the root `AGENTS.md` names the edit points.
- Every type signature in `_ufuncs` must match a loop NumPy already has for that
  ufunc, or the first `patch_numpy_umath()` raises `RuntimeError`. The patcher
  finds the ufuncs by scanning `_ufuncs`, so there is no list to update.
- Keep results consistent with NumPy for every dtype a loop handles, including
  NaN and signed-zero handling.
- Keep the saved original loops so patching stays reversible, and the patch
  lock, which serializes patch/restore and their reference counts. It does not
  make patching safe alongside NumPy calls in other threads; see the warning in
  the `patch_numpy_umath` docstring.
- Patch and restore are balanced per thread: a thread can only restore patches
  it applied.
- Keep both extensions free-threading compatible: `freethreading_compatible=True`
  in `_patch_numpy.pyx` and `Py_MOD_GIL_NOT_USED` in `ufuncsmodule.c`.
- Build flags, including precision and hardening flags, live in `meson.build`;
  the recipes' `conda_build_config.yaml` also sets `/fp:precise` on Windows.
