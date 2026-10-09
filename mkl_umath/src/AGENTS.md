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
- Edit the `.src` templates, not the generated `.c`/`.h`.
- Keep results consistent with NumPy for every dtype a loop handles, including
  NaN and signed-zero handling.
- Keep the patch lock and the saved original loops so patching stays
  thread-safe and reversible.
- Keep both extensions free-threading compatible: `freethreading_compatible=True`
  in `_patch_numpy.pyx` and `Py_MOD_GIL_NOT_USED` in `ufuncsmodule.c`.
- Build flags, including precision and hardening flags, live in `meson.build`.
