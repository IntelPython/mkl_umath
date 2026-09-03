# Triaging Coverity Scan findings

Static analysis runs on [Coverity Scan](https://scan.coverity.com) via
[`.github/workflows/coverity.yml`](../.github/workflows/coverity.yml) (weekly + on
demand). Analysis runs on Black Duck's servers; triage is done in the Scan web UI.

Every finding to date is a false positive in **generated** code — the
Cython-generated `_patch_numpy.c` and the `generate_umath.py`-generated
`__umath_generated.c` — not in code we maintain. This guide records the verified
findings and how to keep triage from resetting.

## Where findings come from

`cov-build` captures four C translation units:

- **Hand-written** `mkl_umath/src/ufuncsmodule.c` (ufunc registration + module
  init). Most likely place for a real bug — review every finding.
- **Template-generated** `mkl_umath_loops.c` (from `mkl_umath_loops.c.src` via
  `_vendored/process_src_template.py`). Not Cython — this is our MKL VM loop logic,
  just type-specialized for float32/float64/complex64/complex128. A real loop bug
  can surface here, so treat it like hand-written code, not boilerplate.
- **Generated** `__umath_generated.c` (from `mkl_umath/generate_umath.py`): the
  ufunc registration tables built by the `make_ufuncs` template. Findings here are
  ~always false positives in shared generator boilerplate.
- **Cython-generated** `_patch_numpy.c` (from `_patch_numpy.pyx`): `__Pyx_*` /
  `__pyx_pw_*` / `__pyx_tp_*` helpers and wrappers are boilerplate — findings here
  are ~always false positives. `__pyx_pf_*` functions are the C translation of our
  `.pyx` bodies; a real `.pyx` bug could surface here, though so far all have been
  false positives too (Coverity can't see Python-level invariants).

## Keeping triage durable: the Cython pin

A Cython *version* bump regenerates `_patch_numpy.c` wholesale, which churns the
Coverity CIDs and silently drops their triage — the same boilerplate then returns
under new CIDs. So **Cython is pinned in `coverity.yml`** (not `pyproject.toml`,
so shipped wheels are unaffected). The pin works only because the build runs with
`--no-build-isolation`; bumping it means re-triaging the boilerplate. The
`__umath_generated.c` findings churn likewise if `generate_umath.py` or the NumPy
codegen it mirrors changes.

## Reducing the noise: a Project Component

**Project Settings → Components** buckets defects by a path regex. Define one to
group (not hide) the generated units so they can be filtered out of view —
path-based, so it survives regeneration:

- **Name:** `Generated-code`  **Path regex:** `.*(_patch_numpy|__umath_generated).*\.c`

This matches only the two generated units. Do **not** group `mkl_umath_loops.c` or
`ufuncsmodule.c` — those carry our loop and module logic. Group only — do **not**
mark it *ignored*, as that also drops the `__pyx_pf_*` bodies (see
[declined](#evaluated-and-declined)).

## Review checklist

Don't blanket-ignore the generated files — prioritise instead:

1. **Findings in `ufuncsmodule.c` and `mkl_umath_loops.c`** — review every one;
   the loop file is our type-specialized MKL VM code, not boilerplate.
2. **High/Medium findings in `__pyx_pf_*`** — verify against `_patch_numpy.pyx`; if
   it's a Python-level invariant Coverity can't see, mark `False Positive` with a
   reason.
3. **Known false-positive families below** — carry the recorded disposition;
   match on **checker + mechanism**, not CID (CIDs reset on a Cython bump, a
   `generate_umath.py` change, or an engine upgrade).

## Known false-positive families

Match on **checker + mechanism**, not CID (CIDs reset on a Cython bump or engine
upgrade). Helper names below are from Cython 3.3.0 and vary between versions.
All Minor severity, no runtime or security impact.

| Family | Checker | Why it's a false positive |
| --- | --- | --- |
| `InitOperators`, `__umath_generated.c` (`make_ufuncs` template) | DEADCODE | The template emits `identity = {expr}; if (has_identity && identity == NULL) return -1;` for every ufunc. For `fmax`/`fmin`/`floor` the identity is `ReorderableNone` = `(Py_INCREF(Py_None), Py_None)`, a non-NULL singleton, so `identity == NULL` is provably false and `return -1` is dead. The check *is* needed for other identities (`PyLong_FromLong`, …). |
| `__pyx_tp_traverse_...__patch_impl` (Cython `tp_traverse` slot) | DEADCODE | Cython emits a uniform base-type traversal preamble `e = __Pyx_call_type_traverse(...); if (e) return e;`. `_patch_impl` derives from `object`, whose traverse contributes nothing, so the helper returns 0 and the early-return is dead. |
| `__pyx_pf_..._is_patched` (Cython codegen for `with self._lock:`) | DEADCODE | Cython expands `with` into try/except/finally with both an exception path and a normal-exit path calling `__exit__`. On the normal-exit path the pending-exception temp is NULL, so the `if (__pyx_t_N) {...}` exception-dispatch body is dead. |
| `__Pyx__Import` (Cython generic import runtime helper) | DEADCODE | With `level == -1` the helper does `package_sep = strchr(__Pyx_MODULE_NAME, '.')` to decide on a relative import. The module is compiled with the plain name `_patch_numpy` (`meson.build`), which has no `.`, so `strchr` always returns NULL and the relative-import branch is dead. The variable is "constant" because the module name is a compile-time constant, not a missing assignment. |
| `_patch_impl.__cinit__`, `_patch_numpy.pyx:117` | FORWARD_NULL | `self.functions` is left NULL only when `expected_count == 0` (the sum of `ntypes` over all ufuncs). The `self.functions[...]` deref runs only inside `for pi in range(...ntypes)`, i.e. only when `ntypes >= 1`, which forces `expected_count >= 1` and a successful malloc (failure raises `MemoryError`). Coverity can't correlate the two loops through the opaque `ntypes`/`getattr`. |

The four DEADCODE families mark **Intentional**, the FORWARD_NULL marks **False
Positive** — all with disposition **Ignore**. Optional hardening for the
FORWARD_NULL: add an explicit `if self.functions is NULL: raise RuntimeError(...)`
guard before the deref. Not required for correctness.

## Evaluated and declined

- **Modeling files** correct the behavior of *called* functions; our FPs are
  intraprocedural (dead branches, compile-time-constant guards, `#if`), which
  models can't reach.
- **Dropping the generated units** (hard exclude), e.g. after `cov-build`:
  ```bash
  cov-manage-emit --dir cov-int --tu-pattern "file('.*(_patch_numpy|__umath_generated).*\\.c')" delete
  ```
  Also drops the `__pyx_pf_*` bodies, so it's disabled in favour of the checklist.
