# AGENTS.md — benchmarks/

ASV performance suite for `mkl_umath`. Paths are relative to this directory,
so the suite itself is under `benchmarks/benchmarks/`.

## Scope
- `asv.conf.json` — ASV configuration, channels, and regression thresholds
- `benchmarks/micro/` — per-ufunc micro-benchmarks
- `benchmarks/npbench/` — end-to-end kernels adapted from npbench
- `benchmarks/_patch_setup.py` — patches NumPy at import
- `requirements.txt` — Python packages the benchmarks import, besides `asv`
- `README.md` — coverage table, threading default, and run commands

## Guardrails
- Run `asv` from this directory, where `asv.conf.json` is, with
  `--python=same` as in `README.md`; without it, asv builds its own conda
  environments (`environment_type` is `conda`).
- Treat `asv.conf.json` as canonical for ASV settings; treat `README.md` as
  canonical for what each module covers.
- Benchmarks run on patched NumPy only: `_patch_setup.py` raises if patching
  fails, so results never silently come from stock NumPy.
- Comparability across machines depends on the thread default in
  `benchmarks/__init__.py` and the warmup call in each benchmark's `setup`.
  Changing either invalidates comparison against existing results — call it
  out explicitly.
- Keep inputs deterministic; benchmarks seed their own RNG.
- Report performance numbers with reproducible context: hardware, thread count,
  versions, and the command used.
- Results under `.asv/` are local artifacts.
