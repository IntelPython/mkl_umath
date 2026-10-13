# AGENTS.md — _vendored/

Build-time template tooling copied from NumPy's `numpy/_build_utils`.

## Files
- `conv_template.py` — expands `/**begin repeat ... end repeat**/` blocks in
  `.src` files
- `process_src_template.py` — command-line wrapper that `meson.build` runs on
  the `.src` files; loads `conv_template.py`
- `README.md` — provenance

## Guardrails
- Prefer updating upstream source when feasible; keep local vendored diffs
  minimal.
- Do not refactor vendored code opportunistically in unrelated PRs.
- `black` and `isort` are configured to skip the vendored files
  (`pyproject.toml`).
