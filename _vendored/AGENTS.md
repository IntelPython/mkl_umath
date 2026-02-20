# AGENTS.md — _vendored/

Vendored dependencies from upstream projects (NumPy).

## Files
- **conv_template.py** — NumPy's template processor (from `numpy.distutils`)
- **__init__.py** — Python package marker

## Why vendored?
- `numpy.distutils` removed in NumPy 2.0+ / Python 3.12+
- Needed for `.src` template processing at build time
- Vendored to maintain build compatibility across NumPy versions

## Maintenance
- Source: NumPy's `numpy/distutils/conv_template.py`
- Update if template syntax changes upstream (rare)
- Do not modify vendored code (keep attribution intact)

## Usage
- Imported by `generate_umath.py` for `.src` → `.c` conversion
- Processes `/**begin repeat ... end repeat**/` blocks
