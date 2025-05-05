# changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [dev] (MM/DD/YY)

### Added
* `mkl_umath` is now aligned with both numpy-1.26,x and numpy-2.x.x [gh-65](https://github.com/IntelPython/mkl_umath/pull/65)

### Changed 
* Migrated from `setup.py` to `pyproject toml` [gh-63](https://github.com/IntelPython/mkl_umath/pull/63)

### Fixed
* Fixed a bug for `mkl_umath.is_patched` function [gh-66](https://github.com/IntelPython/mkl_umath/pull/66)


## [0.1.5] (04/09/2025)

### Fixed
* Fixed failures to import `mkl_umath` from virtual environment on Linux

## [0.1.4] (04/09/2025)

### Added
* Added support for `mkl_umath` out-of-the-box in virtual environments on Windows

### Fixed
* Fixed a bug in in-place addition with negative zeros

## [0.1.2] (10/11/2024)

### Added
* Added support for building with NumPy 2.0 and older

### Changed
* Updated build system from removed NumPy distutils to scikit-build, gain ability to build with Intel LLVM compiler ICX
