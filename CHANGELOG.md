# changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [dev] - YYYY-MM-DD

### Added
* Added mkl implementation for floating point data-types of `exp2`, `log2`, `fabs`, `copysign`, `nextafter`, `fmax`, `fmin` and `remainder` functions [gh-81](https://github.com/IntelPython/mkl_umath/pull/81)
* Added mkl implementation for complex data-types of `conjugate` and `absolute` functions [gh-86](https://github.com/IntelPython/mkl_umath/pull/86)
* Enabled support of Python 3.13 [gh-101](https://github.com/IntelPython/mkl_umath/pull/101)
* Added mkl implementation for complex data-types of `add`, `subtract`, `multiply` and `divide` functions [gh-102](https://github.com/IntelPython/mkl_umath/pull/102)
* Added support for `floor_divide` [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)
* Enabled support for `divmod` [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)
* Added oneMKL implementation for floating point data-types of `arctan2`, `hypot`, `fmod` function [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)

### Changed 
* Dropped support for `maximum` and `minimum` [gh-104](https://github.com/IntelPython/mkl_umath/pull/104)
* Dropped support for COMPLEX LOOPs of `fmax` and `fmin` [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)
* Dropped support for FLOAT LOOPs of `equal` and `not_equal` [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)
* Dropped support for `less`, `less_equal`, `greater`, `greater_equal` [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)
* Dropped support for `isinf`, `isnan` and `isfinite` [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)
* Dropped support for COMPLEX LOOPs of `sign` and `reciprocal` [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)
* Dropped support for `spacing` [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)
* Dropped support for `frexp` [gh-??](https://github.com/IntelPython/mkl_umath/pull/??)

## [0.2.0] - 2025-06-03
This release updates `mkl_umath` to be aligned with both numpy-1.26.x and numpy-2.x.x.

### Added
* The definition of `sign` function for complex floating point data types is updated to match numpy-2.x.x [gh-65](https://github.com/IntelPython/mkl_umath/pull/65)
* `ldexp` function is updated to allow `int64` explicitly similar to numpy-2.x.x behavior [gh-73](https://github.com/IntelPython/mkl_umath/pull/73)

### Changed 
* Migrated from `setup.py` to `pyproject toml` [gh-63](https://github.com/IntelPython/mkl_umath/pull/63)
* Changed to dynamic linking and added interface and threading layers [gh-72](https://github.com/IntelPython/mkl_umath/pull/72)

### Fixed
* Fixed a bug for `mkl_umath.is_patched` function [gh-66](https://github.com/IntelPython/mkl_umath/pull/66)


## [0.1.5] - 2025-04-09

### Fixed
* Fixed failures to import `mkl_umath` from virtual environment on Linux

## [0.1.4] - 2025-04-09

### Added
* Added support for `mkl_umath` out-of-the-box in virtual environments on Windows

### Fixed
* Fixed a bug in in-place addition with negative zeros

## [0.1.2] - 2024-10-11

### Added
* Added support for building with NumPy 2.0 and older

### Changed
* Updated build system from removed NumPy distutils to scikit-build, gain ability to build with Intel LLVM compiler ICX
