[![Conda package](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package.yml/badge.svg)](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package.yml)
[![Editable build using pip and pre-release NumPy](https://github.com/IntelPython/mkl_umath/actions/workflows/build_pip.yaml/badge.svg)](https://github.com/IntelPython/mkl_umath/actions/workflows/build_pip.yaml)
[![Conda package with conda-forge channel only](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package-cf.yml/badge.svg)](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package-cf.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/IntelPython/mkl_umath/badge)](https://securityscorecards.dev/viewer/?uri=github.com/IntelPython/mkl_umath)

# `mkl_umath` --  a NumPy-based Python interface to Intel® oneAPI Math Kernel Library (oneMKL) VM Mathematical Functions

# Introduction
`mkl_umath` started as a part of [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html) optimizations to NumPy, and is now being released as a stand-alone package.
It offers a thin layered python interface to the [Intel® oneAPI Math Kernel Library (oneMKL) VM Mathematical Functions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/vm-mathematical-functions.html) that allows efficient access to computing values of mathematical functions on real and complex vector arguments. As a result, its performance is close to the performance of native C/Intel® oneMKL.

---
# Installation
`mkl_umath` can be installed into conda environment from Intel`s channel using:

```
   conda install -c https://software.repos.intel.com/python/conda mkl_umath
```

To install `mkl_umath` PyPI package please use the following command:

```
   python -m pip install --i https://software.repos.intel.com/python/pypi -extra-index-url https://pypi.org/simple mkl_umath
```

If command above installs NumPy package from the PyPI, please use the following command to install Intel optimized NumPy wheel package from Intel PyPI Cloud:

```
   python -m pip install --i https://software.repos.intel.com/python/pypi -extra-index-url https://pypi.org/simple mkl_umath numpy==<numpy_version>
```

where `<numpy_version>` should be the latest version from https://software.repos.intel.com/python/conda/.

---
# How to use?
## Use `mkl_umath` mathematical functions
A complete list of functions in `mkl_umath` and data types the are supported for is available [here]().

oneMKL is only used for large contiguous arrays when there is no memort overlap.
if input has type that can safely be upcast to one of supported dtypes it will do it.

```python
import mkl_umath, numpy
a = numpy.random.rand(10**4)
res = mkl_umath.exp(a)

# to get supported dtypes
mkl_umath.exp.types
# ['f->f', 'd->d']

# to get function signature
help(mkl_umath.exp)
# Help on ufunc:
# exp = <ufunc 'exp'>
#    exp(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature])
```

## Use `mkl_umath` within NumPy
In addition to diret use of `mkl_umath`, users have the option to temporarily or permanently patch equivalnet NumPy functions to use `mkl_umath` under the hood.

To permanently patch NumPy functions, use the following commands:

```python
import mkl_umath
mkl_umath.is_patched()
# False

mkl_umath.use_in_numpy()  # Enable mkl_umath in NumPy
mkl_umath.is_patched()
# True

# calling numpy.exp will use oneMKL implementation

mkl_umath.restore()  # Disable mkl_umath in NumPy
mkl_umath.is_patched()
# False 
```

or use context manager and decorator to temporarily patch NumPy ufuncs:

```python
import mkl_umath
mkl_umath.is_patched()
# False

with mkl_umath.mkl_umath():  # Enable mkl_umath in NumPy
    print(mkl_umath.is_patched())
    # calling numpy.exp will use oneMKL implementation
# True

# calling numpy.exp will use NumPy own implementation
mkl_umath.is_patched()
# False  
```

---
# Building from source

Intel® C compiler and Intel® oneAPI Math Kernel Library (oneMKL) are required to build `mkl_umath` from source.

If these are installed as part of a `oneAPI` installation, the following packages must also be installed into the environment
- `cmake`
- `ninja`
- `cython`
- `scikit-build`
- `numpy`

If build dependencies are to be installed with Conda, the following packages must be installed from the Intel® channel
- `mkl-devel`
- `tbb-devel`
- `dpcpp_linux-64` (or `dpcpp_win-64` for Windows)
- `numpy-base`

then the remaining dependencies
- `cmake`
- `ninja`
- `cython`
- `scikit-build`

and for `mkl-devel`, `tbb-devel` and `dpcpp_linux-64` in a Conda environment, `MKLROOT` environment variable must be set
On Linux
```sh
export MKLROOT=$CONDA_PREFIX
```

On Windows
```sh
set MKLROOT=%CONDA_PREFIX%
```

If using `oneAPI`, it must be activated in the environment

On Linux
```
source ${ONEAPI_ROOT}/setvars.sh
```

On Windows
```
call "%ONEAPI_ROOT%\setvars.bat"
```

finally, execute
```
CC=icx pip install --no-build-isolation --no-deps .
```
