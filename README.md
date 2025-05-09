[![Conda package](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package.yml/badge.svg)](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package.yml)
[![Editable build using pip and pre-release NumPy](https://github.com/IntelPython/mkl_umath/actions/workflows/build_pip.yaml/badge.svg)](https://github.com/IntelPython/mkl_umath/actions/workflows/build_pip.yaml)
[![Conda package with conda-forge channel only](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package-cf.yml/badge.svg)](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package-cf.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/IntelPython/mkl_umath/badge)](https://securityscorecards.dev/viewer/?uri=github.com/IntelPython/mkl_umath)

# `mkl_umath` --  a NumPy-based Python interface to Intel® oneAPI Math Kernel Library (oneMKL) VM Mathematical Functions

# Introduction
`mkl_umath` started is part of [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html) optimizations to NumPy.
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
## Supported functions
A complete list of `mkl_umath` functions with their supported data types is available [here]().
Note that oneMKL is only used for large contiguous arrays when there is no memory overlap.
These functions can be used in two ways which are explained below.

## Registering `mkl_umath` as the ufunc backend for NumPy
The recommended way to use `mkl_umath` package is through patching stock NumPy. 
To permanently patch stock NumPy ufuncs which are supported in `mkl_umath`, use the following commands:

```python
import mkl_umath
mkl_umath.is_patched()
# False

mkl_umath.use_in_numpy()  # Enable mkl_umath in NumPy
mkl_umath.is_patched()
# True

# calling any ufuncs supported in mkl_umath will use oneMKL implementation

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
    # calling any ufuncs supported in mkl_umath will use oneMKL implementation
# True

# calling numpy.exp will use NumPy own implementation
mkl_umath.is_patched()
# False  
```

Note that when NumPy is patched in this way not only functions that are supported in `mkl_umath` use oneMKL as backend, any other functions that depends on such functions will use oneMKL as well. For instance,

```python
import mkl_umath, numpy

mkl_umath.use_in_numpy()  # Enable mkl_umath in NumPy

# numpy.blackman functions internally depnends on numpy.cos which is a supported function in `mkl_umath`
# so calling numpy.blackman will also use oneMKL for its calculations
%timeit numpy.blackman(size)
# 787 ms ± 1.46 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# compare timing to stock NumPy withou using oneMKL
mkl_umath.restore()
%timeit numpy.blackman(size)
# 3.21 s ± 2.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

## Direct use of `mkl_umath` package
While patching the stock NumPy is the recommended way to leverage `mk_umath`, one can also use `mk_umath` directly as:

It is upcast for instance intefer in addition
```python
import mkl_umath, numpy
a = numpy.random.rand(10**4)
mkl_res = mkl_umath.exp(a)
np_res = numpy.exp(a)
numpy.allclose(mkl_res, np_res)
# True
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
