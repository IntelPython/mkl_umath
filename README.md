[![Conda package](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package.yml/badge.svg)](https://github.com/IntelPython/mkl_umath/actions/workflows/conda-package.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/IntelPython/mkl_umath/badge)](https://securityscorecards.dev/viewer/?uri=github.com/IntelPython/mkl_umath)

# `mkl_umath`

`mkl_umath._ufuncs` exposes [Intel® OneAPI Math Kernel Library (OneMKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
powered version of loops used in the patched version of [NumPy](https://numpy.org), that used to be included in
[Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html).

Patches were factored out per community feedback ([NEP-36](https://numpy.org/neps/nep-0036-fair-play.html)).

`mkl_umath` started as a part of Intel® Distribution for Python* optimizations to NumPy, and is now being released 
as a stand-alone package. It can be installed into conda environment using:

```
   conda install -c https://software.repos.intel.com/python/conda mkl_umath
```

---

To install mkl_umath PyPI package please use following command:

```
   python -m pip install --i https://software.repos.intel.com/python/pypi -extra-index-url https://pypi.org/simple mkl_umath
```

If command above installs NumPy package from the PyPI, please use the following command to install Intel optimized NumPy wheel package from Intel PyPI Cloud:

```
   python -m pip install --i https://software.repos.intel.com/python/pypi -extra-index-url https://pypi.org/simple mkl_umath numpy==<numpy_version>
```

Where `<numpy_version>` should be the latest version from https://software.repos.intel.com/python/conda/

---

## Building

Intel(R) C compiler and Intel(R) OneAPI Math Kernel Library (OneMKL) are required to build `mkl_umath` from source.

If these are installed as part of a `oneAPI` installation, the following packages must also be installed into the environment
- `cmake`
- `ninja`
- `cython`
- `scikit-build`
- `numpy`

If build dependencies are to be installed with Conda, the following packages must be installed from the Intel(R) channel
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
