# `mkl_umath`

`mkl_umath._ufuncs` exposes [Intel(R) Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
powered version of loops used in the patched version of [NumPy](https://numpy.org), that used to be included in
[Intel(R) Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html).

Patches were factored out per community feedback ([NEP-36](https://numpy.org/neps/nep-0036-fair-play.html)).

`mkl_umath` started as a part of Intel (R) Distribution for Python* optimizations to NumPy, and is now being released 
as a stand-alone package. It can be installed into conda environment using 

```
   conda install -c intel mkl_umath
```

---

To install mkl_umath Pypi package please use following command:

```
   python -m pip install --i https://pypi.anaconda.org/intel/simple -extra-index-url https://pypi.org/simple mkl_umath
```

If command above installs NumPy package from the Pypi, please use following command to install Intel optimized NumPy wheel package from Anaconda Cloud:

```
   python -m pip install --i https://pypi.anaconda.org/intel/simple -extra-index-url https://pypi.org/simple mkl_umath numpy==<numpy_version>
```

Where `<numpy_version>` should be the latest version from https://anaconda.org/intel/numpy

---

## Building

Intel(R) C compiler and Intel(R) Math Kernel Library are required to build `mkl_umath` from source:

```sh
# ensure that MKL is installed, icc is activated
export MKLROOT=$CONDA_PREFIX
python setup.py config_cc --compiler=intelem build_ext --inplace
```
