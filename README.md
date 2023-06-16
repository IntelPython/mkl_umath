# `mkl_umath`

`mkl_umath._ufuncs` exposes [Intel(R) Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
powered version of loops used in the patched version of [NumPy](https://numpy.org), that used to be included in
[Intel(R) Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html).

Patches were factored out per community feedback ([NEP-36](https://numpy.org/neps/nep-0036-fair-play.html)).

## Building

Intel(R) C compiler and Intel(R) Math Kernel Library are required to build `mkl_umath` from source:

```sh
# ensure that MKL is installed, icc is activated
export MKLROOT=$CONDA_PREFIX
python setup.py config_cc --compiler=intelem build_ext --inplace
```
