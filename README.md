# `mkl_umath`

To build it:

```sh
# ensure that MKL is installed, icc is activated
MKLROOT=$CONDA_PREFIX python setup.py config_cc --compiler=intelem build_ext --inplace
```

`mkl_umath._ufuncs` exposes MKL-enabled version of loops used in patched NumPy,
that used to be included in Intel(R) Distribution for Python