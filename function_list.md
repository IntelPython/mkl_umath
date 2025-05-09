In the following a list of current functions available in `mkl_umath` is provided.
In the list the supported data type are indicated. However, if input has type that can safely be upcast to one of supported dtypes, `mkl_umath` still can be used it will do it.

Also Note that oneMKL is only used for large contiguous arrays when there is no memort overlap.

The follwoing example shows how to get the supported dtypes and function signature for a function:

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


| function | data types | implementation | note |
|----------|----------|----------|----------|
| sqrt | s, d | oneMKL | - |
| exp | s, d | oneMKL | - |
| exp2 | s, d | oneMKL | - |
| expm1 | s, d | oneMKL | - |
| log | s, d | oneMKL | - |
| log2 | s, d | oneMKL | - |
| log10 | s, d | oneMKL | - |
| log1p | s, d | oneMKL | - |
| cos | s, d | oneMKL | - |
| sin | s, d | oneMKL | - |
| tan | s, d | oneMKL | - |
| arccos | s, d | oneMKL | - |
| arcsin | s, d | oneMKL | - |
| arctan | s, d | oneMKL | - |
| cosh | s, d | oneMKL | - |
| sinh | s, d | oneMKL | - |
| tanh | s, d | oneMKL | - |
| arccosh | s, d | oneMKL | - |
| arcsinh | s, d | oneMKL | - |
| arctanh | s, d | oneMKL | - |
| fabs | s, d | oneMKL | - |
| floor | s, d | oneMKL | - |
| ceil | s, d | oneMKL | - |
| rint | s, d | oneMKL | - |
| trunc | s, d | oneMKL | - |
| cbrt | s, d | oneMKL | - |
| add | s, d | oneMKL | - |
| subtract | s, d | oneMKL | - |
| multiply | s, d | oneMKL | - |
| divide | s, d | oneMKL | - |
| equal | s, d | primitive | remove it |
| not_equal | s, d | primitive | remove it |
| less | s, d | primitive | remove it |
| less_equal | s, d | primitive | remove it |
| greater | s, d | primitive | remove it |
| greater_equal | s, d | primitive | remove it |
| logical_and | s, d | primitive | keep it |
| logical_or | s, d | primitive | keep it |
| logical_xor | s, d | primitive | keep it |
| logical_not | s, d | primitive | keep it |
| isnan | s, d | C built-in | remove it |
| isinf | s, d | C built-in | remove it |
| isfinite | s, d | C built-in | remove it |
| signbit | s, d | C built-in | remove it |
| spacing | s, d | C built-in | remove it |
| copysign | s, d | oneMKL | - |
| nextafter | s, d | oneMKL | - |
| maximum | s, d | primitive | remove it |
| minimum | s, d | primitive | remove it |
| fmax | s, d | oneMKL | - |
| fmin | s, d | oneMKL | - |
| remainder | s, d | oneMKL | - |
| divmod | s, d | primitive | keep it |
| square | s, d | oneMKL | - |
| reciprocal | s, d | oneMKL | - |
| conjugate | s, d | primitive | keep it |
| absolute | s, d | oneMKL | - |
| positive | s, d | primitive | keep it |
| negative | s, d | primitive | keep it |
| sign | s, d | primitive | keep it |
| modf | s, d | primitive | add oneMKL implementation |
| frexp | s, d | C built-in | remove it |
| ldexp | s, d | C built-in | ?? |
| pow | s, d | - | add oneMKL implementation |
| i0 | s, d | - | add oneMKL implementation |
| hypot | s, d | oneMKL | - |
| atan2 | s, d | oneMKL | - |
| fmod | s, d | - | add oneMKL implementation |
| floor_divide | s, d | primitive | - |

| function | data types | implementation | note |
|----------|----------|----------|----------|
| add | c, z | oneMKL | - |
| subtract | c, z | oneMKL | - |
| multiply | c, z | oneMKL | - |
| divide | c, z | oneMKL | - |
| equal | c, z | primitive | keep it |
| not_equal | c, z | primitive | keep it |
| less | c, z | primitive | remove it |
| less_equal | c, z | primitive | remove it |
| greater | c, z | primitive | remove it |
| greater_equal | c, z | primitive | remove it |
| logical_and | c, z | primitive | keep it |
| logical_or | c, z | primitive | keep it |
| logical_xor | c, z | primitive | keep it |
| logical_not | c, z | primitive | keep it |
| isnan | c, z | C built-in | remove it |
| isinf | c, z | C built-in | remove it |
| isfinite | c, z | C built-in | remove it |
| maximum | c, z | primitive | remove it |
| minimum | c, z | primitive | remove it |
| fmax | c, z | primitive | remove it |
| fmin | c, z | primitive | remove it |
| square | c, z | C built-in | keep it |
| reciprocal | c, z | primitive | remove it |
| conjugate | c, z | oneMKL | - |
| absolute | c, z | oneMKL | - |
| sign | c, z | primitive | remove it |
| pow | c, z | - | add oneMKL implementation |
| sqrt | c, z | - | add oneMKL implementation |
| exp | c, z | - | add oneMKL implementation |
| log | c, z | - | add oneMKL implementation |
| log10 | c, z | - | add oneMKL implementation |
| cos | c, z | - | add oneMKL implementation |
| sin | c, z | - | add oneMKL implementation |
| tan | c, z | - | add oneMKL implementation |
| arccos | c, z | - | add oneMKL implementation |
| arcsin | c, z | - | add oneMKL implementation |
| arctan | c, z | - | add oneMKL implementation |
| cosh | c, z | - | add oneMKL implementation |
| sinh | c, z | - | add oneMKL implementation |
| tanh | c, z | - | add oneMKL implementation |
| arccosh | c, z | - | add oneMKL implementation |
| arcsinh | c, z | - | add oneMKL implementation |
| arctanh | c, z | - | add oneMKL implementation |
