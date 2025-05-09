In the following a list of current functions available in `mkl_umath` is provided.
In the list the supported data type are indicated. However, if input has type that can safely be upcast to one of supported dtypes, `mkl_umath` still can be used.

Also Note that oneMKL is only used for large contiguous arrays when there is no memort overlap.

The follwoing example shows how to get the supported dtypes and function signature for a function:

```python
import mkl_umath

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
| absolute | s, d, c, z | oneMKL | - |
| add | s, d, c, z | oneMKL | - |
| arccos | s, d, c, z | oneMKL | - |
| arccosh | s, d, c, z | oneMKL | - |
| arcsin | s, d, c, z | oneMKL | - |
| arcsinh | s, d, c, z | oneMKL | - |
| arctan | s, d, c, z | oneMKL | - |
| arctan2 | s, d | oneMKL | - |
| arctanh | s, d, c, z | oneMKL | - |
| cbrt | s, d | oneMKL | - |
| ceil | s, d | oneMKL | - |
| conjugate | s, d, c, z | oneMKL | float dtypes use primitive implementation |
| copysign | s, d | oneMKL | - |
| cos | s, d, c, z | oneMKL | - |
| cosh | s, d, c, z | oneMKL | - |
| divide | s, d, c, z | oneMKL | - |
| divmod | s, d | primitive | - |
| equal | c, z | primitive | - |
| exp | s, d, c, z | oneMKL | - |
| exp2 | s, d | oneMKL | - |
| expm1 | s, d | oneMKL | - |
| fabs | s, d | oneMKL | - |
| float_power | d, z | oneMKL | - |
| floor | s, d | oneMKL | - |
| floor_divide | s, d | primitive | - |
| fmax | s, d | oneMKL | - |
| fmin | s, d | oneMKL | - |
| fmod | s, d | oneMKL | - |
| hypot | s, d | oneMKL | - |
| ldexp | s, d | C built-in | ?? |
| log | s, d, c, z | oneMKL | - |
| log2 | s, d | oneMKL | - |
| log10 | s, d, c, z | oneMKL | - |
| log1p | s, d | oneMKL | - |
| logical_and | s, d, c, z | primitive | - |
| logical_or | s, d, c, z | primitive | - |
| logical_xor | s, d, c, z | primitive | - |
| logical_not | s, d, c, z | primitive | - |
| modf | s, d | oneMKL | - |
| multiply | s, d, c, z | oneMKL | - |
| negative | s, d | primitive | - |
| nextafter | s, d | oneMKL | - |
| not_equal | c, z | primitive | - |
| positive | s, d | primitive | - |
| power | s, d, c, z | oneMKL | - |
| reciprocal | s, d | oneMKL | - |
| remainder | s, d | oneMKL | - |
| rint | s, d | oneMKL | - |
| sign | s, d | primitive | - |
| sin | s, d, c, z | oneMKL | - |
| sinh | s, d, c, z | oneMKL | - |
| sqrt | s, d, c, z | oneMKL | - |
| square | s, d, c, z | oneMKL | complex dtypes use primitive implementation |
| subtract | s, d, c, z | oneMKL | - |
| tan | s, d, c, z | oneMKL | - |
| tanh | s, d, c, z | oneMKL | - |
| trunc | s, d | oneMKL | - |

Tasks:
-) clean up, new test for both float and double
-) complex loop for add, subtract, divide, multiply
-) float loop hypot using c function
-) remove comparison
-) remove minimum and maximum
-) remove signbit
-) enable divmod
?) float loop floordivide and remove its patch in numpy-recipe
-) complex loop exp using c function
-) float loop arctan2 using c function
-) float and complex loop power
-) float and complex loop float_power
-) complex loop log using c function
-) complex loop sqrt using c function
-) complex loop trig and hyper using c function
-) float loop fmod using c function
-) complex loop log10
-) MKL for float loop modf
-) remove float/complex loop is family
-) remove spacing and its definition
?) remove frexp (more timing example)
-) remove complex loop fmax and fmin
-) remove complex loop sign
-) remove complex loop reciprocal
24) decide about ldexp
25) documentation
26) compile win
27) add numpy tests
28) clean up numpy-recipe
29) update size tolerance +,-,*,/

divide is normal definition, true divide is its alias. binary with one output

fmod and remainder return remainder but with sign of dividend and diviser respectively. Both are binary with one output

floor_divide: returns quotient (pairs with remainder). binary with one output

divmod returns quotient and remainder simultaneously. binary with two outputs

modf returns fractional and integral parts of an array, unary with two output
