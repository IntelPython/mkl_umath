This is a list of OneMKL VM functions that can be used in `mkl_umath`.


This is a list of current functions available in `mkl_umath`

| function name | data types | proposal |
|----------|----------|----------|
| sqrt  | s, d  | - |
| invsqrt  | s, d  | remove - not exposed - no equivalent in `numpy.ufuncs` |
| exp  | s, d  | - |
| exp2  | s, d  | implement MKL version |
| expm1  | s, d  | - |
| erf  | s, d  | remove - not exposed - no equivalent in `numpy.ufuncs`  |
| log  | s, d  | - |
| log2  | s, d  | implement MKL version |
| log10  | s, d  | - |
| log1p  | s, d  | - |
| cos  | s, d  | extend for c, z dtypes |
| sin  | s, d  | extend for c, z dtypes |
| tan  | s, d  | extend for c, z dtypes |
| arccos  | s, d  | extend for c, z dtypes |
| arcsin  | s, d  | extend for c, z dtypes |
| arctan  | s, d  | extend for c, z dtypes |
| cosh  | s, d  | extend for c, z dtypes |
| sinh  | s, d  | extend for c, z dtypes |
| tanh  | s, d  | extend for c, z dtypes |
| arccosh  | s, d  | extend for c, z dtypes |
| arcsinh  | s, d  | extend for c, z dtypes |
| arctanh  | s, d  | extend for c, z dtypes |
| fabs  | s, d  | implement MKL version (use abs function) |
| floor  | s, d  | - |
| ceil  | s, d  | - |
| rint  | s, d  | - |
| trunc  | s, d  | - |
| cbrt  | s, d  | - |
| add  | s, d, c, z  | extend to use OneMKL for c, z dtypes |
| subtract  | s, d, c, z  | extend to use OneMKL for c, z dtypes |
| multiply  | s, d, c, z  | extend to use OneMKL for c, z dtypes |
| divide  | s, d, c, z  | extend to use OneMKL for c, z dtypes |
| equal  | s, d, c, z  | - |
| not_equal  | s, d, c, z  | - |
| less  | s, d, c, z  | - |
| less_equal  | s, d, c, z  | - |
| greater  | s, d, c, z  | - |
| greater_equal  | s, d, c, z  | - |
| logical_and  | s, d, c, z  | NumPy implementation is updated, is `mkl_umath` performance better? |
| logical_or  | s, d, c, z  | NumPy implementation is updated, is `mkl_umath` performance better? |
| logical_xor  | s, d, c, z  | NumPy implementation is updated, is `mkl_umath` performance better? |
| logical_not  | s, d, c, z  | NumPy implementation is updated, is `mkl_umath` performance better? |
| isnan  | s, d, c, z  | - |
| isinf  | s, d, c, z  | - |
| isfinite  | s, d, c, z  | - |
| signbit  | s, d  | - |
| spacing  | s, d  | - |
| copysign  | s, d  | implement MKL version |
| nextafter  | s, d  | implement MKL version |
| maximum  | s, d, c, z  | NumPy implementation is updated |
| minimum  | s, d, c, z  | NumPy implementation is updated |
| fmax  | s, d, c, z  | extend to use OneMKL for s, d dtypes |
| fmin  | s, d  | extend to use OneMKL for s, d dtypes |
| floor_divide  | s, d, c, z  | only in header no implementation |
| remainder  | s, d  | implement MKL version |
| divmod  | s, d  | - |
| square  | s, d, c, z  | - |
| reciprocal  | s, d, c, z  | - |
| ones_like  | s, d, c, z  | remove - not exposed - no equivalent in `numpy.ufuncs`  |
| conjugate  | s, d, c, z  | extend to use OneMKL for c, z dtypes |
| absolute  | s, d, c, z  | extend to use OneMKL for c, z dtypes |
| positive  | s, d  | - |
| negative  | s, d  | NumPy implementation is updated |
| sign  | s, d, c, z  | - |
| modf  | s, d  | implement MKL version |
| frexp  | s, d  | - |
| ldexp  | s, d  | - |
| arg  | c, z  | remove? - not exposed - no equivalent in `numpy.ufuncs` - can be used for angle? |


The following functions are part of `numpy.ufuncs` and are available in OneMKL VM functions but not in `mkl_umath`:

| function name | data types | proposal |
|----------|----------|----------|
| pow  | s, d, c, z  | - |
| atan2  | s, d  | - |
| fmod  | s, d  | - |
| hypot  | s, d  | - |
| i0  | s, d  | - |
| exp  | c, z  | - |
| ln  | c, z  | - |
| log10  | c, z  | - |
| sqrt  | c, z  | - |
| hyperbolic functions  | c, z  | - |
| trigonometric functions  | c, z  | - |
