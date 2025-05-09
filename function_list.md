This is a list of current functions available in `mkl_umath`

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
| equal | s, d | primitive | - |
| not_equal | s, d | primitive | - |
| less | s, d | primitive | - |
| less_equal | s, d | primitive | - |
| greater | s, d | primitive | - |
| greater_equal | s, d | primitive | - |
| logical_and | s, d | primitive | - |
| logical_or | s, d | primitive | - |
| logical_xor | s, d | primitive | - |
| logical_not | s, d | primitive | - |
| isnan | s, d | C built-in | - |
| isinf | s, d | C built-in | - |
| isfinite | s, d | C built-in | - |
| signbit | s, d | C built-in | not exposed |
| spacing | s, d | C built-in | - |
| copysign | s, d | oneMKL | - |
| nextafter | s, d | oneMKL | - |
| maximum | s, d | primitive | - |
| minimum | s, d | primitive | - |
| fmax | s, d | oneMKL | - |
| fmin | s, d | oneMKL | - |
| remainder | s, d | oneMKL | - |
| divmod | s, d | primitive | not exposed |
| square | s, d | oneMKL | - |
| reciprocal | s, d | oneMKL | - |
| conjugate | s, d | primitive | - |
| absolute | s, d | oneMKL | - |
| positive | s, d | primitive | - |
| negative | s, d | primitive | - |
| sign | s, d | primitive | - |
| modf | s, d | primitive | add oneMKL implementation |
| frexp | s, d | C built-in | - |
| ldexp | s, d | C built-in | - |
| pow | s, d | - | add oneMKL implementation |
| i0 | s, d | - | add oneMKL implementation |
| hypot | s, d | - | add oneMKL implementation |
| atan2 | s, d | - | add oneMKL implementation^ |
| fmod | s, d | - | add oneMKL implementation |
| floor_divide | s, d | - | add primitive implementation^^ |

| function | data types | implementation | note |
|----------|----------|----------|----------|
| add | c, z | oneMKL | - |
| subtract | c, z | oneMKL | - |
| multiply | c, z | oneMKL | - |
| divide | c, z | oneMKL | - |
| equal | c, z | primitive | - |
| not_equal | c, z | primitive | - |
| less | c, z | primitive | - |
| less_equal | c, z | primitive | - |
| greater | c, z | primitive | - |
| greater_equal | c, z | primitive | - |
| logical_and | c, z | primitive | - |
| logical_or | c, z | primitive | - |
| logical_xor | c, z | primitive | - |
| logical_not | c, z | primitive | - |
| isnan | c, z | C built-in | - |
| isinf | c, z | C built-in | - |
| isfinite | c, z | C built-in | - |
| maximum | c, z | primitive | - |
| minimum | c, z | primitive | - |
| square | c, z | C built-in | - |
| reciprocal | c, z | primitive | - |
| conjugate | c, z | oneMKL | - |
| absolute | c, z | oneMKL | - |
| sign | c, z | primitive | - |
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
| floor_divide | c, z | - | add primitive implementation^^ |

^ NumPy implementation can be similar to `arg`

^^ https://github.com/intel-innersource/libraries.python.intel.condarecipes.numpy-recipe/blob/gold/2021/patches/mkl_umath_remove_floor_divide.patch