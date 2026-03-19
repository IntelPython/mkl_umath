# Copyright (c) 2019, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Implementation of Numpy universal math functions using Intel(R) MKL and
Intel(R) C compiler runtime.
"""

from . import _init_helper
from ._patch_numpy import (
    is_patched,
    mkl_umath,
    patch_numpy_umath,
    restore,
    restore_numpy_umath,
    use_in_numpy,
)
from ._ufuncs import (
    absolute,
    add,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    cbrt,
    ceil,
    conjugate,
    copysign,
    cos,
    cosh,
    divide,
    equal,
    exp,
    exp2,
    expm1,
    fabs,
    floor,
    fmax,
    fmin,
    frexp,
    greater,
    greater_equal,
    isfinite,
    isinf,
    isnan,
    ldexp,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    modf,
    multiply,
    negative,
    nextafter,
    not_equal,
    positive,
    reciprocal,
    rint,
    sign,
    sin,
    sinh,
    spacing,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    trunc,
)
from ._version import __version__

__all__ = [
    "absolute",
    "add",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cbrt",
    "ceil",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "floor",
    "fmax",
    "fmin",
    "frexp",
    "greater",
    "greater_equal",
    "isfinite",
    "isinf",
    "isnan",
    "ldexp",
    "less",
    "less_equal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "modf",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "positive",
    "reciprocal",
    "rint",
    "sign",
    "sin",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "trunc",
    "is_patched",
    "mkl_umath",
    "patch_numpy_umath",
    "restore",
    "restore_numpy_umath",
    "use_in_numpy",
]

del _init_helper
