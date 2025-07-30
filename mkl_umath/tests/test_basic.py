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

import pytest
import numpy as np
import mkl_umath._ufuncs as mu
import mkl_umath._patch as mp

np.random.seed(42)

def get_args(args_str, size, low, high):
    args = []
    a = np.random.uniform(low, high, size)
    b = np.random.uniform(low, high, size)
    for s in args_str:
        if s == "f":
            args.append(np.single(a))
        elif s == "d":
            args.append(np.double(a))
        elif s == "F":
            args.append(np.single(a) + np.single(b) * 1j)
        elif s == "D":
            args.append(np.double(a) + np.double(b) * 1j)
        elif s == "i":
            args.append(np.int_(np.random.randint(low=1, high=10)))
        elif s == "l":
            args.append(np.dtype("long").type(np.random.randint(low=1, high=10)))
        elif s == "q":
            args.append(np.int64(np.random.randint(low=1, high=10)))
        else:
            raise ValueError("Unexpected type specified!")
    return tuple(args)

umaths = [i for i in dir(mu) if isinstance(getattr(mu, i), np.ufunc)]

mkl_cases = {}
fall_back_cases = {}
for umath in umaths:
    mkl_umath = getattr(mu, umath)
    types = mkl_umath.types
    size_mkl = 8192 + 1
    for type_ in types:
        args_str = type_[:type_.find("->")]
        if umath in ["arccos", "arcsin", "arctanh"]:
            low = -1; high = 1
        elif umath in ["log", "log10", "log1p", "log2", "sqrt"]:
            low = 0; high = 10
        elif umath == "arccosh":
            low = 1; high = 10
        else:
            low = -10; high = 10
        
        if umath in ["add", "subtract", "multiply"]:
            size_mkl = 10**5 + 1
        # when size > size_mkl, mkl is used (if supported), 
        # otherwise it falls back on numpy algorithm
        args_mkl = get_args(args_str, size_mkl, low, high)
        args_fall_back = get_args(args_str, 100, low, high)
        mkl_cases[(umath, type_)] = args_mkl
        fall_back_cases[(umath, type_)] = args_fall_back     

test_mkl = {**mkl_cases}
test_fall_back = {**fall_back_cases}

def get_id(val):
    return val.__str__()

@pytest.mark.parametrize("case", test_mkl, ids=get_id)
def test_mkl_umath(case):
    umath, _ = case
    args = test_mkl[case]
    mkl_umath = getattr(mu, umath)
    np_umath = getattr(np, umath)
    
    mkl_res = mkl_umath(*args)
    np_res = np_umath(*args)
       
    assert np.allclose(mkl_res, np_res), f"Results for '{umath}' do not match"


@pytest.mark.parametrize("case", test_fall_back, ids=get_id)
def test_fall_back_umath(case):
    umath, _ = case
    args = test_fall_back[case]
    mkl_umath = getattr(mu, umath)
    np_umath = getattr(np, umath)
    
    mkl_res = mkl_umath(*args)
    np_res = np_umath(*args)

    assert np.allclose(mkl_res, np_res), f"Results for '{umath}' do not match"    


@pytest.mark.parametrize("func", ["add", "subtract", "multiply", "divide"])
@pytest.mark.parametrize("size", [10**5 + 1, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scalar(func, size, dtype):
    a = np.random.uniform(-10, 10, size).astype(dtype)

    # testing implementation in IS_BINARY_CONT_S1 branch
    mkl_res = getattr(mu, func)(a, 1.0)
    np_res = getattr(np, func)(a, 1.0)
    assert np.allclose(mkl_res, np_res), f"Results for '{func}(array, scalar)' do not match"

    # testing implementation in IS_BINARY_CONT_S2 branch
    mkl_res = getattr(mu, func)(1.0, a)
    np_res = getattr(np, func)(1.0, a)
    assert np.allclose(mkl_res, np_res), f"Results for '{func}(scalar, array)' do not match"


@pytest.mark.parametrize("func", ["add", "subtract", "multiply", "divide"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_strided(func, dtype):
    # testing implementation in rthe final else branch
    a = np.random.uniform(-10, 10, 100)[::2].astype(dtype)
    b = np.random.uniform(-10, 10, 100)[::2].astype(dtype)

    mkl_res = getattr(mu, func)(a, b)
    np_res = getattr(np, func)(a, b)
    assert np.allclose(mkl_res, np_res), f"Results for '{func}[strided]' do not match"


@pytest.mark.parametrize("func", ["add", "subtract", "multiply", "divide", "fmax", "fmin"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reduce_float(func, dtype):
    # testing implementation in IS_BINARY_REDUCE branch
    a = np.random.uniform(-10, 10, 50).astype(dtype)
    mkl_func = getattr(mu, func)
    np_func = getattr(np, func)

    mkl_res = mkl_func.reduce(a)
    np_res = np_func.reduce(a)
    assert np.allclose(mkl_res, np_res), f"Results for '{func}[reduce]' do not match"


@pytest.mark.parametrize("func", ["add", "subtract"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_reduce_complex(func, dtype):
    # testing implementation in IS_BINARY_REDUCE branch
    a = np.random.uniform(-10, 10, 100) + 1j * np.random.uniform(-10, 10, 100)
    a = a.astype(dtype)
    mkl_func = getattr(mu, func)
    np_func = getattr(np, func)

    mkl_res = mkl_func.reduce(a)
    np_res = np_func.reduce(a)
    assert np.allclose(mkl_res, np_res), f"Results for '{func}[reduce]' do not match"


def test_patch():
    mp.restore()
    assert not mp.is_patched()

    mp.use_in_numpy()  # Enable mkl_umath in Numpy
    assert mp.is_patched()

    mp.restore()  # Disable mkl_umath in Numpy
    assert not mp.is_patched()
