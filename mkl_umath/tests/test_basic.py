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

def get_args(args_str):
    args = []
    for s in args_str:
        if s == 'f':
            args.append(np.single(np.random.random_sample()))
        elif s == 'd':
            args.append(np.double(np.random.random_sample()))
        elif s == 'F':
            args.append(np.single(np.random.random_sample()) + np.single(np.random.random_sample()) * 1j)
        elif s == 'D':
            args.append(np.double(np.random.random_sample()) + np.double(np.random.random_sample()) * 1j)
        elif s == 'i':
            args.append(np.int_(np.random.randint(low=1, high=10)))
        elif s == 'l':
            args.append(np.dtype('long').type(np.random.randint(low=1, high=10)))
        elif s == 'q':
            args.append(np.int64(np.random.randint(low=1, high=10)))
        else:
            raise ValueError("Unexpected type specified!")
    return tuple(args)

umaths = [i for i in dir(mu) if isinstance(getattr(mu, i), np.ufunc)]
umaths.remove('arccosh') # expects input greater than 1

generated_cases = {}
for umath in umaths:
    mkl_umath = getattr(mu, umath)
    types = mkl_umath.types
    for type_ in types:
        args_str = type_[:type_.find('->')]
        args = get_args(args_str)
        generated_cases[(umath, type_)] = args

additional_cases = {
    ('arccosh', 'f->f'): (np.single(np.random.random_sample() + 1),),
    ('arccosh', 'd->d'): (np.double(np.random.random_sample() + 1),),
}

test_cases = {**generated_cases, **additional_cases}

def get_id(val):
    return val.__str__()

@pytest.mark.parametrize("case", test_cases, ids=get_id)
def test_umath(case):
    umath, _ = case
    args = test_cases[case]
    mkl_umath = getattr(mu, umath)
    np_umath = getattr(np, umath)
    
    mkl_res = mkl_umath(*args)
    np_res = np_umath(*args)
       
    assert np.allclose(mkl_res, np_res), f"Results for '{umath}': mkl_res: {mkl_res}, np_res: {np_res}"

def test_patch():
    mp.restore()
    assert not mp.is_patched()

    mp.use_in_numpy()  # Enable mkl_umath in Numpy
    assert mp.is_patched()

    mp.restore()  # Disable mkl_umath in Numpy
    assert not mp.is_patched()
