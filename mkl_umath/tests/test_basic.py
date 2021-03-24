# Copyright (c) 2019-2021, Intel Corporation
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

import numpy as np
import mkl_umath._ufuncs as mu
import numpy.core.umath as nu

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
            args.append(np.int(np.random.randint(low=1, high=10)))
        elif s == 'l':
            args.append(np.long(np.random.randint(low=1, high=10)))
        else:
            raise ValueError("Unexpected type specified!")
    return tuple(args)

umaths = [i for i in dir(mu) if isinstance(getattr(mu, i), np.ufunc)]

umaths.remove('arccosh') # expects input greater than 1

# dictionary with test cases
# (umath, types) : args
generated_cases = {}
for umath in umaths:
    mkl_umath = getattr(mu, umath)
    types = mkl_umath.types
    for type in types:
        args_str = type[:type.find('->')]
        args = get_args(args_str)
        generated_cases[(umath, type)] = args

additional_cases = {
('arccosh', 'f->f') : (np.single(np.random.random_sample() + 1),),
('arccosh', 'd->d') : (np.double(np.random.random_sample() + 1),),
}

test_cases = {}
for d in (generated_cases, additional_cases):
    test_cases.update(d)

for case in test_cases:
    umath = case[0]
    type = case[1]
    args = test_cases[case]
    mkl_umath = getattr(mu, umath)
    np_umath = getattr(nu, umath)
    print('*'*80)
    print(umath, type)
    print("args", args)
    mkl_res = mkl_umath(*args)
    np_res = np_umath(*args)
    print("mkl res", mkl_res)
    print("npy res", np_res)

    assert np.array_equal(mkl_res, np_res)

print("Test cases count:", len(test_cases))
print("All looks good!")
