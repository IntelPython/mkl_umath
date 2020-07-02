#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

# distutils: language = c
# cython: language_level=2

import mkl_umath._ufuncs as mu
import numpy.core.umath as nu

cimport numpy as cnp
import numpy as np

from libc.stdlib cimport malloc, free

cnp.import_umath()


ctypedef struct function_info:
    cnp.PyUFuncGenericFunction original_function
    cnp.PyUFuncGenericFunction patch_function
    int* signature


cdef class patch:
    cdef int functions_count
    cdef function_info* functions

    functions_dict = {}

    def __cinit__(self):
        cdef int pi, oi

        umaths = [i for i in dir(mu) if isinstance(getattr(mu, i), np.ufunc)]
        self.functions_count = 0
        for umath in umaths:
            mkl_umath = getattr(mu, umath)
            self.functions_count = self.functions_count + mkl_umath.ntypes

        self.functions = <function_info *> malloc(self.functions_count * sizeof(function_info))

        func_number = 0
        for umath in umaths:
            patch_umath = getattr(mu, umath)
            c_patch_umath = <cnp.ufunc>patch_umath
            c_orig_umath = <cnp.ufunc>getattr(nu, umath)
            nargs = c_patch_umath.nargs
            for pi in range(c_patch_umath.ntypes):
                oi = 0
                while oi < c_orig_umath.ntypes:
                    found = True
                    for i in range(c_patch_umath.nargs):
                        if c_patch_umath.types[pi * nargs + i] != c_orig_umath.types[oi * nargs + i]:
                            found = False
                            break
                    if found == True:
                        break
                    oi = oi + 1
                if oi < c_orig_umath.ntypes:
                    self.functions[func_number].original_function = c_orig_umath.functions[oi]
                    self.functions[func_number].patch_function = c_patch_umath.functions[pi]
                    self.functions[func_number].signature = <int *> malloc(nargs * sizeof(int))
                    for i in range(nargs):
                        self.functions[func_number].signature[i] = c_patch_umath.types[pi * nargs + i]
                    self.functions_dict[(umath, patch_umath.types[pi])] = func_number
                    func_number = func_number + 1
                else:
                    raise RuntimeError("Unable to find original function for: " + umath + " " + patch_umath.types[pi])

    def __dealloc__(self):
        for i in range(self.functions_count):
            free(self.functions[i].signature)
        free(self.functions)

    def do_patch(self):
        cdef int res
        cdef cnp.PyUFuncGenericFunction temp
        cdef cnp.PyUFuncGenericFunction function
        cdef int* signature

        for func in self.functions_dict:
            np_umath = getattr(nu, func[0])
            index = self.functions_dict[func]
            function = self.functions[index].patch_function
            signature = self.functions[index].signature
            res = cnp.PyUFunc_ReplaceLoopBySignature(np_umath, function, signature, &temp)

    def do_unpatch(self):
        cdef int res
        cdef cnp.PyUFuncGenericFunction temp
        cdef cnp.PyUFuncGenericFunction function
        cdef int* signature

        for func in self.functions_dict:
            np_umath = getattr(nu, func[0])
            index = self.functions_dict[func]
            function = self.functions[index].original_function
            signature = self.functions[index].signature
            res = cnp.PyUFunc_ReplaceLoopBySignature(np_umath, function, signature, &temp)


from threading import local as threading_local
_tls = threading_local()


def _is_tls_initialized():
    return (getattr(_tls, 'initialized', None) is not None) and (_tls.initialized == True)


def _initialize_tls():
    _tls.patch = patch()
    _tls.initialized = True

    
def do_patch():
    if not _is_tls_initialized():
        _initialize_tls()
    _tls.patch.do_patch()


def do_unpatch():
    if not _is_tls_initialized():
        _initialize_tls()
    _tls.patch.do_unpatch()
