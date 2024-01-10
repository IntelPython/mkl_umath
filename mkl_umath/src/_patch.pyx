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
    cdef bint _is_patched

    functions_dict = dict()

    def __cinit__(self):
        cdef int pi, oi

        self._is_patched = False

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
            res = cnp.PyUFunc_ReplaceLoopBySignature(<cnp.ufunc>np_umath, function, signature, &temp)

        self._is_patched = True

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

        self._is_patched = False

    def is_patched(self):
        return self._is_patched

from threading import local as threading_local
_tls = threading_local()


def _is_tls_initialized():
    return (getattr(_tls, 'initialized', None) is not None) and (_tls.initialized == True)


def _initialize_tls():
    _tls.patch = patch()
    _tls.initialized = True


def use_in_numpy():
    '''
    Enables using of mkl_umath in Numpy.
    '''
    if not _is_tls_initialized():
        _initialize_tls()
    _tls.patch.do_patch()


def restore():
    '''
    Disables using of mkl_umath in Numpy.
    '''
    if not _is_tls_initialized():
        _initialize_tls()
    _tls.patch.do_unpatch()


def is_patched():
    '''
    Returns whether Numpy has been patched with mkl_umath.
    '''
    if not _is_tls_initialized():
        _initialize_tls()
    _tls.patch.is_patched()

from contextlib import ContextDecorator

class mkl_umath(ContextDecorator):
    def __enter__(self):
        use_in_numpy()
        return self

    def __exit__(self, *exc):
        restore()
        return False
