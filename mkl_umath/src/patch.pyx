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

funcs_dict = {}

ctypedef struct function_info:
    cnp.PyUFuncGenericFunction np_function
    cnp.PyUFuncGenericFunction mkl_function
    int* signature

cdef function_info* functions

def fill_functions():
    global functions

    umaths = [i for i in dir(mu) if isinstance(getattr(mu, i), np.ufunc)]
    funcs_count = 0
    for umath in umaths:
        mkl_umath = getattr(mu, umath)
        types = mkl_umath.types
        for type in types:
            funcs_count = funcs_count + 1

    functions = <function_info *> malloc(funcs_count * sizeof(function_info))

    func_number = 0
    for umath in umaths:
        mkl_umath = getattr(mu, umath)
        np_umath = getattr(nu, umath)
        c_mkl_umath = <cnp.ufunc>mkl_umath
        c_np_umath = <cnp.ufunc>np_umath
        for type in mkl_umath.types:
            np_index = np_umath.types.index(type)
            functions[func_number].np_function = c_np_umath.functions[np_index]
            mkl_index = mkl_umath.types.index(type)
            functions[func_number].mkl_function = c_mkl_umath.functions[mkl_index]

            nargs = c_mkl_umath.nargs
            functions[func_number].signature = <int *> malloc(nargs * sizeof(int))
            for i in range(nargs):
                functions[func_number].signature[i] = c_mkl_umath.types[mkl_index*nargs + i]

            funcs_dict[(umath, type)] = func_number
            func_number = func_number + 1


fill_functions()

cdef c_do_patch():
    cdef int res
    cdef cnp.PyUFuncGenericFunction temp
    cdef cnp.PyUFuncGenericFunction function
    cdef int* signature

    global functions

    for func in funcs_dict:
        np_umath = getattr(nu, func[0])
        index = funcs_dict[func]
        function = functions[index].mkl_function
        signature = functions[index].signature
        res = cnp.PyUFunc_ReplaceLoopBySignature(np_umath, function, signature, &temp)


cdef c_do_unpatch():
    cdef int res
    cdef cnp.PyUFuncGenericFunction temp
    cdef cnp.PyUFuncGenericFunction function
    cdef int* signature

    global functions

    for func in funcs_dict:
        np_umath = getattr(nu, func[0])
        index = funcs_dict[func]
        function = functions[index].np_function
        signature = functions[index].signature
        res = cnp.PyUFunc_ReplaceLoopBySignature(np_umath, function, signature, &temp)


def do_patch():
    c_do_patch()

def do_unpatch():
    c_do_unpatch()
