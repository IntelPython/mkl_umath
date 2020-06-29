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

cimport cpython.pycapsule

cnp.import_umath()

ctypedef struct function_info:
    cnp.PyUFuncGenericFunction np_function
    cnp.PyUFuncGenericFunction mkl_function
    int* signature

ctypedef struct functions_struct:
    int count
    function_info* functions


cdef const char *capsule_name = "functions_cache"


cdef void _capsule_destructor(object caps):
    cdef functions_struct* fs

    if (caps is None):
        print("Nothing to destroy")
        return
    fs = <functions_struct *>cpython.pycapsule.PyCapsule_GetPointer(caps, capsule_name)
    for i in range(fs[0].count):
        free(fs[0].functions[i].signature)
    free(fs[0].functions)
    free(fs)


from threading import local as threading_local
_tls = threading_local()


def _is_tls_initialized():
    return (getattr(_tls, 'initialized', None) is not None) and (_tls.initialized == True)


def _initialize_tls():
    cdef functions_struct* fs
    cdef int funcs_count

    _tls.functions_dict = {}

    umaths = [i for i in dir(mu) if isinstance(getattr(mu, i), np.ufunc)]
    funcs_count = 0
    for umath in umaths:
        mkl_umath = getattr(mu, umath)
        funcs_count = funcs_count + mkl_umath.ntypes

    fs = <functions_struct *> malloc(sizeof(functions_struct))
    fs[0].count = funcs_count
    fs[0].functions = <function_info *> malloc(funcs_count * sizeof(function_info))

    func_number = 0
    for umath in umaths:
        mkl_umath = getattr(mu, umath)
        np_umath = getattr(nu, umath)
        c_mkl_umath = <cnp.ufunc>mkl_umath
        c_np_umath = <cnp.ufunc>np_umath
        for type in mkl_umath.types:
            np_index = np_umath.types.index(type)
            fs[0].functions[func_number].np_function = c_np_umath.functions[np_index]
            mkl_index = mkl_umath.types.index(type)
            fs[0].functions[func_number].mkl_function = c_mkl_umath.functions[mkl_index]

            nargs = c_mkl_umath.nargs
            fs[0].functions[func_number].signature = <int *> malloc(nargs * sizeof(int))
            for i in range(nargs):
                fs[0].functions[func_number].signature[i] = c_mkl_umath.types[mkl_index*nargs + i]

            _tls.functions_dict[(umath, type)] = func_number
            func_number = func_number + 1

    _tls.functions_capsule = cpython.pycapsule.PyCapsule_New(<void *>fs, capsule_name, &_capsule_destructor)

    _tls.initialized = True


def _get_func_dict():
    if not _is_tls_initialized():
        _initialize_tls()
    return _tls.functions_dict


cdef function_info* _get_functions():
    cdef function_info* functions
    cdef functions_struct* fs

    if not _is_tls_initialized():
        _initialize_tls()

    capsule = _tls.functions_capsule
    if (not cpython.pycapsule.PyCapsule_IsValid(capsule, capsule_name)):
        raise ValueError("Internal Error: invalid capsule stored in TLS")
    fs = <functions_struct *>cpython.pycapsule.PyCapsule_GetPointer(capsule, capsule_name)
    return fs[0].functions


cdef void c_do_patch():
    cdef int res
    cdef cnp.PyUFuncGenericFunction temp
    cdef cnp.PyUFuncGenericFunction function
    cdef int* signature

    funcs_dict = _get_func_dict()
    functions = _get_functions()

    for func in funcs_dict:
        np_umath = getattr(nu, func[0])
        index = funcs_dict[func]
        function = functions[index].mkl_function
        signature = functions[index].signature
        res = cnp.PyUFunc_ReplaceLoopBySignature(np_umath, function, signature, &temp)


cdef void c_do_unpatch():
    cdef int res
    cdef cnp.PyUFuncGenericFunction temp
    cdef cnp.PyUFuncGenericFunction function
    cdef int* signature

    funcs_dict = _get_func_dict()
    functions = _get_functions()

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
