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

cdef extern from "loops_intel.h":
     cnp.PyUFuncGenericFunction get_func_by_name(char*)

cnp.import_umath()

def _get_func_name(name, type):
    if type.startswith('f'):
        type_str = 'FLOAT'
    elif type.startswith('d'):
        type_str = 'DOUBLE'
    elif type.startswith('F'):
        type_str = 'CFLOAT'
    elif type.startswith('D'):
        type_str = 'CDOUBLE'
    else:
        raise ValueError("_get_func_name: Unexpected type specified!")
    func_name = type_str + '_' + name
    if type.startswith('fl') or type.startswith('dl'):
        func_name = func_name + '_long'
    return func_name


cdef void _fill_signature(signature_str, int* signature):
    for i in range(len(signature_str)):
        if signature_str[i] == 'f':
            signature[i] = cnp.NPY_FLOAT
        elif signature_str[i] == 'd':
            signature[i] = cnp.NPY_DOUBLE
        elif signature_str[i] == 'F':
            signature[i] = cnp.NPY_CFLOAT
        elif signature_str[i] == 'D':
            signature[i] = cnp.NPY_CDOUBLE
        elif signature_str[i] == 'i':
            signature[i] = cnp.NPY_INT
        elif signature_str[i] == 'l':
            signature[i] = cnp.NPY_LONG
        elif signature_str[i] == '?':
            signature[i] = cnp.NPY_BOOL
        else:
            raise ValueError("_fill_signature: Unexpected type specified!")


cdef cnp.PyUFuncGenericFunction fooSaved

funcs_dict = {}
cdef cnp.PyUFuncGenericFunction* originalFuncs


cdef c_do_patch():
    cdef int res

    global originalFuncs

    umaths = [i for i in dir(mu) if isinstance(getattr(mu, i), np.ufunc)]
    func_number = 0
    for umath in umaths:
        mkl_umath = getattr(mu, umath)
        types = mkl_umath.types
        for type in types:
            funcs_dict[(umath, type)] = func_number
            func_number = func_number + 1
    originalFuncs = <cnp.PyUFuncGenericFunction *> malloc(len(funcs_dict) * sizeof(cnp.PyUFuncGenericFunction))

    for func in funcs_dict:
        umath = func[0]
        type = func[1]
        np_umath = getattr(nu, umath)
        signature_str = type.replace('->', '')
        signature = <int *> malloc(len(signature_str) * sizeof(int))
        _fill_signature(signature_str, signature)
        ufunc_name = _get_func_name(umath, type)
        ufunc = get_func_by_name(str.encode(ufunc_name))
        res = cnp.PyUFunc_ReplaceLoopBySignature(np_umath, ufunc, signature, &(originalFuncs[funcs_dict[func]]))
        free(signature)


cdef c_do_unpatch():
    cdef int res
    cdef cnp.PyUFuncGenericFunction temp

    global originalFuncs

    for func in funcs_dict:
        umath = func[0]
        type = func[1]
        np_umath = getattr(nu, umath)
        signature_str = type.replace('->', '')
        signature = <int *> malloc(len(signature_str) * sizeof(int))
        _fill_signature(signature_str, signature)
        res = cnp.PyUFunc_ReplaceLoopBySignature(np_umath, originalFuncs[funcs_dict[(umath, type)]], signature, &temp)
        free(signature)
    free(originalFuncs)


def do_patch():
    c_do_patch()

def do_unpatch():
    c_do_unpatch()
