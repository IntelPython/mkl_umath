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

# distutils: language = c
# cython: language_level=3

import warnings
from contextlib import ContextDecorator
from threading import Lock, local

import mkl_umath._ufuncs as mu

cimport numpy as cnp

import numpy as np

from libc.stdlib cimport free, malloc

cnp.import_umath()

ctypedef struct function_info:
    cnp.PyUFuncGenericFunction original_function
    cnp.PyUFuncGenericFunction patch_function
    int* signature


cdef class _patch_impl:
    cdef int functions_count
    cdef function_info* functions

    functions_dict = dict()

    def __cinit__(self):
        cdef int pi, oi

        umaths = [i for i in dir(mu) if isinstance(getattr(mu, i), np.ufunc)]
        self.functions_count = 0
        for umath in umaths:
            mkl_umath_func = getattr(mu, umath)
            self.functions_count += mkl_umath_func.ntypes

        self.functions = <function_info *> malloc(
            self.functions_count * sizeof(function_info)
        )

        func_number = 0
        for umath in umaths:
            patch_umath = getattr(mu, umath)
            c_patch_umath = <cnp.ufunc>patch_umath
            c_orig_umath = <cnp.ufunc>getattr(np, umath)
            nargs = c_patch_umath.nargs
            for pi in range(c_patch_umath.ntypes):
                oi = 0
                while oi < c_orig_umath.ntypes:
                    found = True
                    for i in range(c_patch_umath.nargs):
                        if (
                            c_patch_umath.types[pi * nargs + i]
                            != c_orig_umath.types[oi * nargs + i]
                        ):
                            found = False
                            break
                    if found is True:
                        break
                    oi = oi + 1
                if oi < c_orig_umath.ntypes:
                    self.functions[func_number].original_function = (
                        c_orig_umath.functions[oi]
                    )
                    self.functions[func_number].patch_function = (
                        c_patch_umath.functions[pi]
                    )
                    self.functions[func_number].signature = (
                        <int *> malloc(nargs * sizeof(int))
                    )
                    for i in range(nargs):
                        self.functions[func_number].signature[i] = (
                            c_patch_umath.types[pi * nargs + i]
                        )
                    self.functions_dict[(umath, patch_umath.types[pi])] = (
                        func_number
                    )
                    func_number = func_number + 1
                else:
                    raise RuntimeError(
                        f"Unable to find original function for: {umath} "
                        f"{patch_umath.types[pi]}"
                    )

    def __dealloc__(self):
        for i in range(self.functions_count):
            free(self.functions[i].signature)
        free(self.functions)

    cdef int _replace_loop(
        self,
        object func,
        cnp.PyUFuncGenericFunction function,
    ) except -1:
        cdef int res
        cdef cnp.PyUFuncGenericFunction temp
        cdef int* signature

        np_umath = getattr(np, func[0])
        index = self.functions_dict[func]
        signature = self.functions[index].signature
        res = cnp.PyUFunc_ReplaceLoopBySignature(
            <cnp.ufunc>np_umath, function, signature, &temp
        )
        return res

    def do_patch(self):
        cdef int index

        for func in self.functions_dict:
            index = self.functions_dict[func]
            if self._replace_loop(
                func, self.functions[index].patch_function
            ) != 0:
                raise RuntimeError(
                    f"Failed to patch {func[0]} with signature {func[1]}. "
                    "NumPy may be partially restored or in an invalid state."
                )

    def do_unpatch(self):
        cdef int index

        for func in self.functions_dict:
            index = self.functions_dict[func]
            if self._replace_loop(
                func, self.functions[index].original_function
            ) != 0:
                raise RuntimeError(
                    f"Failed to restore {func[0]} with signature {func[1]}. "
                    "NumPy may be partially restored or in an invalid state."
                )


class _GlobalPatch:
    def __init__(self):
        self._lock = Lock()
        self._patch_count = 0
        self._tls = local()
        self._patcher = None

    def do_patch(self, verbose=False):
        with self._lock:
            local_count = getattr(self._tls, "local_count", 0)
            if self._patch_count == 0:
                if verbose:
                    print(
                        "Now patching NumPy ufuncs with mkl_umath loops."
                    )
                    print(
                        "Please direct bug reports to "
                        "https://github.com/IntelPython/mkl_umath"
                    )
                if self._patcher is None:
                    # lazy initialization of the patcher to save memory
                    self._patcher = _patch_impl()
                self._patcher.do_patch()

            self._patch_count += 1
            self._tls.local_count = local_count + 1

    def do_restore(self, verbose=False):
        with self._lock:
            local_count = getattr(self._tls, "local_count", 0)
            if local_count <= 0:
                warnings.warn(
                    "restore_numpy_umath called more times than "
                    "patch_numpy_umath in this thread.",
                    RuntimeWarning,
                    stacklevel=1,  # Cython does not add a stacklevel
                )
                return

            next_patch_count = self._patch_count - 1
            if next_patch_count == 0:
                if verbose:
                    print("Now restoring original NumPy loops.")
                self._patcher.do_unpatch()

            self._tls.local_count -= 1
            self._patch_count = next_patch_count

    def is_patched(self):
        with self._lock:
            return self._patch_count > 0


_patch = _GlobalPatch()


def patch_numpy_umath(verbose=False):
    """
    Patch NumPy's ufuncs with mkl_umath's loops.

    Parameters
    ----------
    verbose : bool, optional
        print message when starting the patching process.

    Notes
    -----
    This function uses reference-counted semantics. Each call increments a
    global patch counter. Restoration requires a matching number of calls
    between `patch_numpy_umath` and `restore_numpy_umath`.

    ⚠️ Warning
    -------------------------
    If used in a multi-threaded program, ALL concurrent threads executing NumPy
    operations must either have applied the patch prior to execution, or run
    entirely within the `mkl_umath` context manager. Executing standard NumPy
    calls in one thread while another thread is actively patching or unpatching
    will lead to undefined behavior at best, and segmentation faults at worst.
    For this reason, it is recommended to prefer the `mkl_umath` context
    manager.

    Examples
    --------
    >>> import mkl_umath
    >>> mkl_umath.is_patched()
    # False

    >>> mkl_umath.patch_numpy_umath()  # Enable mkl_umath in Numpy
    >>> mkl_umath.is_patched()
    # True

    >>> mkl_umath.restore_numpy_umath()  # Disable mkl_umath in Numpy
    >>> mkl_umath.is_patched()
    # False
    """
    _patch.do_patch(verbose=verbose)


def restore_numpy_umath(verbose=False):
    """
    Restore NumPy's ufuncs to the original loops.

    Parameters
    ----------
    verbose : bool, optional
        print message when starting restoration process.

    Notes
    -----
    This function uses reference-counted semantics. Each call decrements a
    global patch counter. Restoration requires a matching number of calls
    between `patch_numpy_umath` and `restore_numpy_umath`.

    ⚠️ Warning
    -------------------------
    If used in a multi-threaded program, ALL concurrent threads executing NumPy
    operations must either have applied the patch prior to execution, or run
    entirely within the `mkl_umath` context manager. Executing standard NumPy
    calls in one thread while another thread is actively patching or unpatching
    will lead to undefined behavior at best, and segmentation faults at worst.
    For this reason, it is recommended to prefer the `mkl_umath` context
    manager.

    Examples
    --------
    >>> import mkl_umath
    >>> mkl_umath.is_patched()
    # False

    >>> mkl_umath.patch_numpy_umath()  # Enable mkl_umath in Numpy
    >>> mkl_umath.is_patched()
    # True

    >>> mkl_umath.restore_numpy_umath()  # Disable mkl_umath in Numpy
    >>> mkl_umath.is_patched()
    # False
    """
    _patch.do_restore(verbose=verbose)


def use_in_numpy():
    """
    Deprecated alias for patch_numpy_umath.

    See patch_numpy_umath for details and examples.
    """
    warnings.warn(
        "use_in_numpy is deprecated since mkl_umath 0.4.0 and will be removed "
        "in a future release. Use `patch_numpy_umath` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    patch_numpy_umath()


def restore():
    """
    Deprecated alias for restore_numpy_umath.

    See restore_numpy_umath for details and examples.
    """
    warnings.warn(
        "restore is deprecated since mkl_umath 0.4.0 and will be "
        "removed in a future release. Use `restore_numpy_umath` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    restore_numpy_umath()


def is_patched():
    """
    Return True if NumPy umath loops have been patched by mkl_umath.

    Examples
    --------
    >>> import mkl_umath
    >>> mkl_umath.is_patched()
    # False

    >>> mkl_umath.patch_numpy_umath()  # Enable mkl_umath in Numpy
    >>> mkl_umath.is_patched()
    # True

    >>> mkl_umath.restore_numpy_umath()  # Disable mkl_umath in Numpy
    >>> mkl_umath.is_patched()
    # False
    """
    return _patch.is_patched()


class mkl_umath(ContextDecorator):
    """
    Context manager and decorator to temporarily patch NumPy ufuncs
    with MKL-based implementations.

    ⚠️ Warning
    -------------------------
    If used in a multi-threaded program, ALL concurrent threads executing NumPy
    operations must either have applied the patch prior to execution, or run
    entirely within the `mkl_umath` context manager. Executing standard NumPy
    calls in one thread while another thread is actively patching or unpatching
    will lead to undefined behavior at best, and segmentation faults at worst.
    For this reason, it is recommended to prefer the `mkl_umath` context
    manager.

    Examples
    --------
    >>> import mkl_umath
    >>> mkl_umath.is_patched()
    # False

    >>> with mkl_umath.mkl_umath():  # Enable mkl_umath in Numpy
    >>>     print(mkl_umath.is_patched())
    # True

    >>> mkl_umath.is_patched()
    # False
    """
    def __enter__(self):
        patch_numpy_umath()
        return self

    def __exit__(self, *exc):
        restore_numpy_umath()
        return False
