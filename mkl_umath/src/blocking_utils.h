/*
 * Copyright (c) 2019, Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Intel Corporation nor the names of its contributors
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "numpy/arrayobject.h"

/* Adapated from NumPy's source code. 
 * https://github.com/numpy/numpy/blob/main/LICENSE.txt */

/*
 * Return number of elements that must be peeled from the start of 'addr' with
 * 'nvals' elements of size 'esize' in order to reach blockable alignment.
 * The required alignment in bytes is passed as the 'alignment' argument and
 * must be a power of two. This function is used to prepare an array for
 * blocking. See the 'npy_blocked_end' function documentation below for an
 * example of how this function is used.
 */
static NPY_INLINE npy_intp
npy_aligned_block_offset(const void * addr, const npy_uintp esize,
                         const npy_uintp alignment, const npy_uintp nvals)
{
    npy_uintp offset, peel;

    offset = (npy_uintp)addr & (alignment - 1);
    peel = offset ? (alignment - offset) / esize : 0;
    peel = (peel <= nvals) ? peel : nvals;
    assert(peel <= NPY_MAX_INTP);
    return (npy_intp)peel;
}

/*
 * Return upper loop bound for an array of 'nvals' elements
 * of size 'esize' peeled by 'offset' elements and blocking to
 * a vector size of 'vsz' in bytes
 *
 * example usage:
 * npy_intp i;
 * double v[101];
 * npy_intp esize = sizeof(v[0]);
 * npy_intp peel = npy_aligned_block_offset(v, esize, 16, n);
 * // peel to alignment 16
 * for (i = 0; i < peel; i++)
 *   <scalar-op>
 * // simd vectorized operation
 * for (; i < npy_blocked_end(peel, esize, 16, n); i += 16 / esize)
 *   <blocked-op>
 * // handle scalar rest
 * for(; i < n; i++)
 *   <scalar-op>
 */
static NPY_INLINE npy_intp
npy_blocked_end(const npy_uintp peel, const npy_uintp esize,
                const npy_uintp vsz, const npy_uintp nvals)
{
    npy_uintp ndiff = nvals - peel;
    npy_uintp res = (ndiff - ndiff % (vsz / esize));

    assert(nvals >= peel);
    assert(res <= NPY_MAX_INTP);
    return (npy_intp)(res);
}

