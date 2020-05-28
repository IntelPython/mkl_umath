#include "numpy/arrayobject.h"

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

