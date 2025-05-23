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

/**
 * Macros to help build fast ufunc inner loops.
 *
 * These expect to have access to the arguments of a typical ufunc loop,
 *
 *     char **args
 *     npy_intp *dimensions
 *     npy_intp *steps
 */
#ifndef _NPY_UMATH_FAST_LOOP_MACROS_H_
#define _NPY_UMATH_FAST_LOOP_MACROS_H_

#if defined(__ICC) || defined(__INTEL_COMPILER)
#define NPY_PRAGMA_VECTOR _Pragma("vector")
#define NPY_PRAGMA_NOVECTOR _Pragma("novector")
#define NPY_ASSUME_ALIGNED(p, b) __assume_aligned((p), (b));
#elif defined(__clang__)
#define NPY_PRAGMA_VECTOR _Pragma("clang loop vectorize(enable)")
#define NPY_PRAGMA_NOVECTOR _Pragma("clang loop vectorize(disable)")
#define NPY_ASSUME_ALIGNED(p, b)
#else
#define NPY_PRAGMA_VECTOR _Pragma("GCC ivdep")
#define NPY_PRAGMA_NOVECTOR
#define NPY_ASSUME_ALIGNED(p, b) 
#endif

/* Adapated from NumPy's source code. 
 * https://github.com/numpy/numpy/blob/main/LICENSE.txt */

/**
 * Simple unoptimized loop macros that iterate over the ufunc arguments in
 * parallel.
 * @{
 */

/** (<ignored>) -> (op1) */
#define OUTPUT_LOOP\
    char *op1 = args[1];\
    npy_intp os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, op1 += os1)

/** (ip1) -> (op1) */
#define UNARY_LOOP\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; ++i, ip1 += is1, op1 += os1)

#define UNARY_LOOP_VECTORIZED(tin, tout)\
    tin *ip1 = (tin *) args[0];\
    tout *op1 = (tout *) args[1];		\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    NPY_PRAGMA_VECTOR\
    for(i = 0; i < n; ++i, ++ip1, ++op1)

#define UNARY_LOOP_DISPATCH(tin, tout, cond, body)\
    if (cond) {\
        UNARY_LOOP_VECTORIZED(tin, tout) { body; }\
    } else {\
        UNARY_LOOP { body; }\
    }

/** (ip1) -> (op1, op2) */
#define UNARY_LOOP_TWO_OUT\
    char *ip1 = args[0], *op1 = args[1], *op2 = args[2];\
    npy_intp is1 = steps[0], os1 = steps[1], os2 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; ++i, ip1 += is1, op1 += os1, op2 += os2)

/** (ip1, ip2) -> (op1) */
#define BINARY_LOOP\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; ++i, ip1 += is1, ip2 += is2, op1 += os1)

/** (ip1, ip2) -> (op1, op2) */
#define BINARY_LOOP_TWO_OUT\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2], *op2 = args[3];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2], os2 = steps[3];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; ++i, ip1 += is1, ip2 += is2, op1 += os1, op2 += os2)

/** (ip1, ip2, ip3) -> (op1) */
#define TERNARY_LOOP\
    char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];\
    npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; ++i, ip1 += is1, ip2 += is2, ip3 += is3, op1 += os1)

/** @} */

/* unary loop input and output contiguous */
#define IS_UNARY_CONT(tin, tout) (steps[0] == sizeof(tin) && \
                                  steps[1] == sizeof(tout))

#define IS_OUTPUT_CONT(tout) (steps[1] == sizeof(tout))

#define IS_BINARY_REDUCE ((args[0] == args[2])\
        && (steps[0] == steps[2])\
        && (steps[0] == 0))

/* binary loop input and output contiguous */
#define IS_BINARY_CONT(tin, tout) (steps[0] == sizeof(tin) && \
                                   steps[1] == sizeof(tin) && \
                                   steps[2] == sizeof(tout))

/* binary loop input and output contiguous with first scalar */
#define IS_BINARY_CONT_S1(tin, tout) (steps[0] == 0 && \
                                   steps[1] == sizeof(tin) && \
                                   steps[2] == sizeof(tout))

/* binary loop input and output contiguous with second scalar */
#define IS_BINARY_CONT_S2(tin, tout) (steps[0] == sizeof(tin) && \
                                   steps[1] == 0 && \
                                   steps[2] == sizeof(tout))

/*
 * loop with contiguous specialization
 * op should be the code working on `tin in` and
 * storing the result in `tout *out`
 * combine with NPY_GCC_OPT_3 to allow autovectorization
 * should only be used where its worthwhile to avoid code bloat
 */
#define BASE_UNARY_LOOP(tin, tout, op) \
    UNARY_LOOP { \
        const tin in = *(tin *)ip1; \
        tout *out = (tout *)op1; \
        op; \
    }

#define UNARY_LOOP_FAST(tin, tout, op)          \
    do { \
        /* condition allows compiler to optimize the generic macro */ \
        if (IS_UNARY_CONT(tin, tout)) { \
            if (args[0] == args[1]) { \
                BASE_UNARY_LOOP(tin, tout, op) \
            } \
            else { \
                BASE_UNARY_LOOP(tin, tout, op) \
            } \
        } \
        else { \
            BASE_UNARY_LOOP(tin, tout, op) \
        } \
    } \
    while (0)

/*
 * loop with contiguous specialization
 * op should be the code working on `tin in1`, `tin in2` and
 * storing the result in `tout *out`
 * combine with NPY_GCC_OPT_3 to allow autovectorization
 * should only be used where its worthwhile to avoid code bloat
 */
#define BASE_BINARY_LOOP(tin, tout, op) \
    BINARY_LOOP { \
        const tin in1 = *(tin *)ip1; \
        const tin in2 = *(tin *)ip2; \
        tout *out = (tout *)op1; \
        op; \
    }

/*
 * unfortunately gcc 6/7 regressed and we need to give it additional hints to
 * vectorize inplace operations (PR80198)
 * must only be used after op1 == ip1 or ip2 has been checked
 * TODO: using ivdep might allow other compilers to vectorize too
 */
#if __GNUC__ >= 6
#define IVDEP_LOOP _Pragma("GCC ivdep")
#else
#define IVDEP_LOOP
#endif
#define BASE_BINARY_LOOP_INP(tin, tout, op) \
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    IVDEP_LOOP \
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) { \
        const tin in1 = *(tin *)ip1; \
        const tin in2 = *(tin *)ip2; \
        tout *out = (tout *)op1; \
        op; \
    }

#define BASE_BINARY_LOOP_S(tin, tout, cin, cinp, vin, vinp, op) \
    const tin cin = *(tin *)cinp; \
    BINARY_LOOP { \
        const tin vin = *(tin *)vinp; \
        tout *out = (tout *)op1; \
        op; \
    }

/* PR80198 again, scalar works without the pragma */
#define BASE_BINARY_LOOP_S_INP(tin, tout, cin, cinp, vin, vinp, op) \
    const tin cin = *(tin *)cinp; \
    BINARY_LOOP { \
        const tin vin = *(tin *)vinp; \
        tout *out = (tout *)vinp; \
        op; \
    }

#define BINARY_LOOP_FAST(tin, tout, op)         \
    do { \
        /* condition allows compiler to optimize the generic macro */ \
        if (IS_BINARY_CONT(tin, tout)) { \
            if (abs_ptrdiff(args[2], args[0]) == 0 && \
                    abs_ptrdiff(args[2], args[1]) >= NPY_MAX_SIMD_SIZE) { \
                BASE_BINARY_LOOP_INP(tin, tout, op) \
            } \
            else if (abs_ptrdiff(args[2], args[1]) == 0 && \
                         abs_ptrdiff(args[2], args[0]) >= NPY_MAX_SIMD_SIZE) { \
                BASE_BINARY_LOOP_INP(tin, tout, op) \
            } \
            else { \
                BASE_BINARY_LOOP(tin, tout, op) \
            } \
        } \
        else if (IS_BINARY_CONT_S1(tin, tout)) { \
            if (abs_ptrdiff(args[2], args[1]) == 0) { \
                BASE_BINARY_LOOP_S_INP(tin, tout, in1, args[0], in2, ip2, op) \
            } \
            else { \
                BASE_BINARY_LOOP_S(tin, tout, in1, args[0], in2, ip2, op) \
            } \
        } \
        else if (IS_BINARY_CONT_S2(tin, tout)) { \
            if (abs_ptrdiff(args[2], args[0]) == 0) { \
                BASE_BINARY_LOOP_S_INP(tin, tout, in2, args[1], in1, ip1, op) \
            } \
            else { \
                BASE_BINARY_LOOP_S(tin, tout, in2, args[1], in1, ip1, op) \
            }\
        } \
        else { \
            BASE_BINARY_LOOP(tin, tout, op) \
        } \
    } \
    while (0)

#define BINARY_REDUCE_LOOP_INNER\
    char *ip2 = args[1]; \
    npy_intp is2 = steps[1]; \
    npy_intp n = dimensions[0]; \
    npy_intp i; \
    for(i = 0; i < n; i++, ip2 += is2)

#define BINARY_REDUCE_LOOP(TYPE)\
    char *iop1 = args[0]; \
    TYPE io1 = *(TYPE *)iop1; \
    BINARY_REDUCE_LOOP_INNER


#endif /* _NPY_UMATH_FAST_LOOP_MACROS_H_ */
