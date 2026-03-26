#ifndef MKL_UMATH_NPYV_H
#define MKL_UMATH_NPYV_H

/*
 * Simplified SIMD vectorization wrapper for mkl_umath
 * Using direct AVX2 intrinsics instead of NumPy's complex npyv layer
 *
 * This is a proof-of-concept focusing on FLOAT add operation only.
 */

#include "numpy/npy_common.h"
#include <immintrin.h>  // AVX2 intrinsics

/*
 * Check if AVX2 is available at compile time
 */
#ifdef __AVX2__
  #define NPYV_CAN_VECTORIZE_FLOAT 1
  #define NPYV_CAN_VECTORIZE_DOUBLE 1

  // AVX2 vector lanes
  #define npyv_nlanes_f32 8   // 256 bits / 32 bits
  #define npyv_nlanes_f64 4   // 256 bits / 64 bits
  #define npyv_nlanes_u8 32   // 256 bits / 8 bits

  // Type definitions
  typedef __m256 npyv_f32;
  typedef __m256d npyv_f64;

  // Load operations
  #define npyv_load_f32(ptr) _mm256_loadu_ps((const float*)(ptr))
  #define npyv_load_f64(ptr) _mm256_loadu_pd((const double*)(ptr))

  // Store operations
  #define npyv_store_f32(ptr, vec) _mm256_storeu_ps((float*)(ptr), (vec))
  #define npyv_store_f64(ptr, vec) _mm256_storeu_pd((double*)(ptr), (vec))

  // Arithmetic operations
  #define npyv_add_f32(a, b) _mm256_add_ps((a), (b))
  #define npyv_add_f64(a, b) _mm256_add_pd((a), (b))

  // Set all lanes to same value
  #define npyv_setall_f32(val) _mm256_set1_ps(val)
  #define npyv_setall_f64(val) _mm256_set1_pd(val)

  // Conditional load/store (simplified - just use regular ops for prototype)
  static inline npyv_f32 npyv_load_tillz_f32(const float *ptr, npy_intp len) {
      if (len >= 8) return npyv_load_f32(ptr);
      // For partial loads, create mask and zero out unused elements
      float temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (npy_intp i = 0; i < len && i < 8; i++) temp[i] = ptr[i];
      return _mm256_loadu_ps(temp);
  }

  static inline npyv_f64 npyv_load_tillz_f64(const double *ptr, npy_intp len) {
      if (len >= 4) return npyv_load_f64(ptr);
      double temp[4] = {0, 0, 0, 0};
      for (npy_intp i = 0; i < len && i < 4; i++) temp[i] = ptr[i];
      return _mm256_loadu_pd(temp);
  }

  static inline void npyv_store_till_f32(float *ptr, npy_intp len, npyv_f32 vec) {
      if (len >= 8) {
          npyv_store_f32(ptr, vec);
      } else {
          float temp[8];
          npyv_store_f32(temp, vec);
          for (npy_intp i = 0; i < len && i < 8; i++) ptr[i] = temp[i];
      }
  }

  static inline void npyv_store_till_f64(double *ptr, npy_intp len, npyv_f64 vec) {
      if (len >= 4) {
          npyv_store_f64(ptr, vec);
      } else {
          double temp[4];
          npyv_store_f64(temp, vec);
          for (npy_intp i = 0; i < len && i < 4; i++) ptr[i] = temp[i];
      }
  }

  // Cleanup (no-op for AVX2)
  #define npyv_cleanup() do {} while(0)

#else
  // SIMD not available
  #define NPYV_CAN_VECTORIZE_FLOAT 0
  #define NPYV_CAN_VECTORIZE_DOUBLE 0
#endif

/*
 * Memory overlap detection (copied from NumPy)
 */
static inline int
is_mem_overlap(const void *src, npy_intp src_step,
               const void *dst, npy_intp dst_step, npy_intp len)
{
    const char *src_ptr = (const char *)src;
    const char *dst_ptr = (const char *)dst;

    if (src_ptr == dst_ptr) {
        return 0; // Same pointer, always safe
    }

    npy_intp src_size = len * src_step;
    npy_intp dst_size = len * dst_step;

    // Check if ranges overlap
    return !((src_ptr + src_size <= dst_ptr) || (dst_ptr + dst_size <= src_ptr));
}

#endif /* MKL_UMATH_NPYV_H */
