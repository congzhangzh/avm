/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include <assert.h>
#include <immintrin.h>

#include "config/avm_config.h"
#include "config/avm_dsp_rtcd.h"

#include "avm/avm_integer.h"
#include "avm_ports/mem.h"

// High bit depth 4-way SAD (one source block against four references) with
// 512-bit vectors: 32 pixels a vector, one row of a block 32 wide or more,
// two rows of a block 16 wide.
//
// Only the 4-way versions: in the encoder (EPYC 9645, realtime speed 9,
// 1440p) they took 1.2-1.3x fewer cycles than AVX2, while single SADs were
// up to 10 % slower than AVX2's. The references sit at any pixel offset, so
// almost every 64-byte load spans two cache lines; the 4-way versions load
// the source once for four references and come out ahead.
//
// |src - ref| of a pixel is at most 4095 (12 bits), so eight of them add up
// to 32760 and still fit a positive int16. Up to eight vectors of absolute
// differences are summed in 16-bit lanes, then _mm512_madd_epi16 against ones
// widens the pairs to 32 bits. The largest block (256x256 at 12 bits) sums
// to 268M, well inside an int32.

#define MAX_VECTORS_16BIT 8

// Two rows of 16 pixels in one vector.
static AVM_FORCE_INLINE __m512i load_16x2(const uint16_t *p, int stride) {
  const __m256i lo = _mm256_loadu_si256((const __m256i *)p);
  const __m256i hi = _mm256_loadu_si256((const __m256i *)(p + stride));
  return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

// Sum of |src - ref| over a w x h block against four references; w is 16 or
// a multiple of 32, and h is even when w is 16. Each source vector is loaded
// once for all four. Inlined with constant w and h, so the loops unroll.
static AVM_FORCE_INLINE void highbd_sad_x4d_avx512(
    const uint16_t *src, int src_stride, const uint16_t *const ref_array[],
    int ref_stride, uint32_t *sad_array, int w, int h, int shift) {
  const __m512i ones = _mm512_set1_epi16(1);
  const uint16_t *ref[4] = { ref_array[0], ref_array[1], ref_array[2],
                             ref_array[3] };
  __m512i sum[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(),
                     _mm512_setzero_si512(), _mm512_setzero_si512() };
  if (w == 16) {
    assert(h % 2 == 0);
    const int rows = AVMMIN(2 * MAX_VECTORS_16BIT, h);
    for (int y = 0; y < h; y += rows) {
      __m512i acc[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(),
                         _mm512_setzero_si512(), _mm512_setzero_si512() };
      for (int i = 0; i < rows; i += 2) {
        const __m512i s = load_16x2(src, src_stride);
        for (int k = 0; k < 4; ++k) {
          const __m512i d = _mm512_abs_epi16(
              _mm512_sub_epi16(s, load_16x2(ref[k], ref_stride)));
          acc[k] = _mm512_add_epi16(acc[k], d);
          ref[k] += 2 * ref_stride;
        }
        src += 2 * src_stride;
      }
      for (int k = 0; k < 4; ++k)
        sum[k] = _mm512_add_epi32(sum[k], _mm512_madd_epi16(acc[k], ones));
    }
  } else {
    assert(w % 32 == 0 && w / 32 <= MAX_VECTORS_16BIT);
    const int rows = AVMMIN(MAX_VECTORS_16BIT / (w / 32), h);
    for (int y = 0; y < h; y += rows) {
      __m512i acc[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(),
                         _mm512_setzero_si512(), _mm512_setzero_si512() };
      for (int i = 0; i < rows; ++i) {
        for (int x = 0; x < w; x += 32) {
          const __m512i s = _mm512_loadu_si512((const void *)(src + x));
          for (int k = 0; k < 4; ++k) {
            const __m512i r = _mm512_loadu_si512((const void *)(ref[k] + x));
            acc[k] = _mm512_add_epi16(acc[k],
                                      _mm512_abs_epi16(_mm512_sub_epi16(s, r)));
          }
        }
        src += src_stride;
        for (int k = 0; k < 4; ++k) ref[k] += ref_stride;
      }
      for (int k = 0; k < 4; ++k)
        sum[k] = _mm512_add_epi32(sum[k], _mm512_madd_epi16(acc[k], ones));
    }
  }
  for (int k = 0; k < 4; ++k)
    sad_array[k] = (uint32_t)_mm512_reduce_add_epi32(sum[k]) << shift;
}

// The skip versions take every other row and double the sum, as the C
// versions do.
#define HIGHBD_SAD_X4D_AVX512(w, h)                                           \
  void avm_highbd_sad##w##x##h##x4d_avx512(                                   \
      const uint16_t *src, int src_stride, const uint16_t *const ref_array[], \
      int ref_stride, uint32_t *sad_array) {                                  \
    highbd_sad_x4d_avx512(src, src_stride, ref_array, ref_stride, sad_array,  \
                          w, h, 0);                                           \
  }                                                                           \
  void avm_highbd_sad_skip_##w##x##h##x4d_avx512(                             \
      const uint16_t *src, int src_stride, const uint16_t *const ref_array[], \
      int ref_stride, uint32_t *sad_array) {                                  \
    highbd_sad_x4d_avx512(src, 2 * src_stride, ref_array, 2 * ref_stride,     \
                          sad_array, w, h / 2, 1);                            \
  }

HIGHBD_SAD_X4D_AVX512(16, 4)
HIGHBD_SAD_X4D_AVX512(16, 8)
HIGHBD_SAD_X4D_AVX512(16, 16)
HIGHBD_SAD_X4D_AVX512(16, 32)
HIGHBD_SAD_X4D_AVX512(16, 64)
HIGHBD_SAD_X4D_AVX512(32, 4)
HIGHBD_SAD_X4D_AVX512(32, 8)
HIGHBD_SAD_X4D_AVX512(32, 16)
HIGHBD_SAD_X4D_AVX512(32, 32)
HIGHBD_SAD_X4D_AVX512(32, 64)
HIGHBD_SAD_X4D_AVX512(64, 4)
HIGHBD_SAD_X4D_AVX512(64, 8)
HIGHBD_SAD_X4D_AVX512(64, 16)
HIGHBD_SAD_X4D_AVX512(64, 32)
HIGHBD_SAD_X4D_AVX512(64, 64)
HIGHBD_SAD_X4D_AVX512(64, 128)
HIGHBD_SAD_X4D_AVX512(128, 64)
HIGHBD_SAD_X4D_AVX512(128, 128)
HIGHBD_SAD_X4D_AVX512(128, 256)
HIGHBD_SAD_X4D_AVX512(256, 128)
HIGHBD_SAD_X4D_AVX512(256, 256)
