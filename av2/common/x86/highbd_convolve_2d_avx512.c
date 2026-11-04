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
#include "config/av2_rtcd.h"

#include "avm_dsp/avm_dsp_common.h"
#include "avm_dsp/avm_filter.h"
#include "avm_ports/mem.h"
#include "av2/common/convolve.h"
#include "av2/common/filter.h"

// The x and y bodies use 64-byte aligned stack space, which makes their
// prologue long (stack realignment, saved registers). They are kept out of
// line, so the blocks that go straight to AVX2 do not pay for it.
#if defined(_MSC_VER)
#define CONV_NOINLINE __declspec(noinline)
#else
#define CONV_NOINLINE __attribute__((noinline))
#endif

// High bit depth single-reference sub-pixel convolutions (x, y and 2-D) with
// 512-bit vectors.
//
// Exact: every pixel goes through the same integer steps as in
// av2_highbd_convolve_*_sr_avx2(): the same _mm*_madd_epi16 coefficient
// pairs, 32-bit sums (wrap-around addition is associative, so the order of
// the adds does not matter), rounding constants and shifts, saturating packs
// and clip. Only the lane that carries a pixel differs, so the output is
// bit-identical to AVX2 (and so to C) for any input.
//
// The filters are bound by shuffles more than by multiplies, so the layouts
// are chosen to need few of them:
// - Width a multiple of 16: a vector holds 2 rows of 16 pixels, row k in the
//   low half and row k + 1 in the high half, the AVX2 layout (2 rows of 8)
//   twice as wide. Its vertical filter then gets the vector "one row down"
//   from a load, as AVX2 does: of the source for the y filter, of the 16-wide
//   intermediate buffer for the 2-D one.
// - Width 8 (2-D and x only; height a multiple of 4): a vector holds 4 rows
//   of 8 pixels, one in each 128-bit lane. The 2-D filter keeps its
//   intermediate rows in registers and shifts them a row at a time with
//   _mm512_alignr_epi64.
// Other blocks (2xN, 4xN, Nx2, 8-wide y) go to the AVX2 versions.
//
// Source reads stay what AVX2 reads: 256-bit loads of 16 pixels a row, of the
// same rows and columns (none past them). 512-bit loads of reference pixels
// at arbitrary offsets almost always span two cache lines, which made 512-bit
// SAD slower than AVX2 in the encoder.

// The pair (f[k], f[k + 1]) in every 32-bit lane, the operand of
// _mm512_madd_epi16 for two adjacent taps.
static AVM_FORCE_INLINE __m512i coeff_pair(const int16_t *f, int k) {
  const uint32_t lo = (uint16_t)f[k];
  const uint32_t hi = (uint16_t)f[k + 1];
  return _mm512_set1_epi32((int32_t)(lo | (hi << 16)));
}

// The coefficient pairs of a tap-long filter, as prepare_coeffs(),
// prepare_coeffs_6t(), prepare_coeffs_4t() and prepare_coeffs_bilinear() in
// avm_dsp/x86/convolve_avx2.h pick them from the 8-tap kernel: the taps
// (8 - tap) / 2 .. (8 + tap) / 2 - 1, the others being zero.
static AVM_FORCE_INLINE void prepare_coeffs_avx512(
    const InterpFilterParams *filter_params, int subpel_qn, int tap,
    __m512i *coeffs) {
  const int16_t *f = av2_get_interp_filter_subpel_kernel(
      filter_params, subpel_qn & SUBPEL_MASK);
  const int first = (8 - tap) / 2;
  for (int m = 0; m < tap / 2; ++m) coeffs[m] = coeff_pair(f, first + 2 * m);
}

static AVM_FORCE_INLINE __m512i load_2x256(const uint16_t *p0,
                                           const uint16_t *p1) {
  return _mm512_inserti64x4(
      _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)p0)),
      _mm256_loadu_si256((const __m256i *)p1), 1);
}

// Source of the horizontal filter for 2 rows of 16 output pixels: lo = pixels
// 0-15 and hi = 8-23 of the row at p (low half) and of the next one (high
// half), so that each 128-bit lane of lo:hi holds the 16 pixels of 8 outputs.
// The next row is not read (zero) unless two is set.
static AVM_FORCE_INLINE void load_16x2(const uint16_t *p, ptrdiff_t stride,
                                       int two, __m512i *lo, __m512i *hi) {
  if (two) {
    *lo = load_2x256(p, p + stride);
    *hi = load_2x256(p + 8, p + stride + 8);
  } else {
    *lo = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)p));
    *hi = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(p + 8)));
    *lo = _mm512_inserti64x4(*lo, _mm256_setzero_si256(), 1);
    *hi = _mm512_inserti64x4(*hi, _mm256_setzero_si256(), 1);
  }
}

// Source of the horizontal filter for 4 rows of 8 output pixels: pixels 0-7
// (lo) and 8-15 (hi) of the rows 0..3 at p, row r in lane r. Rows at or past
// n are not read (zero).
static AVM_FORCE_INLINE void load_16x4(const uint16_t *p, ptrdiff_t stride,
                                       int n, __m512i *lo, __m512i *hi) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i r0 = _mm256_loadu_si256((const __m256i *)p);
  const __m256i r1 =
      n > 1 ? _mm256_loadu_si256((const __m256i *)(p + stride)) : zero;
  const __m256i r2 =
      n > 2 ? _mm256_loadu_si256((const __m256i *)(p + 2 * stride)) : zero;
  const __m256i r3 =
      n > 3 ? _mm256_loadu_si256((const __m256i *)(p + 3 * stride)) : zero;
  const __m512i r01 = _mm512_inserti64x4(_mm512_castsi256_si512(r0), r1, 1);
  const __m512i r23 = _mm512_inserti64x4(_mm512_castsi256_si512(r2), r3, 1);
  *lo = _mm512_shuffle_i64x2(r01, r23, 0x88);
  *hi = _mm512_shuffle_i64x2(r01, r23, 0xdd);
}

// Horizontal filter sums of the even (0, 2, 4, 6) and odd (1, 3, 5, 7) output
// pixels of each 128-bit lane, from the 16 pixels of the lane in lo:hi, as
// CONVOLVE_HORIZ_FILTER_* in avm_dsp/x86/convolve_avx2.h.
static AVM_FORCE_INLINE void hfilter_lanes(__m512i lo, __m512i hi, int tap,
                                           const __m512i *c, __m512i *even,
                                           __m512i *odd) {
  if (tap == 8) {
    *even = _mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_madd_epi16(lo, c[0]),
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 4), c[1])),
        _mm512_add_epi32(
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 8), c[2]),
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 12), c[3])));
    *odd = _mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 2), c[0]),
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 6), c[1])),
        _mm512_add_epi32(
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 10), c[2]),
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 14), c[3])));
  } else if (tap == 6) {
    *even = _mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_madd_epi16(lo, c[0]),
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 4), c[1])),
        _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 8), c[2]));
    *odd = _mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 2), c[0]),
            _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 6), c[1])),
        _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 10), c[2]));
  } else if (tap == 4) {
    *even = _mm512_add_epi32(
        _mm512_madd_epi16(lo, c[0]),
        _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 4), c[1]));
    *odd = _mm512_add_epi32(
        _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 2), c[0]),
        _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 6), c[1]));
  } else {
    *even = _mm512_madd_epi16(lo, c[0]);
    *odd = _mm512_madd_epi16(_mm512_alignr_epi8(hi, lo, 2), c[0]);
  }
}

// Vertical filter sums of the 8 pixels of each 128-bit lane, from u[m] = the
// tap rows 2m and 2m + 1 interleaved: lo for pixels 0-3, hi for 4-7.
static AVM_FORCE_INLINE void vfilter_lanes(const __m512i *ulo,
                                           const __m512i *uhi, int tap,
                                           const __m512i *c, __m512i *sum_lo,
                                           __m512i *sum_hi) {
  __m512i a = _mm512_madd_epi16(ulo[0], c[0]);
  __m512i b = _mm512_madd_epi16(uhi[0], c[0]);
  for (int m = 1; m < tap / 2; ++m) {
    a = _mm512_add_epi32(a, _mm512_madd_epi16(ulo[m], c[m]));
    b = _mm512_add_epi32(b, _mm512_madd_epi16(uhi[m], c[m]));
  }
  *sum_lo = a;
  *sum_hi = b;
}

// Back to 8 pixels in order in each lane, from packs(even, odd).
static AVM_FORCE_INLINE __m512i interleave_even_odd(__m512i v) {
  const __m512i order = _mm512_broadcast_i32x4(
      _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15));
  return _mm512_shuffle_epi8(v, order);
}

// 32-bit sums to pixels: round, pack with signed saturation, clip to
// [0, clip], as the AVX2 vertical filters do.
static AVM_FORCE_INLINE __m512i round_pack_clip(__m512i lo, __m512i hi,
                                                __m512i round_const,
                                                __m128i round_shift,
                                                __m512i clip) {
  lo = _mm512_sra_epi32(_mm512_add_epi32(lo, round_const), round_shift);
  hi = _mm512_sra_epi32(_mm512_add_epi32(hi, round_const), round_shift);
  const __m512i res = _mm512_min_epi16(_mm512_packs_epi32(lo, hi), clip);
  return _mm512_max_epi16(res, _mm512_setzero_si512());
}

static AVM_FORCE_INLINE void store_16x2(uint16_t *d, ptrdiff_t stride,
                                        __m512i v) {
  _mm256_storeu_si256((__m256i *)d, _mm512_castsi512_si256(v));
  _mm256_storeu_si256((__m256i *)(d + stride), _mm512_extracti64x4_epi64(v, 1));
}

static AVM_FORCE_INLINE void store_8x4(uint16_t *d, ptrdiff_t stride,
                                       __m512i v) {
  _mm_storeu_si128((__m128i *)d, _mm512_castsi512_si128(v));
  _mm_storeu_si128((__m128i *)(d + stride), _mm512_extracti32x4_epi32(v, 1));
  _mm_storeu_si128((__m128i *)(d + 2 * stride),
                   _mm512_extracti32x4_epi32(v, 2));
  _mm_storeu_si128((__m128i *)(d + 3 * stride),
                   _mm512_extracti32x4_epi32(v, 3));
}

// The rows 1-4, 2-5 and 3-6 (k = 1, 2, 3) of a 4-row group a and the next b.
#define ROWS_FROM(b, a, k) _mm512_alignr_epi64((b), (a), 2 * (k))

// Rounding constants of the 2-D filter, as
// av2_highbd_convolve_2d_sr_specialized_avx2().
typedef struct {
  __m512i round_const_x, round_const_y, clip_pixel;
  __m128i round_shift_x, round_shift_y;
  __m512i coeffs_x[4], coeffs_y[4];
} Convolve2DConsts;

static AVM_FORCE_INLINE void init_2d_consts(
    Convolve2DConsts *k, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, const ConvolveParams *conv_params, int bd, int tap_x,
    int tap_y) {
  assert(bd + FILTER_BITS + 2 - conv_params->round_0 <= 16);
  k->round_const_x = _mm512_set1_epi32(((1 << conv_params->round_0) >> 1) +
                                       (1 << (bd + FILTER_BITS - 1)));
  k->round_shift_x = _mm_cvtsi32_si128(conv_params->round_0);
  k->round_const_y = _mm512_set1_epi32(
      ((1 << conv_params->round_1) >> 1) -
      (1 << (bd + 2 * FILTER_BITS - conv_params->round_0 - 1)));
  k->round_shift_y = _mm_cvtsi32_si128(conv_params->round_1);
  k->clip_pixel = _mm512_set1_epi16((1 << bd) - 1);
  prepare_coeffs_avx512(filter_params_x, subpel_x_qn, tap_x, k->coeffs_x);
  prepare_coeffs_avx512(filter_params_y, subpel_y_qn, tap_y, k->coeffs_y);
}

// Horizontal pass of the 2-D filter on prepared lo:hi: 16-bit intermediate
// pixels, in order, as AVX2 stores them in im_block.
static AVM_FORCE_INLINE __m512i hfilter_2d(__m512i lo, __m512i hi, int tap,
                                           const Convolve2DConsts *k) {
  __m512i even, odd;
  hfilter_lanes(lo, hi, tap, k->coeffs_x, &even, &odd);
  even = _mm512_sra_epi32(_mm512_add_epi32(even, k->round_const_x),
                          k->round_shift_x);
  odd = _mm512_sra_epi32(_mm512_add_epi32(odd, k->round_const_x),
                         k->round_shift_x);
  return interleave_even_odd(_mm512_packs_epi32(even, odd));
}

// 2-D, width a multiple of 16, height even. Strips of 16 columns: the
// horizontal pass writes the im_h intermediate rows of the strip to im_block
// (16 pixels a row, 2 rows a vector); the vertical pass loads "rows k, k + 1"
// at any k from it.
static AVM_FORCE_INLINE void convolve_2d_sr_w16(const uint16_t *src_ptr,
                                                ptrdiff_t stride, uint16_t *dst,
                                                int dst_stride, int w, int h,
                                                int tap_x, int tap_y,
                                                const Convolve2DConsts *k) {
  DECLARE_ALIGNED(64, int16_t, im_block[(MAX_SB_SIZE + MAX_FILTER_TAP) * 16]);
  const int im_h = h + tap_y - 1;
  const int pairs = tap_y / 2;
  for (int j = 0; j < w; j += 16) {
    for (int i = 0; i < im_h; i += 2) {
      __m512i lo, hi;
      load_16x2(src_ptr + i * stride + j, stride, i + 1 < im_h, &lo, &hi);
      _mm512_store_si512((__m512i *)&im_block[i * 16],
                         hfilter_2d(lo, hi, tap_x, k));
    }
    __m512i ulo[4], uhi[4];
    for (int m = 0; m < pairs - 1; ++m) {
      const __m512i r0 =
          _mm512_load_si512((const __m512i *)&im_block[2 * m * 16]);
      const __m512i r1 =
          _mm512_loadu_si512((const __m512i *)&im_block[(2 * m + 1) * 16]);
      ulo[m] = _mm512_unpacklo_epi16(r0, r1);
      uhi[m] = _mm512_unpackhi_epi16(r0, r1);
    }
    for (int i = 0; i < h; i += 2) {
      const int16_t *im = &im_block[(i + 2 * (pairs - 1)) * 16];
      const __m512i r0 = _mm512_load_si512((const __m512i *)im);
      const __m512i r1 = _mm512_loadu_si512((const __m512i *)(im + 16));
      ulo[pairs - 1] = _mm512_unpacklo_epi16(r0, r1);
      uhi[pairs - 1] = _mm512_unpackhi_epi16(r0, r1);
      __m512i sum_lo, sum_hi;
      vfilter_lanes(ulo, uhi, tap_y, k->coeffs_y, &sum_lo, &sum_hi);
      store_16x2(dst + i * dst_stride + j, dst_stride,
                 round_pack_clip(sum_lo, sum_hi, k->round_const_y,
                                 k->round_shift_y, k->clip_pixel));
      for (int m = 0; m < pairs - 1; ++m) {
        ulo[m] = ulo[m + 1];
        uhi[m] = uhi[m + 1];
      }
    }
  }
}

// Intermediate rows 4g..4g+3 of the 2-D filter at p = the first of them, of
// which n exist (the others are not read).
static AVM_FORCE_INLINE __m512i hgroup_2d(const uint16_t *p, ptrdiff_t stride,
                                          int n, int tap,
                                          const Convolve2DConsts *k) {
  __m512i lo, hi;
  load_16x4(p, stride, n, &lo, &hi);
  return hfilter_2d(lo, hi, tap, k);
}

// 2-D, width 8, height a multiple of 4. The intermediate rows stay in
// registers: a0, a1 = rows 4p..4p+3 and 4p+4..4p+7 for output rows 4p..4p+3,
// where output row 4p + r takes its tap rows from lane r on.
static AVM_FORCE_INLINE void convolve_2d_sr_w8(const uint16_t *s,
                                               ptrdiff_t stride, uint16_t *d,
                                               int dst_stride, int h, int tap_x,
                                               int tap_y,
                                               const Convolve2DConsts *k) {
  const int im_h = h + tap_y - 1;
#define HGROUP(g) \
  hgroup_2d(s + (g) * 4 * stride, stride, im_h - (g) * 4, tap_x, k)
#define VOUT(i)                                                    \
  do {                                                             \
    __m512i sum_lo, sum_hi;                                        \
    vfilter_lanes(ulo, uhi, tap_y, k->coeffs_y, &sum_lo, &sum_hi); \
    store_8x4(d + (i) * dst_stride, dst_stride,                    \
              round_pack_clip(sum_lo, sum_hi, k->round_const_y,    \
                              k->round_shift_y, k->clip_pixel));   \
  } while (0)

  __m512i ulo[4], uhi[4];
  __m512i a0 = HGROUP(0);
  __m512i a1 = HGROUP(1);
  if (tap_y == 8 || tap_y == 6) {
    // u[0], u[1]: tap rows 0-1 and 2-3; each output group then needs the
    // rows 4-7 (8-tap) or 4-5 (6-tap) from the next group a2.
    ulo[0] = _mm512_unpacklo_epi16(a0, ROWS_FROM(a1, a0, 1));
    uhi[0] = _mm512_unpackhi_epi16(a0, ROWS_FROM(a1, a0, 1));
    ulo[1] = _mm512_unpacklo_epi16(ROWS_FROM(a1, a0, 2), ROWS_FROM(a1, a0, 3));
    uhi[1] = _mm512_unpackhi_epi16(ROWS_FROM(a1, a0, 2), ROWS_FROM(a1, a0, 3));
    for (int i = 0; i < h; i += 4) {
      const __m512i a2 = HGROUP(i / 4 + 2);
      const __m512i b1 = ROWS_FROM(a2, a1, 1);
      const __m512i b2 = ROWS_FROM(a2, a1, 2);
      const __m512i b3 = ROWS_FROM(a2, a1, 3);
      ulo[2] = _mm512_unpacklo_epi16(a1, b1);
      uhi[2] = _mm512_unpackhi_epi16(a1, b1);
      if (tap_y == 8) {
        ulo[3] = _mm512_unpacklo_epi16(b2, b3);
        uhi[3] = _mm512_unpackhi_epi16(b2, b3);
        VOUT(i);
        ulo[0] = ulo[2];
        uhi[0] = uhi[2];
        ulo[1] = ulo[3];
        uhi[1] = uhi[3];
      } else {
        VOUT(i);
        ulo[0] = ulo[2];
        uhi[0] = uhi[2];
        ulo[1] = _mm512_unpacklo_epi16(b2, b3);
        uhi[1] = _mm512_unpackhi_epi16(b2, b3);
      }
      a1 = a2;
    }
  } else {
    for (int i = 0; i < h; i += 4) {
      if (i) a1 = HGROUP(i / 4 + 1);
      const __m512i b1 = ROWS_FROM(a1, a0, 1);
      ulo[0] = _mm512_unpacklo_epi16(a0, b1);
      uhi[0] = _mm512_unpackhi_epi16(a0, b1);
      if (tap_y == 4) {
        const __m512i b2 = ROWS_FROM(a1, a0, 2);
        const __m512i b3 = ROWS_FROM(a1, a0, 3);
        ulo[1] = _mm512_unpacklo_epi16(b2, b3);
        uhi[1] = _mm512_unpackhi_epi16(b2, b3);
      }
      VOUT(i);
      a0 = a1;
    }
  }
#undef HGROUP
#undef VOUT
}

static AVM_FORCE_INLINE void highbd_convolve_2d_sr_avx512_core(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, ConvolveParams *conv_params, int bd, int tap_x,
    int tap_y) {
  const ptrdiff_t stride = src_stride;
  const uint16_t *const src_ptr =
      src - (tap_y / 2 - 1) * stride - (tap_x / 2 - 1);
  Convolve2DConsts k;
  init_2d_consts(&k, filter_params_x, filter_params_y, subpel_x_qn, subpel_y_qn,
                 conv_params, bd, tap_x, tap_y);
  if ((w & 15) == 0) {
    convolve_2d_sr_w16(src_ptr, stride, dst, dst_stride, w, h, tap_x, tap_y,
                       &k);
  } else {
    for (int j = 0; j < w; j += 8) {
      convolve_2d_sr_w8(src_ptr + j, stride, dst + j, dst_stride, h, tap_x,
                        tap_y, &k);
    }
  }
}

// One copy for each pair of filter lengths, so the tap branches fold away.
#define CONVOLVE_2D_SR_TAPS(tx, ty)                                          \
  static void highbd_convolve_2d_sr_avx512_##tx##_##ty(                      \
      const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride,    \
      int w, int h, const InterpFilterParams *filter_params_x,               \
      const InterpFilterParams *filter_params_y, const int subpel_x_qn,      \
      const int subpel_y_qn, ConvolveParams *conv_params, int bd) {          \
    highbd_convolve_2d_sr_avx512_core(                                       \
        src, src_stride, dst, dst_stride, w, h, filter_params_x,             \
        filter_params_y, subpel_x_qn, subpel_y_qn, conv_params, bd, tx, ty); \
  }

#define CONVOLVE_2D_SR_TAPS_Y(tx) \
  CONVOLVE_2D_SR_TAPS(tx, 2)      \
  CONVOLVE_2D_SR_TAPS(tx, 4)      \
  CONVOLVE_2D_SR_TAPS(tx, 6)      \
  CONVOLVE_2D_SR_TAPS(tx, 8)

CONVOLVE_2D_SR_TAPS_Y(2)
CONVOLVE_2D_SR_TAPS_Y(4)
CONVOLVE_2D_SR_TAPS_Y(6)
CONVOLVE_2D_SR_TAPS_Y(8)

typedef void (*highbd_convolve_2d_sr_fn)(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, ConvolveParams *conv_params, int bd);

#define TAPS_ROW(tx)                       \
  { highbd_convolve_2d_sr_avx512_##tx##_2, \
    highbd_convolve_2d_sr_avx512_##tx##_4, \
    highbd_convolve_2d_sr_avx512_##tx##_6, \
    highbd_convolve_2d_sr_avx512_##tx##_8 }

// [tap_x / 2 - 1][tap_y / 2 - 1]
static const highbd_convolve_2d_sr_fn convolve_2d_sr_avx512_fns[4][4] = {
  TAPS_ROW(2), TAPS_ROW(4), TAPS_ROW(6), TAPS_ROW(8)
};

// The blocks the 512-bit layouts take: 16n x 2m, or 8 x 4m.
static INLINE int avx512_block(int w, int h) {
  return ((w & 15) == 0 && (h & 1) == 0) || (w == 8 && (h & 3) == 0);
}

void av2_highbd_convolve_2d_sr_avx512(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, ConvolveParams *conv_params, int bd) {
  if (!avx512_block(w, h) || filter_params_x->taps != SUBPEL_TAPS ||
      filter_params_y->taps != SUBPEL_TAPS) {
    av2_highbd_convolve_2d_sr_avx2(src, src_stride, dst, dst_stride, w, h,
                                   filter_params_x, filter_params_y,
                                   subpel_x_qn, subpel_y_qn, conv_params, bd);
    return;
  }
  // Both filters BILINEAR take a separate AVX2 function, which computes the
  // same integers as the 2-tap case here.
  const int tap_x = get_filter_tap(filter_params_x, subpel_x_qn);
  const int tap_y = get_filter_tap(filter_params_y, subpel_y_qn);
  convolve_2d_sr_avx512_fns[tap_x / 2 - 1][tap_y / 2 - 1](
      src, src_stride, dst, dst_stride, w, h, filter_params_x, filter_params_y,
      subpel_x_qn, subpel_y_qn, conv_params, bd);
}

// Constants of the x filter, as av2_highbd_convolve_x_sr_avx2().
typedef struct {
  __m512i round_const_x, round_const_bits, clip_pixel;
  __m128i round_shift_x, round_shift_bits;
  __m512i coeffs_x[4];
} ConvolveXConsts;

static AVM_FORCE_INLINE void init_x_consts(
    ConvolveXConsts *k, const InterpFilterParams *filter_params_x,
    const int subpel_x_qn, const ConvolveParams *conv_params, int bd) {
  assert(bd + FILTER_BITS + 2 - conv_params->round_0 <= 16);
  const int bits = FILTER_BITS - conv_params->round_0;
  assert(bits >= 0);
  k->round_const_x = _mm512_set1_epi32(((1 << conv_params->round_0) >> 1));
  k->round_shift_x = _mm_cvtsi32_si128(conv_params->round_0);
  k->round_const_bits = _mm512_set1_epi32((1 << bits) >> 1);
  k->round_shift_bits = _mm_cvtsi32_si128(bits);
  k->clip_pixel = _mm512_set1_epi16(bd == 10 ? 1023 : (bd == 12 ? 4095 : 255));
  prepare_coeffs_avx512(filter_params_x, subpel_x_qn, 8, k->coeffs_x);
}

static AVM_FORCE_INLINE __m512i x_filter(__m512i lo, __m512i hi,
                                         const ConvolveXConsts *k) {
  __m512i even, odd;
  hfilter_lanes(lo, hi, 8, k->coeffs_x, &even, &odd);
  even = _mm512_sra_epi32(_mm512_add_epi32(even, k->round_const_x),
                          k->round_shift_x);
  odd = _mm512_sra_epi32(_mm512_add_epi32(odd, k->round_const_x),
                         k->round_shift_x);
  even = _mm512_sra_epi32(_mm512_add_epi32(even, k->round_const_bits),
                          k->round_shift_bits);
  odd = _mm512_sra_epi32(_mm512_add_epi32(odd, k->round_const_bits),
                         k->round_shift_bits);
  const __m512i res = interleave_even_odd(_mm512_packs_epi32(even, odd));
  return _mm512_max_epi16(_mm512_min_epi16(res, k->clip_pixel),
                          _mm512_setzero_si512());
}

static CONV_NOINLINE void convolve_x_sr_avx512_body(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x, const int subpel_x_qn,
    ConvolveParams *conv_params, int bd) {
  if (!avx512_block(w, h) || filter_params_x->taps != SUBPEL_TAPS) {
    av2_highbd_convolve_x_sr_avx2(src, src_stride, dst, dst_stride, w, h,
                                  filter_params_x, subpel_x_qn, conv_params,
                                  bd);
    return;
  }
  // As av2_highbd_convolve_x_sr_avx2(): always the 8 taps.
  const ptrdiff_t stride = src_stride;
  const uint16_t *const src_ptr = src - (SUBPEL_TAPS / 2 - 1);
  ConvolveXConsts k;
  init_x_consts(&k, filter_params_x, subpel_x_qn, conv_params, bd);

  if ((w & 15) == 0) {
    for (int i = 0; i < h; i += 2) {
      const uint16_t *const s = src_ptr + i * stride;
      uint16_t *const d = dst + i * dst_stride;
      for (int j = 0; j < w; j += 16) {
        __m512i lo, hi;
        load_16x2(s + j, stride, 1, &lo, &hi);
        store_16x2(d + j, dst_stride, x_filter(lo, hi, &k));
      }
    }
  } else {
    for (int i = 0; i < h; i += 4) {
      __m512i lo, hi;
      load_16x4(src_ptr + i * stride, stride, 4, &lo, &hi);
      store_8x4(dst + i * dst_stride, dst_stride, x_filter(lo, hi, &k));
    }
  }
}

static CONV_NOINLINE void convolve_y_sr_avx512_body(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_y, const int subpel_y_qn,
    int bd) {
  if ((w & 15) || (h & 1) || filter_params_y->taps != SUBPEL_TAPS) {
    av2_highbd_convolve_y_sr_avx2(src, src_stride, dst, dst_stride, w, h,
                                  filter_params_y, subpel_y_qn, bd);
    return;
  }
  // As av2_highbd_convolve_y_sr_avx2(): always the 8 taps, the source rows
  // -3 .. h + 3. r[k] = rows k (low half) and k + 1 (high half), 16 pixels
  // each; u[m] interleaves r[i + 2m] and r[i + 2m + 1], the tap rows 2m and
  // 2m + 1 of output rows i (low half) and i + 1 (high half).
  const ptrdiff_t stride = src_stride;
  const uint16_t *const src_ptr = src - (SUBPEL_TAPS / 2 - 1) * stride;
  const __m128i round_shift_bits = _mm_cvtsi32_si128(FILTER_BITS);
  const __m512i round_const_bits = _mm512_set1_epi32((1 << FILTER_BITS) >> 1);
  const __m512i clip_pixel =
      _mm512_set1_epi16(bd == 10 ? 1023 : (bd == 12 ? 4095 : 255));
  __m512i coeffs_y[4];
  prepare_coeffs_avx512(filter_params_y, subpel_y_qn, 8, coeffs_y);

  for (int j = 0; j < w; j += 16) {
    const uint16_t *const s = src_ptr + j;
    __m512i ulo[4], uhi[4];
    for (int m = 0; m < 3; ++m) {
      const __m512i r0 =
          load_2x256(s + 2 * m * stride, s + (2 * m + 1) * stride);
      const __m512i r1 =
          load_2x256(s + (2 * m + 1) * stride, s + (2 * m + 2) * stride);
      ulo[m] = _mm512_unpacklo_epi16(r0, r1);
      uhi[m] = _mm512_unpackhi_epi16(r0, r1);
    }
    for (int i = 0; i < h; i += 2) {
      const uint16_t *const p = s + (i + 6) * stride;
      const __m512i r0 = load_2x256(p, p + stride);
      const __m512i r1 = load_2x256(p + stride, p + 2 * stride);
      ulo[3] = _mm512_unpacklo_epi16(r0, r1);
      uhi[3] = _mm512_unpackhi_epi16(r0, r1);
      __m512i sum_lo, sum_hi;
      vfilter_lanes(ulo, uhi, 8, coeffs_y, &sum_lo, &sum_hi);
      store_16x2(dst + i * dst_stride + j, dst_stride,
                 round_pack_clip(sum_lo, sum_hi, round_const_bits,
                                 round_shift_bits, clip_pixel));
      for (int m = 0; m < 3; ++m) {
        ulo[m] = ulo[m + 1];
        uhi[m] = uhi[m + 1];
      }
    }
  }
}

void av2_highbd_convolve_x_sr_avx512(const uint16_t *src, int src_stride,
                                     uint16_t *dst, int dst_stride, int w,
                                     int h,
                                     const InterpFilterParams *filter_params_x,
                                     const int subpel_x_qn,
                                     ConvolveParams *conv_params, int bd) {
  if (!avx512_block(w, h) || filter_params_x->taps != SUBPEL_TAPS) {
    av2_highbd_convolve_x_sr_avx2(src, src_stride, dst, dst_stride, w, h,
                                  filter_params_x, subpel_x_qn, conv_params,
                                  bd);
    return;
  }
  convolve_x_sr_avx512_body(src, src_stride, dst, dst_stride, w, h,
                            filter_params_x, subpel_x_qn, conv_params, bd);
}

void av2_highbd_convolve_y_sr_avx512(const uint16_t *src, int src_stride,
                                     uint16_t *dst, int dst_stride, int w,
                                     int h,
                                     const InterpFilterParams *filter_params_y,
                                     const int subpel_y_qn, int bd) {
  if ((w & 15) || (h & 1) || filter_params_y->taps != SUBPEL_TAPS) {
    av2_highbd_convolve_y_sr_avx2(src, src_stride, dst, dst_stride, w, h,
                                  filter_params_y, subpel_y_qn, bd);
    return;
  }
  convolve_y_sr_avx512_body(src, src_stride, dst, dst_stride, w, h,
                            filter_params_y, subpel_y_qn, bd);
}
