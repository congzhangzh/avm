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
#include "av2/common/av2_txfm.h"
#include "av2/common/common_data.h"
#include "av2/common/txb_common.h"
#include "av2/encoder/x86/highbd_fwd_txfm_avx2.h"
#include "avm/avm_integer.h"
#include "avm_ports/mem.h"

// Transposes the 16x16 block of 32-bit values in r[] (r[k] lane l = element
// (k, l)) and stores row l of the result, { r[0][l] .. r[15][l] }, at
// dst + l * stride.
static AVM_FORCE_INLINE void transpose_store_16x16(const __m512i *r, int *dst,
                                                   int stride) {
  __m512i t[16], u[16];
  for (int k = 0; k < 16; k += 2) {
    // 128-bit lane L: { r[k][4L], r[k+1][4L], r[k][4L+1], r[k+1][4L+1] } and
    // the same for 4L+2, 4L+3.
    t[k] = _mm512_unpacklo_epi32(r[k], r[k + 1]);
    t[k + 1] = _mm512_unpackhi_epi32(r[k], r[k + 1]);
  }
  for (int q = 0; q < 16; q += 4) {
    // u[q + s] lane L: rows q .. q+3 of column 4L + s.
    u[q + 0] = _mm512_unpacklo_epi64(t[q + 0], t[q + 2]);
    u[q + 1] = _mm512_unpackhi_epi64(t[q + 0], t[q + 2]);
    u[q + 2] = _mm512_unpacklo_epi64(t[q + 1], t[q + 3]);
    u[q + 3] = _mm512_unpackhi_epi64(t[q + 1], t[q + 3]);
  }
  for (int s = 0; s < 4; s++) {
    // Column 4L + s is lane L of u[s], u[4 + s], u[8 + s] and u[12 + s]:
    // transpose those four vectors as 4x4 blocks of 128-bit lanes.
    const __m512i v0 = _mm512_shuffle_i32x4(u[s], u[4 + s], 0x44);
    const __m512i v1 = _mm512_shuffle_i32x4(u[s], u[4 + s], 0xee);
    const __m512i v2 = _mm512_shuffle_i32x4(u[8 + s], u[12 + s], 0x44);
    const __m512i v3 = _mm512_shuffle_i32x4(u[8 + s], u[12 + s], 0xee);
    _mm512_storeu_si512((__m512i *)(dst + (0 + s) * stride),
                        _mm512_shuffle_i32x4(v0, v2, 0x88));
    _mm512_storeu_si512((__m512i *)(dst + (4 + s) * stride),
                        _mm512_shuffle_i32x4(v0, v2, 0xdd));
    _mm512_storeu_si512((__m512i *)(dst + (8 + s) * stride),
                        _mm512_shuffle_i32x4(v1, v3, 0x88));
    _mm512_storeu_si512((__m512i *)(dst + (12 + s) * stride),
                        _mm512_shuffle_i32x4(v1, v3, 0xdd));
  }
}

// Input sample n of 16 consecutive lines: int16_t residuals or int values.
static AVM_FORCE_INLINE __m512i load_line16(const void *src, int is_int16,
                                            ptrdiff_t offset) {
  if (is_int16) {
    return _mm512_cvtepi16_epi32(
        _mm256_loadu_si256((const __m256i *)((const int16_t *)src + offset)));
  }
  return _mm512_loadu_si512((const __m512i *)((const int *)src + offset));
}

#define MUL_ACC(acc, x, coef) \
  acc = _mm512_add_epi32(_mm512_mullo_epi32(x, _mm512_set1_epi32(coef)), acc)

// The 64-point forward DCT-II of 16 lines (sample n of line l at
// src[n * stride + l]), keeping the first 32 outputs as
// fwd_txfm_dct2_size64_c() does when zero_line is set (always, for a
// 64-length transform): output k of line l goes to dst[l * dst_stride + k].
//
// Each output is the same sum of tx_mat products as in the C version, plus
// the rounding offset, all in wrapping 32-bit arithmetic, then shifted. The
// terms are added in another order than in C and AVX2, but addition and
// multiplication modulo 2^32 are associative and commutative, so every lane
// ends up with the same bits.
static AVM_FORCE_INLINE void fwd_dct64_16lines(const void *src, int is_int16,
                                               ptrdiff_t stride, int *dst,
                                               int dst_stride, int shift,
                                               __m512i v_offset) {
  const int *tx_mat = tx_kernel_dct2_size64[FWD_TXFM][0];
  __m512i a[32], c[16], e[8], g[4];
  __m512i out[32];
  __m512i acc[16];

  // Odd outputs 1, 3, .., 31 from b[n] = x[n] - x[63 - n]; b[n] is used as
  // soon as it is made, so only the 16 sums stay in registers.
  for (int k = 0; k < 16; k++) acc[k] = v_offset;
  for (int n = 0; n < 32; n++) {
    const __m512i x0 = load_line16(src, is_int16, n * stride);
    const __m512i x1 = load_line16(src, is_int16, (63 - n) * stride);
    const __m512i b = _mm512_sub_epi32(x0, x1);
    a[n] = _mm512_add_epi32(x0, x1);
    MUL_ACC(acc[0], b, tx_mat[1 * 64 + n]);
    MUL_ACC(acc[1], b, tx_mat[3 * 64 + n]);
    MUL_ACC(acc[2], b, tx_mat[5 * 64 + n]);
    MUL_ACC(acc[3], b, tx_mat[7 * 64 + n]);
    MUL_ACC(acc[4], b, tx_mat[9 * 64 + n]);
    MUL_ACC(acc[5], b, tx_mat[11 * 64 + n]);
    MUL_ACC(acc[6], b, tx_mat[13 * 64 + n]);
    MUL_ACC(acc[7], b, tx_mat[15 * 64 + n]);
    MUL_ACC(acc[8], b, tx_mat[17 * 64 + n]);
    MUL_ACC(acc[9], b, tx_mat[19 * 64 + n]);
    MUL_ACC(acc[10], b, tx_mat[21 * 64 + n]);
    MUL_ACC(acc[11], b, tx_mat[23 * 64 + n]);
    MUL_ACC(acc[12], b, tx_mat[25 * 64 + n]);
    MUL_ACC(acc[13], b, tx_mat[27 * 64 + n]);
    MUL_ACC(acc[14], b, tx_mat[29 * 64 + n]);
    MUL_ACC(acc[15], b, tx_mat[31 * 64 + n]);
  }
  for (int k = 0; k < 16; k++)
    out[2 * k + 1] = _mm512_srai_epi32(acc[k], shift);

  // Outputs 2, 6, .., 30 from d[n] = a[n] - a[31 - n].
  for (int k = 0; k < 8; k++) acc[k] = v_offset;
  for (int n = 0; n < 16; n++) {
    const __m512i d = _mm512_sub_epi32(a[n], a[31 - n]);
    c[n] = _mm512_add_epi32(a[n], a[31 - n]);
    MUL_ACC(acc[0], d, tx_mat[2 * 64 + n]);
    MUL_ACC(acc[1], d, tx_mat[6 * 64 + n]);
    MUL_ACC(acc[2], d, tx_mat[10 * 64 + n]);
    MUL_ACC(acc[3], d, tx_mat[14 * 64 + n]);
    MUL_ACC(acc[4], d, tx_mat[18 * 64 + n]);
    MUL_ACC(acc[5], d, tx_mat[22 * 64 + n]);
    MUL_ACC(acc[6], d, tx_mat[26 * 64 + n]);
    MUL_ACC(acc[7], d, tx_mat[30 * 64 + n]);
  }
  for (int k = 0; k < 8; k++) out[4 * k + 2] = _mm512_srai_epi32(acc[k], shift);

  // Outputs 4, 12, 20, 28 from f[n] = c[n] - c[15 - n].
  for (int k = 0; k < 4; k++) acc[k] = v_offset;
  for (int n = 0; n < 8; n++) {
    const __m512i f = _mm512_sub_epi32(c[n], c[15 - n]);
    e[n] = _mm512_add_epi32(c[n], c[15 - n]);
    MUL_ACC(acc[0], f, tx_mat[4 * 64 + n]);
    MUL_ACC(acc[1], f, tx_mat[12 * 64 + n]);
    MUL_ACC(acc[2], f, tx_mat[20 * 64 + n]);
    MUL_ACC(acc[3], f, tx_mat[28 * 64 + n]);
  }
  for (int k = 0; k < 4; k++) out[8 * k + 4] = _mm512_srai_epi32(acc[k], shift);

  // Outputs 8 and 24 from h[n] = e[n] - e[7 - n].
  acc[0] = acc[1] = v_offset;
  for (int n = 0; n < 4; n++) {
    const __m512i h = _mm512_sub_epi32(e[n], e[7 - n]);
    g[n] = _mm512_add_epi32(e[n], e[7 - n]);
    MUL_ACC(acc[0], h, tx_mat[8 * 64 + n]);
    MUL_ACC(acc[1], h, tx_mat[24 * 64 + n]);
  }
  out[8] = _mm512_srai_epi32(acc[0], shift);
  out[24] = _mm512_srai_epi32(acc[1], shift);

  // Outputs 0 and 16.
  const __m512i i0 = _mm512_add_epi32(g[0], g[3]);
  const __m512i i1 = _mm512_add_epi32(g[1], g[2]);
  const __m512i u0 = _mm512_sub_epi32(g[0], g[3]);
  const __m512i u1 = _mm512_sub_epi32(g[1], g[2]);
  acc[0] = acc[1] = v_offset;
  MUL_ACC(acc[0], i0, tx_mat[0 * 64 + 0]);
  MUL_ACC(acc[0], i1, tx_mat[0 * 64 + 1]);
  MUL_ACC(acc[1], u0, tx_mat[16 * 64 + 0]);
  MUL_ACC(acc[1], u1, tx_mat[16 * 64 + 1]);
  out[0] = _mm512_srai_epi32(acc[0], shift);
  out[16] = _mm512_srai_epi32(acc[1], shift);

  transpose_store_16x16(out, dst, dst_stride);
  transpose_store_16x16(out + 16, dst + 16, dst_stride);
}

#undef MUL_ACC

// fwd_txfm_dct2_size64_c() with zero_line set, on lines lines (a multiple of
// 16) of src (int16_t when is_int16, else int), sample n of line l at
// src[n * stride + l]; line l is written to dst + l * dst_stride.
static AVM_FORCE_INLINE void fwd_dct64_avx512(const void *src, int is_int16,
                                              ptrdiff_t stride, int *dst,
                                              int dst_stride, int shift,
                                              int lines) {
  assert(lines % 16 == 0);
  const __m512i v_offset = _mm512_set1_epi32(shift > 0 ? 1 << (shift - 1) : 0);
  for (int l = 0; l < lines; l += 16) {
    const void *s = is_int16 ? (const void *)((const int16_t *)src + l)
                             : (const void *)((const int *)src + l);
    fwd_dct64_16lines(s, is_int16, stride, dst + l * dst_stride, dst_stride,
                      shift, v_offset);
  }
}

// fwd_txfm_avx2() with AVX-512 64-point passes for the transforms that have
// one over 16 lines or more: 64x64, 64x32, 32x64, 64x16 and 16x64. The
// others, and the 64-point passes over 4 or 8 lines, take the AVX2 path.
//
// The output matches fwd_txfm_avx2() in the coefficients kept, the first
// min(w, 32) * min(h, 32); what it leaves after them differs, as it does
// between fwd_txfm_avx2() and fwd_txfm_c(), and nothing reads it:
// - a 64-point column pass reads the residuals itself, so the int copy of
//   them in coeff is made only when the column pass is a shorter one;
// - a 64-point row pass writes its 32 outputs per line at a stride of 32,
//   which is the packed layout fwd_txfm_avx2() gets by moving the rows after
//   it.
// buf is not cleared: each pass reads only lines 0 .. lines - skip_line - 1 of
// the first 32 (or tx size) samples, all of which the pass before writes.
#if defined(_MSC_VER)
#define FWD_TXFM_NOINLINE __declspec(noinline)
#else
#define FWD_TXFM_NOINLINE __attribute__((noinline))
#endif

// The 64-point sizes this file handles: 64x64, 64x32, 32x64, 64x16, 16x64.
static AVM_INLINE int is_dct64_size(int width, int height) {
  return (height == 64 && width >= 16) || (width == 64 && height >= 16);
}

// The work for the 64-point sizes, kept out of line: its 64-byte aligned
// stack buffer makes the prologue long (stack realignment, saved registers),
// which every other transform would otherwise pay on its way to AVX2.
static FWD_TXFM_NOINLINE void fwd_txfm_dct64_avx512(const int16_t *resi,
                                                    tran_low_t *coeff,
                                                    int diff_stride,
                                                    TxfmParam *txfm_param) {
  const TX_SIZE tx_size = txfm_param->tx_size;
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const int col64 = height == 64 && width >= 16;
  const int row64 = width == 64 && height >= 16;

  const TX_TYPE tx_type = txfm_param->tx_type;
  int tx_type_row = g_hor_tx_type[tx_type];
  int tx_type_col = g_ver_tx_type[tx_type];
  if ((col64 && tx_type_col != DCT2) || (row64 && tx_type_row != DCT2)) {
    fwd_txfm_avx2(resi, coeff, diff_stride, txfm_param);
    return;
  }

  if (txfm_param->use_ddt) {
    // The pass that is not 64 points long may be 16 points long.
    if (REPLACE_ADST16 && width == 16 &&
        (tx_type_row == DST7 || tx_type_row == DCT8)) {
      tx_type_row = (tx_type_row == DST7) ? DDTX : FDDT;
    }
    if (REPLACE_ADST16 && height == 16 &&
        (tx_type_col == DST7 || tx_type_col == DCT8)) {
      tx_type_col = (tx_type_col == DST7) ? DDTX : FDDT;
    }
  }

  const int skip_width = width > 32 ? width - 32 : 0;
  const int skip_height = height > 32 ? height - 32 : 0;
  const int shift_1st = fwd_tx_shift[tx_size][0];
  const int shift_2nd = fwd_tx_shift[tx_size][1];

  DECLARE_ALIGNED(64, int, buf[MAX_TX_SQUARE]);

  if (col64) {
    fwd_dct64_avx512(resi, 1, diff_stride, buf, 64, shift_1st, width);
  } else {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        coeff[(y * width) + x] = resi[(y * diff_stride) + x];
      }
    }
    fwd_transform_1d_avx2(coeff, buf, shift_1st, width, 0, skip_height,
                          tx_type_col, tx_size_high_log2[tx_size] - 2);
  }

  if (row64) {
    fwd_dct64_avx512(buf, 0, height, coeff, 32, shift_2nd,
                     height - skip_height);
  } else {
    fwd_transform_1d_avx2(buf, coeff, shift_2nd, height, skip_height,
                          skip_width, tx_type_row,
                          tx_size_wide_log2[tx_size] - 2);
  }

  // 64x32 and 32x64: scale by 1 / sqrt(2) as fwd_txfm_avx2() does.
  if ((tx_size_wide_log2[tx_size] + tx_size_high_log2[tx_size]) & 1) {
    const __m512i scale = _mm512_set1_epi64((int64_t)NewSqrt2);
    const __m512i round = _mm512_set1_epi64(1LL << (NewSqrt2Bits - 1));
    for (int i = 0; i < 1024; i += 8) {
      // round_shift() of the 64-bit product; only its low 32 bits are kept,
      // so a logical shift gives them as well as an arithmetic one.
      __m512i v = _mm512_cvtepi32_epi64(
          _mm256_loadu_si256((const __m256i *)(coeff + i)));
      v = _mm512_add_epi64(_mm512_mul_epi32(v, scale), round);
      v = _mm512_srli_epi64(v, NewSqrt2Bits);
      _mm256_storeu_si256((__m256i *)(coeff + i), _mm512_cvtepi64_epi32(v));
    }
  }
}

void fwd_txfm_avx512(const int16_t *resi, tran_low_t *coeff, int diff_stride,
                     TxfmParam *txfm_param) {
  const TX_SIZE tx_size = txfm_param->tx_size;
  if (txfm_param->lossless ||
      !is_dct64_size(tx_size_wide[tx_size], tx_size_high[tx_size])) {
    fwd_txfm_avx2(resi, coeff, diff_stride, txfm_param);
    return;
  }
  fwd_txfm_dct64_avx512(resi, coeff, diff_stride, txfm_param);
}
