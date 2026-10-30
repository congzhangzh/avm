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

#include <arm_neon.h>
#include <assert.h>

#include "config/av2_rtcd.h"

#include "av2/common/convolve.h"

// ROUND_POWER_OF_TWO_SIGNED(v, n), lane by lane.
static inline int32x4_t round_signed(int32x4_t v, int32x4_t bias,
                                     int32x4_t neg_n) {
  const int32x4_t mag = vshlq_s32(vaddq_s32(vabsq_s32(v), bias), neg_n);
  return vbslq_s32(vcltq_s32(v, vdupq_n_s32(0)), vnegq_s32(mag), mag);
}

// The symmetric non-separable filter of av2_convolve_symmetric_highbd_c()
// (subtract_center == 0) and av2_convolve_symmetric_subtract_center_highbd_c()
// (subtract_center == 1), eight pixels at a time; returns the number of
// columns done, the rest is left to the C versions. The tap pairs are summed
// in 16 bits as the C does.
static inline int convolve_symmetric_neon(
    const uint16_t *dgd, int stride, const NonsepFilterConfig *filter_config,
    const int16_t *filter, uint16_t *dst, int dst_stride, int bit_depth,
    int block_row_begin, int block_row_end, int block_col_begin,
    int block_col_end, int subtract_center) {
  const int num_sym_taps = filter_config->num_pixels / 2;
  assert(num_sym_taps <= 24);
  int32_t singleton_tap = 1 << filter_config->prec_bits;
  int32_t dc_offset = 0;
  if (filter_config->num_pixels % 2) {
    const int16_t center =
        filter[filter_config->config[filter_config->num_pixels - 1]
                                    [NONSEP_BUF_POS]];
    if (subtract_center)
      dc_offset = center;
    else
      singleton_tap += center;
  }
  int offsets[24];
  int16_t taps[24];
  for (int k = 0; k < num_sym_taps; ++k) {
    offsets[k] = filter_config->config[2 * k][NONSEP_ROW_ID] * stride +
                 filter_config->config[2 * k][NONSEP_COL_ID];
    taps[k] = filter[filter_config->config[2 * k][NONSEP_BUF_POS]];
  }
  const int n = filter_config->prec_bits;
  const int32x4_t bias = vdupq_n_s32((1 << n) >> 1);
  const int32x4_t neg_n = vdupq_n_s32(-n);
  const int32x4_t base = vdupq_n_s32(dc_offset);
  const int32x4_t pix_max = vdupq_n_s32((1 << bit_depth) - 1);
  const int32x4_t zero = vdupq_n_s32(0);
  const int width8 = (block_col_end - block_col_begin) & ~7;

  for (int r = block_row_begin; r < block_row_end; ++r) {
    for (int c = block_col_begin; c < block_col_begin + width8; c += 8) {
      const int16_t *src = (const int16_t *)dgd + r * stride + c;
      const int16x8_t center = vld1q_s16(src);
      int32x4_t lo = vmlaq_n_s32(base, vmovl_s16(vget_low_s16(center)),
                                 singleton_tap);
      int32x4_t hi = vmlaq_n_s32(base, vmovl_s16(vget_high_s16(center)),
                                 singleton_tap);
      for (int k = 0; k < num_sym_taps; ++k) {
        int16x8_t a = vld1q_s16(src - offsets[k]);
        int16x8_t b = vld1q_s16(src + offsets[k]);
        if (subtract_center) {
          a = vsubq_s16(a, center);
          b = vsubq_s16(b, center);
        }
        const int16x8_t sum = vaddq_s16(a, b);
        lo = vmlal_n_s16(lo, vget_low_s16(sum), taps[k]);
        hi = vmlal_n_s16(hi, vget_high_s16(sum), taps[k]);
      }
      lo = vminq_s32(vmaxq_s32(round_signed(lo, bias, neg_n), zero), pix_max);
      hi = vminq_s32(vmaxq_s32(round_signed(hi, bias, neg_n), zero), pix_max);
      vst1q_u16(dst + r * dst_stride + c,
                vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)));
    }
  }
  return width8;
}

void av2_convolve_symmetric_highbd_neon(const uint16_t *dgd, int stride,
                                        const NonsepFilterConfig *filter_config,
                                        const int16_t *filter, uint16_t *dst,
                                        int dst_stride, int bit_depth,
                                        int block_row_begin, int block_row_end,
                                        int block_col_begin,
                                        int block_col_end) {
  assert(!filter_config->subtract_center);
  const int done = convolve_symmetric_neon(
      dgd, stride, filter_config, filter, dst, dst_stride, bit_depth,
      block_row_begin, block_row_end, block_col_begin, block_col_end, 0);
  if (block_col_begin + done < block_col_end)
    av2_convolve_symmetric_highbd_c(dgd, stride, filter_config, filter, dst,
                                    dst_stride, bit_depth, block_row_begin,
                                    block_row_end, block_col_begin + done,
                                    block_col_end);
}

void av2_convolve_symmetric_subtract_center_highbd_neon(
    const uint16_t *dgd, int stride, const NonsepFilterConfig *filter_config,
    const int16_t *filter, uint16_t *dst, int dst_stride, int bit_depth,
    int block_row_begin, int block_row_end, int block_col_begin,
    int block_col_end) {
  assert(filter_config->subtract_center);
  const int done = convolve_symmetric_neon(
      dgd, stride, filter_config, filter, dst, dst_stride, bit_depth,
      block_row_begin, block_row_end, block_col_begin, block_col_end, 1);
  if (block_col_begin + done < block_col_end)
    av2_convolve_symmetric_subtract_center_highbd_c(
        dgd, stride, filter_config, filter, dst, dst_stride, bit_depth,
        block_row_begin, block_row_end, block_col_begin + done, block_col_end);
}
