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
#include <string.h>

#include "config/av2_rtcd.h"

#include "av2/common/gdf_block.h"

// CLIP(x, lo, hi) of restoration.h, lane by lane, including its behaviour
// when lo > hi.
static inline int32x4_t clip_s32(int32x4_t x, int32x4_t lo, int32x4_t hi) {
  const int32x4_t t = vbslq_s32(vcgtq_s32(x, hi), hi, x);
  return vbslq_s32(vcltq_s32(x, lo), lo, t);
}

// Looks up a 16-bit parameter per pixel: tbl holds the GDF_TRAIN_CLS_NUM
// values of one input (twice), idx the byte pairs {2c, 2c + 1} of each
// pixel's class c.
static inline int16x8_t lookup_s16(const int16_t *par, uint8x16_t idx) {
  const int16x4_t p = vld1_s16(par);
  const uint8x16_t tbl = vreinterpretq_u8_s16(vcombine_s16(p, p));
  return vreinterpretq_s16_u8(vqtbl1q_u8(tbl, idx));
}

// Multiplies the clipped input (it fits 16 bits: GDF_TEST_INP_PREC) by the
// weight of each pixel and adds it to the two accumulators.
static inline void mac(int32x4_t *acc, int16x8_t inp, int16x8_t w) {
  acc[0] = vmlal_s16(acc[0], vget_low_s16(inp), vget_low_s16(w));
  acc[1] = vmlal_s16(acc[1], vget_high_s16(inp), vget_high_s16(w));
}

// Same arithmetic as gdf_inference_unit_c(), eight pixels at a time. The
// last (width % 8) columns are left to the C version.
void gdf_inference_unit_neon(const int i_min, const int i_max, const int j_min,
                             const int j_max, const int qp_idx,
                             const uint16_t *rec_pnt, const int rec_stride,
                             uint16_t *const *gdf_lap_pnt,
                             const int gdf_lap_stride,
                             const uint32_t *gdf_cls_pnt,
                             const int gdf_cls_stride, int16_t *err_pnt,
                             const int err_stride, const int pxl_shift,
                             const int ref_dst_idx) {
  const int blk_height = i_max - i_min;
  const int blk_width = j_max - j_min;
  const int width8 = blk_width & ~7;
  if (width8 < blk_width) {
    uint16_t *tail_lap_pnt[GDF_NET_INP_GRD_NUM];
    for (int g = 0; g < GDF_NET_INP_GRD_NUM; g++)
      tail_lap_pnt[g] = gdf_lap_pnt[g] + width8;
    gdf_inference_unit_c(i_min, i_max, j_min + width8, j_max, qp_idx,
                         rec_pnt + width8, rec_stride, tail_lap_pnt,
                         gdf_lap_stride, gdf_cls_pnt + (width8 >> 1),
                         gdf_cls_stride, err_pnt + width8, err_stride,
                         pxl_shift, ref_dst_idx);
  }
  if (width8 == 0) return;

  const int is_intra = ref_dst_idx == 0 ? 1 : 0;
  const int gdf_frm_max =
      is_intra ? GDF_NET_LUT_IDX_INTRA_MAX : GDF_NET_LUT_IDX_INTER_MAX;
  const int gdf_idx_min = -(gdf_frm_max >> 1);
  const int gdf_idx_max = gdf_frm_max - 1 + gdf_idx_min;
  const int gdf_idx_scale = AVMMAX(-gdf_idx_min, gdf_idx_max);
  const int32_t gdf_shift =
      GDF_TEST_INP_PREC - GDF_TRAIN_INP_PREC + GDF_TRAIN_PAR_SCALE_LOG2;
  const int32_t gdf_shift_half = 1 << (gdf_shift - 1);

  const int16_t *alpha, *weight;
  const int32_t *bias;
  const int8_t *gdftable;
  if (is_intra) {
    alpha = gdf_intra_alpha_table[qp_idx];
    weight = gdf_intra_weight_table[qp_idx];
    bias = gdf_intra_bias_table[qp_idx];
    gdftable = gdf_intra_error_table[qp_idx];
  } else {
    const int inter_ref_dst_idx = ref_dst_idx - 1;
    alpha = gdf_inter_alpha_table[inter_ref_dst_idx][qp_idx];
    weight = gdf_inter_weight_table[inter_ref_dst_idx][qp_idx];
    bias = gdf_inter_bias_table[inter_ref_dst_idx][qp_idx];
    gdftable = gdf_inter_error_table[inter_ref_dst_idx][qp_idx];
  }
  int gdf_idx_offset[GDF_NET_LUT_IDX_NUM];
  for (int idx = 0; idx < GDF_NET_LUT_IDX_NUM; idx++) {
    gdf_idx_offset[idx] = 1;
    for (int r_idx = 0; r_idx < GDF_NET_LUT_IDX_NUM - 1 - idx; r_idx++)
      gdf_idx_offset[idx] *= gdf_frm_max;
  }
  const int w_stride = GDF_OPTS_INP_TOT * GDF_TRAIN_CLS_NUM;

  const int32x4_t shift_v = vdupq_n_s32(pxl_shift);
  const int32x4_t inp_lo = vdupq_n_s32(-(1 << (GDF_TEST_INP_PREC - 1)));
  const int32x4_t inp_hi = vdupq_n_s32((1 << (GDF_TEST_INP_PREC - 1)) - 1);
  const int32x4_t zero = vdupq_n_s32(0);
  const int32x4_t lut_hi = vdupq_n_s32(gdf_frm_max - 1);
  const int32x4_t neg_shift = vdupq_n_s32(-gdf_shift);
  const int32x4_t half = vdupq_n_s32(gdf_shift_half);
  const int32x4_t idx_min = vdupq_n_s32(gdf_idx_min);

  uint16_t *lap_pnt[GDF_NET_INP_GRD_NUM];
  memcpy(lap_pnt, gdf_lap_pnt, sizeof(lap_pnt));
  const uint32_t *cls_line = gdf_cls_pnt;
  const int16_t *rec_ptr = (const int16_t *)rec_pnt;

  for (int i = 0; i < blk_height; i++) {
    for (int j = 0; j < width8; j += 8) {
      // Class of each pixel (one per two columns): c0 c0 c1 c1 ... c3 c3,
      // and the byte indices that pick its 16- and 32-bit parameters.
      const uint16x4_t c4 = vmovn_u32(vld1q_u32(cls_line + (j >> 1)));
      const uint16x8_t c8 = vzip1q_u16(vcombine_u16(c4, c4),
                                       vcombine_u16(c4, c4));
      const uint8x16_t idx16 = vreinterpretq_u8_u16(
          vmlaq_n_u16(vdupq_n_u16(0x0100), c8, 0x0202));
      const uint8x16_t idx32_lo = vreinterpretq_u8_u32(vmlaq_n_u32(
          vdupq_n_u32(0x03020100), vmovl_u16(vget_low_u16(c8)), 0x04040404));
      const uint8x16_t idx32_hi = vreinterpretq_u8_u32(vmlaq_n_u32(
          vdupq_n_u32(0x03020100), vmovl_u16(vget_high_u16(c8)), 0x04040404));

      int32x4_t acc[GDF_NET_LUT_IDX_NUM][2];
      for (int idx = 0; idx < GDF_NET_LUT_IDX_NUM; idx++) {
        const uint8x16_t tbl = vreinterpretq_u8_s32(
            vld1q_s32(bias + GDF_TRAIN_CLS_NUM * idx));
        acc[idx][0] = vreinterpretq_s32_u8(vqtbl1q_u8(tbl, idx32_lo));
        acc[idx][1] = vreinterpretq_s32_u8(vqtbl1q_u8(tbl, idx32_hi));
      }

      const int16x8_t rec = vld1q_s16(rec_ptr + j);
      for (int k = 0; k < GDF_OPTS_INP_TOT; k++) {
        const int16x8_t a = lookup_s16(alpha + k * GDF_TRAIN_CLS_NUM, idx16);
        const int32x4_t a_lo = vmovl_s16(vget_low_s16(a));
        const int32x4_t a_hi = vmovl_s16(vget_high_s16(a));
        const int32x4_t na_lo = vnegq_s32(a_lo);
        const int32x4_t na_hi = vnegq_s32(a_hi);
        int32x4_t inp_l, inp_h;
        if (k < GDF_NET_INP_REC_NUM) {
          const int16x8_t fwd = vld1q_s16(
              rec_ptr + j + gdf_guided_sample_coordinates_fwd[k][0] * rec_stride +
              gdf_guided_sample_coordinates_fwd[k][1]);
          const int16x8_t bwd = vld1q_s16(
              rec_ptr + j + gdf_guided_sample_coordinates_bwd[k][0] * rec_stride +
              gdf_guided_sample_coordinates_bwd[k][1]);
          const int32x4_t f_lo = vshlq_s32(
              vsubl_s16(vget_low_s16(fwd), vget_low_s16(rec)), shift_v);
          const int32x4_t f_hi = vshlq_s32(
              vsubl_s16(vget_high_s16(fwd), vget_high_s16(rec)), shift_v);
          const int32x4_t b_lo = vshlq_s32(
              vsubl_s16(vget_low_s16(bwd), vget_low_s16(rec)), shift_v);
          const int32x4_t b_hi = vshlq_s32(
              vsubl_s16(vget_high_s16(bwd), vget_high_s16(rec)), shift_v);
          inp_l = vaddq_s32(clip_s32(f_lo, na_lo, a_lo),
                            clip_s32(b_lo, na_lo, a_lo));
          inp_h = vaddq_s32(clip_s32(f_hi, na_hi, a_hi),
                            clip_s32(b_hi, na_hi, a_hi));
        } else {
          const uint16x8_t lap = vld1q_u16(lap_pnt[k - GDF_NET_INP_REC_NUM] + j);
          const int32x4_t l_lo = vshrq_n_s32(
              vshlq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lap))),
                        shift_v),
              GDF_TRAIN_GRD_SHIFT);
          const int32x4_t l_hi = vshrq_n_s32(
              vshlq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lap))),
                        shift_v),
              GDF_TRAIN_GRD_SHIFT);
          inp_l = clip_s32(l_lo, na_lo, a_lo);
          inp_h = clip_s32(l_hi, na_hi, a_hi);
        }
        inp_l = vminq_s32(vmaxq_s32(inp_l, inp_lo), inp_hi);
        inp_h = vminq_s32(vmaxq_s32(inp_h, inp_lo), inp_hi);
        const int16x8_t inp = vcombine_s16(vmovn_s32(inp_l), vmovn_s32(inp_h));
        for (int idx = 0; idx < GDF_NET_LUT_IDX_NUM; idx++) {
          const int16x8_t w = lookup_s16(
              weight + k * GDF_TRAIN_CLS_NUM + w_stride * idx, idx16);
          mac(acc[idx], inp, w);
        }
      }

      // GDF_NORM_IDX(), then the offset into the error table.
      int32x4_t off[2] = { zero, zero };
      for (int idx = 0; idx < GDF_NET_LUT_IDX_NUM; idx++) {
        for (int h = 0; h < 2; h++) {
          const int32x4_t v = acc[idx][h];
          const int32x4_t mag = vshlq_s32(
              vmlaq_n_s32(half, vabsq_s32(v), gdf_idx_scale), neg_shift);
          const int32x4_t norm =
              vbslq_s32(vcgtq_s32(v, zero), mag, vnegq_s32(mag));
          const int32x4_t lut =
              vminq_s32(vmaxq_s32(vsubq_s32(norm, idx_min), zero), lut_hi);
          off[h] = vmlaq_n_s32(off[h], lut, gdf_idx_offset[idx]);
        }
      }
      int32_t o[8];
      vst1q_s32(o, off[0]);
      vst1q_s32(o + 4, off[1]);
      for (int p = 0; p < 8; p++) err_pnt[j + p] = (int16_t)gdftable[o[p]];
    }
    if (i & 1) {
      cls_line += gdf_cls_stride;
      for (int g = 0; g < GDF_NET_INP_GRD_NUM; g++) lap_pnt[g] += gdf_lap_stride;
    }
    rec_ptr += rec_stride;
    err_pnt += err_stride;
  }
}
