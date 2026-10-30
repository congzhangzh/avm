/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause
 * Clear License was not distributed with this source code in the LICENSE file,
 * you can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.
 * If the Alliance for Open Media Patent License 1.0 was not distributed with
 * this source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

// NEON versions of avm_highbd_lpf_{horizontal,vertical}_generic.
//
// Both kernels filter one 4-sample edge segment: 4 lines run across the edge
// (columns for a horizontal edge, rows for a vertical one) and each line holds
// up to 8 samples on each side. The work is done in a "lines in lanes" form:
// px[PX(k)] is a uint16x4_t holding sample position k (k = -8..7, negative =
// before the edge) of the 4 lines, one line per lane. A horizontal edge loads
// that form directly (one row per position); a vertical edge transposes 4 rows
// into it and back. The filter decision and the filter itself are then shared.
//
// Bit exactness with avm_dsp/loopfilter.c:
// - The decision values (second derivatives and end-direction differences)
//   are computed in 16 bits. Their true values fit in int16 for bit depths up
//   to 12 (the largest is |(s[-1] - s[-8]) - 7 * (s[-1] - s[-2])| <= 8 * 4095),
//   so the wrapping arithmetic gives the exact result. The two lines are
//   averaged with vrhadd, which is (a + b + 1) >> 1 without overflow.
// - The filter runs in 32 bits like the C code; vrshrq_n_s32(x, 11) is
//   ROUND_POWER_OF_TWO(x, 3 + DF_SHIFT), and vqmovun + vmin is
//   clip_pixel_highbd().
// - Like the SSE4.1 version, the filter returns early when every delta is
//   zero. The C version would then still clip each sample to the bit depth,
//   which changes nothing for in-range samples.

#include <arm_neon.h>
#include <assert.h>

#include "config/avm_config.h"
#include "config/avm_dsp_rtcd.h"

#include "avm_dsp/avm_dsp_common.h"
#include "avm_dsp/loopfilter.h"
#include "avm_ports/mem.h"

#define PX(k) ((k) + 8)

static INLINE int all_zero_s32(int32x4_t v) {
#if defined(__aarch64__)
  return vmaxvq_u32(vreinterpretq_u32_s32(v)) == 0;
#else
  const uint32x4_t u = vreinterpretq_u32_s32(v);
  const uint32x2_t r = vorr_u32(vget_low_u32(u), vget_high_u32(u));
  return (vget_lane_u32(r, 0) | vget_lane_u32(r, 1)) == 0;
#endif
}

// Transposes 4 rows of 8 samples. With columns c0..c7, returns
// c04 = (c0 | c4), c15 = (c1 | c5), c26 = (c2 | c6), c37 = (c3 | c7), where
// each half holds the 4 rows of one column.
static INLINE void transpose_rows_4x8(const uint16x8_t r[4], uint16x8_t *c04,
                                      uint16x8_t *c15, uint16x8_t *c26,
                                      uint16x8_t *c37) {
  const uint16x8x2_t t01 = vtrnq_u16(r[0], r[1]);
  const uint16x8x2_t t23 = vtrnq_u16(r[2], r[3]);
  const uint32x4x2_t a = vtrnq_u32(vreinterpretq_u32_u16(t01.val[0]),
                                   vreinterpretq_u32_u16(t23.val[0]));
  const uint32x4x2_t b = vtrnq_u32(vreinterpretq_u32_u16(t01.val[1]),
                                   vreinterpretq_u32_u16(t23.val[1]));
  *c04 = vreinterpretq_u16_u32(a.val[0]);
  *c26 = vreinterpretq_u16_u32(a.val[1]);
  *c15 = vreinterpretq_u16_u32(b.val[0]);
  *c37 = vreinterpretq_u16_u32(b.val[1]);
}

// Inverse of transpose_rows_4x8().
static INLINE void transpose_cols_4x8(uint16x8_t c04, uint16x8_t c15,
                                      uint16x8_t c26, uint16x8_t c37,
                                      uint16x8_t r[4]) {
  const uint32x4x2_t a =
      vtrnq_u32(vreinterpretq_u32_u16(c04), vreinterpretq_u32_u16(c26));
  const uint32x4x2_t b =
      vtrnq_u32(vreinterpretq_u32_u16(c15), vreinterpretq_u32_u16(c37));
  const uint16x8x2_t r01 = vtrnq_u16(vreinterpretq_u16_u32(a.val[0]),
                                     vreinterpretq_u16_u32(b.val[0]));
  const uint16x8x2_t r23 = vtrnq_u16(vreinterpretq_u16_u32(a.val[1]),
                                     vreinterpretq_u16_u32(b.val[1]));
  r[0] = r01.val[0];
  r[1] = r01.val[1];
  r[2] = r23.val[0];
  r[3] = r23.val[1];
}

// Vertical edge: positions -4..3 of rows 0..3 into px.
static INLINE void load_vert_near(const uint16_t *s, int pitch,
                                  uint16x4_t *px) {
  uint16x8_t r[4], c04, c15, c26, c37;
  for (int i = 0; i < 4; ++i) r[i] = vld1q_u16(s + i * pitch - 4);
  transpose_rows_4x8(r, &c04, &c15, &c26, &c37);
  px[PX(-4)] = vget_low_u16(c04);
  px[PX(0)] = vget_high_u16(c04);
  px[PX(-3)] = vget_low_u16(c15);
  px[PX(1)] = vget_high_u16(c15);
  px[PX(-2)] = vget_low_u16(c26);
  px[PX(2)] = vget_high_u16(c26);
  px[PX(-1)] = vget_low_u16(c37);
  px[PX(3)] = vget_high_u16(c37);
}

// Vertical edge: positions -8..-5 and 4..7 of rows 0..3 into px.
static INLINE void load_vert_far(const uint16_t *s, int pitch, uint16x4_t *px) {
  uint16x8_t r[4], c04, c15, c26, c37;
  for (int i = 0; i < 4; ++i) {
    r[i] =
        vcombine_u16(vld1_u16(s + i * pitch - 8), vld1_u16(s + i * pitch + 4));
  }
  transpose_rows_4x8(r, &c04, &c15, &c26, &c37);
  px[PX(-8)] = vget_low_u16(c04);
  px[PX(4)] = vget_high_u16(c04);
  px[PX(-7)] = vget_low_u16(c15);
  px[PX(5)] = vget_high_u16(c15);
  px[PX(-6)] = vget_low_u16(c26);
  px[PX(6)] = vget_high_u16(c26);
  px[PX(-5)] = vget_low_u16(c37);
  px[PX(7)] = vget_high_u16(c37);
}

static INLINE void store_vert_near(uint16_t *s, int pitch,
                                   const uint16x4_t *px) {
  uint16x8_t r[4];
  transpose_cols_4x8(vcombine_u16(px[PX(-4)], px[PX(0)]),
                     vcombine_u16(px[PX(-3)], px[PX(1)]),
                     vcombine_u16(px[PX(-2)], px[PX(2)]),
                     vcombine_u16(px[PX(-1)], px[PX(3)]), r);
  for (int i = 0; i < 4; ++i) vst1q_u16(s + i * pitch - 4, r[i]);
}

static INLINE void store_vert_far(uint16_t *s, int pitch,
                                  const uint16x4_t *px) {
  uint16x8_t r[4];
  transpose_cols_4x8(vcombine_u16(px[PX(-8)], px[PX(4)]),
                     vcombine_u16(px[PX(-7)], px[PX(5)]),
                     vcombine_u16(px[PX(-6)], px[PX(6)]),
                     vcombine_u16(px[PX(-5)], px[PX(7)]), r);
  for (int i = 0; i < 4; ++i) {
    vst1_u16(s + i * pitch - 8, vget_low_u16(r[i]));
    vst1_u16(s + i * pitch + 4, vget_high_u16(r[i]));
  }
}

static INLINE int16x8_t pair_s16(uint16x4_t a, uint16x4_t b) {
  return vreinterpretq_s16_u16(vcombine_u16(a, b));
}

// |a - 2 * b + c|
static INLINE int16x8_t sec_deriv(int16x8_t a, int16x8_t b, int16x8_t c) {
  return vabsq_s16(vaddq_s16(vsubq_s16(a, vshlq_n_s16(b, 1)), c));
}

// |(base - far) - dist * grad|
static INLINE int16x8_t end_dir_diff(int16x8_t base, int16x8_t far,
                                     int16x8_t grad, int16_t dist) {
  return vabsq_s16(vmlsq_n_s16(vsubq_s16(base, far), grad, dist));
}

// (line 0 + line 3 + 1) >> 1 of each half, in lanes 0 and 4.
static INLINE uint16x8_t avg_lines_0_3(int16x8_t v) {
  const uint16x8_t u = vreinterpretq_u16_s16(v);
  return vrhaddq_u16(u, vrev64q_u16(u));
}

// Lanes 0 and 4 of a, b, c, d into lanes 0..7.
static INLINE uint16x8_t gather_lanes_0_4(uint16x8_t a, uint16x8_t b,
                                          uint16x8_t c, uint16x8_t d) {
  const uint16x8_t ab = vuzpq_u16(a, b).val[0];  // a0 . a4 . b0 . b4 .
  const uint16x8_t cd = vuzpq_u16(c, d).val[0];  // c0 . c4 . d0 . d4 .
  return vuzpq_u16(ab, cd).val[0];
}

// Number of samples to modify on each side (filt_choice_highbd() in
// avm_dsp/loopfilter.c), from lines 0 and 3. px[PX(-4)]..px[PX(3)] must be
// loaded; load_far() fills px[PX(-8)]..px[PX(-5)] and px[PX(4)]..px[PX(7)]
// once the 4-sample tests are reached. max_samples_neg must be >= 1.
typedef void (*load_far_fn)(const uint16_t *s, int pitch, int max_samples_neg,
                            int max_samples_pos, uint16x4_t *px);

static INLINE int filt_choice_neon(uint16x4_t *px, const uint16_t *s, int pitch,
                                   load_far_fn load_far, int max_samples_neg,
                                   int max_samples_pos, int q_thresh,
                                   int side_thresh) {
  assert(max_samples_pos == 1 || max_samples_pos == 3 || max_samples_pos == 4 ||
         max_samples_pos == 6 || max_samples_pos == 8);

  if (max_samples_pos < 1 || max_samples_pos < max_samples_neg) return 0;

  const int16x8_t base = pair_s16(px[PX(-1)], px[PX(0)]);
  const int16x8_t grad = vsubq_s16(base, pair_s16(px[PX(-2)], px[PX(1)]));

  // { sd[-2], sd[-1], sd[0], sd[1], diff_neg(3), diff_pos(3), -, - }
  DECLARE_ALIGNED(16, uint16_t, res[8]);
  {
    const int16x8_t sd_m2m1 = sec_deriv(pair_s16(px[PX(-3)], px[PX(-2)]),
                                        pair_s16(px[PX(-2)], px[PX(-1)]), base);
    const int16x8_t sd_p0p1 = sec_deriv(base, pair_s16(px[PX(0)], px[PX(1)]),
                                        pair_s16(px[PX(1)], px[PX(2)]));
    const int16x8_t diff3 =
        end_dir_diff(base, pair_s16(px[PX(-4)], px[PX(3)]), grad, 3);
    const uint16x8_t d3 = avg_lines_0_3(diff3);
    vst1q_u16(res, gather_lanes_0_4(avg_lines_0_3(sd_m2m1),
                                    avg_lines_0_3(sd_p0p1), d3, d3));
  }

  // Testing for 1 sample modification
  if (res[0] > side_thresh || res[3] > side_thresh) return 0;
  if (max_samples_pos == 1) return 1;

  // Testing for 2 sample modification
  const int sd_sum = res[1] + res[2];
  const int side_thresh2 = side_thresh >> 2;
  if (res[0] > side_thresh2 || res[3] > side_thresh2 ||
      sd_sum > q_thresh * DF_6_THRESH)
    return 1;

  // Testing for 3 sample modification
  const int side_thresh3 = side_thresh >> FILT_8_THRESH_SHIFT;
  int end_dir_thresh = (side_thresh * 3) >> 4;
  if (res[0] > side_thresh3 || res[3] > side_thresh3 ||
      sd_sum > q_thresh * DF_8_THRESH ||
      (max_samples_neg > 2 && res[4] > end_dir_thresh) ||
      res[5] > end_dir_thresh)
    return 2;
  if (max_samples_pos == 3) return 3;

  // Testing for 4 sample modification and above
  const int transition = sd_sum << DF_Q_THRESH_SHIFT;
  load_far(s, pitch, max_samples_neg, max_samples_pos, px);

  // { diff_neg(4), diff_pos(4), diff_neg(6), diff_pos(6), diff_neg(7),
  //   diff_pos(7), -, - }
  {
    const int16x8_t diff4 =
        end_dir_diff(base, pair_s16(px[PX(-5)], px[PX(4)]), grad, 4);
    const int16x8_t diff6 =
        end_dir_diff(base, pair_s16(px[PX(-7)], px[PX(6)]), grad, 6);
    const int16x8_t diff7 =
        end_dir_diff(base, pair_s16(px[PX(-8)], px[PX(7)]), grad, 7);
    const uint16x8_t d7 = avg_lines_0_3(diff7);
    vst1q_u16(res, gather_lanes_0_4(avg_lines_0_3(diff4), avg_lines_0_3(diff6),
                                    d7, d7));
  }

  // 4 sample modification
  end_dir_thresh = (side_thresh * 4) >> 4;
  if (transition > q_thresh * q_first[0] ||
      (max_samples_neg >= 4 && res[0] > end_dir_thresh) ||
      res[1] > end_dir_thresh)
    return 3;
  if (max_samples_pos <= 4) return 4;

  // 6 sample modification
  end_dir_thresh = (side_thresh * 6) >> 4;
  if (transition > q_thresh * q_first[2] ||
      (max_samples_neg >= 6 && res[2] > end_dir_thresh) ||
      res[3] > end_dir_thresh)
    return 4;
  if (max_samples_pos <= 6) return 6;

  // 8 sample modification (using diff 7 so it stays within 8 samples)
  end_dir_thresh = (side_thresh * 8) >> 4;
  if (transition > q_thresh * q_first[4] ||
      (max_samples_neg >= 7 && res[4] > end_dir_thresh) ||
      res[5] > end_dir_thresh)
    return 6;
  return MAX_DBL_FLT_LEN;
}

// filt_generic_asym_highbd() on the 4 lines of px. Modifies px[PX(-wn)] ..
// px[PX(wp - 1)] (only the side that is not lossless); returns 0 when no
// sample changes. wn and wp must be >= 1.
static INLINE int filt_generic_asym_neon(uint16x4_t *px, int q_threshold,
                                         int wn, int wp, int bd,
                                         int is_lossless_neg,
                                         int is_lossless_pos) {
  const int32x4_t d1 = vreinterpretq_s32_u32(vsubl_u16(px[PX(0)], px[PX(-1)]));
  const int32x4_t d2 = vreinterpretq_s32_u32(vsubl_u16(px[PX(1)], px[PX(-2)]));
  int32x4_t delta = vsubq_s32(vmulq_n_s32(d1, 3), d2);
  if (all_zero_s32(delta)) return 0;
  delta = vshlq_n_s32(delta, 2);

  const int32_t q_clamp = q_threshold * q_thresh_mults[AVMMAX(wn, wp) - 1];
  delta =
      vmaxq_s32(vminq_s32(delta, vdupq_n_s32(q_clamp)), vdupq_n_s32(-q_clamp));

  const uint16x4_t max_val = vdup_n_u16((uint16_t)((1 << bd) - 1));

  if (!is_lossless_neg) {
    const int32x4_t delta_neg = vmulq_n_s32(delta, w_mult[wn - 1]);
    for (int i = 0; i < wn; ++i) {
      const int32x4_t adj =
          vrshrq_n_s32(vmulq_n_s32(delta_neg, wn - i), 3 + DF_SHIFT);
      // pix + adj, exact in 32 bits.
      const int32x4_t v = vreinterpretq_s32_u32(
          vaddw_u16(vreinterpretq_u32_s32(adj), px[PX(-i - 1)]));
      px[PX(-i - 1)] = vmin_u16(vqmovun_s32(v), max_val);
    }
  }
  if (!is_lossless_pos) {
    const int32x4_t delta_pos = vmulq_n_s32(delta, w_mult[wp - 1]);
    for (int i = 0; i < wp; ++i) {
      const int32x4_t adj =
          vrshrq_n_s32(vmulq_n_s32(delta_pos, wp - i), 3 + DF_SHIFT);
      const int32x4_t v =
          vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(px[PX(i)])), adj);
      px[PX(i)] = vmin_u16(vqmovun_s32(v), max_val);
    }
  }
  return 1;
}

// Horizontal edge: rows -8..-5 and 4..7 that the C decision reads; the others
// get a harmless stand-in so no row outside the C footprint is read.
static void load_horz_far(const uint16_t *s, int pitch, int max_samples_neg,
                          int max_samples_pos, uint16x4_t *px) {
  const uint16x4_t m1 = px[PX(-1)];
  const uint16x4_t p0 = px[PX(0)];
  px[PX(-5)] = max_samples_neg >= 4 ? vld1_u16(s - 5 * pitch) : m1;
  px[PX(-7)] = max_samples_neg >= 6 ? vld1_u16(s - 7 * pitch) : m1;
  px[PX(-8)] = max_samples_neg >= 7 ? vld1_u16(s - 8 * pitch) : m1;
  px[PX(4)] = vld1_u16(s + 4 * pitch);
  px[PX(6)] = max_samples_pos >= 6 ? vld1_u16(s + 6 * pitch) : p0;
  px[PX(7)] = max_samples_pos >= 7 ? vld1_u16(s + 7 * pitch) : p0;
}

static void load_vert_far_fn(const uint16_t *s, int pitch, int max_samples_neg,
                             int max_samples_pos, uint16x4_t *px) {
  (void)max_samples_neg;
  (void)max_samples_pos;
  load_vert_far(s, pitch, px);
}

void avm_highbd_lpf_horizontal_generic_neon(
    uint16_t *s, int pitch, int filt_width_neg, int filt_width_pos,
    const uint16_t *q_thresh, const uint16_t *side_thresh, int bd,
    int is_lossless_neg, int is_lossless_pos) {
  const int filt_neg = (filt_width_neg >> 1) - 1;
  // With filt_neg < 1 the C version modifies nothing.
  if (filt_neg < 1 || (is_lossless_neg && is_lossless_pos)) return;
  if (!*q_thresh || !*side_thresh) return;

  const int max_samples_neg = filt_neg;
  const int max_samples_pos = filt_width_pos / 2 - 1;

  uint16x4_t px[16];
  px[PX(-3)] = vld1_u16(s - 3 * pitch);
  px[PX(-2)] = vld1_u16(s - 2 * pitch);
  px[PX(-1)] = vld1_u16(s - 1 * pitch);
  px[PX(0)] = vld1_u16(s);
  px[PX(1)] = vld1_u16(s + 1 * pitch);
  px[PX(2)] = vld1_u16(s + 2 * pitch);
  px[PX(-4)] = max_samples_neg > 2 ? vld1_u16(s - 4 * pitch) : px[PX(-1)];
  px[PX(3)] = max_samples_pos >= 3 ? vld1_u16(s + 3 * pitch) : px[PX(0)];

  const int filter =
      filt_choice_neon(px, s, pitch, load_horz_far, max_samples_neg,
                       max_samples_pos, *q_thresh, *side_thresh);
  const int wn = AVMMIN(filter, filt_neg);
  if (wn < 1) return;
  const int wp = filter;

  // Reload the rows the filter touches (px may hold stand-ins).
  uint16x4_t f[16];
  const int lo = AVMMAX(wn, 2), hi = AVMMAX(wp, 2);
  for (int k = -lo; k < hi; ++k) f[PX(k)] = vld1_u16(s + k * pitch);

  if (!filt_generic_asym_neon(f, *q_thresh, wn, wp, bd, is_lossless_neg,
                              is_lossless_pos))
    return;

  if (!is_lossless_neg) {
    for (int k = -wn; k < 0; ++k) vst1_u16(s + k * pitch, f[PX(k)]);
  }
  if (!is_lossless_pos) {
    for (int k = 0; k < wp; ++k) vst1_u16(s + k * pitch, f[PX(k)]);
  }
}

void avm_highbd_lpf_vertical_generic_neon(
    uint16_t *s, int pitch, int filt_width_neg, int filt_width_pos,
    const uint16_t *q_thresh, const uint16_t *side_thresh, int bd,
    int is_lossless_neg, int is_lossless_pos) {
  const int filt_neg = (filt_width_neg >> 1) - 1;
  // With filt_neg < 1 the C version modifies nothing.
  if (filt_neg < 1 || (is_lossless_neg && is_lossless_pos)) return;
  if (!*q_thresh || !*side_thresh) return;

  const int max_samples_neg = filt_neg;
  const int max_samples_pos = filt_width_pos / 2 - 1;

  uint16x4_t px[16];
  load_vert_near(s, pitch, px);

  const int filter =
      filt_choice_neon(px, s, pitch, load_vert_far_fn, max_samples_neg,
                       max_samples_pos, *q_thresh, *side_thresh);
  const int wn = AVMMIN(filter, filt_neg);
  if (wn < 1) return;
  const int wp = filter;

  // A filter longer than 4 is only chosen after load_vert_far() ran.
  if (!filt_generic_asym_neon(px, *q_thresh, wn, wp, bd, is_lossless_neg,
                              is_lossless_pos))
    return;

  store_vert_near(s, pitch, px);
  if (wp > 4) store_vert_far(s, pitch, px);
}
