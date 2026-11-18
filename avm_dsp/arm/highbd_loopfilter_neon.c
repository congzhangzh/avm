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

// The scalar tests of filt_choice_highbd() (avm_dsp/loopfilter.c) on the
// line-0/line-3 averages r[i * st]: i = 0..3 are sd[-2], sd[-1], sd[0], sd[1],
// 4 and 5 are the end-direction differences at distance 3 (neg, pos).
// Returns the number of samples to modify, or -1 when the tests at distance
// 4 and above must run; then *transition is set for filt_choice_far().
static INLINE int filt_choice_near(const uint16_t *r, int st,
                                   int max_samples_neg, int max_samples_pos,
                                   int q_thresh, int side_thresh,
                                   int *transition) {
  // Testing for 1 sample modification
  if (r[0] > side_thresh || r[3 * st] > side_thresh) return 0;
  if (max_samples_pos == 1) return 1;

  // Testing for 2 sample modification
  const int sd_sum = r[st] + r[2 * st];
  const int side_thresh2 = side_thresh >> 2;
  if (r[0] > side_thresh2 || r[3 * st] > side_thresh2 ||
      sd_sum > q_thresh * DF_6_THRESH)
    return 1;

  // Testing for 3 sample modification
  const int side_thresh3 = side_thresh >> FILT_8_THRESH_SHIFT;
  const int end_dir_thresh = (side_thresh * 3) >> 4;
  if (r[0] > side_thresh3 || r[3 * st] > side_thresh3 ||
      sd_sum > q_thresh * DF_8_THRESH ||
      (max_samples_neg > 2 && r[4 * st] > end_dir_thresh) ||
      r[5 * st] > end_dir_thresh)
    return 2;
  if (max_samples_pos == 3) return 3;

  *transition = sd_sum << DF_Q_THRESH_SHIFT;
  return -1;
}

// The tests at distance 4 and above, on r[i * st]: i = 0..5 are the
// end-direction differences (neg, pos) at distances 4, 6 and 7.
static INLINE int filt_choice_far(const uint16_t *r, int st,
                                  int max_samples_neg, int max_samples_pos,
                                  int q_thresh, int side_thresh,
                                  int transition) {
  // 4 sample modification
  int end_dir_thresh = (side_thresh * 4) >> 4;
  if (transition > q_thresh * q_first[0] ||
      (max_samples_neg >= 4 && r[0] > end_dir_thresh) || r[st] > end_dir_thresh)
    return 3;
  if (max_samples_pos <= 4) return 4;

  // 6 sample modification
  end_dir_thresh = (side_thresh * 6) >> 4;
  if (transition > q_thresh * q_first[2] ||
      (max_samples_neg >= 6 && r[2 * st] > end_dir_thresh) ||
      r[3 * st] > end_dir_thresh)
    return 4;
  if (max_samples_pos <= 6) return 6;

  // 8 sample modification (using diff 7 so it stays within 8 samples)
  end_dir_thresh = (side_thresh * 8) >> 4;
  if (transition > q_thresh * q_first[4] ||
      (max_samples_neg >= 7 && r[4 * st] > end_dir_thresh) ||
      r[5 * st] > end_dir_thresh)
    return 6;
  return MAX_DBL_FLT_LEN;
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

  int transition = 0;
  const int near = filt_choice_near(res, 1, max_samples_neg, max_samples_pos,
                                    q_thresh, side_thresh, &transition);
  if (near >= 0) return near;

  // Testing for 4 sample modification and above
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

  return filt_choice_far(res, 1, max_samples_neg, max_samples_pos, q_thresh,
                         side_thresh, transition);
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

// Two segments at once: two adjacent segments of one horizontal edge
// (s[0..3] and s[4..7]). r[PX(k)] holds position k of the 8 lines, lanes
// 0..3 the lines of segment 0 and lanes 4..7 those of segment 1. Each
// segment gets exactly what the single-segment kernel would do to it: the
// decision values of both come out of the same vectors, but the decision,
// the filter lengths and the filter stay per segment.

// The cases where the single kernel returns before or at the start of its
// decision (filt_choice_neon() returning 0 up front).
static INLINE int dual_seg_on(int msn, int msp, int q, int side) {
  return msn >= 1 && q && side && msp >= msn;
}

// Fills r[PX(-8)]..r[PX(-5)] and r[PX(4)]..r[PX(7)] once segment h with
// undecided[h] set reaches the tests at distance 4 and above.
typedef void (*load_far8_fn)(const uint16_t *s, int pitch, const int *msn,
                             const int *msp, const int *undecided,
                             uint16x8_t *r);

// filt_choice_neon() for both segments; r[PX(-4)]..r[PX(3)] must be loaded.
static INLINE void dual_filt_choice(uint16x8_t *r, const uint16_t *s, int pitch,
                                    load_far8_fn load_far, const int *msn,
                                    const int *msp, const int *q,
                                    const int *side, int *filter) {
  assert(msp[0] == 1 || msp[0] == 3 || msp[0] == 4 || msp[0] == 6 ||
         msp[0] == 8);
  assert(msp[1] == 1 || msp[1] == 3 || msp[1] == 4 || msp[1] == 6 ||
         msp[1] == 8);
#define S16(k) vreinterpretq_s16_u16(r[PX(k)])
  const int16x8_t m1 = S16(-1), p0 = S16(0);
  const int16x8_t gn = vsubq_s16(m1, S16(-2));
  const int16x8_t gp = vsubq_s16(p0, S16(1));

  // res[2 * i + h]: decision value i of segment h, in the order that
  // filt_choice_near() (then filt_choice_far()) reads.
  DECLARE_ALIGNED(16, uint16_t, res[16]);
  {
    const int16x8_t sd_m2 = sec_deriv(S16(-3), S16(-2), m1);
    const int16x8_t sd_m1 = sec_deriv(S16(-2), m1, p0);
    const int16x8_t sd_p0 = sec_deriv(m1, p0, S16(1));
    const int16x8_t sd_p1 = sec_deriv(p0, S16(1), S16(2));
    const uint16x8_t dn3 = avg_lines_0_3(end_dir_diff(m1, S16(-4), gn, 3));
    const uint16x8_t dp3 = avg_lines_0_3(end_dir_diff(p0, S16(3), gp, 3));
    vst1q_u16(res,
              gather_lanes_0_4(avg_lines_0_3(sd_m2), avg_lines_0_3(sd_m1),
                               avg_lines_0_3(sd_p0), avg_lines_0_3(sd_p1)));
    vst1q_u16(res + 8, gather_lanes_0_4(dn3, dp3, dn3, dp3));
  }

  int transition[2] = { 0, 0 };
  for (int h = 0; h < 2; ++h) {
    filter[h] = filt_choice_near(res + h, 2, msn[h], msp[h], q[h], side[h],
                                 &transition[h]);
  }
  if (filter[0] >= 0 && filter[1] >= 0) return;

  const int undecided[2] = { filter[0] < 0, filter[1] < 0 };
  load_far(s, pitch, msn, msp, undecided, r);
  {
    const uint16x8_t dn4 = avg_lines_0_3(end_dir_diff(m1, S16(-5), gn, 4));
    const uint16x8_t dp4 = avg_lines_0_3(end_dir_diff(p0, S16(4), gp, 4));
    const uint16x8_t dn6 = avg_lines_0_3(end_dir_diff(m1, S16(-7), gn, 6));
    const uint16x8_t dp6 = avg_lines_0_3(end_dir_diff(p0, S16(6), gp, 6));
    const uint16x8_t dn7 = avg_lines_0_3(end_dir_diff(m1, S16(-8), gn, 7));
    const uint16x8_t dp7 = avg_lines_0_3(end_dir_diff(p0, S16(7), gp, 7));
    vst1q_u16(res, gather_lanes_0_4(dn4, dp4, dn6, dp6));
    vst1q_u16(res + 8, gather_lanes_0_4(dn7, dp7, dn7, dp7));
  }
#undef S16
  for (int h = 0; h < 2; ++h) {
    if (!undecided[h]) continue;
    filter[h] = filt_choice_far(res + h, 2, msn[h], msp[h], q[h], side[h],
                                transition[h]);
  }
}

// Filters both segments of r over positions -lo..hi-1 (which must cover
// -2..1 and every active segment's -wn..wp-1) into f0 (segment 0) and f1
// (segment 1); ch[h] is set when segment h changed.
static INLINE void dual_filter(const uint16x8_t *r, int lo, int hi,
                               const int *q, const int *wn, const int *wp,
                               int bd, uint16x4_t *f0, uint16x4_t *f1,
                               int *ch) {
  for (int k = -lo; k < hi; ++k) {
    f0[PX(k)] = vget_low_u16(r[PX(k)]);
    f1[PX(k)] = vget_high_u16(r[PX(k)]);
  }
  ch[0] =
      wn[0] >= 1 && filt_generic_asym_neon(f0, q[0], wn[0], wp[0], bd, 0, 0);
  ch[1] =
      wn[1] >= 1 && filt_generic_asym_neon(f1, q[1], wn[1], wp[1], bd, 0, 0);
}

#if defined(__aarch64__)
// dual_filter() in 16-bit lanes, in place on r, for the common case of both
// segments active with the same lengths wn, wp. Returns 0 (and changes
// nothing) when a clamp bound does not fit in int16; the caller then uses
// dual_filter(). Exact for samples within the bit depth (as the decision):
// - 3 * (p0 - m1) - (p1 - m2) is at most 4 * 4095 in magnitude; vqshl by 2
//   saturates at +-32767, beyond the clamp bound, so the clamp result is
//   that of the C code.
// - vqrdmulh(delta, 16 * m) is (2 * delta * 16 * m + 2^15) >> 16, i.e.
//   ROUND_POWER_OF_TWO(delta * m, 3 + DF_SHIFT) with m = w_mult * (w - i)
//   <= 120, and never saturates.
// - sample +- adj stays within int16 (|adj| <= 32767 * 120 >> 11), and the
//   min/max is clip_pixel_highbd().
static INLINE int dual_filter16_same(uint16x8_t *r, const int *q, int wn,
                                     int wp, int bd, int *ch) {
  const int width = AVMMAX(wn, wp);
  const int c0 = q[0] * q_thresh_mults[width - 1];
  const int c1 = q[1] * q_thresh_mults[width - 1];
  if (c0 > INT16_MAX || c1 > INT16_MAX) return 0;
#define S16(k) vreinterpretq_s16_u16(r[PX(k)])
  const int16x8_t d = vsubq_s16(vmulq_n_s16(vsubq_s16(S16(0), S16(-1)), 3),
                                vsubq_s16(S16(1), S16(-2)));
  const uint16x8_t du = vreinterpretq_u16_s16(d);
  ch[0] = vmaxv_u16(vget_low_u16(du)) != 0;
  ch[1] = vmaxv_u16(vget_high_u16(du)) != 0;
  if (!ch[0] && !ch[1]) return 1;

  const int16x8_t cv =
      vcombine_s16(vdup_n_s16((int16_t)c0), vdup_n_s16((int16_t)c1));
  const int16x8_t delta =
      vmaxq_s16(vminq_s16(vqshlq_n_s16(d, 2), cv), vnegq_s16(cv));
  const int16x8_t zero = vdupq_n_s16(0);
  const int16x8_t max_val = vdupq_n_s16((int16_t)((1 << bd) - 1));

  const int mn = 16 * w_mult[wn - 1];
  for (int i = 0; i < wn; ++i) {
    const int16x8_t adj = vqrdmulhq_n_s16(delta, (int16_t)(mn * (wn - i)));
    const int16x8_t v = vaddq_s16(S16(-i - 1), adj);
    r[PX(-i - 1)] =
        vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(v, zero), max_val));
  }
  const int mp = 16 * w_mult[wp - 1];
  for (int i = 0; i < wp; ++i) {
    const int16x8_t adj = vqrdmulhq_n_s16(delta, (int16_t)(mp * (wp - i)));
    const int16x8_t v = vsubq_s16(S16(i), adj);
    r[PX(i)] = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(v, zero), max_val));
  }
#undef S16
  return 1;
}
#endif  // defined(__aarch64__)

// Horizontal edge: rows -8..-5 and 4..7 that the far tests of a segment
// still deciding read; the others get a stand-in. A row is loaded 8 wide
// when either segment reads it: the other half lies in the same frame row
// next to a segment that is filtered too, and its values only feed tests
// that the segment's own limits rule out.
static void load_horz_far8(const uint16_t *s, int pitch, const int *msn,
                           const int *msp, const int *undecided,
                           uint16x8_t *r) {
  int need_n4 = 0, need_n6 = 0, need_n7 = 0, need_p6 = 0, need_p7 = 0;
  for (int h = 0; h < 2; ++h) {
    if (!undecided[h]) continue;
    need_n4 |= msn[h] >= 4;
    need_n6 |= msn[h] >= 6;
    need_n7 |= msn[h] >= 7;
    need_p6 |= msp[h] >= 6;
    need_p7 |= msp[h] >= 7;
  }
  r[PX(-5)] = need_n4 ? vld1q_u16(s - 5 * pitch) : r[PX(-1)];
  r[PX(-7)] = need_n6 ? vld1q_u16(s - 7 * pitch) : r[PX(-1)];
  r[PX(-8)] = need_n7 ? vld1q_u16(s - 8 * pitch) : r[PX(-1)];
  r[PX(4)] = vld1q_u16(s + 4 * pitch);
  r[PX(6)] = need_p6 ? vld1q_u16(s + 6 * pitch) : r[PX(0)];
  r[PX(7)] = need_p7 ? vld1q_u16(s + 7 * pitch) : r[PX(0)];
}

// Two adjacent segments of one horizontal edge: s[0..3] with the parameters
// of index 0 and s[4..7] with those of index 1. Each segment's stores cover
// exactly the rows the single kernel stores.
void avm_highbd_lpf_horizontal_generic_dual_neon(
    uint16_t *s, int pitch, int filt_width_neg0, int filt_width_pos0,
    const uint16_t *q_thresh0, const uint16_t *side_thresh0,
    int filt_width_neg1, int filt_width_pos1, const uint16_t *q_thresh1,
    const uint16_t *side_thresh1, int bd) {
  const int msn[2] = { (filt_width_neg0 >> 1) - 1, (filt_width_neg1 >> 1) - 1 };
  const int msp[2] = { filt_width_pos0 / 2 - 1, filt_width_pos1 / 2 - 1 };
  const int q[2] = { *q_thresh0, *q_thresh1 };
  const int side[2] = { *side_thresh0, *side_thresh1 };
  const int on0 = dual_seg_on(msn[0], msp[0], q[0], side[0]);
  const int on1 = dual_seg_on(msn[1], msp[1], q[1], side[1]);
  if (!on0 || !on1) {
    if (on0) {
      avm_highbd_lpf_horizontal_generic_neon(s, pitch, filt_width_neg0,
                                             filt_width_pos0, q_thresh0,
                                             side_thresh0, bd, 0, 0);
    } else if (on1) {
      avm_highbd_lpf_horizontal_generic_neon(s + 4, pitch, filt_width_neg1,
                                             filt_width_pos1, q_thresh1,
                                             side_thresh1, bd, 0, 0);
    }
    return;
  }

  uint16x8_t r[16];
  for (int k = -3; k < 3; ++k) r[PX(k)] = vld1q_u16(s + k * pitch);
  r[PX(-4)] = (msn[0] > 2 || msn[1] > 2) ? vld1q_u16(s - 4 * pitch) : r[PX(-1)];
  r[PX(3)] = (msp[0] >= 3 || msp[1] >= 3) ? vld1q_u16(s + 3 * pitch) : r[PX(0)];

  int filter[2];
  dual_filt_choice(r, s, pitch, load_horz_far8, msn, msp, q, side, filter);

  int wn[2], wp[2];
  for (int h = 0; h < 2; ++h) {
    wn[h] = AVMMIN(filter[h], msn[h]);
    wp[h] = filter[h];
  }
  const int act0 = wn[0] >= 1, act1 = wn[1] >= 1;
  if (!act0 && !act1) return;

  // Rows the filters touch: -3..2 are loaded already, the others may hold
  // stand-ins and are reloaded.
  const int lo = AVMMAX(AVMMAX(act0 ? wn[0] : 0, act1 ? wn[1] : 0), 2);
  const int hi = AVMMAX(AVMMAX(act0 ? wp[0] : 0, act1 ? wp[1] : 0), 2);
  for (int k = -lo; k < -3; ++k) r[PX(k)] = vld1q_u16(s + k * pitch);
  for (int k = 3; k < hi; ++k) r[PX(k)] = vld1q_u16(s + k * pitch);

  int ch[2];
#if defined(__aarch64__)
  if (act0 && act1 && wn[0] == wn[1] && wp[0] == wp[1] &&
      dual_filter16_same(r, q, wn[0], wp[0], bd, ch)) {
    // Both segments cover the same rows.
    for (int k = -wn[0]; k < wp[0]; ++k) {
      if (ch[0] && ch[1]) {
        vst1q_u16(s + k * pitch, r[PX(k)]);
      } else if (ch[0]) {
        vst1_u16(s + k * pitch, vget_low_u16(r[PX(k)]));
      } else if (ch[1]) {
        vst1_u16(s + 4 + k * pitch, vget_high_u16(r[PX(k)]));
      }
    }
    return;
  }
#endif
  uint16x4_t f0[16], f1[16];
  dual_filter(r, lo, hi, q, wn, wp, bd, f0, f1, ch);
  if (!ch[0] && !ch[1]) return;

  // Store the rows each changed segment's filter covers, and only those.
  const int sn0 = ch[0] ? wn[0] : 0, sp0 = ch[0] ? wp[0] : 0;
  const int sn1 = ch[1] ? wn[1] : 0, sp1 = ch[1] ? wp[1] : 0;
  for (int k = -AVMMAX(sn0, sn1); k < AVMMAX(sp0, sp1); ++k) {
    const int t0 = k < 0 ? -k <= sn0 : k < sp0;
    const int t1 = k < 0 ? -k <= sn1 : k < sp1;
    if (t0 && t1) {
      vst1q_u16(s + k * pitch, vcombine_u16(f0[PX(k)], f1[PX(k)]));
    } else if (t0) {
      vst1_u16(s + k * pitch, f0[PX(k)]);
    } else if (t1) {
      vst1_u16(s + 4 + k * pitch, f1[PX(k)]);
    }
  }
}
