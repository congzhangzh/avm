/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <string.h>
#include <tuple>

#include "config/avm_config.h"
#include "config/avm_dsp_rtcd.h"

#include "test/acm_random.h"
#include "test/util.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace {
using libavm_test::ACMRandom;
using std::make_tuple;
using std::tuple;

typedef void (*LoopFilterFunc)(uint16_t *s, int pitch, int filt_width_neg,
                               int filt_width_pos, const uint16_t *q_thresh,
                               const uint16_t *side_thresh, int bd,
                               int is_lossless_neg, int is_lossless_pos);

// <test_func, ref_func, bit_depth, filt_len_neg, filt_len_pos>
typedef tuple<LoopFilterFunc, LoopFilterFunc, int, int, int> Params;

const int kNumCoeffs = 1024;

const int number_of_iterations = 10000;

const int kSpeedIterations = 50000000;

void InitLoopFilterInput(uint16_t *s, uint16_t *ref_s, ACMRandom *rnd,
                         const uint8_t limit, const int mask, const int32_t p,
                         const int i, const uint16_t prob) {
  uint16_t tmp_s[kNumCoeffs];

  for (int j = 0; j < kNumCoeffs;) {
    const uint8_t val = rnd->Rand8();
    if ((val & prob) == prob) {
      tmp_s[j] = rnd->Rand16();
      j++;
    } else {
      int k = 0;
      while (k++ < ((val & 0x1f) + 1) && j < kNumCoeffs) {
        if (j < 1) {
          tmp_s[j] = rnd->Rand16();
        } else if (val & 0x20) {  // Increment by a value within the limit.
          tmp_s[j] = static_cast<uint16_t>(tmp_s[j - 1] + (limit - 1));
        } else {  // Decrement by a value within the limit.
          tmp_s[j] = static_cast<uint16_t>(tmp_s[j - 1] - (limit - 1));
        }
        j++;
      }
    }
  }

  for (int j = 0; j < kNumCoeffs;) {
    const uint8_t val = rnd->Rand8();
    if ((val & prob) == prob) {
      j++;
    } else {
      int k = 0;
      while (k++ < ((val & 0x1f) + 1) && j < kNumCoeffs) {
        if (j < 1) {
          tmp_s[j] = rnd->Rand16();
        } else if (val & 0x20) {  // Increment by a value within the limit.
          tmp_s[(j % 32) * 32 + j / 32] = static_cast<uint16_t>(
              tmp_s[((j - 1) % 32) * 32 + (j - 1) / 32] + (limit - 1));
        } else {  // Decrement by a value within the limit.
          tmp_s[(j % 32) * 32 + j / 32] = static_cast<uint16_t>(
              tmp_s[((j - 1) % 32) * 32 + (j - 1) / 32] - (limit - 1));
        }
        j++;
      }
    }
  }

  for (int j = 0; j < kNumCoeffs; j++) {
    if (i % 2) {
      s[j] = tmp_s[j] & mask;
    } else {
      s[j] = tmp_s[p * (j % p) + j / p] & mask;
    }
    ref_s[j] = s[j];
  }
}

class LoopFilterTest : public ::testing::TestWithParam<Params> {
 public:
  virtual void SetUp() {
    func_test_ = GET_PARAM(0);
    func_ref_ = GET_PARAM(1);
    bit_depth_ = static_cast<avm_bit_depth_t>(GET_PARAM(2));
    mask_ = (1 << bit_depth_) - 1;
    filt_width_neg_ = GET_PARAM(3);
    filt_width_pos_ = GET_PARAM(4);
    pitch_ = kNumCoeffs / 32;
  }

  uint8_t GetQpThresh(ACMRandom *rnd) {
    return static_cast<uint8_t>(rnd->PseudoUniform(120));
  }

  uint8_t GetSideThresh(ACMRandom *rnd) {
    return static_cast<uint8_t>(rnd->PseudoUniform(210));
  }

  void InitInput(uint16_t *s, uint16_t *ref_s, ACMRandom *rnd,
                 const uint8_t limit, const int mask, const int32_t p,
                 const int i, const uint16_t prob) {
    InitLoopFilterInput(s, ref_s, rnd, limit, mask, p, i, prob);
  }

 protected:
  void CheckResult();
  void RunSpeedTest();

 private:
  avm_bit_depth_t bit_depth_;
  LoopFilterFunc func_ref_;
  LoopFilterFunc func_test_;
  int filt_width_neg_;
  int filt_width_pos_;
  int pitch_;
  int mask_;
};

void LoopFilterTest::CheckResult() {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  const int count_test_block = number_of_iterations;
  DECLARE_ALIGNED(16, uint16_t, s[kNumCoeffs]);
  DECLARE_ALIGNED(16, uint16_t, ref_s[kNumCoeffs]);
  int err_count_total = 0;
  int first_failure = -1;
  for (int i = 0; i < count_test_block; ++i) {
    int err_count = 0;

    const uint16_t q_thr = GetQpThresh(&rnd);
    const uint16_t side_thr = GetSideThresh(&rnd);

    InitInput(s, ref_s, &rnd, 16, mask_, pitch_, i, 0xc0);

    func_ref_(ref_s + 8 + pitch_ * 8, pitch_, filt_width_neg_, filt_width_pos_,
              &q_thr, &side_thr, bit_depth_, 0, 0);
    func_test_(s + 8 + pitch_ * 8, pitch_, filt_width_neg_, filt_width_pos_,
               &q_thr, &side_thr, bit_depth_, 0, 0);
    for (int j = 0; j < kNumCoeffs; ++j) {
      err_count += ref_s[j] != s[j];
    }
    if (err_count && !err_count_total) {
      first_failure = i;
    }
    err_count_total += err_count;
  }
  EXPECT_EQ(0, err_count_total)
      << "Error: C output doesn't match SIMD loopfilter output. "
      << "First failed at test case " << first_failure;
}
TEST_P(LoopFilterTest, CheckResult) { CheckResult(); }

void LoopFilterTest::RunSpeedTest() {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  DECLARE_ALIGNED(16, uint16_t, s[kNumCoeffs]);
  DECLARE_ALIGNED(16, uint16_t, ref_s[kNumCoeffs]);

  const uint16_t q_thr = 120;
  const uint16_t side_thr = 210;

  // Generate a constant input block by setting limit = 1 in InitInput(). This
  // prevents filt_choice_highbd() from reducing the filter length, but results
  // in delta_m2 = 0. Thus, to measure the correct scaling comment out the early
  // return "if (is_delta_m2_zero == 0xFFFF) return" in
  // filt_generic_asym_highbd_hor_4px_sse4_1() and
  // filt_generic_asym_highbd_ver_4px_sse4_1().
  InitInput(s, ref_s, &rnd, 1, mask_, pitch_, 0, 0xff);

  avm_usec_timer timer;
  avm_usec_timer_start(&timer);
  for (int i = 0; i < kSpeedIterations; ++i) {
    func_ref_(ref_s + 8 + pitch_ * 8, pitch_, filt_width_neg_, filt_width_pos_,
              &q_thr, &side_thr, bit_depth_, 0, 0);
  }
  avm_usec_timer_mark(&timer);
  auto elapsed_time_c = avm_usec_timer_elapsed(&timer);

  avm_usec_timer_start(&timer);
  for (int i = 0; i < kSpeedIterations; ++i) {
    func_test_(s + 8 + pitch_ * 8, pitch_, filt_width_neg_, filt_width_pos_,
               &q_thr, &side_thr, bit_depth_, 0, 0);
  }
  avm_usec_timer_mark(&timer);
  auto elapsed_time_opt = avm_usec_timer_elapsed(&timer);

  float c_time_per_filter_block =
      (float)1000.0 * elapsed_time_c / (kSpeedIterations);
  float opt_time_per_filter_block =
      (float)1000.0 * elapsed_time_opt / (kSpeedIterations);
  float scaling = c_time_per_filter_block / opt_time_per_filter_block;
  printf("\n filt_width_neg=%d, filt_width_pos=%d, bit_depth=%d",
         filt_width_neg_, filt_width_pos_, bit_depth_);
  printf(
      "\n c_time_per_filter_block=%10.5f, opt_time_per_filter_block=%10.5f,"
      "  scaling=%f",
      c_time_per_filter_block, opt_time_per_filter_block, scaling);
}
TEST_P(LoopFilterTest, DISABLED_RunSpeedTest) { RunSpeedTest(); }

typedef void (*LoopFilterDualFunc)(uint16_t *s, int pitch, int filt_width_neg0,
                                   int filt_width_pos0,
                                   const uint16_t *q_thresh0,
                                   const uint16_t *side_thresh0,
                                   int filt_width_neg1, int filt_width_pos1,
                                   const uint16_t *q_thresh1,
                                   const uint16_t *side_thresh1, int bd);

// <test_func, ref_func, bit_depth>
typedef tuple<LoopFilterDualFunc, LoopFilterDualFunc, int> DualParams;

// Two adjacent 4-sample segments of a horizontal edge with independent
// parameters: every pair
// of filter lengths, random thresholds (zero included), on rough and on smooth
// data so that every filter length is chosen, often a different one per
// segment.
class LoopFilterDualTest : public ::testing::TestWithParam<DualParams> {
 public:
  virtual void SetUp() {
    func_test_ = GET_PARAM(0);
    func_ref_ = GET_PARAM(1);
    bit_depth_ = GET_PARAM(2);
  }

 protected:
  // Smooth columns with a step at the edge (row 16 of the 32 rows); the
  // noise and the step differ between groups of 4 columns.
  void InitSmooth(uint16_t *s, ACMRandom *rnd, int pitch) {
    const int mask = (1 << bit_depth_) - 1;
    for (int g = 0; g < 8; ++g) {
      const int noise = 1 << rnd->PseudoUniform(bit_depth_ - 2);
      const int use_noise = rnd->PseudoUniform(3);
      const int step = rnd->PseudoUniform(4) ? rnd->PseudoUniform(64) : 0;
      const int slope = rnd->PseudoUniform(9) - 4;
      const int base = rnd->PseudoUniform(mask + 1);
      for (int c = g * 4; c < g * 4 + 4; ++c) {
        for (int r = 0; r < 32; ++r) {
          int v = base + slope * r + (r >= 16 ? step : 0);
          if (use_noise) v += rnd->PseudoUniform(noise);
          s[r * pitch + c] =
              static_cast<uint16_t>(std::min(std::max(v, 0), mask));
        }
      }
    }
  }

  void CheckResult() {
    static const int kNeg[] = { 0, 4, 6, 8, 10, 14, 18 };
    static const int kPos[] = { 4, 8, 10, 14, 18 };
    const int kIters = 24;
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    DECLARE_ALIGNED(16, uint16_t, s[kNumCoeffs]);
    DECLARE_ALIGNED(16, uint16_t, ref_s[kNumCoeffs]);
    const int pitch = 32;
    const int mask = (1 << bit_depth_) - 1;
    int err_count_total = 0;
    for (int n0 : kNeg) {
      for (int p0 : kPos) {
        for (int n1 : kNeg) {
          for (int p1 : kPos) {
            for (int i = 0; i < kIters; ++i) {
              uint16_t q[2], side[2];
              for (int h = 0; h < 2; ++h) {
                q[h] = static_cast<uint16_t>(rnd.PseudoUniform(120) *
                                             (1 << (bit_depth_ - 8)));
                side[h] = static_cast<uint16_t>(rnd.PseudoUniform(210) *
                                                (1 << (bit_depth_ - 8)));
              }
              if (i & 1) {
                InitSmooth(s, &rnd, pitch);
                memcpy(ref_s, s, sizeof(s));
              } else {
                InitLoopFilterInput(s, ref_s, &rnd, 16, mask, pitch, i / 2,
                                    0xc0);
              }
              // The edge is at row 16, columns 8..15.
              const int off = 8 + pitch * 16;
              uint16_t *const t = s + off;
              uint16_t *const r = ref_s + off;
              func_ref_(r, pitch, n0, p0, &q[0], &side[0], n1, p1, &q[1],
                        &side[1], bit_depth_);
              func_test_(t, pitch, n0, p0, &q[0], &side[0], n1, p1, &q[1],
                         &side[1], bit_depth_);
              int err = 0;
              for (int j = 0; j < kNumCoeffs; ++j) err += ref_s[j] != s[j];
              if (err && !err_count_total) {
                printf(
                    "first failure: bd %d neg0 %d pos0 %d neg1 %d pos1 %d "
                    "iter %d\n",
                    bit_depth_, n0, p0, n1, p1, i);
              }
              err_count_total += err;
            }
          }
        }
      }
    }
    EXPECT_EQ(0, err_count_total)
        << "Error: C output doesn't match SIMD dual loopfilter output.";
  }

  LoopFilterDualFunc func_test_;
  LoopFilterDualFunc func_ref_;
  int bit_depth_;
};
TEST_P(LoopFilterDualTest, CheckResult) { CheckResult(); }

#if HAVE_SSE4_1
const Params kLoopFilterTest[] = {
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 4, 4),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 6, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 8, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 6, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 10, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 14, 14),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 14, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 18, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 4, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 10, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 8, 18, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 4, 4),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 6, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 8, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 6, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 10, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 14, 14),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 14, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 18, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 4, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 10, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 10, 18, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 4, 4),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 6, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 8, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 6, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 10, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 14, 14),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 14, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 18, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 4, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 10, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_sse4_1,
             &avm_highbd_lpf_horizontal_generic_c, 12, 18, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 4, 4),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 6, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 8, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 6, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 10, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 14, 14),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 14, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 18, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 4, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 10, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 8, 18, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 4, 4),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 6, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 8, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 6, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 10, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 14, 14),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 14, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 18, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 4, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 10, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 10, 18, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 4, 4),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 6, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 8, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 6, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 10, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 14, 14),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 14, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 18, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 4, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 10, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_sse4_1,
             &avm_highbd_lpf_vertical_generic_c, 12, 18, 8),
};

INSTANTIATE_TEST_SUITE_P(SSE4_1, LoopFilterTest,
                         ::testing::ValuesIn(kLoopFilterTest));

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, LoopFilterDualTest,
    ::testing::Values(make_tuple(&avm_highbd_lpf_horizontal_generic_dual_sse4_1,
                                 &avm_highbd_lpf_horizontal_generic_dual_c, 8),
                      make_tuple(&avm_highbd_lpf_horizontal_generic_dual_sse4_1,
                                 &avm_highbd_lpf_horizontal_generic_dual_c, 10),
                      make_tuple(&avm_highbd_lpf_horizontal_generic_dual_sse4_1,
                                 &avm_highbd_lpf_horizontal_generic_dual_c,
                                 12)));
#endif  // HAVE_SSE4_1

#if HAVE_NEON
const Params kLoopFilterTestNeon[] = {
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 4, 4),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 6, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 8, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 6, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 10, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 14, 14),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 14, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 18, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 4, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 10, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 8, 18, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 4, 4),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 6, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 8, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 6, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 10, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 14, 14),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 14, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 18, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 4, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 10, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 10, 18, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 4, 4),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 6, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 8, 8),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 6, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 10, 10),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 14, 14),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 14, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 18, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 4, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 10, 18),
  make_tuple(&avm_highbd_lpf_horizontal_generic_neon,
             &avm_highbd_lpf_horizontal_generic_c, 12, 18, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 4, 4),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 6, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 8, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 6, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 10, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 14, 14),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 14, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 18, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 4, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 10, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 8, 18, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 4, 4),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 6, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 8, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 6, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 10, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 14, 14),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 14, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 18, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 4, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 10, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 10, 18, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 4, 4),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 6, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 8, 8),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 6, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 10, 10),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 14, 14),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 14, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 18, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 4, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 10, 18),
  make_tuple(&avm_highbd_lpf_vertical_generic_neon,
             &avm_highbd_lpf_vertical_generic_c, 12, 18, 8),
};

INSTANTIATE_TEST_SUITE_P(NEON, LoopFilterTest,
                         ::testing::ValuesIn(kLoopFilterTestNeon));

INSTANTIATE_TEST_SUITE_P(
    NEON, LoopFilterDualTest,
    ::testing::Values(make_tuple(&avm_highbd_lpf_horizontal_generic_dual_neon,
                                 &avm_highbd_lpf_horizontal_generic_dual_c, 8),
                      make_tuple(&avm_highbd_lpf_horizontal_generic_dual_neon,
                                 &avm_highbd_lpf_horizontal_generic_dual_c, 10),
                      make_tuple(&avm_highbd_lpf_horizontal_generic_dual_neon,
                                 &avm_highbd_lpf_horizontal_generic_dual_c,
                                 12)));
#endif  // HAVE_NEON

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LoopFilterTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LoopFilterDualTest);
}  // namespace
