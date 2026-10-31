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

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "config/avm_config.h"
#include "config/avm_dsp_rtcd.h"

#include "test/acm_random.h"

using libavm_test::ACMRandom;

namespace {

typedef unsigned int (*HighbdAvgFunc)(const uint16_t *s, int p);

class HighbdAvg8x8Test : public ::testing::TestWithParam<HighbdAvgFunc> {
 protected:
  // Compares against the C version on blocks at random offsets of a strided
  // buffer, with random pixels of 8 to 12 bits and with all pixels at the
  // maximum of each bit depth.
  void Check(int bit_depth, bool max) {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int stride = 72;
    uint16_t buf[stride * 16];
    const uint16_t mask = (1 << bit_depth) - 1;
    for (int iter = 0; iter < 1000; ++iter) {
      for (int i = 0; i < stride * 16; ++i)
        buf[i] = max ? mask : rnd.Rand16() & mask;
      const uint16_t *s =
          buf + rnd.PseudoUniform(8) * stride + rnd.PseudoUniform(64);
      ASSERT_EQ(avm_highbd_avg_8x8_c(s, stride), GetParam()(s, stride))
          << "bit depth " << bit_depth << ", iteration " << iter;
    }
  }
};

TEST_P(HighbdAvg8x8Test, MatchesC) {
  for (int bd = 8; bd <= 12; bd += 2) {
    Check(bd, false);
    Check(bd, true);
  }
}

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(SSE2, HighbdAvg8x8Test,
                         ::testing::Values(&avm_highbd_avg_8x8_sse2));
#endif
#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, HighbdAvg8x8Test,
                         ::testing::Values(&avm_highbd_avg_8x8_neon));
#endif

}  // namespace
