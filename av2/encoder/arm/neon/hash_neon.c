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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "config/av2_rtcd.h"

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

// CRC-32C with the Armv8 CRC32C instructions where the target has them (the
// CRC32 extension: Armv8.1-A and later, every Apple silicon core); they
// compute the same CRC-32C (polynomial 0x1EDC6F41) as the table-driven
// av2_get_crc32c_value_c(), which other targets keep using.
uint32_t av2_get_crc32c_value_neon(void *crc_calculator, uint8_t *p,
                                   size_t len) {
#if defined(__ARM_FEATURE_CRC32)
  (void)crc_calculator;
  const uint8_t *buf = p;
  uint32_t crc = 0xFFFFFFFF;
  for (; len >= 8; len -= 8, buf += 8) {
    uint64_t v;
    memcpy(&v, buf, sizeof(v));
    crc = __crc32cd(crc, v);
  }
  if (len >= 4) {
    uint32_t v;
    memcpy(&v, buf, sizeof(v));
    crc = __crc32cw(crc, v);
    len -= 4;
    buf += 4;
  }
  for (; len > 0; len--) crc = __crc32cb(crc, *buf++);
  return crc ^ 0xFFFFFFFF;
#else
  return av2_get_crc32c_value_c(crc_calculator, p, len);
#endif
}
