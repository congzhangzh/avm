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
#ifndef AVM_AV2_ENCODER_X86_HIGHBD_FWD_TXFM_AVX2_H_
#define AVM_AV2_ENCODER_X86_HIGHBD_FWD_TXFM_AVX2_H_

#ifdef __cplusplus
extern "C" {
#endif

// The 1-D passes of fwd_txfm_avx2(), shared with fwd_txfm_avx512().
void fwd_transform_1d_avx2(const int *src, int *dst, int shift, int line,
                           int skip_line, int zero_line,
                           const int tx_type_index, const int size_index);
void fwd_txfm_dct2_size64_avx2(const int *src, int *dst, int shift, int line,
                               int skip_line, int zero_line);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AVM_AV2_ENCODER_X86_HIGHBD_FWD_TXFM_AVX2_H_
