/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
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
#include <limits.h>
#include <string.h>

#include "config/av2_rtcd.h"

#include "av2/encoder/block.h"
#include "av2/encoder/hash.h"
#include "av2/encoder/hash_motion.h"

#define kSrcBits 16
static_assert(kSrcBits == HASH_TABLE_KEY_BITS, "bucket key bits");

static void get_pixels_in_1D_short_array_by_block_2x2(const uint16_t *y_src,
                                                      int stride,
                                                      uint16_t *p_pixels_in1D) {
  const uint16_t *p_pel = y_src;
  int index = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      p_pixels_in1D[index++] = p_pel[j];
    }
    p_pel += stride;
  }
}

static int is_block16_2x2_row_same_value(const uint16_t *p) {
  if (p[0] != p[1] || p[2] != p[3]) {
    return 0;
  }
  return 1;
}

static int is_block16_2x2_col_same_value(const uint16_t *p) {
  if ((p[0] != p[2]) || (p[1] != p[3])) {
    return 0;
  }
  return 1;
}

// the hash value (hash_value1 consists two parts, the first 3 bits relate to
// the block size and the remaining 16 bits are the crc values. This function
// is used to get the first 3 bits.
static int hash_block_size_to_index(int block_size) {
  switch (block_size) {
    case 4: return 0;
    case 8: return 1;
    case 16: return 2;
    case 32: return 3;
    case 64: return 4;
    case 128: return 5;
    case 256: return 6;
    default: return -1;
  }
}

void av2_hash_table_init(IntraBCHashInfo *intrabc_hash_info) {
  if (!intrabc_hash_info->g_crc_initialized) {
    av2_crc_calculator_init(&intrabc_hash_info->crc_calculator1, 24, 0x5D6DCB);
    av2_crc_calculator_init(&intrabc_hash_info->crc_calculator2, 24, 0x864CFB);
    av2_crc32c_calculator_init(&intrabc_hash_info->crc32c_calculator);
    intrabc_hash_info->g_crc_initialized = 1;
  }
  hash_table *const t = &intrabc_hash_info->intrabc_hash_table;
  memset(t, 0, sizeof(*t));
}

void av2_hash_table_clear_all(hash_table *p_hash_table) {
  for (int s = 0; s < HASH_TABLE_NUM_SIZES; s++) {
    avm_free(p_hash_table->start[s]);
    avm_free(p_hash_table->entries[s]);
    p_hash_table->start[s] = NULL;
    p_hash_table->entries[s] = NULL;
    p_hash_table->block_crc[s] = NULL;
  }
  for (int k = 0; k < 2; k++) {
    avm_free(p_hash_table->crc_buf[k]);
    p_hash_table->crc_buf[k] = NULL;
  }
  p_hash_table->frame_buf = NULL;
}

void av2_hash_table_destroy(hash_table *p_hash_table) {
  av2_hash_table_clear_all(p_hash_table);
}

void av2_hash_table_create(hash_table *p_hash_table) {
  av2_hash_table_clear_all(p_hash_table);
}

int32_t av2_hash_table_count(const hash_table *p_hash_table,
                             uint32_t hash_value) {
  const uint32_t *const start = p_hash_table->start[hash_value >> kSrcBits];
  if (start == NULL) return 0;
  const uint32_t key = hash_value & (HASH_TABLE_NUM_KEYS - 1);
  return (int32_t)(start[key + 1] - start[key]);
}

const block_hash *av2_hash_get_first_entry(const hash_table *p_hash_table,
                                           uint32_t hash_value) {
  assert(av2_hash_table_count(p_hash_table, hash_value) > 0);
  const int s = hash_value >> kSrcBits;
  const uint32_t key = hash_value & (HASH_TABLE_NUM_KEYS - 1);
  return p_hash_table->entries[s] + p_hash_table->start[s][key];
}

void av2_generate_block_2x2_hash_value_rows(IntraBCHashInfo *intrabc_hash_info,
                                            const YV12_BUFFER_CONFIG *picture,
                                            uint32_t *pic_block_hash[2],
                                            int8_t *pic_block_same_info[3],
                                            int y_start, int y_stop) {
  const int width = 2;
  const int height = 2;
  const int pic_width = picture->y_crop_width;
  const int x_end = picture->y_crop_width - width + 1;
  const int y_end = AVMMIN(y_stop, picture->y_crop_height - height + 1);
  CRC_CALCULATOR *calc_1 = &intrabc_hash_info->crc_calculator1;
  CRC_CALCULATOR *calc_2 = &intrabc_hash_info->crc_calculator2;
  const int use_crc32c = intrabc_hash_info->use_crc32c;
  void *const crc32c = &intrabc_hash_info->crc32c_calculator;

  const int length = width * 2;
  uint16_t p[4];
  for (int y_pos = AVMMAX(y_start, 0); y_pos < y_end; y_pos++) {
    int pos = y_pos * pic_width;
    for (int x_pos = 0; x_pos < x_end; x_pos++) {
      get_pixels_in_1D_short_array_by_block_2x2(
          picture->y_buffer + y_pos * picture->y_stride + x_pos,
          picture->y_stride, p);
      pic_block_same_info[0][pos] = is_block16_2x2_row_same_value(p);
      pic_block_same_info[1][pos] = is_block16_2x2_col_same_value(p);

      if (use_crc32c) {
        pic_block_hash[0][pos] = pic_block_hash[1][pos] =
            av2_get_crc32c_value(crc32c, (uint8_t *)p, length * sizeof(p[0]));
      } else {
        pic_block_hash[0][pos] =
            av2_get_crc_value(calc_1, (uint8_t *)p, length * sizeof(p[0]));
        pic_block_hash[1][pos] =
            av2_get_crc_value(calc_2, (uint8_t *)p, length * sizeof(p[0]));
      }
      pos++;
    }
  }
}

void av2_generate_block_2x2_hash_value(IntraBCHashInfo *intrabc_hash_info,
                                       const YV12_BUFFER_CONFIG *picture,
                                       uint32_t *pic_block_hash[2],
                                       int8_t *pic_block_same_info[3]) {
  av2_generate_block_2x2_hash_value_rows(intrabc_hash_info, picture,
                                         pic_block_hash, pic_block_same_info, 0,
                                         INT_MAX);
}

void av2_generate_block_hash_value_rows(
    IntraBCHashInfo *intrabc_hash_info, const YV12_BUFFER_CONFIG *picture,
    int block_size, uint32_t *src_pic_block_hash[2],
    uint32_t *dst_pic_block_hash[2], int8_t *src_pic_block_same_info[3],
    int8_t *dst_pic_block_same_info[3], int y_start, int y_stop) {
  CRC_CALCULATOR *calc_1 = &intrabc_hash_info->crc_calculator1;
  CRC_CALCULATOR *calc_2 = &intrabc_hash_info->crc_calculator2;
  const int use_crc32c = intrabc_hash_info->use_crc32c;
  void *const crc32c = &intrabc_hash_info->crc32c_calculator;

  const int pic_width = picture->y_crop_width;
  const int x_end = picture->y_crop_width - block_size + 1;
  const int y_end = AVMMIN(y_stop, picture->y_crop_height - block_size + 1);
  y_start = AVMMAX(y_start, 0);

  const int src_size = block_size >> 1;
  const int quad_size = block_size >> 2;

  uint32_t p[4];
  const int length = sizeof(p);

  for (int y_pos = y_start; y_pos < y_end; y_pos++) {
    int pos = y_pos * pic_width;
    for (int x_pos = 0; x_pos < x_end; x_pos++) {
      p[0] = src_pic_block_hash[0][pos];
      p[1] = src_pic_block_hash[0][pos + src_size];
      p[2] = src_pic_block_hash[0][pos + src_size * pic_width];
      p[3] = src_pic_block_hash[0][pos + src_size * pic_width + src_size];
      if (use_crc32c) {
        dst_pic_block_hash[0][pos] = dst_pic_block_hash[1][pos] =
            av2_get_crc32c_value(crc32c, (uint8_t *)p, length);
      } else {
        dst_pic_block_hash[0][pos] =
            av2_get_crc_value(calc_1, (uint8_t *)p, length);

        p[0] = src_pic_block_hash[1][pos];
        p[1] = src_pic_block_hash[1][pos + src_size];
        p[2] = src_pic_block_hash[1][pos + src_size * pic_width];
        p[3] = src_pic_block_hash[1][pos + src_size * pic_width + src_size];
        dst_pic_block_hash[1][pos] =
            av2_get_crc_value(calc_2, (uint8_t *)p, length);
      }

      dst_pic_block_same_info[0][pos] =
          src_pic_block_same_info[0][pos] &&
          src_pic_block_same_info[0][pos + quad_size] &&
          src_pic_block_same_info[0][pos + src_size] &&
          src_pic_block_same_info[0][pos + src_size * pic_width] &&
          src_pic_block_same_info[0][pos + src_size * pic_width + quad_size] &&
          src_pic_block_same_info[0][pos + src_size * pic_width + src_size];

      dst_pic_block_same_info[1][pos] =
          src_pic_block_same_info[1][pos] &&
          src_pic_block_same_info[1][pos + src_size] &&
          src_pic_block_same_info[1][pos + quad_size * pic_width] &&
          src_pic_block_same_info[1][pos + quad_size * pic_width + src_size] &&
          src_pic_block_same_info[1][pos + src_size * pic_width] &&
          src_pic_block_same_info[1][pos + src_size * pic_width + src_size];
      pos++;
    }
  }

  if (block_size >= 4) {
    const int size_minus_1 = block_size - 1;
    for (int y_pos = y_start; y_pos < y_end; y_pos++) {
      int pos = y_pos * pic_width;
      for (int x_pos = 0; x_pos < x_end; x_pos++) {
        dst_pic_block_same_info[2][pos] =
            (!dst_pic_block_same_info[0][pos] &&
             !dst_pic_block_same_info[1][pos]) ||
            (((x_pos & size_minus_1) == 0) && ((y_pos & size_minus_1) == 0));
        pos++;
      }
    }
  }
}

void av2_generate_block_hash_value(IntraBCHashInfo *intrabc_hash_info,
                                   const YV12_BUFFER_CONFIG *picture,
                                   int block_size,
                                   uint32_t *src_pic_block_hash[2],
                                   uint32_t *dst_pic_block_hash[2],
                                   int8_t *src_pic_block_same_info[3],
                                   int8_t *dst_pic_block_same_info[3]) {
  av2_generate_block_hash_value_rows(intrabc_hash_info, picture, block_size,
                                     src_pic_block_hash, dst_pic_block_hash,
                                     src_pic_block_same_info,
                                     dst_pic_block_same_info, 0, INT_MAX);
}

void av2_hash_count_cols(const uint32_t *pic_hash, const int8_t *pic_is_same,
                         int pic_width, int pic_height, int block_size,
                         int x_start, int x_stop, uint32_t *counts) {
  const int x_end = AVMMIN(x_stop, pic_width - block_size + 1);
  const int y_end = pic_height - block_size + 1;
  memset(counts, 0, sizeof(*counts) * HASH_TABLE_NUM_KEYS);
  x_start = AVMMAX(x_start, 0);
  if (x_start >= x_end) return;
  // The order of counting does not matter: row by row.
  for (int y_pos = 0; y_pos < y_end; y_pos++) {
    const int row = y_pos * pic_width;
    for (int x_pos = x_start; x_pos < x_end; x_pos++) {
      if (pic_is_same[row + x_pos])
        counts[pic_hash[row + x_pos] & (HASH_TABLE_NUM_KEYS - 1)]++;
    }
  }
}

int av2_hash_table_alloc_size(hash_table *p_hash_table, int block_size,
                              uint32_t **counts, int num_ranges) {
  const int s = hash_block_size_to_index(block_size);
  assert(s >= 0);
  assert(p_hash_table->start[s] == NULL);
  uint32_t *const start =
      (uint32_t *)avm_malloc(sizeof(*start) * (HASH_TABLE_NUM_KEYS + 1));
  if (start == NULL) return 0;
  uint32_t total = 0;
  for (int k = 0; k < HASH_TABLE_NUM_KEYS; k++) {
    start[k] = total;
    for (int i = 0; i < num_ranges; i++) {
      const uint32_t n = counts[i][k];
      counts[i][k] = total;
      total += n;
    }
  }
  start[HASH_TABLE_NUM_KEYS] = total;
  block_hash *const entries =
      (block_hash *)avm_malloc(sizeof(*entries) * AVMMAX(total, 1));
  if (entries == NULL) {
    avm_free(start);
    return 0;
  }
  p_hash_table->start[s] = start;
  p_hash_table->entries[s] = entries;
  return 1;
}

void av2_hash_fill_cols(hash_table *p_hash_table, uint32_t *pic_hash[2],
                        const int8_t *pic_is_same, int pic_width,
                        int pic_height, int block_size, int x_start, int x_stop,
                        uint32_t *next_index) {
  const int x_end = AVMMIN(x_stop, pic_width - block_size + 1);
  const int y_end = pic_height - block_size + 1;
  block_hash *const entries =
      p_hash_table->entries[hash_block_size_to_index(block_size)];
  // Column by column, top to bottom: the order the search finds them in.
  for (int x_pos = AVMMAX(x_start, 0); x_pos < x_end; x_pos++) {
    for (int y_pos = 0; y_pos < y_end; y_pos++) {
      const int pos = y_pos * pic_width + x_pos;
      if (pic_is_same[pos]) {
        block_hash *const e = &entries[next_index[pic_hash[0][pos] &
                                                  (HASH_TABLE_NUM_KEYS - 1)]++];
        e->x = x_pos;
        e->y = y_pos;
        e->hash_value2 = pic_hash[1][pos];
      }
    }
  }
}

void av2_add_to_hash_map_by_row_with_precal_data(hash_table *p_hash_table,
                                                 uint32_t *pic_hash[2],
                                                 int8_t *pic_is_same,
                                                 int pic_width, int pic_height,
                                                 int block_size) {
  uint32_t *counts =
      (uint32_t *)avm_malloc(sizeof(*counts) * HASH_TABLE_NUM_KEYS);
  if (counts == NULL) return;
  av2_hash_count_cols(pic_hash[0], pic_is_same, pic_width, pic_height,
                      block_size, 0, INT_MAX, counts);
  if (av2_hash_table_alloc_size(p_hash_table, block_size, &counts, 1)) {
    av2_hash_fill_cols(p_hash_table, pic_hash, pic_is_same, pic_width,
                       pic_height, block_size, 0, INT_MAX, counts);
  }
  avm_free(counts);
}

int av2_hash_get_frame_block_hash(const hash_table *p_hash_table,
                                  const uint16_t *y_src, int stride, int x_pos,
                                  int y_pos, int block_size,
                                  uint32_t *hash_value1,
                                  uint32_t *hash_value2) {
  const int s = av2_hash_block_size_index(block_size);
  if (s < 0 || p_hash_table->block_crc[s] == NULL) return 0;
  const int width = p_hash_table->frame_width;
  if (stride != p_hash_table->frame_stride || x_pos < 0 || y_pos < 0 ||
      x_pos > width - block_size ||
      y_pos > p_hash_table->frame_height - block_size ||
      y_src != p_hash_table->frame_buf + (ptrdiff_t)y_pos * stride + x_pos)
    return 0;
  // The value the table build computed for this block from the same pixels,
  // in the same order as av2_get_block_hash_value(): 2x2 groups, then four
  // half-size blocks (top left, top right, bottom left, bottom right).
  const uint32_t crc = p_hash_table->block_crc[s][y_pos * width + x_pos];
  *hash_value1 = (crc & ((1 << kSrcBits) - 1)) + ((uint32_t)s << kSrcBits);
  *hash_value2 = crc;
  return 1;
}

int av2_hash_is_horizontal_perfect(const YV12_BUFFER_CONFIG *picture,
                                   int block_size, int x_start, int y_start) {
  const int stride = picture->y_stride;
  const uint16_t *p = picture->y_buffer + y_start * stride + x_start;

  for (int i = 0; i < block_size; i++) {
    for (int j = 1; j < block_size; j++) {
      if (p[j] != p[0]) {
        return 0;
      }
    }
    p += stride;
  }

  return 1;
}

int av2_hash_is_vertical_perfect(const YV12_BUFFER_CONFIG *picture,
                                 int block_size, int x_start, int y_start) {
  const int stride = picture->y_stride;
  const uint16_t *p = picture->y_buffer + y_start * stride + x_start;

  for (int i = 0; i < block_size; i++) {
    for (int j = 1; j < block_size; j++) {
      if (p[j * stride + i] != p[i]) {
        return 0;
      }
    }
  }
  return 1;
}

void av2_get_block_hash_value(IntraBCHashInfo *intrabc_hash_info,
                              const uint16_t *y_src, int stride, int block_size,
                              uint32_t *hash_value1, uint32_t *hash_value2) {
  int add_value = hash_block_size_to_index(block_size);
  assert(add_value >= 0);
  add_value <<= kSrcBits;
  const int crc_mask = (1 << kSrcBits) - 1;

  CRC_CALCULATOR *calc_1 = &intrabc_hash_info->crc_calculator1;
  CRC_CALCULATOR *calc_2 = &intrabc_hash_info->crc_calculator2;
  uint32_t **buf_1 = intrabc_hash_info->hash_value_buffer[0];
  uint32_t **buf_2 = intrabc_hash_info->hash_value_buffer[1];
  const int use_crc32c = intrabc_hash_info->use_crc32c;
  void *const crc32c = &intrabc_hash_info->crc32c_calculator;

  // 2x2 subblock hash values in current CU
  int sub_block_in_width = (block_size >> 1);
  uint16_t pixel_to_hash[4];
  for (int y_pos = 0; y_pos < block_size; y_pos += 2) {
    for (int x_pos = 0; x_pos < block_size; x_pos += 2) {
      int pos = (y_pos >> 1) * sub_block_in_width + (x_pos >> 1);
      get_pixels_in_1D_short_array_by_block_2x2(y_src + y_pos * stride + x_pos,
                                                stride, pixel_to_hash);
      assert(pos < AVM_BUFFER_SIZE_FOR_BLOCK_HASH);
      if (use_crc32c) {
        buf_1[0][pos] = buf_2[0][pos] = av2_get_crc32c_value(
            crc32c, (uint8_t *)pixel_to_hash, sizeof(pixel_to_hash));
      } else {
        buf_1[0][pos] = av2_get_crc_value(calc_1, (uint8_t *)pixel_to_hash,
                                          sizeof(pixel_to_hash));
        buf_2[0][pos] = av2_get_crc_value(calc_2, (uint8_t *)pixel_to_hash,
                                          sizeof(pixel_to_hash));
      }
    }
  }

  int src_sub_block_in_width = sub_block_in_width;
  sub_block_in_width >>= 1;

  int src_idx = 1;
  int dst_idx = 0;

  // 4x4 subblock hash values to current block hash values
  uint32_t to_hash[4];
  for (int sub_width = 4; sub_width <= block_size; sub_width *= 2) {
    src_idx = 1 - src_idx;
    dst_idx = 1 - dst_idx;

    int dst_pos = 0;
    for (int y_pos = 0; y_pos < sub_block_in_width; y_pos++) {
      for (int x_pos = 0; x_pos < sub_block_in_width; x_pos++) {
        int srcPos = (y_pos << 1) * src_sub_block_in_width + (x_pos << 1);

        assert(srcPos + 1 < AVM_BUFFER_SIZE_FOR_BLOCK_HASH);
        assert(srcPos + src_sub_block_in_width + 1 <
               AVM_BUFFER_SIZE_FOR_BLOCK_HASH);
        assert(dst_pos < AVM_BUFFER_SIZE_FOR_BLOCK_HASH);
        to_hash[0] = buf_1[src_idx][srcPos];
        to_hash[1] = buf_1[src_idx][srcPos + 1];
        to_hash[2] = buf_1[src_idx][srcPos + src_sub_block_in_width];
        to_hash[3] = buf_1[src_idx][srcPos + src_sub_block_in_width + 1];

        if (use_crc32c) {
          buf_1[dst_idx][dst_pos] = buf_2[dst_idx][dst_pos] =
              av2_get_crc32c_value(crc32c, (uint8_t *)to_hash,
                                   sizeof(to_hash));
          dst_pos++;
          continue;
        }
        buf_1[dst_idx][dst_pos] =
            av2_get_crc_value(calc_1, (uint8_t *)to_hash, sizeof(to_hash));

        to_hash[0] = buf_2[src_idx][srcPos];
        to_hash[1] = buf_2[src_idx][srcPos + 1];
        to_hash[2] = buf_2[src_idx][srcPos + src_sub_block_in_width];
        to_hash[3] = buf_2[src_idx][srcPos + src_sub_block_in_width + 1];
        buf_2[dst_idx][dst_pos] =
            av2_get_crc_value(calc_2, (uint8_t *)to_hash, sizeof(to_hash));
        dst_pos++;
      }
    }

    src_sub_block_in_width = sub_block_in_width;
    sub_block_in_width >>= 1;
  }

  *hash_value1 = (buf_1[dst_idx][0] & crc_mask) + add_value;
  *hash_value2 = buf_2[dst_idx][0];
}
