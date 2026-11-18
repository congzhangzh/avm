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

#ifndef AVM_AV2_ENCODER_HASH_MOTION_H_
#define AVM_AV2_ENCODER_HASH_MOTION_H_

#include "config/avm_config.h"

#include "avm/avm_integer.h"
#include "avm_scale/yv12config.h"
#include "av2/encoder/hash.h"
#ifdef __cplusplus
extern "C" {
#endif

// Block size used for force_integer_mv decisions
#define FORCE_INT_MV_DECISION_BLOCK_SIZE 8

// store a block's hash info.
// x and y are the position from the top left of the picture
// hash_value2 is used to store the second hash value
typedef struct _block_hash {
  int16_t x;
  int16_t y;
  uint32_t hash_value2;
} block_hash;

// Block sizes 4, 8, ..., 256: the top bits of a hash_value1.
#define HASH_TABLE_NUM_SIZES 7
// The low bits of a hash_value1 (a block's first hash) select its bucket.
#define HASH_TABLE_KEY_BITS 16
#define HASH_TABLE_NUM_KEYS (1 << HASH_TABLE_KEY_BITS)

// The blocks of each block size, grouped by bucket: those of bucket k are
// entries[s][start[s][k]] .. entries[s][start[s][k + 1] - 1], in the order
// they were added (column by column, top to bottom). start[s] is NULL until
// size s is added.
typedef struct _hash_table {
  uint32_t *start[HASH_TABLE_NUM_SIZES];
  block_hash *entries[HASH_TABLE_NUM_SIZES];
  // With CRC32C (a block's two hashes are one value), the table also keeps
  // that value for the blocks of the two largest sizes it was built for at
  // every position (y * frame_width + x) of the frame (frame_buf) it was built
  // from, so that the search looks up a block of that frame instead of hashing
  // it again. block_crc[s] is NULL when not kept; the table owns crc_buf.
  const uint32_t *block_crc[HASH_TABLE_NUM_SIZES];
  uint32_t *crc_buf[2];
  const uint16_t *frame_buf;
  int frame_stride;
  int frame_width;
  int frame_height;
} hash_table;

// The index of a block size (4, 8, ..., 256) in hash_table, -1 for others.
static INLINE int av2_hash_block_size_index(int block_size) {
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

struct intrabc_hash_info;

typedef struct intrabc_hash_info {
  // buffer for hash value calculation of a block
  // used only in av2_get_block_hash_value()
  // [first hash/second hash]
  // [two buffers used ping-pong]
  uint32_t *hash_value_buffer[2][2];
  hash_table intrabc_hash_table;

  CRC_CALCULATOR crc_calculator1;
  CRC_CALCULATOR crc_calculator2;
  int g_crc_initialized;
  // Realtime: one CRC32C per block (SSE4.2 where available) instead of the two
  // table-driven 24-bit CRCs, as libaom does; the second hash is then the full
  // CRC32C and the first one its low bits. Set before av2_hash_table_init().
  int use_crc32c;
  CRC32C crc32c_calculator;
} IntraBCHashInfo;

void av2_hash_table_init(IntraBCHashInfo *intra_bc_hash_info);
void av2_hash_table_clear_all(hash_table *p_hash_table);
void av2_hash_table_destroy(hash_table *p_hash_table);
void av2_hash_table_create(hash_table *p_hash_table);
int32_t av2_hash_table_count(const hash_table *p_hash_table,
                             uint32_t hash_value);
// The first of the av2_hash_table_count() blocks with this hash_value1; they
// follow it contiguously.
const block_hash *av2_hash_get_first_entry(const hash_table *p_hash_table,
                                           uint32_t hash_value);
void av2_generate_block_2x2_hash_value(IntraBCHashInfo *intra_bc_hash_info,
                                       const YV12_BUFFER_CONFIG *picture,
                                       uint32_t *pic_block_hash[2],
                                       int8_t *pic_block_same_info[3]);
void av2_generate_block_hash_value(IntraBCHashInfo *intra_bc_hash_info,
                                   const YV12_BUFFER_CONFIG *picture,
                                   int block_size,
                                   uint32_t *src_pic_block_hash[2],
                                   uint32_t *dst_pic_block_hash[2],
                                   int8_t *src_pic_block_same_info[3],
                                   int8_t *dst_pic_block_same_info[3]);
void av2_add_to_hash_map_by_row_with_precal_data(hash_table *p_hash_table,
                                                 uint32_t *pic_hash[2],
                                                 int8_t *pic_is_same,
                                                 int pic_width, int pic_height,
                                                 int block_size);

// The same work split into parts that can run on several threads; the result
// equals the functions above.
// av2_generate_block_2x2_hash_value() and av2_generate_block_hash_value() for
// the block positions of rows [y_start, y_stop) only. The block hash of a row
// reads the previous size's values of rows up to block_size / 2 below it.
void av2_generate_block_2x2_hash_value_rows(IntraBCHashInfo *intra_bc_hash_info,
                                            const YV12_BUFFER_CONFIG *picture,
                                            uint32_t *pic_block_hash[2],
                                            int8_t *pic_block_same_info[3],
                                            int y_start, int y_stop);
void av2_generate_block_hash_value_rows(
    IntraBCHashInfo *intra_bc_hash_info, const YV12_BUFFER_CONFIG *picture,
    int block_size, uint32_t *src_pic_block_hash[2],
    uint32_t *dst_pic_block_hash[2], int8_t *src_pic_block_same_info[3],
    int8_t *dst_pic_block_same_info[3], int y_start, int y_stop);
// Adding one block size in three steps:
// 1. av2_hash_count_cols() counts, per bucket, the blocks to add whose column
//    is in [x_start, x_stop), into counts[HASH_TABLE_NUM_KEYS] (zeroed by it).
// 2. av2_hash_table_alloc_size() takes the counts of consecutive column ranges
//    (counts[0] the leftmost), allocates the size and turns each counts[i]
//    into the first index of range i's blocks in each bucket.
// 3. av2_hash_fill_cols() stores the blocks of one column range at those
//    indices; the ranges can be filled in any order or at the same time.
void av2_hash_count_cols(const uint32_t *pic_hash, const int8_t *pic_is_same,
                         int pic_width, int pic_height, int block_size,
                         int x_start, int x_stop, uint32_t *counts);
int av2_hash_table_alloc_size(hash_table *p_hash_table, int block_size,
                              uint32_t **counts, int num_ranges);
void av2_hash_fill_cols(hash_table *p_hash_table, uint32_t *pic_hash[2],
                        const int8_t *pic_is_same, int pic_width,
                        int pic_height, int block_size, int x_start, int x_stop,
                        uint32_t *next_index);

// check whether the block starts from (x_start, y_start) with the size of
// block_size x block_size has the same color in all rows
int av2_hash_is_horizontal_perfect(const YV12_BUFFER_CONFIG *picture,
                                   int block_size, int x_start, int y_start);
// check whether the block starts from (x_start, y_start) with the size of
// block_size x block_size has the same color in all columns
int av2_hash_is_vertical_perfect(const YV12_BUFFER_CONFIG *picture,
                                 int block_size, int x_start, int y_start);

// Gets the two hashes of the block_size block at y_src as
// av2_get_block_hash_value() computes them, when the table kept them for this
// block of its frame: y_src is frame_buf + y_pos * stride + x_pos. Returns 0
// otherwise.
int av2_hash_get_frame_block_hash(const hash_table *p_hash_table,
                                  const uint16_t *y_src, int stride, int x_pos,
                                  int y_pos, int block_size,
                                  uint32_t *hash_value1, uint32_t *hash_value2);

void av2_get_block_hash_value(IntraBCHashInfo *intrabc_hash_info,
                              const uint16_t *y_src, int stride, int block_size,
                              uint32_t *hash_value1, uint32_t *hash_value2);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AVM_AV2_ENCODER_HASH_MOTION_H_
