#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: block_sum_by_position
 *
 * Math / indexing:
 *   Token t of a [W, T] value matrix occupies the sequence position first_position + t, and its
 *   position block is
 *       block(t) = (first_position + t) / block_tokens                     (integer division).
 *   Writing
 *       block_base = first_position / block_tokens,
 *       block_count = (first_position + T - 1) / block_tokens - block_base + 1,
 *   the Op computes, for every w in [0, W) and b in [0, block_count),
 *       sums[w, b] = sum over t in [0, T) with block(t) == block_base + b of value[w, t].
 *   The declared blocks partition [0, T), so each token contributes exactly once and no declared
 *   column is empty.
 *
 * Logical shapes:
 *   value is a contiguous BF16 matrix [W, T] with element (w, t) at offset w + W*t.
 *   sums is a contiguous FP32 matrix [W, block_count] with element (w, b) at offset w + W*b.
 *   W, T, and block_count are positive; first_position is nonnegative and block_tokens positive.
 *
 * Supported domain:
 *   BF16 value, FP32 sums, both contiguous and non-overlapping. Any positive W, T, and
 *   block_tokens representable by the views, and any first_position with
 *   first_position + T - 1 <= INT32_MAX. Positions are absolute sequence token coordinates, not
 *   window rows, attention-cache rows, or RoPE positions.
 *
 * Numeric:
 *   Each BF16 element is read as its exact FP32 value. The observable result is the FP32 sum of a
 *   block; reduction association, staging precision, and accumulator dtype are implementation
 *   choices and are not part of the contract.
 *
 * Effects:
 *   Completely overwrites sums. value is unchanged. sums and value must not overlap.
 *
 * Workspace:
 *   None. The Op has no state side effect beyond writing sums.
 *
 * Execution:
 *   Asynchronous on the supplied stream. The launch reads only device data the caller supplied,
 *   allocates nothing, and performs no host synchronization, so it is valid inside a CUDA Graph
 *   capture.
 */
void block_sum_by_position(const Tensor& value, Tensor& sums, std::int32_t first_position,
                           std::int32_t block_tokens, cudaStream_t stream);

/**
 * The block_count block_sum_by_position writes for a call whose value matrix has `tokens`
 * columns starting at first_position.
 *
 * Same position and block_tokens domain as the Op entry; it performs no device access.
 */
std::int32_t block_sum_by_position_blocks(std::int32_t first_position, std::int32_t tokens,
                                          std::int32_t block_tokens);

} // namespace ninfer::ops