#pragma once

// ninfer::ops - position block sums of a contiguous BF16 [W, T] value matrix.
// Implements: include/ninfer/ops/block_sum.h

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void block_sum_by_position_kernel(const __nv_bfloat16* value, float* sums,
                                              std::int32_t width, std::int32_t tokens,
                                              std::int32_t first_position,
                                              std::int32_t block_tokens, std::int32_t block_base,
                                              std::int32_t block_count) {
    const std::int32_t w = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (w >= width) { return; }

    const std::int64_t width64 = width;
    for (std::int32_t b = 0; b < block_count; ++b) {
        const std::int64_t block_first = (static_cast<std::int64_t>(block_base) + b) * block_tokens;
        std::int64_t t_begin           = block_first - first_position;
        std::int64_t t_end             = t_begin + block_tokens;
        if (t_begin < 0) { t_begin = 0; }
        if (t_end > tokens) { t_end = tokens; }

        float acc = 0.0f;
        for (std::int64_t t = t_begin; t < t_end; ++t) {
            acc += __bfloat162float(value[width64 * t + w]);
        }
        sums[width64 * b + w] = acc;
    }
}

} // namespace ninfer::ops