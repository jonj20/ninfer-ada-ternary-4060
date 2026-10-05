#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void block_sum_by_position_launch(const Tensor& value, Tensor& sums, std::int32_t first_position,
                                  std::int32_t block_tokens, std::int32_t block_base,
                                  std::int32_t block_count, cudaStream_t stream);

} // namespace ninfer::ops::detail