#include "ops/launcher/block_sum.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/block_sum.cuh"

namespace ninfer::ops::detail {

void block_sum_by_position_launch(const Tensor& value, Tensor& sums, std::int32_t first_position,
                                  std::int32_t block_tokens, std::int32_t block_base,
                                  std::int32_t block_count, cudaStream_t stream) {
    constexpr int block = 256;
    const int grid      = div_up(value.ne[0], block);
    block_sum_by_position_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(value.data), static_cast<float*>(sums.data), value.ne[0],
        value.ne[1], first_position, block_tokens, block_base, block_count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail