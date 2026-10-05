#include "ninfer/ops/block_sum.h"

#include "ops/launcher/block_sum.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_matrix(const Tensor& tensor, DType dtype, const char* label) {
    if (tensor.dtype != dtype || tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("block_sum_by_position: ") + label +
                                    " must be a non-empty contiguous rank-2 matrix");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

struct BlockRange {
    std::int32_t base;
    std::int32_t count;
};

BlockRange resolve_block_range(std::int32_t first_position, std::int32_t tokens,
                               std::int32_t block_tokens) {
    if (tokens <= 0 || block_tokens <= 0 || first_position < 0 ||
        first_position > std::numeric_limits<std::int32_t>::max() - (tokens - 1)) {
        throw std::invalid_argument(
            "block_sum_by_position: position range must stay inside nonnegative I32");
    }
    const std::int64_t base   = static_cast<std::int64_t>(first_position) / block_tokens;
    const std::int64_t top    =
        (static_cast<std::int64_t>(first_position) + tokens - 1) / block_tokens;
    const std::int64_t blocks = top - base + 1;
    if (blocks > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("block_sum_by_position: block count overflows I32");
    }
    return {static_cast<std::int32_t>(base), static_cast<std::int32_t>(blocks)};
}

} // namespace

std::int32_t block_sum_by_position_blocks(std::int32_t first_position, std::int32_t tokens,
                                          std::int32_t block_tokens) {
    return resolve_block_range(first_position, tokens, block_tokens).count;
}

void block_sum_by_position(const Tensor& value, Tensor& sums, std::int32_t first_position,
                           std::int32_t block_tokens, cudaStream_t stream) {
    require_matrix(value, DType::BF16, "value");
    require_matrix(sums, DType::FP32, "sums");
    const BlockRange range = resolve_block_range(first_position, value.ne[1], block_tokens);
    if (sums.ne[0] != value.ne[0] || sums.ne[1] != range.count) {
        throw std::invalid_argument(
            "block_sum_by_position: sums must be [value rows, derived block count]");
    }
    if (overlaps(value, sums)) {
        throw std::invalid_argument("block_sum_by_position: sums must not overlap value");
    }
    detail::block_sum_by_position_launch(value, sums, first_position, block_tokens, range.base,
                                         range.count, stream);
}

} // namespace ninfer::ops