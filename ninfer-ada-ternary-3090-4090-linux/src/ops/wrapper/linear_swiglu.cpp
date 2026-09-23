#include "ninfer/ops/linear_swiglu.h"

#include "ninfer/ops/silu_mul.h"
#include "ops/linear/ternary/ternary_dispatch.h"
#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

} // namespace

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    if (qtype == QType::W8G32_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_swiglu workspace: W8 admits only A16");
        }
        (void)detail::w8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens});
        (void)detail::w8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::Q4G64_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_swiglu workspace: Q4 admits only A16");
        }
        return detail::q4_linear_swiglu_capacity_workspace_bytes(
            gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens, max_tokens);
    }
    if (qtype == QType::PTQ1_0_G128 || qtype == QType::PQ2_0_G128) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_swiglu workspace: ternary admits only A16");
        }
        // 折叠三元路线把整个融合的 gate/up 父权重投影进一个作用域内的 [2*intermediate, T]
        // scratch，随后复用 silu_mul，因此需要那块 scratch 加上激活的旋转缓冲。
        return detail::ternary_projection_workspace_bytes(gate_up_rows, input_rows, max_tokens);
    }
    throw std::invalid_argument("linear_swiglu workspace: unsupported weight format");
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t   = x.ne[1];
    const bool large_shape = x.ne[0] == 5120 && out.ne[0] == 17408 && gate_up_weight.n == 34816 &&
                             gate_up_weight.k == 5120 && gate_up_weight.padded_shape[0] == 34816 &&
                             gate_up_weight.padded_shape[1] == 5120;
    const bool w8_shape = x.ne[0] == 2048 && out.ne[0] == 6144 && gate_up_weight.n == 12288 &&
                          gate_up_weight.k == 2048 && gate_up_weight.padded_shape[0] == 12288 &&
                          gate_up_weight.padded_shape[1] == 2048;
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!large_shape && !w8_shape)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }

    const bool common_row_split =
        gate_up_weight.layout == QuantLayout::RowSplit &&
        gate_up_weight.scale_dtype == DType::FP16 && gate_up_weight.ndim == 2 &&
        gate_up_weight.shape[0] == gate_up_weight.n &&
        gate_up_weight.shape[1] == gate_up_weight.k && gate_up_weight.qdata != nullptr &&
        gate_up_weight.scales != nullptr;
    const bool q4_weight = large_shape && gate_up_weight.qtype == QType::Q4G64_F16S &&
                           gate_up_weight.group_size == 64 && gate_up_weight.group == 64 &&
                           common_row_split;
    const bool w8_weight = w8_shape && gate_up_weight.qtype == QType::W8G32_F16S &&
                           gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
                           gate_up_weight.qhigh == nullptr &&
                           gate_up_weight.high_plane_bytes == 0 && common_row_split;
    const bool ternary_weight =
        large_shape &&
        (gate_up_weight.qtype == QType::PTQ1_0_G128 ||
         gate_up_weight.qtype == QType::PQ2_0_G128) &&
        gate_up_weight.group == 128 && common_row_split;
    if (!q4_weight && !w8_weight && !ternary_weight) {
        throw std::invalid_argument("linear_swiglu: unsupported weight");
    }

    if (ternary_weight) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("ternary linear_swiglu admits only A16");
        }
        // out = silu(gate) * up。没有融合的三元 SwiGLU 核，就用共享三元 GEMM 投影整个融合父权重
        // （它同时把 x 映射进折叠基），再对两个半行区间复用现成的逐元素 silu_mul —— 与 MTP 路径
        // 对它的 W8 gate/up 父权重做法完全一致。
        auto scope     = ws.scope();
        Tensor gate_up = ws.alloc(DType::BF16, {gate_up_weight.n, t});
        detail::ternary_dispatch(x, gate_up_weight, gate_up, policy, &ws, stream);
        const std::int32_t rows = gate_up_weight.n / 2;
        silu_mul(gate_up.slice(0, 0, rows), gate_up.slice(0, rows, rows), out, stream);
        return;
    }

    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("linear_swiglu: Q4/W8 admit only A16");
    }
    if (!aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, w8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }

    if (w8_weight) {
        detail::w8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
    } else {
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream);
    }
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

} // namespace ninfer::ops
