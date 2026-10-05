#include "ninfer/ops/rmsnorm_swiglu_mlp.h"

#include "core/layout.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/common/rowsplit_a8_quantize.h"
#include "ops/linear/t5/t5_project.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::size_t round_up_256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("rmsnorm_swiglu_mlp: invalid compute policy");
}

bool t5_profile(QType gate_up_qtype, QType down_qtype) {
    return gate_up_qtype == QType::T5_G128_FP16 && down_qtype == QType::T5_G128_FP16;
}

// The Qwen3.8 dense MLP: Q4 gate/up [34816,5120] and Q5 down [5120,17408].
constexpr std::int32_t kQ4Q5Hidden       = 5120;
constexpr std::int32_t kQ4Q5GateUpRows   = 34816;
constexpr std::int32_t kQ4Q5Intermediate = 17408;

bool q4_q5_profile(QType gate_up_qtype, QType down_qtype) {
    return gate_up_qtype == QType::Q4_G64_FP16 && down_qtype == QType::Q5_G64_FP16;
}

// Whether both projections resolve to their A8 routes at this width, which the fused route
// replaces; otherwise the block is the plain composition.
bool q4_q5_quantized(std::int32_t tokens, LinearPolicy gate_up_policy, LinearPolicy down_policy) {
    const auto gate_up = detail::q4_linear_swiglu_resolve_plan(
        {kQ4Q5GateUpRows, kQ4Q5Intermediate, kQ4Q5Hidden, kQ4Q5Hidden, tokens, gate_up_policy});
    const auto down = detail::q5_linear_add_resolve_plan(
        {kQ4Q5Hidden, kQ4Q5Intermediate, kQ4Q5Intermediate, tokens, down_policy});
    return gate_up.schedule == detail::Q4LinearSwiGluScheduleId::A8MmaFoldedPipelinedR64C128 &&
           down.schedule == detail::Q5LinearAddScheduleId::A8MmaResidualPipelinedR128C64;
}

std::size_t q4_q5_quantized_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)detail::allocate_a8_g64_activation(layout, kQ4Q5Intermediate, tokens);
    (void)detail::allocate_a8_g64_activation(layout, kQ4Q5Hidden, tokens);
    return layout.peak_bytes(1);
}

// n in BF16, then the activation a in BF16 across both projections, each projection's
// transient storage scoped to it.
std::size_t q4_q5_composed_bytes(LinearPolicy gate_up_policy, LinearPolicy down_policy,
                                 std::int32_t min_tokens, std::int32_t max_tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {kQ4Q5Hidden, max_tokens});
    (void)layout.alloc(DType::BF16, {kQ4Q5Intermediate, max_tokens});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(linear_swiglu_workspace_capacity_bytes(
            QType::Q4_G64_FP16, kQ4Q5GateUpRows, kQ4Q5Hidden, gate_up_policy, min_tokens,
            max_tokens));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(linear_add_workspace_capacity_bytes(
            QType::Q5_G64_FP16, kQ4Q5Hidden, kQ4Q5Intermediate, down_policy, min_tokens,
            max_tokens));
    }
    return layout.peak_bytes(1);
}

void require_q4_q5(const Weight& gate_up, const Weight& down, const Tensor& residual) {
    const auto aligned = [](const void* p) {
        return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & 15) == 0;
    };
    if (gate_up.n != kQ4Q5GateUpRows || gate_up.k != kQ4Q5Hidden ||
        gate_up.padded_shape[1] != kQ4Q5Hidden || gate_up.layout != QuantLayout::RowSplit ||
        !aligned(gate_up.qdata) || !aligned(gate_up.scales) || down.n != kQ4Q5Hidden ||
        down.k != kQ4Q5Intermediate || down.padded_shape[1] != kQ4Q5Intermediate ||
        down.layout != QuantLayout::RowSplit || !aligned(down.qdata) || !aligned(down.qhigh) ||
        !aligned(down.scales)) {
        throw std::invalid_argument(
            "rmsnorm_swiglu_mlp: Q4 gate/up [34816,5120] and Q5 down [5120,17408] expected");
    }
    if (residual.dtype != DType::BF16 || residual.ne[0] != kQ4Q5Hidden || residual.ne[1] <= 0 ||
        residual.ne[2] != 1 || residual.ne[3] != 1 || !residual.is_contiguous() ||
        !aligned(residual.data)) {
        throw std::invalid_argument(
            "rmsnorm_swiglu_mlp: the residual must be contiguous 16-byte aligned BF16 [5120,T]");
    }
}

void q4_q5_mlp(const RmsNormPrologue& norm, const Weight& gate_up, LinearPolicy gate_up_policy,
               const Weight& down, LinearPolicy down_policy, Tensor& residual,
               WorkspaceArena& ws, cudaStream_t stream) {
    require_q4_q5(gate_up, down, residual);
    const std::int32_t tokens = residual.ne[1];
    auto scope                = ws.scope();
    if (!q4_q5_quantized(tokens, gate_up_policy, down_policy)) {
        Tensor normalized = ws.alloc(DType::BF16, {kQ4Q5Hidden, tokens});
        rmsnorm(residual, norm.weight, norm.eps, norm.unit_offset, normalized, stream);
        Tensor activation = ws.alloc(DType::BF16, {kQ4Q5Intermediate, tokens});
        {
            auto call = ws.scope();
            linear_swiglu(normalized, gate_up, activation, gate_up_policy, ws, stream);
        }
        linear_add(activation, down, residual, down_policy, ws, stream);
        return;
    }
    // n and a exist only as the A8 activations the projections would quantize them to.
    detail::A8G64Activation activation =
        detail::allocate_a8_g64_activation(ws, kQ4Q5Intermediate, tokens);
    {
        auto call                         = ws.scope();
        detail::A8G64Activation normalized =
            detail::allocate_a8_g64_activation(ws, kQ4Q5Hidden, tokens);
        detail::rmsnorm_a8_g64_quantize(residual, norm.weight, norm.eps, norm.unit_offset,
                                        normalized, stream);
        detail::q4_linear_swiglu_a8_quantized_mma_folded_pipelined_r64_c128_launch(
            normalized, gate_up, activation, stream);
    }
    detail::q5_linear_add_a8_mma_pipelined_r128_c64_launch(activation, down, residual, stream);
}

} // namespace

bool rmsnorm_swiglu_mlp_accepts(QType gate_up_qtype, LinearPolicy gate_up_policy, QType down_qtype,
                                LinearPolicy down_policy) {
    validate_policy(gate_up_policy);
    validate_policy(down_policy);
    return (t5_profile(gate_up_qtype, down_qtype) || q4_q5_profile(gate_up_qtype, down_qtype)) &&
           allows_a8(gate_up_policy) && allows_a8(down_policy);
}

std::size_t rmsnorm_swiglu_mlp_workspace_capacity_bytes(
    QType gate_up_qtype, QType down_qtype, std::int32_t gate_up_rows, std::int32_t input_rows,
    LinearPolicy gate_up_policy, LinearPolicy down_policy, std::int32_t min_tokens,
    std::int32_t max_tokens) {
    if (!rmsnorm_swiglu_mlp_accepts(gate_up_qtype, gate_up_policy, down_qtype, down_policy)) {
        throw std::invalid_argument("rmsnorm_swiglu_mlp workspace: unregistered profile");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("rmsnorm_swiglu_mlp workspace: invalid interval");
    }
    if (q4_q5_profile(gate_up_qtype, down_qtype)) {
        if (gate_up_rows != kQ4Q5GateUpRows || input_rows != kQ4Q5Hidden) {
            throw std::invalid_argument("rmsnorm_swiglu_mlp workspace: unsupported Q4/Q5 shape");
        }
        // Both routes grow with T; the composed one covers the widths below the A8 threshold.
        std::size_t bytes = 0;
        std::int32_t composed_last = 0;
        for (std::int32_t t = min_tokens; t <= max_tokens; ++t) {
            if (q4_q5_quantized(t, gate_up_policy, down_policy)) { break; }
            composed_last = t;
        }
        if (composed_last >= min_tokens) {
            bytes = q4_q5_composed_bytes(gate_up_policy, down_policy, min_tokens, composed_last);
        }
        if (composed_last < max_tokens) { bytes = std::max(bytes, q4_q5_quantized_bytes(max_tokens)); }
        return bytes;
    }
    if (gate_up_rows <= 0 || gate_up_rows % 2048 || input_rows <= 0 || input_rows % 1024) {
        throw std::invalid_argument("rmsnorm_swiglu_mlp workspace: invalid profile or interval");
    }
    const std::int32_t rows = gate_up_rows / 2;
    // g and u live across both projections; each projection's quantized input is scoped to it.
    return 2 * round_up_256(static_cast<std::size_t>(rows) * max_tokens * 2) +
           std::max(detail::t5_workspace_capacity_bytes(gate_up_policy, input_rows, max_tokens),
                    detail::t5_workspace_capacity_bytes(down_policy, rows, max_tokens));
}

void rmsnorm_swiglu_mlp(const RmsNormPrologue& norm, const Weight& gate_up,
                        LinearPolicy gate_up_policy, const Weight& down, LinearPolicy down_policy,
                        Tensor& residual, WorkspaceArena& ws, cudaStream_t stream) {
    if (!rmsnorm_swiglu_mlp_accepts(gate_up.qtype, gate_up_policy, down.qtype, down_policy)) {
        throw std::invalid_argument("rmsnorm_swiglu_mlp: unregistered weight format or policy");
    }
    if (q4_q5_profile(gate_up.qtype, down.qtype)) {
        q4_q5_mlp(norm, gate_up, gate_up_policy, down, down_policy, residual, ws, stream);
        return;
    }
    const std::int32_t hidden = residual.ne[0];
    const std::int32_t tokens = residual.ne[1];
    if (gate_up.n % 2048 || gate_up.k != hidden || down.n != hidden || down.k != gate_up.n / 2) {
        throw std::invalid_argument("rmsnorm_swiglu_mlp: gate/up [2M,D] and down [D,M] expected");
    }
    if (tokens <= 0) { throw std::invalid_argument("rmsnorm_swiglu_mlp: T must be positive"); }
    // The residual is both the normalized input and the accumulated output; the t5 projections
    // validate it, the norm weight and their weights.
    auto scope             = ws.scope();
    Tensor gate            = ws.alloc(DType::BF16, {down.k, tokens});
    Tensor up              = ws.alloc(DType::BF16, {down.k, tokens});
    Tensor* gate_up_rows[] = {&gate, &up};
    detail::t5_project_rmsnorm(residual, norm.weight, norm.eps, norm.unit_offset, gate_up,
                               gate_up_rows, /*accumulate=*/false, gate_up_policy, &ws, stream);
    Tensor* delta[] = {&residual};
    detail::t5_project_swiglu(gate, up, down, delta, /*accumulate=*/true, down_policy, &ws, stream);
}

} // namespace ninfer::ops
