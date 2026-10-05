// rmsnorm_swiglu_mlp on the Qwen3.8 dense MLP (Q4 gate/up [34816,5120], Q5 down [5120,17408])
// under AllowA8, against an FP64 oracle of its documented semantics: n = rmsnorm(residual) and
// a = SiLU(g) * u are rounded to BF16, and each projection multiplies the A8 quantization of its
// BF16 input (tests/ops/a8_g64_reference.h) from 129 columns on, otherwise the input itself. The
// block must also equal rmsnorm(), linear_swiglu() and linear_add() composed, bit for bit.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_swiglu_mlp.h"
#include "ops/a8_g64_reference.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kIntermediate = 17408;
constexpr std::int32_t kGateUpRows   = 2 * kIntermediate;
constexpr float kEps                 = 1e-6f;

// One BF16 unit roundoff of the delta plus the BF16 roundings of the stored delta and residual.
constexpr ReductionCriterion kCriterion{2.0 / 256.0, 2.0 / 256.0, 4.0 / 256.0};

std::vector<std::uint16_t> bf16_values(std::size_t count, std::uint32_t seed, float low,
                                       float high) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<std::uint16_t> bits(count);
    for (auto& b : bits) b = f32_to_bf16(dist(rng));
    return bits;
}

void parallel_rows(std::int32_t rows, const std::function<void(std::int32_t)>& body) {
    const std::int32_t threads =
        std::max(1, static_cast<std::int32_t>(std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    for (std::int32_t w = 0; w < threads; ++w) {
        workers.emplace_back([&, w] {
            for (std::int32_t row = w; row < rows; row += threads) body(row);
        });
    }
    for (auto& worker : workers) worker.join();
}

// out[s][t][r] = sum_c W(r, c) in[s][t][c] for the inputs [T][K] of each set s, FP64.
std::vector<std::vector<double>> project(const quantized_weight::PackedWeight& w,
                                         const std::vector<const std::vector<double>*>& inputs,
                                         std::int32_t tokens) {
    const std::int32_t rows = w.weight.n, k = w.weight.k;
    std::vector<std::vector<double>> out(inputs.size(),
                                         std::vector<double>(std::size_t(rows) * tokens));
    parallel_rows(rows, [&](std::int32_t row) {
        std::vector<double> weights(static_cast<std::size_t>(k));
        for (std::int32_t c = 0; c < k; ++c) weights[c] = quantized_weight::logical_weight_fp64(w, row, c);
        for (std::size_t s = 0; s < inputs.size(); ++s) {
            for (std::int32_t t = 0; t < tokens; ++t) {
                const double* x = &(*inputs[s])[std::size_t(t) * k];
                double sum      = 0.0;
                for (std::int32_t c = 0; c < k; ++c) sum += weights[c] * x[c];
                out[s][std::size_t(t) * rows + row] = sum;
            }
        }
    });
    return out;
}

std::vector<double> as_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) values[i] = bf16_to_f32(bits[i]);
    return values;
}

double silu(double v) { return v / (1.0 + std::exp(-v)); }

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        // A16 widths (decode, verification, the A16 boundary) and A8 widths (its first column,
        // one full and a partial 128-token tile).
        const std::vector<std::int32_t> token_cases{1, 64, 128, 129, 300};
        const std::int32_t tokens = token_cases.back();
        const ops::LinearPolicy policy = ops::LinearPolicy::AllowA8;
        if (!ops::rmsnorm_swiglu_mlp_accepts(QType::Q4_G64_FP16, policy, QType::Q5_G64_FP16,
                                             policy) ||
            ops::rmsnorm_swiglu_mlp_accepts(QType::Q4_G64_FP16, ops::LinearPolicy::A16Only,
                                            QType::Q5_G64_FP16, policy)) {
            std::cerr << "Q4/Q5 rmsnorm_swiglu_mlp registration mismatch\n";
            return 1;
        }

        // Hashed codes; scales that keep g and u within SiLU's curved range and the update
        // comparable to the residual.
        quantized_weight::PatternedWeightOptions gate_up_options, down_options;
        gate_up_options.row_split_scale = quantized_weight::RowSplitScalePattern::Small;
        gate_up_options.row_split_codes = quantized_weight::RowSplitCodePattern::Hashed;
        down_options.row_split_scale    = quantized_weight::RowSplitScalePattern::Tiny;
        down_options.row_split_codes    = quantized_weight::RowSplitCodePattern::Hashed;
        auto gate_up = quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, kGateUpRows,
                                                               kHidden, 1409U, gate_up_options);
        auto down = quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, kHidden,
                                                            kIntermediate, 1427U, down_options);
        const auto residual = bf16_values(std::size_t(kHidden) * tokens, 31U, -1.0f, 1.0f);
        const auto gain     = bf16_values(kHidden, 32U, -0.5f, 0.5f);

        // Oracle: n in BF16, then both activation paths.
        std::vector<std::uint16_t> n_bits(residual.size());
        for (std::int32_t t = 0; t < tokens; ++t) {
            const std::uint16_t* row = &residual[std::size_t(t) * kHidden];
            double squares           = 0.0;
            for (std::int32_t c = 0; c < kHidden; ++c) {
                const double v = bf16_to_f32(row[c]);
                squares += v * v;
            }
            const double inverse = 1.0 / std::sqrt(squares / kHidden + double(kEps));
            for (std::int32_t c = 0; c < kHidden; ++c) {
                const double v = double(bf16_to_f32(row[c])) * inverse *
                                 (1.0 + double(bf16_to_f32(gain[c])));
                n_bits[std::size_t(t) * kHidden + c] = f32_to_bf16(static_cast<float>(v));
            }
        }
        const auto n16 = as_double(n_bits);
        const auto n8  = a8_g64_dequantized(n_bits, kHidden, tokens);
        const auto gu  = project(gate_up, {&n16, &n8}, tokens);
        std::vector<std::uint16_t> a16_bits(std::size_t(kIntermediate) * tokens), a8_bits(a16_bits.size());
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t i = 0; i < kIntermediate; ++i) {
                const std::size_t g = std::size_t(t) * kGateUpRows + i;
                const std::size_t a = std::size_t(t) * kIntermediate + i;
                a16_bits[a] = f32_to_bf16(static_cast<float>(silu(gu[0][g]) * gu[0][g + kIntermediate]));
                a8_bits[a]  = f32_to_bf16(static_cast<float>(silu(gu[1][g]) * gu[1][g + kIntermediate]));
            }
        }
        const auto a16   = as_double(a16_bits);
        const auto a8    = a8_g64_dequantized(a8_bits, kIntermediate, tokens);
        const auto delta = project(down, {&a16, &a8}, tokens);

        GuardedDeviceBuffer device_gate_up(gate_up.payload.size()), device_down(down.payload.size());
        device_gate_up.copy_from_host(gate_up.payload.data(), gate_up.payload.size());
        device_down.copy_from_host(down.payload.data(), down.payload.size());
        const Weight gate_up_weight = gate_up.device_weight(device_gate_up.data());
        const Weight down_weight    = down.device_weight(device_down.data());
        DeviceBuffer device_gain(gain.size() * 2);
        device_gain.copy_from_host(gain.data(), gain.size() * 2);
        const ops::RmsNormPrologue norm{Tensor(device_gain.p, DType::BF16, {kHidden}), kEps, true};

        int failures = 0;
        for (const std::int32_t t : token_cases) {
            const bool quantized = t >= kA8G64MinTokens;
            const std::string label =
                "Q4/Q5 rmsnorm_swiglu_mlp T=" + std::to_string(t) + (quantized ? " A8" : " A16");
            const std::size_t elements = std::size_t(kHidden) * t;
            GuardedDeviceBuffer fused(elements * 2), composed(elements * 2);
            fused.copy_from_host(residual.data(), elements * 2);
            composed.copy_from_host(residual.data(), elements * 2);
            Tensor fused_residual(fused.data(), DType::BF16, {kHidden, t});
            Tensor composed_residual(composed.data(), DType::BF16, {kHidden, t});

            const std::size_t capacity = ops::rmsnorm_swiglu_mlp_workspace_capacity_bytes(
                QType::Q4_G64_FP16, QType::Q5_G64_FP16, kGateUpRows, kHidden, policy, policy, t, t);
            WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
            ops::rmsnorm_swiglu_mlp(norm, gate_up_weight, policy, down_weight, policy,
                                    fused_residual, workspace, nullptr);
            cuda_synchronize();
            if (workspace.used() != 0 || workspace.peak_used() != capacity) {
                std::cerr << label << ": workspace query/execution mismatch (peak "
                          << workspace.peak_used() << ", capacity " << capacity << ")\n";
                ++failures;
            }

            DeviceBuffer normalized(elements * 2), activation(std::size_t(kIntermediate) * t * 2);
            Tensor n(normalized.p, DType::BF16, {kHidden, t});
            Tensor a(activation.p, DType::BF16, {kIntermediate, t});
            WorkspaceArena scratch(std::max<std::size_t>(
                std::max(ops::linear_swiglu_workspace_capacity_bytes(
                             QType::Q4_G64_FP16, kGateUpRows, kHidden, policy, t, t),
                         ops::linear_add_workspace_capacity_bytes(
                             QType::Q5_G64_FP16, kHidden, kIntermediate, policy, t, t)),
                256));
            ops::rmsnorm(composed_residual, norm.weight, kEps, true, n, nullptr);
            ops::linear_swiglu(n, gate_up_weight, a, policy, scratch, nullptr);
            ops::linear_add(a, down_weight, composed_residual, policy, scratch, nullptr);
            cuda_synchronize();

            std::vector<std::uint16_t> got(elements), reference_bits(elements);
            fused.copy_to_host(got.data(), elements * 2);
            composed.copy_to_host(reference_bits.data(), elements * 2);
            if (std::memcmp(got.data(), reference_bits.data(), elements * 2) != 0) {
                std::cerr << label << ": differs from rmsnorm + linear_swiglu + linear_add\n";
                ++failures;
            }
            // The update itself against the oracle's delta.
            std::vector<double> update(elements), expected(elements);
            for (std::int32_t token = 0; token < t; ++token) {
                for (std::int32_t row = 0; row < kHidden; ++row) {
                    const std::size_t i = std::size_t(token) * kHidden + row;
                    update[i]   = double(bf16_to_f32(got[i])) - double(bf16_to_f32(residual[i]));
                    expected[i] = delta[quantized ? 1 : 0][i];
                }
            }
            failures += verify_reduction(label, update, expected, kCriterion);
            failures += fused.verify_guards(label);
        }
        failures += device_gate_up.verify_guards("gate_up weight");
        failures += device_down.verify_guards("down weight");
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4/Q5 rmsnorm_swiglu_mlp\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q4/Q5 rmsnorm_swiglu_mlp test failed: " << error.what() << '\n';
        return 1;
    }
}
