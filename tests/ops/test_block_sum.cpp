#include "core/device.h"
#include "ninfer/ops/block_sum.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// FP32 accumulation of at most 128 BF16-decoded terms bounded by [-8, 8] stays far below this
// absolute difference from an FP64 sum of the same values, while a token assigned to the wrong
// block, a repeated token, or a dropped token moves a column by an amount of order 8.
constexpr PointwiseCriterion kCriterion{/*absolute*/ 2.0e-2, /*relative*/ 1.0e-4};

struct Geometry {
    std::int32_t width;
    std::int32_t tokens;
    std::int32_t first_position;
    std::int32_t block_tokens;
};

std::vector<std::uint16_t> encode(const std::vector<float>& values) {
    std::vector<std::uint16_t> encoded(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) encoded[i] = f32_to_bf16(values[i]);
    return encoded;
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    const std::vector<float> raw = from_device<float>(device, count);
    return std::vector<double>(raw.begin(), raw.end());
}

std::vector<double> as_double(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

// Independent FP64 oracle: every token of the call joins exactly the column its own position
// block selects.
std::vector<double> reference_sums(const Geometry& geometry, const std::vector<float>& value,
                                   std::int32_t blocks) {
    const std::size_t width  = static_cast<std::size_t>(geometry.width);
    const std::size_t tokens = static_cast<std::size_t>(geometry.tokens);
    const std::int64_t base  = geometry.first_position / geometry.block_tokens;

    std::vector<double> reference(width * static_cast<std::size_t>(blocks), 0.0);
    for (std::size_t t = 0; t < tokens; ++t) {
        const std::int64_t position = geometry.first_position + static_cast<std::int64_t>(t);
        const std::size_t column = static_cast<std::size_t>(position / geometry.block_tokens - base);
        for (std::size_t w = 0; w < width; ++w) {
            reference[w + width * column] += static_cast<double>(value[w + width * t]);
        }
    }
    return reference;
}

int sum_case(const Geometry& geometry, std::uint32_t seed) {
    const std::size_t width  = static_cast<std::size_t>(geometry.width);
    const std::size_t tokens = static_cast<std::size_t>(geometry.tokens);

    std::vector<float> value(width * tokens);
    fill_uniform(value, seed, -8.0f, 8.0f);
    round_to_bf16(value);

    const std::int32_t blocks =
        ops::block_sum_by_position_blocks(geometry.first_position, geometry.tokens,
                                          geometry.block_tokens);
    const std::vector<double> reference = reference_sums(geometry, value, blocks);

    GuardedDeviceBuffer device_value(value.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_sums(reference.size() * sizeof(float));
    const std::vector<std::uint16_t> encoded = encode(value);
    device_value.copy_from_host(encoded.data(), encoded.size() * sizeof(std::uint16_t));
    device_sums.fill(0x7f);

    Tensor value_tensor(device_value.data(), DType::BF16, {geometry.width, geometry.tokens});
    Tensor sums_tensor(device_sums.data(), DType::FP32, {geometry.width, blocks});
    ops::block_sum_by_position(value_tensor, sums_tensor, geometry.first_position,
                               geometry.block_tokens, nullptr);
    cuda_synchronize();

    const std::string label = "block_sum_by_position W=" + std::to_string(geometry.width) +
                              " T=" + std::to_string(geometry.tokens) + " first=" +
                              std::to_string(geometry.first_position) + " block_tokens=" +
                              std::to_string(geometry.block_tokens) + " blocks=" +
                              std::to_string(blocks);
    int failures = verify_pointwise(label, read_f32(device_sums.data(), reference.size()),
                                    reference, kCriterion);
    failures += verify_exact((label + " preserves value").c_str(),
                             from_device_bf16(device_value.data(), value.size()),
                             as_double(value));
    failures += device_value.verify_guards((label + " value").c_str());
    failures += device_sums.verify_guards((label + " sums").c_str());
    return failures;
}

// One block holding exactly one token is an exact identity: it proves the declared column count is
// correct and that every declared column is written instead of keeping its initial content.
int single_token_case(std::int32_t width, std::int32_t first_position) {
    std::vector<float> value(static_cast<std::size_t>(width));
    fill_uniform(value, 9001u + static_cast<std::uint32_t>(width), -8.0f, 8.0f);
    round_to_bf16(value);

    GuardedDeviceBuffer device_value(value.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_sums(value.size() * sizeof(float));
    const std::vector<std::uint16_t> encoded = encode(value);
    device_value.copy_from_host(encoded.data(), encoded.size() * sizeof(std::uint16_t));
    device_sums.fill(0x7f);

    Tensor value_tensor(device_value.data(), DType::BF16, {width, 1});
    Tensor sums_tensor(device_sums.data(), DType::FP32, {width, 1});
    ops::block_sum_by_position(value_tensor, sums_tensor, first_position, 128, nullptr);
    cuda_synchronize();

    const std::string label = "block_sum_by_position single token W=" + std::to_string(width) +
                              " first=" + std::to_string(first_position);
    int failures = verify_exact(label.c_str(), from_device<float>(device_sums.data(), value.size()),
                                value);
    failures += device_value.verify_guards((label + " value").c_str());
    failures += device_sums.verify_guards((label + " sums").c_str());
    return failures;
}

int rejection_cases() {
    GuardedDeviceBuffer device_value(256);
    GuardedDeviceBuffer device_sums(256);
    device_value.fill(0);
    device_sums.fill(0);

    Tensor value(device_value.data(), DType::BF16, {4, 3});
    Tensor sums(device_sums.data(), DType::FP32, {4, 2});

    auto is_rejected = [&](Tensor v, Tensor s, std::int32_t first_position,
                           std::int32_t block_tokens) {
        try {
            ops::block_sum_by_position(v, s, first_position, block_tokens, nullptr);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };
    auto expect_rejected = [&](bool rejected, const char* what) {
        if (rejected) return 0;
        std::cerr << "block_sum_by_position accepted " << what << '\n';
        return 1;
    };
    auto expect_accepted = [&] {
        try {
            ops::block_sum_by_position(value, sums, 0, 2, nullptr);
        } catch (const std::invalid_argument& error) {
            std::cerr << "block_sum_by_position rejected a conforming call: " << error.what() << '\n';
            return 1;
        }
        return 0;
    };

    Tensor permuted = value;
    permuted.nb[0] *= 2;

    int failures = 0;
    failures += expect_rejected(is_rejected(Tensor(nullptr, DType::BF16, {4, 3}), sums, 0, 2),
                                "a null value");
    failures += expect_rejected(is_rejected(value, Tensor(nullptr, DType::FP32, {4, 2}), 0, 2),
                                "a null sums");
    failures += expect_rejected(
        is_rejected(Tensor(device_value.data(), DType::FP32, {4, 3}), sums, 0, 2),
        "an FP32 value");
    failures += expect_rejected(is_rejected(value, Tensor(device_sums.data(), DType::BF16, {4, 2}),
                                            0, 2),
                                "a BF16 sums");
    failures += expect_rejected(
        is_rejected(Tensor(device_value.data(), DType::BF16, {4, 3, 2}), sums, 0, 2),
        "a rank-3 value");
    failures += expect_rejected(is_rejected(value, permuted, 0, 2), "a non-contiguous value");
    failures += expect_rejected(is_rejected(value, sums, 0, 0), "a zero block_tokens");
    failures += expect_rejected(is_rejected(value, sums, 0, -4), "a negative block_tokens");
    failures += expect_rejected(is_rejected(value, sums, -1, 2), "a negative first_position");
    failures += expect_rejected(
        is_rejected(value, sums, std::numeric_limits<std::int32_t>::max(), 2),
        "a first_position whose position range overflows I32");
    failures += expect_rejected(is_rejected(value, sums, 0, 3), "a wrong derived block count");
    failures += expect_rejected(
        is_rejected(value, Tensor(device_sums.data(), DType::FP32, {3, 2}), 0, 2),
        "a sums row count that differs from the value rows");
    failures += expect_rejected(
        is_rejected(Tensor(device_value.data(), DType::BF16, {4, 3}),
                    Tensor(device_value.data(), DType::FP32, {4, 2}), 0, 2),
        "sums aliasing value");
    failures += expect_accepted();
    cuda_synchronize();
    failures += device_value.verify_guards("block_sum_by_position accepted value");
    failures += device_sums.verify_guards("block_sum_by_position accepted sums");
    return failures;
}

// The decode path launches this Op inside a CUDA Graph capture, so a captured launch must replay
// against changed device data with no host round trip and no captured synchronization.
int graph_capture_case() {
    constexpr std::int32_t kWidth = 128;
    constexpr std::int32_t kTokens = 4;
    constexpr std::int32_t kFirstPosition = 5;
    constexpr std::int32_t kBlockTokens = 3;
    const Geometry geometry{kWidth, kTokens, kFirstPosition, kBlockTokens};
    const std::int32_t blocks =
        ops::block_sum_by_position_blocks(kFirstPosition, kTokens, kBlockTokens);

    std::vector<float> value(static_cast<std::size_t>(kWidth) * kTokens);
    fill_uniform(value, 4242u, -8.0f, 8.0f);
    round_to_bf16(value);

    DeviceBuffer device_value(value.size() * sizeof(std::uint16_t));
    DeviceBuffer device_sums(static_cast<std::size_t>(kWidth) * blocks * sizeof(float));
    const std::vector<std::uint16_t> initial = encode(value);
    device_value.copy_from_host(initial.data(), device_value.bytes);
    device_sums.fill(0x7f);

    Tensor value_tensor(device_value.p, DType::BF16, {kWidth, kTokens});
    Tensor sums_tensor(device_sums.p, DType::FP32, {kWidth, blocks});

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto launch = [&] {
        ops::block_sum_by_position(value_tensor, sums_tensor, kFirstPosition, kBlockTokens, stream);
    };

    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

    int failures = 0;
    for (int replay = 0; replay < 3; ++replay) {
        for (auto& x : value) x += 0.125f * static_cast<float>(replay + 1);
        round_to_bf16(value);
        const std::vector<std::uint16_t> encoded = encode(value);
        CUDA_CHECK(cudaMemcpyAsync(device_value.p, encoded.data(), device_value.bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const std::vector<double> reference = reference_sums(geometry, value, blocks);
        const std::string label = "block_sum_by_position graph replay " + std::to_string(replay);
        failures += verify_pointwise(label, read_f32(device_sums.p, reference.size()),
                                     reference, kCriterion);
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}

// The ordinary decode route captures this launch into a CUDA Graph, which bakes the position
// argument while the lane's real position advances every step. That is only sound because a
// one-token range reduces to a single block whose slice is the whole column for *every* position:
// block_count is always 1, and the clamped token window is always [0, 1). This case pins that
// invariant, so the graph argument rests on a tested property rather than on the derivation alone.
int decode_position_invariance_case() {
    constexpr std::int32_t kWidth = 64;
    constexpr std::int32_t kBlockTokens = 128;
    const std::vector<std::int32_t> positions{0, 1, 5, 127, 128, 129, 255, 4096, 1 << 20};

    std::vector<float> value(static_cast<std::size_t>(kWidth));
    fill_uniform(value, 77u, -4.0f, 4.0f);
    round_to_bf16(value);

    DeviceBuffer device_value(value.size() * sizeof(std::uint16_t));
    DeviceBuffer device_sums(static_cast<std::size_t>(kWidth) * sizeof(float));
    const std::vector<std::uint16_t> initial = encode(value);
    device_value.copy_from_host(initial.data(), device_value.bytes);

    Tensor value_tensor(device_value.p, DType::BF16, {kWidth, 1});
    Tensor sums_tensor(device_sums.p, DType::FP32, {kWidth, 1});

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int failures = 0;
    // The reference is the single token's own values, independent of any position.
    std::vector<double> expected(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) { expected[i] = static_cast<double>(value[i]); }

    std::vector<double> first_result;
    for (const std::int32_t position : positions) {
        // The declared shape must stay [W, 1] for every position, or a graph replay would be
        // reading a different extent than the capture validated.
        if (ops::block_sum_by_position_blocks(position, 1, kBlockTokens) != 1) {
            ++failures;
            std::cerr << "FAIL: a one-token range must declare exactly one block, position "
                      << position << '\n';
            continue;
        }
        device_sums.fill(0x7f);
        ops::block_sum_by_position(value_tensor, sums_tensor, position, kBlockTokens, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::vector<double> actual = read_f32(device_sums.p, expected.size());
        failures += verify_pointwise("one-token reduction at position " + std::to_string(position),
                                     actual, expected, kCriterion);
        if (first_result.empty()) {
            first_result = actual;
        } else {
            failures += verify_pointwise("one-token reduction is position invariant",
                                         actual, first_result, kCriterion);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += sum_case({1, 1, 0, 128}, 11u);
    failures += sum_case({4, 7, 0, 3}, 12u);
    failures += sum_case({4, 129, 100, 128}, 13u);
    failures += sum_case({64, 128, 0, 128}, 14u);
    failures += sum_case({64, 128, 63, 128}, 15u);
    failures += sum_case({128, 1024, 0, 128}, 16u);
    failures += sum_case({128, 1024, 4096, 128}, 17u);
    failures += sum_case({512, 4096, 0, 128}, 18u);
    failures += sum_case({3, 100, 1, 3}, 19u);
    failures += sum_case({17, 33, 262143, 128}, 20u);
    failures += sum_case({1, 1024, 131071, 1}, 21u);
    failures += single_token_case(1, 0);
    failures += single_token_case(5, 1000000);
    failures += rejection_cases();
    failures += graph_capture_case();
    failures += decode_position_invariance_case();
    std::cout << (failures ? "FAIL" : "OK") << " block_sum\n";
    return failures ? 1 : 0;
}