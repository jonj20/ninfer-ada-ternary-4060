// Accuracy of the int8-activation decode path against the bf16-activation path.
//
// Build (see run_checks.sh):
//   nvcc -O2 -std=c++17 -arch=sm_89 -I<repo>/src \
//        tools/verify/ternary_gemv/activation_quant_accuracy.cu -o activation_quant_accuracy
//
// The int8 path exists because decode is instruction-issue bound, not bandwidth bound: one dp4a
// carries four weight-activation products where the bf16 path needs an FFMA plus an int-to-float
// per weight. This script is how the resulting quantization error was measured.
//
// Greedy token-for-token equality is NOT a usable acceptance test here: a 1e-3 logit difference
// flips an argmax and the sequences then diverge completely (observed divergence at token 2). What
// matters is the distribution of per-row error and the preserved global magnitude.
//
// Reference points from the RTX 4060 Laptop run in docs/4060-开发跟踪.md §9.1: median per-row
// ~1.3%, p90 ~6%, RMS ratio ~0.991.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"

namespace {

using ninfer::ops::detail::kGemvWarpsPerBlock;

void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,
                     cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

void compare(int n, int k) {
    const int groups = k / 128;
    std::mt19937 rng(7);

    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * groups * 24);
    std::vector<std::uint8_t> high(static_cast<std::size_t>(n) * groups * 2);
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(n) * groups * 2);
    std::vector<__nv_bfloat16> x(k);
    for (auto& v : codes) v = static_cast<std::uint8_t>(rng() & 0xFF);
    for (auto& v : high) v = static_cast<std::uint8_t>(rng() & 0xFF);
    // Weight scales near N(0, 0.02) so the products resemble real activations.
    std::normal_distribution<float> weight_scale(0.0f, 0.02f);
    for (std::size_t i = 0; i < scales.size(); i += 2) {
        const std::uint16_t bits = __half_as_ushort(__float2half(weight_scale(rng)));
        scales[i] = static_cast<std::uint8_t>(bits & 0xFF);
        scales[i + 1] = static_cast<std::uint8_t>(bits >> 8);
    }
    std::normal_distribution<float> activation(0.0f, 1.0f);
    for (auto& v : x) v = __float2bfloat16_rn(activation(rng));

    __nv_bfloat16* dx = nullptr;
    __nv_bfloat16* d_bf16 = nullptr;
    __nv_bfloat16* d_dp4a = nullptr;
    std::uint8_t* dc = nullptr;
    std::uint8_t* dh = nullptr;
    std::uint8_t* ds = nullptr;
    std::int8_t* dq = nullptr;
    float* dqs = nullptr;
    check(cudaMalloc(&dx, static_cast<std::size_t>(k) * 2));
    check(cudaMalloc(&d_bf16, static_cast<std::size_t>(n) * 2));
    check(cudaMalloc(&d_dp4a, static_cast<std::size_t>(n) * 2));
    check(cudaMalloc(&dc, codes.size()));
    check(cudaMalloc(&dh, high.size()));
    check(cudaMalloc(&ds, scales.size()));
    check(cudaMalloc(&dq, static_cast<std::size_t>(k)));
    check(cudaMalloc(&dqs, static_cast<std::size_t>(groups) * 4));
    check(cudaMemcpy(dx, x.data(), static_cast<std::size_t>(k) * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dh, high.data(), high.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(ds, scales.data(), scales.size(), cudaMemcpyHostToDevice));

    const int grid = (n + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    ninfer::ops::detail::ternary_ptq1_gemv_kernel<1>
        <<<grid, kGemvWarpsPerBlock * 32>>>(dx, dc, dh, ds, d_bf16, n, groups, 1, n);
    ninfer::ops::detail::ternary_ptq1_quantize_act_kernel<<<1, 256>>>(dx, dq, dqs, groups);
    ninfer::ops::detail::ternary_ptq1_gemv_dp4a_kernel<<<grid, kGemvWarpsPerBlock * 32>>>(
        dq, dqs, dc, dh, ds, d_dp4a, n, groups);
    check(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> a(n);
    std::vector<__nv_bfloat16> b(n);
    check(cudaMemcpy(a.data(), d_bf16, static_cast<std::size_t>(n) * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), d_dp4a, static_cast<std::size_t>(n) * 2, cudaMemcpyDeviceToHost));

    std::vector<double> rel;
    rel.reserve(n);
    double sum_bf16 = 0.0;
    double sum_dp4a = 0.0;
    for (int i = 0; i < n; ++i) {
        const double va = __bfloat162float(a[i]);
        const double vb = __bfloat162float(b[i]);
        sum_bf16 += va * va;
        sum_dp4a += vb * vb;
        rel.push_back(std::fabs(va - vb) / (std::fabs(va) + 1e-6));
    }
    std::sort(rel.begin(), rel.end());
    std::printf("N=%-6d K=%-6d  median %.5f  p90 %.5f  p99 %.5f  max %.2f   "
                "RMS(bf16)=%.5f RMS(dp4a)=%.5f ratio=%.5f\n",
                n, k, rel[n / 2], rel[static_cast<int>(n * 0.9)],
                rel[static_cast<int>(n * 0.99)], rel[n - 1],
                std::sqrt(sum_bf16 / n), std::sqrt(sum_dp4a / n),
                std::sqrt(sum_dp4a / sum_bf16));
    std::printf("    (large p99/max are rows whose true value is near zero: cancellation inflates\n"
                "     the relative error while the absolute error stays negligible. The RMS ratio\n"
                "     is the number that tracks preserved magnitude.)\n");

    cudaFree(dx); cudaFree(d_bf16); cudaFree(d_dp4a);
    cudaFree(dc); cudaFree(dh); cudaFree(ds); cudaFree(dq); cudaFree(dqs);
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3) {
        compare(std::atoi(argv[1]), std::atoi(argv[2]));
        return 0;
    }
    std::printf("int8-activation path vs bf16-activation path\n");
    compare(4096, 5120);
    compare(2048, 17408);
    return 0;
}
