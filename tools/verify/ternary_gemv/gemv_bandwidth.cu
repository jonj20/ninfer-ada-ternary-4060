// Bandwidth of the PTQ1_0 decode GEMV kernels, on payloads chosen to be honest about DRAM.
//
// Build (see run_checks.sh):
//   nvcc -O3 -std=c++17 -arch=sm_89 -I<repo>/src \
//        tools/verify/ternary_gemv/gemv_bandwidth.cu -o gemv_bandwidth
//
//   gemv_bandwidth            # the shape table below
//   gemv_bandwidth <n> <k>    # one shape, for quick A/B while tuning
//
// Three measurement hazards, each of which produced a wrong conclusion before this harness existed:
//
// 1. Payload size vs cache. A 39 MB tensor is only 1.2x this card's 32 MB L2, so repeated
//    iterations partially hit cache and inflate the number. The table therefore mixes the real
//    model shapes with 78/156/312 MB payloads. The ceiling to compare against comes from
//    tools/hbm_bandwidth_probe.cu: 249.6 GB/s on an RTX 4060 Laptop (resident-grid uint4 read).
//
// 2. SM clock state. This laptop part swings between roughly 1.68 and 2.01 GHz with the power state
//    and the kernel is partly instruction-issue bound, so a single timing carries about +/-20% noise.
//    Every measurement is repeated and reported as best and median, the convention
//    tools/hbm_bandwidth_probe.cu already uses.
//
// 3. Normalization. "eff GB/s" counts the PTQ1_0 weight bytes actually read, rows * k * 28/128,
//    even though a given variant may touch only some of those planes.
//
// The int8 row includes the activation quantization launch, because that is what the engine pays.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"

namespace {

using ninfer::ops::detail::kGemvWarpsPerBlock;

constexpr int kTrials = 5;

struct Payload {
    int n;
    int k;
    const char* label;
};

void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,
                     cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

struct Result {
    float median_ms;
    float best_ms;
};

// Times `launch` over several trials; each trial runs `iters` back-to-back launches.
Result measure(const std::function<void()>& launch, int iters) {
    std::vector<float> samples;
    samples.reserve(kTrials);
    cudaEvent_t a;
    cudaEvent_t b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int warm = 0; warm < 3; ++warm) { launch(); }
    check(cudaDeviceSynchronize());
    for (int t = 0; t < kTrials; ++t) {
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) { launch(); }
        cudaEventRecord(b);
        check(cudaEventSynchronize(b));
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        samples.push_back(ms / iters);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2], samples.front()};
}

void report(const char* tag, const Payload& p, const Result& r, double bytes) {
    std::printf("  %-5s N=%-7d K=%-6d %-22s %8.2f us  median %6.1f GB/s  best %6.1f GB/s\n",
                tag, p.n, p.k, p.label, r.median_ms * 1e3f,
                bytes / (r.median_ms * 1e-3) / 1e9, bytes / (r.best_ms * 1e-3) / 1e9);
}

void run_shape(const Payload& p) {
    const int groups = p.k / 128;
    const int grid = (p.n + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    // Fewer iterations for the big payloads to keep the run quick; they are bandwidth-bound anyway.
    const int iters = (p.n > 100000) ? 15 : (p.n > 40000 ? 30 : 80);

    __nv_bfloat16* dx = nullptr;
    __nv_bfloat16* dout = nullptr;
    std::uint8_t* dc = nullptr;
    std::uint8_t* dh = nullptr;
    std::uint8_t* ds = nullptr;
    std::int8_t* dq = nullptr;
    float* dqs = nullptr;
    check(cudaMalloc(&dx, static_cast<std::size_t>(p.k) * 2));
    check(cudaMalloc(&dout, static_cast<std::size_t>(p.n) * 2));
    check(cudaMalloc(&dc, static_cast<std::size_t>(p.n) * groups * 24));
    check(cudaMalloc(&dh, static_cast<std::size_t>(p.n) * groups * 2));
    check(cudaMalloc(&ds, static_cast<std::size_t>(p.n) * groups * 2));
    check(cudaMalloc(&dq, static_cast<std::size_t>(p.k)));
    check(cudaMalloc(&dqs, static_cast<std::size_t>(groups) * 4));

    std::vector<__nv_bfloat16> hx(p.k, __float2bfloat16_rn(0.5f));
    std::vector<std::uint8_t> hc(static_cast<std::size_t>(p.n) * groups * 24, 0x55);
    std::vector<std::uint8_t> hh(static_cast<std::size_t>(p.n) * groups * 2, 0);
    check(cudaMemcpy(dx, hx.data(), static_cast<std::size_t>(p.k) * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc, hc.data(), hc.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dh, hh.data(), hh.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(ds, hh.data(), hh.size(), cudaMemcpyHostToDevice));

    const double bytes = static_cast<double>(p.n) * p.k * 28.0 / 128.0;

    const Result bf16 = measure([&] {
        ninfer::ops::detail::ternary_ptq1_gemv_kernel<1>
            <<<grid, kGemvWarpsPerBlock * 32>>>(dx, dc, dh, ds, dout, p.n, groups, 1, p.n);
    }, iters);
    report("bf16", p, bf16, bytes);
    check(cudaGetLastError());

    const Result dp4a = measure([&] {
        ninfer::ops::detail::ternary_ptq1_quantize_act_kernel<<<1, 256>>>(dx, dq, dqs, groups);
        ninfer::ops::detail::ternary_ptq1_gemv_dp4a_kernel<<<grid, kGemvWarpsPerBlock * 32>>>(
            dq, dqs, dc, dh, ds, dout, p.n, groups);
    }, iters);
    report("dp4a", p, dp4a, bytes);
    check(cudaGetLastError());

    cudaFree(dx); cudaFree(dout); cudaFree(dc); cudaFree(dh); cudaFree(ds);
    cudaFree(dq); cudaFree(dqs);
}

} // namespace

int main(int argc, char** argv) {
    const Payload table[] = {
        {34816, 5120, "MLP gate_up (real)"},
        {5120, 17408, "MLP down (real)"},
        {5120, 5120, "attn/GDN (real)"},
        {69632, 5120, "2.4x L2"},
        {139264, 5120, "4.9x L2"},
        {278528, 5120, "9.8x L2"},
    };
    if (argc >= 3) {
        const Payload one[] = {{std::atoi(argv[1]), std::atoi(argv[2]), "custom"}};
        run_shape(one[0]);
        return 0;
    }
    std::printf("PTQ1_0 decode GEMV bandwidth, median of %d trials "
                "(weight bytes = rows*k*28/128; ceiling ~249.6 GB/s)\n", kTrials);
    for (const Payload& p : table) { run_shape(p); }
    return 0;
}
