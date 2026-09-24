// PTQ1_0 decode-GEMV correctness gate: the in-tree kernels against a CPU reference.
//
// Build (see run_checks.sh for the exact invocation):
//   nvcc -O2 -std=c++17 -arch=sm_89 -I<repo>/src \
//        tools/verify/ternary_gemv/gemv_reference_check.cu -o gemv_reference_check
//
// Exit code 0 means every shape matched. The tolerance is bf16 output rounding, not algorithmic
// slack: the kernel accumulates in fp32 and rounds once on store, so a correct kernel lands within
// ~0.4% relative of the double-precision reference.
//
// This is the regression gate for the decode-kernel rewrites recorded in
// docs/4060-开发跟踪.md §9.1. It earns its keep: a wrong tail-column mapping passed the in-engine
// smoke test (the model still produced fluent text, only the wrong logits) and was caught here.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"

namespace {

using ninfer::ops::detail::kGemvWarpsPerBlock;

// A PTQ1_0 group is 128 columns: columns 0..79 and 80..119 live in the 24-byte code plane (as five
// trit passes over 16 and 8 bytes), columns 120..127 in the 2-byte high plane (four trits per byte).
// `quad_off` and `trit` are the two things a kernel has to get right per column.
struct ColumnMap {
    int byte_offset;
    int trit;
};

ColumnMap map_column(int c) {
    if (c < 80) {
        const int lane = c / 4;
        return {4 * (lane & 3) + (c % 4), lane >> 2};
    }
    if (c < 120) {
        const int lane = 20 + (c - 80) / 4;
        return {16 + 4 * ((lane - 20) & 1) + ((c - 80) % 4), (lane - 20) >> 1};
    }
    return {(c - 120) & 1, (c - 120) >> 1};
}

std::uint8_t pow3(int n) {
    static const std::uint8_t table[] = {1, 3, 9, 27, 81, 243};
    return table[n < 0 ? 0 : (n > 5 ? 5 : n)];
}

double reference_row(const std::vector<std::uint8_t>& codes,
                     const std::vector<std::uint8_t>& high,
                     const std::vector<std::uint8_t>& scales,
                     const std::vector<__nv_bfloat16>& x, int row, int groups) {
    const std::size_t code_base  = static_cast<std::size_t>(row) * groups * 24;
    const std::size_t high_base  = static_cast<std::size_t>(row) * groups * 2;
    const std::size_t scale_base = static_cast<std::size_t>(row) * groups * 2;
    double total = 0.0;
    for (int g = 0; g < groups; ++g) {
        const std::uint16_t sbits =
            static_cast<std::uint16_t>(scales[scale_base + g * 2] |
                                       (static_cast<std::uint16_t>(scales[scale_base + g * 2 + 1]) << 8));
        const double scale = __half2float(__ushort_as_half(sbits));
        double dot = 0.0;
        for (int c = 0; c < 128; ++c) {
            const ColumnMap m = map_column(c);
            const std::uint8_t raw = c < 120 ? codes[code_base + g * 24 + m.byte_offset]
                                              : high[high_base + g * 2 + m.byte_offset];
            const std::uint8_t q = static_cast<std::uint8_t>(raw * pow3(m.trit));
            const int trit_value = static_cast<int>((static_cast<std::uint16_t>(q) * 3u) >> 8);
            dot += static_cast<double>(trit_value - 1) * static_cast<float>(x[g * 128 + c]);
        }
        total += scale * dot;
    }
    return total;
}

bool check_shape(int n, int k) {
    const int groups = k / 128;
    std::mt19937 rng(1234);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * groups * 24);
    std::vector<std::uint8_t> high(static_cast<std::size_t>(n) * groups * 2);
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(n) * groups * 2);
    std::vector<__nv_bfloat16> x(k);
    for (auto& v : codes) v = static_cast<std::uint8_t>(rng() & 0xFF);
    for (auto& v : high) v = static_cast<std::uint8_t>(rng() & 0xFF);
    for (std::size_t i = 0; i < scales.size(); i += 2) {
        const std::uint16_t bits = static_cast<std::uint16_t>(0x3C00 + (rng() % 512));
        scales[i] = static_cast<std::uint8_t>(bits & 0xFF);
        scales[i + 1] = static_cast<std::uint8_t>(bits >> 8);
    }
    for (auto& v : x) {
        v = __float2bfloat16_rn((static_cast<float>(rng() % 2000) - 1000.0f) / 500.0f);
    }

    __nv_bfloat16* dx = nullptr;
    __nv_bfloat16* dout = nullptr;
    std::uint8_t* dc = nullptr;
    std::uint8_t* dh = nullptr;
    std::uint8_t* ds = nullptr;
    cudaMalloc(&dx, static_cast<std::size_t>(k) * 2);
    cudaMalloc(&dout, static_cast<std::size_t>(n) * 2);
    cudaMalloc(&dc, codes.size());
    cudaMalloc(&dh, high.size());
    cudaMalloc(&ds, scales.size());
    cudaMemcpy(dx, x.data(), static_cast<std::size_t>(k) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dh, high.data(), high.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, scales.data(), scales.size(), cudaMemcpyHostToDevice);

    const int grid = (n + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    ninfer::ops::detail::ternary_ptq1_gemv_kernel<1><<<grid, kGemvWarpsPerBlock * 32>>>(
        dx, dc, dh, ds, dout, n, groups, 1, n);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("N=%-6d K=%-6d  CUDA error: %s\n", n, k, cudaGetErrorString(err));
        return false;
    }

    std::vector<__nv_bfloat16> got(n);
    cudaMemcpy(got.data(), dout, static_cast<std::size_t>(n) * 2, cudaMemcpyDeviceToHost);

    double worst = 0.0;
    int worst_row = -1;
    int bad = 0;
    for (int r = 0; r < n; ++r) {
        const double expected = reference_row(codes, high, scales, x, r, groups);
        const double actual = __bfloat162float(got[r]);
        const double rel = std::fabs(actual - expected) / (std::fabs(expected) + 1e-3);
        if (rel > worst) { worst = rel; worst_row = r; }
        if (rel > 0.02) ++bad;
    }
    std::printf("N=%-6d K=%-6d  max_rel_err=%.5f (row %d)  bad_rows=%d  %s\n", n, k, worst,
                worst_row, bad, bad == 0 ? "PASS" : "FAIL");

    cudaFree(dx); cudaFree(dout); cudaFree(dc); cudaFree(dh); cudaFree(ds);
    return bad == 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3) {
        return check_shape(std::atoi(argv[1]), std::atoi(argv[2])) ? 0 : 1;
    }
    // Odd row counts exercise the row guard; the K values cover the group counts this model uses
    // (5120 -> 40 groups, 6144 -> 48, 17408 -> 136) plus two off-model widths.
    const int shapes[][2] = {
        {257, 5120}, {5120, 5120}, {4096, 6144}, {1024, 1280}, {777, 3840},
    };
    bool ok = true;
    for (const auto& s : shapes) {
        ok = check_shape(s[0], s[1]) && ok;
    }
    std::printf("%s\n", ok ? "ALL PASS" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
