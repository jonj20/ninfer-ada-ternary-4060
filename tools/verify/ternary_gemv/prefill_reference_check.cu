// PTQ1_0 批量 prefill 内核的数值对拍：内核 vs CPU 双精度参考。
//
// 为什么要有这个门禁：prefill 内核同时踩了三件事——四列 quad 的三值解码列映射、dp4a 的字节序、
// 以及 qh 那 8 列的 FFMA 累加。开发中真的踩中过一个：tail 在 token 循环里写成了赋值而不是累加，
// 循环结束后只剩最后一个 token 的 tail 值并被加到所有 token 上；端到端冒烟完全看不出来
// （生成的文本仍然连贯），只有逐元素对拍能抓到。
//
// 覆盖的形状刻意包含边界：单 token tile 装不满的 token 数（37）、行数不是 CTA 行块倍数（67）、
// 只有一组 K（没有中区）、多组 K，以及 tail 列被清零 / 不清零两种情况。

#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"
#include "ops/linear/ternary/ternary_rowsplit_prefill.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using ninfer::ops::detail::kPtq1PrefillQuads;
using ninfer::ops::detail::Ptq1QuantThreads;

void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

// CPU 侧参考解码：与 PTQ1SimtDecodeAtom::decode_one 同源（dequantize_row_ptq1_0 的逐权重形式）
float host_scale(const std::uint8_t* p) {
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(p);
    return __half2float(__ushort_as_half(bits));
}

float host_decode_one(const std::uint8_t* codes, const std::uint8_t* high, float scale, int index) {
    static const unsigned kPow3[5] = {1u, 3u, 9u, 27u, 81u};
    std::uint8_t raw;
    int trit;
    if (index < 80) {
        trit = index >> 4;
        raw = codes[index & 15];
    } else if (index < 120) {
        const int local = index - 80;
        trit = local >> 3;
        raw = codes[16 + (local & 7)];
    } else {
        const int local = index - 120;
        trit = local >> 1;
        raw = high[local & 1];
    }
    const std::uint8_t q = static_cast<std::uint8_t>(raw * kPow3[trit]);
    const int xi = static_cast<int>((static_cast<std::uint16_t>(q) * 3u) >> 8);
    return static_cast<float>(xi - 1) * scale;
}

struct Shape {
    int rows;
    int k;
    int tokens;
    int zero_tail;
};

}  // namespace

int main() {
    constexpr int kR = 8, kTT = 4, kRG = 8, kTG = 16;
    using Tile = ninfer::ops::detail::Ptq1PrefillTile<kR, kTT, kRG, kTG>;

    const Shape shapes[] = {
        {1, 128, 1, 1},    {1, 128, 1, 0},     {8, 128, 8, 1},     {8, 128, 8, 0},
        {64, 512, 64, 1},  {64, 512, 64, 0},   {64, 640, 32, 1},   {64, 640, 37, 1},
        {67, 640, 37, 1},  {64, 640, 128, 1},  {64, 640, 128, 0},  {67, 640, 37, 0},
    };

    std::printf("PTQ1_0 prefill 内核 vs CPU 参考（整表最大幅度归一，容差 2%% = bf16 舍入量级）\n");
    int failures = 0;
    for (const Shape& s : shapes) {
        const int groups = s.k / 128;
        std::vector<float> hx(static_cast<size_t>(s.k) * s.tokens);
        for (size_t i = 0; i < hx.size(); ++i) {
            hx[i] = std::sin(0.7f * static_cast<float>(i)) * (1.0f + 0.3f * (i % 5));
            const size_t col = i % static_cast<size_t>(s.k);
            if (s.zero_tail && (col % 128) >= 120) { hx[i] = 0.0f; }
        }
        std::vector<std::uint8_t> hc(static_cast<size_t>(s.rows) * groups * 24);
        std::vector<std::uint8_t> hh(static_cast<size_t>(s.rows) * groups * 2);
        std::vector<std::uint16_t> hs(static_cast<size_t>(s.rows) * groups);
        for (size_t i = 0; i < hc.size(); ++i) { hc[i] = static_cast<std::uint8_t>(i * 37 + 11); }
        for (size_t i = 0; i < hh.size(); ++i) { hh[i] = static_cast<std::uint8_t>(i * 53 + 7); }
        for (size_t i = 0; i < hs.size(); ++i) { hs[i] = static_cast<std::uint16_t>(0x3400 + (i % 64)); }

        __nv_bfloat16* dx = nullptr;
        std::int8_t* dq = nullptr;
        float* dqs = nullptr;
        std::uint8_t *dc = nullptr, *dh = nullptr, *ds = nullptr;
        __nv_bfloat16* dout = nullptr;
        check(cudaMalloc(&dx, hx.size() * 2));
        check(cudaMalloc(&dq, hx.size()));
        check(cudaMalloc(&dqs, static_cast<size_t>(s.tokens) * groups * 4));
        check(cudaMalloc(&dc, hc.size()));
        check(cudaMalloc(&dh, hh.size()));
        check(cudaMalloc(&ds, hs.size() * 2));
        check(cudaMalloc(&dout, static_cast<size_t>(s.rows) * s.tokens * 2));
        std::vector<__nv_bfloat16> hbf(hx.size());
        for (size_t i = 0; i < hx.size(); ++i) { hbf[i] = __float2bfloat16_rn(hx[i]); }
        check(cudaMemcpy(dx, hbf.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dc, hc.data(), hc.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dh, hh.data(), hh.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(ds, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
        check(cudaMemset(dout, 0, static_cast<size_t>(s.rows) * s.tokens * 2));

        const int qgrid = (s.tokens * groups + Ptq1QuantThreads / 32 - 1) / (Ptq1QuantThreads / 32);
        ninfer::ops::detail::ternary_ptq1_quantize_act_batch_kernel<<<qgrid, Ptq1QuantThreads>>>(
            dx, dq, dqs, s.k, s.tokens, groups);
        const dim3 grid((s.rows + Tile::kRowsPerCta - 1) / Tile::kRowsPerCta,
                        (s.tokens + Tile::kTokensPerCta - 1) / Tile::kTokensPerCta, 1u);
        ninfer::ops::detail::ternary_ptq1_prefill_dp4a_kernel<kR, kTT, kRG, kTG>
            <<<grid, Tile::kThreads>>>(dq, dqs, dc, dh, ds, dout, s.rows, s.k, s.tokens, groups,
                                       s.rows);
        check(cudaDeviceSynchronize());
        check(cudaGetLastError());

        std::vector<__nv_bfloat16> hn(static_cast<size_t>(s.rows) * s.tokens);
        std::vector<std::int8_t> hq(static_cast<size_t>(s.k) * s.tokens);
        std::vector<float> hqs(static_cast<size_t>(s.tokens) * groups);
        check(cudaMemcpy(hn.data(), dout, hn.size() * 2, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hq.data(), dq, hq.size(), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hqs.data(), dqs, hqs.size() * 4, cudaMemcpyDeviceToHost));

        double ref_absmax = 0.0, err = 0.0, se_ref = 0.0, se_new = 0.0;
        int worst_t = -1, worst_row = -1, n = 0;
        for (int t = 0; t < s.tokens; ++t) {
            for (int row = 0; row < s.rows; ++row) {
                double acc = 0.0;
                for (int g = 0; g < groups; ++g) {
                    double dot = 0.0;
                    for (int j = 0; j < 128; ++j) {
                        const float w = host_decode_one(
                            hc.data() + (static_cast<size_t>(row) * groups + g) * 24,
                            hh.data() + (static_cast<size_t>(row) * groups + g) * 2,
                            host_scale(reinterpret_cast<const std::uint8_t*>(
                                hs.data() + static_cast<size_t>(row) * groups + g)),
                            j);
                        dot += static_cast<double>(w) *
                               hq[static_cast<size_t>(t) * s.k + g * 128 + j];
                    }
                    acc += dot * hqs[static_cast<size_t>(t) * groups + g];
                }
                const double got =
                    __bfloat162float(hn[static_cast<size_t>(t) * s.rows + row]);
                ref_absmax = std::max(ref_absmax, std::fabs(acc));
                if (std::fabs(got - acc) > err) {
                    err = std::fabs(got - acc);
                    worst_t = t;
                    worst_row = row;
                }
                se_ref += acc * acc;
                se_new += got * got;
                ++n;
            }
        }
        const double rel = err / (ref_absmax > 0 ? ref_absmax : 1.0);
        const double rms_ratio = std::sqrt(se_ref / n) / (std::sqrt(se_new / n) + 1e-30);
        const bool pass = rel < 0.02;
        if (!pass) { ++failures; }
        std::printf("  rows=%-4d k=%-6d tokens=%-4d tail=%s  max_abs_err=%.3e (%.2e x max)  "
                    "RMS 比=%.5f  最差(token=%d,row=%d)  %s\n",
                    s.rows, s.k, s.tokens, s.zero_tail ? "清零" : "生效", err, rel, rms_ratio,
                    worst_t, worst_row, pass ? "PASS" : "FAIL");

        cudaFree(dx); cudaFree(dq); cudaFree(dqs); cudaFree(dc); cudaFree(dh); cudaFree(ds);
        cudaFree(dout);
    }
    if (failures == 0) {
        std::printf("prefill_reference: PASS\n");
        return 0;
    }
    std::printf("prefill_reference: FAIL（%d 个形状）\n", failures);
    return 1;
}
