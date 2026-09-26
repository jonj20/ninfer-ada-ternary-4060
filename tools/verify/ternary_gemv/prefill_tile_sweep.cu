// PTQ1_0 批量内核的 tile 形状扫描：同一份权重下不同 tile 的耗时 / 等效带宽 / CTA 数。
//
// 为什么需要这个工具：批量内核的三个 tile 维度是绑在一起的——
//   kRowsPerCta  = R * ROW_GROUPS
//   kTokensPerCta = TT * TOK_GROUPS
//   kThreads     = ROW_GROUPS * TOK_GROUPS
// 而 grid = (div_up(rows, kRowsPerCta), div_up(tokens, kTokensPerCta))。两条约束同时成立：
//
//   1. grid.y 必须为 1，也就是 kTokensPerCta >= T。同一 grid.x 的各个 y 会重读同一批权重行，
//      所以 kTokensPerCta < T 会让权重流量按 grid.y 翻倍。T=9 用 kTokensPerCta=4 会读三遍。
//   2. 在 kTokensPerCta 刚好覆盖 T 的前提下，kRowsPerCta 越小 CTA 数越多，DRAM 延迟藏得越好。
//      权重流量与 CTA 数是两头：CTA 太少会退化成延迟受限（实测 21 GB/s，而 decode 的 GEMV
//      是 136 GB/s），CTA 太多则每 CTA 的行块太小、复用不足。
//
// 这两条就是开发中把小 T 验证从 921 ms 降到 344 ms 的依据。改 launch_ternary_gemm_t8 里的
// tile 分派之后，用这个工具复核「当前选择是否仍然最优」。
//
// 判据：只报数，不判定。性能噪声在本卡约 ±20%（SM 频率随电源在 1.68–2.01 GHz 摆动），
// 硬阈值会误报；但当某个候选比当前分派快 15% 以上时会显式标出，由人决定要不要换。
// 数值正确性由 prefill_reference_check.cu 把关，两者职责不重叠。

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"
#include "ops/linear/ternary/ternary_rowsplit_prefill.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

using namespace ninfer::ops::detail;

namespace {

void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

float time_ms(const std::function<void()>& fn, int iters) {
    float best = 1.0e30f;
    for (int t = 0; t < iters; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        check(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<float, std::milli>(t1 - t0).count());
    }
    return best;
}

struct Shape {
    int rows;
    int k;
    int name_id;
    int tokens;
};

// 名字与权重行宽绑定，见 main 里的 shapes 表。
const char* shape_name(int id) {
    switch (id) {
        case 0: return "MLP gate_up";
        case 1: return "MLP down";
        case 2: return "attn/GDN";
        case 3: return "LM head";
        default: return "?";
    }
}

// 权重字节：codes 24 + high 2 + scales 2 每 (row, group)。
double weight_bytes(const Shape& s, int groups) {
    return static_cast<double>(s.rows) * groups * 28.0;
}

struct Result {
    float ms;
    long ctas;
    long threads;
};

template <int R, int TT, int ROW_GROUPS, int TOK_GROUPS>
Result run_tile(const Shape& s, int groups, const __nv_bfloat16* x_bf16, std::int8_t* xq,
                float* xq_scale, const std::uint8_t* codes, const std::uint8_t* high,
                const std::uint8_t* scales, __nv_bfloat16* out, int iters) {
    using Tile = Ptq1PrefillTile<R, TT, ROW_GROUPS, TOK_GROUPS>;
    const dim3 grid(static_cast<unsigned>((s.rows + Tile::kRowsPerCta - 1) / Tile::kRowsPerCta),
                    static_cast<unsigned>((s.tokens + Tile::kTokensPerCta - 1) /
                                          Tile::kTokensPerCta),
                    1u);
    const int qgrid = (s.tokens * groups + static_cast<int>(Ptq1QuantThreads) / 32 - 1) /
                      (static_cast<int>(Ptq1QuantThreads) / 32);
    const float ms = time_ms(
        [&] {
            ternary_ptq1_quantize_act_batch_kernel<<<qgrid, Ptq1QuantThreads>>>(
                x_bf16, xq, xq_scale, s.k, s.tokens, groups);
            ternary_ptq1_prefill_dp4a_kernel<R, TT, ROW_GROUPS, TOK_GROUPS>
                <<<grid, Tile::kThreads>>>(xq, xq_scale, codes, high, scales, out, s.rows, s.k,
                                            s.tokens, groups, s.rows);
        },
        iters);
    check(cudaGetLastError());
    return Result{ms, static_cast<long>(grid.x) * grid.y,
                  static_cast<long>(grid.x) * grid.y * Tile::kThreads};
}

}  // namespace

int main(int argc, char** argv) {
    // tokens 属于形状；分派按 T 分档，所以每档单独扫。
    struct Case {
        int tokens;
        const char* bucket;
    };
    const Case cases[] = {
        {2, "T=2   (draft=1 验证)"}, {3, "T=3   (draft=2 验证)"}, {4, "T=4   (draft=3 验证)"},
        {9, "T=9   (server prefill)"}, {16, "T=16"},              {32, "T=32"},
        {64, "T=64"},
    };
    const int kCaseCount = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

    const Shape shapes[] = {
        {34816, 5120, 0},   // MLP gate_up
        {5120, 17408, 1},   // MLP down
        {5120, 5120, 2},    // attn / GDN
        {248320, 5120, 3},  // LM head
    };
    const int kShapeCount = static_cast<int>(sizeof(shapes) / sizeof(shapes[0]));

    int iters = 5;
    int only_case = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--case" && i + 1 < argc) {
            only_case = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "用法：%s [--iters N] [--case 0..%d]\n", argv[0], kCaseCount - 1);
            return 2;
        }
    }

    std::printf("PTQ1_0 批量内核 tile 扫描（本机噪声约 ±20%%，只报数不判定）\n");
    std::printf("约束：kTokensPerCta >= T 才能让 grid.y=1，否则权重流量按 grid.y 翻倍\n\n");

    for (int ci = 0; ci < kCaseCount; ++ci) {
        if (only_case >= 0 && ci != only_case) { continue; }
        const int tokens = cases[ci].tokens;
        std::printf("########## %s ##########\n", cases[ci].bucket);

        for (int si = 0; si < kShapeCount; ++si) {
            Shape s = shapes[si];
            s.tokens = tokens;
            const int groups = s.k / 128;

            std::vector<__nv_bfloat16> x_in(static_cast<std::size_t>(s.k) * tokens);
            for (std::size_t i = 0; i < x_in.size(); ++i) {
                x_in[i] = __float2bfloat16_rn(std::sin(0.01f * static_cast<float>(i)));
            }
            std::vector<std::int8_t> xq(static_cast<std::size_t>(s.k) * tokens);
            std::vector<float> xq_scale(static_cast<std::size_t>(tokens) * groups);
            std::vector<std::uint8_t> codes(static_cast<std::size_t>(s.rows) * groups * 24);
            std::vector<std::uint8_t> high(static_cast<std::size_t>(s.rows) * groups * 2);
            std::vector<std::uint8_t> scales(static_cast<std::size_t>(s.rows) * groups * 2);
            std::vector<__nv_bfloat16> out(static_cast<std::size_t>(s.rows) * tokens);
            for (std::size_t i = 0; i < codes.size(); ++i) {
                codes[i] = static_cast<std::uint8_t>(i * 37 + 11);
            }
            for (std::size_t i = 0; i < high.size(); ++i) {
                high[i] = static_cast<std::uint8_t>(i * 53 + 7);
            }
            for (std::size_t i = 0; i < scales.size(); ++i) {
                scales[i] = static_cast<std::uint8_t>(i * 29 + 3);
            }
            __nv_bfloat16* d_x_in = nullptr;
            std::int8_t* d_xq = nullptr;
            float* d_scale = nullptr;
            std::uint8_t *d_codes = nullptr, *d_high = nullptr, *d_scales = nullptr;
            __nv_bfloat16* d_out = nullptr;
            check(cudaMalloc(&d_x_in, x_in.size() * 2));
            check(cudaMalloc(&d_xq, xq.size()));
            check(cudaMalloc(&d_scale, xq_scale.size() * 4));
            check(cudaMalloc(&d_codes, codes.size()));
            check(cudaMalloc(&d_high, high.size()));
            check(cudaMalloc(&d_scales, scales.size()));
            check(cudaMalloc(&d_out, out.size() * 2));
            check(cudaMemcpy(d_x_in, x_in.data(), x_in.size() * 2, cudaMemcpyHostToDevice));
            check(cudaMemcpy(d_xq, xq.data(), xq.size(), cudaMemcpyHostToDevice));
            check(cudaMemcpy(d_scale, xq_scale.data(), xq_scale.size() * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice));
            check(cudaMemcpy(d_high, high.data(), high.size(), cudaMemcpyHostToDevice));
            check(cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice));

            const double wb = weight_bytes(s, groups);
            std::printf("=== %s  rows=%d k=%d T=%d  权重 %.1f MB ===\n", shape_name(s.name_id),
                        s.rows, s.k, tokens, wb / 1e6);

            // 参考平铺内核（launch_by_qtype<8> 的那条）作为量级参照。
            {
                const float ms = time_ms(
                    [&] {
                        ternary_rowsplit_gemm_kernel<PTQ1RowSplitStorage, PTQ1SimtDecodeAtom, 8>
                            <<<dim3(s.rows, (tokens + 7) / 8), PTQ1RowSplitStorage::kGroupK>>>(
                                reinterpret_cast<const __nv_bfloat16*>(d_xq), d_codes, d_high,
                                d_scales, d_out, s.rows, s.k, tokens, groups, s.rows);
                    },
                    iters);
                check(cudaGetLastError());
                std::printf("  %-22s %9.3f ms  %7.1f GB/s   (参考平铺内核，非本路径)\n", "reference t8",
                            ms, wb / (ms * 1e-3) / 1e9);
            }

            float best_ms = 1.0e30f;
            char best_tag[64] = "";
            float cur_ms = 0.0f;
            char cur_tag[64] = "";

#define SWEEP(R, TT, RG, TG, IS_CURRENT)                                                 \
    do {                                                                                 \
        const Result r =                                                                  \
            run_tile<R, TT, RG, TG>(s, groups, d_x_in, d_xq, d_scale, d_codes, d_high,    \
                                    d_scales, d_out, iters);                               \
        char tag[64];                                                                      \
        std::snprintf(tag, sizeof(tag), "R%d TT%d RG%d TG%d", R, TT, RG, TG);               \
        std::printf("  %-22s %9.3f ms  %7.1f GB/s  CTA=%-6ld thr=%-8ld", tag, r.ms,         \
                    wb / (r.ms * 1e-3) / 1e9, r.ctas, r.threads);                          \
        if (r.ms < best_ms) { best_ms = r.ms; std::snprintf(best_tag, sizeof(best_tag),   \
                                                            "%s", tag); }                  \
        if (IS_CURRENT) { cur_ms = r.ms; std::snprintf(cur_tag, sizeof(cur_tag), "%s",    \
                                                       tag); }                             \
    } while (0)

            // 引擎当前分派（见 ternary_rowsplit_gemm.cu 的 launch_ternary_gemm_t8）：
            //   T >= 64 -> <8,4,8,16>；9..63 -> <8,4,8,4>；3..4 -> <1,1,16,4>；2 -> <1,1,16,2>
            const bool current_2 = tokens == 2;
            const bool current_4 = tokens == 3 || tokens == 4;
            const bool current_9_63 = tokens >= 9 && tokens < 64;
            const bool current_64 = tokens >= 64;

            if (current_2) { SWEEP(1, 1, 16, 2, true); }
            if (current_4) { SWEEP(1, 1, 16, 4, true); }
            if (current_9_63) { SWEEP(8, 4, 8, 4, true); }
            if (current_64) { SWEEP(8, 4, 8, 16, true); }

            // 备选：同 kTokensPerCta 下换 kRowsPerCta / 线程数。
            // <8,4,8,4> 是 2026-09-26 之前 T<16 一律用的配置，保留在扫描里当基线：
            // kRowsPerCta=64 让 n 大时只剩几十个 CTA，是小 T 退化成延迟受限的直接原因。
            SWEEP(8, 4, 8, 4, false);
            if (tokens <= 2) {
                SWEEP(2, 1, 16, 2, false);
                SWEEP(1, 1, 32, 2, false);
                SWEEP(1, 1, 8, 2, false);
                SWEEP(4, 1, 8, 2, false);
                SWEEP(1, 2, 8, 4, false);
                SWEEP(2, 1, 8, 4, false);
                SWEEP(1, 1, 16, 4, false);
            } else if (tokens <= 4) {
                SWEEP(2, 1, 8, 4, false);
                SWEEP(1, 1, 8, 4, false);
                SWEEP(4, 1, 8, 4, false);
                SWEEP(2, 1, 16, 4, false);
                SWEEP(1, 1, 32, 4, false);
                SWEEP(1, 1, 16, 2, false);
                SWEEP(2, 1, 16, 2, false);
                SWEEP(1, 2, 8, 4, false);
            } else {
                SWEEP(8, 4, 8, 16, false);
                SWEEP(4, 2, 8, 16, false);
                SWEEP(2, 2, 8, 16, false);
                SWEEP(1, 2, 8, 16, false);
                SWEEP(1, 4, 8, 8, false);
                SWEEP(2, 4, 8, 8, false);
                SWEEP(1, 1, 16, 16, false);
                SWEEP(1, 2, 16, 8, false);
            }
#undef SWEEP

            std::printf("  -> 当前分派 %s: %.3f ms；本档最优 %s: %.3f ms", cur_tag, cur_ms, best_tag,
                        best_ms);
            if (cur_ms > best_ms * 1.15f) {
                std::printf("   <<< 有候选快 %.0f%%，值得复核分派", (cur_ms / best_ms - 1.0) * 100.0);
            }
            std::printf("\n\n");

            cudaFree(d_x_in);
            cudaFree(d_xq);
            cudaFree(d_scale);
            cudaFree(d_codes);
            cudaFree(d_high);
            cudaFree(d_scales);
            cudaFree(d_out);
        }
    }
    return 0;
}
