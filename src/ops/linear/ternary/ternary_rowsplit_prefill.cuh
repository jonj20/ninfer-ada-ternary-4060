// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

// PTQ1_0 批量 prefill 解码（T >= 16）。
//
// 为什么需要它：参考分块 GEMM（ternary_rowsplit_gemm_kernel）每个 CTA 只做一个输出行，于是激活向量
// 被重复读 rows 次——gate_up 形状每层约 108 GB 的 L2 流量，实测只有 0.27 TMAC/s（0.4% 的 FFMA 利用率）。
// 解码路径（ternary_ptq1_gemv_recurrence_kernel）那套 int8 激活 + dp4a 当时只接了 T == 1，预填充完全没吃到。
//
// 本内核把两件事一起做：
//   * 每 CTA 覆盖 ROW_GROUPS*R 个输出行，激活流量降为 rows/(ROW_GROUPS*R) 次；
//   * 激活按 128 列组量化成 int8，一个 dp4a 覆盖 4 个权重×1 个 token，替代每权重每 token 一次 FFMA。
// 实测 sm_89：gate_up x11.7、MLP down x12.1、attn/GDN x11.0、LM head x11.4（对参考 t8 内核）。
//
// 激活布局与权重平面布局沿用仓内既有约定：x 是 [K, T] 且 K 连续（token*k + column），输出 [N, T]
// 同理（token*out_row_stride + row）；权重的三个平面按 [rows, groups] 行主序。

#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// 前 120 列 = 30 个四列 quad（走 dp4a），qh 的 8 列解码形状不同，逐 token 用 FFMA 累加。
inline constexpr int kPtq1PrefillQuads = 30;

// 批量量化 kernel 的线程数（必须是 32 的倍数；一个 warp 处理一个 (token, group)）。
inline constexpr int Ptq1QuantThreads = 256;

// 每 CTA 的形状：R 个输出行/线程、TT 个 token/线程，线程按 ROW_GROUPS（行组）× TOK_GROUPS（token 组）排布。
template <int R, int TT, int ROW_GROUPS, int TOK_GROUPS>
struct Ptq1PrefillTile {
    static constexpr int kRowsPerCta  = R * ROW_GROUPS;
    static constexpr int kTokensPerCta = TT * TOK_GROUPS;
    static constexpr int kThreads      = ROW_GROUPS * TOK_GROUPS;
};

// 批量激活量化：x[K,T] bf16 -> xq[K,T] int8 + xqs[T, groups] fp32。
// 一个 (token, group) 由一个 warp 处理（4 个值/线程 + warp 内归约），按 blockIdx 分配，
// 保证每个 (token, group) 只算一次。
__global__ __launch_bounds__(256)
void ternary_ptq1_quantize_act_batch_kernel(const __nv_bfloat16* __restrict__ x,
                                           std::int8_t* __restrict__ xq,
                                           float* __restrict__ xq_scale, std::int32_t k,
                                           std::int32_t tokens, std::int32_t groups) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int nwarps = static_cast<int>(blockDim.x) >> 5;
    const int total = tokens * groups;
    for (int idx = static_cast<int>(blockIdx.x) * nwarps + warp; idx < total;
         idx += static_cast<int>(gridDim.x) * nwarps) {
        const int t = idx / groups;
        const int g = idx - t * groups;
        const int base = g * 128 + lane * 4;
        // 4 个 bf16 = 8 字节：float2 天然满足 8 字节对齐（float4 要 16 字节而 lane*4 只有 8 的倍数）
        const float2 raw =
            *reinterpret_cast<const float2*>(x + static_cast<std::int64_t>(t) * k + base);
        const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float2 lo = __bfloat1622float2(bp[0]);
        const float2 hi = __bfloat1622float2(bp[1]);
        const float f[4] = {lo.x, lo.y, hi.x, hi.y};
        float m = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { m = fmaxf(m, fabsf(f[i])); }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o)); }
        const float inv = (m > 0.0f) ? (127.0f / m) : 0.0f;
        xq_scale[idx] = m * (1.0f / 127.0f);
        int q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, f[i] * inv)));
        }
        *reinterpret_cast<int*>(xq + static_cast<std::int64_t>(t) * k + base) =
            (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | ((q[3] & 0xFF) << 24);
    }
}

// 一个四列 quad 的三值解码成 4 个 int8（{-1,0,1}）。
// q < 20 取前区（trit = q>>2，字节 4*(q&3)），q < 30 取中区（trit 与字节由局部偏移推出），
// 与解码 GEMV 的列映射逐位一致。
__device__ __forceinline__ int ptq1_prefill_decode_quad(const std::uint8_t* __restrict__ c, int q) {
    const int local = (q < 20) ? 0 : 4 * q - 80;
    const int byte_off = (q < 20) ? (4 * (q & 3)) : (16 + (local & 7));
    const int trit = (q < 20) ? (q >> 2) : (local >> 3);
    const std::uint32_t w = *reinterpret_cast<const std::uint32_t*>(c + byte_off);
    std::uint32_t v_lo = __byte_perm(w, 0u, 0x4140u);
    std::uint32_t v_hi = __byte_perm(w, 0u, 0x4342u);
    const unsigned p3 = ternary_pow3(trit);
    v_lo = (v_lo * p3) & 0x00FF00FFu;
    v_hi = (v_hi * p3) & 0x00FF00FFu;
    v_lo *= 3u;
    v_hi *= 3u;
    return __vsub4(__byte_perm(v_lo, v_hi, 0x7531u), 0x01010101u);
}

// qh 的 4 个 trit（连续乘三得到），权重为 trit-1。
__device__ __forceinline__ void ptq1_prefill_tail4(std::uint32_t byte, float (&w)[4]) {
    std::uint32_t q = byte & 0xFFu;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        w[t] = static_cast<float>((q * 3u) >> 8) - 1.0f;
        q = (q * 3u) & 0xFFu;
    }
}

template <int R, int TT, int ROW_GROUPS, int TOK_GROUPS>
__global__ __launch_bounds__(Ptq1PrefillTile<R, TT, ROW_GROUPS, TOK_GROUPS>::kThreads)
void ternary_ptq1_prefill_dp4a_kernel(const std::int8_t* __restrict__ xq,
                                      const float* __restrict__ xq_scale,
                                      const std::uint8_t* __restrict__ codes,
                                      const std::uint8_t* __restrict__ high,
                                      const std::uint8_t* __restrict__ scales,
                                      __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                      std::int32_t k, std::int32_t tokens,
                                      std::int32_t groups_per_row, std::int32_t out_row_stride) {
    using Tile = Ptq1PrefillTile<R, TT, ROW_GROUPS, TOK_GROUPS>;
    const int tid = static_cast<int>(threadIdx.x);
    const int row0 = static_cast<int>(blockIdx.x) * Tile::kRowsPerCta + (tid % ROW_GROUPS) * R;
    const int tok0 = static_cast<int>(blockIdx.y) * Tile::kTokensPerCta + (tid / ROW_GROUPS) * TT;

    float acc[R][TT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < TT; ++t) { acc[r][t] = 0.0f; }
    }

    const std::uint8_t* cr = codes + static_cast<std::int64_t>(row0) * groups_per_row * 24;
    const std::uint8_t* hr = high + static_cast<std::int64_t>(row0) * groups_per_row * 2;
    const std::uint8_t* sr = scales + static_cast<std::int64_t>(row0) * groups_per_row * 2;

    for (int g = 0; g < groups_per_row; ++g) {
        // 越界 token 的下标夹到 0：这些结果不会写出，但避免越界读
        int tc[TT];
        float as[TT];
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            tc[t] = (tok0 + t < tokens) ? t : 0;
            as[t] = xq_scale[static_cast<std::int64_t>(tok0 + tc[t]) * groups_per_row + g];
        }
        const std::int8_t* xg = xq + static_cast<std::int64_t>(tok0) * k + g * 128;

#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= rows) { break; }
            const std::uint8_t* cg =
                cr + static_cast<std::int64_t>(r) * groups_per_row * 24 + g * 24;
            const float ws =
                ternary_scale(sr + static_cast<std::int64_t>(r) * groups_per_row * 2 + g * 2);

            int sum[TT];
#pragma unroll
            for (int t = 0; t < TT; ++t) { sum[t] = 0; }
#pragma unroll
            for (int q = 0; q < kPtq1PrefillQuads; ++q) {
                const int wq = ptq1_prefill_decode_quad(cg, q);
                const std::int8_t* aqp = xg + q * 4;
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    sum[t] = __dp4a(wq, *reinterpret_cast<const int*>(aqp + tc[t] * k), sum[t]);
                }
            }

            const std::uint16_t hv = *reinterpret_cast<const std::uint16_t*>(
                hr + static_cast<std::int64_t>(r) * groups_per_row * 2 + g * 2);
            float wb[4];
            float wa[4];
            ptq1_prefill_tail4(hv & 0xFFu, wb);
            ptq1_prefill_tail4((hv >> 8) & 0xFFu, wa);
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                const std::int8_t* ap = xg + 120 + tc[t] * k;
                const float tv =
                    fmaf(wb[0], static_cast<float>(ap[0]),
                    fmaf(wa[0], static_cast<float>(ap[1]),
                    fmaf(wb[1], static_cast<float>(ap[2]),
                    fmaf(wa[1], static_cast<float>(ap[3]),
                    fmaf(wb[2], static_cast<float>(ap[4]),
                    fmaf(wa[2], static_cast<float>(ap[5]),
                    fmaf(wb[3], static_cast<float>(ap[6]),
                         wa[3] * static_cast<float>(ap[7]))))))));
                if (tok0 + t < tokens) {
                    acc[r][t] = fmaf(ws * as[t], static_cast<float>(sum[t]) + tv, acc[r][t]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (row0 + r < rows && tok0 + t < tokens) {
                out[static_cast<std::int64_t>(tok0 + t) * out_row_stride + row0 + r] =
                    __float2bfloat16_rn(acc[r][t]);
            }
        }
    }
}

} // namespace ninfer::ops::detail
