// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

// Decode-shaped ternary GEMV (T == 1) for PQ2_0: one warp owns one output row.
//
// Why: the reference kernel in ternary_rowsplit_gemm.cuh gives each output row a full 128-thread
// CTA and seven block-wide barriers. That is fine for correctness but leaves decode far from the
// card's measured bandwidth. Here each lane covers four consecutive weights of every 128-group, so
// a warp reads exactly one 32-byte code span plus one 2-byte scale per group and reduces through
// shuffles with no __syncthreads at all.
//
// PQ2_0 packing makes the mapping exact rather than approximate: a group is 32 bytes holding four
// two-bit codes each, so for lane l the four columns 4l..4l+3 live entirely in byte l. Lane l
// loads one byte, decodes four weights, and consumes four bf16 activations.
//
// Occupancy, not instruction count, is what this kernel is tuned for: a GEMV needs many warps in
// flight to cover DRAM latency. A two-rows-per-warp variant (which reuses the activation loads)
// measured SLOWER (32.8 vs 46.4 t/s) because the extra accumulators and row bases cost registers
// and dropped the resident warp count. So: one row per warp, and only the micro-optimisations that
// remove instructions without adding state -- one 16-bit scale load, paired bf16 activation
// loads, and the per-group scale multiply hoisted out of the four FMAs.
//
// LAYOUT: ninfer/ggml put ne[0] on the contiguous axis, so a [k, 1] activation keeps element
// (column, 0) at column, and a one-token output row is simply out[row].

#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Warps per block for the GEMV below.
inline constexpr int kGemvWarpsPerBlock = 8;

inline constexpr int kGemvCodeBytesPerGroup  = 32;
inline constexpr int kGemvScaleBytesPerGroup = 2;
inline constexpr int kGemvGroupK             = 128;

__device__ __forceinline__ float gemv_scale(const std::uint8_t* scale_ptr) {
    // One 16-bit load instead of two byte loads plus a shift/or; the scale plane is 2-byte aligned.
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);
    return __half2float(__ushort_as_half(bits));
}

__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                             const std::uint8_t* __restrict__ codes,
                             const std::uint8_t* __restrict__ scales,
                             __nv_bfloat16* __restrict__ out, std::int32_t rows,
                             std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low        = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 high = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));

        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                               fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                                    low.y,
                                    fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                                         high.x,
                                         static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                             high.y)));
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup), dot,
                           accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// ---------------------------------------------------------------------------
// PTQ1_0 warp-per-row decode GEMV (T == 1).
//
// Shape identical to the PQ2_0 kernel: lane l owns columns 4l..4l+3 of every 128-group, so the
// activation reads stay two contiguous float2 at base = group*128 + 4l. The base-3 decode is
// the part that differs, and it takes the four-wide SIMD trick from llama.cpp's
// vec_dot_ptq1_0_q8_1_multi instead of the reference kernel's per-weight scalar stage walk
// (branchy index < 80 / < 120 / qh tail + a pow3 fan-out + two multiplies per weight):
//   * widen four code bytes into 16-bit lanes (__byte_perm 0x4140 / 0x4342)
//   * multiply every lane by 3^trit and keep the low byte (carry shield 0x00FF00FF)
//   * multiply by three one more time; the high byte of each 16-bit lane is now
//     ((raw * 3^trit mod 256) * 3) >> 8 = the ternary code of the reference decode
//   * __vsub4 subtracts one byte-wise, landing four weights in {-1, 0, 1} in one register.
// Per-lane decode cost drops from ~15 instructions per weight to ~6 per FOUR weights.
//
// Per-lane mapping inside a group (codes = 24-byte qs plane, high = 2-byte qh plane):
//   lane  0..19  qs front quad 4*(lane&3),  trit lane>>2          -> columns 4l..4l+3 (0..79)
//   lane 20..29  qs mid quad   16 + 4*((lane-20)&1), trit (lane-20)>>1 -> columns 80..119
//   lane 30..31  qh tail, scalar, columns 120..127  (8 of 128 columns, decode cost negligible)
// The four output bytes of one SIMD step map to the four consecutive columns, so the FMA chain
// feeds lo.x/lo.y/hi.x/hi.y in weight order exactly as the PQ2_0 kernel does.
template <int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_ptq1_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ high,
                              const std::uint8_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out, std::int32_t rows,
                              std::int32_t groups_per_row, std::int32_t tokens,
                              std::int32_t out_row_stride) {
    static_assert(kT >= 1 && kT <= 8, "tile size must be small enough to keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * 24;
    const std::uint8_t* high_row =
        high + static_cast<std::int64_t>(warp) * groups_per_row * 2;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * 2;

    // Per-lane guilding for the SIMD quad: which 4-byte span and which trit index.
    const bool is_tail      = lane >= 30;
    const int  quad_off     = lane < 20 ? 4 * (lane & 3)
                                        : 16 + 4 * ((lane - 20) & 1);
    const int  trit         = lane < 20 ? lane >> 2 : (lane - 20) >> 1;
    const std::uint8_t pow3 = is_tail ? 1u
                                      : trit <= 0 ? 1u : trit == 1 ? 3u : trit == 2 ? 9u : trit == 3 ? 27u : 81u;
    std::uint32_t v_lo = 0u, v_hi = 0u;

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);

        if (!is_tail) {
            // One 4-byte load per group per lane: bytes 4*(lane&3)..+3 (front) or 16+.. (mid).
            v_lo = __byte_perm(*reinterpret_cast<const std::uint32_t*>(code_row + group * 24 + quad_off), 0u, 0x4140u);
            v_hi = __byte_perm(*reinterpret_cast<const std::uint32_t*>(code_row + group * 24 + quad_off), 0u, 0x4342u);
            v_lo = (v_lo * pow3) & 0x00FF00FFu;
            v_hi = (v_hi * pow3) & 0x00FF00FFu;
            v_lo *= 3u;
            v_hi *= 3u;
        }

#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK;
                const float2 lo = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 hi = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));

                float w0, w1, w2, w3;
                if (!is_tail) {
                    // Four signed bytes of the SIMD step: each is one weight in {-1, 0, 1}.
                    const std::uint32_t q =
                        __vsub4(__byte_perm(v_lo, v_hi, 0x7531u), 0x01010101u);
                    w0 = static_cast<float>(static_cast<std::int8_t>(q & 0xFFu));
                    w1 = static_cast<float>(static_cast<std::int8_t>((q >> 8) & 0xFFu));
                    w2 = static_cast<float>(static_cast<std::int8_t>((q >> 16) & 0xFFu));
                    w3 = static_cast<float>(static_cast<std::int8_t>((q >> 24) & 0xFFu));
                } else {
                    // qh tail: column 4l+j = 120 + n*2 + h, byte qh[h] at trit n.
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int c_local = (lane - 30) * 4 + j;
                        const int n_tail  = c_local >> 1;
                        const int h_tail  = c_local & 1;
                        const std::uint8_t raw = high_row[group * 2 + h_tail];
                        const std::uint8_t qb  = static_cast<std::uint8_t>(raw * ternary_pow3(n_tail));
                        const int xi = static_cast<int>((static_cast<std::uint16_t>(qb) * 3u) >> 8);
                        if (j == 0) { w0 = static_cast<float>(xi - 1); }
                        else if (j == 1) { w1 = static_cast<float>(xi - 1); }
                        else if (j == 2) { w2 = static_cast<float>(xi - 1); }
                        else { w3 = static_cast<float>(xi - 1); }
                    }
                }
                const float dot = fmaf(w0, lo.x, fmaf(w1, lo.y, fmaf(w2, hi.x, w3 * hi.y)));
                accumulator[t] = fmaf(scale, dot, accumulator[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0 && t < tokens) {
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}

// Small-token-tile variant, for the speculative VERIFY pass (T = draft + 1, i.e. 2..4).
//
// A per-token GEMV would re-read every weight for every token, and weights are exactly what the
// decode path is bound by, so a verify round would cost T full decode passes and speculation could
// never pay for itself (measured: 15.7 t/s with MTP N=2 against 62.1 t/s without). This kernel
// keeps the weights single-read: the code byte and scale are loaded once per group and reused for
// all kT tokens, while each token contributes its own four activations.
template <int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row, std::int32_t tokens,
                                  std::int32_t out_row_stride) {
    static_assert(kT >= 1 && kT <= 8, "tile size must be small enough to keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        // Weights: one byte of codes + one 16-bit scale, reused across every token in the tile.
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const float weight0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
        const float weight1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
        const float weight2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
        const float weight3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);

        const std::int32_t base = group * kGemvGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK;
                const float2 low = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 high = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
                const float dot = fmaf(weight0, low.x,
                                       fmaf(weight1, low.y, fmaf(weight2, high.x, weight3 * high.y)));
                accumulator[t] = fmaf(scale, dot, accumulator[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0 && t < tokens) {
            // Token-major output: element (row, token) lives at token * out_row_stride + row.
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}

} // namespace ninfer::ops::detail
