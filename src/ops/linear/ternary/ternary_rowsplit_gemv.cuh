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

// The four qh trits packed in one high byte, as weights in {-1, 0, 1}. Trit t sits at bit offset
// 2t, so repeated multiplication by three walks them instead of four independent pow3 products.
__device__ __forceinline__ void tail_trits4(std::uint32_t byte, float (&w)[4]) {
    std::uint32_t q = byte & 0xFFu;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        w[t] = static_cast<float>((q * 3u) >> 8) - 1.0f;
        q = (q * 3u) & 0xFFu;
    }
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

    if constexpr (kT == 1) {
        // Single-token decode path. Fully branch-free on purpose: an in-loop `if (!is_tail)`
        // forces BSSY/BSYNC reconvergence barriers around every unrolled group, which stops the
        // compiler hoisting the weight/activation loads across iterations (measured 47 GB/s
        // branchy vs 110 GB/s branch-free on sm_89).
        float acc_front = 0.0f;
        // Lane mask is a loop invariant: hoisting the select out of the group loop keeps the body
        // free of branches (an in-loop select re-materialises as BSSY/BSYNC and costs ~2x).
        const float weight_mask = is_tail ? 0.0f : 1.0f;
#pragma unroll 8
        for (int group = 0; group < groups_per_row; ++group) {
            const std::int32_t base = group * kGemvGroupK + lane * 4;
            const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);

            // One 4-byte load per group per lane: bytes 4*(lane&3)..+3 (front) or 16+.. (mid).
            // Lanes 30/31 re-read the mid-quad bytes here; their weights are masked out below.
            v_lo = __byte_perm(*reinterpret_cast<const std::uint32_t*>(code_row + group * 24 + quad_off), 0u, 0x4140u);
            v_hi = __byte_perm(*reinterpret_cast<const std::uint32_t*>(code_row + group * 24 + quad_off), 0u, 0x4342u);
            v_lo = (v_lo * pow3) & 0x00FF00FFu;
            v_hi = (v_hi * pow3) & 0x00FF00FFu;
            v_lo *= 3u;
            v_hi *= 3u;

            // Four bf16 activations (columns 4l..4l+3) are 8-byte aligned: one 64-bit load,
            // de-pack the two float2 pairs in registers.
            const std::uint64_t ract =
                *reinterpret_cast<const std::uint64_t*>(x + base);
            const __nv_bfloat162* racq = reinterpret_cast<const __nv_bfloat162*>(&ract);
            const float2 lo = __bfloat1622float2(racq[0]);
            const float2 hi = __bfloat1622float2(racq[1]);

            // Four signed bytes of the SIMD step: each is one weight in {-1, 0, 1}.
            const std::uint32_t q =
                __vsub4(__byte_perm(v_lo, v_hi, 0x7531u), 0x01010101u);
            const float w0 = static_cast<float>(static_cast<std::int8_t>(q & 0xFFu));
            const float w1 = static_cast<float>(static_cast<std::int8_t>((q >> 8) & 0xFFu));
            const float w2 = static_cast<float>(static_cast<std::int8_t>((q >> 16) & 0xFFu));
            const float w3 = static_cast<float>(static_cast<std::int8_t>((q >> 24) & 0xFFu));

            // Lanes 30/31 own no front/mid columns, so weight_mask zeroes the bytes they re-read.
            const float dot = fmaf(w0, lo.x, fmaf(w1, lo.y, fmaf(w2, hi.x, w3 * hi.y))) *
                              weight_mask;
            acc_front = fmaf(scale, dot, acc_front);
        }

        // The 8 qh tail columns (120..127) live in the 2-byte high plane, not the code plane, so
        // they need a different decode. Folding that into the group loop costs ~45% of throughput
        // (measured 110 -> 57 GB/s on the gate_up shape: the kernel is instruction-issue bound, so
        // every extra instruction in that loop costs directly), so they get their own pass.
        //
        // Lane l owns whole groups l, l+32, ... and takes all 8 tail columns of a group at once:
        // one 16-byte load covers the 8 activations, one 32-bit load covers the 2 high bytes and
        // one covers the 2 scale bytes. That is 3 loads per group for the whole warp instead of the
        // 3 scalar loads per group per lane a column-per-lane split needs (30 loads per lane for
        // gpr=40). The warp reduction then sums every lane's partial, so each (group, column) pair
        // is still visited exactly once.
        float acc_tail = 0.0f;
        for (int group = lane; group < groups_per_row; group += 32) {
            // 16 bytes = 8 bf16, i.e. exactly the 8 tail columns; element 120 sits at byte 240
            // within the group, so the 128-bit load is naturally aligned.
            const uint4 av = *reinterpret_cast<const uint4*>(
                x + group * kGemvGroupK + (kGemvGroupK - 8));
            // 2-byte load: the high plane has a 2-byte group stride, so only 16-bit aligned.
            const std::uint16_t hv =
                *reinterpret_cast<const std::uint16_t*>(high_row + group * 2);
            const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);

            const __nv_bfloat162* ab = reinterpret_cast<const __nv_bfloat162*>(&av);
            const float2 a0 = __bfloat1622float2(ab[0]);
            const float2 a1 = __bfloat1622float2(ab[1]);
            const float2 a2 = __bfloat1622float2(ab[2]);
            const float2 a3 = __bfloat1622float2(ab[3]);
            // trit t of a high byte is successive multiplication by 3: one chain gives all four.
            float wb[4];
            float wa[4];
            tail_trits4(hv & 0xFFu, wb);
            tail_trits4((hv >> 8) & 0xFFu, wa);
            const float dot =
                fmaf(wb[0], a0.x, fmaf(wa[0], a0.y, fmaf(wb[1], a1.x, fmaf(wa[1], a1.y,
                fmaf(wb[2], a2.x, fmaf(wa[2], a2.y, fmaf(wb[3], a3.x, wa[3] * a3.y)))))));
            acc_tail = fmaf(scale, dot, acc_tail);
        }

        float value = acc_front + acc_tail;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            out[warp] = __float2bfloat16_rn(value);
        }
        return;
    }

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

// ---------------------------------------------------------------------------
// int8-activation decode GEMV (T == 1): the same PTQ1_0 math with an integer dot product.
//
// Why: the bf16 path above spends one FFMA plus one int-to-float conversion per weight, ~30
// instructions per 128-group, and on sm_89 that instruction pressure -- not DRAM -- is what caps
// decode (measured 2.2 of 4 IPC at 98 GB/s on the 39 MB gate_up tensor). Quantizing the activation
// vector to int8 once per linear op lets one dp4a carry four weight-activation products, which
// takes the inner loop to ~15 instructions per group (measured 151.7 GB/s, i.e. DRAM-bound).
// This mirrors what llama.cpp's mmvq does with q8_1 activations.
//
// Quantization is per 128-column group, matching the weight group so one scale multiply covers both
// sides. Warp w owns groups w, w+8, ... and lane holds four of the 128 values, so the activation is
// read once and the reduction stays inside the warp (no block barrier).
__global__ __launch_bounds__(256)
void ternary_ptq1_quantize_act_kernel(const __nv_bfloat16* __restrict__ x,
                                       std::int8_t* __restrict__ xq, float* __restrict__ xq_scale,
                                       std::int32_t groups) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (int g = warp; g < groups; g += 8) {
        const std::int32_t base = g * kGemvGroupK + lane * 4;
        const float2 lo = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 hi = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));
        float m = fmaxf(fmaxf(fabsf(lo.x), fabsf(lo.y)), fmaxf(fabsf(hi.x), fabsf(hi.y)));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
        }
        const float inv = (m > 0.0f) ? (127.0f / m) : 0.0f;
        xq_scale[g] = m * (1.0f / 127.0f);
        const int q0 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, lo.x * inv)));
        const int q1 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, lo.y * inv)));
        const int q2 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, hi.x * inv)));
        const int q3 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, hi.y * inv)));
        *reinterpret_cast<int*>(xq + base) = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                                            ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }
}

// Integer dot-product decode GEMV. Same row/lane mapping and same tail pass as the bf16 kernel
// above, so the two produce the same result up to activation quantization.
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_ptq1_gemv_dp4a_kernel(const std::int8_t* __restrict__ xq,
                                   const float* __restrict__ xq_scale,
                                   const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ high,
                                   const std::uint8_t* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                   std::int32_t groups_per_row) {
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

    const int  quad_off = lane < 20 ? 4 * (lane & 3) : 16 + 4 * ((lane - 20) & 1);
    const int  trit     = lane < 20 ? lane >> 2 : (lane - 20) >> 1;
    const unsigned pow3 = trit <= 0 ? 1u : trit == 1 ? 3u : trit == 2 ? 9u : trit == 3 ? 27u : 81u;
    // Lanes 30/31 own no front/mid column; zeroing their quad keeps the mask out of the loop.
    const std::uint32_t quad_mask = (lane >= 30) ? 0u : 0xFFFFFFFFu;

    float acc_front = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const int a_packed =
            *reinterpret_cast<const int*>(xq + group * kGemvGroupK + lane * 4);
        const float a_scale = xq_scale[group];

        const std::uint32_t code_word =
            *reinterpret_cast<const std::uint32_t*>(code_row + group * 24 + quad_off);
        std::uint32_t v_lo = __byte_perm(code_word, 0u, 0x4140u);
        std::uint32_t v_hi = __byte_perm(code_word, 0u, 0x4342u);
        v_lo = (v_lo * pow3) & 0x00FF00FFu;
        v_hi = (v_hi * pow3) & 0x00FF00FFu;
        v_lo *= 3u;
        v_hi *= 3u;
        // Four signed bytes: four weights in {-1, 0, 1}, ready for one dp4a.
        const std::uint32_t q =
            (__vsub4(__byte_perm(v_lo, v_hi, 0x7531u), 0x01010101u)) & quad_mask;

        const int dot = __dp4a(static_cast<int>(q), a_packed, 0);
        acc_front = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup) * a_scale,
                         static_cast<float>(dot), acc_front);
    }

    // qh tail pass, same shape as the bf16 kernel: lane l owns whole groups l, l+32, ... and takes
    // all 8 tail columns of a group with one 16-byte load, so the pass costs 3 loads per group for
    // the warp instead of 3 per lane.
    float acc_tail = 0.0f;
    for (int group = lane; group < groups_per_row; group += 32) {
        const uint4 av = *reinterpret_cast<const uint4*>(
            xq + group * kGemvGroupK + (kGemvGroupK - 16));
        const std::uint16_t hv =
            *reinterpret_cast<const std::uint16_t*>(high_row + group * 2);
        const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup) *
                            xq_scale[group];
        const std::int8_t* ap = reinterpret_cast<const std::int8_t*>(&av) + 8;
        float wb[4];
        float wa[4];
        tail_trits4(hv & 0xFFu, wb);
        tail_trits4((hv >> 8) & 0xFFu, wa);
        const float dot =
            fmaf(wb[0], static_cast<float>(ap[0]), fmaf(wa[0], static_cast<float>(ap[1]),
            fmaf(wb[1], static_cast<float>(ap[2]), fmaf(wa[1], static_cast<float>(ap[3]),
            fmaf(wb[2], static_cast<float>(ap[4]), fmaf(wa[2], static_cast<float>(ap[5]),
            fmaf(wb[3], static_cast<float>(ap[6]), wa[3] * static_cast<float>(ap[7]))))))));
        acc_tail = fmaf(scale, dot, acc_tail);
    }

    float value = acc_front + acc_tail;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(value); }
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
