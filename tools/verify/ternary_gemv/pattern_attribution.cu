// Attribution of the decode-GEMV bandwidth: which part of the memory pattern costs what.
//
// Build (see run_checks.sh, mode "pattern"):
//   nvcc -O3 -std=c++17 -arch=sm_89 tools/verify/ternary_gemv/pattern_attribution.cu \
//        -o pattern_attribution
//
// Each row reads the same addresses in the same layout as the real kernel but with the cheapest
// possible consumer, so the step between rows is the cost of one added piece of work rather than of
// a different access pattern. Run it on a payload well past L2 (the default, 312 MB) or the first
// rows are inflated by cache.
//
// Measured on an RTX 4060 Laptop, 312 MB payload, best of several runs:
//   uint4 contiguous        ~249 GB/s   (same as tools/hbm_bandwidth_probe.cu)
//   code plane only         ~249 GB/s   the dominant stream is already at the ceiling
//   + activation read       ~249 GB/s   the int8 activation vector is free: it stays in cache
//   + high and scale planes ~196 GB/s   two 2-byte-per-group side streams cost 22%
//   + base-3 decode         ~169 GB/s   the trit extraction costs 14%
//   + dp4a and scale fma    ~120 GB/s   the accumulation step costs 29%
//
// The last row is the real inner loop. It is the reason further kernel work has to beat the
// three-plane ceiling of ~196 GB/s rather than the 249 GB/s of a contiguous stream, and the reason
// closing the remaining gap needs a different weight layout (one plane per row), not a better
// schedule: the cost is spread thinly across the side streams, the decode and the accumulation
// rather than sitting in one hot instruction.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

namespace {

constexpr int kWPB = 8;
constexpr int kThreads = kWPB * 32;
constexpr int kGroupK = 128;

void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

__global__ __launch_bounds__(kThreads)
void stream_uint4(const int4* __restrict__ src, int* sink, long chunks) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (long c = i; c < chunks; c += stride) {
        const int4 v = src[c];
        acc += (unsigned)(v.x ^ v.y ^ v.z ^ v.w);
    }
    if (acc == 0xDEADBEEFu) sink[0] = (int)acc;
}

__global__ __launch_bounds__(kThreads)
void stream_code(const unsigned char* __restrict__ codes, int* sink, int rows, int groups) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * kWPB + (threadIdx.x >> 5);
    if (warp >= rows) return;
    const int quad_off = lane < 20 ? 4 * (lane & 3) : 16 + 4 * ((lane - 20) & 1);
    const unsigned char* row = codes + (long)warp * groups * 24;
    unsigned acc = 0;
#pragma unroll 8
    for (int g = 0; g < groups; ++g) {
        acc += *reinterpret_cast<const int*>(row + g * 24 + quad_off);
    }
    if (acc == 0xDEADBEEFu) sink[0] = (int)acc;
}

__global__ __launch_bounds__(kThreads)
void stream_code_act(const unsigned char* __restrict__ codes, const signed char* __restrict__ xq,
                     int* sink, int rows, int groups) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * kWPB + (threadIdx.x >> 5);
    if (warp >= rows) return;
    const int quad_off = lane < 20 ? 4 * (lane & 3) : 16 + 4 * ((lane - 20) & 1);
    const unsigned char* row = codes + (long)warp * groups * 24;
    unsigned acc = 0;
#pragma unroll 8
    for (int g = 0; g < groups; ++g) {
        acc += *reinterpret_cast<const int*>(row + g * 24 + quad_off);
        acc += (unsigned)*reinterpret_cast<const int*>(xq + g * kGroupK + lane * 4);
    }
    if (acc == 0xDEADBEEFu) sink[0] = (int)acc;
}

__global__ __launch_bounds__(kThreads)
void stream_all_planes(const unsigned char* __restrict__ codes,
                       const unsigned char* __restrict__ high,
                       const unsigned char* __restrict__ scales, int* sink, int rows, int groups) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * kWPB + (threadIdx.x >> 5);
    if (warp >= rows) return;
    const int quad_off = lane < 20 ? 4 * (lane & 3) : 16 + 4 * ((lane - 20) & 1);
    const unsigned char* cr = codes + (long)warp * groups * 24;
    const unsigned char* hr = high + (long)warp * groups * 2;
    const unsigned char* sr = scales + (long)warp * groups * 2;
    unsigned acc = 0;
#pragma unroll 8
    for (int g = 0; g < groups; ++g) {
        acc += *reinterpret_cast<const int*>(cr + g * 24 + quad_off);
        acc += *reinterpret_cast<const short*>(hr + g * 2);
        acc += *reinterpret_cast<const short*>(sr + g * 2);
    }
    if (acc == 0xDEADBEEFu) sink[0] = (int)acc;
}

__global__ __launch_bounds__(kThreads)
void stream_decode(const unsigned char* __restrict__ codes, const signed char* __restrict__ xq,
                   int* sink, int rows, int groups) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * kWPB + (threadIdx.x >> 5);
    if (warp >= rows) return;
    const int quad_off = lane < 20 ? 4 * (lane & 3) : 16 + 4 * ((lane - 20) & 1);
    const int trit = lane < 20 ? lane >> 2 : (lane - 20) >> 1;
    const unsigned pow3 = trit <= 0 ? 1u : trit == 1 ? 3u : trit == 2 ? 9u : trit == 3 ? 27u : 81u;
    const unsigned char* cr = codes + (long)warp * groups * 24;
    unsigned acc = 0;
#pragma unroll 8
    for (int g = 0; g < groups; ++g) {
        const unsigned cw = (unsigned)*reinterpret_cast<const int*>(cr + g * 24 + quad_off);
        unsigned v_lo = __byte_perm(cw, 0u, 0x4140u) * pow3; v_lo &= 0x00FF00FFu; v_lo *= 3u;
        unsigned v_hi = __byte_perm(cw, 0u, 0x4342u) * pow3; v_hi &= 0x00FF00FFu; v_hi *= 3u;
        const unsigned q = __vsub4(__byte_perm(v_lo, v_hi, 0x7531u), 0x01010101u) &
                           ((lane >= 30) ? 0u : 0xFFFFFFFFu);
        acc += q;
        acc += (unsigned)*reinterpret_cast<const int*>(xq + g * kGroupK + lane * 4);
    }
    if (acc == 0xDEADBEEFu) sink[0] = (int)acc;
}

__global__ __launch_bounds__(kThreads)
void stream_decode_dp4a(const unsigned char* __restrict__ codes, const signed char* __restrict__ xq,
                        const unsigned char* __restrict__ scales, const float* __restrict__ xq_scale,
                        __nv_bfloat16* out, int rows, int groups) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * kWPB + (threadIdx.x >> 5);
    if (warp >= rows) return;
    const int quad_off = lane < 20 ? 4 * (lane & 3) : 16 + 4 * ((lane - 20) & 1);
    const int trit = lane < 20 ? lane >> 2 : (lane - 20) >> 1;
    const unsigned pow3 = trit <= 0 ? 1u : trit == 1 ? 3u : trit == 2 ? 9u : trit == 3 ? 27u : 81u;
    const unsigned char* cr = codes + (long)warp * groups * 24;
    const unsigned char* sr = scales + (long)warp * groups * 2;
    float acc = 0.0f;
#pragma unroll 8
    for (int g = 0; g < groups; ++g) {
        const int a = *reinterpret_cast<const int*>(xq + g * kGroupK + lane * 4);
        const unsigned cw = (unsigned)*reinterpret_cast<const int*>(cr + g * 24 + quad_off);
        unsigned v_lo = __byte_perm(cw, 0u, 0x4140u) * pow3; v_lo &= 0x00FF00FFu; v_lo *= 3u;
        unsigned v_hi = __byte_perm(cw, 0u, 0x4342u) * pow3; v_hi &= 0x00FF00FFu; v_hi *= 3u;
        const unsigned q = __vsub4(__byte_perm(v_lo, v_hi, 0x7531u), 0x01010101u) &
                           ((lane >= 30) ? 0u : 0xFFFFFFFFu);
        const int dot = __dp4a((int)q, a, 0);
        const unsigned short sb = *reinterpret_cast<const unsigned short*>(sr + g * 2);
        acc = fmaf(__half2float(__ushort_as_half(sb)) * xq_scale[g], (float)dot, acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) out[warp] = __float2bfloat16_rn(acc);
}

template <typename F>
float time_ms(F launch, int iters) {
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(b);
    check(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

} // namespace

int main(int argc, char** argv) {
    // Default payload is deliberately ~10x L2; a 39 MB tensor partially hits cache and hides costs.
    const int rows = argc > 1 ? atoi(argv[1]) : 278528;
    const int k = argc > 2 ? atoi(argv[2]) : 5120;
    const int groups = k / 128;
    const int iters = rows > 100000 ? 15 : 60;
    const double weight_bytes = (double)rows * k * 28.0 / 128.0;

    unsigned char *dc, *dh, *ds;
    signed char* dq;
    __nv_bfloat16* dout;
    float* dqs;
    int* sink;
    check(cudaMalloc(&dc, (size_t)rows * groups * 24));
    check(cudaMalloc(&dh, (size_t)rows * groups * 2));
    check(cudaMalloc(&ds, (size_t)rows * groups * 2));
    check(cudaMalloc(&dq, (size_t)k));
    check(cudaMalloc(&dout, (size_t)rows * 2));
    check(cudaMalloc(&dqs, (size_t)groups * 4));
    check(cudaMalloc(&sink, 4));
    check(cudaMemset(dc, 0x55, (size_t)rows * groups * 24));
    check(cudaMemset(dh, 0, (size_t)rows * groups * 2));
    check(cudaMemset(ds, 0, (size_t)rows * groups * 2));
    check(cudaMemset(dq, 1, (size_t)k));
    check(cudaMemset(dqs, 0, (size_t)groups * 4));

    const int grid = (rows + kWPB - 1) / kWPB;
    std::printf("rows=%d k=%d  weight bytes=%.1f MB  (hbm probe ceiling ~249.6 GB/s)\n",
                rows, k, weight_bytes / 1e6);
    auto row = [&](const char* label, float ms) {
        std::printf("  %-22s %8.2f us  %6.1f GB/s\n", label, ms * 1e3,
                    weight_bytes / (ms * 1e-3) / 1e9);
    };

    {
        int4* big = nullptr;
        const long chunks = (long)(weight_bytes / 16.0);
        check(cudaMalloc(&big, chunks * 16));
        check(cudaMemset(big, 1, chunks * 16));
        const int g2 = 24 * 6;
        row("uint4 contiguous", time_ms([&] { stream_uint4<<<g2, kThreads>>>(big, sink, chunks); }, iters));
        cudaFree(big);
    }
    row("code plane only",
        time_ms([&] { stream_code<<<grid, kThreads>>>(dc, sink, rows, groups); }, iters));
    row("+ activation read",
        time_ms([&] { stream_code_act<<<grid, kThreads>>>(dc, dq, sink, rows, groups); }, iters));
    row("+ high/scale planes",
        time_ms([&] { stream_all_planes<<<grid, kThreads>>>(dc, dh, ds, sink, rows, groups); }, iters));
    row("+ base-3 decode",
        time_ms([&] { stream_decode<<<grid, kThreads>>>(dc, dq, sink, rows, groups); }, iters));
    row("+ dp4a/scale (real loop)",
        time_ms([&] {
            stream_decode_dp4a<<<grid, kThreads>>>(dc, dq, ds, dqs, dout, rows, groups);
        }, iters));
    check(cudaGetLastError());
    return 0;
}
