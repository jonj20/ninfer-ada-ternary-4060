#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/common/rowsplit_a8_quantize.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
void q4_linear_swiglu_mma_folded_pipelined_r64_c128_launch(const Tensor& x, const Weight& w,
                                                           Tensor& out, cudaStream_t stream);
// Wide folded tiles for the leading column blocks plus a narrow route for a small remainder.
void q4_linear_swiglu_mma_folded_pipelined_r64_c128_tail_launch(const Tensor& x, const Weight& w,
                                                                Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);
// The folded pipelined tile over an A8 activation (quantized x of the same K and T).
void q4_linear_swiglu_a8_mma_folded_pipelined_r64_c128_launch(const A8G64Activation& x,
                                                              const Weight& w, Tensor& out,
                                                              cudaStream_t stream);
// The same GEMM whose SwiGLU output is quantized in the epilogue to the A8 activation of the
// down projection (out: K = intermediate, the same T), bitwise equal to a8_g64_quantize of the
// BF16 output of the launch above.
void q4_linear_swiglu_a8_quantized_mma_folded_pipelined_r64_c128_launch(const A8G64Activation& x,
                                                                        const Weight& w,
                                                                        A8G64Activation& out,
                                                                        cudaStream_t stream);

} // namespace ninfer::ops::detail
