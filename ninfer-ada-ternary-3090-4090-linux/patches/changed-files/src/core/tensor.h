#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

enum class QType : std::uint16_t {
    Q4G64_F16S = 0,
    Q5G64_F16S = 1,
    Q6G64_F16S = 2,
    W8G32_F16S = 3,
    BF16_CTRL  = 4,
    FP32_CTRL  = 5,
    I32_CTRL   = 6,
    // Prism 私有三元格式（Bonsai 2 27B）：组宽 128，base-3 三值码 / 2-bit 码。
    // 取值 9/10 与 Ada（sm_89）线保持一致 —— 那条线上 7/8 已被 NVFP4 / FP8_E4M3FN_ROW_BF16S
    // 占用；本线暂未携带这两个格式，留空以免将来并入时与本处冲突。
    PTQ1_0_G128 = 9,
    PQ2_0_G128  = 10,
};

enum class QuantLayout : std::uint16_t {
    RowSplit   = 0,
    Contiguous = 1,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4G64_F16S;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata        = nullptr;
    const void* qhigh        = nullptr;
    const void* scales       = nullptr;
    std::int32_t n           = 0;
    std::int32_t k           = 0;
    std::int32_t group       = 0;
    QuantLayout layout       = QuantLayout::RowSplit;
    DType scale_dtype        = DType::FP32;
    std::int32_t scale_ne[4] = {1, 1, 1, 1};
    std::int64_t scale_nb[4] = {0, 0, 0, 0};

    // 三元权重以"折叠进旋转基"的形式存储：模型实际计算 y = W' * (H * (s * P * x))，
    // 因此每一路喂给折叠权重的激活，都必须在乘之前先经过 (符号, 归一化 Sylvester-Hadamard)
    // 映射；而词嵌入查表得到的行，必须在之后用同一变换的逆映射回原始基。
    //
    // hadamard_signs 指向"本权重输入宽度"对应的符号块（每个宽度占 k/1024 行、每行 1024 个
    // F32 的 ±1），行号由激活所在的 1024 分块下标选出；nullptr 表示不做变换。
    // hadamard_n_blk = k/1024。
    const float* hadamard_signs = nullptr;
    std::int32_t hadamard_n_blk = 0;

    // 折叠基的特征置换 P，作用在符号与旋转之前：
    //
    //     x.view(perm_hd, perm_nk, perm_rep) -> 交换后两轴 -> flatten
    //
    // 只有 GDN 输出投影是"分组列"存储、而运行时产出的是分块 v-head，故只有它需要
    // perm_rep > 1，其余权重一律留 1。几何来自模型头数：
    // perm_hd = K/n_v，perm_nk = n_k，perm_rep = n_v/n_k。
    std::int32_t hadamard_perm_hd  = 0;
    std::int32_t hadamard_perm_nk  = 0;
    std::int32_t hadamard_perm_rep = 1;
};

} // namespace ninfer
