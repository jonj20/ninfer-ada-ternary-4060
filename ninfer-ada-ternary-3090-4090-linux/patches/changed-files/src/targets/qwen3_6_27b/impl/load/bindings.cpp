#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

bool is_early_attention_input(std::size_t layer) {
    return layer == 3 || layer == 7 || layer == 11 || layer == 15 || layer == 19 || layer == 23;
}

bool is_bf16_attention_output(std::size_t layer) { return layer == 3 || layer == 7; }

bool is_bf16_gdn_output(std::size_t layer) { return layer == 4; }
NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::Q6G64_F16S;
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return NumericFormat::W8G32_F16S;
    case WeightsProfile::FoldedTernary:
        // 词表两端在折叠三元制品里是三元张量，但这里返回的值只用于选中"读制品声明"那条绑定
        // 路径（见 is_row_split_grouped），实际格式一律取自制品自己。PTQ1_0_G128 与
        // PQ2_0_G128 走的是同一条路径，所以同一个档案能同时服务两种三元打包。
        return NumericFormat::PTQ1_0_G128;
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

// 分组 row-split 家族在容器层面是可以互换的：每个成员共享 row-split-k128-v1 布局，只在组几何
// 与码打包上不同，而制品会声明它存的是哪一个。所以这里解析【制品声明的】格式，而不是听信计划
// 的预期 —— 这正是同一个运行时既能读 groupwise-int 转换产物（Q4G64/Q5G64/Q6G64/W8G32）、
// 又能读三元移植产物（PTQ1_0_G128/PQ2_0_G128）的原因。
//
// 这个放宽是刻意收窄的：声明出来的格式本身必须也是分组 row-split 格式，而且 bind_tensor()
// 依旧校验布局与形状，所以任一方向的错配仍会被拒。下游一律读 WeightPlan::format，因此在这里
// 解析好就等于传播到了所有 materialized_weight() 调用点，无需逐个改。
bool is_row_split_grouped(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q4G64_F16S:
    case NumericFormat::Q5G64_F16S:
    case NumericFormat::Q6G64_F16S:
    case NumericFormat::W8G32_F16S:
    case NumericFormat::PTQ1_0_G128:
    case NumericFormat::PQ2_0_G128:
        return true;
    default:
        return false;
    }
}

// 折叠基特征置换 P。只有 GDN 输出投影是"分组列"存储、而运行时产出分块 v-head 几何，所以这里
// 同时按对象的【角色】和它的输入宽度设门，而不是只看宽度。头数取自模型：n_v = 48 个 v-head，
// n_k = 16 个 k 组（于是 K = 6144 = 128 * 16 * 3）。
void set_folded_perm(WeightPlan& plan, std::string_view name, std::uint64_t columns) {
    constexpr std::uint64_t kGdnOutputWidth = 6144;
    constexpr std::int32_t kNv             = 48;
    constexpr std::int32_t kNk             = 16;
    if (name.ends_with("/gdn/output") && columns == kGdnOutputWidth) {
        plan.hadamard_perm_hd  = static_cast<std::int32_t>(columns) / kNv;
        plan.hadamard_perm_nk  = kNk;
        plan.hadamard_perm_rep = kNv / kNk;
    }
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    const std::uint64_t columns = shape.size() > 1 ? shape.begin()[1] : 0;
    if (is_row_split_grouped(format)) {
        const artifact::ObjectDescriptor* descriptor = binder.find(name);
        if (descriptor == nullptr ||
            !std::holds_alternative<artifact::TensorDescriptor>(*descriptor)) {
            throw artifact::ArtifactError(std::string(name) + ": expected a tensor object");
        }
        const NumericFormat declared = std::get<artifact::TensorDescriptor>(*descriptor).format;
        if (!is_row_split_grouped(declared)) {
            throw artifact::ArtifactError(std::string(name) +
                                          ": artifact declares a non-grouped format for a "
                                          "row-split weight");
        }
        WeightPlan plan{.object = artifact::bind_device_tensor(binder, name, declared, shape),
                        .format = declared};
        set_folded_perm(plan, name, columns);
        return plan;
    }
    WeightPlan plan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                    .format = format};
    set_folded_perm(plan, name, columns);
    return plan;
}

// --- 折叠（旋转基）符号表 -----------------------------------------------------
//
// 见 bindings.h 对该表的说明。下面这些常量描述三元移植写出的制品契约：1024 宽归一化
// Sylvester-Hadamard 分块、28672 个显式符号，以及恰好把符号切分完的若干宽度。
inline constexpr std::int32_t kHadamardBlockSize = 1024;
inline constexpr std::size_t kHadamardSignValues = 28672;
inline constexpr std::size_t kHadamardWidthCount = 3;

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::size_t offset,
                          const char* label) {
    if (offset + sizeof(std::uint32_t) > bytes.size()) {
        throw artifact::ArtifactError(std::string(label) + " is truncated");
    }
    return std::to_integer<std::uint32_t>(bytes[offset]) |
           (std::to_integer<std::uint32_t>(bytes[offset + 1]) << 8U) |
           (std::to_integer<std::uint32_t>(bytes[offset + 2]) << 16U) |
           (std::to_integer<std::uint32_t>(bytes[offset + 3]) << 24U);
}

// 加载期作用域：只在一次物化过程中安装，随后拆除，因此没有任何全局状态活过一次加载。把符号块
// 挂在这里 —— 构造每个 Weight 的唯一漏斗 —— 正是让三十个 materialized_weight() 调用点无需改动
// 的原因。
struct FoldedSigns {
    const float* base = nullptr;
    std::vector<std::pair<std::int32_t, std::uint64_t>> width_offsets;

    [[nodiscard]] const float* for_width(std::int32_t width) const noexcept {
        for (const auto& entry : width_offsets) {
            if (entry.first == width) { return base + entry.second; }
        }
        return nullptr;
    }
};

FoldedSigns* g_folded_signs = nullptr;

class FoldedSignsScope {
public:
    FoldedSignsScope(const std::optional<HadamardSignsPlan>& plan,
                     const artifact::MaterializedArtifact& materialized) {
        if (!plan.has_value()) { return; }
        table_.base          = static_cast<const float*>(materialized.device_data(plan->values));
        table_.width_offsets = plan->width_offsets;
        if (table_.base != nullptr) { g_folded_signs = &table_; }
    }
    ~FoldedSignsScope() { g_folded_signs = nullptr; }

    FoldedSignsScope(const FoldedSignsScope&)            = delete;
    FoldedSignsScope& operator=(const FoldedSignsScope&) = delete;

private:
    FoldedSigns table_;
};

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    // 三元权重恰好就是折叠权重（制品里 402 个三元张量 = 401 个折叠权重 + 一个做逆映射的词嵌入
    // 表），所以只看格式就能判定是否需要符号块。
    const auto attach_folded_signs = [&](Weight& w) {
        if (g_folded_signs == nullptr) { return; }
        if (plan.format != NumericFormat::PTQ1_0_G128 &&
            plan.format != NumericFormat::PQ2_0_G128) {
            return;
        }
        const float* signs = g_folded_signs->for_width(columns);
        if (signs == nullptr) {
            throw artifact::ArtifactError(
                "folded ternary weight has no sign block for input width " +
                std::to_string(columns));
        }
        w.hadamard_signs    = signs;
        w.hadamard_n_blk    = columns / kHadamardBlockSize;
        w.hadamard_perm_hd  = plan.hadamard_perm_hd;
        w.hadamard_perm_nk  = plan.hadamard_perm_nk;
        w.hadamard_perm_rep = plan.hadamard_perm_rep;
    };

    Weight out = artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
    attach_folded_signs(out);
    return out;
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    // row-split 平面几何是按格式而定的：Q4/Q5/Q6/W8 每组 32 个基础字节，而 PTQ1_0 是 24 基础
    // + 2 高位，PQ2_0 是 32 基础且无高位平面。对 PTQ1_0 套用默认的 32/0 会从载荷里切出错位的
    // 字节区间。
    const std::uint64_t low_group  = block.qtype == QType::PTQ1_0_G128 ? 24 : 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S    ? 8
                                     : block.qtype == QType::Q6G64_F16S  ? 16
                                     : block.qtype == QType::PTQ1_0_G128 ? 2
                                                                         : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, 34816, 5120);
    out.down    = materialized_weight(materialized, plan.down, 5120, 17408);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection)) {
        return SplitAttentionProjectionPayload{
            .query_key  = materialized_weight(materialized, split->query_key, 7168, 5120),
            .gate_value = materialized_weight(materialized, split->gate_value, 7168, 5120),
        };
    }
    const auto& fused = std::get<FusedAttentionProjectionPlan>(plan.projection);
    return FusedAttentionProjectionPayload{
        .query_key_gate_value =
            materialized_weight(materialized, fused.query_key_gate_value, 14336, 5120),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection)) {
        return SplitGdnInputProjectionPayload{
            .query_key = materialized_weight(materialized, split->query_key, 4096, 5120),
            .value_z   = materialized_weight(materialized, split->value_z, 12288, 5120),
        };
    }
    const auto& fused = std::get<FusedGdnInputProjectionPlan>(plan.input_projection);
    return FusedGdnInputProjectionPayload{
        .query_key_value_z =
            materialized_weight(materialized, fused.query_key_value_z, 16384, 5120),
    };
}

void bind_groupwise_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = SplitAttentionProjectionPlan{
                .query_key  = bind_weight(binder, prefix + "attention/query_key",
                                          NumericFormat::Q4G64_F16S, {7168, 5120}),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value",
                                          NumericFormat::Q5G64_F16S, {7168, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                  NumericFormat::Q5G64_F16S, {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.a_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.b_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/b_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.input_projection = SplitGdnInputProjectionPlan{
                .query_key = bind_weight(binder, prefix + "gdn/query_key",
                                         NumericFormat::Q4G64_F16S, {4096, 5120}),
                .value_z   = bind_weight(binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S,
                                         {12288, 5120}),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S, {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S, {34816, 5120});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {5120, 17408});
    }
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

} // namespace

// Stub binding for optional DFlash2 weights to ensure compatibility with
// new Qwen3.8 artifact releases without materializing them on device.
void bind_dflash2_stub(artifact::Binder& binder) {
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape,
                                     artifact::TensorPlacement::ValidateOnly);
    };

    (void)bind("dflash2/feature_projection", NumericFormat::W8G32_F16S, {5120, 25600});
    (void)bind("dflash2/context_norm", NumericFormat::BF16, {5120});
    for (std::size_t layer = 0; layer < 5; ++layer) {
        const std::string prefix = "dflash2/layers/" + std::to_string(layer) + "/";
        (void)bind(prefix + "input_norm", NumericFormat::BF16, {5120});
        (void)bind(prefix + "attention_conv/base_kernel", NumericFormat::BF16, {2, 2, 5120});
        (void)bind(prefix + "attention_conv/kernel_projection", NumericFormat::BF16, {1280, 5120});
        (void)bind(prefix + "attention/query_key_value", NumericFormat::W8G32_F16S, {6144, 5120});
        (void)bind(prefix + "attention/query_norm", NumericFormat::BF16, {128});
        (void)bind(prefix + "attention/key_norm", NumericFormat::BF16, {128});
        (void)bind(prefix + "attention/output", NumericFormat::W8G32_F16S, {5120, 4096});
        (void)bind(prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        (void)bind(prefix + "mlp_conv/base_kernel", NumericFormat::BF16, {2, 2, 5120});
        (void)bind(prefix + "mlp_conv/kernel_projection", NumericFormat::BF16, {1280, 5120});
        (void)bind(prefix + "mlp/gate_up", NumericFormat::W8G32_F16S, {34816, 5120});
        (void)bind(prefix + "mlp/down", NumericFormat::W8G32_F16S, {5120, 17408});
    }
    (void)bind("dflash2/final_norm", NumericFormat::BF16, {5120});
    (void)bind("dflash2/candidate_selector/hidden_projection", NumericFormat::BF16, {256, 5120});
    (void)bind("dflash2/candidate_selector/predecessor_codebook", NumericFormat::BF16, {248320, 256});
    (void)bind("dflash2/candidate_selector/successor_codebook", NumericFormat::BF16, {248320, 256});
}

// 定义在下方，紧挨着消费它的那次物化过程。
std::optional<HadamardSignsPlan> bind_hadamard_signs(artifact::Binder& binder);

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    out.token_embedding =
        bind_weight(binder, "text/token_embedding", vocabulary_format, {248320, 5120});
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
    case WeightsProfile::FoldedTernary:
        // 三档共用同一张文本层对象表：绑定层按对象名与制品声明的格式物化，从不看权重档案，
        // 所以折叠三元制品在这里与 groupwise-int 制品走完全相同的代码。档案的差别只在规划期
        // 的临时容量查询上。
        bind_groupwise_text_layers(binder, out);
        break;
    default:
        throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
    }
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {5120});
    out.output_head = bind_weight(binder, "text/output_head", vocabulary_format, {248320, 5120});
    out.hadamard_signs = bind_hadamard_signs(binder);
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_tensor(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                           {131072, 5120}, proposal_placement);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {131072}, proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids);

    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp                           = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection =
        bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {5120, 10240});
    out.mtp.embedding_norm       = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {5120});
    out.mtp.hidden_norm          = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {5120});
    out.mtp.input_norm           = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {5120});
    out.mtp.query_key_gate_value = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                            NumericFormat::W8G32_F16S, {14336, 5120});
    out.mtp.query_norm = bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.key_norm   = bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.output =
        bind_mtp("mtp/layer/attention/output", NumericFormat::W8G32_F16S, {5120, 6144});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {5120});
    out.mtp.mlp.gate_up = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S, {34816, 5120}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.mlp.down = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/down", NumericFormat::W8G32_F16S, {5120, 17408}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {5120});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2   = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {5120, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {5120}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);

    const bool artifact_has_dflash2 = binder.has_object("dflash2/feature_projection");
    if (artifact_has_dflash2) {
        bind_dflash2_stub(binder);
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

// 当制品携带折叠符号表时把它绑上。权重未折叠的制品直接省略这些对象，而每个对象都必须被绑定，
// 否则 Binder::finish() 会以"有对象未被消费"拒绝该制品 —— 所以只要对象存在，这就不是可选项。
std::optional<HadamardSignsPlan> bind_hadamard_signs(artifact::Binder& binder) {
    if (!binder.contains("text/hadamard_signs")) { return std::nullopt; }

    HadamardSignsPlan plan;
    plan.values = artifact::bind_device_tensor(binder, "text/hadamard_signs", NumericFormat::FP32,
                                               {kHadamardSignValues});
    plan.widths = artifact::bind_tensor(binder, "text/hadamard_widths", NumericFormat::I32,
                                        {kHadamardWidthCount},
                                        artifact::TensorPlacement::ValidateOnly);

    // 宽度只需要到达主机：它们在这里被读出来推导每个宽度的元素偏移，正是这一点让权重仅凭输入
    // 维度就能找到自己的分块。
    const artifact::PayloadSpan payload = binder.payload(plan.widths);
    if (payload.data.size() < kHadamardWidthCount * sizeof(std::uint32_t)) {
        throw artifact::ArtifactError("text/hadamard_widths is shorter than its declared count");
    }
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < kHadamardWidthCount; ++i) {
        const std::uint32_t width =
            read_u32_le(payload.data, i * sizeof(std::uint32_t), "text/hadamard_widths");
        if (width == 0) {
            throw artifact::ArtifactError("text/hadamard_widths entries must be positive");
        }
        plan.width_offsets.emplace_back(static_cast<std::int32_t>(width), offset);
        offset += width;
    }
    if (offset != kHadamardSignValues) {
        throw artifact::ArtifactError("text/hadamard_widths must sum to " +
                                      std::to_string(kHadamardSignValues) + ", got " +
                                      std::to_string(offset));
    }
    return plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);
    // 为下面整段物化过程安装折叠符号表。
    const FoldedSignsScope folded_signs_scope(plan.hadamard_signs, backing);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding        = materialized_weight(backing, plan.token_embedding, 248320, 5120);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {5120});
            target.projection            = load_attention_projection(source.attention, backing);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = materialized_weight(backing, source.attention.output, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {5120});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {48});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {48});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {10240, 4});
            target.projection.a_projection = artifact::materialized_weight(
                backing, source.gdn.a_projection, NumericFormat::BF16, 48, 5120);
            target.projection.b_projection = artifact::materialized_weight(
                backing, source.gdn.b_projection, NumericFormat::BF16, 48, 5120);
            target.projection.input_projection = load_gdn_input_projection(source.gdn, backing);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.output = materialized_weight(backing, source.gdn.output, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {5120});
    output_head = materialized_weight(backing, plan.output_head, 248320, 5120);
    if (plan.features.optimized_proposal()) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_weight(backing, plan.draft_head,
                                                           NumericFormat::Q4G64_F16S, 131072, 5120);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {131072});
    }

    if (plan.features.mtp()) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(
            backing, plan.mtp.input_projection, NumericFormat::W8G32_F16S, 5120, 10240);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.attention.packed = artifact::materialized_weight(
            backing, plan.mtp.query_key_gate_value, NumericFormat::W8G32_F16S, 14336, 5120);
        mtp.attention.query       = row_view(mtp.attention.packed, 0, 6144);
        mtp.attention.key         = row_view(mtp.attention.packed, 6144, 1024);
        mtp.attention.output_gate = row_view(mtp.attention.packed, 7168, 6144);
        mtp.attention.value       = row_view(mtp.attention.packed, 13312, 1024);
        mtp.query_norm =
            artifact::materialized_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {256});
        mtp.key_norm =
            artifact::materialized_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {256});
        mtp.output              = artifact::materialized_weight(backing, plan.mtp.output,
                                                                NumericFormat::W8G32_F16S, 5120, 6144);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {5120});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {5120});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, 5120, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {5120});
    }
}

} // namespace ninfer::targets::qwen3_6_27b::detail
