#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen3_6_27b {

struct Package;

namespace detail {

struct Variant;

// 权重档案描述"制品里存在哪些张量、以及规划期要为它们预留多少临时字节"。行几何不足以区分
// groupwise-int 转换产物与折叠（旋转基）三元产物 —— 两者的 GDN/attention 形状完全相同 —— 所以
// 折叠三元必须有自己的一档，否则规划期的容量查询只能取两者的上界，而引擎要求查询值恰好等于
// 执行高水位。
enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
    GroupwiseIntW8Endpoints,
    FoldedTernary,
};

using Frontend       = qwen3_6::Frontend;
using PreparedPrompt = qwen3_6::PreparedPrompt;
using OutputSession  = qwen3_6::OutputSession;

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_27b::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_27b::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id           = "qwen3.6-27b";
    static constexpr std::string_view target_key         = "qwen3_6_27b";
    static constexpr std::string_view qwen3_8_model_id   = "qwen3.8-27b";
    static constexpr std::string_view qwen3_8_target_key = "qwen3_8_27b";
    // 折叠三元制品的 identity.weights_id。绑定期并不看它 —— 每个权重的格式由制品自己的声明
    // 决定 —— 但规划期的容量查询看不到权重，只能靠它区分两档。
    static constexpr std::string_view folded_weights_id = "folded-ternary";

    using WeightsProfile  = detail::WeightsProfile;
    using LoadPlan        = detail::LoadPlan;
    using LoadedModel     = detail::LoadedModel;
    using Frontend        = detail::Frontend;
    using PreparedPrompt  = detail::PreparedPrompt;
    using OutputSession   = detail::OutputSession;
    using SequencePlanner = qwen3_6::SequencePlanner<detail::Variant>;
    using SequencePlan    = qwen3_6::SequencePlan<detail::Variant>;
    using RequestBasePlan = qwen3_6::RequestBasePlan<detail::Variant>;
    using RequestPlan     = qwen3_6::RequestPlan<detail::Variant>;
    using Program         = qwen3_6::Program<detail::Variant>;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::qwen3_6_27b
} // namespace ninfer
