#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/transfer_work.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer {

struct HostKVPlaneLayout {
    std::size_t offset             = 0;
    std::size_t page_payload_bytes = 0;
    std::size_t head_payload_bytes = 0;

    friend bool operator==(const HostKVPlaneLayout&, const HostKVPlaneLayout&) = default;
};

struct HostKVPageLayout {
    KVPageGeometry geometry;
    std::vector<HostKVPlaneLayout> planes;
    std::size_t page_stride = 0;

    friend bool operator==(const HostKVPageLayout&, const HostKVPageLayout&) = default;
};

[[nodiscard]] HostKVPageLayout plan_host_kv_page_layout(const KVPageGeometry& geometry);

// The arena pins its backing store in chunks instead of committing the whole capacity up front, so
// a large logical window does not have to be resident in host RAM to be addressable. The capacity is
// a ceiling; bytes are pinned only as allocations demand them. This returns the size of the chunk
// pinned eagerly at construction, which is also the growth granularity. Callers that report startup
// pinning cost use it so the reported amount is the amount actually committed.
[[nodiscard]] std::size_t host_kv_arena_first_chunk_bytes(
    std::size_t capacity_bytes, std::span<const HostKVPageLayout> supported_layouts) noexcept;

[[nodiscard]] TransferWork plan_host_kv_transfer_work(const HostKVPageLayout& layout,
                                                      std::uint32_t pages,
                                                      std::uint32_t contiguous_runs);
[[nodiscard]] TransferWork plan_device_kv_copy_work(const HostKVPageLayout& layout,
                                                    std::uint32_t pages);

class HostKVArena;

class HostKVAllocationHandle {
public:
    HostKVAllocationHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(HostKVAllocationHandle,
                                         HostKVAllocationHandle) noexcept = default;

private:
    friend class HostKVArena;
    friend class HostKVAllocation;
    friend class HostKVAllocationView;
    friend class HostKVAllocationConstView;

    HostKVAllocationHandle(const HostKVArena* owner, std::uint32_t descriptor,
                           std::uint32_t generation) noexcept
        : owner_(owner), descriptor_(descriptor), generation_(generation) {}

    const HostKVArena* owner_ = nullptr;
    std::uint32_t descriptor_ = 0;
    std::uint32_t generation_ = 0;
};

class HostKVAllocationView {
public:
    HostKVAllocationView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }

    [[nodiscard]] std::uint32_t page_count() const noexcept { return page_count_; }

    [[nodiscard]] const HostKVPageLayout& layout() const;
    [[nodiscard]] HostKVAllocationView subview(std::uint32_t begin, std::uint32_t count) const;

private:
    friend class HostKVArena;
    friend class HostKVAllocationConstView;

    HostKVAllocationView(HostKVAllocationHandle handle, std::byte* data,
                         const HostKVPageLayout* layout, std::uint32_t page_count) noexcept
        : handle_(handle), data_(data), layout_(layout), page_count_(page_count) {}

    HostKVAllocationHandle handle_;
    std::byte* data_                = nullptr;
    const HostKVPageLayout* layout_ = nullptr;
    std::uint32_t page_count_       = 0;
};

class HostKVAllocationConstView {
public:
    HostKVAllocationConstView() noexcept = default;

    HostKVAllocationConstView(HostKVAllocationView view) noexcept
        : handle_(view.handle_), data_(view.data_), layout_(view.layout_),
          page_count_(view.page_count_) {}

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] const std::byte* data() const noexcept { return data_; }

    [[nodiscard]] std::uint32_t page_count() const noexcept { return page_count_; }

    [[nodiscard]] const HostKVPageLayout& layout() const;
    [[nodiscard]] HostKVAllocationConstView subview(std::uint32_t begin, std::uint32_t count) const;

private:
    friend class HostKVArena;

    HostKVAllocationConstView(HostKVAllocationHandle handle, const std::byte* data,
                              const HostKVPageLayout* layout, std::uint32_t page_count) noexcept
        : handle_(handle), data_(data), layout_(layout), page_count_(page_count) {}

    HostKVAllocationHandle handle_;
    const std::byte* data_          = nullptr;
    const HostKVPageLayout* layout_ = nullptr;
    std::uint32_t page_count_       = 0;
};

class HostKVAllocation {
public:
    HostKVAllocation() noexcept = default;
    ~HostKVAllocation();

    HostKVAllocation(const HostKVAllocation&)            = delete;
    HostKVAllocation& operator=(const HostKVAllocation&) = delete;
    HostKVAllocation(HostKVAllocation&& other) noexcept;
    HostKVAllocation& operator=(HostKVAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] HostKVAllocationHandle handle() const noexcept;
    [[nodiscard]] std::uint32_t page_count() const noexcept;
    bool release() noexcept;

private:
    friend class HostKVArena;

    HostKVAllocation(HostKVArena& owner, std::uint32_t descriptor,
                     std::uint32_t generation) noexcept
        : owner_(&owner), descriptor_(descriptor), generation_(generation) {}

    void disarm() noexcept;

    HostKVArena* owner_       = nullptr;
    std::uint32_t descriptor_ = 0;
    std::uint32_t generation_ = 0;
};

struct HostKVAllocationRequest {
    const HostKVPageLayout* layout = nullptr;
    std::uint32_t pages            = 0;
};

struct HostKVSuballocationRelease {
    HostKVAllocationHandle allocation;
    std::uint32_t begin_page = 0;
    std::uint32_t page_count = 0;
};

class HostKVAllocationRecipe {
public:
    HostKVAllocationRecipe() noexcept                                    = default;
    HostKVAllocationRecipe(HostKVAllocationRecipe&&) noexcept            = default;
    HostKVAllocationRecipe& operator=(HostKVAllocationRecipe&&) noexcept = default;

    HostKVAllocationRecipe(const HostKVAllocationRecipe&)            = delete;
    HostKVAllocationRecipe& operator=(const HostKVAllocationRecipe&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::size_t release_count() const noexcept { return releases_.size(); }

    [[nodiscard]] std::size_t allocation_count() const noexcept { return targets_.size(); }

private:
    struct Target {
        std::uint32_t layout = 0;
        std::uint32_t pages  = 0;
        std::size_t offset   = 0;
        std::size_t bytes    = 0;
    };

    const HostKVArena* owner_     = nullptr;
    std::uint64_t arena_revision_ = 0;
    std::vector<HostKVAllocationHandle> releases_;
    std::vector<Target> targets_;

    friend class HostKVArena;
};

class HostKVArena {
public:
    HostKVArena(std::size_t capacity_bytes, std::span<const HostKVPageLayout> supported_layouts);

    HostKVArena(const HostKVArena&)            = delete;
    HostKVArena& operator=(const HostKVArena&) = delete;
    HostKVArena(HostKVArena&&)                 = delete;
    HostKVArena& operator=(HostKVArena&&)      = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    // Host RAM actually pinned right now. Below capacity_bytes() until allocations force growth.
    [[nodiscard]] std::size_t committed_bytes() const noexcept { return committed_bytes_; }

    [[nodiscard]] std::size_t free_bytes() const noexcept {
        return capacity_bytes_ - occupied_bytes_;
    }

    [[nodiscard]] const HostKVPageLayout* layout_for(const KVPageGeometry& geometry) const noexcept;

    // Widest allocation this arena can serve, in pages. An allocation never spans two growth chunks:
    // each chunk is its own pinned buffer and copy_to_host addresses the view linearly, so the
    // smallest chunk bounds it. Callers that batch work into one allocation (the pressure planner's
    // demotion runs) split their demand at this limit instead of failing to allocate.
    [[nodiscard]] std::uint32_t max_contiguous_pages(const HostKVPageLayout& layout) const noexcept;

    [[nodiscard]] bool can_allocate(const HostKVPageLayout& layout,
                                    std::uint32_t pages) const noexcept;
    [[nodiscard]] std::optional<HostKVAllocation> allocate(const HostKVPageLayout& layout,
                                                           std::uint32_t pages) noexcept;

    [[nodiscard]] std::optional<HostKVAllocationRecipe>
    plan_after_releases(std::span<const HostKVAllocationHandle> proposed_releases,
                        std::span<const HostKVAllocationRequest> target_allocations) const;

    [[nodiscard]] bool can_allocate_after_suballocation_releases(
        std::span<const HostKVSuballocationRelease> proposed_releases,
        std::span<const HostKVAllocationRequest> target_allocations) const;

    // The caller supplies already-sized empty outputs so successful adoption cannot allocate.
    // A false return leaves the arena and every input allocation unchanged.
    [[nodiscard]] bool apply_recipe(HostKVAllocationRecipe&& recipe,
                                    std::span<HostKVAllocation* const> proposed_releases,
                                    std::span<HostKVAllocation> target_allocations) noexcept;

    [[nodiscard]] std::pair<HostKVAllocation, HostKVAllocation> split(HostKVAllocation&& allocation,
                                                                      std::uint32_t page_offset);

    [[nodiscard]] HostKVAllocationView writable_view(HostKVAllocation& allocation);
    [[nodiscard]] HostKVAllocationConstView view(const HostKVAllocation& allocation) const;

private:
    friend class HostKVAllocation;
    friend class HostKVAllocationView;
    friend class HostKVAllocationConstView;

    struct Descriptor {
        std::size_t offset       = 0;
        std::size_t bytes        = 0;
        std::uint32_t layout     = 0;
        std::uint32_t pages      = 0;
        std::uint32_t generation = 1;
        bool active              = false;
    };

    struct FreeExtent {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    [[nodiscard]] std::optional<std::uint32_t>
    find_layout(const HostKVPageLayout& layout) const noexcept;
    [[nodiscard]] std::optional<std::size_t> find_free_extent(std::size_t bytes) const noexcept;
    // Free extents plus the not-yet-pinned chunks, in the single ascending order that find_free_extent
    // walks. Planning and allocation both resolve against this union so a planned offset is the
    // offset a later growth-backed allocation will actually receive.
    [[nodiscard]] std::vector<FreeExtent> available_extents() const;
    [[nodiscard]] std::size_t chunk_count() const noexcept;
    [[nodiscard]] FreeExtent chunk_extent(std::size_t index) const noexcept;
    [[nodiscard]] std::optional<std::size_t> find_growable_chunk(std::size_t bytes) const noexcept;
    bool commit_chunk(std::size_t index);
    [[nodiscard]] bool valid_handle(HostKVAllocationHandle handle) const noexcept;
    [[nodiscard]] std::uint32_t take_descriptor() noexcept;
    bool release_descriptor(std::uint32_t descriptor, std::uint32_t generation) noexcept;
    void insert_free_extent(FreeExtent extent) noexcept;
    // Two adjacent free regions may only be merged when they sit in the same growth chunk. Every
    // chunk is its own pinned buffer and allocation_data resolves a descriptor through that buffer,
    // so a merged extent crossing a boundary would describe memory that is not contiguous, and it
    // would also disagree with the offsets the planner picks from available_extents().
    [[nodiscard]] bool mergeable(std::size_t left, std::size_t right) const noexcept;
    [[nodiscard]] std::byte* allocation_data(const Descriptor& descriptor) const noexcept;
    void bump_revision() noexcept;

    std::vector<PinnedHostBuffer> chunks_;
    std::size_t capacity_bytes_ = 0;
    std::size_t chunk_bytes_ = 0;
    std::size_t committed_bytes_ = 0;
    std::size_t occupied_bytes_ = 0;
    std::vector<HostKVPageLayout> layouts_;
    std::vector<Descriptor> descriptors_;
    std::vector<std::uint32_t> free_descriptors_;
    std::vector<FreeExtent> free_extents_;
    std::uint64_t revision_ = 1;
};

} // namespace ninfer
