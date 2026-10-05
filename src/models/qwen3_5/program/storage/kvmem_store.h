#pragma once

// Block repository and bounded-working-set selection for KVMem-style retrieval.
//
// Pure host logic: no CUDA, no Program dependency, no physical page handles. The executor
// consumes KVMemPlan to decide which logical blocks back the execution-table window this round
// and which blocks must cross the Device/Host tier boundary.
//
// Granularity: selection and scoring work in `block_tokens`-token logical blocks (128 by
// default), while physical storage and transfer stay in the page store's 64-token logical pages.
// One block is exactly two pages, so a block is a natural extent membership.
//
// Window order is always ascending TRUE token position. The executor publishes those blocks into
// a dense window-row prefix through KVMemWindow: the causal mask reads
// `row <= cache_positions[t]`, and renumbering keeps row order identical to true-position order,
// so mask order and visibility order agree while K itself stays baked at its true position.
// See docs/maintainer/kvmem-port.md 2.12.

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct KVMemConfig {
    // Selection/transfer block granularity in tokens. Physical pages stay at 64 tokens.
    std::uint32_t block_tokens = 128;
    // Tokens the retrieved working set may occupy on the Device pool.
    std::uint32_t budget_tokens = 32768;
    // Always-kept prefix. Zero selects one block, matching the reference implementation, so the
    // sink is never accidentally empty.
    std::uint32_t sink_tokens = 0;
    // Always-kept suffix blocks under the Recency policy. Zero keeps no suffix block beyond what
    // the budget already buys from the tail.
    std::uint32_t recent_blocks = 0;
};

struct KVMemBlock {
    std::uint32_t block_id         = 0;
    std::uint32_t orig_pos_start   = 0;
    std::uint32_t n_tokens         = 0;
    bool          in_window        = false;
    bool          host_resident    = false;
    double        score            = 0.0;

    [[nodiscard]] std::uint32_t orig_pos_end() const noexcept {
        return orig_pos_start + n_tokens;
    }
};

// Diff of one selection round. `selected` is ascending true position and defines the window row
// layout: the window row of selected[i] is the sum of n_tokens over selected[0..i).
struct KVMemPlan {
    std::vector<std::uint32_t> stage_in;
    std::vector<std::uint32_t> stage_out;
    std::vector<std::uint32_t> selected;
    std::uint32_t window_tokens = 0;
};

// Renumbered-prefix view of one selection (docs/maintainer/kvmem-port.md 2.12).
//
// A selection is always the union of a retained sink prefix [0, sink_end) and a retained recent
// tail [recent_begin, total_tokens), with a hole between them whenever the budget cannot hold the
// whole context; when it can, both boundaries are zero and rel() is the identity. rel() maps a
// retained absolute position onto a dense prefix of window rows, so the unchanged causal mask
// `row <= cache_positions[t]` selects exactly the retained positions at or before the query.
//
// rel() is defined only for retained positions. Hole positions are never selected and must never
// be queried (invariants I1 and I3); the store validates I2, that both boundaries are multiples
// of the 64-token page, which is what makes a reselect a pure execution-table update with no
// device byte movement.
struct KVMemWindow {
    std::uint32_t sink_end      = 0;
    std::uint32_t recent_begin  = 0;
    std::uint32_t window_tokens = 0;

    [[nodiscard]] bool has_hole() const noexcept { return recent_begin > sink_end; }

    // Window row of a retained absolute position; undefined for hole positions.
    [[nodiscard]] std::uint32_t rel(std::uint32_t abs_pos) const noexcept {
        return abs_pos < sink_end ? abs_pos : abs_pos - recent_begin + sink_end;
    }

    // Whether an absolute position belongs to the retained union over a context of total tokens.
    [[nodiscard]] bool retains(std::uint32_t abs_pos, std::uint32_t total_tokens) const noexcept {
        return abs_pos < sink_end || (abs_pos >= recent_begin && abs_pos < total_tokens);
    }

    // Token delta applied to every recent-tail position. Always a non-positive multiple of the
    // page size, so rel()>>6 == abs>>6 + delta/64 and page offsets are untouched.
    [[nodiscard]] std::int64_t page_shift_tokens() const noexcept {
        return static_cast<std::int64_t>(sink_end) - static_cast<std::int64_t>(recent_begin);
    }
};

class KVMemBlockStore {
public:
    explicit KVMemBlockStore(KVMemConfig cfg);

    [[nodiscard]] const KVMemConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] std::uint32_t block_count() const noexcept {
        return static_cast<std::uint32_t>(blocks_.size());
    }
    [[nodiscard]] std::uint32_t total_tokens() const noexcept { return total_tokens_; }
    [[nodiscard]] const std::vector<KVMemBlock>& blocks() const noexcept { return blocks_; }
    [[nodiscard]] const KVMemBlock& block(std::uint32_t block_id) const;
    [[nodiscard]] std::uint32_t budget_blocks() const noexcept;

    // Blocks that own original token position `pos`, or block_count() when no block does.
    [[nodiscard]] std::uint32_t block_id_containing(std::uint32_t pos) const noexcept;

    // Register newly appended context tokens. Partial trailing blocks extend in place; a full
    // trailing block starts a new one. Returns how many blocks became newly full (0 or 1, since
    // only the trailing block can change state).
    std::uint32_t register_append(std::uint32_t n_tokens);

    // Inverse of register_append: drop trailing blocks so the store holds exactly token_pos
    // tokens. A partially covered trailing block shrinks in place; fully past blocks are popped.
    // Returns the popped blocks so the caller can release their tier slots. token_pos beyond
    // total_tokens is a no-op.
    std::vector<KVMemBlock> truncate_to(std::uint32_t token_pos);

    // Recency selection: keep the sink head, then fill the remaining budget from the newest tail.
    // When the whole context fits the budget every block is selected, which is exact full
    // attention. Produces a diff against the current window membership; it does not apply it.
    [[nodiscard]] KVMemPlan select_recency() const;

    // Apply a plan produced by this store: flips in_window membership and records window order.
    void apply(const KVMemPlan& plan);

    // Derive the renumbered-prefix window for a selection, given the true-position range of the
    // next attention call. Pure: it reads `plan.selected`, the block table and the range only.
    // Throws std::logic_error when a validated invariant would be violated, because every one of
    // them fails silently at output rather than crashing:
    //   I1 the next call's position range lies inside the retained union
    //   I2 both boundaries are multiples of the 64-token page (no byte movement on reselect)
    //   the window fits the configured budget
    [[nodiscard]] KVMemWindow window_for(const KVMemPlan& plan, std::uint32_t call_min_pos,
                                         std::uint32_t call_max_pos) const;

    // Window row of a block under the current membership, or false when it is not in the window.
    [[nodiscard]] bool window_row(std::uint32_t block_id, std::uint32_t& row_out) const noexcept;

    // Selector/scoring hook for a later retrieval round (phase 4b). Overwrites scores and returns
    // a plan over the same budget; identical to select_recency until a scorer exists.
    void set_scores(std::vector<double> scores);
    [[nodiscard]] KVMemPlan select_topk() const;

private:
    void validate_config(const KVMemConfig& cfg) const;
    [[nodiscard]] KVMemPlan finish_plan(std::vector<std::uint32_t> selected) const;
    [[nodiscard]] std::uint32_t sink_blocks() const noexcept;

    KVMemConfig cfg_;
    std::vector<KVMemBlock> blocks_;
    std::uint32_t total_tokens_ = 0;
    std::uint32_t window_tokens_ = 0;
};

// Mean-K retrieval scores (kvmem-port.md 1.4, 2.6 stage 4b). One F32 pre-RoPE key mean per
// selection block, accumulated on the host from device-reduced partial sums, plus the similarity
// ranking a query vector produces over those means. Pure host logic - no CUDA, no Program
// dependency, no physical page handles - so the whole scoring path is unit-testable (2.8).
//
// A block spans capture rounds: every round that touches it contributes a partial sum for the
// tokens of that block it covered, and `finalize` divides by the token count once the block is
// full. `score` ranks blocks by the scaled dot product between the query and their means,
// softmaxed over the blocks that have one; softmax is monotone in the dot product, so the
// selection ranking is the raw inner product and the stored score is its probability.
class KVMemScoreStore {
public:
    // `vector_width` is the captured key width (key heads x head dimension), `head_dim` the
    // per-head dimension the similarity scales by. Both are fixed by the model geometry.
    KVMemScoreStore(std::uint32_t block_tokens, std::uint32_t vector_width, std::uint32_t head_dim);

    [[nodiscard]] std::uint32_t block_tokens() const noexcept { return block_tokens_; }
    [[nodiscard]] std::uint32_t vector_width() const noexcept { return vector_width_; }
    [[nodiscard]] std::uint32_t block_count() const noexcept {
        return static_cast<std::uint32_t>(blocks_.size());
    }

    // Add one device partial sum for `block_id`: the sum over `count` of that block's tokens.
    // Throws when the vector width, a zero count, or the accumulated token count does not fit
    // the block, because a partial sum that silently overflows becomes a wrong mean.
    void accumulate(std::uint32_t block_id, const float* partial_sum, std::uint32_t count);

    // Promote a full block to its mean. A block whose accumulated token count is not the block
    // size (still growing, or truncated) stays unfinalized and cannot be scored.
    void finalize(std::uint32_t block_id);

    // Forget every block at or beyond `count` (context truncation or prefix fork).
    void truncate(std::uint32_t count);

    [[nodiscard]] bool has_mean(std::uint32_t block_id) const;
    [[nodiscard]] std::uint32_t captured_tokens(std::uint32_t block_id) const;
    [[nodiscard]] const std::vector<float>& mean(std::uint32_t block_id) const;

    // Softmaxed similarity of every block mean against `query`. Blocks without a mean score 0,
    // which places them after every scored block while still letting them be selected when the
    // budget has room left.
    [[nodiscard]] std::vector<double> score(const float* query, std::uint32_t width) const;

private:
    struct Block {
        std::vector<float> sum;   // device partial sums, cleared once finalized
        std::vector<float> mean;  // sum / block_tokens, empty until finalized
        std::uint32_t tokens = 0;
    };

    [[nodiscard]] const Block& require_block(std::uint32_t block_id) const;
    [[nodiscard]] Block& require_block(std::uint32_t block_id);

    std::uint32_t block_tokens_ = 0;
    std::uint32_t vector_width_ = 0;
    std::uint32_t head_dim_ = 0;
    std::vector<Block> blocks_;
};

// One block a captured range contributes to, and how much of that range falls inside it.
//
// block_sum_by_position reduces a whole range of tokens into one column per block that range
// touches, so the host side has to recover each column's block identity and token count from the
// same arithmetic the Op used. Deriving it here - pure, from the range alone - keeps the two
// descriptions of the same range from drifting apart.
struct KVMemBlockShare {
    std::uint32_t block_id = 0;
    std::uint32_t tokens   = 0;  // tokens of this block inside the range; always positive
};

// Blocks that the token range [first_position, first_position + tokens) touches, ascending.
// Throws std::invalid_argument for a negative position or an empty range, because such a range
// misattributes partial sums instead of failing.
[[nodiscard]] std::vector<KVMemBlockShare>
kvmem_block_shares(std::int64_t first_position, std::int64_t tokens, std::uint32_t block_tokens);

} // namespace ninfer::models::qwen3_5::detail
