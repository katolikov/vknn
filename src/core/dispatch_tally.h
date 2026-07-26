// Recorded-compute-dispatch accounting, as a backend-free counter. Lives in core — compiled into
// every build, including hosts without the Vulkan backend — so the attribution arithmetic the
// Vulkan segment reports per op is the same code the host tests exercise.
//
// A node's dispatch count is not its node count: one recorded op routinely emits several
// vkCmdDispatch calls (a split-K GEMM's partial + reduce passes, Winograd's transform/GEMM/output
// passes, fused attention's partial + combine, a boundary or layout convert), so a segment's real
// dispatch count runs well above its node count. The counter is the measurement that makes a
// "dispatch-bound" claim checkable instead of inferred.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vknn {

    /// Counts compute dispatches as they are recorded into a command buffer and attributes them to
    /// the op that recorded them.
    ///
    /// Two numbers come out of one recording pass: `runTotal()` — every dispatch recorded since
    /// beginRun(), including the ones no graph node owns (boundary/layout converts, resident-link
    /// copies, decode-chain feedback, output argmax epilogues) — and `nodeDispatches(k)`, the share
    /// recorded between openNode(k) and closeNode(). The two differ by exactly the unattributed
    /// dispatches, which is the point: `runTotal() - nodeTotal()` is the per-run cost the per-op
    /// profile table cannot show.
    ///
    /// Counts ACCUMULATE across openNode/closeNode spans for the same index, so a decode chain that
    /// records its body once per iteration reports each node's total over the whole recording.
    ///
    /// Single-threaded by contract: note() is called from the thread recording the command buffer,
    /// and one counter belongs to one device context (one session's backend), so the recording
    /// thread is the only writer.
    class DispatchTally {
      public:
        /// Count one recorded dispatch. Called from the single dispatch-recording site.
        void note() noexcept {
            ++lifetime_;
            ++sinceRunStart_;
        }

        /// Dispatches recorded over the counter's whole lifetime, across every recording pass.
        uint64_t lifetime() const noexcept {
            return lifetime_;
        }

        /// Start attributing a fresh recording pass over `nodeCount` nodes: clears the per-node
        /// counts and the run total. A re-record (changed boundary buffer, chain reconfiguration)
        /// calls this again and reports the new recording, not the sum of both.
        void beginRun(size_t nodeCount) {
            perNode_.assign(nodeCount, 0);
            sinceRunStart_ = 0;
            openIndex_     = kNoOpenNode;
            openAt_        = 0;
        }

        /// Begin attributing dispatches to node `index`. An index past the beginRun() node count is
        /// ignored (the span still closes cleanly), so a caller cannot corrupt the table.
        void openNode(size_t index) noexcept {
            openIndex_ = index;
            openAt_    = sinceRunStart_;
        }

        /// Attribute everything recorded since openNode() to that node, adding to whatever the node
        /// already holds. A close with no open span is a no-op.
        void closeNode() noexcept {
            if (openIndex_ != kNoOpenNode && openIndex_ < perNode_.size())
            {
                perNode_[openIndex_] += (uint32_t) (sinceRunStart_ - openAt_);
            }
            openIndex_ = kNoOpenNode;
        }

        /// Dispatches attributed to node `index` in the current recording (0 for an unknown index).
        uint32_t nodeDispatches(size_t index) const noexcept {
            return index < perNode_.size() ? perNode_[index] : 0;
        }

        /// Every dispatch recorded since beginRun(), owned by a node or not.
        uint64_t runTotal() const noexcept {
            return sinceRunStart_;
        }

        /// The share of runTotal() attributed to graph nodes; the remainder is the segment's
        /// boundary/epilogue overhead.
        uint64_t nodeTotal() const noexcept {
            uint64_t sum = 0;
            for (uint32_t n: perNode_)
            {
                sum += n;
            }
            return sum;
        }

        /// Nodes the current recording pass covers (what beginRun() was given).
        size_t nodeCount() const noexcept {
            return perNode_.size();
        }

      private:
        /// openIndex_ sentinel for "no span is open".
        static constexpr size_t kNoOpenNode = (size_t) -1;

        uint64_t              lifetime_      = 0;
        uint64_t              sinceRunStart_ = 0;
        std::vector<uint32_t> perNode_;
        size_t                openIndex_ = kNoOpenNode;
        uint64_t              openAt_    = 0;
    };

} // namespace vknn
