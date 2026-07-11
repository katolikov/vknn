// Segment: one compiled, executable run of graph nodes assigned to a single backend.
#pragma once
#include "vknn/exec_context.h"
#include "vknn/io_link.h"
#include "vknn/status.h"
#include "vknn/tensor.h"
#include <string>
#include <vector>

namespace vknn {

    class Backend;

    /// One compiled, executable run of graph nodes assigned to a single Backend. A session's graph is
    /// partitioned into segments (one per contiguous run of same-backend nodes); running the graph
    /// runs its segments in order. Produced by Backend::compileSegment() and owned by the Session.
    class Segment {
      public:
        virtual ~Segment() = default;
        /// Execute this segment's nodes against the run's tensor pool. `ctx` is valid only for the
        /// duration of the call and must not be retained past it.
        virtual void run(ExecContext &ctx) = 0;

        // ---- device-resident output->input links (Session::linkOutputToInput) ----
        // A backend that keeps boundary tensors resident across runs (the Vulkan segment) overrides
        // these; the default says "no device path" and the Session falls back to host-buffer linking
        // (the CPU backend's semantics).

        /// Register `sourceOutput` (one of this segment's boundary outputs) as the resident source
        /// for `destInput` (one of its boundary inputs): the segment copies the declared ranges from
        /// the source's device buffer into the destination's at the START of each run, before any
        /// node executes, and stops downloading the source to host. Returns Unsupported when this
        /// segment has no device-resident path; any other failure fills `whyNot`.
        virtual Status addResidentLink(TensorId sourceOutput, TensorId destInput, std::string &whyNot) {
            (void) sourceOutput;
            (void) destInput;
            (void) whyNot;
            return Status::Unsupported;
        }
        /// Replace the copy ranges of a link registered by addResidentLink(). Offsets/counts are
        /// canonical elements, already bounds-checked by the Session. `rangeSets` holds one set per
        /// decode-chain iteration (set i is applied at the head of iteration i); a single-step
        /// caller passes exactly one set. Sets beyond the segment's recorded chain length are
        /// ignored; recorded iterations past the last provided set apply no copies.
        virtual void setResidentLinkRangeSets(TensorId sourceOutput, TensorId destInput, const std::vector<std::vector<LinkRange>> &rangeSets) {
            (void) sourceOutput;
            (void) destInput;
            (void) rangeSets;
        }
        /// Drop every link registered by addResidentLink() (linked outputs download again, linked
        /// inputs re-pack from host state on the next run).
        virtual void clearResidentLinks() {
        }
        /// Download the current device residency of a linked boundary tensor into `rt.host` (fp32
        /// canonical NCHW). False when this segment holds no buffer for `id`.
        virtual bool downloadResident(TensorId id, RtTensor &rt) {
            (void) id;
            (void) rt;
            return false;
        }

        // ---- device-side output reductions (Session::setOutputArgMax) ----
        // A backend that can reduce a boundary output on-device (the Vulkan segment) overrides
        // these; the default says "no device path" and the Session computes the reduction on the
        // host copy instead.

        /// Register `output` (one of this segment's boundary outputs, flat float storage) for an
        /// on-device argmax epilogue: every run appends one reduction dispatch after the graph's
        /// nodes into the same command stream and stops downloading the full output to host;
        /// readOutputArgMax() returns the last run's result. Unsupported = no device path.
        virtual Status setOutputArgMax(TensorId output, std::string &whyNot) {
            (void) output;
            (void) whyNot;
            return Status::Unsupported;
        }
        /// The last run's argmax of an output registered by setOutputArgMax(): the first-occurrence
        /// (lowest) index of the maximum element and its value widened to fp32. `step` selects a
        /// decode-chain iteration's result slot (0 on the single-step path). False when `output` is
        /// not registered here or `step` is outside the segment's result slots.
        virtual bool readOutputArgMax(TensorId output, int step, int64_t &index, float &value) {
            (void) output;
            (void) step;
            (void) index;
            (void) value;
            return false;
        }

        /// Read back only row `row` of a flat 2-D boundary `output` ([.., R, V]) on the next runs,
        /// copying V elements from device offset row*V instead of the whole [R, V] buffer — the
        /// prefill logits case, where only the last real token's row feeds the first decode token.
        /// `row` < 0 clears the selection (full readback resumes). Unsupported = no device path
        /// (the Session then downloads the full output and the caller slices the host copy).
        virtual Status setOutputRow(TensorId output, int64_t row, std::string &whyNot) {
            (void) output;
            (void) row;
            (void) whyNot;
            return Status::Unsupported;
        }

        // ---- device-resident decode chains (Session::configureDecodeChain, ADR-0015) ----
        // A backend that can record K decode iterations into one pre-recorded command stream (the
        // Vulkan segment) overrides these; the default says "no device path".

        /// Configure this segment to record `steps` decode iterations per run as one command-buffer
        /// chain: between iterations a feedback dispatch writes the previous iteration's argmax
        /// index into `tokenInput`, advances `positionInput` by one, and marks the newly valid
        /// `maskInput` slot; resident-link copies apply their per-iteration range set and the
        /// argmax epilogue lands in its per-iteration result slot. `argMaxOutput` must already be
        /// registered via setOutputArgMax(). Unsupported = no device chain path; any other failure
        /// fills `whyNot`.
        virtual Status configureDecodeChain(TensorId tokenInput, TensorId positionInput, TensorId maskInput, TensorId argMaxOutput, int steps, std::string &whyNot) {
            (void) tokenInput;
            (void) positionInput;
            (void) maskInput;
            (void) argMaxOutput;
            (void) steps;
            (void) whyNot;
            return Status::Unsupported;
        }
        /// Set the next runs' chain window: `basePosition` is iteration 0's absolute decode
        /// position (the feedback dispatches derive iteration i's position as basePosition + i),
        /// and `activeSteps` is how many recorded iterations the next runs execute (a prefix of
        /// the chain; the host provides iteration 0's inputs and range set as usual). Sticky until
        /// changed. Unsupported when no chain is configured.
        virtual Status setDecodeChainWindow(int64_t basePosition, int activeSteps) {
            (void) basePosition;
            (void) activeSteps;
            return Status::Unsupported;
        }

        /// Backend that compiled and owns this segment. Non-owning; the backend outlives the segment.
        Backend *backend = nullptr;
        /// The graph this segment was compiled against, recorded so the segment's captured `Graph &`
        /// can be checked against the session's live bucket graph (they must be the same object: the
        /// graph's address is stable for the session's lifetime). Set by the backend's compileSegment().
        const Graph *compiledGraph = nullptr;
        /// True when this is a CPU fallback segment that exists because the primary backend cannot run
        /// these ops. Drives the fallback warning and the profiler tag.
        bool isFallback = false;
        /// True when the whole graph runs on this (GPU) backend, so 8-bit image graph-inputs are
        /// uploaded raw and converted on the GPU (no host uint8->fp32->fp16 pack). Off whenever any CPU
        /// segment exists, since a CPU consumer needs the fp32 host copy.
        bool ioGpuConvert = false;
        /// Indices into graph.nodes of the nodes this segment executes, in execution order.
        std::vector<int> nodeIdx;
        /// Boundary set: tensor ids this segment consumes from outside the segment.
        std::vector<TensorId> boundaryInputs;
        /// Boundary set: tensor ids this segment produces for consumers outside the segment.
        std::vector<TensorId> boundaryOutputs;
    };

} // namespace vknn
