// Owns the planned graph, the chosen backend(s), caches, and the tensor pool.
#pragma once
#include "vknn/backend.h"
#include "vknn/config.h"
#include "vknn/graph.h"
#include "vknn/io_info.h"
#include "vknn/io_link.h"
#include "vknn/io_tensor.h"
#include "vknn/profiler.h"
#include "vknn/tensor.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    /// One plan-time fallback: a node the requested backend refused, with the engine's reason
    /// (filled by Backend::supportsNode, or by the tiny-GPU-island fold). Empty node names come
    /// from unnamed source nodes; `op` is the ONNX-style spelling (opTypeName).
    struct FallbackReason {
        std::string node;   ///< Node name (may be empty for unnamed nodes).
        std::string op;     ///< Op spelling, e.g. "Conv".
        std::string reason; ///< Why the node does not run on the requested backend.
    };

    /// One compiled plan for a single input-shape set. A Session holds one PlanBucket per declared
    /// shape; run() selects the matching bucket by exact input-shape match and dispatches its
    /// pre-recorded segments. A fixed-shape model has exactly one bucket, so the whole per-shape
    /// machinery collapses to a single map lookup on the hot path.
    ///
    /// Everything shape-dependent lives here: this bucket's optimized graph (its own tensor descs +
    /// folded shape constants), the per-node backend assignment, the compiled segments (which hold a
    /// reference to `graph` for their pre-recorded buffer/dispatch geometry), the runtime tensor pool
    /// sized to `graph`, and the boundary IO descriptions. The backends, caches, pipeline pool, and
    /// device weight pool are shared across buckets on the owning Session — a second bucket reuses
    /// them (autotune sigs and weight-cache keys are shape-safe).
    struct PlanBucket {
        std::string key; ///< Canonical input-shape key (see Session::shapeKey).
        /// This bucket's optimized graph, held by pointer so its address is stable: the compiled
        /// segments capture a `Graph &` into it, and that reference must stay valid across every
        /// PlanBucket move (buildBucket returns by value) and every buckets_ vector reallocation
        /// (a multi-bucket .vxm or prepareShapes appends buckets). A by-value Graph member would
        /// relocate on those moves and dangle the segments' reference — the boundary-residency crash.
        std::unique_ptr<Graph>                graph;
        std::vector<int>                      nodeBackendIdx;       ///< Backend index (into Session::backends_) per node.
        std::vector<std::unique_ptr<Segment>> segments;             ///< Compiled segments, run in order.
        std::vector<RtTensor>                 pool;                 ///< Runtime tensor pool for this bucket, indexed by TensorId.
        std::vector<FallbackReason>           fallbackReasons;      ///< Requested-backend refusals recorded while planning this bucket.
        bool                                  ioGpuConvert = false; ///< Whole graph on one GPU backend: 8-bit inputs upload raw + convert on the GPU.
        /// Graph inputs that reach a Gather node's index operand, paired with the smallest statically
        /// known axis size any of their Gathers selects from. Collected at plan time; run() validates a
        /// bound Int64 input against [-axisSize, axisSize) so an out-of-range index (an out-of-vocab
        /// token id against an embedding table) fails at bind time instead of reading out of bounds.
        std::vector<std::pair<TensorId, int64_t>> gatherIndexAxisSizes;
    };

    /// Owns the planned graph, the chosen backend(s), caches, and the tensor pool.
    class Session {
      public:
        ~Session();
        /// Build a session from an ONNX model file.
        static std::unique_ptr<Session> createFromOnnx(const std::string &path, const Config &cfg);
        /// Build from a pre-optimized ".vxm" file (skips ONNX parsing + graph passes).
        static std::unique_ptr<Session> createFromVxm(const std::string &path, const Config &cfg);
        /// Build from an already-imported graph (testing / surgery).
        static std::unique_ptr<Session> create(Graph &&g, const Config &cfg);
        /// Serialize the optimized graph to a ".vxm" file for fast reloads.
        bool saveOptimized(const std::string &path) const;

        /// Write the unified cache file (cfg.cacheFile) if the cache changed. Called automatically from
        /// ~Session(); also callable manually (e.g. before a checkpoint).
        void updateCache();

        /// Run the model. The bound input shapes select which compiled plan bucket runs: an exact
        /// match dispatches that bucket's pre-recorded segments; no match is Status::InvalidArgument
        /// with the available bucket shapes listed. A fixed-shape model has one bucket and always
        /// matches. To bind a zero-copy output, pre-fill `outputs` with an entry whose name + dmaBufFd
        /// select that output's caller buffer; that output is written into the fd and returned with no
        /// host data. `outputs` is then (re)filled with all results.
        Status run(const std::vector<IOTensor> &inputs, std::vector<IOTensor> &outputs);

        /// Compile an additional plan bucket for a declared input-shape set and cache it on this
        /// session, so a later run() with those shapes dispatches to it. The passes and plan re-run
        /// for the new shapes from the pristine imported graph; this is explicit and never happens
        /// implicitly inside run(). `shapes` maps an input tensor name to its full concrete shape;
        /// inputs left out keep the default (batch-fallback) resolution. Re-declaring a shape already
        /// planned is a no-op. Only sessions built from ONNX (createFromOnnx / create(Graph&&)) can add
        /// buckets — a .vxm session dispatches among its compiled buckets only and returns
        /// Status::Unsupported here.
        Status prepareShapes(const std::map<std::string, Shape> &shapes);

        /// Number of compiled plan buckets. Exactly 1 for a fixed-shape model; grows with each
        /// distinct shape set compiled via prepareShapes() (or loaded from a multi-bucket .vxm).
        size_t bucketCount() const noexcept {
            return buckets_.size();
        }

        // --- ergonomic API: names/shapes/dtypes come from the model; the caller passes only data ---
        /// Model inputs/outputs (name, concrete shape, dtype, element count). Use these to size buffers.
        /// The no-argument forms describe the default (first) bucket; the indexed forms describe any
        /// bucket, which differ per bucket in a multi-graph .vxm (each bucket is its own graph with its
        /// own input/output names) and in shape across shape buckets of one graph.
        std::vector<IOInfo> inputInfo() const;
        std::vector<IOInfo> outputInfo() const;
        std::vector<IOInfo> inputInfo(size_t bucket) const;
        std::vector<IOInfo> outputInfo(size_t bucket) const;
        /// Every compiled bucket's canonical key (see shapeKey), in bucket order. run() dispatches to
        /// the bucket whose key matches the caller's bound input names+shapes.
        std::vector<std::string> bucketKeys() const;
        /// Run with raw fp32 data, one buffer per model input in model order. Names/shapes/dtypes are
        /// filled from the model and the element counts are validated. Outputs come back fully described.
        Status run(const std::vector<std::vector<float>> &inputData, std::vector<IOTensor> &outputs);
        /// Single-input / single-output convenience: feed the input values, get the output values back.
        /// Returns empty on error. The shape is whatever the model declares (see inputInfo()).
        std::vector<float> infer(const std::vector<float> &input);

        /// The optimized graph the default (first) bucket runs (nodes, tensors, and their
        /// descriptors). A multi-bucket session has one such graph per shape; this returns bucket 0's.
        const Graph &graph() const noexcept {
            return *buckets_.front().graph;
        }
        /// The configuration this session was built with.
        const Config &config() const noexcept {
            return cfg_;
        }
        /// Per-op timing sink, populated when profiling is enabled in the Config.
        Profiler &profiler() noexcept {
            return profiler_;
        }
        /// Backend assignment per node, in node order (for reporting fallbacks).
        std::vector<BackendKind> nodeBackends() const;
        /// "<OpType> <node name>" for every node NOT running on the requested backend. A release run
        /// entirely on the GPU reports an empty list.
        std::vector<std::string> fallbackOps() const;
        /// Why each fallback in fallbackOps() happened, recorded once at plan() time: the requested
        /// backend's supportsNode refusal reason, or the tiny-GPU-island fold. Empty when the whole
        /// model runs on the requested backend. Reported for the default (first) bucket.
        const std::vector<FallbackReason> &fallbackReasons() const noexcept {
            return buckets_.front().fallbackReasons;
        }

        // --- engine-resident output->input links ------------------------------------------------
        // A link declares "this graph output feeds that graph input on the NEXT run" so recurrent
        // state (e.g. an autoregressive decoder's KV cache) stays inside the engine instead of
        // round-tripping through the caller every run. Semantics, identical on both backends:
        //   - A linked OUTPUT stays engine-resident: run() returns its IOTensor with name/shape/
        //     dtype but NO data (no device->host download on the GPU, no host donation on the CPU).
        //   - A linked INPUT keeps its engine-side values across runs. At the START of each run —
        //     before any node executes — the declared ranges are copied from the linked output's
        //     resident values (i.e. the PREVIOUS run's result) into it. With no prior run its
        //     values are zero.
        //   - Binding host data for a linked input is allowed and REINITIALIZES its resident state
        //     (the bound bytes form the base; the ranged copies then apply on top — pass empty
        //     ranges when reinitializing to suppress them). Binding a dma-buf fd to a linked tensor
        //     is rejected.
        //   - The copies never change math: the linked path moves the exact device (or host) bytes
        //     the unlinked path would have round-tripped, so results are bit-identical.
        // Declaring no links leaves every run byte-identical to a build without this feature.

        /// Link output `outputName` to input `inputName` with the given copy ranges (canonical
        /// elements; see LinkRange). Calling again for the same pair replaces the ranges — the
        /// per-run way to move the destination slot. On the Vulkan backend both tensors must be
        /// boundary tensors of one GPU segment with equal device element size; a violation is
        /// returned as an error naming both tensors (there is no silent slow path). This form
        /// requires the (outputName, inputName) pair to exist in exactly one plan bucket;
        /// multi-bucket sessions with the pair in several buckets use the bucket overload.
        Status linkOutputToInput(const std::string &outputName, const std::string &inputName, const std::vector<LinkRange> &ranges = {});
        /// Bucket-explicit form of linkOutputToInput() for multi-bucket sessions (bucket indices
        /// follow bucketKeys() order). The link applies only to runs that dispatch to `bucket`.
        Status linkOutputToInput(size_t bucket, const std::string &outputName, const std::string &inputName, const std::vector<LinkRange> &ranges = {});
        /// Copy the CURRENT resident values of a linked tensor (input or output name) into `out`,
        /// in the engine's internal storage dtype (fp32, or int64 for integer tensors). Reads the
        /// state as of the last completed run: a linked input reflects every ranged copy applied so
        /// far; a linked output holds the last run's produced values (whose fold into the input is
        /// still pending until the next run).
        Status readResident(const std::string &name, IOTensor &out);
        /// Remove every link. Linked outputs are returned with data again from the next run on;
        /// previously linked inputs keep their engine-side values until rebound.
        void clearLinks();

        // --- engine-side output reductions --------------------------------------------------------
        // An output registered for argmax stays engine-resident like a linked output: run() returns
        // its IOTensor with name/shape/dtype but NO data, and the engine reduces it to {index, value}
        // instead — on the GPU backend a single dispatch appended to the segment's pre-recorded
        // command stream with an 8-byte readback, replacing the full download of the vector plus the
        // host-side scan (an autoregressive decoder's per-token greedy argmax over the logits). The
        // selected index is the first occurrence of the maximum — identical to a left-to-right host
        // scan with a strictly-greater test over the same values — so a greedy token stream is
        // unchanged by registration.

        /// Register the boundary output `outputName` of `bucket` for engine-side argmax. The output
        /// must be float and effectively one-dimensional (every leading dim 1). On a backend without
        /// a device reduction path the output keeps its host copy and readOutputArgMax() scans it —
        /// same result, host cost. Registration is idempotent.
        Status setOutputArgMax(size_t bucket, const std::string &outputName);
        /// The argmax of a registered output as of the last completed run: first-occurrence index
        /// and the value widened to fp32. NotFound when the name was never registered.
        Status readOutputArgMax(const std::string &outputName, int64_t &index, float &value);

        /// Runtime tensor by name for layer-dump / debugging, or nullptr if no such tensor exists.
        /// The returned data is host-resident.
        const RtTensor *tensor(const std::string &name) const;

        /// True when every compiled segment, in every bucket, references its bucket's live graph
        /// object (Segment::compiledGraph == &bucket.graph). A segment captures a `Graph &` at compile
        /// time and dereferences it at run time (the Vulkan boundary path reads tensor descriptors
        /// through it), so a stale reference is a use-after-free. Guards the address-stability
        /// invariant against a PlanBucket move or a buckets_ reallocation relocating the graph.
        bool segmentGraphsLive() const noexcept {
            for (const PlanBucket &b: buckets_)
            {
                for (const std::unique_ptr<Segment> &s: b.segments)
                {
                    if (s && s->compiledGraph != b.graph.get())
                    {
                        return false;
                    }
                }
            }
            return true;
        }

      private:
        Session() = default;
        /// Ensure the shared backends are instantiated (once per session, first bucket build).
        void ensureBackends();
        /// Assign backends, partition into segments, and compile `g` into a fresh PlanBucket labelled
        /// `key`. Consumes `g` (it becomes the bucket's owned graph). Runs over the shared backends.
        PlanBucket buildBucket(Graph &&g, const std::string &key);
        /// Reassign small CPU-bounded GPU runs in `bucket` to CPU (avoid round trips).
        void foldTinyGpuIslands(PlanBucket &bucket);
        /// Checks a caller-provided input shape against `bucket`'s plan-frozen buffers; the single
        /// point every run() input shape passes through when a bucket is already selected.
        Status validateInputShape(const PlanBucket &bucket, TensorId id, const Shape &got) const;
        /// The canonical key for a shape assignment: each graph input's name and resolved shape. Two
        /// buckets with the same key are the same plan. `graph` supplies the input names/order.
        static std::string shapeKey(const Graph &graph);
        /// The key implied by a run's bound input shapes over `graph`'s inputs (an unbound or empty
        /// caller shape adopts that input's shape in `graph`). run() evaluates this per candidate
        /// bucket — buckets of a multi-graph .vxm have distinct input NAME sets, so the key must be
        /// built from each bucket's own graph, never from bucket 0's. `allowPositional` extends the
        /// forgiving single-input positional match to misnamed callers (homogeneous sessions only).
        static std::string runShapeKey(const Graph &graph, const std::vector<IOTensor> &inputs, bool allowPositional);
        /// True when every bucket exposes bucket 0's input names in the same order (one graph at
        /// several shapes). Falsity marks a multi-graph session, which dispatches strictly by name.
        /// Cached; recomputed when the bucket count changes.
        bool bucketsShareInputNames() const;

        /// One declared output->input link (see linkOutputToInput). `deviceSegment` set = the owning
        /// GPU segment applies the copies on-device; null = the Session copies host storage (the CPU
        /// backend's path).
        struct ResidentLink {
            size_t                 bucket = 0;
            std::string            outputName, inputName;
            TensorId               outId = kNoTensor, inId = kNoTensor;
            Segment               *deviceSegment = nullptr;
            std::vector<LinkRange> ranges;
            bool                   rangesDirty = false; ///< Ranges changed since last pushed to the device segment.
        };
        /// Bounds/overlap validation for a link's ranges against both tensors' logical shapes.
        Status validateLinkRanges(const Graph &g, TensorId outId, TensorId inId, const std::vector<LinkRange> &ranges) const;
        /// The link record for (bucket, tensor id) on the given side, or nullptr when not linked.
        const ResidentLink *linkedOutput(size_t bucket, TensorId id) const;
        const ResidentLink *linkedInput(size_t bucket, TensorId id) const;

        /// One registered engine-side argmax (see setOutputArgMax). `deviceSegment` set = the owning
        /// GPU segment reduces on-device and suppresses the output's download; null = the host copy
        /// stays and readOutputArgMax() scans it (the CPU backend's path).
        struct OutputArgMax {
            size_t      bucket = 0;
            std::string outputName;
            TensorId    outId         = kNoTensor;
            Segment    *deviceSegment = nullptr;
        };
        /// The argmax record for (bucket, output tensor id), or nullptr when not registered.
        const OutputArgMax *argMaxOutput(size_t bucket, TensorId id) const;
        /// Per-run link step: push dirty ranges to device segments and apply host-path copies.
        Status applyResidentLinks(size_t bucketIndex, PlanBucket &bucket);

        Config   cfg_;
        Profiler profiler_;
        // Declaration order matters for teardown: backends_ (owns the VulkanContext) must be
        // destroyed LAST, after the buckets' segments and pools release their device buffers. Members
        // are destroyed in reverse declaration order, so backends_ is declared first here.
        std::vector<std::unique_ptr<Backend>> backends_; // active, in priority order
        std::map<BackendKind, Backend *>      byKind_;
        // One compiled plan per declared input-shape set. buckets_[0] is the default (batch-fallback)
        // plan every fixed-shape model has; more are added by prepareShapes() or a multi-bucket .vxm.
        std::vector<PlanBucket> buckets_;
        // The pristine imported graph (pre-passes), retained only for ONNX-built sessions so
        // prepareShapes() can re-run the pipeline at a new shape. Empty for a .vxm session, which
        // cannot add buckets (its passes were baked at compile time, one shape per stored bucket).
        Graph importedGraph_;
        bool  hasImportedGraph_ = false;
        bool  planned_          = false;
        bool  graphOptimized_   = false; // graph came from .vxm (passes already applied)
        // bucketsShareInputNames() cache: valid while the bucket count equals uniformCheckedFor_.
        mutable size_t uniformCheckedFor_ = 0;
        mutable bool   bucketsUniform_    = true;
        // Declared output->input links, applied at the start of each run of their bucket.
        std::vector<ResidentLink> links_;
        // Registered engine-side output argmax reductions (see setOutputArgMax).
        std::vector<OutputArgMax> argMaxOutputs_;
    };

} // namespace vknn
