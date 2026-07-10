#include "vknn/session.h"
#include "../import/passes.h"
#include "core/quant_int4.h"
#include "vknn/logging.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <limits>
#include <fstream>
#include <set>
#include <sys/stat.h>

namespace vknn {

    // ---- non-fp32 I/O boundary conversion -----------------------------------------------------------
    // The engine computes in fp32 (carrying Int64 for shape/index paths). Model I/O may declare UINT8 /
    // FLOAT16 / integer dtypes; these helpers convert at the graph boundary so any dtype the ONNX declares
    // works end-to-end without a fp32-only assumption.

    // Caller bytes (in dtype `src`) -> internal storage: int64 for Int64, fp32 for everything else.
    // Storage is always sized for `elems`; a short caller buffer is tolerated, not rejected — every
    // branch fills only min(elems, in.size()/bytesPer) elements and zeroes the tail past them, so a bound
    // tensor never carries an earlier run's values into the elements the caller left out.
    static void bindInput(DType src, const std::vector<uint8_t> &in, int64_t elems, RtTensor &rt) {
        if (src == DType::Int64)
        {
            rt.dtype = DType::Int64;
            rt.host.resizeElems(elems, DType::Int64);
            int64_t avail = std::min<int64_t>(elems, (int64_t) (in.size() / 8));
            std::memcpy(rt.host.bytes.data(), in.data(), (size_t) avail * 8);
            if (avail < elems)
            {
                std::memset(rt.host.i64() + avail, 0, (size_t) (elems - avail) * 8);
            }
            return;
        }
        rt.dtype = DType::Float32;
        rt.host.resizeElems(elems, DType::Float32);
        float *f = rt.host.f32();
        // Elements that fit in both the destination (elems) and the caller buffer at bytesPer each.
        auto fitElems = [&](int64_t bytesPer) {
            return std::min<int64_t>(elems, (int64_t) (in.size() / bytesPer));
        };
        int64_t filled = 0;
        switch (src)
        {
            case DType::Float32: {
                filled = fitElems(4);
                std::memcpy(f, in.data(), (size_t) filled * 4);
                break;
            }
            case DType::Float16: {
                const fp16_t *h = reinterpret_cast<const fp16_t *>(in.data());
                filled          = fitElems(2);
                for (int64_t i = 0; i < filled; ++i)
                {
                    f[i] = halfToFloat(h[i]);
                }
                break;
            }
            case DType::UInt8:
                filled = fitElems(1);
                for (int64_t i = 0; i < filled; ++i)
                {
                    f[i] = (float) reinterpret_cast<const uint8_t *>(in.data())[i];
                }
                break;
            case DType::Int8:
                filled = fitElems(1);
                for (int64_t i = 0; i < filled; ++i)
                {
                    f[i] = (float) reinterpret_cast<const int8_t *>(in.data())[i];
                }
                break;
            case DType::Int32:
                filled = fitElems(4);
                for (int64_t i = 0; i < filled; ++i)
                {
                    f[i] = (float) reinterpret_cast<const int32_t *>(in.data())[i];
                }
                break;
            default: {
                filled = fitElems(4);
                std::memcpy(f, in.data(), (size_t) filled * 4);
                break;
            }
        }
        if (filled < elems)
        {
            std::memset(f + filled, 0, (size_t) (elems - filled) * 4);
        }
    }

    // Internal storage (rt.dtype fp32 or int64) -> output bytes in the model's declared dtype `dst`.
    static void readbackOutput(DType dst, RtTensor &rt, int64_t elems, IOTensor &io) {
        io.dtype = dst;
        if (dst == rt.dtype)
        {
            // fast path: rt.host already holds the declared dtype (fp32/fp16/uint8/...). MOVE it into the
            // output instead of copying — rt.host is refilled from the device buffer on the next run before
            // it is read again, so donating its storage here avoids a full-tensor copy of every output.
            io.data      = rt.host.bytes.release();
            rt.hostValid = false;
            return;
        }
        bool srcI64 = rt.dtype == DType::Int64;
        auto srcF32 = [&](int64_t i) -> float {
            return srcI64 ? (float) rt.host.i64()[i] : rt.host.f32()[i];
        };
        auto srcI = [&](int64_t i) -> int64_t {
            return srcI64 ? rt.host.i64()[i] : (int64_t) rt.host.f32()[i];
        };
        io.data.assign((size_t) elems * dtypeSize(dst), 0);
        switch (dst)
        {
            case DType::Float32:
                for (int64_t i = 0; i < elems; ++i)
                {
                    reinterpret_cast<float *>(io.data.data())[i] = srcF32(i);
                }
                break;
            case DType::Float16:
                if (srcI64)
                {
                    for (int64_t i = 0; i < elems; ++i)
                    {
                        reinterpret_cast<fp16_t *>(io.data.data())[i] = floatToHalf(srcF32(i));
                    }
                } else
                {
                    floatToHalfBulk(rt.host.f32(), reinterpret_cast<fp16_t *>(io.data.data()), elems);
                }
                break;
            case DType::UInt8:
                for (int64_t i = 0; i < elems; ++i)
                {
                    int64_t v                                      = srcI(i);
                    reinterpret_cast<uint8_t *>(io.data.data())[i] = (uint8_t) (v < 0 ? 0 : (v > 255 ? 255 : v));
                }
                break;
            case DType::Int8:
                for (int64_t i = 0; i < elems; ++i)
                {
                    reinterpret_cast<int8_t *>(io.data.data())[i] = (int8_t) srcI(i);
                }
                break;
            case DType::Int32:
                for (int64_t i = 0; i < elems; ++i)
                {
                    reinterpret_cast<int32_t *>(io.data.data())[i] = (int32_t) srcI(i);
                }
                break;
            case DType::Int64:
                for (int64_t i = 0; i < elems; ++i)
                {
                    reinterpret_cast<int64_t *>(io.data.data())[i] = srcI(i);
                }
                break;
            default:
                io.data = rt.host.bytes.toVector();
                break;
        }
    }

    // Graph inputs that reach an index-consuming operand — a Gather's index, or a fused Rope's
    // position (the Gather the rope fusion removed) — directly or through value-preserving hops
    // (layout/dtype converts, metadata reshapes), paired with the SMALLEST statically known axis
    // size any of those lookups selects from. Feeds PlanBucket::gatherIndexAxisSizes, which run()
    // checks when the input is bound: an out-of-range index (an out-of-vocab token id against an
    // embedding table) then fails with its value and position instead of reading out of bounds inside
    // the op — the class of crash a caller-supplied id can otherwise trigger.
    static std::vector<std::pair<TensorId, int64_t>> collectGatherIndexAxisSizes(const Graph &g) {
        // Last writer of each tensor, so an index can be traced back to its boundary source.
        std::vector<int> producer(g.tensors.size(), -1);
        for (int ni = 0; ni < (int) g.nodes.size(); ++ni)
        {
            for (TensorId out: g.nodes[ni].outputs)
            {
                if (out != kNoTensor)
                {
                    producer[(size_t) out] = ni;
                }
            }
        }
        // Ops whose output holds the same element values as inputs[0], so a bound the consumer's
        // Gather imposes on its index applies unchanged to the traced tensor.
        auto valuePreserving = [](OpType t) {
            return t == OpType::ConvertLayout || t == OpType::ConvertDtype || t == OpType::Cast || t == OpType::Reshape || t == OpType::Squeeze || t == OpType::Unsqueeze || t == OpType::Flatten || t == OpType::Identity;
        };
        std::vector<std::pair<TensorId, int64_t>> limits;
        for (const Node &nd: g.nodes)
        {
            // Index-consuming ops with a statically bounded lookup: Gather (index operand 1 selects
            // rows of the data's gather axis) and the fused Rope (position operand 1 selects rows of
            // the cos/sin tables — the Gather the fusion removed, so the same bind-time bound holds).
            int64_t axisSize = 0;
            if (nd.type == OpType::Gather && nd.inputs.size() >= 2 && nd.inputs[0] != kNoTensor && nd.inputs[1] != kNoTensor)
            {
                // Axis normalization mirrors GatherCpu: negative counts from the back, then clamps.
                const Shape &data = g.desc(nd.inputs[0]).shape;
                int64_t      rank = (int64_t) data.size();
                int64_t      axis = nd.attr.geti("axis", 0);
                if (axis < 0)
                {
                    axis += rank;
                }
                axis = std::max<int64_t>(0, std::min<int64_t>(axis, rank > 0 ? rank - 1 : 0));
                if (rank == 0 || data[axis] <= 0)
                {
                    continue; // axis size not statically known: nothing to validate against
                }
                axisSize = data[axis];
            } else if (nd.type == OpType::Rope && nd.inputs.size() >= 3 && nd.inputs[1] != kNoTensor && nd.inputs[2] != kNoTensor)
            {
                const Shape &table = g.desc(nd.inputs[2]).shape;
                if (table.empty() || table[0] <= 0)
                {
                    continue;
                }
                axisSize = table[0];
            } else
            {
                continue;
            }
            TensorId traced = nd.inputs[1];
            for (int hop = 0; traced != kNoTensor && hop < 64; ++hop)
            {
                int pi = producer[(size_t) traced];
                if (pi < 0)
                {
                    break; // boundary: a graph input or an initializer
                }
                const Node &pn = g.nodes[(size_t) pi];
                if (!valuePreserving(pn.type) || pn.inputs.empty() || pn.inputs[0] == kNoTensor)
                {
                    traced = kNoTensor; // a real op computes the index: its values are not the input's
                    break;
                }
                traced = pn.inputs[0];
            }
            if (traced == kNoTensor || !g.desc(traced).isInput)
            {
                continue;
            }
            bool merged = false;
            for (auto &lim: limits)
            {
                if (lim.first == traced)
                {
                    lim.second = std::min(lim.second, axisSize);
                    merged     = true;
                    break;
                }
            }
            if (!merged)
            {
                limits.push_back({traced, axisSize});
            }
        }
        return limits;
    }

    Session::~Session() {
        updateCache(); // flush autotune/pipeline changes to the cache file if any
    }

    void Session::updateCache() {
        for (auto &b: backends_)
        {
            b->finalize(); // writes the unified cache file when its contents changed
        }
    }

    std::unique_ptr<Session> Session::createFromOnnx(const std::string &path, const Config &cfg) {
        Config c = cfg;
        if (c.cacheFile.empty())
        {
            c.cacheFile = Runtime::defaultCacheFile(path);
        }
        Graph g = importOnnx(path);
        return create(std::move(g), c);
    }

    std::unique_ptr<Session> Session::createFromVxm(const std::string &path, const Config &cfg) {
        auto s             = std::unique_ptr<Session>(new Session());
        s->graphOptimized_ = true; // passes already baked in; buckets carry one shape each
        s->cfg_            = cfg;
        if (s->cfg_.cacheFile.empty())
        {
            s->cfg_.cacheFile = Runtime::defaultCacheFile(path);
        }
        cfg.applyLogLevel();
        s->profiler_.setEnabled(cfg.profile);
        auto t0 = std::chrono::high_resolution_clock::now();
        // A .vxm carries one already-optimized graph per declared shape (or per graph, for a
        // multi-graph file). Compile every stored bucket over the shared backends so run() can
        // dispatch among them by bound input names+shapes. Buckets are STREAMED from the file and
        // built one at a time: with freeWeightsAfterUpload, host memory peaks at a single bucket's
        // weights instead of the whole file's expansion — the difference between loading and OOMing
        // a multi-bucket LLM on-device. A single-bucket file (every fixed-shape model) yields
        // exactly one PlanBucket, unchanged from before.
        s->ensureBackends();
        bool ok = loadGraphBinBucketsStreamed(path, [&](Graph &&gb, const std::string &, size_t, size_t) {
            std::string key = Session::shapeKey(gb);
            s->buckets_.push_back(s->buildBucket(std::move(gb), key));
            return true;
        });
        if (!ok || s->buckets_.empty())
        {
            VKNN_ERROR << "failed to load .vxm: " << path;
            return nullptr;
        }
        s->planned_ = true;
        s->warnUnmatchedDebugPatterns(); // all buckets built: name the pattern knobs that matched nothing
        auto t1 = std::chrono::high_resolution_clock::now();
        VKNN_INFO << "Session created from .vxm in " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms (" << s->buckets_.size() << " bucket(s))";
        return s;
    }

    bool Session::saveOptimized(const std::string &path) const {
        return saveGraphBin(*buckets_.front().graph, path);
    }

    std::unique_ptr<Session> Session::create(Graph &&g, const Config &cfg) {
        auto s  = std::unique_ptr<Session>(new Session());
        s->cfg_ = cfg;
        cfg.applyLogLevel();
        s->profiler_.setEnabled(cfg.profile);
        // ONNX-built session: retain the pristine imported graph so prepareShapes() can re-run the
        // pass pipeline at a new declared shape. The default bucket is compiled from a copy so the
        // pristine graph (with its weights) survives the passes and the freeWeightsAfterUpload reclaim.
        s->importedGraph_    = g;
        s->hasImportedGraph_ = true;
        auto  t0             = std::chrono::high_resolution_clock::now();
        Graph def            = std::move(g);
        s->ensureBackends();
        std::string key = Session::shapeKey(def); // key reflects the input names before shape resolution
        s->buckets_.push_back(s->buildBucket(std::move(def), key));
        // The default bucket resolved the input shapes (batch fallback + any Config::inputShapes); key
        // the bucket by those resolved shapes so run() dispatch matches concrete caller shapes.
        s->buckets_.front().key = Session::shapeKey(*s->buckets_.front().graph);
        s->planned_             = true;
        s->warnUnmatchedDebugPatterns(); // the default bucket is built: name the pattern knobs that matched nothing
        auto t1 = std::chrono::high_resolution_clock::now();
        VKNN_INFO << "Session created in " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
        return s;
    }

    void Session::foldTinyGpuIslands(PlanBucket &bucket) {
        Graph            &graph_          = *bucket.graph;
        std::vector<int> &nodeBackendIdx_ = bucket.nodeBackendIdx;
        int               cpuIdx          = -1;
        for (size_t i = 0; i < backends_.size(); ++i)
        {
            if (backends_[i]->kind() == BackendKind::Cpu)
            {
                cpuIdx = (int) i;
            }
        }
        if (cpuIdx < 0)
        {
            return; // no CPU backend to fall back to
        }
        auto isCpu = [&](int ni) {
            return backends_[nodeBackendIdx_[ni]]->kind() == BackendKind::Cpu;
        };
        // Approximate per-node work: a conv/gemm output costs Cin*KH*KW per element, everything else ~1.
        auto nodeCost = [&](int ni) -> int64_t {
            const Node &nd       = graph_.nodes[ni];
            int64_t     outElems = nd.outputs.empty() || nd.outputs[0] == kNoTensor ? 0 : numElements(graph_.desc(nd.outputs[0]).shape);
            if ((nd.type == OpType::Conv || nd.type == OpType::Gemm) && nd.inputs.size() > 1)
            {
                const Shape &w = graph_.desc(nd.inputs[1]).shape;
                int64_t      k = 1;
                for (size_t i = 1; i < w.size(); ++i)
                {
                    k *= w[i]; // Cin*KH*KW (Gemm: K)
                }
                return outElems * std::max<int64_t>(k, 1);
            }
            if (nd.type == OpType::ConvGemm && nd.inputs.size() > 1)
            {
                const Shape &w = graph_.desc(nd.inputs[1]).shape; // repacked [K, Cout]: dim 0 IS Cin*KH*KW
                return outElems * std::max<int64_t>(w.empty() ? 1 : w[0], 1);
            }
            return outElems;
        };
        const int64_t kKeepOnGpu = 2'000'000; // work below this isn't worth a CPU<->GPU round trip
        bool          changed    = true;
        while (changed)
        {
            changed = false;
            // contiguous same-backend runs over the (topo-sorted) node order
            std::vector<int> runOf(graph_.nodes.size(), -1);
            int              nRuns = 0;
            for (size_t n = 0; n < graph_.nodes.size(); ++n)
            {
                if (n == 0 || nodeBackendIdx_[n] != nodeBackendIdx_[n - 1])
                {
                    nRuns++;
                }
                runOf[n] = nRuns - 1;
            }
            std::vector<int>              producerRun(graph_.tensors.size(), -1);
            std::vector<std::vector<int>> runNodes(nRuns);
            for (size_t n = 0; n < graph_.nodes.size(); ++n)
            {
                runNodes[runOf[n]].push_back((int) n);
                for (TensorId o: graph_.nodes[n].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producerRun[o] = runOf[n];
                    }
                }
            }
            for (int r = 0; r < nRuns && !changed; ++r)
            {
                if (isCpu(runNodes[r].front()))
                {
                    continue; // already CPU
                }
                std::set<TensorId> internal;
                for (int ni: runNodes[r])
                {
                    for (TensorId o: graph_.nodes[ni].outputs)
                    {
                        if (o != kNoTensor)
                        {
                            internal.insert(o);
                        }
                    }
                }
                bool    touchesGpu = false;
                int64_t work       = 0;
                for (int ni: runNodes[r])
                {
                    work += nodeCost(ni);
                    for (TensorId in: graph_.nodes[ni].inputs) // fed by another GPU run?
                    {
                        if (in != kNoTensor && !internal.count(in) && producerRun[in] >= 0 && !isCpu(runNodes[producerRun[in]].front()))
                        {
                            touchesGpu = true;
                        }
                    }
                }
                for (size_t q = 0; q < graph_.nodes.size() && !touchesGpu; ++q)
                { // consumed by a GPU run?
                    if (runOf[q] == r || isCpu((int) q))
                    {
                        continue;
                    }
                    for (TensorId x: graph_.nodes[q].inputs)
                    {
                        if (x != kNoTensor && internal.count(x))
                        {
                            touchesGpu = true;
                            break;
                        }
                    }
                }
                if (touchesGpu || work >= kKeepOnGpu)
                {
                    continue; // connected to GPU work, or heavy -> keep
                }
                bool cpuOk = true;
                for (int ni: runNodes[r])
                {
                    if (!backends_[cpuIdx]->supportsNode(graph_, graph_.nodes[ni], DType::Float32))
                    {
                        cpuOk = false;
                        break;
                    }
                }
                if (!cpuOk)
                {
                    continue;
                }
                for (int ni: runNodes[r])
                {
                    nodeBackendIdx_[ni] = cpuIdx;
                    if (backends_[cpuIdx]->kind() != cfg_.backend)
                    {
                        bucket.fallbackReasons.push_back({graph_.nodes[ni].name, opTypeName(graph_.nodes[ni].type), "tiny GPU island folded to CPU (round-trip cost exceeds the work)"});
                    }
                }
                changed = true; // restart: the fold may have merged neighbours into a new island
            }
        }
    }

    void Session::ensureBackends() {
        if (!backends_.empty())
        {
            return; // instantiated once per session; every bucket plans over the same backends
        }
        // --- instantiate backends in priority order: primary, fallbacks..., CPU last ---
        std::vector<BackendKind> order;
        order.push_back(cfg_.backend);
        for (auto k: cfg_.fallback)
        {
            order.push_back(k);
        }
        if (cfg_.allowCpuFallback)
        {
            order.push_back(BackendKind::Cpu);
        }
        std::set<BackendKind> seen;
        auto                 &reg = BackendRegistry::instance();
        for (BackendKind k: order)
        {
            if (seen.count(k))
            {
                continue;
            }
            seen.insert(k);
            if (!reg.has(k))
            {
                VKNN_DEBUG << "backend " << backendName(k) << " not registered";
                continue;
            }
            auto b = reg.create(k, cfg_);
            if (!b || !b->available())
            {
                VKNN_WARN << "backend " << backendName(k) << " unavailable; skipping";
                continue;
            }
            byKind_[k] = b.get();
            backends_.push_back(std::move(b));
        }
        if (backends_.empty())
        {
            throw Error(Status::RuntimeError, "no usable backend");
        }
        for (auto &b: backends_)
        {
            b->configure(cfg_); // apply Config (e.g. disableVkOps) before capability queries
        }
        VKNN_INFO << "Active backends (priority): " << [&] {
            std::string s;
            for (auto &b: backends_)
            {
                s += std::string(b->name()) + " ";
            }
            return s;
        }();
    }

    PlanBucket Session::buildBucket(Graph &&g, const std::string &key) {
        PlanBucket bucket;
        bucket.key   = key;
        bucket.graph = std::make_unique<Graph>(std::move(g));
        // Alias the bucket's members under the names the body below was written against, so the plan
        // logic reads exactly as it did when it lived in plan() over the session members.
        Graph                                 &graph_          = *bucket.graph;
        std::vector<int>                      &nodeBackendIdx_ = bucket.nodeBackendIdx;
        std::vector<RtTensor>                 &pool_           = bucket.pool;
        std::vector<std::unique_ptr<Segment>> &segments_       = bucket.segments;

        // --- graph optimization passes (NCHW IR; batch=1 fallback + any Config::inputShapes/dimBindings) ---
        // Skipped when the graph came from a .vxm (passes already applied at save time). The default
        // fusion set is used (the compiler tool sets fusion options explicitly); the ONNX-load path only
        // needs the caller's declared input shapes and symbolic-dim bindings threaded in so a dynamic model
        // resolves instead of silently freezing to a 1x1 plan.
        if (!graphOptimized_)
        {
            PassOptions opt;
            opt.inputShapes = cfg_.inputShapes;
            opt.dimBindings = cfg_.dimBindings;
            runStandardPasses(graph_, opt);
        }
        graph_.topoSort();

        // Selective fp32 storage set. Precision::Normal ("normal") uses the built-in geometry-tail
        // preset when fp32Tensors is empty; an explicit fp32Tensors always wins. Resolved before the
        // view fold: a chain tensor markFp32 would pin must keep its materialized form.
        const bool  vulkanFlat = byKind_.count(BackendKind::Vulkan) && cfg_.flatLayout();
        std::string fp32Marks  = cfg_.fp32Tensors;
        if (fp32Marks.empty() && cfg_.precision == Precision::Normal)
        {
            fp32Marks = mixedPrecisionFp32Tensors();
        }

        // --- RoPE chain fusion: collapse each rotate-half chain (last-axis half Slices, cos/sin
        //     table Gathers, the rotate products, Concat) into ONE Rope node reading the tables by
        //     position, so an LLM decode step spends one dispatch per q/k rope site instead of ~7.
        //     Runs BEFORE the MatMul view fold: the two passes claim disjoint node kinds (the view
        //     fold absorbs Transpose/Expand movers, never Slice/Gather/Concat), and a MatMul chain
        //     walk that stopped at the Concat output stops at the same tensor now produced by Rope.
        //     Load-time only (never serialized); the fp32 pins apply only where markFp32 runs (the
        //     Vulkan fp16-storage path below).
        if (cfg_.ropeFusion())
        {
            fuseRope(graph_, vulkanFlat ? fp32Marks : std::string());
        }

        // --- MatMul operand-view fold: absorb Transpose/Expand chains feeding non-tiled MatMuls into
        //     per-axis stride attrs (core/matmul_view.h), so a GQA decode reads its KV cache in place
        //     instead of materializing the repeat_kv broadcast and attention transposes every token.
        //     Load-time only (never serialized), bit-identical, honored by both backends. The fp32
        //     pins apply only where markFp32 runs (the Vulkan fp16-storage path below).
        if (cfg_.matmulViewFold())
        {
            foldMatMulViews(graph_, vulkanFlat ? fp32Marks : std::string());
        }

        // --- decode-attention fusion: collapse the M == 1 MatMul -> scale/mask -> Softmax -> MatMul
        //     (-> Transpose -> Reshape) chain into one FusedAttention node, consuming the operand-view
        //     strides the fold above composed, so a decode step's attention core is one dispatch per
        //     layer and the score/probability intermediates never touch memory. Load-time only (never
        //     serialized); numerics-changing (fp32 scores + softmax replace the decomposed chain's
        //     fp16 round-trips), so it has its own hint and cache-variant key entry.
        if (cfg_.fusedAttention())
        {
            fuseDecodeAttention(graph_, vulkanFlat ? fp32Marks : std::string());
        }

        // --- Vulkan flat-layout pass: route the generic head ops (Transpose/Slice/Concat/Binary/Softmax)
        //     through flat row-major GPU buffers, inserting ConvertLayout at NC4HW4 boundaries, so the
        //     whole graph runs on the GPU. Must run before the pool + backend assignment (it adds nodes).
        if (vulkanFlat)
        {
            insertLayoutConverts(graph_);
            // Integer index tensors (token ids / positions) must survive to the GPU without an fp16 store
            // that would overflow a value above 65504 to +inf. Pin the Gather index chains to fp32 before
            // markFp32 so the buffer planner sizes them 4-byte and their producers run in fp32.
            pinGatherIndexFp32(graph_);
            // GridSample grids hold normalized sampling coordinates whose fp16 storage quantization
            // drifts the sample point (~0.5 px at 1920-wide inputs). Pin runtime grid chains to fp32
            // the same way; the GridSample shader decodes the grid at its storage precision.
            pinGridSampleGridFp32(graph_);
            // Only a caller-supplied fp32Tensors list takes zero-match accounting; the built-in
            // Precision::Normal preset is engine-owned and exempt from the load-end warning.
            markFp32(graph_, fp32Marks, cfg_.fp32Tensors.empty() ? nullptr : &matchedFp32Patterns_);
            fp32PatternsAccounted_ = fp32PatternsAccounted_ || !cfg_.fp32Tensors.empty();
            graph_.topoSort();
        }

        // --- init tensor pool, load initializers ---
        pool_.resize(graph_.tensors.size());
        for (size_t i = 0; i < pool_.size(); ++i)
        {
            pool_[i].id    = (TensorId) i;
            pool_[i].shape = graph_.tensors[i].shape;
            pool_[i].dtype = graph_.tensors[i].dtype;
        }
        // Initializers are loaded into the pool LATER (after backend assignment), and only the ones a CPU
        // op consumes — GPU ops upload their weights directly from graph_.initializers (the boundary pack
        // skips initializers). This avoids re-materializing every weight in the pool, and lets us free
        // the graph weights after upload — essential for fitting a 965M-param fp16 model on-device.

        // --- per-node backend assignment (highest-priority backend that supports it) ---
        // The backends are configured once in ensureBackends(); each bucket only assigns nodes.
        nodeBackendIdx_.assign(graph_.nodes.size(), -1);
        for (size_t n = 0; n < graph_.nodes.size(); ++n)
        {
            const Node &nd     = graph_.nodes[n];
            DType       dt     = DType::Float32; // compute dtype at IR level
            int         chosen = -1;
            for (size_t bi = 0; bi < backends_.size(); ++bi)
            {
                if (backends_[bi]->supportsNode(graph_, nd, dt))
                {
                    chosen = (int) bi;
                    break;
                }
            }
            if (chosen < 0)
            {
                throw Error(Status::Unsupported, std::string("no backend supports op ") + opTypeName(nd.type) + " (" + nd.name + ")");
            }
            nodeBackendIdx_[n] = chosen;
            // warn if the primary backend couldn't take it; keep the refusal reason for
            // fallbackReasons() (the support report and fallback diagnostics)
            if (backends_[chosen]->kind() != cfg_.backend && byKind_.count(cfg_.backend))
            {
                std::string why;
                if (!byKind_[cfg_.backend]->supportsNode(graph_, nd, dt, &why))
                {
                    bucket.fallbackReasons.push_back({nd.name, opTypeName(nd.type), why});
                    VKNN_WARN_THROTTLE(std::string("fallback_") + opTypeName(nd.type), 2) << "op " << opTypeName(nd.type) << " (" << nd.name << ") not supported by " << backendName(cfg_.backend) << " backend -> falling back to "
                                                                                          << backends_[chosen]->name() << " (" << why << "). Perf note: this op does not run on the requested backend.";
                }
            }
        }

        // --- fold tiny GPU "islands" back to CPU ---
        // A maximal run of GPU nodes that is fed only by CPU output and consumed only by CPU (a true
        // island in the dataflow) costs a CPU->GPU->CPU round trip. When that island does little work (a
        // detection head's DFL conv / sigmoid on tiny tensors) the pack/unpack dwarfs the compute and,
        // worse, stresses the boundary path. Run it on the CPU instead. The heavy backbone/head convs are
        // kept on the GPU — they exceed the work threshold, so this never drags real compute off the
        // accelerator.
        if (cfg_.gpuIslandFold())
        {
            foldTinyGpuIslands(bucket);
        }

        // Zero-match accounting for the dumpTensors/disableVkOps debug knobs, against this bucket's
        // final graph and backend assignment (post island fold). Aggregated across buckets;
        // warnUnmatchedDebugPatterns names the entries that matched nowhere once the whole model is
        // built. Reads only — the plan is untouched.
        accountDebugPatternMatches(bucket);

        // --- materialize int4-quantized weights (vknn_compile -Os) for non-native consumers ---
        // The Vulkan MatMul dequantizes the packed payload in-kernel; every other consumer — a
        // CPU-assigned node, or a GPU op without an int4 kernel (Conv/Gemm) — gets its fp16 bytes
        // reconstructed here, before the pool load and any GPU op prepare, so quantization is
        // invisible to it. Must run after the island fold: that pass reassigns nodes to the CPU,
        // and a reassigned MatMul needs its weight materialized like any other CPU node.
        materializeInt4Weights(graph_, [&](size_t n, const Node &nd) {
            return nd.type == OpType::MatMul && nodeBackendIdx_[n] >= 0 && backends_[nodeBackendIdx_[n]]->kind() == BackendKind::Vulkan;
        });

        // --- load CPU-consumed initializers into the pool (fp16 -> fp32 decode) ---
        // Only weights a CPU-assigned node reads need a host copy; GPU ops upload from
        // graph_.initializers. Every CPU op reads pool payloads through host.f32()/i64() with no
        // per-site dtype handling, so this load is the one place the stored dtype is honored: a
        // Float16 payload decodes to fp32 here (via initFloats, which also recovers a rank-0
        // scalar's element from the payload size), and any other dtype copies through as-is.
        {
            std::set<TensorId> cpuNeeded;
            auto               need = [&](TensorId t) {
                if (t != kNoTensor && graph_.isInitializer(t))
                {
                    cpuNeeded.insert(t);
                }
            };
            for (size_t n = 0; n < graph_.nodes.size(); ++n)
            {
                bool isCpu = nodeBackendIdx_[n] >= 0 && backends_[nodeBackendIdx_[n]]->kind() == BackendKind::Cpu;
                if (!isCpu)
                {
                    continue;
                }
                for (TensorId in: graph_.nodes[n].inputs)
                {
                    need(in);
                }
                // Fusion edges reference tensors outside node.inputs; a legacy .vxm may point them
                // at an initializer, which the op reads from the pool like any other operand.
                need(graph_.nodes[n].fusedResidual);
                need(graph_.nodes[n].fusedBias);
            }
            for (TensorId id: graph_.outputs)
            {
                need(id);
            }
            for (TensorId id: cpuNeeded)
            {
                RtTensor &rt = pool_[id];
                rt.shape     = graph_.tensors[id].shape;
                if (graph_.tensors[id].dtype == DType::Float16)
                {
                    std::vector<float> f = initFloats(graph_, id);
                    rt.host.bytes.resize(f.size() * 4);
                    std::memcpy(rt.host.bytes.data(), f.data(), f.size() * 4);
                    rt.dtype = DType::Float32;
                } else
                {
                    rt.host  = graph_.initializers[id];
                    rt.dtype = graph_.tensors[id].dtype;
                }
                rt.hostValid = true;
            }
        }

        // --- partition into maximal same-backend segments ---
        std::vector<std::vector<int>> parts;
        for (size_t n = 0; n < graph_.nodes.size(); ++n)
        {
            if (parts.empty() || nodeBackendIdx_[n] != nodeBackendIdx_[parts.back().front()])
            {
                parts.push_back({});
            }
            parts.back().push_back((int) n);
        }

        // boundary-set computation: producer map
        std::vector<int> producerSeg(graph_.tensors.size(), -1);
        std::vector<int> nodeToSeg(graph_.nodes.size(), -1);
        for (size_t p = 0; p < parts.size(); ++p)
        {
            for (int ni: parts[p])
            {
                nodeToSeg[ni] = (int) p;
                for (TensorId o: graph_.nodes[ni].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producerSeg[o] = (int) p;
                    }
                }
            }
        }
        std::set<TensorId> graphOutputs(graph_.outputs.begin(), graph_.outputs.end());

        for (size_t p = 0; p < parts.size(); ++p)
        {
            int                bi  = nodeBackendIdx_[parts[p].front()];
            auto               seg = backends_[bi]->compileSegment(parts[p], graph_, cfg_);
            std::set<TensorId> ins, outs;
            std::set<TensorId> internalOut;
            for (int ni: parts[p])
            {
                for (TensorId o: graph_.nodes[ni].outputs)
                {
                    internalOut.insert(o);
                }
            }
            for (int ni: parts[p])
            {
                for (TensorId in: graph_.nodes[ni].inputs)
                {
                    if (in == kNoTensor)
                    {
                        continue;
                    }
                    if (!internalOut.count(in))
                    {
                        ins.insert(in); // produced outside (init/input/other seg)
                    }
                }
                // a fused residual read by this op but produced by ANOTHER segment is also a boundary input.
                TensorId res = graph_.nodes[ni].fusedResidual;
                if (res != kNoTensor && !graph_.isInitializer(res) && !internalOut.count(res))
                {
                    ins.insert(res);
                }
                for (TensorId o: graph_.nodes[ni].outputs)
                {
                    if (o == kNoTensor)
                    {
                        continue;
                    }
                    // consumed outside this segment?
                    bool external = graphOutputs.count(o) > 0;
                    if (!external)
                    {
                        for (size_t q = 0; q < graph_.nodes.size() && !external; ++q)
                        {
                            if (nodeToSeg[q] != (int) p)
                            {
                                for (TensorId x: graph_.nodes[q].inputs)
                                {
                                    if (x == o)
                                    {
                                        external = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (external)
                    {
                        outs.insert(o);
                    }
                }
            }
            seg->boundaryInputs.assign(ins.begin(), ins.end());
            seg->boundaryOutputs.assign(outs.begin(), outs.end());
            // tag a CPU segment as a fallback when the configured primary backend isn't CPU.
            if (backends_[bi]->kind() == BackendKind::Cpu && cfg_.backend != BackendKind::Cpu)
            {
                seg->isFallback = true;
            }
            segments_.push_back(std::move(seg));
        }
        // GPU-side image I/O conversion is safe only when the WHOLE graph runs on a single non-CPU backend:
        // a CPU segment consuming a graph input needs the fp32 host copy that bindInput would otherwise
        // produce. When enabled, 8-bit image graph-inputs are uploaded raw and converted on the GPU (see the
        // boundary_convert staging path in the Vulkan backend), skipping the host uint8->fp32->fp16 pack.
        bucket.ioGpuConvert = cfg_.backend != BackendKind::Cpu;
        for (const auto &seg: segments_)
        {
            if (seg->backend && seg->backend->kind() == BackendKind::Cpu)
            {
                bucket.ioGpuConvert = false;
                break;
            }
        }
        for (const auto &seg: segments_)
        {
            seg->ioGpuConvert = bucket.ioGpuConvert;
        }
        // The pipeline/weight/tuning caches are flushed once at teardown (Session::updateCache, from the
        // destructor), not here, so any autotune/pipeline results land in the unified cache file.

        // --- free the host weights: GPU ops have uploaded them to the device, CPU-consumed ones were
        //     decoded into the pool above. Reclaims the full weight blob (a 965M fp16 model: ~1.9GB) so
        //     only the GPU buffers + activations remain resident. Gated by a config flag so callers that
        //     need weights resident (re-plan, weight introspection) can opt out.
        if (cfg_.freeWeightsAfterUpload)
        {
            // Free the bulk weights — Conv/MatMul/Gemm operands, which their ops upload + cache at compile.
            // KEEP the remaining (small) constants: some ops read their initializers while recording the
            // command buffer, which the zero-copy path re-records, so those initializers must stay
            // resolvable. Keeping them costs little (KB-scale shapes/biases/tables).
            // A fused pointwise-chain epilogue uploads its operands (inputs[pw_opbase..]) lazily while
            // RECORDING, not at prepare — e.g. a PRelu slope folded into a Conv. Those initializers must
            // stay resolvable, so keep any tensor used as an epilogue operand anywhere.
            std::set<TensorId> keepAtRecord;
            for (const auto &nd: graph_.nodes)
            {
                if (!nd.attr.has("pw_steps"))
                {
                    continue;
                }
                int opbase = (int) nd.attr.geti("pw_opbase", (int64_t) nd.inputs.size());
                for (int k = opbase; k < (int) nd.inputs.size(); ++k)
                {
                    if (nd.inputs[k] != kNoTensor)
                    {
                        keepAtRecord.insert(nd.inputs[k]);
                    }
                }
            }
            std::set<TensorId> freeable;
            for (const auto &nd: graph_.nodes)
            {
                if (nd.type == OpType::Conv || nd.type == OpType::MatMul || nd.type == OpType::Gemm || nd.type == OpType::ConvGemm)
                {
                    for (TensorId in: nd.inputs)
                    {
                        if (in != kNoTensor && graph_.isInitializer(in) && !keepAtRecord.count(in))
                        {
                            freeable.insert(in);
                        }
                    }
                }
            }
            size_t freed = 0;
            for (TensorId id: freeable)
            {
                auto it = graph_.initializers.find(id);
                if (it != graph_.initializers.end())
                {
                    freed += it->second.bytes.size();
                    graph_.initializers.erase(it);
                }
            }
            VKNN_INFO << "freed " << freed / (1024 * 1024) << " MB of host weights after upload";
        }

        // Bind-time index bounds: which graph inputs index a Gather, and how many rows it holds.
        bucket.gatherIndexAxisSizes = collectGatherIndexAxisSizes(graph_);

        VKNN_INFO << "Planned " << segments_.size() << " segment(s) over " << graph_.nodes.size() << " nodes [bucket '" << key << "']";
        // One un-throttled fallback summary per bucket: the per-node warnings above are throttled to
        // 2 per op type, so a model with 50 CPU GridSamples prints 2 warnings — without this line a
        // user can believe the whole graph runs on the requested backend while a divergent (or slow)
        // CPU implementation quietly handles part of it. Counts per op type plus the first few node
        // names; the full list stays available via Session::fallbackReasons() and
        // `vknn_compile --support-report`.
        if (!bucket.fallbackReasons.empty() && cfg_.backend != BackendKind::Cpu)
        {
            std::map<std::string, int> perOp;
            for (const FallbackReason &fr: bucket.fallbackReasons)
            {
                ++perOp[fr.op];
            }
            std::string ops;
            for (const auto &kv: perOp)
            {
                ops += (ops.empty() ? "" : ", ") + kv.first + " x" + std::to_string(kv.second);
            }
            std::string   first;
            constexpr int kNamedNodes = 3;
            for (int i = 0; i < kNamedNodes && i < (int) bucket.fallbackReasons.size(); ++i)
            {
                const FallbackReason &fr = bucket.fallbackReasons[(size_t) i];
                first += (first.empty() ? "" : "; ") + fr.node + " (" + fr.reason + ")";
            }
            VKNN_INFO << bucket.fallbackReasons.size() << " node(s) fall back to CPU [bucket '" << key << "']: " << ops << " -- first: " << first
                      << (bucket.fallbackReasons.size() > kNamedNodes ? " ... (full list: Session::fallbackReasons(), vknn_compile --support-report)" : "");
        }
        return bucket;
    }

    // disableVkOps entries, comma-split and whitespace-trimmed exactly as Config::listContains
    // treats them, so the zero-match warning enumerates the same entries the matcher accepts.
    static std::vector<std::string> splitTrimmedEntries(const std::string &list) {
        std::vector<std::string> entries;
        size_t                   pos = 0;
        while (pos <= list.size())
        {
            size_t comma    = list.find(',', pos);
            size_t end      = comma == std::string::npos ? list.size() : comma;
            size_t tokBegin = pos;
            size_t tokEnd   = end;
            while (tokBegin < tokEnd && std::isspace((unsigned char) list[tokBegin]))
            {
                ++tokBegin;
            }
            while (tokEnd > tokBegin && std::isspace((unsigned char) list[tokEnd - 1]))
            {
                --tokEnd;
            }
            if (tokEnd > tokBegin)
            {
                entries.push_back(list.substr(tokBegin, tokEnd - tokBegin));
            }
            if (comma == std::string::npos)
            {
                break;
            }
            pos = comma + 1;
        }
        return entries;
    }

    void Session::accountDebugPatternMatches(const PlanBucket &bucket) {
        if (!byKind_.count(BackendKind::Vulkan))
        {
            return; // dumpTensors and disableVkOps act through the Vulkan backend only
        }
        const Graph &g = *bucket.graph;
        if (!cfg_.dumpTensors.empty())
        {
            const std::vector<std::string> entries = splitPatternList(cfg_.dumpTensors);
            // Mirrors the VulkanSegment activation set (vk_backend.cpp): the dump candidates are
            // the non-initializer tensors a Vulkan-assigned node reads or writes, the fused
            // residual edge included.
            auto accountTensorName = [&](TensorId tid) {
                if (tid == kNoTensor || g.isInitializer(tid))
                {
                    return;
                }
                const std::string &nm = g.desc(tid).name;
                if (nm.empty())
                {
                    return;
                }
                for (const std::string &e: entries)
                {
                    if (nm.find(e) != std::string::npos)
                    {
                        matchedDumpPatterns_.insert(e);
                    }
                }
            };
            for (size_t n = 0; n < g.nodes.size(); ++n)
            {
                if (backends_[(size_t) bucket.nodeBackendIdx[n]]->kind() != BackendKind::Vulkan)
                {
                    continue;
                }
                const Node &nd = g.nodes[n];
                for (TensorId in: nd.inputs)
                {
                    accountTensorName(in);
                }
                for (TensorId o: nd.outputs)
                {
                    accountTensorName(o);
                }
                accountTensorName(nd.fusedResidual);
            }
        }
        if (!cfg_.disableVkOps.empty())
        {
            for (const Node &nd: g.nodes)
            {
                presentOpNames_.insert(opTypeName(nd.type));
            }
        }
    }

    void Session::warnUnmatchedDebugPatterns() const {
        // One load-end sweep over the user-supplied pattern knobs: an entry that matched nothing in
        // ANY bucket names a knob that silently does nothing — the tensor it targeted was renamed
        // or fused/folded away by a pass, or the entry is a typo. Once per pattern per model load.
        if (fp32PatternsAccounted_)
        {
            for (const std::string &e: splitPatternList(cfg_.fp32Tensors))
            {
                if (!matchedFp32Patterns_.count(e))
                {
                    VKNN_WARN << "fp32Tensors pattern '" << e << "' matched no eligible tensor in any bucket -- the fp32 pin does nothing (tensor renamed or fused away, or a typo?)";
                }
            }
        }
        if (byKind_.count(BackendKind::Vulkan))
        {
            if (!cfg_.dumpTensors.empty())
            {
                for (const std::string &e: splitPatternList(cfg_.dumpTensors))
                {
                    if (!matchedDumpPatterns_.count(e))
                    {
                        VKNN_WARN << "dumpTensors pattern '" << e << "' matched no GPU tensor in any bucket -- nothing will be dumped for it (tensor renamed or fused away, or a typo?)";
                    }
                }
            }
            for (const std::string &e: splitTrimmedEntries(cfg_.disableVkOps))
            {
                if (!presentOpNames_.count(e))
                {
                    VKNN_WARN << "disableVkOps entry '" << e << "' matches no op in the model";
                }
            }
        }
    }

    bool Session::bucketsShareInputNames() const {
        // Recomputed only when the bucket count changes (prepareShapes appends; a .vxm load fills
        // once): the classification drives per-run dispatch and must not rescan name lists per token.
        if (uniformCheckedFor_ != buckets_.size())
        {
            uniformCheckedFor_ = buckets_.size();
            bucketsUniform_    = true;
            const Graph &g0    = *buckets_.front().graph;
            for (size_t b = 1; b < buckets_.size() && bucketsUniform_; ++b)
            {
                const Graph &g = *buckets_[b].graph;
                if (g.inputs.size() != g0.inputs.size())
                {
                    bucketsUniform_ = false;
                    break;
                }
                for (size_t i = 0; i < g.inputs.size(); ++i)
                {
                    if (g.desc(g.inputs[i]).name != g0.desc(g0.inputs[i]).name)
                    {
                        bucketsUniform_ = false;
                        break;
                    }
                }
            }
        }
        return bucketsUniform_;
    }

    // The canonical key for a shape assignment: each graph input's "name=DxDxD" joined by ';'. Buckets
    // with equal keys are the same plan; run() builds the same string from the bound input shapes.
    std::string Session::shapeKey(const Graph &graph) {
        std::string k;
        for (size_t i = 0; i < graph.inputs.size(); ++i)
        {
            const TensorDesc &d = graph.desc(graph.inputs[i]);
            if (i)
            {
                k += ';';
            }
            k += d.name;
            k += '=';
            k += shapeStr(d.shape);
        }
        return k;
    }

    // True when every named caller input is an input of `g`. run() requires this of a bucket before
    // comparing shape keys: buckets of a multi-graph .vxm carry disjoint input NAME sets, and a key
    // built over a graph that lacks the caller's names would adopt that graph's defaults for every
    // input and "match" a bucket the caller never addressed. Callers may bind fewer inputs than the
    // graph has (the rest adopt bucket shapes).
    //
    // `allowPositional` extends the forgiving single-input match (a sole caller entry binds a sole
    // graph input whatever its name) to selection. It is true for a HOMOGENEOUS session — every
    // bucket exposes the same input names, i.e. one graph at several shapes — preserving the legacy
    // misnamed-single-input contract there. A heterogeneous (multi-graph) session dispatches named
    // inputs strictly by name: with two single-input graphs of equal input shape, a positional match
    // would silently run whichever bucket loads first. An UNNAMED sole entry stays positional even
    // then (the caller expressed "the only input"; the shape picks the graph).
    static bool runInputsBind(const Graph &g, const std::vector<IOTensor> &inputs, bool allowPositional) {
        const bool singleBind = g.inputs.size() == 1 && inputs.size() == 1;
        for (const IOTensor &io: inputs)
        {
            if (io.name.empty())
            {
                continue; // an unnamed entry never disqualifies; the bind loop resolves it
            }
            bool found = false;
            for (TensorId in: g.inputs)
            {
                if (g.desc(in).name == io.name)
                {
                    found = true;
                    break;
                }
            }
            if (!found && !(singleBind && allowPositional))
            {
                return false;
            }
        }
        return true;
    }

    // True when at least one caller entry binds an input of `g`: by name, or positionally as the
    // sole unnamed entry against a single-input graph. Multi-graph selection requires this of a
    // candidate — a run binding nothing of a bucket must not match it through its own defaults.
    static bool runBindsAny(const Graph &g, const std::vector<IOTensor> &inputs) {
        if (g.inputs.size() == 1 && inputs.size() == 1 && inputs[0].name.empty())
        {
            return true;
        }
        for (const IOTensor &io: inputs)
        {
            if (io.name.empty())
            {
                continue;
            }
            for (TensorId in: g.inputs)
            {
                if (g.desc(in).name == io.name)
                {
                    return true;
                }
            }
        }
        return false;
    }

    std::string Session::runShapeKey(const Graph &g, const std::vector<IOTensor> &inputs, bool allowPositional) {
        // Resolve each of `g`'s inputs to the shape this run binds for it (an unbound or empty
        // caller shape adopts the graph's shape), then build the same key shapeKey() produces.
        // A single-input graph binds by position when the sole caller entry names no input — or, when
        // `allowPositional`, names one the graph does not have (the legacy forgiving match, kept for
        // homogeneous sessions; see runInputsBind).
        const bool  positional = g.inputs.size() == 1 && inputs.size() == 1 && (allowPositional || inputs[0].name.empty());
        const bool  singleBind = positional;
        std::string k;
        for (size_t i = 0; i < g.inputs.size(); ++i)
        {
            const TensorDesc &d     = g.desc(g.inputs[i]);
            Shape             shape = d.shape;
            for (const auto &io: inputs)
            {
                bool match = io.name == d.name || singleBind;
                if (match && !io.shape.empty())
                {
                    shape = io.shape;
                    break;
                }
            }
            if (i)
            {
                k += ';';
            }
            k += d.name;
            k += '=';
            k += shapeStr(shape);
        }
        return k;
    }

    Status Session::prepareShapes(const std::map<std::string, Shape> &shapes) {
        if (!hasImportedGraph_)
        {
            VKNN_ERROR << "prepareShapes: this session was loaded from a .vxm; its buckets are fixed at "
                          "compile time. Recompile with vknn_compile --bucket to add shapes.";
            return Status::Unsupported;
        }
        // Re-run the whole pipeline from the pristine imported graph at the declared shapes, then key
        // the bucket by the shapes the passes actually resolved. If that key already exists this is a
        // no-op (re-declaring a prepared shape is idempotent).
        Config saved     = cfg_;
        cfg_.inputShapes = shapes; // threaded into the passes by buildBucket via cfg_
        Graph       copy = importedGraph_;
        std::string k;
        try
        {
            PlanBucket b = buildBucket(std::move(copy), std::string());
            b.key        = shapeKey(*b.graph);
            k            = b.key;
            for (const auto &existing: buckets_)
            {
                if (existing.key == k)
                {
                    cfg_ = saved;
                    return Status::Ok; // already planned this shape
                }
            }
            buckets_.push_back(std::move(b));
        } catch (const Error &e)
        {
            cfg_ = saved;
            VKNN_ERROR << "prepareShapes: " << e.what();
            return e.status();
        }
        cfg_ = saved;
        VKNN_INFO << "prepareShapes: added bucket '" << k << "' (" << buckets_.size() << " total)";
        return Status::Ok;
    }

    std::vector<BackendKind> Session::nodeBackends() const {
        std::vector<BackendKind> v;
        for (int bi: buckets_.front().nodeBackendIdx)
        {
            v.push_back(bi >= 0 ? backends_[bi]->kind() : BackendKind::Cpu);
        }
        return v;
    }

    std::vector<std::string> Session::fallbackOps() const {
        std::vector<std::string> v;
        const PlanBucket        &b0 = buckets_.front();
        for (size_t n = 0; n < b0.nodeBackendIdx.size() && n < b0.graph->nodes.size(); ++n)
        {
            int bi = b0.nodeBackendIdx[n];
            if (bi >= 0 && backends_[bi]->kind() != cfg_.backend)
            {
                v.push_back(std::string(opTypeName(b0.graph->nodes[n].type)) + " " + b0.graph->nodes[n].name);
            }
        }
        return v;
    }

    const RtTensor *Session::tensor(const std::string &name) const {
        const PlanBucket &b0 = buckets_.front();
        TensorId          id = b0.graph->find(name);
        if (id == kNoTensor)
        {
            return nullptr;
        }
        return &b0.pool[id];
    }

    // ---- engine-resident output->input links ---------------------------------------------------

    namespace {
        bool idInList(const std::vector<TensorId> &v, TensorId id) {
            return std::find(v.begin(), v.end(), id) != v.end();
        }
        // Internal host-storage class of a boundary tensor: int64 for declared Int64 (shape/index
        // tensors), fp32 for everything else — the same rule bindInput/CPU ops follow.
        DType internalStorageDtype(DType declared) {
            return declared == DType::Int64 ? DType::Int64 : DType::Float32;
        }
        // Allocate + zero a linked tensor's host storage when it has never been bound or produced,
        // so the resident state starts as zeros (mirrors the GPU boundary buffers' zero init).
        void ensureResidentHostStorage(const Graph &g, TensorId id, RtTensor &rt) {
            if (rt.hostValid && !rt.host.bytes.empty())
            {
                return;
            }
            rt.shape      = rt.shape.empty() ? g.tensors[id].shape : rt.shape;
            rt.dtype      = internalStorageDtype(g.tensors[id].dtype);
            int64_t elems = rt.shape.empty() ? 1 : numElements(rt.shape);
            rt.host.resizeElems(elems, rt.dtype);
            std::memset(rt.host.bytes.data(), 0, rt.host.bytes.size());
            rt.hostValid = true;
        }
    } // namespace

    Status Session::validateLinkRanges(const Graph &g, TensorId outId, TensorId inId, const std::vector<LinkRange> &ranges) const {
        const int64_t srcElems = numElements(g.tensors[outId].shape);
        const int64_t dstElems = numElements(g.tensors[inId].shape);
        for (const LinkRange &r: ranges)
        {
            if (r.count <= 0 || r.sourceElem < 0 || r.destElem < 0 || r.sourceElem + r.count > srcElems || r.destElem + r.count > dstElems)
            {
                VKNN_ERROR << "link: range [src " << r.sourceElem << ", dst " << r.destElem << ", count " << r.count << ") exceeds '" << g.tensors[outId].name << "' (" << srcElems << " elems) or '"
                           << g.tensors[inId].name << "' (" << dstElems << " elems)";
                return Status::InvalidArgument;
            }
        }
        // Overlapping destination ranges would race on the GPU copy; reject them.
        std::vector<std::pair<int64_t, int64_t>> spans;
        spans.reserve(ranges.size());
        for (const LinkRange &r: ranges)
        {
            spans.emplace_back(r.destElem, r.destElem + r.count);
        }
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size(); ++i)
        {
            if (spans[i].first < spans[i - 1].second)
            {
                VKNN_ERROR << "link: destination ranges overlap at element " << spans[i].first << " of '" << g.tensors[inId].name << "'";
                return Status::InvalidArgument;
            }
        }
        return Status::Ok;
    }

    const Session::ResidentLink *Session::linkedOutput(size_t bucket, TensorId id) const {
        for (const ResidentLink &link: links_)
        {
            if (link.bucket == bucket && link.outId == id)
            {
                return &link;
            }
        }
        return nullptr;
    }

    const Session::ResidentLink *Session::linkedInput(size_t bucket, TensorId id) const {
        for (const ResidentLink &link: links_)
        {
            if (link.bucket == bucket && link.inId == id)
            {
                return &link;
            }
        }
        return nullptr;
    }

    Status Session::linkOutputToInput(const std::string &outputName, const std::string &inputName, const std::vector<LinkRange> &ranges) {
        // Resolve the unique bucket exposing both names; several matches need the bucket overload.
        size_t match = buckets_.size(), matches = 0;
        for (size_t b = 0; b < buckets_.size(); ++b)
        {
            const Graph &g   = *buckets_[b].graph;
            TensorId     out = g.find(outputName), in = g.find(inputName);
            if (out != kNoTensor && in != kNoTensor && idInList(g.outputs, out) && idInList(g.inputs, in))
            {
                match = b;
                ++matches;
            }
        }
        if (matches == 0)
        {
            VKNN_ERROR << "link: no bucket has output '" << outputName << "' and input '" << inputName << "'";
            return Status::NotFound;
        }
        if (matches > 1)
        {
            VKNN_ERROR << "link: output '" << outputName << "' and input '" << inputName << "' exist in " << matches << " buckets; use the bucket-explicit linkOutputToInput overload";
            return Status::InvalidArgument;
        }
        return linkOutputToInput(match, outputName, inputName, ranges);
    }

    Status Session::linkOutputToInput(size_t bucket, const std::string &outputName, const std::string &inputName, const std::vector<LinkRange> &ranges) {
        if (bucket >= buckets_.size())
        {
            VKNN_ERROR << "link: bucket " << bucket << " out of range (have " << buckets_.size() << ")";
            return Status::InvalidArgument;
        }
        PlanBucket    &b     = buckets_[bucket];
        const Graph   &g     = *b.graph;
        const TensorId outId = g.find(outputName);
        const TensorId inId  = g.find(inputName);
        if (outId == kNoTensor || !idInList(g.outputs, outId))
        {
            VKNN_ERROR << "link: '" << outputName << "' is not a graph output of bucket " << bucket;
            return Status::NotFound;
        }
        if (inId == kNoTensor || !idInList(g.inputs, inId))
        {
            VKNN_ERROR << "link: '" << inputName << "' is not a graph input of bucket " << bucket;
            return Status::NotFound;
        }
        if (outId == inId || idInList(g.inputs, outId) || idInList(g.outputs, inId))
        {
            VKNN_ERROR << "link: '" << outputName << "' -> '" << inputName << "' would alias a tensor that is both a graph input and output";
            return Status::InvalidArgument;
        }
        // The copies move raw storage; both sides must live in the same storage class.
        if (internalStorageDtype(g.tensors[outId].dtype) != internalStorageDtype(g.tensors[inId].dtype))
        {
            VKNN_ERROR << "link: dtype mismatch between output '" << outputName << "' (" << dtypeStr(g.tensors[outId].dtype) << ") and input '" << inputName << "' ("
                       << dtypeStr(g.tensors[inId].dtype) << ")";
            return Status::InvalidArgument;
        }
        if (Status rangeStatus = validateLinkRanges(g, outId, inId, ranges); rangeStatus != Status::Ok)
        {
            return rangeStatus;
        }
        // Re-linking an existing pair replaces the ranges (the per-token update path).
        for (ResidentLink &link: links_)
        {
            if (link.bucket == bucket && link.outputName == outputName && link.inputName == inputName)
            {
                link.ranges      = ranges;
                link.rangesDirty = true;
                return Status::Ok;
            }
        }
        // One input takes at most one source: a second link into the same input would make the copy
        // order (and any cross-link range overlap) undefined.
        for (const ResidentLink &existing: links_)
        {
            if (existing.bucket == bucket && existing.inId == inId)
            {
                VKNN_ERROR << "link: input '" << inputName << "' is already linked from output '" << existing.outputName << "'";
                return Status::InvalidArgument;
            }
        }
        ResidentLink link;
        link.bucket      = bucket;
        link.outputName  = outputName;
        link.inputName   = inputName;
        link.outId       = outId;
        link.inId        = inId;
        link.ranges      = ranges;
        link.rangesDirty = true;
        if (cfg_.backend != BackendKind::Cpu)
        {
            // Device path: one segment must both produce the output and consume the input, so the
            // resident copy stays entirely inside its pre-recorded command stream and the values it
            // reads are the previous run's. There is no silent host fallback on a GPU session — a
            // link that cannot be device-resident is an error the caller must see.
            Segment *owner = nullptr;
            for (const std::unique_ptr<Segment> &seg: b.segments)
            {
                if (idInList(seg->boundaryOutputs, outId) && idInList(seg->boundaryInputs, inId))
                {
                    owner = seg.get();
                    break;
                }
            }
            if (!owner)
            {
                VKNN_ERROR << "link: no single segment produces '" << outputName << "' and consumes '" << inputName << "' (a CPU-fallback island splits them); device-resident linking unavailable";
                return Status::Unsupported;
            }
            // Every reader of the linked input must live in that segment (an outside reader would
            // consume a stale host copy), and the linked output must have no readers outside it.
            std::set<int> ownNodes(owner->nodeIdx.begin(), owner->nodeIdx.end());
            for (size_t n = 0; n < g.nodes.size(); ++n)
            {
                if (ownNodes.count((int) n))
                {
                    continue;
                }
                for (TensorId in: g.nodes[n].inputs)
                {
                    if (in == inId || in == outId)
                    {
                        VKNN_ERROR << "link: '" << g.tensors[in].name << "' is read outside the owning GPU segment (node '" << g.nodes[n].name << "'); device-resident linking unavailable";
                        return Status::Unsupported;
                    }
                }
                if (g.nodes[n].fusedResidual == inId || g.nodes[n].fusedResidual == outId)
                {
                    VKNN_ERROR << "link: a linked tensor is read as a fused residual outside the owning GPU segment (node '" << g.nodes[n].name << "'); device-resident linking unavailable";
                    return Status::Unsupported;
                }
            }
            std::string whyNot;
            Status      addStatus = owner->addResidentLink(outId, inId, whyNot);
            if (addStatus != Status::Ok)
            {
                VKNN_ERROR << "link: '" << outputName << "' -> '" << inputName << "': " << (whyNot.empty() ? "backend has no device-resident path" : whyNot);
                return addStatus;
            }
            link.deviceSegment = owner;
        }
        links_.push_back(std::move(link));
        return Status::Ok;
    }

    Status Session::readResident(const std::string &name, IOTensor &out) {
        const ResidentLink *found       = nullptr;
        bool                isInputSide = false;
        for (const ResidentLink &link: links_)
        {
            bool matchIn = link.inputName == name, matchOut = link.outputName == name;
            if (!matchIn && !matchOut)
            {
                continue;
            }
            if (found)
            {
                VKNN_ERROR << "readResident: '" << name << "' is linked more than once; state is ambiguous";
                return Status::InvalidArgument;
            }
            found       = &link;
            isInputSide = matchIn;
        }
        if (!found)
        {
            VKNN_ERROR << "readResident: '" << name << "' is not a linked tensor";
            return Status::NotFound;
        }
        PlanBucket &b   = buckets_[found->bucket];
        TensorId    tid = isInputSide ? found->inId : found->outId;
        RtTensor   &rt  = b.pool[tid];
        if (found->deviceSegment)
        {
            rt.shape = rt.shape.empty() ? b.graph->tensors[tid].shape : rt.shape;
            if (!found->deviceSegment->downloadResident(tid, rt))
            {
                VKNN_ERROR << "readResident: '" << name << "' has no device residency to download";
                return Status::RuntimeError;
            }
        } else
        {
            ensureResidentHostStorage(*b.graph, tid, rt);
        }
        out.name  = name;
        out.shape = rt.shape.empty() ? b.graph->tensors[tid].shape : rt.shape;
        out.dtype = rt.dtype;
        out.data  = rt.host.bytes.toVector();
        return Status::Ok;
    }

    void Session::clearLinks() {
        for (ResidentLink &link: links_)
        {
            if (link.deviceSegment)
            {
                link.deviceSegment->clearResidentLinks();
            }
        }
        links_.clear();
    }

    // ---- engine-side output reductions -----------------------------------------------------------

    const Session::OutputArgMax *Session::argMaxOutput(size_t bucket, TensorId id) const {
        for (const OutputArgMax &reduction: argMaxOutputs_)
        {
            if (reduction.bucket == bucket && reduction.outId == id)
            {
                return &reduction;
            }
        }
        return nullptr;
    }

    Status Session::setOutputArgMax(size_t bucket, const std::string &outputName) {
        if (bucket >= buckets_.size())
        {
            VKNN_ERROR << "argmax: bucket " << bucket << " out of range (have " << buckets_.size() << ")";
            return Status::InvalidArgument;
        }
        PlanBucket    &b     = buckets_[bucket];
        const Graph   &g     = *b.graph;
        const TensorId outId = g.find(outputName);
        if (outId == kNoTensor || !idInList(g.outputs, outId))
        {
            VKNN_ERROR << "argmax: '" << outputName << "' is not a graph output of bucket " << bucket;
            return Status::NotFound;
        }
        if (g.tensors[outId].dtype != DType::Float32 && g.tensors[outId].dtype != DType::Float16)
        {
            VKNN_ERROR << "argmax: '" << outputName << "' is " << dtypeStr(g.tensors[outId].dtype) << "; only float outputs reduce";
            return Status::InvalidArgument;
        }
        // The reduction is over one flat vector: every leading dim must be 1 so the element index IS
        // the last-axis index (a decoder's logits row [1,1,V]). Anything else is ambiguous.
        const Shape  &shape = g.tensors[outId].shape;
        const int64_t elems = shape.empty() ? 0 : numElements(shape);
        if (elems <= 0 || elems != shape.back())
        {
            VKNN_ERROR << "argmax: '" << outputName << "' is not effectively one-dimensional";
            return Status::InvalidArgument;
        }
        if (argMaxOutput(bucket, outId))
        {
            return Status::Ok; // idempotent
        }
        OutputArgMax reduction;
        reduction.bucket     = bucket;
        reduction.outputName = outputName;
        reduction.outId      = outId;
        if (cfg_.backend != BackendKind::Cpu)
        {
            for (const std::unique_ptr<Segment> &seg: b.segments)
            {
                if (!idInList(seg->boundaryOutputs, outId))
                {
                    continue;
                }
                std::string  whyNot;
                const Status st = seg->setOutputArgMax(outId, whyNot);
                if (st == Status::Ok)
                {
                    reduction.deviceSegment = seg.get();
                } else if (st != Status::Unsupported)
                {
                    VKNN_ERROR << "argmax: '" << outputName << "': " << whyNot;
                    return st;
                }
                break;
            }
        }
        if (!reduction.deviceSegment)
        {
            VKNN_INFO << "argmax: '" << outputName << "' reduces on the host copy (no device reduction path)";
        }
        argMaxOutputs_.push_back(std::move(reduction));
        return Status::Ok;
    }

    Status Session::readOutputArgMax(const std::string &outputName, int64_t &index, float &value) {
        for (const OutputArgMax &reduction: argMaxOutputs_)
        {
            if (reduction.outputName != outputName)
            {
                continue;
            }
            if (reduction.deviceSegment && reduction.deviceSegment->readOutputArgMax(reduction.outId, index, value))
            {
                return Status::Ok;
            }
            // Host path: scan the internal fp32 copy with the shader's exact semantics — strictly
            // greater replaces, so the first occurrence of the maximum wins and NaN never does.
            PlanBucket &b  = buckets_[reduction.bucket];
            RtTensor   &rt = b.pool[reduction.outId];
            if (!rt.hostValid || rt.host.bytes.empty())
            {
                VKNN_ERROR << "argmax: '" << outputName << "' has no values to reduce (no completed run)";
                return Status::RuntimeError;
            }
            const float  *data  = reinterpret_cast<const float *>(rt.host.bytes.data());
            const int64_t elems = rt.shape.empty() ? (int64_t) (rt.host.bytes.size() / sizeof(float)) : numElements(rt.shape);
            float         best   = -std::numeric_limits<float>::infinity();
            int64_t       bestAt = -1;
            for (int64_t i = 0; i < elems; ++i)
            {
                if (data[i] > best)
                {
                    best   = data[i];
                    bestAt = i;
                }
            }
            index = bestAt < 0 ? 0 : bestAt;
            value = best;
            return Status::Ok;
        }
        VKNN_ERROR << "argmax: '" << outputName << "' was never registered (setOutputArgMax)";
        return Status::NotFound;
    }

    Status Session::applyResidentLinks(size_t bucketIndex, PlanBucket &bucket) {
        for (ResidentLink &link: links_)
        {
            if (link.bucket != bucketIndex)
            {
                continue;
            }
            RtTensor &src = bucket.pool[link.outId];
            RtTensor &dst = bucket.pool[link.inId];
            if (src.dmaBufFd >= 0 || dst.dmaBufFd >= 0)
            {
                VKNN_ERROR << "run: linked tensor '" << (src.dmaBufFd >= 0 ? link.outputName : link.inputName) << "' cannot also bind a dma-buf fd";
                return Status::InvalidArgument;
            }
            if (link.deviceSegment)
            {
                if (link.rangesDirty)
                {
                    link.deviceSegment->setResidentLinkRanges(link.outId, link.inId, link.ranges);
                    link.rangesDirty = false;
                }
                continue;
            }
            // Host path (CPU backend): the ranged copy moves the exact fp32/int64 storage bytes the
            // caller's own fold would have, so values are identical to the unlinked loop.
            const Graph &g = *bucket.graph;
            ensureResidentHostStorage(g, link.outId, src);
            ensureResidentHostStorage(g, link.inId, dst);
            if (src.dtype != dst.dtype)
            {
                VKNN_ERROR << "run: linked tensors '" << link.outputName << "' (" << dtypeStr(src.dtype) << ") and '" << link.inputName << "' (" << dtypeStr(dst.dtype) << ") hold different storage dtypes";
                return Status::InvalidArgument;
            }
            const size_t elemBytes = dtypeSize(src.dtype);
            for (const LinkRange &r: link.ranges)
            {
                std::memcpy(dst.host.bytes.data() + (size_t) r.destElem * elemBytes, src.host.bytes.data() + (size_t) r.sourceElem * elemBytes, (size_t) r.count * elemBytes);
            }
        }
        return Status::Ok;
    }

    // Buffer sizes, push constants, and dispatch geometry are frozen from a bucket's graph shapes at
    // plan() time, so a caller shape whose packed footprint differs from the bucket would overrun
    // (or misread) the mapped boundary buffer at pack time. Every run() input shape passes through
    // this single check once its bucket is selected; an empty caller shape adopts the bucket's shape
    // and always passes. An unresolved bucket shape (element count <= 0) accepts any caller shape,
    // mirroring the fully-dynamic escape in the vector<float> overload.
    // Every run() input shape passes through this single check once its bucket is selected; an empty
    // caller shape adopts the bucket's shape and always passes. An unresolved bucket shape (element
    // count <= 0) accepts any caller shape, mirroring the fully-dynamic escape in the vector<float>
    // overload. A DIFFERENT caller shape is judged by boundShapeCompatible (nchw.h): the loose
    // footprint-fit dynamic-reshape contract when only CPU segments consume the input (CPU ops
    // recompute geometry from the runtime shape — a session planned at x=[2,6] runs a [2,8] bind),
    // or byte-identical-packing when a GPU segment consumes it (its pack layout, push constants and
    // dispatch geometry are frozen from the planned shape, so a shape that packs differently — even
    // at an equal padded footprint — silently misreads).
    Status Session::validateInputShape(const PlanBucket &bucket, TensorId id, const Shape &got) const {
        const TensorDesc &d = bucket.graph->tensors[id];
        if (got.empty() || got == d.shape || numElements(d.shape) <= 0)
        {
            return Status::Ok;
        }
        bool gpuConsumed = false;
        for (const auto &seg: bucket.segments)
        {
            if (!seg->backend || seg->backend->kind() == BackendKind::Cpu)
            {
                continue;
            }
            for (TensorId t: seg->boundaryInputs)
            {
                if (t == id)
                {
                    gpuConsumed = true;
                    break;
                }
            }
            if (gpuConsumed)
            {
                break;
            }
        }
        if (!boundShapeCompatible(got, d.shape, d.gpuFlat, gpuConsumed))
        {
            VKNN_ERROR << "run: input '" << d.name << "' shape " << shapeStr(got) << " does not " << (gpuConsumed ? "pack like" : "fit") << " the planned shape " << shapeStr(d.shape) << (gpuConsumed ? " (a GPU-consumed input must bind the planned shape, or an N/C/spatial-product-preserving reshape of it)" : "");
            return Status::InvalidArgument;
        }
        return Status::Ok;
    }

    Status Session::run(const std::vector<IOTensor> &inputs, std::vector<IOTensor> &outputs) {
        const bool tm  = cfg_.timing;
        auto       now = [] {
            return std::chrono::high_resolution_clock::now();
        };
        auto tA = now();

        // --- select the plan bucket by the bound input shapes (the W0.1 shape choke point, now the
        //     bucket selector). A fixed-shape model has ONE bucket and takes the fast path below: no
        //     key is built, so the hot path allocates nothing new for a bucket lookup. A multi-bucket
        //     session builds the run's shape key and dispatches by exact match; an unknown shape lists
        //     the available shapes and is rejected with no compute.
        //
        //     Single-bucket runs still honor validateInputShape() (footprint fit), preserving the
        //     pre-bucket contract where a caller may bind a different-but-same-footprint shape (the CPU
        //     dynamic-reshape path). A multi-bucket session requires an exact per-shape plan instead.
        PlanBucket *sel = &buckets_.front();
        if (buckets_.size() > 1)
        {
            sel = nullptr;
            if (bucketsShareInputNames())
            {
                // Homogeneous multi-shape session (one graph at several shapes): the pre-multi-graph
                // semantics verbatim — ONE key built over the default bucket's inputs (an unbound
                // input adopts bucket 0's shape, a misnamed sole entry binds positionally), matched
                // exactly against each bucket's key. Evaluating the key per candidate instead would
                // let a partially-bound run adopt a CANDIDATE's shapes for its unbound inputs and
                // dispatch where the old code errored.
                const std::string wantKey = runShapeKey(*buckets_.front().graph, inputs, true);
                for (auto &b: buckets_)
                {
                    if (b.key == wantKey)
                    {
                        sel = &b;
                        break;
                    }
                }
            } else
            {
                // Multi-graph session: the key is evaluated over each CANDIDATE bucket's own graph
                // (the buckets have disjoint input names, so there is no single key for a run). A
                // bucket is eligible when every named caller input binds to it, it BINDS AT LEAST
                // ONE caller entry (an all-defaults key trivially equals the bucket's own key — a
                // run that binds nothing of a bucket is no claim on it), and the resolved shapes
                // reproduce its key. Named inputs dispatch strictly; an unnamed sole entry binds
                // positionally to a single-input graph, so the shape picks the graph.
                for (auto &b: buckets_)
                {
                    if (!runInputsBind(*b.graph, inputs, false) || !runBindsAny(*b.graph, inputs))
                    {
                        continue;
                    }
                    if (b.key == runShapeKey(*b.graph, inputs, false))
                    {
                        sel = &b;
                        break;
                    }
                }
            }
            if (!sel)
            {
                std::string bound;
                for (const auto &io: inputs)
                {
                    bound += (bound.empty() ? "" : ";") + io.name + "=" + shapeStr(io.shape);
                }
                std::string avail;
                for (size_t i = 0; i < buckets_.size(); ++i)
                {
                    avail += (i ? ", " : "") + buckets_[i].key;
                }
                VKNN_ERROR << "run: bound inputs (" << bound << ") match no compiled bucket (by name and shape); available: " << avail << ". Add the shape with prepareShapes() (ONNX session) or recompile with --bucket/--graph.";
                return Status::InvalidArgument;
            }
        }
        PlanBucket  &bucket      = *sel;
        const size_t bucketIndex = (size_t) (sel - buckets_.data());
        // Alias the selected bucket's state under the names the body below was written against; run()
        // is otherwise unchanged, so a fixed-shape model's single bucket runs exactly as before.
        Graph                                 &graph_        = *bucket.graph;
        std::vector<RtTensor>                 &pool_         = bucket.pool;
        std::vector<std::unique_ptr<Segment>> &segments_     = bucket.segments;
        const bool                             ioGpuConvert_ = bucket.ioGpuConvert;

        ExecContext ctx;
        ctx.pool     = &pool_;
        ctx.graph    = &graph_;
        ctx.config   = &cfg_;
        ctx.profiler = &profiler_;
        profiler_.clear();

        // --- bind inputs (host data, or a zero-copy dma-buf fd) ---
        // Clear any prior zero-copy binding on every input first, so a binding declared in an earlier
        // run can't linger on an input the caller omits this run.
        for (TensorId iid: graph_.inputs)
        {
            pool_[iid].dmaBufFd     = -1;
            pool_[iid].dmaBufFormat = TensorFormat::NCHW;
            pool_[iid].dmaBufDtype  = DType::Float32;
        }
        for (const auto &io: inputs)
        {
            TensorId id = graph_.find(io.name);
            if (id == kNoTensor)
            {
                // fall back to the single graph input
                if (graph_.inputs.size() == 1)
                {
                    id = graph_.inputs[0];
                } else
                {
                    VKNN_ERROR << "input not found: " << io.name;
                    return Status::InvalidArgument;
                }
            }
            RtTensor &rt = pool_[id];
            if (Status vs = validateInputShape(bucket, id, io.shape); vs != Status::Ok)
            {
                return vs;
            }
            rt.shape        = io.shape.empty() ? graph_.tensors[id].shape : io.shape;
            rt.dmaBufFd     = io.dmaBufFd;
            rt.dmaBufFormat = io.dmaBufFormat;
            rt.dmaBufDtype  = io.dmaBufDtype;
            if (io.dmaBufFd >= 0)
            {
                rt.dtype     = io.dtype;
                rt.hostValid = false; // zero-copy: the input comes straight from the fd, no host buffer
            } else if (ioGpuConvert_ && (io.dtype == DType::UInt8 || io.dtype == DType::Int8) && !linkedInput(bucketIndex, id))
            {
                // (A LINKED input takes the fp32 bindInput path below even for 8-bit data: the raw-
                // byte staging convert re-runs every submit and would overwrite the resident state.)
                // Whole-graph GPU run: keep the caller's raw 8-bit bytes (rt.dtype stays the declared 8-bit
                // type) and let the GPU convert them at the boundary — uint8/int8 -> device fp16 + NC4HW4
                // gather — skipping the host uint8->fp32->fp16 pack. The Vulkan backend recognizes the 8-bit
                // rt.dtype, memcpys the raw NCHW bytes into a staging buffer, and dispatches boundary_convert.
                rt.dtype      = io.dtype;
                rt.host.bytes = io.data;
                rt.hostValid  = true;
            } else
            {
                // Convert the caller's bytes (in io.dtype) to the internal compute storage: fp32 for every
                // real/8-bit/32-bit type (the compute path is fp32), int64 for Int64 (shape/index inputs).
                // A UINT8/FLOAT16 image thus enters as fp32 and the model's own Cast handles the rest.
                // A rank-0 scalar input (shape [], numElements() 0) carries its one element, so the host
                // buffer is non-empty and a CPU op reading the operand does not dereference a null pointer.
                int64_t elems = rt.shape.empty() ? 1 : numElements(rt.shape);
                bindInput(io.dtype, io.data, elems, rt);
                rt.hostValid = true;
            }
            rt.deviceValid = false;
            // An Int64 input that indexes a Gather (an embedding lookup's token ids) is validated
            // here against the smallest axis size collected at plan time, so a caller-supplied
            // out-of-range id fails with its value and position before any op reads past the table.
            // ONNX index semantics admit [-axisSize, axisSize): negatives wrap Python-style.
            if (rt.hostValid && rt.dtype == DType::Int64)
            {
                for (const auto &indexLimit: bucket.gatherIndexAxisSizes)
                {
                    if (indexLimit.first != id)
                    {
                        continue;
                    }
                    const int64_t  axisSize = indexLimit.second;
                    const int64_t  elems    = rt.shape.empty() ? 1 : numElements(rt.shape);
                    const int64_t *values   = rt.host.i64();
                    for (int64_t e = 0; e < elems; ++e)
                    {
                        if (values[e] < -axisSize || values[e] >= axisSize)
                        {
                            VKNN_ERROR << "run: input '" << graph_.tensors[id].name << "' element " << e << " = " << values[e] << " is out of range [" << -axisSize << ", " << axisSize << ") for the " << axisSize << "-row Gather axis it indexes";
                            return Status::InvalidArgument;
                        }
                    }
                    break;
                }
            }
        }
        // Read zero-copy output fd bindings from the incoming `outputs` (set before the segments run so
        // the GPU writes into them), before it is cleared and refilled with results below.
        for (TensorId oid: graph_.outputs)
        {
            pool_[oid].dmaBufFd     = -1;
            pool_[oid].dmaBufFormat = TensorFormat::NCHW;
            pool_[oid].dmaBufDtype  = DType::Float32;
        }
        for (const auto &b: outputs)
        {
            if (b.dmaBufFd < 0)
            {
                continue;
            }
            TensorId id = graph_.find(b.name);
            if (id == kNoTensor && graph_.outputs.size() == 1)
            {
                id = graph_.outputs[0];
            }
            if (id != kNoTensor)
            {
                pool_[id].dmaBufFd     = b.dmaBufFd;
                pool_[id].dmaBufFormat = b.dmaBufFormat;
                pool_[id].dmaBufDtype  = b.dmaBufDtype;
            }
        }
        // Reclaim the byte storage the previous run donated to the caller. readbackOutput() moves an
        // output tensor's host bytes into the caller's IOTensor, leaving that tensor's host residency with
        // no allocation; `outputs` carries those buffers back in and is cleared below regardless. Taking
        // them back here — before any segment writes an output — lets the download's resizeElems() land on
        // the byte size already held, so a steady-state loop allocates nothing and zeroes nothing per run.
        // Entries match positionally against graph_.outputs and are rejected on a name mismatch, so a
        // caller that reorders or substitutes the vector merely forgoes the reuse. A tensor whose host
        // residency is live (a graph input also declared an output, a dtype-conversion buffer) keeps what
        // it holds, and a zero-copy output has no host residency to refill.
        for (size_t i = 0, no = std::min(outputs.size(), graph_.outputs.size()); i < no; ++i)
        {
            TensorId  oid = graph_.outputs[i];
            RtTensor &rt  = pool_[oid];
            if (outputs[i].data.empty() || rt.dmaBufFd >= 0 || rt.hostValid || !rt.host.bytes.empty())
            {
                continue;
            }
            if (outputs[i].name != graph_.tensors[oid].name)
            {
                continue;
            }
            rt.host.bytes = std::move(outputs[i].data);
        }

        // --- engine-resident links: push updated copy ranges to the owning GPU segment, or apply
        //     the host-path ranged copies (CPU backend), before anything reads the linked inputs.
        if (!links_.empty())
        {
            if (Status linkStatus = applyResidentLinks(bucketIndex, bucket); linkStatus != Status::Ok)
            {
                return linkStatus;
            }
        }

        auto tB = now();
        // --- run segments in order ---
        try
        {
            bool dbg = cfg_.debugSegments;
            for (size_t si = 0; si < segments_.size(); ++si)
            {
                if (dbg)
                {
                    VKNN_INFO << "RUN segment " << si << "/" << segments_.size() << " backend=" << segments_[si]->backend->name();
                }
                segments_[si]->run(ctx);
            }
        } catch (const std::exception &e)
        {
            VKNN_ERROR << "run failed: " << e.what();
            return Status::RuntimeError;
        }
        auto tC = now();

        // --- layer dump ---
        if (cfg_.layerDump)
        {
            ::mkdir(cfg_.layerDumpDir.c_str(), 0755);
            for (size_t i = 0; i < pool_.size(); ++i)
            {
                RtTensor &rt = pool_[i];
                if (!rt.hostValid || graph_.isInitializer((TensorId) i))
                {
                    continue;
                }
                std::string nm = graph_.tensors[i].name;
                for (char &c: nm)
                {
                    if (c == '/' || c == ':')
                    {
                        c = '_';
                    }
                }
                std::ofstream f(cfg_.layerDumpDir + "/" + nm + ".bin", std::ios::binary);
                if (f)
                {
                    f.write((const char *) rt.host.bytes.data(), rt.host.bytes.size());
                }
            }
            VKNN_INFO << "layer dump written to " << cfg_.layerDumpDir;
        }

        // --- collect outputs ---
        outputs.clear();
        for (TensorId oid: graph_.outputs)
        {
            RtTensor &rt = pool_[oid];
            IOTensor  io;
            io.name         = graph_.tensors[oid].name;
            io.shape        = rt.shape;
            io.dmaBufFd     = rt.dmaBufFd;
            io.dmaBufFormat = rt.dmaBufFormat;
            io.dmaBufDtype  = rt.dmaBufDtype;
            if (linkedOutput(bucketIndex, oid))
            {
                // A linked output stays engine-resident: the entry carries its metadata but no data
                // (readResident() fetches the values when a caller needs them).
                io.dtype = graph_.tensors[oid].dtype;
            } else if (argMaxOutput(bucketIndex, oid))
            {
                // An argmax-registered output is served by readOutputArgMax(): metadata only, no
                // declared-dtype readback (the device path never downloads the vector at all).
                io.dtype = graph_.tensors[oid].dtype;
            } else if (rt.dmaBufFd < 0)
            {
                // Emit the output in the model's DECLARED dtype (e.g. a UINT8 image, FLOAT16 tensor),
                // converting from the internal fp32/int64 storage. Matches the ONNX output contract. A
                // rank-0 scalar output (shape [], numElements() 0) counts its one element so the dtype
                // conversion emits the value rather than an empty buffer.
                int64_t outElems = rt.shape.empty() ? 1 : numElements(rt.shape);
                readbackOutput(graph_.tensors[oid].dtype, rt, outElems, io);
            } else
            {
                io.dtype = rt.dtype; // a bound output lives in the caller's fd, not here
            }
            outputs.push_back(std::move(io));
            rt.dmaBufFd     = -1; // reset for the next run
            rt.dmaBufFormat = TensorFormat::NCHW;
            rt.dmaBufDtype  = DType::Float32;
        }
        if (tm)
        {
            auto tD = now();
            auto ms = [&](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            VKNN_INFO << "sess::run bind=" << ms(tA, tB) << "ms segments=" << ms(tB, tC) << "ms collect=" << ms(tC, tD) << "ms";
        }
        return Status::Ok;
    }

    // --- ergonomic API ------------------------------------------------------------------------------

    static IOInfo ioInfoOf(const Graph &g, TensorId id, Precision prec) {
        IOInfo info;
        info.name  = g.tensors[id].name;
        info.shape = g.tensors[id].shape;
        info.dtype = g.tensors[id].dtype;
        info.elems = numElements(info.shape);
        // Zero-copy boundary buffer the caller provides: the segment's device layout for this tensor at
        // the compute precision (fp16 -> 2 bytes/elem). Flat boundaries are row-major NCHW; the rest are
        // NC4HW4 (channels in groups of 4, padded), whose byte size includes the channel padding.
        int64_t elemSize = (prec == Precision::High) ? 4 : 2;
        info.deviceDtype = (prec == Precision::High) ? DType::Float32 : DType::Float16;
        if (g.desc(id).gpuFlat)
        {
            info.deviceBytes  = info.elems * elemSize;
            info.deviceFormat = TensorFormat::NCHW;
        } else
        {
            NCHW x            = NCHW::from(info.shape);
            info.deviceBytes  = x.n * cBlocks(x.c) * 4 * x.h * x.w * elemSize;
            info.deviceFormat = TensorFormat::NC4HW4;
        }
        return info;
    }

    std::vector<IOInfo> Session::inputInfo() const {
        const Graph        &graph_ = *buckets_.front().graph;
        std::vector<IOInfo> v;
        for (TensorId id: graph_.inputs)
        {
            v.push_back(ioInfoOf(graph_, id, cfg_.precision));
        }
        return v;
    }

    std::vector<IOInfo> Session::outputInfo() const {
        const Graph        &graph_ = *buckets_.front().graph;
        std::vector<IOInfo> v;
        for (TensorId id: graph_.outputs)
        {
            v.push_back(ioInfoOf(graph_, id, cfg_.precision));
        }
        return v;
    }

    std::vector<IOInfo> Session::inputInfo(size_t bucket) const {
        std::vector<IOInfo> v;
        if (bucket >= buckets_.size())
        {
            return v;
        }
        const Graph &g = *buckets_[bucket].graph;
        for (TensorId id: g.inputs)
        {
            v.push_back(ioInfoOf(g, id, cfg_.precision));
        }
        return v;
    }

    std::vector<IOInfo> Session::outputInfo(size_t bucket) const {
        std::vector<IOInfo> v;
        if (bucket >= buckets_.size())
        {
            return v;
        }
        const Graph &g = *buckets_[bucket].graph;
        for (TensorId id: g.outputs)
        {
            v.push_back(ioInfoOf(g, id, cfg_.precision));
        }
        return v;
    }

    std::vector<std::string> Session::bucketKeys() const {
        std::vector<std::string> v;
        for (const PlanBucket &b: buckets_)
        {
            v.push_back(b.key);
        }
        return v;
    }

    Status Session::run(const std::vector<std::vector<float>> &inputData, std::vector<IOTensor> &outputs) {
        // The convenience overload binds one buffer per input in default-bucket order; the built
        // IOTensors carry each input's default-bucket shape so run() dispatches to the default bucket.
        const Graph &graph_ = *buckets_.front().graph;
        if (inputData.size() != graph_.inputs.size())
        {
            VKNN_ERROR << "run: expected " << graph_.inputs.size() << " input(s), got " << inputData.size();
            return Status::InvalidArgument;
        }
        std::vector<IOTensor> ins(graph_.inputs.size());
        for (size_t i = 0; i < graph_.inputs.size(); ++i)
        {
            TensorId          id   = graph_.inputs[i];
            const TensorDesc &d    = graph_.tensors[id];
            int64_t           want = numElements(d.shape);
            // Allow callers not to know the count (want<=0 for fully-dynamic shapes); otherwise validate.
            if (want > 0 && (int64_t) inputData[i].size() != want)
            {
                VKNN_ERROR << "run: input '" << d.name << "' expects " << want << " values, got " << inputData[i].size();
                return Status::InvalidArgument;
            }
            ins[i].name      = d.name;
            ins[i].shape     = d.shape;
            ins[i].dtype     = DType::Float32;
            const uint8_t *p = reinterpret_cast<const uint8_t *>(inputData[i].data());
            ins[i].data.assign(p, p + inputData[i].size() * sizeof(float));
        }
        return run(ins, outputs);
    }

    std::vector<float> Session::infer(const std::vector<float> &input) {
        std::vector<IOTensor> outs;
        if (run({input}, outs) != Status::Ok || outs.empty())
        {
            return {};
        }
        const float *o = outs[0].f32();
        return std::vector<float>(o, o + numElements(outs[0].shape));
    }

} // namespace vknn
