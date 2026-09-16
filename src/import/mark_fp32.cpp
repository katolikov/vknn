#include "import/mod_integer_operands.h"
#include "passes_internal.h"

namespace vknn {

    // The Config::fp32Tensors matcher: a comma list of substrings; a leading '-' marks an EXCLUDE
    // (a name with an excluded substring is never marked even if it matches an include), so a
    // fragile sub-region can be carved out. Shared with the fusion pass's compile-time prediction
    // (pwTensorIsFp32) — the two MUST agree or a fused unit can span a tensor markFp32 pins.
    bool fp32NameMatch(const std::string &nm, const std::string &substrs) {
        if (nm.empty() || substrs.empty())
        {
            return false;
        }
        std::vector<std::string> incl, excl;
        for (size_t p = 0, c;; p = c + 1)
        {
            c             = substrs.find(',', p);
            std::string s = substrs.substr(p, c == std::string::npos ? c : c - p);
            if (!s.empty())
            {
                (s[0] == '-' ? excl : incl).push_back(s[0] == '-' ? s.substr(1) : s);
            }
            if (c == std::string::npos)
            {
                break;
            }
        }
        for (const auto &s: excl)
        {
            if (nm.find(s) != std::string::npos)
            {
                return false;
            }
        }
        for (const auto &s: incl)
        {
            if (nm.find(s) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    // Split a comma-separated pattern list into its non-empty entries, as typed (an exclude entry
    // keeps its leading '-'). The zero-match accounting below and the session's load-end warning
    // both enumerate entries through this, so a warned entry is exactly what the caller wrote.
    std::vector<std::string> splitPatternList(const std::string &patterns) {
        std::vector<std::string> entries;
        for (size_t p = 0, c;; p = c + 1)
        {
            c             = patterns.find(',', p);
            std::string s = patterns.substr(p, c == std::string::npos ? c : c - p);
            if (!s.empty())
            {
                entries.push_back(std::move(s));
            }
            if (c == std::string::npos)
            {
                break;
            }
        }
        return entries;
    }

    // Selective fp32: mark every activation tensor whose name contains one of the comma-separated
    // substrings (Config::fp32Tensors) so its buffer stays fp32 under fp16 compute, then bridge the
    // fp16/fp32 frontier with ConvertDtype nodes — for each node, any activation input whose storage
    // dtype differs from the node's (its output[0]) gets a convert, exactly mirroring insertLayoutConverts.
    // Initializers are skipped: ops upload them at the node's precision (env.useFp16). Runs at load, after
    // insertLayoutConverts, so it operates on the final flat names.
    void markFp32(Graph &g, const std::string &substrs, std::set<std::string> *matchedPatterns) {
        // Substring marks from Config::fp32Tensors are additive on top of any storage an earlier pass
        // already pinned to fp32 (pinGatherIndexFp32's index chains). Only flat tensors are eligible for a
        // substring mark: the flat transformer/geometry kernels all #include precision.glsl so an fp32
        // SPIR-V variant exists, whereas the NC4HW4 conv family (conv/wino/dwconv/fc/pool) is hand-written
        // fp16-only. Marking an NC4HW4 tensor would request a non-existent fp32 kernel.
        int marked = 0;
        if (!substrs.empty())
        {
            // Per-entry zero-match accounting (matchedPatterns non-null): an entry is matched when its
            // substring occurs in an ELIGIBLE tensor's name — the same non-initializer flat set the
            // marking consults — so the session's load-end warning names exactly the entries that
            // cannot affect this model. Exclude entries account the same way (an exclude matching
            // nothing is an inert knob too) and are recorded as typed, '-' included.
            std::vector<std::string> entries, entryText;
            if (matchedPatterns)
            {
                entries = splitPatternList(substrs);
                for (const std::string &e: entries)
                {
                    entryText.push_back(e[0] == '-' ? e.substr(1) : e);
                }
            }
            auto matches = [&](const std::string &nm) {
                return fp32NameMatch(nm, substrs);
            };
            for (auto &t: g.tensors)
            {
                if (t.isInitializer || !t.gpuFlat)
                {
                    continue;
                }
                if (matches(t.name))
                {
                    t.storeFp32 = true;
                    ++marked;
                }
                for (size_t e = 0; e < entries.size(); ++e)
                {
                    if (!entryText[e].empty() && t.name.find(entryText[e]) != std::string::npos)
                    {
                        matchedPatterns->insert(entries[e]);
                    }
                }
            }
            if (!marked)
            {
                VKNN_INFO << "markFp32: no tensor matched fp32Tensors=\"" << substrs << "\"";
            }
        }
        // The frontier walk below runs whether or not a substring matched, so a chain pinned by an earlier
        // pass still gets its ConvertDtype bridges. With nothing anywhere in fp32 it is a pure no-op.
        // A kernel writes every output in ONE storage precision (its outputs[0]'s), so all outputs
        // of a multi-output node must share a mark — a fused unit's exported stream (pw_outs) pinned
        // differently from the main output would be written half-empty (fp16 stores into an fp32
        // buffer) or overrun (the reverse). Align to outputs[0]; consumers needing the other
        // precision get their ConvertDtype from the frontier walk below.
        for (auto &nd: g.nodes)
        {
            if (nd.outputs.size() < 2 || nd.outputs[0] == kNoTensor)
            {
                continue;
            }
            bool nodeFp32 = g.desc(nd.outputs[0]).storeFp32;
            for (size_t k = 1; k < nd.outputs.size(); ++k)
            {
                if (nd.outputs[k] != kNoTensor)
                {
                    g.desc(nd.outputs[k]).storeFp32 = nodeFp32;
                }
            }
        }
        // (source tensor, wantFp32) -> already-converted tensor, so one frontier tensor consumed at a
        // given precision by several nodes is converted once and the result shared.
        std::map<std::pair<TensorId, bool>, TensorId> cache;
        // New ConvertDtype nodes are buffered rather than appended in-place: mutating g.nodes while the
        // range-for below iterates it would invalidate that loop. They are spliced in after the walk.
        std::vector<Node> converts;
        int               n = 0;
        // Route one tensor read through a ConvertDtype when its storage precision differs from the
        // precision the reader's kernel runs in, reusing an already-converted copy from `cache`. `ref`
        // is the reader's reference (a node input or a fused edge) and is rewired in place.
        auto convertRead = [&](TensorId &ref, bool wantFp32) {
            if (ref == kNoTensor || g.isInitializer(ref))
            {
                return; // initializers upload at the node's precision (env.useFp16)
            }
            if (g.desc(ref).storeFp32 == wantFp32)
            {
                return;
            }
            auto key = std::make_pair(ref, wantFp32);
            auto it  = cache.find(key);
            if (it == cache.end())
            {
                TensorDesc d    = g.desc(ref);
                d.name          = g.desc(ref).name + (wantFp32 ? "#f32" : "#f16") + std::to_string(n);
                d.isInitializer = d.isInput = d.isOutput = false;
                d.storeFp32                              = wantFp32;
                d.gpuFlat                                = g.desc(ref).gpuFlat; // dtype change only, same layout
                TensorId t2                              = g.addTensor(d);
                Node     cv;
                cv.type    = OpType::ConvertDtype;
                cv.name    = "cvtdt" + std::to_string(n++);
                cv.inputs  = {ref};
                cv.outputs = {t2};
                converts.push_back(cv);
                it = cache.emplace(key, t2).first;
            }
            ref = it->second;
        };
        for (auto &nd: g.nodes)
        {
            if (nd.outputs.empty() || nd.outputs[0] == kNoTensor)
            {
                continue;
            }
            bool nodeFp32 = g.desc(nd.outputs[0]).storeFp32; // the precision this node's kernel runs in
            for (size_t inIdx = 0; inIdx < nd.inputs.size(); ++inIdx)
            {
                // A Gather reads its index (input 1) as fp32 no matter the kernel's compute precision
                // (gather.comp binding 1 is float), so a pinned fp32 index must not be narrowed back to
                // fp16 by a frontier convert -- that would re-overflow a large token id to +inf.
                // Rope reads its position tensor at the same slot under the same contract (rope.comp
                // binding 1 is float).
                if ((nd.type == OpType::Gather || nd.type == OpType::Rope) && inIdx == 1)
                {
                    continue;
                }
                // An fp16 GridSample decodes its grid (input 1) at the grid's OWN storage precision
                // (gridsample_fp16.comp reads raw words per the GRID_FP32 spec constant), so a grid
                // pinned fp32 by pinGridSampleGridFp32 must not be narrowed back to fp16 -- that would
                // re-quantize the sampling coordinates the pin exists to protect. An fp32 GridSample
                // (nodeFp32) keeps the bridge: gridsample.comp reads the grid as float only.
                if (nd.type == OpType::GridSample && inIdx == 1 && !nodeFp32)
                {
                    continue;
                }
                // ArgMax/ArgMin read their data (input 0) at the data tensor's OWN storage precision (the
                // kernel selects its fp16 or fp32 variant from it), while pinIntegerResultsFp32 pins the
                // indices output -- and so the node -- to fp32. A bridge here would only copy the whole
                // data tensor to fp32 for a comparison scan that needs no extra precision.
                if ((nd.type == OpType::ArgMax || nd.type == OpType::ArgMin) && inIdx == 0)
                {
                    continue;
                }
                convertRead(nd.inputs[inIdx], nodeFp32);
            }
            // Fused residual/bias edges are reads outside the inputs list (rewireTensor's contract)
            // and the kernel decodes them at ITS storage precision, so they take the same bridges as
            // any input. An edge mirrored into inputs (a conv residual doubling at the bias slot; the
            // conv kernel tests inputs[2] != fusedResidual to tell the two apart) resolves through the
            // same cache entry, so the mirrored entry and the edge stay one tensor.
            convertRead(nd.fusedResidual, nodeFp32);
            convertRead(nd.fusedBias, nodeFp32);
        }
        if (!converts.empty())
        {
            // The converts are appended at the tail (out of dependency order), then topoSort restores a
            // valid execution order so each ConvertDtype runs before the node that reads its output.
            for (auto &c: converts)
            {
                g.nodes.push_back(std::move(c));
            }
            g.topoSort();
        }
        VKNN_INFO << "markFp32: marked " << marked << " tensor(s) fp32, inserted " << converts.size() << " convert(s)";
    }

    void pinGatherIndexFp32(Graph &g) {
        // Last writer of each tensor, so the index can be traced back to its boundary source.
        std::vector<int> producer(g.tensors.size(), -1);
        for (int ni = 0; ni < (int) g.nodes.size(); ++ni)
        {
            for (TensorId o: g.nodes[ni].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[(size_t) o] = ni;
                }
            }
        }
        int pinned = 0;
        for (const auto &nd: g.nodes)
        {
            // Rope (the fused rotate-half chain) reads its position tensor as input 1 exactly like
            // Gather reads its index there, and its shader binds that buffer as fp32 for the same
            // reason — an integer position must never round through fp16 storage.
            const bool indexed = nd.type == OpType::Gather || nd.type == OpType::Rope;
            if (!indexed || nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor)
            {
                continue;
            }
            // A constant index is uploaded fp32 by the op itself, so only a runtime index needs pinning.
            // Walk it up through pure passthrough producers (the ConvertLayout the flat pass splices in) to
            // the boundary input, pinning every hop to fp32 so an integer index never rounds to fp16 (a
            // token id above 65504 would otherwise store as +inf and the lookup would read the wrong row).
            TensorId t = nd.inputs[1];
            for (int hop = 0; t != kNoTensor && !g.isInitializer(t) && hop < 64; ++hop)
            {
                if (g.desc(t).storeFp32)
                {
                    break; // already pinned (a shared index, or a prior hop)
                }
                g.desc(t).storeFp32 = true;
                ++pinned;
                int p = producer[(size_t) t];
                if (p < 0)
                {
                    break; // graph-input boundary: pinned, nothing upstream to follow
                }
                const Node &pn = g.nodes[(size_t) p];
                if (pn.type == OpType::ConvertLayout && pn.inputs.size() == 1)
                {
                    t = pn.inputs[0]; // same values, different layout -- keep pinning toward the source
                } else
                {
                    break; // a real op computes the index; its (now-pinned) output runs fp32 via nodeFp32
                }
            }
        }
        if (pinned)
        {
            VKNN_INFO << "pinGatherIndexFp32: pinned " << pinned << " index tensor(s) to fp32";
        }
    }

    void pinGridSampleGridFp32(Graph &g) {
        // Last writer of each tensor, so the grid can be traced back toward its source.
        std::vector<int> producer(g.tensors.size(), -1);
        for (int ni = 0; ni < (int) g.nodes.size(); ++ni)
        {
            for (TensorId o: g.nodes[ni].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[(size_t) o] = ni;
                }
            }
        }
        int pinned = 0;
        for (const auto &nd: g.nodes)
        {
            if (nd.type != OpType::GridSample || nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor)
            {
                continue;
            }
            // A warp-mode GridSample (fuseGridSampleWarp) has no runtime grid to pin: input 1 is the
            // NCHW flow (read fp16 in the NC4HW4 activation layout) and the base grid (input 2, uploaded
            // fp32 by the op) carries the coordinates' full precision. The fp16 sampler reproduces the
            // split Mul's fp16-rounded product, so flow must stay fp16 — pinning it would diverge.
            if (nd.attr.has("warp"))
            {
                continue;
            }
            // The grid holds normalized sampling COORDINATES: fp16 storage quantizes them at ~4.9e-4
            // near |g|=1, which drifts the sample point by up to ~0.5 px at 1920-wide inputs — a direct
            // warp-quality (UV) loss. A constant grid is uploaded fp32 by the op itself, so only a
            // runtime grid (optical flow) needs pinning. Walk it up through pure passthrough producers
            // (the ConvertLayout the flat pass splices in) pinning every hop, but only while the hop is
            // FLAT: the NC4HW4 conv family is hand-written fp16-only, so pinning a conv output would
            // request a non-existent fp32 kernel (the frontier walk in markFp32 bridges the boundary
            // with a ConvertDtype instead). Mirrors pinGatherIndexFp32.
            TensorId t = nd.inputs[1];
            for (int hop = 0; t != kNoTensor && !g.isInitializer(t) && g.desc(t).gpuFlat && hop < 64; ++hop)
            {
                if (g.desc(t).storeFp32)
                {
                    break; // already pinned (a shared grid, or a prior hop)
                }
                g.desc(t).storeFp32 = true;
                ++pinned;
                int p = producer[(size_t) t];
                if (p < 0)
                {
                    break; // graph-input boundary: pinned, nothing upstream to follow
                }
                const Node &pn = g.nodes[(size_t) p];
                if (pn.type == OpType::ConvertLayout && pn.inputs.size() == 1)
                {
                    t = pn.inputs[0]; // same values, different layout -- keep pinning toward the source
                } else
                {
                    break; // a real op computes the grid; its (now-pinned) output runs fp32 via nodeFp32
                }
            }
        }
        if (pinned)
        {
            VKNN_INFO << "pinGridSampleGridFp32: pinned " << pinned << " grid tensor(s) to fp32";
        }
    }

    namespace {
        // ONNX Mod `fmod` value (also the attribute's default) selecting the integer remainder whose sign
        // follows the divisor; fmod == 1 is the C fmod, integer-valued only on integer operands.
        constexpr int64_t kModIntegerRemainder = 0;

        // ONNX TensorProto.DataType codes: FLOAT (the Cast `to` default) and the integer element types.
        constexpr int64_t kOnnxFloat  = 1;
        constexpr int64_t kOnnxUInt8  = 2;
        constexpr int64_t kOnnxInt8   = 3;
        constexpr int64_t kOnnxUInt16 = 4;
        constexpr int64_t kOnnxInt16  = 5;
        constexpr int64_t kOnnxInt32  = 6;
        constexpr int64_t kOnnxInt64  = 7;
        constexpr int64_t kOnnxUInt32 = 12;
        constexpr int64_t kOnnxUInt64 = 13;

        bool castTargetsInteger(const Node &nd) {
            const int64_t to = nd.attr.geti("to", kOnnxFloat);
            return to == kOnnxUInt8 || to == kOnnxInt8 || to == kOnnxUInt16 || to == kOnnxInt16 || to == kOnnxInt32 || to == kOnnxInt64 || to == kOnnxUInt32 || to == kOnnxUInt64;
        }

        // A graph-declared wide integer dtype (graph inputs and initializers carry theirs; a runtime
        // intermediate is typed only where an import rule stamps it). 8-bit values are exact in fp16.
        bool typedWideInteger(const Graph &g, TensorId t) {
            return t != kNoTensor && (g.desc(t).dtype == DType::Int64 || g.desc(t).dtype == DType::Int32);
        }

        // Operand count of a movement op that reads its values from operand 0 alone.
        constexpr size_t kSingleDataOperand = 1;

        // The value-preserving ops an integer region extends through, as the data operand slots
        // [first, end): every output holds element values copied from those operands (the other inputs
        // are shape/index/axis parameters), so integer data makes an integer result and an integer result
        // has integer data. Layout converts, Identity, metadata reshapes, Cast (its input may be float, so
        // the walk treats it one-way; see pinIntegerResultsFp32), movement and selection. Empty for every
        // other op, and for a movement op hosting a fused pointwise chain, which computes new values.
        std::pair<size_t, size_t> integerPassthroughDataSlots(const Node &nd) {
            if (nd.attr.has("pw_steps"))
            {
                return {0, 0};
            }
            const size_t coreInputs = pwCoreInputs(nd);
            switch (nd.type)
            {
                case OpType::ConvertLayout:
                case OpType::Identity:
                case OpType::Reshape:
                case OpType::Flatten:
                case OpType::Squeeze:
                case OpType::Unsqueeze:
                case OpType::Cast:
                case OpType::Slice:
                case OpType::Transpose:
                case OpType::Expand:
                case OpType::Tile:
                case OpType::Split:
                case OpType::Gather: // data operand 0; the index (operand 1) is pinned by pinGatherIndexFp32
                    return {0, std::min(kSingleDataOperand, coreInputs)};
                case OpType::Concat:
                    return {0, coreInputs};
                default:
                    return {0, 0};
            }
        }
    } // namespace

    void pinIntegerResultsFp32(Graph &g) {
        // Last writer of each tensor and every (node, input slot) reading it, so the region can be
        // followed toward sources and toward consumers.
        std::vector<int>                                 producer(g.tensors.size(), -1);
        std::vector<std::vector<std::pair<int, size_t>>> readers(g.tensors.size());
        for (int ni = 0; ni < (int) g.nodes.size(); ++ni)
        {
            for (TensorId o: g.nodes[ni].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[(size_t) o] = ni;
                }
            }
            for (size_t slot = 0; slot < g.nodes[ni].inputs.size(); ++slot)
            {
                if (g.nodes[ni].inputs[slot] != kNoTensor)
                {
                    readers[(size_t) g.nodes[ni].inputs[slot]].push_back({ni, slot});
                }
            }
        }
        // Whether a tensor may take fp32 storage: a runtime tensor that is flat (every flat kernel has an
        // fp32 variant), or an NC4HW4 tensor no fp16-only kernel writes -- a graph input (the boundary
        // packs it at its storage precision) or the output of a layout convert, a metadata reshape (a
        // buffer copy) or a Cast (cast.comp has an fp32 variant). The layout pass keeps an agnostic
        // chain rooted at a graph input NC4HW4 until a reader changes the channel count, so an integer
        // input often reaches its flat integer op through such hops. The NC4HW4 conv family is
        // hand-written fp16-only, so its outputs are never pinned; markFp32 bridges them instead.
        auto storageCanPin = [&](TensorId t) {
            if (t == kNoTensor || g.isInitializer(t))
            {
                return false;
            }
            if (g.desc(t).gpuFlat)
            {
                return true;
            }
            const int p = producer[(size_t) t];
            if (p < 0)
            {
                return true;
            }
            const Node &pn = g.nodes[(size_t) p];
            if (pn.attr.has("pw_steps"))
            {
                return false;
            }
            switch (pn.type)
            {
                case OpType::ConvertLayout:
                case OpType::Reshape:
                case OpType::Flatten:
                case OpType::Squeeze:
                case OpType::Unsqueeze:
                case OpType::Cast:
                    return true;
                default:
                    return false;
            }
        };
        int  pinned = 0;
        auto pin    = [&](TensorId t) {
            if (!storageCanPin(t) || g.desc(t).storeFp32)
            {
                return;
            }
            g.desc(t).storeFp32 = true;
            ++pinned;
        };

        // How far the walk follows a tensor: an Integer tensor extends the region toward its sources and
        // its consumers; an Upstream tensor (a Cast's operand, which may hold float values) is followed
        // only toward its source, so fp32 never spreads into the float readers beside it. Ordered so a
        // tensor reached both ways is walked at the wider reach.
        enum class Reach : uint8_t { None, Upstream, Integer };
        std::vector<Reach>                      reach(g.tensors.size(), Reach::None);
        std::vector<std::pair<TensorId, Reach>> pending;
        auto                                    enqueue = [&](TensorId t, Reach r) {
            if (t != kNoTensor)
            {
                pending.push_back({t, r});
            }
        };

        // Seeds: integer results, and the operands of the ops whose operands are integers too. A constant
        // operand is uploaded by the op itself at the node's (pinned) precision, so only runtime tensors
        // are pinned.
        for (const Node &nd: g.nodes)
        {
            bool integerResult   = false;
            bool integerOperands = false;
            switch (nd.type)
            {
                case OpType::ArgMax:
                case OpType::ArgMin:
                    // int64 indices; the data is integer only where its dtype says so (a float scan
                    // needs no extra precision, so markFp32 leaves it at its own storage precision).
                    integerResult = true;
                    if (!nd.inputs.empty() && typedWideInteger(g, nd.inputs[0]))
                    {
                        enqueue(nd.inputs[0], Reach::Integer);
                    }
                    break;
                case OpType::Mod:
                    // fmod 0 is the integer remainder; fmod 1 is integer-valued on integer operands,
                    // which modOperandsAreInteger resolves from dtypes and producers exactly as the
                    // Mod kernels do.
                    integerResult   = nd.attr.geti("fmod", kModIntegerRemainder) == kModIntegerRemainder || modOperandsAreInteger(g, nd, producer);
                    integerOperands = integerResult;
                    break;
                case OpType::BitShift:
                case OpType::BitwiseAnd:
                case OpType::BitwiseOr:
                case OpType::BitwiseXor:
                case OpType::BitwiseNot:
                    integerResult   = true;
                    integerOperands = true;
                    break;
                default:
                    break;
            }
            if (integerResult)
            {
                for (TensorId o: nd.outputs)
                {
                    enqueue(o, Reach::Integer);
                }
            }
            if (integerOperands)
            {
                for (TensorId in: nd.inputs)
                {
                    enqueue(in, Reach::Integer);
                }
            }
        }

        // Flood the region. Every tensor is walked at most once per reach, so the walk terminates. The
        // region stops at a tensor that cannot take fp32 storage (an initializer, or an output of the
        // fp16-only NC4HW4 conv family: markFp32's frontier convert bridges it), at a producer that
        // computes new values (its pinned output runs fp32 via nodeFp32), and at a reader whose result
        // is float (a Cast to a float type, or any computing op), in front of which markFp32 places the
        // fp32->fp16 bridge.
        while (!pending.empty())
        {
            const auto [t, r] = pending.back();
            pending.pop_back();
            if (!storageCanPin(t) || reach[(size_t) t] >= r)
            {
                continue;
            }
            reach[(size_t) t] = r;
            pin(t);

            const int p = producer[(size_t) t];
            if (p >= 0)
            {
                const Node &pn = g.nodes[(size_t) p];
                // markFp32 gives every output of a node outputs[0]'s precision, so a pin on a secondary
                // output survives only if the producer's primary output is pinned too.
                if (pn.outputs.size() > 1 && pn.outputs[0] != t)
                {
                    pin(pn.outputs[0]);
                }
                const auto [firstSlot, endSlot] = integerPassthroughDataSlots(pn);
                const bool viaCast              = pn.type == OpType::Cast;
                for (size_t slot = firstSlot; slot < endSlot; ++slot)
                {
                    enqueue(pn.inputs[slot], viaCast ? Reach::Upstream : r);
                }
                if (r == Reach::Integer && !viaCast && firstSlot < endSlot)
                {
                    for (TensorId sibling: pn.outputs)
                    {
                        enqueue(sibling, Reach::Integer); // the other parts of a Split
                    }
                }
            }
            if (r != Reach::Integer)
            {
                continue;
            }
            for (const auto &[readerIndex, slot]: readers[(size_t) t])
            {
                const Node &rn                  = g.nodes[(size_t) readerIndex];
                const auto [firstSlot, endSlot] = integerPassthroughDataSlots(rn);
                if (slot < firstSlot || slot >= endSlot)
                {
                    continue; // not a value-preserving read (a parameter slot, or a computing op)
                }
                const bool viaCast = rn.type == OpType::Cast;
                if (viaCast && !castTargetsInteger(rn))
                {
                    continue; // a float result leaves the integer region
                }
                for (TensorId o: rn.outputs)
                {
                    enqueue(o, Reach::Integer);
                }
                for (size_t sibling = firstSlot; sibling < endSlot && !viaCast; ++sibling)
                {
                    enqueue(rn.inputs[sibling], Reach::Integer); // the other parts of a Concat
                }
            }
        }
        if (pinned)
        {
            VKNN_INFO << "pinIntegerResultsFp32: pinned " << pinned << " integer tensor(s) to fp32";
        }
    }

    void planFlatLayoutAndStorage(Graph &g, const std::string &fp32Marks, std::set<std::string> *matchedPatterns) {
        insertLayoutConverts(g);
        // Integer index tensors (token ids / positions) must survive to the GPU without an fp16 store
        // that would overflow a value above 65504 to +inf. Pin the Gather index chains to fp32 before
        // markFp32 so the buffer planner sizes them 4-byte and their producers run in fp32.
        pinGatherIndexFp32(g);
        // GridSample grids hold normalized sampling coordinates whose fp16 storage quantization drifts
        // the sample point (~0.5 px at 1920-wide inputs). Pin runtime grid chains to fp32 the same way;
        // the GridSample shader decodes the grid at its storage precision.
        pinGridSampleGridFp32(g);
        // Integer results (ArgMax/ArgMin indices, integer Mod, bit shifts and bitwise ops), their
        // integer operands, and the value-preserving region around them are exact only up to 2^11 in
        // fp16 storage; pin them to fp32 the same way before markFp32 bridges the frontier.
        pinIntegerResultsFp32(g);
        markFp32(g, fp32Marks, matchedPatterns);
        g.topoSort();
    }

} // namespace vknn
