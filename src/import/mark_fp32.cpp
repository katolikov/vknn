#include "passes_internal.h"
#include <unordered_set>

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
                // pinned fp32 by pinSampleCoordFp32 must not be narrowed back to fp16 -- that would
                // re-quantize the sampling coordinates the pin exists to protect. An fp32 GridSample
                // (nodeFp32) keeps the bridge: gridsample.comp reads the grid as float only.
                if (nd.type == OpType::GridSample && inIdx == 1 && !nodeFp32)
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

    bool coordinateTransparentOp(OpType op) {
        switch (op)
        {
            case OpType::Add:
            case OpType::Binary:
            case OpType::Unary:
            case OpType::Clip:
            // The rest of the elementwise family. A flow clamped by a Relu or scaled by a PRelu is
            // ordinary coordinate algebra, and both have fp32 kernels; leaving them out made each one
            // a wall the pin stopped at, for no reason the arithmetic can name.
            case OpType::Relu:
            case OpType::PRelu:
            case OpType::Concat:
            case OpType::Slice:
            case OpType::Split:
            case OpType::Expand:
            case OpType::Tile:
            case OpType::Reshape:
            case OpType::Transpose:
            case OpType::Unsqueeze:
            case OpType::Squeeze:
            case OpType::Cast:
            case OpType::Where:
            case OpType::FusedPointwise:
            case OpType::Reduce:
            case OpType::ConvertLayout:
            // Movement that carries values verbatim: a pixel-shuffle upsample of a flow, a padded
            // coordinate field, a channel remap, a rank change. Each has kernels at both precisions
            // and none of them touches the VALUE, only where it sits.
            case OpType::DepthToSpace:
            case OpType::Pad:
            case OpType::ChannelShuffle:
            case OpType::Flatten:
            // A Resize carries coordinates from one grid density to another; it is as much a step of
            // the coordinate algebra as a multiply is, and it has fp32 kernels. Left out of this set
            // it became a WALL: the pin stopped there, so the tensor on the far side kept the session
            // precision and markFp32 spliced a ConvertDtype across the seam. Every Resize inside a
            // flow chain cost one such bridge.
            case OpType::Resize:
                return true;
            default:
                return false;
        }
    }

    // ONNX TensorProto dtype codes a Cast `to` attribute carries. Only the codes whose CPU/GPU
    // implementations TRUNCATE need the pin; the float codes are a same-precision copy.
    constexpr int64_t kOnnxDtypeFloat  = 1;
    constexpr int64_t kOnnxDtypeUint8  = 2;
    constexpr int64_t kOnnxDtypeInt8   = 3;
    constexpr int64_t kOnnxDtypeUint16 = 4;
    constexpr int64_t kOnnxDtypeInt16  = 5;
    constexpr int64_t kOnnxDtypeInt32  = 6;
    constexpr int64_t kOnnxDtypeInt64  = 7;
    constexpr int64_t kOnnxDtypeBool   = 9;
    constexpr int64_t kOnnxDtypeUint32 = 12;
    constexpr int64_t kOnnxDtypeUint64 = 13;
    // Hop ceiling of a pin walk, shared with the index and coordinate pins: a movement chain longer
    // than this is pathological and the walk stops rather than scanning an adversarial graph.
    constexpr int kFp32PinMaxHops = 64;

    /// True when a Cast to `onnxDtype` truncates its input toward zero rather than copying it at
    /// the same precision. Truncation is discontinuous, so any storage rounding that crosses an
    /// integer boundary becomes a full-unit output error — which is why the operand cone of such a
    /// Cast is pinned to fp32 exactly like an index or a sampling coordinate.
    bool castTargetTruncates(int64_t onnxDtype) {
        switch (onnxDtype)
        {
            case kOnnxDtypeUint8:
            case kOnnxDtypeInt8:
            case kOnnxDtypeUint16:
            case kOnnxDtypeInt16:
            case kOnnxDtypeInt32:
            case kOnnxDtypeInt64:
            case kOnnxDtypeBool:
            case kOnnxDtypeUint32:
            case kOnnxDtypeUint64:
                return true;
            default:
                return false; // FLOAT / FLOAT16 / DOUBLE: a same-precision copy
        }
    }

    /// True when a Unary step is discontinuous in its input: it maps an interval to one value and
    /// jumps at the boundary, so a storage rounding that crosses a boundary is a full-unit output
    /// error. Trunc is what foldIntRoundtripCast leaves behind when it collapses a
    /// float->wide-int->float Cast pair, so a graph can carry this hazard with no Cast node left.
    bool unaryStepIsDiscontinuous(int32_t subOp) {
        switch ((UnaryType) subOp)
        {
            case UnaryType::Floor:
            case UnaryType::Ceil:
            case UnaryType::Round:
            case UnaryType::Trunc:
            case UnaryType::Sign:
                return true;
            default:
                return false;
        }
    }

    /// True when an op reproduces its input values unchanged and only moves or re-labels them, so a
    /// pin may follow it upstream without changing what any kernel computes.
    bool movementOnlyOp(OpType op) {
        switch (op)
        {
            case OpType::ConvertLayout:
            case OpType::Reshape:
            case OpType::Transpose:
            case OpType::Squeeze:
            case OpType::Unsqueeze:
            case OpType::Slice:
            case OpType::Concat:
            case OpType::Expand:
            case OpType::Tile:
                return true;
            default:
                return false;
        }
    }

    void pinDiscontinuousStepFp32(Graph &g) {
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
            if (nd.inputs.empty() || nd.inputs[0] == kNoTensor)
            {
                continue;
            }
            const bool truncatingCast = nd.type == OpType::Cast && castTargetTruncates(nd.attr.geti("to", kOnnxDtypeFloat));
            const bool steppingUnary  = nd.type == OpType::Unary && unaryStepIsDiscontinuous(nd.subOp);
            if (!truncatingCast && !steppingUnary)
            {
                continue; // a float-target Cast and a smooth unary both carry rounding proportionally
            }
            // The result is a logical integer carried in a float slot, so an fp16 store on the
            // OUTPUT would round it a second time (fp16 spaces integers past 2048).
            if (nd.outputs[0] != kNoTensor && !g.desc(nd.outputs[0]).storeFp32)
            {
                g.desc(nd.outputs[0]).storeFp32 = true;
                ++pinned;
            }
            // Walk the operand back through pure movement producers: a rounding one hop upstream
            // reaches trunc() as the same full-unit error.
            TensorId t = nd.inputs[0];
            for (int hop = 0; t != kNoTensor && !g.isInitializer(t) && hop < kFp32PinMaxHops; ++hop)
            {
                if (g.desc(t).storeFp32)
                {
                    break; // already pinned (a shared operand, or a prior hop)
                }
                g.desc(t).storeFp32 = true;
                ++pinned;
                int p = producer[(size_t) t];
                if (p < 0)
                {
                    break; // graph-input boundary: pinned, nothing upstream to follow
                }
                const Node &pn = g.nodes[(size_t) p];
                if (movementOnlyOp(pn.type) && pn.inputs.size() >= 1)
                {
                    t = pn.inputs[0]; // same values, different placement -- keep pinning to the source
                } else
                {
                    break; // a computing producer runs at its own precision; markFp32 bridges it
                }
            }
        }
        if (pinned != 0)
        {
            VKNN_INFO << "pinDiscontinuousStepFp32: pinned " << pinned << " tensor(s) around truncating Cast / stepping Unary nodes";
        }
    }

    void pinSampleCoordFp32(Graph &g) {
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
        // Seeds: every GridSample coordinate operand. The warp variant's base grid (input 2) is a
        // constant the op already uploads fp32; the runtime operands are the plain grid and the
        // warp flow at input 1.
        std::vector<TensorId> walk;
        for (const auto &nd: g.nodes)
        {
            if (nd.type != OpType::GridSample || nd.inputs.size() < 2)
            {
                continue;
            }
            for (size_t i = 1; i < nd.inputs.size(); ++i)
            {
                if (nd.inputs[i] != kNoTensor && !g.isInitializer(nd.inputs[i]))
                {
                    walk.push_back(nd.inputs[i]);
                }
            }
        }
        int                          pinned = 0;
        std::unordered_set<TensorId> seen;
        while (!walk.empty())
        {
            TensorId t = walk.back();
            walk.pop_back();
            if (t == kNoTensor || !seen.insert(t).second || g.isInitializer(t))
            {
                continue;
            }
            const int p = producer[(size_t) t];
            // A hop produced by a non-transparent, non-flat op (the NC4HW4 conv family) stays at
            // the session precision: those kernels have no fp32 twins, and markFp32 bridges the
            // fp16 -> fp32 boundary with a ConvertDtype where the pinned algebra reads it.
            if (p >= 0 && !coordinateTransparentOp(g.nodes[(size_t) p].type) && !g.desc(t).gpuFlat)
            {
                continue;
            }
            if (!g.desc(t).storeFp32)
            {
                g.desc(t).storeFp32 = true;
                ++pinned;
            }
            // A GPU kernel stores every one of its streams at ONE precision: pinning a node's
            // output pins its sibling outputs too (a fused unit's export streams, a Split's other
            // parts), or the fp32 kernel would write 4-byte values into buffers planned fp16.
            if (p >= 0)
            {
                for (TensorId sib: g.nodes[(size_t) p].outputs)
                {
                    if (sib != kNoTensor && !g.desc(sib).storeFp32)
                    {
                        g.desc(sib).storeFp32 = true;
                        ++pinned;
                    }
                }
            }
            // A Reduce is the cone's upstream BOUNDARY, not a corridor: its output (a zoom-box
            // extent, a mask centroid) is a coordinate and pins, but its INPUT domain is image
            // content - walking through would pin whole mask-image chains, paying fp32 traffic on
            // image-sized tensors for values whose fp16 noise the reduction itself averages away.
            if (p >= 0 && coordinateTransparentOp(g.nodes[(size_t) p].type) && g.nodes[(size_t) p].type != OpType::Reduce)
            {
                for (TensorId in: g.nodes[(size_t) p].inputs)
                {
                    walk.push_back(in);
                }
            }
        }
        if (pinned)
        {
            VKNN_INFO << "pinSampleCoordFp32: pinned " << pinned << " coordinate tensor(s) to fp32";
        }
    }

} // namespace vknn
