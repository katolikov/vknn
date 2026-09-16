#include "core/segment_constant_operands.h"
#include "import/mod_integer_operands.h"
#include "import/onnx/onnx_types.h"
#include "passes_internal.h"
#include "vknn/binary_type.h"
#include "vknn/reduce_type.h"
#include "vknn/unary_type.h"
#include <algorithm>
#include <limits>

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

        // A graph-declared wide integer dtype (graph inputs, graph outputs and initializers carry theirs; a
        // runtime intermediate is typed only where an import rule stamps it). 8-bit values are exact in fp16,
        // and an INT8/UINT8 tensor binds as fp32 and computes as a float on the CPU.
        bool typedWideInteger(const Graph &g, TensorId t) {
            return t != kNoTensor && (g.desc(t).dtype == DType::Int64 || g.desc(t).dtype == DType::Int32);
        }

        // Whether a Mod node computes integers: fmod 0 is the integer remainder, and fmod 1 is
        // integer-valued on integer operands, which modOperandsAreInteger resolves from dtypes and
        // producers exactly as the Mod kernels do.
        bool modComputesIntegers(const Graph &g, const Node &mod, const std::vector<int> &producer) {
            return mod.attr.geti("fmod", kModIntegerRemainder) == kModIntegerRemainder || modOperandsAreInteger(g, mod, producer);
        }

        // Operand slots an IntegerDataSlots mask names, one bit per slot. An op whose data operands are all
        // of its operands (a variadic Concat) sets everyCoreOperand instead of a mask.
        constexpr size_t kMaskedOperandSlots = 32;
        // IntegerDataSlots::dataOutputEnd of an op whose every output holds element values of its data.
        constexpr size_t kEveryOutput = std::numeric_limits<size_t>::max();

        // The data operand of a movement op, a reduction and a unary function, and the base of a Pow.
        constexpr size_t kLeadingOperandSlot = 0;
        // Where: operand 0 is the condition; operands 1 and 2 are the selected values.
        constexpr size_t kWhereTrueValueSlot  = 1;
        constexpr size_t kWhereFalseValueSlot = 2;
        // Clip data, min and max; Range start, limit and delta: three operands of one element type.
        constexpr size_t kSecondOperandSlot = 1;
        constexpr size_t kThirdOperandSlot  = 2;
        // Pow's exponent: read exactly, but it does not type the result.
        constexpr size_t kPowExponentSlot = 1;
        // Pad: operand 1 holds the pads; operand 2 is the constant fill value, of the data's element type.
        constexpr size_t kPadFillValueSlot = 2;
        // ScatterND: operand 1 holds the indices; operand 2 holds the updates, of the data's element type.
        constexpr size_t kScatterUpdatesSlot = 2;
        // TopK: output 0 holds the selected values; outputs from here on (the int64 indices) do not.
        constexpr size_t kTopKValuesOutputEnd = 1;
        // TopK output slot holding the int64 indices (output 0 holds the values).
        constexpr size_t kTopKIndicesOutput = 1;
        // The first output after a node's primary output 0.
        constexpr size_t kFirstSecondaryOutput = 1;

        constexpr uint32_t operandSlotBit(size_t slot) {
            return uint32_t {1} << slot;
        }

        // How a node's result relates to integer element values in its data operand slots.
        enum class IntegerFlow : uint8_t {
            None,     // the result is a float, or the node hosts a fused pointwise chain
            Moves,    // copies or selects element values: integer data makes an integer result and back
            Computes, // integer arithmetic: integer operands make an integer result and back
            Casts,    // the result has the Cast target's element type; the operand may hold floats
            Compares, // a 0/1 result that is right only when the compared integers are read exactly
        };

        // A node's IntegerFlow, the data operand slots it applies to, the operands an integer computation
        // reads exactly without taking its element type from them (Pow's exponent: (-1)^2049 is -1, but
        // fp16 stores 2049 as 2048), and the outputs holding the data's element values. The other inputs
        // are shape, index, axis, condition, count or fill-geometry parameters, and operands from
        // pwCoreInputs on feed a fused unit.
        struct IntegerDataSlots {
            IntegerFlow flow             = IntegerFlow::None;
            size_t      coreOperands     = 0;
            bool        everyCoreOperand = false;
            uint32_t    dataSlotMask     = 0;
            uint32_t    exactSlotMask    = 0;
            size_t      dataOutputEnd    = kEveryOutput;

            bool readsData(size_t slot) const noexcept {
                return slot < coreOperands && (everyCoreOperand || (slot < kMaskedOperandSlots && (dataSlotMask & operandSlotBit(slot)) != 0));
            }
            bool readsExactlyWithoutTyping(size_t slot) const noexcept {
                return slot < coreOperands && slot < kMaskedOperandSlots && (exactSlotMask & operandSlotBit(slot)) != 0;
            }
            bool writesData(size_t outputIndex) const noexcept {
                return outputIndex < dataOutputEnd;
            }
        };

        // The integer role of every op the region extends through. The movement and selection ops are the
        // ones modOperandsAreInteger resolves as sharing their data's element type (layout and dtype
        // converts, Identity, metadata reshapes, Slice, Transpose, Expand, Tile, Split, Gather, Pad and its
        // fill value, DepthToSpace, ScatterND and its updates, TopK's values, Concat, Where's values), plus
        // ChannelShuffle, a channel permutation: their GPU kernels copy element values at the node's
        // storage precision, and the CPU op keeps int64 elements wherever it has an int64 path. Add, Binary
        // (Add/Sub/Mul/Div/Max/Min/Pow) and ReduceSum/Max/Min/Prod store an int64 result when an operand is
        // Int64 on the CPU, Range does when its operands are, and Clip, Neg and Abs keep fp32-carried
        // integers (an INT32 input) integer-valued. Every other op is None: float math (ReduceMean/L2, the
        // other unary functions, MatMul, ...), and a node hosting a fused pointwise chain, whose steps
        // compute new values of their own type.
        IntegerDataSlots integerDataSlots(const Node &nd) {
            if (nd.attr.has("pw_steps"))
            {
                return {};
            }
            auto slots = [&](IntegerFlow flow, uint32_t dataSlotMask) {
                IntegerDataSlots dataSlots;
                dataSlots.flow         = flow;
                dataSlots.coreOperands = std::min(nd.inputs.size(), (size_t) pwCoreInputs(nd));
                dataSlots.dataSlotMask = dataSlotMask;
                return dataSlots;
            };
            auto everyOperand = [&](IntegerFlow flow) {
                IntegerDataSlots dataSlots = slots(flow, 0);
                dataSlots.everyCoreOperand = true;
                return dataSlots;
            };
            const uint32_t leading    = operandSlotBit(kLeadingOperandSlot);
            const uint32_t threeSlots = leading | operandSlotBit(kSecondOperandSlot) | operandSlotBit(kThirdOperandSlot);
            switch (nd.type)
            {
                case OpType::ConvertLayout:
                case OpType::ConvertDtype:
                case OpType::Identity:
                case OpType::Reshape:
                case OpType::Flatten:
                case OpType::Squeeze:
                case OpType::Unsqueeze:
                case OpType::Slice:
                case OpType::Transpose:
                case OpType::Expand:
                case OpType::Tile:
                case OpType::Split:
                case OpType::DepthToSpace:
                case OpType::ChannelShuffle:
                case OpType::Gather: // data operand 0; the index (operand 1) is pinned by pinGatherIndexFp32
                    return slots(IntegerFlow::Moves, leading);
                case OpType::Pad:
                    return slots(IntegerFlow::Moves, leading | operandSlotBit(kPadFillValueSlot));
                case OpType::ScatterND:
                    return slots(IntegerFlow::Moves, leading | operandSlotBit(kScatterUpdatesSlot));
                case OpType::TopK: {
                    // The values are copied from the data; the indices are integers whatever the data holds
                    // (producesIntegers), so a float score tensor never joins through them.
                    IntegerDataSlots topKSlots = slots(IntegerFlow::Moves, leading);
                    topKSlots.dataOutputEnd    = kTopKValuesOutputEnd;
                    return topKSlots;
                }
                case OpType::Concat:
                    return everyOperand(IntegerFlow::Moves);
                case OpType::Where:
                    return slots(IntegerFlow::Moves, operandSlotBit(kWhereTrueValueSlot) | operandSlotBit(kWhereFalseValueSlot));
                case OpType::Cast:
                    return slots(IntegerFlow::Casts, leading);
                case OpType::Add:
                    return everyOperand(IntegerFlow::Computes);
                case OpType::Binary: {
                    // Div and Pow on an Int64 operand keep the CPU op (vkNodeGate), whose int64 result feeds
                    // the region like any other. Pow's result has its base's element type, whatever the
                    // exponent's.
                    if ((BinaryType) nd.subOp != BinaryType::Pow)
                    {
                        return everyOperand(IntegerFlow::Computes);
                    }
                    IntegerDataSlots powSlots = slots(IntegerFlow::Computes, leading);
                    powSlots.exactSlotMask    = operandSlotBit(kPowExponentSlot);
                    return powSlots;
                }
                case OpType::Reduce:
                    switch ((ReduceType) nd.subOp)
                    {
                        case ReduceType::Sum:
                        case ReduceType::Max:
                        case ReduceType::Min:
                        case ReduceType::Prod:
                            return slots(IntegerFlow::Computes, leading);
                        default:
                            return {}; // Mean and L2 are float results on integer data too (the CPU op stores fp32)
                    }
                case OpType::Clip:
                case OpType::Range:
                    return slots(IntegerFlow::Computes, threeSlots);
                case OpType::Unary:
                    switch ((UnaryType) nd.subOp)
                    {
                        case UnaryType::Neg:
                        case UnaryType::Abs:
                            return slots(IntegerFlow::Computes, leading);
                        default:
                            return {}; // the other unary functions are float math
                    }
                case OpType::Equal:
                case OpType::Greater:
                case OpType::GreaterEqual:
                case OpType::Less:
                case OpType::LessEqual:
                    return slots(IntegerFlow::Compares, leading | operandSlotBit(kSecondOperandSlot));
                default:
                    return {};
            }
        }

        // Whether a GPU kernel uploads a constant operand itself, at the node's own precision (operandBuf,
        // or a constant operand buffer of its own). The segment fills a constant operand 0 into a shared
        // activation buffer at the segment's storage precision for the kernels that read operand 0 through
        // env.devBuf, so such a kernel reads that constant correctly only while it runs at the segment's
        // precision: the NC4HW4 Concat (which reads every part that way) and Split, Reduce, DepthToSpace,
        // TopK, the layout converts and the float kernels. A kernel not listed here counts as one of them.
        bool uploadsConstantOperandsItself(const Graph &g, const Node &nd) {
            switch (nd.type)
            {
                case OpType::Reshape:
                case OpType::Flatten:
                case OpType::Squeeze:
                case OpType::Unsqueeze:
                case OpType::Cast:
                case OpType::Unary:
                case OpType::ChannelShuffle:
                case OpType::Slice:
                case OpType::Transpose:
                case OpType::Expand:
                case OpType::Tile:
                case OpType::Pad:
                case OpType::Gather:
                case OpType::ScatterND:
                case OpType::Where:
                case OpType::Clip:
                case OpType::Range:
                case OpType::Equal:
                case OpType::Greater:
                case OpType::GreaterEqual:
                case OpType::Less:
                case OpType::LessEqual:
                case OpType::ArgMax:
                case OpType::ArgMin:
                case OpType::Mod:
                case OpType::BitShift:
                case OpType::BitwiseAnd:
                case OpType::BitwiseOr:
                case OpType::BitwiseXor:
                case OpType::BitwiseNot:
                case OpType::MatMul:
                case OpType::Einsum:
                case OpType::And:
                case OpType::Or:
                case OpType::Xor:
                case OpType::Not:
                    return true;
                case OpType::Add:
                case OpType::Binary:
                case OpType::Concat:
                case OpType::Split:
                    // The flat kernels upload constants themselves; the NC4HW4 ones read the segment's buffer.
                    return !nd.outputs.empty() && nd.outputs[0] != kNoTensor && g.desc(nd.outputs[0]).gpuFlat;
                default:
                    return false;
            }
        }

        // Whether a node reads a constant operand through the segment's shared activation buffer (the slots
        // segmentFilledConstantOperandEnd names), so its output must keep the segment's storage precision.
        bool readsConstantAtSegmentPrecision(const Graph &g, const Node &nd) {
            const size_t filledEnd = segmentFilledConstantOperandEnd(g, nd);
            for (size_t slot = 0; slot < filledEnd; ++slot)
            {
                if (nd.inputs[slot] != kNoTensor && g.isInitializer(nd.inputs[slot]))
                {
                    return !uploadsConstantOperandsItself(g, nd);
                }
            }
            return false;
        }

        // Whether output `output` of `nd` holds integers whatever its operands hold: a Cast to an integer
        // type, Shape, ArgMax/ArgMin, TopK's indices, an integer-filled ConstantOfShape, the bitwise ops and
        // an integer Mod.
        bool producesIntegers(const Graph &g, const Node &nd, TensorId output, const std::vector<int> &producer) {
            if (nd.attr.has("pw_steps"))
            {
                return false;
            }
            switch (nd.type)
            {
                case OpType::Cast:
                    return onnx::castTargetsIntegerElementType(nd);
                case OpType::Shape:
                case OpType::ArgMax:
                case OpType::ArgMin:
                case OpType::BitShift:
                case OpType::BitwiseAnd:
                case OpType::BitwiseOr:
                case OpType::BitwiseXor:
                case OpType::BitwiseNot:
                    return true;
                case OpType::ConstantOfShape: {
                    // The importer records an integer fill `value` as an Ints attribute and a float fill as Floats.
                    const auto fill = nd.attr.map.find("value");
                    return fill != nd.attr.map.end() && fill->second.kind == Attr::Ints;
                }
                case OpType::TopK:
                    return nd.outputs.size() > kTopKIndicesOutput && nd.outputs[kTopKIndicesOutput] == output;
                case OpType::Mod:
                    return modComputesIntegers(g, nd, producer);
                default:
                    return false;
            }
        }
    } // namespace

    void pinIntegerResultsFp32(Graph &g) {
        // Last writer of each tensor and every (node, input slot) reading it, so the region can be
        // followed toward sources and toward consumers.
        std::vector<int>                                 producer(g.tensors.size(), -1);
        std::vector<std::vector<std::pair<int, size_t>>> readers(g.tensors.size());
        for (int nodeIndex = 0; nodeIndex < (int) g.nodes.size(); ++nodeIndex)
        {
            for (TensorId output: g.nodes[nodeIndex].outputs)
            {
                if (output != kNoTensor)
                {
                    producer[(size_t) output] = nodeIndex;
                }
            }
            for (size_t slot = 0; slot < g.nodes[nodeIndex].inputs.size(); ++slot)
            {
                if (g.nodes[nodeIndex].inputs[slot] != kNoTensor)
                {
                    readers[(size_t) g.nodes[nodeIndex].inputs[slot]].push_back({nodeIndex, slot});
                }
            }
        }
        // Whether a tensor may take fp32 storage: a runtime tensor that is flat (every flat kernel has an
        // fp32 variant), or an NC4HW4 tensor no fp16-only kernel writes -- a graph input (the boundary
        // packs it at its storage precision), the output of a layout convert, a metadata reshape (a buffer
        // copy) or a Cast (cast.comp has an fp32 variant), or of Add, Binary, Unary, Concat or ChannelShuffle,
        // whose NC4HW4 kernels have fp32 variants too, or of a Split, whose NC4HW4 kernel copies channel
        // blocks at the node's element width (the segment planner keeps an fp32 Concat or Split out of its
        // zero-copy views). The layout pass keeps an NC4HW4 input's layout through an agnostic reshape, Cast
        // or ChannelShuffle only while nc4PackingIdentical holds (the output stores every element at the
        // input's NC4HW4 position, with the same footprint), and packs a graph input flat when every read of
        // it, through such hops, is flat; a same-shape rank-4 Add/Binary of runtime operands runs NC4HW4, and
        // a channel-axis Concat or Split of 4-aligned parts does too, so an integer region often crosses such
        // tensors. The NC4HW4 conv family is hand-written fp16-only,
        // so its outputs are never pinned; markFp32 bridges them instead. In either layout, a node reading a
        // constant through the segment's shared activation buffer (readsConstantAtSegmentPrecision: an
        // NC4HW4 Concat with a constant part, among others) keeps the segment's precision, since an fp32
        // kernel would read that fp16-filled buffer as fp32.
        auto storageCanPin = [&](TensorId tensor) {
            if (tensor == kNoTensor || g.isInitializer(tensor))
            {
                return false;
            }
            const int producerIndex = producer[(size_t) tensor];
            if (producerIndex >= 0 && readsConstantAtSegmentPrecision(g, g.nodes[(size_t) producerIndex]))
            {
                return false;
            }
            if (g.desc(tensor).gpuFlat || producerIndex < 0)
            {
                return true;
            }
            const Node &producerNode = g.nodes[(size_t) producerIndex];
            if (producerNode.attr.has("pw_steps"))
            {
                return false;
            }
            switch (producerNode.type)
            {
                case OpType::ConvertLayout:
                case OpType::Reshape:
                case OpType::Flatten:
                case OpType::Squeeze:
                case OpType::Unsqueeze:
                case OpType::Cast:
                case OpType::Add:
                case OpType::Binary:
                case OpType::Unary:
                case OpType::Concat:
                case OpType::Split:
                case OpType::ChannelShuffle:
                    return true;
                default:
                    return false;
            }
        };
        int  pinned = 0;
        auto pin    = [&](TensorId tensor) {
            if (!storageCanPin(tensor) || g.desc(tensor).storeFp32)
            {
                return;
            }
            g.desc(tensor).storeFp32 = true;
            ++pinned;
        };

        // Integer-valued tensors: Int32/Int64-typed tensors (initializers included) and integer results
        // (producesIntegers), followed forward through every Moves and Computes reader. Membership alone
        // pins nothing: a value becomes part of the region where a node needs it exact (the seeds below), so
        // an integer graph input read only by a Cast to a float type keeps its storage precision.
        std::vector<char>     integerValued(g.tensors.size(), 0);
        std::vector<TensorId> unfollowed;
        auto                  markIntegerValued = [&](TensorId tensor) {
            if (tensor != kNoTensor && !integerValued[(size_t) tensor])
            {
                integerValued[(size_t) tensor] = 1;
                unfollowed.push_back(tensor);
            }
        };
        for (TensorId tensor = 0; tensor < (TensorId) g.tensors.size(); ++tensor)
        {
            if (typedWideInteger(g, tensor))
            {
                markIntegerValued(tensor);
            }
        }
        for (const Node &nd: g.nodes)
        {
            for (TensorId output: nd.outputs)
            {
                if (output != kNoTensor && producesIntegers(g, nd, output, producer))
                {
                    markIntegerValued(output);
                }
            }
        }
        while (!unfollowed.empty())
        {
            const TensorId tensor = unfollowed.back();
            unfollowed.pop_back();
            for (const auto &[readerIndex, slot]: readers[(size_t) tensor])
            {
                const Node            &readerNode = g.nodes[(size_t) readerIndex];
                const IntegerDataSlots dataSlots  = integerDataSlots(readerNode);
                if ((dataSlots.flow == IntegerFlow::Moves || dataSlots.flow == IntegerFlow::Computes) && dataSlots.readsData(slot))
                {
                    for (size_t outputIndex = 0; outputIndex < readerNode.outputs.size(); ++outputIndex)
                    {
                        if (dataSlots.writesData(outputIndex))
                        {
                            markIntegerValued(readerNode.outputs[outputIndex]);
                        }
                    }
                }
            }
        }
        auto readsIntegerValues = [&](const Node &nd, const IntegerDataSlots &dataSlots) {
            for (size_t slot = 0; slot < nd.inputs.size(); ++slot)
            {
                if (dataSlots.readsData(slot) && nd.inputs[slot] != kNoTensor && integerValued[(size_t) nd.inputs[slot]])
                {
                    return true;
                }
            }
            return false;
        };

        // How far the walk follows a tensor: an Integer tensor extends the region toward its sources and
        // its consumers; an Upstream tensor (a Cast's operand, which may hold float values, or a comparison
        // result) is followed only toward its source, so fp32 never spreads into the float readers beside
        // it. Ordered so a tensor reached both ways is walked at the wider reach.
        enum class Reach : uint8_t { None, Upstream, Integer };
        std::vector<Reach>                      reach(g.tensors.size(), Reach::None);
        std::vector<std::pair<TensorId, Reach>> pending;
        auto                                    enqueue = [&](TensorId tensor, Reach tensorReach) {
            if (tensor != kNoTensor)
            {
                pending.push_back({tensor, tensorReach});
            }
        };
        // The operands of a node computing integers: its data operands are integers, and an operand it only
        // reads exactly is followed toward its source unless it holds integers itself.
        auto enqueueIntegerOperands = [&](const Node &nd, const IntegerDataSlots &dataSlots) {
            for (size_t slot = 0; slot < nd.inputs.size(); ++slot)
            {
                const TensorId operand = nd.inputs[slot];
                if (dataSlots.readsData(slot))
                {
                    enqueue(operand, Reach::Integer);
                    continue;
                }
                if (dataSlots.readsExactlyWithoutTyping(slot))
                {
                    enqueue(operand, operand != kNoTensor && integerValued[(size_t) operand] ? Reach::Integer : Reach::Upstream);
                }
            }
        };
        // The outputs of a node holding its data's element values, at one reach.
        auto enqueueDataOutputs = [&](const Node &nd, const IntegerDataSlots &dataSlots, Reach tensorReach) {
            for (size_t outputIndex = 0; outputIndex < nd.outputs.size(); ++outputIndex)
            {
                if (dataSlots.writesData(outputIndex))
                {
                    enqueue(nd.outputs[outputIndex], tensorReach);
                }
            }
        };

        // Seeds: the integer results of the integer ops and the operands of the ops whose operands are
        // integers too; the integer data a ranking reads (ArgMax/ArgMin, TopK); every Computes node reading
        // an integer value (its result and its operands); the operands of a comparison reading an integer
        // value, with its 0/1 result pinned so the kernel compares at fp32; and every graph output holding
        // integer values, which reaches the caller exactly even when only movement ops (a Gather from an
        // int64 table, a Where or Concat of int64 inputs) lie between it and its integer sources. A constant
        // operand is uploaded by the op itself at the node's (pinned) precision, so only runtime tensors are
        // pinned.
        for (TensorId graphOutput: g.outputs)
        {
            if (graphOutput != kNoTensor && integerValued[(size_t) graphOutput])
            {
                enqueue(graphOutput, Reach::Integer);
            }
        }
        for (const Node &nd: g.nodes)
        {
            bool integerResult   = false;
            bool integerOperands = false;
            switch (nd.type)
            {
                case OpType::ArgMax:
                case OpType::ArgMin:
                    // int64 indices; the data joins only when it holds integers (a float scan needs no extra
                    // precision, so markFp32 leaves it at its own storage precision).
                    integerResult = true;
                    if (!nd.inputs.empty() && nd.inputs[0] != kNoTensor && integerValued[(size_t) nd.inputs[0]])
                    {
                        enqueue(nd.inputs[0], Reach::Integer);
                    }
                    break;
                case OpType::TopK:
                    // The ranking compares the data: integer data is read exactly (fp16 stores 70000 and
                    // 70001 as one value), and the values follow it into the region. Float scores keep their
                    // storage precision, and the indices are pinned only where a reader needs them exact.
                    if (!nd.inputs.empty() && nd.inputs[0] != kNoTensor && integerValued[(size_t) nd.inputs[0]])
                    {
                        enqueue(nd.inputs[0], Reach::Integer);
                    }
                    break;
                case OpType::Mod:
                    integerResult   = modComputesIntegers(g, nd, producer);
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
                for (TensorId output: nd.outputs)
                {
                    enqueue(output, Reach::Integer);
                }
            }
            if (integerOperands)
            {
                for (TensorId operand: nd.inputs)
                {
                    enqueue(operand, Reach::Integer);
                }
            }
            const IntegerDataSlots dataSlots = integerDataSlots(nd);
            if ((dataSlots.flow == IntegerFlow::Computes || dataSlots.flow == IntegerFlow::Compares) && readsIntegerValues(nd, dataSlots))
            {
                enqueueIntegerOperands(nd, dataSlots);
                enqueueDataOutputs(nd, dataSlots, dataSlots.flow == IntegerFlow::Computes ? Reach::Integer : Reach::Upstream);
            }
        }

        // Flood the region. Every tensor is walked at most once per reach, so the walk terminates. The
        // region stops at a tensor that cannot take fp32 storage (an initializer, or an output of the
        // fp16-only NC4HW4 conv family: markFp32's frontier convert bridges it), at a producer that
        // computes new float values (its pinned output runs fp32 via nodeFp32), and at a reader whose
        // result is float (a Cast to a float type, or any float op), in front of which markFp32 places the
        // fp32->fp16 bridge.
        while (!pending.empty())
        {
            const auto [tensor, tensorReach] = pending.back();
            pending.pop_back();
            if (!storageCanPin(tensor) || reach[(size_t) tensor] >= tensorReach)
            {
                continue;
            }
            reach[(size_t) tensor] = tensorReach;
            pin(tensor);

            const int producerIndex = producer[(size_t) tensor];
            if (producerIndex >= 0)
            {
                const Node            &producerNode = g.nodes[(size_t) producerIndex];
                const IntegerDataSlots dataSlots    = integerDataSlots(producerNode);
                const size_t outputIndex = (size_t) (std::find(producerNode.outputs.begin(), producerNode.outputs.end(), tensor) - producerNode.outputs.begin());
                // A secondary output that holds no data values (TopK's indices) says nothing about the
                // operands: the pinned output runs its node fp32 via nodeFp32.
                const IntegerFlow producerFlow = dataSlots.writesData(outputIndex) ? dataSlots.flow : IntegerFlow::None;
                switch (producerFlow)
                {
                    case IntegerFlow::Moves:
                        for (size_t slot = 0; slot < producerNode.inputs.size(); ++slot)
                        {
                            if (dataSlots.readsData(slot))
                            {
                                enqueue(producerNode.inputs[slot], tensorReach);
                            }
                        }
                        if (tensorReach == Reach::Integer)
                        {
                            enqueueDataOutputs(producerNode, dataSlots, Reach::Integer); // the other parts of a Split
                        }
                        break;
                    case IntegerFlow::Computes:
                        // An integer result has integer operands. An Upstream result (a Cast operand) may be
                        // float arithmetic, which keeps its operands at their storage precision.
                        if (tensorReach == Reach::Integer)
                        {
                            enqueueIntegerOperands(producerNode, dataSlots);
                        }
                        break;
                    case IntegerFlow::Casts:
                        for (size_t slot = 0; slot < producerNode.inputs.size(); ++slot)
                        {
                            if (dataSlots.readsData(slot))
                            {
                                enqueue(producerNode.inputs[slot], Reach::Upstream);
                            }
                        }
                        break;
                    default:
                        break; // a comparison or a float op: the pinned output runs fp32 via nodeFp32
                }
            }
            if (tensorReach != Reach::Integer)
            {
                continue;
            }
            for (const auto &[readerIndex, slot]: readers[(size_t) tensor])
            {
                const Node            &readerNode = g.nodes[(size_t) readerIndex];
                const IntegerDataSlots dataSlots  = integerDataSlots(readerNode);
                if (!dataSlots.readsData(slot))
                {
                    continue; // a parameter slot, or a float reader
                }
                Reach resultReach = Reach::Integer;
                switch (dataSlots.flow)
                {
                    case IntegerFlow::Casts:
                        if (!onnx::castTargetsIntegerElementType(readerNode))
                        {
                            continue; // a float result leaves the integer region
                        }
                        break;
                    case IntegerFlow::Compares:
                        resultReach = Reach::Upstream; // a 0/1 result, pinned without spreading
                        break;
                    default:
                        break;
                }
                enqueueDataOutputs(readerNode, dataSlots, resultReach);
                if (dataSlots.flow != IntegerFlow::Casts)
                {
                    enqueueIntegerOperands(readerNode, dataSlots); // the other parts of a Concat, the other operands
                }
            }
        }

        // markFp32 gives every output of a node outputs[0]'s precision, so a pin on a secondary output --
        // from this region, or from pinGatherIndexFp32 on a TopK's indices -- survives only if the node's
        // primary output is pinned too. A Gather reads its index as fp32 whatever its storage, so an
        // unpinned TopK would hand it fp16 bytes.
        for (const Node &nd: g.nodes)
        {
            if (nd.outputs.empty() || nd.outputs[0] == kNoTensor || g.desc(nd.outputs[0]).storeFp32)
            {
                continue;
            }
            for (size_t outputIndex = kFirstSecondaryOutput; outputIndex < nd.outputs.size(); ++outputIndex)
            {
                if (nd.outputs[outputIndex] != kNoTensor && g.desc(nd.outputs[outputIndex]).storeFp32)
                {
                    pin(nd.outputs[0]);
                    break;
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
        // Integer values (ArgMax/ArgMin indices, integer Mod, bit shifts and bitwise ops, integer arithmetic
        // and the operands of integer comparisons), and the value-preserving region around them, are exact
        // only up to 2^11 in fp16 storage; pin them to fp32 the same way before markFp32 bridges the frontier.
        pinIntegerResultsFp32(g);
        markFp32(g, fp32Marks, matchedPatterns);
        g.topoSort();
    }

} // namespace vknn
