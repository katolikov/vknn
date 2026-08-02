#include "core/matmul_tile.h"
#include "core/quant_weights.h"
#include "passes_internal.h"
#include <algorithm>
#include <map>

namespace vknn {

    // Is this tensor forced to fp32 storage by the selective-fp32 preset (Precision::Normal), or
    // already marked storeFp32 by an earlier pass run? A fused kernel stores its whole unit in one
    // dtype, so fusing across such a tensor would round an intermediate that must stay fp32 to fp16 —
    // breaking bit-exactness. Uses fp32NameMatch — the exact include/exclude rule markFp32 applies
    // at load; a no-op for models without those names (e.g. CNNs).
    static bool pwTensorIsFp32(const Graph &g, TensorId t) {
        if (t == kNoTensor)
        {
            return false;
        }
        if (g.desc(t).storeFp32)
        {
            return true;
        }
        static const std::string marks = mixedPrecisionFp32Tensors();
        return fp32NameMatch(g.desc(t).name, marks);
    }

    static bool pwFloatDtype(DType d) {
        return d == DType::Float32 || d == DType::Float16;
    }

    // Clip bounds resolved from initializer inputs (opset >= 11) or attributes (older opsets);
    // attributes win. Returns false when a bound is a RUNTIME tensor (not encodable as a step).
    static bool pwClipBounds(const Graph &g, const Node &n, float &lo, float &hi) {
        lo = -3.4e38f;
        hi = 3.4e38f;
        // A bound must be a constant scalar with a materialized payload to encode as a step; a runtime
        // tensor or an empty (unresolved rank-0) buffer is not fusable, so refuse rather than index [0].
        if (n.inputs.size() > 1 && n.inputs[1] != kNoTensor)
        {
            if (!g.isInitializer(n.inputs[1]) || g.initializers.at(n.inputs[1]).bytes.empty())
            {
                return false;
            }
            lo = g.initializers.at(n.inputs[1]).f32()[0];
        }
        if (n.inputs.size() > 2 && n.inputs[2] != kNoTensor)
        {
            if (!g.isInitializer(n.inputs[2]) || g.initializers.at(n.inputs[2]).bytes.empty())
            {
                return false;
            }
            hi = g.initializers.at(n.inputs[2]).f32()[0];
        }
        if (n.attr.has("min"))
        {
            lo = n.attr.getf("min", lo);
        }
        if (n.attr.has("max"))
        {
            hi = n.attr.getf("max", hi);
        }
        return true;
    }

    // Per-element ops eligible to join a fused-pointwise unit. Every input and the output must be
    // float-typed (comparisons on int/shape tensors and int Casts stay out), the output shape must
    // be resolved, and the output must not be fp32-pinned. The eligible OpType set is the descriptor's
    // pwMember flag; the shape/dtype/bound checks below are the per-node part.
    static bool pwEligibleNode(const Graph &g, const Node &n) {
        if (!opDescriptor(n.type).pwMember)
        {
            return false;
        }
        if (n.outputs.size() != 1 || n.outputs[0] == kNoTensor || n.inputs.empty())
        {
            return false;
        }
        const TensorDesc &od = g.desc(n.outputs[0]);
        if (od.shape.empty() || !pwFloatDtype(od.dtype) || pwTensorIsFp32(g, n.outputs[0]))
        {
            return false;
        }
        for (TensorId t: n.inputs)
        {
            if (t != kNoTensor && !pwFloatDtype(g.desc(t).dtype))
            {
                return false;
            }
        }
        if (n.type == OpType::Clip)
        {
            float lo, hi;
            if (!pwClipBounds(g, n, lo, hi))
            {
                return false; // runtime clip bounds can't encode as step params
            }
        }
        return true;
    }

    // Ops whose kernel can apply a pointwise-unit epilogue at its store (the unit folds into the
    // producer instead of a standalone node). An epilogue-capable type's GPU kernel family carries an
    // _epi variant reading pw_steps; anything else keeps the standalone FusedPointwise node. The set
    // is the descriptor's pwEpilogue flag (Conv/Gemm/MatMul/ConvGemm/ConvTranspose/FusedDwPw/Softmax/
    // LayerNorm/Reduce/GridSample/Resize/MaxPool/AvgPool/GlobalAvgPool/Transpose/Slice/Concat).
    static bool pwEpilogueCapable(OpType t) {
        return opDescriptor(t).pwEpilogue;
    }

    // Structural mirror of the Vulkan segment's zero-copy Concat test (vk_segment.cpp, "Zero-copy
    // Concat/Split"): parts that are contiguous slices of the whole in the stored byte layout can
    // become sub-buffer views into one arena, and the Concat then records NO dispatches. A concat
    // carrying pw_steps is excluded from that aliasing (its kernel must run to apply the chain), so
    // hosting a unit on an aliasable concat trades an elided node for one copy dispatch PER PART —
    // a DenseNet dense block pays hundreds of dispatches for it. The unit is cheaper standalone:
    // the parts alias, the concat vanishes, and the unit runs as ONE FusedPointwise dispatch over
    // the whole. This predicate covers the structural half only (layout, alignment, axis); the
    // planner's segment-scoped conditions (produced in-segment, liveness, fp32 pins) can still
    // refuse a view at load, which costs one extra dispatch, never correctness.
    static bool pwConcatPartsCanAlias(const Graph &g, const Node &n) {
        if (n.type != OpType::Concat || n.outputs.size() != 1 || n.inputs.empty() || n.fusedResidual != kNoTensor)
        {
            return false;
        }
        const TensorId whole = n.outputs[0];
        if (whole == kNoTensor || g.isInitializer(whole))
        {
            return false;
        }
        const Shape &ws   = g.desc(whole).shape;
        const int    rank = (int) ws.size();
        int64_t      axis = n.attr.geti("axis", 1);
        if (axis < 0)
        {
            axis += rank;
        }
        if (axis < 0 || axis >= rank || numElements(ws) <= 0)
        {
            return false;
        }
        if (gpuFlatNode(g, n))
        {
            int64_t outer = 1;
            for (int d = 0; d < (int) axis; ++d)
            {
                outer *= ws[d];
            }
            if (outer != 1)
            {
                return false; // slices interleave along an inner axis: not contiguous slabs
            }
        } else if (rank != 4 || ws[0] != 1 || axis != 1)
        { return false; }
        int64_t axisSum = 0;
        for (TensorId t: n.inputs)
        {
            // A part must come from a producing node: a graph input or initializer never gets a
            // view (the planner's producedHere test), so such a concat keeps hosting the unit.
            if (t == kNoTensor || g.isInitializer(t))
            {
                return false;
            }
            bool isGraphInput = false;
            for (TensorId gi: g.inputs)
            {
                isGraphInput = isGraphInput || gi == t;
            }
            if (isGraphInput)
            {
                return false;
            }
            const Shape &ss = g.desc(t).shape;
            if ((int) ss.size() != rank || numElements(ss) <= 0)
            {
                return false;
            }
            for (int d = 0; d < rank; ++d)
            {
                if (d == (int) axis ? ss[d] < 1 : ss[d] != ws[d])
                {
                    return false;
                }
            }
            if (!gpuFlatNode(g, n) && ss[1] % kNC4Block != 0)
            {
                return false; // an unaligned slice straddles a channel block
            }
            axisSum += ss[axis];
        }
        return axisSum == ws[axis];
    }

    // Broadcast class of tensor `t` against the unit's run shape (see the kPwBcast* constants).
    // A rank<4 CONSTANT is judged by its right-aligned rank-4 interpretation, which is exactly how
    // pwOperandBuf packs it; rank<4 RUNTIME tensors reach here already right-aligned behind the
    // explicit Reshape rightAlignPwOperands inserts (their device packing must match the reading).
    // The spatial and *Splat classes are limited to a single batch: the
    // NC4HW4 kernel recovers the pixel as vecIdx % HW, which drops the batch index, so N>1 would
    // alias batch 0's values across the whole run and stays general. The Row/Col classes carry the
    // batch in their channel-block index, so they have no such restriction. The older classes are
    // tested first, so every shape that classified before keeps its class and its encoded bytes.
    static int pwBcastClass(const Graph &g, TensorId t, const Shape &run) {
        const Shape &s = g.desc(t).shape;
        if (s == run)
        {
            return kPwBcastSame;
        }
        if (numElements(s) <= 1)
        {
            return kPwBcastScalar;
        }
        if (run.size() == 4)
        {
            Shape rs = s;
            if (g.isInitializer(t) && rs.size() < 4)
            {
                rs.insert(rs.begin(), 4 - rs.size(), 1); // right-align the constant into NCHW
            }
            if (rs.size() == 4)
            {
                if (rs[0] == run[0] && rs[1] == run[1] && rs[2] == 1 && rs[3] == 1)
                {
                    return kPwBcastChannel;
                }
                if (run[0] == 1 && rs[0] == 1 && rs[1] == 1 && rs[2] == run[2] && rs[3] == run[3])
                {
                    return kPwBcastSpatial;
                }
                if (rs[0] == run[0] && rs[1] == run[1] && rs[2] == run[2] && rs[3] == 1)
                {
                    return kPwBcastRow;
                }
                if (rs[0] == run[0] && rs[1] == run[1] && rs[2] == 1 && rs[3] == run[3])
                {
                    return kPwBcastCol;
                }
                if (run[0] == 1 && rs[0] == 1 && rs[1] == 1 && rs[2] == run[2] && rs[3] == 1)
                {
                    return kPwBcastRowSplat;
                }
                if (run[0] == 1 && rs[0] == 1 && rs[1] == 1 && rs[2] == 1 && rs[3] == run[3])
                {
                    return kPwBcastColSplat;
                }
                // Any remaining 1-or-full axis mask has a closed-form packed index: buildPwPlan
                // derives per-axis vec4-space strides from the operand shape (zero on broadcast
                // axes), so the NC4HW4 kernel addresses it without the flat div/mod walk. Tested
                // after every named class, so shapes that classified before keep their encodings.
                bool oneOrFull = rs.size() == kNchwRank;
                for (size_t k = 0; oneOrFull && k < kNchwRank; ++k)
                {
                    oneOrFull = rs[k] == 1 || rs[k] == run[k];
                }
                if (oneOrFull)
                {
                    return kPwBcastPacked;
                }
            }
        }
        return kPwBcastGeneral;
    }

    namespace {

        // One planned fused unit: the encoded program plus everything emission needs. Budgets
        // (steps/operands/registers/exports) are enforced during planning; a prefix that fits is
        // emitted and the remaining members re-enter the pool as seeds for the next unit.
        struct PwUnit {
            std::vector<int64_t>  steps;    // 8 ints per step
            std::vector<float>    params;   // 2 floats per step
            std::vector<TensorId> operands; // node.inputs[1+k] (standalone) / appended at opbase (attach)
            std::vector<TensorId> exports;  // node.outputs[1+o]
            std::vector<int64_t>  outSteps; // pw_outs: emitted step index (or kPwRefEntry) per export
            TensorId              entry   = kNoTensor;
            TensorId              mainOut = kNoTensor;
            /// First operand classified kPwBcastGeneral, the one that set nc4Ok false (see
            /// warnFlatForcedUnit). kNoTensor when the unit has none.
            TensorId generalOperand = kNoTensor;
            bool     nc4Ok = false, flatOk = false;
            bool     ok = false;
        };

        // Plan a unit over `members` (node indices, ascending = emission order) with run shape
        // `run`. Values flow: each member's result lands in the accumulator; a result consumed only
        // by the immediately following member rides the accumulator, anything with a later or
        // second internal reader takes one of kPwMaxRegs registers, and a member that must first
        // LOAD extra broadcast operands cannot receive its predecessor through the accumulator
        // (loads pass through it), so those predecessors take registers too. External readers of a
        // member value (or a graph-output use) make it an export.
        struct PwPlanner {
            const Graph             &g;
            const Shape             &run;
            const std::vector<int>  &producer;
            const std::vector<int>  &consumerCount;
            const std::vector<char> &isGraphOut;
            bool                     strict;

            PwPlanner(const Graph &g_, const Shape &run_, const std::vector<int> &prod, const std::vector<int> &cc, const std::vector<char> &go, bool strict_):
                g(g_), run(run_), producer(prod), consumerCount(cc), isGraphOut(go), strict(strict_) {
            }

            // The member's data inputs (excluding Clip bound inputs, which encode as params).
            static std::vector<TensorId> dataInputs(const Node &n) {
                if (n.type == OpType::Clip || n.type == OpType::Relu || n.type == OpType::Unary)
                {
                    return {n.inputs[0]};
                }
                return std::vector<TensorId>(n.inputs.begin(), n.inputs.end());
            }

            // External full-size runtime inputs that can stream as the unit's entry, in member
            // (emission) order. The caller plans once per candidate: the entry choice decides which
            // producer an epilogue attach can target.
            std::vector<TensorId> entryCandidates(const std::vector<int> &members) const {
                std::set<int>         memberSet(members.begin(), members.end());
                std::vector<TensorId> cands;
                for (int m: members)
                {
                    for (TensorId t: dataInputs(g.nodes[m]))
                    {
                        int p = (t >= 0 && t < (TensorId) producer.size()) ? producer[t] : -1;
                        if (t != kNoTensor && !(p >= 0 && memberSet.count(p)) && !g.isInitializer(t) && g.desc(t).shape == run && !pwTensorIsFp32(g, t) &&
                            std::find(cands.begin(), cands.end(), t) == cands.end())
                        {
                            cands.push_back(t);
                        }
                    }
                }
                return cands;
            }

            PwUnit plan(const std::vector<int> &members, TensorId entry) {
                PwUnit        u;
                std::set<int> memberSet(members.begin(), members.end());
                // internal readers per member value (member-granular), and reads of each tensor
                // by unit members (to detect external readers of the entry later)
                std::map<TensorId, std::vector<int>> internalReaders; // value tensor -> member node idx list
                std::map<TensorId, int>              unitReads;       // tensor -> read count inside the unit
                for (int m: members)
                {
                    for (TensorId t: dataInputs(g.nodes[m]))
                    {
                        if (t == kNoTensor)
                        {
                            continue;
                        }
                        unitReads[t]++;
                        int p = (t >= 0 && t < (TensorId) producer.size()) ? producer[t] : -1;
                        if (p >= 0 && memberSet.count(p))
                        {
                            internalReaders[g.nodes[p].outputs[0]].push_back(m);
                        }
                    }
                }

                u.entry = entry;
                if (u.entry == kNoTensor)
                {
                    return u; // nothing to stream through the kernel
                }

                // Register planning (member-granular). loadsOf[m] = extra broadcast operands that
                // need LOAD steps; a member with loads clobbers the accumulator before its own step.
                std::map<int, int> loadsOf;
                for (int m: members)
                {
                    int bcastOperands = 0;
                    for (TensorId t: dataInputs(g.nodes[m]))
                    {
                        int  p        = (t >= 0 && t < (TensorId) producer.size()) ? producer[t] : -1;
                        bool internal = p >= 0 && memberSet.count(p);
                        if (!internal && t != u.entry && t != kNoTensor && pwBcastClass(g, t, run) != 0)
                        {
                            bcastOperands++;
                        }
                    }
                    loadsOf[m] = bcastOperands > 1 ? bcastOperands - 1 : 0;
                }
                // Which member values need a register: any internal-read value that is not
                // "single reader == the immediately next member, and that member has no loads".
                std::map<TensorId, int> regOf; // value tensor -> register
                {
                    std::vector<char> regFree(kPwMaxRegs, 1);
                    // free registers at the right time: track last internal reader per value
                    std::map<int, std::vector<TensorId>> freeAfter; // member -> values whose last read is here
                    std::vector<TensorId>                needsReg;
                    for (int mi = 0; mi < (int) members.size(); ++mi)
                    {
                        int      m   = members[mi];
                        TensorId val = g.nodes[m].outputs[0];
                        auto     it  = internalReaders.find(val);
                        if (it == internalReaders.end())
                        {
                            continue;
                        }
                        const auto &rd      = it->second;
                        bool        accOnly = rd.size() == 1 && mi + 1 < (int) members.size() && rd[0] == members[mi + 1] && loadsOf[members[mi + 1]] == 0;
                        if (!accOnly)
                        {
                            int last = rd.back();
                            for (int r: rd)
                            {
                                last = std::max(last, r);
                            }
                            freeAfter[last].push_back(val);
                            needsReg.push_back(val);
                        }
                    }
                    for (int mi = 0; mi < (int) members.size(); ++mi)
                    {
                        int      m   = members[mi];
                        TensorId val = g.nodes[m].outputs[0];
                        if (std::find(needsReg.begin(), needsReg.end(), val) != needsReg.end())
                        {
                            int r = -1;
                            for (int k = 0; k < kPwMaxRegs; ++k)
                            {
                                if (regFree[k])
                                {
                                    r = k;
                                    break;
                                }
                            }
                            if (r < 0)
                            {
                                return u; // register pressure: caller retries a shorter prefix
                            }
                            regFree[r] = 0;
                            regOf[val] = r;
                        }
                        auto fa = freeAfter.find(m);
                        if (fa != freeAfter.end())
                        {
                            for (TensorId v: fa->second)
                            {
                                regFree[regOf[v]] = 1;
                            }
                        }
                    }
                }

                // Emission. Transient LOAD results reuse a scratch register that is dead between
                // members; loads bind it just before the consuming step.
                auto operandRef = [&](TensorId t) -> int64_t {
                    for (size_t k = 0; k < u.operands.size(); ++k)
                    {
                        if (u.operands[k] == t)
                        {
                            return kPwRefOp0 - (1 + (int64_t) k);
                        }
                    }
                    u.operands.push_back(t);
                    return kPwRefOp0 - (int64_t) u.operands.size();
                };
                bool                        hasClass2 = false;
                TensorId                    accVal    = kNoTensor; // which member value the accumulator holds
                std::map<TensorId, int64_t> emittedStepOf;
                for (int mi = 0; mi < (int) members.size(); ++mi)
                {
                    int         m  = members[mi];
                    const Node &nd = g.nodes[m];
                    auto        di = dataInputs(nd);

                    // resolve each data input to a ref + (for operands) its broadcast class
                    struct Src {
                        int64_t ref     = kPwRefNone;
                        int     bc      = 0;
                        bool    operand = false;
                    };
                    std::vector<Src> src(di.size());
                    int              bcastLeft = 1; // one strided operand per step; extras LOAD first
                    // scratch register for loads: any register free during this member
                    for (size_t k = 0; k < di.size(); ++k)
                    {
                        TensorId t = di[k];
                        int      p = (t >= 0 && t < (TensorId) producer.size()) ? producer[t] : -1;
                        if (p >= 0 && memberSet.count(p))
                        {
                            TensorId val = g.nodes[p].outputs[0];
                            auto     rg  = regOf.find(val);
                            if (rg != regOf.end())
                            {
                                src[k].ref = kPwRefReg0 - rg->second;
                            } else if (accVal == val)
                            {
                                src[k].ref = kPwRefAcc;
                            } else
                            {
                                return PwUnit {}; // planning bug guard: value neither in acc nor a register
                            }
                        } else if (t == u.entry)
                        {
                            src[k].ref = kPwRefEntry;
                        } else
                        {
                            src[k].operand = true;
                            src[k].bc      = pwBcastClass(g, t, run);
                            if (pwTensorIsFp32(g, t))
                            {
                                return PwUnit {};
                            }
                            src[k].ref = operandRef(t);
                            if (src[k].bc == kPwBcastGeneral)
                            {
                                hasClass2 = true;
                                if (u.generalOperand == kNoTensor)
                                {
                                    u.generalOperand = t;
                                }
                            }
                        }
                    }
                    // LOAD steps for extra broadcast operands (beyond the one the step carries).
                    // Scratch registers are dead outside this member; regs planned for live values
                    // stay reserved (conservatively for the whole unit).
                    std::set<int> scratchUsed;
                    for (auto &rv: regOf)
                    {
                        scratchUsed.insert(rv.second);
                    }
                    for (size_t k = 0; k < di.size(); ++k)
                    {
                        if (!src[k].operand || src[k].bc == 0)
                        {
                            continue;
                        }
                        if (bcastLeft > 0)
                        {
                            bcastLeft--;
                            continue;
                        }
                        int scratch = -1;
                        for (int r = 0; r < kPwMaxRegs; ++r)
                        {
                            if (!scratchUsed.count(r))
                            {
                                scratch = r;
                                break;
                            }
                        }
                        if (scratch < 0 || (int) u.steps.size() / 8 >= kPwMaxSteps)
                        {
                            return PwUnit {};
                        }
                        scratchUsed.insert(scratch);
                        u.steps.insert(u.steps.end(), {kPwKindLoad, 0, src[k].ref, kPwRefNone, kPwRefNone, (int64_t) scratch, (int64_t) src[k].bc, 1});
                        u.params.insert(u.params.end(), {0, 0});
                        src[k].ref     = kPwRefReg0 - scratch;
                        src[k].operand = false;
                        src[k].bc      = 0;
                    }

                    // the member's own step
                    int64_t kind = 0, code = 0;
                    float   p0 = 0, p1 = 0;
                    int64_t sA = kPwRefNone, sB = kPwRefNone, sC = kPwRefNone;
                    switch (nd.type)
                    {
                        case OpType::Add:
                            kind = kPwKindBinary;
                            code = (int64_t) BinaryType::Add;
                            sA   = src[0].ref;
                            sB   = src[1].ref;
                            break;
                        case OpType::Binary:
                            kind = kPwKindBinary;
                            code = nd.subOp;
                            sA   = src[0].ref;
                            sB   = src[1].ref;
                            break;
                        case OpType::PRelu:
                            kind = kPwKindBinary;
                            code = kPwBinPRelu;
                            sA   = src[0].ref;
                            sB   = src[1].ref;
                            break;
                        case OpType::Greater:
                            kind = kPwKindBinary;
                            code = kPwBinGreater;
                            sA   = src[0].ref;
                            sB   = src[1].ref;
                            break;
                        case OpType::GreaterEqual:
                            kind = kPwKindBinary;
                            code = kPwBinGreaterEqual;
                            sA   = src[0].ref;
                            sB   = src[1].ref;
                            break;
                        case OpType::Less:
                            kind = kPwKindBinary;
                            code = kPwBinLess;
                            sA   = src[0].ref;
                            sB   = src[1].ref;
                            break;
                        case OpType::LessEqual:
                            kind = kPwKindBinary;
                            code = kPwBinLessEqual;
                            sA   = src[0].ref;
                            sB   = src[1].ref;
                            break;
                        case OpType::Equal:
                            kind = kPwKindBinary;
                            code = kPwBinEqual;
                            sA   = src[0].ref;
                            sB   = src[1].ref;
                            break;
                        case OpType::Where:
                            kind = kPwKindSelect;
                            sA   = src[0].ref;
                            sB   = src[1].ref;
                            sC   = src[2].ref;
                            break;
                        case OpType::Unary:
                            kind = kPwKindUnary;
                            code = nd.subOp;
                            p0   = nd.actLo;
                            p1   = nd.actHi;
                            sA   = src[0].ref;
                            break;
                        case OpType::Relu:
                            kind = kPwKindAct;
                            code = (int64_t) ActType::Relu;
                            sA   = src[0].ref;
                            break;
                        case OpType::Clip: {
                            kind = kPwKindAct;
                            code = (int64_t) ActType::Clip;
                            sA   = src[0].ref;
                            pwClipBounds(g, nd, p0, p1);
                            break;
                        }
                        default:
                            return PwUnit {};
                    }
                    // the ONE strided operand this step carries (bcastSrc marks the field)
                    int64_t bc = 0, bsrc = 0;
                    for (size_t k = 0; k < di.size(); ++k)
                    {
                        if (src[k].operand && src[k].bc != 0)
                        {
                            bc = src[k].bc;
                            // field of the ref: srcA=1, srcB=2, srcC=3 (Clip/Relu/Unary have 1 input)
                            bsrc = (int64_t) k + 1;
                        }
                    }
                    if ((int) u.steps.size() / 8 >= kPwMaxSteps)
                    {
                        return PwUnit {};
                    }
                    TensorId val = nd.outputs[0];
                    auto     rg  = regOf.find(val);
                    int64_t  dst = rg != regOf.end() ? (int64_t) rg->second : (int64_t) kPwRefNone;
                    // Fast mode folds a swish diamond into ONE step: [unary Sigmoid/HardSigmoid X]
                    // followed by [Mul(X, acc)] becomes [unary SiLU/HardSwish X] — one expression,
                    // one VM step, matching the retired fuseSwish's cost. Bytes intentionally
                    // differ from the two rounded steps; --strict-fuse keeps those.
                    if (!strict && kind == kPwKindBinary && code == (int64_t) BinaryType::Mul && u.steps.size() >= 8 && nd.fusedAct == ActType::None)
                    {
                        size_t  ps    = u.steps.size() - 8;
                        int64_t pKind = u.steps[ps], pCode = u.steps[ps + 1], pSrcA = u.steps[ps + 2], pDst = u.steps[ps + 5];
                        bool    sig = pKind == kPwKindUnary && pCode == (int64_t) UnaryType::Sigmoid;
                        bool hsig = pKind == kPwKindUnary && pCode == (int64_t) UnaryType::HardSigmoid && u.params[u.params.size() - 2] == 1.0f / 6.0f && u.params.back() == 0.5f;
                        // the gate value must feed ONLY this Mul (accumulator-carried, no register,
                        // no export) and the Mul's other source must be the gated value itself
                        bool gateOk  = (sig || hsig) && pDst == kPwRefNone && accVal != kNoTensor && consumerCount[accVal] == 1;
                        bool diamond = gateOk && ((sA == pSrcA && sB == kPwRefAcc) || (sB == pSrcA && sA == kPwRefAcc));
                        if (diamond)
                        {
                            int64_t pBc = u.steps[ps + 6], pBsrc = u.steps[ps + 7];
                            u.steps.resize(ps);
                            u.params.resize(u.params.size() - 2);
                            u.steps.insert(u.steps.end(), {kPwKindUnary, (int64_t) (sig ? UnaryType::SiLU : UnaryType::HardSwish), pSrcA, kPwRefNone, kPwRefNone, dst, pBc, pBsrc});
                            u.params.insert(u.params.end(), {0, 0});
                            emittedStepOf[val] = (int64_t) u.steps.size() / 8 - 1;
                            accVal             = val;
                            continue;
                        }
                    }
                    u.steps.insert(u.steps.end(), {kind, code, sA, sB, sC, nd.fusedAct == ActType::None ? dst : (int64_t) kPwRefNone, bc, bsrc});
                    u.params.insert(u.params.end(), {p0, p1});
                    // a fused activation folded onto the member (fuseActivations targets Add) is an
                    // invisible extra op: encode it or it is silently dropped with the node
                    if (nd.fusedAct != ActType::None)
                    {
                        if ((int) u.steps.size() / 8 >= kPwMaxSteps)
                        {
                            return PwUnit {};
                        }
                        u.steps.insert(u.steps.end(), {kPwKindAct, (int64_t) nd.fusedAct, kPwRefAcc, kPwRefNone, kPwRefNone, dst, 0, 0});
                        u.params.insert(u.params.end(), {nd.actLo, nd.actHi});
                    }
                    emittedStepOf[val] = (int64_t) u.steps.size() / 8 - 1;
                    accVal             = val;
                }

                if ((int) u.operands.size() > kPwMaxOperands)
                {
                    return PwUnit {};
                }

                // Exports: member values with readers outside the unit or a graph-output use; the
                // last member's value is the node's main output and needs no export slot.
                u.mainOut = g.nodes[members.back()].outputs[0];
                for (int m: members)
                {
                    TensorId val      = g.nodes[m].outputs[0];
                    int      internal = 0;
                    auto     ir       = internalReaders.find(val);
                    if (ir != internalReaders.end())
                    {
                        internal = (int) ir->second.size();
                    }
                    bool external = consumerCount[val] > internal || (val < (TensorId) isGraphOut.size() && isGraphOut[val]);
                    if (external && val != u.mainOut)
                    {
                        if ((int) u.exports.size() >= kPwMaxOuts)
                        {
                            return PwUnit {};
                        }
                        u.exports.push_back(val);
                        u.outSteps.push_back(emittedStepOf[val]);
                    }
                }

                u.nc4Ok  = run.size() == 4 && !hasClass2;
                u.flatOk = (int) run.size() <= kPwMaxRank;
                if (!u.nc4Ok && !u.flatOk)
                {
                    return PwUnit {};
                }
                u.ok = true;
                return u;
            }
        };

        /// Report a rank-4 unit that a general-broadcast operand pushed onto the flat kernel. Such a
        /// unit gives up vec4 addressing for the WHOLE region (one element per thread instead of
        /// four), pays a per-axis integer div/mod walk per element per step that reads the operand,
        /// and is excluded from the flexible layout re-vote, so a ConvertLayout appears on each of
        /// its full-size edges. The message names the anchor node and the operand shape that caused
        /// it, which is the whole diagnosis for a model whose bytes are not available to inspect.
        void warnFlatForcedUnit(const Graph &g, const Node &anchor, TensorId general, const Shape &run, bool nc4Ok) {
            if (nc4Ok || general == kNoTensor || run.size() != 4)
            {
                return; // NC4HW4-expressible, or a rank the blocked path has no form for anyway
            }
            VKNN_INFO << "fusePointwiseChains: unit at '" << anchor.name << "' runs " << shapeStr(run) << " on the flat kernel -- operand '" << g.desc(general).name << "' "
                      << shapeStr(g.desc(general).shape) << " has no blocked-layout index, so the whole unit loses vec4 and picks up a layout convert on each side";
        }

        /// Rewire every rank<4 RUNTIME operand of a rank-4 pointwise run through an explicit Reshape
        /// to its right-aligned rank-4 NumPy interpretation ([H,W] -> [1,1,H,W], [C,1,1] ->
        /// [1,C,1,1]). pwBcastClass may only judge a tensor by right-alignment when its device bytes
        /// follow that reading: initializers do (pwOperandBuf packs them right-aligned at upload),
        /// but a runtime tensor's NC4HW4 packing follows NCHW::from's LEFT-aligned rank<4 mapping,
        /// so without the Reshape such operands classify kPwBcastGeneral and flat-force their unit.
        /// The Reshape puts the right-aligned reading into the graph itself: the layout passes then
        /// convert the small operand into the unit's world instead of dropping the whole unit to the
        /// flat kernel. Reshape is layout-agnostic and elides to a buffer alias at record time, so
        /// the node adds nothing to the data path. Only readers in the pw-eligible set are rewired;
        /// every other consumer keeps the original tensor.
        void rightAlignPwOperands(Graph &g) {
            struct PendingReshape {
                Node     node;
                TensorId source;
            };
            std::map<TensorId, TensorId> aligned; // original tensor -> its rank-4 view
            std::vector<PendingReshape>  pending;
            for (auto &nd: g.nodes)
            {
                if (!pwEligibleNode(g, nd) || PwPlanner::dataInputs(nd).size() < 2)
                {
                    continue;
                }
                const Shape run = g.desc(nd.outputs[0]).shape;
                if (run.size() != kNchwRank)
                {
                    continue;
                }
                for (TensorId &in: nd.inputs)
                {
                    if (in == kNoTensor || g.isInitializer(in) || pwTensorIsFp32(g, in))
                    {
                        continue;
                    }
                    const Shape &s = g.desc(in).shape;
                    if (s.empty() || s.size() >= kNchwRank || numElements(s) <= 1)
                    {
                        continue;
                    }
                    Shape rs = s;
                    rs.insert(rs.begin(), kNchwRank - rs.size(), 1);
                    bool broadcastLegal = true;
                    for (size_t k = 0; k < kNchwRank; ++k)
                    {
                        broadcastLegal = broadcastLegal && (rs[k] == 1 || rs[k] == run[k]);
                    }
                    if (!broadcastLegal)
                    {
                        continue; // malformed broadcast: leave it for the plan builder's diagnostics
                    }
                    auto it = aligned.find(in);
                    if (it == aligned.end())
                    {
                        TensorDesc d;
                        d.name        = g.desc(in).name + "#pwr4";
                        d.shape       = rs;
                        d.dtype       = g.desc(in).dtype;
                        TensorId view = g.addTensor(d);

                        TensorDesc sd;
                        sd.name            = d.name + "#shape";
                        sd.shape           = {(int64_t) kNchwRank};
                        sd.dtype           = DType::Int64;
                        sd.isInitializer   = true;
                        TensorId   shapeId = g.addTensor(sd);
                        HostBuffer hb;
                        hb.resizeElems((int64_t) kNchwRank, DType::Int64);
                        for (size_t k = 0; k < kNchwRank; ++k)
                        {
                            hb.i64()[k] = rs[k];
                        }
                        g.initializers[shapeId] = std::move(hb);

                        PendingReshape pr;
                        pr.node.type    = OpType::Reshape;
                        pr.node.name    = d.name;
                        pr.node.inputs  = {in, shapeId};
                        pr.node.outputs = {view};
                        pr.source       = in;
                        pending.push_back(std::move(pr));
                        it = aligned.emplace(in, view).first;
                    }
                    in = it->second;
                }
            }
            if (pending.empty())
            {
                return;
            }
            // Splice each Reshape right after its source's producer (graph inputs at the front), so
            // node order stays topological for the index-interval convexity walk below. Insertions
            // run back-to-front so earlier positions stay valid.
            std::map<TensorId, size_t> producerAt;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                for (TensorId o: g.nodes[i].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producerAt[o] = i;
                    }
                }
            }
            std::stable_sort(pending.begin(), pending.end(), [&](const PendingReshape &a, const PendingReshape &b) {
                auto   pa = producerAt.find(a.source), pb = producerAt.find(b.source);
                size_t ia = pa == producerAt.end() ? 0 : pa->second + 1;
                size_t ib = pb == producerAt.end() ? 0 : pb->second + 1;
                return ia < ib;
            });
            for (size_t i = pending.size(); i-- > 0;)
            {
                auto   pa  = producerAt.find(pending[i].source);
                size_t pos = pa == producerAt.end() ? 0 : pa->second + 1;
                g.nodes.insert(g.nodes.begin() + (long) pos, std::move(pending[i].node));
            }
            VKNN_INFO << "fusePointwiseChains: right-aligned " << pending.size() << " rank<4 runtime operand(s) behind Reshape views";
        }

        // ---- Rank collapse for pointwise runs beyond the plan's rank budget --------------------

        /// Right-aligned rank-`rank` reading of `s` (leading axes filled with 1) -- the NumPy
        /// broadcast interpretation the axis-grouping rule and the grouped-view shapes use.
        Shape pwAlignShape(const Shape &s, size_t rank) {
            Shape aligned = s;
            aligned.insert(aligned.begin(), rank - aligned.size(), 1);
            return aligned;
        }

        /// A node the rank-collapse pre-pass may rewire: pointwise-fusable, output rank above
        /// kPwMaxRank with a positive element count, and every data edge present, un-pinned, and
        /// (when non-scalar) a 1-or-full broadcast of the run. A malformed broadcast (an axis
        /// neither 1 nor the run's extent) has no per-group closed form, so its node stays at full
        /// rank and keeps the pre-collapse diagnostics.
        bool pwRankCollapsible(const Graph &g, const Node &n) {
            if (!pwEligibleNode(g, n))
            {
                return false;
            }
            const Shape &run = g.desc(n.outputs[0]).shape;
            if ((int) run.size() <= kPwMaxRank || numElements(run) <= 0)
            {
                return false;
            }
            for (TensorId t: PwPlanner::dataInputs(n))
            {
                if (t == kNoTensor || pwTensorIsFp32(g, t))
                {
                    return false;
                }
                const Shape &s = g.desc(t).shape;
                if (numElements(s) <= 1)
                {
                    continue; // scalars are shape-independent and ride along untouched
                }
                if (s.size() > run.size())
                {
                    return false;
                }
                const Shape aligned = pwAlignShape(s, run.size());
                for (size_t k = 0; k < run.size(); ++k)
                {
                    if (aligned[k] != 1 && aligned[k] != run[k])
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        /// Record the axis-grouping cuts one operand forces on the run: walking the run's axes left
        /// to right, a cut lands immediately left of every non-trivial axis whose state (broadcast
        /// 1 vs the run's full extent) differs from the previous non-trivial axis. Run axes of
        /// extent 1 are trivial -- 1 and full coincide there, so they constrain nothing and attach
        /// to the group on their left. This canonical cut placement makes the grouping a pure
        /// function of the shapes. cutAfter[k] set means axes k and k+1 may not share a group.
        void pwMarkGroupCuts(const Shape &aligned, const Shape &run, std::vector<char> &cutAfter) {
            bool havePrev = false, prevFull = false;
            for (size_t k = 0; k < run.size(); ++k)
            {
                if (run[k] == 1)
                {
                    continue;
                }
                const bool full = aligned[k] == run[k];
                if (havePrev && full != prevFull)
                {
                    cutAfter[k - 1] = 1;
                }
                havePrev = true;
                prevFull = full;
            }
        }

        /// Per-group extent products of `aligned` under the cut set (group boundaries fall after
        /// every set cutAfter[k]). Within one group an operand is uniformly broadcast or uniformly
        /// full (pwMarkGroupCuts places a cut between any two different-state axes), so each
        /// product is 1 or the run group's extent -- a NumPy-legal broadcast of the grouped run.
        Shape pwGroupShape(const Shape &aligned, const std::vector<char> &cutAfter) {
            Shape   grouped;
            int64_t extent = aligned[0];
            for (size_t k = 1; k < aligned.size(); ++k)
            {
                if (cutAfter[k - 1])
                {
                    grouped.push_back(extent);
                    extent = aligned[k];
                } else
                {
                    extent *= aligned[k];
                }
            }
            grouped.push_back(extent);
            return grouped;
        }

        /// Give each grouped constant view its bytes, once the region rewiring is complete.
        ///
        /// A grouped view is a reshaped reading of the SAME payload, so the view TAKES the source's
        /// bytes whenever the collapse consumed the source's last reference -- the common case,
        /// where duplicating them would double a weight's host footprint and its serialized size
        /// for nothing. A source some other consumer still reads at the full rank keeps its own
        /// payload and its view gets a duplicate: two shapes, two tensors, both live. Reachability
        /// is pruneDeadInitializers' rule (node inputs, the fused residual/bias edges, the
        /// quantized-weight side tensors named by attributes, and the graph's input/output lists),
        /// so a source this drops is exactly one that pass would drop.
        ///
        /// @param groupedConstants (source constant, grouped view) in creation order; a source with
        ///                         several views keeps its bytes for all but the last.
        void attachGroupedConstantPayloads(Graph &g, const std::vector<std::pair<TensorId, TensorId>> &groupedConstants) {
            if (groupedConstants.empty())
            {
                return;
            }
            std::set<TensorId> referenced(g.outputs.begin(), g.outputs.end());
            referenced.insert(g.inputs.begin(), g.inputs.end());
            for (const Node &nd: g.nodes)
            {
                referenced.insert(nd.inputs.begin(), nd.inputs.end());
                referenced.insert(nd.fusedResidual);
                referenced.insert(nd.fusedBias);
                if (nd.attr.has(kWq))
                {
                    for (const char *key: {kWqScales, kWqOidx, kWqOval, kWqLut})
                    {
                        referenced.insert((TensorId) nd.attr.geti(key, kNoTensor));
                    }
                }
            }
            std::map<TensorId, int> viewsLeft;
            for (const auto &pair: groupedConstants)
            {
                viewsLeft[pair.first]++;
            }
            for (const auto &pair: groupedConstants)
            {
                const TensorId source = pair.first, view = pair.second;
                const bool     lastView = --viewsLeft[source] == 0;
                if (lastView && !referenced.count(source))
                {
                    g.initializers[view]         = std::move(g.initializers.at(source));
                    g.desc(source).isInitializer = false;
                    g.initializers.erase(source);
                } else
                {
                    // The source outlives this view — another grouping wants it at a different
                    // shape, or a reader outside the region still names it. The two tensors differ
                    // only in shape, so they SHARE one payload: a copy would duplicate every weight
                    // byte in host memory and again in the artifact. Both sides stay
                    // copy-on-write, so a later mutation on either takes its own copy.
                    auto shared = g.initializers.at(source).bytes.shareBytes();
                    if (shared)
                    {
                        g.initializers[view].bytes.setSharedBytes(std::move(shared));
                    } else
                    {
                        // File-backed or empty: copying the buffer already copies a handle, not
                        // bytes.
                        g.initializers[view] = g.initializers.at(source);
                    }
                }
            }
        }

        /// Collapse every maximal same-run-shape pointwise region whose rank exceeds kPwMaxRank
        /// onto the coarsest axis grouping ALL of its members' data inputs admit, so the region
        /// becomes fusable: PwPlanner refuses any rank>kPwMaxRank run outright (no flat form, no
        /// NC4 form), which otherwise decays a rank-5/6 elementwise block into per-op dispatches.
        ///
        /// Rewiring keeps the grouped shape interior to the region: each member's output moves to
        /// a fresh "#pwrc" tensor at the grouped rank and intra-region edges connect DIRECTLY at
        /// that shape; boundary inputs enter through collapse Reshape views (an initializer
        /// instead becomes a grouped-shape constant carrying the source's bytes -- no runtime node,
        /// and no second payload unless another consumer still reads the source at full rank); and the
        /// original full-rank output tensor -- its id, name, and desc untouched, so graph outputs
        /// keep their contract -- is regenerated by an expand Reshape only where a reader outside
        /// the region (or a graph-output use) still needs it. Reshape is layout-agnostic and
        /// aliases at record, so every view is free on the data path. Regions with no legal
        /// grouping within the budget are left at full rank, unchanged.
        void collapsePwRunRanks(Graph &g) {
            const size_t                  nodeCount = g.nodes.size();
            std::vector<int>              producer(g.tensors.size(), -1);
            std::vector<std::vector<int>> readerNodes(g.tensors.size());
            for (size_t i = 0; i < nodeCount; ++i)
            {
                for (TensorId o: g.nodes[i].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producer[o] = (int) i;
                    }
                }
                // Reads include the fused residual/bias edges: they are tensor references OUTSIDE
                // the inputs list (rewireTensor's contract), and the dead-code passes count them as
                // live uses. A member value read only through one of them is still an external use,
                // and without the expand Reshape it would be left with no producer at all.
                auto addRead = [&](TensorId t) {
                    if (t != kNoTensor && t < (TensorId) readerNodes.size())
                    {
                        readerNodes[t].push_back((int) i);
                    }
                };
                for (TensorId t: g.nodes[i].inputs)
                {
                    addRead(t);
                }
                addRead(g.nodes[i].fusedResidual);
                addRead(g.nodes[i].fusedBias);
            }
            std::vector<char> isGraphOut(g.tensors.size(), 0);
            for (TensorId go: g.outputs)
            {
                if (go != kNoTensor)
                {
                    isGraphOut[go] = 1;
                }
            }

            struct PendingReshape {
                Node   node;
                size_t pos;  // node-list index the Reshape lands at (pre-insertion index space)
                int    tier; // expands (0) precede collapses (1) at one pos: a collapse may read an expand's output
            };
            std::vector<PendingReshape>                    pending;
            std::map<std::pair<TensorId, Shape>, TensorId> collapsedView; // (source, grouped shape) -> view
            // (source constant, its grouped copy), in creation order. The payload is attached after
            // the rewiring, when the source's surviving references are known.
            std::vector<std::pair<TensorId, TensorId>> groupedConstants;
            std::vector<char>                          visited(nodeCount, 0);
            int                                        regions = 0, collapsedNodes = 0;

            auto shapeInitializer = [&g](const std::string &name, const Shape &s) {
                TensorDesc sd;
                sd.name          = name;
                sd.shape         = {(int64_t) s.size()};
                sd.dtype         = DType::Int64;
                sd.isInitializer = true;
                TensorId   id    = g.addTensor(sd);
                HostBuffer hb;
                hb.resizeElems((int64_t) s.size(), DType::Int64);
                for (size_t k = 0; k < s.size(); ++k)
                {
                    hb.i64()[k] = s[k];
                }
                g.initializers[id] = std::move(hb);
                return id;
            };

            for (size_t seed = 0; seed < nodeCount; ++seed)
            {
                if (visited[seed] || !pwRankCollapsible(g, g.nodes[seed]))
                {
                    continue;
                }
                const Shape run = g.desc(g.nodes[seed].outputs[0]).shape;

                // ---- maximal same-run-shape collapsible region over def-use edges, fanout included
                // (the same connectivity the fusion's component walk uses, so a region collapses to
                // ONE grouping and its interior edges never need a reshape) ----
                std::set<int>    region {(int) seed};
                std::vector<int> work {(int) seed};
                while (!work.empty())
                {
                    int cur = work.back();
                    work.pop_back();
                    auto grow = [&](int j) {
                        if (j >= 0 && !visited[j] && !region.count(j) && pwRankCollapsible(g, g.nodes[j]) && g.desc(g.nodes[j].outputs[0]).shape == run)
                        {
                            region.insert(j);
                            work.push_back(j);
                        }
                    };
                    for (TensorId t: PwPlanner::dataInputs(g.nodes[cur]))
                    {
                        grow(t >= 0 && t < (TensorId) producer.size() ? producer[t] : -1);
                    }
                    for (int j: readerNodes[g.nodes[cur].outputs[0]])
                    {
                        grow(j);
                    }
                }
                for (int m: region)
                {
                    visited[m] = 1;
                }

                // ---- the coarsest grouping every member's every data input admits ----
                std::vector<char> cutAfter(run.size() - 1, 0);
                for (int m: region)
                {
                    for (TensorId t: PwPlanner::dataInputs(g.nodes[m]))
                    {
                        const Shape &s = g.desc(t).shape;
                        if (numElements(s) > 1)
                        {
                            pwMarkGroupCuts(pwAlignShape(s, run.size()), run, cutAfter);
                        }
                    }
                }
                const int groups = 1 + (int) std::count(cutAfter.begin(), cutAfter.end(), (char) 1);
                if (groups > kPwMaxRank)
                {
                    VKNN_INFO << "fusePointwiseChains: run " << shapeStr(run) << " at '" << g.nodes[*region.begin()].name << "' admits no axis grouping within rank " << kPwMaxRank << " -- the region stays at full rank, unfused";
                    continue;
                }
                const Shape groupedRun = pwGroupShape(run, cutAfter);

                // ---- rewire: fresh grouped output per member first, so interior edges resolve ----
                std::map<TensorId, TensorId> groupedOut; // member's full-rank output -> grouped output
                for (int m: region)
                {
                    TensorId   out = g.nodes[m].outputs[0];
                    TensorDesc d;
                    d.name  = g.desc(out).name + "#pwrc";
                    d.shape = groupedRun;
                    d.dtype = g.desc(out).dtype;
                    groupedOut.emplace(out, g.addTensor(d));
                }
                TensorId runShapeInit = kNoTensor; // shared by the region's expand Reshapes
                for (int m: region)                // std::set iterates ascending = topological
                {
                    Node &nd = g.nodes[m];
                    // Data inputs only: Clip carries its bounds at inputs[1..2] and Relu/Unary are
                    // single-input, mirroring PwPlanner::dataInputs by index.
                    const bool   firstInputOnly = nd.type == OpType::Clip || nd.type == OpType::Relu || nd.type == OpType::Unary;
                    const size_t dataCount      = firstInputOnly ? 1 : nd.inputs.size();
                    for (size_t k = 0; k < dataCount; ++k)
                    {
                        TensorId &in       = nd.inputs[k];
                        auto      interior = groupedOut.find(in);
                        if (interior != groupedOut.end())
                        {
                            in = interior->second; // intra-region edge: connect at the grouped shape
                            continue;
                        }
                        const Shape inShape = g.desc(in).shape;
                        if (numElements(inShape) <= 1)
                        {
                            continue; // scalar: shape-independent
                        }
                        const Shape grouped = pwGroupShape(pwAlignShape(inShape, run.size()), cutAfter);
                        const auto  key     = std::make_pair(in, grouped);
                        auto        it      = collapsedView.find(key);
                        if (it == collapsedView.end())
                        {
                            TensorDesc d;
                            d.name  = g.desc(in).name + "#pwrc" + shapeStr(grouped);
                            d.shape = grouped;
                            d.dtype = g.desc(in).dtype;
                            if (g.isInitializer(in))
                            {
                                // A constant view needs no runtime node: same bytes, grouped shape.
                                // The entry exists from here on (so isInitializer holds through the
                                // rest of the rewiring, which reads shapes only); the payload is
                                // attached at the end of the pass, moved rather than copied
                                // whenever the collapse consumed the source's last reference.
                                d.isInitializer = true;
                                TensorId copy   = g.addTensor(d);
                                g.initializers.emplace(copy, HostBuffer {});
                                groupedConstants.emplace_back(in, copy);
                                it = collapsedView.emplace(key, copy).first;
                            } else
                            {
                                TensorId       view = g.addTensor(d);
                                PendingReshape pr;
                                pr.node.type    = OpType::Reshape;
                                pr.node.name    = d.name;
                                pr.node.inputs  = {in, shapeInitializer(d.name + "#shape", grouped)};
                                pr.node.outputs = {view};
                                int p           = (in >= 0 && in < (TensorId) producer.size()) ? producer[in] : -1;
                                pr.pos          = p < 0 ? 0 : (size_t) p + 1;
                                pr.tier         = 1;
                                pending.push_back(std::move(pr));
                                it = collapsedView.emplace(key, view).first;
                            }
                        }
                        in = it->second;
                    }
                    // The output moves to the grouped tensor; the full-rank value is regenerated
                    // only where a reader outside the region (or a graph-output use) needs it.
                    TensorId orig = nd.outputs[0];
                    nd.outputs[0] = groupedOut.at(orig);
                    bool external = orig < (TensorId) isGraphOut.size() && isGraphOut[orig];
                    for (int j: readerNodes[orig])
                    {
                        external = external || !region.count(j);
                    }
                    if (external)
                    {
                        if (runShapeInit == kNoTensor)
                        {
                            runShapeInit = shapeInitializer(g.desc(orig).name + "#pwrc.run#shape", run);
                        }
                        PendingReshape pr;
                        pr.node.type    = OpType::Reshape;
                        pr.node.name    = g.desc(orig).name + "#pwrc.expand";
                        pr.node.inputs  = {groupedOut.at(orig), runShapeInit};
                        pr.node.outputs = {orig};
                        pr.pos          = (size_t) m + 1;
                        pr.tier         = 0;
                        pending.push_back(std::move(pr));
                    }
                    collapsedNodes++;
                }
                regions++;
            }
            if (regions == 0)
            {
                return;
            }
            // Splice each Reshape right after its input's producer (graph inputs at the front) so
            // node order stays topological; expands land before collapses at the same slot because
            // a later region's collapse view can read the full-rank tensor an expand regenerates.
            // Insertions run back-to-front so earlier positions stay valid.
            std::stable_sort(pending.begin(), pending.end(), [](const PendingReshape &a, const PendingReshape &b) {
                return a.pos != b.pos ? a.pos < b.pos : a.tier < b.tier;
            });
            for (size_t i = pending.size(); i-- > 0;)
            {
                g.nodes.insert(g.nodes.begin() + (long) pending[i].pos, std::move(pending[i].node));
            }
            attachGroupedConstantPayloads(g, groupedConstants);
            VKNN_INFO << "fusePointwiseChains: collapsed " << collapsedNodes << " rank>" << kPwMaxRank << " pointwise node(s) across " << regions << " region(s)";
        }
    } // namespace

    /// One general pointwise fusion: grow each maximal same-shape per-element region — Binary/Add/
    /// Unary/Clip/Relu/PRelu/Where/Greater/GreaterEqual/Less/LessEqual/Equal over one run shape,
    /// connected through def-use edges in either direction, fanout included — and emit it as a
    /// single fused unit: folded into an epilogue-capable producer's store when the region's entry
    /// stream comes from one (pwEpilogueCapable), otherwise a standalone FusedPointwise node.
    /// Internal fanout rides the unit's registers; values consumed outside the region (or graph
    /// outputs) are exported as extra output streams (pw_outs), stored TO_STORE-rounded so they are
    /// byte-identical to the tensors the unfused graph would materialize. A region that exceeds the
    /// step/operand/register/export budgets is emitted as its largest fitting prefix and the
    /// remaining members seed the next unit.
    ///
    /// Legality: regions are convex — an external node that transitively depends on a region value
    /// and feeds a later region member (only possible inside the region's node-index interval, as
    /// node order approximates topological order) excludes the fed member. Chains never grow across
    /// fp32-pinned tensors, int-typed tensors, unresolved shapes, or runtime Clip bounds. A
    /// standalone unit is emitted at the LAST member's slot, so every operand is produced before it;
    /// external consumers of exported values are re-ordered by the load-time topoSort when needed.
    ///
    /// Rounding discipline (strictFuse): in strict mode the unit's entry value is rounded to the
    /// byte the producer would store and every step result passes TO_STORE
    /// (shaders/pw_epilogue.glsl), reproducing each fp16 store of the unfused graph bit for bit, in
    /// the same order. In the default fast mode the unit carries the pw_relax attr: the entry still
    /// rounds to the producer's store byte (inter-unit tensors stay on the unfused trajectory), but
    /// the steps chain unrounded in fp32 registers and the unit rounds ONCE per stored stream —
    /// fewer roundings than the unfused graph on every multi-step chain, and byte-identical to it
    /// for single-step units and chains ending in a monotone activation.
    ///
    /// Precondition: shapes inferred and const-folding done (this runs LAST among the standard
    /// passes). Postcondition: fully-merged members are erased; the producer (or the anchor slot's
    /// FusedPointwise node) yields the unit's main output and carries pw_steps/pw_params/pw_outs —
    /// plus pw_flat on standalone nodes for the load-time layout classifier.
    void fusePointwiseChains(Graph &g, bool strictFuse) {
        // Rank collapse runs BEFORE right-alignment: alignment engages only on kNchwRank runs, so
        // it must observe each run's FINAL rank -- a region collapsed to rank 4 joins the alignment
        // (and NC4 classification) population, while the reverse order would skip the original
        // rank-5 form and never revisit the rank-4 run the collapse produces.
        collapsePwRunRanks(g);   // rank>kPwMaxRank pointwise regions -> coarsest grouped rank<=kPwMaxRank views
        rightAlignPwOperands(g); // rank<4 runtime operands -> explicit right-aligned rank-4 views
        std::vector<int>  producer, consumerCount;
        std::vector<char> isGraphOut;
        auto              rebuild = [&]() {
            producer.assign(g.tensors.size(), -1);
            consumerCount.assign(g.tensors.size(), 0);
            isGraphOut.assign(g.tensors.size(), 0);
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                for (TensorId o: g.nodes[i].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producer[o] = (int) i;
                    }
                }
            }
            for (size_t j = 0; j < g.nodes.size(); ++j)
            {
                for (TensorId in: g.nodes[j].inputs)
                {
                    if (in != kNoTensor && in < (TensorId) consumerCount.size())
                    {
                        consumerCount[in]++;
                    }
                }
            }
            for (TensorId go: g.outputs)
            {
                if (go != kNoTensor)
                {
                    consumerCount[go]++;
                    isGraphOut[go] = 1;
                }
            }
        };
        rebuild();

        std::set<int> removed;
        int           fused = 0, attached = 0;

        // readers built once against the ORIGINAL nodes; emission only removes members and rewires
        // through preserved tensor ids, so reader NODE INDICES stay valid (removed ones are skipped).
        std::vector<std::vector<int>> readers(g.tensors.size());
        for (size_t j = 0; j < g.nodes.size(); ++j)
        {
            for (TensorId in: g.nodes[j].inputs)
            {
                if (in != kNoTensor && in < (TensorId) readers.size())
                {
                    readers[in].push_back((int) j);
                }
            }
        }

        std::vector<char> visited(g.nodes.size(), 0);
        for (size_t seed = 0; seed < g.nodes.size(); ++seed)
        {
            if (visited[seed] || removed.count((int) seed) || !pwEligibleNode(g, g.nodes[seed]))
            {
                continue;
            }
            const Shape run = g.desc(g.nodes[seed].outputs[0]).shape;

            // ---- collect the undirected same-shape component ----
            std::set<int>    inComp;
            std::vector<int> work {(int) seed};
            inComp.insert((int) seed);
            while (!work.empty())
            {
                int cur = work.back();
                work.pop_back();
                const Node &nd = g.nodes[cur];
                for (TensorId t: PwPlanner::dataInputs(nd))
                {
                    if (t == kNoTensor || pwTensorIsFp32(g, t))
                    {
                        continue;
                    }
                    int p = (t >= 0 && t < (TensorId) producer.size()) ? producer[t] : -1;
                    if (p >= 0 && !visited[p] && !removed.count(p) && !inComp.count(p) && pwEligibleNode(g, g.nodes[p]) && g.desc(g.nodes[p].outputs[0]).shape == run)
                    {
                        inComp.insert(p);
                        work.push_back(p);
                    }
                }
                TensorId ot = nd.outputs[0];
                if (!pwTensorIsFp32(g, ot))
                {
                    for (int j: readers[ot])
                    {
                        if (!visited[j] && !removed.count(j) && !inComp.count(j) && pwEligibleNode(g, g.nodes[j]) && g.desc(g.nodes[j].outputs[0]).shape == run)
                        {
                            inComp.insert(j);
                            work.push_back(j);
                        }
                    }
                }
            }

            // ---- convexity: exclude members fed by externals that depend on the region ----
            std::vector<int> members(inComp.begin(), inComp.end()); // std::set iterates ascending
            {
                std::set<int>      kept(members.begin(), members.end());
                std::set<TensorId> tainted;
                int                lo = members.front(), hi = members.back();
                for (int j = lo; j <= hi; ++j)
                {
                    if (removed.count(j))
                    {
                        continue;
                    }
                    const Node &nd = g.nodes[j];
                    if (kept.count(j))
                    {
                        bool bad = false;
                        for (TensorId t: nd.inputs)
                        {
                            if (t != kNoTensor && tainted.count(t))
                            {
                                bad = true;
                            }
                        }
                        if (bad)
                        {
                            kept.erase(j);
                            tainted.insert(nd.outputs[0]);
                        }
                    } else
                    {
                        bool dep = false;
                        for (TensorId t: nd.inputs)
                        {
                            if (t == kNoTensor)
                            {
                                continue;
                            }
                            int p = (t >= 0 && t < (TensorId) producer.size()) ? producer[t] : -1;
                            if (tainted.count(t) || (p >= 0 && kept.count(p)))
                            {
                                dep = true;
                            }
                        }
                        if (dep)
                        {
                            for (TensorId o: nd.outputs)
                            {
                                if (o != kNoTensor)
                                {
                                    tainted.insert(o);
                                }
                            }
                        }
                    }
                }
                members.assign(kept.begin(), kept.end());
            }
            if (members.empty())
            {
                continue;
            }

            // ---- can the entry's producer host the unit as its store epilogue? ----
            auto canHostUnit = [&](const PwUnit &u, const std::vector<int> &mem, int &prod, bool &entryExp) -> bool {
                prod     = (u.entry >= 0 && u.entry < (TensorId) producer.size()) ? producer[u.entry] : -1;
                entryExp = false;
                bool ok  = prod >= 0 && !removed.count(prod) && pwEpilogueCapable(g.nodes[prod].type) && !g.nodes[prod].attr.has("pw_steps") &&
                          g.nodes[prod].outputs.size() == 1 && g.nodes[prod].outputs[0] == u.entry;
                // The register-tiled MatMul kernel (matmul_tiled, chosen for M,N,K >=
                // kTiledMatMulMin — the same constant matmul.cpp gates on) has no register
                // headroom for the VM at its per-thread register micro-tile store loop — an
                // attached unit collapses the GEMM's occupancy and costs far more than a
                // standalone dispatch. Such units run standalone; small matmuls (the
                // 1-thread-per-output kernel) still host epilogues.
                // An aliasable Concat must not host: pw_steps forces one copy dispatch per part,
                // while an unhosted concat elides into arena views and the unit runs as a single
                // standalone dispatch (see pwConcatPartsCanAlias).
                if (ok && g.nodes[prod].type == OpType::Concat && pwConcatPartsCanAlias(g, g.nodes[prod]))
                {
                    ok = false;
                }
                if (ok && g.nodes[prod].type == OpType::MatMul)
                {
                    const Node  &P  = g.nodes[prod];
                    const Shape &as = g.desc(P.inputs[0]).shape;
                    const Shape &bs = g.desc(P.inputs[1]).shape;
                    if (as.size() >= 2 && bs.size() >= 2)
                    {
                        int64_t M = as[as.size() - 2], K = as[as.size() - 1], N = bs[bs.size() - 1];
                        if (M >= kTiledMatMulMin && N >= kTiledMatMulMin && K >= kTiledMatMulMin)
                        {
                            ok = false;
                        }
                    }
                }
                // every appended operand must already be available before the producer, otherwise
                // the load-time topoSort sinks the producer and its widely-consumed output's
                // lifetime blows up the buffer pool (VK_ERROR_OUT_OF_HOST_MEMORY on large models)
                for (size_t k = 0; ok && k < u.operands.size(); ++k)
                {
                    TensorId op = u.operands[k];
                    if (op != kNoTensor && !g.isInitializer(op))
                    {
                        int pop = (op >= 0 && op < (TensorId) producer.size()) ? producer[op] : -1;
                        if (pop >= prod)
                        {
                            ok = false;
                        }
                    }
                }
                // the producer's kernel executes the steps in ITS layout world; cross-world folding
                // is exact (layout only changes indexing) provided the steps are expressible there
                if (ok)
                {
                    ok = gpuFlatNode(g, g.nodes[prod]) ? u.flatOk : u.nc4Ok;
                }
                // an entry consumed outside the unit (or declared a graph output) must keep
                // materializing: export the entry value itself on an extra output stream
                if (ok)
                {
                    int unitEntryReads = 0;
                    for (int m: mem)
                    {
                        for (TensorId t: PwPlanner::dataInputs(g.nodes[m]))
                        {
                            if (t == u.entry)
                            {
                                unitEntryReads++;
                            }
                        }
                    }
                    bool external = consumerCount[u.entry] > unitEntryReads || (u.entry < (TensorId) isGraphOut.size() && isGraphOut[u.entry]);
                    if (external)
                    {
                        if ((int) u.exports.size() >= kPwMaxOuts)
                        {
                            ok = false;
                        } else
                        {
                            entryExp = true;
                        }
                    }
                }
                return ok;
            };

            // ---- emit the largest fitting prefix; the rest re-enters the pool. Each prefix is
            // planned once per entry candidate: the first attachable plan wins, else the first
            // encodable one runs standalone. ----
            PwPlanner planner(g, run, producer, consumerCount, isGraphOut, strictFuse);
            PwUnit    unit;
            int       hostProd      = -1;
            bool      entryExported = false;
            bool      haveUnit      = false;
            for (size_t cut = members.size(); cut >= 1 && !haveUnit; --cut)
            {
                std::vector<int> prefix(members.begin(), members.begin() + (long) cut);
                PwUnit           fallback;
                for (TensorId cand: planner.entryCandidates(prefix))
                {
                    PwUnit u = planner.plan(prefix, cand);
                    if (!u.ok)
                    {
                        continue;
                    }
                    int  prod;
                    bool eExp;
                    if (canHostUnit(u, prefix, prod, eExp))
                    {
                        unit          = u;
                        hostProd      = prod;
                        entryExported = eExp;
                        haveUnit      = true;
                        break;
                    }
                    if (!fallback.ok)
                    {
                        fallback = u;
                    }
                }
                if (!haveUnit && fallback.ok)
                {
                    unit     = fallback;
                    hostProd = -1;
                    haveUnit = true;
                }
                if (haveUnit)
                {
                    members = prefix;
                }
            }
            if (!haveUnit)
            {
                visited[seed] = 1; // not encodable at all: leave the seed unfused
                continue;
            }
            warnFlatForcedUnit(g, g.nodes[members.back()], unit.generalOperand, run, unit.nc4Ok);

            // Fast mode: a lone initializer-bias Add on a MatMul folds onto the kernel's native
            // bias input — matmul[_tiled]_bias adds it in the fp32 accumulator with one store
            // rounding, the fp32-chained semantics at zero VM cost. The register-tiled GEMM cannot
            // host a VM unit (canHostUnit refuses it), so without this fold every transformer-layer
            // bias would run as its own dispatch.
            if (!strictFuse && unit.exports.empty() && members.size() == 1 && unit.operands.size() == 1 && unit.steps.size() == 8)
            {
                int prod = (unit.entry >= 0 && unit.entry < (TensorId) producer.size()) ? producer[unit.entry] : -1;
                bool addStep = unit.steps[0] == kPwKindBinary && unit.steps[1] == (int64_t) BinaryType::Add && ((unit.steps[2] == kPwRefEntry && unit.steps[3] <= kPwRefOp0) || (unit.steps[3] == kPwRefEntry && unit.steps[2] <= kPwRefOp0));
                if (addStep && prod >= 0 && !removed.count(prod) && g.isInitializer(unit.operands[0]))
                {
                    Node &P       = g.nodes[prod];
                    bool  soleUse = consumerCount[unit.entry] == 1 && !(unit.entry < (TensorId) isGraphOut.size() && isGraphOut[unit.entry]);
                    bool hostable = P.type == OpType::MatMul && P.fusedBias == kNoTensor && !P.attr.has("pw_steps") && P.outputs.size() == 1 && P.outputs[0] == unit.entry && soleUse;
                    const Shape &os = g.desc(unit.mainOut).shape;
                    const Shape &bs = g.desc(unit.operands[0]).shape;
                    if (hostable && !os.empty() && !bs.empty() && bs.back() == os.back() && numElements(bs) == os.back())
                    {
                        P.fusedBias = unit.operands[0];
                        P.inputs.push_back(unit.operands[0]); // keep the bias live for DCE/allocation
                        P.outputs[0] = unit.mainOut;
                        removed.insert(members[0]);
                        fused++;
                        attached++;
                        rebuild();
                        continue;
                    }
                }
            }

            if (hostProd >= 0)
            {
                int prod = hostProd;
                // Inline-activation fast path: a lone Relu — or Clip whose bounds round-trip fp16
                // exactly — after a Conv/Gemm folds onto the kernel's own fusedAct epilogue instead
                // of a pw unit (no plan SSBO, no extra bindings). Byte-safe: a monotone clamp with
                // exactly-representable bounds commutes with RTE rounding, so act applied in the
                // fp32 accumulator stores the same bytes the unfused act-of-rounded-value would.
                if (members.size() == 1 && unit.operands.empty() && unit.exports.empty() && !entryExported)
                {
                    Node       &P        = g.nodes[prod];
                    const Node &mn       = g.nodes[members[0]];
                    bool        hostable = (P.type == OpType::Conv || P.type == OpType::Gemm) && P.fusedAct == ActType::None && P.fusedResidual == kNoTensor;
                    auto        rep      = [](float v) {
                        return v == halfToFloat(floatToHalf(v));
                    };
                    ActType act = ActType::None;
                    float   lo = 0, hi = 0;
                    if (hostable && mn.type == OpType::Relu)
                    {
                        act = ActType::Relu;
                    } else if (hostable && mn.type == OpType::Clip)
                    {
                        pwClipBounds(g, mn, lo, hi);
                        if (rep(lo) && rep(hi))
                        {
                            act = (lo == 0.f && hi == 6.f) ? ActType::Relu6 : ActType::Clip;
                        }
                    }
                    if (act != ActType::None)
                    {
                        P.fusedAct   = act;
                        P.actLo      = lo;
                        P.actHi      = hi;
                        P.outputs[0] = unit.mainOut; // consumers already read the act's tensor id
                        removed.insert(members[0]);
                        fused++;
                        attached++;
                        rebuild();
                        continue;
                    }
                }
                Node                &P      = g.nodes[prod];
                int                  opbase = (int) P.inputs.size();
                std::vector<int64_t> es     = unit.steps;
                for (TensorId op: unit.operands)
                {
                    P.inputs.push_back(op);
                }
                for (int s = 0; s < (int) es.size() / 8; ++s)
                {
                    for (int f = 2; f <= 4; ++f)
                    {
                        int64_t ref = es[s * 8 + f];
                        if (ref <= kPwRefOp0)
                        {
                            es[s * 8 + f] = kPwRefOp0 - (opbase + (int) (kPwRefOp0 - ref) - 1);
                        }
                    }
                }
                {
                    Attr a;
                    a.kind                 = Attr::Ints;
                    a.ints                 = es;
                    P.attr.map["pw_steps"] = a;
                }
                {
                    Attr a;
                    a.kind                  = Attr::Floats;
                    a.floats                = unit.params;
                    P.attr.map["pw_params"] = a;
                }
                {
                    Attr a;
                    a.kind                  = Attr::Int;
                    a.i                     = opbase;
                    P.attr.map["pw_opbase"] = a;
                }
                if (!strictFuse)
                {
                    Attr a;
                    a.kind                 = Attr::Int;
                    a.i                    = 1;
                    P.attr.map["pw_relax"] = a;
                }
                P.outputs.assign(1, unit.mainOut);
                std::vector<int64_t>  outs = unit.outSteps;
                std::vector<TensorId> exp  = unit.exports;
                if (entryExported)
                {
                    exp.push_back(unit.entry); // the producer's own value keeps its tensor id
                    outs.push_back(kPwRefEntry);
                }
                for (TensorId e: exp)
                {
                    P.outputs.push_back(e);
                }
                if (!outs.empty())
                {
                    Attr a;
                    a.kind                = Attr::Ints;
                    a.ints                = outs;
                    P.attr.map["pw_outs"] = a;
                }
                for (int m: members)
                {
                    removed.insert(m);
                }
                fused++;
                attached++;
                rebuild();
                continue;
            }

            // ---- standalone unit. A multi-member region is always worth a node. A SINGLE member
            // is worth one exactly when it would otherwise run on the flat kernel with a broadcast
            // or constant operand (gpuFlatNode: initializer-operand Binary/Add, non-channel
            // broadcasts, Where and the comparisons have no NC4 kernel of their own): as a
            // one-step NC4-expressible unit it joins the flexible layout re-vote, runs packed, and
            // sheds the full-size ConvertLayout pair the flat node would carry. Everything the
            // flat classifier already serves in NC4 (same-shape runtime Binary, lone activations)
            // keeps its original node, so existing encodings are untouched. ----
            if (members.size() < 2)
            {
                const bool broadcastSingle = unit.nc4Ok && !unit.operands.empty() && PwPlanner::dataInputs(g.nodes[members[0]]).size() > 1 && gpuFlatNode(g, g.nodes[members[0]]);
                if (!broadcastSingle)
                {
                    visited[members[0]] = 1;
                    continue;
                }
            }
            Node fn;
            fn.type   = OpType::FusedPointwise;
            fn.name   = g.nodes[members.front()].name + "#pwunit";
            fn.inputs = {unit.entry};
            for (TensorId op: unit.operands)
            {
                fn.inputs.push_back(op);
            }
            fn.outputs = {unit.mainOut};
            for (TensorId e: unit.exports)
            {
                fn.outputs.push_back(e);
            }
            {
                Attr a;
                a.kind                  = Attr::Ints;
                a.ints                  = unit.steps;
                fn.attr.map["pw_steps"] = a;
            }
            {
                Attr a;
                a.kind                   = Attr::Floats;
                a.floats                 = unit.params;
                fn.attr.map["pw_params"] = a;
            }
            if (!unit.outSteps.empty())
            {
                Attr a;
                a.kind                 = Attr::Ints;
                a.ints                 = unit.outSteps;
                fn.attr.map["pw_outs"] = a;
            }
            {
                Attr a;
                a.kind                 = Attr::Int;
                a.i                    = unit.nc4Ok ? 0 : 1; // NC4HW4 when expressible (conv-adjacent graphs live there)
                fn.attr.map["pw_flat"] = a;
            }
            if (!strictFuse)
            {
                Attr a;
                a.kind                  = Attr::Int;
                a.i                     = 1;
                fn.attr.map["pw_relax"] = a;
            }
            int anchor = members.back();
            for (int m: members)
            {
                if (m != anchor)
                {
                    removed.insert(m);
                }
            }
            g.nodes[anchor] = fn;
            visited[anchor] = 1;
            fused++;
            rebuild();
        }

        if (fused)
        {
            std::vector<Node> keptNodes;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!removed.count((int) i))
                {
                    keptNodes.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(keptNodes);
            VKNN_INFO << "fusePointwiseChains: fused " << fused << " unit(s), " << attached << " into producer epilogues";
        }
    }

} // namespace vknn
