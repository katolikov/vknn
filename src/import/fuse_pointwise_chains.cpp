#include "passes_internal.h"
#include "core/matmul_tile.h"
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
        if (n.inputs.size() > 1 && n.inputs[1] != kNoTensor)
        {
            if (!g.isInitializer(n.inputs[1]))
            {
                return false;
            }
            lo = g.initializers.at(n.inputs[1]).f32()[0];
        }
        if (n.inputs.size() > 2 && n.inputs[2] != kNoTensor)
        {
            if (!g.isInitializer(n.inputs[2]))
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
    // be resolved, and the output must not be fp32-pinned.
    static bool pwEligibleNode(const Graph &g, const Node &n) {
        switch (n.type)
        {
            case OpType::Binary:
            case OpType::Add:
            case OpType::Unary:
            case OpType::Clip:
            case OpType::Relu:
            case OpType::PRelu:
            case OpType::Where:
            case OpType::Greater:
            case OpType::GreaterEqual:
            case OpType::Less:
            case OpType::LessEqual:
            case OpType::Equal:
                break;
            default:
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
    // producer instead of a standalone node). Every listed type's GPU kernel family carries an _epi
    // variant reading pw_steps; anything else keeps the standalone FusedPointwise node.
    static bool pwEpilogueCapable(OpType t) {
        switch (t)
        {
            case OpType::MatMul:
            case OpType::Gemm:
            case OpType::Conv:
            case OpType::ConvGemm:
            case OpType::ConvTranspose:
            case OpType::FusedDwPw:
            case OpType::Softmax:
            case OpType::LayerNorm:
            case OpType::Reduce:
            case OpType::GridSample:
            case OpType::Resize:
            case OpType::MaxPool:
            case OpType::AvgPool:
            case OpType::GlobalAvgPool:
            case OpType::Transpose: // flat_gather _epi: fold consumers into the Transpose/Slice store,
            case OpType::Slice:     // dropping their dispatch AND the materialized gather output
            case OpType::Concat:    // per-part stores apply the unit in output space (concat/flat_scatter _epi)
                return true;
            default:
                return false;
        }
    }

    // Broadcast class of tensor `t` against the unit's run shape: 0 same-shape, 3 scalar splat,
    // 1 per-channel (rank-4 [N,C,1,1]; a rank<4 CONSTANT that right-aligns to [1,C,1,1] with N==1
    // also qualifies — pwOperandBuf packs it by that interpretation), 2 general (flat-only).
    static int pwBcastClass(const Graph &g, TensorId t, const Shape &run) {
        const Shape &s = g.desc(t).shape;
        if (s == run)
        {
            return 0;
        }
        if (numElements(s) <= 1)
        {
            return 3;
        }
        if (run.size() == 4)
        {
            if (s.size() == 4 && s[0] == run[0] && s[1] == run[1] && s[2] == 1 && s[3] == 1)
            {
                return 1;
            }
            if (g.isInitializer(t) && run[0] == 1 && s.size() < 4)
            {
                Shape rs(4, 1);
                for (size_t k = 0; k < s.size(); ++k)
                {
                    rs[4 - s.size() + k] = s[k];
                }
                if (rs[0] == 1 && rs[1] == run[1] && rs[2] == 1 && rs[3] == 1)
                {
                    return 1;
                }
            }
        }
        return 2;
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
            bool                  nc4Ok = false, flatOk = false;
            bool                  ok = false;
        };

        // Plan a unit over `members` (node indices, ascending = emission order) with run shape
        // `run`. Values flow: each member's result lands in the accumulator; a result consumed only
        // by the immediately following member rides the accumulator, anything with a later or
        // second internal reader takes one of kPwMaxRegs registers, and a member that must first
        // LOAD extra broadcast operands cannot receive its predecessor through the accumulator
        // (loads pass through it), so those predecessors take registers too. External readers of a
        // member value (or a graph-output use) make it an export.
        struct PwPlanner {
            const Graph                        &g;
            const Shape                        &run;
            const std::vector<int>             &producer;
            const std::vector<int>             &consumerCount;
            const std::vector<char>            &isGraphOut;
            bool                                strict;

            PwPlanner(const Graph &g_, const Shape &run_, const std::vector<int> &prod, const std::vector<int> &cc, const std::vector<char> &go, bool strict_)
                : g(g_), run(run_), producer(prod), consumerCount(cc), isGraphOut(go), strict(strict_) {}

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
                        if (t != kNoTensor && !(p >= 0 && memberSet.count(p)) && !g.isInitializer(t) && g.desc(t).shape == run && !pwTensorIsFp32(g, t) && std::find(cands.begin(), cands.end(), t) == cands.end())
                        {
                            cands.push_back(t);
                        }
                    }
                }
                return cands;
            }

            PwUnit plan(const std::vector<int> &members, TensorId entry) {
                PwUnit           u;
                std::set<int>    memberSet(members.begin(), members.end());
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
                        int p = (t >= 0 && t < (TensorId) producer.size()) ? producer[t] : -1;
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
                            regFree[r]  = 0;
                            regOf[val]  = r;
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
                bool     hasClass2 = false;
                TensorId accVal    = kNoTensor; // which member value the accumulator holds
                std::map<TensorId, int64_t> emittedStepOf;
                for (int mi = 0; mi < (int) members.size(); ++mi)
                {
                    int         m  = members[mi];
                    const Node &nd = g.nodes[m];
                    auto        di = dataInputs(nd);

                    // resolve each data input to a ref + (for operands) its broadcast class
                    struct Src {
                        int64_t ref = kPwRefNone;
                        int     bc  = 0;
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
                            if (src[k].bc == 2)
                            {
                                hasClass2 = true;
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
                        size_t  ps     = u.steps.size() - 8;
                        int64_t pKind  = u.steps[ps], pCode = u.steps[ps + 1], pSrcA = u.steps[ps + 2], pDst = u.steps[ps + 5];
                        bool    sig    = pKind == kPwKindUnary && pCode == (int64_t) UnaryType::Sigmoid;
                        bool    hsig   = pKind == kPwKindUnary && pCode == (int64_t) UnaryType::HardSigmoid && u.params[u.params.size() - 2] == 1.0f / 6.0f && u.params.back() == 0.5f;
                        // the gate value must feed ONLY this Mul (accumulator-carried, no register,
                        // no export) and the Mul's other source must be the gated value itself
                        bool    gateOk = (sig || hsig) && pDst == kPwRefNone && accVal != kNoTensor && consumerCount[accVal] == 1;
                        bool    diamond = gateOk && ((sA == pSrcA && sB == kPwRefAcc) || (sB == pSrcA && sA == kPwRefAcc));
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
                    TensorId val = g.nodes[m].outputs[0];
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
                bool ok  = prod >= 0 && !removed.count(prod) && pwEpilogueCapable(g.nodes[prod].type) && !g.nodes[prod].attr.has("pw_steps") && g.nodes[prod].outputs.size() == 1 && g.nodes[prod].outputs[0] == u.entry;
                // The register-tiled MatMul kernel (matmul_tiled, chosen for M,N,K >=
                // kTiledMatMulMin — the same constant matmul.cpp gates on) has no register
                // headroom for the VM at its per-thread register micro-tile store loop — an
                // attached unit collapses the GEMM's occupancy and costs far more than a
                // standalone dispatch. Such units run standalone; small matmuls (the
                // 1-thread-per-output kernel) still host epilogues.
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

            // Fast mode: a lone initializer-bias Add on a MatMul folds onto the kernel's native
            // bias input — matmul[_tiled]_bias adds it in the fp32 accumulator with one store
            // rounding, the fp32-chained semantics at zero VM cost. The register-tiled GEMM cannot
            // host a VM unit (canHostUnit refuses it), so without this fold every transformer-layer
            // bias would run as its own dispatch.
            if (!strictFuse && unit.exports.empty() && members.size() == 1 && unit.operands.size() == 1 && unit.steps.size() == 8)
            {
                int  prod    = (unit.entry >= 0 && unit.entry < (TensorId) producer.size()) ? producer[unit.entry] : -1;
                bool addStep = unit.steps[0] == kPwKindBinary && unit.steps[1] == (int64_t) BinaryType::Add && ((unit.steps[2] == kPwRefEntry && unit.steps[3] <= kPwRefOp0) || (unit.steps[3] == kPwRefEntry && unit.steps[2] <= kPwRefOp0));
                if (addStep && prod >= 0 && !removed.count(prod) && g.isInitializer(unit.operands[0]))
                {
                    Node &P        = g.nodes[prod];
                    bool  soleUse  = consumerCount[unit.entry] == 1 && !(unit.entry < (TensorId) isGraphOut.size() && isGraphOut[unit.entry]);
                    bool  hostable = P.type == OpType::MatMul && P.fusedBias == kNoTensor && !P.attr.has("pw_steps") && P.outputs.size() == 1 && P.outputs[0] == unit.entry && soleUse;
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
                    Node       &P  = g.nodes[prod];
                    const Node &mn = g.nodes[members[0]];
                    bool hostable  = (P.type == OpType::Conv || P.type == OpType::Gemm) && P.fusedAct == ActType::None && P.fusedResidual == kNoTensor;
                    auto rep       = [](float v) { return v == halfToFloat(floatToHalf(v)); };
                    ActType act    = ActType::None;
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
                std::vector<int64_t> outs = unit.outSteps;
                std::vector<TensorId> exp = unit.exports;
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

            // ---- standalone unit (worth a node only for >= 2 members) ----
            if (members.size() < 2)
            {
                visited[members[0]] = 1;
                continue;
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
                a.kind                = Attr::Ints;
                a.ints                = unit.outSteps;
                fn.attr.map["pw_outs"] = a;
            }
            {
                Attr a;
                a.kind                = Attr::Int;
                a.i                   = unit.nc4Ok ? 0 : 1; // NC4HW4 when expressible (conv-adjacent graphs live there)
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
            g.nodes[anchor]  = fn;
            visited[anchor]  = 1;
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
