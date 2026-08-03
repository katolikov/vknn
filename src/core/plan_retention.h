// What a session must still hold on the host once a plan bucket is built.
//
// Two independent retention rules, both pure functions of the graph (no backend, no device state), so
// the policy is testable on its own and has exactly one definition:
//   - reclaimableInitializers(): the initializer payloads a built bucket may drop.
//   - importedGraphCanPlanNewShapes(): whether the pristine imported graph can still yield a bucket
//     that the default one does not already cover.
#pragma once
#include "vknn/graph.h"
#include <set>
#include <vector>

namespace vknn {

    /// Op types whose prepare() reads every constant input it needs and uploads it to the device,
    /// never touching the host payload again.
    ///
    /// This is the ONLY set whose weights a bucket may reclaim, and it is deliberately a short
    /// allow-list of the weighted ops: conv and matmul families, whose prepacked blobs are the bytes
    /// worth reclaiming at all. Everything else keeps its payload, because an op is free to
    /// materialize a constant operand on its FIRST record -- flat Concat parts, pointwise unit
    /// operands and pad values all do -- and that record happens after the reclaim.
    inline bool opUploadsConstantsAtPrepare(OpType type) {
        switch (type)
        {
            case OpType::Conv:
            case OpType::ConvGemm:
            case OpType::ConvTranspose:
            case OpType::MatMul:
            case OpType::Gemm:
            case OpType::FusedDwPw:
            case OpType::FusedSE:
            case OpType::BatchNorm:
                return true;
            default:
                return false;
        }
    }

    /// Initializers whose host payload must stay resolvable after a bucket's segments are built.
    ///
    /// The rule is an allow-list of what may be DROPPED, not of what must be kept. An op may resolve
    /// a constant operand while recording rather than while preparing, and the first record runs
    /// after the reclaim, so a payload dropped on the assumption that "every GPU op has taken its
    /// device copy by now" is a null buffer at dispatch -- not a diagnosable error, a crash. Only a
    /// weighted op reading a constant in a weight position is known to be finished with the host
    /// bytes, so only that is freed; everything else is kept.
    ///
    /// Kept, then:
    ///   - any operand of a node on the CPU backend (its pool entry is a copy, but the graph entry
    ///     keeps `isInitializer` true for the boundary/dump paths that key off it),
    ///   - a node's fused residual/bias edge, referenced outside `inputs`,
    ///   - a graph input or output that is itself a constant,
    ///   - a pointwise unit's operands, named either as the tail past pwCoreInputs() or by a step's
    ///     srcA/srcB/srcC field (operand i encoded as kPwRefOp0 - i),
    ///   - every constant read by an op outside opUploadsConstantsAtPrepare(), whatever it is.
    ///
    /// @param g            The bucket's planned graph.
    /// @param nodeRunsOnCpu One entry per node: true when the node is assigned to the CPU backend. A
    ///                     shorter vector treats the missing tail as CPU-assigned (the conservative
    ///                     reading — those nodes' operands are kept).
    inline std::set<TensorId> initializersNeededAfterPlanning(const Graph &g, const std::vector<bool> &nodeRunsOnCpu) {
        std::set<TensorId> needed;
        auto               keep = [&](TensorId t) {
            if (t != kNoTensor && g.isInitializer(t))
            {
                needed.insert(t);
            }
        };
        for (size_t n = 0; n < g.nodes.size(); ++n)
        {
            const Node &nd = g.nodes[n];
            // A fused residual/bias rides an edge outside node.inputs, so it is never covered by the
            // operand walks below and is kept for every node regardless of backend.
            keep(nd.fusedResidual);
            keep(nd.fusedBias);
            const bool onCpu = n >= nodeRunsOnCpu.size() || nodeRunsOnCpu[n];
            // Input 0 is the activation stream; a weighted op's constants sit past it, and anything
            // at or past pwCoreInputs() is a fused-epilogue operand this op resolves while recording.
            const size_t coreInputs    = pwCoreInputs(nd);
            const bool   uploadsAtPrep = !onCpu && opUploadsConstantsAtPrepare(nd.type);
            for (size_t k = 0; k < nd.inputs.size(); ++k)
            {
                const bool weightPosition = uploadsAtPrep && k >= 1 && k < coreInputs;
                if (!weightPosition)
                {
                    keep(nd.inputs[k]);
                }
            }
            if (onCpu)
            {
                continue;
            }
            // A pointwise unit resolves its constant operands lazily while RECORDING (pwOperandBuf
            // materializes one on first use), so their payloads must outlive planning. Which inputs
            // those are is stated by the plan itself: a step's srcA/srcB/srcC field encodes operand
            // i as kPwRefOp0 - i, indexing node.inputs. Reading the step words rather than assuming
            // a contiguous tail covers both carriers of a unit — an epilogue host, whose operands
            // start at pwCoreInputs(), and a standalone FusedPointwise node, whose plan may name any
            // input — and a future step layout that names one somewhere else again.
            const std::vector<int64_t> &steps = nd.attr.getints("pw_steps");
            for (size_t s = 0; s + kPwStepInts <= steps.size(); s += kPwStepInts)
            {
                for (int field = kPwStepSrcA; field <= kPwStepSrcC; ++field)
                {
                    const int64_t ref = steps[s + (size_t) field];
                    if (ref > kPwRefOp0)
                    {
                        continue; // accumulator, entry value, register, or unused slot
                    }
                    const size_t operandIndex = (size_t) (kPwRefOp0 - ref);
                    if (operandIndex < nd.inputs.size())
                    {
                        keep(nd.inputs[operandIndex]);
                    }
                }
            }
        }
        for (TensorId id: g.outputs)
        {
            keep(id);
        }
        return needed;
    }

    /// The complement of initializersNeededAfterPlanning() over the graph's initializers: every payload
    /// the built bucket can drop. Derived from the needed set rather than from a list of weighted op
    /// types, so a weighted op added later is reclaimed without touching this rule.
    inline std::set<TensorId> reclaimableInitializers(const Graph &g, const std::vector<bool> &nodeRunsOnCpu) {
        const std::set<TensorId> needed = initializersNeededAfterPlanning(g, nodeRunsOnCpu);
        std::set<TensorId>       freeable;
        for (const auto &kv: g.initializers)
        {
            if (!needed.count(kv.first))
            {
                freeable.insert(kv.first);
            }
        }
        return freeable;
    }

    /// Whether re-running the pass pipeline on the pristine imported graph at caller-declared input
    /// shapes can produce a plan the default bucket does not already hold.
    ///
    /// Shape resolution fills only dimensions the model left dynamic: inferShapes skips every axis whose
    /// extent is already non-negative, so neither a declared shape nor a symbolic-dim binding can move a
    /// fully static input. A graph with no dynamic input axis therefore plans one and only one bucket,
    /// and the pristine copy of its weights is dead once that bucket is built.
    inline bool importedGraphCanPlanNewShapes(const Graph &g) {
        for (TensorId in: g.inputs)
        {
            if (in == kNoTensor)
            {
                continue;
            }
            for (int64_t extent: g.desc(in).shape)
            {
                if (extent < 0)
                {
                    return true;
                }
            }
        }
        return false;
    }

} // namespace vknn
