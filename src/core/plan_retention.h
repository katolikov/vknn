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

    /// Initializers whose host payload must stay resolvable after a bucket's segments are built.
    ///
    /// Every GPU op has taken its device copy by the time a segment finishes building: the flat path
    /// (uploadInit / uploadInitRaw) drops the payload inline, the prepacked path holds only the device
    /// buffer, and an operand resolved while RECORDING goes through operandBuf, which memoizes the
    /// device buffer in the op instance — so a re-record never re-reads host bytes. What still needs a
    /// host payload afterwards is therefore exactly:
    ///   - an operand of a node assigned to the CPU backend (its runtime pool entry is a copy, but the
    ///     graph entry keeps `isInitializer` true for the boundary/dump paths that key off it),
    ///   - a node's fused residual/bias edge, which is referenced outside `inputs`,
    ///   - a graph output that is itself a constant,
    ///   - a fused pointwise-chain operand (an input at or past pwCoreInputs()), which an epilogue
    ///     uploads lazily while recording.
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
            if (onCpu)
            {
                for (TensorId in: nd.inputs)
                {
                    keep(in);
                }
                continue;
            }
            // A pointwise unit resolves its constant operands lazily while RECORDING (pwOperandBuf
            // materializes one on first use), so their payloads must outlive planning. Which inputs
            // those are is stated by the plan itself: a step's srcA/srcB/srcC field encodes operand
            // i as kPwRefOp0 - i, indexing node.inputs. Reading the step words rather than assuming
            // a contiguous tail covers both carriers of a unit — an epilogue host, whose operands
            // start at pwCoreInputs(), and a standalone FusedPointwise node, whose plan may name any
            // input — and a future step layout that names one somewhere else again.
            // The contiguous operand tail an epilogue host declares through pw_opbase.
            for (size_t k = pwCoreInputs(nd); k < nd.inputs.size(); ++k)
            {
                keep(nd.inputs[k]);
            }
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
