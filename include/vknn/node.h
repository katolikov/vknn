// IR node struct: an op instance with its tensor ids, attributes, and fusion metadata.
#pragma once
#include "vknn/act_type.h"
#include "vknn/attributes.h"
#include "vknn/op_type.h"
#include "vknn/tensor_id.h"
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {

    /// One operator instance in the graph IR: an op code plus the tensor ids it reads and writes,
    /// its parsed attributes, and metadata describing any pointwise/activation ops fused into it.
    /// Tensors are referenced by id into Graph::tensors rather than by pointer.
    struct Node {
        OpType                type = OpType::Unknown; ///< Op code; OpType::Unknown until the importer resolves it.
        std::string           name;                   ///< Source name of the op (diagnostics; may be empty).
        std::vector<TensorId> inputs;                 ///< Operand tensor ids. May carry fused pointwise-chain operands past pwCoreInputs().
        std::vector<TensorId> outputs;                ///< Result tensor ids.
        Attributes            attr;                   ///< Parsed op attributes (kernel shape, pads, fusion plan, ...).
        /// Activation folded into this op's epilogue, or ActType::None when unfused.
        ActType fusedAct = ActType::None;
        /// Clamp/param bounds for fusedAct (e.g. Relu6/Clip min and max), in [actLo, actHi].
        float actLo = 0, actHi = 0;
        /// For kUnary/kBinary: the UnaryType/BinaryType code. For unary ops with params (LeakyRelu/Elu
        /// alpha, HardSigmoid alpha/beta) the params live in actLo/actHi.
        int32_t subOp = 0;
        /// Conv only: a residual tensor fused into the epilogue (out = act(conv + residual)).
        /// kNoTensor when no residual is fused.
        TensorId fusedResidual = kNoTensor;
        /// MatMul only: a rank-1 [N] bias initializer added into the fp32 accumulator before the store
        /// (out[...,n] = matmul + bias[n]). kNoTensor when no bias is fused.
        TensorId fusedBias = kNoTensor;
    };

    /// The node's own operand count, excluding any fused pointwise-chain operands.
    ///
    /// fusePointwiseChains appends its chain-operand tensors to node.inputs (from index pw_opbase on)
    /// so liveness/DCE/scheduling see them; any positional read of an op's optional inputs (bias,
    /// scales, axes, ...) must bound the search by this value, not by inputs.size().
    ///
    /// @param n Node to inspect.
    /// @returns inputs.size() when no chain is fused; otherwise the pw_opbase split point, clamped to
    ///          be non-negative.
    inline size_t pwCoreInputs(const Node &n) {
        if (!n.attr.has("pw_steps"))
        {
            return n.inputs.size();
        }
        int64_t base = n.attr.geti("pw_opbase", (int64_t) n.inputs.size());
        // Clamp both ends: an out-of-range pw_opbase from a crafted/bit-flipped .vxm must not let a
        // caller iterate n.inputs past its end (e.g. concat's range-ctor) or before its begin.
        base = base < 0 ? 0 : base;
        return (size_t) base < n.inputs.size() ? (size_t) base : n.inputs.size();
    }

} // namespace vknn
