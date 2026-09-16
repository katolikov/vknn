// Device buffer of a constant operand of the flat logical kernels (Or, Xor, Not). The source rule lives in
// logical_geometry.h (constantOperandSource), where the host tests run it; this header applies it to a
// segment's VkOpEnv.
#pragma once
#include "logical_geometry.h"
#include "vk_op_common.h"
#include <memory>
#include <optional>
#include <string>

namespace vknn { namespace logical {

    /// Device buffer of constant operand `id` of the logical node labelled `nodeLabel` (e.g. "Or 'mask'"),
    /// at the node's storage precision (env.useFp16): the canonical 1.0 / 0.0 upload of its host payload
    /// (rank-0 safe), or the flat device copy an earlier consumer uploaded when that upload released the
    /// payload.
    /// @throws Error(InvalidArgument) when the payload does not cover the operand and no device copy
    ///         reads correctly (constantOperandSource).
    inline std::shared_ptr<vk::Buffer> constantOperandBuffer(VkOpEnv &env, TensorId id, const std::string &nodeLabel) {
        const Graph                      &g                = *env.graph;
        const TensorDesc                 &desc             = g.desc(id);
        const std::shared_ptr<vk::Buffer> sharedDeviceCopy = env.lookupFlatWeight ? env.lookupFlatWeight(id) : nullptr;
        std::optional<size_t>             sharedDeviceBytes;
        if (sharedDeviceCopy)
        {
            sharedDeviceBytes = sharedDeviceCopy->bytes();
        }
        const std::string operandLabel = nodeLabel + ": constant operand '" + desc.name + "'";
        const ConstantOperandSource source = constantOperandSource(g.initializers.at(id).bytes.size(), desc.shape, desc.dtype, sharedDeviceBytes, env.useFp16, operandLabel);
        if (source == ConstantOperandSource::SharedDeviceCopy)
        {
            return sharedDeviceCopy;
        }
        return upload(*env.ctx, canonicalConstantOperand(initFloats(g, id), flatElementCount(desc.shape)), env.useFp16);
    }

}} // namespace vknn::logical
