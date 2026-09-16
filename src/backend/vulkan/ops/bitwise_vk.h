// Shared pieces of the flat Vulkan bitwise kernels: the two-operand broadcast geometry and constant
// operand upload (BitwiseAnd/Or/Xor, BitShift), the BitwiseAnd/Or/Xor kernel parameterized by its operator
// specialization constant, and constant operand upload for the 1:1 BitwiseNot.
//
// The ops compute integers on float lanes and write non-0/1 values; pinIntegerResultsFp32 stores their
// outputs (and operands) at fp32 at load, so env.useFp16 is false for them in an fp16 segment and they run
// the fp32 variants. The fp16 variants remain valid (exact for integers within +-2048) for an unpinned
// node. A constant operand uploads at the node's precision through initFloats + upload(), which decodes
// every initializer dtype (Int64 lanes included) and keeps a rank-0 scalar's value.
#pragma once
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn { namespace bitwise_vk {

    /// Specialization-constant values selecting the operator of shaders/bitwise.comp (its kOperator*).
    constexpr uint32_t kBitwiseOperatorAnd = 0;
    constexpr uint32_t kBitwiseOperatorOr  = 1;
    constexpr uint32_t kBitwiseOperatorXor = 2;

    /// Operand input slots and the SSBO count of a two-operand flat broadcast kernel: operand A (binding
    /// 0), operand B (binding 1), output (binding 2), geometry (binding 3).
    constexpr int      kOperandA             = 0;
    constexpr int      kOperandB             = 1;
    constexpr int      kOperandCount         = 2;
    constexpr uint32_t kBroadcastBufferCount = 4;

    /// Upload constant initializer `t` flat at the node's precision (rank-0 safe: a scalar keeps its one
    /// element).
    inline std::shared_ptr<vk::Buffer> uploadConstantOperand(VkOpEnv &env, TensorId t) {
        const Graph       &g      = *env.graph;
        std::vector<float> values = initFloats(g, t);
        values.resize((size_t) std::max<int64_t>(1, numElements(g.desc(t).shape)));
        return upload(*env.ctx, values, env.useFp16);
    }

    /// Output extent, per-operand broadcast strides (geometry SSBO) and constant operand buffers of a
    /// two-operand flat broadcast kernel, in the layout shaders/bitwise.comp and shaders/bitshift.comp
    /// decode (outDim = g[0..rank), aStride = g[rank..2*rank), bStride = g[2*rank..3*rank)).
    struct BroadcastOperands {
        int                         rank  = 0;
        int                         total = 0;
        std::shared_ptr<vk::Buffer> geometry;
        std::shared_ptr<vk::Buffer> constants[kOperandCount];

        void prepare(const Node &node, VkOpEnv &env) {
            const Graph &g   = *env.graph;
            const Shape  out = g.desc(node.outputs[0]).shape;
            rank             = (int) out.size();
            total            = (int) numElements(out);
            std::vector<int32_t> outDim(rank);
            std::vector<int32_t> strides[kOperandCount];
            for (int k = 0; k < rank; ++k)
            {
                outDim[k] = (int) out[k];
            }
            for (int operand = 0; operand < kOperandCount; ++operand)
            {
                const TensorId       t     = node.inputs[operand];
                const Shape          shape = g.desc(t).shape;
                std::vector<int64_t> padded(rank, 1); // right-aligned into the output rank, leading dims 1
                for (int k = 0; k < (int) shape.size(); ++k)
                {
                    padded[rank - (int) shape.size() + k] = shape[k];
                }
                const std::vector<int64_t> rowStride = flat::rowStrides(padded);
                strides[operand].resize(rank);
                for (int k = 0; k < rank; ++k)
                {
                    // A size-1 axis gets stride 0 so every output coordinate along it reads the same element.
                    strides[operand][k] = padded[k] == 1 ? 0 : (int) rowStride[k];
                }
                if (g.isInitializer(t))
                {
                    constants[operand] = uploadConstantOperand(env, t);
                }
            }
            geometry = flat::uploadFlatGeom(env, {outDim, strides[kOperandA], strides[kOperandB]});
        }

        /// Device buffer of operand `operand`: its uploaded constant, or the activation buffer.
        vk::Buffer *operandBuffer(const Node &node, VkOpEnv &env, int operand) const {
            return constants[operand] ? constants[operand].get() : env.devBuf(node.inputs[operand]);
        }
    };

    /// BitwiseAnd / BitwiseOr / BitwiseXor on shaders/bitwise.comp; the operator is the kernel's
    /// specialization constant, so each op shares one source with no runtime operator branch.
    struct BitwiseBinaryVk: VulkanOp {
        /// Byte-matches the push_constant block of shaders/bitwise.comp.
        struct PushConstants {
            int rank, total;
        } pc {};
        const uint32_t                       operatorCode;
        BroadcastOperands                    operands;
        std::shared_ptr<vk::ComputePipeline> pipe;

        explicit BitwiseBinaryVk(uint32_t bitwiseOperator): operatorCode(bitwiseOperator) {
        }

        void prepare(const Node &node, VkOpEnv &env) override {
            operands.prepare(node, env);
            pc.rank  = operands.rank;
            pc.total = operands.total;
            pipe     = env.pipeline(shader("bitwise", env.useFp16), kBroadcastBufferCount, sizeof(PushConstants), std::vector<uint32_t> {operatorCode});
        }

        void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
            // One flat invocation per output element; bitwise.comp is local_size_x=256 == flat::kFlatLocalSize.
            pipe->dispatch(cmd,
                           {operands.operandBuffer(node, env, kOperandA)->handle(), operands.operandBuffer(node, env, kOperandB)->handle(),
                            env.devBuf(node.outputs[0])->handle(), operands.geometry->handle()},
                           &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
        }
    };

}} // namespace vknn::bitwise_vk
