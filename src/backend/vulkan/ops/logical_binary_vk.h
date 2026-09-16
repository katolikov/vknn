// Flat (row-major) two-operand logical kernel shared by Or and Xor on the GPU, with N-D broadcasting at
// any rank. Each op file derives from LogicalBinaryVk with its shader stem; the shaders differ only in the
// boolean combine.
//
// Layout byte-matches shaders/or.comp and shaders/xor.comp:
//   binding 0  STORE a[]   operand A (activation buffer, or the canonical constant upload)
//   binding 1  STORE b[]   operand B (same)
//   binding 2  STORE d[]   output, 1.0 / 0.0
//   binding 3  int g[]     geometry: outDim, aStride, bStride packed back to back (logical_geometry.h)
//   push constant { int rank; int total; }
// The node always runs on the flat path (descriptor LayoutClass::Flat). Constant operands upload in
// prepare() at the node's storage precision after logical::canonicalConstantOperand (rank-0 safe).
#pragma once
#include "flat_ops.h"
#include "logical_geometry.h"
#include "vk_op_common.h"
#include <string>

namespace vknn {

    struct LogicalBinaryVk: VulkanOp {
        /// Descriptor bindings of the shader: two operands, the output, the geometry SSBO.
        static constexpr uint32_t kBindingCount = 4;
        static constexpr size_t   kOperandCount = 2;
        static constexpr int      kOperandA     = 0;
        static constexpr int      kOperandB     = 1;

        struct PC {
            int rank, total;
        } pc {};
        std::shared_ptr<vk::ComputePipeline> pipe;
        std::shared_ptr<vk::Buffer>          geom;
        std::shared_ptr<vk::Buffer>          constBuf[kOperandCount];
        const char                          *shaderStem;

        explicit LogicalBinaryVk(const char *stem): shaderStem(stem) {
        }

        void prepare(const Node &node, VkOpEnv &env) override {
            const Graph &g = *env.graph;
            if (node.inputs.size() < kOperandCount || node.inputs[kOperandA] == kNoTensor || node.inputs[kOperandB] == kNoTensor || node.outputs.empty() || node.outputs[0] == kNoTensor)
            {
                throw Error(Status::InvalidArgument, std::string(opTypeName(node.type)) + " '" + node.name + "': expects " + std::to_string(kOperandCount) + " operands and one output");
            }
            const std::string                    nodeLabel = std::string(opTypeName(node.type)) + " '" + node.name + "'";
            const Shape                          out       = g.desc(node.outputs[0]).shape;
            const logical::FlatBroadcastGeometry geometry =
                logical::flatBroadcastGeometry(out, g.desc(node.inputs[kOperandA]).shape, g.desc(node.inputs[kOperandB]).shape, nodeLabel);
            pc.rank  = geometry.rank;
            pc.total = (int) logical::flatElementCount(out);
            for (int operand = kOperandA; operand <= kOperandB; ++operand)
            {
                TensorId id = node.inputs[(size_t) operand];
                if (g.isInitializer(id))
                {
                    const std::vector<float> canonical = logical::canonicalConstantOperand(initFloats(g, id), logical::flatElementCount(g.desc(id).shape));
                    constBuf[operand]                  = upload(*env.ctx, canonical, env.useFp16);
                }
            }
            geom = flat::uploadFlatGeom(env, {geometry.outDim, geometry.aStride, geometry.bStride});
            pipe = env.pipeline(shader(shaderStem, env.useFp16), kBindingCount, sizeof(PC), std::vector<uint32_t> {});
        }

        void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
            auto operandBuffer = [&](int operand) {
                return constBuf[operand] ? constBuf[operand].get() : env.devBuf(node.inputs[(size_t) operand]);
            };
            // Buffers in binding order (operand A, operand B, output, geometry). One flat invocation per output
            // element; the shaders are local_size_x=256 == flat::kFlatLocalSize.
            pipe->dispatch(cmd, {operandBuffer(kOperandA)->handle(), operandBuffer(kOperandB)->handle(), env.devBuf(node.outputs[0])->handle(), geom->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
        }
    };

} // namespace vknn
