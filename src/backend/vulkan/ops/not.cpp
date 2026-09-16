// Flat (row-major) boolean NOT on the GPU: out = (x != 0) ? 0 : 1, a pure 1:1 elementwise map. The node
// always runs on the flat path (descriptor LayoutClass::Flat). No broadcasting and no axis, so it needs
// no geometry SSBO: the push constant carries only the element count.
//
// Layout byte-matches shaders/not.comp:
//   binding 0  STORE operand[]  operand (activation buffer, or the constant operand buffer)
//   binding 1  STORE result[]   output, 1.0 / 0.0
//   push constant { int total; }
// A constant operand (one too large for constFold's bounded logical group) resolves in prepare() through
// logical::constantOperandBuffer at the node's storage precision (logical_constant_vk.h): the canonical
// upload keeps the truth of values an fp16 upload would round to zero, and a payload an earlier
// consumer's upload released reads that consumer's device copy or fails the prepare, never as all false.
#include "flat_ops.h"
#include "logical_constant_vk.h"
#include "logical_geometry.h"
#include "vk_op_common.h"
#include <string>

namespace vknn {
    namespace {

        struct NotVk: VulkanOp {
            /// Descriptor bindings of the shader: the operand and the output.
            static constexpr uint32_t kBindingCount = 2;

            struct PC {
                int total;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          constBuf; // set when the operand is a constant initializer

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph      &g         = *env.graph;
                const std::string nodeLabel = "Not '" + node.name + "'";
                if (node.inputs.empty() || node.inputs[0] == kNoTensor || node.outputs.empty() || node.outputs[0] == kNoTensor)
                {
                    throw Error(Status::InvalidArgument, nodeLabel + ": expects one operand and one output");
                }
                pc.total    = logical::shaderElementCount(g.desc(node.outputs[0]).shape, nodeLabel);
                TensorId id = node.inputs[0];
                if (g.isInitializer(id))
                {
                    constBuf = logical::constantOperandBuffer(env, id, nodeLabel);
                }
                pipe = env.pipeline(shader("not", env.useFp16), kBindingCount, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *source      = constBuf ? constBuf.get() : env.devBuf(node.inputs[0]);
                vk::Buffer *destination = env.devBuf(node.outputs[0]);
                // One flat invocation per element; not.comp is local_size_x=256 == flat::kFlatLocalSize.
                pipe->dispatch(cmd, {source->handle(), destination->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Not, NotVk);
} // namespace vknn
