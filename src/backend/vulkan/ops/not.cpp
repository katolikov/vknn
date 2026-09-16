// Flat (row-major) boolean NOT on the GPU: out = (x != 0) ? 0 : 1, a pure 1:1 elementwise map. The node
// always runs on the flat path (descriptor LayoutClass::Flat). No broadcasting and no axis, so it needs
// no geometry SSBO: the push constant carries only the element count.
//
// Layout byte-matches shaders/not.comp:
//   binding 0  STORE s[]   operand (activation buffer, or the canonical constant upload)
//   binding 1  STORE d[]   output, 1.0 / 0.0
//   push constant { int total; }
// A constant operand (one too large for constFold's bounded logical group) uploads in prepare() at the
// node's storage precision after logical::canonicalConstantOperand, which is rank-0 safe and keeps the
// truth of values an fp16 upload would round to zero.
#include "flat_ops.h"
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
                const Graph &g = *env.graph;
                if (node.inputs.empty() || node.inputs[0] == kNoTensor || node.outputs.empty() || node.outputs[0] == kNoTensor)
                {
                    throw Error(Status::InvalidArgument, "Not '" + node.name + "': expects one operand and one output");
                }
                const Shape   out   = g.desc(node.outputs[0]).shape;
                const int64_t total = logical::flatElementCount(out);
                if (total > logical::kMaxShaderElements)
                {
                    throw Error(Status::InvalidArgument, "Not '" + node.name + "': " + std::to_string(total) + " elements exceed the shader index range");
                }
                pc.total    = (int) total;
                TensorId id = node.inputs[0];
                if (g.isInitializer(id))
                {
                    const std::vector<float> canonical = logical::canonicalConstantOperand(initFloats(g, id), logical::flatElementCount(g.desc(id).shape));
                    constBuf                           = upload(*env.ctx, canonical, env.useFp16);
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
