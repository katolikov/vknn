// Flat ConstantOfShape on the GPU: fill the plan-time output size with the scalar `value` attr.
// Most ConstantOfShape nodes const-fold; one survives when its element count exceeds the fold
// bound, and this kernel keeps it off the CPU. A float or an integer `value` both fill as the
// compute-precision float — an integer index/shape magnitude is exact in the float, and the graph
// boundary repacks the declared int32/int64 dtype on readback. Only an unresolved output size
// (data-dependent shape input) stays on the exact CPU op.
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        // 1D workgroup width; matches `layout(local_size_x = 256)` in shaders/constant_of_shape.comp.
        // The dispatch group count is derived from this exact value so the launched thread grid covers
        // every output element with no gap or overshoot.
        constexpr uint32_t kConstantOfShapeLocalSize = 256;

        struct ConstantOfShapeVk: VulkanOp {
            // Mirrors constant_of_shape.comp's push_constant block. `count` is the flat (logical)
            // output element count; `value` stays fp32 even under fp16 storage because the shader casts
            // it once via STORE(pc.value) at write time.
            struct PC {
                uint32_t count;
                float    value;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;

            void prepare(const Node &node, VkOpEnv &env) override {
                // The shader writes a plain linear buffer (d[i]), so the count is the logical element
                // product, not an NC4HW4 packed count.
                pc.count = (uint32_t) numElements(env.graph->desc(node.outputs[0]).shape);
                // The importer records an integer `value` as an Ints attribute and a float `value` as
                // Floats. Carry either as the compute-precision float fill; an integer index/shape value
                // is exact in the float and the graph boundary repacks the declared int dtype on readback.
                // A missing/empty `value` defaults to 0 per ONNX.
                auto it  = node.attr.map.find("value");
                pc.value = 0.f;
                if (it != node.attr.map.end())
                {
                    if (it->second.kind == Attr::Ints && !it->second.ints.empty())
                    {
                        pc.value = (float) it->second.ints[0];
                    } else if (!it->second.floats.empty())
                    { pc.value = it->second.floats[0]; }
                }
                pipe = env.pipeline(shader("constant_of_shape", env.useFp16), 1, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                pipe->dispatch(cmd, {env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.count, kConstantOfShapeLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ConstantOfShape, ConstantOfShapeVk);
} // namespace vknn
