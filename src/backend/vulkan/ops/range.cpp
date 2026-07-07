// Flat Range on the GPU: y[i] = start + i * delta over the plan-time output size. start/delta are
// scalar operands — constants upload once (int64/int32 lanes decode to fp32 via initFloats), runtime
// scalars bind from their flat buffers (the static plan fixes the element count; the values may still
// be computed on-GPU). An integer start/delta generates the ramp in compute-precision float — exact
// for the index magnitudes these produce — and the graph boundary repacks the declared int32/int64
// dtype on readback. Only a Range whose size cannot resolve at plan time stays on the exact CPU op.
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        // 1D workgroup width; matches `layout(local_size_x = 256)` in shaders/range.comp. The dispatch
        // group count is derived from this exact value so the launched thread grid covers every output
        // element (one thread per element) with no gap or overshoot.
        constexpr uint32_t kRangeLocalSize = 256;

        struct RangeVk: VulkanOp {
            // Mirrors range.comp's push_constant block. `total` is the flat (logical) output element
            // count; the plan fixes it, so the kernel never needs the ONNX `limit` operand at runtime.
            struct PC {
                int total;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          startHold, deltaHold;

            void prepare(const Node &node, VkOpEnv &env) override {
                pc.total = (int) numElements(env.graph->desc(node.outputs[0]).shape);
                pipe     = env.pipeline(shader("range", env.useFp16), 3, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // ONNX Range inputs are (start, limit, delta). Only start and delta feed the kernel:
                // the plan-time output size already encodes the element count, so `limit` (inputs[1]) is
                // unused here. operandBuf() resolves each scalar to a device buffer — an activation keeps
                // its own buffer, a constant uploads flat into the *Hold shared_ptr on first use.
                vk::Buffer *s = operandBuf(env, node.inputs[0], startHold);
                vk::Buffer *d = operandBuf(env, node.inputs[2], deltaHold);
                // Bind order {start, delta, output} matches range.comp bindings 0/1/2; groups() rounds
                // pc.total up to whole workgroups of kRangeLocalSize.
                pipe->dispatch(cmd, {s->handle(), d->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, kRangeLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Range, RangeVk);
} // namespace vknn
