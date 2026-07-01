// Cast on the GPU. A float->float cast is a pure buffer copy (the segment runs in one precision, so the
// cast is a no-op). A float->integer cast (UINT8/INT8/INT16/INT32/INT64/BOOL) truncates toward zero and
// clamps to the target range via cast.comp, keeping compute-precision storage — the flat path carries a
// logical integer as a truncated fp32/fp16 value, and the graph boundary repacks it to the declared dtype.
// supportsNode keeps a cast whose INPUT is int64 (a CPU int64 shape/index tensor) on the CPU op.
#include "vk_op_common.h"
#include "vknn/op.h"
#include <limits>

namespace vknn {
    namespace {

        struct CastOp: VulkanOp {
            struct PC {
                int   total;
                float lo, hi;
            } pc {};
            bool                                 truncate = false;
            std::unique_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          hold0; // when input is a constant initializer

            void prepare(const Node &node, VkOpEnv &env) override {
                int64_t to = node.attr.geti("to", 1); // ONNX TensorProto dtype
                // integer targets that need a value truncation (not a same-precision copy)
                switch (to)
                {
                    case 2:
                        truncate = true;
                        pc.lo    = 0.0f;
                        pc.hi    = 255.0f;
                        break; // UINT8
                    case 3:
                        truncate = true;
                        pc.lo    = -128.0f;
                        pc.hi    = 127.0f;
                        break; // INT8
                    case 4:
                        truncate = true;
                        pc.lo    = 0.0f;
                        pc.hi    = 65535.0f;
                        break; // UINT16
                    case 5:
                        truncate = true;
                        pc.lo    = -32768.0f;
                        pc.hi    = 32767.0f;
                        break; // INT16
                    case 9:
                        truncate = true;
                        pc.lo    = 0.0f;
                        pc.hi    = 1.0f;
                        break; // BOOL
                    case 6:
                    case 7:
                    case 12:
                    case 13:
                        truncate = true;
                        pc.lo    = -3.4e38f;
                        pc.hi    = 3.4e38f;
                        break; // INT32/INT64/UINT32/UINT64: truncate only
                    default:
                        truncate = false;
                        break; // FLOAT/FLOAT16/DOUBLE -> copy
                }
                if (truncate)
                {
                    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("cast", env.useFp16), 2, sizeof(PC), std::vector<uint32_t> {}, env.cache->handle());
                }
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = operandBuf(env, node.inputs[0], hold0);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                if (!truncate)
                {
                    VkBufferCopy c {0, 0, std::min(src->bytes(), dst->bytes())};
                    vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
                    return;
                }
                // Truncate every storage slot of the output buffer (element-wise, layout-agnostic; NC4HW4
                // channel padding is truncated harmlessly). elemSize matches the compute precision.
                int elemSize = env.useFp16 ? 2 : 4;
                pc.total     = (int) (dst->bytes() / elemSize);
                pipe->dispatch(cmd, {src->handle(), dst->handle()}, &pc, sizeof(pc), groups(pc.total, 256));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Cast, CastOp);
} // namespace vknn
