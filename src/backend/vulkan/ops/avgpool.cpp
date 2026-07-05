// Windowed AveragePool2D on the GPU (NC4HW4). GlobalAveragePool has its own op; this is the
// kernel/stride/pad form used by Inception/SqueezeNet.
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct AvgPoolOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            AvgPC                                pc {};
            PwEpi                                epi;
            int64_t                              total = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                NCHW x    = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                NCHW y    = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                auto ints = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto ks  = ints("kernel_shape", {1, 1});
                auto st  = ints("strides", {1, 1});
                auto pad = ints("pads", {0, 0, 0, 0});
                // Positional fields map 1:1 to AvgPC / the shader's PC block (N,C,H,W,OH,OW,KH,KW,SH,SW,PT,PL,
                // countIncludePad). count_include_pad follows ONNX: 0 (default) divides by the in-bounds window
                // count, 1 divides by the full KH*KW, so its value changes the numerical result.
                pc       = {(int) x.n,
                            (int) x.c,
                            (int) x.h,
                            (int) x.w,
                            (int) y.h,
                            (int) y.w,
                            (int) ks[0],
                            (int) ks[1],
                            (int) st[0],
                            (int) st[1],
                            (int) pad[0],
                            (int) pad[1],
                            (int) node.attr.geti("count_include_pad", 0)};
                // One thread per NC4HW4 output block-pixel: a thread emits one vec4 (four packed channels),
                // so the dispatch counts channel blocks (ceil(C/4)) rather than channels.
                total    = x.n * cBlocks(x.c) * y.h * y.w;
                epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader((std::string("avgpool2d") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(AvgPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *src  = env.devBuf(node.inputs[0]);
                vk::Buffer           *dst  = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {src->handle(), dst->handle()};
                epi.append(bufs, node, env, dst->handle());
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, 64));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::AvgPool, AvgPoolOp);
} // namespace vknn
