// ConvTranspose2D on the GPU via the flat (row-major NCHW) path: one thread per output element,
// gather form (see shaders/convtranspose.comp). Weights/bias are constant initializers uploaded
// flat in prepare(). Gated by supportsNode to 4D input + constant weight; everything else (runtime
// weight, non-4D) falls back to the CPU oracle. Push-constant block byte-matches the shader.
#include "core/conv_geom.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct ConvTransposeVk: VulkanOp {
            struct PC {
                int N, Cin, Cout, H, W, outH, outW, kH, kW;
                int sh, sw, pt, pl, dh, dw, inCg, outCg, total, hasBias;
            } pc {};
            std::unique_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          wbuf, bbuf;

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g   = *env.graph;
                Shape        in  = g.desc(node.inputs[0]).shape;  // [N, Cin, H, W]
                Shape        w   = g.desc(node.inputs[1]).shape;  // [Cin, Cout/group, kH, kW]
                Shape        out = g.desc(node.outputs[0]).shape; // [N, Cout, outH, outW]

                auto ints = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto    strides = ints("strides", {1, 1});
                auto    dil     = ints("dilations", {1, 1});
                int64_t group   = node.attr.geti("group", 1);

                ConvTransposeGeom geom = convTransposeGeom(in[2], in[3], w[2], w[3], node.attr);

                pc.N     = (int) in[0];
                pc.Cin   = (int) in[1];
                pc.H     = (int) in[2];
                pc.W     = (int) in[3];
                pc.Cout  = (int) out[1];
                pc.outH  = (int) out[2];
                pc.outW  = (int) out[3];
                pc.kH    = (int) w[2];
                pc.kW    = (int) w[3];
                pc.sh    = (int) strides[0];
                pc.sw    = (int) strides[1];
                pc.pt    = (int) geom.padH;
                pc.pl    = (int) geom.padW;
                pc.dh    = (int) dil[0];
                pc.dw    = (int) dil[1];
                pc.outCg = (int) w[1];
                pc.inCg  = (int) (in[1] / group);
                pc.total = (int) numElements(out);

                std::vector<float> wv = initFloats(g, node.inputs[1]);
                wbuf                  = upload(*env.ctx, wv, env.useFp16);

                const bool hasBias    = node.inputs.size() > 2 && node.inputs[2] != kNoTensor && g.isInitializer(node.inputs[2]);
                pc.hasBias            = hasBias ? 1 : 0;
                std::vector<float> bv = hasBias ? initFloats(g, node.inputs[2]) : std::vector<float>(pc.Cout, 0.f);
                bv.resize(pc.Cout);
                bbuf = upload(*env.ctx, bv, env.useFp16);

                pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("convtranspose", env.useFp16), 4, sizeof(PC), std::vector<uint32_t> {}, env.cache->handle());
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                pipe->dispatch(cmd, {env.devBuf(node.inputs[0])->handle(), wbuf->handle(), bbuf->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, 256));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ConvTranspose, ConvTransposeVk);
} // namespace vknn
