// ConvGemm: a Conv lowered by lowerConv to one implicit-GEMM kernel over the receptive field
// (out[m, oc] = sum_k patch(m, k) * Wt[k, oc]). The weights arrive repacked [K][Cout] from convert
// time, both panels stage through LDS, and the fp32 K reduction runs in a fixed chunked order —
// deterministic, fp16-floor equivalent to the direct Conv kernel (the summation order differs).
// Serves the KxK shapes the Winograd path rejects (strided, dilated, shallow, non-square).
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        // Mirror of conv_gemm.comp's push_constant block and tile geometry.
        struct ConvGemmPC {
            int   C, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW;
            int   act, hasBias;
            float actLo, actHi;
        };
        constexpr int kTileM = 64, kTileN = 64;

        struct ConvGemmOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          wt, bs;
            ConvGemmPC                           pc {};
            PwEpi                                epi;
            uint32_t                             gx = 0, gy = 0, gz = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g   = *env.graph;
                NCHW         x   = NCHW::from(g.desc(node.inputs[0]).shape);
                Shape        out = g.desc(node.outputs[0]).shape;
                NCHW         y   = NCHW::from(out);
                auto         a   = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto k = a("kernel_shape", {1, 1}), st = a("strides", {1, 1});
                auto p = a("pads", {0, 0, 0, 0}), dl = a("dilations", {1, 1});

                pc = {(int) x.c,  (int) x.h,  (int) x.w,  (int) y.c,  (int) y.h,  (int) y.w,
                      (int) k[0], (int) k[1], (int) st[0], (int) st[1], (int) p[0], (int) p[1],
                      (int) dl[0], (int) dl[1], (int) node.fusedAct, node.inputs.size() > 2 && node.inputs[2] != kNoTensor ? 1 : 0,
                      node.actLo, node.actHi};

                wt = uploadInit(env, node.inputs[1], g.desc(node.inputs[1]).shape);
                if (pc.hasBias)
                {
                    bs = uploadInit(env, node.inputs[2], g.desc(node.inputs[2]).shape);
                }
                epi.prepare(node, env, false, out);
                pipe = env.pipeline(shader((std::string("conv_gemm") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(ConvGemmPC), std::vector<uint32_t> {});

                int64_t M = y.h * y.w;
                gx        = (uint32_t) ((y.c + kTileN - 1) / kTileN);
                gy        = (uint32_t) ((M + kTileM - 1) / kTileM);
                gz        = (uint32_t) y.n;
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *dst = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs {env.devBuf(node.inputs[0])->handle(), wt->handle(), pc.hasBias ? bs->handle() : dst->handle(), dst->handle()};
                epi.append(bufs, node, env, dst->handle());
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), gx, gy, gz);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ConvGemm, ConvGemmOp);
} // namespace vknn
