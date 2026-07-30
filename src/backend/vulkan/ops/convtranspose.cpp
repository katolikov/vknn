// ConvTranspose2D on the GPU via the flat (row-major NCHW) path: one thread per output element,
// gather form (see shaders/convtranspose.comp). The weight (ONNX [Cin, Cout/group, kH, kW]) and the
// optional bias are read flat by the shader. A constant weight/bias uploads flat in prepare(); a
// RUNTIME (computed) weight/bias binds its activation buffer at dispatch — the shader indexes ws[]/bs[]
// identically in both cases. Gated by supportsNode to a 4D input; a non-4D input falls back to the CPU
// oracle. Push-constant block byte-matches the shader.
#include "core/conv_geom.h"
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        // Local workgroup size along x; matches local_size_x in shaders/convtranspose.comp.
        constexpr uint32_t kConvTransposeLocalSize = 256;

        struct ConvTransposeVk: VulkanOp {
            struct PC {
                int N, Cin, Cout, H, W, outH, outW, kH, kW;
                int sh, sw, pt, pl, dh, dw, inCg, outCg, total, hasBias;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          wbuf, bbuf; // hold the CONST weight / const-or-zero bias; null for a runtime operand
            bool                                 rtWeight = false, rtBias = false;
            PwEpi                                epi;

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

                // A constant weight uploads flat here; a runtime weight binds its activation buffer at
                // dispatch (record()).
                rtWeight = !g.isInitializer(node.inputs[1]);
                if (!rtWeight)
                {
                    std::vector<float> wv = initFloats(g, node.inputs[1]);
                    wbuf                  = upload(*env.ctx, wv, env.useFp16);
                }

                // Bias binding (binding 2) is unconditional in the descriptor set, so a Cout-length
                // buffer must exist even when the node has no bias: absent-bias uploads zeros. A runtime
                // bias binds its activation buffer at dispatch (a real Cout-length tensor). pc.hasBias
                // gates the shader read of bs[oc].
                const bool hasBias = pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor;
                pc.hasBias         = hasBias ? 1 : 0;
                rtBias             = hasBias && !g.isInitializer(node.inputs[2]);
                if (!rtBias)
                {
                    std::vector<float> bv = hasBias ? initFloats(g, node.inputs[2]) : std::vector<float>(pc.Cout, 0.f);
                    bv.resize(pc.Cout);
                    bbuf = upload(*env.ctx, bv, env.useFp16);
                }

                epi.prepare(node, env, /*flat=*/true, out);
                pipe = env.pipeline(shader((std::string("convtranspose") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                VkBuffer              dst  = env.devBuf(node.outputs[0])->handle();
                VkBuffer              w    = rtWeight ? env.devBuf(node.inputs[1])->handle() : wbuf->handle();
                VkBuffer              b    = rtBias ? env.devBuf(node.inputs[2])->handle() : bbuf->handle();
                std::vector<VkBuffer> bufs = {env.devBuf(node.inputs[0])->handle(), w, b, dst};
                epi.append(bufs, node, env, dst);
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.total, kConvTransposeLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ConvTranspose, ConvTransposeVk);
} // namespace vknn
