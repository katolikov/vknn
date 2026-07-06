// Inference BatchNorm on the GPU. For CONSTANT params (the common case) the 4 inputs fold into a
// per-channel affine (scale,bias) on the host, then y = scale[c]*x + bias[c] runs over NC4HW4
// (batchnorm.comp). For RUNTIME params (a computed gamma/beta/mean/var, not baked initializers) the
// fold can't happen on the host, so the 4 params bind as SSBOs and batchnorm_rt.comp folds per channel
// at dispatch. For pre-activation nets (DenseNet) whose BN can't fold into a preceding conv. inputs:
// X, gamma, beta, mean, var.
#include "vk_op_common.h"
#include <cmath>

namespace vknn {
    namespace {

        // Shader workgroup size (matches local_size_x in shaders/batchnorm*.comp); one invocation
        // per packed NC4HW4 element.
        constexpr uint32_t kBnLocalSize = 256;

        // Push constants for the const-fold shader's gid -> channel decode: `count` is the total packed
        // NC4HW4 element count (loop bound), `Cb` the number of 4-channel blocks, `HW` the spatial
        // slab size. The shader recovers a channel from a flat gid as c = ((gid/4/HW) % Cb)*4 + gid%4.
        struct BnPC {
            int count, Cb, HW;
        };
        // Runtime-param variant adds the logical channel count C (to bound the param read to real
        // channels) and eps (folded on the host in the const path, applied in-shader here).
        struct BnRtPC {
            int   count, Cb, HW, C;
            float eps;
        };

        struct BatchNormOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          scaleBuf, biasBuf; // const path: host-folded affine
            bool                                 runtime = false;   // any of the 4 params is a runtime tensor
            BnPC                                 pc {};
            BnRtPC                               rtpc {};

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                NCHW         x = NCHW::from(g.desc(node.inputs[0]).shape);
                // Cb = ceil(C/4) blocks; padded rounds the channel count up to a multiple of 4 so the
                // per-channel scale/bias arrays cover the NC4HW4 padding lanes. Those pad lanes stay
                // zero (scale=0, bias=0 below), so any padded output they produce is harmless.
                int64_t Cb = cBlocks(x.c), padded = Cb * 4;
                float   eps = node.attr.getf("epsilon", 1e-5f);

                for (int i = 1; i <= 4; ++i)
                {
                    if (!g.isInitializer(node.inputs[i]))
                    {
                        runtime = true;
                    }
                }
                if (runtime)
                {
                    // Runtime params: bind gamma/beta/mean/var directly and fold per channel in the
                    // shader. The params are 1-D [C]; in NC4HW4 (H=W=1) channel c lands at index c.
                    rtpc = {(int) (x.n * Cb * 4 * x.h * x.w), (int) Cb, (int) (x.h * x.w), (int) x.c, eps};
                    pipe = env.pipeline(shader("batchnorm_rt", env.useFp16), 6, sizeof(BnRtPC), std::vector<uint32_t> {});
                    return;
                }

                std::vector<float> gammaV = initFloats(g, node.inputs[1]);
                std::vector<float> betaV  = initFloats(g, node.inputs[2]);
                std::vector<float> meanV  = initFloats(g, node.inputs[3]);
                std::vector<float> varV   = initFloats(g, node.inputs[4]);
                const float       *gamma  = gammaV.data();
                const float       *beta   = betaV.data();
                const float       *mean   = meanV.data();
                const float       *var    = varV.data();
                std::string        tag    = node.name;
                // Fold BatchNorm's 4 params into a per-channel affine y = scale[c]*x + bias[c]:
                //   scale = gamma / sqrt(var + eps),  bias = beta - mean*scale.
                // Precomputed on the host and uploaded once so the shader is a plain multiply-add.
                scaleBuf = uploadCached(env, tag + "#bn_scale", [&] {
                    std::vector<float> a(padded, 0.f);
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        a[c] = gamma[c] / std::sqrt(var[c] + eps);
                    }
                    return a;
                });
                biasBuf  = uploadCached(env, tag + "#bn_bias", [&] {
                    std::vector<float> b(padded, 0.f);
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        b[c] = beta[c] - mean[c] * (gamma[c] / std::sqrt(var[c] + eps));
                    }
                    return b;
                });
                pc       = {(int) (x.n * Cb * 4 * x.h * x.w), (int) Cb, (int) (x.h * x.w)};
                pipe     = env.pipeline(shader("batchnorm", env.useFp16), 4, sizeof(BnPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (runtime)
                {
                    pipe->dispatch(cmd,
                                   {env.devBuf(node.inputs[0])->handle(), env.devBuf(node.inputs[1])->handle(), env.devBuf(node.inputs[2])->handle(), env.devBuf(node.inputs[3])->handle(),
                                    env.devBuf(node.inputs[4])->handle(), env.devBuf(node.outputs[0])->handle()},
                                   &rtpc, sizeof(rtpc), groups(rtpc.count, kBnLocalSize));
                    return;
                }
                pipe->dispatch(cmd, {env.devBuf(node.inputs[0])->handle(), scaleBuf->handle(), biasBuf->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.count, kBnLocalSize));
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::BatchNorm, BatchNormOp);

} // namespace vknn
