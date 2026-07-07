// Channel-axis Concat on the GPU (NC4HW4). Each input is copied into the output at its channel-
// block offset. Only valid when every input's channel count is a multiple of 4 (block alignment);
// the backend's supportsNode() enforces that, otherwise it falls back to the CPU concat. A fused
// pointwise unit (pw_steps) applies at each part's stores, indexed in output space, so a unit
// attached to the Concat (e.g. a lowered BatchNorm + activation) costs no extra dispatch. Inputs
// from pwCoreInputs on are the unit's operands, not concatenated parts.
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        // Local workgroup size along x for the NC4HW4 path; matches local_size_x in shaders/concat.comp.
        constexpr uint32_t kConcatLocalSize = 64;

        // Mirror of the concat shader's push_constant block. Cib/Cob are input/output channel-block
        // counts (channels/4 in NC4HW4), cbOff is this input's starting channel block in the output,
        // and HW is the flattened spatial extent. One invocation copies one [n][cb][hw] vec4 element.
        struct ConcatPC {
            int N, Cib, Cob, cbOff, HW;
        };

        struct ConcatOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::vector<ConcatPC>                parts; // one per concatenated input
            std::vector<int64_t>                 partGroups;
            PwEpi                                epi;
            flat::Concat                         flatImpl;
            bool                                 flat = false;

            void prepare(const Node &node, VkOpEnv &env) override {
                if (opIsFlat(node, env))
                {
                    flat = true;
                    flatImpl.prepare(node, env);
                    return;
                }
                Shape out = env.graph->desc(node.outputs[0]).shape;
                epi.prepare(node, env, false, out);
                NCHW   y     = NCHW::from(out);
                int    Cob   = (int) cBlocks(y.c), HW = (int) (y.h * y.w);
                int    cbOff = 0;
                size_t nIn   = (size_t) pwCoreInputs(node);
                for (size_t e = 0; e < nIn && e < node.inputs.size(); ++e)
                {
                    NCHW xi  = NCHW::from(env.graph->desc(node.inputs[e]).shape);
                    int  Cib = (int) cBlocks(xi.c);
                    parts.push_back({(int) y.n, Cib, Cob, cbOff, HW});
                    // One invocation per [n][cb][hw] vec4; kConcatLocalSize matches the shader's local_size_x.
                    partGroups.push_back(groups((int64_t) y.n * Cib * HW, kConcatLocalSize));
                    // Advance the output channel-block cursor so the next input lands after this one.
                    cbOff += Cib;
                }
                pipe = env.pipeline(shader((std::string("concat") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(ConcatPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (flat)
                {
                    flatImpl.record(cmd, node, env);
                    return;
                }
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                // Each input writes a disjoint channel-block range of the output, so no barriers between them.
                for (size_t i = 0; i < parts.size(); ++i)
                {
                    vk::Buffer           *src = env.devBuf(node.inputs[i]);
                    std::vector<VkBuffer> bufs {src->handle(), dst->handle()};
                    epi.append(bufs, node, env, dst->handle());
                    pipe->dispatch(cmd, bufs, &parts[i], sizeof(ConcatPC), (uint32_t) partGroups[i]);
                }
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Concat, ConcatOp);
} // namespace vknn
