// Elementwise Clip on the GPU. Bounds (min=input[1], max=input[2]) are constant scalars read in
// prepare() and baked into the push constant; an absent bound is -/+inf.
//
// A clamp reads element i and writes element i, so it computes the same answer whatever layout its
// tensor carries -- the layout pass therefore lets it ADOPT its input's, and a Clip between blocked
// neighbours needs no converts. What it must respect is the STORED element count: a blocked buffer
// pads its channel axis, and walking the logical count would leave those lanes undefined.
//
// When a bound is a RUNTIME tensor (computed min/max, not an initializer) the value isn't known in
// prepare(), so the op switches to clip_rt.comp: the bounds bind as single-element SSBOs read at
// dispatch. An absent or constant bound in that path is materialised as a one-element buffer holding
// the -/+inf or the fixed value, so the shader always reads binding 1/2. The all-constant/absent case
// keeps the baked push-constant kernel (clip.comp) byte-for-byte unchanged.
#include "blocked_extent.h"
#include "vk_op_common.h"
#include "vknn/op.h"
#include <limits>

namespace vknn {
    namespace {

        // Local workgroup size along x; matches local_size_x in shaders/clip.comp and clip_rt.comp.
        constexpr uint32_t kClipLocalSize = 256;

        // True iff input `idx` is a runtime (non-initializer) tensor that must be read at dispatch.
        inline bool runtimeBound(const Graph &g, const Node &node, int idx) {
            return (int) node.inputs.size() > idx && node.inputs[idx] != kNoTensor && !g.isInitializer(node.inputs[idx]);
        }

        struct ClipOp: VulkanOp {
            // Field order/types mirror clip.comp's push_constant block { int total; float lo, hi; }.
            // total is the flat element count of the output (one lane per element on the row-major path).
            struct PC {
                int   total;
                float lo, hi;
            } pc {};
            struct RtPC {
                int total; // clip_rt.comp reads lo/hi from SSBOs, so only the element count is a constant
            } rtpc {};
            bool                                 runtime = false; // a runtime min or max => bind bounds as SSBOs
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          hold0;
            std::shared_ptr<vk::Buffer>          loBuf, hiBuf; // runtime-path bound buffers (const/absent bounds too)

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                pc.lo          = -std::numeric_limits<float>::infinity();
                pc.hi          = std::numeric_limits<float>::infinity();
                auto scalar    = [&](int idx, float &dst) {
                    if ((int) node.inputs.size() > idx && node.inputs[idx] != kNoTensor && g.isInitializer(node.inputs[idx]))
                    {
                        std::vector<float> v = initFloats(g, node.inputs[idx]);
                        if (!v.empty())
                        {
                            dst = v[0];
                        }
                    }
                };
                scalar(1, pc.lo); // min
                scalar(2, pc.hi); // max
                // also support the Relu6-style float attributes (older Clip opset)
                if (node.attr.has("min"))
                {
                    pc.lo = node.attr.getf("min", pc.lo);
                }
                if (node.attr.has("max"))
                {
                    pc.hi = node.attr.getf("max", pc.hi);
                }
                pc.total   = (int) storedElemCount(g.desc(node.outputs[0]).shape, g.desc(node.outputs[0]).gpuFlat);
                rtpc.total = pc.total;

                runtime = runtimeBound(g, node, 1) || runtimeBound(g, node, 2);
                if (!runtime)
                {
                    pipe = env.pipeline(shader("clip", env.useFp16), 2, sizeof(PC), std::vector<uint32_t> {});
                    return;
                }
                // Runtime path: a runtime bound binds directly at dispatch (record()); a constant or
                // absent bound is uploaded once as a single-element buffer holding pc.lo / pc.hi so the
                // shader unconditionally reads binding 1/2.
                if (!runtimeBound(g, node, 1))
                {
                    loBuf = upload(*env.ctx, {pc.lo}, env.useFp16);
                }
                if (!runtimeBound(g, node, 2))
                {
                    hiBuf = upload(*env.ctx, {pc.hi}, env.useFp16);
                }
                pipe = env.pipeline(shader("clip_rt", env.useFp16), 4, sizeof(RtPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // operandBuf (not devBuf) so a constant-initializer input[0] uploads flat into hold0 on
                // first use instead of null-crashing. One flat 1D grid of kClipLocalSize-lane workgroups spans total.
                VkBuffer src = operandBuf(env, node.inputs[0], hold0)->handle();
                if (!runtime)
                {
                    pipe->dispatch(cmd, {src, env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, kClipLocalSize));
                    return;
                }
                VkBuffer lo = loBuf ? loBuf->handle() : env.devBuf(node.inputs[1])->handle();
                VkBuffer hi = hiBuf ? hiBuf->handle() : env.devBuf(node.inputs[2])->handle();
                pipe->dispatch(cmd, {src, lo, hi, env.devBuf(node.outputs[0])->handle()}, &rtpc, sizeof(rtpc), groups(rtpc.total, kClipLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Clip, ClipOp);
} // namespace vknn
