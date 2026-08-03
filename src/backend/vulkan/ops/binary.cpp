// Elementwise binary family on the GPU (Mul/Sub/Div/Max/Min/Pow). Handles same-shape inputs and
// the channel-broadcast case (second operand [N,C,1,1], the Squeeze-Excite scale). Other broadcast
// patterns fall back to the CPU binary op (gated in the backend's supportsNode()).
//
// Either operand may be a CONSTANT. A constant is not an activation, so it has no pooled buffer to
// bind; it is packed into the blocked layout once at prepare and uploaded through the weight cache.
// Without that the whole node had to run flat, which cost a layout convert on the way in and another
// on the way out for the sake of one uploaded tensor.
#include "core/boundary_pack.h"
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        // Local workgroup size along x for the NC4HW4 path; matches local_size_x in shaders/binary.comp.

        struct BinaryPC {
            int count, HW, op;
        };

        struct BinaryOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            BinaryPC                             pc {};
            flat::Binary                         flatImpl;
            bool                                 flat = false;
            // Blocked uploads of constant operands, by input index; null where the operand is an
            // activation and binds from the pool.
            std::shared_ptr<vk::Buffer> constBuf[2];

            void prepare(const Node &node, VkOpEnv &env) override {
                if (opIsFlat(node, env))
                {
                    flat = true;
                    flatImpl.prepare(node, env);
                    return;
                }
                NCHW y = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                NCHW a = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                NCHW b = NCHW::from(env.graph->desc(node.inputs[1]).shape);
                // Spatial extent per channel block. The shader divides the linear vec4 index by HW to
                // collapse a broadcast [N,C,1,1] operand to its matching per-channel-block element.
                int HW = (int) (y.h * y.w);
                // 0 = same shape, 1 = A is the [N,C,1,1] broadcast operand, 2 = B is.
                uint32_t bcast = 0;
                if (y.h * y.w != 1)
                {
                    if (a.h * a.w == 1)
                    {
                        bcast = 1;
                    } else if (b.h * b.w == 1)
                    { bcast = 2; }
                }
                // One GPU thread per NC4HW4 vec4 slot: cBlocks(y.c) packs four channels into each vec4,
                // so the thread count is y.n * ceil(C/4) * HW. The int64 product guards the multiply from
                // overflow before it is truncated to the shader's int `count`.
                pc   = {(int) ((int64_t) y.n * cBlocks(y.c) * HW), HW, node.subOp};
                pipe = env.pipeline(shader("binary", env.useFp16), 3, sizeof(BinaryPC), std::vector<uint32_t> {bcast, env.flatLocalSize});
                // A constant operand carries its own NCHW: the same-shape case packs the run shape,
                // the [N,C,1,1] case packs one element per channel. packNc4 pads the channel axis to
                // whole blocks exactly as the activation buffers are packed, so the shader indexes
                // both operands identically.
                const Graph &g = *env.graph;
                for (int i = 0; i < 2; ++i)
                {
                    const TensorId t = node.inputs[(size_t) i];
                    if (t == kNoTensor || !g.isInitializer(t))
                    {
                        continue;
                    }
                    const NCHW   shape = NCHW::from(g.desc(t).shape);
                    const bool   fp16  = env.useFp16;
                    const size_t elems = (size_t) formatElems(TensorFormat::NC4HW4, shape);
                    constBuf[i]        = uploadCached(env, "binaryNc4/" + g.desc(t).name, [&] {
                        const std::vector<float> src = initFloats(g, t);
                        std::vector<float>       packed(elems, 0.f);
                        // packNc4 writes at the storage width; pack fp32 here and let uploadCached's
                        // fp16 conversion happen on the way out, so one host path serves both.
                        boundary::packNc4(src.data(), packed.data(), shape, /*fp16=*/false, 1);
                        return packed;
                    });
                }
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (flat)
                {
                    flatImpl.record(cmd, node, env);
                    return;
                }
                vk::Buffer *a = constBuf[0] ? constBuf[0].get() : env.devBuf(node.inputs[0]);
                vk::Buffer *b = constBuf[1] ? constBuf[1].get() : env.devBuf(node.inputs[1]);
                vk::Buffer *c = env.devBuf(node.outputs[0]);
                // groups() rounds pc.count up to whole env.flatLocalSize workgroups.
                pipe->dispatch(cmd, {a->handle(), b->handle(), c->handle()}, &pc, sizeof(pc), groups(pc.count, env.flatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Binary, BinaryOp);
} // namespace vknn
