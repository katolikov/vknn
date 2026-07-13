// Rope (fused rotate-half rotary embedding) on the GPU via the FLAT (row-major) path. ONE dispatch
// applies the whole rotate-half chain a lowered contrib RotaryEmbedding expands to (last-axis half
// Slices, cos/sin table Gathers by position, the two rotate products, Concat): each thread computes
// one output element as x1*cos - x2*sin (low half) or x1*sin + x2*cos (high half), reading the
// cos/sin row straight from the table at the row's position id. Positions are ALWAYS fp32 (the
// gather.comp index convention: an integer position must never round through fp16 storage —
// pinGatherIndexFp32 pins the runtime chain, a constant uploads as fp32 here). The tables bind at
// compute precision: constants upload flat through the content-pooled uploadInit (48 rope sites of
// one decoder share ONE device copy of each table); a runtime (computed) table binds its activation
// buffer. All arithmetic is fp32; the store rounds once through TO_STORE. Output shape == input
// shape. Created only by fuseRope at session load.
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        // Local workgroup size along x; matches local_size_x in shaders/rope.comp.
        constexpr uint32_t kRopeLocalSize = 256;

        struct RopeOp: VulkanOp {
            // Field order/types mirror rope.comp's push_constant block. total = output elements;
            // head/halfDim = the per-head width and its half; headsPerPos = the H axis extent
            // (consecutive head rows sharing one position); tableRows = the table height, for the
            // negative-position wrap.
            struct PC {
                int total, head, halfDim, headsPerPos, tableRows;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          posBuf;   // const positions uploaded as fp32; null when runtime
            std::shared_ptr<vk::Buffer>          holdX;    // const data operand
            std::shared_ptr<vk::Buffer>          holdCos;  // const cos table
            std::shared_ptr<vk::Buffer>          holdSin;  // const sin table

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph  &g       = *env.graph;
                const Shape  &s       = g.desc(node.inputs[0]).shape;
                const int     rank    = (int) s.size();
                const int64_t halfDim = node.attr.geti("half", 0);
                const Shape  &ts      = g.desc(node.inputs[2]).shape;

                TensorId pid = node.inputs[1];
                if (g.isInitializer(pid))
                { // const positions -> upload as fp32 (decode int64/fp16 as needed), like GatherOp
                    const HostBuffer  &hb  = g.initializers.at(pid);
                    DType              pdt = g.desc(pid).dtype;
                    int64_t            n   = std::max<int64_t>(numElements(g.desc(pid).shape), 1);
                    std::vector<float> pv((size_t) n);
                    for (int64_t i = 0; i < n; ++i)
                    {
                        if (pdt == DType::Int64)
                        {
                            pv[(size_t) i] = (float) hb.i64()[i];
                        } else if (pdt == DType::Float16)
                        {
                            pv[(size_t) i] = halfToFloatAt(hb.bytes.data(), i);
                        } else
                        {
                            pv[(size_t) i] = hb.f32()[i];
                        }
                    }
                    posBuf = upload(*env.ctx, pv, false); // positions are always fp32 (rope.comp binding 1)
                }

                pc = {(int) numElements(g.desc(node.outputs[0]).shape), (int) (halfDim * 2), (int) halfDim,
                      (int) (rank >= 2 ? s[rank - 2] : 1), (int) (ts.empty() ? 0 : ts[0])};
                pipe = env.pipeline(shader("rope", env.useFp16), 5, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // Bind order must match the shader: x, positions, cos table, sin table, out.
                vk::Buffer *pos = posBuf ? posBuf.get() : env.devBuf(node.inputs[1]);
                pipe->dispatch(cmd,
                               {operandBuf(env, node.inputs[0], holdX)->handle(), pos->handle(),
                                operandBuf(env, node.inputs[2], holdCos)->handle(),
                                operandBuf(env, node.inputs[3], holdSin)->handle(),
                                env.devBuf(node.outputs[0])->handle()},
                               &pc, sizeof(pc), groups(pc.total, kRopeLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Rope, RopeOp);
} // namespace vknn
