// Flat row-major GPU Gather (one op per file). Honors `axis` and a scalar-or-N-D index. The index
// is passed to the kernel as a float buffer so ONE kernel serves both cases: a constant index
// (attention Q/K/V split, uploaded here as float) and a runtime float index activation (RoPE
// position lookup, read straight from its device buffer). Mirrors GatherCpu's shape/scatter math.
#include "vk_op_common.h"
#include "vknn/op.h"
#include <algorithm>
#include <vector>

namespace vknn {
    namespace {

        // Local workgroup size along x; matches local_size_x in shaders/gather.comp.
        constexpr uint32_t kGatherLocalSize = 256;

        struct GatherOp: VulkanOp {
            // Field order/types mirror gather.comp's push_constant block. The data tensor is
            // flattened around `axis` into [outer, axisSize, inner]: outer = product of dims before
            // axis, axisSize = the gathered dim, inner = product of dims after axis. The kernel maps
            // each of `total` output elements to (o, i, in) and reads data[o*axisSize*inner +
            // idx[i]*inner + in], so this decomposition is all it needs to index either operand.
            struct PC {
                int total, outer, axisSize, inner, nIdx;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          idxBuf; // const index uploaded as float; null when index is activation
            std::shared_ptr<vk::Buffer>          hold0;  // const data operand

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g    = *env.graph;
                Shape        d    = g.desc(node.inputs[0]).shape;
                Shape        out  = g.desc(node.outputs[0]).shape;
                int          rank = (int) d.size();
                int          axis = (int) node.attr.geti("axis", 0);
                // Normalize the ONNX axis: negative counts from the back, then clamp into
                // [0, rank-1] so a malformed/out-of-range attribute can never index outside `d`.
                if (axis < 0)
                {
                    axis += rank;
                }
                if (axis < 0)
                {
                    axis = 0;
                }
                if (axis >= rank)
                {
                    axis = rank > 0 ? rank - 1 : 0;
                }
                int64_t axisSize = rank > 0 ? d[axis] : 1;
                int64_t outer    = 1;
                for (int k = 0; k < axis; ++k)
                {
                    outer *= d[k];
                }
                int64_t inner = 1;
                for (int k = axis + 1; k < rank; ++k)
                {
                    inner *= d[k];
                }

                TensorId iid = node.inputs[1];
                // A scalar index has a rank-0 shape (numElements == 1); the max() keeps nIdx >= 1
                // so the empty-shape scalar case still allocates and dispatches one index lane.
                int64_t nIdx = std::max<int64_t>(numElements(g.desc(iid).shape), 1);
                if (g.isInitializer(iid))
                { // const index -> upload as float (decode int64/fp16 as needed)
                    const HostBuffer  &hb  = g.initializers.at(iid);
                    DType              idt = g.desc(iid).dtype;
                    std::vector<float> iv((size_t) nIdx);
                    for (int64_t i = 0; i < nIdx; ++i)
                    {
                        if (idt == DType::Int64)
                        {
                            iv[(size_t) i] = (float) hb.i64()[i];
                        } else if (idt == DType::Float16)
                        {
                            iv[(size_t) i] = halfToFloatAt(hb.bytes.data(), i);
                        } else
                        {
                            iv[(size_t) i] = hb.f32()[i];
                        }
                        // A constant index is fully known here: an out-of-range one is a hard error at
                        // prepare time, matching GatherCpu's contract. The kernel's clamp only covers a
                        // runtime index activation, which the host cannot see before dispatch. Guarded
                        // on a statically known axis size (> 0); an unresolved dim has nothing to check.
                        const int64_t rawIndex = idt == DType::Int64 ? hb.i64()[i] : (int64_t) iv[(size_t) i];
                        if (axisSize > 0 && (rawIndex < -axisSize || rawIndex >= axisSize))
                        {
                            throw Error(Status::InvalidArgument, "Gather '" + node.name + "': constant index " + std::to_string(rawIndex) + " at position " + std::to_string(i) + " is out of range [" + std::to_string(-axisSize) + ", " + std::to_string(axisSize) + ") for axis " + std::to_string(axis) + " of size " + std::to_string(axisSize));
                        }
                    }
                    idxBuf = upload(*env.ctx, iv, false); // index is always fp32 (gather.comp binding 1)
                }

                pc   = {(int) numElements(out), (int) outer, (int) axisSize, (int) inner, (int) nIdx};
                pipe = env.pipeline(shader("gather", env.useFp16), 3, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // Const index uses the float buffer uploaded in prepare(); a runtime index reads its
                // device buffer directly. Either way the kernel treats index[] as float (see header).
                vk::Buffer *idx = idxBuf ? idxBuf.get() : env.devBuf(node.inputs[1]);
                // groups() rounds pc.total (one thread per output element) up to whole kGatherLocalSize
                // workgroups. Buffer order matches gather.comp: data, index, out.
                pipe->dispatch(cmd, {operandBuf(env, node.inputs[0], hold0)->handle(), idx->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, kGatherLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Gather, GatherOp);
} // namespace vknn
