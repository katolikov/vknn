// Distribution-focal decode on the GPU (core/dfl.h): one dispatch over the blocked [N, S*B, L]
// map, one thread per stored output lane (image, side, anchor), reading the B bins of its side
// from their channel blocks at the same anchor. Replaces the head's Reshape, two Transposes,
// Softmax and bin Conv (four flat dispatches and their layout converts) and keeps the softmax
// and its expectation in fp32.
#include "core/dfl.h"
#include "dispatch_extent.h" // flat::dispatchExtentRefusal
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // Field order/types mirror dfl.comp's push_constant block: the blocked geometry of the
        // [N, S*B, L] input (CbIn channel blocks) and the [N, S, L] output (CbOut blocks).
        struct DflPC {
            int      N, S, B, L, CbIn, CbOut;
            uint32_t total; // N*CbOut*4*L stored lanes (checked against the int32 domain; the shader compares the unsigned gid to it)
        };
        struct FusedDflOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          weights;
            DflPC                                pc {};
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Graph  &g    = *env.graph;
                const Shape  &x    = g.desc(node.inputs[0]).shape; // [N, S*B, L]
                const int64_t bins = node.attr.geti("bins", 0);
                if (x.size() != 3 || bins < kDflMinBins || x[1] % bins != 0)
                {
                    throw Error(Status::Unsupported, "FusedDfl '" + node.name + "': input is not a [N, sides*bins, anchors] map with bins " + std::to_string(bins));
                }
                const int64_t sides = x[1] / bins;
                const int64_t cbIn = cBlocks(x[1]), cbOut = cBlocks(sides);
                const int64_t total = x[0] * cbOut * kNC4Block * x[2];
                // The kernel indexes both maps in signed int (the flat family's domain): the output
                // lane count is the dispatch extent, the input lane count its largest index.
                const int64_t inputLanes = x[0] * cbIn * kNC4Block * x[2];
                if (!flat::dispatchExtentFits(total))
                {
                    throw Error(Status::Unsupported, flat::dispatchExtentRefusal("FusedDfl '" + node.name + "'", "output lane count", total));
                }
                if (!flat::dispatchExtentFits(inputLanes))
                {
                    throw Error(Status::Unsupported, flat::dispatchExtentRefusal("FusedDfl '" + node.name + "'", "input lane count", inputLanes));
                }
                pc = {(int) x[0], (int) sides, (int) bins, (int) x[2], (int) cbIn, (int) cbOut, (uint32_t) total};
                std::vector<float> w = initFloats(g, node.inputs[1]);
                w.resize((size_t) bins, 0.f);
                weights = upload(*env.ctx, w, env.useFp16);
                pipe    = env.pipeline(shader("dfl", env.useFp16), 3, sizeof(DflPC), std::vector<uint32_t> {flat::flatLocalSizeFor(env.ctx->caps())});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                std::vector<VkBuffer> bufs = {env.devBuf(node.inputs[0])->handle(), weights->handle(), env.devBuf(node.outputs[0])->handle()};
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups((int64_t) pc.total, flat::flatLocalSizeFor(env.ctx->caps())));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::FusedDfl, FusedDflOp);
} // namespace vknn
