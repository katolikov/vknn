// DepthToSpace on the GPU: [N,C,H,W] -> [N,C/b^2,H*b,W*b]. Two paths, chosen by depthToSpaceIsNc4.
//
// Packed (both channel counts 4-aligned): one thread per output NC4HW4 block-pixel assembles the
// block's four source channels and stores one quad. Nothing converts around it.
//
// Flat row-major (anything else): one thread per output element does the DCR/CRD index remap, and
// the layout pass converts at each NC4HW4 boundary.
//
// The two kernels take different push-constant blocks, so the pipeline and the pushed block are one
// choice carried on the op instance as a DepthToSpaceDispatchPlan (depth_to_space_plan.h).
#include "depth_to_space_plan.h"
#include "dispatch_extent.h"
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/error.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct DepthToSpaceOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            DepthToSpaceDispatchPlan             plan;
            DepthToSpaceFlatPC                   flatPc {};
            DepthToSpacePackedPC                 packedPc {};
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                int  b = (int) node.attr.geti("blocksize", 1);
                if (b < 1)
                {
                    b = 1;
                }
                // Output channels shrink by b^2 while each spatial dim grows by b, so total element
                // count is preserved; the shader reads mode to pick the DCR vs CRD channel unpacking.
                int C2 = (int) x.c / (b * b), OH = (int) x.h * b, OW = (int) x.w * b;
                int mode = node.attr.gets("mode", "DCR") == "CRD" ? 1 : 0;
                // Lane counts stay int64 until the extent gate has cleared them: one lane per output
                // element for the flat kernel, one per output NC4HW4 block-pixel for the packed one.
                const int64_t flatLanes   = (int64_t) x.n * C2 * OH * OW;
                const int64_t packedLanes = (int64_t) x.n * cBlocks(C2) * OH * OW;
                plan                      = planDepthToSpaceDispatch(depthToSpaceIsNc4(*env.graph, node), packedLanes, flatLanes);
                // Both counts are narrowed into an int push-constant field, so both are checked; the
                // packed count is the flat count over the NC4HW4 block width, so the flat one binds.
                if (!flat::dispatchExtentFits(flatLanes) || !flat::dispatchExtentFits(packedLanes))
                {
                    throw Error(Status::Unsupported, flat::dispatchExtentRefusal("DepthToSpace '" + node.name + "'", "output element count", flatLanes));
                }
                flatPc   = {(int) flatLanes, (int) x.n, (int) x.c, (int) x.h, (int) x.w, C2, OH, OW, b, mode};
                packedPc = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, C2, OH, OW, b, mode};
                if (plan.path == DepthToSpacePath::kPackedNc4)
                {
                    pipe = env.pipeline(shader("depth_to_space_nc4", env.useFp16), 2, plan.pushConstantBytes(), std::vector<uint32_t> {env.convLocalSize});
                    return;
                }
                pipe = env.pipeline(shader("flat_depth_to_space", env.useFp16), 2, plan.pushConstantBytes(), std::vector<uint32_t> {env.flatLocalSize});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                std::vector<VkBuffer> bufs = {env.devBuf(node.inputs[0])->handle(), env.devBuf(node.outputs[0])->handle()};
                // The path, the pushed block and its size all come from the plan prepare() built, so a
                // zero-lane dispatch still pushes the block the pipeline's range was created for.
                if (plan.path == DepthToSpacePath::kPackedNc4)
                {
                    pipe->dispatch(cmd, bufs, &packedPc, plan.pushConstantBytes(), groups(plan.laneCount, env.convLocalSize));
                    return;
                }
                // One thread per output element; flat_depth_to_space.comp takes the family width as its
                // workgroup-size spec constant (env.flatLocalSize), and groups() rounds the lane count
                // up to whole workgroups.
                pipe->dispatch(cmd, bufs, &flatPc, plan.pushConstantBytes(), groups(plan.laneCount, env.flatLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::DepthToSpace, DepthToSpaceOp);
} // namespace vknn
