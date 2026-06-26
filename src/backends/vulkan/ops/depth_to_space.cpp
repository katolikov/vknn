// DepthToSpace on the GPU (FLAT row-major path): [N,C,H,W] -> [N,C/b^2,H*b,W*b]. One thread per
// output element does the DCR/CRD index remap into the source. The layout pass marks the output
// flat (channel count changes) and inserts ConvertLayout at the NC4HW4 boundary.
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vx/op.h"

namespace vx {
namespace {
struct D2sPC {
  int total, N, C, H, W, C2, OH, OW, b, mode;
};
struct DepthToSpaceOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  D2sPC pc{};
  void prepare(const Node& node, VkOpEnv& env) override {
    NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
    int b = (int)node.attr.geti("blocksize", 1);
    if (b < 1) b = 1;
    int C2 = (int)x.c / (b * b), OH = (int)x.h * b, OW = (int)x.w * b;
    int mode = node.attr.gets("mode", "DCR") == "CRD" ? 1 : 0;
    pc = {(int)((int64_t)x.n * C2 * OH * OW), (int)x.n, (int)x.c, (int)x.h, (int)x.w,
          C2, OH, OW, b, mode};
    pipe = std::make_unique<vk::ComputePipeline>(
        *env.ctx, shader("flat_depth_to_space", env.useFp16), 2, sizeof(D2sPC),
        std::vector<uint32_t>{}, env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    pipe->dispatch(cmd,
                   {env.devBuf(node.inputs[0])->handle(), env.devBuf(node.outputs[0])->handle()},
                   &pc, sizeof(pc), groups(pc.total, 256));
  }
};
}  // namespace
VX_REGISTER_VK_OP(OpType::kDepthToSpace, DepthToSpaceOp);
}  // namespace vx
