// GlobalAveragePool: average each channel over H*W. One workgroup per (n, channel-block); its
// threads cooperatively reduce the spatial dimension (see shaders/avgpool.comp).
#include "vk_op_common.h"

namespace vknn {
namespace {

struct GlobalAvgPoolOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  PoolPC pc{};
  int64_t total = 0;

  void prepare(const Node& node, VkOpEnv& env) override {
    NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
    pc = {(int)x.n, (int)x.c, (int)x.h, (int)x.w};
    total = x.n * cBlocks(x.c);
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("avgpool", env.useFp16), 2,
                                                 sizeof(PoolPC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = env.devBuf(node.inputs[0]);
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    // one workgroup per (n, channel-block); the workgroup's 256 threads reduce H*W together
    pipe->dispatch(cmd, {src->handle(), dst->handle()}, &pc, sizeof(pc), (uint32_t)total);
  }
};

}  // namespace

VKNN_REGISTER_VK_OP(OpType::kGlobalAvgPool, GlobalAvgPoolOp);

}  // namespace vknn
