// Elementwise binary family on the GPU (Mul/Sub/Div/Max/Min/Pow). Handles same-shape inputs and
// the channel-broadcast case (second operand [N,C,1,1], the Squeeze-Excite scale). Other broadcast
// patterns fall back to the CPU binary op (gated in the backend's supportsNode()).
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vx {
namespace {

struct BinaryPC {
  int count, HW, op;
};

struct BinaryOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  BinaryPC pc{};
  flat::Binary flatImpl;
  bool flat = false;

  void prepare(const Node& node, VkOpEnv& env) override {
    if (opIsFlat(node, env)) { flat = true; flatImpl.prepare(node, env); return; }
    NCHW y = NCHW::from(env.graph->desc(node.outputs[0]).shape);
    NCHW a = NCHW::from(env.graph->desc(node.inputs[0]).shape);
    NCHW b = NCHW::from(env.graph->desc(node.inputs[1]).shape);
    int HW = (int)(y.h * y.w);
    // 0 = same shape, 1 = A is the [N,C,1,1] broadcast operand, 2 = B is.
    uint32_t bcast = 0;
    if (y.h * y.w != 1) {
      if (a.h * a.w == 1) bcast = 1;
      else if (b.h * b.w == 1) bcast = 2;
    }
    pc = {(int)((int64_t)y.n * cBlocks(y.c) * HW), HW, node.subOp};
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("binary", env.useFp16), 3,
                                                 sizeof(BinaryPC), std::vector<uint32_t>{bcast},
                                                 env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    if (flat) { flatImpl.record(cmd, node, env); return; }
    vk::Buffer* a = env.devBuf(node.inputs[0]);
    vk::Buffer* b = env.devBuf(node.inputs[1]);
    vk::Buffer* c = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {a->handle(), b->handle(), c->handle()}, &pc, sizeof(pc),
                   groups(pc.count, 256));
  }
};

}  // namespace
VX_REGISTER_VK_OP(OpType::kBinary, BinaryOp);
}  // namespace vx
