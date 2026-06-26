// Residual add over NC4HW4 buffers. Both inputs have the same packed layout, so it's a
// flat elementwise add over the buffer.
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
namespace {

struct AddPC {
  uint32_t count;
  int act;
  float actLo, actHi;
};

struct AddOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  AddPC pc{};
  flat::Binary flatImpl;
  bool flat = false;

  void prepare(const Node& node, VkOpEnv& env) override {
    if (opIsFlat(node, env)) {
      flat = true;
      flatImpl.prepare(node, env);
      return;
    }
    pc = {(uint32_t)packedElems(env.graph->desc(node.outputs[0]).shape), (int)node.fusedAct,
          node.actLo, node.actHi};
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("add", env.useFp16), 3,
                                                 sizeof(AddPC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    if (flat) {
      flatImpl.record(cmd, node, env);
      return;
    }
    vk::Buffer* a = env.devBuf(node.inputs[0]);
    vk::Buffer* b = env.devBuf(node.inputs[1]);
    vk::Buffer* y = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {a->handle(), b->handle(), y->handle()}, &pc, sizeof(pc),
                   groups(pc.count, 256));
  }
};

}  // namespace

VKNN_REGISTER_VK_OP(OpType::kAdd, AddOp);

}  // namespace vknn
