// Flat elementwise Clip on the GPU. Bounds (min=input[1], max=input[2]) are constant scalars read in
// prepare() and baked into the push constant; an absent bound is -/+inf. Runs on the flat row-major
// path (the geometry-tail Clip is a rank-6 tensor that can't be NC4HW4-packed).
#include <limits>
#include "vk_op_common.h"
#include "vx/op.h"

namespace vx {
namespace {

struct ClipOp : VulkanOp {
  struct PC { int total; float lo, hi; } pc{};
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer> hold0;

  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    pc.lo = -std::numeric_limits<float>::infinity();
    pc.hi = std::numeric_limits<float>::infinity();
    auto scalar = [&](int idx, float& dst) {
      if ((int)node.inputs.size() > idx && node.inputs[idx] != kNoTensor &&
          g.isInitializer(node.inputs[idx])) {
        std::vector<float> v = initFloats(g, node.inputs[idx]);
        if (!v.empty()) dst = v[0];
      }
    };
    scalar(1, pc.lo);  // min
    scalar(2, pc.hi);  // max
    // also support the Relu6-style float attributes (older Clip opset)
    if (node.attr.has("min")) pc.lo = node.attr.getf("min", pc.lo);
    if (node.attr.has("max")) pc.hi = node.attr.getf("max", pc.hi);
    pc.total = (int)numElements(g.desc(node.outputs[0]).shape);
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("clip", env.useFp16), 2,
                                                 sizeof(PC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    pipe->dispatch(cmd,
                   {operandBuf(env, node.inputs[0], hold0)->handle(),
                    env.devBuf(node.outputs[0])->handle()},
                   &pc, sizeof(pc), groups(pc.total, 256));
  }
};

}  // namespace
VX_REGISTER_VK_OP(OpType::kClip, ClipOp);
}  // namespace vx
