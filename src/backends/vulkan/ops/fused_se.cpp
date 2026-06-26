// Fused Squeeze-Excite scale on the GPU. One dispatch (one workgroup per image) replaces the
// GAP->FC->relu->FC->hardsigmoid chain. Weights uploaded once (fp16). The following Mul is
// unchanged.
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
namespace {
struct SePC {
  int N, C, Cr;
  float alpha, beta;
};
struct FusedSeOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer> w1, b1, w2, b2;
  SePC pc{};
  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    NCHW x = NCHW::from(g.desc(node.inputs[0]).shape);
    int64_t C = x.c, Cr = g.desc(node.inputs[3]).shape[0];  // W2 is [C][Cr] -> rows=C; use W1 rows
    Cr = g.desc(node.inputs[1]).shape[0];                   // W1 is [Cr][C] -> Cr = rows of W1
    pc = {(int)x.n, (int)C, (int)Cr, node.actLo, node.actHi};
    w1 = uploadCached(env, node.name + "#w1", [&] { return initFloats(g, node.inputs[1]); });
    w2 = uploadCached(env, node.name + "#w2", [&] { return initFloats(g, node.inputs[3]); });
    b1 = uploadCached(env, node.name + "#b1", [&] {
      std::vector<float> v(Cr, 0.f);
      if (node.inputs[2] != kNoTensor) {
        std::vector<float> t = initFloats(g, node.inputs[2]);
        for (int64_t i = 0; i < Cr && i < (int64_t)t.size(); ++i)
          v[i] = t[i];
      }
      return v;
    });
    b2 = uploadCached(env, node.name + "#b2", [&] {
      std::vector<float> v(C, 0.f);
      if (node.inputs[4] != kNoTensor) {
        std::vector<float> t = initFloats(g, node.inputs[4]);
        for (int64_t i = 0; i < C && i < (int64_t)t.size(); ++i)
          v[i] = t[i];
      }
      return v;
    });
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("fused_se", env.useFp16), 6,
                                                 sizeof(SePC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* f = env.devBuf(node.inputs[0]);
    vk::Buffer* s = env.devBuf(node.outputs[0]);
    pipe->dispatch(
        cmd, {f->handle(), w1->handle(), b1->handle(), w2->handle(), b2->handle(), s->handle()},
        &pc, sizeof(pc), (uint32_t)pc.N);
  }
};
}  // namespace
VKNN_REGISTER_VK_OP(OpType::kFusedSE, FusedSeOp);
}  // namespace vknn
