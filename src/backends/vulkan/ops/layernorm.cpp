// LayerNormalization on the GPU via the FLAT (row-major) path. One thread per OUTER row (outer =
// product of dims before `axis`); each walks its normalized block (norm = product of dims from `axis`
// to the end) computing mean then variance, then writes (x-mean)/sqrt(var+eps)*gamma + beta.
// gamma (Scale) and beta (B, optional) are 1-D [norm] initializers, uploaded as flat constant buffers
// in prepare(). Output shape == input shape. supportsNode gates to scale/bias-as-initializers.
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vx/op.h"

namespace vx {
namespace {

struct LnPC {
  int outer, norm, hasBeta;
  float eps;
};

struct LayerNormOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  LnPC pc{};
  std::shared_ptr<vk::Buffer> gammaBuf, betaBuf;

  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    Shape s = g.desc(node.inputs[0]).shape;
    int rank = (int)s.size();
    int64_t axis = node.attr.geti("axis", -1);
    if (axis < 0) axis += rank;
    if (axis < 0) axis = 0;
    int64_t norm = 1;
    for (int k = (int)axis; k < rank; ++k) norm *= s[k];
    if (norm < 1) norm = 1;
    int64_t outer = numElements(s) / norm;
    bool hasBeta = node.inputs.size() > 2 && node.inputs[2] != kNoTensor &&
                   g.isInitializer(node.inputs[2]);
    pc = {(int)outer, (int)norm, hasBeta ? 1 : 0, node.attr.getf("epsilon", 1e-5f)};

    // gamma (Scale) is a 1-D [norm] initializer; upload it flat.
    std::vector<float> gv = initFloats(g, node.inputs[1]);
    gv.resize(norm);
    gammaBuf = upload(*env.ctx, gv, env.useFp16);
    // beta (optional): real bias, or a zero buffer so binding 2 is always valid.
    if (hasBeta) {
      std::vector<float> bv = initFloats(g, node.inputs[2]);
      bv.resize(norm);
      betaBuf = upload(*env.ctx, bv, env.useFp16);
    } else {
      betaBuf = upload(*env.ctx, std::vector<float>((size_t)norm, 0.0f), env.useFp16);
    }
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("flat_layernorm", env.useFp16), 4,
                                                 sizeof(LnPC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    pipe->dispatch(cmd,
                   {env.devBuf(node.inputs[0])->handle(), gammaBuf->handle(), betaBuf->handle(),
                    env.devBuf(node.outputs[0])->handle()},
                   &pc, sizeof(pc), groups(pc.outer, 256));
  }
};

}  // namespace
VX_REGISTER_VK_OP(OpType::kLayerNorm, LayerNormOp);
}  // namespace vx
