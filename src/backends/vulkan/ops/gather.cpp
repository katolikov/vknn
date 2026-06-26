// Flat row-major GPU Gather (one op per file). Honors `axis` and a scalar-or-N-D index. The index
// is passed to the kernel as a float buffer so ONE kernel serves both cases: a constant index
// (attention Q/K/V split, uploaded here as float) and a runtime float index activation (RoPE
// position lookup, read straight from its device buffer). Mirrors GatherCpu's shape/scatter math.
#include <algorithm>
#include <vector>

#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
namespace {

struct GatherOp : VulkanOp {
  struct PC {
    int total, outer, axisSize, inner, nIdx;
  } pc{};
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer>
      idxBuf;  // const index uploaded as float; null when index is activation
  std::shared_ptr<vk::Buffer> hold0;  // const data operand

  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    Shape d = g.desc(node.inputs[0]).shape;
    Shape out = g.desc(node.outputs[0]).shape;
    int rank = (int)d.size();
    int axis = (int)node.attr.geti("axis", 0);
    if (axis < 0)
      axis += rank;
    if (axis < 0)
      axis = 0;
    if (axis >= rank)
      axis = rank > 0 ? rank - 1 : 0;
    int64_t axisSize = rank > 0 ? d[axis] : 1;
    int64_t outer = 1;
    for (int k = 0; k < axis; ++k)
      outer *= d[k];
    int64_t inner = 1;
    for (int k = axis + 1; k < rank; ++k)
      inner *= d[k];

    TensorId iid = node.inputs[1];
    int64_t nIdx = std::max<int64_t>(numElements(g.desc(iid).shape), 1);
    if (g.isInitializer(iid)) {  // const index -> upload as float (decode int64/fp16 as needed)
      const HostBuffer& hb = g.initializers.at(iid);
      DType idt = g.desc(iid).dtype;
      std::vector<float> iv((size_t)nIdx);
      for (int64_t i = 0; i < nIdx; ++i) {
        if (idt == DType::kInt64)
          iv[(size_t)i] = (float)hb.i64()[i];
        else if (idt == DType::kFloat16)
          iv[(size_t)i] = halfToFloat(reinterpret_cast<const fp16_t*>(hb.bytes.data())[i]);
        else
          iv[(size_t)i] = hb.f32()[i];
      }
      idxBuf = upload(*env.ctx, iv, env.useFp16);
    }

    pc = {(int)numElements(out), (int)outer, (int)axisSize, (int)inner, (int)nIdx};
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("gather", env.useFp16), 3,
                                                 sizeof(PC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* idx = idxBuf ? idxBuf.get() : env.devBuf(node.inputs[1]);
    pipe->dispatch(cmd,
                   {operandBuf(env, node.inputs[0], hold0)->handle(), idx->handle(),
                    env.devBuf(node.outputs[0])->handle()},
                   &pc, sizeof(pc), groups(pc.total, 256));
  }
};

}  // namespace
VKNN_REGISTER_VK_OP(OpType::kGather, GatherOp);
}  // namespace vknn
