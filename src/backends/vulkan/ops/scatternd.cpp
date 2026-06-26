// ScatterND on the FLAT row-major GPU path. Two dispatches: (1) copy data -> out, (2) scatter the
// updates into out at each index row. The indices operand is a constant int64 initializer here
// (uploaded as a raw int32 buffer); runtime indices fall back to the CPU op (see supportsNode).
#include <vector>

#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
namespace {

constexpr int kMaxR = 8;  // must match dataDim[8]/stride[8] in shaders/scatternd*.comp
struct CopyPC {
  uint32_t count;
};
struct ScatterPC {
  uint32_t total;
  int q, sliceSize, rank;
  int dataDim[kMaxR];
  int stride[kMaxR];
};

struct ScatterNDOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> copyPipe;
  std::unique_ptr<vk::ComputePipeline> scatterPipe;
  std::shared_ptr<vk::Buffer>
      idxBuf;  // const index uploaded as float; null when index is activation
  std::shared_ptr<vk::Buffer> holdData;
  std::shared_ptr<vk::Buffer> holdUpd;
  CopyPC copyPc{};
  ScatterPC pc{};

  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    const Shape& ds = g.desc(node.inputs[0]).shape;
    const Shape& is = g.desc(node.inputs[1]).shape;
    int dr = (int)ds.size();

    int q = is.empty() ? 1 : (int)is.back();
    int64_t rows = 1;
    for (size_t i = 0; i + 1 < is.size(); ++i)
      rows *= is[i];

    std::vector<int64_t> stride(dr, 1);
    for (int k = dr - 2; k >= 0; --k)
      stride[k] = stride[k + 1] * ds[k + 1];
    int64_t sliceSize = 1;
    for (int k = q; k < dr; ++k)
      sliceSize *= ds[k];

    pc.total = (uint32_t)(rows * sliceSize);
    pc.q = q;
    pc.sliceSize = (int)sliceSize;
    pc.rank = dr;
    for (int k = 0; k < kMaxR; ++k) {
      pc.dataDim[k] = k < dr ? (int)ds[k] : 1;
      pc.stride[k] = k < dr ? (int)stride[k] : 0;
    }
    copyPc.count = (uint32_t)numElements(ds);

    // Index: a constant initializer is uploaded as float; a runtime float index activation (e.g.
    // the decoder/camera ScatterNDs) is read straight from its device buffer. Both feed the
    // kernel's float IDX binding (the kernel truncates to int), so one kernel serves both.
    TensorId iid = node.inputs[1];
    if (g.isInitializer(iid)) {
      const HostBuffer& ib = g.initializers.at(iid);
      DType idt = g.desc(iid).dtype;
      int64_t nIdx = (int64_t)numElements(is);
      std::vector<float> idxf((size_t)std::max<int64_t>(nIdx, 4), 0.f);
      for (int64_t i = 0; i < nIdx; ++i) {
        if (idt == DType::kInt64)
          idxf[(size_t)i] = (float)ib.i64()[i];
        else if (idt == DType::kFloat16)
          idxf[(size_t)i] = halfToFloat(reinterpret_cast<const fp16_t*>(ib.bytes.data())[i]);
        else
          idxf[(size_t)i] = ib.f32()[i];
      }
      idxBuf = upload(*env.ctx, idxf, env.useFp16);
    }

    copyPipe = std::make_unique<vk::ComputePipeline>(
        *env.ctx, shader("scatternd_copy", env.useFp16), 2, sizeof(CopyPC), std::vector<uint32_t>{},
        env.cache->handle());
    scatterPipe = std::make_unique<vk::ComputePipeline>(
        *env.ctx, shader("scatternd", env.useFp16), 3, sizeof(ScatterPC), std::vector<uint32_t>{},
        env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* data = operandBuf(env, node.inputs[0], holdData);
    vk::Buffer* updates = operandBuf(env, node.inputs[2], holdUpd);
    vk::Buffer* idx = idxBuf ? idxBuf.get() : env.devBuf(node.inputs[1]);
    vk::Buffer* out = env.devBuf(node.outputs[0]);
    // Pass 1: out = copy(data).
    copyPipe->dispatch(cmd, {data->handle(), out->handle()}, &copyPc, sizeof(copyPc),
                       groups(copyPc.count, 256));
    // The framework only barriers BETWEEN nodes (read-after-write across ops); two dispatches
    // inside one record() are NOT auto-barriered. Pass 2 scatters into the SAME `out` buffer pass 1
    // wrote, so without this compute->compute barrier the dispatches can overlap and read stale
    // data.
    vk::computeBarrier(cmd);
    // Pass 2: scatter updates into out at the index rows.
    if (pc.total > 0)
      scatterPipe->dispatch(cmd, {updates->handle(), idx->handle(), out->handle()}, &pc, sizeof(pc),
                            groups(pc.total, 256));
  }
};

}  // namespace
VKNN_REGISTER_VK_OP(OpType::kScatterND, ScatterNDOp);
}  // namespace vknn
