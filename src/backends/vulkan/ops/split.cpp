// Split on the GPU, two paths:
//  - NC4HW4 channel split (4D, axis=1, every output's channels a multiple of 4): each output is a
//    contiguous range of channel-blocks, a plain buffer copy (keeps YOLO's C2f blocks on the GPU).
//  - Flat row-major split (any other axis, e.g. the encoder's last-axis splits): each output is a
//    Slice along the split axis, dispatched via the flat_gather shader (base = axis offset).
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vx {
namespace {

struct SplitOp : VulkanOp {
  bool flat_ = false;
  // ---- NC4HW4 channel path ----
  struct Part { int outIdx; int64_t blockOff, cbk; };
  std::vector<Part> parts_;
  NCHW x_{};
  int elem_ = 4;
  int64_t cbTotal_ = 0, hw_ = 0;
  // ---- flat path ----
  struct FPC { int rank, total, base; int outDim[flat::kMaxRank]; int inStride[flat::kMaxRank]; };
  std::vector<FPC> fpcs_;
  std::vector<int> foutIdx_;
  std::vector<std::unique_ptr<vk::ComputePipeline>> fpipes_;
  std::shared_ptr<vk::Buffer> hold0_;

  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    flat_ = !node.outputs.empty() && node.outputs[0] != kNoTensor && g.desc(node.outputs[0]).gpuFlat;
    if (flat_) {
      Shape in = g.desc(node.inputs[0]).shape;
      int rank = (int)in.size();
      int axis = (int)node.attr.geti("axis", 0);
      if (axis < 0) axis += rank;
      auto inStride = flat::rowStrides(in);
      int64_t offset = 0;
      for (size_t k = 0; k < node.outputs.size(); ++k) {
        if (node.outputs[k] == kNoTensor) continue;
        Shape out = g.desc(node.outputs[k]).shape;
        FPC pc{};
        pc.rank = rank;
        pc.total = (int)numElements(out);
        pc.base = (int)(offset * inStride[axis]);
        for (int d = 0; d < rank; ++d) { pc.outDim[d] = (int)out[d]; pc.inStride[d] = (int)inStride[d]; }
        fpcs_.push_back(pc);
        foutIdx_.push_back((int)k);
        offset += out[axis];
        fpipes_.push_back(std::make_unique<vk::ComputePipeline>(
            *env.ctx, shader("flat_gather", env.useFp16), 2, sizeof(FPC), std::vector<uint32_t>{},
            env.cache->handle()));
      }
      return;
    }
    // NC4HW4 channel split
    x_ = NCHW::from(g.desc(node.inputs[0]).shape);
    elem_ = env.useFp16 ? 2 : 4;
    cbTotal_ = cBlocks(x_.c);
    hw_ = x_.h * x_.w;
    int64_t blk = 0;
    for (size_t k = 0; k < node.outputs.size(); ++k) {
      if (node.outputs[k] == kNoTensor) continue;
      int64_t ck = NCHW::from(g.desc(node.outputs[k]).shape).c;
      int64_t cbk = cBlocks(ck);
      parts_.push_back({(int)k, blk, cbk});
      blk += cbk;
    }
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    if (flat_) {
      vk::Buffer* src = operandBuf(env, node.inputs[0], hold0_);
      for (size_t i = 0; i < fpcs_.size(); ++i)
        fpipes_[i]->dispatch(cmd, {src->handle(), env.devBuf(node.outputs[foutIdx_[i]])->handle()},
                             &fpcs_[i], sizeof(FPC), groups(fpcs_[i].total, 256));
      return;
    }
    vk::Buffer* src = env.devBuf(node.inputs[0]);
    for (const Part& p : parts_) {
      vk::Buffer* dst = env.devBuf(node.outputs[p.outIdx]);
      for (int64_t n = 0; n < x_.n; ++n) {
        VkBufferCopy c{};
        c.srcOffset = (VkDeviceSize)((n * cbTotal_ + p.blockOff) * hw_ * 4 * elem_);
        c.dstOffset = (VkDeviceSize)((n * p.cbk) * hw_ * 4 * elem_);
        c.size = (VkDeviceSize)(p.cbk * hw_ * 4 * elem_);
        vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
      }
    }
  }
};

}  // namespace

VX_REGISTER_VK_OP(OpType::kSplit, SplitOp);

}  // namespace vx
