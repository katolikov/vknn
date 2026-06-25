// Split along the channel axis in NC4HW4. The packed layout is [N][Cblock][H*W][4], so when every
// output's channel count is a multiple of 4 each output is a CONTIGUOUS range of channel-blocks —
// a plain buffer copy per output, no shader. (Non-4-aligned channel splits fall back to the CPU op
// via supportsNode.) This keeps C2f blocks (YOLO) fully on the GPU instead of round-tripping to CPU.
#include "vk_op_common.h"

namespace vx {
namespace {

struct SplitOp : VulkanOp {
  struct Part {
    int outIdx;
    int64_t blockOff, cbk;  // first channel-block + block count for this output
  };
  std::vector<Part> parts_;
  NCHW x_{};
  int elem_ = 4;
  int64_t cbTotal_ = 0, hw_ = 0;

  void prepare(const Node& node, VkOpEnv& env) override {
    x_ = NCHW::from(env.graph->desc(node.inputs[0]).shape);
    elem_ = env.useFp16 ? 2 : 4;
    cbTotal_ = cBlocks(x_.c);
    hw_ = x_.h * x_.w;
    int64_t blk = 0;
    for (size_t k = 0; k < node.outputs.size(); ++k) {
      if (node.outputs[k] == kNoTensor) continue;
      int64_t ck = NCHW::from(env.graph->desc(node.outputs[k]).shape).c;
      int64_t cbk = cBlocks(ck);
      parts_.push_back({(int)k, blk, cbk});
      blk += cbk;
    }
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
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
