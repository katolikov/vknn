// GridSample on the GPU (NC4HW4 data, flat constant grid). Gated to a constant grid in supportsNode
// (the grid's [N,H,W,2] layout can't go through NC4HW4 input packing; a constant is uploaded raw
// here, like conv weights). Runtime grids fall back to the CPU op.
#include "vk_op_common.h"
#include "vx/op.h"

namespace vx {
namespace {
struct GsPC { int N, C, Hin, Win, OH, OW, align; };
struct GridSampleOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer> gridBuf;
  GsPC pc{};
  int64_t total = 0;
  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    NCHW x = NCHW::from(g.desc(node.inputs[0]).shape);
    const Shape& gs = g.desc(node.inputs[1]).shape;  // [N,Hout,Wout,2]
    int OH = (int)gs[1], OW = (int)gs[2];
    pc = {(int)x.n, (int)x.c, (int)x.h, (int)x.w, OH, OW, (int)node.attr.geti("align_corners", 0)};
    std::string mode = node.attr.gets("mode", "bilinear");
    uint32_t MODE = (mode == "nearest") ? 1u : 0u;
    std::string pad = node.attr.gets("padding_mode", "zeros");
    uint32_t PAD = pad == "border" ? 1u : (pad == "reflection" ? 2u : 0u);
    // upload the constant grid as a raw fp32 buffer (2 floats per output pixel)
    std::vector<float> gv = initFloats(g, node.inputs[1]);
    int64_t nf = (int64_t)gv.size();
    gridBuf = std::make_shared<vk::Buffer>(*env.ctx, std::max<int64_t>(nf * 4, 16), vk::MemPref::kAuto);
    gridBuf->upload(gv.data(), nf * 4);
    total = (int64_t)x.n * cBlocks(x.c) * OH * OW;
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("gridsample", env.useFp16), 3,
                                                 sizeof(GsPC), std::vector<uint32_t>{MODE, PAD},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* s = env.devBuf(node.inputs[0]);
    vk::Buffer* d = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {s->handle(), gridBuf->handle(), d->handle()}, &pc, sizeof(pc),
                   groups(total, 64));
  }
};
}  // namespace
VX_REGISTER_VK_OP(OpType::kGridSample, GridSampleOp);
}  // namespace vx
