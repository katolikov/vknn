// vxrt — Vulkan compute operators (NC4HW4, fp32). Conv (general/depthwise), Add,
// GlobalAveragePool, Gemm/FC, Reshape. Weights are prepacked + uploaded in prepare().
#include <cstring>
#include <vector>
#include "vk_backend.h"
#include "vx/logging.h"

namespace vx {

// fp16 shader variants are added separately; off until registered.
bool vxVulkanFp16Available() { return false; }

namespace {

// ---- push-constant layouts (must match shaders/*.comp) ----
struct ConvPC {
  int N, Cin, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act;
  float actLo, actHi;
};
struct DwPC {
  int N, C, H, W, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act, pad0;
  float actLo, actHi;
};
struct PoolPC { int N, C, H, W; };
struct FcPC { int Cin, Cout, act; float actLo, actHi; };

static std::vector<int64_t> ints(const Node& n, const char* k, std::vector<int64_t> d) {
  const auto& v = n.attr.getints(k); return v.empty() ? d : v;
}
static uint32_t groups64(int64_t total) { return (uint32_t)((total + 63) / 64); }

// upload a host float vector into a new device buffer
static std::shared_ptr<vk::Buffer> uploadFloats(vk::VulkanContext& ctx,
                                                 const std::vector<float>& data) {
  auto b = std::make_shared<vk::Buffer>(ctx, std::max<size_t>(data.size(), 4) * 4,
                                        vk::MemPref::kAuto);
  b->upload(data.data(), data.size() * 4);
  return b;
}

// ============================ Conv (general + depthwise) ============================
struct ConvVulkanOp : VulkanOp {
  bool depthwise = false;
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer> wbuf, bbuf;
  ConvPC pc{};
  DwPC dpc{};
  int64_t total = 0;

  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    TensorId xid = node.inputs[0], wid = node.inputs[1];
    NCHW x = NCHW::from(g.desc(xid).shape);
    NCHW y = NCHW::from(g.desc(node.outputs[0]).shape);
    const Shape& ws = g.desc(wid).shape;  // [Cout, Cin/g, KH, KW]
    int64_t Cout = ws[0], inCg = ws[1], KH = ws[2], KW = ws[3];
    auto st = ints(node, "strides", {1, 1});
    auto pad = ints(node, "pads", {0, 0, 0, 0});
    auto dil = ints(node, "dilations", {1, 1});
    int64_t group = node.attr.geti("group", 1);
    depthwise = (group == x.c && group == Cout && inCg == 1);

    const float* wsrc = g.initializers.at(wid).f32();
    // bias (optional) -> padded to multiple of 4
    int64_t Coutb = cBlocks(Cout);
    std::vector<float> bias(Coutb * 4, 0.f);
    if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor) {
      const float* bsrc = g.initializers.at(node.inputs[2]).f32();
      for (int64_t i = 0; i < Cout; ++i) bias[i] = bsrc[i];
    }
    bbuf = uploadFloats(*env.ctx, bias);

    if (depthwise) {
      // weight [C,1,KH,KW] -> [Cb][KH][KW][4]
      int64_t Cb = cBlocks(x.c);
      std::vector<float> wp(Cb * KH * KW * 4, 0.f);
      for (int64_t c = 0; c < x.c; ++c) {
        int64_t cb = c / 4, l = c % 4;
        for (int64_t ky = 0; ky < KH; ++ky)
          for (int64_t kx = 0; kx < KW; ++kx)
            wp[(((cb * KH + ky) * KW + kx) * 4) + l] = wsrc[((c * KH + ky) * KW + kx)];
      }
      wbuf = uploadFloats(*env.ctx, wp);
      dpc = {(int)x.n, (int)x.c, (int)x.h, (int)x.w, (int)y.h, (int)y.w, (int)KH, (int)KW,
             (int)st[0], (int)st[1], (int)pad[0], (int)pad[1], (int)dil[0], (int)dil[1],
             (int)node.fusedAct, 0, node.actLo, node.actHi};
      total = x.n * Cb * y.h * y.w;
      pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "dwconv", 4, sizeof(DwPC),
                                                   std::vector<uint32_t>{}, env.cache->handle());
    } else {
      // general group=1: weight [Cout,Cin,KH,KW] -> [Cout][Cinb][KH][KW][4]
      int64_t Cinb = cBlocks(x.c);
      std::vector<float> wp((size_t)Cout * Cinb * KH * KW * 4, 0.f);
      for (int64_t oc = 0; oc < Cout; ++oc)
        for (int64_t ic = 0; ic < x.c; ++ic) {
          int64_t icb = ic / 4, l = ic % 4;
          for (int64_t ky = 0; ky < KH; ++ky)
            for (int64_t kx = 0; kx < KW; ++kx)
              wp[(((((oc * Cinb + icb) * KH + ky) * KW + kx) * 4) + l)] =
                  wsrc[(((oc * x.c + ic) * KH + ky) * KW + kx)];
        }
      wbuf = uploadFloats(*env.ctx, wp);
      pc = {(int)x.n, (int)x.c, (int)x.h, (int)x.w, (int)Cout, (int)y.h, (int)y.w, (int)KH,
            (int)KW, (int)st[0], (int)st[1], (int)pad[0], (int)pad[1], (int)dil[0], (int)dil[1],
            (int)node.fusedAct, node.actLo, node.actHi};
      total = x.n * Coutb * y.h * y.w;
      pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "conv", 4, sizeof(ConvPC),
                                                   std::vector<uint32_t>{}, env.cache->handle());
    }
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = env.devBuf(node.inputs[0]);
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    std::vector<VkBuffer> bufs = {src->handle(), wbuf->handle(), bbuf->handle(), dst->handle()};
    if (depthwise)
      pipe->dispatch(cmd, bufs, &dpc, sizeof(dpc), groups64(total));
    else
      pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups64(total));
  }
};

// ============================ Add (NC4HW4 elementwise) ============================
struct AddVulkanOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  uint32_t count = 0;
  void prepare(const Node& node, VkOpEnv& env) override {
    count = (uint32_t)packedElems(env.graph->desc(node.outputs[0]).shape);
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "add", 3, sizeof(uint32_t),
                                                 std::vector<uint32_t>{}, env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* a = env.devBuf(node.inputs[0]);
    vk::Buffer* b = env.devBuf(node.inputs[1]);
    vk::Buffer* y = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {a->handle(), b->handle(), y->handle()}, &count, sizeof(count),
                   groups256(count));
  }
  static uint32_t groups256(int64_t total) { return (uint32_t)((total + 255) / 256); }
};

// ============================ GlobalAveragePool ============================
struct AvgPoolVulkanOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  PoolPC pc{};
  int64_t total = 0;
  void prepare(const Node& node, VkOpEnv& env) override {
    NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
    pc = {(int)x.n, (int)x.c, (int)x.h, (int)x.w};
    total = x.n * cBlocks(x.c);
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "avgpool", 2, sizeof(PoolPC),
                                                 std::vector<uint32_t>{}, env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = env.devBuf(node.inputs[0]);
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {src->handle(), dst->handle()}, &pc, sizeof(pc), groups64(total));
  }
};

// ============================ Gemm / FC (classifier) ============================
struct GemmVulkanOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer> wbuf, bbuf;
  FcPC pc{};
  int64_t Cout = 0;
  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    const Shape& ws = g.desc(node.inputs[1]).shape;  // [Cout,Cin] (transB=1) or [Cin,Cout]
    int64_t transB = node.attr.geti("transB", 0);
    int64_t Cin, CoutL;
    if (transB) { CoutL = ws[0]; Cin = ws[1]; } else { Cin = ws[0]; CoutL = ws[1]; }
    Cout = CoutL;
    const float* wsrc = g.initializers.at(node.inputs[1]).f32();
    std::vector<float> wp((size_t)CoutL * Cin);
    for (int64_t oc = 0; oc < CoutL; ++oc)
      for (int64_t ic = 0; ic < Cin; ++ic)
        wp[oc * Cin + ic] = transB ? wsrc[oc * Cin + ic] : wsrc[ic * CoutL + oc];
    wbuf = uploadFloats(*env.ctx, wp);
    std::vector<float> bias(CoutL, 0.f);
    if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor) {
      const float* bsrc = g.initializers.at(node.inputs[2]).f32();
      for (int64_t i = 0; i < CoutL; ++i) bias[i] = bsrc[i];
    }
    bbuf = uploadFloats(*env.ctx, bias);
    pc = {(int)Cin, (int)CoutL, (int)node.fusedAct, node.actLo, node.actHi};
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "fc", 4, sizeof(FcPC),
                                                 std::vector<uint32_t>{}, env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = env.devBuf(node.inputs[0]);
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {src->handle(), wbuf->handle(), bbuf->handle(), dst->handle()}, &pc,
                   sizeof(pc), groups64(Cout));
  }
};

// ============================ Reshape (packed alias via copy) ============================
struct ReshapeVulkanOp : VulkanOp {
  size_t bytes = 0;
  void prepare(const Node& node, VkOpEnv& env) override {
    int el = env.useFp16 ? 2 : 4;
    bytes = (size_t)packedElems(env.graph->desc(node.outputs[0]).shape) * el;
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = env.devBuf(node.inputs[0]);
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    size_t n = std::min({bytes, src->bytes(), dst->bytes()});
    VkBufferCopy c{0, 0, n};
    vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
  }
};

}  // namespace

VX_REGISTER_VK_OP(OpType::kConv, ConvVulkanOp);
VX_REGISTER_VK_OP(OpType::kAdd, AddVulkanOp);
VX_REGISTER_VK_OP(OpType::kGlobalAvgPool, AvgPoolVulkanOp);
VX_REGISTER_VK_OP(OpType::kGemm, GemmVulkanOp);
VX_REGISTER_VK_OP(OpType::kReshape, ReshapeVulkanOp);

}  // namespace vx
