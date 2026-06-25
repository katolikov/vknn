// Generic flat (row-major) GPU op implementations for the detection-head graph: Transpose, Slice,
// Concat, broadcasting Binary/Add, non-channel Softmax. The layout pass (insertLayoutConverts) stores
// these tensors as flat buffers and inserts ConvertLayout at the NC4HW4 boundary, so a whole YOLO
// head runs on the GPU. Ops with an existing NC4HW4 kernel (concat/binary/add/softmax) hold one of
// these and dispatch to it when their tensors are flat; Transpose/Slice are flat-only.
#pragma once
#include "import/passes.h"  // readI64Param
#include "vk_op_common.h"

namespace vx {

// A node runs the flat path iff the layout pass marked its (first) output as a flat GPU tensor.
inline bool opIsFlat(const Node& node, VkOpEnv& env) {
  return !node.outputs.empty() && node.outputs[0] != kNoTensor &&
         env.graph->desc(node.outputs[0]).gpuFlat;
}

namespace flat {

constexpr int kMaxRank = 6;
inline std::vector<int64_t> rowStrides(const Shape& s) {
  std::vector<int64_t> st(s.size(), 1);
  for (int k = (int)s.size() - 2; k >= 0; --k) st[k] = st[k + 1] * s[k + 1];
  return st;
}

// ---- Transpose / Slice: out[i] = in[base + sum outCoord_k * inStride_k] ----
struct Gather {
  struct PC { int rank, total, base, outDim[kMaxRank], inStride[kMaxRank]; } pc{};
  std::unique_ptr<vk::ComputePipeline> pipe;
  void prepare(const Node& node, VkOpEnv& env) {
    const Graph& g = *env.graph;
    Shape in = g.desc(node.inputs[0]).shape, out = g.desc(node.outputs[0]).shape;
    auto inStride = rowStrides(in);
    int rank = (int)out.size();
    pc.rank = rank;
    pc.total = (int)numElements(out);
    pc.base = 0;
    if (node.type == OpType::kTranspose) {
      const auto& perm = node.attr.getints("perm");
      for (int k = 0; k < rank; ++k) {
        int p = perm.empty() ? rank - 1 - k : (int)perm[k];
        pc.outDim[k] = (int)in[p];
        pc.inStride[k] = (int)inStride[p];
      }
    } else {  // kSlice
      int r = (int)in.size();
      auto starts = readI64Param(g, node, "starts", 1), ends = readI64Param(g, node, "ends", 2);
      auto axes = readI64Param(g, node, "axes", 3), steps = readI64Param(g, node, "steps", 4);
      std::vector<int64_t> start(r, 0), step(r, 1);
      for (size_t a = 0; a < starts.size() && a < ends.size(); ++a) {
        int ax = axes.empty() ? (int)a : (int)(axes[a] < 0 ? axes[a] + r : axes[a]);
        if (ax < 0 || ax >= r) continue;
        int s0 = (int)(starts[a] < 0 ? starts[a] + in[ax] : starts[a]);
        start[ax] = std::max(0, std::min(s0, (int)in[ax]));
        step[ax] = (int)(steps.size() > a ? steps[a] : 1);
      }
      for (int k = 0; k < rank; ++k) {
        pc.outDim[k] = (int)out[k];
        pc.inStride[k] = (int)(inStride[k] * step[k]);
        pc.base += (int)(start[k] * inStride[k]);
      }
    }
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("flat_gather", env.useFp16), 2,
                                                 sizeof(PC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) {
    pipe->dispatch(cmd, {env.devBuf(node.inputs[0])->handle(), env.devBuf(node.outputs[0])->handle()},
                   &pc, sizeof(pc), groups(pc.total, 256));
  }
};

// ---- Concat: scatter each input into the output at its axis offset ----
struct Concat {
  struct PC { int rank, total, base, inDim[kMaxRank], outStride[kMaxRank]; };
  std::vector<std::unique_ptr<vk::ComputePipeline>> pipes;
  std::vector<PC> pcs;
  std::vector<int> inIdx;
  void prepare(const Node& node, VkOpEnv& env) {
    const Graph& g = *env.graph;
    Shape out = g.desc(node.outputs[0]).shape;
    int rank = (int)out.size();
    int64_t axis = node.attr.geti("axis", 1);
    if (axis < 0) axis += rank;
    auto outStride = rowStrides(out);
    int64_t offset = 0;
    for (size_t e = 0; e < node.inputs.size(); ++e) {
      if (node.inputs[e] == kNoTensor) continue;
      Shape in = g.desc(node.inputs[e]).shape;
      PC pc{};
      pc.rank = rank;
      pc.total = (int)numElements(in);
      pc.base = (int)(offset * outStride[axis]);
      for (int k = 0; k < rank; ++k) {
        pc.inDim[k] = (int)in[k];
        pc.outStride[k] = (int)outStride[k];
      }
      pcs.push_back(pc);
      inIdx.push_back((int)e);
      offset += in[axis];
      pipes.push_back(std::make_unique<vk::ComputePipeline>(
          *env.ctx, shader("flat_scatter", env.useFp16), 2, sizeof(PC), std::vector<uint32_t>{},
          env.cache->handle()));
    }
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) {
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    for (size_t i = 0; i < pcs.size(); ++i)
      pipes[i]->dispatch(cmd, {env.devBuf(node.inputs[inIdx[i]])->handle(), dst->handle()}, &pcs[i],
                         sizeof(PC), groups(pcs[i].total, 256));
  }
};

// ---- broadcasting Binary / Add (handles a constant operand by uploading it flat) ----
struct Binary {
  struct PC { int rank, total, op, outDim[kMaxRank], aStride[kMaxRank], bStride[kMaxRank]; } pc{};
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer> constBuf[2];
  void prepare(const Node& node, VkOpEnv& env) {
    const Graph& g = *env.graph;
    Shape out = g.desc(node.outputs[0]).shape;
    int rank = (int)out.size();
    pc.rank = rank;
    pc.total = (int)numElements(out);
    pc.op = node.type == OpType::kAdd ? kBAdd : node.subOp;
    for (int k = 0; k < rank; ++k) pc.outDim[k] = (int)out[k];
    auto setup = [&](TensorId t, int which) {
      Shape s = g.desc(t).shape;
      std::vector<int64_t> ps(rank, 1);  // left-pad to out rank
      for (int k = 0; k < (int)s.size(); ++k) ps[rank - (int)s.size() + k] = s[k];
      auto st = rowStrides(ps);
      for (int k = 0; k < rank; ++k) {
        int stride = ps[k] == 1 ? 0 : (int)st[k];
        (which == 0 ? pc.aStride : pc.bStride)[k] = stride;
      }
      if (g.isInitializer(t)) {
        const HostBuffer& hb = g.initializers.at(t);
        constBuf[which] = upload(*env.ctx, std::vector<float>(hb.f32(), hb.f32() + numElements(s)),
                                 env.useFp16);
      }
    };
    setup(node.inputs[0], 0);
    setup(node.inputs[1], 1);
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("flat_binary", env.useFp16), 3,
                                                 sizeof(PC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) {
    auto buf = [&](int e) { return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]); };
    pipe->dispatch(cmd, {buf(0)->handle(), buf(1)->handle(), env.devBuf(node.outputs[0])->handle()},
                   &pc, sizeof(pc), groups(pc.total, 256));
  }
};

// ---- Softmax over an arbitrary axis ----
struct Softmax {
  struct PC { int outer, axis, inner; } pc{};
  std::unique_ptr<vk::ComputePipeline> pipe;
  void prepare(const Node& node, VkOpEnv& env) {
    Shape s = env.graph->desc(node.inputs[0]).shape;
    int rank = (int)s.size();
    int64_t axis = node.attr.geti("axis", -1);
    if (axis < 0) axis += rank;
    int64_t outer = 1, inner = 1;
    for (int k = 0; k < (int)axis; ++k) outer *= s[k];
    for (int k = (int)axis + 1; k < rank; ++k) inner *= s[k];
    pc = {(int)outer, (int)s[axis], (int)inner};
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("flat_softmax", env.useFp16), 2,
                                                 sizeof(PC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) {
    pipe->dispatch(cmd, {env.devBuf(node.inputs[0])->handle(), env.devBuf(node.outputs[0])->handle()},
                   &pc, sizeof(pc), groups((int64_t)pc.outer * pc.inner, 256));
  }
};

}  // namespace flat
}  // namespace vx
