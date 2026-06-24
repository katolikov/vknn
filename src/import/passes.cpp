#include "passes.h"
#include <algorithm>
#include <cmath>
#include <set>
#include "backends/cpu/cpu_backend.h"
#include "vx/logging.h"

namespace vx {

void inferShapes(Graph& g, int64_t batch) {
  // Resolve dynamic dims on inputs to `batch`.
  for (TensorId in : g.inputs) {
    auto& s = g.desc(in).shape;
    for (auto& d : s)
      if (d < 0) d = batch;
  }
  auto SH = [&](TensorId id) -> Shape& { return g.desc(id).shape; };
  for (auto& nd : g.nodes) {
    if (nd.outputs.empty()) continue;
    TensorId o = nd.outputs[0];
    switch (nd.type) {
      case OpType::kConv: {
        NCHW x = NCHW::from(SH(nd.inputs[0]));
        const Shape& w = SH(nd.inputs[1]);
        if (w.size() < 4) break;
        int64_t outC = w[0], kh = w[2], kw = w[3];
        auto ints = [&](const char* k, std::vector<int64_t> d) {
          const auto& v = nd.attr.getints(k);
          return v.empty() ? d : v;
        };
        auto st = ints("strides", {1, 1});
        auto pad = ints("pads", {0, 0, 0, 0});
        auto dil = ints("dilations", {1, 1});
        int64_t oh = (x.h + pad[0] + pad[2] - (dil[0] * (kh - 1) + 1)) / st[0] + 1;
        int64_t ow = (x.w + pad[1] + pad[3] - (dil[1] * (kw - 1) + 1)) / st[1] + 1;
        SH(o) = {x.n, outC, oh, ow};
        break;
      }
      case OpType::kClip:
      case OpType::kRelu:
      case OpType::kBatchNorm:
      case OpType::kIdentity:
      case OpType::kUnary:
        SH(o) = SH(nd.inputs[0]);
        break;
      case OpType::kBinary:
      case OpType::kAdd: {
        const Shape& a = SH(nd.inputs[0]);
        const Shape& b = SH(nd.inputs[1]);
        SH(o) = (numElements(a) >= numElements(b)) ? a : b;
        break;
      }
      case OpType::kGlobalAvgPool: {
        NCHW x = NCHW::from(SH(nd.inputs[0]));
        SH(o) = {x.n, x.c, 1, 1};
        break;
      }
      case OpType::kMaxPool:
      case OpType::kAvgPool: {
        NCHW x = NCHW::from(SH(nd.inputs[0]));
        auto ints = [&](const char* k, std::vector<int64_t> d) {
          const auto& v = nd.attr.getints(k);
          return v.empty() ? d : v;
        };
        auto ks = ints("kernel_shape", {1, 1});
        auto st = ints("strides", {1, 1});
        auto pad = ints("pads", {0, 0, 0, 0});
        int64_t oh = (x.h + pad[0] + pad[2] - ks[0]) / st[0] + 1;
        int64_t ow = (x.w + pad[1] + pad[3] - ks[1]) / st[1] + 1;
        SH(o) = {x.n, x.c, oh, ow};
        break;
      }
      case OpType::kGemm: {
        const Shape& a = SH(nd.inputs[0]);
        const Shape& w = SH(nd.inputs[1]);
        int64_t transB = nd.attr.geti("transB", 0);
        int64_t M = a.empty() ? 1 : a[0];
        int64_t N = w.size() < 2 ? 0 : (transB ? w[0] : w[1]);
        SH(o) = {M, N};
        break;
      }
      case OpType::kFlatten: {
        const Shape& a = SH(nd.inputs[0]);
        int64_t axis = nd.attr.geti("axis", 1), d0 = 1, d1 = 1;
        for (int64_t i = 0; i < (int64_t)a.size(); ++i) (i < axis ? d0 : d1) *= a[i];
        SH(o) = {d0, d1};
        break;
      }
      case OpType::kReshape: {
        TensorId sid = nd.inputs[1];
        if (!g.isInitializer(sid)) break;  // shape becomes const after constFold; 2nd pass fills it
        const HostBuffer& hb = g.initializers[sid];
        const Shape& in = SH(nd.inputs[0]);
        int64_t rank = numElements(g.desc(sid).shape);
        if (rank <= 0) rank = (int64_t)(hb.bytes.size() / 8);
        Shape out(rank);
        int64_t known = 1, infer = -1;
        for (int64_t i = 0; i < rank; ++i) {
          int64_t d = hb.i64()[i];
          if (d == 0) d = (i < (int64_t)in.size()) ? in[i] : 1;
          out[i] = d;
          if (d == -1)
            infer = i;
          else
            known *= d;
        }
        if (infer >= 0) out[infer] = numElements(in) / std::max<int64_t>(known, 1);
        SH(o) = out;
        break;
      }
      default:
        break;  // shape-path ops resolved by constFold
    }
  }
}

void constFold(Graph& g) {
  std::set<TensorId> known;
  for (auto& kv : g.initializers) known.insert(kv.first);
  std::vector<RtTensor> pool(g.tensors.size());
  for (size_t i = 0; i < pool.size(); ++i) {
    pool[i].id = (TensorId)i;
    pool[i].shape = g.tensors[i].shape;
    pool[i].dtype = g.tensors[i].dtype;
    if (g.isInitializer((TensorId)i)) {
      pool[i].host = g.initializers[i];
      pool[i].hostValid = true;
    }
  }
  ExecContext ctx;
  ctx.pool = &pool;
  ctx.graph = &g;
  Config cfg;
  ctx.config = &cfg;

  std::set<int> removeNodes;
  auto foldable = [&](const Node& nd) {
    switch (nd.type) {
      case OpType::kConstant:
        return true;
      case OpType::kShape:
        return !g.desc(nd.inputs[0]).shape.empty();  // shape known
      case OpType::kGather:
      case OpType::kUnsqueeze:
      case OpType::kConcat: {
        for (TensorId in : nd.inputs)
          if (in != kNoTensor && !known.count(in)) return false;
        return true;
      }
      default:
        return false;
    }
  };

  for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
    Node& nd = g.nodes[ni];
    if (!foldable(nd)) continue;
    // ensure Shape's input has a shape-only RtTensor
    if (nd.type == OpType::kShape) pool[nd.inputs[0]].shape = g.desc(nd.inputs[0]).shape;
    auto op = CpuOpRegistry::instance().create(nd.type);
    if (!op) continue;
    try {
      op->run(nd, ctx);
    } catch (...) {
      continue;
    }
    for (TensorId o : nd.outputs) {
      if (o == kNoTensor) continue;
      g.initializers[o] = pool[o].host;
      g.desc(o).isInitializer = true;
      g.desc(o).shape = pool[o].shape;
      g.desc(o).dtype = pool[o].dtype;
      known.insert(o);
    }
    removeNodes.insert((int)ni);
  }
  if (!removeNodes.empty()) {
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); ++i)
      if (!removeNodes.count((int)i)) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    VX_INFO << "constFold: folded " << removeNodes.size() << " shape-path node(s)";
  }
}

void foldBatchNorm(Graph& g) {
  // Fold BN(Conv(x)) -> Conv with adjusted weights/bias. MobileNetV2 ships BN pre-folded,
  // so this is typically a no-op here, but implemented for correctness on other models.
  int folded = 0;
  std::set<int> remove;
  // map tensor -> producing node index
  std::vector<int> producer(g.tensors.size(), -1);
  for (size_t i = 0; i < g.nodes.size(); ++i)
    for (TensorId o : g.nodes[i].outputs)
      if (o != kNoTensor) producer[o] = (int)i;

  for (size_t i = 0; i < g.nodes.size(); ++i) {
    Node& bn = g.nodes[i];
    if (bn.type != OpType::kBatchNorm) continue;
    int pi = producer[bn.inputs[0]];
    if (pi < 0 || g.nodes[pi].type != OpType::kConv) continue;
    Node& conv = g.nodes[pi];
    // only fold if conv output feeds only this BN
    int consumers = 0;
    for (auto& nn : g.nodes)
      for (TensorId in : nn.inputs)
        if (in == conv.outputs[0]) consumers++;
    if (consumers != 1) continue;

    const auto& scale = g.initializers[bn.inputs[1]].f32();
    const auto& bias = g.initializers[bn.inputs[2]].f32();
    const auto& mean = g.initializers[bn.inputs[3]].f32();
    const auto& var = g.initializers[bn.inputs[4]].f32();
    float eps = bn.attr.getf("epsilon", 1e-5f);
    int64_t outC = g.desc(conv.inputs[1]).shape[0];
    int64_t perOC = numElements(g.desc(conv.inputs[1]).shape) / outC;
    HostBuffer& W = g.initializers[conv.inputs[1]];
    // ensure bias exists
    TensorId biasId =
        (conv.inputs.size() > 2 && conv.inputs[2] != kNoTensor) ? conv.inputs[2] : kNoTensor;
    HostBuffer biasBuf;
    if (biasId == kNoTensor) {
      biasBuf.resizeElems(outC, DType::kFloat32);
    }
    HostBuffer& Bb = (biasId == kNoTensor) ? biasBuf : g.initializers[biasId];
    for (int64_t oc = 0; oc < outC; ++oc) {
      float a = scale[oc] / std::sqrt(var[oc] + eps);
      float* w = W.f32() + oc * perOC;
      for (int64_t k = 0; k < perOC; ++k) w[k] *= a;
      Bb.f32()[oc] = (Bb.f32()[oc] - mean[oc]) * a + bias[oc];
    }
    if (biasId == kNoTensor) {
      TensorDesc d;
      d.name = conv.name + "_bias";
      d.shape = {outC};
      d.isInitializer = true;
      TensorId nb = g.addTensor(d);
      g.initializers[nb] = std::move(biasBuf);
      if (conv.inputs.size() < 3) conv.inputs.resize(3, kNoTensor);
      conv.inputs[2] = nb;
    }
    // rewire: BN consumers now read conv output
    TensorId bnOut = bn.outputs[0], convOut = conv.outputs[0];
    for (auto& nn : g.nodes)
      for (TensorId& in : nn.inputs)
        if (in == bnOut) in = convOut;
    for (TensorId& go : g.outputs)
      if (go == bnOut) go = convOut;
    remove.insert((int)i);
    folded++;
  }
  if (folded) {
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); ++i)
      if (!remove.count((int)i)) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    VX_INFO << "foldBatchNorm: folded " << folded << " BN node(s) into Conv";
  }
}

void fuseActivations(Graph& g) {
  std::vector<int> producer(g.tensors.size(), -1);
  for (size_t i = 0; i < g.nodes.size(); ++i)
    for (TensorId o : g.nodes[i].outputs)
      if (o != kNoTensor) producer[o] = (int)i;
  std::set<int> remove;
  int fused = 0;
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    Node& act = g.nodes[i];
    if (act.type != OpType::kClip && act.type != OpType::kRelu) continue;
    int pi = producer[act.inputs[0]];
    if (pi < 0) continue;
    Node& prod = g.nodes[pi];
    if (prod.type != OpType::kConv && prod.type != OpType::kGemm && prod.type != OpType::kAdd)
      continue;
    if (prod.fusedAct != ActType::kNone) continue;
    // producer output must feed only this activation
    int consumers = 0;
    for (auto& nn : g.nodes)
      for (TensorId in : nn.inputs)
        if (in == prod.outputs[0]) consumers++;
    for (TensorId go : g.outputs)
      if (go == prod.outputs[0]) consumers++;
    if (consumers != 1) continue;

    if (act.type == OpType::kRelu) {
      prod.fusedAct = ActType::kRelu;
    } else {
      float lo = 0, hi = 6;  // default relu6
      if (act.inputs.size() > 1 && act.inputs[1] != kNoTensor && g.isInitializer(act.inputs[1]))
        lo = g.initializers[act.inputs[1]].f32()[0];
      if (act.inputs.size() > 2 && act.inputs[2] != kNoTensor && g.isInitializer(act.inputs[2]))
        hi = g.initializers[act.inputs[2]].f32()[0];
      if (act.attr.has("min")) lo = act.attr.getf("min", lo);
      if (act.attr.has("max")) hi = act.attr.getf("max", hi);
      prod.fusedAct = (lo == 0.f && hi == 6.f) ? ActType::kRelu6 : ActType::kClip;
      prod.actLo = lo;
      prod.actHi = hi;
    }
    // rewire consumers of act output to producer output
    TensorId actOut = act.outputs[0], prodOut = prod.outputs[0];
    for (auto& nn : g.nodes)
      for (TensorId& in : nn.inputs)
        if (in == actOut) in = prodOut;
    for (TensorId& go : g.outputs)
      if (go == actOut) go = prodOut;
    remove.insert((int)i);
    fused++;
  }
  if (fused) {
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); ++i)
      if (!remove.count((int)i)) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    VX_INFO << "fuseActivations: fused " << fused << " activation(s) into Conv/Gemm";
  }
}

void eliminateIdentity(Graph& g) {
  // Drop Identity nodes by pointing their consumers (and any graph output) straight at the input.
  std::set<int> remove;
  int n = 0;
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    Node& id = g.nodes[i];
    if (id.type != OpType::kIdentity) continue;
    if (id.inputs.empty() || id.outputs.empty()) continue;
    TensorId in = id.inputs[0], out = id.outputs[0];
    for (auto& nn : g.nodes)
      for (TensorId& x : nn.inputs)
        if (x == out) x = in;
    for (TensorId& go : g.outputs)
      if (go == out) go = in;
    remove.insert((int)i);
    ++n;
  }
  if (n) {
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); ++i)
      if (!remove.count((int)i)) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    VX_INFO << "eliminateIdentity: removed " << n << " Identity node(s)";
  }
}

void eliminateDeadNodes(Graph& g) {
  std::set<TensorId> live(g.outputs.begin(), g.outputs.end());
  bool changed = true;
  std::vector<int> producer(g.tensors.size(), -1);
  for (size_t i = 0; i < g.nodes.size(); ++i)
    for (TensorId o : g.nodes[i].outputs)
      if (o != kNoTensor) producer[o] = (int)i;
  // propagate liveness backward
  while (changed) {
    changed = false;
    for (auto& nd : g.nodes) {
      bool nodeLive = false;
      for (TensorId o : nd.outputs)
        if (o != kNoTensor && live.count(o)) nodeLive = true;
      if (!nodeLive) continue;
      for (TensorId in : nd.inputs)
        if (in != kNoTensor && !live.count(in)) {
          live.insert(in);
          changed = true;
        }
    }
  }
  std::vector<Node> kept;
  int removed = 0;
  for (auto& nd : g.nodes) {
    bool nodeLive = false;
    for (TensorId o : nd.outputs)
      if (o != kNoTensor && live.count(o)) nodeLive = true;
    if (nodeLive)
      kept.push_back(nd);
    else
      removed++;
  }
  if (removed) {
    g.nodes = std::move(kept);
    VX_INFO << "eliminateDeadNodes: removed " << removed << " node(s)";
  }
}

void runStandardPasses(Graph& g, int64_t batch) {
  inferShapes(g, batch);
  eliminateIdentity(g);
  foldBatchNorm(g);
  fuseActivations(g);
  constFold(g);
  eliminateDeadNodes(g);
  inferShapes(g, batch);  // refresh shapes after fusion/folding
}

}  // namespace vx
