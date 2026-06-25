#include "passes.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include "backends/cpu/cpu_backend.h"
#include "vx/logging.h"

namespace vx {

// Read an int64 list parameter from either a node attribute (older opsets) or an initializer input
// (opset 10+/13+ moved Slice/Pad/Reduce params to inputs). Returns empty if neither is present.
static std::vector<int64_t> readI64Param(const Graph& g, const Node& nd, const char* attrName,
                                         int inputIdx) {
  const auto& av = nd.attr.getints(attrName);
  if (!av.empty()) return av;
  if (inputIdx >= 0 && inputIdx < (int)nd.inputs.size() && nd.inputs[inputIdx] != kNoTensor) {
    auto it = g.initializers.find(nd.inputs[inputIdx]);
    if (it != g.initializers.end()) {
      const HostBuffer& hb = it->second;
      if (g.tensors[nd.inputs[inputIdx]].dtype == DType::kInt64) {
        int64_t n = (int64_t)hb.bytes.size() / 8;
        return std::vector<int64_t>(hb.i64(), hb.i64() + n);
      }
      int64_t n = (int64_t)hb.bytes.size() / 4;
      std::vector<int64_t> out;
      const float* f = hb.f32();
      for (int64_t i = 0; i < n; ++i) out.push_back((int64_t)f[i]);
      return out;
    }
  }
  return {};
}

// Redirect every reference to tensor `from` so it points at `to`: node inputs, the fused-residual
// edge (NOT in the inputs list on every op), and graph outputs. Fusion passes that delete a node and
// fold its output into a producer MUST use this — rewiring only node.inputs leaves a stale
// fusedResidual edge dangling at a dead tensor (the cause of a hard crash in conv residual reads).
static void rewireTensor(Graph& g, TensorId from, TensorId to) {
  if (from == to || from == kNoTensor) return;
  for (auto& nn : g.nodes) {
    for (TensorId& in : nn.inputs)
      if (in == from) in = to;
    if (nn.fusedResidual == from) nn.fusedResidual = to;
  }
  for (TensorId& go : g.outputs)
    if (go == from) go = to;
}

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
      case OpType::kPRelu:
        SH(o) = SH(nd.inputs[0]);
        break;
      case OpType::kGridSample: {
        const Shape& xs = SH(nd.inputs[0]);
        const Shape& gs = SH(nd.inputs[1]);
        if (xs.size() == 4 && gs.size() == 4) SH(o) = {xs[0], xs[1], gs[1], gs[2]};
        break;
      }
      case OpType::kCast: {
        SH(o) = SH(nd.inputs[0]);
        break;
      }
      case OpType::kFusedSE: {
        NCHW x = NCHW::from(SH(nd.inputs[0]));
        SH(o) = {x.n, x.c, 1, 1};  // channel scale
        break;
      }
      case OpType::kFusedDwPw: {
        NCHW x = NCHW::from(SH(nd.inputs[0]));  // expanded input [N,E,H,W]
        const Shape& pw = SH(nd.inputs[3]);     // project weights [Cout,E,1,1]
        auto a = [&](const char* k, std::vector<int64_t> d) {
          const auto& v = nd.attr.getints(k); return v.empty() ? d : v;
        };
        auto k = a("kernel_shape", {3, 3}), st = a("strides", {1, 1});
        auto pad = a("pads", {0, 0, 0, 0}), dil = a("dilations", {1, 1});
        int64_t oh = (x.h + pad[0] + pad[2] - (dil[0] * (k[0] - 1) + 1)) / st[0] + 1;
        int64_t ow = (x.w + pad[1] + pad[3] - (dil[1] * (k[1] - 1) + 1)) / st[1] + 1;
        SH(o) = {x.n, pw.empty() ? x.c : pw[0], oh, ow};
        break;
      }
      case OpType::kSplit: {
        const Shape& a = SH(nd.inputs[0]);
        if (a.empty()) break;
        int64_t rank = (int64_t)a.size();
        int64_t axis = nd.attr.geti("axis", 0);
        if (axis < 0) axis += rank;
        std::vector<int64_t> sp = readI64Param(g, nd, "split", 1);
        int64_t nout = (int64_t)nd.outputs.size();
        if (sp.empty() && nout > 0) { int64_t each = a[axis] / nout; for (int64_t k = 0; k < nout; ++k) sp.push_back(each); }
        for (int64_t k = 0; k < nout && k < (int64_t)sp.size(); ++k) {
          if (nd.outputs[k] == kNoTensor) continue;
          Shape os = a; os[axis] = sp[k]; SH(nd.outputs[k]) = os;
        }
        break;
      }
      case OpType::kTranspose: {
        const Shape& a = SH(nd.inputs[0]);
        const auto& perm = nd.attr.getints("perm");
        Shape out(a.size());
        for (size_t i = 0; i < a.size(); ++i)
          out[i] = perm.empty() ? a[a.size() - 1 - i] : a[perm[i]];
        SH(o) = out;
        break;
      }
      case OpType::kReduce: {
        const Shape& a = SH(nd.inputs[0]);
        int64_t rank = (int64_t)a.size();
        std::vector<int64_t> axes = readI64Param(g, nd, "axes", 1);
        if (axes.empty())
          for (int64_t i = 0; i < rank; ++i) axes.push_back(i);  // reduce all
        std::set<int64_t> ax;
        for (int64_t v : axes) ax.insert(v < 0 ? v + rank : v);
        bool keep = nd.attr.geti("keepdims", 1) != 0;
        Shape out;
        for (int64_t i = 0; i < rank; ++i) {
          if (ax.count(i)) { if (keep) out.push_back(1); }
          else out.push_back(a[i]);
        }
        if (out.empty()) out.push_back(1);
        SH(o) = out;
        break;
      }
      case OpType::kPad: {
        const Shape& a = SH(nd.inputs[0]);
        int64_t rank = (int64_t)a.size();
        std::vector<int64_t> pads = readI64Param(g, nd, "pads", 1);
        Shape out = a;
        if ((int64_t)pads.size() >= 2 * rank)
          for (int64_t i = 0; i < rank; ++i) out[i] = a[i] + pads[i] + pads[i + rank];
        SH(o) = out;
        break;
      }
      case OpType::kSlice: {
        const Shape& a = SH(nd.inputs[0]);
        int64_t rank = (int64_t)a.size();
        std::vector<int64_t> starts = readI64Param(g, nd, "starts", 1);
        std::vector<int64_t> ends = readI64Param(g, nd, "ends", 2);
        std::vector<int64_t> axes = readI64Param(g, nd, "axes", 3);
        std::vector<int64_t> steps = readI64Param(g, nd, "steps", 4);
        Shape out = a;
        // starts/ends/axes/steps come from initializer inputs that may be only partially
        // const-foldable (the YOLO DFL head leaves some runtime); only bound a dim when BOTH
        // its start and end are known, and never index a param vector past its length.
        for (size_t k = 0; k < starts.size() && k < ends.size(); ++k) {
          int64_t ax = axes.empty() ? (int64_t)k : (k < axes.size() ? axes[k] : (int64_t)k);
          if (ax < 0) ax += rank;
          if (ax < 0 || ax >= rank) continue;
          int64_t step = (k < steps.size()) ? steps[k] : 1;
          int64_t dim = a[ax];
          int64_t st = starts[k] < 0 ? starts[k] + dim : starts[k];
          int64_t en = ends[k] < 0 ? ends[k] + dim : ends[k];
          st = std::max<int64_t>(0, std::min(st, dim));
          en = std::max<int64_t>(0, std::min(en, dim));
          int64_t n = step > 0 ? (en - st + step - 1) / step : 0;
          out[ax] = std::max<int64_t>(0, n);
        }
        SH(o) = out;
        break;
      }
      case OpType::kResize: {
        // output = round(input * scales) or explicit sizes; scales/sizes are initializer inputs.
        Shape s = SH(nd.inputs[0]);
        if (s.size() == 4) {
          // ONNX Resize inputs: X, roi, scales, sizes (some optional/empty). Prefer sizes if given.
          auto getInit = [&](int idx, std::vector<float>& f, std::vector<int64_t>& i64) {
            if (idx >= (int)nd.inputs.size() || nd.inputs[idx] == kNoTensor) return false;
            auto it = g.initializers.find(nd.inputs[idx]);
            if (it == g.initializers.end()) return false;
            int64_t n = (int64_t)it->second.bytes.size() /
                        (g.tensors[nd.inputs[idx]].dtype == DType::kInt64 ? 8 : 4);
            if (g.tensors[nd.inputs[idx]].dtype == DType::kInt64) {
              const int64_t* p = it->second.i64();
              for (int64_t k = 0; k < n; ++k) i64.push_back(p[k]);
            } else {
              const float* p = it->second.f32();
              for (int64_t k = 0; k < n; ++k) f.push_back(p[k]);
            }
            return true;
          };
          std::vector<float> sizesF, scalesF;
          std::vector<int64_t> sizesI, scalesI;
          if (nd.inputs.size() >= 4 && getInit(3, sizesF, sizesI) && (sizesI.size() == 4 || sizesF.size() == 4)) {
            for (int k = 0; k < 4; ++k) s[k] = sizesI.size() == 4 ? sizesI[k] : (int64_t)sizesF[k];
          } else if (getInit(2, scalesF, scalesI) && scalesF.size() == 4) {
            for (int k = 0; k < 4; ++k) s[k] = (int64_t)(SH(nd.inputs[0])[k] * scalesF[k]);
          }
        }
        SH(o) = s;
        break;
      }
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
      case OpType::kConcat: {
        Shape s = SH(nd.inputs[0]);
        if (!s.empty()) {
          int64_t axis = nd.attr.geti("axis", 1);
          if (axis < 0) axis += (int64_t)s.size();
          int64_t sum = 0;
          for (TensorId in : nd.inputs) {
            const Shape& si = SH(in);
            if (si.empty()) { sum = -1; break; }
            sum += si[axis];
          }
          if (sum >= 0) { s[axis] = sum; SH(o) = s; }
        }
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

int constFold(Graph& g) {
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
      // Any op whose every input is a known constant can be evaluated now. This collapses the
      // shape-arithmetic that detection heads (YOLO) build at runtime — Shape/Gather feeding scalar
      // Binary/Add to derive per-level strides — into plain constants, so those ops never need a
      // backend at all (neither CPU nor GPU).
      case OpType::kGather:
      case OpType::kUnsqueeze:
      case OpType::kConcat:
      case OpType::kBinary:
      case OpType::kAdd:
      case OpType::kReshape:
      case OpType::kSlice:
      case OpType::kTranspose:
      case OpType::kCast:
      case OpType::kReduce: {
        if (nd.inputs.empty()) return false;
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
  return (int)removeNodes.size();
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

void fuseResidualAdd(Graph& g) {
  // Fuse  out = Add(pointwise_conv(x), residual)  into the conv's epilogue, so the conv writes
  // conv+residual directly (saves the Add's full read+write). Only for 1x1 stride-1 pad-0 group-1
  // convs (the ones our conv1x1/split-K kernels run and support a residual on).
  std::vector<int> producer(g.tensors.size(), -1);
  for (size_t i = 0; i < g.nodes.size(); ++i)
    for (TensorId o : g.nodes[i].outputs)
      if (o != kNoTensor) producer[o] = (int)i;
  auto convEligible = [&](const Node& c) {
    if (c.type != OpType::kConv || c.fusedResidual != kNoTensor) return false;
    auto ints = [&](const char* k, std::vector<int64_t> d) {
      const auto& v = c.attr.getints(k); return v.empty() ? d : v;
    };
    auto k = ints("kernel_shape", {1, 1}), s = ints("strides", {1, 1});
    auto p = ints("pads", {0, 0, 0, 0});
    return c.attr.geti("group", 1) == 1 && k[0] == 1 && k[1] == 1 && s[0] == 1 && s[1] == 1 &&
           p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 0;
  };
  std::set<int> remove;
  int fused = 0;
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    Node& add = g.nodes[i];
    if (add.type != OpType::kAdd || add.inputs.size() != 2) continue;
    int p0 = producer[add.inputs[0]], p1 = producer[add.inputs[1]];
    int ci = -1;
    TensorId residual = kNoTensor;
    if (p0 >= 0 && convEligible(g.nodes[p0])) { ci = p0; residual = add.inputs[1]; }
    else if (p1 >= 0 && convEligible(g.nodes[p1])) { ci = p1; residual = add.inputs[0]; }
    if (ci < 0) continue;
    // the conv output must feed only this Add
    int consumers = 0;
    for (auto& nn : g.nodes)
      for (TensorId in : nn.inputs)
        if (in == g.nodes[ci].outputs[0]) consumers++;
    for (TensorId go : g.outputs)
      if (go == g.nodes[ci].outputs[0]) consumers++;
    if (consumers != 1) continue;
    Node& conv = g.nodes[ci];
    conv.fusedResidual = residual;
    conv.inputs.push_back(residual);  // keep it live for DCE / buffer allocation / scheduling
    if (conv.fusedAct == ActType::kNone) {  // carry any activation that was folded into the Add
      conv.fusedAct = add.fusedAct;
      conv.actLo = add.actLo;
      conv.actHi = add.actHi;
    }
    TensorId addOut = add.outputs[0], convOut = conv.outputs[0];
    for (auto& nn : g.nodes)
      for (TensorId& in : nn.inputs)
        if (in == addOut) in = convOut;
    for (TensorId& go : g.outputs)
      if (go == addOut) go = convOut;
    remove.insert((int)i);
    fused++;
  }
  if (fused) {
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); ++i)
      if (!remove.count((int)i)) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    VX_INFO << "fuseResidualAdd: fused " << fused << " residual Add(s) into Conv";
  }
}

void fuseSqueezeExcite(Graph& g) {
  // Collapse the Squeeze-Excite scale chain GlobalAvgPool -> Conv1x1(+relu) -> Conv1x1 ->
  // HardSigmoid into ONE kFusedSE node that emits the channel scale. The following channel-broadcast
  // Mul (scale * feature) is left intact. MobileNetV3 has ~11 of these tiny multi-dispatch chains.
  std::vector<int> producer(g.tensors.size(), -1);
  std::vector<int> consumers(g.tensors.size(), 0);
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    for (TensorId o : g.nodes[i].outputs)
      if (o != kNoTensor) producer[o] = (int)i;
    for (TensorId in : g.nodes[i].inputs)
      if (in != kNoTensor) consumers[in]++;
  }
  auto single = [&](TensorId t) { return t != kNoTensor && consumers[t] == 1; };
  std::set<int> remove;
  int fused = 0;
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    Node& hs = g.nodes[i];
    if (hs.type != OpType::kUnary || hs.subOp != kUHardSigmoid) continue;
    int p2 = producer[hs.inputs[0]];
    if (p2 < 0 || g.nodes[p2].type != OpType::kConv || !single(g.nodes[p2].outputs[0])) continue;
    Node& conv2 = g.nodes[p2];
    int p1 = producer[conv2.inputs[0]];
    if (p1 < 0 || g.nodes[p1].type != OpType::kConv || !single(g.nodes[p1].outputs[0])) continue;
    Node& conv1 = g.nodes[p1];
    if (conv1.fusedAct != ActType::kRelu) continue;
    // Require a GlobalAvgPool feeding conv1, but KEEP it (its reduction is parallel). We fuse only
    // the tiny [N,C,1,1] middle: conv1(relu)->conv2->hardsigmoid -> one kernel that reads the pooled
    // avg directly. Fusing the pool into one workgroup regressed; this keeps GAP + Mul parallel.
    int pg = producer[conv1.inputs[0]];
    if (pg < 0 || g.nodes[pg].type != OpType::kGlobalAvgPool) continue;
    TensorId avg = conv1.inputs[0];      // the pooled [N,C,1,1] tensor
    auto bias = [](const Node& c) { return c.inputs.size() > 2 ? c.inputs[2] : kNoTensor; };
    Node se;
    se.type = OpType::kFusedSE;
    se.name = conv1.name + "#se";
    se.inputs = {avg, conv1.inputs[1], bias(conv1), conv2.inputs[1], bias(conv2)};
    se.outputs = {hs.outputs[0]};        // the scale tensor (Mul still consumes it)
    se.actLo = hs.actLo;                  // hardsigmoid alpha/beta
    se.actHi = hs.actHi;
    g.nodes[i] = se;                      // replace hardsigmoid node with the fused node
    remove.insert(p1);                    // remove conv1 + conv2 (gap stays)
    remove.insert(p2);
    fused++;
  }
  if (fused) {
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); ++i)
      if (!remove.count((int)i)) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    VX_INFO << "fuseSqueezeExcite: fused " << fused << " SE chain(s)";
  }
}

void fuseSwish(Graph& g) {
  // Fuse the elementwise self-gating activation Mul(x, HardSigmoid(x)) = HardSwish and
  // Mul(x, Sigmoid(x)) = SiLU/Swish. If x is produced by a Conv/Gemm (consumed only by the gate +
  // the Mul), fold it into that op's epilogue activation (removes 2 dispatches + the intermediate);
  // otherwise collapse the [gate, Mul] pair into a single unary op. MobileNetV3 has ~21 of these.
  std::vector<int> producer(g.tensors.size(), -1);
  std::vector<int> consumers(g.tensors.size(), 0);
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    for (TensorId o : g.nodes[i].outputs) if (o != kNoTensor) producer[o] = (int)i;
    for (TensorId in : g.nodes[i].inputs) if (in != kNoTensor) consumers[in]++;
  }
  std::set<int> remove;
  int fusedC = 0, fusedU = 0;
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    Node& M = g.nodes[i];
    if (M.type != OpType::kBinary || M.subOp != kBMul || M.inputs.size() != 2) continue;
    int sigIdx = -1;
    TensorId x = kNoTensor;
    for (int e = 0; e < 2; ++e) {
      int pc = producer[M.inputs[e]];
      if (pc < 0 || g.nodes[pc].type != OpType::kUnary) continue;
      int su = g.nodes[pc].subOp;
      if ((su == kUHardSigmoid || su == kUSigmoid) && g.nodes[pc].inputs[0] == M.inputs[1 - e]) {
        sigIdx = pc; x = M.inputs[1 - e]; break;
      }
    }
    if (sigIdx < 0) continue;
    if (consumers[g.nodes[sigIdx].outputs[0]] != 1) continue;  // gate feeds only this Mul
    ActType act = g.nodes[sigIdx].subOp == kUHardSigmoid ? ActType::kHardSwish : ActType::kSiLU;
    int px = producer[x];
    bool fuseConv = px >= 0 && (g.nodes[px].type == OpType::kConv || g.nodes[px].type == OpType::kGemm) &&
                    g.nodes[px].fusedAct == ActType::kNone && consumers[x] == 2;
    if (fuseConv) {
      g.nodes[px].fusedAct = act;
      // Fold the Mul output onto the conv output. Must also patch any fusedResidual edge that
      // pointed at the Mul output (a residual already folded into a later conv by fuseResidualAdd),
      // else that conv reads a dead tensor -> crash.
      rewireTensor(g, M.outputs[0], x);
      remove.insert((int)i);
      remove.insert(sigIdx);
      fusedC++;
    } else {
      // collapse [gate, Mul] -> one unary
      M.type = OpType::kUnary;
      M.subOp = (act == ActType::kHardSwish) ? kUHardSwish : kUSiLU;
      M.inputs = {x};
      remove.insert(sigIdx);
      fusedU++;
    }
  }
  if (!remove.empty()) {
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); ++i) if (!remove.count((int)i)) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    VX_INFO << "fuseSwish: fused " << fusedC << " into conv, collapsed " << fusedU << " to unary";
  }
}

void fuseDwPw(Graph& g) {
  // Fuse depthwise-3x3 conv (D) -> 1x1 project conv (P) into one kFusedDwPw node, so the expanded
  // intermediate (D's output, the block's largest activation) never hits global memory and a
  // dispatch+barrier is removed. Only when D's output feeds ONLY P, D's activation is a plain
  // ActType (Relu/Relu6/Clip/None), and P is a stride-1 pad-0 group-1 pointwise conv.
  std::vector<int> producer(g.tensors.size(), -1);
  std::vector<int> consumers(g.tensors.size(), 0);
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    for (TensorId o : g.nodes[i].outputs) if (o != kNoTensor) producer[o] = (int)i;
    for (TensorId in : g.nodes[i].inputs) if (in != kNoTensor) consumers[in]++;
  }
  auto ints = [](const Node& n, const char* k, std::vector<int64_t> d) {
    const auto& v = n.attr.getints(k); return v.empty() ? d : v;
  };
  std::set<int> remove;
  int fused = 0;
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    Node& P = g.nodes[i];
    if (P.type != OpType::kConv) continue;
    const Shape& pw = g.desc(P.inputs[1]).shape;  // [Cout, Cin, 1, 1]
    if (pw.size() != 4 || pw[2] != 1 || pw[3] != 1) continue;
    if (P.attr.geti("group", 1) != 1) continue;
    auto ps = ints(P, "strides", {1, 1}), pp = ints(P, "pads", {0, 0, 0, 0});
    if (ps[0] != 1 || ps[1] != 1 || pp[0] || pp[1] || pp[2] || pp[3]) continue;
    int di = producer[P.inputs[0]];
    if (di < 0 || g.nodes[di].type != OpType::kConv) continue;
    Node& D = g.nodes[di];
    if (consumers[D.outputs[0]] != 1) continue;  // D feeds only P
    const Shape& dw = g.desc(D.inputs[1]).shape;  // [C,1,KH,KW]
    NCHW dx = NCHW::from(g.desc(D.inputs[0]).shape);
    bool depthwise = (D.attr.geti("group", 1) == dx.c && dw.size() == 4 && dw[1] == 1);
    if (!depthwise) continue;
    // D's activation must be parameterless (None/Relu/Relu6); skip custom Clip and hardswish-dw.
    if (D.fusedAct != ActType::kNone && D.fusedAct != ActType::kRelu &&
        D.fusedAct != ActType::kRelu6)
      continue;
    auto bias = [](const Node& c) { return c.inputs.size() > 2 ? c.inputs[2] : kNoTensor; };
    Node f;
    f.type = OpType::kFusedDwPw;
    f.name = D.name + "#dwpw";
    f.inputs = {D.inputs[0], D.inputs[1], bias(D), P.inputs[1], bias(P)};
    if (P.fusedResidual != kNoTensor) { f.inputs.push_back(P.fusedResidual); f.fusedResidual = P.fusedResidual; }
    f.outputs = {P.outputs[0]};
    f.attr = D.attr;                 // carry dw strides/pads/kernel for shape inference + kernel
    f.subOp = (int32_t)D.fusedAct;   // dw activation
    f.fusedAct = P.fusedAct;         // project activation
    f.actLo = P.actLo; f.actHi = P.actHi;
    g.nodes[i] = f;                  // replace P with the fused node
    remove.insert(di);              // remove D
    fused++;
  }
  if (fused) {
    std::vector<Node> kept;
    for (size_t i = 0; i < g.nodes.size(); ++i) if (!remove.count((int)i)) kept.push_back(g.nodes[i]);
    g.nodes = std::move(kept);
    VX_INFO << "fuseDwPw: fused " << fused << " depthwise+project pair(s)";
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
  fuseResidualAdd(g);
  if (!std::getenv("VXRT_NO_FUSE_SWISH")) fuseSwish(g);  // HardSwish/SiLU into conv epilogue (default on)
  if (std::getenv("VXRT_FUSE_SE")) fuseSqueezeExcite(g);
  if (std::getenv("VXRT_FUSE_DWPW")) fuseDwPw(g);
  // Iterate fold+infer: folding a Shape/Gather/Concat chain turns a dynamic Reshape's shape input
  // into a constant, which lets the next inferShapes resolve that Reshape statically, which in turn
  // exposes more foldable shape ops downstream (YOLO's DFL/box-decode head). Converges in a couple
  // rounds; the loop just runs until constFold stops removing nodes.
  for (int iter = 0; iter < 8; ++iter) {
    if (constFold(g) == 0) break;
    inferShapes(g, batch);
  }
  eliminateDeadNodes(g);
  inferShapes(g, batch);  // refresh shapes after fusion/folding
}

}  // namespace vx
