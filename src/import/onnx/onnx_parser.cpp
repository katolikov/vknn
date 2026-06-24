// minimal, dependency-free ONNX (protobuf) importer.
//
// Parses the protobuf wire format directly (varint / length-delimited / fixed32/64),
// reading only the fields vxrt needs: GraphProto nodes, initializers (TensorProto),
// inputs/outputs (ValueInfoProto), and NodeProto attributes. Sufficient for MobileNetV2
// and CNNs in general. No protobuf library / no generated code required.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "vx/graph.h"
#include "vx/logging.h"

namespace vx {
namespace onnx {

// ----------------------------- wire reader -----------------------------
class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}
  bool eof() const { return p_ >= end_; }

  uint64_t varint() {
    uint64_t v = 0;
    int shift = 0;
    while (p_ < end_) {
      uint8_t b = *p_++;
      v |= (uint64_t)(b & 0x7F) << shift;
      if (!(b & 0x80)) break;
      shift += 7;
    }
    return v;
  }
  uint32_t fixed32() {
    uint32_t v = 0;
    std::memcpy(&v, p_, 4);
    p_ += 4;
    return v;
  }
  uint64_t fixed64() {
    uint64_t v = 0;
    std::memcpy(&v, p_, 8);
    p_ += 8;
    return v;
  }
  // returns (field number, wire type)
  bool tag(uint32_t& field, uint32_t& wire) {
    if (eof()) return false;
    uint64_t t = varint();
    field = (uint32_t)(t >> 3);
    wire = (uint32_t)(t & 7);
    return true;
  }
  // length-delimited region
  Reader sub() {
    uint64_t len = varint();
    Reader r(p_, (size_t)len);
    p_ += len;
    return r;
  }
  std::string str() {
    uint64_t len = varint();
    std::string s((const char*)p_, (size_t)len);
    p_ += len;
    return s;
  }
  std::vector<uint8_t> bytes() {
    uint64_t len = varint();
    std::vector<uint8_t> b(p_, p_ + len);
    p_ += len;
    return b;
  }
  // skip a field of given wire type
  void skip(uint32_t wire) {
    switch (wire) {
      case 0:
        varint();
        break;
      case 1:
        p_ += 8;
        break;
      case 5:
        p_ += 4;
        break;
      case 2: {
        uint64_t l = varint();
        p_ += l;
        break;
      }
      default:
        break;
    }
  }
  const uint8_t* cur() const { return p_; }
  const uint8_t* end() const { return end_; }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
};

// ----------------------------- TensorProto -----------------------------
// fields: 1=dims(int64 repeated/packed), 2=data_type(int32), 4=float_data(packed),
// 7=int64_data(packed), 8=name(string), 9=raw_data(bytes)
struct TensorProto {
  std::vector<int64_t> dims;
  int32_t dataType = 1;  // 1=FLOAT, 7=INT64
  std::string name;
  std::vector<uint8_t> raw;
  std::vector<float> floatData;
  std::vector<int64_t> int64Data;
};

static TensorProto parseTensor(Reader r) {
  TensorProto t;
  uint32_t f, w;
  while (r.tag(f, w)) {
    switch (f) {
      case 1:  // dims
        if (w == 2) {
          Reader s = r.sub();
          while (!s.eof()) t.dims.push_back((int64_t)s.varint());
        } else
          t.dims.push_back((int64_t)r.varint());
        break;
      case 2:
        t.dataType = (int32_t)r.varint();
        break;
      case 4:  // float_data (packed or single)
        if (w == 2) {
          Reader s = r.sub();
          while (!s.eof()) {
            uint32_t b = s.fixed32();
            float fv;
            std::memcpy(&fv, &b, 4);
            t.floatData.push_back(fv);
          }
        } else {
          uint32_t b = r.fixed32();
          float fv;
          std::memcpy(&fv, &b, 4);
          t.floatData.push_back(fv);
        }
        break;
      case 7:  // int64_data
        if (w == 2) {
          Reader s = r.sub();
          while (!s.eof()) t.int64Data.push_back((int64_t)s.varint());
        } else
          t.int64Data.push_back((int64_t)r.varint());
        break;
      case 8:
        t.name = r.str();
        break;
      case 9:
        t.raw = r.bytes();
        break;
      default:
        r.skip(w);
        break;
    }
  }
  return t;
}

// Materialize a TensorProto into a float32 HostBuffer (raw_data or typed data).
static void fillHostFloat(const TensorProto& t, HostBuffer& hb, int64_t elems) {
  hb.resizeElems(elems, DType::kFloat32);
  float* dst = hb.f32();
  if (!t.raw.empty()) {
    if (t.dataType == 1) {  // FLOAT raw
      std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), (size_t)elems * 4));
    } else if (t.dataType == 7) {  // INT64 raw
      const int64_t* s = reinterpret_cast<const int64_t*>(t.raw.data());
      for (int64_t i = 0; i < elems; ++i) dst[i] = (float)s[i];
    }
  } else if (!t.floatData.empty()) {
    for (int64_t i = 0; i < elems && i < (int64_t)t.floatData.size(); ++i) dst[i] = t.floatData[i];
  } else if (!t.int64Data.empty()) {
    for (int64_t i = 0; i < elems && i < (int64_t)t.int64Data.size(); ++i)
      dst[i] = (float)t.int64Data[i];
  }
}

// Materialize as int64 (for shape tensors).
static void fillHostI64(const TensorProto& t, HostBuffer& hb, int64_t elems) {
  hb.resizeElems(elems, DType::kInt64);
  int64_t* dst = hb.i64();
  if (!t.raw.empty() && t.dataType == 7) {
    std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), (size_t)elems * 8));
  } else if (!t.int64Data.empty()) {
    for (int64_t i = 0; i < elems && i < (int64_t)t.int64Data.size(); ++i) dst[i] = t.int64Data[i];
  }
}

// ----------------------------- AttributeProto -----------------------------
// fields: 1=name, 2=f(float32), 3=i(int64), 4=s(bytes), 7=floats(packed),
// 8=ints(packed), 5=t(TensorProto), 20=type(int32)
static void parseAttr(Reader r, Node& node) {
  std::string name;
  Attr a;
  uint32_t f, w;
  TensorProto tp;
  bool hasTp = false;
  while (r.tag(f, w)) {
    switch (f) {
      case 1:
        name = r.str();
        break;
      case 2: {
        uint32_t b = r.fixed32();
        std::memcpy(&a.f, &b, 4);
        a.kind = Attr::kFloat;
        break;
      }
      case 3:
        a.i = (int64_t)r.varint();
        a.kind = Attr::kInt;
        break;
      case 4: {
        auto b = r.bytes();
        a.str.assign((const char*)b.data(), b.size());
        a.kind = Attr::kString;
        break;
      }
      case 7: {
        Reader s = r.sub();
        while (!s.eof()) {
          uint32_t b = s.fixed32();
          float fv;
          std::memcpy(&fv, &b, 4);
          a.floats.push_back(fv);
        }
        a.kind = Attr::kFloats;
        break;
      }
      case 8: {
        if (w == 2) {
          Reader s = r.sub();
          while (!s.eof()) a.ints.push_back((int64_t)s.varint());
        } else
          a.ints.push_back((int64_t)r.varint());
        a.kind = Attr::kInts;
        break;
      }
      case 5: {
        tp = parseTensor(r.sub());
        hasTp = true;
        break;
      }
      default:
        r.skip(w);
        break;
    }
  }
  if (hasTp) {
    // store as a float-ints attribute when it's a small shape/scalar constant
    a.kind = Attr::kFloats;
    int64_t n = 1;
    for (auto d : tp.dims) n *= d;
    if (tp.dims.empty())
      n = std::max<int64_t>(1, (int64_t)std::max(tp.floatData.size(), tp.int64Data.size()));
    if (!tp.int64Data.empty() || tp.dataType == 7) {
      a.kind = Attr::kInts;
      if (!tp.int64Data.empty())
        a.ints = tp.int64Data;
      else if (!tp.raw.empty()) {
        const int64_t* s = (const int64_t*)tp.raw.data();
        for (int64_t i = 0; i < n; ++i) a.ints.push_back(s[i]);
      }
    } else {
      if (!tp.floatData.empty())
        a.floats = tp.floatData;
      else if (!tp.raw.empty()) {
        const float* s = (const float*)tp.raw.data();
        for (int64_t i = 0; i < n; ++i) a.floats.push_back(s[i]);
      }
    }
  }
  node.attr.map[name] = a;
}

// ----------------------------- ValueInfoProto -----------------------------
// field 1 = name, field 2 = type(TypeProto); TypeProto field1=tensor_type;
// Tensor field1=elem_type(int32), field2=shape(TensorShapeProto);
// TensorShapeProto field1=dim(repeated Dimension); Dimension field1=dim_value(int64).
static void parseValueInfo(Reader r, std::string& name, Shape& shape, int32_t& elem) {
  uint32_t f, w;
  while (r.tag(f, w)) {
    if (f == 1)
      name = r.str();
    else if (f == 2) {  // TypeProto
      Reader tp = r.sub();
      uint32_t f2, w2;
      while (tp.tag(f2, w2)) {
        if (f2 == 1) {  // tensor_type
          Reader tt = tp.sub();
          uint32_t f3, w3;
          while (tt.tag(f3, w3)) {
            if (f3 == 1)
              elem = (int32_t)tt.varint();
            else if (f3 == 2) {  // shape
              Reader sh = tt.sub();
              uint32_t f4, w4;
              while (sh.tag(f4, w4)) {
                if (f4 == 1) {  // dim
                  Reader dim = sh.sub();
                  uint32_t f5, w5;
                  int64_t val = -1;
                  while (dim.tag(f5, w5)) {
                    if (f5 == 1)
                      val = (int64_t)dim.varint();  // dim_value
                    else if (f5 == 2) {
                      dim.str();
                      val = -1;
                    }  // dim_param (dynamic)
                    else
                      dim.skip(w5);
                  }
                  shape.push_back(val);
                } else
                  sh.skip(w4);
              }
            } else
              tt.skip(w3);
          }
        } else
          tp.skip(w2);
      }
    } else
      r.skip(w);
  }
}

// ----------------------------- NodeProto -----------------------------
// fields: 1=input(string repeated), 2=output(string repeated), 3=name, 4=op_type, 5=attribute,
// 7=domain
static void parseNode(Reader r, Graph& g, Node& node) {
  uint32_t f, w;
  std::string opType;
  std::vector<std::string> ins, outs;
  while (r.tag(f, w)) {
    switch (f) {
      case 1:
        ins.push_back(r.str());
        break;
      case 2:
        outs.push_back(r.str());
        break;
      case 3:
        node.name = r.str();
        break;
      case 4:
        opType = r.str();
        break;
      case 5:
        parseAttr(r.sub(), node);
        break;
      default:
        r.skip(w);
        break;
    }
  }
  node.type = opTypeFromOnnx(opType);
  if (node.type == OpType::kUnary) {
    node.subOp = unaryFromOnnx(opType);
    // params (defaults per ONNX): LeakyRelu alpha=0.01, Elu alpha=1.0, HardSigmoid alpha,beta
    if (node.subOp == kULeakyRelu) node.actLo = node.attr.getf("alpha", 0.01f);
    else if (node.subOp == kUElu) node.actLo = node.attr.getf("alpha", 1.0f);
    else if (node.subOp == kUHardSigmoid) {
      node.actLo = node.attr.getf("alpha", 0.2f);
      node.actHi = node.attr.getf("beta", 0.5f);
    }
  } else if (node.type == OpType::kBinary) {
    node.subOp = binaryFromOnnx(opType);
  }
  if (node.type == OpType::kUnknown)
    VX_WARN << "unknown ONNX op '" << opType << "' (node " << node.name << ")";
  for (auto& s : ins) node.inputs.push_back(s.empty() ? kNoTensor : g.findOrAdd(s));
  for (auto& s : outs) node.outputs.push_back(g.findOrAdd(s));
  if (node.name.empty()) node.name = opType + "_" + std::to_string(g.nodes.size());
}

// ----------------------------- GraphProto -----------------------------
// fields: 1=node, 2=name, 5=initializer, 11=input, 12=output, 13=value_info
static void parseGraph(Reader r, Graph& g) {
  uint32_t f, w;
  struct PendingInit {
    std::string name;
    TensorProto tp;
  };
  std::vector<PendingInit> inits;
  std::vector<Node> nodes;
  while (r.tag(f, w)) {
    switch (f) {
      case 1: {
        Node n;
        parseNode(r.sub(), g, n);
        nodes.push_back(std::move(n));
        break;
      }
      case 5: {
        TensorProto tp = parseTensor(r.sub());
        std::string nm = tp.name;
        inits.push_back({nm, std::move(tp)});
        break;
      }
      case 11: {
        std::string nm;
        Shape sh;
        int32_t el = 1;
        parseValueInfo(r.sub(), nm, sh, el);
        TensorId id = g.findOrAdd(nm);
        g.desc(id).shape = sh;
        g.desc(id).isInput = true;
        g.inputs.push_back(id);
        break;
      }
      case 12: {
        std::string nm;
        Shape sh;
        int32_t el = 1;
        parseValueInfo(r.sub(), nm, sh, el);
        TensorId id = g.findOrAdd(nm);
        g.desc(id).shape = sh;
        g.desc(id).isOutput = true;
        g.outputs.push_back(id);
        break;
      }
      case 13: {
        std::string nm;
        Shape sh;
        int32_t el = 1;
        parseValueInfo(r.sub(), nm, sh, el);
        if (!nm.empty()) {
          TensorId id = g.findOrAdd(nm);
          if (g.desc(id).shape.empty()) g.desc(id).shape = sh;
        }
        break;
      }
      default:
        r.skip(w);
        break;
    }
  }
  g.nodes = std::move(nodes);

  // Materialize initializers.
  for (auto& pi : inits) {
    TensorId id = g.findOrAdd(pi.name);
    auto& d = g.desc(id);
    d.isInitializer = true;
    d.shape = pi.tp.dims;
    int64_t n = 1;
    for (auto x : pi.tp.dims) n *= x;
    if (pi.tp.dims.empty()) n = 1;
    HostBuffer hb;
    if (pi.tp.dataType == 7) {
      d.dtype = DType::kInt64;
      fillHostI64(pi.tp, hb, n);
    } else {
      d.dtype = DType::kFloat32;
      fillHostFloat(pi.tp, hb, n);
    }
    g.initializers[id] = std::move(hb);
  }
  // Remove initializer ids from the graph input list (ONNX lists them in both).
  std::vector<TensorId> realInputs;
  for (TensorId id : g.inputs)
    if (!g.isInitializer(id)) realInputs.push_back(id);
  g.inputs = realInputs;
}

}  // namespace onnx

// Public entry point.
Graph importOnnx(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error(Status::kIoError, "cannot open ONNX file: " + path);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  Graph g;
  // ModelProto: field 7 = graph (GraphProto). Skip everything else.
  onnx::Reader r(buf.data(), buf.size());
  uint32_t fld, wire;
  bool foundGraph = false;
  while (r.tag(fld, wire)) {
    if (fld == 7 && wire == 2) {
      onnx::parseGraph(r.sub(), g);
      foundGraph = true;
    } else
      r.skip(wire);
  }
  if (!foundGraph) throw Error(Status::kInvalidArgument, "no GraphProto in ONNX model");
  g.topoSort();
  VX_INFO << "Imported ONNX: " << g.nodes.size() << " nodes, " << g.initializers.size()
          << " initializers, " << g.inputs.size() << " inputs, " << g.outputs.size() << " outputs";
  return g;
}

}  // namespace vx
