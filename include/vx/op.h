// vxrt — operator types, fused-activation codes, attributes, and IR node.
#pragma once
#include <map>
#include <string>
#include <vector>
#include "vx/common.h"

namespace vx {

using TensorId = int32_t;
constexpr TensorId kNoTensor = -1;

/// Fused activation applied after an op (kept in sync with shaders/common.glsl).
enum class ActType : int32_t { kNone = 0, kRelu = 1, kRelu6 = 2, kClip = 3 };

/// Operator types. Add a new value here + a name mapping + register kernels.
enum class OpType {
  kUnknown = 0,
  kConv,  // Conv2D (incl. depthwise via group, pointwise 1x1)
  kClip,  // Clip / Relu6
  kRelu,
  kAdd,  // elementwise add (residual)
  kGlobalAvgPool,
  kAvgPool,
  kMaxPool,
  kGemm,
  kMatMul,
  kReshape,
  kFlatten,
  kSoftmax,
  kBatchNorm,
  kConcat,
  kPad,
  kIdentity,
  kConstant,
  kShape,
  kGather,
  kUnsqueeze,
  // layout conversion nodes (inserted by the layout pass)
  kConvertLayout,
};

const char* opTypeName(OpType t);
OpType opTypeFromOnnx(const std::string& s);

/// A single attribute value (subset needed for CNNs).
struct Attr {
  enum Kind { kNone, kInt, kFloat, kInts, kFloats, kString } kind = kNone;
  int64_t i = 0;
  float f = 0;
  std::vector<int64_t> ints;
  std::vector<float> floats;
  std::string str;
};

struct Attributes {
  std::map<std::string, Attr> map;
  bool has(const std::string& k) const { return map.count(k) > 0; }
  int64_t geti(const std::string& k, int64_t d = 0) const {
    auto it = map.find(k);
    return it == map.end() ? d : it->second.i;
  }
  float getf(const std::string& k, float d = 0) const {
    auto it = map.find(k);
    return it == map.end() ? d : it->second.f;
  }
  const std::vector<int64_t>& getints(const std::string& k) const {
    static const std::vector<int64_t> e;
    auto it = map.find(k);
    return it == map.end() ? e : it->second.ints;
  }
  std::string gets(const std::string& k, const std::string& d = "") const {
    auto it = map.find(k);
    return it == map.end() ? d : it->second.str;
  }
};

/// IR node. References tensors by id into Graph::tensors.
struct Node {
  OpType type = OpType::kUnknown;
  std::string name;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  Attributes attr;
  // Fusion metadata filled by graph passes:
  ActType fusedAct = ActType::kNone;
  float actLo = 0, actHi = 0;
};

}  // namespace vx
