// Op types, the fused-activation codes, the attribute bag, and the IR node struct.
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
  kUnary,   // elementwise unary family (Sigmoid/Tanh/HardSwish/...), see UnaryType
  kBinary,  // elementwise binary family (Mul/Sub/Div/Max/Min/Pow), see BinaryType
  kPRelu,   // y = x>0 ? x : slope*x, slope per-channel
  kResize,  // Resize/Upsample (nearest/linear), spatial
  kGridSample,  // sample input at grid coords (CPU)
  kTranspose,   // permute dims (CPU)
  kSlice,       // strided slice (CPU)
  kReduce,      // ReduceMean/Sum/Max/Min/Prod, see ReduceType
  kCast,        // dtype cast (CPU)
  kSplit,       // split along an axis into N outputs (CPU)
  // layout conversion nodes (inserted by the layout pass)
  kConvertLayout,
};

// Sub-codes for the kUnary/kBinary families (kept in sync with shaders/common.glsl).
enum UnaryType {
  kUSigmoid = 0, kUTanh = 1, kUHardSwish = 2, kUHardSigmoid = 3, kULeakyRelu = 4, kUElu = 5,
  kUAbs = 6, kUNeg = 7, kUExp = 8, kULog = 9, kUSqrt = 10, kUFloor = 11, kUCeil = 12, kURelu = 13
};
enum BinaryType { kBMul = 0, kBSub = 1, kBDiv = 2, kBMax = 3, kBMin = 4, kBPow = 5, kBAdd = 6 };
enum ReduceType { kRMean = 0, kRSum = 1, kRMax = 2, kRMin = 3, kRProd = 4 };

const char* opTypeName(OpType t);
OpType opTypeFromOnnx(const std::string& s);
// Returns the UnaryType/BinaryType code for an ONNX op name, or -1 if not in that family.
int unaryFromOnnx(const std::string& s);
int binaryFromOnnx(const std::string& s);
int reduceFromOnnx(const std::string& s);

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
  // For kUnary/kBinary: the UnaryType/BinaryType code. For unary ops with params (LeakyRelu/Elu
  // alpha, HardSigmoid alpha/beta) the params live in actLo/actHi.
  int32_t subOp = 0;
  // Conv only: a residual tensor fused into the epilogue (out = act(conv + residual)); set by the
  // residual-Add fusion pass. kNoTensor when not fused.
  TensorId fusedResidual = kNoTensor;
};

}  // namespace vx
