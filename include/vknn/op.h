// Op types, the fused-activation codes, the attribute bag, and the IR node struct.
#pragma once
#include <map>
#include <string>
#include <vector>

#include "vknn/common.h"

namespace vknn {

using TensorId = int32_t;
constexpr TensorId kNoTensor = -1;

/// Fused activation applied after an op (kept in sync with shaders/common.glsl).
enum class ActType : int32_t {
  kNone = 0,
  kRelu = 1,
  kRelu6 = 2,
  kClip = 3,
  kHardSwish = 4,
  kSiLU = 5
};

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
  kEinsum,  // einsum: outer-product (RoPE) + batched mat-vec/matmul (geometry tail)
  kReshape,
  kExpand,   // broadcast X to a target shape (numpy broadcasting), flat gather
  kTile,     // repeat X along each dim by `repeats`, flat gather
  kSqueeze,  // remove size-1 dims (metadata reshape / flat copy)
  kFlatten,
  kSoftmax,
  kLayerNorm,  // LayerNormalization: normalize over axes [axis..end], y=(x-mean)/sqrt(var+eps)*g+b
  kBatchNorm,
  kConcat,
  kPad,
  kIdentity,
  kConstant,
  kShape,
  kGather,
  kUnsqueeze,
  kUnary,            // elementwise unary family (Sigmoid/Tanh/HardSwish/...), see UnaryType
  kBinary,           // elementwise binary family (Mul/Sub/Div/Max/Min/Pow), see BinaryType
  kPRelu,            // y = x>0 ? x : slope*x, slope per-channel
  kResize,           // Resize/Upsample (nearest/linear), spatial
  kGridSample,       // sample input at grid coords (CPU)
  kTranspose,        // permute dims (CPU)
  kSlice,            // strided slice (CPU)
  kReduce,           // ReduceMean/Sum/Max/Min/Prod/L2, see ReduceType
  kDepthToSpace,     // [N,C,H,W] -> [N,C/b^2,H*b,W*b], DCR|CRD (flat index remap)
  kCast,             // dtype cast (CPU)
  kSplit,            // split along an axis into N outputs (CPU)
  kWhere,            // cond ? X : Y, elementwise with full broadcasting (flat path)
  kEqual,            // A == B -> 1.0/0.0, elementwise with broadcasting (flat path)
  kConstantOfShape,  // emit a tensor of the given shape filled with a scalar value
  kEyeLike,          // identity-like matrix (ones on a diagonal) matching the input shape
  kScatterND,        // copy data, then scatter update slices at N-D index rows
  kFusedSE,          // fused Squeeze-Excite scale: GAP->FC->relu->FC->hardsigmoid (one kernel)
  kFusedDwPw,        // fused depthwise-3x3 + 1x1-project (expanded intermediate stays on-chip)
  // layout conversion nodes (inserted by the layout pass)
  kConvertLayout,
};

// Sub-codes for the kUnary/kBinary/kReduce families, stored in Node::subOp. The integer values are
// kept in sync with the switch in shaders/common.glsl, so they are fixed and explicit. kInvalid
// (-1) marks "this ONNX op is not in the family".
enum class UnaryType : int32_t {
  kInvalid = -1,
  kSigmoid = 0,
  kTanh = 1,
  kHardSwish = 2,
  kHardSigmoid = 3,
  kLeakyRelu = 4,
  kElu = 5,
  kAbs = 6,
  kNeg = 7,
  kExp = 8,
  kLog = 9,
  kSqrt = 10,
  kFloor = 11,
  kCeil = 12,
  kRelu = 13,
  kSiLU = 14,
  kErf = 15,
  kCos = 16,
  kSin = 17,
  kReciprocal = 18,
  kSoftplus = 19  // transformer: GELU, RoPE, ...
};
enum class BinaryType : int32_t {
  kInvalid = -1,
  kMul = 0,
  kSub = 1,
  kDiv = 2,
  kMax = 3,
  kMin = 4,
  kPow = 5,
  kAdd = 6
};
enum class ReduceType : int32_t {
  kInvalid = -1,
  kMean = 0,
  kSum = 1,
  kMax = 2,
  kMin = 3,
  kProd = 4,
  kL2 = 5
};

const char* opTypeName(OpType t);
OpType opTypeFromOnnx(const std::string& s);
// The UnaryType/BinaryType/ReduceType for an ONNX op name, or k*::kInvalid if not in that family.
UnaryType unaryFromOnnx(const std::string& s);
BinaryType binaryFromOnnx(const std::string& s);
ReduceType reduceFromOnnx(const std::string& s);

/// A single attribute value (subset needed for CNNs).
struct Attr {
  enum Kind { kNone, kInt, kFloat, kInts, kFloats, kString } kind = kNone;
  int64_t i = 0;
  float f = 0;
  std::vector<int64_t> ints;
  std::vector<float> floats;
  std::string str;
  // For tensor-valued attributes (a Constant node's `value`): the tensor's dims, so the op can
  // emit the right shape instead of a flat 1-D vector (multi-dim constants like anchor grids).
  std::vector<int64_t> shape;
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

}  // namespace vknn
