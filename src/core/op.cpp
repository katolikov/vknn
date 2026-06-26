#include "vknn/op.h"

#include <unordered_map>

namespace vknn {

const char* opTypeName(OpType t) {
  switch (t) {
    case OpType::kConv:
      return "Conv";
    case OpType::kClip:
      return "Clip";
    case OpType::kRelu:
      return "Relu";
    case OpType::kAdd:
      return "Add";
    case OpType::kGlobalAvgPool:
      return "GlobalAveragePool";
    case OpType::kAvgPool:
      return "AveragePool";
    case OpType::kMaxPool:
      return "MaxPool";
    case OpType::kGemm:
      return "Gemm";
    case OpType::kMatMul:
      return "MatMul";
    case OpType::kEinsum:
      return "Einsum";
    case OpType::kReshape:
      return "Reshape";
    case OpType::kExpand:
      return "Expand";
    case OpType::kTile:
      return "Tile";
    case OpType::kSqueeze:
      return "Squeeze";
    case OpType::kFlatten:
      return "Flatten";
    case OpType::kSoftmax:
      return "Softmax";
    case OpType::kLayerNorm:
      return "LayerNormalization";
    case OpType::kBatchNorm:
      return "BatchNormalization";
    case OpType::kConcat:
      return "Concat";
    case OpType::kPad:
      return "Pad";
    case OpType::kIdentity:
      return "Identity";
    case OpType::kConstant:
      return "Constant";
    case OpType::kShape:
      return "Shape";
    case OpType::kGather:
      return "Gather";
    case OpType::kUnsqueeze:
      return "Unsqueeze";
    case OpType::kUnary:
      return "Unary";
    case OpType::kBinary:
      return "Binary";
    case OpType::kPRelu:
      return "PRelu";
    case OpType::kResize:
      return "Resize";
    case OpType::kGridSample:
      return "GridSample";
    case OpType::kTranspose:
      return "Transpose";
    case OpType::kSlice:
      return "Slice";
    case OpType::kReduce:
      return "Reduce";
    case OpType::kDepthToSpace:
      return "DepthToSpace";
    case OpType::kCast:
      return "Cast";
    case OpType::kSplit:
      return "Split";
    case OpType::kWhere:
      return "Where";
    case OpType::kEqual:
      return "Equal";
    case OpType::kConstantOfShape:
      return "ConstantOfShape";
    case OpType::kEyeLike:
      return "EyeLike";
    case OpType::kScatterND:
      return "ScatterND";
    case OpType::kFusedSE:
      return "FusedSE";
    case OpType::kFusedDwPw:
      return "FusedDwPw";
    case OpType::kConvertLayout:
      return "ConvertLayout";
    default:
      return "Unknown";
  }
}

UnaryType unaryFromOnnx(const std::string& s) {
  using U = UnaryType;
  static const std::unordered_map<std::string, UnaryType> m = {{"Sigmoid", U::kSigmoid},
                                                               {"Tanh", U::kTanh},
                                                               {"HardSwish", U::kHardSwish},
                                                               {"HardSigmoid", U::kHardSigmoid},
                                                               {"LeakyRelu", U::kLeakyRelu},
                                                               {"Elu", U::kElu},
                                                               {"Abs", U::kAbs},
                                                               {"Neg", U::kNeg},
                                                               {"Exp", U::kExp},
                                                               {"Log", U::kLog},
                                                               {"Sqrt", U::kSqrt},
                                                               {"Floor", U::kFloor},
                                                               {"Ceil", U::kCeil},
                                                               {"Erf", U::kErf},
                                                               {"Cos", U::kCos},
                                                               {"Sin", U::kSin},
                                                               {"Reciprocal", U::kReciprocal},
                                                               {"Softplus", U::kSoftplus}};
  auto it = m.find(s);
  return it == m.end() ? U::kInvalid : it->second;
}
ReduceType reduceFromOnnx(const std::string& s) {
  using R = ReduceType;
  if (s == "ReduceMean")
    return R::kMean;
  if (s == "ReduceSum")
    return R::kSum;
  if (s == "ReduceMax")
    return R::kMax;
  if (s == "ReduceMin")
    return R::kMin;
  if (s == "ReduceProd")
    return R::kProd;
  if (s == "ReduceL2")
    return R::kL2;
  return R::kInvalid;
}
BinaryType binaryFromOnnx(const std::string& s) {
  using B = BinaryType;
  static const std::unordered_map<std::string, BinaryType> m = {{"Mul", B::kMul}, {"Sub", B::kSub},
                                                                {"Div", B::kDiv}, {"Max", B::kMax},
                                                                {"Min", B::kMin}, {"Pow", B::kPow}};
  auto it = m.find(s);
  return it == m.end() ? B::kInvalid : it->second;
}

OpType opTypeFromOnnx(const std::string& s) {
  static const std::unordered_map<std::string, OpType> m = {
      {"Conv", OpType::kConv},
      {"Clip", OpType::kClip},
      {"Relu", OpType::kRelu},
      {"Add", OpType::kAdd},
      {"GlobalAveragePool", OpType::kGlobalAvgPool},
      // ReduceMean over the spatial dims (keepdims) is exactly a global average pool; that's how
      // it shows up in ResNet exports, so we route it to the same kernel.
      {"ReduceMean", OpType::kGlobalAvgPool},
      {"AveragePool", OpType::kAvgPool},
      {"MaxPool", OpType::kMaxPool},
      {"Gemm", OpType::kGemm},
      {"MatMul", OpType::kMatMul},
      {"Einsum", OpType::kEinsum},
      {"Reshape", OpType::kReshape},
      {"Expand", OpType::kExpand},
      {"Tile", OpType::kTile},
      {"Squeeze", OpType::kSqueeze},
      {"Flatten", OpType::kFlatten},
      {"Softmax", OpType::kSoftmax},
      {"LayerNormalization", OpType::kLayerNorm},
      {"BatchNormalization", OpType::kBatchNorm},
      {"Concat", OpType::kConcat},
      {"Pad", OpType::kPad},
      {"Identity", OpType::kIdentity},
      {"Constant", OpType::kConstant},
      {"Shape", OpType::kShape},
      {"Gather", OpType::kGather},
      {"Unsqueeze", OpType::kUnsqueeze},
      {"PRelu", OpType::kPRelu},
      {"Resize", OpType::kResize},
      {"Upsample", OpType::kResize},
      {"GridSample", OpType::kGridSample},
      {"Transpose", OpType::kTranspose},
      {"Slice", OpType::kSlice},
      {"DepthToSpace", OpType::kDepthToSpace},
      {"Cast", OpType::kCast},
      {"Split", OpType::kSplit},
      {"Where", OpType::kWhere},
      {"Equal", OpType::kEqual},
      {"ConstantOfShape", OpType::kConstantOfShape},
      {"EyeLike", OpType::kEyeLike},
      {"ScatterND", OpType::kScatterND},
  };
  auto it = m.find(s);
  if (it != m.end())
    return it->second;
  if (s == "ReduceSum" || s == "ReduceMax" || s == "ReduceMin" || s == "ReduceProd" ||
      s == "ReduceL2")
    return OpType::kReduce;
  if (unaryFromOnnx(s) != UnaryType::kInvalid)
    return OpType::kUnary;
  if (binaryFromOnnx(s) != BinaryType::kInvalid)
    return OpType::kBinary;
  return OpType::kUnknown;
}

}  // namespace vknn
