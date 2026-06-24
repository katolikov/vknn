#include "vx/op.h"
#include <unordered_map>

namespace vx {

const char* opTypeName(OpType t) {
  switch (t) {
    case OpType::kConv: return "Conv";
    case OpType::kClip: return "Clip";
    case OpType::kRelu: return "Relu";
    case OpType::kAdd: return "Add";
    case OpType::kGlobalAvgPool: return "GlobalAveragePool";
    case OpType::kAvgPool: return "AveragePool";
    case OpType::kMaxPool: return "MaxPool";
    case OpType::kGemm: return "Gemm";
    case OpType::kMatMul: return "MatMul";
    case OpType::kReshape: return "Reshape";
    case OpType::kFlatten: return "Flatten";
    case OpType::kSoftmax: return "Softmax";
    case OpType::kBatchNorm: return "BatchNormalization";
    case OpType::kConcat: return "Concat";
    case OpType::kPad: return "Pad";
    case OpType::kIdentity: return "Identity";
    case OpType::kConstant: return "Constant";
    case OpType::kShape: return "Shape";
    case OpType::kGather: return "Gather";
    case OpType::kUnsqueeze: return "Unsqueeze";
    case OpType::kConvertLayout: return "ConvertLayout";
    default: return "Unknown";
  }
}

OpType opTypeFromOnnx(const std::string& s) {
  static const std::unordered_map<std::string, OpType> m = {
      {"Conv", OpType::kConv},
      {"Clip", OpType::kClip},
      {"Relu", OpType::kRelu},
      {"Add", OpType::kAdd},
      {"GlobalAveragePool", OpType::kGlobalAvgPool},
      {"AveragePool", OpType::kAvgPool},
      {"MaxPool", OpType::kMaxPool},
      {"Gemm", OpType::kGemm},
      {"MatMul", OpType::kMatMul},
      {"Reshape", OpType::kReshape},
      {"Flatten", OpType::kFlatten},
      {"Softmax", OpType::kSoftmax},
      {"BatchNormalization", OpType::kBatchNorm},
      {"Concat", OpType::kConcat},
      {"Pad", OpType::kPad},
      {"Identity", OpType::kIdentity},
      {"Constant", OpType::kConstant},
      {"Shape", OpType::kShape},
      {"Gather", OpType::kGather},
      {"Unsqueeze", OpType::kUnsqueeze},
  };
  auto it = m.find(s);
  return it == m.end() ? OpType::kUnknown : it->second;
}

}  // namespace vx
