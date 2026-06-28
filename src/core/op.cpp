#include "vknn/op.h"
#include <unordered_map>

namespace vknn {

    const char *opTypeName(OpType t) {
        switch (t)
        {
            case OpType::Conv:
                return "Conv";
            case OpType::Clip:
                return "Clip";
            case OpType::Relu:
                return "Relu";
            case OpType::Add:
                return "Add";
            case OpType::GlobalAvgPool:
                return "GlobalAveragePool";
            case OpType::AvgPool:
                return "AveragePool";
            case OpType::MaxPool:
                return "MaxPool";
            case OpType::Gemm:
                return "Gemm";
            case OpType::MatMul:
                return "MatMul";
            case OpType::Einsum:
                return "Einsum";
            case OpType::Reshape:
                return "Reshape";
            case OpType::Expand:
                return "Expand";
            case OpType::Tile:
                return "Tile";
            case OpType::Squeeze:
                return "Squeeze";
            case OpType::Flatten:
                return "Flatten";
            case OpType::Softmax:
                return "Softmax";
            case OpType::LayerNorm:
                return "LayerNormalization";
            case OpType::BatchNorm:
                return "BatchNormalization";
            case OpType::Concat:
                return "Concat";
            case OpType::Pad:
                return "Pad";
            case OpType::Identity:
                return "Identity";
            case OpType::Constant:
                return "Constant";
            case OpType::Shape:
                return "Shape";
            case OpType::Gather:
                return "Gather";
            case OpType::Unsqueeze:
                return "Unsqueeze";
            case OpType::Unary:
                return "Unary";
            case OpType::Binary:
                return "Binary";
            case OpType::PRelu:
                return "PRelu";
            case OpType::Resize:
                return "Resize";
            case OpType::GridSample:
                return "GridSample";
            case OpType::Transpose:
                return "Transpose";
            case OpType::Slice:
                return "Slice";
            case OpType::Reduce:
                return "Reduce";
            case OpType::DepthToSpace:
                return "DepthToSpace";
            case OpType::Cast:
                return "Cast";
            case OpType::Split:
                return "Split";
            case OpType::Where:
                return "Where";
            case OpType::Equal:
                return "Equal";
            case OpType::ConstantOfShape:
                return "ConstantOfShape";
            case OpType::EyeLike:
                return "EyeLike";
            case OpType::ScatterND:
                return "ScatterND";
            case OpType::FusedSE:
                return "FusedSE";
            case OpType::FusedDwPw:
                return "FusedDwPw";
            case OpType::ConvertLayout:
                return "ConvertLayout";
            default:
                return "Unknown";
        }
    }

    UnaryType unaryFromOnnx(const std::string &s) {
        using U = UnaryType;
        static const std::unordered_map<std::string, UnaryType> m = {{"Sigmoid", U::Sigmoid}, {"Tanh", U::Tanh}, {"HardSwish", U::HardSwish}, {"HardSigmoid", U::HardSigmoid}, {"LeakyRelu", U::LeakyRelu}, {"Elu", U::Elu}, {"Abs", U::Abs}, {"Neg", U::Neg}, {"Exp", U::Exp}, {"Log", U::Log}, {"Sqrt", U::Sqrt}, {"Floor", U::Floor}, {"Ceil", U::Ceil}, {"Erf", U::Erf}, {"Cos", U::Cos}, {"Sin", U::Sin}, {"Reciprocal", U::Reciprocal}, {"Softplus", U::Softplus}};
        auto it = m.find(s);
        return it == m.end() ? U::Invalid : it->second;
    }
    ReduceType reduceFromOnnx(const std::string &s) {
        using R = ReduceType;
        if (s == "ReduceMean")
        {
            return R::Mean;
        }
        if (s == "ReduceSum")
        {
            return R::Sum;
        }
        if (s == "ReduceMax")
        {
            return R::Max;
        }
        if (s == "ReduceMin")
        {
            return R::Min;
        }
        if (s == "ReduceProd")
        {
            return R::Prod;
        }
        if (s == "ReduceL2")
        {
            return R::L2;
        }
        return R::Invalid;
    }
    BinaryType binaryFromOnnx(const std::string &s) {
        using B                                                     = BinaryType;
        static const std::unordered_map<std::string, BinaryType> m  = {{"Mul", B::Mul}, {"Sub", B::Sub}, {"Div", B::Div},
                                                                       {"Max", B::Max}, {"Min", B::Min}, {"Pow", B::Pow}};
        auto                                                     it = m.find(s);
        return it == m.end() ? B::Invalid : it->second;
    }

    OpType opTypeFromOnnx(const std::string &s) {
        static const std::unordered_map<std::string, OpType> m = {
            {"Conv", OpType::Conv},
            {"Clip", OpType::Clip},
            {"Relu", OpType::Relu},
            {"Add", OpType::Add},
            {"GlobalAveragePool", OpType::GlobalAvgPool},
            // ReduceMean over the spatial dims (keepdims) is exactly a global average pool; that's how
            // it shows up in ResNet exports, so we route it to the same kernel.
            {"ReduceMean", OpType::GlobalAvgPool},
            {"AveragePool", OpType::AvgPool},
            {"MaxPool", OpType::MaxPool},
            {"Gemm", OpType::Gemm},
            {"MatMul", OpType::MatMul},
            {"Einsum", OpType::Einsum},
            {"Reshape", OpType::Reshape},
            {"Expand", OpType::Expand},
            {"Tile", OpType::Tile},
            {"Squeeze", OpType::Squeeze},
            {"Flatten", OpType::Flatten},
            {"Softmax", OpType::Softmax},
            {"LayerNormalization", OpType::LayerNorm},
            {"BatchNormalization", OpType::BatchNorm},
            {"Concat", OpType::Concat},
            {"Pad", OpType::Pad},
            {"Identity", OpType::Identity},
            {"Constant", OpType::Constant},
            {"Shape", OpType::Shape},
            {"Gather", OpType::Gather},
            {"Unsqueeze", OpType::Unsqueeze},
            {"PRelu", OpType::PRelu},
            {"Resize", OpType::Resize},
            {"Upsample", OpType::Resize},
            {"GridSample", OpType::GridSample},
            {"Transpose", OpType::Transpose},
            {"Slice", OpType::Slice},
            {"DepthToSpace", OpType::DepthToSpace},
            {"Cast", OpType::Cast},
            {"Split", OpType::Split},
            {"Where", OpType::Where},
            {"Equal", OpType::Equal},
            {"ConstantOfShape", OpType::ConstantOfShape},
            {"EyeLike", OpType::EyeLike},
            {"ScatterND", OpType::ScatterND},
        };
        auto it = m.find(s);
        if (it != m.end())
        {
            return it->second;
        }
        if (s == "ReduceSum" || s == "ReduceMax" || s == "ReduceMin" || s == "ReduceProd" || s == "ReduceL2")
        {
            return OpType::Reduce;
        }
        if (unaryFromOnnx(s) != UnaryType::Invalid)
        {
            return OpType::Unary;
        }
        if (binaryFromOnnx(s) != BinaryType::Invalid)
        {
            return OpType::Binary;
        }
        return OpType::Unknown;
    }

} // namespace vknn
