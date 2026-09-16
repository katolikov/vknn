// Integer element-type resolution for ONNX Mod; see mod_integer_operands.h for the rules.
#include "import/mod_integer_operands.h"
#include "vknn/binary_type.h"
#include <algorithm>

namespace vknn {
    namespace {

        // ONNX TensorProto.DataType codes: FLOAT (the Cast `to` default) and the integer element types a
        // Cast can target. BOOL is not one of them: Mod is not defined on booleans.
        constexpr int64_t kOnnxFloat  = 1;
        constexpr int64_t kOnnxUInt8  = 2;
        constexpr int64_t kOnnxInt8   = 3;
        constexpr int64_t kOnnxUInt16 = 4;
        constexpr int64_t kOnnxInt16  = 5;
        constexpr int64_t kOnnxInt32  = 6;
        constexpr int64_t kOnnxInt64  = 7;
        constexpr int64_t kOnnxUInt32 = 12;
        constexpr int64_t kOnnxUInt64 = 13;

        // Operand count of ONNX Mod: dividend (input 0) and divisor (input 1).
        constexpr size_t kModOperandCount = 2;
        // modTensorProducers entry of a tensor no node writes.
        constexpr int kNoProducer = -1;
        // TopK output slot holding the int64 indices (output 0 holds the values).
        constexpr size_t kTopKIndicesOutput = 1;
        // Where operand slots holding the selected values [first, end) (operand 0 is the condition).
        constexpr size_t kWhereFirstValueOperand = 1;
        constexpr size_t kWhereEndValueOperand   = 3;
        // Operand slot range [0, end) of an op whose result copies or computes from operand 0 alone
        // (the other operands are shape, index, axis or count parameters).
        constexpr size_t kLeadingOperandEnd = 1;

        bool integerDType(DType dtype) {
            return dtype == DType::Int64 || dtype == DType::Int32 || dtype == DType::Int8 || dtype == DType::UInt8;
        }

        bool castTargetsIntegerElements(const Node &cast) {
            const int64_t to = cast.attr.geti("to", kOnnxFloat);
            return to == kOnnxUInt8 || to == kOnnxInt8 || to == kOnnxUInt16 || to == kOnnxInt16 || to == kOnnxInt32 || to == kOnnxInt64 || to == kOnnxUInt32 || to == kOnnxUInt64;
        }

        // What a producer's output says about its element type.
        enum class ElementTypeSource : uint8_t {
            Integer,        // the op yields integers whatever its operands hold
            SharedOperands, // the result's element type is the one of the operands in [firstOperand, endOperand)
            Unresolved,     // float, or not derivable from the IR
        };

        struct ProducerElementType {
            ElementTypeSource source       = ElementTypeSource::Unresolved;
            size_t            firstOperand = 0;
            size_t            endOperand   = 0;
        };

        ProducerElementType producerElementType(const Node &producer, TensorId output) {
            if (producer.attr.has("pw_steps"))
            {
                return {}; // a fused pointwise chain computes new values of its own type
            }
            const size_t operandCount = producer.inputs.size();
            auto         shared       = [&](size_t firstOperand, size_t endOperand) {
                return ProducerElementType {ElementTypeSource::SharedOperands, firstOperand, std::min(endOperand, operandCount)};
            };
            const ProducerElementType integer {ElementTypeSource::Integer, 0, 0};
            switch (producer.type)
            {
                case OpType::Cast:
                    return castTargetsIntegerElements(producer) ? integer : ProducerElementType {};
                case OpType::Shape:
                case OpType::ArgMax:
                case OpType::ArgMin:
                case OpType::BitShift:
                case OpType::BitwiseAnd:
                case OpType::BitwiseOr:
                case OpType::BitwiseXor:
                case OpType::BitwiseNot:
                    return integer;
                case OpType::ConstantOfShape: {
                    // The importer records an integer fill `value` as an Ints attribute (the CPU kernel's
                    // int64 output switch) and a float fill as Floats.
                    const auto fill = producer.attr.map.find("value");
                    return fill != producer.attr.map.end() && fill->second.kind == Attr::Ints ? integer : ProducerElementType {};
                }
                case OpType::TopK:
                    if (producer.outputs.size() > kTopKIndicesOutput && producer.outputs[kTopKIndicesOutput] == output)
                    {
                        return integer;
                    }
                    return shared(0, kLeadingOperandEnd);
                case OpType::ConvertLayout:
                case OpType::ConvertDtype:
                case OpType::Identity:
                case OpType::Reshape:
                case OpType::Flatten:
                case OpType::Squeeze:
                case OpType::Unsqueeze:
                case OpType::Slice:
                case OpType::Transpose:
                case OpType::Expand:
                case OpType::Tile:
                case OpType::Split:
                case OpType::Gather:
                case OpType::Pad:
                case OpType::DepthToSpace:
                case OpType::ScatterND:
                case OpType::Range:
                    return shared(0, kLeadingOperandEnd);
                case OpType::Where:
                    return shared(kWhereFirstValueOperand, kWhereEndValueOperand);
                case OpType::Concat:
                case OpType::Add:
                case OpType::Mod:
                    return shared(0, operandCount);
                case OpType::Binary:
                    // Pow's exponent may have another element type than its base; the result has the base's.
                    return shared(0, (BinaryType) producer.subOp == BinaryType::Pow ? kLeadingOperandEnd : operandCount);
                default:
                    return {};
            }
        }

    } // namespace

    std::vector<int> modTensorProducers(const Graph &g) {
        std::vector<int> producers(g.tensors.size(), kNoProducer);
        for (int nodeIndex = 0; nodeIndex < (int) g.nodes.size(); ++nodeIndex)
        {
            for (TensorId output: g.nodes[(size_t) nodeIndex].outputs)
            {
                if (output >= 0 && (size_t) output < producers.size())
                {
                    producers[(size_t) output] = nodeIndex;
                }
            }
        }
        return producers;
    }

    bool modOperandsAreInteger(const Graph &g, const Node &mod, const std::vector<int> &producers) {
        const auto validTensor = [&](TensorId tensor) {
            return tensor >= 0 && (size_t) tensor < g.tensors.size();
        };
        if (!mod.outputs.empty() && validTensor(mod.outputs[0]) && integerDType(g.desc(mod.outputs[0]).dtype))
        {
            return true;
        }
        std::vector<char>     visited(g.tensors.size(), 0);
        std::vector<TensorId> pending;
        for (size_t slot = 0; slot < std::min(mod.inputs.size(), kModOperandCount); ++slot)
        {
            pending.push_back(mod.inputs[slot]);
        }
        while (!pending.empty())
        {
            const TensorId tensor = pending.back();
            pending.pop_back();
            if (!validTensor(tensor) || visited[(size_t) tensor])
            {
                continue;
            }
            visited[(size_t) tensor] = 1;
            const TensorDesc &desc   = g.desc(tensor);
            if (integerDType(desc.dtype))
            {
                return true;
            }
            const int producerIndex = (size_t) tensor < producers.size() ? producers[(size_t) tensor] : kNoProducer;
            if (desc.isInput || desc.isInitializer || g.isInitializer(tensor) || producerIndex == kNoProducer || (size_t) producerIndex >= g.nodes.size())
            {
                continue; // a declared non-integer source
            }
            const Node               &producer    = g.nodes[(size_t) producerIndex];
            const ProducerElementType elementType = producerElementType(producer, tensor);
            if (elementType.source == ElementTypeSource::Integer)
            {
                return true;
            }
            for (size_t slot = elementType.firstOperand; slot < elementType.endOperand; ++slot)
            {
                pending.push_back(producer.inputs[slot]);
            }
        }
        return false;
    }

    bool modOperandsAreInteger(const Graph &g, const Node &mod) {
        return modOperandsAreInteger(g, mod, modTensorProducers(g));
    }

} // namespace vknn
