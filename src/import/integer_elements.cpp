// Integer element values and int64 storage per tensor; see integer_elements.h for the rules.
#include "import/integer_elements.h"
#include "import/onnx/onnx_types.h"
#include "vknn/binary_type.h"
#include "vknn/reduce_type.h"
#include "vknn/unary_type.h"
#include <algorithm>
#include <utility>

namespace vknn {
    namespace {

        // Operand slots an ElementRule mask names, one bit per slot.
        constexpr size_t kMaskedOperandSlots = 32;
        // The data operand of a movement op, a reduction, a unary function; the base of a Pow.
        constexpr size_t kLeadingOperandSlot = 0;
        // Binary operands: the two sides of an Add, Mod, bitwise op or Binary.
        constexpr size_t kSecondOperandSlot = 1;
        // Pad's constant fill value and ScatterND's updates: the element type of the data.
        constexpr size_t kThirdOperandSlot = 2;
        // TopK output slot holding the int64 indices (output 0 holds the values).
        constexpr size_t kTopKIndicesOutput = 1;

        constexpr uint32_t operandSlotBit(size_t slot) {
            return uint32_t {1} << slot;
        }

        // What a producer says about the fact on one of its outputs.
        struct ElementRule {
            enum class Kind : uint8_t {
                Holds,        // the output has the fact whatever the operands hold
                Lacks,        // the output lacks the fact
                AnyOperand,   // the output has the fact when an operand in the mask has it
                EveryOperand, // the output has the fact when every operand in the mask has it
            };
            Kind     kind             = Kind::Lacks;
            uint32_t operandMask      = 0;
            bool     everyCoreOperand = false; // the mask is every core operand (a variadic Concat)
        };

        ElementRule holdsRule() {
            ElementRule rule;
            rule.kind = ElementRule::Kind::Holds;
            return rule;
        }

        ElementRule anyOperandRule(uint32_t operandMask) {
            ElementRule rule;
            rule.kind        = ElementRule::Kind::AnyOperand;
            rule.operandMask = operandMask;
            return rule;
        }

        ElementRule anyCoreOperandRule() {
            ElementRule rule      = anyOperandRule(0);
            rule.everyCoreOperand = true;
            return rule;
        }

        bool integerLabel(DType dtype) {
            return dtype == DType::Int64 || dtype == DType::Int32 || dtype == DType::Int8 || dtype == DType::UInt8;
        }

        // A Constant or ConstantOfShape whose `value` the importer recorded as integers (Ints); a float value
        // is Floats.
        bool integerValueAttribute(const Node &node) {
            const auto value = node.attr.map.find("value");
            return value != node.attr.map.end() && value->second.kind == Attr::Ints;
        }

        bool integerReduction(const Node &node) {
            switch ((ReduceType) node.subOp)
            {
                case ReduceType::Sum:
                case ReduceType::Max:
                case ReduceType::Min:
                case ReduceType::Prod:
                    return true;
                default:
                    return false; // Mean and L2 are float results on integer data too
            }
        }

        constexpr uint32_t kLeading   = operandSlotBit(kLeadingOperandSlot);
        constexpr uint32_t kBothSides = operandSlotBit(kLeadingOperandSlot) | operandSlotBit(kSecondOperandSlot);

        ElementRule integerValuesRule(const Node &producer, TensorId output) {
            switch (producer.type)
            {
                case OpType::Cast:
                    return onnx::castTargetsIntegerElementType(producer) ? holdsRule() : ElementRule {};
                case OpType::Shape:
                case OpType::ArgMax:
                case OpType::ArgMin:
                case OpType::BitShift:
                case OpType::BitwiseAnd:
                case OpType::BitwiseOr:
                case OpType::BitwiseXor:
                case OpType::BitwiseNot:
                    return holdsRule();
                case OpType::Constant:
                case OpType::ConstantOfShape:
                    return integerValueAttribute(producer) ? holdsRule() : ElementRule {};
                case OpType::TopK:
                    if (producer.outputs.size() > kTopKIndicesOutput && producer.outputs[kTopKIndicesOutput] == output)
                    {
                        return holdsRule();
                    }
                    return anyOperandRule(kLeading);
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
                case OpType::DepthToSpace:
                case OpType::ChannelShuffle:
                case OpType::Relu:
                case OpType::Clip:
                    return anyOperandRule(kLeading);
                case OpType::Unary:
                    switch ((UnaryType) producer.subOp)
                    {
                        case UnaryType::Neg:
                        case UnaryType::Abs:
                            return anyOperandRule(kLeading);
                        default:
                            return {}; // the other unary functions are float math
                    }
                case OpType::Reduce:
                    return integerReduction(producer) ? anyOperandRule(kLeading) : ElementRule {};
                case OpType::Pad:
                case OpType::ScatterND:
                    return anyOperandRule(kLeading | operandSlotBit(kThirdOperandSlot));
                case OpType::Range:
                    return anyOperandRule(kBothSides | operandSlotBit(kThirdOperandSlot));
                case OpType::Where:
                    return anyOperandRule(operandSlotBit(kSecondOperandSlot) | operandSlotBit(kThirdOperandSlot));
                case OpType::Concat:
                case OpType::Add:
                case OpType::Mod:
                    return anyCoreOperandRule();
                case OpType::Binary:
                    // Pow's exponent may have another element type than its base; the result has the base's.
                    return (BinaryType) producer.subOp == BinaryType::Pow ? anyOperandRule(kLeading) : anyCoreOperandRule();
                default:
                    return {};
            }
        }

        ElementRule int64StorageRule(const Node &producer, TensorId output) {
            switch (producer.type)
            {
                case OpType::Cast: {
                    // The CPU Cast stores every integer target and BOOL as int64.
                    const int64_t to = producer.attr.geti("to", (int64_t) onnx::OnnxType::Float);
                    return onnx::isIntegerElementType(to) || to == (int64_t) onnx::OnnxType::Bool ? holdsRule() : ElementRule {};
                }
                case OpType::Shape:
                case OpType::ArgMax:
                case OpType::ArgMin:
                    return holdsRule();
                case OpType::Constant:
                case OpType::ConstantOfShape:
                    return integerValueAttribute(producer) ? holdsRule() : ElementRule {};
                case OpType::TopK:
                    if (producer.outputs.size() > kTopKIndicesOutput && producer.outputs[kTopKIndicesOutput] == output)
                    {
                        return holdsRule();
                    }
                    return anyOperandRule(kLeading);
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
                case OpType::ScatterND:
                case OpType::BitwiseNot:
                    return anyOperandRule(kLeading);
                case OpType::Reduce:
                    return integerReduction(producer) ? anyOperandRule(kLeading) : ElementRule {};
                case OpType::Add:
                case OpType::Mod:
                case OpType::BitShift:
                case OpType::BitwiseAnd:
                case OpType::BitwiseOr:
                case OpType::BitwiseXor:
                    return anyOperandRule(kBothSides);
                case OpType::Binary:
                    // A float base raised to an int64 exponent is a float power.
                    return anyOperandRule((BinaryType) producer.subOp == BinaryType::Pow ? kLeading : kBothSides);
                case OpType::Concat:
                    return anyCoreOperandRule();
                case OpType::Where:
                    return anyOperandRule(operandSlotBit(kSecondOperandSlot) | operandSlotBit(kThirdOperandSlot));
                case OpType::Range: {
                    ElementRule rule = anyOperandRule(kBothSides | operandSlotBit(kThirdOperandSlot));
                    rule.kind        = ElementRule::Kind::EveryOperand;
                    return rule;
                }
                default:
                    return {};
            }
        }

    } // namespace

    std::vector<int> tensorProducers(const Graph &g) {
        std::vector<int> producers(g.tensors.size(), kNoTensorProducer);
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

    ElementFactResolver::ElementFactResolver(const Graph &g, ElementFact fact): ElementFactResolver(g, fact, tensorProducers(g)) {
    }

    ElementFactResolver::ElementFactResolver(const Graph &g, ElementFact fact, std::vector<int> producers):
        graph_(g), fact_(fact), producers_(std::move(producers)), resolution_(g.tensors.size(), Resolution::Unresolved) {
        producers_.resize(g.tensors.size(), kNoTensorProducer);
    }

    bool ElementFactResolver::holds(TensorId tensor) {
        const auto validTensor = [&](TensorId id) {
            return id >= 0 && (size_t) id < resolution_.size();
        };
        if (!validTensor(tensor))
        {
            return false;
        }
        // One pending tensor of the depth-first walk: the operands its producer's rule reads, and the next
        // one to inspect. The walk keeps its own stack, so a long producer chain cannot exhaust the call
        // stack.
        struct PendingTensor {
            TensorId              tensor;
            std::vector<TensorId> operands;
            size_t                nextOperand  = 0;
            bool                  everyOperand = false;
        };
        std::vector<PendingTensor> pending;
        pending.push_back({tensor, {}, 0, false});
        while (!pending.empty())
        {
            const size_t   top     = pending.size() - 1;
            const TensorId current = pending[top].tensor;
            if (resolution_[(size_t) current] == Resolution::Unresolved)
            {
                const TensorDesc &desc          = graph_.desc(current);
                const bool        labelHolds    = fact_ == ElementFact::IntegerValues ? integerLabel(desc.dtype) : desc.dtype == DType::Int64;
                const int         producerIndex = producers_[(size_t) current];
                if (labelHolds)
                {
                    resolution_[(size_t) current] = Resolution::Holds;
                    pending.pop_back();
                    continue;
                }
                if (desc.isInput || desc.isInitializer || graph_.isInitializer(current) || producerIndex == kNoTensorProducer)
                {
                    resolution_[(size_t) current] = Resolution::Lacks; // a declared source without the fact
                    pending.pop_back();
                    continue;
                }
                const Node &producer = graph_.nodes[(size_t) producerIndex];
                const ElementRule rule = producer.attr.has("pw_steps") ? ElementRule {} : (fact_ == ElementFact::IntegerValues ? integerValuesRule(producer, current) : int64StorageRule(producer, current));
                if (rule.kind == ElementRule::Kind::Holds || rule.kind == ElementRule::Kind::Lacks)
                {
                    resolution_[(size_t) current] = rule.kind == ElementRule::Kind::Holds ? Resolution::Holds : Resolution::Lacks;
                    pending.pop_back();
                    continue;
                }
                const size_t coreOperands = std::min(producer.inputs.size(), (size_t) pwCoreInputs(producer));
                for (size_t slot = 0; slot < coreOperands; ++slot)
                {
                    if (rule.everyCoreOperand || (slot < kMaskedOperandSlots && (rule.operandMask & operandSlotBit(slot)) != 0))
                    {
                        pending[top].operands.push_back(producer.inputs[slot]);
                    }
                }
                pending[top].everyOperand     = rule.kind == ElementRule::Kind::EveryOperand;
                resolution_[(size_t) current] = Resolution::Pending;
            }
            if (resolution_[(size_t) current] != Resolution::Pending)
            {
                pending.pop_back(); // resolved while waiting on the stack
                continue;
            }
            bool decided  = false;
            bool deferred = false;
            while (pending[top].nextOperand < pending[top].operands.size())
            {
                const TensorId operand    = pending[top].operands[pending[top].nextOperand];
                Resolution     resolution = validTensor(operand) ? resolution_[(size_t) operand] : Resolution::Lacks;
                if (resolution == Resolution::Unresolved)
                {
                    pending.push_back({operand, {}, 0, false});
                    deferred = true;
                    break;
                }
                if (resolution == Resolution::Pending)
                {
                    resolution = Resolution::Lacks; // a producer cycle carries no element type
                }
                ++pending[top].nextOperand;
                if (!pending[top].everyOperand && resolution == Resolution::Holds)
                {
                    resolution_[(size_t) current] = Resolution::Holds;
                    decided                       = true;
                    break;
                }
                if (pending[top].everyOperand && resolution == Resolution::Lacks)
                {
                    resolution_[(size_t) current] = Resolution::Lacks;
                    decided                       = true;
                    break;
                }
            }
            if (deferred)
            {
                continue;
            }
            if (!decided)
            {
                const bool everyOperandHeld   = pending[top].everyOperand && !pending[top].operands.empty();
                resolution_[(size_t) current] = everyOperandHeld ? Resolution::Holds : Resolution::Lacks;
            }
            pending.pop_back();
        }
        return resolution_[(size_t) tensor] == Resolution::Holds;
    }

    std::vector<char> resolveElementFact(const Graph &g, ElementFact fact) {
        ElementFactResolver resolver(g, fact);
        std::vector<char>   facts(g.tensors.size(), 0);
        for (TensorId tensor = 0; tensor < (TensorId) g.tensors.size(); ++tensor)
        {
            facts[(size_t) tensor] = resolver.holds(tensor) ? 1 : 0;
        }
        return facts;
    }

} // namespace vknn
