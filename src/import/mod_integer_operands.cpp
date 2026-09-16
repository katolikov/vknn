// Integer element-type resolution for ONNX Mod; see mod_integer_operands.h for the rules.
#include "import/mod_integer_operands.h"
#include "import/integer_elements.h"
#include <algorithm>

namespace vknn {
    namespace {

        // Operand count of ONNX Mod: dividend (input 0) and divisor (input 1).
        constexpr size_t kModOperandCount = 2;

        bool integerDType(DType dtype) {
            return dtype == DType::Int64 || dtype == DType::Int32 || dtype == DType::Int8 || dtype == DType::UInt8;
        }

    } // namespace

    std::vector<int> modTensorProducers(const Graph &g) {
        return tensorProducers(g);
    }

    bool modOperandsAreInteger(const Graph &g, const Node &mod, const std::vector<int> &producers) {
        if (!mod.outputs.empty() && mod.outputs[0] >= 0 && (size_t) mod.outputs[0] < g.tensors.size() && integerDType(g.desc(mod.outputs[0]).dtype))
        {
            return true;
        }
        ElementFactResolver integerValues(g, ElementFact::IntegerValues, producers);
        for (size_t slot = 0; slot < std::min(mod.inputs.size(), kModOperandCount); ++slot)
        {
            if (integerValues.holds(mod.inputs[slot]))
            {
                return true;
            }
        }
        return false;
    }

    bool modOperandsAreInteger(const Graph &g, const Node &mod) {
        return modOperandsAreInteger(g, mod, modTensorProducers(g));
    }

} // namespace vknn
