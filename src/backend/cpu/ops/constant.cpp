// Constant: emit the node's stored value (int64 or float vector).
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        /// ONNX Constant: a source node with no inputs whose single output is a compile-time literal
        /// carried in the `value` attribute. The reference just materializes that attribute into the
        /// output tensor; the import path has already flattened the tensor's elements into `ints`
        /// (int64 list) or `floats` (float32 list) and stashed its dims in `value.shape`.
        struct ConstantCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                RtTensor &Y  = ctx.t(node.outputs[0]);
                auto      it = node.attr.map.find("value");
                // Emit the value with its original dims when known; a flat [n] tensor broadcasts wrong for a
                // multi-dim constant (e.g. a (1,2,8400) anchor grid against (1,2,8400) features -> bogus
                // shape).
                auto shapeOf = [&](size_t n) -> Shape {
                    if (it != node.attr.map.end() && !it->second.shape.empty())
                    {
                        return it->second.shape;
                    }
                    return {(int64_t) n};
                };
                if (it != node.attr.map.end() && it->second.kind == Attr::Ints)
                {
                    const auto &v = it->second.ints;
                    int64_t    *y = cpu::allocOutI64(Y, shapeOf(v.size()));
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        y[i] = v[i];
                    }
                } else if (it != node.attr.map.end() && it->second.kind == Attr::Floats)
                {
                    const auto &v = it->second.floats;
                    float      *y = cpu::allocOut(Y, shapeOf(v.size()));
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        y[i] = v[i];
                    }
                } else
                {
                    // No usable `value` attribute (absent, or a kind this reference does not carry):
                    // emit an empty int64 tensor of shape [0] so downstream shape inference stays
                    // well-defined rather than reading an unsized output.
                    cpu::allocOutI64(Y, {0});
                }
            }
            // The output dtype is fixed by the stored attribute's kind (int64 for Ints, fp32 for
            // Floats), not chosen by the caller, so every requested dtype is trivially accepted.
            bool supportsDType(DType) const override {
                return true;
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Constant, ConstantCpu);
} // namespace vknn
