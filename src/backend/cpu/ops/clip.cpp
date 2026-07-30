// Clip(x, min, max): elementwise clamp of X into [min, max]. A missing bound is unbounded on that
// side, so ReLU6 (min=0, max=6, as MobileNetV2 emits) and plain ReLU (min=0, no max) are both Clips.
//
// The bounds arrive two ways depending on opset: opset >= 11 passes min/max as optional tensor
// INPUTS (inputs[1], inputs[2]); opset <= 6 passes them as float ATTRIBUTES. Both forms are read
// here, and the attribute read runs last, so an attribute wins over an input for the same bound.
#include "backend/cpu/cpu_backend.h"
#include <limits>

namespace vknn {
    namespace {

        struct ClipCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                // Default an absent bound to +/-infinity so the clamp below is a no-op on that side.
                float lo = -std::numeric_limits<float>::infinity();
                float hi = std::numeric_limits<float>::infinity();
                // min/max are scalar tensors; kNoTensor marks an optional input the model omitted
                // (an unsupplied middle input still occupies a slot), so both existence and the
                // sentinel are checked before dereferencing element [0].
                if (node.inputs.size() > 1 && node.inputs[1] != kNoTensor)
                {
                    lo = ctx.t(node.inputs[1]).host.f32()[0];
                }
                if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor)
                {
                    hi = ctx.t(node.inputs[2]).host.f32()[0];
                }
                // Legacy attribute form; applied after the inputs so it takes precedence (see header).
                if (node.attr.has("min"))
                {
                    lo = node.attr.getf("min", lo);
                }
                if (node.attr.has("max"))
                {
                    hi = node.attr.getf("max", hi);
                }
                int64_t      n = cpu::elemCount(X.shape); // a rank-0 scalar carries its one element
                float       *y = cpu::allocOut(Y, X.shape);
                const float *x = X.host.f32();
                for (int64_t i = 0; i < n; ++i)
                {
                    float v = x[i];
                    // Clamp low first, then high. Both comparisons are false for a NaN input, so a
                    // NaN falls through to the trailing v and propagates unchanged.
                    y[i] = v < lo ? lo : (v > hi ? hi : v);
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Clip, ClipCpu);
} // namespace vknn
