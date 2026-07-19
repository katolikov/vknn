// Elementwise unary family (Sigmoid/Tanh/HardSwish/HardSigmoid/LeakyRelu/Elu/Abs/Neg/Exp/Log/
// Sqrt/Floor/Ceil/Relu). One op, switched on node.subOp; params (alpha/beta) in actLo/actHi.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <cmath>

namespace vknn {
    namespace {

        /// Evaluate one elementwise unary activation in the element/accumulation type T. T=float
        /// reproduces the GPU shader's fp32 arithmetic bit-for-bit (the CPU op is the GPU oracle);
        /// T=double evaluates the same formula in real fp64 for a double-precision input (so, e.g.,
        /// Sign reads the true sign of a fp64 value rather than one that narrowing to fp32 might flip).
        /// @param x  Input value.
        /// @param op UnaryType sub-code (Node::subOp).
        /// @param a  First parameter (Node::actLo): LeakyRelu/Elu alpha, HardSigmoid alpha. Ignored by
        ///           parameter-free ops.
        /// @param b  Second parameter (Node::actHi): HardSigmoid beta. Ignored by every other op.
        /// @returns  op(x). Invalid (and any unlisted code) passes x through unchanged.
        template<typename T>
        static T unary(T x, UnaryType op, T a, T b) {
            switch (op)
            {
                case UnaryType::Sigmoid:
                    return T(1) / (T(1) + std::exp(-x));
                case UnaryType::Tanh:
                    return std::tanh(x);
                case UnaryType::HardSwish:
                    return x * std::min(std::max(x + T(3), T(0)), T(6)) / T(6);
                case UnaryType::HardSigmoid:
                    // clamp(alpha*x + beta, 0, 1); alpha in a, beta in b (ONNX defaults 0.2, 0.5).
                    return std::min(std::max(a * x + b, T(0)), T(1));
                case UnaryType::LeakyRelu:
                    return x > T(0) ? x : a * x; // negative slope alpha in a
                case UnaryType::Elu:
                    return x > T(0) ? x : a * (std::exp(x) - T(1)); // saturation scale alpha in a
                case UnaryType::Abs:
                    return std::fabs(x);
                case UnaryType::Neg:
                    return -x;
                case UnaryType::Exp:
                    return std::exp(x);
                case UnaryType::Log:
                    return std::log(x);
                case UnaryType::Sqrt:
                    return std::sqrt(x);
                case UnaryType::Floor:
                    return std::floor(x);
                case UnaryType::Ceil:
                    return std::ceil(x);
                case UnaryType::Relu:
                    return x > T(0) ? x : T(0);
                case UnaryType::SiLU:
                    return x / (T(1) + std::exp(-x));
                case UnaryType::Erf:
                    return std::erf(x);
                case UnaryType::Cos:
                    return std::cos(x);
                case UnaryType::Sin:
                    return std::sin(x);
                case UnaryType::Reciprocal:
                    return T(1) / x;
                case UnaryType::Softplus:
                    // log(1 + exp(x)), evaluated as max(x,0) + log1p(exp(-|x|)) so the exp never
                    // overflows for large positive x and stays accurate for large negative x.
                    return std::max(x, T(0)) + std::log1p(std::exp(-std::fabs(x)));
                case UnaryType::Round:
                    // Nearest integer, ties to even (the FE_TONEAREST default); agrees bitwise with
                    // GLSL roundEven, including the sign of a zero result (-0.5 -> -0.0).
                    return std::nearbyint(x);
                case UnaryType::Sign:
                    // 1/-1 for nonzero, +-0 and NaN pass through unchanged — the same expression the
                    // GLSL evaluator uses, so CPU and GPU agree bitwise on every input.
                    return x > T(0) ? T(1) : (x < T(0) ? T(-1) : x);
                case UnaryType::Trunc:
                    // Round toward zero (drop the fraction); agrees bitwise with GLSL trunc and with
                    // the float->wide-int->float Cast pair foldIntRoundtripCast collapses into it.
                    return std::trunc(x);
                case UnaryType::Invalid:
                    break;
            }
            return x;
        }

        struct UnaryCpu: CpuOp {
            // Apply the selected activation independently to each element; output keeps the input shape
            // and dtype. subOp selects the op and actLo/actHi carry its parameters (see unary()). A
            // real-fp64 input is evaluated in double; otherwise the fp32 path stays the GPU oracle.
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X  = ctx.t(node.inputs[0]);
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                int64_t         n  = cpu::elemCount(X.shape); // a rank-0 scalar carries its one element
                const UnaryType op = (UnaryType) node.subOp;
                if (X.dtype == DType::Float64)
                {
                    double       *y = cpu::allocOutF64(Y, X.shape);
                    const double *x = X.host.f64();
                    for (int64_t i = 0; i < n; ++i)
                    {
                        y[i] = unary<double>(x[i], op, (double) node.actLo, (double) node.actHi);
                    }
                    return;
                }
                float       *y = cpu::allocOut(Y, X.shape);
                const float *x = X.host.f32();
                for (int64_t i = 0; i < n; ++i)
                {
                    y[i] = unary<float>(x[i], op, node.actLo, node.actHi);
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Unary, UnaryCpu);
} // namespace vknn
