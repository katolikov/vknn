// Elementwise unary family (Sigmoid/Tanh/HardSwish/HardSigmoid/LeakyRelu/Elu/Abs/Neg/Exp/Log/
// Sqrt/Floor/Ceil/Relu). One op, switched on node.subOp; params (alpha/beta) in actLo/actHi.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <cmath>

namespace vknn {
    namespace {

        static float unary(float x, UnaryType op, float a, float b) {
            switch (op)
            {
                case UnaryType::Sigmoid:
                    return 1.f / (1.f + std::exp(-x));
                case UnaryType::Tanh:
                    return std::tanh(x);
                case UnaryType::HardSwish:
                    return x * std::min(std::max(x + 3.f, 0.f), 6.f) / 6.f;
                case UnaryType::HardSigmoid:
                    return std::min(std::max(a * x + b, 0.f), 1.f);
                case UnaryType::LeakyRelu:
                    return x > 0 ? x : a * x;
                case UnaryType::Elu:
                    return x > 0 ? x : a * (std::exp(x) - 1.f);
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
                    return x > 0 ? x : 0;
                case UnaryType::SiLU:
                    return x / (1.f + std::exp(-x));
                case UnaryType::Erf:
                    return std::erf(x);
                case UnaryType::Cos:
                    return std::cos(x);
                case UnaryType::Sin:
                    return std::sin(x);
                case UnaryType::Reciprocal:
                    return 1.f / x;
                case UnaryType::Softplus:
                    return std::max(x, 0.f) + std::log1p(std::exp(-std::fabs(x)));
                case UnaryType::Invalid:
                    break;
            }
            return x;
        }

        struct UnaryCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                int64_t         n = X.elems();
                float          *y = cpu::allocOut(Y, X.shape);
                const float    *x = X.host.f32();
                for (int64_t i = 0; i < n; ++i)
                {
                    y[i] = unary(x[i], (UnaryType) node.subOp, node.actLo, node.actHi);
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Unary, UnaryCpu);
} // namespace vknn
