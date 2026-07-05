// FusedPointwise: run a per-element step chain (pw_steps/pw_params) in fp32. The CPU op is the
// correctness oracle for the fused-epilogue chain; applyPwEpilogue is also the shared applier a
// later phase's executor hook calls to run an epilogue carried by a producer node.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>

namespace vknn {
    namespace {

        /// Apply one binary step `a OP b`, where `op` is a BinaryType wire code.
        /// The `default`/fall-through returns `a + b`: Add is encoded as BinaryType::Add (6) by the
        /// fuser (binaryFromOnnx() never yields it), so it lands here rather than in an explicit case.
        float pwBinary(float a, float b, int op) {
            switch ((BinaryType) op)
            {
                case BinaryType::Mul:
                    return a * b;
                case BinaryType::Sub:
                    return a - b;
                case BinaryType::Div:
                    return a / b;
                case BinaryType::Max:
                    return std::max(a, b);
                case BinaryType::Min:
                    return std::min(a, b);
                case BinaryType::Pow:
                    return std::pow(a, b);
                default:
                    break;
            }
            return a + b;
        }

        /// Apply one unary step to `x`, where `op` is a UnaryType wire code. `a`/`b` are the step's two
        /// scalar params (pw_params): the negative slope for LeakyRelu/Elu and the affine scale/bias for
        /// HardSigmoid; unused by the parameterless activations. Invalid/unrecognized codes pass `x`
        /// through unchanged.
        float pwUnary(float x, int op, float a, float b) {
            switch ((UnaryType) op)
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
                    // Numerically-stable log(1 + exp(x)): factoring out exp(max(x,0)) leaves the
                    // exponent argument always <= 0, so exp() never overflows for large positive x.
                    return std::max(x, 0.f) + std::log1p(std::exp(-std::fabs(x)));
                case UnaryType::Invalid:
                    break;
            }
            return x;
        }

        /// Apply one fused-activation step to `x`, where `act` is an ActType wire code. `lo`/`hi` are the
        /// clamp bounds, read only by ActType::Clip; the other activations ignore them. Relu6 uses the
        /// fixed [0, 6] range mandated by the op. None/unrecognized codes pass `x` through unchanged.
        float pwAct(float x, int act, float lo, float hi) {
            switch ((ActType) act)
            {
                case ActType::Relu:
                    return x > 0 ? x : 0;
                case ActType::Relu6:
                    return x < 0 ? 0 : (x > 6 ? 6 : x);
                case ActType::Clip:
                    return x < lo ? lo : (x > hi ? hi : x);
                case ActType::HardSwish:
                    return x * std::min(std::max(x + 3.f, 0.f), 6.f) / 6.f;
                case ActType::SiLU:
                    return x / (1.f + std::exp(-x));
                case ActType::None:
                    break;
            }
            return x;
        }

    } // namespace

    // Apply pw_steps/pw_params in place on node.outputs[0] (already holds the head/primary
    // result). Step operands (node.inputs[1..]) broadcast against the output shape per bcastMode:
    // 0=same-shape, 1=channel[N,C,1,1], 2=general (all handled by the same NumPy-style stride
    // computation from the operand's own shape, so this reference reads no per-mode fast path).
    //
    // Encoding (produced by the pointwise-chain fuser, src/import/fuse_pointwise_chains.cpp): pw_steps
    // is 4 ints per step [kind, code, operandIndex, unused] and pw_params is 2 floats per step
    // [p0, p1]. `kind` selects the step family:
    //   0 = binary, chain value on the LEFT:  acc = pwBinary(acc, operand, code)
    //   3 = binary REVERSED (chain value was on the RHS of a non-commutative op such as Sub/Div/Pow):
    //       acc = pwBinary(operand, acc, code)
    //   1 = unary activation:   acc = pwUnary(acc, code, p0, p1)
    //   2 = fused activation:   acc = pwAct(acc, code, p0, p1)
    // For kinds 0/3 `operandIndex` indexes node.inputs; for 1/2 it is 0 and p0/p1 carry any params.
    void applyPwEpilogue(const Node &node, ExecContext &ctx) {
        RtTensor       &Y    = ctx.t(node.outputs[0]);
        const Shape    &out  = Y.shape;
        int64_t         n    = numElements(out);
        float           *y   = Y.host.f32();
        size_t           rank = out.size();
        const auto      &st  = node.attr.getints("pw_steps");
        const auto      &pr  = node.attr.getfloats("pw_params");
        int              nSteps = (int) (st.size() / 4);

        // NumPy-style broadcast strides for an operand of shape `s` against the rank-`rank` output.
        // `s` is right-aligned to the output axes (leading axes it lacks, off = rank - s.size(), are
        // treated as size 1); an axis of extent 1 gets stride 0 so every output index along it re-reads
        // the same operand element, while other axes get the operand's own row-major stride.
        auto broadcastStrides = [&](const Shape &s) {
            std::vector<int64_t> ob(rank, 0);
            int64_t               stride = 1;
            size_t                 off    = rank - s.size();
            for (int i = (int) rank - 1; i >= 0; --i)
            {
                int64_t d = (i < (int) off) ? 1 : s[i - off];
                ob[i]     = (d == 1) ? 0 : stride;
                stride *= d;
            }
            return ob;
        };

        for (int64_t lin = 0; lin < n; ++lin)
        {
            float acc = y[lin];
            for (int s = 0; s < nSteps; ++s)
            {
                int   kind = (int) st[s * 4 + 0];
                int   code = (int) st[s * 4 + 1];
                int   oi   = (int) st[s * 4 + 2];
                float p0   = pr[s * 2 + 0];
                float p1   = pr[s * 2 + 1];
                if (kind == 0 || kind == 3)
                {
                    const RtTensor &O  = ctx.t(node.inputs[oi]);
                    auto            ob = broadcastStrides(O.shape);
                    // Decompose the flat output index `lin` into its per-axis coordinates and
                    // reassemble the operand offset from the broadcast strides. For output axis d the
                    // coordinate is (lin / prod(out[d+1..])) % out[d]; multiplying by ob[d] (which is 0
                    // on a broadcast axis) accumulates the operand's read position.
                    int64_t io = 0;
                    for (size_t d = 0; d < rank; ++d)
                    {
                        int64_t stride = 1;
                        for (size_t e = d + 1; e < rank; ++e)
                        {
                            stride *= out[e];
                        }
                        io += ((lin / stride) % out[d]) * ob[d];
                    }
                    // kind 3 is the reversed (operand-on-left) form for non-commutative binaries.
                    float b = O.host.f32()[io];
                    acc     = kind == 3 ? pwBinary(b, acc, code) : pwBinary(acc, b, code);
                }
                else if (kind == 1)
                {
                    acc = pwUnary(acc, code, p0, p1);
                }
                else if (kind == 2)
                {
                    acc = pwAct(acc, code, p0, p1);
                }
            }
            y[lin] = acc;
        }
    }

    namespace {
        struct FusedPointwiseCpu: CpuOp {
            // The standalone node has no producer to seed the output, so the head input (inputs[0], the
            // chain's primary full-size stream) is copied into the output verbatim to play the role of
            // the "primary result"; applyPwEpilogue() then folds pw_steps over it in place.
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                float          *y = cpu::allocOut(Y, X.shape);
                const float    *x = X.host.f32();
                int64_t         n = numElements(X.shape);
                for (int64_t i = 0; i < n; ++i)
                {
                    y[i] = x[i];
                }
                applyPwEpilogue(node, ctx);
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::FusedPointwise, FusedPointwiseCpu);
} // namespace vknn
