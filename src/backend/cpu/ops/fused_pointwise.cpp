// FusedPointwise: run a per-element step chain (pw_steps/pw_params) in fp32. The CPU op is the
// correctness oracle for the fused-epilogue chain; applyPwEpilogue is also the shared applier a
// later phase's executor hook calls to run an epilogue carried by a producer node.
#include "backend/cpu/broadcast.h"
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>

namespace vknn {
    namespace {

        /// Apply one binary step `a OP b`, where `op` is a BinaryType wire code or one of the
        /// pw-step-only kPwBin* codes (comparisons and PRelu, which fuse through this space).
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
            if (op == kPwBinGreater)
            {
                return a > b ? 1.f : 0.f;
            }
            if (op == kPwBinGreaterEqual)
            {
                return a >= b ? 1.f : 0.f;
            }
            if (op == kPwBinEqual)
            {
                return a == b ? 1.f : 0.f;
            }
            if (op == kPwBinPRelu)
            {
                return a > 0.f ? a : b * a;
            }
            if (op == kPwBinLess)
            {
                return a < b ? 1.f : 0.f;
            }
            if (op == kPwBinLessEqual)
            {
                return a <= b ? 1.f : 0.f;
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
                case UnaryType::Round:
                    // Nearest integer, ties to even (the FE_TONEAREST default); agrees bitwise with
                    // GLSL roundEven, including the sign of a zero result (-0.5 -> -0.0).
                    return std::nearbyint(x);
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

    // Apply pw_steps/pw_params/pw_outs in place on node.outputs[0] (already holds the entry /
    // primary result) and fill any extra output streams (node.outputs[1..]).
    //
    // Encoding (produced by the pointwise fuser, src/import/fuse_pointwise_chains.cpp): pw_steps is
    // 8 ints per step [kind, code, srcA, srcB, srcC, dst, bcast, bcastSrc] and pw_params is 2 floats
    // per step [p0, p1]. Sources reference the accumulator (kPwRefAcc), the entry value
    // (kPwRefEntry), a register (kPwRefReg0 - r), or a tensor operand (kPwRefOp0 - i, i indexing
    // node.inputs). `kind` selects the step family:
    //   kPwKindBinary: acc = pwBinary(srcA, srcB, code)
    //   kPwKindUnary:  acc = pwUnary(srcA, code, p0, p1)
    //   kPwKindAct:    acc = pwAct(srcA, code, p0, p1)
    //   kPwKindSelect: acc = srcA != 0 ? srcB : srcC
    //   kPwKindLoad:   acc = srcA
    // A dst >= 0 additionally copies the step result to that register. Every tensor operand
    // broadcasts against the output shape by the same NumPy-style stride computation regardless of
    // the bcast/bcastSrc fields (those drive the GPU kernels' fast paths; this reference reads no
    // per-mode special case). pw_outs lists, per extra output stream, the step whose value it
    // stores (kPwRefEntry stores the entry value itself).
    void applyPwEpilogue(const Node &node, ExecContext &ctx) {
        RtTensor    &Y      = ctx.t(node.outputs[0]);
        const Shape &out    = Y.shape;
        int64_t      n      = numElements(out);
        float       *y      = Y.host.f32();
        size_t       rank   = out.size();
        const auto  &st     = node.attr.getints("pw_steps");
        const auto  &pr     = node.attr.getfloats("pw_params");
        const auto  &po     = node.attr.getints("pw_outs");
        int          nSteps = (int) (st.size() / 8);

        // NumPy-style broadcast strides for operand tensor `t` of shape `s` against the rank-`rank`
        // output. `s` is right-aligned to the output axes (leading axes it lacks, off = rank -
        // s.size(), are treated as size 1); an axis of extent 1 gets stride 0 so every output index
        // along it re-reads the same operand element, while other axes get the operand's own
        // row-major stride. A valid broadcast requires each right-aligned operand axis to be 1 or
        // equal to the output extent (and the operand rank cannot exceed the output rank); a
        // non-conforming extent is a malformed graph that the index reassembly below would turn into
        // an out-of-bounds read, so it is rejected here (once per operand, before the element loop).
        auto broadcastStrides = [&](const Shape &s, TensorId t) {
            if (s.size() > rank)
            {
                throw Error(Status::InvalidArgument, "FusedPointwise (" + node.name + ") operand tensor " + std::to_string(t) + " shape " + shapeStr(s) + " has higher rank than output " + shapeStr(out));
            }
            std::vector<int64_t> ob(rank, 0);
            int64_t              stride = 1;
            size_t               off    = rank - s.size();
            for (int i = (int) rank - 1; i >= 0; --i)
            {
                int64_t d = (i < (int) off) ? 1 : s[i - off];
                if (d != 1 && d != out[i])
                {
                    throw Error(Status::InvalidArgument, "FusedPointwise (" + node.name + ") operand tensor " + std::to_string(t) + " shape " + shapeStr(s) + " is not broadcast-compatible with output " + shapeStr(out) + " (axis " + std::to_string(i) + ": " + std::to_string(d) + " vs " + std::to_string(out[i]) + ")");
                }
                ob[i] = (d == 1) ? 0 : stride;
                stride *= d;
            }
            return ob;
        };
        // Hoist per-source operand pointers and broadcast strides out of the element loop; a
        // non-operand source keeps a null pointer and resolves from acc/entry/registers instead.
        // `slot` is the source's index into the BroadcastWalk built below, which carries the operand
        // offsets across the element sweep so the loop never unravels `lin` per source.
        struct SrcRef {
            int                  ref  = kPwRefNone;
            const float         *p    = nullptr;
            size_t               slot = 0;
            std::vector<int64_t> ob;
        };
        std::vector<SrcRef> src((size_t) nSteps * 3);
        for (int s = 0; s < nSteps; ++s)
        {
            for (int f = 0; f < 3; ++f)
            {
                SrcRef &r = src[(size_t) s * 3 + f];
                r.ref     = (int) st[s * 8 + 2 + f];
                if (r.ref <= kPwRefOp0)
                {
                    const RtTensor &O = ctx.t(node.inputs[kPwRefOp0 - r.ref]);
                    // The pool holds every pw operand as valid fp32 (the session decodes fp16
                    // initializers at load; activations are fp32 by construction). Anything else
                    // here is a wrong-payload bug upstream — fail loudly rather than read
                    // reinterpreted or missing bytes as values.
                    if (!O.hostValid || O.dtype != DType::Float32 || O.host.bytes.size() < 4)
                    {
                        throw Error(Status::RuntimeError, "FusedPointwise operand tensor " + std::to_string(node.inputs[kPwRefOp0 - r.ref]) + " (" + node.name + ") has no fp32 host payload");
                    }
                    r.p               = O.host.f32();
                    r.ob              = broadcastStrides(O.shape, node.inputs[kPwRefOp0 - r.ref]);
                }
            }
        }
        // One walker slot per operand source; `src` is sized up front so the borrowed `ob` pointers
        // stay valid for the walker's lifetime.
        std::vector<const int64_t *> srcStrides;
        for (SrcRef &r: src)
        {
            if (r.p)
            {
                r.slot = srcStrides.size();
                srcStrides.push_back(r.ob.data());
            }
        }
        cpu::BroadcastWalk walk(out, std::move(srcStrides));
        walk.seek(0);

        // Extra output streams share the unit's output shape (the fuser only exports same-shape
        // step values); allocate them here since the producing op only writes outputs[0].
        int    numOuts = std::min((int) po.size(), (int) kPwMaxOuts);
        float *outPtr[kPwMaxOuts] = {};
        int    outStep[kPwMaxOuts] = {};
        for (int o = 0; o < numOuts; ++o)
        {
            outPtr[o]  = cpu::allocOut(ctx.t(node.outputs[1 + o]), out);
            outStep[o] = (int) po[o];
        }

        for (int64_t lin = 0; lin < n; ++lin, walk.next())
        {
            float entry = y[lin];
            float acc   = entry;
            float reg[kPwMaxRegs] = {};
            for (int o = 0; o < numOuts; ++o)
            {
                if (outStep[o] == kPwRefEntry)
                {
                    outPtr[o][lin] = entry;
                }
            }
            auto value = [&](const SrcRef &r) -> float {
                if (r.p)
                {
                    // The walker already holds this operand's broadcast source offset for `lin`
                    // (0 strides collapse its broadcast axes onto element 0).
                    return r.p[walk.offset(r.slot)];
                }
                if (r.ref == kPwRefAcc)
                {
                    return acc;
                }
                if (r.ref == kPwRefEntry)
                {
                    return entry;
                }
                if (r.ref <= kPwRefReg0 && r.ref > kPwRefReg0 - kPwMaxRegs)
                {
                    return reg[kPwRefReg0 - r.ref];
                }
                return 0.f;
            };
            for (int s = 0; s < nSteps; ++s)
            {
                int   kind = (int) st[s * 8 + 0];
                int   code = (int) st[s * 8 + 1];
                int   dst  = (int) st[s * 8 + 5];
                float p0   = pr[s * 2 + 0];
                float p1   = pr[s * 2 + 1];
                float va   = value(src[(size_t) s * 3 + 0]);
                if (kind == kPwKindBinary)
                {
                    acc = pwBinary(va, value(src[(size_t) s * 3 + 1]), code);
                } else if (kind == kPwKindUnary)
                {
                    acc = pwUnary(va, code, p0, p1);
                } else if (kind == kPwKindAct)
                {
                    acc = pwAct(va, code, p0, p1);
                } else if (kind == kPwKindSelect)
                {
                    acc = va != 0.f ? value(src[(size_t) s * 3 + 1]) : value(src[(size_t) s * 3 + 2]);
                } else
                {
                    acc = va; // kPwKindLoad
                }
                if (dst >= 0 && dst < kPwMaxRegs)
                {
                    reg[dst] = acc;
                }
                for (int o = 0; o < numOuts; ++o)
                {
                    if (outStep[o] == s)
                    {
                        outPtr[o][lin] = acc;
                    }
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
