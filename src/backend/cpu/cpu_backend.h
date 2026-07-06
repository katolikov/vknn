// CPU reference backend + scalar/NEON operator registry.
//
// Adding a CPU op (see docs/ADDING_AN_OPERATOR.md):
//   1. subclass CpuOp, implement run().
//   2. VKNN_REGISTER_CPU_OP(OpType::Foo, FooCpuOp);
// No edits to core dispatch are required.
#pragma once
#include "vknn/backend.h"
#include <functional>
#include <map>
#include <memory>

namespace vknn {

    /// One operator implementation for the CPU backend. `run` reads inputs and writes outputs
    /// (host buffers, NCHW canonical). Shape inference is the op's responsibility.
    ///
    /// This is the numeric reference backend: results here define correctness, and it also serves
    /// as the terminal fallback for any node the GPU backend declines. Kernels compute in scalar
    /// (optionally NEON) fp32/int64 with host-visible buffers.
    class CpuOp {
      public:
        virtual ~CpuOp()                                     = default;
        virtual void run(const Node &node, ExecContext &ctx) = 0;
    };

    using CpuOpFactory = std::function<std::unique_ptr<CpuOp>()>;

    /// Process-wide map from `OpType` to a factory that builds that op's CPU kernel. Populated at
    /// static-init time by the `VKNN_REGISTER_CPU_OP` registrars below, then queried by the backend
    /// when planning each node.
    class CpuOpRegistry {
      public:
        static CpuOpRegistry &instance();
        /// Register (or replace) the factory for op type `t`.
        void reg(OpType t, CpuOpFactory f) {
            factories_[t] = std::move(f);
        }
        /// True if a CPU kernel is registered for `t`; used to decide GPU-vs-CPU placement and
        /// whether a fallback path exists.
        bool has(OpType t) const noexcept {
            return factories_.count(t) > 0;
        }
        /// Instantiate a fresh kernel for `t`, or nullptr when none is registered. Each call builds
        /// a new instance so per-node kernel state never aliases across the graph.
        std::unique_ptr<CpuOp> create(OpType t) const {
            auto it = factories_.find(t);
            return it == factories_.end() ? nullptr : it->second();
        }

      private:
        std::map<OpType, CpuOpFactory> factories_;
    };

    /// Static-init hook: constructing one registers `f` for op type `t`. A file-scope instance
    /// (emitted by `VKNN_REGISTER_CPU_OP`) runs its constructor before `main`, so every linked op
    /// self-registers without any central dispatch table to edit.
    struct CpuOpRegistrar {
        CpuOpRegistrar(OpType t, CpuOpFactory f) {
            CpuOpRegistry::instance().reg(t, std::move(f));
        }
    };
#define VKNN_REGISTER_CPU_OP(OPTYPE, CLASS)                            \
    static ::vknn::CpuOpRegistrar _vx_cpuop_reg_##CLASS(OPTYPE, []() { \
        return std::unique_ptr<::vknn::CpuOp>(new CLASS());            \
    })

    // ---- helpers shared by CPU ops ----
    namespace cpu {
        /// Size `rt`'s host buffer to `shape`, mark its host copy valid, and hand back a typed
        /// fp32 pointer to element 0. The op writes its result straight through this pointer.
        float *allocOut(RtTensor &rt, const Shape &shape);
        /// Int64 counterpart of `allocOut`, for ops emitting index/shape tensors (Shape, ArgMax,
        /// NonZero, …) rather than fp32 activations.
        int64_t *allocOutI64(RtTensor &rt, const Shape &shape);
        /// Fold a fused activation over `n` contiguous fp32 elements in place. `lo`/`hi` supply the
        /// runtime clamp bounds for ActType::Clip (ONNX Clip min/max); activations whose bounds are
        /// intrinsic (Relu, Relu6, HardSwish, SiLU) ignore them.
        void applyAct(float *p, int64_t n, ActType act, float lo, float hi);
        /// Reinterpret X under a new `shape` with identical dtype and byte layout, copying its raw
        /// bytes into Y. Backs the pure metadata reshapes (Reshape/Flatten/Squeeze/Unsqueeze) whose
        /// element order is unchanged, so no per-element conversion is needed.
        void copyAs(const RtTensor &X, RtTensor &Y, const Shape &shape);
    } // namespace cpu

    /// Apply a fused pointwise-epilogue chain in place to `node.outputs[0]`, which must already hold
    /// the head/primary result. The chain is carried on the node as parallel attributes: `pw_steps`
    /// (the ordered op codes) and `pw_params` (their scalar operands). Shared by the standalone
    /// FusedPointwise CPU op and the executor hook that runs an epilogue attached to a producer node,
    /// keeping both paths byte-identical to the reference.
    void applyPwEpilogue(const Node &node, ExecContext &ctx);

} // namespace vknn
