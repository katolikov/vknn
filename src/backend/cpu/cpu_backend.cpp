#include "cpu_backend.h"
#include "vknn/logging.h"
#include "vknn/profiler.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace vknn {

    CpuOpRegistry &CpuOpRegistry::instance() {
        static CpuOpRegistry r;
        return r;
    }

    namespace cpu {
        float *allocOut(RtTensor &rt, const Shape &shape) {
            rt.shape = shape;
            rt.dtype = DType::Float32;
            rt.host.resizeElems(elemCount(shape), DType::Float32);
            rt.hostValid   = true;
            rt.deviceValid = false;
            return rt.host.f32();
        }
        int64_t *allocOutI64(RtTensor &rt, const Shape &shape) {
            rt.shape = shape;
            rt.dtype = DType::Int64;
            rt.host.resizeElems(elemCount(shape), DType::Int64);
            rt.hostValid   = true;
            rt.deviceValid = false;
            return rt.host.i64();
        }
        /// Apply a fused activation to the `n` contiguous elements at `p` in place. `lo`/`hi` are the
        /// clamp bounds and are read only by ActType::Clip; the other cases carry their bounds in the
        /// formula. Unrecognized activations (default) leave the buffer untouched (identity).
        void applyAct(float *p, int64_t n, ActType act, float lo, float hi) {
            switch (act)
            {
                case ActType::Relu:
                    // max(x, 0).
                    for (int64_t i = 0; i < n; ++i)
                    {
                        p[i] = p[i] > 0 ? p[i] : 0;
                    }
                    break;
                case ActType::Relu6:
                    // clamp(x, 0, 6): ReLU with a hard ceiling of 6, common in mobile CNNs.
                    for (int64_t i = 0; i < n; ++i)
                    {
                        float v = p[i];
                        p[i]    = v < 0 ? 0 : (v > 6 ? 6 : v);
                    }
                    break;
                case ActType::Clip:
                    // clamp(x, lo, hi) using the runtime-supplied bounds (ONNX Clip min/max).
                    for (int64_t i = 0; i < n; ++i)
                    {
                        float v = p[i];
                        p[i]    = v < lo ? lo : (v > hi ? hi : v);
                    }
                    break;
                case ActType::HardSwish:
                    // x * relu6(x + 3) / 6, the piecewise-linear approximation of SiLU used by
                    // HardSwish. relu6 is spelled here as min(max(x+3, 0), 6).
                    for (int64_t i = 0; i < n; ++i)
                    {
                        float v = p[i];
                        p[i]    = v * std::min(std::max(v + 3.f, 0.f), 6.f) / 6.f;
                    }
                    break;
                case ActType::SiLU:
                    // x * sigmoid(x) = x / (1 + exp(-x)), a.k.a. Swish.
                    for (int64_t i = 0; i < n; ++i)
                    {
                        p[i] = p[i] / (1.f + std::exp(-p[i]));
                    }
                    break;
                default:
                    break;
            }
        }
        void copyAs(const RtTensor &X, RtTensor &Y, const Shape &shape) {
            Y.shape = shape;
            Y.dtype = X.dtype;
            Y.host.resizeElems(elemCount(shape), X.dtype);
            Y.hostValid   = true;
            Y.deviceValid = false;
            // Pure metadata reshapes preserve element count, so the two byte spans are equal in size;
            // the min() guards against a caller passing a mismatched `shape` by copying only the
            // overlap rather than reading or writing past either buffer, and zeroing whatever of the
            // destination the overlap leaves uncovered.
            const size_t overlap = std::min(Y.host.bytes.size(), X.host.bytes.size());
            std::memcpy(Y.host.bytes.data(), X.host.bytes.data(), overlap);
            if (overlap < Y.host.bytes.size())
            {
                std::memset(Y.host.bytes.data() + overlap, 0, Y.host.bytes.size() - overlap);
            }
        }
    } // namespace cpu

    // --------------------------- CpuSegment ---------------------------
    /// A contiguous run of graph nodes executed on the CPU reference path. Construction eagerly
    /// instantiates one CpuOp per node (parallel to `nodeIdx`), so run() is a straight-line dispatch
    /// with no per-node lookup.
    class CpuSegment: public Segment {
      public:
        CpuSegment(const std::vector<int> &idx, Graph &g): g_(g) {
            nodeIdx = idx;
            for (int i: idx)
            {
                auto op = CpuOpRegistry::instance().create(g.nodes[i].type);
                ops_.push_back(std::move(op));
            }
        }
        void run(ExecContext &ctx) override {
            for (size_t k = 0; k < nodeIdx.size(); ++k)
            {
                const Node &node = ctx.graph->nodes[nodeIdx[k]];
                if (ctx.config && ctx.config->debugSegments)
                {
                    std::string sh;
                    for (auto t: node.inputs)
                    {
                        sh += std::to_string(t) + ":[";
                        if (t >= 0)
                        {
                            const RtTensor &rt = ctx.t(t);
                            for (auto d: rt.shape)
                            {
                                sh += std::to_string(d) + ",";
                            }
                            sh += rt.hostValid ? "]h " : "]NOHOST ";
                        } else
                        {
                            sh += "?] ";
                        }
                    }
                    VKNN_INFO << "  cpuop " << opTypeName(node.type) << " '" << node.name << "' ins=" << sh;
                }
                CpuOp *op = ops_[k].get();
                if (!op)
                {
                    throw Error(Status::Unsupported, std::string("no CPU kernel for op ") + opTypeName(node.type) + " (" + node.name + ")");
                }
                auto t0 = std::chrono::high_resolution_clock::now();
                op->run(node, ctx);
                // A producer may carry a fused pointwise-chain epilogue (attr pw_steps); apply it
                // in-place after the op runs. FusedPointwise applies its own chain, so skip it here.
                if (node.type != OpType::FusedPointwise && node.attr.has("pw_steps"))
                {
                    applyPwEpilogue(node, ctx);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                if (ctx.profiler && ctx.profiler->enabled())
                {
                    OpRecord r;
                    r.name     = node.name;
                    r.type     = node.type;
                    r.backend  = backend ? backend->name() : "CPU";
                    r.cpuMs    = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    r.fellBack = isFallback;
                    ctx.profiler->add(r);
                }
            }
        }

      private:
        Graph                              &g_;
        std::vector<std::unique_ptr<CpuOp>> ops_;
    };

    // --------------------------- CpuBackend ---------------------------
    class CpuBackend: public Backend {
      public:
        // Config is unused by the CPU backend (no creation-time device settings) but accepted to match
        // the backend factory signature.
        explicit CpuBackend(const Config & = {}) {
        }
        BackendKind kind() const override {
            return BackendKind::Cpu;
        }
        const char *name() const override {
            return "CPU";
        }
        bool available() const override {
            return true;
        }
        bool supports(OpType t, DType dt) const override {
            auto &r = CpuOpRegistry::instance();
            if (!r.has(t))
            {
                return false;
            }
            // A registered kernel exists; the CPU reference path additionally requires the tensor to
            // be one of the dtypes the kernels operate on (fp32 activations, int64/int32 index/shape
            // tensors). Other dtypes fall through to no CPU support.
            return dt == DType::Float32 || dt == DType::Int64 || dt == DType::Int32;
        }
        std::unique_ptr<Segment> compileSegment(const std::vector<int> &idx, Graph &g, const Config &) override {
            auto s           = std::make_unique<CpuSegment>(idx, g);
            s->backend       = this;
            s->compiledGraph = &g;
            return s;
        }
    };

    VKNN_REGISTER_BACKEND(BackendKind::Cpu, CpuBackend);

} // namespace vknn
