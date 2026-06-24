// vxrt — CPU reference backend + scalar/NEON operator registry.
//
// Adding a CPU op (see docs/ADDING_AN_OPERATOR.md):
//   1. subclass CpuOp, implement run().
//   2. VX_REGISTER_CPU_OP(OpType::kFoo, FooCpuOp);
// No edits to core dispatch are required.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include "vx/backend.h"

namespace vx {

/// One operator implementation for the CPU backend. `run` reads inputs and writes outputs
/// (host buffers, NCHW canonical). Shape inference is the op's responsibility.
class CpuOp {
 public:
  virtual ~CpuOp() = default;
  virtual void run(const Node& node, ExecContext& ctx) = 0;
  // Which dtypes this op supports (for capability/fallback). Default: fp32.
  virtual bool supportsDType(DType dt) const { return dt == DType::kFloat32 || dt == DType::kInt64; }
};

using CpuOpFactory = std::function<std::unique_ptr<CpuOp>()>;

class CpuOpRegistry {
 public:
  static CpuOpRegistry& instance();
  void reg(OpType t, CpuOpFactory f) { factories_[t] = std::move(f); }
  bool has(OpType t) const { return factories_.count(t) > 0; }
  std::unique_ptr<CpuOp> create(OpType t) const {
    auto it = factories_.find(t);
    return it == factories_.end() ? nullptr : it->second();
  }
 private:
  std::map<OpType, CpuOpFactory> factories_;
};

struct CpuOpRegistrar {
  CpuOpRegistrar(OpType t, CpuOpFactory f) { CpuOpRegistry::instance().reg(t, std::move(f)); }
};
#define VX_REGISTER_CPU_OP(OPTYPE, CLASS)                                       \
  static ::vx::CpuOpRegistrar _vx_cpuop_reg_##CLASS(OPTYPE, []() {              \
    return std::unique_ptr<::vx::CpuOp>(new CLASS());                           \
  })

// ---- helpers shared by CPU ops ----
namespace cpu {
// Ensure rt has a host fp32 buffer of `elems` (allocates if needed). Returns float*.
float* allocOut(RtTensor& rt, const Shape& shape);
int64_t* allocOutI64(RtTensor& rt, const Shape& shape);
// Apply fused activation in place.
void applyAct(float* p, int64_t n, ActType act, float lo, float hi);
}  // namespace cpu

}  // namespace vx
