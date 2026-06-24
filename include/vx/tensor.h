// vxrt — tensor descriptor (IR) and runtime tensor (execution-time storage).
#pragma once
#include <cstring>
#include <memory>
#include <vector>
#include "vx/common.h"
#include "vx/dtype.h"
#include "vx/op.h"
#include "vx/tensor_format.h"

namespace vx {

/// Compile-time description of a tensor in the graph.
struct TensorDesc {
  std::string name;
  Shape shape;  // logical NCHW shape (may have dynamic dims as -1)
  DType dtype = DType::kFloat32;
  TensorFormat format = TensorFormat::kNCHW;
  bool isInput = false;
  bool isOutput = false;
  bool isInitializer = false;
};

/// Host-side raw bytes (initializers, I/O, CPU compute results). Logical layout = NCHW.
struct HostBuffer {
  std::vector<uint8_t> bytes;
  void resizeElems(int64_t n, DType dt) { bytes.assign((size_t)n * dtypeSize(dt), 0); }
  float* f32() { return reinterpret_cast<float*>(bytes.data()); }
  const float* f32() const { return reinterpret_cast<const float*>(bytes.data()); }
  int64_t* i64() { return reinterpret_cast<int64_t*>(bytes.data()); }
  const int64_t* i64() const { return reinterpret_cast<const int64_t*>(bytes.data()); }
};

/// Forward declaration: the Vulkan backend attaches its device storage here as an
/// opaque handle, keeping the core free of Vulkan types.
struct DeviceStorage;

/// Runtime tensor: may be resident on host and/or device. Tracks validity + the
/// device layout/dtype so cross-backend handoff can convert correctly.
struct RtTensor {
  TensorId id = kNoTensor;
  Shape shape;
  DType dtype = DType::kFloat32;

  // ---- host residency (canonical NCHW, fp32 for compute/IO) ----
  HostBuffer host;
  bool hostValid = false;

  // ---- device residency (managed by a backend) ----
  std::shared_ptr<DeviceStorage> device;  // null until a backend allocates it
  TensorFormat deviceFormat = TensorFormat::kUnknown;
  DType deviceDtype = DType::kFloat32;
  bool deviceValid = false;

  int64_t elems() const { return numElements(shape); }
  void allocHost() {
    host.resizeElems(elems(), dtype);
    hostValid = true;
  }
};

}  // namespace vx
