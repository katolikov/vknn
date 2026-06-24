// ENN / NPU backend (Samsung Exynos Neural Network).
//
// Status: documented STUB (see docs/adr/0007 + LIMITATIONS.md). The ENN runtime libraries
// are present on-device, but (1) no public ENN C++ headers are available to us and (2) there is
// no on-device NNC compiler (NNC is produced by an offline Samsung SDK tool). A real ENN path
// needs both. This backend therefore:
//   * subclasses Backend and registers in the backend registry (proves the plug-in path),
//   * is selectable via config.backend = ENN with NO changes to core dispatch,
//   * probes (dlopen) the on-device ENN libs and reports what's present,
//   * declines all ops (supports() == false) so execution falls back to Vulkan/CPU, and
//   * returns a clear "unavailable" if asked to compile/run.
// Replacing the body with a real implementation requires only this file.
#include <dlfcn.h>
#include "vx/backend.h"
#include "vx/logging.h"

namespace vx {
namespace {

const char* kEnnLibs[] = {
    "libenn_public_api_cpp.so",
    "libenn_engine.so",
    "libenn_model.so",
    "libenn_user_driver_gpu.so",
    "libenn_user_driver_unified.so",
};

class EnnBackend : public Backend {
 public:
  EnnBackend() { probe(); }

  BackendKind kind() const override { return BackendKind::kEnn; }
  const char* name() const override { return "ENN(stub)"; }

  // Selectable + instantiable (so the plug-in path is exercised). It declines all ops, so
  // the configured fallback (Vulkan/CPU) executes the graph.
  bool available() const override { return true; }

  bool supports(OpType, DType) const override { return false; }

  std::unique_ptr<Segment> compileSegment(const std::vector<int>&, Graph&, const Config&) override {
    throw Error(Status::kUnsupported,
                "ENN backend requires an NNC-format model; no on-device NNC compiler / public "
                "headers available on this device (see LIMITATIONS.md). Use VULKAN or CPU.");
  }

 private:
  void probe() {
    int found = 0;
    std::string present;
    for (const char* lib : kEnnLibs) {
      void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
      if (h) {
        ++found;
        present += std::string(lib) + " ";
        dlclose(h);
      }
    }
    if (found)
      VX_INFO
          << "ENN backend: probed " << found << "/" << (int)(sizeof(kEnnLibs) / sizeof(*kEnnLibs))
          << " runtime libs present [" << present << "] - but NNC model + public headers are "
          << "unavailable on-device, so ENN execution is stubbed (falls back). See LIMITATIONS.md";
    else
      VX_INFO
          << "ENN backend: no ENN runtime libs reachable via dlopen (vendor namespace). Stubbed.";
  }
};

VX_REGISTER_BACKEND(BackendKind::kEnn, EnnBackend);

}  // namespace
}  // namespace vx
