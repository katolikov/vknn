// vx_ion_zerocopy - demonstrate Exynos ION / DMA-BUF zero-copy import into Vulkan.
//
// Mode A (library-allocated): IonBuffer::alloc -> CPU memcpy -> import fd into Vulkan ->
//   GPU compute (add) reads the ION buffer directly (no staging) -> verify vs CPU.
// Mode B (user-supplied fd): wrap an existing dma-buf fd -> import -> verify.
// Both are validated against the staged (regular VkBuffer + upload) path.
#include <cstdio>
#if defined(VXRT_ENABLE_VULKAN)
#include <cmath>
#include <unistd.h>
#include <vector>
#include "vx/ion.h"
#include "vx/logging.h"
#include "backends/vulkan/vk_buffer.h"
#include "backends/vulkan/vk_command.h"
#include "backends/vulkan/vk_pipeline.h"

using namespace vx;

static bool runAdd(vk::VulkanContext& ctx, vk::CommandRunner& runner, vk::ComputePipeline& add,
                   VkBuffer a, VkBuffer b, VkBuffer c, uint32_t n) {
  uint32_t count = n;
  runner.oneShot([&](VkCommandBuffer cmd) {
    add.dispatch(cmd, {a, b, c}, &count, sizeof(count), (n + 255) / 256);
  });
  return true;
}

int main() {
  vk::VulkanContext ctx;
  if (!ctx.initialized()) {
    fprintf(stderr, "no Vulkan\n");
    return 1;
  }
  vk::CommandRunner runner(ctx);
  vk::ComputePipeline add(ctx, "add", 3, sizeof(uint32_t));

  const uint32_t N = 1 << 16;
  const size_t bytes = (size_t)N * 4;
  std::vector<float> A(N), B(N);
  for (uint32_t i = 0; i < N; ++i) {
    A[i] = (float)i * 0.01f;
    B[i] = (float)(N - i) * 0.02f;
  }

  // shared operand B + output, used by all paths
  vk::Buffer bBuf(ctx, bytes), cBuf(ctx, bytes, vk::MemPref::kReadback);
  bBuf.upload(B.data(), bytes);
  std::vector<float> got(N);
  auto verify = [&](const char* tag) {
    cBuf.download(got.data(), bytes);
    double maxErr = 0;
    for (uint32_t i = 0; i < N; ++i)
      maxErr = std::max(maxErr, (double)std::fabs(got[i] - (A[i] + B[i])));
    bool ok = maxErr < 1e-3;
    printf("  [%s] maxAbsErr=%.3e => %s\n", tag, maxErr, ok ? "PASS" : "FAIL");
    return ok;
  };

  // ---------- Staged reference path ----------
  printf("== staged path (baseline) ==\n");
  {
    vk::Buffer aStg(ctx, bytes);
    aStg.upload(A.data(), bytes);
    runAdd(ctx, runner, add, aStg.handle(), bBuf.handle(), cBuf.handle(), N);
    verify("staged");
  }

  bool importedOk = false;
  int savedFd = -1;

  // ---------- Mode A: library-allocated ION ----------
  printf("== Mode A: library-allocated ION (dma-heap) ==\n");
  auto ionA = IonBuffer::alloc(bytes);
  if (ionA && ionA->data()) {
    memcpy(ionA->data(), A.data(), bytes);  // CPU fills the ION buffer
    std::unique_ptr<vk::Buffer> imp(vk::Buffer::importDmaBufFd(ctx, ionA->fd(), bytes));
    if (imp) {
      importedOk = true;
      savedFd = ionA->fd();
      runAdd(ctx, runner, add, imp->handle(), bBuf.handle(), cBuf.handle(), N);
      printf("  imported dma-buf fd %d into Vulkan (zero-copy, no staging)\n", ionA->fd());
      verify("ion-mode-A");
    } else {
      printf(
          "  dma-buf import into Vulkan unsupported on this driver -> would fall back to "
          "staging\n");
    }
  } else {
    printf("  dma-heap allocation unavailable -> ION zero-copy not testable here\n");
  }

  // ---------- Mode B: user-supplied fd ----------
  printf("== Mode B: user-supplied fd (wrapFd) ==\n");
  if (importedOk && savedFd >= 0) {
    int dupFd = ::dup(savedFd);  // simulate an fd from elsewhere
    auto ionB = IonBuffer::wrapFd(dupFd, bytes, /*takeOwnership=*/true);
    // data already present in the shared buffer; re-write to be explicit
    if (ionB->data()) memcpy(ionB->data(), A.data(), bytes);
    std::unique_ptr<vk::Buffer> impB(vk::Buffer::importDmaBufFd(ctx, dupFd, bytes));
    if (impB) {
      runAdd(ctx, runner, add, impB->handle(), bBuf.handle(), cBuf.handle(), N);
      verify("ion-mode-B");
    }
  } else {
    printf("  (skipped: import path not available)\n");
  }

  printf("== ION zero-copy demo %s ==\n",
         importedOk ? "OK" : "(import unsupported; see LIMITATIONS.md)");
  return 0;
}
#else
int main() {
  printf("vx_ion_zerocopy: built without Vulkan (Android-only feature)\n");
  return 0;
}
#endif
