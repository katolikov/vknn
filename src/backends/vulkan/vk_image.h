// Minimal 2D fp16 texture (RGBA16F) used to test the image-backend hypothesis: store NC4HW4 as a
// texture (each texel = 4 channels) so reads hit the GPU's texture cache. Kept GENERAL-layout and
// storage-typed so shaders use imageLoad/imageStore without samplers or layout transitions.
#pragma once
#include "vk_command.h"
#include "vk_context.h"

namespace vx {
namespace vk {

class Image {
 public:
  Image(VulkanContext& ctx, int width, int height);
  ~Image();
  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;

  VkImageView view() const { return view_; }
  int width() const { return w_; }
  int height() const { return h_; }

  // Transition UNDEFINED -> GENERAL (run once before use).
  void toGeneral(CommandRunner& runner);
  // host fp16 RGBA texels (w*h*4 uint16) -> image, and back.
  void upload(CommandRunner& runner, const uint16_t* rgba);
  void download(CommandRunner& runner, uint16_t* rgba);

  // Whether the device supports RGBA16F storage images at all.
  static bool supported(VulkanContext& ctx);

 private:
  VulkanContext& ctx_;
  int w_, h_;
  VkImage img_ = VK_NULL_HANDLE;
  VkDeviceMemory mem_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
};

}  // namespace vk
}  // namespace vx
