#include "vk_image.h"
#include "vk_buffer.h"

namespace vknn { namespace vk {

    static const VkFormat kFmt = VK_FORMAT_R16G16B16A16_SFLOAT;

    bool Image::supported(VulkanContext &ctx) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice(), kFmt, &p);
        return (p.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
    }

    Image::Image(VulkanContext &ctx, int width, int height): ctx_(ctx), w_(width), h_(height) {
        VkImageCreateInfo ici {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = kFmt;
        ici.extent        = {(uint32_t) w_, (uint32_t) h_, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(ctx_.device(), &ici, nullptr, &img_));

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(ctx_.device(), img_, &req);
        uint32_t    typeIdx = 0;
        const auto &mp      = ctx_.memProps();
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        {
            if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            {
                typeIdx = i;
                break;
            }
        }
        VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = typeIdx;
        VK_CHECK(vkAllocateMemory(ctx_.device(), &ai, nullptr, &mem_));
        VK_CHECK(vkBindImageMemory(ctx_.device(), img_, mem_, 0));

        VkImageViewCreateInfo vci {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image            = img_;
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vci.format           = kFmt;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(ctx_.device(), &vci, nullptr, &view_));
    }

    Image::~Image() {
        if (view_)
        {
            vkDestroyImageView(ctx_.device(), view_, nullptr);
        }
        if (img_)
        {
            vkDestroyImage(ctx_.device(), img_, nullptr);
        }
        if (mem_)
        {
            vkFreeMemory(ctx_.device(), mem_, nullptr);
        }
    }

    void Image::toGeneral(CommandRunner &runner) {
        runner.oneShot([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier b {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
            b.image            = img_;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b.srcAccessMask    = 0;
            b.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        });
    }

    void Image::upload(CommandRunner &runner, const uint16_t *rgba) {
        size_t bytes = (size_t) w_ * h_ * 4 * 2;
        Buffer staging(ctx_, bytes, MemPref::kAuto);
        staging.upload(rgba, bytes);
        runner.oneShot([&](VkCommandBuffer cmd) {
            VkBufferImageCopy c {};
            c.bufferRowLength   = w_;
            c.bufferImageHeight = h_;
            c.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.imageExtent       = {(uint32_t) w_, (uint32_t) h_, 1};
            vkCmdCopyBufferToImage(cmd, staging.handle(), img_, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        });
    }

    void Image::download(CommandRunner &runner, uint16_t *rgba) {
        size_t bytes = (size_t) w_ * h_ * 4 * 2;
        Buffer staging(ctx_, bytes, MemPref::kReadback);
        runner.oneShot([&](VkCommandBuffer cmd) {
            VkBufferImageCopy c {};
            c.bufferRowLength   = w_;
            c.bufferImageHeight = h_;
            c.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.imageExtent       = {(uint32_t) w_, (uint32_t) h_, 1};
            vkCmdCopyImageToBuffer(cmd, img_, VK_IMAGE_LAYOUT_GENERAL, staging.handle(), 1, &c);
        });
        staging.download(rgba, bytes);
    }

}} // namespace vknn::vk
