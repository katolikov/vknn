// VkOpEnv bridges into VulkanBackend and the op-registry singleton; this TU sees both sides of the
// VkOpEnv <-> VulkanBackend circularity complete.
#include "vk_op_env.h"
#include "vk_backend.h"
#include "vk_weight_cache.h"

namespace vknn {

    // ============================ VkOpRegistry ============================
    VkOpRegistry &VkOpRegistry::instance() {
        static VkOpRegistry r;
        return r;
    }

    // ============================ VkOpEnv ============================
    bool VkOpEnv::reuseTuned(const std::string &sig, int &out) const {
        if (!weights)
        {
            return false;
        }
        int level  = -1;
        int cached = weights->tuned(sig, 0, &level);
        if (level >= 0 && (tuning == Tuning::None || level >= (int) tuning))
        {
            out = cached;
            return true;
        }
        return false;
    }

    std::shared_ptr<vk::ComputePipeline> VkOpEnv::pipeline(const std::string &shaderName, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &specData, uint32_t requiredSubgroupSize) const {
        return backend->sharedPipeline(shaderName, numBuffers, pushConstBytes, specData, cache ? cache->handle() : VK_NULL_HANDLE, requiredSubgroupSize);
    }

    std::shared_ptr<vk::Buffer> VkOpEnv::uploadPooled(const void *data, size_t bytes) const {
        return backend->uploadPooled(data, bytes);
    }

    std::shared_ptr<vk::Buffer> VkOpEnv::acquireWeight(const std::string &key, bool fp16, std::function<std::shared_ptr<vk::Buffer>()> make) const {
        return backend->acquireWeight(key, fp16, make);
    }

    std::shared_ptr<vk::Buffer> VkOpEnv::uploadWeightDeviceOnly(const void *src, size_t srcBytes, size_t bufferBytes) const {
        return backend->stageWeightToDevice(src, srcBytes, bufferBytes);
    }

} // namespace vknn
