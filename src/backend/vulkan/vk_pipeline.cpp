#include "vk_pipeline.h"
#include <cstdio>
#include <fstream>

namespace vknn { namespace vk {

    // ----------------------------- PipelineCache -----------------------------
    PipelineCache::PipelineCache(VulkanContext &ctx, const std::vector<char> &initialData): ctx_(ctx) {
        diskBytes_ = initialData.size();
        if (diskBytes_)
        {
            VKNN_INFO << "Loaded pipeline cache (" << diskBytes_ << " bytes)";
        }
        VkPipelineCacheCreateInfo ci {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = initialData.size();
        ci.pInitialData    = initialData.empty() ? nullptr : initialData.data();
        VK_CHECK(vkCreatePipelineCache(ctx_.device(), &ci, nullptr, &cache_));
    }

    std::vector<char> PipelineCache::getData() const {
        size_t sz = 0;
        vkGetPipelineCacheData(ctx_.device(), cache_, &sz, nullptr);
        std::vector<char> data(sz);
        vkGetPipelineCacheData(ctx_.device(), cache_, &sz, data.data());
        return data;
    }

    PipelineCache::~PipelineCache() {
        if (cache_)
        {
            vkDestroyPipelineCache(ctx_.device(), cache_, nullptr);
        }
    }

    // ----------------------------- ComputePipeline -----------------------------
    ComputePipeline::ComputePipeline(VulkanContext &ctx, const std::string &shaderName, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &specData, VkPipelineCache cache):
        ctx_(ctx), numBuffers_(numBuffers), name_(shaderName) {
        auto it = embeddedShaders().find(shaderName);
        if (it == embeddedShaders().end())
        {
            throw Error(Status::kNotFound, "shader not found: " + shaderName);
        }
        const auto &spv = it->second;

        VkShaderModuleCreateInfo smci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode    = spv.data();
        VK_CHECK(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &module_));

        std::vector<VkDescriptorSetLayoutBinding> binds(numBuffers);
        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            binds[i].binding         = i;
            binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo slci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        slci.bindingCount = numBuffers;
        slci.pBindings    = binds.data();
        bool usePush      = ctx_.caps().pushDescriptor && ctx_.cmdPushDescriptorSet;
        if (usePush)
        {
            slci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        }
        VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device(), &slci, nullptr, &setLayout_));

        VkPushConstantRange pcr {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = pushConstBytes;
        VkPipelineLayoutCreateInfo plci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &setLayout_;
        if (pushConstBytes > 0)
        {
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcr;
        }
        VK_CHECK(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &layout_));

        // Specialization constants: consecutive uint32 at ids 0..N-1.
        std::vector<VkSpecializationMapEntry> specEntries(specData.size());
        for (size_t i = 0; i < specData.size(); ++i)
        {
            specEntries[i] = {(uint32_t) i, (uint32_t) (i * sizeof(uint32_t)), sizeof(uint32_t)};
        }
        VkSpecializationInfo specInfo {};
        specInfo.mapEntryCount = (uint32_t) specEntries.size();
        specInfo.pMapEntries   = specEntries.data();
        specInfo.dataSize      = specData.size() * sizeof(uint32_t);
        specInfo.pData         = specData.data();

        VkPipelineShaderStageCreateInfo stage {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module_;
        stage.pName  = "main";
        if (!specData.empty())
        {
            stage.pSpecializationInfo = &specInfo;
        }

        VkComputePipelineCreateInfo cpci {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage  = stage;
        cpci.layout = layout_;
        VK_CHECK(vkCreateComputePipelines(ctx_.device(), cache, 1, &cpci, nullptr, &pipeline_));
    }

    ComputePipeline::~ComputePipeline() {
        if (pipeline_)
        {
            vkDestroyPipeline(ctx_.device(), pipeline_, nullptr);
        }
        if (layout_)
        {
            vkDestroyPipelineLayout(ctx_.device(), layout_, nullptr);
        }
        if (setLayout_)
        {
            vkDestroyDescriptorSetLayout(ctx_.device(), setLayout_, nullptr);
        }
        if (module_)
        {
            vkDestroyShaderModule(ctx_.device(), module_, nullptr);
        }
    }

    void ComputePipeline::dispatch(VkCommandBuffer cmd, const std::vector<VkBuffer> &buffers, const void *pushConst, uint32_t pcBytes, uint32_t gx, uint32_t gy, uint32_t gz) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        std::vector<VkDescriptorBufferInfo> infos(buffers.size());
        std::vector<VkWriteDescriptorSet>   writes(buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i)
        {
            infos[i]                  = {buffers[i], 0, VK_WHOLE_SIZE};
            writes[i]                 = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstBinding      = (uint32_t) i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &infos[i];
        }
        ctx_.cmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, (uint32_t) writes.size(), writes.data());
        if (pcBytes > 0)
        {
            vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcBytes, pushConst);
        }
        // A 1D dispatch whose x group count exceeds the device limit is illegal and is silently
        // dropped on this driver (producing zeroed output). Spill the overflow into the y dimension;
        // the flat shaders recover the linear id via gl_GlobalInvocationID.x + y*numGroups.x*wgSize.x,
        // so this is a no-op (gy stays 1) for every dispatch that already fits.
        const uint32_t maxX = ctx_.caps().maxWorkGroupCount[0];
        if (gy == 1 && gz == 1 && maxX > 0 && gx > maxX)
        {
            gy = (gx + maxX - 1) / maxX;
            gx = (gx + gy - 1) / gy; // gx <= maxX, gx*gy >= original group count
            VKNN_INFO << "dispatch split for " << name_ << ": gx=" << gx << " gy=" << gy << " (max " << maxX << ")";
        }
        vkCmdDispatch(cmd, gx, gy, gz);
    }

}} // namespace vknn::vk
