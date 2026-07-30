#include "vk_pipeline.h"
#include <cstdio>
#include <fstream>

namespace vknn { namespace vk {

    namespace {
        // Every kernel exposes a single compute entry point named "main" (glslang default).
        constexpr const char *kEntryPoint = "main";
        // All bindings are storage buffers in the compute stage (the engine binds no images/samplers here).
        constexpr VkDescriptorType kBindingType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // The bits enum (not VkShaderStageFlags) so it also assigns to VkPipelineShaderStageCreateInfo::stage.
        constexpr VkShaderStageFlagBits kComputeStage = VK_SHADER_STAGE_COMPUTE_BIT;
    } // namespace

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
        // Two-call idiom: query the size, then read. A driver may legally report a smaller size on the
        // second call, so the buffer is shrunk to what was actually written.
        size_t sz = 0;
        if (vkGetPipelineCacheData(ctx_.device(), cache_, &sz, nullptr) != VK_SUCCESS || sz == 0)
        {
            return {};
        }
        std::vector<char> data(sz);
        if (vkGetPipelineCacheData(ctx_.device(), cache_, &sz, data.data()) != VK_SUCCESS)
        {
            return {};
        }
        data.resize(sz);
        return data;
    }

    size_t PipelineCache::currentBytes() const noexcept {
        size_t sz = 0;
        if (vkGetPipelineCacheData(ctx_.device(), cache_, &sz, nullptr) != VK_SUCCESS)
        {
            return 0;
        }
        return sz;
    }

    PipelineCache::~PipelineCache() {
        if (cache_)
        {
            vkDestroyPipelineCache(ctx_.device(), cache_, nullptr);
        }
    }

    // ----------------------------- ComputePipeline -----------------------------
    ComputePipeline::ComputePipeline(VulkanContext &ctx, const std::string &shaderName, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &specData, VkPipelineCache cache, uint32_t requiredSubgroupSize):
        ctx_(ctx), numBuffers_(numBuffers), name_(shaderName) {
        // A throwing constructor does not run the destructor, so reclaim any handle already created
        // before letting the exception propagate.
        try
        {
            auto it = embeddedShaders().find(shaderName);
            if (it == embeddedShaders().end())
            {
                throw Error(Status::NotFound, "shader not found: " + shaderName);
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
                binds[i].descriptorType  = kBindingType;
                binds[i].descriptorCount = 1;
                binds[i].stageFlags      = kComputeStage;
            }
            VkDescriptorSetLayoutCreateInfo slci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            slci.bindingCount = numBuffers;
            slci.pBindings    = binds.data();
            // Push descriptors bind buffers inline at dispatch time (no descriptor pool/sets to manage),
            // which is why dispatch() can take raw VkBuffers. Requires VK_KHR_push_descriptor.
            const bool usePush = ctx_.caps().pushDescriptor && ctx_.cmdPushDescriptorSet;
            if (!usePush)
            {
                // dispatch() binds inputs exclusively through vkCmdPushDescriptorSetKHR; there is no
                // descriptor-pool/allocate fallback, so a device without VK_KHR_push_descriptor cannot
                // run this pipeline. Reject it here with the shader named rather than null-calling the
                // missing function pointer on the first dispatch().
                VKNN_ERROR << "shader " << shaderName << " requires VK_KHR_push_descriptor, which this device does not expose";
                throw Error(Status::Unsupported, "shader " + shaderName + " requires VK_KHR_push_descriptor (not available on this device)");
            }
            slci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device(), &slci, nullptr, &setLayout_));

            // Several kernel PC blocks exceed the 128-byte Vulkan-guaranteed minimum; a range
            // above the device cap is invalid at layout creation, so it is rejected here with
            // the shader named rather than left to a driver error (or silence) later.
            if (pushConstBytes > ctx_.caps().maxPushConstantsSize)
            {
                VKNN_ERROR << "shader " << shaderName << " needs " << pushConstBytes << " B of push constants but the device caps at " << ctx_.caps().maxPushConstantsSize << " B";
                throw Error(Status::Unsupported, "push-constant block of shader " + shaderName + " (" + std::to_string(pushConstBytes) + " B) exceeds device maxPushConstantsSize (" +
                                                     std::to_string(ctx_.caps().maxPushConstantsSize) + " B)");
            }
            VkPushConstantRange pcr {};
            pcr.stageFlags = kComputeStage;
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
            stage.stage  = kComputeStage;
            stage.module = module_;
            stage.pName  = kEntryPoint;
            if (!specData.empty())
            {
                stage.pSpecializationInfo = &specInfo;
            }
            // Pinned subgroup width (cooperative-matrix kernels are written for one exact width;
            // a driver free to pick another would compute wrong lane mappings). The caller gates
            // on caps().subgroupSizeControl + requiredSubgroupSizeCompute and the [min,max] range,
            // so an unsupported request never reaches pipeline creation.
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo requiredSize {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
            if (requiredSubgroupSize > 0)
            {
                requiredSize.requiredSubgroupSize = requiredSubgroupSize;
                stage.pNext                       = &requiredSize;
            }

            VkComputePipelineCreateInfo cpci {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage  = stage;
            cpci.layout = layout_;
            VK_CHECK(vkCreateComputePipelines(ctx_.device(), cache, 1, &cpci, nullptr, &pipeline_));
        } catch (...)
        {
            destroy();
            throw;
        }
    }

    void ComputePipeline::destroy() noexcept {
        if (pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(ctx_.device(), pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        if (layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(ctx_.device(), layout_, nullptr);
            layout_ = VK_NULL_HANDLE;
        }
        if (setLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(ctx_.device(), setLayout_, nullptr);
            setLayout_ = VK_NULL_HANDLE;
        }
        if (module_ != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(ctx_.device(), module_, nullptr);
            module_ = VK_NULL_HANDLE;
        }
    }

    ComputePipeline::~ComputePipeline() {
        destroy();
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
            writes[i].descriptorType  = kBindingType;
            writes[i].pBufferInfo     = &infos[i];
        }
        ctx_.cmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, (uint32_t) writes.size(), writes.data());
        if (pcBytes > 0)
        {
            vkCmdPushConstants(cmd, layout_, kComputeStage, 0, pcBytes, pushConst);
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
        // The engine's only dispatch site, so this counter is the complete recorded-dispatch total.
        // The recording segment reads it around each op to attribute the count per node.
        ctx_.dispatchTally().note();
    }

}} // namespace vknn::vk
