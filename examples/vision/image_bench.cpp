// vknn_image_bench - compares VkImage against SSBO storage for a 1x1 conv on a given GPU.
// Runs the same 1x1 conv four ways on one shape: (a) the SSBO c-tiled kernel at OCB=1, (b) the same
// SSBO kernel at OCB=2 (the register tiling the image kernels use, so the read path is the only
// difference left), (c) an image-backed c8w4 kernel reading via storage-image imageLoad, and (d) a
// sampler2D/texelFetch kernel that goes through the true read-only texture cache (as MNN does).
// Each variant is verified against a CPU reference (cosine similarity) and timed as ms/iter; the
// image paths run only when RGBA16F storage images are supported. The relative speedups isolate
// whether the texture cache beats plain SSBO reads.
//
// All arms are timed round-robin rather than one to completion at a time: a clock or thermal step
// part-way through the run would otherwise land entirely on whichever arm was being timed then.
// Each arm's reported figure is the median over the rounds.
//
// Usage: vknn_image_bench [Cin Cout H W] [iters] [rounds]
// The four shape arguments are all-or-nothing: pass all of Cin/Cout/H/W together or none (defaults
// 256/256/14/14); iters defaults to 200 and rounds to 5, each read only when the preceding argument
// is present.
#include "vknn/dtype.h"
#include "vknn/logging.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#if defined(VKNN_ENABLE_VULKAN)
#include "backend/vulkan/vk_buffer.h"
#include "backend/vulkan/vk_command.h"
#include "backend/vulkan/vk_image.h"
#include "backend/vulkan/vk_pipeline.h"

using namespace vknn;

namespace {

    constexpr int kChannelBlock  = 4;  ///< NC4HW4 packs 4 channels per texel/vec4.
    constexpr int kWorkgroupSize = 64; ///< local_size_x of every kernel raced here.
    constexpr int kWidthTile     = 4;  ///< WTILE: output pixels per thread, in both the SSBO and image kernels.
    constexpr int kOcbScalar     = 1;  ///< OCB=1: the SSBO kernel's classic one-block-per-thread body.
    constexpr int kOcbTiled      = 2;  ///< OCB=2: matches the image kernels' c8 (2 channel-blocks per thread).
    constexpr int kHasResidual   = 0;  ///< HAS_RES specialization: no fused residual in this bench.

    constexpr int kDefaultCin    = 256;
    constexpr int kDefaultCout   = 256;
    constexpr int kDefaultExtent = 14;
    constexpr int kDefaultIters  = 200;
    constexpr int kDefaultRounds = 5;

    /// conv1x1_fp16 binds src/wt/bias/dst; binding 4 (residual) stays unbound because HAS_RES == 0.
    constexpr uint32_t kConvBuffers   = 4;
    constexpr int      kImageBindings = 3;    ///< input, kernel, output.
    constexpr int      kDumpValues    = 6;    ///< leading outputs printed beside the reference when an arm fails to verify.
    constexpr double   kVerifyCosine  = 0.99; ///< an arm below this is reported with a value dump.

    /// Push-constant block of conv1x1_fp16.comp / conv1x1.comp (ConvPC in the conv op).
    struct ConvPushConstants {
        int   N, Cin, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act;
        float actLo, actHi;
    };

    /// Push-constant block of conv1x1_img.comp / conv1x1_tex.comp.
    struct ImagePushConstants {
        int W, H, Cin, Cout;
    };

    /// A compute pipeline over image descriptors, which vk::ComputePipeline (storage buffers only)
    /// cannot express. Owns its handles; destroy() releases them.
    struct ImagePipeline {
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout      layout    = VK_NULL_HANDLE;
        VkShaderModule        module    = VK_NULL_HANDLE;
        VkPipeline            pipeline  = VK_NULL_HANDLE;

        void destroy(VkDevice device) {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyShaderModule(device, module, nullptr);
            vkDestroyPipelineLayout(device, layout, nullptr);
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
        }
    };

    /// Build a push-descriptor compute pipeline whose bindings 0..kImageBindings-1 take the given
    /// descriptor types (combined sampler for the sampled arm, storage image for the imageLoad arm).
    ImagePipeline makeImagePipeline(vk::VulkanContext &ctx, const char *shaderName, const VkDescriptorType *types) {
        ImagePipeline                out {};
        VkDescriptorSetLayoutBinding bindings[kImageBindings] {};
        for (int i = 0; i < kImageBindings; ++i)
        {
            bindings[i].binding         = i;
            bindings[i].descriptorType  = types[i];
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo setInfo {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        setInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        setInfo.bindingCount = kImageBindings;
        setInfo.pBindings    = bindings;
        vkCreateDescriptorSetLayout(ctx.device(), &setInfo, nullptr, &out.setLayout);

        VkPushConstantRange        range {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ImagePushConstants)};
        VkPipelineLayoutCreateInfo layoutInfo {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount         = 1;
        layoutInfo.pSetLayouts            = &out.setLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges    = &range;
        vkCreatePipelineLayout(ctx.device(), &layoutInfo, nullptr, &out.layout);

        const auto              &spv = embeddedShaders().at(shaderName);
        VkShaderModuleCreateInfo moduleInfo {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = spv.size() * sizeof(uint32_t);
        moduleInfo.pCode    = spv.data();
        vkCreateShaderModule(ctx.device(), &moduleInfo, nullptr, &out.module);

        VkComputePipelineCreateInfo pipelineInfo {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipelineInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = out.module;
        pipelineInfo.stage.pName  = "main";
        pipelineInfo.layout       = out.layout;
        vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &out.pipeline);
        return out;
    }

    double median(std::vector<double> v) {
        if (v.empty())
        {
            return -1.0;
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }

} // namespace

int main(int argc, char **argv) {
    int Cin    = argc > 4 ? atoi(argv[1]) : kDefaultCin;
    int Cout   = argc > 4 ? atoi(argv[2]) : kDefaultCout;
    int H      = argc > 4 ? atoi(argv[3]) : kDefaultExtent;
    int W      = argc > 4 ? atoi(argv[4]) : kDefaultExtent;
    int iters  = argc > 5 ? atoi(argv[5]) : kDefaultIters;
    int rounds = argc > 6 ? atoi(argv[6]) : kDefaultRounds;
    int Cinb = (Cin + kChannelBlock - 1) / kChannelBlock, Coutb = (Cout + kChannelBlock - 1) / kChannelBlock;

    vk::VulkanContext ctx;
    if (!ctx.initialized())
    {
        fprintf(stderr, "no Vulkan\n");
        return 1;
    }
    vk::CommandRunner runner(ctx);
    const bool        haveImages = vk::Image::supported(ctx);
    printf("conv1x1  Cin=%d Cout=%d HxW=%dx%d  iters=%d rounds=%d  (RGBA16F storage supported=%d)\n", Cin, Cout, H, W, iters, rounds, (int) haveImages);

    // ---- reference data ----
    std::vector<float> in(Cin * H * W), wt(Cout * Cin), ref(Cout * H * W, 0.f);
    for (int i = 0; i < Cin * H * W; ++i)
    {
        in[i] = ((i * 37) % 23 - 11) * 0.03f;
    }
    for (int i = 0; i < Cout * Cin; ++i)
    {
        wt[i] = ((i * 13) % 17 - 8) * 0.02f;
    }
    for (int oc = 0; oc < Cout; ++oc)
    {
        for (int p = 0; p < H * W; ++p)
        {
            float acc = 0;
            for (int ic = 0; ic < Cin; ++ic)
            {
                acc += wt[oc * Cin + ic] * in[ic * H * W + p];
            }
            ref[oc * H * W + p] = acc;
        }
    }
    auto cosErr = [&](const std::vector<float> &v) {
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < ref.size(); ++i)
        {
            dot += v[i] * ref[i];
            na += v[i] * v[i];
            nb += ref[i] * ref[i];
        }
        return dot / (sqrt(na) * sqrt(nb) + 1e-12);
    };
    auto report = [&](const char *label, double ms, const std::vector<float> &out, double baseMs) {
        double cosine = cosErr(out);
        printf("%-26s %8.4f ms/iter  cosine=%.5f", label, ms, cosine);
        if (baseMs > 0 && ms > 0)
        {
            printf("  => %.2fx vs SSBO(OCB=1) %s", baseMs / ms, ms < baseMs ? "FASTER" : "slower");
        }
        printf("\n");
        if (!(cosine > kVerifyCosine))
        {
            printf("    got:");
            for (int i = 0; i < kDumpValues; ++i)
            {
                printf(" %g", out[i]);
            }
            printf("\n    ref:");
            for (int i = 0; i < kDumpValues; ++i)
            {
                printf(" %g", ref[i]);
            }
            printf("\n");
        }
    };

    // ============ SSBO resources (NC4HW4, fp16 storage) ============
    auto pack = [](const float *s, fp16_t *d, int C, int HW) { // NCHW -> NC4HW4
        int Cb = (C + kChannelBlock - 1) / kChannelBlock;
        for (int cb = 0; cb < Cb; ++cb)
        {
            for (int p = 0; p < HW; ++p)
            {
                for (int l = 0; l < kChannelBlock; ++l)
                {
                    int c                                = cb * kChannelBlock + l;
                    d[(cb * HW + p) * kChannelBlock + l] = floatToHalf(c < C ? s[c * HW + p] : 0.f);
                }
            }
        }
    };
    vk::Buffer inB(ctx, (size_t) Cinb * H * W * kChannelBlock * sizeof(fp16_t));
    vk::Buffer outB(ctx, (size_t) Coutb * H * W * kChannelBlock * sizeof(fp16_t), vk::MemPref::kReadback);
    vk::Buffer wB(ctx, (size_t) Cout * Cinb * kChannelBlock * sizeof(fp16_t));
    vk::Buffer bB(ctx, (size_t) Coutb * kChannelBlock * sizeof(fp16_t));
    {
        std::vector<fp16_t> t((size_t) Cinb * H * W * kChannelBlock);
        pack(in.data(), t.data(), Cin, H * W);
        inB.upload(t.data(), t.size() * sizeof(fp16_t));
    }
    {
        std::vector<fp16_t> t((size_t) Cout * Cinb * kChannelBlock, 0); // weight [oc][icb][4ic]
        for (int oc = 0; oc < Cout; ++oc)
        {
            for (int ic = 0; ic < Cin; ++ic)
            {
                t[(size_t) (oc * Cinb + ic / kChannelBlock) * kChannelBlock + ic % kChannelBlock] = floatToHalf(wt[oc * Cin + ic]);
            }
        }
        wB.upload(t.data(), t.size() * sizeof(fp16_t));
    }
    {
        std::vector<fp16_t> z((size_t) Coutb * kChannelBlock, 0);
        bB.upload(z.data(), z.size() * sizeof(fp16_t));
    }

    ConvPushConstants convPc {1, Cin, H, W, Cout, H, W, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0.f, 0.f};
    const int64_t     HWt = (int64_t) H * W, nTiles = (HWt + kWidthTile - 1) / kWidthTile;

    vk::ComputePipeline ssboScalar(ctx, "conv1x1_fp16", kConvBuffers, sizeof(ConvPushConstants), {kHasResidual, kWidthTile, kOcbScalar});
    vk::ComputePipeline ssboTiled(ctx, "conv1x1_fp16", kConvBuffers, sizeof(ConvPushConstants), {kHasResidual, kWidthTile, kOcbTiled});

    const int64_t ocbGroups    = (Coutb + kOcbTiled - 1) / kOcbTiled;
    const auto    scalarGroups = (uint32_t) ((Coutb * nTiles + kWorkgroupSize - 1) / kWorkgroupSize);
    const auto    tiledGroups  = (uint32_t) ((ocbGroups * nTiles + kWorkgroupSize - 1) / kWorkgroupSize);

    auto recordSsbo = [&](vk::ComputePipeline &pipe, uint32_t groups) {
        VkCommandBuffer cmd = runner.allocate();
        runner.begin(cmd);
        for (int i = 0; i < iters; ++i)
        {
            pipe.dispatch(cmd, {inB.handle(), wB.handle(), bB.handle(), outB.handle()}, &convPc, sizeof(convPc), groups);
            vk::computeBarrier(ctx, cmd);
        }
        runner.end(cmd);
        return cmd;
    };
    VkCommandBuffer scalarCmd = recordSsbo(ssboScalar, scalarGroups);
    VkCommandBuffer tiledCmd  = recordSsbo(ssboTiled, tiledGroups);

    auto readbackSsbo = [&](std::vector<float> &dst) {
        std::vector<fp16_t> t((size_t) Coutb * H * W * kChannelBlock);
        outB.download(t.data(), t.size() * sizeof(fp16_t));
        for (int oc = 0; oc < Cout; ++oc)
        {
            for (int p = 0; p < H * W; ++p)
            {
                dst[oc * H * W + p] = halfToFloat(t[(size_t) ((oc / kChannelBlock) * H * W + p) * kChannelBlock + oc % kChannelBlock]);
            }
        }
    };

    // ============ Image resources (texture-backed c8w4) ============
    std::unique_ptr<vk::Image> inI, kI, outI;
    ImagePipeline              storagePipe {}, samplerPipe {};
    VkSampler                  sampler    = VK_NULL_HANDLE;
    VkCommandBuffer            storageCmd = VK_NULL_HANDLE, samplerCmd = VK_NULL_HANDLE;
    ImagePushConstants         imagePc {W, H, Cin, Cout};
    if (haveImages)
    {
        inI  = std::make_unique<vk::Image>(ctx, W * Cinb, H);
        kI   = std::make_unique<vk::Image>(ctx, Cin, Coutb);
        outI = std::make_unique<vk::Image>(ctx, W * Coutb, H);
        inI->toGeneral(runner);
        kI->toGeneral(runner);
        outI->toGeneral(runner);
        {
            std::vector<fp16_t> t((size_t) W * Cinb * H * kChannelBlock, 0); // input image [w+icb*W, h]
            for (int icb = 0; icb < Cinb; ++icb)
            {
                for (int h = 0; h < H; ++h)
                {
                    for (int w = 0; w < W; ++w)
                    {
                        for (int l = 0; l < kChannelBlock; ++l)
                        {
                            int c                                                            = icb * kChannelBlock + l;
                            t[(size_t) (h * (W * Cinb) + (w + icb * W)) * kChannelBlock + l] = floatToHalf(c < Cin ? in[c * H * W + h * W + w] : 0.f);
                        }
                    }
                }
            }
            inI->upload(runner, t.data());
        }
        {
            std::vector<fp16_t> t((size_t) Cin * Coutb * kChannelBlock, 0); // kernel image [ic, ocb] = 4 oc
            for (int ocb = 0; ocb < Coutb; ++ocb)
            {
                for (int ic = 0; ic < Cin; ++ic)
                {
                    for (int l = 0; l < kChannelBlock; ++l)
                    {
                        int oc                                           = ocb * kChannelBlock + l;
                        t[(size_t) (ocb * Cin + ic) * kChannelBlock + l] = floatToHalf(oc < Cout ? wt[oc * Cin + ic] : 0.f);
                    }
                }
            }
            kI->upload(runner, t.data());
        }

        const VkDescriptorType storageTypes[kImageBindings] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        const VkDescriptorType samplerTypes[kImageBindings] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        storagePipe = makeImagePipeline(ctx, "conv1x1_img", storageTypes);
        samplerPipe = makeImagePipeline(ctx, "conv1x1_tex", samplerTypes);
        {
            VkSamplerCreateInfo si {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            si.magFilter = si.minFilter = VK_FILTER_NEAREST;
            si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            vkCreateSampler(ctx.device(), &si, nullptr, &sampler);
        }

        const int  channelTiles = (Coutb + kOcbTiled - 1) / kOcbTiled, widthTiles = (W + kWidthTile - 1) / kWidthTile;
        const auto imageGroups = (uint32_t) ((channelTiles * widthTiles * H + kWorkgroupSize - 1) / kWorkgroupSize);

        auto recordImage = [&](const ImagePipeline &pipe, const VkDescriptorType *types, VkSampler s) {
            VkCommandBuffer cmd = runner.allocate();
            VkDescriptorImageInfo info[kImageBindings] = {{s, inI->view(), VK_IMAGE_LAYOUT_GENERAL}, {s, kI->view(), VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, outI->view(), VK_IMAGE_LAYOUT_GENERAL}};
            VkWriteDescriptorSet writes[kImageBindings] {};
            for (int i = 0; i < kImageBindings; ++i)
            {
                writes[i]                 = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                writes[i].dstBinding      = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType  = types[i];
                writes[i].pImageInfo      = &info[i];
            }
            runner.begin(cmd);
            for (int i = 0; i < iters; ++i)
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
                ctx.cmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, kImageBindings, writes);
                vkCmdPushConstants(cmd, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ImagePushConstants), &imagePc);
                vkCmdDispatch(cmd, imageGroups, 1, 1);
                vk::computeBarrier(ctx, cmd);
            }
            runner.end(cmd);
            return cmd;
        };
        storageCmd = recordImage(storagePipe, storageTypes, VK_NULL_HANDLE);
        samplerCmd = recordImage(samplerPipe, samplerTypes, sampler);
    }

    auto readbackImage = [&](std::vector<float> &dst) {
        std::vector<fp16_t> t((size_t) W * Coutb * H * kChannelBlock);
        outI->download(runner, t.data());
        for (int oc = 0; oc < Cout; ++oc)
        {
            for (int h = 0; h < H; ++h)
            {
                for (int w = 0; w < W; ++w)
                {
                    dst[oc * H * W + h * W + w] = halfToFloat(t[(size_t) (h * (W * Coutb) + (w + (oc / kChannelBlock) * W)) * kChannelBlock + oc % kChannelBlock]);
                }
            }
        }
    };

    // ============ interleaved timing ============
    // Warm every arm first so no arm pays pipeline compilation or clock ramp inside a timed round.
    runner.submitAndWait(scalarCmd);
    runner.submitAndWait(tiledCmd);
    if (haveImages)
    {
        runner.submitAndWait(storageCmd);
        runner.submitAndWait(samplerCmd);
    }
    std::vector<double> scalarMs, tiledMs, storageMs, samplerMs;
    for (int r = 0; r < rounds; ++r)
    {
        scalarMs.push_back(runner.submitAndWait(scalarCmd) / iters);
        tiledMs.push_back(runner.submitAndWait(tiledCmd) / iters);
        if (haveImages)
        {
            storageMs.push_back(runner.submitAndWait(storageCmd) / iters);
            samplerMs.push_back(runner.submitAndWait(samplerCmd) / iters);
        }
    }

    // ============ verify: run each arm last, then read its output back ============
    std::vector<float> scalarOut(Cout * H * W), tiledOut(Cout * H * W);
    runner.submitAndWait(scalarCmd);
    readbackSsbo(scalarOut);
    runner.submitAndWait(tiledCmd);
    readbackSsbo(tiledOut);

    double baseMs = median(scalarMs);
    report("SSBO(OCB=1)", baseMs, scalarOut, -1);
    report("SSBO(OCB=2)", median(tiledMs), tiledOut, baseMs);
    if (haveImages)
    {
        std::vector<float> storageOut(Cout * H * W), samplerOut(Cout * H * W);
        runner.submitAndWait(storageCmd);
        readbackImage(storageOut);
        runner.submitAndWait(samplerCmd);
        readbackImage(samplerOut);
        report("IMAGE(imageLoad)", median(storageMs), storageOut, baseMs);
        report("TEXFETCH(sampler2D)", median(samplerMs), samplerOut, baseMs);
    }
    vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &scalarCmd);
    vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &tiledCmd);
    if (haveImages)
    {
        vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &storageCmd);
        vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &samplerCmd);
        vkDestroySampler(ctx.device(), sampler, nullptr);
        storagePipe.destroy(ctx.device());
        samplerPipe.destroy(ctx.device());
    }
    return 0;
}
#else
int main() {
    printf("built without Vulkan\n");
    return 0;
}
#endif
