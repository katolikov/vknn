// vknn_microbench — STAGE-0 standalone kernel microbench (Android/Vulkan only).
//
// Times a compute shader loaded from a .spv file on disk at a given conv/gemm shape, GPU-timestamped,
// min-of-N with a serializing barrier between dispatches. Reads a manifest of runs and emits CSV. This
// exists to answer the channel-block-width question (does computing OCB output channels per thread beat
// the production 4:1 reuse?) WITHOUT touching the engine's C4-hardcoded plumbing. Throwaway scaffolding:
// not wired into inference; the winning widths get productionized in Stage 1.
//
// Manifest line (whitespace-separated; '#' comment lines and blanks ignored):
//   label spv N Cin H W Cout KH KW SH SW PT PL DH DW OCB LSX iters warmup verify
// OCB is baked into the .spv (one variant per width); the harness needs it only to size the dispatch.
#include "backend/vulkan/vk_buffer.h"
#include "backend/vulkan/vk_context.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace vknn;

#define MBCHK(x)                                                              \
    do                                                                        \
    {                                                                         \
        VkResult _r = (x);                                                    \
        if (_r != VK_SUCCESS)                                                 \
        {                                                                     \
            fprintf(stderr, "VK fail %d at %s:%d\n", _r, __FILE__, __LINE__); \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

static std::vector<uint32_t> loadSpv(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        fprintf(stderr, "cannot open spv %s\n", path.c_str());
        std::abort();
    }
    size_t sz = (size_t) f.tellg();
    f.seekg(0);
    std::vector<uint32_t> v(sz / 4);
    f.read((char *) v.data(), (std::streamsize) sz);
    return v;
}

// float -> IEEE fp16 bit pattern (via the native __fp16 rounding).
static inline uint16_t f2h(float x) noexcept {
    __fp16   h = (__fp16) x;
    uint16_t o;
    std::memcpy(&o, &h, 2);
    return o;
}
// IEEE fp16 bit pattern -> float.
static inline float h2f(uint16_t x) noexcept {
    __fp16 h;
    std::memcpy(&h, &x, 2);
    return (float) h;
}
// deterministic bounded pseudo-random in [-0.5,0.5] from an index (no RNG, reproducible across devices)
static inline float synth(uint32_t i) noexcept {
    uint32_t h = i * 2654435761u + 1013904223u;
    h ^= h >> 15;
    return (float) (h & 0xffff) / 65535.0f - 0.5f;
}

struct PC {
    int32_t N, Cin, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act;
    float   actLo, actHi;
};

struct Run {
    std::string label, spv;
    int         N, Cin, H, W, Cout, KH, KW, SH, SW, PT, PL, DH, DW, OCB, LSX, iters, warmup, verify;
};

int main(int argc, char **argv) {
    if (argc < 2)
    {
        fprintf(stderr, "usage: vknn_microbench <manifest>\n");
        return 2;
    }
    using namespace vknn::vk;
    VulkanContext ctx;
    if (!ctx.initialized())
    {
        fprintf(stderr, "Vulkan init failed\n");
        return 1;
    }
    const auto &caps   = ctx.caps();
    double      period = caps.timestampPeriod;
    fprintf(stderr, "device=%s subgroup=%u tsPeriod=%.3fns tsSupported=%d maxWGCount0=%u\n", caps.deviceName.c_str(), caps.subgroupSize, period, caps.timestampSupported, caps.maxWorkGroupCount[0]);
    printf("label,device,OCB,LSX,OH,OW,ms_min,ms_med,gflops,maxabserr\n");

    std::ifstream mf(argv[1]);
    if (!mf)
    {
        fprintf(stderr, "cannot open manifest %s\n", argv[1]);
        return 2;
    }
    std::string line;
    while (std::getline(mf, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream is(line);
        Run                r {};
        if (!(is >> r.label >> r.spv >> r.N >> r.Cin >> r.H >> r.W >> r.Cout >> r.KH >> r.KW >> r.SH >> r.SW >> r.PT >> r.PL >> r.DH >> r.DW >> r.OCB >> r.LSX >> r.iters >> r.warmup >> r.verify))
        {
            continue;
        }

        // Channel-block geometry: NC4HW4 packs channels in groups of 4, so Cin/Cout round up to whole
        // vec4 blocks. OCB output channels per thread => OCB/4 blocks per group; Cout blocks round up to
        // a whole number of OCB groups and the padded channel count zero-fills the tail lanes.
        int Cinb = (r.Cin + 3) / 4, Coutb = (r.Cout + 3) / 4;
        int OCB4     = r.OCB / 4;
        int ocGroups = (Coutb + OCB4 - 1) / OCB4;
        int padCoutb = ocGroups * OCB4, padCout = padCoutb * 4;
        // Standard conv output-extent formula (dilated kernel, padding, stride).
        int OH = (r.H + 2 * r.PT - ((r.KH - 1) * r.DH + 1)) / r.SH + 1;
        int OW = (r.W + 2 * r.PL - ((r.KW - 1) * r.DW + 1)) / r.SW + 1;

        size_t srcHalfs = (size_t) r.N * Cinb * 4 * r.H * r.W;
        size_t wtVecs   = (size_t) padCout * Cinb * r.KH * r.KW; // f16vec4 count
        size_t biasVecs = (size_t) padCoutb;
        size_t dstHalfs = (size_t) r.N * Coutb * 4 * OH * OW;

        // --- host synthetic data (zero the channel-padding lanes so the CPU reference is exact) ---
        std::vector<uint16_t> hsrc(srcHalfs), hwt(wtVecs * 4), hbias(biasVecs * 4, 0), hdst(dstHalfs, 0);
        for (int n = 0; n < r.N; ++n)
        {
            for (int icb = 0; icb < Cinb; ++icb)
            {
                for (int p = 0; p < r.H * r.W; ++p)
                {
                    for (int l = 0; l < 4; ++l)
                    {
                        int    ic  = icb * 4 + l;
                        size_t idx = (((size_t) (n * Cinb + icb)) * r.H * r.W + p) * 4 + l;
                        hsrc[idx]  = f2h(ic < r.Cin ? synth((uint32_t) (idx * 3 + 7)) : 0.0f);
                    }
                }
            }
        }
        for (int oc = 0; oc < padCout; ++oc)
        {
            for (int icb = 0; icb < Cinb; ++icb)
            {
                for (int t = 0; t < r.KH * r.KW; ++t)
                {
                    for (int l = 0; l < 4; ++l)
                    {
                        int    ic        = icb * 4 + l;
                        size_t vec       = (((size_t) oc * Cinb + icb) * r.KH * r.KW + t);
                        hwt[vec * 4 + l] = f2h((oc < r.Cout && ic < r.Cin) ? synth((uint32_t) (vec * 5 + l * 131 + 1)) : 0.0f);
                    }
                }
            }
        }
        for (int ocb = 0; ocb < Coutb; ++ocb)
        {
            for (int l = 0; l < 4; ++l)
            {
                hbias[ocb * 4 + l] = f2h(synth((uint32_t) (ocb * 4 + l + 999)));
            }
        }

        // Byte sizes: fp16 halfs are 2 bytes each, an f16vec4 is 8 bytes. dst is host-readable for the
        // optional CPU-reference verify.
        Buffer src(ctx, srcHalfs * 2, MemPref::kAuto);
        Buffer wt(ctx, wtVecs * 8, MemPref::kAuto);
        Buffer bias(ctx, biasVecs * 8, MemPref::kAuto);
        Buffer dst(ctx, dstHalfs * 2, MemPref::kReadback, 0, true);
        src.upload(hsrc.data(), hsrc.size() * 2);
        wt.upload(hwt.data(), hwt.size() * 2);
        bias.upload(hbias.data(), hbias.size() * 2);

        // --- pipeline (classic descriptor set; no push-descriptor dependency) ---
        auto                     spv = loadSpv(r.spv);
        VkShaderModuleCreateInfo smci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv.size() * 4;
        smci.pCode    = spv.data();
        VkShaderModule mod;
        MBCHK(vkCreateShaderModule(ctx.device(), &smci, nullptr, &mod));

        VkDescriptorSetLayoutBinding binds[4];
        for (int i = 0; i < 4; ++i)
        {
            binds[i] = {(uint32_t) i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo slci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        slci.bindingCount = 4;
        slci.pBindings    = binds;
        VkDescriptorSetLayout setLayout;
        MBCHK(vkCreateDescriptorSetLayout(ctx.device(), &slci, nullptr, &setLayout));

        VkPushConstantRange        pcr {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
        VkPipelineLayoutCreateInfo plci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &setLayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        VkPipelineLayout layout;
        MBCHK(vkCreatePipelineLayout(ctx.device(), &plci, nullptr, &layout));

        uint32_t                        lsx = (uint32_t) r.LSX;
        VkSpecializationMapEntry        sme {0, 0, sizeof(uint32_t)};
        VkSpecializationInfo            spec {1, &sme, sizeof(uint32_t), &lsx};
        VkPipelineShaderStageCreateInfo stage {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage               = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module              = mod;
        stage.pName               = "main";
        stage.pSpecializationInfo = &spec;
        VkComputePipelineCreateInfo cpci {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage  = stage;
        cpci.layout = layout;
        VkPipeline pipe;
        MBCHK(vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

        VkDescriptorPoolSize       psz {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo dpci {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets       = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &psz;
        VkDescriptorPool dpool;
        MBCHK(vkCreateDescriptorPool(ctx.device(), &dpci, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsai {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool     = dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &setLayout;
        VkDescriptorSet dset;
        MBCHK(vkAllocateDescriptorSets(ctx.device(), &dsai, &dset));
        VkBuffer               bufs[4] = {src.handle(), wt.handle(), bias.handle(), dst.handle()};
        VkDescriptorBufferInfo dbi[4];
        VkWriteDescriptorSet   wds[4];
        for (int i = 0; i < 4; ++i)
        {
            dbi[i]                 = {bufs[i], 0, VK_WHOLE_SIZE};
            wds[i]                 = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wds[i].dstSet          = dset;
            wds[i].dstBinding      = (uint32_t) i;
            wds[i].descriptorCount = 1;
            wds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wds[i].pBufferInfo     = &dbi[i];
        }
        vkUpdateDescriptorSets(ctx.device(), 4, wds, 0, nullptr);

        // --- dispatch geometry (mirror the engine's 2D split when gx overflows maxWorkGroupCount[0]) ---
        uint64_t totalThreads = (uint64_t) r.N * ocGroups * OH * OW;
        uint32_t gtotal       = (uint32_t) ((totalThreads + lsx - 1) / lsx);
        uint32_t gx = gtotal, gy = 1;
        uint32_t maxc0 = caps.maxWorkGroupCount[0] ? caps.maxWorkGroupCount[0] : 65535u;
        if (gx > maxc0)
        {
            gy = (gx + maxc0 - 1) / maxc0;
            gx = maxc0;
        }

        int                   total = r.warmup + r.iters;
        VkQueryPoolCreateInfo qpci {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = (uint32_t) (2 * total);
        VkQueryPool qpool;
        MBCHK(vkCreateQueryPool(ctx.device(), &qpci, nullptr, &qpool));

        VkCommandPoolCreateInfo cpc {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpc.queueFamilyIndex = ctx.computeQueueFamily();
        VkCommandPool cpool;
        MBCHK(vkCreateCommandPool(ctx.device(), &cpc, nullptr, &cpool));
        VkCommandBufferAllocateInfo cbai {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        MBCHK(vkAllocateCommandBuffers(ctx.device(), &cbai, &cmd));

        PC pc {r.N, r.Cin, r.H, r.W, r.Cout, OH, OW, r.KH, r.KW, r.SH, r.SW, r.PT, r.PL, r.DH, r.DW, 0, 0.0f, 0.0f};

        VkCommandBufferBeginInfo bi {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        MBCHK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdResetQueryPool(cmd, qpool, 0, (uint32_t) (2 * total));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        VkMemoryBarrier mb {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        for (int k = 0; k < total; ++k)
        {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, (uint32_t) (2 * k));
            vkCmdDispatch(cmd, gx, gy, 1);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, (uint32_t) (2 * k + 1));
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
        MBCHK(vkEndCommandBuffer(cmd));

        VkFenceCreateInfo fci {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence           fence;
        MBCHK(vkCreateFence(ctx.device(), &fci, nullptr, &fence));
        VkSubmitInfo si {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        MBCHK(vkQueueSubmit(ctx.computeQueue(), 1, &si, fence));
        MBCHK(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX));

        std::vector<uint64_t> ts(2 * total, 0);
        MBCHK(vkGetQueryPoolResults(ctx.device(), qpool, 0, (uint32_t) (2 * total), ts.size() * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        // Convert each dispatch's timestamp delta (ticks * period ns) to ms, skipping the warmup
        // iterations. Report min (least-perturbed run) and the sorted-middle median.
        std::vector<double> ms;
        for (int k = r.warmup; k < total; ++k)
        {
            ms.push_back((double) (ts[2 * k + 1] - ts[2 * k]) * period / 1e6);
        }
        std::sort(ms.begin(), ms.end());
        double msMin = ms.front(), msMed = ms[ms.size() / 2];
        double flops  = 2.0 * r.N * r.Cout * OH * OW * r.Cin * r.KH * r.KW;
        double gflops = flops / (msMin * 1e6);

        // --- optional CPU-reference verify (small shapes only) ---
        double maxerr = -1.0;
        if (r.verify)
        {
            dst.download(hdst.data(), hdst.size() * 2);
            maxerr = 0.0;
            for (int n = 0; n < r.N; ++n)
            {
                for (int oc = 0; oc < r.Cout; ++oc)
                {
                    for (int oy = 0; oy < OH; ++oy)
                    {
                        for (int ox = 0; ox < OW; ++ox)
                        {
                            int   ocb = oc / 4, l = oc % 4;
                            float acc = h2f(hbias[ocb * 4 + l]);
                            for (int ic = 0; ic < r.Cin; ++ic)
                            {
                                for (int ky = 0; ky < r.KH; ++ky)
                                {
                                    int iy = oy * r.SH - r.PT + ky * r.DH;
                                    if (iy < 0 || iy >= r.H)
                                    {
                                        continue;
                                    }
                                    for (int kx = 0; kx < r.KW; ++kx)
                                    {
                                        int ix = ox * r.SW - r.PL + kx * r.DW;
                                        if (ix < 0 || ix >= r.W)
                                        {
                                            continue;
                                        }
                                        int    icb = ic / 4, il = ic % 4;
                                        size_t sidx = (((size_t) (n * Cinb + icb)) * r.H * r.W + iy * r.W + ix) * 4 + il;
                                        size_t vec  = (((size_t) oc * Cinb + icb) * r.KH * r.KW + (ky * r.KW + kx));
                                        acc += h2f(hsrc[sidx]) * h2f(hwt[vec * 4 + il]);
                                    }
                                }
                            }
                            size_t didx = (((size_t) (n * Coutb + ocb)) * OH * OW + oy * OW + ox) * 4 + l;
                            float  got  = h2f(hdst[didx]);
                            maxerr      = std::max(maxerr, (double) std::fabs(got - acc));
                        }
                    }
                }
            }
        }

        printf("%s,%s,%d,%d,%d,%d,%.5f,%.5f,%.1f,%.4g\n", r.label.c_str(), caps.deviceName.c_str(), r.OCB, r.LSX, OH, OW, msMin, msMed, gflops, maxerr);
        fflush(stdout);

        vkDestroyFence(ctx.device(), fence, nullptr);
        vkDestroyCommandPool(ctx.device(), cpool, nullptr);
        vkDestroyQueryPool(ctx.device(), qpool, nullptr);
        vkDestroyDescriptorPool(ctx.device(), dpool, nullptr);
        vkDestroyPipeline(ctx.device(), pipe, nullptr);
        vkDestroyPipelineLayout(ctx.device(), layout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device(), setLayout, nullptr);
        vkDestroyShaderModule(ctx.device(), mod, nullptr);
    }
    return 0;
}
