// vx_image_bench - settle the "does VkImage beat SSBO on this GPU?" question with a real number.
// Runs the SAME 1x1 conv two ways on representative shapes: (a) our SSBO c-tiled kernel,
// (b) an image-backed c8w4 kernel reading via imageLoad (texture cache). Verifies both against a
// CPU reference and reports timing. Usage: vx_image_bench [Cin Cout H W] [iters]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vx/dtype.h"
#include "vx/logging.h"
#if defined(VXRT_ENABLE_VULKAN)
#include "backends/vulkan/vk_buffer.h"
#include "backends/vulkan/vk_command.h"
#include "backends/vulkan/vk_image.h"
#include "backends/vulkan/vk_pipeline.h"

using namespace vx;

int main(int argc, char** argv) {
  int Cin = argc > 4 ? atoi(argv[1]) : 256;
  int Cout = argc > 4 ? atoi(argv[2]) : 256;
  int H = argc > 4 ? atoi(argv[3]) : 14;
  int W = argc > 4 ? atoi(argv[4]) : 14;
  int iters = argc > 5 ? atoi(argv[5]) : 200;
  int Cinb = (Cin + 3) / 4, Coutb = (Cout + 3) / 4;

  vk::VulkanContext ctx;
  if (!ctx.initialized()) {
    fprintf(stderr, "no Vulkan\n");
    return 1;
  }
  vk::CommandRunner runner(ctx);
  printf("conv1x1  Cin=%d Cout=%d HxW=%dx%d  iters=%d   (RGBA16F storage supported=%d)\n", Cin,
         Cout, H, W, iters, (int)vk::Image::supported(ctx));

  // ---- reference data ----
  std::vector<float> in(Cin * H * W), wt(Cout * Cin), ref(Cout * H * W, 0.f);
  for (int i = 0; i < Cin * H * W; ++i)
    in[i] = ((i * 37) % 23 - 11) * 0.03f;
  for (int i = 0; i < Cout * Cin; ++i)
    wt[i] = ((i * 13) % 17 - 8) * 0.02f;
  for (int oc = 0; oc < Cout; ++oc)
    for (int p = 0; p < H * W; ++p) {
      float acc = 0;
      for (int ic = 0; ic < Cin; ++ic)
        acc += wt[oc * Cin + ic] * in[ic * H * W + p];
      ref[oc * H * W + p] = acc;
    }
  auto cosErr = [&](const std::vector<float>& v) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      dot += v[i] * ref[i];
      na += v[i] * v[i];
      nb += ref[i] * ref[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-12);
  };

  // ============ SSBO path (our current kernel) ============
  double ssboMs = 0;
  std::vector<float> ssboOut(Cout * H * W);
  {
    auto pack = [](const float* s, fp16_t* d, int C, int HW) {  // NCHW -> NC4HW4
      int Cb = (C + 3) / 4;
      for (int cb = 0; cb < Cb; ++cb)
        for (int p = 0; p < HW; ++p)
          for (int l = 0; l < 4; ++l) {
            int c = cb * 4 + l;
            d[(cb * HW + p) * 4 + l] = floatToHalf(c < C ? s[c * HW + p] : 0.f);
          }
    };
    vk::Buffer inB(ctx, (size_t)Cinb * H * W * 4 * 2),
        outB(ctx, (size_t)Coutb * H * W * 4 * 2, vk::MemPref::kReadback);
    vk::Buffer wB(ctx, (size_t)Cout * Cinb * 4 * 2), bB(ctx, (size_t)Coutb * 4 * 2);
    {
      std::vector<fp16_t> t(Cinb * H * W * 4);
      pack(in.data(), t.data(), Cin, H * W);
      inB.upload(t.data(), t.size() * 2);
    }
    {
      std::vector<fp16_t> t(Cout * Cinb * 4, 0);  // weight [Cout][Cinb][4ic]
      for (int oc = 0; oc < Cout; ++oc)
        for (int ic = 0; ic < Cin; ++ic)
          t[(oc * Cinb + ic / 4) * 4 + ic % 4] = floatToHalf(wt[oc * Cin + ic]);
      wB.upload(t.data(), t.size() * 2);
    }
    {
      std::vector<fp16_t> z(Coutb * 4, 0);
      bB.upload(z.data(), z.size() * 2);
    }

    vk::ComputePipeline pipe(ctx, "conv1x1", 4, sizeof(int) * 18);
    struct PC {
      int N, Cin, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act;
      float lo, hi;
    } pc{1, Cin, H, W, Cout, H, W, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0};
    int64_t HWt = (int64_t)H * W, nTiles = (HWt + 3) / 4, total = Coutb * nTiles;
    VkCommandBuffer cmd = runner.allocate();
    runner.begin(cmd);
    for (int i = 0; i < iters; ++i) {
      pipe.dispatch(cmd, {inB.handle(), wB.handle(), bB.handle(), outB.handle()}, &pc, sizeof(pc),
                    (uint32_t)((total + 63) / 64));
      vk::computeBarrier(cmd);
    }
    runner.end(cmd);
    runner.submitAndWait(cmd);  // warm
    ssboMs = runner.submitAndWait(cmd) / iters;
    vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &cmd);
    std::vector<fp16_t> t(Coutb * H * W * 4);
    outB.download(t.data(), t.size() * 2);
    for (int oc = 0; oc < Cout; ++oc)
      for (int p = 0; p < H * W; ++p)
        ssboOut[oc * H * W + p] = halfToFloat(t[((oc / 4) * H * W + p) * 4 + oc % 4]);
  }

  // ============ Image path (texture-backed c8w4) ============
  double imgMs = -1;
  std::vector<float> imgOut(Cout * H * W);
  if (vk::Image::supported(ctx)) {
    vk::Image inI(ctx, W * Cinb, H), kI(ctx, Cin, Coutb), outI(ctx, W * Coutb, H);
    inI.toGeneral(runner);
    kI.toGeneral(runner);
    outI.toGeneral(runner);
    {
      std::vector<fp16_t> t((size_t)W * Cinb * H * 4, 0);  // input image [w+icb*W, h]
      for (int icb = 0; icb < Cinb; ++icb)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
            for (int l = 0; l < 4; ++l) {
              int c = icb * 4 + l;
              t[(h * (W * Cinb) + (w + icb * W)) * 4 + l] =
                  floatToHalf(c < Cin ? in[c * H * W + h * W + w] : 0.f);
            }
      inI.upload(runner, t.data());
    }
    {
      std::vector<fp16_t> t((size_t)Cin * Coutb * 4, 0);  // kernel image [ic, ocb] = 4 oc
      for (int ocb = 0; ocb < Coutb; ++ocb)
        for (int ic = 0; ic < Cin; ++ic)
          for (int l = 0; l < 4; ++l) {
            int oc = ocb * 4 + l;
            t[(ocb * Cin + ic) * 4 + l] = floatToHalf(oc < Cout ? wt[oc * Cin + ic] : 0.f);
          }
      kI.upload(runner, t.data());
    }

    // standalone storage-image pipeline
    VkDescriptorSetLayoutBinding b[3]{};
    for (int i = 0; i < 3; ++i) {
      b[i].binding = i;
      b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      b[i].descriptorCount = 1;
      b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    sl.bindingCount = 3;
    sl.pBindings = b;
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(ctx.device(), &sl, nullptr, &dsl);
    struct IPC {
      int W, H, Cin, Cout;
    } ipc{W, H, Cin, Cout};
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IPC)};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &dsl;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    VkPipelineLayout playout;
    vkCreatePipelineLayout(ctx.device(), &pl, nullptr, &playout);
    const auto& spv = embeddedShaders().at("conv1x1_img");
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = spv.size() * 4;
    sm.pCode = spv.data();
    VkShaderModule mod;
    vkCreateShaderModule(ctx.device(), &sm, nullptr, &mod);
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod;
    cp.stage.pName = "main";
    cp.layout = playout;
    VkPipeline pipe;
    vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &cp, nullptr, &pipe);

    int C8 = (Coutb + 1) / 2, W4 = (W + 3) / 4;
    uint32_t groups = (C8 * W4 * H + 63) / 64;
    auto rec = [&](VkCommandBuffer cmd) {
      VkDescriptorImageInfo ii[3] = {{VK_NULL_HANDLE, inI.view(), VK_IMAGE_LAYOUT_GENERAL},
                                     {VK_NULL_HANDLE, kI.view(), VK_IMAGE_LAYOUT_GENERAL},
                                     {VK_NULL_HANDLE, outI.view(), VK_IMAGE_LAYOUT_GENERAL}};
      VkWriteDescriptorSet w[3]{};
      for (int i = 0; i < 3; ++i) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &ii[i];
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      ctx.cmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 3, w);
      vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IPC), &ipc);
      vkCmdDispatch(cmd, groups, 1, 1);
    };
    VkCommandBuffer cmd = runner.allocate();
    runner.begin(cmd);
    for (int i = 0; i < iters; ++i) {
      rec(cmd);
      vk::computeBarrier(cmd);
    }
    runner.end(cmd);
    runner.submitAndWait(cmd);
    imgMs = runner.submitAndWait(cmd) / iters;
    vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &cmd);

    std::vector<fp16_t> t((size_t)W * Coutb * H * 4);
    outI.download(runner, t.data());
    for (int oc = 0; oc < Cout; ++oc)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          imgOut[oc * H * W + h * W + w] =
              halfToFloat(t[(h * (W * Coutb) + (w + (oc / 4) * W)) * 4 + oc % 4]);
    vkDestroyPipeline(ctx.device(), pipe, nullptr);
    vkDestroyShaderModule(ctx.device(), mod, nullptr);
    vkDestroyPipelineLayout(ctx.device(), playout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device(), dsl, nullptr);

    // ---- sampler2D / texelFetch path (true texture cache, like MNN) ----
    VkSampler samp;
    {
      VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
      si.magFilter = si.minFilter = VK_FILTER_NEAREST;
      si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      vkCreateSampler(ctx.device(), &si, nullptr, &samp);
    }
    VkDescriptorSetLayoutBinding tb[3]{};
    tb[0].binding = 0;
    tb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    tb[1].binding = 1;
    tb[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    tb[2].binding = 2;
    tb[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    for (int i = 0; i < 3; ++i) {
      tb[i].descriptorCount = 1;
      tb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo tsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    tsl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    tsl.bindingCount = 3;
    tsl.pBindings = tb;
    VkDescriptorSetLayout tdsl;
    vkCreateDescriptorSetLayout(ctx.device(), &tsl, nullptr, &tdsl);
    VkPushConstantRange tpcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IPC)};
    VkPipelineLayoutCreateInfo tpl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    tpl.setLayoutCount = 1;
    tpl.pSetLayouts = &tdsl;
    tpl.pushConstantRangeCount = 1;
    tpl.pPushConstantRanges = &tpcr;
    VkPipelineLayout tlayout;
    vkCreatePipelineLayout(ctx.device(), &tpl, nullptr, &tlayout);
    const auto& tspv = embeddedShaders().at("conv1x1_tex");
    VkShaderModuleCreateInfo tsm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    tsm.codeSize = tspv.size() * 4;
    tsm.pCode = tspv.data();
    VkShaderModule tmod;
    vkCreateShaderModule(ctx.device(), &tsm, nullptr, &tmod);
    VkComputePipelineCreateInfo tcp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    tcp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    tcp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    tcp.stage.module = tmod;
    tcp.stage.pName = "main";
    tcp.layout = tlayout;
    VkPipeline tpipe;
    vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &tcp, nullptr, &tpipe);
    auto recT = [&](VkCommandBuffer cmd) {
      VkDescriptorImageInfo ti[3] = {{samp, inI.view(), VK_IMAGE_LAYOUT_GENERAL},
                                     {samp, kI.view(), VK_IMAGE_LAYOUT_GENERAL},
                                     {VK_NULL_HANDLE, outI.view(), VK_IMAGE_LAYOUT_GENERAL}};
      VkWriteDescriptorSet w[3]{};
      for (int i = 0; i < 3; ++i) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = tb[i].descriptorType;
        w[i].pImageInfo = &ti[i];
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tpipe);
      ctx.cmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tlayout, 0, 3, w);
      vkCmdPushConstants(cmd, tlayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IPC), &ipc);
      vkCmdDispatch(cmd, groups, 1, 1);
    };
    VkCommandBuffer tc = runner.allocate();
    runner.begin(tc);
    for (int i = 0; i < iters; ++i) {
      recT(tc);
      vk::computeBarrier(tc);
    }
    runner.end(tc);
    runner.submitAndWait(tc);
    double texMs = runner.submitAndWait(tc) / iters;
    vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &tc);
    std::vector<fp16_t> tt((size_t)W * Coutb * H * 4);
    outI.download(runner, tt.data());
    std::vector<float> texOut(Cout * H * W);
    for (int oc = 0; oc < Cout; ++oc)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          texOut[oc * H * W + h * W + w] =
              halfToFloat(tt[(h * (W * Coutb) + (w + (oc / 4) * W)) * 4 + oc % 4]);
    printf("TEXFETCH(sampler2D): %.4f ms/iter cosine=%.5f => %.2fx vs SSBO %s\n", texMs,
           cosErr(texOut), ssboMs / texMs, texMs < ssboMs ? "FASTER" : "slower");
    vkDestroyPipeline(ctx.device(), tpipe, nullptr);
    vkDestroyShaderModule(ctx.device(), tmod, nullptr);
    vkDestroyPipelineLayout(ctx.device(), tlayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device(), tdsl, nullptr);
    vkDestroySampler(ctx.device(), samp, nullptr);
  }

  printf("SSBO : %.4f ms/iter   cosine=%.5f\n", ssboMs, cosErr(ssboOut));
  if (imgMs >= 0)
    printf("IMAGE(storage/imageLoad): %.4f ms/iter cosine=%.5f => %.2fx vs SSBO %s\n", imgMs,
           cosErr(imgOut), ssboMs / imgMs, imgMs < ssboMs ? "FASTER" : "slower");
  return 0;
}
#else
int main() {
  printf("built without Vulkan\n");
  return 0;
}
#endif
