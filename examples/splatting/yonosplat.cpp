// vknn_yonosplat - the full YoNoSplat 3D Gaussian-Splatting pipeline in one program, all on the
// GPU:
//   image + intrinsics --> encoder (vknn Vulkan) --> Gaussians --> Vulkan rasterizer --> rendered
//   view.
// The encoder runs as a normal vknn Session; its 6 Gaussian outputs feed the from-scratch Vulkan
// compute rasterizer (raster_core.h: preprocess -> GPU tile-bin -> stable radix sort -> per-tile
// alpha compositing), which the app-demo JNI bridge shares.
// The rendered view is written as a PPM. See scripts/yonosplat/ for how the encoder .vxm + inputs
// are made.
//
//   vknn_yonosplat <encoder.vxm> <image.bin> <intrinsics.bin> <out.ppm> [--extr extr.bin]
//     [--view N] [--render S] [--repeat N] [--raw out.f32] [--packed out.u32]
// image.bin = fp32 [1,V,3,224,224], intrinsics.bin = fp32 [1,V,3,3] (normalized). extr.bin
// (optional) = fp32 [V,4,4] camera-to-world (the encoder's predicted pose, dumpable via
// Config::dumpTensors); identity if omitted. Renders view N.
// --render S rasterizes at SxS instead of the encoder's 224 (the normalized intrinsics scale
// with the render size). --repeat N renders N times, times each, and verifies the fp32 outputs
// are byte-identical (determinism gate). --raw dumps the fp32 render; --packed additionally runs
// the packed-ARGB path, dumps it, and verifies it equals the fp32 render's round-half-up 8-bit
// quantization exactly.
#include "vknn/session.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#if defined(VKNN_ENABLE_VULKAN)
#include "raster_core.h"
#endif

using namespace vknn;

// Reads an entire file into a byte vector; returns an empty vector if the file cannot be opened.
static std::vector<uint8_t> readFile(const std::string &p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f)
    {
        return {};
    }
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> v((size_t) n);
    f.read(reinterpret_cast<char *>(v.data()), n);
    return v;
}
// Returns the argument following the flag k on the command line, or the default d if k is absent.
static const char *opt(int c, char **v, const char *k, const char *d) noexcept {
    for (int i = 1; i < c - 1; ++i)
    {
        if (!strcmp(v[i], k))
        {
            return v[i + 1];
        }
    }
    return d;
}

int main(int argc, char **argv) {
#if !defined(VKNN_ENABLE_VULKAN)
    fprintf(stderr, "vknn_yonosplat needs Vulkan\n");
    return 1;
#else
    if (argc < 5)
    {
        printf("usage: vknn_yonosplat <encoder.vxm> <image.bin> <intrinsics.bin> <out.ppm> [--extr "
               "extr.bin] [--view N] [--render S] [--repeat N] [--raw out.f32] [--packed out.u32]\n");
        return 1;
    }
    std::string enc = argv[1], imgp = argv[2], intp = argv[3], outp = argv[4];
    std::string extp       = opt(argc, argv, "--extr", "");
    std::string rawPath    = opt(argc, argv, "--raw", "");
    std::string packedPath = opt(argc, argv, "--packed", "");
    int         view       = atoi(opt(argc, argv, "--view", "0"));
    int         renderSide = atoi(opt(argc, argv, "--render", "224"));
    int         repeat     = atoi(opt(argc, argv, "--repeat", "1"));
    if (renderSide <= 0)
    {
        renderSide = 224;
    }
    const int   H = renderSide, W = renderSide;
    const float NEAR = 0.2f;

    // ---------- 1. encoder: image + intrinsics -> 6 Gaussian outputs (vknn Vulkan) ----------
    Config cfg;
    cfg.backend                = BackendKind::Vulkan;
    cfg.precision              = Precision::Low;
    cfg.freeWeightsAfterUpload = true;
    auto sess                  = Runtime::load(enc, cfg);
    if (!sess)
    {
        fprintf(stderr, "failed to load encoder %s\n", enc.c_str());
        return 1;
    }
    std::vector<IOTensor> ins, outs;
    for (auto &info: sess->inputInfo())
    {
        IOTensor t;
        t.name  = info.name;
        t.shape = info.shape;
        t.dtype = DType::Float32;
        t.data  = readFile(info.name == sess->inputInfo()[0].name ? imgp : intp);
        t.data.resize((size_t) numElements(info.shape) * 4, 0);
        ins.push_back(std::move(t));
    }
    printf("[encoder] running on GPU ...\n");
    if (sess->run(ins, outs) != Status::Ok)
    {
        fprintf(stderr, "encoder run failed\n");
        return 2;
    }
    auto get = [&](const char *nm) -> const float * {
        for (auto &o: outs)
        {
            if (o.name == nm)
            {
                return o.f32();
            }
        }
        fprintf(stderr, "encoder output '%s' missing\n", nm);
        return nullptr;
    };
    const float *means = get("means"), *covs = get("covariances"), *harm = get("harmonics"), *opac = get("opacities");
    if (!means || !covs || !harm || !opac)
    {
        return 2;
    }
    int N = 0;
    for (auto &o: outs)
    {
        if (o.name == "means")
        {
            N = (int) (numElements(o.shape) / 3);
        }
    }
    printf("[encoder] %d Gaussians\n", N);
    std::vector<float> mv(means, means + (size_t) N * 3), cvv(covs, covs + (size_t) N * 9), opv(opac, opac + (size_t) N), cols((size_t) N * 3);
    for (size_t i = 0; i < (size_t) N * 3; ++i)
    {
        cols[i] = std::max(0.f, raster::kC0 * harm[i] + 0.5f);
    }
    std::vector<float> intr(36, 0);
    {
        auto d = readFile(intp);
        memcpy(intr.data(), d.data(), std::min(d.size(), intr.size() * 4));
    }
    std::vector<float> extr;
    if (!extp.empty())
    {
        auto d = readFile(extp);
        extr.assign((size_t) (view + 1) * 16, 0);
        memcpy(extr.data(), d.data(), std::min(d.size(), extr.size() * 4));
    }
    sess.reset(); // free the encoder (+ its Vulkan context) before the rasterizer

    // ---------- 2. camera: w2c = rigid inverse of c2w, K in pixels ----------
    float c2w[16];
    if (!extr.empty())
    {
        memcpy(c2w, &extr[(size_t) view * 16], 64);
    } else
    {
        memset(c2w, 0, 64);
        c2w[0] = c2w[5] = c2w[10] = c2w[15] = 1.f;
    } // identity
    const float *K = &intr[(size_t) view * 9];
    float        Rt[9], tp[3];
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            Rt[r * 3 + c] = c2w[c * 4 + r];
        }
    }
    for (int r = 0; r < 3; ++r)
    {
        tp[r] = -(Rt[r * 3] * c2w[3] + Rt[r * 3 + 1] * c2w[7] + Rt[r * 3 + 2] * c2w[11]);
    }
    float fx = K[0] * W, fy = K[4] * H, cx = K[2] * W, cy = K[5] * H;

    // ---------- 3. rasterizer: Gaussians -> view, entirely on the GPU (raster_core.h) ----------
    raster::Rasterizer rz(H, W, NEAR);
    if (!rz.ok())
    {
        fprintf(stderr, "no Vulkan for rasterizer\n");
        return 1;
    }
    rz.setGaussians(mv.data(), cvv.data(), cols.data(), opv.data(), N);
    std::vector<float> img((size_t) H * W * 3);
    raster::Stats      st;
    raster::Result     rr = rz.render(c2w, fx, fy, cx, cy, img.data(), &st);
    if (rr == raster::Result::SortLimitExceeded)
    {
        fprintf(stderr, "[rasterizer] %u tile entries exceed the 2^30 sort limit\n", st.entries);
        return 3;
    }
    if (rr != raster::Result::Ok)
    {
        fprintf(stderr, "no Vulkan for rasterizer\n");
        return 1;
    }
    printf("[rasterizer] %u tile entries, sort capacity %lld\n", st.entries, (long long) st.cap);
    if (st.emitted > (uint32_t) st.cap)
    {
        fprintf(stderr, "[rasterizer] WARNING: %u tile entries exceed capacity %lld - %u dropped, image is incomplete\n", st.emitted, (long long) st.cap,
                st.emitted - (uint32_t) st.cap);
    }
    printf("[rasterizer] view %d rendered %dx%d on GPU in %.2f ms (count %.2f + main %.2f)\n", view, W, H, st.msCount + st.msMain, st.msCount, st.msMain);

    // --repeat: re-render the same pose and require byte-identical fp32 output (determinism
    // gate). Repeats also report the steady-state time, where the count pass is elided.
    for (int rerun = 1; rerun < repeat; ++rerun)
    {
        std::vector<float> imgRepeat(img.size());
        raster::Stats      stRepeat;
        if (rz.render(c2w, fx, fy, cx, cy, imgRepeat.data(), &stRepeat) != raster::Result::Ok)
        {
            fprintf(stderr, "[determinism] repeat %d render failed\n", rerun);
            return 4;
        }
        printf("[rasterizer] repeat %d rendered in %.2f ms (count %.2f + main %.2f)\n", rerun, stRepeat.msCount + stRepeat.msMain, stRepeat.msCount, stRepeat.msMain);
        if (memcmp(img.data(), imgRepeat.data(), img.size() * 4) != 0)
        {
            fprintf(stderr, "[determinism] repeat %d fp32 output DIFFERS\n", rerun);
            return 5;
        }
        printf("[determinism] repeat %d fp32 output byte-identical\n", rerun);
    }

    if (!rawPath.empty())
    {
        std::ofstream rawFile(rawPath, std::ios::binary);
        rawFile.write(reinterpret_cast<const char *>(img.data()), (std::streamsize)(img.size() * 4));
        printf("[raw] fp32 render -> %s\n", rawPath.c_str());
    }

    // --packed: the GPU-packed ARGB path must equal the fp32 render's round-half-up 8-bit
    // quantization exactly (byte = trunc(clamp(c, 0, 1) * 255 + 0.5) in fp32).
    if (!packedPath.empty())
    {
        std::vector<uint32_t> packed((size_t) H * W);
        raster::Stats         stPacked;
        if (rz.renderPacked(c2w, fx, fy, cx, cy, packed.data(), &stPacked) != raster::Result::Ok)
        {
            fprintf(stderr, "[packed] render failed\n");
            return 6;
        }
        printf("[packed] rendered in %.2f ms (count %.2f + main %.2f)\n", stPacked.msCount + stPacked.msMain, stPacked.msCount, stPacked.msMain);
        size_t mismatches = 0;
        for (size_t i = 0; i < packed.size(); ++i)
        {
            auto quantize = [&](int channel) -> uint32_t {
                // Separate statements keep the mul and add unfused (matches the shader's
                // contraction-disabled quantization bit-for-bit).
                float scaled = std::min(1.0f, std::max(0.0f, img[i * 3 + (size_t) channel])) * 255.0f;
                scaled += 0.5f;
                return (uint32_t) scaled;
            };
            const uint32_t expected = 0xff000000u | (quantize(0) << 16) | (quantize(1) << 8) | quantize(2);
            if (packed[i] != expected)
            {
                ++mismatches;
            }
        }
        std::ofstream packedFile(packedPath, std::ios::binary);
        packedFile.write(reinterpret_cast<const char *>(packed.data()), (std::streamsize)(packed.size() * 4));
        printf("[packed] ARGB render -> %s, %zu/%zu pixels differ from the quantized fp32 render\n", packedPath.c_str(), mismatches, packed.size());
        if (mismatches != 0)
        {
            return 7;
        }
    }

    // ---------- 4. save PPM ----------
    std::ofstream f(outp, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (size_t i = 0; i < img.size(); ++i)
    {
        f.put((char) (uint8_t) (std::min(1.f, std::max(0.f, img[i])) * 255 + 0.5f));
    }
    printf("[done] image -> %s  (mean=%.4f)\n", outp.c_str(), [&] {
        double s = 0;
        for (float x: img)
        {
            s += x;
        }
        return s / img.size();
    }());
    return 0;
#endif
}
