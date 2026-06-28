// vknn_zerocopy_cache - declared-format zero-copy: the caller owns a DMA-BUF per input/output and
// DECLARES its layout (NCHW / NHWC / NC4HW4) and dtype (fp32 / fp16). vknn binds the fd directly when
// the declared format matches the model's device-native boundary, or converts on the GPU otherwise —
// no host copy either way. A unified per-model cache file makes the second load fast.
//
// It SELF-VERIFIES: it runs once the normal way (host fp32 in -> outputs) and once per declared format
// (the same inputs packed into caller DMA-BUFs), and checks each zero-copy output matches the host
// output (cosine >= 0.999; fp16-free paths are bit-exact).
//
//   vknn_zerocopy_cache model.vxm [cache.file]
#include "vknn/dtype.h"
#include "vknn/logging.h"
#include "vknn/model.h"
#include "vknn/session.h"
#include "vknn/tensor_format.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <functional>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace vknn;

// The caller's allocator (NOT part of vknn): a DMA-BUF from /dev/dma_heap/system, CPU-mapped. In a real
// app these fds come from the camera / gralloc / ION stack. Returns fd=-1 if unavailable.
struct UserBuf {
    int    fd    = -1;
    void  *map   = nullptr;
    size_t bytes = 0;
};
static UserBuf allocDmaBuf(size_t bytes) {
    UserBuf b;
    b.bytes = bytes;
    struct alloc_data {
        uint64_t len;
        uint32_t fd;
        uint32_t fd_flags;
        uint64_t heap_flags;
    } d {};
    constexpr unsigned long kIoctlAlloc = (3UL << 30) | (sizeof(alloc_data) << 16) | ('H' << 8) | 0x0;
    int                     h           = ::open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (h < 0)
    {
        return b;
    }
    d.len      = bytes;
    d.fd_flags = O_RDWR | O_CLOEXEC;
    if (::ioctl(h, kIoctlAlloc, &d) == 0)
    {
        b.fd  = (int) d.fd;
        b.map = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, b.fd, 0);
        if (b.map == MAP_FAILED)
        {
            b.map = nullptr;
        }
    }
    ::close(h);
    return b;
}
static void freeDmaBuf(UserBuf &b) {
    if (b.map)
    {
        ::munmap(b.map, b.bytes);
    }
    if (b.fd >= 0)
    {
        ::close(b.fd);
    }
}

// Minimal loader for a C-order fp32 .npy (v1.0): skip the header, read the rest as float.
static std::vector<float> loadNpyF32(const std::string &path) {
    FILE *f = ::fopen(path.c_str(), "rb");
    if (!f)
    {
        return {};
    }
    unsigned char h[10];
    if (::fread(h, 1, 10, f) != 10 || h[0] != 0x93)
    {
        ::fclose(f);
        return {};
    }
    long hlen = (long) (h[8] | (h[9] << 8));
    ::fseek(f, 0, SEEK_END);
    long end = ::ftell(f);
    long off = 10 + hlen;
    ::fseek(f, off, SEEK_SET);
    std::vector<float> data((size_t) ((end - off) / 4));
    if (::fread(data.data(), 4, data.size(), f) != data.size())
    {
        data.clear();
    }
    ::fclose(f);
    return data;
}

static std::string shpJoin(const Shape &s) {
    std::string r;
    for (size_t i = 0; i < s.size(); ++i)
    {
        r += (i ? "x" : "") + std::to_string(s[i]);
    }
    return r.empty() ? "scalar" : r;
}

// Element index of (n,c,h,w) in a declared boundary layout (matches the boundary_convert shader).
static int64_t encodeIdx(TensorFormat fmt, const NCHW &x, int64_t n, int64_t c, int64_t h, int64_t w) {
    if (fmt == TensorFormat::NHWC)
    {
        return ((n * x.h + h) * x.w + w) * x.c + c;
    }
    if (fmt == TensorFormat::NC4HW4)
    {
        int64_t Cb = cBlocks(x.c), cb = c / 4, l = c % 4;
        return ((((n * Cb + cb) * x.h + h) * x.w + w) * 4) + l;
    }
    return ((n * x.c + c) * x.h + h) * x.w + w; // NCHW
}
static size_t declBytes(TensorFormat fmt, DType dt, const Shape &shape) {
    return (size_t) (formatElems(fmt, NCHW::from(shape)) * dtypeSize(dt));
}
// NCHW fp32 -> declared (fmt, dtype) bytes the caller puts in its DMA-BUF.
static void packDecl(const Shape &shape, TensorFormat fmt, DType dt, const float *nchw, void *dst) {
    NCHW    x     = NCHW::from(shape);
    int64_t total = formatElems(fmt, x);
    for (int64_t i = 0; i < total; ++i) // zero (NC4HW4 padding lanes stay 0)
    {
        if (dt == DType::Float16)
        {
            ((fp16_t *) dst)[i] = floatToHalf(0.f);
        } else
        {
            ((float *) dst)[i] = 0.f;
        }
    }
    for (int64_t n = 0; n < x.n; ++n)
    {
        for (int64_t c = 0; c < x.c; ++c)
        {
            for (int64_t h = 0; h < x.h; ++h)
            {
                for (int64_t w = 0; w < x.w; ++w)
                {
                    float   v   = nchw[((n * x.c + c) * x.h + h) * x.w + w];
                    int64_t idx = encodeIdx(fmt, x, n, c, h, w);
                    if (dt == DType::Float16)
                    {
                        ((fp16_t *) dst)[idx] = floatToHalf(v);
                    } else
                    {
                        ((float *) dst)[idx] = v;
                    }
                }
            }
        }
    }
}
// declared (fmt, dtype) DMA-BUF bytes -> NCHW fp32 for comparison against the host path.
static void unpackDecl(const Shape &shape, TensorFormat fmt, DType dt, const void *src, float *nchw) {
    NCHW x = NCHW::from(shape);
    for (int64_t n = 0; n < x.n; ++n)
    {
        for (int64_t c = 0; c < x.c; ++c)
        {
            for (int64_t h = 0; h < x.h; ++h)
            {
                for (int64_t w = 0; w < x.w; ++w)
                {
                    int64_t idx                               = encodeIdx(fmt, x, n, c, h, w);
                    float   v                                 = (dt == DType::Float16) ? halfToFloat(((const fp16_t *) src)[idx]) : ((const float *) src)[idx];
                    nchw[((n * x.c + c) * x.h + h) * x.w + w] = v;
                }
            }
        }
    }
}

static double cosine(const float *a, const float *b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; ++i)
    {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
        {
            continue; // diagnosed separately as non-finite
        }
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    double d = std::sqrt(na) * std::sqrt(nb);
    return d > 0 ? dot / d : 1.0;
}

struct Decl {
    const char  *name;
    TensorFormat fmt;
    DType        dt;
};

// Run one declared-format pass and compare every output to the host reference. zcIn/zcOut select whether
// the inputs / outputs use caller DMA-BUFs in the declared format (the other side uses host data), so the
// input convert and the output convert can be exercised in isolation.
static bool runMode(Model &net, const Decl &decl, bool zcIn, bool zcOut, const std::vector<IOInfo> &inInfo, const std::vector<IOInfo> &outInfo, const std::vector<std::vector<float>> &refIn, const std::vector<Tensor> &hostOut) {
    std::vector<UserBuf> inBufs, outBufs;
    std::vector<Tensor>  inputs, outputs;
    for (size_t i = 0; i < inInfo.size(); ++i)
    {
        if (zcIn)
        {
            UserBuf b = allocDmaBuf(declBytes(decl.fmt, decl.dt, inInfo[i].shape));
            if (b.fd < 0 || !b.map)
            {
                fprintf(stderr, "DMA-BUF alloc failed (need /dev/dma_heap/system; Android only)\n");
                return false;
            }
            packDecl(inInfo[i].shape, decl.fmt, decl.dt, refIn[i].data(), b.map);
            inBufs.push_back(b);
            inputs.push_back(Tensor::fromDmaBuf(b.fd, inInfo[i].shape, inInfo[i].name, decl.fmt, decl.dt));
        } else
        {
            inputs.push_back(Tensor(refIn[i], inInfo[i].shape, inInfo[i].name));
        }
    }
    for (size_t i = 0; zcOut && i < outInfo.size(); ++i)
    {
        UserBuf b = allocDmaBuf(declBytes(decl.fmt, decl.dt, outInfo[i].shape));
        if (b.fd < 0 || !b.map)
        {
            fprintf(stderr, "DMA-BUF alloc failed\n");
            return false;
        }
        outBufs.push_back(b);
        outputs.push_back(Tensor::toDmaBuf(b.fd, outInfo[i].shape, outInfo[i].name, decl.fmt, decl.dt));
    }
    auto result = net.run(inputs, outputs);

    bool   ok       = true;
    double worstCos = 1.0, worstAbs = 0;
    for (size_t i = 0; i < outInfo.size(); ++i)
    {
        const Tensor *h = findTensor(hostOut, outInfo[i].name);
        if (!h)
        {
            ok = false;
            continue;
        }
        std::vector<float> zc((size_t) outInfo[i].elems);
        if (zcOut)
        {
            unpackDecl(outInfo[i].shape, decl.fmt, decl.dt, outBufs[i].map, zc.data());
        } else
        {
            const Tensor *r = findTensor(result, outInfo[i].name);
            if (!r || (int64_t) r->values().size() < outInfo[i].elems)
            {
                ok = false;
                continue;
            }
            std::copy(r->values().begin(), r->values().begin() + outInfo[i].elems, zc.begin());
        }
        double  cos    = cosine(zc.data(), h->values().data(), outInfo[i].elems);
        double  mx     = 0;
        int64_t worstK = -1, nBad = 0;
        for (int64_t k = 0; k < outInfo[i].elems; ++k)
        {
            bool fz = std::isfinite(zc[(size_t) k]), fh = std::isfinite(h->values()[(size_t) k]);
            if (!fz)
            {
                ++nBad;
            }
            if (fz && fh)
            {
                double e = std::fabs((double) zc[(size_t) k] - (double) h->values()[(size_t) k]);
                if (e > mx)
                {
                    mx     = e;
                    worstK = k;
                }
            }
        }
        bool outOk = nBad == 0 && cos >= 0.999;
        ok         = ok && outOk;
        worstCos   = std::min(worstCos, cos);
        worstAbs   = std::max(worstAbs, mx);
        if (!outOk || mx > 1.0)
        {
            printf("      out[%zu] '%s' %s: cos=%.5f maxAbsErr=%.3e nonfiniteZc=%lld", i, outInfo[i].name.c_str(), shpJoin(outInfo[i].shape).c_str(), cos, mx, (long long) nBad);
            if (worstK >= 0)
            {
                printf(" worst k=%lld zc=%.4f host=%.4f", (long long) worstK, zc[(size_t) worstK], h->values()[(size_t) worstK]);
            }
            printf("\n");
        }
    }
    printf("    %-8s %s cos=%.5f maxAbsErr=%.3e\n", zcIn && zcOut ? "in+out" : zcIn ? "in-only" : "out-only", ok ? "OK" : "FAIL", worstCos, worstAbs);
    for (auto &b: inBufs)
    {
        freeDmaBuf(b);
    }
    for (auto &b: outBufs)
    {
        freeDmaBuf(b);
    }
    return ok;
}

int main(int argc, char **argv) {
    // positional: model [cache] [input.npy ...]; flags: --fmt N (0-3) --mode M (0=in,1=out,2=both)
    // select one declared format / one zero-copy side so each run is a fresh process (no carried buffer
    // or imported-fd state between configurations).
    std::vector<std::string> pos;
    int                      onlyFmt = -1, onlyMode = -1;
    for (int a = 1; a < argc; ++a)
    {
        std::string s = argv[a];
        if (s == "--fmt" && a + 1 < argc)
        {
            onlyFmt = atoi(argv[++a]);
        } else if (s == "--mode" && a + 1 < argc)
        {
            onlyMode = atoi(argv[++a]);
        } else
        {
            pos.push_back(s);
        }
    }
    if (pos.empty())
    {
        printf("usage: %s model.vxm [cache.file] [input.npy ...] [--fmt N] [--mode M]\n", argv[0]);
        return 1;
    }
    Config cfg;
    if (pos.size() > 1)
    {
        cfg.cacheFile = pos[1]; // else Model::load uses "<model>.cache"
    }
    Model net = Model::load(pos[0], cfg);
    if (!net)
    {
        fprintf(stderr, "failed to load %s\n", pos[0].c_str());
        return 1;
    }
    Session *sess    = net.session();
    auto     inInfo  = sess->inputInfo();
    auto     outInfo = sess->outputInfo();
    printf("model %s: %zu inputs, %zu outputs\n", pos[0].c_str(), inInfo.size(), outInfo.size());
    for (size_t i = 0; i < inInfo.size(); ++i)
    {
        printf("  in[%zu] '%s' shape=%s deviceFormat=%s deviceBytes=%lld\n", i, inInfo[i].name.c_str(), shpJoin(inInfo[i].shape).c_str(), formatStr(inInfo[i].deviceFormat),
               (long long) inInfo[i].deviceBytes);
    }
    for (size_t i = 0; i < outInfo.size(); ++i)
    {
        printf("  out[%zu] '%s' shape=%s deviceFormat=%s deviceBytes=%lld\n", i, outInfo[i].name.c_str(), shpJoin(outInfo[i].shape).c_str(),
               formatStr(outInfo[i].deviceFormat), (long long) outInfo[i].deviceBytes);
    }

    // reference fp32 inputs: real .npy files (pos[2..], one per model input) keep a model in its valid
    // range; otherwise a ramp (fine for classifiers, out-of-distribution for some encoders).
    bool                            haveFiles = pos.size() >= 2 + inInfo.size();
    std::vector<std::vector<float>> refIn(inInfo.size());
    std::vector<Tensor>             hostInputs;
    for (size_t i = 0; i < inInfo.size(); ++i)
    {
        refIn[i].resize((size_t) inInfo[i].elems);
        if (haveFiles)
        {
            std::vector<float> npy = loadNpyF32(pos[2 + i]);
            if ((int64_t) npy.size() < inInfo[i].elems)
            {
                fprintf(stderr, "input file %s has %zu floats, need %lld\n", pos[2 + i].c_str(), npy.size(), (long long) inInfo[i].elems);
                return 1;
            }
            std::copy(npy.begin(), npy.begin() + inInfo[i].elems, refIn[i].begin());
        } else
        {
            for (int64_t k = 0; k < inInfo[i].elems; ++k)
            {
                refIn[i][(size_t) k] = (float) ((k * 37) % 255) / 255.f - 0.5f;
            }
        }
        hostInputs.push_back(Tensor(refIn[i], inInfo[i].shape, inInfo[i].name));
    }

    // (1) host path — normal fp32 in, fp32 out (the reference; this is the non-zero-copy path)
    auto hostOut = net.run(hostInputs);

    // If a <output>_gold.npy sits next to us (e.g. the encoder goldens), check the non-zero-copy host
    // path against it — confirms the model itself is correct, independent of the zero-copy boundary.
    for (size_t i = 0; i < outInfo.size(); ++i)
    {
        std::vector<float> gold = loadNpyF32(outInfo[i].name + "_gold.npy");
        const Tensor      *h    = findTensor(hostOut, outInfo[i].name);
        if (!gold.empty() && h && (int64_t) gold.size() >= outInfo[i].elems)
        {
            printf("  non-zero-copy '%s' vs golden: cos=%.5f\n", outInfo[i].name.c_str(), cosine(h->values().data(), gold.data(), outInfo[i].elems));
        }
    }

    // (2) declared-format zero-copy — one caller DMA-BUF per input/output, in each declared format,
    // exercising the input convert and the output convert in isolation and together.
    const Decl decls[] = {
        {"NCHW   fp32", TensorFormat::NCHW, DType::Float32},
        {"NCHW   fp16", TensorFormat::NCHW, DType::Float16},
        {"NHWC   fp32", TensorFormat::NHWC, DType::Float32},
        {"NC4HW4 fp16", TensorFormat::NC4HW4, DType::Float16},
    };

    const bool modeOn[3] = {onlyMode < 0 || onlyMode == 0, onlyMode < 0 || onlyMode == 1, onlyMode < 0 || onlyMode == 2};
    bool       allOk     = true;
    for (int f = 0; f < 4; ++f)
    {
        if (onlyFmt >= 0 && onlyFmt != f)
        {
            continue;
        }
        const Decl &decl = decls[f];
        printf("  declared %s\n", decl.name);
        if (modeOn[0])
        {
            allOk &= runMode(net, decl, true, false, inInfo, outInfo, refIn, hostOut);
        }
        if (modeOn[1])
        {
            allOk &= runMode(net, decl, false, true, inInfo, outInfo, refIn, hostOut);
        }
        if (modeOn[2])
        {
            allOk &= runMode(net, decl, true, true, inInfo, outInfo, refIn, hostOut);
        }
    }

    printf("declared-format zero-copy %s host  (all formats)\n", allOk ? "MATCHES" : "DIFFERS FROM");

    // runtime: host path (packs the input, unpacks every output) vs device-native zero-copy (the GPU
    // reads/writes the caller's DMA-BUFs directly — no host pack/unpack). Same GPU compute either way;
    // the difference is the boundary I/O the zero-copy path removes.
    auto wallMs = [](const std::function<void()> &body, int iters) {
        body(); // warm
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            body();
        }
        return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count() / iters;
    };
    std::vector<UserBuf> zbIn, zbOut;
    std::vector<Tensor>  zIn, zOut;
    bool                 zok = true;
    for (size_t i = 0; i < inInfo.size() && zok; ++i)
    {
        UserBuf b = allocDmaBuf((size_t) inInfo[i].deviceBytes);
        zok       = b.fd >= 0 && b.map;
        if (zok)
        {
            packDecl(inInfo[i].shape, inInfo[i].deviceFormat, inInfo[i].deviceDtype, refIn[i].data(), b.map);
            zbIn.push_back(b);
            zIn.push_back(Tensor::fromDmaBuf(b.fd, inInfo[i].shape, inInfo[i].name, TensorFormat::Auto));
        }
    }
    for (size_t i = 0; i < outInfo.size() && zok; ++i)
    {
        UserBuf b = allocDmaBuf((size_t) outInfo[i].deviceBytes);
        zok       = b.fd >= 0 && b.map;
        if (zok)
        {
            zbOut.push_back(b);
            zOut.push_back(Tensor::toDmaBuf(b.fd, outInfo[i].shape, outInfo[i].name, TensorFormat::Auto));
        }
    }
    if (zok)
    {
        double hostMs = wallMs(
            [&] {
                net.run(hostInputs);
            },
            5);
        double zcMs = wallMs(
            [&] {
                net.run(zIn, zOut);
            },
            5);
        printf("runtime (mean of 5): host=%.2f ms  zero-copy=%.2f ms  (%.1f%% of host)\n", hostMs, zcMs, 100.0 * zcMs / hostMs);
    }
    for (auto &b: zbIn)
    {
        freeDmaBuf(b);
    }
    for (auto &b: zbOut)
    {
        freeDmaBuf(b);
    }

    printf("cache file: %s\n", sess->config().cacheFile.c_str());
    return allOk ? 0 : 3;
}
