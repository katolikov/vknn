// vknn_benchmark - run a model on-device from one JSON config: feed .npy or raw (.bin/.raw) inputs (or
// none, for a runtime-only measurement), optionally save outputs as .npy / .raw / .png, compare each
// named output against a golden (cosine / PSNR / SNR / relL2 / max|d|), and write a result .json with
// per-call timing and (optional) per-operator profiling.
//
//   vknn_benchmark config.json
//
// Config (flat; paths relative to the config file or absolute). run.py generates this from its richer
// model/convert/device/stages config, but it is hand-writable too:
//   {
//     "model": "encoder8_fp16.vxm",         // .onnx or .vxm (required)
//     "backend": "vulkan", "precision": "fp16",
//     "no_weight_cache": true,
//     "cache": "model.cache",               // unified per-model cache file (default "<model>.cache")
//     "generate_cache": false,              // populate the cache first (untimed), then time a warm load
//     "max_submit_nodes": 500,              // 0 = single submit
//     "timing": true,                       // engine prints pack/submit/unpack
//     "profile": false,                     // per-operator GPU timing in the result json
//     "inputs": ["image.npy", "intr.raw"],  // .npy (shape from header) or raw .bin/.raw (model shape);
//                                           // or { "image": "image.npy" }; OMIT for runtime-only
//     "save": ["npy", "raw", "png"],        // output formats to write (optional)
//     "save_dir": "out",                    // where to write them (default ".")
//     "golden": { "means": "means_gold.npy" },          // output-name -> golden (optional)
//     "metrics": ["cosine","psnr","snr","relL2","max"], // which to report (default: all)
//     "tolerance": 0.999,                   // cosine pass threshold
//     "result": "result.json"               // write timing + profile + metrics here (optional)
//   }
//
// Exit code 0 iff every checked output passes (cosine >= tolerance and no NaN), or there is nothing
// to check (runtime-only).
#include "core/json.h"
#include "vknn/dtype.h"
#include "vknn/session.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    using Clock = std::chrono::high_resolution_clock;
    double msSince(Clock::time_point a) {
        return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    }

    std::string dirOf(const std::string &p) {
        auto s = p.find_last_of("/\\");
        return s == std::string::npos ? std::string() : p.substr(0, s + 1);
    }
    std::string baseName(const std::string &p) {
        auto s = p.find_last_of("/\\");
        return s == std::string::npos ? p : p.substr(s + 1);
    }
    std::string resolve(const std::string &base, const std::string &p) {
        return (p.empty() || p[0] == '/') ? p : base + p;
    }
    std::string sanitize(std::string s) {
        for (char &c: s)
        {
            if (c == '/' || c == ':')
            {
                c = '_';
            }
        }
        return s;
    }
    bool endsWith(const std::string &s, const char *suf) {
        size_t n = strlen(suf);
        return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
    }

    std::string readFile(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        return f ? std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()) : std::string();
    }

    // ---- input loaders: .npy (any numeric dtype -> fp32) or raw .bin (fp32) ----
    struct Tensor {
        std::vector<int64_t> shape; // empty for raw .bin
        std::vector<float>   data;
    };

    bool loadNpy(const std::string &path, Tensor &out, std::string &err) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            err = "cannot open " + path;
            return false;
        }
        char magic[6] = {};
        f.read(magic, 6);
        if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        {
            err = path + ": not a .npy file";
            return false;
        }
        uint8_t  major = 0, minor = 0;
        uint32_t hlen = 0;
        f.read(reinterpret_cast<char *>(&major), 1);
        f.read(reinterpret_cast<char *>(&minor), 1);
        if (major >= 2)
        {
            f.read(reinterpret_cast<char *>(&hlen), 4);
        } else
        {
            uint16_t h = 0;
            f.read(reinterpret_cast<char *>(&h), 2);
            hlen = h;
        }
        std::string hdr(hlen, '\0');
        f.read(&hdr[0], hlen);
        if (hdr.find("'fortran_order': True") != std::string::npos)
        {
            err = path + ": fortran_order=True unsupported";
            return false;
        }
        std::string descr;
        if (auto d = hdr.find("'descr'"); d != std::string::npos)
        {
            auto q = hdr.find('\'', hdr.find(':', d) + 1);
            descr  = hdr.substr(q + 1, hdr.find('\'', q + 1) - q - 1);
        }
        auto lp = hdr.find('(', hdr.find("'shape'")), rp = hdr.find(')', lp);
        out.shape.clear();
        for (size_t i = lp + 1; i < rp; ++i)
        {
            if (isdigit(hdr[i]))
            {
                int64_t v = 0;
                while (i < rp && isdigit(hdr[i]))
                {
                    v = v * 10 + (hdr[i++] - '0');
                }
                out.shape.push_back(v);
            }
        }
        int64_t n = 1;
        for (auto d: out.shape)
        {
            n *= d;
        }
        if (out.shape.empty())
        {
            n = 0;
        }
        out.data.resize((size_t) std::max<int64_t>(n, 0));
        char              tc    = descr.size() >= 2 ? descr[1] : 'f';
        int               bytes = descr.size() >= 3 ? descr[2] - '0' : 4;
        std::vector<char> raw((size_t) n * (size_t) bytes);
        f.read(raw.data(), (std::streamsize) raw.size());
        for (int64_t i = 0; i < n; ++i)
        {
            const char *p = raw.data() + (size_t) i * bytes;
            float       v = 0.f;
            if (tc == 'f' && bytes == 4)
            {
                std::memcpy(&v, p, 4);
            } else if (tc == 'f' && bytes == 2)
            {
                fp16_t h;
                std::memcpy(&h, p, 2);
                v = halfToFloat(h);
            } else if (tc == 'f' && bytes == 8)
            {
                double d;
                std::memcpy(&d, p, 8);
                v = (float) d;
            } else if (tc == 'i' && bytes == 8)
            {
                int64_t d;
                std::memcpy(&d, p, 8);
                v = (float) d;
            } else if (tc == 'i' && bytes == 4)
            {
                int32_t d;
                std::memcpy(&d, p, 4);
                v = (float) d;
            } else if (bytes == 1)
            {
                v = tc == 'u' ? (float) (uint8_t) *p : (float) (int8_t) *p;
            } else
            {
                err = path + ": unsupported dtype '" + descr + "'";
                return false;
            }
            out.data[(size_t) i] = v;
        }
        return true;
    }

    bool loadRaw(const std::string &path, int64_t elems, Tensor &out, std::string &err) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            err = "cannot open " + path;
            return false;
        }
        out.data.assign((size_t) elems, 0.f);
        f.read(reinterpret_cast<char *>(out.data.data()), (std::streamsize) (elems * 4));
        return true;
    }

    void saveNpy(const std::string &path, const Shape &shape, const float *data, size_t n) {
        std::string sh = "(";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
        }
        sh += ")";
        std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
        while ((10 + hdr.size() + 1) % 64 != 0)
        {
            hdr += ' ';
        }
        hdr += '\n';
        std::ofstream f(path, std::ios::binary);
        f.write("\x93NUMPY", 6);
        uint8_t ver[2] = {1, 0};
        f.write(reinterpret_cast<char *>(ver), 2);
        uint16_t hl = (uint16_t) hdr.size();
        f.write(reinterpret_cast<char *>(&hl), 2);
        f.write(hdr.data(), (std::streamsize) hdr.size());
        f.write(reinterpret_cast<const char *>(data), (std::streamsize) (n * 4));
    }

    // ---- minimal PNG writer (zlib stored/uncompressed blocks: valid PNG, no deps) ----
    uint32_t crc32of(const uint8_t *p, size_t n, uint32_t crc = 0xffffffffu) {
        static uint32_t T[256];
        static bool     init = false;
        if (!init)
        {
            for (uint32_t i = 0; i < 256; ++i)
            {
                uint32_t c = i;
                for (int k = 0; k < 8; ++k)
                {
                    c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
                }
                T[i] = c;
            }
            init = true;
        }
        for (size_t i = 0; i < n; ++i)
        {
            crc = T[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
        }
        return crc;
    }
    void putBE(std::vector<uint8_t> &v, uint32_t x) {
        v.push_back(x >> 24);
        v.push_back(x >> 16);
        v.push_back(x >> 8);
        v.push_back(x);
    }
    void chunk(std::ofstream &f, const char *type, const std::vector<uint8_t> &data) {
        std::vector<uint8_t> len;
        putBE(len, (uint32_t) data.size());
        f.write(reinterpret_cast<char *>(len.data()), 4);
        std::vector<uint8_t> tc(type, type + 4);
        std::vector<uint8_t> crcbuf = tc;
        crcbuf.insert(crcbuf.end(), data.begin(), data.end());
        f.write(type, 4);
        f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
        std::vector<uint8_t> crc;
        putBE(crc, crc32of(crcbuf.data(), crcbuf.size()) ^ 0xffffffffu);
        f.write(reinterpret_cast<char *>(crc.data()), 4);
    }
    void writePng(const std::string &path, int w, int h, int ch, const std::vector<uint8_t> &rgb) {
        // raw scanlines: filter byte 0 + row bytes
        std::vector<uint8_t> raw;
        raw.reserve((size_t) h * (w * ch + 1));
        for (int y = 0; y < h; ++y)
        {
            raw.push_back(0);
            raw.insert(raw.end(), rgb.begin() + (size_t) y * w * ch, rgb.begin() + (size_t) (y + 1) * w * ch);
        }
        // zlib stored blocks + adler32
        std::vector<uint8_t> z;
        z.push_back(0x78);
        z.push_back(0x01);
        size_t off = 0;
        while (off < raw.size())
        {
            size_t  n     = std::min<size_t>(65535, raw.size() - off);
            uint8_t final = (off + n >= raw.size()) ? 1 : 0;
            z.push_back(final);
            z.push_back(n & 0xff);
            z.push_back((n >> 8) & 0xff);
            z.push_back(~n & 0xff);
            z.push_back((~n >> 8) & 0xff);
            z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
            off += n;
        }
        uint32_t a = 1, b = 0;
        for (uint8_t c: raw)
        {
            a = (a + c) % 65521;
            b = (b + a) % 65521;
        }
        putBE(z, (b << 16) | a);

        std::ofstream f(path, std::ios::binary);
        const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
        f.write(reinterpret_cast<const char *>(sig), 8);
        std::vector<uint8_t> ihdr;
        putBE(ihdr, w);
        putBE(ihdr, h);
        ihdr.push_back(8);                             // bit depth
        ihdr.push_back(ch == 1 ? 0 : ch == 3 ? 2 : 6); // color type: gray / RGB / RGBA
        ihdr.insert(ihdr.end(), {0, 0, 0});
        chunk(f, "IHDR", ihdr);
        chunk(f, "IDAT", z);
        chunk(f, "IEND", {});
    }

    // Interpret an output tensor as an image and write a PNG (per-image min-max normalised to 0..255).
    // Handles [..,C,H,W] (C in {1,3,4}) and [..,H,W,C]; otherwise returns false.
    bool saveOutputPng(const std::string &path, const Shape &shape, const float *d) {
        int r = (int) shape.size();
        if (r < 2)
        {
            return false;
        }
        int64_t W = shape[r - 1], H = shape[r - 2], C = 1;
        bool    hwc = false;
        if (r >= 3 && (shape[r - 1] == 1 || shape[r - 1] == 3 || shape[r - 1] == 4) && shape[r - 1] < shape[r - 2])
        {
            C   = shape[r - 1];
            W   = shape[r - 2];
            H   = shape[r - 3];
            hwc = true; // [..,H,W,C]
        } else if (r >= 3 && (shape[r - 3] == 1 || shape[r - 3] == 3 || shape[r - 3] == 4))
        {
            C = shape[r - 3]; // [..,C,H,W]
        }
        if (W <= 0 || H <= 0 || (C != 1 && C != 3 && C != 4) || W * H * C > 64 * 1024 * 1024)
        {
            return false;
        }
        size_t plane = (size_t) W * H;
        float  lo = 1e30f, hi = -1e30f;
        for (size_t i = 0; i < plane * C; ++i)
        {
            lo = std::min(lo, d[i]);
            hi = std::max(hi, d[i]);
        }
        float                scale = hi > lo ? 255.f / (hi - lo) : 0.f;
        std::vector<uint8_t> img((size_t) W * H * C);
        for (int64_t y = 0; y < H; ++y)
        {
            for (int64_t x = 0; x < W; ++x)
            {
                for (int64_t c = 0; c < C; ++c)
                {
                    float v                  = hwc ? d[(y * W + x) * C + c] : d[c * plane + y * W + x];
                    img[(y * W + x) * C + c] = (uint8_t) std::lround(std::min(255.f, std::max(0.f, (v - lo) * scale)));
                }
            }
        }
        writePng(path, (int) W, (int) H, (int) C, img);
        return true;
    }

    struct Metrics {
        double cosine = 0, relL2 = 0, maxAbs = 0, psnr = 0, snr = 0;
        int    nan    = 0;
        bool   sizeOk = true;
    };
    Metrics compareAll(const float *a, size_t na, const float *b, size_t nb) {
        Metrics m;
        m.sizeOk   = na == nb;
        size_t n   = std::min(na, nb);
        double dot = 0, sa = 0, sb = 0, err = 0, lo = 1e300, hi = -1e300;
        for (size_t i = 0; i < n; ++i)
        {
            double x = a[i], y = b[i];
            if (std::isnan(x))
            {
                ++m.nan;
                continue;
            }
            dot += x * y;
            sa += x * x;
            sb += y * y;
            double e = x - y;
            err += e * e;
            m.maxAbs = std::max(m.maxAbs, std::fabs(e));
            lo       = std::min(lo, y);
            hi       = std::max(hi, y);
        }
        m.cosine    = dot / (std::sqrt(sa) * std::sqrt(sb) + 1e-12);
        m.relL2     = std::sqrt(err) / (std::sqrt(sb) + 1e-12);
        double mse  = err / std::max<size_t>(1, n);
        double peak = hi - lo;
        m.psnr      = (mse > 0 && peak > 0) ? 20.0 * std::log10(peak / std::sqrt(mse)) : 1e9;
        m.snr       = err > 0 ? 10.0 * std::log10(sb / err) : 1e9;
        return m;
    }

} // namespace

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s config.json\n", argv[0]);
        return 1;
    }
    std::string cfgPath = argv[1], base = dirOf(cfgPath);
    std::string text = readFile(cfgPath);
    if (text.empty())
    {
        fprintf(stderr, "cannot read config %s\n", cfgPath.c_str());
        return 1;
    }
    JsonValue js = JsonParser::parse(text);
    if (!js.isObject())
    {
        fprintf(stderr, "config is not a JSON object\n");
        return 1;
    }
    auto str = [&](const char *k, const std::string &d) {
        auto *j = js.get(k);
        return j ? j->asStr(d) : d;
    };
    auto flag = [&](const char *k, bool d) {
        auto *j = js.get(k);
        return j ? j->asBool(d) : d;
    };

    std::string model = str("model", "");
    if (model.empty())
    {
        fprintf(stderr, "config missing \"model\"\n");
        return 1;
    }
    Config cfg;
    cfg.backend                = backendFromStr(str("backend", "vulkan"));
    cfg.precision              = str("precision", "fp16") == "fp32" ? Precision::Fp32 : Precision::Fp16;
    cfg.freeWeightsAfterUpload = true;
    cfg.cacheWeights           = !flag("no_weight_cache", true);
    cfg.timing                 = flag("timing", false);
    cfg.profile                = flag("profile", false);
    // "winograd": "auto"|"on"|"off" forces the 3x3-conv kernel choice. "on"/"off" skip the per-shape
    // timing measurement, so the kernel selection (and the output bits) is deterministic across runs.
    cfg.winograd = winogradFromStr(str("winograd", "auto"));
    cfg.tuning   = tuningFromStr(str("tuning", "fast"));
    if (auto *j = js.get("max_submit_nodes"))
    {
        cfg.maxSubmitNodes = (int) j->asNum(cfg.maxSubmitNodes);
    }
    double tol = js.get("tolerance") ? js.get("tolerance")->asNum(0.999) : 0.999;

    // Unified per-model cache file. "cache": path (default "<model>.cache" next to the model). With
    // "generate_cache": true, populate it first in an UNTIMED throwaway load (shader compile + conv
    // autotune + Winograd transform), written when that session is destroyed, so the timed load below is
    // a warm start — the cache-build cost is excluded from timing_ms.
    std::string modelPath = resolve(base, model);
    std::string cacheFile = str("cache", "");
    if (!cacheFile.empty())
    {
        cacheFile = resolve(base, cacheFile);
    }
    if (flag("generate_cache", false))
    {
        printf("generating cache (untimed) ...\n");
        Runtime::load(modelPath, cfg, cacheFile); // throwaway; ~Session writes the cache
    }

    auto t0   = Clock::now();
    auto sess = Runtime::load(modelPath, cfg, cacheFile);
    if (!sess)
    {
        fprintf(stderr, "failed to load %s\n", model.c_str());
        return 1;
    }
    double loadMs = msSince(t0);
    auto   infos  = sess->inputInfo();

    // --- inputs: .npy / raw .bin / none (runtime-only) ---
    const JsonValue      *jin     = js.get("inputs");
    bool                  haveIns = jin && ((jin->type == JsonValue::kArray && !jin->arr.empty()) || (jin->type == JsonValue::kObject && !jin->obj.empty()));
    std::vector<IOTensor> ins;
    for (size_t i = 0; i < infos.size(); ++i)
    {
        IOTensor t;
        t.name  = infos[i].name;
        t.shape = infos[i].shape;
        t.dtype = DType::Float32;
        t.data.assign((size_t) infos[i].elems * 4, 0); // default zeros (runtime-only)
        if (haveIns)
        {
            std::string p;
            if (jin->type == JsonValue::kArray)
            {
                if (i < jin->arr.size())
                {
                    p = jin->arr[i].asStr("");
                }
            } else if (auto *j = jin->get(infos[i].name))
            { p = j->asStr(""); }
            if (p.empty())
            {
                fprintf(stderr, "no input for '%s'\n", infos[i].name.c_str());
                return 1;
            }
            Tensor      tn;
            std::string err;
            bool        ok = endsWith(p, ".npy") ? loadNpy(resolve(base, p), tn, err) : loadRaw(resolve(base, p), infos[i].elems, tn, err);
            if (!ok)
            {
                fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            if ((int64_t) tn.data.size() != infos[i].elems)
            {
                fprintf(stderr, "input '%s': %s has %zu elems, model expects %lld\n", infos[i].name.c_str(), p.c_str(), tn.data.size(), (long long) infos[i].elems);
                return 1;
            }
            std::memcpy(t.data.data(), tn.data.data(), t.data.size());
            printf("input  '%s'  %s  <- %s\n", t.name.c_str(), shapeStr(t.shape).c_str(), baseName(p).c_str());
        }
        ins.push_back(std::move(t));
    }
    if (!haveIns)
    {
        printf("(no inputs given -> zero-filled, runtime-only)\n");
    }

    std::vector<IOTensor> outs;
    auto                  t1 = Clock::now();
    if (sess->run(ins, outs) != Status::Ok)
    {
        fprintf(stderr, "inference failed\n");
        return 2;
    }
    double runMs = msSince(t1);
    printf("load %.1f ms   run %.1f ms\n", loadMs, runMs);

    // --- save outputs (npy / raw / png) ---
    std::vector<std::string> saveFmts;
    if (auto *j = js.get("save"))
    {
        if (j->type == JsonValue::kArray)
        {
            for (auto &e: j->arr)
            {
                saveFmts.push_back(e.asStr(""));
            }
        }
    }
    std::string saveDir = str("save_dir", str("save_outputs", "."));
    bool        wantNpy = false, wantRaw = false, wantPng = false;
    for (auto &s: saveFmts)
    {
        wantNpy = wantNpy || s == "npy";
        wantRaw = wantRaw || s == "raw" || s == "bin";
        wantPng = wantPng || s == "png";
    }
    std::map<std::string, std::vector<std::string>> saved;
    if (haveIns && (wantNpy || wantRaw || wantPng))
    {
        for (auto &o: outs)
        {
            std::string nm  = sanitize(o.name);
            size_t      cnt = o.data.size() / 4;
            if (wantNpy)
            {
                std::string p = resolve(base, saveDir) + "/" + nm + ".npy";
                saveNpy(p, o.shape, o.f32(), cnt);
                saved[o.name].push_back(baseName(p));
            }
            if (wantRaw)
            {
                std::string   p = resolve(base, saveDir) + "/" + nm + ".raw";
                std::ofstream rf(p, std::ios::binary);
                rf.write(reinterpret_cast<const char *>(o.f32()), (std::streamsize) (cnt * 4)); // fp32 row-major
                saved[o.name].push_back(baseName(p));
            }
            if (wantPng)
            {
                std::string p = resolve(base, saveDir) + "/" + nm + ".png";
                if (saveOutputPng(p, o.shape, o.f32()))
                {
                    saved[o.name].push_back(baseName(p));
                }
            }
            printf("output '%s'  %s\n", o.name.c_str(), shapeStr(o.shape).c_str());
        }
    }

    // --- golden comparison ---
    const JsonValue         *jgold = js.get("golden") ? js.get("golden") : js.get("outputs");
    std::vector<std::string> wantMetrics;
    if (auto *j = js.get("metrics"))
    {
        if (j->type == JsonValue::kArray)
        {
            for (auto &e: j->arr)
            {
                wantMetrics.push_back(e.asStr(""));
            }
        }
    }
    if (wantMetrics.empty())
    {
        wantMetrics = {"cosine", "psnr", "snr", "relL2", "max"};
    }
    auto wants = [&](const char *m) {
        for (auto &s: wantMetrics)
        {
            if (s == m)
            {
                return true;
            }
        }
        return false;
    };

    std::map<std::string, Metrics> results;
    bool                           allOk = true;
    if (haveIns && jgold && jgold->type == JsonValue::kObject && !jgold->obj.empty())
    {
        printf("\nvalidation vs golden (tolerance cos >= %.4f):\n", tol);
        for (auto &kv: jgold->obj)
        {
            IOTensor *o = nullptr;
            for (auto &t: outs)
            {
                if (t.name == kv.first)
                {
                    o = &t;
                }
            }
            if (!o)
            {
                printf("  %-16s NOT-AN-OUTPUT\n", kv.first.c_str());
                allOk = false;
                continue;
            }
            Tensor      g;
            std::string err;
            if (!loadNpy(resolve(base, kv.second.asStr("")), g, err))
            {
                fprintf(stderr, "%s\n", err.c_str());
                allOk = false;
                continue;
            }
            Metrics m         = compareAll(o->f32(), o->data.size() / 4, g.data.data(), g.data.size());
            results[kv.first] = m;
            bool ok           = m.sizeOk && m.nan == 0 && m.cosine >= tol;
            allOk             = allOk && ok;
            std::string line  = "  " + kv.first;
            line.resize(18, ' ');
            char        buf[256];
            std::string detail;
            if (wants("cosine"))
            {
                snprintf(buf, sizeof(buf), " cos=%.6f", m.cosine);
                detail += buf;
            }
            if (wants("psnr"))
            {
                snprintf(buf, sizeof(buf), " psnr=%.2fdB", m.psnr);
                detail += buf;
            }
            if (wants("snr"))
            {
                snprintf(buf, sizeof(buf), " snr=%.2fdB", m.snr);
                detail += buf;
            }
            if (wants("relL2"))
            {
                snprintf(buf, sizeof(buf), " relL2=%.3e", m.relL2);
                detail += buf;
            }
            if (wants("max"))
            {
                snprintf(buf, sizeof(buf), " max|d|=%.3e", m.maxAbs);
                detail += buf;
            }
            printf("%s%s nan=%d  %s\n", line.c_str(), detail.c_str(), m.nan, !m.sizeOk ? "SIZE-MISMATCH" : (ok ? "PASS" : "FAIL"));
        }
        printf("%s\n", allOk ? "ALL OUTPUTS PASS" : "SOME OUTPUTS FAIL");
    }

    // --- result json: timing + (optional) per-op profile + per-output metrics ---
    std::string resPath = str("result", "");
    if (!resPath.empty())
    {
        std::ofstream r(resolve(base, resPath));
        r << "{\n";
        r << "  \"model\": \"" << sanitize(baseName(model)) << "\",\n";
        r << "  \"backend\": \"" << backendName(cfg.backend) << "\",\n";
        r << "  \"timing_ms\": { \"load\": " << loadMs << ", \"run\": " << runMs << " },\n";
        if (cfg.profile)
        {
            std::map<std::string, double> byType;
            r << "  \"profile\": [";
            const auto &recs = sess->profiler().records();
            for (size_t i = 0; i < recs.size(); ++i)
            {
                const auto &op = recs[i];
                byType[opTypeName(op.type)] += op.gpuMs > 0 ? op.gpuMs : 0;
                r << (i ? ",\n    " : "\n    ") << "{ \"name\": \"" << sanitize(op.name) << "\", \"type\": \"" << opTypeName(op.type) << "\", \"gpu_ms\": " << op.gpuMs << ", \"cpu_ms\": " << op.cpuMs << " }";
            }
            r << "\n  ],\n";
            r << "  \"profile_by_type_ms\": {";
            bool first = true;
            for (auto &kv: byType)
            {
                r << (first ? " " : ", ") << "\"" << kv.first << "\": " << kv.second;
                first = false;
            }
            r << " },\n";
            r << "  \"gpu_total_ms\": " << sess->profiler().totalGpuMs() << ",\n";
        }
        r << "  \"outputs\": [";
        bool first = true;
        for (auto &o: outs)
        {
            r << (first ? "\n    " : ",\n    ") << "{ \"name\": \"" << sanitize(o.name) << "\", \"shape\": [";
            for (size_t k = 0; k < o.shape.size(); ++k)
            {
                r << (k ? ", " : "") << o.shape[k];
            }
            r << "]";
            if (saved.count(o.name))
            {
                r << ", \"saved\": [";
                for (size_t k = 0; k < saved[o.name].size(); ++k)
                {
                    r << (k ? ", " : "") << "\"" << saved[o.name][k] << "\"";
                }
                r << "]";
            }
            if (results.count(o.name))
            {
                const Metrics &m = results[o.name];
                r << ", \"metrics\": { \"cosine\": " << m.cosine << ", \"psnr\": " << m.psnr << ", \"snr\": " << m.snr << ", \"relL2\": " << m.relL2 << ", \"max\": " << m.maxAbs << ", \"nan\": " << m.nan << " }";
                r << ", \"pass\": " << ((m.sizeOk && m.nan == 0 && m.cosine >= tol) ? "true" : "false");
            }
            r << " }";
            first = false;
        }
        r << "\n  ]\n}\n";
        printf("wrote result json -> %s\n", resPath.c_str());
    }

    return allOk ? 0 : 3;
}
