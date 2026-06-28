// vknn_validate - run a model from a small JSON config, feeding .npy inputs and checking each output
// against a golden .npy. The .npy header carries shape + dtype, so the user never hand-specifies them.
//
//   vknn_validate config.json
//
// config.json (paths are relative to the config file's directory, or absolute):
//   {
//     "model": "encoder8_fp16.vxm",        // .onnx or .vxm (required)
//     "backend": "vulkan",                  // vulkan | cpu        (default vulkan)
//     "precision": "fp16",                  // fp16 | fp32         (default fp16)
//     "no_weight_cache": true,              // optional (default true)
//     "tolerance": 0.999,                   // cosine pass threshold (default 0.999)
//     "inputs": ["image.npy", "intr.npy"],  // positional (model input order); or { "image": "image.npy" }
//     "outputs": { "means": "means_gold.npy", "scales": "scales_gold.npy" }, // name -> golden (optional)
//     "save_outputs": "out_dir"             // optional: write every output as <name>.npy here
//   }
//
// Exit code 0 iff every checked output passes (cosine >= tolerance and no NaN).
#include "core/json.h"
#include "vknn/dtype.h"
#include "vknn/session.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    std::string dirOf(const std::string &p) {
        auto s = p.find_last_of("/\\");
        return s == std::string::npos ? std::string() : p.substr(0, s + 1);
    }
    std::string resolve(const std::string &base, const std::string &p) {
        return (p.empty() || p[0] == '/') ? p : base + p;
    }

    std::string readFile(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            return {};
        }
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    // ---- minimal .npy (v1.0/2.0) reader: any numeric dtype -> fp32, C-order only ----
    struct Npy {
        std::vector<int64_t> shape;
        std::vector<float>   data;
        int64_t              elems() const {
            int64_t n = 1;
            for (auto d: shape)
            {
                n *= d;
            }
            return shape.empty() ? 0 : n;
        }
    };

    bool loadNpy(const std::string &path, Npy &out, std::string &err) {
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

        auto field = [&](const char *key) -> std::string {
            auto k = hdr.find(key);
            if (k == std::string::npos)
            {
                return {};
            }
            k = hdr.find(':', k) + 1;
            while (k < hdr.size() && (hdr[k] == ' ' || hdr[k] == '\''))
            {
                ++k;
            }
            auto e = k;
            while (e < hdr.size() && hdr[e] != '\'' && hdr[e] != ',' && hdr[e] != '}' && hdr[e] != ')')
            {
                ++e;
            }
            return hdr.substr(k, e - k);
        };
        std::string descr = field("descr");
        if (hdr.find("'fortran_order': True") != std::string::npos)
        {
            err = path + ": fortran_order=True unsupported";
            return false;
        }
        // shape tuple, e.g. (1, 8, 3, 224, 224)
        auto sp = hdr.find("'shape'");
        auto lp = hdr.find('(', sp), rp = hdr.find(')', lp);
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
        int64_t n = out.elems();
        out.data.resize((size_t) std::max<int64_t>(n, 0));

        // descr like "<f4", "<f2", "<f8", "<i8", "<i4", "|i1", "|u1" (assume little-endian / native)
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

    void saveNpy(const std::string &path, const Shape &shape, const std::vector<float> &data) {
        std::string sh = "(";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
        }
        sh += ")";
        std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
        size_t      pre = 10; // magic(6)+ver(2)+len(2)
        while ((pre + hdr.size() + 1) % 64 != 0)
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
        f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * 4));
    }

    void compare(const std::string &name, const std::vector<float> &a, const std::vector<float> &b, double tol, bool &allOk) {
        size_t n   = std::min(a.size(), b.size());
        double dot = 0, na = 0, nb = 0, maxd = 0, errn = 0;
        int    nan = 0;
        for (size_t i = 0; i < n; ++i)
        {
            double x = a[i], y = b[i];
            if (std::isnan(x))
            {
                ++nan;
                continue;
            }
            dot += x * y;
            na += x * x;
            nb += y * y;
            maxd = std::max(maxd, std::fabs(x - y));
            errn += (x - y) * (x - y);
        }
        double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
        double rel = std::sqrt(errn) / (std::sqrt(nb) + 1e-12);
        bool   ok  = nan == 0 && a.size() == b.size() && cos >= tol;
        allOk      = allOk && ok;
        printf("  %-16s cos=%.6f  relL2=%.3e  max|d|=%.3e  nan=%d  %s\n", name.c_str(), cos, rel, maxd, nan, a.size() != b.size() ? "SIZE-MISMATCH" : (ok ? "PASS" : "FAIL"));
    }

} // namespace

int main(int argc, char **argv) {
    if (argc < 2)
    {
        printf("usage: %s config.json\n", argv[0]);
        return 1;
    }
    std::string cfgPath = argv[1];
    std::string base    = dirOf(cfgPath);
    std::string text    = readFile(cfgPath);
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

    std::string model = str("model", "");
    if (model.empty())
    {
        fprintf(stderr, "config missing \"model\"\n");
        return 1;
    }
    Config cfg;
    cfg.backend                = backendFromStr(str("backend", "vulkan"));
    cfg.precision              = str("precision", "fp16") == "fp32" ? Precision::kFp32 : Precision::kFp16;
    cfg.freeWeightsAfterUpload = true;
    if (auto *j = js.get("no_weight_cache"))
    {
        cfg.cacheWeights = !j->asBool(true);
    }
    if (auto *j = js.get("timing"))
    {
        cfg.timing = j->asBool(false); // engine prints pack / submit+gpu / unpack
    }
    double tol = js.get("tolerance") ? js.get("tolerance")->asNum(0.999) : 0.999;

    auto sess = Runtime::load(resolve(base, model), cfg);
    if (!sess)
    {
        fprintf(stderr, "failed to load %s\n", model.c_str());
        return 1;
    }
    auto infos = sess->inputInfo();

    // Resolve each model input to a .npy path: "inputs" is either an array (positional) or an object
    // keyed by input name.
    std::vector<IOTensor> ins;
    const JsonValue      *jin = js.get("inputs");
    for (size_t i = 0; i < infos.size(); ++i)
    {
        std::string npyPath;
        if (jin && jin->type == JsonValue::kArray)
        {
            if (i < jin->arr.size())
            {
                npyPath = jin->arr[i].asStr("");
            }
        } else if (jin && jin->type == JsonValue::kObject)
        {
            if (auto *j = jin->get(infos[i].name))
            {
                npyPath = j->asStr("");
            }
        }
        if (npyPath.empty())
        {
            fprintf(stderr, "no input .npy for '%s'\n", infos[i].name.c_str());
            return 1;
        }
        Npy         npy;
        std::string err;
        if (!loadNpy(resolve(base, npyPath), npy, err))
        {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        if (npy.elems() != infos[i].elems)
        {
            fprintf(stderr, "input '%s': %s has %lld elems, model expects %lld\n", infos[i].name.c_str(), npyPath.c_str(), (long long) npy.elems(),
                    (long long) infos[i].elems);
            return 1;
        }
        IOTensor t;
        t.name  = infos[i].name;
        t.shape = infos[i].shape;
        t.dtype = DType::kFloat32;
        t.data.resize(npy.data.size() * 4);
        std::memcpy(t.data.data(), npy.data.data(), t.data.size());
        printf("input  '%s'  %s  <- %s\n", t.name.c_str(), shapeStr(t.shape).c_str(), npyPath.c_str());
        ins.push_back(std::move(t));
    }

    std::vector<IOTensor> outs;
    if (sess->run(ins, outs) != Status::kOk)
    {
        fprintf(stderr, "inference failed\n");
        return 2;
    }

    std::string saveDir = str("save_outputs", "");
    if (!saveDir.empty())
    {
        for (auto &o: outs)
        {
            std::string safe = o.name;
            for (char &c: safe)
            {
                if (c == '/' || c == ':')
                {
                    c = '_';
                }
            }
            saveNpy(resolve(base, saveDir) + "/" + safe + ".npy", o.shape, std::vector<float>(o.f32(), o.f32() + o.data.size() / 4));
            printf("output '%s'  %s  -> %s/%s.npy\n", o.name.c_str(), shapeStr(o.shape).c_str(), saveDir.c_str(), safe.c_str());
        }
    }

    const JsonValue *jout = js.get("outputs");
    if (!jout || jout->type != JsonValue::kObject || jout->obj.empty())
    {
        printf("(no \"outputs\" goldens to validate)\n");
        return 0;
    }
    printf("\nvalidation vs golden .npy (tolerance cos >= %.4f):\n", tol);
    bool allOk = true;
    for (auto &kv: jout->obj)
    {
        const std::string &oname = kv.first;
        IOTensor          *o     = nullptr;
        for (auto &t: outs)
        {
            if (t.name == oname)
            {
                o = &t;
                break;
            }
        }
        if (!o)
        {
            printf("  %-16s NOT-AN-OUTPUT\n", oname.c_str());
            allOk = false;
            continue;
        }
        Npy         g;
        std::string err;
        if (!loadNpy(resolve(base, kv.second.asStr("")), g, err))
        {
            fprintf(stderr, "%s\n", err.c_str());
            allOk = false;
            continue;
        }
        compare(oname, std::vector<float>(o->f32(), o->f32() + o->data.size() / 4), g.data, tol, allOk);
    }
    printf("%s\n", allOk ? "ALL OUTPUTS PASS" : "SOME OUTPUTS FAIL");
    return allOk ? 0 : 3;
}
