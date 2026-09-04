// Implementation of the Model/Tensor facade on top of Session.
#include "vknn/model.h"
#include "vknn/ion.h"
#include "vknn/session.h"
#include <algorithm>
#include <map>
#include <cstring>

namespace vknn {

    // Join extents with 'x' (e.g. "1x3x224x224"). An empty shape is a rank-0 tensor, rendered "scalar".
    static std::string shapeJoin(const std::vector<int64_t> &s) {
        std::string r;
        for (size_t i = 0; i < s.size(); ++i)
        {
            r += (i ? "x" : "") + std::to_string(s[i]);
        }
        return r.empty() ? "scalar" : r;
    }

    std::string TensorInfo::shapeString() const {
        return shapeJoin(shape);
    }

    // ----------------------------- Tensor -----------------------------
    Tensor::Tensor(std::vector<float> data, std::vector<int64_t> shape, std::string name):
        name_(std::move(name)), shape_(std::move(shape)), data_(std::move(data)) {
    }

    Tensor::Tensor(std::vector<float> data): shape_ {(int64_t) data.size()}, data_(std::move(data)) {
    }

    Tensor Tensor::fromDmaBuf(int fd, std::vector<int64_t> shape, std::string name, TensorFormat layout, DType dtype) {
        Tensor t;
        t.shape_        = std::move(shape);
        t.name_         = std::move(name);
        t.fd_           = fd;
        t.dmaBufFormat_ = layout;
        t.dmaBufDtype_  = dtype;
        return t;
    }

    // A rank-0 shape holds one element (the value a scalar output carries), as a host Tensor built
    // from a one-element vector does.
    int64_t Tensor::shapeElems() const noexcept {
        int64_t n = 1;
        for (int64_t d: shape_)
        {
            n *= d;
        }
        return n;
    }
    Tensor Tensor::pinned(std::shared_ptr<PinnedHostMemory> block, std::vector<int64_t> shape, std::string name) {
        Tensor t;
        t.shape_  = std::move(shape);
        t.name_   = std::move(name);
        t.pinned_ = std::move(block);
        return t;
    }
    Tensor Tensor::pinned(std::vector<int64_t> shape, std::string name) {
        int64_t n = 1;
        for (int64_t d: shape)
        {
            n *= d;
        }
        std::shared_ptr<PinnedHostMemory> block = PinnedHostMemory::alloc((size_t) std::max<int64_t>(n, 0) * sizeof(float));
        if (!block)
        {
            // Out of memory: a host tensor of the same size keeps data() valid and the run correct
            // (one copy per direction instead of none).
            return Tensor(std::vector<float>((size_t) std::max<int64_t>(n, 0), 0.f), std::move(shape), std::move(name));
        }
        return pinned(std::move(block), std::move(shape), std::move(name));
    }
    Tensor Tensor::toPinned(std::vector<int64_t> shape, std::string name) {
        return pinned(std::move(shape), std::move(name));
    }
    Tensor Tensor::toDmaBuf(int fd, std::vector<int64_t> shape, std::string name, TensorFormat layout, DType dtype) {
        // same carrier; output vs input is by list position in Model::run
        return fromDmaBuf(fd, std::move(shape), std::move(name), layout, dtype);
    }

    const Tensor *findTensor(const std::vector<Tensor> &tensors, const std::string &name) {
        for (const auto &t: tensors)
        {
            if (t.name() == name)
            {
                return &t;
            }
        }
        return nullptr;
    }

    std::string Tensor::shapeString() const {
        return shapeJoin(shape_);
    }

    int64_t Tensor::argmax() const {
        const int64_t n = size();
        if (n <= 0)
        {
            return -1;
        }
        const float *v = data();
        return (int64_t) (std::max_element(v, v + n) - v);
    }

    float Tensor::max() const {
        const int64_t n = size();
        return n <= 0 ? 0.f : *std::max_element(data(), data() + n);
    }

    // ----------------------------- Model -----------------------------
    Model Model::load(const std::string &onnxPath, Precision precision) {
        Config cfg; // defaults: Vulkan backend, CPU fallback
        cfg.precision = precision;
        return load(onnxPath, cfg);
    }

    static bool endsWith(const std::string &s, const std::string &suf) {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    }

    Model Model::load(const std::string &path, const Config &cfg) {
        Model m;
        // .vxm = pre-optimized binary (skip ONNX parse + passes); anything else = ONNX.
        auto s  = endsWith(path, ".vxm") ? Session::createFromVxm(path, cfg) : Session::createFromOnnx(path, cfg);
        m.sess_ = std::shared_ptr<Session>(s.release());
        return m;
    }

    bool Model::save(const std::string &vxmPath) const {
        return sess_ && sess_->saveOptimized(vxmPath);
    }

    static std::vector<TensorInfo> toInfos(const std::vector<IOInfo> &v) {
        std::vector<TensorInfo> out;
        for (const auto &i: v)
        {
            TensorInfo t;
            t.name  = i.name;
            t.shape = i.shape;
            t.dtype = DType::Float32; // values cross the high-level API as fp32
            t.count = i.elems;
            out.push_back(std::move(t));
        }
        return out;
    }

    std::vector<TensorInfo> Model::inputs() const {
        return sess_ ? toInfos(sess_->inputInfo()) : std::vector<TensorInfo> {};
    }
    std::vector<TensorInfo> Model::outputs() const {
        return sess_ ? toInfos(sess_->outputInfo()) : std::vector<TensorInfo> {};
    }

    // Product of the extents. An empty shape reports 0 (unknown/unshaped), not the empty product 1.
    static int64_t elemCount(const Shape &s) {
        int64_t n = 1;
        for (int64_t d: s)
        {
            n *= d;
        }
        return s.empty() ? 0 : n;
    }

    std::vector<Tensor> Model::run(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs) {
        if (!sess_)
        {
            return {};
        }
        // Build IOTensors, filling name/shape from the model where the caller left them blank.
        auto                  info = sess_->inputInfo();
        std::vector<IOTensor> ins(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const Tensor &t = inputs[i];
            ins[i].name     = !t.name().empty() ? t.name() : (i < info.size() ? info[i].name : "");
            ins[i].shape    = !t.shape().empty() ? t.shape() : (i < info.size() ? info[i].shape : Shape {});
            ins[i].dtype    = DType::Float32;
            if (t.dmaBufFd() >= 0)
            {
                ins[i].dmaBufFd     = t.dmaBufFd(); // zero-copy: the engine reads this fd as the GPU input buffer
                ins[i].dmaBufFormat = t.dmaBufFormat();
                ins[i].dmaBufDtype  = t.dmaBufDtype();
            } else if (t.pinnedBlock())
            {
                ins[i].pinned = t.pinnedBlock(); // pinned: the engine binds (or copies from) the block
            } else
            {
                const uint8_t *p = reinterpret_cast<const uint8_t *>(t.data());
                ins[i].data.assign(p, p + t.size() * sizeof(float));
            }
        }
        // Pre-fill `outs` with zero-copy output bindings; run() writes each bound output into the caller's
        // fd and refills `outs` with the results.
        // A pinned output block holds fp32 (Tensor::data() reads it as such). The session writes an
        // output at the model's declared dtype, so only an fp32-declared output binds its block for
        // the run; any other is read back through the host and widened into the block below.
        const std::vector<IOInfo> outInfo = sess_->outputInfo();
        // The session reports each output under the model's name; a binding may carry an empty
        // name for a single-output model, so resolve it once and key everything by the model's.
        auto resolvedName = [&](const std::string &name) {
            return name.empty() && outInfo.size() == 1 ? outInfo[0].name : name;
        };
        auto declaredDtype = [&](const std::string &name) {
            for (const IOInfo &oi: outInfo)
            {
                if (oi.name == name)
                {
                    return oi.dtype;
                }
            }
            return DType::Float32;
        };
        std::vector<IOTensor>                                    outs;
        std::map<std::string, std::shared_ptr<PinnedHostMemory>> widenInto; // non-fp32 outputs: block filled after the run
        for (const auto &o: outputs)
        {
            if (o.dmaBufFd() >= 0 || o.pinnedBlock())
            {
                IOTensor b;
                b.name         = o.name();
                b.shape        = o.shape();
                b.dmaBufFd     = o.dmaBufFd();
                b.dmaBufFormat = o.dmaBufFormat();
                b.dmaBufDtype  = o.dmaBufDtype();
                if (o.pinnedBlock() && o.dmaBufFd() < 0)
                {
                    const std::string name = resolvedName(o.name());
                    if (declaredDtype(name) == DType::Float32)
                    {
                        b.pinned = o.pinnedBlock();
                    } else
                    {
                        widenInto[name] = o.pinnedBlock();
                    }
                }
                outs.push_back(std::move(b));
            }
        }
        if (sess_->run(ins, outs) != Status::Ok)
        {
            return {};
        }
        std::vector<Tensor> result;
        for (auto &o: outs)
        {
            if (o.dmaBufFd >= 0)
            {
                result.emplace_back(std::vector<float> {}, o.shape, o.name); // delivered to the caller's fd
            } else if (o.pinned)
            {
                result.push_back(Tensor::pinned(o.pinned, o.shape, o.name)); // delivered into the caller's block
            } else if (auto wi = widenInto.find(o.name); wi != widenInto.end())
            {
                // A non-fp32 output bound to a block: widen the declared-dtype bytes into it.
                std::vector<float> wide = o.toFloat32();
                std::memcpy(wi->second->data(), wide.data(), std::min(wide.size() * sizeof(float), wi->second->bytes()));
                result.push_back(Tensor::pinned(wi->second, o.shape, o.name));
            } else
            {
                // Widen by the declared dtype: a non-fp32 output (fp16 logits, uint8/int64) read as raw
                // fp32 would 2x/4x-OOB-read its buffer or return garbage. toFloat32 sizes from the
                // payload length, so it also preserves a rank-0 scalar; the fp32 path stays bit-exact.
                result.emplace_back(o.toFloat32(), o.shape, o.name);
            }
        }
        return result;
    }

    std::vector<Tensor> Model::run(const Tensor &input) {
        return run(std::vector<Tensor> {input}, {});
    }

    Tensor Model::run(const std::vector<float> &input) {
        auto  info  = sess_ ? sess_->inputInfo() : std::vector<IOInfo> {};
        Shape shape = info.empty() ? Shape {(int64_t) input.size()} : info[0].shape;
        auto  outs  = run(std::vector<Tensor> {Tensor(input, shape)});
        return outs.empty() ? Tensor {} : outs[0];
    }

} // namespace vknn
