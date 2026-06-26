// Implementation of the friendly Model/Tensor facade on top of Session.
#include "vx/model.h"

#include <algorithm>

#include "vx/ion.h"
#include "vx/session.h"

namespace vx {

static std::string shapeJoin(const std::vector<int64_t>& s) {
  std::string r;
  for (size_t i = 0; i < s.size(); ++i)
    r += (i ? "x" : "") + std::to_string(s[i]);
  return r.empty() ? "scalar" : r;
}

std::string TensorInfo::shapeString() const {
  return shapeJoin(shape);
}

// ----------------------------- Tensor -----------------------------
Tensor::Tensor(std::vector<float> data, std::vector<int64_t> shape, std::string name)
    : name_(std::move(name)), shape_(std::move(shape)), data_(std::move(data)) {}

Tensor::Tensor(std::vector<float> data) : shape_{(int64_t)data.size()}, data_(std::move(data)) {}

Tensor Tensor::fromDmaBuf(int fd, std::vector<int64_t> shape, std::string name) {
  Tensor t;
  t.shape_ = std::move(shape);
  t.name_ = std::move(name);
  t.fd_ = fd;
  return t;
}

const Tensor* findTensor(const std::vector<Tensor>& tensors, const std::string& name) {
  for (const auto& t : tensors)
    if (t.name() == name)
      return &t;
  return nullptr;
}

std::string Tensor::shapeString() const {
  return shapeJoin(shape_);
}

int64_t Tensor::argmax() const {
  if (data_.empty())
    return -1;
  return (int64_t)(std::max_element(data_.begin(), data_.end()) - data_.begin());
}

float Tensor::max() const {
  return data_.empty() ? 0.f : *std::max_element(data_.begin(), data_.end());
}

// ----------------------------- Model -----------------------------
Model Model::load(const std::string& onnxPath, Precision precision) {
  Config cfg;  // defaults: Vulkan backend, CPU fallback
  cfg.precision = precision;
  return load(onnxPath, cfg);
}

static bool endsWith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

Model Model::load(const std::string& path, const Config& cfg) {
  Model m;
  // .vxm = pre-optimized binary (skip ONNX parse + passes); anything else = ONNX.
  auto s = endsWith(path, ".vxm") ? Session::createFromVxm(path, cfg)
                                  : Session::createFromOnnx(path, cfg);
  m.sess_ = std::shared_ptr<Session>(s.release());
  return m;
}

bool Model::save(const std::string& vxmPath) const {
  return sess_ && sess_->saveOptimized(vxmPath);
}

static std::vector<TensorInfo> toInfos(const std::vector<IOInfo>& v) {
  std::vector<TensorInfo> out;
  for (const auto& i : v) {
    TensorInfo t;
    t.name = i.name;
    t.shape = i.shape;
    t.dtype = DType::kFloat32;  // values cross the high-level API as fp32
    t.count = i.elems;
    out.push_back(std::move(t));
  }
  return out;
}

std::vector<TensorInfo> Model::inputs() const {
  return sess_ ? toInfos(sess_->inputInfo()) : std::vector<TensorInfo>{};
}
std::vector<TensorInfo> Model::outputs() const {
  return sess_ ? toInfos(sess_->outputInfo()) : std::vector<TensorInfo>{};
}

std::vector<Tensor> Model::run(const std::vector<Tensor>& inputs) {
  if (!sess_)
    return {};
  // Build IOTensors, filling name/shape from the model where the caller left them blank.
  auto info = sess_->inputInfo();
  std::vector<IOTensor> ins(inputs.size());
  std::vector<std::unique_ptr<IonBuffer>> dmabufs;  // keep mappings alive for the run
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Tensor& t = inputs[i];
    ins[i].name = !t.name().empty() ? t.name() : (i < info.size() ? info[i].name : "");
    ins[i].shape = !t.shape().empty() ? t.shape() : (i < info.size() ? info[i].shape : Shape{});
    ins[i].dtype = DType::kFloat32;
    if (t.dmaBufFd() >= 0) {
      // zero-copy: read the input straight from the DMA-BUF (no caller-side host buffer).
      int64_t n = numElements(ins[i].shape);
      auto ion = IonBuffer::wrapFd(t.dmaBufFd(), (size_t)n * sizeof(float));
      if (ion && ion->data()) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(ion->data());
        ins[i].data.assign(p, p + n * sizeof(float));
        dmabufs.push_back(std::move(ion));
      }
    } else {
      const uint8_t* p = reinterpret_cast<const uint8_t*>(t.data());
      ins[i].data.assign(p, p + t.size() * sizeof(float));
    }
  }
  std::vector<IOTensor> outs;
  if (sess_->run(ins, outs) != Status::kOk)
    return {};
  std::vector<Tensor> result;
  for (auto& o : outs) {
    int64_t n = 1;
    for (int64_t d : o.shape)
      n *= d;
    if (o.shape.empty())
      n = (int64_t)(o.data.size() / sizeof(float));
    const float* f = o.f32();
    result.emplace_back(std::vector<float>(f, f + n), o.shape, o.name);
  }
  return result;
}

std::vector<Tensor> Model::run(const Tensor& input) {
  return run(std::vector<Tensor>{input});
}

Tensor Model::run(const std::vector<float>& input) {
  auto info = sess_ ? sess_->inputInfo() : std::vector<IOInfo>{};
  Shape shape = info.empty() ? Shape{(int64_t)input.size()} : info[0].shape;
  auto outs = run(std::vector<Tensor>{Tensor(input, shape)});
  return outs.empty() ? Tensor{} : outs[0];
}

}  // namespace vx
