// The friendly, high-level API. Load a model and run it — no tensor names, shapes, or dtypes to
// wire up by hand; everything is read from the model. This is the API most users should reach for.
// (The lower-level Session/IOTensor in session.h stay available for advanced control.)
//
//   vx::Model net = vx::Model::load("mobilenet.onnx");        // precision auto, Vulkan if available
//   vx::Tensor out = net.run(pixels);                          // pixels = std::vector<float>, NCHW
//   int cls = out.argmax();                                    // done
//
//   for (auto& in : net.inputs()) printf("%s %s\n", in.name.c_str(), in.shapeString().c_str());
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "vx/config.h"
#include "vx/dtype.h"

namespace vx {

class Session;

/// Describes one model input or output (read from the model — you never set these yourself).
struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  DType dtype = DType::kFloat32;     // the tensor's element type (values cross the API as fp32)
  int64_t count = 0;                 // number of elements
  std::string shapeString() const;   // e.g. "1x3x224x224"
};

/// A tensor going in or out of a model. Carries its own shape + data; all the accessors you'd want
/// are here so you never poke at raw buffers or recompute strides. Data is row-major fp32 (NCHW).
class Tensor {
 public:
  Tensor() = default;
  Tensor(std::vector<float> data, std::vector<int64_t> shape, std::string name = "");
  /// Wrap raw values with a 1-D shape (handy for quick inputs).
  explicit Tensor(std::vector<float> data);
  /// Zero-copy input from a DMA-BUF fd (e.g. a camera/ION buffer). vxrt imports the fd instead of
  /// copying a host buffer. `name` selects which model input this feeds (optional for single-input).
  /// The fd's memory is read as row-major fp32 in the given shape.
  static Tensor fromDmaBuf(int fd, std::vector<int64_t> shape, std::string name = "");
  int dmaBufFd() const { return fd_; }

  const std::string& name() const { return name_; }
  const std::vector<int64_t>& shape() const { return shape_; }
  std::string shapeString() const;
  int rank() const { return (int)shape_.size(); }
  int64_t dim(int i) const { return (i >= 0 && i < rank()) ? shape_[i] : 1; }
  int64_t size() const { return (int64_t)data_.size(); }  // total element count
  bool empty() const { return data_.empty(); }

  const float* data() const { return data_.data(); }
  float* data() { return data_.data(); }
  const std::vector<float>& values() const { return data_; }
  float operator[](int64_t i) const { return data_[i]; }

  /// Index of the largest value — the usual "predicted class" for a classifier output.
  int64_t argmax() const;
  /// Largest value.
  float max() const;

 private:
  std::string name_;
  std::vector<int64_t> shape_;
  std::vector<float> data_;
  int fd_ = -1;  // DMA-BUF fd for zero-copy input (-1 = host data in data_)
};

/// A loaded, ready-to-run model. Copyable handle (shares the underlying engine).
class Model {
 public:
  /// Load an ONNX model. Picks the Vulkan backend if available (CPU fallback), and the given
  /// precision (Auto = fp16 on GPU). This is all the configuration most users need.
  static Model load(const std::string& onnxPath, Precision precision = Precision::kAuto);
  /// Advanced: full control via Config.
  static Model load(const std::string& onnxPath, const Config& cfg);
  /// Save the optimized model to a ".vxm" file. A later Model::load() on that path skips ONNX
  /// parsing + graph passes (faster load). Returns false on error.
  bool save(const std::string& vxmPath) const;

  bool ok() const { return sess_ != nullptr; }
  explicit operator bool() const { return ok(); }

  /// What the model expects / produces — names, shapes, dtypes. You don't have to set any of it.
  std::vector<TensorInfo> inputs() const;
  std::vector<TensorInfo> outputs() const;

  /// Run with one input tensor; returns all outputs (named + shaped).
  std::vector<Tensor> run(const Tensor& input);
  /// Run with several inputs (matched to the model's inputs in order).
  std::vector<Tensor> run(const std::vector<Tensor>& inputs);
  /// Simplest form: raw values in (shaped to the model's single input), first output back.
  Tensor run(const std::vector<float>& input);

  /// Escape hatch to the low-level engine.
  Session* session() const { return sess_.get(); }

 private:
  std::shared_ptr<Session> sess_;
};

/// Find a tensor by name in a run() result (for multi-output models). Returns nullptr if absent.
const Tensor* findTensor(const std::vector<Tensor>& tensors, const std::string& name);

}  // namespace vx
