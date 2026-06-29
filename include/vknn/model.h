// High-level API. Load a model and run it — no tensor names, shapes, or dtypes to wire up by hand;
// everything is read from the model. The lower-level Session / IOTensor in session.h remain
// available for advanced control.
//
//   vknn::Model net = vknn::Model::load("mobilenet.onnx");
//   vknn::Tensor out = net.run(pixels);   // pixels: std::vector<float>, NCHW
//   int cls = out.argmax();
#pragma once
#include "vknn/config.h"
#include "vknn/dtype.h"
#include "vknn/tensor_format.h"
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    class Session;

    /// Describes one model input or output (read from the model — you never set these yourself).
    struct TensorInfo {
        std::string          name;
        std::vector<int64_t> shape;
        DType                dtype = DType::Float32; // the tensor's element type (values cross the API as fp32)
        int64_t              count = 0;              // number of elements
        std::string          shapeString() const;    // e.g. "1x3x224x224"
    };

    /// A tensor going in or out of a model. Carries its own shape + data; all the accessors you'd want
    /// are here so you never poke at raw buffers or recompute strides. Data is row-major fp32 (NCHW).
    class Tensor {
      public:
        Tensor() = default;
        Tensor(std::vector<float> data, std::vector<int64_t> shape, std::string name = "");
        /// Wrap raw values with a 1-D shape (handy for quick inputs).
        explicit Tensor(std::vector<float> data);
        /// Zero-copy INPUT from a DMA-BUF fd (e.g. a camera/ION buffer the caller owns). vknn reads the
        /// input straight from the fd instead of from a caller host buffer. `name` selects which model
        /// input this feeds (optional for single-input). `layout`/`dtype` declare the fd's bytes: when
        /// they match the model's device-native boundary (see IOInfo::deviceFormat/deviceDtype) the fd is
        /// bound directly, otherwise the GPU converts on read. TensorFormat::Auto means "already
        /// device-native — bind directly".
        static Tensor fromDmaBuf(int fd, std::vector<int64_t> shape, std::string name = "", TensorFormat layout = TensorFormat::NCHW, DType dtype = DType::Float32);
        /// Zero-copy OUTPUT binding: pass in Model::run()'s `outputs` list to have vknn write that
        /// output straight into the caller's DMA-BUF fd, no host output buffer. `name` selects which
        /// model output (required when the model has several). `layout`/`dtype` declare the fd's bytes;
        /// the GPU converts the device-native result into them, or writes directly when they match
        /// (or layout is Auto).
        static Tensor toDmaBuf(int fd, std::vector<int64_t> shape, std::string name = "", TensorFormat layout = TensorFormat::NCHW, DType dtype = DType::Float32);
        int dmaBufFd() const {
            return fd_;
        }
        TensorFormat dmaBufFormat() const {
            return dmaBufFormat_;
        }
        DType dmaBufDtype() const {
            return dmaBufDtype_;
        }

        const std::string &name() const {
            return name_;
        }
        const std::vector<int64_t> &shape() const {
            return shape_;
        }
        std::string shapeString() const;
        int         rank() const {
            return (int) shape_.size();
        }
        int64_t dim(int i) const {
            return (i >= 0 && i < rank()) ? shape_[i] : 1;
        }
        int64_t size() const {
            return (int64_t) data_.size();
        } // total element count
        bool empty() const {
            return data_.empty();
        }

        const float *data() const {
            return data_.data();
        }
        float *data() {
            return data_.data();
        }
        const std::vector<float> &values() const {
            return data_;
        }
        float operator[](int64_t i) const {
            return data_[i];
        }

        /// Index of the largest value — the usual "predicted class" for a classifier output.
        int64_t argmax() const;
        /// Largest value.
        float max() const;

      private:
        std::string          name_;
        std::vector<int64_t> shape_;
        std::vector<float>   data_;
        int                  fd_           = -1;                 // DMA-BUF fd for zero-copy I/O (-1 = host data in data_)
        TensorFormat         dmaBufFormat_ = TensorFormat::NCHW; // declared layout of the fd's bytes
        DType                dmaBufDtype_  = DType::Float32;     // declared dtype of the fd's bytes
    };

    /// A loaded, ready-to-run model. Copyable handle (shares the underlying engine).
    class Model {
      public:
        /// Load an ONNX model. Picks the Vulkan backend if available (CPU fallback), and the given
        /// precision tier (Low = fp16, Normal = fp16 + selective fp32, High = fp32).
        static Model load(const std::string &onnxPath, Precision precision = Precision::Low);
        /// Advanced: full control via Config.
        static Model load(const std::string &onnxPath, const Config &cfg);
        /// Save the optimized model to a ".vxm" file. A later Model::load() on that path skips ONNX
        /// parsing + graph passes (faster load). Returns false on error.
        bool save(const std::string &vxmPath) const;

        bool ok() const {
            return sess_ != nullptr;
        }
        explicit operator bool() const {
            return ok();
        }

        /// What the model expects / produces — names, shapes, dtypes. You don't have to set any of it.
        std::vector<TensorInfo> inputs() const;
        std::vector<TensorInfo> outputs() const;

        /// Run with one input tensor; returns all outputs (named + shaped).
        std::vector<Tensor> run(const Tensor &input);
        /// Run with several inputs (matched to the model's inputs in order). Inputs may be host tensors
        /// or DMA-BUF inputs (Tensor::fromDmaBuf). Optional `outputs` are DMA-BUF output bindings
        /// (Tensor::toDmaBuf): each named output is written straight into the caller's fd, and the
        /// returned Tensor for it carries no host copy (empty data). Outputs without a binding come back
        /// as host tensors as usual.
        std::vector<Tensor> run(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs = {});
        /// Simplest form: raw values in (shaped to the model's single input), first output back.
        Tensor run(const std::vector<float> &input);

        /// Escape hatch to the low-level engine.
        Session *session() const {
            return sess_.get();
        }

      private:
        std::shared_ptr<Session> sess_;
    };

    /// Find a tensor by name in a run() result (for multi-output models). Returns nullptr if absent.
    const Tensor *findTensor(const std::vector<Tensor> &tensors, const std::string &name);

} // namespace vknn
