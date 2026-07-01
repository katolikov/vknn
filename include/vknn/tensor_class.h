#pragma once
#include "vknn/dtype.h"
#include "vknn/tensor_format.h"
#include <string>
#include <vector>

namespace vknn {

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

} // namespace vknn
