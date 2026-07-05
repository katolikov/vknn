#pragma once
#include "vknn/dtype.h"
#include "vknn/tensor_format.h"
#include <string>
#include <vector>

namespace vknn {

    /// A tensor going in or out of a model. Carries its own shape + data; all the accessors you'd want
    /// are here so you never poke at raw buffers or recompute strides. Data is row-major fp32 (NCHW).
    ///
    /// A tensor is in one of two modes. In the default host mode it owns its element data in a
    /// std::vector<float>. In zero-copy mode (built via fromDmaBuf()/toDmaBuf()) it instead carries a
    /// DMA-BUF fd and a declared layout/dtype, and its host vector is empty; the data accessors below
    /// then return an empty/zero view.
    class Tensor {
      public:
        Tensor() = default;
        /// Take ownership of `data` (row-major fp32) with the given `shape` and optional `name`.
        Tensor(std::vector<float> data, std::vector<int64_t> shape, std::string name = "");
        /// Wrap raw values with a 1-D shape (handy for quick inputs).
        explicit Tensor(std::vector<float> data);
        /// Zero-copy INPUT from a DMA-BUF fd (e.g. a camera/ION buffer the caller owns). vknn reads the
        /// input straight from the fd instead of from a caller host buffer. `name` selects which model
        /// input this feeds (optional for single-input). `layout`/`dtype` declare the fd's bytes: when
        /// they match the model's device-native boundary (see IOInfo::deviceFormat/deviceDtype) the fd is
        /// bound directly, otherwise the GPU converts on read. TensorFormat::Auto means "already
        /// device-native — bind directly".
        /// @param fd     Caller-owned DMA-BUF file descriptor; the caller retains ownership.
        /// @param shape  Logical NCHW shape of the tensor the fd holds.
        /// @param name   Target model input; may be empty for a single-input model.
        /// @param layout Declared layout of the fd's bytes.
        /// @param dtype  Declared element type of the fd's bytes.
        /// @returns A zero-copy input Tensor carrying the fd (its host data vector is empty).
        static Tensor fromDmaBuf(int fd, std::vector<int64_t> shape, std::string name = "", TensorFormat layout = TensorFormat::NCHW, DType dtype = DType::Float32);
        /// Zero-copy OUTPUT binding: pass in Model::run()'s `outputs` list to have vknn write that
        /// output straight into the caller's DMA-BUF fd, no host output buffer. `name` selects which
        /// model output (required when the model has several). `layout`/`dtype` declare the fd's bytes;
        /// the GPU converts the device-native result into them, or writes directly when they match
        /// (or layout is Auto).
        /// @param fd     Caller-owned DMA-BUF file descriptor; the caller retains ownership.
        /// @param shape  Logical NCHW shape to write into the fd.
        /// @param name   Target model output; required when the model has several outputs.
        /// @param layout Declared layout of the fd's bytes.
        /// @param dtype  Declared element type of the fd's bytes.
        /// @returns A zero-copy output Tensor carrying the fd (its host data vector is empty).
        static Tensor toDmaBuf(int fd, std::vector<int64_t> shape, std::string name = "", TensorFormat layout = TensorFormat::NCHW, DType dtype = DType::Float32);
        /// DMA-BUF file descriptor for zero-copy I/O, or -1 for a host-data tensor.
        int dmaBufFd() const noexcept {
            return fd_;
        }
        /// Declared layout of the DMA-BUF fd's bytes (meaningful only when dmaBufFd() >= 0).
        TensorFormat dmaBufFormat() const noexcept {
            return dmaBufFormat_;
        }
        /// Declared element type of the DMA-BUF fd's bytes (meaningful only when dmaBufFd() >= 0).
        DType dmaBufDtype() const noexcept {
            return dmaBufDtype_;
        }

        /// Tensor name; ties this tensor to a model input/output by name, empty if unnamed.
        const std::string &name() const noexcept {
            return name_;
        }
        /// Logical shape (NCHW), one extent per dimension.
        const std::vector<int64_t> &shape() const noexcept {
            return shape_;
        }
        /// Human-readable shape rendering, e.g. "[1, 3, 224, 224]".
        std::string shapeString() const;
        /// Number of dimensions in shape().
        int rank() const noexcept {
            return (int) shape_.size();
        }
        /// Extent of dimension `i`, or 1 when `i` is out of range (so missing dims read as broadcastable).
        int64_t dim(int i) const noexcept {
            return (i >= 0 && i < rank()) ? shape_[i] : 1;
        }
        /// Total element count of the host data (0 for a zero-copy DMA-BUF tensor).
        int64_t size() const noexcept {
            return (int64_t) data_.size();
        }
        /// True when there is no host element data.
        bool empty() const noexcept {
            return data_.empty();
        }

        /// Pointer to the first host element (row-major fp32), or an empty-vector pointer if there is none.
        const float *data() const noexcept {
            return data_.data();
        }
        /// Mutable pointer to the first host element (row-major fp32).
        float *data() noexcept {
            return data_.data();
        }
        /// The host element data as a vector.
        const std::vector<float> &values() const noexcept {
            return data_;
        }
        /// Element `i` of the host data. Precondition: 0 <= i < size(); the index is not bounds-checked.
        float operator[](int64_t i) const noexcept {
            return data_[i];
        }

        /// Index of the largest value — the usual "predicted class" for a classifier output.
        /// @returns The argmax index, or -1 when the tensor has no host data.
        int64_t argmax() const;
        /// Largest host value, or 0 when the tensor has no host data.
        float max() const;

      private:
        std::string          name_;
        std::vector<int64_t> shape_;
        std::vector<float>   data_;
        int                  fd_           = -1;                 ///< DMA-BUF fd for zero-copy I/O (-1 = host data in data_).
        TensorFormat         dmaBufFormat_ = TensorFormat::NCHW; ///< Declared layout of the fd's bytes.
        DType                dmaBufDtype_  = DType::Float32;      ///< Declared dtype of the fd's bytes.
    };

} // namespace vknn
