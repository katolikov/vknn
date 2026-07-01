#pragma once
#include "vknn/config.h"
#include "vknn/tensor_class.h"
#include "vknn/tensor_info.h"
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    class Session;

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

} // namespace vknn
