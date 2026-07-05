#pragma once
#include "vknn/config.h"
#include "vknn/tensor_class.h"
#include "vknn/tensor_info.h"
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    class Session;

    /// A loaded, ready-to-run model. Copyable handle that shares the underlying engine: copies refer
    /// to the same Session, so running one copy is indistinguishable from running another.
    class Model {
      public:
        /// Load an ONNX model. Picks the Vulkan backend if available (CPU fallback), and the given
        /// precision tier (Low = fp16, Normal = fp16 + selective fp32, High = fp32).
        /// @param onnxPath  Filesystem path to the ".onnx" model (or a ".vxm" produced by save()).
        /// @param precision Numeric precision tier for the compiled graph.
        /// @returns A Model handle; test ok() (or the bool conversion) to confirm the load succeeded.
        static Model load(const std::string &onnxPath, Precision precision = Precision::Low);
        /// Advanced: full control via Config.
        /// @param onnxPath Filesystem path to the ".onnx" model (or a ".vxm" produced by save()).
        /// @param cfg      Backend selection, precision, hints, and every other tunable knob.
        /// @returns A Model handle; test ok() (or the bool conversion) to confirm the load succeeded.
        static Model load(const std::string &onnxPath, const Config &cfg);
        /// Save the optimized model to a ".vxm" file. A later Model::load() on that path skips ONNX
        /// parsing + graph passes (faster load).
        /// @param vxmPath Destination path for the serialized optimized model.
        /// @returns True on success, false on error (e.g. the path is unwritable).
        bool save(const std::string &vxmPath) const;

        /// True once a model is loaded (the shared engine is present).
        bool ok() const noexcept {
            return sess_ != nullptr;
        }
        /// Same as ok(); enables `if (model)`.
        explicit operator bool() const noexcept {
            return ok();
        }

        /// What the model expects / produces — names, shapes, dtypes. You don't have to set any of it.
        /// @returns The model's declared input tensors, in binding order.
        std::vector<TensorInfo> inputs() const;
        /// @returns The model's declared output tensors, in binding order.
        std::vector<TensorInfo> outputs() const;

        /// Run with one input tensor; returns all outputs (named + shaped).
        /// @param input Single host input tensor for a single-input model.
        /// @returns Every output tensor, each carrying its name and shape.
        std::vector<Tensor> run(const Tensor &input);
        /// Run with several inputs (matched to the model's inputs in order). Inputs may be host tensors
        /// or DMA-BUF inputs (Tensor::fromDmaBuf). Optional `outputs` are DMA-BUF output bindings
        /// (Tensor::toDmaBuf): each named output is written straight into the caller's fd, and the
        /// returned Tensor for it carries no host copy (empty data). Outputs without a binding come back
        /// as host tensors as usual.
        /// @param inputs  Inputs matched to the model's inputs in order (host or DMA-BUF tensors).
        /// @param outputs Optional DMA-BUF output bindings, matched to outputs by name.
        /// @returns Every output tensor; DMA-BUF-bound outputs return with empty host data.
        std::vector<Tensor> run(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs = {});
        /// Simplest form: raw values in (shaped to the model's single input), first output back.
        /// @param input Raw float values, reshaped to the model's single input.
        /// @returns The model's first output tensor.
        Tensor run(const std::vector<float> &input);

        /// Escape hatch to the low-level engine.
        /// @returns The shared Session, or nullptr when the model is not loaded (see ok()).
        Session *session() const noexcept {
            return sess_.get();
        }

      private:
        std::shared_ptr<Session> sess_;
    };

} // namespace vknn
