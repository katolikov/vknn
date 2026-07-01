#pragma once
#include "vknn/tensor_class.h"
#include <string>
#include <vector>

namespace vknn {

    /// Find a tensor by name in a run() result (for multi-output models). Returns nullptr if absent.
    const Tensor *findTensor(const std::vector<Tensor> &tensors, const std::string &name);

} // namespace vknn
