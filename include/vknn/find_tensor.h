#pragma once
#include "vknn/tensor_class.h"
#include <string>
#include <vector>

namespace vknn {

    /// Look up a tensor by name in a run() result — the convenient way to pick one output out of a
    /// multi-output model without indexing by position.
    /// @param tensors The result list to search (typically Model::run()'s return value).
    /// @param name    The exact tensor name to match (see IOInfo::name / Model::outputs()).
    /// @returns A borrowed pointer to the first tensor whose name() equals `name`, or nullptr if none
    ///          matches. The pointer aliases an element of `tensors` and is invalidated by any later
    ///          modification of that vector; it is never owned by the caller.
    const Tensor *findTensor(const std::vector<Tensor> &tensors, const std::string &name);

} // namespace vknn
