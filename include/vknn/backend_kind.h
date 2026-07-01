// Backend selection enum plus its string helpers.
#pragma once
#include <string>

namespace vknn {

    enum class BackendKind { Vulkan = 0, Cpu = 1 };

    const char *backendName(BackendKind k);
    BackendKind backendFromStr(const std::string &s);

} // namespace vknn
