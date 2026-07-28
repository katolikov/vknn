// Engine version accessor. Its own translation unit, so the string is part of the compiled library
// and stays greppable in a binary with `strings`.
#include "vknn/version.h"

namespace vknn {

    const char *vknnVersion() {
        return VKNN_VERSION_STRING;
    }

} // namespace vknn
