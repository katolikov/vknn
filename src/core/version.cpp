// The engine version string.
//
// Defined in its own translation unit so the string lives in the compiled engine: a caller that
// links a prebuilt library reports the version of the library it actually loaded, not the version
// of the header it was compiled against. That difference is the whole point of exposing it, and it
// also leaves the version greppable in any binary with `strings`.
#include "vknn/version.h"

namespace vknn {

    const char *vknnVersion() {
        return VKNN_VERSION_STRING;
    }

} // namespace vknn
