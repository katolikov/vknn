// Status codes returned across the public API, plus statusStr() to spell them.
#pragma once

namespace vknn {

    /// Status codes returned across the public API. Ok is guaranteed to be 0, so a Status is
    /// falsy exactly on success and every error is truthy.
    enum class Status {
        Ok = 0,          ///< The operation completed successfully.
        InvalidArgument, ///< A caller-supplied argument was malformed, out of range, or inconsistent.
        Unsupported,     ///< The requested operation, op, dtype, or format is not implemented on this path.
        NotFound,        ///< A named resource (file, tensor, cache entry) does not exist.
        RuntimeError,    ///< Execution failed for a reason not covered by a more specific code.
        DeviceLost,      ///< The Vulkan device was lost (reset, TDR, or driver fault); the context is unusable.
        IoError,         ///< A filesystem or stream read/write failed.
    };

    /// Readable spelling of a Status enumerator, matching the enumerator name (e.g. "DeviceLost").
    /// @param s Status to name.
    /// @returns A static string literal owned by the program; never null. Unrecognized values yield "?".
    inline const char *statusStr(Status s) noexcept {
        switch (s)
        {
            case Status::Ok:
                return "Ok";
            case Status::InvalidArgument:
                return "InvalidArgument";
            case Status::Unsupported:
                return "Unsupported";
            case Status::NotFound:
                return "NotFound";
            case Status::RuntimeError:
                return "RuntimeError";
            case Status::DeviceLost:
                return "DeviceLost";
            case Status::IoError:
                return "IoError";
        }
        return "?";
    }

} // namespace vknn
