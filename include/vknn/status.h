// Status codes returned across the public API, plus their string names.
#pragma once

namespace vknn {

    /// Status codes returned across the public API.
    enum class Status {
        Ok = 0,
        InvalidArgument,
        Unsupported,
        NotFound,
        RuntimeError,
        DeviceLost,
        IoError,
    };

    inline const char *statusStr(Status s) {
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
