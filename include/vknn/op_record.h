// Per-op timing record: CPU wall clock plus GPU timestamp, dispatch dims, and IO bytes.
#pragma once
#include "vknn/op.h"
#include <array>
#include <cstdint>
#include <string>

namespace vknn {

    struct OpRecord {
        std::string             name;
        OpType                  type = OpType::Unknown;
        std::string             backend;
        double                  cpuMs    = 0.0;
        double                  gpuMs    = -1.0; // <0 => not measured
        std::array<uint32_t, 3> dispatch = {0, 0, 0};
        int64_t                 bytesIO  = 0;
        bool                    fellBack = false; // primary backend couldn't run -> CPU
    };

} // namespace vknn
