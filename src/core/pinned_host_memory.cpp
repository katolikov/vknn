// See include/vknn/pinned_host_memory.h. The block is zeroed at allocation so its pages are
// resident before the first run (a first-touch fault inside a run would land in the run's wall).
#include "vknn/pinned_host_memory.h"
#include <cstdlib>
#include <cstring>

namespace vknn {
    std::shared_ptr<PinnedHostMemory> PinnedHostMemory::alloc(size_t bytes) {
        const size_t granule  = kPinnedHostAlignment;
        const size_t capacity = ((bytes > 0 ? bytes : 1) + granule - 1) / granule * granule;
        void        *ptr      = nullptr;
        if (posix_memalign(&ptr, granule, capacity) != 0 || ptr == nullptr)
        {
            return nullptr;
        }
        std::memset(ptr, 0, capacity);
        return std::shared_ptr<PinnedHostMemory>(new PinnedHostMemory(ptr, bytes, capacity));
    }

    PinnedHostMemory::~PinnedHostMemory() {
        std::free(ptr_);
    }
} // namespace vknn
