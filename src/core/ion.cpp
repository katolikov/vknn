#include "vknn/ion.h"
#include "vknn/logging.h"
#include <sys/mman.h>
#include <unistd.h>

namespace vknn {

    std::unique_ptr<IonBuffer> IonBuffer::wrapFd(int fd, size_t bytes, bool takeOwnership) {
        auto b   = std::unique_ptr<IonBuffer>(new IonBuffer());
        b->fd_   = fd;
        b->size_ = bytes;
        b->owns_ = takeOwnership;
        b->map_  = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (b->map_ == MAP_FAILED)
        {
            b->map_ = nullptr;
            VKNN_WARN << "ION: wrapFd mmap failed (fd " << fd << ", " << bytes << " bytes)";
        }
        return b;
    }

    IonBuffer::~IonBuffer() {
        if (map_)
        {
            ::munmap(map_, size_);
        }
        if (owns_ && fd_ >= 0)
        {
            ::close(fd_);
        }
    }

} // namespace vknn
