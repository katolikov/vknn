#include "vknn/mapped_file.h"
#include "vknn/logging.h"

#if defined(_WIN32)
#    define VKNN_HAVE_MMAP 0
#else
#    define VKNN_HAVE_MMAP 1
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace vknn {

    std::shared_ptr<const MappedFile> MappedFile::open(const std::string &path) {
#if !VKNN_HAVE_MMAP
        (void) path;
        return nullptr;
#else
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            return nullptr;
        }
        struct stat info {};
        if (fstat(fd, &info) != 0 || info.st_size <= 0)
        {
            ::close(fd);
            return nullptr;
        }
        const size_t bytes = (size_t) info.st_size;
        void        *base  = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd); // the mapping keeps its own reference to the file
        if (base == MAP_FAILED)
        {
            return nullptr;
        }
        VKNN_INFO << "mapped " << path << " (" << bytes / (1024 * 1024) << " MB) read-only";
        return std::shared_ptr<const MappedFile>(new MappedFile(static_cast<const unsigned char *>(base), bytes));
#endif
    }

    MappedFile::~MappedFile() {
#if VKNN_HAVE_MMAP
        if (base_)
        {
            munmap(const_cast<unsigned char *>(base_), size_);
        }
#endif
    }

} // namespace vknn
