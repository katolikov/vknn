// Backend-level device-weight pool: one uploaded copy of a weight/bias/transformed-weight buffer
// shared by every op instance that references the same initializer, pack-kind and precision.
//
// Motivation. Runtime dynamic shapes plan one segment set per shape bucket (ADR-0012). Each bucket
// re-instantiates the ops, and each op instance would otherwise upload its own device copy of every
// weight through uploadCached() — N buckets would multiply weight VRAM. Weight bytes are
// activation-shape-independent (the pack depends only on the weight tensor and the compute
// precision), so the buckets can share one upload. This pool is that sharing point: acquire() keys a
// buffer by the weight-cache key (which already encodes the initializer identity and the pack-kind,
// e.g. "<model>/<node>#w" or "…#wino2") plus the storage precision, and returns the same buffer to
// every caller of that key.
//
// Lifetime. The pool holds each buffer WEAKLY, exactly like VulkanBackend::uploadPooled: the owners
// are the op instances that acquired it, so a buffer is freed when its last user drops it and the
// process-wide allocation count is unaffected for a single-bucket (fixed-shape) model — the one op
// instance holds the one buffer, byte-for-byte the same allocation as before this pool existed.
//
// Handle-agnostic by design. The template parameter is the buffer type; nothing here includes a
// Vulkan header, so the keying/refcount logic is host-testable with a stand-in buffer and the Vulkan
// backend instantiates it with vk::Buffer.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace vknn {

    /// Refcounted, weakly-held device-buffer pool keyed by (weight-cache key, precision). Not thread
    /// safe: it is populated during single-threaded plan/prepare, like the rest of the backend's pools.
    template <typename Buf> class DeviceWeightPool {
      public:
        /// Return the buffer for (`key`, `fp16`), constructing it via `make` on a miss. Repeat callers
        /// with the same key+precision get the same shared buffer (the factory runs once while any user
        /// is live); a different key or precision gets its own buffer. When every prior user has dropped
        /// the buffer, the next acquire rebuilds it rather than returning a dangling handle.
        template <typename Fn> std::shared_ptr<Buf> acquire(const std::string &key, bool fp16, Fn make) {
            std::string full = key;
            full += fp16 ? "|h" : "|f"; // precision splits the key: fp16 and fp32 packs are distinct buffers
            auto it = pool_.find(full);
            if (it != pool_.end())
            {
                if (auto b = it->second.lock())
                {
                    return b;
                }
            }
            std::shared_ptr<Buf> b = make();
            pool_[full]            = b;
            return b;
        }

        /// Live entry count (including expired weak slots); test/introspection only.
        size_t size() const {
            return pool_.size();
        }

      private:
        std::map<std::string, std::weak_ptr<Buf>> pool_;
    };

} // namespace vknn
