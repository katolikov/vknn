// Prepacked-weight + autotune cache backing warm session starts.
#pragma once
#include "core/cache_codec.h"
#include <map>
#include <string>
#include <vector>

namespace vknn {

    /// In-memory cache of prepacked weights (keyed by op+role+shape) and autotuned workgroup sizes.
    /// Skips the host repacking + per-shape autotune on warm session creation. It maps to/from one
    /// CacheVariant of the multi-variant model cache (see cache_codec.h).
    class WeightCache {
      public:
        // Clear and set whether prepacked weights are retained for saving. `enabled` is true when a
        // persistent cache file is in use; without a file, weights are uploaded and freed (never
        // retained) to avoid ballooning RAM (a 965M model would hold ~3.85GB of prepacked fp32).
        void reset(bool enabled);
        // Populate from a cached variant (warm start), then retain for the next save.
        void loadFrom(const CacheVariant &v);
        // Copy the retained weights + autotune table into a variant for serialization.
        void writeInto(CacheVariant &v) const;
        bool enabled() const {
            return enabled_;
        }
        bool dirty() const {
            return dirty_;
        }
        /// Clear the dirty mark once the retained content has reached the cache file, so a later flush
        /// with nothing new to add costs nothing.
        void markSaved() const {
            dirty_ = false;
        }
        bool get(const std::string &key, std::vector<float> &out) const;
        void put(const std::string &key, const std::vector<float> &data);
        // autotune table: op-signature -> chosen kernel value, plus the Tuning level each entry was
        // measured at. `level` (when non-null) receives the cached entry's level, or -1 on a miss —
        // the pick sites re-sweep when the requested level exceeds it (a fast entry does not serve a
        // heavy request) and reuse it under Tuning::None (none runs no new sweep but honors a cached
        // one). A legacy entry with no stored level reads back as Fast.
        int  tuned(const std::string &sig, int dflt, int *level = nullptr) const;
        void setTuned(const std::string &sig, int val, int level);

      private:
        std::map<std::string, std::vector<float>> weights_;
        std::map<std::string, int>                tune_;
        std::map<std::string, int>                tuneLevel_;       // op-signature -> Tuning level it was measured at
        bool                                      enabled_ = false; // retain prepacked weights for saving
        mutable bool                              dirty_   = false;
    };

} // namespace vknn
