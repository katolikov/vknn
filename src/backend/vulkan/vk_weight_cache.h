// Prepacked-weight + autotune cache backing warm session starts.
#pragma once
#include "core/cache_codec.h"
#include "vknn/logging.h"
#include "vknn/tuning.h"
#include <map>
#include <string>
#include <vector>

namespace vknn {

    /// In-memory cache of prepacked weights (keyed by op+role+shape) and autotuned workgroup sizes.
    /// Skips the host repacking + per-shape autotune on warm session creation. It maps to/from one
    /// CacheVariant of the multi-variant model cache (see cache_codec.h).
    ///
    /// Free of Vulkan types, like the device-weight pool it feeds, so its retention accounting is
    /// exercised on host (tests/test_weight_cache_retention.cpp).
    ///
    /// The retained prepacked blobs are fp32 and weight-sized, so they are held only for as long as
    /// they are needed to reach the cache file: releaseRetained() drops them once a save has confirmed
    /// them on disk, and anything put() afterwards accumulates as a delta that the next save merges
    /// into the stored variant (mergeSessionArtifacts, vk_cache_image.h). The autotune tables are
    /// key-sized and stay whole.
    class WeightCache {
      public:
        // Clear and set whether prepacked weights are retained for saving. `enabled` is true when a
        // persistent cache file is in use; without a file, weights are uploaded and freed (never
        // retained) to avoid ballooning RAM (a 965M model would hold ~3.85GB of prepacked fp32).
        void reset(bool enabled) {
            weights_.clear();
            tune_.clear();
            tuneLevel_.clear();
            enabled_ = enabled;
            dirty_   = false;
        }
        // Populate from a cached variant (warm start), then retain for the next save. The variant is
        // consumed: its weight map moves in rather than doubling for the duration of the load.
        void loadFrom(CacheVariant &&v) {
            weights_ = std::move(v.weights);
            tune_.clear();
            tuneLevel_.clear();
            for (const auto &kv: v.tune)
            {
                tune_[kv.first] = (int) kv.second;
                // A cache written before the level field carries no tunelvl entry; treat it as Fast (the
                // production default nearly every legacy cache was measured at) so a warm Fast load reuses
                // it, while a Heavy request still re-sweeps it once and upgrades the stored level.
                auto lit             = v.tuneLevel.find(kv.first);
                tuneLevel_[kv.first] = lit != v.tuneLevel.end() ? (int) lit->second : (int) Tuning::Fast;
            }
            enabled_ = true;
            dirty_   = false;
            VKNN_INFO << "WeightCache: loaded " << weights_.size() << " prepacked weights, " << tune_.size() << " tuning entries";
        }
        // Copy the retained weights + autotune table into a variant for serialization.
        void writeInto(CacheVariant &v) const {
            v.weights = weights_;
            v.tune.clear();
            v.tuneLevel.clear();
            for (const auto &kv: tune_)
            {
                v.tune[kv.first] = (int32_t) kv.second;
                auto lit         = tuneLevel_.find(kv.first);
                if (lit != tuneLevel_.end())
                {
                    v.tuneLevel[kv.first] = (int32_t) lit->second;
                }
            }
        }
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
        /// Drop the retained prepacked blobs. Valid only once a save has confirmed them in the cache
        /// file: a later get() then misses and the op recomputes its prepack (the device-weight pool
        /// answers first for a weight already uploaded, so a miss is reached only by a genuinely new
        /// one), and the next save merges whatever accumulated since into the stored variant. The
        /// autotune tables and the enabled flag survive, so the cache keeps collecting.
        void releaseRetained() {
            weights_.clear();
        }
        /// Bytes of prepacked weight payload currently retained. The accounting the reclaim is measured
        /// against: it is weight-sized while a save is pending and zero once one has landed.
        size_t retainedBytes() const {
            size_t total = 0;
            for (const auto &kv: weights_)
            {
                total += kv.second.size() * sizeof(float);
            }
            return total;
        }
        /// Number of prepacked blobs currently retained.
        size_t retainedCount() const {
            return weights_.size();
        }
        bool get(const std::string &key, std::vector<float> &out) const {
            auto it = weights_.find(key);
            if (it == weights_.end())
            {
                return false;
            }
            out = it->second;
            return true;
        }
        void put(const std::string &key, const std::vector<float> &data) {
            weights_[key] = data;
            dirty_        = true;
        }
        // autotune table: op-signature -> chosen kernel value, plus the Tuning level each entry was
        // measured at. `level` (when non-null) receives the cached entry's level, or -1 on a miss —
        // the pick sites re-sweep when the requested level exceeds it (a fast entry does not serve a
        // heavy request) and reuse it under Tuning::None (none runs no new sweep but honors a cached
        // one). A legacy entry with no stored level reads back as Fast.
        int tuned(const std::string &sig, int dflt, int *level = nullptr) const {
            auto it = tune_.find(sig);
            if (it == tune_.end())
            {
                if (level)
                {
                    *level = -1;
                }
                return dflt;
            }
            if (level)
            {
                auto lit = tuneLevel_.find(sig);
                *level   = lit != tuneLevel_.end() ? lit->second : (int) Tuning::Fast;
            }
            return it->second;
        }
        void setTuned(const std::string &sig, int val, int level) {
            tune_[sig]      = val;
            tuneLevel_[sig] = level;
            dirty_          = true;
        }

      private:
        std::map<std::string, std::vector<float>> weights_;
        std::map<std::string, int>                tune_;
        std::map<std::string, int>                tuneLevel_;       // op-signature -> Tuning level it was measured at
        bool                                      enabled_ = false; // retain prepacked weights for saving
        mutable bool                              dirty_   = false;
    };

} // namespace vknn
