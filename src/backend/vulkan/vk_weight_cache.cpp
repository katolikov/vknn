#include "vk_weight_cache.h"
#include "vknn/logging.h"
#include "vknn/tuning.h"

namespace vknn {

    void WeightCache::reset(bool enabled) {
        weights_.clear();
        tune_.clear();
        tuneLevel_.clear();
        enabled_ = enabled;
        dirty_   = false;
    }
    void WeightCache::loadFrom(const CacheVariant &v) {
        weights_ = v.weights;
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
    void WeightCache::writeInto(CacheVariant &v) const {
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
    bool WeightCache::get(const std::string &key, std::vector<float> &out) const {
        auto it = weights_.find(key);
        if (it == weights_.end())
        {
            return false;
        }
        out = it->second;
        return true;
    }
    void WeightCache::put(const std::string &key, const std::vector<float> &data) {
        weights_[key] = data;
        dirty_        = true;
    }
    int WeightCache::tuned(const std::string &sig, int dflt, int *level) const {
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
    void WeightCache::setTuned(const std::string &sig, int val, int level) {
        tune_[sig]      = val;
        tuneLevel_[sig] = level;
        dirty_          = true;
    }

} // namespace vknn
