#include "vk_tune_race.h"
#include <algorithm>

namespace vknn { namespace vk {

    namespace {
        // Middle sample of a copy of `values` (mean of the two middle ones for an even count).
        double medianOf(std::vector<double> values) {
            std::sort(values.begin(), values.end());
            const size_t mid = values.size() / 2;
            return values.size() % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
        }
    } // namespace

    std::vector<double> raceCandidates(int count, const std::function<double(int)> &submitOnce, int rounds) {
        if (count <= 0)
        {
            return {};
        }
        const int evenRounds = std::max(2, rounds + (rounds & 1));
        // Warm-up: one discarded submit per candidate, in race order. Nothing here is recorded,
        // so the first-use pipeline cost and the clock ramp land outside every estimate.
        for (int candidate = 0; candidate < count; ++candidate)
        {
            submitOnce(candidate);
        }
        std::vector<std::vector<double>> samples((size_t) count);
        for (int round = 0; round < evenRounds; ++round)
        {
            const bool reversed = (round % 2) == 1;
            for (int slot = 0; slot < count; ++slot)
            {
                const int candidate = reversed ? (count - 1 - slot) : slot;
                samples[(size_t) candidate].push_back(submitOnce(candidate));
            }
        }
        std::vector<double> estimates((size_t) count);
        for (int candidate = 0; candidate < count; ++candidate)
        {
            estimates[(size_t) candidate] = medianOf(samples[(size_t) candidate]);
        }
        return estimates;
    }

}} // namespace vknn::vk
