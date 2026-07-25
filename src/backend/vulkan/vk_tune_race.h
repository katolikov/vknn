// Shared timing discipline for the load-time kernel races (matmul.cpp pickTile, the conv.cpp
// tile/local-size races, conv_gemm.cpp pickVariant).
//
// Timing one candidate to completion and then the next hands every later candidate a GPU whose
// clocks the earlier candidates already moved: the device ramps up over the first submits and
// throttles back under sustained load, and that drift is comparable to the differences the races
// are trying to resolve. raceCandidates removes the order term instead of hoping it is small - the
// candidates alternate one submit at a time and the round order reverses every other round, so
// every candidate occupies the same mean position in the race.
#pragma once

#include <functional>
#include <vector>

namespace vknn { namespace vk {

    /// Recorded rounds behind one candidate's estimate. Rounds alternate forward and reversed
    /// candidate order, so a monotone clock drift over a forward/reverse pair contributes the
    /// same amount to every candidate and cancels to first order; the count must stay even for
    /// that cancellation to be exact. Four rounds leave two samples on each side of the median.
    constexpr int kTuneRaceRounds = 4;

    /// Race `count` candidates and return one millisecond estimate per candidate (index-aligned).
    /// `submitOnce(index)` records and submits that candidate's work once and returns the wall
    /// time of the submit; the caller keeps ownership of pipelines, scratch buffers and geometry.
    ///
    /// Every candidate gets one discarded warm-up submit before the recorded rounds, so first-use
    /// pipeline costs and the initial clock ramp are paid before any candidate is on the clock.
    /// The reported estimate is the median of the candidate's per-round samples: unlike a min it
    /// does not reward whichever candidate happened to catch the one quiet moment, and unlike a
    /// mean a single contaminated sample (an OS scheduling hiccup, a throttle step) does not move
    /// it. `rounds` is rounded up to the next even count.
    std::vector<double> raceCandidates(int count, const std::function<double(int)> &submitOnce, int rounds = kTuneRaceRounds);

}} // namespace vknn::vk
